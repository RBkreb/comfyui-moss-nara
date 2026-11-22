"""End-to-end check of the plugin against a real smoke-exported NaRA+SORA file.

No ComfyUI runtime: the plugin's file parsing, variant detection, SORA split and the
per-layer conditioning are exercised against the trainer's own numbers.

    python scripts/check_plugin_file.py <lycoris .safetensors>
"""

import importlib.util
import sys
from pathlib import Path

import torch
from safetensors import safe_open

PLUGIN = Path(__file__).resolve().parent.parent / "nara.py"
DEFAULT = (
    "D:/comfyui/KohakuMossTrainer/runs/smokeB/out/smoke-b-lokr-nara-sora-eva"
    "/export/smoke-b-lokr-nara-sora-eva-lycoris.safetensors"
)


def load_plugin():
    spec = importlib.util.spec_from_file_location("moss_nara_plugin", PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    stubs = {}
    for name in (
        "comfy",
        "comfy.model_management",
        "comfy.model_prefetch",
        "comfy.ops",
        "comfy.weight_adapter",
        "comfy.weight_adapter.base",
    ):
        stubs[name] = type(sys)(name)
    stubs["comfy.weight_adapter.base"].WeightAdapterBase = type("Base", (), {})
    stubs["comfy.model_prefetch"].pause_malloc_graph = lambda: __import__(
        "contextlib"
    ).nullcontext()
    stubs["comfy.model_management"].get_torch_device = lambda: torch.device("cpu")
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def main(path: str) -> int:
    mod = load_plugin()
    with safe_open(path, framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        meta = f.metadata() or {}

    hyper_t, rest = mod.split_hyper(tensors)
    layers = mod.group_nara_layers(rest)

    print(f"file: {Path(path).name}")
    print(f"  tensors {len(tensors)}, hyper {len(hyper_t)}, layers {len(layers)}")
    print(f"  meta: format={meta.get('kohakumoss_format')} algo={meta.get('kohakumoss_algo')}")

    fails = []
    if not hyper_t:
        fails.append("no nara_hyper tensors (the loader would refuse the file)")

    variants, with_sora, without_sora = set(), 0, 0
    for stem, parts in layers.items():
        sora, factors = mod.split_sora(parts)
        if sora:
            with_sora += 1
        else:
            without_sora += 1
        if not set(factors) & set(mod.SORA_NAMES):
            variants.add(mod.variant_of(factors))
        else:
            fails.append(f"{stem}: SORA keys leaked into the factor set")

    print(f"  variants {sorted(variants)}")
    print(f"  layers with SORA {with_sora}, without {without_sora}")

    # the hypernetwork rebuilds from the recorded config and moves with sigma (a
    # trained run has pushed C off the identity it starts at)
    hyper = mod.hyper_from(hyper_t, meta)
    sig = torch.tensor([0.0, 0.5, 1.0])
    cores = hyper(sig)
    eye = torch.eye(hyper.dim)
    devs = [(c - eye).abs().max().item() for c in cores]
    print(f"  hyper dim {hyper.dim}, cores {tuple(cores.shape)}, "
          f"max|C-I| by sigma {[f'{d:.4f}' for d in devs]}")
    if not any(d > 0 for d in devs):
        fails.append("the hypernetwork still emits the identity everywhere")

    # one real layer: the conditioning through the plugin's class
    stem, parts = next(iter(layers.items()))
    sora, factors = mod.split_sora(parts)
    cond = mod.SoraConditioning(sora)
    print(f"  sample layer {stem}")
    print(f"    keys {sorted(factors)} + sora {sorted(sora)}")
    print(f"    flags: post={cond.post} pre={cond.pre}")

    # a numeric check on that layer: Pre-Diag at m=1 must be d*base + delta
    r = torch.randn(3, 8, 64)
    if cond.pre:
        d = cond.diag.float()
        base = torch.randn(3, 8, d.numel())
        delta = torch.randn_like(base)
        got = cond.apply(base + delta, base, 1.0)
        want = d * base + delta if not cond.post else None
        if want is not None:
            err = (got - want).abs().max().item()
            ok = err < 1e-5
            print(f"    pre-diag m=1: {'ok ' if ok else 'FAIL'} max|err|={err:.2e}")
            if not ok:
                fails.append("pre-diag formula")

    if fails:
        print("\nFAILED:\n  " + "\n  ".join(fails))
        return 1
    print("\nfile loads as NaRA+SORA, all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
