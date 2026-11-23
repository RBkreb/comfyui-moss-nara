"""Does the plugin's bypass path reproduce the TRAINER's forward for one layer?

The trainer is the reference: build one LoKr-NaRA+SORA layer there, run it, then run
the plugin's bypass math on the same weights and compare. This is the check that the
plugin's SORA conditioning is wired the way training defined it.
"""

import contextlib
import importlib.util
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "D:/comfyui/KohakuMossTrainer/src")

from kohakumoss.adapters import AdapterConfig, apply
from kohakumoss.adapters.nara import clear_context, set_context
from kohakumoss.model.linear import MossLinear
from kohakumoss.types.seqinfo import build_seqinfo

PLUGIN = "D:/comfyui/ComfyUI/custom_nodes/comfyui-moss-nara/nara.py"
D = 3072
DIM = 16
FACTOR = 8


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
    stubs["comfy.model_prefetch"].pause_malloc_graph = contextlib.nullcontext
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


def trainer_layer():
    lin = MossLinear(D, D, bias=False)
    lin.init_weights()
    block = nn.Module()
    block.proj = nn.ModuleDict({"q": lin})
    root = nn.Module()
    root.layers = nn.ModuleList([block])
    root.requires_grad_(False)
    cfg = AdapterConfig(
        algo="lokr",
        nara=True,
        preset="attn",
        dim=DIM,
        alpha=float(DIM),
        backend="torch",
        sora_post_rank=2,
        sora_pre_diag=True,
        algo_kwargs={"factor": FACTOR},
    )
    aset = apply(root, cfg, dtype=torch.float32, device="cpu")
    t = aset.targets[0]
    torch.manual_seed(4)
    with torch.no_grad():
        for _, p in aset.named_trainable():
            p.add_(torch.randn(p.shape) * 0.05)
        if getattr(t.lyco, "sora_diag", None) is not None:
            t.lyco.sora_diag.uniform_(0.7, 1.3)
    aset.bind()
    return t, aset


def main() -> int:
    plugin = load_plugin()
    t, aset = trainer_layer()
    lyco = t.lyco

    seq = build_seqinfo(torch.tensor([[2, 2], [2, 2]]), torch.tensor([2, 2]))
    sigma = torch.tensor([0.3, 0.8])
    set_context(sigma, seq)
    x = torch.randn(12, D)
    with torch.no_grad():
        want = t.linear(x).clone()
    clear_context()

    # the plugin's view of the same module
    cond = plugin.SoraConditioning(
        {
            "sora_C": lyco.sora_C.detach(),
            "sora_D": lyco.sora_D.detach(),
            "sora_s_p0": lyco.sora_s_p0,
            "sora_diag": lyco.sora_diag.detach() if lyco.sora_pre_diag else None,
        }
    )
    core = lyco.core_at(0.5).detach()  # one fixed sigma for the plugin's static path
    set_context(torch.tensor([0.5, 0.5]), seq)
    with torch.no_grad():
        want_mid = t.linear(x).clone()
    clear_context()

    base = F.linear(x, t.base_weight(torch.float32))
    delta = plugin.part_delta(
        "lokr",
        {k: v.detach() for k, v in plugin_weights(lyco).items()},
        core,
        float(lyco.scale),
        torch.float32,
    )
    delta = F.linear(x, delta)
    got = cond.apply(base + delta, base, 1.0)

    err = (got - want_mid).abs().max().item()
    scale = want_mid.abs().max().item()
    ok = err / scale < 2e-4
    print("trainer vs plugin, one LoKr-NaRA+SORA layer at sigma=0.5")
    print(f"  max|err| {err:.3e}  (relative {err / scale:.2e})  {'ok' if ok else 'FAIL'}")
    print(f"  (reference at the run's own per-sample sigma differs, as it must: "
          f"{(want - want_mid).abs().max().item():.3e})")
    return 0 if ok else 1


def plugin_weights(lyco):
    """The per-layer tensor names the plugin's part_delta expects."""
    out = {"alpha": torch.tensor(float(lyco.lora_dim))}
    w1 = lyco.lokr_w1 if lyco.use_w1 else lyco.lokr_w1_a @ lyco.lokr_w1_b
    out["lokr_w1"] = w1
    out["lokr_w2_a"] = lyco.lokr_w2_a
    out["lokr_w2_b"] = lyco.lokr_w2_b
    return out


if __name__ == "__main__":
    raise SystemExit(main())
