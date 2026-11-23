"""Does the plugin's dynamic path reproduce the TRAINER for LoRA + Pre-Diag with no NaRA?

That is exactly this run's adapter: ALGO="lora", SORA_PRE_DIAG=True, NARA=False, so the
export carries sora_diag but no nara_hyper.*. The trainer's reference is a real forward
at a per-sample sigma; the plugin is run with hyper=None (C(sigma) = I) through the same
NaraAdapter.bypass_forward the Loader builds.
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
DIM = 32
OUT = D
M = 0.7  # a strength the test actually exercises (not 1.0, where Pre-Diag is identity)


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
    # a submodule registered in sys.modules is not an attribute of its parent package,
    # and nara.py calls comfy.model_prefetch.pause_malloc_graph by attribute
    for name in ("model_prefetch", "model_management", "ops", "weight_adapter"):
        setattr(stubs["comfy"], name, stubs[f"comfy.{name}"])
    stubs["comfy.weight_adapter"].base = stubs["comfy.weight_adapter.base"]
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
    """One LoRA + Pre-Diag layer exactly as this run configured it (NARA=False)."""
    lin = MossLinear(D, D, bias=False)
    lin.init_weights()
    block = nn.Module()
    block.proj = nn.ModuleDict({"q": lin})
    root = nn.Module()
    root.layers = nn.ModuleList([block])
    root.requires_grad_(False)
    cfg = AdapterConfig(
        algo="lora",
        nara=False,
        preset="attn",
        dim=DIM,
        alpha=float(DIM),
        backend="torch",
        sora_post_rank=0,
        sora_pre_diag=True,
        algo_kwargs={},
    )
    aset = apply(root, cfg, dtype=torch.float32, device="cpu")
    t = aset.targets[0]
    torch.manual_seed(4)
    with torch.no_grad():
        for _, p in aset.named_trainable():
            p.add_(torch.randn(p.shape) * 0.05)
        diag = getattr(t.lyco, "sora_diag", None)
        if diag is not None:
            diag.uniform_(0.5, 1.5)
    aset.bind()
    return t, aset


def main() -> int:
    plugin = load_plugin()
    t, _ = trainer_layer()
    lyco = t.lyco

    if not lyco.sora_pre_diag:
        print("FAIL: the trainer layer is not in Pre-Diag mode")
        return 1
    print(f"trainer layer: lora dim {lyco.lora_dim}, pre_diag True, "
          f"sora_diag range [{float(lyco.sora_diag.min()):.3f}, {float(lyco.sora_diag.max()):.3f}]")

    seq = build_seqinfo(torch.tensor([[2, 2], [2, 2]]), torch.tensor([2, 2]))
    set_context(torch.tensor([0.3, 0.8]), seq)
    x = torch.randn(12, D)
    with torch.no_grad():
        want = t.linear(x).clone()
    clear_context()

    # ---- the plugin's path: NaraAdapter with hyper=None (no C(sigma) axis) ----
    factors = {k: v.detach() for k, v in plugin_weights(lyco).items()}
    sora = {"sora_diag": lyco.sora_diag.detach()}
    alpha = float(lyco.scale) * lyco.lora_dim  # plugin re-derives scale as alpha / rank
    parts = [(None, alpha, factors, None, sora)]
    adapter = plugin.NaraAdapter(set(factors), parts, None, "lora", 0.5)
    adapter.multiplier = M

    def org_forward(xx, *a, **kw):
        return F.linear(xx, t.base_weight(torch.float32))

    plugin.set_sigma(0.5)  # any value: there is no hypernetwork to read it
    with torch.no_grad():
        got = adapter.bypass_forward(org_forward, x)
    plugin.set_sigma(None)

    # the trainer at the same strength: m scales both the delta and Pre-Diag's (d - 1)
    set_context(torch.tensor([0.3, 0.8]), seq)
    base = F.linear(x, t.base_weight(torch.float32))
    dm_ref = 1.0 + M * (lyco.sora_diag.to(torch.float32) - 1.0)
    delta_ref = F.linear(F.linear(x, lyco.lora_down.weight.float()) @ torch.eye(DIM),
                         lyco.lora_up.weight.float()) * (float(lyco.scale) * M)
    want_m = (dm_ref * base + delta_ref).to(base.dtype)
    clear_context()

    err = (got - want_m).abs().max().item()
    scale = want_m.abs().max().item()
    ok = err / scale < 2e-4
    print(f"plugin(hyper=None) vs trainer reference at strength m={M}")
    print(f"  max|err| {err:.3e}  relative {err / scale:.2e}  {'ok' if ok else 'FAIL'}")
    print(f"  (m=1.0 trainer forward differs by {(want - want_m).abs().max().item():.3e}: "
          f"Pre-Diag is the identity only at m=0/1, so m={M} is the real test)")

    # control: the same plugin adapter with Pre-Diag removed must match plain LoRA
    plain = plugin.NaraAdapter(set(factors), [(None, alpha, factors, None, {})], None, "lora", 0.5)
    plain.multiplier = M
    with torch.no_grad():
        got_plain = plain.bypass_forward(org_forward, x)
    delta_only = F.linear(
        F.linear(x, lyco.lora_down.weight.float()) @ torch.eye(DIM), lyco.lora_up.weight.float()
    ) * (float(lyco.scale) * M)
    err2 = (got_plain - (base + delta_only)).abs().max().item()
    print(f"control (sora stripped): max|err| {err2:.3e}  "
          f"relative {err2 / base.abs().max():.2e}  {'ok' if err2 / base.abs().max() < 2e-4 else 'FAIL'}")

    # the effect size: how far Pre-Diag moves the output at this diag spread
    moved = (want_m - (base + delta_only)).abs().max().item()
    print(f"Pre-Diag effect at this diag spread: max|out - plain_lora| {moved:.3e} "
          f"({100 * moved / scale:.2f}% of the output scale)")

    return 0 if ok else 1


def plugin_weights(lyco):
    """The per-layer tensor names the plugin reads for a lora variant. LyCORIS stores
    `lyco.lora_down` / `lyco.lora_up` as nn.Linear modules, so take their `.weight`."""
    return {
        "lora_down.weight": lyco.lora_down.weight,
        "lora_up.weight": lyco.lora_up.weight,
    }


if __name__ == "__main__":
    raise SystemExit(main())
