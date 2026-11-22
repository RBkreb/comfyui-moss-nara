"""GPU device test: the SORA tensors stay on CPU (BypassForwardHook only moves
`adapter.weights`), so any place that multiplies them with a cuda activation must place
them itself. This reproduces the real failure and then checks the fix.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/check_loader_device.py
"""

import importlib.util
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

NARA = "D:/comfyui/ComfyUI/custom_nodes/comfyui-moss-nara/nara.py"
FILE = "D:/comfyui/ComfyUI/models/loras/moss/lora-prediag-rigid-ema-lycoris-fp32.safetensors"
D = 3072
M = 0.7


def load_plugin():
    spec = importlib.util.spec_from_file_location("moss_nara_plugin", NARA)
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
    stubs["comfy.model_management"].get_torch_device = lambda: torch.device("cuda")
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


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no cuda")
        return 0
    dev = torch.device("cuda")
    plugin = load_plugin()

    with safe_open(FILE, "pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        meta = f.metadata() or {}
    tensors, meta = plugin.normalize_ema(tensors, meta, FILE)
    _, rest = plugin.split_hyper(tensors)
    layers = plugin.group_nara_layers(rest)

    stem = "lora_unet_layers_0_proj_q"
    parts = layers[stem]
    sora, factors = plugin.split_sora(parts)
    alpha = float(factors.pop("alpha").item()) if "alpha" in factors else None

    # what the hook does: move `weights` to the device, leave the SORA tensors alone
    moved = {k: v.to(dev) for k, v in factors.items()}
    adapter = plugin.NaraAdapter(set(moved), [(None, alpha, moved, None, sora)], None, "lora", 0.5)
    adapter.multiplier = M
    adapter.weights = tuple(moved.values())  # exactly what _move_adapter_weights_to_device binds

    print("devices after the hook:")
    print(f"  weights (moved by hook) : {adapter.weights[0].device}")
    print(f"  sora_diag (untouched)   : {adapter.sora[0].diag.device}  <- the hazard")

    # the output is bf16, so the reference can only be matched to bf16's own resolution:
    # a bf16 ULP at 1.0 is 7.8e-3, and every rounding of base / delta / d_m base compounds
    # it. The check is that the math is the right math, not that it is bf16-exact.
    base_w = (torch.randn(D, D, device=dev, generator=torch.Generator(dev).manual_seed(0)) * 0.02).to(torch.bfloat16)
    x = torch.randn(8, D, device=dev, dtype=torch.bfloat16, generator=torch.Generator(dev).manual_seed(1))

    def org_forward(xx, *a, **kw):
        return F.linear(xx, base_w)

    plugin.set_sigma(0.5)
    try:
        with torch.no_grad():
            got = adapter.bypass_forward(org_forward, x)
    except RuntimeError as err:
        plugin.set_sigma(None)
        print(f"\nFAIL: {err}")
        print("the SORA tensors must be placed on the activation device at use time")
        return 1
    plugin.set_sigma(None)

    ts = adapter._tensors(adapter.parts[0][2])
    base = org_forward(x)
    dm = 1.0 + M * (sora["sora_diag"].float().to(dev) - 1.0)
    rank = ts["lora_down.weight"].shape[0]
    scale = (1.0 if alpha is None else alpha / rank) * M
    xf = x.float()
    delta = (
        F.linear(F.linear(xf, ts["lora_down.weight"].float()), ts["lora_up.weight"].float())
        * scale
    )
    want = dm * base.float() + delta.float()
    err = (got.float() - want).abs().max().item()
    rel = err / want.abs().max().item()
    plain = base.float() + delta.float()
    moved_by = (want - plain).abs().max().item()

    # bf16's own resolution near 1.0 sets the floor: |a| ~ 1 rounds to 7.8e-3, and the
    # base / delta / dm*base roundings compound to a few ULP. Compare against the
    # tensor's ULP scale rather than an absolute epsilon.
    ulp = 2 ** (-8)  # bf16 has 8 explicit mantissa bits
    tol = 4 * ulp
    print(f"\nbypass forward on {dev} (bf16 base, fp32 SORA)")
    print(f"  relative error {rel:.2e}  (bf16 tolerance {tol:.1e})  "
          f"{'ok' if rel < tol else 'FAIL'}")
    print(f"  output {tuple(got.shape)} {got.dtype} on {got.device}")
    print(f"  Pre-Diag moved it by {moved_by:.3e} "
          f"({100 * moved_by / want.abs().max().item():.2f}% of scale)")
    return 0 if rel < tol else 1


if __name__ == "__main__":
    sys.exit(main())
