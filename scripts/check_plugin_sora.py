"""Plugin-side SORA check, no ComfyUI: the plugin's SoraConditioning must reproduce the
trainer's fp64 formulas for one linear.

    python scripts/check_plugin_sora.py
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PLUGIN = Path("D:/comfyui/ComfyUI/custom_nodes/comfyui-moss-nara/nara.py")
OUT, IN, RP = 24, 32, 2


def load_plugin():
    spec = importlib.util.spec_from_file_location("moss_nara_plugin", PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    stubs = {
        "comfy": type(sys)("comfy"),
        "comfy.model_management": type(sys)("comfy.model_management"),
        "comfy.model_prefetch": type(sys)("comfy.model_prefetch"),
        "comfy.ops": type(sys)("comfy.ops"),
        "comfy.weight_adapter": type(sys)("comfy.weight_adapter"),
        "comfy.weight_adapter.base": type(sys)("comfy.weight_adapter.base"),
    }

    class Base:
        pass

    stubs["comfy.weight_adapter.base"].WeightAdapterBase = Base
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


def ref_rotation(c, d, s_p0, out_dim):
    cf, df = c.double(), d.double()
    s_p = min(1.0, s_p0 / (float(cf.norm()) + float(df.norm()) + 1e-6))
    eye = torch.eye(out_dim, dtype=torch.float64)
    return eye + s_p * (df @ cf.T - cf @ df.T)


def main() -> int:
    mod = load_plugin()
    torch.manual_seed(5)
    c = torch.randn(OUT, RP) * 0.05
    d = torch.randn(OUT, RP) * 0.05
    diag = torch.rand(OUT) * 2.0

    x = torch.randn(6, IN)
    w = torch.randn(OUT, IN) * 0.1
    dw = torch.randn(OUT, IN) * 0.02
    base = F.linear(x, w)
    total = F.linear(x, w + dw)

    fails = []

    def check(name, got, want, tol):
        err = (got.double() - want).abs().max().item()
        ok = err < tol
        print(f"  {name:34s} {'ok ' if ok else 'FAIL'} max|err|={err:.3e}")
        if not ok:
            fails.append(name)

    # --- Post-Ortho only ---
    cond = mod.SoraConditioning({"sora_C": c, "sora_D": d, "sora_s_p0": torch.tensor(1.0)})
    got = cond.apply(total, base, 1.0)
    want = F.linear(x.double(), ref_rotation(c, d, 1.0, OUT) @ (w + dw).double())
    check("post-ortho m=1", got, want, 2e-6)

    # --- Pre-Diag only: d_m * base + delta ---
    cond_pre = mod.SoraConditioning({"sora_diag": diag})
    got = cond_pre.apply(total, base, 1.0)
    want = diag.double() * F.linear(x.double(), w.double()) + F.linear(
        x.double(), dw.double()
    )
    check("pre-diag m=1", got, want, 2e-6)

    # --- both: R (diag(d) W x + dW x) ---
    cond_both = mod.SoraConditioning(
        {"sora_C": c, "sora_D": d, "sora_s_p0": torch.tensor(1.0), "sora_diag": diag}
    )
    got = cond_both.apply(total, base, 1.0)
    rot = ref_rotation(c, d, 1.0, OUT)
    want = F.linear(
        x.double(), rot @ (diag.double().reshape(-1, 1) * w.double() + dw.double())
    )
    check("post + pre m=1", got, want, 2e-6)

    # --- post-ortho at m=0 is the identity ---
    got = mod.SoraConditioning(
        {"sora_C": c, "sora_D": d, "sora_s_p0": torch.tensor(1.0)}
    ).apply(total, base, 0.0)
    check("post-ortho m=0 = plain", got, total.double(), 2e-6)

    # --- the bound: a huge C tracks s_p0 ---
    big = torch.randn(OUT, RP) * 100
    got = mod.SoraConditioning(
        {"sora_C": big, "sora_D": d, "sora_s_p0": torch.tensor(0.5)}
    ).apply(total, base, 1.0)
    want = F.linear(x.double(), ref_rotation(big, d, 0.5, OUT) @ (w + dw).double())
    check("post-ortho s_p0 bound", got, want, 2e-6)

    # --- split_sora separates exactly ---
    parts = {
        "lokr_w1_a": torch.randn(4, 2),
        "lokr_w2_b": torch.randn(2, 8),
        "alpha": torch.tensor(4.0),
        "sora_C": c,
        "sora_D": d,
        "sora_s_p0": torch.tensor(1.0),
    }
    sora, rest = mod.split_sora(parts)
    ok = set(sora) == {"sora_C", "sora_D", "sora_s_p0"} and not (
        set(rest) & set(mod.SORA_NAMES)
    )
    print(f"  {'split_sora separates':34s} {'ok ' if ok else 'FAIL'}")
    if not ok:
        fails.append("split_sora")

    print("\nFAILED: " + ", ".join(fails) if fails else "\nall checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
