"""Load every real adapter in models/loras/moss through the plugin's parse path.

The failure this catches: `_out_features` / `_in_features` unpacked the per-slice tuple
as 4 values while `_map_nara_layers` builds 5 (offset, alpha, factors, dora, sora). Only
a DoRA file reaches that code (the `has_dora` branch), so a LoRA-only test missed it.
"""

import importlib.util
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

NARA = "D:/comfyui/ComfyUI/custom_nodes/comfyui-moss-nara/nara.py"
NODES = "D:/comfyui/ComfyUI/custom_nodes/comfyui-moss-nara/nodes.py"
LORAS = "D:/comfyui/ComfyUI/models/loras/moss"
FILES = [
    "lora-nara-dora-moss2-1500.safetensors",
    "lora-nara-dora-moss2-1500-ema.safetensors",
    "lora-prediag-rigid-lycoris-fp32.safetensors",
    "lora-prediag-rigid-ema-lycoris-fp32.safetensors",
    "moss-ani-1000-nara.safetensors",
]
D = 3072


def _stub_comfy():
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
    for name in ("model_prefetch", "model_management", "ops", "weight_adapter"):
        setattr(stubs["comfy"], name, stubs[f"comfy.{name}"])
    stubs["comfy.weight_adapter"].base = stubs["comfy.weight_adapter.base"]
    return stubs


def load_plugin():
    spec = importlib.util.spec_from_file_location("moss_nara_plugin", NARA)
    mod = importlib.util.module_from_spec(spec)
    stubs = _stub_comfy()
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


def load_nodes_map(plugin):
    import ast

    src = open(NODES, encoding="utf-8").read()
    fn = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "_map_nara_layers"
    )
    code = ast.get_source_segment(src, fn)

    fake_lora = type(sys)("comfy.lora")
    fake_lora.model_lora_keys_unet = lambda model, key_map=None: {
        f"lora_unet_layers_{i}_proj_{n}": f"layers.{i}.proj.{n}.weight"
        for i in range(TEST_LAYERS)
        for n in ("q", "k", "v", "o", "ffn_down", "ffn_up_gate")
    }
    stubs = _stub_comfy()
    stubs["comfy"].lora = fake_lora
    stubs["comfy.lora"] = fake_lora
    ns = {**{k: getattr(plugin, k) for k in dir(plugin)}, "torch": torch,
          "logging": __import__("logging")}
    ns["comfy"] = stubs["comfy"]
    exec(compile(code, NODES, "exec"), ns)
    return ns["_map_nara_layers"]


TEST_LAYERS = 3


class Linear(nn.Module):
    def __init__(self, out_f, in_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)


class Patcher:
    def __init__(self):
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(TEST_LAYERS):
            blk = nn.Module()
            blk.proj = nn.ModuleDict(
                {n: Linear(D, D) for n in ("q", "k", "v", "o", "ffn_down", "ffn_up_gate")}
            )
            self.model.layers.append(blk)


def main() -> int:
    plugin = load_plugin()
    map_layers = load_nodes_map(plugin)
    bad = 0
    for name in FILES:
        path = f"{LORAS}/{name}"
        try:
            with safe_open(path, "pt") as f:
                tensors = {k: f.get_tensor(k) for k in f.keys()}
                meta = f.metadata() or {}
        except Exception as err:
            print(f"{name:<46} SKIP ({type(err).__name__})")
            continue

        hyper_t, _ = plugin.split_hyper(tensors)
        dyn = bool(hyper_t) or plugin.carries_sora(tensors)
        route = "dynamic" if dyn else "static"
        try:
            tensors, meta = plugin.normalize_ema(tensors, meta, path)
            if not dyn:
                print(f"{name:<46} {route} (native patches)")
                continue
            hyper_t, rest = plugin.split_hyper(tensors)
            hyper = plugin.hyper_from(hyper_t, meta) if hyper_t else None
            layers = plugin.group_nara_layers(rest)
            groups, loaded, variant = map_layers(Patcher(), layers)
            if not groups:
                print(f"{name:<46} {route} but 0 groups (key map miss)")
                continue
            # build every adapter, exactly as _load_nara does
            n_dora = n_sora = 0
            for wkey, parts in groups.items():
                parts.sort(key=lambda p: p[0][1] if p[0] is not None else 0)
                ad = plugin.NaraAdapter(loaded, parts, hyper, variant, 0.5)
                n_dora += int(ad.has_dora)
                n_sora += sum(1 for c in ad.sora if c.any)
            flags = []
            if hyper is not None:
                flags.append("hyper")
            if n_sora:
                flags.append("sora")
            if n_dora:
                flags.append("dora")
            print(f"{name:<46} {route} {variant} groups={len(groups)} "
                  f"sora={n_sora} dora={n_dora} [{'/'.join(flags) or 'plain'}] ok")
        except Exception as err:
            bad += 1
            print(f"{name:<46} {route} FAIL: {type(err).__name__}: {err}")
    print()
    print("all files parsed" if bad == 0 else f"{bad} file(s) failed")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
