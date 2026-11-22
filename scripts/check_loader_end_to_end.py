"""End-to-end: load the real fp32 export through the patched plugin's LOADER PATH.

No ComfyUI runtime (the model would need a GPU and the whole stack), so this drives the
same functions MossNara_Loader.execute calls -- normalize_ema, split_hyper, carries_sora,
group_nara_layers, _map_nara_layers and NaraAdapter construction -- on a synthetic key
map standing in for comfy.lora.model_lora_keys_unet, then runs one bypass forward and
checks the emitted Pre-Diag conditioning is the trainer's formula.
"""

import importlib.util
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

NARA = "D:/comfyui/ComfyUI/custom_nodes/comfyui-moss-nara/nara.py"
NODES = "D:/comfyui/ComfyUI/custom_nodes/comfyui-moss-nara/nodes.py"
FILE = "D:/comfyui/ComfyUI/models/loras/moss/lora-prediag-rigid-ema-lycoris-fp32.safetensors"
D = 3072
M = 0.7


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
    """`_map_nara_layers` lives in nodes.py (needs the whole ComfyUI node API), so lift
    just that function's source and exec it against the plugin's globals plus a stub
    comfy.lora that resolves the same way `model_lora_keys_unet` does for a linear."""
    import ast

    src = open(NODES, encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_map_nara_layers"
    )
    code = ast.get_source_segment(src, fn)

    fake_lora = type(sys)("comfy.lora")

    def model_lora_keys_unet(model, key_map=None):
        key_map = {} if key_map is None else key_map
        for i in range(len(model.layers)):
            for n in ("q", "k", "v", "o", "ffn_down", "ffn_up_gate"):
                key_map[f"lora_unet_layers_{i}_proj_{n}"] = f"layers.{i}.proj.{n}.weight"
        return key_map

    fake_lora.model_lora_keys_unet = model_lora_keys_unet
    # the plugin loader popped its stubs, so re-install them for this call
    stubs = _stub_comfy()
    stubs["comfy"].lora = fake_lora
    stubs["comfy.lora"] = fake_lora
    comfy = stubs["comfy"]
    logging_stub = type(sys)("logging")
    ns = {
        **{k: getattr(plugin, k) for k in dir(plugin)},
        "torch": torch,
        "logging": __import__("logging"),
    }
    ns["comfy"] = comfy  # after the plugin's own names: `dir(plugin)` also has `comfy`
    exec(compile(code, NODES, "exec"), ns)
    return ns["_map_nara_layers"]


class Linear(nn.Module):
    """torch.empty leaves the weight at whatever the allocator hands back (often zeros,
    which would make every comparison here trivially 0), so initialize it."""

    def __init__(self, out_f, in_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = None

    def forward(self, x):
        return F.linear(x, self.weight)


class Model(nn.Module):
    """`MossNara_Loader` sees a comfy ModelPatcher; `_map_nara_layers` reads `model.model`
    and calls `get_submodule` on it, so a tiny holder is enough."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(2):
            blk = nn.Module()
            blk.proj = nn.ModuleDict(
                {n: Linear(D, D) for n in ("q", "k", "v", "o", "ffn_down", "ffn_up_gate")}
            )
            self.layers.append(blk)


class Patcher:
    """What the node's `model` argument actually is: an object with `.model`."""

    def __init__(self):
        self.model = Model()

    def get_submodule(self, path):
        return self.model.get_submodule(path)


def main() -> int:
    plugin = load_plugin()

    with safe_open(FILE, "pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        meta = f.metadata() or {}
    print(f"file: {FILE.rsplit('/', 1)[1]}  tensors {len(tensors)}  meta {sorted(meta)[:3]}")

    tensors, meta = plugin.normalize_ema(tensors, meta, FILE)
    hyper_t, _ = plugin.split_hyper(tensors)
    routed = "dynamic" if (hyper_t or plugin.carries_sora(tensors)) else "static"
    print(f"loader routing: hyper tensors {len(hyper_t)}, carries_sora "
          f"{plugin.carries_sora(tensors)} -> {routed}")
    if routed != "dynamic":
        print("FAIL: the loader would still take the static path and drop sora_diag")
        return 1

    # the loader's own steps, with a synthetic key map for the first two blocks
    model = Patcher()

    _, rest = plugin.split_hyper(tensors)
    layers = plugin.group_nara_layers(rest)
    map_layers = load_nodes_map(plugin)
    groups, loaded, variant = map_layers(model, layers)
    print(f"matched {len(groups)} weights of {len(layers)} in the file, variant {variant}")

    covered = sum(len(parts) for parts in groups.values())
    with_diag = sum(1 for parts in groups.values() for *_, s in parts if s)
    print(f"slices covered {covered}, slices carrying SORA {with_diag}")
    if with_diag != covered:
        print("FAIL: some matched slice lost its SORA tensors")

    # one real bypass forward on layer 0 / q through the patched NaraAdapter
    wk = "layers.0.proj.q.weight"
    parts = sorted(groups[wk], key=lambda p: p[0][1] if p[0] is not None else 0)
    adapter = plugin.NaraAdapter(loaded, parts, None, variant, 0.5)
    adapter.multiplier = M
    module = model.model.layers[0].proj["q"]
    x = torch.randn(8, D)

    def org_forward(xx, *a, **kw):
        return F.linear(xx, module.weight.float())

    plugin.set_sigma(0.5)
    with torch.no_grad():
        got = adapter.bypass_forward(org_forward, x)
    plugin.set_sigma(None)

    ts = parts[0][2]
    sora = parts[0][4]
    base = org_forward(x)
    dm = 1.0 + M * (sora["sora_diag"].float() - 1.0)
    alpha = float(parts[0][1]) if parts[0][1] is not None else None
    rank = ts["lora_down.weight"].shape[0]
    scale = (1.0 if alpha is None else alpha / rank) * M
    delta = F.linear(F.linear(x, ts["lora_down.weight"].float()), ts["lora_up.weight"].float()) * scale
    want = dm * base + delta

    err = (got - want).abs().max().item()
    rel = err / want.abs().max().item()
    plain = base + delta
    moved = (want - plain).abs().max().item()
    print(f"bypass forward (hyper=None, m={M}): max|err| {err:.3e} relative {rel:.2e} "
          f"{'ok' if rel < 2e-4 else 'FAIL'}")
    print(f"Pre-Diag moved the output by {moved:.3e} "
          f"({100 * moved / want.abs().max().item():.2f}% of scale) -- 0.0 would mean it is inert")
    return 0 if rel < 2e-4 and moved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
