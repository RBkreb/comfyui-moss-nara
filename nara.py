"""NaRA (Noise-aware Rank Adaptation) for MOSS-Image, ported from KohakuMossTrainer.

h = W0 x + s B C(sigma) A x, with C(sigma) = I + c_scale F(e_sigma) produced by one
shared hypernetwork. Runs as a bypass weight adapter: base weights (including
quantized ones) are never modified, the sigma-dependent path is added in forward.
"""

import json

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

import comfy.model_management
from comfy.weight_adapter.base import WeightAdapterBase

EMBEDS = ("fourier", "mlp", "raw")
HYPER_STEM = "nara_hyper"


class NaraHyper(nn.Module):
    """sigma [B] -> C [B, dim, dim] in fp32; same structure as the trainer's."""

    def __init__(
        self,
        dim,
        hidden1=256,
        hidden2=512,
        embed_dim=64,
        embed="fourier",
        fourier_scale=16.0,
        c_scale=1.0,
    ):
        super().__init__()
        if embed not in EMBEDS:
            raise ValueError(f"nara_embed={embed!r} not in {EMBEDS}")
        if embed == "fourier" and embed_dim % 2:
            raise ValueError(f"nara_embed_dim must be even for fourier, got {embed_dim}")
        self.dim = dim
        self.c_scale = float(c_scale)
        self.embed = embed
        if embed == "fourier":
            self.register_buffer("fourier_w", torch.empty(1, embed_dim // 2))
            input_dim = embed_dim
        elif embed == "mlp":
            self.embed_proj = nn.Linear(1, embed_dim)
            input_dim = embed_dim
        else:
            input_dim = 1
        self.mapper = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.SiLU(),
            nn.Linear(hidden1, hidden2),
            nn.SiLU(),
            nn.Linear(hidden2, dim * dim),
        )
        self.register_buffer("eye", torch.eye(dim), persistent=False)

    def forward(self, sigma):
        x = sigma.float().reshape(-1, 1)
        if self.embed == "fourier":
            proj = x @ self.fourier_w.to(x.dtype)
            x = torch.cat((proj.sin(), proj.cos()), dim=-1)
        elif self.embed == "mlp":
            x = F.silu(self.embed_proj(x))
        flat = self.mapper(x)
        return self.eye + self.c_scale * flat.view(-1, self.dim, self.dim)


def hyper_from(tensors, meta):
    """Rebuild the hypernetwork from its tensors and the configuration the file records."""
    spec = meta.get("kohakumoss_spec")
    if spec:
        fields = json.loads(spec)
        dim, kwargs = int(fields["dim"]), fields.get("algo_kwargs", {})
    elif "ss_network_args" in meta:
        dim = int(meta["ss_network_dim"])
        kwargs = json.loads(meta["ss_network_args"])
    else:
        raise ValueError("NaRA file has no recorded configuration (kohakumoss_spec or ss_network_args)")
    kwargs = {k[5:]: v for k, v in kwargs.items() if k.startswith("nara_")}
    hyper = NaraHyper(dim, **kwargs)
    hyper.load_state_dict(tensors)
    return hyper


def load_nara(path):
    """(hypernetwork, [(lora key, up, down, alpha)]) from a NaRA safetensors file."""
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        keys = list(f.keys())
        prefix = HYPER_STEM + "."
        hyper_tensors = {k[len(prefix):]: f.get_tensor(k) for k in keys if k.startswith(prefix)}
        if not hyper_tensors:
            raise ValueError(
                f"{path}: no {HYPER_STEM}.* tensors; a fixed-sigma NaRA export is a plain LoRA, load it with LoraLoader"
            )
        layers = []
        for k in keys:
            if not k.endswith(".lora_up.weight"):
                continue
            stem = k[: -len(".lora_up.weight")]
            up = f.get_tensor(k)
            down = f.get_tensor(stem + ".lora_down.weight")
            alpha_name = stem + ".alpha"
            alpha = f.get_tensor(alpha_name).item() if alpha_name in keys else None
            layers.append((stem, up, down, alpha))
    return hyper_from(hyper_tensors, meta), layers


# The sampling sigma of the current model evaluation, set by the NaRA KSampler's
# predict-noise wrapper. One value per evaluation: cond and uncond share it.
_sigma = None


def set_sigma(sigma):
    global _sigma
    _sigma = sigma


def _core(hyper, sigma):
    """C(sigma), cached per hypernetwork (several adapters may share one sigma)."""
    cache = getattr(hyper, "_core_cache", None)
    if cache is None or cache[0] != sigma:
        device = next(hyper.parameters()).device
        s = torch.tensor([sigma], device=device, dtype=torch.float32)
        cache = (sigma, hyper(s)[0])
        hyper._core_cache = cache
    return cache[1]


class NaraAdapter(WeightAdapterBase):
    """NaRA over one weight: `parts` are (offset, alpha, up, down) per adapted slice
    (a fused qkv weight carries three). `weights` holds the (up, down) pairs flat so
    BypassForwardHook moves them to the compute device; parts index into it."""

    name = "nara"

    def __init__(self, loaded_keys, parts, hyper, export_sigma=0.5):
        self.loaded_keys = loaded_keys
        self.parts = parts
        self.hyper = hyper
        self.export_sigma = float(export_sigma)
        self.weights = tuple(t for _, _, up, down in parts for t in (up, down))

    def h(self, x, base_out):
        if _sigma is None:
            raise RuntimeError("NaRA: no sampling sigma set; sample with the NaRA KSampler")
        core = _core(self.hyper, _sigma).to(device=x.device, dtype=x.dtype)
        multiplier = getattr(self, "multiplier", 1.0)
        out = None
        for i, (offset, alpha, _, _) in enumerate(self.parts):
            up = self.weights[2 * i].to(x.dtype)
            down = self.weights[2 * i + 1].to(x.dtype)
            scale = (alpha / down.shape[0] if alpha is not None else 1.0) * multiplier
            hidden = F.linear(x, down) @ core
            delta = F.linear(hidden, up) * scale
            if offset is None:
                out = delta if out is None else out + delta
            else:
                if out is None:
                    out = torch.zeros_like(base_out)
                out.narrow(-1, offset[1], offset[2]).add_(delta)
        return out

    def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
        """Merge mode: NaRA at the fixed export_sigma as a plain LoRA diff."""
        device = weight.device
        hyper_device = next(self.hyper.parameters()).device
        s = torch.tensor([self.export_sigma], device=hyper_device, dtype=torch.float32)
        core = self.hyper(s)[0].to(device=device, dtype=intermediate_dtype)
        for i, (off, alpha, _, _) in enumerate(self.parts):
            up = comfy.model_management.cast_to_device(self.weights[2 * i], device, intermediate_dtype)
            down = comfy.model_management.cast_to_device(self.weights[2 * i + 1], device, intermediate_dtype)
            w = weight if off is None else weight.narrow(*off)
            scale = alpha / down.shape[0] if alpha is not None else 1.0
            diff = (up @ core.T) @ down
            w += function((strength * scale * diff).type(w.dtype))
        return weight