"""NaRA (Noise-aware Rank Adaptation) for MOSS-Image, ported from KohakuMossTrainer.

h = W0 x + s B C(sigma) A x, with C(sigma) = I + c_scale F(e_sigma) produced by one
shared hypernetwork. Runs as a bypass weight adapter: base weights (including
quantized ones) are never modified, the sigma-dependent path is added in forward.
LoHa / LoKr variants carry the core in their rank factors the same way the trainer
does; DoRA normalizes after the live BA*C(sigma), per forward, in fp32.
"""

import json
import logging
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
from comfy.weight_adapter.base import WeightAdapterBase

EMBEDS = ("fourier", "mlp", "raw")
HYPER_STEM = "nara_hyper"

# Per-layer tensor names of the trainer's NaRA exports (lycoris / raw state).
LORA_NAMES = ("lora_up.weight", "lora_down.weight")
LOHA_NAMES = ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")
LOKR_NAMES = ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b")
COMMON_NAMES = ("alpha", "dora_scale")
# SORA (arXiv:2511.00051) conditioning of any additive kind: a bounded skew rotation of
# the layer's whole output and / or a learned per-channel scale of the base output.
SORA_NAMES = ("sora_C", "sora_D", "sora_diag", "sora_s_p0")
NARA_NAMES = LORA_NAMES + LOHA_NAMES + LOKR_NAMES + COMMON_NAMES + SORA_NAMES


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


def read_adapter(path):
    """(tensors, metadata) of an adapter file, on CPU."""
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    return tensors, meta


def normalize_ema(tensors, meta, path):
    """An EMA checkpoint's tensors carry an `ema<decay>/` prefix and no metadata:
    strip it, and take the configuration plus the hypernetwork's fixed buffers
    (fourier_w is never averaged) from the sibling trainable.safetensors. A copy
    with the metadata and buffers already embedded needs no sibling."""
    prefixed = [k for k in tensors if k.startswith("ema") and "/" in k]
    if not prefixed:
        return tensors, meta
    prefix = prefixed[0].split("/", 1)[0] + "/"
    stripped = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
    have_meta = bool(meta.get("kohakumoss_spec") or meta.get("ss_network_args"))
    sibling = os.path.join(os.path.dirname(path), "trainable.safetensors")
    if not have_meta and not os.path.isfile(sibling):
        raise ValueError(f"{path}: an EMA state without metadata needs its sibling trainable.safetensors")
    if os.path.isfile(sibling):
        with safe_open(sibling, framework="pt", device="cpu") as f:
            if not have_meta:
                meta = f.metadata() or {}
            for k in f.keys():
                if k.startswith(HYPER_STEM + ".") and k not in stripped:
                    stripped[k] = f.get_tensor(k)
    return stripped, meta


def split_hyper(tensors):
    """(hypernetwork tensors under their own names, the per-layer tensors)."""
    prefix = HYPER_STEM + "."
    hyper = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
    rest = {k: v for k, v in tensors.items() if not k.startswith(prefix)}
    return hyper, rest


def group_nara_layers(tensors):
    """stem -> {tensor name: tensor} of the per-layer NaRA tensors."""
    layers = {}
    for k, v in tensors.items():
        for name in NARA_NAMES:
            suffix = "." + name
            if k.endswith(suffix):
                layers.setdefault(k[: -len(suffix)], {})[name] = v
                break
    return layers


def variant_of(parts):
    """The NaRA variant of one layer's tensor set."""
    if "lora_up.weight" in parts:
        return "lora"
    if "hada_w1_a" in parts:
        return "loha"
    if parts.keys() & {"lokr_w1", "lokr_w1_a", "lokr_w2_a"}:
        if "lokr_w2" in parts:
            raise ValueError("NaRA LoKr needs a factorized w2 (lokr_w2_a / lokr_w2_b)")
        return "lokr"
    raise KeyError(f"no NaRA variant matches {sorted(parts)}")


def rank_of(variant, parts):
    if variant == "lora":
        return parts["lora_down.weight"].shape[0]
    if variant == "loha":
        return parts["hada_w1_b"].shape[0]
    return parts["lokr_w2_b"].shape[0]


def part_delta(variant, parts, core, scale, dtype):
    """One layer's dense dW at the given core, in `dtype` (fp32 for the DoRA norm)."""
    if variant == "lora":
        up = parts["lora_up.weight"].to(dtype)
        down = parts["lora_down.weight"].to(dtype)
        return (up @ core.T @ down) * scale
    if variant == "loha":
        w1a, w1b = parts["hada_w1_a"].to(dtype), parts["hada_w1_b"].to(dtype)
        w2a, w2b = parts["hada_w2_a"].to(dtype), parts["hada_w2_b"].to(dtype)
        return ((w1a @ core.T @ w1b) * (w2a @ core.T @ w2b)) * scale
    w1 = (
        parts["lokr_w1"].to(dtype)
        if "lokr_w1" in parts
        else parts["lokr_w1_a"].to(dtype) @ parts["lokr_w1_b"].to(dtype)
    )
    w2 = parts["lokr_w2_a"].to(dtype) @ core.T @ parts["lokr_w2_b"].to(dtype)
    return torch.kron(w1, w2) * scale


def split_sora(parts):
    """(the SORA tensors of one layer, the remaining per-layer set)."""
    sora = {k: parts[k] for k in SORA_NAMES if k in parts}
    return sora, {k: v for k, v in parts.items() if k not in SORA_NAMES}


def map_nara_layers(key_map, layers):
    """Resolve file layers to weight keys; group slices of one weight (fused qkv).

    `key_map` is what `comfy.lora.model_lora_keys_unet` returns: stem -> weight key,
    or a tuple when several slices fuse into one weight. Passing the map in (rather
    than walking the model) keeps this usable without a ComfyUI runtime.
    """
    groups = {}
    loaded_keys = set()
    variant = None
    for stem, parts in layers.items():
        target = key_map.get(stem)
        if target is None:
            logging.warning("[moss-nara] key not found in model: %s", stem)
            continue
        v = variant_of(parts)
        if variant is None:
            variant = v
        elif variant != v:
            raise ValueError(f"{stem}: mixed NaRA variants ({variant} and {v}) in one file")
        loaded_keys.update(stem + "." + name for name in parts)
        weight_key, offset = target if isinstance(target, tuple) else (target, None)
        alpha = parts.get("alpha")
        alpha = float(alpha.item()) if alpha is not None else None
        dora = parts.get("dora_scale")
        sora, factors = split_sora(parts)
        groups.setdefault(weight_key, []).append(
            (offset, alpha, factors, dora, sora or None)
        )
    return groups, loaded_keys, variant


class SoraConditioning:
    """SORA's two switches on one weight, ported from the trainer's `adapters/sora.py`.

    Post-Ortho: y -> y + m (R - I) y with R = I + s_p (D C^T - C D^T), a bounded skew
    rotation of the whole layer output (base + delta). Pre-Diag: the base output's
    per-channel scale d replaces DoRA, y -> d_m fn(x) + (1 - d_m) delta_m(x) with
    d_m = 1 + m (d - 1).

    Both read the layer's output and its own bypass delta, never a dense weight, so
    they stack on quantized bases. All math fp32; the rank term is never materialised
    as an out x out matrix."""

    def __init__(self, sora: dict):
        self.C = sora.get("sora_C")
        self.D = sora.get("sora_D")
        s_p0 = sora.get("sora_s_p0")
        self.s_p0 = 1.0 if s_p0 is None else float(s_p0.reshape(()))
        self.diag = sora.get("sora_diag")
        self._factors_cache = None
        self._diag_cache = None

    @property
    def post(self) -> bool:
        return self.C is not None

    @property
    def pre(self) -> bool:
        return self.diag is not None

    @property
    def any(self) -> bool:
        return self.post or self.pre

    def _factors(self, device=None):
        """(C, D, s_p) fp32: s_p = min(1, s_p0 / (||C|| + ||D||)), the trainer's bound.
        The SORA tensors are read from the file on CPU and are NOT part of the adapter's
        `weights` tuple, so BypassForwardHook never moves them: place them on the
        activation's device once and cache, they are constant across forwards."""
        cache = self._factors_cache
        if cache is not None and cache[0] == device:
            return cache[1]
        c = self.C.float().to(device)
        d = self.D.float().to(device)
        s_p = min(1.0, self.s_p0 / (float(c.norm()) + float(d.norm()) + 1e-6))
        factors = (c, d, s_p)
        self._factors_cache = (device, factors)
        return factors

    def rotate_out(self, y):
        """(R - I) y for activation rows y (..., out), fp32."""
        c, d, s_p = self._factors(y.device)
        yf = y.float()
        return s_p * ((yf @ c) @ d.T - (yf @ d) @ c.T)

    def apply(self, total, base_out, m: float):
        """The conditioning on one slice's output at strength m. `total` is
        base + delta, `base_out` the base alone (the same columns).

        Post-Ortho: rotate the whole output, `total + m (R - I) total`. Pre-Diag: the
        trainer's `d_m fn(x) + (1 - d_m) delta(x)` collapses to `d_m base + delta` —
        the base output scaled per channel, the delta left at full strength — so the
        delta is NOT scaled with it."""
        out = total
        if self.pre:
            cache = self._diag_cache
            if cache is None or cache[0] != (out.device, m):
                diag = self.diag.float().to(out.device)
                cache = ((out.device, m), 1.0 + m * (diag - 1.0))
                self._diag_cache = cache
            d_m = cache[1]
            out = (d_m.to(out.dtype) * base_out + (out - base_out).float()).to(
                out.dtype
            )
        if self.post:
            out = (out.float() + m * self.rotate_out(out)).to(out.dtype)
        return out


def bake_tensors(variant, parts, core):
    """The fixed-sigma plain-adapter tensors: the core folded into the up factors
    (the trainer's comfy export rule), alpha and dora_scale carried over."""
    out = {k: v for k, v in parts.items() if k in COMMON_NAMES}
    if variant == "lora":
        out["lora_up.weight"] = parts["lora_up.weight"].float() @ core.T
        out["lora_down.weight"] = parts["lora_down.weight"]
    elif variant == "loha":
        out["hada_w1_a"] = parts["hada_w1_a"].float() @ core.T
        out["hada_w1_b"] = parts["hada_w1_b"]
        out["hada_w2_a"] = parts["hada_w2_a"].float() @ core.T
        out["hada_w2_b"] = parts["hada_w2_b"]
    else:
        for name in ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2_b"):
            if name in parts:
                out[name] = parts[name]
        out["lokr_w2_a"] = parts["lokr_w2_a"].float() @ core.T
    return out


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


def carries_sora(tensors) -> bool:
    """The file holds SORA tensors. Such a file needs the dynamic path even without a
    hypernetwork: ComfyUI's native patches have no slot for `sora_diag` / `sora_C`, so
    a static load drops them (and warns once per layer)."""
    return any(k.rsplit(".", 1)[-1] in SORA_NAMES for k in tensors)


class NaraAdapter(WeightAdapterBase):
    """NaRA over one weight: `parts` are (offset, alpha, {name: tensor}, dora_scale)
    per adapted slice (a fused qkv weight carries three). `weights` holds every
    tensor flat so BypassForwardHook moves them to the compute device; parts index
    into it. DoRA normalizes after the live core in fp32 through the Gram expansion
    of ||W + dW(sigma)||; the static base terms (||W||^2, W down^T / W^T up and the
    factor Gram) are cached per adapter at first use and the ratios per (device,
    core), so cond and uncond share one r x r norm pass per step. LoHa / LoKr keep
    the dense per-forward norm."""

    name = "nara"

    def __init__(self, loaded_keys, parts, hyper, variant, export_sigma=0.5):
        self.loaded_keys = loaded_keys
        # None for a file with no nara_hyper.* (a plain adapter carrying SORA): Pre-Diag
        # and Post-Ortho read the base output and the rank factors, never C(sigma), so
        # the core is the identity. A NaRA file always passes its hypernetwork.
        self.hyper = hyper
        self.variant = variant
        self.export_sigma = float(export_sigma)
        self.weights = tuple(
            t
            for _, _, ts, ds, _ in parts
            for t in list(ts.values()) + ([ds] if ds is not None else [])
        )
        self.parts = []
        self.sora = []
        i = 0
        for offset, alpha, ts, ds, sora in parts:
            idx = {name: i + j for j, name in enumerate(ts)}
            i += len(ts)
            dora_idx = None if ds is None else i
            if ds is not None:
                i += 1
            self.parts.append((offset, alpha, idx, dora_idx))
            # SORA conditioning per slice, parallel to `parts`. It never enters
            # `weights`: the rotation reads factors, not the flat tensor list.
            self.sora.append(SoraConditioning(sora or {}))
        self.has_dora = any(d is not None for _, _, _, d in self.parts)
        self.wd_on_out = True
        if self.has_dora:
            ds = self.weights[next(d for _, _, _, d in self.parts if d is not None)]
            out_features = self._out_features(parts[0])
            in_features = self._in_features(parts[0])
            if ds.shape[0] == out_features:
                self.wd_on_out = True
            elif ds.numel() == in_features:
                self.wd_on_out = False
            else:
                raise ValueError(f"dora_scale shape {tuple(ds.shape)} matches neither axis ({out_features}, {in_features})")
        self._norm_cache = None
        self._norms_cache = None
        self._eye_cache = None
        # Hot-path metadata: indices into `weights` stay valid when the bypass hook
        # replaces the tuple with device/dtype copies.
        self.multiplier = 1.0
        self._lora = []
        if variant == "lora":
            for offset, alpha, idx, _ in self.parts:
                self._lora.append(
                    (offset, self._scale(alpha, self._tensors(idx)), idx["lora_up.weight"], idx["lora_down.weight"])
                )
            self.rank = self.weights[self._lora[0][3]].shape[0]
        self._conds = [c for c in self.sora if c.any]
        self._need_base = any(c.pre for c in self._conds)
        # Fused epilogue: accumulate the up projection into the layer output instead
        # of materialising the delta; off by default (rounding order changes).
        self.fused = False
        self._core_typed_cache = None
        self._module = None

    @staticmethod
    def _out_features(part):
        offset, _, ts, *_ = part
        if offset is not None:
            return offset[2]
        if "lora_up.weight" in ts:
            return ts["lora_up.weight"].shape[0]
        if "hada_w1_a" in ts:
            return ts["hada_w1_a"].shape[0]
        w1 = ts["lokr_w1"] if "lokr_w1" in ts else ts["lokr_w1_a"]
        return w1.shape[0] * ts["lokr_w2_a"].shape[0]

    @staticmethod
    def _in_features(part):
        _, _, ts, *_ = part
        if "lora_down.weight" in ts:
            return ts["lora_down.weight"].shape[1]
        if "hada_w1_b" in ts:
            return ts["hada_w1_b"].shape[1]
        w1 = ts["lokr_w1"] if "lokr_w1" in ts else ts["lokr_w1_a"]
        return w1.shape[1] * ts["lokr_w2_b"].shape[1]

    def _tensors(self, idx):
        return {name: self.weights[i] for name, i in idx.items()}

    def _scale(self, alpha, ts):
        return 1.0 if alpha is None else alpha / rank_of(self.variant, ts)

    def _core_at(self, rank, device=None):
        """The current C(sigma) for this adapter: I(rank) when it has no hypernetwork,
        which is exactly what a file without `nara_hyper.*` means (no noise axis).
        The identity is cached per (rank, device); a hypernetwork's core comes from
        its own parameters, already on the compute device."""
        if self.hyper is None:
            cache = self._eye_cache
            if cache is None or cache[0] != (rank, device):
                cache = ((rank, device), torch.eye(rank, dtype=torch.float32, device=device))
                self._eye_cache = cache
            return cache[1]
        if _sigma is None:
            raise RuntimeError("NaRA: no sampling sigma set; sample with the NaRA KSampler")
        return _core(self.hyper, _sigma)

    def _core_typed(self, core, dtype, device):
        """The core in the activation dtype, cached per sigma: the cast is bit-exact
        and cond/uncond share one core tensor, so each layer casts it once per step."""
        cache = self._core_typed_cache
        if cache is None or cache[0] is not core or cache[1] != (dtype, device):
            cache = (core, (dtype, device), core.to(device=device, dtype=dtype))
            self._core_typed_cache = cache
        return cache[2]

    def _lora_hidden(self, x, down_idx, core):
        """x @ (down C(sigma))^T of one slice, in x's dtype (the 32-wide hot GEMM)."""
        down = self.weights[down_idx]
        if down.dtype != x.dtype:
            down = down.to(x.dtype)
        hidden = F.linear(x, down)
        if self.hyper is not None:
            hidden = hidden @ self._core_typed(core, x.dtype, x.device)
        return hidden

    def _delta(self, x, ts, core, scale):
        """The slice's bypass delta in x's dtype (LoRA keeps the factorized chain)."""
        if self.variant == "lora":
            up = ts["lora_up.weight"].to(x.dtype)
            down = ts["lora_down.weight"].to(x.dtype)
            hidden = F.linear(x, down)
            # no hypernetwork means the core is the exact identity: skip the matmul
            if self.hyper is not None:
                hidden = hidden @ core.to(device=x.device, dtype=x.dtype)
            return F.linear(hidden, up) * scale
        core_x = core.to(device=x.device, dtype=x.dtype)
        dw = part_delta(self.variant, ts, core_x, scale, x.dtype)
        return F.linear(x, dw)

    def _add_delta(self, x, base_out):
        """Fused NaRA: each slice's up GEMM accumulates into the base output, one
        read/write of the activation instead of a separate delta tensor."""
        core = self._core_at(self.rank, x.device)
        for offset, scale, up_idx, down_idx in self._lora:
            up = self.weights[up_idx]
            if up.dtype != x.dtype:
                up = up.to(x.dtype)
            hidden = self._lora_hidden(x, down_idx, core).view(-1, up.shape[1])
            ys = base_out if offset is None else base_out.narrow(-1, offset[1], offset[2])
            ys.view(-1, ys.shape[-1]).addmm_(hidden, up.T, beta=1.0, alpha=scale * self.multiplier)
        return base_out

    def h(self, x, base_out):
        m = self.multiplier
        if self.variant != "lora":
            return self._h_generic(x, base_out, m)
        core = self._core_at(self.rank, x.device)
        out = None
        for offset, scale, up_idx, down_idx in self._lora:
            up = self.weights[up_idx]
            if up.dtype != x.dtype:
                up = up.to(x.dtype)
            delta = F.linear(self._lora_hidden(x, down_idx, core), up) * (scale * m)
            if offset is None:
                out = delta if out is None else out + delta
            else:
                if out is None:
                    out = torch.zeros_like(base_out)
                out.narrow(-1, offset[1], offset[2]).add_(delta)
        return out

    def _h_generic(self, x, base_out, m):
        out = None
        for offset, alpha, idx, _ in self.parts:
            ts = self._tensors(idx)
            core = self._core_at(rank_of(self.variant, ts), x.device)
            delta = self._delta(x, ts, core, self._scale(alpha, ts) * m)
            if offset is None:
                out = delta if out is None else out + delta
            else:
                if out is None:
                    out = torch.zeros_like(base_out)
                out.narrow(-1, offset[1], offset[2]).add_(delta)
        return out

    def bypass_forward(self, org_forward, x, *args, **kwargs):
        m = self.multiplier
        conds = self._conds
        if self.has_dora:
            plain = self._forward_plain(org_forward, x, *args, **kwargs)
            # A DoRA path's output is not base + delta, so fall back to the raw
            # module forward: it holds no adapter of this one, only an earlier
            # stack member.
            base = org_forward(x, *args, **kwargs) if self._need_base else None
        else:
            base_out = org_forward(x, *args, **kwargs)
            if self.fused and self.variant == "lora" and not self._need_base:
                with comfy.model_prefetch.pause_malloc_graph():
                    plain = self._add_delta(x, base_out)
                base = None
            else:
                # the adapter's own allocations pause the model's malloc graph, as
                # kitchen ops do, or every step re-records/breaks
                with comfy.model_prefetch.pause_malloc_graph():
                    delta = self.h(x, base_out)
                    plain = base_out.add_(delta)
                    # Pre-Diag scales the base output and leaves the delta at full
                    # strength; reuse the one delta instead of recomputing h.
                    base = plain - delta if self._need_base else None
        if not conds:
            return plain
        if len(conds) == 1 and self.parts[0][0] is None:
            return self.sora[0].apply(plain, plain if base is None else base, m)
        result = plain.float()
        for (offset, _, _, _), cond in zip(self.parts, self.sora):
            if not cond.any:
                continue
            sl = None if offset is None else slice(offset[1], offset[1] + offset[2])
            total = result if sl is None else result[..., sl]
            b = total if base is None else (base if sl is None else base[..., sl])
            cond_out = cond.apply(total, b, m)
            if sl is None:
                result = cond_out.float()
            else:
                result[..., sl] = cond_out.float()
        return result.to(plain.dtype)

    def _forward_plain(self, org_forward, x, *args, **kwargs):
        if self.variant == "lora":
            core = self._core_at(self.rank, x.device)
        else:
            core = self._core_at(
                rank_of(self.variant, self._tensors(self.parts[0][2])), x.device
            )
        m = self.multiplier
        # org_forward may be another bypass hook when adapters stack; the DoRA
        # norm reads the raw module's weight (earlier deltas are not folded in)
        module = self._module
        if module is None:
            module = org_forward.__self__
            while not hasattr(module, "weight") and hasattr(module, "original_forward"):
                module = module.original_forward.__self__
            self._module = module
        if self.variant == "lora":
            if self.wd_on_out:
                return self._dora_out(org_forward, x, module, core, m, args, kwargs)
            return self._dora_in(org_forward, x, module, core, m, args, kwargs)
        with comfy.model_prefetch.pause_malloc_graph():
            w, bias, stream = comfy.ops.cast_bias_weight(module, input=x, dtype=torch.float32, offloadable=True)
            try:
                if self.wd_on_out:
                    return self._dora_out_dense(org_forward, x, w, bias, core, m, args, kwargs)
                return self._dora_in_dense(org_forward, x, w, core, m, args, kwargs)
            finally:
                comfy.ops.uncast_bias_weight(module, w, bias, stream)

    def _statics(self, module, device):
        """The frozen base terms of the Gram expansion, fp32, grouped by slice shape
        so a step batches them: per group (W down^T, up, down down^T, ||W||^2, mag)
        on the output axis or (W^T up, down, up^T up, ||W||^2, mag) on the input
        axis, plus each part's span in the group-flattened ratio vector. One dequant
        at first use serves every step; rebuilt if the device moves."""
        cache = self._norm_cache
        if cache is not None and cache[0] == device:
            return cache[1], cache[2]
        w, bias, stream = comfy.ops.cast_bias_weight(module, dtype=torch.float32, device=device, offloadable=True)
        try:
            groups = {}
            for pi, (offset, _, idx, dora_idx) in enumerate(self.parts):
                ts = self._tensors(idx)
                wp = w if offset is None else w.narrow(0, offset[1], offset[2])
                mag = self.weights[dora_idx].to(device=device, dtype=torch.float32).reshape(-1)
                if self.wd_on_out:
                    up = ts["lora_up.weight"].to(device=device, dtype=torch.float32)
                    down = ts["lora_down.weight"].to(device=device, dtype=torch.float32)
                    entry = (wp @ down.T, up, down @ down.T, (wp * wp).sum(dim=1), mag)
                else:
                    up = ts["lora_up.weight"].to(device=device, dtype=torch.float32)
                    down = ts["lora_down.weight"].to(device=device, dtype=torch.float32)
                    entry = (wp.T @ up, down, up.T @ up, (wp * wp).sum(dim=0), mag)
                groups.setdefault(entry[0].shape, []).append((pi, entry))
        finally:
            comfy.ops.uncast_bias_weight(module, w, bias, stream)
        stacked, spans, pos = [], [None] * len(self.parts), 0
        for items in groups.values():
            tensors = [torch.stack([e[i] for _, e in items]) for i in range(5)]
            rows = tensors[0].shape[1]
            for j, (pi, _) in enumerate(items):
                spans[pi] = (pos + j * rows, rows)
            pos += len(items) * rows
            stacked.append(tuple(tensors))
        self._norm_cache = (device, stacked, spans)
        return stacked, spans

    def _norms(self, module, device, core):
        """mag / ||W + dW(sigma)|| per DoRA row, fp32, batched per shape group:
        ||W + dW||^2 = ||W||^2 + 2<W, dW> + ||dW||^2 with the live core. Cached per
        (device, core): cond and uncond of a step share one cached core tensor."""
        cache = self._norms_cache
        if cache is not None and cache[0] == device and cache[1] is core:
            return cache[2]
        eps = torch.finfo(torch.float32).eps
        groups, spans = self._statics(module, device)
        flat = []
        for a, b, g, base2, mag in groups:
            if self.wd_on_out:
                gram = core.T @ g @ core
                u = b @ gram
                cross = ((a @ core) * b).sum(-1)
                delta = (u * b).sum(-1)
            else:
                cross = ((core @ a.transpose(1, 2)) * b).sum(1)
                gram = core @ g @ core.T
                delta = ((b.transpose(1, 2) @ gram) * b.transpose(1, 2)).sum(-1)
            norm = (base2 + 2 * cross + delta).clamp_min(0).sqrt() + eps
            flat.append((mag / norm).reshape(-1))
        flat = torch.cat(flat)
        ratios = [flat[s : s + n] for s, n in spans]
        self._norms_cache = (device, core, ratios)
        return ratios

    def _dora_out(self, org_forward, x, module, core, m, args, kwargs):
        """y + m ((r - 1) Wx + r dy), r = mag / ||W + dW(sigma)|| per output row.
        The norm is fp32; the epilogue runs in the activation dtype with r folded
        into a row scale on y and on the up factor (the output is rounded to that
        dtype anyway). The base output is updated in place — no per-part temporary
        the size of the activation. `fused` lets the up GEMM accumulate straight
        into it through addmm instead of materialising dy."""
        y = org_forward(x, *args, **kwargs)
        with comfy.model_prefetch.pause_malloc_graph():
            for (offset, scale, up_idx, down_idx), r in zip(self._lora, self._norms(module, x.device, core)):
                a = (1 + m * (r - 1)).to(y.dtype)
                b = (m * r).to(y.dtype)
                up = self.weights[up_idx]
                up = up * b.to(up.dtype).reshape(-1, 1)
                if up.dtype != x.dtype:
                    up = up.to(x.dtype)
                hidden = self._lora_hidden(x, down_idx, core)
                ys = y if offset is None else y.narrow(-1, offset[1], offset[2])
                ys.mul_(a)
                if self.fused:
                    ys.view(-1, ys.shape[-1]).addmm_(
                        hidden.view(-1, hidden.shape[-1]), up.T, beta=1.0, alpha=scale
                    )
                else:
                    ys.add_(F.linear(hidden, up).mul_(scale))
                if module.bias is not None:
                    bs = module.bias if offset is None else module.bias.narrow(0, offset[1], offset[2])
                    ys.sub_((m * (r - 1)).to(y.dtype) * bs.to(y.dtype))
            return y

    def _dora_in(self, org_forward, x, module, core, m, args, kwargs):
        """W'x = (W + dW)(c x), c = mag / ||W + dW(sigma)|| per input column; one
        scaled-input base forward per slice (fused qkv slices scale differently)."""
        y = org_forward(x, *args, **kwargs)
        with comfy.model_prefetch.pause_malloc_graph():
            out = y.float()
            for (offset, alpha, idx, _), c in zip(self.parts, self._norms(module, x.device, core)):
                ts = self._tensors(idx)
                scale = self._scale(alpha, ts)
                xc = (x.float() * c).to(x.dtype)
                yc = org_forward(xc, *args, **kwargs)
                dy = self._delta(xc, ts, core, scale)
                if offset is None:
                    out = out + m * (yc.float() + dy.float() - y.float())
                else:
                    sl = slice(offset[1], offset[1] + offset[2])
                    out[..., sl] += m * (yc[..., sl].float() + dy.float() - y[..., sl].float())
            return out.to(y.dtype)

    def _dora_out_dense(self, org_forward, x, w, bias, core, m, args, kwargs):
        """LoHa / LoKr output axis: dense dW, one fp32 base read per forward."""
        y = org_forward(x, *args, **kwargs)
        eps = torch.finfo(torch.float32).eps
        ratios, deltas = [], []
        for offset, alpha, idx, dora_idx in self.parts:
            ts = self._tensors(idx)
            scale = self._scale(alpha, ts)
            dw = part_delta(self.variant, ts, core, scale, torch.float32)
            wp = w if offset is None else w.narrow(0, offset[1], offset[2])
            norm = (wp + dw).reshape(wp.shape[0], -1).norm(dim=1) + eps
            ratios.append(self.weights[dora_idx].float().reshape(-1).to(norm.device) / norm)
            deltas.append(self._delta(x, ts, core, scale))
        ratio = torch.cat(ratios) if len(ratios) > 1 else ratios[0]
        dy = torch.cat(deltas, dim=-1) if len(deltas) > 1 else deltas[0]
        yf = y.float()
        wx = yf if bias is None else yf - bias.float()
        out = yf + m * ((ratio - 1) * wx + ratio * dy.float())
        return out.to(y.dtype)

    def _dora_in_dense(self, org_forward, x, w, core, m, args, kwargs):
        """LoHa / LoKr input axis: dense dW, one fp32 base read per forward."""
        y = org_forward(x, *args, **kwargs)
        eps = torch.finfo(torch.float32).eps
        out = y.float()
        for offset, alpha, idx, dora_idx in self.parts:
            ts = self._tensors(idx)
            scale = self._scale(alpha, ts)
            dw = part_delta(self.variant, ts, core, scale, torch.float32)
            wp = w if offset is None else w.narrow(0, offset[1], offset[2])
            norm = (wp + dw).norm(dim=0) + eps
            c = self.weights[dora_idx].float().reshape(-1).to(norm.device) / norm
            xc = (x.float() * c).to(x.dtype)
            yc = org_forward(xc, *args, **kwargs)
            dy = self._delta(xc, ts, core, scale)
            if offset is None:
                out = out + m * (yc.float() + dy.float() - y.float())
            else:
                sl = slice(offset[1], offset[1] + offset[2])
                out[..., sl] += m * (yc[..., sl].float() + dy.float() - y[..., sl].float())
        return out.to(y.dtype)

    def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
        """Merge mode: NaRA at the fixed export_sigma as a plain adapter diff. DoRA
        merges through the Bake node (its magnitude needs the base weight)."""
        if self.has_dora:
            raise ValueError("NaRA+DoRA cannot merge directly; use NaRA Bake (fixed sigma)")
        device = weight.device
        hyper_device = next(self.hyper.parameters()).device
        s = torch.tensor([self.export_sigma], device=hyper_device, dtype=torch.float32)
        core = self.hyper(s)[0].to(device=device, dtype=intermediate_dtype)
        for off, alpha, idx, _ in self.parts:
            ts = {name: comfy.model_management.cast_to_device(t, device, intermediate_dtype) for name, t in self._tensors(idx).items()}
            diff = part_delta(self.variant, ts, core, self._scale(alpha, ts), intermediate_dtype)
            wp = weight if off is None else weight.narrow(*off)
            wp += function((strength * diff).type(wp.dtype))
        return weight
