import json
import logging
import uuid

import torch
from comfy_api.latest import ComfyExtension, io

import comfy.lora
import comfy.lora_convert
import comfy.model_base
import comfy.model_management
import comfy.ops
import comfy.samplers
import folder_paths
import nodes
from comfy.patcher_extension import PatcherInjection, WrappersMP
from comfy.weight_adapter.bypass import BypassInjectionManager

from .nara import (
    NaraAdapter,
    bake_tensors,
    carries_sora,
    group_nara_layers,
    hyper_from,
    map_nara_layers,
    normalize_ema,
    part_delta,
    rank_of,
    read_adapter,
    set_sigma,
    split_hyper,
    split_sora,
    variant_of,
)

INJECTION_PREFIX = "moss_nara_"


def _check_moss(model):
    base = model.model
    if isinstance(base, comfy.model_base.MOSSImage):
        return
    # The standalone moss-image loaders define their own MOSSImage(BaseModel) beside the
    # core one: same weights and layer names, different class object, so isinstance fails.
    diffusion = getattr(base, "diffusion_model", None)
    if type(base).__name__ == "MOSSImage" and hasattr(diffusion, "video_embedder") and hasattr(diffusion, "layers"):
        return
    raise ValueError("NaRA nodes expect a MOSS-Image model")


def _export_sigma(meta):
    spec = meta.get("kohakumoss_spec")
    if spec:
        return float(json.loads(spec).get("algo_kwargs", {}).get("export_sigma", 0.5))
    args = meta.get("ss_network_args")
    if args:
        return float(json.loads(args).get("export_sigma", 0.5))
    return 0.5


def _map_nara_layers(model, layers):
    """`map_nara_layers` against this model's key map."""
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    return map_nara_layers(key_map, layers)


def _load_nara(model, tensors, meta, strength_model):
    hyper_tensors, rest = split_hyper(tensors)
    # A file with SORA but no nara_hyper.* is a plain adapter plus Pre-Diag / Post-Ortho
    # (the trainer's NARA=False run). Those read the base output and the rank factors,
    # not C(sigma), so it still needs this dynamic path: ComfyUI's native patches have no
    # slot for sora_diag / sora_C and would drop them with a warning per layer.
    hyper = hyper_from(hyper_tensors, meta) if hyper_tensors else None
    layers = group_nara_layers(rest)
    groups, loaded_keys, variant = _map_nara_layers(model, layers)
    if not groups:
        raise ValueError("no NaRA layer matched this model")
    has_dora = any(
        ds is not None for parts in groups.values() for _, _, _, ds, _ in parts
    )
    sora_conds = sum(1 for parts in groups.values() for *_, s in parts if s)
    logging.info(
        "[moss-nara] loaded %s-%s%s%s (%d weights, dynamic core)",
        "NaRA" if hyper is not None else "SORA",
        variant,
        "+DoRA" if has_dora else "",
        "+SORA" if sora_conds else "",
        len(groups),
    )

    manager = BypassInjectionManager()
    for weight_key, parts in groups.items():
        parts.sort(key=lambda p: p[0][1] if p[0] is not None else 0)
        adapter = NaraAdapter(loaded_keys, parts, hyper, variant, _export_sigma(meta))
        manager.add_adapter(weight_key, adapter, strength=strength_model)
    inner = manager.create_injections(model.model)[0]

    def inject(patcher):
        inner.inject(patcher)
        if hyper is not None:
            hyper.to(comfy.model_management.get_torch_device())

    def eject(patcher):
        inner.eject(patcher)
        if hyper is not None:
            hyper.to("cpu")
            hyper._core_cache = None

    m = model.clone()
    m.set_injections(f"{INJECTION_PREFIX}{uuid.uuid4().hex[:8]}", [PatcherInjection(inject, eject)])
    return m


def _load_static(model, tensors, strength_model):
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    patches = comfy.lora.load_lora(comfy.lora_convert.convert_lora(tensors), key_map)
    if not patches:
        raise ValueError("no adapter layer matched this model")
    kinds = sorted({type(v).__name__ for v in patches.values() if hasattr(v, "name")})
    logging.info("[moss-nara] loaded static adapter via native patches: %s (%d weights)", ", ".join(kinds), len(patches))
    m = model.clone()
    m.add_patches(patches, strength_model)
    return m


class MossNaraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MossNara_Loader",
            display_name="Adapter Loader (MOSS-Image, NaRA auto)",
            category="moss_nara",
            description="Load a MOSS-Image adapter, detecting its type automatically: NaRA (lora/loha/lokr, with or without DoRA) runs as a dynamic bypass adapter — sample with the NaRA KSampler Advanced; plain lora/loha/lokr (with or without DoRA) loads through ComfyUI's native patches and works with any sampler.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-10.0, max=10.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model) -> io.NodeOutput:
        if strength_model == 0.0:
            return io.NodeOutput(model)
        _check_moss(model)
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        tensors, meta = read_adapter(path)
        tensors, meta = normalize_ema(tensors, meta, path)
        hyper_tensors, _ = split_hyper(tensors)
        if hyper_tensors or carries_sora(tensors):
            return io.NodeOutput(_load_nara(model, tensors, meta, strength_model))
        return io.NodeOutput(_load_static(model, tensors, strength_model))


def _has_nara(model):
    return any(k.startswith(INJECTION_PREFIX) for k in model.injections)


class MossNaraKSamplerAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MossNara_KSamplerAdvanced",
            display_name="NaRA KSampler Advanced (MOSS-Image)",
            category="moss_nara",
            description="ComfyUI's advanced KSampler with NaRA support: when the upstream model carries a NaRA adapter, each denoising step's sigma feeds the hypernetwork so the core C(sigma) follows the noise level. Without a NaRA adapter it behaves exactly like the native KSamplerAdvanced.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("add_noise", options=["enable", "disable"]),
                io.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=35, min=1, max=10000),
                io.Float.Input("cfg", default=3.5, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="sgm_uniform"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Int.Input("start_at_step", default=0, min=0, max=10000),
                io.Int.Input("end_at_step", default=10000, min=0, max=10000),
                io.Combo.Input("return_with_leftover_noise", options=["disable", "enable"]),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise) -> io.NodeOutput:
        nara = _has_nara(model)
        m = model
        if nara:
            m = model.clone()

            def sigma_wrapper(executor, x, timestep, model_options={}, seed=None):
                set_sigma(float(timestep))
                return executor(x, timestep, model_options, seed)

            m.add_wrapper_with_key(WrappersMP.PREDICT_NOISE, "moss_nara_sigma", sigma_wrapper)
        try:
            out = nodes.common_ksampler(
                m, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                denoise=1.0,
                disable_noise=add_noise == "disable",
                start_step=start_at_step,
                last_step=end_at_step,
                force_full_denoise=return_with_leftover_noise == "disable",
            )[0]
        finally:
            if nara:
                set_sigma(None)
        return io.NodeOutput(out)


class MossNaraKSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MossNara_KSampler",
            display_name="NaRA KSampler (MOSS-Image)",
            category="moss_nara",
            description="ComfyUI's standard KSampler with NaRA support: when the upstream model carries a NaRA adapter, each denoising step's sigma feeds the hypernetwork so the core C(sigma) follows the noise level. Without a NaRA adapter it behaves exactly like the native KSampler.",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise) -> io.NodeOutput:
        nara = _has_nara(model)
        m = model
        if nara:
            m = model.clone()

            def sigma_wrapper(executor, x, timestep, model_options={}, seed=None):
                set_sigma(float(timestep))
                return executor(x, timestep, model_options, seed)

            m.add_wrapper_with_key(WrappersMP.PREDICT_NOISE, "moss_nara_sigma", sigma_wrapper)
        try:
            out = nodes.common_ksampler(
                m, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                denoise=denoise,
            )[0]
        finally:
            if nara:
                set_sigma(None)
        return io.NodeOutput(out)


def _baked_dora_scale(model, weight_key, offset, dw, dora):
    """The DoRA magnitude in ComfyUI's normalization: output-axis DoRA divides by
    ||W|| where the trainer divides by ||W + dW||, so fold that ratio in; the input
    axis already normalizes by ||W + dW|| and carries over unchanged."""
    module = model.model.get_submodule(weight_key[: -len(".weight")])
    device = comfy.model_management.get_torch_device()
    w, bias, stream = comfy.ops.cast_bias_weight(module, dtype=torch.float32, device=device, offloadable=True)
    try:
        wp = w if offset is None else w.narrow(0, offset[1], offset[2])
        if dora.shape[0] != wp.shape[0]:
            return dora.float()
        eps = torch.finfo(torch.float32).eps
        row = wp.reshape(wp.shape[0], -1)
        merged = row + dw.to(row.device).reshape(row.shape)
        ratio = (row.norm(dim=1) + eps) / (merged.norm(dim=1) + eps)
        return dora.float() * ratio.to(dora.device).reshape(dora.shape)
    finally:
        comfy.ops.uncast_bias_weight(module, w, bias, stream)


class MossNaraBake(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MossNara_Bake",
            display_name="Adapter Bake (MOSS-Image, fixed sigma)",
            category="moss_nara",
            description="Bake a NaRA adapter (lora/loha/lokr, with or without DoRA) at one fixed noise level into a plain adapter: the core C(sigma) folds into the up factors, DoRA magnitudes convert to ComfyUI's normalization. Works with any sampler, but gives up the per-step sigma adaptation.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Float.Input("sigma", default=0.5, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model, sigma) -> io.NodeOutput:
        if strength_model == 0.0:
            return io.NodeOutput(model)
        _check_moss(model)
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        tensors, meta = read_adapter(path)
        tensors, meta = normalize_ema(tensors, meta, path)
        hyper_tensors, rest = split_hyper(tensors)
        if not hyper_tensors:
            raise ValueError(f"{lora_name}: not a NaRA file (no nara_hyper tensors); load it with the Adapter Loader")
        hyper = hyper_from(hyper_tensors, meta)
        core = hyper(torch.tensor([sigma], dtype=torch.float32))[0]
        layers = group_nara_layers(rest)
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        baked = {}
        matched = 0
        for stem, parts in layers.items():
            target = key_map.get(stem)
            if target is None:
                logging.warning("[moss-nara] key not found in model: %s", stem)
                continue
            sora, factors = split_sora(parts)
            if sora:
                # A SORA layer's rotation touches the base weight, which a static
                # load_lora patch set cannot express: bake it through the dynamic
                # loader (sample as usual) or export the trainer's 'comfy' format.
                logging.warning(
                    "[moss-nara] %s carries SORA; baking at a fixed sigma drops it", stem
                )
            variant = variant_of(factors)
            out = bake_tensors(variant, factors, core)
            dora = factors.get("dora_scale")
            if dora is not None:
                weight_key, offset = target if isinstance(target, tuple) else (target, None)
                alpha = float(factors["alpha"].item()) if "alpha" in factors else None
                scale = 1.0 if alpha is None else alpha / rank_of(variant, factors)
                dw = part_delta(variant, factors, core, scale, torch.float32)
                out["dora_scale"] = _baked_dora_scale(model, weight_key, offset, dw, dora)
            for name, value in out.items():
                baked[f"{stem}.{name}"] = value
            matched += 1
        if not matched:
            raise ValueError(f"{lora_name}: no NaRA layer matched this model")
        logging.info("[moss-nara] baked NaRA at sigma=%.3f (%d weights)", sigma, matched)
        patches = comfy.lora.load_lora(baked, key_map)
        m = model.clone()
        m.add_patches(patches, strength_model)
        return io.NodeOutput(m)


class MossNaraExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MossNaraLoader, MossNaraKSampler, MossNaraKSamplerAdvanced, MossNaraBake]