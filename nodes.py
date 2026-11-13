import logging
import uuid

import torch
from comfy_api.latest import ComfyExtension, io

import comfy.lora
import comfy.model_base
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
from comfy.patcher_extension import PatcherInjection, WrappersMP
from comfy.weight_adapter.bypass import BypassInjectionManager
from comfy.weight_adapter.lora import LoRAAdapter

from .nara import NaraAdapter, load_nara, set_sigma


def _map_layers(model, layers):
    """Resolve file layers to model weight keys; group slices of one weight (fused qkv)."""
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    groups = {}
    loaded_keys = set()
    for stem, up, down, alpha in layers:
        target = key_map.get(stem)
        if target is None:
            logging.warning("[moss-nara] key not found in model: %s", stem)
            continue
        loaded_keys.update((stem + ".lora_up.weight", stem + ".lora_down.weight", stem + ".alpha"))
        weight_key, offset = target if isinstance(target, tuple) else (target, None)
        groups.setdefault(weight_key, []).append((offset, alpha, up, down))
    return groups, loaded_keys


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


class MossNaraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MossNara_Loader",
            display_name="NaRA Loader (MOSS-Image)",
            category="moss_nara",
            description="Load a NaRA adapter (noise-aware LoRA with a shared hypernetwork) onto a MOSS-Image model. Sample with the NaRA KSampler so the core C(sigma) follows the denoising noise level.",
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
        hyper, layers = load_nara(path)
        groups, loaded_keys = _map_layers(model, layers)
        if not groups:
            raise ValueError(f"{lora_name}: no NaRA layer matched this model")

        manager = BypassInjectionManager()
        for weight_key, parts in groups.items():
            parts.sort(key=lambda p: p[0][1] if p[0] is not None else 0)
            manager.add_adapter(weight_key, NaraAdapter(loaded_keys, parts, hyper), strength=strength_model)
        inner = manager.create_injections(model.model)[0]

        def inject(patcher):
            inner.inject(patcher)
            hyper.to(comfy.model_management.get_torch_device())

        def eject(patcher):
            inner.eject(patcher)
            hyper.to("cpu")

        m = model.clone()
        m.set_injections(f"moss_nara_{uuid.uuid4().hex[:8]}", [PatcherInjection(inject, eject)])
        return io.NodeOutput(m)


def _sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial", None), latent.get("downscale_ratio_temporal", None))
    batch_inds = latent.get("batch_index", None)
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    noise_mask = latent.get("noise_mask", None)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


class MossNaraKSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MossNara_KSampler",
            display_name="NaRA KSampler (MOSS-Image)",
            category="moss_nara",
            description="KSampler that feeds each denoising step's sigma to the NaRA hypernetwork, so the adapter's core C(sigma) updates during sampling. Required for NaRA-loaded models.",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=25, min=1, max=10000),
                io.Float.Input("cfg", default=2.5, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="er_sde"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise) -> io.NodeOutput:
        m = model.clone()

        def sigma_wrapper(executor, x, timestep, model_options={}, seed=None):
            set_sigma(float(timestep))
            return executor(x, timestep, model_options, seed)

        m.add_wrapper_with_key(WrappersMP.PREDICT_NOISE, "moss_nara_sigma", sigma_wrapper)
        try:
            out = _sample(m, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
        finally:
            set_sigma(None)
        return io.NodeOutput(out)


class MossNaraBake(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MossNara_Bake",
            display_name="NaRA Bake (fixed sigma)",
            category="moss_nara",
            description="Apply NaRA as a plain LoRA at one fixed noise level (up = B C(sigma)^T). Works with any sampler, but gives up the per-step sigma adaptation.",
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
        hyper, layers = load_nara(path)
        core = hyper(torch.tensor([sigma], dtype=torch.float32))[0]
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        patches = {}
        for stem, up, down, alpha in layers:
            target = key_map.get(stem)
            if target is None:
                logging.warning("[moss-nara] key not found in model: %s", stem)
                continue
            patches[target] = LoRAAdapter(set(), (up.float() @ core.T, down, alpha, None, None, None))
        if not patches:
            raise ValueError(f"{lora_name}: no NaRA layer matched this model")
        m = model.clone()
        m.add_patches(patches, strength_model)
        return io.NodeOutput(m)


class MossNaraExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MossNaraLoader, MossNaraKSampler, MossNaraBake]