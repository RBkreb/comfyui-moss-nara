# comfyui-moss-nara

NaRA (Noise-aware Rank Adaptation) inference nodes for **MOSS-Image** in ComfyUI.

NaRA is a LoRA variant whose low-rank core depends on the denoising noise level:

```
h = W0 x + s B C(sigma) A x
C(sigma) = I + c_scale F(e_sigma)
```

`C` comes from one small shared hypernetwork trained alongside the A/B factors
(see the NaRA adapter in KohakuMossTrainer). A plain LoRA export freezes `C` at a
single sigma; these nodes keep it live during sampling.

## Nodes

| Node | What it does |
|------|--------------|
| **NaRA Loader (MOSS-Image)** | Loads a NaRA safetensors file onto a MOSS-Image model as a bypass adapter. Base weights (bf16 or int8 quantized) are never modified. |
| **NaRA KSampler (MOSS-Image)** | KSampler that feeds every step's sigma to the hypernetwork, so `C(sigma)` follows the schedule. Required when sampling a NaRA-loaded model. |
| **NaRA Bake (fixed sigma)** | Applies NaRA as a plain LoRA at one fixed noise level (`up = B C(sigma)^T`). Works with any sampler; gives up per-step adaptation. |

Usage: `UNETLoader -> NaRA Loader -> NaRA KSampler -> VAEDecode`. Everything else
(text encode, SLG, shift, VAE) stays as in the standard MOSS-Image workflow.

## File formats

Accepted: KohakuMossTrainer NaRA checkpoints and LyCORIS-style exports that carry
the `nara_hyper.*` tensors plus a recorded configuration (`kohakumoss_spec` or
`ss_network_args` / `ss_network_dim` metadata). Per-layer keys use the
`lora_unet_layers_N_proj_*` convention; fused q/k/v checkpoints are supported.

A fixed-sigma ComfyUI export (no `nara_hyper.*`) is an ordinary LoRA — load it
with the standard LoraLoader instead.

## Notes

- The hypernetwork is evaluated once per distinct sigma and shared by all layers;
  the per-layer overhead is a rank-sized matmul chain in bypass mode.
- `strength_model` scales the adapter output exactly like a LoRA strength.
- Sigma convention matches training: flow-matching noise level in [0, 1]
  (1 = pure noise, 0 = clean image).