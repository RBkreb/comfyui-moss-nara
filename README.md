# comfyui-moss-nara

面向 ComfyUI 中 **MOSS-Image** 模型的 NaRA（Noise-aware Rank Adaptation，噪声感知秩适配）推理节点。

NaRA 是一种 LoRA 变体：它的低秩核心会随去噪噪声水平变化。

```
h = W0 x + s B C(sigma) A x
C(sigma) = I + c_scale F(e_sigma)
```

其中 `C` 来自一个与 A/B 因子一同训练的小型共享超网络（见 KohakuMossTrainer 里的 NaRA 适配器）。普通的 LoRA 导出会把 `C` 冻结在单一 sigma 上；本节点在采样过程中让 `C` 保持随噪声水平实时变化。

## 节点

| 节点 | 作用 |
|------|------|
| **NaRA Loader (MOSS-Image)** | 以 bypass 适配器的方式，把 NaRA safetensors 文件加载到 MOSS-Image 模型上。基础权重（bf16 或 int8 量化）不会被修改。 |
| **NaRA KSampler (MOSS-Image)** | 每一步都把该步的 sigma 喂给超网络，使 `C(sigma)` 跟随采样调度变化。采样 NaRA 加载后的模型时必须使用此节点。 |
| **NaRA Bake (fixed sigma)** | 在单一噪声水平下把 NaRA 当作普通 LoRA 应用（`up = B C(sigma)^T`）。可搭配任意采样器，但放弃了逐 step 的噪声自适应。 |

用法：`UNETLoader -> NaRA Loader -> NaRA KSampler -> VAEDecode`。其余部分（文本编码、SLG、shift、VAE）与标准 MOSS-Image 工作流保持一致。

加载器既可用 ComfyUI 核心的 `UNETLoader`，也可用独立 `moss-image` 节点包提供的 `MOSSImageUNETLoader`，两者均受支持。

## 文件格式

支持 KohakuMossTrainer 的 NaRA checkpoint，以及 LyCORIS 风格的导出——前提是文件带有 `nara_hyper.*` 张量并记录了配置（`kohakumoss_spec`，或 `ss_network_args` / `ss_network_dim` 元数据）。逐层键名使用 `lora_unet_layers_N_proj_*` 约定；fused q/k/v 的 checkpoint 也受支持。

固定 sigma 的 ComfyUI 导出（不含 `nara_hyper.*`）就是一个普通 LoRA，请改用标准的 LoraLoader 加载。

## 环境要求

- **ComfyUI ≥ v0.42.0**：需要核心的 `comfy.model_base.MOSSImage`（MOSS-Image 支持在该版本引入）。
- **torch** 与 **safetensors** 都随 ComfyUI 一并安装，本节点包不引入任何额外的第三方依赖，因此**不需要** `requirements.txt`。请勿手动 `pip install torch`，否则可能覆盖掉与当前 CUDA 版本匹配的 torch 构建。
- 使用 ComfyUI 的 V3 节点 API（`comfy_api.latest`），该 API 内置于 ComfyUI，无需单独安装。

## 说明

- 超网络在每个不同的 sigma 上只求值一次，并由所有层共享；在 bypass 模式下，每层的额外开销只是一段秩大小的矩阵乘链。
- `strength_model` 对适配器输出的缩放方式与 LoRA 的 strength 完全一致。
- sigma 约定与训练一致：flow-matching 的噪声水平，范围 [0, 1]（1 = 纯噪声，0 = 干净图像）。
