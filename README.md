# comfyui-moss-nara

**MOSS-Image** 适配器推理节点：NaRA（Noise-aware Rank Adaptation，噪声感知秩适配）与
loha / lokr / lora × DoRA 组合的自动识别加载。

NaRA 的低秩核心随去噪噪声水平变化：

```
h = W0 x + s B C(sigma) A x
C(sigma) = I + c_scale F(e_sigma)
```

`C` 来自与 A/B 因子一同训练的小型共享超网络。普通导出把 `C` 冻结在单一 sigma；
本节点在采样过程中让 `C` 跟随噪声水平实时变化。LoHa / LoKr 变体把核心放在各自的
秩通道里（与 KohakuMossTrainer 一致）。DoRA 的归一化发生在活的 `BA*C(sigma)` 之后：
每步、fp32 重新计算 `m / ||W + dW(sigma)||`，不使用任何静态 BA 的范数缓存。

## 节点

| 节点 | 作用 |
|------|------|
| **Adapter Loader (MOSS-Image, NaRA auto)** | 自动识别适配器类型：含 `nara_hyper.*` 的文件按 NaRA（lora/loha/lokr，可带 DoRA）以 bypass 方式加载；普通 lora/loha/lokr（可带 DoRA）走 ComfyUI 原生 patch，任意采样器可用 |
| **NaRA KSampler Advanced (MOSS-Image)** | 原生高级 KSampler 的等价物（add_noise / start_at_step / end_at_step / leftover noise）。上游模型含 NaRA 时把每步 sigma 喂给超网络；不含 NaRA 时行为与原生 KSamplerAdvanced 完全一致 |
| **Adapter Bake (MOSS-Image, fixed sigma)** | 把 NaRA（lora/loha/lokr，可带 DoRA）在固定 sigma 烘焙成普通适配器：核心折入 up 因子，DoRA 幅度换算为 ComfyUI 的归一化语义。任意采样器可用，但放弃逐步 sigma 自适应 |

用法：`Moss Loader -> Adapter Loader -> NaRA KSampler Advanced`。
量化（int8 ConvRot 等）基模全程不被修改：NaRA 走 bypass，DoRA 范数经
`cast_bias_weight` 瞬时反量化读取，fp32 计算。

## 环境
- **ComfyUI ≥ v0.42.0**

## 说明

- 超网络在每个不同的 sigma 上只求值一次，所有层共享；bypass 模式下每层额外开销是
  一段秩大小的矩阵乘链（DoRA 层另有一次瞬时 fp32 范数）。
- `strength_model` 与 LoRA strength 语义一致；DoRA 的 multiplier 插值与 trainer 相同。
- 同一层堆叠多个 NaRA 时，DoRA 范数读原始基模权重（不折入先前适配器的 delta）。

## 参考
- https://github.com/generaldi/NaRA