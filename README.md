# comfyui-moss-nara

 **MOSS-Image**  NaRA（Noise-aware Rank Adaptation，噪声感知秩适配）推理节点。

NaRA 是一种 LoRA 变体：它的低秩核心会随去噪噪声水平变化。

```
h = W0 x + s B C(sigma) A x
C(sigma) = I + c_scale F(e_sigma)
```

其中 `C` 来自一个与 A/B 因子一同训练的小型共享超网络。普通的 LoRA 导出会把 `C` 冻结在单一 sigma 上；本节点在采样过程中让 `C` 保持随噪声水平实时变化。

## 节点

| 节点 | 作用 |
|------|------|
| **NaRA Loader (MOSS-Image)** | 以 bypass 适配器的方式，把 NaRA 加载到 MOSS-Image 模型上 |
| **NaRA KSampler (MOSS-Image)** | 使 `C` 跟随采样调度变化。NaRA 加载后的模型时必须使用此节点 |
| **NaRA Bake (fixed sigma)** | 在单一噪声水平下把 NaRA 当作普通 LoRA 应用（`up = B C(sigma)^T`） |

用法：`Moss Loader -> NaRA Loader -> NaRA KSampler`。

## 环境
- **ComfyUI ≥ v0.42.0**
## 说明

- 超网络在每个不同的 sigma 上只求值一次，并由所有层共享；在 bypass 模式下，每层的额外开销是一段秩大小的矩阵乘链。
- `strength_model` 对适配器输出的缩放方式与 LoRA 的 strength 完全一致。

## 参考
- https://github.com/generaldi/NaRA