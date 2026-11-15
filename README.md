# comfyui-moss-nara

**MOSS-Image** 适配器推理节点：NaRA（Noise-aware Rank Adaptation，噪声感知秩适配）

loha / lokr / lora × DoRA 组合的自动识别加载。

NaRA 的低秩核心随去噪噪声水平变化：

```
h = W0 x + s B C(sigma) A x
C(sigma) = I + c_scale F(e_sigma)
```

`C` 来自与 A/B 因子一同训练的小型共享超网络。
本节点在采样过程中让 `C` 跟随噪声水平实时变化。LoHa / LoKr 变体把核心放在各自的
秩通道里。DoRA 的归一化发生在 `BA*C(sigma)` 之后。

## 节点

| 节点 | 作用 |
|------|------|
| **Adapter Loader (MOSS-Image, NaRA auto)** | 自动识别适配器类型：含 `nara_hyper.*` 的文件以 bypass 方式加载；普通 lora/loha/lokr走 ComfyUI 原生 patch |
| **NaRA KSampler Advanced (MOSS-Image)** | 原生高级 KSampler 的等价物。上游模型含 NaRA 时以每步 sigma 计算超网络；不含 NaRA 时行为与原生 KSamplerAdvanced 完全一致 |
| **Adapter Bake (MOSS-Image, fixed sigma)** | 将 NaRA 在固定 sigma 烘焙成普通适配器：核心折入 up 因子，DoRA 幅度换算为 ComfyUI 的归一化语义 |

用法：`Moss Loader -> Adapter Loader -> NaRA KSampler Advanced`。

## 环境
- **ComfyUI ≥ v0.42.0**

## 说明

- 超网络在每个不同的 sigma 上只求值一次，所有层共享；bypass 模式下每层额外开销是
  一段秩大小的矩阵乘链。DoRA 范数用 Gram 展开（fp32）：静态基项（||W||²、W·downᵀ 等）
  在适配器首次使用时缓存，每步只剩按形状分批的小矩阵乘；输出轴 epilogue 以激活 dtype 运行。
- `strength_model` 与 LoRA strength 语义一致；DoRA 的 multiplier 插值与 trainer 相同。
- 同一层堆叠多个 NaRA 时，DoRA 范数读原始基模权重。

## 参考
- https://github.com/generaldi/NaRA