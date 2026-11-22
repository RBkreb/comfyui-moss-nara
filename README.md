# comfyui-moss-nara

**MOSS-Image** 适配器推理节点：NaRA + SORA

loha / lokr / lora × DoRA × SORA 组合的自动识别加载。


## 安装
```bash
cd Comfyui/custom_nodes
git clone https://github.com/RBkreb/comfyui-moss-nara.git
```
重启comfyui
## 节点

| 节点 | 作用 |
|------|------|
| **Adapter Loader (MOSS-Image, NaRA auto)** | 自动识别适配器类型：含 `nara_hyper.*` **或** SORA 张量的文件以 bypass 方式加载；其余普通 lora/loha/lokr 走 ComfyUI 原生 patch |
| **NaRA KSampler (MOSS-Image)** | 原生标准 KSampler 的等价物。上游模型含 NaRA 时以每步 sigma 计算超网络；不含 NaRA 时行为与原生 KSampler 一致 |
| **NaRA KSampler Advanced (MOSS-Image)** | 原生高级 KSampler 的等价物。上游模型含 NaRA 时以每步 sigma 计算超网络；不含 NaRA 时行为与原生 KSamplerAdvanced 一致 |
| **Adapter Bake (MOSS-Image, fixed sigma)** |  固定 NaRA sigma 制成普通适配器：核心折入 up 因子，DoRA 幅度换算为 ComfyUI 的归一化语义。**SORA 无法兼容**，会逐层告警并丢弃 SORA 分量 |

用法：`Moss Loader -> Adapter Loader -> NaRA KSampler`。

## 环境
- **ComfyUI ≥ v0.42.0**

## 说明

- `strength_model` 与 LoRA strength 一致；DoRA 的 multiplier 插值与 trainer 相同。
- **SORA 参数**（`sora_C` / `sora_D` / `sora_diag` / `sora_s_p0`）与 factor 张量分开解析。文件带 SORA 但无 `nara_hyper.*`走动态路径，此时 C(σ)=I：Pre-Diag / Post-Ortho 只读 base 输出与 rank 因子。


## 自检脚本

| 脚本 | 检查 |
|------|------|
| `scripts/check_plugin_sora.py` | SORA 公式对 fp64 参考实现（Post-Ortho / Pre-Diag / 组合 / m=0 / s_p0 上界 / 键分离） |
| `scripts/check_plugin_file.py [file]` | 真实导出文件的解析：超网络重建、变体识别、SORA 键无泄漏 |
| `scripts/check_plugin_vs_trainer.py` | 插件 bypass 路径对 **trainer 前向**逐层比对（单层相对误差 ~1e-7） |
| `scripts/check_plugin_lora_prediag.py` | 纯 LoRA + Pre-Diag（无超网络，C(σ)=I）对 trainer 前向比对，并验证 Pre-Diag 未被静默丢弃 |
| `scripts/check_loader_end_to_end.py [file]` |  Loader 的路由 + 一次 bypass 前向；确认 `sora_diag` 全部保留 |
| `scripts/check_loader_device.py` | **需要 GPU**：底座位于 cuda、SORA 张量位于 CPU 进行单次 bypass 前向 |

## 示例
[DoRA+NaRA](https://huggingface.co/RBXLSP/TextureLux-DoRAN)

[MossTrainer](https://github.com/RBkreb/KohakuMossTrainer)