# Seed17 主实验配置目录

本目录采用完整配置描述，不使用单独的 `global`、`worst`、`hard` 等简称作为配置名。

## 统一主实验协议

除非表格特别说明，所有主实验均满足：

- seed：`17`
- source checkpoint：seed17 的 crop-trained message MSE + global image MSE 模型 epoch 100
- 输入：`128×128 RGB`，`64-bit message`
- 训练噪声：`RandomCrop(0.3, 1.0)`
- continuation：20 epochs，`lr=1e-4`
- batch size：16，单 GPU，deterministic
- optimizer：恢复 source checkpoint 的 Adam state
- message loss：`MSE(decoded_message, message)`，weight `10.0`
- evaluation：固定 manifest，报告 PSNR、SSIM、LPIPS、Worst/Top25/P90/P95/P99 局部失真和 BER@30/35/40/50/70/100

`Local-specific gain = ΔWorstPSNR − ΔPSNR`，只作为内部诊断，不作为正式论文指标。

## 当前主实验与 Soft ablation

| 完整配置名称 | 局部目标 | 关键参数 | PSNR | ΔPSNR | Worst PSNR | ΔWorstPSNR | Top25 MSE | ΔTop25 MSE | BER@30 | 状态 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `seed17_continuation_message_mse10_global_image_mse1_local_none` | 无局部项 | global=1.0，local=0 | 36.227 | 0 | 35.316 | 0 | 0.001176 | 0 | 0.11313 | controlled reference |
| `seed17_hard_patch16_stride16_top25_global_weight0.5_local_weight0.5` | non-overlap patch MSE 的 top25% 均值 | patch=16，stride=16，selected=16/64 | 36.383 | +0.156 | 35.519 | +0.203 | 0.001122 | -0.000054 | 0.11319 | current hard reference |
| `seed17_soft_patch16_stride16_normalized_detached_softmax_temperature0.25_global_weight0.5_local_weight0.5` | normalized patch MSE 的 detached softmax 加权均值 | patch=16，stride=16，T=0.25，N_eff≈28 | 36.314 | +0.086 | 35.451 | +0.136 | 0.001140 | -0.000036 | 0.11338 | selected soft ablation |
| `seed17_soft_patch16_stride16_normalized_detached_softmax_temperature0.5_global_weight0.5_local_weight0.5` | normalized patch MSE 的 detached softmax 加权均值 | patch=16，stride=16，T=0.5，N_eff≈46 | 36.305 | +0.077 | 35.420 | +0.104 | 0.001148 | -0.000028 | 0.11300 | completed soft ablation |
| `seed17_soft_patch16_stride16_normalized_detached_softmax_temperature1.0_global_weight0.5_local_weight0.5` | normalized patch MSE 的 detached softmax 加权均值 | patch=16，stride=16，T=1.0，N_eff≈58 | 36.250 | +0.023 | 35.353 | +0.037 | 0.001166 | -0.000010 | 0.11319 | completed soft ablation |
| `seed17_hard_patch16_stride8_top25_global_weight0.5_local_weight0.5` | overlapping patch MSE 的 top25% 均值 | patch=16，stride=8，225 patches，selected=57 | 36.411 | +0.183 | 35.549 | +0.233 | 0.001115 | -0.000062 | 0.11319 | completed P0 |
| `seed17_hard_patch16_stride8_top10_global_weight0.5_local_weight0.5` | overlapping patch MSE 的 top10% 均值 | patch=16，stride=8，225 patches，selected=23 | 36.407 | +0.180 | 35.563 | +0.247 | 0.001111 | -0.000065 | 0.11294 | completed P0; best current |
| `seed17_multiscale_patch16_stride16_top25_weight0.7_plus_patch32_stride32_top25_weight0.3_global_weight0.5_local_weight0.5` | `0.7 × Patch16 Top25 + 0.3 × Patch32 Top25` | two non-overlap scales，global=0.5，local=0.5 | 36.356 | +0.129 | 35.491 | +0.175 | 0.001130 | -0.000047 | 0.11306 | completed P1 |
| `seed17_excess_patch16_stride16_threshold1_scale3_global_weight0.5_local_weight0.5` | `mu_ref × mean(ReLU(e_i / mu_ref − 1)^2) × 3` | patch=16，stride=16，threshold=1.0，fixed scale=3.0 | 35.864 | -0.363 | 35.030 | -0.285 | 0.001256 | +0.000080 | 0.11294 | completed P1；reject |

以上结果全部来自同一个 seed17 source checkpoint，不与历史 full-scratch 或 DataParallel 结果混合。Soft temperature 不再继续扩展搜索；T=0.25 仅作为已有 ablation 保留。

## 下一阶段配置名称

下一阶段仍然只使用 seed17、同一 source checkpoint、同一 continuation protocol。每个新实验单独改变一个因素。

| 完整配置名称 | 局部目标 | 关键参数 | 目的 |
|---|---|---|---|
| `seed17_hard_patch16_stride8_top25_global_weight0.5_local_weight0.5` | overlapping patch MSE 的 top25% 均值 | patch=16，stride=8，225 patches，selected=57 | 已完成：ΔWorstPSNR=+0.233 dB，作为当前 P0 结果 |
| `seed17_multiscale_patch16_stride16_top25_weight0.7_plus_patch32_stride32_top25_weight0.3_global_weight0.5_local_weight0.5` | `0.7 × L_patch16_top25 + 0.3 × L_patch32_top25` | patch16/patch32 均 non-overlap | 已完成：ΔWorstPSNR=+0.175 dB，未超过现有 Hard 或 overlap |
| `seed17_excess_patch16_stride16_threshold1_scale3_global_weight0.5_local_weight0.5` | `mu_ref × mean(ReLU(e_i / (stop_gradient(mu)+eps) − 1)^2) × 3` | patch=16，stride=16，无 top-k ranking，固定 scale=3.0 | 已完成但未改善 tail：ΔWorstPSNR=-0.285 dB，不进入候选 |

### 新实验的额外检查

- overlapping patch 必须记录实际 patch 数、实际 selected 数、每像素覆盖次数和 local loss normalization。
- Excess Distortion Loss 训练前必须记录 raw magnitude 以及 watermarked image / encoder gradient norm，并校准到 Hard Patch16 reference 的量级。
- 不同时组合 warm-up；warm-up 暂列 P2。
- 不运行 seed29/41，不继续 soft temperature grid，不重新搜索 Patch8、Patch64 或大规模 top-k ratio。

## 历史 exploratory 配置的完整解释

下列结果只用于理解趋势，不属于当前 seed17 主实验结论：

| 历史结果中的旧标签 | 完整含义 |
|---|---|
| `crop_global_baseline` | crop-trained encoder-decoder；message MSE weight=10；global image MSE weight=1；no local loss |
| `patch_mean` | non-overlap patch32 MSE 的全图有效像素加权平均；global image MSE weight=0；local image MSE weight=1；该目标与 global MSE 数学等价 |
| `worst_patch32_top25_weight50` | non-overlap patch32；每图 16 个 patch 中选误差最大的 4 个；global weight=0.5；local weight=0.5 |
| `worst_patch16_top25_weight50` | non-overlap patch16；每图 64 个 patch 中选误差最大的 16 个；global weight=0.5；local weight=0.5 |
| `worst_patch16_top25_weight25` | non-overlap patch16；top25%；global weight=0.75；local weight=0.25 |
| `worst_patch16_top25_weight75` | non-overlap patch16；top25%；global weight=0.25；local weight=0.75 |
| `worst_patch32_top25_weight25` | non-overlap patch32；top25%；global weight=0.75；local weight=0.25 |
| `worst_patch32_top50_weight50` | non-overlap patch32；每图选误差最大的 8 个 patch；global weight=0.5；local weight=0.5 |
| `worst_patch32_top10_weight50` | non-overlap patch32；每图选误差最大的 2 个 patch；global weight=0.5；local weight=0.5 |
| `worst_patch64_top25_weight50` | non-overlap patch64；每图 4 个 patch 中选 1 个；global weight=0.5；local weight=0.5 |
| `worst_patch8_top10/top25/top50` | non-overlap patch8；128×128 图像对应 256 个 patch；分别选择 top10%、top25% 或 top50% |
| `nocrop_global` | identity noise；message MSE weight=10；global image MSE weight=1；没有 RandomCrop，不是 crop-robust 主实验 |

历史 full-scratch 数值保留在 [matched_seed_summary.md](matched_seed_summary.md)，但后续汇报优先使用本目录中的完整配置名称和 seed17 controlled 结果。

## seed17 新方法最终比较

| 完整配置 | ΔPSNR | ΔWorstPSNR | Local-specific gain | ΔTop25 MSE | ΔP95 MSE | ΔBER30 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| `seed17_hard_patch16_stride16_top25_global_weight0.5_local_weight0.5` | +0.156 | +0.203 | +0.047 | -0.000054 | -0.000078 | +0.00006 | hard reference |
| `seed17_hard_patch16_stride8_top25_global_weight0.5_local_weight0.5` | +0.183 | +0.233 | +0.050 | -0.000062 | -0.000090 | +0.00006 | **best current** |
| `seed17_multiscale_patch16_stride16_top25_weight0.7_plus_patch32_stride32_top25_weight0.3_global_weight0.5_local_weight0.5` | +0.129 | +0.175 | +0.046 | -0.000047 | -0.000069 | -0.00006 | below overlap |
| `seed17_excess_patch16_stride16_threshold1_scale3_global_weight0.5_local_weight0.5` | -0.363 | -0.285 | +0.078 | +0.000080 | +0.000079 | -0.00019 | reject |

Excess 的 Local-specific gain 数值表面上为正，是因为全局和 worst 指标同步下降；不能将其解释为局部 tail 改善。它的 Top25/P95 MSE 以及 local SSIM 均恶化，说明固定 batch 上的 gradient-scale 校准没有解决训练过程中 active excess 区域和 loss magnitude 的变化问题。

## Patch16 local weight 0.25 的结果状态

完整配置含义为：seed17、source 为 crop-trained message MSE + global image MSE epoch100 checkpoint、continuation 20 epochs、patch16、stride16、top25%、global image-loss weight=0.75、local image-loss weight=0.25、message-loss weight=10、RandomCrop(0.3,1.0)。

该配置目前**没有正式 controlled fixed-manifest 主实验结果**。现有 seed17 记录属于历史 full-scratch exploratory validation：epoch100 的 validation PSNR 为 `33.033 dB`，validation Worst PSNR 为 `31.752 dB`，validation BER 为 `0`。这些 validation 数值不能与上方 controlled fixed-manifest 指标直接比较，也不能作为当前主方法结论。
