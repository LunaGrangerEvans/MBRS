# Current controlled seed17 master table

This is the only formal comparison table for the current project. All rows use the same seed17 epoch-100 crop-trained source checkpoint, 20 continuation epochs, restored Adam state, single GPU, deterministic data order, and the fixed evaluation manifest. No seed29/41 or historical DataParallel result is included.

| Complete configuration | ΔPSNR | ΔWorstPSNR | Local-specific gain | ΔTop25 MSE | ΔP95 MSE | Δlocal SSIM | Δlocal LPIPS | ΔBER30 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `seed17_hard_patch16_stride16_top25_global_weight0.5_local_weight0.5` | +0.156 | +0.203 | +0.047 | -0.000054 | -0.000078 | +0.00219 | -0.000065 | +0.000063 | reference |
| **`seed17_hard_patch16_stride8_top25_global_weight0.5_local_weight0.5`** | **+0.183** | **+0.233** | **+0.050** | **-0.000062** | **-0.000090** | **+0.00263** | **-0.000065** | +0.000063 | **best current** |
| **`seed17_hard_patch16_stride8_top10_global_weight0.5_local_weight0.5`** | **+0.180** | **+0.247** | **+0.067** | **-0.000065** | **-0.000091** | **+0.00217** | **-0.000016** | -0.000188 | **best current** |
| `seed17_gradientaware_patch16_stride8_top10_alpha2_global_weight0.5_local_weight0.5` | +0.070 | +0.117 | +0.047 | -0.000031 | -0.000049 | +0.00216 | -0.000127 | +0.000312 | perceptual tradeoff; reject as main |
| `seed17_soft_patch16_stride16_normalized_detached_softmax_temperature0.25_global_weight0.5_local_weight0.5` | +0.086 | +0.136 | +0.049 | -0.000036 | -0.000062 | +0.00088 | -0.000054 | +0.000250 | ablation |
| `seed17_soft_patch16_stride16_normalized_detached_softmax_temperature0.5_global_weight0.5_local_weight0.5` | +0.077 | +0.104 | +0.027 | -0.000028 | -0.000040 | +0.00101 | -0.000064 | -0.000125 | ablation |
| `seed17_soft_patch16_stride16_normalized_detached_softmax_temperature1.0_global_weight0.5_local_weight0.5` | +0.023 | +0.037 | +0.014 | -0.000010 | -0.000003 | +0.00036 | -0.000073 | +0.000063 | ablation |
| `seed17_multiscale_patch16_stride16_top25_weight0.7_plus_patch32_stride32_top25_weight0.3_global_weight0.5_local_weight0.5` | +0.129 | +0.175 | +0.046 | -0.000047 | -0.000069 | +0.00181 | -0.000073 | -0.000063 | below overlap |
| `seed17_excess_patch16_stride16_threshold1_scale3_global_weight0.5_local_weight0.5` | -0.363 | -0.285 | +0.078* | +0.000080 | +0.000080 | -0.00595 | approximately 0 | -0.000190 | reject |

`Local-specific gain = ΔWorstPSNR − ΔPSNR`. The Excess value marked `*` is not a positive tail result: both global and worst quality deteriorated, while Top25/P95 MSE and local SSIM also worsened.
