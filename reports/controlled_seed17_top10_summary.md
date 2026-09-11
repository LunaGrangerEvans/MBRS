# Controlled seed17 continuation summary

Primary comparison is against the equal-length Global continuation control. All branches use the same seed17 epoch-100 source checkpoint, restored Adam/BatchNorm state, 20 continuation epochs, LR 1e-4, one GPU, and fixed evaluation masks.

## Quality and robustness

| Branch | PSNR | Δ | Worst/top25 PSNR | Δ | SSIM | Δ | LPIPS | Δ | BER30 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Global continuation | 36.227 | +0.000 | 35.316 | +0.000 | 0.9437 | +0.0000 | 0.00234 | +0.00000 | 0.11313 | +0.00000 |
| Hard P16-T25-L50 | 36.383 | +0.156 | 35.519 | +0.203 | 0.9454 | +0.0017 | 0.00219 | -0.00015 | 0.11319 | +0.00006 |
| Soft P16-T0.5-L50 | 36.305 | +0.077 | 35.420 | +0.104 | 0.9445 | +0.0008 | 0.00223 | -0.00012 | 0.11300 | -0.00013 |
| Hard patch16 stride8 top25 global0.5 local0.5 | 36.411 | +0.183 | 35.549 | +0.233 | 0.9457 | +0.0020 | 0.00220 | -0.00015 | 0.11319 | +0.00006 |
| Multiscale patch16 top25 weight0.7 plus patch32 top25 weight0.3 global0.5 local0.5 | 36.356 | +0.129 | 35.491 | +0.175 | 0.9451 | +0.0014 | 0.00219 | -0.00015 | 0.11306 | -0.00006 |
| Excess patch16 stride16 threshold1 scale3 global0.5 local0.5 | 35.864 | -0.363 | 35.030 | -0.285 | 0.9392 | -0.0045 | 0.00234 | -0.00000 | 0.11294 | -0.00019 |
| Hard patch16 stride8 top10 global0.5 local0.5 | 36.407 | +0.180 | 35.563 | +0.247 | 0.9454 | +0.0017 | 0.00221 | -0.00014 | 0.11294 | -0.00019 |

## Local tail

| Branch | Top10 MSE | Top25 MSE | P90 | P95 | P99 | Mean max MSE | Tail/global | Local SSIM top25 | Local LPIPS top25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Global continuation | 0.001247 | 0.001176 | 0.001353 | 0.001613 | 0.002113 | 0.001309 | 1.224 | 0.9244 | 0.00113 |
| Hard P16-T25-L50 | 0.001187 | 0.001122 | 0.001314 | 0.001535 | 0.001980 | 0.001245 | 1.209 | 0.9265 | 0.00107 |
| Soft P16-T0.5-L50 | 0.001216 | 0.001148 | 0.001324 | 0.001573 | 0.002039 | 0.001274 | 1.216 | 0.9254 | 0.00107 |
| Hard patch16 stride8 top25 global0.5 local0.5 | 0.001179 | 0.001115 | 0.001309 | 0.001523 | 0.001965 | 0.001236 | 1.208 | 0.9270 | 0.00107 |
| Multiscale patch16 top25 weight0.7 plus patch32 top25 weight0.3 global0.5 local0.5 | 0.001195 | 0.001130 | 0.001320 | 0.001544 | 0.001984 | 0.001253 | 1.210 | 0.9262 | 0.00106 |
| Excess patch16 stride16 threshold1 scale3 global0.5 local0.5 | 0.001325 | 0.001256 | 0.001439 | 0.001693 | 0.002152 | 0.001387 | 1.203 | 0.9184 | 0.00108 |
| Hard patch16 stride8 top10 global0.5 local0.5 | 0.001173 | 0.001111 | 0.001301 | 0.001522 | 0.001949 | 0.001230 | 1.204 | 0.9265 | 0.00112 |

## BER curve

| Branch | BER30 | BER35 | BER40 | BER50 | BER70 | BER100 |
|---|---:|---:|---:|---:|---:|---:|
| Global continuation | 0.11313 | 0.08300 | 0.04163 | 0.00144 | 0.00000 | 0.00000 |
| Hard P16-T25-L50 | 0.11319 | 0.08331 | 0.04169 | 0.00156 | 0.00000 | 0.00000 |
| Soft P16-T0.5-L50 | 0.11300 | 0.08300 | 0.04125 | 0.00150 | 0.00000 | 0.00000 |
| Hard patch16 stride8 top25 global0.5 local0.5 | 0.11319 | 0.08313 | 0.04156 | 0.00156 | 0.00000 | 0.00000 |
| Multiscale patch16 top25 weight0.7 plus patch32 top25 weight0.3 global0.5 local0.5 | 0.11306 | 0.08313 | 0.04169 | 0.00144 | 0.00000 | 0.00000 |
| Excess patch16 stride16 threshold1 scale3 global0.5 local0.5 | 0.11294 | 0.08306 | 0.04144 | 0.00144 | 0.00000 | 0.00000 |
| Hard patch16 stride8 top10 global0.5 local0.5 | 0.11294 | 0.08319 | 0.04181 | 0.00150 | 0.00000 | 0.00000 |

## Training behavior

| Branch | Final message raw | Global raw | Local raw | Weighted image | Total | PSNR step std | BER step std | Soft max weight | Soft N_eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Global continuation | 0.013794 | 0.001134 | 0.000000 | 0.001134 | 0.139079 | 0.2657 | 0.02092 | 0.00000 | 0.00 |
| Hard P16-T25-L50 | 0.013806 | 0.001084 | 0.001454 | 0.001269 | 0.139330 | 0.2050 | 0.02077 | 0.00000 | 0.00 |
| Soft P16-T0.5-L50 | 0.013786 | 0.001108 | 0.001379 | 0.001244 | 0.139103 | 0.2252 | 0.02081 | 0.07150 | 46.05 |
| Hard patch16 stride8 top25 global0.5 local0.5 | 0.013805 | 0.001079 | 0.001456 | 0.001267 | 0.139318 | 0.1784 | 0.02079 | 0.00000 | 0.00 |
| Multiscale patch16 top25 weight0.7 plus patch32 top25 weight0.3 global0.5 local0.5 | 0.013799 | 0.001096 | 0.001448 | 0.001272 | 0.139259 | 0.2460 | 0.02075 | 0.00000 | 0.00 |
| Excess patch16 stride16 threshold1 scale3 global0.5 local0.5 | 0.013777 | 0.001201 | 0.000164 | 0.000682 | 0.138456 | 0.2193 | 0.02081 | 0.00000 | 0.00 |
| Hard patch16 stride8 top10 global0.5 local0.5 | 0.013807 | 0.001074 | 0.001612 | 0.001343 | 0.139414 | 0.1614 | 0.02079 | 0.00000 | 0.00 |

## Soft T=0.5 go/no-go

- worst/top25 PSNR improves: PASS
- top25 and P95 MSE decrease: PASS
- global PSNR within -0.2 dB: PASS
- BER30 within +0.005: PASS
- local SSIM or LPIPS improves: PASS
- Soft oscillation no worse than Hard: PASS

Decision: **GO** for temperature screening.

## Figures

All residual panels use a shared scale. Aligned example index: 7.

- `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/controlled_seed17_top10/aligned_watermark_residual_worst_patch.png`
- `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/controlled_seed17_top10/patch_mse_histogram_cdf.png`
- `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/controlled_seed17_top10/patch_mse_percentile_curve.png`
- `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/controlled_seed17_top10/tail_global_ratio_cdf.png`
