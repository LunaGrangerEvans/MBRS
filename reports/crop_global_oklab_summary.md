# Crop Hard Top10 + global OKLab result

QUALITY AND COLOR IMPROVE; THE COMPLETE PRESERVATION GATE IS NOT MET

This run starts from the same seed17 crop-trained Global epoch100 source as the incumbent, restores Adam/BN state, and continues20 epochs at LR1e-4, batch16, RandomCrop(0.3,1.0).

Loss: `10 message MSE + 0.5 global RGB MSE + 0.5 Hard patch16/stride8/Top10 MSE + 0.029574882 global OKLab distance`.

The added term is a mean of per-pixel standard OKLab Euclidean distances. It does not replace the global RGB or local pixel-tail term. Chroma-only and signed biases are diagnostics. CIEDE2000 is evaluation-only.

## Fixed validation metrics

| Method | PSNR ↑ | SSIM ↑ | 3-scale MS-SSIM ↑ | Full LPIPS ↓ | Top25 local PSNR ↑ | P95 native32 MSE ↓ | Gini ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global continuation | 36.222526 | 0.952949 | 0.984118 | 0.002233172 | 35.425925 | 0.000310843 | 0.100938 |
| Hard16 stride8 Top10 (incumbent) | 36.410613 | 0.954982 | 0.984839 | 0.002140650 | 35.652988 | 0.000294555 | 0.094569 |
| Hard16 stride8 Top10 + global OKLab | 36.650197 | 0.957396 | 0.985650 | 0.001940944 | 35.879871 | 0.000280816 | 0.097303 |

| Method | CIEDE2000 global ↓ | 5×5 patch Top10 ↓ | Patch P95 ↓ | Max patch ↓ | Global OKLab ↓ | Chroma-only ↓ | Signed CIELAB da/db |
|---|---:|---:|---:|---:|---:|---:|---|
| Global continuation | 3.654601 | 5.314376 | 5.207602 | 7.275265 | 0.0134476 | 0.0111339 | -0.365335 / +0.421903 |
| Hard16 stride8 Top10 (incumbent) | 3.598625 | 5.230329 | 5.124218 | 7.198253 | 0.0132469 | 0.0109530 | -0.405426 / +0.448987 |
| Hard16 stride8 Top10 + global OKLab | 3.465417 | 5.051264 | 4.946862 | 6.990734 | 0.0126752 | 0.0105642 | -0.343891 / +0.423177 |

| Method | BER100 | BER70 | BER50 | BER40 | BER30 |
|---|---:|---:|---:|---:|---:|
| Global continuation | 0.0000000 | 0.0000000 | 0.0158750 | 0.0435000 | 0.1110000 |
| Hard16 stride8 Top10 (incumbent) | 0.0000000 | 0.0000000 | 0.0158750 | 0.0435625 | 0.1110000 |
| Hard16 stride8 Top10 + global OKLab | 0.0000000 | 0.0000000 | 0.0158750 | 0.0437500 | 0.1108125 |

## Compared with incumbent Hard Top10

- global_psnr: +0.239584907
- top25_local_psnr: +0.226883537
- global_ssim: +0.002414118
- patch_mse_p95: -0.000013739
- gini: +0.002734541
- ciede2000_global: -0.133208737 (-3.70%)
- ciede2000_top10: -0.179065247 (-3.42%)
- ciede2000_p95: -0.177356164 (-3.46%)
- ber30: -0.000187500

Per-image fractions with lower values (all50 validation images):

- ciede2000_global: 100%.
- ciede2000_top10: 100%.
- ciede2000_p95: 100%.
- patch_mse_p95: 100%.
- gini: 14%.
- full_lpips: 92%.

## Validation and debugging record

- `validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5`: preservation=False, color=False; improved-color images=100%.
- `validation_seed17_crop_hard16_stride8_top10_global_oklab_g25`: preservation=False, color=True; improved-color images=100%.

| Validation configuration | lambda | PSNR | Top25 local PSNR | Gini | Global CIEDE2000 | Top10 CIEDE2000 | BER30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global continuation | 0 | 36.222526 | 35.425925 | 0.100938 | 3.654601 | 5.314376 | 0.1110000 |
| Hard16 stride8 Top10 (incumbent) | 0 | 36.410613 | 35.652988 | 0.094569 | 3.598625 | 5.230329 | 0.1110000 |
| Hard16 stride8 Top10 + global OKLab | 0.029574882 | 36.650197 | 35.879871 | 0.097303 | 3.465417 | 5.051264 | 0.1108125 |
| Hard16 stride8 Top10 + global OKLab | 0.059149764 | 36.818150 | 36.032252 | 0.100017 | 3.365012 | 4.920614 | 0.1108125 |

These are validation-only endpoint results. No candidate passed the full validation gate, so no candidate formal-test evaluation was opened and the incumbent paper evidence remains unchanged.

## Interpretation

The candidate does not meet every preservation/color-benefit tolerance. Retain the incumbent paper method and report the measured trade-off. No extra test-driven parameter scans are performed.

The acceptance tolerances are engineering criteria, not p-values or a human perceptual study. Lower CIEDE2000 does not prove visible improvement for every observer. The 5×5 color grid differs from native32 MSE/concentration/perceptual evaluation.

This study adds an image regularizer; it has no matched-strength extra-RGB-regularizer control. Gains cannot all be attributed exclusively to the geometry of OKLab. In training logs, `global_oklab`/`global_chroma` are the actual new diagnostics; the inherited `local_oklab` field in topk mode is a legacy alias of raw local MSE and must not be interpreted as a color metric.

## Why this is a concentration trade-off

Gini describes relative inequality, not absolute error. The candidate reduces per-image P95 pixel error and color error on all50 validation images, while its Gini can increase. Compared with Global continuation, even the half-weight candidate retains a lower mean Gini (0.097303 versus0.100938), but less of the reduction achieved by incumbent Hard Top10 (0.094569). Thus the original concentration benefit is partly reduced, not evidence that absolute errors increased.

A reference-content-only diagnostic groups native32 patches by original brightness/gradient quartiles. At the larger weight, dark/bright group mean MSE decreases9.46%/8.52% and low/high-gradient group MSE decreases9.27%/8.63%. At half weight the respective reductions are5.65%/5.08% and5.58%/5.12%. All grouped mean errors decline; the differences are modest and do not establish artifact relocation as a sole cause. Unequal regional reductions are consistent with the changed normalized distribution. See [region diagnostics](crop_global_oklab_region_diagnostics.csv).

The legacy JPEG+OKLab implementation used an incorrect XYZ matrix on linear RGB. The new module follows the [author’s linear-sRGB implementation](https://bottosson.github.io/posts/oklab/) and has independent primary-color/neutral/gradient tests. Old JPEG outcomes are not evidence against standard OKLab.

## Files and metric contract

- [protocol](crop_global_oklab_protocol.md)
- [main CSV](crop_global_oklab_main_table.csv)
- [per-image CSV](crop_global_oklab_per_image.csv)
- Complete validation/calibration/provenance: `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab`.
- Fixed validation qualitative examples and color distribution: `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/crop_global_oklab`.
- Quality: clipped RGB[0,1], current frozen evaluator; PSNR from dataset mean MSE, localPSNR mean per-image dB; CIEDE2000 from CIELAB D65, color5×5 stride1, Top10 ceil1538/15376. BER: the existing normalized float output, five fixed masks per ratio, no resize-back. No metric implementation tuned on these results.
