# Patch distortion tail analysis

Protocol: fixed test manifest, seed17-aligned images/messages, 32×32 non-overlapping patches, 50 images. Local SSIM and LPIPS are evaluation-only metrics.

## Tail statistics

| Model | Global MSE | Top25 MSE | Max MSE | Top25/global | Max/global | Patch CV | Top25 local SSIM | Worst local SSIM | Top25 local LPIPS | Worst local LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No-crop Global | 0.000734 | 0.001014 | 0.001197 | 1.374 | 1.625 | 0.280 | 0.9457 | 0.9332 | 0.00177 | 0.00284 |
| Crop-trained Global | 0.001407 | 0.001670 | 0.001832 | 1.181 | 1.287 | 0.142 | 0.8949 | 0.8785 | 0.00262 | 0.00341 |
| Patch16 top25 weight50 | 0.001168 | 0.001410 | 0.001529 | 1.190 | 1.281 | 0.142 | 0.9076 | 0.8938 | 0.00196 | 0.00274 |

## Claim check

- Crop-trained Global minus No-crop Global mean top25/global ratio: -0.1923; paired t-test p=2.258e-15.
- Patch16 candidate minus Crop-trained Global mean top25/global ratio: +0.0082; paired t-test p=0.3719.
- The concentration statistic does not support claim A. Use claim B: crop-robust watermarking degrades visual quality, while a global average objective does not explicitly control the local-distortion tail.

## Independent local perceptual evaluation

LPIPS status: enabled (AlexNet). These metrics are not used for training and therefore provide an independent check against optimizing only worst-patch MSE.

## Figures

- `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/local_loss_diagnostics_20260908/patch_mse_histogram_cdf.png`
- `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/local_loss_diagnostics_20260908/worst_global_ratio_cdf.png`
- `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/local_loss_diagnostics_20260908/residual_tail_example.png` (aligned image index 7)
