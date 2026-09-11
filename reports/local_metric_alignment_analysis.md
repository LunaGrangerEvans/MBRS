# Local metric alignment analysis

This is an analysis-only comparison at the formal 32×32 patch scale on the fixed seed17 manifest. For each image, patch MSE rank was compared with patch LPIPS rank and local SSIM degradation rank.

| Method | Mean Spearman MSE↔LPIPS | Mean Spearman MSE↔SSIM degradation | Top25 Jaccard MSE/LPIPS | Top25 Jaccard MSE/SSIM degradation |
|---|---:|---:|---:|---:|
| Global continuation | -0.092 | -0.008 | 0.149 | 0.159 |
| Hard Patch16 stride16 | -0.100 | -0.045 | 0.151 | 0.157 |
| Hard Patch16 stride8 | -0.099 | -0.042 | 0.151 | 0.152 |
| Excess | -0.170 | -0.041 | 0.119 | 0.136 |

The correlations are weak and mostly negative. MSE-worst patches are therefore not reliably LPIPS-worst or SSIM-worst patches in this setting. The low Top25 Jaccard overlap reinforces the mismatch.

This does not justify immediately adding LPIPS to training: LPIPS is not valid on the current 16×16 patch size because its pooling hierarchy collapses the spatial support. It does justify treating MSE-tail optimization as a proxy objective whose perceptual validity must be demonstrated, not assumed.

## Decision

The local metric is now a stronger bottleneck candidate than the backbone. The next research decision should be whether the paper studies MSE-defined distortion tail explicitly, or whether it should redefine the tail using a larger perceptual/structural region before adding a training term.
