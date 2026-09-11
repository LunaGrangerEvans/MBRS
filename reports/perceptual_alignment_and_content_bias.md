# Perceptual alignment and content-bias analysis

Analysis uses the fixed seed17 manifest. LPIPS and SSIM rank comparisons use 32×32 patches because the LPIPS AlexNet feature hierarchy is not valid on 16×16 patches. MSE/content correlations are also computed at 16×16.

## MSE versus original-image content

For the current best overlap checkpoint at 32×32:

- MSE vs local variance: Spearman `0.269`
- MSE vs gradient magnitude: `0.402`
- MSE vs edge density: `0.294`
- MSE vs high-frequency energy: `0.398`

At 16×16 the corresponding values are approximately `0.317`, `0.332`, `0.232`, and `0.301`.

This is evidence of moderate content bias: high-MSE regions are more likely to be textured or edge-rich. It is not proof that the loss is entirely selecting texture; the correlations are not near one.

## MSE versus perceptual/structural metrics

| Method | Patch size | MSE↔LPIPS Spearman | MSE↔SSIM-degradation Spearman | MSE/LPIPS Top25 Jaccard | MSE/SSIM Top25 Jaccard |
|---|---:|---:|---:|---:|---:|
| Global continuation | 32 | -0.002 | +0.008 | 0.149 | 0.159 |
| Hard Patch16 stride8 | 32 | -0.018 | -0.056 | 0.151 | 0.152 |

The near-zero rank correlations and low Jaccard overlaps mean MSE-worst regions are not reliably LPIPS-worst or SSIM-worst regions. This is a stronger limitation than simple random noise: the ranking targets are materially different.

## Interpretation

The current loss is content-sensitive and only weakly perceptually aligned. Therefore the current paper should explicitly call the target **pixel-MSE local distortion tail**, not human-perceived artifact severity.

No perceptual training loss should be added yet. First decide whether the research contribution is about a measurable MSE tail or about perceptually aligned local artifacts.

CSV data: [perceptual_alignment_and_content_bias.csv](perceptual_alignment_and_content_bias.csv).

Plots are stored under `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/perceptual_alignment/`.
