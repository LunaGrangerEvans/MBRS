# Perceptual-tail visualization summary

> 本报告于本轮证据审计修正相关类型和paired网格混用问题。后续引用以 [逐图证据审计](perceptual_evidence_audit.md) 的CSV为准。旧定性panel存在独立热图缩放，不能称为统一尺度；直接看图改用 [实际RGB观察页](artifact_observation_guide.md)。

This report uses the same fixed formal test manifest and existing controlled checkpoints. No model was trained and no selector/loss was changed. The primary comparison is Global continuation versus Hard Patch16/stride8/Top10. Hard Top25 and Gradient-aware Top10 alpha2 are auxiliary references.

## Final distributions

| Method | Top10 LPIPS mean | LPIPS TailGap | Bottom10 SSIM mean | SSIM TailGap |
|---|---:|---:|---:|---:|
| Global continuation | 0.001356 | 0.000846 | 0.91760 | 0.034947 |
| Hard stride8 Top25 | 0.001293 | 0.000816 | 0.92063 | 0.033598 |
| Hard stride8 Top10 | 0.001356 | 0.000870 | 0.92007 | 0.033969 |
| Gradient-aware Top10 alpha2 | 0.001205 | 0.000739 | 0.92013 | 0.033450 |

Lower is better for LPIPS and both tail gaps; higher is better for Bottom10 SSIM. The LPIPS distributions visibly overlap substantially; the plots do not hide that overlap.

## Trajectory interpretation

The PSNR/perceptual scatter plots use E1/E5/E10/E15/E20 points with epoch labels and only light chronological lines:

- `psnr_vs_top10_lpips.png`
- `psnr_vs_lpips_tail_gap.png`
- `psnr_vs_bottom10_ssim.png`
- `psnr_vs_ssim_tail_gap.png`

They are under `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/perceptual_tail_visualization/`.

## Concentration versus perceptual salience

The final-image scatter plots are:

- `gini_vs_lpips_tail_gap.png`
- `top10mean_vs_top10_lpips.png`
- `energyshare_vs_lpips_tail_gap.png`
- `cv_vs_ssim_tail_gap.png`

Across methods, concentration–perception correlations are weak. For Hard Top10, representative 32×32 correlations are approximately:

- Gini vs LPIPS TailGap: Spearman `-0.221513`
- Top10/Mean vs Top10 LPIPS: Spearman `-0.094454`
- Top10 energy share vs LPIPS TailGap: Spearman `-0.213349`
- CV vs SSIM TailGap: Spearman `-0.088788`

These do not support a strong claim that residual concentration is a reliable perceptual-salience surrogate.

## Paired Global versus Hard Top10

Per-image paired plots:

- `paired_gini_global_vs_top10.png`
- `paired_top10mean_global_vs_top10.png`
- `paired_energyshare_global_vs_top10.png`
- `paired_lpips_global_vs_top10.png`

For the final fixed test images:

- Hard Top10 has lower Gini on `98%` of images at 32×32; median paired Gini change is `-0.007719`. The earlier `100%` and `-0.00974` values belong to the separate 16×16 grid.
- Hard Top10 has lower Top10 LPIPS on `64%` of images; median paired change is approximately `-0.000041`, while the mean change is near zero.

Thus residual concentration improves consistently, but perceptual-tail improvement is heterogeneous and small.

## Qualitative evidence

The fixed predetermined images are test indices 7, 20, and 42. No image was selected by visual quality or method outcome. The legacy panel is:

`qualitative_residual_perceptual_tail_v2.png`

It includes Original, Global/Hard watermarked images, residuals, patch-energy heatmaps, worst LPIPS locations, and worst SSIM locations. The legacy heatmaps were independently scaled, so do not use their color intensity to compare methods. For direct visual comparison use artifact_observation_guide.md. The legacy directory is `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/perceptual_tail_visualization/`.

## Required conclusions

### DOES HARD TOP10 SHIFT LPIPS TAIL DISTRIBUTION?

**WEAK.** The final Top10 LPIPS mean is effectively unchanged from Global and the distribution overlap is large. Per-image improvement occurs in 64% of images, but the aggregate mean is nearly flat.

### DOES HARD TOP10 IMPROVE SSIM TAIL DISTRIBUTION?

**YES, weakly.** Bottom10 SSIM increases from `0.91760` to `0.92007` and SSIM TailGap decreases from `0.034947` to `0.033969`.

### AT COMPARABLE GLOBAL PSNR, IS LPIPS TAIL BETTER?

**MIXED.** At the closest saved early PSNR pair, LPIPS tail is slightly worse for Hard Top10; at the final checkpoint, the aggregate LPIPS is nearly unchanged but per-image improvement is 64%.

### AT COMPARABLE GLOBAL PSNR, IS SSIM TAIL BETTER?

**MIXED to weakly positive.** The matched early pair shows a small SSIM-tail improvement, but not a large effect.

### IS GINI CORRELATED WITH LPIPS TAIL?

**WEAK, negative.** Hard Top10 Gini vs LPIPS TailGap Spearman is `-0.221513`.

### IS TOP10/MEAN CORRELATED WITH TOP10 LPIPS?

**WEAK.** Hard Top10 Spearman is `-0.094454`.

### PERCENT OF TEST IMAGES WITH LOWER GINI UNDER HARD TOP10

`98%` at the formal 32×32 grid; `100%` at the supplementary 16×16 grid.

### PERCENT OF TEST IMAGES WITH LOWER TOP10 LPIPS UNDER HARD TOP10

`64%`.

### WHAT DO THE NEW FIGURES SUPPORT?

They support a consistent reduction in pixel residual-energy concentration and a weak/heterogeneous improvement in SSIM tail. They show that Hard Top10 is not merely a global-MSE reduction: normalized concentration metrics improve beyond the global energy change.

### WHAT DO THEY NOT SUPPORT?

They do not support a strong claim that lower residual concentration reliably means lower perceptual artifact severity. LPIPS tail distributions overlap heavily and concentration–perception correlations are weak.

## Paper-level recommendation

Keep Hard Top10 as the pixel-tail main method. Phrase the perceptual claim conservatively:

> Hard Top10 consistently reduces the measured residual-energy concentration and provides weak SSIM-tail support, while LPIPS-tail improvement remains mixed and image-dependent.

Do not present residual concentration as a validated perceptual proxy without additional human or task-specific artifact annotation.
