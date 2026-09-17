# Corrected PSNR Bootstrap Using the Frozen Paper Estimator

## Estimator

The frozen paper computes one PSNR per method from the aggregate clipped-RGB MSE:

```text
MSE_paper = mean_i(global_mse_i)
PSNR_paper = 10 log10(1 / MSE_paper)
```

For each of 20,000 paired image-index bootstrap replicates, this analysis resamples the 50 Global/Ours image pairs, computes each method's mean `global_mse`, applies the formula above, and then takes `Ours PSNR − Global PSNR`. It does not modify the original bootstrap outputs for any other metric.

- Input per-image CSV: `reports/final_ours_project_test_per_image.csv`
- Resamples: `20,000`
- Analysis seed: `20260916`
- Observed Global PSNR: `36.263172371770 dB`
- Observed Ours PSNR: `36.854908880639 dB`
- Observed difference: `0.591736508869 dB`
- 95% percentile interval: `[0.552852364558, 0.635779193394] dB`
- Bootstrap standard error: `0.021363447558 dB`
- Interval crosses zero: `no`

## Exact reason for the prior mismatch

The original `bootstrap_analysis.py` loads the `global_psnr` field from the per-image CSV and computes `mean_i(PSNR_ours,i − PSNR_global,i)`. That is the mean of per-image PSNR differences. The frozen paper path in `experiments/freeze_paper_evidence.py`, function `summarize`, first averages the per-image `global_mse` values and then overwrites `global_psnr` with `-10*log10(mean_i(global_mse_i))`. These estimators are nonlinear and therefore differ numerically.

Both paths use the same clipped RGB range derived from normalized tensors, the same 128×128×3 image dimensions, and equal image weights. Because every image has the same dimensions, the paper's mean per-image MSE is also the pixel-weighted aggregate MSE over the 50 images. The mismatch is not caused by clipping, normalization, image size, or unequal pixel counts.

## Compatibility

The original PSNR bootstrap interval is valid for the distinct estimand “mean per-image PSNR difference,” but it is not the exact frozen-paper PSNR estimand. The corrected files in this directory are the compatible PSNR bootstrap result. All original bootstrap intervals for Top-25 Local PSNR, P95, P99, LPIPS, CIEDE2000, Gini, and BER remain unchanged.
