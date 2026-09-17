# PSNR Estimator Audit

## Finding

The apparent `+0.596953233 dB` versus `+0.591737 dB` discrepancy is an estimator mismatch, not a clipping, normalization, image-size, or pairing mismatch. The original bootstrap computes a mean of per-image PSNR differences. The frozen paper computes PSNR from the aggregate mean MSE.

## Frozen paper formula and code path

The frozen paper path is the `summarize` function in `experiments/freeze_paper_evidence.py`. After the per-image evaluator produces `global_mse` records, `summarize` sets:

```text
MSE_paper = mean_i(global_mse_i)
PSNR_paper = -10 log10(MSE_paper)
             = 10 log10(1 / mean_i(global_mse_i))
```

The source code explicitly assigns `result['global_psnr'] = float(-10 * np.log10(np.mean([r['global_mse'] for r in rows])))`. The per-image records are produced by `evaluate` in `experiments/evaluate_extended_image_quality.py`, where each image is first converted from normalized `[-1,1]` to clipped RGB `[0,1]`, then `global_mse` is the mean squared RGB error over that image. All images have the same 128×128×3 dimensions, so equal image weighting is identical to pixel-weighted aggregate MSE here.

## Original bootstrap formula and code path

The original path is `load_rows`, `main`, and `ci_record` in `paper_extension_study/bootstrap/bootstrap_analysis.py`. It loads the `global_psnr` field from `reports/final_ours_project_test_per_image.csv`, forms per-image paired differences:

```text
d_i = PSNR_ours,i - PSNR_global,i
```

and then computes each bootstrap replicate as:

```text
d_boot = mean_{i sampled with replacement}(d_i)
```

Thus its observed value is `mean_i(PSNR_ours,i − PSNR_global,i)`, equivalently the difference of mean per-image PSNR values. It does not recompute PSNR from resampled mean MSE.

## Exact source of the numerical mismatch

PSNR is nonlinear in MSE:

```text
mean_i[10 log10(1 / MSE_i)] != 10 log10(1 / mean_i[MSE_i])
```

The original bootstrap therefore estimates a different quantity. The other metrics in the original bootstrap are already resampled at the image level using their per-image values and are not changed by this correction.

No metric-space preprocessing mismatch was found:

- both paths use clipped RGB `[0,1]` derived from normalized tensors;
- both use the same 50 paired project-test images;
- both use 128×128 RGB images with equal dimensions;
- both preserve Global/Ours image pairing;
- the mismatch is not caused by pixel weighting, image dimensions, or tensor normalization.

## Corrected result

Because the estimands differ, the original PSNR interval is not compatible with the frozen paper PSNR estimator. The corrected result is saved separately, without deleting or overwriting the original bootstrap files:

- `bootstrap_psnr_paper_estimator.csv`
- `bootstrap_psnr_paper_estimator.npz`
- `bootstrap_psnr_paper_estimator_summary.md`
- `bootstrap_psnr_paper_estimator_metadata.json`

The corrected analysis uses 20,000 paired image-index resamples and analysis seed `20260916`. For each replicate it recomputes the paper estimator from the resampled mean MSE for Global and Ours, then differences the two PSNR values. The corrected summary file contains the exact observed difference, percentile interval, standard error, and zero-crossing status.

The corrected PSNR bootstrap is compatible with the formal paper estimator. The original bootstrap outputs for all non-PSNR metrics remain authoritative extension artifacts and are unchanged.
