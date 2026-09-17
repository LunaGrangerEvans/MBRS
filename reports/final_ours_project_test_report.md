# Formal project-test result for the frozen final-method candidate

## Classification: **FINAL METHOD ACCEPTED**

This decision uses one evaluation of the frozen g25 configuration on the existing fixed 50-image project-test manifest. No project-test result was used to retune or select a checkpoint.

- Manifest: `/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_manifest.pt`; SHA-256 `36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339`.
- Raw outputs and original evaluator CSV: `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/test_seed17_crop_hard16_stride8_top10_global_oklab_g25`.
- Expanded CSV with P90/P95/P99 and per-image records: `/root/workspace/GLX/icassp/MBRS/reports/final_ours_project_test_per_image.csv`.

## Required comparison

| Method | PSNR | SSIM | LPIPS | Top-25 local PSNR | P95 MSE | P99 MSE | Global CIEDE2000 | Top10 CIEDE2000 | Gini | BER30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MBRS crop-trained global | 36.263172 | 0.950759 | 0.00234960 | 35.522213 | 3.016531971e-04 | 3.199585360e-04 | 3.535857 | 5.159104 | 0.095007 | 0.1131250 |
| Hard Local-Tail | 36.442300 | 0.952541 | 0.00221518 | 35.754376 | 2.844551472e-04 | 3.010726967e-04 | 3.484790 | 5.075666 | 0.086897 | 0.1129375 |
| Ours | 36.854909 | 0.957018 | 0.00191711 | 36.141895 | 2.619925590e-04 | 2.782322409e-04 | 3.254091 | 4.772229 | 0.092130 | 0.1131250 |

## Full metric summary

P90/P95/P99 are means of the per-image percentiles over the 16 native 32×32 patches, matching the frozen validation evaluator's primary aggregation. The expanded CSV also preserves pooled percentile values as `pooled_patch_mse_p90/p95/p99`. `population CV` is the population standard deviation divided by the pooled mean over the same 800 patch values. Gini, Top10/Mean, and Top10 energy share remain mean per-image normalized diagnostics.

| Method | P90 MSE | P95 MSE | P99 MSE | Population CV | Gini | CV (mean/image) | Top10/Mean | Top10 energy share | CIEDE2000 P95 | BER100 | BER70 | BER50 | BER40 | BER30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MBRS crop-trained global | 2.868657853e-04 | 3.016531971e-04 | 3.199585360e-04 | 0.341332 | 0.095007 | 0.173718 | 1.295702 | 0.161963 | 5.056442 | 0.0000000 | 0.0000000 | 0.0014375 | 0.0416250 | 0.1131250 |
| Hard Local-Tail | 2.713995232e-04 | 2.844551472e-04 | 3.010726967e-04 | 0.320981 | 0.086897 | 0.159015 | 1.270921 | 0.158865 | 4.974994 | 0.0000000 | 0.0000000 | 0.0015000 | 0.0418125 | 0.1129375 |
| Ours | 2.495647391e-04 | 2.619925590e-04 | 2.782322409e-04 | 0.347564 | 0.092130 | 0.168775 | 1.287692 | 0.160961 | 4.672281 | 0.0000000 | 0.0000000 | 0.0015000 | 0.0416875 | 0.1131250 |

## Frozen acceptance checks

The pre-frozen rule requires Local PSNR ≥ Hard Local-Tail, P95 MSE ≤ Hard Local-Tail, global PSNR ≥ Hard Local-Tail or no meaningful degradation, LPIPS ≤ Hard Local-Tail or no meaningful degradation, and absolute BER30 degradation ≤ 0.002. CIEDE2000 is expected but not a hard criterion; Gini is diagnostic only and is not a gate.

| Criterion | Ours value | Hard Local-Tail value | Status |
|---|---:|---:|:---:|
| Local PSNR ≥ Hard Local-Tail | 36.141895 | 35.754376 | **PASS** |
| P95 MSE ≤ Hard Local-Tail | 2.619925590e-04 | 2.844551472e-04 | **PASS** |
| Global PSNR ≥ Hard Local-Tail | 36.854909 | 36.442300 | **PASS** |
| LPIPS ≤ Hard Local-Tail | 0.00191711 | 0.00221518 | **PASS** |
| |Δ BER30| ≤ 0.002 | 0.0001875 | 0.0020000 | **PASS** |

## Hard Local-Tail → Ours deltas

- Δ PSNR: `+0.412608788 dB`.
- Δ Local PSNR: `+0.387518925 dB`.
- Relative P95 reduction: `+7.897%`.
- Relative P99 reduction: `+7.586%`.
- Relative LPIPS reduction: `+13.456%`.
- Relative global CIEDE2000 reduction: `+6.620%`.
- Relative Top10 CIEDE2000 reduction: `+5.978%`.
- Δ Gini: `+0.005233313` (+6.022%). Gini is reported as a diagnostic, not used for acceptance.
- Δ BER30: `+0.000187500`.

## Interpretation

Absolute local-tail magnitude and normalized residual concentration are distinct. The final decision is therefore based on absolute Local PSNR/P95/P99 and perceptual fidelity plus BER preservation; a Gini increase is not hidden, but it does not overturn an otherwise passing final-method rule.

The formal classification is **FINAL METHOD ACCEPTED**. The paper method is now Ours = Hard Local-Tail + OKLab; Hard Local-Tail is retained as the intermediate ablation.
