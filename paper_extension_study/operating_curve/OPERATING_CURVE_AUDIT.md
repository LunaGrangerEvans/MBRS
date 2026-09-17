# Operating-Curve Audit

This audit checks the registered operating-curve implementation and preserves the raw measured points. It does not select a project-test point, retune alpha, or claim curve dominance.

## Protocol checks

| Check | Result | Evidence |
|---|---|---|
| Alpha grid frozen before project-test | PASS | `FROZEN_ALPHA_GRID.txt` was pre-registered before the run; `PRE_REGISTERED_PLAN.md` fixes it and the curve script evaluates validation before project-test. |
| Grid selected using validation only | PASS | `curve_metadata.json` records `project_test_used_to_choose_alpha_grid: false`. |
| Alpha range | PASS | Values are 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.00; none exceed 1. |
| Per-image alpha tuning | PASS | One global alpha is applied per method/point; no per-image alpha field or tuning loop exists. |
| Identical grid for Global/Ours | PASS | 18 rows = 9 alpha values × 2 methods; summary CSV pairs both methods at every alpha. |
| Project-test ordering | PASS | The registered script writes validation results before loading/evaluating project-test data, after reading the frozen grid. |
## Measured project-test summary

`operating_curve_summary.csv` reports every project-test alpha point with the requested paired fields. Same-alpha rows are descriptive only; they are not treated as matched-PSNR comparisons when their PSNR differs materially.

## Common PSNR overlap

The measured Global PSNR range is `36.263172–39.351276 dB`; the measured Ours range is `36.854909–39.942909 dB`. Their common overlap is `36.854909–39.351276 dB`.

To avoid same-alpha pairing at materially different PSNR, this audit uses linear interpolation within the measured overlap only. It evaluates five evenly spaced common PSNR values, including the overlap endpoints. No extrapolation is used and all raw points remain in `project_test_curve.csv`.

| Common PSNR | Global local PSNR | Ours local PSNR | Global P95 | Ours P95 | Global LPIPS | Ours LPIPS | Global CIEDE | Ours CIEDE | Global BER30 | Ours BER30 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 36.854909 | 36.113913 | 36.141895 | 2.633000125e-04 | 2.619925669e-04 | 0.00204688 | 0.00191711 | 3.310267 | 3.254091 | 0.113125000 | 0.113125000 |
| 37.479001 | 36.737975 | 36.766343 | 2.283274070e-04 | 2.269621937e-04 | 0.00177164 | 0.00165880 | 3.087771 | 3.034859 | 0.113086595 | 0.113152368 |
| 38.103093 | 37.362065 | 37.390692 | 1.976759913e-04 | 1.968043320e-04 | 0.00153148 | 0.00143694 | 2.878460 | 2.830356 | 0.113062500 | 0.113189980 |
| 38.727185 | 37.986126 | 38.014912 | 1.711061319e-04 | 1.703382371e-04 | 0.00132367 | 0.00124193 | 2.682649 | 2.637967 | 0.113062500 | 0.113527252 |
| 39.351276 | 38.610058 | 38.639097 | 1.481526451e-04 | 1.474368035e-04 | 0.00114442 | 0.00107453 | 2.499856 | 2.458204 | 0.113000000 | 0.113625000 |

## Fractions across the comparable interpolated points

- Ours has better Top-25 Local PSNR at `5/5` points (`100%`).
- Ours has lower P95 at `5/5` points (`100%`).
- Ours has lower LPIPS at `5/5` points (`100%`).
- Ours has lower Global CIEDE2000 at `5/5` points (`100%`).
- BER30 is better for Ours at `0/5` points, tied at `1/5`, and worse at `4/5` points (ties use absolute tolerance `1e-9`).

## Interpretation

The local/perceptual advantage persists across the overlapping operating range under this interpolation diagnostic. BER30 is mixed and does not support BER dominance or equivalence. This audit therefore supports only a cautious range-level local/perceptual statement; it does not claim total curve dominance.
