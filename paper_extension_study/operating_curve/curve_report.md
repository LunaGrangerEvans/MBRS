# Fidelity–Robustness Operating Curve

This is a pre-registered residual-strength analysis of the frozen MBRS crop-trained Global and frozen Ours checkpoints. No model was retrained. The alpha grid was frozen before project-test evaluation and is identical for validation and project-test.

- Residual scaling: `x_alpha = x + alpha (x_hat - x)` in the normalized tensor, followed by clipped-RGB quality evaluation.
- Frozen alpha grid: `0.700, 0.750, 0.800, 0.850, 0.900, 0.925, 0.950, 0.975, 1.000`.
- Validation is used for calibration/context; project-test is evaluated only after the grid is fixed.
- Curves use identical axes for Global and Ours within each figure.
- No dominance claim is made by this report; the curves are descriptive across the registered range.

## Project-test curve values

| Method | α | PSNR | BER30 | P95 MSE | Top-25 Local PSNR | LPIPS | Global CIEDE2000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MBRS crop-trained global | 0.700 | 39.351276 | 0.1130000 | 1.481526451e-04 | 38.610058 | 0.00114442 | 2.499856 |
| MBRS crop-trained global | 0.750 | 38.753648 | 0.1130625 | 1.700010267e-04 | 38.012589 | 0.00131503 | 2.674419 |
| MBRS crop-trained global | 0.800 | 38.194694 | 0.1130625 | 1.933426716e-04 | 37.453667 | 0.00149755 | 2.848257 |
| MBRS crop-trained global | 0.850 | 37.669764 | 0.1130625 | 2.181753060e-04 | 36.928730 | 0.00169199 | 3.021343 |
| MBRS crop-trained global | 0.900 | 37.174943 | 0.1131250 | 2.445089394e-04 | 36.433929 | 0.00189860 | 3.193652 |
| MBRS crop-trained global | 0.925 | 36.937791 | 0.1131250 | 2.582363200e-04 | 36.196793 | 0.00200711 | 3.279507 |
| MBRS crop-trained global | 0.950 | 36.706996 | 0.1131250 | 2.723368848e-04 | 35.966003 | 0.00211786 | 3.365162 |
| MBRS crop-trained global | 0.975 | 36.482224 | 0.1131250 | 2.868102324e-04 | 35.741245 | 0.00223185 | 3.450613 |
| MBRS crop-trained global | 1.000 | 36.263172 | 0.1131250 | 3.016532065e-04 | 35.522213 | 0.00234960 | 3.535857 |
| Ours | 0.700 | 39.942909 | 0.1136250 | 1.286472140e-04 | 39.230755 | 0.00093658 | 2.299035 |
| Ours | 0.750 | 39.345297 | 0.1136250 | 1.476266815e-04 | 38.633118 | 0.00107592 | 2.459812 |
| Ours | 0.800 | 38.786391 | 0.1135625 | 1.679044280e-04 | 38.074127 | 0.00122397 | 2.619973 |
| Ours | 0.850 | 38.261482 | 0.1132500 | 1.894817648e-04 | 37.549141 | 0.00138320 | 2.779500 |
| Ours | 0.900 | 37.766676 | 0.1130625 | 2.123573376e-04 | 37.054150 | 0.00155107 | 2.938373 |
| Ours | 0.925 | 37.529533 | 0.1131250 | 2.242806445e-04 | 36.816902 | 0.00163900 | 3.017559 |
| Ours | 0.950 | 37.298739 | 0.1132500 | 2.365280726e-04 | 36.585983 | 0.00172944 | 3.096575 |
| Ours | 0.975 | 37.073965 | 0.1131250 | 2.490985486e-04 | 36.361080 | 0.00182154 | 3.175419 |
| Ours | 1.000 | 36.854909 | 0.1131250 | 2.619925669e-04 | 36.141895 | 0.00191711 | 3.254091 |

## Interpretation boundary

The operating curve tests behavior over a fixed residual-strength range. It does not retune the frozen method, select a project-test point, or replace the natural frozen or internal matched-PSNR paper results. Figure 2 remains the separate external 512×512 display-space comparison.
