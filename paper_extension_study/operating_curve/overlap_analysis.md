# Operating-Curve Overlap Diagnostic

This is a descriptive interpolation diagnostic over the registered project-test operating curves. It does not add alpha points, select a test operating point, or establish formal dominance.

The measured PSNR ranges overlap from `36.854909` to `39.351276 dB`. Across five evenly spaced PSNR locations in that overlap, linear interpolation of the already measured curves gives Ours lower P95 MSE, higher Top-25 Local PSNR, lower LPIPS, and lower Global CIEDE2000 than the Global curve. BER30 is mixed and is slightly higher for Ours at most interior interpolated locations; the measured BER30 ranges are `0.11299999999999999–0.11312500000000002` for Global and `0.11306249999999998–0.113625` for Ours.

| Common PSNR | Global BER30 | Ours BER30 | Global P95 | Ours P95 | Global local PSNR | Ours local PSNR | Global LPIPS | Ours LPIPS | Global CIEDE | Ours CIEDE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 36.854909 | 0.113125000 | 0.113125000 | 2.63300012e-04 | 2.61992567e-04 | 36.113913 | 36.141895 | 0.00204688 | 0.00191711 | 3.310267 | 3.254091 |
| 37.479001 | 0.113086595 | 0.113152368 | 2.28327407e-04 | 2.26962194e-04 | 36.737975 | 36.766343 | 0.00177164 | 0.00165880 | 3.087771 | 3.034859 |
| 38.103093 | 0.113062500 | 0.113189980 | 1.97675991e-04 | 1.96804332e-04 | 37.362065 | 37.390692 | 0.00153148 | 0.00143694 | 2.878460 | 2.830356 |
| 38.727185 | 0.113062500 | 0.113527252 | 1.71106132e-04 | 1.70338237e-04 | 37.986127 | 38.014912 | 0.00132367 | 0.00124193 | 2.682649 | 2.637967 |
| 39.351276 | 0.113000000 | 0.113625000 | 1.48152645e-04 | 1.47436803e-04 | 38.610058 | 38.639097 | 0.00114442 | 0.00107453 | 2.499856 | 2.458204 |

The interpolation suggests a favorable local/perceptual tradeoff across the common range, but it does not support an unconditional BER30 dominance claim. Cite the raw `project_test_curve.csv` values alongside this diagnostic.
