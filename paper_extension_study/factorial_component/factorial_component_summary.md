# Fixed-RGB-Weight Factorial Component Summary

Only the four seed17 continuation branches with RGB/global weight 0.5 are included. The frozen Global row with RGB weight 1.0 is intentionally excluded from this factorial design. Values are natural 128×128 project-test results; the formal frozen Hard/Ours rows come from the authoritative bundle, while the two control rows come from the completed extension runs.

| Method | Tail | OKLab | PSNR | Top-25 Local PSNR | P95 MSE | P99 MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini | BER30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RGB0.5-Control | 0 | 0 | 35.164486 | 34.447721 | 3.813118118e-04 | 4.026092408e-04 | 0.00261828 | 4.027174 | 5.858652 | 0.088316 | 0.1130625 |
| Global+OKLab-only | 0 | 1 | 35.943842 | 35.207559 | 3.227863646e-04 | 3.422455385e-04 | 0.00236554 | 3.577205 | 5.215556 | 0.093363 | 0.1132500 |
| Hard Local-Tail | 1 | 0 | 36.442300 | 35.754376 | 2.844551472e-04 | 3.010726967e-04 | 0.00221518 | 3.484790 | 5.075666 | 0.086897 | 0.1129375 |
| Ours | 1 | 1 | 36.854909 | 36.141895 | 2.619925590e-04 | 2.782322409e-04 | 0.00191711 | 3.254091 | 4.772229 | 0.092130 | 0.1131250 |

## Simple effects

Effects are second cell minus first cell; for lower-is-better metrics, negative is favorable.

- `OKLab_without_Tail` (M01 - M00): `Global+OKLab-only − RGB0.5-Control`.
  - PSNR: `+0.779356563`.
  - Top-25 Local PSNR: `+0.759838602`.
  - P95 MSE: `-5.852544724e-05`.
  - P99 MSE: `-6.036370233e-05`.
  - LPIPS: `-0.000252740`.
  - Global CIEDE2000: `-0.449969101`.
  - Top10 CIEDE2000: `-0.643096260`.
  - Gini: `+0.005047799`.
  - BER30: `+0.000187500`.
- `OKLab_with_Tail` (M11 - M10): `Ours − Hard Local-Tail`.
  - PSNR: `+0.412608788`.
  - Top-25 Local PSNR: `+0.387518925`.
  - P95 MSE: `-2.246258817e-05`.
  - P99 MSE: `-2.284045571e-05`.
  - LPIPS: `-0.000298064`.
  - Global CIEDE2000: `-0.230699067`.
  - Top10 CIEDE2000: `-0.303437085`.
  - Gini: `+0.005233313`.
  - BER30: `+0.000187500`.
- `Tail_without_OKLab` (M10 - M00): `Hard Local-Tail − RGB0.5-Control`.
  - PSNR: `+1.277814533`.
  - Top-25 Local PSNR: `+1.306654791`.
  - P95 MSE: `-9.685666460e-05`.
  - P99 MSE: `-1.015365441e-04`.
  - LPIPS: `-0.000403106`.
  - Global CIEDE2000: `-0.542383244`.
  - Top10 CIEDE2000: `-0.782986015`.
  - Gini: `-0.001418607`.
  - BER30: `-0.000125000`.
- `Tail_with_OKLab` (M11 - M01): `Ours − Global+OKLab-only`.
  - PSNR: `+0.911066758`.
  - Top-25 Local PSNR: `+0.934335114`.
  - P95 MSE: `-6.079380553e-05`.
  - P99 MSE: `-6.401329751e-05`.
  - LPIPS: `-0.000448430`.
  - Global CIEDE2000: `-0.323113211`.
  - Top10 CIEDE2000: `-0.443326840`.
  - Gini: `-0.001233093`.
  - BER30: `-0.000125000`.
