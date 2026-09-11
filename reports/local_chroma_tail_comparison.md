# Local chroma-tail comparison (validation only)

One seed17 continuation was run from the common epoch100 source. The loss keeps the existing Hard RGB tail and adds one conservative Top10 chroma-tail term. These results use the fixed validation manifest and are not formal-test main results.

| Method | PSNR | Top25 local PSNR | Global CIEDE2000 | CIEDE Top10 | CIEDE P95 | Global chroma | Chroma Top10 | P95 MSE | Gini | BER30 | BER40 | BER50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hard Top10 incumbent | 36.410612515 | 35.652987524 | 3.598625340 | 5.230329256 | 5.124217739 | 0.010952966 | 0.015500122 | 0.000294555 | 0.094568882 | 0.1110000 | 0.0435625 | 0.0158750 |
| Hard Top10 + global OKLab | 36.650197422 | 35.879871061 | 3.465416603 | 5.051264009 | 4.946861575 | 0.010564240 | 0.015017297 | 0.000280816 | 0.097303423 | 0.1108125 | 0.0437500 | 0.0158750 |
| Hard Top10 + local chroma-tail | 36.535881338 | 35.776874823 | 3.533374786 | 5.132034798 | 5.027569327 | 0.010751649 | 0.015221145 | 0.000286894 | 0.095257628 | 0.1109375 | 0.0436250 | 0.0158125 |

## Delta versus incumbent Hard Top10

### Hard Top10 + global OKLab
- global_psnr: +0.239584907
- top25_local_psnr: +0.226883537
- global_ciede2000: -0.133208737
- ciede2000_top10: -0.179065247
- global_chroma: -0.000388726
- chroma_top10: -0.000482826
- patch_mse_p95: -0.000013739
- gini: +0.002734541
- ber30: -0.000187500
- ber40: +0.000187500
- ber50: +0.000000000
- images with lower ciede2000_top10: 100.0%
- images with lower chroma_top10: 100.0%
- images with lower patch_mse_p95: 100.0%
- images with lower gini: 14.0%
### Hard Top10 + local chroma-tail
- global_psnr: +0.125268824
- top25_local_psnr: +0.123887299
- global_ciede2000: -0.065250554
- ciede2000_top10: -0.098294458
- global_chroma: -0.000201316
- chroma_top10: -0.000278977
- patch_mse_p95: -0.000007661
- gini: +0.000688745
- ber30: -0.000062500
- ber40: +0.000062500
- ber50: -0.000062500
- images with lower ciede2000_top10: 98.0%
- images with lower chroma_top10: 96.0%
- images with lower patch_mse_p95: 100.0%
- images with lower gini: 34.0%

## Interpretation

The local chroma-tail model is the direct target of this experiment: it selects the largest 5×5 chroma distances inside the same Patch16/stride8/Top10-style local grid and optimizes only that chroma term in addition to the inherited RGB objective. A lower CIEDE2000 or chroma-only value is the desired color-tail direction; Gini and P95 MSE are checked for collateral effects.

No model selection was performed using the formal test manifest. The global-OKLab row is the previously completed validation extension at lambda0.029574882. The comparison is therefore a validation comparison among three frozen endpoints, not a new formal ranking.

Sources: [validation candidates](crop_global_oklab_validation_candidates.csv), [per-image comparison](local_chroma_tail_comparison.csv), [calibration](crop_global_oklab/calibration.json), and the local-chroma checkpoint under `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_local_chroma/`.
