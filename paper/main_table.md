# Frozen main table

Eight existing seed17 epoch20 checkpoints, one shared epoch100 source. Image weights are global/local 0.5/0.5 except Global 1/0; message MSE weight 10; LR=1e-4; batch16. No new training.

All quality values use the current clipped-RGB evaluator. PSNR is computed from dataset mean MSE. Top25 local PSNR is the **mean of per-image** PSNR of the four highest-MSE native32 patches. P95 is the mean per-image P95, not a pooled percentile. Gini and other concentration metrics use 16 native32 non-overlap patches. BER replays the frozen unquantized normalized-output mask protocol. Units and aggregation are defined in draft §5.

| Method | PSNR ↑ | SSIM ↑ | 3-scale MS-SSIM ↑ | LPIPS ↓ | Top25 local PSNR ↑ | P95 MSE ↓ | Gini ↓ | Top10/Mean ↓ | BER50 ↓ | BER30 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Global continuation (RGB image weight 1)** | 36.263172 | 0.950759 | 0.983036 | 0.002349598 | 35.522213 | 0.000301653 | 0.095007 | 1.295702 | 0.001437 | 0.113125 |
| Hard MSE, patch16 / stride16 / Top25% | 36.418147 | 0.952313 | 0.983600 | 0.002200142 | 35.716045 | 0.000287625 | 0.089031 | 1.278384 | 0.001563 | 0.113187 |
| Hard MSE, patch16 / stride8 / Top25% | 36.445329 | 0.952631 | 0.983713 | 0.002205026 | 35.747490 | 0.000285606 | 0.088604 | 1.277077 | 0.001563 | 0.113187 |
| **Hard MSE, patch16 / stride8 / Top10% (main)** | 36.442300 | 0.952541 | 0.983690 | 0.002215177 | 35.754376 | 0.000284455 | 0.086897 | 1.270921 | 0.001500 | 0.112937 |
| Soft normalized weighting, patch16 / stride16 / T=0.25 | 36.350297 | 0.951587 | 0.983356 | 0.002211002 | 35.646696 | 0.000291918 | 0.089091 | 1.277649 | 0.001500 | 0.113375 |
| Multi-scale hard Top25%, patch16/stride16 (0.7) + patch32/stride32 (0.3) | 36.391394 | 0.952069 | 0.983516 | 0.002198487 | 35.688070 | 0.000289416 | 0.089119 | 1.278703 | 0.001437 | 0.113062 |
| Excess, patch16 / stride16 / threshold1 / scale3 | 35.903160 | 0.946346 | 0.981532 | 0.002351493 | 35.210189 | 0.000321119 | 0.086731 | 1.268679 | 0.001437 | 0.112937 |
| Gradient-aware MSE, patch16 / stride8 / Top10% / alpha2 | 36.332801 | 0.952017 | 0.983473 | 0.002199208 | 35.624195 | 0.000293327 | 0.089410 | 1.278817 | 0.001500 | 0.113437 |

Bold identifies the fixed primary comparison, not column winners. Soft T=0.25 is the historical best local-pixel-tail Soft setting, not a new selection. The CSV additionally contains CV, energy share, perceptual tails, all five BER values, and run identifiers.

Source: [main_table.csv](main_table.csv), [per-image evidence](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv), [provenance](evidence_manifest.json).
