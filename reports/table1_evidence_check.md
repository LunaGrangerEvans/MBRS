# Table I frozen-evidence check

## Evidence boundary

- Protocol: `paper-freeze-v1; current clipped RGB evaluator`; `training=false`.
- Primary 50-image evidence: `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv`.
- Primary evidence SHA-256: `dc332a5ca6fe36a1a791944ffef6bd2f3c47d8b47327871171426b3bef96c7c6` (matches `paper/evidence_manifest.json`).
- Frozen aggregate cross-check: `/root/workspace/GLX/icassp/MBRS/paper/main_table.csv`.
- Aggregate SHA-256: `5d5829cb57b796accbeedfb9ce6b9fa28c1f65873451619c382576caed07c2cb` (matches `paper/evidence_manifest.json`).
- Legacy metrics were not read or used.
- PSNR is recomputed as `10 log10(1 / mean(global_mse))`; SSIM, LPIPS, Top-25 local PSNR, P95 MSE, Gini, and BER@30% are means over the 50 frozen per-image rows.

## Cell-by-cell verification

Every cell uses the primary path `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` and is independently matched to the corresponding run/column in `/root/workspace/GLX/icassp/MBRS/paper/main_table.csv`.

| Method | Cell | Source column / aggregation | Verified value | Expected value | Status |
|---|---|---|---:|---:|:---:|
| Global continuation | PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `10 log10(1 / mean(global_mse))` | 36.263172 | 36.263172 | **PASS** |
| Global continuation | SSIM | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(global_ssim), n=50` | 0.950759 | 0.950759 | **PASS** |
| Global continuation | LPIPS | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(full_lpips), n=50` | 0.002350 | 0.002350 | **PASS** |
| Global continuation | Top-25 local PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(top25_local_psnr), n=50` | 35.522213 | 35.522213 | **PASS** |
| Global continuation | P95 MSE | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(patch_mse_p95), n=50` | 3.016532e-4 | 3.016532e-4 | **PASS** |
| Global continuation | Gini | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(gini), n=50` | 0.095007 | 0.095007 | **PASS** |
| Global continuation | BER@30% | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(ber30), n=50` | 0.1131250 | 0.1131250 | **PASS** |
| Hard P16/S16/Top-25% | PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `10 log10(1 / mean(global_mse))` | 36.418147 | 36.418147 | **PASS** |
| Hard P16/S16/Top-25% | SSIM | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(global_ssim), n=50` | 0.952313 | 0.952313 | **PASS** |
| Hard P16/S16/Top-25% | LPIPS | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(full_lpips), n=50` | 0.002200 | 0.002200 | **PASS** |
| Hard P16/S16/Top-25% | Top-25 local PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(top25_local_psnr), n=50` | 35.716045 | 35.716045 | **PASS** |
| Hard P16/S16/Top-25% | P95 MSE | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(patch_mse_p95), n=50` | 2.876246e-4 | 2.876246e-4 | **PASS** |
| Hard P16/S16/Top-25% | Gini | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(gini), n=50` | 0.089031 | 0.089031 | **PASS** |
| Hard P16/S16/Top-25% | BER@30% | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(ber30), n=50` | 0.1131875 | 0.1131875 | **PASS** |
| Hard P16/S8/Top-25% | PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `10 log10(1 / mean(global_mse))` | 36.445329 | 36.445329 | **PASS** |
| Hard P16/S8/Top-25% | SSIM | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(global_ssim), n=50` | 0.952631 | 0.952631 | **PASS** |
| Hard P16/S8/Top-25% | LPIPS | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(full_lpips), n=50` | 0.002205 | 0.002205 | **PASS** |
| Hard P16/S8/Top-25% | Top-25 local PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(top25_local_psnr), n=50` | 35.747490 | 35.747490 | **PASS** |
| Hard P16/S8/Top-25% | P95 MSE | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(patch_mse_p95), n=50` | 2.856057e-4 | 2.856057e-4 | **PASS** |
| Hard P16/S8/Top-25% | Gini | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(gini), n=50` | 0.088604 | 0.088604 | **PASS** |
| Hard P16/S8/Top-25% | BER@30% | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(ber30), n=50` | 0.1131875 | 0.1131875 | **PASS** |
| Hard P16/S8/Top-10% (Ours) | PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `10 log10(1 / mean(global_mse))` | 36.442300 | 36.442300 | **PASS** |
| Hard P16/S8/Top-10% (Ours) | SSIM | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(global_ssim), n=50` | 0.952541 | 0.952541 | **PASS** |
| Hard P16/S8/Top-10% (Ours) | LPIPS | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(full_lpips), n=50` | 0.002215 | 0.002215 | **PASS** |
| Hard P16/S8/Top-10% (Ours) | Top-25 local PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(top25_local_psnr), n=50` | 35.754376 | 35.754376 | **PASS** |
| Hard P16/S8/Top-10% (Ours) | P95 MSE | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(patch_mse_p95), n=50` | 2.844551e-4 | 2.844551e-4 | **PASS** |
| Hard P16/S8/Top-10% (Ours) | Gini | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(gini), n=50` | 0.086897 | 0.086897 | **PASS** |
| Hard P16/S8/Top-10% (Ours) | BER@30% | `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` -> `mean(ber30), n=50` | 0.1129375 | 0.1129375 | **PASS** |

## Column-best audit

| Column | True best method | Value |
|---|---|---:|
| PSNR | Hard P16/S8/Top-25% | 36.445329 |
| SSIM | Hard P16/S8/Top-25% | 0.952631 |
| LPIPS | Hard P16/S16/Top-25% | 0.002200 |
| Top-25 local PSNR | Hard P16/S8/Top-10% (Ours) | 35.754376 |
| P95 MSE | Hard P16/S8/Top-10% (Ours) | 2.844551e-4 |
| Gini | Hard P16/S8/Top-10% (Ours) | 0.086897 |
| BER@30% | Hard P16/S8/Top-10% (Ours) | 0.1129375 |

Only the true best numeric cell in each column is bold. The Ours method name is bold, but its PSNR, SSIM, and LPIPS values are not bold because they are not column-best.

## Formatting checks

- Uses `table*`, `booktabs`, `tabular*`, and `\textwidth`.
- Caption is above the table and the label is `tab:main_results`.
- Uses `\scriptsize`; no `resizebox` and no vertical rules.
- The notation footnote defines P16, S8/S16, and Top-k.
