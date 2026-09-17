# Table II validation-only evidence check

> **VALIDATION ONLY.** This report and Table II are isolated from the formal-test evidence used by Table I.

## Endpoint identity

- Validation manifest: `/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/validation_manifest.pt`; SHA-256 `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`; 50 images.
- Global OKLab config: `/root/workspace/GLX/icassp/MBRS/experiments/config_seed17_crop_hard16_stride8_top10_global_oklab_g25.json`; run `seed17_crop_hard16_stride8_top10_global_oklab_g25`; `lambda_OK=0.059149764`.
- Local chroma-tail config: `/root/workspace/GLX/icassp/MBRS/experiments/config_seed17_crop_hard16_stride8_top10_local_chroma.json`; run `seed17_crop_hard16_stride8_top10_local_chroma`; `local_loss_mode=chroma_topk`, P16/S8/Top-10%, `lambda=0.014221079`.
- Local chroma-tail checkpoint: `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_local_chroma/checkpoint_0020.pth`; SHA-256 `92ba2b77aacf3e80394e28b39f4a7ad9d5dcbb64e8d956e43d3fa95ba43ee2d7`.
- Latest-endpoint check: the mounted artifact search contains one completed `*local*chroma*outputs.pt` endpoint, under `validation_seed17_crop_hard16_stride8_top10_local_chroma`; its artifacts are newer than the global-OKLab validation artifacts and match the run/config/checkpoint above.
- Provenance caveat: the local-chroma `summary.csv`/`per_image.csv` retained the producer's stale display string `Hard16 stride8 Top10 + global OKLab`. Identity is instead established by the containing run directory, summary `run` field, provenance checkpoint entry, local-chroma config, and dedicated local-chroma comparison artifact. The numeric block is internally consistent across all of them.

## Locked source artifacts

| Artifact | SHA-256 | Status |
|---|---|:---:|
| `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/per_image.csv` | `910fa34e715b451dc096a7c779f177d571ad3c4eab76cea8fa73382b883f42d1` | **PASS** |
| `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/summary.csv` | `890a83e0adff70b9f4eb71417f451080dedd3d1dbca3ad6d899e858eb8e578a6` | **PASS** |
| `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/provenance.json` | `3075d566c65cae862d50c38a7c37c5890546bf09e7b9161568e2f23e03fa78db` | **PASS** |
| `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_local_chroma/per_image.csv` | `1c053d6e2d26a15fb9a58a4b454f453fe3750150be5d63bfc22820cbe145eeec` | **PASS** |
| `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_local_chroma/summary.csv` | `2dc472853e750ffc4f737d0c9d5d9349141ffd4d7310ba9bbfec5eb1c127f588` | **PASS** |
| `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_local_chroma/provenance.json` | `515ec0ee2ec66ebd141dafff2d16c5df193f5a3b3a9f233c8ea0a3e8a45c757d` | **PASS** |
| `/root/workspace/GLX/icassp/MBRS/experiments/config_seed17_crop_hard16_stride8_top10_global_oklab_g25.json` | `0f1b3fcbb2f53ac346c18b9699bc7d7d49b5964a039cd0b779e1b3cf429e02fd` | **PASS** |
| `/root/workspace/GLX/icassp/MBRS/experiments/config_seed17_crop_hard16_stride8_top10_local_chroma.json` | `46b40cc38edc231df13079faa85e4093a9ddf726f1d62a5cb5c1e89e1a37c8b6` | **PASS** |

## Cell-by-cell verification

PSNR is recomputed as `10 log10(1 / mean(global_mse))`; all other cells are means over the 50 per-image validation rows. Every aggregate is also matched to the corresponding `summary.csv` row.

| Method | Cell | Source artifact/path and aggregation | Verified | Expected | Status |
|---|---|---|---:|---:|:---:|
| Hard Top-10% | PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/per_image.csv` -> `10 log10(1 / mean(global_mse))` | 36.410612515 | 36.4106 | **PASS** |
| Hard Top-10% | Top-25 local PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/per_image.csv` -> `mean(top25_local_psnr), n=50` | 35.652987524 | 35.6530 | **PASS** |
| Hard Top-10% | Global CIEDE2000 | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/per_image.csv` -> `mean(ciede2000_global), n=50` | 3.598625340 | 3.5986 | **PASS** |
| Hard Top-10% | Top-10 CIEDE2000 | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/per_image.csv` -> `mean(ciede2000_top10), n=50` | 5.230329256 | 5.2303 | **PASS** |
| Hard Top-10% | Gini | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/per_image.csv` -> `mean(gini), n=50` | 0.094568882 | 0.09457 | **PASS** |
| Hard Top-10% | BER@30% | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/per_image.csv` -> `mean(ber30), n=50` | 0.111000000 | 0.11100 | **PASS** |
| + Global OKLab | PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/per_image.csv` -> `10 log10(1 / mean(global_mse))` | 36.818149934 | 36.8182 | **PASS** |
| + Global OKLab | Top-25 local PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/per_image.csv` -> `mean(top25_local_psnr), n=50` | 36.032251722 | 36.0323 | **PASS** |
| + Global OKLab | Global CIEDE2000 | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/per_image.csv` -> `mean(ciede2000_global), n=50` | 3.365011549 | 3.3650 | **PASS** |
| + Global OKLab | Top-10 CIEDE2000 | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/per_image.csv` -> `mean(ciede2000_top10), n=50` | 4.920614367 | 4.9206 | **PASS** |
| + Global OKLab | Gini | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/per_image.csv` -> `mean(gini), n=50` | 0.100017246 | 0.10002 | **PASS** |
| + Global OKLab | BER@30% | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/per_image.csv` -> `mean(ber30), n=50` | 0.110812500 | 0.11081 | **PASS** |
| + Local chroma-tail | PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_local_chroma/per_image.csv` -> `10 log10(1 / mean(global_mse))` | 36.535881338 | 36.5359 | **PASS** |
| + Local chroma-tail | Top-25 local PSNR | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_local_chroma/per_image.csv` -> `mean(top25_local_psnr), n=50` | 35.776874823 | 35.7769 | **PASS** |
| + Local chroma-tail | Global CIEDE2000 | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_local_chroma/per_image.csv` -> `mean(ciede2000_global), n=50` | 3.533374786 | 3.5334 | **PASS** |
| + Local chroma-tail | Top-10 CIEDE2000 | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_local_chroma/per_image.csv` -> `mean(ciede2000_top10), n=50` | 5.132034798 | 5.1320 | **PASS** |
| + Local chroma-tail | Gini | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_local_chroma/per_image.csv` -> `mean(gini), n=50` | 0.095257628 | 0.09526 | **PASS** |
| + Local chroma-tail | BER@30% | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_local_chroma/per_image.csv` -> `mean(ber30), n=50` | 0.110937500 | 0.11094 | **PASS** |

Expected manuscript values are checked within one unit of their last reported decimal. Global OKLab PSNR is `36.818149934` in the frozen artifact; the retained manuscript display `36.8182` reflects the project's existing intermediate six-decimal value `36.818150`. This display-only double rounding does not affect any ranking.

## Column-best audit

| Column | True best row | Value |
|---|---|---:|
| PSNR | + Global OKLab | 36.8182 |
| Top-25 local PSNR | + Global OKLab | 36.0323 |
| Global CIEDE2000 | + Global OKLab | 3.3650 |
| Top-10 CIEDE2000 | + Global OKLab | 4.9206 |
| Gini | Hard Top-10% | 0.09457 |
| BER@30% | + Global OKLab | 0.11081 |

The Hard Top-10% incumbent is correctly bold only for Gini. Global OKLab is correctly bold for PSNR, Top-25 local PSNR, both CIEDE2000 columns, and BER@30%.

## Separation and formatting checks

- The caption begins with bold `VALIDATION ONLY` and refers only to the fixed validation manifest.
- The mandatory note states that these rows are development-validation evidence and are not part of Table I's formal-test comparison.
- Uses `table*`, `booktabs`, `tabular*`, `\textwidth`, and `\scriptsize`; no vertical rules and no `resizebox`.
- Label is `tab:color_extension`; caption is above the table.
