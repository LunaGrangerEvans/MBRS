# Extended metrics and external baselines final summary

## Phase A results

The frozen formal manifest contains 50 images. Extended evaluation used RGB [0,1], full-image SSIM, valid 3-scale MS-SSIM with weights [0.3,0.3,0.4], native32 patch SSIM, full-image LPIPS, and native32 patch LPIPS. No training was run.

| Method | PSNR | SSIM | MS-SSIM | LPIPS | Bottom10 SSIM | Top10 LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Global continuation | 36.263172 | 0.950759 | 0.983036 | 0.00234960 | 0.902273 | 0.00136127 |
| Hard Patch16 stride8 Top25 | 36.445329 | 0.952631 | 0.983713 | 0.00220503 | 0.905780 | 0.00129904 |
| Hard Patch16 stride8 Top10 | 36.442300 | 0.952541 | 0.983690 | 0.00221518 | 0.905504 | 0.00136614 |
| Gradient-aware Top10 alpha2 | 36.332801 | 0.952017 | 0.983473 | 0.00219921 | 0.905216 | 0.00121320 |

DISTS and GMSD are N/A because no mature implementation was installed and no implementation was copied. Patch MS-SSIM is omitted because 32×32 support is insufficient for a defensible multi-scale calculation.

The Phase A machine-readable outputs are:

- [per-image metrics](/mnt/wmcontent/GLX/icassp/MBRS/reports/extended_image_quality/per_image.csv)
- [summary CSV](/mnt/wmcontent/GLX/icassp/MBRS/reports/extended_image_quality/summary.csv)
- [sanity JSON](/mnt/wmcontent/GLX/icassp/MBRS/reports/extended_image_quality/sanity.json)

Interpretation: Hard Top10 supports SSIM/MS-SSIM and full-image LPIPS improvement over Global, while its native32 Top10 LPIPS is essentially unchanged. The perceptual evidence is mixed, not a uniform win.

## Phase B results

- TrustMark Q/P: official isolated environment configured, official example completed, official encoder/decoder hashes verified, 50 outputs per model saved, and fixed crop decode evaluated. Status: completed reference inference.
- StegaStamp: official checkout exists, but no verified SavedModel checkpoint is available at the pinned checkout. Status: blocked.
- HiDDeN: no verified official pretrained checkpoint was found in scope. Status: requires retraining; no retraining was started.

Actual TrustMark values and crop results are in [final_external_baseline_summary.md](final_external_baseline_summary.md), [external_baseline_main_table.md](external_baseline_main_table.md), and `/mnt/wmcontent/GLX/icassp/MBRS/reports/external_baseline_metrics/`.

The audit artifacts are:

- [compatibility audit](external_baseline_compatibility_audit.md)
- [compatibility CSV](external_baseline_compatibility_audit.csv)
- [crop protocol](external_crop_protocol.md)
- [external main table](external_baseline_main_table.md)
- [external reproduction notes](../external_baselines/REPRODUCE.md)

## Supported claims

- The current Hard Top10 method improves the measured MBRS local pixel-tail under the frozen controlled protocol with small BER change.
- Extended full-reference metrics show SSIM/MS-SSIM and full-image LPIPS support, but native local LPIPS is mixed.
- No strict external baseline ranking is available because TrustMark is a payload/ECC-mismatched reference comparison.

## Partially supported claims

- The method improves perceptual quality: full-image metrics improve, while local LPIPS tail is not consistently better.
- The method is visually less artifact-prone: actual RGB observation and perceptual tails show only weak/mixed evidence.

## Unsupported claims

- Strict superiority over StegaStamp, TrustMark or HiDDeN.
- Large human-perceived artifact reduction.
- DISTS/GMSD-based superiority; those metrics were unavailable.
