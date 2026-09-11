# Final external-baseline summary

Audit date: 2026-09-10. No new MBRS model was trained. The fixed formal manifest contains 50 images; TrustMark crop results use five fixed repeats per crop area.

## Which baselines actually ran

- **TrustMark Q: COMPLETED**. Official Adobe checkout, official encoder/decoder files, isolated environment, official example, 50-image encode/decode, saved outputs, and crop evaluation all completed.
- **TrustMark P: COMPLETED** under the same conditions.
- **StegaStamp: BLOCKED**. The official checkout has no verified pretrained SavedModel checkpoint at the pinned commit; no unverified model was downloaded. An independent Python 3.7/TF1 environment specification is recorded, but the configured conda mirror failed incomplete/timeout downloads, so no partial environment is treated as valid.
- **HiDDeN: REQUIRES RETRAINING**. No verified pretrained checkpoint and reproducible official inference path were available in scope; this phase did not train it.

## Frozen image-quality results

These are not strict cross-method rankings. The image-quality columns use the same frozen RGB `[0,1]` definition for MBRS and TrustMark; DISTS/GMSD remain `N/A`.

| Method | Class | PSNR | SSIM | 3-scale MS-SSIM | LPIPS | Top25 local PSNR | P95 patch MSE | Gini | CV | Top10/Mean | Top10 energy share | Bottom10 SSIM | Top10 LPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MBRS Global continuation | STRICT internal | 36.263172 | 0.950759 | 0.983036 | 0.00234960 | 35.522213 | 0.000301653 | 0.095007 | 0.173718 | 1.295702 | 0.161963 | 0.902273 | 0.00136127 |
| MBRS Hard Patch16 / stride8 / Top10 / MSE / 0.5/0.5 | STRICT internal | 36.442300 | 0.952541 | 0.983690 | 0.00221518 | 35.754376 | 0.000284455 | 0.086897 | 0.159015 | 1.270921 | 0.158865 | 0.905504 | 0.00136614 |
| TrustMark Q | REFERENCE only | 42.632394 | 0.990987 | 0.995099 | 0.00096475 | 39.707519 | 0.000130456 | 0.355392 | 0.675501 | 2.344941 | 0.293118 | 0.977458 | 0.00039995 |
| TrustMark P | REFERENCE only | 48.308034 | 0.997511 | 0.998849 | 0.00030470 | 46.572266 | 0.000025546 | 0.196823 | 0.361190 | 1.681308 | 0.210164 | 0.994554 | 0.00022314 |

TrustMark's larger PSNR is a quality/reference observation, not evidence of strict superiority: it carries a different protected payload, uses different preprocessing/resolution, and exposes ECC-corrected decode rather than raw 64-bit BER.

## Crop robustness

| Method | 100% | 70% | 50% | 40% | 30% |
|---|---:|---:|---:|---:|---:|
| MBRS Global continuation | BER 0.000000 | BER 0.000000 | BER 0.001438 | BER 0.041625 | BER 0.113125 |
| MBRS Hard Top10 | BER 0.000000 | BER 0.000000 | BER 0.001500 | BER 0.041813 | BER 0.112938 |
| TrustMark Q | exact 100.0% / detect 100.0% | exact 97.2% / detect 97.2% | exact 58.4% / detect 59.2% | exact 11.2% / detect 12.4% | exact 0.0% / detect 1.2% |
| TrustMark P | exact 96.0% / detect 96.0% | exact 84.0% / detect 84.4% | exact 36.8% / detect 38.0% | exact 4.4% / detect 4.8% | exact 0.0% / detect 2.8% |

The crop trends are reported descriptively only. TrustMark exact-message success cannot be numerically ranked against MBRS raw BER.

## Metric-consistency correction

The old audit used direct `[-1,1]` tensor MSE/PSNR and a project SSIM implementation with Gaussian 5×5, zero padding, `max_value=2`. The frozen external/extended definition clips to RGB `[0,1]`, uses dataset-wide mean MSE, and uses `pytorch_msssim` valid Gaussian windows. On the same Global and Hard Top10 checkpoints this changes PSNR by about `+0.0358/+0.0351 dB` and Bottom10 SSIM by `-0.01533/-0.01457`. These are definition changes, not model changes. The full audit is [external_metric_consistency_audit.md](external_metric_consistency_audit.md).

All new external tables use only the frozen definition. The old controlled master remains the lineage source for training deltas and raw BER, and its legacy absolute image-quality values are not silently mixed with the new external rows.

## Claims supported by this phase

- TrustMark Q/P can be reproduced from official source code and official declared model files in an isolated environment.
- TrustMark Q/P provide reference image-quality and crop decode measurements on the fixed formal images.
- MBRS Hard Top10 remains the controlled internal pixel-tail method under the seed17 protocol.
- External methods must be labeled `REFERENCE` when payload/ECC and decoder semantics are not equivalent.

## Claims not supported

- Strict superiority of MBRS over TrustMark, StegaStamp, or HiDDeN.
- A strict BER ranking between MBRS and TrustMark.
- Any StegaStamp performance claim without a verified checkpoint.
- Any HiDDeN performance claim without a verified checkpoint or a separately approved retraining study.
- DISTS/GMSD superiority, because those metrics were not available.
