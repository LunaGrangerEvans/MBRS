# Quality–robustness Pareto analysis

No model was trained. Image quality uses the frozen RGB `[0,1]` extended metric definition. Robustness uses the latest fixed evaluator for MBRS and the verified TrustMark pre-ECC audit for TrustMark.

## TrustMark raw-bit audit

The official pipeline exposes decoder logits before thresholding and BCH. The raw bit is `decoder_logit > 0` and is compared with the expected 100-bit BCH protected packet. This is verified pre-ECC output, not a guessed or hacked signal.

| Method | Crop | Raw BER ↓ | Raw bit accuracy ↑ | Detection success | ECC exact success | Trials |
|---|---:|---:|---:|---:|---:|---:|
| TrustMark P | 100% | 0.004000 | 0.996000 | 0.9600 | 0.9600 | 250 |
| TrustMark P | 70% | 0.018440 | 0.981560 | 0.8440 | 0.8400 | 250 |
| TrustMark P | 50% | 0.079400 | 0.920600 | 0.3800 | 0.3680 | 250 |
| TrustMark P | 40% | 0.163600 | 0.836400 | 0.0480 | 0.0440 | 250 |
| TrustMark P | 30% | 0.250120 | 0.749880 | 0.0280 | 0.0000 | 250 |
| TrustMark Q | 100% | 0.003800 | 0.996200 | 1.0000 | 1.0000 | 250 |
| TrustMark Q | 70% | 0.007880 | 0.992120 | 0.9720 | 0.9720 | 250 |
| TrustMark Q | 50% | 0.051920 | 0.948080 | 0.5920 | 0.5840 | 250 |
| TrustMark Q | 40% | 0.116400 | 0.883600 | 0.1240 | 0.1120 | 250 |
| TrustMark Q | 30% | 0.190520 | 0.809480 | 0.0120 | 0.0000 | 250 |

## Unified MBRS and TrustMark table

See [quality_robustness_pareto_table.csv](quality_robustness_pareto_table.csv). MBRS exact decode is `N/A` because it reports raw 64-bit BER without ECC; TrustMark exact success is ECC-corrected message success.

| Method | Comparison | Crop | PSNR | Top25 local PSNR | Gini | LPIPS | Raw BER | Bit accuracy | Exact decode success |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MBRS Global continuation | STRICT internal | 100% | 36.263172 | 35.522213 | 0.095007 | 0.00234960 | 0.000000 | 1.000000 | N/A |
| MBRS Global continuation | STRICT internal | 70% | 36.263172 | 35.522213 | 0.095007 | 0.00234960 | 0.000000 | 1.000000 | N/A |
| MBRS Global continuation | STRICT internal | 50% | 36.263172 | 35.522213 | 0.095007 | 0.00234960 | 0.001437 | 0.998563 | N/A |
| MBRS Global continuation | STRICT internal | 40% | 36.263172 | 35.522213 | 0.095007 | 0.00234960 | 0.041625 | 0.958375 | N/A |
| MBRS Global continuation | STRICT internal | 30% | 36.263172 | 35.522213 | 0.095007 | 0.00234960 | 0.113125 | 0.886875 | N/A |
| MBRS Hard Patch16 / stride8 / Top10 / MSE / image-local 0.5/0.5 | STRICT internal | 100% | 36.442300 | 35.754376 | 0.086897 | 0.00221518 | 0.000000 | 1.000000 | N/A |
| MBRS Hard Patch16 / stride8 / Top10 / MSE / image-local 0.5/0.5 | STRICT internal | 70% | 36.442300 | 35.754376 | 0.086897 | 0.00221518 | 0.000000 | 1.000000 | N/A |
| MBRS Hard Patch16 / stride8 / Top10 / MSE / image-local 0.5/0.5 | STRICT internal | 50% | 36.442300 | 35.754376 | 0.086897 | 0.00221518 | 0.001500 | 0.998500 | N/A |
| MBRS Hard Patch16 / stride8 / Top10 / MSE / image-local 0.5/0.5 | STRICT internal | 40% | 36.442300 | 35.754376 | 0.086897 | 0.00221518 | 0.041813 | 0.958187 | N/A |
| MBRS Hard Patch16 / stride8 / Top10 / MSE / image-local 0.5/0.5 | STRICT internal | 30% | 36.442300 | 35.754376 | 0.086897 | 0.00221518 | 0.112938 | 0.887062 | N/A |
| TrustMark Q | REFERENCE | 100% | 42.632394 | 39.707519 | 0.355392 | 0.00096475 | 0.003800 | 0.996200 | 1.0000 |
| TrustMark Q | REFERENCE | 70% | 42.632394 | 39.707519 | 0.355392 | 0.00096475 | 0.007880 | 0.992120 | 0.9720 |
| TrustMark Q | REFERENCE | 50% | 42.632394 | 39.707519 | 0.355392 | 0.00096475 | 0.051920 | 0.948080 | 0.5840 |
| TrustMark Q | REFERENCE | 40% | 42.632394 | 39.707519 | 0.355392 | 0.00096475 | 0.116400 | 0.883600 | 0.1120 |
| TrustMark Q | REFERENCE | 30% | 42.632394 | 39.707519 | 0.355392 | 0.00096475 | 0.190520 | 0.809480 | 0.0000 |
| TrustMark P | REFERENCE | 100% | 48.308034 | 46.572266 | 0.196823 | 0.00030470 | 0.004000 | 0.996000 | 0.9600 |
| TrustMark P | REFERENCE | 70% | 48.308034 | 46.572266 | 0.196823 | 0.00030470 | 0.018440 | 0.981560 | 0.8400 |
| TrustMark P | REFERENCE | 50% | 48.308034 | 46.572266 | 0.196823 | 0.00030470 | 0.079400 | 0.920600 | 0.3680 |
| TrustMark P | REFERENCE | 40% | 48.308034 | 46.572266 | 0.196823 | 0.00030470 | 0.163600 | 0.836400 | 0.0440 |
| TrustMark P | REFERENCE | 30% | 48.308034 | 46.572266 | 0.196823 | 0.00030470 | 0.250120 | 0.749880 | 0.0000 |

## Core observations

- Hard Top10 versus Global improves frozen image quality: PSNR `+0.179128 dB`, Top25 local PSNR `+0.232163 dB`, Gini `-0.008110`, and full LPIPS `-0.00013442`.
- At 70% crop, Hard Top10 and Global both have raw bit accuracy `1.000000`; at 50/40/30%, Hard Top10 changes BER by only `+0.000063/+0.000188/-0.000188`, respectively.
- TrustMark Q has lower raw BER than P at every partial crop, while P has better image quality. This is a clear Q-versus-P quality–robustness trade-off within TrustMark.
- TrustMark points occupy a higher-quality region, but they remain REFERENCE points because the packet is 100 protected bits with BCH-5/61 data bits, not MBRS raw 64-bit BER.
- The plots use marker shape for method family and color for crop ratio; no strict superiority boundary is drawn.

## Claim boundary

The evidence supports an internal Hard Top10 quality–robustness operating point: better frozen quality with essentially preserved MBRS raw BER. It does not support an ours-over-TrustMark strict or human-perceptual superiority claim.
