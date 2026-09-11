# External reference table

TrustMark Q/P: official released models, strength unchanged, 100 transmitted bits including 61 data bits, BCH parity and schema. Every cross-family comparison is **REFERENCE**. PNG output and native decoder resizing differ from MBRS tensor inference. No strict ranking is made.

| Method | PSNR ↑ | Top25 local PSNR ↑ | Gini ↓ | Full LPIPS ↓ |
|---|---:|---:|---:|---:|
| Global continuation (RGB image weight 1) | 36.263172 | 35.522213 | 0.095007 | 0.002349598 |
| Hard MSE, patch16 / stride8 / Top10% (main) | 36.442300 | 35.754376 | 0.086897 | 0.002215177 |
| TrustMark Q | 42.293755 | 39.707519 | 0.355392 | 0.000964749 |
| TrustMark P | 48.107281 | 46.572266 | 0.196823 | 0.000304701 |

| Method | Retained area | Pre-ECC packet BER ↓ | Packet bit accuracy ↑ | ECC exact message ↑ | Decode flag ↑ |
|---|---:|---:|---:|---:|---:|
| TrustMark Q | 100% | 0.003800 | 0.996200 | 1.000000 | 1.000000 |
| TrustMark Q | 70% | 0.007880 | 0.992120 | 0.972000 | 0.972000 |
| TrustMark Q | 50% | 0.051920 | 0.948080 | 0.584000 | 0.592000 |
| TrustMark Q | 40% | 0.116400 | 0.883600 | 0.112000 | 0.124000 |
| TrustMark Q | 30% | 0.190520 | 0.809480 | 0.000000 | 0.012000 |
| TrustMark P | 100% | 0.004000 | 0.996000 | 0.960000 | 0.960000 |
| TrustMark P | 70% | 0.018440 | 0.981560 | 0.840000 | 0.844000 |
| TrustMark P | 50% | 0.079400 | 0.920600 | 0.368000 | 0.380000 |
| TrustMark P | 40% | 0.163600 | 0.836400 | 0.044000 | 0.048000 |
| TrustMark P | 30% | 0.250120 | 0.749880 | 0.000000 | 0.028000 |

Quality recomputed by aggregation from saved per-image records: older TrustMark PSNR used mean image PSNR; here it uses dataset mean MSE exactly as MBRS. No encoder or strength change. Raw-bit/detection records are replayed from the verified official pre-ECC audit, not newly optimized. The decode flag is not evidence of correct payload or a calibrated detection rate on unwatermarked images.

StegaStamp: BLOCKED (no verified checkpoint in the audited environment). HiDDeN: REQUIRES RETRAINING in the audited setup. Neither receives numeric performance claims.
