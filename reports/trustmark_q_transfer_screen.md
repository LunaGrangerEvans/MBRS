# TrustMark-Q bounded transfer screen

**Status: completed bounded screen; no long training started.**

This is an **official TrustMark-Q weights initialized custom transfer** experiment. It is not official TrustMark fine-tuning: the public training loss/data/noise pipeline is incomplete, as documented in [the prior audit](strong_backbone_transfer_audit.md).

## Fixed protocol

- Seed: `170917`; train data: first `64` lexicographically sorted DIV2K train PNGs; evaluation: the fixed 50-image manifest.
- Budget: `2` epochs, batch `4`, Adam encoder-only, LR `1e-06`; all three arms use the same order and masks.
- Training crop: same deterministic `RandomCrop(0.3, 1.0)` mask sequence for every arm; no JPEG or extra augmentation.
- Decoder: fixed Q ResNet-50, `eval()` mode, no decoder gradients or BatchNorm updates.
- Arm differences only: Q-control has base proxy; Hard adds `0.5 × Hard local-tail RGB (P16/stride8/Top10)`; OKLab adds fixed `0.05 × global OKLab`.
- Evaluation crop: fixed `100/70/50/40/30%` masks, five repeats per ratio; raw bits are the official decoder logits thresholded before BCH/ECC.

## Clean image quality and color

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Top25 local PSNR ↑ | P95 MSE ↓ | Gini ↓ | CIEDE2000 mean ↓ | CIEDE2000 P95 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q-control | 42.605575 | 0.990362 | 0.00103866 | 39.439787 | 0.000141198 | 0.396997 | 0.452890 | 1.452663 |
| Q-hard-local-tail | 42.604832 | 0.990363 | 0.00103906 | 39.439306 | 0.000141215 | 0.396973 | 0.453748 | 1.452982 |
| Q-hard-local-tail-oklab | 42.605551 | 0.990364 | 0.00103889 | 39.439884 | 0.000141192 | 0.396987 | 0.453284 | 1.452710 |

## Crop raw-bit accuracy / BER

| Method | 100% acc / BER | 70% acc / BER | 50% acc / BER | 40% acc / BER | 30% acc / BER |
|---|---:|---:|---:|---:|---:|
| Q-control | 0.998000 / 0.002000 | 0.995240 / 0.004760 | 0.955760 / 0.044240 | 0.893000 / 0.107000 | 0.815760 / 0.184240 |
| Q-hard-local-tail | 0.997600 / 0.002400 | 0.995120 / 0.004880 | 0.955560 / 0.044440 | 0.892600 / 0.107400 | 0.815600 / 0.184400 |
| Q-hard-local-tail-oklab | 0.997800 / 0.002200 | 0.995240 / 0.004760 | 0.955680 / 0.044320 | 0.892560 / 0.107440 | 0.816280 / 0.183720 |

## Direction readout

- Hard local-tail local-distortion direction: **not confirmed**. Compared with Q-control, ΔTop25 local PSNR = `-0.000482`, ΔP95 MSE = `+0.000000017`, ΔGini = `-0.000024`.
- OKLab color direction: **positive** versus Hard. ΔCIEDE2000 mean versus Hard = `-0.000464`; ΔCIEDE2000 P95 = `-0.000272`.
- Robustness: maximum absolute BER change versus Q-control is `0.000400` for Hard and `0.000520` for Hard+OKLab. This is a descriptive short-screen result, not a significance test.
- Overall bounded-screen direction: **not stable enough to justify long training**. No parameter grid or retry was run.

## Interpretation and next decision

The screen answers only whether the already-selected loss direction survives a stronger fixed Q backbone under the same custom training bridge. It does not establish official TrustMark training fidelity or a general quality/crop ranking against MBRS, because TrustMark uses a 100-bit BCH-protected message while MBRS uses raw 64-bit BER.

Long training should be considered only if Hard improves the local-tail metrics without a meaningful BER increase and the OKLab arm lowers CIEDE2000 without reversing the quality/robustness direction. If either condition fails here, retain the Q-control/custom objective and do not escalate the backbone experiment.

## Artifacts

- [Complete per-trial CSV](trustmark_q_transfer_screen.csv)
- [Per-arm summary CSV](trustmark_q_transfer_screen_summary.csv)
- [Provenance JSON](trustmark_q_transfer_screen_provenance.json)
- [Runner](../external_baselines/run_trustmark_q_transfer_screen.py)

## Limits

- Two epochs on a 64-image training subset is a bounded screen, not a convergence study.
- Base image/message terms are the transparent RGB-MSE + BCE-logit proxy from the adapter; the missing official `ImageSecretLoss` is not fabricated.
- CIEDE2000 is evaluation-only (CIELAB D65, skimage); OKLab is the only optional color term optimized.
