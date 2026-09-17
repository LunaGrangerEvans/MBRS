# Bootstrap Interpretation Check

This summary uses the corrected paper-estimator PSNR bootstrap and the unchanged original bootstrap outputs for all other metrics. Differences are `Ours − MBRS crop-trained global`; positive favors Ours for PSNR/local PSNR, while negative favors Ours for lower-is-better metrics.

| Metric | Observed difference | 95% paired bootstrap interval | Interval includes zero? |
|---|---:|---:|:---:|
| PSNR | +0.591736509 dB | [+0.552852365, +0.635779193] dB | No |
| Top-25 Local PSNR | +0.619682010 dB | [+0.579132280, +0.664227127] dB | No |
| P95 patch MSE | -3.966063807e-05 | [-4.334515075e-05, -3.620615269e-05] | No |
| P99 patch MSE | -4.172629507e-05 | [-4.586718574e-05, -3.781528063e-05] | No |
| LPIPS | -0.000432485252 | [-0.000557054864, -0.000341752803] | No |
| Global CIEDE2000 | -0.281766059 | [-0.305251388, -0.261962791] | No |
| Top10 CIEDE2000 | -0.386875057 | [-0.415715378, -0.361141439] | No |
| Gini | -0.002876527 | [-0.004419131, -0.001287979] | No |
| BER30 | approximately 0 | [-0.000500000, +0.000562500] | Yes |

The paired bootstrap interval excludes zero for every listed fidelity/tail metric and includes zero for BER30. This supports the cautious sentence: **“Fidelity improvements are stable under paired image-level resampling, whereas no directional BER improvement is established.”** The BER interval containing zero is not evidence of equivalence, and no improved crop-robustness claim is made.

The corrected PSNR result is generated in `bootstrap_psnr_paper_estimator_summary.md`; the full unchanged non-PSNR bootstrap table is in `bootstrap_results.csv` and `bootstrap_summary.md`.
