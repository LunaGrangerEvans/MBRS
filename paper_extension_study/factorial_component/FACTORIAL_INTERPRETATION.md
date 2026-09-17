# Factorial Component Interpretation

This interpretation uses only the four fixed-RGB-weight seed17 cells audited in `PROTOCOL_AUDIT.md`. `M00` is RGB0.5-Control, `M01` is Global+OKLab-only, `M10` is Hard Local-Tail, and `M11` is Ours. The frozen Global row with RGB weight 1.0 is not part of these pure component contrasts.

## 1. OKLab effect with Tail off

The pure fixed-weight comparison is **Global+OKLab-only − RGB0.5-Control**, not Global+OKLab-only versus the frozen Global row. Point effects are:

- PSNR `+0.779357 dB`;
- Top-25 Local PSNR `+0.759839 dB`;
- P95 `-5.852545e-05`, P99 `-6.036370e-05`;
- LPIPS `-0.000252740`;
- Global CIEDE2000 `-0.449969`, Top10 CIEDE2000 `-0.643096`;
- Gini `+0.005048`;
- BER30 `+0.0001875`.

The factorial bootstrap intervals exclude zero for the listed fidelity/color metrics and for the Gini increase; the BER30 interval includes zero. Thus OKLab has an independent measured effect at fixed RGB weight when Tail is off, but its Gini direction is unfavorable and no BER direction is established.

## 2. OKLab effect with Tail on

The comparison **Ours − Hard Local-Tail** gives:

- PSNR `+0.412609 dB`;
- Top-25 Local PSNR `+0.387519 dB`;
- P95 `-2.246259e-05`, P99 `-2.284046e-05`;
- LPIPS `-0.000298064`;
- Global CIEDE2000 `-0.230699`, Top10 CIEDE2000 `-0.303437`;
- Gini `+0.005233`;
- BER30 `+0.0001875`.

Intervals exclude zero for the fidelity/color metrics and for the Gini increase, while BER30 includes zero. OKLab retains a favorable measured refinement when Tail is present, but its standalone effect is smaller than with Tail off on PSNR, local PSNR, P95, and P99.

## 3. Tail effect with OKLab off

The comparison **Hard Local-Tail − RGB0.5-Control** gives:

- PSNR `+1.277815 dB`;
- Top-25 Local PSNR `+1.306655 dB`;
- P95 `-9.685666e-05`, P99 `-1.015365e-04`;
- LPIPS `-0.000403106`;
- Global CIEDE2000 `-0.542383`, Top10 CIEDE2000 `-0.782986`;
- Gini `-0.001419`;
- BER30 `-0.0001250`.

Intervals exclude zero for PSNR, local PSNR, P95, P99, LPIPS, and CIEDE2000. The Gini and BER30 intervals include zero. Tail has the strongest fixed-weight point effect on the local-tail metrics.

## 4. Tail effect with OKLab on

The comparison **Ours − Global+OKLab-only** gives:

- PSNR `+0.911067 dB`;
- Top-25 Local PSNR `+0.934335 dB`;
- P95 `-6.079381e-05`, P99 `-6.401330e-05`;
- LPIPS `-0.000448430`;
- Global CIEDE2000 `-0.323113`, Top10 CIEDE2000 `-0.443327`;
- Gini `-0.001233`;
- BER30 `-0.0001250`.

Intervals exclude zero for PSNR, local PSNR, P95, P99, LPIPS, and CIEDE2000. Gini and BER30 intervals include zero. Tail remains the larger local-tail effect when OKLab is on.

## 5. Component strongest on local-tail metrics

Hard Local-Tail is the stronger component for local-tail metrics at this fixed RGB weight. Its Tail-off effect is `+1.306655 dB` in Top-25 Local PSNR versus `+0.759839 dB` for OKLab-off Tail-off, and its P95/P99 reductions are larger in magnitude. This directly supports describing Tail as explicit upper-tail local-distortion control.

## 6. Component strongest on color/perceptual metrics

No single component dominates every color/perceptual metric, although Tail has the larger point effect on LPIPS, Global CIEDE2000, and Top10 CIEDE2000 in both contexts. OKLab still contributes a measured Top10 CIEDE2000 reduction without Tail (`-0.643096`) and with Tail (`-0.303437`). The safe conclusion is that both terms contribute measured fidelity/color changes, with Tail driving much of the local and extreme-color-tail movement; do not claim an isolated OKLab win over frozen Global.

## 7. Gini behavior

OKLab increases Gini in both fixed-weight contexts (`+0.005048` without Tail and `+0.005233` with Tail), with bootstrap intervals excluding zero. Tail point effects reduce Gini slightly, but both Tail Gini intervals include zero. Gini is therefore a diagnostic, not a primary objective or a universal concentration claim.

## 8. BER30 behavior

All five factorial BER30 effect intervals include zero. Point effects are small and mixed: OKLab is `+0.0001875` in both contexts, while Tail is `-0.0001250` in both contexts. These results do not establish a directional BER30 effect, BER equivalence, or stronger crop robustness.

## 9. Additive, diminishing, or mixed effects

The interaction contrast is:

```text
M11 - M10 - M01 + M00
```

Key interaction results:

| Metric | Interaction | 95% interval | Reading |
|---|---:|---:|---|
| PSNR | `-0.366748 dB` | `[-0.380423, -0.351979]` | non-additive diminishing returns |
| Top-25 Local PSNR | `-0.372320 dB` | `[-0.390011, -0.353527]` | non-additive diminishing returns |
| P95 MSE | `+3.606286e-05` | `[+3.413000e-05, +3.828961e-05]` | diminishing favorable reduction; lower is better |
| P99 MSE | `+3.752325e-05` | `[+3.526294e-05, +3.996619e-05]` | diminishing favorable reduction; lower is better |
| LPIPS | `-4.532402e-05` | `[-1.119358e-04, +1.866620e-05]` | interval includes zero; uncertain |
| Global CIEDE2000 | `+0.219270` | `[+0.206887, +0.231136]` | diminishing favorable reduction; lower is better |
| Top10 CIEDE2000 | `+0.339659` | `[+0.322037, +0.355784]` | diminishing favorable reduction; lower is better |
| Gini | `+0.000186` | `[-0.000648, +0.001019]` | interval includes zero |
| BER30 | approximately `0` | `[-0.000813, +0.000813]` | interval includes zero |

Thus the combined effects are non-additive and mostly show diminishing returns on fidelity/color metrics, with LPIPS, Gini, and BER30 mixed or uncertain. Positive/negative interaction signs are not intrinsically good or bad; metric direction matters.

## 10. Preferred wording

**“Complementary effects” remains the preferred wording, with qualification.** Tail provides the clearest explicit local-tail contribution, while OKLab contributes an additional measured color/perceptual refinement in the Tail-on comparison. The interaction analysis does not justify “synergy”; several contrasts show diminishing returns and some diagnostics are mixed.

The comparison Global+OKLab-only versus frozen Global remains confounded by the frozen Global RGB weight of 1.0 and must not be called a pure OKLab effect.
