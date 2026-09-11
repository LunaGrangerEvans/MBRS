# Validation-driven single correction

The first candidate uses lambda0.059149764, fixed by training-only gradient calibration. Its epoch20 evaluation on the fixed validation manifest gives, relative to incumbent Hard Top10:

- Global PSNR +0.407537dB; Top25 native32 local PSNR +0.379264dB.
- Global CIEDE2000 3.598625→3.365012; Top10 color error5.230329→4.920614. All50 images improve Top10 color error.
- BER50 −0.0000625, BER40 +0.0002500, BER30 −0.0001875; within the predeclared preservation tolerances.
- Gini0.094569→0.100017 (+5.76%), exceeding the predeclared2% tolerance.

The candidate lowers absolute error but increases its relative spatial inequality. The additional color term affects regions differently; this explanation is consistent with the measurements, not a proved causal mechanism. Its source-point color/image gradient cosine is about0.4, so the objectives are only partly aligned even at initialization.

Apply the one predeclared correction: lambda/2=0.029574882. This is a fresh20-epoch continuation from the same epoch100 source with identical optimizer initialization, not a resume from the first candidate. All other training variables stay fixed. Do not inspect formal-test candidate metrics or make any further parameter changes to decide this retry.

The existing Hard Top10 remains the manuscript method unless the new candidate meets the complete decision rule. A failure to meet the5% color target or2% concentration tolerance will be reported rather than hidden.

## Completed half-weight result

The second continuation completed20 epochs. At lambda0.029574882, PSNR increases0.239585dB and Top25 local PSNR increases0.226884dB relative to incumbent. Global CIEDE2000 drops3.70%, Top10 color error drops3.42%, and all50 validation images improve both color metrics and P95 native32 MSE. BER100/70/50 stays identical; BER40 changes+0.0001875 and BER30−0.0001875.

Gini increases2.89% to0.097303. Halving the color weight reduces the Gini regression, but also reduces color benefit. Both candidates miss the strict complete gate: the strong candidate fails Gini preservation; the half-weight candidate fails the2% Gini tolerance and the5% color-benefit target. No extra grid or third run was launched. Report the half-weight model as the more conservative diagnostic comparison, retain incumbent as paper main method, and keep candidate formal-test evaluation unopened under this protocol.

This is not a JPEG-like robustness collapse: BER is preserved within the declared tolerances. The remaining trade-off is between absolute quality/color fidelity and relative concentration. Whether the modest Gini increase is acceptable is an application preference, not something to hide by changing the success thresholds after measurement.
