# Tail-specificity analysis

This analysis uses existing seed17 controlled checkpoints at the formal 32×32 evaluation scale. Values are relative MSE reductions versus Global continuation.

| Method | P10 | P25 | P50 | P75 | P90 | P95 | P99 | Tail-specificity diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hard Patch16 stride16 | +0.5% | +1.7% | +3.6% | +4.9% | +2.8% | +4.8% | +6.3% | +2.0 percentage points |
| Hard Patch16 stride8 | +1.0% | +2.1% | +4.2% | +5.6% | +3.2% | +5.6% | +7.0% | +2.1 percentage points |
| Soft T=0.25 | -0.7% | -0.1% | +2.0% | +3.8% | +1.9% | +3.9% | +4.7% | +2.5 percentage points |
| Multi-scale | +0.1% | +1.0% | +3.0% | +4.3% | +2.4% | +4.3% | +6.1% | +2.3 percentage points |

The current methods do preferentially reduce the upper tail, but the effect is modest. The distribution also shifts globally: P50/P75 improve substantially, so the `+0.233 dB` Worst PSNR gain is not a pure tail-only effect.

The overlap method has the strongest absolute tail reduction and remains the best current method. Its extra tail emphasis over the lower distribution is only about two percentage points, consistent with the small local-specific gain.

Machine-readable percentile values are in [tail_specificity_analysis.csv](tail_specificity_analysis.csv).
