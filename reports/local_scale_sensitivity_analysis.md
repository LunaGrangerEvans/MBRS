# Local-scale sensitivity analysis

This is an analysis-only re-evaluation of existing seed17 controlled checkpoints. Training was not changed. All scales use non-overlapping evaluation patches and the same fixed manifest.

## Worst-PSNR improvement relative to Global continuation

| Method | 8×8 patches | 16×16 patches | 32×32 patches | 64×64 patches |
|---|---:|---:|---:|---:|
| Hard Patch16 stride16 | +0.218 | +0.218 | +0.203 | +0.194 |
| Soft T=0.25 | +0.151 | +0.150 | +0.136 | +0.128 |
| Hard Patch16 stride8 | **+0.248** | **+0.248** | **+0.233** | **+0.221** |
| Multi-scale | +0.189 | +0.189 | +0.175 | +0.167 |
| Excess | -0.257 | -0.263 | -0.285 | -0.306 |

## Interpretation

The overlap branch is consistently strongest at every analysis scale. Its gain is slightly larger at 8×8/16×16 than at the formal 32×32 evaluation scale, which indicates a small training/evaluation scale mismatch. However, the difference is only about `0.015 dB`, so this mismatch does not explain the entire small effect size.

The percentile distributions support the same conclusion: the overlap branch reduces P95 patch MSE at 16×16 by about `0.000086` and at 32×32 by about `0.000090` relative to the Global continuation. The method is not secretly producing a large hidden gain at a smaller scale.

## Decision

Keep 32×32 as the primary cross-method metric for comparability with the existing study, but report 16×16 analysis as a diagnostic sensitivity result. Do not claim that evaluation mismatch is the main bottleneck.
