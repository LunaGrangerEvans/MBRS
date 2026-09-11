# Color-bias Go/No-Go audit

Analysis-only audit of the frozen Global continuation and Hard Patch16/stride8/Top10 outputs on all 50 fixed formal test images. No training was run and no image was selected based on its result.

Protocol: RGB tensors clipped to [0,1], CIEDE2000 computed from CIELAB, and 5×5 sliding patches with stride 1. Top10% means the highest `ceil(15376×0.10)=1538` patch scores per image.

## Aggregate results

| Method | Global CIEDE2000 mean | Patch CIEDE2000 mean | Patch CIEDE2000 P95 | Top10 patch CIEDE2000 | Max patch CIEDE2000 | Mean signed Δa | Mean signed Δb |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global continuation | 3.535857 | 3.586801 | 5.056442 | 5.159104 | 7.013435 | -0.260175 | 0.366304 |
| Hard Patch16 stride8 Top10 | 3.484790 | 3.533332 | 4.974994 | 5.075666 | 6.930025 | -0.325777 | 0.396867 |

## Hard Top10 versus Global

- Top10 CIEDE2000 change: `-0.083438`.
- P95 patch CIEDE2000 change: `-0.081447`.
- Global mean CIEDE2000 change: `-0.051067`.
- The audit is a color-distortion diagnostic, not a claim that CIEDE2000 is the training objective.

## Go/No-Go decision

**GO for one controlled JPEG + OKLab-tail continuation.** The frozen outputs contain non-zero global and local CIEDE2000 tails, so a color-tail intervention is measurable. This Go decision does not assume that the color-tail loss will improve the tail after training.

## Qualitative outputs

Fixed predetermined images 07, 20, and 42 are rendered under `visualizations/color_bias_audit/`. Each panel includes Original, both watermarked images, CIEDE2000 heatmaps, and the automatically selected worst 5×5 color patch zoom.

Machine-readable per-image values: [color_bias_audit_per_image.csv](color_bias_audit_per_image.csv).
