# Final Figure 2 selection

The final five-column Figure 2 uses only the fixed 50-image validation manifest. Project-test images are not used for qualitative sample or ROI selection.

- Manifest: `/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/validation_manifest.pt`; SHA-256 `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`.
- Candidate criterion: balanced score of local PSNR benefit (35%), P95 local-tail reduction (25%), visible color-error reduction by ROI CIEDE2000 (20%), residual diagnostic clarity (10%), and content texture/diversity (10%).
- ROIs are selected from the Hard Local-Tail → Ours comparison on native 128×128 clipped RGB outputs, then shared across all five columns within each sample.
- The selection is illustrative rather than representative; the balanced score avoids selecting only the two largest numeric gains.

| Figure sample | Image ID | Validation index | ROI `(x,y,w,h)` | Δ local PSNR | P95 reduction | ROI CIEDE2000 reduction | ROI chroma reduction | ROI residual-energy reduction | Rationale |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `0814` | 13 | `(88,48,32,32)` | +0.723 dB | +2.835e-05 | +0.476 | +0.490 | +9.400e-04 | balanced local-tail/color/residual criterion with content diversity |
| 2 | `0839` | 38 | `(88,16,32,32)` | +1.097 dB | +2.068e-05 | +0.216 | +0.182 | +9.187e-04 | balanced local-tail/color/residual criterion with content diversity |

## Reader-facing boundary

The final figure has exactly five columns: Original, HiDDeN-64 (external), MaskWM-D_64 (external), MBRS crop-trained global, and Ours. Ours is the frozen Hard Local-Tail + global OKLab method. Hard Local-Tail is kept for the quantitative ablation chain, not as a sixth final-figure column.

Caption text used in the figure: “Qualitative comparison on fixed validation samples. External methods use their respective released/retrained inference protocols and are included as reference baselines rather than strictly matched training comparisons. Red boxes indicate shared ROIs. Local-error heatmaps show mean absolute RGB difference inside the shared ROI, amplified ×10 for visibility with one shared scale per sample; the Original column is not applicable. Ours denotes the proposed Hard Local-Tail model with OKLab color-aware regularization.”
