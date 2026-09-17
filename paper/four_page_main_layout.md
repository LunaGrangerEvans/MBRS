# Compact four-page main-paper layout

This is the production set requested for the four-page body: exactly two figures and two tables. The values are frozen reads from the current evidence files; no new training or evaluation is part of this layout.

## Selected assets

- Figure 1, final method framework: [PNG](figures/figure1_final_method.png) · [PDF](figures/figure1_final_method.pdf) · [renderer](../paper/render_fig1_final_method.py)
- Figure 2, final external + internal comparison: [PPTX](figures/figure2_final.pptx) · [PNG](figures/figure2_final.png) · [PDF](figures/figure2_final.pdf) · [renderer](../paper/render_fig2_final.py)
- Table 1, formal final-method ablation: [LaTeX](tables/final_method_ablation.tex)
- Table 2, formal color-fidelity comparison: [LaTeX](tables/final_method_color.tex)
- Render provenance: [JSON](figures/provenance.json)

## Figure 1 — method framework

The main path is `cover x + message m → crop-trained MBRS encoder → watermarked image ŷ`. A secondary crop-mask/decoder branch keeps the crop-trained message objective visible. The three image-supervision cards are:

1. Global RGB MSE: mean distortion over all pixels.
2. Hard Patch16/stride8/Top10 local-tail MSE: 225 overlapping candidates, 23 selected worst patches.
3. Global OKLab: color/lightness residual regularization used by the frozen final Ours method.

The bottom objective makes the scope explicit: message recovery remains part of training, while the paper's image argument is the progression from mean to local-tail to color-aware fidelity. The three colored callouts are the intended visual takeaway: global controls the mean, Hard Local-Tail controls the upper tail, and OKLab controls color artifacts.

**Caption.** *Frozen final-method framework for crop-trained MBRS. The encoder maps a cover image and message to a watermarked image; a random crop and decoder provide the message-recovery objective. Image supervision combines global RGB MSE, overlapping Patch16/stride8 Hard Local-Tail MSE, and global OKLab fidelity. Global controls average distortion, Hard Local-Tail targets the upper tail, and OKLab targets color/lightness residuals. All three fidelity branches are training losses; no additional inference module is introduced.*

## Table 1 — main formal result

Place immediately after the method paragraph that defines the local-tail objective. It contains the three formal project-test rows needed to show the progression from MBRS crop-trained global to Hard Local-Tail to Ours. TrustMark and HiDDeN are intentionally absent from the main body and remain supplementary reference material.

## Figure 2 — qualitative comparison

The figure is a full-width two-sample, five-column panel selected from the fixed validation manifest. Columns are `Original`, `HiDDeN-64 (external)`, `MaskWM-D_64 (external)`, `MBRS crop-trained global`, and `Ours`. Each sample has a full-image row, direct ROI zoom, and an explicitly labelled local-error heatmap row. The Original cell in the error row is marked `—`; the four method heatmaps share one scale per sample. No method-specific ROI, sharpening, saturation, or contrast adjustment is used.

**Caption.** *Qualitative comparison on fixed validation samples. External methods use their respective released/retrained inference protocols and are included as reference baselines rather than strictly matched training comparisons. Red boxes indicate shared ROIs. Local-error heatmaps show mean absolute RGB difference inside the shared ROI, amplified ×10 for visibility with one shared scale per sample; the Original column is not applicable. Ours denotes the proposed Hard Local-Tail model with OKLab color-aware regularization.*

## Table 2 — final color fidelity

Place after the qualitative discussion, as the final result object before the conclusion. The formal color table compares Hard Local-Tail with Ours and records the color-fidelity gain alongside PSNR, local PSNR, LPIPS, and BER30. Gini remains a reported secondary diagnostic.

## Four-page order

1. Introduction and problem setup; end with the one-paragraph contribution statement.
2. Method section with Figure 1; follow with Table 1 and the formal-test result paragraph.
3. Qualitative result paragraph with final Figure 2; discuss external references as contextual and local-error heatmaps as visualization-only.
4. Color-fidelity subsection with Table 2; short limitations/conclusion. Put TrustMark/HiDDeN comparison and all other ablations in supplementary material.

The body therefore follows `method figure → main table → qualitative figure → color table`, with external baselines retained as reference context rather than a formal main-table claim.
