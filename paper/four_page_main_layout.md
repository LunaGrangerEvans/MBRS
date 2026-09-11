# Compact four-page main-paper layout

This is the production set requested for the four-page body: exactly two figures and two tables. The values are frozen reads from the current evidence files; no new training or evaluation is part of this layout.

## Selected assets

- Figure 1, method framework: [PNG](figures/fig01_method_framework.png) · [PDF](figures/fig01_method_framework.pdf) · [renderer](../experiments/render_four_page_main_assets.py)
- Figure 2, five fixed samples with shared local zoom: [PNG](figures/fig02_qualitative_comparison.png) · [PDF](figures/fig02_qualitative_comparison.pdf) · [renderer](../experiments/render_four_page_main_assets.py)
- Table 1, formal main result: [LaTeX](table1_main_results.tex)
- Table 2, validation-only color extension: [LaTeX](table2_color_extension.tex)
- Render provenance: [JSON](figures/provenance.json)

## Figure 1 — method framework

The main path is `cover x + message m → crop-trained MBRS encoder → watermarked image ŷ`. A secondary crop-mask/decoder branch keeps the crop-trained message objective visible. The three image-supervision cards are:

1. Global RGB MSE: mean distortion over all pixels.
2. Hard Patch16/stride8/Top10 local-tail MSE: 225 overlapping candidates, 23 selected worst patches.
3. Optional global OKLab: color/lightness residual regularization, marked as a validation extension.

The bottom objective makes the scope explicit: message recovery remains part of training, while the paper's image argument is the complementarity of mean, local tail, and color branches. The three colored callouts are the intended visual takeaway: global controls the mean, Hard Top10 controls the worst local tail, and OKLab controls color artifacts.

**Caption.** *Method framework for crop-trained MBRS. The encoder maps a cover image and message to a watermarked image; a random crop and decoder provide the message-recovery objective. Image supervision compares the watermarked image with the cover using global RGB MSE, overlapping Patch16/stride8 hard Top10 local-tail MSE, and an optional global OKLab branch. Global controls average distortion, Hard Top10 targets the worst local regions, and OKLab targets color/lightness residuals. The OKLab branch is a validation-only extension in this study.*

## Table 1 — main formal result

Place immediately after the method paragraph that defines the local-tail objective. It contains only the four formal-test rows needed to show the progression from Global continuation to the proposed overlap/ratio setting. TrustMark and HiDDeN are intentionally absent from the main body and remain supplementary reference material.

## Figure 2 — qualitative comparison

The figure is a full-width 5×8 panel: five predetermined validation indices `0, 10, 20, 30, 40`; each method has a full-frame panel and a shared 32×32 zoom. The four method groups are `Original`, `Global baseline`, `Hard Top10`, and `Hard Top10 + global OKLab (λ=0.059149764)`. The red rectangle indicates the same row-level ROI in all four columns; it is selected once from the incumbent Hard Top10 CIEDE2000 tail. No per-method crop selection, residual amplification, sharpening, or contrast normalization is used.

**Caption.** *Qualitative RGB comparison on five fixed validation samples. Each row uses the same image and message. For each row, the right-hand panel of every method shows the same native 32×32 crop, selected from the incumbent Hard Top10 CIEDE2000 tail. The OKLab result is validation-only and uses λ=0.059149764. The examples illustrate local color-block and residual differences; they are not an observer study or a claim of uniform perceptual superiority.*

## Table 2 — color extension

Place after the qualitative discussion, as the final result object before the conclusion. Label it explicitly `validation-only`; the strong OKLab row is included because it matches the requested color-error contrast, while the local chroma-tail row shows the more conservative concentration trade-off. The table's message is intentionally narrow: global OKLab improves CIEDE2000 most, and local chroma-tail keeps Gini closer to the incumbent Hard Top10 value.

## Four-page order

1. Introduction and problem setup; end with the one-paragraph contribution statement.
2. Method section with Figure 1; follow with Table 1 and the formal-test result paragraph.
3. Qualitative result paragraph with Figure 2; discuss that color blocks are illustrative and validation-only for OKLab.
4. Color extension subsection with Table 2; short limitations/conclusion. Put TrustMark/HiDDeN comparison and all other ablations in supplementary material.

The body therefore follows `method figure → main table → qualitative figure → color table`, as requested, without spending a full table on external baselines.
