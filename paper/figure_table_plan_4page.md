# Four-page paper figure/table freeze

The requested main-paper set is now frozen to exactly **two figures and two tables**. The production details, captions, and page order live in [four_page_main_layout.md](four_page_main_layout.md).

## Figure 1 — method framework

- [PNG](figures/figure1_final_method.png)
- [PDF](figures/figure1_final_method.pdf)
- [Source](../paper/render_fig1_final_method.py)

The diagram shows `cover + message → crop-trained MBRS encoder → watermarked image`, a secondary crop/decoder message path, and the three image branches: Global RGB average distortion, Hard Local-Tail upper-tail distortion, and Ours' OKLab color-aware refinement. The visual callouts explicitly state: Global controls the mean; Hard Local-Tail controls the upper tail; OKLab controls color-aware fidelity.

## Table 1 — main formal result

- [LaTeX](tables/final_method_ablation.tex)
- Source values: [final_ours_project_test_report.md](../reports/final_ours_project_test_report.md)

The compact table contains the three-row progression MBRS crop-trained global → Hard Local-Tail → Ours. Columns are PSNR, SSIM, LPIPS, Top25 local PSNR, P95 MSE, Gini, and BER30.

## Figure 2 — qualitative comparison

- [Editable PPTX](figures/figure2_final.pptx)
- [PNG](figures/figure2_final.png)
- [PDF](figures/figure2_final.pdf)
- [Source](../paper/render_fig2_final.py)

Two samples are selected from all 50 validation images using a balanced local-tail, P95, color-error, residual-clarity, and content-diversity criterion. Columns are Original, HiDDeN-64 (external), MaskWM-D_64 (external), MBRS crop-trained global, and Ours. Full, direct ROI zoom, and residual ×10 rows are shown. Ours is the frozen Hard Local-Tail + global OKLab method.

## Table 2 — final color fidelity

- [LaTeX](tables/final_method_color.tex)
- Source values: [final_ours_project_test_report.md](../reports/final_ours_project_test_report.md)

The final color table contains the formal Hard Local-Tail and Ours rows, with PSNR, local PSNR, global/Top10 CIEDE2000, LPIPS, and BER30. The prior validation-only color-extension table remains supplementary historical evidence.

## Four-page order

1. Introduction and setup.
2. Method + Figure 1, then Table 1 and the formal result paragraph.
3. Qualitative result paragraph + Figure 2.
4. Color extension + Table 2, limitations, and conclusion.

TrustMark/HiDDeN comparison, the full eight-row ablation, and detailed provenance remain supplementary. This replaces the previous five-figure main-body proposal; older freeze assets remain available for audit but are not part of this four-page set.
