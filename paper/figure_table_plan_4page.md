# Four-page paper figure/table freeze

The requested main-paper set is now frozen to exactly **two figures and two tables**. The production details, captions, and page order live in [four_page_main_layout.md](four_page_main_layout.md).

## Figure 1 — method framework

- [PNG](figures/fig01_method_framework.png)
- [PDF](figures/fig01_method_framework.pdf)
- [Source](../experiments/render_four_page_main_assets.py)

The diagram shows `cover + message → crop-trained MBRS encoder → watermarked image`, a secondary crop/decoder message path, and the three image branches: Global RGB MSE, Hard Patch16/stride8/Top10 local-tail MSE, and optional OKLab color regularization. The visual callouts explicitly state: Global controls the mean; Hard Top10 controls the worst local tail; OKLab controls color artifacts.

## Table 1 — main formal result

- [LaTeX](table1_main_results.tex)
- Source values: [main_table.csv](main_table.csv)

The compact table keeps only Global continuation, Hard P16/S16/Top25, Hard P16/S8/Top25, and Hard P16/S8/Top10 (Ours). Columns are PSNR, SSIM, LPIPS, Top25 local PSNR, P95 MSE, Gini, and BER30.

## Figure 2 — qualitative comparison

- [PNG](figures/fig02_qualitative_comparison.png)
- [PDF](figures/fig02_qualitative_comparison.pdf)
- [Source](../experiments/render_four_page_main_assets.py)

Five fixed validation samples (`0, 10, 20, 30, 40`) are arranged as rows. The four method groups are Original, Global baseline, Hard Top10, and Hard Top10 + global OKLab (`λ=0.059149764`). Each group has a full image and the same row-level 32×32 zoom; the ROI is selected once from the incumbent Hard Top10 CIEDE2000 tail and shared across methods.

## Table 2 — color extension

- [LaTeX](table2_color_extension.tex)
- Sources: [crop_global_oklab_validation_candidates.csv](../reports/crop_global_oklab_validation_candidates.csv) and [local_chroma_tail_summary.csv](../reports/local_chroma_tail_summary.csv)

This is explicitly validation-only. It contains Hard Top10, + Global OKLab (`λ=0.059149764`), and + Local chroma-tail, with PSNR, local PSNR, global/Top10 CIEDE2000, Gini, and BER30.

## Four-page order

1. Introduction and setup.
2. Method + Figure 1, then Table 1 and the formal result paragraph.
3. Qualitative result paragraph + Figure 2.
4. Color extension + Table 2, limitations, and conclusion.

TrustMark/HiDDeN comparison, the full eight-row ablation, and detailed provenance remain supplementary. This replaces the previous five-figure main-body proposal; older freeze assets remain available for audit but are not part of this four-page set.
