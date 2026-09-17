# Figure 2 generation notes

## Figure design

- Layout: upper qualitative section plus three lower quantitative panels.
- Final size: `7.16 in × 6.55 in`.
- PNG: `2506 × 2292 px` at `350` DPI.
- PDF/SVG retain vector text and quantitative curves; qualitative image tiles are embedded raster evidence.

## Upper qualitative section

- Selection source: `reports/final_figure2_selection.md`.
- Fixed validation sample: ID `0814`, index `13`.
- Shared source-space ROI: `(x=88, y=48, width=32, height=32)` on 128×128 images.
- Columns: Original, HiDDeN-64, MaskWM-D_64, MBRS crop-trained Global, Ours.
- Mean display PSNR on the common 512×512 canvas: HiDDeN-64 `30.140571` dB; MaskWM-D_64 `38.988421` dB; Global `40.160126` dB; Ours `40.666174` dB.
- ROI mean absolute RGB error: HiDDeN-64 `0.019601174`; MaskWM-D_64 `0.006362271`; Global `0.007322781`; Ours `0.006436666`.
- Local-error maps use one linear color range across all method columns and are amplified ×10 only for visualization.
- External methods are contextual references under different training/preprocessing protocols; the panel does not claim strict external superiority.

## Lower quantitative section

- Evidence space: natural frozen 128×128 project-test set, exactly 50 paired image indices.
- Global source: `paper_extension_study/continuation_seed/global/seed17/project_test_per_image.csv`.
- Ours source: `paper_extension_study/continuation_seed/ours/seed17/project_test_per_image.csv`.
- P95 paired scatter: `50/50` points below identity (lower is better).
- Top-25 local-PSNR paired scatter: `50/50` points above identity (higher is better).
- Local-tail profile: each image contributes sixteen non-overlapping 32×32 patch MSE values; values are sorted within image and averaged by rank over 50 images.
- Worst-quartile mean patch MSE: Global `2.918315877e-04`; Ours `2.538156114e-04`; relative reduction `13.027%`.
- The worst-quartile band spans the 75th–100th percentile. No smoothing, regression, pooling across images before ranking, or sample filtering is used.

## Safeguards

- No model was retrained.
- No checkpoint, alpha, hyperparameter, image, or ROI was newly selected from project-test results.
- The Prism manuscript and `paper_bundle/` were not modified.
