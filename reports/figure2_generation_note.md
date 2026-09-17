# Figure 2 generation note

Figure 2 uses only the fixed 50-image validation manifest (SHA-256 `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`). It is a qualitative illustration, not a formal quantitative benchmark panel.

Selected samples and shared ROIs:

- **Sample 1:** `0829` (validation index `28`), ROI `(68,52,32,32)`. Selected by the direct MBRS crop-trained global → Ours ROI improvement ranking, with P95 error-tail, color-error, and textured-content tie-break terms.
- **Sample 2:** `0803` (validation index `2`), ROI `(44,24,32,32)`. Selected by the direct MBRS crop-trained global → Ours ROI improvement ranking, with P95 error-tail, color-error, and textured-content tie-break terms.

The five columns are Original, HiDDeN-64, MaskWM-D_64, MBRS crop-trained global, and Ours. Ours is the final Hard Local-Tail + OKLab output. Each sample uses one shared ROI across all methods; Local error heatmaps show the corresponding displayed watermarked image's mean absolute RGB difference from the displayed original inside that ROI, amplified ×10 for visibility. The Original column is marked — because its error is not applicable.

Variant recommendation: **`figure2_final_v2.png` / `figure2_final_v2.pdf`**; `figure2_final.png` / `figure2_final.pdf` use the same v2 layout. V2 gives the two sample groups a clearer separation while preserving aligned, borderless tiles and the exact requested row labels. V1 is retained as the compact alternative.

Generated files:

- `visualizations/paper/figure2_final.png`
- `visualizations/paper/figure2_final.pdf`
- `visualizations/paper/figure2_final_v1.png`
- `visualizations/paper/figure2_final_v1.pdf`
- `visualizations/paper/figure2_final_v2.png`
- `visualizations/paper/figure2_final_v2.pdf`
- `visualizations/paper/figure2_manifest.json`
