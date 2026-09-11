# Paper figure and table plan

## Figures

| Figure | Scientific message | Methods | Metrics/data | Existing status | Missing |
|---|---|---|---|---|---|
| Fig.1 Motivation | Crop robustness raises visual distortion and creates a measurable local tail | no-crop Global, crop-trained Global | cover, encoded image, residual, 32×32 heatmap, BER curve | partial figures exist | final aligned examples and fixed captions |
| Fig.2 Method | Global image loss leaves local tail implicit; Hard overlap adds explicit local selection | Global, Hard stride16, Hard stride8 | framework diagrams and loss equations | framework SVGs exist | update GLX figures with final controlled labels |
| Fig.3 Distribution | Methods change the right tail differently | Global, Hard stride16, overlap, Soft, multi-scale, Excess | patch-MSE histogram/CDF | controlled figures exist | consolidate all branches on one scale |
| Fig.4 Percentile curve | Best method is tail-biased or merely shifts all pixels | same controlled branches | P50–P99 relative MSE reduction | missing | generate from scale-analysis arrays |
| Fig.5 Crop robustness | Tail control does not damage message recovery | same controlled branches | BER@30/35/40/50/70/100 | existing controlled tables | final plot with explicit configuration names |
| Fig.6 Qualitative residuals | Improvement is spatially visible and not cherry-picked | Global, Hard stride16, overlap, best candidate | shared residual scale, aligned image index plus several fixed examples | figures exist per branch | choose examples by predeclared rule |
| Fig.7 Selected/excess maps | Show coverage and redundancy of selected regions | Hard stride16 vs overlap | selected patch map, union mask, multiplicity map, components | missing | generate analysis-only maps |

## Tables

| Table | Content | Required status |
|---|---|---|
| Table 1 | Current controlled master comparison | available in `current_controlled_master_table.*` |
| Table 2 | Patch size/stride/loss ablation | available, but historical rows must be labeled exploratory |
| Table 3 | Local SSIM/LPIPS and MSE tail metrics | available for controlled branches |
| Table 4 | Crop robustness curve | available for controlled branches |
| Table 5 | Experiment protocol and reproducibility controls | missing; should list exact source/checkpoint/device/manifest |

## Minimum paper package

The minimum credible paper needs Fig.1–5, Tables 1–3, explicit controlled protocol, and a limitation paragraph acknowledging seed17-only evaluation and modest effect size.
