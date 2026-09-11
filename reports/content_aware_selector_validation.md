# Content-aware selector validation

This is an analysis-only validation on the seed17 validation split. No formal test images were used to choose the selector, and no model was retrained.

All selectors use Patch16, stride8, Top10%, with per-image robust P5–P95 normalization of original-image content features to approximately [0,1].

## Selector definitions

- MSE: `S_mse = MSE`
- Gradient-aware: `S_grad = MSE / (1 + alpha * G_norm)`
- Gradient+HF: `A = 0.5 G_norm + 0.5 H_norm`; `S_gh = MSE / (1 + alpha A)`

LPIPS alignment uses the same 16×16 crops resized to 32×32 before AlexNet-LPIPS evaluation. This is an alignment proxy because the LPIPS network cannot directly process 16×16 crops.

## Alignment and content-bias results

### Top10 selection

| Selector | LPIPS Spearman | SSIM-degradation Spearman | MSE/LPIPS Jaccard | MSE/SSIM Jaccard | Gradient correlation | HF correlation | Unique coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSE | 0.151 | 0.109 | 0.112 | 0.106 | +0.207 | +0.090 | 19.4% |
| Gradient, α=0.5 | 0.324 | 0.506 | 0.155 | 0.238 | -0.191 | -0.212 | 20.2% |
| Gradient, α=1.0 | 0.375 | 0.666 | 0.166 | 0.312 | -0.403 | -0.368 | 20.0% |
| **Gradient, α=2.0** | **0.387** | **0.750** | **0.168** | **0.350** | -0.572 | -0.487 | 19.9% |
| Gradient+HF, α=0.5 | 0.296 | 0.451 | 0.151 | 0.204 | -0.148 | -0.229 | 20.1% |
| Gradient+HF, α=1.0 | 0.339 | 0.600 | 0.157 | 0.265 | -0.345 | -0.399 | 20.1% |
| Gradient+HF, α=2.0 | 0.347 | 0.681 | 0.162 | 0.301 | -0.511 | -0.535 | 19.8% |

### Top25 selection

The same ranking holds at Top25. Gradient α=2.0 reaches approximately `0.286` MSE/LPIPS Jaccard and `0.5108` MSE/SSIM-degradation Jaccard, versus MSE `0.205` and `0.206`.

## Selection-collapse check

Gradient α=2.0 does not collapse spatially: it still selects 23 patches and covers about 20% of the image. It does, however, strongly shift content selection toward smooth regions:

- selected normalized gradient mean: `0.105` vs all-patch mean `0.461`;
- selected normalized HF mean: `0.146` vs all-patch mean `0.428`.

This is not a spatial collapse, but it is a strong content prior. It may be desirable if smooth-region artifacts are perceptually more visible, or harmful if it simply suppresses legitimate texture distortion. That must be checked after training, not assumed from alignment alone.

## Decision

**BEST SELECTOR:** Gradient-only normalized selector

**BEST ALPHA:** `2.0`

**MSE→LPIPS ALIGNMENT:** `0.151 → 0.387` Spearman; Top10 Jaccard `0.112 → 0.168`

**MSE→SSIM ALIGNMENT:** `0.109 → 0.750` Spearman; Top10 Jaccard `0.106 → 0.350`

**CONTENT BIAS:** Original MSE selects above-average texture/gradient regions; Gradient α=2 reverses this bias and strongly favors smooth regions.

**TRAIN ONE CONTENT-AWARE MODEL?** **YES, one controlled validation run is justified; do not run a grid.**

**WHY:** Gradient-only α=2 improves both perceptual rank alignment measures and both tail-overlap measures, while Gradient+HF is consistently weaker. The spatial coverage remains non-degenerate, but the smooth-region bias must be tested for actual visual benefit.

## Suggested single training configuration

`seed17 + Patch16 + stride8 + Top10 + Gradient-aware selector alpha2 + global/local 0.5/0.5`

All other variables must remain identical to the current seed17 Top10 controlled protocol. This report does not authorize automatic training; the next run should be started only as this one isolated configuration.

Qualitative outputs are under `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/content_aware_selector_validation/`. Machine-readable values are in [content_aware_selector_validation.csv](content_aware_selector_validation.csv).
