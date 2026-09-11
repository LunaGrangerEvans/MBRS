# Bottleneck priority matrix

| Category | Evidence | Expected upside | Implementation risk | Paper value | GPU cost | Assessment |
|---|---|---|---|---|---|---|
| Patch localization | Medium | Medium | Medium | Medium | Low/medium | Overlap helps only 0.030 dB; boundary issue exists but is not primary. |
| Tail selection | Medium | Medium | Low/medium | High | Low/medium | Overlap coverage is 38.9%; Top25 may be too broad. |
| Local metric | Strong | High | Medium | High | Low for diagnostics, medium for training | MSE rank has near-zero/negative alignment with LPIPS and SSIM degradation. |
| Loss weighting | Medium | Low/medium | Low | Medium | Low | W0.75 does not help; W0.5 is already adequate. |
| Training schedule | Weak/medium | Low/medium | Low | Medium | Medium | Controlled curves show no decisive early shock requiring warm-up. |
| Watermark architecture/backbone | Weak | Unknown/high | High | High | High | No evidence yet that backbone capacity is the limiting factor. |
| Crop augmentation | Medium | High but changes task | Medium | Medium | Medium | The crop protocol defines the difficulty and may be the main source of distortion headroom. |
| Evaluation mismatch | Medium | Medium | Low | High | Low | Smaller-scale analysis gives only a modest extra gain; report sensitivity, do not redefine silently. |
| Problem headroom | Strong | N/A | N/A | High | None | Crop-Global tail ratio is only about 1.18 and baseline is already fairly uniform. |

## Primary bottleneck

The primary bottleneck is the combination of limited baseline tail headroom and weak alignment between pixel-MSE tail and perceptual artifact tail. Patch localization and loss form are secondary.
