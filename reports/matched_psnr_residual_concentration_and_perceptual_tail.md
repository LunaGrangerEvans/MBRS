# Matched-PSNR residual concentration and perceptual-tail analysis

> 审计注：历史correlation CSV中的corrcoef值为Pearson，不是Spearman。准确的逐图分网格配对值和Spearman见 [perceptual_evidence_audit.md](perceptual_evidence_audit.md)。本报告matched pair是最小PSNR gap的epoch1/epoch1，仅说明早期checkpoint，不能推断最终模型在matched-PSNR下的感知优势。下文top10 PSNR与正式Top25 WorstPSNR不是同一指标。

This report uses existing seed17 checkpoints only. No training was started. The primary comparison is Global continuation versus Hard Patch16/stride8/Top10; Hard Top25 and Gradient-aware alpha2 are auxiliary references.

## 1. Global distortion

At the final continuation checkpoint on the formal fixed manifest:

| Method | Global MSE | Global PSNR | BER30 |
|---|---:|---:|---:|
| Global continuation | 0.0009535 | 36.227 | 0.11313 |
| Hard stride8 Top10 | 0.0009148 | 36.407 | 0.11294 |
| Hard stride8 Top25 | 0.0009141 | 36.411 | 0.11319 |
| Gradient-aware Top10 alpha2 | 0.0009383 | 36.297 | 0.11344 |

Hard Top10 reduces total residual energy relative to Global. Gradient-aware alpha2 reduces it less and has a small BER regression.

## 2. Residual concentration

The following uses fixed non-overlap evaluation grids. The formal comparison is 32×32; 16×16 is a scale diagnostic.

### Final 32×32 grid

| Method | Top10/Mean | P95/Mean | P99/Mean | Max/Mean | CV | Gini | Top10 energy share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global continuation | 1.2935 | 1.2645 | 1.3341 | 1.3515 | 0.1722 | 0.0940 | 0.1617 |
| Hard stride8 Top25 | 1.2736 | 1.2454 | 1.3129 | 1.3298 | 0.1602 | 0.0873 | 0.1592 |
| **Hard stride8 Top10** | **1.2674** | **1.2398** | **1.3061** | **1.3227** | **0.1569** | **0.0855** | **0.1584** |
| Gradient-aware Top10 alpha2 | 1.2748 | 1.2467 | 1.3143 | 1.3312 | 0.1610 | 0.0878 | 0.1594 |

Hard Top10 lowers every normalized concentration metric. This is evidence that the method does more than reduce only the scalar global MSE, although the absolute changes are modest.

At 16×16, Hard Top10 shows the same direction: Top10/Mean, P95/Mean, CV, Gini, and Top10 energy share all decrease relative to Global.

## 3. Matched-PSNR comparison

Using saved checkpoints only, the smallest global-PSNR gap for Global versus Hard Top10 is the epoch1/epoch1 pair:

- Global: `34.8788 dB`
- Hard Top10: `34.8812 dB`
- absolute gap: `0.0024 dB`

At this pair, 32×32 concentration is slightly lower for Hard Top10:

- Top10/Mean: `1.24385 → 1.24047`
- P95/Mean: `1.21659 → 1.21358`
- CV: `0.14618 → 0.14416`
- Gini: `0.07986 → 0.07872`
- Top10 energy share: `0.15548 → 0.15506`
- Worst/top10 PSNR: `33.8928 → 33.9070 dB`

This is directionally positive but very small because the closest pair occurs early in the continuation trajectory. It does not establish a large matched-PSNR effect.

## 4. PSNR–concentration trajectories

The generated curves use all saved checkpoints 1, 5, 10, 15, and 20:

- `psnr_vs_p32_top10_psnr.png`
- `psnr_vs_p32_top10_over_mean.png`
- `psnr_vs_p32_gini.png`
- `psnr_vs_p32_top10_energy_share.png`

They are stored under `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/matched_psnr_residual_concentration/`.

The trajectories indicate that Hard Top10 generally tracks a lower concentration curve than Global at comparable PSNR, but the separation is narrow rather than dramatic.

## 5. Local perceptual tail

At the closest epoch1 matched-PSNR pair:

| Method | Top10 LPIPS | LPIPS tail gap | Bottom10 SSIM | SSIM tail gap |
|---|---:|---:|---:|---:|
| Global | 0.002323 | 0.001304 | 0.89303 | 0.04415 |
| Hard Top10 | 0.002410 | 0.001369 | 0.89354 | 0.04384 |

At this strictly matched early pair, SSIM tail is slightly better but LPIPS tail is worse. At the final checkpoint, Hard Top10 has better native local LPIPS than Global, but this coincides with a global PSNR increase, so it is not a pure matched-PSNR perceptual proof.

The fixed qualitative panel and distributions are stored under the same visualization directory:

- `patch_energy_cdf.png`
- `lorenz_curve.png`
- `top10_over_mean_distribution.png`
- `lpips_tail_distribution.png`
- `fixed_qualitative_residual_and_perceptual_tail.png`

## 6. Concentration–perception correlation

At final 32×32 patches, the analysis compares Top10/Mean, P95/Mean, Gini, CV, and Top10 energy share against LPIPS and SSIM tail gaps. The correlations are descriptive and image-level; they do not establish causality.

The result is not strong enough to claim that residual concentration is a reliable perceptual-artifact surrogate. This agrees with the earlier MSE/LPIPS/SSIM rank-overlap analysis.

## 7. Gradient-aware negative ablation

Gradient-aware alpha2 improves selector alignment but is worse as an end-to-end method:

- native32 ΔWorstPSNR: `+0.117 dB` versus MSE Top10 `+0.247 dB`;
- native32 ΔPSNR: `+0.070 dB`;
- BER30 change: `+0.000312` versus Global;
- Top25/P95 pixel-tail reductions are smaller;
- selector alignment improves, but that improvement does not translate into better overall local quality.

This supports the statement:

> Better selector alignment alone does not guarantee better final watermark imperceptibility.

## 8. Final interpretation

### CURRENT BEST METHOD

Hard Patch16/stride8/Top10 with global/local image weights 0.5/0.5.

### DOES OUR METHOD REDUCE GLOBAL RESIDUAL ENERGY?

**YES.** Global MSE decreases from `0.0009535` to `0.0009148` and PSNR rises by `+0.180 dB`.

### DOES OUR METHOD REDUCE NORMALIZED RESIDUAL CONCENTRATION?

**YES, modestly.** Top10/Mean, P95/Mean, P99/Mean, CV, Gini, and Top10 energy share all decrease.

### AT MATCHED GLOBAL PSNR, IS WORST-PATCH PSNR BETTER?

**YES, marginally.** The closest saved pair has a `0.0024 dB` global gap and a `+0.014 dB` Hard Top10 advantage in 32×32 top10 PSNR.

### AT MATCHED GLOBAL PSNR, IS RESIDUAL CONCENTRATION LOWER?

**YES, marginally.** All reported normalized concentration metrics move in the favorable direction at the closest pair.

### AT MATCHED GLOBAL PSNR, IS LPIPS TAIL BETTER?

**NO at the closest early pair.** LPIPS Top10 and LPIPS tail gap are slightly worse for Hard Top10. Final-checkpoint LPIPS improves, but that comparison includes a global PSNR increase.

### AT MATCHED GLOBAL PSNR, IS SSIM TAIL BETTER?

**Slightly yes** at the closest pair, but the effect is small.

### IS RESIDUAL CONCENTRATION ASSOCIATED WITH PERCEPTUAL TAIL?

**WEAK.** The direction is not strong enough to use residual concentration as a perceptual-artifact surrogate.

### DOES THIS SUPPORT “REDISTRIBUTES RESIDUAL ENERGY”?

**PARTIALLY.** Normalized concentration decreases beyond the global MSE reduction, but the matched-PSNR perceptual evidence is mixed.

### DOES THIS SUPPORT “REDUCES LOCALLY SALIENT ARTIFACTS”?

**PARTIALLY at best.** Native final LPIPS improves, but matched-PSNR LPIPS does not; no strong perceptual claim is justified.

### CURRENT PAPER CLAIM

Controlled fine-grained local-tail supervision reduces measured residual-energy concentration and crop-robust pixel tail with negligible BER cost. Perceptual artifact reduction is suggestive but not conclusively demonstrated.

### CURRENT LIMITATION

The evaluation is seed17-only, the LPIPS patch analysis includes a declared 16→32 proxy where needed, and residual concentration is only weakly aligned with perceptual tail severity.

Machine-readable metrics: [matched_psnr_residual_concentration_metrics.csv](matched_psnr_residual_concentration_metrics.csv). Matching pairs: [matched_psnr_matching_pairs.json](matched_psnr_matching_pairs.json).
