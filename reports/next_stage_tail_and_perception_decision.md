# Next-stage tail and perception decision

## 1. Current strongest evidence

The current best controlled method is seed17 Hard Patch16/stride8/Top25/global-local0.5. It achieves `ΔWorstPSNR=+0.233 dB`, `ΔPSNR=+0.183 dB`, and `ΔP95 MSE=-0.000090` with BER30 change `+0.00006`.

## 2. Why improvement remains limited

- The crop-trained Global baseline is already moderately uniform; tail headroom is limited.
- Top25 with overlap supervises a broad 38.9% unique pixel union and has 2.30× mean redundancy.
- The tail improvement is partly a distribution-wide shift; local-specific gain is only about `+0.050 dB`.
- MSE ranking is moderately content-biased and weakly aligned with LPIPS/SSIM ranking.

## 3. Is Top25 too broad?

Yes. Under stride8, Top25 selects 57 of 225 patches, covers 38.9% of pixels, and has a largest connected selected component covering about 62% of the image. It is not a narrow extreme-tail mask.

## 4. Best reduced Top-ratio candidate

Top10% is the best candidate from analysis-only coverage:

- 23 selected patches out of 225;
- 19.4% mean unique pixel coverage;
- 1.88× mean overlap multiplicity;
- about 4.56 connected selected components;
- non-degenerate region count.

Top15% is less selective at 26.1% coverage. No ratio grid should be run.

## 5. MSE alignment with LPIPS and SSIM

Alignment is weak. At 32×32, current-best MSE versus LPIPS Spearman is `-0.018`, MSE versus SSIM degradation is `-0.056`, and Top25 Jaccard overlap is approximately `0.15` for both comparisons.

## 6. Content bias

MSE has moderate correlation with original content activity: for the current best at 32×32, MSE versus gradient magnitude is `0.402`, and MSE versus high-frequency energy is `0.398`. The loss is therefore partly attracted to textured/edge-rich regions.

## 7. Is current +0.233 dB truly tail-specific?

Partly. The upper percentiles improve more than the median, but P50/P75 also improve. The overlap method has a tail-specificity diagnostic of only about two percentage points, so most of the result is not an isolated extreme-tail correction.

## 8. Should one reduced-ratio training be launched?

**YES — exactly one run.** The spatial coverage analysis supports a single Top10% test. No other ratio, loss, or architecture should be changed in this run.

## 9. Exact configuration

`seed17_hard_patch16_stride8_top10_global_weight0.5_local_weight0.5`

- same seed17 epoch100 source checkpoint;
- same 20-epoch deterministic continuation;
- same restored Adam/BatchNorm state;
- same RandomCrop(0.3,1.0), batch size, device, data order, and fixed manifest;
- Patch16, stride8, Top10%, global/local image weights 0.5/0.5.

## 10. What result justifies continuing Hard Top-k

- ΔWorstPSNR greater than `+0.233 dB`, ideally at least `+0.30 dB`;
- local-specific gain greater than `+0.050 dB`;
- Top25/P95/P99 MSE reductions larger than overlap Top25;
- BER30 absolute change within `±0.002`;
- no degradation in local SSIM/LPIPS.

## 11. What terminates Top-ratio tuning

If Top10 fails to exceed overlap or improves MSE while worsening local SSIM/LPIPS, stop Top-ratio tuning. Do not run Top12/Top15/Top20 afterward.

## 12. Is metric alignment now the more important problem?

Yes. If Top10 fails, the next research advance should come from redefining the local artifact risk, not from another ranking ratio.

## 13. Best next-generation loss concept

The next concept should be a content/perceptual proxy that remains simple and differentiable:

1. first test content bias using local gradient/variance normalization;
2. if bias is confirmed, use a detached content-normalized tail score;
3. keep MSE as the differentiable base and use structural/perceptual metrics for validation before considering training terms.

## 14. Recommended paper framing

The paper should present explicit local-tail control for crop-robust watermarking, with overlap as the current method and MSE/perceptual mismatch as a limitation and motivation for the next generation. Do not claim crop-induced concentration or broad perceptual superiority.

## Terminal summary

### CURRENT BOTTLENECK

Top25 overlap coverage is broad, and pixel-MSE tail ranking is weakly aligned with perceptual artifact ranking.

### TOP25 COVERAGE DIAGNOSIS

Too broad under stride8: 38.9% unique pixel coverage and 2.30× redundancy.

### BEST NEXT TOP RATIO

Top10%, selected because it yields 19.4% unique coverage with 23 selected patches.

### MSE-PERCEPTUAL ALIGNMENT

Weak: near-zero/negative rank correlations and about 0.15 Top25 Jaccard overlap.

### CONTENT BIAS

Moderate: MSE correlates with gradient magnitude and high-frequency energy around 0.40 at 32×32.

### TAIL-SPECIFICITY

Positive but modest: overlap improves upper percentiles more than the median, with only about two percentage points of extra tail reduction.

### RUN ONE NEW TRAINING?

YES, exactly one reduced-ratio overlap run.

### IF YES, EXACT CONFIG

`seed17_hard_patch16_stride8_top10_global_weight0.5_local_weight0.5`.

### STOP HARD-TOPK TUNING IF

Top10 does not exceed `+0.233 dB` Worst PSNR or worsens local SSIM/LPIPS.

### NEXT-GENERATION METHOD DIRECTION

Content/perceptually aligned local-tail scoring, after analysis-first validation.

### RECOMMENDED PAPER STORY

Crop-robust watermarking incurs local visual-tail risk; explicit fine-grained supervision reduces that measured tail with negligible BER cost, while current pixel-MSE tail selection has clear perceptual and content-alignment limitations.

## Top10 result update

The one approved reduced-ratio run is complete. The configuration was `seed17_hard_patch16_stride8_top10_global_weight0.5_local_weight0.5`.

- ΔPSNR: `+0.180 dB`
- ΔWorstPSNR: `+0.247 dB`
- Local-specific gain: `+0.067 dB`
- ΔTop25 MSE: `-0.000065`
- ΔP95 MSE: `-0.000091`
- ΔBER30: `-0.000188`
- local SSIM improved; local LPIPS did not worsen materially

This exceeds the previous Top25 overlap reference (`+0.233 dB` Worst PSNR) and meets the predefined engineering success criteria. Top-ratio tuning is now stopped: no Top12/15/20 follow-up is justified in this phase.
