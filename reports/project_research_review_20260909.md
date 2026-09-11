# MBRS crop-robust local-distortion project review — 2026-09-09

## 1. Executive summary

The project has a credible controlled seed17 experiment, a reproducible baseline/continuation protocol, and a measurable local-tail effect. The strongest current method is Hard Patch16 with 50% overlap and Top10% selection. It improves formal 32×32 Worst PSNR by `+0.247 dB`, Top25 MSE by about `5.5%`, and P95 MSE while changing BER30 by only `-0.00019`.

The effect is real in the controlled setting but modest. The main reason is not simply insufficient backbone capacity. The crop-trained Global baseline already has a relatively mild distortion tail, overlap covers a broad and redundant region, and pixel-MSE rankings align poorly with perceptual patch rankings. The project should now be framed as explicit local-tail control under crop robustness, not general PSNR improvement or proof that crop training concentrates artifacts.

## 2. Original idea and hypothesis change

Original hypothesis: crop training concentrates artifacts into local regions.

Data-supported revision: crop training raises absolute visual distortion, while a global average image objective leaves upper-tail local risk implicit. The concentration-ratio claim is unsupported: no-crop top25/global is about 1.374, crop-trained Global is about 1.181.

## 3. Trustworthy versus discardable evidence

Trustworthy formal evidence:

- seed17 deterministic continuation branches from one epoch-100 source checkpoint;
- same Adam/BatchNorm state, duration, device, manifest, and crop protocol;
- fixed-manifest evaluation of Global, Hard, Soft, overlap, multi-scale, and Excess.

Exploratory only:

- historical DataParallel and non-deterministic full-scratch runs;
- duplicated parallel runs;
- cross-seed means and p-values, which are not part of the current teacher-defined objective;
- incomplete Patch16/local-weight0.25 full-scratch result.

## 4. Strongest current evidence

- Hard Patch16 stride8 is best: `ΔPSNR +0.183`, `ΔWorstPSNR +0.233`, `ΔTop25 MSE -0.000062`, `ΔP95 MSE -0.000090`, BER30 `+0.00006`.
- Reducing the same overlap configuration to Top10% improves the result to `ΔPSNR +0.180`, `ΔWorstPSNR +0.247`, `ΔTop25 MSE -0.000065`, `ΔP95 MSE -0.000091`, BER30 `-0.00019`.
- At 16×16 analysis scale, overlap reaches about `+0.248 dB` Worst PSNR; at 32×32 it is `+0.233 dB`.
- Selected overlap patches cover about 38.9% of pixels with multiplicity 2.30, so the method is not a tiny-region detector.
- MSE-to-LPIPS and MSE-to-SSIM rank correlations are near zero or negative at 32×32.
- Multi-scale underperforms overlap; Excess fails despite initial gradient-scale calibration.

## 5. Why the effect remains small

1. The crop-trained baseline is not extremely tail-concentrated; there is limited headroom.
2. Hard Top25 on overlap selects many correlated patches and covers a broad union.
3. Global PSNR improves together with the tail; local-specific gain is only about `+0.050 dB`.
4. The optimized MSE tail is weakly aligned with perceptual artifact rankings.
5. Excess normalization is not dynamically stable across training.

## 6. Primary and secondary bottlenecks

Primary: task headroom and metric alignment.

Secondary: tail selectivity/coverage and evaluation-scale definition.

Not currently supported as primary bottlenecks: backbone capacity, optimizer choice, or need for a larger architecture.

## 7. Current best method

`seed17_hard_patch16_stride8_top10_global_weight0.5_local_weight0.5`.

It is the best current engineering candidate, not yet a large-effect or general-purpose method.

## 8. Next three experiments, design only

1. **Content/activity diagnostic**: no training initially; measure whether MSE-worst regions are high-texture/edge regions. Only design a content-normalized loss if the bias is strong.
2. **Larger-region structural tail diagnostic**: compare 32×32 and larger SSIM/LPIPS tails before adding any perceptual training term.
3. **Perceptual-tail definition**: only after diagnostics, decide whether the paper should remain MSE-tail focused or move to a structural/content proxy.

Do not combine methods or launch another grid before these decisions are reviewed.

## 9. Directions not worth GPU now

- seed29/41;
- more Soft temperatures;
- Excess threshold/scale grids;
- Patch8/Patch64 sweeps;
- GAN, discriminator, frequency loss, backbone changes;
- full-scratch matrices;
- unmotivated warm-up combinations.

## 10. Paper contribution assessment

`+0.247 dB` alone is not a strong main-method contribution. The result becomes defensible if presented as a controlled local-tail intervention with:

- Top25/P95/P99 reductions;
- local SSIM/LPIPS support where available;
- BER preservation;
- residual and selected-region visualizations;
- explicit limitations about seed17-only and MSE/perceptual mismatch.

The paper claim should be:

> Explicit fine-grained local-tail supervision can reduce crop-induced high-distortion regions with negligible message-recovery cost under a controlled MBRS protocol.

The project cannot currently claim a large perceptual improvement, generalization across datasets/backbones, or statistical superiority across seeds.

## 11. Minimum viable paper

- Motivation: crop robustness versus local visual tail.
- Method: explicit local tail, Hard Patch16, overlap.
- Controlled table: Global, Hard stride16, Soft T0.25, overlap, multi-scale, Excess.
- Distribution/CDF and percentile curves.
- BER crop curve.
- Residual/heatmap/selected-region visualizations with fixed selection rule.
- Reproducibility and limitation section.

## 12. Final decision

The project is scientifically viable as a focused local-tail control study, but not yet as a broad claim of superior watermarking quality. The next decision should target selectivity or metric alignment, not model size. No new formal training should start before the analysis-only evidence and paper framing are accepted.

## Required final statements

### CURRENT RESEARCH QUESTION

Can explicit local-tail supervision reduce high-distortion watermark regions under crop-robust training while preserving BER and global quality?

### CURRENT BEST METHOD

Seed17 Hard Patch16, stride8, Top25%, global/local image weights 0.5/0.5.

### WHAT IT ACTUALLY IMPROVES

32×32 Worst PSNR by `+0.247 dB`, Top25 MSE by about `5.5%`, P95 MSE, local SSIM, and selected residual tail, with negligible BER change.

### WHY THE EFFECT IS STILL SMALL

The baseline tail has limited headroom, overlap selection is spatially redundant, and MSE tail rankings are weakly perceptual.

### PRIMARY BOTTLENECK

Problem headroom plus local metric alignment, not backbone capacity.

### MOST PROMISING NEXT IDEA

Content/perceptual metric alignment analysis; Top-ratio selectivity has now been tested once and should not be further swept.

### NEXT EXPERIMENT #1

Content/activity correlation diagnostic on the fixed manifest.

### NEXT EXPERIMENT #2

32×32/larger structural-tail diagnostic.

### NEXT EXPERIMENT #3

Define a perceptual/content proxy only if diagnostics justify it; do not start training automatically.

### DO NOT SPEND GPU ON

Seed29/41, temperature grids, Excess grids, Patch8/Patch64 grids, GAN/frequency/backbone changes, or combined-method searches.

### PAPER CLAIM WE CAN ALREADY SUPPORT

Controlled fine-grained local-tail supervision reduces measured crop-robust watermark distortion tail with negligible BER cost.

### PAPER CLAIM WE STILL CANNOT SUPPORT

Large perceptual improvement, crop-induced concentration, cross-seed statistical superiority, or generalization beyond the current MBRS setting.

### WHAT WOULD MAKE THIS A STRONG PAPER

A clear tail-risk motivation, controlled protocol, one selective local-tail method, perceptual alignment evidence, full distribution visualizations, and honest limitation of the claim to the demonstrated setting.
