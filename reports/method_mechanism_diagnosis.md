# Method mechanism diagnosis

## Hard Patch16 stride16

- Intended mechanism: select the 16 highest-error regions among 64 non-overlapping Patch16 regions.
- Actual gradient behavior: local gradient support is 25% of pixels; selected-set identity changes as errors move.
- Result: `ΔWorstPSNR=+0.203 dB` with nearly unchanged BER.
- Explanation: finer spatial averaging than Patch32/64 gives a less coarse target, but the method still improves global quality at the same time.
- Revisit: useful reference, not sufficient as a large-effect main contribution.

## Hard Patch16 stride8

- Intended mechanism: reduce boundary fragmentation by using 50% overlap.
- Actual spatial behavior: 225 candidate patches, 57 selected; unique selected pixel coverage is about 38.9%, average multiplicity about 2.30, and about 4.36 connected selected regions per image.
- Result: `ΔWorstPSNR=+0.233 dB`, the best current result.
- Explanation: overlap improves localization, but the selected area is broad and highly redundant, so the objective is not narrowly focused on a few independent artifacts.
- Revisit: current best; the next variable worth studying is selectivity/coverage, not another backbone.

## Soft T=0.25 / 0.5 / 1.0

- Intended mechanism: smooth the discontinuity of hard top-k while retaining tail emphasis.
- Actual behavior: decreasing temperature makes the weighting more concentrated. T=0.25 is closest to hard selection; T=1.0 is close to uniform weighting.
- Result: T=0.25/0.5/1.0 produce `+0.136/+0.104/+0.037 dB` Worst PSNR.
- Explanation: soft weighting is stable but does not beat hard selection. The temperature trend is internally consistent, so further temperature search is low value.
- Revisit: retain T=0.25 as an ablation only.

## Multi-scale Patch16 + Patch32

- Intended mechanism: Patch16 handles small artifacts while Patch32 handles broader structure.
- Actual behavior: the 30% Patch32 term dilutes the fine-grained Patch16 signal.
- Result: `ΔWorstPSNR=+0.175 dB`, below both single-scale Hard and overlap.
- Explanation: the larger scale adds a more global objective without improving the target metric enough to compensate.
- Revisit: not worth further mixture-weight search under the current formulation.

## Excess Distortion Loss

- Intended mechanism: penalize regions whose error exceeds the detached per-image mean, avoiding discrete top-k ranking.
- Initial diagnostic: after fixed scale 3, encoded-image gradient norm was `1.74e-4`, close to Hard `1.92e-4`.
- Training behavior: final raw Excess local loss was approximately `0.000164`, versus Hard local raw loss approximately `0.001454`.
- Result: `ΔPSNR=-0.363 dB`, `ΔWorstPSNR=-0.285 dB`, Top25/P95 MSE increased, and local SSIM fell.
- Explanation: initial gradient calibration did not remain valid. The active excess set and normalized magnitude change during continuation; the loss becomes sparse and weak in the later trajectory. It also does not explicitly target the upper-ranked tail, so errors can remain high without producing enough excess signal.
- Revisit: reject this exact formulation. Do not rescue it with a grid.

## Historical Patch32/64 and Patch8

- Patch32/64 select too few large regions: 4 or 1 patch at Top25%, producing coarse, high-variance spatial updates.
- Patch8 produces many correlated micro-regions and historically large topology-dependent failures.
- These results support a spatial-granularity explanation, but historical execution confounds prevent them from being causal controlled evidence.
