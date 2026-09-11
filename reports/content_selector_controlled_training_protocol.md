# Frozen single controlled training and final-test interpretation

Frozen after validation screening and before training/test, 2026-09-09.

## Exactly one selected formulation

`seed17_contentaware_gradient_patch16_stride8_top10_alpha1_global0.5_local0.5`

Score `S_i = stop_gradient(MSE_i)/(1 + Ghat_i)`; Ghat uses the original image and the 5th/95th percentile normalization in the preregistered protocol. Select the largest 23 of 225 patch scores. Local loss is the mean of ORIGINAL differentiable MSE over those indices. No gradient through normalization, indices, or original activity. The training path computes only gradient activity, omitting HF and diagnostic statistics.

Source: `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth` (SHA-256 `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907`). Restore model/BatchNorm and Adam moments/steps using the existing trainer. Seed17, 20 continuation epochs, LR1e-4, batch16, workers0, same sorted files, generator seeds, crop/message RNG order, `RandomCrop(0.3,1.0)`, message weight10, global/local0.5. One visible GPU, deterministic algorithms, no scheduler/backbone change. Content calculations consume no randomness. No scratch run, alpha grid, or extra seeds.

The existing trainer is reused without changing its control flow. Its `contentaware_gradient` loss mode is fixed to the selected grid/alpha and rejects another grid. Historical loss modes retain their previous computations. Output directory must be absent at launch. Evaluate scheduled epoch20; no validation-based checkpoint selection or early stopping.

## One formal final test

After epoch20 only, evaluate four frozen checkpoints on the existing formal fixed test manifest and crop35/40 extension: Global continuation, overlapping Hard Top25, overlapping Hard Top10, selected content-aware Top10. Persist encoded outputs and per-image diagnostics so report/figure work needs no new test pass. Do not use test outcomes to modify the formulation or retry a variant.

Report native32 nonoverlapping metrics for direct comparability to the existing master table, AND separate Patch16/stride8 mechanism metrics. Do not mix these grids. The local LPIPS metric at Patch16 is the declared 16→32 bilinear proxy; native32 LPIPS remains the direct perceptual-tail result. Compute Jaccard using patch-index sets, not overlap-union pixels.

## Interpretation fixed before test

Compare primarily against Hard Top10 and secondarily against Global/Top25. A paper upgrade requires convergent evidence: increased LPIPS/SSIM selection alignment, improvement of native local SSIM and LPIPS/tails, preserved crop BER, and no material pixel-tail deterioration. Report actual deltas even when small, plus image-paired bootstrap CIs where per-image quantities are available. An increased selector Jaccard alone is insufficient because changing the ranking rule can raise alignment without improving the image.

Operational noninferiority tolerances against Hard Top10: WorstPSNR >= -0.01 dB, Top25/P95/P99 MSE <= +1% relative, and every measured BER <= +0.001 absolute (0.1 percentage point). These are engineering tolerances, not proof of statistical equivalence. Strong perceptual evidence requires a native32 local LPIPS Top25 decrease and native32 local SSIM bottom25 increase whose paired 95% CIs exclude zero, with the other reported perceptual tails moving consistently. If only one perceptual metric improves, or pixel-tail/BER noninferiority fails, retain the pixel-tail method as paper main method and describe this experiment as a tradeoff/failed normalization direction. No post-test tuning.

The test partition has been examined in previous research stages; it is held out from this phase's selector/alpha choice, not globally pristine. Single-seed, 50-image results and proxy LPIPS do not establish human-perceived artifact severity.
