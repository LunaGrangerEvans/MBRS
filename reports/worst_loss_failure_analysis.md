# Why the default worst-patch loss failed

## Executive diagnosis

The default patch32/top25/weight50 loss did not fail because of an arithmetic bug or because its scalar value overwhelmed the message objective. It failed through a combination of three effects:

1. Hard top-k changes the spatial objective discontinuously as patch ranks change.
2. At patch32 only four blocks determine the local term; at patch64 one block determines it. This creates a coarse, high-variance spatial signal even though the selected pixel fraction remains 25%.
3. The experiment runtime is not reproducible enough to isolate that effect. In particular, nominally identical seed/config runs used different single-GPU versus `DataParallel` topologies with BatchNorm and sometimes diverged by more than 6 dB.

The promising Patch16 result is consistent with averaging more spatial units, but it is not yet a clean causal proof. A deterministic continuation-control experiment is needed next.

## Matched-seed outcome

All deltas below are paired against the crop-trained Global model at the same seed.

| Configuration | ΔPSNR mean ± std | ΔWorstPSNR mean ± std | ΔBER30 mean ± std | Direction of ΔWorst |
|---|---:|---:|---:|---|
| Default patch32/top25/weight50 | -2.054 ± 1.412 | -1.961 ± 1.615 | +0.00346 ± 0.00955 | 0/3 positive |
| Patch16/top25/weight50 | +0.701 ± 1.038 | +0.890 ± 0.997 | +0.00092 ± 0.00209 | 2/3 positive; seed29 ≈ 0 |
| Patch16/top25/weight75 | -0.055 ± 0.619 | +0.101 ± 0.741 | -0.00125 ± 0.00348 | 2/3 positive, below effect threshold |
| Patch32/top25/weight25 | -0.750 ± 2.120 | -0.586 ± 1.919 | -0.00119 ± 0.01023 | 2/3 positive but seed29 collapses |
| Patch32/top50/weight50 | -0.297 ± 0.972 | -0.264 ± 0.777 | -0.00085 ± 0.00561 | 1/3 positive |

Patch16/weight50 passes the internal engineering screen, but a paired one-sample t-test with only three seeds gives `p=0.363` for ΔPSNR and `p=0.262` for ΔWorstPSNR. This is not a paper-level significance result.

## Raw and weighted loss magnitude

For default Worst seed17:

| Epoch | Weighted message | Weighted global | Weighted local | Total |
|---:|---:|---:|---:|---:|
| 1 | 1.59475 | 0.09989 | 0.13932 | 1.83395 |
| 10 | 0.18712 | 0.01157 | 0.01417 | 0.21286 |
| 40 | 0.11426 | 0.00360 | 0.00441 | 0.12228 |
| 100 | 0.10487 | 0.00142 | 0.00170 | 0.10799 |

The local term contributes about 7.6% of the scalar total at epoch 1 and 1.6% at epoch 100. It does not dominate the scalar objective. The message term remains dominant throughout training.

On a fixed diagnostic batch, however, the weighted local encoder-gradient norm divided by the weighted global encoder-gradient norm was `1.41`, `1.06`, `1.02`, and `1.46` at epochs 1, 10, 40, and 100. Thus local loss is small relative to message loss but as influential as the global image regularizer on the encoder's spatial update.

For Weight25, the corresponding local/global gradient ratios were `0.45`, `0.35`, `0.50`, and `0.35`. This confirms that reducing the local coefficient materially restores the global regularizer's control, even though Weight25 did not replicate reliably across seeds.

## Hard top-k selection behavior

The local gradient touches exactly 25% of encoded-image pixels for patch16, patch32, and patch64. The difference is the number and granularity of selected regions:

| Configuration | Selected patches/image | Local-gradient pixel support |
|---|---:|---:|
| patch16/top25 | 16 of 64 | 25% |
| patch32/top25 | 4 of 16 | 25% |
| patch64/top25 | 1 of 4 | 25% |

On the fixed diagnostic batch, selected-set Jaccard overlap across checkpoints was:

| Configuration | epoch1→10 | 10→40 | 40→100 |
|---|---:|---:|---:|
| patch32/weight50 | 0.295 | 0.410 | 0.286 |
| patch16/weight50 | 0.502 | 0.298 | 0.491 |
| patch64/weight50 | 0.625 | 0.625 | 0.625 |
| patch32/weight25 | 0.230 | 0.230 | 0.376 |

Low overlap confirms that the identity of selected patches is a moving target. Patch64's apparently higher overlap is not smoothness: because only one patch is selected, each image contributes either overlap 1 or overlap 0. A switch replaces the entire local target for that image.

## Reproducibility confound

Two nominally identical configurations were accidentally executed both through the multi-GPU candidate suite and single-GPU parallel suite. The model contains BatchNorm, so `DataParallel` changes per-replica BatchNorm batch statistics (batch 8 per GPU versus batch 16 on one GPU) in addition to CUDA nondeterminism.

Examples at epoch 100:

| Same nominal config/seed | Multi-GPU candidate PSNR / worst | Single-GPU parallel PSNR / worst |
|---|---:|---:|
| patch16/weight50 seed29 | 34.421 / 33.788 | 34.590 / 33.772 |
| patch16/weight50 seed41 | 36.199 / 35.528 | 29.765 / 28.582 |
| patch32/weight25 seed29 | 31.631 / 31.015 | 32.562 / 31.353 |
| patch32/weight25 seed41 | 34.714 / 33.966 | 30.556 / 29.381 |

This topology effect is larger than the method effect. Future matched comparisons must use one fixed device topology and deterministic controls.

## Hypothesis verdicts

### H1: local weight is too large and disrupts Global/message optimization

Partially supported. Weight50 local loss does not dominate the message term, but it exceeds the global image-loss gradient on the encoder at several checkpoints. Weight25 restores a global-dominant image gradient. The default failure cannot be attributed to scalar magnitude alone.

### H2: hard top-k is a moving target with high gradient variance

Supported, with a causality caveat. Patch membership has low overlap between fixed-image checkpoint evaluations, and final outcomes have large seed variance. Runtime nondeterminism is a competing cause, so a controlled continuation experiment is required to isolate top-k variance.

### H3: patch16 is more stable because it averages more patches

Provisionally supported. Patch16 averages 16 selected patches versus four for patch32 and is the only configuration passing the matched-seed engineering screen. It does not cover more pixels; the benefit is finer spatial averaging. Seed29's near-zero ΔWorst and the topology confound prevent a stronger conclusion.

### H4: patch64/top25 is effectively one-worst-patch optimization

Supported structurally and by the available seed17 result. It selects one of four patches and produced ΔWorstPSNR `-1.826 dB`. Three-seed evidence is absent, so this should remain an ablation observation rather than a statistical claim.

### H5: local MSE adds no new perceptual information

Confirmed mathematically. It uses the same squared residual as global MSE and only redistributes spatial gradient weight. Independent local SSIM/LPIPS evaluation is therefore necessary.

## Implication

Do not add GAN, frequency, or perceptual training losses yet. First make training topology/reproducibility controlled, then compare Global continuation versus Patch16/top25 continuation with a smaller local coefficient or warm-up. If hard selection remains unstable under that protocol, replace it with a soft-tail spatial reweighting loss.
