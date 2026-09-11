# Frozen ablations

All deltas are candidate minus Global under the same frozen evaluator. These are existing configurations, not a new search.

| Configuration | ΔPSNR | ΔTop25 local PSNR | Δlocal PSNR − Δglobal PSNR | ΔP95 MSE | ΔGini | ΔBER30 |
|---|---:|---:|---:|---:|---:|---:|
| Hard MSE, patch16 / stride16 / Top25% | +0.154975 | +0.193832 | +0.038857 | -0.000014029 | -0.005976 | +0.0000625 |
| Hard MSE, patch16 / stride8 / Top25% | +0.182157 | +0.225278 | +0.043121 | -0.000016048 | -0.006403 | +0.0000625 |
| Hard MSE, patch16 / stride8 / Top10% (main) | +0.179128 | +0.232163 | +0.053035 | -0.000017198 | -0.008110 | -0.0001875 |
| Soft normalized weighting, patch16 / stride16 / T=0.25 | +0.087124 | +0.124483 | +0.037359 | -0.000009735 | -0.005915 | +0.0002500 |
| Multi-scale hard Top25%, patch16/stride16 (0.7) + patch32/stride32 (0.3) | +0.128221 | +0.165858 | +0.037636 | -0.000012237 | -0.005888 | -0.0000625 |
| Excess, patch16 / stride16 / threshold1 / scale3 | -0.360013 | -0.312024 | +0.047989 | +0.000019466 | -0.008276 | -0.0001875 |
| Gradient-aware MSE, patch16 / stride8 / Top10% / alpha2 | +0.069629 | +0.101983 | +0.032354 | -0.000008327 | -0.005597 | +0.0003125 |

The difference of the two PSNR deltas is descriptive: their aggregation differs, so it is not a matched-PSNR causal estimate. Training overlap and selection ratio are ablated through the listed controlled pairs.

Soft: detached softmax of per-image mean-normalized patch MSE / T. Multi-scale: 0.7×patch16 + 0.3×patch32 hard Top25 losses. Excess: 3×detached patch-mean MSE×mean(ReLU(patch MSE / detached mean − 1)²). Gradient-aware: original-image activity only changes ranking; selected raw patch MSE remains the optimized loss.
