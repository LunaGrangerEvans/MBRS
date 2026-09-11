# Local-loss code and experiment integrity audit

## Scope and verdict

Audited files:

- `experiments/config_global_128_m64.json`
- `experiments/train_local_patch.py`
- `experiments/losses.py`
- `experiments/evaluate_all_checkpoints.py`
- `experiments/evaluate_crop.py`
- `utils/Dataloader.py`
- `network/noise_layers/crop.py`

The local-loss reductions are mathematically correct for non-overlapping patches, and `patch_mean_mse` is exactly the same objective as global MSE. The main integrity risk is reproducibility rather than a reduction bug: CUDA determinism is not enabled, data paths are enumerated with unsorted `os.listdir`, RNG state is not checkpointed/restored, and the crop implementation samples one rectangle for the entire batch. Existing Global and PatchMean runs therefore cannot be treated as a clean deterministic equivalence test even though their objectives are identical.

## Actual training objective

For an encoded image `y`, cover image `x`, message `m`, and decoded message `m_hat`, the code minimizes

```text
L_total = c_message * L_message + c_global * L_global + c_local * L_local

L_message = mean((m_hat - m)^2)
L_global  = mean((y - x)^2) over batch, channel, height, width

L_local = 0                                  when mode = none
        = L_global                           when mode = mean
        = batch_mean(weighted mean of each image's selected top-k patch MSE)
                                                when mode = topk
```

The message coefficient is `10.0` in all audited configs. Important image-loss coefficients are:

| Configuration | `c_global` | `c_local` | Effective image objective |
|---|---:|---:|---|
| Global | 1.00 | 0.00 | `L_global` |
| PatchMean | 0.00 | 1.00 | exactly `L_global` |
| Default Worst | 0.50 | 0.50 | `0.5 L_global + 0.5 L_topk` |
| Weight25 | 0.75 | 0.25 | `0.75 L_global + 0.25 L_topk` |
| Weight75 | 0.25 | 0.75 | `0.25 L_global + 0.75 L_topk` |

Representative raw and weighted epoch-averaged training values from seed17 logs:

| Run / epoch | Raw message | Raw global | Raw local | Weighted message | Weighted global | Weighted local | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global / 1 | 0.159835 | 0.227277 | 0 | 1.59835 | 0.227277 | 0 | 1.82563 |
| Global / 100 | 0.010151 | 0.002028 | 0 | 0.101507 | 0.002028 | 0 | 0.103534 |
| Worst32-w50 / 1 | 0.159475 | 0.199780 | 0.278633 | 1.59475 | 0.099890 | 0.139317 | 1.83395 |
| Worst32-w50 / 100 | 0.010487 | 0.002835 | 0.003399 | 0.104873 | 0.001418 | 0.001699 | 0.107990 |
| Patch16-w50 / 1 | 0.159491 | 0.190580 | 0.303255 | 1.59491 | 0.095290 | 0.151628 | 1.84183 |
| Patch16-w50 / 100 | 0.010219 | 0.001367 | 0.001771 | 0.102194 | 0.000684 | 0.000886 | 0.103763 |
| Patch32-w25 / 1 | 0.159522 | 0.205413 | 0.294579 | 1.59522 | 0.154060 | 0.073645 | 1.82293 |
| Patch32-w25 / 100 | 0.010163 | 0.001846 | 0.002288 | 0.101630 | 0.001384 | 0.000572 | 0.103586 |

The message term dominates the scalar total loss. The local term is not numerically dominant over the message term, but at weight50 its encoder-gradient norm is typically comparable to or larger than the global image-loss gradient, so it can materially change the spatial optimization path.

## PatchMean equivalence

`_patch_scores_and_weights` computes a per-patch sum of squared residuals and divides by valid pixels times channels. `patch_mean_raw_mse` then weights each patch by its valid-pixel count before averaging over images. Algebraically this reconstructs the sum over all pixels and channels divided by their count, which is global MSE.

The implementation independently computes this reduction, asserts absolute difference `< 1e-6`, and then deliberately assigns `patch_mean_mse = global_mse` to preserve the exact floating-point reduction path used by Global.

The minimal test covers patch16, patch32, patch64 at 128×128 and a non-divisible 130×126 boundary case. Observed errors:

| Shape | Patch | Patches | Absolute error | Relative error |
|---|---:|---:|---:|---:|
| 128×128 | 16 | 64 | 0.000e+00 | 0.000e+00 |
| 128×128 | 32 | 16 | 1.192e-07 | 5.975e-08 |
| 128×128 | 64 | 4 | 0.000e+00 | 0.000e+00 |
| 130×126 | 32 | 20 | 0.000e+00 | 0.000e+00 |

Conclusion: there is no PatchMean-vs-Global mathematical difference under the current code. Their multi-seed performance differences are evidence of execution nondeterminism or run-history differences, not evidence that patch averaging is a new objective.

## Patch extraction and reduction

- Extraction uses `F.unfold(kernel_size=patch_size, stride=patch_size)`: patches do not overlap.
- For 128×128 and patch sizes 16, 32, or 64, coverage is exact and no padding occurs.
- For non-divisible dimensions, only the bottom and right are zero-padded. A matching valid-pixel mask excludes padded values from patch denominators.
- An assertion checks that valid coverage equals `height * width`.
- Reduction order is: squared residual → sum channels/pixels inside each patch → normalize to patch MSE → select patches independently per image → valid-pixel-weighted selected-patch mean per image → batch mean.
- Raw local MSE remains in the same units as global MSE, but its expected value changes with patch count and top-k count. Smaller `k` estimates a more extreme tail and is not numerically interchangeable with a larger `k`.

## Hard top-k behavior

Top-k selection is performed independently for every image (`dim=1`), not across the batch. The count is `ceil(number_of_patches * ratio)` with a minimum of one.

| Patch size | Patch grid | Total patches/image | top25 count | Selected pixel fraction |
|---:|---:|---:|---:|---:|
| 16 | 8×8 | 64 | 16 | 25% |
| 32 | 4×4 | 16 | 4 | 25% |
| 64 | 2×2 | 4 | 1 | 25% |

`torch.topk` indices are discrete and do not receive gradients. Gradients flow through selected values only; non-selected patches receive zero local-loss gradient. Because the global term is also present in top-k configurations, non-selected pixels still receive global-image gradients.

All three top25 variants touch 25% of pixels, so Patch16 is not better because it covers more area. Its plausible stability advantage is that its local estimate averages 16 spatial units instead of 4, while Patch64 makes a single block carry the entire local term. A boundary crossing in Patch64 replaces 100% of the selected set; Patch16 changes the objective more gradually.

## Randomness and reproducibility

Implemented seeding:

- Python `random.seed(seed)`
- `numpy.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`

Missing or fragile controls:

- `torch.use_deterministic_algorithms(True)` is not enabled.
- cuDNN deterministic/benchmark flags are not set.
- Multi-GPU runs use `DataParallel`, which can introduce nondeterministic CUDA reduction behavior.
- The dataset uses unsorted `os.listdir`; index-to-image mapping can change with filesystem enumeration.
- DataLoader has no explicit `generator` or `worker_init_fn`. Current configs use `num_workers=0`, but changing it weakens reproducibility further.
- Training checkpoints save model, optimizer, epoch, and config, but not Python/NumPy/Torch/CUDA RNG states. A resumed run is not equivalent to uninterrupted training.
- Dataset augmentation uses torchvision `RandomCrop` (Torch RNG), while watermark crop noise uses NumPy RNG.
- `RandomCrop` samples one rectangle and broadcasts it to the entire batch, so images in a batch do not receive independent crop locations/sizes.
- The sampled height and width ratios are independently uniform in `[sqrt(min_area), sqrt(max_area)]`; retained area itself is not uniformly distributed in `[min_area, max_area]`.

If Global and PatchMean run with the same seed, code revision, file ordering, device topology, and deterministic kernels, their trajectories should be identical up to negligible floating-point effects. The observed divergence from epoch 1 confirms that at least one execution condition was not deterministic/equivalent. Those runs should not be used to argue that PatchMean has a distinct effect.

## Uniform evaluation integrity

- The fixed manifest makes current checkpoint comparisons consistent: the final report contains all 28 new candidate/parallel epoch-100 runs.
- Canonical local evaluation is fixed at patch32/top25 regardless of training patch size, which is correct for cross-config comparison.
- Quality PSNR is calculated per batch and then sample-weighted across batches, rather than converting a single dataset-wide MSE. Comparisons remain valid under the shared protocol, but the aggregation definition should be documented.
- Attack masks are generated once per batch and broadcast across images in that batch. The reported 250 decoded samples are therefore not 250 independent crop geometries.
- Crop starts use an exclusive upper bound without `+1`, so the final legal start coordinate is never sampled except for full-image crops.
- Multiple suite scripts can append to the same JSONL without a file lock. The current final file is complete, but future concurrent evaluation should use separate outputs followed by a deterministic merge.

## Audit actions

- Strengthened `experiments/test_patch_mse.py` to report independent absolute/relative equivalence errors for patch16/32/64 and a boundary case.
- Did not alter training semantics during this audit.
