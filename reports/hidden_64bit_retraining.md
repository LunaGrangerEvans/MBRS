# HiDDeN reimplementation, 64-bit retrained

审计日期：2026-09-11。本文结果严格标注为 **HiDDeN reimplementation,
64-bit retrained**，不是 official pretrained baseline，也不是原论文结果的
严格复现。

## 结论先行

本次单 seed 64-bit HiDDeN retraining 已完成并可被可信地运行：从随机初始化开始，
在当前 MBRS 的 DIV2K train/validation split 上训练 200 epochs，并在固定 50-image
test manifest 上完成全部要求的 frozen evaluation。

它的实现证据是完整的，但测试性能较弱，尤其是 message recovery：

| 指标 | epoch-200 test 值 |
|---|---:|
| PSNR (dB) | 28.834154 |
| SSIM | 0.918691 |
| LPIPS | 0.069035 |
| Top25 local PSNR (dB) | 27.937939 |
| P95 native32 patch MSE | 0.001848372 |
| Gini | 0.143546 |
| BER100 | 0.299063 |
| BER70 | 0.282500 |
| BER50 | 0.290625 |
| BER40 | 0.326188 |
| BER30 | 0.393813 |

这些数字来自真实 checkpoint 和真实固定 manifest，已写入
[`hidden_64bit_results.csv`](hidden_64bit_results.csv)。它们不是现有 30-bit
checkpoint 的数字，也没有使用 local-tail 或 OKLab loss。

## 1. Source and implementation identity

训练基于公开的 [`ando-khachatryan/HiDDeN`](https://github.com/ando-khachatryan/HiDDeN)
PyTorch port，local commit 为：

```text
556f76dd0602e4351ed4c19bd2ee87d50ef3c0de
```

原作者仓库是 [`jirenz/HiDDeN`](https://github.com/jirenz/HiDDeN)，local commit 为：

```text
9baa4a79dffe293f7553068f92eb7527534cba26
```

原作者仓库是 Lua/Torch7 实现，README 没有发布 pretrained model；本次实际训练
使用的是作者 README 明确链接的 PyTorch port。port README 自身说明它是 work in
progress、没有完全复现原论文，因此本结果不使用 “official” 这个称呼。

上游 port 源码文件保持未修改。所有训练、数据、noise 适配和评测代码均在：

- [`external_baselines/run_hidden_64bit_retraining.py`](../external_baselines/run_hidden_64bit_retraining.py)
- [`external_baselines/hidden_adapter.py`](../external_baselines/hidden_adapter.py)

## 2. Data and fixed-message protocol

| Split | Source | Count | Usage |
|---|---|---:|---|
| train | `/mnt/wmcontent/GLX/icassp/MBRS/datasets/train` | 800 | training |
| validation | `/mnt/wmcontent/GLX/icassp/MBRS/datasets/validation` | 50 | per-epoch monitoring only |
| test | `uniform_eval_manifest.pt` | 50 | final frozen evaluation |

All inputs are RGB `128×128`, normalized to `[-1,1]`. The final test manifest is the
current MBRS fixed manifest:

```text
path: /mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_manifest.pt
SHA-256: 36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339
images: 50 × 3 × 128 × 128
messages: 50 × 64
seed: 17
```

Training and validation messages are fixed, not regenerated on every batch. For each
filename, the adapter hashes `HiDDeN-64|17|filename` with SHA-256 and takes the first
64 bits. The test messages are taken directly from the fixed manifest. This satisfies
the requested fixed 64-bit protocol, but differs from the original port's training
loop, which samples a fresh random message for each batch.

The input transform keeps the current MBRS geometry: resize to `140×140`, then
train-time random crop or validation-time center crop to `128×128`, followed by
`ToTensor` and channel normalization.

## 3. Architecture and loss

The model is the port's original HiDDeN architecture, instantiated from scratch:

```text
encoder: 4 ConvBNReLU blocks, 64 channels
message injection: concatenate 64 message channels with image/features
encoder output: 3-channel image
decoder: 7 ConvBNReLU blocks, 64 channels, global average pool, 64→64 linear
discriminator: 3 ConvBNReLU blocks, 64 channels, global average pool, linear logit
```

The training objective is the port's original non-VGG adversarial objective:

```text
L = 0.001 * adversarial BCE-with-logits
  + 0.7   * encoder image MSE
  + 1.0   * decoder message MSE
```

No local patch-tail loss, OKLab term, perceptual loss, external pretrained model, or
existing HiDDeN checkpoint was loaded. Optimizers are the port's Adam defaults; the
run uses batch size 32, one visible GPU, seed 17, and 200 epochs.

## 4. Required crop/noise adaptation and differences from the port

The training channel is an adapter-owned `UnifiedRandomCropNoiser` with the current
MBRS `RandomCrop(0.3,1.0)` semantics:

- height and width ratios are independently sampled uniformly from
  `[sqrt(0.3), 1]`;
- one rectangle is sampled per batch, matching the current MBRS implementation's
  batch-shared mask behavior;
- the retained rectangle is kept on the original `128×128` canvas;
- pixels outside the rectangle are zeroed in normalized tensor space, which is the
  current project crop protocol;
- the custom noiser is always applied.

This differs from the original port in explicit, recorded ways:

| Area | Original port | This reimplementation |
|---|---|---|
| Noise selection | `Noiser` randomly selects from Identity and configured layers | always-on current-project RandomCrop |
| Crop operation | `Crop` changes tensor spatial size | mask outside rectangle, retain canvas |
| Noise recipe | port experiments may combine crop/cropout/dropout/resize/JPEG | only the unified MBRS RandomCrop channel |
| Message sampling | fresh random message per batch | fixed deterministic 64-bit message per file |
| Dataset | documented 10k COCO train / 1k validation | current 800/50 DIV2K split |
| Framework | documented PyTorch 1.0 | isolated PyTorch 2.4.1+cu121 |

The crop adaptation is therefore necessary for a fair common protocol, but this is
not a claim that the original HiDDeN paper's full combined-noise training recipe was
reproduced.

## 5. Compatibility fix and training evidence

The unmodified port's `Hidden.train_on_batch` creates discriminator labels with
integer fill values. Under PyTorch 2.4, these become `Long` tensors and
`BCEWithLogitsLoss` raises:

```text
RuntimeError: result type Float can't be cast to the desired output type Long
```

The adapter sets `model.cover_label = 1.0` and `model.encoded_label = 0.0` in memory;
the upstream file is not edited. This only resolves dtype compatibility and does not
change the loss formula.

Training output:

```text
output: /mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_64bit_retrained
checkpoint: hidden_64bit_epoch_200.pyt
checkpoint SHA-256: a2181ef05278de48d5af28129474d508a3178b849b9b8655f8a3cb46c671a7e8
elapsed training time: 922.540 seconds
epochs: 200
seed: 17
```

The full history is in
`/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_64bit_retrained/training_history.csv`.
Selected monitoring values:

| Epoch | Train loss | Validation decoder MSE | Validation bit accuracy |
|---:|---:|---:|---:|
| 1 | 0.413830 | 0.277447 | 0.506673 |
| 25 | 0.245957 | 0.248062 | 0.558512 |
| 50 | 0.232788 | 0.250380 | 0.574626 |
| 100 | 0.194231 | 0.256954 | 0.607069 |
| 150 | 0.156638 | 0.223553 | 0.643745 |
| 186 | 0.135840 | 0.190338 | 0.731445 |
| 200 | 0.128496 | 0.204694 | 0.726074 |

There were no NaN/Inf losses and no sustained near-random plateau. The automatic
non-convergence stop was not triggered, so the run completed as planned. Validation
accuracy is only a training-channel monitoring signal; it must not be substituted for
the final fixed-test crop BER.

## 6. Frozen evaluation definition

The adapter uses the current MBRS frozen metric semantics, with the port-specific
decoder rule explicitly retained:

- quality tensors are converted to RGB `[0,1]` and clipped before quality metrics;
- PSNR is computed from the dataset mean per-image RGB MSE;
- SSIM uses `pytorch_msssim.ssim`, `win_size=7`, `data_range=1`;
- LPIPS uses `lpips.LPIPS(net="alex", version="0.1")`;
- Top25 local PSNR uses the four highest-MSE native non-overlapping `32×32` patches
  per image, converts each image's selected mean MSE to PSNR, then averages dB;
- P95 MSE is the mean of per-image 95th percentiles over the 16 native32 patches;
- Gini uses the same 16 native32 patch-MSE values and current concentration formula;
- crop BER uses the fixed five-repeat masks at 100/70/50/40/30% and decodes the
  unquantized normalized masked tensor;
- HiDDeN message bits use the port's documented `round().clip(0,1)` rule, rather than
  MBRS's sigmoid/threshold implementation, because the HiDDeN port decoder emits
  unconstrained regression values.

The 40% masks come from the existing project extension:

```text
path: /mnt/wmcontent/GLX/icassp/MBRS/reports/controlled_crop35_40_manifest.pt
SHA-256: ffa19caecb5d97e717208d3aad7755e67d4b027d79c7c405e83b82bc6a995dde
```

Its seed, sample count, repeats, and batch layout were checked against the base
manifest. No 35% result is included because it was not requested.

## 7. Final result

The machine-readable summary is
[`hidden_64bit_results.csv`](hidden_64bit_results.csv):

```csv
method,status,seed,epoch,checkpoint,checkpoint_sha256,samples,message_length,psnr,ssim,lpips,top25_local_psnr,p95_mse,gini,ber100,ber70,ber50,ber40,ber30
"HiDDeN reimplementation, 64-bit retrained",completed,17,200,/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_64bit_retrained/hidden_64bit_epoch_200.pyt,a2181ef05278de48d5af28129474d508a3178b849b9b8655f8a3cb46c671a7e8,50,64,28.83415412902832,0.9186909198760986,0.06903545558452606,27.937938690185547,0.0018483723979443312,0.14354593997255327,0.2990625,0.2825,0.290625,0.3261875,0.3938125
```

The literal CSV is authoritative.

Quality metrics were independently run through the current MBRS
`experiments.evaluate_extended_image_quality.evaluate` function using the saved
encoded tensors. Cross-check absolute differences were below `4.1e-6` for the
quality metrics; the LPIPS difference is numerical variation between the isolated
PyTorch 2.4 environment and the MBRS evaluator environment. Crop BER was recomputed
directly from the saved encoded tensors and the fixed masks.

## 8. Interpretation and boundary

This run establishes a usable, reproducible **64-bit retrained HiDDeN
reimplementation** under the requested common data and evaluation protocol. It does
not establish a strong watermarking baseline: the final test BER remains high even
at 100% retained area, and the image quality is substantially below the current MBRS
reference models.

The result should be retained as a strict single-seed implementation reference. No
parameter grid, second seed, checkpoint selection, local-tail objective, OKLab
objective, or claim of official pretrained HiDDeN performance is justified by this
run.
