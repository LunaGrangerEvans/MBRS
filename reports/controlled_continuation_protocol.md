# Controlled continuation protocol

## Objective

Isolate the effect of hard versus soft local-tail weighting from additional training time. For each seed, all branches start from the exact same crop-trained Global epoch-100 checkpoint and differ only in the image-loss objective.

This protocol does not use warm-up, full-scratch training, `DataParallel`, or hyperparameter search.

## Source checkpoints

| Seed | Source checkpoint | Epoch | SHA-256 |
|---:|---|---:|---|
| 17 | `optimization_global_seed17_128_m64_crop/checkpoint_0100.pth` | 100 | `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907` |
| 29 | `optimization_global_seed29_128_m64_crop/checkpoint_0100.pth` | 100 | `f840be4fa0345035b31dcf23e2616388103b5e397a601b9e578f51e00a51f44c` |
| 41 | `optimization_global_seed41_128_m64_crop/checkpoint_0100.pth` | 100 | `b81a5859e8cc9ca08b8af98b65696a978d74600c55733d8bc537e3a8c1ae1cc6` |

The controlled study uses seed17 only under the current single-seed policy. Seed29/41 branches are not launched. Historical multi-seed results remain available for context but are not extended in this round.

## Three seed17 branches

| Variable | Global continuation | Hard P16-T25-L50 | Soft P16-T0.5-L50 |
|---|---:|---:|---:|
| Source model/optimizer | same seed17 epoch-100 checkpoint | same | same |
| Continuation epochs | 20 | 20 | 20 |
| Learning rate | 1e-4 | 1e-4 | 1e-4 |
| Message coefficient | 10.0 | 10.0 | 10.0 |
| Global coefficient | 1.0 | 0.5 | 0.5 |
| Local coefficient | 0.0 | 0.5 | 0.5 |
| Patch size | n/a | 16 | 16 |
| Tail selection | none | hard top 25% | detached normalized softmax |
| Temperature | n/a | n/a | 0.5 |
| Crop augmentation | `RandomCrop(0.3, 1.0)` | same | same |
| Batch size | 16 | 16 | 16 |

## State restoration

- Model parameters are restored from the source checkpoint.
- BatchNorm affine parameters and running mean/variance are part of the model state and are restored identically. They continue updating in training mode.
- Adam optimizer state, including moments and step counters, is restored identically.
- The configured continuation LR `1e-4` is explicitly written into every restored optimizer parameter group.
- There is no learning-rate scheduler in `train_local_patch.py`; scheduler restoration is therefore not applicable.
- Continuation epoch numbering restarts at 1 for all branches, while `config.resolved.json` records the source checkpoint path and source epoch.

## Randomness controls

- Each branch runs in a fresh process and calls Python, NumPy, Torch CPU, and Torch CUDA seeding with seed17.
- `torch.use_deterministic_algorithms(True)`, deterministic cuDNN, disabled cuDNN benchmark, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are enabled.
- Exactly one GPU is exposed with `CUDA_VISIBLE_DEVICES`; the config requires `torch.cuda.device_count() == 1`, preventing accidental `DataParallel`.
- Dataset filenames are sorted only for configs marked `deterministic: true`.
- Train and validation DataLoaders use explicit independent generators derived from the same experiment seed and use `num_workers=0`.
- All branches use identical model construction and source loading, so model-construction RNG consumption is the same.
- Dataset random crop uses the same Torch RNG initialization in each branch. Message generation uses the same Torch RNG initialization and call order.
- Watermark crop augmentation uses NumPy; each branch starts from the same NumPy state and executes one crop call per batch. Local-loss computation consumes no RNG, so crop streams remain aligned across branches.

The optimization paths diverge after the first update by design, but data indices, augmentation draws, messages, and crop masks stay branch-aligned.

## Dataset and evaluation identity

Filename-list digests under stable sorting:

| Split | Images | Filename SHA-256 |
|---|---:|---|
| train | 800 | `5b90cdb3ce0a36c91ddda938b79f32d9fc1c7277db6db3c54099261892e12459` |
| validation | 50 | `3bad101174f98d85be1c4668988e8bc6e23528c9bbd1e8e3cb6abfd827ba203c` |
| test | 50 | `d0dd9c594f2a6ca890d0fbc863e5b74b917a9e292dc80d5109c24b1a5694fbeb` |

The existing fixed evaluation manifest is reused for images, messages, and crop30/50/70/100 masks:

`uniform_eval_manifest.pt` SHA-256: `36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339`

Crop35 and crop40 masks will be generated once as a deterministic extension and persisted before evaluating any branch. The same extension file is then reused for all methods.

## Logging

Every train/validation epoch records:

- raw message, global, and local loss;
- weighted message, global, local, and combined image loss;
- total loss;
- PSNR, worst-patch PSNR, bit accuracy, and BER;
- effective global/local coefficients.

Soft-tail additionally records mean/max soft weight and effective patch count `1 / sum(w_i^2)`. Separate historical gradient diagnostics already exist; per-step multi-objective gradient probing is omitted from formal training to avoid changing runtime and memory behavior.

## Isolation from the legacy suite

The running full-scratch P16/T25/L25 suite remains exploratory. Its outputs and logs use different directory prefixes and are excluded from controlled summaries. Controlled runs must not start while another MBRS process is using the selected GPU unless an explicit resource check confirms isolation.
