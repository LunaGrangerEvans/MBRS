# Content selector split protocol

Audited on 2026-09-09. Selector features, alpha values, and the conditional training decision use only the project validation partition. The existing formal test manifest is excluded from this phase's selector design and parameter selection.

## Actual data partitions

The project is `/root/workspace/GLX/icassp/MBRS`; its `datasets` link resolves to `/mnt/wmcontent/GLX/icassp/MBRS/datasets`. The split construction is recorded in [download_div2k.sh](../scripts/download_div2k.sh). The [complete split manifest](content_selector_split_manifest.csv) records all 900 filenames, source paths, and image-file SHA-256 hashes.

| Project split | Images | Inclusive image IDs | Original source | Role in this phase |
|---|---:|---|---|---|
| train | 800 | `0001.png`–`0800.png` | `DIV2K_train_HR` | Existing training data; excluded from selector analysis |
| validation | 50 | `0801.png`–`0850.png` | `DIV2K_valid_HR` | Selector design, activity comparison, alpha selection, fixed qualitative examples |
| test | 50 | `0851.png`–`0900.png` | `DIV2K_valid_HR` | Final evaluation only after a selected formulation passes the gate and completes training |

The local validation and test partitions divide the 100 public DIV2K validation images. The project's test partition is not the official DIV2K test HR set. Reading all of `raw/DIV2K_valid_HR` for selector selection would include the project test images and is prohibited.

Live filesystem checks found all 900 expected images present. There are no duplicate filenames, resolved source paths, or identical image-file SHA-256 hashes between any pair of partitions. Each partition also has no internal byte-identical duplicates. This verifies exact file identity separation; it is not a near-duplicate visual-content audit.

Sorted filename digests use UTF-8 `filename + newline` for every entry, including the last:

| Split | Filename-list SHA-256 |
|---|---|
| train | `5b90cdb3ce0a36c91ddda938b79f32d9fc1c7277db6db3c54099261892e12459` |
| validation | `3bad101174f98d85be1c4668988e8bc6e23528c9bbd1e8e3cb6abfd827ba203c` |
| test | `d0dd9c594f2a6ca890d0fbc863e5b74b917a9e292dc80d5109c24b1a5694fbeb` |

These match the existing [controlled continuation protocol](controlled_continuation_protocol.md).

## Frozen validation manifest

The [preregistered analysis protocol](content_selector_preregistered_protocol.md) fixes the seven candidate rows, perceptual metric definitions, uncertainty calculation, selection rule, and training gate before selector results are computed. Its SHA-256 is `46006f013e1e3dc46a019a1f4319475c90a521b08f322a17f92e523a25113061`.

The new [validation manifest](/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/validation_manifest.pt) contains exactly the 50 sorted validation images and no train or test source entries. Its independently verified SHA-256 is `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`. The [provenance record](/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/provenance.json) binds this manifest to the protocol and analysis checkpoint.

Each image has one fixed view: convert to RGB, resize to 140×140 with PIL bilinear interpolation, center crop `[6:134, 6:134]` to 128×128, and normalize to `[-1,1]`. This view is deliberately deterministic; the training loader instead uses random crops after the same resize dimensions. A dedicated CPU Torch generator with seed `170901` produces the fixed 64-bit messages. The manifest stores image tensors, messages, source filenames and paths, source image hashes, the transform description, and predetermined example indices.

The qualitative examples are sorted indices `0, 10, 20, 30, 40`: `0801.png`, `0811.png`, `0821.png`, `0831.png`, and `0841.png`. They are fixed independently of selector scores and visual appearance. Every method uses these same examples and the Patch16/stride8 grid.

## Analysis checkpoint provenance

The analysis checkpoint is the user-specified incumbent, not a checkpoint selected using the new validation analysis:

`/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50/checkpoint_0020.pth`

Its verified SHA-256 is `74fd12a6147b20e68477d9439cf6c416181bc48f07cfa3c3384989f48850f92c`. The checkpoint records epoch 20, seed17, Patch16, stride8, Top10, global/local weights 0.5/0.5, and the deterministic controlled configuration. Its resolved run configuration identifies the source as:

`/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth`

The source checkpoint's verified SHA-256 is `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907`. Both checkpoints contain model and Adam state. Candidate scoring is analysis-only: it does not update these states or select additional checkpoints.

## Existing formal test manifests

[evaluate_all_checkpoints.py](../experiments/evaluate_all_checkpoints.py) creates the existing fixed manifest explicitly from `dataset_path/test`. The persisted artifacts are under mounted storage:

| Artifact | SHA-256 |
|---|---|
| [uniform_eval_manifest.pt](/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_manifest.pt) | `36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339` |
| [controlled_crop35_40_manifest.pt](/mnt/wmcontent/GLX/icassp/MBRS/reports/controlled_crop35_40_manifest.pt) | `ffa19caecb5d97e717208d3aad7755e67d4b027d79c7c405e83b82bc6a995dde` |

The base manifest holds 50 image/message pairs at 128×128 and 64 bits, seed17, with five fixed attack repeats. The extension holds crop35/40 masks for the same seed, sample count, repeats, and batch size 16. The historical base manifest does not embed source filenames or crop coordinates, so its source partition is established by its generator and matching historical artifact hash, rather than a newly inferred image ordering. This audit reads identity metadata, not test metrics for selection.

## Evidence boundary and conditional use

Validation images `0801`–`0850` have already been monitored during earlier training epochs. In [train_local_patch.py](../experiments/train_local_patch.py), validation runs with `model.eval()` and `torch.no_grad()`; checkpoint saves follow the configured epoch schedule. This is development validation independent of training updates and of the project test images, not previously untouched data.

Historical fixed-test diagnostics and method/ratio comparisons informed the research history. Therefore, the defensible claim is that the project test partition is held out from **this phase's selector formulation and alpha selection**. It is not a globally pristine blind test across all earlier research decisions.

Only the frozen validation results may determine the content feature and alpha. If the preregistered gate fails, launch no training, perform no new formal test evaluation, and stop content-normalization tuning. If it passes, freeze exactly one formulation before the single authorized controlled continuation; the existing fixed test manifests may then be used for its final evaluation. No test result may reopen feature, alpha, or Top-ratio selection.
