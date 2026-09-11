# HiDDeN compatibility audit

Audit date: 2026-09-11. Scope: HiDDeN external-baseline compatibility in the
MBRS project. No MBRS source file, MBRS environment, or upstream HiDDeN source
file was modified.

## Executive result

- The minimum CUDA smoke test passed with one fixed MBRS-manifest image and the
  first 30 bits of its message.
- A valid 64-bit unified baseline was not produced. The available trained
  checkpoint is 30-bit, while the fixed MBRS manifest is 64-bit. The guarded
  evaluation command stopped before calculating any metric.
- The original author repository contains no released pretrained model. Its
  README says “Coming soon...” under pretrained models.
- The usable checkpoint is a repository-tracked checkpoint from the community
  PyTorch port linked by the original author repository. It is suitable for a
  compatibility smoke test, but it is not an official author checkpoint and is
  not reported as a formal MBRS baseline.

## 1. Existing project audit

Before this audit, the MBRS project had no `external_baselines/repos/HiDDeN`, no
HiDDeN checkpoint, and no HiDDeN-specific environment. The existing
[`external_baselines/REPRODUCE.md`](../external_baselines/REPRODUCE.md) already
recorded HiDDeN as requiring retraining and having no verified checkpoint in
scope.

The current MBRS fixed manifest is:

| Field | Value |
|---|---:|
| Images | 50 |
| Image tensor | `50 × 3 × 128 × 128` |
| Message length | 64 bits |
| Crop repeats | 5 |
| Crop levels | 100%, 70%, 50%, 40%, 30% |

The manifest is at `/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_manifest.pt`.

## 2. Sources and provenance

### Original author repository

- Repository: [`jirenz/HiDDeN`](https://github.com/jirenz/HiDDeN)
- Authors named by the repository: Jiren Zhu, Russell Kaplan, Justin Johnson,
  and Li Fei-Fei
- Local clone commit: `9baa4a79dffe293f7553068f92eb7527534cba26`
- Implementation: Lua + Torch7 (`th`, `nn`, `cunn`, `cudnn`, `cutorch`)
- Local README status: work in progress; pretrained models are listed as
  “Coming soon...”.
- Local machine status: `th` and `luarocks` are not installed.

The paper/source identity is also supported by the
[ECCV 2018 paper record](https://openaccess.thecvf.com/content_ECCV_2018/html/Jiren_Zhu_HiDDeN_Hiding_Data_ECCV_2018_paper.html).

### Practical PyTorch port

- Repository: [`ando-khachatryan/HiDDeN`](https://github.com/ando-khachatryan/HiDDeN)
- Local clone commit: `556f76dd0602e4351ed4c19bd2ee87d50ef3c0de`
- The original author README explicitly links this repository under “Other
  implementations”.
- The port README explicitly says it is a work in progress and does not fully
  reproduce the original paper. It is therefore treated as a known,
  source-identifiable community port, not as the official implementation.
- The port README says its experiment folders contain settings, logs, and
  trained-model checkpoints. The selected checkpoint and its metadata are
  Git-tracked in that repository.

Selected smoke checkpoint:

```text
experiments/no-noise adam-eps-1e-4/checkpoints/no-noise--epoch-200.pyt
SHA-256: 71118462194d647f6f5c08dd35f95d4f53db11a9bd2758f3bd336b46aafa5916
Git-tracked blob: ec9743f6009ca07340cb6983c90d9237a66926af
```

The checkpoint was loaded with `weights_only=True` and its state dictionary
was loaded with `strict=True`. Its adjacent options file resolves to:

```text
H=W=128
message_length=30
encoder_blocks=4, encoder_channels=64
decoder_blocks=7, decoder_channels=64
decoder.linear.weight shape=(30, 30)
encoder-decoder state keys=95
checkpoint epoch=200
```

This verifies structure compatibility with the port's own model. It does not
verify equivalence to the unavailable official author checkpoint.

## 3. Isolated environment

Environment prefix: `external_baselines/envs/hidden`.

The environment is separate from the MBRS environment and was created from a
CUDA-capable Python 3.10 base, then given the additional metric packages
needed by this adapter. The dependency note is
[`external_baselines/hidden_environment.yml`](../external_baselines/hidden_environment.yml).

Resolved runtime:

| Package/runtime | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.4.1+cu121 |
| torchvision | 0.19.1+cu121 |
| CUDA runtime reported by PyTorch | 12.1 |
| NumPy | 2.2.6 |
| Pillow | 12.3.0 |
| SciPy | 1.13.1 |
| scikit-image | 0.24.0 |
| LPIPS | 0.1.4 |
| pytorch-msssim | 1.0.0 |
| GPU | NVIDIA A800-SXM4-80GB × 2 |

The adapter puts the vendored HiDDeN port at the front of `sys.path`, avoiding
the MBRS project's same-named `utils` package. The source-preserving adapter
files are:

- [`external_baselines/hidden_adapter.py`](../external_baselines/hidden_adapter.py)
- [`external_baselines/run_hidden_smoke.py`](../external_baselines/run_hidden_smoke.py)
- [`external_baselines/run_hidden_evaluation.py`](../external_baselines/run_hidden_evaluation.py)

## 4. Smoke and evaluation attempts

Smoke command:

```bash
PYTHONPATH="$PWD/external_baselines" \
  external_baselines/envs/hidden/bin/python \
  external_baselines/run_hidden_smoke.py --device cuda
```

Result: exit code 0. Details are in
[`reports/hidden_smoke_test.md`](hidden_smoke_test.md).

Guarded unified-evaluation command:

```bash
PYTHONPATH="$PWD/external_baselines" \
  external_baselines/envs/hidden/bin/python \
  external_baselines/run_hidden_evaluation.py --device cuda
```

Result: exit code 2, intentional guard. The output artifact is
`/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_evaluation_guard.json`.
The adapter refused to calculate PSNR, SSIM, LPIPS, Top25 local PSNR, or crop
BER because `30 != 64`. No unified-evaluation baseline numbers exist.

## 5. Training feasibility and decision

The original author source defaults to a 30-bit message and 200 epochs, while
the linked PyTorch port documents 10,000 COCO training images, 1,000 validation
images, and 200–400 epoch experiments. The current project has 800 DIV2K
training images, not the original 10k COCO training population.

On the A800, the reproducible five-step batch-32 64-bit training benchmark in
[`external_baselines/estimate_hidden_training.py`](../external_baselines/estimate_hidden_training.py)
measured:

```text
0.103983 seconds/step
4.7 GiB peak allocated GPU memory
```

Extrapolated training-only time for 10,000 images is approximately 1.8–3.7
hours for batch 32 and 200–400 epochs; validation, image loading, checkpoint
writes, and retries add overhead. The first unmodified modern-PyTorch training
attempt exposed a source compatibility issue: the port creates integer labels
with `torch.full`, which PyTorch 2.4 passes to BCE-with-logits as `Long`, causing
the expected `result type Float can't be cast to Long` error. The benchmark
used float labels in memory only; it did not edit upstream code or create a
trained checkpoint. The machine-readable measurement is at
`/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_training_benchmark.json`.

Decision: **do not start full training in this audit**. It would create a new
64-bit retraining on a different dataset/protocol, not an official HiDDeN
reproduction. It becomes worthwhile only if the project explicitly accepts a
new, clearly labelled 64-bit HiDDeN retraining protocol and supplies/chooses a
training dataset and fair comparison policy.

## Final compatibility status

| Capability | Status |
|---|---|
| Locate official source | PASS |
| Confirm official pretrained checkpoint | NOT AVAILABLE |
| Confirm known port checkpoint provenance | PASS, qualified |
| Load model/checkpoint structure | PASS, strict |
| One-image embed/decode on CUDA | PASS, 30-bit only |
| Current 50-image 64-bit unified evaluation | BLOCKED before metrics |
| Official reproduction claim | NOT SUPPORTED |
