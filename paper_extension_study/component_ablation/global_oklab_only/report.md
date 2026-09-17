# Global + OKLab Only Component Ablation

## Scope

This is a descriptive seed17 controlled continuation, not a candidate-selection run. It tests the missing Tail-off / OKLab-on cell while leaving the frozen `lambda_OK = 0.059149764` unchanged. It does not modify the frozen final method or `paper_bundle/`.

## Objective and protocol

```text
L = 10 L_msg + 0.5 L_RGB + 0.059149764 L_OKLab
```

- same MBRS backbone;
- same epoch-100 crop-trained Global source checkpoint;
- source model, BatchNorm state, and Adam state restored;
- seed17;
- `RandomCrop(0.3, 1.0)`;
- 20 continuation epochs;
- `lr = 1e-4`, batch size 16, workers 0;
- deterministic settings and one visible GPU;
- no `L_tail` term and no additional inference module;
- fixed validation and project-test manifests/masks; project-test was not used for selection.

Source and final checkpoint provenance is recorded in `source_checkpoint_provenance.md`, `source_checkpoint_sha256.txt`, and `checkpoint_sha256.txt`. The exact resolved training configuration is `config.resolved.json`; the registered input configuration is `config.json`.

## New branch metrics

| Split | PSNR | Top-25 Local PSNR | P95 MSE | P99 MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | BER30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 35.928561 | 35.140262 | 3.297063091e-04 | 3.476742709e-04 | 0.00222304 | 3.691406 | 5.363056 | 0.1108750 |
| Project-test | 35.943842 | 35.207559 | 3.227863646e-04 | 3.422455385e-04 | 0.00236554 | 3.577205 | 5.215556 | 0.1132500 |

The full per-image outputs and exact summary values are in `validation_per_image.csv`, `project_test_per_image.csv`, `validation_metrics.json`, and `project_test_metrics.json`.

## 2×2 component table: project-test

The Global, Hard, and Ours rows below are copied from the authoritative frozen formal project-test table. The Global+OKLab-only row is the new extension branch evaluated under the same fixed evaluator family.

| Tail | OKLab | Method | PSNR | Top-25 Local PSNR | P95 MSE | P99 MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | BER30 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | off | MBRS crop-trained global | 36.263172 | 35.522213 | 3.016531971e-04 | 3.199585360e-04 | 0.00234960 | 3.535857 | 5.159104 | 0.1131250 |
| off | on | Global + OKLab | 35.943842 | 35.207559 | 3.227863646e-04 | 3.422455385e-04 | 0.00236554 | 3.577205 | 5.215556 | 0.1132500 |
| on | off | Hard Local-Tail | 36.442300 | 35.754376 | 2.844551472e-04 | 3.010726967e-04 | 0.00221518 | 3.484790 | 5.075666 | 0.1129375 |
| on | on | Ours | 36.854909 | 36.141895 | 2.619925590e-04 | 2.782322409e-04 | 0.00191711 | 3.254091 | 4.772229 | 0.1131250 |

## Interpretation

The new branch is evaluated descriptively only. Its project-test values do not select a new method, retune OKLab, or replace the frozen component chain. The table supports discussing the contribution of the two components, subject to the exact metric-space and aggregation boundaries in `paper/paper_evidence_audit.md`.

The formal frozen Ours result remains the authoritative paper result. In particular, this extension branch must not be substituted for the frozen Ours row merely because it uses the same OKLab coefficient.

## Reproducibility artifacts

- Training command: `train_command.txt`
- Training log: `train.log`
- Evaluation command: `eval_command.txt`
- Evaluation log: `eval.log`
- Environment and deterministic settings: `environment_info.txt`, `config.resolved.json`
- Final checkpoint: `checkpoint_0020.pth`
- Evaluator: `../evaluate_component.py`
