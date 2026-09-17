# Global-RGB0.5-Control

## Scope

This is a registered causal control, not a candidate final method. It tests whether changing only the global RGB loss coefficient can explain the observed paper improvement. It does not modify the frozen final method or `paper_bundle/`.

## Objective and protocol

```text
L = 10 L_msg + 0.5 L_RGB
```

- same MBRS backbone and epoch-100 crop-trained Global source;
- source model, BatchNorm state, and Adam state restored;
- seed17;
- `RandomCrop(0.3, 1.0)`;
- 20 continuation epochs;
- `lr = 1e-4`, batch size 16, workers 0;
- deterministic settings and one visible GPU;
- no local-tail term, no OKLab term, and no additional inference module;
- fixed validation and project-test manifests/masks; project-test was not used for selection.

Source and final checkpoint hashes are in `source_checkpoint_provenance.md`, `source_checkpoint_sha256.txt`, and `checkpoint_sha256.txt`. The exact configurations are `config.json` and `config.resolved.json`.

## Metrics

| Split | PSNR | Top-25 Local PSNR | P95 MSE | P99 MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | BER30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 35.160083 | 34.401580 | 3.861125620e-04 | 4.047923086e-04 | 0.00251539 | 4.156380 | 6.022150 | 0.1113125 |
| Project-test | 35.164486 | 34.447721 | 3.813118118e-04 | 4.026092408e-04 | 0.00261828 | 4.027174 | 5.858652 | 0.1130625 |

Full per-image rows and exact summaries are in `validation_per_image.csv`, `project_test_per_image.csv`, `validation_metrics.json`, and `project_test_metrics.json`.

## Interpretation boundary

This control is descriptive evidence about loss-weight confounding. It is not a new final method and must not be promoted, selected, or used to retune the frozen method. Its results remain supplementary until explicitly approved.

## Reproducibility artifacts

- Training command: `train_command.txt`
- Training log: `train.log`
- Evaluation command: `eval_command.txt`
- Evaluation log: `eval.log`
- Environment and deterministic settings: `environment_info.txt`, `config.resolved.json`
- Final checkpoint: `checkpoint_0020.pth`
- Evaluator: `../evaluate_component.py`
