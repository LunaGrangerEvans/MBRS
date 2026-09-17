# Continuation-Seed Sensitivity Launch Gate

## Gate result

**PASS — continuation-seed sensitivity may launch under the explicitly registered protocol.**

This gate authorizes only the requested experiment named **stochastic continuation-seed sensitivity from a common frozen source checkpoint**. It is not full multi-seed reproducibility and does not alter the frozen paper method.

## Checks

| Gate | Result | Evidence |
|---|---|---|
| Factorial protocol audit | PASS | `factorial_component/PROTOCOL_AUDIT.md` reports no scientifically relevant parity failure. |
| Factorial bootstrap | PASS | 20,000 paired image-level resamples with corrected paper PSNR estimator completed under `factorial_component/bootstrap/`. |
| Overhead benchmark | PASS | 20 warm-ups and 100 measured iterations completed; results are under `overhead/`. |
| Frozen-method validity contradiction | PASS | Controls and interaction analysis add limits but do not invalidate the frozen method. |
| Disk space | PASS | Sufficient workspace disk remains for six new epoch-20 checkpoints and logs. |
| GPU availability | PASS | A800 GPUs are available; one visible GPU will be assigned per run. |
| Common source checkpoint | PASS | Seed17 epoch-100 crop-trained Global source exists with audited SHA-256 `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907`. |
| RNG/protocol implementability | PASS | Existing trainer seeds Python, NumPy, Torch CPU/CUDA, and DataLoader generators; local-tail/OKLab losses consume no RNG; all runs use identical registered configs except continuation seed. |

## Launch constraints

- Use the same source checkpoint for all Global, Hard, and Ours runs at seeds17, 23, and42.
- Reuse valid frozen seed17 outputs where exact protocol parity is confirmed; do not rerun seed17 unnecessarily.
- Launch only new seed23 and seed42 runs for all three methods.
- Do not construct seed-specific source checkpoints.
- Do not tune or select any hyperparameter, checkpoint, seed, or result using project-test data.
- Stop any specific run if RNG call order or source restoration cannot be verified exactly.
- Keep all new runs under `paper_extension_study/continuation_seed/` and preserve logs, source hashes, configs, checkpoints, metrics, and RNG audits.

## Frozen method configurations

- Global: `10 L_msg + 1.0 L_RGB`.
- Hard: `10 L_msg + 0.5 L_RGB + 0.5 L_tail` with P16/S8 hard Top-10%, 225 candidates, 23 selected patches.
- Ours: `10 L_msg + 0.5 L_RGB + 0.5 L_tail + 0.059149764 L_OKLab` with the same local-tail definition.
- All: 20 epochs, `lr = 1e-4`, batch size 16, workers 0, `RandomCrop(0.3, 1.0)`, restored Adam/BatchNorm/model state, deterministic settings.
