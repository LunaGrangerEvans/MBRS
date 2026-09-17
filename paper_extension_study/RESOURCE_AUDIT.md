# Phase 0 Resource and Protocol Audit

This inventory was collected before launching any new extension experiment. It is execution metadata, not paper evidence. Paper claims remain governed only by `paper_bundle/`, `paper/paper_evidence_audit.md`, and `paper/PAPER_RULES.md`.

## Execution entry points inspected

- `experiments/train_local_patch.py`: controlled continuation training entry point with `--config`, `--output-dir`, and `--init-checkpoint`.
- `experiments/evaluate_controlled_seed17.py`: fixed controlled project-test evaluator for compatible continuation branches.
- `experiments/run_matched_psnr_comparison.py`: validation-first source-resolution residual-strength calibration and project-test comparison.
- `experiments/extend_external_matched_psnr.py`: external Figure 2 matched-display calibration/renderer; not used to alter the frozen Figure 2.

These repository files are tooling references only. They are not authoritative evidence sources for the paper.

## Frozen assets located for execution

The exact frozen source/checkpoint paths and hashes are recorded in the bundled evidence. The seed17 epoch-100 crop-trained Global source, seed17 Global continuation, and seed17 Ours checkpoints are present and readable.

The bundle-referenced fixed validation/project-test manifests are present. The bundle-referenced expanded per-image project-test CSV is present at the workspace path named by the bundled formal report.

## Seed-source gate

Exact epoch-100 source checkpoints were found for seed17, seed29, and seed41. No exact epoch-100 source checkpoint was found for seed23 or seed42. Therefore the multi-seed phase is blocked under the pre-registered same-source policy; no seed23/42 substitute will be used.

## Compute and storage

- GPUs: 2 × NVIDIA A800-SXM4-80GB.
- Observed free GPU memory: approximately 79.7 GiB per GPU at audit time.
- Workspace filesystem: approximately 467 GiB available.
- Workspace project tree: approximately 43 MiB.
- Host experiment/data mount: approximately 197 GiB currently used; large generated outputs must remain on the host mount according to project storage rules.

## Safety constraints

- Do not modify `paper_bundle/`.
- Do not modify the Prism manuscript or existing paper assets.
- Do not overwrite historical output directories.
- Write all new analysis outputs and logs under `paper_extension_study/`.
- Do not evaluate project-test data before phase-specific choices are frozen.
