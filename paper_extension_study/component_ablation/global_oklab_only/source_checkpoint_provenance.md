# Source checkpoint provenance

This branch must initialize from the exact epoch-100 crop-trained Global source referenced by the authoritative paper bundle.

- Path: `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth`
- SHA-256: `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907`
- Source epoch: `100`
- Required state: model/BatchNorm state and Adam optimizer state restored; continuation schedule restarts at epoch 1.
- Paper authority: `paper_bundle/evidence/final_method_frozen_config.md` and `paper_bundle/evidence/controlled_continuation_protocol.md`.

This branch is a descriptive component ablation only. It cannot replace or retune the frozen Ours method.
