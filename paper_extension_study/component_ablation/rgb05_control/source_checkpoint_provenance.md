# Source checkpoint provenance

This control initializes from the exact epoch-100 crop-trained Global source referenced by the authoritative paper bundle.

- Path: `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth`
- SHA-256: `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907`
- Source epoch: `100`
- Required state: model/BatchNorm state and Adam optimizer state restored; continuation schedule restarts at epoch 1.
- Registered objective: `L = 10 L_msg + 0.5 L_RGB` with no `L_tail` and no `L_OKLab`.

This branch is a causal-control experiment only. It cannot replace or retune the frozen Ours method.
