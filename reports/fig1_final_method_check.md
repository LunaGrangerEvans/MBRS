# Final Figure 1 method check

The formal frozen g25 evaluation passed, so the framework now defines `Ours` as Hard Local-Tail plus global OKLab regularization.

- Frozen config: `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/config.resolved.json`; SHA-256 `bfe4ccf7efa339e0eacf1bc19dc77aa3f88f8d1dce3dcd105bc62c6e3f314fa6`.
- Output PDF: `/root/workspace/GLX/icassp/MBRS/paper/figures/figure1_final_method.pdf`; SHA-256 `2136e888079a6fb0e3e5ec796947d85b32be368678164ed047c99a5113a556c3`.
- Output PNG: `/root/workspace/GLX/icassp/MBRS/paper/figures/figure1_final_method.png`; SHA-256 `5130e4d9d807b3dd4e1b829a38cf8af9e8f05d70d724c008347208a8c947c398`.
- Branch A: Global RGB — Average distortion control.
- Branch B: Hard Local-Tail — Upper-tail local distortion control; P16/S8, 225 candidates, hard Top-10% (23 patches).
- Branch C: OKLab — Color-aware fidelity control; frozen lambda 0.059149764.
- Objective uses symbolic weights: `L = w_msg L_msg + w_g L_RGB + w_t L_tail + lambda_OK L_OKLab`.
- Local-tail and OKLab are training losses; the diagram explicitly states that no additional inference module is introduced.
