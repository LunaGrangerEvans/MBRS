#!/usr/bin/env python3
"""Prepare exact continuation-seed sensitivity run directories."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_extension_study/continuation_seed"
SOURCE = "/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth"
SOURCE_SHA = "1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907"
FROZEN_RUNS = {
    "global": "/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/controlled_seed17_global_continuation/checkpoint_0020.pth",
    "hard": "/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50/checkpoint_0020.pth",
    "ours": "/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/checkpoint_0020.pth",
}

COMMON = {
    "dataset_path": "/mnt/wmcontent/GLX/icassp/MBRS/datasets",
    "H": 128,
    "W": 128,
    "message_length": 64,
    "batch_size": 16,
    "num_workers": 0,
    "epochs": 20,
    "lr": 0.0001,
    "noise_layers": ["RandomCrop(0.3, 1.0)"],
    "message_loss_weight": 10.0,
    "deterministic": True,
    "deterministic_warn_only": False,
    "require_single_gpu": True,
    "save_every": 5,
}
METHOD_CONFIGS = {
    "global": {"global_loss_weight": 1.0, "local_loss_weight": 0.0, "local_loss_mode": "none", "local_patch_size": 16, "local_patch_stride": 8, "local_topk_ratio": 0.1},
    "hard": {"global_loss_weight": 0.5, "local_loss_weight": 0.5, "local_loss_mode": "topk", "local_patch_size": 16, "local_patch_stride": 8, "local_topk_ratio": 0.1},
    "ours": {"global_loss_weight": 0.5, "local_loss_weight": 0.5, "local_loss_mode": "topk", "local_patch_size": 16, "local_patch_stride": 8, "local_topk_ratio": 0.1, "global_oklab_weight": 0.059149764, "global_oklab_version": "linear_srgb_author_matrix_v1"},
}

RNG_AUDIT = """# RNG and Alignment Audit

This run is part of **stochastic continuation-seed sensitivity from a common frozen source checkpoint**, not full multi-seed reproducibility.

- Python, NumPy, Torch CPU, and Torch CUDA seeds are set to the run seed.
- Deterministic cuDNN/algorithm settings are enabled; one visible GPU is used.
- DataLoader uses `num_workers=0`, sorted deterministic filenames, and generators derived as `seed + 200003` for train and `seed + 300007` for validation.
- Message generation uses Torch RNG in the fixed trainer call order.
- `RandomCrop(0.3, 1.0)` uses NumPy RNG in the fixed trainer call order.
- Local-tail selection and global OKLab computation consume no RNG.
- Global, Hard, and Ours use the same source, data pipeline, batch size, and call order within a continuation seed; only the registered objective and continuation seed vary.

If any call-order or source-restoration mismatch is discovered, stop that run and document it rather than continuing.
"""


def main() -> None:
    for method, method_config in METHOD_CONFIGS.items():
        for seed in (17, 23, 42):
            run_dir = OUT / method / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            config = {"name": f"continuation_{method}_seed{seed}", "seed": seed, **COMMON, **method_config}
            (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            (run_dir / "source_checkpoint.md").write_text(
                f"# Common frozen source\n\n- Path: `{SOURCE}`\n- SHA-256: `{SOURCE_SHA}`\n- Source epoch: `100`\n- State restoration: model, BatchNorm, and Adam state; continuation schedule restarts at epoch 1.\n",
                encoding="utf-8",
            )
            (run_dir / "source_checkpoint_sha256.txt").write_text(f"{SOURCE_SHA}  {SOURCE}\n", encoding="utf-8")
            (run_dir / "rng_audit.md").write_text(RNG_AUDIT + f"\nRegistered continuation seed: `{seed}`.\n", encoding="utf-8")
            command = f"CUDA_VISIBLE_DEVICES=<GPU> CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 experiments/train_local_patch.py --config {run_dir}/config.json --output-dir {run_dir} --device cuda --init-checkpoint {SOURCE}"
            (run_dir / "launch_command.template.txt").write_text(command + "\n", encoding="utf-8")
            status = "reused_frozen" if seed == 17 else "planned"
            payload = {"status": status, "method": method, "seed": seed, "source_checkpoint": SOURCE, "source_checkpoint_sha256": SOURCE_SHA}
            if seed == 17:
                payload["existing_frozen_checkpoint"] = FROZEN_RUNS[method]
            (run_dir / "status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
