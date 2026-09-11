#!/usr/bin/env python3
"""Guarded HiDDeN evaluation adapter for the MBRS fixed manifest.

The command refuses to emit baseline metrics when the model payload width and
the manifest payload width differ.  This is intentional: the available
checkpoint is 30-bit while the formal MBRS manifest is 64-bit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hidden_adapter import (
    MANIFEST_DEFAULT,
    checkpoint_provenance,
    load_manifest,
    load_model,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--options",
        type=Path,
        default=Path("external_baselines/repos/HiDDeN-pytorch/experiments/no-noise adam-eps-1e-4/options-and-config.pickle"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("external_baselines/repos/HiDDeN-pytorch/experiments/no-noise adam-eps-1e-4/checkpoints/no-noise--epoch-200.pyt"),
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_evaluation_guard.json"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable")

    manifest = load_manifest(args.manifest)
    model, config, checkpoint, _ = load_model(args.options, args.checkpoint, torch.device(args.device))
    model_bits = int(config.message_length)
    manifest_bits = int(manifest["message_length"])
    result = {
        "status": "blocked" if model_bits != manifest_bits else "ready",
        "reason": (
            "checkpoint/model message length does not match the fixed manifest; "
            "no metrics were computed"
            if model_bits != manifest_bits
            else "message length matches; metric execution may proceed"
        ),
        "model_message_length": model_bits,
        "manifest_message_length": manifest_bits,
        "manifest_samples": int(manifest["samples"]),
        "requested_metrics": [
            "PSNR",
            "SSIM",
            "LPIPS",
            "Top25 local PSNR",
            "crop BER at 100/70/50/40/30%",
        ],
        "provenance": checkpoint_provenance(args.options, args.checkpoint, config, checkpoint),
    }
    write_json(args.output_json, result)
    print(result["status"], result["reason"])
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
