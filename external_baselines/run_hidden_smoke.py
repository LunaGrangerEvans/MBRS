#!/usr/bin/env python3
"""Run one real HiDDeN embed/decode smoke test using a port checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hidden_adapter import (
    MANIFEST_DEFAULT,
    bit_predictions,
    checkpoint_provenance,
    encode,
    load_manifest,
    load_model,
    psnr_per_sample,
    save_rgb,
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
        "--output-dir",
        type=Path,
        default=Path("/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_smoke"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable")
    device = torch.device(args.device)

    manifest = load_manifest(args.manifest)
    model, config, checkpoint, _ = load_model(args.options, args.checkpoint, device)
    if int(config.message_length) > int(manifest["message_length"]):
        raise ValueError("checkpoint message length exceeds manifest message length")
    message = manifest["messages"][0:1, : int(config.message_length)].float()
    image = manifest["images"][0:1].float()
    encoded = encode(model, image, message, device)
    decoded = model.decoder(encoded.to(device)).cpu()
    predicted = bit_predictions(decoded)
    accuracy = float((predicted == message).float().mean())
    psnr = float(psnr_per_sample(encoded, image)[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(args.output_dir / "watermarked.png", encoded)
    result = {
        "status": "passed",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_shape": list(image.shape),
        "message_length_used": int(config.message_length),
        "manifest_message_length": int(manifest["message_length"]),
        "psnr_db": psnr,
        "bit_accuracy": accuracy,
        "decoded_min": float(decoded.min()),
        "decoded_max": float(decoded.max()),
        "watermarked_min_normalized": float(encoded.min()),
        "watermarked_max_normalized": float(encoded.max()),
        "message": message[0].tolist(),
        "decoded_bits": predicted[0].tolist(),
        "provenance": checkpoint_provenance(args.options, args.checkpoint, config, checkpoint),
    }
    write_json(args.output_dir / "result.json", result)
    print(f"status=passed psnr_db={psnr:.6f} bit_accuracy={accuracy:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
