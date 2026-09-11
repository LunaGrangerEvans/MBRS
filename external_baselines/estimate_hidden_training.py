#!/usr/bin/env python3
"""Estimate HiDDeN training cost without producing a trained model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from hidden_adapter import write_json, load_port_modules


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_training_benchmark.json"),
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable")
    device = torch.device(args.device)
    options, _, _, noiser_module = load_port_modules()
    config = options.HiDDenConfiguration(
        H=128,
        W=128,
        message_length=64,
        encoder_blocks=4,
        encoder_channels=64,
        decoder_blocks=7,
        decoder_channels=64,
        use_discriminator=True,
        use_vgg=False,
        discriminator_blocks=3,
        discriminator_channels=64,
        decoder_loss=1,
        encoder_loss=0.7,
        adversarial_loss=1e-3,
    )
    hidden_module = __import__("model.hidden", fromlist=["Hidden"])
    model = hidden_module.Hidden(config, device, noiser_module.Noiser([], device), None)
    # The old source relied on PyTorch 1.0's floating default for these labels.
    # Setting the values, rather than modifying the upstream file, keeps this
    # benchmark source-preserving on modern PyTorch.
    model.cover_label = 1.0
    model.encoded_label = 0.0
    images = torch.rand(args.batch_size, 3, 128, 128, device=device) * 2 - 1
    messages = torch.randint(0, 2, (args.batch_size, 64), device=device).float()
    for _ in range(2):
        model.train_on_batch([images, messages])
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(args.steps):
        model.train_on_batch([images, messages])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    seconds_per_step = elapsed / args.steps
    estimates = []
    for dataset_size, batch_size, epochs in (
        (10000, 32, 200),
        (10000, 32, 300),
        (10000, 32, 400),
        (10000, 48, 300),
        (10000, 48, 400),
    ):
        steps = ((dataset_size + batch_size - 1) // batch_size) * epochs
        estimates.append(
            {
                "dataset_images": dataset_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "training_steps": steps,
                "training_hours": steps * seconds_per_step / 3600,
            }
        )
    result = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": seconds_per_step,
        "images_per_second": args.batch_size * args.steps / elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "message_length": 64,
        "estimates": estimates,
        "note": "Benchmark only; no checkpoint was saved and no baseline metric was produced.",
    }
    write_json(args.output_json, result)
    print(
        "seconds_per_step={:.6f} peak_memory_mb={}".format(
            seconds_per_step,
            "N/A" if result["peak_memory_bytes"] is None else f"{result['peak_memory_bytes'] / 1024**2:.1f}",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
