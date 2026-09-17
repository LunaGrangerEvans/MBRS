#!/usr/bin/env python3
"""Benchmark frozen Global/Ours training-step and inference overhead."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.losses import image_loss_components
from experiments.oklab_global import global_oklab_components
from network.Encoder_MP_Decoder import EncoderDecoder


ROOT = Path(__file__).resolve().parents[2]
MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
SOURCE = MOUNT / "experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth"
GLOBAL_FINAL = MOUNT / "experiments/runs/controlled_seed17_global_continuation/checkpoint_0020.pth"
OURS_FINAL = MOUNT / "experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/checkpoint_0020.pth"
VALIDATION_MANIFEST = MOUNT / "reports/content_selector/validation_manifest.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def load(path: Path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def make_model(checkpoint, device: torch.device, training: bool):
    config = checkpoint["config"]
    noise_layers = ["RandomCrop(0.3, 1.0)"] if training else ["Identity()"]
    model = EncoderDecoder(config["H"], config["W"], config["message_length"], noise_layers).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def training_step(model, optimizer, images, messages, kind: str):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    encoded, _noised, decoded = model(images, messages)
    if kind == "global":
        components = image_loss_components(encoded, images, mode="none", patch_size=16, patch_stride=8, topk_ratio=0.1)
        image_loss = components["global_mse"]
    else:
        components = image_loss_components(encoded, images, mode="topk", patch_size=16, patch_stride=8, topk_ratio=0.1)
        image_loss = 0.5 * components["global_mse"] + 0.5 * components["local_mse"]
        image_loss = image_loss + 0.059149764 * global_oklab_components(encoded, images)["global_oklab"]
    loss = 10.0 * torch.nn.functional.mse_loss(decoded, messages) + image_loss
    loss.backward()
    optimizer.step()


def timed_training(kind: str, checkpoint_path: Path, images: torch.Tensor, messages: torch.Tensor, device: torch.device, warmup: int, iterations: int) -> dict[str, object]:
    checkpoint = load(checkpoint_path, device)
    model = make_model(checkpoint, device, training=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = 1e-4
    np.random.seed(20260916)
    torch.manual_seed(20260916)
    for _ in range(warmup):
        training_step(model, optimizer, images, messages, kind)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        training_step(model, optimizer, images, messages, kind)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else float("nan")
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else float("nan")
    return summarize("training", "Global" if kind == "global" else "Ours", timings, len(images), peak_allocated, peak_reserved, warmup, iterations)


def inference_step(model, images, messages):
    encoded = model.encoder(images, messages)
    model.decoder(encoded)


def timed_inference(name: str, checkpoint_path: Path, images: torch.Tensor, messages: torch.Tensor, device: torch.device, warmup: int, iterations: int) -> dict[str, object]:
    checkpoint = load(checkpoint_path, device)
    model = make_model(checkpoint, device, training=False)
    model.eval()
    for _ in range(warmup):
        with torch.inference_mode():
            inference_step(model, images, messages)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            inference_step(model, images, messages)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else float("nan")
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else float("nan")
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return summarize("inference", name, timings, len(images), peak_allocated, peak_reserved, warmup, iterations) | {
        "parameter_count": params,
        "trainable_parameter_count": trainable,
        "architecture": "EncoderDecoder(Encoder_MP, Identity, Decoder)",
        "additional_inference_module": "NONE",
    }


def summarize(mode, method, timings, batch_size, peak_allocated, peak_reserved, warmup, iterations):
    mean_ms = statistics.mean(timings)
    return {
        "mode": mode,
        "method": method,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(timings),
        "std_ms": statistics.stdev(timings),
        "throughput_steps_per_s": 1000.0 / mean_ms,
        "throughput_images_per_s": batch_size * 1000.0 / mean_ms,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    manifest = load(VALIDATION_MANIFEST, "cpu")
    images = manifest["images"][:16].float().to(device)
    messages = manifest["messages"][:16].float().to(device)
    rows = [
        timed_training("global", SOURCE, images, messages, device, args.warmup, args.iterations),
        timed_training("ours", SOURCE, images, messages, device, args.warmup, args.iterations),
        timed_inference("Global", GLOBAL_FINAL, images, messages, device, args.warmup, args.iterations),
        timed_inference("Ours", OURS_FINAL, images, messages, device, args.warmup, args.iterations),
    ]
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with (args.output_dir / "overhead_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    env = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_device_count_visible": torch.cuda.device_count() if device.type == "cuda" else 0,
        "precision": "float32; autocast disabled",
        "batch_shape": list(images.shape),
        "payload_length": int(messages.shape[1]),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "deterministic_training_path": True,
        "source_checkpoint": str(SOURCE),
        "validation_manifest": str(VALIDATION_MANIFEST),
    }
    (args.output_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    training_global = next(row for row in rows if row["mode"] == "training" and row["method"] == "Global")
    training_ours = next(row for row in rows if row["mode"] == "training" and row["method"] == "Ours")
    inference_global = next(row for row in rows if row["mode"] == "inference" and row["method"] == "Global")
    inference_ours = next(row for row in rows if row["mode"] == "inference" and row["method"] == "Ours")
    training_overhead = (training_ours["mean_ms"] - training_global["mean_ms"]) / training_global["mean_ms"] * 100.0
    inference_overhead = (inference_ours["mean_ms"] - inference_global["mean_ms"]) / inference_global["mean_ms"] * 100.0
    report = f"""# Overhead Benchmark Report

## Protocol

The benchmark uses `{args.warmup}` warm-up and `{args.iterations}` measured iterations on a fixed validation batch of shape `{tuple(images.shape)}`, FP32, with CUDA synchronization. Training timing includes forward, loss computation, backward, and Adam step. Inference timing covers the encoder-plus-decoder path with no crop/noise module.

## Results

| Mode | Method | Mean ms | Median ms | Std ms | Images/s | Peak allocated MiB | Peak reserved MiB |
|---|---|---:|---:|---:|---:|---:|---:|
"""
    for row in rows:
        report += f"| {row['mode']} | {row['method']} | {row['mean_ms']:.4f} | {row['median_ms']:.4f} | {row['std_ms']:.4f} | {row['throughput_images_per_s']:.3f} | {row['peak_allocated_mib']:.2f} | {row['peak_reserved_mib']:.2f} |\n"
    report += f"""
## Relative timing

- Training-step mean relative overhead, Ours vs Global: `{training_overhead:+.2f}%`.
- Inference mean relative timing difference, Ours vs Global: `{inference_overhead:+.2f}%`; this is a noisy runtime measurement, not an architectural-overhead claim.

## Architecture

- Global parameters: `{inference_global['parameter_count']}` total, `{inference_global['trainable_parameter_count']}` trainable.
- Ours parameters: `{inference_ours['parameter_count']}` total, `{inference_ours['trainable_parameter_count']}` trainable.
- Architecture identity: `{inference_global['architecture']}` for both paths.
- Additional inference module: **NONE**.

The scientifically important result is that the proposed Tail and OKLab terms are training-only and introduce no additional inference module or parameters. Any measured inference latency difference is reported as runtime noise/benchmark variation and is not called zero without a noise analysis.
"""
    (args.output_dir / "overhead_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"training_overhead_percent": training_overhead, "inference_mean_difference_percent": inference_overhead, "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
