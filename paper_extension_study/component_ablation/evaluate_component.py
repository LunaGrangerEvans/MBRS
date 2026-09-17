#!/usr/bin/env python3
"""Evaluate one extension checkpoint on fixed validation and project-test manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import deltaE_ciede2000, rgb2lab

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.analyze_patch_distortion import extract_patches
from experiments.evaluate_extended_image_quality import evaluate, encode, load_file, load_model
from experiments.losses import patch_mse_per_sample


MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
VALIDATION_MANIFEST = MOUNT / "reports/content_selector/validation_manifest.pt"
VALIDATION_MASKS = MOUNT / "reports/crop_global_oklab/validation_crop_masks.pt"
PROJECT_TEST_MANIFEST = MOUNT / "reports/uniform_eval_manifest.pt"
PROJECT_TEST_EXTENSION = MOUNT / "reports/controlled_crop35_40_manifest.pt"
BER_LEVELS = (100, 70, 50, 40, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def load_checkpoint_model(path: Path, device: torch.device):
    checkpoint = load_file(path, device)
    config = checkpoint["config"]
    from network.Encoder_MP_Decoder import EncoderDecoder

    model = EncoderDecoder(config["H"], config["W"], config["message_length"], ["Identity()"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def load_masks(split: str, batch_size: int) -> dict[str, object]:
    if split == "validation":
        masks = load_file(VALIDATION_MASKS, "cpu")
    else:
        base = load_file(PROJECT_TEST_MANIFEST, "cpu")
        extension = load_file(PROJECT_TEST_EXTENSION, "cpu")
        if extension["base_manifest_seed"] != base["seed"] or extension["samples"] != base["samples"] or extension["batch_size"] != batch_size:
            raise ValueError("project-test crop extension is incompatible with fixed manifest")
        masks = {**base["attack_masks"], **extension["attack_masks"]}
    for ratio in BER_LEVELS:
        key = f"crop_{ratio}"
        if key not in masks:
            raise ValueError(f"missing fixed mask set {key} for {split}")
    return masks


def crop_ber_per_image(model, encoded: torch.Tensor, messages: torch.Tensor, masks: dict[str, object], ratio: int, device: torch.device, batch_size: int) -> np.ndarray:
    errors = np.zeros(len(messages), dtype=np.int64)
    mask_set = masks[f"crop_{ratio}"]
    with torch.no_grad():
        for repeat in range(5):
            for batch_index, start in enumerate(range(0, len(messages), batch_size)):
                stop = min(start + batch_size, len(messages))
                masked = encoded[start:stop].to(device) * mask_set[repeat][batch_index].to(device)
                predicted = model.decoder(masked).gt(0.5).cpu()
                target = messages[start:stop].gt(0.5).cpu()
                errors[start:stop] += (predicted != target).sum(dim=1).numpy()
    return errors.astype(np.float64) / (messages.shape[1] * 5)


def add_tail_and_color_metrics(rows: list[dict[str, object]], encoded: torch.Tensor, images: torch.Tensor) -> None:
    reference = ((images + 1.0) / 2.0).clamp(0.0, 1.0)
    value = ((encoded + 1.0) / 2.0).clamp(0.0, 1.0)
    patch_scores = patch_mse_per_sample(value, reference, 32).detach().cpu().numpy()
    for index, row in enumerate(rows):
        ref = reference[index].permute(1, 2, 0).numpy()
        out = value[index].permute(1, 2, 0).numpy()
        ciede = deltaE_ciede2000(rgb2lab(ref), rgb2lab(out)).astype(np.float64)
        ciede_patch = F.avg_pool2d(
            torch.from_numpy(ciede).unsqueeze(0).unsqueeze(0),
            kernel_size=5,
            stride=1,
        ).flatten().numpy()
        row["patch_mse_p99"] = float(np.percentile(patch_scores[index], 99))
        row["ciede2000_global"] = float(ciede.mean())
        top_count = max(1, int(math.ceil(ciede_patch.size * 0.10)))
        row["ciede2000_top10"] = float(np.sort(ciede_patch)[-top_count:].mean())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields: list[str] = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"samples": len(rows)}
    metric_fields = (
        "global_ssim", "global_ms_ssim", "full_lpips", "top25_local_psnr",
        "patch_mse_p95", "patch_mse_p99", "ciede2000_global", "ciede2000_top10",
        "gini", "ber100", "ber70", "ber50", "ber40", "ber30",
    )
    for field in metric_fields:
        result[field] = float(np.mean([float(row[field]) for row in rows]))
    result["global_mse"] = float(np.mean([float(row["global_mse"]) for row in rows]))
    result["global_psnr"] = float(10.0 * np.log10(1.0 / max(float(result["global_mse"]), 1e-12)))
    return result


def evaluate_split(split: str, model, images: torch.Tensor, messages: torch.Tensor, masks: dict[str, object], device: torch.device, batch_size: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    encoded = encode(model, images, messages, device, batch_size)
    rows = [dict(row) for row in evaluate(encoded, images, device, batch_size)]
    add_tail_and_color_metrics(rows, encoded, images)
    for ratio in BER_LEVELS:
        values = crop_ber_per_image(model, encoded, messages, masks, ratio, device, batch_size)
        for index, row in enumerate(rows):
            row[f"ber{ratio}"] = float(values[index])
        assert abs(float(np.mean(values)) - float(np.mean([row[f"ber{ratio}"] for row in rows]))) < 1e-12
    for row in rows:
        row["split"] = split
    return rows, summarize(rows)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    all_rows: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, dict[str, object]] = {}
    for split, manifest_path in (("validation", VALIDATION_MANIFEST), ("project_test", PROJECT_TEST_MANIFEST)):
        manifest = load_file(manifest_path, "cpu")
        images = manifest["images"].float()
        messages = manifest["messages"].float()
        masks = load_masks(split, args.batch_size)
        rows, summary = evaluate_split(split, model, images, messages, masks, device, args.batch_size)
        all_rows[split] = rows
        summaries[split] = summary
        write_csv(args.output_dir / f"{split}_per_image.csv", rows)
        (args.output_dir / f"{split}_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_config": checkpoint.get("config"),
        "device": str(device),
        "batch_size": args.batch_size,
        "validation_manifest": str(VALIDATION_MANIFEST),
        "project_test_manifest": str(PROJECT_TEST_MANIFEST),
        "project_test_crop_extension": str(PROJECT_TEST_EXTENSION),
        "validation_crop_masks": str(VALIDATION_MASKS),
        "metric_evaluator": "experiments.evaluate_extended_image_quality.evaluate plus fixed CIEDE2000 5x5 stride-1 top10/P99/image-level BER extension",
        "project_test_used_for_selection": False,
        "paper_bundle_unchanged": True,
    }
    (args.output_dir / "evaluation_metadata.json").write_text(json.dumps(json_safe(metadata), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
