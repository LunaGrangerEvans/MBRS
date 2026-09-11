#!/usr/bin/env python3
"""Evaluate saved external-baseline outputs with the frozen extended metric definitions."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in {"", None}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate_extended_image_quality import evaluate, load_file
from experiments.losses import patch_mse_per_sample


ROOT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
MANIFEST = ROOT / "reports/uniform_eval_manifest.pt"
OUTPUT_ROOT = ROOT / "external_baselines/outputs"
OUT = ROOT / "reports/external_baseline_metrics"


def energy_metrics(scores):
    x = np.asarray(scores, dtype=np.float64)
    n = x.shape[1]
    k = max(1, math.ceil(n * 0.10))
    sorted_x = np.sort(x, axis=1)
    total = x.sum(axis=1)
    mean = x.mean(axis=1)
    return {
        "gini": ((2 * np.arange(1, n + 1) - n - 1) * sorted_x).sum(axis=1) / (n * total),
        "cv": x.std(axis=1) / np.maximum(mean, 1e-12),
        "top10_over_mean": sorted_x[:, -k:].mean(axis=1) / np.maximum(mean, 1e-12),
        "top10_energy_share": sorted_x[:, -k:].sum(axis=1) / np.maximum(total, 1e-12),
    }


def load_outputs(output_dir, count):
    values = []
    for index in range(count):
        from PIL import Image
        image = np.asarray(Image.open(output_dir / f"image_{index:03d}.png").convert("RGB"), dtype=np.float32)
        values.append(torch.from_numpy(image).permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = torch.device(args.device)
    manifest = load_file(MANIFEST, "cpu")
    images = manifest["images"].float()
    rows = []
    summaries = []
    for mode in ("Q", "P"):
        output_dir = OUTPUT_ROOT / f"trustmark_{mode}"
        encoded = load_outputs(output_dir, len(images))
        base_rows = evaluate(encoded, images, device, args.batch_size)
        patch_scores = patch_mse_per_sample(encoded, images, 32).numpy()
        concentration = energy_metrics(patch_scores)
        for index, row in enumerate(base_rows):
            row = {"method": f"TrustMark {mode}", **row}
            row.update({key: float(values[index]) for key, values in concentration.items()})
            row["top25_local_psnr"] = 10 * math.log10(1.0 / max(row["patch_mse_top25"], 1e-12))
            rows.append(row)
        cur = [row for row in rows if row["method"] == f"TrustMark {mode}"]
        keys = [
            "global_psnr", "global_ssim", "global_ms_ssim", "full_lpips",
            "top25_local_psnr", "patch_mse_p95", "gini", "cv",
            "top10_over_mean", "top10_energy_share", "patch_ssim_bottom10_mean",
            "patch_lpips_top10_mean",
        ]
        summary = {"method": f"TrustMark {mode}"}
        for key in keys:
            values = np.asarray([item[key] for item in cur], dtype=float)
            summary[key] = float(np.nanmean(values))
        summaries.append(summary)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUT / "provenance.json").write_text(json.dumps({
        "manifest": str(MANIFEST),
        "manifest_sha256": __import__("hashlib").sha256(MANIFEST.read_bytes()).hexdigest(),
        "input": "saved TrustMark Q/P uint8 PNG outputs",
        "metrics": "same frozen extended definitions: RGB [0,1], SSIM win7, 3-scale MS-SSIM [0.3,0.3,0.4], native32 patch LPIPS/SSIM, native32 MSE concentration",
        "dists": "N/A; unavailable",
        "gmsd": "N/A; unavailable",
    }, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
