#!/usr/bin/env python3
"""Evaluate a pre-registered residual-strength grid on validation and project-test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.analyze_patch_distortion import extract_patches
from experiments.losses import patch_mse_per_sample
from network.Encoder_MP_Decoder import EncoderDecoder


MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
VALIDATION_MANIFEST = MOUNT / "reports/content_selector/validation_manifest.pt"
VALIDATION_MASKS = MOUNT / "reports/crop_global_oklab/validation_crop_masks.pt"
PROJECT_TEST_MANIFEST = MOUNT / "reports/uniform_eval_manifest.pt"
PROJECT_TEST_EXTENSION = MOUNT / "reports/controlled_crop35_40_manifest.pt"
GLOBAL_CHECKPOINT = MOUNT / "experiments/runs/controlled_seed17_global_continuation/checkpoint_0020.pth"
OURS_CHECKPOINT = MOUNT / "experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/checkpoint_0020.pth"
RATIOS = (100, 70, 50, 40, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def load_file(path: Path, device="cpu"):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: Path, device: torch.device):
    checkpoint = load_file(path, device)
    config = checkpoint["config"]
    model = EncoderDecoder(config["H"], config["W"], config["message_length"], ["Identity()"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def encode(model, images: torch.Tensor, messages: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            stop = min(start + batch_size, len(images))
            encoded.append(model.encoder(images[start:stop].to(device), messages[start:stop].to(device)).cpu())
    return torch.cat(encoded)


def rgb(value: torch.Tensor) -> torch.Tensor:
    return ((value.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def parse_grid(path: Path) -> list[float]:
    values = [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("alpha grid must be non-empty and lie in [0,1]")
    if values != sorted(set(values)):
        raise ValueError("alpha grid must be sorted and unique")
    return values


def load_split(split: str):
    if split == "validation":
        manifest = load_file(VALIDATION_MANIFEST)
        masks = load_file(VALIDATION_MASKS)
        mask_paths = [VALIDATION_MASKS]
    else:
        manifest = load_file(PROJECT_TEST_MANIFEST)
        extension = load_file(PROJECT_TEST_EXTENSION)
        if extension["base_manifest_seed"] != manifest["seed"] or extension["samples"] != manifest["samples"] or extension["batch_size"] != 16:
            raise ValueError("project-test crop extension is incompatible")
        masks = {**manifest["attack_masks"], **extension["attack_masks"]}
        mask_paths = [PROJECT_TEST_MANIFEST, PROJECT_TEST_EXTENSION]
    if tuple(manifest["images"].shape) != (50, 3, 128, 128) or tuple(manifest["messages"].shape) != (50, 64):
        raise ValueError(f"{split} manifest dimensions do not match frozen contract")
    for ratio in RATIOS:
        key = f"crop_{ratio}"
        if key not in masks or len(masks[key]) != 5:
            raise ValueError(f"missing/incompatible {split} {key} masks")
    return manifest, masks, mask_paths


def lpips_values(metric, values: torch.Tensor, reference: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    results = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            results.append(metric((values[start:stop] * 2.0 - 1.0).to(device), (reference[start:stop] * 2.0 - 1.0).to(device)).flatten().cpu().numpy())
    return np.concatenate(results).astype(np.float64)


def quality_summary(values: torch.Tensor, reference: torch.Tensor, metric, device: torch.device, batch_size: int) -> dict[str, float]:
    patch_scores = patch_mse_per_sample(values, reference, 32, 32).detach().cpu().numpy()
    global_mse = (values - reference).square().mean(dim=(1, 2, 3)).numpy()
    top25 = max(1, int(math.ceil(patch_scores.shape[1] * 0.25)))
    ordered = np.sort(patch_scores, axis=1)
    local_psnr = 10.0 * np.log10(1.0 / np.maximum(ordered[:, -top25:].mean(axis=1), 1e-12))
    p95 = np.percentile(patch_scores, 95, axis=1)
    p99 = np.percentile(patch_scores, 99, axis=1)
    ciede = []
    for index in range(len(values)):
        ref = reference[index].permute(1, 2, 0).numpy()
        out = values[index].permute(1, 2, 0).numpy()
        ciede.append(float(deltaE_ciede2000(rgb2lab(ref), rgb2lab(out)).mean()))
    return {
        "global_mse": float(global_mse.mean()),
        "psnr": float(10.0 * np.log10(1.0 / max(float(global_mse.mean()), 1e-12))),
        "top25_local_psnr": float(local_psnr.mean()),
        "p95": float(p95.mean()),
        "p99": float(p99.mean()),
        "lpips": float(lpips_values(metric, values, reference, device, batch_size).mean()),
        "ciede2000_global": float(np.mean(ciede)),
    }


def ber_summary(model, values: torch.Tensor, messages: torch.Tensor, masks: dict[str, object], device: torch.device, batch_size: int) -> dict[str, float]:
    result = {}
    with torch.inference_mode():
        for ratio in RATIOS:
            errors = np.zeros(len(messages), dtype=np.int64)
            for repeat in range(5):
                for batch_index, start in enumerate(range(0, len(messages), batch_size)):
                    stop = min(start + batch_size, len(messages))
                    masked = values[start:stop].to(device) * masks[f"crop_{ratio}"][repeat][batch_index].to(device)
                    predicted = model.decoder(masked).gt(0.5).cpu()
                    target = messages[start:stop].gt(0.5).cpu()
                    errors[start:stop] += (predicted != target).sum(dim=1).numpy()
            result[f"ber{ratio}"] = float(np.mean(errors / (messages.shape[1] * 5)))
    return result


def evaluate_split(split: str, models: dict[str, object], grid: list[float], device: torch.device, batch_size: int) -> list[dict[str, object]]:
    manifest, masks, _ = load_split(split)
    images = manifest["images"].float()
    messages = manifest["messages"].float()
    encoded = {name: encode(model, images, messages, device, batch_size) for name, model in models.items()}
    import lpips

    metric = lpips.LPIPS(net="alex", version="0.1", spatial=False, verbose=False).to(device).eval()
    rows = []
    reference = rgb(images)
    for name, model in models.items():
        for alpha in grid:
            blended = rgb(images + float(alpha) * (encoded[name] - images))
            row = {"split": split, "method": name, "alpha": alpha}
            row.update(quality_summary(blended, reference, metric, device, batch_size))
            row.update(ber_summary(model, images + float(alpha) * (encoded[name] - images), messages, masks, device, batch_size))
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_figures(rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("ber30", "BER30", "lower"),
        ("p95", "P95 MSE", "lower"),
        ("top25_local_psnr", "Top-25 Local PSNR", "higher"),
        ("lpips", "LPIPS", "lower"),
        ("ciede2000_global", "Global CIEDE2000", "lower"),
    )
    for field, ylabel, _direction in metrics:
        figure, axis = plt.subplots(figsize=(6.0, 4.2))
        for method in ("MBRS crop-trained global", "Ours"):
            subset = sorted((row for row in rows if row["method"] == method), key=lambda row: float(row["psnr"]))
            axis.plot([row["psnr"] for row in subset], [row[field] for row in subset], marker="o", linewidth=1.8, label=method)
        axis.set_xlabel("PSNR (dB)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(out_dir / f"{field}_vs_psnr.png", dpi=220)
        figure.savefig(out_dir / f"{field}_vs_psnr.pdf")
        plt.close(figure)


def write_report(path: Path, grid: list[float], rows: list[dict[str, object]]) -> None:
    lines = [
        "# Fidelity–Robustness Operating Curve",
        "",
        "This is a pre-registered residual-strength analysis of the frozen MBRS crop-trained Global and frozen Ours checkpoints. No model was retrained. The alpha grid was frozen before project-test evaluation and is identical for validation and project-test.",
        "",
        "- Residual scaling: `x_alpha = x + alpha (x_hat - x)` in the normalized tensor, followed by clipped-RGB quality evaluation.",
        f"- Frozen alpha grid: `{', '.join(f'{value:.3f}' for value in grid)}`.",
        "- Validation is used for calibration/context; project-test is evaluated only after the grid is fixed.",
        "- Curves use identical axes for Global and Ours within each figure.",
        "- No dominance claim is made by this report; the curves are descriptive across the registered range.",
        "",
        "## Project-test curve values",
        "",
        "| Method | α | PSNR | BER30 | P95 MSE | Top-25 Local PSNR | LPIPS | Global CIEDE2000 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted((r for r in rows if r["split"] == "project_test"), key=lambda r: (r["method"], float(r["alpha"]))):
        lines.append(f"| {row['method']} | {float(row['alpha']):.3f} | {float(row['psnr']):.6f} | {float(row['ber30']):.7f} | {float(row['p95']):.9e} | {float(row['top25_local_psnr']):.6f} | {float(row['lpips']):.8f} | {float(row['ciede2000_global']):.6f} |")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The operating curve tests behavior over a fixed residual-strength range. It does not retune the frozen method, select a project-test point, or replace the natural frozen or internal matched-PSNR paper results. Figure 2 remains the separate external 512×512 display-space comparison.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = parse_grid(args.alpha_grid)
    (args.output_dir / "alpha_grid.txt").write_text(args.alpha_grid.read_text(encoding="utf-8"), encoding="utf-8")
    device = torch.device(args.device)
    models = {
        "MBRS crop-trained global": load_model(GLOBAL_CHECKPOINT, device),
        "Ours": load_model(OURS_CHECKPOINT, device),
    }
    validation_rows = evaluate_split("validation", models, grid, device, args.batch_size)
    write_csv(args.output_dir / "validation_curve.csv", validation_rows)
    project_rows = evaluate_split("project_test", models, grid, device, args.batch_size)
    write_csv(args.output_dir / "project_test_curve.csv", project_rows)
    all_rows = validation_rows + project_rows
    write_figures(all_rows, args.output_dir / "figures")
    provenance = {
        "alpha_grid_file": str(args.alpha_grid),
        "alpha_grid_sha256": sha256(args.alpha_grid),
        "alpha_values": grid,
        "validation_manifest": str(VALIDATION_MANIFEST),
        "validation_masks": str(VALIDATION_MASKS),
        "project_test_manifest": str(PROJECT_TEST_MANIFEST),
        "project_test_crop_extension": str(PROJECT_TEST_EXTENSION),
        "checkpoints": {"MBRS crop-trained global": str(GLOBAL_CHECKPOINT), "Ours": str(OURS_CHECKPOINT)},
        "project_test_used_to_choose_alpha_grid": False,
        "retraining": False,
        "quality_space": "native 128x128 clipped RGB",
        "ber_unit": "image-level aggregate of fixed 5-repeat 64-bit masks",
    }
    (args.output_dir / "curve_metadata.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir / "curve_report.md", grid, all_rows)
    print(json.dumps({"validation_rows": len(validation_rows), "project_test_rows": len(project_rows), "alpha_grid": grid}, indent=2), flush=True)


if __name__ == "__main__":
    main()
