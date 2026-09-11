#!/usr/bin/env python3
"""Evaluate the two frozen JPEG/OKLab continuations on the formal manifest."""
import csv
import io
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

if __package__ in {"", None}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate_extended_image_quality import load_file
from network.Encoder_MP_Decoder import EncoderDecoder


ROOT = Path("/root/workspace/GLX/icassp/MBRS")
MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
MANIFEST = MOUNT / "reports/uniform_eval_manifest.pt"
RUNS = {
    "JPEG Global": MOUNT / "experiments/runs/seed17_jpeg_global_continuation/checkpoint_0020.pth",
    "JPEG OKLab-tail": MOUNT / "experiments/runs/seed17_jpeg_oklab_tail_continuation/checkpoint_0020.pth",
}
OUT_REPORT = ROOT / "reports/jpeg_oklab_experiment_summary.md"
OUT_CSV = ROOT / "reports/jpeg_oklab_per_image.csv"
OUT_TABLE = ROOT / "reports/jpeg_oklab_main_table.csv"
FIG_DIR = MOUNT / "visualizations/jpeg_oklab"


def load_model(path, device):
    checkpoint = load_file(path, device)
    config = checkpoint["config"]
    model = EncoderDecoder(config["H"], config["W"], config["message_length"], ["Identity()"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def encode(model, images, messages, device):
    values = []
    with torch.no_grad():
        for start in range(0, len(images), 16):
            values.append(model.encoder(images[start:start + 16].to(device), messages[start:start + 16].to(device)).cpu())
    return torch.cat(values)


def real_jpeg(images, quality):
    outputs = []
    for tensor in images:
        array = (((tensor.clamp(-1, 1).permute(1, 2, 0) + 1) / 2) * 255).round().byte().numpy()
        buffer = io.BytesIO()
        Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        decoded = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32) / 127.5 - 1
        outputs.append(torch.from_numpy(decoded).permute(2, 0, 1))
    return torch.stack(outputs)


def bit_metrics(model, attacked, messages, device):
    values = []
    with torch.no_grad():
        for start in range(0, len(attacked), 16):
            values.append(model.decoder(attacked[start:start + 16].to(device)).cpu())
    decoded = torch.cat(values)
    target = messages.gt(0.5)
    predicted = decoded.gt(0.5)
    accuracy = float((predicted == target).float().mean())
    return {"bit_accuracy": accuracy, "ber": 1.0 - accuracy}


def ciede_metrics(encoded, original, patch_size=5):
    rows = []
    maps = []
    for index in range(len(original)):
        ref = (((original[index] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy())
        wm = (((encoded[index] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy())
        ref_lab = rgb2lab(ref).astype(np.float32)
        wm_lab = rgb2lab(wm).astype(np.float32)
        de = deltaE_ciede2000(ref_lab, wm_lab).astype(np.float32)
        patch_scores = torch.from_numpy(de)[None, None]
        patch_scores = torch.nn.functional.avg_pool2d(patch_scores, patch_size, stride=1).flatten().numpy()
        k = max(1, math.ceil(len(patch_scores) * 0.1))
        rows.append({
            "global_ciede2000_mean": float(de.mean()),
            "patch_ciede2000_mean": float(patch_scores.mean()),
            "patch_ciede2000_p95": float(np.percentile(patch_scores, 95)),
            "top10_patch_ciede2000": float(np.sort(patch_scores)[-k:].mean()),
            "max_patch_ciede2000": float(patch_scores.max()),
            "mean_signed_delta_a": float((wm_lab[..., 1] - ref_lab[..., 1]).mean()),
            "mean_signed_delta_b": float((wm_lab[..., 2] - ref_lab[..., 2]).mean()),
        })
        maps.append(de)
    return rows, maps


def quality_metrics(encoded, original, device):
    from experiments.evaluate_extended_image_quality import evaluate
    return evaluate(encoded, original, device, 16)


def render_figures(encoded_by_method, original, quality_rows, color_maps, robustness):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    methods = list(encoded_by_method)
    for index in [7, 20, 42]:
        fig, axes = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
        original_rgb = (((original[index] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy())
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title("Original")
        for col, method in enumerate(methods, start=1):
            rgb = (((encoded_by_method[method][index] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy())
            axes[0, col].imshow(rgb)
            axes[0, col].set_title(method)
        axes[0, 3].axis("off")
        for col, method in enumerate(methods):
            color_map = color_maps[method][index]
            axes[1, col].imshow(color_map, cmap="magma", vmin=0, vmax=4)
            axes[1, col].set_title(f"{method} CIEDE2000")
            y, x = np.unravel_index(np.argmax(color_map), color_map.shape)
            y = min(y, color_map.shape[0] - 5)
            x = min(x, color_map.shape[1] - 5)
            axes[1, col].add_patch(plt.Rectangle((x, y), 5, 5, fill=False, edgecolor="cyan", linewidth=1.2))
            rgb = (((encoded_by_method[method][index] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy())
            crop_ref = original_rgb[y:y + 5, x:x + 5]
            crop_wm = rgb[y:y + 5, x:x + 5]
            axes[2, col].imshow(np.concatenate([crop_ref, crop_wm], axis=1), interpolation="nearest")
            axes[2, col].set_title(f"{method} worst 5×5\nOriginal | WM")
        axes[1, 2].axis("off")
        axes[1, 3].axis("off")
        axes[2, 0].axis("off")
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"JPEG/OKLab formal comparison — fixed image {index:02d}", fontsize=14)
        fig.savefig(FIG_DIR / f"qualitative_color_tail_{index:02d}.png", dpi=180)
        plt.close(fig)

    summary = {}
    for method in methods:
        rows = [r for r in quality_rows if r["method"] == method]
        summary[method] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key not in {"method", "image_index"}}
    metrics = [
        ("global_ciede2000_mean", "Global CIEDE2000 mean"),
        ("patch_ciede2000_mean", "Patch CIEDE2000 mean"),
        ("patch_ciede2000_p95", "Patch CIEDE2000 P95"),
        ("top10_patch_ciede2000", "Top10 patch CIEDE2000"),
        ("max_patch_ciede2000", "Max patch CIEDE2000"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4), constrained_layout=True)
    for ax, (key, title) in zip(axes, metrics):
        ax.boxplot([[r[key] for r in quality_rows if r["method"] == method] for method in methods], labels=methods)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.2)
    fig.savefig(FIG_DIR / "ciede2000_distribution.png", dpi=180)
    plt.close(fig)

    attacks = ["Identity", "RealJPEG50", "RealJPEG30"]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = np.arange(len(attacks))
    width = 0.35
    for offset, method in zip([-width / 2, width / 2], methods):
        values = [
            next(r["bit_accuracy"] for r in robustness if r["method"] == method and r["attack"] == attack)
            for attack in attacks
        ]
        ax.bar(x + offset, values, width, label=method)
    ax.set_xticks(x, attacks)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Watermark bit accuracy ↑")
    ax.set_title("Identity / RealJPEG50 / RealJPEG30 robustness")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(FIG_DIR / "q50_q30_robustness_comparison.png", dpi=180)
    plt.close(fig)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_file(MANIFEST, "cpu")
    original = manifest["images"].float()
    messages = manifest["messages"].float()
    encoded_by_method = {}
    models = {}
    all_rows = []
    color_maps = {}
    clean_quality = {}
    robustness = []
    for method, checkpoint in RUNS.items():
        model = load_model(checkpoint, device)
        models[method] = model
        encoded = encode(model, original, messages, device)
        encoded_by_method[method] = encoded
        quality = quality_metrics(encoded, original, device)
        colors, maps = ciede_metrics(encoded, original)
        color_maps[method] = maps
        for index, (qrow, crow) in enumerate(zip(quality, colors)):
            all_rows.append({"method": method, "image_index": index, **qrow, **crow})
        clean_quality[method] = quality
        for attack_name, attacked in [("Identity", encoded), ("RealJPEG50", real_jpeg(encoded, 50)), ("RealJPEG30", real_jpeg(encoded, 30))]:
            metric = bit_metrics(model, attacked, messages, device)
            robustness.append({"method": method, "attack": attack_name, **metric})
    render_figures(encoded_by_method, original, all_rows, color_maps, robustness)

    fields = list(all_rows[0])
    with OUT_CSV.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    summary_rows = []
    for method in RUNS:
        rows = [row for row in all_rows if row["method"] == method]
        item = {"method": method}
        for key in fields[2:]:
            item[key] = float(np.mean([row[key] for row in rows]))
        for attack in ["Identity", "RealJPEG50", "RealJPEG30"]:
            item[f"{attack}_accuracy"] = next(r["bit_accuracy"] for r in robustness if r["method"] == method and r["attack"] == attack)
            item[f"{attack}_ber"] = next(r["ber"] for r in robustness if r["method"] == method and r["attack"] == attack)
        summary_rows.append(item)
    with OUT_TABLE.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    robust_path = ROOT / "reports/jpeg_oklab_robustness.csv"
    with robust_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(robustness[0]))
        writer.writeheader()
        writer.writerows(robustness)

    base, proposed = summary_rows
    lower = {"global_ciede2000_mean", "patch_ciede2000_mean", "patch_ciede2000_p95", "top10_patch_ciede2000", "max_patch_ciede2000"}
    improvement = {}
    for key in lower:
        improvement[key] = float(np.mean([r[key] < b[key] for r, b in zip(
            [x for x in all_rows if x["method"] == proposed["method"]],
            [x for x in all_rows if x["method"] == base["method"]],
        )]))
    lines = [
        "# JPEG + OKLab-tail experiment summary",
        "",
        "Two seed17 controlled continuations were trained for 20 epochs from the same crop-trained Global epoch100 checkpoint. No new seed, full-scratch run, Q30 training, or parameter grid was used.",
        "",
        "Training attack: `Combined([Jpeg(50), Identity()])`, using the existing differentiable/simulated JPEG implementation and the existing Identity layer. Evaluation uses clean Identity, real PIL JPEG quality50, and real PIL JPEG quality30.",
        "",
        "## Clean quality and color tail",
        "",
        "| Method | PSNR | SSIM | MS-SSIM | LPIPS | Global CIEDE2000 | Patch mean CIEDE2000 | P95 CIEDE2000 | Top10 CIEDE2000 | Max CIEDE2000 | Mean Δa | Mean Δb |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary_rows:
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {:.8f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
            item["method"], item["global_psnr"], item["global_ssim"], item["global_ms_ssim"], item["full_lpips"],
            item["global_ciede2000_mean"], item["patch_ciede2000_mean"], item["patch_ciede2000_p95"], item["top10_patch_ciede2000"], item["max_patch_ciede2000"], item["mean_signed_delta_a"], item["mean_signed_delta_b"]))
    lines += ["", "## Robustness", "", "| Method | Identity Acc / BER | RealJPEG50 Acc / BER | RealJPEG30 Acc / BER |", "|---|---:|---:|---:|"]
    for item in summary_rows:
        lines.append(f"| {item['method']} | {item['Identity_accuracy']:.6f} / {item['Identity_ber']:.6f} | {item['RealJPEG50_accuracy']:.6f} / {item['RealJPEG50_ber']:.6f} | {item['RealJPEG30_accuracy']:.6f} / {item['RealJPEG30_ber']:.6f} |")
    lines += ["", "## Per-image color-tail improvement", ""]
    for key, value in improvement.items():
        lines.append(f"- {key}: {value * 100:.1f}% of formal images improved under OKLab-tail.")
    lines += ["", "## Decision", "", "The experiment is intentionally evaluated as one controlled direction. Success requires lower local CIEDE2000 tails without a material JPEG50 accuracy loss and without obvious clean-quality degradation. No further alpha, top-ratio, patch-size, or color-space search is authorized in this phase.", "", "Qualitative figures and CIEDE2000 distributions are under `visualizations/jpeg_oklab/`. Machine-readable per-image data are in [jpeg_oklab_per_image.csv](jpeg_oklab_per_image.csv), the summary table is [jpeg_oklab_main_table.csv](jpeg_oklab_main_table.csv), and robustness details are [jpeg_oklab_robustness.csv](jpeg_oklab_robustness.csv)."]
    OUT_REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary": summary_rows, "robustness": robustness, "improvement": improvement}, indent=2))


if __name__ == "__main__":
    main()
