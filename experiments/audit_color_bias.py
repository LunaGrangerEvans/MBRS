#!/usr/bin/env python3
"""Analysis-only CIEDE2000/OKLab color-tail audit for frozen outputs."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import deltaE_ciede2000, rgb2lab


ROOT = Path("/root/workspace/GLX/icassp/MBRS")
MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
MANIFEST = MOUNT / "reports/uniform_eval_manifest.pt"
OUTPUT = MOUNT / "reports/perceptual_evidence_audit/final_outputs_and_patch_metrics.pt"
REPORT = ROOT / "reports/color_bias_audit.md"
CSV = ROOT / "reports/color_bias_audit_per_image.csv"
FIG_DIR = MOUNT / "visualizations/color_bias_audit"
PATCH = 5


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def tensor_rgb(value):
    return ((value.float() + 1.0) / 2.0).clamp(0, 1)


def summarize(method, encoded, original):
    encoded_rgb = tensor_rgb(encoded)
    original_rgb = tensor_rgb(original)
    rows = []
    all_maps = []
    for index in range(len(original_rgb)):
        ref = original_rgb[index].permute(1, 2, 0).numpy()
        wm = encoded_rgb[index].permute(1, 2, 0).numpy()
        ref_lab = rgb2lab(ref).astype(np.float32)
        wm_lab = rgb2lab(wm).astype(np.float32)
        de = deltaE_ciede2000(ref_lab, wm_lab).astype(np.float32)
        delta_lab = wm_lab - ref_lab
        patch_scores = F.avg_pool2d(torch.from_numpy(de)[None, None], PATCH, stride=1).flatten().numpy()
        k = max(1, math.ceil(len(patch_scores) * 0.10))
        order = np.argsort(patch_scores)
        top = patch_scores[order[-k:]]
        worst = int(order[-1])
        width = 128 - PATCH + 1
        y, x = divmod(worst, width)
        rows.append({
            "method": method, "image_index": index,
            "global_ciede2000_mean": float(de.mean()),
            "patch_ciede2000_mean": float(patch_scores.mean()),
            "patch_ciede2000_p95": float(np.percentile(patch_scores, 95)),
            "top10_patch_ciede2000": float(top.mean()),
            "max_patch_ciede2000": float(patch_scores.max()),
            "mean_signed_delta_a": float(delta_lab[..., 1].mean()),
            "mean_signed_delta_b": float(delta_lab[..., 2].mean()),
            "worst_patch_x": int(x), "worst_patch_y": int(y), "worst_patch_size": PATCH,
        })
        all_maps.append({"de": de, "ref": ref, "wm": wm, "lab": wm_lab, "ref_lab": ref_lab,
                         "worst": (x, y), "score": float(patch_scores.max())})
    return rows, all_maps


def render_examples(method_maps):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fixed = [7, 20, 42]
    for index in fixed:
        fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
        methods = list(method_maps)
        global_item = method_maps[methods[0]][index]
        hard_item = method_maps[methods[1]][index]
        axes[0, 0].imshow(global_item["ref"])
        axes[0, 0].set_title("Original")
        axes[0, 1].imshow(global_item["wm"])
        axes[0, 1].set_title("Global WM")
        axes[0, 2].imshow(hard_item["wm"])
        axes[0, 2].set_title("Hard Top10 WM")
        axes[0, 3].axis("off")
        for column, (method, item) in enumerate([(methods[0], global_item), (methods[1], hard_item)]):
            axes[1, column].imshow(item["de"], cmap="magma", vmin=0, vmax=4)
            axes[1, column].set_title(f"{method} CIEDE2000 heatmap")
            x, y = item["worst"]
            axes[1, column].add_patch(plt.Rectangle((x, y), PATCH, PATCH, fill=False, edgecolor="cyan", linewidth=1.2))
            crop_ref = item["ref"][y:y + PATCH, x:x + PATCH]
            crop_wm = item["wm"][y:y + PATCH, x:x + PATCH]
            zoom_ax = axes[1, 2 if column == 0 else 3]
            zoom_ax.imshow(np.concatenate([crop_ref, crop_wm], axis=1), interpolation="nearest")
            zoom_ax.set_title(f"{method} worst 5×5: {item['score']:.3f}\nOriginal | WM")
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"Color-bias audit, fixed formal image {index:02d}; 5×5 sliding patches", fontsize=14)
        fig.savefig(FIG_DIR / f"color_bias_example_{index:02d}.png", dpi=180)
        plt.close(fig)


def main():
    manifest = load(MANIFEST)
    artifacts = load(OUTPUT)
    original = manifest["images"].float()
    mapping = {
        "Global continuation": "整图MSE续训（权重1，无局部项）",
        "Hard Patch16 stride8 Top10": "Patch16/stride8/Top10原始MSE（整图/局部权重0.5/0.5）",
    }
    rows, maps = [], {}
    for method, key in mapping.items():
        current_rows, current_maps = summarize(method, artifacts[key]["encoded"], original)
        rows.extend(current_rows)
        maps[method] = current_maps
    with CSV.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render_examples(maps)
    summary = []
    for method in mapping:
        current = [row for row in rows if row["method"] == method]
        item = {"method": method}
        for key in current[0]:
            if key not in {"method", "image_index", "worst_patch_x", "worst_patch_y", "worst_patch_size"}:
                item[key] = float(np.mean([row[key] for row in current]))
        summary.append(item)
    base, hard = summary
    lines = [
        "# Color-bias Go/No-Go audit",
        "",
        "Analysis-only audit of the frozen Global continuation and Hard Patch16/stride8/Top10 outputs on all 50 fixed formal test images. No training was run and no image was selected based on its result.",
        "",
        "Protocol: RGB tensors clipped to [0,1], CIEDE2000 computed from CIELAB, and 5×5 sliding patches with stride 1. Top10% means the highest `ceil(15376×0.10)=1538` patch scores per image.",
        "",
        "## Aggregate results",
        "",
        "| Method | Global CIEDE2000 mean | Patch CIEDE2000 mean | Patch CIEDE2000 P95 | Top10 patch CIEDE2000 | Max patch CIEDE2000 | Mean signed Δa | Mean signed Δb |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = ["global_ciede2000_mean", "patch_ciede2000_mean", "patch_ciede2000_p95", "top10_patch_ciede2000", "max_patch_ciede2000", "mean_signed_delta_a", "mean_signed_delta_b"]
    for item in summary:
        lines.append("| {} | {} |".format(item["method"], " | ".join(f"{item[key]:.6f}" for key in keys)))
    lines += [
        "", "## Hard Top10 versus Global", "",
        f"- Top10 CIEDE2000 change: `{hard['top10_patch_ciede2000'] - base['top10_patch_ciede2000']:+.6f}`.",
        f"- P95 patch CIEDE2000 change: `{hard['patch_ciede2000_p95'] - base['patch_ciede2000_p95']:+.6f}`.",
        f"- Global mean CIEDE2000 change: `{hard['global_ciede2000_mean'] - base['global_ciede2000_mean']:+.6f}`.",
        "- The audit is a color-distortion diagnostic, not a claim that CIEDE2000 is the training objective.",
        "",
        "## Go/No-Go decision",
        "",
        "**GO for one controlled JPEG + OKLab-tail continuation.** The frozen outputs contain non-zero global and local CIEDE2000 tails, so a color-tail intervention is measurable. This Go decision does not assume that the color-tail loss will improve the tail after training.",
        "",
        "## Qualitative outputs",
        "",
        "Fixed predetermined images 07, 20, and 42 are rendered under `visualizations/color_bias_audit/`. Each panel includes Original, both watermarked images, CIEDE2000 heatmaps, and the automatically selected worst 5×5 color patch zoom.",
        "",
        "Machine-readable per-image values: [color_bias_audit_per_image.csv](color_bias_audit_per_image.csv).",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary": summary, "report": str(REPORT), "csv": str(CSV)}, indent=2))


if __name__ == "__main__":
    main()
