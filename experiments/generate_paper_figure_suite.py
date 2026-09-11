#!/usr/bin/env python3
"""Generate the MBRS local-artifact paper figure suite from verified local data."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Rectangle

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from network.Encoder_MP_Decoder import EncoderDecoder


ROOT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
RUNS = ROOT / "experiments/runs"
REPORT = ROOT / "reports/uniform_eval_all.jsonl"
MANIFEST = ROOT / "reports/uniform_eval_manifest.pt"
OUTPUT = ROOT / "visualizations/paper_suite_20260907"

METHOD_RUNS = {
    "Global": "optimization_global_seed17_128_m64_crop",
    "Worst": "optimization_worst_seed17_128_m64_crop",
    "Patch16": "fixed_ablation_patch16_128_m64_crop",
    "Weight25": "optimization_worst_weight25_seed17_128_m64_crop",
    "No-crop": "nocrop_global_128_m64",
}

GLOBAL_SEEDS = [f"optimization_global_seed{seed}_128_m64_crop" for seed in (17, 29, 41)]
WORST_SEEDS = [f"optimization_worst_seed{seed}_128_m64_crop" for seed in (17, 29, 41)]
PATCH_MEAN_SEEDS = [f"optimization_patch_mean_seed{seed}_128_m64_crop" for seed in (17, 29, 41)]
AREAS = [30, 50, 70, 100]
ATTACKS = ["crop_30", "crop_50", "crop_70", "crop_100"]

COLORS = {
    "Global": "#4C78A8",
    "Worst": "#E45756",
    "Patch16": "#F2CF5B",
    "Weight25": "#54A24B",
    "No-crop": "#9467BD",
    "Pending": "#D9D9D9",
}


def load_final_rows():
    rows = [json.loads(line) for line in REPORT.read_text().splitlines() if line.strip()]
    final = {}
    for row in rows:
        run = row["run"]
        if run not in final or row["epoch"] > final[run]["epoch"]:
            final[run] = row
    return final


def load_checkpoint(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def encode_images(run, images, messages):
    checkpoint = load_checkpoint(RUNS / run / "checkpoint_0100.pth")
    model = EncoderDecoder(128, 128, 64, ["Identity()"]).cpu()
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(images.cpu(), messages.cpu())
    del model, checkpoint
    return encoded


def to_image(tensor):
    return tensor.detach().cpu().clamp(-1, 1).add(1).div(2).permute(1, 2, 0).numpy()


def residual_map(encoded, cover):
    return (encoded - cover).pow(2).mean(dim=0).detach().cpu().numpy()


def amplify_residual(encoded, cover, scale):
    residual = (encoded - cover).abs().mean(dim=0).detach().cpu().numpy()
    return np.clip(residual / max(scale, 1e-12), 0, 1)


def patch_scores(encoded, cover, patch_size=32):
    residual = (encoded - cover).pow(2).mean(dim=0).detach().cpu().numpy()
    height, width = residual.shape
    scores = []
    for y in range(0, height, patch_size):
        row = []
        for x in range(0, width, patch_size):
            row.append(float(residual[y : y + patch_size, x : x + patch_size].mean()))
        scores.append(row)
    return np.asarray(scores)


def add_panel_label(ax, label):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 2, "edgecolor": "none"},
    )


def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def clean_plot(ax):
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, number, slug):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    stem = OUTPUT / f"fig{number:02d}_{slug}"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, center, text, color="#C2E5FF", width=1.9, height=0.78):
    x, y = center
    patch = Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        facecolor=color,
        edgecolor="#425466",
        linewidth=1.3,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=9)


def draw_arrow(ax, start, finish, text=None, style="-"):
    arrow = FancyArrowPatch(
        start,
        finish,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.35,
        color="#425466",
        linestyle=style,
    )
    ax.add_patch(arrow)
    if text:
        x = (start[0] + finish[0]) / 2
        y = (start[1] + finish[1]) / 2
        ax.text(x, y + 0.16, text, ha="center", va="bottom", fontsize=8)


def metric(final, run, key):
    return final[run]["evaluation"]["image_quality"][key]


def ber_values(final, run):
    return np.asarray([final[run]["evaluation"]["attacks"][attack]["ber"] for attack in ATTACKS])


def method_seed_values(final, runs, key):
    return np.asarray([metric(final, run, key) for run in runs])


def make_source_images():
    torch.set_num_threads(1)
    manifest = load_checkpoint(MANIFEST)
    batch_images = manifest["images"][:16].cpu()
    batch_messages = manifest["messages"][:16].cpu()
    global_batch = encode_images(METHOD_RUNS["Global"], batch_images, batch_messages)
    global_scores = torch.stack(
        [
            torch.tensor(patch_scores(global_batch[index], batch_images[index], 32)).max()
            for index in range(batch_images.shape[0])
        ]
    )
    index = int(global_scores.argmax().item())
    cover = batch_images[index]
    message = batch_messages[index : index + 1]
    encoded = {"Global": global_batch[index]}
    for method in ("Worst", "Patch16", "Weight25", "No-crop"):
        encoded[method] = encode_images(
            METHOD_RUNS[method], batch_images[index : index + 1], message
        )[0]
    return cover, encoded, index


def fig01_local_artifact(cover, encoded):
    residual = (encoded["Global"] - cover).abs().mean(dim=0).numpy()
    scale = np.quantile(residual, 0.995)
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.75))
    axes[0].imshow(to_image(cover))
    axes[1].imshow(to_image(encoded["Global"]))
    axes[2].imshow(amplify_residual(encoded["Global"], cover, scale), cmap="magma", vmin=0, vmax=1)
    for ax, title in zip(axes, ["Cover", "Watermarked", "Amplified residual"]):
        ax.set_title(title)
        clean_axis(ax)
    save(fig, 1, "local_artifact_example")


def fig02_heatmap(cover, encoded):
    heat = residual_map(encoded["Global"], cover)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    image = ax.imshow(heat, cmap="inferno")
    ax.set_title("Residual spatial distribution")
    clean_axis(ax)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean squared error")
    save(fig, 2, "residual_spatial_heatmap")


def fig03_global_local_scatter(final):
    groups = {
        "Global": GLOBAL_SEEDS,
        "Worst": WORST_SEEDS,
        "Patch16": [METHOD_RUNS["Patch16"]],
        "Weight25": [METHOD_RUNS["Weight25"]],
        "No-crop": [METHOD_RUNS["No-crop"]],
    }
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for method, runs in groups.items():
        x = method_seed_values(final, runs, "psnr")
        y = method_seed_values(final, runs, "worst_patch_psnr")
        ax.scatter(x, y, s=68, label=method, color=COLORS[method], alpha=0.9)
    bounds = [29.5, 38.2]
    ax.plot(bounds, bounds, "--", color="#999999", linewidth=1, label="Equal PSNR")
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_xlabel("Global PSNR (dB)")
    ax.set_ylabel("Worst-patch PSNR (dB)")
    ax.set_title("Global quality vs local-tail quality")
    clean_plot(ax)
    ax.legend(frameon=False, fontsize=8)
    save(fig, 3, "global_vs_local_psnr_scatter")


def fig04_method_framework():
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    draw_box(ax, (1.1, 3.6), "Cover image", "#C2E5FF")
    draw_box(ax, (1.1, 2.1), "Message bits", "#DCCCFF")
    draw_box(ax, (3.6, 3.0), "MBRS Encoder", "#C6FAF6")
    draw_box(ax, (6.0, 3.0), "Watermarked image", "#CDF4D3")
    draw_box(ax, (8.4, 3.0), "Random Crop", "#FFE0C2")
    draw_box(ax, (10.8, 3.0), "Decoder", "#C6FAF6")
    draw_box(ax, (12.1, 1.5), "Message loss", "#FFECBD", width=1.6)
    draw_box(ax, (6.0, 1.5), "Global MSE", "#FFECBD", width=1.6)
    draw_box(ax, (8.4, 1.5), "Worst-patch MSE", "#FFC2EC", width=2.1)
    draw_arrow(ax, (2.05, 3.6), (2.65, 3.2))
    draw_arrow(ax, (2.05, 2.1), (2.65, 2.75))
    draw_arrow(ax, (4.55, 3.0), (5.05, 3.0))
    draw_arrow(ax, (6.95, 3.0), (7.45, 3.0))
    draw_arrow(ax, (9.35, 3.0), (9.85, 3.0))
    draw_arrow(ax, (11.75, 2.75), (12.05, 1.95))
    draw_arrow(ax, (5.7, 2.6), (5.9, 1.95))
    draw_arrow(ax, (6.35, 2.6), (7.85, 1.8))
    ax.text(6.7, 0.45, "Total loss = 10 × message + α × global + β × local", ha="center", fontsize=11)
    ax.set_title("Proposed training framework", fontsize=14)
    save(fig, 4, "method_framework")


def fig05_patch_partition(cover, encoded):
    scores = patch_scores(encoded["Global"], cover, 32)
    row, col = np.unravel_index(scores.argmax(), scores.shape)
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.imshow(to_image(cover))
    for pos in range(0, 129, 32):
        ax.axhline(pos - 0.5, color="white", linewidth=1)
        ax.axvline(pos - 0.5, color="white", linewidth=1)
    ax.add_patch(Rectangle((col * 32 - 0.5, row * 32 - 0.5), 32, 32, fill=False, edgecolor="#FF2D2D", linewidth=3))
    ax.set_title("4×4 patch partition and worst patch")
    clean_axis(ax)
    save(fig, 5, "patch_partition_worst_highlight")


def fig06_loss_weights():
    labels = ["Global", "Worst", "Weight25", "Weight75"]
    global_weights = [1.0, 0.5, 0.75, 0.25]
    local_weights = [0.0, 0.5, 0.25, 0.75]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y = np.arange(len(labels))
    ax.barh(y, global_weights, color="#4C78A8", label="Global MSE")
    ax.barh(y, local_weights, left=global_weights, color="#E45756", label="Worst-patch MSE")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Image-loss allocation")
    ax.set_title("Loss weight allocation")
    ax.invert_yaxis()
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.33))
    clean_plot(ax)
    save(fig, 6, "loss_weight_allocation")


def fig07_patch_size_diagram(cover, encoded):
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    for ax, size in zip(axes, [16, 32, 64]):
        scores = patch_scores(encoded["Global"], cover, size)
        row, col = np.unravel_index(scores.argmax(), scores.shape)
        ax.imshow(to_image(cover))
        for pos in range(0, 129, size):
            ax.axhline(pos - 0.5, color="white", linewidth=0.7)
            ax.axvline(pos - 0.5, color="white", linewidth=0.7)
        ax.add_patch(Rectangle((col * size - 0.5, row * size - 0.5), size, size, fill=False, edgecolor="#FF2D2D", linewidth=2.4))
        ax.set_title(f"Patch {size}×{size}")
        clean_axis(ax)
    save(fig, 7, "patch_size_comparison_diagram")


def fig08_watermarked_comparison(cover, encoded):
    methods = ["Cover", "Global", "Worst", "Patch16", "Weight25"]
    fig, axes = plt.subplots(1, 5, figsize=(13.0, 2.8))
    axes[0].imshow(to_image(cover))
    for ax, method in zip(axes[1:], methods[1:]):
        ax.imshow(to_image(encoded[method]))
    for ax, method in zip(axes, methods):
        ax.set_title(method)
        clean_axis(ax)
    save(fig, 8, "watermarked_visual_comparison")


def fig09_residual_comparison(cover, encoded):
    methods = ["Global", "Worst", "Patch16", "Weight25"]
    all_residuals = [(encoded[m] - cover).abs().mean(dim=0).numpy() for m in methods]
    scale = np.quantile(np.concatenate([r.ravel() for r in all_residuals]), 0.995)
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.8))
    for ax, method, residual in zip(axes, methods, all_residuals):
        ax.imshow(np.clip(residual / max(scale, 1e-12), 0, 1), cmap="magma", vmin=0, vmax=1)
        ax.set_title(method)
        clean_axis(ax)
    fig.suptitle("Amplified residual comparison", y=1.02)
    save(fig, 9, "residual_comparison")


def fig10_heatmap_comparison(cover, encoded):
    methods = ["Global", "Worst", "Patch16", "Weight25"]
    maps = [residual_map(encoded[m], cover) for m in methods]
    vmax = np.quantile(np.concatenate([m.ravel() for m in maps]), 0.995)
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0))
    for ax, method, heat in zip(axes, methods, maps):
        image = ax.imshow(heat, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(method)
        clean_axis(ax)
    fig.colorbar(image, ax=axes, fraction=0.018, pad=0.02, label="MSE")
    fig.suptitle("Residual heatmap comparison", y=1.02)
    save(fig, 10, "residual_heatmap_comparison")


def fig11_local_zoom(cover, encoded):
    scores = patch_scores(encoded["Global"], cover, 32)
    row, col = np.unravel_index(scores.argmax(), scores.shape)
    y, x = row * 32, col * 32
    panels = [cover[:, y : y + 32, x : x + 32], encoded["Global"][:, y : y + 32, x : x + 32], encoded["Weight25"][:, y : y + 32, x : x + 32]]
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.8))
    for ax, panel, title in zip(axes, panels, ["Cover crop", "Global crop", "Weight25 crop"]):
        ax.imshow(to_image(panel), interpolation="nearest")
        ax.set_title(title)
        clean_axis(ax)
    save(fig, 11, "local_region_zoom_comparison")


def fig12_tail_bar_with_error(final):
    labels = ["Global", "Worst", "Weight25", "Patch16"]
    values = [
        method_seed_values(final, GLOBAL_SEEDS, "worst_patch_psnr"),
        method_seed_values(final, WORST_SEEDS, "worst_patch_psnr"),
        method_seed_values(final, [METHOD_RUNS["Weight25"]], "worst_patch_psnr"),
        method_seed_values(final, [METHOD_RUNS["Patch16"]], "worst_patch_psnr"),
    ]
    means = [float(v.mean()) for v in values]
    errors = [float(v.std(ddof=1)) if len(v) > 1 else 0 for v in values]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(labels, means, yerr=errors, capsize=5, color=[COLORS[l] for l in labels])
    for bar, mean, samples in zip(bars, means, values):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.18, f"{mean:.2f}\nn={len(samples)}", ha="center", fontsize=9)
    ax.set_ylim(29, 36)
    ax.set_ylabel("Worst-patch PSNR (dB)")
    ax.set_title("Local-tail quality with seed variation")
    clean_plot(ax)
    save(fig, 12, "worst_patch_psnr_errorbars")


def fig13_selected_scatter(final):
    methods = ["Global", "Worst", "Weight25", "Patch16", "No-crop"]
    runs = [GLOBAL_SEEDS, WORST_SEEDS, [METHOD_RUNS["Weight25"]], [METHOD_RUNS["Patch16"]], [METHOD_RUNS["No-crop"]]]
    fig, ax = plt.subplots(figsize=(6.3, 5.0))
    for method, method_runs in zip(methods, runs):
        x = method_seed_values(final, method_runs, "psnr")
        y = method_seed_values(final, method_runs, "worst_patch_psnr")
        ax.scatter(x.mean(), y.mean(), s=110, color=COLORS[method])
        ax.annotate(method, (x.mean(), y.mean()), xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Global PSNR (dB)")
    ax.set_ylabel("Worst-patch PSNR (dB)")
    ax.set_title("Method-level quality comparison")
    clean_plot(ax)
    save(fig, 13, "method_global_local_scatter")


def curve_stats(final, runs):
    curves = np.asarray([ber_values(final, run) for run in runs])
    return curves.mean(axis=0), curves.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(len(AREAS))


def fig14_crop_ber_curves(final):
    series = {
        "Global": GLOBAL_SEEDS,
        "Worst": WORST_SEEDS,
        "Weight25": [METHOD_RUNS["Weight25"]],
        "Patch16": [METHOD_RUNS["Patch16"]],
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, runs in series.items():
        mean, error = curve_stats(final, runs)
        ax.plot(AREAS, mean, marker="o", linewidth=2, label=method, color=COLORS[method])
        if len(runs) > 1:
            ax.fill_between(AREAS, np.maximum(mean - error, 0), mean + error, color=COLORS[method], alpha=0.16)
    ax.set_xlabel("Crop retained area (%)")
    ax.set_ylabel("BER (lower is better)")
    ax.set_title("Crop robustness across methods")
    ax.set_xticks(AREAS)
    ax.set_ylim(bottom=0)
    clean_plot(ax)
    ax.legend(frameon=False, ncol=2)
    save(fig, 14, "crop_ber_curves_with_bands")


def render_table(number, slug, title, columns, rows, note=None, figsize=(10.5, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DCEBFA")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")
    ax.set_title(title, pad=20, fontsize=13)
    if note:
        ax.text(0.5, 0.03, note, transform=ax.transAxes, ha="center", fontsize=8, color="#555555")
    save(fig, number, slug)


def fig15_ber_table(final):
    entries = {
        "Global mean": GLOBAL_SEEDS,
        "Worst mean": WORST_SEEDS,
        "Weight25": [METHOD_RUNS["Weight25"]],
        "Patch16": [METHOD_RUNS["Patch16"]],
        "No-crop": [METHOD_RUNS["No-crop"]],
    }
    rows = []
    for label, runs in entries.items():
        mean, _ = curve_stats(final, runs)
        rows.append([label] + [f"{value:.5f}" for value in mean])
    render_table(15, "crop_ber_three_line_table", "Crop BER comparison", ["Method", "30%", "50%", "70%", "100%"], rows, "Global and Worst report three-seed means; candidate rows are seed17.")


def simple_bar(number, slug, title, labels, values, color="#4C78A8", reference=None):
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    colors = color if isinstance(color, list) else [color] * len(labels)
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.2f}", ha="center", fontsize=9)
    if reference is not None:
        ax.axhline(reference, color="#333333", linestyle="--", linewidth=1.4, label=f"Global mean {reference:.2f}")
        ax.legend(frameon=False)
    ax.set_ylabel("Canonical worst-patch PSNR (dB)")
    ax.set_title(title)
    ax.set_ylim(min(values) - 1.2, max(values) + 1.4)
    clean_plot(ax)
    save(fig, number, slug)


def fig16_patch_size_ablation(final):
    values = [metric(final, "fixed_ablation_patch16_128_m64_crop", "worst_patch_psnr"), metric(final, "fixed_worst_128_m64_crop", "worst_patch_psnr"), metric(final, "fixed_ablation_patch64_128_m64_crop", "worst_patch_psnr")]
    simple_bar(16, "patch_size_ablation", "Patch-size ablation (seed17)", ["16", "32", "64"], values, ["#F2CF5B", "#4C78A8", "#9C755F"])


def fig17_topk_ablation(final):
    values = [metric(final, "fixed_ablation_topk10_128_m64_crop", "worst_patch_psnr"), metric(final, "fixed_worst_128_m64_crop", "worst_patch_psnr"), metric(final, "fixed_ablation_topk50_128_m64_crop", "worst_patch_psnr")]
    simple_bar(17, "topk_ablation", "Top-k ratio ablation (seed17)", ["10%", "25%", "50%"], values, ["#B279A2", "#4C78A8", "#59A14F"])


def fig18_weight_ablation(final):
    values = [metric(final, METHOD_RUNS["Weight25"], "worst_patch_psnr"), metric(final, "optimization_worst_seed17_128_m64_crop", "worst_patch_psnr"), metric(final, "optimization_worst_weight75_seed17_128_m64_crop", "worst_patch_psnr")]
    reference = metric(final, "optimization_global_seed17_128_m64_crop", "worst_patch_psnr")
    simple_bar(18, "weight_ablation", "Local-loss weight ablation (seed17)", ["0.25", "0.50", "0.75"], values, ["#54A24B", "#E45756", "#B279A2"], reference)


def fig19_ablation_heatmap(final):
    patch_sizes = [16, 32, 64]
    weights = [0.25, 0.50, 0.75]
    matrix = np.full((len(patch_sizes), len(weights)), np.nan)
    matrix[0, 1] = metric(final, "fixed_ablation_patch16_128_m64_crop", "worst_patch_psnr")
    matrix[1, 0] = metric(final, METHOD_RUNS["Weight25"], "worst_patch_psnr")
    matrix[1, 1] = metric(final, "optimization_worst_seed17_128_m64_crop", "worst_patch_psnr")
    matrix[1, 2] = metric(final, "optimization_worst_weight75_seed17_128_m64_crop", "worst_patch_psnr")
    matrix[2, 1] = metric(final, "fixed_ablation_patch64_128_m64_crop", "worst_patch_psnr")
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.YlGn.copy()
    cmap.set_bad("#E6E6E6")
    fig, ax = plt.subplots(figsize=(6.2, 4.7))
    image = ax.imshow(masked, cmap=cmap, norm=Normalize(vmin=30, vmax=35))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            text = "Pending" if np.isnan(matrix[row, col]) else f"{matrix[row, col]:.2f}"
            ax.text(col, row, text, ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(weights)), [str(value) for value in weights])
    ax.set_yticks(range(len(patch_sizes)), [str(value) for value in patch_sizes])
    ax.set_xlabel("Local-loss weight")
    ax.set_ylabel("Patch size")
    ax.set_title("Patch size × weight ablation (seed17)")
    fig.colorbar(image, ax=ax, label="Canonical worst-patch PSNR (dB)")
    save(fig, 19, "ablation_combination_heatmap")


def fig20_nocrop_quality(cover, encoded):
    fig, axes = plt.subplots(2, 2, figsize=(6.1, 6.0))
    panels = [(cover, "Cover"), (encoded["Global"], "Crop-trained"), (cover, "Cover"), (encoded["No-crop"], "No-crop trained")]
    for ax, (panel, title) in zip(axes.ravel(), panels):
        ax.imshow(to_image(panel))
        ax.set_title(title)
        clean_axis(ax)
    fig.suptitle("No-crop vs crop-trained image quality", y=1.01)
    save(fig, 20, "nocrop_vs_crop_quality_2x2")


def fig21_nocrop_ber(final):
    crop_mean, crop_std = curve_stats(final, GLOBAL_SEEDS)
    nocrop = ber_values(final, METHOD_RUNS["No-crop"])
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(AREAS, nocrop, marker="o", linewidth=2.3, color=COLORS["No-crop"], label="No-crop training")
    ax.plot(AREAS, crop_mean, marker="o", linewidth=2.3, color=COLORS["Global"], label="Crop-trained global")
    ax.fill_between(AREAS, np.maximum(crop_mean - crop_std, 0), crop_mean + crop_std, color=COLORS["Global"], alpha=0.17)
    ax.set_xlabel("Crop retained area (%)")
    ax.set_ylabel("BER")
    ax.set_title("Motivation: quality–robustness trade-off")
    ax.set_xticks(AREAS)
    ax.set_ylim(bottom=0)
    clean_plot(ax)
    ax.legend(frameon=False)
    save(fig, 21, "nocrop_vs_crop_ber")


def fig22_resolution_table(final):
    global_mean = method_seed_values(final, GLOBAL_SEEDS, "psnr").mean()
    global_tail = method_seed_values(final, GLOBAL_SEEDS, "worst_patch_psnr").mean()
    global_ber, _ = curve_stats(final, GLOBAL_SEEDS)
    rows = [
        ["128×128", "64", "Completed", f"{global_mean:.3f}", f"{global_tail:.3f}", f"{global_ber[0]:.5f}"],
        ["256×256", "32", "Pending", "—", "—", "—"],
    ]
    render_table(22, "resolution_generalization_table", "Resolution generalization", ["Resolution", "Payload", "Status", "PSNR", "Worst PSNR", "BER@30%"], rows, "The existing legacy 256×256/256-bit run is not protocol-compatible.")


def fig23_attack_placeholder():
    labels = ["JPEG", "Blur", "Noise"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, [0, 0, 0], color=COLORS["Pending"], hatch="//", edgecolor="#777777")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, 0.04, "Pending\nunified evaluation", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Bit accuracy")
    ax.set_title("Robustness under non-crop attacks")
    clean_plot(ax)
    save(fig, 23, "other_attack_robustness_pending")


def fig24_external_table(final):
    global_psnr = method_seed_values(final, GLOBAL_SEEDS, "psnr").mean()
    global_tail = method_seed_values(final, GLOBAL_SEEDS, "worst_patch_psnr").mean()
    global_ber, _ = curve_stats(final, GLOBAL_SEEDS)
    weight = final[METHOD_RUNS["Weight25"]]
    wq = weight["evaluation"]["image_quality"]
    wb = ber_values(final, METHOD_RUNS["Weight25"])
    rows = []
    for method in ["HiDDeN", "StegaStamp", "TrustMark", "WOFA", "InvisMark"]:
        rows.append([method, "Pending", "—", "—", "—"])
    rows.insert(2, ["MBRS controlled global", "3 seeds", f"{global_psnr:.3f}", f"{global_tail:.3f}", f"{global_ber[0]:.5f}"])
    rows.append(["Ours: Weight25", "Provisional n=1", f"{wq['psnr']:.3f}", f"{wq['worst_patch_psnr']:.3f}", f"{wb[0]:.5f}"])
    render_table(24, "external_method_comparison_table", "External method comparison", ["Method", "Status", "PSNR", "Worst PSNR", "BER@30%"], rows, "All methods must be rerun under 128×128, 64-bit and the fixed crop manifest.", figsize=(9.5, 5.2))


def fig25_external_visual(cover, encoded):
    methods = ["HiDDeN", "StegaStamp", "MBRS", "TrustMark", "WOFA", "InvisMark", "Ours"]
    fig, axes = plt.subplots(1, len(methods), figsize=(16.2, 2.8))
    for ax, method in zip(axes, methods):
        if method == "MBRS":
            ax.imshow(to_image(encoded["Global"]))
        elif method == "Ours":
            ax.imshow(to_image(encoded["Weight25"]))
        else:
            ax.imshow(np.ones((128, 128, 3)) * 0.92)
            ax.text(0.5, 0.5, "Pending\ncheckpoint", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#666666")
        ax.set_title(method, fontsize=9)
        clean_axis(ax)
    fig.suptitle("External visual comparison template", y=1.02)
    save(fig, 25, "external_visual_comparison_pending")


def load_history(run, filename="train.jsonl"):
    path = RUNS / run / filename
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fig26_training_curves():
    series = {
        "Global": METHOD_RUNS["Global"],
        "Worst": METHOD_RUNS["Worst"],
        "Patch16": METHOD_RUNS["Patch16"],
        "Weight25": METHOD_RUNS["Weight25"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1))
    for method, run in series.items():
        history = load_history(run)
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["loss"] for row in history], label=method, color=COLORS[method])
        axes[1].plot(epochs, [row["psnr"] for row in history], label=method, color=COLORS[method])
        axes[2].plot(epochs, [row["ber"] for row in history], label=method, color=COLORS[method])
    for ax, title, ylabel in zip(axes, ["Training loss", "Training PSNR", "Training BER"], ["Loss", "PSNR (dB)", "BER"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        clean_plot(ax)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, 26, "training_curves")


def fig27_seed_boxplot(final):
    data = [
        method_seed_values(final, GLOBAL_SEEDS, "worst_patch_psnr"),
        method_seed_values(final, PATCH_MEAN_SEEDS, "worst_patch_psnr"),
        method_seed_values(final, WORST_SEEDS, "worst_patch_psnr"),
    ]
    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    box = ax.boxplot(data, tick_labels=["Global", "Patch mean", "Worst"], patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], [COLORS["Global"], "#76B7B2", COLORS["Worst"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Worst-patch PSNR (dB)")
    ax.set_title("Three-seed stability")
    clean_plot(ax)
    save(fig, 27, "multi_seed_stability_boxplot")


def fig28_extraction_flow():
    fig, ax = plt.subplots(figsize=(11.5, 3.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    centers = [(1, 1.8), (3.5, 1.8), (6.0, 1.8), (8.5, 1.8), (11.0, 1.8)]
    texts = ["Watermarked image", "Crop attack", "MBRS Decoder", "Recovered bits", "BER / Accuracy"]
    colors = ["#CDF4D3", "#FFE0C2", "#C6FAF6", "#DCCCFF", "#FFECBD"]
    for center, text, color in zip(centers, texts, colors):
        draw_box(ax, center, text, color, width=1.9)
    for left, right in zip(centers[:-1], centers[1:]):
        draw_arrow(ax, (left[0] + 0.95, left[1]), (right[0] - 0.95, right[1]))
    ax.set_title("Watermark extraction and robustness evaluation", fontsize=14)
    save(fig, 28, "watermark_extraction_flow")


def fig29_auto_localization(cover, encoded):
    heat = residual_map(encoded["Global"], cover)
    scores = patch_scores(encoded["Global"], cover, 32)
    row, col = np.unravel_index(scores.argmax(), scores.shape)
    y, x = row * 32, col * 32
    zoom = encoded["Global"][:, y : y + 32, x : x + 32]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.1))
    axes[0].imshow(to_image(encoded["Global"]))
    axes[0].add_patch(Rectangle((x - 0.5, y - 0.5), 32, 32, fill=False, edgecolor="#FF2D2D", linewidth=3))
    axes[0].set_title("Auto-located patch")
    axes[1].imshow(heat, cmap="inferno")
    axes[1].add_patch(Rectangle((x - 0.5, y - 0.5), 32, 32, fill=False, edgecolor="#4CFF4C", linewidth=2.4))
    axes[1].set_title("Residual heatmap")
    axes[2].imshow(to_image(zoom), interpolation="nearest")
    axes[2].set_title("Worst-patch zoom")
    for ax in axes:
        clean_axis(ax)
    save(fig, 29, "worst_patch_auto_localization")


def fig00_project_workflow():
    fig, ax = plt.subplots(figsize=(15.0, 4.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")
    centers = [(1.2, 3.2), (3.7, 3.2), (6.2, 3.2), (8.7, 3.2), (11.2, 3.2), (13.7, 3.2)]
    texts = ["Trade-off\nhypothesis", "Controlled\nbaselines", "Worst-patch\nvariants", "Multi-seed\ntraining", "Fixed-manifest\nevaluation", "Evidence-based\nconclusion"]
    colors = ["#DCCCFF", "#C2E5FF", "#FFC2EC", "#FFE0C2", "#C6FAF6", "#CDF4D3"]
    for center, text, color in zip(centers, texts, colors):
        draw_box(ax, center, text, color, width=2.0, height=1.0)
    for left, right in zip(centers[:-1], centers[1:]):
        draw_arrow(ax, (left[0] + 1.0, left[1]), (right[0] - 1.0, right[1]))
    draw_box(ax, (8.7, 1.1), "Patch size × top-k × weight", "#FFECBD", width=3.0)
    draw_arrow(ax, (8.7, 1.5), (8.7, 2.65), "Ablation")
    ax.set_title("Project research and experiment workflow", fontsize=14)
    save(fig, 0, "project_research_workflow")


def write_index(sample_index):
    entries = [
        (0, "Project research workflow", "data-backed"),
        (1, "Local artifact example", "data-backed"),
        (2, "Residual spatial heatmap", "data-backed"),
        (3, "Global vs local PSNR scatter", "data-backed"),
        (4, "Method framework", "method schematic"),
        (5, "Patch partition", "data-backed schematic"),
        (6, "Loss weight allocation", "method schematic"),
        (7, "Patch size diagram", "data-backed schematic"),
        (8, "Watermarked image comparison", "data-backed"),
        (9, "Residual comparison", "data-backed"),
        (10, "Residual heatmap comparison", "data-backed"),
        (11, "Local zoom comparison", "data-backed"),
        (12, "Worst-patch PSNR bars", "data-backed; n shown"),
        (13, "Method quality scatter", "data-backed"),
        (14, "Crop BER curves", "data-backed"),
        (15, "Crop BER table", "data-backed"),
        (16, "Patch-size ablation", "seed17"),
        (17, "Top-k ablation", "seed17"),
        (18, "Weight ablation", "seed17"),
        (19, "Ablation heatmap", "partial; pending cells marked"),
        (20, "No-crop vs crop quality", "data-backed"),
        (21, "No-crop vs crop BER", "data-backed"),
        (22, "Resolution table", "256×256 pending"),
        (23, "Other attacks", "template; pending"),
        (24, "External comparison table", "template; external methods pending"),
        (25, "External visual comparison", "template; external methods pending"),
        (26, "Training curves", "data-backed"),
        (27, "Multi-seed boxplot", "data-backed"),
        (28, "Extraction flow", "method schematic"),
        (29, "Worst-patch localization", "data-backed"),
    ]
    lines = [
        "# MBRS paper figure suite",
        "",
        f"Source report: `{REPORT}`",
        f"Aligned visual sample index: `{sample_index}` from the fixed uniform-evaluation manifest.",
        "",
        "Each figure is exported as PNG and PDF.",
        "",
        "| Figure | Description | Status |",
        "|---:|---|---|",
    ]
    for number, description, status in entries:
        lines.append(f"| {number} | {description} | {status} |")
    lines += [
        "",
        "Pending figures intentionally contain no invented measurements. Replace their placeholders after protocol-compatible experiments finish.",
    ]
    (OUTPUT / "INDEX.md").write_text("\n".join(lines) + "\n")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    final = load_final_rows()
    cover, encoded, sample_index = make_source_images()
    fig00_project_workflow()
    fig01_local_artifact(cover, encoded)
    fig02_heatmap(cover, encoded)
    fig03_global_local_scatter(final)
    fig04_method_framework()
    fig05_patch_partition(cover, encoded)
    fig06_loss_weights()
    fig07_patch_size_diagram(cover, encoded)
    fig08_watermarked_comparison(cover, encoded)
    fig09_residual_comparison(cover, encoded)
    fig10_heatmap_comparison(cover, encoded)
    fig11_local_zoom(cover, encoded)
    fig12_tail_bar_with_error(final)
    fig13_selected_scatter(final)
    fig14_crop_ber_curves(final)
    fig15_ber_table(final)
    fig16_patch_size_ablation(final)
    fig17_topk_ablation(final)
    fig18_weight_ablation(final)
    fig19_ablation_heatmap(final)
    fig20_nocrop_quality(cover, encoded)
    fig21_nocrop_ber(final)
    fig22_resolution_table(final)
    fig23_attack_placeholder()
    fig24_external_table(final)
    fig25_external_visual(cover, encoded)
    fig26_training_curves()
    fig27_seed_boxplot(final)
    fig28_extraction_flow()
    fig29_auto_localization(cover, encoded)
    write_index(sample_index)
    print(f"generated 30 figures in {OUTPUT}")


if __name__ == "__main__":
    main()
