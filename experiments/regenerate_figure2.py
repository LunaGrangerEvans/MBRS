#!/usr/bin/env python3
"""Regenerate the six-column, validation-only Figure 2 candidates.

All image inputs are persisted outputs.  The script never opens a project-test
manifest, retrains a model, or writes formal Table I inputs.  It intentionally
keeps the old Figure 2 renderers untouched because their labels and operating
points predate the external-baseline comparison requested here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.generate_fig2_qualitative_search import Panel, make_pptx


PROJECT = Path(__file__).resolve().parents[1]
MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
MANIFEST = MOUNT / "reports/content_selector/validation_manifest.pt"
MASKWM_PROVENANCE = MOUNT / "external_baselines/outputs/maskwm_validation/D_64bits/provenance.json"
MASKWM_128 = MASKWM_PROVENANCE.parent
MASKWM_NATIVE = MASKWM_128 / "native_512"
MASKWM_CHECKPOINT = MOUNT / "external_baselines/checkpoints/maskwm/D_64bits.pth"
HIDDEN_CACHE = MOUNT / "results/fig2_stress/external_baselines/hidden_64bit_validation.pt"
HIDDEN_CHECKPOINT = MOUNT / "external_baselines/outputs/hidden_64bit_retrained/hidden_64bit_epoch_200.pyt"
GLOBAL_RESULTS = MOUNT / "results/fig2_stress/promoted_ours-base_wmsg80/outputs.pt"
HARD_RESULTS = MOUNT / "results/fig2_stress/promoted_ours_wmsg80/outputs.pt"
GLOBAL_CHECKPOINT = MOUNT / "checkpoints/fig2_stress/promoted_ours-base_wmsg80/checkpoint_0020.pth"
HARD_CHECKPOINT = MOUNT / "checkpoints/fig2_stress/promoted_ours_wmsg80/checkpoint_0020.pth"
GLOBAL_CONFIG = GLOBAL_CHECKPOINT.parent / "config.json"
HARD_CONFIG = HARD_CHECKPOINT.parent / "config.json"
OKLAB_DIR = MOUNT / "reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25"
OKLAB_RESULTS = OKLAB_DIR / "seed17_crop_hard16_stride8_top10_global_oklab_g25_outputs.pt"
OKLAB_RUN = MOUNT / "experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25"
OKLAB_CHECKPOINT = OKLAB_RUN / "checkpoint_0020.pth"
OKLAB_CONFIG = OKLAB_RUN / "config.resolved.json"
OUTPUT_DIR = PROJECT / "visualizations/figure2"
AUDIT_NOTE = PROJECT / "reports/figure2_maskwm_consistency_check.md"
REPORT = PROJECT / "reports/figure2_regeneration_report.md"

METHODS = (
    "Original",
    "HiDDeN-64 (external)",
    "MaskWM-D_64 (external)",
    "MBRS crop-trained global",
    "Hard Local-Tail (Ours)",
    "Hard Local-Tail + OKLab",
)
SOURCE_KEYS = ("original", "hidden", "maskwm", "global", "hard", "oklab")
ROI_SIZES = (32, 40, 48, 64)
DISPLAY_SIZE = 512
ZOOM_FACTOR = 8
RED = "#e5483c"
INK = "#202934"
MUTED = "#66727d"
PAD = (248, 249, 251)


def load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def psnr(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(float(mse), 1e-12))


def rgb_from_model(value: torch.Tensor) -> torch.Tensor:
    return ((value.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def load_encoded(path: Path) -> torch.Tensor:
    payload = load(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("encoded"), torch.Tensor):
        raise ValueError(f"{path} does not contain an encoded tensor")
    return rgb_from_model(payload["encoded"])


def load_maskwm_128() -> torch.Tensor:
    values = []
    for index in range(50):
        path = MASKWM_128 / f"image_{index:03d}.png"
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        if array.shape != (128, 128, 3):
            raise ValueError(f"unexpected MaskWM 128 image shape: {path} {array.shape}")
        values.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
    return torch.stack(values)


def load_maskwm_native() -> torch.Tensor:
    values = []
    for index in range(50):
        path = MASKWM_NATIVE / f"image_{index:03d}.png"
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        if array.shape != (DISPLAY_SIZE, DISPLAY_SIZE, 3):
            raise ValueError(f"unexpected MaskWM native image shape: {path} {array.shape}")
        values.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
    return torch.stack(values)


def display_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or value.shape[1:] != (3, 128, 128):
        raise ValueError(f"expected NCHW 128 RGB tensor, got {tuple(value.shape)}")
    return F.interpolate(value.float().clamp(0, 1), size=(DISPLAY_SIZE, DISPLAY_SIZE), mode="bilinear", align_corners=False, antialias=True)


def integral_image(array: np.ndarray) -> np.ndarray:
    return np.pad(array.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def integral_mean(integral: np.ndarray, x: int, y: int, size: int) -> float:
    total = integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x]
    return float(total / (size * size))


def choose_roi(original: torch.Tensor, global_value: torch.Tensor, hard_value: torch.Tensor) -> tuple[tuple[int, int, int, int], dict[str, float]]:
    """Search shared ROIs; kept separate from ranking so it is easy to audit."""
    improvement = ((global_value - original).square() - (hard_value - original).square()).mean(0).numpy()
    texture_source = original.mean(0).numpy()
    gy, gx = np.gradient(texture_source)
    texture = np.sqrt(gx * gx + gy * gy)
    improvement_integral = integral_image(improvement)
    texture_integral = integral_image(texture)
    texture_floor = max(0.002, float(np.percentile(texture, 30)))
    winners: list[tuple[float, int, int, int, float]] = []
    by_size: dict[str, float] = {}
    for size in ROI_SIZES:
        margin = 4 if size >= 48 else 8
        candidates: list[tuple[float, int, int, float]] = []
        for y in range(margin, 128 - size - margin + 1, 4):
            for x in range(margin, 128 - size - margin + 1, 4):
                content = integral_mean(texture_integral, x, y, size)
                if content >= texture_floor:
                    candidates.append((integral_mean(improvement_integral, x, y, size), x, y, content))
        if not candidates:
            candidates = [
                (integral_mean(improvement_integral, x, y, size), x, y, 0.0)
                for y in range(margin, 128 - size - margin + 1, 4)
                for x in range(margin, 128 - size - margin + 1, 4)
            ]
        score, x, y, content = max(candidates, key=lambda item: item[0])
        by_size[f"roi{size}_improvement"] = score
        winners.append((score, x, y, size, content))
    score, x, y, size, content = max(winners, key=lambda item: item[0])
    return (x, y, size, size), {"roi_mse_improvement": score, "roi_texture_energy": content, **by_size}


def patch_map(value: torch.Tensor, reference: torch.Tensor) -> np.ndarray:
    return F.avg_pool2d((value - reference).square().mean(0, keepdim=True).unsqueeze(0), 32, 32)[0, 0].numpy()


def gini(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    total = ordered.sum()
    if total <= 1e-15:
        return 0.0
    n = len(ordered)
    return float(((2 * np.arange(1, n + 1) - n - 1) * ordered).sum() / (n * total))


def local_stats(original: torch.Tensor, global_value: torch.Tensor, hard_value: torch.Tensor, roi: tuple[int, int, int, int]) -> dict[str, float]:
    x, y, width, height = roi
    global_patch = patch_map(global_value, original)
    hard_patch = patch_map(hard_value, original)
    top_global = float(np.sort(global_patch.reshape(-1))[-4:].mean())
    top_hard = float(np.sort(hard_patch.reshape(-1))[-4:].mean())
    residual_global = (global_value - original).abs().mean(0)
    residual_hard = (hard_value - original).abs().mean(0)
    local_global = float(((global_value[:, y:y + height, x:x + width] - original[:, y:y + height, x:x + width]) ** 2).mean())
    local_hard = float(((hard_value[:, y:y + height, x:x + width] - original[:, y:y + height, x:x + width]) ** 2).mean())

    def tv(value: torch.Tensor) -> float:
        return float(value[:, 1:].sub(value[:, :-1]).abs().mean() + value[1:, :].sub(value[:-1, :]).abs().mean())

    tv_global = tv(residual_global[y:y + height, x:x + width])
    tv_hard = tv(residual_hard[y:y + height, x:x + width])
    return {
        "local_psnr_global": psnr(local_global),
        "local_psnr_hard": psnr(local_hard),
        "delta_local_psnr": psnr(top_hard) - psnr(top_global),
        "p95_global": float(np.percentile(global_patch, 95)),
        "p95_hard": float(np.percentile(hard_patch, 95)),
        "delta_p95_reduction": float(np.percentile(global_patch, 95) - np.percentile(hard_patch, 95)),
        "roi_mse_improvement": local_global - local_hard,
        "roi_residual_energy_reduction": float((residual_global[y:y + height, x:x + width] - residual_hard[y:y + height, x:x + width]).mean()),
        "roi_structured_tv_reduction": tv_global - tv_hard,
        "global_patch_mse_mean": float(global_patch.mean()),
        "hard_patch_mse_mean": float(hard_patch.mean()),
        "global_gini": gini(global_patch),
        "hard_gini": gini(hard_patch),
    }


def color_stats(reference: torch.Tensor, hard_value: torch.Tensor, oklab_value: torch.Tensor, roi: tuple[int, int, int, int]) -> dict[str, float]:
    x, y, width, height = roi
    ref = reference.permute(1, 2, 0).numpy()
    hard = hard_value.permute(1, 2, 0).numpy()
    oklab = oklab_value.permute(1, 2, 0).numpy()
    ref_lab = rgb2lab(ref)
    hard_lab = rgb2lab(hard)
    oklab_lab = rgb2lab(oklab)
    hard_ciede = deltaE_ciede2000(ref_lab, hard_lab)
    oklab_ciede = deltaE_ciede2000(ref_lab, oklab_lab)
    hard_chroma = np.sqrt(((hard_lab[..., 1:] - ref_lab[..., 1:]) ** 2).sum(-1))
    oklab_chroma = np.sqrt(((oklab_lab[..., 1:] - ref_lab[..., 1:]) ** 2).sum(-1))
    sl = np.s_[y:y + height, x:x + width]
    return {
        "roi_ciede2000_hard": float(hard_ciede[sl].mean()),
        "roi_ciede2000_oklab": float(oklab_ciede[sl].mean()),
        "roi_ciede2000_reduction": float(hard_ciede[sl].mean() - oklab_ciede[sl].mean()),
        "roi_chroma_error_hard": float(hard_chroma[sl].mean()),
        "roi_chroma_error_oklab": float(oklab_chroma[sl].mean()),
        "roi_chroma_reduction": float(hard_chroma[sl].mean() - oklab_chroma[sl].mean()),
    }


def sample_record(index: int, manifest: dict, original: torch.Tensor, values: dict[str, torch.Tensor]) -> dict[str, Any]:
    roi, roi_info = choose_roi(original[index], values["global"][index], values["hard"][index])
    stats = local_stats(original[index], values["global"][index], values["hard"][index], roi)
    color = color_stats(original[index], values["hard"][index], values["oklab"][index], roi)
    local_errors = {
        key: float(((values[key][index, :, roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] - original[index, :, roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]) ** 2).mean())
        for key in SOURCE_KEYS[1:]
    }
    return {
        "image_id": Path(manifest["sources"][index]["filename"]).stem,
        "validation_index": index,
        "source_filename": manifest["sources"][index]["filename"],
        "roi_x": roi[0],
        "roi_y": roi[1],
        "roi_width": roi[2],
        "roi_height": roi[3],
        "roi": roi,
        **roi_info,
        **stats,
        **color,
        "local_mse": local_errors,
        "semantic_proxy_mean_rgb": float(original[index].mean()),
        "semantic_proxy_texture": float(np.abs(np.gradient(original[index].mean(0).numpy())).mean()),
    }


def normalized(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def rank_candidates(manifest: dict, original: torch.Tensor, values: dict[str, torch.Tensor]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [sample_record(index, manifest, original, values) for index in range(50)]
    rows.sort(key=lambda row: (row["delta_local_psnr"], row["delta_p95_reduction"], row["roi_mse_improvement"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank_main"] = rank
    chosen = [rows[0]]
    for row in rows[1:]:
        distance = abs(row["semantic_proxy_mean_rgb"] - chosen[0]["semantic_proxy_mean_rgb"]) + 4.0 * abs(row["semantic_proxy_texture"] - chosen[0]["semantic_proxy_texture"])
        row["content_diversity_distance"] = distance
        if distance >= 0.01:
            chosen.append(row)
            break
    return rows, chosen[:2]


def rank_external(rows: list[dict[str, Any]], main_selected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_ids = {row["validation_index"] for row in main_selected}
    for row in rows:
        local = row["local_mse"]
        external_range = abs(local["hidden"] - local["maskwm"])
        external_gap = max(0.0, (local["hidden"] + local["maskwm"]) / 2.0 - local["hard"])
        row["external_local_range"] = external_range
        row["external_vs_hard_local_gap"] = external_gap
        row["external_score_raw"] = 0.45 * external_range + 0.35 * external_gap + 0.20 * max(0.0, row["roi_mse_improvement"])
    spread = normalized([row["external_local_range"] for row in rows])
    gap = normalized([row["external_vs_hard_local_gap"] for row in rows])
    gain = normalized([max(0.0, row["roi_mse_improvement"]) for row in rows])
    for row, a, b, c in zip(rows, spread, gap, gain):
        row["external_selection_score"] = 0.45 * a + 0.35 * b + 0.20 * c
    ranked = sorted(rows, key=lambda row: (row["external_selection_score"], row["external_local_range"]), reverse=True)
    chosen: list[dict[str, Any]] = []
    for row in ranked:
        if row["validation_index"] in main_ids and len(chosen) < 2:
            continue
        chosen.append(row)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        chosen = ranked[:2]
    for rank, row in enumerate(ranked, 1):
        row["rank_external"] = rank
    return ranked, chosen


def to_pil(value: torch.Tensor) -> Image.Image:
    array = np.rint(value.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def zoom_canvas(value: torch.Tensor, roi: tuple[int, int, int, int]) -> Image.Image:
    x, y, width, height = roi
    crop = to_pil(value[:, y * 4:(y + height) * 4, x * 4:(x + width) * 4])
    crop = crop.resize((width * ZOOM_FACTOR, height * ZOOM_FACTOR), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (DISPLAY_SIZE, DISPLAY_SIZE), PAD)
    canvas.paste(crop, ((DISPLAY_SIZE - crop.width) // 2, (DISPLAY_SIZE - crop.height) // 2))
    return canvas


def residual_canvas(value: torch.Tensor, reference: torch.Tensor, roi: tuple[int, int, int, int], vmax: float) -> Image.Image:
    x, y, width, height = roi
    residual = (value - reference).abs().mean(0).numpy() * 10.0
    crop = np.clip(residual[y * 4:(y + height) * 4, x * 4:(x + width) * 4] / max(vmax, 1e-12), 0, 1)
    color = (plt.get_cmap("magma")(crop)[..., :3] * 255.0).round().astype(np.uint8)
    image = Image.fromarray(color, "RGB").resize((width * ZOOM_FACTOR, height * ZOOM_FACTOR), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (DISPLAY_SIZE, DISPLAY_SIZE), PAD)
    canvas.paste(image, ((DISPLAY_SIZE - image.width) // 2, (DISPLAY_SIZE - image.height) // 2))
    return canvas


def header_labels(psnr_values: dict[str, float], ber30: dict[str, float]) -> list[str]:
    labels = []
    for method, key in zip(METHODS, SOURCE_KEYS):
        if key == "original":
            labels.append(f"{method}\nPSNR ∞ (reference)")
        elif key == "global":
            labels.append(f"{method}\nPSNR {psnr_values[key]:.2f} dB\nBER30 {ber30[key]:.3f}")
        elif key == "hard":
            labels.append(f"{method}\nPSNR {psnr_values[key]:.2f} dB\nBER30 {ber30[key]:.3f}")
        elif key == "oklab":
            labels.append(f"{method}\nPSNR {psnr_values[key]:.2f} dB\nvalidation-only")
        else:
            labels.append(f"{method}\nPSNR {psnr_values[key]:.2f} dB")
    return labels


def footer_lines() -> list[str]:
    return [
        "Fixed validation manifest; same host and shared ROI within each sample.",
        "External methods are reference baselines under released checkpoints; not strictly matched training baselines.",
        "Residual ×10 is visualization-only after RGB[0,1] normalization; each sample uses one shared colormap/range.",
        "Hard Local-Tail + OKLab is validation-only; samples and ROIs are deliberate metric-selected best cases, not representative evidence.",
    ]


def render_candidate(name: str, selected: list[dict[str, Any]], display: dict[str, torch.Tensor], psnr_values: dict[str, float], ber30: dict[str, float]) -> list[Path]:
    rows = []
    panels: list[list[Panel]] = []
    residual_ranges: list[float] = []
    original_display = display["original"]
    for sample_number, record in enumerate(selected, 1):
        roi = tuple(int(record[key]) for key in ("roi_x", "roi_y", "roi_width", "roi_height"))
        roi_max = 0.0
        for key in SOURCE_KEYS:
            roi_max = max(roi_max, float(((display[key][record["validation_index"]] - original_display[record["validation_index"]]).abs().mean(0)[roi[1] * 4:(roi[1] + roi[3]) * 4, roi[0] * 4:(roi[0] + roi[2]) * 4] * 10).max()))
        residual_ranges.append(max(roi_max, 1e-8))
        rows.extend([
            f"Sample {sample_number} · {record['image_id']} · full",
            f"Sample {sample_number} · {record['image_id']} · ROI zoom ({ZOOM_FACTOR}×)",
            f"Sample {sample_number} · {record['image_id']} · residual ×10\nvisualization-only",
        ])
        full_panels = []
        zoom_panels = []
        residual_panels = []
        index = record["validation_index"]
        for key in SOURCE_KEYS:
            full = to_pil(display[key][index])
            zoom = zoom_canvas(display[key][index], roi)
            residual = residual_canvas(display[key][index], original_display[index], roi, residual_ranges[-1])
            full_panels.append(Panel(full, label=f"{name} {key} sample {sample_number} full", roi=(roi[0] * 4, roi[1] * 4, roi[2] * 4, roi[3] * 4), source_size=(DISPLAY_SIZE, DISPLAY_SIZE)))
            zoom_panels.append(Panel(zoom, label=f"{name} {key} sample {sample_number} zoom"))
            residual_panels.append(Panel(residual, label=f"{name} {key} sample {sample_number} residual"))
        panels.extend((full_panels, zoom_panels, residual_panels))

    labels = header_labels(psnr_values, ber30)
    footer = "\n".join(footer_lines())
    figure = plt.figure(figsize=(16.8, 10.1), facecolor="white")
    grid = figure.add_gridspec(6, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.10], left=0.074, right=0.985, top=0.79, bottom=0.145, wspace=0.045, hspace=0.22)
    axes: list[list[Any]] = []
    for row_index, row_panels in enumerate(panels):
        row_axes = []
        for col_index, panel in enumerate(row_panels):
            ax = figure.add_subplot(grid[row_index, col_index])
            image = np.asarray(panel.image)
            if row_index % 3 == 2:
                ax.imshow(image, interpolation="nearest", aspect="equal")
            else:
                ax.imshow(image, interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d7dde2")
                spine.set_linewidth(0.7)
            if row_index % 3 == 0:
                rx, ry, rw, rh = panel.roi or (0, 0, 0, 0)
                ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False, edgecolor=RED, linewidth=2.0))
            if row_index == 0:
                ax.set_title(labels[col_index], fontsize=9.0, color=INK, fontweight="bold", pad=10, linespacing=1.15)
            row_axes.append(ax)
        axes.append(row_axes)

    figure.canvas.draw()
    for row_index, row_axes in enumerate(axes):
        position = row_axes[0].get_position()
        figure.text(0.004, position.y0 + position.height / 2.0, rows[row_index], ha="left", va="center", fontsize=8.5 if row_index % 3 != 2 else 8.0, color=INK if row_index % 3 != 2 else MUTED, linespacing=1.15)
        if row_index % 3 == 2:
            cax = figure.add_subplot(grid[row_index, 6])
            scalar = matplotlib.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=residual_ranges[row_index // 3]), cmap="magma")
            colorbar = figure.colorbar(scalar, cax=cax)
            colorbar.ax.tick_params(labelsize=6, colors=MUTED, length=2)
            colorbar.set_label("×10", fontsize=7, color=MUTED, labelpad=2)
    figure.text(0.074, 0.965, f"Figure 2 — {name}", fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    figure.text(0.074, 0.935, "Local-tail fidelity under matched crop robustness; native outputs plus a shared residual diagnostic", fontsize=9.5, color=MUTED, ha="left", va="top")
    figure.text(0.985, 0.965, "red outline = shared ROI", fontsize=8, color=RED, ha="right", va="top")
    for line_index, line in enumerate(footer_lines()):
        figure.text(0.074, 0.105 - line_index * 0.018, line, fontsize=7.2, color=MUTED, ha="left", va="top")
    figure.text(0.985, 0.105, f"PSNR labels use the displayed 512×512 canvas; residual range S1/S2 = {residual_ranges[0]:.3f}/{residual_ranges[1]:.3f}", fontsize=6.8, color=MUTED, ha="right", va="top")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / f"figure2_candidate_{name}"
    figure.savefig(stem.with_suffix(".png"), dpi=350, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    make_pptx(stem.with_name(stem.name + "_pptx").with_suffix(".pptx"), f"Figure 2 — {name}", labels, rows, panels, footer)
    return [stem.with_suffix(".png"), stem.with_suffix(".pdf"), stem.with_name(stem.name + "_pptx").with_suffix(".pptx")]


def display_psnr(display: dict[str, torch.Tensor]) -> dict[str, float]:
    reference = display["original"]
    result = {}
    for key in SOURCE_KEYS[1:]:
        result[key] = psnr(float((display[key] - reference).square().mean()))
    return result


def read_ber30() -> dict[str, float]:
    global_metrics = json.loads((GLOBAL_RESULTS.parent / "metrics.json").read_text())
    hard_metrics = json.loads((HARD_RESULTS.parent / "metrics.json").read_text())
    oklab_rows = list(csv.DictReader((PROJECT / "reports/crop_global_oklab_validation_candidates.csv").open(newline="")))
    oklab_row = next(row for row in oklab_rows if row["global_oklab_weight"] == "0.059149764")
    return {"global": float(global_metrics["ber30"]), "hard": float(hard_metrics["ber30"]), "oklab": float(oklab_row["ber30"])}


def existing_maskwm_metric_audit(manifest: dict) -> dict[str, Any]:
    provenance = json.loads(MASKWM_PROVENANCE.read_text())
    manifest_hash = sha256(MANIFEST)
    current_psnr_csv = PROJECT / "visualizations/external_qualitative/per_image_psnr.csv"
    current_rows = {int(row["validation_index"]): row for row in csv.DictReader(current_psnr_csv.open(newline=""))}
    host_native = display_batch(rgb_from_model(manifest["images"]))
    native = load_maskwm_native()
    current_indices = (0, 10, 20, 30, 40)
    errors = []
    for index in current_indices:
        calculated = psnr(float((native[index] - host_native[index]).square().mean()))
        shown = float(current_rows[index]["MaskWM-D_64_psnr_db"])
        errors.append(abs(calculated - shown))

    montage = MOUNT / "visualizations/external_qualitative/qualitative_5x6_main.png"
    panel_mse: list[float] = []
    if montage.is_file():
        montage_image = np.asarray(Image.open(montage).convert("RGB"), dtype=np.int16)
        for row, index in enumerate(current_indices):
            panel = montage_image[(285 + row * 583):(800 + row * 583), 1395:1911]
            reference = np.asarray(Image.open(MASKWM_NATIVE / f"image_{index:03d}.png").convert("RGB"), dtype=np.int16)
            best = min(
                float(np.mean((panel[dy:dy + 512, dx:dx + 512] - reference) ** 2))
                for dy in range(panel.shape[0] - 512 + 1)
                for dx in range(panel.shape[1] - 512 + 1)
            )
            panel_mse.append(best)

    legacy_csv = PROJECT / "reports/maskwm_results.csv"
    legacy_rows = list(csv.DictReader(legacy_csv.open(newline=""))) if legacy_csv.is_file() else []
    legacy_manifest_hashes = sorted({row.get("manifest_sha256", "") for row in legacy_rows})
    return {
        "provenance_manifest_sha256": provenance.get("manifest_sha256"),
        "expected_manifest_sha256": manifest_hash,
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "output_128_count": len(list(MASKWM_128.glob("image_*.png"))),
        "output_native_512_count": len(list(MASKWM_NATIVE.glob("image_*.png"))),
        "shown_psnr_max_abs_error_db": max(errors) if errors else None,
        "existing_montage_panel_mse_max": max(panel_mse) if panel_mse else None,
        "legacy_report_manifest_sha256": legacy_manifest_hashes,
        "legacy_report_matches_current": manifest_hash in legacy_manifest_hashes,
        "pass": provenance.get("manifest_sha256") == manifest_hash and max(errors) < 1e-3 and len(panel_mse) == 5 and max(panel_mse) < 500.0,
        "figure_metric_scope": "The existing qualitative figure prints PSNR only; SSIM and LPIPS are not shown in its annotations.",
    }


def write_audit_note(audit: dict[str, Any]) -> None:
    lines = [
        "# Figure 2 MaskWM consistency check",
        "",
        "Audit scope: current Figure 2 qualitative assets and the fixed content-selector validation manifest only. No project-test sample or formal Table I input was opened.",
        "",
        "## Verdict",
        "",
        f"**{'PASS' if audit['pass'] else 'FAIL'} for the current visual asset.** The current MaskWM provenance manifest hash matches the fixed validation manifest, all 50 evaluated 128×128 outputs and 50 native 512×512 display outputs are present, and the five displayed PSNR values re-compute from the corresponding native display images.",
        "",
        "## Evidence",
        "",
        f"- Fixed manifest: `{MANIFEST}`",
        f"- Fixed manifest SHA-256: `{audit['expected_manifest_sha256']}`",
        f"- MaskWM checkpoint: `{MASKWM_CHECKPOINT}`",
        f"- MaskWM checkpoint SHA-256: `{audit['checkpoint_sha256']}`",
        f"- MaskWM provenance manifest SHA-256: `{audit['provenance_manifest_sha256']}`",
        f"- Evaluated 128×128 output files: `{audit['output_128_count']}`; native 512×512 display files: `{audit['output_native_512_count']}`.",
        f"- Existing qualitative PSNR annotation maximum absolute re-computation error: `{audit['shown_psnr_max_abs_error_db']:.3e} dB`.",
        f"- Existing montage MaskWM panel maximum rasterized-panel MSE against its native source: `{audit['existing_montage_panel_mse_max']:.3f}` uint8²; the nonzero value is expected from montage rasterization and ROI overlay.",
        "- The existing qualitative figure prints PSNR only; SSIM and LPIPS are not shown in its annotations.",
        "",
        "## Logged scope issue",
        "",
        f"`reports/maskwm_results.csv` records legacy reference rows with manifest hash(es) `{', '.join(audit['legacy_report_manifest_sha256'])}`. They do not match the current fixed validation manifest (`{audit['expected_manifest_sha256']}`) and were not used for the regenerated figures. This is a stale/out-of-scope report row, not a mismatch in the current displayed MaskWM asset.",
        "",
        "The regenerated Figure 2 uses the current `maskwm_validation/D_64bits` provenance and outputs, so the stale legacy row cannot silently supply a metric or image.",
    ]
    AUDIT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def output_metrics(display: dict[str, torch.Tensor], original: torch.Tensor) -> dict[str, dict[str, float]]:
    result = {}
    for key in SOURCE_KEYS[1:]:
        value = display[key]
        reference = display["original"]
        mse = (value - reference).square().mean((1, 2, 3)).numpy()
        ssim = [
            structural_similarity(reference[index].permute(1, 2, 0).numpy(), value[index].permute(1, 2, 0).numpy(), data_range=1.0, channel_axis=2, win_size=7)
            for index in range(50)
        ]
        result[key] = {"psnr": psnr(float(mse.mean())), "ssim": float(np.mean(ssim)), "mse": float(mse.mean())}
    return result


def write_ranking_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist ranking evidence without nested Python-only structures."""
    flattened = []
    for row in rows:
        item = {key: value for key, value in row.items() if key not in {"roi", "local_mse"}}
        item.update({f"roi_local_mse_{key}": value for key, value in row["local_mse"].items()})
        flattened.append(item)
    fields = []
    for row in flattened:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened)


def write_report(audit: dict[str, Any], main_rows: list[dict[str, Any]], main_selected: list[dict[str, Any]], external_rows: list[dict[str, Any]], external_selected: list[dict[str, Any]], display_metrics: dict[str, dict[str, float]], ber30: dict[str, float], figure_paths: list[Path]) -> None:
    def sample_table(title: str, selected: list[dict[str, Any]]) -> list[str]:
        lines = [f"### {title}", "", "| Sample | Validation index | ROI `(x,y,w,h)` | Δ local PSNR (dB) | Δ P95 MSE | ROI residual-energy reduction | Structured-TV reduction | OKLab CIEDE2000 reduction | Selection rationale |", "|---|---:|---|---:|---:|---:|---:|---:|---|"]
        for row in selected:
            rationale = (
                "largest internal local-tail gain with positive P95 reduction and reduced structured residual proxy"
                if title == "Candidate_main" else
                "strong external-method disagreement plus lower Hard Local-Tail local error; retained as a reference contrast"
            )
            lines.append(f"| `{row['image_id']}` | {row['validation_index']} | `({row['roi_x']},{row['roi_y']},{row['roi_width']},{row['roi_height']})` | {row['delta_local_psnr']:+.3f} | {row['delta_p95_reduction']:+.3e} | {row['roi_residual_energy_reduction']:+.3e} | {row['roi_structured_tv_reduction']:+.3e} | {row['roi_ciede2000_reduction']:+.3f} | {rationale} |")
        return lines

    internal_sources = [
        ("MBRS crop-trained global", GLOBAL_CHECKPOINT, GLOBAL_CONFIG, GLOBAL_RESULTS),
        ("Hard Local-Tail (Ours)", HARD_CHECKPOINT, HARD_CONFIG, HARD_RESULTS),
        ("Hard Local-Tail + OKLab", OKLAB_CHECKPOINT, OKLAB_CONFIG, OKLAB_RESULTS),
    ]
    lines = [
        "# Figure 2 regeneration report",
        "",
        "The requested six-column Figure 2 was regenerated from persisted outputs on the fixed 50-image validation manifest. No project-test sample, formal Table I result, or checkpoint was modified.",
        "",
        "## Recommendation",
        "",
        "Use `figure2_candidate_main` in the main paper. It directly supports the primary local-tail claim with the promoted matched crop-robustness strong-embedding pair; `figure2_candidate_external` is the alternate when external-method contrast is the editorial priority. Both are deliberately metric-selected best-case qualitative evidence and must not be described as representative sampling.",
        "",
        "## MaskWM consistency",
        "",
        f"- Current MaskWM visual asset: **{'PASS' if audit['pass'] else 'FAIL'}**; see [`figure2_maskwm_consistency_check.md`](figure2_maskwm_consistency_check.md).",
        f"- Current provenance and display outputs use the fixed manifest hash `{audit['expected_manifest_sha256']}`.",
        f"- Recomputed PSNR annotation error is `{audit['shown_psnr_max_abs_error_db']:.3e} dB`; the existing figure shows PSNR only, not SSIM or LPIPS.",
        f"- The legacy `maskwm_results.csv` hash is out of scope/current-manifest mismatched (`{', '.join(audit['legacy_report_manifest_sha256'])}`); it was not consumed.",
        "",
        "## Fixed protocol",
        "",
        f"- Manifest: `{MANIFEST}`",
        f"- Manifest SHA-256: `{audit['expected_manifest_sha256']}`",
        "- Split: validation; 50 images; 64-bit messages; source size 128×128.",
        "- Selection and rendering use saved tensors/PNG outputs only. No project-test samples, attack realizations, manual edits, per-method ROI, or per-method residual normalization were used.",
        "",
        "## Checkpoints, configs, and exact outputs",
        "",
        "| Method | Checkpoint | Config | Exact visual output | Status |",
        "|---|---|---|---|---|",
        f"| HiDDeN-64 (external) | `{HIDDEN_CHECKPOINT}` | `{HIDDEN_CHECKPOINT.parent / 'training_config.json'}` | `{HIDDEN_CACHE}` | external reference; current-manifest cache |",
        f"| MaskWM-D_64 (external) | `{MASKWM_CHECKPOINT}` | `{MASKWM_PROVENANCE}` | `{MASKWM_128}` and `{MASKWM_NATIVE}` | external reference under released checkpoint |",
    ]
    for label, checkpoint, config, output in internal_sources:
        lines.append(f"| {label} | `{checkpoint}` | `{config}` | `{output}` | {'validation-only extension' if 'OKLab' in label else 'validation-only strong-embedding visualization'} |")
    lines += [
        "",
        "### Strong-embedding operating point",
        "",
        "The two internal comparison columns use the promoted `w_msg=80`, `λ_img=1` pair from the same crop-trained epoch-100 source. This is a validation-only visualization setting, not a replacement for formal Table I.",
        "",
        "| Method | BER30 | PSNR (frozen 128 source metric) | Local PSNR | Checkpoint SHA-256 |",
        "|---|---:|---:|---:|---|",
        f"| MBRS crop-trained global | {ber30['global']:.6f} | 31.486392 | 30.780588 | `{sha256(GLOBAL_CHECKPOINT)}` |",
        f"| Hard Local-Tail (Ours) | {ber30['hard']:.6f} | 32.353024 | 31.610333 | `{sha256(HARD_CHECKPOINT)}` |",
        "",
        f"Hard Local-Tail + OKLab uses the validation-only `g25` run (`λ_OKLab=0.059149764`, checkpoint `{OKLAB_CHECKPOINT}`). It showed the clearest measured color benefit among the available OKLab variants, but failed the complete preservation gate because Gini increased beyond tolerance; it is included only as a validation-only extension.",
        "",
        "## Display metrics used in the figure headers",
        "",
        "PSNR labels in the PNG/PDF/PPTX are computed on the displayed 512×512 canvas used by every column. This makes the annotation correspond to the rendered panel; the strong-pair frozen 128-source metrics are recorded above.",
        "",
        "| Column | Display PSNR | Display SSIM | BER30 annotation |",
        "|---|---:|---:|---:|",
        "| Original | reference | reference | — |",
        f"| HiDDeN-64 (external) | {display_metrics['hidden']['psnr']:.3f} dB | {display_metrics['hidden']['ssim']:.5f} | — |",
        f"| MaskWM-D_64 (external) | {display_metrics['maskwm']['psnr']:.3f} dB | {display_metrics['maskwm']['ssim']:.5f} | — |",
        f"| MBRS crop-trained global | {display_metrics['global']['psnr']:.3f} dB | {display_metrics['global']['ssim']:.5f} | {ber30['global']:.3f} |",
        f"| Hard Local-Tail (Ours) | {display_metrics['hard']['psnr']:.3f} dB | {display_metrics['hard']['ssim']:.5f} | {ber30['hard']:.3f} |",
        f"| Hard Local-Tail + OKLab | {display_metrics['oklab']['psnr']:.3f} dB | {display_metrics['oklab']['ssim']:.5f} | validation-only |",
        "",
    ]
    lines += sample_table("Candidate_main", main_selected)
    lines += ["", "The complete main and external rankings are persisted in `reports/figure2_candidate_main_ranking.csv` and `reports/figure2_candidate_external_ranking.csv`. Main ranking priority was local PSNR improvement, then P95 local patch-MSE reduction, ROI residual-energy reduction, and a structured-residual TV proxy.", ""]
    lines += sample_table("Candidate_external", external_selected)
    lines += ["", "Candidate_external prioritizes external local-error disagreement and the gap between the external references and Hard Local-Tail, while retaining the internal ROI gain as a secondary term. It uses the same fixed manifest, shared host, shared ROI per sample, and same normalization policy.", "", "## Generated files", ""]
    for path in figure_paths:
        lines.append(f"- [`{path.relative_to(PROJECT)}`]({path}) — SHA-256 `{sha256(path)}`")
    lines.append("- `reports/figure2_candidate_main_ranking.csv` and `reports/figure2_candidate_external_ranking.csv` — complete selection rankings")
    lines += [
        "",
        "## Figure annotations and honesty boundary",
        "",
        "- Exact method names in all generated files are: Original; HiDDeN-64 (external); MaskWM-D_64 (external); MBRS crop-trained global; Hard Local-Tail (Ours); Hard Local-Tail + OKLab.",
        "- Full rows show the same host canvas and a shared red ROI rectangle. Zoom rows use the same 8× source-pixel scale; smaller ROIs are centered on a common canvas.",
        "- Residual rows show mean absolute RGB residual after normalization, multiplied by ×10 for visualization only. Each sample has one shared magma range across all six methods.",
        "- External methods are reference baselines under released checkpoints, not strictly matched training baselines. OKLab is validation-only. Selection is disclosed as best-case and non-representative.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest = load(MANIFEST)
    if manifest.get("split") != "validation" or len(manifest.get("images", [])) != 50:
        raise RuntimeError("Figure 2 requires the fixed 50-image validation manifest")
    original = rgb_from_model(manifest["images"])
    values = {
        "original": original,
        "hidden": load_encoded(HIDDEN_CACHE),
        "maskwm": load_maskwm_128(),
        "global": load_encoded(GLOBAL_RESULTS),
        "hard": load_encoded(HARD_RESULTS),
        "oklab": load_encoded(OKLAB_RESULTS),
    }
    if any(value.shape != original.shape for value in values.values()):
        raise RuntimeError(f"source shape mismatch: {[tuple(value.shape) for value in values.values()]}")
    for output_path in (HIDDEN_CACHE, GLOBAL_RESULTS, HARD_RESULTS, OKLAB_RESULTS):
        payload = load(output_path)
        if isinstance(payload, dict) and isinstance(payload.get("images"), torch.Tensor):
            if not torch.equal(payload["images"].float(), manifest["images"].float()):
                raise RuntimeError(f"saved output host mismatch: {output_path}")

    audit = existing_maskwm_metric_audit(manifest)
    write_audit_note(audit)
    main_rows, main_selected = rank_candidates(manifest, original, values)
    external_rows, external_selected = rank_external(main_rows, main_selected)
    write_ranking_csv(PROJECT / "reports/figure2_candidate_main_ranking.csv", main_rows)
    write_ranking_csv(PROJECT / "reports/figure2_candidate_external_ranking.csv", external_rows)

    display = {
        "original": display_batch(values["original"]),
        "hidden": display_batch(values["hidden"]),
        "maskwm": load_maskwm_native(),
        "global": display_batch(values["global"]),
        "hard": display_batch(values["hard"]),
        "oklab": display_batch(values["oklab"]),
    }
    psnr_values = display_psnr(display)
    ber30 = read_ber30()
    generated = render_candidate("main", main_selected, display, psnr_values, ber30)
    generated += render_candidate("external", external_selected, display, psnr_values, ber30)
    display_metrics = output_metrics(display, original)
    write_report(audit, main_rows, main_selected, external_rows, external_selected, display_metrics, ber30, generated)

    print(f"MaskWM check: {'PASS' if audit['pass'] else 'FAIL'}")
    print(f"Candidate_main samples: {[row['image_id'] for row in main_selected]}")
    print(f"Candidate_external samples: {[row['image_id'] for row in external_selected]}")
    for path in generated:
        print(path)
    print("Recommendation: visualizations/figure2/figure2_candidate_main.{png,pdf} for the main paper; the external candidate is the contrast-focused alternate.")


if __name__ == "__main__":
    main()
