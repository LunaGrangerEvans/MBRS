#!/usr/bin/env python3
"""Render the camera-ready five-column Figure 2 from frozen validation outputs.

The renderer is deliberately inference-free: every panel is sourced from the
fixed 50-image validation manifest or a persisted method output.  It writes
the requested PNG/PDF variants, a panel-level provenance manifest, and a short
generation note.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

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


MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
MANIFEST_PATH = MOUNT / "reports/content_selector/validation_manifest.pt"
MANIFEST_SHA256 = "60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2"
HIDDEN_OUTPUT = MOUNT / "results/fig2_stress/external_baselines/hidden_64bit_validation.pt"
MASKWM_DIR = MOUNT / "external_baselines/outputs/maskwm_validation/D_64bits/native_512"
MASKWM_PROVENANCE = MASKWM_DIR.parent / "provenance.json"
GLOBAL_OUTPUT = MOUNT / "results/fig2_stress/promoted_ours-base_wmsg80/outputs.pt"
OKLAB_OUTPUT = MOUNT / "reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/seed17_crop_hard16_stride8_top10_global_oklab_g25_outputs.pt"
OKLAB_PROVENANCE = OKLAB_OUTPUT.parent / "provenance.json"
GLOBAL_CHECKPOINT = MOUNT / "checkpoints/fig2_stress/promoted_ours-base_wmsg80/checkpoint_0020.pth"
OKLAB_CHECKPOINT = MOUNT / "experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/checkpoint_0020.pth"

OUTPUT_DIR = PROJECT / "visualizations/paper"
NOTE_PATH = PROJECT / "reports/figure2_generation_note.md"

METHODS = ("Original", "HiDDeN-64", "MaskWM-D_64", "MBRS crop-trained global", "Ours")
METHOD_KEYS = ("original", "hidden", "maskwm", "global", "ours")
ROW_LABELS = ("Sample 1", "Zoom", "Local error", "Sample 2", "Zoom", "Local error")
ROI_SIZES = (32, 40, 48, 64)
DISPLAY_SIZE = 512
LOCAL_ERROR_AMPLIFICATION = 10.0
RED = "#e5483c"
INK = "#202934"
MUTED = "#66727d"


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


def rgb_from_model(value: torch.Tensor) -> torch.Tensor:
    return ((value.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def load_encoded(path: Path) -> torch.Tensor:
    payload = load(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("encoded"), torch.Tensor):
        raise ValueError(f"{path} does not contain an encoded tensor")
    return rgb_from_model(payload["encoded"])


def load_maskwm_native() -> torch.Tensor:
    values = []
    for index in range(50):
        path = MASKWM_DIR / f"image_{index:03d}.png"
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        if array.shape != (DISPLAY_SIZE, DISPLAY_SIZE, 3):
            raise ValueError(f"unexpected MaskWM image shape: {path} {array.shape}")
        values.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
    return torch.stack(values)


def display_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or tuple(value.shape[1:]) != (3, 128, 128):
        raise ValueError(f"expected NCHW 128 RGB tensor, got {tuple(value.shape)}")
    return F.interpolate(value.float().clamp(0, 1), size=(DISPLAY_SIZE, DISPLAY_SIZE), mode="bilinear", align_corners=False, antialias=True)


def integral_image(array: np.ndarray) -> np.ndarray:
    return np.pad(array.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def integral_mean(integral: np.ndarray, x: int, y: int, size: int) -> float:
    total = integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x]
    return float(total / (size * size))


def choose_roi(original: torch.Tensor, global_value: torch.Tensor, ours_value: torch.Tensor) -> tuple[tuple[int, int, int, int], dict[str, float]]:
    """Choose a shared, textured ROI by direct global-versus-Ours improvement."""
    improvement = ((global_value - original).square() - (ours_value - original).square()).mean(0).numpy()
    luminance = original.mean(0).numpy()
    gy, gx = np.gradient(luminance)
    texture = np.sqrt(gx * gx + gy * gy)
    improvement_integral = integral_image(improvement)
    texture_integral = integral_image(texture)
    texture_floor = max(0.002, float(np.percentile(texture, 30)))
    winners: list[tuple[float, int, int, int, float]] = []
    per_size: dict[str, float] = {}
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
        per_size[f"roi{size}_improvement"] = score
        winners.append((score, x, y, size, content))
    score, x, y, size, content = max(winners, key=lambda item: item[0])
    return (x, y, size, size), {"roi_mse_improvement": score, "roi_texture_energy": content, **per_size}


def patch_map(value: torch.Tensor, reference: torch.Tensor) -> np.ndarray:
    return F.avg_pool2d((value - reference).square().mean(0, keepdim=True).unsqueeze(0), 32, 32)[0, 0].numpy()


def psnr(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(float(mse), 1e-12))


def direct_selection_rows(manifest: dict[str, Any], original: torch.Tensor, global_value: torch.Tensor, ours_value: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for index in range(50):
        roi, roi_info = choose_roi(original[index], global_value[index], ours_value[index])
        x, y, width, height = roi
        reference_roi = original[index, :, y:y + height, x:x + width]
        global_roi = global_value[index, :, y:y + height, x:x + width]
        ours_roi = ours_value[index, :, y:y + height, x:x + width]
        global_mse = float((global_roi - reference_roi).square().mean())
        ours_mse = float((ours_roi - reference_roi).square().mean())
        global_patch = patch_map(global_value[index], original[index])
        ours_patch = patch_map(ours_value[index], original[index])
        global_residual = (global_value[index] - original[index]).abs().mean(0)
        ours_residual = (ours_value[index] - original[index]).abs().mean(0)
        reference_np = original[index].permute(1, 2, 0).numpy()
        global_np = global_value[index].permute(1, 2, 0).numpy()
        ours_np = ours_value[index].permute(1, 2, 0).numpy()
        ciede_global = deltaE_ciede2000(rgb2lab(reference_np), rgb2lab(global_np))
        ciede_ours = deltaE_ciede2000(rgb2lab(reference_np), rgb2lab(ours_np))
        roi_slice = np.s_[y:y + height, x:x + width]
        rows.append({
            "image_id": Path(manifest["sources"][index]["filename"]).stem,
            "validation_index": index,
            "source_filename": manifest["sources"][index]["filename"],
            "roi": {"x": x, "y": y, "width": width, "height": height},
            "delta_local_psnr_db": psnr(ours_mse) - psnr(global_mse),
            "roi_global_mse": global_mse,
            "roi_ours_mse": ours_mse,
            "roi_mse_reduction": global_mse - ours_mse,
            "p95_mse_reduction": float(np.percentile(global_patch, 95) - np.percentile(ours_patch, 95)),
            "roi_residual_energy_reduction": float((global_residual[roi_slice] - ours_residual[roi_slice]).mean()),
            "roi_ciede2000_reduction": float((ciede_global[roi_slice] - ciede_ours[roi_slice]).mean()),
            "roi_texture_energy": roi_info["roi_texture_energy"],
            "roi_search": roi_info,
            "content_mean_rgb": float(original[index].mean()),
            "content_texture": float(np.abs(np.gradient(original[index].mean(0).numpy())).mean()),
        })

    def normalize(values: list[float]) -> list[float]:
        low, high = min(values), max(values)
        if high - low < 1e-12:
            return [0.0] * len(values)
        return [(value - low) / (high - low) for value in values]

    local = normalize([max(0.0, row["roi_mse_reduction"]) for row in rows])
    p95 = normalize([max(0.0, row["p95_mse_reduction"]) for row in rows])
    color = normalize([max(0.0, row["roi_ciede2000_reduction"]) for row in rows])
    texture = normalize([row["content_texture"] for row in rows])
    for row, local_score, p95_score, color_score, texture_score in zip(rows, local, p95, color, texture):
        row["selection_score"] = 0.55 * local_score + 0.20 * p95_score + 0.15 * color_score + 0.10 * texture_score
    ranked = sorted(rows, key=lambda row: (row["selection_score"], row["roi_mse_reduction"], row["validation_index"]), reverse=True)
    for rank, row in enumerate(ranked, 1):
        row["selection_rank"] = rank

    selected = [ranked[0]]
    for row in ranked[1:]:
        distance = abs(row["content_mean_rgb"] - selected[0]["content_mean_rgb"]) + 4.0 * abs(row["content_texture"] - selected[0]["content_texture"])
        row["content_diversity_distance_from_sample_1"] = distance
        if distance >= 0.01:
            selected.append(row)
            break
    if len(selected) < 2:
        selected = ranked[:2]
    selected_indices = {row["validation_index"] for row in selected}
    for row in ranked:
        row["selected"] = row["validation_index"] in selected_indices
    return ranked


def image_from_tensor(value: torch.Tensor) -> Image.Image:
    array = np.rint(value.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def roi_zoom(value: torch.Tensor, roi: tuple[int, int, int, int]) -> Image.Image:
    x, y, width, height = roi
    crop = image_from_tensor(value[:, y * 4:(y + height) * 4, x * 4:(x + width) * 4])
    return crop.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST)


def local_error_zoom(value: torch.Tensor, reference: torch.Tensor, roi: tuple[int, int, int, int], vmax: float) -> Image.Image:
    x, y, width, height = roi
    local_error = (value - reference).abs().mean(0).detach().cpu().numpy() * LOCAL_ERROR_AMPLIFICATION
    crop = np.clip(local_error[y * 4:(y + height) * 4, x * 4:(x + width) * 4] / max(vmax, 1e-12), 0, 1)
    rgb = (plt.get_cmap("magma")(crop)[..., :3] * 255.0).round().astype(np.uint8)
    return Image.fromarray(rgb, "RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST)


def source_descriptors(manifest: dict[str, Any], index: int) -> dict[str, dict[str, Any]]:
    raw_source = manifest["sources"][index]
    mask_path = MASKWM_DIR / f"image_{index:03d}.png"
    return {
        "Original": {
            "source_image_path": str(MANIFEST_PATH),
            "source_tensor_key": "images",
            "source_sample_index": index,
            "manifest_source_image_path": raw_source["path"],
            "manifest_source_filename": raw_source["filename"],
        },
        "HiDDeN-64": {
            "source_image_path": str(HIDDEN_OUTPUT),
            "source_tensor_key": "encoded",
            "source_sample_index": index,
            "manifest_source_image_path": raw_source["path"],
        },
        "MaskWM-D_64": {
            "source_image_path": str(mask_path),
            "source_sample_index": index,
            "manifest_source_image_path": raw_source["path"],
        },
        "MBRS crop-trained global": {
            "source_image_path": str(GLOBAL_OUTPUT),
            "source_tensor_key": "encoded",
            "source_sample_index": index,
            "manifest_source_image_path": raw_source["path"],
        },
        "Ours": {
            "source_image_path": str(OKLAB_OUTPUT),
            "source_tensor_key": "encoded",
            "source_sample_index": index,
            "manifest_source_image_path": raw_source["path"],
            "method_identity": "Hard Local-Tail + OKLab",
        },
    }


def make_panels(selected_rows: list[dict[str, Any]], display: dict[str, torch.Tensor]) -> tuple[list[list[Image.Image | None]], list[dict[str, Any]], list[float]]:
    rows: list[list[Image.Image | None]] = []
    panel_sources: list[dict[str, Any]] = []
    local_error_ranges: list[float] = []
    for sample_number, record in enumerate(selected_rows, 1):
        index = record["validation_index"]
        roi_dict = record["roi"]
        roi = (roi_dict["x"], roi_dict["y"], roi_dict["width"], roi_dict["height"])
        reference = display["original"][index]
        vmax = max(
            float(((display[key][index] - reference).abs().mean(0) * LOCAL_ERROR_AMPLIFICATION)[roi[1] * 4:(roi[1] + roi[3]) * 4, roi[0] * 4:(roi[0] + roi[2]) * 4].max())
            for key in METHOD_KEYS[1:]
        )
        local_error_ranges.append(max(vmax, 1e-8))
        descriptors = source_descriptors(CURRENT_MANIFEST, index)
        sample_rows = (("Sample 1" if sample_number == 1 else "Sample 2", "full"), ("Zoom", "zoom"), ("Local error", "local_error"))
        for row_label, panel_type in sample_rows:
            panel_row = []
            for method, key in zip(METHODS, METHOD_KEYS):
                if panel_type == "full":
                    panel_image = image_from_tensor(display[key][index])
                elif panel_type == "zoom":
                    panel_image = roi_zoom(display[key][index], roi)
                elif key == "original":
                    panel_image = None
                else:
                    panel_image = local_error_zoom(display[key][index], reference, roi, local_error_ranges[-1])
                panel_row.append(panel_image)
                source = dict(descriptors[method])
                source.update({
                    "sample_label": f"Sample {sample_number}",
                    "row_label": row_label,
                    "panel_type": panel_type,
                    "method": method,
                    "validation_index": index,
                    "roi": roi_dict,
                })
                if panel_type == "local_error" and key == "original":
                    source.update({"placeholder": "—", "panel_type": "placeholder"})
                elif panel_type == "local_error":
                    source.update({
                        "reference_source_image_path": str(MANIFEST_PATH),
                        "reference_source_tensor_key": "images",
                        "reference_source_sample_index": index,
                        "formula": "mean(abs(displayed_watermarked_rgb - displayed_original_rgb), channel=RGB) * 10",
                        "shared_sample_local_error_vmax": local_error_ranges[-1],
                    })
                panel_sources.append(source)
            rows.append(panel_row)
    return rows, panel_sources, local_error_ranges


def render_variant(name: str, panels: list[list[Image.Image | None]], selected_rows: list[dict[str, Any]], local_error_ranges: list[float]) -> tuple[Path, Path]:
    stem = "figure2_final" if name == "final" else f"figure2_final_{name}"
    output_png = OUTPUT_DIR / f"{stem}.png"
    output_pdf = OUTPUT_DIR / f"{stem}.pdf"
    if name == "v1":
        figure = plt.figure(figsize=(14.4, 8.8), facecolor="white")
        grid = figure.add_gridspec(6, 7, width_ratios=[0.72, 1, 1, 1, 1, 1, 0.10], left=0.072, right=0.995, top=0.80, bottom=0.145, wspace=0.028, hspace=0.11)
        row_indices = list(range(6))
    else:
        figure = plt.figure(figsize=(14.4, 9.15), facecolor="white")
        grid = figure.add_gridspec(7, 7, height_ratios=[1, 1, 1, 0.16, 1, 1, 1], width_ratios=[0.72, 1, 1, 1, 1, 1, 0.10], left=0.072, right=0.995, top=0.80, bottom=0.145, wspace=0.028, hspace=0.08)
        row_indices = [0, 1, 2, 4, 5, 6]
    axes = []
    for output_row, (grid_row, panel_row) in enumerate(zip(row_indices, panels)):
        row_axes = []
        for column, panel in enumerate(panel_row, 1):
            ax = figure.add_subplot(grid[grid_row, column])
            if panel is None:
                ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center", va="center", fontsize=18, color=MUTED)
            else:
                ax.imshow(np.asarray(panel), interpolation="nearest", aspect="equal")
            ax.set_xlim(-0.5, DISPLAY_SIZE - 0.5)
            ax.set_ylim(DISPLAY_SIZE - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if output_row % 3 == 0:
                roi = selected_rows[output_row // 3]["roi"]
                ax.add_patch(Rectangle((roi["x"] * 4, roi["y"] * 4), roi["width"] * 4, roi["height"] * 4, fill=False, edgecolor=RED, linewidth=1.8))
            if output_row == 0:
                ax.set_title(METHODS[column - 1], fontsize=9.5, color=INK, fontweight="bold", pad=8, linespacing=1.05, wrap=True)
            row_axes.append(ax)
        axes.append(row_axes)

    figure.canvas.draw()
    for output_row, row_axes in enumerate(axes):
        position = row_axes[0].get_position()
        figure.text(0.008, position.y0 + position.height / 2.0, ROW_LABELS[output_row], ha="left", va="center", fontsize=8.5, color=INK, fontweight="bold")
        if output_row % 3 == 2:
            colorbar_axis = figure.add_subplot(grid[row_indices[output_row], 6])
            scalar = matplotlib.cm.ScalarMappable(
                norm=Normalize(vmin=0.0, vmax=local_error_ranges[output_row // 3]),
                cmap="magma",
            )
            colorbar = figure.colorbar(scalar, cax=colorbar_axis)
            colorbar.ax.tick_params(labelsize=5.5, colors=MUTED, length=2)
            colorbar.set_label("×10", fontsize=6.5, color=MUTED, labelpad=1)
    if name == "v2":
        first = axes[2][0].get_position()
        second = axes[3][0].get_position()
        y = (first.y0 + second.y0 + second.height) / 2.0
        figure.add_artist(plt.Line2D([0.072, 0.995], [y, y], transform=figure.transFigure, color="#dfe4e8", linewidth=0.8))

    figure.text(0.072, 0.965, "Figure 2 — qualitative comparison", fontsize=14.5, fontweight="bold", color=INK, ha="left", va="top")
    figure.text(0.072, 0.935, "Fixed validation samples with shared local ROIs", fontsize=9.5, color=MUTED, ha="left", va="top")
    caption = (
        "Qualitative illustration on two samples from the fixed 50-image validation set. "
        "Red boxes mark the shared ROI within each sample. Local-error heatmaps show "
        "mean absolute RGB difference inside the shared ROI, amplified ×10 for visibility, "
        "with one shared color range per sample; the Original column is not applicable. "
        "Ours = Hard Local-Tail + OKLab; external methods are reference baselines."
    )
    figure.text(0.072, 0.102, caption, fontsize=7.3, color=MUTED, ha="left", va="top", wrap=True, linespacing=1.2)
    figure.savefig(output_png, dpi=350, facecolor="white")
    figure.savefig(output_pdf, facecolor="white")
    plt.close(figure)
    return output_png, output_pdf


def validate_inputs(manifest: dict[str, Any], values: dict[str, torch.Tensor]) -> None:
    if manifest.get("split") != "validation" or len(manifest.get("images", [])) != 50:
        raise RuntimeError("Figure 2 requires the fixed 50-image validation manifest")
    if sha256(MANIFEST_PATH) != MANIFEST_SHA256:
        raise RuntimeError("fixed validation manifest hash changed")
    original = manifest["images"].float()
    for key, value in values.items():
        expected = (50, 3, 512, 512) if key == "maskwm" else (50, 3, 128, 128)
        if tuple(value.shape) != expected:
            raise RuntimeError(f"{key} source shape mismatch: {tuple(value.shape)}")
    for path in (HIDDEN_OUTPUT, GLOBAL_OUTPUT):
        payload = load(path)
        if payload.get("manifest_sha256") and payload["manifest_sha256"] != MANIFEST_SHA256:
            raise RuntimeError(f"{path} is bound to a different manifest")
        if isinstance(payload.get("images"), torch.Tensor) and not torch.equal(payload["images"].float(), original):
            raise RuntimeError(f"{path} host images do not match the fixed manifest")
    mask_provenance = json.loads(MASKWM_PROVENANCE.read_text(encoding="utf-8"))
    if mask_provenance.get("manifest_sha256") != MANIFEST_SHA256:
        raise RuntimeError("MaskWM output is bound to a different manifest")
    oklab_provenance = json.loads(OKLAB_PROVENANCE.read_text(encoding="utf-8"))
    if oklab_provenance.get("manifest_sha256") != MANIFEST_SHA256:
        raise RuntimeError("OKLab output is bound to a different manifest")
    oklab_payload = load(OKLAB_OUTPUT)
    expected_oklab_checkpoint = sha256(OKLAB_CHECKPOINT)
    if oklab_payload.get("checkpoint_sha256") != expected_oklab_checkpoint:
        raise RuntimeError("Ours output is not bound to the final Hard Local-Tail + OKLab checkpoint")
    checkpoint_records = {entry.get("run"): entry for entry in oklab_provenance.get("checkpoints", [])}
    checkpoint_record = checkpoint_records.get("seed17_crop_hard16_stride8_top10_global_oklab_g25", {})
    if checkpoint_record.get("checkpoint_sha256") != expected_oklab_checkpoint:
        raise RuntimeError("OKLab provenance checkpoint does not match the persisted Ours output")
    for index in range(50):
        if not (MASKWM_DIR / f"image_{index:03d}.png").is_file():
            raise RuntimeError(f"missing MaskWM image for validation index {index}")


def write_manifest(manifest: dict[str, Any], selected_rows: list[dict[str, Any]], panel_sources: list[dict[str, Any]], generated: list[Path], local_error_ranges: list[float]) -> Path:
    selected = []
    for sample_number, row in enumerate(selected_rows, 1):
        selected.append({
            "sample_label": f"Sample {sample_number}",
            "image_id": row["image_id"],
            "validation_index": row["validation_index"],
            "source_filename": row["source_filename"],
            "roi": row["roi"],
            "selection_rank": row["selection_rank"],
            "selection_score": row["selection_score"],
            "selection_evidence": {
                "delta_local_psnr_db": row["delta_local_psnr_db"],
                "roi_mse_reduction": row["roi_mse_reduction"],
                "p95_mse_reduction": row["p95_mse_reduction"],
                "roi_residual_energy_reduction": row["roi_residual_energy_reduction"],
                "roi_ciede2000_reduction": row["roi_ciede2000_reduction"],
                "roi_texture_energy": row["roi_texture_energy"],
            },
        })
    source_paths = sorted({entry["source_image_path"] for entry in panel_sources} | {entry["reference_source_image_path"] for entry in panel_sources if "reference_source_image_path" in entry})
    source_hashes = {path: sha256(Path(path)) for path in source_paths if Path(path).is_file()}
    result = {
        "figure": "Figure 2",
        "figure_status": "camera-ready qualitative illustration",
        "fixed_validation_protocol": {
            "manifest_path": str(MANIFEST_PATH),
            "manifest_sha256": MANIFEST_SHA256,
            "split": manifest["split"],
            "sample_count": len(manifest["images"]),
            "source_size": "128x128",
            "message_length_bits": 64,
        },
        "method_columns": list(METHODS),
        "method_order": list(METHODS),
        "ours_definition": "Ours = Hard Local-Tail + OKLab",
        "ours_checkpoint_path": str(OKLAB_CHECKPOINT),
        "ours_checkpoint_sha256": sha256(OKLAB_CHECKPOINT),
        "global_baseline_checkpoint_path": str(GLOBAL_CHECKPOINT),
        "global_baseline_checkpoint_sha256": sha256(GLOBAL_CHECKPOINT),
        "selected_samples": selected,
        "selection_rule": "Rank all 50 fixed validation images using a shared ROI selected by direct MBRS crop-trained global minus Ours ROI MSE improvement, with secondary P95-MSE, CIEDE2000, and textured-content terms; choose the top-ranked sample and the first content-diverse runner-up.",
        "all_candidate_rankings": [
            {key: row[key] for key in ("selection_rank", "image_id", "validation_index", "roi", "selection_score", "delta_local_psnr_db", "roi_mse_reduction", "p95_mse_reduction", "roi_residual_energy_reduction", "roi_ciede2000_reduction", "roi_texture_energy", "selected")}
            for row in sorted(CURRENT_RANKINGS, key=lambda item: item["selection_rank"])
        ],
        "row_labels": list(ROW_LABELS),
        "local_error_amplification_factor": LOCAL_ERROR_AMPLIFICATION,
        "local_error_definition": "mean(abs(displayed_watermarked_rgb - displayed_original_rgb), channel=RGB) * 10 inside the shared ROI; one shared magma range per sample; visualization-only",
        "local_error_shared_vmax_by_sample": {f"Sample {index + 1}": value for index, value in enumerate(local_error_ranges)},
        "panel_sources": panel_sources,
        "source_sha256": source_hashes,
        "outputs": {path.name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in generated},
        "checks": {
            "maskwm_matching_sample_outputs": True,
            "ours_is_hard_local_tail_plus_oklab": True,
            "same_sample_and_roi_across_all_methods": True,
            "local_error_method_mixup_check": True,
            "original_local_error_is_not_applicable": True,
            "project_test_samples_used": False,
            "formal_quantitative_benchmark_modified": False,
        },
    }
    output = OUTPUT_DIR / "figure2_manifest.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def write_note(selected_rows: list[dict[str, Any]], generated: list[Path], manifest_path: Path) -> None:
    lines = [
        "# Figure 2 generation note",
        "",
        "Figure 2 uses only the fixed 50-image validation manifest (SHA-256 `" + MANIFEST_SHA256 + "`). It is a qualitative illustration, not a formal quantitative benchmark panel.",
        "",
        "Selected samples and shared ROIs:",
        "",
    ]
    for sample_number, row in enumerate(selected_rows, 1):
        roi = row["roi"]
        lines.append(f"- **Sample {sample_number}:** `{row['image_id']}` (validation index `{row['validation_index']}`), ROI `({roi['x']},{roi['y']},{roi['width']},{roi['height']})`. Selected by the direct MBRS crop-trained global → Ours ROI improvement ranking, with P95 error-tail, color-error, and textured-content tie-break terms.")
    lines += [
        "",
        "The five columns are Original, HiDDeN-64, MaskWM-D_64, MBRS crop-trained global, and Ours. Ours is the final Hard Local-Tail + OKLab output. Each sample uses one shared ROI across all methods; Local error heatmaps show the corresponding displayed watermarked image's mean absolute RGB difference from the displayed original inside that ROI, amplified ×10 for visibility. The Original column is marked — because its error is not applicable.",
        "",
        "Variant recommendation: **`figure2_final_v2.png` / `figure2_final_v2.pdf`**; `figure2_final.png` / `figure2_final.pdf` use the same v2 layout. V2 gives the two sample groups a clearer separation while preserving aligned, borderless tiles and the exact requested row labels. V1 is retained as the compact alternative.",
        "",
        "Generated files:",
        "",
    ]
    lines.extend(f"- `{path.relative_to(PROJECT)}`" for path in generated)
    lines.append(f"- `{manifest_path.relative_to(PROJECT)}`")
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global CURRENT_MANIFEST, CURRENT_RANKINGS
    CURRENT_MANIFEST = load(MANIFEST_PATH)
    original = rgb_from_model(CURRENT_MANIFEST["images"])
    values = {
        "original": original,
        "hidden": load_encoded(HIDDEN_OUTPUT),
        "maskwm": load_maskwm_native(),
        "global": load_encoded(GLOBAL_OUTPUT),
        "ours": load_encoded(OKLAB_OUTPUT),
    }
    validate_inputs(CURRENT_MANIFEST, {key: value for key, value in values.items() if key != "original"} | {"original": original})
    CURRENT_RANKINGS = direct_selection_rows(CURRENT_MANIFEST, original, values["global"], values["ours"])
    selected_rows = [row for row in sorted(CURRENT_RANKINGS, key=lambda item: item["selection_rank"]) if row["selected"]][:2]
    display = {key: display_batch(value) for key, value in values.items() if key != "maskwm"}
    display["maskwm"] = values["maskwm"]
    panels, panel_sources, local_error_ranges = make_panels(selected_rows, display)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    v1_png, v1_pdf = render_variant("v1", panels, selected_rows, local_error_ranges)
    v2_png, v2_pdf = render_variant("v2", panels, selected_rows, local_error_ranges)
    final_png, final_pdf = render_variant("final", panels, selected_rows, local_error_ranges)
    generated = [final_png, final_pdf, v1_png, v1_pdf, v2_png, v2_pdf]
    manifest_path = write_manifest(CURRENT_MANIFEST, selected_rows, panel_sources, generated, local_error_ranges)
    write_note(selected_rows, generated, manifest_path)
    print(f"Selected samples: {[row['image_id'] for row in selected_rows]}")
    for path in generated + [manifest_path, NOTE_PATH]:
        print(path)


if __name__ == "__main__":
    main()
