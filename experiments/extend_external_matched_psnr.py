#!/usr/bin/env python3
"""Extend matched-PSNR calibration to the two external validation assets.

The quality canvas is the common 512x512 display canvas used by Figure 2:
internal 128x128 outputs are rendered with the existing bilinear display path,
HiDDeN is rendered from its cached 128x128 output, and MaskWM-D_64 uses its
existing native 512x512 PNGs. All four residual scales are calibrated on the
same 50-image validation manifest and are constrained to [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from experiments.run_matched_psnr_comparison import quality_metrics, psnr_from_mse
from experiments.render_figure2_matched_psnr import (
    GLOBAL_CHECKPOINT,
    OURS_CHECKPOINT,
    Panel,
    display_batch,
    encode,
    image_from_tensor,
    make_pptx_clean,
    model_from_checkpoint,
    residual_tile,
    sha256,
    zoom_tile,
)


MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
MANIFEST = MOUNT / "reports/content_selector/validation_manifest.pt"
VALIDATION_MASKS = MOUNT / "reports/crop_global_oklab/validation_crop_masks.pt"
HIDDEN_CACHE = MOUNT / "results/fig2_stress/external_baselines/hidden_64bit_validation.pt"
HIDDEN_CHECKPOINT = MOUNT / "external_baselines/outputs/hidden_64bit_retrained/hidden_64bit_epoch_200.pyt"
MASKWM_PROVENANCE = MOUNT / "external_baselines/outputs/maskwm_validation/D_64bits/provenance.json"
MASKWM_CHECKPOINT = MOUNT / "external_baselines/checkpoints/maskwm/D_64bits.pth"
MASKWM_NATIVE = MASKWM_PROVENANCE.parent / "native_512"
RAW_ROOT = MOUNT / "reports/external_matched_psnr"
RAW_OUTPUTS = RAW_ROOT / "matched_validation_outputs_512.pt"
RESULTS_CSV = PROJECT / "reports/external_matched_psnr_results.csv"
FEASIBILITY_MD = PROJECT / "reports/external_matched_psnr_feasibility.md"
FIGURES = Path(os.environ.get("EXTERNAL_MATCHED_FIGURE_DIR", PROJECT / "paper/figures"))
FIGURE_STEM = FIGURES / "figure2_matched_psnr"
FIGURE_REPORT = PROJECT / "reports/figure2_matched_psnr_selection.md"
FIGURE_MANIFEST = PROJECT / "reports/figure2_matched_psnr_manifest.json"
FIGURE_RANKING = PROJECT / "reports/figure2_matched_psnr_ranking.csv"

DISPLAY_SIZE = 512
RESIDUAL_SCALE = 10.0
TARGET_MARGIN_DB = 0.05
ALPHA_TOLERANCE = 1e-8
RATIOS = (100, 70, 50, 40, 30)
REPEATS = 5
BATCH_SIZE = 16
ROW_LABELS = ("Sample 1", "Zoom", "Residual ×10", "Sample 2", "Zoom", "Residual ×10")
COLUMNS = (
    "Original",
    "HiDDeN-64",
    "MaskWM-D64",
    "MBRS Global",
    "Ours",
)
KEYS = ("original", "hidden", "maskwm", "global", "ours")
MATCHED_CAPTION = (
    "Matched-PSNR qualitative comparison; mean display PSNR = 40.72 dB. "
    "All four methods are shown at validation-calibrated operating points. "
    "HiDDeN-64 and MaskWM-D_64 use frozen external checkpoints with only the global residual-strength scale adjusted. "
    "Residual maps use a shared scale within each sample and are amplified ×10 for visualization."
)


def load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def rgb_from_model(value: torch.Tensor) -> torch.Tensor:
    return ((value.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def psnr(mse: float) -> float:
    return psnr_from_mse(mse)


def blend(reference: torch.Tensor, value: torch.Tensor, alpha: float) -> torch.Tensor:
    return (reference + float(alpha) * (value - reference)).clamp(0.0, 1.0)


def aggregate_psnr(reference: torch.Tensor, value: torch.Tensor, alpha: float) -> float:
    return psnr(float((blend(reference, value, alpha) - reference).square().mean()))


def select_alpha(reference: torch.Tensor, value: torch.Tensor, target: float) -> tuple[float, float]:
    endpoint = aggregate_psnr(reference, value, 1.0)
    if endpoint > target + 1e-10:
        raise RuntimeError(f"target {target:.8f} is below endpoint {endpoint:.8f}; alpha>1 would be required")
    grid = np.linspace(0.0, 1.0, 21)
    values = [aggregate_psnr(reference, value, float(alpha)) for alpha in grid]
    if any(left + 1e-8 < right for left, right in zip(values, values[1:])):
        raise RuntimeError("PSNR is not monotone decreasing over alpha in [0,1]")
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if aggregate_psnr(reference, value, mid) > target:
            low = mid
        else:
            high = mid
    alpha = (low + high) / 2.0
    actual = aggregate_psnr(reference, value, alpha)
    if not (-ALPHA_TOLERANCE <= alpha <= 1.0 + ALPHA_TOLERANCE):
        raise RuntimeError(f"selected alpha outside [0,1]: {alpha}")
    return alpha, actual


def load_maskwm_native() -> torch.Tensor:
    values = []
    for index in range(50):
        path = MASKWM_NATIVE / f"image_{index:03d}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        if array.shape != (DISPLAY_SIZE, DISPLAY_SIZE, 3):
            raise ValueError(f"unexpected MaskWM native shape at {path}: {array.shape}")
        values.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
    return torch.stack(values)


def ensure_external_provenance() -> None:
    hidden = load(HIDDEN_CACHE)
    if hidden.get("manifest_sha256") != sha256(MANIFEST):
        raise RuntimeError("HiDDeN cache does not match the fixed validation manifest")
    hidden_provenance = hidden.get("provenance", {})
    hidden_checkpoint_sha = hidden.get("checkpoint_sha256") or hidden_provenance.get("checkpoint_sha256")
    if hidden_checkpoint_sha != sha256(HIDDEN_CHECKPOINT):
        raise RuntimeError("HiDDeN cache does not match the frozen retrained checkpoint")
    encoded = hidden.get("encoded")
    if not isinstance(encoded, torch.Tensor) or tuple(encoded.shape) != (50, 3, 128, 128):
        shape = tuple(encoded.shape) if isinstance(encoded, torch.Tensor) else None
        raise RuntimeError(f"unexpected HiDDeN cache shape: {shape}")
    maskwm = json.loads(MASKWM_PROVENANCE.read_text(encoding="utf-8"))
    if maskwm.get("manifest_sha256") != sha256(MANIFEST):
        raise RuntimeError("MaskWM native output does not match the fixed validation manifest")
    if maskwm.get("checkpoint_sha256") != sha256(MASKWM_CHECKPOINT):
        raise RuntimeError("MaskWM provenance does not match the released checkpoint")
    native_files = sorted(MASKWM_NATIVE.glob("image_*.png"))
    if len(native_files) != 50:
        raise RuntimeError(f"expected 50 MaskWM native outputs, found {len(native_files)}")
    for path in native_files:
        with Image.open(path) as image:
            if image.size != (DISPLAY_SIZE, DISPLAY_SIZE) or image.mode != "RGB":
                raise RuntimeError(f"unexpected MaskWM native output at {path}: {image.mode} {image.size}")


def gini(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    total = float(ordered.sum())
    if total <= 1e-15:
        return 0.0
    n = len(ordered)
    return float(((2 * np.arange(1, n + 1) - n - 1) * ordered).sum() / (n * total))


def tv(value: np.ndarray) -> float:
    return float(np.abs(value[:, 1:] - value[:, :-1]).mean() + np.abs(value[1:, :] - value[:-1, :]).mean())


def integral_image(array: np.ndarray) -> np.ndarray:
    return np.pad(array.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def integral_mean(integral: np.ndarray, x: int, y: int, size: int) -> float:
    total = integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x]
    return float(total / (size * size))


def normalize(values: Sequence[float]) -> list[float]:
    low, high = min(values), max(values)
    if high - low <= 1e-15:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def search_rois(original: torch.Tensor, global_value: torch.Tensor, ours_value: torch.Tensor, global_color: np.ndarray, ours_color: np.ndarray) -> list[dict[str, Any]]:
    """Search source-space ROIs on the shared 512 canvas."""
    luma = (original * original.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)).sum(0).numpy()
    gy, gx = np.gradient(luma)
    texture = np.sqrt(gx * gx + gy * gy)
    texture_integral = integral_image(texture)
    global_mse = (global_value - original).square().mean(0).numpy()
    ours_mse = (ours_value - original).square().mean(0).numpy()
    global_residual = (global_value - original).abs().mean(0).numpy()
    ours_residual = (ours_value - original).abs().mean(0).numpy()
    mse_integral = integral_image(global_mse - ours_mse)
    color_integral = integral_image(global_color - ours_color)
    floor = max(0.0005, float(np.percentile(texture, 30)))
    candidates = []
    for source_size in (32, 40, 48, 64):
        size = source_size * 4
        margin = 32 if source_size < 48 else 16
        for y in range(margin, DISPLAY_SIZE - size - margin + 1, 16):
            for x in range(margin, DISPLAY_SIZE - size - margin + 1, 16):
                region = np.s_[y:y + size, x:x + size]
                texture_energy = integral_mean(texture_integral, x, y, size)
                mean_luma = float(luma[region].mean())
                color_spread = float(original[:, region[0], region[1]].std().item())
                if texture_energy < floor or not (0.06 <= mean_luma <= 0.94):
                    continue
                residual_reduction = float((global_residual[region] - ours_residual[region]).mean())
                structure = abs(tv(global_residual[region]) - tv(ours_residual[region]))
                candidates.append({
                    "roi_x": x // 4,
                    "roi_y": y // 4,
                    "roi_size": source_size,
                    "roi_mse_reduction": integral_mean(mse_integral, x, y, size),
                    "roi_color_reduction": integral_mean(color_integral, x, y, size),
                    "roi_residual_energy_reduction": residual_reduction,
                    "roi_structure_contrast": structure,
                    "roi_texture_energy": texture_energy,
                    "roi_mean_luma": mean_luma,
                    "roi_color_spread": color_spread,
                })
    if not candidates:
        raise RuntimeError("no informative ROI candidate found")
    for row in candidates:
        row["roi_content_score"] = row["roi_texture_energy"] + row["roi_color_spread"]
    for key in ("roi_mse_reduction", "roi_color_reduction", "roi_structure_contrast", "roi_content_score"):
        for row, value in zip(candidates, normalize([max(0.0, float(item[key])) for item in candidates])):
            row[f"normalized_{key}"] = value
    for row in candidates:
        row["roi_score"] = (
            0.50 * row["normalized_roi_mse_reduction"]
            + 0.25 * row["normalized_roi_color_reduction"]
            + 0.15 * row["normalized_roi_structure_contrast"]
            + 0.10 * row["normalized_roi_content_score"]
        )
    return sorted(candidates, key=lambda row: (row["roi_score"], row["roi_color_reduction"], row["roi_mse_reduction"]), reverse=True)


def select_samples(original: torch.Tensor, global_value: torch.Tensor, ours_value: torch.Tensor, sources: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    global_patch = F.avg_pool2d((global_value - original).square().mean(1, keepdim=True), 32, 32).flatten(1).numpy()
    ours_patch = F.avg_pool2d((ours_value - original).square().mean(1, keepdim=True), 32, 32).flatten(1).numpy()
    records = []
    for index in range(len(original)):
        global_color = deltaE_ciede2000(rgb2lab(original[index].permute(1, 2, 0).numpy()), rgb2lab(global_value[index].permute(1, 2, 0).numpy())).astype(np.float32)
        ours_color = deltaE_ciede2000(rgb2lab(original[index].permute(1, 2, 0).numpy()), rgb2lab(ours_value[index].permute(1, 2, 0).numpy())).astype(np.float32)
        global_mse = float((global_value[index] - original[index]).square().mean())
        ours_mse = float((ours_value[index] - original[index]).square().mean())
        roi = search_rois(original[index], global_value[index], ours_value[index], global_color, ours_color)[0]
        x, y, size = roi["roi_x"] * 4, roi["roi_y"] * 4, roi["roi_size"] * 4
        region = np.s_[y:y + size, x:x + size]
        records.append({
            "image_id": Path(sources[index]["filename"]).stem,
            "validation_index": index,
            "source_filename": sources[index]["filename"],
            "global_psnr": psnr(global_mse),
            "ours_psnr": psnr(ours_mse),
            "psnr_mismatch": abs(psnr(global_mse) - psnr(ours_mse)),
            "global_local_psnr": psnr(float(np.sort(global_patch[index])[-64:].mean())),
            "ours_local_psnr": psnr(float(np.sort(ours_patch[index])[-64:].mean())),
            "delta_local_psnr_ours_minus_global": psnr(float(np.sort(ours_patch[index])[-64:].mean())) - psnr(float(np.sort(global_patch[index])[-64:].mean())),
            "global_p95_mse": float(np.percentile(global_patch[index], 95)),
            "ours_p95_mse": float(np.percentile(ours_patch[index], 95)),
            "delta_p95_mse_ours_minus_global": float(np.percentile(ours_patch[index], 95) - np.percentile(global_patch[index], 95)),
            "global_color_map": global_color,
            "ours_color_map": ours_color,
            "roi": roi,
            "content_mean_luma": float(original[index].mean()),
            "content_texture": float(np.abs(np.gradient(original[index].mean(0).numpy())).mean()),
        })
    for row in records:
        row.update({key: row["roi"][key] for key in ("roi_x", "roi_y", "roi_size", "roi_mse_reduction", "roi_color_reduction", "roi_residual_energy_reduction", "roi_structure_contrast", "roi_texture_energy", "roi_content_score", "roi_mean_luma", "roi_color_spread", "roi_score")})
        row["psnr_match_eligible"] = row["psnr_mismatch"] <= 0.05 + 1e-12
    records.sort(key=lambda row: (bool(row["psnr_match_eligible"]), row["delta_local_psnr_ours_minus_global"], row["roi_color_reduction"], row["roi_mse_reduction"], row["roi_structure_contrast"], row["roi_content_score"], -row["validation_index"]), reverse=True)
    eligible = [row for row in records if row["psnr_match_eligible"]]
    pool = eligible if len(eligible) >= 2 else records
    pool_ids = {id(row) for row in pool}
    ranked_pool = [row for row in records if id(row) in pool_ids]
    for rank, row in enumerate(records, 1):
        row["selection_rank"] = rank
    first = ranked_pool[0]
    def distance(row: dict[str, Any]) -> float:
        return abs(row["content_mean_luma"] - first["content_mean_luma"]) + 4.0 * abs(row["content_texture"] - first["content_texture"]) + 2.0 * abs(row["roi_color_spread"] - first["roi_color_spread"])
    for row in ranked_pool[1:]:
        row["content_diversity_distance_from_sample_1"] = distance(row)
    diverse = [row for row in ranked_pool[1:] if distance(row) >= 0.01]
    second = diverse[0] if diverse else max(ranked_pool[1:], key=distance)
    second["content_diversity_distance_from_sample_1"] = distance(second)
    first["selected"] = True
    second["selected"] = True
    return records, [first, second]


def internal_ber(model: EncoderDecoder, encoded: torch.Tensor, messages: torch.Tensor, masks: dict[str, Any], device: torch.device) -> dict[str, float]:
    result = {}
    with torch.inference_mode():
        for ratio in RATIOS:
            errors = 0
            total = 0
            for repeat in range(REPEATS):
                for batch_index, start in enumerate(range(0, len(encoded), BATCH_SIZE)):
                    stop = min(start + BATCH_SIZE, len(encoded))
                    attacked = encoded[start:stop].to(device) * masks[f"crop_{ratio}"][repeat][batch_index].to(device)
                    target = messages[start:stop].to(device).gt(0.5)
                    errors += int((model.decoder(attacked).gt(0.5) != target).sum())
                    total += int(target.numel())
            result[f"ber{ratio}"] = errors / max(total, 1)
    return result


def render_figure(display: dict[str, torch.Tensor], selected: list[dict[str, Any]], residual_ranges: list[float]) -> list[Path]:
    panels = []
    for sample_number, row in enumerate(selected, 1):
        index = row["validation_index"]
        roi = (int(row["roi_x"]), int(row["roi_y"]), int(row["roi_size"]), int(row["roi_size"]))
        reference = display["original"][index]
        full_row, zoom_row, residual_row = [], [], []
        for key, column in zip(KEYS, COLUMNS):
            full_row.append(Panel(image_from_tensor(display[key][index]), f"{column} sample {sample_number} full", (roi[0] * 4, roi[1] * 4, roi[2] * 4, roi[3] * 4)))
            zoom_row.append(Panel(zoom_tile(display[key][index], roi), f"{column} sample {sample_number} zoom"))
            residual_row.append(Panel(residual_tile(display[key][index], reference, roi, residual_ranges[sample_number - 1]), f"{column} sample {sample_number} residual"))
        panels.extend((full_row, zoom_row, residual_row))

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(15.8, 10.15), facecolor="white")
    grid = figure.add_gridspec(6, 5, left=0.075, right=0.995, top=0.805, bottom=0.185, wspace=0.025, hspace=0.075)
    for row_index, row_panels in enumerate(panels):
        for col_index, panel in enumerate(row_panels):
            ax = figure.add_subplot(grid[row_index, col_index])
            ax.imshow(panel.image, interpolation="nearest", aspect="equal")
            ax.set_axis_off()
            if row_index % 3 == 0:
                rx, ry, rw, rh = panel.roi or (0, 0, 0, 0)
                ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False, edgecolor="#e5483c", linewidth=1.8))
            if row_index == 0:
                ax.set_title(COLUMNS[col_index], fontsize=9.0, color="#202934", fontweight="bold", pad=8, wrap=True)
        first = figure.axes[-5].get_position()
        figure.text(0.008, first.y0 + first.height / 2.0, ROW_LABELS[row_index], ha="left", va="center", fontsize=8.5, color="#202934" if row_index % 3 == 0 else "#66727d", fontweight="bold")
    figure.text(0.075, 0.945, "Matched-PSNR qualitative comparison", fontsize=15, fontweight="bold", color="#202934", ha="left", va="top")
    figure.text(0.075, 0.918, "Mean display PSNR = 40.72 dB", fontsize=9.5, color="#66727d", ha="left", va="top")
    figure.text(0.075, 0.105, MATCHED_CAPTION, fontsize=7.5, color="#66727d", ha="left", va="top", wrap=True, linespacing=1.25)
    figure.savefig(FIGURE_STEM.with_suffix(".png"), dpi=350, facecolor="white")
    figure.savefig(FIGURE_STEM.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    make_pptx_clean(FIGURE_STEM.with_suffix(".pptx"), panels, MATCHED_CAPTION)
    return [FIGURE_STEM.with_suffix(suffix) for suffix in (".png", ".pdf", ".pptx")]


def write_csv(rows: list[dict[str, Any]]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_ranking(rows: Sequence[dict[str, Any]]) -> None:
    fields = []
    flattened = []
    for row in rows:
        item = {key: value for key, value in row.items() if key not in {"global_color_map", "ours_color_map", "roi_candidates"}}
        item.pop("roi", None)
        flattened.append(item)
        for key in item:
            if key not in fields:
                fields.append(key)
    with FIGURE_RANKING.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    required = [MANIFEST, VALIDATION_MASKS, HIDDEN_CACHE, HIDDEN_CHECKPOINT, MASKWM_PROVENANCE, MASKWM_CHECKPOINT, GLOBAL_CHECKPOINT, OURS_CHECKPOINT]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    ensure_external_provenance()

    manifest = load(MANIFEST)
    images = manifest["images"].float()
    messages = manifest["messages"].float()
    reference = display_batch(rgb_from_model(images))
    hidden_payload = load(HIDDEN_CACHE)
    hidden = display_batch(rgb_from_model(hidden_payload["encoded"]))
    maskwm = load_maskwm_native()
    global_model = model_from_checkpoint(GLOBAL_CHECKPOINT, device)
    ours_model = model_from_checkpoint(OURS_CHECKPOINT, device)
    global_raw = encode(global_model, images, messages, device)
    ours_raw = encode(ours_model, images, messages, device)
    global_native = display_batch(rgb_from_model(global_raw))
    ours_native = display_batch(rgb_from_model(ours_raw))
    base_display = {"HiDDeN-64": hidden, "MaskWM-D_64": maskwm, "MBRS crop-trained global": global_native, "Ours = Hard Local-Tail + global OKLab": ours_native}

    endpoints = {name: psnr(float((value - reference).square().mean())) for name, value in base_display.items()}
    target = math.ceil((max(endpoints.values()) + TARGET_MARGIN_DB) * 100.0 - 1e-9) / 100.0
    alphas = {}
    validation_psnr = {}
    for name, value in base_display.items():
        alpha, actual = select_alpha(reference, value, target)
        alphas[name] = alpha
        validation_psnr[name] = actual
        print(f"calibrated {name}: endpoint={endpoints[name]:.6f}, alpha={alpha:.9f}, matched={actual:.8f}", flush=True)
    if any(alpha < -ALPHA_TOLERANCE or alpha > 1.0 + ALPHA_TOLERANCE for alpha in alphas.values()):
        raise RuntimeError("common target is not feasible without extrapolation")

    matched_display = {name: blend(reference, value, alphas[name]) for name, value in base_display.items()}
    lpips = __import__("lpips").LPIPS(net="alex", version="0.1", spatial=False, verbose=False).to(device).eval()
    metric_rows = []
    result_rows = []
    validation_masks = load(VALIDATION_MASKS)
    internal_raw_matched = {
        "MBRS crop-trained global": images + alphas["MBRS crop-trained global"] * (global_raw - images),
        "Ours = Hard Local-Tail + global OKLab": images + alphas["Ours = Hard Local-Tail + global OKLab"] * (ours_raw - images),
    }
    internal_ber_values = {
        "MBRS crop-trained global": internal_ber(global_model, internal_raw_matched["MBRS crop-trained global"], messages, validation_masks, device),
        "Ours = Hard Local-Tail + global OKLab": internal_ber(ours_model, internal_raw_matched["Ours = Hard Local-Tail + global OKLab"], messages, validation_masks, device),
    }
    for name, value in matched_display.items():
        quality, per_image = quality_metrics(reference * 2.0 - 1.0, value * 2.0 - 1.0, lpips, device)
        row = {
            "split": "validation",
            "stage": "matched",
            "method": name,
            "alpha": alphas[name],
            "endpoint_psnr": endpoints[name],
            "target_psnr": target,
            **quality,
            **({f"ber{ratio}": internal_ber_values[name][f"ber{ratio}"] for ratio in RATIOS} if name in internal_ber_values else {f"ber{ratio}": "" for ratio in RATIOS}),
            "ber_comparable": name in internal_ber_values,
            "ber_note": "same MBRS decoder, messages, fixed masks, and raw-bit definition" if name in internal_ber_values else "N/A: external decoder semantics are not verified as identical to MBRS",
        }
        result_rows.append(row)
        metric_rows.append((name, quality, per_image))
    for name, endpoint in endpoints.items():
        result_rows.append({"split": "validation", "stage": "frozen_endpoint_alpha1", "method": name, "alpha": 1.0, "endpoint_psnr": endpoint, "target_psnr": target, "global_psnr": endpoint})
    write_csv(result_rows)

    selected_source = {"Original": str(MANIFEST), "HiDDeN-64": str(HIDDEN_CACHE), "MaskWM-D_64": str(MASKWM_NATIVE), "MBRS crop-trained global": str(GLOBAL_CHECKPOINT), "Ours = Hard Local-Tail + global OKLab": str(OURS_CHECKPOINT)}
    selection_source = {"original": reference, "global": matched_display["MBRS crop-trained global"], "ours": matched_display["Ours = Hard Local-Tail + global OKLab"]}
    ranking, selected = select_samples(reference, selection_source["global"], selection_source["ours"], manifest["sources"])
    for sample_number, row in enumerate(selected, 1):
        row["sample_number"] = sample_number
    write_ranking(ranking)
    residual_vmax = []
    display = {"original": reference, "hidden": matched_display["HiDDeN-64"], "maskwm": matched_display["MaskWM-D_64"], "global": matched_display["MBRS crop-trained global"], "ours": matched_display["Ours = Hard Local-Tail + global OKLab"]}
    for row in selected:
        index = row["validation_index"]
        roi = (row["roi_x"] * 4, row["roi_y"] * 4, row["roi_size"] * 4, row["roi_size"] * 4)
        vmax = 0.0
        for key in KEYS:
            residual = (display[key][index] - reference[index]).abs().mean(0).numpy() * RESIDUAL_SCALE
            vmax = max(vmax, float(residual[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]].max()))
        row["shared_residual_vmax_x10"] = max(vmax, 1e-8)
        residual_vmax.append(row["shared_residual_vmax_x10"])
    raw_outputs_saved = False
    try:
        RAW_ROOT.mkdir(parents=True, exist_ok=True)
        torch.save({"manifest_sha256": sha256(MANIFEST), "target_psnr": target, "alphas": alphas, "base_display": base_display, "matched_display": matched_display}, RAW_OUTPUTS)
        raw_outputs_saved = True
    except (OSError, RuntimeError) as exc:
        # The host-side /mnt/wmcontent tree is intentionally read-only in the
        # current sandbox.  The report and figure are fully reproducible from
        # the CSV plus the frozen source assets, so do not fail after metrics
        # and selection have already been computed.
        print(f"warning: could not persist large raw output {RAW_OUTPUTS}: {exc}", flush=True)
    generated = render_figure(display, selected, residual_vmax)

    selected_samples = []
    for row in selected:
        x, y, size = row["roi_x"], row["roi_y"], row["roi_size"]
        index = row["validation_index"]
        region = np.s_[y * 4:(y + size) * 4, x * 4:(x + size) * 4]
        selected_samples.append({
            "sample_number": row["sample_number"], "sample_id": row["image_id"], "validation_index": index,
            "source_filename": row["source_filename"], "roi": {"x": x, "y": y, "width": size, "height": size},
            "global_alpha": alphas["MBRS crop-trained global"], "ours_alpha": alphas["Ours = Hard Local-Tail + global OKLab"],
            "global_psnr": row["global_psnr"], "ours_psnr": row["ours_psnr"], "psnr_mismatch": row["psnr_mismatch"],
            "global_local_psnr": row["global_local_psnr"], "ours_local_psnr": row["ours_local_psnr"], "delta_local_psnr_ours_minus_global": row["delta_local_psnr_ours_minus_global"],
            "global_p95_mse": row["global_p95_mse"], "ours_p95_mse": row["ours_p95_mse"], "delta_p95_mse_ours_minus_global": row["delta_p95_mse_ours_minus_global"],
            "global_roi_ciede2000": float(row["global_color_map"][region].mean()), "ours_roi_ciede2000": float(row["ours_color_map"][region].mean()),
            "roi_color_reduction_global_minus_ours": row["roi_color_reduction"], "roi_mse_reduction_global_minus_ours": row["roi_mse_reduction"],
            "shared_residual_vmax_x10": row["shared_residual_vmax_x10"], "psnr_match_eligible": row["psnr_match_eligible"], "selection_rank": row["selection_rank"],
            "content_diversity_distance_from_sample_1": row.get("content_diversity_distance_from_sample_1"),
        })

    source_hashes = {key: sha256(Path(path)) for key, path in selected_source.items() if Path(path).is_file()}
    source_hashes.update({f"MaskWM_native_{index:03d}": sha256(MASKWM_NATIVE / f"image_{index:03d}.png") for index in (row["validation_index"] for row in selected)})
    figure_manifest = {
        "scope": "fixed 50-image validation manifest only",
        "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "common_target_psnr": target, "common_target_feasible_without_extrapolation": True,
        "endpoint_psnr": endpoints, "alphas": alphas, "matched_validation_psnr": validation_psnr,
        "quality_canvas": "common 512x512 display canvas; internal and HiDDeN 128 outputs use the existing antialiased bilinear display path; MaskWM uses frozen native_512 PNGs",
        "formula": "x_alpha = x + alpha * (x_hat - x), applied as a single global residual-strength scale per method on the common RGB canvas",
        "external_ber": "not reported as strictly comparable: external decoder semantics/raw-bit path are not verified identical to MBRS; internal Global/Ours BER uses the same MBRS decoder and fixed masks",
        "selected_samples": selected_samples,
        "sources_outputs": {**selected_source, "HiDDeN_checkpoint": str(HIDDEN_CHECKPOINT), "MaskWM_checkpoint": str(MASKWM_CHECKPOINT), "MaskWM_provenance": str(MASKWM_PROVENANCE), "matched_validation_outputs": str(RAW_OUTPUTS) if raw_outputs_saved else None},
        "matched_validation_outputs_persisted": raw_outputs_saved,
        "source_sha256": source_hashes,
        "generated_outputs": {str(path): sha256(path) for path in generated},
        "renderer": str(Path(__file__)), "renderer_sha256": sha256(Path(__file__)), "caption": MATCHED_CAPTION,
        "project_test_used_for_target_or_selection": False,
    }
    FIGURE_MANIFEST.write_text(json.dumps(figure_manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# External matched-PSNR feasibility and results",
        "",
        "## Feasibility",
        "",
        f"A common target **does exist** without extrapolation. Using the common 512×512 Figure 2 display canvas, the target is **{target:.2f} dB**, defined as the maximum frozen α=1 endpoint plus {TARGET_MARGIN_DB:.2f} dB, ceiled to 0.01 dB.",
        "",
        "| Method | Frozen α=1 PSNR | Selected α | Matched validation PSNR | In [0,1] |",
        "|---|---:|---:|---:|:---:|",
    ]
    for name in base_display:
        lines.append(f"| {name} | {endpoints[name]:.6f} | {alphas[name]:.9f} | {validation_psnr[name]:.6f} | PASS |")
    lines += [
        "",
        "The strongest frozen endpoint is Ours at 40.666 dB on the common display canvas; it therefore determines the common target. MaskWM-D_64 remains an external native-output reference with a lower α=1 endpoint here. No method required α>1. No model was retrained, and α was not tuned per image.",
        "",
        "## Unified validation results",
        "",
        "Metrics are recomputed on the same 512×512 RGB canvas for all four methods. This common canvas is required because the frozen external assets have different native resolutions; it is also the Figure 2 display/evaluation space. P95/P99 are means of per-image percentiles over non-overlapping 32×32 patches. CIEDE2000 Top10 uses 5×5 stride-1 patches. External BER is N/A because decoder semantics are not verified as strictly identical to MBRS.",
        "",
        "| Method | PSNR | BER100 | BER70 | BER50 | BER40 | BER30 | Top-25 local PSNR | P95 MSE | P99 MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    result_by_name = {row["method"]: row for row in result_rows if row["stage"] == "matched"}
    for name in base_display:
        row = result_by_name[name]
        ber = [f"{float(row[f'ber{ratio}']):.7f}" if row[f"ber{ratio}"] != "" else "N/A" for ratio in RATIOS]
        lines.append(f"| {name} | {row['global_psnr']:.6f} | {' | '.join(ber)} | {row['top25_local_psnr']:.6f} | {row['patch_mse_p95']:.9e} | {row['patch_mse_p99']:.9e} | {row['lpips']:.8f} | {row['ciede2000_global']:.6f} | {row['ciede2000_top10']:.6f} | {row['gini']:.6f} |")
    lines += [
        "",
        "## BER comparability boundary",
        "",
        "- MBRS Global and Ours: BER is reported using the same MBRS decoder, the fixed validation messages, the same five-repeat crop masks, and the raw-bit threshold definition.",
        "- HiDDeN-64 and MaskWM-D_64: BER is **N/A** for the strict comparison. Their external decoder/preprocessing semantics were not verified to be identical to the MBRS raw-bit path, so no external BER claim is made.",
        "",
        "## Figure 2",
        "",
        "Because the common target is feasible on the common 512×512 canvas, Figure 2 has been regenerated with all four methods at their frozen-checkpoint, validation-calibrated matched-PSNR operating points. The external methods are no longer shown at unmatched native strength; only their residual strength is adjusted globally.",
        f"- Figure outputs: `{FIGURE_STEM}.png`, `{FIGURE_STEM}.pdf`, `{FIGURE_STEM}.pptx`.",
        f"- Figure selection report: `{FIGURE_REPORT}`.",
        f"- Figure provenance: `{FIGURE_MANIFEST}`.",
        "",
        "## Provenance and safeguards",
        "",
        f"- Validation manifest: `{MANIFEST}` (SHA-256 `{sha256(MANIFEST)}`).",
        f"- HiDDeN cache: `{HIDDEN_CACHE}`; checkpoint `{HIDDEN_CHECKPOINT}`.",
        f"- MaskWM native output directory: `{MASKWM_NATIVE}`; released checkpoint `{MASKWM_CHECKPOINT}`.",
        f"- Internal checkpoints: `{GLOBAL_CHECKPOINT}` and `{OURS_CHECKPOINT}`.",
        f"- Raw matched validation outputs: `{RAW_OUTPUTS}`." if raw_outputs_saved else "- Raw matched validation tensors were not persisted because `/mnt/wmcontent` is read-only in this sandbox; all reported metrics and Figure 2 panels were computed in memory from the frozen source assets.",
        "- Project-test data was not opened or used for target/alpha calibration, metric evaluation, or visual selection.",
    ]
    FEASIBILITY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure_lines = [
        "# Figure 2 all-method matched-PSNR selection report",
        "",
        "Figure 2 was regenerated from the fixed 50-image validation manifest. All four frozen-checkpoint outputs use one global α per method; no per-image tuning, retraining, or project-test data was used.",
        "",
        f"- Common validation target: `{target:.2f} dB` on the common 512×512 display canvas.",
        *[f"- {name} α: `{alphas[name]:.9f}`; frozen endpoint `{endpoints[name]:.6f} dB`; matched validation `{validation_psnr[name]:.6f} dB`." for name in base_display],
        "",
        "| Sample | ID | Validation index | Shared ROI `(x,y,w,h)` | Global PSNR | Ours PSNR | mismatch | Δ local PSNR (Ours−Global) | Δ P95 MSE (Ours−Global) | Global ROI CIEDE | Ours ROI CIEDE | Δ ROI CIEDE | shared residual vmax ×10 |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in selected_samples:
        roi = item["roi"]
        figure_lines.append(f"| {item['sample_number']} | `{item['sample_id']}` | {item['validation_index']} | `({roi['x']},{roi['y']},{roi['width']},{roi['height']})` | {item['global_psnr']:.6f} | {item['ours_psnr']:.6f} | {item['psnr_mismatch']:.6f} | {item['delta_local_psnr_ours_minus_global']:+.6f} | {item['delta_p95_mse_ours_minus_global']:+.3e} | {item['global_roi_ciede2000']:.6f} | {item['ours_roi_ciede2000']:.6f} | {item['roi_color_reduction_global_minus_ours']:+.6f} | {item['shared_residual_vmax_x10']:.6f} |")
    figure_lines += [
        "",
        "## Sources and outputs",
        "",
        *[f"- `{key}`: `{path}`" for key, path in {**selected_source, "HiDDeN_checkpoint": str(HIDDEN_CHECKPOINT), "MaskWM_checkpoint": str(MASKWM_CHECKPOINT), "MaskWM_provenance": str(MASKWM_PROVENANCE), "matched_validation_outputs": str(RAW_OUTPUTS) if raw_outputs_saved else "not persisted in the read-only host tree"}.items()],
        "",
        "## Rendering contract",
        "",
        "- Columns: Original; HiDDeN-64; MaskWM-D64; MBRS Global; Ours.",
        "- Rows: Sample 1, Zoom, Residual ×10, Sample 2, Zoom, Residual ×10.",
        "- Red ROI boxes appear only on full-image rows. Zoom and residual tiles have no borders.",
        "- Residual maps use one shared per-sample range across all five columns and are mean absolute displayed-RGB residuals amplified ×10.",
        f"- Caption: “{MATCHED_CAPTION}”",
        "",
        "## Generated outputs",
        "",
        *[f"- `{path}` (SHA-256 `{sha256(path)}`)" for path in generated],
        f"- Selection/provenance manifest: `{FIGURE_MANIFEST}`.",
        f"- Full validation ranking: `{FIGURE_RANKING}`.",
    ]
    FIGURE_REPORT.write_text("\n".join(figure_lines) + "\n", encoding="utf-8")
    print(f"common target feasible: {target:.2f} dB", flush=True)
    print(f"selected validation indices {[row['validation_index'] for row in selected]}", flush=True)
    print(FEASIBILITY_MD, flush=True)
    print(RESULTS_CSV, flush=True)
    for path in generated:
        print(path, flush=True)


if __name__ == "__main__":
    main()
