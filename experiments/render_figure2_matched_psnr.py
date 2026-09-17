#!/usr/bin/env python3
"""Render Figure 2 from validation-calibrated matched-PSNR outputs.

Only the fixed 50-image validation manifest is opened. The two internal
frozen checkpoints are re-encoded, the supplied global residual scales are
applied once per method, and two content-diverse samples are selected from
those matched outputs. HiDDeN and MaskWM are loaded from their existing frozen
validation assets without rescaling their residual strength.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile
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

from experiments.losses import patch_mse_per_sample
from network.Encoder_MP_Decoder import EncoderDecoder
from paper import render_fig2_qualitative_editable as ooxml


MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
MANIFEST = MOUNT / "reports/content_selector/validation_manifest.pt"
MATCHED_CSV = PROJECT / "reports/matched_psnr_alpha_residual.csv"
HIDDEN_CACHE = MOUNT / "results/fig2_stress/external_baselines/hidden_64bit_validation.pt"
HIDDEN_CHECKPOINT = MOUNT / "external_baselines/outputs/hidden_64bit_retrained/hidden_64bit_epoch_200.pyt"
MASKWM_PROVENANCE = MOUNT / "external_baselines/outputs/maskwm_validation/D_64bits/provenance.json"
MASKWM_CHECKPOINT = MOUNT / "external_baselines/checkpoints/maskwm/D_64bits.pth"
MASKWM_NATIVE = MASKWM_PROVENANCE.parent / "native_512"
GLOBAL_CHECKPOINT = MOUNT / "experiments/runs/controlled_seed17_global_continuation/checkpoint_0020.pth"
OURS_CHECKPOINT = MOUNT / "experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/checkpoint_0020.pth"
RAW_ROOT = MOUNT / "reports/figure2_matched_psnr"
MATCHED_OUTPUTS = RAW_ROOT / "matched_internal_validation_outputs.pt"
OUTPUT_DIR = PROJECT / "paper/figures"
OUTPUT_STEM = OUTPUT_DIR / "figure2_matched_psnr"
SELECTION_REPORT = PROJECT / "reports/figure2_matched_psnr_selection.md"
SELECTION_MANIFEST = PROJECT / "reports/figure2_matched_psnr_manifest.json"
RANKING_CSV = PROJECT / "reports/figure2_matched_psnr_ranking.csv"

GLOBAL_ALPHA = 0.927892238
OURS_ALPHA = 0.994025141
DISPLAY_SIZE = 512
RESIDUAL_SCALE = 10.0
RED = "#e5483c"
INK = "#202934"
MUTED = "#66727d"
PAD = (248, 249, 251)

COLUMNS = (
    "Original",
    "HiDDeN-64",
    "MaskWM-D_64",
    "MBRS crop-trained global",
    "Ours = Hard Local-Tail + global OKLab",
)
KEYS = ("original", "hidden", "maskwm", "global", "ours")
INTERNAL_KEYS = ("global", "ours")
ROW_LABELS = ("Sample 1", "Zoom", "Residual ×10", "Sample 2", "Zoom", "Residual ×10")


def load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_from_model(value: torch.Tensor) -> torch.Tensor:
    return ((value.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def display_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or value.shape[1:] != (3, 128, 128):
        raise ValueError(f"expected NCHW 128 RGB tensor, got {tuple(value.shape)}")
    return F.interpolate(value.float().clamp(0.0, 1.0), size=(DISPLAY_SIZE, DISPLAY_SIZE), mode="bilinear", align_corners=False, antialias=True)


def psnr(mse: float) -> float:
    return float(10.0 * math.log10(1.0 / max(float(mse), 1e-12)))


def model_from_checkpoint(path: Path, device: torch.device) -> EncoderDecoder:
    checkpoint = load(path, device)
    config = checkpoint["config"]
    model = EncoderDecoder(config["H"], config["W"], config["message_length"], ["Identity()"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def encode(model: EncoderDecoder, images: torch.Tensor, messages: torch.Tensor, device: torch.device) -> torch.Tensor:
    values = []
    with torch.inference_mode():
        for start in range(0, len(images), 16):
            values.append(model.encoder(images[start:start + 16].to(device), messages[start:start + 16].to(device)).cpu())
    return torch.cat(values)


def matched_output(images: torch.Tensor, encoded: torch.Tensor, alpha: float) -> torch.Tensor:
    return ((images + alpha * (encoded - images) + 1.0) / 2.0).clamp(0.0, 1.0)


def ciede_map(reference: torch.Tensor, value: torch.Tensor) -> np.ndarray:
    left = reference.permute(1, 2, 0).numpy()
    right = value.permute(1, 2, 0).numpy()
    return deltaE_ciede2000(rgb2lab(left), rgb2lab(right)).astype(np.float32)


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


def roi_candidates(
    original: torch.Tensor,
    global_value: torch.Tensor,
    ours_value: torch.Tensor,
    global_color: np.ndarray,
    ours_color: np.ndarray,
) -> list[dict[str, Any]]:
    """Search shared ROIs with content eligibility and metric-first scoring."""
    luminance = (original * original.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)).sum(0).numpy()
    gy, gx = np.gradient(luminance)
    texture = np.sqrt(gx * gx + gy * gy)
    texture_integral = integral_image(texture)
    mse_global = (global_value - original).square().mean(0).numpy()
    mse_ours = (ours_value - original).square().mean(0).numpy()
    residual_global = (global_value - original).abs().mean(0).numpy()
    residual_ours = (ours_value - original).abs().mean(0).numpy()
    mse_gain_integral = integral_image(mse_global - mse_ours)
    color_gain_integral = integral_image(global_color - ours_color)
    texture_floor = max(0.002, float(np.percentile(texture, 30)))
    candidates = []
    for size in (32, 40, 48, 64):
        margin = 8 if size < 48 else 4
        for y in range(margin, 128 - size - margin + 1, 4):
            for x in range(margin, 128 - size - margin + 1, 4):
                texture_energy = integral_mean(texture_integral, x, y, size)
                region = np.s_[y:y + size, x:x + size]
                mean_luma = float(luminance[region].mean())
                color_spread = float(original[:, region[0], region[1]].std().item())
                # Exclude nearly black/empty ROIs while preserving low-texture
                # but chromatic content. The fallback below handles rare flat images.
                if texture_energy < texture_floor or not (0.06 <= mean_luma <= 0.94):
                    continue
                residual_energy_reduction = float((residual_global[region] - residual_ours[region]).mean())
                structure_contrast = abs(tv(residual_global[region]) - tv(residual_ours[region]))
                candidates.append({
                    "roi_x": x, "roi_y": y, "roi_size": size,
                    "roi_mse_reduction": integral_mean(mse_gain_integral, x, y, size),
                    "roi_color_reduction": integral_mean(color_gain_integral, x, y, size),
                    "roi_residual_energy_reduction": residual_energy_reduction,
                    "roi_structure_contrast": structure_contrast,
                    "roi_texture_energy": texture_energy,
                    "roi_mean_luma": mean_luma,
                    "roi_color_spread": color_spread,
                })
    if not candidates:
        raise RuntimeError("no informative validation ROI candidates found")
    for row in candidates:
        row["roi_content_score"] = row["roi_texture_energy"] + row["roi_color_spread"]
    for key in ("roi_mse_reduction", "roi_color_reduction", "roi_structure_contrast", "roi_content_score"):
        values = normalize([max(0.0, float(row[key])) for row in candidates])
        for row, value in zip(candidates, values):
            row[f"normalized_{key}"] = value
    for row in candidates:
        row["roi_score"] = (
            0.50 * row["normalized_roi_mse_reduction"]
            + 0.25 * row["normalized_roi_color_reduction"]
            + 0.15 * row["normalized_roi_structure_contrast"]
            + 0.10 * row["normalized_roi_content_score"]
        )
    return sorted(candidates, key=lambda row: (row["roi_score"], row["roi_color_reduction"], row["roi_mse_reduction"]), reverse=True)


def diagnostics(original: torch.Tensor, global_value: torch.Tensor, ours_value: torch.Tensor) -> list[dict[str, Any]]:
    global_patch = patch_mse_per_sample(global_value, original, 32, 32).numpy()
    ours_patch = patch_mse_per_sample(ours_value, original, 32, 32).numpy()
    global_color = [ciede_map(original[index], global_value[index]) for index in range(len(original))]
    ours_color = [ciede_map(original[index], ours_value[index]) for index in range(len(original))]
    rows = []
    for index in range(len(original)):
        global_mse = float((global_value[index] - original[index]).square().mean())
        ours_mse = float((ours_value[index] - original[index]).square().mean())
        global_local = psnr(float(np.sort(global_patch[index])[-4:].mean()))
        ours_local = psnr(float(np.sort(ours_patch[index])[-4:].mean()))
        roi_rows = roi_candidates(original[index], global_value[index], ours_value[index], global_color[index], ours_color[index])
        roi = roi_rows[0]
        mismatch = abs(psnr(global_mse) - psnr(ours_mse))
        row = {
            "image_id": str(index),
            "validation_index": index,
            "global_psnr": psnr(global_mse),
            "ours_psnr": psnr(ours_mse),
            "psnr_mismatch": mismatch,
            "global_local_psnr": global_local,
            "ours_local_psnr": ours_local,
            "delta_local_psnr_ours_minus_global": ours_local - global_local,
            "global_p95_mse": float(np.percentile(global_patch[index], 95)),
            "ours_p95_mse": float(np.percentile(ours_patch[index], 95)),
            "delta_p95_mse_ours_minus_global": float(np.percentile(ours_patch[index], 95) - np.percentile(global_patch[index], 95)),
            "global_color_map": global_color[index],
            "ours_color_map": ours_color[index],
            "roi": roi,
            "roi_candidates": roi_rows,
            "content_mean_luma": float(((original[index] * original.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)).sum(0)).mean()),
            "content_texture": float(np.abs(np.gradient(original[index].mean(0).numpy())).mean()),
        }
        rows.append(row)
    return rows


def rank_and_select(rows: list[dict[str, Any]], sources: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for row in rows:
        index = row["validation_index"]
        row["image_id"] = Path(sources[index]["filename"]).stem
        roi = row["roi"]
        row.update({key: roi[key] for key in ("roi_x", "roi_y", "roi_size", "roi_mse_reduction", "roi_color_reduction", "roi_residual_energy_reduction", "roi_structure_contrast", "roi_texture_energy", "roi_content_score", "roi_mean_luma", "roi_color_spread", "roi_score")})
        row["psnr_match_eligible"] = row["psnr_mismatch"] <= 0.05 + 1e-12
    eligible = [row for row in rows if row["psnr_match_eligible"]]
    pool = eligible if len(eligible) >= 2 else rows
    rows.sort(key=lambda row: (
        bool(row["psnr_match_eligible"]),
        row["delta_local_psnr_ours_minus_global"],
        row["roi_color_reduction"],
        row["roi_mse_reduction"],
        row["roi_structure_contrast"],
        row["roi_content_score"],
        -row["validation_index"],
    ), reverse=True)
    pool_ids = {id(row) for row in pool}
    ranked_pool = [row for row in rows if id(row) in pool_ids]
    for rank, row in enumerate(rows, 1):
        row["selection_rank"] = rank
    first = ranked_pool[0]
    def distance(row: dict[str, Any]) -> float:
        return (
            abs(row["content_mean_luma"] - first["content_mean_luma"])
            + 4.0 * abs(row["content_texture"] - first["content_texture"])
            + 2.0 * abs(row["roi_color_spread"] - first["roi_color_spread"])
        )
    for row in ranked_pool[1:]:
        row["content_diversity_distance_from_sample_1"] = distance(row)
    diverse = [row for row in ranked_pool[1:] if distance(row) >= 0.01]
    second = diverse[0] if diverse else max(ranked_pool[1:], key=distance)
    second["content_diversity_distance_from_sample_1"] = distance(second)
    first["selected"] = True
    second["selected"] = True
    return rows, [first, second]


def load_external(manifest: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_payload = load(HIDDEN_CACHE)
    if hidden_payload.get("manifest_sha256") != sha256(MANIFEST):
        raise RuntimeError("HiDDeN cache is bound to a different validation manifest")
    hidden = display_batch(rgb_from_model(hidden_payload["encoded"]))
    values = []
    for index in range(50):
        path = MASKWM_NATIVE / f"image_{index:03d}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        if array.shape != (DISPLAY_SIZE, DISPLAY_SIZE, 3):
            raise ValueError(f"unexpected MaskWM native shape {array.shape} at {path}")
        values.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
    maskwm = torch.stack(values)
    provenance = json.loads(MASKWM_PROVENANCE.read_text(encoding="utf-8"))
    if provenance.get("manifest_sha256") != sha256(MANIFEST):
        raise RuntimeError("MaskWM native outputs are bound to a different validation manifest")
    return hidden, maskwm


def image_from_tensor(value: torch.Tensor) -> Image.Image:
    array = np.rint(value.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def zoom_tile(value: torch.Tensor, roi: tuple[int, int, int, int]) -> Image.Image:
    x, y, width, height = roi
    crop = image_from_tensor(value[:, y * 4:(y + height) * 4, x * 4:(x + width) * 4])
    return crop.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST)


def residual_tile(value: torch.Tensor, reference: torch.Tensor, roi: tuple[int, int, int, int], vmax: float) -> Image.Image:
    x, y, width, height = roi
    residual = (value - reference).abs().mean(0).numpy() * RESIDUAL_SCALE
    crop = np.clip(residual[y * 4:(y + height) * 4, x * 4:(x + width) * 4] / max(vmax, 1e-12), 0.0, 1.0)
    color = (plt.get_cmap("magma")(crop)[..., :3] * 255.0).round().astype(np.uint8)
    return Image.fromarray(color, "RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST)


@dataclass
class Panel:
    image: Image.Image
    label: str
    roi: tuple[int, int, int, int] | None = None
    source_size: tuple[int, int] = (DISPLAY_SIZE, DISPLAY_SIZE)


CAPTION = (
    "Global and Ours are shown at validation-calibrated matched-PSNR operating points. "
    "HiDDeN-64 and MaskWM-D_64 are external references shown at their respective frozen operating points. "
    "Residual maps use a shared scale within each sample and are amplified ×10 for visualization."
)


def make_pptx_clean(output: Path, panels: list[list[Panel]], footer: str) -> None:
    tree = ooxml.group_tree()
    relationships = [("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml")]
    media: dict[str, bytes] = {}
    shape_id, rel_index = 2, 2
    nrows, ncols = len(panels), len(COLUMNS)
    left, right, top, bottom = 1.05, 0.14, 0.90, 0.55
    gap_x, gap_y = 0.10, 0.055
    box_w = (ooxml.SLIDE_W - left - right - (ncols - 1) * gap_x) / ncols
    box_h = (ooxml.SLIDE_H - top - bottom - (nrows - 1) * gap_y) / nrows
    image_size = min(box_w, box_h)
    ooxml.add_text_box(tree, shape_id, left, 7.15, ooxml.SLIDE_W - left - right, 0.25, "Figure 2 — matched-PSNR qualitative comparison", size=12, bold=True)
    shape_id += 1
    for col, label in enumerate(COLUMNS):
        x = left + col * (box_w + gap_x) + (box_w - image_size) / 2.0
        ooxml.add_text_box(tree, shape_id, x, 6.82, image_size, 0.28, label, size=7, bold=True)
        shape_id += 1
    for row, row_panels in enumerate(panels):
        y = bottom + (nrows - row - 1) * (box_h + gap_y) + (box_h - image_size) / 2.0
        ooxml.add_text_box(tree, shape_id, 0.03, y, 0.92, image_size, ROW_LABELS[row], size=7, bold=True, color=ooxml.MUTED, rotate=1)
        shape_id += 1
        for col, panel in enumerate(row_panels):
            x = left + col * (box_w + gap_x) + (box_w - image_size) / 2.0
            buffer = __import__("io").BytesIO()
            panel.image.save(buffer, format="PNG")
            media_name = f"panel_{len(media) + 1}.png"
            media[media_name] = buffer.getvalue()
            rel_id = f"rId{rel_index}"
            rel_index += 1
            relationships.append((rel_id, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"../media/{media_name}"))
            ooxml.add_picture(tree, shape_id, x, y, image_size, image_size, rel_id, panel.label)
            shape_id += 1
            if panel.roi is not None:
                rx, ry, rw, rh = panel.roi
                sw, sh = panel.source_size
                ooxml.add_rect(tree, shape_id, x + image_size * rx / sw, y + image_size * (1 - (ry + rh) / sh), image_size * rw / sw, image_size * rh / sh, line_color=ooxml.RED, line_width=11430)
                shape_id += 1
    ooxml.add_text_box(tree, shape_id, left, 0.08, ooxml.SLIDE_W - left - right, 0.38, footer, size=6, color=ooxml.MUTED, align="l")
    content_types = [(f"/ppt/media/{name}", "image/png", "override") for name in media]
    content_types += [
        ("/ppt/presentation.xml", "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml", "override"),
        ("/ppt/slides/slide1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slide+xml", "override"),
        ("/ppt/slideLayouts/slideLayout1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml", "override"),
        ("/ppt/slideMasters/slideMaster1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml", "override"),
        ("/ppt/theme/theme1.xml", "application/vnd.openxmlformats-officedocument.theme+xml", "override"),
        ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml", "override"),
        ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml", "override"),
    ]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", ooxml.content_types_xml(content_types))
        archive.writestr("_rels/.rels", ooxml.rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/package/2006/relationships/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", "http://schemas.openxmlformats.org/package/2006/relationships/extended-properties", "docProps/app.xml"),
        ]))
        archive.writestr("ppt/presentation.xml", ooxml.presentation_xml())
        archive.writestr("ppt/_rels/presentation.xml.rels", ooxml.rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml"),
            ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", "slides/slide1.xml"),
        ]))
        archive.writestr("ppt/slides/slide1.xml", ooxml.slide_xml(tree))
        archive.writestr("ppt/slides/_rels/slide1.xml.rels", ooxml.rels_xml(relationships))
        archive.writestr("ppt/slideLayouts/slideLayout1.xml", ooxml.slide_layout_xml())
        archive.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", ooxml.rels_xml([("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "../slideMasters/slideMaster1.xml")]))
        archive.writestr("ppt/slideMasters/slideMaster1.xml", ooxml.slide_master_xml())
        archive.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", ooxml.rels_xml([("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml")]))
        archive.writestr("ppt/theme/theme1.xml", ooxml.theme_xml())
        archive.writestr("docProps/core.xml", ooxml.core_props_xml())
        archive.writestr("docProps/app.xml", ooxml.app_props_xml())
        for name, data in media.items():
            archive.writestr(f"ppt/media/{name}", data)


def render_figure(display: dict[str, torch.Tensor], selected: list[dict[str, Any]], residual_vmax: list[float]) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panels: list[list[Panel]] = []
    for sample_number, row in enumerate(selected):
        index = row["validation_index"]
        roi = (int(row["roi_x"]), int(row["roi_y"]), int(row["roi_size"]), int(row["roi_size"]))
        reference = display["original"][index]
        full_row = []
        zoom_row = []
        residual_row = []
        for key, column in zip(KEYS, COLUMNS):
            full_row.append(Panel(image_from_tensor(display[key][index]), f"{column} sample {sample_number} full", (roi[0] * 4, roi[1] * 4, roi[2] * 4, roi[3] * 4)))
            zoom_row.append(Panel(zoom_tile(display[key][index], roi), f"{column} sample {sample_number} zoom"))
            residual_row.append(Panel(residual_tile(display[key][index], reference, roi, residual_vmax[sample_number]), f"{column} sample {sample_number} residual"))
        panels.extend((full_row, zoom_row, residual_row))

    fig = plt.figure(figsize=(15.8, 10.15), facecolor="white")
    grid = fig.add_gridspec(6, 5, left=0.075, right=0.995, top=0.805, bottom=0.185, wspace=0.025, hspace=0.075)
    for row_index, row_panels in enumerate(panels):
        for col_index, panel in enumerate(row_panels):
            ax = fig.add_subplot(grid[row_index, col_index])
            ax.imshow(panel.image, interpolation="nearest", aspect="equal")
            ax.set_axis_off()
            if row_index % 3 == 0:
                rx, ry, rw, rh = panel.roi or (0, 0, 0, 0)
                ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False, edgecolor=RED, linewidth=1.8))
            if row_index == 0:
                ax.set_title(COLUMNS[col_index], fontsize=9.0, color=INK, fontweight="bold", pad=8, wrap=True)
        first = fig.axes[-5].get_position()
        fig.text(0.008, first.y0 + first.height / 2.0, ROW_LABELS[row_index], ha="left", va="center", fontsize=8.5, color=INK if row_index % 3 == 0 else MUTED, fontweight="bold")
    fig.text(0.075, 0.945, "Figure 2 — matched-PSNR qualitative comparison", fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.075, 0.918, "Fixed validation samples; shared ROI per sample; external references retained at frozen operating points", fontsize=9.5, color=MUTED, ha="left", va="top")
    fig.text(0.075, 0.105, CAPTION, fontsize=7.5, color=MUTED, ha="left", va="top", wrap=True, linespacing=1.25)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=350, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    make_pptx_clean(OUTPUT_STEM.with_suffix(".pptx"), panels, CAPTION)
    return [OUTPUT_STEM.with_suffix(suffix) for suffix in (".png", ".pdf", ".pptx")]


def write_ranking(rows: Sequence[dict[str, Any]]) -> None:
    flattened = []
    for row in rows:
        flattened.append({key: value for key, value in row.items() if key not in {"global_color_map", "ours_color_map", "roi_candidates"}})
    fields = []
    for row in flattened:
        for key in row:
            if key not in fields:
                fields.append(key)
    with RANKING_CSV.open("w", newline="", encoding="utf-8") as stream:
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
    required = [MANIFEST, MATCHED_CSV, HIDDEN_CACHE, HIDDEN_CHECKPOINT, MASKWM_PROVENANCE, MASKWM_CHECKPOINT, GLOBAL_CHECKPOINT, OURS_CHECKPOINT]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    matched_rows = list(csv.DictReader(MATCHED_CSV.open(newline="", encoding="utf-8")))
    alpha_rows = {row["method"]: row for row in matched_rows if row["stage"] == "validation_selected"}
    expected_names = ("MBRS crop-trained global", "Ours = Hard Local-Tail + global OKLab")
    if any(name not in alpha_rows for name in expected_names):
        raise RuntimeError("matched-PSNR CSV does not contain both frozen validation operating points")
    if abs(float(alpha_rows[expected_names[0]]["alpha"]) - GLOBAL_ALPHA) > 1e-9 or abs(float(alpha_rows[expected_names[1]]["alpha"]) - OURS_ALPHA) > 1e-9:
        raise RuntimeError("supplied alpha constants do not match the frozen matched-PSNR report")

    manifest = load(MANIFEST)
    if manifest.get("split") != "validation" or len(manifest.get("images", [])) != 50:
        raise RuntimeError("Figure 2 requires the fixed 50-image validation manifest")
    images = manifest["images"].float()
    messages = manifest["messages"].float()
    if tuple(images.shape) != (50, 3, 128, 128) or tuple(messages.shape) != (50, 64):
        raise RuntimeError("validation manifest dimensions differ from the frozen 50x128x128/64 contract")

    print(f"recomputing matched outputs on {device}", flush=True)
    global_model = model_from_checkpoint(GLOBAL_CHECKPOINT, device)
    ours_model = model_from_checkpoint(OURS_CHECKPOINT, device)
    global_encoded = encode(global_model, images, messages, device)
    ours_encoded = encode(ours_model, images, messages, device)
    global_value = matched_output(images, global_encoded, GLOBAL_ALPHA)
    ours_value = matched_output(images, ours_encoded, OURS_ALPHA)
    del global_model, ours_model, global_encoded, ours_encoded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = diagnostics(rgb_from_model(images), global_value, ours_value)
    ranked, selected = rank_and_select(rows, manifest["sources"])
    for sample_number, row in enumerate(selected, 1):
        row["sample_number"] = sample_number
    write_ranking(ranked)

    hidden, maskwm = load_external(manifest)
    display = {
        "original": display_batch(rgb_from_model(images)),
        "hidden": hidden,
        "maskwm": maskwm,
        "global": display_batch(global_value),
        "ours": display_batch(ours_value),
    }
    residual_vmax = []
    for row in selected:
        index = row["validation_index"]
        roi = (int(row["roi_x"]), int(row["roi_y"]), int(row["roi_size"]), int(row["roi_size"]))
        reference = display["original"][index]
        vmax = 0.0
        for key in KEYS:
            residual = (display[key][index] - reference).abs().mean(0).numpy() * RESIDUAL_SCALE
            vmax = max(vmax, float(residual[roi[1] * 4:(roi[1] + roi[3]) * 4, roi[0] * 4:(roi[0] + roi[2]) * 4].max()))
        row["shared_residual_vmax_x10"] = max(vmax, 1e-8)
        residual_vmax.append(row["shared_residual_vmax_x10"])

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({
        "manifest_sha256": sha256(MANIFEST),
        "global_checkpoint_sha256": sha256(GLOBAL_CHECKPOINT),
        "ours_checkpoint_sha256": sha256(OURS_CHECKPOINT),
        "global_alpha": GLOBAL_ALPHA,
        "ours_alpha": OURS_ALPHA,
        "global_matched_rgb": global_value,
        "ours_matched_rgb": ours_value,
    }, MATCHED_OUTPUTS)
    generated = render_figure(display, selected, residual_vmax)

    source_paths = {
        "validation_manifest": str(MANIFEST),
        "HiDDeN_cache": str(HIDDEN_CACHE),
        "HiDDeN_checkpoint": str(HIDDEN_CHECKPOINT),
        "MaskWM_provenance": str(MASKWM_PROVENANCE),
        "MaskWM_checkpoint": str(MASKWM_CHECKPOINT),
        "MaskWM_native_output_directory": str(MASKWM_NATIVE),
        "MBRS_global_checkpoint": str(GLOBAL_CHECKPOINT),
        "Ours_checkpoint": str(OURS_CHECKPOINT),
        "matched_internal_output": str(MATCHED_OUTPUTS),
    }
    selected_manifest = []
    for row in selected:
        selected_manifest.append({
            "sample_number": row["sample_number"],
            "sample_id": row["image_id"],
            "validation_index": row["validation_index"],
            "source_filename": manifest["sources"][row["validation_index"]]["filename"],
            "roi": {"x": row["roi_x"], "y": row["roi_y"], "width": row["roi_size"], "height": row["roi_size"]},
            "global_alpha": GLOBAL_ALPHA,
            "ours_alpha": OURS_ALPHA,
            "global_psnr": row["global_psnr"],
            "ours_psnr": row["ours_psnr"],
            "psnr_mismatch": row["psnr_mismatch"],
            "global_local_psnr": row["global_local_psnr"],
            "ours_local_psnr": row["ours_local_psnr"],
            "delta_local_psnr_ours_minus_global": row["delta_local_psnr_ours_minus_global"],
            "global_p95_mse": row["global_p95_mse"],
            "ours_p95_mse": row["ours_p95_mse"],
            "delta_p95_mse_ours_minus_global": row["delta_p95_mse_ours_minus_global"],
            "global_roi_mse": row["roi_mse_reduction"] + 0.0,
            "ours_roi_mse": None,
            "roi_mse_reduction_global_minus_ours": row["roi_mse_reduction"],
            "roi_color_reduction_global_minus_ours": row["roi_color_reduction"],
            "roi_residual_energy_reduction_global_minus_ours": row["roi_residual_energy_reduction"],
            "roi_structure_contrast": row["roi_structure_contrast"],
            "shared_residual_vmax_x10": row["shared_residual_vmax_x10"],
            "selection_rank": row["selection_rank"],
            "psnr_match_eligible": row["psnr_match_eligible"],
            "content_diversity_distance_from_sample_1": row.get("content_diversity_distance_from_sample_1"),
        })
    # Fill exact ROI MSE values after selecting the ROI, keeping the report
    # explicit about the direction of every reduction.
    for item, row in zip(selected_manifest, selected):
        x, y, size = row["roi_x"], row["roi_y"], row["roi_size"]
        reference = rgb_from_model(images)[row["validation_index"]][:, y:y + size, x:x + size]
        global_roi = global_value[row["validation_index"]][:, y:y + size, x:x + size]
        ours_roi = ours_value[row["validation_index"]][:, y:y + size, x:x + size]
        item["global_roi_mse"] = float((global_roi - reference).square().mean())
        item["ours_roi_mse"] = float((ours_roi - reference).square().mean())
        item["global_roi_ciede2000"] = float(row["global_color_map"][y:y + size, x:x + size].mean())
        item["ours_roi_ciede2000"] = float(row["ours_color_map"][y:y + size, x:x + size].mean())

    manifest_payload = {
        "scope": "fixed 50-image validation manifest only",
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256(MANIFEST),
        "matched_psnr_report": str(MATCHED_CSV),
        "matched_psnr_report_sha256": sha256(MATCHED_CSV),
        "alphas": {"MBRS crop-trained global": GLOBAL_ALPHA, "Ours = Hard Local-Tail + global OKLab": OURS_ALPHA},
        "formula": "x_alpha = x + alpha * (x_hat - x), applied in normalized [-1,1] tensor space before RGB clipping/display",
        "external_operating_point": "HiDDeN-64 and MaskWM-D_64 loaded from existing frozen validation assets; no residual-strength scaling",
        "selection_rule": {
            "eligibility": "per-image matched PSNR mismatch <= 0.05 dB preferred; original ROI mean luminance in [0.06,0.94] and texture above validation-image percentile floor",
            "primary": "higher Ours-minus-Global native32 Top-25 local PSNR",
            "secondary": "higher shared-ROI CIEDE2000 reduction, higher shared-ROI MSE reduction, residual-structure contrast, informative content",
            "diversity": "second sample selected with content distance from sample 1 >= 0.01 when available",
            "same_roi_all_columns": True,
            "project_test_used_for_selection": False,
        },
        "selected_samples": selected_manifest,
        "sources_outputs": source_paths,
        "generated_outputs": {str(path): sha256(path) for path in generated},
        "renderer": str(Path(__file__)),
        "renderer_sha256": sha256(Path(__file__)),
        "caption": CAPTION,
    }
    SELECTION_MANIFEST.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Figure 2 matched-PSNR selection report",
        "",
        "This figure uses only the fixed 50-image validation manifest. Global and Ours are recomputed from their frozen seed17 checkpoints with the report-frozen residual scales; α is fixed per method and is not retuned per image. HiDDeN-64 and MaskWM-D_64 remain at their existing frozen/native operating points.",
        "",
        f"- Global α: `{GLOBAL_ALPHA:.9f}`",
        f"- Ours α: `{OURS_ALPHA:.9f}`",
        f"- Validation manifest SHA-256: `{sha256(MANIFEST)}`",
        f"- Matched internal output: `{MATCHED_OUTPUTS}`",
        "- Project-test images were not opened or used for visual selection.",
        "",
        "## Selected samples and shared ROIs",
        "",
        "| Sample | ID | Validation index | ROI `(x,y,w,h)` | Global PSNR | Ours PSNR | mismatch | Δ local PSNR (Ours−Global) | Δ P95 MSE (Ours−Global) | Global ROI CIEDE | Ours ROI CIEDE | Δ ROI CIEDE (Global−Ours) | shared residual vmax ×10 |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in selected_manifest:
        roi = item["roi"]
        report_lines.append(
            f"| {item['sample_number']} | `{item['sample_id']}` | {item['validation_index']} | `({roi['x']},{roi['y']},{roi['width']},{roi['height']})` | {item['global_psnr']:.6f} | {item['ours_psnr']:.6f} | {item['psnr_mismatch']:.6f} | {item['delta_local_psnr_ours_minus_global']:+.6f} | {item['delta_p95_mse_ours_minus_global']:+.3e} | {item['global_roi_ciede2000']:.6f} | {item['ours_roi_ciede2000']:.6f} | {item['roi_color_reduction_global_minus_ours']:+.6f} | {item['shared_residual_vmax_x10']:.6f} |")
    report_lines += [
        "",
        "## Exact sources and outputs",
        "",
    ]
    for key, path in source_paths.items():
        report_lines.append(f"- `{key}`: `{path}`")
    report_lines += [
        "",
        "## Layout and rendering contract",
        "",
        "- Columns: Original; HiDDeN-64; MaskWM-D_64; MBRS crop-trained global; Ours = Hard Local-Tail + global OKLab.",
        "- Rows: Sample 1, Zoom, Residual ×10, Sample 2, Zoom, Residual ×10.",
        "- Red ROI boxes appear only on full-image rows. Zoom and residual tiles have no borders.",
        "- Each sample uses one shared residual range across all five columns; residual maps are mean absolute displayed-RGB residuals amplified ×10.",
        f"- Caption: “{CAPTION}”",
        "",
        "## Generated files",
        "",
    ]
    for path in generated:
        report_lines.append(f"- `{path}` (SHA-256 `{sha256(path)}`)")
    report_lines += [
        "",
        f"Complete machine-readable selection/provenance: `{SELECTION_MANIFEST}`.",
        f"Complete validation ranking: `{RANKING_CSV}`.",
    ]
    SELECTION_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"selected validation indices {[row['validation_index'] for row in selected]}", flush=True)
    for path in generated:
        print(path, flush=True)


if __name__ == "__main__":
    main()
