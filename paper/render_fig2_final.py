#!/usr/bin/env python3
"""Render the final five-column qualitative Figure 2 from frozen outputs."""

# The project-root path is inserted before local package imports so the file
# can be executed directly from the paper/ directory.
# ruff: noqa: E402

from __future__ import annotations

import io
import csv
import sys
import zipfile
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.generate_fig2_qualitative_search import Panel  # noqa: E402
from experiments.regenerate_figure2 import (
    PROJECT,
    MOUNT,
    OKLAB_RESULTS,
    display_batch,
    load,
    load_encoded,
    load_maskwm_native,
    local_stats,
    choose_roi,
    color_stats,
    psnr,
    sha256,
)
from paper import render_fig2_qualitative_editable as ooxml  # noqa: E402


VALIDATION_MANIFEST = MOUNT / "reports/content_selector/validation_manifest.pt"
HIDDEN_CACHE = MOUNT / "results/fig2_stress/external_baselines/hidden_64bit_validation.pt"
GLOBAL_OUTPUT = MOUNT / "reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/controlled_seed17_global_continuation_outputs.pt"
OUTPUT = PROJECT / "paper/figures"
SELECTION_REPORT = PROJECT / "reports/final_figure2_selection.md"

METHODS = (
    "Original",
    "HiDDeN-64 (external)",
    "MaskWM-D_64 (external)",
    "MBRS crop-trained global",
    "Ours",
)
KEYS = ("original", "hidden", "maskwm", "global", "ours")
DISPLAY_SIZE = 512
LOCAL_ERROR_AMPLIFICATION = 10.0
RED = "#e5483c"
INK = "#202934"
MUTED = "#66727d"


def to_pil(value: torch.Tensor) -> Image.Image:
    array = np.rint(value.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def crop_direct(value: torch.Tensor, roi: tuple[int, int, int, int]) -> Image.Image:
    x, y, width, height = roi
    crop = to_pil(value[:, y * 4:(y + height) * 4, x * 4:(x + width) * 4])
    return crop.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST)


def local_error_crop(value: torch.Tensor, reference: torch.Tensor, roi: tuple[int, int, int, int], vmax: float) -> Image.Image:
    x, y, width, height = roi
    local_error = (value - reference).abs().mean(0).numpy() * LOCAL_ERROR_AMPLIFICATION
    crop = np.clip(local_error[y * 4:(y + height) * 4, x * 4:(x + width) * 4] / max(vmax, 1e-12), 0, 1)
    rgb = (plt.get_cmap("magma")(crop)[..., :3] * 255.0).round().astype(np.uint8)
    return Image.fromarray(rgb, "RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST)


def local_error_placeholder() -> Image.Image:
    image = Image.new("RGB", (DISPLAY_SIZE, DISPLAY_SIZE), "white")
    draw = ImageDraw.Draw(image)
    center = DISPLAY_SIZE // 2
    draw.line((center - 22, center, center + 22, center), fill=MUTED, width=6)
    return image


def normalized(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def select_samples(manifest: dict[str, Any], original: torch.Tensor, hard: torch.Tensor, ours: torch.Tensor) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    for index in range(50):
        roi, roi_info = choose_roi(original[index], hard[index], ours[index])
        local = local_stats(original[index], hard[index], ours[index], roi)
        color = color_stats(original[index], hard[index], ours[index], roi)
        x, y, width, height = roi
        hard_residual = (hard[index] - original[index]).abs().mean(0)
        roi_clarity = float(hard_residual[y:y + height, x:x + width].mean())
        records.append({
            "image_id": Path(manifest["sources"][index]["filename"]).stem,
            "validation_index": index,
            "roi_x": x,
            "roi_y": y,
            "roi_width": width,
            "roi_height": height,
            "roi": roi,
            **roi_info,
            "local_psnr_hard": local["local_psnr_global"],
            "local_psnr_ours": local["local_psnr_hard"],
            "delta_local_psnr": local["local_psnr_hard"] - local["local_psnr_global"],
            "p95_hard": local["p95_global"],
            "p95_ours": local["p95_hard"],
            "p95_reduction": local["p95_global"] - local["p95_hard"],
            "roi_residual_energy_hard": roi_clarity,
            "roi_residual_energy_reduction": local["roi_residual_energy_reduction"],
            "roi_structured_tv_reduction": local["roi_structured_tv_reduction"],
            "roi_ciede2000_reduction": color["roi_ciede2000_reduction"],
            "roi_chroma_reduction": color["roi_chroma_reduction"],
            "content_texture": float(np.abs(np.gradient(original[index].mean(0).numpy())).mean()),
            "content_mean": float(original[index].mean()),
        })
    local = normalized([max(0.0, row["delta_local_psnr"]) for row in records])
    p95 = normalized([max(0.0, row["p95_reduction"]) for row in records])
    color = normalized([max(0.0, row["roi_ciede2000_reduction"]) for row in records])
    clarity = normalized([row["roi_residual_energy_hard"] for row in records])
    texture = normalized([row["content_texture"] for row in records])
    for row, a, b, c, d, e in zip(records, local, p95, color, clarity, texture):
        row["selection_score"] = 0.35 * a + 0.25 * b + 0.20 * c + 0.10 * d + 0.10 * e
    ranked = sorted(records, key=lambda row: (row["selection_score"], row["delta_local_psnr"], row["p95_reduction"]), reverse=True)
    selected = [ranked[0]]
    for row in ranked[1:]:
        distance = abs(row["content_mean"] - selected[0]["content_mean"]) + 4.0 * abs(row["content_texture"] - selected[0]["content_texture"])
        row["content_diversity_distance"] = distance
        if distance >= 0.01:
            selected.append(row)
            break
    if len(selected) < 2:
        selected = ranked[:2]
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return ranked, selected


def write_selection_report(ranked: list[dict[str, Any]], selected: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# Final Figure 2 selection",
        "",
        "The final five-column Figure 2 uses only the fixed 50-image validation manifest. Project-test images are not used for qualitative sample or ROI selection.",
        "",
        f"- Manifest: `{VALIDATION_MANIFEST}`; SHA-256 `{sha256(VALIDATION_MANIFEST)}`.",
        "- Candidate criterion: balanced score of local PSNR benefit (35%), P95 local-tail reduction (25%), visible color-error reduction by ROI CIEDE2000 (20%), residual diagnostic clarity (10%), and content texture/diversity (10%).",
        "- ROIs are selected from the Hard Local-Tail → Ours comparison on native 128×128 clipped RGB outputs, then shared across all five columns within each sample.",
        "- The selection is illustrative rather than representative; the balanced score avoids selecting only the two largest numeric gains.",
        "",
        "| Figure sample | Image ID | Validation index | ROI `(x,y,w,h)` | Δ local PSNR | P95 reduction | ROI CIEDE2000 reduction | ROI chroma reduction | ROI residual-energy reduction | Rationale |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for sample, row in enumerate(selected, 1):
        lines.append(f"| {sample} | `{row['image_id']}` | {row['validation_index']} | `({row['roi_x']},{row['roi_y']},{row['roi_width']},{row['roi_height']})` | {row['delta_local_psnr']:+.3f} dB | {row['p95_reduction']:+.3e} | {row['roi_ciede2000_reduction']:+.3f} | {row['roi_chroma_reduction']:+.3f} | {row['roi_residual_energy_reduction']:+.3e} | balanced local-tail/color/residual criterion with content diversity |")
    lines += [
        "",
        "## Reader-facing boundary",
        "",
        "The final figure has exactly five columns: Original, HiDDeN-64 (external), MaskWM-D_64 (external), MBRS crop-trained global, and Ours. Ours is the frozen Hard Local-Tail + global OKLab method. Hard Local-Tail is kept for the quantitative ablation chain, not as a sixth final-figure column.",
        "",
        "Caption text used in the figure: “Qualitative comparison on fixed validation samples. External methods use their respective released/retrained inference protocols and are included as reference baselines rather than strictly matched training comparisons. Red boxes indicate shared ROIs. Local-error heatmaps show mean absolute RGB difference inside the shared ROI, amplified ×10 for visibility with one shared scale per sample; the Original column is not applicable. Ours denotes the proposed Hard Local-Tail model with OKLab color-aware regularization.”",
    ]
    SELECTION_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def header_labels(display: dict[str, torch.Tensor], ber30: dict[str, float]) -> list[str]:
    reference = display["original"]
    values = {
        key: psnr(float((display[key] - reference).square().mean()))
        for key in KEYS[1:]
    }
    return [
        "Original\nPSNR ∞ (reference)",
        f"HiDDeN-64\n(external)\nPSNR {values['hidden']:.2f} dB",
        f"MaskWM-D_64\n(external)\nPSNR {values['maskwm']:.2f} dB",
        f"MBRS crop-trained global\nPSNR {values['global']:.2f} dB\nBER30 {ber30['global']:.3f}",
        f"Ours\nPSNR {values['ours']:.2f} dB\nBER30 {ber30['ours']:.3f}",
    ]


def caption() -> str:
    return ("Qualitative comparison on fixed validation samples. External methods use released/retrained inference protocols and are reference baselines, not strictly matched training comparisons. Red boxes indicate shared ROIs. Local-error heatmaps show mean absolute RGB difference inside the shared ROI, amplified ×10 for visibility with one shared scale per sample; the Original column is not applicable. Ours denotes Hard Local-Tail with OKLab color-aware regularization.")


def colorbar_bytes() -> bytes:
    gradient = np.linspace(1.0, 0.0, 256, dtype=np.float32)[:, None]
    rgb = (plt.get_cmap("magma")(gradient)[..., :3] * 255.0).round().astype(np.uint8)
    image = Image.fromarray(np.repeat(rgb, 18, axis=1), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_final_pptx(output: Path, selected: list[dict[str, Any]], panels: list[list[Panel]], labels: list[str], local_error_ranges: list[float]) -> None:
    tree = ooxml.group_tree()
    relationships = [("rId1", "http://schemas.openxmlformats.org/presentationml/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml")]
    media: dict[str, bytes] = {}
    rel_index, shape_id = 2, 2
    left, right, bottom, top = 0.92, 0.38, 0.42, 1.18
    gap_x, gap_y = 0.10, 0.08
    box_w = (ooxml.SLIDE_W - left - right - 4 * gap_x) / 5
    box_h = (ooxml.SLIDE_H - top - bottom - 5 * gap_y) / 6
    image_size = min(box_w, box_h)
    ooxml.add_text_box(tree, shape_id, left, 7.10, ooxml.SLIDE_W - left - right, 0.28, "Figure 2 — final qualitative comparison", size=12, bold=True)
    shape_id += 1
    for col, label_text in enumerate(labels):
        x = left + col * (box_w + gap_x) + (box_w - image_size) / 2
        ooxml.add_text_box(tree, shape_id, x, 6.76, image_size, 0.34, label_text, size=8, bold=True)
        shape_id += 1
    row_labels = ("Sample 1", "Zoom", "Local error", "Sample 2", "Zoom", "Local error")
    for row_index, row_label in enumerate(row_labels):
        y = bottom + (6 - row_index - 1) * (box_h + gap_y) + (box_h - image_size) / 2
        ooxml.add_text_box(tree, shape_id, 0.06, y, 0.72, image_size, row_label, size=7.5, bold=True)
        shape_id += 1
        for col, panel in enumerate(panels[row_index]):
            x = left + col * (box_w + gap_x) + (box_w - image_size) / 2
            data = io.BytesIO()
            panel.image.save(data, format="PNG")
            media_name = f"panel_{len(media) + 1}.png"
            media[media_name] = data.getvalue()
            rel_id = f"rId{rel_index}"
            rel_index += 1
            relationships.append((rel_id, "http://schemas.openxmlformats.org/presentationml/2006/relationships/image", f"../media/{media_name}"))
            ooxml.add_picture(tree, shape_id, x, y, image_size, image_size, rel_id, panel.label or f"Panel {row_index + 1},{col + 1}")
            shape_id += 1
            if row_index % 3 == 0 and panel.roi is not None:
                rx, ry, rw, rh = panel.roi
                ooxml.add_rect(tree, shape_id, x + image_size * rx / 512.0, y + image_size * (1 - (ry + rh) / 512.0), image_size * rw / 512.0, image_size * rh / 512.0, line_color="E5483C", line_width=15240)
                shape_id += 1
        if row_index % 3 == 2:
            bar_name = f"colorbar_{row_index}.png"
            media[bar_name] = colorbar_bytes()
            rel_id = f"rId{rel_index}"
            rel_index += 1
            relationships.append((rel_id, "http://schemas.openxmlformats.org/presentationml/2006/relationships/image", f"../media/{bar_name}"))
            ooxml.add_picture(tree, shape_id, ooxml.SLIDE_W - 0.22, y, 0.10, image_size, rel_id, f"Local error colorbar {row_index}")
            shape_id += 1
            value = local_error_ranges[row_index // 3]
            ooxml.add_text_box(tree, shape_id, ooxml.SLIDE_W - 0.38, y + image_size - 0.08, 0.15, 0.15, f"{value:.2f}", size=5, color="66727D")
            shape_id += 1
            ooxml.add_text_box(tree, shape_id, ooxml.SLIDE_W - 0.38, y - 0.07, 0.15, 0.15, "0", size=5, color="66727D")
            shape_id += 1

    footer = "\n".join([
        "Fixed validation samples; shared red ROIs.",
        "External methods are reference baselines under released/retrained protocols, not strictly matched training comparisons.",
        "Local error ×10 uses a shared per-sample scale; Original is not applicable; Ours = Hard Local-Tail + OKLab.",
    ])
    ooxml.add_text_box(tree, shape_id, left, 0.06, ooxml.SLIDE_W - left - right, 0.30, footer, size=6.5, color="66727D")
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
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
        ]))
        archive.writestr("ppt/presentation.xml", ooxml.presentation_xml())
        archive.writestr("ppt/_rels/presentation.xml.rels", ooxml.rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/presentationml/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml"),
            ("rId2", "http://schemas.openxmlformats.org/presentationml/2006/relationships/slide", "slides/slide1.xml"),
        ]))
        archive.writestr("ppt/slides/slide1.xml", ooxml.slide_xml(tree))
        archive.writestr("ppt/slides/_rels/slide1.xml.rels", ooxml.rels_xml(relationships))
        archive.writestr("ppt/slideLayouts/slideLayout1.xml", ooxml.slide_layout_xml())
        archive.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", ooxml.rels_xml([("rId1", "http://schemas.openxmlformats.org/presentationml/2006/relationships/slideMaster", "../slideMasters/slideMaster1.xml")]))
        archive.writestr("ppt/slideMasters/slideMaster1.xml", ooxml.slide_master_xml())
        archive.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", ooxml.rels_xml([("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml")]))
        archive.writestr("ppt/theme/theme1.xml", ooxml.theme_xml())
        archive.writestr("docProps/core.xml", ooxml.core_props_xml())
        archive.writestr("docProps/app.xml", ooxml.app_props_xml())
        for name, data in media.items():
            archive.writestr(f"ppt/media/{name}", data)


def render() -> None:
    manifest = load(VALIDATION_MANIFEST)
    original = ((manifest["images"].float() + 1.0) / 2.0).clamp(0, 1)
    hard = load_encoded(GLOBAL_OUTPUT.parent / "controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_outputs.pt")
    ours = load_encoded(OKLAB_RESULTS)
    ranked, selected = select_samples(manifest, original, hard, ours)
    write_selection_report(ranked, selected, manifest)

    hidden = load_encoded(HIDDEN_CACHE)
    display = {
        "original": display_batch(original),
        "hidden": display_batch(hidden),
        "maskwm": load_maskwm_native(),
        "global": display_batch(load_encoded(GLOBAL_OUTPUT)),
        "ours": display_batch(ours),
    }
    ber_rows = list(csv.DictReader((MOUNT / "reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/summary.csv").open(newline="")))
    ber30 = {
        "global": float(next(row for row in ber_rows if row["run"] == "controlled_seed17_global_continuation")["ber30"]),
        "ours": float(next(row for row in ber_rows if row["run"] == "seed17_crop_hard16_stride8_top10_global_oklab_g25")["ber30"]),
    }
    labels = header_labels(display, ber30)
    panels: list[list[Panel]] = []
    local_error_ranges = []
    for sample_number, record in enumerate(selected, 1):
        index = record["validation_index"]
        roi = tuple(int(record[key]) for key in ("roi_x", "roi_y", "roi_width", "roi_height"))
        reference = display["original"][index]
        vmax = max(float(((display[key][index] - reference).abs().mean(0)[roi[1] * 4:(roi[1] + roi[3]) * 4, roi[0] * 4:(roi[0] + roi[2]) * 4] * 10).max()) for key in KEYS)
        local_error_ranges.append(max(vmax, 1e-8))
        full, zoom, local_error = [], [], []
        for key in KEYS:
            full.append(Panel(to_pil(display[key][index]), label=f"{METHODS[KEYS.index(key)]} sample {sample_number} full", roi=(roi[0] * 4, roi[1] * 4, roi[2] * 4, roi[3] * 4), source_size=(512, 512)))
            zoom.append(Panel(crop_direct(display[key][index], roi), label=f"{METHODS[KEYS.index(key)]} sample {sample_number} zoom"))
            panel = local_error_placeholder() if key == "original" else local_error_crop(display[key][index], reference, roi, local_error_ranges[-1])
            local_error.append(Panel(panel, label=f"{METHODS[KEYS.index(key)]} sample {sample_number} local error"))
        panels.extend((full, zoom, local_error))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "figure2_final"
    figure = plt.figure(figsize=(14.6, 8.8), facecolor="white")
    grid = figure.add_gridspec(6, 6, width_ratios=[1, 1, 1, 1, 1, 0.10], left=0.095, right=0.985, top=0.79, bottom=0.15, wspace=0.035, hspace=0.18)
    axes = []
    for row_index, row_panels in enumerate(panels):
        row_axes = []
        for col_index, panel in enumerate(row_panels):
            ax = figure.add_subplot(grid[row_index, col_index])
            ax.imshow(np.asarray(panel.image), interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index % 3 == 0:
                rx, ry, rw, rh = panel.roi or (0, 0, 0, 0)
                ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False, edgecolor=RED, linewidth=1.8))
            if row_index == 0:
                ax.set_title(labels[col_index], fontsize=8.6, color=INK, fontweight="bold", pad=9, linespacing=1.1)
            row_axes.append(ax)
        axes.append(row_axes)
    figure.canvas.draw()
    row_labels = ("Sample 1", "Zoom", "Local error", "Sample 2", "Zoom", "Local error")
    for row_index, row_axes in enumerate(axes):
        position = row_axes[0].get_position()
        figure.text(0.008, position.y0 + position.height / 2.0, row_labels[row_index], ha="left", va="center", fontsize=8.2 if row_index % 3 else 8.6, color=INK if row_index % 3 else INK, fontweight="bold")
        if row_index % 3 == 2:
            cax = figure.add_subplot(grid[row_index, 5])
            scalar = matplotlib.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=local_error_ranges[row_index // 3]), cmap="magma")
            colorbar = figure.colorbar(scalar, cax=cax)
            colorbar.ax.tick_params(labelsize=5.5, colors=MUTED, length=2)
            colorbar.set_label("×10", fontsize=6.5, color=MUTED, labelpad=1)
    figure.text(0.095, 0.965, "Figure 2 — qualitative comparison", fontsize=14.5, fontweight="bold", color=INK, ha="left", va="top")
    figure.text(0.095, 0.936, "Global average control → hard local-tail control → color-aware refinement", fontsize=9.2, color=MUTED, ha="left", va="top")
    figure.text(0.985, 0.965, "red outline = shared ROI", fontsize=7.5, color=RED, ha="right", va="top")
    figure.text(0.095, 0.108, caption(), fontsize=6.8, color=MUTED, ha="left", va="top", wrap=True)
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    figure.savefig(stem.with_suffix(".png"), dpi=350, facecolor="white")
    plt.close(figure)
    make_final_pptx(stem.with_suffix(".pptx"), selected, panels, labels, local_error_ranges)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".pptx"))


if __name__ == "__main__":
    render()
