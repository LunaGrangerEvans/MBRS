#!/usr/bin/env python3
"""Generate the composite paper Figure 2 from frozen MBRS artifacts.

Top: one preregistered validation example with contextual external references.
Bottom: paired natural-project-test evidence for Global versus Ours.

The script performs no training, checkpoint selection, or parameter tuning.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")

VALIDATION_MANIFEST = MOUNT / "reports/content_selector/validation_manifest.pt"
HIDDEN_CACHE = MOUNT / "results/fig2_stress/external_baselines/hidden_64bit_validation.pt"
MASKWM_PROVENANCE = MOUNT / "external_baselines/outputs/maskwm_validation/D_64bits/provenance.json"
MASKWM_NATIVE = MOUNT / "external_baselines/outputs/maskwm_validation/D_64bits/native_512"
VALIDATION_OUTPUT_DIR = MOUNT / "reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25"
GLOBAL_VALIDATION_OUTPUT = VALIDATION_OUTPUT_DIR / "controlled_seed17_global_continuation_outputs.pt"
OURS_VALIDATION_OUTPUT = VALIDATION_OUTPUT_DIR / "seed17_crop_hard16_stride8_top10_global_oklab_g25_outputs.pt"

PROJECT_TEST_MANIFEST = MOUNT / "reports/uniform_eval_manifest.pt"
PROJECT_TEST_OUTPUT_DIR = MOUNT / "reports/crop_global_oklab/test_seed17_crop_hard16_stride8_top10_global_oklab_g25"
GLOBAL_PROJECT_TEST_OUTPUT = PROJECT_TEST_OUTPUT_DIR / "controlled_seed17_global_continuation_outputs.pt"
OURS_PROJECT_TEST_OUTPUT = PROJECT_TEST_OUTPUT_DIR / "seed17_crop_hard16_stride8_top10_global_oklab_g25_outputs.pt"
GLOBAL_PER_IMAGE = PROJECT / "paper_extension_study/continuation_seed/global/seed17/project_test_per_image.csv"
OURS_PER_IMAGE = PROJECT / "paper_extension_study/continuation_seed/ours/seed17/project_test_per_image.csv"
SELECTION_REPORT = PROJECT / "reports/final_figure2_selection.md"

FIGURE_STEM = OUTPUT_DIR / "figure2_final"
CAPTION_PATH = OUTPUT_DIR / "figure2_caption.txt"
NOTES_PATH = OUTPUT_DIR / "figure2_generation_notes.md"
PAIRED_CSV = OUTPUT_DIR / "figure2_paired_points.csv"
PROFILE_CSV = OUTPUT_DIR / "figure2_local_tail_profile.csv"

FIGURE_WIDTH_IN = 7.16
FIGURE_HEIGHT_IN = 6.55
PNG_DPI = 350
DISPLAY_SIZE = 512
ERROR_AMPLIFICATION = 10.0

SAMPLE_INDEX = 13
SAMPLE_ID = "0814"
ROI_SOURCE = (88, 48, 32, 32)

GLOBAL_COLOR = "#0072B2"
OURS_COLOR = "#D55E00"
IDENTITY_COLOR = "#7A858F"
MUTED = "#66727D"
INK = "#202934"
ROI_COLOR = "#E5483C"
TAIL_COLOR = "#F3E4BD"
FAVORABLE = "#167C5A"

METHOD_KEYS = ("original", "hidden", "maskwm", "global", "ours")
METHOD_LABELS = {
    "original": "Original",
    "hidden": "HiDDeN-64",
    "maskwm": "MaskWM-D_64",
    "global": "MBRS Global",
    "ours": "Ours",
}


def load_torch(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_from_model(value: torch.Tensor) -> torch.Tensor:
    return ((value.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def display_image(value: torch.Tensor) -> torch.Tensor:
    if tuple(value.shape) != (3, 128, 128):
        raise ValueError(f"expected a 3x128x128 tensor, received {tuple(value.shape)}")
    return F.interpolate(
        value.unsqueeze(0),
        size=(DISPLAY_SIZE, DISPLAY_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]


def tensor_to_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def load_maskwm(index: int) -> torch.Tensor:
    path = MASKWM_NATIVE / f"image_{index:03d}.png"
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if array.shape != (DISPLAY_SIZE, DISPLAY_SIZE, 3):
        raise ValueError(f"unexpected MaskWM image shape at {path}: {array.shape}")
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def load_qualitative_data() -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, float]]:
    manifest = load_torch(VALIDATION_MANIFEST)
    hidden_payload = load_torch(HIDDEN_CACHE)
    global_payload = load_torch(GLOBAL_VALIDATION_OUTPUT)
    ours_payload = load_torch(OURS_VALIDATION_OUTPUT)

    if tuple(manifest["images"].shape) != (50, 3, 128, 128):
        raise ValueError("validation manifest does not satisfy the frozen 50-image contract")
    if Path(manifest["sources"][SAMPLE_INDEX]["filename"]).stem != SAMPLE_ID:
        raise ValueError("selected validation sample no longer matches the audited sample ID")
    if hidden_payload.get("manifest_sha256") != sha256(VALIDATION_MANIFEST):
        raise ValueError("HiDDeN cache is not aligned to the fixed validation manifest")
    for payload, name in ((global_payload, "Global"), (ours_payload, "Ours")):
        if tuple(payload["encoded"].shape) != (50, 3, 128, 128):
            raise ValueError(f"{name} validation output has an unexpected shape")

    reference_native = rgb_from_model(manifest["images"])
    hidden_native = rgb_from_model(hidden_payload["encoded"])
    global_native = rgb_from_model(global_payload["encoded"])
    ours_native = rgb_from_model(ours_payload["encoded"])

    display = {
        "original": display_image(reference_native[SAMPLE_INDEX]),
        "hidden": display_image(hidden_native[SAMPLE_INDEX]),
        "maskwm": load_maskwm(SAMPLE_INDEX),
        "global": display_image(global_native[SAMPLE_INDEX]),
        "ours": display_image(ours_native[SAMPLE_INDEX]),
    }

    # Compute mean display PSNR on the same 512x512 canvas used by the
    # qualitative panel. This is contextual validation evidence only.
    squared_error_sums = {key: 0.0 for key in METHOD_KEYS[1:]}
    element_count = 50 * 3 * DISPLAY_SIZE * DISPLAY_SIZE
    with torch.inference_mode():
        for index in range(50):
            reference = display_image(reference_native[index])
            candidates = {
                "hidden": display_image(hidden_native[index]),
                "maskwm": load_maskwm(index),
                "global": display_image(global_native[index]),
                "ours": display_image(ours_native[index]),
            }
            for key, value in candidates.items():
                squared_error_sums[key] += float((value - reference).square().sum())
    psnr = {
        key: -10.0 * np.log10(squared_error_sums[key] / element_count)
        for key in METHOD_KEYS[1:]
    }

    x, y, width, height = (value * 4 for value in ROI_SOURCE)
    reference = display["original"]
    roi_mae = {}
    for key in METHOD_KEYS[1:]:
        residual = (display[key] - reference).abs().mean(dim=0)
        roi_mae[key] = float(residual[y:y + height, x:x + width].mean())
    return display, psnr, roi_mae


def load_per_image(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row["image_index"]))
    if len(rows) != 50 or [int(row["image_index"]) for row in rows] != list(range(50)):
        raise ValueError(f"{path} does not contain exactly one ordered row for indices 0-49")
    if {row["split"] for row in rows} != {"project_test"}:
        raise ValueError(f"{path} contains a non-project-test split")
    return rows


def patch_mse_grid(value: torch.Tensor, reference: torch.Tensor) -> np.ndarray:
    if tuple(value.shape) != (50, 3, 128, 128) or value.shape != reference.shape:
        raise ValueError("project-test output tensors do not satisfy the frozen shape contract")
    residual = (value - reference).square().mean(dim=1, keepdim=True)
    return F.avg_pool2d(residual, kernel_size=32, stride=32).flatten(1).numpy()


def load_quantitative_data() -> dict[str, np.ndarray]:
    global_rows = load_per_image(GLOBAL_PER_IMAGE)
    ours_rows = load_per_image(OURS_PER_IMAGE)
    result = {
        "image_index": np.arange(50),
        "global_p95": np.array([float(row["patch_mse_p95"]) for row in global_rows]),
        "ours_p95": np.array([float(row["patch_mse_p95"]) for row in ours_rows]),
        "global_local_psnr": np.array([float(row["top25_local_psnr"]) for row in global_rows]),
        "ours_local_psnr": np.array([float(row["top25_local_psnr"]) for row in ours_rows]),
    }

    manifest = load_torch(PROJECT_TEST_MANIFEST)
    reference = rgb_from_model(manifest["images"])
    global_value = rgb_from_model(load_torch(GLOBAL_PROJECT_TEST_OUTPUT)["encoded"])
    ours_value = rgb_from_model(load_torch(OURS_PROJECT_TEST_OUTPUT)["encoded"])
    global_patches = patch_mse_grid(global_value, reference)
    ours_patches = patch_mse_grid(ours_value, reference)
    if global_patches.shape != (50, 16) or ours_patches.shape != (50, 16):
        raise ValueError("expected sixteen non-overlapping 32x32 patches per image")

    # Lock the profile derivation to the published per-image metrics.
    global_p95 = np.percentile(global_patches, 95, axis=1)
    ours_p95 = np.percentile(ours_patches, 95, axis=1)
    global_local = 10.0 * np.log10(1.0 / np.sort(global_patches, axis=1)[:, -4:].mean(axis=1))
    ours_local = 10.0 * np.log10(1.0 / np.sort(ours_patches, axis=1)[:, -4:].mean(axis=1))
    np.testing.assert_allclose(global_p95, result["global_p95"], rtol=0.0, atol=5e-10)
    np.testing.assert_allclose(ours_p95, result["ours_p95"], rtol=0.0, atol=5e-10)
    np.testing.assert_allclose(global_local, result["global_local_psnr"], rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(ours_local, result["ours_local_psnr"], rtol=0.0, atol=1e-5)

    result["profile_percentile"] = np.arange(1, 17, dtype=float) / 16.0 * 100.0
    result["global_profile"] = np.sort(global_patches, axis=1).mean(axis=0)
    result["ours_profile"] = np.sort(ours_patches, axis=1).mean(axis=0)
    return result


def style_quant_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#5F6B76")
    axis.spines["bottom"].set_color("#5F6B76")
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.grid(color="#C7D0D9", linewidth=0.45, alpha=0.45)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=8.0, width=0.65, length=3, pad=2)


def scatter_identity(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    favorable_side: str,
) -> None:
    low = min(float(x.min()), float(y.min()))
    high = max(float(x.max()), float(y.max()))
    padding = 0.07 * (high - low)
    low -= padding
    high += padding
    axis.plot([low, high], [low, high], color=IDENTITY_COLOR, linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
    axis.scatter(x, y, s=24, color=OURS_COLOR, edgecolor="white", linewidth=0.55, alpha=0.88, zorder=3)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title, loc="left", fontsize=9.3, fontweight="semibold", pad=5)
    axis.set_xlabel(xlabel, fontsize=8.2, labelpad=3)
    axis.set_ylabel(ylabel, fontsize=8.2, labelpad=3)
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    style_quant_axis(axis)
    count = int(np.sum(y < x)) if favorable_side == "below" else int(np.sum(y > x))
    axis.text(
        0.05,
        0.93,
        f"{count}/50 {favorable_side} identity",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color=FAVORABLE,
        fontweight="semibold",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.90},
    )


def render_qualitative(
    figure: plt.Figure,
    spec: Any,
    display: dict[str, torch.Tensor],
    psnr: dict[str, float],
    roi_mae: dict[str, float],
) -> None:
    grid = spec.subgridspec(
        3,
        6,
        width_ratios=(1, 1, 1, 1, 1, 0.07),
        wspace=0.055,
        hspace=0.11,
    )
    x, y, width, height = (value * 4 for value in ROI_SOURCE)
    reference = display["original"]
    residual_maps = {
        key: (display[key] - reference).abs().mean(dim=0).numpy() * ERROR_AMPLIFICATION
        for key in METHOD_KEYS
    }
    residual_vmax = max(
        float(residual_maps[key][y:y + height, x:x + width].max())
        for key in METHOD_KEYS[1:]
    )
    norm = Normalize(vmin=0.0, vmax=residual_vmax)
    column_titles = {
        "original": "Original\nreference",
        "hidden": f"HiDDeN-64\n{psnr['hidden']:.2f} dB",
        "maskwm": f"MaskWM-D_64\n{psnr['maskwm']:.2f} dB",
        "global": f"MBRS Global\n{psnr['global']:.2f} dB",
        "ours": f"Ours\n{psnr['ours']:.2f} dB",
    }
    title_colors = {
        "original": INK,
        "hidden": MUTED,
        "maskwm": MUTED,
        "global": GLOBAL_COLOR,
        "ours": OURS_COLOR,
    }
    row_axes: list[list[plt.Axes]] = []
    for row in range(3):
        axes = []
        for column, key in enumerate(METHOD_KEYS):
            axis = figure.add_subplot(grid[row, column])
            if row == 0:
                axis.imshow(tensor_to_array(display[key]), interpolation="nearest")
                axis.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=ROI_COLOR, linewidth=1.45))
                axis.set_title(
                    column_titles[key],
                    fontsize=8.1,
                    color=title_colors[key],
                    fontweight="semibold",
                    pad=4.5,
                    linespacing=1.05,
                )
            elif row == 1:
                crop = tensor_to_array(display[key])
                axis.imshow(crop[y:y + height, x:x + width], interpolation="nearest")
            else:
                crop = residual_maps[key][y:y + height, x:x + width]
                axis.imshow(crop, cmap="magma", norm=norm, interpolation="nearest")
                if key == "original":
                    label = "reference"
                else:
                    label = f"ROI MAE {roi_mae[key] * 1e3:.2f}×10⁻³"
                axis.text(
                    0.5,
                    -0.055,
                    label,
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    color=MUTED if key not in ("global", "ours") else title_colors[key],
                )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            axes.append(axis)
        row_axes.append(axes)

    color_axis = figure.add_subplot(grid[2, 5])
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap="magma")
    colorbar = figure.colorbar(scalar, cax=color_axis)
    colorbar.ax.tick_params(labelsize=6.2, colors=MUTED, length=2, pad=1)
    colorbar.ax.set_title("×10", fontsize=6.5, color=MUTED, pad=2)

    figure.canvas.draw()
    row_labels = ("Full image", "Shared ROI", "Local error")
    for label, axes in zip(row_labels, row_axes):
        position = axes[0].get_position()
        figure.text(
            position.x0 - 0.010,
            position.y0 + position.height / 2.0,
            label,
            ha="right",
            va="center",
            fontsize=7.5,
            color=INK if label == "Full image" else MUTED,
            fontweight="semibold",
        )


def render_figure(
    display: dict[str, torch.Tensor],
    psnr: dict[str, float],
    roi_mae: dict[str, float],
    data: dict[str, np.ndarray],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    figure = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), facecolor="white")
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=(3.15, 2.05),
        left=0.084,
        right=0.965,
        top=0.895,
        bottom=0.085,
        hspace=0.29,
    )
    render_qualitative(figure, outer[0], display, psnr, roi_mae)

    figure.text(
        0.084,
        0.982,
        "Qualitative local-distortion comparison",
        fontsize=11.0,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    figure.text(
        0.084,
        0.954,
        "Fixed validation sample 0814 · frozen operating points · mean PSNR on common 512×512 display canvas",
        fontsize=7.3,
        color=MUTED,
        ha="left",
        va="top",
    )
    figure.text(
        0.965,
        0.982,
        "red outline = shared 32×32 ROI",
        fontsize=7.0,
        color=ROI_COLOR,
        ha="right",
        va="top",
    )

    bottom = outer[1].subgridspec(1, 3, wspace=0.48)
    p95_axis = figure.add_subplot(bottom[0, 0])
    local_axis = figure.add_subplot(bottom[0, 1])
    profile_axis = figure.add_subplot(bottom[0, 2])

    scatter_identity(
        p95_axis,
        data["global_p95"] * 1e4,
        data["ours_p95"] * 1e4,
        "(a) P95 patch MSE",
        "Global P95 (×10⁻⁴)",
        "Ours P95 (×10⁻⁴)",
        "below",
    )
    scatter_identity(
        local_axis,
        data["global_local_psnr"],
        data["ours_local_psnr"],
        "(b) Top-25 local PSNR",
        "Global local PSNR (dB)",
        "Ours local PSNR (dB)",
        "above",
    )

    percentile = data["profile_percentile"]
    global_profile = data["global_profile"] * 1e4
    ours_profile = data["ours_profile"] * 1e4
    profile_axis.axvspan(75.0, 100.0, color=TAIL_COLOR, alpha=0.70, linewidth=0, zorder=0)
    profile_axis.plot(
        percentile,
        global_profile,
        color=GLOBAL_COLOR,
        marker="o",
        markersize=3.3,
        linewidth=1.45,
        label="Global",
        zorder=3,
    )
    profile_axis.plot(
        percentile,
        ours_profile,
        color=OURS_COLOR,
        marker="s",
        markersize=3.3,
        linewidth=1.45,
        linestyle=(0, (4, 2)),
        label="Ours",
        zorder=3,
    )
    profile_axis.fill_between(percentile, ours_profile, global_profile, color=FAVORABLE, alpha=0.08, zorder=1)
    profile_axis.set_title("(c) Local-tail profile", loc="left", fontsize=9.3, fontweight="semibold", pad=5)
    profile_axis.set_xlabel("Sorted patch-error percentile", fontsize=8.2, labelpad=3)
    profile_axis.set_ylabel("Mean patch MSE (×10⁻⁴)", fontsize=8.2, labelpad=3)
    profile_axis.set_xlim(5.0, 101.0)
    profile_axis.set_xticks([6.25, 25, 50, 75, 100], labels=["6", "25", "50", "75", "100"])
    profile_axis.yaxis.set_major_locator(MaxNLocator(5))
    style_quant_axis(profile_axis)
    profile_axis.legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=2.0, ncol=1)
    profile_axis.text(
        87.5,
        0.96,
        "worst 25%",
        transform=profile_axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.0,
        color="#8A6D28",
        fontweight="semibold",
    )
    tail_reduction = 1.0 - float(data["ours_profile"][-4:].mean() / data["global_profile"][-4:].mean())
    profile_axis.text(
        87.5,
        0.075,
        f"tail mean\n−{tail_reduction * 100:.1f}%",
        transform=profile_axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=FAVORABLE,
        fontweight="semibold",
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_STEM.with_suffix(".png"), dpi=PNG_DPI, facecolor="white")
    figure.savefig(FIGURE_STEM.with_suffix(".pdf"), facecolor="white")
    figure.savefig(FIGURE_STEM.with_suffix(".svg"), facecolor="white")
    plt.close(figure)


def write_data_exports(data: dict[str, np.ndarray]) -> None:
    with PAIRED_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_index", "global_p95_patch_mse", "ours_p95_patch_mse", "global_top25_local_psnr", "ours_top25_local_psnr"))
        for index in range(50):
            writer.writerow(
                (
                    int(data["image_index"][index]),
                    f"{data['global_p95'][index]:.17g}",
                    f"{data['ours_p95'][index]:.17g}",
                    f"{data['global_local_psnr'][index]:.17g}",
                    f"{data['ours_local_psnr'][index]:.17g}",
                )
            )
    with PROFILE_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("rank", "percentile", "global_mean_patch_mse", "ours_mean_patch_mse"))
        for rank in range(16):
            writer.writerow(
                (
                    rank + 1,
                    f"{data['profile_percentile'][rank]:.6f}",
                    f"{data['global_profile'][rank]:.17g}",
                    f"{data['ours_profile'][rank]:.17g}",
                )
            )


def write_caption_and_notes(
    psnr: dict[str, float],
    roi_mae: dict[str, float],
    data: dict[str, np.ndarray],
) -> None:
    p95_count = int(np.sum(data["ours_p95"] < data["global_p95"]))
    local_count = int(np.sum(data["ours_local_psnr"] > data["global_local_psnr"]))
    tail_global = float(data["global_profile"][-4:].mean())
    tail_ours = float(data["ours_profile"][-4:].mean())
    tail_reduction = 1.0 - tail_ours / tail_global
    caption = (
        "Fig. 2. Qualitative and paired local-tail distortion analysis. Top: a fixed validation sample at the frozen operating points; "
        "HiDDeN-64 and MaskWM-D_64 are contextual external references under their released/retrained protocols, not strictly matched training comparisons. "
        "Red boxes mark one shared 32×32 ROI, and local-error maps show mean absolute RGB error amplified ×10 with a common scale. "
        "Bottom: natural 128×128 project-test evidence over 50 paired images. (a) Per-image P95 patch MSE; lower is better and all 50 points fall below identity. "
        "(b) Top-25 local PSNR; higher is better and all 50 points lie above identity. (c) Mean rank-ordered patch MSE over sixteen non-overlapping 32×32 patches per image; "
        "the shaded region denotes the worst quartile. Ours denotes Hard Local-Tail + global OKLab."
    )
    CAPTION_PATH.write_text(caption + "\n", encoding="utf-8")

    notes = f"""# Figure 2 generation notes

## Figure design

- Layout: upper qualitative section plus three lower quantitative panels.
- Final size: `{FIGURE_WIDTH_IN:.2f} in × {FIGURE_HEIGHT_IN:.2f} in`.
- PNG: `{round(FIGURE_WIDTH_IN * PNG_DPI)} × {round(FIGURE_HEIGHT_IN * PNG_DPI)} px` at `{PNG_DPI}` DPI.
- PDF/SVG retain vector text and quantitative curves; qualitative image tiles are embedded raster evidence.

## Upper qualitative section

- Selection source: `reports/final_figure2_selection.md`.
- Fixed validation sample: ID `{SAMPLE_ID}`, index `{SAMPLE_INDEX}`.
- Shared source-space ROI: `(x={ROI_SOURCE[0]}, y={ROI_SOURCE[1]}, width={ROI_SOURCE[2]}, height={ROI_SOURCE[3]})` on 128×128 images.
- Columns: Original, HiDDeN-64, MaskWM-D_64, MBRS crop-trained Global, Ours.
- Mean display PSNR on the common 512×512 canvas: HiDDeN-64 `{psnr['hidden']:.6f}` dB; MaskWM-D_64 `{psnr['maskwm']:.6f}` dB; Global `{psnr['global']:.6f}` dB; Ours `{psnr['ours']:.6f}` dB.
- ROI mean absolute RGB error: HiDDeN-64 `{roi_mae['hidden']:.9f}`; MaskWM-D_64 `{roi_mae['maskwm']:.9f}`; Global `{roi_mae['global']:.9f}`; Ours `{roi_mae['ours']:.9f}`.
- Local-error maps use one linear color range across all method columns and are amplified ×10 only for visualization.
- External methods are contextual references under different training/preprocessing protocols; the panel does not claim strict external superiority.

## Lower quantitative section

- Evidence space: natural frozen 128×128 project-test set, exactly 50 paired image indices.
- Global source: `paper_extension_study/continuation_seed/global/seed17/project_test_per_image.csv`.
- Ours source: `paper_extension_study/continuation_seed/ours/seed17/project_test_per_image.csv`.
- P95 paired scatter: `{p95_count}/50` points below identity (lower is better).
- Top-25 local-PSNR paired scatter: `{local_count}/50` points above identity (higher is better).
- Local-tail profile: each image contributes sixteen non-overlapping 32×32 patch MSE values; values are sorted within image and averaged by rank over 50 images.
- Worst-quartile mean patch MSE: Global `{tail_global:.9e}`; Ours `{tail_ours:.9e}`; relative reduction `{tail_reduction * 100:.3f}%`.
- The worst-quartile band spans the 75th–100th percentile. No smoothing, regression, pooling across images before ranking, or sample filtering is used.

## Safeguards

- No model was retrained.
- No checkpoint, alpha, hyperparameter, image, or ROI was newly selected from project-test results.
- The Prism manuscript and `paper_bundle/` were not modified.
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")


def main() -> None:
    torch.set_num_threads(4)
    required = (
        VALIDATION_MANIFEST,
        HIDDEN_CACHE,
        MASKWM_PROVENANCE,
        GLOBAL_VALIDATION_OUTPUT,
        OURS_VALIDATION_OUTPUT,
        PROJECT_TEST_MANIFEST,
        GLOBAL_PROJECT_TEST_OUTPUT,
        OURS_PROJECT_TEST_OUTPUT,
        GLOBAL_PER_IMAGE,
        OURS_PER_IMAGE,
        SELECTION_REPORT,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    display, psnr, roi_mae = load_qualitative_data()
    data = load_quantitative_data()
    write_data_exports(data)
    write_caption_and_notes(psnr, roi_mae, data)
    render_figure(display, psnr, roi_mae, data)
    print(f"wrote {FIGURE_STEM.with_suffix('.pdf')}")
    print(f"wrote {FIGURE_STEM.with_suffix('.png')}")
    print(f"wrote {FIGURE_STEM.with_suffix('.svg')}")
    print(f"wrote {CAPTION_PATH}")
    print(f"wrote {NOTES_PATH}")
    print(f"wrote {PAIRED_CSV}")
    print(f"wrote {PROFILE_CSV}")


if __name__ == "__main__":
    main()
