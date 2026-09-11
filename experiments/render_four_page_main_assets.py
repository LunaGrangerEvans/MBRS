#!/usr/bin/env python3
"""Render the compact two-figure main-paper asset set.

The renderer intentionally reads only frozen formal/validation artifacts.  It
does not train, re-evaluate checkpoints, or write to the host-mounted output
tree.  Outputs are small paper assets under ``paper/figures``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab


PROJECT = Path(__file__).resolve().parents[1]
PAPER = PROJECT / "paper"
OUT = PAPER / "figures"
MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
MANIFEST = MOUNT / "reports/content_selector/validation_manifest.pt"
VALIDATION_ROOT = (
    MOUNT
    / "reports/crop_global_oklab/"
    / "validation_seed17_crop_hard16_stride8_top10_global_oklab_g25"
)
SAMPLES = [0, 10, 20, 30, 40]

METHOD_FILES = {
    "Global baseline": VALIDATION_ROOT / "controlled_seed17_global_continuation_outputs.pt",
    "Hard Top10": VALIDATION_ROOT / "controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_outputs.pt",
    "Hard Top10 + global OKLab": VALIDATION_ROOT / "seed17_crop_hard16_stride8_top10_global_oklab_g25_outputs.pt",
}

COLORS = {
    "ink": "#1D2733",
    "muted": "#5D6B79",
    "line": "#7D8A96",
    "global": "#2D6A9F",
    "hard": "#C15B32",
    "oklab": "#3B826E",
    "message": "#6C5A9C",
    "soft_blue": "#EAF2F8",
    "soft_orange": "#FBEDE4",
    "soft_green": "#EAF5F0",
    "soft_purple": "#F0ECF8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor to clipped HWC float RGB."""

    return ((tensor.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def psnr(reference: np.ndarray, output: np.ndarray) -> float:
    mse = max(float(np.square(reference - output).mean()), 1e-12)
    return float(10.0 * np.log10(1.0 / mse))


def load_encoded(path: Path) -> list[np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return [rgb(item) for item in payload["encoded"]]


def ciede_map(reference: np.ndarray, output: np.ndarray) -> np.ndarray:
    return deltaE_ciede2000(rgb2lab(reference), rgb2lab(output)).astype(np.float32)


def fixed_rois(originals: list[np.ndarray], incumbent: list[np.ndarray]) -> dict[int, tuple[int, int, int, int]]:
    """Choose one shared native32 ROI per row using the incumbent's color tail."""

    rois: dict[int, tuple[int, int, int, int]] = {}
    for index in SAMPLES:
        error = ciede_map(originals[index], incumbent[index])
        y, x = np.unravel_index(int(np.argmax(error)), error.shape)
        rois[index] = (int((y // 32) * 32), int((x // 32) * 32), 32, 32)
    return rois


def add_box(ax, center, width, height, text, face, edge=COLORS["line"], fontsize=10, weight="normal"):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", color=COLORS["ink"], fontsize=fontsize, weight=weight, linespacing=1.25)


def add_arrow(ax, start, end, color=COLORS["line"], style="-|>", lw=1.5, linestyle="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=lw,
            linestyle=linestyle,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def render_method_figure() -> Path:
    """Render Figure 1: the global/local/color objective structure."""

    fig, ax = plt.subplots(figsize=(14, 5.7))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 5.7)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(0.25, 5.38, "MBRS crop-trained encoder with complementary image supervision", fontsize=15, weight="bold", color=COLORS["ink"])
    ax.text(0.25, 5.08, "The same watermarked image is constrained at the mean, at its worst local tail, and optionally in a perceptual color space.", fontsize=9.5, color=COLORS["muted"])

    # Main forward path.
    add_box(ax, (1.35, 4.08), 1.8, 0.84, "Cover image x\n128×128 RGB", COLORS["soft_blue"], edge=COLORS["global"], weight="bold")
    add_box(ax, (1.35, 2.86), 1.8, 0.84, "Message m\n64 bits", COLORS["soft_purple"], edge=COLORS["message"], weight="bold")
    add_box(ax, (3.65, 3.48), 1.8, 1.05, "Crop-trained\nMBRS encoder", "#F5F7F9", edge=COLORS["ink"], weight="bold")
    add_box(ax, (6.05, 3.48), 1.95, 1.05, "Watermarked image\nŷ = E(x, m)", COLORS["soft_orange"], edge=COLORS["hard"], weight="bold")
    add_arrow(ax, (2.28, 4.08), (2.72, 3.72), color=COLORS["global"])
    add_arrow(ax, (2.28, 2.86), (2.72, 3.24), color=COLORS["message"])
    add_arrow(ax, (4.56, 3.48), (5.05, 3.48), color=COLORS["ink"])

    # Crop/message branch is deliberately secondary to the three image branches.
    add_box(ax, (8.45, 4.23), 1.75, 0.7, "Random crop mask\nM ⊙ ŷ", COLORS["soft_purple"], edge=COLORS["message"], fontsize=9.5)
    add_box(ax, (10.65, 4.23), 1.75, 0.7, "Decoder\nmessage logits", COLORS["soft_purple"], edge=COLORS["message"], fontsize=9.5)
    add_box(ax, (12.75, 4.23), 1.65, 0.7, "L_message", COLORS["soft_purple"], edge=COLORS["message"], fontsize=9.5, weight="bold")
    add_arrow(ax, (7.03, 3.9), (7.55, 4.1), color=COLORS["message"])
    add_arrow(ax, (9.35, 4.23), (9.72, 4.23), color=COLORS["message"])
    add_arrow(ax, (11.58, 4.23), (11.9, 4.23), color=COLORS["message"])

    # Reference bus for all image losses.
    ax.plot([1.35, 1.35], [3.66, 1.58], color=COLORS["global"], linewidth=1.2, linestyle="--")
    ax.text(0.35, 1.42, "reference x", color=COLORS["global"], fontsize=9, ha="left", va="center")
    ax.plot([1.35, 11.9], [1.58, 1.58], color=COLORS["global"], linewidth=1.0, linestyle="--", alpha=0.8)
    ax.plot([6.05, 6.05], [2.95, 1.58], color=COLORS["hard"], linewidth=1.2)

    # Three supervision cards.
    add_box(ax, (3.55, 1.08), 2.7, 0.9, "Global RGB MSE\nmean over all pixels\ncontrols average distortion", COLORS["soft_blue"], edge=COLORS["global"], fontsize=9.5, weight="bold")
    add_box(ax, (7.15, 1.08), 3.25, 0.9, "Hard local-tail MSE\nPatch16 / stride8 → 225 candidates\nTop10% = 23 worst patches", COLORS["soft_orange"], edge=COLORS["hard"], fontsize=9.2, weight="bold")
    add_box(ax, (11.1, 1.08), 2.65, 0.9, "Optional OKLab branch\ncolor/lightness residuals\nvalidation extension", COLORS["soft_green"], edge=COLORS["oklab"], fontsize=9.3, weight="bold")
    add_arrow(ax, (2.0, 1.58), (2.2, 1.46), color=COLORS["global"], linestyle="--")
    add_arrow(ax, (6.05, 2.95), (7.15, 1.56), color=COLORS["hard"])
    add_arrow(ax, (7.0, 1.58), (9.75, 1.58), color=COLORS["hard"], linestyle="--")
    add_arrow(ax, (9.75, 1.58), (10.05, 1.46), color=COLORS["oklab"], linestyle="--")

    # Combined objective footer.
    add_box(ax, (7.0, 0.15), 7.0, 0.58, r"$\mathcal{L}=\lambda_m\mathcal{L}_{message}+0.5\mathcal{L}_{RGB}+0.5\mathcal{L}_{Hard}+\lambda_{OKLab}\mathcal{L}_{OKLab}$", "#F5F7F9", edge=COLORS["ink"], fontsize=10, weight="bold")
    ax.text(0.25, 0.18, "Global controls the mean", fontsize=9, color=COLORS["global"], weight="bold", va="center")
    ax.text(0.25, -0.03, "Hard Top10 controls the worst local tail", fontsize=9, color=COLORS["hard"], weight="bold", va="center")
    ax.text(0.25, -0.24, "OKLab controls color artifacts", fontsize=9, color=COLORS["oklab"], weight="bold", va="center")
    ax.text(13.75, 1.08, "image branches\ntraining only", fontsize=8.5, color=COLORS["muted"], ha="right", va="center")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    path = OUT / "fig01_method_framework"
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def render_qualitative_figure() -> Path:
    """Render Figure 2: five fixed validation samples plus shared local zooms."""

    manifest = torch.load(MANIFEST, map_location="cpu", weights_only=False)
    originals = [rgb(item) for item in manifest["images"]]
    encoded = {name: load_encoded(path) for name, path in METHOD_FILES.items()}
    rois = fixed_rois(originals, encoded["Hard Top10"])

    method_order = list(METHOD_FILES)
    fig, axes = plt.subplots(len(SAMPLES), 8, figsize=(15.2, 10.6), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.985, "Qualitative RGB comparison on fixed validation samples", ha="center", va="top", fontsize=15, weight="bold", color=COLORS["ink"])
    fig.text(0.5, 0.962, "Each row uses the same image/message. Right-hand panels show one shared 32×32 crop selected from the incumbent Hard Top10 CIEDE2000 tail.", ha="center", va="top", fontsize=9.2, color=COLORS["muted"])

    # Two image columns per method: full frame and the shared zoom.
    group_centers = [0.09, 0.34, 0.59, 0.84]
    group_titles = ["Original", "Global baseline", "Hard Top10", "Hard Top10 + global OKLab\n(validation, λ=0.05915)"]
    for center, title in zip(group_centers, group_titles):
        fig.text(center, 0.932, title, ha="center", va="center", fontsize=10, weight="bold", color=COLORS["ink"])
    for col in range(8):
        fig.text((col + 0.5) / 8.0, 0.914, "full" if col % 2 == 0 else "shared zoom", ha="center", va="center", fontsize=7.5, color=COLORS["muted"])

    for row, index in enumerate(SAMPLES):
        y, x, h, w = rois[index]
        all_images = [originals[index]] + [encoded[name][index] for name in method_order]
        for group, image in enumerate(all_images):
            full_ax = axes[row, group * 2]
            zoom_ax = axes[row, group * 2 + 1]
            full_ax.imshow(image, interpolation="nearest")
            full_ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#E04F3F", linewidth=1.2))
            full_ax.axis("off")
            zoom_ax.imshow(image[y : y + h, x : x + w], interpolation="nearest")
            zoom_ax.axis("off")
            if group == 0:
                full_ax.set_ylabel(f"sample {index:02d}", rotation=90, fontsize=8.5, color=COLORS["ink"], labelpad=8)
        # Put compact per-row PSNR values under the full-frame method panels.
        for group, image in enumerate(all_images[1:], start=1):
            axes[row, group * 2].text(
                0.5,
                -0.035,
                f"{psnr(originals[index], image):.2f} dB",
                transform=axes[row, group * 2].transAxes,
                ha="center",
                va="top",
                fontsize=7.3,
                color=COLORS["muted"],
            )

    fig.text(0.5, 0.012, "The OKLab column is a validation-only color extension; no residual amplification, sharpening, or per-method ROI selection is used.", ha="center", va="bottom", fontsize=8.5, color=COLORS["muted"])
    fig.subplots_adjust(left=0.035, right=0.99, top=0.895, bottom=0.045, wspace=0.08, hspace=0.26)
    path = OUT / "fig02_qualitative_comparison"
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_provenance(method_path: Path, qualitative_path: Path) -> None:
    files = [MANIFEST, *METHOD_FILES.values(), PAPER / "main_table.csv", PAPER / "../reports/crop_global_oklab_validation_candidates.csv", PAPER / "../reports/local_chroma_tail_summary.csv", Path(__file__)]
    provenance = {
        "samples": SAMPLES,
        "method_figure": {"png": str(method_path.with_suffix(".png")), "pdf": str(method_path.with_suffix(".pdf"))},
        "qualitative_figure": {"png": str(qualitative_path.with_suffix(".png")), "pdf": str(qualitative_path.with_suffix(".pdf"))},
        "qualitative_roi_rule": "For each sample, select the native 32x32 block containing the maximum CIEDE2000 pixel of incumbent Hard Top10; share that ROI across all four columns.",
        "oklab_weight": 0.059149764,
        "source_sha256": {str(path): sha256(path) for path in files if path.exists()},
    }
    outputs = [method_path.with_suffix(".png"), method_path.with_suffix(".pdf"), qualitative_path.with_suffix(".png"), qualitative_path.with_suffix(".pdf")]
    provenance["output_sha256"] = {str(path): sha256(path) for path in outputs}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    method_path = render_method_figure()
    qualitative_path = render_qualitative_figure()
    write_provenance(method_path, qualitative_path)
    print(method_path.with_suffix(".png"))
    print(qualitative_path.with_suffix(".png"))


if __name__ == "__main__":
    main()
