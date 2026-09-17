#!/usr/bin/env python3
"""Build the paper Fig. 2 from the frozen external-qualitative montage.

The raster panels are lossless crops of the already-rendered qualitative
artifact.  No model is loaded, no inference is run, and no image adjustment is
applied.  Titles, PSNR labels, row labels, and the footer are newly typeset so
that text remains vectorized in the PDF.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIR = Path("/mnt/wmcontent/GLX/icassp/MBRS/visualizations/external_qualitative")
FROZEN_MONTAGE = FROZEN_DIR / "qualitative_5x6_main.png"
FROZEN_MONTAGE_SHA256 = "4a39a0c4e8559a403b822bb37db11014a82f3deb3d56ef97b7cf5a068aa9ff3e"
PSNR_CSV = FROZEN_DIR / "per_image_psnr.csv"
OUTPUT_STEM = PROJECT_ROOT / "paper/figures/fig2_external_qualitative"

SAMPLES = (0, 10, 20)
COLUMNS = (
    "Original",
    "HiDDeN-64",
    "MaskWM-D_64",
    "MBRS Global",
    "Hard Top10",
    "Hard Top10 + OKLab",
)

# Pixel-exact panel bounds in the frozen 3960 x 3300 montage.  PIL boxes use
# exclusive right/lower edges.  These bounds omit the old title, labels, and
# whitespace while retaining each actual-RGB panel and its shared ROI outline.
X_BOUNDS = ((86, 602), (741, 1257), (1395, 1911), (2050, 2566), (2704, 3220), (3359, 3875))
Y_BOUNDS = ((285, 800), (868, 1383), (1451, 1966))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_psnr() -> dict[int, dict[str, float]]:
    with PSNR_CSV.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        int(row["validation_index"]): {
            method: float(row[f"{method}_psnr_db"])
            for method in COLUMNS
            if method != "Original"
        }
        for row in rows
    }


def main() -> None:
    actual_hash = sha256(FROZEN_MONTAGE)
    if actual_hash != FROZEN_MONTAGE_SHA256:
        raise RuntimeError(
            "Frozen qualitative montage changed: "
            f"expected {FROZEN_MONTAGE_SHA256}, found {actual_hash}"
        )

    montage = Image.open(FROZEN_MONTAGE).convert("RGB")
    if montage.size != (3960, 3300):
        raise RuntimeError(f"Unexpected montage size: {montage.size}")
    psnr = load_psnr()

    # ICASSP-style two-column span.  Explicit placement gives compact, stable
    # gutters and preserves the source panel aspect ratio (516:515).
    figure_width = 7.16
    figure_height = 4.42
    left, right = 0.31, 0.035
    bottom, top = 0.24, 0.25
    col_gap, row_gap = 0.025, 0.20
    image_width = (figure_width - left - right - 5 * col_gap) / 6
    image_height = image_width * 515 / 516

    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    for row, (sample, y_bounds) in enumerate(zip(SAMPLES, Y_BOUNDS)):
        y = figure_height - top - (row + 1) * image_height - row * row_gap
        fig.text(
            0.055 / figure_width,
            (y + image_height / 2) / figure_height,
            f"Sample {sample}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=6.5,
        )
        for col, (method, x_bounds) in enumerate(zip(COLUMNS, X_BOUNDS)):
            x = left + col * (image_width + col_gap)
            panel = montage.crop((x_bounds[0], y_bounds[0], x_bounds[1], y_bounds[1]))
            if panel.size != (516, 515):
                raise RuntimeError(f"Unexpected panel size for {sample}/{method}: {panel.size}")

            ax = fig.add_axes(
                [x / figure_width, y / figure_height, image_width / figure_width, image_height / figure_height]
            )
            ax.imshow(panel, interpolation="none", aspect="equal")
            ax.set_axis_off()

            if row == 0:
                fig.text(
                    (x + image_width / 2) / figure_width,
                    (y + image_height + 0.055) / figure_height,
                    method,
                    ha="center",
                    va="bottom",
                    fontsize=7.6,
                )
            if method != "Original":
                fig.text(
                    (x + image_width / 2) / figure_width,
                    (y - 0.032) / figure_height,
                    f"{psnr[sample][method]:.2f} dB",
                    ha="center",
                    va="top",
                    fontsize=6.6,
                )

    fig.text(
        0.5,
        0.025,
        "Same host per row; fixed shared ROIs; no residual amplification.",
        ha="center",
        va="bottom",
        fontsize=6.7,
    )

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=350, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=350, facecolor="white")
    plt.close(fig)

    print(OUTPUT_STEM.with_suffix(".pdf"))
    print(OUTPUT_STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
