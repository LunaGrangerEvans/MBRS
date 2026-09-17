#!/usr/bin/env python3
"""Render the frozen three-stage MBRS final-method framework."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from paper.render_fig1_method import arrow, box, label, routed_arrow  # noqa: E402


OUTPUT = PROJECT / "paper/figures"
REPORT = PROJECT / "reports/fig1_final_method_check.md"
CONFIG = Path("/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/config.resolved.json")

BLUE = "#dfeaf4"
BLUE_EDGE = "#41627d"
GRAY = "#eef1f3"
GRAY_EDGE = "#59636b"
ORANGE = "#f8e4cc"
ORANGE_EDGE = "#b96922"
GREEN = "#edf5ee"
GREEN_EDGE = "#4f845d"
INK = "#20262b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def render() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.16, 4.62), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.55)
    ax.axis("off")

    label(ax, (0.16, 7.36), "FROZEN FINAL METHOD", ha="left", fontsize=8.8, color=BLUE_EDGE, weight="bold")
    label(ax, (11.84, 7.36), "SHARED MBRS ENCODER / DECODER", ha="right", fontsize=7.4, color=GRAY_EDGE, weight="bold")

    nodes = (
        (0.93, "Cover $x$ +\n64-bit message $m$", 1.55, BLUE, BLUE_EDGE),
        (2.52, "MBRS\nEncoder", 1.00, BLUE, BLUE_EDGE),
        (4.08, "Watermarked\nimage $\\hat{x}$", 1.28, BLUE, BLUE_EDGE),
        (5.83, "RandomCrop\nchannel\n(training)", 1.45, GRAY, GRAY_EDGE),
        (7.54, "MBRS\nDecoder", 1.00, BLUE, BLUE_EDGE),
        (9.36, "Message\nrecovery", 1.28, BLUE, BLUE_EDGE),
    )
    for x, text, width, face, edge in nodes:
        box(ax, (x, 6.67), width, 0.76, text, face, edge, fontsize=6.1)
    for start, end in ((1.70, 2.02), (3.02, 3.44), (4.72, 5.10), (6.56, 7.04), (8.07, 8.72)):
        arrow(ax, (start, 6.67), (end, 6.67))
    label(ax, (5.83, 6.08), "crop channel is training-time only", fontsize=6.7, color=GRAY_EDGE, weight="bold")

    label(ax, (4.08, 5.50), "clean $\\hat{x}$ and host $x$ feed three training-loss branches", fontsize=7.4, color=ORANGE_EDGE, weight="bold")
    arrow(ax, (4.08, 6.25), (4.08, 5.68), color=ORANGE_EDGE, lw=1.2)
    ax.plot([4.08, 4.08], [5.68, 4.93], color=ORANGE_EDGE, linewidth=1.2)
    ax.plot([1.85, 10.24], [4.93, 4.93], color=ORANGE_EDGE, linewidth=1.2)
    ax.add_patch(plt.Circle((4.08, 4.93), 0.07, facecolor=ORANGE_EDGE, edgecolor=ORANGE_EDGE))
    for x in (2.05, 6.05, 10.05):
        arrow(ax, (x, 4.93), (x, 4.46), color=ORANGE_EDGE, lw=1.15)

    box(ax, (2.05, 3.66), 3.35, 1.22, "A · Average distortion control\nGlobal RGB reconstruction\n$L_{\\mathrm{RGB}} = \\mathrm{MSE}(\\hat{x},x)$", ORANGE, ORANGE_EDGE, fontsize=6.8)
    box(ax, (6.05, 3.66), 3.35, 1.22, "B · Upper-tail local distortion control\nP16 / S8 · 225 candidates\nHard Top-10% = 23 highest-error patches", ORANGE, ORANGE_EDGE, fontsize=6.55)
    box(ax, (10.05, 3.66), 3.35, 1.22, "C · Color-aware fidelity control\nGlobal linear-sRGB OKLab distance\n$\\lambda_{\\mathrm{OK}}L_{\\mathrm{OKLab}}$", GREEN, GREEN_EDGE, fontsize=6.55)

    label(ax, (2.05, 2.74), "$L_{\\mathrm{tail}} = \\mathrm{mean}$ selected patch MSE", fontsize=6.7, color=ORANGE_EDGE)
    label(ax, (10.05, 2.74), "$\\lambda_{\\mathrm{OK}}=0.059149764$", fontsize=6.7, color=GREEN_EDGE, weight="bold")
    box(ax, (6.05, 1.37), 7.90, 1.25, "Ours · final objective\n$L = w_{msg}L_{msg} + w_gL_{RGB} + w_tL_{tail} + \\lambda_{OK}L_{OKLab}$\nGlobal average control  →  hard local-tail control  →  color-aware refinement\ntraining losses only  ·  no additional inference module", GRAY, GRAY_EDGE, fontsize=6.55)
    routed_arrow(ax, ((2.05, 3.05), (2.05, 2.48), (4.06, 1.91)), color=ORANGE_EDGE, lw=1.05)
    routed_arrow(ax, ((6.05, 3.05), (6.05, 1.97)), color=ORANGE_EDGE, lw=1.05)
    routed_arrow(ax, ((10.05, 3.05), (10.05, 2.48), (8.04, 1.91)), color=GREEN_EDGE, lw=1.05)
    label(ax, (6.05, 0.48), "Local-tail and OKLab are training losses only; inference remains the adopted MBRS encoder → crop channel → decoder path.", fontsize=6.55, color=GRAY_EDGE)

    pdf = OUTPUT / "figure1_final_method.pdf"
    png = OUTPUT / "figure1_final_method.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.055, facecolor="white")
    fig.savefig(png, dpi=360, bbox_inches="tight", pad_inches=0.055, facecolor="white")
    plt.close(fig)
    report = "\n".join([
        "# Final Figure 1 method check",
        "",
        "The formal frozen g25 evaluation passed, so the framework now defines `Ours` as Hard Local-Tail plus global OKLab regularization.",
        "",
        f"- Frozen config: `{CONFIG}`; SHA-256 `{sha256(CONFIG)}`.",
        f"- Output PDF: `{pdf}`; SHA-256 `{sha256(pdf)}`.",
        f"- Output PNG: `{png}`; SHA-256 `{sha256(png)}`.",
        "- Branch A: Global RGB — Average distortion control.",
        "- Branch B: Hard Local-Tail — Upper-tail local distortion control; P16/S8, 225 candidates, hard Top-10% (23 patches).",
        "- Branch C: OKLab — Color-aware fidelity control; frozen lambda 0.059149764.",
        "- Objective uses symbolic weights: `L = w_msg L_msg + w_g L_RGB + w_t L_tail + lambda_OK L_OKLab`.",
        "- Local-tail and OKLab are training losses; the diagram explicitly states that no additional inference module is introduced.",
    ])
    REPORT.write_text(report + "\n", encoding="utf-8")
    print(pdf)
    print(png)


if __name__ == "__main__":
    render()
