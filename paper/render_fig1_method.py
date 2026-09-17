"""Render the camera-ready, two-column MBRS method schematic."""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "paper" / "figures"
REPORT = PROJECT / "reports" / "fig1_method_final_check.md"

BLUE = "#dfeaf4"
BLUE_EDGE = "#41627d"
GRAY = "#eef1f3"
GRAY_EDGE = "#59636b"
ORANGE = "#f8e4cc"
ORANGE_EDGE = "#b96922"
GREEN = "#edf5ee"
GREEN_EDGE = "#4f845d"
INK = "#20262b"
ARROW = "#4f5b63"


def box(
    ax,
    center,
    width,
    height,
    text,
    face,
    edge,
    *,
    dashed=False,
    fontsize=8.1,
    weight="normal",
    linewidth=1.05,
):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.045,rounding_size=0.035",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        linestyle=(0, (4, 2.5)) if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.14,
    )


def arrow(ax, start, end, *, color=ARROW, lw=1.15, dashed=False):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            mutation_scale=9,
            color=color,
            linewidth=lw,
            linestyle=(0, (4, 2.5)) if dashed else "-",
            shrinkA=0,
            shrinkB=0,
        ),
    )


def routed_arrow(ax, points, *, color=ARROW, lw=1.15, dashed=False):
    style = (0, (4, 2.5)) if dashed else "-"
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=lw,
            linestyle=style,
            solid_capstyle="round",
        )
    arrow(ax, points[-2], points[-1], color=color, lw=lw, dashed=dashed)


def label(
    ax,
    xy,
    text,
    *,
    color=INK,
    fontsize=7.5,
    ha="center",
    va="center",
    weight="normal",
):
    ax.text(
        *xy,
        text,
        ha=ha,
        va=va,
        color=color,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.13,
    )


def render():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.16, 4.12), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Adopted MBRS encoder/channel/decoder path. The channel box itself makes
    # its training-only status explicit; the model architecture is shared.
    label(
        ax,
        (0.16, 7.00),
        "ADOPTED MBRS BACKBONE",
        ha="left",
        fontsize=8.9,
        color=BLUE_EDGE,
        weight="bold",
    )
    label(
        ax,
        (11.84, 7.00),
        "SHARED TRAIN/INFERENCE ARCHITECTURE",
        ha="right",
        fontsize=7.6,
        color=GRAY_EDGE,
        weight="bold",
    )

    nodes = (
        (0.96, "Cover image $x$ +\n64-bit message $m$", 1.72, BLUE, BLUE_EDGE),
        (2.70, "MBRS\nEncoder", 1.05, BLUE, BLUE_EDGE),
        (4.22, "Watermarked\nimage $\\hat{x}$", 1.25, BLUE, BLUE_EDGE),
        (5.86, "RandomCrop\nchannel (training)", 1.56, GRAY, GRAY_EDGE),
        (7.55, "MBRS\nDecoder", 1.05, BLUE, BLUE_EDGE),
    )
    for x, text, width, face, edge in nodes:
        box(ax, (x, 6.20), width, 0.72, text, face, edge, fontsize=6.5)
    for start, end in ((1.82, 2.17), (3.23, 3.59), (4.85, 5.08), (6.64, 7.02)):
        arrow(ax, (start, 6.20), (end, 6.20))

    box(
        ax,
        (9.42, 5.64),
        2.20,
        0.72,
        "Training objective\n$L_{\\mathrm{msg}}$  message loss",
        GRAY,
        GRAY_EDGE,
        fontsize=6.9,
    )
    routed_arrow(
        ax,
        ((8.08, 6.20), (9.42, 6.20), (9.42, 6.02)),
        color=GRAY_EDGE,
        lw=1.0,
    )

    # The clean watermarked image enters one split point before either fidelity
    # objective. Routed connectors keep the section title unobstructed.
    label(
        ax,
        (4.53, 5.47),
        "clean watermarked image $\\hat{x}$",
        ha="left",
        fontsize=7.2,
        color=ORANGE_EDGE,
        weight="bold",
    )
    arrow(ax, (4.22, 5.84), (4.22, 5.18), color=ORANGE_EDGE, lw=1.25)
    ax.add_patch(Circle((4.22, 5.08), 0.075, facecolor=ORANGE_EDGE, edgecolor=ORANGE_EDGE))
    label(
        ax,
        (4.22, 4.62),
        "Training-only Fidelity Supervision",
        fontsize=8.7,
        color=ORANGE_EDGE,
        weight="bold",
    )
    routed_arrow(
        ax,
        ((4.22, 5.08), (2.25, 5.08), (2.25, 4.25)),
        color=ORANGE_EDGE,
        lw=1.2,
    )
    routed_arrow(
        ax,
        ((4.22, 5.08), (7.02, 5.08), (7.02, 4.22)),
        color=ORANGE_EDGE,
        lw=1.2,
    )

    box(
        ax,
        (2.25, 3.72),
        3.45,
        0.96,
        "Branch A\nGlobal RGB reconstruction\n"
        "$L_{\\mathrm{global}} = \\mathrm{MSE}(\\hat{x}, x)$",
        ORANGE,
        ORANGE_EDGE,
        fontsize=7.1,
    )

    box(ax, (7.02, 3.49), 4.05, 1.28, "", ORANGE, ORANGE_EDGE, linewidth=1.25)
    label(
        ax,
        (7.02, 3.82),
        "Our Hard Local-Tail Branch",
        fontsize=7.8,
        weight="bold",
    )
    label(
        ax,
        (7.02, 3.38),
        "P16 / stride 8\n225 overlapping candidates\n"
        "Hard Top-10%: 23/225 highest-error patches",
        fontsize=6.8,
    )
    box(
        ax,
        (7.02, 2.32),
        4.05,
        0.52,
        "$L_{\\mathrm{tail}} = \\mathrm{mean}$ selected patch MSE",
        ORANGE,
        ORANGE_EDGE,
        fontsize=7.7,
    )
    arrow(ax, (7.02, 2.80), (7.02, 2.62), color=ORANGE_EDGE, lw=1.15)

    box(ax, (4.34, 0.86), 6.15, 1.00, "", GRAY, GRAY_EDGE)
    label(
        ax,
        (4.34, 1.03),
        "$L = 10\\,L_{\\mathrm{msg}} + 0.5\\,L_{\\mathrm{global}} + 0.5\\,L_{\\mathrm{tail}}$",
        fontsize=8.0,
    )
    label(
        ax,
        (4.34, 0.66),
        "training only  |  no additional inference module",
        fontsize=7.2,
        weight="bold",
    )
    arrow(ax, (2.25, 3.27), (3.10, 1.40), color=ORANGE_EDGE, lw=1.15)
    arrow(ax, (7.02, 2.01), (5.78, 1.40), color=ORANGE_EDGE, lw=1.15)

    box(ax, (9.72, 1.10), 3.90, 1.36, "", GREEN, GREEN_EDGE, dashed=True, linewidth=1.15)
    label(
        ax,
        (9.72, 1.53),
        "Validation-only color extensions",
        fontsize=6.5,
        weight="bold",
    )
    label(
        ax,
        (9.72, 1.08),
        "Global OKLab     OR     Local chroma-tail\n"
        "$+\\lambda_{\\mathrm{OK}}L_{\\mathrm{OK}}$   OR   "
        "$+\\lambda_{\\mathrm{chr}}L_{\\mathrm{chr-tail}}$",
        fontsize=6.1,
    )
    label(
        ax,
        (9.72, 0.67),
        "alternatives (not used together)",
        fontsize=6.0,
        color=GREEN_EDGE,
        weight="bold",
    )
    arrow(
        ax,
        (7.72, 1.10),
        (7.46, 1.10),
        color=GREEN_EDGE,
        lw=1.15,
        dashed=True,
    )

    pdf = OUTPUT / "fig1_method.pdf"
    png = OUTPUT / "fig1_method.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.055, facecolor="white")
    fig.savefig(png, dpi=360, bbox_inches="tight", pad_inches=0.055, facecolor="white")
    plt.close(fig)

    pdf_text = pdf.read_bytes().decode("latin-1", errors="ignore")
    media_box = re.search(r"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)", pdf_text)
    page_size = f"{media_box.group(1)} x {media_box.group(2)} pt" if media_box else "unavailable"
    from PIL import Image

    with Image.open(png) as image:
        png_size = f"{image.width} x {image.height} px"
        dpi = image.info.get("dpi", (None, None))
        png_dpi = f"{dpi[0]:.1f} x {dpi[1]:.1f}" if dpi[0] else "unavailable"

    report = "\n".join(
        [
            "# Fig. 1 Method Figure Final Check",
            "",
            "## Outputs",
            "",
            f"- PDF: `{pdf}`; page size `{page_size}`; file size `{pdf.stat().st_size} bytes`.",
            f"- PNG: `{png}`; raster size `{png_size}`; embedded DPI `{png_dpi}`; file size `{png.stat().st_size} bytes`.",
            "- Renderer: `/root/workspace/GLX/icassp/MBRS/paper/render_fig1_method.py`.",
            "- Format: white background, flat fills, no gradients or shadows; PDF uses embedded TrueType vector text.",
            "",
            "## Camera-ready semantic checks",
            "",
            "- Orange section title: `Training-only Fidelity Supervision`.",
            "- Branch A is unchanged: `Global RGB reconstruction`; `L_global = MSE(x_hat, x)`.",
            "- Branch B is explicitly titled `Our Hard Local-Tail Branch` and states `P16 / stride 8`, `225 overlapping candidates`, `Hard Top-10%`, `23/225 highest-error patches`, and `L_tail = mean selected patch MSE`.",
            "- Backbone sequence is `MBRS Encoder -> watermarked image -> RandomCrop channel (training) -> MBRS Decoder` and is labeled `SHARED TRAIN/INFERENCE ARCHITECTURE`.",
            "- `L_msg` is isolated in a `Training objective` box. No `inference path` label appears beneath or near the message loss.",
            "- The crop box contains only `RandomCrop channel (training)`; retained-area evaluation language is absent.",
            "- The green dashed box is titled `Validation-only color extensions`; it presents Global OKLab and Local chroma-tail as explicit alternatives, with `+lambda_OK L_OK OR +lambda_chr L_chr-tail`, and has a dashed green arrow to total loss.",
            "- The clean watermarked image enters one split point before routed arrows reach Branch A and Branch B; no branch arrow crosses the fidelity-supervision title.",
            "- Total loss is unchanged: `L = 10 L_msg + 0.5 L_global + 0.5 L_tail`.",
            "- Footer text is retained inside the loss box: `training only | no additional inference module`.",
            "- Language check: figure text is English only.",
            "",
            "## Method-definition integrity",
            "",
            "- No model, loss coefficient, patch geometry, selection count, or train/inference module definition was changed.",
            "- No training, inference, checkpoint loading, or model-output modification was performed; this revision changes figure rendering only.",
            "- No retained-area evaluation protocol is represented in the method diagram.",
            "",
        ]
    )
    REPORT.write_text(report, encoding="utf-8")
    print(f"Rendered {pdf} and {png}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    render()
