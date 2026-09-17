#!/usr/bin/env python3
"""Generate the final ICASSP Figure 2 from the frozen operating-curve data.

This renderer is deliberately data-only: it reads the persisted project-test
CSV and its paired summary, connects measured points with straight segments,
and uses linear interpolation only at the five PSNR values declared by the
operating-curve audit. It never evaluates a model or changes an alpha value.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "project_test_curve.csv"
SUMMARY_CSV = ROOT / "operating_curve_summary.csv"
ALPHA_GRID = ROOT / "FROZEN_ALPHA_GRID.txt"
AUDIT_MD = ROOT / "OPERATING_CURVE_AUDIT.md"
METADATA_JSON = ROOT / "curve_metadata.json"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_STEM = OUTPUT_DIR / "figure2_operating_curve"
CAPTION_PATH = OUTPUT_DIR / "figure2_caption.txt"
NOTES_PATH = OUTPUT_DIR / "figure2_generation_notes.md"

GLOBAL_METHOD = "MBRS crop-trained global"
OURS_METHOD = "Ours"
METHODS = (GLOBAL_METHOD, OURS_METHOD)
ALPHAS = np.array([0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.00])
MATCHED_PSNR = np.array([36.854909, 37.479001, 38.103093, 38.727185, 39.351276])
COMMON_OVERLAP = (float(MATCHED_PSNR[0]), float(MATCHED_PSNR[-1]))

FIGURE_WIDTH_IN = 7.16
FIGURE_HEIGHT_IN = 3.30
PNG_DPI = 300

COLORS = {
    GLOBAL_METHOD: "#0072B2",  # blue; circle + solid line
    OURS_METHOD: "#D55E00",  # vermillion; square + dashed line
}
MARKERS = {GLOBAL_METHOD: "o", OURS_METHOD: "s"}
LINESTYLES = {GLOBAL_METHOD: "-", OURS_METHOD: (0, (4.0, 2.2))}
DISPLAY_NAMES = {
    GLOBAL_METHOD: "MBRS crop-trained Global",
    OURS_METHOD: "Ours = Hard Local-Tail + global OKLab",
}

METRICS = (
    ("top25_local_psnr", "Top-25 Local PSNR (dB) $\u2191", "(a) Top-25 Local PSNR", (35.30, 39.45)),
    ("lpips", "LPIPS $\u2193", "(b) LPIPS", (0.00085, 0.00245)),
    ("ber30", "BER@30 $\u2193", "(c) BER@30", (0.11295, 0.11370)),
)

CAPTION = (
    "Fig. 2. Fidelity\u2013robustness operating-point analysis on the frozen "
    "project-test set. Residual strength is varied using the preregistered "
    "global alpha grid without retraining or per-image tuning. Across the "
    "common global-PSNR range, Ours maintains higher Top-25 local PSNR and "
    "lower LPIPS than the crop-trained Global baseline, while BER@30 is tied "
    "at the natural operating point and shows a small unfavorable shift at "
    "most matched-PSNR points. Curves connect measured operating points; "
    "matched comparisons use linear interpolation only within the measured "
    "PSNR overlap."
)


def load_alpha_grid() -> np.ndarray:
    values = [
        float(line.strip())
        for line in ALPHA_GRID.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    actual = np.array(values, dtype=float)
    np.testing.assert_allclose(actual, ALPHAS, rtol=0.0, atol=0.0)
    return actual


def load_rows() -> dict[str, dict[str, np.ndarray]]:
    with DATA_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 18:
        raise ValueError(f"expected 18 project-test rows, found {len(rows)}")
    if {row["method"] for row in rows} != set(METHODS):
        raise ValueError("project-test CSV contains an unexpected method name")
    if {row["split"] for row in rows} != {"project_test"}:
        raise ValueError("figure data must come only from project_test rows")

    result: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        subset.sort(key=lambda row: float(row["alpha"]))
        alpha = np.array([float(row["alpha"]) for row in subset])
        np.testing.assert_allclose(alpha, ALPHAS, rtol=0.0, atol=0.0)
        result[method] = {
            "alpha": alpha,
            "psnr": np.array([float(row["psnr"]) for row in subset]),
            "top25_local_psnr": np.array([float(row["top25_local_psnr"]) for row in subset]),
            "lpips": np.array([float(row["lpips"]) for row in subset]),
            "ber30": np.array([float(row["ber30"]) for row in subset]),
        }
    return result


def verify_summary(rows: dict[str, dict[str, np.ndarray]]) -> None:
    """Check that the paired summary agrees with the raw plotted CSV."""
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    if len(summary) != 9:
        raise ValueError(f"expected 9 paired summary rows, found {len(summary)}")
    summary_alpha = np.array([float(row["alpha"]) for row in summary])
    np.testing.assert_allclose(summary_alpha, ALPHAS, rtol=0.0, atol=0.0)
    fields = {
        GLOBAL_METHOD: {
            "psnr": "Global PSNR",
            "top25_local_psnr": "Global Top-25 Local PSNR",
            "lpips": "Global LPIPS",
            "ber30": "Global BER30",
        },
        OURS_METHOD: {
            "psnr": "Ours PSNR",
            "top25_local_psnr": "Ours Top-25 Local PSNR",
            "lpips": "Ours LPIPS",
            "ber30": "Ours BER30",
        },
    }
    for metric in fields[GLOBAL_METHOD]:
        for method in METHODS:
            summary_values = np.array([float(row[fields[method][metric]]) for row in summary])
            np.testing.assert_allclose(summary_values, rows[method][metric], rtol=0.0, atol=1e-15)


def ascending_x(series: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    order = np.argsort(series["psnr"])
    return series["psnr"][order], {key: value[order] for key, value in series.items()}


def interpolate(series: dict[str, np.ndarray], metric: str) -> np.ndarray:
    x, ordered = ascending_x(series)
    targets = MATCHED_PSNR.copy()
    # The audit prints endpoints to six decimals. Snap only a target that is
    # within that display-rounding tolerance to the corresponding measured
    # endpoint; this keeps the comparison inside the measured curve without
    # silently extrapolating the raw data.
    endpoint_tolerance = 1e-6
    if targets[0] < x[0]:
        if abs(targets[0] - x[0]) > endpoint_tolerance:
            raise ValueError("matched PSNR points fall outside a measured curve")
        targets[0] = x[0]
    if targets[-1] > x[-1]:
        if abs(targets[-1] - x[-1]) > endpoint_tolerance:
            raise ValueError("matched PSNR points fall outside a measured curve")
        targets[-1] = x[-1]
    if targets[0] < x[0] or targets[-1] > x[-1]:
        raise ValueError("matched PSNR points fall outside a measured curve")
    return np.interp(targets, x, ordered[metric])


def style_axes(axis: plt.Axes, metric_key: str, ylabel: str, title: str, y_limits: tuple[float, float]) -> None:
    axis.set_title(title, loc="left", fontsize=9.5, fontweight="semibold", pad=5.0)
    axis.set_xlim(36.10, 40.05)
    axis.set_ylim(*y_limits)
    axis.set_xlabel("Global PSNR (dB)", fontsize=9.0, labelpad=2.5)
    axis.set_ylabel(ylabel, fontsize=9.0, labelpad=3.0)
    axis.tick_params(axis="both", which="major", labelsize=9.0, length=3.0, width=0.65, pad=2.0)
    axis.grid(axis="y", color="#B8C2CC", linewidth=0.45, alpha=0.42)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#5F6B76")
    axis.spines["bottom"].set_color("#5F6B76")
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    # One-decimal labels at 1-dB spacing stay readable in the narrow panels.
    axis.set_xticks([36.5, 37.5, 38.5, 39.5])
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if metric_key == "lpips":
        axis.yaxis.set_major_locator(FixedLocator([0.0010, 0.0015, 0.0020, 0.0024]))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    elif metric_key == "ber30":
        axis.yaxis.set_major_locator(FixedLocator([0.1130, 0.1132, 0.1134, 0.1136]))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))


def plot_metric(axis: plt.Axes, rows: dict[str, dict[str, np.ndarray]], metric: str) -> None:
    axis.axvspan(
        COMMON_OVERLAP[0],
        COMMON_OVERLAP[1],
        color="#DCE6F2",
        alpha=0.28,
        linewidth=0,
        zorder=0,
    )
    for method in METHODS:
        x, ordered = ascending_x(rows[method])
        axis.plot(
            x,
            ordered[metric],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.65,
            marker=MARKERS[method],
            markersize=4.25,
            markerfacecolor=COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.65,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=3,
        )

        # Hollow markers are the five audited linear interpolations, not new
        # measured points. They are intentionally subtle and sit above lines.
        matched_y = interpolate(rows[method], metric)
        axis.plot(
            MATCHED_PSNR,
            matched_y,
            linestyle="None",
            marker=MARKERS[method],
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor=COLORS[method],
            markeredgewidth=0.9,
            zorder=4,
        )

        # A larger diamond identifies the raw alpha=1 natural operating point.
        alpha_one = int(np.flatnonzero(np.isclose(rows[method]["alpha"], 1.0, atol=0.0))[0])
        axis.plot(
            [rows[method]["psnr"][alpha_one]],
            [rows[method][metric][alpha_one]],
            linestyle="None",
            marker="D",
            markersize=6.0,
            markerfacecolor="white",
            markeredgecolor=COLORS[method],
            markeredgewidth=1.0,
            zorder=5,
        )


def write_caption_and_notes(rows: dict[str, dict[str, np.ndarray]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CAPTION_PATH.write_text(CAPTION + "\n", encoding="utf-8")
    global_x, _ = ascending_x(rows[GLOBAL_METHOD])
    ours_x, _ = ascending_x(rows[OURS_METHOD])
    matched_lines = []
    for target in MATCHED_PSNR:
        matched_lines.append(f"  - {target:.6f} dB")
    notes = f"""# Figure 2 generation notes

## Provenance

- Raw plotted values: `project_test_curve.csv` (project-test rows only).
- Paired cross-check: `operating_curve_summary.csv`.
- Frozen grid: `FROZEN_ALPHA_GRID.txt`.
- Interpolation and interpretation audit: `OPERATING_CURVE_AUDIT.md`.
- Frozen-run metadata: `curve_metadata.json`.
- No file under `paper_bundle/` and no Prism manuscript file was modified.

## Data and transformations

- Methods plotted: `MBRS crop-trained Global` and `Ours = Hard Local-Tail + global OKLab`.
- Metrics plotted exactly: `psnr` on the x-axis; `top25_local_psnr`, `lpips`, and `ber30` on the three y-axes.
- Alpha grid used exactly: `0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.00`.
- Raw measured point count: 9 Global points + 9 Ours points. All raw points remain visible.
- Global measured PSNR range: `{global_x[0]:.6f}\u2013{global_x[-1]:.6f} dB`.
- Ours measured PSNR range: `{ours_x[0]:.6f}\u2013{ours_x[-1]:.6f} dB`.
- Common PSNR overlap: `{COMMON_OVERLAP[0]:.6f}\u2013{COMMON_OVERLAP[1]:.6f} dB`.
- Matched-PSNR markers shown: yes, hollow method-shaped markers at these five audited points:
{chr(10).join(matched_lines)}
- Interpolation: `numpy.interp` linear interpolation of each measured curve within the common overlap only. No extrapolation, spline, polynomial, regression, or smoothing was used.
- Endpoint precision note: the audit's lower endpoint is printed as `36.854909`, while the raw Ours endpoint is `36.854909401406786`; this 6-decimal display-rounding difference is snapped to the measured endpoint before interpolation. No extrapolation is performed.

## Visual encoding

- Figure layout: three horizontal panels, shared two-method legend, white background, compact conference typography.
- Global: blue circle markers with a solid line; Ours: vermillion square markers with a dashed line. Marker and line differences preserve grayscale readability.
- Alpha=1 natural operating points: larger hollow diamonds over the corresponding raw points for both methods.
- Common-overlap shading: used in all three panels as a subtle light-blue vertical band from `{COMMON_OVERLAP[0]:.6f}` to `{COMMON_OVERLAP[1]:.6f}` dB; it does not alter data or axes.
- Identical x-axis limits in all panels: `36.10–40.05 dB`.
- Y-axis limits: Top-25 Local PSNR `35.30–39.45 dB`; LPIPS `0.00085–0.00245`; BER@30 `0.11295–0.11370`.
- BER@30 uses normal numeric ordering and four-decimal ticks; no visual inversion, clipping, normalization, or compression was applied.

## Outputs

- PDF: `figure2_operating_curve.pdf` (vector lines/text suitable for LaTeX inclusion).
- SVG: `figure2_operating_curve.svg` (editable vector backup).
- PNG: `figure2_operating_curve.png` at {PNG_DPI} DPI.
- Figure dimensions: `{FIGURE_WIDTH_IN:.2f} in \u00d7 {FIGURE_HEIGHT_IN:.2f} in`; PNG raster dimensions `{int(FIGURE_WIDTH_IN * PNG_DPI)} \u00d7 {int(FIGURE_HEIGHT_IN * PNG_DPI)} px`.
- Generation did not retrain, retune, select new operating points, or change any alpha value.
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")


def render(rows: dict[str, dict[str, np.ndarray]]) -> None:
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
    figure, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), squeeze=False)
    axes = axes[0]
    for axis, (metric, ylabel, title, y_limits) in zip(axes, METRICS):
        style_axes(axis, metric, ylabel, title, y_limits)
        plot_metric(axis, rows, metric)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[GLOBAL_METHOD],
            linestyle=LINESTYLES[GLOBAL_METHOD],
            marker=MARKERS[GLOBAL_METHOD],
            markersize=4.3,
            markerfacecolor=COLORS[GLOBAL_METHOD],
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=1.65,
            label=DISPLAY_NAMES[GLOBAL_METHOD],
        ),
        Line2D(
            [0],
            [0],
            color=COLORS[OURS_METHOD],
            linestyle=LINESTYLES[OURS_METHOD],
            marker=MARKERS[OURS_METHOD],
            markersize=4.3,
            markerfacecolor=COLORS[OURS_METHOD],
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=1.65,
            label=DISPLAY_NAMES[OURS_METHOD],
        ),
        Line2D(
            [0],
            [0],
            color="#38434D",
            linestyle="None",
            marker="D",
            markersize=5.2,
            markerfacecolor="white",
            markeredgecolor="#38434D",
            markeredgewidth=0.9,
            label=r"$\alpha=1$ natural point",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        fontsize=9.0,
        handlelength=2.2,
        columnspacing=1.25,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(left=0.080, right=0.998, bottom=0.225, top=0.815, wspace=0.56)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=PNG_DPI, facecolor="white")
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white")
    figure.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor="white")
    plt.close(figure)


def main() -> None:
    load_alpha_grid()
    rows = load_rows()
    verify_summary(rows)
    write_caption_and_notes(rows)
    render(rows)
    print(f"wrote {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"wrote {OUTPUT_STEM.with_suffix('.png')}")
    print(f"wrote {OUTPUT_STEM.with_suffix('.svg')}")
    print(f"wrote {CAPTION_PATH}")
    print(f"wrote {NOTES_PATH}")


if __name__ == "__main__":
    main()
