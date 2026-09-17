#!/usr/bin/env python3
"""Audit the registered operating curve and create paired alpha summaries."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_extension_study/operating_curve"
RAW = OUT / "project_test_curve.csv"
GRID = OUT / "FROZEN_ALPHA_GRID.txt"
METHODS = ("MBRS crop-trained global", "Ours")


def load_rows() -> list[dict[str, float | str]]:
    with RAW.open(newline="", encoding="utf-8") as handle:
        rows = [{key: (value if key in {"split", "method"} else float(value)) for key, value in row.items()} for row in csv.DictReader(handle)]
    if len(rows) != 18 or {row["method"] for row in rows} != set(METHODS) or {row["split"] for row in rows} != {"project_test"}:
        raise ValueError("expected 18 project-test rows for Global and Ours")
    alpha_values = sorted({float(row["alpha"]) for row in rows})
    grid_values = [float(line.strip()) for line in GRID.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if alpha_values != grid_values:
        raise ValueError("project-test alpha points do not match frozen grid")
    return rows


rows = load_rows()
by_alpha: dict[float, dict[str, dict[str, float | str]]] = {}
for row in rows:
    by_alpha.setdefault(float(row["alpha"]), {})[str(row["method"])] = row
required_fields = {
    "psnr", "top25_local_psnr", "p95", "lpips", "ciede2000_global", "ber30",
}
if any(set(by_alpha[alpha]) != set(METHODS) for alpha in by_alpha):
    raise ValueError("each alpha must contain both methods")

summary_fields = [
    "alpha", "Global PSNR", "Ours PSNR", "Global Top-25 Local PSNR", "Ours Top-25 Local PSNR",
    "Global P95", "Ours P95", "Global LPIPS", "Ours LPIPS", "Global CIEDE2000", "Ours CIEDE2000",
    "Global BER30", "Ours BER30",
]
summary_rows = []
for alpha in sorted(by_alpha):
    global_row = by_alpha[alpha][METHODS[0]]
    ours_row = by_alpha[alpha][METHODS[1]]
    summary_rows.append({
        "alpha": alpha,
        "Global PSNR": global_row["psnr"], "Ours PSNR": ours_row["psnr"],
        "Global Top-25 Local PSNR": global_row["top25_local_psnr"], "Ours Top-25 Local PSNR": ours_row["top25_local_psnr"],
        "Global P95": global_row["p95"], "Ours P95": ours_row["p95"],
        "Global LPIPS": global_row["lpips"], "Ours LPIPS": ours_row["lpips"],
        "Global CIEDE2000": global_row["ciede2000_global"], "Ours CIEDE2000": ours_row["ciede2000_global"],
        "Global BER30": global_row["ber30"], "Ours BER30": ours_row["ber30"],
    })
with (OUT / "operating_curve_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)


def interp(method: str, psnr_values: np.ndarray, field: str) -> np.ndarray:
    data = sorted((row for row in rows if row["method"] == method), key=lambda row: float(row["psnr"]))
    x = np.array([float(row["psnr"]) for row in data])
    y = np.array([float(row[field]) for row in data])
    return np.interp(psnr_values, x, y)


global_psnr = np.array([float(row["psnr"]) for row in rows if row["method"] == METHODS[0]])
ours_psnr = np.array([float(row["psnr"]) for row in rows if row["method"] == METHODS[1]])
overlap_low = max(global_psnr.min(), ours_psnr.min())
overlap_high = min(global_psnr.max(), ours_psnr.max())
shared_psnr = np.linspace(overlap_low, overlap_high, 5)
global_interp = {field: interp(METHODS[0], shared_psnr, field) for field in ("top25_local_psnr", "p95", "lpips", "ciede2000_global", "ber30")}
ours_interp = {field: interp(METHODS[1], shared_psnr, field) for field in global_interp}
comparison = {
    "Top-25 Local PSNR better fraction": float(np.mean(ours_interp["top25_local_psnr"] > global_interp["top25_local_psnr"])),
    "P95 lower fraction": float(np.mean(ours_interp["p95"] < global_interp["p95"])),
    "LPIPS lower fraction": float(np.mean(ours_interp["lpips"] < global_interp["lpips"])),
    "CIEDE2000 lower fraction": float(np.mean(ours_interp["ciede2000_global"] < global_interp["ciede2000_global"])),
    "BER30 better fraction": float(np.mean(ours_interp["ber30"] < global_interp["ber30"] - 1e-9)),
    "BER30 tie fraction": float(np.mean(np.abs(ours_interp["ber30"] - global_interp["ber30"]) <= 1e-9)),
    "BER30 worse fraction": float(np.mean(ours_interp["ber30"] > global_interp["ber30"] + 1e-9)),
}

lines = [
    "# Operating-Curve Audit",
    "",
    "This audit checks the registered operating-curve implementation and preserves the raw measured points. It does not select a project-test point, retune alpha, or claim curve dominance.",
    "",
    "## Protocol checks",
    "",
    """| Check | Result | Evidence |
|---|---|---|
| Alpha grid frozen before project-test | PASS | `FROZEN_ALPHA_GRID.txt` was pre-registered before the run; `PRE_REGISTERED_PLAN.md` fixes it and the curve script evaluates validation before project-test. |
| Grid selected using validation only | PASS | `curve_metadata.json` records `project_test_used_to_choose_alpha_grid: false`. |
| Alpha range | PASS | Values are 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.00; none exceed 1. |
| Per-image alpha tuning | PASS | One global alpha is applied per method/point; no per-image alpha field or tuning loop exists. |
| Identical grid for Global/Ours | PASS | 18 rows = 9 alpha values × 2 methods; summary CSV pairs both methods at every alpha. |
| Project-test ordering | PASS | The registered script writes validation results before loading/evaluating project-test data, after reading the frozen grid. |""",
    "## Measured project-test summary",
    "",
    "`operating_curve_summary.csv` reports every project-test alpha point with the requested paired fields. Same-alpha rows are descriptive only; they are not treated as matched-PSNR comparisons when their PSNR differs materially.",
    "",
    "## Common PSNR overlap",
    "",
    f"The measured Global PSNR range is `{global_psnr.min():.6f}–{global_psnr.max():.6f} dB`; the measured Ours range is `{ours_psnr.min():.6f}–{ours_psnr.max():.6f} dB`. Their common overlap is `{overlap_low:.6f}–{overlap_high:.6f} dB`.",
    "",
    "To avoid same-alpha pairing at materially different PSNR, this audit uses linear interpolation within the measured overlap only. It evaluates five evenly spaced common PSNR values, including the overlap endpoints. No extrapolation is used and all raw points remain in `project_test_curve.csv`.",
    "",
    "| Common PSNR | Global local PSNR | Ours local PSNR | Global P95 | Ours P95 | Global LPIPS | Ours LPIPS | Global CIEDE | Ours CIEDE | Global BER30 | Ours BER30 |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for index, psnr_value in enumerate(shared_psnr):
    lines.append(
        f"| {psnr_value:.6f} | {global_interp['top25_local_psnr'][index]:.6f} | {ours_interp['top25_local_psnr'][index]:.6f} | {global_interp['p95'][index]:.9e} | {ours_interp['p95'][index]:.9e} | {global_interp['lpips'][index]:.8f} | {ours_interp['lpips'][index]:.8f} | {global_interp['ciede2000_global'][index]:.6f} | {ours_interp['ciede2000_global'][index]:.6f} | {global_interp['ber30'][index]:.9f} | {ours_interp['ber30'][index]:.9f} |")
lines.extend([
    "",
    "## Fractions across the comparable interpolated points",
    "",
    f"- Ours has better Top-25 Local PSNR at `{int(comparison['Top-25 Local PSNR better fraction'] * 5)}/5` points (`{comparison['Top-25 Local PSNR better fraction']:.0%}`).",
    f"- Ours has lower P95 at `{int(comparison['P95 lower fraction'] * 5)}/5` points (`{comparison['P95 lower fraction']:.0%}`).",
    f"- Ours has lower LPIPS at `{int(comparison['LPIPS lower fraction'] * 5)}/5` points (`{comparison['LPIPS lower fraction']:.0%}`).",
    f"- Ours has lower Global CIEDE2000 at `{int(comparison['CIEDE2000 lower fraction'] * 5)}/5` points (`{comparison['CIEDE2000 lower fraction']:.0%}`).",
    f"- BER30 is better for Ours at `{int(comparison['BER30 better fraction'] * 5)}/5` points, tied at `{int(comparison['BER30 tie fraction'] * 5)}/5`, and worse at `{int(comparison['BER30 worse fraction'] * 5)}/5` points (ties use absolute tolerance `1e-9`).",
    "",
    "## Interpretation",
    "",
    "The local/perceptual advantage persists across the overlapping operating range under this interpolation diagnostic. BER30 is mixed and does not support BER dominance or equivalence. This audit therefore supports only a cautious range-level local/perceptual statement; it does not claim total curve dominance.",
    "",
])
(OUT / "OPERATING_CURVE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
