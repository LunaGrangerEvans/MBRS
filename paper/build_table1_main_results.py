#!/usr/bin/env python3
"""Build Table I from the frozen paper-freeze-v1 evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
FROZEN = Path("/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv")
AGGREGATED = PROJECT / "paper/main_table.csv"
MANIFEST = PROJECT / "paper/evidence_manifest.json"
OUTPUT = PROJECT / "paper/tables/table1_main_results.tex"
REPORT = PROJECT / "reports/table1_evidence_check.md"

METHODS = (
    (
        "Global continuation",
        "Global continuation (RGB image weight 1)",
        "controlled_seed17_global_continuation",
    ),
    (
        "Hard P16/S16/Top-25%",
        "Hard MSE, patch16 / stride16 / Top25%",
        "controlled_seed17_hard_p16_t25_l50",
    ),
    (
        "Hard P16/S8/Top-25%",
        "Hard MSE, patch16 / stride8 / Top25%",
        "controlled_seed17_hard_patch16_stride8_top25_weight50",
    ),
    (
        "Hard P16/S8/Top-10% (Ours)",
        "Hard MSE, patch16 / stride8 / Top10% (main)",
        "controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50",
    ),
)

METRICS = (
    ("PSNR", "global_psnr", "max", 6),
    ("SSIM", "global_ssim", "max", 6),
    ("LPIPS", "full_lpips", "min", 6),
    ("Top-25 local PSNR", "top25_local_psnr", "max", 6),
    ("P95 MSE", "patch_mse_p95", "min", 10),
    ("Gini", "gini", "min", 6),
    ("BER@30%", "ber30", "min", 7),
)

EXPECTED = {
    "Global continuation": (36.263172, 0.950759, 0.002350, 35.522213, 3.016532e-4, 0.095007, 0.1131250),
    "Hard P16/S16/Top-25%": (36.418147, 0.952313, 0.002200, 35.716045, 2.876246e-4, 0.089031, 0.1131875),
    "Hard P16/S8/Top-25%": (36.445329, 0.952631, 0.002205, 35.747490, 2.856057e-4, 0.088604, 0.1131875),
    "Hard P16/S8/Top-10% (Ours)": (36.442300, 0.952541, 0.002215, 35.754376, 2.844551e-4, 0.086897, 0.1129375),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def aggregate(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "global_psnr": 10.0 * math.log10(1.0 / mean(rows, "global_mse")),
        "global_ssim": mean(rows, "global_ssim"),
        "full_lpips": mean(rows, "full_lpips"),
        "top25_local_psnr": mean(rows, "top25_local_psnr"),
        "patch_mse_p95": mean(rows, "patch_mse_p95"),
        "gini": mean(rows, "gini"),
        "ber30": mean(rows, "ber30"),
    }


def displayed(metric: str, value: float) -> str:
    if metric == "P95 MSE":
        return f"{value * 1e4:.6f}e-4"
    digits = next(item[3] for item in METRICS if item[0] == metric)
    return f"{value:.{digits}f}"


def latex_value(metric: str, value: float) -> str:
    if metric == "P95 MSE":
        return f"{value * 1e4:.6f}$\\times10^{{-4}}$"
    digits = next(item[3] for item in METRICS if item[0] == metric)
    return f"{value:.{digits}f}"


def main() -> None:
    provenance = json.loads(MANIFEST.read_text())
    expected_hashes = provenance["artifacts"]
    assert provenance["protocol"] == "paper-freeze-v1; current clipped RGB evaluator"
    assert provenance["training"] is False
    assert sha256(FROZEN) == expected_hashes[str(FROZEN)]
    assert sha256(AGGREGATED) == expected_hashes[str(AGGREGATED)]

    with FROZEN.open(newline="") as stream:
        per_image = list(csv.DictReader(stream))
    with AGGREGATED.open(newline="") as stream:
        frozen_aggregate = {row["run"]: row for row in csv.DictReader(stream)}

    values: dict[str, dict[str, float]] = {}
    checks: list[tuple[str, str, str, float, float, str]] = []
    for display_name, evidence_name, run in METHODS:
        method_rows = [row for row in per_image if row["method"] == evidence_name]
        assert len(method_rows) == 50, (display_name, len(method_rows))
        computed = aggregate(method_rows)
        values[display_name] = computed
        for position, (label, key, _, digits) in enumerate(METRICS):
            frozen_value = float(frozen_aggregate[run][key])
            assert math.isclose(computed[key], frozen_value, rel_tol=0.0, abs_tol=1e-12)
            expected = EXPECTED[display_name][position]
            tolerance = 0.5 * 10 ** (-digits)
            status = "PASS" if abs(computed[key] - expected) < tolerance else "FAIL"
            checks.append((display_name, label, key, computed[key], expected, status))
    assert all(item[-1] == "PASS" for item in checks)

    winners: dict[str, str] = {}
    for label, key, direction, _ in METRICS:
        chooser = max if direction == "max" else min
        winners[key] = chooser(METHODS, key=lambda method: values[method[0]][key])[0]

    latex_rows = []
    for display_name, _, _ in METHODS:
        method_cell = display_name.replace("%", r"\%")
        if display_name.endswith("(Ours)"):
            method_cell = rf"\textbf{{{method_cell}}}"
        cells = [method_cell]
        for label, key, _, _ in METRICS:
            cell = latex_value(label, values[display_name][key])
            if winners[key] == display_name:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        latex_rows.append("    " + " & ".join(cells) + r" \\")

    latex = "\n".join(
        [
            "% Generated from paper-freeze-v1; requires \\usepackage{booktabs}.",
            r"\begin{table*}[t]",
            r"  \centering",
            r"  \caption{Controlled formal-test comparison on the fixed 50-image project-test manifest. All methods start from the same crop-trained source checkpoint and use the same continuation budget.}",
            r"  \label{tab:main_results}",
            r"  \scriptsize",
            r"  \setlength{\tabcolsep}{3.0pt}",
            r"  \renewcommand{\arraystretch}{1.10}",
            r"  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}",
            r"    \toprule",
            r"    Method & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & Top-25 local PSNR $\uparrow$ & P95 MSE $\downarrow$ & Gini $\downarrow$ & BER@30\% $\downarrow$ \\",
            r"    \midrule",
            *latex_rows,
            r"    \bottomrule",
            r"  \end{tabular*}",
            r"  \vspace{2pt}",
            r"  \parbox{\textwidth}{\scriptsize P16 = 16$\times$16 training patch; S8/S16 = stride; Top-$k$ = highest raw-MSE patch fraction selected during training.}",
            r"\end{table*}",
            "",
        ]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(latex)

    report_lines = [
        "# Table I frozen-evidence check",
        "",
        "## Evidence boundary",
        "",
        f"- Protocol: `{provenance['protocol']}`; `training=false`.",
        f"- Primary 50-image evidence: `{FROZEN}`.",
        f"- Primary evidence SHA-256: `{sha256(FROZEN)}` (matches `paper/evidence_manifest.json`).",
        f"- Frozen aggregate cross-check: `{AGGREGATED}`.",
        f"- Aggregate SHA-256: `{sha256(AGGREGATED)}` (matches `paper/evidence_manifest.json`).",
        "- Legacy metrics were not read or used.",
        "- PSNR is recomputed as `10 log10(1 / mean(global_mse))`; SSIM, LPIPS, Top-25 local PSNR, P95 MSE, Gini, and BER@30% are means over the 50 frozen per-image rows.",
        "",
        "## Cell-by-cell verification",
        "",
        "Every cell uses the primary path `/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv` and is independently matched to the corresponding run/column in `/root/workspace/GLX/icassp/MBRS/paper/main_table.csv`.",
        "",
        "| Method | Cell | Source column / aggregation | Verified value | Expected value | Status |",
        "|---|---|---|---:|---:|:---:|",
    ]
    for method, label, key, verified, expected, status in checks:
        aggregation = "10 log10(1 / mean(global_mse))" if key == "global_psnr" else f"mean({key}), n=50"
        report_lines.append(
            f"| {method} | {label} | `{FROZEN}` -> `{aggregation}` | "
            f"{displayed(label, verified)} | {displayed(label, expected)} | **{status}** |"
        )
    report_lines.extend(
        [
            "",
            "## Column-best audit",
            "",
            "| Column | True best method | Value |",
            "|---|---|---:|",
        ]
    )
    for label, key, _, _ in METRICS:
        winner = winners[key]
        report_lines.append(f"| {label} | {winner} | {displayed(label, values[winner][key])} |")
    report_lines.extend(
        [
            "",
            "Only the true best numeric cell in each column is bold. The Ours method name is bold, but its PSNR, SSIM, and LPIPS values are not bold because they are not column-best.",
            "",
            "## Formatting checks",
            "",
            "- Uses `table*`, `booktabs`, `tabular*`, and `\\textwidth`.",
            "- Caption is above the table and the label is `tab:main_results`.",
            "- Uses `\\scriptsize`; no `resizebox` and no vertical rules.",
            "- The notation footnote defines P16, S8/S16, and Top-k.",
            "",
        ]
    )
    REPORT.write_text("\n".join(report_lines))

    print(OUTPUT)
    print(REPORT)


if __name__ == "__main__":
    main()
