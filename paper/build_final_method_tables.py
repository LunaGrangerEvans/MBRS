#!/usr/bin/env python3
"""Build camera-ready table fragments from the frozen formal result."""

from __future__ import annotations

import csv
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PER_IMAGE = PROJECT / "reports/final_ours_project_test_per_image.csv"
OUT = PROJECT / "paper/tables"
REPORT = PROJECT / "reports/final_method_tables_check.md"
METHODS = ("MBRS crop-trained global", "Hard Local-Tail", "Ours")


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def aggregate(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "psnr": -10.0 * math.log10(mean(rows, "global_mse")),
        "ssim": mean(rows, "global_ssim"),
        "lpips": mean(rows, "full_lpips"),
        "local_psnr": mean(rows, "top25_local_psnr"),
        "p95": mean(rows, "patch_mse_p95"),
        "p99": mean(rows, "patch_mse_p99"),
        "gini": mean(rows, "gini"),
        "ciede_global": mean(rows, "ciede2000_global"),
        "ciede_top10": mean(rows, "ciede2000_top10"),
        "ber30": mean(rows, "ber30"),
    }


def main() -> None:
    rows = list(csv.DictReader(PER_IMAGE.open(newline="")))
    values = {method: aggregate([row for row in rows if row["method"] == method]) for method in METHODS}
    OUT.mkdir(parents=True, exist_ok=True)
    ablation = "\n".join([
        "% Generated from the frozen one-pass formal project-test evaluation; requires booktabs.",
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Frozen formal project-test ablation on the fixed 50-image manifest. The progression is global RGB reconstruction, Hard Local-Tail supervision, and the final Ours method with global OKLab regularization.}",
        r"  \label{tab:final_method_ablation}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}",
        r"    \toprule",
        r"    Method & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & Top-25 local PSNR $\uparrow$ & P95 MSE $\downarrow$ & Gini $\downarrow$ & BER@30 $\downarrow$ \\",
        r"    \midrule",
        f"    MBRS crop-trained global & {values['MBRS crop-trained global']['psnr']:.4f} & {values['MBRS crop-trained global']['ssim']:.4f} & {values['MBRS crop-trained global']['lpips']:.7f} & {values['MBRS crop-trained global']['local_psnr']:.4f} & {values['MBRS crop-trained global']['p95']:.3e} & {values['MBRS crop-trained global']['gini']:.4f} & {values['MBRS crop-trained global']['ber30']:.5f} " + r"\\",
        f"    Hard Local-Tail & {values['Hard Local-Tail']['psnr']:.4f} & {values['Hard Local-Tail']['ssim']:.4f} & {values['Hard Local-Tail']['lpips']:.7f} & {values['Hard Local-Tail']['local_psnr']:.4f} & {values['Hard Local-Tail']['p95']:.3e} & {values['Hard Local-Tail']['gini']:.4f} & {values['Hard Local-Tail']['ber30']:.5f} " + r"\\",
        rf"    \textbf{{Ours}} & \textbf{{{values['Ours']['psnr']:.4f}}} & \textbf{{{values['Ours']['ssim']:.4f}}} & \textbf{{{values['Ours']['lpips']:.7f}}} & \textbf{{{values['Ours']['local_psnr']:.4f}}} & \textbf{{{values['Ours']['p95']:.3e}}} & {values['Ours']['gini']:.4f} & {values['Ours']['ber30']:.5f} " + r"\\",
        r"    \bottomrule",
        r"  \end{tabular*}",
        r"  \vspace{2pt}",
        r"  \parbox{\textwidth}{\scriptsize P95 is the mean of per-image P95 values over the native 32$\times$32 evaluation grid; BER is raw 64-bit BER under the fixed rectangle-mask protocol.}",
        r"\end{table*}",
        "",
    ])
    color = "\n".join([
        "% Generated from the frozen one-pass formal project-test evaluation; requires booktabs.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Formal project-test color-fidelity comparison for the intermediate Hard Local-Tail method and final Ours method.}",
        r"  \label{tab:final_method_color}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    Method & Global CIEDE2000 $\downarrow$ & Top10 CIEDE2000 $\downarrow$ & PSNR $\uparrow$ & Local PSNR $\uparrow$ & LPIPS $\downarrow$ & BER30 $\downarrow$ \\",
        r"    \midrule",
        f"    Hard Local-Tail & {values['Hard Local-Tail']['ciede_global']:.4f} & {values['Hard Local-Tail']['ciede_top10']:.4f} & {values['Hard Local-Tail']['psnr']:.4f} & {values['Hard Local-Tail']['local_psnr']:.4f} & {values['Hard Local-Tail']['lpips']:.7f} & {values['Hard Local-Tail']['ber30']:.5f} " + r"\\",
        rf"    \textbf{{Ours}} & \textbf{{{values['Ours']['ciede_global']:.4f}}} & \textbf{{{values['Ours']['ciede_top10']:.4f}}} & \textbf{{{values['Ours']['psnr']:.4f}}} & \textbf{{{values['Ours']['local_psnr']:.4f}}} & \textbf{{{values['Ours']['lpips']:.7f}}} & {values['Ours']['ber30']:.5f} " + r"\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])
    (OUT / "final_method_ablation.tex").write_text(ablation, encoding="utf-8")
    (OUT / "final_method_color.tex").write_text(color, encoding="utf-8")
    report = "\n".join([
        "# Final-method table check",
        "",
        f"- Source: `{PER_IMAGE}`; 150 rows, 50 per method.",
        "- Table 1 fragment: `paper/tables/final_method_ablation.tex`.",
        "- Color fragment: `paper/tables/final_method_color.tex`.",
        "- The existing historical Table I fragments were not overwritten; these are the candidate final-method snippets from the new frozen evidence.",
        "- Exact reader-facing names: MBRS crop-trained global, Hard Local-Tail, Ours.",
    ])
    REPORT.write_text(report + "\n", encoding="utf-8")
    print(OUT / "final_method_ablation.tex")
    print(OUT / "final_method_color.tex")


if __name__ == "__main__":
    main()
