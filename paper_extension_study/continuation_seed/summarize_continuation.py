#!/usr/bin/env python3
"""Summarize continuation-seed sensitivity from common-source endpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_extension_study/continuation_seed"
METHOD_DIRS = {"Global": "global", "Hard Local-Tail": "hard", "Ours": "ours"}
SEEDS = (17, 23, 42)
METRICS = {
    "PSNR": "global_psnr",
    "Top-25 Local PSNR": "top25_local_psnr",
    "P95": "patch_mse_p95",
    "P99": "patch_mse_p99",
    "LPIPS": "full_lpips",
    "Global CIEDE2000": "ciede2000_global",
    "Top10 CIEDE2000": "ciede2000_top10",
    "Gini": "gini",
    "BER30": "ber30",
}
LOWER = {"P95", "P99", "LPIPS", "Global CIEDE2000", "Top10 CIEDE2000", "Gini", "BER30"}


def load_values() -> dict[str, dict[int, dict[str, float]]]:
    values = {}
    for method, directory in METHOD_DIRS.items():
        values[method] = {}
        for seed in SEEDS:
            path = OUT / directory / f"seed{seed}" / "project_test_metrics.json"
            data = json.loads(path.read_text())
            values[method][seed] = {metric: float(data[field]) for metric, field in METRICS.items()}
    return values


def fmt(value: float, metric: str) -> str:
    if metric in {"P95", "P99"}:
        return f"{value:.9e}"
    if metric in {"LPIPS", "BER30"}:
        return f"{value:.8f}"
    return f"{value:.6f}"


values = load_values()
summary_fields = ["Method", "Seed", *METRICS]
summary_rows = []
for method in METHOD_DIRS:
    for seed in SEEDS:
        summary_rows.append({"Method": method, "Seed": seed, **values[method][seed]})
with (OUT / "continuation_seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)

effects = {
    "Ours - Global": ("Ours", "Global"),
    "Hard - Global": ("Hard Local-Tail", "Global"),
    "Ours - Hard": ("Ours", "Hard Local-Tail"),
}
effect_rows = []
for effect, (second, first) in effects.items():
    for seed in SEEDS:
        effect_rows.append({"Effect": effect, "Seed": seed, **{metric: values[second][seed][metric] - values[first][seed][metric] for metric in METRICS}})
with (OUT / "continuation_seed_effects.csv").open("w", newline="", encoding="utf-8") as handle:
    fields = ["Effect", "Seed", *METRICS]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(effect_rows)

lines = [
    "# Stochastic Continuation-Seed Sensitivity Report",
    "",
    "This report measures stochastic continuation-seed sensitivity from the same frozen seed17 epoch-100 crop-trained Global source checkpoint. It is not full end-to-end multi-seed reproducibility. Seed17 reuses the valid frozen endpoint; seeds23 and42 are new continuations from the identical source.",
    "",
    "- Seeds: `17, 23, 42`.",
    "- Source SHA-256: `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907`.",
    "- All methods: 20 epochs, `lr=1e-4`, batch16, workers0, `RandomCrop(0.3,1.0)`, restored model/BatchNorm/Adam state, deterministic settings.",
    "- No seed, checkpoint, or hyperparameter was selected using project-test performance.",
    "- All values below are natural source-resolution 128×128 project-test metrics.",
    "",
    "## Individual seed values",
    "",
    "| Method | Seed | PSNR | Top-25 Local PSNR | P95 | P99 | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini | BER30 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in summary_rows:
    lines.append("| " + row["Method"] + " | " + str(row["Seed"]) + " | " + " | ".join(fmt(row[metric], metric) for metric in METRICS) + " |")

lines.extend(["", "## Mean ± standard deviation across continuation seeds", "", "| Method | " + " | ".join(METRICS) + " |", "|---|" + "---:|" * len(METRICS)])
for method in METHOD_DIRS:
    cells = []
    for metric in METRICS:
        array = np.array([values[method][seed][metric] for seed in SEEDS])
        cells.append(f"{fmt(float(array.mean()), metric)} ± {fmt(float(array.std(ddof=1)), metric)}")
    lines.append("| " + method + " | " + " | ".join(cells) + " |")

lines.extend(["", "## Paired method differences by continuation seed", "", "Differences are computed within each continuation seed. For PSNR and local PSNR, positive favors the first method; for lower-is-better metrics, negative favors the first method.", ""])
for effect, (second, first) in effects.items():
    lines.append(f"### {effect}")
    lines.append("")
    lines.append("| Seed | " + " | ".join(METRICS) + " |")
    lines.append("|---:|" + "---:|" * len(METRICS))
    for seed in SEEDS:
        row = next(row for row in effect_rows if row["Effect"] == effect and row["Seed"] == seed)
        lines.append("| " + str(seed) + " | " + " | ".join(fmt(row[metric], metric) for metric in METRICS) + " |")
    lines.append("")

primary = ("PSNR", "Top-25 Local PSNR", "P95", "P99", "LPIPS", "Global CIEDE2000", "Top10 CIEDE2000")
for effect, (second, first) in (("Ours - Global", effects["Ours - Global"]), ("Hard - Global", effects["Hard - Global"]), ("Ours - Hard", effects["Ours - Hard"])):
    consistent = []
    for metric in primary:
        deltas = [values[second][seed][metric] - values[first][seed][metric] for seed in SEEDS]
        favorable = all(delta > 0 for delta in deltas) if metric not in LOWER else all(delta < 0 for delta in deltas)
        if favorable:
            consistent.append(metric)
    lines.append(f"For `{effect}`, the favorable direction is consistent across all three continuation seeds for: {', '.join(consistent)}.")
    lines.append("")

lines.extend([
    "## Interpretation",
    "",
    "The primary fidelity trend for Ours versus Global persists across all three stochastic continuation seeds from a common frozen source checkpoint: PSNR and Top-25 Local PSNR are higher, while P95, P99, LPIPS, and CIEDE2000 are lower for Ours in every seed. This supports the cautious sentence: **“The fidelity trend persists across stochastic continuation seeds from a common frozen source checkpoint.”**",
    "",
    "This consistency does not extend to every diagnostic. Gini is mixed for Ours versus Global, and BER30 is mixed (`0`, positive, and negative small differences across seeds). No BER or Gini superiority claim is made. The report also does not claim full end-to-end reproducibility because the source model was held fixed.",
    "",
    "The full per-seed rows are in `continuation_seed_summary.csv`; paired deltas are in `continuation_seed_effects.csv`.",
    "",
])
(OUT / "CONTINUATION_SEED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
