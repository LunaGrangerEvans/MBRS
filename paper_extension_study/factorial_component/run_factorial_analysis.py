#!/usr/bin/env python3
"""Construct and bootstrap the fixed-RGB-weight 2x2 Tail/OKLab factorial."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_extension_study/factorial_component"
FROZEN_PER_IMAGE = ROOT / "reports/final_ours_project_test_per_image.csv"
CONTROL_RUNS = {
    "RGB0.5-Control": OUT.parent / "component_ablation/rgb05_control/project_test_per_image.csv",
    "Global+OKLab-only": OUT.parent / "component_ablation/global_oklab_only/project_test_per_image.csv",
}
METHODS = ("RGB0.5-Control", "Global+OKLab-only", "Hard Local-Tail", "Ours")
FIELDS = {
    "PSNR": "global_mse",
    "Top-25 Local PSNR": "top25_local_psnr",
    "P95 MSE": "patch_mse_p95",
    "P99 MSE": "patch_mse_p99",
    "LPIPS": "full_lpips",
    "Global CIEDE2000": "ciede2000_global",
    "Top10 CIEDE2000": "ciede2000_top10",
    "Gini": "gini",
    "BER30": "ber30",
}
LOWER = {"P95 MSE", "P99 MSE", "LPIPS", "Global CIEDE2000", "Top10 CIEDE2000", "Gini", "BER30"}
EFFECTS = {
    "OKLab_without_Tail": ("M01 - M00", ("Global+OKLab-only", "RGB0.5-Control")),
    "OKLab_with_Tail": ("M11 - M10", ("Ours", "Hard Local-Tail")),
    "Tail_without_OKLab": ("M10 - M00", ("Hard Local-Tail", "RGB0.5-Control")),
    "Tail_with_OKLab": ("M11 - M01", ("Ours", "Global+OKLab-only")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def read_rows(path: Path, source_method: str | None = None) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        method = source_method or row["method"]
        if method not in {"Hard Local-Tail", "Ours"} and source_method is None:
            continue
        item = {"method": method, "image_index": int(row["image_index"])}
        for field in set(FIELDS.values()):
            item[field] = float(row[field])
        output.append(item)
    return output


def load_input() -> dict[str, dict[int, dict[str, float]]]:
    grouped: dict[str, dict[int, dict[str, float]]] = {method: {} for method in METHODS}
    for row in read_rows(CONTROL_RUNS["RGB0.5-Control"], "RGB0.5-Control") + read_rows(CONTROL_RUNS["Global+OKLab-only"], "Global+OKLab-only") + read_rows(FROZEN_PER_IMAGE):
        method = str(row["method"])
        if method not in grouped:
            continue
        index = int(row["image_index"])
        if index in grouped[method]:
            raise ValueError(f"duplicate {method} image index {index}")
        grouped[method][index] = {field: float(row[field]) for field in FIELDS.values()}
    for method in METHODS:
        if sorted(grouped[method]) != list(range(50)):
            raise ValueError(f"{method} does not have paired indexes 0..49")
    with (OUT / "factorial_input_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["method", "image_index", *FIELDS.values()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHODS:
            for index in range(50):
                writer.writerow({"method": method, "image_index": index, **grouped[method][index]})
    return grouped


def metric_matrix(grouped: dict[str, dict[int, dict[str, float]]], metric: str, indexes: np.ndarray | None = None) -> dict[str, np.ndarray]:
    indexes = np.arange(50) if indexes is None else indexes
    field = FIELDS[metric]
    if metric == "PSNR":
        return {method: 10.0 * np.log10(1.0 / np.maximum(np.array([grouped[method][int(i)][field] for i in indexes]).mean(), 1e-12)) for method in METHODS}
    return {method: np.array([grouped[method][int(i)][field] for i in indexes]).mean() for method in METHODS}


def observed_effect(metric: str, grouped: dict[str, dict[int, dict[str, float]]], indices: np.ndarray) -> float:
    values = metric_matrix(grouped, metric, indices)
    return float(values["Ours"] - values["Hard Local-Tail"] - values["Global+OKLab-only"] + values["RGB0.5-Control"])


def simple_observed(metric: str, grouped: dict[str, dict[int, dict[str, float]]], second: str, first: str) -> float:
    values = metric_matrix(grouped, metric)
    return float(values[second] - values[first])


def make_point_summary(grouped: dict[str, dict[int, dict[str, float]]]) -> None:
    fields = ["Method", "Tail", "OKLab", *FIELDS.keys()]
    rows = []
    factors = {"RGB0.5-Control": (0, 0), "Global+OKLab-only": (0, 1), "Hard Local-Tail": (1, 0), "Ours": (1, 1)}
    for method in METHODS:
        values = metric_matrix(grouped, "PSNR")
        row = {"Method": method, "Tail": factors[method][0], "OKLab": factors[method][1]}
        row["PSNR"] = values[method]
        for metric in list(FIELDS)[1:]:
            row[metric] = metric_matrix(grouped, metric)[method]
        rows.append(row)
    with (OUT / "factorial_component_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Fixed-RGB-Weight Factorial Component Summary",
        "",
        "Only the four seed17 continuation branches with RGB/global weight 0.5 are included. The frozen Global row with RGB weight 1.0 is intentionally excluded from this factorial design. Values are natural 128×128 project-test results; the formal frozen Hard/Ours rows come from the authoritative bundle, while the two control rows come from the completed extension runs.",
        "",
        "| Method | Tail | OKLab | PSNR | Top-25 Local PSNR | P95 MSE | P99 MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini | BER30 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + row["Method"] + " | " + " | ".join([
            str(row["Tail"]), str(row["OKLab"]), f"{row['PSNR']:.6f}", f"{row['Top-25 Local PSNR']:.6f}",
            f"{row['P95 MSE']:.9e}", f"{row['P99 MSE']:.9e}", f"{row['LPIPS']:.8f}", f"{row['Global CIEDE2000']:.6f}",
            f"{row['Top10 CIEDE2000']:.6f}", f"{row['Gini']:.6f}", f"{row['BER30']:.7f}",
        ]) + " |")
    lines.extend(["", "## Simple effects", "", "Effects are second cell minus first cell; for lower-is-better metrics, negative is favorable.", ""])
    for effect, (contrast, (second, first)) in EFFECTS.items():
        lines.append(f"- `{effect}` ({contrast}): `{second} − {first}`.")
        for metric in FIELDS:
            delta = simple_observed(metric, grouped, second, first)
            lines.append(f"  - {metric}: `{delta:+.9e}`." if metric in {"P95 MSE", "P99 MSE"} else f"  - {metric}: `{delta:+.9f}`.")
    lines.append("")
    (OUT / "factorial_component_summary.md").write_text("\n".join(lines), encoding="utf-8")


def bootstrap(grouped: dict[str, dict[int, dict[str, float]]], resamples: int, seed: int) -> None:
    bootstrap_dir = OUT / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 50, size=(resamples, 50))
    records = []
    distributions = {}
    for metric in FIELDS:
        # Build method-by-replicate values with the paper PSNR estimator.
        method_values: dict[str, np.ndarray] = {}
        for method in METHODS:
            field = FIELDS[metric]
            values = np.array([grouped[method][i][field] for i in range(50)], dtype=np.float64)
            if metric == "PSNR":
                method_values[method] = 10.0 * np.log10(1.0 / np.maximum(values[indices].mean(axis=1), 1e-12))
            else:
                method_values[method] = values[indices].mean(axis=1)
        effect_arrays = {
            "OKLab_without_Tail": method_values["Global+OKLab-only"] - method_values["RGB0.5-Control"],
            "OKLab_with_Tail": method_values["Ours"] - method_values["Hard Local-Tail"],
            "Tail_without_OKLab": method_values["Hard Local-Tail"] - method_values["RGB0.5-Control"],
            "Tail_with_OKLab": method_values["Ours"] - method_values["Global+OKLab-only"],
            "interaction": method_values["Ours"] - method_values["Hard Local-Tail"] - method_values["Global+OKLab-only"] + method_values["RGB0.5-Control"],
        }
        for effect, values in effect_arrays.items():
            key = f"{effect}__{metric.replace(' ', '_').replace('-', '_')}"
            distributions[key] = values
            low, high = np.percentile(values, [2.5, 97.5])
            if effect == "interaction":
                contrast = "M11 - M10 - M01 + M00"
                observed = observed_effect(metric, grouped, np.arange(50))
            else:
                contrast, (second, first) = EFFECTS[effect]
                observed = simple_observed(metric, grouped, second, first)
            records.append({
                "effect": effect,
                "contrast": contrast,
                "metric": metric,
                "direction": "lower_is_better" if metric in LOWER else "higher_is_better",
                "n_images": 50,
                "n_bootstrap": resamples,
                "analysis_seed": seed,
                "observed_estimate": observed,
                "ci_2_5_percentile": float(low),
                "ci_97_5_percentile": float(high),
                "bootstrap_standard_error": float(np.std(values, ddof=1)),
                "ci_includes_zero": bool(low <= 0 <= high),
            })
    with (bootstrap_dir / "factorial_bootstrap_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(bootstrap_dir / "factorial_bootstrap_distributions.npz", **distributions)
    metadata = {
        "input": str(OUT / "factorial_input_per_image.csv"),
        "n_images": 50,
        "n_bootstrap": resamples,
        "analysis_seed": seed,
        "unit_of_resampling": "paired image index",
        "psnr_estimator": "10*log10(1 / mean_i(global_mse_i)) per bootstrap replicate",
        "metrics": list(FIELDS),
        "effects": {name: contrast for name, (contrast, _) in EFFECTS.items()} | {"interaction": "M11 - M10 - M01 + M00"},
        "individual_bits_are_not_bootstrap_units": True,
        "project_test_used_for_selection": False,
        "paper_bundle_unchanged": True,
    }
    (bootstrap_dir / "factorial_bootstrap_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Factorial Component Bootstrap Summary",
        "",
        "The four fixed-RGB-weight cells are bootstrapped at the paired image level using the corrected paper PSNR estimator. For PSNR, each replicate recomputes aggregate mean MSE per cell and then converts it to PSNR. Other metrics use their documented image-level values; BER30 resamples per-image BER contributions.",
        "",
        f"- Resamples: `{resamples:,}`",
        f"- Analysis seed: `{seed}`",
        "- Effects: OKLab without Tail, OKLab with Tail, Tail without OKLab, Tail with OKLab, and the interaction contrast.",
        "- Interval language is limited to “interval excludes/includes zero.”",
        "",
        "| Effect | Metric | Observed | 2.5 percentile | 97.5 percentile | Bootstrap SE | Interval includes zero? |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    for record in records:
        lines.append(f"| {record['effect']} | {record['metric']} | {float(record['observed_estimate']):.9g} | {float(record['ci_2_5_percentile']):.9g} | {float(record['ci_97_5_percentile']):.9g} | {float(record['bootstrap_standard_error']):.9g} | {'Yes' if record['ci_includes_zero'] else 'No'} |")
    lines.extend(["", "The factorial bootstrap quantifies uncertainty in cell contrasts; it does not change the frozen method, establish universal superiority, or justify the word “synergy” by itself.", ""])
    (bootstrap_dir / "factorial_bootstrap_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    grouped = load_input()
    make_point_summary(grouped)
    bootstrap(grouped, args.resamples, args.seed)
    print(json.dumps({"methods": METHODS, "resamples": args.resamples, "seed": args.seed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
