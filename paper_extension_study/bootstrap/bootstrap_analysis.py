#!/usr/bin/env python3
"""Paired image-level bootstrap for the frozen project-test per-image CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


COMPARISONS = {
    "global_vs_ours": ("MBRS crop-trained global", "Ours"),
    "global_vs_hard": ("MBRS crop-trained global", "Hard Local-Tail"),
    "hard_vs_ours": ("Hard Local-Tail", "Ours"),
}

QUALITY_METRICS = {
    "PSNR": "global_psnr",
    "Top-25 Local PSNR": "top25_local_psnr",
    "P95 patch MSE": "patch_mse_p95",
    "P99 patch MSE": "patch_mse_p99",
    "LPIPS": "full_lpips",
    "Global CIEDE2000": "ciede2000_global",
    "Top10 CIEDE2000": "ciede2000_top10",
    "Gini": "gini",
}
BER_METRICS = {
    "BER100": "ber100",
    "BER70": "ber70",
    "BER50": "ber50",
    "BER40": "ber40",
    "BER30": "ber30",
}
LOWER_IS_BETTER = {
    "P95 patch MSE",
    "P99 patch MSE",
    "LPIPS",
    "Global CIEDE2000",
    "Top10 CIEDE2000",
    "Gini",
    "BER100",
    "BER70",
    "BER50",
    "BER40",
    "BER30",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, dict[int, dict[str, float]]] = {}
    for row in rows:
        method = row["method"]
        index = int(row["image_index"])
        if index in grouped.setdefault(method, {}):
            raise ValueError(f"duplicate image index {index} for {method}")
        grouped[method][index] = {key: float(value) for key, value in row.items() if key not in {"method", "run", "image_index"}}
    if set(grouped) != {"MBRS crop-trained global", "Hard Local-Tail", "Ours"}:
        raise ValueError(f"unexpected methods: {sorted(grouped)}")
    indexes = sorted(next(iter(grouped.values())))
    if len(indexes) != 50 or indexes != list(range(50)):
        raise ValueError(f"expected image indexes 0..49, got {indexes}")
    for method, method_rows in grouped.items():
        if sorted(method_rows) != indexes:
            raise ValueError(f"image pairing mismatch for {method}")
    return grouped


def ci_record(
    comparison: str,
    first: str,
    second: str,
    metric: str,
    values: np.ndarray,
    bootstrap: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "comparison": comparison,
        "first_method": first,
        "second_method": second,
        "difference": f"{second} - {first}",
        "metric": metric,
        "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
        "n_images": int(values.shape[0]),
        "n_bootstrap": resamples,
        "analysis_seed": seed,
        "observed_mean_difference": float(np.mean(values)),
        "ci_2_5_percentile": float(low),
        "ci_97_5_percentile": float(high),
        "bootstrap_standard_error": float(np.std(bootstrap, ddof=1)),
        "ci_crosses_zero": bool(low <= 0.0 <= high),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_rows(args.input)
    indexes = np.arange(50)
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.integers(0, len(indexes), size=(args.resamples, len(indexes)))

    all_metrics = {**QUALITY_METRICS, **BER_METRICS}
    records: list[dict[str, object]] = []
    distributions: dict[str, np.ndarray] = {}
    for comparison, (first, second) in COMPARISONS.items():
        for metric, field in all_metrics.items():
            first_values = np.array([grouped[first][int(i)][field] for i in indexes], dtype=np.float64)
            second_values = np.array([grouped[second][int(i)][field] for i in indexes], dtype=np.float64)
            paired_difference = second_values - first_values
            bootstrap = paired_difference[sample_indices].mean(axis=1)
            key = f"{comparison}__{metric.replace(' ', '_').replace('-', '_')}"
            distributions[key] = bootstrap
            records.append(ci_record(comparison, first, second, metric, paired_difference, bootstrap, args.resamples, args.seed))

    with (args.out_dir / "bootstrap_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(records[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(args.out_dir / "bootstrap_distributions.npz", **distributions)
    metadata = {
        "input": str(args.input),
        "input_is_authoritative_bundle_provenance": True,
        "unit_of_resampling": "project-test image index",
        "n_images": 50,
        "n_bootstrap": args.resamples,
        "analysis_seed": args.seed,
        "difference_convention": "second_method_minus_first_method",
        "comparisons": {name: list(pair) for name, pair in COMPARISONS.items()},
        "quality_metrics": QUALITY_METRICS,
        "ber_metrics": BER_METRICS,
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "ci_method": "percentile interval from paired image-level bootstrap distribution",
        "individual_bits_are_not_bootstrap_units": True,
        "project_test_used_for_hyperparameter_selection": False,
        "paper_bundle_unchanged": True,
    }
    (args.out_dir / "bootstrap_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Paired Image-Level Bootstrap Summary",
        "",
        "This supplementary analysis reuses the frozen 50-image project-test per-image output explicitly referenced by the authoritative bundle. It does not rerun training or select any hyperparameter. Each bootstrap resample samples image indices with replacement and preserves method pairing; individual bits are never treated as IID bootstrap units.",
        "",
        f"- Resamples: `{args.resamples:,}`",
        f"- Analysis seed: `{args.seed}`",
        "- Difference convention: `second method − first method`.",
        "- For higher-is-better metrics, positive differences favor the second method; for lower-is-better metrics, negative differences favor the second method.",
        "- The interval wording below is descriptive: an interval either excludes or includes zero. No p-value or generic statistical-significance claim is made.",
        "",
        "## Results",
        "",
        "| Comparison | Metric | Observed difference | 2.5 percentile | 97.5 percentile | Bootstrap SE | CI crosses zero | Direction |",
        "|---|---|---:|---:|---:|---:|:---:|---|",
    ]
    for record in records:
        lines.append(
            "| {comparison} | {metric} | {observed_mean_difference:.9g} | {ci_2_5_percentile:.9g} | {ci_97_5_percentile:.9g} | {bootstrap_standard_error:.9g} | {crosses} | {direction} |".format(
                comparison=record["comparison"],
                metric=record["metric"],
                observed_mean_difference=record["observed_mean_difference"],
                ci_2_5_percentile=record["ci_2_5_percentile"],
                ci_97_5_percentile=record["ci_97_5_percentile"],
                bootstrap_standard_error=record["bootstrap_standard_error"],
                crosses="yes" if record["ci_crosses_zero"] else "no",
                direction=record["direction"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The observed differences and paired bootstrap intervals quantify image-level stability of the frozen comparisons. They do not alter the frozen method, replace the formal project-test table, establish universal superiority, or authorize retuning. Use the exact metric-space labels from the paper evidence audit when citing these results.",
            "",
        ]
    )
    (args.out_dir / "bootstrap_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
