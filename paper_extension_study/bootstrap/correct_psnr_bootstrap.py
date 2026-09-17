#!/usr/bin/env python3
"""Corrected paired bootstrap using the frozen paper's aggregate-MSE PSNR estimator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


GLOBAL = "MBRS crop-trained global"
OURS = "Ours"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def load_mse(path: Path) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, dict[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["method"], {})[int(row["image_index"])] = float(row["global_mse"])
    expected = {GLOBAL, OURS}
    if not expected.issubset(grouped):
        raise ValueError(f"missing required Global/Ours rows; found {sorted(grouped)}")
    indexes = sorted(grouped[GLOBAL])
    if indexes != list(range(50)) or sorted(grouped[OURS]) != indexes:
        raise ValueError("expected paired image indexes 0..49")
    return (
        np.array([grouped[GLOBAL][index] for index in indexes], dtype=np.float64),
        np.array([grouped[OURS][index] for index in indexes], dtype=np.float64),
    )


def paper_psnr(mse: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_mse, ours_mse = load_mse(args.input)
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, 50, size=(args.resamples, 50))
    global_boot_mse = global_mse[indices].mean(axis=1)
    ours_boot_mse = ours_mse[indices].mean(axis=1)
    global_boot_psnr = paper_psnr(global_boot_mse)
    ours_boot_psnr = paper_psnr(ours_boot_mse)
    difference = ours_boot_psnr - global_boot_psnr
    observed_global = float(paper_psnr(np.array([global_mse.mean()]))[0])
    observed_ours = float(paper_psnr(np.array([ours_mse.mean()]))[0])
    observed_difference = observed_ours - observed_global
    low, high = np.percentile(difference, [2.5, 97.5])
    standard_error = float(np.std(difference, ddof=1))

    rows = []
    for replicate in range(args.resamples):
        rows.append({
            "replicate": replicate,
            "global_bootstrap_mse": global_boot_mse[replicate],
            "ours_bootstrap_mse": ours_boot_mse[replicate],
            "global_paper_psnr": global_boot_psnr[replicate],
            "ours_paper_psnr": ours_boot_psnr[replicate],
            "ours_minus_global_paper_psnr": difference[replicate],
        })
    with (args.output_dir / "bootstrap_psnr_paper_estimator.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output_dir / "bootstrap_psnr_paper_estimator.npz",
        global_bootstrap_mse=global_boot_mse,
        ours_bootstrap_mse=ours_boot_mse,
        global_paper_psnr=global_boot_psnr,
        ours_paper_psnr=ours_boot_psnr,
        ours_minus_global_paper_psnr=difference,
    )
    metadata = {
        "input": str(args.input),
        "comparison": "Ours - MBRS crop-trained global",
        "n_images": 50,
        "n_bootstrap": args.resamples,
        "analysis_seed": args.seed,
        "unit_of_resampling": "paired image index",
        "paper_estimator": "10*log10(1 / mean_i(global_mse_i))",
        "global_mse_source_field": "global_mse from clipped RGB per-image evaluator",
        "difference_estimator": "paper_psnr(ours bootstrap mean MSE) - paper_psnr(global bootstrap mean MSE)",
        "original_other_metric_bootstraps_unchanged": True,
        "project_test_used_for_selection": False,
    }
    (args.output_dir / "bootstrap_psnr_paper_estimator_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary = f"""# Corrected PSNR Bootstrap Using the Frozen Paper Estimator

## Estimator

The frozen paper computes one PSNR per method from the aggregate clipped-RGB MSE:

```text
MSE_paper = mean_i(global_mse_i)
PSNR_paper = 10 log10(1 / MSE_paper)
```

For each of 20,000 paired image-index bootstrap replicates, this analysis resamples the 50 Global/Ours image pairs, computes each method's mean `global_mse`, applies the formula above, and then takes `Ours PSNR − Global PSNR`. It does not modify the original bootstrap outputs for any other metric.

- Input per-image CSV: `{args.input}`
- Resamples: `{args.resamples:,}`
- Analysis seed: `{args.seed}`
- Observed Global PSNR: `{observed_global:.12f} dB`
- Observed Ours PSNR: `{observed_ours:.12f} dB`
- Observed difference: `{observed_difference:.12f} dB`
- 95% percentile interval: `[{low:.12f}, {high:.12f}] dB`
- Bootstrap standard error: `{standard_error:.12f} dB`
- Interval crosses zero: `{'yes' if low <= 0 <= high else 'no'}`

## Exact reason for the prior mismatch

The original `bootstrap_analysis.py` loads the `global_psnr` field from the per-image CSV and computes `mean_i(PSNR_ours,i − PSNR_global,i)`. That is the mean of per-image PSNR differences. The frozen paper path in `experiments/freeze_paper_evidence.py`, function `summarize`, first averages the per-image `global_mse` values and then overwrites `global_psnr` with `-10*log10(mean_i(global_mse_i))`. These estimators are nonlinear and therefore differ numerically.

Both paths use the same clipped RGB range derived from normalized tensors, the same 128×128×3 image dimensions, and equal image weights. Because every image has the same dimensions, the paper's mean per-image MSE is also the pixel-weighted aggregate MSE over the 50 images. The mismatch is not caused by clipping, normalization, image size, or unequal pixel counts.

## Compatibility

The original PSNR bootstrap interval is valid for the distinct estimand “mean per-image PSNR difference,” but it is not the exact frozen-paper PSNR estimand. The corrected files in this directory are the compatible PSNR bootstrap result. All original bootstrap intervals for Top-25 Local PSNR, P95, P99, LPIPS, CIEDE2000, Gini, and BER remain unchanged.
"""
    (args.output_dir / "bootstrap_psnr_paper_estimator_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({
        "observed_global_psnr": observed_global,
        "observed_ours_psnr": observed_ours,
        "observed_difference": observed_difference,
        "ci_2_5": float(low),
        "ci_97_5": float(high),
        "bootstrap_standard_error": standard_error,
        "ci_crosses_zero": bool(low <= 0 <= high),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
