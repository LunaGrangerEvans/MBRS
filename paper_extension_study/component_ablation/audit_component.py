#!/usr/bin/env python3
"""Audit and summarize the registered component-control project-test rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_extension_study/component_ablation"

METRICS = (
    "PSNR", "Top-25 Local PSNR", "P95 MSE", "P99 MSE", "LPIPS",
    "Global CIEDE2000", "Top10 CIEDE2000", "Gini", "BER30",
)

FROZEN = {
    "MBRS crop-trained Global": {
        "PSNR": 36.263172, "Top-25 Local PSNR": 35.522213, "P95 MSE": 3.016531971e-4,
        "P99 MSE": 3.199585360e-4, "LPIPS": 0.00234960, "Global CIEDE2000": 3.535857,
        "Top10 CIEDE2000": 5.159104, "Gini": 0.095007, "BER30": 0.1131250,
    },
    "Hard Local-Tail": {
        "PSNR": 36.442300, "Top-25 Local PSNR": 35.754376, "P95 MSE": 2.844551472e-4,
        "P99 MSE": 3.010726967e-4, "LPIPS": 0.00221518, "Global CIEDE2000": 3.484790,
        "Top10 CIEDE2000": 5.075666, "Gini": 0.086897, "BER30": 0.1129375,
    },
    "Ours": {
        "PSNR": 36.854909, "Top-25 Local PSNR": 36.141895, "P95 MSE": 2.619925590e-4,
        "P99 MSE": 2.782322409e-4, "LPIPS": 0.00191711, "Global CIEDE2000": 3.254091,
        "Top10 CIEDE2000": 4.772229, "Gini": 0.092130, "BER30": 0.1131250,
    },
}


def extension_row(directory: str) -> dict[str, float]:
    data = json.loads((OUT / directory / "project_test_metrics.json").read_text())
    return {
        "PSNR": data["global_psnr"],
        "Top-25 Local PSNR": data["top25_local_psnr"],
        "P95 MSE": data["patch_mse_p95"],
        "P99 MSE": data["patch_mse_p99"],
        "LPIPS": data["full_lpips"],
        "Global CIEDE2000": data["ciede2000_global"],
        "Top10 CIEDE2000": data["ciede2000_top10"],
        "Gini": data["gini"],
        "BER30": data["ber30"],
    }


rows = {
    "MBRS crop-trained Global": FROZEN["MBRS crop-trained Global"],
    "Global-RGB0.5-Control": extension_row("rgb05_control"),
    "Global + OKLab only": extension_row("global_oklab_only"),
    "Hard Local-Tail": FROZEN["Hard Local-Tail"],
    "Ours = Hard Local-Tail + global OKLab": FROZEN["Ours"],
}
baseline = rows["MBRS crop-trained Global"]
delta_names = [f"Delta {metric} vs Global" for metric in METRICS]
fields = ["Method", *METRICS, *delta_names]
with (OUT / "component_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for method, values in rows.items():
        row = {"Method": method}
        row.update(values)
        row.update({f"Delta {metric} vs Global": values[metric] - baseline[metric] for metric in METRICS})
        writer.writerow(row)


def fmt(value: float, metric: str) -> str:
    if metric in {"P95 MSE", "P99 MSE"}:
        return f"{value:.9e}"
    if metric in {"LPIPS", "BER30"}:
        return f"{value:.8f}"
    return f"{value:.6f}"


lines = [
    "# Component Ablation Audit",
    "",
    "This audit uses the authoritative frozen rows from `paper_bundle/` and the two registered seed17 control runs under `paper_extension_study/component_ablation/`. It does not modify the frozen final method, paper bundle, or Prism manuscript.",
    "",
    "## Rows and provenance",
    "",
    "- **MBRS crop-trained Global**, **Hard Local-Tail**, and **Ours**: formal natural project-test rows from `paper_bundle/evidence/final_ours_project_test_report.md` and `paper/paper_evidence_audit.md`; shared controlled source/protocol identity is defined by the bundled frozen-method and continuation reports.",
    "- **Global + OKLab only**: `global_oklab_only/`, registered objective `10 L_msg + 0.5 L_RGB + 0.059149764 L_OKLab`, no local-tail term.",
    "- **Global-RGB0.5-Control**: `rgb05_control/`, registered objective `10 L_msg + 0.5 L_RGB`, no local-tail or OKLab term.",
    "- The formal frozen rows are used as the paper-authoritative comparison. The two new rows are extension evidence and cannot replace Ours.",
    "- Shared source checkpoint: `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth`, SHA-256 `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907`; new branch configs, checkpoints, logs, per-image rows, and split summaries are stored inside the corresponding extension directories.",
    "",
    "## Protocol parity",
    "",
    """| Field | Frozen Global / Hard / Ours | Global + OKLab only | Global-RGB0.5-Control | Difference |
|---|---|---|---|---|
| Source checkpoint | Same seed17 epoch-100 crop-trained Global source under bundled protocol | Same path/hash; restored | Same path/hash; restored | None |
| Continuation | 20 epochs | 20 epochs | 20 epochs | None |
| Learning rate | 1e-4 | 1e-4 | 1e-4 | None |
| Batch / workers | 16 / 0 | 16 / 0 | 16 / 0 | None |
| Crop augmentation | RandomCrop(0.3, 1.0) | Same | Same | None |
| Seed | 17 | 17 | 17 | None |
| Model / BatchNorm / Adam restoration | Restored under bundled protocol | Restored | Restored | None |
| Validation/project-test identity | Fixed messages and fixed crop masks | Same fixed evaluator/manifests | Same fixed evaluator/manifests | None |
| Objective | Global, Hard, or frozen Ours term set | Tail off; OKLab on | Tail off; OKLab off; RGB weight 0.5 | Objective terms only |""",
    "The bundled continuation report also contains a Hard Top-25% lineage branch. That branch is not substituted for the formal frozen Hard/Ours rows here; the final method remains P16/S8 hard Top-10% plus global OKLab.",
    "",
    "## Project-test component summary",
    "",
    "The `component_summary.csv` file contains the requested metrics and deltas relative to MBRS crop-trained Global. Lower is better for P95, P99, LPIPS, CIEDE2000, Gini, and BER30; higher is better for PSNR and Top-25 Local PSNR.",
    "",
    "| Method | PSNR | Top-25 Local PSNR | P95 MSE | P99 MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini | BER30 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for method, values in rows.items():
    lines.append("| " + method + " | " + " | ".join(fmt(values[metric], metric) for metric in METRICS) + " |")

lines.extend([
    "",
    "## Answers to the registered questions",
    "",
    "### A. Does RGB0.5-Control reproduce most of the Hard/Ours gain?",
    "",
    "No. Relative to the frozen Global row, RGB0.5-Control is lower in PSNR and local PSNR and higher in P95, P99, LPIPS, and CIEDE2000. BER30 is close (`0.1130625` versus `0.1131250`) but does not rescue the quality result. A simple global RGB-weight reduction does not reproduce the Hard/Ours pattern.",
    "",
    "### B. Does Global+OKLab-only improve color/perceptual fidelity without reproducing the full local-tail behavior?",
    "",
    "Not relative to the frozen Global baseline in this run. Global+OKLab-only is lower in PSNR/local PSNR and has higher P95, P99, LPIPS, and CIEDE2000 than Global. It is better than RGB0.5-Control on these metrics, but that comparison is not evidence of improvement over the frozen Global baseline. The standalone OKLab control therefore does not show an independent color/perceptual gain under this continuation.",
    "",
    "### C. Does Hard Local-Tail provide a distinct local-tail/concentration contribution?",
    "",
    "Yes, descriptively. Hard Local-Tail improves Top-25 Local PSNR by `+0.232163 dB`, lowers P95 by `1.719804989e-05`, lowers P99 by `1.888583935e-05`, and lowers Gini by `0.008109840` relative to Global. It also improves LPIPS/CIEDE2000, but the clearest component-specific evidence is the local-tail and concentration movement.",
    "",
    "### D. Does Ours show complementary effects from Tail + OKLab?",
    "",
    "The measured results support cautious language about complementary effects, not synergy. Relative to Hard Local-Tail, Ours improves PSNR, local PSNR, P95, P99, LPIPS, and both CIEDE2000 metrics, while BER30 remains effectively tied. Ours has higher Gini than Hard, so the complement is not uniformly favorable on every diagnostic.",
    "",
    "### E. Are any component results negative or mixed?",
    "",
    "Yes. Both Tail-off controls are negative relative to the frozen Global quality row. Ours versus Hard has a Gini increase, and BER30 differences are small/mixed. These outcomes remain visible and are not converted into a universal or causal superiority claim.",
    "",
    "### F. Is the current final method interpretation still supported?",
    "",
    "Yes, with narrower wording. The formal frozen Ours row still supports the interpretation that Hard Local-Tail supplies explicit absolute local-tail control and the combined Ours objective improves measured perceptual/color fidelity under the frozen protocol. The new component controls do not justify claiming that OKLab alone improves the frozen Global baseline, nor do they establish statistical interaction or “synergy.”",
    "",
    "## Boundary",
    "",
    "All rows in this audit are natural source-resolution project-test results; they must not be mixed with the internal matched-PSNR or external 512×512 display-space results. Gini remains diagnostic only. The extension rows are supplementary and do not retune or replace the frozen paper evidence.",
])
(OUT / "COMPONENT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
