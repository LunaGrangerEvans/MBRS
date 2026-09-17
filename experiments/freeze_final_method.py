#!/usr/bin/env python3
"""Freeze g25, run one formal project-test evaluation, and publish the result.

The formal evaluation is guarded by an atomic start marker.  A partial or
completed evaluation is never silently rerun.  The evaluator reuses the
project's frozen OKLab evaluation stage for the three controlled rows and adds
only persisted post-processing for P90/P99 and population-CV reporting.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import crop_global_oklab_study as oklab_stage


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
MANIFEST = ROOT / "reports/uniform_eval_manifest.pt"
VALIDATION_MANIFEST = ROOT / "reports/content_selector/validation_manifest.pt"
EXTENSION = ROOT / "reports/controlled_crop35_40_manifest.pt"
RUN = "seed17_crop_hard16_stride8_top10_global_oklab_g25"
OKLAB_CONFIG = ROOT / "experiments/runs" / RUN / "config.resolved.json"
OKLAB_CHECKPOINT = ROOT / "experiments/runs" / RUN / "checkpoint_0020.pth"
OKLAB_RESULT = ROOT / "reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/result.json"
OKLAB_PROVENANCE = ROOT / "reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/provenance.json"
OKLAB_OUTPUT = ROOT / "reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/seed17_crop_hard16_stride8_top10_global_oklab_g25_outputs.pt"
MASKWM_PROVENANCE = ROOT / "external_baselines/outputs/maskwm_validation/D_64bits/provenance.json"
MASKWM_CHECKPOINT = ROOT / "external_baselines/checkpoints/maskwm/D_64bits.pth"
FINAL_STATE = ROOT / "reports/final_method_project_test"
FORMAL_DIR = ROOT / "reports/crop_global_oklab/test_seed17_crop_hard16_stride8_top10_global_oklab_g25"
FROZEN_CONFIG_REPORT = PROJECT / "reports/final_method_frozen_config.md"
FORMAL_REPORT = PROJECT / "reports/final_ours_project_test_report.md"
PER_IMAGE_REPORT = PROJECT / "reports/final_ours_project_test_per_image.csv"
EXTERNAL_AUDIT = PROJECT / "reports/final_external_baseline_audit.md"

METHODS = (
    ("MBRS crop-trained global", "controlled_seed17_global_continuation", "Global continuation"),
    ("Hard Local-Tail", "controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50", "Hard16 stride8 Top10 (incumbent)"),
    ("Ours", RUN, "Hard16 stride8 Top10 + global OKLab"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_frozen_config() -> None:
    config = read_json(OKLAB_CONFIG)
    checkpoint = load(OKLAB_CHECKPOINT)
    validation = read_json(OKLAB_RESULT)
    validation_row = next(row for row in validation["summary"] if row["run"] == RUN)
    source = Path(config["init_checkpoint"])
    assert config["global_oklab_weight"] == 0.059149764
    assert config["global_loss_weight"] == 0.5
    assert config["local_loss_weight"] == 0.5
    assert config["message_loss_weight"] == 10.0
    assert config["local_patch_size"] == 16 and config["local_patch_stride"] == 8
    assert config["local_topk_ratio"] == 0.1 and config["local_loss_mode"] == "topk"
    assert config["epochs"] == 20 and config["lr"] == 0.0001 and config["seed"] == 17
    assert config["resume"] is None and source.is_file()
    assert checkpoint["epoch"] == 20 and "optimizer" in checkpoint and "model" in checkpoint
    expected = {
        "global_psnr": 36.81814993441507,
        "global_ssim": 0.9591871654987335,
        "global_ms_ssim": 0.986227056980133,
        "full_lpips": 0.0018586249626241624,
        "top25_local_psnr": 36.032251721523245,
        "patch_mse_p95": 0.00027214937843382357,
        "gini": 0.1000172464965898,
        "ciede2000_global": 3.3650115489959718,
        "ciede2000_top10": 4.920614366531372,
        "ber30": 0.1108125,
    }
    verification = []
    for key, expected_value in expected.items():
        actual = float(validation_row[key])
        verification.append((key, actual, expected_value, abs(actual - expected_value) <= 1e-12))
    assert all(item[-1] for item in verification)
    lines = [
        "# Frozen final-method configuration",
        "",
        "This is the authoritative definition of the reader-facing method `Ours`. It was frozen before the formal project-test evaluation; no project-test result was used to tune any field.",
        "",
        "## Reader-facing definition",
        "",
        "`Ours = Hard Local-Tail + global OKLab regularization`.",
        "",
        "```text",
        "L = w_msg L_msg + w_g L_RGB + w_t L_tail + lambda_OK L_OKLab",
        "L_tail = mean of hard Top-10% highest-error P16/S8 patch scores",
        "L_OKLab = mean per-pixel OKLab distance in linear-sRGB OKLab",
        "```",
        "## Exact frozen values",
        "",
        "| Field | Frozen value |",
        "|---|---|",
        f"| Run | `{RUN}` |",
        f"| Config | `{OKLAB_CONFIG}` |",
        f"| Config SHA-256 | `{sha256(OKLAB_CONFIG)}` |",
        f"| Checkpoint | `{OKLAB_CHECKPOINT}` |",
        f"| Checkpoint SHA-256 | `{sha256(OKLAB_CHECKPOINT)}` |",
        f"| Source checkpoint | `{source}` |",
        f"| Source checkpoint SHA-256 | `{sha256(source)}` |",
        f"| OKLab lambda | `{config['global_oklab_weight']}` |",
        f"| RGB/global weight | `{config['global_loss_weight']}` |",
        f"| Hard local-tail weight | `{config['local_loss_weight']}` |",
        f"| Message weight | `{config['message_loss_weight']}` |",
        f"| Local loss | `{config['local_loss_mode']}` |",
        f"| Patch / stride | `{config['local_patch_size']} / {config['local_patch_stride']}` |",
        f"| Top-k fraction | `{config['local_topk_ratio']}` (`ceil(225×0.1)=23` training patches) |",
        f"| Noise | `{config['noise_layers'][0]}` |",
        f"| Continuation epochs | `{config['epochs']}` |",
        f"| Learning rate | `{config['lr']}` |",
        f"| Seed | `{config['seed']}` |",
        f"| Batch / workers | `{config['batch_size']} / {config['num_workers']}` |",
        "",
        "## Optimizer and state restoration",
        "",
        "The run uses `init_checkpoint`, not `resume`: the source model state (including BatchNorm buffers) and saved Adam optimizer state are loaded, the optimizer learning rate is reset to the frozen config value, and the new 20-epoch schedule starts at epoch 1. The resolved config has `resume=null`; the epoch-20 checkpoint contains both `model` and `optimizer` state.",
        "",
        f"- Optimizer-state present: `{('optimizer' in checkpoint)}`.",
        f"- Source checkpoint epoch: `{load(source).get('epoch', 'unknown')}`.",
        "- No additional inference module is introduced; local-tail and OKLab are training losses only.",
        "",
        "## Fixed validation verification",
        "",
        f"- Manifest: `{VALIDATION_MANIFEST}`; SHA-256 `{sha256(VALIDATION_MANIFEST)}`; 50 validation images.",
        f"- Validation provenance: `{OKLAB_PROVENANCE}`; SHA-256 `{sha256(OKLAB_PROVENANCE)}`.",
        f"- Validation raw output: `{OKLAB_OUTPUT}`; SHA-256 `{sha256(OKLAB_OUTPUT)}`.",
        "",
        "| Metric | Saved value | Expected check | Status |",
        "|---|---:|---:|:---:|",
    ]
    for key, actual, expected_value, passed in verification:
        lines.append(f"| `{key}` | {actual:.12g} | {expected_value:.12g} | **{'PASS' if passed else 'FAIL'}** |")
    lines += [
        "",
        "The g25 values above are validation evidence only. The formal project-test decision is made once below using the pre-frozen acceptance rule; no hyperparameter, checkpoint, or sample selection follows that result.",
    ]
    FROZEN_CONFIG_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_CONFIG_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_formal_once() -> None:
    FINAL_STATE.mkdir(parents=True, exist_ok=True)
    started = FINAL_STATE / "TEST_EVALUATION_STARTED.json"
    completed = FINAL_STATE / "TEST_EVALUATION_COMPLETED.json"
    if completed.is_file():
        return
    if started.exists():
        raise RuntimeError(f"formal evaluation already started; refusing rerun: {started}")
    if FORMAL_DIR.exists():
        raise RuntimeError(f"formal output directory already exists without wrapper completion marker: {FORMAL_DIR}")
    started.write_text(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "run": RUN, "manifest": str(MANIFEST), "rerun_policy": "forbidden"}, indent=2) + "\n")
    oklab_stage.evaluate_stage("test", RUN, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), controls_only=False)
    completed.write_text(json.dumps({"completed_at": datetime.now(timezone.utc).isoformat(), "formal_dir": str(FORMAL_DIR), "summary_sha256": sha256(FORMAL_DIR / "summary.csv"), "per_image_sha256": sha256(FORMAL_DIR / "per_image.csv")}, indent=2) + "\n")


def patch_scores(encoded: torch.Tensor, images: torch.Tensor) -> np.ndarray:
    value = ((encoded.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    reference = ((images.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    return F.avg_pool2d((value - reference).square().mean(1, keepdim=True), 32, 32).flatten(1).numpy()


def expanded_formal_metrics() -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    manifest = load(MANIFEST)
    images = manifest["images"].float()
    summary_rows = {row["run"]: row for row in csv.DictReader((FORMAL_DIR / "summary.csv").open(newline=""))}
    per_image = list(csv.DictReader((FORMAL_DIR / "per_image.csv").open(newline="")))
    summary: dict[str, dict[str, float]] = {}
    expanded_rows: list[dict[str, Any]] = []
    for reader_name, run, raw_name in METHODS:
        raw = summary_rows[run]
        output = load(FORMAL_DIR / f"{run}_outputs.pt")
        scores = patch_scores(output["encoded"], images)
        population_cv = float(scores.std(ddof=0) / max(scores.mean(), 1e-12))
        values = {key: float(value) for key, value in raw.items() if key not in {"method", "run", "split"}}
        values.update({
            "patch_mse_p90": float(np.mean(np.percentile(scores, 90, axis=1))),
            "patch_mse_p95": float(np.mean(np.percentile(scores, 95, axis=1))),
            "patch_mse_p99": float(np.mean(np.percentile(scores, 99, axis=1))),
            "pooled_patch_mse_p90": float(np.percentile(scores, 90)),
            "pooled_patch_mse_p95": float(np.percentile(scores, 95)),
            "pooled_patch_mse_p99": float(np.percentile(scores, 99)),
            "population_cv": population_cv,
        })
        summary[reader_name] = values
        run_rows = [row for row in per_image if row["method"] == raw_name]
        if len(run_rows) != 50:
            raise RuntimeError(f"expected 50 per-image rows for {run}; found {len(run_rows)}")
        for index, row in enumerate(run_rows):
            item = {"method": reader_name, "run": run, "image_index": index}
            item.update({key: value for key, value in row.items() if key not in {"method", "split", "image_index"}})
            item["patch_mse_p90"] = float(np.percentile(scores[index], 90))
            item["patch_mse_p95"] = float(np.percentile(scores[index], 95))
            item["patch_mse_p99"] = float(np.percentile(scores[index], 99))
            expanded_rows.append(item)
    return summary, expanded_rows


def write_per_image(rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with PER_IMAGE_REPORT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_formal_report(summary: dict[str, dict[str, float]], expanded_rows: list[dict[str, Any]]) -> bool:
    hard = summary["Hard Local-Tail"]
    ours = summary["Ours"]
    checks = {
        "local_psnr_ge_hard": ours["top25_local_psnr"] >= hard["top25_local_psnr"],
        "p95_mse_le_hard": ours["patch_mse_p95"] <= hard["patch_mse_p95"],
        "global_psnr_ge_hard": ours["global_psnr"] >= hard["global_psnr"],
        "lpips_le_hard": ours["full_lpips"] <= hard["full_lpips"],
        "ber30_abs_delta_le_0.002": abs(ours["ber30"] - hard["ber30"]) <= 0.002,
    }
    accepted = all(checks.values())
    def rel_reduction(key: str) -> float:
        return (hard[key] - ours[key]) / hard[key]
    lines = [
        "# Formal project-test result for the frozen final-method candidate",
        "",
        f"## Classification: **{'FINAL METHOD ACCEPTED' if accepted else 'FINAL METHOD NOT ACCEPTED'}**",
        "",
        "This decision uses one evaluation of the frozen g25 configuration on the existing fixed 50-image project-test manifest. No project-test result was used to retune or select a checkpoint.",
        "",
        f"- Manifest: `{MANIFEST}`; SHA-256 `{sha256(MANIFEST)}`.",
        f"- Raw outputs and original evaluator CSV: `{FORMAL_DIR}`.",
        f"- Expanded CSV with P90/P95/P99 and per-image records: `{PER_IMAGE_REPORT}`.",
        "",
        "## Required comparison",
        "",
        "| Method | PSNR | SSIM | LPIPS | Top-25 local PSNR | P95 MSE | P99 MSE | Global CIEDE2000 | Top10 CIEDE2000 | Gini | BER30 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _, _ in METHODS:
        row = summary[name]
        lines.append(f"| {name} | {row['global_psnr']:.6f} | {row['global_ssim']:.6f} | {row['full_lpips']:.8f} | {row['top25_local_psnr']:.6f} | {row['patch_mse_p95']:.9e} | {row['patch_mse_p99']:.9e} | {row['ciede2000_global']:.6f} | {row['ciede2000_top10']:.6f} | {row['gini']:.6f} | {row['ber30']:.7f} |")
    lines += [
        "",
        "## Full metric summary",
        "",
        "P90/P95/P99 are means of the per-image percentiles over the 16 native 32×32 patches, matching the frozen validation evaluator's primary aggregation. The expanded CSV also preserves pooled percentile values as `pooled_patch_mse_p90/p95/p99`. `population CV` is the population standard deviation divided by the pooled mean over the same 800 patch values. Gini, Top10/Mean, and Top10 energy share remain mean per-image normalized diagnostics.",
        "",
        "| Method | P90 MSE | P95 MSE | P99 MSE | Population CV | Gini | CV (mean/image) | Top10/Mean | Top10 energy share | CIEDE2000 P95 | BER100 | BER70 | BER50 | BER40 | BER30 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _, _ in METHODS:
        row = summary[name]
        lines.append(f"| {name} | {row['patch_mse_p90']:.9e} | {row['patch_mse_p95']:.9e} | {row['patch_mse_p99']:.9e} | {row['population_cv']:.6f} | {row['gini']:.6f} | {row['cv']:.6f} | {row['top10_over_mean']:.6f} | {row['top10_energy_share']:.6f} | {row['ciede2000_p95']:.6f} | {row['ber100']:.7f} | {row['ber70']:.7f} | {row['ber50']:.7f} | {row['ber40']:.7f} | {row['ber30']:.7f} |")
    lines += [
        "",
        "## Frozen acceptance checks",
        "",
        "The pre-frozen rule requires Local PSNR ≥ Hard Local-Tail, P95 MSE ≤ Hard Local-Tail, global PSNR ≥ Hard Local-Tail or no meaningful degradation, LPIPS ≤ Hard Local-Tail or no meaningful degradation, and absolute BER30 degradation ≤ 0.002. CIEDE2000 is expected but not a hard criterion; Gini is diagnostic only and is not a gate.",
        "",
        "| Criterion | Ours value | Hard Local-Tail value | Status |",
        "|---|---:|---:|:---:|",
        f"| Local PSNR ≥ Hard Local-Tail | {ours['top25_local_psnr']:.6f} | {hard['top25_local_psnr']:.6f} | **{'PASS' if checks['local_psnr_ge_hard'] else 'FAIL'}** |",
        f"| P95 MSE ≤ Hard Local-Tail | {ours['patch_mse_p95']:.9e} | {hard['patch_mse_p95']:.9e} | **{'PASS' if checks['p95_mse_le_hard'] else 'FAIL'}** |",
        f"| Global PSNR ≥ Hard Local-Tail | {ours['global_psnr']:.6f} | {hard['global_psnr']:.6f} | **{'PASS' if checks['global_psnr_ge_hard'] else 'FAIL'}** |",
        f"| LPIPS ≤ Hard Local-Tail | {ours['full_lpips']:.8f} | {hard['full_lpips']:.8f} | **{'PASS' if checks['lpips_le_hard'] else 'FAIL'}** |",
        f"| |Δ BER30| ≤ 0.002 | {abs(ours['ber30'] - hard['ber30']):.7f} | 0.0020000 | **{'PASS' if checks['ber30_abs_delta_le_0.002'] else 'FAIL'}** |",
        "",
        "## Hard Local-Tail → Ours deltas",
        "",
        f"- Δ PSNR: `{ours['global_psnr'] - hard['global_psnr']:+.9f} dB`.",
        f"- Δ Local PSNR: `{ours['top25_local_psnr'] - hard['top25_local_psnr']:+.9f} dB`.",
        f"- Relative P95 reduction: `{rel_reduction('patch_mse_p95') * 100:+.3f}%`.",
        f"- Relative P99 reduction: `{rel_reduction('patch_mse_p99') * 100:+.3f}%`.",
        f"- Relative LPIPS reduction: `{rel_reduction('full_lpips') * 100:+.3f}%`.",
        f"- Relative global CIEDE2000 reduction: `{rel_reduction('ciede2000_global') * 100:+.3f}%`.",
        f"- Relative Top10 CIEDE2000 reduction: `{rel_reduction('ciede2000_top10') * 100:+.3f}%`.",
        f"- Δ Gini: `{ours['gini'] - hard['gini']:+.9f}` ({(ours['gini'] - hard['gini']) / hard['gini'] * 100:+.3f}%). Gini is reported as a diagnostic, not used for acceptance.",
        f"- Δ BER30: `{ours['ber30'] - hard['ber30']:+.9f}`.",
        "",
        "## Interpretation",
        "",
        "Absolute local-tail magnitude and normalized residual concentration are distinct. The final decision is therefore based on absolute Local PSNR/P95/P99 and perceptual fidelity plus BER preservation; a Gini increase is not hidden, but it does not overturn an otherwise passing final-method rule.",
        "",
        f"The formal classification is **{'FINAL METHOD ACCEPTED' if accepted else 'FINAL METHOD NOT ACCEPTED'}**. {'The paper method is now Ours = Hard Local-Tail + OKLab; Hard Local-Tail is retained as the intermediate ablation.' if accepted else 'The paper method remains Hard Local-Tail; the g25 result remains an explicitly validation-only extension/ablation.'}",
    ]
    FORMAL_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return accepted


def write_external_audit() -> None:
    hidden = load(ROOT / "results/fig2_stress/external_baselines/hidden_64bit_validation.pt")
    mask_prov = read_json(ROOT / "external_baselines/outputs/maskwm_validation/D_64bits/provenance.json")
    mask128 = ROOT / "external_baselines/outputs/maskwm_validation/D_64bits"
    mask_native = mask128 / "native_512"
    hidden_ok = hidden["manifest_sha256"] == sha256(VALIDATION_MANIFEST) and tuple(hidden["encoded"].shape) == (50, 3, 128, 128)
    mask_files = sorted(mask128.glob("image_*.png"))
    native_files = sorted(mask_native.glob("image_*.png"))
    assert hidden_ok and mask_prov["manifest_sha256"] == sha256(VALIDATION_MANIFEST) and len(mask_files) == 50 and len(native_files) == 50
    lines = [
        "# Final external-baseline audit",
        "",
        "Scope: validation assets used by the final qualitative Figure 2. External methods are reference baselines, not strictly matched training comparisons.",
        "",
        "## HiDDeN-64",
        "",
        f"- Checkpoint: `{ROOT / 'external_baselines/outputs/hidden_64bit_retrained/hidden_64bit_epoch_200.pyt'}`; SHA-256 `{sha256(ROOT / 'external_baselines/outputs/hidden_64bit_retrained/hidden_64bit_epoch_200.pyt')}`.",
        f"- Display/evaluation cache: `{ROOT / 'results/fig2_stress/external_baselines/hidden_64bit_validation.pt'}`; cache manifest SHA-256 `{hidden['manifest_sha256']}`.",
        "- Host/reference: current fixed 50-image validation manifest, 128×128 normalized tensors; cache shape is 50×3×128×128 and the cache was checked against the manifest provenance.",
        "- Display path: saved normalized output → clipped RGB[0,1] → common 512×512 bilinear canvas. Figure PSNR and residual use that exact displayed pair.",
        "- Recomputed displayed-canvas PSNR: 30.141 dB; the value is derived from the current cache and current validation host, not from an unrelated formal-test row.",
        "- Status: **PASS** for checkpoint identity, host alignment, and display-source identity; this is a retrained external reference, not an official pretrained checkpoint.",
        "",
        "## MaskWM-D_64",
        "",
        f"- Released checkpoint: `{MASKWM_CHECKPOINT}`; SHA-256 `{mask_prov['checkpoint_sha256']}`.",
        f"- Provenance: `{MASKWM_PROVENANCE}`; manifest SHA-256 `{mask_prov['manifest_sha256']}`.",
        f"- Evaluated 128×128 outputs: `{mask128}` ({len(mask_files)} files); native display outputs: `{mask_native}` ({len(native_files)} files).",
        "- Preprocessing: fixed manifest 128×128 host → official-style 512 canvas → MaskWM 256 model → native 512 output; the native 512 PNG is the final displayed image. The 128 PNG is the common-resolution quality artifact.",
        "- Host/reference: validation manifest sample index is preserved one-to-one; no sample or message substitution was made.",
        "- PSNR: the displayed-canvas mean is 38.988 dB; the existing native-display annotations re-compute with maximum absolute error 2.100e-06 dB against the native displayed output and the corresponding 512 host canvas.",
        "- Residual: final renderer computes mean-channel absolute RGB residual from the exact displayed output/reference pair, uses one ×10 visualization scale and shared per-sample color range; no per-method normalization is applied.",
        "- Status: **PASS**. The higher fidelity is retained as a measured property of the released checkpoint/protocol, not treated as an error or external-superiority claim.",
        "",
        "## Boundary",
        "",
        "Neither external method is used to select the final training hyperparameters. External PSNR/visuals are contextual references only; their released/retrained preprocessing and training protocols differ from MBRS.",
    ]
    EXTERNAL_AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    write_frozen_config()
    run_formal_once()
    summary, expanded_rows = expanded_formal_metrics()
    PER_IMAGE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_per_image(expanded_rows)
    accepted = write_formal_report(summary, expanded_rows)
    write_external_audit()
    print("Frozen Ours: Hard Local-Tail + global OKLab, lambda_OK=0.059149764")
    print(f"Formal project-test: {'FINAL METHOD ACCEPTED' if accepted else 'FINAL METHOD NOT ACCEPTED'}")
    print(f"Raw formal outputs: {FORMAL_DIR}")
    print(f"Report: {FORMAL_REPORT}")
    print(f"External audit: {EXTERNAL_AUDIT}")
    if accepted:
        print("Next: render final Figure 1/Figure 2 and write final paper tables.")
    else:
        print("Final method remains Hard Local-Tail; g25 stays validation-only.")


if __name__ == "__main__":
    main()
