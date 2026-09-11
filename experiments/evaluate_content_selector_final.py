#!/usr/bin/env python3
"""One-time fixed-test evaluation of four frozen seed17 epoch-20 branches.

Run only after controlled training completes. --check-ready checks prerequisites
without opening the formal test manifest. There is deliberately no force/rerun
flag: the persistent evaluation guard survives failures as well as completion.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.analyze_patch_distortion import extract_patches, ssim_per_sample
from experiments.content_selector import (
    content_activity, patches, selected_indices, selector_scores,
)
from experiments.evaluate_controlled_seed17 import (
    ATTACK_AREAS, ROOT, evaluate_branch, load_attack_masks, load_model,
    metric_batches, torch_load, training_diagnostics,
)

PROJECT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/content_selector_final"
MANIFEST = ROOT / "reports/uniform_eval_manifest.pt"
EXTENSION = ROOT / "reports/controlled_crop35_40_manifest.pt"
DECISION = ROOT / "reports/content_selector/decision.json"
BRANCHES = {
    "global_continuation": "controlled_seed17_global_continuation",
    "hard_p16_s8_top25": "controlled_seed17_hard_patch16_stride8_top25_weight50",
    "hard_p16_s8_top10": "controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50",
    "content_gradient_alpha1_top10": "seed17_contentaware_gradient_patch16_stride8_top10_alpha1_global0.5_local0.5",
}
BATCH_SIZE = 16
REPEATS = 5


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_ready():
    """Validate the frozen decision/checkpoints; do not deserialize test data."""
    required = [MANIFEST, EXTENSION, DECISION]
    checkpoints = {
        name: ROOT / "experiments/runs" / run / "checkpoint_0020.pth"
        for name, run in BRANCHES.items()
    }
    for path in [*required, *checkpoints.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    decision = json.loads(DECISION.read_text())
    selected = decision["selected"]
    if not (decision["training_justified"] and selected["feature"] == "gradient"
            and selected["alpha"] == 1.0 and selected["gate_pass"]):
        raise ValueError("validation decision is not the frozen gradient alpha1 formulation")
    diagnostics = {}
    for name, path in checkpoints.items():
        checkpoint = torch_load(path, "cpu")
        if checkpoint.get("epoch") != 20:
            raise ValueError(f"not final epoch 20: {path}")
        diagnostics[name] = training_diagnostics(path.parent)
        del checkpoint
    return checkpoints, diagnostics


def claim_test_once(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is atomic and prevents concurrent launches too.
    with (output_dir / "TEST_EVALUATION_STARTED.json").open("x") as handle:
        json.dump({"started_at": datetime.now(timezone.utc).isoformat(),
                   "pid": os.getpid(), "status": "started",
                   "rerun_policy": "No automatic rerun; inspect partial artifacts after failure."},
                  handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def tail_values(values, prefix, higher_is_worse=True):
    """Per-image own-metric tails; aggregate percentiles pool all patch values."""
    ordered = torch.sort(values, dim=1, descending=higher_is_worse).values
    count = values.shape[1]
    per_image = {
        f"{prefix}_mean": values.mean(1),
        f"{prefix}_worst": ordered[:, 0],
        f"{prefix}_top10": ordered[:, :int(np.ceil(count * .10))].mean(1),
        f"{prefix}_top25": ordered[:, :int(np.ceil(count * .25))].mean(1),
    }
    summary = {key: float(value.mean()) for key, value in per_image.items()}
    for percentile in (90, 95, 99):
        key = f"{prefix}_p{percentile}"
        per_image[key] = torch.quantile(values, percentile / 100, dim=1)
        summary[key] = float(np.percentile(values.numpy().reshape(-1), percentile))
    return summary, per_image


def local_perceptual(encoded_patches, original_patches, metric, device, proxy=False):
    ssim_values, lpips_values = [], []
    with torch.no_grad():
        for start in range(0, len(original_patches), BATCH_SIZE * 4):
            left = encoded_patches[start:start + BATCH_SIZE * 4].to(device)
            right = original_patches[start:start + BATCH_SIZE * 4].to(device)
            ssim_values.append(ssim_per_sample(left, right).cpu())
            if proxy:
                left = F.interpolate(left, size=32, mode="bilinear", align_corners=False)
                right = F.interpolate(right, size=32, mode="bilinear", align_corners=False)
            lpips_values.append(metric(left, right).flatten().cpu())
    return torch.cat(ssim_values), torch.cat(lpips_values)


def alignment(scores, target):
    correlations = []
    selected = selected_indices(scores).numpy()
    target_selected = selected_indices(target).numpy()
    overlap = []
    for score, perceptual, left, right in zip(
            scores.numpy(), target.numpy(), selected, target_selected):
        correlations.append(float(spearmanr(score, perceptual).statistic)
                            if np.ptp(score) > 0 and np.ptp(perceptual) > 0 else 0.0)
        left, right = set(left), set(right)
        overlap.append(len(left & right) / len(left | right))
    return torch.tensor(correlations), torch.tensor(overlap)


def additional_metrics(images, encoded, metric, device, content_aware):
    """Reuse encoded outputs; neither encoder nor decoder is called here."""
    summary, per_image, tensors = {}, {}, {}
    for grid, size in (("native32_s32", 32), ("mechanism16_s8", 16)):
        if size == 32:
            left, right = extract_patches(encoded, 32), extract_patches(images, 32)
        else:
            left, right = patches(encoded).flatten(0, 1), patches(images).flatten(0, 1)
        patch_count = len(left) // len(images)
        mse = (left - right).square().mean((1, 2, 3)).reshape(len(images), patch_count)
        ssim, lpips = local_perceptual(left, right, metric, device, proxy=size == 16)
        ssim, lpips = ssim.reshape_as(mse), lpips.reshape_as(mse)
        lpips_name = "lpips_proxy16to32" if size == 16 else "lpips"
        for label, values, higher_is_worse in (
                ("mse", mse, True), ("ssim", ssim, False), (lpips_name, lpips, True)):
            aggregate, individual = tail_values(values, f"{grid}_{label}", higher_is_worse)
            summary.update(aggregate)
            per_image.update(individual)
        tensors[grid] = {"mse": mse, "ssim": ssim, lpips_name: lpips}
        if size == 16:
            activity = content_activity(images)
            feature, alpha = ("gradient", 1.0) if content_aware else ("mse", 0.0)
            scores = selector_scores(mse, activity, feature, alpha)
            indices = selected_indices(scores)
            tensors[grid].update(selector_scores=scores, selected_top10_indices=indices,
                                 activity=activity)
            for target_name, target in (("lpips_proxy16to32", lpips), ("ssim_degradation", 1 - ssim)):
                rho, jaccard = alignment(scores, target)
                per_image[f"{grid}_score_{target_name}_spearman"] = rho
                per_image[f"{grid}_selected_top10_{target_name}_jaccard"] = jaccard
            for label, values in (("mse", mse), ("ssim", ssim), (lpips_name, lpips),
                                  *activity.items()):
                per_image[f"{grid}_selected_top10_{label}"] = values.gather(1, indices).mean(1)
    # Keep pooled percentile summaries above, while averaging image-level alignment.
    for key, value in per_image.items():
        summary.setdefault(key, float(value.mean()))
    global_mse = (encoded - images).square().mean((1, 2, 3))
    per_image["global_mse"] = global_mse
    per_image["psnr"] = 10 * torch.log10(4 / global_mse)
    per_image["worst_psnr"] = 10 * torch.log10(4 / per_image["native32_s32_mse_top25"])
    per_image["ssim"] = metric_batches(ssim_per_sample, encoded, images, device, BATCH_SIZE)
    per_image["lpips"] = metric_batches(metric, encoded, images, device, BATCH_SIZE)
    return summary, per_image, tensors


def comparison_deltas(metrics, reference):
    control = metrics[reference]
    result = {}
    for name, values in metrics.items():
        row = {key: float(value - control[key]) for key, value in values.items()}
        row["local_specific_gain_db"] = row["worst_psnr"] - row["psnr"]
        result[name] = row
    return result


def evaluate_with_per_image_ber(model, images, messages, masks, metric, device):
    """Observe existing decoder calls; never add a second attack evaluation."""
    errors = {name: torch.zeros(len(images)) for name in ATTACK_AREAS}
    names = list(ATTACK_AREAS)
    calls = 0

    def capture(_module, _inputs, decoded):
        nonlocal calls
        batch_index, within_batch = divmod(calls, len(names) * REPEATS)
        attack_index = within_batch // REPEATS
        start = batch_index * BATCH_SIZE
        stop = start + len(decoded)
        target = messages[start:stop].gt(.5)
        errors[names[attack_index]][start:stop] += (decoded.detach().cpu().gt(.5) != target).sum(1)
        calls += 1

    hook = model.decoder.register_forward_hook(capture)
    try:
        result = evaluate_branch(model, images, messages, masks, metric, device,
                                 BATCH_SIZE, 32, REPEATS)
    finally:
        hook.remove()
    expected = int(np.ceil(len(images) / BATCH_SIZE)) * len(names) * REPEATS
    if calls != expected:
        raise RuntimeError("legacy decoder call ordering changed; per-image BER is invalid")
    per_image = {"ber" + name.split("_")[-1]: value / (REPEATS * messages.shape[1])
                 for name, value in errors.items()}
    for key, values in per_image.items():
        if not np.isclose(float(values.mean()), result[0][key], atol=1e-7):
            raise RuntimeError(f"per-image {key} does not reproduce legacy aggregate")
    return result, per_image


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--check-ready", action="store_true")
    args = parser.parse_args()
    checkpoints, diagnostics = check_ready()
    if args.check_ready:
        print("Ready: four frozen epoch-20 checkpoints; formal test tensors not opened.")
        return
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible GPU required; set CUDA_VISIBLE_DEVICES")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    import lpips
    metric = lpips.LPIPS(net="alex", version="0.1", spatial=False, verbose=False).to(device).eval()
    claim_test_once(OUT)
    manifest = torch_load(MANIFEST, "cpu")
    if manifest["repeats"] != REPEATS or manifest.get("batch_size", BATCH_SIZE) != BATCH_SIZE:
        raise ValueError("formal manifest repeats/batch size differ from frozen protocol")
    images, messages = manifest["images"].float(), manifest["messages"].float()
    if images.shape != (50, 3, 128, 128) or messages.shape != (50, 64):
        raise ValueError("formal fixed-test tensor dimensions differ from frozen protocol")
    attack_masks = load_attack_masks(manifest, EXTENSION, REPEATS, BATCH_SIZE)
    metrics, per_image_rows = {}, []
    provenance = {
        "manifest": str(MANIFEST), "manifest_sha256": digest(MANIFEST),
        "attack_extension": str(EXTENSION), "attack_extension_sha256": digest(EXTENSION),
        "validation_decision_sha256": digest(DECISION),
        "evaluator_sha256": digest(Path(__file__)),
        "selector_sha256": digest(PROJECT / "experiments/content_selector.py"),
        "batch_size": BATCH_SIZE, "repeats": REPEATS, "epoch": 20,
        "checkpoints": {name: {"path": str(path), "sha256": digest(path)}
                        for name, path in checkpoints.items()},
    }
    torch.save({"images": images, "messages": messages}, OUT / "fixed_inputs.pt")
    for name, path in checkpoints.items():
        print(f"Final evaluation: {name}", flush=True)
        model = load_model(path, device)
        (legacy, encoded, patch_mse), individual_ber = evaluate_with_per_image_ber(
            model, images, messages, attack_masks, metric, device)
        extra, individual, tensors = additional_metrics(
            images, encoded, metric, device, name == "content_gradient_alpha1_top10")
        individual.update(individual_ber)
        metrics[name] = {**legacy, **extra}
        torch.save({"encoded": encoded, "native32_legacy_patch_mse": patch_mse,
                    "grids": tensors, "per_image": individual}, OUT / f"{name}_tensors.pt")
        for index in range(len(images)):
            per_image_rows.append({"branch": name, "manifest_index": index,
                                   **{key: float(value[index]) for key, value in individual.items()}})
        (OUT / f"{name}_metrics.json").write_text(json.dumps(metrics[name], indent=2) + "\n")
        del model, encoded, tensors
    deltas = {
        "vs_global_continuation": comparison_deltas(metrics, "global_continuation"),
        "vs_hard_top10": comparison_deltas(metrics, "hard_p16_s8_top10"),
    }
    result = {
        "provenance": provenance, "metrics": metrics, "deltas": deltas,
        "training_diagnostics": diagnostics,
        "metric_definitions": {
            "legacy": "Unprefixed metrics are unchanged evaluate_branch native32/stride32 results.",
            "native32_s32": "16 nonoverlapping native RGB32 patches/image; LPIPS AlexNet v0.1.",
            "mechanism16_s8": "225 overlapping patches/image; native16 SSIM; LPIPS 16->32 bilinear proxy, align_corners=False, no clipping.",
            "selector": "Global and Hard use MSE; content branch uses MSE/(1+normalized original gradient). Top10 selects 23/225; stable row-major ties.",
            "tail_aggregation": "Mean/worst/top10/top25 are image means. SSIM tails select lowest SSIM; MSE/LPIPS select highest. P90/P95/P99 pool patch values; CSV per-image percentiles are separate.",
            "worst_psnr": "10*log10(4 / mean per-image native32 Top25 raw MSE), legacy definition.",
            "alignment": "Mean per-image Spearman with LPIPS or 1-SSIM, and Top10 patch-index Jaccard; constant-vector Spearman is 0.",
            "local_specific_gain_db": "Delta WorstPSNR minus delta PSNR, separately against each reference.",
            "ber": f"Identical fixed masks, batch16 repeats5; attacks {list(ATTACK_AREAS)}.",
        },
        "decision": "Descriptive final results only; no unfrozen success thresholds or further tuning.",
    }
    rows = []
    for name, values in metrics.items():
        row = {"branch": name, **values}
        for reference, changes in deltas.items():
            row.update({f"delta_{reference}_{key}": value for key, value in changes[name].items()})
        rows.append(row)
    write_csv(OUT / "summary.csv", rows)
    write_csv(OUT / "per_image.csv", per_image_rows)
    with (OUT / "metrics.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    (OUT / "TEST_EVALUATION_COMPLETED.json").write_text(json.dumps({
        "completed_at": datetime.now(timezone.utc).isoformat(), "branches": list(metrics),
        "metrics_sha256": digest(OUT / "metrics.json"),
    }, indent=2) + "\n")
    print(f"Saved final artifacts: {OUT}", flush=True)


if __name__ == "__main__":
    main()
