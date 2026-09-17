#!/usr/bin/env python3
"""Calibrate and evaluate a matched-PSNR pair using frozen seed17 models.

Validation is used only to select the shared target and the two residual scales.
The selected scales are then applied once to the fixed project-test manifest.
No model parameters are changed and no project-test value is consulted during
calibration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from experiments.losses import patch_mse_per_sample
from network.Encoder_MP_Decoder import EncoderDecoder


MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
VALIDATION_MANIFEST = MOUNT / "reports/content_selector/validation_manifest.pt"
VALIDATION_MASKS = MOUNT / "reports/crop_global_oklab/validation_crop_masks.pt"
PROJECT_TEST_MANIFEST = MOUNT / "reports/uniform_eval_manifest.pt"
PROJECT_TEST_EXTENSION = MOUNT / "reports/controlled_crop35_40_manifest.pt"
RAW_ROOT = MOUNT / "reports/matched_psnr_alpha_residual"
REPORT_CSV = PROJECT / "reports/matched_psnr_alpha_residual.csv"
REPORT_MD = PROJECT / "reports/matched_psnr_alpha_residual.md"

METHODS = {
    "MBRS crop-trained global": MOUNT / "experiments/runs/controlled_seed17_global_continuation/checkpoint_0020.pth",
    "Ours = Hard Local-Tail + global OKLab": MOUNT / "experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/checkpoint_0020.pth",
}
RATIOS = (100, 70, 50, 40, 30)
BATCH_SIZE = 16
REPEATS = 5
PATCH_SIZE = 32
TARGET_MARGIN_DB = 0.05


def load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_from_checkpoint(path: Path, device: torch.device) -> EncoderDecoder:
    checkpoint = load(path, device)
    config = checkpoint["config"]
    model = EncoderDecoder(config["H"], config["W"], config["message_length"], ["Identity()"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def encode(model: EncoderDecoder, images: torch.Tensor, messages: torch.Tensor, device: torch.device) -> torch.Tensor:
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(images), BATCH_SIZE):
            stop = start + BATCH_SIZE
            encoded.append(model.encoder(images[start:stop].to(device), messages[start:stop].to(device)).cpu())
    return torch.cat(encoded)


def rgb(value: torch.Tensor) -> torch.Tensor:
    return ((value.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def psnr_from_mse(mse: float) -> float:
    return float(10.0 * math.log10(1.0 / max(float(mse), 1e-12)))


def blended(images: torch.Tensor, encoded: torch.Tensor, alpha: float) -> torch.Tensor:
    return images + float(alpha) * (encoded - images)


def global_psnr(images: torch.Tensor, encoded: torch.Tensor, alpha: float) -> float:
    value = rgb(blended(images, encoded, alpha))
    reference = rgb(images)
    return psnr_from_mse(float((value - reference).square().mean()))


def select_alpha(images: torch.Tensor, encoded: torch.Tensor, target: float) -> tuple[float, float]:
    """Solve for alpha in [0, 1], assuming residual strength is monotone."""
    endpoint = global_psnr(images, encoded, 1.0)
    if endpoint > target + 1e-10:
        raise RuntimeError(f"target {target:.8f} dB is below the alpha=1 endpoint {endpoint:.8f} dB")
    # The target is deliberately above both alpha=1 endpoints. Check the
    # bracket before bisection so a hidden monotonicity issue cannot pass.
    grid = np.linspace(0.0, 1.0, 21)
    grid_psnr = [global_psnr(images, encoded, float(alpha)) for alpha in grid]
    if any(left + 1e-8 < right for left, right in zip(grid_psnr, grid_psnr[1:])):
        raise RuntimeError("validation PSNR is not monotone decreasing over alpha in [0, 1]")
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if global_psnr(images, encoded, mid) > target:
            low = mid
        else:
            high = mid
    alpha = (low + high) / 2.0
    return alpha, global_psnr(images, encoded, alpha)


def gini(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    total = float(ordered.sum())
    if total <= 1e-15:
        return 0.0
    count = len(ordered)
    return float(((2 * np.arange(1, count + 1) - count - 1) * ordered).sum() / (count * total))


def color_metrics(reference: torch.Tensor, value: torch.Tensor) -> tuple[float, float]:
    reference_np = reference.permute(1, 2, 0).numpy()
    value_np = value.permute(1, 2, 0).numpy()
    distance = deltaE_ciede2000(rgb2lab(reference_np), rgb2lab(value_np)).astype(np.float32)
    patch_scores = torch.from_numpy(distance)[None, None]
    patch_scores = torch.nn.functional.avg_pool2d(patch_scores, 5, stride=1).flatten().numpy()
    top_count = max(1, math.ceil(len(patch_scores) * 0.10))
    return float(distance.mean()), float(np.sort(patch_scores)[-top_count:].mean())


def quality_metrics(
    images: torch.Tensor,
    encoded: torch.Tensor,
    lpips_metric: Any,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    reference = rgb(images)
    value = rgb(encoded)
    per_image_mse = (value - reference).square().mean((1, 2, 3)).numpy()
    patch_values = patch_mse_per_sample(value, reference, PATCH_SIZE, PATCH_SIZE).numpy()
    top_count = max(1, math.ceil(patch_values.shape[1] * 0.25))
    top_values = np.sort(patch_values, axis=1)[:, -top_count:].mean(axis=1)
    lpips_values = []
    with torch.inference_mode():
        for start in range(0, len(value), BATCH_SIZE):
            score = lpips_metric(
                (value[start:start + BATCH_SIZE] * 2.0 - 1.0).to(device),
                (reference[start:start + BATCH_SIZE] * 2.0 - 1.0).to(device),
            ).flatten().cpu().numpy()
            lpips_values.extend(float(item) for item in score)

    rows = []
    for index in range(len(images)):
        ciede_global, ciede_top10 = color_metrics(reference[index], value[index])
        rows.append({
            "manifest_index": index,
            "global_mse": float(per_image_mse[index]),
            "global_psnr": psnr_from_mse(float(per_image_mse[index])),
            "top25_local_psnr": psnr_from_mse(float(top_values[index])),
            "patch_mse_p95": float(np.percentile(patch_values[index], 95)),
            "patch_mse_p99": float(np.percentile(patch_values[index], 99)),
            "lpips": float(lpips_values[index]),
            "ciede2000_global": ciede_global,
            "ciede2000_top10": ciede_top10,
            "gini": gini(patch_values[index]),
        })

    summary = {
        "global_mse": float(per_image_mse.mean()),
        "global_psnr": psnr_from_mse(float(per_image_mse.mean())),
        "top25_local_psnr": float(np.mean([row["top25_local_psnr"] for row in rows])),
        "patch_mse_p95": float(np.mean([row["patch_mse_p95"] for row in rows])),
        "patch_mse_p99": float(np.mean([row["patch_mse_p99"] for row in rows])),
        "lpips": float(np.mean([row["lpips"] for row in rows])),
        "ciede2000_global": float(np.mean([row["ciede2000_global"] for row in rows])),
        "ciede2000_top10": float(np.mean([row["ciede2000_top10"] for row in rows])),
        "gini": float(np.mean([row["gini"] for row in rows])),
    }
    return summary, rows


def load_split(split: str) -> tuple[dict[str, Any], dict[str, Any], Path, list[Path]]:
    if split == "validation":
        manifest_path = VALIDATION_MANIFEST
        masks_path = VALIDATION_MASKS
        manifest = load(manifest_path)
        masks = load(masks_path)
        mask_files = [masks_path]
    else:
        manifest_path = PROJECT_TEST_MANIFEST
        manifest = load(manifest_path)
        masks = dict(manifest["attack_masks"])
        extension = load(PROJECT_TEST_EXTENSION)
        masks.update(extension["attack_masks"])
        mask_files = [manifest_path, PROJECT_TEST_EXTENSION]
    images = manifest["images"]
    messages = manifest["messages"]
    if tuple(images.shape) != (50, 3, 128, 128) or tuple(messages.shape) != (50, 64):
        raise ValueError(f"{split} manifest dimensions are not the frozen 50x128x128/64 contract")
    for ratio in RATIOS:
        key = f"crop_{ratio}"
        if key not in masks or len(masks[key]) != REPEATS:
            raise ValueError(f"missing/incompatible {split} {key} masks")
        if any(len(batch_masks) != math.ceil(len(images) / BATCH_SIZE) for batch_masks in masks[key]):
            raise ValueError(f"incompatible {split} {key} batch-mask count")
    return manifest, masks, manifest_path, mask_files


def ber_metrics(
    model: EncoderDecoder,
    encoded: torch.Tensor,
    messages: torch.Tensor,
    masks: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    errors = {ratio: np.zeros(len(messages), dtype=np.int64) for ratio in RATIOS}
    with torch.inference_mode():
        for ratio in RATIOS:
            key = f"crop_{ratio}"
            for repeat in range(REPEATS):
                for batch_index, start in enumerate(range(0, len(encoded), BATCH_SIZE)):
                    stop = min(start + BATCH_SIZE, len(encoded))
                    mask = masks[key][repeat][batch_index].to(device)
                    decoded = model.decoder(encoded[start:stop].to(device) * mask).cpu().gt(0.5)
                    target = messages[start:stop].gt(0.5)
                    errors[ratio][start:stop] += (decoded != target).sum(1).numpy()
    per_image = {f"ber{ratio}": (errors[ratio] / (REPEATS * messages.shape[1])).tolist() for ratio in RATIOS}
    summary = {f"ber{ratio}": float(np.mean(per_image[f"ber{ratio}"])) for ratio in RATIOS}
    return summary, per_image


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def write_report(
    target: float,
    calibration: dict[str, dict[str, float]],
    final: dict[str, dict[str, float]],
    provenance: dict[str, Any],
) -> None:
    names = list(METHODS)
    val_gap = abs(calibration[names[0]]["selected_psnr"] - calibration[names[1]]["selected_psnr"])
    test_gap = abs(final[names[0]]["global_psnr"] - final[names[1]]["global_psnr"])
    lower = {
        "ber100", "ber70", "ber50", "ber40", "ber30", "patch_mse_p95", "patch_mse_p99",
        "lpips", "ciede2000_global", "ciede2000_top10", "gini",
    }
    higher = {"global_psnr", "top25_local_psnr"}
    wins = {}
    for metric in (*lower, *higher):
        left, right = final[names[0]][metric], final[names[1]][metric]
        if abs(right - left) <= 1e-10:
            wins[metric] = "tie"
        else:
            wins[metric] = "Ours" if (right < left if metric in lower else right > left) else "MBRS global"
    lines = [
        "# Matched-PSNR residual-strength comparison",
        "",
        "## Result",
        "",
        f"Validation selected a shared target of **{target:.2f} dB**. The target is the higher α=1 validation PSNR plus {TARGET_MARGIN_DB:.2f} dB, ceiled to 0.01 dB; both selected scales are inside [0, 1], so no extrapolation was used.",
        "",
        f"The frozen project-test PSNR mismatch is **{test_gap:.6f} dB** ({'PASS' if test_gap <= 0.05 else 'FAIL'} for the requested ≤0.05 dB aim).",
        "",
        "## Validation calibration",
        "",
        "| Method | α=1 PSNR | Selected α | Selected PSNR | Target | Validation gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        row = calibration[name]
        lines.append(f"| {name} | {row['endpoint_psnr']:.6f} | {row['alpha']:.9f} | {row['selected_psnr']:.6f} | {target:.2f} | {abs(row['selected_psnr'] - target):.8f} |")

    lines += [
        "",
        "## Frozen project-test comparison",
        "",
        "| Method | α | PSNR | BER100 | BER70 | BER50 | BER40 | BER30 | Top-25 local PSNR | P95 patch MSE | P99 patch MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        row = final[name]
        lines.append(
            "| {} | {:.9f} | {:.6f} | {:.7f} | {:.7f} | {:.7f} | {:.7f} | {:.7f} | {:.6f} | {:.9e} | {:.9e} | {:.8f} | {:.6f} | {:.6f} | {:.6f} |".format(
                name, row["alpha"], row["global_psnr"], row["ber100"], row["ber70"], row["ber50"], row["ber40"], row["ber30"],
                row["top25_local_psnr"], row["patch_mse_p95"], row["patch_mse_p99"], row["lpips"], row["ciede2000_global"], row["ciede2000_top10"], row["gini"],
            )
        )

    lines += ["", "## Ours minus MBRS crop-trained global", ""]
    for metric in ("global_psnr", "ber100", "ber70", "ber50", "ber40", "ber30", "top25_local_psnr", "patch_mse_p95", "patch_mse_p99", "lpips", "ciede2000_global", "ciede2000_top10", "gini"):
        delta = final[names[1]][metric] - final[names[0]][metric]
        winner = wins[metric]
        verdict = "tie" if winner == "tie" else f"{winner} is better under the metric direction"
        lines.append(f"- `{metric}`: {delta:+.9e} ({verdict}).")

    crop_wins = sum(wins[f"ber{ratio}"] == "Ours" for ratio in RATIOS)
    crop_ties = sum(wins[f"ber{ratio}"] == "tie" for ratio in RATIOS)
    crop_losses = len(RATIOS) - crop_wins - crop_ties
    tail_wins = sum(wins[key] == "Ours" for key in ("top25_local_psnr", "patch_mse_p95", "patch_mse_p99"))
    lines += [
        "",
        "## Answer to the key question",
        "",
        f"At matched global PSNR, Ours strictly wins {crop_wins}/{len(RATIOS)} requested crop-BER levels, ties {crop_ties}, and loses {crop_losses}; it wins {tail_wins}/3 requested local-tail metrics (Top-25 local PSNR, P95, P99). Gini is treated only as a secondary diagnostic.",
        "",
        "Calibration used only the fixed validation manifest. The project-test manifest, messages, and crop masks were opened only after α values were frozen; neither project-test PSNR nor any other project-test metric was used to choose the target or scales.",
        "",
        "## Frozen protocol and provenance",
        "",
        f"- Validation manifest: `{provenance['validation_manifest']}` (SHA-256 `{provenance['validation_manifest_sha256']}`).",
        f"- Validation crop masks: `{provenance['validation_masks']}` (SHA-256 `{provenance['validation_masks_sha256']}`).",
        f"- Project-test manifest: `{provenance['project_test_manifest']}` (SHA-256 `{provenance['project_test_manifest_sha256']}`).",
        f"- Project-test crop-40 extension: `{provenance['project_test_extension']}` (SHA-256 `{provenance['project_test_extension_sha256']}`).",
        f"- Evaluator: `{provenance['evaluator']}` (SHA-256 `{provenance['evaluator_sha256']}`).",
        "- Both methods use the same fixed messages within each split, the same five-repeat masks, affine normalized-tensor residual scaling `x + α(x̂ − x)`, clipped-RGB quality evaluation, native 32×32 non-overlap patch metrics, mean per-image P95/P99, full-image LPIPS, standard CIEDE2000 with 5×5 stride-1 Top-10 patches, and BER on the unquantized normalized tensor.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    required = [*METHODS.values(), VALIDATION_MANIFEST, VALIDATION_MASKS, PROJECT_TEST_MANIFEST, PROJECT_TEST_EXTENSION]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    validation, _, _, _ = load_split("validation")
    validation_images = validation["images"].float()
    validation_messages = validation["messages"].float()

    models = {}
    outputs = {"validation": {}}
    for name, checkpoint in METHODS.items():
        print(f"encoding {name}", flush=True)
        model = model_from_checkpoint(checkpoint, device)
        outputs["validation"][name] = encode(model, validation_images, validation_messages, device)
        models[name] = model

    endpoints = {name: global_psnr(validation_images, outputs["validation"][name], 1.0) for name in METHODS}
    target = math.ceil((max(endpoints.values()) + TARGET_MARGIN_DB) * 100.0 - 1e-9) / 100.0
    calibration = {}
    selected = {}
    for name in METHODS:
        alpha, selected_psnr = select_alpha(validation_images, outputs["validation"][name], target)
        calibration[name] = {"endpoint_psnr": endpoints[name], "alpha": alpha, "selected_psnr": selected_psnr}
        selected[name] = alpha
        print(f"calibrated {name}: alpha={alpha:.9f}, validation_psnr={selected_psnr:.8f}", flush=True)

    # Do not deserialize project-test tensors until both target and alphas are
    # frozen from validation-only evidence.
    project_test, _, _, _ = load_split("project_test")
    test_images = project_test["images"].float()
    test_messages = project_test["messages"].float()
    outputs["project_test"] = {}
    for name in METHODS:
        outputs["project_test"][name] = encode(models[name], test_images, test_messages, device)

    import lpips
    lpips_metric = lpips.LPIPS(net="alex", version="0.1", spatial=False, verbose=False).to(device).eval()
    all_rows = []
    final = {}
    raw = {"target_psnr": target, "calibration": calibration, "splits": {}}
    for split, images, messages in (("validation", validation_images, validation_messages), ("project_test", test_images, test_messages)):
        _, masks, manifest_path, mask_files = load_split(split)
        raw["splits"][split] = {}
        for name in METHODS:
            alpha = selected[name]
            model = models[name]
            value = blended(images, outputs[split][name], alpha)
            quality, per_image = quality_metrics(images, value, lpips_metric, device)
            robustness, ber_per_image = ber_metrics(model, value, messages, masks, device)
            summary = {**quality, **robustness, "alpha": alpha, "split": split, "method": name}
            raw["splits"][split][name] = {"encoded": value, "summary": summary, "per_image": per_image, "ber_per_image": ber_per_image}
            if split == "validation":
                row = {"split": split, "stage": "validation_selected", **summary, "target_psnr": target, "validation_psnr_error": abs(quality["global_psnr"] - target)}
                all_rows.append(row)
            else:
                final[name] = summary
                all_rows.append({"split": split, "stage": "project_test_final", "target_psnr": target, **summary})
            print(f"evaluated {split} / {name}", flush=True)

    for name in METHODS:
        all_rows.append({
            "split": "validation", "stage": "validation_endpoint_alpha1", "method": name,
            "alpha": 1.0, "target_psnr": target, "global_psnr": calibration[name]["endpoint_psnr"],
        })

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({
        "target_psnr": target,
        "calibration": calibration,
        "selected_alpha": selected,
        "validation": {name: raw["splits"]["validation"][name]["encoded"] for name in METHODS},
        "project_test": {name: raw["splits"]["project_test"][name]["encoded"] for name in METHODS},
    }, RAW_ROOT / "matched_outputs.pt")
    raw_json = {
        "target_psnr": target,
        "calibration": calibration,
        "selected_alpha": selected,
        "final": final,
    }
    (RAW_ROOT / "summary.json").write_text(json.dumps(raw_json, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "validation_manifest": str(VALIDATION_MANIFEST),
        "validation_manifest_sha256": sha256(VALIDATION_MANIFEST),
        "validation_masks": str(VALIDATION_MASKS),
        "validation_masks_sha256": sha256(VALIDATION_MASKS),
        "project_test_manifest": str(PROJECT_TEST_MANIFEST),
        "project_test_manifest_sha256": sha256(PROJECT_TEST_MANIFEST),
        "project_test_extension": str(PROJECT_TEST_EXTENSION),
        "project_test_extension_sha256": sha256(PROJECT_TEST_EXTENSION),
        "evaluator": str(Path(__file__)),
        "evaluator_sha256": sha256(Path(__file__)),
        "checkpoints": {name: {"path": str(path), "sha256": sha256(path)} for name, path in METHODS.items()},
        "target_rule": "ceil((max(validation alpha=1 PSNR) + 0.05 dB) * 100) / 100",
        "no_project_test_tuning": True,
    }
    (RAW_ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    write_csv(REPORT_CSV, all_rows)
    write_report(target, calibration, final, provenance)
    print(f"saved {REPORT_MD}", flush=True)
    print(f"saved {REPORT_CSV}", flush=True)


if __name__ == "__main__":
    main()
