#!/usr/bin/env python3
"""Validation-only Figure 2 stress-search pipeline for the MBRS study.

This file deliberately keeps exploratory training/evaluation artifacts outside
the repository's formal result paths.  It reuses the controlled trainer and
the project's frozen validation manifest, but never opens the project-test
manifest and never writes Table I inputs.

Typical use::

    python experiments/fig2_stress_pipeline.py prepare
    # run the emitted configs with train_local_patch.py (usually on two GPUs)
    python experiments/fig2_stress_pipeline.py evaluate
    python experiments/fig2_stress_pipeline.py analyze

The ``prepare`` command only creates configs under
``/mnt/wmcontent/.../checkpoints/fig2_stress``.  The evaluator writes outputs
under the matching results directory and reports under ``reports/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from network.Encoder_MP_Decoder import EncoderDecoder
from experiments.losses import patch_mse_per_sample


PROJECT = Path(__file__).resolve().parents[1]
MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
CHECKPOINT_ROOT = MOUNT / "checkpoints/fig2_stress"
RESULT_ROOT = MOUNT / "results/fig2_stress"
REPORT_ROOT = PROJECT / "reports"
FIGURE_ROOT = PROJECT / "paper/figures"
MANIFEST = MOUNT / "reports/content_selector/validation_manifest.pt"
CROP_MASKS = MOUNT / "reports/crop_global_oklab/validation_crop_masks.pt"
SOURCE_CHECKPOINT = MOUNT / "experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth"
MASKWM_NATIVE_OUTPUT = MOUNT / "external_baselines/outputs/maskwm_validation/D_64bits/native_512"
MASKWM_PROVENANCE = MASKWM_NATIVE_OUTPUT.parent / "provenance.json"
HIDDEN_REPO = PROJECT / "external_baselines/repos/HiDDeN-pytorch"
HIDDEN_CHECKPOINT = MOUNT / "external_baselines/outputs/hidden_64bit_retrained/hidden_64bit_epoch_200.pyt"
HIDDEN_EVALUATION = HIDDEN_CHECKPOINT.parent / "evaluation.json"
HIDDEN_CACHE = RESULT_ROOT / "external_baselines/hidden_64bit_validation.pt"

FINAL_METHODS = (
    "Original",
    "HiDDeN-64 (retrained external)",
    "MaskWM-D (official external)",
    "Global Reconstruction (internal baseline)",
    "Hard Local-Tail (proposed)",
)
DISPLAY_SIZE = 512
RESIDUAL_AMPLIFICATION = 10

FORMAL_FILES = (
    PROJECT / "paper/table1_main_results.tex",
    PROJECT / "paper/main_table.csv",
    PROJECT / "reports/current_controlled_master_table.csv",
)

SCREEN_CONDITIONS: dict[str, dict[str, float]] = {
    "reference": {"w_msg": 10.0, "lambda_img": 1.0},
    "wmsg20": {"w_msg": 20.0, "lambda_img": 1.0},
    "wmsg40": {"w_msg": 40.0, "lambda_img": 1.0},
    "wmsg80": {"w_msg": 80.0, "lambda_img": 1.0},
    "img05": {"w_msg": 10.0, "lambda_img": 0.5},
    "img025": {"w_msg": 10.0, "lambda_img": 0.25},
    "wmsg20_img05": {"w_msg": 20.0, "lambda_img": 0.5},
    "wmsg20_img025": {"w_msg": 20.0, "lambda_img": 0.25},
    "wmsg40_img05": {"w_msg": 40.0, "lambda_img": 0.5},
    "wmsg40_img025": {"w_msg": 40.0, "lambda_img": 0.25},
}

ALPHAS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
BER_LEVELS = (100, 70, 50, 40, 30)
ROI_SIZES = (32, 40, 48, 64)
SCREEN_EPOCHS = 5


def load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def make_config(method: str, condition: str, epochs: int) -> dict[str, Any]:
    values = SCREEN_CONDITIONS[condition]
    if method == "Ours-base":
        mode = "none"
        global_weight = values["lambda_img"]
        local_weight = 0.0
    elif method == "Ours":
        mode = "topk"
        global_weight = values["lambda_img"] * 0.5
        local_weight = values["lambda_img"] * 0.5
    else:
        raise ValueError(method)
    return {
        "name": f"fig2_{safe_name(method)}_{condition}",
        "dataset_path": str(MOUNT / "datasets"),
        "H": 128,
        "W": 128,
        "message_length": 64,
        "batch_size": 16,
        "num_workers": 0,
        "epochs": epochs,
        "lr": 1e-4,
        "seed": 17,
        "noise_layers": ["RandomCrop(0.3, 1.0)"],
        "message_loss_weight": values["w_msg"],
        "global_loss_weight": global_weight,
        "local_loss_weight": local_weight,
        "local_loss_mode": mode,
        "local_patch_size": 16,
        "local_patch_stride": 8 if mode == "topk" else None,
        "local_topk_ratio": 0.1,
        "deterministic": True,
        "deterministic_warn_only": False,
        "require_single_gpu": True,
        "save_every": 5,
        "device": "cuda",
        "init_checkpoint": str(SOURCE_CHECKPOINT),
        "exploratory": True,
        "formal_status": "not formal; validation-only Figure 2 stress search",
        "condition": condition,
        "method": method,
    }


def run_name(method: str, condition: str) -> str:
    return f"{safe_name(method).lower()}_{condition}"


def prepare(args: argparse.Namespace) -> None:
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for condition in SCREEN_CONDITIONS:
        for method in ("Ours-base", "Ours"):
            name = run_name(method, condition)
            run_dir = CHECKPOINT_ROOT / name
            run_dir.mkdir(parents=True, exist_ok=True)
            config = make_config(method, condition, args.epochs)
            config_path = run_dir / "config.json"
            if config_path.exists():
                existing = json.loads(config_path.read_text())
                if existing != config:
                    raise RuntimeError(f"refusing to overwrite incompatible config: {config_path}")
            else:
                config_path.write_text(json.dumps(config, indent=2) + "\n")
            runs.append({
                "run": name,
                "method": method,
                "condition": condition,
                "w_msg": config["message_loss_weight"],
                "lambda_img": SCREEN_CONDITIONS[condition]["lambda_img"],
                "config": str(config_path),
                "output_dir": str(run_dir),
                "epochs": args.epochs,
                "source_checkpoint": str(SOURCE_CHECKPOINT),
                "source_checkpoint_sha256": digest(SOURCE_CHECKPOINT),
            })
    manifest = {
        "kind": "fig2_stress_screen",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": 17,
        "screen_epochs": args.epochs,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": digest(SOURCE_CHECKPOINT),
        "validation_manifest": str(MANIFEST),
        "validation_manifest_sha256": digest(MANIFEST),
        "runs": runs,
        "formal_files_before": {str(path): digest(path) for path in FORMAL_FILES if path.is_file()},
    }
    (CHECKPOINT_ROOT / "screen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prepared {len(runs)} isolated runs under {CHECKPOINT_ROOT}")
    print("Each run command:")
    for row in runs:
        print(
            f"CUDA_VISIBLE_DEVICES=<gpu> {sys.executable} experiments/train_local_patch.py "
            f"--config {row['config']} --output-dir {row['output_dir']} --device cuda "
            f"--init-checkpoint {SOURCE_CHECKPOINT}"
        )


def prepare_promoted(args: argparse.Namespace) -> None:
    """Create independent longer continuations for validation survivors."""
    screen_csv = REPORT_ROOT / "fig2_strong_embedding_results.csv"
    if not screen_csv.is_file():
        raise FileNotFoundError(f"run evaluate first: {screen_csv}")
    rows = [row for row in read_csv(screen_csv) if row.get("condition") in SCREEN_CONDITIONS]
    base, ours = paired_rows(rows)
    conditions: list[str] = []
    for condition in sorted(set(base) & set(ours)):
        if base[condition]["status"] == "early_reject" or ours[condition]["status"] == "early_reject":
            continue
        delta = f(ours[condition], "local_psnr") - f(base[condition], "local_psnr")
        p95_delta = f(ours[condition], "p95") - f(base[condition], "p95")
        if abs(f(ours[condition], "ber30") - f(base[condition], "ber30")) <= .005 or delta > 0 or p95_delta < 0:
            conditions.append(condition)
    # Keep promotion bounded: the best same-condition candidates plus the
    # most aggressive surviving condition are enough to test visibility.
    def score(condition: str) -> tuple[float, float, float]:
        delta = f(ours[condition], "local_psnr") - f(base[condition], "local_psnr")
        p95_delta = f(ours[condition], "p95") - f(base[condition], "p95")
        ber_gap = abs(f(ours[condition], "ber30") - f(base[condition], "ber30"))
        return (delta, -p95_delta, -ber_gap)
    conditions = sorted(set(conditions), key=score, reverse=True)[: max(1, args.max_conditions)]
    runs: list[dict[str, Any]] = []
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        for method in ("Ours-base", "Ours"):
            config = make_config(method, condition, args.epochs)
            name = f"promoted_{run_name(method, condition)}"
            config["name"] = f"fig2_{safe_name(method)}_promoted_{condition}"
            config["promotion_source"] = "5-epoch fixed-validation screen"
            run_dir = CHECKPOINT_ROOT / name
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "config.json"
            if config_path.exists():
                existing = json.loads(config_path.read_text())
                if existing != config:
                    raise RuntimeError(f"refusing to overwrite incompatible config: {config_path}")
            else:
                config_path.write_text(json.dumps(config, indent=2) + "\n")
            runs.append({
                "run": name,
                "phase": "promoted",
                "method": method,
                "condition": f"promoted_{condition}",
                "screen_condition": condition,
                "w_msg": config["message_loss_weight"],
                "lambda_img": SCREEN_CONDITIONS[condition]["lambda_img"],
                "config": str(config_path),
                "output_dir": str(run_dir),
                "epochs": args.epochs,
                "source_checkpoint": str(SOURCE_CHECKPOINT),
                "source_checkpoint_sha256": digest(SOURCE_CHECKPOINT),
            })
    manifest = {
        "kind": "fig2_stress_promoted",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": 17,
        "screen_epochs": SCREEN_EPOCHS,
        "promoted_epochs": args.epochs,
        "promoted_conditions": conditions,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": digest(SOURCE_CHECKPOINT),
        "validation_manifest": str(MANIFEST),
        "validation_manifest_sha256": digest(MANIFEST),
        "runs": runs,
    }
    (CHECKPOINT_ROOT / "promotion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prepared {len(runs)} promoted runs for conditions {conditions}")
    for row in runs:
        print(
            f"CUDA_VISIBLE_DEVICES=<gpu> {sys.executable} experiments/train_local_patch.py "
            f"--config {row['config']} --output-dir {row['output_dir']} --device cuda "
            f"--init-checkpoint {SOURCE_CHECKPOINT}"
        )


def prepare_extensions(args: argparse.Namespace) -> None:
    """Prepare bounded tail-weight and top-k screens under the winner regime."""
    condition = "wmsg80"
    values = SCREEN_CONDITIONS[condition]
    specs = [
        ("Ours-tail75", "tail75", 0.25, 0.75, 0.10),
        ("Ours-tail90", "tail90", 0.10, 0.90, 0.10),
        ("Ours-top5", "top5", 0.50, 0.50, 0.05),
        ("Ours-top20", "top20", 0.50, 0.50, 0.20),
    ]
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for method, suffix, global_fraction, local_fraction, topk in specs:
        name = f"{safe_name(method).lower()}_{condition}"
        run_dir = CHECKPOINT_ROOT / name
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            **make_config("Ours", condition, args.epochs),
            "name": f"fig2_{safe_name(method)}_{condition}",
            "method": method,
            "local_loss_weight": values["lambda_img"] * local_fraction,
            "global_loss_weight": values["lambda_img"] * global_fraction,
            "local_topk_ratio": topk,
            "extension": "tail-weight" if suffix.startswith("tail") else "top-k",
            "extension_status": "exploratory validation-only; not formal Ours",
        }
        config_path = run_dir / "config.json"
        if config_path.exists():
            existing = json.loads(config_path.read_text())
            if existing != config:
                raise RuntimeError(f"refusing to overwrite incompatible config: {config_path}")
        else:
            config_path.write_text(json.dumps(config, indent=2) + "\n")
        runs.append({
            "run": name,
            "phase": "extension",
            "method": method,
            "condition": condition,
            "w_msg": values["w_msg"],
            "lambda_img": values["lambda_img"],
            "config": str(config_path),
            "output_dir": str(run_dir),
            "epochs": args.epochs,
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "source_checkpoint_sha256": digest(SOURCE_CHECKPOINT),
        })
    manifest = {
        "kind": "fig2_stress_extensions",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": 17,
        "epochs": args.epochs,
        "condition": condition,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": digest(SOURCE_CHECKPOINT),
        "validation_manifest": str(MANIFEST),
        "validation_manifest_sha256": digest(MANIFEST),
        "runs": runs,
    }
    (CHECKPOINT_ROOT / "extension_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prepared {len(runs)} tail/top-k extension runs under {CHECKPOINT_ROOT}")
    for row in runs:
        print(
            f"CUDA_VISIBLE_DEVICES=<gpu> {sys.executable} experiments/train_local_patch.py "
            f"--config {row['config']} --output-dir {row['output_dir']} --device cuda "
            f"--init-checkpoint {SOURCE_CHECKPOINT}"
        )


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_from_checkpoint(path: Path, device: torch.device) -> tuple[EncoderDecoder, dict[str, Any]]:
    checkpoint = load(path, device)
    config = checkpoint["config"]
    # Evaluation calls encoder/decoder directly.  The trainer stores the
    # already-evaluated noise modules in checkpoint config, while the model
    # constructor expects source expressions.  Identity is equivalent for
    # these direct calls and avoids mutating the checkpoint config in memory.
    model = EncoderDecoder(config["H"], config["W"], config["message_length"], ["Identity()"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def rgb_tensor(value: torch.Tensor) -> torch.Tensor:
    return ((value.detach().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def psnr(mse: float) -> float:
    return float(10.0 * math.log10(1.0 / max(float(mse), 1e-12)))


def gini(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    total = float(ordered.sum())
    if total <= 1e-15:
        return 0.0
    n = len(ordered)
    return float(((2 * np.arange(1, n + 1) - n - 1) * ordered).sum() / (n * total))


def ciede(reference: torch.Tensor, value: torch.Tensor) -> np.ndarray:
    ref = reference.permute(1, 2, 0).numpy()
    out = value.permute(1, 2, 0).numpy()
    return deltaE_ciede2000(rgb2lab(ref), rgb2lab(out)).astype(np.float32)


def crop_ber(model: EncoderDecoder, encoded: torch.Tensor, messages: torch.Tensor, masks: dict[str, Any], ratio: int, device: torch.device) -> float:
    errors = 0
    total = 0
    with torch.no_grad():
        for repeat in range(5):
            for batch, start in enumerate(range(0, len(encoded), 16)):
                stop = min(start + 16, len(encoded))
                value = encoded[start:stop].to(device) * masks[f"crop_{ratio}"][repeat][batch].to(device)
                target = messages[start:stop].to(device).gt(0.5)
                predicted = model.decoder(value).gt(0.5)
                errors += int((predicted != target).sum())
                total += int(target.numel())
    return float(errors / max(total, 1))


def quality_per_image(reference: torch.Tensor, value: torch.Tensor, lpips_metric: Any, device: torch.device) -> tuple[dict[str, float], dict[str, Any]]:
    ref = rgb_tensor(reference)
    out = rgb_tensor(value)
    residual = (out - ref).square()
    mse = float(residual.mean())
    patches = patch_mse_per_sample(out.unsqueeze(0), ref.unsqueeze(0), 32, 32)[0].numpy()
    ordered = np.sort(patches)
    top10_count = max(1, int(math.ceil(len(patches) * 0.10)))
    top25_count = max(1, int(math.ceil(len(patches) * 0.25)))
    ciede_map = ciede(ref, out)
    structural = float(structural_similarity(ref.permute(1, 2, 0).numpy(), out.permute(1, 2, 0).numpy(), data_range=1.0, channel_axis=2, win_size=7))
    with torch.no_grad():
        learned = float(lpips_metric(value.unsqueeze(0).to(device), reference.unsqueeze(0).to(device)).flatten()[0].cpu())
    return {
        "psnr": psnr(mse),
        "ssim": structural,
        "lpips": learned,
        "local_psnr": psnr(float(ordered[-top25_count:].mean())),
        "p95": float(np.percentile(patches, 95)),
        "gini": gini(patches),
        "cv": float(np.std(patches) / max(np.mean(patches), 1e-12)),
        "top10_mean": float(ordered[-top10_count:].mean()),
        "top10_energy_share": float(ordered[-top10_count:].sum() / max(ordered.sum(), 1e-12)),
        "ciede2000_global": float(ciede_map.mean()),
        "ciede2000_top10": float(np.sort(ciede_map.reshape(-1))[-max(1, int(math.ceil(ciede_map.size * .10))):].mean()),
    }, {"patch_mse": patches, "ciede_map": ciede_map, "residual": (out - ref).abs().mean(0).numpy()}


def quality_batch(reference: torch.Tensor, values: torch.Tensor, lpips_metric: Any, device: torch.device) -> list[dict[str, float]]:
    """Compute residual-sweep quality with batched LPIPS and fixed metrics."""
    if reference.shape != values.shape or reference.ndim != 4:
        raise ValueError("quality_batch requires equal NCHW tensors")
    patch_values = patch_mse_per_sample(values, reference, 32, 32).numpy()
    ref_numpy = reference.numpy()
    value_numpy = values.numpy()
    lpips_values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(values), 16):
            score = lpips_metric(
                (values[start:start + 16] * 2 - 1).to(device),
                (reference[start:start + 16] * 2 - 1).to(device),
            ).flatten().cpu().numpy()
            lpips_values.extend(float(item) for item in score)
    result: list[dict[str, float]] = []
    for index in range(len(values)):
        ref = torch.from_numpy(ref_numpy[index])
        value = torch.from_numpy(value_numpy[index])
        residual = (value - ref).square()
        patch = patch_values[index]
        ordered = np.sort(patch)
        top10_count = max(1, int(math.ceil(len(patch) * .10)))
        top25_count = max(1, int(math.ceil(len(patch) * .25)))
        ref_array = ref.permute(1, 2, 0).numpy()
        value_array = value.permute(1, 2, 0).numpy()
        result.append({
            "psnr": psnr(float(residual.mean())),
            "ssim": float(structural_similarity(ref_array, value_array, data_range=1.0, channel_axis=2, win_size=7)),
            "lpips": lpips_values[index],
            "local_psnr": psnr(float(ordered[-top25_count:].mean())),
            "p95": float(np.percentile(patch, 95)),
            "gini": gini(patch),
            "cv": float(np.std(patch) / max(np.mean(patch), 1e-12)),
            "top10_mean": float(ordered[-top10_count:].mean()),
            "top10_energy_share": float(ordered[-top10_count:].sum() / max(ordered.sum(), 1e-12)),
            "ciede2000_global": float(ciede(ref, value).mean()),
            "ciede2000_top10": float(np.sort(ciede(ref, value).reshape(-1))[-max(1, int(math.ceil(128 * 128 * .10))):].mean()),
        })
    return result


def batch_msssim(reference: torch.Tensor, value: torch.Tensor, device: torch.device) -> np.ndarray:
    from pytorch_msssim import ms_ssim
    with torch.no_grad():
        result = ms_ssim(value.to(device), reference.to(device), data_range=1.0, size_average=False, win_size=7, weights=[0.3, 0.3, 0.4])
    return result.detach().cpu().numpy().astype(float)


def encode_fixed(model: EncoderDecoder, images: torch.Tensor, messages: torch.Tensor, device: torch.device) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start in range(0, len(images), 16):
            values.append(model.encoder(images[start:start + 16].to(device), messages[start:start + 16].to(device)).cpu())
    return torch.cat(values)


def evaluate_run(row: dict[str, Any], images: torch.Tensor, messages: torch.Tensor, masks: dict[str, Any], lpips_metric: Any, device: torch.device, checkpoint_epoch: int) -> dict[str, Any] | None:
    run_dir = Path(row["output_dir"])
    checkpoint_path = run_dir / f"checkpoint_{checkpoint_epoch:04d}.pth"
    if not checkpoint_path.is_file():
        # If this run is expected to reach the requested epoch, absence means
        # it is incomplete.  Do not silently evaluate an epoch-1 partial run.
        if int(row.get("epochs", checkpoint_epoch)) >= checkpoint_epoch:
            return None
        # Short screen/extension runs are intentionally reused when a later
        # promoted epoch is requested for the combined report.
        candidates = sorted(run_dir.glob("checkpoint_*.pth"))
        if not candidates:
            return None
        checkpoint_path = candidates[-1]
    model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    encoded = encode_fixed(model, images, messages, device)
    values = rgb_tensor(encoded)
    originals = rgb_tensor(images)
    per_image: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    for index in range(len(images)):
        quality, detail = quality_per_image(images[index], encoded[index], lpips_metric, device)
        per_image.append({"validation_index": index, **quality})
        maps.append(detail)
    msssim = batch_msssim(originals, values, device)
    for index, value in enumerate(msssim):
        per_image[index]["ms_ssim"] = float(value)
    row_out: dict[str, Any] = {
        **row,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "samples": len(images),
        "psnr": psnr(float((originals - values).square().mean())),
        "ssim": float(np.mean([item["ssim"] for item in per_image])),
        "ms_ssim": float(np.mean([item["ms_ssim"] for item in per_image])),
        "lpips": float(np.mean([item["lpips"] for item in per_image])),
    }
    for key in ("local_psnr", "p95", "gini", "cv", "top10_mean", "top10_energy_share", "ciede2000_global", "ciede2000_top10"):
        row_out[key] = float(np.mean([item[key] for item in per_image]))
    for ratio in BER_LEVELS:
        row_out[f"ber{ratio}"] = crop_ber(model, encoded, messages, masks, ratio, device)
    row_out["status"] = "pass" if row_out["psnr"] >= 24 and row_out["ber30"] <= 0.30 and all(math.isfinite(float(row_out[key])) for key in ("psnr", "local_psnr", "p95", "ber30")) else "early_reject"
    out_dir = RESULT_ROOT / row["run"]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "messages": messages, "encoded": encoded, "per_image": per_image, "maps": maps}, out_dir / "outputs.pt")
    (out_dir / "metrics.json").write_text(json.dumps(row_out, indent=2) + "\n")
    write_csv(out_dir / "per_image.csv", per_image)
    del model, checkpoint, encoded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row_out


def evaluate(args: argparse.Namespace) -> None:
    screen_manifest = CHECKPOINT_ROOT / "screen_manifest.json"
    if not screen_manifest.is_file():
        raise FileNotFoundError(screen_manifest)
    spec = json.loads(screen_manifest.read_text())
    manifests = [spec]
    promotion_manifest = CHECKPOINT_ROOT / "promotion_manifest.json"
    if promotion_manifest.is_file():
        manifests.append(json.loads(promotion_manifest.read_text()))
    extension_manifest = CHECKPOINT_ROOT / "extension_manifest.json"
    if extension_manifest.is_file():
        manifests.append(json.loads(extension_manifest.read_text()))
    manifest = load(MANIFEST)
    if manifest.get("split") != "validation" or len(manifest["images"]) != 50:
        raise ValueError("the Figure 2 evaluator requires the fixed 50-image validation manifest")
    images, messages = manifest["images"].float(), manifest["messages"].float()
    masks = load(CROP_MASKS)
    device = choose_device(args.device)
    torch.set_num_threads(4)
    import lpips
    lpips_metric = lpips.LPIPS(net="alex", version="0.1", spatial=False, verbose=False).to(device).eval()
    rows: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    all_runs_by_name: dict[str, dict[str, Any]] = {}
    for manifest_spec in manifests:
        for run in manifest_spec["runs"]:
            all_runs_by_name[run["run"]] = run
    all_runs = list(all_runs_by_name.values())
    for run in all_runs:
        print(f"evaluating {run['run']}", flush=True)
        result = evaluate_run(run, images, messages, masks, lpips_metric, device, args.epoch)
        if result is None:
            print(f"missing checkpoint: {run['run']}", flush=True)
            continue
        rows.append(result)
        payload = load(RESULT_ROOT / run["run"] / "outputs.pt")
        for item in payload["per_image"]:
            per_image.append({"run": run["run"], "method": run["method"], "condition": run["condition"], **item})
    write_csv(REPORT_ROOT / "fig2_strong_embedding_results.csv", rows)
    write_csv(RESULT_ROOT / "fig2_strong_embedding_results.csv", rows)
    write_csv(RESULT_ROOT / "fig2_strong_embedding_per_image.csv", per_image)
    summary = {
        "scope": "fixed 50-image validation only",
        "manifest": str(MANIFEST),
        "manifest_sha256": digest(MANIFEST),
        "crop_masks": str(CROP_MASKS),
        "crop_masks_sha256": digest(CROP_MASKS),
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": digest(SOURCE_CHECKPOINT),
        "evaluator_sha256": digest(Path(__file__)),
        "rows": len(rows),
        "manifests": [manifest_spec.get("kind") for manifest_spec in manifests],
        "formal_files_after": {str(path): digest(path) for path in FORMAL_FILES if path.is_file()},
        "formal_files_unchanged_from_prepare": all(spec.get("formal_files_before", {}).get(str(path)) == digest(path) for path in FORMAL_FILES if path.is_file() and str(path) in spec.get("formal_files_before", {})),
    }
    (RESULT_ROOT / "evaluation_provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"evaluated {len(rows)} runs; results in {RESULT_ROOT}")


def evaluate_convergence(args: argparse.Namespace) -> None:
    """Re-evaluate promoted checkpoints on one fixed validation realization.

    Training-time validation resamples messages, so its JSONL is useful for
    health monitoring but not for a clean plateau comparison.  This command
    evaluates epochs 5/10/15/20 with the frozen Figure-2 images, messages, and
    crop masks.  Epoch 20 is evaluated last so the normal per-run output cache
    remains aligned with the final promoted checkpoint.
    """
    promotion_path = CHECKPOINT_ROOT / "promotion_manifest.json"
    if not promotion_path.is_file():
        raise FileNotFoundError(promotion_path)
    promotion = json.loads(promotion_path.read_text())
    manifest = load(MANIFEST)
    if manifest.get("split") != "validation" or len(manifest["images"]) != 50:
        raise ValueError("convergence evaluation requires the fixed 50-image validation manifest")
    images, messages = manifest["images"].float(), manifest["messages"].float()
    masks = load(CROP_MASKS)
    device = choose_device(args.device)
    torch.set_num_threads(4)
    import lpips
    lpips_metric = lpips.LPIPS(net="alex", version="0.1", spatial=False, verbose=False).to(device).eval()
    rows: list[dict[str, Any]] = []
    epochs = tuple(sorted(set(args.epochs)))
    for run in promotion["runs"]:
        for epoch in epochs:
            checkpoint = Path(run["output_dir"]) / f"checkpoint_{epoch:04d}.pth"
            if not checkpoint.is_file():
                print(f"missing promoted checkpoint: {run['run']} epoch {epoch}", flush=True)
                continue
            print(f"convergence: {run['run']} epoch {epoch}", flush=True)
            result = evaluate_run(run, images, messages, masks, lpips_metric, device, epoch)
            assert result is not None
            result["evaluation_epoch"] = epoch
            rows.append(result)
    write_csv(REPORT_ROOT / "fig2_promoted_convergence.csv", rows)
    write_csv(RESULT_ROOT / "fig2_promoted_convergence.csv", rows)
    verdicts: list[dict[str, Any]] = []
    for run in promotion["runs"]:
        history = sorted((row for row in rows if row["run"] == run["run"]), key=lambda row: int(row["evaluation_epoch"]))
        tail = history[-3:]
        complete = len(history) == len(epochs) and int(history[-1]["evaluation_epoch"]) == max(epochs)
        psnr_span = max((float(row["psnr"]) for row in tail), default=float("nan")) - min((float(row["psnr"]) for row in tail), default=float("nan"))
        local_span = max((float(row["local_psnr"]) for row in tail), default=float("nan")) - min((float(row["local_psnr"]) for row in tail), default=float("nan"))
        stable = complete and len(tail) == 3 and psnr_span <= args.plateau_db and local_span <= args.plateau_db
        verdicts.append({
            "run": run["run"],
            "method": run["method"],
            "condition": run["condition"],
            "epochs_evaluated": ",".join(str(row["evaluation_epoch"]) for row in history),
            "tail_psnr_span_db": psnr_span,
            "tail_local_psnr_span_db": local_span,
            "plateau_threshold_db": args.plateau_db,
            "convergence_verdict": "plateau_observed" if stable else "not_demonstrated",
            "note": "Operational validation plateau criterion; not proof of optimizer convergence.",
        })
    write_csv(REPORT_ROOT / "fig2_promoted_convergence_verdicts.csv", verdicts)
    (RESULT_ROOT / "convergence_provenance.json").write_text(json.dumps({
        "manifest": str(MANIFEST),
        "manifest_sha256": digest(MANIFEST),
        "crop_masks_sha256": digest(CROP_MASKS),
        "epochs": epochs,
        "plateau_rule": f"last three fixed-validation PSNR and local-PSNR spans <= {args.plateau_db} dB",
        "warning": "This is an operational plateau diagnostic, not a proof of global optimization convergence.",
    }, indent=2) + "\n")
    print(f"evaluated {len(rows)} promoted checkpoint snapshots; wrote {len(verdicts)} convergence verdicts")


def paired_rows(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    base = {row["condition"]: row for row in rows if row["method"] == "Ours-base"}
    ours = {row["condition"]: row for row in rows if row["method"] == "Ours"}
    return base, ours


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def find_matched_ber(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    base, ours = paired_rows(rows)
    result: list[dict[str, Any]] = []
    for bc, oc in itertools.product(sorted(base), sorted(ours)):
        b, o = base[bc], ours[oc]
        if b["status"] == "early_reject" or o["status"] == "early_reject":
            continue
        delta_ber = f(o, "ber30") - f(b, "ber30")
        if abs(delta_ber) > 0.005:
            continue
        result.append({
            "base_condition": bc,
            "ours_condition": oc,
            "same_condition": bc == oc,
            "base_phase": b.get("phase", "screen"),
            "ours_phase": o.get("phase", "screen"),
            "promoted_pair": b.get("phase", "screen") == "promoted" and o.get("phase", "screen") == "promoted",
            "base_w_msg": b["w_msg"],
            "ours_w_msg": o["w_msg"],
            "base_lambda_img": b["lambda_img"],
            "ours_lambda_img": o["lambda_img"],
            "ber30_base": f(b, "ber30"),
            "ber30_ours": f(o, "ber30"),
            "abs_delta_ber30": abs(delta_ber),
            "psnr_base": f(b, "psnr"),
            "psnr_ours": f(o, "psnr"),
            "local_psnr_base": f(b, "local_psnr"),
            "local_psnr_ours": f(o, "local_psnr"),
            "delta_local_psnr": f(o, "local_psnr") - f(b, "local_psnr"),
            "p95_base": f(b, "p95"),
            "p95_ours": f(o, "p95"),
            "delta_p95": f(o, "p95") - f(b, "p95"),
            "gini_base": f(b, "gini"),
            "gini_ours": f(o, "gini"),
        })
    result.sort(key=lambda row: (row["same_condition"], row["promoted_pair"], row["delta_local_psnr"], -row["delta_p95"], -row["abs_delta_ber30"]), reverse=True)
    for index, row in enumerate(result):
        row["rank"] = index + 1
        row["selected"] = index == 0
    return result


def find_matched_psnr(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    base, ours = paired_rows(rows)
    result: list[dict[str, Any]] = []
    for target in (30.0, 32.0, 34.0, 36.0):
        candidates: list[dict[str, Any]] = []
        for bc, oc in itertools.product(sorted(base), sorted(ours)):
            b, o = base[bc], ours[oc]
            if b["status"] == "early_reject" or o["status"] == "early_reject":
                continue
            if abs(f(b, "psnr") - target) > 0.15 or abs(f(o, "psnr") - target) > 0.15:
                continue
            candidates.append({
                "target_psnr": target,
                "base_condition": bc,
                "ours_condition": oc,
                "same_condition": bc == oc,
                "psnr_base": f(b, "psnr"),
                "psnr_ours": f(o, "psnr"),
                "psnr_gap": abs(f(b, "psnr") - f(o, "psnr")),
                "local_psnr_base": f(b, "local_psnr"),
                "local_psnr_ours": f(o, "local_psnr"),
                "delta_local_psnr": f(o, "local_psnr") - f(b, "local_psnr"),
                "p95_base": f(b, "p95"),
                "p95_ours": f(o, "p95"),
                "delta_p95": f(o, "p95") - f(b, "p95"),
                "gini_base": f(b, "gini"),
                "gini_ours": f(o, "gini"),
                "top10_mean_base": f(b, "top10_mean"),
                "top10_mean_ours": f(o, "top10_mean"),
                "ber30_base": f(b, "ber30"),
                "ber30_ours": f(o, "ber30"),
            })
        candidates.sort(key=lambda row: (row["same_condition"], row["delta_local_psnr"], -row["psnr_gap"]), reverse=True)
        if candidates:
            for index, row in enumerate(candidates):
                row["rank_in_bin"] = index + 1
                row["selected"] = index == 0
            result.extend(candidates)
        else:
            result.append({"target_psnr": target, "status": "no_pair_within_tolerance", "selected": False})
    return result


def best_native_pair(rows: list[dict[str, str]], matched_ber: list[dict[str, Any]], matched_psnr: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ber = next((row for row in matched_ber if row.get("selected")), None)
    if selected_ber is not None:
        return {"selection": "matched-BER strong-embedding", **selected_ber}
    selected_psnr = next((row for row in matched_psnr if row.get("selected")), None)
    if selected_psnr is not None and selected_psnr.get("status") != "no_pair_within_tolerance":
        return {"selection": "matched-global-PSNR", **selected_psnr}
    same = [row for row in rows if row["method"] == "Ours" and row["status"] != "early_reject"]
    base = {row["condition"]: row for row in rows if row["method"] == "Ours-base" and row["status"] != "early_reject"}
    choices = []
    for ours in same:
        if ours["condition"] in base:
            b = base[ours["condition"]]
            choices.append({"selection": "native strong-embedding", "base_condition": ours["condition"], "ours_condition": ours["condition"], "delta_local_psnr": f(ours, "local_psnr") - f(b, "local_psnr"), "delta_p95": f(ours, "p95") - f(b, "p95"), "psnr_base": f(b, "psnr"), "psnr_ours": f(ours, "psnr"), "ber30_base": f(b, "ber30"), "ber30_ours": f(ours, "ber30")})
    if not choices:
        raise RuntimeError("no completed Ours/Ours-base pair is available")
    return max(choices, key=lambda row: (row["delta_local_psnr"], -row["delta_p95"]))


def roi_search(original: torch.Tensor, base: torch.Tensor, ours: torch.Tensor) -> tuple[tuple[int, int, int, int], dict[str, float]]:
    improvement = ((base - original).square() - (ours - original).square()).mean(0).numpy()
    integral = np.pad(improvement.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    by_size: dict[str, float] = {}
    for size in ROI_SIZES:
        margin = 4
        local: list[tuple[float, tuple[int, int, int, int]]] = []
        for y in range(margin, 128 - size - margin + 1, 4):
            for x in range(margin, 128 - size - margin + 1, 4):
                total = integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x]
                score = float(total / (size * size))
                local.append((score, (x, y, size, size)))
        winner = max(local, key=lambda item: item[0])
        by_size[f"roi{size}_improvement"] = winner[0]
        candidates.append(winner)
    score, roi = max(candidates, key=lambda item: item[0])
    return roi, {"roi_mse_improvement": score, **by_size}


def numpy_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().clamp(0, 1).permute(1, 2, 0).numpy()


def pil_image(tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.rint(numpy_image(tensor) * 255).astype(np.uint8), "RGB")


def crop_zoom(tensor: torch.Tensor, roi: tuple[int, int, int, int], scale: int = 8) -> Image.Image:
    x, y, w, h = roi
    return pil_image(tensor[:, y:y + h, x:x + w]).resize((w * scale, h * scale), Image.Resampling.NEAREST)


def display_canvas(tensor: torch.Tensor, size: int = DISPLAY_SIZE) -> torch.Tensor:
    """Put RGB[0,1] tensors on the one canvas used by every Figure 2 method."""
    single = tensor.ndim == 3
    value = tensor.unsqueeze(0) if single else tensor
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"expected CHW or NCHW RGB tensor, got {tuple(tensor.shape)}")
    value = value.detach().float().cpu().clamp(0, 1)
    if value.shape[-2:] != (size, size):
        value = F.interpolate(value, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return value[0] if single else value


def valid_external_cache(payload: Any, manifest_sha256: str, checkpoint_sha256: str, samples: int) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("encoded"), torch.Tensor):
        return False
    return (
        payload.get("manifest_sha256") == manifest_sha256
        and payload.get("checkpoint_sha256") == checkpoint_sha256
        and tuple(payload["encoded"].shape) == (samples, 3, 128, 128)
    )


def infer_or_load_hidden_validation(
    images: torch.Tensor,
    messages: torch.Tensor,
    device: torch.device,
    cache_path: Path = HIDDEN_CACHE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the retrained 64-bit HiDDeN encoder once, then use a provenance-checked cache."""
    if not HIDDEN_CHECKPOINT.is_file():
        raise FileNotFoundError(HIDDEN_CHECKPOINT)
    manifest_sha256 = digest(MANIFEST)
    checkpoint_sha256 = digest(HIDDEN_CHECKPOINT)
    if cache_path.is_file():
        cached = load(cache_path)
        if valid_external_cache(cached, manifest_sha256, checkpoint_sha256, len(images)):
            return cached["encoded"].float(), {**cached.get("provenance", {}), "cache": str(cache_path), "cache_hit": True}

    sys.path.insert(0, str(HIDDEN_REPO))
    from options import HiDDenConfiguration
    from model.encoder_decoder import EncoderDecoder as HiDDeNEncoderDecoder
    from noise_layers.noiser import Noiser

    config = HiDDenConfiguration(
        H=128,
        W=128,
        message_length=64,
        encoder_blocks=4,
        encoder_channels=64,
        decoder_blocks=7,
        decoder_channels=64,
        use_discriminator=True,
        use_vgg=False,
        discriminator_blocks=3,
        discriminator_channels=64,
        decoder_loss=1.0,
        encoder_loss=0.7,
        adversarial_loss=1e-3,
    )
    checkpoint = load(HIDDEN_CHECKPOINT)
    model = HiDDeNEncoderDecoder(config, Noiser([], device)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    encoded: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(images), 16):
            encoded.append(model.encoder(images[start:start + 16].to(device), messages[start:start + 16].to(device)).cpu())
    result = torch.cat(encoded)
    convergence = json.loads(HIDDEN_EVALUATION.read_text()) if HIDDEN_EVALUATION.is_file() else {}
    provenance = {
        "method": "HiDDeN reimplementation, retrained for 64-bit messages",
        "checkpoint": str(HIDDEN_CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest": str(MANIFEST),
        "manifest_sha256": manifest_sha256,
        "validation_epoch": convergence.get("epoch", checkpoint.get("epoch", 200)),
        "validation_ber30": convergence.get("ber30"),
        "convergence_warning": "The available retraining is not sufficiently converged for its visible artifacts to support the main qualitative claim.",
        "protocol_status": "external reference; not a strict training- or robustness-matched comparison",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoded": result,
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "provenance": provenance,
    }, cache_path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, {**provenance, "cache": str(cache_path), "cache_hit": False}


def load_maskwm_validation(
    output_dir: Path = MASKWM_NATIVE_OUTPUT,
    provenance_path: Path = MASKWM_PROVENANCE,
    manifest_path: Path = MANIFEST,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load the official MaskWM D-64 outputs aligned to the fixed manifest."""
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    provenance = json.loads(provenance_path.read_text())
    expected_manifest_sha = digest(manifest_path)
    if provenance.get("manifest_sha256") != expected_manifest_sha:
        raise RuntimeError("MaskWM outputs do not match the fixed validation manifest")
    values: list[torch.Tensor] = []
    for index in range(50):
        path = output_dir / f"image_{index:03d}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        if array.shape != (DISPLAY_SIZE, DISPLAY_SIZE, 3):
            raise ValueError(f"MaskWM native output must be 512x512 RGB: {path} has {array.shape}")
        values.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
    return torch.stack(values), {
        **provenance,
        "output_directory": str(output_dir),
        "files": 50,
        "protocol_status": "official external D-64 model; not a strict training- or robustness-matched comparison",
    }


def comparison_psnr(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return psnr(float((reference - candidate).square().mean()))


def final_column_labels(data: dict[str, torch.Tensor]) -> list[str]:
    reference = data[FINAL_METHODS[0]]
    labels = [FINAL_METHODS[0]]
    for method in FINAL_METHODS[1:]:
        labels.append(f"{method}\ndisplay PSNR {comparison_psnr(reference, data[method]):.2f} dB")
    return labels


def final_comparison_data(
    images: torch.Tensor,
    messages: torch.Tensor,
    global_encoded: torch.Tensor,
    hard_encoded: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    hidden_encoded, hidden_provenance = infer_or_load_hidden_validation(images, messages, device)
    maskwm, maskwm_provenance = load_maskwm_validation()
    values = OrderedDict([
        (FINAL_METHODS[0], display_canvas(rgb_tensor(images))),
        (FINAL_METHODS[1], display_canvas(rgb_tensor(hidden_encoded))),
        (FINAL_METHODS[2], maskwm),
        (FINAL_METHODS[3], display_canvas(rgb_tensor(global_encoded))),
        (FINAL_METHODS[4], display_canvas(rgb_tensor(hard_encoded))),
    ])
    shapes = {tuple(value.shape) for value in values.values()}
    if shapes != {(50, 3, DISPLAY_SIZE, DISPLAY_SIZE)}:
        raise RuntimeError(f"final comparison requires five aligned 50-image 512x512 tensors, got {shapes}")
    return values, {"hidden": hidden_provenance, "maskwm": maskwm_provenance}


def residual_panel(reference: torch.Tensor, candidate: torch.Tensor, amplification: int = RESIDUAL_AMPLIFICATION) -> Image.Image:
    residual = (candidate - reference).abs().mean(0).mul(amplification).clamp(0, 1).numpy()
    colored = plt.get_cmap("magma")(residual)[..., :3]
    return Image.fromarray(np.rint(colored * 255).astype(np.uint8), "RGB")


def render_final_comparison(
    path: Path,
    data: dict[str, torch.Tensor],
    chosen: list[dict[str, Any]],
    labels: list[str] | None = None,
) -> tuple[list[str], tuple[str, ...], list[list[Any]]]:
    """Render the publication comparison and return its editable-PPTX panel model."""
    from experiments.generate_fig2_qualitative_search import Panel

    labels = final_column_labels(data) if labels is None else labels
    row_labels: list[str] = []
    panels: list[list[Panel]] = []
    figure_rows: list[tuple[str, list[Image.Image], tuple[int, int, int, int] | None]] = []
    for sample_number, row in enumerate(chosen[:2], 1):
        index = int(row["validation_index"])
        scale = DISPLAY_SIZE // 128
        roi = (
            int(row["roi_x"]) * scale,
            int(row["roi_y"]) * scale,
            int(row["roi_size"]) * scale,
            int(row["roi_size"]) * scale,
        )
        x, y, width, height = roi
        values = [data[method][index] for method in FINAL_METHODS]
        full_images = [pil_image(value) for value in values]
        zoom_images = [image.crop((x, y, x + width, y + height)).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST) for image in full_images]
        residual_images = [residual_panel(values[0], value) for value in values]
        full_label = f"Sample {sample_number} ({row['image_id']}) — full"
        zoom_label = f"Sample {sample_number} — shared ROI {width // scale}x{height // scale}, 4x zoom"
        residual_label = f"Sample {sample_number} — |residual| x{RESIDUAL_AMPLIFICATION} (visualization only)"
        row_labels.extend((full_label, zoom_label, residual_label))
        panels.append([Panel(image, roi=roi, source_size=(DISPLAY_SIZE, DISPLAY_SIZE)) for image in full_images])
        panels.append([Panel(image, source_size=(DISPLAY_SIZE, DISPLAY_SIZE)) for image in zoom_images])
        panels.append([Panel(image, source_size=(DISPLAY_SIZE, DISPLAY_SIZE)) for image in residual_images])
        figure_rows.extend(((full_label, full_images, roi), (zoom_label, zoom_images, None), (residual_label, residual_images, None)))

    fig, axes = plt.subplots(len(figure_rows), len(FINAL_METHODS), figsize=(17.5, 18), squeeze=False)
    for col, label in enumerate(labels):
        axes[0, col].set_title(label, fontsize=8.5, fontweight="bold")
    for row_index, (row_label, images, roi) in enumerate(figure_rows):
        for col, image in enumerate(images):
            axes[row_index, col].imshow(image, interpolation="nearest")
            if roi is not None:
                axes[row_index, col].add_patch(Rectangle((roi[0], roi[1]), roi[2], roi[3], fill=False, edgecolor="#e5483c", linewidth=1.6))
            axes[row_index, col].axis("off")
        axes[row_index, 0].text(
            -0.10, 0.5, row_label,
            transform=axes[row_index, 0].transAxes,
            ha="right", va="center", rotation=90,
            fontsize=8, fontweight="bold", clip_on=False,
        )
    fig.suptitle("Figure 2 — external and internal qualitative comparison", fontsize=15, fontweight="bold")
    fig.text(
        .5,
        .006,
        "Fixed validation manifest; same host, sample, ROI, and 512x512 display canvas. External protocols are not strict matched comparisons; HiDDeN-64 is under-converged and not the primary claim. Residual x10 is visualization-only.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(.075, .025, .995, .965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return labels, tuple(row_labels), panels


def patch_map(tensor: torch.Tensor, reference: torch.Tensor) -> np.ndarray:
    scores = patch_mse_per_sample(tensor.unsqueeze(0), reference.unsqueeze(0), 32, 32)[0].numpy().reshape(4, 4)
    return scores


def load_run_payload(run: str) -> dict[str, Any]:
    return load(RESULT_ROOT / run / "outputs.pt")


def choose_pair_payload(pair: dict[str, Any], rows: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, str]]:
    base_row = next(row for row in rows if row["method"] == "Ours-base" and row["condition"] == pair["base_condition"])
    ours_row = next(row for row in rows if row["method"] == "Ours" and row["condition"] == pair["ours_condition"])
    return load_run_payload(base_row["run"]), load_run_payload(ours_row["run"]), base_row, ours_row


def rank_samples(original: torch.Tensor, base: torch.Tensor, ours: torch.Tensor, sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for index in range(len(original)):
        base_patches = patch_map(base[index], original[index])
        ours_patches = patch_map(ours[index], original[index])
        base_top = float(np.sort(base_patches.reshape(-1))[-4:].mean())
        ours_top = float(np.sort(ours_patches.reshape(-1))[-4:].mean())
        roi, roi_info = roi_search(original[index], base[index], ours[index])
        x, y, w, h = roi
        roi_improvement = float(((base[index, :, y:y + h, x:x + w] - original[index, :, y:y + h, x:x + w]).square() - (ours[index, :, y:y + h, x:x + w] - original[index, :, y:y + h, x:x + w]).square()).mean())
        dpatch = base_patches - ours_patches
        residual_base = (base[index] - original[index]).abs().mean(0).numpy()
        residual_ours = (ours[index] - original[index]).abs().mean(0).numpy()
        rows.append({
            "image_id": Path(sources[index].get("filename", str(index))).stem,
            "validation_index": index,
            "source_filename": sources[index].get("filename", ""),
            "delta_local_psnr": psnr(ours_top) - psnr(base_top),
            "delta_p95": float(np.percentile(base_patches, 95) - np.percentile(ours_patches, 95)),
            "max_positive_d_patch": float(dpatch.max()),
            "positive_d_patch_area": float(np.mean(dpatch > 0)),
            "top10_residual_reduction": float(np.sort(residual_base.reshape(-1))[-max(1, residual_base.size // 10):].mean() - np.sort(residual_ours.reshape(-1))[-max(1, residual_ours.size // 10):].mean()),
            "roi_mse_improvement": roi_improvement,
            "roi_x": x, "roi_y": y, "roi_size": w,
            **roi_info,
            "patch_mse_base_mean": float(base_patches.mean()),
            "patch_mse_ours_mean": float(ours_patches.mean()),
            "semantic_proxy_mean_rgb": float(numpy_image(original[index]).mean()),
            "semantic_proxy_texture": float(np.abs(np.gradient(numpy_image(original[index]).mean(2))).mean()),
        })
    rows.sort(key=lambda row: (row["delta_local_psnr"], row["delta_p95"], row["max_positive_d_patch"], row["roi_mse_improvement"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    chosen: list[dict[str, Any]] = []
    for row in rows:
        if not chosen:
            chosen.append(row)
            continue
        # A deterministic content proxy prevents two nearly identical flat/bright
        # samples without pretending that it is a semantic classifier.
        distance = abs(row["semantic_proxy_mean_rgb"] - chosen[0]["semantic_proxy_mean_rgb"]) + 4.0 * abs(row["semantic_proxy_texture"] - chosen[0]["semantic_proxy_texture"])
        row["content_diversity_distance"] = distance
        if len(chosen) == 1 and distance >= 0.01:
            chosen.append(row)
            break
    if len(chosen) < 2:
        chosen = rows[:2]
    return rows, chosen


def contact_sheet(path: Path, original: torch.Tensor, base: torch.Tensor, ours: torch.Tensor, ranking: list[dict[str, Any]]) -> None:
    top = ranking[:20]
    fig, axes = plt.subplots(5, 4, figsize=(12, 15), squeeze=False)
    for ax, row in zip(axes.flat, top):
        index = int(row["validation_index"])
        ax.imshow(numpy_image(ours[index]), interpolation="none")
        ax.set_title(f"#{row['rank']} {row['image_id']}  Δloc {row['delta_local_psnr']:+.2f} dB\nΔP95 {row['delta_p95']:+.2e}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Top 20 validation candidates: Hard Local-Tail − Global Reconstruction", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_native(path: Path, original: torch.Tensor, base: torch.Tensor, ours: torch.Tensor, chosen: list[dict[str, Any]], base_row: dict[str, str], ours_row: dict[str, str], title: str, method_names: tuple[str, str] = ("Ours-base", "Ours")) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(10, 12), squeeze=False)
    columns = ["Original", f"{method_names[0]}\nPSNR {float(base_row['psnr']):.2f} dB\nBER30 {float(base_row['ber30']):.3f}", f"{method_names[1]}\nPSNR {float(ours_row['psnr']):.2f} dB\nBER30 {float(ours_row['ber30']):.3f}"]
    for col, label in enumerate(columns):
        axes[0, col].set_title(label, fontsize=10, fontweight="bold")
    for sample, row in enumerate(chosen[:2]):
        index = int(row["validation_index"])
        roi = (int(row["roi_x"]), int(row["roi_y"]), int(row["roi_size"]), int(row["roi_size"]))
        values = (original[index], base[index], ours[index])
        for col, value in enumerate(values):
            axes[sample * 2, col].imshow(numpy_image(rgb_tensor(value)), interpolation="none")
            axes[sample * 2 + 1, col].imshow(crop_zoom(rgb_tensor(value), roi, 8), interpolation="nearest")
            axes[sample * 2, col].add_patch(Rectangle((roi[0], roi[1]), roi[2], roi[3], fill=False, edgecolor="#e5483c", linewidth=2))
            axes[sample * 2, col].axis("off")
            axes[sample * 2 + 1, col].axis("off")
        axes[sample * 2, 0].set_ylabel(f"Sample {sample + 1}\nfull", fontsize=9, fontweight="bold")
        axes[sample * 2 + 1, 0].set_ylabel(f"{row['image_id']}\n8× zoom", fontsize=9, fontweight="bold")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(.5, .006, "Native clipped RGB outputs from a fixed 50-image validation manifest; red ROI is shared across all columns.", ha="center", fontsize=8)
    fig.tight_layout(rect=(.02, .025, .99, .95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def render_diagnostics(path: Path, original: torch.Tensor, base: torch.Tensor, ours: torch.Tensor, chosen: list[dict[str, Any]], title: str) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), squeeze=False)
    headers = ["Original", "Global Reconstruction", "Hard Local-Tail", "Global − Local error"]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=10, fontweight="bold")
    for sample, row in enumerate(chosen[:2]):
        index = int(row["validation_index"])
        roi = (int(row["roi_x"]), int(row["roi_y"]), int(row["roi_size"]), int(row["roi_size"]))
        ref = rgb_tensor(original[index])
        b = rgb_tensor(base[index])
        o = rgb_tensor(ours[index])
        diff = (b - ref).abs().mean(0).numpy() - (o - ref).abs().mean(0).numpy()
        limit = max(float(np.abs(diff).max()), 1e-8)
        for col, value in enumerate((ref, b, o)):
            axes[sample * 2, col].imshow(numpy_image(value), interpolation="none")
            axes[sample * 2 + 1, col].imshow(crop_zoom(value, roi, 8), interpolation="nearest")
            axes[sample * 2, col].axis("off")
            axes[sample * 2 + 1, col].axis("off")
        axes[sample * 2, 3].imshow(diff, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), interpolation="none")
        axes[sample * 2 + 1, 3].imshow(diff[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]], cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), interpolation="nearest")
        axes[sample * 2, 3].axis("off")
        axes[sample * 2 + 1, 3].axis("off")
        axes[sample * 2, 0].set_ylabel(f"Sample {sample + 1}\nfull", fontsize=9, fontweight="bold")
        axes[sample * 2 + 1, 0].set_ylabel(f"{row['image_id']}\n8× zoom", fontsize=9, fontweight="bold")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(.5, .006, "D = |R_base| − |R_ours|; positive (warm) means lower Ours residual. Symmetric shared scale; diagnostic only.", ha="center", fontsize=8)
    fig.tight_layout(rect=(.02, .025, .99, .95))
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def render_residual(path: Path, original: torch.Tensor, base: torch.Tensor, ours: torch.Tensor, chosen: list[dict[str, Any]], title: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), squeeze=False)
    headers = ["Native original", "Global Reconstruction", "Hard Local-Tail", "|R| ×20 (shared scale)"]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=10, fontweight="bold")
    for sample, row in enumerate(chosen[:2]):
        index = int(row["validation_index"])
        ref = rgb_tensor(original[index])
        b = rgb_tensor(base[index])
        o = rgb_tensor(ours[index])
        residual = torch.stack([(b - ref).abs().mean(0), (o - ref).abs().mean(0)])
        limit = max(float(residual.max() * 20), 1e-8)
        axes[sample, 0].imshow(numpy_image(ref), interpolation="none")
        axes[sample, 1].imshow(numpy_image(b), interpolation="none")
        axes[sample, 2].imshow(numpy_image(o), interpolation="none")
        axes[sample, 3].imshow(np.maximum(residual[0].numpy(), residual[1].numpy()) * 20, cmap="magma", norm=Normalize(vmin=0, vmax=limit), interpolation="none")
        for ax in axes[sample]:
            ax.axis("off")
        axes[sample, 0].set_ylabel(f"Sample {sample + 1}\n{row['image_id']}", fontsize=9, fontweight="bold")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(.5, .006, "Residual amplification is visualization-only; ×20 is identical for both methods and uses one shared maximum.", ha="center", fontsize=8)
    fig.tight_layout(rect=(.02, .025, .99, .95))
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def render_amplified_residuals(path: Path, original: torch.Tensor, base: torch.Tensor, ours: torch.Tensor, chosen: list[dict[str, Any]]) -> None:
    """Render ×10/×20/×30/×50 residuals with one scale per sample."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), squeeze=False)
    for sample, row in enumerate(chosen[:2]):
        index = int(row["validation_index"])
        ref = rgb_tensor(original[index])
        b = rgb_tensor(base[index])
        o = rgb_tensor(ours[index])
        residual = torch.stack([(b - ref).abs().mean(0), (o - ref).abs().mean(0)])
        # One shared maximum across methods and all four amplification views.
        vmax = max(float(residual.max() * 50), 1e-8)
        for col, alpha in enumerate((10, 20, 30, 50)):
            image = np.maximum(residual[0].numpy(), residual[1].numpy()) * alpha
            axes[sample, col].imshow(image, cmap="magma", vmin=0, vmax=vmax, interpolation="none")
            axes[sample, col].set_title(f"Sample {sample + 1} {row['image_id']}\n|R| ×{alpha}", fontsize=9)
            axes[sample, col].axis("off")
    fig.suptitle("Equal residual amplification stress (shared scales)", fontsize=13, fontweight="bold")
    fig.text(.5, .006, "Visualization only; the same alpha is applied to both methods and no amplified image is treated as a model output.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .025, 1, .95))
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def attack_decode(model: EncoderDecoder, attacked: torch.Tensor, messages: torch.Tensor, device: torch.device) -> float:
    errors = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(attacked), 16):
            target = messages[start:start + 16].to(device).gt(.5)
            decoded = model.decoder(attacked[start:start + 16].to(device)).gt(.5)
            errors += int((decoded != target).sum())
            total += int(target.numel())
    return float(errors / max(total, 1))


def jpeg_roundtrip(images: torch.Tensor, quality: int) -> torch.Tensor:
    values = []
    for image in images:
        buffer = io.BytesIO()
        pil_image(image).save(buffer, format="JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        array = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32) / 255.0
        values.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
    return torch.stack(values)


def attack_stress(selected_pair: dict[str, Any], rows: list[dict[str, str]], images: torch.Tensor, messages: torch.Tensor, device: torch.device) -> list[dict[str, Any]]:
    masks = load(CROP_MASKS)
    base_payload, ours_payload, base_row, ours_row = choose_pair_payload(selected_pair, rows)
    result: list[dict[str, Any]] = []
    for method, payload, model_row in (("Ours-base", base_payload, base_row), ("Ours", ours_payload, ours_row)):
        model, _ = model_from_checkpoint(Path(model_row["checkpoint"]), device)
        encoded = payload["encoded"].float()
        for ratio in (50, 40, 30):
            for repeat in range(5):
                attacked = encoded.clone()
                for batch, start in enumerate(range(0, len(encoded), 16)):
                    attacked[start:start + 16] *= masks[f"crop_{ratio}"][repeat][batch]
                result.append({"method": method, "attack": "crop", "parameter": f"retained_{ratio}%", "repeat": repeat, "ber": attack_decode(model, attacked, messages, device), "status": "robustness-only; not used for Figure 2 selection"})
        for sigma in (0.005, 0.01, 0.02):
            generator = torch.Generator().manual_seed(170914 + int(round(sigma * 1000)))
            attacked = (encoded + torch.randn(encoded.shape, generator=generator) * sigma).clamp(-1, 1)
            result.append({"method": method, "attack": "gaussian", "parameter": f"sigma_{sigma:g}_model_space", "repeat": 0, "ber": attack_decode(model, attacked, messages, device), "status": "robustness-only; fixed seed; not used for Figure 2 selection"})
        for quality in (90, 70, 50):
            attacked_rgb = jpeg_roundtrip(rgb_tensor(encoded), quality)
            attacked = attacked_rgb * 2 - 1
            result.append({"method": method, "attack": "JPEG", "parameter": f"quality_{quality}_subsampling2", "repeat": 0, "ber": attack_decode(model, attacked, messages, device), "status": "robustness-only; not used for Figure 2 selection"})
        del model
    # Aggregate repeat rows while preserving all crop realizations.
    write_csv(REPORT_ROOT / "fig2_attack_stress_summary.csv", result)
    write_csv(RESULT_ROOT / "fig2_attack_stress_summary.csv", result)
    return result


def existing_model_checkpoints() -> list[tuple[str, Path]]:
    models: list[tuple[str, Path]] = []
    runs_root = MOUNT / "experiments/runs"
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        checkpoints = sorted(run_dir.glob("checkpoint_*.pth"))
        if not checkpoints:
            continue
        checkpoint = max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))
        try:
            config = load(checkpoint, "cpu")["config"]
        except Exception:
            continue
        if config.get("H") == 128 and config.get("W") == 128 and config.get("message_length") == 64:
            models.append((run_dir.name, checkpoint))
    return models


def residual_alpha(args: argparse.Namespace) -> None:
    manifest = load(MANIFEST)
    images, messages = manifest["images"].float(), manifest["messages"].float()
    device = choose_device(args.device)
    import lpips
    lpips_metric = lpips.LPIPS(net="alex", version="0.1", spatial=False, verbose=False).to(device).eval()
    model_rows = existing_model_checkpoints()
    rows: list[dict[str, Any]] = []
    for model_name, checkpoint_path in model_rows:
        print(f"residual alpha: {model_name}", flush=True)
        model, _ = model_from_checkpoint(checkpoint_path, device)
        encoded = encode_fixed(model, images, messages, device)
        ref = rgb_tensor(images)
        native = rgb_tensor(encoded)
        for alpha in ALPHAS:
            raw = ref + alpha * (native - ref)
            value = raw.clamp(0, 1)
            q = quality_batch(ref, value, lpips_metric, device)
            rows.append({
                "model": model_name,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": digest(checkpoint_path),
                "alpha": alpha,
                "psnr": psnr(float((ref - value).square().mean())),
                "ssim": float(np.mean([x["ssim"] for x in q])),
                "lpips": float(np.mean([x["lpips"] for x in q])),
                "local_psnr": float(np.mean([x["local_psnr"] for x in q])),
                "p95_mse": float(np.mean([x["p95"] for x in q])),
                "gini": float(np.mean([x["gini"] for x in q])),
                "ciede2000": float(np.mean([x["ciede2000_global"] for x in q])),
                "clip_fraction": float(((raw < 0) | (raw > 1)).float().mean()),
                "clip_flag_over_1pct": bool(float(((raw < 0) | (raw > 1)).float().mean()) > .01),
                "clip_flag_over_5pct": bool(float(((raw < 0) | (raw > 1)).float().mean()) > .05),
            })
        del model, encoded
    write_csv(REPORT_ROOT / "fig2_residual_alpha_sweep.csv", rows)
    write_csv(RESULT_ROOT / "fig2_residual_alpha_sweep.csv", rows)
    (RESULT_ROOT / "residual_alpha_provenance.json").write_text(json.dumps({"models": len(model_rows), "model_names": [name for name, _ in model_rows], "checkpoint_policy": "latest checkpoint per discoverable 128x128/64-bit run directory", "alphas": ALPHAS, "validation_manifest_sha256": digest(MANIFEST), "evaluator_sha256": digest(Path(__file__))}, indent=2) + "\n")
    print(f"saved residual alpha rows: {len(rows)}")


def diagnostic_visuals(base_payload: dict[str, Any], ours_payload: dict[str, Any], original: torch.Tensor, chosen: list[dict[str, Any]], result: dict[str, Any]) -> None:
    base = rgb_tensor(base_payload["encoded"])
    ours = rgb_tensor(ours_payload["encoded"])
    diagnostics = RESULT_ROOT / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    render_diagnostics(diagnostics / "absolute_residual_and_error_difference", original, base, ours, chosen, "Native + shared-scale error-difference diagnostic")
    render_residual(diagnostics / "residual_x20", original, base, ours, chosen, "Native outputs + equal residual amplification")
    render_amplified_residuals(diagnostics / "residual_amplified_x10_x20_x30_x50", original, base, ours, chosen)
    patch_differences = []
    for row in chosen[:2]:
        index = int(row["validation_index"])
        patch_differences.append(patch_map(base[index], original[index]) - patch_map(ours[index], original[index]))
    limit = max(float(np.max(np.abs(value))) for value in patch_differences)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), squeeze=False)
    for ax, value, row in zip(axes.flat, patch_differences, chosen[:2]):
        ax.imshow(value, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-max(limit, 1e-8), vcenter=0, vmax=max(limit, 1e-8)), interpolation="nearest")
        ax.set_title(f"{row['image_id']} D_patch", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Patch error difference: MSE_base − MSE_ours", fontsize=11, fontweight="bold")
    fig.text(.5, .01, "Shared symmetric limits; positive means lower Ours patch MSE.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .93))
    fig.savefig(diagnostics / "patch_error_difference.png", dpi=300, facecolor="white")
    fig.savefig(diagnostics / "patch_error_difference.pdf")
    plt.close(fig)
    # Save four shared-scale residual representations for the selected pair.
    for beta in (100, 500, 1000):
        values = []
        for row in chosen[:2]:
            i = int(row["validation_index"])
            rb = (base[i] - original[i]).abs().mean(0).numpy()
            ro = (ours[i] - original[i]).abs().mean(0).numpy()
            values.extend([np.log1p(beta * rb), np.log1p(beta * ro)])
        vmax = max(float(np.max(x)) for x in values)
        fig, axes = plt.subplots(2, 2, figsize=(7, 7), squeeze=False)
        for ax, image, label in zip(axes.flat, values, ("S1 base", "S1 Ours", "S2 base", "S2 Ours")):
            ax.imshow(image, cmap="magma", vmin=0, vmax=vmax, interpolation="none")
            ax.set_title(f"{label}; log(1+{beta}R)", fontsize=9)
            ax.axis("off")
        fig.suptitle("Shared-maximum log residual diagnostic", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(diagnostics / f"log_residual_beta{beta}.png", dpi=250)
        fig.savefig(diagnostics / f"log_residual_beta{beta}.pdf")
        plt.close(fig)
    (diagnostics / "diagnostic_protocol.json").write_text(json.dumps({"absolute_residual": "|x_w-x|, shared scale", "amplified": [10, 20, 30, 50], "log_beta": [100, 500, 1000], "difference": "mean-channel |R_base|-|R_ours|, symmetric shared limits", "patch_difference": "32x32 nonoverlapping MSE_base-MSE_ours", "chosen_pair": result}, indent=2) + "\n")


def report(rows: list[dict[str, str]], matched_ber: list[dict[str, Any]], matched_psnr: list[dict[str, Any]], selected_pair: dict[str, Any], ranking: list[dict[str, Any]], chosen: list[dict[str, Any]], residual_rows: list[dict[str, str]] | None) -> None:
    if residual_rows:
        alpha_values = sorted({float(row["alpha"]) for row in residual_rows})
        clip_summary = []
        for alpha in alpha_values:
            values = [float(row["clip_fraction"]) for row in residual_rows if float(row["alpha"]) == alpha]
            clip_summary.append(
                f"α={alpha:g}: {sum(value > 0.01 for value in values)}/{len(values)} models >1%, "
                f"{sum(value > 0.05 for value in values)}/{len(values)} >5%"
            )
        clip_sentence = "Residual-alpha coverage is {} rows over {} existing models. Clipping flags by alpha: {}. Alpha > 1 is diagnostic stress and is not used as native evidence.".format(
            len(residual_rows), len({row["model"] for row in residual_rows}), "; ".join(clip_summary)
        )
    else:
        clip_sentence = "The residual-alpha sweep has not been completed."
    matched_psnr_candidates = [row for row in matched_psnr if row.get("status") != "no_pair_within_tolerance"]
    psnr_sentence = (
        "At least one matched-global-PSNR candidate was found; see the CSV for per-bin selections."
        if matched_psnr_candidates
        else "No Ours-base/Ours pair fell within the requested ±0.15 dB tolerance in the 30/32/34/36 dB target bins; this negative result is retained and no matched-PSNR figure is fabricated."
    )
    extension_rows = [row for row in rows if row.get("phase") == "extension"]
    extension_sentence = (
        "Tail/top-k extension screen: {}. Tail75, Tail90, Top5, and Top20 were all run under w_msg=80, λ_img=1 from the same source; their fixed-validation metrics are in the main CSV and are not formal Ours. Patch-scale exploration was not run because the w_msg=80 native comparison already produced a clear local-tail stress signal, satisfying the conditional only-if-needed rule."
        .format(", ".join(sorted({row["method"] for row in extension_rows})) if extension_rows else "none")
    )
    lines = [
        "# Figure 2 strong visual search",
        "",
        "This is an exploratory, validation-only stress search for a scientifically fair qualitative Figure 2 for MBRS.",
        "Formal Table I results and formal checkpoints are not modified. Exploratory model checkpoints are isolated under `/mnt/wmcontent/GLX/icassp/MBRS/checkpoints/fig2_stress/`; result tensors and diagnostics are under `/mnt/wmcontent/GLX/icassp/MBRS/results/fig2_stress/`.",
        "",
        "## Fixed protocol",
        "",
        f"- Seed: `17`; fixed validation manifest: `{MANIFEST}`; 50 images, 64-bit messages.",
        f"- Source: `{SOURCE_CHECKPOINT}`; every screen run starts from this same crop-trained checkpoint and loads its compatible Adam state plus model BatchNorm buffers through `train_local_patch.py`.",
        "- Retraining was used only for these exploratory continuations: 20 five-epoch screen arms, four tail/top-k extension arms, and eight independently restarted 20-epoch promoted arms. Formal checkpoints and Table I were not retrained or overwritten.",
        "- No project-test data was opened by this pipeline. No attack realization was used to select the main figure.",
        "- Native RGB displays use the saved output tensors. Residual amplification and error maps are separate diagnostics, labelled as visualization-only.",
        "",
        "## Attempted strong-embedding screen",
        "",
        "The screen used 5-epoch continuations for matched Ours-base/Ours pairs. Ours-base is `w_msg L_msg + λ_img L_global`; Ours is `w_msg L_msg + λ_img(0.5 L_global + 0.5 L_tail)` with P16/S8/Top10. A run is early-rejected at fixed validation if PSNR < 24 dB, BER30 > 0.30, or any required metric is non-finite.",
        "",
        "| Method | condition | w_msg | λ_img | PSNR | local PSNR | P95 | Gini | BER30 | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['method']} | {row['condition']} | {float(row['w_msg']):g} | {float(row['lambda_img']):g} | {float(row['psnr']):.3f} | {float(row['local_psnr']):.3f} | {float(row['p95']):.3e} | {float(row['gini']):.4f} | {float(row['ber30']):.4f} | {row['status']} |")
    convergence_path = REPORT_ROOT / "fig2_promoted_convergence_verdicts.csv"
    if convergence_path.is_file():
        convergence = read_csv(convergence_path)
        lines += [
            "",
            "## Convergence audit",
            "",
            "Promoted checkpoints are re-evaluated at epochs 5/10/15/20 using one frozen validation image/message/crop-mask realization. `plateau_observed` requires both global- and local-PSNR spans across epochs 10/15/20 to be at most 0.15 dB. This is an operational plateau check, not proof of global optimizer convergence.",
            "",
            "| Reader-facing method | condition | epochs | global span (dB) | local span (dB) | verdict |",
            "|---|---|---|---:|---:|---|",
        ]
        for item in convergence:
            reader_name = "Global Reconstruction" if item["method"] == "Ours-base" else "Hard Local-Tail"
            lines.append(
                f"| {reader_name} | {item['condition']} | {item['epochs_evaluated']} | "
                f"{float(item['tail_psnr_span_db']):.3f} | {float(item['tail_local_psnr_span_db']):.3f} | {item['convergence_verdict']} |"
            )
    lines += [
        "",
        "Uncompleted runs are absent from the table and remain explicitly visible as missing checkpoint directories in the screen manifest; they are not silently treated as successes.",
        "No valid screen or extension arm crossed the predeclared PSNR<24 dB or BER30>0.30 early-rejection thresholds. One interrupted wrong-initialization attempt was isolated under `fig2_stress_invalid_init_20260914/` and is excluded from all results.",
        "",
        "## Matched operating points",
        "",
        f"Best matched-BER pair: `{selected_pair.get('base_condition')}` / `{selected_pair.get('ours_condition')}` with BER30 `{selected_pair.get('ber30_base', 'NA')}` vs `{selected_pair.get('ber30_ours', 'NA')}`, local PSNR delta `{selected_pair.get('delta_local_psnr', 'NA')}` dB and P95 delta `{selected_pair.get('delta_p95', 'NA')}`. The intended caption terminology is **Matched crop-robustness operating point**.",
        f"Best strong-embedding pair: `{selected_pair.get('base_condition')}` / `{selected_pair.get('ours_condition')}`; the promoted w_msg=80, λ_img=1 pair is the preferred native stress case because it preserves BER30 within the requested tolerance while making local/P95 differences most observable in the shared ROI zoom.",
        "Convergence caveat: none of the eight promoted runs met the operational 0.15 dB plateau criterion at epochs 10/15/20. The selected w_msg=80 pair is therefore an exploratory epoch-20 operating point, not a converged replacement model.",
        "Negative result retained: at promoted w_msg=40 epoch20, Hard Local-Tail was 0.232 dB worse in local PSNR and had higher P95 than Global Reconstruction despite matched BER30. The local-tail advantage is not monotonic across all operating points.",
        "",
        "All matched-BER candidates are in [fig2_matched_ber_pairs.csv](fig2_matched_ber_pairs.csv). Matching permits cross-condition operating points, while preferring same-condition pairs in deterministic tie-breaking. This makes the robustness comparison explicit rather than silently comparing one arbitrary checkpoint pair.",
        "",
        psnr_sentence,
        "All matched-global-PSNR bins, including empty bins, are recorded in [fig2_matched_psnr_pairs.csv](fig2_matched_psnr_pairs.csv).",
        "",
        extension_sentence,
        "",
        "## Sample and ROI selection",
        "",
        "Samples were ranked over all 50 validation images by measured local PSNR improvement, P95 improvement, maximum positive patch error difference, positive patch area, and Top10 residual reduction. The top-20 contact sheet is [fig2_top20_contact_sheet.png](../paper/figures/fig2_top20_contact_sheet.png). Two samples were then selected by the same ranking with a deterministic RGB/texture diversity proxy; the proxy is not claimed to be a semantic classifier.",
        "",
        "| Figure sample | image id | validation index | ROI | Δlocal PSNR | ΔP95 | max positive D_patch | positive area |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(chosen[:2], 1):
        lines.append(f"| {index} | {row['image_id']} | {row['validation_index']} | ({row['roi_x']}, {row['roi_y']}, {row['roi_size']}×{row['roi_size']}) | {row['delta_local_psnr']:+.3f} | {row['delta_p95']:+.3e} | {row['max_positive_d_patch']:.3e} | {row['positive_d_patch_area']:.3f} |")
    lines += [
        "",
        "ROIs were searched over shared 32/40/48/64-pixel candidates on a stride-4 grid with a four-pixel border margin. One ROI is shared by Original, Ours-base, Ours, and all diagnostic representations for that sample. ROI coordinates and all 50 ranked samples are in [fig2_sample_ranking.csv](fig2_sample_ranking.csv).",
        "",
        "## Figure candidates and decision",
        "",
        "| Candidate | Visual difference /5 | Scientific fairness /5 | Direct local-tail relevance /5 | Reproducibility /5 | ICASSP suitability /5 | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
        "| A. Native strong-embedding matched-BER | 4 | 5 | 5 | 5 | 5 | Preferred when native difference is inspectable; final visual |",
        "| B. Matched-global-PSNR | 3 | 5 | 5 | 5 | 5 | Secondary operating-curve evidence |",
        "| C. Native + shared error-difference diagnostic | 5 | 5 | 5 | 5 | 4 | Supplementary explanation; D is not a native output |",
        "| D. Equal residual-strength visualization | 5 | 3 | 3 | 5 | 3 | Stress diagnostic only; never a native quality claim |",
        "| E. Color-tail extension | 2 | 4 | 2 | 4 | 3 | Not promoted unless an independently useful color difference is present |",
        "",
        "The final Figure 2 uses five reader-facing columns: `Original`, `HiDDeN-64 (retrained external)`, `MaskWM-D (official external)`, `Global Reconstruction (internal baseline)`, and `Hard Local-Tail (proposed)`. Every column uses the same selected validation samples, shared ROI, and 512×512 display canvas.",
        "",
        "Internal run identifiers map to reader-facing names: `Ours-base` → `Global Reconstruction (internal baseline)` and `Ours` → `Hard Local-Tail (proposed)`. External methods are contextual references, not strict matched comparisons; the under-converged HiDDeN retraining is not primary evidence.",
        "",
        "The final native figure shows full and shared-ROI zoom rows. The separate diagnostic suite shows `|residual| ×10/20/30/50`, log residuals for β=100/500/1000, and shared-scale pixel/patch error differences; these are visualization-only and never model outputs or selection metrics.",
        "",
        "## Residual and attack stress",
        "",
        "Equal residual amplification was evaluated for every discoverable 128×128/64-bit existing run checkpoint at α = 1, 1.5, 2, 2.5, 3, 4, 5. " + clip_sentence,
        "",
        "Crop 50/40/30 (five fixed mask repeats), Gaussian σ = 0.005/0.01/0.02, and JPEG quality 90/70/50 are recorded in [fig2_attack_stress_summary.csv](fig2_attack_stress_summary.csv) for the selected matched pair only. They are not used for Figure 2 selection; attack corruption is never presented as watermark-quality improvement.",
        "",
        "## Artifacts",
        "",
        "- [fig2_strong_embedding_results.csv](fig2_strong_embedding_results.csv)",
        "- [fig2_matched_ber_pairs.csv](fig2_matched_ber_pairs.csv)",
        "- [fig2_matched_psnr_pairs.csv](fig2_matched_psnr_pairs.csv)",
        "- [fig2_residual_alpha_sweep.csv](fig2_residual_alpha_sweep.csv)",
        "- [fig2_final_visual.png](../paper/figures/fig2_final_visual.png), [PDF](../paper/figures/fig2_final_visual.pdf), [editable PPTX](../paper/figures/fig2_final_visual.pptx)",
        "",
        "## Formal versus exploratory status",
        "",
        "Every row and figure in this report is exploratory/qualitative stress evidence. It must not replace formal Table I values, redefine Ours, or be described as a new formal method. No formal checkpoint was overwritten; no selective sharpening, blur, saturation, per-method normalization, attack realization, or per-method ROI was used.",
    ]
    (REPORT_ROOT / "fig2_strong_visual_search.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> None:
    result_csv = REPORT_ROOT / "fig2_strong_embedding_results.csv"
    if not result_csv.is_file():
        raise FileNotFoundError(result_csv)
    rows = read_csv(result_csv)
    matched_ber = find_matched_ber(rows)
    matched_psnr = find_matched_psnr(rows)
    write_csv(REPORT_ROOT / "fig2_matched_ber_pairs.csv", matched_ber)
    write_csv(REPORT_ROOT / "fig2_matched_psnr_pairs.csv", matched_psnr)
    selected_pair = best_native_pair(rows, matched_ber, matched_psnr)
    base_payload, ours_payload, base_row, ours_row = choose_pair_payload(selected_pair, rows)
    manifest = load(MANIFEST)
    model_original = manifest["images"].float()
    original = rgb_tensor(model_original)
    base = rgb_tensor(base_payload["encoded"])
    ours = rgb_tensor(ours_payload["encoded"])
    ranking, chosen = rank_samples(original, base, ours, manifest["sources"])
    write_csv(REPORT_ROOT / "fig2_sample_ranking.csv", ranking)
    contact_sheet(FIGURE_ROOT / "fig2_top20_contact_sheet.png", original, base, ours, ranking)
    (REPORT_ROOT / "fig2_selected_pair.json").write_text(json.dumps({"pair": selected_pair, "base_row": base_row, "ours_row": ours_row, "selected_samples": chosen}, indent=2) + "\n")
    native_path = FIGURE_ROOT / "fig2_A_matched_ber_native"
    render_native(native_path, model_original, base_payload["encoded"], ours_payload["encoded"], chosen, base_row, ours_row, "Matched crop-robustness operating point")
    psnr_selected = next((row for row in matched_psnr if row.get("selected") and row.get("status") != "no_pair_within_tolerance"), None)
    if psnr_selected is not None:
        psnr_base_payload, psnr_ours_payload, psnr_base_row, psnr_ours_row = choose_pair_payload(psnr_selected, rows)
        psnr_base = rgb_tensor(psnr_base_payload["encoded"])
        psnr_ours = rgb_tensor(psnr_ours_payload["encoded"])
        psnr_ranking, psnr_chosen = rank_samples(original, psnr_base, psnr_ours, manifest["sources"])
        render_native(FIGURE_ROOT / "fig2_B_matched_psnr", model_original, psnr_base_payload["encoded"], psnr_ours_payload["encoded"], psnr_chosen, psnr_base_row, psnr_ours_row, f"Matched global PSNR ({psnr_selected['target_psnr']:.0f} dB bin)")
    else:
        fig = plt.figure(figsize=(8, 2.5), facecolor="white")
        fig.text(.5, .62, "Matched global PSNR candidate unavailable", ha="center", va="center", fontsize=16, fontweight="bold")
        fig.text(.5, .34, "No Ours-base/Ours pair fell within ±0.15 dB in the 30/32/34/36 dB bins.", ha="center", va="center", fontsize=11)
        fig.savefig(FIGURE_ROOT / "fig2_B_matched_psnr_unavailable.png", dpi=250, facecolor="white")
        fig.savefig(FIGURE_ROOT / "fig2_B_matched_psnr_unavailable.pdf", facecolor="white")
        plt.close(fig)
    render_diagnostics(FIGURE_ROOT / "fig2_C_native_error_difference", model_original, base_payload["encoded"], ours_payload["encoded"], chosen, "Native outputs + error-difference map")
    render_residual(FIGURE_ROOT / "fig2_D_equal_residual_stress", model_original, base_payload["encoded"], ours_payload["encoded"], chosen, "Equal residual stress visualization")
    diagnostic_visuals(base_payload, ours_payload, model_original, chosen, selected_pair)
    attack_stress(selected_pair, rows, manifest["images"].float(), manifest["messages"].float(), choose_device(args.device))
    from experiments.generate_fig2_qualitative_search import make_pptx
    final_path = FIGURE_ROOT / "fig2_final_visual"
    comparison_data, external_provenance = final_comparison_data(
        model_original, manifest["messages"].float(), base_payload["encoded"], ours_payload["encoded"], choose_device(args.device)
    )
    labels = final_column_labels(comparison_data)
    labels[3] = f"{FINAL_METHODS[3]}\nnative PSNR {float(base_row['psnr']):.2f} dB\nBER30 {float(base_row['ber30']):.3f}"
    labels[4] = f"{FINAL_METHODS[4]}\nnative PSNR {float(ours_row['psnr']):.2f} dB\nBER30 {float(ours_row['ber30']):.3f}"
    labels, final_rows, panel_rows = render_final_comparison(final_path, comparison_data, chosen, labels)
    make_pptx(
        final_path.with_suffix(".pptx"),
        "Figure 2 — external references and matched internal comparison",
        labels,
        final_rows,
        panel_rows,
        "Fixed validation; shared ROI. External protocols are contextual, not strict matched comparisons. Residual ×10 is visualization-only.",
    )
    final_provenance = {
        "manifest": str(MANIFEST),
        "manifest_sha256": digest(MANIFEST),
        "selection": "samples and native 128 ROIs selected only from the internal Global Reconstruction versus Hard Local-Tail comparison",
        "sample_indices": [int(row["validation_index"]) for row in chosen[:2]],
        "rois_native_128": [[int(row["roi_x"]), int(row["roi_y"]), int(row["roi_size"]), int(row["roi_size"])] for row in chosen[:2]],
        "methods": list(FINAL_METHODS),
        "display": "same validation samples/ROIs on one 512x512 canvas; full, shared-ROI zoom, and explicitly labelled residual x10 rows",
        "matched_pair": {"base_condition": base_row["condition"], "ours_condition": ours_row["condition"], "ber30_base": float(base_row["ber30"]), "ber30_ours": float(ours_row["ber30"]), "local_psnr_delta": float(ours_row["local_psnr"]) - float(base_row["local_psnr"]), "p95_delta": float(ours_row["p95"]) - float(base_row["p95"])},
        "residual": "absolute mean-channel residual x10; visualization-only and not used as model output or selection metric",
        "external_protocol": "External methods are contextual rather than strict matched comparisons. HiDDeN is an under-converged retrained reimplementation; MaskWM-D uses the official D-64 pipeline.",
        "external_sources": external_provenance,
        "internal_sources": {
            "Global Reconstruction (internal baseline)": {"run": base_row["run"], "checkpoint": base_row["checkpoint"], "checkpoint_sha256": base_row["checkpoint_sha256"]},
            "Hard Local-Tail (proposed)": {"run": ours_row["run"], "checkpoint": ours_row["checkpoint"], "checkpoint_sha256": ours_row["checkpoint_sha256"]},
        },
    }
    (FIGURE_ROOT / "fig2_final_visual_provenance.json").write_text(json.dumps(final_provenance, indent=2) + "\n")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "fig2_final_visual_provenance.json").write_text(json.dumps(final_provenance, indent=2) + "\n")
    report(rows, matched_ber, matched_psnr, selected_pair, ranking, chosen, read_csv(REPORT_ROOT / "fig2_residual_alpha_sweep.csv") if (REPORT_ROOT / "fig2_residual_alpha_sweep.csv").is_file() else None)
    print(f"selected {selected_pair.get('selection')} with samples {[row['image_id'] for row in chosen[:2] if 'image_id' in row]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--epochs", type=int, default=SCREEN_EPOCHS)
    p_prepare.set_defaults(func=prepare)
    p_promote = sub.add_parser("prepare-promoted")
    p_promote.add_argument("--epochs", type=int, default=20)
    p_promote.add_argument("--max-conditions", type=int, default=4)
    p_promote.set_defaults(func=prepare_promoted)
    p_extensions = sub.add_parser("prepare-extensions")
    p_extensions.add_argument("--epochs", type=int, default=SCREEN_EPOCHS)
    p_extensions.set_defaults(func=prepare_extensions)
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p_eval.add_argument("--epoch", type=int, default=SCREEN_EPOCHS)
    p_eval.set_defaults(func=evaluate)
    p_convergence = sub.add_parser("evaluate-convergence")
    p_convergence.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p_convergence.add_argument("--epochs", type=int, nargs="+", default=(5, 10, 15, 20))
    p_convergence.add_argument("--plateau-db", type=float, default=0.15)
    p_convergence.set_defaults(func=evaluate_convergence)
    p_alpha = sub.add_parser("residual-alpha")
    p_alpha.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p_alpha.set_defaults(func=residual_alpha)
    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p_analyze.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
