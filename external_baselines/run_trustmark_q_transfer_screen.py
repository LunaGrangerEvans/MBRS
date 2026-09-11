#!/usr/bin/env python3
"""Run a fixed, three-arm TrustMark-Q transfer screen.

This is deliberately a bounded custom experiment.  Every arm starts from the
same official Q encoder/decoder weights, uses the same fixed train order,
messages, crop masks, optimizer and epoch count, and differs only in the
additional image loss term.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from skimage.color import deltaE_ciede2000, rgb2lab
from torch import Tensor
from torchvision import transforms

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from external_baselines.trustmark_q_adapter import (  # noqa: E402
    TrustMarkQAdapter,
    as_device,
    make_optimizer,
    normalized_tensor_to_pil,
)
from experiments.evaluate_extended_image_quality import evaluate  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
DEFAULT_MANIFEST = DEFAULT_ROOT / "reports/uniform_eval_manifest.pt"
DEFAULT_CROP_MANIFEST = DEFAULT_ROOT / "reports/controlled_crop35_40_manifest.pt"
DEFAULT_TRAIN_DIR = DEFAULT_ROOT / "datasets/train"
DEFAULT_CSV = PROJECT_ROOT / "reports/trustmark_q_transfer_screen.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports/trustmark_q_transfer_screen_summary.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports/trustmark_q_transfer_screen.md"
DEFAULT_PROVENANCE = PROJECT_ROOT / "reports/trustmark_q_transfer_screen_provenance.json"

SEED = 170917
TRAIN_CROP_MIN_AREA = 0.30
TRAIN_CROP_MAX_AREA = 1.00
CROP_KEYS = ("crop_100", "crop_70", "crop_50", "crop_40", "crop_30")
ARMS = (
    ("Q-control", 0.0, 0.0),
    ("Q-hard-local-tail", 0.5, 0.0),
    ("Q-hard-local-tail-oklab", 0.5, 0.05),
)


def load_torch(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_images(train_dir: Path, count: int, resolution: int = 128) -> Tensor:
    paths = sorted(train_dir.glob("*.png"))[:count]
    if len(paths) != count:
        raise RuntimeError(f"expected {count} train PNGs in {train_dir}, found {len(paths)}")
    values = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = ImageOps.fit(
            image,
            (resolution, resolution),
            method=Image.Resampling.BILINEAR,
            centering=(0.5, 0.5),
        )
        values.append(transforms.ToTensor()(image).mul(2.0).sub(1.0))
    return torch.stack(values)


def packet_messages(adapter: TrustMarkQAdapter, bits: Tensor) -> Tensor:
    packets = []
    for row in bits.round().to(torch.int64).tolist():
        payload = "".join(str(int(bit)) for bit in row[:61])
        packets.append(adapter.trustmark.ecc.encode_binary([payload])[0])
    return torch.from_numpy(np.stack(packets)).float()


def make_training_masks(
    count: int,
    epochs: int,
    height: int,
    width: int,
    seed: int,
) -> list[Tensor]:
    """Precompute the same RandomCrop(0.3, 1.0) mask sequence for all arms."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    min_side = TRAIN_CROP_MIN_AREA**0.5
    max_side = TRAIN_CROP_MAX_AREA**0.5
    masks = []
    for _epoch in range(epochs):
        epoch_masks = []
        for _sample in range(count):
            height_ratio = min_side + (max_side - min_side) * torch.rand((), generator=generator).item()
            width_ratio = min_side + (max_side - min_side) * torch.rand((), generator=generator).item()
            remaining_height = int(height_ratio * height)
            remaining_width = int(width_ratio * width)
            if remaining_height == height:
                height_start = 0
            else:
                height_start = int(torch.randint(0, height - remaining_height, (), generator=generator).item())
            if remaining_width == width:
                width_start = 0
            else:
                width_start = int(torch.randint(0, width - remaining_width, (), generator=generator).item())
            mask = torch.zeros(1, height, width)
            mask[:, height_start:height_start + remaining_height, width_start:width_start + remaining_width] = 1
            epoch_masks.append(mask)
        masks.append(torch.stack(epoch_masks))
    return masks


def make_orders(count: int, epochs: int, seed: int) -> list[Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randperm(count, generator=generator) for _ in range(epochs)]


def cpu_state(module: torch.nn.Module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def run_arm(
    name: str,
    local_weight: float,
    oklab_weight: float,
    source_encoder,
    source_decoder,
    train_images: Tensor,
    train_messages: Tensor,
    orders: list[Tensor],
    train_masks: list[Tensor],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
):
    seed_everything(SEED)
    adapter = TrustMarkQAdapter(
        device=str(device),
        local_loss_weight=local_weight,
        global_oklab_weight=oklab_weight,
    )
    adapter.encoder.load_state_dict(source_encoder)
    adapter.decoder.load_state_dict(source_decoder)
    optimizer = make_optimizer(adapter, lr=lr)
    losses = []
    started = time.perf_counter()
    step = 0
    for epoch in range(epochs):
        order = orders[epoch]
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            cover = train_images[indices].to(device, non_blocking=device.type == "cuda")
            message = train_messages[indices].to(device, non_blocking=device.type == "cuda")
            decoder_mask = train_masks[epoch][indices].to(device, non_blocking=device.type == "cuda")
            metrics = adapter.train_step(
                optimizer,
                cover,
                message,
                decoder_mask=decoder_mask,
            )
            losses.append(metrics)
            step += 1
        print(f"{name}: epoch {epoch + 1}/{epochs}, loss={losses[-1]['loss']:.6f}", flush=True)
    elapsed = time.perf_counter() - started
    return adapter, {
        "method": name,
        "train_epochs": epochs,
        "train_images": len(train_images),
        "batch_size": batch_size,
        "train_steps": step,
        "lr": lr,
        "train_first_loss": losses[0]["loss"],
        "train_last_loss": losses[-1]["loss"],
        "train_elapsed_s": elapsed,
    }


def masked_pil(image: Image.Image, mask: Tensor) -> Image.Image:
    array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    masked = array * mask.cpu().numpy().astype(np.float32)[..., None]
    output = np.rint(((masked + 1.0) * 127.5).clip(0, 255)).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


def official_decoder_input(adapter: TrustMarkQAdapter, image: Image.Image) -> Tensor:
    processed = adapter.trustmark.get_the_image_for_processing(image)
    resolution = int(adapter.trustmark.model_resolution_dec)
    resized = processed.resize((resolution, resolution), Image.Resampling.BILINEAR)
    return transforms.ToTensor()(resized).unsqueeze(0).mul(2.0).sub(1.0)[0]


def quality_color_metrics(outputs: Tensor, references: Tensor, device: torch.device):
    rows = evaluate(outputs.cpu(), references.cpu(), device, 16)
    color_rows = []
    output_rgb = ((outputs + 1.0) / 2.0).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    reference_rgb = ((references + 1.0) / 2.0).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    for encoded, reference in zip(output_rgb, reference_rgb):
        distance = deltaE_ciede2000(rgb2lab(encoded), rgb2lab(reference))
        color_rows.append({
            "ciede2000_global": float(distance.mean()),
            "ciede2000_p95": float(np.percentile(distance, 95)),
            "ciede2000_max": float(distance.max()),
        })
    return [{**quality, **color} for quality, color in zip(rows, color_rows)]


def raw_crop_trials(
    adapter: TrustMarkQAdapter,
    outputs: Tensor,
    expected_packets: np.ndarray,
    manifest: dict,
    crop_manifest: dict,
    device: torch.device,
    batch_size: int = 64,
):
    clean_pils = [normalized_tensor_to_pil(image) for image in outputs]
    specs = []
    manifest_batch_size = int(manifest.get("batch_size", 16))
    for ratio in CROP_KEYS:
        mask_manifest = manifest if ratio in manifest["attack_masks"] else crop_manifest
        for repeat, batch_masks in enumerate(mask_manifest["attack_masks"][ratio]):
            for index, clean in enumerate(clean_pils):
                mask = batch_masks[index // manifest_batch_size][0, 0]
                attacked = clean if ratio == "crop_100" and bool(mask.all()) else masked_pil(clean, mask)
                specs.append((index, ratio, repeat, attacked))

    decoded_bits = []
    for start in range(0, len(specs), batch_size):
        input_batch = torch.stack([
            official_decoder_input(adapter, item[3]) for item in specs[start:start + batch_size]
        ]).to(device)
        with torch.no_grad():
            logits = adapter.decoder.decoder(input_batch)
        decoded_bits.append((logits > 0).cpu().numpy().astype(np.int64))
    decoded_bits = np.concatenate(decoded_bits, axis=0)

    rows = []
    for spec, predicted in zip(specs, decoded_bits):
        index, ratio, repeat, _image = spec
        expected = expected_packets[index]
        raw_ber = float(np.not_equal(predicted, expected).mean())
        decoded, detected, schema = adapter.trustmark.ecc.decode_bitstream(
            predicted[None, :], MODE="binary"
        )[0]
        payload = "".join(str(int(bit)) for bit in expected[:61])
        rows.append({
            "image_index": index,
            "crop_ratio": ratio.removeprefix("crop_"),
            "repeat": repeat,
            "raw_bit_accuracy": 1.0 - raw_ber,
            "raw_ber": raw_ber,
            "detection_success": int(bool(detected)),
            "exact_decode_success": int(bool(detected and decoded == payload)),
            "schema": int(schema),
        })
    return rows


def aggregate_rows(method, quality_rows, raw_rows, train_info):
    result = {**train_info, "method": method}
    for key in (
        "global_psnr", "global_ssim", "full_lpips", "top25_local_psnr",
        "patch_mse_p95", "gini", "ciede2000_global", "ciede2000_p95", "ciede2000_max",
    ):
        result[key] = float(np.mean([row[key] for row in quality_rows]))
    for ratio in CROP_KEYS:
        label = ratio.removeprefix("crop_")
        group = [row for row in raw_rows if row["crop_ratio"] == label]
        for key in ("raw_bit_accuracy", "raw_ber", "detection_success", "exact_decode_success"):
            result[f"{key}_{label}"] = float(np.mean([row[key] for row in group]))
    return result


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary_rows, provenance):
    by_method = {row["method"]: row for row in summary_rows}
    control = by_method["Q-control"]
    hard = by_method["Q-hard-local-tail"]
    oklab = by_method["Q-hard-local-tail-oklab"]

    def delta(method, key):
        return by_method[method][key] - control[key]

    hard_local_improved = (
        delta("Q-hard-local-tail", "top25_local_psnr") > 0
        and delta("Q-hard-local-tail", "patch_mse_p95") < 0
        and delta("Q-hard-local-tail", "gini") < 0
    )
    oklab_color_improved = (
        oklab["ciede2000_global"] < hard["ciede2000_global"]
        and oklab["ciede2000_p95"] < hard["ciede2000_p95"]
    )
    max_hard_ber_delta = max(
        abs(hard[f"raw_ber_{ratio}"] - control[f"raw_ber_{ratio}"])
        for ratio in ("100", "70", "50", "40", "30")
    )
    max_oklab_ber_delta = max(
        abs(oklab[f"raw_ber_{ratio}"] - control[f"raw_ber_{ratio}"])
        for ratio in ("100", "70", "50", "40", "30")
    )
    direction_stable = hard_local_improved and oklab_color_improved and max_oklab_ber_delta <= 0.005

    lines = [
        "# TrustMark-Q bounded transfer screen",
        "",
        "**Status: completed bounded screen; no long training started.**",
        "",
        "This is an **official TrustMark-Q weights initialized custom transfer** experiment. It is not official TrustMark fine-tuning: the public training loss/data/noise pipeline is incomplete, as documented in [the prior audit](strong_backbone_transfer_audit.md).",
        "",
        "## Fixed protocol",
        "",
        f"- Seed: `{provenance['seed']}`; train data: first `{provenance['train_images']}` lexicographically sorted DIV2K train PNGs; evaluation: the fixed 50-image manifest.",
        f"- Budget: `{provenance['epochs']}` epochs, batch `{provenance['batch_size']}`, Adam encoder-only, LR `{provenance['lr']}`; all three arms use the same order and masks.",
        f"- Training crop: same deterministic `RandomCrop({TRAIN_CROP_MIN_AREA}, 1.0)` mask sequence for every arm; no JPEG or extra augmentation.",
        "- Decoder: fixed Q ResNet-50, `eval()` mode, no decoder gradients or BatchNorm updates.",
        "- Arm differences only: Q-control has base proxy; Hard adds `0.5 × Hard local-tail RGB (P16/stride8/Top10)`; OKLab adds fixed `0.05 × global OKLab`.",
        "- Evaluation crop: fixed `100/70/50/40/30%` masks, five repeats per ratio; raw bits are the official decoder logits thresholded before BCH/ECC.",
        "",
        "## Clean image quality and color",
        "",
        "| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Top25 local PSNR ↑ | P95 MSE ↓ | Gini ↓ | CIEDE2000 mean ↓ | CIEDE2000 P95 ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['global_psnr']:.6f} | {row['global_ssim']:.6f} | {row['full_lpips']:.8f} | {row['top25_local_psnr']:.6f} | {row['patch_mse_p95']:.9f} | {row['gini']:.6f} | {row['ciede2000_global']:.6f} | {row['ciede2000_p95']:.6f} |"
        )
    lines += [
        "",
        "## Crop raw-bit accuracy / BER",
        "",
        "| Method | 100% acc / BER | 70% acc / BER | 50% acc / BER | 40% acc / BER | 30% acc / BER |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        values = []
        for ratio in ("100", "70", "50", "40", "30"):
            values.append(f"{row[f'raw_bit_accuracy_{ratio}']:.6f} / {row[f'raw_ber_{ratio}']:.6f}")
        lines.append(f"| {row['method']} | " + " | ".join(values) + " |")

    lines += [
        "",
        "## Direction readout",
        "",
        f"- Hard local-tail local-distortion direction: **{'positive' if hard_local_improved else 'not confirmed'}**. Compared with Q-control, ΔTop25 local PSNR = `{delta('Q-hard-local-tail', 'top25_local_psnr'):+.6f}`, ΔP95 MSE = `{delta('Q-hard-local-tail', 'patch_mse_p95'):+.9f}`, ΔGini = `{delta('Q-hard-local-tail', 'gini'):+.6f}`.",
        f"- OKLab color direction: **{'positive' if oklab_color_improved else 'not confirmed'}** versus Hard. ΔCIEDE2000 mean versus Hard = `{oklab['ciede2000_global'] - hard['ciede2000_global']:+.6f}`; ΔCIEDE2000 P95 = `{oklab['ciede2000_p95'] - hard['ciede2000_p95']:+.6f}`.",
        f"- Robustness: maximum absolute BER change versus Q-control is `{max_hard_ber_delta:.6f}` for Hard and `{max_oklab_ber_delta:.6f}` for Hard+OKLab. This is a descriptive short-screen result, not a significance test.",
        f"- Overall bounded-screen direction: **{'stable enough for a carefully controlled next screen' if direction_stable else 'not stable enough to justify long training'}**. No parameter grid or retry was run.",
        "",
        "## Interpretation and next decision",
        "",
        "The screen answers only whether the already-selected loss direction survives a stronger fixed Q backbone under the same custom training bridge. It does not establish official TrustMark training fidelity or a general quality/crop ranking against MBRS, because TrustMark uses a 100-bit BCH-protected message while MBRS uses raw 64-bit BER.",
        "",
        "Long training should be considered only if Hard improves the local-tail metrics without a meaningful BER increase and the OKLab arm lowers CIEDE2000 without reversing the quality/robustness direction. If either condition fails here, retain the Q-control/custom objective and do not escalate the backbone experiment.",
        "",
        "## Artifacts",
        "",
        "- [Complete per-trial CSV](trustmark_q_transfer_screen.csv)",
        "- [Per-arm summary CSV](trustmark_q_transfer_screen_summary.csv)",
        "- [Provenance JSON](trustmark_q_transfer_screen_provenance.json)",
        "- [Runner](../external_baselines/run_trustmark_q_transfer_screen.py)",
        "",
        "## Limits",
        "",
        "- Two epochs on a 64-image training subset is a bounded screen, not a convergence study.",
        "- Base image/message terms are the transparent RGB-MSE + BCE-logit proxy from the adapter; the missing official `ImageSecretLoss` is not fabricated.",
        "- CIEDE2000 is evaluation-only (CIELAB D65, skimage); OKLab is the only optional color term optimized.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--crop-manifest", type=Path, default=DEFAULT_CROP_MANIFEST)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--train-images", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-provenance", type=Path, default=DEFAULT_PROVENANCE)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.train_images <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("train-images, epochs and batch-size must be positive")
    for manifest_path in (args.manifest, args.crop_manifest):
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
    if not args.train_dir.is_dir():
        raise FileNotFoundError(args.train_dir)

    seed_everything(SEED)
    device = as_device(args.device)
    manifest = load_torch(args.manifest)
    crop_manifest = load_torch(args.crop_manifest)
    eval_images = manifest["images"].float()
    eval_bits = manifest["messages"].float()
    if tuple(eval_images.shape[-2:]) != (128, 128):
        raise ValueError(f"expected 128x128 eval manifest, got {tuple(eval_images.shape[-2:])}")
    if crop_manifest.get("samples") != manifest.get("samples"):
        raise ValueError("crop manifest sample count does not match evaluation manifest")
    if crop_manifest.get("base_manifest_seed") != manifest.get("seed"):
        raise ValueError("crop manifest seed does not match evaluation manifest")

    # One official Q construction supplies the exact source states for every arm.
    source = TrustMarkQAdapter(device=str(device), local_loss_weight=0.0, global_oklab_weight=0.0)
    source_encoder = cpu_state(source.encoder)
    source_decoder = cpu_state(source.decoder)
    eval_packets = packet_messages(source, eval_bits).numpy().astype(np.int64)
    train_images = load_train_images(args.train_dir, args.train_images)
    train_rng = np.random.default_rng(SEED)
    train_bits = torch.from_numpy(train_rng.integers(0, 2, size=(args.train_images, 64))).float()
    train_messages = packet_messages(source, train_bits)
    orders = make_orders(args.train_images, args.epochs, SEED + 1)
    train_masks = make_training_masks(args.train_images, args.epochs, 128, 128, SEED + 2)

    all_rows = []
    summary_rows = []
    provenance = {
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_images": args.train_images,
        "train_dir": str(args.train_dir),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "crop_manifest": str(args.crop_manifest),
        "crop_manifest_sha256": sha256(args.crop_manifest),
        "official_q_encoder_md5": "700328b8754db934b2f6cb5e5185d81f",
        "official_q_decoder_md5": "4ced90e9cfe13e3295ad082887fe9187",
        "device": str(device),
        "training_crop": "RandomCrop(0.3,1.0), deterministic precomputed mask sequence",
        "evaluation_crop": "fixed manifest masks crop_100/crop_70/crop_50/crop_40/crop_30, 5 repeats",
        "arms": [
            {"method": name, "local_loss_weight": local, "global_oklab_weight": oklab}
            for name, local, oklab in ARMS
        ],
        "official_fine_tuning": False,
        "custom_transfer": True,
        "long_training_started": False,
    }

    for name, local_weight, oklab_weight in ARMS:
        adapter, train_info = run_arm(
            name,
            local_weight,
            oklab_weight,
            source_encoder,
            source_decoder,
            train_images,
            train_messages,
            orders,
            train_masks,
            device,
            args.epochs,
            args.batch_size,
            args.lr,
        )
        adapter.eval_mode()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(eval_images), args.batch_size):
                cover = eval_images[start:start + args.batch_size].to(device)
                message = eval_packets[start:start + args.batch_size]
                message = torch.from_numpy(message).float().to(device)
                outputs.append(adapter.differentiable_encode(cover, message).cpu())
        outputs = torch.cat(outputs)
        quality_rows = quality_color_metrics(outputs, eval_images, device)
        raw_rows = raw_crop_trials(adapter, outputs, eval_packets, manifest, crop_manifest, device)
        summary = aggregate_rows(name, quality_rows, raw_rows, train_info)
        summary_rows.append(summary)
        for raw in raw_rows:
            quality = quality_rows[raw["image_index"]]
            all_rows.append({
                "method": name,
                **quality,
                **raw,
                "train_epochs": args.epochs,
                "train_images": args.train_images,
                "train_batch_size": args.batch_size,
                "train_steps": train_info["train_steps"],
                "train_last_loss": train_info["train_last_loss"],
            })
        provenance.setdefault("arms_runtime", []).append(summary)
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(args.output_csv, all_rows)
    write_csv(args.output_summary, summary_rows)
    args.output_provenance.parent.mkdir(parents=True, exist_ok=True)
    args.output_provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    write_report(args.output_report, summary_rows, provenance)
    print(json.dumps({"summary": summary_rows, "csv": str(args.output_csv), "report": str(args.output_report)}, indent=2))


if __name__ == "__main__":
    main()
