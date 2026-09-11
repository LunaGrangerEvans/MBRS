#!/usr/bin/env python3
"""Train and evaluate one explicitly labelled 64-bit HiDDeN reimplementation.

The upstream HiDDeN-PyTorch model files remain untouched.  This adapter owns
the requested DIV2K data split, fixed per-image messages, current-project
RandomCrop semantics, compatibility shims, training loop, and frozen metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from hidden_adapter import (
    MANIFEST_DEFAULT,
    bit_predictions,
    load_manifest,
    load_port_modules,
    sha256,
    to_rgb01,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
TRAIN_DIR = MOUNT / "datasets/train"
VAL_DIR = MOUNT / "datasets/validation"
TEST_DIR = MOUNT / "datasets/test"
DEFAULT_OUTPUT = MOUNT / "external_baselines/outputs/hidden_64bit_retrained"
DEFAULT_RESULTS = ROOT / "reports/hidden_64bit_results.csv"
SEED = 17
EVAL_AREAS = (100, 70, 50, 40, 30)


class FixedMessageImageDataset(Dataset):
    """DIV2K split with one deterministic 64-bit message per filename."""

    def __init__(self, directory: Path, train: bool, seed: int):
        self.directory = directory
        self.files = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() or path.is_symlink()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not self.files:
            raise FileNotFoundError(f"no images found in {directory}")
        size = 128
        # Decode and resize once. The original port decodes every image on
        # every epoch, which is needlessly slow for the local DIV2K PNG split.
        # RandomCrop remains per access and uses the same 140 -> 128 geometry.
        resize = transforms.Resize((int(size * 1.1), int(size * 1.1)))
        resized = []
        for path in self.files:
            with Image.open(path) as image:
                tensor = transforms.ToTensor()(resize(image.convert("RGB")))
            resized.append(tensor)
        self.images = torch.stack(resized).sub_(0.5).div_(0.5)
        self.train = train
        self.messages = torch.stack(
            [self._message_for(path.name, seed) for path in self.files]
        )

    @staticmethod
    def _message_for(name: str, seed: int) -> torch.Tensor:
        digest = hashlib.sha256(f"HiDDeN-64|{seed}|{name}".encode()).digest()
        bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))[:64]
        return torch.from_numpy(bits.astype(np.float32))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        image = self.images[index]
        if self.train:
            max_offset = image.shape[-1] - 128
            top = int(torch.randint(max_offset + 1, (1,)))
            left = int(torch.randint(max_offset + 1, (1,)))
            tensor = image[:, top : top + 128, left : left + 128]
        else:
            offset = (image.shape[-1] - 128) // 2
            tensor = image[:, offset : offset + 128, offset : offset + 128]
        return tensor.clone(), self.messages[index].clone()


class UnifiedRandomCropNoiser(nn.Module):
    """Always apply the MBRS current-project RandomCrop mask semantics."""

    def __init__(self, min_area: float = 0.3, max_area: float = 1.0):
        super().__init__()
        if not 0 < min_area <= max_area <= 1:
            raise ValueError("crop area must satisfy 0 < min_area <= max_area <= 1")
        self.min_area = min_area
        self.max_area = max_area

    def forward(self, noised_and_cover):
        encoded, cover = noised_and_cover
        _, _, height, width = encoded.shape
        min_side = self.min_area**0.5
        max_side = self.max_area**0.5
        height_ratio = np.random.uniform(min_side, max_side)
        width_ratio = np.random.uniform(min_side, max_side)
        kept_height = int(height_ratio * height)
        kept_width = int(width_ratio * width)
        if kept_height == height:
            h_start = 0
        else:
            h_start = np.random.randint(0, height - kept_height)
        if kept_width == width:
            w_start = 0
        else:
            w_start = np.random.randint(0, width - kept_width)
        mask = torch.zeros_like(encoded)
        mask[:, :, h_start : h_start + kept_height, w_start : w_start + kept_width] = 1
        return [encoded * mask, cover]


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--val-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument(
        "--attack-extension",
        type=Path,
        default=MOUNT / "reports/controlled_crop35_40_manifest.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def make_config(options_module):
    return options_module.HiDDenConfiguration(
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


def finite_losses(losses: dict[str, Any]) -> bool:
    return all(math.isfinite(float(value)) for value in losses.values())


def aggregate(losses: list[dict[str, Any]]) -> dict[str, float]:
    keys = losses[0].keys()
    return {key: float(np.mean([float(row[key]) for row in losses])) for key in keys}


def build_loaders(args):
    train_dataset = FixedMessageImageDataset(args.train_dir, train=True, seed=args.seed)
    val_dataset = FixedMessageImageDataset(args.val_dir, train=False, seed=args.seed)
    train_generator = torch.Generator().manual_seed(args.seed + 101)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=args.device == "cuda",
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device == "cuda",
    )
    return train_dataset, val_dataset, train_loader, val_loader


def train_model(args, model, train_loader, val_loader, device):
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        training_rows = []
        for images, messages in train_loader:
            images = images.to(device, non_blocking=True)
            messages = messages.to(device, non_blocking=True)
            losses, _ = model.train_on_batch([images, messages])
            if not finite_losses(losses):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}: {losses}")
            training_rows.append(losses)

        validation_rows = []
        for images, messages in val_loader:
            images = images.to(device, non_blocking=True)
            messages = messages.to(device, non_blocking=True)
            losses, _ = model.validate_on_batch([images, messages])
            if not finite_losses(losses):
                raise RuntimeError(f"non-finite validation loss at epoch {epoch}: {losses}")
            validation_rows.append(losses)

        train_mean = aggregate(training_rows)
        val_mean = aggregate(validation_rows)
        row = {
            "epoch": epoch,
            "train_loss": train_mean["loss           "],
            "train_encoder_mse": train_mean["encoder_mse    "],
            "train_decoder_mse": train_mean["dec_mse        "],
            "train_bit_error": train_mean["bitwise-error  "],
            "val_loss": val_mean["loss           "],
            "val_encoder_mse": val_mean["encoder_mse    "],
            "val_decoder_mse": val_mean["dec_mse        "],
            "val_bit_error": val_mean["bitwise-error  "],
            "val_bit_accuracy": 1.0 - val_mean["bitwise-error  "],
        }
        history.append(row)
        print(
            "epoch={epoch:03d} train_loss={train_loss:.5f} "
            "val_dec_mse={val_decoder_mse:.5f} val_bit_acc={val_bit_accuracy:.5f}".format(**row),
            flush=True,
        )

        # A conservative automatic stop for unmistakable collapse. Normal
        # early learning is allowed; only a sustained near-random plateau is
        # treated as non-convergence.
        if epoch >= 30:
            recent = history[-10:]
            if (
                max(item["val_bit_accuracy"] for item in recent) < 0.51
                and min(item["val_decoder_mse"] for item in recent)
                >= history[0]["val_decoder_mse"] * 0.98
            ):
                raise RuntimeError(
                    "training stopped as non-converged: validation bit accuracy "
                    "stayed near random and decoder MSE did not improve"
                )
    return history, time.perf_counter() - started


def save_training_artifacts(args, model, config, history, elapsed, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "hidden_64bit_epoch_200.pyt"
    torch.save(
        {
            "format": "hidden-reimplementation-64bit-retrained-v1",
            "epoch": args.epochs,
            "seed": args.seed,
            "config": vars(config),
            "model": model.encoder_decoder.state_dict(),
            "discriminator": model.discriminator.state_dict(),
            "optimizer_enc_dec": model.optimizer_enc_dec.state_dict(),
            "optimizer_discriminator": model.optimizer_discrim.state_dict(),
            "history": history,
        },
        checkpoint,
    )
    with (output_dir / "training_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_json(
        output_dir / "training_config.json",
        {
            "method": "HiDDeN reimplementation, 64-bit retrained",
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_dir": str(args.train_dir),
            "val_dir": str(args.val_dir),
            "train_images": 800,
            "validation_images": 50,
            "input": "128x128 RGB normalized to [-1,1]",
            "message": "fixed deterministic 64-bit message per filename",
            "noise": "always-on current-project RandomCrop(0.3,1.0) mask",
            "loss": "0.001 adversarial BCE + 0.7 encoder MSE + 1.0 decoder MSE",
            "local_tail": False,
            "oklab": False,
            "elapsed_seconds": elapsed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "upstream_port_commit": "556f76dd0602e4351ed4c19bd2ee87d50ef3c0de",
        },
    )
    return checkpoint


def load_eval_model(checkpoint: Path, options_module, noiser_module, device):
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    config = make_config(options_module)
    model = __import__("model.encoder_decoder", fromlist=["EncoderDecoder"]).EncoderDecoder(
        config, noiser_module.Noiser([], device)
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload, config


def encode_clean(model, images, messages, device, batch_size=16):
    result = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            result.append(
                model.encoder(images[start : start + batch_size].to(device), messages[start : start + batch_size].to(device)).cpu()
            )
    return torch.cat(result)


def patch_grid(tensor, patch_size=32):
    unfolded = F.unfold(tensor, kernel_size=patch_size, stride=patch_size)
    count = unfolded.shape[-1]
    return unfolded.transpose(1, 2).reshape(
        tensor.shape[0], count, tensor.shape[1], patch_size, patch_size
    )


def gini_per_image(values):
    ordered = np.sort(np.asarray(values, dtype=np.float64), axis=1)
    count = ordered.shape[1]
    total = ordered.sum(axis=1)
    return ((2 * np.arange(1, count + 1) - count - 1) * ordered).sum(axis=1) / (
        count * np.maximum(total, 1e-12)
    )


def evaluate_model(args, model, manifest, device, output_dir, checkpoint):
    images = manifest["images"].float()
    messages = manifest["messages"].float()
    if int(manifest["message_length"]) != 64:
        raise ValueError("the fixed manifest is not 64-bit")
    encoded = encode_clean(model, images, messages, device)
    cover_rgb = to_rgb01(images)
    encoded_rgb = to_rgb01(encoded)
    per_image_mse = (encoded_rgb - cover_rgb).square().mean(dim=(1, 2, 3))
    patches_encoded = patch_grid(encoded_rgb)
    patches_cover = patch_grid(cover_rgb)
    patch_mse = (patches_encoded - patches_cover).square().mean(dim=(2, 3, 4))
    top4_mse = torch.topk(patch_mse, 4, dim=1).values.mean(dim=1)

    from pytorch_msssim import ssim

    with torch.no_grad():
        ssim_values = []
        for start in range(0, len(images), 16):
            ssim_values.append(
                ssim(
                    encoded_rgb[start : start + 16].to(device),
                    cover_rgb[start : start + 16].to(device),
                    data_range=1.0,
                    size_average=False,
                    win_size=7,
                ).cpu()
            )
    ssim_values = torch.cat(ssim_values)

    import lpips

    lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    encoded_lpips = encoded_rgb * 2.0 - 1.0
    cover_lpips = cover_rgb * 2.0 - 1.0
    lpips_values = []
    with torch.no_grad():
        for start in range(0, len(images), 16):
            lpips_values.append(
                lpips_metric(
                    encoded_lpips[start : start + 16].to(device),
                    cover_lpips[start : start + 16].to(device),
                ).flatten().cpu()
            )
    lpips_values = torch.cat(lpips_values)

    crop_ber = {}
    with torch.no_grad():
        for area in EVAL_AREAS:
            mask_batches = manifest["attack_masks"][f"crop_{area}"]
            expected_batches = math.ceil(len(images) / 16)
            if len(mask_batches) != 5 or any(len(item) != expected_batches for item in mask_batches):
                raise ValueError(f"manifest mask layout incompatible for crop_{area}")
            errors = 0
            total = 0
            for repeat in range(5):
                for batch_index, start in enumerate(range(0, len(images), 16)):
                    stop = start + 16
                    mask = mask_batches[repeat][batch_index].to(device)
                    decoded = model.decoder(encoded[start:stop].to(device) * mask)
                    predicted = bit_predictions(decoded)
                    target = messages[start:stop].to(device)
                    errors += int((predicted != target).sum())
                    total += target.numel()
            crop_ber[f"ber{area}"] = errors / total

    patch_mse_np = patch_mse.numpy()
    per_image_rows = []
    per_image_psnr = -10.0 * torch.log10(per_image_mse.clamp_min(1e-12))
    per_image_local_psnr = -10.0 * torch.log10(top4_mse.clamp_min(1e-12))
    for index in range(len(images)):
        per_image_rows.append(
            {
                "image_index": index,
                "global_mse": float(per_image_mse[index]),
                "global_psnr": float(per_image_psnr[index]),
                "ssim": float(ssim_values[index]),
                "lpips": float(lpips_values[index]),
                "top25_local_psnr": float(per_image_local_psnr[index]),
                "p95_mse": float(np.percentile(patch_mse_np[index], 95)),
                "gini": float(gini_per_image(patch_mse_np[[index]])[0]),
            }
        )

    summary = {
        "method": "HiDDeN reimplementation, 64-bit retrained",
        "status": "completed",
        "seed": args.seed,
        "epoch": args.epochs,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "samples": len(images),
        "message_length": 64,
        "psnr": float(-10.0 * torch.log10(per_image_mse.mean().clamp_min(1e-12))),
        "ssim": float(ssim_values.mean()),
        "lpips": float(lpips_values.mean()),
        "top25_local_psnr": float(per_image_local_psnr.mean()),
        "p95_mse": float(np.mean([row["p95_mse"] for row in per_image_rows])),
        "gini": float(np.mean([row["gini"] for row in per_image_rows])),
        **crop_ber,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoded": encoded,
            "images": images,
            "messages": messages,
            "manifest_sha256": sha256(args.manifest),
            "attack_extension_sha256": sha256(args.attack_extension),
        },
        output_dir / "encoded_outputs.pt",
    )
    with (output_dir / "per_image_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)
    write_json(output_dir / "evaluation.json", summary)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", newline="") as stream:
        fields = [
            "method",
            "status",
            "seed",
            "epoch",
            "checkpoint",
            "checkpoint_sha256",
            "samples",
            "message_length",
            "psnr",
            "ssim",
            "lpips",
            "top25_local_psnr",
            "p95_mse",
            "gini",
            "ber100",
            "ber70",
            "ber50",
            "ber40",
            "ber30",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: summary.get(field, "") for field in fields})
    return summary


def load_eval_manifest(args):
    manifest = load_manifest(args.manifest)
    extension = torch.load(str(args.attack_extension), map_location="cpu", weights_only=True)
    if (
        extension["base_manifest_seed"] != manifest["seed"]
        or extension["samples"] != manifest["samples"]
        or extension["repeats"] != manifest["repeats"]
        or extension["batch_size"] != 16
    ):
        raise ValueError("crop35/40 manifest extension is incompatible with the fixed manifest")
    manifest["attack_masks"] = {
        **manifest["attack_masks"],
        **extension["attack_masks"],
    }
    return manifest


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.epochs != 200:
        raise ValueError("this controlled run is fixed to 200 epochs")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    set_seed(args.seed)
    options_module, _, _, noiser_module = load_port_modules()
    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint")
        eval_model, _, _ = load_eval_model(args.checkpoint, options_module, noiser_module, device)
        summary = evaluate_model(
            args,
            eval_model,
            load_eval_manifest(args),
            device,
            args.output_dir,
            args.checkpoint,
        )
        print(json.dumps(summary, indent=2))
        return 0
    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args)
    config = make_config(options_module)
    model_module = __import__("model.hidden", fromlist=["Hidden"])
    model = model_module.Hidden(config, device, UnifiedRandomCropNoiser(0.3, 1.0), None)
    # PyTorch 1.0 inferred floating labels from integer fill values. Modern
    # PyTorch infers Long here, which BCEWithLogitsLoss rejects. This is an
    # adapter-only compatibility assignment; upstream files are untouched.
    model.cover_label = 1.0
    model.encoded_label = 0.0
    print(
        f"training random-init HiDDeN-64 on {len(train_dataset)} train / "
        f"{len(val_dataset)} val images, device={device}",
        flush=True,
    )
    history, elapsed = train_model(args, model, train_loader, val_loader, device)
    checkpoint = save_training_artifacts(args, model, config, history, elapsed, args.output_dir)
    eval_model, _, _ = load_eval_model(checkpoint, options_module, noiser_module, device)
    summary = evaluate_model(args, eval_model, load_eval_manifest(args), device, args.output_dir, checkpoint)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
