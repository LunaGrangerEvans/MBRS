#!/usr/bin/env python3
"""Run the official MaskWM checkpoint on the MBRS fixed manifests.

This adapter preserves the official 64-bit model contract and never edits the
MaskWM checkout.  Fixed-manifest image metrics are computed after exporting a
128x128 RGB view so they can use the existing MBRS frozen evaluator.  Crop
decoding is explicitly a separate MaskWM-on-MBRS rectangle-canvas probe: it
uses the MBRS normalized zero-fill mask, not MaskWM's native CropResize layer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
MASKWM_REPO = MOUNT / "external_repos/maskwm"
CHECKPOINT_ROOT = MOUNT / "external_baselines/checkpoints/maskwm"
OUTPUT_ROOT = MOUNT / "external_baselines/outputs/maskwm"
MANIFESTS = {
    "test": MOUNT / "reports/uniform_eval_manifest.pt",
    "validation": MOUNT / "reports/content_selector/validation_manifest.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = torch.load(path, map_location="cpu", weights_only=False)
    if manifest["images"].shape != (50, 3, 128, 128):
        raise ValueError(f"unexpected image tensor shape: {manifest['images'].shape}")
    if manifest["messages"].shape != (50, 64):
        raise ValueError(f"unexpected message tensor shape: {manifest['messages'].shape}")
    return manifest


def load_model(model_name: str, checkpoint: Path, device: torch.device):
    # Import the official namespace before any project module named models.
    sys.path.insert(0, str(MASKWM_REPO))
    from models.Mask_Model import WatermarkModel

    config_path = MASKWM_REPO / f"configs/model/{model_name}.yaml"
    config = OmegaConf.load(config_path)
    model = WatermarkModel(**config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config


def to_native_512(images: torch.Tensor) -> torch.Tensor:
    return TF.resize(images, [512, 512], interpolation=InterpolationMode.BILINEAR, antialias=True)


def to_rgb8(image: torch.Tensor) -> np.ndarray:
    return (
        ((image.detach().cpu() + 1) / 2).clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .byte()
        .numpy()
    )


def save_rgb_batch(directory: Path, images: torch.Tensor, start: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for offset, image in enumerate(images):
        Image.fromarray(to_rgb8(image), mode="RGB").save(directory / f"image_{start + offset:03d}.png")


def load_saved_outputs(directory: Path, count: int) -> torch.Tensor:
    values = []
    for index in range(count):
        image = np.asarray(Image.open(directory / f"image_{index:03d}.png").convert("RGB"), dtype=np.float32)
        values.append(torch.from_numpy(image).permute(2, 0, 1) / 127.5 - 1)
    return torch.stack(values)


def expanded_masks(manifest: dict, extension: dict | None) -> dict[str, torch.Tensor]:
    source = dict(manifest["attack_masks"])
    if extension is not None:
        source["crop_40"] = extension["attack_masks"]["crop_40"]
    result = {}
    for name in ("crop_100", "crop_70", "crop_50", "crop_40", "crop_30"):
        nested = source[name]
        repeat_values = []
        for repeat in nested:
            values = []
            for batch_index, batch_mask in enumerate(repeat):
                start = batch_index * 16
                stop = min(start + 16, int(manifest["samples"]))
                values.extend([batch_mask[0]] * max(0, stop - start))
            if len(values) != int(manifest["samples"]):
                raise ValueError(f"expanded {name} repeat has {len(values)} samples")
            repeat_values.append(torch.stack(values).float())
        result[name] = torch.stack(repeat_values)
    return result


def bit_accuracy(target: torch.Tensor, decoded: torch.Tensor) -> tuple[int, int]:
    correct = int((target.gt(0.5) == decoded.gt(0.5)).sum().item())
    return correct, int(target.numel())


def run_model(
    model_name: str,
    manifest: dict,
    extension: dict | None,
    device: torch.device,
    batch_size: int,
    run_crops: bool,
    output_root: Path,
    manifest_path: Path,
) -> dict:
    checkpoint = CHECKPOINT_ROOT / f"{model_name}.pth"
    model, config = load_model(model_name, checkpoint, device)
    images = manifest["images"].float()
    messages = manifest["messages"].float()
    output_dir = output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    native_output_dir = output_dir / "native_512"
    native_output_dir.mkdir(parents=True, exist_ok=True)
    masks = expanded_masks(manifest, extension) if run_crops else {}
    crop_correct = {name: 0 for name in masks}
    crop_total = {name: 0 for name in masks}
    clean_correct = 0
    clean_total = 0

    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            stop = min(start + batch_size, len(images))
            image = to_native_512(images[start:stop].to(device))
            message = messages[start:stop].to(device)
            image_256 = F.interpolate(image, size=[256, 256], mode="bilinear", align_corners=False)
            is_ed = model_name.startswith("ED_")
            if is_ed:
                full_mask = torch.ones((stop - start, 1, 256, 256), device=device)
                wm_256 = model.encoder(
                    image_256, message, full_mask, use_jnd=True, jnd_factor=1.75, blue=True
                )
            else:
                wm_256 = model.encoder(image_256, message, use_jnd=True, jnd_factor=1.3, blue=True)
            wm_512 = (
                F.interpolate(wm_256 - image_256, size=[512, 512], mode="bilinear", align_corners=False)
                + image
            ).clamp(-1, 1)
            save_rgb_batch(native_output_dir, wm_512, start)
            wm_128 = TF.resize(
                wm_512, [128, 128], interpolation=InterpolationMode.BILINEAR, antialias=True
            ).cpu()
            save_rgb_batch(output_dir, wm_128, start)

            clean_256 = TF.resize(
                wm_512, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            decoded_clean, _ = model.decoder(clean_256)
            correct, total = bit_accuracy(message, decoded_clean)
            clean_correct += correct
            clean_total += total

            for crop_name, repeat_masks in masks.items():
                for repeat_masks_one in repeat_masks:
                    mask_512 = F.interpolate(
                        repeat_masks_one[start:stop].to(device), size=[512, 512], mode="nearest"
                    )
                    # Exact MBRS frozen crop semantics: retain a rectangle on
                    # the original canvas and zero the exterior in normalized
                    # tensor space.  This is deliberately not MaskWM's native
                    # fused-image (original-background) inference protocol.
                    fused = wm_512 * mask_512
                    fused_256 = TF.resize(
                        fused, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True
                    )
                    decoded, _ = model.decoder(fused_256)
                    correct, total = bit_accuracy(message, decoded)
                    crop_correct[crop_name] += correct
                    crop_total[crop_name] += total

    roundtrip_reference = TF.resize(
        to_native_512(images), [128, 128], interpolation=InterpolationMode.BILINEAR, antialias=True
    ).cpu()
    quality = evaluate_saved_quality(output_dir, roundtrip_reference, device, batch_size)
    row = {
        "method": model_name,
        "classification": "REFERENCE",
        "status": "completed_fixed_manifest_reference",
        "payload_bits": int(config.wm_enc_config.message_length),
        "native_model_resolution": 256,
        "official_default_canvas": 512,
        "fixed_eval_output_resolution": 128,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "manifest_sha256": sha256(manifest_path),
        "global_psnr": quality["global_psnr"],
        "global_ssim": quality["global_ssim"],
        "global_ms_ssim": quality["global_ms_ssim"],
        "full_lpips": quality["full_lpips"],
        "top25_local_psnr": quality["top25_local_psnr"],
        "patch_mse_p95": quality["patch_mse_p95"],
        "gini": quality["gini"],
        "clean_bit_accuracy": clean_correct / clean_total,
        "clean_ber": 1 - clean_correct / clean_total,
        "crop_protocol": "MaskWM-on-MBRS rectangle canvas: wm*mask with normalized zero outside; not native CropResize",
        "quality_reference_view": "official-style 128->512->128 roundtrip of each manifest host; avoids a resampling floor being attributed to watermark",
    }
    for crop_name in ("crop_100", "crop_70", "crop_50", "crop_40", "crop_30"):
        if crop_name in crop_total:
            accuracy = crop_correct[crop_name] / crop_total[crop_name]
            suffix = crop_name.removeprefix("crop_")
            row[f"bit_accuracy_{suffix}"] = accuracy
            row[f"ber_{suffix}"] = 1 - accuracy
        else:
            row[f"bit_accuracy_{crop_name.removeprefix('crop_')}"] = np.nan
            row[f"ber_{crop_name.removeprefix('crop_')}"] = np.nan
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "input_view": "manifest 128x128 [-1,1] -> official-style tensor resize to 512 -> model 256",
                "output_view": "official-style 512 canvas -> antialiased bilinear resize to 128 -> uint8 RGB PNG",
                "native_output_view": "official-style 512 canvas saved losslessly as uint8 RGB PNG under native_512/",
                "quality_reference_view": "same 128->512->128 roundtrip applied to host; frozen evaluator formulas retained",
                "message_contract": "fixed MBRS manifest 64-bit messages; native MaskWM 64-bit variant",
                "embedding": "D: jnd_factor=1.3, no encoder mask; ED: full encoder mask, jnd_factor=1.75",
                "crop_protocol": row["crop_protocol"],
            },
            indent=2,
        )
        + "\n"
    )
    return row


def evaluate_saved_quality(output_dir: Path, originals: torch.Tensor, device: torch.device, batch_size: int) -> dict:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.evaluate_extended_image_quality import evaluate

    encoded = load_saved_outputs(output_dir, len(originals))
    rows = evaluate(encoded, originals, device, batch_size)
    global_mse = float(np.mean([row["global_mse"] for row in rows]))
    return {
        "global_psnr": float(10 * np.log10(1 / max(global_mse, 1e-12))),
        "global_ssim": float(np.mean([row["global_ssim"] for row in rows])),
        "global_ms_ssim": float(np.mean([row["global_ms_ssim"] for row in rows])),
        "full_lpips": float(np.mean([row["full_lpips"] for row in rows])),
        "top25_local_psnr": float(np.mean([row["top25_local_psnr"] for row in rows])),
        "patch_mse_p95": float(np.mean([row["patch_mse_p95"] for row in rows])),
        "gini": float(np.mean([row["gini"] for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=("D_64bits", "ED_64bits", "all"), default="D_64bits")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--manifest", choices=tuple(MANIFESTS), default="test")
    parser.add_argument("--run-crops", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    manifest_path = MANIFESTS[args.manifest]
    manifest = load_manifest(manifest_path)
    extension_path = MOUNT / "reports/controlled_crop35_40_manifest.pt"
    extension = torch.load(extension_path, map_location="cpu", weights_only=False) if args.run_crops else None
    names = ("D_64bits", "ED_64bits") if args.model_name == "all" else (args.model_name,)
    rows = [
        run_model(
            name,
            manifest,
            extension,
            device,
            args.batch_size,
            args.run_crops,
            args.output_root,
            manifest_path,
        )
        for name in names
    ]
    if args.manifest == "test":
        output_csv = PROJECT_ROOT / "reports/maskwm_results.csv"
        fields = list(rows[0])
        with output_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(output_csv)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
