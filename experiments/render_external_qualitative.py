#!/usr/bin/env python3
"""Render the pre-registered six-column qualitative comparison.

The five validation samples and the shared ROI rule are inherited from the
existing MBRS qualitative protocol.  MaskWM is the official global D_64bits
variant; ED_64bits remains a separate adaptive/local reference in the audit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from matplotlib.patches import Rectangle
from skimage.color import deltaE_ciede2000, rgb2lab
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
VALIDATION_MANIFEST = MOUNT / "reports/content_selector/validation_manifest.pt"
MASKWM_OUTPUT = MOUNT / "external_baselines/outputs/maskwm_validation/D_64bits/native_512"
MASKWM_REPO = MOUNT / "external_repos/maskwm"
MASKWM_CHECKPOINT = MOUNT / "external_baselines/checkpoints/maskwm/D_64bits.pth"
HIDDEN_CHECKPOINT = MOUNT / "external_baselines/outputs/hidden_64bit_retrained/hidden_64bit_epoch_200.pyt"
HIDDEN_OPTIONS = PROJECT_ROOT / "external_baselines/repos/HiDDeN-pytorch/experiments/no-noise adam-eps-1e-4/options-and-config.pickle"
MBRS_DIR = MOUNT / "reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5"
OUT = MOUNT / "visualizations/external_qualitative"
SAMPLES = [0, 10, 20, 30, 40]
PREDECLARED_JND_GRID = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0, 1.10, 1.20, 1.30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb(tensor: torch.Tensor) -> np.ndarray:
    return ((tensor.detach().cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = max(float(np.square(reference - candidate).mean()), 1e-12)
    return float(10 * np.log10(1 / mse))


def load_pt_encoded(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False)["encoded"]


def load_maskwm() -> torch.Tensor:
    values = []
    for index in range(50):
        image = np.asarray(Image.open(MASKWM_OUTPUT / f"image_{index:03d}.png").convert("RGB"), dtype=np.float32)
        values.append(torch.from_numpy(image).permute(2, 0, 1) / 127.5 - 1)
    return torch.stack(values)


def upsample_for_display(tensor: torch.Tensor) -> list[np.ndarray]:
    values = TF.resize(tensor, [512, 512], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return [rgb(item) for item in values]


def load_hidden_validation(images: torch.Tensor, messages: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, dict]:
    # This duplicates only the small construction contract from the existing
    # 64-bit retraining adapter.  The upstream HiDDeN source is not modified.
    sys.path.insert(0, str(PROJECT_ROOT / "external_baselines/repos/HiDDeN-pytorch"))
    from options import HiDDenConfiguration
    from model.encoder_decoder import EncoderDecoder
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
    payload = torch.load(HIDDEN_CHECKPOINT, map_location="cpu", weights_only=True)
    model = EncoderDecoder(config, Noiser([], device)).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(images), 16):
            values.append(model.encoder(images[start:start + 16].to(device), messages[start:start + 16].to(device)).cpu())
    encoded = torch.cat(values)
    return encoded, {
        "checkpoint": str(HIDDEN_CHECKPOINT),
        "checkpoint_sha256": sha256(HIDDEN_CHECKPOINT),
        "contract": "HiDDeN reimplementation, 64-bit retrained; decoder/source message contract unchanged",
    }


def fixed_rois(originals: list[np.ndarray], incumbent: list[np.ndarray]) -> dict[int, tuple[int, int, int, int]]:
    rois = {}
    for index in SAMPLES:
        delta = deltaE_ciede2000(rgb2lab(originals[index]), rgb2lab(incumbent[index]))
        y, x = np.unravel_index(int(np.argmax(delta)), delta.shape)
        # The selection is made on the registered 128x128 view.  The common
        # 512x512 display canvas scales that native 32x32 ROI by four.
        rois[index] = (int((x // 32) * 128), int((y // 32) * 128), 128, 128)
    return rois


def maskwm_strength_sweep(images: torch.Tensor, messages: torch.Tensor, targets: list[float], device: torch.device) -> tuple[list[np.ndarray], list[dict]]:
    sys.path.insert(0, str(MASKWM_REPO))
    from models.Mask_Model import WatermarkModel

    config = OmegaConf.load(MASKWM_REPO / "configs/model/D_64bits.yaml")
    model = WatermarkModel(**config)
    state = torch.load(MASKWM_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    source = TF.resize(images.to(device), [512, 512], interpolation=InterpolationMode.BILINEAR, antialias=True)
    source_display = [rgb(item) for item in source.cpu()]
    image_256 = F.interpolate(source, size=[256, 256], mode="bilinear", align_corners=False)
    message = messages.to(device)
    candidates: dict[float, list[np.ndarray]] = {}
    scores: dict[float, list[float]] = {}
    with torch.inference_mode():
        for factor in PREDECLARED_JND_GRID:
            wm_256 = model.encoder(image_256, message, use_jnd=True, jnd_factor=factor, blue=True)
            wm_512 = (F.interpolate(wm_256 - image_256, size=[512, 512], mode="bilinear", align_corners=False) + source).clamp(-1, 1)
            candidates[factor] = [rgb(item) for item in wm_512.cpu()]
            scores[factor] = [psnr(source_display[index], candidates[factor][index]) for index in range(len(images))]
    selected = []
    rows = []
    for sample_position, index in enumerate(SAMPLES):
        factor = min(PREDECLARED_JND_GRID, key=lambda value: abs(scores[value][sample_position] - targets[sample_position]))
        selected.append(candidates[factor][sample_position])
        rows.append({
            "validation_index": index,
            "target_psnr_db": targets[sample_position],
            "jnd_factor_selected": factor,
            "maskwm_actual_psnr_db": scores[factor][sample_position],
        })
    return selected, rows


def render_grid(data: dict[str, list[np.ndarray]], rois: dict[int, tuple[int, int, int, int]], path: Path, zoom: bool = False) -> None:
    order = ["Original", "HiDDeN-64", "MaskWM-D_64", "MBRS Global", "Hard Top10", "Hard Top10 + OKLab"]
    fig, axes = plt.subplots(len(SAMPLES), len(order), figsize=(18, 15), squeeze=False)
    for row, index in enumerate(SAMPLES):
        x, y, width, height = rois[index]
        for col, method in enumerate(order):
            image = data[method][index]
            shown = image[y:y + height, x:x + width] if zoom else image
            axes[row, col].imshow(shown, interpolation="nearest")
            if row == 0:
                axes[row, col].set_title(method, fontsize=10)
            if not zoom:
                axes[row, col].add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="#e24a33", linewidth=1.2))
            axes[row, col].axis("off")
            if not zoom and method != "Original":
                axes[row, col].text(0.5, -0.03, f"{psnr(data['Original'][index], image):.2f} dB", transform=axes[row, col].transAxes, ha="center", va="top", fontsize=7)
        axes[row, 0].set_ylabel(f"sample {index:02d}", fontsize=9)
    title = "External qualitative comparison: fixed validation samples" if not zoom else "Shared fixed 32x32 zoom crops"
    fig.suptitle(title, fontsize=15)
    footer = "No residual amplification; same host/message per row. ROI fixed from incumbent Hard Top10 CIEDE2000 tail." if not zoom else "Nearest-neighbor display only; ROI selected once and shared across all methods."
    fig.text(0.5, 0.01, footer, ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def render_matched(data: dict[str, list[np.ndarray]], matched_maskwm: list[np.ndarray], path: Path) -> None:
    order = ["Original", "Hard Top10", "MaskWM-D_64 (matched)"]
    fig, axes = plt.subplots(len(SAMPLES), 3, figsize=(10, 15), squeeze=False)
    for row, index in enumerate(SAMPLES):
        reference = data["Original"][index]
        ours = data["Hard Top10"][index]
        matched = matched_maskwm[row]
        values = {"Original": reference, "Hard Top10": ours, "MaskWM-D_64 (matched)": matched}
        for col, method in enumerate(order):
            axes[row, col].imshow(values[method])
            axes[row, col].axis("off")
            score = "" if method == "Original" else f"\n{psnr(reference, values[method]):.2f} dB"
            axes[row, col].set_title(f"{method}{score}", fontsize=9)
    fig.suptitle("Matched-PSNR diagnostic: MaskWM-D_64 versus Hard Top10", fontsize=14)
    fig.text(0.5, 0.01, "Official D_64 JND-factor grid; no decoder or training change. Not the native/default figure.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = torch.load(VALIDATION_MANIFEST, map_location="cpu", weights_only=False)
    images, messages = manifest["images"].float(), manifest["messages"].float()
    originals_128 = [rgb(item) for item in images]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden, hidden_provenance = load_hidden_validation(images, messages, device)
    mbrs_hard_128 = [rgb(item) for item in load_pt_encoded(MBRS_DIR / "controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_outputs.pt")]
    data = {
        "Original": upsample_for_display(images),
        "HiDDeN-64": upsample_for_display(hidden),
        "MaskWM-D_64": [rgb(item) for item in load_maskwm()],
        "MBRS Global": upsample_for_display(load_pt_encoded(MBRS_DIR / "controlled_seed17_global_continuation_outputs.pt")),
        "Hard Top10": upsample_for_display(load_pt_encoded(MBRS_DIR / "controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_outputs.pt")),
        "Hard Top10 + OKLab": upsample_for_display(load_pt_encoded(MBRS_DIR / "seed17_crop_hard16_stride8_top10_global_oklab_g12p5_outputs.pt")),
    }
    rois = fixed_rois(originals_128, mbrs_hard_128)
    render_grid(data, rois, OUT / "qualitative_5x6_main.png")
    render_grid(data, rois, OUT / "qualitative_5x6_zoom4x.png", zoom=True)
    target_psnr = [psnr(data["Original"][index], data["Hard Top10"][index]) for index in SAMPLES]
    matched_maskwm, matched_rows = maskwm_strength_sweep(images[SAMPLES], messages[SAMPLES], target_psnr, device)
    render_matched(data, matched_maskwm, OUT / "matched_psnr_maskwm_vs_ours.png")
    with (OUT / "matched_psnr.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matched_rows[0]))
        writer.writeheader()
        writer.writerows(matched_rows)
    psnr_rows = []
    for index in SAMPLES:
        row = {"validation_index": index}
        for method, values in data.items():
            row[f"{method}_psnr_db"] = psnr(data["Original"][index], values[index])
        psnr_rows.append(row)
    with (OUT / "per_image_psnr.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(psnr_rows[0]))
        writer.writeheader()
        writer.writerows(psnr_rows)
    source_paths = {
        "validation_manifest": VALIDATION_MANIFEST,
        "maskwm_validation_outputs": MASKWM_OUTPUT,
        "mbrs_global": MBRS_DIR / "controlled_seed17_global_continuation_outputs.pt",
        "mbrs_hard_top10": MBRS_DIR / "controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_outputs.pt",
        "mbrs_hard_top10_oklab": MBRS_DIR / "seed17_crop_hard16_stride8_top10_global_oklab_g12p5_outputs.pt",
        "hidden_checkpoint": HIDDEN_CHECKPOINT,
    }
    (OUT / "provenance.json").write_text(json.dumps({
        "sample_indices": SAMPLES,
        "manifest": str(VALIDATION_MANIFEST),
        "manifest_sha256": sha256(VALIDATION_MANIFEST),
        "roi_rule": "native 32x32 block containing maximum CIEDE2000 pixel of incumbent Hard Top10; shared across methods",
        "display": "actual RGB PNG; no residual amplification; zoom uses nearest-neighbor",
        "maskwm_variant": "D_64bits, official global model; ED_64bits excluded from main qualitative figure",
        "matched_psnr": "official encoder JND factor sweep over predefined grid; no decoder/training change",
        "maskwm_provenance": {"path": str(MASKWM_OUTPUT), "files": 50, "sample_provenance": str(MASKWM_OUTPUT / "provenance.json")},
        "hidden_provenance": hidden_provenance,
        "source_sha256": {str(path): sha256(path) for path in source_paths.values() if path.is_file()},
    }, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# External qualitative comparison\n\n"
        "Five pre-registered validation indices: 0, 10, 20, 30, 40. Each row uses the same host image and fixed 64-bit message.\n\n"
        "- `qualitative_5x6_main.png`: Original, HiDDeN-64, MaskWM-D_64, MBRS Global, Hard Top10, Hard Top10 + OKLab.\n"
        "- `qualitative_5x6_zoom4x.png`: the same fixed 32x32 ROI for every method.\n"
        "- `matched_psnr_maskwm_vs_ours.png`: official D_64 JND-factor grid selected near the per-sample Hard Top10 PSNR.\n"
        "- `per_image_psnr.csv` and `matched_psnr.csv`: numeric provenance for the figure labels.\n"
    )
    print(OUT / "qualitative_5x6_main.png")
    print(OUT / "qualitative_5x6_zoom4x.png")
    print(OUT / "matched_psnr_maskwm_vs_ours.png")


if __name__ == "__main__":
    main()
