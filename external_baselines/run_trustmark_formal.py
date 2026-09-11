#!/usr/bin/env python3
"""Run official TrustMark Q/P on the frozen MBRS formal manifest.

This script intentionally runs in the isolated TrustMark environment.  It does
not import MBRS models and does not train anything.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from trustmark import TrustMark


ROOT = Path("/root/workspace/GLX/icassp/MBRS")
DEFAULT_MANIFEST = Path("/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_manifest.pt")
DEFAULT_CROP_MANIFEST = Path("/mnt/wmcontent/GLX/icassp/MBRS/reports/controlled_crop35_40_manifest.pt")
DEFAULT_OUTPUT = Path("/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs")
AREAS = ("crop_30", "crop_40", "crop_50", "crop_70", "crop_100")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = ((tensor.detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1.0) * 127.5)
    return Image.fromarray(np.rint(array).clip(0, 255).astype(np.uint8), mode="RGB")


def masked_pil(image: Image.Image, mask: torch.Tensor) -> Image.Image:
    array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    masked = array * mask.numpy().astype(np.float32)[..., None]
    return Image.fromarray(np.rint(((masked + 1.0) * 127.5).clip(0, 255)).astype(np.uint8), mode="RGB")


def decode(tm, image: Image.Image, expected: str):
    secret, present, schema = tm.decode(image, MODE="binary", DETECTFIRST=False, ROTATION=False)
    return {
        "present": bool(present),
        "schema": int(schema),
        "exact": bool(present and secret == expected),
        "decoded_length": len(secret) if isinstance(secret, str) else 0,
    }


def run_mode(mode: str, manifest, crop_manifest, manifest_path: Path, out_root: Path, device: str):
    out_dir = out_root / f"trustmark_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = manifest["images"].float()
    messages = manifest["messages"].float()
    batch_size = int(manifest.get("batch_size", 16))
    tm = TrustMark(
        verbose=True,
        model_type=mode,
        encoding_type=TrustMark.Encoding.BCH_5,
        loadRemover=False,
        loadBBoxDetector=False,
        device=device,
    )
    capacity = int(tm.schemaCapacity())
    rows = []
    random.seed(170917)
    for index in range(len(images)):
        # TrustMark BCH-5 protects 61 data bits by default.  This is a
        # reference comparison, not the MBRS raw 64-bit payload.
        bits = messages[index].round().to(torch.int64).tolist()[:capacity]
        secret = "".join(str(int(bit)) for bit in bits)
        cover = tensor_to_pil(images[index])
        stego = tm.encode(cover, secret, MODE="binary")
        output_path = out_dir / f"image_{index:03d}.png"
        stego.save(output_path)
        clean = decode(tm, stego.convert("RGB"), secret)
        crop_results = {}
        for name in AREAS:
            repeat_results = []
            mask_source = crop_manifest if name not in manifest["attack_masks"] else manifest
            for repeat, batch_masks in enumerate(mask_source["attack_masks"][name]):
                mask = batch_masks[index // batch_size][0, 0]
                attacked = masked_pil(stego, mask)
                repeat_results.append(decode(tm, attacked, secret))
            crop_results[name] = repeat_results
        rows.append({
            "image_index": index,
            "output": str(output_path),
            "output_sha256": digest(output_path),
            "payload_bits": capacity,
            "clean": clean,
            "crop": crop_results,
        })
        if (index + 1) % 5 == 0:
            print(f"{mode}: {index + 1}/{len(images)}", flush=True)
    summary = {"mode": mode, "images": len(rows), "payload_bits": capacity}
    for key in ["clean", *AREAS]:
        values = [row["clean"] if key == "clean" else row["crop"][key] for row in rows]
        flat = values if key == "clean" else [item for group in values for item in group]
        summary[key] = {
            "detection_rate": float(np.mean([item["present"] for item in flat])),
            "exact_rate": float(np.mean([item["exact"] for item in flat])),
            "trials": len(flat),
        }
    provenance = {
        "repository": "https://github.com/adobe/trustmark",
        "manifest": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "model_type": mode,
        "device": device,
        "encoding": "BCH_5 with 100 internal bits, 61-bit data capacity",
        "input_transform": "fixed manifest tensor [-1,1] -> uint8 RGB PNG",
        "crop_transform": "same fixed manifest image-space zero mask, masked normalized image converted to uint8",
        "output_dir": str(out_dir),
    }
    (out_dir / "per_image.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--crop-manifest", type=Path, default=DEFAULT_CROP_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["Q", "P", "both"], default="both")
    args = parser.parse_args()
    manifest = torch.load(args.manifest, map_location="cpu", weights_only=False)
    crop_manifest = torch.load(args.crop_manifest, map_location="cpu", weights_only=False)
    if crop_manifest.get("base_manifest_seed") != manifest.get("seed"):
        raise ValueError("crop manifest seed does not match formal manifest")
    if crop_manifest.get("samples") != manifest.get("samples"):
        raise ValueError("crop manifest sample count does not match formal manifest")
    modes = ["Q", "P"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_mode(mode, manifest, crop_manifest, args.manifest, args.output_root, args.device)


if __name__ == "__main__":
    main()
