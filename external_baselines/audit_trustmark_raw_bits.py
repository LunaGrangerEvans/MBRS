#!/usr/bin/env python3
"""Audit TrustMark pre-ECC decoder bits using the official decoder path.

The official TrustMark ``subimage_decode`` thresholds
``tm.decoder.decoder(stego)`` before passing the 100-bit packet to BCH.  This
script makes that intermediate tensor explicit; it does not alter the model,
strength, preprocessing, or crop masks.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from trustmark import TrustMark


MANIFEST = Path("/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_manifest.pt")
CROP_MANIFEST = Path("/mnt/wmcontent/GLX/icassp/MBRS/reports/controlled_crop35_40_manifest.pt")
OUTPUT_ROOT = Path("/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs")
REPORT_ROOT = Path("/mnt/wmcontent/GLX/icassp/MBRS/reports")
AREAS = {"100%": "crop_100", "70%": "crop_70", "50%": "crop_50", "40%": "crop_40", "30%": "crop_30"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masked_pil(image, mask):
    array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    masked = array * mask.numpy().astype(np.float32)[..., None]
    return Image.fromarray(np.rint(((masked + 1.0) * 127.5).clip(0, 255)).astype(np.uint8), mode="RGB")


@torch.no_grad()
def official_raw_decoder(tm, image):
    # This is the exact preprocessing and decoder call in official
    # trustmark.py::subimage_decode, stopping before threshold/ECC.
    processed = tm.get_the_image_for_processing(image)
    resized = processed.resize((tm.model_resolution_dec, tm.model_resolution_dec), Image.BILINEAR)
    tensor = transforms.ToTensor()(resized).unsqueeze(0).to(tm.decoder.device) * 2.0 - 1.0
    logits = tm.decoder.decoder(tensor)
    bits = (logits > 0).to(torch.int64).cpu().numpy()[0]
    return logits.float().cpu().numpy()[0], bits


def run_mode(mode, manifest, crop_manifest, device):
    tm = TrustMark(verbose=True, model_type=mode, encoding_type=TrustMark.Encoding.BCH_5,
                   loadRemover=False, loadBBoxDetector=False, device=device)
    images = manifest["images"].float()
    messages = manifest["messages"].float()
    batch_size = int(manifest.get("batch_size", 16))
    capacity = int(tm.schemaCapacity())
    rows = []
    output_dir = OUTPUT_ROOT / f"trustmark_{mode}"
    for index in range(len(images)):
        output = Image.open(output_dir / f"image_{index:03d}.png").convert("RGB")
        secret = "".join(str(int(x)) for x in messages[index].round().to(torch.int64).tolist()[:capacity])
        expected_packet = tm.ecc.encode_binary([secret])[0].astype(np.int64)
        for ratio, mask_key in AREAS.items():
            source = manifest if mask_key in manifest["attack_masks"] else crop_manifest
            for repeat, batch_masks in enumerate(source["attack_masks"][mask_key]):
                attacked = masked_pil(output, batch_masks[index // batch_size][0, 0])
                logits, predicted = official_raw_decoder(tm, attacked)
                raw_errors = predicted != expected_packet
                decoded, detected, schema = tm.decode(attacked, MODE="binary", DETECTFIRST=False, ROTATION=False)
                exact = bool(detected and decoded == secret)
                rows.append({
                    "method": f"TrustMark {mode}", "image_index": index, "crop_ratio": ratio,
                    "repeat": repeat, "protected_bits": int(len(expected_packet)),
                    "raw_ber": float(raw_errors.mean()), "raw_bit_accuracy": float(1.0 - raw_errors.mean()),
                    "detection_success": int(bool(detected)), "exact_decode_success": int(exact),
                    "ecc_corrected_success": int(exact), "schema": int(schema),
                    "logit_mean": float(logits.mean()), "logit_abs_mean": float(np.abs(logits).mean()),
                })
        if (index + 1) % 5 == 0:
            print(f"{mode}: {index + 1}/50", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["Q", "P", "both"], default="both")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = torch.load(MANIFEST, map_location="cpu", weights_only=False)
    crop_manifest = torch.load(CROP_MANIFEST, map_location="cpu", weights_only=False)
    rows = []
    modes = ["Q", "P"] if args.mode == "both" else [args.mode]
    for mode in modes:
        rows.extend(run_mode(mode, manifest, crop_manifest, args.device))
    out_dir = REPORT_ROOT / "trustmark_raw_bits"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (out_dir / "per_trial.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in sorted({row["method"] for row in rows}):
        for ratio in AREAS:
            group = [row for row in rows if row["method"] == method and row["crop_ratio"] == ratio]
            summary.append({
                "method": method, "crop_ratio": ratio, "protected_bits": 100,
                "raw_ber": float(np.mean([row["raw_ber"] for row in group])),
                "raw_bit_accuracy": float(np.mean([row["raw_bit_accuracy"] for row in group])),
                "detection_success": float(np.mean([row["detection_success"] for row in group])),
                "exact_decode_success": float(np.mean([row["exact_decode_success"] for row in group])),
                "ecc_corrected_success": float(np.mean([row["ecc_corrected_success"] for row in group])),
                "trials": len(group),
            })
    with (out_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "provenance.json").write_text(json.dumps({
        "official_source": "trustmark.py::subimage_decode, stopping before threshold/ECC",
        "manifest": str(MANIFEST), "manifest_sha256": digest(MANIFEST),
        "crop_manifest": str(CROP_MANIFEST), "crop_protocol": "same fixed image-space masks as MBRS evaluator",
        "raw_definition": "threshold official decoder logits at >0 and compare 100 protected BCH packet bits before DataLayer.decode_bitstream",
        "probability_status": "official pipeline exposes logits; sigmoid probabilities are not used for BER",
        "no_training": True,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
