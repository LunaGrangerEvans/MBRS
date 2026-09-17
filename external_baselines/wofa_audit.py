#!/usr/bin/env python3
"""Audit the author-released WOFA inference bundle without modifying it.

The script deliberately stops short of claiming message accuracy: the current
author checkout contains embedder/extractor weights but no encoder/decoder
weights.  The optional component smoke only checks tensor I/O and finiteness
for the two released modules.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


ROOT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
DEFAULT_REPO = ROOT / "external_repos/wofa"
DEFAULT_OUTPUT = ROOT / "external_baselines/outputs/wofa_smoke"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(value: torch.Tensor) -> dict:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(value.min()),
        "max": float(value.max()),
        "finite": bool(torch.isfinite(value).all()),
    }


def load_official_module(path: Path, device: torch.device):
    # These are full serialized nn.Module objects from the pinned author
    # checkout.  weights_only=False is intentional and must not be generalized
    # to untrusted checkpoints.
    return torch.load(path, map_location=device, weights_only=False).eval().to(device)


def image_tensor(path: Path, device: torch.device, fit: bool = False) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if fit:
        image = ImageOps.fit(image, (200, 200))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def component_smoke(repo: Path, device: torch.device) -> dict:
    inference = repo / "inference"
    embedder = load_official_module(inference / "saved_models/embedder.pth", device)
    extractor = load_official_module(inference / "saved_models/extractor.pth", device)
    original = image_tensor(inference / "I_ori.png", device, fit=True)
    background = image_tensor(inference / "I_bg.png", device)
    mask_array = np.asarray(Image.open(inference / "I_mask.png").convert("L"), dtype=np.float32)
    mask = torch.from_numpy(mask_array / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    # This pattern is synthetic and intentionally not decoded as a message.
    pattern = torch.zeros((1, 1, 200, 200), device=device)
    with torch.inference_mode():
        residual = embedder((pattern, original))
        watermarked = (original + residual).clamp(0, 1)
        fused = mask * watermarked + (1 - mask) * background
        recovered_pattern = extractor(fused)

    return {
        "status": "component_smoke_pass_end_to_end_blocked",
        "device": str(device),
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "input_contract": {
            "image": "RGB 200x200 float [0,1]",
            "pattern": "synthetic zero 1x200x200; not a decoded message",
        },
        "embedder_output": stats(residual),
        "watermarked": stats(watermarked),
        "mask": stats(mask),
        "fused": stats(fused),
        "extractor_output": stats(recovered_pattern),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "component_smoke.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--skip-component", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    inference = repo / "inference"
    sys.path.insert(0, str(inference))
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    expected = [
        inference / "saved_models/embedder.pth",
        inference / "saved_models/extractor.pth",
        inference / "model.py",
        inference / "Stage1_Model.py",
        inference / "inf4all.py",
    ]
    required_missing = [
        inference / "models/encoder.pth",
        inference / "models/decoder.pth",
    ]
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    result = {
        "status": "end_to_end_blocked",
        "repo": str(repo),
        "repo_commit": commit,
        "expected_files": {
            str(path.relative_to(repo)): {
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None,
            }
            for path in expected
        },
        "missing_required_for_message": [
            str(path.relative_to(repo)) for path in required_missing if not path.is_file()
        ],
        "checkpoint_policy": "Only author-repository Git-tracked weights are loaded; no Baidu-share file is used.",
    }
    if not args.skip_component:
        result["component_smoke"] = component_smoke(repo, device)
        result["status"] = result["component_smoke"]["status"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
