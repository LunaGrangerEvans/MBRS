#!/usr/bin/env python3
"""Small, source-preserving adapter for the public HiDDeN PyTorch port.

This module deliberately imports the vendored upstream port at runtime.  It
does not copy or modify the HiDDeN model implementation and never imports the
MBRS ``utils`` package, whose name would otherwise shadow the port's module.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
HIDDEN_PORT = REPO_ROOT / "external_baselines/repos/HiDDeN-pytorch"
MANIFEST_DEFAULT = Path("/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_manifest.pt")


def add_port_to_import_path() -> None:
    port = str(HIDDEN_PORT)
    if port in sys.path:
        sys.path.remove(port)
    sys.path.insert(0, port)


def load_port_modules():
    """Load the port modules with the port directory ahead of MBRS imports."""
    add_port_to_import_path()
    options = importlib.import_module("options")
    port_utils = importlib.import_module("utils")
    encoder_decoder = importlib.import_module("model.encoder_decoder")
    noiser = importlib.import_module("noise_layers.noiser")
    return options, port_utils, encoder_decoder, noiser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        manifest = torch.load(str(path), map_location="cpu")
    required = {"images", "messages", "message_length", "H", "W", "attack_masks"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"manifest is missing keys: {sorted(missing)}")
    if tuple(manifest["images"].shape[1:]) != (3, manifest["H"], manifest["W"]):
        raise ValueError("manifest image shape does not match H/W metadata")
    if manifest["messages"].shape[1] != manifest["message_length"]:
        raise ValueError("manifest message tensor does not match metadata")
    return manifest


def load_model(options_path: Path, checkpoint_path: Path, device: torch.device):
    options, port_utils, encoder_decoder_module, noiser_module = load_port_modules()
    _, config, noise_config = port_utils.load_options(str(options_path))
    model = encoder_decoder_module.EncoderDecoder(
        config, noiser_module.Noiser(noise_config, device)
    ).to(device)
    # The checkpoint is a dictionary of tensors and optimizer state.  Loading
    # it with weights_only=True avoids executing arbitrary pickle payloads.
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "enc-dec-model" not in checkpoint:
        raise ValueError("checkpoint does not have the HiDDeN port enc-dec-model key")
    model.load_state_dict(checkpoint["enc-dec-model"], strict=True)
    model.eval()
    return model, config, checkpoint, options


def encode(model, images: torch.Tensor, messages: torch.Tensor, device: torch.device) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start in range(0, len(images), 16):
            values.append(
                model.encoder(images[start : start + 16].to(device), messages[start : start + 16].to(device))
                .cpu()
            )
    return torch.cat(values)


def decode(model, encoded: torch.Tensor, device: torch.device) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start in range(0, len(encoded), 16):
            values.append(model.decoder(encoded[start : start + 16].to(device)).cpu())
    return torch.cat(values)


def to_rgb01(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def bit_predictions(decoded: torch.Tensor) -> torch.Tensor:
    # This is the port's documented convention: round then clip [0, 1].
    return decoded.round().clamp(0.0, 1.0)


def psnr_per_sample(encoded: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    mse = (to_rgb01(encoded) - to_rgb01(images)).square().mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def save_rgb(path: Path, tensor: torch.Tensor) -> None:
    array = (to_rgb01(tensor[0]).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def checkpoint_provenance(
    options_path: Path, checkpoint_path: Path, config: Any, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    return {
        "port_repo": str(HIDDEN_PORT),
        "port_repo_remote": "https://github.com/ando-khachatryan/HiDDeN",
        "options_file": str(options_path),
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_git_tracked": True,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "configuration": dict(vars(config)),
        "model_state_key_count": len(checkpoint["enc-dec-model"]),
        "decoder_message_projection_shape": list(
            checkpoint["enc-dec-model"]["decoder.linear.weight"].shape
        ),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2) + "\n")
