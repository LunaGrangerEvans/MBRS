#!/usr/bin/env python3
"""Run a one-batch backward + one-epoch TrustMark-Q adapter smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from external_baselines.trustmark_q_adapter import (  # noqa: E402
    TrustMarkQAdapter,
    as_device,
    make_optimizer,
    normalized_tensor_to_pil,
    pil_to_normalized_tensor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "external_baselines" / "repos" / "trustmark" / "images" / "ufo_240.jpg"
DEFAULT_CHECKPOINT = Path("/tmp/trustmark_q_local_tail_adapter_smoke.pth")


def valid_internal_messages(adapter: TrustMarkQAdapter) -> torch.Tensor:
    payloads = ["0" * 61, "01" * 30 + "0"]
    packets = [adapter.trustmark.ecc.encode_binary([payload])[0] for payload in payloads]
    return torch.from_numpy(np.stack(packets)).float().to(adapter.device)


def decode_result(adapter: TrustMarkQAdapter, stego: torch.Tensor, message: torch.Tensor):
    pil = normalized_tensor_to_pil(stego[0])
    decoded, present, schema = adapter.trustmark.decode(
        pil, MODE="binary", DETECTFIRST=False, ROTATION=False
    )
    expected_payload = "".join(str(int(bit)) for bit in message[0].round().cpu().tolist())
    expected = expected_payload[: adapter.trustmark.schemaCapacity()]
    return {
        "present": bool(present),
        "schema": int(schema),
        "exact_first_payload": bool(present and decoded == expected),
        "decoded_length": len(decoded) if isinstance(decoded, str) else 0,
    }


def scalar_objective(adapter, cover, message):
    adapter.eval_mode()
    with torch.no_grad():
        return float(adapter.objective(cover, message)["loss"].cpu())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--global-oklab-weight", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = as_device(args.device)
    torch.manual_seed(170917)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    adapter = TrustMarkQAdapter(
        device=str(device),
        local_loss_weight=0.5,
        global_oklab_weight=args.global_oklab_weight,
    )
    cover_one = pil_to_normalized_tensor(Image.open(args.image)).to(device)
    cover = torch.stack([cover_one, cover_one], dim=0)
    message = valid_internal_messages(adapter)
    optimizer = make_optimizer(adapter, lr=args.lr)

    # Explicit one-batch backward probe.
    adapter.train_mode()
    optimizer.zero_grad(set_to_none=True)
    with torch.enable_grad():
        backward_metrics = adapter.objective(cover, message)
        backward_loss = backward_metrics["loss"]
        backward_loss.backward()
        gradients = [
            parameter.grad
            for parameter in adapter.encoder.parameters()
            if parameter.requires_grad
        ]
        finite_gradient_count = sum(
            gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
            for gradient in gradients
        )
        if finite_gradient_count != len(gradients):
            raise RuntimeError("one-batch backward did not reach every encoder parameter")
        optimizer.step()
    post_backward_loss = scalar_objective(adapter, cover, message)

    # One epoch is deliberately one repeated batch: this is a smoke test, not
    # a dataset run.  It keeps runtime bounded and makes loss direction clear.
    epoch_pre_loss = scalar_objective(adapter, cover, message)
    epoch_step = adapter.train_step(optimizer, cover, message)
    epoch_post_loss = scalar_objective(adapter, cover, message)

    with torch.no_grad():
        adapter.eval_mode()
        final = adapter.objective(cover, message)
        decode = decode_result(adapter, final["stego"], message)

    adapter.save_checkpoint(args.checkpoint, optimizer, epoch=1, step=2)
    restored = TrustMarkQAdapter(
        device=str(device),
        local_loss_weight=0.5,
        global_oklab_weight=args.global_oklab_weight,
    )
    restored_optimizer = make_optimizer(restored, lr=args.lr)
    checkpoint = restored.load_adapter_checkpoint(args.checkpoint, restored_optimizer)
    with torch.no_grad():
        restored.eval_mode()
        restored_stego = restored.differentiable_encode(cover, message)
    restore_max_abs_diff = float((restored_stego - final["stego"]).abs().max().cpu())

    result = {
        "status": "pass",
        "device": str(device),
        "image": str(args.image),
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "encoder_trainable_parameters": sum(parameter.numel() for parameter in adapter.encoder.parameters()),
        "encoder_gradient_parameters": int(finite_gradient_count),
        "encoder_gradient_parameter_total": len(gradients),
        "patch_grid": "P16/stride8/Top10",
        "topk_count_at_256": int(backward_metrics["topk_count"].item()),
        "one_batch_backward_loss": float(backward_loss.detach().cpu()),
        "post_backward_loss": post_backward_loss,
        "epoch_pre_loss": epoch_pre_loss,
        "epoch_train_step_loss": epoch_step["loss"],
        "epoch_post_loss": epoch_post_loss,
        "loss_decreased_after_backward": post_backward_loss < float(backward_loss.detach().cpu()),
        "loss_decreased_during_epoch": epoch_post_loss < epoch_pre_loss,
        "decode": decode,
        "checkpoint_restore_format": checkpoint["format"],
        "checkpoint_restore_max_abs_stego_difference": restore_max_abs_diff,
        "checkpoint_restore_exact": restore_max_abs_diff == 0.0,
        "long_training_started": False,
        "objective_note": "RGB MSE + BCE logits base proxy; Hard local-tail RGB additive; optional global OKLab additive",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
