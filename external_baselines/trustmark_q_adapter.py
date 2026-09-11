"""Minimal differentiable adapter for research-only TrustMark-Q transfer.

The public TrustMark package exposes split encoder/decoder checkpoints and a
forward pass, but not the original training loss/data runner.  This module
therefore keeps the published modules and message dimensionality intact while
making the smallest explicit objective needed to test local-tail transfer.

It is intentionally not presented as a reimplementation of TrustMark's
private/original training pipeline.  The base image term is RGB MSE and the
base message term is BCE on the published decoder logits; the requested Hard
Patch16/stride8/Top10 term and optional global OKLab term are additive.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRUSTMARK_PYTHON = PROJECT_ROOT / "external_baselines" / "repos" / "trustmark" / "python"
if str(TRUSTMARK_PYTHON) not in sys.path:
    sys.path.insert(0, str(TRUSTMARK_PYTHON))

from trustmark import TrustMark  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.oklab_global import global_oklab_components  # noqa: E402


def load_checkpoint(path: Path, device: torch.device):
    """Load a locally produced adapter checkpoint across torch versions."""
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def hard_local_tail_rgb_loss(
    stego: Tensor,
    cover: Tensor,
    patch_size: int = 16,
    patch_stride: int = 8,
    topk_ratio: float = 0.10,
) -> Tuple[Tensor, int, Tensor]:
    """Return Hard local-tail RGB MSE using P16 / stride8 / Top10.

    The score is the mean RGB MSE in each overlapping patch.  Patch indices
    are selected from detached scores by ``topk``; gradients flow through the
    selected RGB MSE values, matching the existing MBRS hard-tail objective.
    """
    if stego.shape != cover.shape:
        raise ValueError("stego and cover must have the same shape")
    if stego.ndim != 4 or stego.shape[1] != 3:
        raise ValueError("expected NCHW RGB tensors")
    if patch_size <= 0 or patch_stride <= 0:
        raise ValueError("patch_size and patch_stride must be positive")
    if not 0 < topk_ratio <= 1:
        raise ValueError("topk_ratio must be in (0, 1]")
    if stego.shape[-2] < patch_size or stego.shape[-1] < patch_size:
        raise ValueError("patch_size must fit inside the image")

    per_pixel_rgb_mse = (stego - cover).square().mean(dim=1, keepdim=True)
    patch_scores = F.avg_pool2d(
        per_pixel_rgb_mse,
        kernel_size=patch_size,
        stride=patch_stride,
    ).flatten(1)
    patch_count = patch_scores.shape[1]
    topk_count = max(1, int(math.ceil(patch_count * topk_ratio)))
    selected = torch.topk(
        patch_scores, topk_count, dim=1, largest=True, sorted=False
    ).values
    return selected.mean(), topk_count, patch_scores


def pil_to_normalized_tensor(image: Image.Image, resolution: int = 256) -> Tensor:
    """Convert an RGB PIL image to the TrustMark encoder's [-1, 1] tensor."""
    image = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BILINEAR)
    return transforms.ToTensor()(image).mul(2.0).sub(1.0)


def normalized_tensor_to_pil(image: Tensor) -> Image.Image:
    """Quantize one [-1, 1] CHW tensor as the official PIL decode path does."""
    array = (
        image.detach().cpu().clamp(-1.0, 1.0).permute(1, 2, 0).numpy() + 1.0
    ) * 127.5
    return Image.fromarray(array.round().clip(0, 255).astype("uint8"), mode="RGB")


class TrustMarkQAdapter(nn.Module):
    """Keep TrustMark-Q's published encoder/decoder and add MBRS losses."""

    FORMAT = "trustmark-q-local-tail-adapter-v1"

    def __init__(
        self,
        device: str = "cpu",
        model_type: str = "Q",
        base_image_weight: float = 1.5,
        base_message_weight: float = 20.0,
        local_loss_weight: float = 0.5,
        global_oklab_weight: float = 0.0,
    ) -> None:
        if model_type != "Q":
            raise ValueError("this minimal adapter is intentionally scoped to TrustMark-Q")
        super().__init__()
        self.device = torch.device(device)
        self.base_image_weight = float(base_image_weight)
        self.base_message_weight = float(base_message_weight)
        self.local_loss_weight = float(local_loss_weight)
        self.global_oklab_weight = float(global_oklab_weight)

        tm = TrustMark(
            verbose=False,
            model_type=model_type,
            loadRemover=False,
            loadBBoxDetector=False,
            device=str(self.device),
        )
        self.trustmark = tm
        self.encoder = tm.encoder
        self.decoder = tm.decoder
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # The decoder is the fixed message contract during the first transfer
        # probe.  Keeping it in eval mode also prevents one-batch BN statistics
        # from changing the official decoder behaviour.
        self.decoder.eval()
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)

    def train_mode(self) -> None:
        """Set only the encoder to training mode; keep the decoder frozen."""
        self.encoder.train()
        self.decoder.eval()

    def eval_mode(self) -> None:
        self.encoder.eval()
        self.decoder.eval()

    def differentiable_encode(self, cover: Tensor, message: Tensor) -> Tensor:
        """Mirror TrustMark's 256px residual post-processing without no_grad.

        TrustMark always runs its encoder at 256x256 and then merges the
        residual back at the input resolution.  Accepting the original
        resolution here is important for the 128x128 MBRS screen: local-tail
        patches then retain the MBRS P16/stride8 geometry, while the backbone
        still sees the official 256px input.
        """
        if cover.ndim != 4 or cover.shape[1] != 3:
            raise ValueError("cover must be NCHW RGB")
        if message.ndim != 2 or message.shape[0] != cover.shape[0] or message.shape[1] != 100:
            raise ValueError("TrustMark-Q expects a Bx100 internal message tensor")
        encoder_cover = cover
        if cover.shape[-2:] != (256, 256):
            encoder_cover = F.interpolate(
                cover, size=(256, 256), mode="bilinear", align_corners=False
            )
        raw_stego, _ = self.encoder(encoder_cover, message)
        residual = raw_stego.clamp(-1.0, 1.0) - encoder_cover
        # This is the channel-mean removal used by trustmark.py::encode before
        # resizing/blending.  Merge at the original input resolution.
        residual = residual - residual.mean(dim=(2, 3), keepdim=True)
        residual = F.interpolate(
            residual, size=cover.shape[-2:], mode="bilinear", align_corners=False
        )
        return (cover + residual).clamp(-1.0, 1.0)

    def decode_logits(self, stego: Tensor) -> Tensor:
        """Use the published decoder resolution before threshold/ECC."""
        decoder_input = stego
        resolution = int(self.trustmark.model_resolution_dec)
        if stego.shape[-2:] != (resolution, resolution):
            decoder_input = F.interpolate(
                stego, size=(resolution, resolution), mode="bilinear", align_corners=False
            )
        return self.decoder.decoder(decoder_input)

    def objective(
        self,
        cover: Tensor,
        message: Tensor,
        decoder_stego: Optional[Tensor] = None,
        decoder_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute the transparent base + local-tail transfer objective."""
        if decoder_stego is not None and decoder_mask is not None:
            raise ValueError("pass decoder_stego or decoder_mask, not both")
        stego = self.differentiable_encode(cover, message)
        if decoder_mask is not None:
            if decoder_mask.ndim != 4 or decoder_mask.shape[0] != stego.shape[0] or decoder_mask.shape[1] != 1:
                raise ValueError("decoder_mask must have shape Bx1xHxW")
            if decoder_mask.shape[-2:] != stego.shape[-2:]:
                raise ValueError("decoder_mask must match stego spatial dimensions")
            decoder_stego = stego * decoder_mask
        if decoder_stego is None:
            decoder_stego = stego
        if decoder_stego.shape != stego.shape:
            raise ValueError("decoder_stego must have the same shape as stego")
        logits = self.decode_logits(decoder_stego)
        image_loss = F.mse_loss(stego, cover)
        message_loss = F.binary_cross_entropy_with_logits(logits, message)
        local_loss, topk_count, _ = hard_local_tail_rgb_loss(stego, cover)

        weighted_image_loss = self.base_image_weight * image_loss
        weighted_local_loss = self.local_loss_weight * local_loss
        color_loss = stego.new_zeros(())
        if self.global_oklab_weight > 0:
            color_loss = global_oklab_components(stego, cover)["global_oklab"]
        weighted_color_loss = self.global_oklab_weight * color_loss
        weighted_message_loss = self.base_message_weight * message_loss
        total = (
            weighted_image_loss
            + weighted_message_loss
            + weighted_local_loss
            + weighted_color_loss
        )
        return {
            "loss": total,
            "stego": stego,
            "logits": logits,
            "image_loss": image_loss,
            "message_loss": message_loss,
            "local_tail_rgb_loss": local_loss,
            "global_oklab_loss": color_loss,
            "weighted_image_loss": weighted_image_loss,
            "weighted_message_loss": weighted_message_loss,
            "weighted_local_loss": weighted_local_loss,
            "weighted_global_oklab_loss": weighted_color_loss,
            "topk_count": stego.new_tensor(topk_count),
        }

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        cover: Tensor,
        message: Tensor,
        decoder_stego: Optional[Tensor] = None,
        decoder_mask: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Run one finite, gradient-bearing optimizer step."""
        self.train_mode()
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            metrics = self.objective(
                cover,
                message,
                decoder_stego=decoder_stego,
                decoder_mask=decoder_mask,
            )
            if not torch.isfinite(metrics["loss"]):
                raise FloatingPointError("non-finite TrustMark-Q adapter loss")
            metrics["loss"].backward()
            gradients = [
                parameter.grad
                for parameter in self.encoder.parameters()
                if parameter.requires_grad
            ]
            finite_gradients = [
                gradient
                for gradient in gradients
                if gradient is not None and torch.isfinite(gradient).all()
            ]
            metrics["grad_param_count"] = metrics["loss"].new_tensor(len(finite_gradients))
            metrics["grad_param_total"] = metrics["loss"].new_tensor(len(gradients))
            if len(finite_gradients) != len(gradients):
                raise RuntimeError("not every trainable TrustMark-Q encoder parameter received a finite gradient")
            optimizer.step()
        return {
            key: float(value.detach().cpu())
            for key, value in metrics.items()
            if torch.is_tensor(value) and value.ndim == 0
        }

    def checkpoint_payload(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
    ) -> Dict[str, object]:
        """Capture enough state to continue this adapter's research run."""
        return {
            "format": self.FORMAT,
            "model_type": "Q",
            "epoch": int(epoch),
            "step": int(step),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "objective": {
                "base_image_weight": self.base_image_weight,
                "base_message_weight": self.base_message_weight,
                "local_loss_weight": self.local_loss_weight,
                "global_oklab_weight": self.global_oklab_weight,
                "patch_size": 16,
                "patch_stride": 8,
                "topk_ratio": 0.10,
            },
        }

    def save_checkpoint(
        self,
        path: Path,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(optimizer, epoch, step), path)

    def load_adapter_checkpoint(
        self,
        path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, object]:
        checkpoint = load_checkpoint(path, self.device)
        if checkpoint.get("format") != self.FORMAT:
            raise ValueError("unsupported TrustMark-Q adapter checkpoint format")
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.decoder.eval()
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint


def make_optimizer(adapter: TrustMarkQAdapter, lr: float = 1e-6):
    """Create the conservative encoder-only optimizer for the first probe."""
    return torch.optim.Adam(
        (parameter for parameter in adapter.encoder.parameters() if parameter.requires_grad),
        lr=lr,
    )


def as_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
