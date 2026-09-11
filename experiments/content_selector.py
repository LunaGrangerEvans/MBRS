"""Detached original-content ranking on the fixed Patch16/stride8 grid."""

import math

import torch
import torch.nn.functional as F


def patches(images):
    if images.ndim != 4 or images.shape[-2:] != (128, 128):
        raise ValueError("content selector protocol requires NCHW 128x128 images")
    return F.unfold(images, 16, stride=8).transpose(1, 2).reshape(
        images.shape[0], 225, images.shape[1], 16, 16
    )


@torch.no_grad()
def robust_normalize(values):
    low = torch.quantile(values, 0.05, dim=1, keepdim=True)
    high = torch.quantile(values, 0.95, dim=1, keepdim=True)
    span = high - low
    return torch.where(
        span > 1e-12, ((values - low) / span.clamp_min(1e-12)).clamp(0, 1), 0
    )


@torch.no_grad()
def content_activity(original, diagnostics=True):
    rgb = (original.detach() + 1) / 2
    weights = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    gray = patches((rgb * weights).sum(1, keepdim=True)).squeeze(2)
    dx = gray[:, :, :-1, 1:] - gray[:, :, :-1, :-1]
    dy = gray[:, :, 1:, :-1] - gray[:, :, :-1, :-1]
    gradient_sites = (dx.square() + dy.square()).sqrt()
    gradient = gradient_sites.mean((-2, -1))
    if not diagnostics:
        return {"gradient_normalized": robust_normalize(gradient)}
    laplacian = (
        gray[:, :, 1:-1, :-2] + gray[:, :, 1:-1, 2:]
        + gray[:, :, :-2, 1:-1] + gray[:, :, 2:, 1:-1]
        - 4 * gray[:, :, 1:-1, 1:-1]
    )
    hf = laplacian.square().mean((-2, -1))
    threshold = torch.quantile(gradient_sites.flatten(1), 0.75, dim=1)
    return {
        "gradient": gradient,
        "hf": hf,
        "variance": gray.var((-2, -1), unbiased=False),
        "edge_density": (gradient_sites > threshold[:, None, None, None]).float().mean((-2, -1)),
        "gradient_normalized": robust_normalize(gradient),
        "hf_normalized": robust_normalize(hf),
    }


@torch.no_grad()
def selector_scores(mse, activity, feature="mse", alpha=0.0):
    if feature == "mse":
        return mse.detach()
    if feature not in {"gradient", "gradient_hf"} or alpha not in {0.5, 1.0, 2.0}:
        raise ValueError("outside the frozen analysis candidate set")
    value = activity["gradient_normalized"]
    if feature == "gradient_hf":
        value = (value + activity["hf_normalized"]) / 2
    return mse.detach() / (1 + alpha * value)


def selected_indices(scores, ratio=0.1):
    if ratio not in {0.1, 0.25}:
        raise ValueError("only Top10 and diagnostic Top25 are in scope")
    return torch.argsort(scores.detach(), dim=1, descending=True, stable=True)[
        :, :math.ceil(scores.shape[1] * ratio)
    ]
