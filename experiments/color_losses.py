"""Differentiable OKLab local-tail losses for the frozen seed17 study."""
import math

import torch
import torch.nn.functional as F


def rgb_to_oklab(rgb, eps=1e-12):
    """Convert sRGB in [0,1] NCHW tensors to OKLab NCHW tensors."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb_to_oklab expects NCHW RGB tensors")
    rgb = rgb.clamp(0, 1)
    threshold = 0.04045
    linear = torch.where(
        rgb <= threshold,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).clamp_min(eps).pow(2.4),
    )
    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    l_channel = 0.8189330101 * r + 0.3618667424 * g - 0.1288597137 * b
    m_channel = 0.0329845436 * r + 0.9293118715 * g + 0.0361456387 * b
    s_channel = 0.0482003018 * r + 0.2643662691 * g + 0.6338517070 * b
    l_ = torch.sign(l_channel) * l_channel.abs().clamp_min(eps).pow(1.0 / 3.0)
    m_ = torch.sign(m_channel) * m_channel.abs().clamp_min(eps).pow(1.0 / 3.0)
    s_ = torch.sign(s_channel) * s_channel.abs().clamp_min(eps).pow(1.0 / 3.0)
    return torch.cat(
        [
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ],
        dim=1,
    )


def oklab_pixel_distances(encoded, cover, eps=1e-12):
    encoded_lab = rgb_to_oklab(((encoded + 1.0) / 2.0).clamp(0, 1), eps=eps)
    cover_lab = rgb_to_oklab(((cover + 1.0) / 2.0).clamp(0, 1), eps=eps)
    delta = encoded_lab - cover_lab
    distance = delta.square().sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
    chroma = delta[:, 1:3].square().sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
    return distance, chroma


def oklab_tail_components(encoded, cover, patch_size=5, patch_stride=1, topk_ratio=0.1, eps=1e-12):
    """Return differentiable 5x5 sliding OKLab tail and diagnostics."""
    if patch_size <= 0 or patch_stride <= 0:
        raise ValueError("patch_size and patch_stride must be positive")
    distances, chroma = oklab_pixel_distances(encoded, cover, eps=eps)
    patch_scores = F.avg_pool2d(distances, patch_size, stride=patch_stride).flatten(1)
    chroma_scores = F.avg_pool2d(chroma, patch_size, stride=patch_stride).flatten(1)
    mse_pixels = (encoded - cover).square().mean(dim=1, keepdim=True)
    mse_scores = F.avg_pool2d(mse_pixels, patch_size, stride=patch_stride).flatten(1)
    count = patch_scores.shape[1]
    k = max(1, math.ceil(count * topk_ratio))
    selected = torch.topk(patch_scores, k, dim=1, largest=True, sorted=False).values
    selected_chroma = torch.topk(chroma_scores, k, dim=1, largest=True, sorted=False).values
    selected_mse = torch.topk(mse_scores, k, dim=1, largest=True, sorted=False).values
    return {
        "global_mse": (encoded - cover).square().mean(),
        "patch_mean_mse": mse_scores.mean(),
        "worst_patch_mse": selected_mse.mean(),
        "local_mse": selected.mean(),
        "local_oklab": selected.mean(),
        "local_chroma": selected_chroma.mean(),
        "global_oklab": distances.mean(),
        "patch_oklab": patch_scores,
        "patch_chroma": chroma_scores,
        "selected_patch_count": k,
        "soft_weight_mean": encoded.new_zeros(()),
        "soft_weight_max": encoded.new_zeros(()),
        "soft_effective_patches": encoded.new_zeros(()),
    }
