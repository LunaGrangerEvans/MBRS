"""Global OKLab constraint to accompany crop Hard16/stride8/Top10 losses.

Conversion follows Björn Ottosson's linear-sRGB implementation (2021 matrices):
https://bottosson.github.io/posts/oklab/
This module is independent of the historical conversion in color_losses.py.
"""

import torch
import torch.nn.functional as F


_LMS_EPS = 1e-12
_DISTANCE_EPS = 1e-12


def rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert float32/float64 sRGB [0, 1], NCHW, to OKLab NCHW.

    Inputs are clamped to [0, 1]. Apply the sRGB inverse transfer before
    the author's linear-RGB-to-LMS matrix and standard M2 matrix.

    The cube root has an infinite derivative at zero. Below LMS=1e-12,
    use x * eps**(-2/3), joining the exact cube root continuously at eps.
    This preserves exact black and finite, nonzero gradients there; only
    extremely dark values (neutral sRGB below 1.292e-11) are approximated.
    Clamp the power's input even in the unselected torch.where branch to
    avoid 0 * inf in backward. All outputs retain their autograd graph;
    this is an actual forward regularization, not a detached surrogate.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb_to_oklab expects NCHW RGB tensors")
    if not rgb.is_floating_point():
        raise TypeError("rgb_to_oklab expects floating-point RGB tensors")
    rgb = rgb.clamp(0, 1)
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )
    r, g, b = linear.unbind(dim=1)
    lms = torch.stack(
        (
            0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
            0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
            0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b,
        ),
        dim=1,
    )
    roots = torch.where(
        lms < _LMS_EPS,
        lms * _LMS_EPS ** (-2.0 / 3.0),
        lms.clamp_min(_LMS_EPS).pow(1.0 / 3.0),
    )
    l_root, m_root, s_root = roots.unbind(dim=1)
    return torch.stack(
        (
            0.2104542553 * l_root + 0.7936177850 * m_root - 0.0040720468 * s_root,
            1.9779984951 * l_root - 2.4285922050 * m_root + 0.4505937099 * s_root,
            0.0259040371 * l_root + 0.7827717662 * m_root - 0.8086757660 * s_root,
        ),
        dim=1,
    )


def global_oklab_components(encoded: torch.Tensor, cover: torch.Tensor) -> dict:
    """Return four differentiable scalar means for matching NCHW [-1, 1] images.

    Delta is OKLab(encoded) - OKLab(cover), after mapping to sRGB [0, 1].
    global_oklab averages sqrt(dL² + da² + db² + 1e-12) - 1e-6 over
    every pixel and batch item. global_chroma uses only da² + db² with
    the same smoothing. These are means of pixel distances, not distances
    between mean colors. Subtracting sqrt(eps) makes identity exactly zero
    while eps keeps backward finite at zero distance. signed_oklab_da/db
    are the signed mean channel differences, also attached to autograd.

    No patch selection or weighting is performed here: callers can add
    global_oklab to the existing Hard16/stride8/Top10 objective.
    """
    if encoded.shape != cover.shape:
        raise ValueError("encoded and cover images must have the same shape")
    delta = rgb_to_oklab((encoded + 1.0) / 2.0) - rgb_to_oklab((cover + 1.0) / 2.0)
    distance = (delta.square().sum(dim=1) + _DISTANCE_EPS).sqrt()
    chroma = (delta[:, 1:3].square().sum(dim=1) + _DISTANCE_EPS).sqrt()
    return {
        "global_oklab": (distance - _DISTANCE_EPS**0.5).mean(),
        "global_chroma": (chroma - _DISTANCE_EPS**0.5).mean(),
        "signed_oklab_da": delta[:, 1].mean(),
        "signed_oklab_db": delta[:, 2].mean(),
    }


def chroma_tail_components(
    encoded: torch.Tensor,
    cover: torch.Tensor,
    patch_size: int = 16,
    patch_stride: int = 8,
    topk_ratio: float = 0.1,
) -> dict:
    """Return a differentiable Top-k local OKLab chroma-tail objective.

    Chroma distance is sqrt((delta-a)^2 + (delta-b)^2) per pixel. Patch
    scores are the mean pixel chroma distance in the overlapping patch grid;
    only the highest raw chroma patches contribute to local_chroma.
    """
    if encoded.shape != cover.shape:
        raise ValueError("encoded and cover images must have the same shape")
    if patch_size != 16 or patch_stride != 8 or topk_ratio != 0.1:
        raise ValueError("frozen chroma-tail requires Patch16/stride8/Top10")
    encoded_lab = rgb_to_oklab((encoded + 1.0) / 2.0)
    cover_lab = rgb_to_oklab((cover + 1.0) / 2.0)
    delta = encoded_lab - cover_lab
    chroma_pixels = (delta[:, 1:3].square().sum(dim=1, keepdim=True) + _DISTANCE_EPS).sqrt()
    patch_scores = F.avg_pool2d(chroma_pixels, patch_size, stride=patch_stride).flatten(1)
    count = patch_scores.shape[1]
    k = max(1, int(torch.ceil(patch_scores.new_tensor(count * topk_ratio)).item()))
    selected = torch.topk(patch_scores, k, dim=1, largest=True, sorted=False).values
    return {
        "local_chroma": (selected - _DISTANCE_EPS**0.5).mean(),
        "patch_chroma": patch_scores,
        "selected_chroma_patch_count": k,
    }
