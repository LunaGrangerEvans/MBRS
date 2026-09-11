#!/usr/bin/env python3
"""Minimal correctness checks for global/patch image reductions."""

import math
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.losses import _patch_scores_and_weights, image_loss_components


def check_shape(height, width, patch_size):
	encoded = torch.randn(2, 3, height, width, requires_grad=True)
	cover = torch.randn(2, 3, height, width)
	patch_scores, patch_weights = _patch_scores_and_weights(
		encoded, cover, patch_size
	)


	independent_patch_mean = (
		(patch_scores * patch_weights.unsqueeze(0)).sum(dim=1)
		/ patch_weights.sum()
	).mean()
	components = image_loss_components(
		encoded, cover, mode="mean", patch_size=patch_size, topk_ratio=0.25
	)
	global_value = components["global_mse"].item()
	patch_value = independent_patch_mean.item()
	absolute_error = abs(global_value - patch_value)
	relative_error = absolute_error / max(abs(global_value), 1e-12)
	assert absolute_error < 1e-6, absolute_error
	assert components["worst_patch_mse"].item() >= components["patch_mean_mse"].item()
	components["local_mse"].backward()
	assert encoded.grad is not None and encoded.grad.abs().sum().item() > 0
	print(
		"ok shape={}x{} patch={} patches={} global={:.10f} "
		"patch_mean={:.10f} abs_error={:.3e} rel_error={:.3e} worst={:.10f}".format(
			height,
			width,
			patch_size,
			patch_scores.shape[1],
			global_value,
			patch_value,
			absolute_error,
			relative_error,
			components["worst_patch_mse"].item(),
		)
	)


def check_overlap():
	encoded = torch.randn(1, 3, 128, 128, requires_grad=True)
	cover = torch.randn(1, 3, 128, 128)
	patch_scores, patch_weights = _patch_scores_and_weights(
		encoded, cover, patch_size=16, patch_stride=8
	)
	assert patch_scores.shape[1] == 225, patch_scores.shape
	assert patch_weights.min().item() == 256
	assert patch_weights.max().item() == 256
	components = image_loss_components(
		encoded,
		cover,
		mode="topk",
		patch_size=16,
		patch_stride=8,
		topk_ratio=0.25,
	)
	assert components["worst_patch_mse"].item() >= components["global_mse"].item()
	components["local_mse"].backward()
	assert encoded.grad is not None and encoded.grad.abs().sum().item() > 0
	print(
		"ok overlap patch=16 stride=8 patches={} selected={} coverage={}x".format(
			patch_scores.shape[1],
			max(1, int(math.ceil(patch_scores.shape[1] * 0.25))),
			int(patch_weights.sum().item() / (128 * 128)),
		)
	)


if __name__ == "__main__":
	torch.manual_seed(20260908)
	for patch_size in (16, 32, 64):
		check_shape(128, 128, patch_size)
	# Also lock the valid-pixel weighting behavior at non-divisible boundaries.
	check_shape(130, 126, 32)
	check_overlap()
