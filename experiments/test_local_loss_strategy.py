#!/usr/bin/env python3
"""Fast checks for local-weight scheduling and soft-tail gradients."""

import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.losses import image_loss_components
from experiments.train_local_patch import image_loss_weights


def check_schedule():
	config = {
		"epochs": 100,
		"global_loss_weight": 0.75,
		"local_loss_weight": 0.25,
		"local_weight_schedule": {
			"type": "linear_warmup",
			"start_fraction": 0.4,
			"end_fraction": 0.6,
			"target_weight": 0.25,
			"preserve_total_image_weight": True,
			"total_image_weight": 1.0,
		},
	}
	expected = {
		1: (1.0, 0.0),
		40: (1.0, 0.0),
		50: (0.875, 0.125),
		60: (0.75, 0.25),
		100: (0.75, 0.25),
	}
	for epoch, target in expected.items():
		actual = image_loss_weights(config, epoch)
		assert all(abs(a - b) < 1e-12 for a, b in zip(actual, target)), (
			epoch,
			actual,
			target,
		)
		print("schedule epoch={} global={:.3f} local={:.3f}".format(epoch, *actual))


def gradient_support(mode, base, detach_weights=True):
	encoded = base.clone().requires_grad_(True)
	cover = torch.zeros_like(encoded)
	components = image_loss_components(
		encoded,
		cover,
		mode=mode,
		patch_size=32,
		topk_ratio=0.25,
		soft_tail_temperature=0.5,
		soft_tail_detach_weights=detach_weights,
	)
	gradient = torch.autograd.grad(components["local_mse"], encoded)[0]
	return components, float((gradient.abs() > 0).float().mean()), gradient


def check_soft_tail():
	base = torch.randn(2, 3, 128, 128)
	hard, hard_support, _ = gradient_support("topk", base)
	soft_detached, soft_support, detached_gradient = gradient_support(
		"soft_tail", base, True
	)
	soft_full, full_support, full_gradient = gradient_support(
		"soft_tail", base, False
	)
	assert abs(hard_support - 0.25) < 1e-6, hard_support
	assert soft_support == 1.0, soft_support
	assert full_support == 1.0, full_support
	assert soft_detached["soft_tail_mse"] >= soft_detached["global_mse"]
	assert abs(soft_detached["soft_weight_mean"].item() - 1 / 16) < 1e-7
	assert 1 < soft_detached["soft_effective_patches"].item() <= 16
	assert not torch.allclose(detached_gradient, full_gradient)
	print(
		"gradient_support hard={:.3f} soft_detached={:.3f} soft_full={:.3f}".format(
			hard_support, soft_support, full_support
		)
	)
	print(
		"soft_tail global={:.6f} hard_topk={:.6f} soft={:.6f}".format(
			soft_detached["global_mse"],
			hard["worst_patch_mse"],
			soft_detached["soft_tail_mse"],
		)
	)
	print(
		"soft_weights mean={:.6f} max={:.6f} effective_patches={:.2f}".format(
			soft_detached["soft_weight_mean"],
			soft_detached["soft_weight_max"],
			soft_detached["soft_effective_patches"],
		)
	)


def check_multiscale():
	encoded = torch.randn(2, 3, 128, 128, requires_grad=True)
	cover = torch.zeros_like(encoded)
	components = image_loss_components(
		encoded,
		cover,
		mode="multiscale",
		patch_size=16,
		topk_ratio=0.25,
		multi_scale_patch_sizes=[16, 32],
		multi_scale_patch_strides=[16, 32],
		multi_scale_weights=[0.7, 0.3],
	)
	assert components["local_mse"].item() > 0
	components["local_mse"].backward()
	assert encoded.grad is not None and encoded.grad.abs().sum().item() > 0
	print(
		"multiscale local={:.6f} global={:.6f}".format(
			components["local_mse"], components["global_mse"]
		)
	)


def check_excess():
	encoded = torch.randn(2, 3, 128, 128, requires_grad=True)
	cover = torch.zeros_like(encoded)
	components = image_loss_components(
		encoded,
		cover,
		mode="excess",
		patch_size=16,
		patch_stride=16,
		topk_ratio=0.25,
		excess_threshold=1.0,
		excess_loss_scale=3.0,
	)
	assert components["local_mse"].item() >= 0
	components["local_mse"].backward()
	assert encoded.grad is not None and encoded.grad.abs().sum().item() > 0
	print(
		"excess local={:.6f} global={:.6f}".format(
			components["local_mse"], components["global_mse"]
		)
	)


if __name__ == "__main__":
	torch.manual_seed(20260908)
	check_schedule()
	check_soft_tail()
	check_multiscale()
	check_excess()
