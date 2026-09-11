import math

import torch
import torch.nn.functional as F


def _patch_scores_and_weights(
	encoded_images, cover_images, patch_size, patch_stride=None
):
	"""Return valid-pixel-weighted MSE scores for a patch grid."""
	if encoded_images.shape != cover_images.shape:
		raise ValueError("encoded and cover images must have the same shape")
	if patch_size <= 0:
		raise ValueError("patch_size must be positive")
	if patch_stride is None:
		patch_stride = patch_size
	if patch_stride <= 0:
		raise ValueError("patch_stride must be positive")
	_, channels, height, width = encoded_images.shape
	pad_height = (patch_size - height % patch_size) % patch_size
	pad_width = (patch_size - width % patch_size) % patch_size
	residual = (encoded_images - cover_images).pow(2)
	residual = F.pad(residual, (0, pad_width, 0, pad_height))
	valid = encoded_images.new_ones((1, 1, height, width))
	valid = F.pad(valid, (0, pad_width, 0, pad_height))

	patches = F.unfold(residual, kernel_size=patch_size, stride=patch_stride)
	patches = patches.view(
		encoded_images.shape[0], channels, patch_size * patch_size, -1
	)
	patch_sums = patches.sum(dim=2).sum(dim=1)
	patch_weights = F.unfold(
		valid, kernel_size=patch_size, stride=patch_stride
	).sum(dim=1).squeeze(0)
	covered_pixels = patch_weights.sum().item()
	if patch_stride == patch_size:
		assert covered_pixels == height * width, (
			"patch coverage mismatch: covered_pixels={}, expected_pixels={}".format(
				covered_pixels, height * width
			)
		)
	else:
		assert covered_pixels >= height * width, (
			"overlap coverage mismatch: covered_pixels={}, expected_at_least={}".format(
				covered_pixels, height * width
			)
		)
	patch_scores = patch_sums / (patch_weights.unsqueeze(0) * channels).clamp_min(1)
	return patch_scores, patch_weights


def patch_mse_per_sample(
	encoded_images, cover_images, patch_size, patch_stride=None
):
	"""Return one valid-pixel-weighted MSE score per patch and sample."""
	patch_scores, _ = _patch_scores_and_weights(
		encoded_images, cover_images, patch_size, patch_stride
	)
	return patch_scores


def _topk_patch_mean(patch_scores, patch_weights, topk_ratio):
	"""Return the per-image weighted mean of the highest-error patches."""
	patch_count = patch_scores.shape[1]
	topk_count = max(1, int(math.ceil(patch_count * topk_ratio)))
	top_values, top_indices = torch.topk(
		patch_scores, topk_count, dim=1, largest=True, sorted=False
	)
	top_weights = patch_weights.unsqueeze(0).expand_as(patch_scores).gather(
		1, top_indices
	)
	return (
		(top_values * top_weights).sum(dim=1)
		/ top_weights.sum(dim=1).clamp_min(1)
	).mean()


def image_loss_components(
	encoded_images,
	cover_images,
	mode="none",
	patch_size=32,
	patch_stride=None,
	topk_ratio=0.25,
	multi_scale_patch_sizes=None,
	multi_scale_patch_strides=None,
	multi_scale_weights=None,
	excess_threshold=1.0,
	excess_epsilon=1e-12,
	excess_loss_scale=1.0,
	content_selector_feature="gradient",
	content_selector_alpha=1.0,
	soft_tail_temperature=0.5,
	soft_tail_detach_weights=True,
):
	"""Compute global, average-patch, and worst-patch image distortion."""
	if mode not in {
		"none",
		"mean",
		"topk",
		"soft_tail",
		"multiscale",
		"excess",
		"contentaware_gradient",
	}:
		raise ValueError(
			"mode must be one of: none, mean, topk, soft_tail, multiscale, excess, contentaware_gradient"
		)
	if not 0 < topk_ratio <= 1:
		raise ValueError("topk_ratio must be in (0, 1]")
	if soft_tail_temperature <= 0:
		raise ValueError("soft_tail_temperature must be positive")

	global_mse = F.mse_loss(encoded_images, cover_images)
	patch_scores, patch_weights = _patch_scores_and_weights(
		encoded_images, cover_images, patch_size, patch_stride
	)
	patch_stride = patch_size if patch_stride is None else patch_stride
	patch_mean_raw_mse = (
		(patch_scores * patch_weights.unsqueeze(0)).sum(dim=1)
		/ patch_weights.sum().clamp_min(1)
	).mean()
	difference = (global_mse - patch_mean_raw_mse).abs().item()
	assertion_message = (
		"global_loss={:.10g}, patch_mean_loss={:.10g}, abs_diff={:.10g} "
		"(tolerance=1e-6)".format(
			global_mse.item(), patch_mean_raw_mse.item(), difference
		)
	)
	if patch_stride == patch_size and difference >= 1e-6:
		print("PATCH_MSE_ASSERTION_FAILED " + assertion_message, flush=True)
	if patch_stride == patch_size:
		assert difference < 1e-6, assertion_message
	# Use the global reduction as the canonical value. The independently
	# computed patch reduction is still asserted above, while this avoids
	# changing the optimizer's floating-point reduction path in mean mode.
	patch_mean_mse = (
		global_mse if patch_stride == patch_size else patch_mean_raw_mse
	)
	worst_patch_mse = _topk_patch_mean(
		patch_scores, patch_weights, topk_ratio
	)
	multiscale_local_mse = global_mse.detach() * 0
	excess_local_mse = global_mse.detach() * 0
	if mode == "excess":
		mean_patch_error = patch_scores.mean(dim=1, keepdim=True).detach()
		normalized_error = patch_scores / mean_patch_error.clamp_min(excess_epsilon)
		excess = F.relu(normalized_error - excess_threshold)
		excess_local_mse = (
			mean_patch_error * excess.pow(2)
		).mean() * excess_loss_scale
	if mode == "multiscale":
		if not multi_scale_patch_sizes or not multi_scale_weights:
			raise ValueError(
				"multiscale mode requires multi_scale_patch_sizes and "
				"multi_scale_weights"
			)
		if len(multi_scale_patch_sizes) != len(multi_scale_weights):
			raise ValueError("multiscale sizes and weights must have equal length")
		if multi_scale_patch_strides is None:
			multi_scale_patch_strides = list(multi_scale_patch_sizes)
		if len(multi_scale_patch_strides) != len(multi_scale_patch_sizes):
			raise ValueError("multiscale strides must match patch sizes")
		weight_sum = sum(float(value) for value in multi_scale_weights)
		if weight_sum <= 0:
			raise ValueError("multiscale weights must sum to a positive value")
		for size, stride, weight in zip(
			multi_scale_patch_sizes,
			multi_scale_patch_strides,
			multi_scale_weights,
		):
			scale_scores, scale_weights = _patch_scores_and_weights(
				encoded_images, cover_images, int(size), int(stride)
			)
			multiscale_local_mse = multiscale_local_mse + (
				float(weight)
				/ weight_sum
				* _topk_patch_mean(scale_scores, scale_weights, topk_ratio)
			)
	soft_tail_mse = global_mse.detach() * 0
	soft_weight_mean = global_mse.detach() * 0
	soft_weight_max = global_mse.detach() * 0
	soft_effective_patches = global_mse.detach() * 0
	if mode == "soft_tail":
		# Normalize each image's patch scores by its detached mean so a fixed
		# temperature has comparable selectivity throughout training. Detaching
		# the weights gives a smooth spatial reweighting objective without the
		# additional gradient-through-attention term.
		normalized_scores = patch_scores / patch_scores.mean(
			dim=1, keepdim=True
		).detach().clamp_min(1e-12)
		soft_tail_weights = torch.softmax(
			normalized_scores / soft_tail_temperature, dim=1
		)
		if soft_tail_detach_weights:
			soft_tail_weights = soft_tail_weights.detach()
		soft_weight_mean = soft_tail_weights.mean()
		soft_weight_max = soft_tail_weights.max(dim=1).values.mean()
		soft_effective_patches = (
			1 / soft_tail_weights.pow(2).sum(dim=1).clamp_min(1e-12)
		).mean()
		soft_valid_weights = soft_tail_weights * patch_weights.unsqueeze(0)
		soft_tail_mse = (
			(soft_valid_weights * patch_scores).sum(dim=1)
			/ soft_valid_weights.sum(dim=1).clamp_min(1)
		).mean()
	if mode == "none":
		local_mse = global_mse.detach() * 0
	elif mode == "mean":
		local_mse = patch_mean_mse
	elif mode == "topk":
		local_mse = worst_patch_mse
	elif mode == "contentaware_gradient":
		from experiments.content_selector import (
			content_activity, selected_indices, selector_scores,
		)

		if patch_size != 16 or patch_stride != 8 or topk_ratio != 0.1:
			raise ValueError("frozen content-aware formulation requires Patch16/stride8/Top10")
		activity = content_activity(cover_images, diagnostics=False)
		selection_scores = selector_scores(
			patch_scores,
			activity,
			content_selector_feature,
			content_selector_alpha,
		)
		indices = selected_indices(selection_scores)
		# Only the detached indices change; optimize the original patch MSE.
		local_mse = patch_scores.gather(1, indices).mean()
	elif mode == "multiscale":
		local_mse = multiscale_local_mse
	elif mode == "excess":
		local_mse = excess_local_mse
	else:
		local_mse = soft_tail_mse

	return {
		"global_mse": global_mse,
		"patch_mean_mse": patch_mean_mse,
		"worst_patch_mse": worst_patch_mse,
		"soft_tail_mse": soft_tail_mse,
		"soft_weight_mean": soft_weight_mean,
		"soft_weight_max": soft_weight_max,
		"soft_effective_patches": soft_effective_patches,
		"local_mse": local_mse,
	}


def mse_to_psnr(mse, max_value=2.0):
	"""Convert MSE in [-1, 1] tensor space to dB."""
	return 10 * torch.log10((max_value * max_value) / mse.clamp_min(1e-12))


def ssim_index(input_images, target_images, window_size=5, max_value=2.0):
	"""Compute a dependency-free SSIM index for NCHW tensors."""
	if input_images.shape != target_images.shape:
		raise ValueError("SSIM inputs must have the same shape")
	if window_size <= 0 or window_size % 2 == 0:
		raise ValueError("SSIM window_size must be a positive odd number")
	coords = torch.arange(window_size, device=input_images.device, dtype=input_images.dtype)
	coords = coords - window_size // 2
	one_dim = torch.exp(-(coords * coords) / (2 * 1.5 * 1.5))
	window = one_dim[:, None] * one_dim[None, :]
	window = window / window.sum()
	channels = input_images.shape[1]
	window = window.view(1, 1, window_size, window_size).expand(channels, 1, -1, -1)
	padding = window_size // 2
	mu_input = F.conv2d(input_images, window, padding=padding, groups=channels)
	mu_target = F.conv2d(target_images, window, padding=padding, groups=channels)
	input_sq = F.conv2d(input_images * input_images, window, padding=padding, groups=channels)
	target_sq = F.conv2d(target_images * target_images, window, padding=padding, groups=channels)
	cross = F.conv2d(input_images * target_images, window, padding=padding, groups=channels)
	variance_input = input_sq - mu_input * mu_input
	variance_target = target_sq - mu_target * mu_target
	covariance = cross - mu_input * mu_target
	c1 = (0.01 * max_value) ** 2
	c2 = (0.03 * max_value) ** 2
	ssim = ((2 * mu_input * mu_target + c1) * (2 * covariance + c2)) / (
		(mu_input * mu_input + mu_target * mu_target + c1)
		* (variance_input + variance_target + c2)
	)
	return ssim.mean()
