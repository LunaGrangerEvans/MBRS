#!/usr/bin/env python3
"""Measure per-image local-distortion tails for selected epoch-100 models."""

import argparse
import csv
import math
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.losses import patch_mse_per_sample
from network.Encoder_MP_Decoder import EncoderDecoder


ROOT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
DEFAULT_MODELS = OrderedDict(
	[
		("No-crop Global", ROOT / "experiments/runs/nocrop_global_128_m64/checkpoint_0100.pth"),
		(
			"Crop-trained Global",
			ROOT / "experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth",
		),
		(
			"Patch16 top25 weight50",
			ROOT / "experiments/runs/fixed_ablation_patch16_128_m64_crop/checkpoint_0100.pth",
		),
	]
)


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--manifest",
		type=Path,
		default=ROOT / "reports/uniform_eval_manifest.pt",
	)
	parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--patch-size", type=int, default=32)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--report", type=Path, default=Path("reports/patch_distortion_analysis.md"))
	parser.add_argument("--per-image-csv", type=Path, default=Path("reports/patch_distortion_per_image.csv"))
	parser.add_argument("--tail-csv", type=Path, default=Path("reports/patch_distortion_tail_stats.csv"))
	parser.add_argument(
		"--figure-dir",
		type=Path,
		default=ROOT / "visualizations/local_loss_diagnostics_20260908",
	)
	parser.add_argument("--skip-lpips", action="store_true")
	return parser.parse_args()


def torch_load(path, device):
	try:
		return torch.load(str(path), map_location=device, weights_only=False)
	except TypeError:
		return torch.load(str(path), map_location=device)


def extract_patches(images, patch_size):
	unfolded = F.unfold(images, kernel_size=patch_size, stride=patch_size)
	batch, _, patch_count = unfolded.shape
	return unfolded.transpose(1, 2).reshape(
		batch * patch_count, images.shape[1], patch_size, patch_size
	)


def ssim_per_sample(input_images, target_images, window_size=5, max_value=2.0):
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
	return ssim.mean(dim=(1, 2, 3))


def lpips_per_patch(input_patches, target_patches, device, batch_size):
	import lpips

	metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
	values = []
	with torch.no_grad():
		for start in range(0, input_patches.shape[0], batch_size):
			stop = start + batch_size
			value = metric(
				input_patches[start:stop].to(device),
				target_patches[start:stop].to(device),
			)
			values.append(value.flatten().cpu())
	return torch.cat(values)


def load_encoded(checkpoint_path, images, messages, device, batch_size):
	checkpoint = torch_load(checkpoint_path, device)
	config = checkpoint.get("config", {})
	model = EncoderDecoder(
		int(config.get("H", images.shape[-2])),
		int(config.get("W", images.shape[-1])),
		int(config.get("message_length", messages.shape[-1])),
		["Identity()"],
	).to(device)
	model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
	model.eval()
	encoded = []
	with torch.no_grad():
		for start in range(0, images.shape[0], batch_size):
			stop = start + batch_size
			encoded.append(
				model.encoder(
					images[start:stop].to(device), messages[start:stop].to(device)
				).cpu()
			)
	return torch.cat(encoded)


def calculate_model_metrics(name, images, encoded, patch_size, lpips_values):
	patch_scores = patch_mse_per_sample(encoded, images, patch_size).detach().cpu()
	patch_count = patch_scores.shape[1]
	top10_count = max(1, math.ceil(patch_count * 0.10))
	top25_count = max(1, math.ceil(patch_count * 0.25))
	global_mse = (encoded - images).pow(2).mean(dim=(1, 2, 3))
	patch_std = patch_scores.std(dim=1, unbiased=False)
	top10 = torch.topk(patch_scores, top10_count, dim=1).values.mean(dim=1)
	top25 = torch.topk(patch_scores, top25_count, dim=1).values.mean(dim=1)
	maximum = patch_scores.max(dim=1).values
	encoded_patches = extract_patches(encoded, patch_size)
	image_patches = extract_patches(images, patch_size)
	with torch.no_grad():
		patch_ssim = ssim_per_sample(encoded_patches, image_patches).reshape(-1, patch_count)
	local_ssim_top25 = torch.topk(
		patch_ssim, top25_count, dim=1, largest=False
	).values.mean(dim=1)
	local_ssim_worst = patch_ssim.min(dim=1).values
	if lpips_values is None:
		local_lpips_top25 = torch.full_like(global_mse, float("nan"))
		local_lpips_worst = torch.full_like(global_mse, float("nan"))
	else:
		patch_lpips = lpips_values.reshape(-1, patch_count)
		local_lpips_top25 = torch.topk(
			patch_lpips, top25_count, dim=1, largest=True
		).values.mean(dim=1)
		local_lpips_worst = patch_lpips.max(dim=1).values
	rows = []
	for index in range(images.shape[0]):
		rows.append(
			{
				"model": name,
				"image_index": index,
				"global_mse": float(global_mse[index]),
				"patch_mean_mse": float(patch_scores[index].mean()),
				"top10_patch_mse": float(top10[index]),
				"top25_patch_mse": float(top25[index]),
				"max_patch_mse": float(maximum[index]),
				"patch_mse_std": float(patch_std[index]),
				"patch_mse_cv": float(patch_std[index] / global_mse[index].clamp_min(1e-12)),
				"worst_global_ratio": float(top25[index] / global_mse[index].clamp_min(1e-12)),
				"max_global_ratio": float(maximum[index] / global_mse[index].clamp_min(1e-12)),
				"local_ssim_top25": float(local_ssim_top25[index]),
				"local_ssim_worst": float(local_ssim_worst[index]),
				"local_lpips_top25": float(local_lpips_top25[index]),
				"local_lpips_worst": float(local_lpips_worst[index]),
			}
		)
	return rows, patch_scores.numpy()


def write_csv(path, rows):
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def summarize(rows, patch_scores_by_model):
	metrics = [
		"global_mse",
		"top10_patch_mse",
		"top25_patch_mse",
		"max_patch_mse",
		"patch_mse_std",
		"patch_mse_cv",
		"worst_global_ratio",
		"max_global_ratio",
		"local_ssim_top25",
		"local_ssim_worst",
		"local_lpips_top25",
		"local_lpips_worst",
	]
	by_model = OrderedDict((name, []) for name in DEFAULT_MODELS)
	for row in rows:
		by_model[row["model"]].append(row)
	summaries = []
	for name, model_rows in by_model.items():
		summary = {"model": name, "images": len(model_rows)}
		for metric in metrics:
			values = np.asarray([row[metric] for row in model_rows], dtype=float)
			summary[metric + "_mean"] = float(np.nanmean(values))
			summary[metric + "_std"] = float(np.nanstd(values, ddof=1))
		patch_values = patch_scores_by_model[name].reshape(-1)
		for percentile in (50, 90, 95, 99):
			summary["patch_mse_p{}".format(percentile)] = float(
				np.percentile(patch_values, percentile)
			)
		summaries.append(summary)
	return summaries, by_model


def plot_distributions(figure_dir, patch_scores_by_model, by_model):
	figure_dir.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
	for name, scores in patch_scores_by_model.items():
		flat = scores.reshape(-1)
		axes[0].hist(flat, bins=60, density=True, histtype="step", linewidth=1.8, label=name)
		x = np.sort(flat)
		y = np.arange(1, len(x) + 1) / len(x)
		axes[1].plot(x, y, linewidth=1.8, label=name)
	for axis in axes:
		axis.set_xscale("log")
		axis.grid(alpha=0.25)
		axis.set_xlabel("32x32 patch MSE")
	axes[0].set_ylabel("Density")
	axes[1].set_ylabel("CDF")
	axes[0].set_title("Patch-distortion histogram")
	axes[1].set_title("Patch-distortion CDF")
	axes[1].legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(figure_dir / "patch_mse_histogram_cdf.png", dpi=220)
	plt.close(fig)

	fig, axis = plt.subplots(figsize=(6.2, 4.2))
	for name, model_rows in by_model.items():
		x = np.sort([row["worst_global_ratio"] for row in model_rows])
		y = np.arange(1, len(x) + 1) / len(x)
		axis.plot(x, y, linewidth=1.8, label=name)
	axis.set_xlabel("Top-25% patch MSE / global MSE")
	axis.set_ylabel("CDF across images")
	axis.grid(alpha=0.25)
	axis.legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(figure_dir / "worst_global_ratio_cdf.png", dpi=220)
	plt.close(fig)


def plot_example(figure_dir, images, encoded_by_model, patch_scores_by_model, patch_size):
	baseline = "Crop-trained Global"
	ratios = patch_scores_by_model[baseline].max(axis=1) / (
		(encoded_by_model[baseline] - images).pow(2).mean(dim=(1, 2, 3)).numpy() + 1e-12
	)
	index = int(np.argmax(ratios))
	fig, axes = plt.subplots(len(encoded_by_model), 3, figsize=(9, 3 * len(encoded_by_model)))
	cover = images[index].permute(1, 2, 0).numpy() * 0.5 + 0.5
	residuals = {
		name: (encoded[index] - images[index]).pow(2).mean(dim=0).numpy()
		for name, encoded in encoded_by_model.items()
	}
	shared_vmax = float(np.percentile(np.concatenate([r.ravel() for r in residuals.values()]), 99.5))
	for row_index, (name, encoded) in enumerate(encoded_by_model.items()):
		watermarked = encoded[index].permute(1, 2, 0).numpy() * 0.5 + 0.5
		residual = residuals[name]
		worst_index = int(np.argmax(patch_scores_by_model[name][index]))
		patches_per_row = images.shape[-1] // patch_size
		patch_row, patch_col = divmod(worst_index, patches_per_row)
		axes[row_index, 0].imshow(np.clip(watermarked, 0, 1))
		axes[row_index, 1].imshow(residual, cmap="magma", vmin=0, vmax=shared_vmax)
		axes[row_index, 2].imshow(np.clip(watermarked, 0, 1))
		axes[row_index, 2].add_patch(
			plt.Rectangle(
				(patch_col * patch_size, patch_row * patch_size),
				patch_size,
				patch_size,
				fill=False,
				edgecolor="cyan",
				linewidth=2,
			)
		)
		axes[row_index, 0].set_ylabel(name)
		for axis in axes[row_index]:
			axis.set_xticks([])
			axis.set_yticks([])
	axes[0, 0].set_title("Watermarked")
	axes[0, 1].set_title("Squared residual heatmap")
	axes[0, 2].set_title("Maximum-MSE patch")
	fig.suptitle("Aligned test image {} (cover shared across models)".format(index), y=0.995)
	fig.tight_layout()
	fig.savefig(figure_dir / "residual_tail_example.png", dpi=220)
	plt.close(fig)
	return index, cover


def paired_test(by_model, metric, left, right):
	from scipy.stats import ttest_rel

	left_values = np.asarray([row[metric] for row in by_model[left]])
	right_values = np.asarray([row[metric] for row in by_model[right]])
	result = ttest_rel(left_values, right_values)
	return float(np.mean(left_values - right_values)), float(result.pvalue)


def write_report(path, summaries, by_model, figure_dir, example_index, lpips_enabled):
	crop_vs_nocrop_ratio, crop_vs_nocrop_p = paired_test(
		by_model, "worst_global_ratio", "Crop-trained Global", "No-crop Global"
	)
	local_vs_crop_ratio, local_vs_crop_p = paired_test(
		by_model, "worst_global_ratio", "Patch16 top25 weight50", "Crop-trained Global"
	)
	lines = [
		"# Patch distortion tail analysis",
		"",
		"Protocol: fixed test manifest, seed17-aligned images/messages, 32×32 non-overlapping patches, 50 images. Local SSIM and LPIPS are evaluation-only metrics.",
		"",
		"## Tail statistics",
		"",
		"| Model | Global MSE | Top25 MSE | Max MSE | Top25/global | Max/global | Patch CV | Top25 local SSIM | Worst local SSIM | Top25 local LPIPS | Worst local LPIPS |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
	]
	for summary in summaries:
		lines.append(
			"| {} | {:.6f} | {:.6f} | {:.6f} | {:.3f} | {:.3f} | {:.3f} | {:.4f} | {:.4f} | {:.5f} | {:.5f} |".format(
				summary["model"],
				summary["global_mse_mean"],
				summary["top25_patch_mse_mean"],
				summary["max_patch_mse_mean"],
				summary["worst_global_ratio_mean"],
				summary["max_global_ratio_mean"],
				summary["patch_mse_cv_mean"],
				summary["local_ssim_top25_mean"],
				summary["local_ssim_worst_mean"],
				summary["local_lpips_top25_mean"],
				summary["local_lpips_worst_mean"],
			)
		)
	lines += [
		"",
		"## Claim check",
		"",
		"- Crop-trained Global minus No-crop Global mean top25/global ratio: {:+.4f}; paired t-test p={:.4g}.".format(crop_vs_nocrop_ratio, crop_vs_nocrop_p),
		"- Patch16 candidate minus Crop-trained Global mean top25/global ratio: {:+.4f}; paired t-test p={:.4g}.".format(local_vs_crop_ratio, local_vs_crop_p),
	]
	if crop_vs_nocrop_ratio > 0 and crop_vs_nocrop_p < 0.05:
		lines.append("- The concentration statistic supports claim A for this seed/test set, but cross-seed confirmation is still required.")
	else:
		lines.append("- The concentration statistic does not support claim A. Use claim B: crop-robust watermarking degrades visual quality, while a global average objective does not explicitly control the local-distortion tail.")
	lines += [
		"",
		"## Independent local perceptual evaluation",
		"",
		"LPIPS status: {}. These metrics are not used for training and therefore provide an independent check against optimizing only worst-patch MSE.".format("enabled (AlexNet)" if lpips_enabled else "skipped"),
		"",
		"## Figures",
		"",
		"- `{}`".format(figure_dir / "patch_mse_histogram_cdf.png"),
		"- `{}`".format(figure_dir / "worst_global_ratio_cdf.png"),
		"- `{}` (aligned image index {})".format(figure_dir / "residual_tail_example.png", example_index),
	]
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n")


def main():
	args = parse_args()
	device = torch.device(args.device)
	manifest = torch_load(args.manifest, "cpu")
	images = manifest["images"].float()
	messages = manifest["messages"].float()
	all_rows = []
	patch_scores_by_model = OrderedDict()
	encoded_by_model = OrderedDict()
	lpips_enabled = not args.skip_lpips
	for name, checkpoint_path in DEFAULT_MODELS.items():
		print("encoding", name, checkpoint_path, flush=True)
		encoded = load_encoded(
			checkpoint_path, images, messages, device, args.batch_size
		)
		encoded_by_model[name] = encoded
		encoded_patches = extract_patches(encoded, args.patch_size)
		image_patches = extract_patches(images, args.patch_size)
		lpips_values = None
		if lpips_enabled:
			lpips_values = lpips_per_patch(
				encoded_patches, image_patches, device, args.batch_size * 4
			)
		rows, patch_scores = calculate_model_metrics(
			name, images, encoded, args.patch_size, lpips_values
		)
		all_rows.extend(rows)
		patch_scores_by_model[name] = patch_scores
	write_csv(args.per_image_csv, all_rows)
	summaries, by_model = summarize(all_rows, patch_scores_by_model)
	write_csv(args.tail_csv, summaries)
	plot_distributions(args.figure_dir, patch_scores_by_model, by_model)
	example_index, _ = plot_example(
		args.figure_dir, images, encoded_by_model, patch_scores_by_model, args.patch_size
	)
	write_report(
		args.report,
		summaries,
		by_model,
		args.figure_dir,
		example_index,
		lpips_enabled,
	)
	print("saved", args.report)
	print("saved", args.per_image_csv)
	print("saved", args.tail_csv)


if __name__ == "__main__":
	main()
