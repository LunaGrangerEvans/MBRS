#!/usr/bin/env python3
"""Evaluate controlled seed17 continuation branches on one fixed manifest."""

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.analyze_patch_distortion import extract_patches, ssim_per_sample
from experiments.losses import patch_mse_per_sample
from network.Encoder_MP_Decoder import EncoderDecoder


ROOT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
BRANCHES = OrderedDict(
	[
		("Global continuation", "controlled_seed17_global_continuation"),
		("Hard P16-T25-L50", "controlled_seed17_hard_p16_t25_l50"),
		("Soft P16-T0.5-L50", "controlled_seed17_soft_p16_t05_l50"),
		(
			"Hard patch16 stride8 top25 global0.5 local0.5",
			"controlled_seed17_hard_patch16_stride8_top25_weight50",
		),
		(
			"Multiscale patch16 top25 weight0.7 plus patch32 top25 weight0.3 global0.5 local0.5",
			"controlled_seed17_multiscale_patch16_patch32_top25_weight70_weight30_global_weight50_local_weight50",
		),
		(
			"Excess patch16 stride16 threshold1 scale3 global0.5 local0.5",
			"controlled_seed17_excess_patch16_stride16_global_weight50_local_weight50",
		),
		(
			"Hard patch16 stride8 top10 global0.5 local0.5",
			"controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50",
		),
		(
			"Gradient-aware patch16 stride8 top10 alpha2 global0.5 local0.5",
			"seed17_contentaware_gradient_patch16_stride8_top10_alpha2_global0.5_local0.5",
		),
	]
)
ATTACK_AREAS = OrderedDict(
	[
		("crop_30", 0.30),
		("crop_35", 0.35),
		("crop_40", 0.40),
		("crop_50", 0.50),
		("crop_70", 0.70),
		("crop_100", 1.00),
	]
)
LOWER_IS_BETTER = {
	"lpips",
	"local_lpips_top25",
	"local_lpips_worst",
	"top10_patch_mse",
	"top25_patch_mse",
	"patch_mse_p90",
	"patch_mse_p95",
	"patch_mse_p99",
	"max_patch_mse",
	"tail_global_ratio",
	"ber30",
	"ber35",
	"ber40",
	"ber50",
	"ber70",
	"ber100",
}


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--patch-size", type=int, default=32)
	parser.add_argument("--repeats", type=int, default=5)
	parser.add_argument(
		"--manifest", type=Path, default=ROOT / "reports/uniform_eval_manifest.pt"
	)
	parser.add_argument(
		"--attack-extension",
		type=Path,
		default=ROOT / "reports/controlled_crop35_40_manifest.pt",
	)
	parser.add_argument(
		"--output-json",
		type=Path,
		default=ROOT / "reports/controlled_seed17_metrics.json",
	)
	parser.add_argument(
		"--output-csv", type=Path, default=Path("reports/controlled_seed17_summary.csv")
	)
	parser.add_argument(
		"--output-markdown",
		type=Path,
		default=Path("reports/controlled_seed17_summary.md"),
	)
	parser.add_argument(
		"--conclusion",
		type=Path,
		default=Path("reports/controlled_hard_vs_soft_conclusion.md"),
	)
	parser.add_argument(
		"--figure-dir",
		type=Path,
		default=ROOT / "visualizations/controlled_seed17",
	)
	return parser.parse_args()


def torch_load(path, device):
	try:
		return torch.load(str(path), map_location=device, weights_only=False)
	except TypeError:
		return torch.load(str(path), map_location=device)


def make_masks(area, manifest, repeats, batch_size, generator):
	height = int(math.sqrt(area) * manifest["H"])
	width = int(math.sqrt(area) * manifest["W"])
	batch_count = math.ceil(manifest["samples"] / batch_size)
	all_masks = []
	for _ in range(repeats):
		batch_masks = []
		for _ in range(batch_count):
			h_start = (
				0
				if height == manifest["H"]
				else int(generator.randint(0, manifest["H"] - height + 1))
			)
			w_start = (
				0
				if width == manifest["W"]
				else int(generator.randint(0, manifest["W"] - width + 1))
			)
			mask = torch.zeros(1, 1, manifest["H"], manifest["W"])
			mask[:, :, h_start : h_start + height, w_start : w_start + width] = 1
			batch_masks.append(mask)
		all_masks.append(batch_masks)
	return all_masks


def load_attack_masks(base_manifest, extension_path, repeats, batch_size):
	if extension_path.is_file():
		extension = torch_load(extension_path, "cpu")
		if (
			extension["base_manifest_seed"] != base_manifest["seed"]
			or extension["samples"] != base_manifest["samples"]
			or extension["repeats"] != repeats
			or extension["batch_size"] != batch_size
		):
			raise ValueError("controlled crop35/40 extension is incompatible")
	else:
		generator = np.random.RandomState(int(base_manifest["seed"]) + 240011)
		extension = {
			"base_manifest_seed": base_manifest["seed"],
			"samples": base_manifest["samples"],
			"repeats": repeats,
			"batch_size": batch_size,
			"attack_masks": {
				"crop_35": make_masks(0.35, base_manifest, repeats, batch_size, generator),
				"crop_40": make_masks(0.40, base_manifest, repeats, batch_size, generator),
			},
		}
		extension_path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(extension, extension_path)
	masks = {
		"crop_30": base_manifest["attack_masks"]["crop_30"],
		"crop_35": extension["attack_masks"]["crop_35"],
		"crop_40": extension["attack_masks"]["crop_40"],
		"crop_50": base_manifest["attack_masks"]["crop_50"],
		"crop_70": base_manifest["attack_masks"]["crop_70"],
		"crop_100": base_manifest["attack_masks"]["crop_100"],
	}
	return masks


def load_model(checkpoint_path, device):
	checkpoint = torch_load(checkpoint_path, device)
	config = checkpoint["config"]
	model = EncoderDecoder(
		config["H"], config["W"], config["message_length"], ["Identity()"]
	).to(device)
	model.load_state_dict(checkpoint["model"])
	model.eval()
	return model


def metric_batches(metric, left, right, device, batch_size):
	values = []
	with torch.no_grad():
		for start in range(0, left.shape[0], batch_size):
			stop = start + batch_size
			values.append(
				metric(left[start:stop].to(device), right[start:stop].to(device))
				.flatten()
				.cpu()
			)
	return torch.cat(values)


def training_diagnostics(run_dir):
	train_rows = [json.loads(line) for line in (run_dir / "train.jsonl").open() if line.strip()]
	val_rows = [json.loads(line) for line in (run_dir / "val.jsonl").open() if line.strip()]
	if len(train_rows) != 20 or len(val_rows) != 20:
		raise ValueError("controlled run must contain exactly 20 train/val epochs: {}".format(run_dir))
	psnr_steps = np.diff([row["psnr"] for row in val_rows])
	ber_steps = np.diff([row["ber"] for row in val_rows])
	last_train = train_rows[-1]
	return {
		"final_message_loss": last_train["message_loss"],
		"final_global_loss": last_train["global_mse"],
		"final_local_loss": last_train["local_mse"],
		"final_weighted_image_loss": last_train["weighted_image_loss"],
		"final_total_loss": last_train["loss"],
		"val_psnr_step_std": float(np.std(psnr_steps, ddof=1)),
		"val_ber_step_std": float(np.std(ber_steps, ddof=1)),
		"soft_weight_mean": last_train.get("soft_weight_mean", 0.0),
		"soft_weight_max": last_train.get("soft_weight_max", 0.0),
		"soft_effective_patches": last_train.get("soft_effective_patches", 0.0),
	}


def evaluate_branch(
	model,
	images,
	messages,
	attack_masks,
	lpips_metric,
	device,
	batch_size,
	patch_size,
	repeats,
):
	encoded_batches = []
	with torch.no_grad():
		for start in range(0, images.shape[0], batch_size):
			stop = start + batch_size
			encoded_batches.append(
				model.encoder(
					images[start:stop].to(device), messages[start:stop].to(device)
				).cpu()
			)
	encoded = torch.cat(encoded_batches)
	global_mse_per_image = (encoded - images).pow(2).mean(dim=(1, 2, 3))
	patch_scores = patch_mse_per_sample(encoded, images, patch_size).cpu()
	patch_count = patch_scores.shape[1]
	top10_count = max(1, math.ceil(patch_count * 0.10))
	top25_count = max(1, math.ceil(patch_count * 0.25))
	top10 = torch.topk(patch_scores, top10_count, dim=1).values.mean(dim=1)
	top25 = torch.topk(patch_scores, top25_count, dim=1).values.mean(dim=1)
	max_patch = patch_scores.max(dim=1).values
	ssim = ssim_per_sample(encoded.to(device), images.to(device)).cpu()
	lpips_values = metric_batches(
		lpips_metric, encoded, images, device, batch_size
	)
	encoded_patches = extract_patches(encoded, patch_size)
	image_patches = extract_patches(images, patch_size)
	local_ssim = ssim_per_sample(
		encoded_patches.to(device), image_patches.to(device)
	).cpu().reshape(-1, patch_count)
	local_lpips = metric_batches(
		lpips_metric, encoded_patches, image_patches, device, batch_size * 4
	).reshape(-1, patch_count)
	local_ssim_top25 = torch.topk(
		local_ssim, top25_count, dim=1, largest=False
	).values.mean(dim=1)
	local_ssim_worst = local_ssim.min(dim=1).values
	local_lpips_top25 = torch.topk(
		local_lpips, top25_count, dim=1
	).values.mean(dim=1)
	local_lpips_worst = local_lpips.max(dim=1).values

	attack_correct = {name: 0 for name in ATTACK_AREAS}
	attack_bits = {name: 0 for name in ATTACK_AREAS}
	with torch.no_grad():
		for batch_index, start in enumerate(range(0, images.shape[0], batch_size)):
			stop = start + batch_size
			batch_encoded = encoded[start:stop].to(device)
			batch_messages = messages[start:stop].to(device)
			for attack_name in ATTACK_AREAS:
				for repeat in range(repeats):
					mask = attack_masks[attack_name][repeat][batch_index].to(device)
					decoded = model.decoder(batch_encoded * mask)
					predicted = decoded.gt(0.5)
					target = batch_messages.gt(0.5)
					attack_correct[attack_name] += int((predicted == target).sum())
					attack_bits[attack_name] += target.numel()

	flat_patch_scores = patch_scores.numpy().reshape(-1)
	global_mse = float(global_mse_per_image.mean())
	top25_mse = float(top25.mean())
	metrics = {
		"psnr": 10 * math.log10(4.0 / global_mse),
		"ssim": float(ssim.mean()),
		"lpips": float(lpips_values.mean()),
		"worst_psnr": 10 * math.log10(4.0 / top25_mse),
		"local_ssim_top25": float(local_ssim_top25.mean()),
		"local_ssim_worst": float(local_ssim_worst.mean()),
		"local_lpips_top25": float(local_lpips_top25.mean()),
		"local_lpips_worst": float(local_lpips_worst.mean()),
		"top10_patch_mse": float(top10.mean()),
		"top25_patch_mse": top25_mse,
		"patch_mse_p90": float(np.percentile(flat_patch_scores, 90)),
		"patch_mse_p95": float(np.percentile(flat_patch_scores, 95)),
		"patch_mse_p99": float(np.percentile(flat_patch_scores, 99)),
		"max_patch_mse": float(max_patch.mean()),
		"tail_global_ratio": float((top25 / global_mse_per_image).mean()),
	}
	for attack_name in ATTACK_AREAS:
		metrics["ber" + attack_name.split("_")[-1]] = 1 - (
			attack_correct[attack_name] / attack_bits[attack_name]
		)
	return metrics, encoded, patch_scores.numpy()


def deltas(metrics_by_branch):
	control = metrics_by_branch["Global continuation"]
	return {
		name: {
			key: value - control[key]
			for key, value in metrics.items()
			if isinstance(value, (float, int)) and key in control
		}
		for name, metrics in metrics_by_branch.items()
	}


def soft_go_no_go(
	metrics_by_branch, branch_deltas, soft_name="Soft P16-T0.5-L50"
):
	control = metrics_by_branch["Global continuation"]
	hard = metrics_by_branch["Hard P16-T25-L50"]
	soft = metrics_by_branch[soft_name]
	soft_delta = branch_deltas[soft_name]
	checks = OrderedDict(
		[
			("worst/top25 PSNR improves", soft_delta["worst_psnr"] > 0),
			(
				"top25 and P95 MSE decrease",
				soft["top25_patch_mse"] < control["top25_patch_mse"]
				and soft["patch_mse_p95"] < control["patch_mse_p95"],
			),
			("global PSNR within -0.2 dB", soft_delta["psnr"] >= -0.2),
			("BER30 within +0.005", soft_delta["ber30"] <= 0.005),
			(
				"local SSIM or LPIPS improves",
				soft_delta["local_ssim_top25"] > 0
				or soft_delta["local_lpips_top25"] < 0,
			),
			(
				"Soft oscillation no worse than Hard",
				soft["val_psnr_step_std"] <= 1.25 * hard["val_psnr_step_std"]
				and soft["val_ber_step_std"] <= 1.25 * hard["val_ber_step_std"],
			),
		]
	)
	return checks, all(checks.values())


def plot_figures(figure_dir, images, encoded_by_branch, patch_scores_by_branch, metrics_by_branch, patch_size):
	figure_dir.mkdir(parents=True, exist_ok=True)
	control_name = "Global continuation"
	control_global = (
		(encoded_by_branch[control_name] - images).pow(2).mean(dim=(1, 2, 3)).numpy()
	)
	control_ratios = patch_scores_by_branch[control_name].max(axis=1) / (control_global + 1e-12)
	image_index = int(np.argmax(control_ratios))
	residuals = {
		name: (encoded[image_index] - images[image_index]).pow(2).mean(dim=0).numpy()
		for name, encoded in encoded_by_branch.items()
	}
	shared_vmax = float(
		np.percentile(np.concatenate([value.ravel() for value in residuals.values()]), 99.5)
	)
	branch_count = len(encoded_by_branch)
	fig, axes = plt.subplots(
		branch_count,
		4,
		figsize=(12, max(9, branch_count * 3)),
		squeeze=False,
	)
	original = images[image_index].permute(1, 2, 0).numpy() * 0.5 + 0.5
	for row, (name, encoded) in enumerate(encoded_by_branch.items()):
		watermarked = encoded[image_index].permute(1, 2, 0).numpy() * 0.5 + 0.5
		worst_index = int(np.argmax(patch_scores_by_branch[name][image_index]))
		patches_per_row = images.shape[-1] // patch_size
		patch_row, patch_col = divmod(worst_index, patches_per_row)
		axes[row, 0].imshow(np.clip(original, 0, 1))
		axes[row, 1].imshow(np.clip(watermarked, 0, 1))
		axes[row, 2].imshow(residuals[name], cmap="magma", vmin=0, vmax=shared_vmax)
		axes[row, 3].imshow(np.clip(watermarked, 0, 1))
		axes[row, 3].add_patch(
			plt.Rectangle(
				(patch_col * patch_size, patch_row * patch_size),
				patch_size,
				patch_size,
				fill=False,
				edgecolor="cyan",
				linewidth=2,
			)
		)
		axes[row, 0].set_ylabel(name)
		for axis in axes[row]:
			axis.set_xticks([])
			axis.set_yticks([])
	for axis, title in zip(
		axes[0], ["Original", "Watermarked", "Residual (shared scale)", "Max-MSE patch"]
	):
		axis.set_title(title)
	fig.suptitle("Controlled seed17, aligned test image {}".format(image_index))
	fig.tight_layout()
	fig.savefig(figure_dir / "aligned_watermark_residual_worst_patch.png", dpi=220)
	plt.close(fig)

	fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
	for name, scores in patch_scores_by_branch.items():
		flat = scores.reshape(-1)
		axes[0].hist(flat, bins=60, density=True, histtype="step", linewidth=1.8, label=name)
		x = np.sort(flat)
		axes[1].plot(x, np.arange(1, len(x) + 1) / len(x), linewidth=1.8, label=name)
	for axis in axes:
		axis.set_xscale("log")
		axis.set_xlabel("32x32 patch MSE")
		axis.grid(alpha=0.25)
	axes[0].set_ylabel("Density")
	axes[1].set_ylabel("CDF")
	axes[0].set_title("Patch MSE histogram")
	axes[1].set_title("Patch MSE CDF")
	axes[1].legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(figure_dir / "patch_mse_histogram_cdf.png", dpi=220)
	plt.close(fig)

	fig, axis = plt.subplots(figsize=(6.4, 4.4))
	percentiles = np.linspace(50, 100, 101)
	for name, scores in patch_scores_by_branch.items():
		axis.plot(percentiles, np.percentile(scores.reshape(-1), percentiles), label=name)
	axis.set_xlabel("Patch-MSE percentile")
	axis.set_ylabel("Patch MSE")
	axis.grid(alpha=0.25)
	axis.legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(figure_dir / "patch_mse_percentile_curve.png", dpi=220)
	plt.close(fig)

	fig, axis = plt.subplots(figsize=(6.4, 4.4))
	for name, scores in patch_scores_by_branch.items():
		global_values = (
			(encoded_by_branch[name] - images).pow(2).mean(dim=(1, 2, 3)).numpy()
		)
		top25_count = max(1, math.ceil(scores.shape[1] * 0.25))
		top25_values = np.sort(scores, axis=1)[:, -top25_count:].mean(axis=1)
		ratios = np.sort(top25_values / (global_values + 1e-12))
		axis.plot(ratios, np.arange(1, len(ratios) + 1) / len(ratios), label=name)
	axis.set_xlabel("Top25 patch MSE / global MSE")
	axis.set_ylabel("CDF across images")
	axis.grid(alpha=0.25)
	axis.legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(figure_dir / "tail_global_ratio_cdf.png", dpi=220)
	plt.close(fig)
	return image_index


def write_csv(path, metrics_by_branch, branch_deltas):
	path.parent.mkdir(parents=True, exist_ok=True)
	metric_names = list(next(iter(metrics_by_branch.values())))
	fields = ["branch", *metric_names, *("delta_" + key for key in metric_names)]
	with path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fields)
		writer.writeheader()
		for name, metrics in metrics_by_branch.items():
			row = {"branch": name, **metrics}
			row.update({"delta_" + key: value for key, value in branch_deltas[name].items()})
			writer.writerow(row)


def signed(value, digits=4):
	return ("{:+.%df}" % digits).format(value)


def write_markdown(path, metrics_by_branch, branch_deltas, checks, go, figure_dir, image_index):
	lines = [
		"# Controlled seed17 continuation summary",
		"",
		"Primary comparison is against the equal-length Global continuation control. All branches use the same seed17 epoch-100 source checkpoint, restored Adam/BatchNorm state, 20 continuation epochs, LR 1e-4, one GPU, and fixed evaluation masks.",
		"",
		"## Quality and robustness",
		"",
		"| Branch | PSNR | Δ | Worst/top25 PSNR | Δ | SSIM | Δ | LPIPS | Δ | BER30 | Δ |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
	]
	for name, metrics in metrics_by_branch.items():
		delta = branch_deltas[name]
		lines.append(
			"| {} | {:.3f} | {} | {:.3f} | {} | {:.4f} | {} | {:.5f} | {} | {:.5f} | {} |".format(
				name,
				metrics["psnr"], signed(delta["psnr"], 3),
				metrics["worst_psnr"], signed(delta["worst_psnr"], 3),
				metrics["ssim"], signed(delta["ssim"], 4),
				metrics["lpips"], signed(delta["lpips"], 5),
				metrics["ber30"], signed(delta["ber30"], 5),
			)
		)
	lines += [
		"",
		"## Local tail",
		"",
		"| Branch | Top10 MSE | Top25 MSE | P90 | P95 | P99 | Mean max MSE | Tail/global | Local SSIM top25 | Local LPIPS top25 |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
	]
	for name, metrics in metrics_by_branch.items():
		lines.append(
			"| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.3f} | {:.4f} | {:.5f} |".format(
				name,
				metrics["top10_patch_mse"], metrics["top25_patch_mse"],
				metrics["patch_mse_p90"], metrics["patch_mse_p95"],
				metrics["patch_mse_p99"], metrics["max_patch_mse"],
				metrics["tail_global_ratio"], metrics["local_ssim_top25"],
				metrics["local_lpips_top25"],
			)
		)
	lines += [
		"",
		"## BER curve",
		"",
		"| Branch | BER30 | BER35 | BER40 | BER50 | BER70 | BER100 |",
		"|---|---:|---:|---:|---:|---:|---:|",
	]
	for name, metrics in metrics_by_branch.items():
		lines.append(
			"| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |".format(
				name, *(metrics["ber" + suffix] for suffix in ("30", "35", "40", "50", "70", "100"))
			)
		)
	lines += [
		"",
		"## Training behavior",
		"",
		"| Branch | Final message raw | Global raw | Local raw | Weighted image | Total | PSNR step std | BER step std | Soft max weight | Soft N_eff |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
	]
	for name, metrics in metrics_by_branch.items():
		lines.append(
			"| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.4f} | {:.5f} | {:.5f} | {:.2f} |".format(
				name,
				metrics["final_message_loss"], metrics["final_global_loss"],
				metrics["final_local_loss"], metrics["final_weighted_image_loss"],
				metrics["final_total_loss"], metrics["val_psnr_step_std"],
				metrics["val_ber_step_std"], metrics["soft_weight_max"],
				metrics["soft_effective_patches"],
			)
		)
	lines += ["", "## Soft T=0.5 go/no-go", ""]
	for label, passed in checks.items():
		lines.append("- {}: {}".format(label, "PASS" if passed else "FAIL"))
	lines.append("")
	lines.append("Decision: **{}** for temperature screening.".format("GO" if go else "NO-GO"))
	lines += [
		"",
		"## Figures",
		"",
		"All residual panels use a shared scale. Aligned example index: {}.".format(image_index),
		"",
		"- `{}`".format(figure_dir / "aligned_watermark_residual_worst_patch.png"),
		"- `{}`".format(figure_dir / "patch_mse_histogram_cdf.png"),
		"- `{}`".format(figure_dir / "patch_mse_percentile_curve.png"),
		"- `{}`".format(figure_dir / "tail_global_ratio_cdf.png"),
	]
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n")


def write_seed17_conclusion(path, metrics, branch_deltas, checks, go):
	control = metrics["Global continuation"]
	hard = metrics["Hard P16-T25-L50"]
	soft = metrics["Soft P16-T0.5-L50"]
	hard_delta = branch_deltas["Hard P16-T25-L50"]
	soft_delta = branch_deltas["Soft P16-T0.5-L50"]
	lines = [
		"# Controlled hard-vs-soft conclusion",
		"",
		"Status: seed17 gate complete; multi-seed conclusion {}.".format(
			"pending temperature screening" if go else "stopped by no-go"
		),
		"",
		"1. Global continuation is the control at PSNR {:.3f}, WorstPSNR {:.3f}, BER30 {:.5f}; source-checkpoint gain is deferred to the final multi-seed evaluator.".format(
			control["psnr"], control["worst_psnr"], control["ber30"]
		),
		"2. Hard versus control: ΔPSNR {:+.3f}, ΔWorstPSNR {:+.3f}, ΔBER30 {:+.5f}.".format(
			hard_delta["psnr"], hard_delta["worst_psnr"], hard_delta["ber30"]
		),
		"3. Soft training oscillation versus Hard: PSNR-step std {:.4f} vs {:.4f}; BER-step std {:.5f} vs {:.5f}.".format(
			soft["val_psnr_step_std"], hard["val_psnr_step_std"],
			soft["val_ber_step_std"], hard["val_ber_step_std"],
		),
		"4. Soft right-tail deltas: Top25MSE {:+.6f}, P95MSE {:+.6f}.".format(
			soft_delta["top25_patch_mse"], soft_delta["patch_mse_p95"]
		),
		"5. Soft local-versus-global gains: ΔWorstPSNR {:+.3f} versus ΔPSNR {:+.3f}.".format(
			soft_delta["worst_psnr"], soft_delta["psnr"]
		),
		"6. Soft BER30 delta: {:+.5f}.".format(soft_delta["ber30"]),
		"7. Independent local perceptual deltas: local SSIM {:+.4f}, local LPIPS {:+.5f}.".format(
			soft_delta["local_ssim_top25"], soft_delta["local_lpips_top25"]
		),
		"8. Evidence is not paper-ready until a final temperature is selected and seed29/41 paired results exist.",
		"9. Warm-up remains out of scope for this round.",
		"10. Current claim remains: global average objectives do not explicitly control the high-distortion local tail.",
		"",
		"Soft T=0.5 decision: **{}**. Checks: {}.".format(
			"GO" if go else "NO-GO",
			", ".join("{}={}".format(name, passed) for name, passed in checks.items()),
		),
	]
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n")


def main():
	args = parse_args()
	if args.device == "cuda" and torch.cuda.device_count() != 1:
		raise RuntimeError("controlled evaluation requires exactly one visible GPU")
	device = torch.device(args.device)
	manifest = torch_load(args.manifest, "cpu")
	if manifest["repeats"] != args.repeats:
		raise ValueError("repeat count must match the fixed manifest")
	images = manifest["images"].float()
	messages = manifest["messages"].float()
	attack_masks = load_attack_masks(
		manifest, args.attack_extension, args.repeats, args.batch_size
	)
	import lpips

	lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
	metrics_by_branch = OrderedDict()
	encoded_by_branch = OrderedDict()
	patch_scores_by_branch = OrderedDict()
	for name, run_name in BRANCHES.items():
		run_dir = ROOT / "experiments/runs" / run_name
		checkpoint_path = run_dir / "checkpoint_0020.pth"
		if not checkpoint_path.is_file():
			raise FileNotFoundError(checkpoint_path)
		print("evaluating", name, checkpoint_path, flush=True)
		model = load_model(checkpoint_path, device)
		metrics, encoded, patch_scores = evaluate_branch(
			model, images, messages, attack_masks, lpips_metric, device,
			args.batch_size, args.patch_size, args.repeats,
		)
		metrics.update(training_diagnostics(run_dir))
		metrics_by_branch[name] = metrics
		encoded_by_branch[name] = encoded
		patch_scores_by_branch[name] = patch_scores
	branch_deltas = deltas(metrics_by_branch)
	checks, go = soft_go_no_go(metrics_by_branch, branch_deltas)
	image_index = plot_figures(
		args.figure_dir, images, encoded_by_branch, patch_scores_by_branch,
		metrics_by_branch, args.patch_size,
	)
	write_csv(args.output_csv, metrics_by_branch, branch_deltas)
	write_markdown(
		args.output_markdown, metrics_by_branch, branch_deltas, checks, go,
		args.figure_dir, image_index,
	)
	write_seed17_conclusion(
		args.conclusion, metrics_by_branch, branch_deltas, checks, go
	)
	args.output_json.parent.mkdir(parents=True, exist_ok=True)
	args.output_json.write_text(
		json.dumps(
			{
				"metrics": metrics_by_branch,
				"deltas_vs_global_continuation": branch_deltas,
				"soft_go_no_go_checks": checks,
				"soft_temperature_screening_go": go,
				"figure_dir": str(args.figure_dir),
			},
			indent=2,
		)
		+ "\n"
	)
	print("soft_temperature_screening_go={}".format(go))
	print("saved", args.output_markdown)
	print("saved", args.output_csv)
	print("saved", args.output_json)


if __name__ == "__main__":
	main()
