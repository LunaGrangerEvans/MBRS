#!/usr/bin/env python3
"""Select at most one soft-tail temperature after the seed17 gate."""

import argparse
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

from experiments.evaluate_controlled_seed17 import (
	ROOT,
	deltas,
	evaluate_branch,
	load_attack_masks,
	load_model,
	soft_go_no_go,
	torch_load,
	training_diagnostics,
	write_csv,
)


BRANCHES = OrderedDict(
	[
		("Global continuation", "controlled_seed17_global_continuation"),
		("Hard P16-T25-L50", "controlled_seed17_hard_p16_t25_l50"),
		("Soft P16-T0.25-L50", "controlled_seed17_soft_p16_t025_l50"),
		("Soft P16-T0.5-L50", "controlled_seed17_soft_p16_t05_l50"),
		("Soft P16-T1.0-L50", "controlled_seed17_soft_p16_t10_l50"),
	]
)
SOFT_TEMPERATURES = OrderedDict(
	[
		("Soft P16-T0.25-L50", 0.25),
		("Soft P16-T0.5-L50", 0.5),
		("Soft P16-T1.0-L50", 1.0),
	]
)


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--patch-size", type=int, default=32)
	parser.add_argument("--repeats", type=int, default=5)
	parser.add_argument(
		"--output-json",
		type=Path,
		default=ROOT / "reports/controlled_soft_temperature_seed17.json",
	)
	parser.add_argument(
		"--output-csv",
		type=Path,
		default=Path("reports/controlled_soft_temperature_seed17.csv"),
	)
	parser.add_argument(
		"--output-markdown",
		type=Path,
		default=Path("reports/controlled_soft_temperature_seed17.md"),
	)
	parser.add_argument(
		"--figure-dir",
		type=Path,
		default=ROOT / "visualizations/controlled_soft_temperature_seed17",
	)
	return parser.parse_args()


def make_plots(figure_dir, patch_scores, encoded, images):
	figure_dir.mkdir(parents=True, exist_ok=True)
	fig, axis = plt.subplots(figsize=(6.6, 4.5))
	percentiles = np.linspace(50, 100, 101)
	for name, scores in patch_scores.items():
		axis.plot(percentiles, np.percentile(scores.reshape(-1), percentiles), label=name)
	axis.set_xlabel("Patch-MSE percentile")
	axis.set_ylabel("Patch MSE")
	axis.grid(alpha=0.25)
	axis.legend(fontsize=7)
	fig.tight_layout()
	fig.savefig(figure_dir / "temperature_percentile_curve.png", dpi=220)
	plt.close(fig)

	fig, axis = plt.subplots(figsize=(6.6, 4.5))
	for name, scores in patch_scores.items():
		global_values = (encoded[name] - images).pow(2).mean(dim=(1, 2, 3)).numpy()
		count = max(1, math.ceil(scores.shape[1] * 0.25))
		top25 = np.sort(scores, axis=1)[:, -count:].mean(axis=1)
		ratios = np.sort(top25 / (global_values + 1e-12))
		axis.plot(ratios, np.arange(1, len(ratios) + 1) / len(ratios), label=name)
	axis.set_xlabel("Top25 patch MSE / global MSE")
	axis.set_ylabel("CDF across images")
	axis.grid(alpha=0.25)
	axis.legend(fontsize=7)
	fig.tight_layout()
	fig.savefig(figure_dir / "temperature_tail_global_cdf.png", dpi=220)
	plt.close(fig)


def write_markdown(path, metrics, branch_deltas, checks_by_soft, selected, figure_dir):
	lines = [
		"# Controlled soft-tail temperature screen: seed17",
		"",
		"All soft branches start from the same Global checkpoint and use the same 20-epoch continuation protocol. Selection is restricted to T=0.25, 0.5, and 1.0.",
		"",
		"| Branch | PSNR | ΔPSNR | Worst PSNR | ΔWorst | Top25 MSE | ΔTop25 | P95 MSE | ΔP95 | BER30 | ΔBER30 | Soft max weight | N_eff | Gate |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
	]
	for name, temperature in SOFT_TEMPERATURES.items():
		value = metrics[name]
		delta = branch_deltas[name]
		passed = all(checks_by_soft[name].values())
		lines.append(
			"| T={:.2g} | {:.3f} | {:+.3f} | {:.3f} | {:+.3f} | {:.6f} | {:+.6f} | {:.6f} | {:+.6f} | {:.5f} | {:+.5f} | {:.5f} | {:.2f} | {} |".format(
				temperature,
				value["psnr"], delta["psnr"], value["worst_psnr"],
				delta["worst_psnr"], value["top25_patch_mse"],
				delta["top25_patch_mse"], value["patch_mse_p95"],
				delta["patch_mse_p95"], value["ber30"], delta["ber30"],
				value["soft_weight_max"], value["soft_effective_patches"],
				"PASS" if passed else "FAIL",
			)
		)
	lines += [
		"",
		"Selected final Soft temperature: **{}**.".format(
			"none" if selected is None else selected
		),
		"",
		"Selection rule: among temperatures passing the seed17 quality/BER/oscillation gate, choose the lowest top25 patch MSE. No additional temperatures are considered.",
		"",
		"Figures:",
		"",
		"- `{}`".format(figure_dir / "temperature_percentile_curve.png"),
		"- `{}`".format(figure_dir / "temperature_tail_global_cdf.png"),
	]
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n")


def main():
	args = parse_args()
	if args.device == "cuda" and torch.cuda.device_count() != 1:
		raise RuntimeError("controlled evaluation requires exactly one visible GPU")
	device = torch.device(args.device)
	manifest = torch_load(ROOT / "reports/uniform_eval_manifest.pt", "cpu")
	images = manifest["images"].float()
	messages = manifest["messages"].float()
	attack_masks = load_attack_masks(
		manifest,
		ROOT / "reports/controlled_crop35_40_manifest.pt",
		args.repeats,
		args.batch_size,
	)
	import lpips

	lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
	metrics = OrderedDict()
	encoded = OrderedDict()
	patch_scores = OrderedDict()
	for name, run_name in BRANCHES.items():
		run_dir = ROOT / "experiments/runs" / run_name
		checkpoint = run_dir / "checkpoint_0020.pth"
		model = load_model(checkpoint, device)
		value, branch_encoded, branch_scores = evaluate_branch(
			model, images, messages, attack_masks, lpips_metric, device,
			args.batch_size, args.patch_size, args.repeats,
		)
		value.update(training_diagnostics(run_dir))
		metrics[name] = value
		encoded[name] = branch_encoded
		patch_scores[name] = branch_scores
	branch_deltas = deltas(metrics)
	checks_by_soft = OrderedDict()
	qualifying = []
	for name, temperature in SOFT_TEMPERATURES.items():
		checks, passed = soft_go_no_go(metrics, branch_deltas, soft_name=name)
		checks_by_soft[name] = checks
		if passed:
			qualifying.append((metrics[name]["top25_patch_mse"], temperature, name))
	selected = min(qualifying)[1] if qualifying else None
	make_plots(args.figure_dir, patch_scores, encoded, images)
	write_csv(args.output_csv, metrics, branch_deltas)
	write_markdown(
		args.output_markdown, metrics, branch_deltas, checks_by_soft,
		selected, args.figure_dir,
	)
	args.output_json.parent.mkdir(parents=True, exist_ok=True)
	args.output_json.write_text(
		json.dumps(
			{
				"metrics": metrics,
				"deltas_vs_global_continuation": branch_deltas,
				"checks": checks_by_soft,
				"selected_temperature": selected,
			},
			indent=2,
		)
		+ "\n"
	)
	print("selected_temperature={}".format(selected))
	print("saved", args.output_markdown)


if __name__ == "__main__":
	main()
