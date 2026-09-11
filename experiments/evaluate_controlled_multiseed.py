#!/usr/bin/env python3
"""Evaluate final controlled Global/Hard/Soft branches across three seeds."""

import argparse
import csv
import json
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

import torch

if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate_controlled_seed17 import (
	ROOT,
	evaluate_branch,
	load_attack_masks,
	load_model,
	torch_load,
)


SEEDS = (17, 29, 41)
METRICS = (
	"psnr",
	"worst_psnr",
	"top25_patch_mse",
	"patch_mse_p95",
	"ssim",
	"local_ssim_top25",
	"lpips",
	"local_lpips_top25",
	"ber30",
)
LOWER_IS_BETTER = {
	"top25_patch_mse",
	"patch_mse_p95",
	"lpips",
	"local_lpips_top25",
	"ber30",
}


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
	parser.add_argument("--temperature", type=float, required=True, choices=[0.25, 0.5, 1.0])
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--patch-size", type=int, default=32)
	parser.add_argument("--repeats", type=int, default=5)
	parser.add_argument(
		"--output-json",
		type=Path,
		default=ROOT / "reports/controlled_hard_vs_soft_metrics.json",
	)
	parser.add_argument(
		"--output-csv",
		type=Path,
		default=Path("reports/controlled_hard_vs_soft_paired.csv"),
	)
	parser.add_argument(
		"--output-markdown",
		type=Path,
		default=Path("reports/controlled_hard_vs_soft_conclusion.md"),
	)
	return parser.parse_args()


def temperature_suffix(temperature):
	return {0.25: "t025", 0.5: "t05", 1.0: "t10"}[temperature]


def run_paths(seed, temperature):
	return OrderedDict(
		[
			(
				"Source Global",
				ROOT / "experiments/runs/optimization_global_seed{}_128_m64_crop/checkpoint_0100.pth".format(seed),
			),
			(
				"Global continuation",
				ROOT / "experiments/runs/controlled_seed{}_global_continuation/checkpoint_0020.pth".format(seed),
			),
			(
				"Hard P16-T25-L50",
				ROOT / "experiments/runs/controlled_seed{}_hard_p16_t25_l50/checkpoint_0020.pth".format(seed),
			),
			(
				"Soft P16-T{}-L50".format(temperature),
				ROOT
				/ "experiments/runs/controlled_seed{}_soft_p16_{}_l50/checkpoint_0020.pth".format(
					seed, temperature_suffix(temperature)
				),
			),
		]
	)


def paired_deltas(metrics_by_seed, candidate, reference):
	return {
		metric: [
			metrics_by_seed[seed][candidate][metric]
			- metrics_by_seed[seed][reference][metric]
			for seed in SEEDS
		]
		for metric in METRICS
	}


def summarize(values):
	return statistics.fmean(values), statistics.stdev(values)


def improvement_count(metric, values):
	if metric in LOWER_IS_BETTER:
		return sum(value < 0 for value in values)
	return sum(value > 0 for value in values)


def write_csv(path, source_delta, hard_delta, soft_delta):
	path.parent.mkdir(parents=True, exist_ok=True)
	fields = ["branch", "row", *METRICS]
	with path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fields)
		writer.writeheader()
		for name, deltas in (
			("Global continuation - Source Global", source_delta),
			("Hard - Global continuation", hard_delta),
			("Soft - Global continuation", soft_delta),
		):
			for index, seed in enumerate(SEEDS):
				writer.writerow(
					{"branch": name, "row": "seed{}".format(seed), **{m: deltas[m][index] for m in METRICS}}
				)
			writer.writerow(
				{"branch": name, "row": "mean", **{m: summarize(deltas[m])[0] for m in METRICS}}
			)
			writer.writerow(
				{"branch": name, "row": "std", **{m: summarize(deltas[m])[1] for m in METRICS}}
			)


def table_row(name, deltas):
	parts = [name]
	for metric in METRICS:
		mean, std = summarize(deltas[metric])
		parts.append("{:+.5f} ± {:.5f}".format(mean, std))
	return "| " + " | ".join(parts) + " |"


def direction_text(deltas):
	return ", ".join(
		"{} {}/3".format(metric, improvement_count(metric, values))
		for metric, values in deltas.items()
	)


def write_report(path, temperature, metrics_by_seed, source_delta, hard_delta, soft_delta):
	hard_worst_std = summarize(hard_delta["worst_psnr"])[1]
	soft_worst_std = summarize(soft_delta["worst_psnr"])[1]
	hard_tail_std = summarize(hard_delta["top25_patch_mse"])[1]
	soft_tail_std = summarize(soft_delta["top25_patch_mse"])[1]
	soft_more_stable = soft_worst_std < hard_worst_std and soft_tail_std < hard_tail_std
	soft_means = {metric: summarize(values)[0] for metric, values in soft_delta.items()}
	hard_means = {metric: summarize(values)[0] for metric, values in hard_delta.items()}
	soft_tail_success = (
		soft_means["worst_psnr"] > 0
		and soft_means["top25_patch_mse"] < 0
		and soft_means["patch_mse_p95"] < 0
		and soft_means["psnr"] >= -0.2
		and soft_means["ber30"] <= 0.005
		and improvement_count("worst_psnr", soft_delta["worst_psnr"]) >= 2
		and (
			soft_means["local_ssim_top25"] > 0
			or soft_means["local_lpips_top25"] < 0
		)
	)
	local_specific = soft_means["worst_psnr"] > soft_means["psnr"] + 0.1
	ber_held = soft_means["ber30"] <= 0.005
	evidence_ready = soft_tail_success and ber_held
	warmup_worthwhile = evidence_ready and not soft_more_stable
	global_gain = {metric: summarize(values)[0] for metric, values in source_delta.items()}
	lines = [
		"# Controlled hard-vs-soft conclusion",
		"",
		"Final Soft temperature: `{}`. All comparisons are paired by seed and use equal 20-epoch continuation from the same seed-specific Global checkpoint.".format(temperature),
		"",
		"## Mean paired deltas",
		"",
		"Positive PSNR/SSIM and negative MSE/LPIPS/BER are improvements.",
		"",
		"| Comparison | ΔPSNR | ΔWorstPSNR | ΔTop25MSE | ΔP95MSE | ΔSSIM | ΔlocalSSIM | ΔLPIPS | ΔlocalLPIPS | ΔBER30 |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
		table_row("Global continuation - source", source_delta),
		table_row("Hard - Global continuation", hard_delta),
		table_row("Soft - Global continuation", soft_delta),
		"",
		"Improvement directions:",
		"",
		"- Hard: {}.".format(direction_text(hard_delta)),
		"- Soft: {}.".format(direction_text(soft_delta)),
		"",
		"## Required questions",
		"",
		"1. **Global continuation gain.** ΔPSNR {:+.3f} dB, ΔWorstPSNR {:+.3f} dB, and ΔBER30 {:+.5f} versus the original Global checkpoints.".format(
			global_gain["psnr"], global_gain["worst_psnr"], global_gain["ber30"]
		),
		"2. **Hard's real gain after controlling training time.** ΔPSNR {:+.3f} dB and ΔWorstPSNR {:+.3f} dB versus Global continuation; ΔBER30 {:+.5f}.".format(
			hard_means["psnr"], hard_means["worst_psnr"], hard_means["ber30"]
		),
		"3. **Soft stability.** {}. ΔWorstPSNR std Hard={:.3f}, Soft={:.3f}; ΔTop25MSE std Hard={:.6f}, Soft={:.6f}.".format(
			"Soft is more stable on both tracked tail statistics" if soft_more_stable else "Soft is not more stable on both tracked tail statistics",
			hard_worst_std, soft_worst_std, hard_tail_std, soft_tail_std,
		),
		"4. **Right-tail suppression.** {}. Mean ΔTop25MSE={:+.6f}, ΔP95MSE={:+.6f}.".format(
			"Supported" if soft_means["top25_patch_mse"] < 0 and soft_means["patch_mse_p95"] < 0 else "Not supported",
			soft_means["top25_patch_mse"], soft_means["patch_mse_p95"],
		),
		"5. **Local specificity.** {}. Soft ΔWorstPSNR={:+.3f} dB versus ΔPSNR={:+.3f} dB.".format(
			"Tail gain exceeds global gain" if local_specific else "Improvement is not clearly local-specific",
			soft_means["worst_psnr"], soft_means["psnr"],
		),
		"6. **BER preservation.** {} with mean ΔBER30={:+.5f}.".format(
			"Yes" if ber_held else "No", soft_means["ber30"]
		),
		"7. **Independent perceptual support.** ΔlocalSSIM={:+.4f}, ΔlocalLPIPS={:+.5f}; {}.".format(
			soft_means["local_ssim_top25"], soft_means["local_lpips_top25"],
			"at least one supports the PSNR result" if soft_means["local_ssim_top25"] > 0 or soft_means["local_lpips_top25"] < 0 else "neither supports the PSNR result",
		),
		"8. **Ready for the paper's main method?** {}.".format(
			"Yes as controlled three-seed evidence, while avoiding claims of broad statistical significance" if evidence_ready else "No; the controlled evidence does not clear the method gate"
		),
		"9. **Test local warm-up next?** {}.".format(
			"Yes, because Soft has value but residual seed variance remains" if warmup_worthwhile else "Not yet; first resolve the failed gate or retain the simpler constant-weight method"
		),
		"10. **Most defensible claim.** Crop-robust watermarking degrades visual quality; a controlled local-tail objective {} reduce high-distortion patch errors without materially changing crop BER.".format(
			"can" if evidence_ready else "has not yet been shown to"
		),
	]
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n")
	return {
		"soft_more_stable": soft_more_stable,
		"soft_tail_success": soft_tail_success,
		"ber_held": ber_held,
		"evidence_ready": evidence_ready,
		"warmup_worthwhile": warmup_worthwhile,
	}


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
	metrics_by_seed = OrderedDict()
	for seed in SEEDS:
		metrics_by_seed[seed] = OrderedDict()
		for name, checkpoint in run_paths(seed, args.temperature).items():
			print("evaluating seed={} branch={}".format(seed, name), flush=True)
			model = load_model(checkpoint, device)
			metrics, _, _ = evaluate_branch(
				model, images, messages, attack_masks, lpips_metric, device,
				args.batch_size, args.patch_size, args.repeats,
			)
			metrics_by_seed[seed][name] = metrics
	soft_name = "Soft P16-T{}-L50".format(args.temperature)
	source_delta = paired_deltas(
		metrics_by_seed, "Global continuation", "Source Global"
	)
	hard_delta = paired_deltas(
		metrics_by_seed, "Hard P16-T25-L50", "Global continuation"
	)
	soft_delta = paired_deltas(metrics_by_seed, soft_name, "Global continuation")
	write_csv(args.output_csv, source_delta, hard_delta, soft_delta)
	decisions = write_report(
		args.output_markdown,
		args.temperature,
		metrics_by_seed,
		source_delta,
		hard_delta,
		soft_delta,
	)
	args.output_json.parent.mkdir(parents=True, exist_ok=True)
	args.output_json.write_text(
		json.dumps(
			{
				"temperature": args.temperature,
				"metrics_by_seed": metrics_by_seed,
				"global_continuation_minus_source": source_delta,
				"hard_minus_global_continuation": hard_delta,
				"soft_minus_global_continuation": soft_delta,
				"decisions": decisions,
			},
			indent=2,
		)
		+ "\n"
	)
	global_psnr = summarize(source_delta["psnr"])[0]
	hard_worst = summarize(hard_delta["worst_psnr"])[0]
	soft_worst = summarize(soft_delta["worst_psnr"])[0]
	soft_ber = summarize(soft_delta["ber30"])[0]
	print("CONTROLLED BASELINE")
	print("* mean PSNR gain vs source: {:+.3f} dB".format(global_psnr))
	print("HARD RESULT")
	print("* mean WorstPSNR gain vs control: {:+.3f} dB".format(hard_worst))
	print("SOFT RESULT")
	print("* T={} mean WorstPSNR gain: {:+.3f} dB; mean BER30 delta: {:+.5f}".format(args.temperature, soft_worst, soft_ber))
	print("* evidence_ready={}".format(decisions["evidence_ready"]))


if __name__ == "__main__":
	main()
