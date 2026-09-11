#!/usr/bin/env python3
"""Build paired-seed summaries from the existing uniform evaluation report."""

import argparse
import csv
import json
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_all.jsonl")
DEFAULT_CSV = Path("reports/matched_seed_summary.csv")
DEFAULT_MARKDOWN = Path("reports/matched_seed_summary.md")
SEEDS = (17, 29, 41)
METRICS = ("psnr", "worst_psnr", "ssim", "ber30", "ber50")


CONFIG_RUNS = OrderedDict(
	[
		(
			"crop_global_baseline",
			{
				17: "optimization_global_seed17_128_m64_crop",
				29: "optimization_global_seed29_128_m64_crop",
				41: "optimization_global_seed41_128_m64_crop",
			},
		),
		(
			"patch_mean",
			{
				17: "optimization_patch_mean_seed17_128_m64_crop",
				29: "optimization_patch_mean_seed29_128_m64_crop",
				41: "optimization_patch_mean_seed41_128_m64_crop",
			},
		),
		(
			"worst_patch32_top25_weight50",
			{
				17: "optimization_worst_seed17_128_m64_crop",
				29: "optimization_worst_seed29_128_m64_crop",
				41: "optimization_worst_seed41_128_m64_crop",
			},
		),
		(
			"worst_patch16_top25_weight50",
			{
				17: "fixed_ablation_patch16_128_m64_crop",
				29: "candidate_worst_patch16_weight50_seed29_128_m64_crop",
				41: "candidate_worst_patch16_weight50_seed41_128_m64_crop",
			},
		),
		(
			"worst_patch16_top25_weight25",
			{
				seed: "candidate_worst_patch16_weight25_seed{}_128_m64_crop".format(seed)
				for seed in SEEDS
			},
		),
		(
			"worst_patch16_top25_weight75",
			{
				seed: "candidate_worst_patch16_weight75_seed{}_128_m64_crop".format(seed)
				for seed in SEEDS
			},
		),
		(
			"worst_patch32_top25_weight25",
			{
				17: "optimization_worst_weight25_seed17_128_m64_crop",
				29: "candidate_worst_patch32_weight25_seed29_128_m64_crop",
				41: "candidate_worst_patch32_weight25_seed41_128_m64_crop",
			},
		),
		(
			"worst_patch32_top50_weight50",
			{
				17: "fixed_ablation_topk50_128_m64_crop",
				29: "candidate_worst_patch32_topk50_seed29_128_m64_crop",
				41: "candidate_worst_patch32_topk50_seed41_128_m64_crop",
			},
		),
		(
			"worst_patch8_top10_weight50",
			{
				seed: "parallel_patch8_topk10_weight50_seed{}_128_m64_crop".format(seed)
				for seed in SEEDS
			},
		),
		(
			"worst_patch8_top25_weight25",
			{
				seed: "parallel_patch8_worst_weight25_seed{}_128_m64_crop".format(seed)
				for seed in SEEDS
			},
		),
		(
			"worst_patch8_top25_weight50",
			{
				seed: "parallel_patch8_worst_weight50_seed{}_128_m64_crop".format(seed)
				for seed in SEEDS
			},
		),
		(
			"worst_patch8_top25_weight75",
			{
				seed: "parallel_patch8_worst_weight75_seed{}_128_m64_crop".format(seed)
				for seed in SEEDS
			},
		),
		(
			"worst_patch8_top50_weight50",
			{
				seed: "parallel_patch8_topk50_weight50_seed{}_128_m64_crop".format(seed)
				for seed in SEEDS
			},
		),
		(
			"worst_patch64_top25_weight50",
			{17: "fixed_ablation_patch64_128_m64_crop"},
		),
		(
			"worst_patch32_top10_weight50",
			{17: "fixed_ablation_topk10_128_m64_crop"},
		),
		("nocrop_global", {17: "nocrop_global_128_m64"}),
	]
)


IGNORED_DUPLICATE_RUNS = {
	"parallel_worst_patch16_weight50_seed29_128_m64_crop",
	"parallel_worst_patch16_weight50_seed41_128_m64_crop",
	"parallel_worst_patch32_weight25_seed29_128_m64_crop",
	"parallel_worst_patch32_weight25_seed41_128_m64_crop",
}


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
	parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
	parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
	return parser.parse_args()


def row_values(row):
	quality = row["evaluation"]["image_quality"]
	attacks = row["evaluation"]["attacks"]
	return {
		"psnr": quality["psnr"],
		"worst_psnr": quality["worst_patch_psnr"],
		"ssim": quality["ssim"],
		"ber30": attacks["crop_30"]["ber"],
		"ber50": attacks["crop_50"]["ber"],
	}


def mean(values):
	return statistics.fmean(values) if values else None


def std(values):
	return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def fmt(value, digits=4):
	return "—" if value is None else ("{:.%df}" % digits).format(value)


def fmt_signed(value, digits):
	return "—" if value is None else ("{:+.%df}" % digits).format(value)


def load_final_rows(path):
	by_run = defaultdict(list)
	with path.open() as file:
		for line in file:
			if line.strip():
				row = json.loads(line)
				by_run[row["run"]].append(row)
	return {
		run: max(rows, key=lambda item: item["epoch"])
		for run, rows in by_run.items()
	}


def build_records(final_rows):
	baseline_runs = CONFIG_RUNS["crop_global_baseline"]
	records = OrderedDict()
	for config_name, seed_runs in CONFIG_RUNS.items():
		seed_records = OrderedDict()
		for seed in SEEDS:
			run = seed_runs.get(seed)
			row = final_rows.get(run) if run else None
			if row is None or row["epoch"] < 100:
				seed_records[seed] = None
				continue
			values = row_values(row)
			baseline_row = final_rows.get(baseline_runs[seed])
			if baseline_row is None or baseline_row["epoch"] < 100:
				raise RuntimeError("missing completed Global baseline for seed {}".format(seed))
			baseline = row_values(baseline_row)
			seed_records[seed] = {
				"run": run,
				"values": values,
				"deltas": {
					metric: values[metric] - baseline[metric] for metric in METRICS
				},
			}
		records[config_name] = seed_records
	return records


def summary_for(seed_records, key):
	return {
		metric: [
			record[key][metric]
			for record in seed_records.values()
			if record is not None
		]
		for metric in METRICS
	}


def screening_result(seed_records):
	available = [record for record in seed_records.values() if record is not None]
	if len(available) != 3:
		return "INCOMPLETE"
	deltas = summary_for(seed_records, "deltas")
	checks = {
		"dWorst>+0.3": mean(deltas["worst_psnr"]) > 0.3,
		"dPSNR>-0.2": mean(deltas["psnr"]) > -0.2,
		"dBER30<+0.005": mean(deltas["ber30"]) < 0.005,
		"worst direction>=2/3": sum(value > 0 for value in deltas["worst_psnr"]) >= 2,
	}
	failed = [name for name, passed in checks.items() if not passed]
	return "PASS" if not failed else "FAIL: " + ", ".join(failed)


def write_csv(path, records):
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = ["config", "row", "run", *METRICS, *("delta_" + m for m in METRICS)]
	with path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writeheader()
		for config_name, seed_records in records.items():
			for seed, record in seed_records.items():
				row = {"config": config_name, "row": "seed{}".format(seed)}
				if record is not None:
					row["run"] = record["run"]
					row.update(record["values"])
					row.update({"delta_" + key: value for key, value in record["deltas"].items()})
				writer.writerow(row)
			for label, reducer in (("mean", mean), ("std", std)):
				values = summary_for(seed_records, "values")
				deltas = summary_for(seed_records, "deltas")
				row = {"config": config_name, "row": label, "run": ""}
				row.update({metric: reducer(items) for metric, items in values.items()})
				row.update({"delta_" + metric: reducer(items) for metric, items in deltas.items()})
				writer.writerow(row)


def write_markdown(path, records, final_rows):
	lines = [
		"# Matched-seed experiment summary",
		"",
		"All values come from epoch-100 rows in the fixed-manifest uniform evaluation. Deltas are candidate minus the crop-trained Global baseline at the same seed. Negative BER deltas are better.",
		"",
		"## Configuration-level screening",
		"",
		"| Configuration | Seeds | PSNR | Worst PSNR | BER@30 | ΔPSNR | ΔWorst | ΔBER@30 | Screening |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---|",
	]
	for config_name, seed_records in records.items():
		available = [seed for seed, record in seed_records.items() if record is not None]
		values = summary_for(seed_records, "values")
		deltas = summary_for(seed_records, "deltas")
		screening = (
			"REFERENCE"
			if config_name == "crop_global_baseline"
			else screening_result(seed_records)
		)
		lines.append(
			"| {} | {} | {} ± {} | {} ± {} | {} ± {} | {} | {} | {} | {} |".format(
				config_name,
				",".join(map(str, available)) or "none",
				fmt(mean(values["psnr"]), 3),
				fmt(std(values["psnr"]), 3),
				fmt(mean(values["worst_psnr"]), 3),
				fmt(std(values["worst_psnr"]), 3),
				fmt(mean(values["ber30"]), 5),
				fmt(std(values["ber30"]), 5),
				fmt_signed(mean(deltas["psnr"]), 3),
				fmt_signed(mean(deltas["worst_psnr"]), 3),
				fmt_signed(mean(deltas["ber30"]), 5),
				screening,
			)
		)

	lines += [
		"",
		"Internal screening requires mean ΔWorstPSNR > +0.3 dB, mean ΔPSNR > -0.2 dB, mean ΔBER30 < +0.005, and positive ΔWorstPSNR in at least 2/3 seeds. It is not a statistical-significance test.",
		"",
		"## Per-seed matched comparisons",
	]
	for config_name, seed_records in records.items():
		lines += [
			"",
			"### {}".format(config_name),
			"",
			"| Row | PSNR | Worst PSNR | SSIM | BER@30 | BER@50 | ΔPSNR | ΔWorst | ΔSSIM | ΔBER30 | ΔBER50 |",
			"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
		]
		for seed, record in seed_records.items():
			if record is None:
				lines.append("| seed{} | incomplete | — | — | — | — | — | — | — | — | — |".format(seed))
				continue
			v = record["values"]
			d = record["deltas"]
			lines.append(
				"| seed{} | {:.3f} | {:.3f} | {:.4f} | {:.5f} | {:.5f} | {:+.3f} | {:+.3f} | {:+.4f} | {:+.5f} | {:+.5f} |".format(
					seed, v["psnr"], v["worst_psnr"], v["ssim"], v["ber30"], v["ber50"],
					d["psnr"], d["worst_psnr"], d["ssim"], d["ber30"], d["ber50"],
				)
			)
		for label, reducer in (("mean", mean), ("std", std)):
			v = summary_for(seed_records, "values")
			d = summary_for(seed_records, "deltas")
			lines.append(
				"| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
					label,
					fmt(reducer(v["psnr"]), 3), fmt(reducer(v["worst_psnr"]), 3),
					fmt(reducer(v["ssim"]), 4), fmt(reducer(v["ber30"]), 5),
					fmt(reducer(v["ber50"]), 5), fmt(reducer(d["psnr"]), 3),
					fmt(reducer(d["worst_psnr"]), 3), fmt(reducer(d["ssim"]), 4),
					fmt(reducer(d["ber30"]), 5), fmt(reducer(d["ber50"]), 5),
				)
			)

	missing = []
	for config_name, seed_records in records.items():
		for seed, record in seed_records.items():
			if record is None:
				missing.append("{} seed{}".format(config_name, seed))
	lines += [
		"",
		"## Integrity notes",
		"",
		"- Incomplete combinations: {}.".format(", ".join(missing) if missing else "none"),
		"- The four `parallel_worst_patch16/32` runs are duplicate executions of seed/config pairs already represented by canonical `candidate_*` runs and are excluded from matched three-seed means.",
		"- The existing `motivation_reflection.md` candidate-family aggregation merges those duplicates, producing n=5 for two nominal three-seed configurations; do not use those n=5 rows for formal comparison.",
	]
	unknown_duplicates = sorted(IGNORED_DUPLICATE_RUNS - set(final_rows))
	if unknown_duplicates:
		lines.append("- Expected duplicate runs absent from input: {}.".format(", ".join(unknown_duplicates)))
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n")


def main():
	args = parse_args()
	final_rows = load_final_rows(args.input)
	records = build_records(final_rows)
	write_csv(args.csv, records)
	write_markdown(args.markdown, records, final_rows)
	print("saved", args.csv)
	print("saved", args.markdown)


if __name__ == "__main__":
	main()
