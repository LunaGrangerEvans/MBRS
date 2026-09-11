#!/usr/bin/env python3
"""Summarize canonical checkpoint evaluations and reflect on the motivation."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_all.jsonl"),
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("/mnt/wmcontent/GLX/icassp/MBRS/reports/motivation_reflection.md"),
	)
	return parser.parse_args()


def mean_std(values):
	if not values:
		return "n/a"
	mean = statistics.fmean(values)
	std = statistics.stdev(values) if len(values) > 1 else 0.0
	return "{:.3f} ± {:.3f}".format(mean, std)


def family_name(run):
	if run.startswith("candidate_"):
		parts = run[len("candidate_") :].split("_seed", 1)
		return "candidate_{}".format(parts[0])
	if run.startswith("optimization_"):
		parts = run[len("optimization_") :].split("_seed", 1)
		return "optimization_{}".format(parts[0])
	if run.startswith("fixed_"):
		return "fixed_{}".format(run[len("fixed_") :].rsplit("_128_m64_crop", 1)[0])
	if run == "nocrop_global_128_m64":
		return "nocrop_global"
	return run


CANDIDATE_REUSED_SEED17 = {
	"fixed_ablation_patch16_128_m64_crop": "candidate_worst_patch16_weight50",
	"optimization_worst_weight25_seed17_128_m64_crop": "candidate_worst_patch32_weight25",
	"fixed_ablation_topk50_128_m64_crop": "candidate_worst_patch32_topk50",
}


def candidate_family_name(run):
	if run in CANDIDATE_REUSED_SEED17:
		return CANDIDATE_REUSED_SEED17[run]
	if run.startswith("candidate_"):
		return family_name(run)
	if run.startswith("parallel_"):
		parts = run[len("parallel_") :].split("_seed", 1)
		return "candidate_{}".format(parts[0])
	return None


def row_values(row):
	quality = row["evaluation"]["image_quality"]
	attacks = row["evaluation"]["attacks"]
	return {
		"psnr": quality["psnr"],
		"worst_psnr": quality["worst_patch_psnr"],
		"ssim": quality["ssim"],
		"ber30": attacks["crop_30"]["ber"],
		"ber50": attacks["crop_50"]["ber"],
		"ber70": attacks["crop_70"]["ber"],
	}


def main():
	args = parse_args()
	if not args.input.is_file():
		raise FileNotFoundError(args.input)
	rows = [json.loads(line) for line in args.input.open() if line.strip()]
	by_run = defaultdict(list)
	for row in rows:
		by_run[row["run"]].append(row)
	final_rows = {
		run: max(run_rows, key=lambda row: row["epoch"])
		for run, run_rows in by_run.items()
	}
	by_family = defaultdict(list)
	by_candidate_family = defaultdict(list)
	for run, row in final_rows.items():
		by_family[family_name(run)].append(row_values(row))
		candidate_family = candidate_family_name(run)
		if candidate_family is not None:
			by_candidate_family[candidate_family].append(row_values(row))

	lines = [
		"# 统一 checkpoint 评估与论文动机反思",
		"",
		"评估协议：固定测试 manifest；128×128、64 bit；5 次固定裁剪；局部指标统一为 32×32 patch 的 top-25% worst patch。",
		"",
		"## 各实验最终 checkpoint",
		"",
		"| run | epoch | PSNR | worst patch PSNR | SSIM | BER@30% | BER@50% |",
		"|---|---:|---:|---:|---:|---:|---:|",
	]
	for run in sorted(final_rows):
		row = final_rows[run]
		values = row_values(row)
		lines.append(
			"| {} | {} | {:.3f} | {:.3f} | {:.4f} | {:.5f} | {:.5f} |".format(
				run,
				row["epoch"],
				values["psnr"],
				values["worst_psnr"],
				values["ssim"],
				values["ber30"],
				values["ber50"],
			)
		)

	lines += [
		"",
		"## 按实验族汇总",
		"",
		"| 实验族 | runs | PSNR | worst patch PSNR | SSIM | BER@30% | BER@50% |",
		"|---|---:|---:|---:|---:|---:|---:|",
	]
	for family in sorted(by_family):
		values = by_family[family]
		lines.append(
			"| {} | {} | {} | {} | {} | {} | {} |".format(
				family,
				len(values),
				mean_std([value["psnr"] for value in values]),
				mean_std([value["worst_psnr"] for value in values]),
				mean_std([value["ssim"] for value in values]),
				mean_std([value["ber30"] for value in values]),
				mean_std([value["ber50"] for value in values]),
			)
		)

	lines += [
		"",
		"## 候选方向（合并复用的 seed17）",
		"",
		"这里将已有的 seed17 checkpoint 与本轮新跑的 seed29/41 合并；这些方向均按 canonical 32×32/top25% 指标评估。",
		"",
		"| 候选方向 | runs | PSNR | worst patch PSNR | SSIM | BER@30% | BER@50% |",
		"|---|---:|---:|---:|---:|---:|---:|",
	]
	for family in sorted(by_candidate_family):
		values = by_candidate_family[family]
		lines.append(
			"| {} | {} | {} | {} | {} | {} | {} |".format(
				family,
				len(values),
				mean_std([value["psnr"] for value in values]),
				mean_std([value["worst_psnr"] for value in values]),
				mean_std([value["ssim"] for value in values]),
				mean_std([value["ber30"] for value in values]),
				mean_std([value["ber50"] for value in values]),
			)
		)

	def family_values(name):
		return by_family.get(name, [])

	nocrop = family_values("nocrop_global")
	crop_global = family_values("optimization_global") or family_values("fixed_global")
	crop_worst = family_values("optimization_worst") or family_values("fixed_worst")
	lines += ["", "## 动机检查", ""]
	if nocrop and crop_global:
		n = nocrop[0]
		g = statistics.fmean(value["worst_psnr"] for value in crop_global)
		g_psnr = statistics.fmean(value["psnr"] for value in crop_global)
		g_ber = statistics.fmean(value["ber30"] for value in crop_global)
		lines.append(
			"无剪裁 baseline 与裁剪训练 global 的 canonical 指标差值："
			" worst patch PSNR {:+.3f} dB，global PSNR {:+.3f} dB，BER@30% {:+.5f}。".format(
				n["worst_psnr"] - g,
				n["psnr"] - g_psnr,
				n["ber30"] - g_ber,
			)
		)
		if n["worst_psnr"] > g and n["ber30"] > g_ber:
			lines.append(
				"初步支持 idea 的基本矛盾：无剪裁训练局部质量更好，但裁剪鲁棒性更弱。"
			)
		else:
			lines.append(
				"无剪裁对照没有同时呈现“局部质量更好、裁剪鲁棒性更弱”的清晰模式，论文动机需要谨慎表述。"
			)
	else:
		lines.append("无剪裁或裁剪训练 global 结果尚未齐全，暂不能判断动机。")
	if crop_worst and crop_global:
		w = statistics.fmean(value["worst_psnr"] for value in crop_worst)
		g = statistics.fmean(value["worst_psnr"] for value in crop_global)
		lines.append(
			"worst 相对 crop-trained global 的平均 canonical worst patch PSNR 差值：{:+.3f} dB。".format(
				w - g
			)
		)
		if w > g:
			lines.append("worst 方向在当前结果上改善局部质量，但仍需结合 BER 和跨 seed 方差判断。")
		else:
			lines.append("worst 方向尚未改善局部质量，不能据此支持原始核心假设。")
	lines += [
		"",
		"报告由统一 manifest 生成；不同训练 config 的 patch 参数不会改变本报告的 canonical 评估口径。",
	]
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text("\n".join(lines) + "\n")
	print("saved", args.output)


if __name__ == "__main__":
	main()
