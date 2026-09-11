#!/usr/bin/env python3
"""Generate the three requested paper figures from the uniform evaluation report."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPORT = Path("/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_all.jsonl")
OUTPUT = Path("/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_figures")


def load_final_rows():
	rows = [json.loads(line) for line in REPORT.open() if line.strip()]
	final = {}
	for row in rows:
		if row["run"] not in final or row["epoch"] > final[row["run"]]["epoch"]:
			final[row["run"]] = row
	return final


def ber_curve(row):
	attacks = row["evaluation"]["attacks"]
	return [attacks["crop_30"]["ber"], attacks["crop_50"]["ber"], attacks["crop_70"]["ber"], attacks["crop_100"]["ber"]]


def style_axis(ax):
	ax.grid(axis="y", alpha=0.25, linewidth=0.8)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)


def save_ber_curve(final):
	areas = np.array([30, 50, 70, 100])
	nocrop = np.array(ber_curve(final["nocrop_global_128_m64"]))
	global_names = [
		"optimization_global_seed17_128_m64_crop",
		"optimization_global_seed29_128_m64_crop",
		"optimization_global_seed41_128_m64_crop",
	]
	global_rows = [final[name] for name in global_names]
	global_seed_bers = np.array([ber_curve(row) for row in global_rows])
	crop = np.mean(global_seed_bers, axis=0)
	write_ber_table(areas, nocrop, global_names, global_seed_bers, crop)

	fig, ax = plt.subplots(figsize=(7.2, 4.6))
	ax.plot(areas, nocrop, marker="o", linewidth=2.5, markersize=7, label="No-crop training")
	ax.plot(areas, crop, marker="o", linewidth=2.5, markersize=7, label="Crop-trained global")
	for x, y in zip(areas, nocrop):
		ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
	for x, y in zip(areas, crop):
		ax.annotate(f"{y:.4f}", (x, y), xytext=(0, -17), textcoords="offset points", ha="center", fontsize=9)
	ax.set_xlabel("Crop retained area (%)")
	ax.set_ylabel("BER (lower is better)")
	ax.set_title("Crop robustness: no-crop vs crop-trained model")
	ax.set_xticks(areas)
	ax.set_ylim(bottom=0)
	style_axis(ax)
	ax.legend(frameon=False)
	fig.tight_layout()
	fig.savefig(OUTPUT / "ber_vs_crop_area.png", dpi=300, bbox_inches="tight")
	plt.close(fig)
	return areas, nocrop, crop


def write_ber_table(areas, nocrop, global_names, global_seed_bers, crop):
	"""Persist the exact plotted values with unambiguous attack-area columns."""
	OUTPUT.mkdir(parents=True, exist_ok=True)
	csv_lines = [
		"retained_area_percent,nocrop_ber,global_seed17_ber,global_seed29_ber,global_seed41_ber,global_mean_ber"
	]
	md_lines = [
		"# BER values used in the crop-area figure",
		"",
		"BER is error rate (lower is better). Each attack point contains 250 decoded samples.",
		"",
		"| Retained area | No-crop BER | Global seed17 BER | Global seed29 BER | Global seed41 BER | Global mean BER |",
		"|---:|---:|---:|---:|---:|---:|",
	]
	for index, area in enumerate(areas):
		values = [
			float(area),
			float(nocrop[index]),
			float(global_seed_bers[0, index]),
			float(global_seed_bers[1, index]),
			float(global_seed_bers[2, index]),
			float(crop[index]),
		]
		csv_lines.append(",".join("{:.10g}".format(value) for value in values))
		md_lines.append(
			"| {}% | {:.7f} | {:.7f} | {:.7f} | {:.7f} | {:.7f} |".format(
				int(area), *values[1:]
			)
		)
	(OUTPUT / "ber_vs_crop_area_values.csv").write_text("\n".join(csv_lines) + "\n")
	(OUTPUT / "ber_vs_crop_area_values.md").write_text("\n".join(md_lines) + "\n")


def save_worst_stability():
	seeds = np.array([17, 29, 41])
	values = np.array([32.17, 30.10, 33.04])
	global_reference = 33.73

	fig, ax = plt.subplots(figsize=(6.4, 4.4))
	colors = ["#4C78A8", "#F58518", "#54A24B"]
	ax.plot(seeds, values, color="#333333", linewidth=1.8, zorder=1)
	ax.scatter(seeds, values, s=110, c=colors, edgecolor="white", linewidth=1.2, zorder=2)
	for seed, value in zip(seeds, values):
		ax.annotate(f"{value:.2f}", (seed, value), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10)
	ax.axhline(global_reference, color="#D62728", linestyle="--", linewidth=2, label="Global mean = 33.73 dB")
	ax.set_xlabel("Seed")
	ax.set_ylabel("Worst-patch PSNR (dB)")
	ax.set_title("Worst-patch stability across seeds")
	ax.set_xticks(seeds)
	ax.set_ylim(29, 35)
	style_axis(ax)
	ax.legend(frameon=False, loc="lower right")
	fig.tight_layout()
	fig.savefig(OUTPUT / "worst_patch_stability.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def save_method_bars():
	labels = ["Global", "Worst", "Weight25", "Patch16"]
	values = np.array([33.73, 31.77, 34.45, 34.54])
	colors = ["#4C78A8", "#E45756", "#54A24B", "#F2CF5B"]

	fig, ax = plt.subplots(figsize=(7.0, 4.5))
	bars = ax.bar(labels, values, color=colors, width=0.64)
	for bar, value in zip(bars, values):
		ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
	ax.set_ylabel("Worst-patch PSNR (dB)")
	ax.set_title("Worst-patch quality comparison")
	ax.set_ylim(29, 36)
	style_axis(ax)
	fig.tight_layout()
	fig.savefig(OUTPUT / "worst_patch_method_comparison.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def save_combined(final):
	areas, nocrop, crop = save_ber_curve(final)
	seeds = np.array([17, 29, 41])
	stability = np.array([32.17, 30.10, 33.04])
	methods = ["Global", "Worst", "Weight25", "Patch16"]
	method_values = np.array([33.73, 31.77, 34.45, 34.54])

	fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
	axes[0].plot(areas, nocrop, marker="o", linewidth=2.2, label="No-crop training")
	axes[0].plot(areas, crop, marker="o", linewidth=2.2, label="Crop-trained global")
	axes[0].set_xlabel("Retained area (%)")
	axes[0].set_ylabel("BER")
	axes[0].set_title("BER vs crop area")
	axes[0].set_xticks(areas)
	axes[0].set_ylim(bottom=0)
	axes[0].legend(frameon=False, fontsize=8)

	axes[1].plot(seeds, stability, color="#333333", marker="o", linewidth=2.0)
	axes[1].axhline(33.73, color="#D62728", linestyle="--", linewidth=1.8, label="Global = 33.73")
	axes[1].set_xlabel("Seed")
	axes[1].set_ylabel("Worst-patch PSNR (dB)")
	axes[1].set_title("Worst stability")
	axes[1].set_xticks(seeds)
	axes[1].legend(frameon=False, fontsize=8)

	axes[2].bar(methods, method_values, color=["#4C78A8", "#E45756", "#54A24B", "#F2CF5B"])
	for index, value in enumerate(method_values):
		axes[2].text(index, value + 0.1, f"{value:.2f}", ha="center", fontsize=9)
	axes[2].set_ylabel("Worst-patch PSNR (dB)")
	axes[2].set_title("Method comparison")
	axes[2].set_ylim(29, 36)

	for ax in axes:
		style_axis(ax)
	fig.tight_layout()
	fig.savefig(OUTPUT / "paper_figures_combined.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def main():
	OUTPUT.mkdir(parents=True, exist_ok=True)
	final = load_final_rows()
	save_combined(final)
	save_worst_stability()
	save_method_bars()
	print("saved figures to", OUTPUT)


if __name__ == "__main__":
	main()
