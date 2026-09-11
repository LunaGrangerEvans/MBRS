#!/usr/bin/env python3
"""Build quality--robustness tables and figures without training."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/root/workspace/GLX/icassp/MBRS")
MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
QUALITY = ROOT / "reports/extended_image_quality_summary.csv"
TM_QUALITY = MOUNT / "reports/external_baseline_metrics/summary.csv"
TM_RAW = MOUNT / "reports/trustmark_raw_bits/summary.csv"
MBRS_BER = MOUNT / "reports/controlled_seed17_gradientaware_alpha2_metrics.json"
TABLE = ROOT / "reports/quality_robustness_pareto_table.csv"
SUMMARY = ROOT / "reports/quality_robustness_pareto_summary.md"
FIG_DIR = MOUNT / "visualizations/external_baselines"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def f(row, key):
    return float(row[key])


def add_row(rows, method, comparison, payload, quality, crop, raw_ber, bit_accuracy, exact):
    rows.append({
        "method": method, "comparison_type": comparison, "payload_ecc": payload,
        "psnr": f(quality, "global_psnr"), "top25_local_psnr": f(quality, "top25_local_psnr"),
        "gini": f(quality, "gini"), "lpips": f(quality, "full_lpips"),
        "crop_ratio": crop, "raw_ber": raw_ber, "bit_accuracy": bit_accuracy,
        "exact_decode_success": exact,
    })


def main():
    quality = {row["method"]: row for row in read_csv(QUALITY)}
    external_quality = {row["method"]: row for row in read_csv(TM_QUALITY)}
    raw = {(row["method"], row["crop_ratio"]): row for row in read_csv(TM_RAW)}
    mb = json.loads(MBRS_BER.read_text())
    mb_metrics = mb["metrics"]
    mb_methods = {
        "MBRS Global continuation": "Global continuation",
        "MBRS Hard Patch16 / stride8 / Top10 / MSE / image-local 0.5/0.5": "Hard Patch16 stride8 Top10",
    }
    rows = []
    ratios = ["100%", "70%", "50%", "40%", "30%"]
    for method, qname in mb_methods.items():
        q = quality[qname]
        m = mb_metrics["Global continuation" if method.startswith("MBRS Global") else "Hard patch16 stride8 top10 global0.5 local0.5"]
        for ratio in ratios:
            key = "ber" + ratio.rstrip("%")
            ber = float(m[key])
            add_row(rows, method, "STRICT internal", "64 raw bits; no ECC", q, ratio, ber, 1.0 - ber, "N/A")
    for method in ["TrustMark Q", "TrustMark P"]:
        q = external_quality[method]
        for ratio in ratios:
            r = raw[(method, ratio)]
            add_row(rows, method, "REFERENCE", "100 internal bits; BCH-5; 61 data bits", q, ratio,
                    f(r, "raw_ber"), f(r, "raw_bit_accuracy"), f(r, "exact_decode_success"))
    with TABLE.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    methods = [
        "MBRS Global continuation", "MBRS Hard Patch16 / stride8 / Top10 / MSE / image-local 0.5/0.5",
        "TrustMark Q", "TrustMark P",
    ]
    colors = {"100%": "#777777", "70%": "#1f77b4", "50%": "#ff7f0e", "40%": "#2ca02c", "30%": "#d62728"}
    markers = {methods[0]: "o", methods[1]: "s", methods[2]: "^", methods[3]: "D"}
    labels = {methods[0]: "MBRS Global", methods[1]: "MBRS Hard Top10", methods[2]: "TrustMark Q (REF)", methods[3]: "TrustMark P (REF)"}
    metrics = [("psnr", "PSNR (dB) ↑", "higher is better"),
               ("top25_local_psnr", "Top25 local PSNR (dB) ↑", "higher is better"),
               ("gini", "Gini ↓", "lower is better"),
               ("lpips", "Full LPIPS ↓", "lower is better")]

    def plot_panels(path, selected_ratios, title):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (metric, xlabel, direction) in zip(axes.flat, metrics):
            for method in methods:
                for ratio in selected_ratios:
                    row = next(item for item in rows if item["method"] == method and item["crop_ratio"] == ratio)
                    ax.scatter(row[metric], row["bit_accuracy"], s=70, marker=markers[method],
                               color=colors[ratio], edgecolor="black", linewidth=0.35,
                               label=f"{labels[method]} / {ratio}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Raw bit accuracy ↑")
            ax.set_title(f"{xlabel} — {direction}")
            ax.grid(alpha=0.25)
            ax.set_ylim(0.70, 1.01)
        handles = []
        for method in methods:
            handles.append(plt.Line2D([], [], marker=markers[method], color="white", markerfacecolor="#555555",
                                      markeredgecolor="black", linestyle="None", label=labels[method]))
        for ratio in selected_ratios:
            handles.append(plt.Line2D([], [], marker="o", color=colors[ratio], linestyle="None", label=f"crop {ratio}"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4, fontsize=8, frameon=True)
        fig.suptitle(title + "\nTrustMark is REFERENCE only; raw bit semantics differ from MBRS raw 64-bit BER", fontsize=13)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.88, hspace=0.34, wspace=0.25)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    plot_panels(FIG_DIR / "quality_robustness_tradeoff.png", ["70%", "50%", "40%", "30%"], "Quality–robustness trade-off across crop ratios")
    for ratio in ["70%", "50%", "40%", "30%"]:
        plot_panels(FIG_DIR / f"quality_robustness_tradeoff_crop{ratio.rstrip('%')}.png", [ratio], f"Quality–robustness trade-off at crop {ratio}")

    def row(method, ratio):
        return next(item for item in rows if item["method"] == method and item["crop_ratio"] == ratio)
    hard_quality = quality["Hard Patch16 stride8 Top10"]
    global_quality = quality["Global continuation"]
    lines = [
        "# Quality–robustness Pareto analysis",
        "",
        "No model was trained. Image quality uses the frozen RGB `[0,1]` extended metric definition. Robustness uses the latest fixed evaluator for MBRS and the verified TrustMark pre-ECC audit for TrustMark.",
        "",
        "## TrustMark raw-bit audit",
        "",
        "The official pipeline exposes decoder logits before thresholding and BCH. The raw bit is `decoder_logit > 0` and is compared with the expected 100-bit BCH protected packet. This is verified pre-ECC output, not a guessed or hacked signal.",
        "",
        "| Method | Crop | Raw BER ↓ | Raw bit accuracy ↑ | Detection success | ECC exact success | Trials |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in read_csv(TM_RAW):
        lines.append(f"| {r['method']} | {r['crop_ratio']} | {float(r['raw_ber']):.6f} | {float(r['raw_bit_accuracy']):.6f} | {float(r['detection_success']):.4f} | {float(r['exact_decode_success']):.4f} | {r['trials']} |")
    lines += ["", "## Unified MBRS and TrustMark table", "", "See [quality_robustness_pareto_table.csv](quality_robustness_pareto_table.csv). MBRS exact decode is `N/A` because it reports raw 64-bit BER without ECC; TrustMark exact success is ECC-corrected message success.", ""]
    lines += ["| Method | Comparison | Crop | PSNR | Top25 local PSNR | Gini | LPIPS | Raw BER | Bit accuracy | Exact decode success |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        exact = r["exact_decode_success"] if r["exact_decode_success"] == "N/A" else f"{float(r['exact_decode_success']):.4f}"
        lines.append(f"| {r['method']} | {r['comparison_type']} | {r['crop_ratio']} | {r['psnr']:.6f} | {r['top25_local_psnr']:.6f} | {r['gini']:.6f} | {r['lpips']:.8f} | {r['raw_ber']:.6f} | {r['bit_accuracy']:.6f} | {exact} |")
    lines += ["", "## Core observations", "", f"- Hard Top10 versus Global improves frozen image quality: PSNR `{float(hard_quality['global_psnr']) - float(global_quality['global_psnr']):+.6f} dB`, Top25 local PSNR `{float(hard_quality['top25_local_psnr']) - float(global_quality['top25_local_psnr']):+.6f} dB`, Gini `{float(hard_quality['gini']) - float(global_quality['gini']):+.6f}`, and full LPIPS `{float(hard_quality['full_lpips']) - float(global_quality['full_lpips']):+.8f}`.", "- At 70% crop, Hard Top10 and Global both have raw bit accuracy `1.000000`; at 50/40/30%, Hard Top10 changes BER by only `+0.000063/+0.000188/-0.000188`, respectively.", "- TrustMark Q has lower raw BER than P at every partial crop, while P has better image quality. This is a clear Q-versus-P quality–robustness trade-off within TrustMark.", "- TrustMark points occupy a higher-quality region, but they remain REFERENCE points because the packet is 100 protected bits with BCH-5/61 data bits, not MBRS raw 64-bit BER.", "- The plots use marker shape for method family and color for crop ratio; no strict superiority boundary is drawn.", "", "## Claim boundary", "", "The evidence supports an internal Hard Top10 quality–robustness operating point: better frozen quality with essentially preserved MBRS raw BER. It does not support an ours-over-TrustMark strict or human-perceptual superiority claim."]
    SUMMARY.write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE}")
    print(f"wrote {SUMMARY}")
    print(f"wrote {FIG_DIR / 'quality_robustness_tradeoff.png'}")


if __name__ == "__main__":
    main()
