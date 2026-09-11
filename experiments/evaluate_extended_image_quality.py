#!/usr/bin/env python3
"""Read-only extended full-reference evaluation for frozen checkpoints."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from pytorch_msssim import ms_ssim, ssim

if __package__ in {"", None}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.analyze_patch_distortion import extract_patches
from experiments.losses import patch_mse_per_sample
from network.Encoder_MP_Decoder import EncoderDecoder

ROOT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
METHODS = {
    "Global continuation": "controlled_seed17_global_continuation",
    "Hard Patch16 stride8 Top25": "controlled_seed17_hard_patch16_stride8_top25_weight50",
    "Hard Patch16 stride8 Top10": "controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50",
    "Gradient-aware Top10 alpha2": "seed17_contentaware_gradient_patch16_stride8_top10_alpha2_global0.5_local0.5",
}


def concentration_metrics(scores):
    values = np.asarray(scores, dtype=np.float64)
    count = values.shape[1]
    k = max(1, math.ceil(count * 0.10))
    ordered = np.sort(values, axis=1)
    total = values.sum(axis=1)
    mean = values.mean(axis=1)
    return {
        "gini": ((2 * np.arange(1, count + 1) - count - 1) * ordered).sum(axis=1) / (count * total),
        "cv": values.std(axis=1) / np.maximum(mean, 1e-12),
        "top10_over_mean": ordered[:, -k:].mean(axis=1) / np.maximum(mean, 1e-12),
        "top10_energy_share": ordered[:, -k:].sum(axis=1) / np.maximum(total, 1e-12),
    }


def load_file(path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def load_model(run, device):
    ck = load_file(ROOT / "experiments/runs" / run / "checkpoint_0020.pth", device)
    c = ck["config"]
    model = EncoderDecoder(c["H"], c["W"], c["message_length"], ["Identity()"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def encode(model, images, messages, device, batch_size):
    values = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            values.append(model.encoder(images[start:start + batch_size].to(device),
                                        messages[start:start + batch_size].to(device)).cpu())
    return torch.cat(values)


def patch_lpips(encoded, images, device, batch_size):
    import lpips
    metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    # LPIPS expects [-1, 1], while the frozen metric contract first clips RGB
    # tensors to [0, 1].  Convert the clipped tensors back to LPIPS' input range.
    encoded = (((encoded + 1) / 2).clamp(0, 1) * 2) - 1
    images = (((images + 1) / 2).clamp(0, 1) * 2) - 1
    encoded_patches = extract_patches(encoded, 32)
    image_patches = extract_patches(images, 32)
    values = []
    with torch.no_grad():
        for start in range(0, len(encoded_patches), batch_size * 4):
            values.append(metric(encoded_patches[start:start + batch_size * 4].to(device),
                                 image_patches[start:start + batch_size * 4].to(device)).flatten().cpu())
    return torch.cat(values).reshape(len(images), 16).numpy()


def full_lpips(encoded, images, device, batch_size):
    import lpips
    metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    encoded = (((encoded + 1) / 2).clamp(0, 1) * 2) - 1
    images = (((images + 1) / 2).clamp(0, 1) * 2) - 1
    values = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            values.append(metric(encoded[start:start + batch_size].to(device),
                                 images[start:start + batch_size].to(device)).flatten().cpu())
    return torch.cat(values).numpy()


def evaluate(encoded, images, device, batch_size):
    left = ((encoded + 1) / 2).clamp(0, 1)
    right = ((images + 1) / 2).clamp(0, 1)
    mse = (left - right).square().mean((1, 2, 3)).numpy()
    weights = torch.tensor([0.3, 0.3, 0.4], device=device)
    full_ssim, full_ms = [], []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            a, b = left[start:start + batch_size].to(device), right[start:start + batch_size].to(device)
            full_ssim.append(ssim(a, b, data_range=1.0, size_average=False, win_size=7).cpu())
            full_ms.append(ms_ssim(a, b, data_range=1.0, size_average=False, win_size=7, weights=weights).cpu())
    encoded_patches = extract_patches(left, 32)
    image_patches = extract_patches(right, 32)
    patch_ssim = []
    with torch.no_grad():
        for start in range(0, len(encoded_patches), batch_size * 4):
            a = encoded_patches[start:start + batch_size * 4].to(device)
            b = image_patches[start:start + batch_size * 4].to(device)
            patch_ssim.append(ssim(a, b, data_range=1.0, size_average=False, win_size=5).cpu())
    patch_ssim = torch.cat(patch_ssim).reshape(len(images), 16).numpy()
    patch_mse = patch_mse_per_sample(left, right, 32).numpy()
    concentration = concentration_metrics(patch_mse)
    local_lpips = patch_lpips(encoded, images, device, batch_size)
    full_lpips_values = full_lpips(encoded, images, device, batch_size)
    k = 2
    bottom = np.sort(patch_ssim, axis=1)[:, :k]
    top_lpips = np.sort(local_lpips, axis=1)[:, -k:]
    rows = []
    for i in range(len(images)):
        rows.append({
            "image_index": i,
            "global_mse": float(mse[i]),
            "global_psnr": 10 * math.log10(1 / max(float(mse[i]), 1e-12)),
            "global_ssim": float(full_ssim[0].new_tensor(full_ssim).reshape(-1)[i]) if False else float(torch.cat(full_ssim)[i]),
            "global_ms_ssim": float(torch.cat(full_ms)[i]),
            "global_dists": np.nan,
            "global_gmsd": np.nan,
            "full_lpips": float(full_lpips_values[i]),
            "patch_ssim_mean": float(patch_ssim[i].mean()),
            "patch_ssim_median": float(np.median(patch_ssim[i])),
            "patch_ssim_p10": float(np.percentile(patch_ssim[i], 10)),
            "patch_ssim_bottom10_mean": float(bottom[i].mean()),
            "patch_ssim_min": float(patch_ssim[i].min()),
            "ssim_tail_gap": float(np.median(patch_ssim[i]) - bottom[i].mean()),
            "patch_lpips_top10_mean": float(top_lpips[i].mean()),
            "patch_lpips_max": float(local_lpips[i].max()),
            "patch_mse_top25": float(np.sort(patch_mse[i])[-4:].mean()),
            "patch_mse_p95": float(np.percentile(patch_mse[i], 95)),
            "top25_local_psnr": float(10 * math.log10(1 / max(float(np.sort(patch_mse[i])[-4:].mean()), 1e-12))),
            "gini": float(concentration["gini"][i]),
            "cv": float(concentration["cv"][i]),
            "top10_over_mean": float(concentration["top10_over_mean"][i]),
            "top10_energy_share": float(concentration["top10_energy_share"][i]),
        })
    return rows


def sanity(device):
    x = torch.rand(2, 3, 128, 128, device=device)
    weights = torch.tensor([0.3, 0.3, 0.4], device=device)
    same_s = float(ssim(x, x, data_range=1.0, size_average=False, win_size=7).mean())
    same_m = float(ms_ssim(x, x, data_range=1.0, size_average=False, win_size=7, weights=weights).mean())
    noisy = (x + torch.randn_like(x) * 0.3).clamp(0, 1)
    noisy_s = float(ssim(x, noisy, data_range=1.0, size_average=False, win_size=7).mean())
    noisy_m = float(ms_ssim(x, noisy, data_range=1.0, size_average=False, win_size=7, weights=weights).mean())
    result = {"ssim_identity": same_s, "ms_ssim_identity_3scale": same_m,
              "ssim_noise": noisy_s, "ms_ssim_noise_3scale": noisy_m}
    assert abs(same_s - 1) < 1e-5 and abs(same_m - 1) < 1e-5
    assert noisy_s < same_s and noisy_m < same_m
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/uniform_eval_manifest.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/extended_image_quality")
    args = parser.parse_args()
    device = torch.device(args.device)
    manifest = load_file(args.manifest, "cpu")
    images, messages = manifest["images"].float(), manifest["messages"].float()
    sanity_values = sanity(device)
    all_rows = []
    for method, run in METHODS.items():
        print("evaluating", method, flush=True)
        encoded = encode(load_model(run, device), images, messages, device, args.batch_size)
        all_rows.extend({"method": method, **row} for row in evaluate(encoded, images, device, args.batch_size))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0])
    with (args.output_dir / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    base = [row for row in all_rows if row["method"] == "Global continuation"]
    metrics = ["global_psnr", "global_ssim", "global_ms_ssim", "full_lpips", "top25_local_psnr",
               "patch_ssim_bottom10_mean", "ssim_tail_gap", "patch_lpips_top10_mean",
               "patch_mse_top25", "patch_mse_p95", "gini", "cv", "top10_over_mean", "top10_energy_share"]
    lower = {"full_lpips", "ssim_tail_gap", "patch_lpips_top10_mean", "patch_mse_top25", "patch_mse_p95"}
    summary = []
    for method in METHODS:
        cur = [row for row in all_rows if row["method"] == method]
        item = {"method": method}
        for metric in metrics:
            values = np.array([row[metric] for row in cur], dtype=float)
            baseline = np.array([row[metric] for row in base], dtype=float)
            if metric == "global_psnr":
                item[metric] = float(10 * np.log10(1 / np.nanmean([row["global_mse"] for row in cur])))
                base_metric = float(10 * np.log10(1 / np.nanmean([row["global_mse"] for row in base])))
                item["delta_" + metric] = item[metric] - base_metric
            else:
                item[metric] = float(np.nanmean(values))
                item["delta_" + metric] = float(np.nanmean(values - baseline))
            item["improved_fraction_" + metric] = float(np.nanmean(values < baseline) if metric in lower else np.nanmean(values > baseline))
        summary.append(item)
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "sanity.json").write_text(json.dumps({
        "sanity": sanity_values,
        "metric_status": {
            "dists": "N/A: no mature implementation installed; no implementation added",
            "gmsd": "N/A: no mature implementation installed; no implementation added",
        },
        "input_range": "[0,1]", "ms_ssim": "3-scale weights [0.3,0.3,0.4]",
        "patch_grid": "native 32x32 non-overlap", "test_images": len(images),
    }, indent=2) + "\n")
    lines = ["# Extended full-reference image quality", "", "Frozen controlled checkpoints and fixed formal manifest only. No training was run.", "",
             "## Sanity", ""] + [f"- {key}: {value:.8f}" for key, value in sanity_values.items()]
    lines += ["- DISTS: N/A; no mature implementation installed and no implementation added.",
              "- GMSD: N/A; no mature implementation installed and no implementation added.", "",
              "## Summary", "", "| Method | PSNR | SSIM | MS-SSIM | DISTS | GMSD | LPIPS | Top25 local PSNR | Bottom10 SSIM | Top10 LPIPS | Gini | CV | Top10/Mean | Top10 energy share |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['method']} | {row['global_psnr']:.6f} | {row['global_ssim']:.6f} | {row['global_ms_ssim']:.6f} | N/A | N/A | {row['full_lpips']:.8f} | {row['top25_local_psnr']:.6f} | {row['patch_ssim_bottom10_mean']:.6f} | {row['patch_lpips_top10_mean']:.8f} | {row['gini']:.6f} | {row['cv']:.6f} | {row['top10_over_mean']:.6f} | {row['top10_energy_share']:.6f} |")
    lines += ["", "## Direction", "",
              "SSIM/MS-SSIM higher is better; DISTS/GMSD/LPIPS lower is better; Bottom10 SSIM higher is better; SSIM TailGap and perceptual tail metrics lower are better.",
              "", "## Scope", "",
              "Native32 patch LPIPS and SSIM are reported. MS-SSIM is full-image only because patch support is not sufficient. DISTS/GMSD remain N/A due environment availability."]
    (args.output_dir / "extended_image_quality.md").write_text("\n".join(lines) + "\n")
    print("saved", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
