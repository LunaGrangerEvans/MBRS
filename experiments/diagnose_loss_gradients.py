#!/usr/bin/env python3
"""Measure raw/weighted loss and gradient scales on a fixed diagnostic batch."""

import argparse
import csv
import json
import math
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.losses import _patch_scores_and_weights, image_loss_components
from network.Encoder_MP_Decoder import EncoderDecoder


ROOT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
RUNS = OrderedDict(
	[
		("global", "optimization_global_seed17_128_m64_crop"),
		("worst_patch32_weight50", "optimization_worst_seed17_128_m64_crop"),
		("patch16_weight50", "fixed_ablation_patch16_128_m64_crop"),
		("patch64_weight50", "fixed_ablation_patch64_128_m64_crop"),
		("patch32_weight25", "optimization_worst_weight25_seed17_128_m64_crop"),
	]
)
EPOCHS = (1, 10, 40, 100)


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument(
		"--output", type=Path, default=Path("reports/loss_gradient_diagnostics.csv")
	)
	parser.add_argument(
		"--selection-output",
		type=Path,
		default=Path("reports/topk_selection_stability.csv"),
	)
	return parser.parse_args()


def torch_load(path, device):
	try:
		return torch.load(str(path), map_location=device, weights_only=False)
	except TypeError:
		return torch.load(str(path), map_location=device)


def seed_everything(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def tensor_grad_norm(loss, tensor):
	gradient = torch.autograd.grad(loss, tensor, retain_graph=True)[0]
	return float(gradient.norm()), float((gradient.abs() > 0).float().mean())


def parameter_grad_norm(loss, parameters):
	gradients = torch.autograd.grad(
		loss, parameters, retain_graph=True, allow_unused=True
	)
	squared = sum(
		float(gradient.detach().pow(2).sum())
		for gradient in gradients
		if gradient is not None
	)
	return math.sqrt(squared)


def topk_indices(encoded, images, patch_size, topk_ratio):
	patch_scores, _ = _patch_scores_and_weights(encoded, images, patch_size)
	k = max(1, math.ceil(patch_scores.shape[1] * topk_ratio))
	return torch.topk(patch_scores, k, dim=1, sorted=True).indices.detach().cpu()


def mean_jaccard(left, right):
	values = []
	for left_row, right_row in zip(left, right):
		left_set = set(left_row.tolist())
		right_set = set(right_row.tolist())
		values.append(len(left_set & right_set) / len(left_set | right_set))
	return float(np.mean(values))


def run_checkpoint(label, run_name, epoch, images, messages, device):
	run_dir = ROOT / "experiments/runs" / run_name
	config = json.loads((run_dir / "config.resolved.json").read_text())
	checkpoint_path = run_dir / "checkpoint_{:04d}.pth".format(epoch)
	checkpoint = torch_load(checkpoint_path, device)
	seed_everything(20260908 + epoch)
	model = EncoderDecoder(
		config["H"], config["W"], config["message_length"], config["noise_layers"]
	).to(device)
	model.load_state_dict(checkpoint["model"])
	model.train()
	encoded, _, decoded = model(images, messages)
	components = image_loss_components(
		encoded,
		images,
		mode="topk",
		patch_size=config["local_patch_size"],
		topk_ratio=config["local_topk_ratio"],
	)
	losses = OrderedDict(
		[
			("message", F.mse_loss(decoded, messages)),
			("global", components["global_mse"]),
			("local", components["worst_patch_mse"]),
		]
	)
	coefficients = {
		"message": float(config["message_loss_weight"]),
		"global": float(config["global_loss_weight"]),
		"local": float(config["local_loss_weight"]),
	}
	parameters = tuple(model.encoder.parameters())
	rows = []
	for loss_name, loss in losses.items():
		image_grad_norm, image_grad_support = tensor_grad_norm(loss, encoded)
		encoder_grad_norm = parameter_grad_norm(loss, parameters)
		coefficient = coefficients[loss_name]
		rows.append(
			{
				"label": label,
				"run": run_name,
				"epoch": epoch,
				"loss": loss_name,
				"raw_value": float(loss.detach()),
				"coefficient": coefficient,
				"weighted_value": coefficient * float(loss.detach()),
				"encoded_grad_norm": image_grad_norm,
				"weighted_encoded_grad_norm": coefficient * image_grad_norm,
				"encoded_grad_support": image_grad_support,
				"encoder_grad_norm": encoder_grad_norm,
				"weighted_encoder_grad_norm": coefficient * encoder_grad_norm,
			}
		)
	indices = topk_indices(
		encoded, images, config["local_patch_size"], config["local_topk_ratio"]
	)
	return rows, indices


def write_csv(path, rows):
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def main():
	args = parse_args()
	device = torch.device(args.device)
	manifest = torch_load(ROOT / "reports/uniform_eval_manifest.pt", "cpu")
	images = manifest["images"][: args.batch_size].to(device)
	messages = manifest["messages"][: args.batch_size].to(device)
	rows = []
	selection_rows = []
	for label, run_name in RUNS.items():
		previous_indices = None
		previous_epoch = None
		for epoch in EPOCHS:
			print("diagnosing", label, "epoch", epoch, flush=True)
			checkpoint_rows, indices = run_checkpoint(
				label, run_name, epoch, images, messages, device
			)
			rows.extend(checkpoint_rows)
			if previous_indices is not None:
				selection_rows.append(
					{
						"label": label,
						"run": run_name,
						"from_epoch": previous_epoch,
						"to_epoch": epoch,
						"mean_topk_jaccard": mean_jaccard(previous_indices, indices),
						"selected_patches": indices.shape[1],
					}
				)
			previous_indices = indices
			previous_epoch = epoch
	write_csv(args.output, rows)
	write_csv(args.selection_output, selection_rows)
	print("saved", args.output)
	print("saved", args.selection_output)


if __name__ == "__main__":
	main()
