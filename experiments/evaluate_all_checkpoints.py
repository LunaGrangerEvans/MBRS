#!/usr/bin/env python3
"""Evaluate every local experiment checkpoint under one fixed test protocol."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.losses import image_loss_components, mse_to_psnr, ssim_index
from network.Encoder_MP_Decoder import EncoderDecoder
from utils.Dataloader import MBRSDataset


CANONICAL_PATCH_SIZE = 32
CANONICAL_TOPK_RATIO = 0.25
ATTACK_AREAS = {
	"identity": None,
	"crop_30": 0.3,
	"crop_50": 0.5,
	"crop_70": 0.7,
	"crop_100": 1.0,
}


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--runs-root",
		type=Path,
		default=Path("/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs"),
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_all.jsonl"),
	)
	parser.add_argument(
		"--manifest",
		type=Path,
		default=Path("/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_manifest.pt"),
	)
	parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
	parser.add_argument("--repeats", type=int, default=5)
	return parser.parse_args()


def load_checkpoint(path, device):
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


def load_config(run_dir):
	path = run_dir / "config.resolved.json"
	if not path.is_file():
		raise FileNotFoundError("missing config.resolved.json: {}".format(path))
	with path.open("r") as file:
		return json.load(file)


def make_test_manifest(config, path, repeats):
	seed_everything(int(config["seed"]))
	dataset = MBRSDataset(
		str(Path(config["dataset_path"]) / "test"), config["H"], config["W"]
	)
	loader = DataLoader(
		dataset,
		batch_size=config["batch_size"],
		shuffle=False,
		num_workers=config.get("num_workers", 0),
	)
	images = torch.cat([batch for batch in loader], dim=0)
	messages = torch.randint(
		0, 2, (images.shape[0], config["message_length"]), dtype=torch.float
	)
	generator = np.random.RandomState(int(config["seed"]) + 100003)
	attack_masks = {}
	for name, area in ATTACK_AREAS.items():
		if area is None:
			attack_masks[name] = None
			continue
		side = math.sqrt(area)
		height = int(side * config["H"])
		width = int(side * config["W"])
		masks = []
		for _ in range(repeats):
			batch_masks = []
			for _ in range(math.ceil(len(dataset) / config["batch_size"])):
				h_start = (
					0
					if height == config["H"]
					else int(generator.randint(0, config["H"] - height))
				)
				w_start = (
					0
					if width == config["W"]
					else int(generator.randint(0, config["W"] - width))
				)
				mask = torch.zeros(1, 1, config["H"], config["W"])
				mask[:, :, h_start : h_start + height, w_start : w_start + width] = 1
				batch_masks.append(mask)
			masks.append(batch_masks)
		attack_masks[name] = masks
	manifest = {
		"seed": int(config["seed"]),
		"samples": len(dataset),
		"repeats": repeats,
		"H": config["H"],
		"W": config["W"],
		"message_length": config["message_length"],
		"images": images,
		"messages": messages,
		"attack_masks": attack_masks,
	}
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(manifest, path)
	return manifest


def load_or_make_manifest(config, path, repeats):
	if path.is_file():
		try:
			manifest = torch.load(str(path), map_location="cpu", weights_only=False)
		except TypeError:
			manifest = torch.load(str(path), map_location="cpu")
		if (
			manifest["samples"] == 50
			and manifest["H"] == config["H"]
			and manifest["W"] == config["W"]
			and manifest["message_length"] == config["message_length"]
			and manifest["repeats"] == repeats
		):
			return manifest
	return make_test_manifest(config, path, repeats)


def bit_accuracy(messages, decoded_messages):
	return float(
		(decoded_messages.gt(0.5) == messages.gt(0.5)).float().mean().item()
	)


def evaluate_checkpoint(checkpoint_path, config, manifest, device, repeats):
	images = manifest["images"].to(device)
	messages = manifest["messages"].to(device)
	model = EncoderDecoder(
		config["H"], config["W"], config["message_length"], ["Identity()"]
	).to(device)
	checkpoint = load_checkpoint(checkpoint_path, device)
	model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
	model.eval()

	quality_total = {
		"global_mse": 0.0,
		"patch_mean_mse": 0.0,
		"worst_patch_mse": 0.0,
		"psnr": 0.0,
		"worst_patch_psnr": 0.0,
		"ssim": 0.0,
	}
	attack_total = {
		name: {"bit_accuracy": 0.0, "ber": 0.0, "samples": 0}
		for name in ATTACK_AREAS
	}
	loader = DataLoader(
		range(images.shape[0]), batch_size=config["batch_size"], shuffle=False
	)
	with torch.no_grad():
		for batch_index, indices in enumerate(loader):
			indices = indices.tolist()
			batch_images = images[indices]
			batch_messages = messages[indices]
			encoded_images = model.encoder(batch_images, batch_messages)
			components = image_loss_components(
				encoded_images,
				batch_images,
				mode="topk",
				patch_size=CANONICAL_PATCH_SIZE,
				topk_ratio=CANONICAL_TOPK_RATIO,
			)
			quality = {
				"global_mse": components["global_mse"].item(),
				"patch_mean_mse": components["patch_mean_mse"].item(),
				"worst_patch_mse": components["worst_patch_mse"].item(),
				"psnr": mse_to_psnr(components["global_mse"].detach()).item(),
				"worst_patch_psnr": mse_to_psnr(
					components["worst_patch_mse"].detach()
				).item(),
				"ssim": ssim_index(encoded_images, batch_images, window_size=5).item(),
			}
			for key, value in quality.items():
				quality_total[key] += value * len(indices)

			for name, area in ATTACK_AREAS.items():
				for repeat in range(repeats):
					if area is None:
						noised_images = encoded_images
					else:
						mask = manifest["attack_masks"][name][repeat][batch_index].to(device)
						noised_images = encoded_images * mask
					decoded_messages = model.decoder(noised_images)
					accuracy = bit_accuracy(batch_messages, decoded_messages)
					attack_total[name]["bit_accuracy"] += accuracy * len(indices)
					attack_total[name]["ber"] += (1 - accuracy) * len(indices)
					attack_total[name]["samples"] += len(indices)

	quality_count = images.shape[0]
	return {
		"checkpoint": str(checkpoint_path),
		"run": checkpoint_path.parent.name,
		"epoch": int(checkpoint_path.stem.split("_")[-1]),
		"train_config": {
			"name": config.get("name"),
			"seed": config.get("seed"),
			"noise_layers": config.get("noise_layers"),
			"local_loss_mode": config.get("local_loss_mode"),
			"local_patch_size": config.get("local_patch_size"),
			"local_topk_ratio": config.get("local_topk_ratio"),
			"global_loss_weight": config.get("global_loss_weight"),
			"local_loss_weight": config.get("local_loss_weight"),
			"local_weight_schedule": config.get("local_weight_schedule"),
			"soft_tail_temperature": config.get("soft_tail_temperature"),
			"soft_tail_detach_weights": config.get("soft_tail_detach_weights"),
			"deterministic": config.get("deterministic"),
		},
		"evaluation": {
			"protocol": "fixed_manifest",
			"samples": quality_count,
			"repeats": repeats,
			"patch_size": CANONICAL_PATCH_SIZE,
			"topk_ratio": CANONICAL_TOPK_RATIO,
			"image_quality": {
				key: value / quality_count for key, value in quality_total.items()
			},
			"attacks": {
				name: {
					"noise": "Identity()" if area is None else "RandomCrop({0}, {0})".format(area),
					"samples": values["samples"],
					"bit_accuracy": values["bit_accuracy"] / values["samples"],
					"ber": values["ber"] / values["samples"],
				}
				for name, area in ATTACK_AREAS.items()
				for values in [attack_total[name]]
			},
		},
	}


def main():
	args = parse_args()
	if args.repeats <= 0:
		raise ValueError("repeats must be positive")
	if args.device == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA requested but unavailable")
	device = torch.device(args.device)
	checkpoint_paths = sorted(args.runs_root.glob("*/checkpoint_*.pth"))
	if not checkpoint_paths:
		raise RuntimeError("no checkpoints found below {}".format(args.runs_root))
	first_config = load_config(checkpoint_paths[0].parent)
	manifest = load_or_make_manifest(first_config, args.manifest, args.repeats)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	evaluated = {}
	if args.output.is_file():
		with args.output.open() as file:
			for line in file:
				if line.strip():
					row = json.loads(line)
					evaluated[row["checkpoint"]] = row
	count = 0
	for checkpoint_path in checkpoint_paths:
		checkpoint_key = str(checkpoint_path)
		if checkpoint_key in evaluated:
			continue
		config = load_config(checkpoint_path.parent)
		if (
			config["H"] != manifest["H"]
			or config["W"] != manifest["W"]
			or config["message_length"] != manifest["message_length"]
		):
			raise ValueError("incompatible dimensions in {}".format(checkpoint_path))
		print("evaluating {}".format(checkpoint_path), flush=True)
		row = evaluate_checkpoint(
			checkpoint_path, config, manifest, device, args.repeats
		)
		with args.output.open("a") as file:
			file.write(json.dumps(row) + "\n")
			file.flush()
		count += 1
	print(
		"uniform_evaluation_complete total={} newly_evaluated={} output={}".format(
			len(checkpoint_paths), count, args.output
		),
		flush=True,
	)


if __name__ == "__main__":
	main()
