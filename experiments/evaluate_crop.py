#!/usr/bin/env python3
"""Evaluate a local-artifact checkpoint under a common crop protocol."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from network.Encoder_MP_Decoder import EncoderDecoder
from utils.Dataloader import MBRSDataset

from experiments.losses import image_loss_components, mse_to_psnr, ssim_index


DEFAULT_ATTACKS = {
	"identity": "Identity()",
	"crop_30": "RandomCrop(0.3, 0.3)",
	"crop_50": "RandomCrop(0.5, 0.5)",
	"crop_70": "RandomCrop(0.7, 0.7)",
	"crop_100": "RandomCrop(1.0, 1.0)",
}


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, required=True)
	parser.add_argument("--checkpoint", type=Path, required=True)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
	parser.add_argument("--max-batches", type=int, default=0)
	parser.add_argument("--repeats", type=int, default=1)
	parser.add_argument("--save-examples", type=Path, default=None)
	return parser.parse_args()


def choose_device(requested):
	if requested == "cpu":
		return torch.device("cpu")
	if requested == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA was requested but is not available")
		return torch.device("cuda")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path):
	with path.open("r") as file:
		return json.load(file)


def load_checkpoint(path, device):
	"""Load trusted local checkpoints across PyTorch 1.x and 2.6+."""
	try:
		return torch.load(str(path), map_location=device, weights_only=False)
	except TypeError:
		# PyTorch 1.x does not have the weights_only keyword.
		return torch.load(str(path), map_location=device)


def seed_everything(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def bit_accuracy(messages, decoded_messages):
	return float(
		(decoded_messages.gt(0.5) == messages.gt(0.5)).float().mean().item()
	)


def summarize_image_quality(encoded_images, images, config):
	components = image_loss_components(
		encoded_images,
		images,
		mode=config["local_loss_mode"],
		patch_size=config["local_patch_size"],
		patch_stride=config.get("local_patch_stride"),
		topk_ratio=config["local_topk_ratio"],
		multi_scale_patch_sizes=config.get("multi_scale_patch_sizes"),
		multi_scale_patch_strides=config.get("multi_scale_patch_strides"),
		multi_scale_weights=config.get("multi_scale_weights"),
		excess_threshold=config.get("excess_threshold", 1.0),
		excess_epsilon=config.get("excess_epsilon", 1e-12),
		excess_loss_scale=config.get("excess_loss_scale", 1.0),
		content_selector_feature=config.get("content_selector_feature", "gradient"),
		content_selector_alpha=config.get("content_selector_alpha", 1.0),
		soft_tail_temperature=config.get("soft_tail_temperature", 0.5),
		soft_tail_detach_weights=config.get("soft_tail_detach_weights", True),
	)
	ssim = ssim_index(encoded_images.detach(), images, window_size=5).item()
	return {
		"global_mse": components["global_mse"].item(),
		"patch_mean_mse": components["patch_mean_mse"].item(),
		"worst_patch_mse": components["worst_patch_mse"].item(),
		"psnr": mse_to_psnr(components["global_mse"].detach()).item(),
		"worst_patch_psnr": mse_to_psnr(
			components["worst_patch_mse"].detach()
		).item(),
		"ssim": ssim,
	}


def main():
	args = parse_args()
	if args.max_batches < 0 or args.repeats <= 0:
		raise ValueError("max-batches must be non-negative and repeats must be positive")
	config = load_config(args.config)
	device = choose_device(args.device)
	seed_everything(config["seed"])
	attacks = config.get("eval_attacks", DEFAULT_ATTACKS)
	if not isinstance(attacks, dict) or not attacks:
		raise ValueError("eval_attacks must be a non-empty object")

	dataset = MBRSDataset(
		str(Path(config["dataset_path"]) / "test"), config["H"], config["W"]
	)
	loader = DataLoader(
		dataset,
		batch_size=config["batch_size"],
		shuffle=False,
		num_workers=config.get("num_workers", 0),
		pin_memory=device.type == "cuda",
	)
	checkpoint = load_checkpoint(args.checkpoint, device)
	model = EncoderDecoder(config["H"], config["W"], config["message_length"], ["Identity()"])
	model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
	model.to(device)
	model.eval()
	attack_models = {}
	for name, expression in attacks.items():
		attack_model = EncoderDecoder(
			config["H"], config["W"], config["message_length"], [expression]
		).to(device)
		attack_model.encoder.load_state_dict(model.encoder.state_dict())
		attack_model.decoder.load_state_dict(model.decoder.state_dict())
		attack_model.eval()
		attack_models[name] = attack_model

	quality_total = {name: 0.0 for name in [
		"global_mse", "patch_mean_mse", "worst_patch_mse", "psnr",
		"worst_patch_psnr", "ssim"
	]}
	attack_total = {
		name: {"bit_accuracy": 0.0, "ber": 0.0} for name in attacks
	}
	quality_count = 0
	attack_count = {name: 0 for name in attacks}
	example = None

	with torch.no_grad():
		for batch_index, images in enumerate(loader):
			if args.max_batches and batch_index >= args.max_batches:
				break
			images = images.to(device, non_blocking=device.type == "cuda")
			messages = torch.randint(
				0, 2, (images.shape[0], config["message_length"]), device=device
			).float()
			encoded_images = model.encoder(images, messages)
			quality = summarize_image_quality(encoded_images, images, config)
			for key, value in quality.items():
				quality_total[key] += value * images.shape[0]
			quality_count += images.shape[0]

			if example is None and args.save_examples is not None:
				example = {
					"images": images.cpu(),
					"encoded_images": encoded_images.cpu(),
					"messages": messages.cpu(),
				}

			for name, expression in attacks.items():
				for _ in range(args.repeats):
					attack_model = attack_models[name]
					noised_images = attack_model.noise([encoded_images, images])
					decoded_messages = attack_model.decoder(noised_images)
					accuracy = bit_accuracy(messages, decoded_messages)
					attack_total[name]["bit_accuracy"] += accuracy * images.shape[0]
					attack_total[name]["ber"] += (1 - accuracy) * images.shape[0]
					attack_count[name] += images.shape[0]
					if example is not None and name not in example:
						example[name] = noised_images.cpu()

	if quality_count == 0:
		raise RuntimeError("no test batches were processed")
	result = {
		"checkpoint": str(args.checkpoint),
		"device": str(device),
		"samples": quality_count,
		"repeats": args.repeats,
		"image_quality": {
			key: value / quality_count for key, value in quality_total.items()
		},
		"attacks": {
			name: {
				"noise": expression,
				"samples": attack_count[name],
				"bit_accuracy": attack_total[name]["bit_accuracy"] / attack_count[name],
				"ber": attack_total[name]["ber"] / attack_count[name],
			}
			for name, expression in attacks.items()
		},
	}
	args.output.parent.mkdir(parents=True, exist_ok=True)
	with args.output.open("w") as file:
		json.dump(result, file, indent=2)
	if example is not None:
		args.save_examples.parent.mkdir(parents=True, exist_ok=True)
		torch.save(example, args.save_examples)
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()
