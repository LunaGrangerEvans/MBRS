#!/usr/bin/env python3
"""Train one controlled MBRS image-loss variant for the ICASSP study."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from network.Encoder_MP_Decoder import EncoderDecoder
from utils.Dataloader import MBRSDataset

from experiments.losses import image_loss_components, mse_to_psnr
from experiments.color_losses import oklab_tail_components
from experiments.oklab_global import chroma_tail_components


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
	parser.add_argument("--max-train-batches", type=int, default=0)
	parser.add_argument("--max-val-batches", type=int, default=0)
	parser.add_argument("--epochs", type=int, default=0)
	parser.add_argument("--resume", type=Path, default=None)
	parser.add_argument(
		"--init-checkpoint",
		type=Path,
		default=None,
		help="load model/optimizer state but restart the configured epoch schedule at 1",
	)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--global-loss-weight", type=float, default=None)
	parser.add_argument("--local-loss-weight", type=float, default=None)
	return parser.parse_args()


def load_config(path):
	with path.open("r") as file:
		config = json.load(file)
	config["noise_layers"] = list(config["noise_layers"])
	return config


def choose_device(requested):
	if requested == "cpu":
		return torch.device("cpu")
	if requested == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA was requested but is not available")
		return torch.device("cuda")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, device):
	"""Load trusted local checkpoints across PyTorch 1.x and 2.6+."""
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


def configure_determinism(config):
	if not config.get("deterministic", False):
		return
	os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms(
		True, warn_only=config.get("deterministic_warn_only", False)
	)


def seed_worker(_worker_id):
	worker_seed = torch.initial_seed() % (2**32)
	random.seed(worker_seed)
	np.random.seed(worker_seed)


def image_loss_weights(config, epoch):
	"""Return effective global/local coefficients for one epoch."""
	target_local = float(config["local_loss_weight"])
	target_global = float(config["global_loss_weight"])
	schedule = config.get("local_weight_schedule")
	if not schedule or schedule.get("type", "constant") == "constant":
		return target_global, target_local
	if schedule.get("type") != "linear_warmup":
		raise ValueError("local_weight_schedule.type must be constant or linear_warmup")
	start_fraction = float(schedule["start_fraction"])
	end_fraction = float(schedule["end_fraction"])
	if not 0 <= start_fraction < end_fraction <= 1:
		raise ValueError("warm-up fractions must satisfy 0 <= start < end <= 1")
	target_local = float(schedule.get("target_weight", target_local))
	progress = epoch / float(config["epochs"])
	if progress <= start_fraction:
		local_weight = 0.0
	elif progress >= end_fraction:
		local_weight = target_local
	else:
		local_weight = target_local * (
			(progress - start_fraction) / (end_fraction - start_fraction)
		)
	if schedule.get("preserve_total_image_weight", False):
		total_weight = float(
			schedule.get("total_image_weight", target_global + target_local)
		)
		global_weight = total_weight - local_weight
	else:
		global_weight = target_global
	if global_weight < 0 or local_weight < 0:
		raise ValueError("effective image-loss weights must be non-negative")
	return global_weight, local_weight


def bit_metrics(messages, decoded_messages):
	predicted = decoded_messages.gt(0.5)
	target = messages.gt(0.5)
	accuracy = (predicted == target).float().mean()
	return float(accuracy.item()), float((1 - accuracy).item())


def compute_image_components(encoded_images, cover_images, config):
	if config["local_loss_mode"] == "oklab_topk":
		return oklab_tail_components(
			encoded_images, cover_images,
			patch_size=config["local_patch_size"],
			patch_stride=config.get("local_patch_stride", 1),
			topk_ratio=config["local_topk_ratio"],
		)
	if config["local_loss_mode"] == "chroma_topk":
		components = image_loss_components(
			encoded_images, cover_images, mode="topk",
			patch_size=config["local_patch_size"],
			patch_stride=config.get("local_patch_stride"),
			topk_ratio=config["local_topk_ratio"],
		)
		components.update(chroma_tail_components(
			encoded_images, cover_images,
			patch_size=config["local_patch_size"],
			patch_stride=config.get("local_patch_stride", 8),
			topk_ratio=config["local_topk_ratio"],
		))
		return components
	return image_loss_components(
		encoded_images, cover_images, mode=config["local_loss_mode"],
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


def batch_metrics(
	encoded_images, cover_images, messages, decoded_messages, config, components=None
):
	if components is None:
		components = compute_image_components(encoded_images, cover_images, config)
	bit_accuracy, ber = bit_metrics(messages, decoded_messages)
	global_psnr = mse_to_psnr(components["global_mse"].detach()).item()
	worst_psnr = mse_to_psnr(components["worst_patch_mse"].detach()).item()
	return {
		"bit_accuracy": bit_accuracy,
		"ber": ber,
		"global_mse": components["global_mse"].detach().item(),
		"patch_mean_mse": components["patch_mean_mse"].detach().item(),
		"worst_patch_mse": components["worst_patch_mse"].detach().item(),
		"psnr": global_psnr,
		"worst_patch_psnr": worst_psnr,
		"soft_weight_mean": components["soft_weight_mean"].detach().item(),
		"soft_weight_max": components["soft_weight_max"].detach().item(),
		"soft_effective_patches": components["soft_effective_patches"].detach().item(),
	}


def average_metrics(total, count):
	return {key: value / count for key, value in total.items()}


def run_epoch(
	model, loader, optimizer, device, config, train, max_batches, loss_weights
):
	if train:
		model.train()
	else:
		model.eval()
	total = {
		"loss": 0.0,
		"message_loss": 0.0,
		"global_mse": 0.0,
		"local_mse": 0.0,
		"weighted_message_loss": 0.0,
		"weighted_global_loss": 0.0,
		"weighted_local_loss": 0.0,
		"weighted_image_loss": 0.0,
		"global_loss_weight": 0.0,
		"local_loss_weight": 0.0,
		"bit_accuracy": 0.0,
		"ber": 0.0,
		"patch_mean_mse": 0.0,
		"worst_patch_mse": 0.0,
		"psnr": 0.0,
		"worst_patch_psnr": 0.0,
		"soft_weight_mean": 0.0,
		"soft_weight_max": 0.0,
		"soft_effective_patches": 0.0,
	}
	count = 0
	for batch_index, images in enumerate(loader):
		if max_batches and batch_index >= max_batches:
			break
		images = images.to(device, non_blocking=device.type == "cuda")
		messages = torch.randint(
			0, 2, (images.shape[0], config["message_length"]), device=device
		).float()
		context = torch.enable_grad() if train else torch.no_grad()
		with context:
			encoded_images, _, decoded_messages = model(images, messages)
			components = compute_image_components(encoded_images, images, config)
			message_loss = F.mse_loss(decoded_messages, messages)
			weighted_message_loss = config["message_loss_weight"] * message_loss
			weighted_global_loss = loss_weights["global"] * components["global_mse"]
			weighted_local_loss = loss_weights["local"] * components["local_mse"]
			weighted_image_loss = weighted_global_loss + weighted_local_loss
			if config.get("global_chroma_weight", 0.0) > 0:
				weighted_image_loss = weighted_image_loss + (
					config["global_chroma_weight"] * components["local_chroma"]
				)
			color_diagnostics = None
			if config.get("global_oklab_weight", 0.0) > 0:
				from experiments.oklab_global import global_oklab_components
				color_diagnostics = global_oklab_components(encoded_images, images)
				weighted_image_loss = weighted_image_loss + (
					config["global_oklab_weight"] * color_diagnostics["global_oklab"]
				)
			loss = weighted_message_loss + weighted_image_loss
			if not torch.isfinite(loss):
				raise FloatingPointError("non-finite training/validation loss")
			if train:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		metrics = batch_metrics(
			encoded_images,
			images,
			messages,
			decoded_messages,
			config,
			components=components,
		)
		batch_size = images.shape[0]
		total["loss"] += loss.detach().item() * batch_size
		total["message_loss"] += message_loss.detach().item() * batch_size
		total["global_mse"] += metrics["global_mse"] * batch_size
		total["local_mse"] += components["local_mse"].detach().item() * batch_size
		total["weighted_message_loss"] += weighted_message_loss.detach().item() * batch_size
		total["weighted_global_loss"] += weighted_global_loss.detach().item() * batch_size
		total["weighted_local_loss"] += weighted_local_loss.detach().item() * batch_size
		total["weighted_image_loss"] += weighted_image_loss.detach().item() * batch_size
		total["global_loss_weight"] += loss_weights["global"] * batch_size
		total["local_loss_weight"] += loss_weights["local"] * batch_size
		for key in metrics:
			if key not in {"global_mse"}:
				total[key] += metrics[key] * batch_size
		total.setdefault("local_oklab", 0.0)
		total.setdefault("local_chroma", 0.0)
		total["local_oklab"] += components.get("local_oklab", components["local_mse"]).detach().item() * batch_size
		total["local_chroma"] += components.get("local_chroma", components["local_mse"].detach() * 0).detach().item() * batch_size
		if color_diagnostics is not None:
			for key, value in color_diagnostics.items():
				total[key] = total.get(key, 0.0) + float(value.detach()) * batch_size
			total["weighted_global_oklab"] = total.get("weighted_global_oklab", 0.0) + (
				config["global_oklab_weight"] * float(color_diagnostics["global_oklab"].detach()) * batch_size
			)
		count += batch_size
	if count == 0:
		raise RuntimeError("no batches were processed")
	return average_metrics(total, count), count


def main():
	args = parse_args()
	skip_marker = args.output_dir / "SKIP_RUN"
	if skip_marker.is_file():
		print("skip_run marker={}".format(skip_marker), flush=True)
		return
	config = load_config(args.config)
	if args.epochs:
		config["epochs"] = args.epochs
	if args.seed is not None:
		config["seed"] = args.seed
	if args.global_loss_weight is not None:
		config["global_loss_weight"] = args.global_loss_weight
	if args.local_loss_weight is not None:
		config["local_loss_weight"] = args.local_loss_weight
	if config["local_loss_mode"] == "none" and config["local_loss_weight"] != 0:
		raise ValueError("local_loss_weight must be zero for local_loss_mode=none")
	if args.resume is not None and args.init_checkpoint is not None:
		raise ValueError("--resume and --init-checkpoint are mutually exclusive")
	if args.max_train_batches < 0 or args.max_val_batches < 0:
		raise ValueError("batch limits must be non-negative")

	device = choose_device(args.device)
	if (
		config.get("require_single_gpu", False)
		and device.type == "cuda"
		and torch.cuda.device_count() != 1
	):
		raise RuntimeError(
			"controlled run requires exactly one visible GPU; found {}".format(
				torch.cuda.device_count()
			)
		)
	seed_everything(config["seed"])
	configure_determinism(config)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	with (args.output_dir / "config.resolved.json").open("w") as file:
		json.dump(
			{
				**config,
				"device": str(device),
				"resume": str(args.resume) if args.resume is not None else None,
				"init_checkpoint": (
					str(args.init_checkpoint)
					if args.init_checkpoint is not None
					else None
				),
			},
			file,
			indent=2,
		)

	train_dataset = MBRSDataset(
		os.path.join(config["dataset_path"], "train"),
		config["H"],
		config["W"],
		sort_files=config.get("deterministic", False),
	)
	val_dataset = MBRSDataset(
		os.path.join(config["dataset_path"], "validation"),
		config["H"],
		config["W"],
		sort_files=config.get("deterministic", False),
	)
	loader_kwargs = {
		"batch_size": config["batch_size"],
		"num_workers": config.get("num_workers", 0),
		"pin_memory": device.type == "cuda",
	}
	if config.get("deterministic", False):
		train_generator = torch.Generator().manual_seed(config["seed"] + 200003)
		val_generator = torch.Generator().manual_seed(config["seed"] + 300007)
		train_loader = DataLoader(
			train_dataset,
			shuffle=True,
			generator=train_generator,
			worker_init_fn=seed_worker,
			**loader_kwargs,
		)
		val_loader = DataLoader(
			val_dataset,
			shuffle=False,
			generator=val_generator,
			worker_init_fn=seed_worker,
			**loader_kwargs,
		)
	else:
		train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
		val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

	model = EncoderDecoder(
		config["H"],
		config["W"],
		config["message_length"],
		config["noise_layers"],
	).to(device)
	if device.type == "cuda" and torch.cuda.device_count() > 1:
		model = torch.nn.DataParallel(model)
	optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
	start_epoch = 1
	checkpoint_path = args.resume or args.init_checkpoint
	if checkpoint_path is not None:
		checkpoint = load_checkpoint(checkpoint_path, device)
		model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
		model_to_load.load_state_dict(checkpoint["model"])
		if "optimizer" in checkpoint:
			optimizer.load_state_dict(checkpoint["optimizer"])
			for parameter_group in optimizer.param_groups:
				parameter_group["lr"] = config["lr"]
		if args.resume is not None:
			start_epoch = int(checkpoint["epoch"]) + 1
			print("resumed_from={} next_epoch={}".format(args.resume, start_epoch), flush=True)
		else:
			print(
				"initialized_from={} source_epoch={} schedule_epoch=1".format(
					args.init_checkpoint, checkpoint.get("epoch", "unknown")
				),
				flush=True,
			)

	train_log = args.output_dir / "train.jsonl"
	val_log = args.output_dir / "val.jsonl"
	started = time.time()
	print(
		"experiment={} device={} train_images={} val_images={} mode={} patch={} topk={}".format(
			config["name"],
			device,
			len(train_dataset),
			len(val_dataset),
			config["local_loss_mode"],
			config["local_patch_size"],
			config["local_topk_ratio"],
		),
		flush=True,
	)
	for epoch in range(start_epoch, config["epochs"] + 1):
		global_weight, local_weight = image_loss_weights(config, epoch)
		loss_weights = {"global": global_weight, "local": local_weight}
		train_metrics, train_count = run_epoch(
			model,
			train_loader,
			optimizer,
			device,
			config,
			train=True,
			max_batches=args.max_train_batches,
			loss_weights=loss_weights,
		)
		val_metrics, val_count = run_epoch(
			model,
			val_loader,
			optimizer,
			device,
			config,
			train=False,
			max_batches=args.max_val_batches,
			loss_weights=loss_weights,
		)
		row = {
			"epoch": epoch,
			"elapsed_s": round(time.time() - started, 3),
			"samples": train_count,
			**train_metrics,
		}
		with train_log.open("a") as file:
			file.write(json.dumps(row) + "\n")
		row = {
			"epoch": epoch,
			"elapsed_s": round(time.time() - started, 3),
			"samples": val_count,
			**val_metrics,
		}
		with val_log.open("a") as file:
			file.write(json.dumps(row) + "\n")
		print(
			"epoch={} train_loss={:.5f} train_ber={:.5f} val_ber={:.5f} "
			"val_psnr={:.3f} val_worst_psnr={:.3f} global_weight={:.4f} local_weight={:.4f}".format(
				epoch,
				train_metrics["loss"],
				train_metrics["ber"],
				val_metrics["ber"],
				val_metrics["psnr"],
				val_metrics["worst_patch_psnr"],
				global_weight,
				local_weight,
			),
			flush=True,
		)
		if epoch == 1 or epoch % config["save_every"] == 0 or epoch == config["epochs"]:
			model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
			torch.save(
				{
					"epoch": epoch,
					"model": model_to_save.state_dict(),
				"optimizer": optimizer.state_dict(),
				"config": config,
			},
			args.output_dir / ("checkpoint_{:04d}.pth".format(epoch)),
			)


if __name__ == "__main__":
	main()
