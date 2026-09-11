#!/usr/bin/env python3
"""Render cover, watermark, residual, and crop examples into a grid."""

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def to_display(images):
	return images.clamp(-1, 1).add(1).div(2)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--examples", type=Path, required=True)
	parser.add_argument("--output", type=Path, required=True)
	args = parser.parse_args()
	examples = torch.load(str(args.examples), map_location="cpu")
	cover = to_display(examples["images"])
	encoded = to_display(examples["encoded_images"])
	residual = (examples["encoded_images"] - examples["images"]).abs()
	residual_max = residual.flatten(1).max(dim=1)[0].clamp_min(1e-8)
	residual = residual / residual_max.view(-1, 1, 1, 1)
	panels = [cover, encoded, residual]
	for name in sorted(key for key in examples if key not in {"images", "encoded_images", "messages"}):
		panels.append(to_display(examples[name]))
	row = torch.cat(panels, dim=3)
	grid = make_grid(row, nrow=1, padding=4, pad_value=1)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	save_image(grid, str(args.output))
	print("saved", args.output)


if __name__ == "__main__":
	main()
