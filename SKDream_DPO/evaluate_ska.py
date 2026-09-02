#!/usr/bin/env python3
"""Evaluate skeleton-image alignment with the SKA scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from skalign.model import SkalignModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--ska-checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-repo", type=Path)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path, help="Defaults to <image-dir>/ska_results.json"
    )
    return parser.parse_args()


def load_models(args: argparse.Namespace, device: torch.device):
    if args.dinov2_repo:
        dino = torch.hub.load(
            str(args.dinov2_repo), "dinov2_vitl14_reg", pretrained=False, source="local"
        )
    else:
        dino = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitl14_reg",
            pretrained=False,
            source="github",
        )
    dino.load_state_dict(torch.load(args.dino_checkpoint, map_location="cpu"), strict=True)
    dino.requires_grad_(False).eval().to(device)

    scorer = SkalignModel(1024, 3)
    scorer.load_state_dict(torch.load(args.ska_checkpoint, map_location="cpu"))
    scorer.requires_grad_(False).eval().to(device)
    return dino, scorer


def open_rgb_foreground(path: Path) -> Image.Image:
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
    black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    return Image.alpha_composite(black, rgba).convert("RGB")


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def mean_by_key(scores: dict[str, float], metadata: dict, key: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for item_id, score in scores.items():
        value = metadata[item_id].get(key)
        if value is not None:
            buckets.setdefault(str(value), []).append(score)
    return {name: float(np.mean(values)) for name, values in sorted(buckets.items())}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    with args.eval_json.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    dino, scorer = load_models(args, device)

    scores = {}
    for item_id in sorted(metadata):
        item_dir = args.image_dir / item_id
        conditions = [
            transform(open_rgb(item_dir / f"cond_{view}.png"))
            for view in range(args.num_views)
        ]
        images = [
            transform(open_rgb_foreground(item_dir / f"gen_{repeat}_{view}.png"))
            for repeat in range(args.num_repeats)
            for view in range(args.num_views)
        ]
        condition_tensor = torch.stack(conditions * args.num_repeats).to(device)
        image_tensor = torch.stack(images).to(device)

        with torch.inference_mode():
            image_features = scorer(dino(image_tensor, return_ret=True)["x_norm_patchtokens"])
            condition_features = scorer(
                dino(condition_tensor, return_ret=True)["x_norm_patchtokens"]
            )
            score = torch.cosine_similarity(
                image_features, condition_features, dim=-1
            ).mean()
        scores[item_id] = float(score)
        print(f"{item_id}: {scores[item_id]:.4f}")

    result = {
        "mean": float(np.mean(list(scores.values()))),
        "by_class": mean_by_key(scores, metadata, "class"),
        "by_subclass": mean_by_key(
            {
                item_id: score
                for item_id, score in scores.items()
                if metadata[item_id].get("class") == 0
            },
            metadata,
            "sub_class",
        ),
        "per_item": scores,
    }
    output_path = args.output or args.image_dir / "ska_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Mean SKA: {result['mean']:.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
