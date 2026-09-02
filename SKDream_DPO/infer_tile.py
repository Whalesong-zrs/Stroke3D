#!/usr/bin/env python3
"""Upscale SKDream multi-view images for texture refinement."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image


DEFAULT_NEGATIVE_PROMPT = (
    "low poly, white model, noise, strange color, ugly, oversaturated, "
    "doubled face, b&w, sepia, freckles, paintings, sketches, worst quality, "
    "low quality, lowres, monochrome, grayscale, error, blurry, artifacts"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing eval.json")
    parser.add_argument(
        "--image-dir",
        "--save_dir",
        dest="image_dir",
        type=Path,
        required=True,
        help="Directory of per-object SKDream outputs",
    )
    parser.add_argument("--num-views", "--num_view", dest="num_views", type=int, default=4)
    parser.add_argument("--repeat-num", "--repeat_num", dest="repeat_num", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--tile-scale", "--cond_scale", dest="tile_scale", type=float, default=1.0)
    parser.add_argument("--canny-scale", type=float, default=0.5)
    parser.add_argument(
        "--negative-prompt",
        "--neg_prompt",
        dest="negative_prompt",
        default="",
        help="Use 'default' for the historical negative prompt",
    )
    parser.add_argument("--base-model", default="ckpt/stable-diffusion-v1-5")
    parser.add_argument("--tile-controlnet", default="ckpt/control_v11f1e_sd15_tile")
    parser.add_argument("--canny-controlnet", default="ckpt/sd-controlnet-canny")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Diffusers to download missing model files",
    )
    parser.add_argument("--disable-xformers", action="store_true")
    return parser.parse_args()


def canny_condition(image: Image.Image) -> Image.Image:
    edges = cv2.Canny(np.asarray(image), 100, 200)
    return Image.fromarray(np.repeat(edges[..., None], 3, axis=2))


def view_prompt_index(angle: float) -> int:
    if angle < 45 or angle > 315:
        return 0
    if angle <= 135:
        return 1
    if angle < 225:
        return 2
    return 3


def main() -> None:
    args = parse_args()
    if args.num_views < 1 or args.repeat_num < 1:
        raise ValueError("num_views and repeat_num must be positive")

    eval_path = args.data_dir.expanduser() / "eval.json"
    with eval_path.open(encoding="utf-8") as handle:
        evaluation = json.load(handle)

    if args.negative_prompt == "default":
        negative_prompt = DEFAULT_NEGATIVE_PROMPT
    else:
        negative_prompt = args.negative_prompt or None

    local_files_only = not args.allow_download
    controlnets = [
        ControlNetModel.from_pretrained(
            args.tile_controlnet,
            torch_dtype=torch.float16,
            local_files_only=local_files_only,
        ),
        ControlNetModel.from_pretrained(
            args.canny_controlnet,
            torch_dtype=torch.float16,
            local_files_only=local_files_only,
        ),
    ]
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        custom_pipeline="stable_diffusion_controlnet_img2img",
        controlnet=controlnets,
        torch_dtype=torch.float16,
        local_files_only=local_files_only,
        requires_safety_checker=False,
        safety_checker=None,
    ).to(torch.device(f"cuda:{args.gpu}"))
    if not args.disable_xformers:
        pipeline.enable_xformers_memory_efficient_attention()

    view_names = ["front", "side", "back", "side"]
    object_dirs = sorted(path for path in args.image_dir.expanduser().iterdir() if path.is_dir())
    for index, object_dir in enumerate(object_dirs, start=1):
        item_id = object_dir.name
        if item_id not in evaluation:
            raise KeyError(f"Missing caption for {item_id!r} in {eval_path}")

        with (object_dir / "cam_dict.pkl").open("rb") as handle:
            camera = pickle.load(handle)
        azimuths = camera["azimuth"]
        if len(azimuths) < args.num_views:
            raise ValueError(f"Camera metadata has fewer than {args.num_views} views: {object_dir}")

        base_prompt = evaluation[item_id]["caption"].split(",", maxsplit=1)[0]
        print(f"[{index}/{len(object_dirs)}] {item_id}")
        for repeat in range(args.repeat_num):
            for view in range(args.num_views):
                view_name = view_names[view_prompt_index(float(azimuths[view]))]
                prompt = f"{base_prompt}, {view_name} view"
                input_path = object_dir / f"gen_{repeat}_{view}.png"
                with Image.open(input_path) as source:
                    image = source.convert("RGB").resize(
                        (args.resolution, args.resolution), Image.Resampling.LANCZOS
                    )

                result = pipeline(
                    prompt,
                    image=image,
                    controlnet_conditioning_image=[image, canny_condition(image)],
                    controlnet_conditioning_scale=[args.tile_scale, args.canny_scale],
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=negative_prompt,
                    width=args.resolution,
                    height=args.resolution,
                ).images[0]
                result.save(object_dir / f"tile_{repeat}_{view}.png")


if __name__ == "__main__":
    main()
