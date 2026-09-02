#!/usr/bin/env python3
"""Generate skeleton-conditioned multi-view images with SKDream."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import skimage.io as io
import torch
import torchvision.transforms.functional as transform_functional
from PIL import Image
from rembg import new_session, remove

import skeleton.render_one_pose as skeleton_renderer
from skdream.pipeline_skdream import load_skdream_pipeline
from skdream.utils.camera import create_camera_to_world_matrix
from skdream.utils.projection import perspective, rotate_x, rotate_y, translate


DEFAULT_NEGATIVE_PROMPT = (
    "low poly, white model, noise, strange color, ugly, oversaturated, doubled face, "
    "black and white, sepia, freckles, paintings, sketches, worst quality, low quality, "
    "low resolution, monochrome, grayscale, error, blurry, artifacts"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controlnet", required=True, help="SKDream checkpoint or Hub ID")
    parser.add_argument(
        "--base-model",
        default="lzq49/mvdream-sd21-diffusers",
        help="MVDream Diffusers checkpoint or Hub ID",
    )
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def camera_bundle(elevation: float, azimuth: float, num_views: int) -> dict:
    distance = 2.5
    projection = perspective(np.deg2rad(30), 1.0, 0.5, 1000)
    result = {
        "mv": [],
        "mvp": [],
        "campos": [],
        "c2w": [],
        "elevation": [],
        "azimuth": [],
        "distance": [],
    }
    for view in range(num_views):
        view_azimuth = azimuth + view * (360.0 / num_views)
        model_view = translate(0, 0, -distance) @ (
            rotate_x(-np.deg2rad(elevation)) @ rotate_y(-np.deg2rad(view_azimuth))
        )
        result["mv"].append(model_view)
        result["mvp"].append(projection @ model_view)
        result["campos"].append(torch.linalg.inv(model_view)[:3, 3])
        result["c2w"].append(
            torch.as_tensor(
                create_camera_to_world_matrix(elevation, view_azimuth, 1),
                dtype=torch.float32,
            )
        )
        result["elevation"].append(elevation)
        result["azimuth"].append(view_azimuth)
        result["distance"].append(distance)
    for key in ("mv", "mvp", "campos", "c2w"):
        result[key] = torch.stack(result[key])
    return result


def render_conditions(skeleton_file: Path, camera: dict, output_dir: Path) -> list[Image.Image]:
    joints, bones, parts = skeleton_renderer.get_skeleton_info(str(skeleton_file))
    joints = torch.as_tensor(joints)
    projected, depth = skeleton_renderer.project_joints(joints, camera["mvp"])
    conditions = []
    for view in range(camera["mvp"].shape[0]):
        sorted_bones = skeleton_renderer.sort_bones_depth(depth[view], bones)
        canvas = skeleton_renderer.draw_ccm_with_depth(
            np.zeros((512, 512, 3), dtype=np.uint8),
            joints,
            projected[view],
            sorted_bones,
            parts,
            skeleton_renderer.process_depth(depth[view]),
        )
        condition = Image.fromarray(canvas, mode="RGBA").resize(
            (256, 256), Image.Resampling.NEAREST
        )
        condition.save(output_dir / f"cond_{view}.png")
        conditions.append(condition)
    return conditions


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    with (args.data_dir / "eval.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    item_ids = sorted(metadata)
    if args.limit is not None:
        item_ids = item_ids[: args.limit]

    pipeline = load_skdream_pipeline(
        pretrained_controlnet_name_or_path=args.controlnet,
        pretrained_model_name_or_path=args.base_model,
        num_views=args.num_views,
        weights_dtype=torch.float16,
        device=device,
    )
    rembg_session = new_session("isnet-general-use")
    rng = np.random.default_rng(args.seed)

    for item_index, item_id in enumerate(item_ids):
        item_dir = args.output_dir / item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        expected = [
            item_dir / f"gen_{repeat}_{view}.png"
            for repeat in range(args.num_repeats)
            for view in range(args.num_views)
        ]
        if all(path.is_file() for path in expected):
            print(f"Skipping complete item: {item_id}")
            continue

        item_meta = metadata[item_id]
        elevation = float(item_meta.get("elevation", rng.integers(0, 30)))
        azimuth = float(item_meta.get("azimuth", rng.integers(0, 360)))
        camera = camera_bundle(elevation, azimuth, args.num_views)
        with (item_dir / "cam_dict.pkl").open("wb") as handle:
            pickle.dump(camera, handle)
        conditions = render_conditions(
            args.data_dir / "cano_sk" / f"{item_id}.txt", camera, item_dir
        )
        condition_tensor = torch.stack(
            [transform_functional.to_tensor(image) for image in conditions]
        ).unsqueeze(0)
        condition_tensor = condition_tensor.to(device) * 2 - 1
        camera_to_world = camera["c2w"].reshape(1, args.num_views, -1).to(device)

        for repeat in range(args.num_repeats):
            generator = torch.Generator(device=device).manual_seed(
                args.seed + item_index * args.num_repeats + repeat
            )
            images = pipeline(
                prompt=item_meta["caption"],
                negative_prompt=args.negative_prompt or None,
                hint=condition_tensor,
                c2ws=camera_to_world,
                guidance_scale=7.5,
                controlnet_conditioning_scale=args.conditioning_scale,
                guess_mode=False,
                blind_control_until_step=None,
                output_type="numpy",
                generator=generator,
            ).images
            images = np.clip(images * 255, 0, 255).astype(np.uint8)
            for view, image in enumerate(images):
                mask = remove(image, only_mask=True, session=rembg_session)
                mask = np.asarray(mask)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                rgba = np.concatenate([image, mask[..., None]], axis=-1)
                io.imsave(
                    item_dir / f"gen_{repeat}_{view}.png", rgba, check_contrast=False
                )
        print(f"Generated: {item_id}")


if __name__ == "__main__":
    main()
