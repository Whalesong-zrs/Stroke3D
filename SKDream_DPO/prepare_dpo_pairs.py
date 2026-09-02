#!/usr/bin/env python3
"""Construct SKA-ranked preference pairs for SKDream DPO training."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import skimage.io as io
import torch
import torchvision.transforms as transforms
from PIL import Image
from rembg import new_session, remove
from tqdm import tqdm

import skeleton.render_one_pose as skeleton_renderer
from skalign.model import SkalignModel
from skdream.pipeline_skdream import load_skdream_pipeline
from skdream.utils.camera import create_camera_to_world_matrix
from skdream.utils.projection import perspective, rotate_x, rotate_y, translate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate two SKDream samples per skeleton and rank them with SKA."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controlnet", required=True, help="SFT SKDream checkpoint or Hub ID")
    parser.add_argument(
        "--base-model",
        default="lzq49/mvdream-sd21-diffusers",
        help="MVDream Diffusers checkpoint or Hub ID",
    )
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--ska-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dinov2-repo",
        type=Path,
        help="Optional local DINOv2 checkout. Omit to load facebookresearch/dinov2 from GitHub.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--score-rembg-model", default="u2net")
    parser.add_argument("--output-rembg-model", default="isnet-general-use")
    parser.add_argument(
        "--limit", type=int, help="Only process the first N IDs (useful for a smoke test)."
    )
    args = parser.parse_args()

    if args.num_views < 1:
        parser.error("--num-views must be positive")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if args.margin < 0:
        parser.error("--margin cannot be negative")
    return args


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def stable_seed(base_seed: int, item_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def generate_conditions_and_camera(
    item_id: str,
    skeleton_file: Path,
    item_meta: dict,
    output_dir: Path,
    num_views: int,
    rng: np.random.Generator,
) -> tuple[list[Image.Image], dict]:
    condition_dir = output_dir / "skeleton_d" / item_id
    meta_dir = output_dir / "meta" / item_id
    condition_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    elevation = item_meta.get("elevation")
    azimuth = item_meta.get("azimuth")
    elevation = int(rng.integers(0, 30)) if elevation is None else float(elevation)
    azimuth = int(rng.integers(0, 360)) if azimuth is None else float(azimuth)
    distance = 2.5
    projection = perspective(np.deg2rad(30), 1.0, 0.5, 1000)
    camera = {
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
        camera["mv"].append(model_view)
        camera["mvp"].append(projection @ model_view)
        camera["campos"].append(torch.linalg.inv(model_view)[:3, 3])
        camera["c2w"].append(
            torch.as_tensor(
                create_camera_to_world_matrix(elevation, view_azimuth, cam_dist=1),
                dtype=torch.float32,
            )
        )
        camera["elevation"].append(elevation)
        camera["azimuth"].append(view_azimuth)
        camera["distance"].append(distance)

    for key in ("mv", "mvp", "campos", "c2w"):
        camera[key] = torch.stack(camera[key])
    with (meta_dir / "cam_dict.pkl").open("wb") as handle:
        pickle.dump(camera, handle)

    joints, bones, parts = skeleton_renderer.get_skeleton_info(str(skeleton_file))
    joints = torch.as_tensor(joints)
    projected, depth = skeleton_renderer.project_joints(joints, camera["mvp"])
    conditions = []
    for view in range(num_views):
        sorted_bones = skeleton_renderer.sort_bones_depth(depth[view], bones)
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        canvas = skeleton_renderer.draw_ccm_with_depth(
            canvas,
            joints,
            projected[view],
            sorted_bones,
            parts,
            skeleton_renderer.process_depth(depth[view]),
        )
        image = Image.fromarray(canvas, mode="RGBA").resize(
            (256, 256), resample=Image.Resampling.NEAREST
        )
        image.save(condition_dir / f"cond_{view}.png")
        conditions.append(image)

    return conditions, camera


def generate_samples(
    pipeline,
    prompt: str,
    conditions: list[Image.Image],
    camera_to_world: torch.Tensor,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> np.ndarray:
    condition = torch.stack(
        [transforms.functional.to_tensor(image.convert("RGBA")) for image in conditions]
    ).unsqueeze(0)
    condition = condition.to(pipeline.device) * 2 - 1
    images = pipeline(
        prompt=prompt,
        negative_prompt=args.negative_prompt or None,
        hint=condition,
        c2ws=camera_to_world.reshape(1, args.num_views, -1).to(pipeline.device),
        guidance_scale=7.5,
        controlnet_conditioning_scale=args.conditioning_scale,
        guess_mode=False,
        blind_control_until_step=None,
        output_type="numpy",
        generator=generator,
    ).images
    return np.clip(images * 255, 0, 255).astype(np.uint8)


def alignment_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


def evaluate_alignment(
    generated: np.ndarray,
    conditions: list[Image.Image],
    dino,
    ska_model,
    transform,
    device: torch.device,
    rembg_session,
) -> float:
    foreground_images = []
    for array in generated:
        image = Image.fromarray(array)
        mask = remove(image, only_mask=True, session=rembg_session)
        foreground = Image.new("RGB", image.size)
        foreground.paste(image, mask=mask)
        foreground_images.append(transform(foreground))

    image_tensor = torch.stack(foreground_images).to(device)
    condition_tensor = torch.stack(
        [transform(image.convert("RGB")) for image in conditions]
    ).to(device)
    with torch.inference_mode():
        image_features = ska_model(dino(image_tensor, return_ret=True)["x_norm_patchtokens"])
        condition_features = ska_model(
            dino(condition_tensor, return_ret=True)["x_norm_patchtokens"]
        )
        score = torch.cosine_similarity(image_features, condition_features, dim=-1).mean()
    return float(score)


def load_alignment_models(args: argparse.Namespace, device: torch.device):
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

    ska_model = SkalignModel(1024, 3)
    ska_model.load_state_dict(torch.load(args.ska_checkpoint, map_location="cpu"))
    ska_model.requires_grad_(False).eval().to(device)
    return dino, ska_model


def save_image_set(images: np.ndarray, output_dir: Path, rembg_session) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for view, image in enumerate(images):
        rgba = remove(image, session=rembg_session)
        io.imsave(output_dir / f"gen_{view}.png", np.asarray(rgba), check_contrast=False)


def sample_is_complete(output_dir: Path, item_id: str, num_views: int) -> bool:
    expected = [output_dir / "meta" / item_id / "cam_dict.pkl"]
    expected += [output_dir / "win_mv" / item_id / f"gen_{view}.png" for view in range(num_views)]
    expected += [output_dir / "lose_mv" / item_id / f"gen_{view}.png" for view in range(num_views)]
    expected += [
        output_dir / "skeleton_d" / item_id / f"cond_{view}.png"
        for view in range(num_views)
    ]
    return all(path.is_file() for path in expected)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("win_mv", "lose_mv", "skeleton_d", "meta"):
        (args.output_dir / directory).mkdir(exist_ok=True)

    evaluation = load_json(args.data_dir / "eval.json")
    item_ids = sorted(evaluation)
    if args.limit is not None:
        item_ids = item_ids[: args.limit]

    captions = {item_id: evaluation[item_id]["caption"] for item_id in item_ids}
    write_json(args.output_dir / "dpo_texturig_captions.json", captions)
    write_json(args.output_dir / "train_eval.json", {"train": item_ids, "eval": []})
    score_path = args.output_dir / "pair_scores.json"
    scores = load_json(score_path) if score_path.exists() else {}

    pipeline = load_skdream_pipeline(
        pretrained_controlnet_name_or_path=args.controlnet,
        pretrained_model_name_or_path=args.base_model,
        num_views=args.num_views,
        weights_dtype=torch.float16,
        device=device,
    )
    dino, ska_model = load_alignment_models(args, device)
    transform = alignment_transform()
    score_rembg_session = new_session(args.score_rembg_model)
    output_rembg_session = new_session(args.output_rembg_model)

    margin_count = 0
    progress = tqdm(item_ids, desc="Preparing SKA-DPO pairs")
    for item_id in progress:
        if sample_is_complete(args.output_dir, item_id, args.num_views) and item_id in scores:
            margin_count += bool(scores[item_id].get("margin_met"))
            continue

        item_seed = stable_seed(args.seed, item_id)
        rng = np.random.default_rng(item_seed)
        conditions, camera = generate_conditions_and_camera(
            item_id,
            args.data_dir / "cano_sk" / f"{item_id}.txt",
            evaluation[item_id],
            args.output_dir,
            args.num_views,
            rng,
        )

        for attempt in range(1, args.max_attempts + 1):
            generator_a = torch.Generator(device=device).manual_seed(item_seed + 2 * attempt)
            generator_b = torch.Generator(device=device).manual_seed(item_seed + 2 * attempt + 1)
            sample_a = generate_samples(
                pipeline, captions[item_id], conditions, camera["c2w"], args, generator_a
            )
            sample_b = generate_samples(
                pipeline, captions[item_id], conditions, camera["c2w"], args, generator_b
            )
            score_a = evaluate_alignment(
                sample_a,
                conditions,
                dino,
                ska_model,
                transform,
                device,
                score_rembg_session,
            )
            score_b = evaluate_alignment(
                sample_b,
                conditions,
                dino,
                ska_model,
                transform,
                device,
                score_rembg_session,
            )
            difference = abs(score_a - score_b)
            progress.set_postfix(
                item=item_id[:8], attempt=attempt, score_gap=f"{difference:.4f}"
            )
            if difference >= args.margin:
                break

        margin_met = difference >= args.margin
        margin_count += margin_met
        winner, loser = (sample_a, sample_b) if score_a >= score_b else (sample_b, sample_a)
        save_image_set(winner, args.output_dir / "win_mv" / item_id, output_rembg_session)
        save_image_set(loser, args.output_dir / "lose_mv" / item_id, output_rembg_session)

        scores[item_id] = {
            "attempts": attempt,
            "candidate_a_ska": score_a,
            "candidate_b_ska": score_b,
            "score_gap": difference,
            "requested_margin": args.margin,
            "margin_met": margin_met,
        }
        write_json(score_path, scores)

    print(
        f"Prepared {len(item_ids)} pairs; {margin_count} met the requested "
        f"SKA margin of {args.margin:.3f}."
    )


if __name__ == "__main__":
    main()
