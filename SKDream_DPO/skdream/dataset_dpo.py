"""Dataset loader for SKA-DPO preference pairs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from skdream.utils.camera import create_camera_to_world_matrix


class SKDreamDatasetDPO(Dataset):
    """Load paired winner/loser multi-view images and skeleton conditions.

    Expected directory layout is documented in ``docs/DATA_FORMAT.md``.
    """

    def __init__(
        self,
        root_dir: str | Path,
        tokenizer: Any,
        cond_channels: int = 4,
        transform: Callable | None = None,
        cond_transform: Callable | None = None,
        num_views: int = 4,
        split: str = "train",
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.num_views = num_views
        self.cond_channels = cond_channels
        self.transform = transform
        self.cond_transform = cond_transform

        if cond_channels not in {1, 3, 4, 5}:
            raise ValueError("cond_channels must be one of 1, 3, 4, or 5")
        if num_views < 1:
            raise ValueError("num_views must be positive")

        split_path = self.root_dir / "train_eval.json"
        caption_path = self.root_dir / "dpo_texturig_captions.json"
        try:
            with split_path.open(encoding="utf-8") as handle:
                splits = json.load(handle)
            with caption_path.open(encoding="utf-8") as handle:
                captions = json.load(handle)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing DPO metadata under {self.root_dir}. "
                "Run tools/validate_dpo_dataset.py for a complete check."
            ) from exc

        if split not in splits:
            raise KeyError(f"Split {split!r} is not present in {split_path}")

        self.file_ids = [str(item_id) for item_id in splits[split]]
        missing_captions = [item_id for item_id in self.file_ids if item_id not in captions]
        if missing_captions:
            preview = ", ".join(missing_captions[:5])
            raise ValueError(f"Missing captions for {len(missing_captions)} IDs: {preview}")

        self.captions = [captions[item_id] for item_id in self.file_ids]
        self.input_ids = self.tokenize_captions(tokenizer, self.captions)

        self.winner_dir = self.root_dir / "win_mv"
        self.loser_dir = self.root_dir / "lose_mv"
        self.condition_dir = self.root_dir / "skeleton_d"
        self.meta_dir = self.root_dir / "meta"

    def __len__(self) -> int:
        return len(self.file_ids)

    @staticmethod
    def _open_image(path: Path) -> Image.Image:
        try:
            with Image.open(path) as image:
                return image.copy()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing dataset image: {path}") from exc

    @staticmethod
    def _rgb_on_gray(image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")

    @staticmethod
    def _binary_condition(image: Image.Image) -> Image.Image:
        gray = np.asarray(image.convert("L"))
        return Image.fromarray(np.where(gray > 1, 255, 0).astype(np.uint8), mode="L")

    def _convert_condition(self, image: Image.Image) -> Image.Image | np.ndarray:
        if self.cond_channels == 4:
            return image.convert("RGBA")
        if self.cond_channels == 3:
            return image.convert("RGB")

        binary = self._binary_condition(image)
        if self.cond_channels == 1:
            return binary

        rgba = np.asarray(image.convert("RGBA"))
        mask = np.asarray(binary)[..., None]
        return np.concatenate([rgba, mask], axis=-1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item_id = self.file_ids[index]
        view_ids = range(self.num_views)

        winners = [
            self._rgb_on_gray(self._open_image(self.winner_dir / item_id / f"gen_{view}.png"))
            for view in view_ids
        ]
        losers = [
            self._rgb_on_gray(self._open_image(self.loser_dir / item_id / f"gen_{view}.png"))
            for view in view_ids
        ]
        conditions = [
            self._convert_condition(
                self._open_image(self.condition_dir / item_id / f"cond_{view}.png")
            )
            for view in view_ids
        ]

        if self.transform is not None:
            winners = [self.transform(image) for image in winners]
            losers = [self.transform(image) for image in losers]
        if self.cond_transform is not None:
            conditions = [self.cond_transform(image) for image in conditions]

        meta_path = self.meta_dir / item_id / "cam_dict.pkl"
        try:
            # This is the historical Stroke3D camera format. Only load metadata
            # from a trusted dataset release because Python pickle is executable.
            with meta_path.open("rb") as handle:
                camera_meta = pickle.load(handle)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing camera metadata: {meta_path}") from exc

        if len(camera_meta.get("elevation", [])) < self.num_views or len(
            camera_meta.get("azimuth", [])
        ) < self.num_views:
            raise ValueError(f"Camera metadata has fewer than {self.num_views} views: {meta_path}")

        cameras = [
            torch.as_tensor(
                create_camera_to_world_matrix(
                    camera_meta["elevation"][view], camera_meta["azimuth"][view], 1
                )
            )
            for view in view_ids
        ]

        return {
            "win_image": torch.stack(winners),
            "lose_image": torch.stack(losers),
            "conditioning_pixel_values": torch.stack(conditions),
            "cameras": torch.stack(cameras),
            "input_ids": self.input_ids[index],
            "caption": self.captions[index],
            "file_id": item_id,
        }

    @staticmethod
    def tokenize_captions(tokenizer: Any, captions: list[str]) -> torch.Tensor:
        return tokenizer(
            text=captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate paired samples without emitting per-batch debug output."""

    return {
        "win_values": torch.stack([item["win_image"] for item in examples])
        .contiguous()
        .float(),
        "lose_values": torch.stack([item["lose_image"] for item in examples])
        .contiguous()
        .float(),
        "conditioning_pixel_values": torch.stack(
            [item["conditioning_pixel_values"] for item in examples]
        )
        .contiguous()
        .float(),
        "cameras": torch.stack([item["cameras"] for item in examples]).contiguous().float(),
        "input_ids": torch.stack([item["input_ids"] for item in examples]),
        "captions": [item["caption"] for item in examples],
        "file_ids": [item["file_id"] for item in examples],
    }
