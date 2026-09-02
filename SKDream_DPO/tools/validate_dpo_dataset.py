#!/usr/bin/env python3
"""Validate an on-disk SKA-DPO dataset and optionally write a manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Write the validation summary as JSON")
    return parser.parse_args()


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def directory_ids(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {entry.name for entry in path.iterdir() if entry.is_dir()}


def tree_stats(root: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for current_root, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(current_root) / filename
            file_count += 1
            total_bytes += path.stat().st_size
    return file_count, total_bytes


def main() -> None:
    args = parse_args()
    root = args.data_dir.expanduser().resolve()
    errors: list[str] = []

    required_json = [root / "train_eval.json", root / "dpo_texturig_captions.json"]
    for path in required_json:
        if not path.is_file():
            errors.append(f"missing metadata file: {path.name}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)

    splits = load_json(required_json[0])
    captions = load_json(required_json[1])
    if "train" not in splits or not isinstance(splits["train"], list):
        errors.append("train_eval.json must contain a list named 'train'")
        train_ids: list[str] = []
    else:
        train_ids = [str(item_id) for item_id in splits["train"]]

    if len(train_ids) != len(set(train_ids)):
        errors.append("train split contains duplicate IDs")
    train_set = set(train_ids)
    caption_set = set(captions)
    missing_captions = sorted(train_set - caption_set)
    extra_captions = sorted(caption_set - train_set)
    if missing_captions:
        errors.append(f"missing captions for {len(missing_captions)} training IDs")
    if extra_captions:
        errors.append(f"captions contain {len(extra_captions)} IDs outside the train split")

    layouts = {
        "win_mv": "gen_{view}.png",
        "lose_mv": "gen_{view}.png",
        "skeleton_d": "cond_{view}.png",
        "meta": "cam_dict.pkl",
    }
    for directory, pattern in layouts.items():
        directory_path = root / directory
        ids = directory_ids(directory_path)
        missing_ids = sorted(train_set - ids)
        extra_ids = sorted(ids - train_set)
        if missing_ids:
            errors.append(f"{directory}: missing {len(missing_ids)} sample directories")
        if extra_ids:
            errors.append(f"{directory}: contains {len(extra_ids)} unexpected sample directories")

        for item_id in train_ids:
            item_dir = directory_path / item_id
            if directory == "meta":
                expected = [item_dir / pattern]
            else:
                expected = [item_dir / pattern.format(view=view) for view in range(args.num_views)]
            missing_files = [path.name for path in expected if not path.is_file()]
            if missing_files:
                errors.append(f"{directory}/{item_id}: missing {', '.join(missing_files)}")
            if len(errors) >= 100:
                errors.append("stopped after 100 validation errors")
                break
        if len(errors) >= 100:
            break

    file_count, total_bytes = tree_stats(root)
    summary = {
        "dataset_root": str(root),
        "train_samples": len(train_ids),
        "eval_samples": len(splits.get("eval", [])),
        "num_views": args.num_views,
        "files": file_count,
        "bytes": total_bytes,
        "validation": "ok" if not errors else "failed",
        "errors": errors,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
