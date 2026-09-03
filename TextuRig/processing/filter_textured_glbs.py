#!/usr/bin/env python3
"""Classify GLBs as textured/color-bearing or solid grayscale using Blender."""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm


WORKER = Path(__file__).with_name("worker_filter_texture.py")


def run_one(task: tuple[Path, Path, Path, Path, bool]) -> tuple[str, str | None]:
    blender, source, textured_dir, plain_dir, move = task
    command = [
        str(blender), "--background", "--python", str(WORKER), "--",
        "--input", str(source), "--textured-dir", str(textured_dir),
        "--plain-dir", str(plain_dir), "--operation", "move" if move else "copy",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return source.name, None if result.returncode == 0 else result.stderr[-4000:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--textured-dir", type=Path, required=True)
    parser.add_argument("--plain-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--move", action="store_true", help="Move inputs instead of the safe default: copy."
    )
    args = parser.parse_args()
    if not args.blender.is_file():
        raise FileNotFoundError(args.blender)
    args.textured_dir.mkdir(parents=True, exist_ok=True)
    args.plain_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (args.blender, source, args.textured_dir, args.plain_dir, args.move)
        for source in sorted(args.input_dir.glob("*.glb"))
    ]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for filename, error in tqdm(executor.map(run_one, tasks), total=len(tasks), desc="GLBs"):
            if error:
                failures.append((filename, error))
    if failures:
        for filename, error in failures:
            print(f"ERROR {filename}: {error}")
        raise SystemExit(f"{len(failures)} GLBs failed")


if __name__ == "__main__":
    main()

