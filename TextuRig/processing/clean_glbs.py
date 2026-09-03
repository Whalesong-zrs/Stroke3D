#!/usr/bin/env python3
"""Run the per-file GLB cleaner through isolated Blender processes."""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm


WORKER = Path(__file__).with_name("worker_clean_glb.py")


def run_one(task: tuple[Path, Path, Path]) -> tuple[str, str | None]:
    blender, source, destination = task
    command = [
        str(blender), "--background", "--python", str(WORKER), "--",
        "--input", str(source), "--output", str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return source.name, None if result.returncode == 0 else result.stderr[-4000:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.blender.is_file():
        raise FileNotFoundError(args.blender)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for source in sorted(args.input_dir.glob("*.glb")):
        destination = args.output_dir / source.name
        if args.overwrite or not destination.exists():
            tasks.append((args.blender, source, destination))
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

