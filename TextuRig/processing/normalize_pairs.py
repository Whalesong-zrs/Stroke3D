#!/usr/bin/env python3
"""Normalize paired skeletons and GLBs using skeleton-derived transforms."""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm


WORKER = Path(__file__).with_name("worker_normalize_mesh.py")


def read_skeleton(path: Path) -> tuple[list[str], np.ndarray, dict[str, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    joints: list[list[float]] = []
    indices: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if parts and parts[0] == "joints":
            indices[parts[1]] = len(joints)
            joints.append([float(value) for value in parts[-3:]])
    if not joints:
        raise ValueError(f"no joints in {path}")
    return lines, np.asarray(joints, dtype=np.float64), indices


def write_skeleton(path: Path, lines: list[str], joints: np.ndarray, indices: dict[str, int]) -> None:
    output = list(lines)
    for line_number, line in enumerate(lines):
        parts = line.split()
        if parts and parts[0] == "joints":
            xyz = joints[indices[parts[1]]]
            output[line_number] = f"joints {parts[1]} {' '.join(map(str, xyz))}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def run_one(task: tuple[Path, Path, Path, Path, Path, float, float]) -> tuple[str, str | None]:
    blender, skeleton, mesh, skeleton_out, mesh_out, skeleton_degrees, mesh_degrees = task
    sample_id = skeleton.stem
    try:
        lines, joints, indices = read_skeleton(skeleton)
        low, high = joints.min(axis=0), joints.max(axis=0)
        shift = (low + high) / 2.0
        largest_side = float(np.max(high - low))
        scale = 1.0 / largest_side if largest_side > 1e-6 else 1.0
        angle = np.deg2rad(skeleton_degrees)
        rotation = np.asarray(
            [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
        )
        write_skeleton(skeleton_out, (lines), (joints - shift) * scale @ rotation, indices)
        command = [
            str(blender), "--background", "--python", str(WORKER), "--",
            "--input", str(mesh), "--output", str(mesh_out),
            "--shift", *[str(value) for value in shift], "--scale", str(scale),
            "--rotation-x-deg", str(mesh_degrees),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            return sample_id, result.stderr[-4000:]
        return sample_id, None
    except Exception as error:
        return sample_id, str(error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--skeletons-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skeleton-rotation-x-deg", type=float, default=90.0)
    parser.add_argument("--mesh-rotation-x-deg", type=float, default=180.0)
    args = parser.parse_args()
    if not args.blender.is_file():
        raise FileNotFoundError(args.blender)
    skeletons_out, assets_out = args.output_dir / "skeletons", args.output_dir / "assets"
    tasks = []
    for skeleton in sorted(args.skeletons_dir.glob("*.txt")):
        mesh = args.assets_dir / f"{skeleton.stem}_mesh.glb"
        if mesh.is_file():
            tasks.append(
                (
                    args.blender, skeleton, mesh, skeletons_out / skeleton.name,
                    assets_out / mesh.name, args.skeleton_rotation_x_deg,
                    args.mesh_rotation_x_deg,
                )
            )
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for sample_id, error in tqdm(executor.map(run_one, tasks), total=len(tasks), desc="pairs"):
            if error:
                failures.append((sample_id, error))
    if failures:
        for sample_id, error in failures:
            print(f"ERROR {sample_id}: {error}")
        raise SystemExit(f"{len(failures)} pairs failed")


if __name__ == "__main__":
    main()

