#!/usr/bin/env python3
"""Copy mesh/skeleton pairs whose skeleton fits within the mesh bounding box."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm


def skeleton_extent(path: Path) -> float:
    joints = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] == "joints":
            joints.append([float(value) for value in parts[-3:]])
    if not joints:
        raise ValueError("skeleton has no joints")
    xyz = np.asarray(joints, dtype=np.float64)
    return float(np.max(xyz.max(axis=0) - xyz.min(axis=0)))


def mesh_extent(path: Path) -> float:
    loaded = trimesh.load(path, force="mesh", process=False)
    return float(np.max(loaded.bounds[1] - loaded.bounds[0]))


def check_pair(task: tuple[str, Path, Path]) -> tuple[str, bool, str | None]:
    sample_id, mesh, skeleton = task
    try:
        return sample_id, skeleton_extent(skeleton) <= mesh_extent(mesh), None
    except Exception as error:  # report corrupt inputs without aborting the batch
        return sample_id, False, str(error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--skeletons-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--copy-skeletons", action="store_true")
    args = parser.parse_args()

    tasks = []
    for skeleton in sorted(args.skeletons_dir.glob("*.txt")):
        sample_id = skeleton.stem
        mesh = args.assets_dir / f"{sample_id}_mesh.glb"
        if mesh.is_file():
            tasks.append((sample_id, mesh, skeleton))
    if not tasks:
        raise FileNotFoundError("no matching <id>_mesh.glb and <id>.txt pairs")

    assets_out = args.output_dir / "assets"
    skeletons_out = args.output_dir / "skeletons"
    assets_out.mkdir(parents=True, exist_ok=True)
    if args.copy_skeletons:
        skeletons_out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"accepted": [], "rejected": [], "errors": {}}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for sample_id, accepted, error in tqdm(
            executor.map(check_pair, tasks), total=len(tasks), desc="pairs"
        ):
            if error:
                report["errors"][sample_id] = error  # type: ignore[index]
            elif accepted:
                report["accepted"].append(sample_id)  # type: ignore[union-attr]
                shutil.copy2(args.assets_dir / f"{sample_id}_mesh.glb", assets_out)
                if args.copy_skeletons:
                    shutil.copy2(args.skeletons_dir / f"{sample_id}.txt", skeletons_out)
            else:
                report["rejected"].append(sample_id)  # type: ignore[union-attr]
    with (args.output_dir / "filter_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(
        f"accepted={len(report['accepted'])}, rejected={len(report['rejected'])}, "
        f"errors={len(report['errors'])}"
    )


if __name__ == "__main__":
    main()
