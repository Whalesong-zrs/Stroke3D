#!/usr/bin/env python3
"""Build deterministic TextuRig release shards from a caption ID manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path
from typing import BinaryIO, Iterable


class HashingWriter:
    """File-like writer that hashes bytes while tarfile streams them."""

    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.digest = hashlib.sha256()

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        return self.raw.write(data)

    def tell(self) -> int:
        return self.raw.tell()

    def flush(self) -> None:
        self.raw.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package only IDs present in captions.json. Each sample contributes "
            "one <id>_mesh.glb and one <id>.txt skeleton; renderings are excluded."
        )
    )
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--skeletons-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-shard", type=int, default=500)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing shard files."
    )
    return parser.parse_args()


def load_captions(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        captions = json.load(handle)
    if not isinstance(captions, dict) or not captions:
        raise ValueError("captions must be a non-empty JSON object keyed by sample ID")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in captions.items()):
        raise ValueError("every caption entry must map a string ID to a string caption")
    return captions


def add_regular_file(archive: tarfile.TarFile, source: Path, arcname: str) -> None:
    stat = source.stat()
    info = tarfile.TarInfo(arcname)
    info.size = stat.st_size
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_shard(
    output_path: Path,
    sample_ids: list[str],
    assets_dir: Path,
    skeletons_dir: Path,
) -> tuple[str, int]:
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with partial.open("wb") as raw:
        writer = HashingWriter(raw)
        with tarfile.open(fileobj=writer, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for sample_id in sample_ids:
                add_regular_file(
                    archive,
                    assets_dir / f"{sample_id}_mesh.glb",
                    f"assets/{sample_id}_mesh.glb",
                )
                add_regular_file(
                    archive,
                    skeletons_dir / f"{sample_id}.txt",
                    f"skeletons/{sample_id}.txt",
                )
        checksum = writer.digest.hexdigest()
    os.replace(partial, output_path)
    return checksum, output_path.stat().st_size


def main() -> None:
    args = parse_args()
    if args.samples_per_shard < 1:
        raise ValueError("--samples-per-shard must be positive")

    captions = load_captions(args.captions)
    sample_ids = sorted(captions)
    missing: list[str] = []
    records: list[dict[str, object]] = []
    for sample_id in sample_ids:
        asset = args.assets_dir / f"{sample_id}_mesh.glb"
        skeleton = args.skeletons_dir / f"{sample_id}.txt"
        if not asset.is_file() or not skeleton.is_file():
            missing.append(sample_id)
            continue
        records.append(
            {
                "id": sample_id,
                "caption": captions[sample_id],
                "asset": f"assets/{sample_id}_mesh.glb",
                "asset_bytes": asset.stat().st_size,
                "skeleton": f"skeletons/{sample_id}.txt",
                "skeleton_bytes": skeleton.stat().st_size,
            }
        )
    if missing:
        preview = ", ".join(missing[:20])
        raise FileNotFoundError(f"{len(missing)} caption IDs lack a pair: {preview}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = args.output_dir / "shards"
    shards_dir.mkdir(exist_ok=True)
    with (args.output_dir / "captions.json").open("w", encoding="utf-8") as handle:
        json.dump({key: captions[key] for key in sample_ids}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    record_by_id = {str(record["id"]): record for record in records}
    id_chunks = list(chunks(sample_ids, args.samples_per_shard))
    shard_count = len(id_chunks)
    shard_records: list[dict[str, object]] = []
    checksums: list[tuple[str, str]] = []
    for index, shard_ids in enumerate(id_chunks):
        filename = f"texturig-{index:05d}-of-{shard_count:05d}.tar"
        output_path = shards_dir / filename
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} already exists; use --overwrite after checking it"
            )
        print(f"[{index + 1}/{shard_count}] writing {filename} ({len(shard_ids)} samples)", flush=True)
        checksum, archive_bytes = build_shard(
            output_path, shard_ids, args.assets_dir, args.skeletons_dir
        )
        asset_bytes = sum(int(record_by_id[key]["asset_bytes"]) for key in shard_ids)
        skeleton_bytes = sum(int(record_by_id[key]["skeleton_bytes"]) for key in shard_ids)
        shard_records.append(
            {
                "file": f"shards/{filename}",
                "sha256": checksum,
                "archive_bytes": archive_bytes,
                "sample_count": len(shard_ids),
                "first_id": shard_ids[0],
                "last_id": shard_ids[-1],
                "asset_bytes": asset_bytes,
                "skeleton_bytes": skeleton_bytes,
            }
        )
        checksums.append((checksum, f"shards/{filename}"))

    manifest = {
        "format_version": 1,
        "sample_count": len(records),
        "asset_bytes": sum(int(record["asset_bytes"]) for record in records),
        "skeleton_bytes": sum(int(record["skeleton_bytes"]) for record in records),
        "naming": {
            "asset": "assets/<id>_mesh.glb",
            "skeleton": "skeletons/<id>.txt",
        },
        "shards": shard_records,
        "samples": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    for path in (args.output_dir / "captions.json", manifest_path):
        checksums.append((hashlib.sha256(path.read_bytes()).hexdigest(), path.name))
    with (args.output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for checksum, filename in checksums:
            handle.write(f"{checksum}  {filename}\n")

    total = int(manifest["asset_bytes"]) + int(manifest["skeleton_bytes"])
    print(
        f"complete: {len(records)} samples, {shard_count} shards, "
        f"{total:,} source bytes",
        flush=True,
    )


if __name__ == "__main__":
    main()
