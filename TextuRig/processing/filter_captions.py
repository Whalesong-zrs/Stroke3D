#!/usr/bin/env python3
"""Keep caption entries whose <id>_mesh.glb exists in a selected directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.captions.open(encoding="utf-8") as handle:
        captions = json.load(handle)
    selected_ids = {
        path.name[: -len("_mesh.glb")]
        for path in args.assets_dir.glob("*_mesh.glb")
    }
    filtered = {key: captions[key] for key in sorted(selected_ids) if key in captions}
    missing = sorted(selected_ids.difference(captions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(filtered, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {len(filtered)} captions to {args.output}")
    if missing:
        print(f"warning: {len(missing)} selected assets have no caption")


if __name__ == "__main__":
    main()

