"""Blender worker used by normalize_pairs.py."""

from __future__ import annotations

import argparse
import math
import os
import sys

import bpy


def apply_parent_transform(objects, location=(0, 0, 0), scale=(1, 1, 1), rotation_x=0.0):
    parent = bpy.data.objects.new("Stroke3DTransform", None)
    bpy.context.collection.objects.link(parent)
    parent.location = location
    parent.scale = scale
    parent.rotation_euler[0] = rotation_x
    for obj in objects:
        obj.parent = parent
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.data.objects.remove(parent, do_unlink=True)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shift", nargs=3, type=float, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--rotation-x-deg", type=float, default=180.0)
    args = parser.parse_args(argv)

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=args.input)
    roots = [
        obj for obj in bpy.context.scene.objects
        if obj.type not in {"CAMERA", "LIGHT"} and obj.parent is None
    ]
    if not roots:
        raise ValueError("GLB contains no transformable root objects")
    apply_parent_transform(roots, location=tuple(-value for value in args.shift))
    roots = [
        obj for obj in bpy.context.scene.objects
        if obj.type not in {"CAMERA", "LIGHT"} and obj.parent is None
    ]
    apply_parent_transform(
        roots,
        scale=(args.scale,) * 3,
        rotation_x=math.radians(args.rotation_x_deg),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=args.output, use_selection=False, export_format="GLB")


if __name__ == "__main__":
    main()

