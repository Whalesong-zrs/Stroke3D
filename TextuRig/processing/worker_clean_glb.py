"""Blender worker used by clean_glbs.py; run through Blender, not CPython."""

from __future__ import annotations

import argparse
import os
import sys

import bpy


def reset_scene() -> None:
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def main_mesh():
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    return max(meshes, key=lambda obj: len(obj.data.vertices), default=None)


def process(source: str, destination: str) -> None:
    reset_scene()
    bpy.ops.import_scene.gltf(filepath=source)
    principal = main_mesh()
    if principal is None:
        raise ValueError("GLB contains no mesh")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj != principal and obj.type != "ARMATURE":
            obj.select_set(True)
    bpy.ops.object.delete()

    bpy.context.view_layer.objects.active = principal
    principal.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_linked(delimit=set())
    bpy.ops.mesh.select_all(action="INVERT")
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    principal.location = (0, 0, 0)
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            obj.location = (0, 0, 0)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=destination, use_selection=True, export_apply=False, export_format="GLB"
    )


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    process(args.input, args.output)


if __name__ == "__main__":
    main()

