"""Blender worker used by filter_textured_glbs.py."""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import bpy


def clear_scene() -> None:
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def color_kind() -> str:
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            material = slot.material
            if material and material.use_nodes:
                if any(node.type == "TEX_IMAGE" and node.image for node in material.node_tree.nodes):
                    return "texture"
        color_attributes = getattr(obj.data, "color_attributes", None)
        if color_attributes and len(color_attributes):
            return "vertex_color"
        if getattr(obj.data, "vertex_colors", None) and len(obj.data.vertex_colors):
            return "vertex_color"
        for slot in obj.material_slots:
            material = slot.material
            if not material or not material.use_nodes:
                continue
            node = next(
                (item for item in material.node_tree.nodes if item.type == "BSDF_PRINCIPLED"),
                None,
            )
            if node and node.inputs.get("Base Color"):
                red, green, blue = node.inputs["Base Color"].default_value[:3]
                if max(abs(red - green), abs(green - blue), abs(red - blue)) >= 0.01:
                    return "material_color"
    return "solid_grayscale"


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--textured-dir", required=True)
    parser.add_argument("--plain-dir", required=True)
    parser.add_argument("--operation", choices=("copy", "move"), default="copy")
    args = parser.parse_args(argv)
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=args.input)
    output_dir = args.plain_dir if color_kind() == "solid_grayscale" else args.textured_dir
    os.makedirs(output_dir, exist_ok=True)
    destination = os.path.join(output_dir, os.path.basename(args.input))
    if args.operation == "move":
        shutil.move(args.input, destination)
    else:
        shutil.copy2(args.input, destination)


if __name__ == "__main__":
    main()
