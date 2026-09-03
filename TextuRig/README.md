# TextuRig

TextuRig is the textured, rigged asset collection used to augment SKDream in
Stroke3D. It was curated from the rigged Objaverse-XL subset identified by
UniRig. The released selection contains 6,633 captioned samples.

## Download

The data are hosted in the `texturig/` directory of the
[`zhaors00/Stroke3D`](https://huggingface.co/zhaors00/Stroke3D/tree/main/texturig)
Hugging Face repository. Rendered PNG images are not part of the release.

Each uncompressed tar shard contains:

```text
assets/<id>_mesh.glb
skeletons/<id>.txt
```

`captions.json` maps each ID to its caption, while `manifest.json` records the
member paths, byte sizes, and shard checksums. Extract all shards into the same
directory to reconstruct the layout above.

## Processing pipeline

The scripts under [`processing/`](processing/) are cleaned versions of the
utilities used during curation. All paths are command-line arguments. Caption
generation reads the Gemini key from `GOOGLE_API_KEY`; no credential is stored
in the repository.

Typical stages are:

1. `index_skeletons.py`: replace source joint names with stable `jointN` names.
2. `clean_glbs.py`: keep the principal mesh and armature in each GLB.
3. `filter_textured_glbs.py`: classify assets by texture, vertex color, or
   non-grayscale material color. Inputs are copied by default.
4. `filter_captions.py`: intersect the caption map with the selected GLB IDs.
5. `filter_geometry_pairs.py`: optionally reject pairs whose skeleton bounding
   box exceeds the mesh bounding box.
6. `normalize_pairs.py`: optionally apply the historical skeleton-driven
   centering, scaling, and rotation used for SKDream preparation.
7. `prepare_release.py`: validate the 6,633 pairs and build deterministic tar
   shards without renderings.

Run `python <script> --help` for complete arguments. The Blender controllers
require Blender 3.6 or a compatible release. The non-Blender utilities use the
packages listed in `processing/requirements.txt`.

## Skeleton text format

```text
joints joint0 x y z
joints joint1 x y z
root joint0
hier joint0 joint1
```

## Source assets and licenses

TextuRig is derived from Objaverse-XL assets and the subset identified by
UniRig. Source assets may have different per-object licenses. No blanket license
is asserted over third-party meshes; users are responsible for following the
applicable upstream license and terms for each asset. Stroke3D-authored
processing code is provided for research reproducibility.

