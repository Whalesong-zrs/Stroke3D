# Stroke3D: SKA-DPO for SKDream

This repository contains the SKDream preference-optimization component of
**Stroke3D: Lifting 2D Strokes into Rigged 3D Model via Latent Diffusion Models**
(ICLR 2026). SKA-DPO improves skeleton-conditioned multi-view generation by ranking
sample pairs with a skeleton-image alignment (SKA) scorer and applying a diffusion
DPO objective.

[[Paper](https://openreview.net/forum?id=VgOWxor3LV)]
[[Project page](https://whalesong-zrs.github.io/Stroke3D_project_page/)]
[[Full Stroke3D code](https://github.com/Whalesong-zrs/Stroke3D)]

## Scope

This module contains the reproducible SKA-DPO path and the original downstream
SKDream mesh-synthesis components:

- SKDream model and multi-view attention modules;
- preference-pair construction with a configurable SKA margin;
- the SKA scorer and evaluation script;
- the paired dataset loader and diffusion-DPO trainer;
- skeleton-conditioned multi-view inference;
- InstantMesh-based multi-view reconstruction;
- optional tile upscaling and UV/texture refinement;
- the historical `objsk_eval2` evaluation inputs.

Checkpoints and generated training data are not committed to this Git repository;
the released files are linked below.
The mesh code is kept for end-to-end SKDream compatibility, but has additional
third-party dependencies and licensing notices; see [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Released assets

- [Final SKDream/SKA-DPO ControlNet checkpoint](https://huggingface.co/zhaors00/Stroke3D/tree/main/ckpt/SKDream-SKA-DPO)
- [2,000-pair SKA-DPO training data](https://huggingface.co/zhaors00/Stroke3D/tree/main/data/DPO-Data)

The DPO archive contains four preferred, four rejected, and four skeleton-condition
PNG images per example, plus captions and camera metadata. It does not contain the
upstream OBJ assets. Download it with:

```bash
hf download zhaors00/Stroke3D \
  data/DPO-Data/stroke3d_ska_dpo_margin_0.10.tar \
  --local-dir stroke3d-assets
```

## Environment

The experiments were developed with Python 3.10, PyTorch 2.0.1, and CUDA. Create an
isolated environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For InstantMesh reconstruction and texture refinement, install the additional
dependencies in `requirements-mesh.txt`. The differentiable renderer also requires
CUDA-compatible builds of
[`nvdiffrast`](https://github.com/NVlabs/nvdiffrast) and
[`tiny-cuda-nn`](https://github.com/NVlabs/tiny-cuda-nn).

The pipeline uses the MVDream Diffusers checkpoint
[`lzq49/mvdream-sd21-diffusers`](https://huggingface.co/lzq49/mvdream-sd21-diffusers),
an SFT SKDream controlnet, the
[`dinov2_vitl14_reg`](https://github.com/facebookresearch/dinov2) checkpoint, and the
SKA scorer checkpoint. The Stroke3D data and final SKA-DPO weights are organized under
the `data/` and `ckpt/` folders in the single Hugging Face repository
[`zhaors00/Stroke3D`](https://huggingface.co/zhaors00/Stroke3D). Review the component
cards and upstream licenses before use.

## 1. Build preference pairs

The input directory must contain `eval.json` and `cano_sk/<id>.txt`. Each `eval.json`
entry needs a `caption`; optional `elevation` and `azimuth` values control the first
view. The paper uses a score margin of **0.10**, with up to three sampling attempts.

```bash
export DATA_DIR=/path/to/skeleton_caption_inputs
export OUTPUT_DIR=/path/to/ska_dpo_pairs
export SFT_CONTROLNET=/path/to/sft_skdream
export DINO_CHECKPOINT=/path/to/dinov2_vitl14_reg4_pretrain.pth
export SKA_CHECKPOINT=/path/to/cosa.pth
bash scripts/prepare_dpo_data.sh
```

To reproduce an ablation without keeping duplicate scripts, override `MARGIN`, for
example `MARGIN=0.15 bash scripts/prepare_dpo_data.sh`. See
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) for the generated layout.

Validate the dataset before training:

```bash
python tools/validate_dpo_dataset.py /path/to/ska_dpo_pairs
```

## 2. Train SKA-DPO

The wrapper contains the paper configuration: 1,000 optimization steps, learning rate
`5e-6`, four views, batch size 4, gradient accumulation 2, and checkpoints every 500
steps.

```bash
export DPO_DATA_DIR=/path/to/ska_dpo_pairs
export SFT_CONTROLNET=/path/to/sft_skdream
export OUTPUT_DIR=outputs/ska_dpo_margin_0.10
bash scripts/train_ska_dpo.sh
```

The trainer writes resumable checkpoints to `checkpoint-<step>/` and exports the final
inference-only controlnet to `<OUTPUT_DIR>/skdream/`. `--beta_dpo` controls the
diffusion-DPO inverse temperature and defaults to the experimental value `5000`.

## 3. Inference and evaluation

```bash
python infer_mv.py \
  --data-dir /path/to/eval_inputs \
  --output-dir eval/ska_dpo \
  --controlnet zhaors00/Stroke3D \
  --controlnet-subfolder ckpt/SKDream-SKA-DPO

export EVAL_JSON=/path/to/eval_inputs/eval.json
export IMAGE_DIR=eval/ska_dpo
export DINO_CHECKPOINT=/path/to/dinov2_vitl14_reg4_pretrain.pth
export SKA_CHECKPOINT=/path/to/cosa.pth
bash scripts/evaluate_ska.sh
```

Set `DINOV2_REPO=/path/to/dinov2` in the preparation or evaluation wrappers to use a
local DINOv2 checkout instead of downloading its Torch Hub definition.

## 4. Mesh reconstruction and texture refinement

The optional downstream path converts SKDream multi-view images into a coarse mesh,
upscales the views with Tile/Canny ControlNet, and optimizes UV textures. The restored
historical entry points are:

- `infer_rec.py`: InstantMesh reconstruction;
- `infer_tile.py`: view-aware image upscaling;
- `infer_refine.py` and `uv_refine.py`: differentiable texture refinement;
- `infer_mesh_{origin,sft,dpo,sft_dpo}.sh`: compatibility aliases for the common
  wrapper below.

The bundled InstantMesh tree preserves the downstream reconstruction code from
the historical project backup. UV refinement receives the repeat index through
`--repeat_idx`, so object IDs containing underscores and repeat counts above ten
are handled without path rewriting.

Configure all machine-specific paths through environment variables:

```bash
export DATA_DIR=/path/to/eval_inputs
export MV_DIR=/path/to/skdream_multiview_outputs
export COARSE_DIR=/path/to/coarse_meshes
export REFINE_DIR=/path/to/refined_meshes
export GPU=0
export REPEAT_NUM=4
bash scripts/run_mesh_pipeline.sh
```

`DATA_DIR` must contain `eval.json`; the included `objsk_eval2/` directory provides
the historical evaluation skeletons and metadata. Model locations in
`config/instant-mesh-large.yaml` and the Tile/Canny checkpoint arguments of
`infer_tile.py` may be overridden for a local installation. Set any of
`RUN_RECONSTRUCTION`, `RUN_TILING`, or `RUN_REFINEMENT` to `0` to skip that stage.

## Citation

```bibtex
@inproceedings{zhao2026stroke3d,
  title     = {Stroke3D: Lifting 2D Strokes into Rigged 3D Model via Latent Diffusion Models},
  author    = {Zhao, Ruisi and Zheng, Haoren and Yang, Zongxin and Fan, Hehe and Yang, Yi},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=VgOWxor3LV}
}
```

## Acknowledgements

This implementation builds on SKDream, MVDream, Hugging Face Diffusers, DINOv2, and
the Diffusion-DPO training formulation. See [`THIRD_PARTY.md`](THIRD_PARTY.md) for
upstream links and licensing notes.
