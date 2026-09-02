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

The public code is intentionally limited to the reproducible SKA-DPO path:

- SKDream model and multi-view attention modules;
- preference-pair construction with a configurable SKA margin;
- the SKA scorer and evaluation script;
- the paired dataset loader and diffusion-DPO trainer;
- skeleton-conditioned multi-view inference.

Mesh reconstruction, texture refinement, embedded evaluation assets, checkpoints,
and generated data are not committed to this code repository.

## Environment

The experiments were developed with Python 3.10, PyTorch 2.0.1, and CUDA. Create an
isolated environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pipeline uses the MVDream Diffusers checkpoint
[`lzq49/mvdream-sd21-diffusers`](https://huggingface.co/lzq49/mvdream-sd21-diffusers),
an SFT SKDream controlnet, the
[`dinov2_vitl14_reg`](https://github.com/facebookresearch/dinov2) checkpoint, and the
SKA scorer checkpoint. The Stroke3D data and final SKA-DPO weights will be distributed
separately after their release metadata and licenses have been reviewed.

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
  --controlnet /path/to/final_skdream

export EVAL_JSON=/path/to/eval_inputs/eval.json
export IMAGE_DIR=eval/ska_dpo
export DINO_CHECKPOINT=/path/to/dinov2_vitl14_reg4_pretrain.pth
export SKA_CHECKPOINT=/path/to/cosa.pth
bash scripts/evaluate_ska.sh
```

Set `DINOV2_REPO=/path/to/dinov2` in the preparation or evaluation wrappers to use a
local DINOv2 checkout instead of downloading its Torch Hub definition.

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
