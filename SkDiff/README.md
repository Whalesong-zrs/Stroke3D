# SkDiff

SkDiff is the Stroke3D skeleton pipeline for lifting a 2D stroke projection to
a rigged 3D skeleton. It contains the released Sk-VAE and Sk-DiT inference
code, together with the historical training and ControlNet utilities.

## 1. Install the environment

Run the commands below from this directory. Install a CUDA-compatible PyTorch
and PyTorch Geometric build for your system first, then install the remaining
dependencies:

```bash
cd SkDiff
pip install -r requirements.txt
```

The CLIP text encoder is loaded from `openai/clip-vit-large-patch14` through
Hugging Face Transformers on the first run, so the environment must be able to
access the Hugging Face Hub (or have that model cached locally).

## 2. Download checkpoints and data

Create the local asset directories and download the released checkpoints:

```bash
mkdir -p ckpts datasets
wget -O ckpts/Sk-VAE.pt 'https://huggingface.co/zhaors00/Stroke3D/resolve/main/ckpt/Sk-VAE/Sk-VAE.pt?download=true'
wget -O ckpts/Sk-DiT.pt 'https://huggingface.co/zhaors00/Stroke3D/resolve/main/ckpt/Sk-DiT/Sk-DiT.pt?download=true'
```

The files are also available at [Sk-VAE](https://huggingface.co/zhaors00/Stroke3D/blob/main/ckpt/Sk-VAE/Sk-VAE.pt)
and [Sk-DiT](https://huggingface.co/zhaors00/Stroke3D/blob/main/ckpt/Sk-DiT/Sk-DiT.pt).

Download and extract the shared skeleton dataset into `SkDiff/datasets`:

[Stroke3D-Skeleton-Data.tar.gz](https://huggingface.co/zhaors00/Stroke3D/blob/main/data/Skeleton-Data/Stroke3D-Skeleton-Data.tar.gz)

```bash
wget -O datasets/Stroke3D-Skeleton-Data.tar.gz 'https://huggingface.co/zhaors00/Stroke3D/resolve/main/data/Skeleton-Data/Stroke3D-Skeleton-Data.tar.gz?download=true'
tar -xzf datasets/Stroke3D-Skeleton-Data.tar.gz -C datasets
```

The extracted dataset directory should contain `train_meta.json`,
`val_meta.json`, `test_meta.json`, `test_common_meta.json`, and the `train/`,
`train_diverse/`, `test/`, and `test_diverse/` skeleton directories.
`test_common_meta.json` aggregates the samples that both Stroke3D
and the other baselines can infer; it is the test set used for the
paper's reported metrics.

## 3. Infer 3D skeletons from 2D strokes

Run all commands from `SkDiff` so that the relative paths in the YAML files
resolve correctly. Each command projects the test-set 3D skeleton onto one
plane and reconstructs a 3D skeleton with Sk-DiT:

```bash
python scripts/sample_diff.py --config configs/sample/sample_diff_xy.yml
python scripts/sample_diff.py --config configs/sample/sample_diff_xz.yml
python scripts/sample_diff.py --config configs/sample/sample_diff_yz.yml
```

The generated skeletons are written to `outputs/skeleton_xy`,
`outputs/skeleton_xz`, and `outputs/skeleton_yz`; corresponding visualizations
are written to the `outputs/vis_*` directories.

## 4. Evaluate Chamfer distance

After all three views have finished, run:

```bash
python scripts/eval_chamfer.py
```

The evaluator matches predictions with the ground-truth skeletons from
`test_common_meta.json`, picks the best available result among the three
projection views for each skeleton, and reports joint-to-joint, joint-to-bone,
and bone-to-bone Chamfer distances. These are the same paper-test-set metrics.
Results are saved to `outputs/chamfer/stroke3d.json`.

## Code layout

- `models/vae.py`: Sk-VAE implementation.
- `models/diffusion.py` and `models/latent_model.py`: Sk-DiT components.
- `scripts/sample_diff.py`: three-view inference entry point.
- `scripts/eval_chamfer.py`: multi-view Chamfer evaluation.
- `scripts/train_vae.py` and `scripts/train_diff.py`: training entry points.

The ControlNet files and older configuration files are retained for historical
reproducibility and are not required for the released three-view inference
workflow.
