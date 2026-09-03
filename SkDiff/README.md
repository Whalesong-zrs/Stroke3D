# SkDiff archival code

This folder contains the historical Stroke3D code for the skeleton variational
autoencoder (Sk-VAE), skeleton latent diffusion transformer (Sk-DiT), and the
associated ControlNet experiments. It is retained for reproducibility and is not
the actively maintained SKA-DPO module.

## Released assets

- [Sk-VAE checkpoint](https://huggingface.co/zhaors00/Stroke3D/tree/main/ckpt/Sk-VAE)
- [Sk-DiT checkpoint](https://huggingface.co/zhaors00/Stroke3D/tree/main/ckpt/Sk-DiT)
- [Shared Sk-VAE/Sk-DiT skeleton data](https://huggingface.co/zhaors00/Stroke3D/tree/main/data/Skeleton-Data)

The shared data archive contains skeleton-coordinate text files and JSON metadata;
it does not contain rendered PNG images, TextuRig assets, or DPO preference pairs.

## Code layout

- `models/vae.py`: Sk-VAE model implementation.
- `models/diffusion.py` and `models/latent_model.py`: skeleton diffusion models.
- `models/controlnet.py`: skeleton ControlNet implementation.
- `scripts/train_vae.py`: Sk-VAE training entry point.
- `scripts/train_diff.py`: Sk-DiT training entry point.
- `scripts/train_controlnet.py`: historical ControlNet training entry point.
- `scripts/sample_*.py`: sampling and qualitative evaluation utilities.

Install the historical dependencies with `pip install -r requirements.txt`.
Several scripts predate the cleaned release and may still require explicit local
paths or environment-specific adjustments. Review their command-line arguments and
configuration before running them.
