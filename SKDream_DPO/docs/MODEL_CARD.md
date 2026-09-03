# Stroke3D SKA-DPO SKDream weights

## Model description

These weights contain the SKDream multi-view controlnet fine-tuned with SKA-DPO for
Stroke3D. The model takes text, four projected skeleton conditions, and synchronized
camera matrices, and produces four multi-view images.

The selected checkpoint is the 1,000-step model trained on 2,000 preference pairs built
with a requested SKA margin of 0.10. Training uses a learning rate of `5e-6`, per-device
batch size 4, gradient accumulation 2, and diffusion-DPO inverse temperature 5000.

## Files

- `config.json`: Diffusers model configuration.
- `diffusion_pytorch_model.safetensors`: SKDream controlnet parameters.

The checkpoint is not a standalone text-to-image pipeline. Load it together with the
MVDream Diffusers base model through `load_skdream_pipeline` or pass it to `infer_mv.py`.
For the shared Stroke3D Hub repository, use repository ID `zhaors00/Stroke3D` and
subfolder `ckpt/SKDream-SKA-DPO`.

## Limitations

The model is designed for four-view skeleton-conditioned generation at 256 x 256. It
inherits limitations and usage restrictions from its base model, SFT checkpoint,
training data, DINOv2 features, SKA scorer, and foreground-removal models.

## Licensing note

Only inference files are distributed; optimizer state is excluded. No blanket license
is asserted for upstream base or SFT checkpoint components. Users must review their
licenses and provenance before redistribution or commercial use.
