# Third-party components

This repository contains adapted research code and interfaces with external models.
Users are responsible for complying with the licenses and terms of every checkpoint
and dataset they download.

- The training entry point is adapted from Hugging Face Diffusers ControlNet training
  examples and retains their Apache-2.0 source header:
  <https://github.com/huggingface/diffusers>
- SKDream provides the skeleton-conditioned multi-view generation foundation:
  <https://skdream3d.github.io/>
- MVDream provides the base multi-view diffusion checkpoint:
  <https://github.com/bytedance/MVDream>
- DINOv2 provides the visual backbone used by the SKA scorer:
  <https://github.com/facebookresearch/dinov2>
- The DPO objective follows Diffusion-DPO:
  <https://github.com/SalesforceAIResearch/DiffusionDPO>

No third-party checkpoints or datasets are stored in this Git repository. A
repository-wide license for the Stroke3D-authored code should be selected by the
copyright holders before calling the release complete.
