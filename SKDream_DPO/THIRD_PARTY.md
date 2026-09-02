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
- The downstream multi-view reconstruction module is derived from InstantMesh:
  <https://github.com/TencentARC/InstantMesh>
- Differentiable rendering and texture refinement use NVIDIA nvdiffrast and code
  carrying file-level NVIDIA copyright notices:
  <https://github.com/NVlabs/nvdiffrast>
- UV parameterization and CUDA neural textures use xatlas and tiny-cuda-nn:
  <https://github.com/jpcy/xatlas>
  <https://github.com/NVlabs/tiny-cuda-nn>

No third-party checkpoints are stored in this Git repository. The restored
`objsk_eval2/` directory contains historical evaluation inputs; its provenance and
redistribution status should be verified before a packaged data release. A
repository-wide license for the Stroke3D-authored code should also be selected by
the copyright holders. Preserve all file-level copyright headers in `instantmesh/`,
`geometry/`, and `render/`, and verify their redistribution terms before release.
