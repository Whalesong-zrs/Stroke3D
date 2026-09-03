# Stroke3D: Lifting 2D Strokes into Rigged 3D Model via Latent Diffusion Models

Ruisi Zhao, Haoren Zheng, Zongxin Yang, Hehe Fan, Yi Yang  
**ICLR 2026**

[[OpenReview](https://openreview.net/forum?id=VgOWxor3LV)]
[[Project page](https://whalesong-zrs.github.io/Stroke3D_project_page/)]

## Code

- [`SkDiff/`](SkDiff/) is the legacy archival snapshot of the skeleton latent
  VAE, diffusion, and ControlNet components. It is not maintained in this
  release and may still contain environment-specific historical settings. See
  [`SkDiff/README.md`](SkDiff/README.md) for its scope, entry points, and assets.
- [`SKDream_DPO/`](SKDream_DPO/) contains the cleaned SKDream/SKA-DPO pipeline
  for preference-pair construction, training, inference, and evaluation. See
  [`SKDream_DPO/README.md`](SKDream_DPO/README.md) for setup and commands.
- [`TextuRig/`](TextuRig/) contains the cleaned TextuRig curation utilities and
  documents the 6,633-sample skeleton/GLB release format. See
  [`TextuRig/README.md`](TextuRig/README.md) for the data format and processing
  pipeline.

This front page is an overview. Detailed environments, data layouts, and commands
are maintained in the README inside each code subfolder. Datasets and checkpoints
are not stored in Git; they are released in
[`zhaors00/Stroke3D`](https://huggingface.co/zhaors00/Stroke3D) under `data/` and
`ckpt/`.

## Release status

- [x] Code release
- [x] Checkpoints: [Sk-VAE](https://huggingface.co/zhaors00/Stroke3D/tree/main/ckpt/Sk-VAE), [Sk-DiT](https://huggingface.co/zhaors00/Stroke3D/tree/main/ckpt/Sk-DiT), and [SKDream/SKA-DPO](https://huggingface.co/zhaors00/Stroke3D/tree/main/ckpt/SKDream-SKA-DPO)
- [x] Data: [shared skeleton training data](https://huggingface.co/zhaors00/Stroke3D/tree/main/data/Skeleton-Data), [SKA-DPO preference pairs](https://huggingface.co/zhaors00/Stroke3D/tree/main/data/DPO-Data), and [TextuRig](https://huggingface.co/zhaors00/Stroke3D/tree/main/data/TextuRig)

## Citation

```bibtex
@inproceedings{zhao2026stroke3d,
  title     = {Stroke3D: Lifting 2D Strokes into Rigged 3D Model via Latent Diffusion Models},
  author    = {Zhao, Ruisi and Zheng, Haoren and Yang, Zongxin and Fan, Hehe and Yang, Yi},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=VgOWxor3LV}
}
```

## Contact

For questions, email Ruisi Zhao at zhaors00@zju.edu.cn or open an issue.
