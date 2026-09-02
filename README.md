# Stroke3D: Lifting 2D Strokes into Rigged 3D Model via Latent Diffusion Models

Ruisi Zhao, Haoren Zheng, Zongxin Yang, Hehe Fan, Yi Yang  
**ICLR 2026**

[[OpenReview](https://openreview.net/forum?id=VgOWxor3LV)]
[[Project page](https://whalesong-zrs.github.io/Stroke3D_project_page/)]

## Code

- [`SkDiff/`](SkDiff/) is the legacy archival snapshot of the skeleton latent
  VAE, diffusion, and ControlNet components. It is not maintained in this
  release and may still contain environment-specific historical settings.
- [`SKDream_DPO/`](SKDream_DPO/) contains the cleaned SKDream/SKA-DPO pipeline
  for preference-pair construction, training, inference, and evaluation.

The maintained open-source path is `SKDream_DPO`; follow its README for the
environment, data layout, and commands. Datasets and checkpoints are not stored
in this Git repository. They will be collected in the single Hugging Face model
repository [`zhaors00/Stroke3D`](https://huggingface.co/zhaors00/Stroke3D), with
separate subfolders for each component.

## Release status

- [x] Code release
- [x] Skeleton checkpoints: [Sk-VAE](https://drive.google.com/file/d/17v0J-oBpKQFxzEmfVINp0gbK-oAKylcO/view?usp=sharing) and [Sk-Diff](https://drive.google.com/file/d/1CVAq1dHnv34oXtPfIYSJYCgP00McMeiB/view?usp=drive_link)
- [ ] SKA-DPO checkpoint release
- [ ] Data release

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
