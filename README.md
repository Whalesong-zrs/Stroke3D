# Stroke3D: Lifting 2D Strokes into Rigged 3D Model via Latent Diffusion Models

Ruisi Zhao, Haoren Zheng, Zongxin Yang, Hehe Fan, Yi Yang  
**ICLR 2026**

[[OpenReview](https://openreview.net/forum?id=VgOWxor3LV)]
[[Project page](https://whalesong-zrs.github.io/Stroke3D_project_page/)]

## Code

- [`SkDiff/`](SkDiff/) contains the skeleton latent VAE, diffusion, and
  ControlNet components.
- [`SKDream_DPO/`](SKDream_DPO/) contains the cleaned SKDream/SKA-DPO pipeline
  for preference-pair construction, training, inference, and evaluation.

Please follow the README inside each module for its environment, data layout,
and commands. Datasets and the SKA-DPO checkpoint are not stored in this Git
repository; their links will be added after the release metadata and licenses
have been reviewed.

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
