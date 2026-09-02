# SKA-DPO data format

The training loader expects one caption, one camera bundle, four winner images, four
loser images, and four skeleton-condition images per training ID. The number of views
is configurable, but the Stroke3D experiments use four.

```text
dataset_root/
├── dpo_texturig_captions.json
├── train_eval.json
├── pair_scores.json                 # optional; written by prepare_dpo_pairs.py
├── win_mv/<id>/gen_0.png ... gen_3.png
├── lose_mv/<id>/gen_0.png ... gen_3.png
├── skeleton_d/<id>/cond_0.png ... cond_3.png
└── meta/<id>/cam_dict.pkl
```

`dpo_texturig_captions.json` maps each ID to a text prompt. `train_eval.json` contains
`train` and `eval` ID lists. The current training script consumes the `train` split.

`cam_dict.pkl` stores the synchronized camera values used for the four views. It must
contain `elevation` and `azimuth` sequences with at least `num_views` elements. It is
a legacy Python pickle and must only be loaded from a trusted Stroke3D release; never
replace it with an untrusted pickle.

Validate a prepared dataset before training:

```bash
python tools/validate_dpo_dataset.py /path/to/dataset_root
```

The preference builder records both SKA scores, their gap, the requested margin, and
the number of sampling attempts in `pair_scores.json`. The historical Stroke3D release
data predates that sidecar, so the file is optional for training.
