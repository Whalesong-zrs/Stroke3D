# Stroke3D SKA-DPO preference pairs

## Summary

This dataset contains the preference pairs used to fine-tune the SKDream component of
Stroke3D with SKA-DPO. It contains 2,000 skeleton-caption examples. Each example has
four synchronized skeleton conditions, four preferred multi-view images, four rejected
multi-view images, and camera metadata.

Candidate pairs were sampled from the TextuRig-fine-tuned SKDream model and ranked by
the skeleton-image alignment (SKA) scorer. The final paper setting requested a score
margin of 0.10 and retried sampling up to three times before retaining the last pair.

## Contents

- `dpo_texturig_captions.json`: prompt keyed by sample ID.
- `train_eval.json`: 2,000 training IDs and five external evaluation IDs.
- `win_mv/`: preferred RGBA views.
- `lose_mv/`: rejected RGBA views.
- `skeleton_d/`: RGBA skeleton conditions; the fourth channel encodes depth.
- `meta/`: pickled camera dictionaries.

The historical dataset does not include per-pair SKA score sidecars. New data generated
with `prepare_dpo_pairs.py` records these values in `pair_scores.json`.

## Intended use and limitations

The data is intended for research on skeleton-conditioned multi-view generation and
preference optimization. Rankings inherit limitations from SKDream, foreground
segmentation, DINOv2, and the learned SKA scorer. A score preference is not a general
measure of image quality.

## Licensing note

No blanket dataset license is asserted. Users must review the applicable
TextuRig/Objaverse-XL source-asset terms before redistribution or commercial use. The
camera metadata uses legacy Python pickle files and must only be loaded from a trusted
Stroke3D release.
