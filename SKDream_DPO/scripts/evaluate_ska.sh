#!/usr/bin/env bash
set -euo pipefail

: "${EVAL_JSON:?Set EVAL_JSON to the evaluation metadata file.}"
: "${IMAGE_DIR:?Set IMAGE_DIR to the generated multi-view result directory.}"
: "${DINO_CHECKPOINT:?Set DINO_CHECKPOINT to the DINOv2 checkpoint.}"
: "${SKA_CHECKPOINT:?Set SKA_CHECKPOINT to the SKA scorer checkpoint.}"

extra_args=()
if [[ -n "${DINOV2_REPO:-}" ]]; then
  extra_args+=(--dinov2-repo "${DINOV2_REPO}")
fi

python evaluate_ska.py \
  --eval-json "${EVAL_JSON}" \
  --image-dir "${IMAGE_DIR}" \
  --dino-checkpoint "${DINO_CHECKPOINT}" \
  --ska-checkpoint "${SKA_CHECKPOINT}" \
  --num-views "${NUM_VIEWS:-4}" \
  --num-repeats "${NUM_REPEATS:-1}" \
  --device "${DEVICE:-cuda}" \
  "${extra_args[@]}"
