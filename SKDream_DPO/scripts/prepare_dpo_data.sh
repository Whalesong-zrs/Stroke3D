#!/usr/bin/env bash
set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to the directory containing eval.json and cano_sk/.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for the generated preference dataset.}"
: "${SFT_CONTROLNET:?Set SFT_CONTROLNET to the SFT SKDream checkpoint or Hub ID.}"
: "${DINO_CHECKPOINT:?Set DINO_CHECKPOINT to the DINOv2 checkpoint.}"
: "${SKA_CHECKPOINT:?Set SKA_CHECKPOINT to the SKA scorer checkpoint.}"

BASE_MODEL="${BASE_MODEL:-lzq49/mvdream-sd21-diffusers}"
DEVICE="${DEVICE:-cuda}"
MARGIN="${MARGIN:-0.10}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

extra_args=()
if [[ -n "${DINOV2_REPO:-}" ]]; then
  extra_args+=(--dinov2-repo "${DINOV2_REPO}")
fi

python prepare_dpo_pairs.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --controlnet "${SFT_CONTROLNET}" \
  --base-model "${BASE_MODEL}" \
  --dino-checkpoint "${DINO_CHECKPOINT}" \
  --ska-checkpoint "${SKA_CHECKPOINT}" \
  --device "${DEVICE}" \
  --margin "${MARGIN}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  "${extra_args[@]}"
