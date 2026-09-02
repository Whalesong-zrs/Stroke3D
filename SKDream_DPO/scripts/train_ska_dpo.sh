#!/usr/bin/env bash
set -euo pipefail

: "${DPO_DATA_DIR:?Set DPO_DATA_DIR to a validated preference dataset.}"
: "${SFT_CONTROLNET:?Set SFT_CONTROLNET to the SFT SKDream checkpoint or Hub ID.}"

BASE_MODEL="${BASE_MODEL:-lzq49/mvdream-sd21-diffusers}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ska_dpo_margin_0.10}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

accelerate launch --num_processes "${NUM_PROCESSES}" train_skdream_dpo.py \
  --mixed_precision fp16 \
  --pretrained_model_name_or_path "${BASE_MODEL}" \
  --controlnet_model_name_or_path "${SFT_CONTROLNET}" \
  --data_root_dir "${DPO_DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --cond_channels 4 \
  --num_views 4 \
  --resolution 256 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 100 \
  --learning_rate 5e-6 \
  --train_batch_size 4 \
  --max_train_steps 1000 \
  --dataloader_num_workers 16 \
  --checkpointing_steps 500 \
  --checkpoints_total_limit 2 \
  --gradient_accumulation_steps 2 \
  --tracker_project_name skdream \
  --report_to tensorboard
