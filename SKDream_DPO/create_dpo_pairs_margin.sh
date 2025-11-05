#!/bin/bash

# --- 在这里配置你要使用的GPU ---
# 例如，使用第 0 张卡

GPUS="0" 

# --- 配置你的项目路径和参数 (请使用绝对路径) ---
# DATA_DIR="/home/zrs/mix_data/dpo_dataset3/"
# OUTPUT_DIR="/home/zrs/mix_data/dpo_dataset3/origin_model_output"
# MVC_CKPT="/data05/xyy/3D/skdream_model/ckpt/skdream/"

# export CUDA_VISIBLE_DEVICES=0
# MARGIN=0.1

# export CUDA_VISIBLE_DEVICES=1
# MARGIN=0.05

# export CUDA_VISIBLE_DEVICES=2
# MARGIN=0.15

export CUDA_VISIBLE_DEVICES=3
MARGIN=0.2

DATA_DIR="/home/zrs/mix_data/dpo_dataset3/"
OUTPUT_DIR="/home/zrs/mix_data/dpo_dataset3/sft_model_output_${MARGIN}"
MVC_CKPT="/home/zrs/skdream_model/tmp_v3/checkpoint-9000/skdream"



echo "启动单进程任务，使用 GPU: ${GPUS}"
echo "SKA Score Margin设置为: ${MARGIN}"

# 设置CUDA_VISIBLE_DEVICES确保脚本只看到你指定的GPU
# 直接运行Python脚本，并传入margin参数
CUDA_VISIBLE_DEVICES=${GPUS} python create_dpo_pairs_margin.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --mvc_ckpt "${MVC_CKPT}" \
  --gpu ${GPUS} \
  --margin ${MARGIN}

echo "任务执行完毕。"