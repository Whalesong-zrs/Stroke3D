#!/bin/bash

# --- 在这里配置你要使用的GPU ---
# 例如，使用第 0 张卡
GPUS="0" 

# --- 配置你的项目路径和参数 (请使用绝对路径) ---
# DATA_DIR="/home/zrs/mix_data/dpo_dataset3/test_set"
# OUTPUT_DIR="/home/zrs/mix_data/dpo_dataset3/test_set"
DATA_DIR="/home/zrs/mix_data/dpo_dataset3/"
OUTPUT_DIR="/home/zrs/mix_data/dpo_dataset3/origin_model_output"
MVC_CKPT="/data05/xyy/3D/skdream_model/ckpt/skdream/"

echo "启动单进程任务，使用 GPU: ${GPUS}"

# 设置CUDA_VISIBLE_DEVICES确保脚本只看到你指定的GPU
# 直接运行Python脚本，不再需要并行参数
CUDA_VISIBLE_DEVICES=${GPUS} python create_dpo_pairs.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --mvc_ckpt "${MVC_CKPT}" \
  --gpu ${GPUS}

echo "任务执行完毕。"