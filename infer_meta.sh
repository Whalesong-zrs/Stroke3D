#!/bin/bash

# 脚本设置：遇到任何错误则立即退出
set -e

# --- 1. 静态配置 ---
# 定义检查点所在的根目录
CKPT_ROOT_DIR="/data05/xyy/3D/skdream_model/ckpt/skdream/"

# >>> 新增：定义输出文件夹的根目录和前缀
# 您可以修改下面的值来改变输出文件夹的名称和位置
OUTPUT_ROOT_DIR="/home/zrs/mix_data/dpo_dataset3"
MV_FOLDER_PREFIX="res_mv"

# 定义固定的数据目录和通用参数
DATA_DIR="/home/zrs/mix_data/dpo_dataset3"
REPEAT_NUM=4
NUM_VIEW=4

# 设置使用的GPU
export CUDA_VISIBLE_DEVICES=1

# --- 2. 主循环：遍历所有检查点 ---
# 使用 seq 命令生成一个从 1000 到 10000，步长为 1000 的数字序列

# 打印当前正在处理的检查点信息，方便追踪进度
echo "=============================================================="
echo "=============== Processing Checkpoint==============="
echo "=============================================================="

# --- 3. 动态路径配置 ---
# 根据当前步数构建完整的检查点路径
MVC_CKPT="${CKPT_ROOT_DIR}"

# >>> 修改：根据顶部定义的前缀和步数，为当前检查点的输出创建唯一的目录名
MV_DIR="${OUTPUT_ROOT_DIR}/${MV_FOLDER_PREFIX}"

# 运行前检查检查点目录是否存在，如果不存在则打印警告并跳过
if [ ! -d "$MVC_CKPT" ]; then
    echo "Warning: Checkpoint directory ${MVC_CKPT} not found. Skipping."
    continue # 跳过当前循环，继续下一个
fi

# --- 4. 依次执行各个阶段的Python脚本 ---

# --- 阶段 1: 骨骼条件下的多视图生成 ---
echo "--- Running Multi-view generation ---"
python infer_meta.py \
    --data_dir $DATA_DIR \
    --save_dir $MV_DIR \
    --num_view $NUM_VIEW \
    --repeat_num $REPEAT_NUM \
    --neg_prompt 'default' \
    --cond_scale 1.0 \
    --gpu 0 \
    --mvc_ckpt "$MVC_CKPT"
