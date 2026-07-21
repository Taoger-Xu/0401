#!/bin/bash
# VisionZip evaluation via the lmms-eval framework (mirrors CLSE's llava_lmms_eval.sh).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/lmms_eval.sh vanilla
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/lmms_eval.sh vz64
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/lmms_eval.sh vz128
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/lmms_eval.sh vz192
set -e

CONFIG="${1:-vanilla}"

case "$CONFIG" in
  vanilla) unset VZ_DOMINANT VZ_CONTEXTUAL ;;
  vz64)    export VZ_DOMINANT=54  VZ_CONTEXTUAL=10 ;;
  vz128)   export VZ_DOMINANT=108 VZ_CONTEXTUAL=20 ;;
  vz192)   export VZ_DOMINANT=162 VZ_CONTEXTUAL=30 ;;
  *) echo "unknown config: $CONFIG"; exit 1 ;;
esac

NUM_GPUS="${NUM_GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/jk/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
MODEL_PATH="/home/jk/models/llava-v1.5-7b"
TASKS="${TASKS:-gqa,mmbench_en_dev,mmbench_cn_dev,mme,pope,scienceqa_img,textvqa_val,vizwiz_vqa_val,ocrbench}"
BASE_OUTPUT_PATH="${OUTPUT_DIR:-logs/lmms-eval/${CONFIG}}"

mkdir -p "$BASE_OUTPUT_PATH"
PORT=$((10000 + RANDOM % 55000))

accelerate launch --num_processes=$NUM_GPUS \
    --main_process_port $PORT \
    eval/lmms_eval_entry.py \
    --model llava \
    --model_args "pretrained=${MODEL_PATH}" \
    --tasks $TASKS \
    --batch_size 1 \
    --output_path "$BASE_OUTPUT_PATH" \
    --log_samples
echo "[$CONFIG] lmms-eval run finished."
