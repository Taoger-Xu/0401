#!/usr/bin/env bash
# Offline full evaluation of idea4 on local Qwen2.5-VL-7B.
# Usage: CUDA_VISIBLE_DEVICES=0,1,... bash scripts/qwen2_5_vl_idea4_eval.sh [K] [tasks]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PYBIN:-/home/jk/miniconda3/envs/clse_qwen/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/jk/models/qwen2.5-vl-7b}"
K="${1:-192}"
TASKS="${2:-gqa,mmbench_en_dev,mme,pope,scienceqa_img,textvqa_val,vizwiz_vqa_val,ocrbench}"
RUN_TAG="qwen2_5_vl_idea4_k${K}_rho${IDEA_RHO:-0.5}_lambda${IDEA_LAMBDA:-0.5}"
RESULT_DIR="${ROOT}/docs/idea4/${RUN_TAG}"
LOG_DIR="${ROOT}/logs"

export HF_HOME="${HF_HOME:-/home/jk/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/jk/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IDEA_QWEN_K="$K"
export IDEA_RHO="${IDEA_RHO:-0.5}"
export IDEA_LAMBDA="${IDEA_LAMBDA:-0.5}"
export IDEA_COVER_FACTOR="${IDEA_COVER_FACTOR:-3.0}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RESULT_DIR" "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/${RUN_TAG}.log") 2>&1
echo "[$(date -Is)] model=$MODEL_PATH tasks=$TASKS K=$K rho=$IDEA_RHO lambda=$IDEA_LAMBDA"

IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"
PORT="${PORT:-$((20000 + RANDOM % 30000))}"
"$PYBIN" -m accelerate.commands.launch --num_processes "$NUM_GPUS" --main_process_port "$PORT" \
  "${ROOT}/eval/qwen2_5_vl_idea4_entry.py" \
  --model qwen2_5_vl \
  --model_args "pretrained=${MODEL_PATH},attn_implementation=eager" \
  --tasks "$TASKS" --batch_size 1 \
  --output_path "$RESULT_DIR" --log_samples
echo "[$(date -Is)] evaluation finished: $RESULT_DIR"
