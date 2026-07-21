#!/usr/bin/env bash
# Offline VANILLA (no idea4 / no pruning) evaluation of local Qwen2.5-VL-7B.
# Baseline reference for scripts/qwen2_5_vl_idea4_eval.sh under an identical
# protocol. Usage: CUDA_VISIBLE_DEVICES=0,1,... bash scripts/qwen2_5_vl_vanilla_eval.sh [tasks]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PYBIN:-/home/jk/miniconda3/envs/clse_qwen/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/jk/models/qwen2.5-vl-7b}"
TASKS="${1:-gqa,mme,pope,scienceqa_img,textvqa_val,vizwiz_vqa_val,ocrbench}"
RUN_TAG="qwen2_5_vl_vanilla"
RESULT_DIR="${ROOT}/docs/idea4/${RUN_TAG}"
LOG_DIR="${ROOT}/logs"

export HF_HOME="${HF_HOME:-/home/jk/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/jk/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RESULT_DIR" "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/${RUN_TAG}.log") 2>&1
echo "[$(date -Is)] VANILLA model=$MODEL_PATH tasks=$TASKS"

IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"
PORT="${PORT:-$((20000 + RANDOM % 30000))}"
"$PYBIN" -m accelerate.commands.launch --num_processes "$NUM_GPUS" --main_process_port "$PORT" \
  "${ROOT}/eval/qwen2_5_vl_vanilla_entry.py" \
  --model qwen2_5_vl \
  --model_args "pretrained=${MODEL_PATH},attn_implementation=eager" \
  --tasks "$TASKS" --batch_size 1 \
  --output_path "$RESULT_DIR" --log_samples
echo "[$(date -Is)] evaluation finished: $RESULT_DIR"
