#!/bin/bash
# Usage: bash scripts/v1_5/eval/llavabench.sh <config>
# NOTE: scoring needs a GPT-4 judge (llava/eval/eval_gpt_review_bench.py), which
# requires an OpenAI API key. This script only produces the answer file.
set -e

CONFIG="${1:-vanilla}"
MODEL_PATH="/home/jk/models/llava-v1.5-7b"

case "$CONFIG" in
  vanilla) VZ_ARGS="" ;;
  vz64)    VZ_ARGS="--vz-dominant 54 --vz-contextual 10" ;;
  vz128)   VZ_ARGS="--vz-dominant 108 --vz-contextual 20" ;;
  vz192)   VZ_ARGS="--vz-dominant 162 --vz-contextual 30" ;;
  *) echo "unknown config: $CONFIG"; exit 1 ;;
esac

CKPT="llava-v1.5-7b-${CONFIG}"

python -m eval.model_vqa \
    --model-path "$MODEL_PATH" \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    $VZ_ARGS
