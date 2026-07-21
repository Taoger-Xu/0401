#!/bin/bash
# Usage: bash scripts/v1_5/eval/sqa.sh <config>
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

python -m eval.model_vqa_science \
    --model-path "$MODEL_PATH" \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${CKPT}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    $VZ_ARGS

python -m llava.eval.eval_science_qa \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${CKPT}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}_output.json \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}_result.json
