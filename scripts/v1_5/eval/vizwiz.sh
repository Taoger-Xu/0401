#!/bin/bash
# Usage: bash scripts/v1_5/eval/vizwiz.sh <config>
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

python -m eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    $VZ_ARGS

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${CKPT}.json
