#!/bin/bash
# Usage: bash scripts/v1_5/eval/vqav2.sh <config>
# NOTE: test-dev2015 has no local ground truth; scoring requires submitting
# playground/data/eval/vqav2/answers_upload/<SPLIT>/<CKPT>.json to the
# official VQA challenge eval server (https://eval.ai/web/challenges/challenge-page/830).
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
SPLIT="llava_vqav2_mscoco_test-dev2015"

python -m eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
    --image-folder ./playground/data/eval/vqav2/test2015 \
    --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    $VZ_ARGS

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT --dir ./playground/data/eval/vqav2
