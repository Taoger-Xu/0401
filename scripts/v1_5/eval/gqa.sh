#!/bin/bash
# Usage: bash scripts/v1_5/eval/gqa.sh <config>
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
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"

python -m eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
    --image-folder ./playground/data/eval/gqa/data/images \
    --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    $VZ_ARGS

python scripts/convert_gqa_for_eval.py \
    --src ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl \
    --dst $GQADIR/testdev_balanced_predictions_${CKPT}.json

cd $GQADIR
# NOTE: the shared testdev_balanced_questions.json here only has 500 of the
# official 12578 questions (see docs/result.md). Score against the full set
# rebuilt from /home/jk/datasets/lmms-lab___gqa instead.
python eval/eval.py --tier testdev_balanced \
    --questions /home/jk/work/paper/VisionZip/gqa_gt/testdev_balanced_questions.json \
    --predictions testdev_balanced_predictions_${CKPT}.json
