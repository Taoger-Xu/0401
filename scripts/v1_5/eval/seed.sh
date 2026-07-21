#!/bin/bash
# Usage: bash scripts/v1_5/eval/seed.sh <config>
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

# LLaVA-1.5 is image-only; llava-seed-bench.jsonl also lists video questions
# whose frames were never extracted here, so we restrict to the image subset
# (matches how LLaVA-1.5 SEED-Bench numbers are normally reported).
python -m eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file /home/jk/work/paper/VisionZip/seed_gt/llava-seed-bench-image-only.jsonl \
    --image-folder ./playground/data/eval/seed_bench \
    --answers-file ./playground/data/eval/seed_bench/answers/${CKPT}/merge.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    $VZ_ARGS

python scripts/convert_seed_for_submission.py \
    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file ./playground/data/eval/seed_bench/answers/${CKPT}/merge.jsonl \
    --result-upload-file ./playground/data/eval/seed_bench/answers_upload/${CKPT}.jsonl
