#!/bin/bash
# Usage: bash scripts/v1_5/eval/mme.sh <config>
# <config> in: vanilla | vz64 | vz128 | vz192
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
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    $VZ_ARGS

cd ./playground/data/eval/MME
# NOTE: convert_answer_to_mme.py assumes questions_answers_YN/ subfolders that
# are missing for artwork/celebrity/landmark/posters/scene in this shared
# dataset copy (see docs/exp.md section 11). Use the ground truth bundled in
# eval_tool/Your_Results instead, which covers all 14 categories.
python /home/jk/work/paper/VisionZip/eval/convert_mme_for_eval.py --experiment ${CKPT}
cd eval_tool
python calculation.py --results_dir answers/${CKPT}
