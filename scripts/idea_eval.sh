#!/bin/bash
# Evaluate the idea1/idea2/idea3 spatial-redundancy pruners via lmms-eval,
# writing logs into the matching docs/ideaN/logs/ directory.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/idea_eval.sh <idea1|idea2|idea3> <K> [tasks]
# e.g.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/idea_eval.sh idea3 128
#   CUDA_VISIBLE_DEVICES=1 bash scripts/idea_eval.sh idea2 192 gqa,pope
set -e

IDEA="${1:?need idea1|idea2|idea3}"
K="${2:?need token budget K}"
TASKS_ARG="${3:-}"

case "$IDEA" in
  idea1) export IDEA_METHOD=spectral  ;;
  idea2) export IDEA_METHOD=local_var ;;
  idea3) export IDEA_METHOD=cls_mmr   ;;
  idea4) export IDEA_METHOD=anchor_cover ;;
  # idea5 diagnostics (not pruning). Second arg is repurposed:
  #   recon <lowpass>    576-token low-pass reconstruction probe
  #   dctdown <out_grid> DCT-resize compression to out_grid^2 tokens
  recon)   export IDEA_METHOD=lowfreq_recon ;;
  dctdown) export IDEA_METHOD=dct_down ;;
  idea5)
    export IDEA_METHOD=detail_mmr
    export IDEA_GAMMA="${IDEA_GAMMA:-1.0}"          # idea5 default differs from idea6
    export IDEA_DETAIL_P="${IDEA_DETAIL_P:-2.0}"
    export IDEA_DETAIL_SRC="${IDEA_DETAIL_SRC:-local_var}"
    ;;
  idea6) export IDEA_METHOD=sem_var ;;
  *) echo "unknown idea: $IDEA (use idea1|idea2|idea3|idea4|idea5|idea6|recon|dctdown)"; exit 1 ;;
esac
export IDEA_K="$K"
if [ "$IDEA_METHOD" = "lowfreq_recon" ]; then
  export IDEA_LOWPASS="$K"
  export IDEA_K=577                 # CLS + 576 reconstructed tokens (no reduction)
elif [ "$IDEA_METHOD" = "dct_down" ]; then
  export IDEA_OUTGRID="$K"
  export IDEA_K=$((K * K + 1))      # CLS + out_grid^2 tokens
fi
export IDEA_LAMBDA="${IDEA_LAMBDA:-0.5}"
export IDEA_RHO="${IDEA_RHO:-0.5}"
export IDEA_SIGMA="${IDEA_SIGMA:-2.0}"
export IDEA_WVAR="${IDEA_WVAR:-0.3}"
export IDEA_GAMMA="${IDEA_GAMMA:-0.5}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PYBIN:-/home/jk/miniconda3/envs/llava_visiPruner/bin/python}"
NUM_GPUS="${NUM_GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/jk/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
MODEL_PATH="/home/jk/models/llava-v1.5-7b"
TASKS="${TASKS:-${TASKS_ARG:-gqa,mmbench_en_dev,mme,pope,scienceqa_img,textvqa_val}}"
# idea4 rho sweeps go to a distinct dir so they don't overwrite the default rho=0.5
SUFFIX=""
if [ "$IDEA_METHOD" = "anchor_cover" ] && [ "$IDEA_RHO" != "0.5" ]; then
  SUFFIX="_rho${IDEA_RHO}"
fi
# idea6 (w_v, gamma) sweeps go to distinct dirs so the default isn't overwritten
if [ "$IDEA_METHOD" = "sem_var" ] && { [ "$IDEA_WVAR" != "0.3" ] || [ "$IDEA_GAMMA" != "0.5" ]; }; then
  SUFFIX="_wv${IDEA_WVAR}_g${IDEA_GAMMA}"
fi
# idea5 (gamma, p, detail source) sweeps likewise
if [ "$IDEA_METHOD" = "detail_mmr" ] && { [ "$IDEA_GAMMA" != "1.0" ] || [ "$IDEA_DETAIL_P" != "2.0" ] || [ "$IDEA_DETAIL_SRC" != "local_var" ]; }; then
  SUFFIX="_g${IDEA_GAMMA}_p${IDEA_DETAIL_P}_${IDEA_DETAIL_SRC}"
fi
# controlled-merge probe (idea5 §4): pruned tokens averaged into kept ones
export IDEA_MERGE="${IDEA_MERGE:-0}"
if [ "$IDEA_MERGE" = "1" ]; then
  SUFFIX="${SUFFIX}_merge"
fi
if [ "$IDEA_METHOD" = "lowfreq_recon" ]; then
  BASE_OUTPUT_PATH="${OUTPUT_DIR:-${REPO_ROOT}/docs/idea5/logs/recon_lp${K}}"
elif [ "$IDEA_METHOD" = "dct_down" ]; then
  BASE_OUTPUT_PATH="${OUTPUT_DIR:-${REPO_ROOT}/docs/idea5/logs/dctdown_g${K}}"
else
  BASE_OUTPUT_PATH="${OUTPUT_DIR:-${REPO_ROOT}/docs/${IDEA}/logs/k${K}${SUFFIX}}"
fi

mkdir -p "$BASE_OUTPUT_PATH"
PORT=$((10000 + RANDOM % 55000))

echo "[$IDEA K=$K method=$IDEA_METHOD lambda=$IDEA_LAMBDA rho=$IDEA_RHO sigma=$IDEA_SIGMA wvar=$IDEA_WVAR gamma=$IDEA_GAMMA] tasks=$TASKS -> $BASE_OUTPUT_PATH"

"$PYBIN" -m accelerate.commands.launch --num_processes=$NUM_GPUS \
    --main_process_port $PORT \
    "${REPO_ROOT}/eval/lmms_eval_entry.py" \
    --model llava \
    --model_args "pretrained=${MODEL_PATH}" \
    --tasks $TASKS \
    --batch_size 1 \
    --output_path "$BASE_OUTPUT_PATH" \
    --log_samples
echo "[$IDEA K=$K] lmms-eval run finished."
