#!/usr/bin/env bash
# Complete the rho=0 (no spatial coverage) table: the 4 benchmarks not yet run
# at rho=0, across K=192/128/64. Text tasks + GQA/MMStar were already measured.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export IDEA_RHO=0.0 IDEA_LAMBDA=0.5
for K in 192 128 64; do
  echo "[rho0-complete $(date -Is)] === K=$K ==="
  K=$K TASK_LIST="mme pope scienceqa_img vizwiz_vqa_val" bash "${ROOT}/scripts/qwen2_5_vl_idea4_pertask.sh"
done
# K=192/128 still need gqa+mmstar at rho=0 (only K=64 was measured)
for K in 192 128; do
  echo "[rho0-complete $(date -Is)] === K=$K gqa/mmstar ==="
  K=$K TASK_LIST="gqa mmstar" bash "${ROOT}/scripts/qwen2_5_vl_idea4_pertask.sh"
done
echo "[rho0-complete $(date -Is)] done"
