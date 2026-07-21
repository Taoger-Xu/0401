#!/usr/bin/env bash
# Ablation for the OCR-degradation hypothesis: in text-heavy images saliency
# concentrates on text, so idea4's diversity terms may be spending budget on
# non-text regions. Two knobs:
#   rho  - share of budget given to spatial coverage (rho=0 disables coverage)
#   lam  - feature-MMR redundancy penalty (lam=0 -> pure saliency ranking)
# Reference point already measured: K=64 rho=0.5 lam=0.5 -> OCRBench 45.00.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
K="${K:-64}"
TASKS="${TASKS:-ocrbench}"

run_cfg () { # $1=rho $2=lam
  echo "[ablation $(date -Is)] === K=$K rho=$1 lam=$2 tasks=$TASKS ==="
  IDEA_RHO="$1" IDEA_LAMBDA="$2" \
    timeout -k 30 -s TERM 2700 bash "${ROOT}/scripts/qwen2_5_vl_idea4_eval.sh" "$K" "$TASKS"
}

run_cfg 0.0  0.5   # no spatial coverage, keep feature-MMR
run_cfg 0.0  0.0   # pure saliency (no coverage, no MMR)
run_cfg 0.25 0.5   # reduced coverage
echo "[ablation $(date -Is)] done"
