#!/usr/bin/env bash
# Corrected configuration for text-heavy tasks: drop spatial coverage (rho=0)
# but KEEP feature-MMR (lam=0.5), which the ablation showed is beneficial.
# Fills in TextVQA/OCRBench at K=192,128 (K=64 already measured) and adds a
# non-text side-check (GQA, MMStar) to see whether rho=0 is a global win or a
# text-specific trade-off.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export IDEA_RHO=0.0 IDEA_LAMBDA=0.5
r () { echo "[rho0fix $(date -Is)] K=$1 tasks=$2"; timeout -k 30 -s TERM 2700 bash "${ROOT}/scripts/qwen2_5_vl_idea4_eval.sh" "$1" "$2"; }
r 192 textvqa_val
r 192 ocrbench
r 128 textvqa_val
r 128 ocrbench
r 64  gqa
r 64  mmstar
echo "[rho0fix $(date -Is)] done"
