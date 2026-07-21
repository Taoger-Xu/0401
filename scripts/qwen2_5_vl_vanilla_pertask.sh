#!/usr/bin/env bash
# Resilient per-task driver for the VANILLA Qwen2.5-VL baseline.
# One lmms-eval invocation per task (small cross-GPU gather, independent
# result files, one retry each). P2P disabled to dodge the inactive-NVLink
# peer-access crash. mmbench_en_dev is skipped (needs OpenAI API / server).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0

read -ra TASKS <<< "${TASK_LIST:-gqa mme mmstar pope scienceqa_img textvqa_val vizwiz_vqa_val ocrbench}"
SUMMARY="${ROOT}/logs/qwen2_5_vl_vanilla_pertask_summary.log"
: > "$SUMMARY"
echo "[$(date -Is)] VANILLA per-task driver start; gpus=$CUDA_VISIBLE_DEVICES" | tee -a "$SUMMARY"

for t in "${TASKS[@]}"; do
  ok=0
  for attempt in 1 2; do
    echo "[$(date -Is)] >>> task=$t attempt=$attempt" | tee -a "$SUMMARY"
    if bash "${ROOT}/scripts/qwen2_5_vl_vanilla_eval.sh" "$t"; then
      echo "[$(date -Is)] OK task=$t (attempt $attempt)" | tee -a "$SUMMARY"
      ok=1
      break
    else
      echo "[$(date -Is)] FAIL task=$t (attempt $attempt)" | tee -a "$SUMMARY"
      sleep 10
    fi
  done
  [ "$ok" -eq 0 ] && echo "[$(date -Is)] GAVE UP task=$t" | tee -a "$SUMMARY"
done

echo "[$(date -Is)] VANILLA per-task driver done" | tee -a "$SUMMARY"
