#!/usr/bin/env bash
# Resilient per-task driver for idea4 on Qwen2.5-VL.
# Runs each task in its own lmms-eval invocation so the cross-GPU result gather
# stays small (avoids the NCCL internal error seen when gathering all tasks at
# once) and each task's results persist independently. Retries a task once on
# failure. P2P disabled to dodge the inactive-NVLink peer-access crash.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Only 5 GPUs: GPU 7 is faulty (Xid / CUDA-unknown-error hangs) and running all
# cards drew enough power to trip an outage on 2026-07-18, so cap the load.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export IDEA_RHO="${IDEA_RHO:-0.5}"
export IDEA_LAMBDA="${IDEA_LAMBDA:-0.5}"
K="${K:-192}"
# Per-task wall-clock cap so a wedged-GPU hang can't stall the sweep for hours.
TASK_TIMEOUT="${TASK_TIMEOUT:-2700}"

cleanup_procs() {
  pkill -9 -f "qwen2_5_vl_idea4_entry.py" 2>/dev/null
  pkill -9 -f "accelerate.commands.launch" 2>/dev/null
  sleep 5
}

# True only when a results JSON for this (K, task) actually exists.
have_result() { # $1=K $2=task
  ROOT="$ROOT" python3 - "$1" "$2" "${IDEA_RHO}" "${IDEA_LAMBDA}" <<'PY'
import json,glob,os,sys
K,task,rho,lam=sys.argv[1:5]
d=f"{os.environ['ROOT']}/docs/idea4/qwen2_5_vl_idea4_k{K}_rho{rho}_lambda{lam}"
for f in glob.glob(f"{d}/**/*results*.json",recursive=True):
    try:
        if task in json.load(open(f)).get("results",{}): sys.exit(0)
    except Exception: pass
sys.exit(1)
PY
}

# NOTE: mmbench_en_dev is skipped here — offline it falls back to the OpenAI GPT
# API for answer extraction (401 Unauthorized, ~20s/item of retries) and cannot
# be scored without an API key or MMBench-server submission. gqa already done.
# Override with TASK_LIST env if needed.
read -ra TASKS <<< "${TASK_LIST:-gqa mme mmstar pope scienceqa_img textvqa_val vizwiz_vqa_val ocrbench}"
SUMMARY="${ROOT}/logs/qwen2_5_vl_idea4_pertask_summary.log"
: > "$SUMMARY"
echo "[$(date -Is)] per-task driver start; K=$K rho=$IDEA_RHO lambda=$IDEA_LAMBDA gpus=$CUDA_VISIBLE_DEVICES" | tee -a "$SUMMARY"

for t in "${TASKS[@]}"; do
  ok=0
  for attempt in 1 2 3; do
    echo "[$(date -Is)] >>> task=$t attempt=$attempt (timeout=${TASK_TIMEOUT}s gpus=$CUDA_VISIBLE_DEVICES)" | tee -a "$SUMMARY"
    rc=0
    timeout -k 30 -s TERM "$TASK_TIMEOUT" bash "${ROOT}/scripts/qwen2_5_vl_idea4_eval.sh" "$K" "$t" || rc=$?
    # Exit code alone is NOT trustworthy: lmms-eval catches a failed cross-GPU
    # gather (NCCL "internal error", which POPE hits repeatedly) and still exits
    # 0 while writing no results. Require the result JSON to actually contain
    # this task before calling it a success.
    if have_result "$K" "$t"; then
      echo "[$(date -Is)] OK task=$t (attempt $attempt, rc=$rc, result verified)" | tee -a "$SUMMARY"
      ok=1
      break
    elif [ "$rc" -eq 0 ]; then
      echo "[$(date -Is)] FAIL task=$t (attempt $attempt) — exited 0 but NO result written (silent gather failure)" | tee -a "$SUMMARY"
      cleanup_procs
    elif [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      echo "[$(date -Is)] TIMEOUT/HANG task=$t (attempt $attempt, rc=$rc) — cleaning up" | tee -a "$SUMMARY"
      cleanup_procs
    else
      echo "[$(date -Is)] FAIL task=$t (attempt $attempt, rc=$rc)" | tee -a "$SUMMARY"
      cleanup_procs
    fi
  done
  [ "$ok" -eq 0 ] && echo "[$(date -Is)] GAVE UP task=$t" | tee -a "$SUMMARY"
done

echo "[$(date -Is)] per-task driver done" | tee -a "$SUMMARY"
