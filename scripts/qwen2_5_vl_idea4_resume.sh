#!/usr/bin/env bash
# Resume sweep after the GPU-7 fault: finish K=128 (vizwiz + ocrbench were lost
# to the hang), then run K=64 across all 8 offline-scorable benchmarks. Uses the
# hardened per-task driver (GPU 0-6, per-task timeout, cleanup+retry).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER="${ROOT}/scripts/qwen2_5_vl_idea4_pertask.sh"
FULL="gqa mme mmstar pope scienceqa_img textvqa_val vizwiz_vqa_val ocrbench"

echo "[resume $(date -Is)] start"

# 1) K=128 remaining tasks
K=128 TASK_LIST="vizwiz_vqa_val ocrbench" bash "$DRIVER"

# 2) K=64 full
K=64 TASK_LIST="$FULL" bash "$DRIVER"

echo "[resume $(date -Is)] done"
