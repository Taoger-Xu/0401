#!/usr/bin/env bash
# Stage-2 sweep: fill the K=192 MMStar gap, then run idea4 at K=128 and K=64
# across all 8 offline-scorable benchmarks, to build the full retention table
# against the vanilla baseline. Runs each (K, task) via the resilient per-task
# driver. Sequential so the 8 GPUs are never over-subscribed.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER="${ROOT}/scripts/qwen2_5_vl_idea4_pertask.sh"
FULL="gqa mme mmstar pope scienceqa_img textvqa_val vizwiz_vqa_val ocrbench"

echo "[ksweep $(date -Is)] start"

# 1) K=192 was run before MMStar was added — fill just that one task.
K=192 TASK_LIST="mmstar" bash "$DRIVER"

# 2) K=128 full
K=128 TASK_LIST="$FULL" bash "$DRIVER"

# 3) K=64 full
K=64 TASK_LIST="$FULL" bash "$DRIVER"

echo "[ksweep $(date -Is)] done"
