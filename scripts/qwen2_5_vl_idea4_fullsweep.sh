#!/usr/bin/env bash
# Full idea4 sweep AFTER the transformers-4.57.6 fixes (pruning now actually
# fires: _anchor_cover_rect in-place ops, cache_position rebuild, and the entry
# wrapper-module rebind). Runs K=192/128/64 across all 8 offline-scorable
# benchmarks via the hardened per-task driver (GPU 0-6, timeout+cleanup+retry).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER="${ROOT}/scripts/qwen2_5_vl_idea4_pertask.sh"
FULL="gqa mme mmstar pope scienceqa_img textvqa_val vizwiz_vqa_val ocrbench"

echo "[fullsweep $(date -Is)] start"
for K in 192 128 64; do
  echo "[fullsweep $(date -Is)] === K=$K ==="
  K=$K TASK_LIST="$FULL" bash "$DRIVER"
done
echo "[fullsweep $(date -Is)] done"
