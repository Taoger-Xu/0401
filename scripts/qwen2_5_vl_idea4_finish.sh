#!/usr/bin/env bash
# Finish the post-fix idea4 sweep after the 2026-07-18 power outage.
# Remaining work is stated explicitly (verified against the surviving result
# files) rather than auto-detected, so there is no chance of redoing completed
# tasks. Surviving valid: vanilla 8/8, K=192 6/8.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER="${ROOT}/scripts/qwen2_5_vl_idea4_pertask.sh"
FULL="gqa mme mmstar pope scienceqa_img textvqa_val vizwiz_vqa_val ocrbench"

echo "[finish $(date -Is)] start"

echo "[finish $(date -Is)] === K=192 (remaining 2) ==="
K=192 TASK_LIST="vizwiz_vqa_val ocrbench" bash "$DRIVER"

echo "[finish $(date -Is)] === K=128 (full 8) ==="
K=128 TASK_LIST="$FULL" bash "$DRIVER"

echo "[finish $(date -Is)] === K=64 (full 8) ==="
K=64 TASK_LIST="$FULL" bash "$DRIVER"

echo "[finish $(date -Is)] done"
