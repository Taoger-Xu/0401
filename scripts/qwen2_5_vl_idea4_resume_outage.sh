#!/usr/bin/env bash
# Resume the post-fix idea4 sweep after the 2026-07-18 power outage.
# Surviving valid results: vanilla 8/8, K=192 6/8. Remaining work below.
# Skips any (K, task) whose result already exists, so it is safe to re-run
# after another interruption.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER="${ROOT}/scripts/qwen2_5_vl_idea4_pertask.sh"
FULL="gqa mme mmstar pope scienceqa_img textvqa_val vizwiz_vqa_val ocrbench"

have_task () {  # $1=K $2=task -> 0 if result already present
  python3 - "$1" "$2" <<'PY'
import json,glob,sys
K,task=sys.argv[1],sys.argv[2]
d=f"docs/idea4/qwen2_5_vl_idea4_k{K}_rho0.5_lambda0.5"
for f in glob.glob(f"{d}/**/*results*.json",recursive=True):
    try:
        if task in json.load(open(f)).get("results",{}): sys.exit(0)
    except: pass
sys.exit(1)
PY
}

echo "[resume-outage $(date -Is)] start"
for K in 192 128 64; do
  todo=""
  for t in $FULL; do
    if have_task "$K" "$t"; then echo "  skip K=$K $t (already done)"; else todo="$todo $t"; fi
  done
  if [ -z "${todo// /}" ]; then echo "[resume-outage] K=$K already complete"; continue; fi
  echo "[resume-outage $(date -Is)] === K=$K todo:$todo ==="
  K=$K TASK_LIST="$(echo $todo)" bash "$DRIVER"
done
echo "[resume-outage $(date -Is)] done"
