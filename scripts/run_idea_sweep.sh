#!/bin/bash
# Distribute the idea sweep across GPUs, each GPU runs its queue sequentially,
# all queues run in parallel in the background. Logs per config land in
# docs/ideaN/logs/kK/ ; a per-GPU runlog goes to logs/idea_sweep/.
#
# Usage:
#   bash scripts/run_idea_sweep.sh A      # experiment A: 3 methods x {192,128,64}
#   bash scripts/run_idea_sweep.sh B      # experiment B (initial screening): x {288,346}
#   bash scripts/run_idea_sweep.sh all
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHICH="${1:-A}"
GPUS=(${GPUS:-0 1 2 3 4})   # use at most 5 GPUs by default
NG=${#GPUS[@]}

# config list: "idea K"
CONFIGS_A=( "idea1 192" "idea1 128" "idea1 64" "idea2 192" "idea2 128" "idea2 64" "idea3 192" "idea3 128" "idea3 64" )
CONFIGS_B=( "idea1 288" "idea1 346" "idea2 288" "idea2 346" "idea3 288" "idea3 346" )

case "$WHICH" in
  A) CONFIGS=( "${CONFIGS_A[@]}" ) ;;
  B) CONFIGS=( "${CONFIGS_B[@]}" ) ;;
  all) CONFIGS=( "${CONFIGS_A[@]}" "${CONFIGS_B[@]}" ) ;;
  *) echo "unknown: $WHICH (A|B|all)"; exit 1 ;;
esac

SWEEP_LOG="${REPO_ROOT}/logs/idea_sweep"
mkdir -p "$SWEEP_LOG"

# build per-GPU queues (round-robin)
declare -a QUEUE
for i in "${!CONFIGS[@]}"; do
  g=$(( i % NG ))
  QUEUE[$g]="${QUEUE[$g]}|${CONFIGS[$i]}"
done

for g in "${!GPUS[@]}"; do
  gpu=${GPUS[$g]}
  q="${QUEUE[$g]#|}"
  [ -z "$q" ] && continue
  (
    IFS='|' read -ra items <<< "$q"
    for it in "${items[@]}"; do
      set -- $it
      idea=$1; K=$2
      echo "$(date '+%F %T') [gpu$gpu] START $idea K=$K"
      CUDA_VISIBLE_DEVICES=$gpu bash "${REPO_ROOT}/scripts/idea_eval.sh" "$idea" "$K" \
        >> "${SWEEP_LOG}/gpu${gpu}.runlog" 2>&1
      echo "$(date '+%F %T') [gpu$gpu] DONE  $idea K=$K"
    done
  ) >> "${SWEEP_LOG}/gpu${gpu}.runlog" 2>&1 &
  echo "launched gpu$gpu queue: ${q}"
done
wait
echo "sweep $WHICH finished."
