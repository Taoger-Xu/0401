#!/bin/bash
# =============================================================================
# Exp-M3 · 均匀空间位置删除 · GQA / MMBench / ScienceQA 保留比例扫描
# =============================================================================
# NOTE: 此脚本原样迁移自 CLSE 仓库 docs/mov/exp-m3/run_uniform_sweep.sh，仅作
# 数据采集过程的溯源留档——它依赖 CLSE 仓库里的 LLaVA1.5/scripts/v1_5/eval/*.sh
# 和 stage0_prune.py，在本仓库（Fourier-Compressor）中无法直接运行。真实产出
# 已固化在 mov/data/results_uniform_redundancy.csv，由 mov/plot_uniform_redundancy.py
# 读取绘图。
# =============================================================================
# 用 stage0_prune.py::uniform_compress（STAGE0_METHOD=uniform）在 Top-K%
# Tokens = 20/40/60/80/100 五档下分别跑三个 benchmark，为
# results_uniform_redundancy.csv 提供真实数据（当前该 CSV 全部是占位值）。
#
# k 与 Top-K% 的对应（576 * pct/100 四舍五入）：
#   100% -> 576   80% -> 461   60% -> 346   40% -> 230   20% -> 115
#
# 用法（从 LLaVA1.5/ 目录运行）：
#   bash ../docs/mov/exp-m3/run_uniform_sweep.sh gqa
#   bash ../docs/mov/exp-m3/run_uniform_sweep.sh mmbench
#   bash ../docs/mov/exp-m3/run_uniform_sweep.sh sqa
#
# 每格跑完后，把评测脚本打印的 Accuracy 填入
# docs/mov/exp-m3/figures/results_uniform_redundancy.csv 对应行的 score/perf_ratio
# 列（perf_ratio = score / 该任务 topk_pct=100 的 score * 100），并把 placeholder 置 0。
# 全部替换后运行：
#   python ../docs/mov/exp-m3/plot_uniform_redundancy.py \
#       --data ../docs/mov/exp-m3/figures/results_uniform_redundancy.csv \
#       --outdir ../docs/mov/exp-m3/figures/ --final
# =============================================================================
set -u

BENCH="${1:-gqa}"
K_LIST=(576 461 346 230 115)     # 对应 Top-K% = 100/80/60/40/20
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$BENCH" in
  gqa)     EVAL_SH="scripts/v1_5/eval/gqa.sh" ;;
  mmbench) EVAL_SH="scripts/v1_5/eval/mmbench.sh" ;;
  sqa)     EVAL_SH="scripts/v1_5/eval/sqa.sh" ;;   # 若脚本名不同请自行改
  *) echo "未知 benchmark: $BENCH（支持 gqa|mmbench|sqa）"; exit 1 ;;
esac

for k in "${K_LIST[@]}"; do
  echo "───────────────────────────────────────────────"
  echo ">>> benchmark=$BENCH  method=uniform  k=$k  topk_pct=$(python -c "print(round($k/576*100))")"
  echo "───────────────────────────────────────────────"
  STAGE0_K="$k" STAGE0_METHOD=uniform PRUNE=0 bash "$EVAL_SH"
done

echo
echo "扫描结束。请把各格分数填入 docs/mov/exp-m3/figures/results_uniform_redundancy.csv"
echo "然后：python ../docs/mov/exp-m3/plot_uniform_redundancy.py --data ../docs/mov/exp-m3/figures/results_uniform_redundancy.csv --outdir ../docs/mov/exp-m3/figures/ --final"
