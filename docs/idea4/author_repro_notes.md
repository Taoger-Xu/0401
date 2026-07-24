# SCOPE 作者复现备忘录（不进入投稿正文）

本文件保存本机路径、历史故障与内部实验索引。投稿稿件只保留可公开复现所需的协议说明。

## 实现入口

- LLaVA 方法实现：`visionzip/prune_ideas.py` 中的 `select_anchor_cover`（SCOPE 的历史内部函数名），以及
  `_dct_matrix`、`_fps_voronoi_cells`、`lowfreq_reconstruct`。
- LLaVA 注入：`visionzip/idea_inject.py`，读取 `IDEA_METHOD`、`IDEA_K`、`IDEA_RHO`、
  `IDEA_LAMBDA`、`IDEA_SIGMA`。
- Qwen2.5-VL 注入：`Qwen2_5_VL/qwen2_5vl_visionzip.py` 与
  `eval/qwen2_5_vl_idea4_entry.py`。
- Qwen 复现脚本：`scripts/qwen2_5_vl_idea4_eval.sh`。

## 数据索引

- LLaVA：`docs/idea4/logs/{cf3_sigInf_*,tvqa_*,text_salience_*}`；vanilla 见
  `logs/lmms-eval/vanilla` 与 `docs/idea5/logs/noocr_vanilla`。
- MMStar 与 COCO 控制台日志：`logs/{mmstar_*,coco_*}.log`。
- 聚合脚本：`scripts/aggregate_ideas.py`。
- Qwen：`docs/idea4/qwen2_5_vl_idea4_k*_rho*_lambda*/models__qwen2.5-vl-7b/` 与
  `docs/idea4/qwen2_5_vl_vanilla/`。
- 低频证据：`scripts/diag_lowfreq_evidence.py`，输出到
  `docs/idea4/figs/lowfreq_evidence/`。
- 内部方法对照：`docs/idea_summary.md` 与 `docs/idea{1,2,3}/`。

## 已知协议与正确性风险

1. 2026-07-07 前后的 TextVQA prompt 是否含 `Reference OCR token` 不一致。旧协议与无 OCR 新协议不可
   直接比较；以 samples jsonl 的 `input` 字段为判据。
2. 2026-07-18 前的 Qwen SCOPE 运行没有真正触发剪枝，已归档到
   `docs/idea4/_invalid_pre_fix_0718/`，禁止引用。原因包括无效 Tensor API、未更新 `cache_position`，
   以及 lmms-eval wrapper 已提前绑定原始模型类。
3. Qwen 运行必须用断言核对逐样本 `kept==K`，并确认不同 K 的结果确有变化，不能只依赖退出码。
4. 本机 GPU 7 存在 Xid/CUDA 故障，多卡评测需排除该卡并设置任务超时。
5. 本地 Qwen 模型路径为 `/home/jk/models/qwen2.5-vl-7b`；环境使用
   `attn_implementation=eager`、`batch_size=1`。

## 投稿前实验缺口

- `cover_factor` 的严格单变量消融。
- cell 内 `medoid / low-frequency / local saliency` 逐项消融。
- 更完整的 $\lambda$ 与 $\rho$ 敏感性。
- 空间距离、cell 占有率、最近邻特征相似度、注意力捕获率。
- Qwen $\rho=0$ 的完整非文字 benchmark 复验。
- 端到端延迟、吞吐、峰值显存和选择器耗时。
