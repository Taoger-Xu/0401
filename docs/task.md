# Task："进入 LLM 前"空间冗余剪枝方案的通用性对比实验（idea1–4）

## 0. 目标

在 LLaVA-1.5-7B + lmms-eval 框架下，用**同一注入点、同一 token 预算**比较四种在
`mm_projector` 之前对 576 个视觉 token 做 hard pruning 的初筛方案，回答两个问题：

1. **通用性**：在不同剪枝力度（K=192/128/64）下，哪种方案在多个 benchmark 上整体掉分最小、最稳？
2. **初筛可行性**：把这些方案当作"初筛"（只保留 50%/60% token）时，性能是否几乎不掉，从而可以作为后续二级剪枝的前置步骤？

## 1. 四个方案（对应目录）

| 目录 | 方案 | `IDEA_METHOD` | 核心信号 | 说明 |
| --- | --- | --- | --- | --- |
| `docs/idea1/` | Spectral Relay Pruning | `spectral` | DCT 低频稳定性 + FPS 空间 cell medoid | 见 [idea1/idea1.md](idea1/idea1.md) |
| `docs/idea2/` | Local Variation | `local_var` | 8 邻域局部变化度（1−邻居余弦） | 见 [idea2/idea2.md](idea2/idea2.md) |
| `docs/idea3/` | CLS-Attention 冗余感知（MMR） | `cls_mmr` | CLS 注意力显著性 + 冗余惩罚 | 见 [idea3/idea3.md](idea3/idea3.md) |
| `docs/idea4/` | Anchor-Cover 双池初筛 | `anchor_cover` | 覆盖池（FPS cell + 注意力代表）+ 显著池（空间门控 MMR） | 见 [idea4/idea4.md](idea4/idea4.md) |

四者统一：注入点 = CLIP 倒数第二层输出（VisionZip 同款）；输出 = CLS + (K−1) 个**原始** patch token，共 K 个；不合并、不重建。
idea4 是 idea1/idea3 的融合泛化（ρ→1 退化为 idea1，ρ→0 且 σ→∞ 退化为 idea3，ρ→0 且 λ=0 退化为 VisionZip dominant 纯剪枝版），
其主假设是同预算下双多样性（空间覆盖硬保证 + 语义去冗）使其通用性最好。

## 2. 公共设置

- 模型：`/home/jk/models/llava-v1.5-7b`，环境 `llava_visiPruner`
- 框架：lmms-eval，入口 [eval/lmms_eval_entry.py](../eval/lmms_eval_entry.py)（读 `IDEA_METHOD`/`IDEA_K`/`IDEA_LAMBDA`）
- 运行脚本：[scripts/idea_eval.sh](../scripts/idea_eval.sh) `<idea1|idea2|idea3> <K> [tasks]`
- 核心 benchmark（6 项，和 VisionZip 论文对比口径一致，兼顾 VQA / 感知 / 文字 / 科学 / 多选）：
  `gqa, mmbench_en_dev, mme, pope, scienceqa_img, textvqa_val`
- 数据走本地 HF 缓存（`HF_DATASETS_OFFLINE=1`）
- 每个配置的日志落到对应 idea 目录：`docs/idea<N>/logs/k<K>/`

## 3. 基线（复用已有结果，不重跑）

`logs/lmms-eval/{vanilla,vz64,vz128,vz192}/` 已有 VisionZip 与 vanilla 的完整 9-task 结果
（见 [docs/lmms.md](lmms.md)）。对比时直接抽取上面 6 个 benchmark 的分数作为：

- **上界**：vanilla（576 token）
- **同预算对手**：VisionZip vz192 / vz128 / vz64

## 4. 实验 A：剪枝力度扫描（主实验，判"通用性"）

4 方案 × K∈{192, 128, 64} = **12 个配置**，每个跑 6 benchmark。

```bash
# 示例（GPU 与配置对应关系见 scripts/run_idea_sweep.sh）
CUDA_VISIBLE_DEVICES=g bash scripts/idea_eval.sh idea3 192
CUDA_VISIBLE_DEVICES=g bash scripts/idea_eval.sh idea3 128
CUDA_VISIBLE_DEVICES=g bash scripts/idea_eval.sh idea3 64
# idea1 / idea2 / idea4 同理（idea4 需先完成 docs/idea4/idea4.md §7 的实现清单）
```

产出：每个 benchmark 分数 → 汇总成"方案 × K"表，和 vanilla / VisionZip 同表对比。
判据：相对 vanilla 的平均保留率、以及各 benchmark 的最差掉分。

## 5. 实验 B：初筛可行性（判"保留 50%/60% 掉分是否可忽略"）

4 方案 × K∈{288 (50%), 346 (60%)} = **8 个配置**，同样 6 benchmark。

目标结论形式："用方案 X 初筛到 60%（346 token），6 benchmark 平均相对 vanilla 保留 ≥ 99%，
掉分在容差内 → 可作为无损初筛前置。"

## 6. 实验 C（可选）：超参敏感性

- idea3：固定 K=128，`IDEA_LAMBDA∈{0.25, 0.5, 1.0}`，看去冗余强度对通用性的影响，选出默认 λ；
- idea4：固定 K=128，`IDEA_RHO∈{0.25, 0.5, 0.75}`（λ=0.5、σ=2.0 固定），看覆盖池/显著池
  预算配比对不同任务类型的影响（预期：POPE/MME 偏好小 ρ，GQA/MMBench 偏好大 ρ）。

## 6.5 实验 D（idea4 先行）：双多样性诊断

不依赖 benchmark 的选择器行为对比：同一批图（GQA 前 500 张）、同 K=128，对
VisionZip dominant top-K / idea1 / idea3 / idea4 统计 cell 占有率、平均 pairwise 空间距离、
平均最近邻特征相似度、注意力捕获率四项度量（定义见 [idea4/idea4.md](idea4/idea4.md) §6），
验证"idea4 同时保双多样性"的机制性主张。结果落 `docs/idea4/logs/diag/`。

## 7. 交付物

1. 每个方案每档 K 的 lmms-eval 原始日志 → `docs/idea<N>/logs/k<K>/`
2. 每个 `docs/idea<N>/idea<N>.md` 第 6/结果节回填该方案自己的分数表
3. 本仓库根 [docs/task.md](task.md) 末尾 / 或 `docs/idea_summary.md` 汇总三方案 + VisionZip + vanilla 的总对比表与结论

## 8. 进度

- [x] 完善 idea3 方案（CLS-MMR），写入 [idea3/idea3.md](idea3/idea3.md)
- [x] 实现 idea1–3 选择器 [visionzip/prune_ideas.py](../visionzip/prune_ideas.py) + 注入 [visionzip/idea_inject.py](../visionzip/idea_inject.py)
- [x] 入口/脚本打通，端到端 smoke test（idea1–3 × K 都输出正确 token 数）
- [x] 完善 idea4 方案（Anchor-Cover 双池初筛），写入 [idea4/idea4.md](idea4/idea4.md)
- [ ] 实现 idea4 选择器 `select_anchor_cover` + 注入/入口/脚本支持（清单见 [idea4/idea4.md](idea4/idea4.md) §7）
- [ ] idea4 smoke test（K 档输出正确 + 三个极限退化一致性抽查）
- [ ] 实验 D：双多样性诊断（idea4 先行验证，可与实验 A 并行）
- [ ] 实验 A：12 配置扫描（idea1–3 共 9 个 + idea4 共 3 个）
- [ ] 实验 B：8 配置初筛
- [ ] 实验 C：idea3 λ 敏感性 / idea4 ρ 敏感性（可选）
- [ ] 汇总对比表 + 结论
