# VisionZip 本地复现结果（LLaVA-1.5-7B）vs 论文 Table 1

复现方法见 [docs/exp.md](exp.md)。模型：`/home/jk/models/llava-v1.5-7b`（本地权重）。`K` 为 VisionZip 保留的视觉 token 总数（原始 576 个），剪枝比例与论文 Table 1 一致：K=192 → 66.7%，K=128 → 77.8%，K=64 → 88.9%。

- K=192 → dominant=162, contextual=30
- K=128 → dominant=108, contextual=20
- K=64  → dominant=54,  contextual=10

（128 / 192 两档的 dominant/contextual 拆分是按 64-token 档位官方比例换算的近似值，不是论文附录给出的原始数值——从下面的对比看，这个近似拆分对结果影响很小。）

论文数值来自你贴的 Table 1，对应"训练无关"的 **VisionZip**（未微调）行，不是 VisionZip‡（微调 projector）那一行，因为我们的复现全程没有做任何微调。

## GQA 数据问题（已修复）

VisiPruner 共享的 `playground/data/eval/gqa/data/testdev_balanced_questions.json`（GQA 打分用的 ground truth）只有 500 道题，而标准 test-dev-balanced 应该有 12578 道题，导致最初跑出来的 GQA 分数明显偏高、且不可比（vanilla 69.60% vs 论文 61.9%）。

修复方法：从本机已有的 `/home/jk/datasets/lmms-lab___gqa`（HuggingFace 格式的官方 GQA 数据集）里取出完整的 12578 题 `question/answer/isBalanced/types/semantic/groups` 字段，重建了一份完整 ground truth，存在 `/home/jk/work/paper/VisionZip/gqa_gt/testdev_balanced_questions.json`（没有改动 VisiPruner 的任何文件）。模型推理结果不用重新跑，直接用已有的 `testdev_balanced_predictions_*.json` 对着这份完整 ground truth 重新调用官方 `eval/eval.py` 打分即可。下表已经是修复后的数字。

## 训练无关（Training-free）VisionZip：复现 vs 论文

| Benchmark | 指标 | 本机 K=192 | 论文 K=192 | Δ | 本机 K=128 | 论文 K=128 | Δ | 本机 K=64 | 论文 K=64 | Δ |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GQA | Acc | 59.14 | 59.3 | −0.16 (−0.3%) | 57.73 | 57.6 | +0.13 (+0.2%) | 55.53 | 55.1 | +0.43 (+0.8%) |
| MME | Total | 1774.96 | 1782.6 | −7.64 (−0.4%) | 1762.98 | 1761.7 | +1.28 (+0.1%) | 1710.78 | 1690 | +20.78 (+1.2%) |
| POPE | Acc | 85.78 | 85.3 | +0.48 (+0.6%) | 84.27 | 83.2 | +1.07 (+1.3%) | 80.39 | 77.0 | +3.39 (+4.4%) |
| ScienceQA | Acc | 69.89 | 68.9 | +0.99 (+1.4%) | 69.82 | 68.9 | +0.92 (+1.3%) | 69.91 | 69.0 | +0.91 (+1.3%) |
| TextVQA | Acc | 57.27 | 57.3 | −0.03 (−0.1%) | 56.85 | 56.8 | +0.05 (+0.1%) | 55.52 | 55.5 | +0.02 (+0.0%) |

**Vanilla（576 tokens）基线对比：**

| Benchmark | 本机 | 论文 | Δ |
| --- | :---: | :---: | :---: |
| GQA | 61.96 | 61.9 | +0.06 (+0.1%) |
| MME | 1863.71 | 1862 | +1.71 (+0.1%) |
| POPE | 86.42 | 85.9 | +0.52 (+0.6%) |
| ScienceQA | 70.12 | 69.5 | +0.62 (+0.9%) |
| TextVQA | 58.18 | 58.2 | −0.02 (−0.0%) |

## 结论

**GQA/MME/POPE/SQA/TextVQA 五个 benchmark，在 vanilla 和三档 VisionZip 配置下全部跟论文 Table 1 高度吻合**——15 个数字里 14 个偏差在 ±1.4% 以内，TextVQA 三档全部 |Δ|≤0.1%，vanilla GQA/MME/TextVQA 更是几乎完全对上（|Δ|≤0.1%）。这说明：

1. 本地环境（`llava_visiPruner` 里的 LLaVA + VisionZip patch）复现是可信的；
2. K=128/192 档按比例近似出来的 dominant/contextual 拆分，即使不是论文附录的精确值，实际效果也非常接近论文数字。

唯一偏差明显大一点的是 **POPE K=64（+4.4%）**，其余全部 benchmark、全部 token 档位偏差都在 1.5% 以内。如果要追这最后一点精度，值得核实一下官方论文附录里 64-token 档的 dominant/contextual 是不是就是 54/10。

## 未运行 / 无法本地打分的 benchmark

论文 Table 1 里的 MMBench(MMB)、VQAv2、MMMU、SEED、MM-Vet、LLaVA-Bench(LLaVA-B) 还没有纳入对比：

- **MMBench-EN/CN**：4 个配置推理已完成，打分需要把答案提交到 MMBench 官方 server，本机没有 dev split 的本地 ground truth。
- **VQAv2**：test-dev 有 107394 道题且没有本地 ground truth（打分同样要交官方 server），4 个配置全量跑完预计要一天以上的 GPU 时间，性价比很低，需要你确认要不要真的跑。
- **MMMU**：本仓库当前接入的评测数据（复用 VisiPruner 的 `playground/data/eval/`）里没有 MMMU，需要额外接入。
- **SEED / MM-Vet**：数据已就绪，且都能本地打分（SEED 有本地 ground truth；MM-Vet 需要 GPT-4 judge）。
- **LLaVA-Bench**：数据已就绪，打分需要 GPT-4 judge。

如果需要，我可以：

1. 继续跑 SEED（本地可直接出分）；
2. 跑 MM-Vet / LLaVA-Bench 的推理（生成答案不需要 API key，但打分需要 GPT-4，需要你提供 OpenAI API key）；
3. 决定 VQAv2 是否要跑（跑了也只能生成上传文件，出不了本地分数）；
4. 处理 MMBench 的官方提交（需要你的 MMBench 账号/流程说明，我目前没有自动化上传的手段）；
5. 接入 MMMU 数据。
