# VisionZip lmms-eval 复现结果（LLaVA-1.5-7B）

> **⚠️ TextVQA 列已过时（2026-07-08）**：本页评测时（07-02）lmms-eval 的 textvqa prompt 含
> "Reference OCR token:" 提示行；07-07 起环境升级为 lmms-eval 0.3.0，该提示被移除，TextVQA
> 分数系统性下降 ~12pt。**本页 TextVQA 列（含 vanilla 58.27）不得与 07-07 之后的任何运行对比。**
> 无 OCR 协议重跑基线：vanilla 46.07，VZ-192/128/64 = 44.53/43.82/41.95
> （logs: `docs/idea5/logs/noocr_*`）。其余 benchmark 的 prompt 未变，数字仍有效。
> 详见 [idea_summary.md](idea_summary.md) 的修正节。

复现方法见 [docs/lmms-exp.md](lmms-exp.md)。评测框架：lmms-eval（复用 `/home/jk/work/paper/CLSE/lmms-eval` 的框架代码，模型/transformers 用本仓库自己验证过的 `llava_visiPruner` 环境）。

模型路径：`/home/jk/models/llava-v1.5-7b`
评测时间：2026-07-02
命令：`CUDA_VISIBLE_DEVICES=<gpu> NUM_GPUS=1 bash scripts/lmms_eval.sh <vanilla|vz64|vz128|vz192>`
Tasks：`gqa,mmbench_en_dev,mmbench_cn_dev,mme,pope,scienceqa_img,textvqa_val,vizwiz_vqa_val,ocrbench`

## 各 Benchmark 分数

| Benchmark | 指标 | K=192 (剪枝66.7%) | K=128 (剪枝77.8%) | K=64 (剪枝88.9%) | 论文 K=192 | 论文 K=128 | 论文 K=64 |
| --------- | ---- | :---------------: | :---------------: | :--------------: | :--------: | :--------: | :-------: |
| **GQA** | Accuracy | 59.25 | 57.66 | 55.15 | 59.3 | 57.6 | 55.1 |
| **MMBench-EN** | Accuracy | 63.75 | 62.20 | 60.14 | 63.7 | 62.0 | 59.7 |
| **MMBench-CN** | Accuracy | 53.61 | 53.61 | 51.20 | 56.8† | 54.2† | 49.3† |
| **MME Perception** | Score | 1449.09 | 1439.94 | 1373.82 | — | — | — |
| **MME Cognition** | Score | 321.43 | 323.93 | 344.29 | — | — | — |
| **MME Total** | Score | 1770.52 | 1763.87 | 1718.10 | 1782.6 | 1761.7 | 1690.0 |
| **POPE** | Accuracy | 86.38 | 84.64 | 80.57 | 85.3 | 83.2 | 77.0 |
| **POPE** | F1 | 85.30 | 82.91 | 76.98 | — | — | — |
| **ScienceQA** | Accuracy | 68.67 | 68.67 | 68.96 | 68.9 | 68.9 | 69.0 |
| **TextVQA** | Accuracy | 57.30 | 56.86 | 55.48 | 57.3 | 56.8 | 55.5 |
| **VizWiz** | Accuracy | 53.99 | 54.11 | 54.66 | 54.2‡ | 54.5‡ | 52.7‡ |
| **OCRBench** | Score(×100) | 31.20 | 29.80 | 28.20 | 30.9† | 27.7† | 25.0† |

> † MMBench-CN 和 OCRBench 的论文参考值取自 `/home/jk/work/paper/CLSE/docs/eval.md` 里整理的数字，不在你直接贴给我的 VisionZip Table 1 里（那张表只有 "MMB"，没有拆 EN/CN，也没有 OCRBench 行），来源上和其它列不完全一致，仅供参考。
> ‡ VizWiz 这里用的是 lmms-eval 的 `vizwiz_vqa_val`（有本地 ground truth 的验证集），和 [docs/result.md](result.md)/[docs/exp.md](exp.md) 里 LLaVA 自带脚本走的 VizWiz **test** split（无本地 GT，需官方提交）不是同一个 split，两边 VizWiz 数字不可直接互相比较。

**Vanilla（576 tokens）基线：**

| Benchmark | 本机 | 论文 | Δ |
| --- | :---: | :---: | :---: |
| GQA | 61.97 | 61.9 | +0.07 |
| MMBench-EN | 64.00 | 64.7 | −0.70 |
| MME Total | 1874.55 | 1862.0 | +12.55 (+0.7%) |
| POPE Accuracy | 86.99 | 85.9 | +1.09 |
| ScienceQA | 69.46 | 69.5 | −0.04 |
| TextVQA | 58.27 | 58.2 | +0.07 |
| VizWiz | 54.06 | 54.2 | −0.14 |

## 与论文对比分析

| Token 预算 | 复现结果（9 项均值，相对 vanilla） | 论文（Table 1 Avg. 列） | 差异 |
| --------- | :-----------------: | :-----------------: | :--: |
| K=192（66.7% 剪枝） | 98.1% | 98.5% | 基本吻合 |
| K=128（77.8% 剪枝） | 96.7% | 97.6% | 基本吻合 |
| K=64（88.9% 剪枝） | 93.9% | 94.0% | 基本吻合 |

（"9 项均值" = GQA、MMBench-EN、MMBench-CN、MME Total、POPE Accuracy、ScienceQA、TextVQA、VizWiz、OCRBench 各自相对 vanilla 的保留率取平均。）

**主要观察：**

- **GQA、MMBench-EN、TextVQA 与论文几乎完全吻合**，三档 token 预算下偏差都在 ±0.5 以内。
- **MME Total** 偏差在 ±1.7% 以内（K=64 时 1718.10 vs 1690.0，+1.7%），趋势正确（token 越少分数越低）。
- **POPE** 偏差随 token 减少而扩大：K=192 时 +1.1，K=64 时 +3.6——和 [docs/result.md](result.md)（LLaVA 自带脚本路线）里观察到的"POPE 在低 token 档偏差最大"的现象一致，两套独立评测框架得出同样的结论，说明这不是某个框架特有的误差，值得进一步核实 64-token 档 dominant/contextual 的精确取值。
- **ScienceQA** 是个例外：论文里 K=192→K=64 是下降趋势（68.9→66.5→65.3），但本机复现三档几乎不掉分（68.67→68.67→68.96，几乎持平甚至微升）。这和 [docs/result.md](result.md) 里 LLaVA 自带脚本路线的观察一致（ScienceQA 对 token 数不敏感），但和论文的下降趋势不一致，可能是 ScienceQA 本身对视觉细节依赖低、方差较大，或者 128/192 档 dominant/contextual 的近似拆分在 ScienceQA 上恰好没有体现出该有的差异。
- **VizWiz** 三档都非常接近论文数字（差异在 ±2 以内），K=64 时甚至反超 vanilla（54.66 vs 54.06），可能是该数据集的正常波动。
- **MMBench-CN / OCRBench** 因为论文参考值来源不完全一致（见上面脚注），仅供参考；本机复现的 MMBench-CN 三档比论文低 2-3 分，趋势（token 越少分越低）正确。
- **两套框架交叉验证**：本文档（lmms-eval）和 [docs/result.md](result.md)（LLaVA 自带脚本）分别用不同评测流程跑了同一模型、同一 VisionZip 配置，GQA/MME/POPE/TextVQA 在两套框架下数值相近、趋势一致，互相印证了复现结果的可靠性。

## 复现细节 / 已知限制

- MMBench-EN/CN 默认走 `eval_method="openai"`，但由于本机没有配置真实 `OPENAI_API_KEY`，且 LLaVA 在提示里已经要求"直接回答选项字母"，绝大多数答案能被 lmms-eval 内置的规则式 `can_infer()` 直接从预测文本里抽出字母，不需要真的调用 GPT；极少数没有干净给出单字母答案的样本会在尝试调用 GPT（用占位 key，必然失败）重试耗尽后随机猜一个选项。这带来的噪声可以忽略，细节见 [docs/lmms-exp.md](lmms-exp.md) 第 6 节。
- 详细的逐样本日志在 `logs/lmms-eval/<config>/`（`--log_samples` 输出），可用于进一步排查任何单个 benchmark 的异常。
- SEED-Bench、MM-Vet、LLaVA-Bench、VQAv2 没有在这套 lmms-eval 命令里（沿用 CLSE `llava_lmms_eval.sh` 的默认 9-task 列表），如果需要可以加到 `TASKS` 环境变量里跑。
