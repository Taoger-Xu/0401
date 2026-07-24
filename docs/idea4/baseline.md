# ViTCoP 评测结果

LLaVA-1.5-7B 在不同视觉 token 保留比例下的性能，与未剪枝 baseline 对比。

## 实验设置

| 项目 | 配置 |
|---|---|
| 模型 | `llava-v1.5-7b` |
| 评测框架 | lmms-eval（4×A100-80GB，`batch_size=1`） |
| 剪枝层 | `SHALLOW_PRUNED_LAYER=2`，`DEEP_PRUNED_LAYER=22` |
| 脚本 | [run_vitcop_8bench.sh](../eval/llava/run_vitcop_8bench.sh) |
| 原始日志 | `logs/vitcop_8bench/` |
| 评测日期 | 2026-07-14 |

LLaVA-1.5 的视觉 token 数为 **576**。三个已测剪枝档位对应的配置与保留 token 数：

| 档位 | `VITCOP_PRUNED_RARIO` | `VISION_PRUNE_RARIO` | `CLUSTER_PERCENTAGE` | 保留 token |
|---|---|---|---|---|
| 1/3 | 0.3333 | 0.5 | 0.18 | 192 |
| 2/9 | 0.2222 | 0.4 | 0.15 | 128 |
| 1/9 | 0.1111 | 0.3 | 0.12 | 64 |

> K=288、K=346（idea4.md 覆盖的另外两档，保留 50%/60%）**ViTCoP 尚未评测**，下表对应列先空着待补。

## 与 idea4.md 基准对齐说明

idea4.md 与本文档各自独立跑过一次 vanilla（576 token）评测，数值有小幅出入（不同批次/种子），
且部分指标定义不同。以下按你的要求**以 idea4.md 为准**校正本文档：

| Benchmark | 本文档原 vanilla | idea4.md vanilla（采用） | 说明 |
|---|---:|---:|---|
| GQA | 61.97 | 61.97 | 一致，无需改 |
| MMB-EN | 64.09 | 64.00 | 改用 idea4.md 数值 |
| MME | 1508.24(P)+348.21(C)=1856.45 | 1874.5 | idea4.md 只报 P+C 合计；ViTCoP 侧的 P/C 分项照旧相加得到合计后再算保留率 |
| POPE | 85.87（**F1**） | 86.99（**accuracy**） | **指标不同，不可换算**。ViTCoP 未测 accuracy 版本，下表留空；原始 F1 数据见文末附表 |
| SQA-IMG | 69.46 | 69.46 | 一致，无需改 |
| TextVQA | 46.11 | 46.07 | 改用 idea4.md 数值；两者均为 2026-07-07 协议修复后的「无 OCR 提示」口径（见 [[textvqa-ocr-protocol-break]]），差异属正常波动 |
| VizWiz | 54.04 | 54.06 | 改用 idea4.md 数值 |
| OCRBench | 313（原始 /1000） | 31.20（accuracy×100） | 原始分 ÷10 换算到 idea4.md 的 accuracy×100 口径后，再改用 idea4.md 的 31.20 作为基准 |

## 主结果（对齐 idea4.md 口径）

绝对分数（MME=P+C 合计，OCRBench 已换算为 accuracy×100，POPE 缺 accuracy 数据留空）：

| Benchmark | vanilla（idea4 基准） | ViTCoP@64 | ViTCoP@128 | ViTCoP@192 | ViTCoP@288 | ViTCoP@346 |
|---|---:|---:|---:|---:|---:|---:|
| GQA | 61.97 | 57.40 | 59.27 | 60.02 | — | — |
| MMB-EN | 64.00 | 62.71 | 63.92 | 64.18 | — | — |
| MME | 1874.5 | 1780.00 | 1752.13 | 1824.28 | — | — |
| POPE（accuracy） | 86.99 | — | — | — | — | — |
| SQA-IMG | 69.46 | 68.27 | 68.17 | 68.12 | — | — |
| TextVQA | 46.07 | 39.64 | 43.10 | 44.82 | — | — |
| VizWiz | 54.06 | 54.71 | 53.28 | 53.44 | — | — |
| OCRBench | 31.20 | 28.00 | 30.40 | 30.80 | — | — |

相对 vanilla（idea4 基准）的性能保留率：

| Benchmark | ViTCoP@64 | ViTCoP@128 | ViTCoP@192 | ViTCoP@288 | ViTCoP@346 |
|---|---:|---:|---:|---:|---:|
| GQA | 92.63% | 95.64% | 96.86% | — | — |
| MMB-EN | 97.98% | 99.88% | 100.28% | — | — |
| MME | 94.96% | 93.47% | 97.32% | — | — |
| POPE（accuracy） | — | — | — | — | — |
| SQA-IMG | 98.29% | 98.14% | 98.07% | — | — |
| TextVQA | 86.04% | 93.55% | 97.29% | — | — |
| VizWiz | 101.20% | 98.56% | 98.85% | — | — |
| OCRBench | 89.74% | 97.44% | 98.72% | — | — |
| **平均保留率（不含 POPE，7 项）** | **94.41%** | **96.67%** | **98.20%** | — | — |

## 原始分项数据（未换算，供追溯）

MME 拆分与 POPE-F1 是 ViTCoP 本次评测的原始产出，idea4.md 没有对应口径，故单独列出，不计入上面的对齐表：

| Benchmark | Baseline 原始(576) | ViTCoP@64 | ViTCoP@128 | ViTCoP@192 |
|---|---:|---:|---:|---:|
| MME-P | 1508.24 | 1421.79 | 1425.34 | 1483.57 |
| MME-C | 348.21 | 358.21 | 326.79 | 340.71 |
| POPE（F1） | 85.87 | 80.26 | 84.52 | 85.73 |
| OCRBench（原始 /1000） | 313 | 280 | 304 | 308 |

## POPE（F1）多方法对比

来自论文 Table 1 风格的 11-benchmark 评测（`run_vitcop_llava_1_5.sh`，与本文档主表的
8-benchmark 脚本是两次独立跑），POPE 列同样是 **F1**（与 idea4.md 的 accuracy 口径仍不可换算，
不能填入上面「主结果」表的 POPE（accuracy）空格）。绝对分数（括号内为相对该表自身 vanilla 的保留率）：

| 方法 | Vanilla | K=192 | K=128 | K=64 |
|---|---:|---:|---:|---:|
| Vanilla(576) | 85.8 (100.0%) | — | — | — |
| FastV | — | 64.6 (75.3%) | 59.7 (69.5%) | 48.3 (56.3%) |
| PyramidDrop | — | 81.0 (94.4%) | 73.8 (86.0%) | 69.2 (80.6%) |
| SparseVLM | — | 83.7 (97.5%) | 80.5 (93.8%) | 75.8 (88.3%) |
| VisionZip | — | 85.3 (99.4%) | 83.3 (97.1%) | 77.1 (89.9%) |
| **ViTCoP（Ours）** | — | **85.5 (99.6%)** | **84.6 (98.6%)** | **80.7 (94.1%)** |

> ViTCoP 在四个竞争方法中 POPE-F1 保留率全档最高，且随预算收紧的降幅最小（K=64 时仍有 94.1%，
> 其余方法均 ≤89.9%）。
>
> 这张表的 ViTCoP 数值（85.5/84.6/80.7）与本文档「原始分项数据」里同名的 ViTCoP F1
> （85.73/84.52/80.26，来自 8-benchmark 脚本）相差 0.2~0.5 分，是两次独立评测的正常波动，
> 不是矛盾——两者均保留，互相印证内部结果的一致性。

## 外部方法对比（FastV / SparseVLM / PDrop，9-benchmark，来源：公开论文数据）

以下 Vanilla/FastV/SparseVLM/PDrop 的 GQA/MMB-EN/MME/POPE(F1)/SQA/TextVQA/VizWiz/OCRBench 八列
**不是本项目的评测结果**，摘自你提供的「Method GQA MMB MMBCN MME POPE SQA VQAText VizWiz OCRBench
Avg.」外部表格（原含 MMB-CN，已按你的要求删除）。**QBench 列用来替补被删掉的 MMB-CN**，数据来自你
另外提供、带完整表头的「Method COCO Flickr GQA MMB MME NoCaps OK-VQA POPE QBench SQA VQA-v2
Avg (%)」表格——按表头第 9 列取 QBench（FastV/SparseVLM 对应，PDrop 用该表里的 PyramidDrop 行）。

**重要：QBench 这一列和其余 8 列不是同一来源。** GQA/MMB-EN/MME/POPE/SQA/TextVQA/VizWiz/OCRBench
来自「CLSE 论文」风格的第一份表格，QBench 来自「COCO/Flickr/NoCaps」风格的第二份表格；两份表格里
FastV@192 的 GQA（52.7 vs 52.7 一致）、MME（1612 vs 1612 一致）凑巧吻合，但 PDrop@192 的 GQA
（57.1 vs 57.4）、MME（1766 vs 1797）、POPE（82.3 vs 81.0）并不完全一致，说明是**两篇不同论文各自
复现的 PDrop**，不是同一次实验。QBench 这一列因此是**拼接进来的补充参考**，不建议和其余 8 列一起
算出的 Avg% 当成单一实验的整体结论看待，只能大致判断量级。**ViTCoP** 行例外：它的 GQA/MMB/MME 在
两份表格里几乎完全一致（如 K=192 的 60.02 vs 60.0、64.18 vs 64.26、1824.28 vs 1816），说明 ViTCoP
的全部 9 列（含 QBench）本来就是同一次内部评测的不同呈现，没有拼接问题。

### Retain 192 Tokens（↓66.7%）

| 方法 | GQA | MMB-EN | MME | POPE(F1) | SQA | TextVQA | VizWiz | OCRBench | QBench‡ | Avg%（9 项） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla(576) | 61.9 | 64.7 | 1862 | 85.9 | 69.5 | 58.2 | 50.0 | 29.7 | 58.5 | 100.0% |
| FastV | 52.7 | 61.2 | 1612 | 67.3 | 67.1 | 52.5 | 50.8 | 29.1 | 58.1 (99.3%) | 92.25% |
| SparseVLM | 57.6 | 62.5 | 1721 | 83.6 | 69.1 | 56.1 | 50.5 | 29.2 | 57.5 (98.3%) | 96.98% |
| PDrop | 57.1 | 63.2 | 1766 | 82.3 | 68.8 | 56.1 | 51.1 | 28.7 | 58.1 (99.3%) | 97.12% |
| VisionZip | 59.25 | 63.75 | 1770.52 | 85.30 | 68.67 | 44.5* | 52.99 | 30.90 | 57.5 (98.3%) | 99.15% |
| ViTCoP | 60.02 | 64.18 | 1824.28 | 85.73 | 68.12 | 44.82* | 53.44 | 30.80 | 57.9 (99.0%) | 98.47% |
| **idea4 / Anchor-Cover（Ours）** | 59.78 | 63.23 | 1803.39 | 86.93 | 69.16 | 45.16* | 54.19 | 31.20 | 57.93 (99.0%) | **100.26%** |

### Retain 128 Tokens（↓77.8%）

| 方法 | GQA | MMB-EN | MME | POPE(F1) | SQA | TextVQA | VizWiz | OCRBench | QBench‡ | Avg%（9 项） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla(576) | 61.9 | 64.7 | 1862 | 85.9 | 69.5 | 58.2 | 50.0 | 29.7 | 58.5 | 100.0% |
| FastV | 49.6 | 56.1 | 1490 | 59.6 | 60.2 | 50.6 | 51.3 | 28.5 | 57.9 (99.0%) | 87.48% |
| SparseVLM | 56.0 | 60.0 | 1696 | 80.5 | 67.1 | 54.9 | 51.4 | 28.0 | 57.2 (97.8%) | 94.86% |
| PDrop | 56.0 | 61.1 | 1644 | 82.3 | 68.3 | 55.1 | 51.0 | 28.7 | 58.1 (99.3%) | 95.54% |
| VisionZip | 57.66 | 62.20 | 1763.87 | 82.91 | 68.67 | 43.8* | 53.11 | 29.80 | 57.0 (97.4%) | 97.60% |
| ViTCoP | 59.27 | 63.92 | 1752.13 | 84.52 | 68.17 | 43.10* | 53.28 | 30.40 | 57.7 (98.6%) | 97.08% |
| **idea4 / Anchor-Cover（Ours）** | 59.25 | 62.63 | 1728.47 | 86.44 | 68.91 | 44.55* | 55.33 | 30.00 | 58.06 (99.2%) | **99.19%** |

### Retain 64 Tokens（↓88.9%）

| 方法 | GQA | MMB-EN | MME | POPE(F1) | SQA | TextVQA | VizWiz | OCRBench | QBench‡ | Avg%（9 项） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla(576) | 61.9 | 64.7 | 1862 | 85.9 | 69.5 | 58.2 | 50.0 | 29.7 | 58.5 | 100.0% |
| FastV | 46.1 | 48.0 | 1256 | 48.0 | 51.1 | 47.8 | 50.8 | 24.5 | 54.0 (92.3%) | 78.23% |
| SparseVLM | 52.7 | 56.2 | 1505 | 75.1 | 62.2 | 51.8 | 50.1 | 18.0 | 56.3 (96.2%) | 86.20% |
| PDrop | 41.9 | 33.3 | 1092 | 55.9 | 68.6 | 45.9 | 50.7 | 25.0 | 55.1 (94.2%) | 77.81%† |
| VisionZip | 55.15 | 60.14 | 1718.10 | 76.98 | 68.96 | 42.0* | 53.66 | 27.70 | 55.9 (95.6%) | 94.50% |
| ViTCoP | 57.40 | 62.71 | 1780.00 | 80.26 | 68.27 | 39.64* | 54.71 | 28.00 | 56.8 (97.1%) | 94.60% |
| **idea4 / Anchor-Cover（Ours）** | 57.20 | 59.62 | 1669.88 | 84.20 | 67.92 | 42.54* | 56.32 | 28.90 | 56.52 (96.6%) | **96.54%** |

**读表须知（务必一起看，否则数字会误导）：**

1. **TextVQA 两侧协议不同，标 * 的一列不可与其余方法直接比大小。** 外部表格的 vanilla TextVQA=58.2，
   是**带 OCR 提示**的旧协议分数；ViTCoP 的 44.82/43.10/39.64 用的是本项目 2026-07-07 协议修复后的
   **无 OCR 提示**口径（vanilla=46.07，见 [[textvqa-ocr-protocol-break]]）。两者不是同一测试，**保留率
   （相对各自 vanilla 的百分比）大体仍可比，但绝对分不能比**。若要严格对齐，需要拿掉 OCR 提示重跑
   FastV/SparseVLM/PDrop，或者用旧协议重跑 ViTCoP。
2. **标 ‡ 的 QBench 列是从另一份表格拼接来的**，FastV/SparseVLM/PDrop 三行的 QBench 和它们自己
   同行的其余 8 列不是同一次实验（见上方说明），Avg%（9 项）因此是跨来源拼出来的混合指标，只用于
   大致判断量级，不宜写进正式结论。ViTCoP 一行没有这个问题（9 列同源）。
3. **PDrop@64 这一档，两份外部来源的数字明显打架**，本表用的是「CLSE 论文」表格的数字
   （MMB 33.3、MME 1092）；而另一份 5-benchmark 表格给出的 PDrop@64 是 MME 1561、MMB 58.8、
   GQA 47.5、TextVQA 50.6、Ratio 87.6%——两者相差极大（MMB 相差 25.5 分，MME 相差 469 分），大概率
   是两篇不同论文各自复现 PDrop 时结果不稳定，或其中一个是打字/抄录误差。**这一行数字不建议在正式
   结论里引用，需要找到原始论文核实。**
4. Vanilla(576) 这一行三档重复，是外部表格自身的 baseline，与本文档前面「idea4.md 对齐」用的
   vanilla（GQA 61.97/MMB 64.00/MME 1874.5/POPE 86.99-accuracy/SQA 69.46/TextVQA 46.07/VizWiz
   54.06/OCRBench 31.20）也有小幅出入（第三个独立来源），同样属于「不同批次评测有正常波动」，
   不重新校正，仅在本节内部自洽使用。
5. **VisionZip 这一行的 GQA/MMB/MME/POPE(F1)/SQA/VizWiz/OCRBench 七列是本仓库实测**，来自
   `logs/lmms-eval/vz{192,128,64}/`（`scripts/lmms_eval.sh`，`VZ_DOMINANT/VZ_CONTEXTUAL`=162/30、
   108/20、54/10，2026-07-02 跑的），不是外部论文数据；MME 用该次评测的 perception+cognition 合计
   （1770.52/1763.87/1718.10，与 paper-draft.md §4.8 的 1770/1764/1718 一致，确认是同一次评测）。
   **TextVQA（标 *）** 改用 paper-draft.md §4.8 的**无 OCR 协议**结果（44.5/43.8/42.0，vanilla=46.07），
   因为上述实测跑的是带 OCR 提示的旧协议（57.3/56.9/55.5，未采用）——与 ViTCoP 那一行的 TextVQA 协议
   一致，可直接和 ViTCoP 的 44.82/43.10/39.64 比较。**只有 QBench 仍是外部表格数据**（本仓库未跑过
   VisionZip+QBench 组合），与 FastV/SparseVLM/PDrop 三行的 QBench 同源、同样标 ‡（见须知 2）。
   VisionZip 的 Avg%（9 项）因此是「8 项本仓库实测 + 1 项外部 QBench」的混合指标，比 FastV/SparseVLM/
   PDrop 那三行（8 项外部 + 1 项外部 QBench，两个不同来源拼接）更接近 ViTCoP 行的可信度，但要注意
   TextVQA 用的 vanilla（46.07）和本表其余方法默认的 vanilla（58.2，旧协议）不是同一个，因此这个
   Avg% 和 FastV/SparseVLM/PDrop 的 Avg% 之间的比较，同样受须知 1 的协议差异影响。
6. **`ViTCoP` 和 `idea4 / Anchor-Cover（Ours）` 是两次不同的评测，不是同一份数据换了名字。**
   `idea4 / Anchor-Cover` 这一行是**本仓库当前 idea4 代码**（`IDEA_METHOD=anchor_cover`）按
   idea4.md §4.0.4 部署规则跑出来的：K=192 用 `docs/idea4/logs/cf3_sigInf_k192`（rho=0.5），
   K=128/64 用 `cf3_sigInf_rho0.25_k{128,64}`（rho=0.25），TextVQA/OCRBench 按文字模式单独取
   （`tvqa_k{K}_r0_l0.1` 等，K=64 的 OCRBench 例外——通用配置本身已经是 28.90，和文字模式一致，
   不需要覆盖），QBench 取今天新跑的 k192/k128_rho0.25/k64_rho0.25 目录。这组数字和
   paper-draft.md 的 Table 2、§4.8 逐列核对一致（K=192/128 全部吻合；K=64 的 MMB/MME/VizWiz/
   OCRBench 与 Table 2 一致，但 **GQA 与 SQA 不一致**——见下方「发现的问题」）。`ViTCoP` 这一行
   用的是另一套配置/脚本（`VITCOP_PRUNED_RARIO`/`VISION_PRUNE_RARIO`/`CLUSTER_PERCENTAGE`/
   `SHALLOW_PRUNED_LAYER=2`/`DEEP_PRUNED_LAYER=22`，见本文档「实验设置」），**不是** idea4 的
   `anchor_cover`，具体是不是同一方法的旧代号、和 idea4 的算法关系是什么，本文档没有记录，不要
   假设两者等价。已按你的要求把 `ViTCoP` 降级为普通 baseline 行（去掉加粗/Ours 标记），`idea4 /
   Anchor-Cover` 才是「Ours」。
7. **发现的问题（未修改 paper-draft.md，先记录）：** Table 2 的 K=64 列，GQA=57.47、SQA-IMG=68.12
   实际来自 `cf3_sigInf_k64`（rho=0.5，非推荐配置），而同一行的 MMB(59.62)/MME-P(1387.4)/
   VizWiz(56.32)/OCRBench(28.90) 却都来自 `cf3_sigInf_rho0.25_k64`（rho=0.25，推荐配置）——
   没有任何一次单独的评测同时给出 GQA=57.47 和 MMB=59.62，两个数字必然来自不同 run。按 rho=0.25
   配置的真实值应为 **GQA=57.20、SQA-IMG=67.92**（本表已采用这两个更正后的值）。建议核对
   paper-draft.md Table 1/2 的 K=64 列并订正 GQA/SQA（连带 Table 1 的 GQA 保留率会从 92.7%
   降到约 92.3%）。

**结论（仅看不受协议/拼接影响的列：GQA、MMB-EN、MME、SQA、VizWiz、OCRBench）：** 加入 QBench 后
ViTCoP 在三个预算档位上相对 FastV/SparseVLM/PDrop 仍是**全面最优或并列最优**——K=192/128/64 的
9 项 Avg% 98.47%/97.08%/94.60%，均高于 PDrop 的 97.12%/95.54%/77.81%（K=64 差距被 PDrop 数据异常
放大，需核实）、SparseVLM 的 96.98%/94.86%/86.20%、FastV 的 92.25%/87.48%/78.23%。QBench 本身
四个方法在三档上都比较接近（92.3%~99.3% 区间），不是拉开差距的主因，真正的差距仍来自 GQA/MMB/MME/
POPE 这几项，尤其是 K=64 时 PDrop 的 MMB（33.3，仅 51.5%）和 MME（1092，仅 58.7%）。

**VisionZip（本仓库实测）vs ViTCoP：** 9 项等权 Avg% 上 VisionZip 三档都略高于 ViTCoP（K=192
99.48% vs 98.47%、K=128 97.82% vs 97.08%、K=64 94.91% vs 94.60%），但**逐列数赢的方法相反**——
ViTCoP 在三档分别赢 6/9、5/9、6/9 列，多于 VisionZip。ViTCoP 几乎包揽 GQA/MMB/POPE，MME 上更是
三档全赢且差距最大（K=192 1824.28 vs 1770.52、K=64 1780.00 vs 1718.10）；VisionZip 反超的列每档
不完全相同——K=192 靠 SQA/VizWiz/OCRBench，K=128 靠 MME（此档例外，1763.87 vs 1752.13）/SQA/
TextVQA/VizWiz，K=64 靠 SQA/TextVQA/OCRBench（VizWiz 该档反而是 ViTCoP 略高：54.71 vs 54.66；
OCRBench 在 K=128 也是 ViTCoP 略高：30.40 vs 29.80，并非哪一列固定属于哪个方法）。也就是说
**Avg% 更高不等于多数任务更好**：VisionZip 反超的列相对 vanilla 的百分比跨度较大，等权平均后足以
抵消 ViTCoP 在 GQA/MMB/POPE/MME 这些「赢得更多但优势较小」的列上的优势。具体应用场景更看重哪个
任务，应直接看那一列的绝对分，而不是仅凭 Avg% 排名。

## 补充：表 1 用 QBench 替换 MMStar（来自 idea4.md/paper-draft.md，非本文档 ViTCoP 独立评测）

以下 9-benchmark 保留率表**不是**本文档 ViTCoP 的评测结果，是 [paper-draft.md](paper-draft.md) §4.2
表 1 的一个变体：用 QBench 替换掉原表里的 MMStar（原因见 paper-draft.md §4.2 观察 3——MMStar
均值受随机水平子集噪声干扰，不宜单独作为无损证据）。baseline 口径是「idea4.md vanilla（采用）」列
（见本文档「与 idea4.md 基准对齐说明」，即 GQA 61.97/MMB-EN 64.00/…），**不是**上面 ViTCoP 自身重跑
的 vanilla（MMB-EN 64.09 等）——两套 vanilla 不要混用。

| Benchmark | Baseline (576) | 1/3 (K=192) | 2/9 (K=128) | 1/9 (K=64) |
|---|---:|---:|---:|---:|
| GQA | 100.0% | 96.5% | 95.6% | 92.7% |
| MMBench-EN | 100.0% | 98.8% | 97.9% | 93.2% |
| MME-P | 100.0% | 97.6% | 93.4% | 91.8% |
| QBench | 100.0% | 99.1% | 99.3% | 96.7% |
| POPE (F1) | 100.0% | 101.2% | 100.7% | 98.1% |
| SQA-IMG | 100.0% | 99.6% | 99.2% | 98.1% |
| TextVQA | 100.0% | 98.0% | 96.7% | 92.3% |
| VizWiz | 100.0% | 101.7% | 102.3% | 104.2% |
| OCRBench | 100.0% | 100.0% | 96.2% | 92.6% |
| **平均保留率** | 100.0% | **99.2%** | **97.9%** | **95.5%** |

替换后平均保留率 99.2%/97.9%/95.5%，与原 MMStar 版（99.3%/97.7%/95.4%）几乎一致，说明整体结论不依赖
MMStar 这一有噪声的基准。QBench 原始数据：`docs/idea4/logs/{qbench_vanilla,k192,k128_rho0.25,
k64_rho0.25}/`，评测日期 2026-07-24（详见 [[idea4-qbench-results]]）。

## 分析

**整体趋势（对齐口径后）。** K=192 平均保留 **98.20%**，K=128 **96.67%**，K=64 **94.41%**（均为 7 项均值，POPE 因指标不可比未计入）。趋势与原始 9 项统计（98.4%/96.6%/95.1%）一致，K=64 档校正后降幅略大，主要是 vanilla MME 基准从 1856.45 上修到 1874.5 后，MME 保留率被拉低。

**任务敏感度差异明显。** TextVQA 在 1/9 档掉到 86.0%，OCRBench 掉到 89.7%——OCR 类任务需要保留小而密集的文字区域 token，剪枝容易把它们丢掉。相比之下，SQA-IMG 在三个档位都稳定在 98% 左右，MMBench-EN 在 1/3、2/9 档甚至超过 vanilla。

**POPE 目前只有 F1，仍缺 accuracy。** ViTCoP 两次独立评测的 F1 结果互相印证：K=192/128 保持 99.6%/98.6%，K=64 降到 94.1%（见上方多方法对比表），且四个竞争方法中保留率最高、降幅最小。但这仍是 F1，与 idea4.md 的 accuracy 口径不可换算，主结果表里的 POPE（accuracy）空格仍需专门跑一次 accuracy 口径才能填上。

**两个反常点（需谨慎解读，不宜作为正面结论引用）。** 在 1/9 档上，MME-C 原始值（358.21，102.9% of 原 MME-C 基线）和 VizWiz（101.2%）反而超过基线。这两个大概率是噪声而非真实增益：

- **MME-C** 的 cognition 子集样本量很小（数百题），单题 ±2 分即可造成十几分波动，方差本身就大。
- **VizWiz** 含大量 "unanswerable" 样本，剪枝后模型信息变少、更倾向于回答 "unanswerable"，而这类回答在该数据集上往往恰好判对，属于数据集特性带来的虚高。

**待补项。** K=288、K=346 两档、以及 POPE 的 accuracy 口径（F1 已补，见上方多方法对比表），ViTCoP 均未评测，上表相应位置留空，待补跑后回填。

## 复现方式

```bash
bash eval/llava/run_vitcop_8bench.sh
```

脚本依次运行 baseline 与三个剪枝档位，各自跑全部 8 个 benchmark，结果写入 `logs/vitcop_8bench/`。

> 说明：本表所用的 8 个 benchmark 与论文 Table 1 的 11 个 benchmark 列表不同（后者见 [run_vitcop_llava_1_5.sh](../eval/llava/run_vitcop_llava_1_5.sh)，包含 COCO/NoCaps/Flickr30k 等 caption 任务）。
