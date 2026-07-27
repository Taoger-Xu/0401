# Qwen 不同 Token Reduction 档位的性能比较

下图比较各方法在不同视觉 token reduction 档位下的平均性能保留率。横轴采用
原表的 **Token Reduction (↓)** 表述；纵轴为各方法相对其自身 Vanilla
baseline 的平均性能保留率（越高越好）。

![Qwen 不同 Token Reduction 档位下的平均性能雷达图](assets/qwen_token_reduction_radar.svg)

雷达图仅展示 ViTCoP、**Scope (ours)**、PDrop 和 SparseVLM；图中字体统一为
25pt。各轴数值均为相对各自 Vanilla baseline 的性能保留率。

## 作图数据

| Method | All Tokens (100%) | Token Reduction (↓ 66.7%) | Token Reduction (↓ 77.8%) | Token Reduction (↓ 88.9%) |
|---|---:|---:|---:|---:|
| Vanilla | 100.0% | — | — | — |
| FastV (ECCV 2024) | 100.0% | 94.9% | 91.3% | 82.2% |
| PDrop (CVPR 2025) | 100.0% | 93.1% | 89.9% | 81.5% |
| HiRED (AAAI 2025) | 100.0% | 94.1% | 90.8% | 83.6% |
| SparseVLM (ICML 2025) | 100.0% | 96.7% | 91.9% | 78.4% |
| CLSE (ours) | 100.0% | 97.6% | 95.8% | 90.5% |
| ViTCoP (ours) | 100.0% | 73.9% | 69.4% | 63.5% |
| **Scope (ours)** | **100.0%** | **96.0%** | **92.3%** | **84.8%** |

其中，FastV、PDrop、HiRED、SparseVLM 和 CLSE 的数值来自给定的
Qwen2-VL-7B 汇总表；ViTCoP 来自 [`qwen.md`](qwen.md)，Scope 来自
[`qwen-scope.md`](qwen-scope.md)。ViTCoP 和 Scope 的 Average 均直接采用
原文档中的 benchmark retention 算术平均值，没有对绝对分数再次混合平均。

## 各 Benchmark 绝对分数

### Qwen2-VL-7B：外部方法与 CLSE

该组实验使用 GQA、MMBench（MMB）、MMBench-CN（MMBCN）、MME、POPE、
ScienceQA（SQA）和 TextVQA（VQAText）。MME 使用原始总分，其余指标按原始
汇总表记录。Avg. 是相对 Vanilla 的综合性能保留率，并非各列绝对分数的直接
算术平均。

#### All Tokens (100%)

| Method | GQA | MMB | MMBCN | MME | POPE | SQA | VQAText | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 62.2 | 79.1 | 77.8 | 2296 | 87.7 | 85.4 | 81.4 | 100.0% |

#### Token Reduction (↓ 66.7%)

| Method | GQA | MMB | MMBCN | MME | POPE | SQA | VQAText | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FastV (ECCV 2024) | 58.6 | 74.5 | 74.3 | 2083 | 85.6 | 83.1 | 76.1 | 94.9% |
| PDrop (CVPR 2025) | 58.5 | 72.3 | 71.2 | 1991 | 86.1 | 82.8 | 76.4 | 93.1% |
| HiRED (AAAI 2025) | 59.9 | 73.8 | 73.4 | 2158 | 86.6 | 80.7 | 71.0 | 94.1% |
| SparseVLM (ICML 2025) | 59.2 | 77.1 | 76.0 | 2181 | 85.9 | 83.5 | 77.8 | 96.7% |
| CLSE (ours) | 61.0 | 76.5 | 75.5 | 2254 | 86.7 | 84.2 | 78.3 | 97.6% |

#### Token Reduction (↓ 77.8%)

| Method | GQA | MMB | MMBCN | MME | POPE | SQA | VQAText | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FastV (ECCV 2024) | 56.2 | 70.1 | 72.4 | 1957 | 82.7 | 82.3 | 73.0 | 91.3% |
| PDrop (CVPR 2025) | 56.4 | 67.5 | 67.2 | 1945 | 84.2 | 82.0 | 73.7 | 89.9% |
| HiRED (AAAI 2025) | 58.3 | 70.5 | 70.7 | 2086 | 85.4 | 79.6 | 65.4 | 90.8% |
| SparseVLM (ICML 2025) | 54.9 | 73.2 | 73.6 | 2047 | 81.1 | 83.2 | 72.2 | 91.9% |
| CLSE (ours) | 59.8 | 74.1 | 74.2 | 2209 | 85.5 | 83.5 | 76.6 | 95.8% |

#### Token Reduction (↓ 88.9%)

| Method | GQA | MMB | MMBCN | MME | POPE | SQA | VQAText | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FastV (ECCV 2024) | 50.6 | 61.8 | 62.5 | 1760 | 72.2 | 80.5 | 68.6 | 82.2% |
| PDrop (CVPR 2025) | 52.5 | 59.2 | 58.3 | 1756 | 74.1 | 80.2 | 67.7 | 81.5% |
| HiRED (AAAI 2025) | 55.5 | 64.3 | 64.3 | 1834 | 82.8 | 75.7 | 56.5 | 83.6% |
| SparseVLM (ICML 2025) | 47.5 | 59.5 | 61.1 | 1624 | 68.0 | 80.7 | 61.7 | 78.4% |
| CLSE (ours) | 56.1 | 69.3 | 69.2 | 2008 | 81.2 | 82.6 | 73.4 | 90.5% |

### Qwen2.5-VL-7B：ViTCoP

MME-P 使用原始分，OCRBench 使用 `/1000` 分数，其余指标为百分数。Avg. 为
八个 benchmark 相对各自 baseline 的 retention 算术平均值。

| Benchmark | All Tokens (100%) | Token Reduction (↓ 66.7%) | Token Reduction (↓ 77.8%) | Token Reduction (↓ 88.9%) |
|---|---:|---:|---:|---:|
| GQA | 60.45 | 57.40 | 55.12 | 50.72 |
| MME-P | 1691.52 | 1614.84 | 1506.81 | 1396.91 |
| MMStar | 62.53 | 52.26 | 47.32 | 40.89 |
| POPE (F1) | 86.14 | 82.51 | 78.42 | 67.62 |
| SQA-IMG | 76.25 | 70.90 | 68.62 | 67.87 |
| TextVQA | 80.83 | 18.75 | 16.22 | 14.07 |
| VizWiz | 70.90 | 63.73 | 62.53 | 60.83 |
| OCRBench | 828 | 125 | 81 | 42 |
| **Avg. retention** | **100.0%** | **73.9%** | **69.4%** | **63.5%** |

### Qwen2.5-VL-7B：Scope (ours)

Scope 使用与 ViTCoP 相同的八个 benchmark。OCRBench 在该实验记录中以百分制
呈现；Avg. 同样为八个 benchmark retention 的算术平均值。

| Benchmark | All Tokens (100%) | Token Reduction (↓ 66.7%) | Token Reduction (↓ 77.8%) | Token Reduction (↓ 88.9%) |
|---|---:|---:|---:|---:|
| GQA | 58.21 | 56.65 | 55.08 | 52.28 |
| MME-P | 1653.48 | 1592.73 | 1547.43 | 1455.28 |
| MMStar | 58.27 | 54.74 | 52.26 | 47.63 |
| POPE (F1) | 86.48 | 84.98 | 83.58 | 79.59 |
| SQA-IMG | 80.91 | 79.92 | 78.04 | 75.61 |
| TextVQA | 71.24 | 69.65 | 67.67 | 61.17 |
| VizWiz | 66.77 | 65.81 | 65.50 | 64.71 |
| OCRBench | 77.80 | 67.80 | 58.10 | 39.50 |
| **Avg. retention** | **100.0%** | **96.0%** | **92.3%** | **84.8%** |

## 结果概览

- 在 Scope 与外部方法可作趋势参考的三个 reduction 档位上，Scope 的平均保留率为
  96.0%、92.3% 和 84.8%；token 越少，性能平稳下降。
- Scope 在 Token Reduction (↓ 66.7%) 时接近领先方法，但在 Token Reduction
  (↓ 88.9%) 时与 CLSE 的差距扩大，说明极低 token 预算仍是主要压力点。
- 当前 Qwen 版 ViTCoP 明显偏低，主要原因是无 `[CLS]` 架构下使用 L2 norm
  saliency，导致 TextVQA 和 OCRBench 性能大幅下降；这组结果更适合反映当前
  移植实现的局限，而不是原方法能力上限。

## 可比性说明

这是一张**跨实验协议的归一化趋势图**，不应作为严格的同表 SOTA 排名：

- 外部方法与 CLSE 使用 Qwen2-VL-7B，并在 GQA、MMBench、MMBench-CN、MME、
  POPE、SQA 和 TextVQA 共 7 项任务上求平均。
- ViTCoP 与 Scope 使用 Qwen2.5-VL-7B，并在 GQA、MME-P、MMStar、POPE、
  SQA-IMG、TextVQA、VizWiz 和 OCRBench 共 8 项任务上求平均。
- 各曲线都除以各自实验的 Vanilla baseline，因此适合观察随 token 预算收紧时
  的相对退化趋势；若需严格比较方法优劣，应在相同模型、数据集、图像分辨率和
  评测流程下重新运行所有方法。
