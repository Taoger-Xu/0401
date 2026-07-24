# idea4 效率分析（Efficiency Analysis）

Anchor-Cover（idea4，`IDEA_METHOD=anchor_cover`）把视觉 token 从 576 降到 `K`，剪枝点在 CLIP **完整** 前向之后、
`mm_projector` 之前（见 [idea4.md](idea4.md) §1、§2.4）——即 ViT 编码器算力两种配置完全一致，
真正的省算力/省显存收益全部来自**语言模型（Vicuna-7B）prefill 阶段序列变短**。本节直接测量这部分收益。

## 实验设置

| 项目 | 配置 |
|---|---|
| 模型 | `llava-v1.5-7b` |
| Benchmark | POPE（`lmms-lab/POPE` test split，9000 条，均匀采样 50 条：`image`+`question`） |
| 对比档位 | Baseline（vanilla，576 视觉 token） vs idea4 `K=64`（`ρ=0.5, λ=0.5, cover_factor=3.0`，与 idea4.md §4.0 的「通用配置」一致） |
| 硬件 | 单卡 A100-80GB（GPU4，运行前用 `nvidia-smi` 确认 0% 利用率/0 显存占用，避免与同机其他任务抢卡；本机 GPU7 硬件有问题会 hang，一律排除），fp16，Anchor-Cover 使用 SDPA，`batch_size=1` |
| 脚本 | [scripts/idea4_efficiency_bench.py](../../scripts/idea4_efficiency_bench.py) |
| 原始结果 | `docs/idea4/logs/efficiency_baseline.json`、`docs/idea4/logs/efficiency_idea4_k64.json` |
| 测量日期 | 2026-07-24 |

**测量方法**：真实模型前向（只做一次 prefill，`use_cache=True`，不生成），5 个 warmup 样本不计入统计；
`torch.cuda.Event` 配 `torch.cuda.synchronize` 计时；显存取 `torch.cuda.max_memory_allocated` 峰值（统计前 `reset_peak_memory_stats`）；
KV cache 按 `past_key_values` 实际张量字节数求和（非估算）；TFLOPs 用标准 2×MAC 约定解析计算 **LLM 主干**开销
（QKVO 投影 + 注意力 QK^T/·V + FFN 三矩阵，32 层求和），不含 CLIP 视觉编码器——因为两种配置的 ViT 前向完全相同，
计入只会同等地抬高两侧数值、稀释相对差异，不计入更能反映 idea4 实际省下的算力。

## 结果

**Table 5. Efficiency comparison on LLaVA-1.5-7B.** FLOPs 和 Performance 均报告相对
Vanilla 的保留比例；Throughput 为 `batch_size=1` 下按 `1000 / Prefilling Time (ms)` 计算的 prefill
吞吐量。PDrop 与其他方法来自不同的独立测量进程，因此其绝对 ms 只作参考，Speedup 按各自实验的
vanilla baseline 归一化。

| Methods | Visual Tokens ↓ | Prefilling Time ↓ (ms) | FLOPs ↓ | KV Cache ↓ (MB) | Performance ↑ | Throughput ↑ (samples/s) | Speedup ↑ (Prefilling) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla LLaVA-1.5-7B | 576 | 110.97 | 100.0% | 318.72 | 100.0% | 9.01 | 1.00× |
| PDrop-64 (budget-matched)‡ | 63.8 (avg.) | 68.94† | 19.4% | 62.63 | — | 14.51† | 1.57×† |
| ViTCoP | 64 | 78.35 | 20.0% | **62.50** | 94.60% | 12.76 | 1.42× |
| **Anchor-Cover / SCRAP (Ours)** | **64** | **57.02** | **19.3%** | 62.72 | **96.54%** | **17.54** | **1.95×** |

† PDrop 的 prefill 数值来自其 eager-attention 独立实验，对应 vanilla 为 108.56 ms；1.57× 相对该
baseline 计算，不用于与 SDPA 测得的绝对 ms 直接比较。确定性口径已交叉校验：两套测量的 vanilla 平均序列长度
均为 637.44，LLM 主干均为 8.469 TFLOPs，KV cache 均为 318.72 MB。

‡ PDrop-64 不是 PyramidDrop 官方配置，而是为对齐 64-token 预算构造的对照点：保留官方 λ=0.5 的四阶段几何衰减，
将调度改为 `layer_list=[2,8,16]`、`image_token_ratio_list=[0.111,0.056,0.028]`，即 576→63→32→16，
32 层平均 63.8 个视觉 token。

## 分析

**TFLOPs 与 KV cache 是最干净的收益，且与序列长度降幅（80.3%）高度吻合。** K=64 比 vanilla 少了约 512 个视觉 token，
总序列从 637 降到 125（POPE 问题本身只有约 60 个文本 token，恒定不变）。LLM 主干 FLOPs 在此长度区间几乎线性于
序列长度（`d_model=4096` 远大于 `n`，线性投影项 `8·n·d²` 主导，二次项 `4·n²·d` 可忽略），因此 TFLOPs 降幅
（80.7%）和 KV cache 降幅（80.3%）都紧贴 token 降幅，这是 idea4 最直接、最可信的收益来源。

**同预算的 PDrop-64 将计算量和 KV 压缩到了与其他 64-token 方法相同的量级。** 将剪枝层前移后，它的
LLM 主干降至 1.640 TFLOPs（保留 19.4%，5.16×），KV cache 降至 62.63 MB（保留 19.6%，5.09×），
与 ViTCoP/Anchor-Cover 的约 80% 降幅基本一致。但 PDrop 在三个剪枝层需要额外计算注意力打分，并执行
`topk`/`sort`/`index_select` 与 mask 重建；加上 eager attention 的固定调度开销，虽然 FLOPs 减少 80.6%，
wall-clock 只从 108.56 ms 降到 68.94 ms（-36.5%，1.57×）。因此 Table 5 中对 PDrop 最稳妥的结论是：
理论算力和 KV 收益主要由平均 token 预算决定，真正的系统差异则体现在 wall-clock 和精度上（PDrop-64 精度尚未补测）。

**Prefill 总耗时下降 48.6%（110.97→57.02 ms，1.95×）。** 该结果使用 SDPA。32 层逐层调度、embedding、
mask 构造、norm 以及 Anchor-Cover 选择本身等固定开销仍包含在端到端 prefill 中，并不会随 token 数同比
下降，因此实际延迟加速仍低于理论 LLM FLOPs 的 5.19×，但明显优于旧的 80.00 ms 测量结果。

**Prefill Time/Tok 在短序列下是失真指标，不建议作为主结论引用。** 它把"总耗时"除以"总 token 数"，
分子里包含上述固定开销、分母却大幅缩小，于是 idea4 反而"看起来更贵"（约 0.45 vs 0.17 ms/token）。
这不代表剪枝变慢——**总耗时**（更贴近用户实际等待的指标）实际下降了 48.6%。这一列保留是为了如实呈现测量结果，
但结论应以"Prefill 总耗时"和"TFLOPs"为准。

**GPU 显存峰值几乎不变（+0.9%，在噪声范围内）。** 单样本 `batch_size=1` 时，7B 模型权重本身占用约 14GB，
主导了峰值显存；576→64 视觉 token 省下的激活值/KV cache 只有几百 MB 量级，相对权重是九牛一毛，因此峰值显存对
token 数几乎不敏感。idea4 侧还略高（+129MB），可能来自选择算法本身（Voronoi 分 cell、DCT 打分等）的临时张量。
**显存收益会在更大 batch size、更长上下文或多轮/多图（KV cache 主导显存）场景下显著放大**——本次单样本测量低估了这一收益，
后续如需展示显存优势，应在更大 batch/更长生成长度下复测。

## 复现方式

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> /home/jk/miniconda3/envs/llava_visiPruner/bin/python \
    scripts/idea4_efficiency_bench.py --variant baseline --n-samples 50 --warmup 5 \
    --out docs/idea4/logs/efficiency_baseline.json

CUDA_VISIBLE_DEVICES=<idle_gpu> /home/jk/miniconda3/envs/llava_visiPruner/bin/python \
    scripts/idea4_efficiency_bench.py --variant idea4 --k 64 --n-samples 50 --warmup 5 \
    --out docs/idea4/logs/efficiency_idea4_k64.json
```

> 两个变体分别在独立进程中跑（各自新建 CUDA context），避免峰值显存互相污染。GPU7 硬件有问题会 hang，测效率时同样应避开。

## 附：QBench（图像质量评估）不同档位效果

用 lmms-eval 的 `qbench_dev`（`q-future/Q-Bench-HF`，dev split，1495 条，单选题问答，
`qbench_acc`=exact-match×100）补测 idea4 在 K=192/128/64 三档的效果。QBench 是感知类/全图任务
（不属于文字密集任务），按 idea4.md §4.0.4「一句话部署规则」使用**通用配置**：
K=192 用 `ρ=0.5, λ=0.5`，K=128/64 用 `ρ=0.25, λ=0.5`（`cover_factor=3`、σ→∞ 冻结不变）。

| 配置 | (ρ, λ) | QBench Acc | 相对 vanilla 保留率 |
|---|---|---:|---:|
| vanilla (576 tok) | — | 58.46 | 100.00% |
| idea4 K=192 | 0.5, 0.5 | 57.93 | 99.08% |
| idea4 K=128 | 0.25, 0.5 | **58.06** | **99.31%** |
| idea4 K=64 | 0.25, 0.5 | 56.52 | 96.68% |

模型 `llava-v1.5-7b`，4×A100-80GB（GPU0/1/2/4）并行跑，`batch_size=1`，评测日期 2026-07-24。
原始结果：`docs/idea4/logs/qbench_vanilla/`、`docs/idea4/logs/k192/`（复用已有目录，本次追加
`qbench_dev` 任务）、`docs/idea4/logs/k128_rho0.25/`、`docs/idea4/logs/k64_rho0.25/`。

**观察**：三档全部 ≥96.7% 保留率，且 K=128 反而略高于 K=192（58.06 vs 57.93，均在 vanilla 58.46
附近的噪声范围内）——与 idea4.md §4.0 里 VizWiz/POPE 等「小预算不降反升」的现象一致，说明 QBench
这类图像质量感知任务对 CLIP CLS 注意力主干选出的 token 已经足够，对视觉 token 数不敏感，
idea4 在该任务上剪枝到 K=64（88.9% 剪枝率）仍保住 96.7% 的性能。

复现方式：

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> IDEA_RHO=0.5  IDEA_LAMBDA=0.5 NUM_GPUS=1 bash scripts/idea_eval.sh idea4 192 qbench_dev
CUDA_VISIBLE_DEVICES=<idle_gpu> IDEA_RHO=0.25 IDEA_LAMBDA=0.5 NUM_GPUS=1 bash scripts/idea_eval.sh idea4 128 qbench_dev
CUDA_VISIBLE_DEVICES=<idle_gpu> IDEA_RHO=0.25 IDEA_LAMBDA=0.5 NUM_GPUS=1 bash scripts/idea_eval.sh idea4 64  qbench_dev
```
