# Motivation 实验：Vision Token 的低频冗余性

对应论文 3.2 节 "Analyses of Frequency Components" 中与 (i) energy distribution
互补的一个分析：不只看频谱能量集中在低频，而是直接问一个更贴近"能不能压缩"的
问题——**如果只用一个 token 网格里最低频的一小部分 2D-DCT 系数去重建每个
token，重建结果和原始 token 有多像？** 并且这件事必须同时满足两个条件才能称之为
"vision token 经过编码后的自然现象"，而不是个别例子或个别模型的巧合：

1. 在**大量真实图片**（本次用 10,000 张）上都成立；
2. 在**架构上互不相关的视觉编码器**上都成立——本次同时采集了 LLaVA-1.5 的
   CLIP-L/14-336（固定 24×24 网格）和 Qwen2.5-VL 的动态分辨率 ViT（每张图网格
   形状都不同），而不是只测一个模型。

## 1. 实验设计

**两套特征来源**（都不跑完整的语言模型生成，只跑各自的视觉塔，省时间）：

- **LLaVA-1.5 / CLIP-L**：加载 `/home/jk/models/clip-vit-large-patch14-336`，取
  `hidden_states[mm_vision_select_layer]`（`llava-v1.5-7b/config.json` 中
  `mm_vision_select_layer=-2`，`mm_vision_select_feature="patch"`），丢弃 CLS
  token，得到 `[576, 1024]` 的固定 `24×24` 网格特征（projector 之前）。
- **Qwen2.5-VL**：加载 `/home/jk/models/qwen2.5-vl-7b`，直接调用
  `model.visual(pixel_values, grid_thw)`（`fourier_compressor/integrations/qwen_vl/`
  里实际压缩逻辑用的同一个前向接口），得到 `[half_h*half_w, 3584]` 的 patch
  merge 后特征。`half_h, half_w` 由 `image_grid_thw` 给出，**每张图不一样**
  （动态分辨率，`min_pixels=256*28*28`、`max_pixels=2304*28*28`，与
  `examples/infer_qwen2_5_vl.py` 默认值一致）。

**数据**：两个模型采集的是**同一批** 10,000 张图，来自
`lmms-lab___coco-caption2017`（`default` split 下的 `*-test-*.arrow`，14 个
分片、共 40,670 张不重复图片，足够无放回采样 10k），固定种子 `seed=7`。
（注：该数据集打包字段有点误导——真正的逐图唯一 COCO 文件名存在
`question_id` 字段里，`id`/`file_name` 两列在整个 split 里是常量，采集脚本里
写了注释说明。）

**核心指标**：对每张图的 `[H,W,C]` 特征网格做逐通道正交 2D-DCT 得到系数 `F`；
用一个**相对比例** `r`（而不是绝对的 `k`，因为 Qwen 每张图网格形状都不同）
只保留左上角 `round(r·H) × round(r·W)` 低频系数做 IDCT 重建，计算每个 token
的重建向量与原始向量的**余弦相似度**，再对所有 token 取平均。`r` 取
`1/24, 2/24, ..., 24/24` 共 24 个点——LLaVA 这边固定网格是 `24×24`，所以
`r=k/24` 就是直接的 `k`；Qwen 每张图网格不同，但用同一组 `r` 保证两条曲线的
横轴（"保留系数比例"）严格可比，不依赖各自的绝对网格大小。`r=12/24=0.5`
对应保留 25% 的系数面积，是本项目 Stage A 的实际目标压缩比（见
`docs/idea1.md` 的 `k1=12` 设定），在图上专门标出。

余弦相似度是比"能量占比"更严格的逐 token 指标：两个不相关的向量期望余弦
相似度 ≈ 0，可以直接作为"完全不冗余"的参照基线画在图上。

## 2. 图表形式的选择（折线图，而非柱状图/散点图）

- **柱状图**不合适：横轴（保留系数比例）是连续有序的，柱状图会把"早期快速
  上升、后期趋于饱和"这条最关键的趋势切成离散的块。
- **纯散点图**（固定一个比例、画很多点）能体现"很多图"，但丢了"随压缩比例
  变化"这条最能体现"低频冗余"的趋势线。
- **折线图**能把趋势和"多图/多模型都成立"同时说清楚。10,000 张图规模下不再
  适合像小样本（n=100）时那样每张图画一条细线（10,000 条线既不可读也没必要），
  改成论文里常见的**均值线 + ±1 标准差阴影带**：阴影带的宽度直接体现"10,000
  张图之间的差异有多大"，比只画均值更诚实。

## 3. 复现步骤

```bash
# 1) LLaVA-1.5 / CLIP-L，10k 图，需要能跑 CLIPVisionModel 的环境
CUDA_VISIBLE_DEVICES=5 /home/jk/miniconda3/envs/llava_visiPruner/bin/python \
    mov/collect_lowfreq_redundancy.py

# 2) Qwen2.5-VL，10k 图（同一批图），需要 transformers >= 4.49 的环境（clse_qwen）
CUDA_VISIBLE_DEVICES=5 /home/jk/miniconda3/envs/clse_qwen/bin/python \
    mov/collect_lowfreq_redundancy_qwen.py

# 3) 只读盘、出图（matplotlib 环境即可，不需要 GPU）
/home/jk/miniconda3/envs/llava_visiPruner/bin/python mov/plot_lowfreq_redundancy.py
```

- `mov/collect_lowfreq_redundancy.py` → `mov/data/lowfreq_redundancy_curves.npz`
  (`[10000,24]`) + `mov/data/summary.json`。
- `mov/collect_lowfreq_redundancy_qwen.py` → `mov/data/lowfreq_redundancy_curves_qwen.npz`
  (`[10000,24]`，另存每张图实际的 `(half_h, half_w)` 网格形状) + `mov/data/summary_qwen.json`。
- `mov/plot_lowfreq_redundancy.py` → `mov/figs/lowfreq_token_redundancy.png`。

两个采集脚本分别限定在各自能跑通的 conda 环境（`llava_visiPruner` 装了兼容
CLIP 的 transformers 版本；`clse_qwen` 是唯一装了 transformers ≥4.49、能跑
`Qwen2_5_VLForConditionalGeneration` 的环境），复现时按脚本里写的环境路径执行
即可，两次采集都限定物理卡 5。

## 4. 结果

在 Stage A 实际压缩比（保留 25% 系数，`r=0.5`）下，跨 10,000 张图：

| 模型 | 平均余弦保真度 | 标准差 | 10k 图范围 |
| - | - | - | - |
| CLIP-L / LLaVA-1.5（固定 24×24 网格） | 0.765 | 0.026 | [0.666, 0.874] |
| Qwen2.5-VL ViT（动态分辨率网格） | 0.780 | 0.021 | [0.684, 0.858] |

两个架构完全不同的视觉编码器——一个是固定分辨率、单尺度 CLIP-L，一个是
动态分辨率、带窗口注意力的 Qwen2.5-VL ViT——在同一批 10,000 张图上给出几乎
一致的结果（均值相差仅 0.015，标准差相近），且都远高于无关向量的 ≈0 基线。
两条曲线的形状也高度一致：只用 1/24 的系数（近似只保留 DC 分量）时保真度已
有 ~0.53-0.54，25% 系数时到 ~0.76-0.78，说明"少量低频系数解释大部分 token
内容"不是 CLIP/LLaVA 特有的现象，而是自然图像经过 patch 化视觉编码后的
共性——这正是本项目做频谱压缩这件事本身成立的前提。

## 5. 最终图

单张折线图：`mov/figs/lowfreq_token_redundancy.png`

- 横轴：保留的低频 DCT 系数比例（0-100%）。
- 纵轴：逐 token 余弦保真度（相对完整频谱 token 的余弦相似度）。
- **紫色 = Qwen2.5-VL ViT，蓝色 = CLIP-L / LLaVA-1.5**，均为 10,000 张图的
  均值线 + ±1 标准差阴影带。
- 灰色稀疏点线：无关向量的零基线。
- 黑色竖直点线 + 标注：25% 系数（本项目 Stage A 的目标压缩比）。
- 视觉风格参考了强调"训练曲线 + 置信区间阴影 + 参考线"的论文图风格：衬线
  字体、无网格、四边框+向内刻度、图例透明无边框。本机没有装 LaTeX
  (`latex`/`dvipng`)，因此没有启用 `text.usetex`，改用 matplotlib 内置的
  `mathtext.fontset="cm"` 搭配衬线字体做最接近的效果，视觉上足够接近但不是
  真正的 LaTeX 排版，如需完全一致需要在有 LaTeX 环境的机器上重新出图。

## 6. 论文文字草稿（可直接改写进 3.2 节）

> To test whether the frequency concentration observed in vision encoder
> outputs translates into genuine, per-token redundancy -- and whether it is
> an architecture-specific artifact or a general property of patch-based
> visual encoding -- we reconstruct each patch token from only its lowest
> low-frequency 2D-DCT coefficients and measure the per-token cosine fidelity
> to the full-spectrum token, across a sweep of retained-coefficient ratios,
> on 10,000 COCO images. We repeat this measurement on two architecturally
> unrelated vision encoders: LLaVA-1.5's fixed-resolution CLIP-L/14-336 tower
> and Qwen2.5-VL's dynamic-resolution ViT. At a 25% coefficient budget
> (matching our Stage A compression ratio), both encoders converge to a
> nearly identical mean cosine fidelity (CLIP-L: 0.765 +/- 0.026; Qwen2.5-VL:
> 0.780 +/- 0.021, both over n=10,000 images), far above the ~0 similarity
> expected between unrelated feature vectors. The consistency of both the
> absolute fidelity level and the saturating curve shape across two unrelated
> architectures and 10,000 images indicates that low-frequency redundancy in
> vision tokens is a general property of patch-based visual encoding, not an
> artifact of any single model or example.

## 7. 复现涉及的文件

一切都自包含在 `mov/` 目录下：

- `mov/collect_lowfreq_redundancy.py` — LLaVA-1.5/CLIP-L 数据采集（需要 GPU，限定物理卡 5，`llava_visiPruner` 环境）。
- `mov/collect_lowfreq_redundancy_qwen.py` — Qwen2.5-VL 数据采集（需要 GPU，限定物理卡 5，`clse_qwen` 环境）。
- `mov/plot_lowfreq_redundancy.py` — 画图（只读盘，不需要 GPU）。
- `mov/data/` — 原始数据（`lowfreq_redundancy_curves.npz`,
  `summary.json`, `lowfreq_redundancy_curves_qwen.npz`, `summary_qwen.json`）。
- `mov/figs/lowfreq_token_redundancy.png` — 最终图。
