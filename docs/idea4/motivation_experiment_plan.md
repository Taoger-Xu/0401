# SCOPE Motivation Experiments：投稿前执行方案

本文件是作者执行清单，不直接进入论文。目标是让 SCOPE 的每个阶段都由独立观察推出，而不是仅凭最终
benchmark 分数解释算法。

## 0. 总体原则

- 所有选择方法使用同一层视觉特征、同一 token 预算、同一图像集合。
- 机制实验优先使用不依赖任务标签的 GQA/COCO 图像子集，避免用下游答案反向定义“重要性”。
- 主预算至少包含 $K=192$（实用点）和 $K=64$（强剪枝）；集合诊断再补 $K=128$。
- 报告图像级均值与 bootstrap 95% CI；方法对比使用配对检验。
- 预先固定 DCT lowpass、cell 数和权重；不得看完测试结果后重新定义机制指标。
- “支持动机”与“提升下游准确率”分开报告。相关性只能支持解释，组件消融才支持因果增益。

## M0. 局部可替代性是否存在且分布不均匀？

这一步必须与频率无关，否则无法排除“用频率定义冗余、再用频率解释频率”的循环。

### 数据与特征

- GQA val 与 COCO Caption val 各随机 1,000 图，与 M1 使用同一批图像以便配对分析。
- LLaVA-1.5 CLIP 倒数第二层 patch feature，保留 $24\times24$ 网格。

### 指标

1. `local_cosine`：token 与八邻域的平均 cosine similarity（不涉及 DCT）。
2. `neighbor_replacement_error`：删去 token 后用邻域 medoid/均值替代产生的 cosine/$\ell_2$ 误差。
3. 区域类型分组：背景、物体内部、物体边界、文字/细粒度。可用 COCO panoptic 标注或 SAM/边缘检测得到
   分组掩码，只用于分析，不进入方法。

### 必须绘制

- `local_cosine` 与替代误差的全局分布；四类区域的分组箱线图。
- 单张图像上的替代误差热图，与原图并列。

### 判定标准

- 若 `local_cosine` 整体偏高，可支持“高压缩率可行”。
- 若四类区域之间差异显著（配对检验），可支持“需按区域而非全局统一处理”，并同时为文字任务的能力边界
  提供解释。
- 若冗余分布接近均匀，则空间 cell 设计缺乏依据，应重新考虑 Stage I。

## M1. 低频结构是否刻画局部可替代性？

### 数据与特征

- GQA val 随机 1,000 图；COCO Caption val 随机 1,000 图。
- 使用 LLaVA-1.5 CLIP 倒数第二层 patch feature，保留原始 $24\times24$ 网格。
- 复用 `lowfreq_reconstruct` 计算 $L\in\{8,12,16,20\}$ 的高频残差，主结果固定 $L=16$。

### 指标

1. `local_cosine`：token 与八邻域的平均 cosine similarity。
2. `centroid_cosine`：token 与所在 FPS–Voronoi cell centroid 的 cosine similarity。
3. `replacement_error`：用邻域 medoid 替代 token 后的 feature cosine/$\ell_2$ 误差。
4. `representative_stability`：轻微 color jitter、resize、horizontal flip 后，对齐坐标的 cell Top-1 一致率。
5. Spearman correlation：高频残差分别与上述四项计算相关性及 CI。

### 必须绘制

- 高频残差直方图与累计分布。
- 高频残差四分位对应的四项指标柱状图。
- $h_i$–`local_cosine` 与 $h_i$–`replacement_error` 的 hexbin/scatter。
- 原图、低通 PCA feature、残差热图四列定性图。

### 判定标准

- 若低残差组显著具有更高邻域相似、更低替代误差与更高稳定率，可使用“frequency-stable local
  representative”。
- 若只有谱能量集中而与替代误差无稳定关系，则低频只能作为定性观察，不能进入标题或主贡献。

## M1b. 低频项相对既有代表准则是否有独立增益？

M1 只给出相关性，不能说明 $w_f$ 是否必要。本实验在 encoder 特征层直接比较代表选择准则，不经过 LLM，
因此不受下游噪声干扰，是 $w_f$ 的核心证据（M3 的 `w_f=0` 下游消融是补充而非替代）。

### 设置

- 固定同一组 FPS–Voronoi cell（`cover_factor=3`，与主实验一致）与同一 $M$，只改变 cell 内代表准则。
- 准则：Random / Attention-max / Medoid（仅 $m_i$）/ Medoid + 低频（$m_i+w_f\ell_i$）/ 完整 Stage-I 打分。
- 图像集合与 M0、M1 相同，支持逐 cell 配对检验。

### 指标

1. `cell_reconstruction_error`：用代表替代该 cell 全部 token 的平均 cosine/$\ell_2$ 误差。
2. `representative_agreement`：color jitter / resize / flip 后对齐坐标的代表一致率。
3. `rep_cell_similarity`：代表与 cell 内其余 token 的平均 cosine similarity。
4. 按 M0 的区域类型分组重复上述指标，检验低频项是否主要在平滑区域生效。

### 判定标准

- Medoid + 低频相对纯 Medoid 在重建误差与一致率上均改善 → 低频项同时提升代表性与稳定性，可写入标题。
- 仅一致率改善、重建误差持平 → 低频项的作用是**稳定性**，正文措辞必须相应收敛，标题避免暗示代表性增益。
- 两项均无改善 → 不把 spectral 写入标题与主贡献，方法退化为 $w_f=0$ 形式，重写故事主线。

## M2. CLS Top-K 是否产生空间塌缩？

### 对照方法

- Random、Uniform Grid、CLS Top-$K$、Global-MMR、Stage-I Anchors、SCOPE。
- $K\in\{64,128,192\}$，每种方法在相同 1,000 张 GQA 图像上运行。

### 指标实现

1. `attention_capture = sum(a[selected]) / sum(topK(a))`。
2. `saliency_weighted_coverage`：将网格划为 $6\times6$ macro cells，以 cell max attention 为权重，统计
   被选中 cell 的加权覆盖。
3. `pairwise_spatial_distance`：选中 token 两两欧氏距离除以网格对角线。
4. `neighbor_cluster_ratio`：至少有一个八邻域 token 同时被选中的 token 比例。
5. `feature_nn_similarity`：每个选中 token 与其余已选 token 的最大 cosine similarity 的均值。
6. `connected_components`：八邻接选择 mask 的连通分量数与最大分量占比。

### 必须绘制

- 同一图像六种方法的 token overlay，至少覆盖单主体、多主体、小目标、文字、背景占比高五类场景。
- 横轴 attention capture、纵轴 saliency-weighted coverage 的 Pareto 图；点大小表示 feature redundancy。
- 每个 $K$ 下五项机制指标的雷达图或紧凑表格。

### 判定标准

- CLS Top-$K$ 应表现为 attention capture 高，但覆盖/空间距离低、邻域聚集与特征相似度高。
- Global-MMR 若只改善 feature similarity、未同步改善 content-weighted coverage，才能支持显式锚点的必要性。
- Uniform Grid 若覆盖高但 attention capture 低，才能支持 saliency-ranked cells 而非无条件均匀覆盖。

## M3. 两阶段组件是否具有独立贡献？

### 变体矩阵

1. CLS Top-$K$。
2. Uniform Spatial Medoid：空间 cell + medoid，无频率、无显著性、无 MMR。
3. Uniform Spectral：空间 cell + medoid + low-frequency，无显著性、无 MMR。
4. Semantic Anchors w/o Spectral：Stage I 设置 $w_f=0$。
5. Spectral Anchors w/o Medoid：移除 $m_i$。
6. Spectral Anchors w/o Local Attention：设置 $w_a=0$，保留 cell-level saliency ranking。
7. Stage-I Anchors only：$\rho=1$。
8. Global-MMR：$\rho=0$。
9. SCOPE：完整方法。

### 下游任务

- 主任务：GQA、POPE、MME-P、MMBench。
- 覆盖敏感：COCO Caption CIDEr、MMStar coarse/fine-grained perception。
- 边界任务：TextVQA、OCRBench。
- 先在 $K=192$ 跑全矩阵；只有关键变体再补 $K=64$，控制实验成本。

### 报告方式

- 同表报告 attention capture、weighted coverage、feature NN similarity 与下游平均保留率。
- 对 `w_f=0` vs full SCOPE 做逐样本 paired bootstrap，作为低频项独立收益的核心证据。
- 对 $\rho=0$ vs full SCOPE 隔离空间锚点收益；对 $\rho=1$ vs full SCOPE 隔离 Stage II 收益。
- 不允许只报告最优 benchmark；同时列出覆盖敏感任务和文字任务的相反趋势。

## M4. LLM Attention 的位置偏差与 FlashAttention 兼容性

### 位置干预

- 固定图像和 prompt，只改变视觉 token 顺序：raster、reverse-raster、固定随机排列、空间蛇形排列。
- 在 eager attention 下收集若干早/中/晚层 cross-modal attention score。
- 恢复原空间索引后计算：不同排列间 Pearson/Spearman、JS divergence、Top-$K$ Jaccard，以及 attention 与
  绝对序列位置的相关性。
- 至少 500 张图像，每图使用相同随机 permutation seed。

### 系统对照

- 路径 A：eager + `output_attentions=True` + LLM-side pruning。
- 路径 B：SDPA/可返回 attention 的替代实现。
- 路径 C：FlashAttention + SCOPE（encoder-side，不请求 LLM attention map）。
- 固定 batch size、输入长度、输出长度和 GPU；预热 20 次、测量至少 100 次。
- 报告 pruning overhead、prefill latency、end-to-end latency、tokens/s、peak allocated/reserved memory。

### 判定标准

- 若恢复空间索引后 attention 排名随排列显著变化，可支持 positional bias 论断。
- 若获取显式 attention map 必须关闭/绕过 FlashAttention且产生显著系统代价，可支持 incompatibility 论断。
- 若框架能够在 FlashAttention 下无额外代价返回所需 score，则摘要必须弱化“不兼容”表述。

## M5. 建议的论文图表顺序

图表顺序应复现叙事链条：冗余存在 → 谁当代表 → 现有准则为何失败 → 方法。

1. **Figure 1：核心动机图。** 左：替代误差热图（冗余存在且不均匀，M0）；中：CLS Top-$K$ 空间聚集（M2）；
   右：SCOPE 两阶段。
2. **Figure 2：冗余诊断。** M0 的分布图与四类区域分组箱线图。
3. **Figure 3：频率诊断。** 残差分布、$h_i$ 与替代误差的相关性、定性热图，以及 M1b 的准则对比柱状图。
4. **Figure 4：选择集合 Pareto。** attention capture vs content-weighted coverage，颜色表示 feature redundancy。
5. **Table 1：代表选择准则对比。** M1b 指标表。
6. **Table 2：Motivation/Mechanism。** M2 指标表。
7. **Table 3：组件消融。** M3 机制指标 + 下游结果。
8. **Table 4：主 benchmark。** 固定通用配置。
9. **Table 5：效率。** FLOPs、prefill、端到端 latency、显存、选择器 overhead。

## M6. 投稿前最低完成线

P0（没有则核心故事不成立）：M0 冗余存在性、M1 相关性、**M1b 代表准则对比**、M2 集合诊断、
`w_f=0` 消融、$\rho=0$ 消融。

P1（标题强调 efficient/FlashAttention 时必须）：M4 系统对照、端到端 latency、显存、选择器耗时。

P2（增强完整性）：频率 cutoff 敏感性、Qwen 上复现 M1/M2、更多外部 pruning/merging baseline。
