# SCOPE: Spectral-Semantic Anchoring with Global Redundancy-Aware Visual Token Pruning for Efficient Multimodal LLM Inference

> **完整论文草稿**（摘要 / 引言 / 相关工作 / 动机分析 / 方法 / 实验 / 跨架构泛化 / 讨论 / 结论）。
> 正文统一使用 **SCOPE**。标有“待补”的表格是投稿前实验协议与占位符，不包含虚构结果。

---

## 摘要

Reducing visual token redundancy is critical for accelerating Multimodal Large Language Models (MLLMs) without
degrading performance. Visual tokens are highly compressible primarily because most of them are locally
substitutable: patches are cut on a fixed grid regardless of object boundaries, so neighboring tokens within a
homogeneous region largely encode the same evidence. Crucially, this substitutability is unevenly distributed, which
turns pruning into two distinct questions — which regions can be summarized by a single token, and which token
qualifies as that representative. Existing criteria answer neither. [CLS]-based attention ranks tokens independently
of the selected set, so it captures saliency but collapses spatially onto a few dominant regions; uniform spatial
sampling spends budget on background; and global redundancy criteria remove feature duplication without guaranteeing
spatial support. LLM-side attention can additionally be position-sensitive, while extracting explicit attention maps
may require forgoing efficient implementations such as FlashAttention. To answer the second question, we analyze the
spectral characteristics of visual tokens and use low-frequency dominance as a proxy for whether a token captures
the slowly varying visual component shared by its neighborhood. Building on this motivation, we propose SCOPE, a
training-free, two-stage visual
token pruning framework that can be seamlessly integrated into existing MLLMs. In the first stage, Spectral-Semantic
Anchoring (SSA) selects spectrally stable anchors within over-segmented spatial cells, and
then allocates the anchor budget according to cell saliency. In the second stage, Global Redundancy-Aware Visual
Token Pruning (GRVP)
applies Maximal Marginal Relevance (MMR), initialized with these anchors, to select the remaining tokens by balancing
visual saliency and feature diversity. The two stages jointly preserve visual saliency and spatial coverage
while operating entirely before the LLM, without relying on positionally biased cross-modal attention maps.
Extensive experiments across diverse
multimodal benchmarks and multiple MLLMs demonstrate that SCOPE achieves a favorable accuracy–efficiency trade-off.
On LLaVA-1.5-7B, SCOPE prunes 88.9% of the visual tokens and reduces the visual-sequence LLM FLOPs by 89.2% under
our accounting, while retaining 95.1% of the baseline performance on average across nine benchmarks.

---

## 1. 引言

Despite their effectiveness, existing visual token pruning paradigms exhibit two stage-specific limitations arising
from their reliance on attention as a proxy for token saliency. In the vision encoder, [CLS]-to-patch attention
typically concentrates the retained tokens within a few salient regions, without accounting for the substantial
redundancy among spatially adjacent tokens (see Fig. 1a). Consequently, the selected subset may contain repeatedly
encoded evidence from the same object while omitting less salient but distinct regions. In the language model,
cross-modal attention does not necessarily reflect each visual token's contribution to the generated response and
can exhibit a pronounced positional bias in shallow layers (see Fig. 1b). Moreover, extracting explicit attention
maps may require disabling or bypassing efficient implementations such as FlashAttention. These limitations suggest
that effective pruning should consider not only whether a token is salient, but also whether its information is
locally irreplaceable.

This distinction is essential because visual tokens can be discarded at high ratios not simply because most of them
have low saliency, but because many can be substituted by neighboring tokens with little reconstruction error. Several
adjacent tokens may all receive high attention while encoding nearly identical evidence, whereas a moderately
attended token may provide the only evidence for a small or distant object. Inspired by traditional image compression,
where frequency-domain transforms expose spatial redundancy and enable efficient compression, we systematically
investigate whether analogous frequency redundancy exists in visual token representations. We apply a two-dimensional
DCT to the patch representations of multiple images and average the coefficient magnitudes over hidden dimensions.
As shown in Figure 2(a), the resulting spectrum is strongly concentrated at low spatial frequencies. This observation
is complemented by Figure 2(b), where randomly removing tokens within uniformly partitioned spatial cells causes only
limited performance degradation over a broad pruning range. Together, these results show that visual tokens contain
substantial local redundancy and motivate the hypothesis that high-frequency residual can distinguish tokens dominated
by neighborhood-shared components from those dominated by position-sensitive details.
Importantly, frequency structure is intended to measure local substitutability rather than visual saliency: a
spectrally stable token may correspond either to an informative object or to background.
Effective pruning must therefore jointly determine which tokens can represent their neighborhoods, which content
regions should receive spatial support, and which additional tokens contribute evidence not already selected.

To address these limitations, we propose SCOPE, a training-free two-stage visual token pruning method (illustrated in
Figure 5) that identifies essential visual tokens entirely within the vision encoder, before they enter the language
model. In the first stage, we use low-frequency dominance as a proxy for whether a token can stably represent its
local neighborhood, while explicitly separating this property from visual saliency. Based on this design, we
introduce Spectral-Semantic Anchoring
(SSA), which partitions the visual tokens into spatial cells, selects within each cell the token whose representation
is most dominated by the neighborhood-shared low-frequency component, and then allocates the anchor budget according
to cell saliency. This
design preserves spatial coverage of informative regions while avoiding the uniform allocation of tokens to large
background areas. In the second stage, we note that these anchors establish only a content-aware lower bound on the
spatial support of the retained evidence, so the remaining budget should be spent on evidence that
is not yet expressed. We therefore introduce Global Redundancy-Aware Visual Token Pruning (GRVP), an MMR procedure
initialized with the SSA anchors, which selects the remaining tokens by balancing visual saliency and feature diversity,
so that redundancy is
consistently measured against the already retained evidence rather than estimated independently.

本文的主要贡献如下：

1. 我们把视觉 token 剪枝重新表述为**局部可替代性判断**而非 token 显著性排序，并用一组不依赖下游标签的机制分析
   说明冗余在何处、锚点应如何选取，将频谱稳定性限定为 cell 内稳定锚点选择的判据而非 token 显著性。
2. 我们提出训练无关、零参数的 SCOPE：Stage I 的 Spectral-Semantic Anchoring（SSA）建立内容感知的
   空间证据下界，Stage II 的 Global Redundancy-Aware Visual Token Pruning（GRVP）在 SSA 锚点条件下执行
   MMR；方法仅 gather 原始 token，且不引入可学习参数。
3. 我们在固定视觉 token 预算下系统比较 token 显著性排序、均匀覆盖、Global-MMR 与 SCOPE，并在 LLaVA-1.5-7B 的
   九项基准以及 COCO Caption、MMStar 能力分解上报告性能—计算权衡和强剪枝边界。
4. 我们在无 CLS、动态视觉网格的 Qwen2.5-VL-7B 上验证可迁移性，并分析密集文字这一低冗余内容对纯选择
   方法构成的挑战。

### 1.1 评测协议

TextVQA 在不同版本评测协议中是否提供参考 OCR token，会导致不可忽略的分数差异。为保证公平性，本文的
Baseline（vanilla）、VisionZip 与 SCOPE 均在不提供参考 OCR token 的同一协议下重新评测；不同协议的结果不做
交叉比较。其余实现核验与复现细节见附录。

---

## 2. 相关工作

### 2.1 视觉 token 压缩

MLLM 视觉 token 压缩大体可分为选择、合并和学习式压缩。学习式方法能够针对下游目标优化保留内容，但需要
额外训练数据与计算；合并方法将多个 patch 聚合为较少的上下文 token，信息保留较充分，却可能改变特征
分布并引入额外聚合误差；选择方法直接保留原始 token，部署最简单，但要求选择准则准确判断哪些证据可被
丢弃。本文聚焦最后一种设定，并将 VisionZip 作为同一插入位置上的代表性训练无关基线。与其“显著 token
选择 + 剩余 token 合并”不同，SCOPE 全程只选择原始 token，因此方法收益能够直接归因于选择策略。

### 2.2 显著性与冗余感知选择

注意力或 CLS-to-patch 响应为视觉 token 的显著性提供了便捷代理。本文将仅按 $a_i$ 选择最高分 token 的
方法统一称为 **Saliency Top-$K$**；在 LLaVA-1.5 中，它具体等价于 CLS-to-patch attention Top-$K$，
也是 VisionZip dominant-token 选择的纯剪枝对应物。然而，token 显著性排序不考虑已选集合，容易重复保留
同一主体附近的相似 token。MMR 类准则通过惩罚候选与已选集合的
最大特征相似度，能够在显著性和非重复性之间折中。本文将不使用锚点、直接在全部 patch token 上运行该
准则的变体称为 **Global-MMR**，并将其作为无显式空间约束的基线。SCOPE 的 GRVP 虽然采用同一 MMR
形式，但以 SSA 锚点集合初始化；因此，Global-MMR 是 $\rho=0$ 时的退化形式，不是 GRVP 的同义名称。

### 2.3 空间覆盖与频率结构

均匀采样、网格池化或基于聚类的代表选择能够改善空间分布，但把每个位置等量纳入预算会过度保留大面积背景，
且大多只回答“在哪里取代表”，不回答“取哪一个”。我们从视觉特征的空间频率结构回答后一个问题：相邻
patch 可共享的视觉成分通常随空间位置缓慢变化，因而集中在低频部分；边界、纹理与局部噪声等位置敏感细节
则更多表现为高频变化。由此，高频残差较小的 token 更可能由邻域共享成分解释，并在轻微扰动下保持稳定。
本文将该性质统一称为
**频谱稳定性**，并且不把它当作 token 显著性；它只用于在空间 cell 内挑选稳定锚点，而该 cell 是否值得占用
预算仍由 cell 显著性决定。这一职责分离避免了
两个常见极端：只追逐高注意力导致空间塌缩，或只追求均匀覆盖而浪费背景预算。

---

## 3. 动机分析：视觉编码器输出是否存在可利用的低频冗余？

SCOPE 的设计源于一个直接的频域观察：尽管视觉编码器在空间网格上输出大量 patch token，其表示并未在各个
空间频率上均匀分布，而是明显集中于低频区域。本节首先通过多图像频谱统计建立这一观察，再讨论它如何导出
token 级频谱稳定性，以及为何频谱稳定性仍需与内容显著性和集合级去冗余配合。该分析遵循“观察—假设—
验证—设计”的顺序，而不使用最终下游准确率反向论证方法合理性。除 Figure 2(a)–(b) 中的初始冗余分析外，标记为
“待补”的表格均为投稿前实验协议，正式稿中只保留实测结果。

### 3.1 视觉编码器输出呈现低频能量集中

我们首先考察视觉编码器在任何 token 选择发生之前是否已经产生显著冗余。给定图像 $x^{(n)}$，将
Qwen2-VL [Wang et al., 2024] 视觉编码器输出的 patch 表征按原始空间位置重排为
$g^{(n)}\in\mathbb R^{G_h\times G_w\times D}$。我们沿两个空间维度对每个隐藏通道独立施加二维离散余弦
变换（DCT），得到

$$
z^{(n)}_{:,:,d}=\mathbf W_h g^{(n)}_{:,:,d}\mathbf W_w^\top,
\qquad d=1,\ldots,D,
$$

其中 $\mathbf W_h$ 和 $\mathbf W_w$ 分别为两个空间维度上的正交 DCT-II 基矩阵。为聚合不同隐藏维度上的
频率响应，我们对全部隐藏维度的频域系数绝对值取平均，将图像 $x^{(n)}$ 在频率位置 $(u,v)$ 上的响应强度
（下文简称频谱能量）定义为

$$
E^{(n)}_{u,v}=\frac{1}{D}\sum_{d=1}^{D}\left|z^{(n)}_{u,v,d}\right|.
$$

进一步在多张图像上求平均，得到数据集级频谱

$$
\bar E_{u,v}=\frac{1}{N_{\mathrm{img}}}\sum_{n=1}^{N_{\mathrm{img}}}E^{(n)}_{u,v}.
$$

如 Figure 2(a) 所示，跨图像平均后的频谱能量高度集中在左上角，并随空间频率升高快速衰减。这表明视觉
编码器输出主要由空间上缓慢变化的成分构成：尽管编码器产生了稠密的 patch token，相邻 token 仍共享大量
低频视觉信息。换言之，视觉 token 在进入语言模型之前已经表现出显著的统计低频冗余。

频谱集中描述的是表示层的统计结构，但它本身尚不能说明这种冗余是否允许实际删除 token。为此，我们进一步
进行局部随机删除实验。具体而言，将视觉 token 网格均匀划分为一组互不重叠的空间 cell，并在每个 cell 内
随机删除相同比例的 token。该设置使删除位置分散在整个图像上，避免结果由少数高注意力区域或特定选择器
主导。随后逐步提高删除比例，并在多个图像—语言理解基准上评估模型性能。

Figure 2(b) 显示，即使从每个局部 cell 中随机删除相当比例的视觉 token，同一 Qwen2-VL 模型在不同基准
上的性能仍能保持在较高水平，并且在中等删除比例下仅出现有限下降。该结果从功能层面印证了 Figure 2(a)
的频域观察：相邻
patch 不仅在频谱上共享低频成分，其中相当一部分 token 在下游推理中也是局部可替代的。因此，视觉编码器
输出的冗余并非仅表现为系数层面的低频集中，而是能够转化为实际的 token 压缩空间。

综合 Figure 2(a) 与 Figure 2(b)，我们得到第一个核心观察：**少量视觉 token 即可承载图像中的大部分有效
信息，而大量初始 token 主要重复编码局部共享的低频成分。** 然而，随机删除只能证明冗余存在，并不能判断
哪些 token 能够被安全删除。低频冗余在不同空间位置和内容类型上的分布也并不均匀；文字、物体边界和
细粒度结构通常包含更多高频变化。下一节因此从数据集级的低频冗余进一步转向 token 级频谱稳定性，以识别
哪些 token 更适合作为局部锚点。

### 3.2 从低频冗余到 token 级频谱稳定性

Figure 2(a) 描述了多图像平均意义下的整体频谱，却不能直接用于 token 选择。要把数据集级的低频冗余转化
为 cell 内锚点准则，需要估计每个 token 对低频共享成分的符合程度。为此，我们保留
$L_{\mathrm{lp}}\times L_{\mathrm{lp}}$ 低频系数并执行逆 DCT，得到低频重建特征 $\tilde f_i$，进而定义
token $i$ 的高频残差

$$
h_i=\lVert f_i-\tilde f_i\rVert_2 .
$$

高频残差越小，说明该 token 越能由空间上缓慢变化的共享成分解释；反之，较大的残差意味着其包含更多边界、
纹理或局部噪声等位置敏感信息。我们据此将 $\ell_i=-h_i$ 定义为 token 的**频谱稳定性**。若频谱稳定性
能够刻画局部可替代性，则它应与独立的空间域冗余指标保持一致。具体而言，用八邻域 $\mathcal N(i)$ 定义

$$
R_i^{\mathrm{loc}}=\frac{1}{|\mathcal N(i)|}\sum_{j\in\mathcal N(i)}
\cos(\hat f_i,\hat f_j),
$$

并以邻域均值或 medoid 替代 $f_i$ 后产生的 cosine/$\ell_2$ 误差定义邻域替代误差。若上述假设成立，
$h_i$ 应与 $R_i^{\mathrm{loc}}$ 显著负相关、与邻域替代误差正相关，
且低残差 token 在轻微输入扰动后应保持更稳定的 cell 内选择结果。前两项检验高频残差是否对应邻域共享
程度，后一项直接检验该指标能否识别稳定锚点；三者共同构成从频率结构到锚点资格的证据链。

**待补实验 M1：频率—可替代性相关性。** 从 GQA 与 COCO Caption 各采样至少 1,000 张图像，按 $h_i$
四分位分组并报告以下指标；所有相关系数同时给出 bootstrap 95% 置信区间。

| 高频残差分组 | 局部余弦 $R^{loc}$ ↑ | 邻域替代误差 ↓ | 扰动后锚点 Top-1 稳定率 ↑ |
|---|---:|---:|---:|
| Q1（最低残差） | 待补 | 待补 | 待补 |
| Q2 | 待补 | 待补 | 待补 |
| Q3 | 待补 | 待补 | 待补 |
| Q4（最高残差） | 待补 | 待补 | 待补 |
| Spearman $r_s(h,\cdot)$ | 待补 | 待补 | 待补 |

扰动稳定性使用不改变语义的轻微颜色抖动、resize 与水平翻转，并在对齐空间坐标后比较 cell 内锚点。

**待补实验 M1b：锚点选择准则的直接对比。** M1 只给出相关性，尚不能证明频谱稳定性能够改善实际选择。
固定同一组 FPS–Voronoi cell 与同一预算，仅改变 cell 内锚点的选取准则，在 encoder 特征层直接评估锚点
质量，不经过 LLM，因而不受下游噪声干扰：

| cell 内锚点准则 | cell 重建误差 ↓ | 扰动后锚点一致率 ↑ | 锚点—邻域平均相似度 ↑ |
|---|---:|---:|---:|
| Random | 待补 | 待补 | 待补 |
| 显著性最大（LLaVA 中为 cell 内最高 CLS attention） | 待补 | 待补 | 待补 |
| 仅频谱稳定性（$\ell_i$） | 待补 | 待补 | 待补 |
| 频谱稳定性 + cell 内显著性（$w_f\ell_i+w_a a_i$） | 待补 | 待补 | 待补 |
| 完整 SSA 实现（含辅助原型对齐项 $m_i$） | 待补 | 待补 | 待补 |

其中“cell 重建误差”定义为用选中锚点替代该 cell 全部 token 后的平均 cosine/$\ell_2$ 误差。核心比较是
仅频谱稳定性相对于 Random 与显著性最大是否同时降低重建误差并提高扰动一致率；M3 中的 $w_f=0$ 下游消融
则检验该信号对最终任务的贡献。完整实现额外包含 §4.3 披露的辅助原型对齐项 $m_i$，其作用单独报告，但不作为
频谱稳定性动机成立的前提。若频谱稳定性在机制指标与下游结果上均无改善，则不应将 spectral 写入标题与
主贡献，方法应退化为 §4.6 中 $w_f=0$ 的形式。

### 3.3 为什么 token 显著性排序仍然冗余？

在 LLaVA-1.5 中，CLS-to-patch attention 是 token 显著性 $a_i$ 的具体实现。它衡量单个 token 的显著性，
却不感知当前已选集合。我们检验其是否导致两种可区分的冗余：特征重复和空间聚集。将 $24\times24$ 网格
划分为 $6\times6$ 宏观区域，定义：

$$
\mathrm{Cov}_{\mathrm{sal}}(S)=
\frac{\sum_C q_C\,\mathbf 1[S\cap C\neq\varnothing]}{\sum_Cq_C},
\qquad q_C=\max_{i\in C}a_i,
$$

作为内容加权空间覆盖率；同时报告选中 token 的归一化两两空间距离、八邻域聚集率、最近邻特征相似度和
token 显著性质量保留率（在 LLaVA-1.5 中即 CLS attention mass 保留率）。内容加权覆盖避免把“均匀采到
背景”误认为有效空间多样性。

**待补实验 M2：选择集合诊断。** 在相同 $K\in\{64,128,192\}$ 下比较 Random、Uniform Grid、
Saliency Top-$K$、Global-MMR、仅 SSA 和完整 SCOPE。

| 方法 | 显著性质量保留 ↑ | 内容加权覆盖 ↑ | 两两空间距离 ↑ | 八邻域聚集率 ↓ | 最近邻特征相似度 ↓ |
|---|---:|---:|---:|---:|---:|
| Random | 待补 | 待补 | 待补 | 待补 | 待补 |
| Uniform Grid | 待补 | 待补 | 待补 | 待补 | 待补 |
| Saliency Top-$K$ | 待补 | 待补 | 待补 | 待补 | 待补 |
| Global-MMR | 待补 | 待补 | 待补 | 待补 | 待补 |
| SSA | 待补 | 待补 | 待补 | 待补 | 待补 |
| SCOPE | 待补 | 待补 | 待补 | 待补 | 待补 |

除聚合表外，应展示同一组图像上 Saliency Top-$K$、Global-MMR 与 SCOPE 的 token overlay。若 Saliency
Top-$K$ 确实捕获最高的 token 显著性质量、但空间覆盖低且邻域聚集率高，就能直接证明“显著性不等于
非冗余”；若
Global-MMR 降低特征相似度却仍落后于 SCOPE 的内容加权覆盖，则说明无锚点的全局选择不能替代 SSA。

### 3.4 两个阶段是否解决互补问题？

完整方法只有在每个组成信号解决不同失败模式时才不是模块堆叠。为此，需要把 cell 内锚点质量、cell 间
空间分配和全局补充策略分别隔离。所有变体使用相同注入点、token 数和评测协议。

**待补实验 M3：因果消融。** 主表至少在 GQA、POPE、MME-P、MMBench、COCO Caption 上报告 $K=192$
和 $K=64$；机制指标与下游结果必须并列。

| 变体 | 频谱稳定性 | cell 显著性排序 | SSA 锚点 | MMR 准则 | 内容加权覆盖 ↑ | 特征冗余 ↓ | 显著性质量保留 ↑ | 下游均值 ↑ |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|
| Saliency Top-$K$ |  |  |  |  | 待补 | 待补 | 待补 | 待补 |
| SSA w/o Cell Saliency Ranking（$c=1$） | ✓ |  | ✓ |  | 待补 | 待补 | 待补 | 待补 |
| SSA w/o Spectral Stability（$w_f=0$） |  | ✓ | ✓ |  | 待补 | 待补 | 待补 | 待补 |
| SSA | ✓ | ✓ | ✓ |  | 待补 | 待补 | 待补 | 待补 |
| Global-MMR（$\rho=0$） |  |  |  | ✓ | 待补 | 待补 | 待补 | 待补 |
| SCOPE | ✓ | ✓ | ✓ | ✓ | 待补 | 待补 | 待补 | 待补 |

该实验应支持一条逐步推理链，而不是要求所有指标同时最优：不使用 cell 显著性排序的 SSA 主要提高空间
分散性，但可能损失显著性质量；加入 cell 显著性排序应减少背景预算；Global-MMR 主要降低全局特征冗余；
完整 SCOPE 应在显著性质量保留、内容覆盖与特征非重复之间取得最好的联合权衡。还应单独加入 $w_a=0$，
检验 cell 内显著性与频谱稳定性的职责差异；去掉辅助原型对齐项 $m_i$ 的结果作为实现完整性消融另行报告，
但不将其提升为与频谱稳定性并列的设计假设。

### 3.5 为什么在进入 LLM 前完成剪枝？

本文关于 LLM-side attention 的批评需要实证，而不能只依赖常识。位置偏差应通过 token 顺序干预检验：对
同一图像分别使用 raster、reverse-raster 和固定随机排列，记录若干 LLM 层的 image-to-text attention，
计算 attention score 与绝对位置的 Spearman 相关、不同排列间恢复到原空间后的 Jensen–Shannon divergence，
以及 top-$K$ 集合重合率。一个理想的 token 显著性分数应在恢复空间索引后对排列相对稳定。

**待补实验 M4：位置偏差与系统兼容性。** 在相同模型、prompt、输出长度和 GPU 上比较以下路径；预热后
重复至少 100 次，报告均值与标准差。

| 剪枝信号/实现 | FlashAttention | 显式 Attention Map | Top-$K$ 排列一致性 ↑ | Prefill Latency ↓ | Peak Memory ↓ |
|---|:---:|:---:|---:|---:|---:|
| LLM cross-modal attention / eager | ✗ | ✓ | 待补 | 待补 | 待补 |
| LLM cross-modal attention / SDPA | 视实现而定 | ✓ | 待补 | 待补 | 待补 |
| SCOPE / FlashAttention | ✓ | ✗ | 不依赖排列 | 待补 | 待补 |

这一实验要区分两个结论：排列实验验证 attention score 是否具有位置敏感性；系统实验验证为了取得显式
attention map 是否必须关闭或绕过 FlashAttention，以及由此产生的延迟和显存代价。只有两者都有证据，
摘要中的相关论断才足够稳健。

### 3.6 从动机到设计

上述频谱观察与机制诊断对应 SCOPE 的各项设计决策，其推导顺序如下：

| 动机结论 | 对应设计决策 | 实现位置 |
|---|---|---|
| 视觉编码器输出具有显著的低频能量集中（Figure 2(a)） | 用二维 DCT 显式分离低频共享成分与高频残差 | §4.3 频谱分解 |
| 均匀 cell 内随机删除大量 token 后性能仍较稳定（Figure 2(b)） | 以空间 cell 组织局部冗余，并在每个候选区域内选择锚点 | §4.3 空间细分 |
| 高频残差可预测局部可替代性与锚点稳定性（M1/M1b） | SSA 以频谱稳定性 $\ell_i$ 选择 cell 内锚点 | §4.3 频谱稳定性 |
| 均匀覆盖把预算浪费在背景（M2） | 过分割出 $P>M$ 个 cell，再按 cell 显著性 $q(C)=\max_{i\in C}a_i$ 只取 top-$M$ | §4.3 cell 显著性 |
| Saliency Top-$K$ 不感知已选集合、空间塌缩（M2） | 用 SSA 锚点建立内容感知的空间证据下界 | §4.3 → §4.4 初始集合 |
| 特征互异不蕴含空间完整，Global-MMR 无覆盖保证（M2/M3） | 以 SSA 锚点初始化 MMR，而非独立运行两套选择 | §4.4 |
| LLM attention 存在位置偏差且与 FlashAttention 冲突（M4） | 选择完全发生在 encoder 侧、LLM 之前 | §4.1 注入点 |

全文用三类核心信号描述不同选择依据：频谱稳定性 $\ell_i$ 判断 token 能否稳定承载邻域共享成分；token
显著性 $a_i$ 及其 cell 聚合 $q(C_t)$ 衡量内容价值；集合冗余 $d_i(S)$ 判断候选信息是否已被当前集合表达。
SSA 以前两类信号建立锚点集合，GRVP 组合 $a_i$ 与 $d_i(S)$ 补充剩余 token。为忠实对应实验代码，§4.3
还披露一个仅用于 cell 内候选排序的辅助原型对齐项 $m_i$；该项不是独立动机，也不改变上述三类信号的职责
划分。下一节只描述如何实现这两个阶段，不再把算法组件本身当作动机。

---

## 4. 方法

### 4.1 问题设定与设计原则

给定视觉编码器输出的 $N$ 个 patch token，目标是在固定视觉 token 预算 $K$ 下选择 patch 索引集合 $S$，
使 $|S|=K-1$，并与恒保留的 CLS token 组成 $K$ 个视觉 token 送入 LLM。SCOPE 仅执行索引 gather，
不合并或重建被丢弃的 patch token。以 LLaVA-1.5 为例，选择发生在 CLIP 倒数第二层与 `mm_projector`
之间，候选集合包含 $N=576$ 个 patch token；本文报告的 $K$ 始终表示送入 LLM 的视觉 token 总预算。

SCOPE 遵循“先建立内容感知的空间证据下界，再用全局准则分配剩余预算”的原则。Stage I 的
Spectral-Semantic Anchoring（SSA）先选择来自不同显著空间 cell 的锚点；Stage II 的
Global Redundancy-Aware Visual Token Pruning（GRVP）随后在 SSA 锚点条件下补充显著且不重复的证据。
下文统一使用 SSA 和 GRVP 指代两个阶段，并将 $\rho$ 称为**锚点预算比例**。

### 4.2 记号

记 CLIP 倒数第二层 patch 特征 $f_i\in\mathbb R^D$（LLaVA-1.5 中 $D=1024$），其单位化
$\hat f_i=f_i/\lVert f_i\rVert$；
token 显著性 $a_i$ 在 LLaVA-1.5 中由 CLS-to-patch attention
$a_i=\sum_{r=1}^{H}\mathrm{Attn}^{(L_{\mathrm{vit}}-2)}_r[\mathrm{cls},i]$ 给出，其中 $L_{\mathrm{vit}}$
和 $H$ 分别为视觉编码器层数与注意力头数，其 min-max 归一化为 $\hat a_i$；patch
网格坐标 $p_i\in\{0,\dots,23\}^2$。总预算 $K$（CLS 恒保留，实际选 $K{-}1$ 个 patch），按锚点预算比例
$\rho\in[0,1]$ 拆成两部分：

$$
M=\lceil \rho\,(K-1)\rceil\ \ (\text{SSA 锚点预算}),\qquad
B=(K-1)-M\ \ (\text{GRVP 补充预算}).
$$

下文中，“锚点”仅指 SSA 输出的 $S_{\mathrm{anc}}$，“最终集合”仅指 SSA 与 GRVP 共同得到的 $S$；
Global-MMR 专指 $\rho=0$ 时不使用 SSA 锚点的退化基线。“空间覆盖”描述所选集合的性质，而 $\rho$ 始终
表示 SSA 锚点预算比例，不称为覆盖率。

### 4.3 Stage I：Spectral-Semantic Anchoring

SSA 旨在以 $M$ 个锚点建立内容感知的空间证据下界。仅按 token 显著性进行全局排序，会将每个 token 的
分数独立于已选集合进行估计，因而容易在少数显著物体周围重复分配预算；反之，均匀空间采样虽然能够扩大
覆盖范围，却默认所有区域具有相同价值，可能为大面积背景保留不必要的 token。锚点选择因此包含两个不能由
单一排序分数同时回答的问题：哪些内容区域值得被保留，以及每个区域中的哪个 token 能够作为可靠代表。

为分离这两个问题，我们将 SSA 表述为一个双层选择过程。设规则 patch 网格被划分为候选空间 cell
$\mathcal C=\{C_t\}_{t=1}^{P}$。对于每个 $C_t$，先依据锚点效用 $s_i$ 确定候选锚点 $\pi_t$；再依据 cell
显著性 $q(C_t)$ 从 $P$ 个候选空间 cell 中分配 $M$ 个锚点预算：

$$
\pi_t=\arg\max_{i\in C_t}s_i,
\qquad
\mathcal T^*=\arg\max_{\substack{\mathcal T\subseteq\{1,\ldots,P\}\\|\mathcal T|=M}}
\sum_{t\in\mathcal T}q(C_t).
$$

该形式化使“cell 内锚点选择”与“cell 间预算分配”具有明确分工。前者以频谱稳定性为核心，后者衡量显著性；
最终锚点集合为 $S_{\mathrm{anc}}=\{\pi_t\mid t\in\mathcal T^*\}$。

**频谱稳定性。** 相邻 patch 中可共享的视觉成分通常表现为空间上的缓慢变化，而对局部位置更敏感的细节则
更多体现为高频变化。因而，适合作为锚点的 token 应主要承载邻域共享的低频成分，而不是由边界、纹理或
局部噪声主导的高频残差。基于这一性质，我们用 token 无法被低频分量解释的残差衡量其频谱稳定性。将视觉特征
重排为 $g\in\mathbb R^{G\times G\times D}$，其中 LLaVA-1.5 对应 $G=24$。正交 DCT-II 的基矩阵为

$$
\mathbf W_{u,x}=\gamma_u\cos\!\left(\frac{\pi(2x+1)u}{2G}\right),\qquad
\gamma_0=\frac{1}{\sqrt G},\quad
\gamma_{u>0}=\sqrt{\frac{2}{G}}.
$$

对每个特征通道分别计算二维变换 $z_{:,:,d}=\mathbf Wg_{:,:,d}\mathbf W^\top$。令
$\mathcal P_{L_{\mathrm{lp}}}$ 表示仅保留左上角 $L_{\mathrm{lp}}\times L_{\mathrm{lp}}$ 系数的低通算子，
则低频重建可写为

$$
\tilde g_{:,:,d}=\mathbf W^\top\mathcal P_{L_{\mathrm{lp}}}(z_{:,:,d})\mathbf W.
$$

记 $\tilde f_i$ 为 $\tilde g$ 在第 $i$ 个 patch 位置的重建特征。据此定义第 $i$ 个 token 的高频残差
$h_i$ 及频谱稳定性 $\ell_i$ 为

$$
h_i=\lVert f_i-\tilde f_i\rVert_2,
\qquad \ell_i=-h_i.
$$

较大的 $\ell_i$ 表示该 token 的表示主要可由低空间频率解释，因此更适合作为所在 cell 的稳定锚点。这里的
频谱稳定性不等价于 token 显著性：平滑背景同样可能取得较高的 $\ell_i$。因此，$\ell_i$ 只回答“cell 内由谁
充当稳定锚点”，而该 cell 是否值得占用预算仍由后述 cell 显著性决定。默认设置下取
$L_{\mathrm{lp}}=16$；DCT 仅用于计算索引分数，不改变送入 LLM 的原始视觉特征。

在精确实现中，我们还加入一个轻量的**辅助原型对齐项**，用于避免锚点明显偏离当前 cell 的平均特征构型。
令 $\bar f_t=\operatorname{normalize}(\sum_{j\in C_t}\hat f_j)$，并定义
$m_i=\cos(\hat f_i,\bar f_t)$。该项只作为 cell 内排序的辅助约束；SCOPE 关于稳定锚点的核心动机与判据仍是
上述频谱稳定性，而非原型建模。结合频谱稳定性、cell 内 token 显著性与该辅助原型对齐项，token $i\in C_t$ 的
锚点效用为

$$
s_i=w_f\,\mathrm{Norm}_{C_t}(\ell_i)
+w_a\,\mathrm{Norm}_{C_t}(a_i)
+\mathrm{Norm}_{C_t}(m_i).
$$

其中，$\ell_i$ 是选择稳定锚点的核心信号，$a_i$ 避免锚点偏离 cell 内部的显著位置，$m_i$ 仅提供辅助
特征对齐。三个分量均在各自 cell 内进行 min-max 归一化，使其只参与同一 cell 内的候选比较，而不改变
cell 之间的预算分配。

**cell 显著性与锚点分配。** 频谱稳定性回答“谁适合作为该 cell 的稳定锚点”，但不能回答“该 cell 是否值得
保留”。尤其是，
直接从每个 cell 中选取一个代表会将背景与前景等量对待。为此，我们定义 cell 显著性

$$
q(C_t)=\max_{i\in C_t}a_i,
$$

并求解上述 cell 级目标。由于目标对各 cell 可分，$\mathcal T^*$ 等价于选取 $q(C_t)$ 最高的 $M$ 个 cell，从而
得到

$$
S_{\mathrm{anc}}
=\left\{\pi_t\mid t\in
\operatorname{TopM}_{u\in\{1,\ldots,P\}}q(C_u)\right\},
\qquad |S_{\mathrm{anc}}|=M.
$$

候选空间 cell 集合 $\mathcal C$ 通过 patch 坐标上的确定性最远点采样及 Voronoi 归属构造，并设置

$$
P=\min\!\left(N,\max\!\left(M,\operatorname{round}(cM)\right)\right),\qquad c\geq1,
$$

其中 $c$ 为**空间 cell 过分割系数**，实现参数名为 `cover_factor`。$c>1$ 时，过分割产生空间分散的候选
支撑，但不强制每个 cell 占用预算；cell 显著性进一步排除低信息 cell。由此，SSA 以频谱稳定性筛选 cell
内锚点，并以 cell 显著性把有限预算分配给重要区域，从而建立内容感知的空间证据下界，并为 GRVP 提供
初始证据集合。

### 4.4 Stage II：Global Redundancy-Aware Visual Token Pruning

SSA 只建立内容感知的空间证据下界，并未用完全部 token 预算。GRVP 的目标是在 SSA 锚点条件下分配
剩余的 $B$ 个名额，优先补充显著、同时尚未被当前集合表达的视觉信息。直接在剩余 token 中按显著性
排序仍会反复选择同一显著区域中的相似特征；仅追求特征差异则可能保留语义无关的异常或背景 token。因此，
我们将剩余预算表述为一个同时平衡 token 显著性与集合级冗余的增量选择问题。

令初始集合 $S^{(0)}=S_{\mathrm{anc}}$。对于任意尚未选择的 token $i\notin S^{(t)}$，定义其相对于当前集合的
冗余度为

$$
d_i\!\left(S^{(t)}\right)
=\left[\max_{j\in S^{(t)}}\cos(\hat f_i,\hat f_j)\right]_+,
$$

其中 $[x]_+=\max(x,0)$，并约定 $d_i(\varnothing)=0$。最大相似度刻画候选 token 能否被任一已选证据
替代；截断负相似度可避免将语义相反或无关的特征误解释为“负冗余奖励”。在此基础上，我们采用 Maximal
Marginal Relevance（MMR）分数

$$
u_i\!\left(S^{(t)}\right)
=\hat a_i-\lambda d_i\!\left(S^{(t)}\right),
$$

其中第一项衡量候选的 token 显著性，第二项惩罚其与当前已选集合之间的特征重复，$\lambda\geq0$ 控制两者
的权衡。在第 $t$ 次迭代中，选择

$$
i_t^*=\arg\max_{i\notin S^{(t)}}u_i\!\left(S^{(t)}\right),
\qquad
S^{(t+1)}=S^{(t)}\cup\{i_t^*\}.
$$

上述过程迭代 $B$ 次，得到最终 patch 索引集合

$$
S=S^{(B)},\qquad |S|=M+B=K-1.
$$

由于 MMR 由 $S_{\mathrm{anc}}$ 初始化，每个新 token 不仅需要区别于随后选出的补充 token，也需要区别于
SSA 已覆盖的内容。因此两个阶段并非彼此独立的两次筛选，而是通过共享集合状态顺序耦合：SSA 为不同显著
空间 cell 建立内容感知的空间证据下界，GRVP 则把剩余预算分配给锚点之外尚未表达的显著信息。最终，保留的
patch token 按原始 raster 顺序排列，与始终保留的 CLS token 拼接，并经 `mm_projector` 投影至 LLM 的
嵌入空间参与后续多模态推理。

上述分数也可写为等价的凸权重形式

$$
\alpha\hat a_i-(1-\alpha)d_i(S),
\qquad \alpha=\frac{1}{1+\lambda},
$$

其与原式仅相差正常数缩放，不改变候选排序。例如，$\lambda=0.5$ 对应 $\alpha=66.7\%$；当
$\lambda=0$ 时，GRVP 对剩余候选退化为显著性排序；只有同时令 $\rho=0$，完整 SCOPE 才退化为
Saliency Top-$K$（在 LLaVA-1.5 中即 CLS-to-patch attention 排序）。§5.7 使用该参数化解释文字密集
场景中的显著性—冗余权衡。

### 4.5 算法与复杂度

```python
def scope(f, a, K, rho, lam, cover_factor=3.0):
    M = ceil(rho * (K-1))                        # SSA anchor budget
    S = []
    if M > 0:
        P = min(N, max(M, round(cover_factor * M)))
        cells = fps_voronoi_cells(grid=24, budget=P)
        rep = [argmax_{i in c}(norm_c(m_i) + w_f*norm_c(l_i) + w_a*norm_c(a_i)) for c in cells]
        S = [rep[c] for c in topM_cells_by(cell_saliency, M)]      # SSA anchors
    for _ in range((K-1) - M):                   # GRVP
        red = {i: 0 if len(S) == 0 else max_{j in S} relu(cos(f_i, f_j)) for i not in S}
        i = argmax_{i not in S}(minmax(a)_i - lam * red[i])
        S.append(i)
    return sort([CLS] + S)                        # raster order
```

忽略特征维度 $D$ 时，GRVP 的复杂度为 $O(BN)$；实现中维护每个候选与已选集合的运行最大相似度，无需在
每轮重新计算完整相似度矩阵。SSA 还包含 DCT、锚点效用和 cell 显著性计算。若显式计入 $D$ 且令
$N=G^2$，单张图像的总复杂度为 $O(DG^3+BND)$；FPS–Voronoi 划分只依赖网格形状与 $P$，可按配置缓存。
在固定视觉维度和网格下，选择开销随 token 数与预算近似线性增长。剪枝后只有 $K$ 个视觉 token 进入 LLM，
因而直接缩短其输入序列；方法实现与注入位置见附录。

### 4.6 退化形式与消融对应

SCOPE 的若干退化形式对应三类常见选择策略，也为动机实验和消融提供了清晰对照：

| 极限 | 退化为 |
|---|---|
| $\rho=0$ | Global-MMR（无 SSA 锚点的显著性—冗余联合选择） |
| $\rho=0,\ \lambda=0$ | Saliency Top-$K$ |
| $c=1$ | 无候选过分割的均匀空间覆盖，每个 cell 都占用预算 |
| $\rho=1,\ w_a=0,\ c=1$ | 频谱稳定性驱动的空间采样（保留实现中的辅助原型对齐项） |

因此，SCOPE 与 $\rho=0$ 的 Global-MMR 对比直接隔离 SSA 的贡献；$w_f=0$ 则隔离频谱稳定性对 SSA
锚点质量的贡献。这两个对照分别回答“是否需要 SSA 锚点”和“频谱稳定性是否改善锚点质量”，不能互相替代。

### 4.7 超参数

实验固定空间 cell 过分割系数 $c=3$（实现参数 `cover_factor`）、$w_f=w_a=1.0$；辅助原型对齐项的系数固定为
1，且 GRVP 的冗余
惩罚不做空间衰减。方法只保留两个控制量：

- **锚点预算比例 $\rho$（主旋钮）**：控制 SSA 锚点与 GRVP 补充 token 的预算配比。在主预算档中，
  预算越紧，SSA 锚点越容易挤占显著核心，因此使用更小的 $\rho$（§5.5）。
- **冗余权重 $\lambda$（副旋钮）**：控制 GRVP 的显著性—冗余平衡，等价显著性权重
  $\alpha=\tfrac{1}{1+\lambda}$。

通用任务使用 $\lambda=0.5$；主工作点 $K=192$ 取 $\rho=0.5$，更紧预算取 $\rho=0.25$。文字密集任务的信息
分布不同，实验中关闭 SSA 并提高显著性权重更稳健。这里的任务自适应不是额外模块，而是对同一权衡的直接
解释：冗余越低、预算越紧，越不应强制分配大量 token 做空间铺展。

---

## 5. 实验：LLaVA-1.5-7B

### 5.1 实验设置

**模型与实现**：LLaVA-1.5-7B，视觉塔 CLIP-ViT-L/14-336。剪枝在视觉塔与 `mm_projector` 之间插入，
LLM 权重不变、无微调。**基线（Baseline）**为标准 LLaVA 的 576 个 patch token（vanilla）；SCOPE 的
$K$ 个视觉 token 由 1 个 CLS token 和 $K-1$ 个原始 patch token 组成。因此，实验中的 $K$ 与方法章节
一致，始终表示实际送入 LLM 的视觉 token 总数。

**基准与指标**（9 项，均用 lmms-eval 统一评测）：GQA、MMBench-EN、MME-P（perception score）、
MMStar、POPE（F1）、ScienceQA-IMG（下文简称 SQA-IMG）、TextVQA、VizWiz、OCRBench。
GQA/SQA-IMG/TextVQA/VizWiz 为
exact-match×100，MMBench 为 GPT 判分，POPE 为 F1×100，MMStar/OCRBench 为 accuracy×100。VizWiz 用带
本地标注的 val split；MMStar 用 val split（1,500 题，覆盖粗/细粒度感知、实例/逻辑推理、科技、数学
六大能力），可有效抵抗视觉无关的语言先验、更能考察真视觉能力。

此外在 **COCO Caption**（`coco2017_cap_val`，5,000 图，CIDEr/BLEU/METEOR/ROUGE-L）上做补充的生成式
压力测试（§5.4）；因其为生成式 n-gram 指标、与上述判别式准确率不同量纲，**不计入主表平均**。

**预算档**：主表报告 $K\in\{192,128,64\}$，即保留 $1/3$、$2/9$、$1/9$ 的视觉 token；另附
$K\in\{288,346\}$（保留 50%/60%）作为无损前置初筛（§5.8）。

**FLOPs 口径**：遵循 HiMAP 的 LLM attention + MLP 计算方式，单个 LLM 层处理 $n$ 个视觉 token 的 FLOPs 为

$$
\mathrm{FLOPs}_{\mathrm{layer}}(n)=4n d_{\mathrm{model}}^2+2n^2d_{\mathrm{model}}
+2n d_{\mathrm{model}}d_{\mathrm{ff}},
$$

其中 LLaVA-1.5-7B 的 hidden size $d_{\mathrm{model}}=4096$、FFN intermediate size
$d_{\mathrm{ff}}=11008$，共 32 层。
该口径衡量由视觉序列
引起的 LLM 计算，不包含视觉编码器、token 选择器与自回归解码。对应结果为：

| $n$ | 32 层 LLM FLOPs | 相对 $n=576$ 降低 |
|---:|---:|---:|
| 576 | 2.986 TFLOPs | — |
| 192 | 0.976 TFLOPs | 67.3% |
| 128 | 0.649 TFLOPs | 78.3% |
| 64 | 0.323 TFLOPs | 89.2% |

**协议说明（TextVQA）**：见 §1.1。本文所有剪枝运行均为**无 OCR 协议**，故 TextVQA 一律以
**Baseline 46.07** 为参照；其余基准不受影响。

### 5.2 主结果：相对 Baseline 的保留率

表 1 给出 SCOPE 相对 Baseline 的保留率。为避免把任务特定调参混入主结论，所有任务均使用同一套
通用规则：$\lambda=0.5$，$K=192$ 时 $\rho=0.5$，$K=128/64$ 时 $\rho=0.25$。文字感知配置作为独立
分析放在 §5.7，不参与主表平均。

**表 1. 相对 Baseline（576）的保留率（LLaVA-1.5-7B）**

| Benchmark | Baseline (576) | 1/3 (K=192) | 2/9 (K=128) | 1/9 (K=64) |
|---|---:|---:|---:|---:|
| GQA | 100.0% | 96.5% | 95.6% | 92.7% |
| MMBench-EN | 100.0% | 98.8% | 97.9% | 93.2% |
| MME-P | 100.0% | 97.6% | 93.4% | 91.8% |
| MMStar | 100.0% | **100.6%** | 97.6% | 95.3% |
| POPE (F1) | 100.0% | **101.2%** | **100.7%** | 98.1% |
| SQA-IMG | 100.0% | 99.6% | 99.2% | 98.1% |
| TextVQA | 100.0% | 96.4% | 94.6% | 90.1% |
| VizWiz | 100.0% | **101.7%** | **102.3%** | **104.2%** |
| OCRBench | 100.0% | 98.7% | 95.8% | 92.6% |
| **平均保留率** | 100.0% | **99.0%** | **97.5%** | **95.1%** |

**表 2. 绝对分数（同上配置）**

| Benchmark | Baseline | K=192 | K=128 | K=64 |
|---|---:|---:|---:|---:|
| GQA | 61.97 | 59.78 | 59.25 | 57.47 |
| MMBench-EN | 64.00 | 63.23 | 62.63 | 59.62 |
| MME-P | 1511.3 | 1474.5 | 1411.7 | 1387.4 |
| MMStar | 33.56 | 33.76 | 32.74 | 31.98 |
| POPE (F1) | 85.88 | 86.93 | 86.44 | 84.26 |
| SQA-IMG | 69.46 | 69.16 | 68.91 | 68.12 |
| TextVQA | 46.07 | 44.40 | 43.60 | 41.50 |
| VizWiz | 54.06 | 54.99 | 55.33 | 56.32 |
| OCRBench | 31.20 | 30.80 | 29.90 | 28.90 |

**观察**：

1. **实用预算下整体损失有限**：$K=192$ 仅保留三分之一视觉 token，九项平均保留率为 99.0%；$K=128$
   时为 97.5%。该结果来自一套固定配置，不依赖按 benchmark 选择超参数。
2. **收益集中在可过滤冗余的任务**：POPE、VizWiz 与 MMStar 平均分在部分预算超过 Baseline。我们将其
   解释为冗余或干扰被移除的可能效应，而非普遍的“剪枝提升”，因为要求完整细节的任务呈相反趋势。
3. **MMStar 平均分稳健但需分解看**（见 §5.3）：平均保留 100.6%/97.6%/95.3%，但该均值由六个能力子集
   等权平均，其中「科技」「数学」「逻辑推理」三项 LLaVA-1.5-7B 本身就在四选一随机水平（25%）附近甚至
   以下，其"上升"主要是向随机水平回归的噪声，而非真实增益；真正视觉相关的**粗/细粒度感知**随预算单调
   下降。因此 MMStar 平均分不宜单独作为无损证据。
4. **文字任务更依赖配置与预算**：统一配置下，TextVQA 在 $K=192/128$ 保留 96.4%/94.6%，OCRBench
   保留 98.7%/95.8%。§5.7 进一步分析减小 SSA 锚点预算是否能缓解损失。

### 5.3 MMStar 能力子集分解

| MMStar 子集 | Baseline | K=192 | K=128 | K=64 |
|---|---:|---:|---:|---:|
| coarse perception | 63.87 | 61.50 (96.3%) | 59.22 (92.7%) | 53.74 (84.1%) |
| fine-grained perception | 25.63 | 24.44 (95.4%) | 21.73 (84.8%) | 19.43 (75.8%) |
| instance reasoning | 38.89 | 38.13 (98.1%) | 39.01 (100.3%) | 39.92 (102.7%) |
| logical reasoning | 28.92 | 31.42 (108.7%) | 29.27 (101.2%) | 29.52 (102.1%) |
| math | 26.31 | 28.38 (107.8%) | 27.19 (103.3%) | 27.40 (104.1%) |
| science & technology | 17.76 | 18.67 (105.1%) | 20.01 (112.7%) | 21.88 (123.2%) |
| **average** | **33.56** | **33.76 (100.6%)** | **32.74 (97.6%)** | **31.98 (95.3%)** |

分解显示两种相反趋势：**感知类子集**（coarse/fine-grained perception，Baseline 分别 63.87/25.63，显著
高于随机）随预算收紧单调下降，$K{=}64$ 仅保留 84.1%/75.8%，与 GQA/MME-P 的趋势一致；而 science &
technology（Baseline 17.76，**低于**四选一随机 25%）、math（26.31）、logical reasoning（28.92）这三项
Baseline 已在随机水平附近，其分数随剪枝"上升"缺乏视觉可解释性，应视为噪声。因此本文在正文以 MMStar
平均分参与统计、但结论以感知子集与 GQA/MME-P 为准。

### 5.4 COCO Caption：密集描述任务上的压力测试

上述基准均为判别式 QA（exact-match / 多选 / 判分）。为检验剪枝在**生成式密集描述**下的代价，我们在
COCO Caption（`coco2017_cap_val`，全量 5,000 张图）上补充评测。该任务要求模型复述全图内容、无法依赖
语言先验蒙对答案，因此对视觉覆盖的完整性最为敏感。配置沿用与主表相同的各档通用配置（$K{=}192$ 取
$\rho{=}0.5$，$K{=}128/64$ 取 $\rho{=}0.25$；$\lambda{=}0.5$、$c=3$）。
指标为标准 COCO 评测的 CIDEr / BLEU / METEOR / ROUGE-L。

**表 3. COCO Caption 结果（coco2017_cap_val，5,000 图；括号内为相对 Baseline 的保留率）**

| 指标 | Baseline (576) | K=192 | K=128 | K=64 |
|---|---:|---:|---:|---:|
| **CIDEr** | 1.1043 | 1.0763 (97.5%) | 1.0441 (94.6%) | 0.9894 (89.6%) |
| BLEU-4 | 0.2978 | 0.2886 (96.9%) | 0.2791 (93.7%) | 0.2658 (89.3%) |
| BLEU-1 | 0.7310 | 0.7187 (98.3%) | 0.7117 (97.4%) | 0.6958 (95.2%) |
| METEOR | 0.2930 | 0.2873 (98.1%) | 0.2814 (96.0%) | 0.2720 (92.8%) |
| ROUGE-L | 0.5563 | 0.5493 (98.7%) | 0.5422 (97.5%) | 0.5301 (95.3%) |
| *5 指标平均保留率* | 100.0% | *97.9%* | *95.8%* | *92.4%* |

> 本节结果**不计入表 1** 的平均保留率：CIDEr/BLEU 是生成式 n-gram 重合度指标，与判别式准确率量纲与
> 方差性质不同，混入同一均值会失去可比性。此处作为独立的压力测试单列。

COCO 呈现与判别式任务**相反的趋势**：所有指标随预算收紧**单调下降，无任何"剪枝反而更好"的反弹**。
把三类任务并置可看出清晰的分层：

| 任务性质 | $K{=}64$ 保留率 |
|---|---:|
| VizWiz（问题常可跳过视觉干扰） | 104.2% ↑ |
| MMStar average（含随机水平子集） | 95.3% |
| **COCO CIDEr（须复述全图）** | **89.6%** ↓ |
| **MMStar fine-grained perception** | **75.8%** ↓ |

这与 §5.3 的分解相互印证：VizWiz 的超越与 MMStar 平均分的"近无损"，很大程度来自这些任务允许忽略
视觉细节（干扰过滤、语言先验、随机水平子集）；一旦任务强制要求完整的细粒度视觉覆盖（COCO 复述、
MMStar 感知子集），$K{=}64$ 的真实代价就显现为 ~90%（CIDEr）乃至 ~76%（细粒度感知）。

因此我们对适用范围给出保守结论：$K{=}192$ 在生成式描述上仍保留 97.5% CIDEr、可视为近无损；
$K{=}128$（94.6%）可接受；而 $K{=}64$ 的强剪枝**不适用于**密集描述与细粒度感知场景，其在判别式
QA 上的高保留率不应外推到这类任务。

### 5.5 SSA 锚点预算比例 $\rho$

固定 $\lambda=0.5$、$c=3$ 和 $w_f=w_a=1.0$，扫描 $\rho\in\{0.25,0.5\}$，并以五项非文字任务相对
Baseline 的平均保留率为判据：

| K | $\rho{=}0.25$ | $\rho{=}0.5$ | 最优 |
|---|---:|---:|:--:|
| 64 | **94.12** | 93.88 | 0.25 |
| 128 | **97.02** | 96.54 | 0.25 |
| 192 | 97.89 | **98.35** | 0.5 |
| 288 | **98.60** | 98.42 | 0.25 |
| 346 | 99.23 | **99.27** | 0.5 |

在主要工作点 $K=192$，$\rho=0.5$ 优于 $\rho=0.25$；当预算收紧到 128 或 64 时，较小的 SSA 锚点集合更稳健，
符合“内容感知的空间证据下界不应挤占显著核心”的预期。宽预算 $K=288/346$ 的差异仅为 0.04–0.18 个百分点，尚不足以
支持严格单调规律。因此本文使用一条简洁规则：$K=192$ 取 $\rho=0.5$，$K=128/64$ 取 $\rho=0.25$；
更宽预算下两者均较稳定。

### 5.6 SSA 空间 cell 过分割系数 $c$

$c=1$ 时，候选 cell 数等于锚点数，每个 cell 都必须贡献一个 token，本质上接近均匀空间覆盖；$c=3$ 时，
先产生 $3M$ 个候选空间 cell，再按 cell 显著性选择 $M$ 个，使背景 cell 不再自动获得预算。现有预实验支持
后者，正文实验统一固定 $c=3$。不过，当前对照尚未完全隔离其他超参数，因此该结果只作为设计选择说明，
不作为独立贡献证据；正式投稿前仍需补充 $c\in\{1,3\}$ 的单变量消融。

### 5.7 内容边界：密集文字需要更小的 SSA 锚点预算

文字 patch 空间聚集、信息密度高，固定 SSA 锚点预算可能挤占字符证据。为避免在 OCRBench 上过拟合，我们采用
开发—验证分离的选参协议：先在 TextVQA 上选参，再在 OCRBench 上独立验证。

**TextVQA 搜索**（exact-match×100）：

| 配置 | $\rho$ | $\lambda$ | $\alpha$ | K=64 | K=128 | K=192 |
|---|---:|---:|---:|---:|---:|---:|
| 主表 SCOPE† | 按预算 | 0.5 | 66.7% | 41.50 | 43.60 | 44.40 |
| Saliency Top-$K$ | 0 | 0 | 100% | **42.54** | 44.33 | 44.68 |
| Global-MMR | 0 | 0.1 | 90.9% | 42.49 | **44.55** | **45.16** |
| Global-MMR | 0 | 0.25 | 80.0% | 42.38 | 44.04 | 45.06 |

† 主表规则在 $K=64/128$ 时使用 $\rho=0.25$，在 $K=192$ 时使用 $\rho=0.5$。

**OCRBench 独立验证**（accuracy×100，Baseline=31.20；配置只在 TextVQA 上选出，OCRBench 未参与选参）：

| K | 主表 SCOPE | TextVQA 选定配置 | Δ | %van |
|---:|---:|---|---:|---:|
| 64 | 28.90 | $\rho{=}0,\lambda{=}0$：28.50 | -0.40 | 91.35% |
| 128 | 29.90 | $\rho{=}0,\lambda{=}0.1$：30.00 | +0.10 | 96.15% |
| 192 | 30.80 | $\rho{=}0,\lambda{=}0.1$：**31.20** | **+0.40** | **100.00%** |

结果表明，文字任务确实倾向于更小的 SSA 锚点预算比例：在 $K=128/192$ 时，$\rho=0,\lambda=0.1$ 在 TextVQA
上最优，并在未参与选参的 OCRBench 上获得小幅一致收益。Saliency Top-$K$ 并非始终最好，说明即使文字区域信息
密集，少量特征去冗仍有价值。$K=64$ 时两个文字基准的偏好不一致，因此不能宣称存在统一文字配置。基于这
一点，§5.2 主表仍采用固定通用配置，本节仅用于解释内容类型如何改变“SSA 锚点—token 显著性”权衡。

### 5.8 与 Global-MMR、VisionZip 的对比

为比较 SSA 锚点、无锚点的全局去冗和选择—合并策略，我们在多个预算下对比 SCOPE、其
$\rho=0$ 退化形式 Global-MMR 以及 VisionZip。TextVQA 列使用无 OCR 协议（Baseline=46.07）；此表 MME 用
**总分** perception+cognition，与表 1/2 的 MME-P 不同尺度，POPE 用 accuracy。SCOPE 沿用 §5.2 的
主表规则，Global-MMR 固定 $\lambda=0.5$：

| K | 方法 | GQA | MMBench-EN | MME（总分） | POPE（acc.） | SQA-IMG | TextVQA | 非文字平均保留率 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 576 | Baseline（vanilla） | 62.0 | 64.0 | 1875 | 87.0 | 69.5 | 46.1 | 100.0% |
| 192 | VisionZip | 59.3 | **63.8** | 1770 | 86.4 | 68.7 | **44.5** | 97.6% |
| 192 | Global-MMR | 59.5 | 63.0 | 1783 | 87.5 | 68.4 | **45.1** | 97.7% |
| 192 | **SCOPE** | **59.8** | 63.2 | **1803** | **87.6** | **69.2** | 44.4 | **98.4%** |
| 128 | VisionZip | 57.7 | 62.2 | **1764** | 84.6 | 68.7 | **43.8** | 96.1% |
| 128 | Global-MMR | 59.3 | 62.3 | 1719 | **87.2** | 68.8 | **43.9** | 96.8% |
| 128 | **SCOPE** | **59.3** | **62.6** | 1728 | **87.2** | **68.9** | 43.6 | **97.0%** |
| 64 | VisionZip | 55.2 | **60.1** | **1718** | 80.6 | **69.0** | **42.0** | 93.3% |
| 64 | Global-MMR | **57.6** | 59.8 | 1639 | **86.3** | 68.3 | **42.0** | **94.2%** |
| 64 | SCOPE | 57.2 | 59.6 | 1670 | 85.5 | 67.9 | 41.5 | 94.1% |
| 288 | Global-MMR | 61.0 | **63.9** | 1779 | **87.6** | 68.6 | 45.1 | 98.5% |
| 288 | **SCOPE** | 61.0 | 63.8 | **1785** | 87.5 | **68.7** | **45.4** | **98.6%** |
| 346 | Global-MMR | 61.2 | 64.6 | **1831** | **87.3** | 68.6 | 45.6 | **99.3%** |
| 346 | SCOPE | 61.2 | **64.9** | 1824 | 87.1 | **68.7** | 45.6 | 99.3% |

三点发现：

1. **纯选择在非文字任务平均值上更稳健**：SCOPE 与 Global-MMR 在各预算的五项非文字平均保留率
   均高于 VisionZip，但并非每个单项都占优。例如，$K=64$ 时 VisionZip 在 MMB 与 SQA 上更高，而纯选择
   在 POPE 上优势明显。因此证据支持的是整体权衡，而不是“选择必然优于合并”的普遍结论。
2. **SSA 锚点的收益集中在实用预算**：相较 Global-MMR，SCOPE 在 $K=192$ 将非文字平均保留率
   从 97.7% 提升至 98.4%，且五项均不低于前者；$K=128/288$ 仍有小幅平均收益。当预算极紧或接近饱和时，
   两者基本持平，说明 SSA 锚点主要在“预算足以容纳多区域、但仍需选择”的区间发挥作用。
3. **宽预算可作为保守初筛点**：$K=288/346$ 时非文字平均保留率为 98.6–99.3%。这一结果说明方法适合
   用作低风险前置压缩，但本文未报告端到端时延，因而不把性能保留率直接等同于系统加速比。

---

## 6. 跨架构泛化：Qwen2.5-VL-7B

为验证 SCOPE 的两阶段机制并非 LLaVA/CLIP 架构特有，我们将其迁移到
Qwen2.5-VL-7B——一个无 CLS token、采用动态分辨率 NaViT 编码、且视觉塔结构与 CLIP 显著不同的架构。

### 6.1 适配方法

SCOPE 在 Qwen2.5-VL 的 **2×2 PatchMerger 之后、序列拼入语言模型之前**插入。

由于 Qwen2.5-VL 没有 CLS token，本节中的 $K$ 表示最终保留的 PatchMerger token 总数，SSA 与 GRVP 的
预算分别为 $M=\lceil\rho K\rceil$ 和 $B=K-M$。除这一架构适配外，$\rho$ 的含义仍是 SSA 锚点预算比例。

- **token 显著性信号的替代**：Qwen 没有 CLS token，用视觉塔最后一层全注意力 block 中每个合并后 patch
  **被接收到的注意力**（对 head 与 query 位置取平均）定义 $a_i$，替代 LLaVA 中的 CLS-to-patch attention。
- **打分特征的对齐**：将最后一个全注意力 block 的 key 特征按 PatchMerger 的 $2\times2$ 分组进行平均，
  得到与 PatchMerger token 一一对应的打分特征 $f_i^{\mathrm Q}$。SSA 的辅助原型对齐项和 GRVP 的集合冗余均
  基于 $f_i^{\mathrm Q}$ 计算，而最终 gather 的仍是相同索引对应的原始 PatchMerger 输出。
- **SSA 空间 cell 与稳定性指标的适配**：使用每张图各自的动态矩形 NaViT 网格（而非固定
  $24\times24$）执行 FPS–Voronoi 划分。由于 LLaVA 版本的固定网格 DCT 不被直接复用，我们以特征范数
  相对局部 $3\times3$ 均值的负绝对偏差
  $\ell_i^{\mathrm Q}=-\left|\lVert f_i^{\mathrm Q}\rVert_2-
  \operatorname{Avg}_{3\times3}(\lVert f^{\mathrm Q}\rVert_2)_i\right|$
  作为频谱稳定性的局部平滑代理，并保留 cell 内显著性与辅助原型对齐项。cell 显著性分配保持不变。
- **GRVP 不变**：仍采用 §4.4 的显著性—冗余联合准则，并以 SSA 锚点集合初始化。
- **输出形式不变**：被选中的输出仍是原始 PatchMerger token 的纯 gather，无重建、无额外合并。

Qwen 适配与 LLaVA 版本共享相同的 SSA–GRVP 两阶段结构及信号分工，但替换了 token 显著性来源、空间
cell 构造和稳定性代理。因此，该实验检验的是两阶段设计原则的跨架构迁移，而不是固定网格 DCT 指标的
原样迁移。具体实现入口与内部结果路径见作者复现备忘录。

### 6.2 评测与实现核验

我们在推理路径中加入运行时断言，逐样本检查实际保留 token 数是否等于目标预算，并在剪枝后同步更新位置
与缓存索引。所有报告结果均来自通过上述检查的重跑实验；失效的早期运行不参与任何表格或结论。更具体的
版本与故障排查记录保留在复现附录，而不作为方法贡献的一部分。

### 6.3 主结果

我们使用 Qwen2.5-VL-7B，在 lmms-eval 中以 eager attention、batch size 1 评测。为检验直接迁移而非再次
按任务调参，所有任务固定 $\rho=0.5,\lambda=0.5,c=3$，只改变 $K$。Baseline 为相同权重
和协议下的未剪枝模型。

**绝对分数**

| Benchmark | Baseline | K=192 | K=128 | K=64 |
|---|---:|---:|---:|---:|
| GQA | 58.21 | 57.57 | 56.65 | 54.30 |
| MME-P | 1653.48 | 1606.53 | 1590.92 | 1512.53 |
| MMStar | 58.27 | 57.72 | 55.24 | 51.35 |
| POPE (F1) | 86.48 | 85.99 | 85.02 | 83.54 |
| SQA-IMG | 80.91 | 80.57 | 79.18 | 78.14 |
| TextVQA | 71.24 | 66.86 | 63.22 | 54.91 |
| VizWiz | 66.77 | 65.63 | 64.42 | 62.13 |
| OCRBench | 77.80 | 65.60 | 58.00 | 45.00 |

**相对 Baseline 保留率**

| Benchmark | Baseline | K=192 | K=128 | K=64 |
|---|---:|---:|---:|---:|
| GQA | 100.0% | 98.9% | 97.3% | 93.3% |
| MME-P | 100.0% | 97.2% | 96.2% | 91.5% |
| MMStar | 100.0% | 99.1% | 94.8% | 88.1% |
| POPE (F1) | 100.0% | 99.4% | 98.3% | 96.6% |
| SQA-IMG | 100.0% | 99.6% | 97.9% | 96.6% |
| TextVQA | 100.0% | 93.9% | 88.7% | 77.1% |
| VizWiz | 100.0% | 98.3% | 96.5% | 93.1% |
| OCRBench | 100.0% | 84.3% | 74.6% | 57.8% |
| **平均** | **100.0%** | **96.3%** | **93.0%** | **86.8%** |

### 6.4 结果解读

- **K 与 LLaVA-1.5 表不可直接对比**：LLaVA-1.5 每图固定输出 576 个视觉 token，故 $K{=}192/128/64$
  恰好是 1/3、2/9、1/9。Qwen2.5-VL 用动态分辨率 NaViT，剪枝前 token 数逐图不同，同一固定 $K$ 在不同
  benchmark 上对应完全不同的保留比例。实测中位数：**TextVQA 约 999 个 token**（$K{=}64$ 只保留
  6.4%，比 LLaVA 的 11.1% 更苛刻），而 **OCRBench 约 274 个 token**（$K{=}64$ 保留 23.3%，反而比
  LLaVA 更宽松）。
- **密集文字承担主要损失，且不能只用保留比例解释**：OCRBench（84.3% → 57.8%）与 TextVQA
  （93.9% → 77.1%）的退化明显快于其余任务。TextVQA 的部分损失来自较低保留比例，但 OCRBench 的原始
  token 更少、相对预算更宽松，退化仍最严重。这一对照与“文字信息密度高、局部可替代性较低”的解释一致，
  但尚不能证明频谱稳定性或其 Qwen 适配代理是唯一原因。
- **非文字任务相对稳健**：POPE 与 SQA-IMG 在 $K=64$ 时仍保留 96% 以上性能，而文字任务明显更低。
  因此，固定 $K$ 带来的比例变化与内容类型都会影响结果；二者不应相互替代解释。
- **MMBench-EN 缺失**：离线环境下该 benchmark 的答案抽取会回退到 OpenAI API（401 无 key），故本文
  未在 Qwen2.5-VL 上报告该项。

### 6.5 文字任务中的 SSA 锚点预算消融

**分析问题**：文字密集图像中 token 显著性本就集中在文字区域，SCOPE 的 SSA 锚点预算与 GRVP 冗余惩罚
是否会把预算挤占到非文字区域并损害 OCR？为此在 $K{=}64$ 的 OCRBench 上分别消融 $\rho$ 与 $\lambda$：

| $\rho$（SSA 锚点预算比例） | $\lambda$（GRVP 冗余权重） | OCRBench |
|---|---|---:|
| 0.50 | 0.50 | 45.00（当前部署配置） |
| 0.25 | 0.50 | 45.40 |
| **0.00** | **0.50** | **46.30**（最优） |
| 0.00 | 0.00 | 44.70 |

OCRBench 上，减小 SSA 锚点预算比例带来单调改善（45.00 → 45.40 → 46.30）；但在 $\rho=0$ 后进一步
移除 MMR 冗余惩罚会下降到 44.70。这一区分很关键：文字任务反对的并非所有多样性，而是固定分配给 SSA
锚点的预算；Global-MMR 的特征去冗仍能减少重复证据。

**$\rho{=}0$ vs $\rho{=}0.5$ 的配对逐样本显著性检验**：

| 任务 | K | $\rho{=}0.5$ | $\rho{=}0$ | Δ | 配对 p 值 |
|---|---:|---:|---:|---:|---|
| OCRBench | 192 | 65.60 | 66.60 | +1.00 | 0.131（不显著） |
| OCRBench | 128 | 58.00 | 60.20 | +2.20 | 0.009（**）|
| OCRBench | 64 | 45.00 | 46.30 | +1.30 | 0.138（不显著） |
| TextVQA | 192 | 66.86 | 66.81 | −0.05 | 0.874（不显著） |
| TextVQA | 128 | 63.22 | 63.94 | +0.72 | 0.064（不显著） |
| TextVQA | 64 | 54.91 | 56.72 | +1.80 | 0.0001（***）|
| GQA（非文字对照） | 64 | 54.30 | 54.17 | −0.13 | 0.602（不显著） |
| MMStar（非文字对照） | 64 | 51.35 | 51.68 | +0.00 | 1.000（不显著） |

修正后的文字任务保留率（$\rho{=}0$）：OCRBench 85.6/77.4/59.5%，TextVQA 93.8/89.8/79.6%
（对应 $K{=}192/128/64$）。

**该消融能证明与不能证明什么**：

- **效应方向一致但证据强度有限**：6 个文字任务—预算组合中只有 2 个达到统计显著。收益主要出现在
  $K=64$–$128$，符合紧预算下 SSA 锚点成本更高的预期；$K=192$ 时 TextVQA 几乎不变。
- **非文字对照尚不足以确定全局默认值**：GQA 与 MMStar 在 $K=64$ 下对 $\rho$ 不敏感，但仅凭两个任务
  不能说明关闭 SSA 对所有非文字任务都是“免费”的。完整结论需要在 POPE、SQA-IMG、VizWiz 与 MME-P 上
  复验。
- **不能弥合 OCR 差距**：$K{=}64$ 的 OCRBench 从 45.0 升到 46.3，相对 77.8 的 Baseline 仍有巨大差距。预算
  本身仍是主导杠杆，差距悬殊（同为 $\rho{=}0$ 下，$K{=}64\to128\to192$ 给出 46.3 → 60.2 → 66.6）。
  因此，SSA 锚点分配是二阶影响，预算本身仍是主要因素。纯选择无法恢复已经丢弃的字符证据，后续工作需要文字
  区域保护、内容自适应预算或受控合并。

### 6.6 小结：跨架构一致性与差异

Qwen2.5-VL 的意义不在于复现与 LLaVA 完全相同的绝对分数，而在于检验 SSA–GRVP 的两阶段结构能否迁移
到无 CLS、动态网格架构。结果显示，非文字任务在中等预算下仍较稳健，而密集文字在两个架构上都更脆弱。
这支持“可替代性随内容类型变化”的核心动机，但由于 Qwen 版本使用局部平滑稳定性代理而非固定网格 DCT，
该实验不能单独证明频谱稳定性指标的跨架构增益。动态网格还引入了固定 LLaVA 设定
中不可见的因素：相同 $K$ 对不同图像代表不同保留比例。因此，动态分辨率部署更适合按原始 token 数与内容
密度联合分配预算，而非使用单一全局 $K$。

---

## 7. 讨论与局限性

1. **局部可替代性不是普遍性质。** SCOPE 最适合大面积局部同质、可由代表 token 概括的视觉内容。
   密集文字和细粒度结构包含更多不可替代证据；一旦相应 patch 被删除，gather-only 方法无法恢复其信息。
   两个架构上的一致退化支持这一边界，但还不足以证明所有纯选择方法都具有相同上限。更稳妥的后续方向是
   文字感知的局部保护、内容自适应预算，或只对高置信冗余区域执行受控合并。
2. **适用范围的保守声明**。判别式 QA 的高保留率不能外推到所有任务：生成式密集描述（COCO CIDEr
   97.5%/94.6%/89.6%）与细粒度感知（MMStar fine-grained 95.4%/84.8%/75.8%）随预算收紧单调下降、
   无反弹；VizWiz 超越 Baseline、MMStar 平均分近无损这类乐观信号，部分来自这些任务本身允许忽略视觉细节
   （可过滤干扰、可依赖语言先验、含随机水平子集）。综合看，$K{=}192$ 是较保守的工作点
   （固定通用配置下判别式平均 99.0%、COCO CIDEr 97.5%）；$K{=}128$ 可接受；$K{=}64$ 仅推荐用于判别式 QA，
   **不建议**用于密集描述与细粒度感知场景。
3. **关键证据仍需补齐。** 当前稿件在投稿前至少需要完成以下实验：

   - 空间 cell 过分割系数 $c$（实现参数 `cover_factor`）的单变量干净消融尚未完全隔离其余超参，当前
     §5.6 的结论仍为定性描述，正式投稿前需补齐数据表；
   - Global-MMR 的 $\lambda$ 敏感性（$\{0.25,0.5,1.0\}$）与 SCOPE 更细的 $\rho$ 网格
     （$\{0.25,0.75\}$）尚未完整跑完；
   - 需要加入 SSA 锚点效用的逐项消融，并配合 §3.2 的 M1b 在 encoder 特征层直接比较锚点选择准则。其中
     $w_f=0$ 与仅频谱稳定性变体是支撑“频谱稳定性”主线最关键的因果证据；$w_a=0$ 检验语义约束，去掉
     $m_i$ 则用于说明辅助原型对齐项的实际影响。现有频谱统计与定性图只能证明低频主导现象存在，不能单独证明
     频谱稳定性能够改善锚点选择；
   - 需要报告 cell 占有率、两两空间距离、最近邻特征相似度和显著性质量保留率，直接验证方法确实同时改善
     空间覆盖、特征去冗和显著性质量保留；
   - 当前外部对照主要是 VisionZip。正式投稿需要补充同插入点、同视觉 token 预算下的近期训练无关选择与合并
     方法，并统一是否计入 CLS、合并 token 和选择器开销；
   - Qwen2.5-VL 上的 $\rho{=}0$ 目前只在 OCRBench/TextVQA/GQA/MMStar 四项上验证，其余 benchmark
     （POPE、SQA-IMG、VizWiz、MME-P）尚未在 $\rho{=}0$ 下重跑，因此尚不能将其设为跨任务默认值。
4. **尚未报告端到端系统指标。** §5.1 已按统一公式报告视觉序列对应的 LLM FLOPs，但尚未给出端到端
   延迟、吞吐、峰值显存与选择器自身耗时。这些指标不能由 FLOPs 或 token 保留比例直接替代。

---

## 8. 结论

本文从视觉 token 的局部可替代性出发提出 SCOPE：patch 按固定网格切分，同质区域内的多数 token 可由邻近
代表替代，而这种可替代性在图像上分布极不均匀。由此，剪枝的关键不是给 token 排序，而是判断哪些区域可以
只留一个代表、以及谁有资格充当代表。SSA 用频谱稳定性回答“谁能稳定承载邻域共享成分”，用 cell 显著性
回答“哪些 cell 值得预算”，从而建立 SSA 锚点集合；GRVP 随后以该集合初始化 MMR，补充显著且不重复的
证据。具体而言，频谱稳定性 $\ell_i$ 衡量锚点稳定性，$a_i$ 与 $q(C_t)$ 衡量 token/cell 显著性，
$d_i(S)$ 衡量集合级冗余；精确实现中的 $m_i$ 仅作为 cell 内辅助原型对齐项。SSA 与 GRVP 依次组合这些量构造
最终证据子集，而非并列堆叠互不相关的评分项。采用固定通用配置时，LLaVA-1.5-7B 在 $K=192/128/64$ 下
分别保留九项判别式基准平均 **99.0%/97.5%/95.1%** 的性能；$K=64$ 时视觉 token 减少 88.9%，按 §5.1
口径计算的视觉序列对应 LLM FLOPs 降低 89.2%。Qwen2.5-VL 的迁移结果进一步说明 SSA–GRVP 的两阶段结构不依赖固定
网格或 CLS token，但不构成固定网格 DCT 指标跨架构有效性的直接证据。
同时，COCO Caption、MMStar 细粒度感知与文字任务共同划定了适用边界：当图像证据密集且局部冗余较低时，
强剪枝会稳定损失信息。因此，SCOPE 的结论不是“多数 token 都可无损删除”，而是只有在先识别可替代冗余、
再通过 SSA 约束空间证据并由 GRVP 补充非重复证据时，视觉 token 压缩才能在效率与视觉完整性之间取得
可靠平衡。

---

## 附录：复现协议

SCOPE 使用确定性的 FPS–Voronoi 空间划分与贪心选择，不引入可学习参数。LLaVA-1.5-7B 中，方法
作用于 CLIP 倒数第二层输出与多模态投影器之间；Qwen2.5-VL-7B 中，方法作用于 PatchMerger 之后、视觉
序列拼入 LLM 之前。两个 SCOPE 实现均 gather 对应插入位置上的原始视觉 token，不执行特征重建或额外
合并；作为外部基线的 VisionZip 仍遵循其原有的选择—合并流程。

我们对每个评测样本记录剪枝前后 token 数，并用运行时断言确认实际保留数与目标预算一致。Baseline 与方法共享
相同的模型权重、数据划分、prompt 模板和评分脚本。TextVQA 的所有结果均来自不提供参考 OCR token 的同一
协议；协议不一致的历史结果不参与比较。Qwen2.5-VL 的可变视觉网格按样本分别构造 cell，固定 $K$ 因而不
代表固定保留比例。代码发布将包含完整环境、配置、逐样本输出与聚合脚本，以支持表格复算。

LLaVA-1.5 上 §3.1–§3.2 的冗余与频率诊断直接复用推理路径中的 DCT 与 cell 构造实现，保证分析与方法
使用同一套
$L_{\mathrm{lp}}$、空间 cell 过分割系数 $c$（实现参数 `cover_factor`）与 cell 划分。定性可视化包括
原图、cell 边界、
原始特征、低通重建特征与残差热力图；定量统计在 200 张 GQA 图像上汇总 token 级重建残差与频谱能量。
作者内部的
机器路径、历史故障和失效运行索引已移至 [author_repro_notes.md](author_repro_notes.md)，不属于投稿正文。
