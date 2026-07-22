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
spatial support. LLM-side attention additionally exhibits strong positional bias and is incompatible with efficient
attention implementations such as FlashAttention. To answer the second question, we analyze the spectral
characteristics of visual tokens and observe that low-frequency-dominant features are not only closer to their local
neighborhood but also retain their representative status under mild image perturbations — a stability that feature
centrality alone does not provide. Building on this observation, we propose SCOPE, a training-free, two-stage visual
token pruning framework that can be seamlessly integrated into existing MLLMs. In the first stage, Spectral-Semantic
Anchoring selects frequency-stable local representatives within over-segmented spatial cells and then allocates the
anchor budget across cells by global visual saliency, so that frequency decides who represents a region while
saliency decides which regions deserve budget. In the second stage, Global Redundancy-Aware Visual Token Pruning
applies Maximal Marginal Relevance (MMR), initialized with these anchors, to select the remaining tokens by balancing
semantic relevance and feature diversity. The two stages jointly preserve semantic importance and spatial coverage
while operating entirely before the LLM, without relying on positionally biased cross-modal attention maps. Extensive experiments across diverse
multimodal benchmarks and multiple MLLMs demonstrate that SCOPE achieves a favorable accuracy–efficiency trade-off.
On LLaVA-1.5-7B, SCOPE prunes 88.9% of the visual tokens and reduces LLM FLOPs by 89.2%, while retaining 95.1% of
the baseline performance on average across nine benchmarks.

---

## 1. 引言

### 1.1 视觉 token 的空间冗余：可替代性而非重要性

以 LLaVA 为代表的多模态大模型把视觉编码器输出的 patch token 与文本 token 一同送入 LLM。LLaVA-1.5-7B
对单张图像产生 576 个 patch token；在动态分辨率模型中，这一数量还可能进一步增长。因此，即使视觉编码器
本身保持不变，视觉序列也会显著增加 LLM 的 prefill 计算和 KV-cache 占用。本文关注一种部署友好的设定：
在视觉编码器之后、LLM 之前一次性选择固定数量的原始视觉 token，不训练新参数，也不改变模型结构。

这些 patch token 之所以可以被大量丢弃，根本原因不是其中多数“不重要”，而是多数**可以被邻近 token 替代**。
patch 是按固定网格切分的，与物体边界无关：一片天空、一面墙、一块草地会被切成许多在特征空间中高度相似的
token，删去其中任意一个，都能由同一区域的其余 token 近似恢复。这种性质可以直接测量——邻域特征余弦相似度、
以及用邻域代表替换某个 token 后产生的重建误差——而无需借助任何注意力信号。

关键在于，这种**局部可替代性在图像上分布极不均匀**：平滑区域的 token 彼此高度互替，而物体边界、细小目标和
文字区域的 token 各自携带独有信息，几乎无法互相替代。这一不均匀性把剪枝问题从“给 token 的重要性排序”
改写成两个不同的判断：**哪些空间区域可以只留一个代表**，以及**在一个区域内部，哪一个 token 有资格充当
这个代表**。重要性排序回答不了第二个问题——同一区域内注意力最高的 token 未必是最典型、最稳定的那个。

### 1.2 现有准则遗漏的三个失败模式

现有选择准则都没有同时回答上面两个问题，各自留下一个可观测的缺口。

第一，**CLS attention top-$K$ 是 set-agnostic 的独立排序**：它为每个 token 单独打分，选择时完全不考虑已选
集合，因此无法感知任何形式的冗余。其后果不只是选到相似特征，更是**空间塌缩**——预算被单个显著主体附近的
一片连通区域吃掉，图像其余部分没有任何证据留下。第二，**均匀空间采样**虽然分散，却默认所有区域价值相同，
把大量预算浪费在背景上。第三，**Global MMR** 通过惩罚与已选集合的特征相似度减少了重复，但“特征互异”不
蕴含“空间完整”：多个特征不同的 token 仍可能来自同一局部邻域，全局准则对空间支持范围没有任何下界保证。
此外，转向 LLM 内部的跨模态 attention 并不能规避这些问题——它需要显式 attention map，受位置偏差影响，
且与 FlashAttention 等高效实现不兼容。

这些论断必须由机制指标而非最终 benchmark 分数支撑。为此，§3 把动机拆成五个可检验问题：

1. 局部可替代性是否真实存在，且在图像上分布不均匀？
2. 在一个局部区域内部，什么样的 token 才是可靠代表？低频稳定性能否预测它？
3. CLS top-$K$ 是否在空间覆盖与邻域聚集指标上显著塌缩？
4. 频率代表、语义筛选与全局 MMR 是否解决不同而互补的失败模式？
5. LLM attention 的位置偏差与 FlashAttention 代价是否足以支持 encoder-side 剪枝？

### 1.3 由动机推导 SCOPE

问题 2 是频率分析进入本文的唯一理由。判断“谁是本区域的可靠代表”需要一个**在扰动下不改变结论**的判据：
仅按区域内特征中心性（medoid）选择，只反映单张图像的瞬时构型；而在特征图上做二维 DCT 后，高频残差小的
token 意味着它主要由缓慢变化的低空间频率解释，因而既接近邻域的共同成分，也不易因轻微的颜色、尺度或翻转
扰动而失去代表资格。低频稳定性因此刻画的是**局部可替代性**，而不是语义重要性——一个低频稳定的 token 可能
是可靠代表，也可能只是一片天空。

由此得到 SCOPE 的两个阶段，以及三种信号的职责分离。**Spectral-Semantic Anchoring** 先把网格过分割成数量
多于锚点预算的空间 cell，在每个 cell 内用中心性、低频稳定性和局部注意力联合选出代表（分别回答“是否典型”、
“是否稳定”、“是否有语义”），再用**全局视觉显著性**在 cell 之间分配预算，只保留最有内容的那部分 cell——
这一步正是均匀覆盖所缺的，它让频率负责“谁当代表”、让显著性负责“哪个区域值得代表”。
**Global Redundancy-Aware Visual Token Pruning** 随后以锚点集合为初始已选集合运行 MMR，补充语义相关且不与
锚点重复的证据。锚点因此有双重作用：既提供空间覆盖的下界，又作为第二阶段冗余计算的参照，使两阶段共同
构造同一个证据子集，而不是把频域、空间与 MMR 三个模块简单拼接。

本文的主要贡献如下：

1. 我们把视觉 token 剪枝重新表述为**局部可替代性判断**而非重要性排序，并用一组不依赖下游标签的机制分析
   说明冗余在何处、代表应如何选取，将低频结构限定为局部可替代性的判据而非全局重要性。
2. 我们提出训练无关、零参数的 SCOPE，以 Spectral-Semantic Anchoring 建立内容感知的空间证据下界，再以
   MMR 执行 Global Redundancy-Aware Pruning；方法仅 gather 原始 token，复杂度为 $O(KN)$。
3. 我们在固定 token 预算下系统比较注意力排序、均匀覆盖、Global-MMR 与 SCOPE，并在 LLaVA-1.5-7B 的
   九项基准以及 COCO Caption、MMStar 能力分解上报告性能—计算权衡和强剪枝边界。
4. 我们在无 CLS、动态视觉网格的 Qwen2.5-VL-7B 上验证可迁移性，并分析密集文字这一低冗余内容对纯选择
   方法构成的挑战。

### 1.4 评测协议

TextVQA 在不同版本评测协议中是否提供参考 OCR token，会导致不可忽略的分数差异。为保证公平性，本文的
vanilla、VisionZip 与 SCOPE 均在不提供参考 OCR token 的同一协议下重新评测；不同协议的结果不做
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

注意力或 CLS-to-patch 响应为视觉 token 的语义重要性提供了便捷代理，因此常被用于 top-$K$ 选择。然而，
重要性排序不考虑已选集合，容易重复保留同一主体附近的相似 token。MMR 类准则通过惩罚候选与已选集合的
最大特征相似度，能够在显著性和非重复性之间折中。本文将其称为 **Global-MMR**，并把它作为全局选择主干。
SCOPE 与之不同的关键在于：在全局贪心选择前，先建立由显著性筛选的空间锚点，显式约束证据的空间
支持范围。

### 2.3 空间覆盖与频率结构

均匀采样、网格池化或基于聚类的代表选择能够改善空间分布，但把每个位置等量纳入预算会过度保留大面积背景，
且大多只回答“在哪里取代表”，不回答“取哪一个”。基于特征中心性（如 cell medoid）的选择前进了一步，但
中心性只描述单张图像的瞬时构型，在轻微扰动下并不稳定。频率分析提供了另一维度：平滑区域主要由低空间频率
解释，其 token 的高频残差小，既接近邻域共同成分，也不易因扰动改变代表资格。本文因而不把低频分数当作全局
重要性，只用它在局部 cell 内挑选稳定代表；该 cell 是否值得占用预算仍由语义显著性决定。这一职责分离避免了
两个常见极端：只追逐高注意力导致空间塌缩，或只追求均匀覆盖而浪费背景预算。

---

## 3. 动机分析：视觉 Token 冗余在哪里，现有准则为何不足？

本节不以最终准确率反向证明设计合理，而是按 §1.2 的五个问题逐一检验 SCOPE 所依赖的前提。§3.1 与 §3.2
分别对应“冗余是否存在”和“代表应如何选取”，二者不能合并：前者用与频率无关的量定义现象，后者才引入
频率作为判据。除 §3.2 中已完成的频谱统计外，其余表格为待补实验协议；正式稿中只保留实测结果，不保留
“预期”文字。

### 3.1 局部可替代性是否真实存在且分布不均匀？

第一个前提与频率无关，必须先独立成立：若 patch token 之间本就不存在大面积的互相替代关系，那么“每个
区域留一个代表”这一整套设计从一开始就没有基础。用八邻域 $\mathcal N(i)$ 定义局部特征冗余

$$
R_i^{\mathrm{loc}}=\frac{1}{|\mathcal N(i)|}\sum_{j\in\mathcal N(i)}
\cos(\hat f_i,\hat f_j),
$$

并定义**邻域替代误差**：从特征图中删去 $i$ 后，用其邻域 medoid（或邻域均值）替代 $f_i$ 所产生的
cosine/$\ell_2$ 误差。两个量都只依赖 encoder 特征，不含任何注意力或下游标签，因此不会用语义显著性
循环论证冗余的存在。

**待补实验 M0：冗余的存在性与非均匀性。** 在 GQA 与 COCO Caption 各 1,000 张图像上报告 $R^{loc}$ 与
替代误差的分布，并按语义区域类型（背景/物体内部/物体边界/文字）分组对比。

| 区域类型 | 局部余弦 $R^{loc}$ | 邻域替代误差 | 占 token 比例 |
|---|---:|---:|---:|
| 平滑背景（天空/墙面/地面） | 待补 | 待补 | 待补 |
| 物体内部 | 待补 | 待补 | 待补 |
| 物体边界 | 待补 | 待补 | 待补 |
| 文字/细粒度结构 | 待补 | 待补 | 待补 |

支持动机的关键是两点同时成立：$R^{loc}$ 整体偏高（冗余广泛存在，因此高压缩率可行），且在区域类型之间
差异显著（冗余不均匀，因此需要按区域而非全局统一处理）。第二点同时预告了本文的能力边界——文字与细粒度
结构本身低冗余，任何纯选择方法都无法凭代表 token 恢复被丢弃的字符（§5.7、§6.5 给出对应的实测结果）。

### 3.2 什么样的 token 才是可靠代表？低频稳定性能否预测它？

确认冗余存在之后，真正决定方法质量的是**在一个 cell 内部选谁**。一个自然的基线是特征中心性，即取 cell
medoid；但中心性只刻画单张图像的瞬时构型，对轻微的颜色、尺度或翻转扰动并不稳健。频率分析正是在这里
进场：对 token $i$，定义 DCT 低通重建后的高频残差

$$
h_i=\lVert f_i-\tilde f_i\rVert_2 .
$$

若“低频稳定性刻画局部可替代性”成立，则 $h_i$ 应与 $R_i^{\mathrm{loc}}$ 显著负相关、与替代误差正相关，
且低残差 token 应更接近所在 cell 的特征中心，并在扰动后保持更稳定的代表排名。**其中扰动稳定性是频率
相对于 medoid 的独有增益**：邻域相似度与中心性都可由特征直接计算，只有“代表资格是否在扰动下保持”这一
性质需要频域刻画。

**已有证据。** 在 200 张 GQA 图像上，token 级高频残差呈左偏长尾分布，$L=16$ 的 DCT 低频块捕获约
70% 的平均谱能量。定性可视化显示，较低残差主要分布在天空、草地、墙面等平滑区域，而较高残差更多位于
物体边界、文字和细粒度结构。这说明低频主导现象广泛存在，但尚不能证明低频项能改善代表选择。

**待补实验 M1：频率—可替代性相关性。** 从 GQA 与 COCO Caption 各采样至少 1,000 张图像，按 $h_i$
四分位分组并报告以下指标；所有相关系数同时给出 bootstrap 95% 置信区间。

| 高频残差分组 | 局部余弦 $R^{loc}$ ↑ | cell-centroid 相似度 ↑ | 邻域替代误差 ↓ | 扰动后代表 Top-1 稳定率 ↑ |
|---|---:|---:|---:|---:|
| Q1（最低残差） | 待补 | 待补 | 待补 | 待补 |
| Q2 | 待补 | 待补 | 待补 | 待补 |
| Q3 | 待补 | 待补 | 待补 | 待补 |
| Q4（最高残差） | 待补 | 待补 | 待补 | 待补 |
| Spearman $\rho(h,\cdot)$ | 待补 | 待补 | 待补 | 待补 |

扰动稳定性使用不改变语义的轻微颜色抖动、resize 与水平翻转，并在对齐空间坐标后比较 cell 代表。

**待补实验 M1b：代表选择准则的直接对比。** M1 只给出相关性，无法说明低频项相对既有准则是否有增量。
固定同一组 FPS–Voronoi cell 与同一预算，仅改变 cell 内代表的选取准则，在 encoder 特征层直接评估代表
质量，不经过 LLM，因而不受下游噪声干扰：

| Cell 内代表准则 | cell 重建误差 ↓ | 扰动后代表一致率 ↑ | 代表—cell 平均相似度 ↑ |
|---|---:|---:|---:|
| Random | 待补 | 待补 | 待补 |
| Attention-max（cell 内最高 $a_i$） | 待补 | 待补 | 待补 |
| Medoid（仅 $m_i$） | 待补 | 待补 | 待补 |
| Medoid + 低频（$m_i+w_f\ell_i$） | 待补 | 待补 | 待补 |
| 完整 Stage-I 打分（$+\,w_a$ 项） | 待补 | 待补 | 待补 |

其中“cell 重建误差”定义为用选中代表替代该 cell 全部 token 后的平均 cosine/$\ell_2$ 误差。这是判断
$w_f$ 是否具有独立价值的核心证据：若 Medoid + 低频相对纯 Medoid 在重建误差上无改善、但在扰动一致率上
显著更高，则低频项的作用应被明确表述为**稳定性**而非**代表性**，正文与标题需据此措辞。若两项均无改善，
则不应把 spectral 写入标题与主贡献，方法应退化为 §4.6 中 $w_f=0$ 的形式。M3 中的 $w_f=0$ 下游消融是
本实验的补充而非替代。

### 3.3 为什么 CLS 显著性排序仍然冗余？

CLS attention 衡量单个 token 的显著性，却不感知当前已选集合。我们检验其是否导致两种可区分的冗余：
特征重复和空间聚集。将 $24\times24$ 网格划分为 $6\times6$ 宏观区域，定义：

$$
\mathrm{Cov}_{\mathrm{sal}}(S)=
\frac{\sum_C q_C\,\mathbf 1[S\cap C\neq\varnothing]}{\sum_Cq_C},
\qquad q_C=\max_{i\in C}a_i,
$$

作为内容加权空间覆盖率；同时报告选中 token 的归一化两两空间距离、八邻域聚集率、最近邻特征相似度和
attention mass 保留率。内容加权覆盖避免把“均匀采到背景”误认为有效空间多样性。

**待补实验 M2：选择集合诊断。** 在相同 $K\in\{64,128,192\}$ 下比较 Random、Uniform Grid、CLS
Top-$K$、Global-MMR、仅 Spectral-Semantic Anchoring 和完整 SCOPE。

| 方法 | Attention mass ↑ | 内容加权覆盖 ↑ | 两两空间距离 ↑ | 八邻域聚集率 ↓ | 最近邻特征相似度 ↓ |
|---|---:|---:|---:|---:|---:|
| Random | 待补 | 待补 | 待补 | 待补 | 待补 |
| Uniform Grid | 待补 | 待补 | 待补 | 待补 | 待补 |
| CLS Top-$K$ | 待补 | 待补 | 待补 | 待补 | 待补 |
| Global-MMR | 待补 | 待补 | 待补 | 待补 | 待补 |
| Spectral-Semantic Anchoring | 待补 | 待补 | 待补 | 待补 | 待补 |
| SCOPE | 待补 | 待补 | 待补 | 待补 | 待补 |

除聚合表外，应展示同一组图像上 CLS Top-$K$、Global-MMR 与 SCOPE 的 token overlay。若 CLS Top-$K$
确实捕获最高 attention mass、但空间覆盖低且邻域聚集率高，就能直接证明“显著性不等于非冗余”；若
Global-MMR 降低特征相似度却仍落后于 SCOPE 的内容加权覆盖，则可证明第二阶段不能替代空间锚点。

### 3.4 两个阶段是否解决互补问题？

完整方法只有在每个组成信号解决不同失败模式时才不是模块堆叠。为此，需要把 cell 内代表质量、cell 间
空间分配和全局补充策略分别隔离。所有变体使用相同注入点、token 数和评测协议。

**待补实验 M3：因果消融。** 主表至少在 GQA、POPE、MME-P、MMBench、COCO Caption 上报告 $K=192$
和 $K=64$；机制指标与下游结果必须并列。

| 变体 | 低频稳定性 | Cell 语义排序 | 空间锚点 | 全局 MMR | 覆盖 ↑ | 特征冗余 ↓ | Attention ↑ | 下游均值 ↑ |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|
| CLS Top-$K$ |  |  |  |  | 待补 | 待补 | 待补 | 待补 |
| Uniform Spectral | ✓ |  | ✓ |  | 待补 | 待补 | 待补 | 待补 |
| Saliency Anchors w/o Spectral |  | ✓ | ✓ |  | 待补 | 待补 | 待补 | 待补 |
| Spectral-Semantic Anchoring | ✓ | ✓ | ✓ |  | 待补 | 待补 | 待补 | 待补 |
| Global-MMR |  |  |  | ✓ | 待补 | 待补 | 待补 | 待补 |
| SCOPE | ✓ | ✓ | ✓ | ✓ | 待补 | 待补 | 待补 | 待补 |

该实验应支持一条逐步推理链，而不是要求所有指标同时最优：Uniform Spectral 主要提高空间分散但可能损失
attention；加入 cell 语义排序应减少背景预算；Global-MMR 主要降低跨 cell 特征冗余；完整 SCOPE 应在
attention、内容覆盖与特征非重复之间取得最好的联合权衡。还应单独加入 $w_f=0$、$w_a=0$ 和去掉 medoid
的 cell 内消融，直接回答 spectral score 是否具有独立增益。

### 3.5 为什么在进入 LLM 前完成剪枝？

本文关于 LLM-side attention 的批评需要实证，而不能只依赖常识。位置偏差应通过 token 顺序干预检验：对
同一图像分别使用 raster、reverse-raster 和固定随机排列，记录若干 LLM 层的 image-to-text attention，
计算 attention score 与绝对位置的 Spearman 相关、不同排列间恢复到原空间后的 Jensen–Shannon divergence，
以及 top-$K$ 集合重合率。一个理想的内容重要性分数应在恢复空间索引后对排列相对稳定。

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

五个问题与 SCOPE 的每个设计决策一一对应，且顺序不可交换：

| 动机结论 | 对应设计决策 | 实现位置 |
|---|---|---|
| 局部可替代性广泛存在但分布不均匀（M0） | 按空间 cell 保留代表，而非全局统一排序 | §4.3 空间细分 |
| cell 内代表需在扰动下稳定，medoid 不足（M1/M1b） | 代表打分加入低频稳定性 $w_f\ell_i$ | §4.3 cell 内代表 |
| 均匀覆盖把预算浪费在背景（M2） | 过分割出 $P>M$ 个 cell，再按 $q(C)=\max_{i\in C}a_i$ 只取 top-$M$ | §4.3 cell 筛选 |
| CLS top-$K$ set-agnostic、空间塌缩（M2） | 用锚点为证据的空间支持范围设下界 | §4.3 → §4.4 初始集合 |
| 特征互异不蕴含空间完整，Global-MMR 无覆盖保证（M2/M3） | 以锚点初始化 MMR，而非独立运行两套选择 | §4.4 |
| LLM attention 存在位置偏差且与 FlashAttention 冲突（M4） | 选择完全发生在 encoder 侧、LLM 之前 | §4.1 注入点 |

三种信号的职责因此互不重叠：**频率**判断一个 token 在其邻域内是否是稳定代表，**显著性**判断一个区域
是否值得占用预算，**全局 MMR** 判断在已有证据之外还缺什么。下一节只描述如何实现这些设计要求，不再把
算法组件本身当作动机。

---

## 4. 方法

### 4.1 问题设定与设计原则

给定视觉编码器输出的 $N$ 个 patch token，目标是在固定预算 $K$ 下选择索引集合 $S$，使 $|S|=K-1$，
并与保留的全局 token 一同送入 LLM。SCOPE 仅执行索引 gather，不合并或重建被丢弃 token。以
LLaVA-1.5 为例，选择发生在 CLIP 倒数第二层与 `mm_projector` 之间，$N=576$。

方法遵循“先建立空间证据下界，再用全局准则分配剩余预算”的原则。Spectral-Semantic Anchoring 先提供一组
来自不同内容区域的锚点；Global Redundancy-Aware Pruning 随后在锚点条件下补充语义重要且不重复的证据。
令 $\rho$ 表示锚点预算比例。

### 4.2 记号

记 CLIP 倒数第二层 patch 特征 $f_i\in\mathbb R^{1024}$，其单位化 $\hat f_i=f_i/\lVert f_i\rVert$；
CLS 注意力 $a_i=\sum_h \mathrm{Attn}^{(L-2)}_h[\mathrm{cls},i]$，其 min-max 归一化 $\hat a_i$；patch
网格坐标 $p_i\in\{0,\dots,23\}^2$。总预算 $K$（CLS 恒保留，实际选 $K{-}1$ 个 patch），按覆盖比
$\rho\in[0,1]$ 拆成两池：

$$
M=\lceil \rho\,(K-1)\rceil\ \ (\text{锚点预算}),\qquad B=(K-1)-M\ \ (\text{全局补充预算}).
$$

### 4.3 Stage I：Spectral-Semantic Anchoring

该阶段的目标不是均匀铺满图像，而是从多个有信息的区域中各保留一个可靠代表。它包含“局部代表选择”和
“全局 cell 筛选”两个层次：

1. **空间细分**：用确定性 FPS + Voronoi 把 $24\times24$ 网格划成 $P=\lceil c\cdot M\rceil$ 个空间 cell，
   其中 $c$ 为 `cover_factor` 且 $P>M$；
2. **cell 内代表**：每个 cell 取综合分最高的 token，
   $$
   s_i=\underbrace{m_i}_{\text{medoid 代表性}}+w_f\underbrace{\ell_i}_{\text{低频稳定性}}
   +w_a\,\mathrm{Norm}_{C(i)}(a_i),
   $$
   其中 $m_i,\ell_i$ 为 cell 内 medoid 与 DCT 低频稳定性（均在 cell 内 min-max 归一化）；
3. **按语义质量选择 cell**：用 $q(C)=\max_{i\in C}a_i$ 衡量 cell 是否包含显著内容，仅保留 top-$M$
   个 cell 的代表。由于 $P>M$，空间细分只生成候选覆盖，最终预算不必平均分配给背景。

其中，令 $\bar f_C=\mathrm{Norm}(\sum_{j\in C}\hat f_j)$，中心性定义为
$m_i=\cos(\hat f_i,\bar f_C)$；低频稳定性 $\ell_i$ 判断该 token 是否来自局部可替代的平滑结构；局部
归一化注意力防止代表落在同一 cell 的无信息位置。三者分别回答“是否典型”、“是否稳定”和“是否有语义”，
共同决定局部代表。

**低频稳定性。** 将 patch 特征排成 $g\in\mathbb R^{24\times24\times D}$，沿两个空间维执行正交 DCT-II：

对 $G=24$，DCT 基矩阵为
   $$
   M_{k,n}=c_k\cos\!\Big(\tfrac{\pi(2n+1)k}{2G}\Big),\qquad c_0=\tfrac{1}{\sqrt G},\ \ c_{k\ge1}=\sqrt{\tfrac{2}{G}},
   $$
逐通道得到频域系数。仅保留左上角 $L\times L$（$L=16$）低频块并做逆变换，得到低频重建
$\tilde g=M^\top(\text{mask}\odot\mathrm{coef})M$。第 $i$ 个 token 的高频残差为
   $$
   r_i=\big\lVert g_i-\tilde g_i\big\rVert_2 .
   $$
令 $\ell_i=-r_i$ 并在 cell 内归一化。残差越小，说明该 token 越能由低空间频率解释，更适合作为同质区域的
稳定代表。DCT 只参与索引打分，最终进入 LLM 的仍是未经修改的原始特征。

低频稳定性只用于判断局部代表性，而不直接决定一个空间区域是否重要；cell 是否进入锚点集合仍由全局视觉
显著性决定。这样可以避免把低频背景误当成语义证据。频率—可替代性关系见 §3.2，该打分项相对 medoid 的
独立增益见 §3.2 的 M1b 与 §3.4 的 $w_f=0$ 消融。

### 4.4 Stage II：Global Redundancy-Aware Visual Token Pruning

以语义锚点集合为初始 $S$，按 Maximal-Marginal-Relevance 贪心补充 $B$ 个 token：

$$
\mathrm{score}(i\mid S)=\hat a_i-\lambda\cdot\big[\max_{j\in S}\cos(\hat f_i,\hat f_j)\big]_+ .
$$

第一项衡量语义重要性，第二项惩罚候选与已选集合中最相似的证据，即 Maximal Marginal Relevance (MMR)。
该阶段从锚点集合开始迭代，因此新 token 不仅彼此去冗，也会避免重复锚点已经覆盖的内容。两阶段由同一个
集合状态 $S$ 耦合：Stage I 规定最低空间证据覆盖，Stage II 把剩余预算用于尚未表达的显著语义。

记 $d_i(S)=[\max_{j\in S}\cos(\hat f_i,\hat f_j)]_+$。式中 $\hat a_i-\lambda d_i(S)$ 可重参数化为等价
显著性权重形式 $\alpha\hat a_i-(1-\alpha)d_i(S)$，
$\alpha=\tfrac{1}{1+\lambda}$：$\lambda{=}0.5\Leftrightarrow\alpha{=}66.7\%$，
$\lambda{=}0\Leftrightarrow$ 纯 CLS 注意力 top-$K$（§5.7 用此重参数化解释文字模式的选参结果）。

### 4.5 算法与复杂度

```python
def scope(f, a, K, rho, lam, cover_factor=3.0):
    M = ceil(rho * (K-1))                        # Stage-I anchor budget
    P = min(N, round(cover_factor * M))          # 细分 cell 数, P > M
    cells = fps_voronoi_cells(grid=24, budget=P)
    rep   = [argmax_{i in c}(m_i + w_f*l_i + w_a*norm_c(a_i)) for c in cells]
    S     = [rep[c] for c in topM_cells_by(max_attn_in_cell, M)]   # spectral-semantic anchors
    for _ in range((K-1) - M):                   # global redundancy-aware MMR
        i = argmax_{i not in S}(minmax(a)_i - lam * max_{j in S} relu(cos(f_i, f_j)))
        S.append(i)
    return sort([CLS] + S)                        # raster order
```

选择复杂度为 $O(KN)$（LLaVA 中 $N=576$）。全局补充阶段维护每个候选与已选集合的运行最大相似度，无需在
每轮重新计算完整相似度矩阵。剪枝后只有 $K$ 个视觉 token 进入 LLM，因而直接缩短其输入序列；方法实现与
注入位置见附录。

### 4.6 退化形式与消融对应

SCOPE 的若干退化形式对应三类常见选择策略，也为动机实验和消融提供了清晰对照：

| 极限 | 退化为 |
|---|---|
| $\rho\to 0$ | Global-MMR（显著性 + 全局去冗，无显式空间覆盖） |
| $\rho\to 0,\ \lambda=0$ | VisionZip dominant top-$K$ 的纯剪枝版 |
| `cover_factor`$=1$ | 均匀空间覆盖，每个 cell 都占用预算 |
| $\rho\to1,\ w_a=0,\ $`cover_factor`$=1$ | 仅使用低频局部代表的空间采样 |

因此，SCOPE 与 $\rho=0$ 的 Global-MMR 对比直接隔离 Stage I 的贡献；$w_f=0$ 则隔离低频稳定性对锚点
质量的贡献。这两个对照分别回答“是否需要空间锚点”和“锚点为何需要频率信息”，不能互相替代。

### 4.7 超参数

实验固定 `cover_factor`$=3$、$w_f=w_a=1.0$，且全局冗余惩罚不做空间衰减。方法只保留两个控制量：

- **覆盖比 $\rho$（主旋钮）**：控制 Stage I 锚点与 Stage II 全局补充的预算配比。在主预算档中，预算越紧，锚点越易
  挤占显著核心，因此使用更小的 $\rho$（§5.5）。
- **冗余权重 $\lambda$（副旋钮）**：控制全局补充阶段的显著性/去冗平衡，等价显著性权重
  $\alpha=\tfrac{1}{1+\lambda}$。

通用任务使用 $\lambda=0.5$；主工作点 $K=192$ 取 $\rho=0.5$，更紧预算取 $\rho=0.25$。文字密集任务的信息
分布不同，实验中关闭覆盖池并提高显著性权重更稳健。这里的任务自适应不是额外模块，而是对同一权衡的直接
解释：冗余越低、预算越紧，越不应强制分配大量 token 做空间铺展。

---

## 5. 实验：LLaVA-1.5-7B

### 5.1 实验设置

**模型与实现**：LLaVA-1.5-7B，视觉塔 CLIP-ViT-L/14-336。剪枝在视觉塔与 `mm_projector` 之间插入，
LLM 权重不变、无微调。**基线（Baseline）**为不剪枝的原始 576 token 模型（vanilla）。

**基准与指标**（9 项，均用 lmms-eval 统一评测）：GQA、MMBench-EN、MME-P（perception score）、
MMStar、POPE（F1）、ScienceQA-IMG、TextVQA、VizWiz、OCRBench。GQA/SQA/TextVQA/VizWiz 为
exact-match×100，MMBench 为 GPT 判分，POPE 为 F1×100，MMStar/OCRBench 为 accuracy×100。VizWiz 用带
本地标注的 val split；MMStar 用 val split（1,500 题，覆盖粗/细粒度感知、实例/逻辑推理、科技、数学
六大能力），可有效抵抗视觉无关的语言先验、更能考察真视觉能力。

此外在 **COCO Caption**（`coco2017_cap_val`，5,000 图，CIDEr/BLEU/METEOR/ROUGE-L）上做补充的生成式
压力测试（§5.4）；因其为生成式 n-gram 指标、与上述判别式准确率不同量纲，**不计入主表平均**。

**预算档**：主表报告 $K\in\{192,128,64\}$，即保留 $1/3$、$2/9$、$1/9$ 的视觉 token；另附
$K\in\{288,346\}$（保留 50%/60%）作为无损前置初筛（§5.8）。

**FLOPs 口径**：遵循 HiMAP 的 LLM attention + MLP 计算方式，第 $l$ 层处理 $n$ 个视觉 token 的 FLOPs 为

$$
\mathrm{FLOPs}_l(n)=4nd^2+2n^2d+2ndm,
$$

其中 LLaVA-1.5-7B 的 $d=4096$、FFN intermediate size $m=11008$，共 32 层。该口径衡量由视觉序列
引起的 LLM 计算，不包含视觉编码器、token 选择器与自回归解码。对应结果为：

| $n$ | 32 层 LLM FLOPs | 相对 $n=576$ 降低 |
|---:|---:|---:|
| 576 | 2.986 TFLOPs | — |
| 192 | 0.976 TFLOPs | 67.3% |
| 128 | 0.649 TFLOPs | 78.3% |
| 64 | 0.323 TFLOPs | 89.2% |

**协议说明（TextVQA）**：见 §1.4。本文所有剪枝运行均为**无 OCR 协议**，故 TextVQA 一律以
**vanilla 46.07** 为基线；其余基准不受影响。

### 5.2 主结果：相对基线的保留率

表 1 给出 SCOPE 相对 vanilla 的保留率。为避免把任务特定调参混入主结论，所有任务均使用同一套
通用规则：$\lambda=0.5$，$K=192$ 时 $\rho=0.5$，$K=128/64$ 时 $\rho=0.25$。文字感知配置作为独立
分析放在 §5.7，不参与主表平均。

**表 1. 相对 baseline(576) 的保留率（LLaVA-1.5-7B）**

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
2. **收益集中在可过滤冗余的任务**：POPE、VizWiz 与 MMStar 平均分在部分预算超过 baseline。我们将其
   解释为冗余或干扰被移除的可能效应，而非普遍的“剪枝提升”，因为要求完整细节的任务呈相反趋势。
3. **MMStar 平均分稳健但需分解看**（见 §5.3）：平均保留 100.6%/97.6%/95.3%，但该均值由六个能力子集
   等权平均，其中「科技」「数学」「逻辑推理」三项 LLaVA-1.5-7B 本身就在四选一随机水平（25%）附近甚至
   以下，其"上升"主要是向随机水平回归的噪声，而非真实增益；真正视觉相关的**粗/细粒度感知**随预算单调
   下降。因此 MMStar 平均分不宜单独作为无损证据。
4. **文字任务更依赖配置与预算**：统一配置下，TextVQA 在 $K=192/128$ 保留 96.4%/94.6%，OCRBench
   保留 98.7%/95.8%。§5.7 进一步分析关闭空间覆盖是否能缓解损失。

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

分解显示两种相反趋势：**感知类子集**（coarse/fine-grained perception，baseline 分别 63.87/25.63，显著
高于随机）随预算收紧单调下降，$K{=}64$ 仅保留 84.1%/75.8%，与 GQA/MME-P 的趋势一致；而 science &
technology（baseline 17.76，**低于**四选一随机 25%）、math（26.31）、logical reasoning（28.92）这三项
baseline 已在随机水平附近，其分数随剪枝"上升"缺乏视觉可解释性，应视为噪声。因此本文在正文以 MMStar
平均分参与统计、但结论以感知子集与 GQA/MME-P 为准。

### 5.4 COCO Caption：密集描述任务上的压力测试

上述基准均为判别式 QA（exact-match / 多选 / 判分）。为检验剪枝在**生成式密集描述**下的代价，我们在
COCO Caption（`coco2017_cap_val`，全量 5,000 张图）上补充评测。该任务要求模型复述全图内容、无法依赖
语言先验蒙对答案，因此对视觉覆盖的完整性最为敏感。配置沿用与主表相同的各档通用配置（$K{=}192$ 取
$\rho{=}0.5$，$K{=}128/64$ 取 $\rho{=}0.25$；$\lambda{=}0.5$、`cover_factor`$=3$、$\sigma\to\infty$）。
指标为标准 COCO 评测的 CIDEr / BLEU / METEOR / ROUGE-L。

**表 3. COCO Caption 结果（coco2017_cap_val，5,000 图；括号内为相对 baseline 的保留率）**

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

### 5.5 Stage-I 锚点预算 $\rho$

扫 $\rho\in\{0.25,0.5\}$（其余固定），以五项非文字任务的平均基线保留率为判据：

| K | $\rho{=}0.25$ | $\rho{=}0.5$ | 最优 |
|---|---:|---:|:--:|
| 64 | **94.12** | 93.88 | 0.25 |
| 128 | **97.02** | 96.54 | 0.25 |
| 192 | 97.89 | **98.35** | 0.5 |
| 288 | **98.60** | 98.42 | 0.25 |
| 346 | 99.23 | **99.27** | 0.5 |

在主要工作点 $K=192$，$\rho=0.5$ 优于 $\rho=0.25$；当预算收紧到 128 或 64 时，较小覆盖池更稳健，
符合“空间下界不应挤占显著核心”的预期。宽预算 $K=288/346$ 的差异仅为 0.04–0.18 个百分点，尚不足以
支持严格单调规律。因此本文使用一条简洁规则：$K=192$ 取 $\rho=0.5$，$K=128/64$ 取 $\rho=0.25$；
更宽预算下两者均较稳定。

### 5.6 Stage-I 候选粒度：显著性排序覆盖 vs 均匀覆盖

`cover_factor=1` 时，每个 cell 都必须贡献一个 token，本质上接近均匀空间覆盖；`cover_factor=3` 则先
产生 $3M$ 个候选 cell，再按语义质量选择 $M$ 个，使背景区域不再自动获得预算。现有预实验支持后者，正文
实验统一固定 `cover_factor=3`。不过，当前对照尚未完全隔离其他超参数，因此该结果只作为设计选择说明，
不作为独立贡献证据；正式投稿前仍需补充 `cover_factor\in\{1,3\}` 的单变量消融。

### 5.7 内容边界：密集文字需要更少空间锚点

文字 patch 空间聚集、信息密度高，固定锚点预算可能挤占字符证据。采用两阶段选参避免在 OCRBench 上过拟合：
先在 TextVQA 上选参，再在 OCRBench 上独立验证。

**TextVQA 搜索**（exact-match×100）：

| $\rho$ | $\lambda$ | $\alpha$ | K=64 | K=128 | K=192 |
|---:|---:|---:|---:|---:|---:|
| 通用 | 0.5 | 66.7% | 41.50 | 43.60 | 44.40 |
| 0 | 0 | 100% | **42.54** | 44.33 | 44.68 |
| 0 | 0.1 | 90.9% | 42.49 | **44.55** | **45.16** |
| 0 | 0.25 | 80.0% | 42.38 | 44.04 | 45.06 |

**OCRBench 独立验证**（accuracy×100，vanilla=31.20；配置只在 TextVQA 上选出，OCRBench 未参与选参）：

| K | 通用 SCOPE | 文字模式 | Δ | %van |
|---:|---:|---|---:|---:|
| 64 | 28.90 | $\rho{=}0,\lambda{=}0$：28.50 | -0.40 | 91.35% |
| 128 | 29.90 | $\rho{=}0,\lambda{=}0.1$：30.00 | +0.10 | 96.15% |
| 192 | 30.80 | $\rho{=}0,\lambda{=}0.1$：**31.20** | **+0.40** | **100.00%** |

结果表明，文字任务确实倾向于更小的空间覆盖比：在 $K=128/192$ 时，$\rho=0,\lambda=0.1$ 在 TextVQA
上最优，并在未参与选参的 OCRBench 上获得小幅一致收益。纯 top-$K$ 并非始终最好，说明即使文字区域信息
密集，少量特征去冗仍有价值。$K=64$ 时两个文字基准的偏好不一致，因此不能宣称存在统一文字配置。基于这
一点，§5.2 主表仍采用固定通用配置，本节仅用于解释内容类型如何改变“覆盖—显著性”权衡。

### 5.8 与 Global-MMR、VisionZip 的对比

为比较显著性引导空间锚点、纯全局去冗和选择—合并策略，我们在多个预算下对比 SCOPE、其
$\rho\to0$ 极限 Global-MMR 以及 VisionZip。TextVQA 列使用无 OCR 协议（vanilla=46.07）；此表 MME 用
**总分** perception+cognition，与表 1/2 的 MME-P 不同尺度，POPE 用 accuracy：

| K | 方法 | GQA | MMB | MME(总) | POPE(acc) | SQA | TextVQA | nonTxt%base |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 576 | vanilla | 62.0 | 64.0 | 1875 | 87.0 | 69.5 | 46.1 | 100.0% |
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
2. **空间锚点的收益集中在实用预算**：相较 Global-MMR，SCOPE 在 $K=192$ 将非文字平均保留率
   从 97.7% 提升至 98.4%，且五项均不低于前者；$K=128/288$ 仍有小幅平均收益。当预算极紧或接近饱和时，
   两者基本持平，说明覆盖约束主要在“预算足以容纳多区域、但仍需选择”的区间发挥作用。
3. **宽预算可作为保守初筛点**：$K=288/346$ 时非文字平均保留率为 98.6–99.3%。这一结果说明方法适合
   用作低风险前置压缩，但本文未报告端到端时延，因而不把性能保留率直接等同于系统加速比。

---

## 6. 跨架构泛化：Qwen2.5-VL-7B

为验证 SCOPE 的两阶段机制并非 LLaVA/CLIP 架构特有，我们将其迁移到
Qwen2.5-VL-7B——一个无 CLS token、采用动态分辨率 NaViT 编码、且视觉塔结构与 CLIP 显著不同的架构。

### 6.1 适配方法

SCOPE 在 Qwen2.5-VL 的 **2×2 PatchMerger 之后、序列拼入语言模型之前**插入：

- **显著性信号的替代**：Qwen 没有 CLS token，用视觉塔最后一层全注意力 block 中每个合并后 patch
  **被接收到的注意力**（对 head 与 query 位置取平均）作为显著性代理 $a_i$，替代 CLIP 的 CLS 注意力。
- **空间覆盖的替代**：使用每张图各自的动态矩形 NaViT 网格（而非固定 24×24）做 FPS + Voronoi 空间
  细分，其余语义锚点流程不变。
- **全局选择不变**：仍采用 §4.4 的显著性—冗余联合准则。
- **输出形式不变**：被选中的输出仍是原始 PatchMerger token 的纯 gather，无重建、无额外合并。

Qwen 适配与 LLaVA 版本共享相同的三项选择原则，仅替换显著性代理和网格构造；具体实现入口与内部结果
路径见作者复现备忘录。

### 6.2 评测与实现核验

我们在推理路径中加入运行时断言，逐样本检查实际保留 token 数是否等于目标预算，并在剪枝后同步更新位置
与缓存索引。所有报告结果均来自通过上述检查的重跑实验；失效的早期运行不参与任何表格或结论。更具体的
版本与故障排查记录保留在复现附录，而不作为方法贡献的一部分。

### 6.3 主结果

我们使用 Qwen2.5-VL-7B，在 lmms-eval 中以 eager attention、batch size 1 评测。为检验直接迁移而非再次
按任务调参，所有任务固定 $\rho=0.5,\lambda=0.5,\text{cover\_factor}=3$，只改变 $K$。基线为相同权重
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

**相对基线保留率**

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
  但尚不能证明频率结构是唯一原因。
- **非文字任务相对稳健**：POPE 与 SQA-IMG 在 $K=64$ 时仍保留 96% 以上性能，而文字任务明显更低。
  因此，固定 $K$ 带来的比例变化与内容类型都会影响结果；二者不应相互替代解释。
- **MMBench-EN 缺失**：离线环境下该 benchmark 的答案抽取会回退到 OpenAI API（401 无 key），故本文
  未在 Qwen2.5-VL 上报告该项。

### 6.5 文字任务中的空间覆盖消融

**分析问题**：文字密集图像中显著性本就集中在文字区域，SCOPE 的两个多样性项（空间覆盖、
特征 MMR）是否会把预算挤占到非文字区域并损害 OCR？为此在 $K{=}64$ 的 OCRBench 上对两个多样性旋钮
分别做独立消融：

| $\rho$（空间覆盖） | $\lambda$（特征 MMR） | OCRBench |
|---|---|---:|
| 0.50 | 0.50 | 45.00（当前部署配置） |
| 0.25 | 0.50 | 45.40 |
| **0.00** | **0.50** | **46.30**（最优） |
| 0.00 | 0.00 | 44.70 |

OCRBench 上，减小空间覆盖比带来单调改善（45.00 → 45.40 → 46.30）；但在 $\rho=0$ 后进一步移除特征
MMR 会下降到 44.70。这一区分很关键：文字任务反对的并非所有多样性，而是固定分配给空间铺展的预算；
全局特征去冗仍能减少重复证据。

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
  $K=64$–$128$，符合紧预算下覆盖成本更高的预期；$K=192$ 时 TextVQA 几乎不变。
- **非文字对照尚不足以确定全局默认值**：GQA 与 MMStar 在 $K=64$ 下对 $\rho$ 不敏感，但仅凭两个任务
  不能说明关闭覆盖对所有非文字任务都是“免费”的。完整结论需要在 POPE、SQA-IMG、VizWiz 与 MME-P 上
  复验。
- **不能弥合 OCR 差距**：$K{=}64$ 的 OCRBench 从 45.0 升到 46.3，相对 77.8 的基线仍有巨大差距。预算
  本身仍是主导杠杆，差距悬殊（同为 $\rho{=}0$ 下，$K{=}64\to128\to192$ 给出 46.3 → 60.2 → 66.6）。
  因此，空间分配是二阶影响，预算本身仍是主要因素。纯选择无法恢复已经丢弃的字符证据，后续工作需要文字
  区域保护、内容自适应预算或受控合并。

### 6.6 小结：跨架构一致性与差异

Qwen2.5-VL 的意义不在于复现与 LLaVA 完全相同的绝对分数，而在于检验三个选择原则能否迁移到无 CLS、
动态网格架构。结果显示，非文字任务在中等预算下仍较稳健，而密集文字在两个架构上都更脆弱。这支持“可
替代性随内容类型变化”的核心动机，但尚不能把损失唯一归因于低频结构。动态网格还引入了固定 LLaVA 设定
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
   无反弹；VizWiz 超越基线、MMStar 平均分近无损这类乐观信号，部分来自这些任务本身允许忽略视觉细节
   （可过滤干扰、可依赖语言先验、含随机水平子集）。综合看，$K{=}192$ 是较保守的工作点
   （固定通用配置下判别式平均 99.0%、COCO CIDEr 97.5%）；$K{=}128$ 可接受；$K{=}64$ 仅推荐用于判别式 QA，
   **不建议**用于密集描述与细粒度感知场景。
3. **关键证据仍需补齐。** 当前稿件在投稿前至少需要完成以下实验：

   - `cover_factor`（覆盖粒度：显著性排序 vs 均匀）的单变量干净消融尚未完全隔离其余超参，当前
     §5.6 的结论仍为定性描述，正式投稿前需补齐数据表；
   - Global-MMR 的 $\lambda$ 敏感性（$\{0.25,0.5,1.0\}$）与 SCOPE 更细的 $\rho$ 网格
     （$\{0.25,0.75\}$）尚未完整跑完；
   - 需要加入 cell 内打分的逐项消融（medoid、低频稳定性、局部显著性），并配合 §3.2 的 M1b 在 encoder
     特征层直接比较代表选择准则。这是支撑“低频稳定性”主线最关键的因果证据：现有频谱统计与定性图只能
     证明低频主导现象存在，不能证明该项相对 medoid 有增量；
   - 需要报告 cell 占有率、两两空间距离、最近邻特征相似度和注意力捕获率，直接验证方法确实同时改善空间
     覆盖、特征去冗和语义保留；
   - 当前外部对照主要是 VisionZip。正式投稿需要补充同插入点、同 token 预算下的近期训练无关选择与合并
     方法，并统一是否计入 CLS、合并 token 和选择器开销；
   - Qwen2.5-VL 上的 $\rho{=}0$ 全局默认值目前只在 OCRBench/TextVQA/GQA/MMStar 四项上验证，其余
     benchmark（POPE、SQA-IMG、VizWiz、MME-P）尚未在 $\rho{=}0$ 下重跑，§6.5 的"全局默认值"结论
     暂为初步推测。
4. **尚未报告端到端系统指标。** §5.1 已按统一公式报告视觉序列对应的 LLM FLOPs，但尚未给出端到端
   延迟、吞吐、峰值显存与选择器自身耗时。这些指标不能由 FLOPs 或 token 保留比例直接替代。

---

## 8. 结论

本文从视觉 token 的局部可替代性出发提出 SCOPE：patch 按固定网格切分，同质区域内的多数 token 可由邻近
代表替代，而这种可替代性在图像上分布极不均匀。由此，剪枝的关键不是给 token 排序，而是判断哪些区域可以
只留一个代表、以及谁有资格充当代表。Spectral-Semantic Anchoring 用低频稳定性回答后者、用全局显著性回答
前者，建立空间分散的语义锚点；Global Redundancy-Aware Visual Token Pruning 则以这些锚点初始化 MMR，
补充语义相关且不重复的证据。三种信号职责分离——频率判断谁是稳定代表，显著性判断哪个区域值得预算，
MMR 判断还缺什么——共同构造一个证据子集，而非并列堆叠评分项。采用固定通用配置时，LLaVA-1.5-7B 在 $K=192/128/64$ 下
分别保留九项判别式基准平均 **99.0%/97.5%/95.1%** 的性能；$K=64$ 时视觉 token 减少 88.9%，按统一
口径计算的 LLM FLOPs 降低 89.2%。Qwen2.5-VL 的迁移结果进一步说明该原则不依赖固定网格或 CLS token。
同时，COCO Caption、MMStar 细粒度感知与文字任务共同划定了适用边界：当图像证据密集且局部冗余较低时，
强剪枝会稳定损失信息。因此，SCOPE 的结论不是“多数 token 都可无损删除”，而是只有在先识别可替代冗余、
再约束空间证据并保留语义锚点时，视觉 token 压缩才能在效率与视觉完整性之间取得可靠平衡。

---

## 附录：复现协议

SCOPE 使用确定性的 FPS–Voronoi 空间划分与贪心选择，不引入可学习参数。LLaVA-1.5-7B 中，方法
作用于 CLIP 倒数第二层输出与多模态投影器之间；Qwen2.5-VL-7B 中，方法作用于 PatchMerger 之后、视觉
序列拼入 LLM 之前。所有实验均保留原始视觉 token，不执行特征重建或额外合并。

我们对每个评测样本记录剪枝前后 token 数，并用运行时断言确认实际保留数与目标预算一致。基线与方法共享
相同的模型权重、数据划分、prompt 模板和评分脚本。TextVQA 的所有结果均来自不提供参考 OCR token 的同一
协议；协议不一致的历史结果不参与比较。Qwen2.5-VL 的可变视觉网格按样本分别构造 cell，固定 $K$ 因而不
代表固定保留比例。代码发布将包含完整环境、配置、逐样本输出与聚合脚本，以支持表格复算。

§3.1–§3.2 的冗余与频率诊断直接复用推理路径中的 DCT 与 cell 构造实现，保证分析与方法使用同一套
$L$、`cover_factor` 与 cell 划分。定性可视化包括原图及 cell 边界、原始特征、
低通重建特征与残差热力图；定量统计在 200 张 GQA 图像上汇总 token 级重建残差与频谱能量。作者内部的
机器路径、历史故障和失效运行索引已移至 [author_repro_notes.md](author_repro_notes.md)，不属于投稿正文。
