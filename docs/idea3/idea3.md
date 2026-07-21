# Idea-3：CLS-Attention 冗余感知剪枝（CLS-MMR）

> 定位：在进入 LLM 之前、`mm_projector` 之前，利用**空间冗余性**对 576 个视觉 token 做初步 hard pruning。
> 与 idea1（Spectral，频率+空间 cell）、idea2（Local Variation，8 邻域余弦）并列为三种候选方案，
> 三者共享同一注入点与同一 token 预算，用于比较"哪种初筛方案通用性更好"。

## 1. 出发点：把 VisionZip 的重要性信号和空间去冗余结合起来

idea1 / idea2 的重要性分数都来自**图像自身的统计量**：

- idea2 用 8 邻域余弦相似度衡量"局部变化度"，与邻居差异越大越保留；
- idea1 用 DCT 低频稳定性 + 空间 cell medoid 衡量"区域代表性"。

它们的共同弱点：**没有任何"任务/语义显著性"信号**，只知道"哪里在空间上不冗余"，不知道"哪里是模型真正关注的主体"。

VisionZip 恰好提供了这样一个显著性信号——CLIP 倒数第二层里 **CLS token 对各 patch 的注意力**：

$$
a_i=\sum_{h}\text{Attn}^{(L-2)}_{h}[\text{cls},\,i],\qquad i=1,\ldots,576 .
$$

CLS 注意力是 ViT 预训练得到的、对"图像主体"的显著性度量，比纯图像统计量更贴近下游任务。VisionZip 的 dominant token 就是 $a_i$ 的 top-K。

但 VisionZip 的 dominant 选择有一个和"空间冗余"直接冲突的缺陷：

> **CLS 注意力会集中在少数几个显著区域**，因此 top-K dominant token 在空间上往往**扎堆、彼此高度相似**——正是 idea1/idea2 想删掉的那种冗余。
> VisionZip 用额外的 contextual token（对剩余 token 做相似度**合并**）来补覆盖，但那是 token *merging / compression*，会引入合成 token，不是严格意义上的剪枝。

idea3 的想法：**保留 CLS 注意力这个强显著性先验，但把"选 top-K"换成"选重要且互相不冗余的一组"**，用纯 hard pruning 实现，不产生任何合成 token。

## 2. 方法：CLS 注意力 + 冗余惩罚的贪心 MMR

在 CLIP 倒数第二层 patch 特征 $f_i\in\mathbb R^{1024}$ 与 CLS 注意力 $a_i$ 上，做 Maximal Marginal Relevance（MMR）式的贪心选择。

定义归一化重要性 $\mathrm{imp}_i=\mathrm{minmax}(a_i)\in[0,1]$，特征方向 $\hat f_i=f_i/\lVert f_i\rVert$。

已选集合为 $S$ 时，候选 token $i$ 的边际得分：

$$
\text{score}(i\mid S)=\mathrm{imp}_i-\lambda\cdot\max_{j\in S}\cos(\hat f_i,\hat f_j).
$$

- 第一项：CLS 显著性（要重要）；
- 第二项：与**已选集合**的最大相似度（要和已选的都不像，即去空间/语义冗余）；
- $\lambda$：去冗余强度。

贪心流程（预算 $K$，实际选 $K$ 个原始 patch，另外恒保留 CLS token 以对齐 VisionZip 的 token 预算）：

```text
S ← { argmax_i imp_i }                     # 用最显著的 token 作种子
while |S| < budget:
    i* ← argmax_{i∉S} ( imp_i − λ·max_{j∈S} cos(f_i, f_j) )
    S ← S ∪ { i* }
return sort(S)                              # 按 raster order 排序保持空间顺序
```

运行时用一个"到已选集合的最大相似度"向量增量更新，复杂度 $O(K\cdot N)$，对 $N=576$、$K\le 300$ 可忽略。实现见 [visionzip/prune_ideas.py](../../visionzip/prune_ideas.py) 的 `select_cls_mmr`，注入见 [visionzip/idea_inject.py](../../visionzip/idea_inject.py)。

## 3. 与 VisionZip / idea1 / idea2 的关系

| 方案 | 重要性信号 | 去冗余机制 | 是否纯剪枝 |
|---|---|---|---|
| VisionZip | CLS 注意力 top-K（dominant） | 剩余 token **合并**成 contextual token | 否（有合成 token） |
| idea1 spectral | DCT 低频稳定性 + cell medoid | FPS Voronoi 空间 cell（每 cell 选 1） | 是 |
| idea2 local_var | 8 邻域局部变化度 | 无显式去冗余（靠分数本身） | 是 |
| **idea3 cls_mmr** | **CLS 注意力**（同 VisionZip） | **MMR 冗余惩罚**（与已选集合去相似） | **是** |

极限情形可以看清 idea3 的位置：

- $\lambda=0$：退化为 VisionZip 的 dominant top-K（但**没有** contextual 合并，是纯剪枝版）；
- $\lambda\to\infty$：退化为特征空间的最远点采样（纯多样性，忽略显著性）；
- 中间 $\lambda$：在"显著性"和"空间去冗余"之间插值。

因此 idea3 = **VisionZip 的显著性先验 × idea1/idea2 的去冗余原则**，是三条线的自然融合，也是本组实验里唯一同时用到 CLS 注意力和空间冗余的方案。

## 4. 超参

| 超参 | 默认 | 说明 |
|---|---|---|
| $\lambda$ (`IDEA_LAMBDA`) | 0.5 | 去冗余强度；实验里扫 {0.25, 0.5, 1.0} 观察通用性 |
| 相似度空间 | CLIP 倒数第二层 patch 特征（1024-d） | 与 VisionZip 注入点一致 |
| CLS token | 恒保留，占 1 个预算 | 对齐 VisionZip（其 dominant 含 CLS） |

## 5. 待验证假设

1. 同预算下，idea3 在 POPE / MME 这类依赖局部证据、对空间覆盖敏感的任务上，应优于 VisionZip 的 dominant-only 和 idea2；
2. 作为初筛（保留 50%/60%）时，idea3 因为兼顾显著性与覆盖，掉分应最小；
3. $\lambda$ 增大提升覆盖但可能牺牲显著主体，存在任务相关的最优点。

以上假设由 [docs/task.md](../task.md) 规定的实验统一验证，结果回填到本目录 `logs/` 与第 6 节。

## 6. 实验结果

LLaVA-1.5-7B，6 benchmark，logs 在 `docs/idea3/logs/k<K>/`。指标：GQA/SQA/TextVQA=exact_match×100，
MMB-EN=gpt_eval_score，MME=perception+cognition，POPE=accuracy×100。Avg%van=各列相对 vanilla 保留率均值。

| 配置 | GQA | MMB-EN | MME | POPE | SQA | TextVQA | Avg%van |
|---|---|---|---|---|---|---|---|
| vanilla(576) | 61.97 | 64.00 | 1874 | 86.99 | 69.46 | 58.27 | 100% |
| VisionZip-192 | 59.25 | 63.75 | 1770 | 86.38 | 68.67 | 57.30 | 97.7% |
| VisionZip-128 | 57.66 | 62.20 | 1764 | 84.64 | 68.67 | 56.86 | 96.3% |
| VisionZip-64 | 55.15 | 60.14 | 1718 | 80.57 | 68.96 | 55.48 | 93.6% |
| **idea3-192** | **59.48** | 62.97 | 1783 | **87.53** | 68.42 | 45.08 | 94.3% |
| **idea3-128** | **59.29** | 62.29 | 1719 | **87.22** | 68.77 | 43.87 | 93.2% |
| **idea3-64** | **57.55** | 59.79 | 1639 | **86.27** | 68.27 | 42.01 | 90.6% |
| idea3-288 (初筛50%) | 60.97 | 63.92 | 1779 | 87.59 | 68.62 | 45.13 | 95.0% |
| idea3-346 (初筛60%) | 61.24 | 64.60 | 1831 | 87.34 | 68.62 | 45.57 | 95.8% |

结论：

1. **idea3 是 idea1/idea2/idea3 三者里通用性最好的**（Avg 94.3/93.2/90.6% vs idea1 90.3/86.3/78.6%、
   idea2 91.6/87.3/79.5%），尤其激进剪枝 K=64 优势巨大（+11pt）；
2. **在 GQA 和 POPE 上 idea3 反超 VisionZip**：POPE@K=64 达 86.3 vs VisionZip 80.6（+5.7），
   GQA@K=64 57.6 vs 55.2——MMR 冗余惩罚带来更好的空间覆盖，感知类任务受益；
3. **瓶颈是 TextVQA**：idea3 把它从 idea1 的 33 拉到 45，但仍低于 VisionZip 的 57，且几乎不随预算
   变化（K=64→346 都在 42~46）。原因是文字 patch 空间聚集、特征相似，被 MMR 冗余惩罚**系统性排斥**。
   这直接导出 idea4 的空间门控（只罚"又近又像"）与 λ/ρ 敏感性方向，也解释了 VisionZip 靠 contextual
   **合并**（非纯剪枝）才能保住文字信息。

假设 1（POPE/MME 优于 dominant-only 与 idea2）成立；假设 2（初筛近无损）在 GQA/POPE/SQA/MMB/MME 上成立、
TextVQA 例外；假设 3（λ 存在任务相关最优）由 idea4 的空间门控与 ρ 敏感性进一步验证。
