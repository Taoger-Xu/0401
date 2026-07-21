# Idea-1：局部变化度（Local Variation）Token 初筛

## 1. Motivation

### 1.1 视觉 Token 的空间冗余现象

LLaVA-1.5-7B 的 MM Projector 将 CLIP 的 24×24=576 个 patch 特征映射为 4096 维向量。Exp-M3 的实测数据揭示，这 576 个 token **在空间上存在显著冗余**：

**跨图像的平均局部余弦相似度**（8-邻域，Layer 0）：

| 图像 | 内容 | 平均局部余弦相似度 |
|------|------|:---:|
| n195925（室内场景）| 桌椅、书籍 | 0.607 |
| n272313（户外场景）| 滑板、人物 | 0.583 |
| 典型图像均值 | — | ~0.586 |

> 相似度 = 0.58 意味着：平均每个 token 的方向与其 8 个邻居高度一致——**超过一半的 token 携带的信息已被邻居覆盖**。

**跨 LLM 层的空间平滑演化**（Exp-M3 cross-layer 实验）：

```
Layer  0 : local_sim_mean = 0.586  (MM Projector 输出，进入 LLM 前)
Layer 20 : local_sim_mean = 0.560  (U 形谷底)
Layer 31 : local_sim_mean = 0.722  (LLM 最终输出，相邻 token 趋于一致)
```

LLM 的深层表征更加平滑，说明相邻 token 在语义上愈发相似——空间冗余在 LLM 内部被进一步放大。

**DCT 能量集中性**（MM Projector 输出层面）：

```
r=12 的低频 DCT 系数（12×12=144 个，占全部 576 的 25%）
可捕获特征总能量的约 39.7%
```

仅用 25% 的低频系数就能覆盖近 40% 的信息，印证了视觉 token 的空间低频主导性。

### 1.2 从空间冗余到 Token 删除

上述观察引出一个自然假设：

> **如果 token A 的方向与其所有邻居高度一致（局部余弦相似度高），则 A 携带的信息可被邻居代表，删除 A 对 LLM 的损失较小。**

反之，**局部变化度高**（与邻居不相似）的 token 携带独特信息，是更重要的保留候选。

这与 CLSE/PAST 在 LLM 内部基于跨层频谱变化剪枝的逻辑互补：
- **Local Variation（本方法）**：利用 MM Projector 输出的**空间结构**，在进入 LLM 之前做 Stage-0 初筛
- **CLSE/PAST**：利用 LLM 内部跨层的**频谱变化**，在推理过程中做精确剪枝

---

## 2. 方案设计

### 2.1 评分公式

对 MM Projector 输出 $\mathbf{F} \in \mathbb{R}^{H \times W \times D}$（$H=W=24$，$D=4096$），计算每个 token 的**局部变化度分数**：

$$
\text{sim}_{ij} = \frac{1}{|\mathcal{N}_{ij}|} \sum_{(i', j') \in \mathcal{N}_{ij}} \cos(\mathbf{f}_{ij},\ \mathbf{f}_{i'j'})
$$

$$
s_{ij} = 1 - \text{sim}_{ij}
$$

其中 $\mathcal{N}_{ij}$ 为 $(i,j)$ 的 8-连通邻域，边界位置使用 edge padding（用自身边缘值填充）。

**重要性 $s_{ij}$ 越高**，说明该 token 与邻居差异越大，信息越独特，越应保留。

选出 top-$K$ 个原始 token 子集直接输入 LLM：

$$
\text{selected} = \mathbf{F}_{\text{flat}}\left[\text{argsort}(s_{\text{flat}})[-K:]\right]
$$

**保留原始 token 数值**，不做重建，无分布偏移。

### 2.2 代码实现

**向量化 NumPy 实现**（推理时评分）：

```python
def score_local_var(feat_2d: np.ndarray) -> np.ndarray:
    """
    feat_2d : [H, W, D]  MM Projector 输出
    返回    : [H*W]      局部变化度分数（越高越应保留）
    """
    H, W, D = feat_2d.shape
    flat   = feat_2d.reshape(-1, D).astype(np.float32)
    norms  = np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-8
    grid_n = (flat / norms).reshape(H, W, D)          # L2-normalized

    # edge padding：保证所有 token 都有 8 个邻居
    pad = np.pad(grid_n, ((1, 1), (1, 1), (0, 0)), mode="edge")

    sim_sum = np.zeros((H, W), dtype=np.float32)
    for di in range(3):
        for dj in range(3):
            if di == 1 and dj == 1:
                continue
            sim_sum += (grid_n * pad[di:di+H, dj:dj+W, :]).sum(-1)

    local_sim = sim_sum / 8.0
    return (1.0 - local_sim).flatten()
```

> 注：本方案在实验代码里对应 `IDEA_METHOD=local_var`，评分在 CLIP 倒数第二层 patch 特征上计算，
> 输出 CLS + top-(K−1) 个原始 patch，与 idea1/idea3/idea4 同注入点、同预算。实现见
> [visionzip/prune_ideas.py](../../visionzip/prune_ideas.py) 的 `score_local_variation`。

## 3. 实验结果

LLaVA-1.5-7B，6 benchmark，logs 在 `docs/idea2/logs/k<K>/`。

| 配置 | GQA | MMB-EN | MME | POPE | SQA | TextVQA | Avg%van |
|---|---|---|---|---|---|---|---|
| vanilla(576) | 61.97 | 64.00 | 1874 | 86.99 | 69.46 | 58.27 | 100% |
| VisionZip-128 | 57.66 | 62.20 | 1764 | 84.64 | 68.67 | 56.86 | 96.3% |
| idea2-192 | 58.16 | 62.71 | 1718 | 83.96 | 69.16 | 40.72 | 91.6% |
| idea2-128 | 55.96 | 60.05 | 1612 | 81.27 | 67.72 | 36.74 | 87.3% |
| idea2-64 | 52.62 | 53.09 | 1447 | 75.00 | 66.29 | 29.37 | 79.5% |
| idea2-288 (初筛50%) | 60.30 | 63.92 | 1752 | 86.02 | 68.27 | 43.85 | 93.8% |
| idea2-346 (初筛60%) | 60.84 | 64.35 | 1787 | 86.44 | 68.52 | 45.01 | 94.9% |

结论：局部变化度是廉价的纯图像统计信号，无任务/显著性先验。

1. 在 GQA/SQA/MMBench 上尚可，但**在 TextVQA 上崩溃**（K=64 仅 29.4，K=192 也只有 40.7 vs
   VisionZip 57）——偏好边缘/纹理、无显式空间覆盖，会把聚集的小文字区域采样掉；
2. 通用性全面弱于 idea3（cls_mmr）和 idea4（anchor_cover）：缺显著性信号 + 缺覆盖硬保证；
3. 作为**高预算初筛**（K=288/346）尚可用：Avg 93.8/94.9%，但 TextVQA 仍封顶 ~45，不能称无损。

对比见 [docs/idea_summary.md](../idea_summary.md)。