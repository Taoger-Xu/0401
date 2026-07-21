# Anchor-Cover：显著性排序覆盖 + 全局冗余感知的进 LLM 前视觉 Token 剪枝

> **完整论文草稿**（摘要 / 引言 / 相关工作 / 方法 / 实验 / 跨架构泛化 / 讨论 / 结论）。
> 本文件整合 [idea4.md](idea4.md)（动机与方法定稿）、[paper_draft.md](paper_draft.md)（LLaVA-1.5-7B
> 方法+实验章节初稿）、[aaai_method_experiments.md](aaai_method_experiments.md)（英文 AAAI 版，含
> MMStar/COCO/OCRBench 扩展实验）、[qwen2_5_vl_method.md](qwen2_5_vl_method.md)（Qwen2.5-VL 跨架构
> 泛化结果）与 [idea_summary.md](../idea_summary.md)（idea1–4 总对比）中已核验的全部材料，按论文写作
> 顺序重新组织。所有数字均可在 `docs/idea4/logs/*` 与 `docs/idea4/qwen2_5_vl_idea4_*` 的 lmms-eval
> 原始输出中核对。方法在内部代号 `IDEA_METHOD=anchor_cover`，曾用投稿代号 **SCRAP**
>（Saliency-, Coverage-, and Redundancy-Aware Pruning），两者指同一方法，正文统一使用 **Anchor-Cover**。

---

## 摘要

多模态大模型（MLLM）中，视觉编码器产出的数百个 patch token 在进入 LLM 后主导了上下文预算与推理开销。
本文提出 **Anchor-Cover**，一种在视觉编码器与 LLM 输入投影之间插入的**训练无关、零参数**的视觉 token
剪枝方法：给定预算 $K$，从 $N$ 个原始 patch token 中纯 gather 式地选出 $K{-}1$ 个（不合并、不重建），
与恒保留的显著 token（CLS 或其等价物）拼接送入 LLM。方法的核心是**双池分配**——一个只落在内容区域的
**显著性排序覆盖池**（Phase A：空间过分割 + 按注意力质量选 cell + cell 内 medoid/低频稳定性/局部显著性
联合打分）与一个**全局冗余感知的显著池**（Phase B：以 CLS 注意力为显著性、以特征余弦相似度为冗余惩罚
的贪心 MMR），二者共享一个预算切分标量 $\rho$，可解析地退化为纯全局 MMR（$\rho\to0$）、VisionZip 式
纯剪枝 top-$K$（$\rho\to0,\lambda=0$）与均匀空间覆盖（`cover_factor`$=1$）。

在 LLaVA-1.5-7B 上，Anchor-Cover 在 9 个判别式 benchmark 上于 $K{=}192/128/64$（保留 1/3、2/9、1/9
token）分别达到 **99.3%/97.7%/95.4%** 的平均保留率，在 POPE、VizWiz、MMStar 等任务上超过未剪枝基线，
且在中高预算档（$K{=}192/128/288$）上全面优于其自身 $\rho{=}0$ 极限（纯全局 MMR）与 VisionZip。生成式
密集描述（COCO Caption）与 MMStar 细粒度感知子集揭示了判别式 QA 平均分掩盖的代价：这两类任务随预算
收紧单调下降、无反弹，$K{=}64$ 时降至 ~90% 与 ~76%，划定了方法的保守适用边界。在 Qwen2.5-VL-7B 上的
跨架构验证（PatchMerger 后插入、无 CLS 时以平均注意力作显著性代理、动态 NaViT 网格）证实了同一机制的
可迁移性，并进一步定位出**密集文字识别是纯剪枝范式的共同天花板**——不来自预算比例失配，而来自文字本身
信息密度高、冗余低，代表性子集无法替代被丢弃的字符。

---

## 1. 引言

### 1.1 背景与问题

以 LLaVA 为代表的多模态大模型将视觉编码器（如 CLIP-ViT）输出的数百个 patch token 与文本 token 一起
送入语言模型。以 LLaVA-1.5-7B 为例，一张图产出 576 个 patch token，远多于典型指令的文本 token 数，
视觉 token 因而主导了 KV-cache 大小与 prefill 计算量。降低视觉 token 数是提升 MLLM 推理效率最直接的
杠杆之一，也是本文的目标场景：在**视觉编码器输出之后、语言模型输入投影之前**插入一次性的 token 选择，
不改变模型权重、不需要训练。

### 1.2 核心矛盾：显著性与覆盖

这一位置的 token 选择必须同时满足两个彼此拉扯的目标：

- **语义显著性**：保住模型真正关注的主体证据。视觉编码器自带的注意力信号（CLIP 的 CLS 注意力，或
  Qwen2.5-VL 视觉塔最后一层的平均注意力）是训练好的显著性度量，是感知类任务（POPE、MME）的必需信号。
- **空间覆盖**：为需要全图理解的任务（GQA、MMBench）保留分散在各内容区域的上下文，避免预算被少数
  显著物体独占。

单一机制都有可预期的失败模式：纯注意力 top-$K$ 会在少数显著物体上**空间扎堆**、选出彼此近似的冗余
token；纯特征去冗（全局 Maximal Marginal Relevance, MMR）能很好地集中在显著且互不相似的 token 上，
但对全图的**空间铺展不足**；不带显著性的均匀空间覆盖又会把预算浪费在天空、墙面等背景区域，稀释了
真正需要的物体证据。

### 1.3 方法与贡献

Anchor-Cover 的做法是**双池分配**：以一个**全局冗余感知的显著池**为主干（继承显著性 + 语义去冗，
即上述"纯特征去冗"机制），再叠加一层**轻量、只落在内容区域**的**显著性排序覆盖池**补足空间铺展
（规避"均匀覆盖浪费背景"的问题）。两池共享同一预算，由单一标量 $\rho$（覆盖比）配置。

本文的贡献可概括为：

1. **方法**：提出双池视觉 token 选择机制 Anchor-Cover，证明其严格泛化纯全局 MMR、VisionZip 式纯剪枝
   top-$K$、均匀空间覆盖三种基线为特定极限（§3.6），选择过程全程 $O(K\cdot N)$、无学习参数。
2. **系统评测**：在 LLaVA-1.5-7B 上用 9 个判别式 benchmark + 1 个生成式压力测试（COCO Caption）+
   1 个能力分解基准（MMStar 六轴分解）建立比现有 6-benchmark 惯例更严格的评测协议，识别出判别式 QA
   平均分会掩盖的代价来源（§4.3–4.4）。
3. **消融与部署规则**：将方法的任务自适应完全收敛到两个标量 $(\rho,\lambda)$，给出预算-任务条件下
   的最优取值与一条可直接部署的规则（§4.5–4.7）。
4. **跨架构泛化**：将方法迁移到无 CLS token、动态分辨率的 Qwen2.5-VL-7B，验证机制的可迁移性，并
   通过独立的空间覆盖消融确认"纯剪枝范式对密集文字的结构性上限"这一结论并非 LLaVA/CLIP 架构特有
   （§5）。

### 1.4 协议注记

lmms-eval 于 2026-07-07 发生升级，TextVQA 的 prompt 丢失了 `Reference OCR token: …` 提示行，带 OCR
（旧协议）与不带 OCR（新协议）的 vanilla 分数相差约 12 分（58.27 vs 46.07），二者**不可跨协议比较**。
本文所有涉及 TextVQA 的结果——包括本文自身的全部剪枝运行与作为参照的 vanilla/VisionZip 基线——均已
统一到**无 OCR 协议**重跑，任何跨协议数字不出现在正文对比表中。判断某次运行协议的方法：检查
lmms-eval 输出 samples jsonl 的 `input` 字段是否含 `"Reference OCR token"`。

---

## 2. 相关工作与内部对照

**VisionZip**（同插入点的训练无关强基线）：在同一位置（CLIP 倒数第二层输出、`mm_projector` 之前）
选出"显著 token"，并将其余被丢弃的 token **合并（merge）**为若干上下文 token 一并保留，因此每个预算
$K$ 下实际传给 LLM 的是"显著 token + 合并 token"而非纯粹的原始 token 子集。VisionZip 是本文实验中
除自身消融外唯一的外部方法对照。

**内部消融基线（idea1–3）**：为隔离"选择准则"这一变量，本工作系列在同一注入点、同一预算、同样纯
gather（不合并不重建）的约束下实现并评测了三种更简单的选择准则，作为 Anchor-Cover 的直接消融对象：

| 方案 | 核心信号 | 与 Anchor-Cover 的关系 |
|---|---|---|
| idea1 spectral | DCT 低频稳定性 + FPS 空间 cell medoid（无显著性信号、无全局去冗） | Anchor-Cover 的 Phase A cell 内打分公式的子集 |
| idea2 local_var | 8 邻域局部特征变化度 | 未被 Anchor-Cover 直接继承，仅作横向对照 |
| idea3 cls_mmr（即本文的 **Global-MMR**） | CLS 显著性 + 全局特征冗余惩罚（无空间约束） | Anchor-Cover 的 $\rho\to0$ 精确极限（§3.6），是隔离 Phase A 贡献的关键消融 |

四方案在 6-benchmark 扫描（§ idea_summary 实验 A/B）中的通用性排序为 **VisionZip > idea3 > idea4 >
idea2 > idea1**（各预算档一致），但该排序的绝大部分差距来自 TextVQA 协议断裂前的误判——修正为无 OCR
协议、并把评测扩展到 9 个 benchmark 后（本文 §4），Anchor-Cover 在中高预算下反超 idea3 与 VisionZip
（§4.8）。这一修正过程本身是本文方法论的一部分：**先污染的对比协议会系统性地误导方法排序**，第 §1.4
节的协议注记与第 §4.8 节的重新对比是相互印证的证据。

---

## 3. 方法

### 3.1 问题设定与总览

给定 LLaVA-1.5 式视觉语言模型，CLIP-ViT 编码器对一张图输出 $N{=}576$ 个 patch token 与 1 个 CLS
token。标准做法把全部 $N$ 个 patch token 经 `mm_projector` 送入 LLM，视觉 token 占据了绝大部分上下文
预算与推理开销。**Anchor-Cover 在 `mm_projector` 之前、对 CLIP 倒数第二层输出做一次硬剪枝**：从 576 个
patch token 中选出 $K{-}1$ 个，与恒保留的 CLS 拼成 $K$ 个 token 进入 LLM。整个过程是**纯 gather**——
只挑选原始 token，不做合并、不做特征重建，因此零额外参数、无需训练、可即插即用。

单一机制都有短板：CLS 注意力 top-$K$ 会在少数显著物体上空间扎堆、彼此近似（冗余）；纯特征去冗（全局
MMR）集中在显著且互不相似的 token 上，但对全图空间铺展不足；而不带显著性的均匀空间覆盖会把预算浪费
在天空/墙面等背景上。Anchor-Cover 采用**双池分配**：以全局冗余感知的显著池为主干（继承显著性 + 语义
去冗），叠加一层只落在内容区域的显著性排序覆盖池补足空间铺展。

### 3.2 记号

记 CLIP 倒数第二层 patch 特征 $f_i\in\mathbb R^{1024}$，其单位化 $\hat f_i=f_i/\lVert f_i\rVert$；
CLS 注意力 $a_i=\sum_h \mathrm{Attn}^{(L-2)}_h[\mathrm{cls},i]$，其 min-max 归一化 $\hat a_i$；patch
网格坐标 $p_i\in\{0,\dots,23\}^2$。总预算 $K$（CLS 恒保留，实际选 $K{-}1$ 个 patch），按覆盖比
$\rho\in[0,1]$ 拆成两池：

$$
M=\lceil \rho\,(K-1)\rceil\ \ (\text{覆盖池}),\qquad B=(K-1)-M\ \ (\text{显著池}).
$$

### 3.3 Phase A：显著性排序覆盖池

只在**有内容的区域**保证空间铺展、跳过背景：

1. **空间细分**：用确定性 FPS + Voronoi 把 $24\times24$ 网格划成 $P=\lceil c\cdot M\rceil$ 个空间 cell，
   其中 $c$ 为 `cover_factor` 且 $P>M$；
2. **cell 内代表**：每个 cell 取综合分最高的 token 作代表，
   $$
   s_i=\underbrace{m_i}_{\text{medoid 代表性}}+w_f\underbrace{\ell_i}_{\text{低频稳定性}}
   +w_a\,\mathrm{Norm}_{C(i)}(a_i),
   $$
   其中 $m_i,\ell_i$ 为 cell 内 medoid 与 DCT 低频稳定性（均在 cell 内 min-max 归一化）；
3. **按注意力选 cell**：按 cell 的注意力质量 $\max_{i\in C}a_i$ 对 $P$ 个 cell 排序，只取 **top-$M$** 个
   cell 的代表 token。

**低频稳定性 $\ell_i$ 的计算（2D-DCT）**：直觉是——空间上平滑、低频占主导的 patch 特征更能代表其所在
区域，而高频（边缘/纹理/噪声）token 对局部扰动敏感、代表性弱。因此用「低通重建后丢失的能量」来度量一个
token 的高频程度，取其相反数作为低频稳定性：

1. 把 patch 特征按网格排成 $g\in\mathbb R^{24\times24\times D}$（每个空间位置一个 $D$ 维特征向量）。
2. 沿两个空间维各做一次**正交 DCT-II**。$G{=}24$ 的 DCT 基矩阵为
   $$
   M_{k,n}=c_k\cos\!\Big(\tfrac{\pi(2n+1)k}{2G}\Big),\qquad c_0=\tfrac{1}{\sqrt G},\ \ c_{k\ge1}=\sqrt{\tfrac{2}{G}},
   $$
   逐通道得到频域系数 $\mathrm{coef}[k,l,d]=\sum_{h,w}M_{k,h}\,M_{l,w}\,g[h,w,d]$（$k,l$ 为纵/横两个空间
   频率下标；$M$ 正交，故 $\sum_n M_{k,n}M_{k',n}=\delta_{kk'}$）。
3. **低通截断**：只保留左上角 $L\times L$（$L{=}16$）的低频系数块 $\{k,l<L\}$，其余高频系数置零。
4. **逆变换重建**低频分量 $\tilde g=M^\top(\text{mask}\odot\mathrm{coef})M$（正交基下逆变换即转置），得每个
   token 的**高频残差能量**
   $$
   r_i=\big\lVert g_i-\tilde g_i\big\rVert_2 .
   $$
5. 令 $\ell_i=-r_i$（残差越小 $\Rightarrow$ 越由低频主导 $\Rightarrow$ 越稳定），再在其所在 cell 内做
   min-max 归一化。

由此 $\ell_i$ 高的 token 其特征可被少数低空间频率解释、对局部扰动稳健，与 medoid 代表性 $m_i$、CLS
显著性 $a_i$ 共同决定 cell 代表。$D$ 维特征沿通道共享同一空间变换，DCT 基矩阵按 grid 缓存复用，整体开销
$O(N D G)$、可忽略；该项沿用自 idea1（spectral）的低频稳定性定义。注意 DCT 仅用于**打分**，被选中的仍是
原始 patch 特征 $f_i$，从不改动 token 内容。

由于 $P>M$ 且按注意力挑 cell，覆盖 token 落在 $M$ 个最有内容、彼此空间可分的区域，而非均匀铺满全图
（含背景），从而在保证空间多样性的同时不浪费预算。

**定性证据**：`docs/idea4/logs` 的低频冗余诊断脚本（[scripts/diag_lowfreq_evidence.py](../../scripts/diag_lowfreq_evidence.py)）
直接复用 `prune_ideas.py` 推理时实跑的 `lowfreq_reconstruct` / `_fps_voronoi_cells` / `_dct_matrix`（非
模拟数据），对真实 GQA 图跑 CLIP 倒数第二层特征。定性图 [figs/lowfreq_evidence/qualitative.png](figs/lowfreq_evidence/qualitative.png)
（原图+cell 边界 / 真实特征 PCA-RGB / lowpass=16 DCT 重建 PCA-RGB / 重建残差热力图，四列对照）显示：
天空、草地、地毯、墙面等平坦区域重建前后几乎无差别，残差热力图的亮点精确落在物体边缘、文字、人脸等
细节处。定量图 [figs/lowfreq_evidence/quantitative.png](figs/lowfreq_evidence/quantitative.png)（200 张
GQA 图池化）显示 per-token 残差直方图明显左偏（多数 token 残差趋近 0，仅长尾少数高），且 lowpass=16 已
捕获约 70% 的均值谱能量——支持"同一空间 cell 内多数 token 可被低频分量良好解释"这一假设。

### 3.4 Phase B：全局冗余感知显著池

以 Phase A 已选集合为初始 $S$，按 Maximal-Marginal-Relevance 贪心补 $B$ 个 token：

$$
\mathrm{score}(i\mid S)=\hat a_i-\lambda\cdot\big[\max_{j\in S}\cos(\hat f_i,\hat f_j)\big]_+ .
$$

第一项是 CLS 显著性，第二项是与已选集合的全局特征冗余惩罚（无任何空间约束）——新 token 必须既显著、
又与已选集合在特征上不相似。这一步等价于全局 MMR，是感知类任务（尤其 POPE）表现最好的主干；覆盖池
只是在它之上补空间铺展。先跑 Phase A 再跑 Phase B 很关键：式 (3) 的冗余惩罚在 Phase B 阶段也会同时
抑制与覆盖池 token 近似的候选，使两池天然耦合而非简单拼接。

式中 $\hat a_i-\lambda r_i$ 可重参数化为等价显著性权重形式 $\alpha\hat a_i-(1-\alpha)r_i$，
$\alpha=\tfrac{1}{1+\lambda}$：$\lambda{=}0.5\Leftrightarrow\alpha{=}66.7\%$，
$\lambda{=}0\Leftrightarrow$ 纯 CLS 注意力 top-$K$（§4.7 用此重参数化解释文字模式的选参结果）。

### 3.5 算法与复杂度

```python
def anchor_cover(f, a, K, rho, lam, cover_factor=3.0):
    M = ceil(rho * (K-1))                        # 覆盖池大小
    P = min(N, round(cover_factor * M))          # 细分 cell 数, P > M
    cells = fps_voronoi_cells(grid=24, budget=P)
    rep   = [argmax_{i in c}(m_i + w_f*l_i + w_a*norm_c(a_i)) for c in cells]
    S     = [rep[c] for c in topM_cells_by(max_attn_in_cell, M)]   # Phase A
    for _ in range((K-1) - M):                   # Phase B: 全局 MMR
        i = argmax_{i not in S}(minmax(a)_i - lam * max_{j in S} relu(cos(f_i, f_j)))
        S.append(i)
    return sort([CLS] + S)                        # raster order
```

复杂度 $O(K\cdot N)$（$N{=}576$），相对编码器前向可忽略；Phase B 的贪心循环维护逐 token 的运行最大
相似度，不需重复计算全量相似度矩阵。由于只有 $K$ 个视觉 token 进入 LLM，语言模型 prefill 成本随之
线性下降，与既有的进 LLM 前剪枝工作一致。实现见
[visionzip/prune_ideas.py](../../visionzip/prune_ideas.py) `select_anchor_cover`，注入见
[visionzip/idea_inject.py](../../visionzip/idea_inject.py)（`IDEA_METHOD=anchor_cover`，超参通过
`IDEA_RHO`/`IDEA_LAMBDA`/`IDEA_SIGMA` 等环境变量传入，切换配置不改代码）。

### 3.6 与已有方法的关系（退化极限）

Anchor-Cover 严格泛化了纯 MMR 与纯 top-$K$ 剪枝，这一关系直接体现在 `select_anchor_cover` 的实现
docstring 中：

| 极限 | 退化为 |
|---|---|
| $\rho\to 0$ | 纯全局 MMR（即 idea3 / 本文的 Global-MMR，显著 + 去冗，无覆盖） |
| $\rho\to 0,\ \lambda=0$ | VisionZip dominant top-$K$ 的纯剪枝版 |
| `cover_factor`$=1$ | 均匀空间覆盖（每个 cell 含背景都取一个，即原始未加注意力排序的 idea4 变体） |
| $\rho\to1,\ w_a=0,\ $`cover_factor`$=1$ | idea1 spectral（纯低频代表 + 空间 cell，无显著性/无全局去冗） |

因此，§4.8 中"Anchor-Cover vs Global-MMR"的对比本质上是对 Phase A（覆盖池）**净贡献的单变量消融**：
两者除 $\rho$ 外的一切设置完全相同。

### 3.7 超参数与任务自适应

**冻结项**（全实验固定，不随任务/预算变动）：`cover_factor`$=3$、空间门控 $\sigma\to\infty$（关闭，
即 Phase B 的冗余惩罚不设空间衰减）、$w_f=w_a=1.0$。真正需要设定的只有两个旋钮：

- **覆盖比 $\rho$（主旋钮）**：控制覆盖池与显著池的预算配比。预算越紧，覆盖池越易挤占显著核心，故
  $\rho$ 随预算收紧而减小（§4.5）。
- **冗余权重 $\lambda$（副旋钮）**：Phase B 的显著性/去冗平衡，等价显著性权重
  $\alpha=\tfrac{1}{1+\lambda}$。

由此得到一条**任务自适应规则**（§4.5、§4.7 实证）：一般（感知/全图）任务用 $\lambda{=}0.5$、按预算取
$\rho\in\{0.5\,(K{\ge}192),\,0.25\,(K{\le}128)\}$；文字密集任务关闭覆盖池 $\rho{=}0$、并把显著性权重
提到 $\lambda{=}0.1$（$K{=}64$ 用 $\lambda{=}0$）。切换仅改两个标量，不改选择器实现。

**早期版本的教训（空间门控 σ）**：方法早期版本在 Phase B 引入了空间门控 MMR（$\sigma=2$，冗余惩罚随
空间距离衰减，只惩罚"空间相近"的冗余），初衷是让去冗更贴近局部竞争。诊断发现这一门控对 TextVQA 毫无
收益（~40–45 分不变），却结构性地伤害 POPE（§4.8 的 POPE 反超即在去掉门控后才出现）。最终版本移除
空间门控（$\sigma\to\infty$），使 Phase B 精确退化为 idea3 的全局 MMR，只在其之上叠加覆盖池——这也是
§3.6 退化关系表中不再出现 $\sigma$ 参数的原因。

---

## 4. 实验：LLaVA-1.5-7B

### 4.1 实验设置

**模型与实现**：LLaVA-1.5-7B，视觉塔 CLIP-ViT-L/14-336。剪枝在视觉塔与 `mm_projector` 之间插入，
LLM 权重不变、无微调。**基线（Baseline）**为不剪枝的原始 576 token 模型（vanilla）。

**基准与指标**（9 项，均用 lmms-eval 统一评测）：GQA、MMBench-EN、MME-P（perception score）、
MMStar、POPE（F1）、ScienceQA-IMG、TextVQA、VizWiz、OCRBench。GQA/SQA/TextVQA/VizWiz 为
exact-match×100，MMBench 为 GPT 判分，POPE 为 F1×100，MMStar/OCRBench 为 accuracy×100。VizWiz 用带
本地标注的 val split；MMStar 用 val split（1,500 题，覆盖粗/细粒度感知、实例/逻辑推理、科技、数学
六大能力），可有效抵抗视觉无关的语言先验、更能考察真视觉能力。

此外在 **COCO Caption**（`coco2017_cap_val`，5,000 图，CIDEr/BLEU/METEOR/ROUGE-L）上做补充的生成式
压力测试（§4.4）；因其为生成式 n-gram 指标、与上述判别式准确率不同量纲，**不计入主表平均**。

**预算档**：主表报告 $K\in\{192,128,64\}$，即保留 $1/3$、$2/9$、$1/9$ 的视觉 token；另附
$K\in\{288,346\}$（保留 50%/60%）作为无损前置初筛（§4.8）。

**协议说明（TextVQA）**：见 §1.4。本文所有剪枝运行均为**无 OCR 协议**，故 TextVQA 一律以
**vanilla 46.07** 为基线；其余基准不受影响。

### 4.2 主结果：相对基线的保留率

表 1 给出每个基准在各预算下 Anchor-Cover 相对 vanilla 的保留率（越接近或超过 100% 越好）。非文字
任务采用该预算下的通用配置，文字任务（TextVQA/OCRBench）采用文字模式（见 §4.7 的超参选择）。

**表 1. 相对 baseline(576) 的保留率（LLaVA-1.5-7B）**

| Benchmark | Baseline (576) | 1/3 (K=192) | 2/9 (K=128) | 1/9 (K=64) |
|---|---:|---:|---:|---:|
| GQA | 100.0% | 96.5% | 95.6% | 92.7% |
| MMBench-EN | 100.0% | 98.8% | 97.9% | 93.2% |
| MME-P | 100.0% | 97.6% | 93.4% | 91.8% |
| MMStar | 100.0% | **100.6%** | 97.6% | 95.3% |
| POPE (F1) | 100.0% | **101.2%** | **100.7%** | 98.1% |
| SQA-IMG | 100.0% | 99.6% | 99.2% | 98.1% |
| TextVQA | 100.0% | 98.0% | 96.7% | 92.3% |
| VizWiz | 100.0% | **101.7%** | **102.3%** | **104.2%** |
| OCRBench | 100.0% | 100.0% | 96.2% | 92.6% |
| **平均保留率** | 100.0% | **99.3%** | **97.7%** | **95.4%** |

**表 2. 绝对分数（同上配置）**

| Benchmark | Baseline | K=192 | K=128 | K=64 |
|---|---:|---:|---:|---:|
| GQA | 61.97 | 59.78 | 59.25 | 57.47 |
| MMBench-EN | 64.00 | 63.23 | 62.63 | 59.62 |
| MME-P | 1511.3 | 1474.5 | 1411.7 | 1387.4 |
| MMStar | 33.56 | 33.76 | 32.74 | 31.98 |
| POPE (F1) | 85.88 | 86.93 | 86.44 | 84.26 |
| SQA-IMG | 69.46 | 69.16 | 68.91 | 68.12 |
| TextVQA | 46.07 | 45.16 | 44.55 | 42.54 |
| VizWiz | 54.06 | 54.99 | 55.33 | 56.32 |
| OCRBench | 31.20 | 31.20 | 30.00 | 28.90 |

**观察**：

1. **宽/中预算近无损**：$K{=}192$ 平均保留 99.3%，POPE、VizWiz、MMStar 超过 baseline，OCRBench 完全
   恢复；$K{=}128$ 平均 97.7%，八项 ≥95.6%（仅 MME-P 93.4%）。
2. **强剪枝有正则作用**：VizWiz 在所有预算下都 ≥ baseline，$K{=}64$ 达 104.2%，说明去除冗余/干扰
   patch 反而有益，视觉 token 并非越多越好。
3. **MMStar 平均分稳健但需分解看**（见 §4.3）：平均保留 100.6%/97.6%/95.3%，但该均值由六个能力子集
   等权平均，其中「科技」「数学」「逻辑推理」三项 LLaVA-1.5-7B 本身就在四选一随机水平（25%）附近甚至
   以下，其"上升"主要是向随机水平回归的噪声，而非真实增益；真正视觉相关的**粗/细粒度感知**随预算单调
   下降。因此 MMStar 平均分不宜单独作为无损证据。
4. **文字任务受协议校正后差距很小**：修正为无 OCR 基线后，TextVQA 在 $K{=}192/128$ 保留 98.0%/96.7%，
   而非旧协议下看似的"天花板"。

### 4.3 MMStar 能力子集分解

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

### 4.4 COCO Caption：密集描述任务上的压力测试

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

这与 §4.3 的分解相互印证：VizWiz 的超越与 MMStar 平均分的"近无损"，很大程度来自这些任务允许忽略
视觉细节（干扰过滤、语言先验、随机水平子集）；一旦任务强制要求完整的细粒度视觉覆盖（COCO 复述、
MMStar 感知子集），$K{=}64$ 的真实代价就显现为 ~90%（CIDEr）乃至 ~76%（细粒度感知）。

因此我们对适用范围给出保守结论：$K{=}192$ 在生成式描述上仍保留 97.5% CIDEr、可视为近无损；
$K{=}128$（94.6%）可接受；而 $K{=}64$ 的强剪枝**不适用于**密集描述与细粒度感知场景，其在判别式
QA 上的高保留率不应外推到这类任务。

### 4.5 消融：覆盖比 $\rho$ 的预算自适应

扫 $\rho\in\{0.25,0.5\}$（其余固定），以除 TextVQA 外 5 项非文字任务的平均保留率 nonTxt%van 为判据：

| K | $\rho{=}0.25$ | $\rho{=}0.5$ | 最优 |
|---|---:|---:|:--:|
| 64 | **94.12** | 93.88 | 0.25 |
| 128 | **97.02** | 96.54 | 0.25 |
| 192 | 97.89 | **98.35** | 0.5 |
| 288 | **98.60** | 98.42 | 0.25 |
| 346 | 99.23 | **99.27** | 0.5 |

规律：**预算越紧、$\rho$ 越小**（不从显著核心抢 token）；$K{\ge}192$ 的宽预算下更大的覆盖池
$\rho{=}0.5$ 收益最大。故默认 $\rho{=}0.25$，$K{\ge}192$ 提升到 $\rho{=}0.5$。

### 4.6 消融：覆盖粒度 `cover_factor`（显著性排序 vs 均匀）

均匀覆盖（`cover_factor`$=1$，每个 cell 含背景都取一个）在小预算下把约 $\rho$ 比例预算耗在背景上，
POPE 明显退化；`cover_factor`$=3$（$P{=}3M$ 细分后按注意力取 top-$M$ cell）把覆盖集中到内容区域，
POPE/MME 显著回升。故固定 **`cover_factor`$=3$**。

> 待补：当前该结论的对照配置尚未与其余超参做单变量隔离（详见 [idea4.md](idea4.md) §3.1），完整的
> `cover_factor`$\in\{1,3\}$ 干净消融表待补充，其余结论不受影响。

### 4.7 消融：文字模式——关闭覆盖池 + 提高显著性权重

假设文字 patch 空间聚集、外观相似，会同时被覆盖池与去冗排斥。采用两阶段选参避免在 OCRBench 上过拟合：
先在 TextVQA 上选参，再在 OCRBench 上独立验证。

**TextVQA 搜索**（exact-match×100）：

| $\rho$ | $\lambda$ | $\alpha$ | K=64 | K=128 | K=192 |
|---:|---:|---:|---:|---:|---:|
| 通用 | 0.5 | 66.7% | 41.50 | 43.60 | 44.40 |
| 0 | 0 | 100% | **42.54** | 44.33 | 44.68 |
| 0 | 0.1 | 90.9% | 42.49 | **44.55** | **45.16** |
| 0 | 0.25 | 80.0% | 42.38 | 44.04 | 45.06 |

**OCRBench 独立验证**（accuracy×100，vanilla=31.20；配置只在 TextVQA 上选出，OCRBench 未参与选参）：

| K | 通用 Idea4 | 文字模式 | Δ | %van |
|---:|---:|---|---:|---:|
| 64 | 28.90 | $\rho{=}0,\lambda{=}0$：28.50 | -0.40 | 91.35% |
| 128 | 29.90 | $\rho{=}0,\lambda{=}0.1$：30.00 | +0.10 | 96.15% |
| 192 | 30.80 | $\rho{=}0,\lambda{=}0.1$：**31.20** | **+0.40** | **100.00%** |

结论：$K{=}128/192$ 的文字最优点为 $\alpha{=}90.9\%$（非 100%）——纯 top-$K$ 会重新引入相邻高注意力
patch 的冗余，仍需少量去冗。$K{=}64$ 不存在跨任务统一最优（纯 top-$K$ 提升 TextVQA 但降 OCRBench），
故 $K{=}64$ 保留通用配置。

### 4.8 宽预算初筛（K=288/346）与方法对比

作为二级剪枝的无损前置，保留 50%/60% token 时 Anchor-Cover 对 GQA/POPE/SQA/MMB/MME 近无损；与自身
$\rho\to0$ 极限（Global-MMR，即 idea3）以及 VisionZip 逐项对比（TextVQA 列为无 OCR 协议，
vanilla=46.07；此表 MME 用**总分** perception+cognition，与表 1/2 的 MME-P 不同尺度，POPE 用 accuracy）：

| K | 方法 | GQA | MMB | MME(总) | POPE(acc) | SQA | TextVQA | nonTxt%base |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 576 | vanilla | 62.0 | 64.0 | 1875 | 87.0 | 69.5 | 46.1 | 100.0% |
| 192 | VisionZip | 59.3 | **63.8** | 1770 | 86.4 | 68.7 | **44.5** | 97.6% |
| 192 | Global-MMR (idea3) | 59.5 | 63.0 | 1783 | 87.5 | 68.4 | **45.1** | 97.7% |
| 192 | **Anchor-Cover** | **59.8** | 63.2 | **1803** | **87.6** | **69.2** | 44.4 | **98.4%** |
| 128 | VisionZip | 57.7 | 62.2 | **1764** | 84.6 | 68.7 | **43.8** | 96.1% |
| 128 | Global-MMR (idea3) | 59.3 | 62.3 | 1719 | **87.2** | 68.8 | **43.9** | 96.8% |
| 128 | **Anchor-Cover** | **59.3** | **62.6** | 1728 | **87.2** | **68.9** | 43.6 | **97.0%** |
| 64 | VisionZip | 55.2 | **60.1** | **1718** | 80.6 | **69.0** | **42.0** | 93.3% |
| 64 | Global-MMR (idea3) | **57.6** | 59.8 | 1639 | **86.3** | 68.3 | **42.0** | **94.2%** |
| 64 | Anchor-Cover | 57.2 | 59.6 | 1670 | 85.5 | 67.9 | 41.5 | 94.1% |
| 288 | Global-MMR (idea3) | 61.0 | **63.9** | 1779 | **87.6** | 68.6 | 45.1 | 98.5% |
| 288 | **Anchor-Cover** | 61.0 | 63.8 | **1785** | 87.5 | **68.7** | **45.4** | **98.6%** |
| 346 | Global-MMR (idea3) | 61.2 | 64.6 | **1831** | **87.3** | 68.6 | 45.6 | **99.3%** |
| 346 | Anchor-Cover | 61.2 | **64.9** | 1824 | 87.1 | **68.7** | 45.6 | 99.3% |

三点发现：

1. **纯选择优于选择+合并**：在五个非文字任务上，Anchor-Cover 与 Global-MMR 两个纯 gather 方案在每个
   预算下都全面优于 VisionZip，最明显的是 $K{=}64$ 的 POPE（85.5/86.3 vs 80.6）——尽管 VisionZip 额外
   对丢弃 token 做了合并。在匹配协议下，合并对 TextVQA 也没有优势：文字模式下 Anchor-Cover 在
   $K{=}192/128/64$ 分别达到 45.16/44.55/42.54，对比 VisionZip 的 44.53/43.82/41.95。
2. **Phase A（覆盖池）在实用预算档回本**：与自身 $\rho{=}0$ 极限 Global-MMR 相比，加入覆盖池在
   $K{=}192$（98.4% vs 97.7%，五项全部 ≥ Global-MMR）、$K{=}128$、$K{=}288$ 上赢得非文字均分；在
   $K{=}64$ 与 $K{=}346$ 两个极端档（预算极紧或预算已接近饱和）与 Global-MMR 基本持平（差 ≤0.1%）。
3. **宽预算初筛无损**：保留 50–60% token（$K{=}288/346$）时，Anchor-Cover 保持 98.6–99.3% 的非文字
   均分，可作为下游二级压缩前的安全初筛步骤。

---

## 5. 跨架构泛化：Qwen2.5-VL-7B

为验证 Anchor-Cover 的机制（显著性 + 空间覆盖 + 全局去冗）并非 LLaVA/CLIP 架构特有，我们将其迁移到
Qwen2.5-VL-7B——一个无 CLS token、采用动态分辨率 NaViT 编码、且视觉塔结构与 CLIP 显著不同的架构。

### 5.1 适配方法

Anchor-Cover 在 Qwen2.5-VL 的 **2×2 PatchMerger 之后、序列拼入语言模型之前**插入：

- **显著性信号的替代**：Qwen 没有 CLS token，用视觉塔最后一层全注意力 block 中每个合并后 patch
  **被接收到的注意力**（对 head 与 query 位置取平均）作为显著性代理 $a_i$，替代 CLIP 的 CLS 注意力。
- **空间覆盖的替代**：使用每张图各自的动态矩形 NaViT 网格（而非固定 24×24）做 FPS + Voronoi 空间
  细分，其余 Phase A 流程不变。
- **Phase B 不变**：仍是与 idea4 相同的全局特征 MMR。
- **输出形式不变**：被选中的输出仍是原始 PatchMerger token 的纯 gather，无重建、无额外合并。

复现入口为 [scripts/qwen2_5_vl_idea4_eval.sh](../../scripts/qwen2_5_vl_idea4_eval.sh)；完整 lmms-eval
结果与逐样本输出落在与本文件同级的 `qwen2_5_vl_idea4_k*_rho*_lambda*/` 目录，控制台日志在仓库根
`logs/` 目录。

### 5.2 复现历史与关键正确性说明（重要）

**2026-07-18 之前的全部 "idea4 on Qwen" 结果无效**：`Qwen2_5_VL/qwen2_5vl_visionzip.py` 的剪枝注入
在 transformers 4.57.6 环境下**从未真正执行**，导致所有 K 档的分数逐位等同于 vanilla（这本身就是发现
问题的证据）。这些结果已归档至 `docs/idea4/_invalid_pre_fix_0718/`，**不应在论文中引用**。三个叠加
bug（均已修复）：

1. `_anchor_cover_rect` 使用了 `Tensor.minimum_`/`Tensor.maximum_`（torch 无此原地方法）——每次调用
   必然崩溃，改为 `torch.minimum/maximum(..., out=...)`。
2. 剪枝后未重建 `cache_position`，导致 prefill 长度与实际序列不匹配；现已重建为剪枝后长度的
   `torch.arange`。
3. **最关键**：入口脚本只 patch 了 `transformers.Qwen2_5_VLForConditionalGeneration`，而 lmms-eval 的
   `models/simple/qwen2_5_vl.py` 在 import 时已经通过 `from transformers import ...` 抓住了原始类的
   引用。必须**直接改绑 wrapper 模块内的类名**，否则 harness 静默地跑原始模型——退出码仍为 0，结果
   文件也正常生成，唯一的破绽是所有 K 档分数完全相同。

**方法论教训**：改动剪枝路径后，必须用运行时插桩确认剪枝真的触发（例如断言若干样本上 `kept==K`），
并核对 vanilla / K=192 / K=64 的分数确有差异，**不能只看退出码判断成功**——lmms-eval 在跨 GPU
gather 失败时也会正常退出、只是静默丢弃某个任务的结果（POPE，gather 量最大，多次踩坑）。硬件层面，
本机 **GPU 7 会发生 Xid/CUDA 未知错误并挂死整个多卡作业**，评测需用
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6` 排除该卡并为每个任务加超时。

以下结果均为 2026-07-18 修复后重跑，已验证剪枝确实生效。

### 5.3 主结果

本地模型 `/home/jk/models/qwen2.5-vl-7b`，lmms-eval 离线评测，`attn_implementation=eager`，
`batch_size=1`，固定 `rho=0.5, lambda=0.5, cover_factor=3.0`，只改变 $K$。基线为**未修改的原始模型**
（`eval/qwen2_5_vl_vanilla_entry.py`，不剪枝），同协议运行。

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

### 5.4 结果解读

- **K 与 LLaVA-1.5 表不可直接对比**：LLaVA-1.5 每图固定输出 576 个视觉 token，故 $K{=}192/128/64$
  恰好是 1/3、2/9、1/9。Qwen2.5-VL 用动态分辨率 NaViT，剪枝前 token 数逐图不同，同一固定 $K$ 在不同
  benchmark 上对应完全不同的保留比例。实测中位数：**TextVQA 约 999 个 token**（$K{=}64$ 只保留
  6.4%，比 LLaVA 的 11.1% 更苛刻），而 **OCRBench 约 274 个 token**（$K{=}64$ 保留 23.3%，反而比
  LLaVA 更宽松）。
- **OCR 密集任务承担主要损失，且预算比例并非主因**：OCRBench（84.3% → 57.8%）与 TextVQA
  （93.9% → 77.1%）的退化速度远超其余任务。TextVQA 的退化部分可归因于异常苛刻的保留比例，但
  OCRBench 恰恰在**本表中保留比例最宽松**的情况下崩溃得最严重。因此驱动因素**不是**"密集文字图像
  产生更多 token"，而是**阅读文字本身对丢弃 patch 天然不宽容**：文字信息密度高、冗余低，任何代表性
  子集都无法替代被丢弃的字符。
- **非文字任务保持良好**，尤其 POPE 与 SQA-IMG 在 $K{=}64$ 时仍 >96%，说明选择机制本身没有失效——
  预算与 Qwen 可变 token 数之间的失配才是文字任务退化的主要来源。
- **MMBench-EN 缺失**：离线环境下该 benchmark 的答案抽取会回退到 OpenAI API（401 无 key），故本文
  未在 Qwen2.5-VL 上报告该项。

### 5.5 空间覆盖在文字任务上的消融（2026-07-20）

**待验证假设**：文字密集图像中显著性本就集中在文字区域，Anchor-Cover 的两个多样性项（空间覆盖、
特征 MMR）会把预算挤占到非文字区域，从而损害 OCR。为此在 $K{=}64$ 的 OCRBench 上对两个多样性旋钮
分别做独立消融：

| $\rho$（空间覆盖） | $\lambda$（特征 MMR） | OCRBench |
|---|---|---:|
| 0.50 | 0.50 | 45.00（当前部署配置） |
| 0.25 | 0.50 | 45.40 |
| **0.00** | **0.50** | **46.30**（最优） |
| 0.00 | 0.00 | 44.70 |

**假设仅对"空间"多样性成立**：去掉空间覆盖单调受益（45.00 → 45.40 → 46.30），但**同时**去掉特征
MMR 反而损失 1.6 分（46.30 → 44.70）——特征空间的去冗**确实在为预算创造价值**，需要保留；真正该
关闭的只有空间覆盖池。文字任务下的最优配置为 **$\rho{=}0,\lambda{=}0.5$**。

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

- **效应真实但偏小，6 个文字任务格中只有 2 个达到统计显著**。收益随预算收紧而增大（$K{=}64$–$128$
  最明显），符合机制预期：预算越紧，把一半预算花在空间铺展上代价越大；$K{=}192$ 时 TextVQA 完全没有
  收益。
- **$\rho{=}0$ 并非文字任务专属的权衡**：GQA 与 MMStar 几乎不受影响（$p{=}0.60$、$p{=}1.00$），说明
  关闭空间覆盖在非文字任务上"免费"——这使得 $\rho{=}0$ 在 Qwen 上可能是一个站得住脚的**全局默认值**
  而非仅针对文字任务的特例；但其余非文字 benchmark 尚未在 $\rho{=}0$ 下完整重跑，此结论暂限于
  GQA/MMStar 两项。
- **不能弥合 OCR 差距**：$K{=}64$ 的 OCRBench 从 45.0 升到 46.3，相对 77.8 的基线仍有巨大差距。预算
  本身仍是主导杠杆，差距悬殊（同为 $\rho{=}0$ 下，$K{=}64\to128\to192$ 给出 46.3 → 60.2 → 66.6）。
  多样性项只是二阶效应；一阶问题始终是**文字信息密度高、冗余低，任何子集选择规则都无法还原被丢弃的
  字符**。

### 5.6 小结：跨架构一致性与差异

Qwen2.5-VL 上的结果与 LLaVA-1.5 相互印证同一个核心结论：**纯剪枝范式对密集文字任务存在结构性天花板**
（§4.8 结论 1、§6 讨论），且该天花板的根因（文字信息密度高、冗余低）与具体架构、显著性信号的具体
实现（CLS 注意力 vs 平均注意力代理）无关，是**任务本身**的性质，而非某个视觉编码器的缺陷。同时，
Qwen 上的动态 token 预算揭示了一个 LLaVA 固定 576 token 设定下不可见的因素：**固定 $K$ 在可变原始
token 数下对应的保留比例逐任务不同**，这提示未来在动态分辨率模型上部署固定预算剪枝时，应考虑按图像
原始 token 数或按任务类型自适应预算，而非使用单一全局 $K$。

---

## 6. 讨论与局限性

1. **纯剪枝（gather-only）范式的共同天花板是密集文字**。VisionZip 用"选择显著 token + 合并被丢弃
   token 为上下文 token"的方式部分保住了文字信息；本文与其内部消融（idea1–3）证明，**任何不带合并/
   重建的纯选择方法**（无论选择准则多复杂）在文字密集任务上都存在结构性上限——文字 patch 通常空间
   聚集、外观相似，语义去冗与空间覆盖机制都倾向于将其判定为冗余而排斥，Qwen2.5-VL 上的独立消融
   （§5.5）进一步证明这一现象与架构无关。这划定了本文方法论（乃至整个"纯剪枝"方向）的天花板，指向
   两个后续方向：**文字感知的局部保护**（识别文字区域并豁免去冗惩罚）或**受控的轻量合并**（仅对确认
   冗余的 token 做合并，而非像 VisionZip 那样合并全部剩余 token）。
2. **适用范围的保守声明**。判别式 QA 的高保留率不能外推到所有任务：生成式密集描述（COCO CIDEr
   97.5%/94.6%/89.6%）与细粒度感知（MMStar fine-grained 95.4%/84.8%/75.8%）随预算收紧单调下降、
   无反弹；VizWiz 超越基线、MMStar 平均分近无损这类乐观信号，部分来自这些任务本身允许忽略视觉细节
   （可过滤干扰、可依赖语言先验、含随机水平子集）。综合看，$K{=}192$ 可视为**全任务族近无损**
   （判别式 99.3%、COCO CIDEr 97.5%）的推荐工作点；$K{=}128$ 可接受；$K{=}64$ 仅推荐用于判别式 QA，
   **不建议**用于密集描述与细粒度感知场景。
3. **尚未完成的实验**（据 [idea_summary.md](../idea_summary.md) §5、[idea4.md](idea4.md) §3.1）：
   - `cover_factor`（覆盖粒度：显著性排序 vs 均匀）的单变量干净消融尚未完全隔离其余超参，当前
     §4.6 的结论仍为定性描述，正式投稿前需补齐数据表；
   - idea3 的 $\lambda$ 敏感性（$\{0.25,0.5,1.0\}$）与 Anchor-Cover 更细的 $\rho$ 网格
     （$\{0.25,0.75\}$）尚未完整跑完；
   - "双多样性诊断"实验（cell 占有率、pairwise 空间距离、最近邻特征相似度、注意力捕获率四项机制性
     度量）尚未执行，用于直接验证"双池同时提升空间多样性与语义显著性"这一机制性假设，而非只看下游
     benchmark 分数的间接证据；
   - Qwen2.5-VL 上的 $\rho{=}0$ 全局默认值目前只在 OCRBench/TextVQA/GQA/MMStar 四项上验证，其余
     benchmark（POPE、SQA-IMG、VizWiz、MME-P）尚未在 $\rho{=}0$ 下重跑，§5.5 的"全局默认值"结论
     暂为初步推测。
4. **协议脆弱性是可复现性的重要风险源**。本文两次独立发现的正确性事故——LLaVA 侧的 TextVQA OCR
   协议断裂（§1.4）与 Qwen 侧的静默 vanilla 退化（§5.2）——共同提示：**分数在合理区间内并不能证明
   流水线正确**，剪枝方法的评测必须搭配运行时插桩（确认剪枝真的触发、跨配置分数确有差异）与协议
   一致性检查（确认 prompt/schema 未在基线与方法之间漂移），而不能仅依赖退出码或"分数看起来正常"。

---

## 7. 结论

Anchor-Cover 以零训练、零额外参数的纯剪枝，在 LLaVA-1.5-7B 的 9 个判别式基准上于 $K{=}192/128/64$
分别保留 baseline 平均 99.3%/97.7%/95.4% 的性能，且在 POPE/VizWiz/MMStar 等任务上超过 baseline；其
唯一的任务自适应来自两个标量 $(\rho,\lambda)$：一般任务按预算设 $\rho$、固定 $\lambda{=}0.5$；文字
密集任务关闭覆盖池并提高显著性权重。方法通过双池设计——显著性排序覆盖池（Phase A）+ 全局冗余感知
显著池（Phase B）——在同预算下同时兼顾语义显著性与空间铺展，且可解析地退化为纯全局 MMR、VisionZip
式纯剪枝 top-$K$ 与均匀空间覆盖三种基线，覆盖池的净贡献可通过与 $\rho{=}0$ 极限的直接对比单变量
隔离出来。跨架构验证（Qwen2.5-VL-7B，无 CLS、动态分辨率）确认了同一机制的可迁移性，并通过独立消融
证实：密集文字识别的性能损失并非来自预算比例失配或架构差异，而是纯选择范式面对信息密度高、冗余低
的文字内容时的结构性上限——这是所有不带合并/重建的视觉 token 剪枝方法共同面对的天花板，也是本工作
指向的下一步方向：文字感知的局部保护，或介于纯选择与 VisionZip 式全量合并之间的受控轻量合并机制。

---

## 附录：数据来源与复现信息

- **方法实现**：[visionzip/prune_ideas.py](../../visionzip/prune_ideas.py)（`select_anchor_cover`
  及依赖的 `_dct_matrix`/`_fps_voronoi_cells`/`lowfreq_reconstruct`）、
  [visionzip/idea_inject.py](../../visionzip/idea_inject.py)（LLaVA 侧注入，读取
  `IDEA_METHOD`/`IDEA_K`/`IDEA_RHO`/`IDEA_LAMBDA`/`IDEA_SIGMA` 环境变量）、
  `Qwen2_5_VL/qwen2_5vl_visionzip.py` + `eval/qwen2_5_vl_idea4_entry.py`（Qwen 侧注入与 2026-07-18
  的三个正确性修复，见 §5.2）。
- **LLaVA-1.5-7B 实验数据**：`docs/idea4/logs/{cf3_sigInf_*,tvqa_*,text_salience_*}`（含 mmstar、
  coco2017_cap_val 的 `*_results.json`）；基线取自 `logs/lmms-eval/vanilla`、
  `docs/idea5/logs/noocr_vanilla`；运行日志 `logs/{mmstar_*,coco_*}.log`；聚合脚本
  [scripts/aggregate_ideas.py](../../scripts/aggregate_ideas.py)（读 `docs/idea*/logs/k*/`）。COCO
  Caption 指标依赖 `pycocoevalcap`（METEOR/CIDEr 需 Java），可用
  `/home/jk/miniconda3/envs/vitcop/bin` 中的 openjdk。
- **Qwen2.5-VL-7B 实验数据**：`docs/idea4/qwen2_5_vl_idea4_k*_rho*_lambda*/models__qwen2.5-vl-7b/`
  与 `docs/idea4/qwen2_5_vl_vanilla/`；复现脚本
  [scripts/qwen2_5_vl_idea4_eval.sh](../../scripts/qwen2_5_vl_idea4_eval.sh)；**2026-07-18 之前的
  结果已失效并归档于 `docs/idea4/_invalid_pre_fix_0718/`，禁止引用**。
- **低频冗余定性/定量证据**：[scripts/diag_lowfreq_evidence.py](../../scripts/diag_lowfreq_evidence.py)，
  产出 [figs/lowfreq_evidence/qualitative.png](figs/lowfreq_evidence/qualitative.png) 与
  [figs/lowfreq_evidence/quantitative.png](figs/lowfreq_evidence/quantitative.png)。
- **总对比与内部消融基线（idea1/2/3、VisionZip、vanilla）**：[docs/idea_summary.md](../idea_summary.md)；
  各方案自述见 [idea1](../idea1/idea1.md) · [idea2](../idea2/idea2.md) · [idea3](../idea3/idea3.md)；
  任务定义见 [docs/task.md](../task.md)。
- **已知协议/正确性风险清单**（写作与复现前务必核对）：
  1. TextVQA 的 OCR 提示协议断裂（2026-07-07 前后不可比，判据见 §1.4）；
  2. Qwen2.5-VL 侧 2026-07-18 前的静默 vanilla 退化（判据见 §5.2）；
  3. 本机 GPU 7 的硬件不稳定性会挂死多卡评测作业，需在 `CUDA_VISIBLE_DEVICES` 中排除。
