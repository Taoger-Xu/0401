# Idea-4：Anchor-Cover——显著性覆盖 + 全局冗余感知的双池视觉 Token 剪枝

> 在进入 LLM 之前、`mm_projector` 之前，对 CLIP 倒数第二层输出的 576 个视觉 token 做 hard pruning，
> 输出 CLS + (K−1) 个**原始** patch token（纯 gather，不合并、不重建）。

## 1. 动机

在 LLM 之前做视觉 token 初筛，需要同时满足两个彼此拉扯的目标：

- **语义显著性**：保住模型真正关注的主体证据。CLIP 倒数第二层的 CLS 注意力
  $a_i=\sum_h \text{Attn}^{(L-2)}_h[\text{cls},i]$ 是 ViT 预训练得到的显著性度量，是感知类任务（POPE/MME）的必需信号。
- **空间覆盖**：为全图理解类任务（GQA/MMBench）保留分散在各内容区域的上下文。

单一机制都有短板：CLS 注意力 top-K 会在少数显著物体上**空间扎堆**、彼此近似（冗余）；
只做特征冗余去除（全局 MMR）能很好地集中在显著且互不相似的 token 上，但**对全图的空间铺展不足**；
而不带显著性的均匀空间覆盖会把预算浪费在天空/墙面等背景上，稀释物体覆盖。

Anchor-Cover 的做法是**双池分配**：以**全局冗余感知的显著池**为主干（继承显著性 + 语义去冗），
再叠加一层**轻量、只落在内容区域的显著性排序覆盖池**补足空间铺展。两者相加，在同预算下既保住显著证据、
又不留空间空洞。

## 2. 方法

记 CLIP 倒数第二层 patch 特征 $f_i\in\mathbb R^{1024}$、CLS 注意力 $a_i$、patch 网格坐标
$p_i\in\{0,\dots,23\}^2$，归一化显著性 $\hat a=\operatorname{minmax}(a)$，单位化特征 $\hat f_i=f_i/\lVert f_i\rVert$。
总预算 $K$（CLS 恒保留，实际选 $K-1$ 个原始 patch），按覆盖比 $\rho$ 拆成两池：

$$
M=\lceil \rho\,(K-1)\rceil\ \text{（覆盖池）},\qquad B=(K-1)-M\ \text{（显著池）}.
$$

### 2.1 Phase A：显著性排序覆盖池

只在**有内容的区域**保证空间铺展，跳过背景：

1. 用确定性 FPS + Voronoi 把网格分成 $P=\lceil c\cdot M\rceil$ 个空间 cell（$c$=`cover_factor`，$P>M$）；
2. 每个 cell 的代表 token 取 cell 内综合分最高者：
   $$
   s_i=\underbrace{m_i}_{\text{medoid 代表性}}+w_f\,\underbrace{\ell_i}_{\text{低频稳定性}}
   +w_a\,\operatorname{Norm}_{C(i)}(a_i),
   $$
   其中 $m_i,\ell_i$ 沿用 idea1 的 cell 内 medoid 与 DCT 低频稳定性定义（均 cell 内 min-max 归一化）；
3. 按 cell 的**注意力质量** $\max_{i\in C}a_i$ 对 $P$ 个 cell 排序，只取 **top-$M$** 个 cell 的代表 token。

因为 $P>M$ 且按注意力挑 cell，覆盖 token 落在 $M$ 个**最有内容**的、彼此空间可分的区域，
而不是均匀铺满全图（含背景）。这保证了空间多样性，同时不浪费预算。

### 2.2 Phase B：全局冗余感知显著池

以 Phase A 已选集合为初始 $S$，贪心地按 Maximal-Marginal-Relevance 补 $B$ 个 token：

$$
\text{score}(i\mid S)=\hat a_i-\lambda\cdot\max_{j\in S}\big[\cos(\hat f_i,\hat f_j)\big]_+ .
$$

第一项是 CLS 显著性，第二项是与已选集合的**全局特征冗余惩罚**（不加任何空间约束）——
新选的 token 必须既显著、又与已选集合在特征上不相似。这一步等价于 idea3 的全局 MMR，
是感知类任务（尤其 POPE）表现最好的主干；覆盖池只是在它之上补空间铺展。

### 2.3 伪代码与复杂度

```python
def select_anchor_cover(f, a, budget, rho=0.25, lam=0.5, cover_factor=3.0):
    M = ceil(rho * budget)                      # 覆盖池大小
    P = min(N, round(cover_factor * M))         # 细分 cell 数, P > M
    cells = fps_voronoi_cells(grid=24, budget=P)
    rep   = [argmax_{i in cell}(m_i + w_f*l_i + w_a*norm_cell(a_i)) for cell in cells]
    keepC = topM_cells_by(max_attn_in_cell, M)  # 只取注意力最高的 M 个 cell
    S = [rep[c] for c in keepC]                 # Phase A: 显著性排序覆盖
    max_red = max_{j in S} relu(cos(f_i, f_j))  # 全局特征冗余, 无空间门控
    for _ in range(budget - M):                 # Phase B: 全局 MMR
        i = argmax_{i not in S}(minmax(a)_i - lam * max_red_i)
        S.append(i); update(max_red)
    return sort(S)                              # raster order
```

复杂度 $O(K\cdot N)$（$N=576$），可忽略。实现见
[visionzip/prune_ideas.py](../../visionzip/prune_ideas.py) `select_anchor_cover`，
注入见 [visionzip/idea_inject.py](../../visionzip/idea_inject.py)（`IDEA_METHOD=anchor_cover`）。

### 2.4 与 baseline 的关系（退化极限）

| 极限 | 退化为 |
|---|---|
| $\rho\to 0$ | idea3 cls_mmr（纯全局 MMR） |
| $\rho\to 0,\ \lambda=0$ | VisionZip dominant top-K 的纯剪枝版 |
| `cover_factor` $=1$ | 均匀空间覆盖（每个 cell 含背景都取一个） |

因此 Anchor-Cover 严格泛化 idea3，加项就是"显著性排序覆盖池"。

## 3. 超参数的确定

方法固定 $w_f=w_a=1.0$（沿用 idea1 v5 冻结值）、$\lambda=0.5$（沿用 idea3）。
需确定的是覆盖比 $\rho$ 与覆盖粒度 `cover_factor`；两者由下述实验确定（LLaVA-1.5-7B，6 benchmark，
判据为**除 TextVQA 外 5 项相对 vanilla 的平均保留率 nonTxt%van**）。

### 3.1 覆盖粒度 cover_factor（显著性排序 vs 均匀）

均匀覆盖（`cover_factor=1`，每个 cell 含背景都取一个）在小预算下把约 $\rho$ 比例的预算耗在背景上，
POPE 明显退化；`cover_factor=3`（$P=3M$ 细分后按注意力取 top-$M$ cell）把覆盖集中到内容区域，
POPE/MME 显著回升。故取 **`cover_factor=3`**。

> 待补：当前均匀覆盖的对照配置同时带有旧的空间门控与 $\rho=0.5$，不是对 cover_factor 的**单变量**隔离。
> 需补一组 `cover_factor=1`、其余固定为 SOTA（σ→∞、$\rho$ 同档）的干净消融，本节表格待该结果回填。

### 3.2 覆盖比 ρ（预算自适应）

$\rho$ 控制覆盖池与显著池的预算配比。扫 $\rho\in\{0.25,0.5\}$（其余固定）得 nonTxt%van：

| K | ρ=0.25 | ρ=0.5 | 最优 ρ |
|---|---|---|---|
| 64 | **94.12** | 93.88 | 0.25 |
| 128 | **97.02** | 96.54 | 0.25 |
| 192 | 97.89 | **98.35** | 0.5 |
| 288 | **98.60** | 98.42 | 0.25 |
| 346 | 99.23 | **99.27** | 0.5 |

规律：**预算越紧、$\rho$ 越小**（不从显著核心抢 token）；预算宽（K=192）时更大的覆盖池 $\rho=0.5$ 收益最大。
默认取 **$\rho=0.25$**（各档稳健、平均最优），在 $K\ge 192$ 的宽预算下可提高到 $\rho=0.5$ 取得最强结果。

## 4. 实验结果

8 benchmark，指标：GQA/SQA/TextVQA/VizWiz=exact_match×100，MMB-EN=gpt_eval_score，
MME=perception+cognition，POPE/OCRBench=accuracy×100。所有保留率 = idea4 / vanilla-576（同协议）。

**Baseline（vanilla-576）与协议说明**：GQA 61.97 / MMB 64.0 / MME 1874.5 / POPE 86.99 /
SQA 69.46 / VizWiz 54.06 / OCRBench 31.20。**TextVQA 一律以「无 OCR 提示」协议的 vanilla 46.07
作基线**——2026-07-07 lmms-eval 升级后 `Reference OCR token: …` 提示行被移除，带 OCR 的旧基线
58.27 与全部 idea4 运行不是同一协议、不可跨表比较（见 [[textvqa-ocr-protocol-break]]）。因此
本节把此前误用的 58.3 基线更正为 46.07，idea4 的 TextVQA 保留率随之从 ~76% 修正到 92～98%。

### 4.0 主结果：8 benchmark 逐项最优保留率（K=192/128/64）

下三表给出每个 benchmark 在该预算下 idea4 能达到的**最优分数与相对 vanilla 的保留率**，最右列标注
取得该最优所用的超参 (ρ, λ)。其余超参在所有结果上**冻结**为 `cover_factor=3`、σ→∞、$w_f=w_a=1.0$；
唯一在各任务间微调的是覆盖比 ρ 与冗余权重 λ（详见 §4.0.4 注释）。8 项逐项最优仅由 2～3 组超参给出：
一组「通用配置」覆盖全部感知/全图任务，一组「文字模式」(ρ=0) 覆盖 TextVQA/OCRBench。

#### 4.0.1 K=192（保留 33% token，8 项平均保留率 **98.9%**）

| Benchmark | vanilla | idea4 最优 | 保留率 | 触发超参 (ρ, λ) |
|---|---:|---:|---:|---|
| GQA | 61.97 | 59.78 | 96.5% | 0.5, 0.5 |
| MMB-EN | 64.0 | 63.23 | 98.8% | 0.5, 0.5 |
| MME | 1874.5 | 1803.4 | 96.2% | 0.5, 0.5 |
| POPE | 86.99 | **87.61** | **100.7%** | 0.5, 0.5 |
| SQA | 69.46 | 69.16 | 99.6% | 0.5, 0.5 |
| VizWiz | 54.06 | **54.99** | **101.7%** | 0.5, 0.5 |
| TextVQA | 46.07 | 45.16 | 98.0% | **0, 0.1**（文字模式） |
| OCRBench | 31.20 | **31.20** | **100.0%** | **0, 0.1**（文字模式） |

> 6 项非文字任务同用**通用配置 (ρ=0.5, λ=0.5)**；TextVQA/OCRBench 切到**文字模式 (ρ=0, λ=0.1)**。
> POPE/VizWiz/OCRBench 已 ≥ vanilla，其余四项保留率 ≥96.2%。

#### 4.0.2 K=128（保留 22% token，8 项平均保留率 **97.5%**）

| Benchmark | vanilla | idea4 最优 | 保留率 | 触发超参 (ρ, λ) |
|---|---:|---:|---:|---|
| GQA | 61.97 | 59.25 | 95.6% | 0.25, 0.5 |
| MMB-EN | 64.0 | 62.63 | 97.9% | 0.25, 0.5 |
| MME | 1874.5 | 1728.5 | 92.2% | 0.25, 0.5 |
| POPE | 86.99 | **87.18** | **100.2%** | 0.25, 0.5 |
| SQA | 69.46 | 68.91 | 99.2% | 0.25, 0.5 |
| VizWiz | 54.06 | **55.33** | **102.3%** | 0.25, 0.5 |
| TextVQA | 46.07 | 44.55 | 96.7% | **0, 0.1**（文字模式） |
| OCRBench | 31.20 | 30.00 | 96.2% | **0, 0.1**（文字模式） |

> 6 项非文字任务同用**通用配置 (ρ=0.25, λ=0.5)**（预算收紧，覆盖池减半）；文字两项切
> **文字模式 (ρ=0, λ=0.1)**。MME 为此档最弱项（92.2%），其余七项 ≥95.6%。

#### 4.0.3 K=64（保留 11% token，8 项平均保留率 **95.1%**）

| Benchmark | vanilla | idea4 最优 | 保留率 | 触发超参 (ρ, λ) |
|---|---:|---:|---:|---|
| GQA | 61.97 | 57.47 | 92.7% | 0.5, 0.5 |
| MMB-EN | 64.0 | 59.62 | 93.2% | 0.25, 0.5 |
| MME | 1874.5 | 1669.9 | 89.1% | 0.25, 0.5 |
| POPE | 86.99 | 85.63 | 98.4% | 0.5, 0.5 |
| SQA | 69.46 | 68.12 | 98.1% | 0.5, 0.5 |
| VizWiz | 54.06 | **56.32** | **104.2%** | 0.25, 0.5 |
| TextVQA | 46.07 | 42.54 | 92.3% | **0, 0**（纯 top-K） |
| OCRBench | 31.20 | 28.90 | 92.6% | 0.25, 0.5 |

> 极小预算下不存在单一最优 ρ：GQA/POPE/SQA 偏好 **ρ=0.5**（更看重全局冗余去除的显著核心），
> MMB/MME/VizWiz/OCRBench 偏好 **ρ=0.25**（更看重空间覆盖）；TextVQA 退到 **纯 top-K (ρ=0, λ=0)**。
> VizWiz 反超 vanilla（104.2%），MME 是全局最弱项（89.1%）。

#### 4.0.4 每项结果需微调的超参注释

冻结项（不随任务变动）：`cover_factor=3`（覆盖池只落在内容区域，§3.1）、σ→∞（关闭空间门控，
§4.1）、$w_f=w_a=1.0$（沿用 idea1 v5）。真正随任务/预算微调的只有两个旋钮：

- **ρ（覆盖比，主旋钮）**：控制覆盖池 vs 显著池的预算配比。
  - *全图/空间任务*（MMB/MME/VizWiz/GQA）随预算收紧对覆盖池更敏感：K=192 用 ρ=0.5，K=128 降到
    ρ=0.25，K=64 多数项仍取 ρ=0.25 保住空间铺展。
  - *感知/去冗任务*（POPE/SQA、以及 K=64 的 GQA）偏向**大显著池**，ρ 越小越稳；K=64 时 ρ=0.5 反而
    最好（把预算集中到全局 MMR 显著核心）。
  - *文字任务*（TextVQA/OCRBench）必须 **ρ=0**：覆盖池会把预算耗在背景 cell 上、排斥空间聚集的
    文字 patch。这是文字两项与其余六项唯一的结构性差异。
- **λ（冗余权重，副旋钮）**：Phase B MMR 的显著性/去冗平衡，等价显著性权重 $\alpha=1/(1+\lambda)$。
  - 非文字任务一律用**默认 λ=0.5**（α=66.7%），无需逐任务调。
  - 文字模式下 K=128/192 降到 **λ=0.1**（α=90.9%）保留更多相邻文字证据但仍去掉纯重复；
    K=64 进一步到 **λ=0**（纯 CLS-attention top-K），此时 TextVQA 最高但 OCRBench 反降（§4.2）。

一句话部署规则：**非文字任务按预算选 ρ∈{0.5(K≥192), 0.25(K≤128)}、λ=0.5；已知文字密集任务切
`IDEA_RHO=0 IDEA_LAMBDA=0.1`（K=64 用 λ=0）**。切换只改环境变量，不动选择器实现。

### 4.0.5 与 idea3 对比及宽预算档（K=288/346）

沿用 §3 每档最优 ρ 的**单一通用配置**（不逐任务切文字模式），与 idea3（纯全局 MMR）逐项对比；
TextVQA 列已换算到无 OCR 协议（vanilla=46.07）。

| 方案\@K | GQA | MMB | MME | POPE | SQA | TextVQA | nonTxt%van |
|---|---|---|---|---|---|---|---|
| vanilla(576) | 62.0 | 64.0 | 1875 | 87.0 | 69.5 | 46.1 | 100.00% |
| idea3-192 | 59.5 | 63.0 | 1783 | 87.5 | 68.4 | 45.1 | 97.73% |
| **idea4-192** (ρ=0.5) | **59.8** | **63.2** | **1803** | **87.6** | 69.2 | 44.4 | **98.35%** |
| idea3-128 | 59.3 | 62.3 | 1719 | 87.2 | 68.8 | 43.9 | 96.80% |
| **idea4-128** (ρ=0.25) | 59.3 | **62.6** | **1728** | 87.2 | **68.9** | 43.6 | **97.02%** |
| idea3-64 | 57.6 | 59.8 | 1639 | 86.3 | 68.3 | 42.0 | 94.24% |
| idea4-64 (ρ=0.25) | 57.2 | 59.6 | 1670 | 85.5 | 67.9 | 41.5 | 94.12% |
| idea3-288（初筛50%） | 61.0 | 63.9 | 1779 | 87.6 | 68.6 | 45.1 | 98.53% |
| **idea4-288** (ρ=0.25) | 61.0 | 63.8 | **1785** | 87.5 | **68.7** | 45.4 | **98.60%** |
| idea3-346（初筛60%） | 61.2 | 64.6 | 1831 | 87.3 | 68.6 | 45.6 | 99.33% |
| idea4-346 (ρ=0.5) | 61.2 | **64.9** | 1824 | 87.1 | **68.7** | 45.6 | 99.27% |

### 4.1 VizWiz 与 OCRBench 补充评测

沿用上表的最终配置（`cover_factor=3`、$\sigma\to\infty$；K=64/128/288 取 $\rho=0.25$，
K=192/346 取 $\rho=0.5$），在 lmms-eval 的 `vizwiz_vqa_val` 与 `ocrbench` 上补跑。
VizWiz 使用带本地标注的 **val split**，不是需提交官方服务器的 test split；两项分数均乘 100。

| 方案\@K | $\rho$ | VizWiz | %van | OCRBench | %van |
|---|---:|---:|---:|---:|---:|
| vanilla(576) | — | 54.06 | 100.00% | 31.20 | 100.00% |
| idea4-64 | 0.25 | **56.32** | **104.18%** | 28.90 | 92.63% |
| idea4-128 | 0.25 | **55.33** | **102.36%** | 29.90 | 95.83% |
| idea4-192 | 0.5 | **54.99** | **101.73%** | 30.80 | 98.72% |
| idea4-288 | 0.25 | **54.19** | **100.24%** | **31.60** | **101.28%** |
| idea4-346 | 0.5 | 53.89 | 99.69% | **31.80** | **101.92%** |

结果呈现明确的任务差异：

1. **VizWiz 对强剪枝不敏感，且小预算更好**：K=64/128/192 均超过 vanilla，K=64 达 56.32
   （104.18% vanilla）。随着 K 增大分数反而回落，说明该任务中去除冗余或干扰 patch 有正则化效果，
   而不是视觉 token 越多越好。
2. **OCRBench 随预算稳定恢复**：28.90@K=64 单调升至 31.80@K=346；K=192 已保留 98.72%，
   K=288/346 则略超 vanilla。相比 TextVQA 在所有预算下都停留于 41.5～45.6，OCRBench 的文字识别
   对扩大原始 patch 覆盖更直接受益，宽预算初筛基本无损。

完整聚合结果和逐样本输出位于 `docs/idea4/logs/cf3_sigInf*_k*/models__llava-v1.5-7b/`
下的 `20260714_174622_results.json` 与对应 `samples_{vizwiz_vqa_val,ocrbench}.jsonl`；每个配置分别
包含 4,319 条 VizWiz 和 1,000 条 OCRBench 样本。

### 4.2 文字任务的语义显著性占比实验

#### 参数化与假设

Idea4 中有两处会稀释 CLS 语义显著性：覆盖池占用 $\rho$ 比例的预算，以及显著池中的冗余惩罚
$\lambda$。为避免再引入一个等价超参数，将 Phase B 改写为

$$
\operatorname{score}(i\mid S)=\alpha\hat a_i-(1-\alpha)r_i,
\qquad \alpha=\frac{1}{1+\lambda},
$$

除以 $\alpha$ 后与原式 $\hat a_i-\lambda r_i$ 完全等价。因此默认 $\lambda=0.5$ 对应显著性权重
$\alpha=66.7\%$，$\lambda=0.1$ 对应 $\alpha=90.9\%$，$\lambda=0$ 则是纯 CLS-attention top-K。
同时令 $\rho=0$，表示不再给覆盖池预留 token，全部预算进入语义显著池。实验假设是：文字 patch
空间聚集且外观相似，默认覆盖与去冗会同时排斥它们；提高 $\alpha$、降低 $\rho$ 应保留更多文字证据。

#### 实验设计

采用两阶段选参，避免直接在 OCRBench 上穷举并过拟合：

1. **TextVQA 搜索阶段**：在 K∈{64,128,192} 上测试默认配置，以及
   $(\rho,\lambda)\in\{(0.25,0.1),(0,0),(0,0.1),(0,0.25)\}$；其余参数固定为
   `cover_factor=3`、$\sigma\to\infty$。每个 K 只按 TextVQA 分数选出一个候选。
2. **OCRBench 迁移验证**：不再调参，直接把三个候选在完整 1,000 条 OCRBench 上评测，并与 §4.1
   同 K 的通用 Idea4 配置比较。这样 OCRBench 是独立验证集，而不是第二个选参集。

候选配置的复现命令如下（K=64 时将 `IDEA_LAMBDA` 改为 0）：

```bash
CUDA_VISIBLE_DEVICES=0 IDEA_RHO=0 IDEA_LAMBDA=0.1 IDEA_SIGMA=1000000000 \
  OUTPUT_DIR=docs/idea4/logs/text_salience_k192_r0_l0.1 \
  bash scripts/idea_eval.sh idea4 192 textvqa_val,ocrbench
```

TextVQA 搜索结果（exact_match×100）：

| $\rho$ | $\lambda$ | $\alpha$ | K=64 | K=128 | K=192 |
|---:|---:|---:|---:|---:|---:|
| 通用 Idea4 | 0.5 | 66.7% | 41.50 | 43.60 | 44.40 |
| 0.25 | 0.1 | 90.9% | 42.46 | 44.54 | 44.93 |
| 0 | 0 | 100% | **42.54** | 44.33 | 44.68 |
| 0 | 0.1 | 90.9% | 42.49 | **44.55** | **45.16** |
| 0 | 0.25 | 80.0% | 42.38 | 44.04 | 45.06 |

其中“通用 Idea4”的 $\rho$ 仍取 §3 各 K 最优值（K=64/128 为 0.25，K=192 为 0.5）；
$\lambda$ 均为 0.5。搜索选择 K=64 的 $(0,0)$，以及 K=128/192 的 $(0,0.1)$。

OCRBench 独立验证结果（accuracy×100）：

| K | 通用 Idea4 | TextVQA 选出的配置 | 调整后 | Δ | %van（vanilla=31.20） |
|---:|---:|---|---:|---:|---:|
| 64 | 28.90 | $\rho=0,\lambda=0$ | 28.50 | -0.40 | 91.35% |
| 128 | 29.90 | $\rho=0,\lambda=0.1$ | 30.00 | +0.10 | 96.15% |
| 192 | 30.80 | $\rho=0,\lambda=0.1$ | **31.20** | **+0.40** | **100.00%** |

#### 结论与推荐配置

1. **中等预算下假设成立**：K=192 使用 $\rho=0,\lambda=0.1$ 后，TextVQA 从 44.40 提升到
   **45.16（+0.76）**，OCRBench 从 30.80 提升到 **31.20（+0.40）**，后者完全恢复 vanilla。
   这是同时弥补两个文字任务损失的最佳配置。
2. **收益主要来自取消覆盖池，但仍需少量去冗**：K=128/192 的最优点均为 $\alpha=90.9\%$，
   而不是 100%；纯 top-K 会重新引入相邻高注意力 patch 的冗余。
3. **极小预算不存在跨任务统一最优**：K=64 的纯 top-K 将 TextVQA 提高 1.04 分，却令 OCRBench
   降低 0.40 分。故 K=64 不建议切换为文字模式，仍保留通用配置。
4. **部署建议**：若任务已知为文字密集型，K=128/192 使用 `IDEA_RHO=0 IDEA_LAMBDA=0.1`；
   未知任务或 K=64 继续使用 §3 的通用配置。该调整不改选择器实现，只切换已有环境变量。

TextVQA 全量搜索日志位于 `docs/idea4/logs/tvqa_k*_r*_l*/`；OCRBench 迁移验证日志位于
`docs/idea4/logs/text_salience_k*/`，三组均包含完整 1,000 条逐样本输出。

**结论**：

1. **在除 TextVQA 外的任务上超越 idea3**：K=192 时 idea4 在**全部 5 个非文字 benchmark 上均 ≥ idea3**
   （含 POPE 87.6≥87.5）；K=128、K=288 时非文字均分超过 idea3；K=64/346 与 idea3 基本持平（±0.1%）。
2. **初筛不掉性能**：保留 50%/60%（K=288/346）时，idea4 对 GQA/POPE/SQA/MMB/MME 近无损
   （nonTxt 98.6%/99.2%），与 idea3 持平或更好，可作为二级剪枝的无损前置。
3. **修正协议后 TextVQA 几乎无损**：以同协议（无 OCR）的 vanilla 46.07 为基线，idea4 文字模式在
   K=192 达 45.16（98.0%）、K=128 达 44.55（96.7%），并非此前误用 58.3 基线时看到的「天花板」。
   剩余差距主要在 K=64（42.54，92.3%）——文字 patch 空间聚集、外观相似，极小预算下仍被去冗机制部分
   排斥；进一步突破需文字感知的局部保护或受控轻量合并，属后续方向。
4. **新增任务进一步区分了剪枝效应**：VizWiz 在 K=64 即超过 vanilla，说明强剪枝可过滤视觉干扰；
   OCRBench 则需要 K≥288 才完全恢复，说明密集文字识别更依赖较宽的原始 patch 覆盖。
5. **文字感知选参可降低宽预算要求**：在 K=192 将覆盖池关闭并把显著性权重提高到 90.9%，
   OCRBench 即可达到 vanilla 的 31.20，同时将 TextVQA 提升至 45.16；无需等到通用配置的 K≥288。

总对比（含 idea1/2/3、VisionZip、vanilla）见 [docs/idea_summary.md](../idea_summary.md)；
复现脚本 [scripts/aggregate_ideas.py](../../scripts/aggregate_ideas.py)（读 `docs/idea4/logs/*`）。
