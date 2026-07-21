# Idea 2：Spectral Relay Pruning——从频率分析到保守式视觉 Token 剪枝

## 1. 研究定位

本文研究的是严格意义上的 **visual token pruning**，不是 token compression、token merging，也不是 frequency-token reconstruction。

给定视觉 token：

$$
H\in\mathbb{R}^{B\times N\times D},
$$

剪枝操作只允许选择一个原始 token 索引集合：

$$
H_{\mathrm{pruned}}=H[:,S,:],\qquad S\subset\{0,\ldots,N-1\}.
$$

因此必须满足：

- 每个输出 token 都逐元素来自某个原始视觉 token；
- 不对多个 token 求平均；
- 不生成 coarse token、summary token 或 frequency token；
- 不把 IDCT/低通重建结果送入模型；
- 每个保留 token 都能追溯到原始 `24×24` CLIP patch 坐标。

频率分析在本文中的作用不是“压缩图像特征”，而是帮助回答三个剪枝问题：

1. 哪些位置是局部稳定主体，适合作为某一区域的代表 token？
2. 哪些局部高频证据可能承载文字、小物体、边缘或关系信息，需要避免过早删除？
3. 随着 layer 演化，视觉 token 何时从“工作空间”变成“冗余缓存”，可以安全剪掉一部分？

当前已经实现并评测的是 Stage A v5：LLaVA 输入端、projector 前、图像条件的保守式 hard pruning。Stage B/C 仍是后续分阶段剪枝方向。

## 2. 从频率分析得到的核心 insight

最初的直觉是：如果某个 patch 的高频残差大，它可能包含边缘、文字、小物体或局部细节，因此应该优先保留。但本地 LLaVA-1.5-7B 上的实验推翻了这个简单假设。

当前实验形成了四条更可靠的观察。

### 2.1 高频残差大，不等于更应该保留

在输入端 hard pruning 中，直接按 DCT 高频残差做全局 top-k 会明显劣于 random/uniform。原因在于，高频残差会偏向纹理、边缘、局部噪声和背景细节；这些 token 看起来“信息量高”，但并不一定是 VLM 回答问题所需要的主体证据。

这说明频率残差不能被简单解释为全局重要性：

$$
r_i^{\mathrm{freq}}\uparrow
\;\not\Rightarrow\;
i\ \text{should be kept}.
$$

它更像一个“局部变化强度”信号，而不是“任务相关性”信号。

### 2.2 Uniform 很强，说明空间覆盖是输入端剪枝的第一原则

实验中 `uniform` 是非常强的廉价基线。对 GQA 这类普通 VQA，视觉 token 具有很高冗余，均匀保留一部分 token 就能维持接近原始性能。

这暗示输入端剪枝最危险的不是删掉某个高频 token，而是让某些区域完全失去代表：

$$
\text{coverage collapse} \;>\; \text{single-token score error}.
$$

因此 Stage A 的第一原则不应该是“全图找最高分 token”，而应该是“先保证每个空间区域至少有代表 token”。

### 2.3 频率信号仍有价值，但角色必须降级

全局高频 top-k 失败，并不意味着频率分析无用。更合理的用法是把频率从“全局重要性分数”降级为“局部选择先验”。

也就是说，频率不再回答：

> 整张图中哪些 token 最重要？

而只回答：

> 在同一个局部 cell 内，哪个 token 更像稳定主体，适合作为该 cell 的代表？

这个转变是 v5 的关键：频率不主导全局排序，只参与 cell 内的保守选择。

### 2.4 Hard pruning 的删除是不可逆的

compression 可以把多个 token 的信息汇入新 token；hard pruning 不行。一个 token 被删掉后，后续 projector、LLM attention 和 decode KV cache 都无法再访问它。

因此输入端剪枝必须更保守。特别是对于 POPE/MME 这类物体存在性和感知任务，小物体或局部证据被删掉后很难由剩余 token 恢复。这也解释了为什么 v5 在 GQA/TextVQA/MMBench 上接近 uniform，但在 POPE/MME 上仍出现退化。

## 3. 从 insight 到方法：为什么 v5 是合理的

v5 不是从“高频重要”这个直觉直接推出来的，而是从实验负结果中修正出来的。它的设计逻辑是：

```text
全局高频 top-k 失败
        ↓
不能让频率主导全图排序
        ↓
uniform 很强，说明空间覆盖必须优先
        ↓
先把图像划成 K 个空间 cell
        ↓
每个 cell 只选一个原始 token
        ↓
cell 内再用 medoid 代表性 + 低频稳定性选代表 token
```

因此 v5 的核心不是“用频率选重要 token”，而是：

> 用空间覆盖约束避免区域空洞，用局部 medoid 保证代表性，用低频残差选择更稳定的主体 token。

它吸收了 uniform 的强点，又给每个 uniform-like cell 增加一个局部选择机制。

## 4. 频率如何用于剪枝，而不是压缩

### 4.1 低通重建只用于打分

在完整 `24×24` CLIP patch 网格上，对视觉特征做二维 DCT，并保留低频块做重建：

$$
\widehat H=\mathrm{IDCT}\left(\Pi_{k\times k}(\mathrm{DCT}(H))\right).
$$

每个 token 的频率残差定义为：

$$
r_i^{k}=\|H_i-\widehat H_i\|_2.
$$

但这个低通重建结果只用于计算分数，绝不作为模型输入。真正进入 projector 的仍然是原始 token 子集：

```python
features_pruned = features.gather(1, keep_indices)
```

### 4.2 当前 v5 使用低频稳定性，而不是高频重要性

在 v5 中，cell 内的频率项是：

$$
\ell_i=\operatorname{Norm}_{C(i)}(-r_i^{16}).
$$

也就是残差越小，低频稳定性越高。这个设计承认了前面的实验发现：高频残差不适合作为输入端全局保留分数。v5 只在局部 cell 内偏向更稳定、更像主体区域代表的 token。

### 4.3 剪枝后如果继续做频率分析，应转向图频率

一旦从 `576` 个规则 patch 剪成不规则子集，剩余 token 不能再 reshape 成规则方阵。后续 Stage B/C 若继续使用频率，应维护原始二维坐标，在保留集合上构造空间图：

$$
w_{ij}=
\exp\left(-\frac{\|p_i-p_j\|_2^2}{2\sigma_p^2}\right)
\mathbb{1}[j\in\mathcal N_k(i)].
$$

局部图残差可定义为：

$$
r_i^{\mathrm{graph}}
=\left\|H_i-\frac{\sum_j w_{ij}H_j}{\sum_jw_{ij}}\right\|_2.
$$

这为后续深层剪枝保留了分析路径，同时仍不产生任何混合 token。

## 5. Stage A：图像条件的保守剪枝

Stage A v5 固定为 LLaVA-1.5 输入端的 hard pruning：在 CLIP selected layer 输出之后、`mm_projector` 之前，将 `24×24=576` 个视觉 patch token 剪到预算 `K`。

当前策略名为：

```text
cell_medoid_lowfreq
```

冻结配置：

```text
grid       = 24×24
lowpass    = 16
w_freq     = 1.0
position   = CLIP selected layer output, before mm_projector
output     = exactly K original patch tokens
```

### 5.1 输入与输出

输入是 CLIP patch features：

$$
H=\{H_i\}_{i=1}^{576},\qquad H_i\in\mathbb{R}^{D},
$$

以及每个 patch 的二维坐标：

$$
p_i=(x_i,y_i).
$$

输出是 `K` 个原始 token 的索引集合：

$$
S=\{s_1,\ldots,s_K\},\qquad H'=H[S].
$$

输出索引按原始 raster order 排序，以保持视觉 token 的相对空间顺序。

### 5.2 第一步：构造 K 个空间 cell

v5 先在归一化二维坐标上做 deterministic farthest point sampling，得到 `K` 个空间 anchor。然后把每个 patch 分配给最近 anchor，形成 `K` 个 Voronoi cell：

$$
C_m=\{i:\ m=\arg\min_a \|p_i-a\|_2\},\quad m=1,\ldots,K.
$$

这个步骤把 uniform 的强先验显式写进方法里：每个 cell 最终只保留一个 token，因此不会出现全图 top-k 把大量预算集中到少数纹理区域的问题。

### 5.3 第二步：计算 cell 内 medoid 代表性

在每个 cell 内，v5 希望保留“最能代表这一小块区域”的 token，而不是最极端的 token。

先对 feature 做单位化，并计算 cell 中心方向：

$$
\bar h_C=
\operatorname{normalize}\left(
\frac{1}{|C|}\sum_{j\in C}\operatorname{normalize}(H_j)
\right).
$$

每个 token 的 medoid 代表性为：

$$
m_i=\operatorname{Norm}_{C(i)}
\left[
\cos\left(\operatorname{normalize}(H_i),\bar h_{C(i)}\right)
\right].
$$

`m_i` 越高，说明该 token 越接近 cell 内的语义/外观中心，更适合作为该区域的代表。

### 5.4 第三步：计算低频稳定性

在完整 `24×24` 网格上计算 `lowpass=16` 的 DCT 低通重建残差：

$$
\widehat H=\mathrm{IDCT}\left(\Pi_{16\times16}(\mathrm{DCT}(H))\right),
\qquad
r_i^{16}=\|H_i-\widehat H_i\|_2.
$$

cell 内低频稳定分数定义为：

$$
\ell_i=\operatorname{Norm}_{C(i)}(-r_i^{16}).
$$

这一步与早期失败实验直接呼应：v5 不奖励高频残差，而是在同一个局部 cell 内轻微偏向更稳定的低频主体 token。

### 5.5 第四步：每个 cell 选择一个原始 token

cell 内综合分数为：

$$
s_i=m_i+1.0\cdot \ell_i.
$$

每个 cell 输出得分最高的原始 token：

$$
k_m=\arg\max_{i\in C_m}s_i.
$$

最终：

$$
S=\operatorname{sort}(\{k_m\}_{m=1}^{K}),
\qquad
H'=H[S].
$$

等价伪代码：

```python
def cell_medoid_lowfreq(features, budget, grid=24, lowpass=16, w_freq=1.0):
    cells = fps_voronoi_cells(grid=grid, budget=budget)
    residual = dct_residual(features, grid=grid, lowpass=lowpass)
    keep = []
    for cell in cells:
        centroid = normalize(mean(normalize(features[cell]), dim=0))
        medoid = minmax(cosine(normalize(features[cell]), centroid))
        lowfreq = minmax(-residual[cell])
        score = medoid + w_freq * lowfreq
        keep.append(cell[argmax(score)])
    return sort(unique(keep))
```

若某个 cell 只有一个 token，则该 token 必然保留。由于 cell 数等于 budget，输出 token 数严格等于 `K`。

### 5.6 与 token compression 的边界

v5 是剪枝，不是压缩。判据如下：

- 输出长度从 `576` 变为 `K`；
- 每个输出 token 都来自输入 token 的 `gather/index_select`；
- 不引入重建 token、合并 token、平均 token 或 learnable summary token；
- 频率分析只用于选择索引，不改变 feature；
- `mm_projector` 和后续 LLM 只看到更短的视觉序列。

这也解释了 v5 的风险：被删 token 后续完全不可见。对于物体存在性、局部感知和细粒度定位任务，budget 不能压得过低。

### 5.7 LLaVA 注入与正确性确认

v5 通过 monkey patch 注入 LLaVA：

| 目标 | 实现/检查 | 证据 |
|---|---|---|
| projector 前剪枝 | `fourier_compressor/integrations/llava/stage_a_patch.py` patch `LlavaMetaForCausalLM.encode_images` | 先取 `vision_tower(images)`，再 `stage_a_select + apply_selection`，最后调用 `mm_projector` |
| 确认 patch 生效 | `apply_stage_a(config)` 设置 `_fourier_stage_a_patched=True` 并记录 `current_config` | 评测脚本在 `lmms_eval.evaluator.simple_evaluate` 前调用 |
| 确认 hard pruning | `fourier_compressor/prune.py::apply_selection` 使用 `features.gather` | 单元测试验证输出 token 是输入 token 子集 |
| 确认实际 budget | `stage_a_patch.stats()` 记录 `calls/in_tokens/out_tokens` | 结果 JSON 中 `avg_kept_tokens` 严格等于 `K` |
| 确认端到端生效 | `examples/eval_stage_a.py`、`examples/run_stage_a_confirm.py`、`examples/run_v5_budget_sweep.py` | lmms-eval 指标随 `K` 改变 |

已执行的本地检查包括：

- `py_compile` 覆盖评测脚本；
- `tests/test_stage_a.py` 与 `tests/test_spectrum.py` 共 11 个测试通过；
- `K∈{288,230,192,144,128,96,64}` 下均验证输出预算、索引唯一性、排序和原 token 子集属性。

## 6. 当前实验结果与结论

### 6.1 不同剪枝力度：GQA-lite 与 POPE-1000

预算扫描使用本地 LLaVA-1.5-7B、lmms-eval、本地缓存数据，在 GPU `0--4` 上完成。结果文件为：

```text
result/v5_budget_k*_gqa500_pope1000.json
logs/v5_budget_k*_gqa500_pope1000.runlog
```

| K | 剪枝率 | avg kept | GQA-lite EM | POPE acc | POPE precision | POPE recall | POPE F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 88.9% | 64.0 | 65.8% | 77.5% | 88.952% | 62.8% | 73.623% |
| 96 | 83.3% | 96.0 | 68.4% | 81.5% | 89.474% | 71.4% | 79.422% |
| 128 | 77.8% | 128.0 | 68.8% | 81.7% | 88.378% | 73.0% | 79.956% |
| 144 | 75.0% | 144.0 | 68.4% | 81.8% | 88.221% | 73.4% | 80.131% |
| 192 | 66.7% | 192.0 | 69.4% | 84.1% | 88.662% | 78.2% | 83.103% |
| 288 | 50.0% | 288.0 | 69.8% | 84.4% | 90.000% | 77.4% | 83.226% |

由此得到三点结论：

1. GQA-lite 在 `K≈96` 后基本进入平台区，说明普通 VQA 对输入端视觉 token 数并不极端敏感；
2. POPE 明显更依赖局部证据，`K=64` 的 recall 大幅下降，`K=192` 才接近 `K=288`；
3. `K=128` 是高压缩折中点，不是所有任务的最优点。

这与前面的频率 insight 一致：hard pruning 无法恢复被删证据，因此物体存在性任务需要更高的空间采样密度。

### 6.2 K=128 六基准 paired comparison

固定 `K=128` 后，v5 与 `uniform@128` 在六个 benchmark 上做 paired comparison。所有任务使用相同模型、相同 projector 前剪枝位置、相同 decoding 设置，uniform 与 v5 的 `avg_kept_tokens` 均严格为 128。

详细结果见：

```text
result/stage_a_v5_confirmation.md
```

| Benchmark | 样本数 | uniform@128 | v5@128 | 差值 | 判定 |
|---|---:|---:|---:|---:|---|
| GQA exact match | 12,578 | 58.491% | 58.427% | -0.064pt | 容差内 |
| ScienceQA-IMG exact match | 2,017 | 69.261% | 68.716% | -0.545pt | 容差内 |
| TextVQA exact match | 5,000 | 52.002% | 51.116% | -0.886pt | 容差内 |
| POPE F1 | 9,000 | 83.220% | 82.065% | -1.155pt | **失败** |
| MME total | 2,374 | 1684.573 | 1650.848 | 保留 97.998% | **失败** |
| MMBench EN dev | 4,329 | 60.481 | 60.739 | +0.258 | 优于 uniform |

POPE 的 accuracy、precision、recall 差值分别为 `-0.900pt`、`-0.292pt` 和 `-1.667pt`。MME perception 从 `1384.216` 降至 `1353.348`（`-2.230%`），cognition 从 `300.357` 降至 `297.500`（`-0.951%`）。

严格结论是：

> v5 是一个已确认注入、可复现、端到端运行的 hard pruning 策略；它在 GQA、ScienceQA-IMG、TextVQA、MMBench 上接近或略优于 uniform，但在 POPE 和 MME 上仍有可复现退化，不能声称六基准全面表现良好。

### 6.3 当前结果反过来支持了哪些规律

当前实验并没有证明“频率剪枝优于 uniform”，但证明了更有价值的负规律和条件规律：

1. 高频残差不能作为输入端全局保留分数；
2. 空间覆盖是 Stage A 的硬约束，而不是可选正则项；
3. 频率更适合作为 cell 内选择代表 token 的局部稳定性先验；
4. `K=128` 对普通 VQA 足够激进，但对物体存在性和感知任务偏紧；
5. 如果目标是统一覆盖 GQA/SQA/TextVQA/POPE/MME/MMBench，必须引入任务自适应 budget 或更晚的 question-aware pruning。

## 7. Spectral Relay Pruning 的下一步：分阶段剪枝

Stage A v5 只解决输入端、图像条件剪枝。它无法使用问题信息，因为在 LLaVA 序列中视觉 token 位于问题文本之前，CLIP/projector 阶段还没有发生 question-to-vision 交互。

因此更完整的 SRP 假设是：

```text
Stage A：projector 前，图像条件保守剪枝
Stage B：LLM 中层保留较多视觉 token，让问题 token 读取视觉证据
Stage C：深层 question-aware 剪枝，删除已经被问题 token 吸收的冗余视觉 token
```

首个静态版本：

```text
CLIP 576 tokens
      │
      │ Stage A: v5 cell_medoid_lowfreq
      ▼
256 original visual tokens
      │
      │ Stage B: LLM layers 0...23, preserve visual workspace
      ▼
256 contextualized visual tokens
      │
      │ Stage C: question-aware pruning
      ▼
144 original visual tokens
      │
      │ LLM layers 24...31
      ▼
answer
```

这个方向的核心不是让 Stage A 一步达到最终最优，而是承认输入端剪枝缺少任务条件，因此先保守保留，再让问题 token 在中层完成信息读取，最后再删。

## 8. 本仓库实验结果（spectral，与 idea2/3/4 同口径）

> 上文 §6 是外部 `fourier_compressor` 项目里 GQA-lite/POPE-1000 的旧结果。这里补上本仓库
> LLaVA-1.5-7B + lmms-eval 全量 6 benchmark 的结果（`IDEA_METHOD=spectral`，注入点与
> idea2/3/4 完全一致，logs 在 `docs/idea1/logs/k<K>/`），用于四方案通用性横向对比。

| 配置 | GQA | MMB-EN | MME | POPE | SQA | TextVQA | Avg%van |
|---|---|---|---|---|---|---|---|
| vanilla(576) | 61.97 | 64.00 | 1874 | 86.99 | 69.46 | 58.27 | 100% |
| VisionZip-128 | 57.66 | 62.20 | 1764 | 84.64 | 68.67 | 56.86 | 96.3% |
| idea1-192 | 59.11 | 62.54 | 1770 | 85.38 | 68.82 | 33.24 | 90.3% |
| idea1-128 | 58.23 | 60.31 | 1668 | 83.32 | 68.27 | 27.14 | 86.3% |
| idea1-64 | 54.93 | 54.30 | 1507 | 78.81 | 65.69 | 19.03 | 78.6% |
| idea1-288 (初筛50%) | 59.76 | 63.57 | 1781 | 85.08 | 70.10 | 35.81 | 91.8% |
| idea1-346 (初筛60%) | 60.18 | 63.75 | 1780 | 85.01 | 70.45 | 38.54 | 92.8% |

结论：纯空间覆盖（FPS cell + 低频稳定 medoid）在 GQA 上甚至略胜 VisionZip（预算按空间均分对
全图理解够用），但：

1. **TextVQA 崩溃最严重**（K=64 仅 19.0，四方案里最差）——均匀空间采样把聚集的小文字区域整体丢弃，
   印证 §2 的负结论"高频/覆盖不等于任务重要性"；
2. POPE/MME 掉分明显（K=64 POPE 78.8、MME 1507），复现了 §6 里"缺显著性信号 → 感知类退化"的老问题；
3. 作为高预算初筛（288/346）尚可（Avg 91.8/92.8%），但仍是四方案里最弱的初筛器。

这正是 idea4 用"注意力感知的 cell 代表 + 显著池"来修补 idea1 的直接动机。四方案对比见
[docs/idea_summary.md](../idea_summary.md)。
