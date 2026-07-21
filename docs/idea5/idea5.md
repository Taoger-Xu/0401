# Idea-5：细节门控 CLS-MMR（Detail-Gated MMR）——针对 TextVQA 天花板的冗余惩罚豁免

> 定位：在进入 LLM 之前、`mm_projector` 之前，对 576 个视觉 token 做 hard pruning，
> 与 idea1–4 共享同一注入点（CLIP 倒数第二层输出）与同一 token 预算（CLS + K−1 个原始 patch）。
> idea5 以 **idea3（CLS-MMR，四方案通用性最好）为骨架**，直接攻击 idea1–4 全部实验
> 暴露出的唯一共同短板：**TextVQA 天花板**（[idea_summary.md](../idea_summary.md) §3.1/§4）。
>
> 本文档为 v2。v1 草稿（FACT-Pruning：低频冗余 × CLS 注意力 + 4×4 硬覆盖）写于 idea1–4
> 实验完成之前，其核心设计已被数据证伪，处置见 §3。
>
> **⚠️ 状态（2026-07-08）：已关闭。** 本方案的前提——"TextVQA 是纯剪枝的结构性天花板"——
> 被证实为评测协议假象（lmms-eval 升级导致 OCR 提示行丢失，见 §9.2）。统一协议下 idea3
> 的 TextVQA 与 VisionZip 持平、Avg%van 三档全面领先。细节门控与合并变体均无增益（§9.1）。
> §1–§8 保留作设计过程记录；结论以 §9 与 [idea_summary.md](../idea_summary.md) 修正节为准。

## 1. 从 idea1–4 实验结果导出的设计约束

idea1–4 的实验 A（K=192/128/64）+ 实验 B（K=288/346）共 20 个配置全部完成
（[idea_summary.md](../idea_summary.md)），给出五条硬约束，idea5 的每个设计决定都必须对得上号：

| # | 实验事实 | 对 idea5 的约束 |
|---|---|---|
| C1 | CLS 注意力是关键一跃：idea3 把 TextVQA 从 idea1 的 19~33 拉回 42~46，POPE 反超 VisionZip（86.3 vs 80.6 @K=64） | 重要性信号必须保持为 CLS 注意力，不掺杂图像统计量 |
| C2 | 频率/局部变化等图像统计量做**全局重要性**必败：idea1/idea2 TextVQA 崩溃（19.0/29.4 @K=64），Avg 垫底 | 频率类信号只能做**辅助先验**，不得参与全局排序（idea1 §2.3 的"降级"教训） |
| C3 | 硬空间覆盖约束收益不稳、反伤 POPE/MME：idea4 POPE 82.5 vs idea3 86.3 @K=64，总均值低于 idea3 | **不引入任何覆盖池/cell 最低保留约束**；覆盖交给 MMR 自由决定 |
| C4 | TextVQA 是纯剪枝的结构性短板：idea3/idea4 卡在 39~46 且几乎不随预算变化（K=64→346），根因是**文字 patch 空间聚集、外观相似，被 MMR 冗余惩罚系统性排斥** | 突破口不是更强的空间去冗，而是**放宽文字块内部的"近处相似"惩罚** |
| C5 | idea4 的空间门控只放宽了"远处相似"，TextVQA 无改善（42.1 < idea3 45.1 @K=192） | 门控变量必须换：不能按**距离**门控，要按**细节/文字似然**门控 |

一句话：**idea3 已经把"重要且不冗余"做对了，唯一错的是把"文字块内部的相似"也当成了冗余**。
文字区域的 patch 特征相似是 CLIP 编码的伪相似（同为笔画纹理），不是信息冗余——删掉任何一块都
丢失不同的字符。idea5 只修这一处。

## 2. 方法：细节门控的 MMR 冗余惩罚

沿用 idea3 的全部记号：CLIP 倒数第二层 patch 特征 $f_i\in\mathbb R^{1024}$、
CLS 注意力 $a_i=\sum_h\text{Attn}^{(L-2)}_h[\text{cls},i]$、$\hat a=\operatorname{minmax}(a)$、
$\hat f_i=f_i/\lVert f_i\rVert$。

### 2.1 细节分数 $d_i$

新增一个"细节/文字似然"分数，衡量 token 是否处于高频细节区（文字、小物体、精细结构）：

$$
d_i=\Big[\operatorname{minmax}\big(v_i\big)\Big]^{p},\qquad
v_i = 1-\frac{1}{8}\sum_{j\in\mathcal N_8(i)}\cos(\hat f_i,\hat f_j),
$$

即 idea2 的 8 邻域局部变化度（已实现：`score_local_variation`），锐化指数 $p=2$ 用于压低
中等纹理、只保留强细节区的响应。备选来源（消融项）：idea1 的 DCT 高频残差
$\operatorname{minmax}(r_i^{16})$（`lowpass=16`，机制已实现）。

**关键角色限定（呼应 C2）**：$d_i$ **不加进重要性**、不参与全局排序，只用来**门控冗余惩罚**。
这与 idea1 的教训一致——频率信号回答不了"哪里重要"，但可以回答"这里的相似算不算冗余"。

### 2.2 门控 MMR

已选集合 $S$ 时，候选 token $i$ 的边际得分：

$$
\text{score}(i\mid S)=\hat a_i-\lambda\,(1-\gamma\,d_i)\cdot
\max_{j\in S}\big[\cos(\hat f_i,\hat f_j)\big]_+ .
$$

与 idea3 唯一的区别是惩罚项乘了 $(1-\gamma d_i)$：

- $d_i\approx 0$（平滑区/背景）：完整惩罚，行为与 idea3 逐 token 一致——POPE/GQA 优势原样保留；
- $d_i\approx 1$（文字/细节区）：惩罚被豁免，该 token 只要 CLS 注意力足够高就能入选，
  **即使它与已选的相邻文字 token 高度相似**——文字块得以被连片保留；
- 豁免只是"不罚"，不是"加分"：$\text{score}\le\hat a_i$ 恒成立，低注意力的纹理区不会因为
  $d_i$ 高而混进来（这是与 idea2 全局排序失败的本质区别）。

贪心流程、增量更新、复杂度均与 idea3 相同（$O(K\cdot N)$）；实现上是 `select_cls_mmr`
的一行改动：`mmr = imp - lam * (1 - gamma * d) * max_sim`。CLS 恒保留，输出 CLS + (K−1) 个
原始 patch，raster 排序，纯 gather，与 idea1–4 剪枝边界判据完全一致。

### 2.3 退化情形

| 极限 | 退化为 |
|---|---|
| $\gamma=0$ | idea3 cls_mmr（逐 token 严格一致，可作 smoke test） |
| $\gamma=1,\ d\equiv 1$ | VisionZip dominant top-K 纯剪枝版（λ 失效） |
| $\lambda=0$ | 同上（与 idea3 相同的退化） |

直觉上，idea5 在每个 token 处按 $d_i$ 在"idea3 行为"（去冗余优先）和"VisionZip dominant 行为"
（显著性优先、允许扎堆）之间插值：**平滑区当 idea3 用，细节区当 VisionZip 用**。
而实验已分别证明这两种行为各自在哪类区域是对的（C1/C4）。

### 2.4 与 idea3 / idea4 的机制对照

| 方案 | 惩罚的"相似" | TextVQA 病根处理 |
|---|---|---|
| idea3 | 一切特征相似 | 文字块内部相似被当冗余 → 排斥（45.1 @K=192） |
| idea4 | 空间上近且特征相似 | 文字块**又近又像**，惩罚反而全额生效 → 无改善（42.1） |
| **idea5** | **非细节区**的特征相似 | 文字块 $d_i$ 高 → 惩罚豁免 → 连片保留 |

## 3. 对 v1 草稿（FACT-Pruning）的处置

v1 的三个核心设计逐条对照数据：

1. **全局剪枝分数 $P_i=R_i(1-A_i)$（低频冗余 × 低注意力，top-M 剪除）**——废弃。
   两个问题：(a) $R_i$ 作为全局信号属于 C2 证伪范围；(b) 该式对保留集合**没有任何去冗余机制**，
   保留的高注意力 token 会像 VisionZip dominant 一样空间扎堆——而 idea3 的数据表明去冗余
   恰恰是 POPE +5.7、GQA +2.4（vs VisionZip @K=64）的来源。
2. **4×4 cell、每 cell 至少保留 1 个的硬覆盖约束**——废弃，C3 直接证伪（idea4 的覆盖池
   比这更聪明——注意力感知的代表选择——尚且伤 POPE，固定 4×4 只会更差）。
3. **DCT 低频重建机制**——保留但**角色反转**：v1 用低频重建误差找"可剪的冗余"，
   v2 用它（或局部变化度）找"要保护的细节"，且只作惩罚门控。信号复用、语义相反，
   与 idea1 v5"频率降级为局部先验"的路径同构。

v1 中与方法无关但论文化时仍有效的内容（已发表方法对照表、公平比较协议、显著性检验、
oracle-text 上界、审稿质疑预案）浓缩进 §8，其余（K=144 协议、GQA-lite 探索集等与
task.md 框架冲突的实验设计）废弃，统一改用 [task.md](../task.md) 协议。

## 4. 变体：受控轻量合并（天花板探针，非主方法）

[idea_summary.md](../idea_summary.md) §4 给出的第二条出路是"受控的轻量合并"。为把
"选择质量"与"范式天花板"解耦，增加一个诊断变体 `idea5+merge`：

- 主方法选出保留集合 $S$ 后，每个被剪 token 合并（加权平均）进 $S$ 中与其最相似的 token——
  即把 VisionZip 的 contextual 合并嫁接到 idea5 的选择结果上，token 数不变仍为 K；
- **用途**：若 `idea5+merge` 的 TextVQA ≈ VisionZip（~57），说明 idea5 的**选择**已到位、
  剩余差距全部来自"被剪文字信息必须以合并形式保留"，纯剪枝范式的天花板得到定量刻画；
  若仍显著低于 VisionZip，说明选择本身还有失误，回到 §2 迭代；
- 该变体破坏"纯剪枝"边界，只进诊断/分析节，不进主表、不参与四方案通用性排序。

## 5. 超参

| 超参 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| $\lambda$ | `IDEA_LAMBDA` | 0.5 | 去冗余强度，沿用 idea3 冻结值，不重扫 |
| $\gamma$ | `IDEA_GAMMA` | 1.0 | 细节豁免强度；消融 {0.5, 1.0}；$\gamma=0$ 即 idea3 |
| $p$ | `IDEA_DETAIL_P` | 2 | $d_i$ 锐化指数；消融 {1, 2} |
| $d_i$ 来源 | `IDEA_DETAIL_SRC` | `local_var` | 备选 `dct`（高频残差）；默认选 local_var 因 idea2 的 TextVQA（40.7）显著好于 idea1（33.2），说明它对文字区更敏感 |
| CLS token | — | 恒保留 | 占 1 个预算，对齐 idea1–4 / VisionZip |

按 6.6 冻结协议：$\gamma,p$ 及 $d_i$ 来源只在 K=128 一档上确定一次，其余 K 全部沿用。

## 6. 待验证假设与量化判据

基线数字直接取自 [idea_summary.md](../idea_summary.md)：

1. **主假设（TextVQA 修复）**：文字块豁免应使 TextVQA 相对 idea3 显著回升且**随预算增长**
   （idea3 的病态特征是 K=64→346 卡在 42~46 不动）。
   判据：TextVQA@K=192 ≥ 50（收复 idea3→VisionZip 差距 45.1→57.3 的 40% 以上）；
   TextVQA(K=346) − TextVQA(K=64) ≥ 5pt（恢复预算敏感性）。
2. **不伤线（保住 idea3 的强项）**：门控只在高 $d_i$ 区生效，平滑区行为与 idea3 一致。
   判据：每档 K 上 POPE ≥ idea3 − 0.5、GQA ≥ idea3 − 0.5（尤其 POPE@K=64 ≥ 85.8）。
3. **通用性净增**：Avg%van 在三档 K 全面超过 idea3（94.3/93.2/90.6），
   目标逼近 VisionZip（97.7/96.3/93.6）——若 1、2 同时成立，此条自动成立。
4. **机制验证（不依赖 benchmark）**：在 TextVQA 图像上统计"高 $d_i$ 连通区内的保留 token 占比"，
   idea5 应显著高于 idea3/idea4；且 $\gamma$: 0→1 时该占比单调上升。这是比分数更直接的
   机制证据（同 idea4 §6 的诊断思路，脚本可复用 `scripts/diag_dual_diversity.py` 框架）。
5. **天花板刻画**：`idea5+merge` 的 TextVQA 与 VisionZip 的差 ≤ 1pt ⇒ 剩余差距归因于
   合并需求而非选择质量（§4）。

反证条件：若 $\gamma=1$ 下 TextVQA 仍 ≤ 47（豁免无效），优先检查 $d_i$ 热力图是否真的
点亮文字区（换 `dct` 来源重试）；若 TextVQA 升但 POPE/MME 明显掉，说明纹理区被豁免误伤，
增大 $p$ 或对 $d_i$ 加分位数截断后重试；两者都失败则接受"纯剪枝救不了 TextVQA"的结论，
主攻方向转向 §4 的受控合并（升级为主方法、重新划定与 idea1–4 的比较边界）。

## 7. 实验安排

统一设置见 [task.md](../task.md)（注入点、模型、6 benchmark、日志路径与 idea1–4 相同），
日志落 `docs/idea5/logs/k<K>/`。

```bash
# 实验 A：剪枝力度扫描（与 idea1–4 同口径）
CUDA_VISIBLE_DEVICES=g bash scripts/idea_eval.sh idea5 192
CUDA_VISIBLE_DEVICES=g bash scripts/idea_eval.sh idea5 128
CUDA_VISIBLE_DEVICES=g bash scripts/idea_eval.sh idea5 64

# 实验 B：初筛可行性
CUDA_VISIBLE_DEVICES=g bash scripts/idea_eval.sh idea5 288
CUDA_VISIBLE_DEVICES=g bash scripts/idea_eval.sh idea5 346

# 实验 C：γ / p / d 来源消融（固定 K=128；γ=0 即 idea3-128，免跑）
IDEA_GAMMA=0.5                    bash scripts/idea_eval.sh idea5 128
IDEA_DETAIL_P=1                   bash scripts/idea_eval.sh idea5 128
IDEA_DETAIL_SRC=dct               bash scripts/idea_eval.sh idea5 128

# 实验 D：机制诊断（假设 4；只需 TextVQA 图像 + 选择器，不跑 benchmark）
# 实验 E：天花板探针 idea5+merge（只跑 textvqa_val 一项即可回答 §4 的问题）
```

省算优先级：**A(K=128) → D → A(其余) → C → B → E**。先用 K=128 单点 + 诊断确认门控
真的把文字块留下来了，再铺全量；TextVQA 单任务即可预判成败，不必每步跑满 6 benchmark。

实现清单（跑实验前完成）：

- [ ] `visionzip/prune_ideas.py`：新增 `select_detail_mmr(patch_feats, cls_attn, budget, lam, gamma, p, detail_src)`——
      复用 `score_local_variation` / `_dct_matrix` 算 $d_i$，主循环即 `select_cls_mmr`
      加门控因子；
- [ ] `visionzip/idea_inject.py`：`_idea_select` 加 `method == "detail_mmr"` 分支
      （需要 `hidden_states[-2]` 与 `attentions[-2]`，取法同 cls_mmr）；
- [ ] `eval/lmms_eval_entry.py`：读取 `IDEA_GAMMA` / `IDEA_DETAIL_P` / `IDEA_DETAIL_SRC` 透传；
- [ ] `scripts/idea_eval.sh`：`case` 加 `idea5) export IDEA_METHOD=detail_mmr`，日志行打印三个新超参；
- [ ] smoke test：K∈{64,128,192,288,346} 输出 token 数严格为 K、索引唯一、原 token 子集、
      raster 有序；**γ=0 与 idea3 选集逐 token 一致**（退化正确性）；
- [ ] 诊断脚本（实验 D）：$d_i$ 热力图 + 高细节区保留占比统计，落 `docs/idea5/logs/diag/`。

## 8. 论文化备忘（自 v1 浓缩，方法无关、仍有效）

- **已发表方法对照**：encoder 侧 text-agnostic 组（必比）：VisionZip、FasterVLM、LLaVA-PruMerge；
  LLM 侧 text-aware 组（次要参照，声明信息量不对等）：FastV、SparseVLM、PyramidDrop。
  主表 K 对齐 VisionZip 口径 {192,128,64}——与 task.md 实验 A 天然一致。
- **公平协议**：同一评测代码/prompt/greedy decoding；超参只在 K=128 上冻结一次（§5），
  不做 per-benchmark 调参；效率测量同机同精度，选择器自身开销（$d_i$ + MMR）单独报告。
- **叙事定位**：贡献点不是"又一个剪枝分数"，而是**修正 MMR 冗余度量**——"细节区的特征相似
  是伪冗余"。消融链条现成：idea3（γ=0）→ idea4（距离门控，已证无效）→ idea5（细节门控），
  每一步都有六基准数据支撑；idea4 的负结果反而成为"为什么门控变量必须是细节而非距离"的论据。
- **显著性**：主方法确定性无 seed 方差；与 idea3 的逐样本对比用 paired bootstrap / McNemar
  报 p 值（TextVQA n=5000，+5pt 远超噪声水平，但 POPE"不伤线"判定需要检验支撑）。
- **oracle-text 上界与反向剪枝 sanity check**（v1 §7.6）保留为附录级诊断。
- **Limitations 主动声明**：依赖 CLS attention 的架构边界（Qwen 系列需替代信号）、
  text-agnostic 的原理性上界（oracle 差距量化）、固定预算无 per-image 自适应、
  576-token 固定分辨率设置。

## 9. 实验结果

### 9.0 预实验：低频重建探针（2026-07-08，已完成）

回答"文字信息是否住在特征图高频、低频压缩路线对 TextVQA 是否可行"。两组配置，
LLaVA-1.5-7B、textvqa_val 全量 5000 样本，注入点同 idea1–4（CLIP 倒数第二层、projector 前）：

- **探针**（`IDEA_METHOD=lowfreq_recon`）：**不删任何 token**，把 576 个 patch 特征替换为其
  2D-DCT 低通重建（截止 `lp`），CLS 不动，共 577 token 进 LLM；
- **频域压缩**（`IDEA_METHOD=dct_down`）：保留 g×g 低频系数块并在 g×g 网格上逆变换
  （正交 DCT resize），共 g²+1 token 进 LLM——与剪枝同预算的"低频合并"对照。
  logs：`docs/idea5/logs/recon_lp<lp>/`、`dctdown_g<g>/`，脚本 `bash scripts/idea_eval.sh recon|dctdown <lp|g> textvqa_val`。

> **修正（07-08 深夜）**：下表 vanilla 一行最初误用了带 OCR 协议的 58.27；无 OCR 协议
> （本轮所有运行的实际协议）的 vanilla 为 **46.07**（见 §9.2 协议断裂）。表格与结论已按
> 46.07 修正；dctdown 与 idea3 的对比原本就同协议，不受影响。

| 配置 | token 数 | 保留系数占比 | TextVQA | 同预算参照（均无 OCR） |
|---|---:|---:|---:|---|
| vanilla (=lp24，无 OCR) | 577 | 100% | 46.07 | — |
| recon_lp20 | 577 | 69.4% | 43.08 | — |
| recon_lp16 | 577 | 44.4% | 38.92 | — |
| recon_lp12 | 577 | 25.0% | 30.57 | — |
| recon_lp8 | 577 | 11.1% | 19.57 | — |
| dctdown_g14 | 197 | 34.0% | 31.36 | idea3-192 **45.08** / VZ-192 44.53 |
| dctdown_g11 | 122 | 21.0% | 23.96 | idea3-128 **43.87** / VZ-128 43.82 |
| dctdown_g8 | 65 | 11.1% | 15.71 | idea3-64 **42.01** / VZ-64 41.95 |

四条结论（按无 OCR 参照修正）：

1. **文字信息集中在特征图高频**：一个 token 都不删，lp20（截掉最高 4 个频带，保留 69% 系数）
   掉 3.0pt，lp16 掉 7.2pt，lp8 掉 26.5pt——而 idea3 剪到 64 个 token 才掉 4.1pt。
   **保留全部 token 的 lp16（38.9）差于只留 64 token 的剪枝（42.0）**：频域截断比删 89%
   的 token 更伤文字，能量与任务信息在文字区严重脱钩。
2. **低频压缩全面且大幅劣于 hard pruning**：同预算下 dctdown 比 idea3/VZ 低 13~26pt。
   "低频重建更适合压缩"对 TextVQA 不成立，且是方向性错误。
3. **分数跟随频率截止而非 token 数**：dctdown_g14（197 tok，14² 系数）31.4 ≈
   recon_lp12（577 tok，12² 系数）30.6——破坏来自频域截断本身，与 token 削减无关。
4. lp8 的 19.6 与 idea1-64 的 19.0 几乎重合（同协议）——idea1 的 TextVQA 崩溃本质就是
   把高频文字信息当冗余丢弃。高频残差可作细节似然信号，但见 §9.1：已无缺口可救。

### 9.1 主实验（2026-07-08，textvqa_val 全量，全部无 OCR 协议）

detail_mmr 消融 + 归因实验 + VZ 式 contextual 嫁接，与同协议基线同表
（vanilla 46.07；VZ = `noocr_vz*` 重跑值）：

| 策略 \ TextVQA | K=192 | K=128 | K=64 |
|---|---:|---:|---:|
| VisionZip（重跑，无 OCR） | 44.53 | 43.82 | 41.95 |
| idea3（λ=0.5 MMR） | **45.08** | 43.87 | 42.01 |
| λ=0（纯 CLS top-K） | 44.68 | 44.33 | **42.54** |
| idea5 detail_mmr（γ=1, local_var） | 44.52 | 43.69 | 42.07 |
| idea5 detail_mmr（γ=1, dct） | 44.73 | 43.75 | 42.23 |
| idea5 detail_mmr（γ=0.5, local_var） | 44.73 | — | — |
| idea3 + VZ 式 contextual（K−C 选 + C 合并，C=30/20/10） | 44.57 | 43.48 | 41.52 |
| λ=0 + VZ 式 contextual（= VZ 复刻，张量级等价已验证） | — | — | 41.96 |
| idea3 + 朴素合并（并入最近保留 token） | — | 43.19 | 39.97 |
| λ=0 + 朴素合并 | — | 43.54 | 41.00 |

（logs 见 `docs/idea5/logs/` 对应目录；POPE@K=64：idea3+contextual 86.32 ≈ idea3 纯剪枝 86.27。）

**结论：idea5 关闭，方向转移。**

1. **细节门控无增益**：7 个配置与 idea3 全部在 ±0.7pt 内。原因不是门控无效，而是
   **前提不存在**——协议修正后，idea3 的 TextVQA 与 VisionZip 持平（45.08 vs 44.53@192），
   距 vanilla 仅 1.0~4.1pt，本就没有"被 MMR 错杀 12pt"的缺口（§9.2）。
2. **合并不是必需组件**：VZ 式 contextual 嫁接无增益（且张量级复刻证明 VZ 自身在无 OCR
   协议下也只有 42 量级）；朴素合并单调有害。"纯剪枝范式天花板"不成立。
3. **修正后的真实格局**（详见 [idea_summary.md](../idea_summary.md) 修正节）：
   idea3 纯剪枝 Avg%van 三档全部超过 VisionZip（97.75/96.54/93.73 vs 97.41/95.94/92.93），
   idea3-346 初筛 99.26% 近无损（含 TextVQA）。**主线成果就是 idea3 本身**；
   后续工作重心：把 idea3 与 VisionZip 的 6-benchmark 对比在统一协议下做扎实
   （含 λ 敏感性、效率测量、其余基线 FasterVLM/PruMerge），而非继续修补文字问题。

### 9.2 协议断裂记录（根因，2026-07-08）

lmms-eval 于 07-02（vanilla/VZ 基线）与 07-07（idea1–4）之间升级至 0.3.0，textvqa prompt
丢失 "Reference OCR token:" 行（带提示 vanilla 58.27 → 无提示 46.07，差 ~12pt）；其余 5 个
benchmark 的 prompt 逐一比对未变。排查链：VZ 复刻仅 41.96 → 张量级 A/B 证明复刻与 VZ 逐位
一致 → 重跑官方 VZ-64 也仅 42.7@first-500 → 新老日志 prompt 对比锁定 OCR 行。判别协议：
看 samples jsonl 的 `input` 是否含 "Reference OCR token"。07-07 前的 textvqa 数字一律弃用，
统一改用 `docs/idea5/logs/noocr_*` 基线。
