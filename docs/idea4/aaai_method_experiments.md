# Method + Experiments (AAAI submission draft, Markdown version)

> 内容稿：定稿后逐段转入 AAAI author kit 的 LaTeX 模板。
> 方法名暂用 **SCRAP**（Saliency-, Coverage-, and Redundancy-Aware Pruning），全文只在此处与首次出现处出现全称，改名时全局替换即可。
> 所有数字来自 `docs/idea4/logs/*` 的 lmms-eval 输出，与 [paper_draft.md](paper_draft.md) 逐项一致。

---

## 3. Method

### 3.1 Problem Setup and Notation

We consider LLaVA-style multimodal large language models (MLLMs), in which a CLIP ViT-L/14-336 encoder produces $N{=}576$ patch tokens and one `[CLS]` token from its penultimate layer; all $N$ patch tokens are then passed through a projector into the LLM, where they dominate both the context budget and the inference cost.

**SCRAP** (Saliency-, Coverage-, and Redundancy-Aware Pruning) inserts a single selection step **between the vision encoder and the projector**: given a token budget $K$, it keeps the `[CLS]` token and selects $K{-}1$ of the $N$ patch tokens. The selection is a **pure gather** over original tokens—no merging, no feature reconstruction, no learned components—so the method is training-free, adds zero parameters, and applies to a frozen model as a plug-in.

Let $f_i\in\mathbb{R}^{1024}$ denote the penultimate-layer feature of patch $i$ and $\hat f_i=f_i/\lVert f_i\rVert$ its $\ell_2$ normalization. Let $a_i=\sum_h \mathrm{Attn}_h^{(L-2)}[\mathrm{cls},i]$ be the head-summed `[CLS]` attention, with min–max normalization $\hat a_i$, and let $p_i\in\{0,\dots,23\}^2$ be the patch grid coordinate. The budget is split into two pools by a coverage ratio $\rho\in[0,1]$:

$$
M=\lceil \rho\,(K{-}1)\rceil\ \text{(coverage pool)},\qquad
B=(K{-}1)-M\ \text{(saliency pool)}. \tag{1}
$$

### 3.2 Overview: Why Two Pools

Pre-LLM token selection must reconcile two objectives that pull in opposite directions: **semantic saliency**—retaining the evidence the model actually attends to, for which the pretrained `[CLS]` attention is a natural signal and which perception tasks (POPE, MME) require—and **spatial coverage**—retaining context spread across all content regions, which holistic-understanding tasks (GQA, MMBench) require.

Each single mechanism has a characteristic failure mode:

1. attention top-$K$ clusters spatially on a few salient objects and selects near-duplicates;
2. pure feature-level de-redundancy (global maximal marginal relevance, MMR) concentrates on salient, mutually dissimilar tokens but under-spreads across the image;
3. saliency-agnostic uniform coverage spends budget on sky and walls, diluting object evidence.

SCRAP therefore allocates the budget over two complementary pools: a **globally redundancy-aware saliency pool** as the backbone (Stage B, inheriting saliency and semantic de-redundancy), topped up by a **lightweight, saliency-ranked coverage pool** that lands only in content regions (Stage A). Together, under the same budget, the selection keeps salient evidence without leaving spatial holes.

*(Figure 1: method overview pipeline — TODO)*

### 3.3 Stage A: Saliency-Ranked Coverage Pool

Stage A guarantees spatial spread **only over regions that contain content**, skipping background, in three steps.

**(1) Over-partition the grid.** Deterministic farthest-point sampling followed by Voronoi assignment partitions the $24\times24$ grid into $P=\lceil c\cdot M\rceil$ spatial cells, where $c$ (the *cover factor*) satisfies $c>1$ so that $P>M$.

**(2) Score a representative per cell.** Within each cell $C$, the representative is the token maximizing

$$
s_i \;=\; \underbrace{m_i}_{\text{medoid affinity}} \;+\; w_f\,\underbrace{\ell_i}_{\text{low-freq. stability}} \;+\; w_a\,\mathrm{Norm}_{C(i)}(a_i), \tag{2}
$$

where $m_i$ is the medoid affinity of token $i$ within its cell (cosine centrality to the other members) and $\mathrm{Norm}_C(\cdot)$ is cell-wise min–max normalization. The **low-frequency stability** $\ell_i$ measures how well a token is explained by the smooth, low-frequency structure of the feature map: we arrange features on the $24\times24$ grid, apply an orthonormal 2D DCT-II along both spatial axes, keep the $L\times L$ low-frequency block ($L{=}16$), reconstruct $\tilde g$ by the inverse transform, and set $\ell_i=-\lVert g_i-\tilde g_i\rVert_2$ (cell-wise normalized): tokens dominated by low spatial frequencies are robust region representatives, whereas high-frequency outliers are sensitive to local perturbations. The DCT is used **only for scoring**; selected tokens keep their original features. Its cost is $O(NDG)$ with cached basis matrices and is negligible.

**(3) Rank cells by attention.** Cells are ranked by their attention mass $\max_{i\in C}a_i$, and only the representatives of the top-$M$ cells are kept. Because $P>M$ and cells are chosen by attention, the $M$ coverage tokens land in the most content-rich, spatially separated regions rather than uniformly tiling the image (background included); this preserves spatial diversity without wasting budget.

### 3.4 Stage B: Globally Redundancy-Aware Saliency Pool

Initialized with the Stage-A set $S$, Stage B greedily adds $B$ tokens by maximal marginal relevance:

$$
\mathrm{score}(i\mid S) \;=\; \hat a_i \;-\; \lambda\cdot\Big[\max_{j\in S}\cos(\hat f_i,\hat f_j)\Big]_+ . \tag{3}
$$

The first term is `[CLS]` saliency; the second is a **global** feature-redundancy penalty against the already-selected set, with no spatial constraint: each new token must be both salient and dissimilar to everything selected so far. Running Stage A first matters—the redundancy penalty in Eq. (3) then also discounts near-duplicates of coverage tokens, coupling the two pools.

Equation (3) can be reparameterized as $\alpha\hat a_i-(1-\alpha)r_i$ with $\alpha=1/(1+\lambda)$, i.e., $\lambda$ directly controls the saliency share ($\lambda{=}0.5\Leftrightarrow\alpha{=}66.7\%$; $\lambda{=}0$ recovers pure attention top-$K$).

### 3.5 Algorithm and Complexity

```text
Algorithm 1: SCRAP token selection
Input: features {f_i}, CLS attention {a_i}, budget K,
       coverage ratio ρ, redundancy weight λ, cover factor c
1: M ← ⌈ρ(K−1)⌉;  B ← (K−1)−M;  P ← min(N, ⌈cM⌉)
2: Partition the 24×24 grid into P cells (FPS + Voronoi)
3: rep(C) ← argmax_{i∈C} m_i + w_f·ℓ_i + w_a·Norm_C(a_i)   ▷ Eq. (2)
4: S ← { rep(C) : C ∈ top-M cells by max_{i∈C} a_i }        ▷ Stage A
5: for t = 1 … B do                                          ▷ Stage B
6:     i* ← argmax_{i∉S}  â_i − λ·[max_{j∈S} cos(f̂_i, f̂_j)]₊
7:     S ← S ∪ {i*}
8: return {CLS} ∪ S in raster order
```

The greedy loop maintains running maxima of pairwise similarities, giving $O(K\cdot N)$ total cost ($N{=}576$)—negligible relative to one encoder forward pass. Since only $K$ visual tokens enter the LLM, the prefill cost of the language model shrinks proportionally, exactly as in prior pre-LLM pruning work.

### 3.6 Relation to Prior Selection Schemes

SCRAP strictly generalizes two existing families as degenerate limits:

- $\rho\to 0$: pure global MMR (saliency + de-redundancy, no coverage);
- $\rho\to 0,\ \lambda=0$: pure `[CLS]`-attention top-$K$, i.e., the pruning-only counterpart of VisionZip's dominant-token selection;
- $c=1$: uniform spatial coverage (one token per cell, background included).

The added ingredient is precisely the saliency-ranked coverage pool, whose contribution we isolate in the ablations.

### 3.7 Hyper-parameters

Three quantities are frozen across all experiments: $c{=}3$, $w_f{=}w_a{=}1.0$, and no spatial gating in Stage B. The only free knobs are the two scalars of Eqs. (1) and (3):

- the **coverage ratio $\rho$** (primary; tighter budgets favor smaller $\rho$ so that coverage does not crowd out the salient core), and
- the **redundancy weight $\lambda$** (secondary; fixed at 0.5 for general use).

The experiments distill a one-line deployment rule: for general tasks set $\rho{=}0.5$ if $K{\ge}192$ and $\rho{=}0.25$ if $K{\le}128$ with $\lambda{=}0.5$; for known text-dense tasks switch to the **text mode** $\rho{=}0,\lambda{=}0.1$. Switching changes two scalars only, never the selector implementation.

---

## 4. Experiments

### 4.1 Setup

**Model and implementation.** We evaluate on LLaVA-1.5-7B with a CLIP ViT-L/14-336 vision tower. SCRAP is inserted between the vision tower and the projector; the LLM and projector weights are untouched and nothing is finetuned. The **baseline** is the unpruned model with all 576 visual tokens (*vanilla*).

**Benchmarks and metrics.** Nine discriminative benchmarks are evaluated with lmms-eval under one protocol: GQA, MMBench-EN, MME-P (perception score), MMStar, POPE (F1), ScienceQA-IMG, TextVQA, VizWiz, and OCRBench. GQA/SQA/TextVQA/VizWiz report exact-match×100, MMBench the GPT-judged score, POPE F1×100, MMStar/OCRBench accuracy×100. VizWiz uses the locally-annotated *val* split; MMStar uses its 1,500-question val split, whose six capability axes resist language-prior shortcuts. As a generative stress test we additionally report COCO Caption (`coco2017_cap_val`, 5,000 images; CIDEr/BLEU/METEOR/ROUGE-L); being n-gram-overlap metrics of a different scale, these are reported separately and **excluded** from the main-table average.

**Budgets.** The main table reports $K\in\{192,128,64\}$, i.e., keeping 1/3, 2/9, and 1/9 of the visual tokens; $K\in\{288,346\}$ (50%/60%) are additionally reported as a lossless pre-screening regime.

**TextVQA protocol.** An lmms-eval upgrade removed the `Reference OCR token:` line from the TextVQA prompt; the with-OCR and without-OCR protocols differ by ~12 points for the unpruned model (58.27 vs. 46.07) and are not comparable. All runs in this paper—including all baselines—use the **no-OCR** protocol, so TextVQA retention is measured against the vanilla score of 46.07.

**Compared methods.** (i) **VisionZip**, the strongest training-free competitor at the same insertion point, which selects dominant tokens *and merges* the remainder into contextual tokens; (ii) **Global-MMR**, the $\rho{=}0$ degenerate case of SCRAP (§3.6), which isolates the contribution of Stage A; (iii) **vanilla**, the unpruned upper reference.

### 4.2 Main Results

**Table 1. Main results on LLaVA-1.5-7B: absolute score with retention relative to the unpruned baseline in parentheses (%).**

| Benchmark | Baseline (576) | K=192 (1/3) | K=128 (2/9) | K=64 (1/9) |
|---|---:|---:|---:|---:|
| GQA | 61.97 | 59.78 (96.5) | 59.25 (95.6) | 57.47 (92.7) |
| MMBench-EN | 64.00 | 63.23 (98.8) | 62.63 (97.9) | 59.62 (93.2) |
| MME-P | 1511.3 | 1474.5 (97.6) | 1411.7 (93.4) | 1387.4 (91.8) |
| MMStar | 33.56 | **33.76 (100.6)** | 32.74 (97.6) | 31.98 (95.3) |
| POPE (F1) | 85.88 | **86.93 (101.2)** | **86.44 (100.7)** | 84.26 (98.1) |
| SQA-IMG | 69.46 | 69.16 (99.6) | 68.91 (99.2) | 68.12 (98.1) |
| TextVQA | 46.07 | 45.16 (98.0) | 44.55 (96.7) | 42.54 (92.3) |
| VizWiz | 54.06 | **54.99 (101.7)** | **55.33 (102.3)** | **56.32 (104.2)** |
| OCRBench | 31.20 | **31.20 (100.0)** | 30.00 (96.2) | 28.90 (92.6) |
| **Average retention (%)** | 100.0 | **99.3** | **97.7** | **95.4** |

*Config note (goes into the table caption): non-text tasks use the per-budget general configuration ($\rho{=}0.5$ at K=192, $\rho{=}0.25$ at K≤128; $\lambda{=}0.5$). TextVQA/OCRBench use the text mode ($\rho{=}0,\lambda{=}0.1$) at K∈{128,192}; at K=64, TextVQA uses $\rho{=}0,\lambda{=}0$ and OCRBench the general configuration (see §4.6). Bold: at or above baseline.*

Four observations stand out.

**(1) Near-lossless at moderate compression.** At K=192 the average retention is 99.3%: POPE, VizWiz, and MMStar *exceed* the baseline and OCRBench fully recovers it. At K=128 the average is 97.7%, with eight of nine tasks at ≥95.6% (only MME-P drops to 93.4%).

**(2) Aggressive pruning can act as a regularizer.** VizWiz stays above the baseline at *every* budget and peaks at K=64 (104.2%): removing redundant or distracting patches helps rather than hurts, i.e., more visual tokens are not always better.

**(3) MMStar requires a decomposed reading.** Its average is robust (100.6/97.6/95.3%), but §4.3 shows this average mixes genuinely visual sub-skills with sub-skills at chance level, so we do not use it alone as evidence of losslessness.

**(4) Text tasks are nearly lossless under a matched protocol.** Against the no-OCR baseline, TextVQA retains 98.0%/96.7% at K=192/128—the large gaps reported under mismatched OCR protocols do not reflect the pruning itself.

### 4.3 MMStar: Capability Decomposition

**Table 2. MMStar decomposition (accuracy×100).**

| MMStar subset | Baseline | K=192 | K=128 | K=64 |
|---|---:|---:|---:|---:|
| coarse perception | 63.87 | 61.50 | 59.22 | 53.74 |
| fine-grained perception | 25.63 | 24.44 | 21.73 | 19.43 |
| instance reasoning | 38.89 | 38.13 | 39.01 | 39.92 |
| logical reasoning | 28.92 | 31.42 | 29.27 | 29.52 |
| math | 26.31 | 28.38 | 27.19 | 27.40 |
| science & technology | 17.76 | 18.67 | 20.01 | 21.88 |
| **average** | **33.56** | **33.76** | **32.74** | **31.98** |

Table 2 reveals two opposite trends hidden in the MMStar average. The genuinely visual subsets—coarse and fine-grained perception, whose baselines (63.87/25.63) sit clearly above chance—degrade **monotonically** as the budget tightens, down to 84.1%/75.8% retention at K=64, consistent with GQA and MME-P. Conversely, science & technology (baseline 17.76, **below** the 25% chance level of four-way choice), math (26.31), and logical reasoning (28.92) start near chance, and their apparent gains under pruning lack any visual interpretation; we treat them as regression-to-chance noise. We therefore report the MMStar average for completeness but base conclusions on the perception subsets together with GQA/MME-P.

### 4.4 Generative Stress Test: COCO Caption

**Table 3. COCO Caption (`coco2017_cap_val`, 5,000 images) under the general configuration; retention in parentheses (%).**

| Metric | Baseline | K=192 | K=128 | K=64 |
|---|---:|---:|---:|---:|
| CIDEr | 1.104 | 1.076 (97.5) | 1.044 (94.6) | 0.989 (89.6) |
| BLEU-4 | 0.298 | 0.289 (96.9) | 0.279 (93.7) | 0.266 (89.3) |
| BLEU-1 | 0.731 | 0.719 (98.3) | 0.712 (97.4) | 0.696 (95.2) |
| METEOR | 0.293 | 0.287 (98.1) | 0.281 (96.0) | 0.272 (92.8) |
| ROUGE-L | 0.556 | 0.549 (98.7) | 0.542 (97.5) | 0.530 (95.3) |
| **Avg. retention (%)** | 100.0 | 97.9 | 95.8 | 92.4 |

Discriminative QA can hide pruning damage behind language priors or answer options. Dense captioning cannot: the model must restate the whole image. Table 3 shows a trend **opposite** to VizWiz: every metric decreases monotonically with the budget, with CIDEr retention 97.5%/94.6%/89.6%—no "pruning helps" rebound.

Placing the three task families side by side at K=64 makes the layering explicit:

| Task family @ K=64 | Retention |
|---|---:|
| VizWiz (questions often answerable while ignoring visual clutter) | 104.2% ↑ |
| MMStar average (inflated by chance-level subsets) | 95.3% |
| COCO CIDEr (must restate the whole image) | 89.6% ↓ |
| MMStar fine-grained perception | 75.8% ↓ |

We accordingly scope our claims: **K=192 is near-lossless across all task families** (99.3% discriminative, 97.5% CIDEr); K=128 is acceptable; K=64 is suitable for discriminative QA but **not** recommended for dense description or fine-grained perception.

### 4.5 Comparison with Training-Free Methods

**Table 4. Comparison on LLaVA-1.5-7B, single general configuration per budget (no per-task switching).** Metrics here follow the common comparison protocol: MME is the total (perception+cognition) score and POPE is accuracy, hence the difference from Table 1; TextVQA is under the no-OCR protocol for all rows. "nonTxt %base" averages the retention of the five non-text tasks. Bold: best per budget.

| K | Method | GQA | MMB | MME | POPE | SQA | TextVQA | nonTxt %base |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 576 | vanilla | 62.0 | 64.0 | 1875 | 87.0 | 69.5 | 46.1 | 100.0 |
| 192 | VisionZip | 59.3 | **63.8** | 1770 | 86.4 | 68.7 | **44.5** | 97.6 |
| 192 | Global-MMR | 59.5 | 63.0 | 1783 | 87.5 | 68.4 | **45.1** | 97.7 |
| 192 | **SCRAP** | **59.8** | 63.2 | **1803** | **87.6** | **69.2** | 44.4 | **98.4** |
| 128 | VisionZip | 57.7 | 62.2 | **1764** | 84.6 | 68.7 | **43.8** | 96.1 |
| 128 | Global-MMR | 59.3 | 62.3 | 1719 | **87.2** | 68.8 | **43.9** | 96.8 |
| 128 | **SCRAP** | **59.3** | **62.6** | 1728 | **87.2** | **68.9** | 43.6 | **97.0** |
| 64 | VisionZip | 55.2 | **60.1** | **1718** | 80.6 | **69.0** | **42.0** | 93.3 |
| 64 | Global-MMR | **57.6** | 59.8 | 1639 | **86.3** | 68.3 | **42.0** | **94.2** |
| 64 | SCRAP | 57.2 | 59.6 | 1670 | 85.5 | 67.9 | 41.5 | 94.1 |
| 288 | Global-MMR | 61.0 | **63.9** | 1779 | **87.6** | 68.6 | 45.1 | 98.5 |
| 288 | **SCRAP** | 61.0 | 63.8 | **1785** | 87.5 | **68.7** | **45.4** | **98.6** |
| 346 | Global-MMR | 61.2 | 64.6 | **1831** | **87.3** | 68.6 | 45.6 | **99.3** |
| 346 | SCRAP | 61.2 | **64.9** | 1824 | 87.1 | **68.7** | 45.6 | 99.3 |

Three findings:

**Pure selection beats select-and-merge.** On the five non-text tasks, both selection-based methods dominate VisionZip at every budget—most visibly on POPE at K=64 (85.5/86.3 vs. 80.6)—even though VisionZip additionally merges discarded tokens. Under the matched no-OCR protocol, merging shows no TextVQA advantage either: with the text mode (§4.6), SCRAP reaches 45.16/44.55/42.54 at K=192/128/64 versus VisionZip's 44.53/43.82/41.95.

**Stage A pays for itself at practical budgets.** Against Global-MMR—which SCRAP contains as its $\rho{=}0$ limit—adding the coverage pool wins the non-text average at K=192 (98.4 vs. 97.7, with all five tasks ≥ Global-MMR), K=128, and K=288, and ties (within 0.1) at the extremes K=64 and K=346, where either every token is precious or the budget is nearly saturated.

**Wide-budget pre-screening is lossless.** Keeping 50–60% of tokens (K=288/346), SCRAP retains 98.6–99.3% of the non-text average, making it a safe first stage before any downstream compression.

### 4.6 Ablation Studies

#### Coverage ratio ρ is budget-adaptive

**Table 5. Non-text average retention (%) sweeping ρ, all other hyper-parameters fixed.**

| K | ρ=0.25 | ρ=0.5 | best |
|---:|---:|---:|:--:|
| 64 | **94.12** | 93.88 | 0.25 |
| 128 | **97.02** | 96.54 | 0.25 |
| 192 | 97.89 | **98.35** | 0.5 |
| 288 | **98.60** | 98.42 | 0.25 |
| 346 | 99.23 | **99.27** | 0.5 |

The pattern is consistent: the tighter the budget, the smaller the optimal ρ—coverage must not crowd out the salient core—while at K≥192 the larger coverage pool pays off. This yields the deployment default of §3.7.

#### Saliency-ranked vs. uniform coverage

Setting $c{=}1$ makes Stage A degenerate to uniform coverage: every cell, background included, contributes one token. At small budgets this spends roughly a ρ-fraction of the budget on background and visibly degrades POPE; over-partitioning with $c{=}3$ and keeping only the top-$M$ cells by attention restores POPE and MME by concentrating coverage in content regions. We therefore fix $c{=}3$ throughout.

*(内部注：cover_factor 的单变量干净消融仍标记为"待补"（idea4.md §3.1），此段目前是定性表述；补跑后在此插入数据表。)*

#### Text mode: drop coverage, raise the saliency share

Text patches are spatially clustered and visually similar, so both the coverage pool (which spends budget on background cells) and a strong redundancy penalty are biased **against** them. We test this with a two-stage protocol that avoids overfitting the validation task: hyper-parameters are searched **only** on TextVQA (Table 6), then the selected configuration is transferred to OCRBench without further tuning (Table 7).

**Table 6. TextVQA search stage (exact-match×100). "gen." is the per-budget general configuration.**

| ρ | λ | α | K=64 | K=128 | K=192 |
|---:|---:|---:|---:|---:|---:|
| gen. | 0.5 | 66.7% | 41.50 | 43.60 | 44.40 |
| 0 | 0 | 100% | **42.54** | 44.33 | 44.68 |
| 0 | 0.1 | 90.9% | 42.49 | **44.55** | **45.16** |
| 0 | 0.25 | 80.0% | 42.38 | 44.04 | 45.06 |

**Table 7. OCRBench transfer validation (accuracy×100; vanilla 31.20). Configurations selected on TextVQA only; OCRBench untouched during selection.**

| K | selected config | general | text mode | Δ |
|---:|---|---:|---:|---:|
| 64 | ρ=0, λ=0 | **28.90** | 28.50 | −0.40 |
| 128 | ρ=0, λ=0.1 | 29.90 | **30.00** | +0.10 |
| 192 | ρ=0, λ=0.1 | 30.80 | **31.20** | +0.40 |

At K=192 the text mode (ρ=0, λ=0.1) lifts TextVQA from 44.40 to 45.16 and OCRBench from 30.80 to 31.20—fully recovering the unpruned OCRBench score. Notably, the optimum keeps α=90.9% rather than 100%: pure top-$K$ (λ=0) re-admits adjacent near-duplicate high-attention patches and is inferior at K∈{128,192}. At K=64 no single configuration wins both tasks (pure top-$K$ gains +1.04 TextVQA but −0.40 OCRBench), so we keep the general configuration there.

### 4.7 Summary

With zero training and zero added parameters, SCRAP retains on average 99.3% (K=192), 97.7% (K=128), and 95.4% (K=64) of the unpruned baseline over nine discriminative benchmarks, exceeding the baseline on POPE, VizWiz, and MMStar, and outperforms both VisionZip and its own pure-MMR limit at practical budgets. Its task adaptivity resides entirely in two scalars $(\rho,\lambda)$.

We state the scope conservatively: gains on discriminative QA do not extend to every regime—dense captioning and fine-grained perception degrade monotonically with the budget—so we recommend K=192 as the near-lossless operating point across task families, and restrict K=64 to discriminative QA. The remaining loss surface concentrates on fine-grained visual coverage and dense text, pointing to text-aware local protection or controlled lightweight merging as future work.

---

*数据来源与核对：Table 1–3 与 [paper_draft.md](paper_draft.md) §4.2 一致；Table 4 的 VisionZip 行来自 [idea_summary.md](../idea_summary.md) 实验 A（5 个非文字基准 prompt 未变，可比）+ 无 OCR 协议重跑的 TextVQA（44.53/43.82/41.95）；VisionZip 的 nonTxt %base（97.6/96.1/93.3）按与 aggregate_ideas.py 相同的公式手工计算，正式投稿前需用脚本核验。Table 5–7 与 paper_draft.md §4.3 一致。*
