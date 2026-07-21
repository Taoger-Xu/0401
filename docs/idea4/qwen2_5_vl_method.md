# Idea4 on Qwen2.5-VL

This adaptation applies Anchor-Cover after Qwen2.5-VL's 2x2 PatchMerger and
before insertion into the language-model sequence. Qwen has no CLS token, so
the saliency term is the attention received by each merged patch, averaged over
heads and query positions in the final full-attention vision block. Spatial
coverage uses each image's dynamic rectangular NaViT grid. Phase B is the same
global feature-MMR as idea4. Selected outputs are a pure gather of original
PatchMerger tokens; there is no reconstruction or additional merging.

The reproducible offline launcher is `scripts/qwen2_5_vl_idea4_eval.sh`. Full
lmms-eval results and samples are written beside this file in a directory named
`qwen2_5_vl_idea4_k*_rho*_lambda*`; console logs are written to the repository
root `logs/` directory.

## Results (2026-07-18, Qwen2.5-VL-7B)

Local `/home/jk/models/qwen2.5-vl-7b`, lmms-eval offline, `attn_implementation=eager`,
`batch_size=1`, fixed `rho=0.5, lambda=0.5, cover_factor=3.0`; only `K` varies.
Baseline is the *unmodified* model (`eval/qwen2_5_vl_vanilla_entry.py`, no pruning)
run under an identical protocol.

### Absolute scores

| Benchmark | Baseline | K=192 | K=128 | K=64 |
| --- | --- | --- | --- | --- |
| GQA | 58.21 | 57.57 | 56.65 | 54.30 |
| MME-P | 1653.48 | 1606.53 | 1590.92 | 1512.53 |
| MMStar | 58.27 | 57.72 | 55.24 | 51.35 |
| POPE (F1) | 86.48 | 85.99 | 85.02 | 83.54 |
| SQA-IMG | 80.91 | 80.57 | 79.18 | 78.14 |
| TextVQA | 71.24 | 66.86 | 63.22 | 54.91 |
| VizWiz | 66.77 | 65.63 | 64.42 | 62.13 |
| OCRBench | 77.80 | 65.60 | 58.00 | 45.00 |

### Retention relative to baseline

| Benchmark | Baseline | K=192 | K=128 | K=64 |
| --- | --- | --- | --- | --- |
| GQA | 100.0% | 98.9% | 97.3% | 93.3% |
| MME-P | 100.0% | 97.2% | 96.2% | 91.5% |
| MMStar | 100.0% | 99.1% | 94.8% | 88.1% |
| POPE (F1) | 100.0% | 99.4% | 98.3% | 96.6% |
| SQA-IMG | 100.0% | 99.6% | 97.9% | 96.6% |
| TextVQA | 100.0% | 93.9% | 88.7% | 77.1% |
| VizWiz | 100.0% | 98.3% | 96.5% | 93.1% |
| OCRBench | 100.0% | 84.3% | 74.6% | 57.8% |
| **Average** | **100.0%** | **96.3%** | **93.0%** | **86.8%** |

### Reading these numbers

- **K is not comparable to the LLaVA-1.5 table.** LLaVA-1.5 emits a fixed 576
  visual tokens, so K=192/128/64 are exactly 1/3, 2/9 and 1/9. Qwen2.5-VL uses
  dynamic-resolution NaViT, so the pre-pruning token count varies per image and
  a fixed K means a different ratio on every benchmark. Measured medians:
  **TextVQA ≈ 999 tokens** (K=64 keeps only 6.4%, harsher than LLaVA's 11.1%)
  but **OCRBench ≈ 274 tokens** (K=64 keeps 23.3%, *more generous* than LLaVA).
- **OCR-heavy tasks carry the loss, and budget ratio does not explain it.**
  OCRBench (84.3% → 57.8%) and TextVQA (93.9% → 77.1%) degrade far faster than
  the rest. For TextVQA an unusually harsh ratio is part of the story, but
  OCRBench collapses the most while receiving the *most* generous ratio of any
  benchmark here. So the driver is not "dense-text images produce more tokens" —
  reading text is simply intrinsically intolerant of dropping patches: text is
  information-dense and non-redundant, so a representative subset cannot stand in
  for the characters that were discarded.
- **Non-OCR tasks hold up well**, especially POPE and SQA-IMG (>96% even at
  K=64), so the selection is not broken — the budget is simply mismatched to
  Qwen's variable token count.
- **MMBench-EN is absent**: offline it falls back to the OpenAI API for answer
  extraction (401), so it cannot be scored here without an API key or a
  MMBench-server submission.

## Spatial-coverage ablation on text tasks (2026-07-20)

Hypothesis under test: in text-heavy images saliency already concentrates on the
text, so idea4's diversity terms spend budget on non-text regions and hurt OCR.
idea4 has two separate diversity knobs, ablated independently at K=64 on OCRBench:

| rho (spatial coverage) | lam (feature MMR) | OCRBench |
| --- | --- | --- |
| 0.50 | 0.50 | 45.00 (shipped config) |
| 0.25 | 0.50 | 45.40 |
| **0.00** | **0.50** | **46.30** (best) |
| 0.00 | 0.00 | 44.70 |

**The hypothesis holds for *spatial* diversity only.** Removing spatial coverage
helps monotonically (45.00 → 45.40 → 46.30), but also removing feature-MMR costs
1.6 points (46.30 → 44.70). Feature-space diversity is *earning* its budget;
spatial coverage is not. Best config for text: **rho=0, lam=0.5**.

### rho=0 vs rho=0.5, paired per-sample tests

| Task | K | rho=0.5 | rho=0.0 | Δ | paired p |
| --- | --- | --- | --- | --- | --- |
| OCRBench | 192 | 65.60 | 66.60 | +1.00 | 0.131 ns |
| OCRBench | 128 | 58.00 | 60.20 | +2.20 | 0.009 ** |
| OCRBench | 64 | 45.00 | 46.30 | +1.30 | 0.138 ns |
| TextVQA | 192 | 66.86 | 66.81 | −0.05 | 0.874 ns |
| TextVQA | 128 | 63.22 | 63.94 | +0.72 | 0.064 ns |
| TextVQA | 64 | 54.91 | 56.72 | +1.80 | 0.0001 *** |
| GQA (non-text) | 64 | 54.30 | 54.17 | −0.13 | 0.602 ns |
| MMStar (non-text) | 64 | 51.35 | 51.68 | +0.00 | 1.000 ns |

Corrected retention for the text tasks (rho=0): OCRBench 85.6 / 77.4 / 59.5 %
and TextVQA 93.8 / 89.8 / 79.6 % at K=192 / 128 / 64.

### What this does and does not establish

- **Effect is real but small, and only 2 of 6 text cells reach significance.**
  The gain grows as the budget tightens (largest at K=64–128), which fits the
  mechanism: when budget is scarce, spending half of it on spatial spread costs
  the most. At K=192 TextVQA shows no gain at all.
- **rho=0 is not a text-specific trade-off.** GQA and MMStar are unchanged
  (p=0.60, p=1.00), so dropping spatial coverage appears free elsewhere. That
  makes rho=0 a defensible global default on Qwen rather than a per-task hack —
  though the remaining non-text benchmarks have not been re-run at rho=0.
- **It does not close the OCR gap.** OCRBench at K=64 goes 45.0 → 46.3 against a
  77.8 baseline. Budget remains the dominant lever by a wide margin
  (K=64 → 128 → 192 gives 46.3 → 60.2 → 66.6 at rho=0). The diversity terms are
  a second-order effect; the first-order problem is that text is information-
  dense and non-redundant, so no subset-selection rule recovers discarded glyphs.

### Correctness note (important)

Results produced before 2026-07-18 for idea4-on-Qwen are **invalid** — the
pruning never executed and every run silently equalled vanilla (all K produced
bit-identical scores). They are archived under `docs/idea4/_invalid_pre_fix_0718/`.
Three bugs had to be fixed to make the method actually run under
transformers 4.57.6 / torch 2.12:

1. `_anchor_cover_rect` used `Tensor.minimum_` / `Tensor.maximum_`, which do not
   exist in torch — the selection raised on every call. Replaced with
   `torch.minimum/maximum(..., out=...)`.
2. After pruning, `cache_position` still described the unpruned length, so the
   prefill mismatched. It is now rebuilt to the pruned length.
3. `eval/qwen2_5_vl_idea4_entry.py` patched `transformers.Qwen2_5_VLForConditionalGeneration`,
   but lmms-eval's `models/simple/qwen2_5_vl.py` had already bound the *stock*
   class via `from transformers import ...`. The entry now rebinds the name on
   the wrapper modules themselves; without this the harness silently runs vanilla.

When changing this path, verify pruning actually fires (e.g. assert kept tokens
== K on a couple of samples) and that vanilla/K=192/K=64 give *different* scores.
Do not trust exit codes: lmms-eval catches a failed cross-GPU gather and still
exits 0, which silently drops a task's results (POPE, the largest gather, hit
this repeatedly) — verify the result JSON exists instead.
