# idea4 on LLaVA-1.5-13B

## 评测设置

- 模型：`llava-hf/llava-1.5-13b-hf`
- 视觉编码器：CLIP ViT-L/14@336，原始 patch 数为 576
- 方法：idea4 / Anchor-Cover
- 档位：K = 192、128、64（K 包含 1 个 CLS token）
- 通用配置：K=192 使用 $\rho=0.5$；K=128/64 使用 $\rho=0.25$；三档均使用
  $\lambda=0.5$、$\sigma=2.0$、`cover_factor=3`
- 推理：`lmms-eval`，batch size 1，零样本，完整验证集，不使用 `--limit`
- TextVQA：当前仓库统一的无 reference-OCR prompt 协议

## 九项 Benchmark 结果

### 第一阶段：GQA、POPE、ScienceQA-IMG、Q-Bench

| Benchmark | Baseline (576) | K=192 | K=128 | K=64 |
|---|---:|---:|---:|---:|
| GQA (exact match ×100) | 62.57 | 59.24 | 58.37 | 57.01 |
| POPE (F1 ×100) | 84.33 | 82.23 | 81.82 | 78.42 |
| ScienceQA-IMG (exact match ×100) | 71.59 | 71.84 | 71.54 | 71.54 |
| Q-Bench (`qbench_dev`, accuracy ×100) | 62.68 | 60.80 | 60.27 | 59.20 |

第一阶段严格使用 GPU 0–3，最多同时运行四个 13B 评测进程；完整产物位于
`docs/idea4/llava-13b/phase1/`。

第一阶段已全部完成，12 组运行均使用完整验证集，日志中未出现评测错误或 CUDA OOM。

### 后续九项主协议表

| Benchmark | K=192 | K=128 | K=64 |
|---|---:|---:|---:|
| GQA (exact match ×100) | 运行中 | 运行中 | 运行中 |
| MMBench-EN | 运行中 | 运行中 | 运行中 |
| MME-P | 运行中 | 运行中 | 运行中 |
| MMStar (accuracy ×100) | 运行中 | 运行中 | 运行中 |
| POPE (F1 ×100) | 运行中 | 运行中 | 运行中 |
| ScienceQA-IMG (exact match ×100) | 运行中 | 运行中 | 运行中 |
| TextVQA (exact match ×100) | 运行中 | 运行中 | 运行中 |
| VizWiz (exact match ×100) | 运行中 | 运行中 | 运行中 |
| OCRBench (accuracy ×100) | 运行中 | 运行中 | 运行中 |

## 产物位置

- K=192：`docs/idea4/llava-13b/k192/runs/`
- K=128：`docs/idea4/llava-13b/k128/runs/`
- K=64：`docs/idea4/llava-13b/k64/runs/`
- 调度日志：`logs/llava13b/`

> 表格将在三档全量评测完成后由结果 JSON 自动回填。
