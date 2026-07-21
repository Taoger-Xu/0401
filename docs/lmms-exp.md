# 用 lmms-eval 框架复现 VisionZip（计划）

参考 `/home/jk/work/paper/CLSE/docs/eval.md` 的思路：用 lmms-eval 统一评测平台驱动 LLaVA 模型，数据全部走 HuggingFace 本地缓存，一条命令跑多个 benchmark，不依赖 LLaVA 自带脚本那套本地 JSONL + 图像目录结构。这套流程和 [docs/exp.md](exp.md)（LLaVA 自带脚本路线）是两条独立、互不依赖的复现路径。

## 1. 为什么能直接复用 CLSE 的 lmms-eval，但不能复用它的 llava/transformers

- `/home/jk/work/paper/CLSE/lmms-eval` 是标准的 [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) 框架代码（CLSE 没有改动 lmms-eval 本身，只改了自己的 `llava` 包和 `transformers-4.37.2`），可以直接拿来装。
- CLSE 的 `llava` 包（`CLSE/LLaVA1.5/llava`）和 `transformers-4.37.2`（`CLSE/LLaVA1.5/transformers-4.37.2`）被他们自己的剪枝方法（`clse_model.py`、`stage0_prune.py`）修改过，直接复用可能和 VisionZip 的 monkey patch 冲突。所以：
  - lmms-eval **框架代码**复用 CLSE 的（`pip install -e CLSE/lmms-eval`）；
  - **llava 包和 transformers** 继续用 [docs/exp.md](exp.md) 里已经验证过的 `llava_visiPruner` 环境（原版 LLaVA + 标准 `transformers==4.37.2`），不用 CLSE 的版本。

## 2. lmms-eval 如何驱动 VisionZip：monkeypatch 挂钩

lmms-eval 内置的 `--model llava` 封装（`lmms_eval/models/simple/llava.py`）在初始化时直接：
```python
from llava.model.builder import load_pretrained_model
...
load_pretrained_model(pretrained, None, model_name, device_map=..., **llava_model_args)
```
CLSE 的做法是把剪枝开关做成 `llava` 包内部读环境变量（`PRUNE`/`RETAIN_TOKEN`），因为剪枝逻辑本来就写在他们自己的 `llava/model/language_model/clse_model.py` 里。VisionZip 不一样：`visionzip()` 是在模型加载**之后**对已加载模型对象做一次性 monkey patch（改 `CLIPVisionTower.forward`、`LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal` 等类方法），不是在模型类内部读环境变量。

要在不改 lmms-eval、不改 VisiPruner 的 `llava` 包的前提下接入，新增了 [eval/lmms_eval_entry.py](../eval/lmms_eval_entry.py)：

```python
import llava.model.builder as builder
_orig = builder.load_pretrained_model

def _patched(*args, **kwargs):
    tokenizer, model, image_processor, context_len = _orig(*args, **kwargs)
    dominant = os.environ.get("VZ_DOMINANT")
    if dominant is not None:
        from visionzip import visionzip
        model = visionzip(model, dominant=int(dominant), contextual=int(os.environ.get("VZ_CONTEXTUAL", "0")))
    return tokenizer, model, image_processor, context_len

builder.load_pretrained_model = _patched          # 在 lmms_eval 导入 llava 之前打补丁
from lmms_eval.__main__ import cli_evaluate
cli_evaluate()                                      # 和 `python -m lmms_eval` 行为一致
```

用这个脚本代替 `python -m lmms_eval` 作为 `accelerate launch` 的入口，其余参数（`--model llava --model_args ... --tasks ... --batch_size 1 --output_path ... --log_samples`）完全不变。这样 `VZ_DOMINANT`/`VZ_CONTEXTUAL` 环境变量的使用方式和 CLSE 的 `PRUNE`/`RETAIN_TOKEN` 是同一模式，未设置 `VZ_DOMINANT` 时就是 vanilla 基线。

## 3. 环境准备

复用 `llava_visiPruner` 环境（已经装好 vanilla `llava` + `transformers==4.37.2` + `visionzip`，见 exp.md），额外装 lmms-eval：

```bash
conda activate llava_visiPruner
pip install -e /home/jk/work/paper/CLSE/lmms-eval

# lmms-eval 的依赖会把 transformers/tokenizers/numpy 升级到不兼容版本，装完后按 CLSE README
# 的方法把这三个包摁回 VisionZip/LLaVA 验证过的版本（accelerate 可以留新版本，纯框架编排用，
# 不影响模型行为）：
pip install "transformers==4.37.2" "tokenizers==0.15.1" "numpy<2.0.0"
```

验证（已经跑通）：
```bash
python -c "import llava, lmms_eval, visionzip, transformers; print(transformers.__version__)"
# transformers 4.37.2 / tokenizers 0.15.1 / numpy 1.26.4 / accelerate 1.14.0（新版本，无影响）
```

## 4. 数据集

沿用 CLSE 的做法，全部走本机已缓存好的 HuggingFace datasets 缓存，不需要联网下载：
```bash
export HF_DATASETS_CACHE=/home/jk/datasets
export HF_DATASETS_OFFLINE=1
```
本机 `/home/jk/datasets` 下已经有 GQA、MME、POPE、ScienceQA、TextVQA、VizWiz-VQA、MMBench(EN/CN dev)、OCRBench 对应的 lmms-lab / echo840 格式缓存，和 CLSE 用的是同一份数据。

## 5. Task 列表（对齐 CLSE）

```
TASKS=gqa,mmbench_en_dev,mmbench_cn_dev,mme,pope,scienceqa_img,textvqa_val,vizwiz_vqa_val,ocrbench
```

## 6. MMBench 打分是否需要 OpenAI Key

看了 lmms-eval 的 `tasks/mmbench/en_utils.py` / `mmbench_evals.py`：默认 `eval_method="openai"`，但 `extract_answer_from_item` 会先用 `can_infer()` 做纯规则的答案提取（正则找 "A/B/C/D" 单字母），LLaVA 在 MMBench 提示里已经要求"直接回答选项字母"，绝大多数回答能被规则直接命中，**不会真的走到 GPT 调用这一步**。只有极少数模型没有干净给出单个字母的样本才会尝试调 GPT——因为没配置真实 `OPENAI_API_KEY`，这些样本会在重试耗尽后走"随机猜一个字母"的兜底逻辑，不会报错卡住。这与 CLSE 自己文档里"需要 GPT 评分，实际调用的是本地逻辑"的说明一致。

结论：**不需要等 OpenAI API key 就能跑完整套 lmms-eval 评测**，只是极少数 MMBench 边缘样本的分数会有一点随机噪声（可忽略）。

## 7. 运行方式

新增 [scripts/lmms_eval.sh](../scripts/lmms_eval.sh)，用法和 CLSE 的 `llava_lmms_eval.sh` 一致，只是把 `PRUNE=True RETAIN_TOKEN=<K>` 换成 `VZ_DOMINANT`/`VZ_CONTEXTUAL`：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/lmms_eval.sh vanilla
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/lmms_eval.sh vz64    # dominant=54  contextual=10
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/lmms_eval.sh vz128  # dominant=108 contextual=20
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/lmms_eval.sh vz192  # dominant=162 contextual=30
```

结果输出到 `logs/lmms-eval/<config>/`（JSON + `--log_samples` 明细），和 CLSE 的 `logs/` 布局一致。

## 8. 已验证的冒烟测试

用 `--tasks pope --limit 8` 跑通了 vz64 配置的最小验证：模型加载 → VisionZip patch 生效 → 本地 HF 缓存离线读取 POPE → 推理 → 出分，全链路无报错，输出了 `pope_accuracy` 等 5 个指标。确认可以放心跑全量。

## 9. 下一步

1. 对 vanilla / vz64 / vz128 / vz192 四个配置各跑一遍第 7 节的命令（可以分配到不同 GPU 并行跑）；
2. 汇总 9 个 benchmark 分数，整理成和 CLSE `docs/eval.md`、以及本仓库 [docs/result.md](result.md) 类似的对比表，写入 `docs/lmms.md`；
3. 和 VisionZip 论文 Table 1 数值、以及 [docs/result.md](result.md) 里 LLaVA 自带脚本跑出来的结果做交叉验证（两套框架、同一模型、同一 VisionZip 配置，理论上分数应该很接近，除了 MMBench 走的是不同评测路径）。
