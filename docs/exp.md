# VisionZip 本地复现实验指南

本文档说明如何在本机复现 VisionZip（LLaVA-1.5 系列）的评测结果。整体思路是：
**复用 `/home/jk/work/paper/VisiPruner` 已经装好的 LLaVA 评测环境和已下载好的评测数据，只在 VisionZip 仓库里新增"打了 VisionZip 补丁"的评测入口脚本**，不改动 VisiPruner 仓库本身。

## 0. TL;DR

| 需要的东西 | 直接复用 | 路径 |
| --- | --- | --- |
| conda 环境 | ✅ | `llava_visiPruner`（`llava==1.2.2.post1` 可编辑安装、`torch==2.1.2`、`transformers==4.37.2`） |
| LLaVA-1.5-7B 权重 | ✅ | `/home/jk/models/llava-v1.5-7b`（`config.json` 里 `mm_vision_tower` 已指向本地路径，完全离线） |
| CLIP 视觉塔 | ✅ | `/home/jk/models/clip-vit-large-patch14-336` |
| 评测数据（11 个 benchmark） | ✅ | `/home/jk/work/paper/VisiPruner/playground/data/eval/*`（已按 LLaVA 官方 [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) 流程准备好） |
| VisionZip 补丁 | 需要新增 | 本仓库 `eval/` 下的评测入口脚本（见第 3 节） |

## 1. 本机资源盘点

```text
模型：
  /home/jk/models/llava-v1.5-7b            # LLaVA-1.5-7B 权重（本地，非 HF hub id）
  /home/jk/models/clip-vit-large-patch14-336 # CLIP 视觉塔（llava-v1.5-7b 的 config.json 已指向此路径）
  /home/jk/models/llava-next-7b             # 如需复现 LLaVA-NeXT 结果可用
  /home/jk/models/llava-next-video-7b

评测数据（LLaVA 官方目录结构，已由 VisiPruner 项目下载/整理好）：
  /home/jk/work/paper/VisiPruner/playground/data/eval/
    gqa/  MME/  textvqa/  pope/  mmbench/  mmbench_cn/
    scienceqa/  vqav2/  seed_bench/  mm-vet/  vizwiz/
    llava-bench-in-the-wild/  qbench/

conda 环境：
  llava_visiPruner   # python3.10, torch2.1.2, torchvision0.16.2, transformers4.37.2, accelerate0.21.0
                      # llava 包以 `pip install -e .` 方式指向 /home/jk/work/paper/VisiPruner

GPU： 8 × NVIDIA A100-SXM4-80GB（单卡即可跑 7B 模型评测）
```

VisionZip 论文 Table 1（LLaVA-1.5）用的 11 个 benchmark（GQA / MME / TextVQA / POPE / MMBench / MMBench-CN / SQA-IMG / VQAv2 / VizWiz / MM-Vet / SEED-Bench）在 VisiPruner 的 `playground/data/eval` 下**全部已经就绪**（图片、问题文件、打分脚本都在），不需要重新下载。

## 2. 为什么可以直接复用 VisiPruner 的 llava 安装

VisionZip 不是一个独立的推理框架，而是通过 monkey patch 替换官方 [LLaVA](https://github.com/haotian-liu/LLaVA) 源码里的几个方法（见 [visionzip/main.py](../visionzip/main.py)）：

- `transformers.models.clip.modeling_clip.CLIPEncoderLayer.forward` / `CLIPAttention.forward`
- `llava.model.multimodal_encoder.clip_encoder.CLIPVisionTower.forward`
- `llava.model.llava_arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal`

检查过 `/home/jk/work/paper/VisiPruner/llava/model/multimodal_encoder/clip_encoder.py` 和 `llava/model/llava_arch.py`，这两个文件和原版 LLaVA **完全一致**——VisiPruner 自己的剪枝方法并没有改这两处，而是通过给 `model.generate(..., pruning_config=...)` 传参、在自定义的 LLaMA forward 里生效的（`--pruning-config` CLI 参数）。所以：

- 只要**不传 `--pruning-config`**（默认值 `None`），VisiPruner 的剪枝逻辑不会被触发；
- VisionZip 要 patch 的两个类和方法签名与原版 LLaVA 一致，可以直接 patch 成功；
- `transformers==4.37.2` 正是 VisionZip 官方要求的"装 LLaVA 环境"版本，无需另建环境。

结论：**不需要新建 conda 环境，也不需要重新下载/转换任何模型和数据**，直接在 `llava_visiPruner` 环境里装 `visionzip` 包即可。

## 3. 安装 VisionZip

```bash
conda activate llava_visiPruner
cd /home/jk/work/paper/VisionZip
pip install -e .

# 验证两个包都能正常导入，且 llava 指向 VisiPruner 的可编辑安装
python -c "import visionzip, llava; print(visionzip.__file__); print(llava.__file__)"
```

## 4. 接入已就绪的评测数据

用软链接把 VisiPruner 已经准备好的数据接进本仓库，避免重复下载/占用磁盘：

```bash
mkdir -p /home/jk/work/paper/VisionZip/playground
ln -s /home/jk/work/paper/VisiPruner/playground/data /home/jk/work/paper/VisionZip/playground/data
```

同时软链接几个评测会用到、但是放在 VisiPruner `scripts/` 目录下的转换脚本（不是 `llava` 包的一部分）：

```bash
mkdir -p /home/jk/work/paper/VisionZip/scripts
for f in convert_gqa_for_eval.py convert_mmbench_for_submission.py \
         convert_vqav2_for_submission.py convert_seed_for_submission.py \
         convert_vizwiz_for_submission.py; do
    ln -s /home/jk/work/paper/VisiPruner/scripts/$f /home/jk/work/paper/VisionZip/scripts/$f
done
```

（GQA 的 `eval/eval.py`、MME 的 `convert_answer_to_mme.py` + `eval_tool`、MM-Vet 的 `convert_answers.py` 都随数据集目录一起放在 `playground/data/eval/<bench>/` 下，跟着软链接一起带过来了，不用单独处理。）

## 5. 给 LLaVA 评测入口脚本打 VisionZip 补丁

VisionZip 官方 README 给出的接入方式就是在 `load_pretrained_model` 之后加两行：

```python
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

from visionzip import visionzip
model = visionzip(model, dominant=54, contextual=10)
```

LLaVA 官方评测总共用到 4 个入口脚本（`llava/eval/model_vqa_loader.py`、`model_vqa.py`、`model_vqa_science.py`、`model_vqa_mmbench.py`），插入点都是同一行：

```python
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
```

做法：把这 4 个文件从 VisiPruner 拷贝到本仓库 `eval/` 目录（不改动 VisiPruner 源文件），再统一插入一个共享的补丁模块，避免 4 处重复代码。

### 5.1 共享补丁模块 `eval/_visionzip_patch.py`

```python
def add_visionzip_args(parser):
    parser.add_argument("--vz-dominant", type=int, default=None,
                         help="VisionZip dominant token 数（含 CLS token）。不传则不启用 VisionZip，跑 vanilla 基线。")
    parser.add_argument("--vz-contextual", type=int, default=None,
                         help="VisionZip contextual token 数。")

def maybe_apply_visionzip(model, args):
    if args.vz_dominant is None:
        return model
    from visionzip import visionzip
    return visionzip(model, dominant=args.vz_dominant, contextual=args.vz_contextual)
```

### 5.2 拷贝 + 打补丁

```bash
cd /home/jk/work/paper/VisionZip
mkdir -p eval
cp /home/jk/work/paper/VisiPruner/llava/eval/model_vqa_loader.py eval/
cp /home/jk/work/paper/VisiPruner/llava/eval/model_vqa.py eval/
cp /home/jk/work/paper/VisiPruner/llava/eval/model_vqa_science.py eval/
cp /home/jk/work/paper/VisiPruner/llava/eval/model_vqa_mmbench.py eval/
```

对拷贝出的 4 个文件各做 2 处修改（手工编辑即可，改动很小）：

1. 顶部加一行 `from _visionzip_patch import add_visionzip_args, maybe_apply_visionzip`
2. `argparse` 构造 parser 的地方加一行 `add_visionzip_args(parser)`
3. 紧跟 `load_pretrained_model(...)` 那一行之后加一行：

   ```python
   model = maybe_apply_visionzip(model, args)
   ```

四个脚本分别对应哪些 benchmark：

| 脚本 | 对应 benchmark |
| --- | --- |
| `model_vqa_loader.py` | GQA、MME、TextVQA、POPE、VQAv2、SEED-Bench、VizWiz、Q-Bench |
| `model_vqa.py` | MM-Vet、LLaVA-Bench-in-the-Wild |
| `model_vqa_science.py` | ScienceQA |
| `model_vqa_mmbench.py` | MMBench、MMBench-CN |

`model_vqa_mmbench.py` 里没有 `pruning_config` 相关代码（VisiPruner 对它没做改动），其余三个脚本里保留原有 `--pruning-config` 参数即可（不传即为 `None`，两套补丁互不冲突）。

## 6. Token 保留数配置

VisionZip 用 `dominant + contextual` 两个参数控制保留的视觉 token 总数（`dominant` 已包含 1 个 CLS token）。README Quick Start 给出的唯一确切数值是 64-token 档位：

| 保留 token 数 | dominant | contextual | 来源 |
| --- | --- | --- | --- |
| 64 | 54 | 10 | README 官方示例，可直接使用 |
| 128 | 108 | 20 | 按 64-token 档位 dominant:contextual ≈ 5.4:1 的比例换算，**未经论文附录核实** |
| 192 | 162 | 30 | 同上，**未经论文附录核实** |

> ⚠️ 本机没有 VisionZip 论文 PDF，128 / 192 两档的确切 `dominant`/`contextual` 拆分是论文附录 B 里调过的数值，本文档给出的只是按比例换算的近似值。如果需要精确复现 Table 1 的数字，请对照 arXiv:2412.04467 附录 B 核实这两个数字，再更新上表。

## 7. 编写评测运行脚本

在 `scripts/v1_5/eval/` 下新建 shell 脚本，直接参考 VisiPruner 对应脚本（`/home/jk/work/paper/VisiPruner/scripts/v1_5/visiPruner_eval/*.sh`）去掉 `--pruning-config`、把 `--model-path` 换成本地路径、加上 `--vz-dominant/--vz-contextual`。例如 GQA：

```bash
#!/bin/bash
# scripts/v1_5/eval/gqa.sh
CKPT="llava-v1.5-7b-visionzip64"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"

python -m eval.model_vqa_loader \
    --model-path /home/jk/models/llava-v1.5-7b \
    --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
    --image-folder ./playground/data/eval/gqa/data/images \
    --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl \
    --temperature 0 \
    --vz-dominant 54 --vz-contextual 10 \
    --conv-mode vicuna_v1

python scripts/convert_gqa_for_eval.py \
    --src ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl \
    --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR && python eval/eval.py --tier testdev_balanced
```

其余 benchmark 依葫芦画瓢，唯一区别是使用的入口脚本（见第 5.2 节表格）和各自的转换/打分命令，直接照抄 VisiPruner `scripts/v1_5/visiPruner_eval/*.sh` 里对应的转换/打分部分即可（那部分和 VisionZip 无关，不需要改）。跑 vanilla 基线对比时，去掉 `--vz-dominant --vz-contextual` 两个参数即可（`answers-file` 换成不同的 `$CKPT` 名字，避免覆盖）。

各 benchmark 对应的打分方式一览：

| Benchmark | 打分脚本 |
| --- | --- |
| GQA | `playground/data/eval/gqa/data/eval/eval.py`（跟随数据打包） |
| MME | `playground/data/eval/MME/convert_answer_to_mme.py` + `eval_tool/calculation.py` |
| TextVQA | `python -m eval_textvqa`（复用 `llava.eval.eval_textvqa`，可 `python -m llava.eval.eval_textvqa`） |
| POPE | `llava.eval.eval_pope`（`python -m llava.eval.eval_pope`） |
| MMBench / MMBench-CN | `scripts/convert_mmbench_for_submission.py`，结果需上传官方 server 评测 |
| ScienceQA | `llava.eval.eval_science_qa` |
| VQAv2 | `scripts/convert_vqav2_for_submission.py`，结果需上传官方 eval server |
| VizWiz | `scripts/convert_vizwiz_for_submission.py`，结果需上传官方 eval server |
| SEED-Bench | `scripts/convert_seed_for_submission.py` |
| MM-Vet | `playground/data/eval/mm-vet/convert_answers.py`，之后用 GPT-4 打分（需要 OpenAI API key，不在本机离线范围内） |

## 8. 运行评测

```bash
conda activate llava_visiPruner
cd /home/jk/work/paper/VisionZip
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/gqa.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/textvqa.sh
# ... 其余 benchmark 同理
```

建议对每个 benchmark 都跑一次 vanilla（不加 `--vz-dominant/--vz-contextual`）和 VisionZip 64/128/192 三档，方便和论文 Table 1 的百分比对照。

## 9. VisionZip‡（高效微调版）复现要点（可选，进阶）

VisionZip‡ 只微调 cross-modality projector（`mm_projector`），论文里 8×A800 约 30 分钟即可完成。复现思路：

1. 复用 VisiPruner `scripts/v1_5/finetune.sh` / `pretrain.sh` 的 deepspeed 训练流程作为模板；
2. 冻结 LLM 和 vision tower，只放开 `mm_projector` 的梯度；
3. 在训练脚本加载完模型后同样调用 `visionzip(model, dominant=..., contextual=...)`，让 projector 直接在减少 token 后的分布上训练；
4. 训练数据用 LLaVA 的 558K 图文对（pretrain 阶段数据）。

本机目前没有确认是否已下载 LLaVA 预训练/微调用的 558K/665K 数据集，如需复现 VisionZip‡ 需要先确认这部分数据是否存在，本文档暂不展开。

## 10. 实测结果（LLaVA-1.5-7B）

以下是在本机实际跑出来的结果（`vanilla` = 不启用 VisionZip 的基线，576 个视觉 token）：

| Benchmark | vanilla | vz64 (dominant54/contextual10) | vz128 (108/20) | vz192 (162/30) |
| --- | --- | --- | --- | --- |
| GQA (accuracy) | 69.60% | 65.20% (93.7%) | 66.40% (95.4%) | 67.20% (96.6%) |
| TextVQA (accuracy) | 58.18% | 55.52% (95.4%) | 56.85% (97.7%) | 57.27% (98.4%) |
| POPE (3 类平均 accuracy) | 86.42% | 80.39% (93.0%) | 84.27% (97.5%) | 85.78% (99.3%) |
| ScienceQA-IMG (accuracy) | 70.12% | 69.91% (99.7%) | 69.82% (99.6%) | 69.89% (99.7%) |
| MME (Perception+Cognition) | 1863.71 | 1710.78 (91.8%) | 1762.98 (94.6%) | 1774.96 (95.3%) |

括号内是相对 vanilla 的保留百分比，趋势与论文一致：token 越少掉点越多，但 64 token 时仍能保留 ~92-96% 的性能。MMBench / MMBench-CN 的推理已跑完（4 个配置的答案文件都在 `playground/data/eval/mmbench{,_cn}/answers/`），但**打分需要提交官方 server**，本机没有本地 ground truth，无法直接算出 accuracy。

## 11. 复现过程中发现的环境/数据问题

1. **MMBench 转换脚本报 `ModuleNotFoundError: No module named 'openpyxl'`**：`scripts/convert_mmbench_for_submission.py` 用 `pandas.to_excel(..., engine='openpyxl')` 生成上传文件，`llava_visiPruner` 环境里没装这个包。修复：`pip install openpyxl`（纯 Python 包，不影响其他任何东西）。

2. **MME 打分脚本报 `AssertionError: .../landmark/questions_answers_YN`**：VisiPruner 共享的 `MME_Benchmark_release_version` 里 `artwork/celebrity/landmark/posters/scene` 这 5 个"特殊"类别（官方格式需要 `images/` + `questions_answers_YN/` 两个子目录）只有 `images` 这个指向类别目录自身的坏 symlink，`questions_answers_YN/` 整个不存在，导致 `MME/convert_answer_to_mme.py` 里的 `get_gt()` 断言失败。这是共享数据本身缺失，不是 VisionZip 补丁引入的问题（其余 9 个"扁平"类别的图文都在,不受影响)。

   解决办法：MME 官方 `eval_tool/Your_Results/*.txt` 其实自带了全部 14 个类别的 `图片名\t问题\t标准答案`（这是 LaVIN 那套 eval_tool 自带的参考模板,本来就覆盖所有类别),不依赖 `MME_Benchmark_release_version` 目录结构。因此写了 [eval/convert_mme_for_eval.py](../eval/convert_mme_for_eval.py),直接从 `eval_tool/Your_Results/<category>.txt` 取标准答案,和我们生成的 `answers/<CKPT>.jsonl` 按 (类别, 图片名, 问题文本) 拼接,生成 `eval_tool/answers/<CKPT>/<category>.txt`,再喂给官方 `calculation.py`。全部 2374 道题、14 个类别都能对上,不需要改动 VisiPruner 的任何文件。

   用法：

   ```bash
   cd playground/data/eval/MME
   python /home/jk/work/paper/VisionZip/eval/convert_mme_for_eval.py --experiment llava-v1.5-7b-<config>
   cd eval_tool
   python calculation.py --results_dir answers/llava-v1.5-7b-<config>
   ```

   （`scripts/v1_5/eval/mme.sh` 目前仍调用原始 `convert_answer_to_mme.py`，如果重新跑建议把这一步换成 `eval/convert_mme_for_eval.py`。）

3. **GQA 打分明显偏高、且和论文对不上**：VisiPruner 共享的 `playground/data/eval/gqa/data/testdev_balanced_questions.json` 只有 500 道题（标准 test-dev-balanced 应有 12578 道），`eval/eval.py` 只在这 500 题上算 accuracy，导致分数偏高且不可比（vanilla 69.60% vs 论文 61.9%）。

   解决办法：本机 `/home/jk/datasets/lmms-lab___gqa`（HuggingFace 格式的官方 GQA 数据集）里的 `testdev_balanced_instructions` 有完整 12578 题的 `question/answer/isBalanced/types/semantic/groups` 字段，且题目 ID 和我们推理用的 `question_id` 完全对得上。从这份数据重建了完整 ground truth，存放在 `/home/jk/work/paper/VisionZip/gqa_gt/testdev_balanced_questions.json`（没有改动 VisiPruner 任何文件），推理结果不用重新跑，直接对这份完整 GT 重新调用 `eval/eval.py --questions .../gqa_gt/testdev_balanced_questions.json` 打分即可。修复后 vanilla GQA = 61.96%，和论文 61.9% 基本完全一致。`scripts/v1_5/eval/gqa.sh` 已经更新为使用这份完整 GT。

## 12. 常见问题

- **`load_pretrained_model` 报错找不到 vision tower**：确认 `--model-path` 用的是本地绝对路径 `/home/jk/models/llava-v1.5-7b`，而不是 VisiPruner 脚本里默认写的 `liuhaotian/llava-v1.5-7b`（那是 HF hub id，本机没联网会报错）。
- **`batch_size must be 1` 断言失败**：`model_vqa_loader.py` 的 `create_data_loader` 强制 `batch_size=1`，这是 LLaVA 官方限制，不是 VisionZip 引入的问题，不要改这个断言。
- **`--pruning-config` 和 `--vz-dominant` 能否同时使用**：技术上可以同时生效（两者作用在不同代码路径，互不冲突），但复现 VisionZip 论文结果时不需要 VisiPruner 的剪枝逻辑，不要传 `--pruning-config`。
- **重新导入 `visionzip` 补丁是否安全**：`visionzip()` 是全局 monkey patch，同一进程内多次调用是幂等的，但不同评测脚本是各自独立的 Python 进程，互不影响。
