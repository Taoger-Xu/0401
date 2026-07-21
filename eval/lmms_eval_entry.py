"""
Entry point that lets lmms-eval's built-in `--model llava` wrapper run with
VisionZip enabled, without modifying lmms-eval or the llava package.

lmms_eval.models.simple.llava does `from llava.model.builder import
load_pretrained_model` at import time and calls it directly. We monkeypatch
that function *before* lmms-eval's model registry imports it, so every call
returns a VisionZip-patched model when VZ_DOMINANT is set (mirrors CLSE's
PRUNE/RETAIN_TOKEN env-var pattern).

Usage: same argv as `python -m lmms_eval ...`, e.g.
    accelerate launch eval/lmms_eval_entry.py --model llava \
        --model_args pretrained=/home/jk/models/llava-v1.5-7b \
        --tasks gqa,mme,pope --batch_size 1 --output_path logs/ --log_samples
"""
import os
import sys

# accelerate launch spawns subprocesses that don't always pick up pip's
# editable-install import hooks for this repo; add it to sys.path explicitly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import llava.model.builder as builder

_orig_load_pretrained_model = builder.load_pretrained_model


def _patched_load_pretrained_model(*args, **kwargs):
    tokenizer, model, image_processor, context_len = _orig_load_pretrained_model(*args, **kwargs)
    idea_method = os.environ.get("IDEA_METHOD")
    dominant = os.environ.get("VZ_DOMINANT")
    if idea_method is not None:
        # idea1/idea2/idea3/idea4 spatial-redundancy pruning (docs/idea*.md)
        from visionzip import visionzip_idea
        k = int(os.environ["IDEA_K"])
        lam = float(os.environ.get("IDEA_LAMBDA", "0.5"))
        rho = float(os.environ.get("IDEA_RHO", "0.5"))
        sigma = float(os.environ.get("IDEA_SIGMA", "2.0"))
        cover_factor = float(os.environ.get("IDEA_COVER_FACTOR", "3.0"))
        lowpass = int(os.environ.get("IDEA_LOWPASS", "16"))
        out_grid = int(os.environ.get("IDEA_OUTGRID", "14"))
        w_var = float(os.environ.get("IDEA_WVAR", "0.3"))
        gamma = float(os.environ.get("IDEA_GAMMA", "0.5"))
        detail_p = float(os.environ.get("IDEA_DETAIL_P", "2.0"))
        detail_src = os.environ.get("IDEA_DETAIL_SRC", "local_var")
        merge = os.environ.get("IDEA_MERGE", "0") == "1"
        idea_contextual = int(os.environ.get("IDEA_CONTEXTUAL", "0"))
        model = visionzip_idea(model, method=idea_method, k=k, lam=lam, rho=rho,
                               sigma=sigma, cover_factor=cover_factor,
                               lowpass=lowpass, out_grid=out_grid,
                               w_var=w_var, gamma=gamma,
                               detail_p=detail_p, detail_src=detail_src,
                               merge=merge, contextual=idea_contextual)
    elif dominant is not None:
        from visionzip import visionzip
        contextual = int(os.environ.get("VZ_CONTEXTUAL", "0"))
        model = visionzip(model, dominant=int(dominant), contextual=contextual)
    return tokenizer, model, image_processor, context_len


builder.load_pretrained_model = _patched_load_pretrained_model

from lmms_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
