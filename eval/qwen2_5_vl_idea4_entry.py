"""lmms-eval entry point for the local Qwen2.5-VL Anchor-Cover model."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# lmms-eval imports this symbol from the transformers package.  Replace it
# before its lazy model registry is loaded, leaving lmms-eval itself untouched.
import transformers
from Qwen2_5_VL.qwen2_5vl_visionzip import Qwen2_5_VLForConditionalGeneration

transformers.Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration

from lmms_eval.__main__ import cli_evaluate

# Importing cli_evaluate eagerly imports lmms-eval's qwen2_5_vl wrappers, whose
# `from transformers import Qwen2_5_VLForConditionalGeneration` already captured
# the *stock* class before (and independently of) the attribute patch above.
# Rebind the name on the wrapper modules themselves so `from_pretrained` builds
# the Anchor-Cover model.  Without this the harness silently runs vanilla.
for _mod in ("lmms_eval.models.simple.qwen2_5_vl",
             "lmms_eval.models.chat.qwen2_5_vl"):
    try:
        _m = __import__(_mod, fromlist=["Qwen2_5_VLForConditionalGeneration"])
        _m.Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration
    except Exception:
        pass

if __name__ == "__main__":
    cli_evaluate()
