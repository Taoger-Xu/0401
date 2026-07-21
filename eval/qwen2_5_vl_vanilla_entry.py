"""lmms-eval entry point for STOCK Qwen2.5-VL (no idea4 / no pruning).

This is the vanilla baseline: it deliberately does NOT replace
transformers.Qwen2_5_VLForConditionalGeneration, so lmms-eval loads the
unmodified model and every visual token is kept. Use it to produce the
reference numbers that idea4 (scripts/qwen2_5_vl_idea4_eval.sh) is compared
against under an identical protocol.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lmms_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
