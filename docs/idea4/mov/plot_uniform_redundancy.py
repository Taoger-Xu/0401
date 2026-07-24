"""Render the uniform (spatial-position) token-drop performance-retention figure.

Evidence for spatial redundancy in vision-encoder output tokens that does not
rely on any importance criterion: drop tokens at uniformly spaced spatial
positions only (``stage0_prune.py::uniform_compress``, ``STAGE0_METHOD=uniform``
in the companion LLaVA-1.5 checkout) and see how much accuracy survives on
GQA / MMBench / TextVQA.

Consumes ``mov/data/results_uniform_redundancy.csv`` -- real (non-placeholder)
scores for 10 uniform keep-ratios (10%..100% of the 576 tokens) collected via
``mov/run_uniform_sweep.sh`` (kept for provenance; that script only runs
inside the CLSE/LLaVA1.5 checkout, see its header note).

X axis: Top K% Tokens (uniformly kept token percentage).
Y axis: Performance Ratio (%) -- score relative to the 100%-token baseline.

Run: python mov/plot_uniform_redundancy.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_DIR = Path(REPO_ROOT) / "mov" / "data"
OUT_DIR = Path(REPO_ROOT) / "mov" / "figs"
CSV_PATH = DATA_DIR / "results_uniform_redundancy.csv"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
})

TASK_STYLE = {
    "gqa":     dict(label="GQA",     color="#2A77C7", marker="o", ms=7),
    "mmbench": dict(label="MMBench", color="#E37B30", marker="^", ms=7),
    "textvqa": dict(label="TextVQA", color="#3FA34D", marker="s", ms=6.5),
}
TASK_ORDER = ["gqa", "mmbench", "textvqa"]

X_TICKS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
Y_TICKS = [80, 85, 90, 95, 100]


def load(path: Path):
    # data[task] = list of (topk_pct, perf_ratio)
    data = defaultdict(list)
    has_placeholder = False
    with open(path) as f:
        for row in csv.DictReader(r for r in f if not r.lstrip().startswith("#")):
            t = row["task"].strip()
            data[t].append((float(row["topk_pct"]), float(row["perf_ratio"])))
            if int(row.get("placeholder", 0)):
                has_placeholder = True
    for t in data:
        data[t].sort()
    return data, has_placeholder


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, has_placeholder = load(CSV_PATH)

    fig, ax = plt.subplots(figsize=(6.0, 4.6))

    for t in TASK_ORDER:
        if t not in data:
            continue
        st = TASK_STYLE[t]
        xs = [x for x, _ in data[t]]
        ys = [y for _, y in data[t]]
        ax.plot(xs, ys, color=st["color"], lw=2.2, ls="-",
                 marker=st["marker"], ms=st["ms"],
                 mfc=st["color"], mec=st["color"], label=st["label"], zorder=5)

    ax.axhline(100, color="#888", ls=":", lw=1.0, zorder=1)

    ax.set_xlabel("Top K% Tokens", fontsize=11)
    ax.set_ylabel("Performance Ratio (%)", fontsize=11)
    ax.set_xlim(5, 105)
    ax.set_ylim(80, 103)
    ax.set_xticks(X_TICKS)
    ax.set_yticks(Y_TICKS)
    ax.grid(True, ls=":", lw=0.5, color="#bbb", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#444")

    leg = ax.legend(loc="lower right", fontsize=9.5, frameon=True, edgecolor="#888")
    leg.get_frame().set_linewidth(0.6)

    if has_placeholder:
        fig.text(0.5, 0.5, "DRAFT · placeholder numbers",
                  fontsize=30, color="#d33", alpha=0.16,
                  ha="center", va="center", rotation=18, zorder=100)

    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"exp_m3_uniform_redundancy.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
