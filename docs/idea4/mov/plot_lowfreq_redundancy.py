"""Render the vision-token low-frequency-redundancy figure: one line chart.

Consumes the two 10,000-image collections written by
``collect_lowfreq_redundancy.py`` (LLaVA-1.5 / CLIP-L vision tower) and
``collect_lowfreq_redundancy_qwen.py`` (Qwen2.5-VL vision tower) --
``mov/data/summary*.json`` -- and draws a single chart:
mean fidelity-vs-compression-ratio curve per model, +/- 1 std band across
the 10k images, plus a reference line at the "unrelated vectors" baseline.

Both curves share an x-axis of "% of low-frequency 2D-DCT coefficients
retained" even though the two encoders use different, differently-shaped
token grids (LLaVA/CLIP: fixed 24x24; Qwen2.5-VL: dynamic per-image
resolution) -- see the collection scripts' docstrings for how the relative
low-pass ratio ``k/24`` (LLaVA) and ``r`` (Qwen) are made comparable.

Style follows a reference "SDPO paper" line-chart aesthetic (serif font,
boxed spines with inward ticks, no grid, shaded +/-1 std band, sparse-dot
reference line). LaTeX (``text.usetex``) is not available on this machine,
so we approximate the look with matplotlib's built-in "cm" mathtext font
instead of real Computer Modern via usetex.

Run: python mov/plot_lowfreq_redundancy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_DIR = Path(REPO_ROOT) / "mov" / "data"
OUT_DIR = Path(REPO_ROOT) / "mov" / "figs"

# text.usetex is left off: no LaTeX toolchain (latex/dvipng) is installed on
# this machine. mathtext with fontset="cm" gives a close approximation for
# the math-like axis labels without needing a LaTeX binary.
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }
)

C_LLAVA = "#2A6DB5"  # CLIP-L / LLaVA-1.5 vision tower (blue)
C_QWEN = "#7B52AB"  # Qwen2.5-VL vision tower (purple)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    llava = json.loads((DATA_DIR / "summary.json").read_text())
    qwen = json.loads((DATA_DIR / "summary_qwen.json").read_text())

    llava_k = np.array(llava["k_values"])
    llava_x = (llava_k / llava["_meta"]["grid_h"]) ** 2 * 100
    llava_mean = np.array(llava["mean_cos_sim"])
    llava_std = np.array(llava["std_cos_sim"])

    qwen_steps = np.arange(1, qwen["ratio_steps"] + 1)
    qwen_x = (qwen_steps / qwen["ratio_steps"]) ** 2 * 100
    qwen_mean = np.array(qwen["mean_cos_sim"])
    qwen_std = np.array(qwen["std_cos_sim"])

    n_llava = llava["_meta"]["num_images"]
    n_qwen = qwen["_meta"]["num_images"]

    fig, ax = plt.subplots(figsize=(6.5, 4.6))

    ax.fill_between(qwen_x, qwen_mean - qwen_std, qwen_mean + qwen_std, color=C_QWEN, alpha=0.18)
    ax.fill_between(llava_x, llava_mean - llava_std, llava_mean + llava_std, color=C_LLAVA, alpha=0.18)

    ax.plot(qwen_x, qwen_mean, color=C_QWEN, lw=2.5, label=f"Qwen2.5-VL ViT (n={n_qwen:,})")
    ax.plot(llava_x, llava_mean, color=C_LLAVA, lw=2.5, label=f"CLIP-L / LLaVA-1.5 (n={n_llava:,})")

    ax.axhline(
        0.0,
        color="#AAAAAA",
        lw=1.8,
        linestyle=(0, (1, 2)),
        label="unrelated-vector baseline",
    )

    stage_a_index = llava["k_values"].index(llava["_meta"]["stage_a_k"])
    stage_a_x = llava_x[stage_a_index]
    stage_a_y = llava_mean[stage_a_index]
    ax.axvline(stage_a_x, color="black", lw=1.0, linestyle=":", alpha=0.6)
    ax.annotate(
        "25% of coefficients\n(this project's Stage A ratio)",
        xy=(stage_a_x, stage_a_y),
        xytext=(stage_a_x + 10, 0.18),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", alpha=0.7),
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlabel("% of low-frequency DCT coefficients retained", fontsize=13)
    ax.set_ylabel("Mean per-token cosine fidelity", fontsize=13)
    ax.set_title("Vision tokens are low-frequency redundant\nin two unrelated encoders", fontsize=15, pad=7)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    leg = ax.legend(
        fontsize=10,
        loc="lower right",
        framealpha=0,
        edgecolor="none",
        handlelength=2.2,
        borderaxespad=0.5,
        labelspacing=0.4,
    )

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.5)
    ax.tick_params(direction="in", length=5, width=1.2, labelsize=11)
    ax.grid(False)

    fig.tight_layout(pad=0.9)
    out_path = OUT_DIR / "lowfreq_token_redundancy.png"
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
