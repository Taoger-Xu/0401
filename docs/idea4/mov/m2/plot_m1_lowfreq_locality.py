"""M1 quantitative figures + summary stats, read-only (no GPU) from the data
collected by collect_m1_lowfreq_locality.py.

Produces:
  figs/m1_residual_hist_cdf.png   -- h_i(L=16) histogram + CDF
  figs/m1_quartile_bars.png       -- 4 metrics binned by h_i quartile
  figs/m1_scatter.png             -- h_i vs local_cosine / replacement_error hexbin
  data/m1_summary.json               -- correlation table (bootstrap CI), quartile
                                         table, L-sensitivity table, margin diagnostic

Run (CPU-only, no CUDA_VISIBLE_DEVICES restriction needed):
    /home/jk/miniconda3/envs/llava_visiPruner/bin/python \
        docs/idea4/mov/m2/plot_m1_lowfreq_locality.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MOV_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in MOV_DIR.parents if (p / "visionzip").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

from visionzip.prune_ideas import _fps_voronoi_cells

DATA_DIR = MOV_DIR / "data"
FIG_DIR = MOV_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "figure.titlesize": 24,
})

GRID = 24
N_TOKENS = GRID * GRID
P_CELLS = 192
N_BOOT = 2000
RNG = np.random.default_rng(0)

# Nature Publishing Group (npg) categorical palette, fixed assignment (not cycled):
# verified colorblind-separable (CAM02-UCS proxy for OKLab dE, normal + deutan/protan/
# tritan-anomaly simulation) -- every pair >20 against an 8 floor / 15 normal-vision floor.
PALETTE = {
    "local_cosine": "#3C5488",           # npg dark blue
    "centroid_cosine": "#00A087",        # npg teal
    "medoid_cosine": "#E64B35",          # npg red -- same identity as replacement_error (medoid_cosine = 1 - replacement_error_cos)
    "replacement_error_cos": "#E64B35",
    "representative_stability": "#8491B4",  # npg grey-blue, deliberately muted (secondary/weak-effect metric)
    "residual_hist": "#4DBBD5",          # npg cyan
    "residual_cdf": "#E64B35",           # npg red
    "grid_ref": "#B09C85",               # npg tan, used for reference gridlines/annotations
}


# --------------------------------------------------------------------------
# image-block bootstrap Spearman CI (fast: precompute per-image sufficient
# statistics of the *global* rank transform, then bootstrap over images by
# reweighting those statistics -- avoids re-ranking / re-concatenating a
# 1.15M-token array on every one of N_BOOT iterations).
# --------------------------------------------------------------------------
def spearman_with_ci(x, y, n_images, tokens_per_image, n_boot=N_BOOT, seed=0):
    rx = rankdata(x)
    ry = rankdata(y)
    rho_point, _ = spearmanr(x, y)

    rx2 = rx.reshape(n_images, tokens_per_image)
    ry2 = ry.reshape(n_images, tokens_per_image)
    sx = rx2.sum(1); sy = ry2.sum(1)
    sxx = (rx2 ** 2).sum(1); syy = (ry2 ** 2).sum(1)
    sxy = (rx2 * ry2).sum(1)
    n_per_img = tokens_per_image

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_images, size=n_images)
        counts = np.bincount(idx, minlength=n_images).astype(np.float64)
        N = counts.sum() * n_per_img
        Sx = (counts * sx).sum(); Sy = (counts * sy).sum()
        Sxx = (counts * sxx).sum(); Syy = (counts * syy).sum()
        Sxy = (counts * sxy).sum()
        cov = Sxy / N - (Sx / N) * (Sy / N)
        varx = Sxx / N - (Sx / N) ** 2
        vary = Syy / N - (Sy / N) ** 2
        boot[b] = cov / np.sqrt(varx * vary)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(rho_point), float(lo), float(hi)


def spearman_with_ci_cells(x, y, n_images, cells_per_image, n_boot=N_BOOT, seed=0):
    return spearman_with_ci(x, y, n_images, cells_per_image, n_boot=n_boot, seed=seed)


def bin_bootstrap_means(metrics, bin_idx, n_bins, n_images, n_tokens_per_image, n_boot=1000, seed=1):
    """metrics: {name: flat[n_images*n_tokens_per_image] array}. bin_idx: same-shape int array
    in [0, n_bins). Returns {name: {"mean":[n_bins], "lo":[n_bins], "hi":[n_bins]}} using an
    image-block bootstrap (resample whole images, not individual tokens, since tokens within an
    image are not independent)."""
    n = n_images * n_tokens_per_image
    onehot_bin = np.zeros((n_bins, n), dtype=np.float32)
    onehot_bin[bin_idx, np.arange(n)] = 1.0
    onehot_bin_img = onehot_bin.reshape(n_bins, n_images, n_tokens_per_image)
    cnt_per_bin_img = onehot_bin_img.sum(-1)  # [n_bins, n_images]

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n_images, size=(n_images, n_boot))
    C = np.zeros((n_images, n_boot), dtype=np.float32)
    for b in range(n_boot):
        C[:, b] = np.bincount(boot_idx[:, b], minlength=n_images)

    cnt_total = cnt_per_bin_img @ C  # [n_bins, n_boot]
    out = {}
    for name, arr in metrics.items():
        sum_per_bin_img = (onehot_bin_img * arr.reshape(n_images, n_tokens_per_image)).sum(-1)
        sum_total = sum_per_bin_img @ C
        boot_means = sum_total / cnt_total
        point = sum_per_bin_img.sum(1) / cnt_per_bin_img.sum(1)
        out[name] = {
            "mean": point,
            "lo": np.percentile(boot_means, 2.5, axis=1),
            "hi": np.percentile(boot_means, 97.5, axis=1),
        }
    return out


def main():
    d = np.load(DATA_DIR / "m1_tokens.npz")
    n_images = int(d["n_images"])
    h = d["h_L16"]
    local_cos = d["local_cosine"]
    centroid_cos = d["centroid_cosine"]
    rep_cos = d["replacement_error_cos"]
    rep_l2 = d["replacement_error_l2"]
    tok_stab = d["token_stability"]

    # ---------------- correlation table (main L=16) ----------------
    corr_table = {}
    for name, arr in [("local_cosine", local_cos), ("centroid_cosine", centroid_cos),
                       ("replacement_error_cos", rep_cos), ("replacement_error_l2", rep_l2),
                       ("representative_stability_token", tok_stab)]:
        rho, lo, hi = spearman_with_ci(h, arr, n_images, N_TOKENS)
        corr_table[name] = {"rho": rho, "ci95": [lo, hi]}
        print(f"h_L16 vs {name}: rho={rho:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    # winner-level representative_stability (proper granularity: one value per cell)
    winner_h = d["winner_h_L16"]
    winner_stab = d["winner_stability"]
    rho, lo, hi = spearman_with_ci_cells(winner_h, winner_stab, n_images, P_CELLS)
    corr_table["representative_stability_winner"] = {"rho": rho, "ci95": [lo, hi]}
    print(f"winner_h_L16 vs winner_stability: rho={rho:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (n_cells={n_images*P_CELLS})")

    for kind in ["jitter", "resize", "flip"]:
        s = d[f"winner_stability_{kind}"]
        rho, lo, hi = spearman_with_ci_cells(winner_h, s, n_images, P_CELLS)
        corr_table[f"representative_stability_winner_{kind}"] = {
            "rho": rho, "ci95": [lo, hi], "mean_stability": float(s.mean())}
        print(f"  [{kind}] mean_stability={s.mean():.4f}  rho={rho:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    # ---------------- L sensitivity ----------------
    l_sensitivity = {}
    for L in [8, 12, 16, 20]:
        hL = d[f"h_L{L}"]
        r1, _ = spearmanr(hL, local_cos)
        r2, _ = spearmanr(hL, rep_cos)
        r3, _ = spearmanr(hL, centroid_cos)
        l_sensitivity[L] = {"local_cosine": float(r1), "replacement_error_cos": float(r2),
                             "centroid_cosine": float(r3)}
    print("L-sensitivity:", json.dumps(l_sensitivity, indent=2))

    # ---------------- margin diagnostic (explains representative_stability) ----------------
    cells, _ = _fps_voronoi_cells(GRID, P_CELLS, torch.device("cpu"))
    cells_np = cells.numpy()
    cc2 = centroid_cos.reshape(n_images, N_TOKENS)
    h2 = h.reshape(n_images, N_TOKENS)
    margins = np.zeros((n_images, P_CELLS), dtype=np.float64)
    m_winner_h = np.zeros((n_images, P_CELLS), dtype=np.float64)
    for c in range(P_CELLS):
        members = np.where(cells_np == c)[0]
        if len(members) == 1:
            margins[:, c] = 1.0
            m_winner_h[:, c] = h2[:, members[0]]
            continue
        vals = cc2[:, members]
        order = np.argsort(-vals, axis=1)
        margins[:, c] = vals[np.arange(n_images), order[:, 0]] - vals[np.arange(n_images), order[:, 1]]
        m_winner_h[:, c] = h2[np.arange(n_images), members[order[:, 0]]]
    margins_flat = margins.reshape(-1)
    rho_margin_stab, _ = spearmanr(margins_flat, winner_stab)
    rho_h_margin, _ = spearmanr(winner_h, margins_flat)
    med = np.median(margins_flat)
    lo_mask, hi_mask = margins_flat <= med, margins_flat > med
    rho_lo, _ = spearmanr(winner_h[lo_mask], winner_stab[lo_mask])
    rho_hi, _ = spearmanr(winner_h[hi_mask], winner_stab[hi_mask])
    margin_diag = {
        "rho_margin_vs_stability": float(rho_margin_stab),
        "rho_winnerh_vs_margin": float(rho_h_margin),
        "rho_winnerh_vs_stability_low_margin_half": float(rho_lo),
        "rho_winnerh_vs_stability_high_margin_half": float(rho_hi),
        "mean_stability_low_margin_half": float(winner_stab[lo_mask].mean()),
        "mean_stability_high_margin_half": float(winner_stab[hi_mask].mean()),
    }
    print("margin diagnostic:", json.dumps(margin_diag, indent=2))

    # ================================================================
    # Fig A: residual histogram + CDF
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].hist(h, bins=60, color=PALETTE["residual_hist"])
    axes[0].set_xlabel(r"$h_i$ = $\|f_i - \mathrm{lowpass16\ recon}\|$")
    axes[0].set_ylabel("token count")
    axes[0].set_title(f"residual distribution\n(n={len(h):,} tokens, {n_images} images)")

    xs = np.sort(h)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    axes[1].plot(xs, ys, color=PALETTE["residual_cdf"], lw=3)
    for q in [0.25, 0.5, 0.75]:
        axes[1].axhline(q, color="gray", ls=":", lw=1.2)
    axes[1].set_xlabel(r"$h_i$")
    axes[1].set_ylabel("cumulative fraction of tokens")
    axes[1].set_title("cumulative distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m1_residual_hist_cdf.png", dpi=150)
    fig.savefig(FIG_DIR / "m1_residual_hist_cdf.pdf")
    plt.close(fig)

    # ================================================================
    # Fig B: quartile bars
    # ================================================================
    q_edges = np.quantile(h, [0, 0.25, 0.5, 0.75, 1.0])
    q_idx = np.digitize(h, q_edges[1:-1], right=True)  # 0..3
    labels = ["Q1 (lowest h)", "Q2", "Q3", "Q4 (highest h)"]
    quartile_table = {}
    means_local, means_centroid, means_stab, means_rep = [], [], [], []
    for qi in range(4):
        mask = q_idx == qi
        ml, mc, ms, mr = local_cos[mask].mean(), centroid_cos[mask].mean(), tok_stab[mask].mean(), rep_cos[mask].mean()
        means_local.append(ml); means_centroid.append(mc); means_stab.append(ms); means_rep.append(mr)
        quartile_table[labels[qi]] = {"local_cosine": float(ml), "centroid_cosine": float(mc),
                                        "representative_stability": float(ms), "replacement_error_cos": float(mr),
                                        "n_tokens": int(mask.sum())}

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    x = np.arange(4)
    w = 0.25
    axes[0].bar(x - w, means_local, width=w, label="local_cosine", color=PALETTE["local_cosine"])
    axes[0].bar(x, means_centroid, width=w, label="centroid_cosine", color=PALETTE["centroid_cosine"])
    axes[0].bar(x + w, means_stab, width=w, label="representative_stability", color=PALETTE["representative_stability"])
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("mean value\n(higher = more locally replaceable / stable)")
    axes[0].legend(fontsize=18)
    axes[0].set_title("similarity / stability metrics\nby residual quartile")

    axes[1].bar(x, means_rep, width=0.5, color=PALETTE["replacement_error_cos"])
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("mean replacement_error_cos\n(lower = better)")
    axes[1].set_title("neighbor-medoid replacement error\nby residual quartile")
    fig.suptitle("M1: four locality metrics grouped by $h_i$ (L=16) quartile")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG_DIR / "m1_quartile_bars.png", dpi=150)
    fig.savefig(FIG_DIR / "m1_quartile_bars.pdf")
    plt.close(fig)

    # ================================================================
    # Fig C: hexbin scatter
    # ================================================================
    x_clip = float(np.percentile(h, 99))  # top 1% (extreme text/edge outliers) clipped for legibility
    fig, axes = plt.subplots(1, 2, figsize=(22, 9.5))
    hb0 = axes[0].hexbin(h, local_cos, gridsize=60, cmap="viridis", mincnt=1, bins="log", extent=(0, x_clip, -0.2, 1.0))
    axes[0].set_xlabel(r"$h_i$"); axes[0].set_ylabel("local_cosine")
    axes[0].set_title(f"$h_i$ vs local_cosine\n(Spearman={corr_table['local_cosine']['rho']:.2f})")
    fig.colorbar(hb0, ax=axes[0], label="log10(token count)")

    hb1 = axes[1].hexbin(h, rep_cos, gridsize=60, cmap="viridis", mincnt=1, bins="log", extent=(0, x_clip, 0, 1.25))
    axes[1].set_xlabel(r"$h_i$"); axes[1].set_ylabel("replacement_error_cos")
    axes[1].set_title(f"$h_i$ vs replacement_error\n(Spearman={corr_table['replacement_error_cos']['rho']:.2f})")
    fig.colorbar(hb1, ax=axes[1], label="log10(token count)")
    fig.suptitle(f"x-axis clipped at the 99th percentile ($h_i$={x_clip:.0f}); correlations on full, unclipped data", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(FIG_DIR / "m1_scatter.png", dpi=150)
    fig.savefig(FIG_DIR / "m1_scatter.pdf")
    plt.close(fig)

    # ================================================================
    # Fig E (headline, fine-grained line version): percentile-binned redundancy
    # curve. Three independent "similarity to a substitute" measures --
    # local_cosine, centroid_cosine, and medoid_cosine (= 1 - replacement_error_cos,
    # same [0,1]-similarity semantics as the other two) -- plotted against the
    # h_i percentile rank, with image-block-bootstrap 95% CI ribbons. Binning
    # by percentile (not raw h_i) gives every bin equal token count despite
    # h_i's heavy right skew, avoiding the long-tail-squashes-the-plot issue
    # visible in the raw hexbin scatter (m1_scatter.png). Kept as a robustness
    # figure alongside the quartile bar chart below (Fig F), which is the
    # core motivation figure actually used in the writeup.
    # ================================================================
    medoid_cos = 1.0 - rep_cos  # same [0,1] "similarity to substitute" semantics as the other two
    n_bins = 40
    pct = (rankdata(h, method="average") - 0.5) / len(h) * 100.0
    bin_idx40 = np.clip((pct / (100.0 / n_bins)).astype(int), 0, n_bins - 1)
    bin_centers = (np.arange(n_bins) + 0.5) * (100.0 / n_bins)

    metrics3 = {"local_cosine": local_cos, "centroid_cosine": centroid_cos, "medoid_cosine": medoid_cos}
    curve_data = bin_bootstrap_means(metrics3, bin_idx40, n_bins, n_images, N_TOKENS, n_boot=1000, seed=1)

    fig, ax = plt.subplots(figsize=(16, 11))
    colors = {k: PALETTE[k] for k in ["local_cosine", "centroid_cosine", "medoid_cosine"]}
    labels = {"local_cosine": "local_cosine (token vs 8-neighbor mean)",
              "centroid_cosine": "centroid_cosine (token vs cell centroid)",
              "medoid_cosine": "medoid_cosine (token vs neighbor-medoid, = 1 - replacement_error)"}
    for name in ["local_cosine", "centroid_cosine", "medoid_cosine"]:
        cd = curve_data[name]
        ax.plot(bin_centers, cd["mean"], color=colors[name], lw=3.5, label=labels[name])
        ax.fill_between(bin_centers, cd["lo"], cd["hi"], color=colors[name], alpha=0.2, linewidth=0)
    for q, val in zip([25, 50, 75, 90], np.percentile(h, [25, 50, 75, 90])):
        ax.axvline(q, color="gray", ls=":", lw=1.2)
        ax.text(q, 0.02, f"h={val:.0f}", fontsize=16, rotation=90, va="bottom", ha="right", color="gray")
    ax.set_xlabel(r"percentile rank of $h_i$ (low-freq. residual); gray = Q1/Q2/Q3/P90 of $h_i$")
    ax.set_ylabel("mean similarity-to-substitute\n(shaded = 95% bootstrap CI)")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 100)
    ax.legend(loc="upper right", fontsize=17, frameon=False)
    ax.set_title("Three independent local-replaceability measures all decay monotonically\nas the low-frequency residual increases (2,000 images, 1.15M tokens)", fontsize=22)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m1_redundancy_curve.png", dpi=150)
    fig.savefig(FIG_DIR / "m1_redundancy_curve.pdf")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'm1_redundancy_curve.png'}")

    # ================================================================
    # Fig F (core motivation bar chart): same 3 unified similarity-to-substitute
    # metrics as Fig E, but binned into h_i quartiles and rendered as grouped
    # bars with 95% bootstrap CI error bars, plus the Q1->Q4 relative drop
    # annotated above each metric's bars. representative_stability is
    # deliberately NOT in this figure -- its effect size is an order of
    # magnitude smaller (see margin diagnostic, sec 5) and would dilute the
    # headline message; it is reported separately.
    # ================================================================
    q_edges2 = np.quantile(h, [0, 0.25, 0.5, 0.75, 1.0])
    bin_idx4 = np.clip(np.digitize(h, q_edges2[1:-1], right=True), 0, 3)
    bars_data = bin_bootstrap_means(metrics3, bin_idx4, 4, n_images, N_TOKENS, n_boot=2000, seed=2)

    fig, ax = plt.subplots(figsize=(16, 11))
    q_labels = ["Q1\n(lowest residual)", "Q2", "Q3", "Q4\n(highest residual)"]
    x = np.arange(4)
    w = 0.26
    offsets = {"local_cosine": -w, "centroid_cosine": 0, "medoid_cosine": w}
    for name in ["local_cosine", "centroid_cosine", "medoid_cosine"]:
        bd = bars_data[name]
        err = np.stack([bd["mean"] - bd["lo"], bd["hi"] - bd["mean"]])
        ax.bar(x + offsets[name], bd["mean"], width=w, label=labels[name].split(" (")[0],
               color=colors[name], yerr=err, capsize=5, error_kw=dict(lw=1.5))
        drop = (bd["mean"][-1] - bd["mean"][0]) / bd["mean"][0] * 100
        ax.annotate(f"{drop:+.0f}%", xy=(x[-1] + offsets[name], bd["mean"][-1] + err[1, -1] + 0.03),
                    ha="center", fontsize=18, color=colors[name], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(q_labels)
    ax.set_ylabel("mean similarity-to-substitute\n(higher = more locally replaceable)")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", fontsize=18, frameon=False)
    ax.set_title("Local replaceability drops sharply from low- to high-residual tokens\n"
                 "(3 independent measures, quartiles of $h_i$, error bars = 95% bootstrap CI)", fontsize=22)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m1_motivation_bars.png", dpi=150)
    fig.savefig(FIG_DIR / "m1_motivation_bars.pdf")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'm1_motivation_bars.png'}")

    # ---------------- save summary json ----------------
    summary = {
        "n_images": n_images, "n_tokens": int(len(h)), "p_cells": P_CELLS,
        "correlations_L16": corr_table,
        "l_sensitivity": l_sensitivity,
        "quartile_table": quartile_table,
        "margin_diagnostic": margin_diag,
        "redundancy_curve": {
            "n_bins": n_bins,
            "bin_centers_percentile": bin_centers.tolist(),
            **{name: {k: v.tolist() for k, v in cd.items()} for name, cd in curve_data.items()},
        },
        "motivation_bars": {
            "labels": ["Q1", "Q2", "Q3", "Q4"],
            **{name: {k: v.tolist() for k, v in bd.items()} for name, bd in bars_data.items()},
        },
        "gqa_source_note": "lmms-lab/GQA has no `val` config; used test_balanced_images (2987 unique) instead",
    }
    with (DATA_DIR / "m1_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {DATA_DIR / 'm1_summary.json'}")
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
