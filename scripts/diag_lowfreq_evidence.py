#!/usr/bin/env python
"""idea4 motivation figure: visual evidence for "same-cell low-frequency
redundancy" (docs/idea_summary.md §idea4 one-liner).

Claim being illustrated: within a local spatial neighbourhood ("cell"),
most CLIP penultimate-layer patch tokens are well approximated by the
image-wide 16x16 DCT low-pass reconstruction; only tokens that straddle
edges/text/object boundaries have large reconstruction residual. That
residual (negated) is exactly idea4's low-frequency-redundancy score
l_i = Norm_C(i)(-||H_i - IDCT(Pi_16(DCT(H)))_i||).

Two pieces of evidence, both built from real CLIP features (no synthetic
data), reusing the exact functions idea1/idea4 use at inference time:

  1. Per-image qualitative panel (n_images, default 6):
       [a] original image + FPS-Voronoi cell boundaries (the same cell
           partition select_anchor_cover uses for C(i))
       [b] PCA-RGB of the *real* patch features
       [c] PCA-RGB of the *lowpass=16* IDCT(Pi_16(DCT(H))) reconstruction,
           projected through the SAME PCA basis as [b]
       [c] and [b] look near-identical almost everywhere except at edges/
           text -> that's the "low-frequency redundancy" claim, directly
           visible.
       [d] residual heatmap ||H_i - rec_i|| overlaid on the image -- bright
           = non-redundant (kept), dark = redundant (safe to prune from
           this cell's perspective)

  2. Pooled quantitative evidence (n_stats images, default 200, vision-tower
     forward only so this is cheap):
       [a] histogram of per-image-normalised residual across all tokens
           -> should be heavily left-skewed (majority low = redundant)
       [b] DCT energy-compaction curve: mean fraction of total spectral
           energy captured by an LxL low-frequency block, L=1..grid,
           with a marker at L=16 -> justifies the lowpass=16 cutoff used
           throughout idea1/idea4.

Usage:
    python scripts/diag_lowfreq_evidence.py --n_images 6 --n_stats 200
Output:
    docs/idea4/figs/lowfreq_evidence/qualitative.png
    docs/idea4/figs/lowfreq_evidence/quantitative.png
"""
import os
import sys
import argparse

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/jk/datasets")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from visionzip.prune_ideas import lowfreq_reconstruct, _fps_voronoi_cells, _dct_matrix

GRID = 24               # CLIP-L/14 @336 -> 24x24 patch grid
CELL_BUDGET = 48       # illustrative cell count for the qualitative panel
LOWPASS = 16
CLIP_PATH = "/home/jk/models/clip-vit-large-patch14-336"


def load_clip(device):
    from transformers import CLIPVisionModel, CLIPImageProcessor
    proc = CLIPImageProcessor.from_pretrained(CLIP_PATH)
    model = CLIPVisionModel.from_pretrained(CLIP_PATH).to(device).eval()
    return proc, model


@torch.no_grad()
def patch_feats_for(img, proc, model, device):
    px = proc(img, return_tensors="pt")["pixel_values"].to(device)
    out = model(px, output_hidden_states=True)
    hs = out.hidden_states[-2]                 # [1, 577, 1024], penultimate layer
    return hs[:, 1:, :]                        # [1, 576, 1024]


def upsample_nearest(arr2d, size):
    """arr2d: [grid, grid] (or [grid,grid,3]) -> PIL-resized to (size,size)."""
    a = np.asarray(arr2d)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    mode = "RGB" if a.ndim == 3 else "L"
    im = Image.fromarray(a, mode=mode)
    return np.asarray(im.resize((size, size), Image.NEAREST))


def cell_boundary_mask(cells_2d, size):
    """cells_2d: [grid,grid] int cell ids -> thin pixel-space boundary mask.

    Draws the seam between adjacent tokens whose cell id differs, at true
    image resolution (not a NEAREST blow-up of the coarse grid, which paints
    whole 14x14 token blocks instead of a thin line)."""
    g = cells_2d.shape[0]
    stride = size // g
    thickness = max(1, stride // 7)
    mask = np.zeros((size, size), dtype=bool)
    for r in range(g):
        y0, y1 = r * stride, (r + 1) * stride
        for c in range(g - 1):
            if cells_2d[r, c] != cells_2d[r, c + 1]:
                x = (c + 1) * stride
                mask[y0:y1, max(0, x - thickness):x + thickness] = True
    for c in range(g):
        x0, x1 = c * stride, (c + 1) * stride
        for r in range(g - 1):
            if cells_2d[r, c] != cells_2d[r + 1, c]:
                y = (r + 1) * stride
                mask[max(0, y - thickness):y + thickness, x0:x1] = True
    return mask


def pca_rgb(feats_np, ref_pca=None):
    from sklearn.decomposition import PCA
    if ref_pca is None:
        ref_pca = PCA(n_components=3, random_state=0).fit(feats_np)
    proj = ref_pca.transform(feats_np)
    return proj, ref_pca


def normalize_channels(*arrs):
    """Shared min-max across all given [N,3] arrays, per channel."""
    stacked = np.concatenate(arrs, axis=0)
    lo, hi = stacked.min(0, keepdims=True), stacked.max(0, keepdims=True)
    return [ (a - lo) / (hi - lo + 1e-6) for a in arrs ]


def qualitative(images, proc, model, device, out_path, size=336):
    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 3.1, n * 3.1))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["image + cells C(i)", "real features (PCA-RGB)",
                  "16x16 DCT low-pass recon (PCA-RGB)", "residual  ||H_i - rec_i||"]

    cells, _ = _fps_voronoi_cells(GRID, CELL_BUDGET, device)
    cells_2d = cells.view(GRID, GRID).cpu().numpy()

    for row, img in enumerate(images):
        img_rs = img.resize((size, size), Image.BILINEAR)
        img_np = np.asarray(img_rs).astype(np.float32) / 255.0

        feats = patch_feats_for(img, proc, model, device)          # [1,576,1024]
        rec = lowfreq_reconstruct(feats, grid=GRID, lowpass=LOWPASS)  # [1,576,1024]
        residual = (feats.float() - rec).norm(dim=-1)[0].cpu().numpy()  # [576]

        feats_np = feats[0].float().cpu().numpy()
        rec_np = rec[0].float().cpu().numpy()
        proj_real, pca = pca_rgb(feats_np)
        proj_rec, _ = pca_rgb(rec_np, ref_pca=pca)
        proj_real_n, proj_rec_n = normalize_channels(proj_real, proj_rec)

        real_rgb = upsample_nearest(proj_real_n.reshape(GRID, GRID, 3), size)
        rec_rgb = upsample_nearest(proj_rec_n.reshape(GRID, GRID, 3), size)

        res_2d = residual.reshape(GRID, GRID)
        lo, hi = np.percentile(res_2d, 2), np.percentile(res_2d, 98)
        res_n = np.clip((res_2d - lo) / (hi - lo + 1e-6), 0, 1)
        res_up = upsample_nearest(res_n, size).astype(np.float32) / 255.0
        heat = plt.get_cmap("inferno")(res_up)[..., :3]
        overlay = 0.35 * img_np + 0.65 * heat

        # panel a: image + cell boundaries
        edge = cell_boundary_mask(cells_2d, size)
        cell_vis = img_np.copy()
        cell_vis[edge] = [1.0, 1.0, 1.0]

        for c, arr in zip(range(4), [cell_vis, real_rgb, rec_rgb, overlay]):
            ax = axes[row, c]
            ax.imshow(arr)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[c], fontsize=10)

    fig.suptitle("idea4 motivation: same-cell low-frequency redundancy "
                 f"(real CLIP-L/14@336 penultimate features, lowpass={LOWPASS}/{GRID})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


@torch.no_grad()
def quantitative(images, proc, model, device, out_path):
    all_res = []
    energy_frac = np.zeros(GRID)
    M = _dct_matrix(GRID, device)
    n_used = 0
    for img in images:
        feats = patch_feats_for(img, proc, model, device)          # [1,576,1024]
        rec = lowfreq_reconstruct(feats, grid=GRID, lowpass=LOWPASS)
        residual = (feats.float() - rec).norm(dim=-1)[0].cpu().numpy()
        residual = residual / (residual.max() + 1e-6)
        all_res.append(residual)

        g = feats.float().view(1, GRID, GRID, -1)
        coef = torch.einsum("kh,bhwd->bkwd", M, g)
        coef = torch.einsum("lw,bkwd->bkld", M, coef)               # [1,grid,grid,D]
        e2d = coef.pow(2).sum(dim=(0, 3)).cpu().numpy()             # [grid,grid]
        total = e2d.sum()
        for L in range(1, GRID + 1):
            energy_frac[L - 1] += e2d[:L, :L].sum() / (total + 1e-9)
        n_used += 1
    energy_frac /= n_used
    all_res = np.concatenate(all_res)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(all_res, bins=40, color="#3b6fa0")
    axes[0].set_title(f"per-token residual (norm. per image, pooled over {n_used} images)")
    axes[0].set_xlabel("||H_i - lowpass16 recon||  (0=fully redundant, 1=worst in image)")
    axes[0].set_ylabel("token count")

    axes[1].plot(range(1, GRID + 1), energy_frac, marker="o", ms=3, color="#a04b3b")
    axes[1].axvline(LOWPASS, color="gray", ls="--", lw=1)
    axes[1].text(LOWPASS + 0.3, 0.15, f"lowpass={LOWPASS}", fontsize=9)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("mean DCT energy captured by top-LxL freq. block")
    axes[1].set_xlabel("L")
    axes[1].set_ylabel("fraction of total spectral energy")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}  (lowpass={LOWPASS} captures "
          f"{energy_frac[LOWPASS-1]*100:.1f}% of mean spectral energy)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=6)
    ap.add_argument("--n_stats", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=os.path.join(REPO, "docs", "idea4", "figs", "lowfreq_evidence"))
    ap.add_argument("--start", type=int, default=0, help="dataset start index (skip trivial/blank leading images)")
    args = ap.parse_args()

    proc, model = load_clip(args.device)

    from datasets import load_dataset
    ds = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev",
                       cache_dir=os.environ["HF_DATASETS_CACHE"])

    n_stats = min(args.n_stats, len(ds) - args.start)
    stat_images = [ds[args.start + i]["image"].convert("RGB") for i in range(n_stats)]

    # spread the qualitative picks across the stats pool for variety
    q_idx = np.linspace(0, n_stats - 1, args.n_images).astype(int)
    qual_images = [stat_images[i] for i in q_idx]

    qualitative(qual_images, proc, model, args.device,
                os.path.join(args.out_dir, "qualitative.png"))
    quantitative(stat_images, proc, model, args.device,
                 os.path.join(args.out_dir, "quantitative.png"))


if __name__ == "__main__":
    main()
