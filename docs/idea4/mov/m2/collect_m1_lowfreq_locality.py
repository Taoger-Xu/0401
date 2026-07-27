"""M1 motivation experiment: does low-frequency structure characterize local
replaceability of CLIP patch tokens?  (docs/idea4/motivation_experiment_plan.md, M1)

For every image we compute, per patch token i on the 24x24 CLIP-L/14-336
penultimate-layer grid:

  - h_i(L)            : DCT low-pass (L x L) reconstruction residual
                         ||f_i - IDCT(Pi_L(DCT(f)))_i||, L in {8,12,16,20},
                         reusing visionzip.prune_ideas.lowfreq_reconstruct
                         (the exact function idea1/idea4 use at inference
                         time -- no reimplementation).
  - local_cosine_i    : mean cosine similarity to the (up to 8) grid neighbours.
  - centroid_cosine_i : cosine similarity to the centroid of the token's
                         FPS-Voronoi cell (P=192 cells, reusing
                         visionzip.prune_ideas._fps_voronoi_cells -- the same
                         partition used by score_and_select_spectral /
                         select_anchor_cover in the main method at the
                         K=192 "practical" budget named in the plan's M0).
  - replacement_error_i: substitute f_i with the *medoid* of its (up to 8)
                         grid neighbours (the neighbour whose average cosine
                         similarity to the other neighbours is highest -- an
                         actual neighbouring token, not a synthetic average)
                         and measure cos/L2 error between f_i and that medoid.

We additionally pick, per image, the per-cell "Top-1" token under the medoid
criterion (argmax centroid_cosine within the cell -- the same criterion
`score_and_select_spectral` uses before adding the low-frequency term), and
re-derive that Top-1 choice after three mild input perturbations (color
jitter, slight resize, horizontal flip), realigning grid coordinates so the
comparison is apples-to-apples. `representative_stability` of a cell is the
fraction of the 3 perturbations under which the Top-1 choice is unchanged;
it is a cell-level quantity, broadcast to member tokens for the pooled
per-token analyses and also kept separately at winner-only granularity
(see collect() docstring below for exactly which arrays hold which).

Data (documented substitution, see docs/idea4/mov/m2/README.md):
  - "GQA val" in the plan: the lmms-lab/GQA packaging used throughout this
    repo has no `val` config (only train/test/testdev/challenge/submission).
    We use `test_balanced_images` (2,987 unique images, unlabeled -- M1 only
    needs images, no QA labels) and sample 1,000 at random, seed=7.
  - "COCO Caption val": lmms-lab/COCO-Caption2017 `default` config, split
    `val` (5,000 images, genuinely has a `val` split with per-row unique
    `file_name`), sample 1,000 at random, seed=7.

Run (restricted to a free physical GPU; GPU7 is a known-bad card in this
workspace, avoid it -- see memory idea4-qwen-never-ran-prefix):

    CUDA_VISIBLE_DEVICES=0 /home/jk/miniconda3/envs/llava_visiPruner/bin/python \
        docs/idea4/mov/m2/collect_m1_lowfreq_locality.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MOV_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in MOV_DIR.parents if (p / "visionzip").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_DATASETS_CACHE", "/home/jk/datasets")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from visionzip.prune_ideas import lowfreq_reconstruct, _fps_voronoi_cells, _dct_matrix

CLIP_PATH = "/home/jk/models/clip-vit-large-patch14-336"
GRID = 24
N_TOKENS = GRID * GRID
SELECT_LAYER = -2                        # llava-v1.5-7b: mm_vision_select_layer
L_VALUES = (8, 12, 16, 20)
L_MAIN = 16
P_CELLS = 192                             # FPS-Voronoi cell count, matches K=192 practical budget
N_GQA = 1000
N_COCO = 1000
SEED = 7

OUT_DIR = MOV_DIR / "data"


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_image_pool():
    """Returns list of (source, image_id, PIL.Image RGB)."""
    from datasets import load_dataset

    items = []
    rng = np.random.default_rng(SEED)

    gqa = load_dataset("lmms-lab/GQA", "test_balanced_images", split="test",
                        cache_dir=os.environ["HF_DATASETS_CACHE"])
    idx = rng.choice(len(gqa), size=min(N_GQA, len(gqa)), replace=False)
    for i in idx:
        row = gqa[int(i)]
        items.append(("gqa", str(row["id"]), row["image"].convert("RGB")))

    coco = load_dataset("lmms-lab/COCO-Caption2017", "default", split="val",
                         cache_dir=os.environ["HF_DATASETS_CACHE"])
    rng2 = np.random.default_rng(SEED + 1)
    idx2 = rng2.choice(len(coco), size=min(N_COCO, len(coco)), replace=False)
    for i in idx2:
        row = coco[int(i)]
        items.append(("coco", str(row["file_name"]), row["image"].convert("RGB")))

    return items


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------
@torch.no_grad()
def patch_feats_for(pil_img, processor, model, device):
    px = processor(images=pil_img, return_tensors="pt")["pixel_values"].to(device)
    hs = model(px, output_hidden_states=True).hidden_states[SELECT_LAYER]
    return hs[:, 1:, :].float()  # [1, 576, 1024], drop CLS


# --------------------------------------------------------------------------
# per-image metric computation
# --------------------------------------------------------------------------
def neighbor_stack(x, grid):
    """x: [1, N, D] on grid x grid layout -> [8, 1, H, W, D] replicate-padded
    8-neighbour stack, same convention as score_local_variation."""
    B, N, D = x.shape
    g = x.view(B, grid, grid, D)
    pad = F.pad(g.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate").permute(0, 2, 3, 1)
    slabs = []
    for di in range(3):
        for dj in range(3):
            if di == 1 and dj == 1:
                continue
            slabs.append(pad[:, di:di + grid, dj:dj + grid, :])
    return torch.stack(slabs, dim=0)  # [8, B, H, W, D]


def compute_local_and_replacement(feats, grid):
    """feats: [1, N, D] raw (unnormalised). Returns:
      local_cosine [N], replacement_error_cos [N], replacement_error_l2 [N]
    """
    B, N, D = feats.shape
    fn = F.normalize(feats, dim=-1)
    nb_raw = neighbor_stack(feats, grid)   # [8,1,H,W,D]
    nb_fn = neighbor_stack(fn, grid)       # [8,1,H,W,D]

    g_fn = fn.view(B, grid, grid, D)
    local_cosine = torch.stack([(g_fn * nb_fn[k]).sum(-1) for k in range(8)], 0).mean(0)  # [1,H,W]

    # pairwise cosine among the 8 neighbours -> medoid neighbour per token
    sim = torch.einsum("aBHWD,bBHWD->abBHW", nb_fn, nb_fn)      # [8,8,1,H,W]
    avg_sim = (sim.sum(0) - 1.0) / 7.0                           # [8,1,H,W], exclude self (sim=1)
    medoid_idx = avg_sim.argmax(0)                                # [1,H,W]

    nb_raw_flat = nb_raw.view(8, B, grid * grid, D)
    idx_flat = medoid_idx.view(B, grid * grid, 1).expand(B, grid * grid, D)
    medoid_feat = torch.gather(nb_raw_flat, 0, idx_flat.unsqueeze(0).expand(8, -1, -1, -1))[0]  # [B,N,D]

    rep_cos_err = 1.0 - F.cosine_similarity(feats, medoid_feat, dim=-1)     # [B,N]
    rep_l2_err = (feats - medoid_feat).norm(dim=-1)                          # [B,N]

    return (local_cosine.view(B, N)[0].cpu().numpy(),
            rep_cos_err[0].cpu().numpy(),
            rep_l2_err[0].cpu().numpy())


def cell_centroid_cosine_and_winner(feats, cells, onehot, P):
    """feats: [1,N,D] raw. Returns centroid_cosine [N] and winner_idx [P]
    (argmax centroid_cosine token index within each cell, i.e. the medoid
    criterion score_and_select_spectral uses before the low-freq term)."""
    B, N, D = feats.shape
    fn = F.normalize(feats, dim=-1)
    cell_sum = torch.einsum("nc,bnd->bcd", onehot, fn)      # [1,P,D]
    centroid = F.normalize(cell_sum, dim=-1)                 # [1,P,D]
    cells_b = cells.view(1, N)
    cent_per_tok = centroid.gather(1, cells_b.unsqueeze(-1).expand(B, N, D))
    centroid_cosine = (fn * cent_per_tok).sum(-1)[0]          # [N]

    NEG = torch.finfo(centroid_cosine.dtype).min
    score = centroid_cosine.view(1, N)
    cmax = score.new_full((1, P), NEG).scatter_reduce(1, cells_b, score, reduce="amax", include_self=True)
    is_win = score >= cmax.gather(1, cells_b)
    idx_all = torch.arange(N, device=feats.device).view(1, N)
    cand = torch.where(is_win, idx_all, torch.full_like(idx_all, N))
    winner = cand.new_full((1, P), N).scatter_reduce(1, cells_b, cand, reduce="amin", include_self=True)
    winner = winner.clamp(max=N - 1)[0]                        # [P]

    return centroid_cosine.cpu().numpy(), winner.cpu().numpy()


def residuals_all_L(feats, grid, L_values):
    out = {}
    for L in L_values:
        rec = lowfreq_reconstruct(feats, grid=grid, lowpass=L)
        out[L] = (feats - rec).norm(dim=-1)[0].cpu().numpy()
    return out


# --------------------------------------------------------------------------
# perturbations (for representative_stability)
# --------------------------------------------------------------------------
COLOR_JITTER = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03)


def perturb_jitter(img):
    return COLOR_JITTER(img)


def perturb_resize(img):
    w, h = img.size
    scale = 0.92  # mild downscale; CLIPImageProcessor still resizes/crops to 336 after
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)


def perturb_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def realign_flip_feats(feats, grid):
    """feats: [1,N,D] computed from a horizontally-flipped image. The fixed
    FPS-Voronoi cell partition is NOT left-right symmetric (anchors start at
    the top-left corner), so grouping flipped-image tokens by the unflipped
    cell membership mixes physically unrelated regions. Flip the *feature
    grid* back into the original coordinate frame first (mirror the width
    axis) so cell membership / centroid / winner all refer to the same
    physical regions as the unperturbed pass."""
    B, N, D = feats.shape
    g = feats.view(B, grid, grid, D)
    g = torch.flip(g, dims=[2])
    return g.reshape(B, N, D)


# --------------------------------------------------------------------------
# main collection loop
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run (20 images) for correctness check")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    from transformers import CLIPImageProcessor, CLIPVisionModel
    processor = CLIPImageProcessor.from_pretrained(CLIP_PATH)
    model = CLIPVisionModel.from_pretrained(CLIP_PATH, torch_dtype=torch.float32).to(device).eval()

    items = load_image_pool()
    if args.smoke:
        items = items[:10] + items[-10:]
    print(f"collecting on {len(items)} images (device={device})")

    cells, onehot = _fps_voronoi_cells(GRID, P_CELLS, device)   # fixed partition, image-independent
    cells_np = cells.cpu().numpy()

    # pooled per-token arrays
    all_h = {L: [] for L in L_VALUES}
    all_local_cos = []
    all_centroid_cos = []
    all_rep_cos_err = []
    all_rep_l2_err = []
    all_token_stability = []   # broadcast cell-level stability -> member tokens
    all_source = []            # 0=gqa, 1=coco, per token
    all_image_idx = []

    # winner-only arrays (one entry per (image, cell))
    winner_h = {L: [] for L in L_VALUES}
    winner_stability = []
    winner_stability_by_kind = {"jitter": [], "resize": [], "flip": []}
    winner_local_cos = []
    winner_source = []

    meta = []

    with torch.no_grad():
        for img_i, (source, image_id, pil_img) in enumerate(items):
            feats = patch_feats_for(pil_img, processor, model, device)  # [1,576,1024]

            h_by_L = residuals_all_L(feats, GRID, L_VALUES)
            local_cos, rep_cos_err, rep_l2_err = compute_local_and_replacement(feats, GRID)
            centroid_cos, winner = cell_centroid_cosine_and_winner(feats, cells, onehot, P_CELLS)

            # ---- representative_stability under 3 perturbations ----
            agree = np.zeros(P_CELLS, dtype=np.float32)
            for kind, fn_pert, needs_flip_realign in [
                ("jitter", perturb_jitter, False),
                ("resize", perturb_resize, False),
                ("flip", perturb_flip, True),
            ]:
                p_img = fn_pert(pil_img)
                p_feats = patch_feats_for(p_img, processor, model, device)
                if needs_flip_realign:
                    p_feats = realign_flip_feats(p_feats, GRID)
                p_centroid_cos, p_winner = cell_centroid_cosine_and_winner(p_feats, cells, onehot, P_CELLS)
                kind_agree = (p_winner == winner).astype(np.float32)
                winner_stability_by_kind[kind].append(kind_agree)
                agree += kind_agree
            cell_stability = agree / 3.0  # [P]
            token_stability = cell_stability[cells_np]  # [N] broadcast

            src_flag = 0 if source == "gqa" else 1

            for L in L_VALUES:
                all_h[L].append(h_by_L[L])
                winner_h[L].append(h_by_L[L][winner])
            all_local_cos.append(local_cos)
            all_centroid_cos.append(centroid_cos)
            all_rep_cos_err.append(rep_cos_err)
            all_rep_l2_err.append(rep_l2_err)
            all_token_stability.append(token_stability)
            all_source.append(np.full(N_TOKENS, src_flag, dtype=np.int8))
            all_image_idx.append(np.full(N_TOKENS, img_i, dtype=np.int32))

            winner_stability.append(cell_stability)
            winner_local_cos.append(local_cos[winner])
            winner_source.append(np.full(P_CELLS, src_flag, dtype=np.int8))

            meta.append({"source": source, "image_id": image_id})

            if (img_i + 1) % 100 == 0 or (img_i + 1) == len(items):
                print(f"  {img_i + 1}/{len(items)}")

    def cat(lst):
        return np.concatenate(lst, axis=0)

    save_kwargs = dict(
        local_cosine=cat(all_local_cos).astype(np.float32),
        centroid_cosine=cat(all_centroid_cos).astype(np.float32),
        replacement_error_cos=cat(all_rep_cos_err).astype(np.float32),
        replacement_error_l2=cat(all_rep_l2_err).astype(np.float32),
        token_stability=cat(all_token_stability).astype(np.float32),
        source=cat(all_source),
        image_idx=cat(all_image_idx),
        winner_stability=cat(winner_stability).astype(np.float32),
        winner_stability_jitter=cat(winner_stability_by_kind["jitter"]).astype(np.float32),
        winner_stability_resize=cat(winner_stability_by_kind["resize"]).astype(np.float32),
        winner_stability_flip=cat(winner_stability_by_kind["flip"]).astype(np.float32),
        winner_local_cosine=cat(winner_local_cos).astype(np.float32),
        winner_source=cat(winner_source),
        p_cells=P_CELLS,
        grid=GRID,
        l_values=np.array(L_VALUES),
        l_main=L_MAIN,
        n_images=len(items),
    )
    for L in L_VALUES:
        save_kwargs[f"h_L{L}"] = cat(all_h[L]).astype(np.float32)
        save_kwargs[f"winner_h_L{L}"] = cat(winner_h[L]).astype(np.float32)

    out_path = OUT_DIR / ("m1_tokens_smoke.npz" if args.smoke else "m1_tokens.npz")
    np.savez_compressed(out_path, **save_kwargs)
    with (OUT_DIR / ("m1_meta_smoke.json" if args.smoke else "m1_meta.json")).open("w") as f:
        json.dump({"images": meta, "n_images": len(items), "p_cells": P_CELLS,
                    "grid": GRID, "l_values": list(L_VALUES), "l_main": L_MAIN,
                    "seed": SEED}, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
