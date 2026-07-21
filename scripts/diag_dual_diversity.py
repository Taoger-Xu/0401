#!/usr/bin/env python
"""Experiment D (idea4 §6): dual-diversity diagnostic, benchmark-free.

For N GQA images, run the (unpatched) CLIP vision tower to get penultimate
patch features + CLS attention, then for four selectors at K=128 measure:
  - cell occupancy      : fraction of the K FPS reference cells that are covered
  - mean pairwise dist  : mean spatial distance between selected patches
  - mean NN feat sim    : mean nearest-neighbour cosine among selected (lower=less redundant)
  - attention capture   : sum of CLS attention over selected / total

Selectors: VisionZip dominant top-K, idea1 spectral, idea3 cls_mmr, idea4 anchor_cover.
Writes docs/idea4/logs/diag/dual_diversity.md
"""
import os
import sys
import argparse

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/jk/datasets")

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from visionzip.prune_ideas import (
    score_and_select_spectral, select_cls_mmr, select_anchor_cover,
    _fps_voronoi_cells, _grid_coords,
)

K = 128
BUDGET = K - 1
GRID = 24


def selectors(feats, attn):
    topk = attn.topk(BUDGET, dim=1).indices
    return {
        "VisionZip top-K": topk,
        "idea1 spectral": score_and_select_spectral(feats, BUDGET),
        "idea3 cls_mmr": select_cls_mmr(feats, attn, BUDGET),
        "idea4 anchor_cover": select_anchor_cover(feats, attn, BUDGET),
    }


def metrics(sel, feats, attn, coords, cells):
    # sel: [budget] indices for one image
    s = sel.view(-1)
    fn = F.normalize(feats[0].float(), dim=-1)          # [N,D]
    fs = fn[s]                                           # [b,D]
    ps = coords[s]                                       # [b,2]
    # cell occupancy
    occ = cells[s].unique().numel() / BUDGET
    # mean pairwise spatial distance
    d = torch.cdist(ps, ps)
    b = d.shape[0]
    pair = d.sum() / (b * (b - 1))
    # mean nearest-neighbour feature cosine
    sim = fs @ fs.t()
    sim.fill_diagonal_(-1)
    nn = sim.max(dim=1).values.mean()
    # attention capture
    cap = attn[0, s].sum() / attn[0].sum()
    return occ, pair.item(), nn.item(), cap.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=398)
    ap.add_argument("--model", default="/home/jk/models/llava-v1.5-7b")
    args = ap.parse_args()

    from llava.model.builder import load_pretrained_model
    tok, model, image_processor, _ = load_pretrained_model(args.model, None, "llava-v1.5-7b", device_map="cuda:0")
    vt = model.model.vision_tower
    vt.to("cuda:0").eval()
    device = "cuda:0"

    from datasets import load_dataset
    ds = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", cache_dir=os.environ["HF_DATASETS_CACHE"])
    n = min(args.n, len(ds))

    coords = _grid_coords(GRID, device)
    cells, _ = _fps_voronoi_cells(GRID, BUDGET, device)

    names = ["VisionZip top-K", "idea1 spectral", "idea3 cls_mmr", "idea4 anchor_cover"]
    acc = {nm: torch.zeros(4) for nm in names}
    used = 0
    for i in range(n):
        img = ds[i]["image"].convert("RGB")
        px = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(device).half()
        with torch.no_grad():
            out = vt.vision_tower(px, output_hidden_states=True, output_attentions=True)
        feats = out.hidden_states[-2][:, 1:, :]                  # [1,576,D]
        attn = out.attentions[-2][:, :, 0, 1:].sum(dim=1).float()  # [1,576]
        sels = selectors(feats, attn)
        for nm, sel in sels.items():
            acc[nm] += torch.tensor(metrics(sel[0:1] if sel.dim() == 2 else sel, feats, attn, coords, cells))
        used += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}")

    lines = ["| Selector | cell占有率↑ | 平均pairwise空间距离↑ | 平均最近邻特征相似度↓ | 注意力捕获率↑ |",
             "|---|---|---|---|---|"]
    for nm in names:
        o, p, s, c = (acc[nm] / used).tolist()
        lines.append(f"| {nm} | {o:.3f} | {p:.3f} | {s:.3f} | {c:.3f} |")
    table = "\n".join(lines)
    print("\n" + table)

    outdir = os.path.join(REPO, "docs", "idea4", "logs", "diag")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "dual_diversity.md"), "w") as f:
        f.write(f"# idea4 双多样性诊断（{used} 张 GQA 图，K={K}）\n\n")
        f.write("↑ 越高越好，↓ 越低越好。cell 占有率以 K 个 FPS cell 为参照；"
                "最近邻特征相似度衡量语义冗余；注意力捕获率衡量显著性保留。\n\n")
        f.write(table + "\n")
    print(f"\nwrote {os.path.join(outdir, 'dual_diversity.md')}")


if __name__ == "__main__":
    main()
