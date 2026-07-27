"""M1 qualitative figure: original image / real CLIP features (PCA-RGB) /
16x16-DCT low-pass reconstruction (PCA-RGB) / residual heatmap, for a
handful of images spanning the residual distribution.

Reuses the same feature extraction and lowfreq_reconstruct call as
collect_m1_lowfreq_locality.py; images are looked up by (source, image_id)
so this reproduces exactly the images already scored there (picked to
span the low/mid/high mean-residual range via docs/idea4/mov/m2/data/m1_meta.json).

Run:
    CUDA_VISIBLE_DEVICES=6 /home/jk/miniconda3/envs/llava_visiPruner/bin/python \
        docs/idea4/mov/m2/plot_m1_qualitative.py
"""
from __future__ import annotations

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from visionzip.prune_ideas import lowfreq_reconstruct

CLIP_PATH = "/home/jk/models/clip-vit-large-patch14-336"
GRID = 24
LOWPASS = 16
SELECT_LAYER = -2

# (source, image_id) picked to span low -> high mean h_L16, see selection log
PICKS = [
    ("coco", "000000539962.jpg"),
    ("gqa", "n205361"),
    ("gqa", "n142149"),
    ("coco", "000000273198.jpg"),
    ("coco", "000000223188.jpg"),
]


def find_images(picks, device):
    from datasets import load_dataset

    need_gqa = [pid for src, pid in picks if src == "gqa"]
    need_coco = [pid for src, pid in picks if src == "coco"]
    found = {}

    if need_gqa:
        gqa = load_dataset("lmms-lab/GQA", "test_balanced_images", split="test",
                            cache_dir=os.environ["HF_DATASETS_CACHE"])
        want = set(need_gqa)
        for row in gqa:
            if str(row["id"]) in want:
                found[("gqa", str(row["id"]))] = row["image"].convert("RGB")
                want.discard(str(row["id"]))
                if not want:
                    break

    if need_coco:
        coco = load_dataset("lmms-lab/COCO-Caption2017", "default", split="val",
                             cache_dir=os.environ["HF_DATASETS_CACHE"])
        want = set(need_coco)
        for row in coco:
            if row["file_name"] in want:
                found[("coco", row["file_name"])] = row["image"].convert("RGB")
                want.discard(row["file_name"])
                if not want:
                    break

    return [found[p] for p in picks]


def pca_rgb(feats_np, ref_pca=None):
    from sklearn.decomposition import PCA
    if ref_pca is None:
        ref_pca = PCA(n_components=3, random_state=0).fit(feats_np)
    return ref_pca.transform(feats_np), ref_pca


def normalize_channels(*arrs):
    stacked = np.concatenate(arrs, axis=0)
    lo, hi = stacked.min(0, keepdims=True), stacked.max(0, keepdims=True)
    return [(a - lo) / (hi - lo + 1e-6) for a in arrs]


def upsample_nearest(arr2d, size):
    a = np.asarray(arr2d)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    mode = "RGB" if a.ndim == 3 else "L"
    im = Image.fromarray(a, mode=mode)
    return np.asarray(im.resize((size, size), Image.NEAREST))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from transformers import CLIPImageProcessor, CLIPVisionModel
    processor = CLIPImageProcessor.from_pretrained(CLIP_PATH)
    model = CLIPVisionModel.from_pretrained(CLIP_PATH, torch_dtype=torch.float32).to(device).eval()

    images = find_images(PICKS, device)
    n = len(images)
    size = 336
    fig, axes = plt.subplots(n, 4, figsize=(4 * 3.1, n * 3.1))
    col_titles = ["original image", "real features (PCA-RGB)",
                  f"{LOWPASS}x{GRID} DCT low-pass recon (PCA-RGB)", "residual  ||H_i - rec_i||"]

    with torch.no_grad():
        for row, img in enumerate(images):
            img_rs = img.resize((size, size), Image.BILINEAR)
            img_np = np.asarray(img_rs).astype(np.float32) / 255.0

            px = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
            hs = model(px, output_hidden_states=True).hidden_states[SELECT_LAYER]
            feats = hs[:, 1:, :].float()  # [1,576,1024]
            rec = lowfreq_reconstruct(feats, grid=GRID, lowpass=LOWPASS)
            residual = (feats - rec).norm(dim=-1)[0].cpu().numpy()

            feats_np = feats[0].cpu().numpy()
            rec_np = rec[0].cpu().numpy()
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

            for c, arr in zip(range(4), [img_np, real_rgb, rec_rgb, overlay]):
                ax = axes[row, c]
                ax.imshow(arr)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(col_titles[c], fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"mean h={residual.mean():.1f}", fontsize=9)

    fig.suptitle("M1 qualitative: low-frequency residual marks non-redundant (kept-worthy) tokens\n"
                 f"(real CLIP-L/14@336 penultimate features, lowpass={LOWPASS}/{GRID}, rows ordered by increasing mean residual)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = MOV_DIR / "figs" / "m1_qualitative.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
