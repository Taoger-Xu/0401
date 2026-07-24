"""Collect data for the vision-token low-frequency-redundancy line chart.

Question this experiment answers: if a CLIP-L/14-336 patch-token grid (24x24,
576 tokens, taken at LLaVA-1.5's ``mm_vision_select_layer=-2`` -- the same
"clip_embedding" stage already used throughout this repo's spectral analyses)
is reconstructed from only its lowest ``k x k`` 2D-DCT coefficients, how well
does that low-pass reconstruction approximate the original per-token feature?
High per-token cosine similarity at small k means most of a token's content is
already implied by its low-frequency neighbourhood -- i.e. redundant, not
independent information.

For every sampled image we compute, per token, ``cos(f_i, IDCT_k(DCT(f))_i)``
averaged over all 576 tokens, for every ``k = 1..24``. This gives one
fidelity-vs-compression-ratio curve per image, over ``NUM_IMAGES=10,000``
COCO caption2017-test images -- large enough that the aggregate figure (see
``plot_lowfreq_redundancy.py``) reports a mean +/- CI band rather than
per-image lines. The companion script ``collect_lowfreq_redundancy_qwen.py``
runs the same measurement on Qwen2.5-VL's vision tower on the same image
sample, so the figure can show the phenomenon holds across two unrelated
vision encoders.

Model/data are loaded directly (no LLaVA generation needed): only the CLIP
vision tower is required to reproduce the "clip_embedding" features.

Run (restricted to physical GPU 5, matching this workspace's authorization):

    CUDA_VISIBLE_DEVICES=5 python mov/collect_lowfreq_redundancy.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
    raise RuntimeError(
        "This experiment is restricted to physical GPU 5; set CUDA_VISIBLE_DEVICES=5."
    )

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import CLIPImageProcessor, CLIPVisionModel

from fourier_compressor.dct import dct2, idct2

CLIP_PATH = "/home/jk/models/clip-vit-large-patch14-336"
# 14 shards, 40,670 unique images total -- large enough to sample 10k without
# replacement. This is the same image pool used by the Qwen2.5-VL collection
# script (collect_lowfreq_redundancy_qwen.py) so both models are evaluated on
# an identical sample.
COCO_IMAGES_GLOB = (
    "/home/jk/datasets/lmms-lab___coco-caption2017/default/0.0.0/*/"
    "coco-caption2017-test-*.arrow"
)
GRID_H, GRID_W = 24, 24
SELECT_LAYER = -2  # matches llava-v1.5-7b config.json: mm_vision_select_layer
K_VALUES = tuple(range(1, GRID_H + 1))  # full sweep -> smooth per-image curves
STAGE_A_K = 12  # 144/576 = 25% of coefficients -- this project's Stage A ratio
NUM_IMAGES = 10_000  # large-N pool backing the "natural phenomenon across images" claim
SEED = 7

OUT_DIR = Path(REPO_ROOT) / "mov" / "data"


def load_images(count: int, seed: int):
    import glob

    from datasets import concatenate_datasets

    paths = sorted(glob.glob(COCO_IMAGES_GLOB))
    if not paths:
        raise FileNotFoundError(f"no shards matched {COCO_IMAGES_GLOB!r}")
    dataset = concatenate_datasets([Dataset.from_file(p) for p in paths])
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=count, replace=False)
    for i in indices:
        row = dataset[int(i)]
        # NB: the lmms-eval packaging of this split mislabels columns; the
        # per-image COCO filename actually lives in "question_id", not "id"
        # or "file_name" (both constant across the whole split).
        yield str(row["question_id"]), row["image"].convert("RGB")


def fidelity_curve(coeffs: torch.Tensor, original_tokens: torch.Tensor) -> list[float]:
    """Mean per-token cosine fidelity of the k x k low-pass reconstruction, for every k.

    ``coeffs`` is ``[1, C, H, W]`` (full-spectrum DCT), ``original_tokens`` is
    ``[H*W, C]``. Returns a list aligned with :data:`K_VALUES`.
    """
    values = []
    for k in K_VALUES:
        low = torch.zeros_like(coeffs)
        low[..., :k, :k] = coeffs[..., :k, :k]
        recon = idct2(low, norm="ortho")  # [1, C, H, W]
        recon_tokens = recon.reshape(recon.shape[1], -1).transpose(0, 1)  # [H*W, C]
        sim = F.cosine_similarity(original_tokens, recon_tokens, dim=-1)
        values.append(float(sim.mean().item()))
    return values


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    processor = CLIPImageProcessor.from_pretrained(CLIP_PATH)
    model = CLIPVisionModel.from_pretrained(CLIP_PATH, torch_dtype=torch.float32).to(device).eval()

    print(f"sampling {NUM_IMAGES} COCO caption2017-test images (seed={SEED})")

    curves: list[list[float]] = []
    image_ids: list[str] = []

    with torch.no_grad():
        for index, (image_id, image) in enumerate(load_images(NUM_IMAGES, SEED)):
            pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(
                device=device, dtype=torch.float32
            )
            hidden_states = model(pixel_values, output_hidden_states=True).hidden_states
            tokens = hidden_states[SELECT_LAYER][:, 1:, :].float()  # [1, 576, 1024], drop CLS

            grid = tokens.transpose(1, 2).reshape(1, tokens.shape[-1], GRID_H, GRID_W)
            coeffs = dct2(grid, norm="ortho")
            flat_tokens = tokens.squeeze(0)  # [576, 1024]

            curves.append(fidelity_curve(coeffs, flat_tokens))
            image_ids.append(image_id)

            if (index + 1) % 200 == 0 or index + 1 == NUM_IMAGES:
                print(f"  processed {index + 1}/{NUM_IMAGES} images")

    curves_arr = np.array(curves)  # [N, len(K_VALUES)]

    np.savez(
        OUT_DIR / "lowfreq_redundancy_curves.npz",
        curves=curves_arr,
        k_values=np.array(K_VALUES),
        image_ids=np.array(image_ids),
        stage_a_k=STAGE_A_K,
        grid_h=GRID_H,
        grid_w=GRID_W,
    )

    means = curves_arr.mean(axis=0)
    stds = curves_arr.std(axis=0)
    summary = {
        "k_values": list(K_VALUES),
        "mean_cos_sim": means.tolist(),
        "std_cos_sim": stds.tolist(),
        "min_across_images": curves_arr.min(axis=0).tolist(),
        "max_across_images": curves_arr.max(axis=0).tolist(),
        "_meta": {
            "num_images": len(image_ids),
            "seed": SEED,
            "stage_a_k": STAGE_A_K,
            "grid_h": GRID_H,
            "grid_w": GRID_W,
            "select_layer": SELECT_LAYER,
        },
    }
    with (OUT_DIR / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    stage_a_index = K_VALUES.index(STAGE_A_K)
    print(
        f"k={STAGE_A_K} (25% coeffs): mean={means[stage_a_index]:.3f} "
        f"std={stds[stage_a_index]:.3f} "
        f"min={curves_arr[:, stage_a_index].min():.3f} "
        f"max={curves_arr[:, stage_a_index].max():.3f} across {len(image_ids)} images"
    )
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
