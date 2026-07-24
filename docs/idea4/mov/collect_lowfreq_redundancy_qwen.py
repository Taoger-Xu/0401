"""Collect low-frequency-redundancy data from Qwen2.5-VL's vision tower.

Same measurement as ``collect_lowfreq_redundancy.py`` (which uses LLaVA-1.5's
CLIP-L vision tower), run on a different, architecturally unrelated encoder
(Qwen2.5-VL-7B's ViT, dynamic-resolution 14px patches merged 2x2), on the
*same* 10,000-image COCO caption2017-test sample. The point is to show the
low-frequency-redundancy phenomenon is not specific to CLIP/LLaVA.

Qwen's vision tower does not use a fixed 24x24 grid: each image gets its own
post-merge grid ``(half_h, half_w)`` from ``image_grid_thw`` (dynamic
resolution between ``min_pixels`` and ``max_pixels``). To keep a shared,
model-agnostic x-axis with the CLIP/LLaVA curves, we sweep 24 *relative*
low-pass ratios ``r = 1/24, 2/24, ..., 24/24`` (rather than an absolute k),
keeping the top ``round(r*half_h) x round(r*half_w)`` 2D-DCT coefficients of
that image's own grid before reconstructing and measuring per-token cosine
fidelity. ``r`` lines up with CLIP/LLaVA's ``k/24`` exactly, so both curves
share the x-axis "% of low-frequency coefficients retained, r^2".

Run (restricted to physical GPU 5, matching this workspace's authorization;
needs the ``clse_qwen`` conda env, which has a transformers version new
enough for Qwen2.5-VL):

    CUDA_VISIBLE_DEVICES=5 /home/jk/miniconda3/envs/clse_qwen/bin/python \\
        mov/collect_lowfreq_redundancy_qwen.py
"""

from __future__ import annotations

import glob
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
from datasets import Dataset, concatenate_datasets
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from fourier_compressor.dct import dct2, idct2

MODEL_PATH = "/home/jk/models/qwen2.5-vl-7b"
COCO_IMAGES_GLOB = (
    "/home/jk/datasets/lmms-lab___coco-caption2017/default/0.0.0/*/"
    "coco-caption2017-test-*.arrow"
)
MIN_PIXELS = 256 * 28 * 28  # matches examples/infer_qwen2_5_vl.py defaults
MAX_PIXELS = 2304 * 28 * 28
RATIO_STEPS = 24  # aligned with CLIP/LLaVA's k=1..24 out of a 24x24 grid
STAGE_A_RATIO_INDEX = 11  # r = 12/24 = 0.5 -> 25% coefficient area, same as LLaVA k=12
NUM_IMAGES = 10_000
SEED = 7

OUT_DIR = Path(REPO_ROOT) / "mov" / "data"


def load_images(count: int, seed: int):
    paths = sorted(glob.glob(COCO_IMAGES_GLOB))
    if not paths:
        raise FileNotFoundError(f"no shards matched {COCO_IMAGES_GLOB!r}")
    dataset = concatenate_datasets([Dataset.from_file(p) for p in paths])
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=count, replace=False)
    for i in indices:
        row = dataset[int(i)]
        yield str(row["question_id"]), row["image"].convert("RGB")


def fidelity_curve(coeffs: torch.Tensor, original_tokens: torch.Tensor, half_h: int, half_w: int) -> list[float]:
    """Mean per-token cosine fidelity of the low-pass reconstruction, for every ratio step.

    ``coeffs`` is ``[1, C, half_h, half_w]`` (full-spectrum DCT), ``original_tokens``
    is ``[half_h*half_w, C]``.
    """
    values = []
    for step in range(1, RATIO_STEPS + 1):
        ratio = step / RATIO_STEPS
        k_h = max(1, round(ratio * half_h))
        k_w = max(1, round(ratio * half_w))
        low = torch.zeros_like(coeffs)
        low[..., :k_h, :k_w] = coeffs[..., :k_h, :k_w]
        recon = idct2(low, norm="ortho")  # [1, C, half_h, half_w]
        recon_tokens = recon.reshape(recon.shape[1], -1).transpose(0, 1)  # [half_h*half_w, C]
        sim = F.cosine_similarity(original_tokens, recon_tokens, dim=-1)
        values.append(float(sim.mean().item()))
    return values


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).to(device).eval()
    visual = model.visual

    print(f"sampling {NUM_IMAGES} COCO caption2017-test images (seed={SEED})")

    curves: list[list[float]] = []
    image_ids: list[str] = []
    grid_shapes: list[tuple[int, int]] = []

    with torch.no_grad():
        for index, (image_id, image) in enumerate(load_images(NUM_IMAGES, SEED)):
            batch = processor.image_processor(images=image, return_tensors="pt")
            pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
            grid_thw = batch["image_grid_thw"].to(device)
            assert grid_thw.shape[0] == 1 and int(grid_thw[0, 0].item()) == 1, "expected one static image"

            embeds = visual(pixel_values, grid_thw=grid_thw)  # [half_h*half_w, C]
            half_h = int(grid_thw[0, 1].item()) // 2
            half_w = int(grid_thw[0, 2].item()) // 2
            tokens = embeds.float()  # [half_h*half_w, C]

            grid = tokens.transpose(0, 1).reshape(1, tokens.shape[-1], half_h, half_w)
            coeffs = dct2(grid, norm="ortho")

            curves.append(fidelity_curve(coeffs, tokens, half_h, half_w))
            image_ids.append(image_id)
            grid_shapes.append((half_h, half_w))

            if (index + 1) % 200 == 0 or index + 1 == NUM_IMAGES:
                print(f"  processed {index + 1}/{NUM_IMAGES} images")

    curves_arr = np.array(curves)  # [N, RATIO_STEPS]

    np.savez(
        OUT_DIR / "lowfreq_redundancy_curves_qwen.npz",
        curves=curves_arr,
        ratio_steps=RATIO_STEPS,
        image_ids=np.array(image_ids),
        grid_shapes=np.array(grid_shapes),
    )

    means = curves_arr.mean(axis=0)
    stds = curves_arr.std(axis=0)
    summary = {
        "ratio_steps": RATIO_STEPS,
        "mean_cos_sim": means.tolist(),
        "std_cos_sim": stds.tolist(),
        "min_across_images": curves_arr.min(axis=0).tolist(),
        "max_across_images": curves_arr.max(axis=0).tolist(),
        "_meta": {
            "num_images": len(image_ids),
            "seed": SEED,
            "stage_a_ratio_index": STAGE_A_RATIO_INDEX,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
            "model_path": MODEL_PATH,
        },
    }
    with (OUT_DIR / "summary_qwen.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"r=0.5 (25% coeff area): mean={means[STAGE_A_RATIO_INDEX]:.3f} "
        f"std={stds[STAGE_A_RATIO_INDEX]:.3f} "
        f"min={curves_arr[:, STAGE_A_RATIO_INDEX].min():.3f} "
        f"max={curves_arr[:, STAGE_A_RATIO_INDEX].max():.3f} across {len(image_ids)} images"
    )
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
