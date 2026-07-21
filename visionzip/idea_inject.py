# ------------------------------------------------------------------------
# Injection glue that plugs the idea1/idea2/idea3 selectors into the exact
# spot VisionZip uses (CLIPVisionTower.forward), so the whole downstream
# LLaVA/lmms-eval pipeline is reused unchanged. Selected via env vars:
#   IDEA_METHOD in {spectral, local_var, cls_mmr}
#   IDEA_K       total visual tokens the LLM sees (CLS + K-1 patches)
#   IDEA_LAMBDA  redundancy weight for cls_mmr (default 0.5)
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import CLIP_EncoderLayer_forward, CLIPAttention_forward, apply_info
from .prune_ideas import (
    score_local_variation,
    score_and_select_spectral,
    select_cls_mmr,
    select_anchor_cover,
    select_sem_var_mmr,
    select_detail_mmr,
    lowfreq_reconstruct,
)
from .llava_arch import (
    prepare_inputs_labels_for_multimodal_visionzip,
    encode_images_visionzip,
    encode_images_visionzip_multi,
    restore_image_features_sorted,
)


@torch.no_grad()
def CLIPVisionTower_Idea_forward(self, images):
    if type(images) is list:
        out = []
        for image in images:
            fo = self.vision_tower(
                image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                output_hidden_states=True, output_attentions=True)
            out.append(_idea_select(self, fo, image.dtype)[0])
        return out, None

    fo = self.vision_tower(
        images.to(device=self.device, dtype=self.dtype),
        output_hidden_states=True, output_attentions=True)
    hidden_states_save, all_indices = _idea_select(self, fo, images.dtype)
    return hidden_states_save, all_indices


def _idea_select(self, image_forward_outs, out_dtype):
    info = self.vision_tower._info
    method = info["idea_method"]
    K = info["idea_k"]
    lam = info.get("idea_lambda", 0.5)

    hidden_states = image_forward_outs.hidden_states[-2]     # [B, 577, D]
    B, T, D = hidden_states.shape
    cls_tok = hidden_states[:, 0:1, :]                       # [B, 1, D]
    patch_feats = hidden_states[:, 1:, :]                    # [B, 576, D]
    grid = int(round((T - 1) ** 0.5))
    ctx = int(info.get("idea_contextual", 0))
    budget = K - 1 - ctx                                     # patches to keep (CLS is +1)

    # ---- reconstruction methods (not pruning): replace patch tokens by their
    # DCT low-pass reconstruction; token identity is destroyed, so no gather.
    if method in ("lowfreq_recon", "dct_down"):
        with torch.autocast(device_type=patch_feats.device.type, enabled=False):
            if method == "lowfreq_recon":
                rec = lowfreq_reconstruct(patch_feats.float(), grid=grid,
                                          lowpass=info.get("idea_lowpass", 16))
            else:
                rec = lowfreq_reconstruct(patch_feats.float(), grid=grid,
                                          out_grid=info.get("idea_out_grid", 14))
        hidden_states_save = torch.cat([cls_tok, rec.to(cls_tok.dtype)], dim=1).to(out_dtype)
        T_out = hidden_states_save.shape[1]
        all_indices = torch.arange(
            T_out, device=hidden_states.device).view(1, T_out).expand(B, T_out)
        return hidden_states_save, all_indices

    # Selection scoring must run in float32 with autocast OFF: under the model's
    # fp16 autocast, ops like F.normalize silently emit Half while cached DCT /
    # one-hot tensors stay Float32, which breaks einsum (dtype mismatch) and
    # hurts DCT precision. Selection is index-only, so fp32 here is free.
    dev_type = patch_feats.device.type
    with torch.autocast(device_type=dev_type, enabled=False):
        patch_feats_f = patch_feats.float()
        if method == "local_var":
            score = score_local_variation(patch_feats_f, grid=grid)   # [B, N]
            sel = score.topk(budget, dim=1).indices                   # [B, budget]
        elif method == "spectral":
            sel = score_and_select_spectral(patch_feats_f, budget, grid=grid)
        elif method == "cls_mmr":
            attn = image_forward_outs.attentions[-2]                  # [B, heads, T, T]
            cls_attn = attn[:, :, 0, 1:].sum(dim=1).float()           # [B, N]
            sel = select_cls_mmr(patch_feats_f, cls_attn, budget, lam=lam)
        elif method == "detail_mmr":
            attn = image_forward_outs.attentions[-2]                  # [B, heads, T, T]
            cls_attn = attn[:, :, 0, 1:].sum(dim=1).float()           # [B, N]
            sel = select_detail_mmr(patch_feats_f, cls_attn, budget, lam=lam,
                                    gamma=info.get("idea_gamma", 1.0),
                                    p=info.get("idea_detail_p", 2.0),
                                    detail_src=info.get("idea_detail_src", "local_var"),
                                    grid=grid, lowpass=info.get("idea_lowpass", 16))
        elif method == "anchor_cover":
            attn = image_forward_outs.attentions[-2]                  # [B, heads, T, T]
            cls_attn = attn[:, :, 0, 1:].sum(dim=1).float()           # [B, N]
            sel = select_anchor_cover(patch_feats_f, cls_attn, budget, grid=grid,
                                      rho=info.get("idea_rho", 0.5), lam=lam,
                                      sigma=info.get("idea_sigma", 2.0),
                                      cover_factor=info.get("idea_cover_factor", 3.0))
        elif method == "sem_var":
            attn = image_forward_outs.attentions[-2]                  # [B, heads, T, T]
            cls_attn = attn[:, :, 0, 1:].sum(dim=1).float()           # [B, N]
            sel = select_sem_var_mmr(patch_feats_f, cls_attn, budget, grid=grid,
                                     lam=lam, w_v=info.get("idea_w_var", 0.3),
                                     gamma=info.get("idea_gamma", 0.5))
        else:
            raise ValueError(f"unknown IDEA_METHOD: {method}")

    sel, _ = sel.sort(dim=1)                                    # keep raster order
    gather_idx = sel.unsqueeze(-1).expand(-1, -1, D)
    kept = torch.gather(patch_feats, 1, gather_idx)             # [B, budget, D]

    # idea5 §4 controlled-merge probe (NOT pruning): each pruned token is
    # averaged into its most similar kept token. Token count stays `budget`;
    # used only to measure how much of the TextVQA gap needs merging.
    if info.get("idea_merge", False):
        with torch.autocast(device_type=patch_feats.device.type, enabled=False):
            feats_f = patch_feats.float()
            fn = F.normalize(feats_f, dim=-1)                        # [B, N, D]
            kept_f = kept.float()
            kept_fn = F.normalize(kept_f, dim=-1)                    # [B, budget, D]
            sim = torch.einsum("bnd,bkd->bnk", fn, kept_fn)          # [B, N, budget]
            assign = sim.argmax(dim=-1)                              # [B, N]
            kept_mask = torch.zeros(B, patch_feats.shape[1], dtype=torch.bool,
                                    device=patch_feats.device).scatter_(1, sel, True)
            w = (~kept_mask).float()                                 # pruned tokens only
            sums = torch.zeros_like(kept_f).scatter_add_(
                1, assign.unsqueeze(-1).expand(-1, -1, D), feats_f * w.unsqueeze(-1))
            # explicit dtype: the caller may have changed torch's default dtype
            cnt = torch.zeros(B, kept_f.shape[1], device=patch_feats.device,
                              dtype=torch.float32).scatter_add_(1, assign, w)
            kept = ((kept_f + sums) / (1.0 + cnt.unsqueeze(-1))).to(patch_feats.dtype)

    # idea5 §4 VisionZip-style contextual tokens: merge the *pruned* tokens into
    # `ctx` extra tokens (uniform-stride targets among the pruned set, assignment
    # by key-metric similarity, VisionZip's exact recipe) appended after the kept
    # set. Dominant/kept tokens are left untouched.
    ctx_tok = None
    if ctx > 0:
        metric = self.vision_tower.vision_model.encoder.layers[-2].metric  # [B, T, dm]
        with torch.autocast(device_type=patch_feats.device.type, enabled=False):
            N = patch_feats.shape[1]
            keep_mask = torch.zeros(B, N, dtype=torch.bool,
                                    device=patch_feats.device).scatter_(1, sel, True)
            Mp = N - budget                                   # pruned count
            metric_f = metric[:, 1:, :].float()[~keep_mask].view(B, Mp, -1)
            hidden_f = patch_feats.float()[~keep_mask].view(B, Mp, D)
            metric_n = metric_f / metric_f.norm(dim=-1, keepdim=True)
            step = max(1, Mp // ctx)
            tgt = torch.arange(0, Mp, step, device=patch_feats.device)[:ctx]
            is_tgt = torch.isin(torch.arange(Mp, device=patch_feats.device), tgt)
            target_tokens = metric_n[:, tgt, :]
            tokens_to_merge = metric_n[:, ~is_tgt, :]
            similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
            assign_one_hot = torch.zeros(B, tokens_to_merge.shape[1], ctx,
                                         dtype=torch.float32, device=patch_feats.device)
            assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
            counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
            aggregated = torch.bmm(assign_one_hot.transpose(1, 2),
                                   hidden_f[:, ~is_tgt, :]) / counts
            ctx_tok = (hidden_f[:, tgt, :] + aggregated).to(patch_feats.dtype)

    parts = [cls_tok, kept] if ctx_tok is None else [cls_tok, kept, ctx_tok]
    hidden_states_save = torch.cat(parts, dim=1).to(out_dtype)  # [B, K, D]

    all_indices = torch.cat([
        torch.zeros((B, 1), dtype=sel.dtype, device=sel.device),
        sel + 1,
    ], dim=1)
    return hidden_states_save, all_indices


def visionzip_idea(model, method, k, lam=0.5, rho=0.5, sigma=2.0, cover_factor=3.0,
                   lowpass=16, out_grid=14, w_var=0.3, gamma=0.5,
                   detail_p=2.0, detail_src="local_var", merge=False, contextual=0):
    """Patch a loaded LLaVA model to prune visual tokens with the given idea."""
    vt = model.model.vision_tower.vision_tower
    # reuse VisionZip's low-level CLIP patches (metric capture / r schedule);
    # contextual=0 since ideas never merge.
    apply_info(vt, dominant_num=k - 1, contextual_num=0)
    vt._info["idea_method"] = method
    vt._info["idea_k"] = k
    vt._info["idea_lambda"] = lam
    vt._info["idea_rho"] = rho
    vt._info["idea_sigma"] = sigma
    vt._info["idea_cover_factor"] = cover_factor
    vt._info["idea_lowpass"] = lowpass
    vt._info["idea_out_grid"] = out_grid
    vt._info["idea_gamma"] = gamma
    vt._info["idea_detail_p"] = detail_p
    vt._info["idea_detail_src"] = detail_src
    vt._info["idea_w_var"] = w_var
    vt._info["idea_merge"] = merge
    vt._info["idea_contextual"] = contextual

    from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention
    CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
    CLIPAttention.forward = CLIPAttention_forward

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    CLIPVisionTower.forward = CLIPVisionTower_Idea_forward

    from llava.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, "prepare_inputs_labels_for_multimodal"):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_visionzip
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
        LlavaMetaForCausalLM.encode_images_visionzip_multi = encode_images_visionzip_multi
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip
    return model
