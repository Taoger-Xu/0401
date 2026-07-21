# ------------------------------------------------------------------------
# Pre-LLM spatial-redundancy visual-token pruning selectors.
#
# Three hard-pruning strategies, all injected at the same point VisionZip
# uses (CLIP penultimate layer output, before mm_projector). Every strategy
# keeps the CLS token plus (K-1) *original* patch tokens (no merging, no
# reconstruction) so the LLM sees exactly K visual tokens, matching the
# VisionZip token budget for a controlled comparison.
#
#   idea1  spectral   : DCT low-frequency stability + spatial-cell medoid
#   idea2  local_var  : keep tokens that differ most from their 8 neighbours
#   idea3  cls_mmr    : CLS-attention importance with redundancy penalty (MMR)
#
# The scores are computed on the CLIP penultimate patch features
# (hidden_states[-2][:, 1:], 1024-d) -- the natural VisionZip injection point.
# ------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# idea2: Local Variation
# --------------------------------------------------------------------------
def score_local_variation(patch_feats, grid=24):
    """patch_feats: [B, N, D] on a grid x grid layout. Returns [B, N] score,
    higher = more locally distinctive = more worth keeping."""
    B, N, D = patch_feats.shape
    H = W = grid
    fn = F.normalize(patch_feats.float(), dim=-1).view(B, H, W, D)
    # replicate padding so border tokens still see 8 neighbours
    pad = F.pad(fn.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate")
    pad = pad.permute(0, 2, 3, 1)  # [B, H+2, W+2, D]
    sim_sum = torch.zeros(B, H, W, device=patch_feats.device, dtype=torch.float32)
    for di in range(3):
        for dj in range(3):
            if di == 1 and dj == 1:
                continue
            sim_sum += (fn * pad[:, di:di + H, dj:dj + W, :]).sum(-1)
    local_sim = sim_sum / 8.0
    return (1.0 - local_sim).view(B, N)


# --------------------------------------------------------------------------
# idea1: Spectral Relay Pruning (cell_medoid_lowfreq)
# --------------------------------------------------------------------------
_CELL_CACHE = {}      # (grid, budget) -> (LongTensor[N] cell id, FloatTensor[N, budget] one-hot)
_DCT_CACHE = {}       # grid -> orthonormal DCT-II matrix [grid, grid]


def _dct_matrix(grid, device, dtype=torch.float32):
    key = (grid, device)
    M = _DCT_CACHE.get(key)
    if M is None:
        n = torch.arange(grid, dtype=torch.float64).view(1, -1)
        k = torch.arange(grid, dtype=torch.float64).view(-1, 1)
        M = torch.cos(math.pi * (2 * n + 1) * k / (2 * grid))
        M[0, :] *= 1.0 / math.sqrt(grid)
        M[1:, :] *= math.sqrt(2.0 / grid)
        M = M.to(device=device, dtype=dtype)
        _DCT_CACHE[key] = M
    return M


def _fps_voronoi_cells(grid, budget, device):
    """Deterministic farthest-point sampling on the grid coords, then assign
    every patch to its nearest anchor -> `budget` Voronoi cells. Image-independent,
    so cached per (grid, budget)."""
    key = (grid, budget)
    cached = _CELL_CACHE.get(key)
    if cached is not None:
        cells, onehot = cached
        return cells.to(device), onehot.to(device)
    ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1).float()  # [N, 2]
    N = coords.shape[0]
    anchors = [0]  # deterministic start at top-left
    dist = torch.full((N,), float("inf"))
    for _ in range(1, budget):
        d = (coords - coords[anchors[-1]]).pow(2).sum(-1)
        dist = torch.minimum(dist, d)
        anchors.append(int(dist.argmax()))
    anchors_t = torch.tensor(anchors)
    # assign each patch to nearest anchor
    d_all = torch.cdist(coords, coords[anchors_t])  # [N, budget]
    cells = d_all.argmin(dim=1)  # [N] in [0, budget)
    onehot = torch.zeros(N, budget)
    onehot[torch.arange(N), cells] = 1.0
    _CELL_CACHE[key] = (cells, onehot)
    return cells.to(device), onehot.to(device)


def score_and_select_spectral(patch_feats, budget, grid=24, lowpass=16, w_freq=1.0):
    """idea1. Returns LongTensor[B, budget] of selected patch indices (one per
    spatial cell, chosen by medoid representativeness + low-frequency stability)."""
    B, N, D = patch_feats.shape
    device = patch_feats.device
    feats = patch_feats.float()

    # ---- low-frequency residual (used only for scoring, never as input) ----
    M = _dct_matrix(grid, device)
    g = feats.view(B, grid, grid, D)
    # 2D DCT: freq index k along H, l along W. coef[b,k,l,d]
    coef = torch.einsum("kh,bhwd->bkwd", M, g)      # H -> freq k
    coef = torch.einsum("lw,bkwd->bkld", M, coef)   # W -> freq l
    mask = torch.zeros(grid, grid, device=device)
    mask[:lowpass, :lowpass] = 1.0
    coef = coef * mask.view(1, grid, grid, 1)
    # 2D inverse DCT (orthonormal -> inverse is M^T, i.e. contract the freq index)
    rec = torch.einsum("lw,bkld->bkwd", M, coef)    # freq l -> W
    rec = torch.einsum("kh,bkwd->bhwd", M, rec)     # freq k -> H
    residual = (g - rec).reshape(B, N, D).norm(dim=-1)  # [B, N]

    cells, onehot = _fps_voronoi_cells(grid, budget, device)  # [N], [N, budget]
    # explicit .float(): under the caller's fp16 autocast F.normalize can emit
    # Half even from a float32 input, which then mismatches the float32 cached
    # one-hot in the einsum below. Force float32 deterministically.
    fn = F.normalize(feats, dim=-1).float()                    # [B, N, D]
    onehot = onehot.float()
    residual = residual.float()
    cells_b = cells.view(1, N).expand(B, N)                    # [B, N]

    # per-cell centroid direction -> medoid cosine per token
    cell_sum = torch.einsum("nc,bnd->bcd", onehot, fn)         # [B, budget, D]
    centroid = F.normalize(cell_sum, dim=-1)                   # [B, budget, D]
    cent_per_tok = centroid.gather(1, cells_b.unsqueeze(-1).expand(B, N, D))  # [B, N, D]
    medoid = (fn * cent_per_tok).sum(-1)                       # [B, N] cosine
    lowfreq = -residual                                       # [B, N], smaller residual = better

    medoid = _cell_minmax(medoid, cells_b, budget)
    lowfreq = _cell_minmax(lowfreq, cells_b, budget)
    score = medoid + w_freq * lowfreq                          # [B, N]

    # per-cell argmax (deterministic tie-break: smallest index)
    NEG = torch.finfo(score.dtype).min
    cmax = score.new_full((B, budget), NEG).scatter_reduce(
        1, cells_b, score, reduce="amax", include_self=True)   # [B, budget]
    cmax_per_tok = cmax.gather(1, cells_b)                     # [B, N]
    is_win = score >= cmax_per_tok                             # [B, N]
    idx_all = torch.arange(N, device=device).view(1, N).expand(B, N)
    cand = torch.where(is_win, idx_all, torch.full_like(idx_all, N))
    selected = cand.new_full((B, budget), N).scatter_reduce(
        1, cells_b, cand, reduce="amin", include_self=True)    # [B, budget]
    selected = selected.clamp(max=N - 1)                       # guard empty cells
    return selected


# --------------------------------------------------------------------------
# idea4: Anchor-Cover (dual-pool: coverage cells + spatially-gated MMR)
# --------------------------------------------------------------------------
_COORD_CACHE = {}     # grid -> FloatTensor[N, 2] patch grid coords
_GATE_CACHE = {}      # (grid, sigma) -> FloatTensor[N, N] spatial gaussian gate


def _grid_coords(grid, device):
    c = _COORD_CACHE.get(grid)
    if c is None:
        ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        c = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1).float()  # [N,2]
        _COORD_CACHE[grid] = c
    return c.to(device)


def _spatial_gate(grid, sigma, device):
    key = (grid, round(float(sigma), 4))
    g = _GATE_CACHE.get(key)
    if g is None:
        coords = _grid_coords(grid, "cpu")
        d2 = torch.cdist(coords, coords).pow(2)             # [N,N]
        g = torch.exp(-d2 / (2.0 * sigma * sigma))
        _GATE_CACHE[key] = g
    return g.to(device)


def select_anchor_cover(patch_feats, cls_attn, budget, grid=24, rho=0.5,
                        lam=0.5, sigma=2.0, lowpass=16, w_f=1.0, w_a=1.0,
                        cover_factor=3.0):
    """idea4. Coverage pool (salience-ranked spatial cells, attention-aware
    medoid) + salience pool (spatially-gated MMR). Returns LongTensor[B, budget].

    Coverage: partition into P = round(cover_factor*M) cells, keep the top-M
    cells by CLS-attention mass (skips empty background), one representative each.
    cover_factor=1 recovers the original "one token per uniform cell" idea4.

    Degenerate limits:
      rho->1, w_a=0, cover_factor=1 -> idea1 spectral
      rho->0, lam=0                 -> VisionZip dominant top-K (pure pruning)
      rho->0, sigma->inf            -> idea3 cls_mmr
    """
    B, N, D = patch_feats.shape
    device = patch_feats.device
    feats = patch_feats.float()
    fn = F.normalize(feats, dim=-1).float()                 # [B,N,D]
    imp = _minmax(cls_attn.float())                         # [B,N]
    M = int(math.ceil(rho * budget))
    M = max(0, min(M, budget))
    ar = torch.arange(B, device=device)

    chosen = torch.zeros(B, N, dtype=torch.bool, device=device)
    selected = torch.zeros(B, budget, dtype=torch.long, device=device)

    # ---------------- Phase A: coverage pool (top-M salient cells) ----------
    if M > 0:
        P = min(N, max(M, int(round(cover_factor * M))))     # partition granularity
        # low-frequency residual (scoring only)
        Mdct = _dct_matrix(grid, device)
        g = feats.view(B, grid, grid, D)
        coef = torch.einsum("kh,bhwd->bkwd", Mdct, g)
        coef = torch.einsum("lw,bkwd->bkld", Mdct, coef)
        fmask = torch.zeros(grid, grid, device=device)
        fmask[:lowpass, :lowpass] = 1.0
        coef = coef * fmask.view(1, grid, grid, 1)
        rec = torch.einsum("lw,bkld->bkwd", Mdct, coef)
        rec = torch.einsum("kh,bkwd->bhwd", Mdct, rec)
        residual = (g - rec).reshape(B, N, D).norm(dim=-1).float()  # [B,N]

        cells, onehot = _fps_voronoi_cells(grid, P, device)
        onehot = onehot.float()
        cells_b = cells.view(1, N).expand(B, N)
        cell_sum = torch.einsum("nc,bnd->bcd", onehot, fn)
        centroid = F.normalize(cell_sum, dim=-1)
        cent_per_tok = centroid.gather(1, cells_b.unsqueeze(-1).expand(B, N, D))
        medoid = _cell_minmax((fn * cent_per_tok).sum(-1), cells_b, P)
        lowfreq = _cell_minmax(-residual, cells_b, P)
        attn_c = _cell_minmax(imp, cells_b, P)
        score = medoid + w_f * lowfreq + w_a * attn_c        # [B,N]

        # representative token per cell (argmax score, tie-break smallest idx)
        NEG = torch.finfo(score.dtype).min
        cmax = score.new_full((B, P), NEG).scatter_reduce(1, cells_b, score, reduce="amax", include_self=True)
        is_win = score >= cmax.gather(1, cells_b)
        idx_all = torch.arange(N, device=device).view(1, N).expand(B, N)
        cand = torch.where(is_win, idx_all, torch.full_like(idx_all, N))
        cellrep = cand.new_full((B, P), N).scatter_reduce(1, cells_b, cand, reduce="amin", include_self=True)
        cellrep = cellrep.clamp(max=N - 1)                   # [B,P] rep token per cell
        # rank cells by attention mass, keep the M most salient cells
        csal = imp.new_full((B, P), NEG).scatter_reduce(1, cells_b, imp, reduce="amax", include_self=True)
        topM = csal.topk(M, dim=1).indices                   # [B,M]
        cellpick = cellrep.gather(1, topM)                   # [B,M]
        selected[:, :M] = cellpick
        chosen.scatter_(1, cellpick, True)

    # ---------------- Phase B: salience pool (spatially-gated MMR) ------
    Bn = budget - M
    if Bn > 0:
        gate = _spatial_gate(grid, sigma, device)            # [N,N]
        max_red = torch.zeros(B, N, device=device)
        if M > 0:
            # seed running redundancy from the coverage set
            for m in range(M):
                j = selected[:, m]
                red = (torch.einsum("bnd,bd->bn", fn, fn[ar, j]).clamp(min=0)
                       * gate[:, j].transpose(0, 1))
                max_red = torch.maximum(max_red, red)
        else:
            # no coverage pool: seed with the single most salient token (=idea3)
            first = imp.argmax(dim=1)
            selected[:, 0] = first
            chosen[ar, first] = True
            red = (torch.einsum("bnd,bd->bn", fn, fn[ar, first]).clamp(min=0)
                   * gate[:, first].transpose(0, 1))
            max_red = torch.maximum(max_red, red)
        start = M if M > 0 else 1
        for step in range(start, budget):
            mmr = (imp - lam * max_red).masked_fill(chosen, float("-inf"))
            pick = mmr.argmax(dim=1)
            selected[:, step] = pick
            chosen[ar, pick] = True
            red = (torch.einsum("bnd,bd->bn", fn, fn[ar, pick]).clamp(min=0)
                   * gate[:, pick].transpose(0, 1))
            max_red = torch.maximum(max_red, red)

    selected, _ = selected.sort(dim=1)
    return selected


def _cell_minmax(x, cells_b, budget):
    """min-max normalise x [B,N] within each cell id in cells_b [B,N]."""
    B, N = x.shape
    POS = torch.finfo(x.dtype).max
    NEG = torch.finfo(x.dtype).min
    cmin = x.new_full((B, budget), POS).scatter_reduce(1, cells_b, x, reduce="amin", include_self=True)
    cmax = x.new_full((B, budget), NEG).scatter_reduce(1, cells_b, x, reduce="amax", include_self=True)
    lo = cmin.gather(1, cells_b)
    hi = cmax.gather(1, cells_b)
    return (x - lo) / (hi - lo + 1e-6)


def _minmax(x):
    lo = x.min(dim=1, keepdim=True).values
    hi = x.max(dim=1, keepdim=True).values
    return (x - lo) / (hi - lo + 1e-6)


# --------------------------------------------------------------------------
# idea5 diagnostics: DCT low-frequency reconstruction / downsampling.
# Not pruning -- output tokens are linear combinations of input tokens.
# Used to test whether TextVQA information survives low-pass filtering
# (probe, token count unchanged) and whether frequency-domain compression
# beats hard pruning at the same token budget (out_grid variant).
# --------------------------------------------------------------------------
def lowfreq_reconstruct(patch_feats, grid=24, lowpass=16, out_grid=None):
    """Replace patch features by their 2D-DCT low-pass reconstruction.

    out_grid=None: reconstruct on the original grid -> [B, N, D]. Token count
        unchanged; probes how much task information lives above the cutoff.
    out_grid=k: keep the k x k low-frequency coefficient block and inverse-
        transform on a k x k grid (orthonormal DCT resize) -> [B, k*k, D].
        Real compression to k*k tokens.
    """
    B, N, D = patch_feats.shape
    device = patch_feats.device
    g = patch_feats.float().view(B, grid, grid, D)
    M = _dct_matrix(grid, device)
    coef = torch.einsum("kh,bhwd->bkwd", M, g)
    coef = torch.einsum("lw,bkwd->bkld", M, coef)
    if out_grid is None:
        mask = torch.zeros(grid, grid, device=device)
        mask[:lowpass, :lowpass] = 1.0
        coef = coef * mask.view(1, grid, grid, 1)
        rec = torch.einsum("lw,bkld->bkwd", M, coef)
        rec = torch.einsum("kh,bkwd->bhwd", M, rec)
        return rec.reshape(B, N, D)
    k = int(out_grid)
    Mk = _dct_matrix(k, device)
    # orthonormal DCT resize: sqrt(k/grid) amplitude rescale per dimension
    ck = coef[:, :k, :k, :] * (k / grid)
    rec = torch.einsum("lw,bkld->bkwd", Mk, ck)
    rec = torch.einsum("kh,bkwd->bhwd", Mk, rec)
    return rec.reshape(B, k * k, D)


# --------------------------------------------------------------------------
# idea3: CLS-attention importance with redundancy penalty (greedy MMR)
# --------------------------------------------------------------------------
def select_cls_mmr(patch_feats, cls_attn, budget, lam=0.5):
    """idea3. Greedy Maximal-Marginal-Relevance selection.

    VisionZip keeps the top-`budget` patches by CLS attention, which tends to
    cluster on one salient region (spatially redundant). Here we still seed on
    CLS-attention importance but subtract a redundancy term so each newly kept
    token is important *and* dissimilar to the already-kept set:

        score(i | S) = imp_i  -  lam * max_{j in S} cos(f_i, f_j)

    patch_feats: [B, N, D]   cls_attn: [B, N] (per-patch CLS attention sum)
    Returns LongTensor[B, budget] of selected patch indices.
    """
    B, N, D = patch_feats.shape
    device = patch_feats.device
    fn = F.normalize(patch_feats.float(), dim=-1)
    imp = _minmax(cls_attn.float())  # [B, N] in [0,1]

    selected = torch.zeros(B, budget, dtype=torch.long, device=device)
    chosen_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    max_sim = torch.zeros(B, N, device=device)  # running max cos-sim to chosen set

    # seed: highest-importance token
    first = imp.argmax(dim=1)  # [B]
    ar = torch.arange(B, device=device)
    selected[:, 0] = first
    chosen_mask[ar, first] = True
    # update running max sim against the seed
    seed_feat = fn[ar, first]                       # [B, D]
    max_sim = torch.einsum("bnd,bd->bn", fn, seed_feat).clamp(min=0)
    max_sim[ar, first] = 2.0                          # exclude chosen from re-pick

    for step in range(1, budget):
        mmr = imp - lam * max_sim
        mmr = mmr.masked_fill(chosen_mask, float("-inf"))
        pick = mmr.argmax(dim=1)                      # [B]
        selected[:, step] = pick
        chosen_mask[ar, pick] = True
        pick_feat = fn[ar, pick]
        sim_new = torch.einsum("bnd,bd->bn", fn, pick_feat).clamp(min=0)
        max_sim = torch.maximum(max_sim, sim_new)
        max_sim[ar, pick] = 2.0
    return selected


# --------------------------------------------------------------------------
# idea5: detail-gated CLS-MMR (docs/idea5/idea5.md)
# --------------------------------------------------------------------------
def select_detail_mmr(patch_feats, cls_attn, budget, lam=0.5, gamma=1.0, p=2.0,
                      detail_src="local_var", grid=24, lowpass=16):
    """idea5. CLS-MMR whose redundancy penalty is gated by a text/detail score:

        score(i | S) = imp_i - lam * (1 - gamma * d_i) * max_{j in S} cos+(f_i, f_j)

    d_i in [0,1] is a detail/text likelihood (high local variation or high DCT
    high-frequency residual). Similarity inside detail regions (text blocks) is
    encoder pseudo-similarity, not information redundancy (docs/idea5 §9.0), so
    high-d tokens are exempted from the penalty and can be kept densely. d never
    adds to importance: score <= imp_i always, so low-attention texture cannot
    enter on detail alone. gamma=0 reduces exactly to select_cls_mmr.
    """
    B, N, D = patch_feats.shape
    device = patch_feats.device
    fn = F.normalize(patch_feats.float(), dim=-1)
    imp = _minmax(cls_attn.float())

    if detail_src == "dct":
        M = _dct_matrix(grid, device)
        g = patch_feats.float().view(B, grid, grid, D)
        coef = torch.einsum("kh,bhwd->bkwd", M, g)
        coef = torch.einsum("lw,bkwd->bkld", M, coef)
        mask = torch.zeros(grid, grid, device=device)
        mask[:lowpass, :lowpass] = 1.0
        coef = coef * mask.view(1, grid, grid, 1)
        rec = torch.einsum("lw,bkld->bkwd", M, coef)
        rec = torch.einsum("kh,bkwd->bhwd", M, rec)
        det = (g - rec).reshape(B, N, D).norm(dim=-1)      # high-freq residual
    else:
        det = score_local_variation(patch_feats.float(), grid=grid)
    d = _minmax(det.float()).pow(p)
    gate = (1.0 - gamma * d).clamp(min=0.0)                # [B, N]

    selected = torch.zeros(B, budget, dtype=torch.long, device=device)
    chosen = torch.zeros(B, N, dtype=torch.bool, device=device)
    ar = torch.arange(B, device=device)
    first = imp.argmax(dim=1)
    selected[:, 0] = first
    chosen[ar, first] = True
    max_sim = torch.einsum("bnd,bd->bn", fn, fn[ar, first]).clamp(min=0)
    for step in range(1, budget):
        mmr = (imp - lam * gate * max_sim).masked_fill(chosen, float("-inf"))
        pick = mmr.argmax(dim=1)
        selected[:, step] = pick
        chosen[ar, pick] = True
        sim_new = torch.einsum("bnd,bd->bn", fn, fn[ar, pick]).clamp(min=0)
        max_sim = torch.maximum(max_sim, sim_new)
    return selected


# --------------------------------------------------------------------------
# idea6: semantic local variation (idea2 + CLS semantics, two injection points)
# --------------------------------------------------------------------------
def select_sem_var_mmr(patch_feats, cls_attn, budget, grid=24, lam=0.5,
                       w_v=0.3, gamma=0.5):
    """idea6. Fixes idea2's two diagnosed failures by injecting CLS-attention
    semantics at two distinct points of the idea3 MMR skeleton:

      1. no saliency prior -> additive score fusion (additive, not
         multiplicative, so high-attention/low-variation text interiors and
         object fills are not zeroed out):
             imp_i = attn_i + w_v * var_i
      2. clustered text patches are mutually similar, so plain MMR rejects
         them as redundant -> attention shrinks the redundancy penalty:
             mmr_i = imp_i - lam * (1 - gamma * attn_i) * max_{j in S} cos(f_i, f_j)
         Salient tokens (text blocks) may coexist despite high similarity;
         background dedup stays at full strength.

    Degenerate limits: w_v=0, gamma=0 -> idea3 cls_mmr exactly;
                       lam=0, w_v large -> idea2 local_var top-K.

    patch_feats: [B, N, D]   cls_attn: [B, N]
    Returns LongTensor[B, budget] of selected patch indices.
    """
    B, N, D = patch_feats.shape
    device = patch_feats.device
    fn = F.normalize(patch_feats.float(), dim=-1)
    attn = _minmax(cls_attn.float())                            # [B, N]
    var = _minmax(score_local_variation(patch_feats, grid=grid))  # [B, N]
    imp = attn + w_v * var
    protect = lam * (1.0 - gamma * attn)                        # [B, N]

    selected = torch.zeros(B, budget, dtype=torch.long, device=device)
    chosen = torch.zeros(B, N, dtype=torch.bool, device=device)
    ar = torch.arange(B, device=device)
    first = imp.argmax(dim=1)
    selected[:, 0] = first
    chosen[ar, first] = True
    max_sim = torch.einsum("bnd,bd->bn", fn, fn[ar, first]).clamp(min=0)

    for step in range(1, budget):
        mmr = (imp - protect * max_sim).masked_fill(chosen, float("-inf"))
        pick = mmr.argmax(dim=1)
        selected[:, step] = pick
        chosen[ar, pick] = True
        sim_new = torch.einsum("bnd,bd->bn", fn, fn[ar, pick]).clamp(min=0)
        max_sim = torch.maximum(max_sim, sim_new)
    return selected
