#!/usr/bin/env python3
"""Training utilities: losses, alignment model, 8-bit loaders, checkpoints."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trisearch_dataset import (
    ImageCaptionDataset,
    Stage1Collator,
    load_verification_samples,
)

from image_augment import (
    DEFAULT_GRAYSCALE_PROB,
    DEFAULT_IMAGE_FILL_MODE,
    DEFAULT_IMAGE_HFLIP_PROB,
    DEFAULT_IMAGE_MAX_ROTATE_DEG,
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_SCALE_MAX,
    DEFAULT_IMAGE_SCALE_MIN,
    DEFAULT_IMAGE_SHIFT_MAX,
    DEFAULT_IMAGE_STD,
    DEFAULT_PHOTO_BRIGHTNESS,
    DEFAULT_PHOTO_CONTRAST,
    DEFAULT_PHOTO_HUE,
    DEFAULT_PHOTO_SATURATION,
    DEFAULT_PHOTOMETRIC_ENABLED,
    DEFAULT_SPATIAL_BRIGHTNESS,
    DEFAULT_SPATIAL_COLOR,
    DEFAULT_SPATIAL_NOISE_GRID,
    apply_train_image_augmentations,
    random_shift_pixel_values,
    train_image_geometric_augment,
    train_image_photometric_augment,
)

from .inference import (
    EMBED_DIM,
    MAX_TRAINING_PHASE,
    QWEN_DIR,
    QWEN_TOKENIZER_ID,
    SIGLIP_DIR,
    SIGLIP_PROCESSOR_ID,
    checkpoint_is_quantized,
    load_qwen_backbone,
    load_siglip_backbone,
    matryoshka_normalize,
)

DEFAULT_SEED_VISION_DIR = SIGLIP_DIR
DEFAULT_SEED_TEXT_DIR = QWEN_DIR
TRAINED_ROOT = "models/trained"
DEFAULT_TRAINED_DIR = "models/trained/stage1"
LEGACY_CHECKPOINT_DIR = "checkpoints/stage1"
VISION_COMPONENT = "vision_model"
TEXT_COMPONENT = "text_model"
PROJECTION_FILE = "projection_heads.pt"
TRAINING_STATE_FILE = "training_state.pt"
CONFIG_FILE = "stage1_config.json"
# Stage-1 captions are short; 64 avoids mean-MaxSim dilution from long generic tails.
DEFAULT_MAX_INPUT_TOKENS = 64
# Prefix dims only — full embed dim is trained by the main contrastive terms.
# Including 1024 here double-counted full-dim CE and starved small prefixes.
DEFAULT_MATRYOSHKA_DIMS = (64, 128, 256, 512)
# Soft MaxSim: τ_s * logsumexp(sim / τ_s). Smaller τ_s → closer to hard max.
DEFAULT_SOFT_MAXSIM_TEMPERATURE = 0.03
# Caption token-Jaccard above this → treat as non-negative (not a false neg).
DEFAULT_MULTI_POSITIVE_JACCARD = 0.5
# Keep top fraction of SigLIP patches by pre-norm L2 (drop background).
DEFAULT_VISION_PATCH_KEEP_RATIO = 0.75
# Mild spatial dropout; high values (0.3–0.4) weakened content signal in collapse runs.
DEFAULT_VISION_PATCH_DROP_PROB = 0.15
# Merge vision patches into this many similarity centroids before MaxSim (0 = off).
# Cuts redundant background tokens that make MaxSim match every image equally.
DEFAULT_VISION_MERGE_TOKENS = 8
# Softmax temperature for soft assignment into merge centroids.
DEFAULT_VISION_MERGE_ASSIGN_TEMPERATURE = 0.05
# Subtract detached batch mean of unit tokens before InfoNCE (kills domain cone).
DEFAULT_SCORE_CENTER = True
# Mean only the top-k query-token MaxSims (0 = mean all). Drops stopword-like tokens.
DEFAULT_QUERY_MAXSIM_TOPK = 8
# Heatmap sparsity stuck ~1.0 and added noise; off unless debugging demos.
DEFAULT_HEATMAP_SPARSITY_WEIGHT = 0.0
DEFAULT_HEATMAP_SPARSITY_TEMPERATURE = 0.07
# Square normalized-entropy badness (same idea as geo_square): soft when already
# sparse, quadratic shove on diffuse MaxSim heatmaps. Entropy is already in [0,1].
DEFAULT_HEATMAP_SPARSITY_SQUARE = True
# Top-k hardest bank docs kept per query as InfoNCE negatives (0 = use full bank).
DEFAULT_HARD_BANK_NEGATIVES = 32
# Exclude bank cols with score >= pos - margin (logit space) as false negatives.
DEFAULT_BANK_FN_MARGIN = 0.0
# Extra random bank negatives after hard-k (diversity; 0 = hard-only).
DEFAULT_BANK_RANDOM_K = 8
# Flush FIFO bank periodically so a collapsed stretch cannot poison all negatives.
DEFAULT_BANK_CLEAR_STEPS = 250
# Direct margin on score_gap = mean(pos) - mean(finite negs) after /temp + hard-k.
# Hinge ReLU(margin - gap): 0 when gap ≥ margin; pushes out of negative-gap regime.
DEFAULT_GAP_LOSS_WEIGHT = 1.0
DEFAULT_GAP_MARGIN = 0.0

# Embedding geometry regularizer (anti-cone / isotropy).
# InfoNCE alone does *not* absolute-repel embeddings — only ranks pos vs negs.
DEFAULT_EMBEDDING_GEO_WEIGHT = 0.4
DEFAULT_GEO_CENTER_WEIGHT = 2.0
DEFAULT_GEO_VAR_WEIGHT = 4.0
DEFAULT_GEO_VEC_MEAN_WEIGHT = 0.25
DEFAULT_GEO_MAG_FLOOR = 0.05
DEFAULT_GEO_MAG_FLOOR_WEIGHT = 0.1
# Per-dim std target as fraction of ideal isotropic 1/sqrt(D).
DEFAULT_GEO_VAR_RATIO = 0.5
# Soft penalty when |coord| exceeds this * 1/sqrt(D) (stops single-dim domination).
DEFAULT_GEO_MAX_ABS_RATIO = 4.0
DEFAULT_GEO_MAX_ABS_WEIGHT = 0.1
# Wang & Isola uniformity: log E exp(-t ||xi-xj||²) on the unit sphere.
DEFAULT_GEO_UNIFORMITY_WEIGHT = 1.0
DEFAULT_GEO_UNIFORMITY_T = 2.0
DEFAULT_GEO_UNIFORMITY_MAX_SAMPLES = 256
# Prefer sample-level (renormed sequence means) over token-row dilution.
DEFAULT_GEO_TOKEN_WEIGHT = 0.3
DEFAULT_GEO_POOL_WEIGHT = 0.7
# Also regularize a Matryoshka prefix (re-normalized).
DEFAULT_GEO_PREFIX_DIM = 256
DEFAULT_GEO_PREFIX_WEIGHT = 0.5
DEFAULT_GEO_EMA_MOMENTUM = 0.99
# Geo during freeze is OK for monitoring + weak proj shaping; set True to defer.
DEFAULT_GEO_AFTER_UNFREEZE = False
# Square non-negative geo badness: soft when near-isotropic, hard shove on cones.
# Guarantees geo term ≥ 0 so total loss cannot crash from negative geo.
DEFAULT_GEO_SQUARE = True

# Optimizer / schedule defaults (wired by train_stage1 CLI).
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM_STEPS = 8  # effective batch 32
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_PROJECTION_LEARNING_RATE = 5e-5
DEFAULT_MAX_STEPS = 5000
DEFAULT_MATRYOSHKA_WEIGHT = 0.25
DEFAULT_TEMPERATURE = 0.07
DEFAULT_FREEZE_BACKBONE_RATIO = 0.05
DEFAULT_SAVE_STEPS = 200
DEFAULT_LOGGING_STEPS = 10


def pad_token_sequences(
    tokens: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of ``(n_i, D)`` tensors to ``(B, max_n, D)`` + bool mask."""
    if not tokens:
        raise ValueError("pad_token_sequences requires a non-empty token list.")
    device = tokens[0].device
    dtype = tokens[0].dtype
    dim = tokens[0].shape[-1]
    lengths = [0 if t.numel() == 0 else int(t.shape[0]) for t in tokens]
    max_n = max(lengths) if lengths else 0
    batch = len(tokens)
    if max_n == 0:
        padded = tokens[0].new_zeros((batch, 1, dim))
        mask = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        return padded, mask
    padded = tokens[0].new_zeros((batch, max_n, dim))
    mask = torch.zeros(batch, max_n, dtype=torch.bool, device=device)
    for i, t in enumerate(tokens):
        if t.numel() == 0:
            continue
        if t.ndim == 1:
            t = t.unsqueeze(0)
        n = t.shape[0]
        padded[i, :n] = t.to(device=device, dtype=dtype)
        mask[i, :n] = True
    return padded, mask


def soft_or_hard_maxsim(
    sim: torch.Tensor,
    *,
    soft_temperature: float | None = None,
    dim: int = -1,
) -> torch.Tensor:
    """Max (or soft-Max) over ``dim``.

    Soft MaxSim: ``τ_s * logsumexp(sim / τ_s)``. As ``τ_s → 0`` this recovers
    hard max while remaining differentiable w.r.t. all document tokens.
    """
    if soft_temperature is None or soft_temperature <= 0.0:
        return sim.max(dim=dim).values
    tau = float(soft_temperature)
    return tau * torch.logsumexp(sim / tau, dim=dim)


def _mean_topk_query_maxsim(
    per_q: torch.Tensor,
    q_mask: torch.Tensor,
    query_topk: int,
) -> torch.Tensor:
    """Mean of top-``query_topk`` per-query MaxSims; ignore padded query slots.

    ``per_q``: ``(Bq, Bd, Tq)``, ``q_mask``: ``(Bq, Tq)``.
    """
    if query_topk is None or int(query_topk) <= 0:
        q_mask_f = q_mask.to(dtype=per_q.dtype).unsqueeze(1)  # (Bq, 1, Tq)
        per_q = per_q * q_mask_f
        denom = q_mask_f.sum(dim=-1).clamp_min(1.0)
        return per_q.sum(dim=-1) / denom

    k = int(query_topk)
    # Mask padded query positions so they never enter the top-k.
    neg = torch.finfo(per_q.dtype).min
    if not torch.isfinite(torch.tensor(neg, dtype=per_q.dtype)):
        neg = -1e4
    else:
        neg = max(float(neg), -1e4)
    masked = per_q.masked_fill(~q_mask.unsqueeze(1), neg)
    # k cannot exceed Tq; also cannot exceed valid count per row.
    tq = masked.shape[-1]
    k_eff = min(k, tq)
    topv = masked.topk(k_eff, dim=-1).values  # (Bq, Bd, k_eff)
    # Valid top entries are finite and not the pad fill (use q_mask counts).
    n_valid = q_mask.to(dtype=per_q.dtype).sum(dim=-1).clamp_min(1.0)  # (Bq,)
    k_row = n_valid.clamp_max(float(k_eff)).unsqueeze(1)  # (Bq, 1)
    # Zero-out slots beyond n_valid for short sequences (topk may pull pad).
    # Count how many topv are real: topv > neg/2 roughly.
    real = topv > (0.5 * neg)
    # Prefer exact: rank among valid only — sum real values / min(k, n_valid)
    topv = topv.masked_fill(~real, 0.0)
    denom = real.to(dtype=per_q.dtype).sum(dim=-1).clamp_min(1.0)
    # Also clamp denom by k_row for safety
    denom = torch.minimum(denom, k_row.expand_as(denom))
    return topv.sum(dim=-1) / denom


def farthest_point_sample_indices(
    unit_tokens: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Greedy FPS indices on unit vectors ``(N, D)`` → ``(k,)`` long."""
    n = unit_tokens.shape[0]
    k = min(max(int(k), 1), n)
    # Start from max L2 of *pre-unit* proxy: use first dim energy of unit (stable).
    # Prefer token farthest from 0 in original if we only have units: max ||x|| is 1;
    # start from index 0 after sorting by max pairwise diversity seed = argmax variance.
    # Use token with largest coordinate span as seed.
    seed = int(unit_tokens.detach().float().abs().max(dim=-1).values.argmax().item())
    chosen: list[int] = [seed]
    # sims[i,j] = cos
    sims = unit_tokens @ unit_tokens.T
    for _ in range(k - 1):
        # For each point, similarity to nearest chosen; pick the smallest.
        max_sim = sims[:, chosen].max(dim=1).values.clone()
        for c in chosen:
            max_sim[c] = 2.0  # exclude
        chosen.append(int(max_sim.argmin().item()))
    return torch.tensor(chosen, device=unit_tokens.device, dtype=torch.long)


def merge_tokens_by_similarity(
    tokens: torch.Tensor,
    k: int = DEFAULT_VISION_MERGE_TOKENS,
    *,
    assign_temperature: float = DEFAULT_VISION_MERGE_ASSIGN_TEMPERATURE,
) -> torch.Tensor:
    """Reduce ``(N, D)`` tokens to ``(k, D)`` similarity centroids.

    Farthest-point seeds (detached) + soft assignment so gradients flow into
    all patches. Merges redundant background patches that otherwise dominate
    MaxSim against every image equally.
    """
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.numel() == 0:
        return tokens
    n = tokens.shape[0]
    k = min(max(int(k), 1), n)
    if k >= n:
        return tokens

    raw = tokens.float()
    x = F.normalize(raw, dim=-1)
    with torch.no_grad():
        idx = farthest_point_sample_indices(x, k)
        anchors = x.index_select(0, idx).clone()  # (k, D)
    # Soft assignment over clusters for each token.
    tau = max(float(assign_temperature), 1e-4)
    logits = (x @ anchors.T) / tau  # (N, k)
    weights = torch.softmax(logits, dim=1)
    # Pool in *raw* space so bank/Matryoshka still see pre-norm magnitudes.
    centroids = weights.T @ raw  # (k, D)
    return centroids.to(dtype=tokens.dtype)


def mean_unit_token_center(
    *token_lists: list[torch.Tensor],
) -> torch.Tensor | None:
    """Detached mean of all unit tokens across the given lists (domain cone)."""
    parts: list[torch.Tensor] = []
    for tokens in token_lists:
        for t in tokens:
            if t is None or t.numel() == 0:
                continue
            x = t.detach()
            if x.ndim == 1:
                x = x.unsqueeze(0)
            parts.append(x.float().reshape(-1, x.shape[-1]))
    if not parts:
        return None
    return torch.cat(parts, dim=0).mean(dim=0)


def apply_score_center(
    tokens: list[torch.Tensor],
    center: torch.Tensor,
) -> list[torch.Tensor]:
    """Spherical center: ``normalize(v - μ)`` per token row (μ detached)."""
    if center is None:
        return tokens
    c = center.detach().float()
    out: list[torch.Tensor] = []
    for t in tokens:
        if t is None or t.numel() == 0:
            out.append(t)
            continue
        x = t.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
            single = True
        else:
            single = False
        y = F.normalize(x - c.to(device=x.device), dim=-1).to(dtype=t.dtype)
        out.append(y.squeeze(0) if single else y)
    return out


def differentiable_late_interaction_score(
    query: torch.Tensor,
    doc: torch.Tensor,
    *,
    soft_maxsim_temperature: float | None = None,
    query_topk: int = 0,
) -> torch.Tensor:
    """Mean-MaxSim: mean over query tokens of max cosine to any doc token.

    Using the mean (not sum) keeps logits O(1) for InfoNCE with temperature
    ~0.07 even when captions are long, avoiding softmax saturation and
    pathological gradient norms with a large memory bank.

    When ``soft_maxsim_temperature`` is set (>0), hard max is replaced by
    soft MaxSim (τ logsumexp).

    ``query_topk`` > 0 averages only the top-k query-token MaxSims (drops
    stopword-like tokens that match every document's background).
    """
    if query.numel() == 0 or doc.numel() == 0:
        return query.new_zeros(())
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if doc.ndim == 1:
        doc = doc.unsqueeze(0)
    sim = query @ doc.T
    per_q = soft_or_hard_maxsim(
        sim, soft_temperature=soft_maxsim_temperature, dim=1
    )
    k = int(query_topk) if query_topk else 0
    if k > 0 and k < per_q.numel():
        per_q = per_q.topk(k).values
    return per_q.mean()


def build_late_interaction_matrix(
    query_tokens: list[torch.Tensor],
    doc_tokens: list[torch.Tensor],
    *,
    soft_maxsim_temperature: float | None = None,
    query_topk: int = 0,
) -> torch.Tensor:
    """Pairwise mean-MaxSim matrix of shape ``(len(queries), len(docs))``.

    Vectorized via padded tensors + einsum (same math as the per-pair loop).
    ``query_topk`` > 0 → mean only the strongest query-token MaxSims.
    """
    if not query_tokens or not doc_tokens:
        raise ValueError(
            f"Late-interaction matrix needs non-empty query and doc lists "
            f"(got {len(query_tokens)} queries, {len(doc_tokens)} docs)."
        )
    q_pad, q_mask = pad_token_sequences(query_tokens)  # (Bq, Tq, D), (Bq, Tq)
    d_pad, d_mask = pad_token_sequences(doc_tokens)  # (Bd, Td, D), (Bd, Td)

    # sim[b_q, b_d, t_q, t_d] = <q, d>
    # q_pad: (Bq, Tq, D), d_pad: (Bd, Td, D)
    sim = torch.einsum("qtd,usd->quts", q_pad, d_pad)

    # Invalidate padded doc positions before max / soft-max.
    neg_large = torch.finfo(sim.dtype).min if sim.dtype.is_floating_point else -1e9
    # Prefer a large negative that works with fp16/bf16 logsumexp.
    if not torch.isfinite(torch.tensor(neg_large, dtype=sim.dtype)):
        neg_large = -1e4
    else:
        # Clamp for half precision stability in logsumexp.
        neg_large = max(float(neg_large), -1e4)
    sim = sim.masked_fill(~d_mask.unsqueeze(0).unsqueeze(2), neg_large)

    per_q = soft_or_hard_maxsim(
        sim, soft_temperature=soft_maxsim_temperature, dim=-1
    )  # (Bq, Bd, Tq)

    return _mean_topk_query_maxsim(per_q, q_mask, int(query_topk or 0))


def caption_token_jaccard(a: str, b: str) -> float:
    """Token-bag Jaccard similarity in ``[0, 1]``."""
    ta = {t for t in str(a).lower().split() if t}
    tb = {t for t in str(b).lower().split() if t}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def build_multi_positive_mask(
    captions: list[str] | None,
    batch_size: int,
    *,
    jaccard_threshold: float = DEFAULT_MULTI_POSITIVE_JACCARD,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor | None:
    """Build ``(B, B)`` mask of in-batch non-negatives (incl. diagonal).

    Entry ``(i, j)`` is True when caption-i and caption-j are near-duplicates
    by token Jaccard (or ``i == j``). These pairs are *excluded from the
    negative set* in InfoNCE (false-negative softening) — they are not forced
    as extra CE positives.
    """
    if captions is None or jaccard_threshold is None or jaccard_threshold <= 0.0:
        return None
    if len(captions) != batch_size:
        raise ValueError(
            f"captions length {len(captions)} != batch_size {batch_size}"
        )
    if batch_size == 0:
        return None
    device = device or torch.device("cpu")
    mask = torch.eye(batch_size, dtype=dtype, device=device)
    thr = float(jaccard_threshold)
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if caption_token_jaccard(captions[i], captions[j]) >= thr:
                mask[i, j] = True
                mask[j, i] = True
    return mask


def masked_cross_entropy(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    non_negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """CE with optional non-negative masking (false-negatives → -inf logits).

    ``non_negative_mask`` is ``(B, n_docs)`` bool. True means "not a negative":
    off-diagonal non-negatives are filled with ``-inf`` so CE does not push
    them away. The labeled positive column is never masked out.
    """
    if non_negative_mask is None:
        return F.cross_entropy(scores, labels)
    if non_negative_mask.shape != scores.shape:
        raise ValueError(
            f"non_negative_mask shape {tuple(non_negative_mask.shape)} != "
            f"scores shape {tuple(scores.shape)}"
        )
    masked = scores.clone()
    # Exclude true positives from the mask-fill so labels stay finite.
    eye = torch.zeros_like(non_negative_mask)
    eye.scatter_(1, labels.view(-1, 1), True)
    exclude = non_negative_mask & ~eye
    masked = masked.masked_fill(exclude, float("-inf"))
    return F.cross_entropy(masked, labels)


def multi_positive_cross_entropy(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    non_negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-positive InfoNCE: ``log Z - logsumexp(pos)`` per row.

    ``scores``: ``(B, N)`` logits. ``positive_mask``: ``(B, N)`` bool with at
    least one True per row. Optional ``non_negative_mask`` softens other
    near-duplicates (sets them to -inf except true positives).
    """
    if positive_mask.shape != scores.shape:
        raise ValueError(
            f"positive_mask shape {tuple(positive_mask.shape)} != "
            f"scores shape {tuple(scores.shape)}"
        )
    if not bool(positive_mask.any()):
        raise ValueError("positive_mask has no True entries")
    logits = scores
    if non_negative_mask is not None:
        if non_negative_mask.shape != scores.shape:
            raise ValueError(
                f"non_negative_mask shape {tuple(non_negative_mask.shape)} != "
                f"scores shape {tuple(scores.shape)}"
            )
        # Keep positives finite; soft-exclude other non-negatives from the denom.
        exclude = non_negative_mask & ~positive_mask
        logits = scores.masked_fill(exclude, float("-inf"))
    log_z = torch.logsumexp(logits, dim=1)
    pos_logits = logits.masked_fill(~positive_mask, float("-inf"))
    log_pos = torch.logsumexp(pos_logits, dim=1)
    return (log_z - log_pos).mean()


def text_image_positive_mask(
    text_image_ids: torch.Tensor,
    n_images: int,
    n_texts: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """``(n_images, n_texts)`` bool: True when text ``t`` is a positive for image ``i``."""
    if text_image_ids.numel() != n_texts:
        raise ValueError(
            f"text_image_ids length {text_image_ids.numel()} != n_texts {n_texts}"
        )
    dev = device or text_image_ids.device
    ids = text_image_ids.to(device=dev, dtype=torch.long)
    img = torch.arange(n_images, device=dev).unsqueeze(1)  # (B, 1)
    return ids.unsqueeze(0) == img  # (B, T)


def keep_top_patches_by_l2(
    tokens: torch.Tensor,
    keep_ratio: float = DEFAULT_VISION_PATCH_KEEP_RATIO,
) -> torch.Tensor:
    """Keep top-``keep_ratio`` patches by pre-norm L2 (drop background).

    ``tokens`` is ``(P, D)`` *unnormalized* projected patch features. High L2
    magnitude tends to mark contentful patches; low-L2 patches are often
    near-uniform background. Ratio ``>= 1`` keeps all patches. Always keeps
    at least one patch.
    """
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.numel() == 0:
        return tokens
    ratio = float(keep_ratio)
    if ratio >= 1.0 or tokens.shape[0] <= 1:
        return tokens
    ratio = max(0.0, ratio)
    k = max(1, int(round(tokens.shape[0] * ratio)))
    k = min(k, tokens.shape[0])
    if k >= tokens.shape[0]:
        return tokens
    norms = tokens.detach().float().norm(dim=-1)
    # Stable order: topk then sort indices ascending for deterministic layout.
    idx = norms.topk(k, largest=True).indices
    idx, _ = idx.sort()
    return tokens[idx]


def random_drop_patches(
    tokens: torch.Tensor,
    drop_prob: float = DEFAULT_VISION_PATCH_DROP_PROB,
    *,
    training: bool = True,
) -> torch.Tensor:
    """Randomly drop a fraction of patches (train-time spatial dropout).

    Keeps ``round(n * (1 - drop_prob))`` patches (at least 1), in sorted index
    order so layout stays raster-stable for any remaining spatial logic.
    """
    if not training or tokens.numel() == 0:
        return tokens
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    prob = float(drop_prob)
    if prob <= 0.0 or tokens.shape[0] <= 1:
        return tokens
    prob = min(prob, 1.0 - 1e-6)
    n = int(tokens.shape[0])
    k = max(1, int(round(n * (1.0 - prob))))
    k = min(k, n)
    if k >= n:
        return tokens
    idx = torch.randperm(n, device=tokens.device)[:k]
    idx, _ = idx.sort()
    return tokens[idx]


# Image geometric / photometric / shift: see project-root image_augment.py
# (re-exported above for train_stage1 and tests).


def heatmap_sparsity_loss(
    query_tokens: list[torch.Tensor],
    image_tokens: list[torch.Tensor],
    *,
    temperature: float = DEFAULT_HEATMAP_SPARSITY_TEMPERATURE,
) -> torch.Tensor:
    """Punish diffuse positive-pair heatmaps; force selection of few patches.

    For each pair ``(q_i, d_i)`` with L2-normalized tokens:
      ``aff_p = max_q cos(q, p)``  (same patch affinity as demo heatmaps)
      ``p = softmax(aff / τ)`` over patches
      loss_i = H(p) / log(n_p)   (1 = uniform/noisy, 0 = one-hot peak)

    Gradients flow into both query and image embeddings (select blocks on the
    image side and sharper query tokens). Returns 0 if lists are empty/mismatched.
    """
    if not query_tokens or not image_tokens:
        raise ValueError("heatmap_sparsity_loss needs non-empty token lists")
    if len(query_tokens) != len(image_tokens):
        raise ValueError(
            f"query/image batch mismatch: {len(query_tokens)} vs {len(image_tokens)}"
        )
    tau = max(float(temperature), 1e-6)
    losses: list[torch.Tensor] = []
    for q, d in zip(query_tokens, image_tokens):
        if q is None or d is None or q.numel() == 0 or d.numel() == 0:
            continue
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if d.ndim == 1:
            d = d.unsqueeze(0)
        # (n_q, n_p) → max over query tokens → (n_p,)
        aff = (q @ d.T).max(dim=0).values
        n_p = int(aff.numel())
        if n_p <= 1:
            losses.append(aff.new_zeros(()))
            continue
        log_p = F.log_softmax(aff.float() / tau, dim=0)
        p = log_p.exp()
        entropy = -(p * log_p).sum()
        max_h = math.log(n_p)
        losses.append((entropy / max_h).to(dtype=q.dtype))
    if not losses:
        ref = query_tokens[0]
        return ref.new_zeros(()) if torch.is_tensor(ref) else torch.zeros(())
    return torch.stack(losses).mean()


def stack_token_embeddings(
    token_lists: list[list[torch.Tensor]],
) -> torch.Tensor | None:
    """Concatenate multi-token sequences into a single ``(N, D)`` matrix."""
    parts: list[torch.Tensor] = []
    for tokens in token_lists:
        for t in tokens:
            if t is None or t.numel() == 0:
                continue
            if t.ndim == 1:
                parts.append(t.unsqueeze(0))
            else:
                parts.append(t)
    if not parts:
        return None
    return torch.cat(parts, dim=0)


def mean_pool_token_list(tokens: list[torch.Tensor]) -> torch.Tensor | None:
    """Mean-pool each multi-token sequence → ``(B, D)``."""
    if not tokens:
        return None
    rows = []
    for t in tokens:
        if t is None or t.numel() == 0:
            continue
        if t.ndim == 1:
            rows.append(t)
        else:
            rows.append(t.mean(dim=0))
    if not rows:
        return None
    return torch.stack(rows, dim=0)


def embedding_geometry_loss(
    normalized: torch.Tensor,
    *,
    raw: torch.Tensor | None = None,
    ema_mean: torch.Tensor | None = None,
    center_weight: float = DEFAULT_GEO_CENTER_WEIGHT,
    var_weight: float = DEFAULT_GEO_VAR_WEIGHT,
    vec_mean_weight: float = DEFAULT_GEO_VEC_MEAN_WEIGHT,
    var_ratio: float = DEFAULT_GEO_VAR_RATIO,
    mag_floor: float = DEFAULT_GEO_MAG_FLOOR,
    mag_floor_weight: float = DEFAULT_GEO_MAG_FLOOR_WEIGHT,
    max_abs_ratio: float = DEFAULT_GEO_MAX_ABS_RATIO,
    max_abs_weight: float = DEFAULT_GEO_MAX_ABS_WEIGHT,
    uniformity_weight: float = DEFAULT_GEO_UNIFORMITY_WEIGHT,
    uniformity_t: float = DEFAULT_GEO_UNIFORMITY_T,
    uniformity_max_samples: int = DEFAULT_GEO_UNIFORMITY_MAX_SAMPLES,
    ema_blend: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Anti-cone / isotropy regularizer on L2-normalized embeddings.

    Parameters
    ----------
    normalized
        ``(N, D)`` **unit** vectors. Gradients flow. Callers must L2-normalize
        mean-pooled rows before passing them here.
    raw
        Optional matching ``(N, D)`` *pre-norm* projections for a magnitude floor
        (stops dead projections before normalize).
    ema_mean
        Optional detached running mean ``(D,)`` mixed into the center target so
        small micro-batches still see a stable cone direction to cancel.
        Must be a **raw mean of unit vectors** (not itself unit-normalized).

    Terms
    -----
    * **center** — ``||μ||²`` pushes batch (and EMA-blended) mean toward 0 so
      dims are not permanently biased (e.g. all v₀ ≈ 0.05).
    * **variance** — hinge below ``var_ratio / sqrt(D)`` per dim (VICReg-style).
    * **vec_mean** — ``E[(mean_d v_d)²]`` soft anti all-positive / all-negative.
    * **uniformity** — non-negative collapse penalty ``exp(L_W&I)`` where
      ``L_W&I = log E_{i≠j} exp(-t ||xi-xj||²)`` (≈1 when collapsed, ≈0 when spread).
      Always ≥ 0 so geo cannot drive total train loss negative.
    * **mag_floor** — pre-norm ``ReLU(ε - ||z||)`` so projections do not die.
    * **max_abs** — soft penalty when any |coord| ≫ isotropic scale.

    All terms are ≥ 0 (a "badness" score). Callers may square the aggregate for
    soft-when-small / hard-when-large behaviour.
    """
    empty_metrics = {
        "geo_center": 0.0,
        "geo_var": 0.0,
        "geo_vec_mean": 0.0,
        "geo_uniformity": 0.0,
        "geo_uniformity_pen": 0.0,
        "geo_mag_floor": 0.0,
        "geo_max_abs": 0.0,
        "geo_mu_norm": 0.0,
        "geo_min_std": 0.0,
        "geo_mean_abs_mu": 0.0,
    }
    if normalized.ndim == 1:
        normalized = normalized.unsqueeze(0)
    if normalized.numel() == 0 or normalized.shape[0] == 0:
        zero = normalized.new_zeros(())
        return zero, dict(empty_metrics)

    # Work in fp32 for stable stats; cast loss back to normalized dtype.
    v = normalized.float()
    n, dim = v.shape
    inv_sqrt_d = 1.0 / math.sqrt(max(dim, 1))
    var_target = float(var_ratio) * inv_sqrt_d
    max_abs_target = float(max_abs_ratio) * inv_sqrt_d

    batch_mu = v.mean(dim=0)
    if ema_mean is not None and ema_mean.numel() == dim:
        # Blend live mean with detached EMA so B=2 still sees the cone.
        # EMA is a raw mean of unit vectors (‖ema‖ ≤ 1), not a unit direction.
        mu = (1.0 - float(ema_blend)) * batch_mu + float(ema_blend) * ema_mean.float().detach()
    else:
        mu = batch_mu
    center = (mu * mu).sum()

    # Unbiased=False: small-N friendly; matches VICReg practice.
    std = v.std(dim=0, unbiased=False).clamp_min(0.0)
    # Mean hinge over dims under-taxes a few dead axes; also hinge on min std.
    var_per_dim = F.relu(var_target - std).pow(2)
    var_hinge = var_per_dim.mean() + var_per_dim.max()

    vec_mean = v.mean(dim=-1)
    vec_mean_pen = (vec_mean * vec_mean).mean()

    # Uniformity (Wang & Isola 2020) raw log-exp is ≤ 0 when spread and ~0 when
    # collapsed — that drove total train loss negative. Convert to a non-negative
    # collapse penalty: exp(L_W&I) ∈ (0, 1], ≈1 collapsed, ≈0 well-spread.
    unif_raw = v.new_zeros(())
    unif_pen = v.new_zeros(())
    if float(uniformity_weight) > 0.0 and n >= 2:
        max_s = max(int(uniformity_max_samples), 2)
        if n > max_s:
            # Deterministic stride subsample (grad-safe; no randperm graph issues).
            idx = torch.linspace(0, n - 1, steps=max_s, device=v.device).long()
            v_u = v.index_select(0, idx)
        else:
            v_u = v
        n_u = v_u.shape[0]
        # ||xi-xj||² = 2 - 2 xi·xj for unit vectors.
        sim = v_u @ v_u.T
        sq = (2.0 - 2.0 * sim).clamp_min(0.0)
        eye = torch.eye(n_u, dtype=torch.bool, device=v.device)
        t = float(uniformity_t)
        unif_raw = torch.log(
            torch.exp((-t) * sq.masked_select(~eye)).mean().clamp_min(1e-8)
        )
        # Clamp before exp for bf16 safety; still non-negative.
        unif_pen = torch.exp(unif_raw.clamp(max=20.0))

    mag_pen = v.new_zeros(())
    if (
        raw is not None
        and mag_floor_weight > 0.0
        and mag_floor > 0.0
        and raw.numel() > 0
    ):
        r = raw.float()
        if r.ndim == 1:
            r = r.unsqueeze(0)
        if r.shape[0] == n:
            norms = r.norm(dim=-1)
            mag_pen = F.relu(float(mag_floor) - norms).pow(2).mean()

    max_abs_pen = v.new_zeros(())
    if max_abs_weight > 0.0 and max_abs_target > 0.0:
        max_abs_pen = F.relu(v.abs().max(dim=-1).values - max_abs_target).pow(2).mean()

    # All terms ≥ 0: a pure "geometry badness" score (no negative free lunch).
    total = (
        float(center_weight) * center
        + float(var_weight) * var_hinge
        + float(vec_mean_weight) * vec_mean_pen
        + float(uniformity_weight) * unif_pen
        + float(mag_floor_weight) * mag_pen
        + float(max_abs_weight) * max_abs_pen
    )
    total = total.to(dtype=normalized.dtype)

    metrics = {
        "geo_center": float(center.detach()),
        "geo_var": float(var_hinge.detach()),
        "geo_vec_mean": float(vec_mean_pen.detach()),
        # Raw W&I (can be negative) for debugging; pen is what enters the loss.
        "geo_uniformity": float(unif_raw.detach()) if torch.is_tensor(unif_raw) else 0.0,
        "geo_uniformity_pen": float(unif_pen.detach()) if torch.is_tensor(unif_pen) else 0.0,
        "geo_mag_floor": float(mag_pen.detach()) if torch.is_tensor(mag_pen) else 0.0,
        "geo_max_abs": float(max_abs_pen.detach()) if torch.is_tensor(max_abs_pen) else 0.0,
        "geo_mu_norm": float(batch_mu.detach().norm()),
        "geo_min_std": float(std.detach().min()),
        "geo_mean_abs_mu": float(batch_mu.detach().abs().mean()),
    }
    return total, metrics


def update_embedding_ema(
    ema: torch.Tensor,
    batch_mean: torch.Tensor,
    momentum: float = DEFAULT_GEO_EMA_MOMENTUM,
) -> torch.Tensor:
    """In-place EMA update of the embedding mean; returns the updated buffer."""
    if ema.shape != batch_mean.shape:
        raise ValueError(
            f"EMA shape {tuple(ema.shape)} != batch mean {tuple(batch_mean.shape)}"
        )
    m = float(momentum)
    # ema ← m * ema + (1-m) * batch_mean  (both detached)
    ema.mul_(m).add_(batch_mean.detach(), alpha=1.0 - m)
    return ema


@torch.no_grad()
def mean_positive_rank(
    scores: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> float:
    """1-based mean rank of the positive class (1 = best). Lower is better.

    Uses **mid-rank for ties**: when many docs share the positive's score
    (including full collapse where every logit is equal), rank → ``(N+1)/2``
    instead of optimistically reporting 1. Non-finite scores (``-inf`` from hard
    bank mining) are ignored so they neither beat nor tie the positive.
    """
    if scores.ndim != 2 or scores.size(0) == 0:
        return float("nan")
    batch = scores.size(0)
    if labels is None:
        labels = torch.arange(batch, device=scores.device)
    pos = scores.gather(1, labels.view(-1, 1)).squeeze(1)
    finite = torch.isfinite(scores)
    # Treat non-finite as "not comparable" (hard-mined bank columns).
    cmp = scores.masked_fill(~finite, float("-inf"))
    pos_col = pos.unsqueeze(1)
    better = (finite & (cmp > pos_col)).sum(dim=1).to(dtype=torch.float32)
    # Ties include the positive column itself when it is finite.
    tied = (finite & (cmp == pos_col)).sum(dim=1).to(dtype=torch.float32)
    # Average rank of a tied group: better + (tied + 1) / 2.
    ranks = better + (tied + 1.0) * 0.5
    return float(ranks.mean().item())


def score_gap_per_row(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Differentiable per-row gap: ``pos - mean(finite negs)`` on logits ``(B, N)``.

    Non-finite scores (hard-mined bank ``-inf``) are excluded from the neg mean.
    """
    if scores.ndim != 2 or scores.size(0) == 0:
        return scores.new_zeros(())
    pos = scores.gather(1, labels.view(-1, 1)).squeeze(1)
    eye = torch.zeros_like(scores, dtype=torch.bool)
    eye.scatter_(1, labels.view(-1, 1), True)
    neg_mask = torch.isfinite(scores) & ~eye
    # Zero masked entries; divide by count (grad-safe, no nanmean).
    neg_sum = scores.masked_fill(~neg_mask, 0.0).sum(dim=1)
    neg_count = neg_mask.to(dtype=scores.dtype).sum(dim=1).clamp_min(1.0)
    neg_mean = neg_sum / neg_count
    return pos - neg_mean


def gap_hinge_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = DEFAULT_GAP_MARGIN,
) -> torch.Tensor:
    """``mean ReLU(margin - gap_i)`` — 0 iff every row has gap ≥ margin.

    Directly trains the logged ``score_gap`` signal (pos above mean hard negs).
    """
    gap = score_gap_per_row(scores, labels)
    if gap.ndim == 0:
        return gap
    return F.relu(float(margin) - gap).mean()


@torch.no_grad()
def contrastive_score_margin_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Pos vs mean-neg logit stats (after /temperature, after hard-bank mask).

    ``score_gap = mean(pos) - mean(finite off-diagonal)``. Near 0 with high CE
    means the contrastive head has no ranking signal (collapse / chance).
    """
    if scores.ndim != 2 or scores.size(0) == 0:
        return {
            "pos_score": float("nan"),
            "neg_score": float("nan"),
            "score_gap": float("nan"),
        }
    gap = score_gap_per_row(scores, labels)
    pos = scores.gather(1, labels.view(-1, 1)).squeeze(1)
    eye = torch.zeros_like(scores, dtype=torch.bool)
    eye.scatter_(1, labels.view(-1, 1), True)
    neg_mask = torch.isfinite(scores) & ~eye
    neg_sum = scores.masked_fill(~neg_mask, 0.0).sum(dim=1)
    neg_count = neg_mask.to(dtype=scores.dtype).sum(dim=1).clamp_min(1.0)
    neg_mean = neg_sum / neg_count
    return {
        "pos_score": float(pos.mean().item()),
        "neg_score": float(neg_mean.mean().item()),
        "score_gap": float(gap.mean().item()),
    }


class EmbeddingMemoryBank:
    """FIFO queue of detached multi-token embeddings used as contrastive negatives.

    Stores *raw* (pre-normalization) projected token sequences so Matryoshka
    prefixes can be re-derived. Entries never receive gradients — they act as
    a MoCo-style memory bank, giving a large effective negative set without
    holding a large micro-batch of activations.

    **Policy B** (stage-1 default): enqueue every micro-batch into the live FIFO,
    but *score* against a snapshot taken at the start of each gradient-
    accumulation window (see ``snapshot`` / ``Stage1AlignmentModel.begin_accum_window``).
    """

    def __init__(self, capacity: int = 0):
        self.capacity = max(0, int(capacity))
        self._image_raw: list[torch.Tensor] = []
        self._text_raw: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self._image_raw)

    @property
    def enabled(self) -> bool:
        return self.capacity > 0

    def clear(self) -> None:
        self._image_raw.clear()
        self._text_raw.clear()

    def _enqueue(self, bucket: list[torch.Tensor], items: list[torch.Tensor]) -> None:
        if not self.enabled or not items:
            return
        for item in items:
            bucket.append(item.detach().contiguous())
        overflow = len(bucket) - self.capacity
        if overflow > 0:
            del bucket[:overflow]

    def enqueue(
        self,
        *,
        image_raw: list[torch.Tensor] | None = None,
        text_raw: list[torch.Tensor] | None = None,
    ) -> None:
        if image_raw is not None:
            self._enqueue(self._image_raw, image_raw)
        if text_raw is not None:
            self._enqueue(self._text_raw, text_raw)

    def image_raw(self) -> list[torch.Tensor]:
        return list(self._image_raw)

    def text_raw(self) -> list[torch.Tensor]:
        return list(self._text_raw)

    def snapshot(self) -> dict[str, list[torch.Tensor]]:
        """Detach-copy of current bank contents for scoring (policy B)."""
        return {
            "image_raw": [t.detach().contiguous() for t in self._image_raw],
            "text_raw": [t.detach().contiguous() for t in self._text_raw],
        }

    def normalized_images(self, dim: int | None = None) -> list[torch.Tensor]:
        return [matryoshka_normalize(t, dim=dim) for t in self._image_raw]

    def normalized_texts(self, dim: int | None = None) -> list[torch.Tensor]:
        return [matryoshka_normalize(t, dim=dim) for t in self._text_raw]


DEFAULT_MEMORY_BANK_SIZE = 128


def matryoshka_prefix_dims(
    dims: tuple[int, ...] | list[int],
    embed_dim: int = EMBED_DIM,
) -> tuple[int, ...]:
    """Keep only strict prefixes ``0 < d < embed_dim`` (no full-dim double count)."""
    out: list[int] = []
    seen: set[int] = set()
    for d in dims:
        d_int = int(d)
        if d_int <= 0 or d_int >= int(embed_dim) or d_int in seen:
            continue
        seen.add(d_int)
        out.append(d_int)
    return tuple(out)


def apply_hard_bank_mining(
    scores: torch.Tensor,
    n_live: int,
    hard_k: int,
    *,
    labels: torch.Tensor | None = None,
    bank_fn_margin: float | None = DEFAULT_BANK_FN_MARGIN,
    bank_random_k: int = DEFAULT_BANK_RANDOM_K,
) -> torch.Tensor:
    """Select bank negatives for InfoNCE; live columns always kept.

    ``scores`` is ``(B, n_live + n_bank)`` with live docs in ``[:, :n_live]``.

    1. **FN filter** (when ``labels`` is set and ``bank_fn_margin`` is not None):
       bank columns with ``score >= pos - margin`` are set to ``-inf`` so
       near-duplicates of the positive are not treated as hard negatives.
    2. **Hard-k** (when ``hard_k > 0``): keep the top-``hard_k`` hardest
       *remaining* bank columns.
    3. **Random-k** (only with hard mining): keep up to ``bank_random_k``
       additional bank columns drawn uniformly from the non-hard, non-FN
       remainder (diversity so the model is not stuck on a few hard FNs).

    ``hard_k <= 0`` means use the full bank after FN filtering (``bank_random_k``
    is ignored in that mode).
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {tuple(scores.shape)}")
    batch, n_docs = scores.shape
    if n_live < 0 or n_live > n_docs:
        raise ValueError(f"n_live={n_live} invalid for n_docs={n_docs}")
    n_bank = n_docs - n_live
    if n_bank <= 0:
        return scores

    live = scores[:, :n_live]
    bank = scores[:, n_live:].clone()
    neg_large = float("-inf")

    # --- False-negative filter relative to live positive ---
    if labels is not None and bank_fn_margin is not None:
        if labels.shape[0] != batch:
            raise ValueError(
                f"labels batch {labels.shape[0]} != scores batch {batch}"
            )
        # Positive is always among live columns [0, n_live).
        pos = scores.gather(1, labels.view(-1, 1).clamp(max=n_live - 1)).squeeze(1)
        thr = pos.unsqueeze(1) - float(bank_fn_margin)
        # Bank at/above threshold ≈ too similar to positive → not a negative.
        bank = bank.masked_fill(bank >= thr, neg_large)

    hard_k = 0 if hard_k is None else int(hard_k)
    random_k = 0 if bank_random_k is None else max(int(bank_random_k), 0)

    if hard_k <= 0:
        # Full bank after FN filter (CLI: hard_k=0 → use full bank).
        return torch.cat([live, bank], dim=1)

    keep = torch.zeros_like(bank, dtype=torch.bool)
    # Eligible = finite after FN filter.
    eligible = torch.isfinite(bank)

    for i in range(batch):
        elig_idx = eligible[i].nonzero(as_tuple=False).view(-1)
        if elig_idx.numel() == 0:
            continue
        vals = bank[i, elig_idx]
        # Hard: top-k among eligible
        k_h = min(hard_k, int(elig_idx.numel()))
        top = torch.topk(vals, k=k_h, dim=0).indices
        hard_sel = elig_idx[top]
        keep[i, hard_sel] = True
        hard_set = {int(x) for x in hard_sel.tolist()}
        # Random among remaining eligible (diversity; optional).
        if random_k > 0:
            remain = [int(x) for x in elig_idx.tolist() if int(x) not in hard_set]
            if remain:
                k_r = min(random_k, len(remain))
                perm = torch.randperm(len(remain), device=scores.device)[:k_r]
                for j in perm.tolist():
                    keep[i, remain[j]] = True

    bank = bank.masked_fill(~keep, neg_large)
    return torch.cat([live, bank], dim=1)


def _expand_non_negative_mask(
    batch_mask: torch.Tensor | None,
    n_docs: int,
    batch: int,
    *,
    n_live: int | None = None,
) -> torch.Tensor | None:
    """Pad a ``(B, B)`` in-batch mask to ``(B, n_docs)`` with False for extra cols.

    When ``n_live`` is set (e.g. captions + distractors for query→text), the mask
    only covers the first ``batch`` columns; columns ``batch:n_live`` and bank
    columns stay False (always eligible as negatives).
    """
    if batch_mask is None:
        return None
    if batch_mask.shape != (batch, batch):
        raise ValueError(
            f"non_negative_mask expected shape {(batch, batch)}, "
            f"got {tuple(batch_mask.shape)}"
        )
    if n_docs == batch:
        return batch_mask
    live = n_live if n_live is not None else batch
    if live < batch or live > n_docs:
        raise ValueError(f"n_live={live} invalid for batch={batch}, n_docs={n_docs}")
    extra = n_docs - batch
    if extra < 0:
        raise ValueError(f"n_docs {n_docs} < batch {batch}")
    # Pad after the B×B block: distractors (if any) + bank are never multi-pos.
    pad = torch.zeros(
        batch, extra, dtype=batch_mask.dtype, device=batch_mask.device
    )
    return torch.cat([batch_mask, pad], dim=1)


def _maybe_score_center_pair(
    text_tokens: list[torch.Tensor],
    image_tokens: list[torch.Tensor],
    bank_text: list[torch.Tensor],
    bank_image: list[torch.Tensor],
    *,
    score_center: bool,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """Subtract detached live-batch mean so MaxSim cannot match a domain cone."""
    if not score_center:
        return text_tokens, image_tokens, bank_text, bank_image
    center = mean_unit_token_center(text_tokens, image_tokens)
    if center is None:
        return text_tokens, image_tokens, bank_text, bank_image
    return (
        apply_score_center(text_tokens, center),
        apply_score_center(image_tokens, center),
        apply_score_center(bank_text, center) if bank_text else bank_text,
        apply_score_center(bank_image, center) if bank_image else bank_image,
    )


def contrastive_late_interaction_loss(
    text_tokens: list[torch.Tensor],
    image_tokens: list[torch.Tensor],
    temperature: float = 0.07,
    *,
    bank_text_tokens: list[torch.Tensor] | None = None,
    bank_image_tokens: list[torch.Tensor] | None = None,
    soft_maxsim_temperature: float | None = None,
    non_negative_mask: torch.Tensor | None = None,
    hard_bank_k: int = 0,
    bank_fn_margin: float | None = DEFAULT_BANK_FN_MARGIN,
    bank_random_k: int = DEFAULT_BANK_RANDOM_K,
    query_topk: int = 0,
    score_center: bool = DEFAULT_SCORE_CENTER,
    gap_weight: float = DEFAULT_GAP_LOSS_WEIGHT,
    gap_margin: float = DEFAULT_GAP_MARGIN,
    return_metrics: bool = False,
    text_image_ids: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Bidirectional late-interaction InfoNCE with optional memory-bank negatives.

    When the bank is empty this matches the historical square-matrix form
    (t2i on ``S``, i2t on ``S.T``). With a bank, each side scores the live batch
    positives first, then bank docs as extra negatives (labels stay in ``0..B-1``).

    When ``text_image_ids`` is provided (length T = len(text_tokens)), multiple
    texts may map to the same image: t→i uses single-label CE; i→t uses
    multi-positive InfoNCE over all texts of that image.

    Scores use mean-MaxSim (see ``differentiable_late_interaction_score``),
    optionally soft MaxSim. ``non_negative_mask`` is an optional ``(B_img, B_img)``
    bool mask of image-level near-duplicates (expanded to text rows via ids).

    ``hard_bank_k`` keeps only the top-k hardest bank docs per query (see
    ``apply_hard_bank_mining``); live in-batch columns are always kept.
    ``bank_fn_margin`` / ``bank_random_k`` control FN filtering and hard+random
    bank mix (see ``apply_hard_bank_mining``).

    ``score_center`` subtracts the detached mean of live unit tokens before
    scoring (removes domain cone so background MaxSim cannot equate all pairs).

    ``query_topk`` averages only the strongest query-token MaxSims.

    ``gap_weight`` / ``gap_margin`` add ``ReLU(margin - score_gap)`` on the same
    logits as CE (direct training signal for the logged gap).

    When ``return_metrics`` is True, also returns mean positive ranks (1 = best)
    averaged over t2i and i2t — useful as a bank-size-invariant health signal.
    """
    n_text = len(text_tokens)
    n_image = len(image_tokens)
    if n_text < 1 or n_image < 1:
        raise ValueError("Contrastive loss needs non-empty text and image lists.")

    # Square 1:1 path when ids omitted or diagonal.
    use_multipos = False
    if text_image_ids is not None:
        ids = text_image_ids.to(device=text_tokens[0].device, dtype=torch.long).view(-1)
        if ids.numel() != n_text:
            raise ValueError(
                f"text_image_ids length {ids.numel()} != n_text {n_text}"
            )
        if n_text != n_image or not bool(torch.equal(ids, torch.arange(n_image, device=ids.device))):
            use_multipos = True
    else:
        ids = torch.arange(n_image, device=text_tokens[0].device)
        if n_text != n_image:
            raise ValueError(
                f"text/image batch size mismatch: {n_text} vs {n_image} "
                "(pass text_image_ids for multi-text positives)"
            )

    bank_text = bank_text_tokens or []
    bank_image = bank_image_tokens or []
    if n_image < 2 and not bank_text and not bank_image:
        raise ValueError(
            f"Contrastive loss needs batch_size >= 2 (got {n_image} images), "
            "or a non-empty memory bank for negatives."
        )

    text_tokens, image_tokens, bank_text, bank_image = _maybe_score_center_pair(
        text_tokens,
        image_tokens,
        bank_text,
        bank_image,
        score_center=score_center,
    )

    soft_tau = soft_maxsim_temperature
    q_topk = int(query_topk or 0)
    n_image_docs = n_image + len(bank_image)
    n_text_docs = n_text + len(bank_text)
    # t→i labels: each text maps to its image index
    labels_t2i = ids
    # For gap/rank on multipos i2t, use first text index per image as representative.
    primary_text_idx = torch.full((n_image,), -1, device=ids.device, dtype=torch.long)
    for t_i, img_i in enumerate(ids.tolist()):
        if 0 <= img_i < n_image and primary_text_idx[img_i] < 0:
            primary_text_idx[img_i] = t_i
    if bool((primary_text_idx < 0).any()):
        missing = (primary_text_idx < 0).nonzero(as_tuple=False).view(-1).tolist()
        raise ValueError(f"No text positives for image indices {missing}")

    def _combine_ce_and_gap(
        scores_a: torch.Tensor,
        scores_b: torch.Tensor,
        ce: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gap_w = float(gap_weight)
        if gap_w <= 0.0:
            return ce, ce.new_zeros(())
        g = 0.5 * (
            gap_hinge_loss(scores_a, labels_a, margin=gap_margin)
            + gap_hinge_loss(scores_b, labels_b, margin=gap_margin)
        )
        return ce + gap_w * g, g

    # --- Expand image-level non-neg mask to text rows ---
    def _mask_t2i(n_docs: int) -> torch.Tensor | None:
        if non_negative_mask is None:
            return None
        # non_negative_mask is (n_image, n_image); expand rows by text→image
        base = _expand_non_negative_mask(non_negative_mask, n_docs, n_image)
        if base is None:
            return None
        return base[ids]  # (n_text, n_docs)

    def _mask_i2t_live(n_live_text: int, n_docs: int) -> torch.Tensor | None:
        """Soft FN mask on i2t: mark other images' near-dup primary texts."""
        if non_negative_mask is None:
            return None
        # Build (n_image, n_live_text) from image-image mask via text ids
        # entry (i, t) soft if non_neg[i, ids[t]] and i != ids[t]
        img_of_t = ids[:n_live_text]
        soft = non_negative_mask[:, img_of_t]  # (B, T_live)
        # Structural multipos are handled by multipos CE, not this mask.
        if n_docs > n_live_text:
            pad = soft.new_zeros((n_image, n_docs - n_live_text))
            soft = torch.cat([soft, pad], dim=1)
        return soft

    if not use_multipos and not bank_text and not bank_image:
        scores = build_late_interaction_matrix(
            text_tokens,
            image_tokens,
            soft_maxsim_temperature=soft_tau,
            query_topk=q_topk,
        ) / temperature
        labels = torch.arange(n_image, device=scores.device)
        mask = _expand_non_negative_mask(non_negative_mask, n_image, n_image)
        loss_t2i = masked_cross_entropy(scores, labels, non_negative_mask=mask)
        mask_t = mask.T if mask is not None else None
        loss_i2t = masked_cross_entropy(scores.T, labels, non_negative_mask=mask_t)
        ce = 0.5 * (loss_t2i + loss_i2t)
        loss, gap_l = _combine_ce_and_gap(scores, scores.T, ce, labels, labels)
        if not return_metrics:
            return loss
        rank_t2i = mean_positive_rank(scores.detach(), labels)
        rank_i2t = mean_positive_rank(scores.T.detach(), labels)
        m_t2i = contrastive_score_margin_metrics(scores.detach(), labels)
        m_i2t = contrastive_score_margin_metrics(scores.T.detach(), labels)
        return loss, {
            "pos_rank_t2i": rank_t2i,
            "pos_rank_i2t": rank_i2t,
            "pos_rank": 0.5 * (rank_t2i + rank_i2t),
            "n_image_docs": float(n_image_docs),
            "n_text_docs": float(n_text_docs),
            "pos_score": 0.5 * (m_t2i["pos_score"] + m_i2t["pos_score"]),
            "neg_score": 0.5 * (m_t2i["neg_score"] + m_i2t["neg_score"]),
            "score_gap": 0.5 * (m_t2i["score_gap"] + m_i2t["score_gap"]),
            "gap_hinge": float(gap_l.detach()),
            "contrastive_ce": float(ce.detach()),
        }

    image_docs = list(image_tokens) + list(bank_image)
    text_docs = list(text_tokens) + list(bank_text)
    scores_t2i = build_late_interaction_matrix(
        text_tokens,
        image_docs,
        soft_maxsim_temperature=soft_tau,
        query_topk=q_topk,
    ) / temperature
    scores_i2t = build_late_interaction_matrix(
        image_tokens,
        text_docs,
        soft_maxsim_temperature=soft_tau,
        query_topk=q_topk,
    ) / temperature

    scores_t2i = apply_hard_bank_mining(
        scores_t2i,
        n_image if not use_multipos else n_image,
        hard_bank_k,
        labels=labels_t2i if use_multipos else torch.arange(n_image, device=ids.device),
        bank_fn_margin=bank_fn_margin,
        bank_random_k=bank_random_k,
    )
    # For i2t bank mining with multipos: label = first positive text col
    labels_i2t_primary = primary_text_idx
    scores_i2t = apply_hard_bank_mining(
        scores_i2t,
        n_text,
        hard_bank_k,
        labels=labels_i2t_primary if use_multipos else torch.arange(n_image, device=ids.device),
        bank_fn_margin=bank_fn_margin,
        bank_random_k=bank_random_k,
    )

    if use_multipos:
        # t→i: single label per text
        mask_t2i = _mask_t2i(n_image_docs)
        loss_t2i = masked_cross_entropy(
            scores_t2i, labels_t2i, non_negative_mask=mask_t2i
        )
        # i→t: multi-positive over live text columns (+ bank never structural pos)
        pos_mask = text_image_positive_mask(
            ids, n_image, n_text, device=scores_i2t.device
        )
        if n_text_docs > n_text:
            pos_mask = torch.cat(
                [
                    pos_mask,
                    pos_mask.new_zeros((n_image, n_text_docs - n_text)),
                ],
                dim=1,
            )
        mask_i2t = _mask_i2t_live(n_text, n_text_docs)
        loss_i2t = multi_positive_cross_entropy(
            scores_i2t, pos_mask, non_negative_mask=mask_i2t
        )
        ce = 0.5 * (loss_t2i + loss_i2t)
        loss, gap_l = _combine_ce_and_gap(
            scores_t2i,
            scores_i2t,
            ce,
            labels_t2i,
            labels_i2t_primary,
        )
        if not return_metrics:
            return loss
        rank_t2i = mean_positive_rank(scores_t2i.detach(), labels_t2i)
        rank_i2t = mean_positive_rank(scores_i2t.detach(), labels_i2t_primary)
        m_t2i = contrastive_score_margin_metrics(scores_t2i.detach(), labels_t2i)
        m_i2t = contrastive_score_margin_metrics(
            scores_i2t.detach(), labels_i2t_primary
        )
        return loss, {
            "pos_rank_t2i": rank_t2i,
            "pos_rank_i2t": rank_i2t,
            "pos_rank": 0.5 * (rank_t2i + rank_i2t),
            "n_image_docs": float(n_image_docs),
            "n_text_docs": float(n_text_docs),
            "pos_score": 0.5 * (m_t2i["pos_score"] + m_i2t["pos_score"]),
            "neg_score": 0.5 * (m_t2i["neg_score"] + m_i2t["neg_score"]),
            "score_gap": 0.5 * (m_t2i["score_gap"] + m_i2t["score_gap"]),
            "gap_hinge": float(gap_l.detach()),
            "contrastive_ce": float(ce.detach()),
        }

    # Square + bank (original path)
    labels = torch.arange(n_image, device=scores_t2i.device)
    mask_t2i = _expand_non_negative_mask(non_negative_mask, n_image_docs, n_image)
    mask_i2t = _expand_non_negative_mask(
        non_negative_mask.T if non_negative_mask is not None else None,
        n_text_docs,
        n_image,
    )
    loss_t2i = masked_cross_entropy(
        scores_t2i, labels, non_negative_mask=mask_t2i
    )
    loss_i2t = masked_cross_entropy(
        scores_i2t, labels, non_negative_mask=mask_i2t
    )
    ce = 0.5 * (loss_t2i + loss_i2t)
    loss, gap_l = _combine_ce_and_gap(scores_t2i, scores_i2t, ce, labels, labels)
    if not return_metrics:
        return loss
    rank_t2i = mean_positive_rank(scores_t2i.detach(), labels)
    rank_i2t = mean_positive_rank(scores_i2t.detach(), labels)
    m_t2i = contrastive_score_margin_metrics(scores_t2i.detach(), labels)
    m_i2t = contrastive_score_margin_metrics(scores_i2t.detach(), labels)
    return loss, {
        "pos_rank_t2i": rank_t2i,
        "pos_rank_i2t": rank_i2t,
        "pos_rank": 0.5 * (rank_t2i + rank_i2t),
        "n_image_docs": float(n_image_docs),
        "n_text_docs": float(n_text_docs),
        "pos_score": 0.5 * (m_t2i["pos_score"] + m_i2t["pos_score"]),
        "neg_score": 0.5 * (m_t2i["neg_score"] + m_i2t["neg_score"]),
        "score_gap": 0.5 * (m_t2i["score_gap"] + m_i2t["score_gap"]),
        "gap_hinge": float(gap_l.detach()),
        "contrastive_ce": float(ce.detach()),
    }


def paraphrase_contrastive_loss(
    anchor_tokens: list[torch.Tensor],
    positive_tokens: list[torch.Tensor],
    *,
    negative_tokens: list[torch.Tensor] | None = None,
    temperature: float = 0.07,
    bank_doc_tokens: list[torch.Tensor] | None = None,
    soft_maxsim_temperature: float | None = None,
    hard_bank_k: int = 0,
    bank_fn_margin: float | None = DEFAULT_BANK_FN_MARGIN,
    bank_random_k: int = DEFAULT_BANK_RANDOM_K,
    query_topk: int = 0,
    score_center: bool = DEFAULT_SCORE_CENTER,
    gap_weight: float = DEFAULT_GAP_LOSS_WEIGHT,
    gap_margin: float = DEFAULT_GAP_MARGIN,
) -> torch.Tensor:
    """Anchor→positive late-interaction InfoNCE (AllNLI / paraphrase).

    Docs = ``[positives | hard_negatives | bank]``. Labels point at the matching
    positive (diagonal in the positive block). Explicit NLI contradiction
    negatives (when provided) sit after the positive block as live hard negs.
    """
    batch = len(anchor_tokens)
    if batch != len(positive_tokens):
        raise ValueError(
            f"anchor/positive batch mismatch: {batch} vs {len(positive_tokens)}"
        )
    if batch < 1:
        raise ValueError("paraphrase_contrastive_loss needs a non-empty batch")
    negs = negative_tokens or []
    bank = bank_doc_tokens or []
    if batch < 2 and not negs and not bank:
        raise ValueError(
            "paraphrase_contrastive_loss needs batch>=2 or negatives/bank"
        )

    if score_center:
        center = mean_unit_token_center(anchor_tokens, positive_tokens, negs)
        if center is not None:
            anchor_tokens = apply_score_center(anchor_tokens, center)
            positive_tokens = apply_score_center(positive_tokens, center)
            negs = apply_score_center(negs, center) if negs else negs
            bank = apply_score_center(bank, center) if bank else bank

    docs = list(positive_tokens) + list(negs) + list(bank)
    scores = build_late_interaction_matrix(
        anchor_tokens,
        docs,
        soft_maxsim_temperature=soft_maxsim_temperature,
        query_topk=int(query_topk or 0),
    ) / temperature
    labels = torch.arange(batch, device=scores.device)
    scores = apply_hard_bank_mining(
        scores,
        batch + len(negs),  # live = positives + explicit negs
        hard_bank_k,
        labels=labels,
        bank_fn_margin=bank_fn_margin,
        bank_random_k=bank_random_k,
    )
    ce = F.cross_entropy(scores, labels)
    if float(gap_weight) > 0.0:
        g = gap_hinge_loss(scores, labels, margin=gap_margin)
        return ce + float(gap_weight) * g
    return ce


def text_text_contrastive_loss(
    query_tokens: list[torch.Tensor],
    caption_tokens: list[torch.Tensor],
    distractor_tokens: list[torch.Tensor],
    temperature: float = 0.07,
    *,
    bank_doc_tokens: list[torch.Tensor] | None = None,
    soft_maxsim_temperature: float | None = None,
    non_negative_mask: torch.Tensor | None = None,
    hard_bank_k: int = 0,
    bank_fn_margin: float | None = DEFAULT_BANK_FN_MARGIN,
    bank_random_k: int = DEFAULT_BANK_RANDOM_K,
    query_topk: int = 0,
    score_center: bool = DEFAULT_SCORE_CENTER,
    gap_weight: float = DEFAULT_GAP_LOSS_WEIGHT,
    gap_margin: float = DEFAULT_GAP_MARGIN,
) -> torch.Tensor:
    """Query→caption InfoNCE: match own caption; not other captions / distractors / bank.

    Docs are ``[captions | distractors | bank]``. Labels are diagonal into the
    caption block. Multi-positive mask only softens near-duplicate *captions*.
    """
    if len(query_tokens) != len(caption_tokens):
        raise ValueError(
            "query_tokens and caption_tokens must have the same batch size."
        )
    bank_docs = bank_doc_tokens or []
    if len(query_tokens) < 2 and not distractor_tokens and not bank_docs:
        raise ValueError(
            "text-text contrastive loss needs batch_size >= 2 for negatives."
        )
    batch = len(query_tokens)
    if score_center:
        center = mean_unit_token_center(
            query_tokens, caption_tokens, distractor_tokens or []
        )
        if center is not None:
            query_tokens = apply_score_center(query_tokens, center)
            caption_tokens = apply_score_center(caption_tokens, center)
            if distractor_tokens:
                distractor_tokens = apply_score_center(distractor_tokens, center)
            if bank_docs:
                bank_docs = apply_score_center(bank_docs, center)
    n_live = batch + len(distractor_tokens)
    all_docs = list(caption_tokens) + list(distractor_tokens) + list(bank_docs)
    scores = build_late_interaction_matrix(
        query_tokens,
        all_docs,
        soft_maxsim_temperature=soft_maxsim_temperature,
        query_topk=int(query_topk or 0),
    ) / temperature
    labels = torch.arange(scores.size(0), device=scores.device)
    scores = apply_hard_bank_mining(
        scores,
        n_live,
        hard_bank_k,
        labels=labels,
        bank_fn_margin=bank_fn_margin,
        bank_random_k=bank_random_k,
    )
    # Only the in-batch caption columns participate in multi-positive masking.
    mask = _expand_non_negative_mask(
        non_negative_mask, scores.size(1), batch, n_live=n_live
    )
    ce = masked_cross_entropy(scores, labels, non_negative_mask=mask)
    if float(gap_weight) <= 0.0:
        return ce
    return ce + float(gap_weight) * gap_hinge_loss(
        scores, labels, margin=gap_margin
    )


def text_text_matryoshka_loss(
    query_raw: list[torch.Tensor],
    caption_raw: list[torch.Tensor],
    distractor_raw: list[torch.Tensor],
    dims: tuple[int, ...],
    temperature: float,
    dim_weights: list[float] | None = None,
    *,
    bank_doc_raw: list[torch.Tensor] | None = None,
    soft_maxsim_temperature: float | None = None,
    non_negative_mask: torch.Tensor | None = None,
    hard_bank_k: int = 0,
    bank_fn_margin: float | None = DEFAULT_BANK_FN_MARGIN,
    bank_random_k: int = DEFAULT_BANK_RANDOM_K,
    query_topk: int = 0,
    score_center: bool = DEFAULT_SCORE_CENTER,
    gap_weight: float = DEFAULT_GAP_LOSS_WEIGHT,
    gap_margin: float = DEFAULT_GAP_MARGIN,
    embed_dim: int = EMBED_DIM,
) -> torch.Tensor:
    """Mean of prefix-only query→caption CE (full dim handled by the main term)."""
    prefix_dims = matryoshka_prefix_dims(dims, embed_dim=embed_dim)
    if not prefix_dims:
        return query_raw[0].new_zeros(())
    if dim_weights is None:
        dim_weights = [1.0] * len(prefix_dims)
    if len(dim_weights) != len(prefix_dims):
        dim_weights = [1.0] * len(prefix_dims)
    total_weight = sum(dim_weights) or 1.0
    bank_raw = bank_doc_raw or []
    loss = query_raw[0].new_zeros(())
    for dim, weight in zip(prefix_dims, dim_weights):
        query_prefix = [matryoshka_normalize(t, dim=dim) for t in query_raw]
        caption_prefix = [matryoshka_normalize(t, dim=dim) for t in caption_raw]
        distractor_prefix = [
            matryoshka_normalize(t, dim=dim) for t in distractor_raw
        ]
        bank_prefix = [matryoshka_normalize(t, dim=dim) for t in bank_raw]
        loss = loss + weight * text_text_contrastive_loss(
            query_prefix,
            caption_prefix,
            distractor_prefix,
            temperature=temperature,
            bank_doc_tokens=bank_prefix,
            soft_maxsim_temperature=soft_maxsim_temperature,
            non_negative_mask=non_negative_mask,
            hard_bank_k=hard_bank_k,
            bank_fn_margin=bank_fn_margin,
            bank_random_k=bank_random_k,
            query_topk=query_topk,
            score_center=score_center,
            gap_weight=gap_weight,
            gap_margin=gap_margin,
        )
    return loss / total_weight


def matryoshka_loss(
    text_raw: list[torch.Tensor],
    image_raw: list[torch.Tensor],
    dims: tuple[int, ...],
    temperature: float,
    dim_weights: list[float] | None = None,
    *,
    bank_text_raw: list[torch.Tensor] | None = None,
    bank_image_raw: list[torch.Tensor] | None = None,
    soft_maxsim_temperature: float | None = None,
    non_negative_mask: torch.Tensor | None = None,
    hard_bank_k: int = 0,
    bank_fn_margin: float | None = DEFAULT_BANK_FN_MARGIN,
    bank_random_k: int = DEFAULT_BANK_RANDOM_K,
    query_topk: int = 0,
    score_center: bool = DEFAULT_SCORE_CENTER,
    gap_weight: float = DEFAULT_GAP_LOSS_WEIGHT,
    gap_margin: float = DEFAULT_GAP_MARGIN,
    embed_dim: int = EMBED_DIM,
    text_image_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean of prefix-only bidirectional CE (full dim handled by the main term)."""
    prefix_dims = matryoshka_prefix_dims(dims, embed_dim=embed_dim)
    if not prefix_dims:
        return text_raw[0].new_zeros(())
    if dim_weights is None:
        dim_weights = [1.0] * len(prefix_dims)
    if len(dim_weights) != len(prefix_dims):
        dim_weights = [1.0] * len(prefix_dims)
    total_weight = sum(dim_weights) or 1.0
    bank_text = bank_text_raw or []
    bank_image = bank_image_raw or []
    loss = text_raw[0].new_zeros(())
    for dim, weight in zip(prefix_dims, dim_weights):
        text_prefix = [matryoshka_normalize(t, dim=dim) for t in text_raw]
        image_prefix = [matryoshka_normalize(t, dim=dim) for t in image_raw]
        bank_text_prefix = [matryoshka_normalize(t, dim=dim) for t in bank_text]
        bank_image_prefix = [matryoshka_normalize(t, dim=dim) for t in bank_image]
        loss = loss + weight * contrastive_late_interaction_loss(
            text_prefix,
            image_prefix,
            temperature=temperature,
            bank_text_tokens=bank_text_prefix,
            bank_image_tokens=bank_image_prefix,
            soft_maxsim_temperature=soft_maxsim_temperature,
            non_negative_mask=non_negative_mask,
            hard_bank_k=hard_bank_k,
            bank_fn_margin=bank_fn_margin,
            bank_random_k=bank_random_k,
            query_topk=query_topk,
            score_center=score_center,
            gap_weight=gap_weight,
            gap_margin=gap_margin,
            text_image_ids=text_image_ids,
        )
    return loss / total_weight


def combine_full_and_matryoshka(
    full: torch.Tensor,
    matryoshka: torch.Tensor,
    matryoshka_weight: float,
    *,
    has_prefixes: bool = True,
) -> torch.Tensor:
    """Blend full-dim and prefix CE for one retrieval task.

    Gradients from both terms flow into the *same* raw embeddings (prefixes via
    truncate→renormalize). This is error addition at the embedding outputs, not
    a separate multi-task optimizer. ``matryoshka_weight`` scales prefix CE
    relative to full; result is normalized so default weight 1.0 → equal share.
    """
    w = float(matryoshka_weight)
    if w <= 0.0 or not has_prefixes:
        return full
    return (full + w * matryoshka) / (1.0 + w)


def mean_task_losses(
    task_losses: list[tuple[torch.Tensor, float]],
) -> torch.Tensor:
    """Weighted mean of task CEs (default weights 1 → equal error at embeddings).

    ``total = sum_i w_i * L_i / sum_i w_i``. Autograd yields
    ``∂total/∂E = sum_i (w_i/Z) ∂L_i/∂E`` — additive task errors on shared
    embeddings, not independent optimizers.
    """
    active = [(loss, float(w)) for loss, w in task_losses if float(w) > 0.0]
    if not active:
        raise ValueError("mean_task_losses requires at least one positive-weight task")
    z = sum(w for _, w in active)
    acc = active[0][0].new_zeros(())
    for loss, w in active:
        acc = acc + (w / z) * loss
    return acc


class Stage1AlignmentModel(nn.Module):
    """Joint vision + text embedders with trainable Matryoshka projection heads."""

    def __init__(
        self,
        vision_model: nn.Module,
        text_model: nn.Module,
        vision_hidden: int,
        text_hidden: int,
        vision_device: torch.device,
        text_device: torch.device,
        embed_dim: int = EMBED_DIM,
        matryoshka_dims: tuple[int, ...] = DEFAULT_MATRYOSHKA_DIMS,
        temperature: float = 0.07,
        contrastive_weight: float = 1.0,
        matryoshka_weight: float = 1.0,
        text_text_weight: float = 1.0,
        text_text_matryoshka_weight: float | None = None,
        query_image_weight: float = 1.0,
        hard_bank_negatives: int = DEFAULT_HARD_BANK_NEGATIVES,
        bank_fn_margin: float | None = DEFAULT_BANK_FN_MARGIN,
        bank_random_k: int = DEFAULT_BANK_RANDOM_K,
        compute_dtype: torch.dtype = torch.float16,
        memory_bank_size: int = DEFAULT_MEMORY_BANK_SIZE,
        soft_maxsim: bool = True,
        soft_maxsim_temperature: float = DEFAULT_SOFT_MAXSIM_TEMPERATURE,
        multi_positive_jaccard: float = DEFAULT_MULTI_POSITIVE_JACCARD,
        vision_patch_keep_ratio: float = DEFAULT_VISION_PATCH_KEEP_RATIO,
        vision_patch_drop_prob: float = DEFAULT_VISION_PATCH_DROP_PROB,
        vision_merge_tokens: int = DEFAULT_VISION_MERGE_TOKENS,
        vision_merge_assign_temperature: float = DEFAULT_VISION_MERGE_ASSIGN_TEMPERATURE,
        score_center: bool = DEFAULT_SCORE_CENTER,
        query_maxsim_topk: int = DEFAULT_QUERY_MAXSIM_TOPK,
        gap_loss_weight: float = DEFAULT_GAP_LOSS_WEIGHT,
        gap_margin: float = DEFAULT_GAP_MARGIN,
        image_shift_max: int = DEFAULT_IMAGE_SHIFT_MAX,
        image_hflip_prob: float = DEFAULT_IMAGE_HFLIP_PROB,
        image_max_rotate_deg: float = DEFAULT_IMAGE_MAX_ROTATE_DEG,
        image_scale_min: float = DEFAULT_IMAGE_SCALE_MIN,
        image_scale_max: float = DEFAULT_IMAGE_SCALE_MAX,
        image_fill_mode: str = DEFAULT_IMAGE_FILL_MODE,
        image_aug_enabled: bool = True,
        photometric_enabled: bool = DEFAULT_PHOTOMETRIC_ENABLED,
        photo_brightness: float = DEFAULT_PHOTO_BRIGHTNESS,
        photo_contrast: float = DEFAULT_PHOTO_CONTRAST,
        photo_saturation: float = DEFAULT_PHOTO_SATURATION,
        photo_hue: float = DEFAULT_PHOTO_HUE,
        spatial_brightness: float = DEFAULT_SPATIAL_BRIGHTNESS,
        spatial_color: float = DEFAULT_SPATIAL_COLOR,
        spatial_noise_grid: int = DEFAULT_SPATIAL_NOISE_GRID,
        grayscale_prob: float = DEFAULT_GRAYSCALE_PROB,
        image_mean: tuple[float, ...] | list[float] | None = None,
        image_std: tuple[float, ...] | list[float] | None = None,
        heatmap_sparsity_weight: float = DEFAULT_HEATMAP_SPARSITY_WEIGHT,
        heatmap_sparsity_temperature: float = DEFAULT_HEATMAP_SPARSITY_TEMPERATURE,
        heatmap_sparsity_square: bool = DEFAULT_HEATMAP_SPARSITY_SQUARE,
        bank_score_policy: str = "accum_window",
        embedding_geo_weight: float = DEFAULT_EMBEDDING_GEO_WEIGHT,
        geo_center_weight: float = DEFAULT_GEO_CENTER_WEIGHT,
        geo_var_weight: float = DEFAULT_GEO_VAR_WEIGHT,
        geo_vec_mean_weight: float = DEFAULT_GEO_VEC_MEAN_WEIGHT,
        geo_var_ratio: float = DEFAULT_GEO_VAR_RATIO,
        geo_mag_floor: float = DEFAULT_GEO_MAG_FLOOR,
        geo_mag_floor_weight: float = DEFAULT_GEO_MAG_FLOOR_WEIGHT,
        geo_max_abs_ratio: float = DEFAULT_GEO_MAX_ABS_RATIO,
        geo_max_abs_weight: float = DEFAULT_GEO_MAX_ABS_WEIGHT,
        geo_uniformity_weight: float = DEFAULT_GEO_UNIFORMITY_WEIGHT,
        geo_uniformity_t: float = DEFAULT_GEO_UNIFORMITY_T,
        geo_uniformity_max_samples: int = DEFAULT_GEO_UNIFORMITY_MAX_SAMPLES,
        geo_token_weight: float = DEFAULT_GEO_TOKEN_WEIGHT,
        geo_pool_weight: float = DEFAULT_GEO_POOL_WEIGHT,
        geo_prefix_dim: int = DEFAULT_GEO_PREFIX_DIM,
        geo_prefix_weight: float = DEFAULT_GEO_PREFIX_WEIGHT,
        geo_ema_momentum: float = DEFAULT_GEO_EMA_MOMENTUM,
        geo_after_unfreeze: bool = DEFAULT_GEO_AFTER_UNFREEZE,
        geo_square: bool = DEFAULT_GEO_SQUARE,
    ):
        super().__init__()
        self.vision_device = vision_device
        self.text_device = text_device
        self.loss_device = vision_device
        self.compute_dtype = compute_dtype

        self.vision_model = vision_model
        self.text_model = text_model
        # bias=False: L2-normalized embeddings; a free bias mainly adds a shared
        # directional mode (cone) that fights contrastive + geometry losses.
        self.vision_projection = nn.Linear(
            vision_hidden,
            embed_dim,
            bias=False,
            device=vision_device,
            dtype=compute_dtype,
        )
        self.text_projection = nn.Linear(
            text_hidden,
            embed_dim,
            bias=False,
            device=text_device,
            dtype=compute_dtype,
        )
        self.embed_dim = int(embed_dim)
        # Prefix-only dims (full dim trained by main contrastive terms).
        self.matryoshka_dims = matryoshka_prefix_dims(matryoshka_dims, embed_dim=embed_dim)
        self.temperature = temperature
        # Relative task weights in mean_task_losses (default 1 → equal embed errors).
        self.contrastive_weight = float(contrastive_weight)
        self.matryoshka_weight = float(matryoshka_weight)
        self.text_text_weight = float(text_text_weight)
        # Legacy alias: if set, kept for logging only; MRL uses matryoshka_weight.
        self.text_text_matryoshka_weight = (
            float(text_text_matryoshka_weight)
            if text_text_matryoshka_weight is not None
            else float(matryoshka_weight)
        )
        self.query_image_weight = float(query_image_weight)
        self.hard_bank_negatives = int(hard_bank_negatives)
        # None disables FN filter; 0.0 filters bank cols with score >= pos.
        self.bank_fn_margin = (
            None if bank_fn_margin is None else float(bank_fn_margin)
        )
        self.bank_random_k = int(bank_random_k)
        self.memory_bank = EmbeddingMemoryBank(memory_bank_size)
        self.soft_maxsim = bool(soft_maxsim)
        self.soft_maxsim_temperature = float(soft_maxsim_temperature)
        self.multi_positive_jaccard = float(multi_positive_jaccard)
        self.vision_patch_keep_ratio = float(vision_patch_keep_ratio)
        self.vision_patch_drop_prob = float(vision_patch_drop_prob)
        self.vision_merge_tokens = int(vision_merge_tokens)
        self.vision_merge_assign_temperature = float(vision_merge_assign_temperature)
        self.score_center = bool(score_center)
        self.query_maxsim_topk = int(query_maxsim_topk)
        self.gap_loss_weight = float(gap_loss_weight)
        self.gap_margin = float(gap_margin)
        self.image_shift_max = int(image_shift_max)
        self.image_hflip_prob = float(image_hflip_prob)
        self.image_max_rotate_deg = float(image_max_rotate_deg)
        self.image_scale_min = float(image_scale_min)
        self.image_scale_max = float(image_scale_max)
        fill_mode = str(image_fill_mode or DEFAULT_IMAGE_FILL_MODE).lower()
        if fill_mode not in ("random", "mean", "reflect"):
            raise ValueError(
                f"image_fill_mode must be random|mean|reflect, got {image_fill_mode!r}"
            )
        self.image_fill_mode = fill_mode
        self.image_aug_enabled = bool(image_aug_enabled)
        self.photometric_enabled = bool(photometric_enabled)
        self.photo_brightness = float(photo_brightness)
        self.photo_contrast = float(photo_contrast)
        self.photo_saturation = float(photo_saturation)
        self.photo_hue = float(photo_hue)
        self.spatial_brightness = float(spatial_brightness)
        self.spatial_color = float(spatial_color)
        self.spatial_noise_grid = int(spatial_noise_grid)
        self.grayscale_prob = float(grayscale_prob)
        self.image_mean = (
            tuple(float(x) for x in image_mean)
            if image_mean is not None
            else DEFAULT_IMAGE_MEAN
        )
        self.image_std = (
            tuple(float(x) for x in image_std)
            if image_std is not None
            else DEFAULT_IMAGE_STD
        )
        self.heatmap_sparsity_weight = float(heatmap_sparsity_weight)
        self.heatmap_sparsity_temperature = float(heatmap_sparsity_temperature)
        self.heatmap_sparsity_square = bool(heatmap_sparsity_square)
        # "accum_window" = policy B: score snapshot from window start, enqueue every mb.
        # "live" = score against bank after each prior micro-batch enqueue.
        if bank_score_policy not in ("accum_window", "live"):
            raise ValueError(
                f"bank_score_policy must be 'accum_window' or 'live', "
                f"got {bank_score_policy!r}"
            )
        self.bank_score_policy = bank_score_policy
        self._score_bank_snapshot: dict[str, list[torch.Tensor]] | None = None

        # Geometry / anti-cone regularizer (enabled when embedding_geo_weight > 0).
        self.embedding_geo_weight = float(embedding_geo_weight)
        self.geo_center_weight = float(geo_center_weight)
        self.geo_var_weight = float(geo_var_weight)
        self.geo_vec_mean_weight = float(geo_vec_mean_weight)
        self.geo_var_ratio = float(geo_var_ratio)
        self.geo_mag_floor = float(geo_mag_floor)
        self.geo_mag_floor_weight = float(geo_mag_floor_weight)
        self.geo_max_abs_ratio = float(geo_max_abs_ratio)
        self.geo_max_abs_weight = float(geo_max_abs_weight)
        self.geo_uniformity_weight = float(geo_uniformity_weight)
        self.geo_uniformity_t = float(geo_uniformity_t)
        self.geo_uniformity_max_samples = int(geo_uniformity_max_samples)
        self.geo_token_weight = float(geo_token_weight)
        self.geo_pool_weight = float(geo_pool_weight)
        self.geo_prefix_dim = int(geo_prefix_dim)
        self.geo_prefix_weight = float(geo_prefix_weight)
        self.geo_ema_momentum = float(geo_ema_momentum)
        # Defer geo until backbone unfreeze: linear proj cannot break a backbone cone.
        self.geo_after_unfreeze = bool(geo_after_unfreeze)
        self.geo_active = not self.geo_after_unfreeze
        self.geo_square = bool(geo_square)
        # Running mean of normalized embeds (fp32); raw mean of unit vectors, ‖μ‖≤1.
        self.register_buffer(
            "_geo_ema_mean",
            torch.zeros(embed_dim, dtype=torch.float32, device=vision_device),
            persistent=False,
        )
        self.register_buffer(
            "_geo_ema_initialized",
            torch.zeros((), dtype=torch.bool, device=vision_device),
            persistent=False,
        )
        self._init_projection_heads()

    def set_geo_active(self, active: bool) -> None:
        """Enable/disable geometry loss (used to defer geo until backbone unfreeze)."""
        self.geo_active = bool(active)

    def _init_projection_heads(self):
        for proj in (self.vision_projection, self.text_projection):
            nn.init.xavier_uniform_(proj.weight)
            # bias=False by design; keep guard if someone re-enables bias later.
            if getattr(proj, "bias", None) is not None:
                nn.init.zeros_(proj.bias)

    def _soft_tau(self) -> float | None:
        if self.soft_maxsim and self.soft_maxsim_temperature > 0.0:
            return self.soft_maxsim_temperature
        return None

    def begin_accum_window(self) -> None:
        """Policy B: freeze bank contents used for scoring for this accum window.

        Live FIFO still receives enqueues every micro-batch; only the scoring
        view is snapshotted here.
        """
        bank = self.memory_bank
        if bank.enabled and self.bank_score_policy == "accum_window":
            self._score_bank_snapshot = bank.snapshot()
        else:
            self._score_bank_snapshot = None

    def _bank_raw_for_scoring(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        bank = self.memory_bank
        if not bank.enabled:
            return [], []
        if (
            self.bank_score_policy == "accum_window"
            and self._score_bank_snapshot is not None
        ):
            return (
                list(self._score_bank_snapshot["text_raw"]),
                list(self._score_bank_snapshot["image_raw"]),
            )
        return bank.text_raw(), bank.image_raw()

    def _text_backbone(self):
        return getattr(self.text_model, "model", self.text_model)

    def _to_loss(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.loss_device, dtype=self.compute_dtype)

    def _module_dtype(self, module: nn.Module) -> torch.dtype:
        """First floating parameter dtype of ``module``, else ``compute_dtype``."""
        for p in module.parameters():
            if p.is_floating_point():
                return p.dtype
        return self.compute_dtype

    def _select_vision_patches(
        self, vision_raw_i: torch.Tensor
    ) -> torch.Tensor:
        """L2 keep → train drop → similarity-merge to few centroids for MaxSim."""
        kept = keep_top_patches_by_l2(
            vision_raw_i, keep_ratio=self.vision_patch_keep_ratio
        )
        kept = random_drop_patches(
            kept,
            drop_prob=self.vision_patch_drop_prob,
            training=self.training,
        )
        if self.vision_merge_tokens > 0 and kept.shape[0] > self.vision_merge_tokens:
            kept = merge_tokens_by_similarity(
                kept,
                k=self.vision_merge_tokens,
                assign_temperature=self.vision_merge_assign_temperature,
            )
        return kept

    def _geo_kwargs(self) -> dict[str, float | int]:
        return {
            "center_weight": self.geo_center_weight,
            "var_weight": self.geo_var_weight,
            "vec_mean_weight": self.geo_vec_mean_weight,
            "var_ratio": self.geo_var_ratio,
            "mag_floor": self.geo_mag_floor,
            "mag_floor_weight": self.geo_mag_floor_weight,
            "max_abs_ratio": self.geo_max_abs_ratio,
            "max_abs_weight": self.geo_max_abs_weight,
            "uniformity_weight": self.geo_uniformity_weight,
            "uniformity_t": self.geo_uniformity_t,
            "uniformity_max_samples": self.geo_uniformity_max_samples,
        }

    def _empty_geo_metrics(self) -> dict[str, float]:
        return {
            "geo_center": 0.0,
            "geo_var": 0.0,
            "geo_vec_mean": 0.0,
            "geo_uniformity": 0.0,
            "geo_mag_floor": 0.0,
            "geo_max_abs": 0.0,
            "geo_mu_norm": 0.0,
            "geo_min_std": 0.0,
            "geo_mean_abs_mu": 0.0,
            "geo_loss": 0.0,
            "geo_raw": 0.0,
            "geo_token_mu_norm": 0.0,
            "geo_pool_mu_norm": 0.0,
        }

    def _geometry_on_matrix(
        self,
        norm_mat: torch.Tensor,
        *,
        raw_mat: torch.Tensor | None,
        ema: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Full-dim + optional Matryoshka-prefix geometry on one unit-vector matrix."""
        geo_full, metrics = embedding_geometry_loss(
            norm_mat,
            raw=raw_mat,
            ema_mean=ema,
            **self._geo_kwargs(),
        )
        geo = geo_full
        prefix_dim = self.geo_prefix_dim
        if (
            self.geo_prefix_weight > 0.0
            and prefix_dim > 0
            and prefix_dim < norm_mat.shape[-1]
        ):
            prefix = matryoshka_normalize(norm_mat, dim=prefix_dim)
            raw_prefix = None
            if raw_mat is not None and raw_mat.shape[-1] >= prefix_dim:
                # Slice pre-norm coords; do not treat as already unit in prefix space.
                raw_prefix = raw_mat[..., :prefix_dim]
            ema_prefix = None
            if ema is not None and ema.numel() >= prefix_dim:
                # Raw mean slice — NOT unit-normalized (unit EMA permanently
                # floors center at ~0.25 even for isotropic batches).
                ema_prefix = ema[:prefix_dim].float()
            geo_pref, pref_metrics = embedding_geometry_loss(
                prefix,
                raw=raw_prefix,
                ema_mean=ema_prefix,
                **self._geo_kwargs(),
            )
            geo = geo + float(self.geo_prefix_weight) * geo_pref
            metrics = {
                **metrics,
                "geo_prefix_center": pref_metrics["geo_center"],
                "geo_prefix_mu_norm": pref_metrics["geo_mu_norm"],
            }
        return geo, metrics

    def _compute_embedding_geometry(
        self,
        *,
        norm_token_lists: list[list[torch.Tensor]],
        raw_token_lists: list[list[torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Geometry loss on tokens and L2-renormed sequence means (equal share).

        Token rows and sequence-mean rows used to be concatenated, so hundreds of
        patches drowned ~B pooled vectors and unnormalized pools violated the
        unit-sphere assumption. We now:
          1. L2-renormalize mean-pooled sequences.
          2. Score token-level and pool-level geo separately.
          3. Average with ``geo_token_weight`` / ``geo_pool_weight`` (default 0.5/0.5).
        """
        zero = self.vision_projection.weight.new_zeros(())
        empty = self._empty_geo_metrics()
        if self.embedding_geo_weight <= 0.0 or not self.geo_active:
            return zero, empty

        token_norm = stack_token_embeddings(norm_token_lists)
        token_raw = stack_token_embeddings(raw_token_lists)

        pooled_parts: list[torch.Tensor] = []
        for lst in norm_token_lists:
            if not lst:
                continue
            p = mean_pool_token_list(lst)
            if p is None or p.numel() == 0:
                continue
            # Critical: mean of unit tokens is NOT unit — renorm for sphere geo.
            pooled_parts.append(F.normalize(p.float(), dim=-1).to(dtype=p.dtype))
        pooled = torch.cat(pooled_parts, dim=0) if pooled_parts else None

        if token_norm is None and pooled is None:
            return zero, empty

        ema = None
        if self._geo_ema_initialized.item():
            device = (
                token_norm.device if token_norm is not None else pooled.device  # type: ignore[union-attr]
            )
            ema = self._geo_ema_mean.to(device=device)

        branch_losses: list[torch.Tensor] = []
        branch_weights: list[float] = []
        token_metrics: dict[str, float] | None = None
        pool_metrics: dict[str, float] | None = None

        if token_norm is not None and token_norm.shape[0] > 0 and self.geo_token_weight > 0.0:
            raw_mat = None
            if token_raw is not None and token_raw.shape[0] == token_norm.shape[0]:
                raw_mat = token_raw
            g_tok, token_metrics = self._geometry_on_matrix(
                token_norm, raw_mat=raw_mat, ema=ema
            )
            branch_losses.append(g_tok)
            branch_weights.append(float(self.geo_token_weight))

        if (
            pooled is not None
            and pooled.shape[0] >= 2
            and self.geo_pool_weight > 0.0
        ):
            g_pool, pool_metrics = self._geometry_on_matrix(
                pooled, raw_mat=None, ema=ema
            )
            branch_losses.append(g_pool)
            branch_weights.append(float(self.geo_pool_weight))
        elif pooled is not None and pooled.shape[0] == 1 and pool_metrics is None:
            # Single sequence: still report pool μ for logging, no var/unif.
            with torch.no_grad():
                pool_metrics = {
                    "geo_mu_norm": float(pooled.float().mean(dim=0).norm()),
                    "geo_min_std": 0.0,
                    "geo_center": 0.0,
                    "geo_var": 0.0,
                    "geo_uniformity": 0.0,
                    "geo_mean_abs_mu": float(pooled.float().mean(dim=0).abs().mean()),
                }

        if not branch_losses:
            return zero, empty

        w_sum = sum(branch_weights)
        geo_raw = sum(w * L for w, L in zip(branch_weights, branch_losses)) / max(
            w_sum, 1e-8
        )
        # Non-negative badness (all terms ≥ 0). Optional square: small badness is
        # soft (retrieval can lead); large coning is quadratically expensive.
        if self.geo_square:
            geo = geo_raw * geo_raw
        else:
            geo = geo_raw

        # Prefer sample-level (pool) metrics for the headline μ / minσ logs.
        primary = pool_metrics if pool_metrics is not None else token_metrics
        assert primary is not None
        metrics: dict[str, float] = {
            **empty,
            **primary,
            "geo_raw": float(geo_raw.detach()),
            "geo_loss": float(geo.detach()),
        }
        if token_metrics is not None:
            metrics["geo_token_mu_norm"] = float(token_metrics.get("geo_mu_norm", 0.0))
            metrics["geo_token_min_std"] = float(token_metrics.get("geo_min_std", 0.0))
            metrics["geo_token_uniformity"] = float(
                token_metrics.get("geo_uniformity", 0.0)
            )
        if pool_metrics is not None:
            metrics["geo_pool_mu_norm"] = float(pool_metrics.get("geo_mu_norm", 0.0))
            metrics["geo_pool_min_std"] = float(pool_metrics.get("geo_min_std", 0.0))
            metrics["geo_pool_uniformity"] = float(
                pool_metrics.get("geo_uniformity", 0.0)
            )
            # Headline = pool (retrieval-relevant sequence means).
            metrics["geo_mu_norm"] = metrics["geo_pool_mu_norm"]
            metrics["geo_min_std"] = float(pool_metrics.get("geo_min_std", 0.0))
            metrics["geo_uniformity"] = float(pool_metrics.get("geo_uniformity", 0.0))
            metrics["geo_center"] = float(pool_metrics.get("geo_center", 0.0))
            metrics["geo_var"] = float(pool_metrics.get("geo_var", 0.0))

        # EMA: equal blend of token mean and pool mean when both exist (no row-count dilution).
        with torch.no_grad():
            mean_parts: list[torch.Tensor] = []
            if token_norm is not None:
                mean_parts.append(token_norm.detach().float().mean(dim=0))
            if pooled is not None:
                mean_parts.append(pooled.detach().float().mean(dim=0))
            batch_mu = torch.stack(mean_parts, dim=0).mean(dim=0)
            if batch_mu.device != self._geo_ema_mean.device:
                batch_mu = batch_mu.to(self._geo_ema_mean.device)
            if not self._geo_ema_initialized.item():
                self._geo_ema_mean.copy_(batch_mu)
                self._geo_ema_initialized.fill_(True)
            else:
                update_embedding_ema(
                    self._geo_ema_mean, batch_mu, momentum=self.geo_ema_momentum
                )

        return geo, metrics

    def _maybe_augment_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Train-only geometric + photometric + shift; identity in eval."""
        if not self.training or not self.image_aug_enabled:
            return pixel_values
        return apply_train_image_augmentations(
            pixel_values,
            hflip_prob=self.image_hflip_prob,
            max_rotate_deg=self.image_max_rotate_deg,
            scale_min=self.image_scale_min,
            scale_max=self.image_scale_max,
            fill_mode=self.image_fill_mode,
            max_shift=self.image_shift_max,
            photometric=self.photometric_enabled,
            photo_brightness=self.photo_brightness,
            photo_contrast=self.photo_contrast,
            photo_saturation=self.photo_saturation,
            photo_hue=self.photo_hue,
            spatial_brightness=self.spatial_brightness,
            spatial_color=self.spatial_color,
            spatial_noise_grid=self.spatial_noise_grid,
            grayscale_prob=self.grayscale_prob,
            image_mean=self.image_mean,
            image_std=self.image_std,
            enabled=True,
        )

    def encode_images(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        vdtype = self._module_dtype(self.vision_model)
        pixel_values = pixel_values.to(
            device=self.vision_device, dtype=vdtype, non_blocking=True
        )
        pixel_values = self._maybe_augment_images(pixel_values)
        vision_hidden = self.vision_model(
            pixel_values=pixel_values
        ).last_hidden_state.to(dtype=self.compute_dtype)
        vision_raw = self.vision_projection(
            vision_hidden.to(self.vision_projection.weight.dtype)
        )
        out: list[torch.Tensor] = []
        for i in range(vision_raw.size(0)):
            kept = self._select_vision_patches(vision_raw[i])
            out.append(matryoshka_normalize(kept))
        return out

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        input_ids = input_ids.to(self.text_device, non_blocking=True)
        attention_mask = attention_mask.to(self.text_device, non_blocking=True)
        text_hidden = self._text_backbone()(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.to(dtype=self.compute_dtype)
        text_raw = self.text_projection(
            text_hidden.to(self.text_projection.weight.dtype)
        )
        tokens: list[torch.Tensor] = []
        for i in range(text_raw.size(0)):
            mask = attention_mask[i].bool()
            tokens.append(matryoshka_normalize(text_raw[i, mask]))
        return tokens

    def _encode_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        input_ids = input_ids.to(self.text_device, non_blocking=True)
        attention_mask = attention_mask.to(self.text_device, non_blocking=True)
        text_hidden = self._text_backbone()(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.to(dtype=self.compute_dtype)
        text_raw = self.text_projection(
            text_hidden.to(self.text_projection.weight.dtype)
        )
        tokens: list[torch.Tensor] = []
        raw_masked: list[torch.Tensor] = []
        for i in range(text_raw.size(0)):
            mask = attention_mask[i].bool()
            raw_masked.append(text_raw[i, mask])
            tokens.append(matryoshka_normalize(text_raw[i, mask]))
        return tokens, raw_masked

    def encode_text_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Project text ids → (normalized tokens list, raw masked list)."""
        input_ids = input_ids.to(self.text_device, non_blocking=True)
        attention_mask = attention_mask.to(self.text_device, non_blocking=True)
        text_hidden = self._text_backbone()(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.to(dtype=self.compute_dtype)
        text_raw = self.text_projection(
            text_hidden.to(self.text_projection.weight.dtype)
        )
        tokens: list[torch.Tensor] = []
        raw_masked: list[torch.Tensor] = []
        for i in range(text_raw.size(0)):
            mask = attention_mask[i].bool()
            raw_masked.append(text_raw[i, mask])
            tokens.append(matryoshka_normalize(text_raw[i, mask]))
        return tokens, raw_masked

    def compute_paraphrase_loss(
        self,
        anchor_input_ids: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        *,
        negative_input_ids: torch.Tensor | None = None,
        negative_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """AllNLI / paraphrase MaxSim InfoNCE (text tower only)."""
        anchor_tok, _ = self.encode_text_tokens(
            anchor_input_ids, anchor_attention_mask
        )
        pos_tok, _ = self.encode_text_tokens(
            positive_input_ids, positive_attention_mask
        )
        neg_tok: list[torch.Tensor] = []
        if negative_input_ids is not None and negative_attention_mask is not None:
            neg_tok, _ = self.encode_text_tokens(
                negative_input_ids, negative_attention_mask
            )
        bank = self.memory_bank
        score_text_raw, _ = self._bank_raw_for_scoring()
        bank_text = (
            [matryoshka_normalize(self._to_loss(t)) for t in score_text_raw]
            if score_text_raw
            else []
        )
        return paraphrase_contrastive_loss(
            [self._to_loss(t) for t in anchor_tok],
            [self._to_loss(t) for t in pos_tok],
            negative_tokens=[self._to_loss(t) for t in neg_tok] if neg_tok else None,
            temperature=self.temperature,
            bank_doc_tokens=bank_text,
            soft_maxsim_temperature=self._soft_tau(),
            hard_bank_k=self.hard_bank_negatives,
            bank_fn_margin=self.bank_fn_margin,
            bank_random_k=self.bank_random_k,
            query_topk=self.query_maxsim_topk,
            score_center=self.score_center,
            gap_weight=self.gap_loss_weight,
            gap_margin=self.gap_margin,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        query_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        unrelated_input_ids: torch.Tensor | None = None,
        unrelated_attention_mask: torch.Tensor | None = None,
        captions: list[str] | None = None,
        related_queries: list[str] | None = None,
        all_input_ids: torch.Tensor | None = None,
        all_attention_mask: torch.Tensor | None = None,
        text_image_ids: torch.Tensor | None = None,
        all_positive_texts: list[str] | None = None,
        return_loss: bool = True,
    ) -> dict[str, Any]:
        vdtype = self._module_dtype(self.vision_model)
        pixel_values = pixel_values.to(
            device=self.vision_device, dtype=vdtype, non_blocking=True
        )
        input_ids = input_ids.to(self.text_device, non_blocking=True)
        attention_mask = attention_mask.to(self.text_device, non_blocking=True)

        # Train-only: flip / rotate / mild scale-stretch + pad fill + pixel shift.
        pixel_values = self._maybe_augment_images(pixel_values)
        # Augmentations can promote to float32; cast back to vision weight dtype.
        pixel_values = pixel_values.to(dtype=vdtype)

        vision_hidden = self.vision_model(
            pixel_values=pixel_values
        ).last_hidden_state.to(dtype=self.compute_dtype)
        vision_raw = self.vision_projection(
            vision_hidden.to(self.vision_projection.weight.dtype)
        )

        # Multi-text positives: encode all captions/queries for images when present.
        use_all_texts = (
            all_input_ids is not None
            and all_attention_mask is not None
            and text_image_ids is not None
            and all_input_ids.shape[0] >= input_ids.shape[0]
        )
        if use_all_texts:
            assert all_input_ids is not None and all_attention_mask is not None
            text_tokens, text_raw_masked = self.encode_text_tokens(
                all_input_ids, all_attention_mask
            )
            ids_map = text_image_ids.to(device=self.text_device, dtype=torch.long).view(-1)
        else:
            text_tokens, text_raw_masked = self.encode_text_tokens(
                input_ids, attention_mask
            )
            ids_map = torch.arange(
                input_ids.shape[0], device=self.text_device, dtype=torch.long
            )

        # Primary caption tokens (1 per image) for bank + query→caption docs.
        n_images = vision_raw.size(0)
        primary_idx = torch.full(
            (n_images,), -1, device=ids_map.device, dtype=torch.long
        )
        for t_i, img_i in enumerate(ids_map.tolist()):
            if 0 <= img_i < n_images and primary_idx[img_i] < 0:
                primary_idx[img_i] = t_i
        if bool((primary_idx < 0).any()):
            # Fall back: encode primary input_ids if multipos map incomplete.
            prim_tok, prim_raw = self.encode_text_tokens(input_ids, attention_mask)
            primary_text_tokens = prim_tok
            primary_text_raw = prim_raw
        else:
            primary_text_tokens = [text_tokens[int(i)] for i in primary_idx.tolist()]
            primary_text_raw = [text_raw_masked[int(i)] for i in primary_idx.tolist()]

        # Vision: drop background patches (pre-norm L2) before normalize / bank.
        image_raw_kept: list[torch.Tensor] = []
        image_tokens: list[torch.Tensor] = []
        for i in range(vision_raw.size(0)):
            kept = self._select_vision_patches(vision_raw[i])
            image_raw_kept.append(kept)
            image_tokens.append(matryoshka_normalize(kept))

        if not return_loss:
            return {
                "text_embeddings": primary_text_tokens,
                "image_embeddings": image_tokens,
            }

        loss_text_tokens = [self._to_loss(t) for t in text_tokens]
        loss_image_tokens = [self._to_loss(t) for t in image_tokens]
        loss_text_raw = [self._to_loss(t) for t in text_raw_masked]
        loss_image_raw = [self._to_loss(t) for t in image_raw_kept]
        loss_primary_text_tokens = [self._to_loss(t) for t in primary_text_tokens]
        loss_primary_text_raw = [self._to_loss(t) for t in primary_text_raw]

        bank = self.memory_bank
        score_text_raw, score_image_raw = self._bank_raw_for_scoring()
        bank_text_raw = (
            [self._to_loss(t) for t in score_text_raw] if score_text_raw else []
        )
        bank_image_raw = (
            [self._to_loss(t) for t in score_image_raw] if score_image_raw else []
        )
        bank_text_tokens = (
            [matryoshka_normalize(t) for t in bank_text_raw] if bank_text_raw else []
        )
        bank_image_tokens = (
            [matryoshka_normalize(t) for t in bank_image_raw] if bank_image_raw else []
        )

        soft_tau = self._soft_tau()
        hard_k = self.hard_bank_negatives
        bank_fn_m = self.bank_fn_margin
        bank_rand_k = self.bank_random_k
        # Task-specific multi-positive masks (false-neg softening):
        #   cap↔img / q→cap  → caption token-Jaccard
        #   query↔image      → related_query token-Jaccard (not caption text)
        # Using caption Jaccard on query tasks wrongly softens pairs whose
        # search queries are dissimilar (different intents, similar captions).
        non_neg_cap = build_multi_positive_mask(
            captions,
            batch_size=n_images,
            jaccard_threshold=self.multi_positive_jaccard,
            device=loss_image_tokens[0].device,
        )
        non_neg_query = build_multi_positive_mask(
            related_queries,
            batch_size=n_images,
            jaccard_threshold=self.multi_positive_jaccard,
            device=loss_image_tokens[0].device,
        )
        # Fallback: if queries missing, do not reuse caption mask for q↔img.
        # (None → standard single-positive InfoNCE on that task.)

        # ------------------------------------------------------------------
        # Retrieval tasks (all InfoNCE on shared live embeddings):
        #   1) caption(s) ↔ image  — all positive texts match their image
        #   2) query  ↔ image   — query finds the matching image
        #   3) query  → caption — query finds the matching caption
        # Within each task, full-dim CE + prefix Matryoshka CE both backprop into
        # the *same* raw projected tokens (error addition at embedding outputs).
        # Tasks are then mean-combined (default equal weights) so
        #   ∂L/∂E = mean_i ∂L_i/∂E
        # — still one backward, not a weighted soup of unrelated scales.
        # Bank vectors are detached (hard top-k mining selects which bank cols
        # enter the softmax).
        # ------------------------------------------------------------------
        q_topk = self.query_maxsim_topk
        score_center = self.score_center
        gap_w = self.gap_loss_weight
        gap_m = self.gap_margin
        ids_for_loss = ids_map.to(device=loss_image_tokens[0].device)
        contrastive, contrastive_metrics = contrastive_late_interaction_loss(
            loss_text_tokens,
            loss_image_tokens,
            temperature=self.temperature,
            bank_text_tokens=bank_text_tokens,
            bank_image_tokens=bank_image_tokens,
            soft_maxsim_temperature=soft_tau,
            non_negative_mask=non_neg_cap,
            hard_bank_k=hard_k,
            bank_fn_margin=bank_fn_m,
            bank_random_k=bank_rand_k,
            query_topk=q_topk,
            score_center=score_center,
            gap_weight=gap_w,
            gap_margin=gap_m,
            return_metrics=True,
            text_image_ids=ids_for_loss,
        )
        matryoshka = matryoshka_loss(
            loss_text_raw,
            loss_image_raw,
            dims=self.matryoshka_dims,
            temperature=self.temperature,
            bank_text_raw=bank_text_raw,
            bank_image_raw=bank_image_raw,
            soft_maxsim_temperature=soft_tau,
            non_negative_mask=non_neg_cap,
            hard_bank_k=hard_k,
            bank_fn_margin=bank_fn_m,
            bank_random_k=bank_rand_k,
            query_topk=q_topk,
            score_center=score_center,
            gap_weight=gap_w,
            gap_margin=gap_m,
            embed_dim=self.embed_dim,
            text_image_ids=ids_for_loss,
        )
        has_prefixes = bool(self.matryoshka_dims)
        caption_image_task = combine_full_and_matryoshka(
            contrastive,
            matryoshka,
            self.matryoshka_weight,
            has_prefixes=has_prefixes,
        )

        zero = contrastive.new_zeros(())
        text_text = zero
        text_text_matryoshka = zero
        query_image = zero
        query_image_matryoshka = zero
        loss_query_tokens: list[torch.Tensor] = []
        loss_query_raw: list[torch.Tensor] = []
        loss_distractor_tokens: list[torch.Tensor] = []
        loss_distractor_raw: list[torch.Tensor] = []

        has_queries = (
            query_input_ids is not None
            and query_attention_mask is not None
            and unrelated_input_ids is not None
            and unrelated_attention_mask is not None
        )
        want_query_image = has_queries and self.query_image_weight > 0.0
        want_query_caption = has_queries and self.text_text_weight > 0.0

        if has_queries and (want_query_image or want_query_caption):
            query_tokens, query_raw = self._encode_text_batch(
                query_input_ids, query_attention_mask
            )
            distractor_tokens, distractor_raw = self._encode_text_batch(
                unrelated_input_ids, unrelated_attention_mask
            )
            loss_query_tokens = [self._to_loss(t) for t in query_tokens]
            loss_distractor_tokens = [self._to_loss(t) for t in distractor_tokens]
            loss_query_raw = [self._to_loss(t) for t in query_raw]
            loss_distractor_raw = [self._to_loss(t) for t in distractor_raw]

            if want_query_image:
                # Bidirectional query↔image: query finds matching image (and reverse).
                # Multi-pos from related_queries (search intent), not captions.
                query_image = contrastive_late_interaction_loss(
                    loss_query_tokens,
                    loss_image_tokens,
                    temperature=self.temperature,
                    bank_text_tokens=bank_text_tokens,
                    bank_image_tokens=bank_image_tokens,
                    soft_maxsim_temperature=soft_tau,
                    non_negative_mask=non_neg_query,
                    hard_bank_k=hard_k,
                    bank_fn_margin=bank_fn_m,
                    bank_random_k=bank_rand_k,
                    query_topk=q_topk,
                    score_center=score_center,
                    gap_weight=gap_w,
                    gap_margin=gap_m,
                    return_metrics=False,
                )
                query_image_matryoshka = matryoshka_loss(
                    loss_query_raw,
                    loss_image_raw,
                    dims=self.matryoshka_dims,
                    temperature=self.temperature,
                    bank_text_raw=bank_text_raw,
                    bank_image_raw=bank_image_raw,
                    soft_maxsim_temperature=soft_tau,
                    non_negative_mask=non_neg_query,
                    hard_bank_k=hard_k,
                    bank_fn_margin=bank_fn_m,
                    bank_random_k=bank_rand_k,
                    query_topk=q_topk,
                    score_center=score_center,
                    gap_weight=gap_w,
                    gap_margin=gap_m,
                    embed_dim=self.embed_dim,
                )

            if want_query_caption:
                # Query → matching primary caption; not other captions / distractors / bank.
                # Multi-pos softens near-duplicate *captions* (doc-side FNs).
                text_text = text_text_contrastive_loss(
                    loss_query_tokens,
                    loss_primary_text_tokens,
                    loss_distractor_tokens,
                    temperature=self.temperature,
                    bank_doc_tokens=bank_text_tokens,
                    soft_maxsim_temperature=soft_tau,
                    non_negative_mask=non_neg_cap,
                    hard_bank_k=hard_k,
                    bank_fn_margin=bank_fn_m,
                    bank_random_k=bank_rand_k,
                    query_topk=q_topk,
                    score_center=score_center,
                    gap_weight=gap_w,
                    gap_margin=gap_m,
                )
                text_text_matryoshka = text_text_matryoshka_loss(
                    loss_query_raw,
                    loss_primary_text_raw,
                    loss_distractor_raw,
                    dims=self.matryoshka_dims,
                    temperature=self.temperature,
                    bank_doc_raw=bank_text_raw,
                    soft_maxsim_temperature=soft_tau,
                    non_negative_mask=non_neg_cap,
                    hard_bank_k=hard_k,
                    bank_fn_margin=bank_fn_m,
                    bank_random_k=bank_rand_k,
                    query_topk=q_topk,
                    score_center=score_center,
                    gap_weight=gap_w,
                    gap_margin=gap_m,
                    embed_dim=self.embed_dim,
                )

        query_image_task = combine_full_and_matryoshka(
            query_image,
            query_image_matryoshka,
            self.matryoshka_weight,
            has_prefixes=has_prefixes,
        )
        query_caption_task = combine_full_and_matryoshka(
            text_text,
            text_text_matryoshka,
            self.matryoshka_weight,
            has_prefixes=has_prefixes,
        )

        task_losses: list[tuple[torch.Tensor, float]] = [
            (caption_image_task, self.contrastive_weight),
        ]
        if want_query_image:
            task_losses.append((query_image_task, self.query_image_weight))
        if want_query_caption:
            task_losses.append((query_caption_task, self.text_text_weight))

        retrieval_loss = mean_task_losses(task_losses)

        # Geometry / anti-cone: text + image (+ query when available).
        # Queries showed strong dim-0 bias (~0.05) in retrieval probes.
        geo_norm_lists: list[list[torch.Tensor]] = [
            loss_primary_text_tokens,
            loss_image_tokens,
        ]
        geo_raw_lists: list[list[torch.Tensor]] = [
            loss_primary_text_raw,
            loss_image_raw,
        ]
        if loss_query_tokens:
            geo_norm_lists.append(loss_query_tokens)
            geo_raw_lists.append(loss_query_raw)
        # Detached bank tokens enlarge N for variance/center without polluting
        # live gradients; we only use bank via EMA, which tracks historical means.

        geo_loss, geo_metrics = self._compute_embedding_geometry(
            norm_token_lists=geo_norm_lists,
            raw_token_lists=geo_raw_lists,
        )
        # Geo is a regularizer on the same embeddings (adds ∂geo/∂E), scaled
        # separately because its numeric scale is not InfoNCE-comparable.
        loss = retrieval_loss
        if self.embedding_geo_weight > 0.0:
            loss = loss + self.embedding_geo_weight * geo_loss

        # Heatmap sparsity: punish uniform patch MaxSim on positive pairs so
        # gradients favor a few “selected blocks” (cleaner demo heatmaps).
        # Raw value is normalized entropy ∈ [0,1]; optional square like geo.
        sparsity = zero
        sparsity_raw = zero
        if self.heatmap_sparsity_weight > 0.0:
            sp_terms: list[torch.Tensor] = [
                heatmap_sparsity_loss(
                    loss_primary_text_tokens,
                    loss_image_tokens,
                    temperature=self.heatmap_sparsity_temperature,
                )
            ]
            # Query heatmaps are what the demo shows — include when queries exist.
            if loss_query_tokens:
                sp_terms.append(
                    heatmap_sparsity_loss(
                        loss_query_tokens,
                        loss_image_tokens,
                        temperature=self.heatmap_sparsity_temperature,
                    )
                )
            sparsity_raw = torch.stack(sp_terms).mean()
            sparsity = (
                sparsity_raw * sparsity_raw
                if self.heatmap_sparsity_square
                else sparsity_raw
            )
            loss = loss + self.heatmap_sparsity_weight * sparsity

        # Enqueue *after* scoring so the current batch is never its own negative.
        # Policy B: enqueue every micro-batch into the live FIFO (primary texts).
        if bank.enabled:
            bank.enqueue(image_raw=loss_image_raw, text_raw=loss_primary_text_raw)

        return {
            "loss": loss,
            "retrieval_loss": retrieval_loss.detach(),
            "contrastive_loss": contrastive.detach(),
            "matryoshka_loss": matryoshka.detach(),
            "caption_image_loss": caption_image_task.detach(),
            "query_image_loss": query_image.detach(),
            "query_image_matryoshka_loss": query_image_matryoshka.detach(),
            "text_text_loss": text_text.detach(),
            "text_text_matryoshka_loss": text_text_matryoshka.detach(),
            "query_caption_loss": query_caption_task.detach(),
            # Log raw entropy [0,1] for readability; loss uses squared when enabled.
            "heatmap_sparsity_loss": sparsity_raw.detach(),
            "heatmap_sparsity_squared": sparsity.detach(),
            "geo_loss": geo_loss.detach()
            if torch.is_tensor(geo_loss)
            else contrastive.new_zeros(()),
            "geo_mu_norm": geo_metrics.get("geo_mu_norm", 0.0),
            "geo_min_std": geo_metrics.get("geo_min_std", 0.0),
            "geo_mean_abs_mu": geo_metrics.get("geo_mean_abs_mu", 0.0),
            "geo_center": geo_metrics.get("geo_center", 0.0),
            "geo_var": geo_metrics.get("geo_var", 0.0),
            "geo_uniformity": geo_metrics.get("geo_uniformity", 0.0),
            "geo_pool_mu_norm": geo_metrics.get("geo_pool_mu_norm", 0.0),
            "geo_token_mu_norm": geo_metrics.get("geo_token_mu_norm", 0.0),
            "memory_bank_size": len(bank),
            "pos_rank": contrastive_metrics["pos_rank"],
            "pos_rank_t2i": contrastive_metrics["pos_rank_t2i"],
            "pos_rank_i2t": contrastive_metrics["pos_rank_i2t"],
            "n_image_docs": contrastive_metrics["n_image_docs"],
            "n_text_docs": contrastive_metrics["n_text_docs"],
            "pos_score": contrastive_metrics.get("pos_score", float("nan")),
            "neg_score": contrastive_metrics.get("neg_score", float("nan")),
            "score_gap": contrastive_metrics.get("score_gap", float("nan")),
            "gap_hinge": contrastive_metrics.get("gap_hinge", 0.0),
        }


def resolve_training_dtype(bf16: bool) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def gpu_device(index: int, *, fallback: bool = True) -> torch.device:
    """Map a logical GPU index to a torch device.

    When ``fallback`` is True (default) and the requested index is out of range,
    clamp to the last available GPU instead of failing — useful on 1-GPU hosts
    where the CLI defaults are vision=0 / text=1.
    """
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if index >= n:
            if not fallback:
                raise ValueError(
                    f"Requested GPU {index}, but only {n} GPU(s) are available."
                )
            index = max(0, n - 1)
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def load_text_model_for_training(
    model_dir: str,
    tokenizer_id: str,
    max_seq_length: int,
    text_device: torch.device,
    compute_dtype: torch.dtype,
):
    from transformers import AutoTokenizer

    quantized = checkpoint_is_quantized(model_dir)
    init_dir = QWEN_DIR if quantized else model_dir
    dtype_name = str(compute_dtype).replace("torch.", "")
    print(
        f"Loading text model from {model_dir} on {text_device} "
        f"(full {dtype_name} via Unsloth full_finetuning"
        f"{'; dequant 8-bit ckpt, shell ' + init_dir if quantized else ''}"
        f"; AdamW8bit) ..."
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_qwen_backbone(
        model_dir,
        text_device,
        load_in_8bit=False,
        seed_dir=QWEN_DIR if quantized else None,
        tokenizer_id=tokenizer_id,
        max_seq_length=max_seq_length,
        for_training=True,
        compute_dtype=compute_dtype,
    )
    return model, tokenizer, model.config.hidden_size


def load_vision_model_for_training(
    model_dir: str,
    vision_device: torch.device,
    compute_dtype: torch.dtype,
):
    quantized = checkpoint_is_quantized(model_dir)
    init_dir = SIGLIP_DIR if quantized else model_dir
    dtype_name = str(compute_dtype).replace("torch.", "")
    print(
        f"Loading vision model from {model_dir} on {vision_device} "
        f"(full {dtype_name}"
        f"{'; dequant 8-bit ckpt, shell ' + init_dir if quantized else ''}"
        f"; AdamW8bit) ..."
    )
    model = load_siglip_backbone(
        model_dir,
        vision_device,
        load_in_8bit=False,
        seed_dir=SIGLIP_DIR if quantized else None,
        for_training=True,
        compute_dtype=compute_dtype,
    )
    return model, model.config.hidden_size


def _component_dir(checkpoint_root: Path, component: str) -> Path:
    return checkpoint_root / component


def _valid_config_path(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            json.load(fh)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def checkpoint_is_valid(root: Path) -> bool:
    if not root.is_dir():
        return False
    projection = root / PROJECTION_FILE
    vision_weights = _component_dir(root, VISION_COMPONENT) / "model.safetensors"
    text_weights = _component_dir(root, TEXT_COMPONENT) / "model.safetensors"
    vision_config = _component_dir(root, VISION_COMPONENT) / "config.json"
    text_config = _component_dir(root, TEXT_COMPONENT) / "config.json"
    return (
        projection.is_file()
        and vision_weights.is_file()
        and text_weights.is_file()
        and _valid_config_path(vision_config)
        and _valid_config_path(text_config)
    )


def _checkpoint_step_key(path: Path) -> tuple[int, float]:
    """Sort key: prefer higher step-N, then mtime. Stage root counts as step=10**12."""
    name = path.name
    mtime = (path / PROJECTION_FILE).stat().st_mtime if (path / PROJECTION_FILE).is_file() else path.stat().st_mtime
    if name.startswith("step-"):
        try:
            return (int(name.split("-", 1)[1]), mtime)
        except ValueError:
            return (0, mtime)
    # Completed / live stage root (models/trained/stage1) — treat as "current head"
    return (10**12, mtime)


def list_stage_checkpoints(
    stage: int,
    *,
    include_stage_root: bool = True,
) -> list[Path]:
    """Valid checkpoint roots for ``models/trained/stage{N}/`` (+ history/step-*)."""
    if stage < 1:
        return []
    trained = Path(TRAINED_ROOT) / f"stage{stage}"
    candidates: list[Path] = []
    if include_stage_root:
        candidates.append(trained)
        if stage == 1:
            candidates.append(Path(LEGACY_CHECKPOINT_DIR))
    history = trained / "history"
    if history.is_dir():
        candidates.extend(history.glob("step-*"))
    return [p for p in candidates if checkpoint_is_valid(p)]


def list_stage1_checkpoints(*, include_stage_root: bool = True) -> list[Path]:
    """Valid Stage-1 checkpoint roots (history/step-* and optional stage root)."""
    return list_stage_checkpoints(1, include_stage_root=include_stage_root)


def _checkpoint_candidates() -> list[Path]:
    return list_stage1_checkpoints(include_stage_root=True)


def _checkpoint_mtime_key(path: Path) -> float:
    proj = path / PROJECTION_FILE
    if proj.is_file():
        return proj.stat().st_mtime
    return path.stat().st_mtime


def find_latest_checkpoint_for_stage(stage: int) -> Path | None:
    """Most recent valid root under stage{N} (stage dir + history/step-*, by mtime)."""
    valid = list_stage_checkpoints(stage, include_stage_root=True)
    if not valid:
        return None
    return max(valid, key=_checkpoint_mtime_key)


def find_latest_checkpoint() -> Path | None:
    """Most recent valid Stage-1 root among stage dir + history/step-* (by mtime)."""
    return find_latest_checkpoint_for_stage(1)


def find_latest_history_checkpoint() -> Path | None:
    """Latest mid-training snapshot under ``models/trained/stage1/history/step-*``.

    Ignores the completed/live stage root so demos can load the newest
    intermediate save even when ``stage1/`` itself is older or incomplete.
    """
    valid = list_stage1_checkpoints(include_stage_root=False)
    if not valid:
        return None
    return max(valid, key=_checkpoint_step_key)


def find_latest_trained_checkpoint(
    *,
    max_stage: int = MAX_TRAINING_PHASE,
    min_stage: int = 1,
) -> Path | None:
    """Latest valid training checkpoint across stages.

    Scans ``stage{max_stage}`` … ``stage{min_stage}`` (high → low). For the
    highest stage that has any valid checkpoint, returns the newest root among
    that stage's live dir and ``history/step-*`` (by projection mtime).
    """
    hi = max(int(max_stage), int(min_stage))
    lo = min(int(max_stage), int(min_stage))
    lo = max(lo, 1)
    for stage in range(hi, lo - 1, -1):
        found = find_latest_checkpoint_for_stage(stage)
        if found is not None:
            return found
    return None


def resolve_inference_checkpoint(
    *,
    phase: int = 1,
    checkpoint_dir: str | Path | None = None,
    latest_history: bool = False,
    latest_any: bool = False,
    latest_across_stages: bool = False,
) -> Path | None:
    """Resolve a trained checkpoint root for inference/demos.

    Priority:
      1. Explicit ``checkpoint_dir``
      2. ``latest_across_stages`` → stage5…1, newest in highest stage that exists
      3. ``latest_history`` → newest ``stage1/history/step-*``
      4. ``latest_any`` → newest among stage1 root + history (mtime)
      5. ``None`` → caller uses phase-based ``models/trained/stage{N}/``
         (or returns that root when valid)
    """
    if checkpoint_dir:
        root = Path(checkpoint_dir)
        if not checkpoint_is_valid(root):
            raise FileNotFoundError(f"No valid Stage-1 checkpoint at {root}")
        return root
    if latest_across_stages:
        root = find_latest_trained_checkpoint()
        if root is None:
            raise FileNotFoundError(
                f"No valid checkpoints under {TRAINED_ROOT}/stage{{1-{MAX_TRAINING_PHASE}}}. "
                "Train first or use --phase 0 for seed weights."
            )
        return root
    if latest_history:
        root = find_latest_history_checkpoint()
        if root is None:
            raise FileNotFoundError(
                f"No history/step-* checkpoints under {DEFAULT_TRAINED_DIR}/history. "
                "Train longer with periodic saves, or omit --latest-checkpoint."
            )
        return root
    if latest_any:
        return find_latest_checkpoint()
    # Default: completed/live stage dir for phase (not history)
    if phase >= 1:
        root = Path(TRAINED_ROOT) / f"stage{phase}"
        if checkpoint_is_valid(root):
            return root
    return None


def resolve_checkpoint_root(
    *,
    fresh: bool,
    checkpoint_dir: str | None,
) -> Path | None:
    if fresh:
        return None
    if checkpoint_dir:
        root = Path(checkpoint_dir)
        if not checkpoint_is_valid(root):
            raise FileNotFoundError(f"No valid checkpoint at {root}")
        return root
    return find_latest_checkpoint()


def resolve_model_dirs(
    *,
    fresh: bool,
    checkpoint_dir: str | None,
    seed_vision_dir: str,
    seed_text_dir: str,
) -> tuple[Path, Path, Path | None]:
    checkpoint_root = resolve_checkpoint_root(
        fresh=fresh, checkpoint_dir=checkpoint_dir
    )
    if checkpoint_root is not None:
        print(f"Resuming from checkpoint: {checkpoint_root}")
        return (
            _component_dir(checkpoint_root, VISION_COMPONENT),
            _component_dir(checkpoint_root, TEXT_COMPONENT),
            checkpoint_root,
        )
    print(
        f"No checkpoint found — seeding from {seed_vision_dir} and {seed_text_dir}"
    )
    return Path(seed_vision_dir), Path(seed_text_dir), None


def _torch_load(path: Path | str, map_location="cpu", *, weights_only: bool | None = None):
    """torch.load wrapper with weights_only when supported (PyTorch 2.0+)."""
    kwargs: dict[str, Any] = {"map_location": map_location}
    if weights_only is not None:
        try:
            return torch.load(path, weights_only=weights_only, **kwargs)
        except TypeError:
            pass
    return torch.load(path, **kwargs)


def _filter_projection_state_dict(
    module: nn.Module, state: dict[str, Any]
) -> dict[str, Any]:
    """Drop keys the module does not own (e.g. old checkpoints with ``bias``)."""
    allowed = set(module.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in allowed}
    dropped = sorted(set(state.keys()) - allowed)
    if dropped:
        print(f"  projection load: ignored keys {dropped}")
    return filtered


def load_projection_heads(
    alignment_model: Stage1AlignmentModel,
    checkpoint_root: Path,
):
    state = _torch_load(
        checkpoint_root / PROJECTION_FILE, map_location="cpu", weights_only=True
    )
    v_sd = _filter_projection_state_dict(
        alignment_model.vision_projection, state["vision_projection"]
    )
    t_sd = _filter_projection_state_dict(
        alignment_model.text_projection, state["text_projection"]
    )
    alignment_model.vision_projection.load_state_dict(v_sd, strict=True)
    alignment_model.text_projection.load_state_dict(t_sd, strict=True)
    alignment_model.vision_projection.to(
        device=alignment_model.vision_device,
        dtype=alignment_model.compute_dtype,
    )
    alignment_model.text_projection.to(
        device=alignment_model.text_device,
        dtype=alignment_model.compute_dtype,
    )
    print(f"Loaded projection heads from {checkpoint_root / PROJECTION_FILE}")


def save_training_state(
    trained_dir: Path,
    global_step: int,
    optimizer: torch.optim.Optimizer,
):
    torch.save(
        {"global_step": global_step, "optimizer": optimizer.state_dict()},
        trained_dir / TRAINING_STATE_FILE,
    )


def load_training_state(
    checkpoint_root: Path,
    optimizer: torch.optim.Optimizer,
) -> int:
    path = checkpoint_root / TRAINING_STATE_FILE
    if not path.is_file():
        print(f"No training state at {path}; starting from step 0.")
        return 0
    # Optimizer state is not pure tensors; must allow full unpickle.
    state = _torch_load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    step = int(state.get("global_step", 0))
    print(f"Resumed optimizer state from step {step}")
    return step


def _write_snapshot(trained_dir: Path, snapshot_root: Path):
    snapshot_root.mkdir(parents=True, exist_ok=True)
    for name in (VISION_COMPONENT, TEXT_COMPONENT):
        src = trained_dir / name
        dst = snapshot_root / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    for filename in (PROJECTION_FILE, TRAINING_STATE_FILE, CONFIG_FILE):
        src = trained_dir / filename
        if src.is_file():
            shutil.copy2(src, snapshot_root / filename)


def _config_source_dir(component_dir: Path, seed_dir: str) -> str:
    for candidate in (component_dir, Path(seed_dir)):
        if _valid_config_path(candidate / "config.json"):
            return str(candidate)
    raise FileNotFoundError(
        f"No valid config.json in {component_dir} or seed dir {seed_dir}"
    )


def _json_safe(value: Any) -> Any:
    """Convert config values to JSON-serializable forms (HF / Unsloth tolerant)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "value") and not callable(value):
        try:
            return _json_safe(value.value)
        except Exception:
            pass
    if callable(value):
        return None
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _quantization_config_dict(model: nn.Module) -> dict[str, Any] | None:
    """Plain-dict BitsAndBytes config for HF-compatible config.json."""
    config = getattr(model, "config", None)
    if config is None:
        return None
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        # Detect 8-bit weights from the live state dict as a fallback.
        for key in model.state_dict():
            if key.endswith(".SCB"):
                return {
                    "quant_method": "bitsandbytes",
                    "load_in_8bit": True,
                    "load_in_4bit": False,
                    "_load_in_8bit": True,
                    "_load_in_4bit": False,
                    "llm_int8_threshold": 6.0,
                    "llm_int8_has_fp16_weight": False,
                    "llm_int8_enable_fp32_cpu_offload": False,
                    "llm_int8_skip_modules": None,
                    "bnb_4bit_quant_type": "fp4",
                    "bnb_4bit_use_double_quant": False,
                    "bnb_4bit_compute_dtype": "float32",
                    "bnb_4bit_quant_storage": "uint8",
                }
        return None
    if hasattr(qc, "to_dict"):
        raw = qc.to_dict()
    elif isinstance(qc, dict):
        raw = dict(qc)
    else:
        return None
    cleaned = _json_safe(raw)
    if not isinstance(cleaned, dict):
        return None
    # Unsloth injects a non-serializable lambda as get_loading_attributes.
    cleaned.pop("get_loading_attributes", None)
    cleaned = {k: v for k, v in cleaned.items() if v is not None or k.startswith("_")}
    if "quant_method" not in cleaned:
        cleaned["quant_method"] = "bitsandbytes"
    if cleaned.get("load_in_8bit") or cleaned.get("_load_in_8bit"):
        cleaned["load_in_8bit"] = True
        cleaned["_load_in_8bit"] = True
    return cleaned


def _load_base_config_dict(config_source_dir: str) -> dict[str, Any]:
    path = Path(config_source_dir) / "config.json"
    if not _valid_config_path(path):
        raise FileNotFoundError(f"Missing or invalid config.json at {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config.json at {path} is not a JSON object")
    return data


def _write_hf_config_json(
    model: nn.Module,
    output_dir: Path,
    config_source_dir: str,
) -> None:
    """Write a HuggingFace-readable config.json (with quantization_config when 8-bit)."""
    config_dict: dict[str, Any]
    model_config = getattr(model, "config", None)
    if model_config is not None and hasattr(model_config, "to_dict"):
        try:
            config_dict = _json_safe(model_config.to_dict())
            if not isinstance(config_dict, dict):
                raise TypeError("config.to_dict() did not return a dict")
        except Exception:
            config_dict = _load_base_config_dict(config_source_dir)
    else:
        config_dict = _load_base_config_dict(config_source_dir)

    # Drop non-serializable / internal Unsloth patches.
    config_dict = {
        k: v for k, v in config_dict.items()
        if v is not None and not callable(v)
    }

    qc = _quantization_config_dict(model)
    if qc is not None:
        config_dict["quantization_config"] = qc
        # Reflect actual stored precision for external loaders.
        if "dtype" not in config_dict or config_dict.get("dtype") == "float32":
            config_dict["dtype"] = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    else:
        config_dict.pop("quantization_config", None)

    out = output_dir / "config.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(config_dict, fh, indent=2, sort_keys=True)
        fh.write("\n")
    if not _valid_config_path(out):
        raise RuntimeError(f"Failed to write valid config.json at {out}")


def _state_dict_for_safetensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU-contiguous state dict safe for ``safetensors.torch.save_file``."""
    state: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        tensor = value.detach().contiguous().cpu()
        state[key] = tensor
    return state


def _save_component_checkpoint(
    model: nn.Module,
    output_dir: Path,
    config_source_dir: str,
):
    """Save weights + HF config.json for a single tower.

    Uses safetensors + an explicitly written config so 8-bit BitsAndBytes
    checkpoints include ``quantization_config`` (required by HuggingFace and
    tools like Unsloth Studio). Unsloth's live config can contain non-JSON
    callables, so we never rely on ``model.save_pretrained`` alone.
    """
    import safetensors.torch

    output_dir.mkdir(parents=True, exist_ok=True)
    state = _state_dict_for_safetensors(model)
    safetensors.torch.save_file(state, output_dir / "model.safetensors")
    _write_hf_config_json(model, output_dir, config_source_dir)


def _save_tokenizer_files(tokenizer: Any, output_dir: Path) -> None:
    if tokenizer is None:
        return
    if not hasattr(tokenizer, "save_pretrained"):
        return
    tokenizer.save_pretrained(str(output_dir))


def _save_image_processor_files(
    image_processor: Any,
    output_dir: Path,
    *,
    image_size: int | None = None,
) -> None:
    if image_processor is None:
        return
    if image_size is not None and hasattr(image_processor, "size"):
        image_processor.size = {"height": image_size, "width": image_size}
    if hasattr(image_processor, "save_pretrained"):
        image_processor.save_pretrained(str(output_dir))


def _save_generation_config(model: nn.Module, output_dir: Path) -> None:
    gen = getattr(model, "generation_config", None)
    if gen is None or not hasattr(gen, "to_dict"):
        return
    try:
        data = _json_safe(gen.to_dict())
        if not isinstance(data, dict):
            return
        path = output_dir / "generation_config.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except Exception:
        return


def save_stage1_checkpoint(
    trained_dir: Path,
    alignment_model: Stage1AlignmentModel,
    args: Any,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    *,
    tokenizer: Any = None,
    image_processor: Any = None,
):
    """Save Stage-1 towers in a HuggingFace-layout checkpoint directory.

    Layout (per tower)::

        trained_dir/
          vision_model/{config.json, model.safetensors, preprocessor_config.json}
          text_model/{config.json, model.safetensors, tokenizer files, ...}
          projection_heads.pt
          training_state.pt
          stage1_config.json
    """
    trained_dir.mkdir(parents=True, exist_ok=True)

    vision_dir = trained_dir / VISION_COMPONENT
    text_dir = trained_dir / TEXT_COMPONENT
    _save_component_checkpoint(
        alignment_model.vision_model,
        vision_dir,
        _config_source_dir(vision_dir, args.seed_vision_dir),
    )
    _save_component_checkpoint(
        alignment_model.text_model,
        text_dir,
        _config_source_dir(text_dir, args.seed_text_dir),
    )

    vision_image_size = getattr(
        getattr(alignment_model.vision_model, "config", None), "image_size", None
    )
    _save_image_processor_files(
        image_processor, vision_dir, image_size=vision_image_size
    )
    _save_tokenizer_files(tokenizer, text_dir)
    _save_generation_config(alignment_model.text_model, text_dir)

    torch.save(
        {
            "vision_projection": alignment_model.vision_projection.state_dict(),
            "text_projection": alignment_model.text_projection.state_dict(),
        },
        trained_dir / PROJECTION_FILE,
    )
    with open(trained_dir / CONFIG_FILE, "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, default=str)
    save_training_state(trained_dir, global_step, optimizer)

    if global_step > 0 and args.save_steps > 0 and global_step % args.save_steps == 0:
        snapshot = trained_dir / "history" / f"step-{global_step}"
        _write_snapshot(trained_dir, snapshot)

    print(f"Saved Stage 1 checkpoint to {trained_dir}")


def verify_trained_checkpoint(
    trained_dir: Path,
    seed_vision_dir: str,
    seed_text_dir: str,
    tokenizer_id: str = QWEN_TOKENIZER_ID,
    vision_processor_id: str = SIGLIP_PROCESSOR_ID,
    max_text_length: int = DEFAULT_MAX_INPUT_TOKENS,
    bf16: bool = True,
    vision_gpu: int = 0,
    text_gpu: int = 1,
    with_text_queries: bool = False,
) -> None:
    from transformers import AutoImageProcessor

    if not checkpoint_is_valid(trained_dir):
        raise RuntimeError(f"Checkpoint at {trained_dir} is incomplete.")

    compute_dtype = resolve_training_dtype(bf16)
    vision_device = gpu_device(vision_gpu)
    text_device = gpu_device(text_gpu)

    vision_model, vision_hidden = load_vision_model_for_training(
        str(trained_dir / VISION_COMPONENT),
        vision_device=vision_device,
        compute_dtype=compute_dtype,
    )
    text_model, tokenizer, text_hidden = load_text_model_for_training(
        str(trained_dir / TEXT_COMPONENT),
        tokenizer_id=tokenizer_id,
        max_seq_length=max_text_length,
        text_device=text_device,
        compute_dtype=compute_dtype,
    )

    model = Stage1AlignmentModel(
        vision_model=vision_model,
        text_model=text_model,
        vision_hidden=vision_hidden,
        text_hidden=text_hidden,
        vision_device=vision_device,
        text_device=text_device,
        compute_dtype=compute_dtype,
    )
    load_projection_heads(model, trained_dir)

    # Project dataset only (curated Hub); queries already present — no Flickr/OpenRouter.
    rows = load_verification_samples(count=4)
    processor = AutoImageProcessor.from_pretrained(vision_processor_id)
    target_size = vision_model.config.image_size
    processor.size = {"height": target_size, "width": target_size}
    verify_rows = rows
    if with_text_queries:
        from trisearch_dataset import enrich_rows_with_text_queries

        # Curated rows already have queries; this is a no-op / no API.
        verify_rows = enrich_rows_with_text_queries(
            rows,
            max_new_queries=0,
            skip_generation=True,
        )
    dataset = ImageCaptionDataset(
        rows=verify_rows,
        image_processor=processor,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
        with_text_queries=with_text_queries,
    )
    batch = Stage1Collator(
        pad_token_id=tokenizer.pad_token_id or 0,
        with_text_queries=with_text_queries,
    )([dataset[i] for i in range(min(2, len(dataset)))])

    model.eval()
    with torch.no_grad():
        out = model(**batch, return_loss=True)
    loss = float(out["loss"])
    if loss <= 0.0 or not math.isfinite(loss):
        raise RuntimeError(f"Verification forward pass produced invalid loss {loss}")

    print(
        f"Verification OK: loaded {trained_dir} and ran forward pass "
        f"(loss={loss:.4f})."
    )


def is_projection_param_name(name: str) -> bool:
    """True for Matryoshka projection heads (always trained)."""
    return "vision_projection" in name or "text_projection" in name


def _param_can_require_grad(param: torch.Tensor) -> bool:
    """True if PyTorch allows ``requires_grad`` on this tensor.

    bitsandbytes 8-bit ``Int8Params`` store weights as integer dtypes; those
    cannot have ``requires_grad=True`` (only float/complex). Freezing still
    sets ``requires_grad=False`` on them; unfreezing must skip them. Trainable
    float scales / LoRA / projection heads are unaffected.
    """
    return bool(param.is_floating_point() or param.is_complex())


def set_backbone_trainable(
    model: Stage1AlignmentModel,
    trainable: bool,
) -> dict[str, int]:
    """Freeze/unfreeze vision + text towers; projections always stay trainable.

    When frozen, backbone modules are set to ``eval()`` (stable features, no
    dropout) while projection heads remain in train mode.

    Only floating/complex parameters are toggled for ``requires_grad`` (bnb
    int8 storage tensors are left alone — see ``_param_can_require_grad``).
    """
    n_backbone = 0
    n_proj = 0
    n_skipped_nonfloat = 0
    want = bool(trainable)
    for name, param in model.named_parameters():
        if is_projection_param_name(name):
            if _param_can_require_grad(param):
                param.requires_grad = True
            n_proj += param.numel()
            continue

        n_backbone += param.numel()
        if not _param_can_require_grad(param):
            # int8 (or other non-float) storage: cannot enable grads; ensure off
            if param.requires_grad:
                param.requires_grad = False
            n_skipped_nonfloat += 1
            continue
        param.requires_grad = want

    # Keep batch-norm / dropout off on frozen towers for stable semantics.
    if hasattr(model, "vision_model") and model.vision_model is not None:
        model.vision_model.train(mode=want)
    if hasattr(model, "text_model") and model.text_model is not None:
        model.text_model.train(mode=want)
    if hasattr(model, "vision_projection"):
        model.vision_projection.train(True)
    if hasattr(model, "text_projection"):
        model.text_projection.train(True)

    return {
        "backbone_params": n_backbone,
        "projection_params": n_proj,
        "backbone_trainable": int(want),
        "skipped_nonfloat": n_skipped_nonfloat,
    }


def count_trainable_parameters(model: Stage1AlignmentModel) -> tuple[int, int]:
    """Return ``(trainable_numel, total_numel)``."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def resolve_backbone_freeze_steps(
    args: Any,
    total_steps: int,
    *,
    start_step: int = 0,
) -> int:
    """Global step (exclusive) until which backbones stay frozen.

    Priority: ``--freeze-backbone-steps`` if >= 0, else
    ``floor(total_steps * --freeze-backbone-ratio)``. Returns 0 when disabled.
    """
    if getattr(args, "no_freeze_backbone", False):
        return 0
    steps = getattr(args, "freeze_backbone_steps", None)
    if steps is not None and int(steps) >= 0:
        freeze_until = int(steps)
    else:
        ratio = float(getattr(args, "freeze_backbone_ratio", 0.0) or 0.0)
        if ratio <= 0.0:
            return 0
        freeze_until = int(total_steps * ratio)
    # Already past freeze window on resume → no freeze.
    if start_step >= freeze_until:
        return 0
    return max(0, freeze_until)


def build_optimizer(
    model: Stage1AlignmentModel,
    learning_rate: float,
    vision_learning_rate: float,
    projection_learning_rate: float,
    weight_decay: float,
):
    import bitsandbytes as bnb

    # Include all parameters (even if currently frozen) so unfreezing mid-run
    # does not require rebuilding the optimizer. Frozen params simply get no
    # grads until ``set_backbone_trainable(True)``.
    vision_params, text_params, projection_params = [], [], []
    for name, param in model.named_parameters():
        if is_projection_param_name(name):
            projection_params.append(param)
        elif "vision_model" in name:
            vision_params.append(param)
        else:
            text_params.append(param)

    if not projection_params:
        raise RuntimeError(
            "No projection parameters found for the optimizer "
            "(expected vision_projection / text_projection)."
        )
    if not vision_params:
        raise RuntimeError("No vision_model parameters found for the optimizer.")
    if not text_params:
        raise RuntimeError("No text_model parameters found for the optimizer.")

    # Fixed group order for run_training LR updates: 0=vision, 1=text, 2=proj.
    return bnb.optim.AdamW8bit(
        [
            {"params": vision_params, "lr": vision_learning_rate},
            {"params": text_params, "lr": learning_rate},
            {"params": projection_params, "lr": projection_learning_rate},
        ],
        weight_decay=weight_decay,
    )


def sanity_check_loss(model: Stage1AlignmentModel, dataloader: DataLoader):
    batch = next(iter(dataloader))
    model.eval()
    with torch.no_grad():
        outputs = model(**batch, return_loss=True)
    loss = float(outputs["loss"])
    contrastive = float(outputs["contrastive_loss"])
    matryoshka = float(outputs["matryoshka_loss"])
    query_image = float(outputs.get("query_image_loss", 0.0))
    text_text = float(outputs.get("text_text_loss", 0.0))
    text_text_matryoshka = float(outputs.get("text_text_matryoshka_loss", 0.0))
    pos_rank = float(outputs.get("pos_rank", float("nan")))
    n_docs = int(outputs.get("n_image_docs", batch["input_ids"].shape[0]))
    model.train()
    print(
        f"Sanity check (batch={batch['input_ids'].shape[0]}): "
        f"loss={loss:.4f} cap↔img={contrastive:.4f} mrl={matryoshka:.4f} "
        f"q↔img={query_image:.4f} q→cap={text_text:.4f} "
        f"q→cap_mrl={text_text_matryoshka:.4f} "
        f"pos_rank={pos_rank:.2f}/{n_docs}"
    )
    if loss <= 0.0 or not math.isfinite(loss):
        raise RuntimeError(
            "Loss is zero or non-finite before training. "
            "Increase --batch-size to at least 2."
        )


def tokenize_text_pair_batch(
    tokenizer: Any,
    pairs: list[Any],
    *,
    max_length: int = DEFAULT_MAX_INPUT_TOKENS,
) -> dict[str, torch.Tensor]:
    """Tokenize a list of TextPairSample (or anchor/positive/negative tuples)."""
    anchors = []
    positives = []
    negatives = []
    has_neg = False
    for p in pairs:
        if hasattr(p, "anchor"):
            a, b, c = p.anchor, p.positive, p.negative
        else:
            a, b = p[0], p[1]
            c = p[2] if len(p) > 2 else None
        anchors.append(str(a))
        positives.append(str(b))
        if c:
            has_neg = True
            negatives.append(str(c))
        else:
            negatives.append("")  # placeholder; dropped if none have neg

    def _tok(texts: list[str]) -> dict[str, torch.Tensor]:
        out = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    a = _tok(anchors)
    b = _tok(positives)
    result = {
        "anchor_input_ids": a["input_ids"],
        "anchor_attention_mask": a["attention_mask"],
        "positive_input_ids": b["input_ids"],
        "positive_attention_mask": b["attention_mask"],
    }
    if has_neg and all(n for n in negatives):
        c = _tok(negatives)
        result["negative_input_ids"] = c["input_ids"]
        result["negative_attention_mask"] = c["attention_mask"]
    return result


def run_training(
    model: Stage1AlignmentModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: Any,
    start_step: int = 0,
    *,
    tokenizer: Any = None,
    image_processor: Any = None,
    paraphrase_queue: Any = None,
) -> int:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # --max-steps is a total global-step budget (HF Trainer convention), not
    # "additional steps after resume".
    max_steps = args.max_steps if args.max_steps > 0 else None
    total_steps = max_steps or max(
        1,
        int(len(dataloader) * args.num_epochs // max(args.gradient_accumulation_steps, 1)),
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    freeze_until = resolve_backbone_freeze_steps(
        args, total_steps, start_step=start_step
    )
    backbone_frozen = freeze_until > start_step
    # Geo is inert under frozen backbones (linear proj cannot un-cone); defer by default.
    geo_after_unfreeze = bool(getattr(args, "geo_after_unfreeze", DEFAULT_GEO_AFTER_UNFREEZE))
    if hasattr(model, "set_geo_active"):
        if geo_after_unfreeze and backbone_frozen:
            model.set_geo_active(False)
            print(
                "Embedding geometry: deferred until backbone unfreeze "
                f"(step {freeze_until})."
            )
        else:
            model.set_geo_active(True)
            if geo_after_unfreeze and not backbone_frozen:
                print("Embedding geometry: active (backbone already unfrozen).")
            elif not geo_after_unfreeze:
                print("Embedding geometry: active from step 0 (--geo-during-freeze).")
    if backbone_frozen:
        stats = set_backbone_trainable(model, False)
        tr, tot = count_trainable_parameters(model)
        print(
            f"Projection-only phase: freeze vision/text backbones until "
            f"global step {freeze_until} "
            f"(trainable {tr:,} / {tot:,} params; "
            f"proj≈{stats['projection_params']:,})."
        )
    else:
        set_backbone_trainable(model, True)
        if getattr(args, "no_freeze_backbone", False) or freeze_until == 0:
            print("Backbone freeze: disabled (full tower training from start).")
        else:
            print(
                f"Backbone freeze: skipped (resume step {start_step} "
                f">= freeze_until {freeze_until})."
            )

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return args.learning_rate * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    global_step = start_step
    micro_step = 0
    log_loss = 0.0
    log_contrastive = 0.0
    log_matryoshka = 0.0
    log_query_image = 0.0
    log_text_text = 0.0
    log_text_text_matryoshka = 0.0
    log_heatmap_sparsity = 0.0
    log_geo = 0.0
    log_geo_mu = 0.0
    log_geo_min_std = 0.0
    log_pos_rank = 0.0
    log_score_gap = 0.0
    log_paraphrase = 0.0
    log_batches = 0
    bank_clear_steps = int(getattr(args, "bank_clear_steps", DEFAULT_BANK_CLEAR_STEPS) or 0)
    paraphrase_weight = float(getattr(args, "paraphrase_weight", 0.0) or 0.0)
    paraphrase_batch_size = int(
        getattr(args, "paraphrase_batch_size", 0) or args.batch_size
    )
    paraphrase_max_len = int(
        getattr(args, "max_text_length", DEFAULT_MAX_INPUT_TOKENS)
        or DEFAULT_MAX_INPUT_TOKENS
    )

    if max_steps is not None and start_step >= max_steps:
        print(f"Already at step {start_step} (max_steps={max_steps}); nothing to train.")
        return start_step

    accum = max(int(args.gradient_accumulation_steps), 1)

    while True:
        for batch in dataloader:
            # Policy B: snapshot bank for scoring at the start of each accum window.
            if micro_step % accum == 0 and hasattr(model, "begin_accum_window"):
                model.begin_accum_window()

            # model.train() would re-enable backbone dropout; re-assert freeze.
            if backbone_frozen:
                set_backbone_trainable(model, False)

            outputs = model(**batch, return_loss=True)
            loss = outputs["loss"]
            para_val = 0.0
            if (
                paraphrase_queue is not None
                and paraphrase_weight > 0.0
                and tokenizer is not None
            ):
                pairs = paraphrase_queue.pop_batch(paraphrase_batch_size)
                tok = tokenize_text_pair_batch(
                    tokenizer, pairs, max_length=paraphrase_max_len
                )
                para_kwargs = {
                    "anchor_input_ids": tok["anchor_input_ids"],
                    "anchor_attention_mask": tok["anchor_attention_mask"],
                    "positive_input_ids": tok["positive_input_ids"],
                    "positive_attention_mask": tok["positive_attention_mask"],
                }
                if "negative_input_ids" in tok:
                    para_kwargs["negative_input_ids"] = tok["negative_input_ids"]
                    para_kwargs["negative_attention_mask"] = tok[
                        "negative_attention_mask"
                    ]
                para_loss = model.compute_paraphrase_loss(**para_kwargs)
                para_val = float(para_loss.detach().float().item())
                loss = loss + paraphrase_weight * para_loss

            loss_val = loss.detach().float().item()
            if loss_val <= 0.0 or not math.isfinite(loss_val):
                raise RuntimeError(
                    f"Invalid loss {loss_val} at micro-step {micro_step}. "
                    f"batch={batch['input_ids'].shape[0]} — need batch_size >= 2."
                )

            (loss / accum).backward()

            log_loss += loss_val
            log_contrastive += float(outputs["contrastive_loss"])
            log_matryoshka += float(outputs["matryoshka_loss"])
            log_query_image += float(outputs.get("query_image_loss", 0.0))
            log_text_text += float(outputs.get("text_text_loss", 0.0))
            log_text_text_matryoshka += float(
                outputs.get("text_text_matryoshka_loss", 0.0)
            )
            log_heatmap_sparsity += float(
                outputs.get("heatmap_sparsity_loss", 0.0)
            )
            log_geo += float(outputs.get("geo_loss", 0.0))
            log_geo_mu += float(outputs.get("geo_mu_norm", 0.0))
            log_geo_min_std += float(outputs.get("geo_min_std", 0.0))
            log_pos_rank += float(outputs.get("pos_rank", 0.0))
            log_paraphrase += para_val
            gap = outputs.get("score_gap", float("nan"))
            if gap == gap:  # not NaN
                log_score_gap += float(gap)
            log_batches += 1
            micro_step += 1

            if micro_step % accum != 0:
                continue

            text_lr = lr_at(global_step)
            vision_lr = args.vision_learning_rate or text_lr
            schedule_scale = text_lr / max(args.learning_rate, 1e-12)
            proj_lr = args.projection_learning_rate * schedule_scale
            # During proj-only phase, zero backbone LRs (belt-and-suspenders).
            if backbone_frozen:
                optimizer.param_groups[0]["lr"] = 0.0
                optimizer.param_groups[1]["lr"] = 0.0
            else:
                optimizer.param_groups[0]["lr"] = vision_lr
                optimizer.param_groups[1]["lr"] = text_lr
            optimizer.param_groups[2]["lr"] = proj_lr

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=args.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if backbone_frozen and global_step >= freeze_until:
                stats = set_backbone_trainable(model, True)
                backbone_frozen = False
                if geo_after_unfreeze and hasattr(model, "set_geo_active"):
                    model.set_geo_active(True)
                tr, tot = count_trainable_parameters(model)
                skip = stats.get("skipped_nonfloat", 0)
                skip_msg = (
                    f" skipped_nonfloat={skip} (bnb int8 storage)"
                    if skip
                    else ""
                )
                geo_msg = (
                    " Embedding geometry enabled."
                    if geo_after_unfreeze
                    else ""
                )
                print(
                    f"Unfroze vision/text backbones at step {global_step} "
                    f"(trainable {tr:,} / {tot:,} params{skip_msg}). "
                    f"Continuing with full tower fine-tuning.{geo_msg}"
                )

            # Drop stale bank negatives so a collapse episode cannot lock CE at chance.
            if (
                bank_clear_steps > 0
                and global_step > 0
                and global_step % bank_clear_steps == 0
                and hasattr(model, "memory_bank")
            ):
                bank = model.memory_bank
                if bank is not None and len(bank) > 0:
                    n_cleared = len(bank)
                    bank.clear()
                    if hasattr(model, "_score_bank_snapshot"):
                        model._score_bank_snapshot = None
                    print(
                        f"Cleared memory bank at step {global_step} "
                        f"({n_cleared} entries; refresh every {bank_clear_steps} steps)."
                    )

            if global_step == 1 or global_step % args.logging_steps == 0:
                phase = "proj-only" if backbone_frozen else "full"
                msg = (
                    f"step {global_step:5d} | "
                    f"[{phase}] | "
                    f"loss {log_loss / log_batches:.4f} | "
                    f"cap↔img {log_contrastive / log_batches:.4f} | "
                    f"mrl {log_matryoshka / log_batches:.4f}"
                )
                if log_query_image > 0.0 or getattr(args, "query_image_weight", 0.0) > 0.0:
                    msg += f" | q↔img {log_query_image / log_batches:.4f}"
                if log_text_text > 0.0 or getattr(args, "text_text_weight", 0.0) > 0.0:
                    msg += f" | q→cap {log_text_text / log_batches:.4f}"
                if (
                    log_text_text_matryoshka > 0.0
                    or getattr(args, "matryoshka_weight", 0.0) > 0.0
                ):
                    msg += (
                        f" | q→cap_mrl {log_text_text_matryoshka / log_batches:.4f}"
                    )
                if paraphrase_weight > 0.0:
                    msg += f" | para {log_paraphrase / log_batches:.4f}"
                if (
                    log_heatmap_sparsity > 0.0
                    or getattr(args, "heatmap_sparsity_weight", 0.0) > 0.0
                ):
                    msg += (
                        f" | heat_sparse {log_heatmap_sparsity / log_batches:.4f}"
                    )
                if log_geo > 0.0 or getattr(args, "embedding_geo_weight", 0.0) > 0.0:
                    msg += (
                        f" | geo {log_geo / log_batches:.4f}"
                        f" (μ={log_geo_mu / log_batches:.4f}"
                        f" minσ={log_geo_min_std / log_batches:.4f})"
                    )
                bank_len = len(getattr(model, "memory_bank", ()))
                if bank_len or getattr(args, "memory_bank_size", 0):
                    msg += f" | bank {bank_len}"
                n_docs = int(outputs.get("n_image_docs", args.batch_size))
                msg += f" | pos_rank {log_pos_rank / log_batches:.1f}/{n_docs}"
                if log_batches > 0:
                    msg += f" | gap {log_score_gap / log_batches:.3f}"
                msg += f" | grad_norm {float(grad_norm):.4f} | lr {text_lr:.2e}"
                print(msg)
                log_loss = log_contrastive = log_matryoshka = 0.0
                log_query_image = 0.0
                log_text_text = log_text_text_matryoshka = 0.0
                log_heatmap_sparsity = 0.0
                log_geo = log_geo_mu = log_geo_min_std = 0.0
                log_pos_rank = log_score_gap = 0.0
                log_paraphrase = 0.0
                log_batches = 0

            if global_step % args.save_steps == 0:
                save_stage1_checkpoint(
                    Path(args.trained_dir),
                    model,
                    args,
                    global_step,
                    optimizer,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                )

            if max_steps is not None and global_step >= max_steps:
                return global_step

        if max_steps is None:
            break

    return global_step
