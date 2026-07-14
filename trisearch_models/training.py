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

from .inference import (
    EMBED_DIM,
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
DEFAULT_TRAINED_DIR = "models/trained/stage1"
LEGACY_CHECKPOINT_DIR = "checkpoints/stage1"
VISION_COMPONENT = "vision_model"
TEXT_COMPONENT = "text_model"
PROJECTION_FILE = "projection_heads.pt"
TRAINING_STATE_FILE = "training_state.pt"
CONFIG_FILE = "stage1_config.json"
DEFAULT_MAX_INPUT_TOKENS = 256
DEFAULT_MATRYOSHKA_DIMS = (64, 128, 256, 512, 1024)
# Soft MaxSim: τ_s * logsumexp(sim / τ_s). Smaller τ_s → closer to hard max.
DEFAULT_SOFT_MAXSIM_TEMPERATURE = 0.05
# Caption token-Jaccard above this → treat as non-negative (not a false neg).
DEFAULT_MULTI_POSITIVE_JACCARD = 0.5
# Keep top fraction of SigLIP patches by pre-norm L2 (drop background).
DEFAULT_VISION_PATCH_KEEP_RATIO = 0.75
# Embedding geometry regularizer (anti-cone / isotropy). Overall scale.
DEFAULT_EMBEDDING_GEO_WEIGHT = 0.05
DEFAULT_GEO_CENTER_WEIGHT = 1.0
DEFAULT_GEO_VAR_WEIGHT = 1.0
DEFAULT_GEO_VEC_MEAN_WEIGHT = 0.25
DEFAULT_GEO_MAG_FLOOR = 0.05
DEFAULT_GEO_MAG_FLOOR_WEIGHT = 0.1
# Per-dim std target as fraction of ideal isotropic 1/sqrt(D).
DEFAULT_GEO_VAR_RATIO = 0.5
# Soft penalty when |coord| exceeds this * 1/sqrt(D) (stops single-dim domination).
DEFAULT_GEO_MAX_ABS_RATIO = 4.0
DEFAULT_GEO_MAX_ABS_WEIGHT = 0.1
# Also regularize a Matryoshka prefix (re-normalized).
DEFAULT_GEO_PREFIX_DIM = 256
DEFAULT_GEO_PREFIX_WEIGHT = 0.5
DEFAULT_GEO_EMA_MOMENTUM = 0.99


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


def differentiable_late_interaction_score(
    query: torch.Tensor,
    doc: torch.Tensor,
    *,
    soft_maxsim_temperature: float | None = None,
) -> torch.Tensor:
    """Mean-MaxSim: mean over query tokens of max cosine to any doc token.

    Using the mean (not sum) keeps logits O(1) for InfoNCE with temperature
    ~0.07 even when captions are long, avoiding softmax saturation and
    pathological gradient norms with a large memory bank.

    When ``soft_maxsim_temperature`` is set (>0), hard max is replaced by
    soft MaxSim (τ logsumexp).
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
    return per_q.mean()


def build_late_interaction_matrix(
    query_tokens: list[torch.Tensor],
    doc_tokens: list[torch.Tensor],
    *,
    soft_maxsim_temperature: float | None = None,
) -> torch.Tensor:
    """Pairwise mean-MaxSim matrix of shape ``(len(queries), len(docs))``.

    Vectorized via padded tensors + einsum (same math as the per-pair loop).
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

    q_mask_f = q_mask.to(dtype=per_q.dtype).unsqueeze(1)  # (Bq, 1, Tq)
    # Zero-out padded query positions; mean over valid query tokens.
    per_q = per_q * q_mask_f
    denom = q_mask_f.sum(dim=-1).clamp_min(1.0)  # (Bq, 1)
    return per_q.sum(dim=-1) / denom


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
    ema_blend: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Anti-cone / isotropy regularizer on L2-normalized embeddings.

    Parameters
    ----------
    normalized
        ``(N, D)`` unit vectors (token-level and/or mean-pooled). Gradients flow.
    raw
        Optional matching ``(N, D)`` *pre-norm* projections for a magnitude floor
        (stops dead projections before normalize).
    ema_mean
        Optional detached running mean ``(D,)`` mixed into the center target so
        small micro-batches still see a stable cone direction to cancel.

    Terms
    -----
    * **center** — ``||μ||²`` pushes batch (and EMA-blended) mean toward 0 so
      dims are not permanently biased (e.g. all v₀ ≈ 0.05).
    * **variance** — hinge below ``var_ratio / sqrt(D)`` per dim (VICReg-style).
    * **vec_mean** — ``E[(mean_d v_d)²]`` soft anti all-positive / all-negative.
    * **mag_floor** — pre-norm ``ReLU(ε - ||z||)`` so projections do not die.
    * **max_abs** — soft penalty when any |coord| ≫ isotropic scale.
    """
    if normalized.ndim == 1:
        normalized = normalized.unsqueeze(0)
    if normalized.numel() == 0 or normalized.shape[0] == 0:
        zero = normalized.new_zeros(())
        return zero, {
            "geo_center": 0.0,
            "geo_var": 0.0,
            "geo_vec_mean": 0.0,
            "geo_mag_floor": 0.0,
            "geo_max_abs": 0.0,
            "geo_mu_norm": 0.0,
            "geo_min_std": 0.0,
            "geo_mean_abs_mu": 0.0,
        }

    # Work in fp32 for stable stats; cast loss back to normalized dtype.
    v = normalized.float()
    n, dim = v.shape
    inv_sqrt_d = 1.0 / math.sqrt(max(dim, 1))
    var_target = float(var_ratio) * inv_sqrt_d
    max_abs_target = float(max_abs_ratio) * inv_sqrt_d

    batch_mu = v.mean(dim=0)
    if ema_mean is not None and ema_mean.numel() == dim:
        # Blend live mean with detached EMA so B=2 still sees the cone.
        mu = (1.0 - float(ema_blend)) * batch_mu + float(ema_blend) * ema_mean.float().detach()
    else:
        mu = batch_mu
    center = (mu * mu).sum()

    # Unbiased=False: small-N friendly; matches VICReg practice.
    std = v.std(dim=0, unbiased=False).clamp_min(0.0)
    var_hinge = F.relu(var_target - std).pow(2).mean()

    vec_mean = v.mean(dim=-1)
    vec_mean_pen = (vec_mean * vec_mean).mean()

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

    total = (
        float(center_weight) * center
        + float(var_weight) * var_hinge
        + float(vec_mean_weight) * vec_mean_pen
        + float(mag_floor_weight) * mag_pen
        + float(max_abs_weight) * max_abs_pen
    )
    total = total.to(dtype=normalized.dtype)

    metrics = {
        "geo_center": float(center.detach()),
        "geo_var": float(var_hinge.detach()),
        "geo_vec_mean": float(vec_mean_pen.detach()),
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
    """1-based mean rank of the positive class (1 = best). Lower is better."""
    if scores.ndim != 2 or scores.size(0) == 0:
        return float("nan")
    batch = scores.size(0)
    if labels is None:
        labels = torch.arange(batch, device=scores.device)
    pos = scores.gather(1, labels.view(-1, 1)).squeeze(1)
    # Ties: count strictly better scores only (optimistic rank).
    ranks = 1 + (scores > pos.unsqueeze(1)).sum(dim=1).to(dtype=torch.float32)
    return float(ranks.mean().item())


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


def _expand_non_negative_mask(
    batch_mask: torch.Tensor | None,
    n_docs: int,
    batch: int,
) -> torch.Tensor | None:
    """Pad a ``(B, B)`` in-batch mask to ``(B, n_docs)`` with False for bank cols."""
    if batch_mask is None:
        return None
    if batch_mask.shape != (batch, batch):
        raise ValueError(
            f"non_negative_mask expected shape {(batch, batch)}, "
            f"got {tuple(batch_mask.shape)}"
        )
    if n_docs == batch:
        return batch_mask
    extra = n_docs - batch
    if extra < 0:
        raise ValueError(f"n_docs {n_docs} < batch {batch}")
    pad = torch.zeros(
        batch, extra, dtype=batch_mask.dtype, device=batch_mask.device
    )
    return torch.cat([batch_mask, pad], dim=1)


def contrastive_late_interaction_loss(
    text_tokens: list[torch.Tensor],
    image_tokens: list[torch.Tensor],
    temperature: float = 0.07,
    *,
    bank_text_tokens: list[torch.Tensor] | None = None,
    bank_image_tokens: list[torch.Tensor] | None = None,
    soft_maxsim_temperature: float | None = None,
    non_negative_mask: torch.Tensor | None = None,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Bidirectional late-interaction InfoNCE with optional memory-bank negatives.

    When the bank is empty this matches the historical square-matrix form
    (t2i on ``S``, i2t on ``S.T``). With a bank, each side scores the live batch
    positives first, then bank docs as extra negatives (labels stay in ``0..B-1``).

    Scores use mean-MaxSim (see ``differentiable_late_interaction_score``),
    optionally soft MaxSim. ``non_negative_mask`` is an optional ``(B, B)`` bool
    mask of in-batch pairs that must not act as negatives (false-neg softening).

    When ``return_metrics`` is True, also returns mean positive ranks (1 = best)
    averaged over t2i and i2t — useful as a bank-size-invariant health signal.
    """
    batch = len(text_tokens)
    if batch != len(image_tokens):
        raise ValueError(
            f"text/image batch size mismatch: {batch} vs {len(image_tokens)}"
        )
    bank_text = bank_text_tokens or []
    bank_image = bank_image_tokens or []
    if batch < 2 and not bank_text and not bank_image:
        raise ValueError(
            f"Contrastive loss needs batch_size >= 2 (got {batch}), "
            "or a non-empty memory bank for negatives."
        )
    if batch < 1:
        raise ValueError("Contrastive loss needs a non-empty batch.")

    labels = torch.arange(batch, device=text_tokens[0].device)
    n_image_docs = batch + len(bank_image)
    n_text_docs = batch + len(bank_text)
    soft_tau = soft_maxsim_temperature

    if not bank_text and not bank_image:
        scores = build_late_interaction_matrix(
            text_tokens,
            image_tokens,
            soft_maxsim_temperature=soft_tau,
        ) / temperature
        mask = _expand_non_negative_mask(non_negative_mask, batch, batch)
        loss_t2i = masked_cross_entropy(scores, labels, non_negative_mask=mask)
        # i2t uses the transpose; mask also transposed.
        mask_t = mask.T if mask is not None else None
        loss_i2t = masked_cross_entropy(scores.T, labels, non_negative_mask=mask_t)
        loss = 0.5 * (loss_t2i + loss_i2t)
        if not return_metrics:
            return loss
        rank_t2i = mean_positive_rank(scores.detach(), labels)
        rank_i2t = mean_positive_rank(scores.T.detach(), labels)
        return loss, {
            "pos_rank_t2i": rank_t2i,
            "pos_rank_i2t": rank_i2t,
            "pos_rank": 0.5 * (rank_t2i + rank_i2t),
            "n_image_docs": float(n_image_docs),
            "n_text_docs": float(n_text_docs),
        }

    image_docs = list(image_tokens) + list(bank_image)
    text_docs = list(text_tokens) + list(bank_text)
    scores_t2i = build_late_interaction_matrix(
        text_tokens, image_docs, soft_maxsim_temperature=soft_tau
    ) / temperature
    scores_i2t = build_late_interaction_matrix(
        image_tokens, text_docs, soft_maxsim_temperature=soft_tau
    ) / temperature
    mask_t2i = _expand_non_negative_mask(non_negative_mask, n_image_docs, batch)
    mask_i2t = _expand_non_negative_mask(
        non_negative_mask.T if non_negative_mask is not None else None,
        n_text_docs,
        batch,
    )
    loss_t2i = masked_cross_entropy(
        scores_t2i, labels, non_negative_mask=mask_t2i
    )
    loss_i2t = masked_cross_entropy(
        scores_i2t, labels, non_negative_mask=mask_i2t
    )
    loss = 0.5 * (loss_t2i + loss_i2t)
    if not return_metrics:
        return loss
    rank_t2i = mean_positive_rank(scores_t2i.detach(), labels)
    rank_i2t = mean_positive_rank(scores_i2t.detach(), labels)
    return loss, {
        "pos_rank_t2i": rank_t2i,
        "pos_rank_i2t": rank_i2t,
        "pos_rank": 0.5 * (rank_t2i + rank_i2t),
        "n_image_docs": float(n_image_docs),
        "n_text_docs": float(n_text_docs),
    }


def text_text_contrastive_loss(
    query_tokens: list[torch.Tensor],
    caption_tokens: list[torch.Tensor],
    distractor_tokens: list[torch.Tensor],
    temperature: float = 0.07,
    *,
    bank_doc_tokens: list[torch.Tensor] | None = None,
    soft_maxsim_temperature: float | None = None,
    non_negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reward query↔caption matches; penalize wrong captions, distractors, bank."""
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
    all_docs = list(caption_tokens) + list(distractor_tokens) + list(bank_docs)
    scores = build_late_interaction_matrix(
        query_tokens, all_docs, soft_maxsim_temperature=soft_maxsim_temperature
    ) / temperature
    labels = torch.arange(scores.size(0), device=scores.device)
    # Only the in-batch caption columns participate in multi-positive masking.
    mask = _expand_non_negative_mask(non_negative_mask, scores.size(1), batch)
    return masked_cross_entropy(scores, labels, non_negative_mask=mask)


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
) -> torch.Tensor:
    if dim_weights is None:
        dim_weights = [1.0] * len(dims)
    total_weight = sum(dim_weights)
    bank_raw = bank_doc_raw or []
    loss = query_raw[0].new_zeros(())
    for dim, weight in zip(dims, dim_weights):
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
) -> torch.Tensor:
    if dim_weights is None:
        dim_weights = [1.0] * len(dims)
    total_weight = sum(dim_weights)
    bank_text = bank_text_raw or []
    bank_image = bank_image_raw or []
    loss = text_raw[0].new_zeros(())
    for dim, weight in zip(dims, dim_weights):
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
        )
    return loss / total_weight


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
        matryoshka_weight: float = 0.5,
        text_text_weight: float = 1.0,
        text_text_matryoshka_weight: float = 0.5,
        compute_dtype: torch.dtype = torch.float16,
        memory_bank_size: int = DEFAULT_MEMORY_BANK_SIZE,
        soft_maxsim: bool = True,
        soft_maxsim_temperature: float = DEFAULT_SOFT_MAXSIM_TEMPERATURE,
        multi_positive_jaccard: float = DEFAULT_MULTI_POSITIVE_JACCARD,
        vision_patch_keep_ratio: float = DEFAULT_VISION_PATCH_KEEP_RATIO,
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
        geo_prefix_dim: int = DEFAULT_GEO_PREFIX_DIM,
        geo_prefix_weight: float = DEFAULT_GEO_PREFIX_WEIGHT,
        geo_ema_momentum: float = DEFAULT_GEO_EMA_MOMENTUM,
    ):
        super().__init__()
        self.vision_device = vision_device
        self.text_device = text_device
        self.loss_device = vision_device
        self.compute_dtype = compute_dtype

        self.vision_model = vision_model
        self.text_model = text_model
        self.vision_projection = nn.Linear(
            vision_hidden, embed_dim, device=vision_device, dtype=compute_dtype
        )
        self.text_projection = nn.Linear(
            text_hidden, embed_dim, device=text_device, dtype=compute_dtype
        )
        self.matryoshka_dims = matryoshka_dims
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.matryoshka_weight = matryoshka_weight
        self.text_text_weight = text_text_weight
        self.text_text_matryoshka_weight = text_text_matryoshka_weight
        self.memory_bank = EmbeddingMemoryBank(memory_bank_size)
        self.soft_maxsim = bool(soft_maxsim)
        self.soft_maxsim_temperature = float(soft_maxsim_temperature)
        self.multi_positive_jaccard = float(multi_positive_jaccard)
        self.vision_patch_keep_ratio = float(vision_patch_keep_ratio)
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
        self.geo_prefix_dim = int(geo_prefix_dim)
        self.geo_prefix_weight = float(geo_prefix_weight)
        self.geo_ema_momentum = float(geo_ema_momentum)
        # Running mean of normalized token embeds (fp32, loss device).
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

    def _init_projection_heads(self):
        for proj in (self.vision_projection, self.text_projection):
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
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

    def _select_vision_patches(
        self, vision_raw_i: torch.Tensor
    ) -> torch.Tensor:
        """Drop background patches by pre-norm L2 before MaxSim / bank store."""
        return keep_top_patches_by_l2(
            vision_raw_i, keep_ratio=self.vision_patch_keep_ratio
        )

    def _geo_kwargs(self) -> dict[str, float]:
        return {
            "center_weight": self.geo_center_weight,
            "var_weight": self.geo_var_weight,
            "vec_mean_weight": self.geo_vec_mean_weight,
            "var_ratio": self.geo_var_ratio,
            "mag_floor": self.geo_mag_floor,
            "mag_floor_weight": self.geo_mag_floor_weight,
            "max_abs_ratio": self.geo_max_abs_ratio,
            "max_abs_weight": self.geo_max_abs_weight,
        }

    def _compute_embedding_geometry(
        self,
        *,
        norm_token_lists: list[list[torch.Tensor]],
        raw_token_lists: list[list[torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Geometry loss on live tokens (+ optional Matryoshka prefix)."""
        zero = self.vision_projection.weight.new_zeros(())
        if self.embedding_geo_weight <= 0.0:
            return zero, {
                "geo_center": 0.0,
                "geo_var": 0.0,
                "geo_vec_mean": 0.0,
                "geo_mag_floor": 0.0,
                "geo_max_abs": 0.0,
                "geo_mu_norm": 0.0,
                "geo_min_std": 0.0,
                "geo_mean_abs_mu": 0.0,
                "geo_loss": 0.0,
            }

        # Token-level matrix (large N even with micro-batch 2) + sample mean-pools.
        token_norm = stack_token_embeddings(norm_token_lists)
        token_raw = stack_token_embeddings(raw_token_lists)
        pooled_parts = [
            mean_pool_token_list(lst) for lst in norm_token_lists if lst
        ]
        pooled_parts = [p for p in pooled_parts if p is not None]
        if token_norm is None and not pooled_parts:
            return zero, {
                "geo_center": 0.0,
                "geo_var": 0.0,
                "geo_vec_mean": 0.0,
                "geo_mag_floor": 0.0,
                "geo_max_abs": 0.0,
                "geo_mu_norm": 0.0,
                "geo_min_std": 0.0,
                "geo_mean_abs_mu": 0.0,
                "geo_loss": 0.0,
            }

        pieces: list[torch.Tensor] = []
        if token_norm is not None:
            pieces.append(token_norm)
        if pooled_parts:
            pieces.append(torch.cat(pooled_parts, dim=0))
        norm_mat = torch.cat(pieces, dim=0)

        raw_mat = None
        if token_raw is not None and token_norm is not None:
            # Align raw rows to token_norm only (not pooled rows).
            if token_raw.shape[0] == token_norm.shape[0]:
                # Pad raw with zeros for pooled rows so mag_floor only hits tokens.
                pad_n = norm_mat.shape[0] - token_raw.shape[0]
                if pad_n > 0:
                    pad = token_raw.new_zeros((pad_n, token_raw.shape[-1]))
                    # Use norms above floor so padded rows don't contribute.
                    pad = pad + float(self.geo_mag_floor) + 1.0
                    raw_mat = torch.cat([token_raw, pad], dim=0)
                else:
                    raw_mat = token_raw

        ema = None
        if self._geo_ema_initialized.item():
            ema = self._geo_ema_mean.to(device=norm_mat.device)

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
            # Raw prefix uses same slice when available.
            raw_prefix = None
            if raw_mat is not None:
                raw_prefix = raw_mat[..., :prefix_dim]
            ema_prefix = None
            if ema is not None:
                # Re-normalize EMA prefix as a soft center prior for the prefix space.
                ema_prefix = F.normalize(ema[:prefix_dim].float(), dim=-1)
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

        # EMA update from live batch mean (no grad).
        with torch.no_grad():
            batch_mu = norm_mat.detach().float().mean(dim=0)
            if batch_mu.device != self._geo_ema_mean.device:
                batch_mu = batch_mu.to(self._geo_ema_mean.device)
            if not self._geo_ema_initialized.item():
                self._geo_ema_mean.copy_(batch_mu)
                self._geo_ema_initialized.fill_(True)
            else:
                update_embedding_ema(
                    self._geo_ema_mean, batch_mu, momentum=self.geo_ema_momentum
                )

        metrics["geo_loss"] = float(geo.detach())
        return geo, metrics

    def encode_images(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        pixel_values = pixel_values.to(self.vision_device, non_blocking=True)
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
        return_loss: bool = True,
    ) -> dict[str, Any]:
        pixel_values = pixel_values.to(self.vision_device, non_blocking=True)
        input_ids = input_ids.to(self.text_device, non_blocking=True)
        attention_mask = attention_mask.to(self.text_device, non_blocking=True)

        vision_hidden = self.vision_model(
            pixel_values=pixel_values
        ).last_hidden_state.to(dtype=self.compute_dtype)
        text_hidden = self._text_backbone()(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.to(dtype=self.compute_dtype)
        vision_raw = self.vision_projection(
            vision_hidden.to(self.vision_projection.weight.dtype)
        )
        text_raw = self.text_projection(
            text_hidden.to(self.text_projection.weight.dtype)
        )

        # Vision: drop background patches (pre-norm L2) before normalize / bank.
        image_raw_kept: list[torch.Tensor] = []
        image_tokens: list[torch.Tensor] = []
        for i in range(vision_raw.size(0)):
            kept = self._select_vision_patches(vision_raw[i])
            image_raw_kept.append(kept)
            image_tokens.append(matryoshka_normalize(kept))

        text_tokens: list[torch.Tensor] = []
        text_raw_masked: list[torch.Tensor] = []
        for i in range(text_raw.size(0)):
            mask = attention_mask[i].bool()
            text_raw_masked.append(text_raw[i, mask])
            text_tokens.append(matryoshka_normalize(text_raw[i, mask]))

        if not return_loss:
            return {"text_embeddings": text_tokens, "image_embeddings": image_tokens}

        loss_text_tokens = [self._to_loss(t) for t in text_tokens]
        loss_image_tokens = [self._to_loss(t) for t in image_tokens]
        # Per-sample raw projected tokens (unnormalized) for Matryoshka + bank.
        # Image raw uses the same L2-kept patch subset as MaxSim scoring.
        loss_text_raw = [self._to_loss(t) for t in text_raw_masked]
        loss_image_raw = [self._to_loss(t) for t in image_raw_kept]

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
        non_neg = build_multi_positive_mask(
            captions,
            batch_size=len(loss_text_tokens),
            jaccard_threshold=self.multi_positive_jaccard,
            device=loss_text_tokens[0].device,
        )

        contrastive, contrastive_metrics = contrastive_late_interaction_loss(
            loss_text_tokens,
            loss_image_tokens,
            temperature=self.temperature,
            bank_text_tokens=bank_text_tokens,
            bank_image_tokens=bank_image_tokens,
            soft_maxsim_temperature=soft_tau,
            non_negative_mask=non_neg,
            return_metrics=True,
        )
        matryoshka = matryoshka_loss(
            loss_text_raw,
            loss_image_raw,
            dims=self.matryoshka_dims,
            temperature=self.temperature,
            bank_text_raw=bank_text_raw,
            bank_image_raw=bank_image_raw,
            soft_maxsim_temperature=soft_tau,
            non_negative_mask=non_neg,
        )
        loss = (
            self.contrastive_weight * contrastive
            + self.matryoshka_weight * matryoshka
        )

        text_text = contrastive.new_zeros(())
        text_text_matryoshka = contrastive.new_zeros(())
        loss_query_tokens: list[torch.Tensor] = []
        loss_query_raw: list[torch.Tensor] = []
        loss_distractor_tokens: list[torch.Tensor] = []
        loss_distractor_raw: list[torch.Tensor] = []
        has_text_text = (
            query_input_ids is not None
            and query_attention_mask is not None
            and unrelated_input_ids is not None
            and unrelated_attention_mask is not None
            and (
                self.text_text_weight > 0.0
                or self.text_text_matryoshka_weight > 0.0
            )
        )
        if has_text_text:
            query_tokens, query_raw = self._encode_text_batch(
                query_input_ids, query_attention_mask
            )
            distractor_tokens, distractor_raw = self._encode_text_batch(
                unrelated_input_ids, unrelated_attention_mask
            )
            loss_query_tokens = [self._to_loss(t) for t in query_tokens]
            loss_caption_tokens = loss_text_tokens
            loss_distractor_tokens = [self._to_loss(t) for t in distractor_tokens]
            loss_query_raw = [self._to_loss(t) for t in query_raw]
            loss_distractor_raw = [self._to_loss(t) for t in distractor_raw]

            if self.text_text_weight > 0.0:
                text_text = text_text_contrastive_loss(
                    loss_query_tokens,
                    loss_caption_tokens,
                    loss_distractor_tokens,
                    temperature=self.temperature,
                    bank_doc_tokens=bank_text_tokens,
                    soft_maxsim_temperature=soft_tau,
                    non_negative_mask=non_neg,
                )
                loss = loss + self.text_text_weight * text_text
            if self.text_text_matryoshka_weight > 0.0:
                text_text_matryoshka = text_text_matryoshka_loss(
                    loss_query_raw,
                    loss_text_raw,
                    loss_distractor_raw,
                    dims=self.matryoshka_dims,
                    temperature=self.temperature,
                    bank_doc_raw=bank_text_raw,
                    soft_maxsim_temperature=soft_tau,
                    non_negative_mask=non_neg,
                )
                loss = loss + self.text_text_matryoshka_weight * text_text_matryoshka

        # Geometry / anti-cone: text + image (+ query when available).
        # Queries showed strong dim-0 bias (~0.05) in retrieval probes.
        geo_norm_lists: list[list[torch.Tensor]] = [
            loss_text_tokens,
            loss_image_tokens,
        ]
        geo_raw_lists: list[list[torch.Tensor]] = [
            loss_text_raw,
            loss_image_raw,
        ]
        if loss_query_tokens:
            geo_norm_lists.append(loss_query_tokens)
            geo_raw_lists.append(loss_query_raw)
        # Detached bank tokens enlarge N for variance/center without polluting
        # live gradients (concat after stacking inside would need care); we only
        # use bank via EMA, which already tracks historical means.

        geo_loss, geo_metrics = self._compute_embedding_geometry(
            norm_token_lists=geo_norm_lists,
            raw_token_lists=geo_raw_lists,
        )
        if self.embedding_geo_weight > 0.0:
            loss = loss + self.embedding_geo_weight * geo_loss

        # Enqueue *after* scoring so the current batch is never its own negative.
        # Policy B: enqueue every micro-batch into the live FIFO.
        if bank.enabled:
            bank.enqueue(image_raw=loss_image_raw, text_raw=loss_text_raw)

        return {
            "loss": loss,
            "contrastive_loss": contrastive.detach(),
            "matryoshka_loss": matryoshka.detach(),
            "text_text_loss": text_text.detach(),
            "text_text_matryoshka_loss": text_text_matryoshka.detach(),
            "geo_loss": geo_loss.detach()
            if torch.is_tensor(geo_loss)
            else contrastive.new_zeros(()),
            "geo_mu_norm": geo_metrics.get("geo_mu_norm", 0.0),
            "geo_min_std": geo_metrics.get("geo_min_std", 0.0),
            "geo_mean_abs_mu": geo_metrics.get("geo_mean_abs_mu", 0.0),
            "geo_center": geo_metrics.get("geo_center", 0.0),
            "geo_var": geo_metrics.get("geo_var", 0.0),
            "memory_bank_size": len(bank),
            "pos_rank": contrastive_metrics["pos_rank"],
            "pos_rank_t2i": contrastive_metrics["pos_rank_t2i"],
            "pos_rank_i2t": contrastive_metrics["pos_rank_i2t"],
            "n_image_docs": contrastive_metrics["n_image_docs"],
            "n_text_docs": contrastive_metrics["n_text_docs"],
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
    print(
        f"Loading text model from {model_dir} on {text_device} "
        f"(8-bit weights via Unsloth"
        f"{', seed shell from ' + init_dir if quantized else ''}) ..."
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_qwen_backbone(
        model_dir,
        text_device,
        load_in_8bit=True,
        seed_dir=QWEN_DIR if quantized else None,
        tokenizer_id=tokenizer_id,
        max_seq_length=max_seq_length,
        for_training=True,
    )
    return model, tokenizer, model.config.hidden_size


def load_vision_model_for_training(
    model_dir: str,
    vision_device: torch.device,
    compute_dtype: torch.dtype,
):
    quantized = checkpoint_is_quantized(model_dir)
    init_dir = SIGLIP_DIR if quantized else model_dir
    print(
        f"Loading vision model from {model_dir} on {vision_device} "
        f"(8-bit weights"
        f"{', seed shell from ' + init_dir if quantized else ''}) ..."
    )
    model = load_siglip_backbone(
        model_dir,
        vision_device,
        load_in_8bit=True,
        seed_dir=SIGLIP_DIR if quantized else None,
        for_training=True,
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


def list_stage1_checkpoints(*, include_stage_root: bool = True) -> list[Path]:
    """Valid Stage-1 checkpoint roots (history/step-* and optional stage root)."""
    trained = Path(DEFAULT_TRAINED_DIR)
    candidates: list[Path] = []
    if include_stage_root:
        candidates.append(trained)
        candidates.append(Path(LEGACY_CHECKPOINT_DIR))
    history = trained / "history"
    if history.is_dir():
        candidates.extend(history.glob("step-*"))
    return [p for p in candidates if checkpoint_is_valid(p)]


def _checkpoint_candidates() -> list[Path]:
    return list_stage1_checkpoints(include_stage_root=True)


def find_latest_checkpoint() -> Path | None:
    """Most recent valid Stage-1 root among stage dir + history/step-* (by mtime)."""
    valid = list_stage1_checkpoints(include_stage_root=True)
    if not valid:
        return None
    return max(valid, key=lambda p: (p / PROJECTION_FILE).stat().st_mtime)


def find_latest_history_checkpoint() -> Path | None:
    """Latest mid-training snapshot under ``models/trained/stage1/history/step-*``.

    Ignores the completed/live stage root so demos can load the newest
    intermediate save even when ``stage1/`` itself is older or incomplete.
    """
    valid = list_stage1_checkpoints(include_stage_root=False)
    if not valid:
        return None
    return max(valid, key=_checkpoint_step_key)


def resolve_inference_checkpoint(
    *,
    phase: int = 1,
    checkpoint_dir: str | Path | None = None,
    latest_history: bool = False,
    latest_any: bool = False,
) -> Path | None:
    """Resolve a Stage-1 checkpoint root for inference/demos.

    Priority:
      1. Explicit ``checkpoint_dir``
      2. ``latest_history`` → newest ``history/step-*``
      3. ``latest_any`` → newest among stage root + history (mtime)
      4. ``None`` → caller uses phase-based ``models/trained/stage{N}/``
    """
    if checkpoint_dir:
        root = Path(checkpoint_dir)
        if not checkpoint_is_valid(root):
            raise FileNotFoundError(f"No valid Stage-1 checkpoint at {root}")
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
        root = Path(DEFAULT_TRAINED_DIR) if phase == 1 else Path(f"models/trained/stage{phase}")
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


def load_projection_heads(
    alignment_model: Stage1AlignmentModel,
    checkpoint_root: Path,
):
    state = _torch_load(
        checkpoint_root / PROJECTION_FILE, map_location="cpu", weights_only=True
    )
    alignment_model.vision_projection.load_state_dict(state["vision_projection"])
    alignment_model.text_projection.load_state_dict(state["text_projection"])
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


def build_optimizer(
    model: Stage1AlignmentModel,
    learning_rate: float,
    vision_learning_rate: float,
    projection_learning_rate: float,
    weight_decay: float,
):
    import bitsandbytes as bnb

    vision_params, text_params, projection_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vision_projection" in name or "text_projection" in name:
            projection_params.append(param)
        elif "vision_model" in name:
            vision_params.append(param)
        else:
            text_params.append(param)

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
    text_text = float(outputs.get("text_text_loss", 0.0))
    text_text_matryoshka = float(outputs.get("text_text_matryoshka_loss", 0.0))
    pos_rank = float(outputs.get("pos_rank", float("nan")))
    n_docs = int(outputs.get("n_image_docs", batch["input_ids"].shape[0]))
    model.train()
    print(
        f"Sanity check (batch={batch['input_ids'].shape[0]}): "
        f"loss={loss:.4f} contrastive={contrastive:.4f} matryoshka={matryoshka:.4f} "
        f"text_text={text_text:.4f} text_text_matryoshka={text_text_matryoshka:.4f} "
        f"pos_rank={pos_rank:.2f}/{n_docs}"
    )
    if loss <= 0.0 or not math.isfinite(loss):
        raise RuntimeError(
            "Loss is zero or non-finite before training. "
            "Increase --batch-size to at least 2."
        )


def run_training(
    model: Stage1AlignmentModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: Any,
    start_step: int = 0,
    *,
    tokenizer: Any = None,
    image_processor: Any = None,
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
    log_text_text = 0.0
    log_text_text_matryoshka = 0.0
    log_geo = 0.0
    log_geo_mu = 0.0
    log_geo_min_std = 0.0
    log_pos_rank = 0.0
    log_batches = 0

    if max_steps is not None and start_step >= max_steps:
        print(f"Already at step {start_step} (max_steps={max_steps}); nothing to train.")
        return start_step

    accum = max(int(args.gradient_accumulation_steps), 1)

    while True:
        for batch in dataloader:
            # Policy B: snapshot bank for scoring at the start of each accum window.
            if micro_step % accum == 0 and hasattr(model, "begin_accum_window"):
                model.begin_accum_window()

            outputs = model(**batch, return_loss=True)
            loss_val = outputs["loss"].detach().float().item()
            if loss_val <= 0.0 or not math.isfinite(loss_val):
                raise RuntimeError(
                    f"Invalid loss {loss_val} at micro-step {micro_step}. "
                    f"batch={batch['input_ids'].shape[0]} — need batch_size >= 2."
                )

            (outputs["loss"] / accum).backward()

            log_loss += loss_val
            log_contrastive += float(outputs["contrastive_loss"])
            log_matryoshka += float(outputs["matryoshka_loss"])
            log_text_text += float(outputs.get("text_text_loss", 0.0))
            log_text_text_matryoshka += float(
                outputs.get("text_text_matryoshka_loss", 0.0)
            )
            log_geo += float(outputs.get("geo_loss", 0.0))
            log_geo_mu += float(outputs.get("geo_mu_norm", 0.0))
            log_geo_min_std += float(outputs.get("geo_min_std", 0.0))
            log_pos_rank += float(outputs.get("pos_rank", 0.0))
            log_batches += 1
            micro_step += 1

            if micro_step % accum != 0:
                continue

            text_lr = lr_at(global_step)
            vision_lr = args.vision_learning_rate or text_lr
            schedule_scale = text_lr / max(args.learning_rate, 1e-12)
            proj_lr = args.projection_learning_rate * schedule_scale
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

            if global_step == 1 or global_step % args.logging_steps == 0:
                msg = (
                    f"step {global_step:5d} | "
                    f"loss {log_loss / log_batches:.4f} | "
                    f"contrastive {log_contrastive / log_batches:.4f} | "
                    f"matryoshka {log_matryoshka / log_batches:.4f}"
                )
                if log_text_text > 0.0 or getattr(args, "text_text_weight", 0.0) > 0.0:
                    msg += f" | text_text {log_text_text / log_batches:.4f}"
                if (
                    log_text_text_matryoshka > 0.0
                    or getattr(args, "text_text_matryoshka_weight", 0.0) > 0.0
                ):
                    msg += (
                        f" | text_text_m {log_text_text_matryoshka / log_batches:.4f}"
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
                msg += f" | grad_norm {float(grad_norm):.4f} | lr {text_lr:.2e}"
                print(msg)
                log_loss = log_contrastive = log_matryoshka = 0.0
                log_text_text = log_text_text_matryoshka = 0.0
                log_geo = log_geo_mu = log_geo_min_std = 0.0
                log_pos_rank = 0.0
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
