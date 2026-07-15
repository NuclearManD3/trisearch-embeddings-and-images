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
# Prefix dims only — full embed dim is trained by the main contrastive terms.
# Including 1024 here double-counted full-dim CE and starved small prefixes.
DEFAULT_MATRYOSHKA_DIMS = (64, 128, 256, 512)
# Soft MaxSim: τ_s * logsumexp(sim / τ_s). Smaller τ_s → closer to hard max.
DEFAULT_SOFT_MAXSIM_TEMPERATURE = 0.05
# Caption token-Jaccard above this → treat as non-negative (not a false neg).
DEFAULT_MULTI_POSITIVE_JACCARD = 0.5
# Keep top fraction of SigLIP patches by pre-norm L2 (drop background).
DEFAULT_VISION_PATCH_KEEP_RATIO = 0.75
# Train-only: randomly drop this fraction of remaining patches after L2 keep.
DEFAULT_VISION_PATCH_DROP_PROB = 0.40
# Train-only: random integer translate in [-max, max] pixels (reflect pad).
DEFAULT_IMAGE_SHIFT_MAX = 18
# Train-only geometric aug (flip / rotate / mild scale-stretch + pad fill).
DEFAULT_IMAGE_HFLIP_PROB = 0.5
DEFAULT_IMAGE_MAX_ROTATE_DEG = 30.0
DEFAULT_IMAGE_SCALE_MIN = 0.85
DEFAULT_IMAGE_SCALE_MAX = 1.05
# "random" | "mean" | "reflect" fill for rotate/scale gaps.
DEFAULT_IMAGE_FILL_MODE = "random"
# Penalize diffuse positive-pair heatmaps (normalized entropy of patch MaxSim).
DEFAULT_HEATMAP_SPARSITY_WEIGHT = 0.1
DEFAULT_HEATMAP_SPARSITY_TEMPERATURE = 0.07
# Top-k hardest bank docs kept per query as InfoNCE negatives (0 = use full bank).
DEFAULT_HARD_BANK_NEGATIVES = 32
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


def random_shift_pixel_values(
    pixel_values: torch.Tensor,
    max_shift: int = DEFAULT_IMAGE_SHIFT_MAX,
) -> torch.Tensor:
    """Per-image integer shift in ``[-max_shift, max_shift]`` with reflect pad.

    ``pixel_values`` is ``(B, C, H, W)``. Each sample draws independent ``(dy, dx)``.
    Used only during training to reduce grid-position memorization (sub-patch /
    one-patch scale when max_shift ≈ patch size).
    """
    if max_shift is None or int(max_shift) <= 0:
        return pixel_values
    if pixel_values.ndim != 4:
        raise ValueError(
            f"pixel_values must be (B,C,H,W), got shape {tuple(pixel_values.shape)}"
        )
    max_shift = int(max_shift)
    b, _, h, w = pixel_values.shape
    pad = max_shift
    # (left, right, top, bottom)
    padded = F.pad(pixel_values, (pad, pad, pad, pad), mode="reflect")
    out = pixel_values.new_empty(pixel_values.shape)
    for i in range(b):
        dy = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        dx = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        y0 = pad + dy
        x0 = pad + dx
        out[i] = padded[i, :, y0 : y0 + h, x0 : x0 + w]
    return out


def _sample_fill_color(
    image_chw: torch.Tensor,
    mode: str = DEFAULT_IMAGE_FILL_MODE,
) -> list[float]:
    """Per-channel fill for gaps (normalized tensor space).

    * ``random`` — uniform in an expanded range around the image min/max
    * ``mean`` — channel means of the image
    * ``reflect`` — unused here (caller uses reflect pad); falls back to mean
    """
    c = int(image_chw.shape[0])
    mode = (mode or "random").lower()
    if mode == "reflect":
        mode = "mean"
    if mode == "mean":
        return [float(image_chw[ch].mean()) for ch in range(c)]
    # random: sample per channel in [lo-margin, hi+margin]
    fills: list[float] = []
    for ch in range(c):
        lo = float(image_chw[ch].min())
        hi = float(image_chw[ch].max())
        span = max(hi - lo, 1e-3)
        lo_e, hi_e = lo - 0.25 * span, hi + 0.25 * span
        fills.append(float(lo_e + (hi_e - lo_e) * torch.rand(1).item()))
    return fills


def _fit_chw_to_size(
    image_chw: torch.Tensor,
    out_h: int,
    out_w: int,
    fill: list[float],
) -> torch.Tensor:
    """Center-pad (shrink) or center-crop (mild expand) a ``(C,H,W)`` tensor."""
    c, h, w = image_chw.shape
    # Pad if smaller.
    pad_top = max(0, (out_h - h) // 2)
    pad_bottom = max(0, out_h - h - pad_top)
    pad_left = max(0, (out_w - w) // 2)
    pad_right = max(0, out_w - w - pad_left)
    if pad_top or pad_bottom or pad_left or pad_right:
        # F.pad on (C,H,W) uses last dims: (left, right, top, bottom)
        # pad value: use mean fill broadcast — pad with zeros then paint.
        image_chw = F.pad(
            image_chw,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )
        if pad_top:
            for ch, v in enumerate(fill):
                image_chw[ch, :pad_top, :] = v
        if pad_bottom:
            for ch, v in enumerate(fill):
                image_chw[ch, image_chw.shape[1] - pad_bottom :, :] = v
        if pad_left:
            for ch, v in enumerate(fill):
                image_chw[ch, :, :pad_left] = v
        if pad_right:
            for ch, v in enumerate(fill):
                image_chw[ch, :, image_chw.shape[2] - pad_right :] = v
        h, w = image_chw.shape[1], image_chw.shape[2]
    # Center-crop if larger (limited expand path).
    if h > out_h or w > out_w:
        top = max(0, (h - out_h) // 2)
        left = max(0, (w - out_w) // 2)
        image_chw = image_chw[:, top : top + out_h, left : left + out_w]
    # Exact size guard (rounding).
    if image_chw.shape[1] != out_h or image_chw.shape[2] != out_w:
        image_chw = F.interpolate(
            image_chw.unsqueeze(0),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return image_chw


def train_image_geometric_augment(
    pixel_values: torch.Tensor,
    *,
    hflip_prob: float = DEFAULT_IMAGE_HFLIP_PROB,
    max_rotate_deg: float = DEFAULT_IMAGE_MAX_ROTATE_DEG,
    scale_min: float = DEFAULT_IMAGE_SCALE_MIN,
    scale_max: float = DEFAULT_IMAGE_SCALE_MAX,
    fill_mode: str = DEFAULT_IMAGE_FILL_MODE,
) -> torch.Tensor:
    """Train-time geometric aug: H-flip, rotate, anisotropic scale, pad/crop.

    ``pixel_values`` is ``(B, C, H, W)`` (typically processor-normalized).

    Design (content-preserving):
    * Horizontal flip with probability ``hflip_prob``.
    * Rotation uniform in ``[-max_rotate_deg, max_rotate_deg]`` (same output size;
      corners filled).
    * Independent ``scale_x, scale_y`` in ``[scale_min, scale_max]``. Shrink pads
      with fill; mild expand center-crops so labels stay mostly valid
      (``scale_max`` should stay near 1.05–1.10).
    * Fill: ``random`` / ``mean`` channel colors in tensor space (not raw RGB).

    Does **not** apply integer pixel shift — compose with
    ``random_shift_pixel_values`` separately.
    """
    if pixel_values.ndim != 4:
        raise ValueError(
            f"pixel_values must be (B,C,H,W), got shape {tuple(pixel_values.shape)}"
        )
    b, c, h, w = pixel_values.shape
    smin = float(scale_min)
    smax = float(scale_max)
    if smin <= 0 or smax <= 0 or smin > smax:
        raise ValueError(f"invalid scale range [{smin}, {smax}]")
    max_rot = abs(float(max_rotate_deg))
    hflip_p = float(hflip_prob)

    # torchvision is the clean path for rotate on CHW batches.
    try:
        import torchvision.transforms.functional as tvf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "train_image_geometric_augment requires torchvision"
        ) from exc

    out = pixel_values.new_empty(pixel_values.shape)
    for i in range(b):
        img = pixel_values[i]
        fill = _sample_fill_color(img, mode=fill_mode)

        if hflip_p > 0.0 and torch.rand(1).item() < hflip_p:
            img = tvf.hflip(img)

        if max_rot > 0.0:
            angle = float((-max_rot) + (2.0 * max_rot) * torch.rand(1).item())
            if abs(angle) > 1e-6:
                # fill: sequence length C for multi-channel.
                img = tvf.rotate(
                    img,
                    angle=angle,
                    interpolation=tvf.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=fill,
                )

        # Anisotropic scale (independent x/y).
        sx = float(smin + (smax - smin) * torch.rand(1).item())
        sy = float(smin + (smax - smin) * torch.rand(1).item())
        if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
            new_w = max(1, int(round(w * sx)))
            new_h = max(1, int(round(h * sy)))
            img = tvf.resize(
                img,
                [new_h, new_w],
                interpolation=tvf.InterpolationMode.BILINEAR,
                antialias=True,
            )
            img = _fit_chw_to_size(img, h, w, fill=fill)

        out[i] = img
    return out


def apply_train_image_augmentations(
    pixel_values: torch.Tensor,
    *,
    hflip_prob: float = DEFAULT_IMAGE_HFLIP_PROB,
    max_rotate_deg: float = DEFAULT_IMAGE_MAX_ROTATE_DEG,
    scale_min: float = DEFAULT_IMAGE_SCALE_MIN,
    scale_max: float = DEFAULT_IMAGE_SCALE_MAX,
    fill_mode: str = DEFAULT_IMAGE_FILL_MODE,
    max_shift: int = DEFAULT_IMAGE_SHIFT_MAX,
    enabled: bool = True,
) -> torch.Tensor:
    """Full train-time vision aug stack: geometric → integer shift."""
    if not enabled:
        return pixel_values
    x = train_image_geometric_augment(
        pixel_values,
        hflip_prob=hflip_prob,
        max_rotate_deg=max_rotate_deg,
        scale_min=scale_min,
        scale_max=scale_max,
        fill_mode=fill_mode,
    )
    if max_shift and int(max_shift) > 0:
        x = random_shift_pixel_values(x, max_shift=int(max_shift))
    return x


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
) -> torch.Tensor:
    """Keep only the top-``hard_k`` hardest *bank* columns per row.

    ``scores`` is ``(B, n_live + n_bank)`` with live docs in ``[:, :n_live]``.
    Live columns are always kept (in-batch negatives + the labeled positive).
    Bank columns outside each row's top-k hardest scores are set to ``-inf`` so
    they drop out of InfoNCE — easy bank negatives no longer dilute the softmax.

    ``hard_k <= 0`` or ``hard_k >= n_bank`` leaves scores unchanged.
    """
    if hard_k is None or int(hard_k) <= 0:
        return scores
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {tuple(scores.shape)}")
    batch, n_docs = scores.shape
    if n_live < 0 or n_live > n_docs:
        raise ValueError(f"n_live={n_live} invalid for n_docs={n_docs}")
    n_bank = n_docs - n_live
    if n_bank <= 0:
        return scores
    k = min(int(hard_k), n_bank)
    if k >= n_bank:
        return scores

    live = scores[:, :n_live]
    bank = scores[:, n_live:]
    topk_idx = torch.topk(bank, k=k, dim=1).indices  # (B, k)
    keep = torch.zeros_like(bank, dtype=torch.bool)
    keep.scatter_(1, topk_idx, True)
    bank = bank.masked_fill(~keep, float("-inf"))
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
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Bidirectional late-interaction InfoNCE with optional memory-bank negatives.

    When the bank is empty this matches the historical square-matrix form
    (t2i on ``S``, i2t on ``S.T``). With a bank, each side scores the live batch
    positives first, then bank docs as extra negatives (labels stay in ``0..B-1``).

    Scores use mean-MaxSim (see ``differentiable_late_interaction_score``),
    optionally soft MaxSim. ``non_negative_mask`` is an optional ``(B, B)`` bool
    mask of in-batch pairs that must not act as negatives (false-neg softening).

    ``hard_bank_k`` keeps only the top-k hardest bank docs per query (see
    ``apply_hard_bank_mining``); live in-batch columns are always kept.

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
    scores_t2i = apply_hard_bank_mining(scores_t2i, batch, hard_bank_k)
    scores_i2t = apply_hard_bank_mining(scores_i2t, batch, hard_bank_k)
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
    hard_bank_k: int = 0,
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
    n_live = batch + len(distractor_tokens)
    all_docs = list(caption_tokens) + list(distractor_tokens) + list(bank_docs)
    scores = build_late_interaction_matrix(
        query_tokens, all_docs, soft_maxsim_temperature=soft_maxsim_temperature
    ) / temperature
    scores = apply_hard_bank_mining(scores, n_live, hard_bank_k)
    labels = torch.arange(scores.size(0), device=scores.device)
    # Only the in-batch caption columns participate in multi-positive masking.
    mask = _expand_non_negative_mask(
        non_negative_mask, scores.size(1), batch, n_live=n_live
    )
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
    hard_bank_k: int = 0,
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
    embed_dim: int = EMBED_DIM,
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
        compute_dtype: torch.dtype = torch.float16,
        memory_bank_size: int = DEFAULT_MEMORY_BANK_SIZE,
        soft_maxsim: bool = True,
        soft_maxsim_temperature: float = DEFAULT_SOFT_MAXSIM_TEMPERATURE,
        multi_positive_jaccard: float = DEFAULT_MULTI_POSITIVE_JACCARD,
        vision_patch_keep_ratio: float = DEFAULT_VISION_PATCH_KEEP_RATIO,
        vision_patch_drop_prob: float = DEFAULT_VISION_PATCH_DROP_PROB,
        image_shift_max: int = DEFAULT_IMAGE_SHIFT_MAX,
        image_hflip_prob: float = DEFAULT_IMAGE_HFLIP_PROB,
        image_max_rotate_deg: float = DEFAULT_IMAGE_MAX_ROTATE_DEG,
        image_scale_min: float = DEFAULT_IMAGE_SCALE_MIN,
        image_scale_max: float = DEFAULT_IMAGE_SCALE_MAX,
        image_fill_mode: str = DEFAULT_IMAGE_FILL_MODE,
        image_aug_enabled: bool = True,
        heatmap_sparsity_weight: float = DEFAULT_HEATMAP_SPARSITY_WEIGHT,
        heatmap_sparsity_temperature: float = DEFAULT_HEATMAP_SPARSITY_TEMPERATURE,
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
        self.memory_bank = EmbeddingMemoryBank(memory_bank_size)
        self.soft_maxsim = bool(soft_maxsim)
        self.soft_maxsim_temperature = float(soft_maxsim_temperature)
        self.multi_positive_jaccard = float(multi_positive_jaccard)
        self.vision_patch_keep_ratio = float(vision_patch_keep_ratio)
        self.vision_patch_drop_prob = float(vision_patch_drop_prob)
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
        self.heatmap_sparsity_weight = float(heatmap_sparsity_weight)
        self.heatmap_sparsity_temperature = float(heatmap_sparsity_temperature)
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
        """L2 background keep, then train-only random patch dropout."""
        kept = keep_top_patches_by_l2(
            vision_raw_i, keep_ratio=self.vision_patch_keep_ratio
        )
        return random_drop_patches(
            kept,
            drop_prob=self.vision_patch_drop_prob,
            training=self.training,
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

    def _maybe_augment_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Train-only geometric + shift aug; identity in eval."""
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
            enabled=True,
        )

    def encode_images(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        pixel_values = pixel_values.to(self.vision_device, non_blocking=True)
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

        # Train-only: flip / rotate / mild scale-stretch + pad fill + pixel shift.
        pixel_values = self._maybe_augment_images(pixel_values)

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
        hard_k = self.hard_bank_negatives
        non_neg = build_multi_positive_mask(
            captions,
            batch_size=len(loss_text_tokens),
            jaccard_threshold=self.multi_positive_jaccard,
            device=loss_text_tokens[0].device,
        )

        # ------------------------------------------------------------------
        # Retrieval tasks (all InfoNCE on shared live embeddings):
        #   1) caption ↔ image  — every caption matches its image, not others
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
        contrastive, contrastive_metrics = contrastive_late_interaction_loss(
            loss_text_tokens,
            loss_image_tokens,
            temperature=self.temperature,
            bank_text_tokens=bank_text_tokens,
            bank_image_tokens=bank_image_tokens,
            soft_maxsim_temperature=soft_tau,
            non_negative_mask=non_neg,
            hard_bank_k=hard_k,
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
            hard_bank_k=hard_k,
            embed_dim=self.embed_dim,
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
                query_image = contrastive_late_interaction_loss(
                    loss_query_tokens,
                    loss_image_tokens,
                    temperature=self.temperature,
                    bank_text_tokens=bank_text_tokens,
                    bank_image_tokens=bank_image_tokens,
                    soft_maxsim_temperature=soft_tau,
                    non_negative_mask=non_neg,
                    hard_bank_k=hard_k,
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
                    non_negative_mask=non_neg,
                    hard_bank_k=hard_k,
                    embed_dim=self.embed_dim,
                )

            if want_query_caption:
                # Query → matching caption; not other captions / distractors / bank.
                text_text = text_text_contrastive_loss(
                    loss_query_tokens,
                    loss_text_tokens,
                    loss_distractor_tokens,
                    temperature=self.temperature,
                    bank_doc_tokens=bank_text_tokens,
                    soft_maxsim_temperature=soft_tau,
                    non_negative_mask=non_neg,
                    hard_bank_k=hard_k,
                )
                text_text_matryoshka = text_text_matryoshka_loss(
                    loss_query_raw,
                    loss_text_raw,
                    loss_distractor_raw,
                    dims=self.matryoshka_dims,
                    temperature=self.temperature,
                    bank_doc_raw=bank_text_raw,
                    soft_maxsim_temperature=soft_tau,
                    non_negative_mask=non_neg,
                    hard_bank_k=hard_k,
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
        sparsity = zero
        if self.heatmap_sparsity_weight > 0.0:
            sp_terms: list[torch.Tensor] = [
                heatmap_sparsity_loss(
                    loss_text_tokens,
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
            sparsity = torch.stack(sp_terms).mean()
            loss = loss + self.heatmap_sparsity_weight * sparsity

        # Enqueue *after* scoring so the current batch is never its own negative.
        # Policy B: enqueue every micro-batch into the live FIFO.
        if bank.enabled:
            bank.enqueue(image_raw=loss_image_raw, text_raw=loss_text_raw)

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
            "heatmap_sparsity_loss": sparsity.detach(),
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
    log_query_image = 0.0
    log_text_text = 0.0
    log_text_text_matryoshka = 0.0
    log_heatmap_sparsity = 0.0
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
                msg += f" | grad_norm {float(grad_norm):.4f} | lr {text_lr:.2e}"
                print(msg)
                log_loss = log_contrastive = log_matryoshka = 0.0
                log_query_image = 0.0
                log_text_text = log_text_text_matryoshka = 0.0
                log_heatmap_sparsity = 0.0
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
