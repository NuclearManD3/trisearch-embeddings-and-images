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
DEFAULT_MAX_TEXT_LENGTH = 256
DEFAULT_MATRYOSHKA_DIMS = (64, 128, 256, 512, 1024)


def differentiable_late_interaction_score(
    query: torch.Tensor, doc: torch.Tensor
) -> torch.Tensor:
    """Mean-MaxSim: mean over query tokens of max cosine to any doc token.

    Using the mean (not sum) keeps logits O(1) for InfoNCE with temperature
    ~0.07 even when captions are long, avoiding softmax saturation and
    pathological gradient norms with a large memory bank.
    """
    if query.numel() == 0 or doc.numel() == 0:
        return query.new_zeros(())
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if doc.ndim == 1:
        doc = doc.unsqueeze(0)
    sim = query @ doc.T
    return sim.max(dim=1).values.mean()


def build_late_interaction_matrix(
    query_tokens: list[torch.Tensor],
    doc_tokens: list[torch.Tensor],
) -> torch.Tensor:
    """Pairwise mean-MaxSim matrix of shape ``(len(queries), len(docs))``."""
    if not query_tokens or not doc_tokens:
        raise ValueError(
            f"Late-interaction matrix needs non-empty query and doc lists "
            f"(got {len(query_tokens)} queries, {len(doc_tokens)} docs)."
        )
    rows = []
    for query in query_tokens:
        rows.append(torch.stack([
            differentiable_late_interaction_score(query, doc)
            for doc in doc_tokens
        ]))
    return torch.stack(rows)


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

    def normalized_images(self, dim: int | None = None) -> list[torch.Tensor]:
        return [matryoshka_normalize(t, dim=dim) for t in self._image_raw]

    def normalized_texts(self, dim: int | None = None) -> list[torch.Tensor]:
        return [matryoshka_normalize(t, dim=dim) for t in self._text_raw]


DEFAULT_MEMORY_BANK_SIZE = 128


def contrastive_late_interaction_loss(
    text_tokens: list[torch.Tensor],
    image_tokens: list[torch.Tensor],
    temperature: float = 0.07,
    *,
    bank_text_tokens: list[torch.Tensor] | None = None,
    bank_image_tokens: list[torch.Tensor] | None = None,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Bidirectional late-interaction InfoNCE with optional memory-bank negatives.

    When the bank is empty this matches the historical square-matrix form
    (t2i on ``S``, i2t on ``S.T``). With a bank, each side scores the live batch
    positives first, then bank docs as extra negatives (labels stay in ``0..B-1``).

    Scores use mean-MaxSim (see ``differentiable_late_interaction_score``).
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

    if not bank_text and not bank_image:
        scores = build_late_interaction_matrix(text_tokens, image_tokens) / temperature
        loss_t2i = F.cross_entropy(scores, labels)
        loss_i2t = F.cross_entropy(scores.T, labels)
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
    scores_t2i = (
        build_late_interaction_matrix(text_tokens, image_docs) / temperature
    )
    scores_i2t = (
        build_late_interaction_matrix(image_tokens, text_docs) / temperature
    )
    loss_t2i = F.cross_entropy(scores_t2i, labels)
    loss_i2t = F.cross_entropy(scores_i2t, labels)
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
    all_docs = list(caption_tokens) + list(distractor_tokens) + list(bank_docs)
    scores = build_late_interaction_matrix(query_tokens, all_docs) / temperature
    labels = torch.arange(scores.size(0), device=scores.device)
    return F.cross_entropy(scores, labels)


def text_text_matryoshka_loss(
    query_raw: list[torch.Tensor],
    caption_raw: list[torch.Tensor],
    distractor_raw: list[torch.Tensor],
    dims: tuple[int, ...],
    temperature: float,
    dim_weights: list[float] | None = None,
    *,
    bank_doc_raw: list[torch.Tensor] | None = None,
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
        self._init_projection_heads()

    def _init_projection_heads(self):
        for proj in (self.vision_projection, self.text_projection):
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _text_backbone(self):
        return getattr(self.text_model, "model", self.text_model)

    def _to_loss(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.loss_device, dtype=self.compute_dtype)

    def encode_images(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        pixel_values = pixel_values.to(self.vision_device, non_blocking=True)
        vision_hidden = self.vision_model(
            pixel_values=pixel_values
        ).last_hidden_state.to(dtype=self.compute_dtype)
        vision_raw = self.vision_projection(
            vision_hidden.to(self.vision_projection.weight.dtype)
        )
        return [matryoshka_normalize(vision_raw[i]) for i in range(vision_raw.size(0))]

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

        image_tokens = [
            matryoshka_normalize(vision_raw[i]) for i in range(vision_raw.size(0))
        ]
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
        loss_text_raw = [self._to_loss(t) for t in text_raw_masked]
        loss_image_raw = [
            self._to_loss(vision_raw[i]) for i in range(vision_raw.size(0))
        ]

        bank = self.memory_bank
        bank_text_raw = (
            [self._to_loss(t) for t in bank.text_raw()] if bank.enabled else []
        )
        bank_image_raw = (
            [self._to_loss(t) for t in bank.image_raw()] if bank.enabled else []
        )
        bank_text_tokens = (
            [matryoshka_normalize(t) for t in bank_text_raw] if bank_text_raw else []
        )
        bank_image_tokens = (
            [matryoshka_normalize(t) for t in bank_image_raw] if bank_image_raw else []
        )

        contrastive, contrastive_metrics = contrastive_late_interaction_loss(
            loss_text_tokens,
            loss_image_tokens,
            temperature=self.temperature,
            bank_text_tokens=bank_text_tokens,
            bank_image_tokens=bank_image_tokens,
            return_metrics=True,
        )
        matryoshka = matryoshka_loss(
            loss_text_raw,
            loss_image_raw,
            dims=self.matryoshka_dims,
            temperature=self.temperature,
            bank_text_raw=bank_text_raw,
            bank_image_raw=bank_image_raw,
        )
        loss = (
            self.contrastive_weight * contrastive
            + self.matryoshka_weight * matryoshka
        )

        text_text = contrastive.new_zeros(())
        text_text_matryoshka = contrastive.new_zeros(())
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
                )
                loss = loss + self.text_text_matryoshka_weight * text_text_matryoshka

        # Enqueue *after* scoring so the current batch is never its own negative.
        if bank.enabled:
            bank.enqueue(image_raw=loss_image_raw, text_raw=loss_text_raw)

        return {
            "loss": loss,
            "contrastive_loss": contrastive.detach(),
            "matryoshka_loss": matryoshka.detach(),
            "text_text_loss": text_text.detach(),
            "text_text_matryoshka_loss": text_text_matryoshka.detach(),
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


def _checkpoint_candidates() -> list[Path]:
    trained = Path(DEFAULT_TRAINED_DIR)
    candidates = [trained, Path(LEGACY_CHECKPOINT_DIR)]
    history = trained / "history"
    if history.is_dir():
        candidates.extend(sorted(history.glob("step-*")))
    return candidates


def find_latest_checkpoint() -> Path | None:
    valid = [p for p in _checkpoint_candidates() if checkpoint_is_valid(p)]
    if not valid:
        return None
    return max(valid, key=lambda p: (p / PROJECTION_FILE).stat().st_mtime)


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
    max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
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

    rows = load_verification_samples()
    processor = AutoImageProcessor.from_pretrained(vision_processor_id)
    target_size = vision_model.config.image_size
    processor.size = {"height": target_size, "width": target_size}
    verify_rows = rows
    if with_text_queries:
        from trisearch_dataset import enrich_rows_with_text_queries

        verify_rows = enrich_rows_with_text_queries(
            rows,
            max_new_queries=8,
            skip_generation=False,
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
    log_pos_rank = 0.0
    log_batches = 0

    if max_steps is not None and start_step >= max_steps:
        print(f"Already at step {start_step} (max_steps={max_steps}); nothing to train.")
        return start_step

    while True:
        for batch in dataloader:
            outputs = model(**batch, return_loss=True)
            loss_val = outputs["loss"].detach().float().item()
            if loss_val <= 0.0 or not math.isfinite(loss_val):
                raise RuntimeError(
                    f"Invalid loss {loss_val} at micro-step {micro_step}. "
                    f"batch={batch['input_ids'].shape[0]} — need batch_size >= 2."
                )

            (outputs["loss"] / args.gradient_accumulation_steps).backward()

            log_loss += loss_val
            log_contrastive += float(outputs["contrastive_loss"])
            log_matryoshka += float(outputs["matryoshka_loss"])
            log_text_text += float(outputs.get("text_text_loss", 0.0))
            log_text_text_matryoshka += float(
                outputs.get("text_text_matryoshka_loss", 0.0)
            )
            log_pos_rank += float(outputs.get("pos_rank", 0.0))
            log_batches += 1
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps != 0:
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
                bank_len = len(getattr(model, "memory_bank", ()))
                if bank_len or getattr(args, "memory_bank_size", 0):
                    msg += f" | bank {bank_len}"
                n_docs = int(outputs.get("n_image_docs", args.batch_size))
                msg += f" | pos_rank {log_pos_rank / log_batches:.1f}/{n_docs}"
                msg += f" | grad_norm {float(grad_norm):.4f} | lr {text_lr:.2e}"
                print(msg)
                log_loss = log_contrastive = log_matryoshka = 0.0
                log_text_text = log_text_text_matryoshka = 0.0
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
