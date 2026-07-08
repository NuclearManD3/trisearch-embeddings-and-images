#!/usr/bin/env python3
"""
Stage 1: Seeding & Cross-Modal Alignment (training_plan.md §3)

Trains the SigLIP vision embedder and Qwen3-MoE text embedder jointly in a
shared 1024-dim Matryoshka space using:
  - ColBERT-style late-interaction contrastive loss (in-batch negatives)
  - Matryoshka loss across multiple embedding prefix dimensions

Unsloth loads the text tower in 8-bit (bitsandbytes) for memory savings.
The vision tower is also loaded in 8-bit. Each model sits on its own GPU;
loss is computed on the vision GPU.

Defaults use synthetic demo data so this runs immediately:
  python3 train_stage1.py --max-steps 200

For real datasets, pass --use-real-data.
"""

from __future__ import annotations

import os

# Unsloth's torch.compile hooks break bitsandbytes 8-bit SigLIP layers.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import argparse
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, BitsAndBytesConfig, SiglipVisionModel

from trisearch_models import (
    EMBED_DIM,
    QWEN_TOKENIZER_ID,
    SIGLIP_PROCESSOR_ID,
    matryoshka_normalize,
)

warnings.filterwarnings("ignore")

DEFAULT_VISION_DIR = "models/siglip-vision"
DEFAULT_TEXT_DIR = "models/qwen3-moe"
DEFAULT_OUTPUT_DIR = "checkpoints/stage1"
DEFAULT_MAX_TEXT_LENGTH = 256
DEFAULT_MATRYOSHKA_DIMS = (64, 128, 256, 512, 1024)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def late_interaction_score(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """Differentiable ColBERT MaxSim between token sets.

    query / doc : (n, D) and (m, D), L2-normalized along the last dim.
    Returns a scalar score = sum_i max_j <q_i, d_j>.
    """
    if query.numel() == 0 or doc.numel() == 0:
        return query.new_zeros(())
    sim = query @ doc.T
    return sim.max(dim=1).values.sum()


def build_late_interaction_matrix(
    text_tokens: list[torch.Tensor],
    image_tokens: list[torch.Tensor],
) -> torch.Tensor:
    """Pairwise late-interaction scores for a batch.

    Returns a (B, B) matrix where entry [i, j] scores text_i against image_j.
    Built with ``torch.stack`` so the full matrix stays in the autograd graph.
    """
    if len(text_tokens) < 2:
        raise ValueError(
            f"Contrastive loss needs batch_size >= 2 (got {len(text_tokens)}). "
            "In-batch negatives require at least one negative pair."
        )
    rows = []
    for text in text_tokens:
        rows.append(torch.stack([
            late_interaction_score(text, image) for image in image_tokens
        ]))
    return torch.stack(rows)


def contrastive_late_interaction_loss(
    text_tokens: list[torch.Tensor],
    image_tokens: list[torch.Tensor],
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE on late-interaction scores with in-batch negatives."""
    scores = build_late_interaction_matrix(text_tokens, image_tokens) / temperature
    labels = torch.arange(scores.size(0), device=scores.device)
    loss_t2i = F.cross_entropy(scores, labels)
    loss_i2t = F.cross_entropy(scores.T, labels)
    return 0.5 * (loss_t2i + loss_i2t)


def matryoshka_loss(
    text_raw: list[torch.Tensor],
    image_raw: list[torch.Tensor],
    dims: tuple[int, ...],
    temperature: float,
    dim_weights: list[float] | None = None,
) -> torch.Tensor:
    """Contrastive loss at multiple Matryoshka prefix dimensions."""
    if dim_weights is None:
        dim_weights = [1.0] * len(dims)
    total_weight = sum(dim_weights)
    loss = text_raw[0].new_zeros(())
    for dim, weight in zip(dims, dim_weights):
        text_prefix = [matryoshka_normalize(t, dim=dim) for t in text_raw]
        image_prefix = [matryoshka_normalize(t, dim=dim) for t in image_raw]
        loss = loss + weight * contrastive_late_interaction_loss(
            text_prefix, image_prefix, temperature=temperature
        )
    return loss / total_weight


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Stage1AlignmentModel(nn.Module):
    """Joint vision + text embedders with trainable Matryoshka projection heads.

    Vision and text towers live on separate GPUs; contrastive loss is computed
    on ``loss_device`` (the vision GPU) with text activations transferred over.
    """

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
        compute_dtype: torch.dtype = torch.float16,
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
        self._init_projection_heads()

    def _init_projection_heads(self):
        """Fresh 1024-dim heads need a real init — default Linear init is too small."""
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
        vision_raw = self.vision_projection(vision_hidden.to(self.vision_projection.weight.dtype))
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
        text_raw = self.text_projection(text_hidden.to(self.text_projection.weight.dtype))
        tokens: list[torch.Tensor] = []
        for i in range(text_raw.size(0)):
            mask = attention_mask[i].bool()
            tokens.append(matryoshka_normalize(text_raw[i, mask]))
        return tokens

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
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
        vision_raw = self.vision_projection(vision_hidden.to(self.vision_projection.weight.dtype))
        text_raw = self.text_projection(text_hidden.to(self.text_projection.weight.dtype))

        image_tokens = [matryoshka_normalize(vision_raw[i]) for i in range(vision_raw.size(0))]
        text_tokens: list[torch.Tensor] = []
        text_raw_masked: list[torch.Tensor] = []
        for i in range(text_raw.size(0)):
            mask = attention_mask[i].bool()
            text_raw_masked.append(text_raw[i, mask])
            text_tokens.append(matryoshka_normalize(text_raw[i, mask]))

        if not return_loss:
            return {"text_embeddings": text_tokens, "image_embeddings": image_tokens}

        # Compute loss on the vision GPU; move text-side activations across the bus.
        loss_text_tokens = [self._to_loss(t) for t in text_tokens]
        loss_image_tokens = [self._to_loss(t) for t in image_tokens]
        loss_text_raw = [self._to_loss(t) for t in text_raw_masked]
        loss_image_raw = [self._to_loss(t) for t in vision_raw]

        contrastive = contrastive_late_interaction_loss(
            loss_text_tokens, loss_image_tokens, temperature=self.temperature
        )
        matryoshka = matryoshka_loss(
            loss_text_raw,
            loss_image_raw,
            dims=self.matryoshka_dims,
            temperature=self.temperature,
        )
        loss = (
            self.contrastive_weight * contrastive
            + self.matryoshka_weight * matryoshka
        )
        return {
            "loss": loss,
            "contrastive_loss": contrastive.detach(),
            "matryoshka_loss": matryoshka.detach(),
        }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_pil_image(value: Any, image_root: str | None = None) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "bytes" in value:
        import io
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        import io
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        path = Path(value)
        if not path.is_file() and image_root:
            path = Path(image_root) / value
        if path.is_file():
            return Image.open(path).convert("RGB")
        raise FileNotFoundError(f"Image not found: {value}")
    raise TypeError(f"Unsupported image value type: {type(value)}")


def _pick_caption(row: dict[str, Any], caption_column: str) -> str:
    caption = row.get(caption_column, "")
    if isinstance(caption, list):
        caption = caption[0] if caption else ""
    return str(caption).strip()


@dataclass
class DataSourceConfig:
    dataset: str | None = None
    split: str = "train"
    image_column: str = "image"
    caption_column: str = "caption"
    image_root: str | None = None
    max_samples: int | None = None


class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        image_processor,
        tokenizer,
        image_column: str = "image",
        caption_column: str = "caption",
        image_root: str | None = None,
        max_text_length: int = 512,
    ):
        self.rows = rows
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_column = image_column
        self.caption_column = caption_column
        self.image_root = image_root
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image = _load_pil_image(row[self.image_column], self.image_root)
        caption = _pick_caption(row, self.caption_column)

        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"][0]
        text = self.tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_length,
            padding="max_length",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": text["input_ids"][0],
            "attention_mask": text["attention_mask"][0],
        }


class Stage1Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def _load_hf_rows(config: DataSourceConfig) -> list[dict[str, Any]]:
    from datasets import load_dataset

    print(f"Loading dataset {config.dataset!r} (split={config.split}) ...")
    ds = load_dataset(config.dataset, split=config.split, trust_remote_code=True)
    if config.max_samples is not None:
        ds = ds.select(range(min(config.max_samples, len(ds))))
    rows = [dict(ds[i]) for i in range(len(ds))]
    print(f"  -> {len(rows):,} rows")
    return rows


def _load_jsonl_rows(path: str, max_samples: int | None) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    print(f"Loaded {len(rows):,} rows from {path}")
    return rows


def build_demo_rows(count: int = 256) -> list[dict[str, Any]]:
    rng = random.Random(42)
    rows = []
    for i in range(count):
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        image = Image.new("RGB", (384, 384), color)
        rows.append({
            "image": image,
            "caption": f"A satellite view with dominant RGB color {color} (sample {i}).",
        })
    print(f"Built {len(rows):,} synthetic demo rows")
    return rows


def build_mixed_dataset(
    satellite_rows: list[dict[str, Any]],
    general_rows: list[dict[str, Any]],
    satellite_fraction: float,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if not satellite_rows and not general_rows:
        raise ValueError("No training rows available.")
    if not satellite_rows or not general_rows:
        return satellite_rows or general_rows

    total = min(len(satellite_rows), len(general_rows)) * 2
    n_sat = int(total * satellite_fraction)
    n_gen = total - n_sat
    rng = random.Random(seed)
    sat = rng.sample(satellite_rows, min(n_sat, len(satellite_rows)))
    gen = rng.sample(general_rows, min(n_gen, len(general_rows)))

    mixed: list[dict[str, Any]] = []
    sat_i = gen_i = 0
    sat_every = max(1, round(1.0 / satellite_fraction)) if satellite_fraction > 0 else 10**9
    while len(mixed) < total and (sat_i < len(sat) or gen_i < len(gen)):
        if sat_i < len(sat) and (len(mixed) % sat_every == 0 or gen_i >= len(gen)):
            mixed.append(sat[sat_i])
            sat_i += 1
        elif gen_i < len(gen):
            mixed.append(gen[gen_i])
            gen_i += 1
        else:
            break
    rng.shuffle(mixed)
    print(
        f"Mixed dataset: {len(mixed):,} rows "
        f"({satellite_fraction:.0%} satellite target)"
    )
    return mixed


# ---------------------------------------------------------------------------
# Model loading (8-bit full fine-tuning, one GPU per tower)
# ---------------------------------------------------------------------------

def _resolve_dtype(bf16: bool) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _gpu_device(index: int) -> torch.device:
    if torch.cuda.is_available():
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested GPU {index}, but only "
                f"{torch.cuda.device_count()} GPU(s) are available."
            )
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def load_text_model(
    model_dir: str,
    tokenizer_id: str,
    max_seq_length: int,
    text_device: torch.device,
    compute_dtype: torch.dtype,
):
    from transformers import AutoTokenizer
    from unsloth import FastLanguageModel

    device_map = {"": text_device.index} if text_device.type == "cuda" else "cpu"
    print(
        f"Loading text model from {model_dir} on {text_device} "
        f"(8-bit weights via Unsloth) ..."
    )
    # The resized checkpoint has weights only; tokenizer lives on the Hub.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        dtype=compute_dtype,
        full_finetuning=False,
        load_in_4bit=False,
        load_in_8bit=True,
        load_in_16bit=False,
        device_map=device_map,
        use_gradient_checkpointing="unsloth",
        tokenizer_name=tokenizer_id,
        fix_tokenizer=False,
    )
    model = FastLanguageModel.for_training(model)
    hidden_size = model.config.hidden_size
    return model, tokenizer, hidden_size


def load_vision_model(
    model_dir: str,
    vision_device: torch.device,
    compute_dtype: torch.dtype,
):
    device_map = {"": vision_device.index} if vision_device.type == "cuda" else "cpu"
    print(
        f"Loading vision model from {model_dir} on {vision_device} "
        f"(8-bit weights) ..."
    )
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = SiglipVisionModel.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map=device_map,
    )
    model.gradient_checkpointing_enable()
    hidden_size = model.config.hidden_size
    return model, hidden_size


def save_stage1_checkpoint(
    output_dir: str,
    alignment_model: Stage1AlignmentModel,
    args: argparse.Namespace,
):
    import shutil

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vision_dir = out / "vision_model"
    text_dir = out / "text_model"
    alignment_model.vision_model.save_pretrained(vision_dir)
    _save_text_checkpoint(alignment_model.text_model, text_dir, args.text_model_dir)

    torch.save(
        {
            "vision_projection": alignment_model.vision_projection.state_dict(),
            "text_projection": alignment_model.text_projection.state_dict(),
        },
        out / "projection_heads.pt",
    )
    with open(out / "stage1_config.json", "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, default=str)
    print(f"Saved Stage 1 checkpoint to {out}")


def _save_text_checkpoint(text_model: nn.Module, output_dir: Path, source_model_dir: str):
    """Save the Unsloth text model without tripping on non-JSON-serializable config."""
    import shutil

    import safetensors.torch

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        text_model.save_pretrained(str(output_dir))
        return
    except TypeError:
        pass

    safetensors.torch.save_file(
        {k: v.detach().cpu() for k, v in text_model.state_dict().items()},
        output_dir / "model.safetensors",
    )
    source_config = Path(source_model_dir) / "config.json"
    if source_config.is_file():
        shutil.copy(source_config, output_dir / "config.json")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

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


def _sanity_check_loss(model: Stage1AlignmentModel, dataloader: DataLoader):
    """Fail fast if the loss is zero/NaN before loading all 200 steps."""
    batch = next(iter(dataloader))
    model.eval()
    with torch.no_grad():
        outputs = model(**batch, return_loss=True)
    loss = float(outputs["loss"])
    contrastive = float(outputs["contrastive_loss"])
    matryoshka = float(outputs["matryoshka_loss"])
    model.train()
    print(
        f"Sanity check (batch={batch['input_ids'].shape[0]}): "
        f"loss={loss:.4f} contrastive={contrastive:.4f} matryoshka={matryoshka:.4f}"
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
    args: argparse.Namespace,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    max_steps = args.max_steps if args.max_steps > 0 else None
    total_steps = max_steps or (
        len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    )
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return args.learning_rate * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    global_step = 0
    micro_step = 0
    log_loss = 0.0
    log_contrastive = 0.0
    log_matryoshka = 0.0
    log_batches = 0

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
                print(
                    f"step {global_step:5d} | "
                    f"loss {log_loss / log_batches:.4f} | "
                    f"contrastive {log_contrastive / log_batches:.4f} | "
                    f"matryoshka {log_matryoshka / log_batches:.4f} | "
                    f"grad_norm {float(grad_norm):.4f} | "
                    f"lr {text_lr:.2e}"
                )
                log_loss = log_contrastive = log_matryoshka = 0.0
                log_batches = 0

            if global_step % args.save_steps == 0:
                save_stage1_checkpoint(args.output_dir, model, args)

            if max_steps is not None and global_step >= max_steps:
                return global_step

        if max_steps is None:
            break

    return global_step


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-model-dir", default=DEFAULT_VISION_DIR)
    parser.add_argument("--text-model-dir", default=DEFAULT_TEXT_DIR)
    parser.add_argument("--vision-processor-id", default=SIGLIP_PROCESSOR_ID)
    parser.add_argument("--text-tokenizer-id", default=QWEN_TOKENIZER_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--use-real-data", action="store_true",
                        help="Load HF satellite/general datasets instead of demo data.")
    parser.add_argument("--demo", action="store_true",
                        help="Alias for the default synthetic demo dataset.")
    parser.add_argument("--data-jsonl", default=None,
                        help="Local JSONL with image path + caption fields.")
    parser.add_argument("--image-root", default=None,
                        help="Base directory for relative image paths in JSONL.")

    parser.add_argument("--satellite-dataset", default="JessicaYuan/ChatEarthNet")
    parser.add_argument("--satellite-split", default="train")
    parser.add_argument("--satellite-image-column", default="image")
    parser.add_argument("--satellite-caption-column", default="caption")
    parser.add_argument("--satellite-image-root", default=None)

    parser.add_argument("--general-dataset", default="HuggingFaceM4/COCO")
    parser.add_argument("--general-split", default="train")
    parser.add_argument("--general-image-column", default="image")
    parser.add_argument("--general-caption-column", default="sentences")
    parser.add_argument("--satellite-fraction", type=float, default=0.5)

    parser.add_argument("--max-satellite-samples", type=int, default=None)
    parser.add_argument("--max-general-samples", type=int, default=None)
    parser.add_argument("--max-text-length", type=int, default=DEFAULT_MAX_TEXT_LENGTH)

    parser.add_argument("--matryoshka-dims", default="64,128,256,512,1024")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--matryoshka-weight", type=float, default=0.5)

    parser.add_argument("--vision-gpu", type=int, default=0,
                        help="GPU index for the SigLIP vision tower.")
    parser.add_argument("--text-gpu", type=int, default=1,
                        help="GPU index for the Qwen3-MoE text tower.")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Must be >=2 for in-batch contrastive negatives.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="LR for text tower + text projection head.")
    parser.add_argument("--projection-learning-rate", type=float, default=1e-4,
                        help="LR for the fresh 1024-dim projection heads.")
    parser.add_argument("--vision-learning-rate", type=float, default=None,
                        help="Optional LR for vision tower (defaults to --learning-rate).")
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")

    return parser.parse_args()


def _require_path(path: str, label: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{label} not found at {path!r}. "
            "Finish model initialization first or pass --demo for a smoke test."
        )


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    matryoshka_dims = tuple(
        int(x) for x in args.matryoshka_dims.split(",") if x.strip()
    )

    use_demo = not args.use_real_data or args.demo
    if use_demo and not args.data_jsonl:
        mixed_rows = build_demo_rows(count=256)
        image_column = "image"
        caption_column = "caption"
        image_root = None
    elif args.data_jsonl:
        mixed_rows = _load_jsonl_rows(args.data_jsonl, max_samples=None)
        image_column = "image"
        caption_column = "caption"
        image_root = args.image_root
    else:
        satellite_rows = _load_hf_rows(DataSourceConfig(
            dataset=args.satellite_dataset,
            split=args.satellite_split,
            image_column=args.satellite_image_column,
            caption_column=args.satellite_caption_column,
            image_root=args.satellite_image_root,
            max_samples=args.max_satellite_samples,
        ))
        general_rows = _load_hf_rows(DataSourceConfig(
            dataset=args.general_dataset,
            split=args.general_split,
            image_column=args.general_image_column,
            caption_column=args.general_caption_column,
            max_samples=args.max_general_samples,
        ))
        mixed_rows = build_mixed_dataset(
            satellite_rows, general_rows, args.satellite_fraction, seed=args.seed
        )
        image_column = args.satellite_image_column
        caption_column = args.satellite_caption_column
        image_root = args.satellite_image_root

    _require_path(args.text_model_dir, "Text model")
    _require_path(args.vision_model_dir, "Vision model")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be >= 2 for contrastive in-batch negatives.")

    compute_dtype = _resolve_dtype(args.bf16)
    vision_device = _gpu_device(args.vision_gpu)
    text_device = _gpu_device(args.text_gpu)

    text_model, tokenizer, text_hidden = load_text_model(
        model_dir=args.text_model_dir,
        tokenizer_id=args.text_tokenizer_id,
        max_seq_length=args.max_text_length,
        text_device=text_device,
        compute_dtype=compute_dtype,
    )
    vision_model, vision_hidden = load_vision_model(
        model_dir=args.vision_model_dir,
        vision_device=vision_device,
        compute_dtype=compute_dtype,
    )

    image_processor = AutoImageProcessor.from_pretrained(args.vision_processor_id)
    target_size = vision_model.config.image_size
    image_processor.size = {"height": target_size, "width": target_size}

    train_dataset = ImageCaptionDataset(
        rows=mixed_rows,
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_column=image_column,
        caption_column=caption_column,
        image_root=image_root,
        max_text_length=args.max_text_length,
    )

    alignment_model = Stage1AlignmentModel(
        vision_model=vision_model,
        text_model=text_model,
        vision_hidden=vision_hidden,
        text_hidden=text_hidden,
        vision_device=vision_device,
        text_device=text_device,
        matryoshka_dims=matryoshka_dims,
        temperature=args.temperature,
        contrastive_weight=args.contrastive_weight,
        matryoshka_weight=args.matryoshka_weight,
        compute_dtype=compute_dtype,
    )
    alignment_model.vision_projection.to(device=vision_device, dtype=compute_dtype)
    alignment_model.text_projection.to(device=text_device, dtype=compute_dtype)

    vision_lr = args.vision_learning_rate or args.learning_rate
    optimizer = build_optimizer(
        alignment_model,
        learning_rate=args.learning_rate,
        vision_learning_rate=vision_lr,
        projection_learning_rate=args.projection_learning_rate,
        weight_decay=args.weight_decay,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=Stage1Collator(pad_token_id=tokenizer.pad_token_id or 0),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    _sanity_check_loss(alignment_model, dataloader)

    print("\n--- Stage 1 training ---")
    print(f"  samples        : {len(train_dataset):,}")
    print(f"  effective batch: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  max tokens     : {args.max_text_length}")
    print(f"  matryoshka dims: {matryoshka_dims}")
    print(f"  compute dtype  : {compute_dtype}")
    print(f"  vision GPU     : {vision_device}")
    print(f"  text GPU       : {text_device}")
    print(f"  weight precision: 8-bit (bnb)")
    print(f"  optimizer      : AdamW8bit")
    print(f"  output         : {args.output_dir}\n")

    run_training(alignment_model, dataloader, optimizer, args)
    save_stage1_checkpoint(args.output_dir, alignment_model, args)


if __name__ == "__main__":
    main()