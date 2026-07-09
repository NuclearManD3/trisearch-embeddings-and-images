#!/usr/bin/env python3
"""
Stage 1: Seeding & Cross-Modal Alignment (training_plan.md §3)

Trains the SigLIP vision embedder and Qwen3-MoE text embedder jointly in a
shared 1024-dim Matryoshka space using ColBERT-style contrastive loss and
Matryoshka loss. Both towers are loaded in 8-bit via Unsloth/bitsandbytes.

Training data is always real image–caption pairs (HuggingFace datasets or
local JSONL). Checkpoints are written under models/trained/stage1/.

  python3 train_stage1.py --max-steps 10000 --batch-size 4

Pass --fresh to train from seed weights instead of resuming.
"""

from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
#os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import argparse
import logging
import random
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from transformers.utils import logging as transformers_logging

#transformers_logging.set_verbosity_error()
#logging.getLogger("transformers").setLevel(logging.ERROR)
#warnings.filterwarnings("ignore")

from trisearch_dataset import (
    DEFAULT_CURATED_DATASET_DIR,
    DEFAULT_GENERAL_CAPTION_COLUMN,
    DEFAULT_GENERAL_DATASET,
    DEFAULT_GENERAL_SPLIT,
    DEFAULT_OPENROUTER_CONFIG,
    DEFAULT_QUERY_CACHE_PATH,
    OPENROUTER_QUERY_BATCH_SIZE,
    OPENROUTER_QUERY_PARALLELISM,
    DEFAULT_SATELLITE_DATASET,
    DEFAULT_SATELLITE_SPLIT,
    ImageCaptionDataset,
    Stage1Collator,
    enrich_rows_with_text_queries,
    load_stage1_training_rows,
)
from trisearch_models import (
    DEFAULT_MATRYOSHKA_DIMS,
    DEFAULT_MAX_TEXT_LENGTH,
    DEFAULT_MEMORY_BANK_SIZE,
    DEFAULT_SEED_TEXT_DIR,
    DEFAULT_SEED_VISION_DIR,
    DEFAULT_TRAINED_DIR,
    QWEN_TOKENIZER_ID,
    SIGLIP_PROCESSOR_ID,
    Stage1AlignmentModel,
    build_optimizer,
    gpu_device,
    load_projection_heads,
    load_text_model_for_training,
    load_training_state,
    load_vision_model_for_training,
    resolve_model_dirs,
    resolve_training_dtype,
    run_training,
    sanity_check_loss,
    save_stage1_checkpoint,
    verify_trained_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-vision-dir", default=DEFAULT_SEED_VISION_DIR)
    parser.add_argument("--seed-text-dir", default=DEFAULT_SEED_TEXT_DIR)
    parser.add_argument("--vision-model-dir", default=None,
                        help="Deprecated alias for --seed-vision-dir.")
    parser.add_argument("--text-model-dir", default=None,
                        help="Deprecated alias for --seed-text-dir.")
    parser.add_argument("--trained-dir", default=DEFAULT_TRAINED_DIR)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--vision-processor-id", default=SIGLIP_PROCESSOR_ID)
    parser.add_argument("--text-tokenizer-id", default=QWEN_TOKENIZER_ID)

    parser.add_argument("--data-jsonl", default=None,
                        help="Local JSONL with image path + caption fields.")
    parser.add_argument(
        "--curated-dataset-dir",
        default=str(DEFAULT_CURATED_DATASET_DIR),
        help="TriSearch curated dataset from generate_datasets.py "
             f"(default {DEFAULT_CURATED_DATASET_DIR}).",
    )
    parser.add_argument(
        "--no-curated-dataset",
        action="store_true",
        help="Ignore curated export; use legacy HF satellite/general mix.",
    )
    parser.add_argument("--image-root", default=None,
                        help="Base directory for relative image paths in JSONL.")

    parser.add_argument("--satellite-dataset", default=DEFAULT_SATELLITE_DATASET)
    parser.add_argument("--satellite-split", default=DEFAULT_SATELLITE_SPLIT)
    parser.add_argument("--satellite-image-column", default="image")
    parser.add_argument("--satellite-caption-column", default="caption")
    parser.add_argument("--satellite-image-root", default=None,
                        help="Directory of ChatEarthNet PNG files (see also "
                             "--download-satellite-images).")
    parser.add_argument("--download-satellite-images", action="store_true",
                        help="Download/extract ChatEarthNet s2_rgb_images.zip "
                             "into models/data/ChatEarthNet/ when PNGs are missing.")

    parser.add_argument("--general-dataset", default=DEFAULT_GENERAL_DATASET)
    parser.add_argument("--general-split", default=DEFAULT_GENERAL_SPLIT)
    parser.add_argument("--general-image-column", default="image")
    parser.add_argument("--general-caption-column", default=DEFAULT_GENERAL_CAPTION_COLUMN)
    parser.add_argument("--satellite-fraction", type=float, default=0.5)

    parser.add_argument("--max-satellite-samples", type=int, default=None)
    parser.add_argument("--max-general-samples", type=int, default=None)
    parser.add_argument("--max-text-length", type=int, default=DEFAULT_MAX_TEXT_LENGTH)

    parser.add_argument("--matryoshka-dims", default="64,128,256,512,1024")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--matryoshka-weight", type=float, default=0.5)
    parser.add_argument("--text-text-weight", type=float, default=1.0,
                        help="Weight for query↔caption text-text contrastive loss.")
    parser.add_argument("--text-text-matryoshka-weight", type=float, default=0.5,
                        help="Weight for Matryoshka text-text contrastive loss.")
    parser.add_argument("--no-text-text-training", action="store_true",
                        help="Disable text-to-text semantic similarity training.")
    parser.add_argument(
        "--memory-bank-size",
        type=int,
        default=DEFAULT_MEMORY_BANK_SIZE,
        help="FIFO queue of detached embeddings used as extra contrastive "
             f"negatives (default {DEFAULT_MEMORY_BANK_SIZE}). "
             "Gives a large effective negative set without a large micro-batch. "
             "Set 0 to disable.",
    )
    parser.add_argument("--openrouter-config", default=str(DEFAULT_OPENROUTER_CONFIG),
                        help="YAML file with openrouter.api_key and openrouter.model.")
    parser.add_argument("--query-cache", default=str(DEFAULT_QUERY_CACHE_PATH),
                        help="JSONL cache for LLM-generated related/unrelated queries.")
    parser.add_argument("--max-query-gen", type=int, default=None,
                        help="Cap new OpenRouter calls (useful for smoke tests).")
    parser.add_argument("--query-batch-size", type=int,
                        default=OPENROUTER_QUERY_BATCH_SIZE,
                        help="Captions per OpenRouter API request.")
    parser.add_argument("--query-parallelism", type=int,
                        default=OPENROUTER_QUERY_PARALLELISM,
                        help="Concurrent OpenRouter API requests during query gen.")
    parser.add_argument("--skip-query-generation", action="store_true",
                        help="Require all queries in --query-cache; do not call API.")

    parser.add_argument("--vision-gpu", type=int, default=0)
    parser.add_argument("--text-gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--projection-learning-rate", type=float, default=1e-4)
    parser.add_argument("--vision-learning-rate", type=float, default=None)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")

    return parser.parse_args()


def _require_path(path: str, label: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{label} not found at {path!r}. Run the model creation scripts first."
        )


def main():
    args = parse_args()
    if args.vision_model_dir:
        args.seed_vision_dir = args.vision_model_dir
    if args.text_model_dir:
        args.seed_text_dir = args.text_model_dir

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    matryoshka_dims = tuple(
        int(x) for x in args.matryoshka_dims.split(",") if x.strip()
    )

    with_text_queries = not args.no_text_text_training

    mixed_rows, image_column, caption_column, image_root = load_stage1_training_rows(
        data_jsonl=args.data_jsonl,
        curated_dataset_dir=args.curated_dataset_dir,
        prefer_curated=not args.no_curated_dataset,
        image_root=args.image_root,
        satellite_dataset=args.satellite_dataset,
        satellite_split=args.satellite_split,
        satellite_image_column=args.satellite_image_column,
        satellite_caption_column=args.satellite_caption_column,
        satellite_image_root=args.satellite_image_root,
        general_dataset=args.general_dataset,
        general_split=args.general_split,
        general_image_column=args.general_image_column,
        general_caption_column=args.general_caption_column,
        satellite_fraction=args.satellite_fraction,
        max_satellite_samples=args.max_satellite_samples,
        max_general_samples=args.max_general_samples,
        seed=args.seed,
        download_satellite_images=args.download_satellite_images,
    )
    if with_text_queries:
        mixed_rows = enrich_rows_with_text_queries(
            mixed_rows,
            config_path=args.openrouter_config,
            cache_path=args.query_cache,
            max_new_queries=args.max_query_gen,
            skip_generation=args.skip_query_generation,
            caption_column=caption_column,
            query_batch_size=args.query_batch_size,
            query_parallelism=args.query_parallelism,
        )

    vision_load_dir, text_load_dir, checkpoint_root = resolve_model_dirs(
        fresh=args.fresh,
        checkpoint_dir=args.checkpoint_dir,
        seed_vision_dir=args.seed_vision_dir,
        seed_text_dir=args.seed_text_dir,
    )
    if checkpoint_root is None:
        _require_path(args.seed_text_dir, "Seed text model")
        _require_path(args.seed_vision_dir, "Seed vision model")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be >= 2 for contrastive in-batch negatives.")

    compute_dtype = resolve_training_dtype(args.bf16)
    vision_device = gpu_device(args.vision_gpu)
    text_device = gpu_device(args.text_gpu)

    text_model, tokenizer, text_hidden = load_text_model_for_training(
        model_dir=str(text_load_dir),
        tokenizer_id=args.text_tokenizer_id,
        max_seq_length=args.max_text_length,
        text_device=text_device,
        compute_dtype=compute_dtype,
    )
    vision_model, vision_hidden = load_vision_model_for_training(
        model_dir=str(vision_load_dir),
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
        with_text_queries=with_text_queries,
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
        text_text_weight=args.text_text_weight if with_text_queries else 0.0,
        text_text_matryoshka_weight=(
            args.text_text_matryoshka_weight if with_text_queries else 0.0
        ),
        compute_dtype=compute_dtype,
        memory_bank_size=args.memory_bank_size,
    )
    alignment_model.vision_projection.to(device=vision_device, dtype=compute_dtype)
    alignment_model.text_projection.to(device=text_device, dtype=compute_dtype)

    if checkpoint_root is not None:
        load_projection_heads(alignment_model, checkpoint_root)

    vision_lr = args.vision_learning_rate or args.learning_rate
    optimizer = build_optimizer(
        alignment_model,
        learning_rate=args.learning_rate,
        vision_learning_rate=vision_lr,
        projection_learning_rate=args.projection_learning_rate,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    if checkpoint_root is not None:
        start_step = load_training_state(checkpoint_root, optimizer)

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=Stage1Collator(
            pad_token_id=tokenizer.pad_token_id or 0,
            with_text_queries=with_text_queries,
        ),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    sanity_check_loss(alignment_model, dataloader)

    print("\n--- Stage 1 training ---")
    print(f"  samples        : {len(train_dataset):,}")
    print(f"  micro-batch    : {args.batch_size}")
    print(f"  effective batch: {args.batch_size * args.gradient_accumulation_steps}")
    print(
        f"  memory bank    : {args.memory_bank_size} "
        f"(contrastive negatives ≈ micro-batch-1 + bank)"
    )
    print(f"  max tokens     : {args.max_text_length}")
    print(f"  matryoshka dims: {matryoshka_dims}")
    print(f"  text-text train: {with_text_queries}")
    if with_text_queries:
        print(f"  text-text w    : {args.text_text_weight}")
        print(f"  text-text m w  : {args.text_text_matryoshka_weight}")
        print(f"  query cache    : {args.query_cache}")
        print(
            f"  query parallel : {args.query_parallelism} "
            f"x batch {args.query_batch_size}"
        )
    print(f"  compute dtype  : {compute_dtype}")
    print(f"  vision GPU     : {vision_device}")
    print(f"  text GPU       : {text_device}")
    print(f"  weight precision: 8-bit (bnb)")
    print(f"  optimizer      : AdamW8bit")
    print(f"  trained dir    : {args.trained_dir}")
    print(f"  resume step    : {start_step}\n")

    trained_path = Path(args.trained_dir)
    final_step = run_training(
        alignment_model,
        dataloader,
        optimizer,
        args,
        start_step=start_step,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )
    save_stage1_checkpoint(
        trained_path,
        alignment_model,
        args,
        final_step,
        optimizer,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )
    verify_trained_checkpoint(
        trained_path,
        seed_vision_dir=args.seed_vision_dir,
        seed_text_dir=args.seed_text_dir,
        tokenizer_id=args.text_tokenizer_id,
        vision_processor_id=args.vision_processor_id,
        max_text_length=args.max_text_length,
        bf16=args.bf16,
        vision_gpu=args.vision_gpu,
        text_gpu=args.text_gpu,
        with_text_queries=with_text_queries,
    )


if __name__ == "__main__":
    main()
