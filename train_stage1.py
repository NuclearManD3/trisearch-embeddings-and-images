#!/usr/bin/env python3
"""
Stage 1: Seeding & Cross-Modal Alignment (training_plan.md §3)

Trains the SigLIP vision embedder and Qwen3-MoE text embedder jointly in a
shared 1024-dim Matryoshka space using ColBERT-style contrastive loss and
Matryoshka loss. Towers load in full bf16/fp16 (Unsloth full_finetuning) so
every parameter is optimizable; AdamW8bit keeps optimizer state small.

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
    DEFAULT_TRISEARCH_HF_DATASET,
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
    DEFAULT_MAX_INPUT_TOKENS,
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
from trisearch_models.training import (
    DEFAULT_BANK_CLEAR_STEPS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_GEO_WEIGHT,
    DEFAULT_FREEZE_BACKBONE_RATIO,
    DEFAULT_GEO_AFTER_UNFREEZE,
    DEFAULT_GEO_CENTER_WEIGHT,
    DEFAULT_GEO_SQUARE,
    DEFAULT_GEO_EMA_MOMENTUM,
    DEFAULT_GEO_MAG_FLOOR,
    DEFAULT_GEO_MAG_FLOOR_WEIGHT,
    DEFAULT_GEO_MAX_ABS_RATIO,
    DEFAULT_GEO_MAX_ABS_WEIGHT,
    DEFAULT_GEO_POOL_WEIGHT,
    DEFAULT_GEO_PREFIX_DIM,
    DEFAULT_GEO_PREFIX_WEIGHT,
    DEFAULT_GEO_TOKEN_WEIGHT,
    DEFAULT_GEO_UNIFORMITY_MAX_SAMPLES,
    DEFAULT_GEO_UNIFORMITY_T,
    DEFAULT_GEO_UNIFORMITY_WEIGHT,
    DEFAULT_GEO_VAR_RATIO,
    DEFAULT_GEO_VAR_WEIGHT,
    DEFAULT_GEO_VEC_MEAN_WEIGHT,
    DEFAULT_GAP_LOSS_WEIGHT,
    DEFAULT_GAP_MARGIN,
    DEFAULT_GRAD_ACCUM_STEPS,
    DEFAULT_HARD_BANK_NEGATIVES,
    DEFAULT_HEATMAP_SPARSITY_SQUARE,
    DEFAULT_HEATMAP_SPARSITY_TEMPERATURE,
    DEFAULT_HEATMAP_SPARSITY_WEIGHT,
    DEFAULT_IMAGE_FILL_MODE,
    DEFAULT_IMAGE_HFLIP_PROB,
    DEFAULT_IMAGE_MAX_ROTATE_DEG,
    DEFAULT_IMAGE_SCALE_MAX,
    DEFAULT_IMAGE_SCALE_MIN,
    DEFAULT_IMAGE_SHIFT_MAX,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOGGING_STEPS,
    DEFAULT_MATRYOSHKA_WEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_MULTI_POSITIVE_JACCARD,
    DEFAULT_PROJECTION_LEARNING_RATE,
    DEFAULT_SAVE_STEPS,
    DEFAULT_QUERY_MAXSIM_TOPK,
    DEFAULT_SCORE_CENTER,
    DEFAULT_SOFT_MAXSIM_TEMPERATURE,
    DEFAULT_TEMPERATURE,
    DEFAULT_VISION_MERGE_ASSIGN_TEMPERATURE,
    DEFAULT_VISION_MERGE_TOKENS,
    DEFAULT_VISION_PATCH_DROP_PROB,
    DEFAULT_VISION_PATCH_KEEP_RATIO,
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
        "--hf-dataset",
        default=DEFAULT_TRISEARCH_HF_DATASET,
        help="HuggingFace dataset id for curated TriSearch "
             f"(default {DEFAULT_TRISEARCH_HF_DATASET}).",
    )
    parser.add_argument(
        "--curated-dataset-dir",
        default=str(DEFAULT_CURATED_DATASET_DIR),
        help="Optional local curated export (only if --prefer-local-curated).",
    )
    parser.add_argument(
        "--prefer-local-curated",
        action="store_true",
        help="Prefer local --curated-dataset-dir over the Hub dataset when present.",
    )
    parser.add_argument(
        "--curated-split",
        default="train",
        choices=("train", "test", "all"),
        help="Official curated split to train on (default train).",
    )
    parser.add_argument(
        "--no-curated-dataset",
        action="store_true",
        help="EMERGENCY ONLY: ignore TriSearch curated Hub set and use legacy "
             "COCO/SkyScript mix (not for normal training).",
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
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)

    parser.add_argument(
        "--matryoshka-dims",
        default="64,128,256,512",
        help="Prefix dims for Matryoshka CE (full embed dim is trained by the "
             "main contrastive terms and is stripped if listed).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"InfoNCE temperature (default {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=1.0,
        help="Relative weight of the caption↔image retrieval task in the "
             "mean task loss (default 1.0).",
    )
    parser.add_argument(
        "--matryoshka-weight",
        type=float,
        default=DEFAULT_MATRYOSHKA_WEIGHT,
        help="Within each retrieval task, relative weight of mean prefix CE "
             f"vs full-dim CE (default {DEFAULT_MATRYOSHKA_WEIGHT}). "
             "0 disables prefixes.",
    )
    parser.add_argument(
        "--text-text-weight",
        type=float,
        default=1.0,
        help="Relative weight of the query→caption retrieval task (default 1.0).",
    )
    parser.add_argument(
        "--text-text-matryoshka-weight",
        type=float,
        default=None,
        help="Deprecated alias; Matryoshka uses --matryoshka-weight for all tasks.",
    )
    parser.add_argument(
        "--query-image-weight",
        type=float,
        default=1.0,
        help="Relative weight of the query↔image retrieval task (default 1.0). "
             "0 disables query→image training.",
    )
    parser.add_argument(
        "--hard-bank-negatives",
        type=int,
        default=DEFAULT_HARD_BANK_NEGATIVES,
        help="Top-k hardest memory-bank docs kept per query as InfoNCE "
             f"negatives (default {DEFAULT_HARD_BANK_NEGATIVES}). "
             "0 uses the full bank.",
    )
    parser.add_argument(
        "--bank-clear-steps",
        type=int,
        default=DEFAULT_BANK_CLEAR_STEPS,
        help="Clear the memory bank every N global steps so a collapse "
             f"episode cannot poison all negatives (default {DEFAULT_BANK_CLEAR_STEPS}). "
             "0 disables.",
    )
    parser.add_argument("--no-text-text-training", action="store_true",
                        help="Disable query→caption and query↔image text-query training.")
    parser.add_argument(
        "--memory-bank-size",
        type=int,
        default=DEFAULT_MEMORY_BANK_SIZE,
        help="FIFO queue of detached embeddings used as extra contrastive "
             f"negatives (default {DEFAULT_MEMORY_BANK_SIZE}). "
             "Gives a large effective negative set without a large micro-batch. "
             "Set 0 to disable.",
    )
    parser.add_argument(
        "--bank-score-policy",
        choices=("accum_window", "live"),
        default="accum_window",
        help="Memory-bank scoring policy. 'accum_window' (policy B, default): "
             "enqueue every micro-batch, but score against the bank snapshot "
             "from the start of the gradient-accumulation window. "
             "'live': score against bank after each prior micro-batch enqueue.",
    )
    parser.add_argument(
        "--soft-maxsim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use soft MaxSim (τ logsumexp) instead of hard max (default: enabled).",
    )
    parser.add_argument(
        "--soft-maxsim-temperature",
        type=float,
        default=DEFAULT_SOFT_MAXSIM_TEMPERATURE,
        help=f"Soft MaxSim temperature τ_s (default {DEFAULT_SOFT_MAXSIM_TEMPERATURE}). "
             "Smaller → closer to hard max.",
    )
    parser.add_argument(
        "--multi-positive-jaccard",
        type=float,
        default=DEFAULT_MULTI_POSITIVE_JACCARD,
        help="Caption token-Jaccard threshold for multi-positive non-negative "
             f"masking in InfoNCE (default {DEFAULT_MULTI_POSITIVE_JACCARD}). "
             "Pairs at/above threshold are excluded from the negative set. "
             "Set 0 to disable.",
    )
    parser.add_argument(
        "--vision-patch-keep-ratio",
        type=float,
        default=DEFAULT_VISION_PATCH_KEEP_RATIO,
        help="Keep top fraction of SigLIP vision patches by pre-norm L2 "
             f"(drop background; default {DEFAULT_VISION_PATCH_KEEP_RATIO}). "
             "1.0 keeps all patches.",
    )
    parser.add_argument(
        "--vision-patch-drop-prob",
        type=float,
        default=DEFAULT_VISION_PATCH_DROP_PROB,
        help="Train-only: randomly drop this fraction of remaining patches "
             f"after L2 keep (default {DEFAULT_VISION_PATCH_DROP_PROB}). "
             "0 disables. Eval/demo keep all selected patches.",
    )
    parser.add_argument(
        "--vision-merge-tokens",
        type=int,
        default=DEFAULT_VISION_MERGE_TOKENS,
        help="Merge vision patches into this many similarity centroids "
             f"before MaxSim (default {DEFAULT_VISION_MERGE_TOKENS}). "
             "0 disables. Reduces background MaxSim collisions.",
    )
    parser.add_argument(
        "--vision-merge-assign-temperature",
        type=float,
        default=DEFAULT_VISION_MERGE_ASSIGN_TEMPERATURE,
        help="Softmax temperature for soft patch→centroid assignment "
             f"(default {DEFAULT_VISION_MERGE_ASSIGN_TEMPERATURE}).",
    )
    parser.set_defaults(score_center=DEFAULT_SCORE_CENTER)
    parser.add_argument(
        "--score-center",
        action="store_true",
        dest="score_center",
        help="Subtract detached live-batch mean of unit tokens before InfoNCE "
             "(default on). Removes domain cone that makes MaxSim match all images.",
    )
    parser.add_argument(
        "--no-score-center",
        action="store_false",
        dest="score_center",
        help="Disable batch score centering.",
    )
    parser.add_argument(
        "--query-maxsim-topk",
        type=int,
        default=DEFAULT_QUERY_MAXSIM_TOPK,
        help="Mean only the top-k query-token MaxSims "
             f"(default {DEFAULT_QUERY_MAXSIM_TOPK}; 0 = mean all tokens). "
             "Drops stopword-like tokens that match every document.",
    )
    parser.add_argument(
        "--gap-loss-weight",
        type=float,
        default=DEFAULT_GAP_LOSS_WEIGHT,
        help="Weight for score-gap hinge ReLU(margin - gap) on InfoNCE logits "
             f"(default {DEFAULT_GAP_LOSS_WEIGHT}). 0 disables. Directly trains "
             "the logged gap (pos - mean finite negs).",
    )
    parser.add_argument(
        "--gap-margin",
        type=float,
        default=DEFAULT_GAP_MARGIN,
        help="Target minimum score_gap in logit space "
             f"(default {DEFAULT_GAP_MARGIN}; 0 = push gap ≥ 0).",
    )
    parser.add_argument(
        "--image-shift-max",
        type=int,
        default=DEFAULT_IMAGE_SHIFT_MAX,
        help="Train-only: random per-image shift in [-N, N] pixels with "
             f"reflect pad (default {DEFAULT_IMAGE_SHIFT_MAX}). 0 disables.",
    )
    parser.add_argument(
        "--no-image-aug",
        action="store_true",
        help="Disable all train-time geometric image augs (flip/rotate/scale/shift).",
    )
    parser.add_argument(
        "--image-hflip-prob",
        type=float,
        default=DEFAULT_IMAGE_HFLIP_PROB,
        help=f"Train-only horizontal flip probability (default {DEFAULT_IMAGE_HFLIP_PROB}).",
    )
    parser.add_argument(
        "--image-max-rotate-deg",
        type=float,
        default=DEFAULT_IMAGE_MAX_ROTATE_DEG,
        help="Train-only max rotation in degrees, uniform in [-N, N] "
             f"(default {DEFAULT_IMAGE_MAX_ROTATE_DEG}). 0 disables rotate.",
    )
    parser.add_argument(
        "--image-scale-min",
        type=float,
        default=DEFAULT_IMAGE_SCALE_MIN,
        help="Train-only min anisotropic scale (default "
             f"{DEFAULT_IMAGE_SCALE_MIN}). Shrink pads with fill.",
    )
    parser.add_argument(
        "--image-scale-max",
        type=float,
        default=DEFAULT_IMAGE_SCALE_MAX,
        help="Train-only max anisotropic scale (default "
             f"{DEFAULT_IMAGE_SCALE_MAX}). Mild expand center-crops; keep near 1.05.",
    )
    parser.add_argument(
        "--image-fill-mode",
        choices=("random", "mean", "reflect"),
        default=DEFAULT_IMAGE_FILL_MODE,
        help="Fill for rotate/scale gaps: random channel noise, image mean, "
             f"or mean-as-reflect fallback (default {DEFAULT_IMAGE_FILL_MODE}).",
    )
    parser.add_argument(
        "--heatmap-sparsity-weight",
        type=float,
        default=DEFAULT_HEATMAP_SPARSITY_WEIGHT,
        help="Weight for positive-pair heatmap sparsity loss (normalized "
             f"entropy of patch MaxSim; default {DEFAULT_HEATMAP_SPARSITY_WEIGHT}). "
             "Punishes noisy/uniform heatmaps so the model selects few blocks. "
             "0 disables. Squared by default (see --heatmap-sparsity-square).",
    )
    parser.add_argument(
        "--heatmap-sparsity-temperature",
        type=float,
        default=DEFAULT_HEATMAP_SPARSITY_TEMPERATURE,
        help="Softmax temperature for heatmap sparsity "
             f"(default {DEFAULT_HEATMAP_SPARSITY_TEMPERATURE}).",
    )
    parser.set_defaults(heatmap_sparsity_square=DEFAULT_HEATMAP_SPARSITY_SQUARE)
    parser.add_argument(
        "--heatmap-sparsity-square",
        action="store_true",
        dest="heatmap_sparsity_square",
        help="Square heatmap entropy badness before weighting (default on): "
             "soft when already sparse, hard shove on diffuse MaxSim maps.",
    )
    parser.add_argument(
        "--no-heatmap-sparsity-square",
        action="store_false",
        dest="heatmap_sparsity_square",
        help="Use raw normalized entropy without squaring.",
    )
    parser.add_argument(
        "--embedding-geo-weight",
        type=float,
        default=DEFAULT_EMBEDDING_GEO_WEIGHT,
        help="Overall weight for embedding geometry / anti-cone loss "
             f"(default {DEFAULT_EMBEDDING_GEO_WEIGHT}). Set 0 to disable. "
             "Pushes batch mean toward 0, keeps per-dim variance above a floor, "
             "soft anti all-same-sign, pre-norm magnitude floor, soft max-|coord|.",
    )
    parser.add_argument(
        "--geo-center-weight",
        type=float,
        default=DEFAULT_GEO_CENTER_WEIGHT,
        help=f"Relative weight for ||μ||² center term (default {DEFAULT_GEO_CENTER_WEIGHT}).",
    )
    parser.add_argument(
        "--geo-var-weight",
        type=float,
        default=DEFAULT_GEO_VAR_WEIGHT,
        help=f"Relative weight for per-dim variance floor (default {DEFAULT_GEO_VAR_WEIGHT}).",
    )
    parser.add_argument(
        "--geo-vec-mean-weight",
        type=float,
        default=DEFAULT_GEO_VEC_MEAN_WEIGHT,
        help="Relative weight for per-vector mean² (anti all-positive/negative) "
             f"(default {DEFAULT_GEO_VEC_MEAN_WEIGHT}).",
    )
    parser.add_argument(
        "--geo-var-ratio",
        type=float,
        default=DEFAULT_GEO_VAR_RATIO,
        help="Variance floor as fraction of 1/sqrt(D) "
             f"(default {DEFAULT_GEO_VAR_RATIO}).",
    )
    parser.add_argument(
        "--geo-mag-floor",
        type=float,
        default=DEFAULT_GEO_MAG_FLOOR,
        help=f"Pre-norm L2 magnitude floor (default {DEFAULT_GEO_MAG_FLOOR}).",
    )
    parser.add_argument(
        "--geo-mag-floor-weight",
        type=float,
        default=DEFAULT_GEO_MAG_FLOOR_WEIGHT,
        help=f"Relative weight for magnitude floor (default {DEFAULT_GEO_MAG_FLOOR_WEIGHT}).",
    )
    parser.add_argument(
        "--geo-max-abs-ratio",
        type=float,
        default=DEFAULT_GEO_MAX_ABS_RATIO,
        help="Soft max-|coord| threshold as multiple of 1/sqrt(D) "
             f"(default {DEFAULT_GEO_MAX_ABS_RATIO}).",
    )
    parser.add_argument(
        "--geo-max-abs-weight",
        type=float,
        default=DEFAULT_GEO_MAX_ABS_WEIGHT,
        help=f"Relative weight for max-|coord| penalty (default {DEFAULT_GEO_MAX_ABS_WEIGHT}).",
    )
    parser.add_argument(
        "--geo-prefix-dim",
        type=int,
        default=DEFAULT_GEO_PREFIX_DIM,
        help="Also apply geometry loss on this Matryoshka prefix "
             f"(default {DEFAULT_GEO_PREFIX_DIM}; 0 disables).",
    )
    parser.add_argument(
        "--geo-prefix-weight",
        type=float,
        default=DEFAULT_GEO_PREFIX_WEIGHT,
        help="Relative weight of prefix geometry vs full-dim "
             f"(default {DEFAULT_GEO_PREFIX_WEIGHT}).",
    )
    parser.add_argument(
        "--geo-ema-momentum",
        type=float,
        default=DEFAULT_GEO_EMA_MOMENTUM,
        help="EMA momentum for running embedding mean used in center blending "
             f"(default {DEFAULT_GEO_EMA_MOMENTUM}).",
    )
    parser.add_argument(
        "--geo-uniformity-weight",
        type=float,
        default=DEFAULT_GEO_UNIFORMITY_WEIGHT,
        help="Wang–Isola uniformity term weight inside geo "
             f"(default {DEFAULT_GEO_UNIFORMITY_WEIGHT}; 0 disables).",
    )
    parser.add_argument(
        "--geo-uniformity-t",
        type=float,
        default=DEFAULT_GEO_UNIFORMITY_T,
        help=f"Uniformity temperature t in exp(-t||xi-xj||²) "
             f"(default {DEFAULT_GEO_UNIFORMITY_T}).",
    )
    parser.add_argument(
        "--geo-uniformity-max-samples",
        type=int,
        default=DEFAULT_GEO_UNIFORMITY_MAX_SAMPLES,
        help="Max rows subsampled for pairwise uniformity "
             f"(default {DEFAULT_GEO_UNIFORMITY_MAX_SAMPLES}).",
    )
    parser.add_argument(
        "--geo-token-weight",
        type=float,
        default=DEFAULT_GEO_TOKEN_WEIGHT,
        help="Relative weight of token-level geo branch "
             f"(default {DEFAULT_GEO_TOKEN_WEIGHT}).",
    )
    parser.add_argument(
        "--geo-pool-weight",
        type=float,
        default=DEFAULT_GEO_POOL_WEIGHT,
        help="Relative weight of L2-renormed sequence-mean geo branch "
             f"(default {DEFAULT_GEO_POOL_WEIGHT}).",
    )
    parser.set_defaults(geo_after_unfreeze=DEFAULT_GEO_AFTER_UNFREEZE)
    parser.add_argument(
        "--geo-after-unfreeze",
        action="store_true",
        dest="geo_after_unfreeze",
        help="Defer embedding geometry until backbone unfreeze. "
             "Linear proj cannot break a backbone cone during freeze.",
    )
    parser.add_argument(
        "--geo-during-freeze",
        action="store_false",
        dest="geo_after_unfreeze",
        help="Apply geometry loss also during projection-only freeze phase "
             f"(default when geo_after_unfreeze={DEFAULT_GEO_AFTER_UNFREEZE}).",
    )
    parser.set_defaults(geo_square=DEFAULT_GEO_SQUARE)
    parser.add_argument(
        "--geo-square",
        action="store_true",
        dest="geo_square",
        help="Square non-negative geo badness (default on): soft when nearly "
             "isotropic, quadratic shove on severe coning. Geo never negative.",
    )
    parser.add_argument(
        "--no-geo-square",
        action="store_false",
        dest="geo_square",
        help="Use raw non-negative geo badness without squaring.",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Micro-batch size (default {DEFAULT_BATCH_SIZE}; must be >= 2).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUM_STEPS,
        help=f"Grad accumulation (default {DEFAULT_GRAD_ACCUM_STEPS}; "
             f"effective batch = batch-size × this).",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Text/backbone peak LR (default {DEFAULT_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--projection-learning-rate",
        type=float,
        default=DEFAULT_PROJECTION_LEARNING_RATE,
        help=f"Projection-head peak LR (default {DEFAULT_PROJECTION_LEARNING_RATE}).",
    )
    parser.add_argument("--vision-learning-rate", type=float, default=None)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Total global-step budget (default {DEFAULT_MAX_STEPS}).",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03,
                        help="LR warmup as a fraction of total steps (cosine after).")
    parser.add_argument(
        "--freeze-backbone-ratio",
        type=float,
        default=DEFAULT_FREEZE_BACKBONE_RATIO,
        help="Fraction of total steps to train *only* projection heads "
             f"(vision/text towers frozen, eval mode). Default {DEFAULT_FREEZE_BACKBONE_RATIO}. "
             "Overridden by --freeze-backbone-steps. Use 0 or "
             "--no-freeze-backbone to disable.",
    )
    parser.add_argument(
        "--freeze-backbone-steps",
        type=int,
        default=None,
        help="Absolute global-step count for projection-only phase "
             "(overrides --freeze-backbone-ratio). 0 disables.",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        action="store_true",
        help="Train full vision/text towers from step 0 (no proj-only phase).",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=DEFAULT_LOGGING_STEPS,
        help=f"Log every N global steps (default {DEFAULT_LOGGING_STEPS}).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=DEFAULT_SAVE_STEPS,
        help=f"Checkpoint every N global steps (default {DEFAULT_SAVE_STEPS}).",
    )
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
        hf_dataset=None if args.no_curated_dataset else args.hf_dataset,
        prefer_curated=not args.no_curated_dataset,
        prefer_local_curated=args.prefer_local_curated,
        curated_split=args.curated_split,
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
        max_seq_length=args.max_input_tokens,
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
        max_text_length=args.max_input_tokens,
        with_text_queries=with_text_queries,
    )

    query_task_w = 1.0 if with_text_queries else 0.0
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
        text_text_weight=(
            args.text_text_weight * query_task_w if with_text_queries else 0.0
        ),
        text_text_matryoshka_weight=args.text_text_matryoshka_weight,
        query_image_weight=(
            args.query_image_weight * query_task_w if with_text_queries else 0.0
        ),
        hard_bank_negatives=args.hard_bank_negatives,
        compute_dtype=compute_dtype,
        memory_bank_size=args.memory_bank_size,
        soft_maxsim=args.soft_maxsim,
        soft_maxsim_temperature=args.soft_maxsim_temperature,
        multi_positive_jaccard=args.multi_positive_jaccard,
        vision_patch_keep_ratio=args.vision_patch_keep_ratio,
        vision_patch_drop_prob=args.vision_patch_drop_prob,
        vision_merge_tokens=args.vision_merge_tokens,
        vision_merge_assign_temperature=args.vision_merge_assign_temperature,
        score_center=args.score_center,
        query_maxsim_topk=args.query_maxsim_topk,
        gap_loss_weight=args.gap_loss_weight,
        gap_margin=args.gap_margin,
        image_shift_max=args.image_shift_max,
        image_hflip_prob=args.image_hflip_prob,
        image_max_rotate_deg=args.image_max_rotate_deg,
        image_scale_min=args.image_scale_min,
        image_scale_max=args.image_scale_max,
        image_fill_mode=args.image_fill_mode,
        image_aug_enabled=not args.no_image_aug,
        heatmap_sparsity_weight=args.heatmap_sparsity_weight,
        heatmap_sparsity_temperature=args.heatmap_sparsity_temperature,
        heatmap_sparsity_square=args.heatmap_sparsity_square,
        bank_score_policy=args.bank_score_policy,
        embedding_geo_weight=args.embedding_geo_weight,
        geo_center_weight=args.geo_center_weight,
        geo_var_weight=args.geo_var_weight,
        geo_vec_mean_weight=args.geo_vec_mean_weight,
        geo_var_ratio=args.geo_var_ratio,
        geo_mag_floor=args.geo_mag_floor,
        geo_mag_floor_weight=args.geo_mag_floor_weight,
        geo_max_abs_ratio=args.geo_max_abs_ratio,
        geo_max_abs_weight=args.geo_max_abs_weight,
        geo_uniformity_weight=args.geo_uniformity_weight,
        geo_uniformity_t=args.geo_uniformity_t,
        geo_uniformity_max_samples=args.geo_uniformity_max_samples,
        geo_token_weight=args.geo_token_weight,
        geo_pool_weight=args.geo_pool_weight,
        geo_prefix_dim=args.geo_prefix_dim,
        geo_prefix_weight=args.geo_prefix_weight,
        geo_ema_momentum=args.geo_ema_momentum,
        geo_after_unfreeze=args.geo_after_unfreeze,
        geo_square=args.geo_square,
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
        f"(hard top-{args.hard_bank_negatives} bank negs/query; "
        f"in-batch always kept; clear every {args.bank_clear_steps or 'never'})"
    )
    print(f"  bank policy    : {args.bank_score_policy}")
    print(
        f"  soft MaxSim    : {args.soft_maxsim}"
        + (f" (τ_s={args.soft_maxsim_temperature})" if args.soft_maxsim else "")
    )
    print(f"  multi-pos Jac  : {args.multi_positive_jaccard}")
    print(f"  vision keep    : {args.vision_patch_keep_ratio} (L2 background drop)")
    print(
        f"  patch dropout  : {args.vision_patch_drop_prob} "
        f"(train-only random drop after L2 keep)"
    )
    print(
        f"  vision merge   : k={args.vision_merge_tokens} "
        f"(sim centroids; τ_assign={args.vision_merge_assign_temperature})"
    )
    print(
        f"  score center   : {args.score_center} "
        f"(subtract live-batch mean before InfoNCE)"
    )
    print(
        f"  query MaxSim   : topk={args.query_maxsim_topk} "
        f"(0=mean all query tokens)"
    )
    print(
        f"  gap hinge      : weight={args.gap_loss_weight} "
        f"margin={args.gap_margin} (ReLU(m - score_gap))"
    )
    if args.no_image_aug:
        print("  image aug      : disabled (--no-image-aug)")
    else:
        print(
            f"  image aug      : flip p={args.image_hflip_prob}, "
            f"rotate ±{args.image_max_rotate_deg}°, "
            f"scale [{args.image_scale_min}, {args.image_scale_max}] "
            f"(anisotropic), fill={args.image_fill_mode}, "
            f"shift ±{args.image_shift_max} px (train-only)"
        )
    print(
        f"  heatmap sparse : weight={args.heatmap_sparsity_weight} "
        f"({'squared' if args.heatmap_sparsity_square else 'linear'}, "
        f"τ={args.heatmap_sparsity_temperature}; "
        f"entropy of patch MaxSim → select blocks)"
    )
    print(
        f"  emb geometry   : weight={args.embedding_geo_weight} "
        f"(center={args.geo_center_weight}, var={args.geo_var_weight}, "
        f"unif={args.geo_uniformity_weight}, "
        f"token/pool={args.geo_token_weight}/{args.geo_pool_weight}, "
        f"mag_floor={args.geo_mag_floor}, "
        f"prefix={args.geo_prefix_dim}@{args.geo_prefix_weight}, "
        f"{'after-unfreeze' if args.geo_after_unfreeze else 'from-step-0'}, "
        f"{'squared' if args.geo_square else 'linear-badness'})"
    )
    print(f"  max tokens     : {args.max_input_tokens}")
    print(
        f"  matryoshka     : prefixes={alignment_model.matryoshka_dims} "
        f"(full dim via main CE; mrl_w={args.matryoshka_weight})"
    )
    print(
        f"  tasks (equal mean of CE errors at embeddings): "
        f"caption↔image w={args.contrastive_weight}"
    )
    print(f"  query training : {with_text_queries}")
    if with_text_queries:
        print(f"  query↔image w  : {args.query_image_weight}")
        print(f"  query→caption w: {args.text_text_weight}")
        print(f"  query cache    : {args.query_cache}")
        print(
            f"  query parallel : {args.query_parallelism} "
            f"x batch {args.query_batch_size}"
        )
    print(f"  compute dtype  : {compute_dtype}")
    print(f"  vision GPU     : {vision_device}")
    print(f"  text GPU       : {text_device}")
    print(f"  weight precision: full {compute_dtype} (all params trainable)")
    print(f"  optimizer      : AdamW8bit (8-bit moments)")
    if args.no_freeze_backbone or (
        args.freeze_backbone_steps is not None and args.freeze_backbone_steps <= 0
    ) or (
        args.freeze_backbone_steps is None and args.freeze_backbone_ratio <= 0
    ):
        print("  backbone freeze: off (full towers from step 0)")
    elif args.freeze_backbone_steps is not None:
        print(
            f"  backbone freeze: proj-only until step {args.freeze_backbone_steps} "
            f"(then unfreeze)"
        )
    else:
        print(
            f"  backbone freeze: proj-only for first "
            f"{args.freeze_backbone_ratio:.0%} of steps (then unfreeze)"
        )
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
        max_text_length=args.max_input_tokens,
        bf16=args.bf16,
        vision_gpu=args.vision_gpu,
        text_gpu=args.text_gpu,
        with_text_queries=with_text_queries,
    )


if __name__ == "__main__":
    main()
