#!/usr/bin/env python3
"""
Hyperparameter Search Script for:
- SigLIP Vision Embedder   (~1B params, 1024-dim output)
- Qwen3-MoE Text Embedder  (~1B params, 1024-dim output)
- MMDiT Image Generator    (~2B params)

For every family we:
  1. Pull the *real* configuration of a reference (baseline) checkpoint from the
     HuggingFace Hub.
  2. Compute its exact parameter count by instantiating the model on the `meta`
     device (no weights are downloaded, no memory is allocated).
  3. Search a grid of candidate configurations. Every candidate is a full copy of
     the baseline configuration with only the searched keys overridden, and its
     parameter count is measured by actually instantiating the model.
  4. Print a table with every configuration key as a row and one column per
     configuration (the leftmost data column is always the baseline).

No parameter counts are ever estimated with heuristics: every number in the
output comes from a real instantiated model.

Run with: python3 design_model_sizes.py
Requires: transformers, diffusers, torch (recent versions recommended)
"""

import copy
import inspect
import warnings
from itertools import product

import torch
from transformers import (
    AutoConfig, AutoModelForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
    Qwen3MoeConfig, Qwen3MoeForCausalLM,
)
from diffusers.models.transformers import SD3Transformer2DModel

warnings.filterwarnings("ignore")

# Baseline checkpoints (public, non-gated) used as *architecture* references
# for the size-search tables (meta param counts). Weight *seeding* defaults
# live in model_seeding.DEFAULT_QWEN_SEED_ID / create_*.py defaults.
SIGLIP_BASELINE_ID = "google/siglip-so400m-patch14-384"
# Dense Qwen3 ~1.7B: used for shared dims when building MoE candidates, and as
# a small official reference. (No official Qwen3 MoE exists in the ~1–2B band.)
QWEN_BASELINE_ID = "Qwen/Qwen3-1.7B"
# Default *weight seed* for create_qwen3_moe_model (dense→MoE arch-aware).
QWEN_SEED_ID = "Qwen/Qwen3-1.7B"
MMDIT_BASELINE_ID = "v2ray/stable-diffusion-3-medium-diffusers"


# ============================ GENERIC HELPERS ============================

def count_parameters(model):
    """Total parameter count. Works with meta-device models."""
    return sum(p.numel() for p in model.parameters())


def _fmt(value):
    """Format a config value for the table (compact, no truncation of numbers)."""
    if isinstance(value, float):
        # Keep floats readable but exact enough for config purposes.
        return repr(value)
    if isinstance(value, list) and len(value) > 1 and all(x == value[0] for x in value):
        # Losslessly compress a uniform list (e.g. per-layer attention types)
        # so it does not blow up the column width.
        return f"[{value[0]!r}] * {len(value)}"
    return str(value)


def print_config_table(title, notes, baseline_col, candidate_cols):
    """Print a table: config keys on the left, one column per configuration.

    baseline_col / candidate_cols are dicts with keys:
        "header": column title
        "params": total parameter count (int)
        "config": full config dict for that model
        "x_target": multiple of the target size
        "x_baseline": multiple of the baseline size
    The leftmost data column is the baseline.
    """
    columns = [baseline_col] + candidate_cols

    # Union of every config key across all columns -> we show EVERYTHING.
    all_keys = []
    seen = set()
    for col in columns:
        for k in col["config"].keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    all_keys.sort()

    # Derived (summary) rows shown at the top of the table.
    summary_rows = [
        ("TOTAL PARAMETERS", [f"{c['params']:,}" for c in columns]),
        ("total (M)", [f"{c['params'] / 1e6:.2f}M" for c in columns]),
        ("total (B)", [f"{c['params'] / 1e9:.4f}B" for c in columns]),
        ("x target size", [f"{c['x_target']:.3f}x" for c in columns]),
        ("x baseline size", [f"{c['x_baseline']:.3f}x" for c in columns]),
    ]

    config_rows = []
    for key in all_keys:
        if key.startswith('_') or key.startswith('output_'):
            continue
        if key in ('architectures', 'hidden_act', 'is_encoder_decoder', 'transformers_version', 'chunk_size_feed_forward', 'dtype', 'id2label', 'label2id', 'model_type', 'return_dict', 'rope_parameters'):
            continue
        row = []
        for col in columns:
            if key in col["config"]:
                row.append(_fmt(col["config"][key]))
            else:
                row.append("-")
        if all(i in ('', 'None') for i in row):
            continue

        config_rows.append((key, row))

    # Column headers.
    headers = ["CONFIG KEY"] + [c["header"] for c in columns]

    # Compute width for each column.
    all_rows = summary_rows + [("", [""] * len(columns))] + config_rows
    widths = [len(headers[0])]
    for i in range(len(columns)):
        widths.append(len(headers[i + 1]))
    for label, values in all_rows:
        widths[0] = max(widths[0], len(label))
        for i, v in enumerate(values):
            widths[i + 1] = max(widths[i + 1], len(v))

    def render(label, values):
        cells = [label.ljust(widths[0])]
        for i, v in enumerate(values):
            cells.append(v.rjust(widths[i + 1]))
        return " | ".join(cells)

    total_width = sum(widths) + 3 * len(widths)

    print("\n" + "=" * total_width)
    print(title)
    print("=" * total_width)
    for note in notes:
        print(note)
    print("-" * total_width)
    print(render(headers[0], headers[1:]))
    print("-" * total_width)
    for label, values in summary_rows:
        print(render(label, values))
    print("-" * total_width)
    print("CONFIGURATION KEYS")
    print("-" * total_width)
    for label, values in config_rows:
        print(render(label, values))
    print("=" * total_width)


def run_search(baseline_cfg, baseline_params, baseline_name, build_fn,
               to_dict_fn, override_grid, target_params, constraint=None,
               top_k=5, candidate_base_cfg=None):
    """Generic search driver.

    baseline_cfg   : the reference config object (or dict) used for the baseline
                     column (displayed with its real, unmodified parameters).
    baseline_params: exact parameter count of the baseline.
    build_fn(cfg)  : instantiate a model on the meta device from a config object.
    to_dict_fn(cfg): return the full config as a plain dict.
    override_grid  : dict of {key: [values]} describing the search space.
    constraint(cfg): optional predicate; skip candidates that return False.
    candidate_base_cfg: config that every candidate is derived from (deep-copied
                     then overridden). Defaults to baseline_cfg. Needed when the
                     baseline architecture differs from the candidate one (e.g.
                     a dense baseline for a MoE search).
    Returns (baseline_col, candidate_cols) ready for print_config_table.
    """
    if candidate_base_cfg is None:
        candidate_base_cfg = baseline_cfg
    keys = list(override_grid.keys())
    candidates = []
    for combo in product(*[override_grid[k] for k in keys]):
        overrides = dict(zip(keys, combo))
        cfg = copy.deepcopy(candidate_base_cfg)
        for k, v in overrides.items():
            _set_key(cfg, k, v)
        if constraint is not None and not constraint(cfg):
            continue
        try:
            with torch.device("meta"):
                model = build_fn(cfg)
            params = count_parameters(model)
        except Exception:
            # No heuristics: if a config cannot be instantiated, it is skipped.
            continue
        candidates.append((params, cfg))

    # Keep the closest-to-target candidates.
    candidates.sort(key=lambda x: abs(x[0] - target_params))
    candidates = candidates[:top_k]

    baseline_col = {
        "header": "BASELINE",
        "params": baseline_params,
        "config": to_dict_fn(baseline_cfg),
        "cfg_obj": baseline_cfg,
        "build_fn": build_fn,
        "x_target": baseline_params / target_params,
        "x_baseline": 1.0,
    }
    candidate_cols = []
    for i, (params, cfg) in enumerate(candidates):
        candidate_cols.append({
            "header": f"cand {i + 1}",
            "params": params,
            "config": to_dict_fn(cfg),
            "cfg_obj": cfg,
            "build_fn": build_fn,
            "x_target": params / target_params,
            "x_baseline": params / baseline_params,
        })
    return baseline_col, candidate_cols


def _set_key(cfg, key, value):
    """Set a key on either a config object or a plain dict."""
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


# ============================ SIGLIP VISION ============================

def load_siglip_baseline():
    full = AutoConfig.from_pretrained(SIGLIP_BASELINE_ID)
    vision_cfg = SiglipVisionConfig(**full.vision_config.to_dict())
    with torch.device("meta"):
        model = SiglipVisionModel(vision_cfg)
    return vision_cfg, count_parameters(model)


def search_siglip_vision(target_params=1_000_000_000, target_embed_dim=1024, top_k=5):
    baseline_cfg, baseline_params = load_siglip_baseline()

    # Use 540px images with 18px patches -> (540/18)^2 = 30x30 = 900 patch
    # tokens (higher than the baseline's 27x27), giving finer detail and more
    # information per patch at the cost of more compute. Detail processing is
    # important for this model, so image_size/patch_size are fixed here.
    override_grid = {
        "hidden_size": [1024, 1152, 1280, 1536],
        "num_hidden_layers": [24, 27, 32, 40],
        "num_attention_heads": [16],
        "image_size": [540],
        "patch_size": [18],
    }
    override_grid_full = dict(override_grid)

    def constraint(cfg):
        if cfg.hidden_size % cfg.num_attention_heads != 0:
            return False
        # Keep the conventional 4x MLP expansion relative to hidden size.
        cfg.intermediate_size = cfg.hidden_size * 4
        return True

    baseline_col, candidate_cols = run_search(
        baseline_cfg=baseline_cfg,
        baseline_params=baseline_params,
        baseline_name=SIGLIP_BASELINE_ID,
        build_fn=lambda cfg: SiglipVisionModel(cfg),
        to_dict_fn=lambda cfg: cfg.to_dict(),
        override_grid=override_grid_full,
        target_params=target_params,
        constraint=constraint,
        top_k=top_k,
    )
    print_config_table(
        title=f"SigLIP Vision Embedder  ->  target ~{target_params / 1e9:.1f}B params",
        notes=[
            f"Baseline checkpoint : {SIGLIP_BASELINE_ID} (vision tower)",
            f"Baseline parameters : {baseline_params:,} ({baseline_params / 1e6:.2f}M)",
            "Resolution 540px with 18px patches -> 30x30 = 900 patch tokens (higher detail).",
            f"Add a projection head nn.Linear(hidden_size, {target_embed_dim}) for the {target_embed_dim}-dim embedding.",
        ],
        baseline_col=baseline_col,
        candidate_cols=candidate_cols,
    )
    return baseline_col, candidate_cols


# ============================ QWEN3-MOE ============================

def load_qwen_baseline():
    # Load the baseline with its *real* architecture (it may be dense or MoE)
    # so its parameter count is accurate. Forcing a dense checkpoint into
    # Qwen3MoeConfig would inject MoE defaults and grossly inflate the count.
    cfg = AutoConfig.from_pretrained(QWEN_BASELINE_ID)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    return cfg, count_parameters(model)


def build_qwen_candidate_base(baseline_cfg):
    """Build a proper Qwen3-MoE config seeded from the baseline's shared dims.

    The baseline may be a dense Qwen3 model, so MoE candidates cannot be derived
    from it directly. We start from a fresh Qwen3MoeConfig and copy over the
    dimensions that should stay comparable to the baseline (vocabulary, context
    length, attention head geometry, rope/normalization settings).
    """
    base = baseline_cfg.to_dict()
    shared_keys = (
        "vocab_size", "max_position_embeddings", "head_dim", "rope_theta",
        "rope_scaling", "hidden_act", "rms_norm_eps", "tie_word_embeddings",
        "attention_bias", "attention_dropout",
    )
    seed = {k: base[k] for k in shared_keys if k in base and base[k] is not None}
    return Qwen3MoeConfig(**seed)


def search_qwen3_moe(target_params=1_000_000_000, target_embed_dim=1024, top_k=5):
    baseline_cfg, baseline_params = load_qwen_baseline()
    candidate_base_cfg = build_qwen_candidate_base(baseline_cfg)

    override_grid = {
        "hidden_size": [768, 1024, 1280, 1536],
        "num_hidden_layers": [12, 16, 20, 24],
        "num_local_experts": [8, 16, 32, 64],
        "moe_intermediate_size": [384, 512, 768],
        "num_attention_heads": [8, 16],
    }

    def constraint(cfg):
        # Qwen3 uses an explicit head_dim, so hidden_size need not divide heads,
        # but the KV heads must divide the attention heads.
        cfg.num_key_value_heads = max(1, cfg.num_attention_heads // 2)
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0:
            return False
        # Keep the dense (shared) MLP proportional to the expert MLP.
        cfg.intermediate_size = cfg.moe_intermediate_size
        return True

    baseline_col, candidate_cols = run_search(
        baseline_cfg=baseline_cfg,
        baseline_params=baseline_params,
        baseline_name=QWEN_BASELINE_ID,
        build_fn=lambda cfg: Qwen3MoeForCausalLM(cfg),
        to_dict_fn=lambda cfg: cfg.to_dict(),
        override_grid=override_grid,
        target_params=target_params,
        constraint=constraint,
        top_k=top_k,
        candidate_base_cfg=candidate_base_cfg,
    )
    print_config_table(
        title=f"Qwen3-MoE Text Embedder  ->  target ~{target_params / 1e9:.1f}B params",
        notes=[
            f"Baseline checkpoint : {QWEN_BASELINE_ID}",
            f"Baseline parameters : {baseline_params:,} ({baseline_params / 1e6:.2f}M)",
            f"Use mean/last-token pooling + nn.Linear(hidden_size, {target_embed_dim}) for the {target_embed_dim}-dim embedding.",
        ],
        baseline_col=baseline_col,
        candidate_cols=candidate_cols,
    )
    return baseline_col, candidate_cols


# ============================ MMDiT (SD3) ============================

def load_mmdit_baseline():
    with torch.device("meta"):
        model = SD3Transformer2DModel.from_pretrained(
            MMDIT_BASELINE_ID, subfolder="transformer"
        )
    init_keys = set(inspect.signature(SD3Transformer2DModel.__init__).parameters.keys())
    init_cfg = {k: v for k, v in dict(model.config).items() if k in init_keys}
    return init_cfg, count_parameters(model)


def search_mmdit(target_params=2_000_000_000, top_k=5):
    baseline_cfg, baseline_params = load_mmdit_baseline()

    override_grid = {
        "num_layers": [18, 21, 24, 27, 30],
        "num_attention_heads": [16, 20, 24, 28],
        "attention_head_dim": [64, 72, 80, 96],
    }

    def constraint(cfg):
        # In SD3/MMDiT the joint-attention stream width is
        # inner_dim = num_attention_heads * attention_head_dim, and the text
        # (context) stream must share that width. Keep caption_projection_dim in
        # sync with inner_dim, otherwise the joint attention blocks cannot run.
        cfg["caption_projection_dim"] = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        return True

    baseline_col, candidate_cols = run_search(
        baseline_cfg=baseline_cfg,
        baseline_params=baseline_params,
        baseline_name=MMDIT_BASELINE_ID,
        build_fn=lambda cfg: SD3Transformer2DModel(**cfg),
        to_dict_fn=lambda cfg: dict(cfg),
        override_grid=override_grid,
        target_params=target_params,
        constraint=constraint,
        top_k=top_k,
    )
    print_config_table(
        title=f"MMDiT Image Generator  ->  target ~{target_params / 1e9:.1f}B params",
        notes=[
            f"Baseline checkpoint : {MMDIT_BASELINE_ID} (transformer)",
            f"Baseline parameters : {baseline_params:,} ({baseline_params / 1e6:.2f}M)",
        ],
        baseline_col=baseline_col,
        candidate_cols=candidate_cols,
    )
    return baseline_col, candidate_cols


# ============================ MAIN ============================

if __name__ == "__main__":
    print("Model Configuration Search Script")
    print("Targets: 1024-dim embeddings | ~1B params (embedders) | ~2B params (generator)")
    print("All parameter counts come from real instantiated models (no heuristics).")

    search_siglip_vision(target_params=1_000_000_000)
    search_qwen3_moe(target_params=1_000_000_000)
    search_mmdit(target_params=2_000_000_000)
