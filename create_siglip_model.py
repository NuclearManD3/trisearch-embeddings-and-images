#!/usr/bin/env python3
"""
Create a SigLIP vision embedder from a searched candidate configuration and
seed its parameters from an existing pretrained checkpoint.

Architecture-aware seeding:
  - patch embedding: spatial resize (not corner-slice) + channel expand
  - position embedding: 2D grid interpolate + width expand
  - attention: head-block copies when head_dim changes
  - deeper layers: clone last transferred block with residual outs zeroed

Run interactively:      python3 create_siglip_model.py
Run non-interactively:  python3 create_siglip_model.py --candidate 1 \\
                            --seed-model google/siglip-so400m-patch14-384
"""

import argparse
import warnings

from transformers import SiglipVisionModel

from design_model_sizes import SIGLIP_BASELINE_ID, search_siglip_vision
from model_seeding import create_seeded_model

warnings.filterwarnings("ignore")

MODEL_TYPE = "siglip-vision"
DEFAULT_OUTPUT_DIR = "models/siglip-vision"


def load_siglip_source(seed_id):
    """Load only the vision tower of the given checkpoint to seed weights from."""
    return SiglipVisionModel.from_pretrained(seed_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=int,
        default=None,
        help="1-based candidate index (skip the interactive prompt).",
    )
    parser.add_argument(
        "--seed-model",
        type=str,
        default=None,
        help="Checkpoint to seed parameters from "
        f"(default {SIGLIP_BASELINE_ID}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save the created model (default {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--target-params",
        type=int,
        default=1_000_000_000,
        help="Target parameter count for the search.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many candidate configurations to show.",
    )
    parser.add_argument(
        "--init-range",
        type=float,
        default=0.02,
        help="Std for newly initialized weight regions (default 0.02).",
    )
    args = parser.parse_args()

    create_seeded_model(
        search_fn=search_siglip_vision,
        source_loader=load_siglip_source,
        output_dir=args.output_dir,
        default_seed=SIGLIP_BASELINE_ID,
        candidate=args.candidate,
        seed_model=args.seed_model,
        search_kwargs={"target_params": args.target_params, "top_k": args.top_k},
        family="siglip",
        init_range=args.init_range,
    )


if __name__ == "__main__":
    main()
