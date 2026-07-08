#!/usr/bin/env python3
"""
Create an MMDiT (SD3-style) image generator transformer from a searched
candidate configuration and seed its parameters from an existing pretrained
checkpoint.

Workflow:
  1. Run the same candidate search as design_model_sizes.py (MMDiT section).
  2. Let the user pick a candidate configuration.
  3. Ask for a pretrained model to download and seed parameters from.
  4. Build the chosen configuration as a real SD3Transformer2DModel, seed every
     compatible weight from the source checkpoint, and save it under
     models/mmdit.

Run interactively:      python3 create_mmdit_model.py
Run non-interactively:  python3 create_mmdit_model.py --candidate 1 --seed-model v2ray/stable-diffusion-3-medium-diffusers
"""

import argparse
import warnings

from diffusers.models.transformers import SD3Transformer2DModel

from design_model_sizes import MMDIT_BASELINE_ID, search_mmdit
from model_seeding import create_seeded_model

warnings.filterwarnings("ignore")

MODEL_TYPE = "mmdit"
DEFAULT_OUTPUT_DIR = "models/mmdit"


def load_mmdit_source(seed_id):
    """Load the source SD3 transformer to seed weights from.

    Accepts either a full diffusers pipeline repo (weights live in the
    ``transformer`` subfolder) or a bare transformer repo.
    """
    try:
        return SD3Transformer2DModel.from_pretrained(seed_id, subfolder="transformer")
    except Exception:
        return SD3Transformer2DModel.from_pretrained(seed_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=int, default=None,
                        help="1-based candidate index (skip the interactive prompt).")
    parser.add_argument("--seed-model", type=str, default=None,
                        help="Checkpoint to seed parameters from "
                             "(skip the interactive prompt).")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Where to save the created model (default {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--target-params", type=int, default=2_000_000_000,
                        help="Target parameter count for the search.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="How many candidate configurations to show.")
    args = parser.parse_args()

    create_seeded_model(
        search_fn=search_mmdit,
        source_loader=load_mmdit_source,
        output_dir=args.output_dir,
        default_seed=MMDIT_BASELINE_ID,
        candidate=args.candidate,
        seed_model=args.seed_model,
        search_kwargs={"target_params": args.target_params, "top_k": args.top_k},
    )


if __name__ == "__main__":
    main()
