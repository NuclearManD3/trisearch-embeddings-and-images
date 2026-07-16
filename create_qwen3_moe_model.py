#!/usr/bin/env python3
"""
Create a Qwen3-MoE text embedder from a searched candidate configuration and
seed its parameters from an existing pretrained checkpoint.

Workflow:
  1. Run the same candidate search as design_model_sizes.py (Qwen3-MoE section).
  2. Let the user pick a candidate configuration.
  3. Ask for a pretrained model to download and seed parameters from (dense or
     MoE). Architecture-aware transfer maps dense MLP → experts, unpacks packed
     MoE tensors, head-block attention, and enforces expert diversity.
  4. Build the chosen configuration as a real Qwen3MoeForCausalLM, seed weights,
     write seed_report.json under models/qwen3-moe.

Default seed source: ``Qwen/Qwen3-1.7B`` (~1.7B dense). There is no official
Qwen3 MoE in the ~1.1–1.6B range; dense 1.7B is the closest sensible official
checkpoint for transferring embeddings / attention / FFN prior into MoE.

Run interactively:      python3 create_qwen3_moe_model.py
Run non-interactively:  python3 create_qwen3_moe_model.py --candidate 1
"""

import argparse
import warnings

from transformers import AutoModelForCausalLM

from design_model_sizes import QWEN_SEED_ID, search_qwen3_moe
from model_seeding import create_seeded_model

warnings.filterwarnings("ignore")

MODEL_TYPE = "qwen3-moe"
DEFAULT_OUTPUT_DIR = "models/qwen3-moe"


def load_qwen_source(seed_id):
    """Load the source causal-LM (dense or MoE) to seed weights from."""
    return AutoModelForCausalLM.from_pretrained(seed_id)


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
        f"(default {QWEN_SEED_ID}).",
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
        "--strict-experts",
        action="store_true",
        help="Fail if expert diversity check still fails after re-init.",
    )
    parser.add_argument(
        "--init-range",
        type=float,
        default=0.02,
        help="Std for newly initialized weight regions (default 0.02).",
    )
    args = parser.parse_args()

    create_seeded_model(
        search_fn=search_qwen3_moe,
        source_loader=load_qwen_source,
        output_dir=args.output_dir,
        default_seed=QWEN_SEED_ID,
        candidate=args.candidate,
        seed_model=args.seed_model,
        search_kwargs={"target_params": args.target_params, "top_k": args.top_k},
        family="qwen_moe",
        init_range=args.init_range,
        strict_experts=args.strict_experts,
    )


if __name__ == "__main__":
    main()
