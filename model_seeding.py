#!/usr/bin/env python3
"""
Shared helpers for the three model-creation scripts:

- ``select_candidate`` : let the user pick one of the candidate configurations
  produced by the search functions in ``design_model_sizes.py``.
- ``prompt_seed_model``: ask the user which pretrained checkpoint to seed the
  new model's parameters from.
- ``seed_parameters``  : copy every compatible weight from a source (pretrained)
  model into a freshly built target model. Weights are matched by parameter
  name; identically-shaped tensors are copied verbatim, and tensors that only
  differ in size are seeded on their overlapping region (the classic
  "grow/shrink from a smaller/larger checkpoint" trick). Parameters that have no
  counterpart keep their fresh initialization.

None of these use heuristics for parameter counts -- every model involved is a
real, instantiated model.
"""

import os

import torch


def select_candidate(candidate_cols, preset=None):
    """Return the column dict chosen by the user.

    candidate_cols : list of candidate column dicts (as returned by run_search).
    preset         : 1-based index chosen non-interactively (e.g. from argparse).
                     When None, the user is prompted on stdin.
    """
    n = len(candidate_cols)
    if n == 0:
        raise SystemExit("No buildable candidates were found for this target.")

    if preset is not None:
        idx = int(preset)
    else:
        print(f"\nSelect a candidate configuration [1-{n}] "
              f"(the columns 'cand 1' .. 'cand {n}' above):")
        while True:
            raw = input(f"Candidate number [1-{n}] (default 1): ").strip()
            if raw == "":
                idx = 1
                break
            try:
                idx = int(raw)
            except ValueError:
                print("  Please enter a number.")
                continue
            if 1 <= idx <= n:
                break
            print(f"  Out of range, choose between 1 and {n}.")

    if not (1 <= idx <= n):
        raise SystemExit(f"Candidate {idx} is out of range (1-{n}).")

    chosen = candidate_cols[idx - 1]
    print(f"Selected {chosen['header']} "
          f"({chosen['params']:,} params, {chosen['params'] / 1e9:.4f}B).")
    return chosen


def prompt_seed_model(preset=None, default=None):
    """Return the HuggingFace id / local path of the model to seed weights from."""
    if preset is not None:
        seed_id = preset
    else:
        suffix = f" (default {default})" if default else ""
        seed_id = input(f"\nModel to download and seed parameters from{suffix}: ").strip()
        if seed_id == "" and default:
            seed_id = default
    if not seed_id:
        raise SystemExit("A seed model id is required.")
    print(f"Seeding parameters from: {seed_id}")
    return seed_id


@torch.no_grad()
def seed_parameters(target_model, source_model):
    """Copy compatible weights from ``source_model`` into ``target_model``.

    Returns a dict of statistics describing how many tensors were copied
    exactly, copied partially (overlapping slice), or left at their fresh
    initialization.
    """
    source_state = dict(source_model.state_dict())
    target_state = target_model.state_dict()

    stats = {
        "exact": 0,
        "partial": 0,
        "shape_mismatch": 0,
        "no_match": 0,
        "total_target": len(target_state),
    }

    for name, tparam in target_state.items():
        sparam = source_state.get(name)
        if sparam is None:
            stats["no_match"] += 1
            continue
        if tuple(sparam.shape) == tuple(tparam.shape):
            tparam.copy_(sparam.to(tparam.dtype))
            stats["exact"] += 1
        elif sparam.dim() == tparam.dim():
            # Seed the overlapping region so smaller/larger checkpoints still help.
            slices = tuple(slice(0, min(a, b))
                           for a, b in zip(tparam.shape, sparam.shape))
            tparam[slices].copy_(sparam[slices].to(tparam.dtype))
            stats["partial"] += 1
        else:
            stats["shape_mismatch"] += 1

    target_model.load_state_dict(target_state)
    return stats


def report_seeding(stats):
    print("\nParameter seeding summary:")
    print(f"  copied exactly       : {stats['exact']}")
    print(f"  copied partially     : {stats['partial']}")
    print(f"  incompatible shapes  : {stats['shape_mismatch']}")
    print(f"  freshly initialized  : {stats['no_match']}")
    print(f"  total target tensors : {stats['total_target']}")


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def create_seeded_model(search_fn, source_loader, output_dir,
                        default_seed=None, candidate=None, seed_model=None,
                        search_kwargs=None):
    """End-to-end flow shared by the three creation scripts.

    1. Run the candidate search (prints the comparison table).
    2. Let the user select a candidate configuration.
    3. Ask for a pretrained model to seed parameters from.
    4. Build the selected configuration as a real model, seed its weights from
       the chosen checkpoint, and save it to ``output_dir``.

    search_fn     : one of the search_* functions from design_model_sizes; it
                    must return (baseline_col, candidate_cols).
    source_loader : callable(seed_id) -> instantiated pretrained model.
    """
    search_kwargs = dict(search_kwargs or {})
    baseline_col, candidate_cols = search_fn(**search_kwargs)

    chosen = select_candidate(candidate_cols, preset=candidate)
    seed_id = prompt_seed_model(preset=seed_model, default=default_seed)

    print("\nBuilding the target model from the selected configuration "
          "(this materializes real weights) ...")
    target = chosen["build_fn"](chosen["cfg_obj"])
    print(f"Target model built: {sum(p.numel() for p in target.parameters()):,} params.")

    print("Loading the source model to seed parameters from (this downloads "
          "the checkpoint) ...")
    source = source_loader(seed_id)

    stats = seed_parameters(target, source)
    report_seeding(stats)

    ensure_output_dir(output_dir)
    target.save_pretrained(output_dir)
    print(f"\nSaved seeded model to: {output_dir}")
    return target, stats
