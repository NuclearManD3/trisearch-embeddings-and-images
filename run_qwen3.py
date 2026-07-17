#!/usr/bin/env python3
"""
Interactive smoke-test runner for the Qwen3-MoE text embedder.

Type a text query; the resized Qwen3-MoE model produces a list of 1024-dim
Matryoshka token embeddings. Every query is remembered, and for each new query
the two most similar previous queries are reported using ColBERT-style
late-interaction (MaxSim) scoring.

By default loads the newest valid training checkpoint: scans stage5 → stage1
and picks the latest root (stage dir or history/step-*) in the highest stage
that exists. Falls back to seed weights if nothing is trained yet.

Run:  python3 run_qwen3.py
      python3 run_qwen3.py --phase 0          # untrained seed
      python3 run_qwen3.py --phase 1          # stage1 live root only
      python3 run_qwen3.py --checkpoint-dir models/trained/stage1/history/step-6200
      python3 run_qwen3.py --merge-threshold 0.95
Quit: type 'q', 'quit', 'exit', or send EOF (Ctrl-D).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trisearch_models import (
    MAX_TRAINING_PHASE,
    MIN_TRAINING_PHASE,
    LateInteractionStore,
    Qwen3MoeEmbedder,
    default_inference_device,
    describe_phase,
    find_latest_trained_checkpoint,
    resolve_inference_checkpoint,
)


def _infer_phase_from_checkpoint(root: Path) -> int:
    """Best-effort stage number from path (models/trained/stageN/...)."""
    for part in root.resolve().parts:
        if part.startswith("stage") and part[5:].isdigit():
            n = int(part[5:])
            if 1 <= n <= MAX_TRAINING_PHASE:
                return n
    return 1


def _resolve_qwen_load(
    args: argparse.Namespace,
) -> tuple[str | None, str | None, int, str]:
    """Return (model_dir, projection_path, phase, description).

    ``model_dir`` None means use phase-based seed/stage resolution.
    """
    if args.model_dir:
        # Prefer phase>=1 so 8-bit trained dirs load correctly; use --phase 0 for seeds.
        phase = args.phase if args.phase is not None else 1
        return (
            args.model_dir,
            args.projection_path,
            phase,
            f"model override: {args.model_dir}",
        )

    if args.checkpoint_dir:
        root = resolve_inference_checkpoint(checkpoint_dir=args.checkpoint_dir)
        assert root is not None
        phase = _infer_phase_from_checkpoint(root)
        return (
            str(root / "text_model"),
            str(root / "projection_heads.pt"),
            phase,
            f"checkpoint: {root}",
        )

    if args.phase is not None:
        return (
            None,
            None,
            args.phase,
            describe_phase(args.phase, "qwen"),
        )

    # Default: newest trained checkpoint across stages.
    try:
        root = resolve_inference_checkpoint(latest_across_stages=True)
    except FileNotFoundError:
        root = None
    if root is None:
        # resolve_inference_checkpoint raises when empty; keep defensive fallback.
        root = find_latest_trained_checkpoint()
    if root is not None:
        phase = _infer_phase_from_checkpoint(root)
        return (
            str(root / "text_model"),
            str(root / "projection_heads.pt"),
            phase,
            f"latest trained checkpoint: {root}",
        )
    return (
        None,
        None,
        0,
        f"{describe_phase(0, 'qwen')} (no trained checkpoints found)",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        choices=range(MIN_TRAINING_PHASE, MAX_TRAINING_PHASE + 1),
        help="Training phase to load: 0=untrained seed, 1-5=stage N live root. "
             "Default: newest checkpoint across stages (stage5…1).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Override text backbone directory (ignores --phase / auto-latest).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Stage checkpoint root containing text_model/ and "
             "projection_heads.pt (e.g. models/trained/stage1/history/step-6200).",
    )
    parser.add_argument(
        "--projection-path",
        default=None,
        help="Optional projection_heads.pt when using --model-dir.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for the embedder (default: cuda:0 if available, else cpu). "
             "8-bit trained weights need CUDA.",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=1.0,
        help="Cosine-sim threshold to merge consecutive embeddings "
             "(1.0 = merge only identical; lower merges more).",
    )
    args = parser.parse_args()

    if args.model_dir and args.checkpoint_dir:
        raise SystemExit("Use only one of --model-dir or --checkpoint-dir")
    if args.checkpoint_dir and args.phase is not None:
        raise SystemExit("--checkpoint-dir ignores --phase; omit --phase")

    device = args.device or default_inference_device(0)
    model_dir, projection_path, phase, desc = _resolve_qwen_load(args)

    print("Loading Qwen3-MoE text embedder (this may take a moment) ...")
    print(f"  {desc}")
    print(f"  device: {device}")

    kwargs: dict = {"phase": phase, "device": device}
    if model_dir is not None:
        kwargs["model_dir"] = model_dir
    if projection_path is not None:
        kwargs["projection_path"] = projection_path
    elif args.projection_path is not None:
        kwargs["projection_path"] = args.projection_path

    embedder = Qwen3MoeEmbedder(**kwargs)
    store = LateInteractionStore()
    print(f"Ready (merge_threshold={args.merge_threshold}). Enter text to embed.\n")

    while True:
        try:
            text = input("text> ").strip()
        except EOFError:
            print()
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        if not text:
            continue

        embeddings = embedder.embed_text(text, merge_threshold=args.merge_threshold)
        print(
            f"  produced {len(embeddings)} x {embeddings[0].shape[0]}-dim "
            f"Matryoshka embeddings."
        )

        matches = store.most_similar(embeddings, top_k=2)
        if matches:
            print("  top matches among previous queries (late-interaction MaxSim):")
            for rank, (label, score) in enumerate(matches, start=1):
                print(f"    {rank}. score={score:8.4f}  |  {label}")
        else:
            print("  (no previous queries yet)")

        store.add(text, embeddings)
        print()

    print("Bye.")


if __name__ == "__main__":
    main()
