#!/usr/bin/env python3
"""
Interactive smoke-test runner for the Qwen3-MoE text embedder.

Type a text query; the resized Qwen3-MoE model produces a list of 1024-dim
Matryoshka token embeddings. Every query is remembered, and for each new query
the two most similar previous queries are reported using ColBERT-style
late-interaction (MaxSim) scoring.

The model is only resized (un-trained), so the scores are not yet meaningful --
this just confirms the embedder "turns on and runs".

Run:  python3 run_qwen3.py
      python3 run_qwen3.py --merge-threshold 0.95
Quit: type 'q', 'quit', 'exit', or send EOF (Ctrl-D).
"""

import argparse

from trisearch_models import LateInteractionStore, Qwen3MoeEmbedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None,
                        help="Path to the Qwen3-MoE model directory.")
    parser.add_argument("--merge-threshold", type=float, default=1.0,
                        help="Cosine-sim threshold to merge consecutive "
                             "embeddings (1.0 = merge only identical; lower "
                             "merges more).")
    args = parser.parse_args()

    print("Loading Qwen3-MoE text embedder (this may take a moment) ...")
    kwargs = {"model_dir": args.model_dir} if args.model_dir else {}
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
        print(f"  produced {len(embeddings)} x {embeddings[0].shape[0]}-dim "
              f"Matryoshka embeddings.")

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
