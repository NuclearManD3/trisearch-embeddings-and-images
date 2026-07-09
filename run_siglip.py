#!/usr/bin/env python3
"""
Interactive smoke-test runner for the SigLIP vision embedder.

Give it an image -- either a path to an image file, or a pasted base64 string
(optionally a full ``data:image/...;base64,`` data URL). The resized SigLIP
tower produces a list of 1024-dim Matryoshka patch embeddings. Every image is
remembered, and for each new one the two most similar previous images are
reported using ColBERT-style late-interaction (MaxSim) scoring.

The model is only resized (un-trained), so the scores are not yet meaningful --
this just confirms the embedder "turns on and runs".

Run:  python3 run_siglip.py
      python3 run_siglip.py --phase 1   # load stage-1 trained weights
Quit: type 'q', 'quit', 'exit', or send EOF (Ctrl-D).
"""

import argparse
import base64
import io
import os

from PIL import Image

from trisearch_models import (
    MAX_TRAINING_PHASE,
    MIN_TRAINING_PHASE,
    LateInteractionStore,
    SiglipEmbedder,
    describe_phase,
)


def load_image(user_input):
    """Load a PIL image from a file path or a base64 / data-URL string."""
    text = user_input.strip()
    if os.path.isfile(text):
        return Image.open(text), os.path.basename(text)

    # Strip an optional data-URL prefix, then decode base64.
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    try:
        raw = base64.b64decode(text, validate=True)
        return Image.open(io.BytesIO(raw)), f"<base64 {len(raw)} bytes>"
    except Exception as exc:
        raise ValueError(f"Not a readable file path or base64 image: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, default=0,
                        choices=range(MIN_TRAINING_PHASE, MAX_TRAINING_PHASE + 1),
                        help="Training phase to load: 0=untrained seed, "
                             "1=stage-1 trained, 2-5=future stages.")
    parser.add_argument("--model-dir", default=None,
                        help="Override the model directory (ignores --phase "
                             "for backbone weights).")
    args = parser.parse_args()

    print("Loading SigLIP vision embedder (this may take a moment) ...")
    if args.model_dir:
        print(f"  model override: {args.model_dir}")
    else:
        print(f"  {describe_phase(args.phase, 'siglip')}")
    kwargs = {"phase": args.phase}
    if args.model_dir:
        kwargs["model_dir"] = args.model_dir
    embedder = SiglipEmbedder(**kwargs)
    store = LateInteractionStore()
    print("Ready. Enter an image path or a base64-encoded image.\n")

    counter = 0
    while True:
        try:
            user_input = input("image> ").strip()
        except EOFError:
            print()
            break
        if user_input.lower() in ("q", "quit", "exit"):
            break
        if not user_input:
            continue

        try:
            image, label = load_image(user_input)
        except ValueError as exc:
            print(f"  {exc}")
            continue

        counter += 1
        label = f"#{counter} {label}"
        embeddings = embedder.embed_image(image)
        print(f"  produced {len(embeddings)} x {embeddings[0].shape[0]}-dim "
              f"Matryoshka embeddings.")

        matches = store.most_similar(embeddings, top_k=2)
        if matches:
            print("  top matches among previous images (late-interaction MaxSim):")
            for rank, (prev_label, score) in enumerate(matches, start=1):
                print(f"    {rank}. score={score:8.4f}  |  {prev_label}")
        else:
            print("  (no previous images yet)")

        store.add(label, embeddings)
        print()

    print("Bye.")


if __name__ == "__main__":
    main()
