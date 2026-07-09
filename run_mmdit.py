#!/usr/bin/env python3
"""
Interactive smoke-test runner for the MMDiT image generator.

Type a text prompt and the resized MMDiT transformer generates an image, which
is opened in a popup window (and also saved to a temp file as a fallback).

The model is only resized (un-trained), so the image is essentially noise --
this just confirms the generator "turns on and runs".

Run: python3 run_mmdit.py
      python3 run_mmdit.py --phase 2   # once stage-2 generator weights exist
Quit: type 'q', 'quit', 'exit', or send EOF (Ctrl-D).
"""

import argparse
import tempfile

from trisearch_models import (
    MAX_TRAINING_PHASE,
    MIN_TRAINING_PHASE,
    MMDiTGenerator,
    default_inference_device,
    describe_phase,
)


def show_image(image):
    """Open the image in a popup; always print where it was saved too."""
    tmp = tempfile.NamedTemporaryFile(prefix="mmdit_", suffix=".png", delete=False)
    tmp.close()
    image.save(tmp.name)
    print(f"  saved image to: {tmp.name}")
    try:
        image.show(title="MMDiT output")
    except Exception as exc:  # headless / no viewer available
        print(f"  (could not open a popup: {exc})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, default=0,
                        choices=range(MIN_TRAINING_PHASE, MAX_TRAINING_PHASE + 1),
                        help="Training phase to load: 0=untrained seed, "
                             "1-5=future trained generator stages.")
    parser.add_argument("--model-dir", default=None,
                        help="Override the model directory (ignores --phase "
                             "for backbone weights).")
    parser.add_argument("--height", type=int, default=640,
                        help="Output image height in pixels (default 640; "
                             "must be a multiple of 16).")
    parser.add_argument("--width", type=int, default=640,
                        help="Output image width in pixels (default 640; "
                             "must be a multiple of 16).")
    parser.add_argument("--steps", type=int, default=4,
                        help="Number of denoising steps (default 4).")
    parser.add_argument("--device", default=None,
                        help="Device for the generator (default: cuda:0 if "
                             "available, else cpu).")
    args = parser.parse_args()

    device = args.device or default_inference_device(0)
    print("Loading MMDiT generator (this may take a moment) ...")
    if args.model_dir:
        print(f"  model override: {args.model_dir}")
    else:
        print(f"  {describe_phase(args.phase, 'mmdit')}")
    print(f"  device: {device}")
    kwargs = {"phase": args.phase, "device": device}
    if args.model_dir:
        kwargs["model_dir"] = args.model_dir
    generator = MMDiTGenerator(**kwargs)
    print("Ready. Enter a text prompt to generate an image.\n")

    while True:
        try:
            prompt = input("prompt> ").strip()
        except EOFError:
            print()
            break
        if prompt.lower() in ("q", "quit", "exit"):
            break
        if not prompt:
            continue
        print("  generating ...")
        image = generator.generate(text=prompt, height=args.height,
                                   width=args.width,
                                   num_inference_steps=args.steps)
        show_image(image)
        print()

    print("Bye.")


if __name__ == "__main__":
    main()
