#!/usr/bin/env python3
"""
Interactive smoke-test runner for the MMDiT image generator.

Type a text prompt and the resized MMDiT transformer generates an image, which
is opened in a popup window (and also saved to a temp file as a fallback).

The model is only resized (un-trained), so the image is essentially noise --
this just confirms the generator "turns on and runs".

Run: python3 run_mmdit.py
Quit: type 'q', 'quit', 'exit', or send EOF (Ctrl-D).
"""

import argparse
import tempfile

from trisearch_models import MMDiTGenerator


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
    parser.add_argument("--model-dir", default=None,
                        help="Path to the MMDiT model directory.")
    parser.add_argument("--height", type=int, default=640,
                        help="Output image height in pixels (default 640; "
                             "must be a multiple of 16).")
    parser.add_argument("--width", type=int, default=640,
                        help="Output image width in pixels (default 640; "
                             "must be a multiple of 16).")
    parser.add_argument("--steps", type=int, default=4,
                        help="Number of denoising steps (default 4).")
    args = parser.parse_args()

    print("Loading MMDiT generator (this may take a moment) ...")
    kwargs = {"model_dir": args.model_dir} if args.model_dir else {}
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
