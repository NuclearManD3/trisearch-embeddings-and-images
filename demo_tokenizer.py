#!/usr/bin/env python3
"""
Terminal tokenizer playground for the TriSearch text tower (Qwen3 tokenizer).

Prints the tokenized text as the same string with each token color-coded
(ANSI), plus a compact id table. No HTTP server.

  python3 demo_tokenizer.py
  python3 demo_tokenizer.py "a red barn in the snow"
  python3 demo_tokenizer.py --no-lower "Don't You'll"
  python3 demo_tokenizer.py --tokenizer-dir models/trained/stage1/text_model
  echo "hello world" | python3 demo_tokenizer.py -

Tokenizer only (no model weights). Prefers stage-1 / seed tokenizer dirs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from trisearch_models import QWEN_DIR, QWEN_TOKENIZER_ID, resolve_model_dir

# 256-color / basic ANSI backgrounds (cycle).
_ANSI_BG = [
    48,  # dark gray
    22,  # dark green
    23,  # teal
    24,  # blue-teal
    17,  # navy
    52,  # dark red
    53,  # purple
    54,
    58,  # olive
    94,  # orange-brown
    130,
    25,
    26,
    27,
    28,
    29,
]


def _c(text: str, bg: int) -> str:
    # Bright foreground on colored bg for readability in most terminals.
    return f"\033[38;5;15;48;5;{bg}m{text}\033[0m"


def resolve_tokenizer_source(
    *,
    phase: int = 1,
    tokenizer_dir: str | None = None,
    tokenizer_id: str | None = None,
) -> str:
    if tokenizer_dir:
        return str(tokenizer_dir)
    try:
        model_dir = Path(resolve_model_dir(phase, "qwen"))
        if (model_dir / "tokenizer.json").is_file() or (
            model_dir / "tokenizer_config.json"
        ).is_file():
            return str(model_dir)
    except (FileNotFoundError, ValueError):
        pass
    seed = Path(QWEN_DIR)
    if (seed / "tokenizer.json").is_file() or (seed / "tokenizer_config.json").is_file():
        return str(seed)
    return tokenizer_id or QWEN_TOKENIZER_ID


def load_tokenizer(source: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def tokenize_pieces(
    tokenizer,
    text: str,
    *,
    add_special_tokens: bool = False,
) -> list[dict[str, Any]]:
    text = text if text is not None else ""
    try:
        enc = tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = list(enc["input_ids"])
        offsets = list(enc["offset_mapping"])
    except Exception:
        ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        offsets = None

    pieces: list[dict[str, Any]] = []
    for i, tid in enumerate(ids):
        tok_str = tokenizer.convert_ids_to_tokens(int(tid))
        surface = tokenizer.decode(
            [int(tid)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        start = end = None
        span = surface
        if offsets is not None and i < len(offsets):
            start, end = int(offsets[i][0]), int(offsets[i][1])
            if end > start and 0 <= start <= len(text):
                span = text[start:end]
        pieces.append(
            {
                "index": i,
                "id": int(tid),
                "token": tok_str if tok_str is not None else "",
                "surface": surface,
                "span": span,
                "start": start,
                "end": end,
            }
        )
    return pieces


def colorize_ansi(text: str, pieces: list[dict[str, Any]], *, use_color: bool) -> str:
    if not pieces:
        return "(empty)"

    use_offsets = all(
        p.get("start") is not None and p.get("end") is not None for p in pieces
    )
    out: list[str] = []
    if use_offsets and text:
        cursor = 0
        for i, p in enumerate(pieces):
            start, end = p["start"], p["end"]
            bg = _ANSI_BG[i % len(_ANSI_BG)]
            if end > start:
                if start > cursor:
                    out.append(text[cursor:start])
                chunk = text[start:end].replace(" ", "·")
                out.append(_c(chunk, bg) if use_color else f"[{chunk}]")
                cursor = end
            else:
                # special / empty span
                chunk = (p["surface"] or p["token"] or "?").replace(" ", "·")
                out.append(_c(f"«{chunk}»", bg) if use_color else f"«{chunk}»")
        if cursor < len(text):
            out.append(text[cursor:])
    else:
        for i, p in enumerate(pieces):
            bg = _ANSI_BG[i % len(_ANSI_BG)]
            chunk = (p["surface"] or p["token"] or "?").replace(" ", "·")
            out.append(_c(chunk, bg) if use_color else f"[{chunk}]")
    return "".join(out)


def print_table(pieces: list[dict[str, Any]]) -> None:
    print(f"{'#':>4}  {'id':>8}  {'token':<20}  surface")
    print("-" * 56)
    for p in pieces:
        span = (p["span"] or p["surface"] or "").replace("\n", "↵")
        print(f"{p['index']:>4}  {p['id']:>8}  {p['token']:<20}  {span!r}")


def run_once(
    tokenizer,
    text: str,
    *,
    add_special: bool,
    lowercase: bool,
    use_color: bool,
    source: str,
) -> None:
    if lowercase:
        from trisearch_dataset import normalize_training_text

        text = normalize_training_text(text)
    pieces = tokenize_pieces(
        tokenizer, text, add_special_tokens=add_special
    )
    ids = [p["id"] for p in pieces]
    print(f"tokenizer: {source}")
    print(f"chars={len(text)}  tokens={len(pieces)}")
    print()
    print(colorize_ansi(text, pieces, use_color=use_color))
    print()
    print_table(pieces)
    print()
    print(f"input_ids = {ids}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "text",
        nargs="*",
        help="Text to tokenize. Use '-' to read stdin. "
             "If omitted, enter interactive mode.",
    )
    p.add_argument("--phase", type=int, default=1)
    p.add_argument("--tokenizer-dir", type=str, default=None)
    p.add_argument("--tokenizer-id", type=str, default=None)
    p.add_argument(
        "--special",
        action="store_true",
        help="Add special tokens (BOS/EOS if the tokenizer defines them).",
    )
    p.add_argument(
        "--no-lower",
        action="store_true",
        help="Do not lowercase (default lowercases like training).",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors (use [brackets] instead).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = resolve_tokenizer_source(
        phase=args.phase,
        tokenizer_dir=args.tokenizer_dir,
        tokenizer_id=args.tokenizer_id,
    )
    print(f"Loading tokenizer from {source!r} ...", file=sys.stderr)
    tokenizer = load_tokenizer(source)

    use_color = not args.no_color and sys.stdout.isatty()
    lowercase = not args.no_lower

    # Explicit text args
    if args.text:
        if len(args.text) == 1 and args.text[0] == "-":
            raw = sys.stdin.read()
        else:
            raw = " ".join(args.text)
        run_once(
            tokenizer,
            raw,
            add_special=args.special,
            lowercase=lowercase,
            use_color=use_color,
            source=source,
        )
        return 0

    # Interactive REPL
    print(
        "Interactive mode (empty line or Ctrl-D to quit). "
        f"lower={lowercase} special={args.special}",
        file=sys.stderr,
    )
    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        if line == "":
            break
        run_once(
            tokenizer,
            line,
            add_special=args.special,
            lowercase=lowercase,
            use_color=use_color,
            source=source,
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
