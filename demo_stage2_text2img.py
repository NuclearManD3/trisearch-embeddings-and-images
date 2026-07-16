#!/usr/bin/env python3
"""
Stage-2 text→image demo: Qwen token embeddings → MMDiT generation.

Embeds a free-text query with the **Stage-1 text tower** (shared 1024-dim
Matryoshka space), optionally applies the same Stage-2 conditioning recipe
(shuffle / dropout / max tokens / merge), then runs the Stage-2 MMDiT several
times with different seeds to produce a small set of images.

Note: Stage 2 was trained on **vision** patch tokens; text works via the
shared embedding space (cross-modal transfer), not a dedicated text-to-image
objective.

Run::

  python3 demo_stage2_text2img.py
  python3 demo_stage2_text2img.py --generator-dir models/trained/stage2
  python3 demo_stage2_text2img.py --text-phase 1 --num-images 4 --steps 12
"""

from __future__ import annotations

import argparse
import base64
import html
import io
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from PIL import Image

from trisearch_models import (
    MMDiTGenerator,
    Qwen3MoeEmbedder,
    describe_phase,
    resolve_inference_checkpoint,
)
from trisearch_models.inference import (
    CONDITIONING_HEADS_FILE,
    prepare_stage2_condition_tokens,
    resolve_model_dir,
)
from trisearch_models.stage2 import (
    DEFAULT_EMBED_DROPOUT,
    DEFAULT_MAX_COND_TOKENS,
    DEFAULT_MERGE_PROB,
    DEFAULT_STAGE2_DIR,
)

DEFAULT_NUM_IMAGES = 4
DEFAULT_STEPS = 12
DEFAULT_SIZE = 512


def _pil_to_b64(image: Image.Image, *, max_side: int = 384, quality: int = 88) -> str:
    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / max(w, h, 1))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.Resampling.BILINEAR,
        )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resolve_text_ckpt(args: argparse.Namespace) -> Path | None:
    if args.text_checkpoint_dir:
        return Path(args.text_checkpoint_dir)
    try:
        return resolve_inference_checkpoint(
            phase=args.text_phase,
            checkpoint_dir=None,
            latest_history=bool(args.latest_checkpoint),
            latest_any=False,
        )
    except FileNotFoundError:
        return None


def _make_text_embedder(args: argparse.Namespace, device: str) -> tuple[Qwen3MoeEmbedder, str]:
    ckpt = _resolve_text_ckpt(args)
    if ckpt is not None:
        tag = ckpt.name if ckpt.name.startswith("step-") else (ckpt.name or "stage")
        text = Qwen3MoeEmbedder(
            model_dir=str(ckpt / "text_model"),
            phase=max(args.text_phase, 1),
            projection_path=str(ckpt / "projection_heads.pt"),
            device=device,
        )
        return text, tag
    text = Qwen3MoeEmbedder(phase=args.text_phase, device=device)
    return text, f"phase{args.text_phase}"


def _make_generator(args: argparse.Namespace, device: str) -> MMDiTGenerator:
    gen_root = Path(args.generator_dir) if args.generator_dir else None
    if gen_root and (gen_root / "mmdit").is_dir():
        model_dir = str(gen_root / "mmdit")
        cond = (
            str(gen_root / CONDITIONING_HEADS_FILE)
            if (gen_root / CONDITIONING_HEADS_FILE).is_file()
            else None
        )
        print(f"Loading Stage-2 generator from {gen_root}")
        return MMDiTGenerator(
            model_dir=model_dir,
            phase=2,
            device=device,
            conditioning_path=cond,
        )
    print(f"Loading seed MMDiT: {describe_phase(0, 'mmdit')}")
    return MMDiTGenerator(
        model_dir=resolve_model_dir(0, "mmdit"),
        phase=0,
        device=device,
    )


def _embed_query(
    text_model: Qwen3MoeEmbedder,
    query: str,
    *,
    merge_threshold: float,
) -> torch.Tensor:
    """Return ``(1, T, D)`` float32 Matryoshka token embeddings on CPU."""
    tokens = text_model.embed_text(query, merge_threshold=float(merge_threshold))
    stacked = torch.stack(
        [t.detach().float().cpu() if torch.is_tensor(t) else torch.as_tensor(t) for t in tokens],
        dim=0,
    )
    return stacked.unsqueeze(0)


def _generate_set(
    text_model: Qwen3MoeEmbedder,
    generator: MMDiTGenerator,
    query: str,
    *,
    num_images: int,
    steps: int,
    seed: int,
    height: int,
    width: int,
    shuffle: bool,
    embed_dropout: float,
    merge_prob: float,
    max_cond_tokens: int,
    token_merge_threshold: float,
) -> tuple[str, str, list[dict[str, Any]]]:
    q = (query or "").strip()
    if not q:
        return "<p>Enter a text query.</p>", "Empty query.", []

    emb = _embed_query(text_model, q, merge_threshold=token_merge_threshold)
    cond = prepare_stage2_condition_tokens(
        emb,
        shuffle=bool(shuffle),
        drop_prob=float(embed_dropout),
        merge_prob=float(merge_prob),
        max_tokens=int(max_cond_tokens),
        training=True,  # enable random shuffle/dropout/merge like train
    )
    # prepare may return (B,T,D) or (T,D)
    if cond.ndim == 2:
        cond_for_gen = cond
        n_tok = int(cond.shape[0])
    else:
        cond_for_gen = cond[0]
        n_tok = int(cond.shape[1])

    parts: list[str] = []
    lines: list[str] = [
        f"query={q!r} | raw_tokens={emb.shape[1]} | cond_tokens={n_tok} | "
        f"shape={tuple(cond_for_gen.shape)}"
    ]
    state: list[dict[str, Any]] = []
    n = max(1, int(num_images))
    for i in range(n):
        s = int(seed) + i
        # Fresh cond shuffle per image when shuffle is on (re-sample recipe).
        if shuffle or embed_dropout > 0 or merge_prob > 0:
            cond_i = prepare_stage2_condition_tokens(
                emb,
                shuffle=bool(shuffle),
                drop_prob=float(embed_dropout),
                merge_prob=float(merge_prob),
                max_tokens=int(max_cond_tokens),
                training=True,
            )
            if cond_i.ndim == 3:
                cond_i = cond_i[0]
        else:
            cond_i = cond_for_gen

        img = generator.generate(
            embeddings=cond_i,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            seed=s,
            shuffle_embeddings=False,
        )
        b64 = _pil_to_b64(img)
        lines.append(f"  [{i + 1}/{n}] seed={s} size={img.size}")
        parts.append(
            f'<div style="display:inline-block;margin:10px;text-align:center;'
            f'vertical-align:top">'
            f'<img src="data:image/jpeg;base64,{b64}" '
            f'style="max-width:280px;max-height:280px;border-radius:8px;'
            f'box-shadow:0 1px 4px rgba(0,0,0,.15)"/>'
            f'<div style="font-size:12px;margin-top:4px;color:#444">'
            f"#{i + 1} · seed {s}</div></div>"
        )
        state.append({"seed": s, "size": list(img.size)})

    header = (
        f'<div style="margin:8px 0 12px;font-size:14px">'
        f"<b>Query:</b> {html.escape(q)} "
        f"<span style='color:#666'>({n_tok} cond tokens → {n} images)</span>"
        f"</div>"
    )
    return header + "".join(parts), "\n".join(lines), state


def build_ui(
    text_model: Qwen3MoeEmbedder,
    generator: MMDiTGenerator,
    *,
    text_tag: str,
    gen_desc: str,
    default_size: int,
):
    def generate(
        query: str,
        num_images: int,
        steps: int,
        seed: int,
        height: int,
        width: int,
        shuffle: bool,
        drop: float,
        merge: float,
        max_tok: int,
        tok_merge: float,
    ):
        return _generate_set(
            text_model,
            generator,
            query,
            num_images=int(num_images),
            steps=int(steps),
            seed=int(seed),
            height=int(height),
            width=int(width),
            shuffle=bool(shuffle),
            embed_dropout=float(drop),
            merge_prob=float(merge),
            max_cond_tokens=int(max_tok),
            token_merge_threshold=float(tok_merge),
        )

    def regenerate(query, state, num_images, steps, seed, height, width, shuffle, drop, merge, max_tok, tok_merge):
        # Bump base seed so noise + cond sampling both change.
        return generate(
            query,
            num_images,
            steps,
            int(seed) + 17 + (len(state) if state else 0),
            height,
            width,
            shuffle,
            drop,
            merge,
            max_tok,
            tok_merge,
        )

    with gr.Blocks(title="TriSearch Stage-2 text→image") as demo:
        gr.Markdown(
            f"## Stage-2 text → image\n"
            f"**Text embedder:** `{html.escape(text_tag)}` (Qwen Matryoshka tokens)  \n"
            f"**Generator:** {html.escape(gen_desc)}\n\n"
            f"Type a query → text tower embeds → Stage-2 MMDiT samples several "
            f"images (different seeds). Conditioning uses the same shuffle / "
            f"dropout / merge recipe as Stage-2 training (vision-trained; text "
            f"is zero-shot via the shared space)."
        )
        query = gr.Textbox(
            label="Text query",
            placeholder="e.g. a sandy beach with turquoise water, satellite view of farmland",
            lines=2,
        )
        with gr.Row():
            num_images = gr.Slider(
                1, 8, value=DEFAULT_NUM_IMAGES, step=1, label="Number of images"
            )
            steps = gr.Slider(1, 40, value=DEFAULT_STEPS, step=1, label="Denoise steps")
            seed = gr.Number(value=0, label="Base seed", precision=0)
        with gr.Row():
            height = gr.Slider(
                256, 1024, value=default_size, step=64, label="Height"
            )
            width = gr.Slider(
                256, 1024, value=default_size, step=64, label="Width"
            )
        with gr.Row():
            shuffle = gr.Checkbox(value=True, label="Shuffle tokens")
            drop = gr.Slider(
                0.0, 0.5, value=0.0, step=0.05, label="Embed dropout"
            )
            merge = gr.Slider(
                0.0, 1.0, value=0.0, step=0.05, label="Merge-to-one prob"
            )
            max_tok = gr.Slider(
                1,
                128,
                value=DEFAULT_MAX_COND_TOKENS,
                step=1,
                label="Max cond tokens",
            )
            tok_merge = gr.Slider(
                0.5,
                1.0,
                value=1.0,
                step=0.01,
                label="Text consecutive-merge threshold",
            )
        with gr.Row():
            btn = gr.Button("Generate", variant="primary")
            again = gr.Button("Regenerate (new seeds)")
        gallery = gr.HTML()
        log = gr.Textbox(label="Log", lines=10)
        state = gr.State([])

        gen_inputs = [
            query,
            num_images,
            steps,
            seed,
            height,
            width,
            shuffle,
            drop,
            merge,
            max_tok,
            tok_merge,
        ]
        btn.click(
            generate, inputs=gen_inputs, outputs=[gallery, log, state], api_name="generate"
        )
        query.submit(
            generate, inputs=gen_inputs, outputs=[gallery, log, state], api_name=False
        )
        again.click(
            regenerate,
            inputs=[query, state, *gen_inputs[1:]],
            outputs=[gallery, log, state],
            api_name=False,
        )
        gr.Examples(
            examples=[
                ["a red stop sign on a rural road"],
                ["satellite view of green farmland and dirt roads"],
                ["a sandy beach with turquoise water and palm trees"],
                ["dense forest canopy from above"],
            ],
            inputs=[query],
        )
    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text-phase", type=int, default=1)
    p.add_argument(
        "--text-checkpoint-dir",
        default=None,
        help="Stage-1 root (text_model/ + projection_heads.pt).",
    )
    p.add_argument(
        "--latest-checkpoint",
        action="store_true",
        help="Use newest stage1 history/step-* for the text tower.",
    )
    p.add_argument(
        "--generator-dir",
        default=DEFAULT_STAGE2_DIR,
        help="Stage-2 root with mmdit/ + conditioning_heads.pt.",
    )
    p.add_argument("--text-gpu", type=int, default=0)
    p.add_argument("--generator-gpu", type=int, default=1)
    p.add_argument(
        "--device",
        default=None,
        help="Force a single device for both models.",
    )
    p.add_argument("--default-size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7862)
    p.add_argument("--share", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Load models, generate one tiny image, print api info, exit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.default_size % 64 != 0:
        raise SystemExit("--default-size must be a multiple of 64")

    if args.device:
        tdev = gdev = args.device
    elif torch.cuda.is_available():
        n = torch.cuda.device_count()
        tdev = f"cuda:{args.text_gpu if args.text_gpu < n else 0}"
        gdev = f"cuda:{args.generator_gpu if args.generator_gpu < n else 0}"
        # Prefer free GPU for the large generator when possible.
        if n >= 2 and tdev == gdev:
            gdev = f"cuda:{(args.generator_gpu + 1) % n}"
    else:
        tdev = gdev = "cpu"

    print("--- Stage-2 text→image demo ---")
    print(f"  text device      : {tdev}")
    print(f"  generator device : {gdev}")

    text_model, text_tag = _make_text_embedder(args, tdev)
    print(f"  text embedder    : {text_tag}")
    generator = _make_generator(args, gdev)
    gen_desc = args.generator_dir or "seed mmdit"
    print(f"  generator        : {gen_desc}")

    if args.smoke:
        # Minimal forward: 1 image, few steps, small spatial size.
        html_out, log, _ = _generate_set(
            text_model,
            generator,
            "a small red barn in a green field",
            num_images=1,
            steps=2,
            seed=0,
            height=256,
            width=256,
            shuffle=True,
            embed_dropout=0.0,
            merge_prob=0.0,
            max_cond_tokens=32,
            token_merge_threshold=1.0,
        )
        print(log)
        assert "seed=0" in log and "cond_tokens=" in log
        demo = build_ui(
            text_model,
            generator,
            text_tag=text_tag,
            gen_desc=str(gen_desc),
            default_size=256,
        )
        info = demo.get_api_info()
        print(f"Smoke OK: gallery html {len(html_out)} chars; api keys={list(info.keys()) if isinstance(info, dict) else type(info)}")
        return

    demo = build_ui(
        text_model,
        generator,
        text_tag=text_tag,
        gen_desc=str(gen_desc),
        default_size=args.default_size,
    )
    # Touch API wiring once before launch (Gradio event graph).
    _ = demo.get_api_info()
    print(f"Launching on http://{args.host}:{args.port}")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
