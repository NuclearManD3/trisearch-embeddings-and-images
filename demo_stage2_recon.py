#!/usr/bin/env python3
"""
Stage-2 reconstruction demo: plaintext caption search + image | generation.

Uses the **shared** embedding cache from ``trisearch_demo_index`` (same files as
``demo_image_search.py``). For each hit, shows the original image beside an
MMDiT reconstruction from **shuffled** patch embeddings (same recipe as train).

Controls:
  * Search captions by plain text (substring / token overlap)
  * Default 4 results
  * **Reshuffle & regenerate** — new token permutations, same images

Run::

  python3 demo_stage2_recon.py --count 100
  python3 demo_stage2_recon.py --count 200 --generator-dir models/trained/stage2
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

from trisearch_dataset import (
    DEFAULT_TRISEARCH_HF_DATASET,
    open_trisearch_map_dataset,
)
from trisearch_demo_index import (
    DEFAULT_CACHE_DIR,
    IndexedImage,
    ImageSearchIndex,
    build_or_load_index,
    cache_path_for,
)
from trisearch_models import (
    MAX_TRAINING_PHASE,
    MIN_TRAINING_PHASE,
    MMDiTGenerator,
    SiglipEmbedder,
    describe_phase,
    resolve_inference_checkpoint,
)
from trisearch_models.inference import (
    CONDITIONING_HEADS_FILE,
    prepare_stage2_condition_tokens,
    resolve_model_dir,
)

DEFAULT_COUNT = 100
DEFAULT_TOP_K = 4


def _pil_to_b64(image: Image.Image, *, max_side: int = 256, quality: int = 85) -> str:
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


def _resolve_vision_ckpt(args: argparse.Namespace) -> Path | None:
    if args.vision_checkpoint_dir:
        return Path(args.vision_checkpoint_dir)
    try:
        return resolve_inference_checkpoint(
            phase=args.vision_phase,
            checkpoint_dir=None,
            latest_history=bool(args.latest_checkpoint),
            latest_any=False,
        )
    except FileNotFoundError:
        return None


def _make_vision(args: argparse.Namespace, device: str) -> tuple[SiglipEmbedder, str]:
    ckpt = _resolve_vision_ckpt(args)
    if ckpt is not None:
        tag = ckpt.name if ckpt.name.startswith("step-") else (ckpt.name or "stage")
        vision = SiglipEmbedder(
            model_dir=str(ckpt / "vision_model"),
            phase=max(args.vision_phase, 1),
            projection_path=str(ckpt / "projection_heads.pt"),
            device=device,
        )
        return vision, tag
    vision = SiglipEmbedder(phase=args.vision_phase, device=device)
    return vision, f"phase{args.vision_phase}"


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
    # Seed MMDiT (untrained recon)
    print(f"Loading seed MMDiT: {describe_phase(0, 'mmdit')}")
    return MMDiTGenerator(
        model_dir=resolve_model_dir(0, "mmdit"),
        phase=0,
        device=device,
    )


def _gallery_html(
    index: ImageSearchIndex,
    generator: MMDiTGenerator,
    entries: list[IndexedImage],
    *,
    steps: int,
    seed: int,
    shuffle: bool,
    embed_dropout: float,
    merge_prob: float,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Render side-by-side original | recon; return html, log, state rows."""
    parts: list[str] = []
    lines: list[str] = []
    state: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        cap = entry.caption
        orig = index.get_image(entry)
        emb = entry.embeddings.float()
        cond = prepare_stage2_condition_tokens(
            emb,
            shuffle=shuffle,
            drop_prob=embed_dropout,
            merge_prob=merge_prob,
            max_tokens=64,
            training=True,  # allow random shuffle/dropout/merge in demo
        )
        gen_img = generator.generate(
            embeddings=cond,
            num_inference_steps=int(steps),
            seed=int(seed) + i,
            shuffle_embeddings=False,  # already prepared
        )
        # Match display size roughly
        if gen_img.size != orig.size:
            gen_img = gen_img.resize(orig.size, Image.Resampling.BILINEAR)

        o_b64 = _pil_to_b64(orig)
        g_b64 = _pil_to_b64(gen_img)
        lines.append(
            f"{i + 1}. id={entry.image_id} tokens={tuple(cond.shape)} | {cap[:100]}"
        )
        parts.append(
            f'<div style="display:inline-block;margin:10px;text-align:center;'
            f'vertical-align:top;max-width:420px">'
            f'<div style="display:flex;gap:6px;justify-content:center">'
            f'<div><img src="data:image/jpeg;base64,{o_b64}" '
            f'style="max-width:200px;max-height:200px;border-radius:6px"/>'
            f'<div style="font-size:11px;color:#555">original</div></div>'
            f'<div><img src="data:image/jpeg;base64,{g_b64}" '
            f'style="max-width:200px;max-height:200px;border-radius:6px"/>'
            f'<div style="font-size:11px;color:#555">generated (shuffled cond)</div></div>'
            f"</div>"
            f'<div style="font-size:12px;margin-top:4px;max-width:400px">'
            f"{i + 1}. {html.escape(cap[:120])}</div></div>"
        )
        state.append(
            {
                "image_id": entry.image_id,
                "caption": cap,
                "record_id": entry.record_id,
            }
        )
    return "".join(parts), "\n".join(lines), state


def build_ui(
    index: ImageSearchIndex,
    generator: MMDiTGenerator,
    *,
    data_desc: str,
):
    # id → entry for reshuffle
    by_id = {e.image_id: e for e in index.entries}

    def search(
        query: str,
        top_k: int,
        steps: int,
        seed: int,
        shuffle: bool,
        drop: float,
        merge: float,
    ):
        q = (query or "").strip()
        if not q:
            return (
                "<p>Enter a caption search string.</p>",
                "Enter a caption search string.",
                [],
            )
        hits = index.search_captions_plaintext(q, top_k=int(top_k))
        if not hits:
            return (
                f"<p>No captions matched {html.escape(q)!r}.</p>",
                f"No matches for {q!r}",
                [],
            )
        entries = [e for _, e in hits]
        html_out, log, state = _gallery_html(
            index,
            generator,
            entries,
            steps=steps,
            seed=seed,
            shuffle=shuffle,
            embed_dropout=drop,
            merge_prob=merge,
        )
        return html_out, log, state

    def reshuffle(state, steps, seed, shuffle, drop, merge):
        if not state:
            return (
                "<p>Search first, then reshuffle.</p>",
                "No current results.",
                state or [],
            )
        entries: list[IndexedImage] = []
        for row in state:
            eid = int(row["image_id"])
            if eid in by_id:
                entries.append(by_id[eid])
        if not entries:
            return "<p>Entries missing from index.</p>", "Missing entries.", []
        # Bump seed so generation noise + shuffle both change.
        html_out, log, new_state = _gallery_html(
            index,
            generator,
            entries,
            steps=int(steps),
            seed=int(seed) + 17,
            shuffle=bool(shuffle),
            embed_dropout=float(drop),
            merge_prob=float(merge),
        )
        return html_out, log + "\n(reshuffled)", new_state

    with gr.Blocks(title="TriSearch Stage-2 recon") as demo:
        gr.Markdown(
            f"## Stage-2 reconstruction demo\n"
            f"{data_desc}\n\n"
            f"Plaintext **caption** search → original | MMDiT recon from "
            f"**shuffled** vision embeddings (same conditioning recipe as train)."
        )
        query = gr.Textbox(
            label="Caption search",
            placeholder="e.g. farm, beach, runway",
        )
        with gr.Row():
            top_k = gr.Slider(1, 16, value=DEFAULT_TOP_K, step=1, label="Top-k")
            steps = gr.Slider(1, 28, value=8, step=1, label="Denoise steps")
            seed = gr.Number(value=0, label="Seed", precision=0)
        with gr.Row():
            shuffle = gr.Checkbox(value=True, label="Shuffle tokens")
            drop = gr.Slider(
                0.0, 0.5, value=0.0, step=0.05, label="Embed dropout (demo)"
            )
            merge = gr.Slider(
                0.0, 1.0, value=0.0, step=0.05, label="Merge-to-one prob (demo)"
            )
        with gr.Row():
            btn = gr.Button("Search & generate", variant="primary")
            reshuffle_btn = gr.Button("Reshuffle & regenerate")
        gallery = gr.HTML()
        log = gr.Textbox(label="Log", lines=8)
        state = gr.State([])

        inputs = [query, top_k, steps, seed, shuffle, drop, merge]
        btn.click(search, inputs=inputs, outputs=[gallery, log, state], api_name=False)
        query.submit(
            search, inputs=inputs, outputs=[gallery, log, state], api_name=False
        )
        reshuffle_btn.click(
            reshuffle,
            inputs=[state, steps, seed, shuffle, drop, merge],
            outputs=[gallery, log, state],
            api_name=False,
        )
        gr.Examples(
            examples=[
                ["field", 4],
                ["beach", 4],
                ["road", 4],
                ["forest", 4],
            ],
            inputs=[query, top_k],
        )
    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-dataset", default=DEFAULT_TRISEARCH_HF_DATASET)
    p.add_argument("--curated-dataset-dir", default=None)
    p.add_argument("--prefer-local-curated", action="store_true")
    p.add_argument(
        "--curated-split", default="train", choices=("train", "test", "all")
    )
    p.add_argument("--count", type=int, default=DEFAULT_COUNT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--satellite-fraction", type=float, default=0.5)
    p.add_argument("--vision-phase", type=int, default=1)
    p.add_argument("--vision-checkpoint-dir", default=None)
    p.add_argument("--latest-checkpoint", action="store_true")
    p.add_argument(
        "--generator-dir",
        default="models/trained/stage2",
        help="Stage-2 root with mmdit/ + conditioning_heads.pt (seed if missing).",
    )
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--vision-gpu", type=int, default=0)
    p.add_argument("--generator-gpu", type=int, default=1)
    p.add_argument("--device", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be >= 1")

    if args.device:
        vdev, gdev = args.device, args.device
    elif torch.cuda.is_available():
        n = torch.cuda.device_count()
        vdev = f"cuda:{args.vision_gpu if args.vision_gpu < n else 0}"
        gdev = f"cuda:{args.generator_gpu if args.generator_gpu < n else 0}"
    else:
        vdev = gdev = "cpu"

    map_ds = open_trisearch_map_dataset(
        hf_dataset=args.hf_dataset,
        dataset_dir=args.curated_dataset_dir,
        prefer_local=args.prefer_local_curated,
        split=args.curated_split,
        max_samples=args.count,
        seed=args.seed,
        satellite_fraction=args.satellite_fraction,
    )

    vision, vtag = _make_vision(args, vdev)
    cache_file = cache_path_for(
        Path(args.cache_dir),
        dataset=f"trisearch:{args.hf_dataset}",
        split=args.curated_split,
        count=args.count,
        seed=args.seed,
        phase=args.vision_phase,
        satellite_fraction=args.satellite_fraction,
        checkpoint_tag=vtag,
    )
    index = build_or_load_index(
        map_ds=map_ds,
        vision=vision,
        cache_file=cache_file,
        rebuild=args.rebuild_index,
        no_cache=args.no_cache,
        batch_size=args.batch_size,
        quiet=args.quiet,
        meta={
            "dataset": f"trisearch:{args.hf_dataset}",
            "split": args.curated_split,
            "count": args.count,
            "seed": args.seed,
            "phase": args.vision_phase,
            "satellite_fraction": args.satellite_fraction,
            "checkpoint_tag": vtag,
        },
    )
    del vision
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generator = _make_generator(args, gdev)
    generator.eval()

    data_desc = (
        f"**{len(index):,}** embeddings · vision `{vtag}` · "
        f"gen `{args.generator_dir}` · `{args.hf_dataset}`"
    )
    demo = build_ui(index, generator, data_desc=data_desc)
    print(f"Opening Stage-2 recon demo at http://{args.host}:{args.port}")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
