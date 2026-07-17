#!/usr/bin/env python3
"""
Index a sample of the **TriSearch curated dataset**, embed with SigLIP, search
with Qwen3 text (ColBERT-style late interaction).

For each hit, optionally overlay a **query heatmap**: per-patch MaxSim
(max cosine of that SigLIP patch vs any query token), reshaped to the vision
patch grid and blended onto the image. Shows which regions support the match.

Project data
------------
Uses **only** ``NuclearManD/trisearch-dataset-64k-v0.0.1`` (or a local curated
export). COCO / SkyScript / Flickr feed ``generate_datasets.py`` only.

Memory (same policy as training)
--------------------------------
- Opens a **map-style** curated view (HF cache on disk).
- Decodes images **only** for the current embed batch, then drops them.
- Search results re-decode **top-k** images on demand (not the full index).
- Peak image RAM ≈ ``batch_size`` (build) or ``top_k`` (search), not ``count``.
- Embedding cache lives in ``trisearch_demo_index`` (shared with stage-2 demo).

Run::

  python3 demo_image_search.py
  python3 demo_image_search.py --phase 1 --count 1000 --rebuild-index
  python3 demo_image_search.py --latest-checkpoint --count 200 --rebuild-index
  python3 demo_image_search.py --checkpoint-dir models/trained/stage1/history/step-1500
"""

from __future__ import annotations

import argparse
import base64
import html
import io
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from trisearch_dataset import (
    DEFAULT_TRISEARCH_HF_DATASET,
    TriSearchMapDataset,
    normalize_training_text,
    open_trisearch_map_dataset,
)
from trisearch_demo_index import (
    DEFAULT_CACHE_DIR,
    ImageSearchIndex,
    build_or_load_index,
    cache_path_for,
)
from trisearch_models import (
    MAX_TRAINING_PHASE,
    MIN_TRAINING_PHASE,
    Qwen3MoeEmbedder,
    SiglipEmbedder,
    describe_phase,
    overlay_patch_heatmap,
    patch_query_affinity,
    resolve_inference_checkpoint,
)

DEFAULT_COUNT = 100


def _pil_to_jpeg_bytes(image: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _resolve_embedder_checkpoint(args: argparse.Namespace) -> Path | None:
    """Return Stage-1 checkpoint root, or None to use phase seed/stage defaults."""
    if args.phase == 0 and not args.checkpoint_dir and not args.latest_checkpoint:
        return None
    try:
        return resolve_inference_checkpoint(
            phase=args.phase,
            checkpoint_dir=args.checkpoint_dir,
            latest_history=bool(args.latest_checkpoint)
            and not args.checkpoint_dir
            and not args.latest_any,
            latest_any=bool(args.latest_any) and not args.checkpoint_dir,
        )
    except FileNotFoundError:
        if args.latest_checkpoint or args.checkpoint_dir or args.latest_any:
            raise
        return None


def _checkpoint_tag(root: Path | None, phase: int) -> str:
    if root is None:
        return f"phase{phase}"
    if root.name.startswith("step-"):
        return root.name
    return root.name or "stage"


def _make_embedders(
    args: argparse.Namespace,
    *,
    vision_device: str,
    text_device: str,
) -> tuple[SiglipEmbedder, Qwen3MoeEmbedder, str]:
    """Build vision + text embedders from phase and/or a concrete checkpoint root."""
    ckpt = _resolve_embedder_checkpoint(args)
    if ckpt is not None:
        vision_dir = str(ckpt / "vision_model")
        text_dir = str(ckpt / "text_model")
        proj = str(ckpt / "projection_heads.pt")
        print(f"Loading embedders from checkpoint: {ckpt}", flush=True)
        vision = SiglipEmbedder(
            model_dir=vision_dir,
            phase=max(args.phase, 1),
            projection_path=proj,
            device=vision_device,
        )
        text = Qwen3MoeEmbedder(
            model_dir=text_dir,
            phase=max(args.phase, 1),
            projection_path=proj,
            device=text_device,
        )
        return vision, text, _checkpoint_tag(ckpt, args.phase)

    print(
        f"Vision embedder on {vision_device}: {describe_phase(args.phase, 'siglip')}",
        flush=True,
    )
    vision = SiglipEmbedder(phase=args.phase, device=vision_device)
    text = Qwen3MoeEmbedder(phase=args.phase, device=text_device)
    return vision, text, _checkpoint_tag(None, args.phase)


def open_demo_map(args: argparse.Namespace) -> TriSearchMapDataset:
    """Lazy curated view: index selection only; images decode on demand."""
    if args.count < 1:
        raise ValueError("--count must be >= 1")
    return open_trisearch_map_dataset(
        hf_dataset=args.hf_dataset,
        dataset_dir=args.curated_dataset_dir,
        prefer_local=args.prefer_local_curated,
        split=args.curated_split,
        max_samples=args.count,
        seed=args.seed,
        satellite_fraction=args.satellite_fraction,
    )


def _load_index(
    args: argparse.Namespace,
    *,
    vision: SiglipEmbedder | None = None,
    checkpoint_tag: str = "stage",
) -> ImageSearchIndex:
    dataset_label = f"trisearch:{args.hf_dataset}"
    split_label = args.curated_split
    cache_file = cache_path_for(
        Path(args.cache_dir),
        dataset=dataset_label,
        split=split_label,
        count=args.count,
        seed=args.seed,
        phase=args.phase,
        satellite_fraction=args.satellite_fraction,
        checkpoint_tag=checkpoint_tag,
    )
    map_ds = open_demo_map(args)
    return build_or_load_index(
        map_ds=map_ds,
        vision=vision,
        cache_file=cache_file,
        rebuild=args.rebuild_index,
        no_cache=args.no_cache,
        batch_size=args.batch_size,
        quiet=args.quiet,
        meta={
            "dataset": dataset_label,
            "split": split_label,
            "count": args.count,
            "seed": args.seed,
            "phase": args.phase,
            "satellite_fraction": args.satellite_fraction,
            "hf_dataset": args.hf_dataset,
            "checkpoint_tag": checkpoint_tag,
        },
    )


def _vision_patch_grid(vision: SiglipEmbedder) -> tuple[int, int]:
    """``(H, W)`` patch grid from the loaded SigLIP config."""
    cfg = vision.model.config
    image_size = int(getattr(cfg, "image_size", 0) or 0)
    patch_size = int(getattr(cfg, "patch_size", 0) or 0)
    if image_size > 0 and patch_size > 0 and image_size % patch_size == 0:
        side = image_size // patch_size
        return side, side
    return (0, 0)


def _thumb_b64(image: Image.Image, *, max_side: int = 220, quality: int = 80) -> str:
    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / max(w, h, 1))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.Resampling.BILINEAR,
        )
    return base64.b64encode(_pil_to_jpeg_bytes(img, quality=quality)).decode("ascii")


def create_search_fn(
    index: ImageSearchIndex,
    text_embedder: Qwen3MoeEmbedder,
    *,
    patch_grid: tuple[int, int] | None = None,
):
    """Build Gradio search callback with optional MaxSim heatmaps."""

    @torch.no_grad()
    def search(
        query: str,
        top_k: int,
        show_heatmap: bool,
        heatmap_alpha: float,
        heatmap_peak_tau: float,
    ):
        q = (query or "").strip()
        if not q:
            return (
                "<p>Enter a text query to search.</p>",
                "Enter a text query to search.",
            )
        query_embeddings = [
            t.detach().float().cpu() for t in text_embedder.embed_text(q)
        ]
        print("\n".join(str(i) for i in query_embeddings))
        print()
        hits = index.search(query_embeddings, top_k=int(top_k))
        gallery_html_parts = []
        lines = []
        alpha = float(heatmap_alpha)
        for rank, (score, entry) in enumerate(hits, 1):
            cap = normalize_training_text(entry.caption)
            lines.append(f"{rank}. [{score:.3f}] {cap[:120]}")
            img = index.get_image(entry)
            orig_b64 = _thumb_b64(img)

            cell_imgs = [
                (
                    f'<img src="data:image/jpeg;base64,{orig_b64}" '
                    f'title="original" '
                    f'style="max-width:180px;max-height:180px;border-radius:6px;'
                    f'display:block;margin:0 auto"/>'
                )
            ]
            if show_heatmap:
                try:
                    n_p = int(entry.embeddings.shape[0])
                    gh, gw = patch_grid if patch_grid and patch_grid[0] > 0 else (0, 0)
                    grid_hw = (gh, gw) if gh * gw == n_p else None
                    aff = patch_query_affinity(
                        query_embeddings, entry.embeddings, reduce="max"
                    )
                    heat = overlay_patch_heatmap(
                        img,
                        aff,
                        grid_hw=grid_hw,
                        alpha=alpha,
                        mode="peak",
                        peak_temperature=float(heatmap_peak_tau),
                    )
                    heat_b64 = _thumb_b64(heat)
                    cell_imgs.append(
                        f'<img src="data:image/jpeg;base64,{heat_b64}" '
                        f'title="query patch MaxSim heatmap" '
                        f'style="max-width:180px;max-height:180px;border-radius:6px;'
                        f'display:block;margin:0 auto"/>'
                    )
                except Exception as exc:
                    lines.append(f"   (heatmap failed: {exc})")

            width = 190 * len(cell_imgs) + 12
            gallery_html_parts.append(
                f'<div style="display:inline-block;margin:8px;text-align:center;'
                f'vertical-align:top;max-width:{width}px">'
                f'<div style="display:flex;gap:4px;justify-content:center">'
                f'{"".join(cell_imgs)}</div>'
                f'<div style="font-size:12px;max-width:{width}px;margin-top:4px">'
                f"{rank}. [{score:.3f}] {html.escape(cap[:80])}</div>"
                f'<div style="font-size:10px;color:#666">'
                f'{"orig | heatmap" if show_heatmap and len(cell_imgs) > 1 else "orig"}'
                f"</div></div>"
            )
        return "".join(gallery_html_parts), "\n".join(lines)

    return search


def build_ui(
    index: ImageSearchIndex,
    text_embedder: Qwen3MoeEmbedder,
    *,
    phase: int,
    data_desc: str,
    patch_grid: tuple[int, int] | None = None,
):
    search_fn = create_search_fn(index, text_embedder, patch_grid=patch_grid)
    grid_note = ""
    if patch_grid and patch_grid[0] > 0:
        grid_note = (
            f" Heatmaps: MaxSim per SigLIP patch "
            f"({patch_grid[0]}×{patch_grid[1]} grid → image regions)."
        )
    with gr.Blocks(title="TriSearch image search") as demo:
        gr.Markdown(
            f"## TriSearch image search\n"
            f"Phase **{phase}** · **{len(index):,}** indexed · {data_desc}\n\n"
            f"_Lazy map dataset: images decode per embed-batch / top-k only._"
            f"{grid_note}"
        )
        query = gr.Textbox(
            label="Search query",
            placeholder="e.g. aerial view of airport runways",
        )
        with gr.Row():
            top_k = gr.Slider(1, 24, value=12, step=1, label="Top-k")
            show_heatmap = gr.Checkbox(
                value=True,
                label="Show query heatmaps",
                info=(
                    "Peak-relative heat: h=exp((s−s_max)/τ). Only patches near "
                    "the best match on that image light up; high but non-peak "
                    "cosine stays cold."
                ),
            )
            heatmap_alpha = gr.Slider(
                0.15,
                0.85,
                value=0.5,
                step=0.05,
                label="Heatmap opacity",
            )
            heatmap_peak_tau = gr.Slider(
                0.02,
                0.20,
                value=0.06,
                step=0.01,
                label="Heatmap peak τ",
                info="Smaller → colder non-peaks (sharper match focus).",
            )
        btn = gr.Button("Search", variant="primary")
        gallery = gr.HTML()
        results = gr.Textbox(label="Scores", lines=12)
        _evt = dict(api_name=False)
        inputs = [query, top_k, show_heatmap, heatmap_alpha, heatmap_peak_tau]
        btn.click(search_fn, inputs=inputs, outputs=[gallery, results], **_evt)
        query.submit(search_fn, inputs=inputs, outputs=[gallery, results], **_evt)
        gr.Examples(
            examples=[
                ["agricultural fields and farmland", 12],
                ["people on a beach at sunset", 12],
                ["dense forest canopy from above", 8],
                ["a city street with cars", 8],
            ],
            inputs=[query, top_k],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-dataset",
        default=DEFAULT_TRISEARCH_HF_DATASET,
        help=f"TriSearch curated Hub id (default {DEFAULT_TRISEARCH_HF_DATASET}).",
    )
    parser.add_argument(
        "--curated-dataset-dir",
        default=None,
        help="Optional local curated export when --prefer-local-curated.",
    )
    parser.add_argument(
        "--prefer-local-curated",
        action="store_true",
        help="Prefer local curated export over the Hub dataset.",
    )
    parser.add_argument(
        "--curated-split",
        default="train",
        choices=("train", "test", "all"),
        help="Official split to sample (default train).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="How many map indices to embed (default 100). "
        "Not a hard RAM cap — only batch_size images are decoded at once.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--satellite-fraction",
        type=float,
        default=0.5,
        help="Target satellite share when sampling (default 0.5).",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=range(MIN_TRAINING_PHASE, MAX_TRAINING_PHASE + 1),
        help="Training phase when not using --latest-checkpoint / --checkpoint-dir "
        "(default 1 = models/trained/stage1 completed tree).",
    )
    parser.add_argument(
        "--latest-checkpoint",
        action="store_true",
        help="Load newest mid-training snapshot under "
        "models/trained/stage1/history/step-* (not only the completed stage dir).",
    )
    parser.add_argument(
        "--latest-any",
        action="store_true",
        help="Load newest valid checkpoint by mtime among stage1/ and history/step-*.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Explicit Stage-1 checkpoint root "
        "(e.g. models/trained/stage1/history/step-1500).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--vision-gpu", type=int, default=0)
    parser.add_argument("--text-gpu", type=int, default=1)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Images decoded + embedded per step (peak image RAM).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _resolve_demo_devices(args: argparse.Namespace) -> tuple[str, str]:
    if args.device:
        return args.device, args.device
    if not torch.cuda.is_available():
        return "cpu", "cpu"
    n = torch.cuda.device_count()
    vision_gpu = args.vision_gpu if args.vision_gpu < n else 0
    text_gpu = args.text_gpu if args.text_gpu < n else vision_gpu
    return f"cuda:{vision_gpu}", f"cuda:{text_gpu}"


def main():
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if args.latest_checkpoint and args.latest_any:
        raise SystemExit("Use only one of --latest-checkpoint or --latest-any")
    if args.checkpoint_dir and (args.latest_checkpoint or args.latest_any):
        raise SystemExit(
            "--checkpoint-dir overrides --latest-checkpoint / --latest-any"
        )

    vision_device, text_device = _resolve_demo_devices(args)
    args.vision_device = vision_device
    args.text_device = text_device

    vision, text_embedder, ckpt_tag = _make_embedders(
        args, vision_device=vision_device, text_device=text_device
    )
    index = _load_index(args, vision=vision, checkpoint_tag=ckpt_tag)

    data_desc = (
        f"`{args.hf_dataset}` split={args.curated_split} "
        f"sat≈{args.satellite_fraction:.0%} · batch_size={args.batch_size} "
        f"· ckpt={ckpt_tag}"
    )
    print(f"Text embedder ready ({text_device}); checkpoint tag={ckpt_tag}")
    patch_grid = _vision_patch_grid(vision)
    if patch_grid[0] > 0:
        print(
            f"Patch heatmap grid: {patch_grid[0]}×{patch_grid[1]} "
            f"(image_size={vision.model.config.image_size}, "
            f"patch_size={vision.model.config.patch_size})"
        )
    del vision
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    demo = build_ui(
        index,
        text_embedder,
        phase=args.phase,
        data_desc=data_desc,
        patch_grid=patch_grid if patch_grid[0] > 0 else None,
    )
    print(f"\nOpening search UI at http://{args.host}:{args.port}")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
