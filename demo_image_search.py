#!/usr/bin/env python3
"""
Index a random subset of images from a HuggingFace dataset, embed them with the
SigLIP vision tower, then search by text using ColBERT-style late interaction.

Default dataset: ``jxie/flickr8k`` (PIL images + natural-language captions).
For ``JessicaYuan/ChatEarthNet``, pass ``--image-root`` pointing at the PNG
files that accompany the captions.

Run:
  python3 demo_image_search.py
  python3 demo_image_search.py --phase 1 --count 1000
  python3 demo_image_search.py --dataset JessicaYuan/ChatEarthNet --image-root /path/to/images

The Gradio UI opens after indexing (or after loading a cached index).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from PIL import Image
from tqdm import tqdm

from trisearch_dataset import (
    DEFAULT_FLICKR8K_DATASET,
    DEFAULT_FLICKR8K_SPLIT,
    load_dataset_samples,
)
from trisearch_models import (
    MAX_TRAINING_PHASE,
    MIN_TRAINING_PHASE,
    Qwen3MoeEmbedder,
    SiglipEmbedder,
    describe_phase,
    late_interaction_score,
)

DEFAULT_DATASET = DEFAULT_FLICKR8K_DATASET
DEFAULT_SPLIT = DEFAULT_FLICKR8K_SPLIT
DEFAULT_COUNT = 1000
DEFAULT_CACHE_DIR = "models/demo_index"
CACHE_VERSION = 1


@dataclass
class IndexedImage:
    image_id: int
    caption: str
    image: Image.Image
    embeddings: torch.Tensor  # (num_patches, D) float32 on CPU


def _default_vision_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _default_text_device() -> str:
    if torch.cuda.device_count() >= 2:
        return "cuda:1"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _jpeg_bytes_to_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")





def cache_path_for(
    cache_dir: Path,
    *,
    dataset: str,
    split: str,
    count: int,
    seed: int,
    phase: int,
) -> Path:
    safe = dataset.replace("/", "__")
    return cache_dir / f"{safe}_{split}_n{count}_seed{seed}_phase{phase}.pt"


class ImageSearchIndex:
    """In-memory gallery of image patch embeddings for late-interaction search."""

    def __init__(self, entries: list[IndexedImage] | None = None):
        self.entries = entries or []

    def __len__(self) -> int:
        return len(self.entries)

    @torch.no_grad()
    def build(
        self,
        samples: list[dict[str, Any]],
        vision: SiglipEmbedder,
        *,
        batch_size: int = 4,
        show_progress: bool = True,
    ):
        from trisearch_models import matryoshka_normalize

        self.entries = []
        batch_size = max(1, batch_size)
        batches = range(0, len(samples), batch_size)
        if show_progress:
            batches = tqdm(batches, desc="Embedding images", unit="batch")

        for start in batches:
            chunk = samples[start : start + batch_size]
            images = [sample["image"] for sample in chunk]
            inputs = vision.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(vision.device)
            hidden = vision.model(pixel_values=pixel_values).last_hidden_state
            hidden = hidden.to(dtype=vision.projection.weight.dtype)
            projected = vision.projection(hidden)
            normed = matryoshka_normalize(projected)

            for offset, sample in enumerate(chunk):
                patch_list = normed[offset].detach().float().cpu()
                image_id = start + offset
                self.entries.append(
                    IndexedImage(
                        image_id=image_id,
                        caption=sample["caption"],
                        image=sample["image"],
                        embeddings=patch_list,
                    )
                )

    def save(self, path: Path, meta: dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "meta": meta,
            "entries": [
                {
                    "image_id": e.image_id,
                    "caption": e.caption,
                    "image_jpeg": _image_to_jpeg_bytes(e.image),
                    "embeddings": e.embeddings.half(),
                }
                for e in self.entries
            ],
        }
        torch.save(payload, path)
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        print(f"Saved index ({len(self.entries):,} images) to {path}")

    @classmethod
    def load(cls, path: Path) -> tuple[ImageSearchIndex, dict[str, Any]]:
        payload = torch.load(path, map_location="cpu")
        if payload.get("version") != CACHE_VERSION:
            raise ValueError(f"Unsupported cache version in {path}")
        entries = []
        for item in payload["entries"]:
            entries.append(
                IndexedImage(
                    image_id=int(item["image_id"]),
                    caption=str(item["caption"]),
                    image=_jpeg_bytes_to_image(item["image_jpeg"]),
                    embeddings=item["embeddings"].float(),
                )
            )
        return cls(entries), dict(payload["meta"])

    @torch.no_grad()
    def search(
        self,
        query_embeddings: list[torch.Tensor],
        top_k: int = 12,
    ) -> list[tuple[float, IndexedImage]]:
        scored: list[tuple[float, IndexedImage]] = []
        for entry in self.entries:
            score = late_interaction_score(query_embeddings, entry.embeddings)
            scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:top_k]


def build_or_load_index(args: argparse.Namespace) -> ImageSearchIndex:
    cache_file = cache_path_for(
        Path(args.cache_dir),
        dataset=args.dataset,
        split=args.split,
        count=args.count,
        seed=args.seed,
        phase=args.phase,
    )
    if cache_file.is_file() and not args.rebuild_index:
        print(f"Loading cached index from {cache_file} ...")
        index, meta = ImageSearchIndex.load(cache_file)
        print(
            f"  {len(index):,} images | dataset={meta.get('dataset')} "
            f"| phase={meta.get('phase')}"
        )
        return index

    vision_device = args.vision_device or args.device or _default_vision_device()
    print(f"Vision embedder on {vision_device}: {describe_phase(args.phase, 'siglip')}")
    vision = SiglipEmbedder(phase=args.phase, device=vision_device)

    samples = load_dataset_samples(
        dataset=args.dataset,
        split=args.split,
        count=args.count,
        seed=args.seed,
        image_column=args.image_column,
        caption_column=args.caption_column,
        image_root=args.image_root,
    )
    index = ImageSearchIndex()
    index.build(
        samples,
        vision,
        batch_size=args.batch_size,
        show_progress=not args.quiet,
    )
    if not args.no_cache:
        meta = {
            "dataset": args.dataset,
            "split": args.split,
            "count": args.count,
            "seed": args.seed,
            "phase": args.phase,
            "num_indexed": len(index),
        }
        index.save(cache_file, meta)
    return index


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _hits_to_html(hits: list[tuple[float, IndexedImage]]) -> str:
    if not hits:
        return "<p>No results.</p>"
    cards = []
    for rank, (score, hit) in enumerate(hits, start=1):
        caption = html.escape(hit.caption)
        cards.append(
            "<div class='hit-card'>"
            f"<img src='{_image_to_data_url(hit.image)}' "
            f"alt='result {hit.image_id}' />"
            f"<div class='meta'>#{rank} · id {hit.image_id} · "
            f"score {score:.4f}</div>"
            f"<div class='caption'>{caption}</div>"
            "</div>"
        )
    return (
        "<style>"
        ".hit-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px;}"
        ".hit-card{border:1px solid #ddd;border-radius:8px;padding:8px;background:#fafafa;}"
        ".hit-card img{width:100%;height:180px;object-fit:contain;background:#fff;}"
        ".hit-card .meta{font-weight:600;margin-top:8px;font-size:0.9rem;}"
        ".hit-card .caption{margin-top:4px;font-size:0.85rem;color:#444;}"
        "</style>"
        f"<div class='hit-grid'>{''.join(cards)}</div>"
    )


def create_search_fn(index: ImageSearchIndex, text_embedder: Qwen3MoeEmbedder):
    @torch.no_grad()
    def search(query: str, top_k: int):
        query = (query or "").strip()
        if not query:
            return "<p>Enter a text query to search.</p>", "Enter a text query to search."
        query_embeddings = [
            t.detach().float().cpu() for t in text_embedder.embed_text(query)
        ]
        hits = index.search(query_embeddings, top_k=int(top_k))
        lines = [
            f"{rank:2d}. score={score:8.4f}  #{hit.image_id}  {hit.caption[:120]}"
            for rank, (score, hit) in enumerate(hits, start=1)
        ]
        summary = (
            f"Query: {query!r}\n"
            f"Indexed images: {len(index):,}\n"
            f"Top {len(hits)} results (late-interaction MaxSim):\n"
            + "\n".join(lines)
        )
        return _hits_to_html(hits), summary

    return search


def build_ui(index: ImageSearchIndex, text_embedder: Qwen3MoeEmbedder, phase: int):
    search_fn = create_search_fn(index, text_embedder)

    with gr.Blocks(title="TriSearch image search") as demo:
        gr.Markdown(
            f"## TriSearch image search\n"
            f"Search **{len(index):,}** indexed images with natural-language "
            f"queries using ColBERT-style late interaction.\n\n"
            f"Training phase **{phase}** "
            f"(vision + text towers from `{describe_phase(phase, 'siglip')}`)"
        )
        with gr.Row():
            query = gr.Textbox(
                label="Search query",
                placeholder="e.g. a dog running in the snow",
                scale=3,
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=24,
                value=12,
                step=1,
                label="Results",
            )
            btn = gr.Button("Search", variant="primary")
        gallery = gr.HTML(label="Top matches")
        results = gr.Textbox(label="Scores", lines=12)

        btn.click(search_fn, inputs=[query, top_k], outputs=[gallery, results])
        query.submit(search_fn, inputs=[query, top_k], outputs=[gallery, results])

        gr.Examples(
            examples=[
                ["a dog running in the snow", 12],
                ["people on a beach at sunset", 12],
                ["a city street with cars", 8],
                ["trees and grass in a park", 8],
            ],
            inputs=[query, top_k],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"HuggingFace dataset name (default: {DEFAULT_DATASET}).")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help="Number of random images to index.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--caption-column", default=None,
                        help="Caption field name (auto-detected when omitted).")
    parser.add_argument("--image-root", default=None,
                        help="Directory of image files for path-only datasets "
                             "(e.g. ChatEarthNet).")

    parser.add_argument("--phase", type=int, default=1,
                        choices=range(MIN_TRAINING_PHASE, MAX_TRAINING_PHASE + 1),
                        help="Training phase for both embedders (1=stage-1 trained).")
    parser.add_argument("--device", default=None,
                        help="Fallback device for both towers if --vision-gpu/--text-gpu unset.")
    parser.add_argument("--vision-gpu", type=int, default=0,
                        help="GPU index for SigLIP (default 0).")
    parser.add_argument("--text-gpu", type=int, default=1,
                        help="GPU index for Qwen3 (default 1, falls back to 0 on 1-GPU).")

    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true",
                        help="Do not write an embedding cache to disk.")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Ignore any existing cache and re-embed everything.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Images per GPU batch while building the index.")

    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link.")
    return parser.parse_args()


def _resolve_demo_devices(args: argparse.Namespace) -> tuple[str, str]:
    if args.device:
        return args.device, args.device
    vision = (
        f"cuda:{args.vision_gpu}"
        if torch.cuda.is_available()
        else "cpu"
    )
    text_gpu = args.text_gpu
    if torch.cuda.is_available() and text_gpu >= torch.cuda.device_count():
        text_gpu = args.vision_gpu
    text = f"cuda:{text_gpu}" if torch.cuda.is_available() else "cpu"
    return vision, text


def main():
    args = parse_args()
    vision_device, text_device = _resolve_demo_devices(args)
    args.vision_device = vision_device
    args.text_device = text_device
    index = build_or_load_index(args)

    print(f"Text embedder on {text_device}: {describe_phase(args.phase, 'qwen')}")
    text_embedder = Qwen3MoeEmbedder(phase=args.phase, device=text_device)

    demo = build_ui(index, text_embedder, phase=args.phase)
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