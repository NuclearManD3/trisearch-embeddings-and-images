#!/usr/bin/env python3
"""
Index a sample of the **TriSearch curated dataset**, embed with SigLIP, search
with Qwen3 text (ColBERT-style late interaction).

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

Run::

  python3 demo_image_search.py
  python3 demo_image_search.py --phase 1 --count 1000 --rebuild-index
  # Newest mid-training snapshot (history/step-*), not only completed stage1/
  python3 demo_image_search.py --latest-checkpoint --count 200 --rebuild-index
  python3 demo_image_search.py --checkpoint-dir models/trained/stage1/history/step-1500
"""

from __future__ import annotations

import argparse
import base64
import html
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import gradio as gr
import torch
from PIL import Image
from tqdm import tqdm

from trisearch_dataset import (
    DEFAULT_TRISEARCH_HF_DATASET,
    TriSearchMapDataset,
    normalize_training_text,
    open_trisearch_map_dataset,
)
from trisearch_models import (
    MAX_TRAINING_PHASE,
    MIN_TRAINING_PHASE,
    Qwen3MoeEmbedder,
    SiglipEmbedder,
    describe_phase,
    late_interaction_score,
    resolve_inference_checkpoint,
)

DEFAULT_COUNT = 100
DEFAULT_CACHE_DIR = "models/demo_index"
CACHE_VERSION = 4  # embeddings + record id; images re-fetched from map


@dataclass
class IndexedImage:
    """Embedding + caption; pixels are **not** retained (lazy via map index)."""

    image_id: int
    caption: str
    embeddings: torch.Tensor  # (num_patches, D) float32 CPU
    record_id: str = ""


class ImageSearchIndex:
    """Embeddings in RAM; PIL only while embedding a batch or rendering top-k."""

    def __init__(
        self,
        entries: list[IndexedImage] | None = None,
        *,
        image_source: Sequence[Any] | None = None,
    ):
        self.entries = entries or []
        # Map dataset (or list) for on-demand image decode by image_id.
        self.image_source = image_source

    def __len__(self) -> int:
        return len(self.entries)

    def get_image(self, entry: IndexedImage) -> Image.Image:
        if self.image_source is None:
            raise RuntimeError("No image_source attached; cannot decode pixels")
        row = self.image_source[entry.image_id]
        img = row["image"] if isinstance(row, dict) else row
        if not isinstance(img, Image.Image):
            from trisearch_dataset import load_pil_image

            img = load_pil_image(img)
        return img.convert("RGB")

    @torch.no_grad()
    def build_from_map(
        self,
        map_ds: TriSearchMapDataset | Sequence[Any],
        vision: SiglipEmbedder,
        *,
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> None:
        """Embed every map index; keep only tensors + captions (drop PIL after batch)."""
        from trisearch_models import matryoshka_normalize

        self.entries = []
        self.image_source = map_ds
        n = len(map_ds)
        batch_size = max(1, batch_size)
        starts = range(0, n, batch_size)
        if show_progress:
            starts = tqdm(starts, desc="Embedding images", unit="batch")

        for start in starts:
            end = min(start + batch_size, n)
            # Decode only this batch.
            chunk_meta: list[dict[str, Any]] = []
            images: list[Image.Image] = []
            for i in range(start, end):
                row = map_ds[i]
                chunk_meta.append(row)
                images.append(row["image"])

            inputs = vision.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(vision.device)
            hidden = vision.model(pixel_values=pixel_values).last_hidden_state
            hidden = hidden.to(dtype=vision.projection.weight.dtype)
            projected = vision.projection(hidden)
            normed = matryoshka_normalize(projected)

            for offset, row in enumerate(chunk_meta):
                cap = normalize_training_text(
                    row.get("caption")
                    or (row.get("captions") or [""])[0]
                )
                self.entries.append(
                    IndexedImage(
                        image_id=start + offset,
                        caption=cap,
                        embeddings=normed[offset].detach().float().cpu(),
                        record_id=str(row.get("id") or ""),
                    )
                )
            # Explicitly drop batch pixels before next decode.
            del images, chunk_meta, inputs, pixel_values, hidden, projected, normed

        print(
            f"Index built: {len(self):,} embeddings "
            f"(peak image batch ≤ {batch_size}; full decode not retained)",
            flush=True,
        )

    def save(self, path: Path, meta: dict[str, Any]) -> None:
        """Cache embeddings only (no multi-GB image blob dump)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "meta": meta,
            "entries": [
                {
                    "image_id": e.image_id,
                    "caption": e.caption,
                    "record_id": e.record_id,
                    "embeddings": e.embeddings,
                }
                for e in self.entries
            ],
        }
        torch.save(payload, path)
        print(f"Saved index ({len(self):,} embeddings, no full images) to {path}")

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        image_source: Sequence[Any] | None = None,
    ) -> tuple["ImageSearchIndex", dict[str, Any]]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ver = int(payload.get("version", 0))
        if ver < 4:
            raise ValueError(
                f"Index cache {path} is version {ver}; rebuild with --rebuild-index "
                f"(v{CACHE_VERSION}+ stores embeddings only, images from map dataset)."
            )
        entries = [
            IndexedImage(
                image_id=int(raw["image_id"]),
                # Re-normalize so older caches (pre-lowercase) still display lower.
                caption=normalize_training_text(raw.get("caption")),
                embeddings=raw["embeddings"],
                record_id=str(raw.get("record_id") or ""),
            )
            for raw in payload["entries"]
        ]
        return cls(entries, image_source=image_source), dict(payload["meta"])

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


def _pil_to_jpeg_bytes(image: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def cache_path_for(
    cache_dir: Path,
    *,
    dataset: str,
    split: str,
    count: int,
    seed: int,
    phase: int,
    satellite_fraction: float,
    checkpoint_tag: str = "stage",
) -> Path:
    safe = dataset.replace("/", "__").replace(":", "_")
    ckpt = checkpoint_tag.replace("/", "_").replace(" ", "")
    frac = f"{satellite_fraction:.2f}".replace(".", "p")
    return (
        cache_dir
        / f"{safe}_{split}_n{count}_seed{seed}_phase{phase}_sat{frac}"
        f"_ckpt-{ckpt}_v{CACHE_VERSION}.pt"
    )


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
    # e.g. history/step-1500 → step-1500
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


def build_or_load_index(
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

    # Same map used for build and for lazy image fetch after load.
    map_ds = open_demo_map(args)

    if cache_file.is_file() and not args.rebuild_index:
        print(f"Loading cached embeddings from {cache_file} ...")
        try:
            index, meta = ImageSearchIndex.load(cache_file, image_source=map_ds)
            if len(index) != len(map_ds):
                print(
                    f"  cache size {len(index)} != map size {len(map_ds)}; rebuilding"
                )
            else:
                print(
                    f"  {len(index):,} embeddings | dataset={meta.get('dataset')} "
                    f"| phase={meta.get('phase')} | ckpt={meta.get('checkpoint_tag')} "
                    f"| images via map on demand"
                )
                return index
        except ValueError as exc:
            print(f"  {exc}")

    if vision is None:
        vision_device = args.vision_device or args.device or (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        vision, _, checkpoint_tag = _make_embedders(
            args, vision_device=vision_device, text_device=vision_device
        )

    index = ImageSearchIndex()
    index.build_from_map(
        map_ds,
        vision,
        batch_size=args.batch_size,
        show_progress=not args.quiet,
    )
    if not args.no_cache:
        index.save(
            cache_file,
            meta={
                "dataset": dataset_label,
                "split": split_label,
                "count": args.count,
                "seed": args.seed,
                "phase": args.phase,
                "satellite_fraction": args.satellite_fraction,
                "version": CACHE_VERSION,
                "hf_dataset": args.hf_dataset,
                "checkpoint_tag": checkpoint_tag,
            },
        )
    return index


def create_search_fn(index: ImageSearchIndex, text_embedder: Qwen3MoeEmbedder):
    @torch.no_grad()
    def search(query: str, top_k: int):
        q = (query or "").strip()
        if not q:
            return (
                "<p>Enter a text query to search.</p>",
                "Enter a text query to search.",
            )
        query_embeddings = [
            t.detach().float().cpu() for t in text_embedder.embed_text(q)
        ]
        hits = index.search(query_embeddings, top_k=int(top_k))
        gallery_html_parts = []
        lines = []
        for rank, (score, entry) in enumerate(hits, 1):
            cap = normalize_training_text(entry.caption)
            lines.append(f"{rank}. [{score:.3f}] {cap[:120]}")
            # Decode **only** hit images (top-k), not the full index.
            img = index.get_image(entry)
            b64 = base64.b64encode(_pil_to_jpeg_bytes(img, quality=80)).decode("ascii")
            gallery_html_parts.append(
                f'<div style="display:inline-block;margin:6px;text-align:center">'
                f'<img src="data:image/jpeg;base64,{b64}" '
                f'style="max-width:180px;max-height:180px;border-radius:6px"/>'
                f'<div style="font-size:12px;max-width:180px">'
                f"{rank}. {html.escape(cap[:80])}</div></div>"
            )
        return "".join(gallery_html_parts), "\n".join(lines)

    return search


def build_ui(
    index: ImageSearchIndex,
    text_embedder: Qwen3MoeEmbedder,
    *,
    phase: int,
    data_desc: str,
):
    search_fn = create_search_fn(index, text_embedder)
    with gr.Blocks(title="TriSearch image search") as demo:
        gr.Markdown(
            f"## TriSearch image search\n"
            f"Phase **{phase}** · **{len(index):,}** indexed · {data_desc}\n\n"
            f"_Lazy map dataset: images decode per embed-batch / top-k only._"
        )
        query = gr.Textbox(
            label="Search query",
            placeholder="e.g. aerial view of airport runways",
        )
        top_k = gr.Slider(1, 24, value=12, step=1, label="Top-k")
        btn = gr.Button("Search", variant="primary")
        gallery = gr.HTML()
        results = gr.Textbox(label="Scores", lines=12)
        _evt = dict(api_name=False)
        btn.click(search_fn, inputs=[query, top_k], outputs=[gallery, results], **_evt)
        query.submit(
            search_fn, inputs=[query, top_k], outputs=[gallery, results], **_evt
        )
        gr.Examples(
            examples=[
                ["agricultural fields and farmland", 12],
                ["people on a beach at sunset", 12],
                ["dense forest canopy from above", 8],
                ["a city street with cars", 8],
            ],  # keep examples lowercase (matches training text policy)
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
        raise SystemExit("--checkpoint-dir overrides --latest-checkpoint / --latest-any")

    vision_device, text_device = _resolve_demo_devices(args)
    args.vision_device = vision_device
    args.text_device = text_device

    vision, text_embedder, ckpt_tag = _make_embedders(
        args, vision_device=vision_device, text_device=text_device
    )
    index = build_or_load_index(args, vision=vision, checkpoint_tag=ckpt_tag)

    data_desc = (
        f"`{args.hf_dataset}` split={args.curated_split} "
        f"sat≈{args.satellite_fraction:.0%} · batch_size={args.batch_size} "
        f"· ckpt={ckpt_tag}"
    )
    print(f"Text embedder ready ({text_device}); checkpoint tag={ckpt_tag}")

    demo = build_ui(index, text_embedder, phase=args.phase, data_desc=data_desc)
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
