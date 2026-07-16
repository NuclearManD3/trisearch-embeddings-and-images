#!/usr/bin/env python3
"""Shared demo embedding index: build, cache, caption search, image decode.

Used by ``demo_image_search.py`` (retrieval) and ``demo_stage2_recon.py``
(reconstruction). Stores **embeddings only** on disk; PIL pixels are re-decoded
from the map dataset on demand (never a multi‑GB image blob cache).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from tqdm import tqdm

from trisearch_dataset import (
    TriSearchMapDataset,
    normalize_training_text,
)
from trisearch_models import late_interaction_score, matryoshka_normalize

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
        vision: Any,
        *,
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> None:
        """Embed every map index; keep only tensors + captions (drop PIL after batch)."""
        self.entries = []
        self.image_source = map_ds
        n = len(map_ds)
        batch_size = max(1, batch_size)
        starts = range(0, n, batch_size)
        if show_progress:
            starts = tqdm(starts, desc="Embedding images", unit="batch")

        for start in starts:
            end = min(start + batch_size, n)
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
        """ColBERT-style late-interaction ranking (stage-1 retrieval demo)."""
        scored: list[tuple[float, IndexedImage]] = []
        for entry in self.entries:
            score = late_interaction_score(query_embeddings, entry.embeddings)
            scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:top_k]

    def search_captions_plaintext(
        self,
        query: str,
        top_k: int = 4,
    ) -> list[tuple[float, IndexedImage]]:
        """Case-insensitive caption substring / token-overlap ranking.

        Score = 1.0 if query is a contiguous substring of the caption, else
        Jaccard overlap of whitespace tokens (0 if empty). Higher is better.
        """
        q = normalize_training_text(query)
        if not q:
            return []
        q_tokens = {t for t in q.split() if t}
        scored: list[tuple[float, IndexedImage]] = []
        for entry in self.entries:
            cap = normalize_training_text(entry.caption)
            if not cap:
                continue
            if q in cap:
                score = 1.0 + min(len(q) / max(len(cap), 1), 1.0)
            elif q_tokens:
                cap_tokens = {t for t in cap.split() if t}
                if not cap_tokens:
                    continue
                inter = len(q_tokens & cap_tokens)
                if inter == 0:
                    continue
                score = inter / len(q_tokens | cap_tokens)
            else:
                continue
            scored.append((float(score), entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[: max(1, int(top_k))]


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


def build_or_load_index(
    *,
    map_ds: TriSearchMapDataset | Sequence[Any],
    vision: Any | None,
    cache_file: Path,
    rebuild: bool = False,
    no_cache: bool = False,
    batch_size: int = 4,
    quiet: bool = False,
    meta: dict[str, Any] | None = None,
) -> ImageSearchIndex:
    """Load embedding cache if valid; otherwise build with ``vision`` and save."""
    if cache_file.is_file() and not rebuild:
        print(f"Loading cached embeddings from {cache_file} ...")
        try:
            index, loaded_meta = ImageSearchIndex.load(
                cache_file, image_source=map_ds
            )
            if len(index) != len(map_ds):
                print(
                    f"  cache size {len(index)} != map size {len(map_ds)}; rebuilding"
                )
            else:
                print(
                    f"  {len(index):,} embeddings | dataset={loaded_meta.get('dataset')} "
                    f"| phase={loaded_meta.get('phase')} | "
                    f"ckpt={loaded_meta.get('checkpoint_tag')} "
                    f"| images via map on demand"
                )
                return index
        except ValueError as exc:
            print(f"  {exc}")

    if vision is None:
        raise RuntimeError(
            "Vision embedder required to build the demo index "
            "(cache miss or --rebuild-index)."
        )

    index = ImageSearchIndex()
    index.build_from_map(
        map_ds,
        vision,
        batch_size=batch_size,
        show_progress=not quiet,
    )
    if not no_cache:
        index.save(cache_file, meta=meta or {})
    return index
