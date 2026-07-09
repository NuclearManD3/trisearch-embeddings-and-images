#!/usr/bin/env python3
"""
TriSearch curated dataset format (HuggingFace-friendly).

On-disk layout (local or Hub)::

    <root>/
      README.md                 # dataset card
      dataset_info.json         # schema + version + stats
      data/
        train-00000-of-000NN.parquet
        ...
      images/                   # optional sidecar JPEGs (viewer / imagefolder)
        general/<id>.jpg
        satellite/<id>.jpg

Each row (parquet / in-memory) has:

| field            | type           | description                                      |
|------------------|----------------|--------------------------------------------------|
| id               | string         | stable unique id                                 |
| domain           | string         | ``"general"`` or ``"satellite"``                 |
| source           | string         | upstream dataset id                              |
| captions         | list[string]   | ≥2 human/source captions for the image           |
| query            | string         | search-style query that should find this image   |
| unrelated_query  | string         | search-style distractor on a different topic     |
| image            | Image          | RGB, 512×512 (embedded in parquet as JPEG bytes) |

Load with::

    from datasets import load_from_disk, load_dataset
    ds = load_from_disk("models/data/trisearch-v1")
    # or after push_to_hub:
    ds = load_dataset("user/trisearch-v1", split="train")
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

DATASET_FORMAT_VERSION = 1
DEFAULT_IMAGE_SIZE = 512
DEFAULT_DATASET_ROOT = Path("models/data/trisearch-v1")
DOMAIN_GENERAL = "general"
DOMAIN_SATELLITE = "satellite"
VALID_DOMAINS = frozenset({DOMAIN_GENERAL, DOMAIN_SATELLITE})

REQUIRED_FIELDS = (
    "id",
    "domain",
    "source",
    "captions",
    "query",
    "unrelated_query",
    "image",
)


def resize_square_rgb(image: Image.Image, size: int = DEFAULT_IMAGE_SIZE) -> Image.Image:
    """Resize to ``size×size`` RGB by *shrink/expand then center-crop*.

    Pipeline (preserves as much scene as possible vs crop-first):

    1. Convert to RGB.
    2. Uniform scale so the **shortest** side becomes ``size``
       (``scale = size / min(w, h)``). Wide images become ``(>size)×size``;
       tall images become ``size×(>size)``. Upscales if the image is smaller.
    3. Center-crop the long side to ``size×size``.

    Example: ``2000×1000`` → scale 0.512 → ``1024×512`` → crop → ``512×512``
    (keeps full height). Crop-first would take a ``1000×1000`` center then
    downscale and throw away half the width *before* shrinking.
    """
    img = image.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size {img.size}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    short = min(w, h)
    scale = size / float(short)
    new_w = max(size, int(round(w * scale)))
    new_h = max(size, int(round(h * scale)))
    # Guard rounding so the short side is never below ``size``.
    if new_w < size:
        new_w = size
    if new_h < size:
        new_h = size

    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        w, h = img.size

    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def caption_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def caption_jaccard(a: str, b: str) -> float:
    """Token Jaccard similarity in ``[0, 1]`` (1 = identical bag of words)."""
    ta, tb = caption_tokens(a), caption_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def captions_are_near_duplicate(
    a: str,
    b: str,
    *,
    max_jaccard: float = 0.72,
) -> bool:
    """True when two captions are effectively the same (RSICD-style paraphrases)."""
    na = re.sub(r"\s+", " ", str(a).strip().lower().rstrip("."))
    nb = re.sub(r"\s+", " ", str(b).strip().lower().rstrip("."))
    if not na or not nb:
        return True
    if na == nb:
        return True
    # One is a trivial extension of the other.
    if na in nb or nb in na:
        shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
        if len(shorter) >= 12 and len(longer) - len(shorter) <= 24:
            return True
    return caption_jaccard(na, nb) >= max_jaccard


def normalize_captions(
    captions: Sequence[Any],
    *,
    min_count: int = 2,
    max_jaccard: float = 0.72,
) -> list[str]:
    """Deduplicate captions, dropping near-duplicates (not only exact matches).

    RSICD often lists five captions that only differ by casing, punctuation, or
    a single preposition (in/at). Those must not count as multi-caption diversity.
    """
    cleaned: list[str] = []
    for cap in captions:
        text = str(cap).strip()
        if not text:
            continue
        if any(
            captions_are_near_duplicate(text, kept, max_jaccard=max_jaccard)
            for kept in cleaned
        ):
            continue
        cleaned.append(text)
    if len(cleaned) < min_count:
        raise ValueError(
            f"Need at least {min_count} diverse captions, got {len(cleaned)}: {cleaned!r}"
        )
    return cleaned


def caption_set_is_diverse(
    captions: Sequence[str],
    *,
    min_count: int = 2,
    max_jaccard: float = 0.72,
) -> bool:
    try:
        normalize_captions(captions, min_count=min_count, max_jaccard=max_jaccard)
        return True
    except ValueError:
        return False


def validate_record(record: dict[str, Any], *, require_image: bool = True) -> None:
    for key in REQUIRED_FIELDS:
        if key == "image" and not require_image:
            continue
        if key not in record:
            raise ValueError(f"Missing field {key!r}")
    domain = str(record["domain"])
    if domain not in VALID_DOMAINS:
        raise ValueError(f"domain must be one of {sorted(VALID_DOMAINS)}, got {domain!r}")
    captions = record["captions"]
    if not isinstance(captions, (list, tuple)) or len(captions) < 2:
        raise ValueError("captions must be a list of ≥2 strings")
    for field in ("id", "source", "query", "unrelated_query"):
        if not str(record[field]).strip():
            raise ValueError(f"{field} must be non-empty")
    if require_image:
        image = record["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"image must be PIL.Image, got {type(image)}")
        if image.size != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
            raise ValueError(
                f"image must be {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE}, got {image.size}"
            )


def features_spec():
    from datasets import Features, Image as HFImage, Sequence, Value

    return Features({
        "id": Value("string"),
        "domain": Value("string"),
        "source": Value("string"),
        "captions": Sequence(Value("string")),
        "query": Value("string"),
        "unrelated_query": Value("string"),
        "image": HFImage(),
    })


DEFAULT_WRITE_CHUNK = 64  # rows per parquet flush (keeps peak RAM bounded)


def records_to_dataset(records: Iterable[dict[str, Any]]):
    """Build a HuggingFace ``Dataset`` from in-memory records (small batches only)."""
    from datasets import Dataset

    rows = list(records)
    if not rows:
        raise ValueError("No records to export")
    for row in rows:
        validate_record(row, require_image=True)
    return Dataset.from_list(rows, features=features_spec())


def _write_info_and_card(
    output_dir: Path,
    *,
    num_rows: int,
    domains: dict[str, int],
    sources: dict[str, int],
) -> None:
    info = {
        "format_version": DATASET_FORMAT_VERSION,
        "num_rows": num_rows,
        "image_size": DEFAULT_IMAGE_SIZE,
        "domains": domains,
        "sources": sources,
        "fields": list(REQUIRED_FIELDS),
        "hf_disk_path": "hf",
        "parquet_glob": "data/train-*.parquet",
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        _dataset_card_markdown(info), encoding="utf-8"
    )


def save_dataset_streaming(
    staged_rows: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    chunk_size: int = DEFAULT_WRITE_CHUNK,
    write_sidecar_jpegs: bool = True,
    jpeg_quality: int = 92,
    write_hf_arrow: bool = True,
) -> Path:
    """Write dataset from staged rows without loading all images at once.

    Each staged row must include either:
      - ``image``: PIL.Image, or
      - ``image_path``: path to a JPEG/PNG on disk (preferred; low RAM)

    Images are opened, written into parquet shards of ``chunk_size``, then
    released. Peak memory is O(chunk_size) images, not O(N).
    """
    import gc
    import shutil

    from datasets import Dataset

    output_dir = Path(output_dir)
    if output_dir.exists():
        # Remove previous export contents carefully (keep parent).
        for name in ("data", "hf", "images", "metadata.jsonl",
                     "dataset_info.json", "README.md"):
            p = output_dir / name
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_root = output_dir / "images"

    if not staged_rows:
        raise ValueError("No records to export")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    domains: dict[str, int] = {"general": 0, "satellite": 0}
    sources: dict[str, int] = {}
    meta_path = output_dir / "metadata.jsonl"
    parquet_paths: list[Path] = []
    chunk: list[dict[str, Any]] = []
    total = 0
    shard_i = 0

    def _materialize(row: dict[str, Any]) -> dict[str, Any]:
        image = row.get("image")
        if image is None:
            path = row.get("image_path")
            if not path:
                raise ValueError(f"Row {row.get('id')} missing image/image_path")
            image = Image.open(path).convert("RGB")
        if image.size != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
            image = resize_square_rgb(image, DEFAULT_IMAGE_SIZE)
        rec = {
            "id": str(row["id"]),
            "domain": str(row["domain"]),
            "source": str(row["source"]),
            "captions": list(row["captions"]),
            "query": str(row["query"]),
            "unrelated_query": str(row["unrelated_query"]),
            "image": image,
        }
        validate_record(rec, require_image=True)
        return rec

    def _flush() -> None:
        nonlocal chunk, shard_i, total
        if not chunk:
            return
        ds = Dataset.from_list(chunk, features=features_spec())
        # Unknown final shard count: write provisional names, rename later.
        out = data_dir / f"train-part-{shard_i:05d}.parquet"
        ds.to_parquet(str(out))
        parquet_paths.append(out)
        shard_i += 1
        total += len(chunk)
        # Drop PIL refs
        for r in chunk:
            r["image"] = None
        chunk = []
        del ds
        gc.collect()

    with open(meta_path, "w", encoding="utf-8") as meta_fh:
        for row in staged_rows:
            rec = _materialize(row)
            domains[rec["domain"]] = domains.get(rec["domain"], 0) + 1
            sources[rec["source"]] = sources.get(rec["source"], 0) + 1

            if write_sidecar_jpegs:
                side = images_root / rec["domain"] / f"{rec['id']}.jpg"
                side.parent.mkdir(parents=True, exist_ok=True)
                # Prefer already-staged jpeg copy when present.
                src_path = row.get("image_path")
                if src_path and Path(src_path).is_file():
                    shutil.copy2(src_path, side)
                else:
                    rec["image"].save(side, format="JPEG", quality=jpeg_quality)
                meta_fh.write(json.dumps({
                    "file_name": f"images/{rec['domain']}/{rec['id']}.jpg",
                    "id": rec["id"],
                    "domain": rec["domain"],
                    "source": rec["source"],
                    "captions": rec["captions"],
                    "query": rec["query"],
                    "unrelated_query": rec["unrelated_query"],
                }, ensure_ascii=False) + "\n")

            chunk.append(rec)
            if len(chunk) >= chunk_size:
                _flush()
                if total % (chunk_size * 10) == 0:
                    print(f"  wrote {total:,} rows ...", flush=True)

        _flush()

    # Rename parts to train-XXXXX-of-YYYYY.parquet
    n_shards = len(parquet_paths)
    for i, path in enumerate(parquet_paths):
        final = data_dir / f"train-{i:05d}-of-{n_shards:05d}.parquet"
        path.rename(final)

    # Optional Arrow export for load_from_disk. Prefer parquet + metadata.jsonl
    # for large sets (loaders already support both).
    hf_dir = output_dir / "hf"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)

    if write_hf_arrow and write_sidecar_jpegs and meta_path.is_file():
        def _gen():
            with open(meta_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    meta = json.loads(line)
                    img = Image.open(output_dir / meta["file_name"]).convert("RGB")
                    yield {
                        "id": meta["id"],
                        "domain": meta["domain"],
                        "source": meta["source"],
                        "captions": meta["captions"],
                        "query": meta["query"],
                        "unrelated_query": meta["unrelated_query"],
                        "image": img,
                    }

        ds = Dataset.from_generator(_gen, features=features_spec())
        ds.save_to_disk(str(hf_dir))
        del ds
        gc.collect()
    elif write_hf_arrow:
        def _gen_pq():
            for i in range(n_shards):
                p = data_dir / f"train-{i:05d}-of-{n_shards:05d}.parquet"
                shard = Dataset.from_parquet(str(p))
                for ex in shard:
                    yield ex
                del shard
                gc.collect()

        ds = Dataset.from_generator(_gen_pq, features=features_spec())
        ds.save_to_disk(str(hf_dir))
        del ds
        gc.collect()

    _write_info_and_card(
        output_dir, num_rows=total, domains=domains, sources=sources
    )
    print(f"  export complete: {total:,} rows, {n_shards} parquet shard(s)", flush=True)
    return output_dir


def save_dataset(
    records: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    max_shard_size: str = "500MB",
    write_sidecar_jpegs: bool = True,
    chunk_size: int = DEFAULT_WRITE_CHUNK,
    write_hf_arrow: bool = True,
) -> Path:
    """Write HF dataset. Delegates to streaming saver (bounded RAM)."""
    del max_shard_size  # kept for call-site compatibility
    return save_dataset_streaming(
        records,
        output_dir,
        chunk_size=chunk_size,
        write_sidecar_jpegs=write_sidecar_jpegs,
        write_hf_arrow=write_hf_arrow,
    )


def load_dataset_records(
    dataset_dir: str | Path,
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load curated TriSearch dataset from ``save_dataset`` output.

    Prefer ``metadata.jsonl`` + sidecar JPEGs when present (random access,
    lower overhead than decoding every parquet row). ``max_samples`` stops early.
    """
    from datasets import load_from_disk, load_dataset

    root = Path(dataset_dir)
    meta_path = root / "metadata.jsonl"
    records: list[dict[str, Any]] = []

    if meta_path.is_file() and (root / "images").is_dir():
        with open(meta_path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if max_samples is not None and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                meta = json.loads(line)
                img_path = root / meta["file_name"]
                image = Image.open(img_path).convert("RGB")
                rec = {
                    "id": str(meta["id"]),
                    "domain": str(meta["domain"]),
                    "source": str(meta["source"]),
                    "captions": list(meta["captions"]),
                    "query": str(meta["query"]),
                    "unrelated_query": str(meta["unrelated_query"]),
                    "image": image,
                }
                validate_record(rec, require_image=True)
                records.append(rec)
        return records

    if (root / "hf").is_dir():
        ds = load_from_disk(str(root / "hf"))
    elif (root / "data").is_dir() and any((root / "data").glob("*.parquet")):
        ds = load_dataset(
            "parquet",
            data_files=str(root / "data" / "*.parquet"),
            split="train",
        )
    else:
        raise FileNotFoundError(
            f"No TriSearch dataset at {root} (expected hf/ or data/*.parquet)"
        )

    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        image = row["image"]
        if not isinstance(image, Image.Image):
            from trisearch_dataset import load_pil_image

            image = load_pil_image(image)
        rec = {
            "id": str(row["id"]),
            "domain": str(row["domain"]),
            "source": str(row["source"]),
            "captions": list(row["captions"]),
            "query": str(row["query"]),
            "unrelated_query": str(row["unrelated_query"]),
            "image": image.convert("RGB"),
        }
        validate_record(rec, require_image=True)
        records.append(rec)
    return records


def _dataset_card_markdown(info: dict[str, Any]) -> str:
    return f"""---
license: other
task_categories:
  - image-to-text
  - retrieval
language:
  - en
size_categories:
  - 10K<n<100K
tags:
  - remote-sensing
  - satellite
  - multimodal
  - matryoshka
  - trisearch
---

# TriSearch curated Stage-1 dataset

Format version **{info["format_version"]}**.

- **Rows**: {info["num_rows"]:,}
- **Image size**: {info["image_size"]}×{info["image_size"]} RGB
- **Domains**: {info["domains"]}
- **Sources**: {info["sources"]}

## Schema

Each example:

- `image` — RGB JPEG, square {info["image_size"]}
- `captions` — list of ≥2 captions
- `query` — search-style query for this image
- `unrelated_query` — distractor search query
- `domain` — `general` or `satellite`
- `source` — upstream dataset id
- `id` — unique string id

## Load

```python
from datasets import load_from_disk
ds = load_from_disk("hf")  # relative to this folder
# or: load_dataset("parquet", data_files="data/*.parquet", split="train")
```

Built for TriSearch Stage-1 contrastive + text–text training.
"""
