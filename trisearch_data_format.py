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
| image            | Image          | RGB, 1024×1024 (embedded in parquet as JPEG bytes) |

Load with::

    from datasets import load_from_disk, load_dataset
    ds = load_from_disk("models/data/trisearch-v1")
    # or after push_to_hub:
    ds = load_dataset("user/trisearch-v1", split="train")
"""

from __future__ import annotations

import io
import json
import math
import re
import shutil
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

DATASET_FORMAT_VERSION = 1
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_DATASET_ROOT = Path("models/data/trisearch-v1")
DOMAIN_GENERAL = "general"
DOMAIN_SATELLITE = "satellite"
VALID_DOMAINS = frozenset({DOMAIN_GENERAL, DOMAIN_SATELLITE})
# Rows per parquet shard. 256 was a legacy RAM cap when PIL images were held
# in memory (~40MB shards). With staged JPEG-bytes embedding, ~4k–8k rows is
# fine and yields ~0.6–1.2GB shards (better for Hub / fewer files).
DEFAULT_WRITE_CHUNK = 4096
DEFAULT_EXPORT_WORKERS = 16  # parallel image/sidecar I/O during export

# Official Hub splits (frozen for v0.0.1+)
OFFICIAL_SPLIT_SEED = 42
OFFICIAL_TEST_DENOM = 16  # test size = floor(n_domain / 16) per domain
VALID_SPLITS = frozenset({"train", "test"})

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

    Example (size=1024): ``2000×1000`` → ``2048×1024`` → crop → ``1024×1024``
    (keeps full height). Crop-first would discard width early.
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


def _prepare_export_row(
    row: dict[str, Any],
    *,
    write_sidecar_jpegs: bool,
    images_root: Path,
    jpeg_quality: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Materialize one row for parquet (+ optional sidecar). Thread-safe.

    Prefers raw staged JPEG bytes (no PIL decode/re-encode) when
    ``image_path`` points at a ``.jpg`` already produced by staging.
    """
    rid = str(row["id"])
    domain = str(row["domain"])
    source = str(row["source"])
    captions = list(row["captions"])
    query = str(row["query"])
    unrelated = str(row["unrelated_query"])

    if domain not in VALID_DOMAINS:
        raise ValueError(
            f"domain must be one of {sorted(VALID_DOMAINS)}, got {domain!r}"
        )
    if not isinstance(captions, list) or len(captions) < 2:
        raise ValueError(f"captions must be a list of ≥2 strings for {rid}")
    if not query.strip() or not unrelated.strip() or not source.strip():
        raise ValueError(f"empty text field for {rid}")

    jpeg_bytes: bytes | None = None
    src_path = row.get("image_path")
    pil = row.get("image")

    if isinstance(pil, Image.Image):
        if pil.size != (image_size, image_size):
            pil = resize_square_rgb(pil, image_size)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=jpeg_quality)
        jpeg_bytes = buf.getvalue()
        image_field: Any = {"bytes": jpeg_bytes, "path": f"{rid}.jpg"}
    elif src_path:
        path = Path(str(src_path))
        if not path.is_file():
            raise FileNotFoundError(f"{rid}: missing image {path}")
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            # Staging already wrote correct-size JPEGs — copy bytes as-is.
            jpeg_bytes = path.read_bytes()
            if len(jpeg_bytes) < 512:
                raise ValueError(f"{rid}: image too small ({len(jpeg_bytes)} B)")
            image_field = {"bytes": jpeg_bytes, "path": path.name}
        else:
            pil_img = Image.open(path).convert("RGB")
            if pil_img.size != (image_size, image_size):
                pil_img = resize_square_rgb(pil_img, image_size)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=jpeg_quality)
            jpeg_bytes = buf.getvalue()
            image_field = {"bytes": jpeg_bytes, "path": f"{rid}.jpg"}
    else:
        raise ValueError(f"Row {rid} missing image/image_path")

    rec = {
        "id": rid,
        "domain": domain,
        "source": source,
        "captions": captions,
        "query": query,
        "unrelated_query": unrelated,
        "image": image_field,
    }

    meta: dict[str, Any] | None = None
    if write_sidecar_jpegs:
        side = images_root / domain / f"{rid}.jpg"
        if src_path and Path(str(src_path)).is_file() and Path(str(src_path)).suffix.lower() in {
            ".jpg",
            ".jpeg",
        }:
            # copyfile is faster than copy2 (no metadata syscalls).
            shutil.copyfile(str(src_path), side)
        else:
            assert jpeg_bytes is not None
            side.write_bytes(jpeg_bytes)
        meta = {
            "file_name": f"images/{domain}/{rid}.jpg",
            "id": rid,
            "domain": domain,
            "source": source,
            "captions": captions,
            "query": query,
            "unrelated_query": unrelated,
        }
        if row.get("split"):
            meta["split"] = str(row["split"])
    return rec, meta


def assign_official_splits(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = OFFICIAL_SPLIT_SEED,
    test_denom: int = OFFICIAL_TEST_DENOM,
) -> dict[str, str]:
    """Map row ``id`` → ``train``|``test`` (stratified by domain).

    For each domain independently: sort ids, shuffle with
    ``Random(f\"{seed}:{domain}\")``, take the first ``len // test_denom``
    as **test**, remainder as **train**.

    With 32 768 rows/domain and ``test_denom=16`` → 2 048 test + 30 720 train
    per domain (4 096 / 61 440 overall on a full 65 536-row export).
    """
    import random
    from collections import defaultdict

    if test_denom < 2:
        raise ValueError("test_denom must be >= 2")
    by_domain: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        rid = str(row["id"])
        domain = str(row.get("domain", "unknown"))
        by_domain[domain].append(rid)

    id_to_split: dict[str, str] = {}
    for domain in sorted(by_domain.keys()):
        ids = sorted(set(by_domain[domain]))
        rng = random.Random(f"{seed}:{domain}")
        rng.shuffle(ids)
        n_test = len(ids) // test_denom
        for i, rid in enumerate(ids):
            id_to_split[rid] = "test" if i < n_test else "train"
    return id_to_split


def apply_official_splits(
    rows: list[dict[str, Any]],
    *,
    seed: int = OFFICIAL_SPLIT_SEED,
    test_denom: int = OFFICIAL_TEST_DENOM,
    force: bool = False,
) -> dict[str, int]:
    """Set ``row['split']`` in place. Returns counts ``{train, test}``.

    If every row already has a valid split and ``force`` is False, keeps them.
    """
    have_all = all(str(r.get("split", "")) in VALID_SPLITS for r in rows)
    if have_all and not force:
        counts = {"train": 0, "test": 0}
        for r in rows:
            counts[str(r["split"])] += 1
        return counts

    id_to_split = assign_official_splits(rows, seed=seed, test_denom=test_denom)
    counts = {"train": 0, "test": 0}
    for row in rows:
        sp = id_to_split[str(row["id"])]
        row["split"] = sp
        counts[sp] += 1
    return counts


def save_dataset_streaming(
    staged_rows: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    chunk_size: int = DEFAULT_WRITE_CHUNK,
    write_sidecar_jpegs: bool = True,
    jpeg_quality: int = 92,
    write_hf_arrow: bool = True,
    workers: int = DEFAULT_EXPORT_WORKERS,
    split_name: str = "train",
    clear_output: bool = True,
) -> Path:
    """Write dataset from staged rows without loading all images at once.

    Each staged row must include either:
      - ``image``: PIL.Image, or
      - ``image_path``: path to a JPEG/PNG on disk (preferred; low RAM)

    Row prep + sidecar copies run in a thread pool (``workers``, default 16).
    Staged JPEGs are embedded as raw bytes (no PIL re-encode). Peak memory is
    O(chunk_size) rows, not O(N).

    ``split_name`` prefixes parquet files (``train-*.parquet`` / ``test-*.parquet``).
    Set ``clear_output=False`` when appending a second split into the same dir.
    """
    import gc

    from datasets import Dataset

    output_dir = Path(output_dir)
    if clear_output and output_dir.exists():
        # Remove previous export contents carefully (keep parent / progress cache).
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
    n_workers = max(1, min(int(workers), 16))
    prefix = str(split_name or "train")

    domains: dict[str, int] = {"general": 0, "satellite": 0}
    sources: dict[str, int] = {}
    meta_path = output_dir / "metadata.jsonl"
    parquet_paths: list[Path] = []
    total = 0
    shard_i = 0

    if write_sidecar_jpegs:
        (images_root / DOMAIN_GENERAL).mkdir(parents=True, exist_ok=True)
        (images_root / DOMAIN_SATELLITE).mkdir(parents=True, exist_ok=True)

    def _flush(chunk: list[dict[str, Any]]) -> None:
        nonlocal shard_i, total
        if not chunk:
            return
        ds = Dataset.from_list(chunk, features=features_spec())
        out = data_dir / f"{prefix}-part-{shard_i:05d}.parquet"
        ds.to_parquet(str(out))
        parquet_paths.append(out)
        shard_i += 1
        total += len(chunk)
        for r in chunk:
            r["image"] = None
        del ds
        gc.collect()

    n_total = len(staged_rows)
    t0 = time.monotonic()
    print(
        f"  export[{prefix}]: writing {n_total:,} rows (chunk={chunk_size}, "
        f"sidecars={write_sidecar_jpegs}, workers={n_workers}) ...",
        flush=True,
    )

    def _prep(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        return _prepare_export_row(
            row,
            write_sidecar_jpegs=write_sidecar_jpegs,
            images_root=images_root,
            jpeg_quality=jpeg_quality,
        )

    meta_mode = "w" if clear_output else "a"
    with open(meta_path, meta_mode, encoding="utf-8") as meta_fh, ThreadPoolExecutor(
        max_workers=n_workers
    ) as pool:
        for start in range(0, n_total, chunk_size):
            batch_rows = staged_rows[start : start + chunk_size]
            # Parallel I/O: read staged JPEGs + write sidecars.
            prepared = list(pool.map(_prep, batch_rows))
            chunk: list[dict[str, Any]] = []
            for rec, meta in prepared:
                domains[rec["domain"]] = domains.get(rec["domain"], 0) + 1
                sources[rec["source"]] = sources.get(rec["source"], 0) + 1
                if meta is not None:
                    meta_fh.write(
                        json.dumps(meta, ensure_ascii=False) + "\n"
                    )
                chunk.append(rec)
            _flush(chunk)
            elapsed = max(time.monotonic() - t0, 1e-6)
            rate = total / elapsed
            eta = (n_total - total) / rate if rate > 0 else 0
            print(
                f"  wrote[{prefix}] {total:,}/{n_total:,} rows "
                f"({rate:.1f}/s, ETA {eta / 60:.1f} min, "
                f"shard {shard_i})",
                flush=True,
            )

    print(f"  parquet pass complete[{prefix}]: {total:,} rows", flush=True)

    # Rename parts to {split}-XXXXX-of-YYYYY.parquet
    n_shards = len(parquet_paths)
    final_paths: list[Path] = []
    for i, path in enumerate(parquet_paths):
        final = data_dir / f"{prefix}-{i:05d}-of-{n_shards:05d}.parquet"
        path.rename(final)
        final_paths.append(final)

    # Optional Arrow export for load_from_disk. Build from parquet (already
    # has embedded JPEG bytes) — do NOT re-decode sidecars row-by-row.
    hf_dir = output_dir / "hf"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)

    if write_hf_arrow and final_paths:
        print(
            f"  building hf/ from {n_shards} parquet shard(s) ...",
            flush=True,
        )
        t_hf = time.monotonic()
        from datasets import concatenate_datasets

        # Load shards (images stay as encoded bytes; no PIL re-encode).
        if n_shards == 1:
            ds = Dataset.from_parquet(str(final_paths[0]))
        else:
            # Parallel shard loads are I/O bound; cap like export workers.
            def _load_shard(p: Path):
                return Dataset.from_parquet(str(p))

            with ThreadPoolExecutor(max_workers=min(n_workers, n_shards)) as pool:
                shards = list(pool.map(_load_shard, final_paths))
            ds = concatenate_datasets(shards)
            del shards
        print("  saving hf/ to disk ...", flush=True)
        ds.save_to_disk(str(hf_dir))
        del ds
        gc.collect()
        print(f"  hf/ done in {time.monotonic() - t_hf:.1f}s", flush=True)
    elif write_hf_arrow:
        print("  warning: no parquet shards; skipping hf/", flush=True)

    _write_info_and_card(
        output_dir, num_rows=total, domains=domains, sources=sources
    )
    print(
        f"  export complete: {total:,} rows, {n_shards} parquet shard(s), "
        f"{time.monotonic() - t0:.1f}s total",
        flush=True,
    )
    return output_dir


def save_dataset(
    records: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    max_shard_size: str = "500MB",
    write_sidecar_jpegs: bool = True,
    chunk_size: int = DEFAULT_WRITE_CHUNK,
    write_hf_arrow: bool = True,
    workers: int = DEFAULT_EXPORT_WORKERS,
) -> Path:
    """Write HF dataset. Delegates to streaming saver (bounded RAM)."""
    del max_shard_size  # kept for call-site compatibility
    return save_dataset_streaming(
        records,
        output_dir,
        chunk_size=chunk_size,
        write_sidecar_jpegs=write_sidecar_jpegs,
        write_hf_arrow=write_hf_arrow,
        workers=workers,
    )


DEFAULT_IMAGE_CACHE_SIZE = 64


def _meta_fields_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Text fields shared by sidecar metadata and parquet/HF rows."""
    return {
        "id": str(row["id"]),
        "domain": str(row["domain"]),
        "source": str(row["source"]),
        "captions": list(row["captions"]),
        "query": str(row["query"]),
        "unrelated_query": str(row["unrelated_query"]),
    }


def _load_meta_lines(
    meta_path: Path,
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Parse ``metadata.jsonl`` without touching image files."""
    metas: list[dict[str, Any]] = []
    with open(meta_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            metas.append(json.loads(line))
    return metas


def _as_rgb_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    from trisearch_dataset import load_pil_image

    return load_pil_image(image).convert("RGB")


class LazyTriSearchDataset:
    """Random-access curated dataset: metadata in RAM, images on demand.

    Prefer ``metadata.jsonl`` + ``images/`` sidecars (fast path). Falls back to
    HF Arrow disk or parquet shards with random indexing. Recently viewed
    images are kept in a small LRU cache so paging does not re-decode every time.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        max_samples: int | None = None,
        image_cache_size: int = DEFAULT_IMAGE_CACHE_SIZE,
    ) -> None:
        root = Path(dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dataset not found at {root}")
        if image_cache_size < 0:
            raise ValueError(f"image_cache_size must be >= 0, got {image_cache_size}")

        self.root = root
        self._image_cache_size = image_cache_size
        self._image_cache: OrderedDict[int, Image.Image] = OrderedDict()
        self._backend: str
        self._metas: list[dict[str, Any]] = []
        self._hf_ds: Any = None

        meta_path = root / "metadata.jsonl"
        if meta_path.is_file() and (root / "images").is_dir():
            self._backend = "sidecar"
            self._metas = _load_meta_lines(meta_path, max_samples=max_samples)
            if not self._metas:
                raise ValueError(f"Empty metadata at {meta_path}")
            return

        from datasets import load_from_disk, load_dataset

        if (root / "hf").is_dir():
            self._backend = "hf"
            self._hf_ds = load_from_disk(str(root / "hf"))
        elif (root / "data").is_dir() and any((root / "data").glob("*.parquet")):
            self._backend = "parquet"
            self._hf_ds = load_dataset(
                "parquet",
                data_files=str(root / "data" / "*.parquet"),
                split="train",
            )
        else:
            raise FileNotFoundError(
                f"No TriSearch dataset at {root} "
                f"(expected metadata.jsonl+images/, hf/, or data/*.parquet)"
            )

        n = len(self._hf_ds)
        if max_samples is not None:
            n = min(n, max_samples)
        # Text columns only — never touch the image column during open.
        text_cols = [
            c
            for c in (
                "id",
                "domain",
                "source",
                "captions",
                "query",
                "unrelated_query",
            )
            if c in self._hf_ds.column_names
        ]
        meta_view = self._hf_ds.select(range(n)).select_columns(text_cols)
        self._metas = [_meta_fields_from_row(row) for row in meta_view]
        if not self._metas:
            raise ValueError(f"Empty dataset at {root}")

    def __len__(self) -> int:
        return len(self._metas)

    def meta(self, index: int) -> dict[str, Any]:
        """Return text fields only (no image I/O)."""
        if index < 0 or index >= len(self._metas):
            raise IndexError(index)
        m = self._metas[index]
        if self._backend == "sidecar":
            return _meta_fields_from_row(m)
        return dict(m)

    def _decode_image(self, index: int) -> Image.Image:
        if self._backend == "sidecar":
            m = self._metas[index]
            img_path = self.root / m["file_name"]
            # load() forces pixels into RAM then closes the file handle.
            with Image.open(img_path) as im:
                return im.convert("RGB")
        row = self._hf_ds[index]
        return _as_rgb_image(row["image"])

    def get_image(self, index: int) -> Image.Image:
        """Load (or return cached) RGB image for ``index``."""
        if index < 0 or index >= len(self._metas):
            raise IndexError(index)
        if index in self._image_cache:
            self._image_cache.move_to_end(index)
            return self._image_cache[index]

        image = self._decode_image(index)
        if self._image_cache_size > 0:
            self._image_cache[index] = image
            while len(self._image_cache) > self._image_cache_size:
                self._image_cache.popitem(last=False)
        return image

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Full record including image (lazy decode + LRU cache)."""
        rec = self.meta(index)
        rec["image"] = self.get_image(index)
        validate_record(rec, require_image=True)
        return rec

    def clear_image_cache(self) -> None:
        self._image_cache.clear()


def open_lazy_dataset(
    dataset_dir: str | Path,
    *,
    max_samples: int | None = None,
    image_cache_size: int = DEFAULT_IMAGE_CACHE_SIZE,
) -> LazyTriSearchDataset:
    """Open a curated dataset for random access without loading all images."""
    return LazyTriSearchDataset(
        dataset_dir,
        max_samples=max_samples,
        image_cache_size=image_cache_size,
    )


def load_dataset_records(
    dataset_dir: str | Path,
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load curated TriSearch dataset fully into memory (images decoded).

    Prefer :class:`LazyTriSearchDataset` / :func:`open_lazy_dataset` for
    browsing large exports. This helper materializes every row (used by
    tests and training loaders that need the full list).
    """
    lazy = open_lazy_dataset(
        dataset_dir,
        max_samples=max_samples,
        image_cache_size=0,  # no cache; we keep every image on the record list
    )
    return [lazy[i] for i in range(len(lazy))]


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
