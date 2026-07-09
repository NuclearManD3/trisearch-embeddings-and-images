#!/usr/bin/env python3
"""
Build the curated TriSearch Stage-1 dataset (HF-friendly export).

Memory model
------------
Images are **never** all kept in RAM. Flow:

1. Stream sources → resize to 1024 → write staging JPEG to disk → keep metadata only.
2. Diversify captions / generate queries (text-only, low RAM).
3. Export parquet + hf/ in small chunks (``--write-chunk``, default 64).

Sources
-------
- **general**: MS-COCO via a **local on-disk mirror** of ``bitmind/MS-COCO``
  (downloaded once under ``models/data/bitmind-MS-COCO``).
- **satellite**: SkyScript (CSV + local image zips). Optional RSICD fallback.

Examples
--------
  python3 generate_datasets.py --preview --skip-query-generation --allow-rsicd-fallback
  python3 generate_datasets.py --total 65536 --download-skyscript-csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator
from urllib.request import urlopen

from PIL import Image

from trisearch_data_format import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_WRITE_CHUNK,
    DOMAIN_GENERAL,
    DOMAIN_SATELLITE,
    caption_set_is_diverse,
    captions_are_near_duplicate,
    normalize_captions,
    resize_square_rgb,
    save_dataset_streaming,
    validate_record,
)
from trisearch_dataset import (
    DEFAULT_OPENROUTER_CONFIG,
    load_openrouter_config,
    openrouter_diversify_captions,
    openrouter_generate_queries,
    openrouter_generate_queries_batch,
)

COCO_HF_ID = "bitmind/MS-COCO"
DEFAULT_COCO_LOCAL_DIR = Path("models/data/bitmind-MS-COCO")
RSICD_HF_ID = "arampacha/rsicd"
DEFAULT_RSICD_LOCAL_DIR = Path("models/data/arampacha-rsicd")
SKYSCRIPT_CSV_URL = (
    "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/dataframe/"
    "SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv"
)
DEFAULT_SKYSCRIPT_ROOT = Path("models/data/SkyScript")
DEFAULT_TOTAL = 65_536
DEFAULT_SATELLITE_FRACTION = 0.5
STAGING_JPEG_QUALITY = 90
DEFAULT_STAGE_WORKERS = 8


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _pil_from_row_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "bytes" in value:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(value)}")


def _staging_jpeg_ready(path: Path, *, image_size: int) -> bool:
    """True if ``path`` is a usable staged square JPEG (skip re-encode)."""
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        with Image.open(path) as im:
            return im.size == (image_size, image_size)
    except OSError:
        return False


def _save_staging_jpeg(
    image: Image.Image,
    path: Path,
    *,
    image_size: int,
) -> Path:
    """Resize to square and write JPEG; free the large source image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _staging_jpeg_ready(path, image_size=image_size):
        try:
            image.close()
        except Exception:
            pass
        return path
    small = resize_square_rgb(image, image_size)
    small.save(path, format="JPEG", quality=STAGING_JPEG_QUALITY, optimize=True)
    small.close()
    if image is not small:
        try:
            image.close()
        except Exception:
            pass
    return path


def _stage_job_from_path(
    src_path: str,
    dest_path: str,
    image_size: int,
) -> str:
    """Worker: open disk image → stage JPEG (skip if dest already good)."""
    dest = Path(dest_path)
    if _staging_jpeg_ready(dest, image_size=image_size):
        return dest_path
    with Image.open(src_path) as img:
        img.load()
        rgb = img.convert("RGB")
        _save_staging_jpeg(rgb, dest, image_size=image_size)
    return dest_path


def _parallel_stage_path_jobs(
    jobs: list[tuple[str, str, list[str], str, str]],
    *,
    image_size: int,
    workers: int,
) -> list[dict[str, Any]]:
    """Stage many (src_path, dest_path, captions, source, domain) jobs in parallel.

    Each job tuple: (src_path, dest_path, captions, source, domain).
    Returns metadata dicts with ``image_path`` set (no PIL retained).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not jobs:
        return []
    workers = max(1, min(workers, len(jobs)))
    print(
        f"  parallel image staging: {len(jobs):,} jobs, workers={workers}",
        flush=True,
    )
    results: list[dict[str, Any] | None] = [None] * len(jobs)

    def work(i: int) -> tuple[int, dict[str, Any] | None]:
        src, dest, caps, source, domain = jobs[i]
        try:
            _stage_job_from_path(src, dest, image_size)
            return i, {
                "image_path": dest,
                "captions": caps,
                "source": source,
                "domain": domain,
            }
        except OSError as exc:
            print(f"  skip staging {src}: {exc}", flush=True)
            return i, None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(work, i) for i in range(len(jobs))]
        for fut in as_completed(futs):
            i, item = fut.result()
            results[i] = item
            done += 1
            if done == 1 or done % 50 == 0 or done == len(jobs):
                print(f"    staged images {done}/{len(jobs)}", flush=True)

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# MS-COCO — local on-disk mirror (download once)
# ---------------------------------------------------------------------------

def ensure_local_hf_dataset(
    local_dir: Path,
    hf_id: str,
    *,
    split: str = "train",
) -> Any:
    """Load a HF dataset from a local ``save_to_disk`` mirror; download once if missing.

    This is the “local COCO mirror”: Hub is contacted only when ``local_dir`` is
    absent. Later runs read only from disk (no re-download / re-stream).
    """
    from datasets import load_dataset, load_from_disk

    local_dir = Path(local_dir)
    # datasets.save_to_disk writes state.json (Dataset) or dataset_dict.json
    if (local_dir / "state.json").is_file() or (local_dir / "dataset_dict.json").is_file():
        print(f"Loading local dataset mirror from {local_dir} ...", flush=True)
        from datasets import DatasetDict

        ds = load_from_disk(str(local_dir))
        if isinstance(ds, DatasetDict):
            return ds[split]
        return ds

    print(
        f"Local mirror missing at {local_dir}; downloading {hf_id!r} once "
        f"(split={split}) and saving to disk ...",
        flush=True,
    )
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(hf_id, split=split)  # materializes into HF cache, then we mirror
    ds.save_to_disk(str(local_dir))
    print(f"  saved local mirror -> {local_dir}", flush=True)
    return ds


def iter_coco_staged(
    *,
    max_images: int,
    seed: int,
    staging_dir: Path,
    image_size: int,
    workers: int = DEFAULT_STAGE_WORKERS,
    hf_id: str = COCO_HF_ID,
    local_dir: Path = DEFAULT_COCO_LOCAL_DIR,
) -> Iterator[dict[str, Any]]:
    """Stage general COCO images from a local mirror (no Hub re-stream).

    Pass 1: text-only column scan to group multi-captions (no image decode).
    Pass 2: shuffle + take exactly ``max_images`` groups (no overscan).
    Pass 3: parallel resize/JPEG for groups not already staged.
    """
    ds = ensure_local_hf_dataset(local_dir, hf_id, split="train")
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Text-only pass: avoid decoding images while grouping captions.
    text_cols = [c for c in ds.column_names if c != "image"]
    text_ds = ds.select_columns(text_cols) if "image" in ds.column_names else ds

    print(
        f"Grouping COCO multi-captions from local mirror "
        f"({len(text_ds):,} rows, text-only) ...",
        flush=True,
    )
    groups: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    for i in range(len(text_ds)):
        row = text_ds[i]
        cid = row.get("cocoid", row.get("imgid"))
        sent = row.get("sentences") or {}
        if isinstance(sent, dict):
            cap = str(sent.get("raw") or sent.get("caption") or "").strip()
        else:
            cap = str(sent).strip()
        if not cap:
            continue
        if cid not in groups:
            groups[cid] = {"indices": [i], "captions": [cap]}
            order.append(cid)
        else:
            groups[cid]["indices"].append(i)
            if cap not in groups[cid]["captions"]:
                groups[cid]["captions"].append(cap)

    # Need ≥2 caption strings; near-dup diversity may be fixed later by diversify step.
    eligible = [cid for cid in order if len(groups[cid]["captions"]) >= 2]

    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = eligible[:max_images]
    print(
        f"  COCO groups with multi-captions: {len(eligible):,}; "
        f"selected {len(selected):,} (no overscan)",
        flush=True,
    )

    # Parallel stage: load image only for selected groups missing staging files.
    jobs: list[tuple[int, str, list[str]]] = []  # (ds_index, dest, captions)
    ready: list[dict[str, Any]] = []
    for cid in selected:
        g = groups[cid]
        dest = staging_dir / f"coco_{cid}.jpg"
        caps = g["captions"]
        try:
            caps = normalize_captions(caps, min_count=2)
        except ValueError:
            caps = list(dict.fromkeys(c for c in caps if c.strip()))
        if _staging_jpeg_ready(dest, image_size=image_size):
            ready.append({
                "image_path": str(dest),
                "captions": caps,
                "source": hf_id,
                "domain": DOMAIN_GENERAL,
            })
        else:
            jobs.append((g["indices"][0], str(dest), caps))

    if jobs:
        print(
            f"  COCO staging: {len(ready):,} cached, {len(jobs):,} to encode "
            f"(workers={workers})",
            flush=True,
        )
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def work(job: tuple[int, str, list[str]]) -> dict[str, Any] | None:
            idx, dest, caps = job
            try:
                img = _pil_from_row_image(ds[idx]["image"])
                _save_staging_jpeg(img, Path(dest), image_size=image_size)
                return {
                    "image_path": dest,
                    "captions": caps,
                    "source": hf_id,
                    "domain": DOMAIN_GENERAL,
                }
            except Exception as exc:
                print(f"  skip COCO idx={idx}: {exc}", flush=True)
                return None

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = [pool.submit(work, j) for j in jobs]
            for fut in as_completed(futs):
                item = fut.result()
                done += 1
                if item is not None:
                    ready.append(item)
                if done == 1 or done % 50 == 0 or done == len(jobs):
                    print(f"    COCO encode {done}/{len(jobs)}", flush=True)
    else:
        print(f"  COCO staging: all {len(ready):,} already cached", flush=True)

    print(f"  staged {len(ready):,} COCO images", flush=True)
    yield from ready


# ---------------------------------------------------------------------------
# SkyScript
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, *, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        print(f"  using cached {dest}")
        return dest
    print(f"  downloading {url} -> {dest} ...")
    with urlopen(url, timeout=600) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return dest


def ensure_skyscript_csv(root: Path, *, download: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / Path(SKYSCRIPT_CSV_URL).name
    if path.is_file():
        return path
    if not download:
        raise FileNotFoundError(
            f"SkyScript CSV missing at {path}. "
            f"Pass --download-skyscript-csv or place the file manually."
        )
    return download_file(SKYSCRIPT_CSV_URL, path)


def _resolve_skyscript_file(root: Path, filepath: str) -> Path | None:
    """Resolve one CSV filepath without building a full directory index."""
    rel = filepath.lstrip("./")
    candidates = [root / rel, root / Path(rel).name]
    # Common layout: images2/foo.jpg extracted as root/images2/foo.jpg or root/foo.jpg
    name = Path(rel).name
    for sub in ("images2", "images3", "images4", "images5", "images6", "images7"):
        candidates.append(root / sub / name)
        candidates.append(root / sub / rel)
    for c in candidates:
        if c.is_file():
            return c
    return None


def iter_skyscript_staged(
    *,
    root: Path,
    max_images: int,
    seed: int,
    download_csv: bool,
    staging_dir: Path,
    image_size: int,
    workers: int = DEFAULT_STAGE_WORKERS,
) -> Iterator[dict[str, Any]]:
    csv_path = ensure_skyscript_csv(root, download=download_csv)
    if not root.is_dir():
        raise FileNotFoundError(
            f"No SkyScript root at {root}. Download and extract "
            f"images2.zip … images7.zip from SkyScript S3 into that folder.\n"
            f"Pass --allow-rsicd-fallback to use RSICD instead for satellite."
        )

    # Reservoir-sample CSV rows that resolve to an existing file — never load
    # all paths or all pixels into RAM.
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    seen = 0
    missing = 0
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fp = (row.get("filepath") or "").strip()
            if not fp:
                continue
            src = _resolve_skyscript_file(root, fp)
            if src is None:
                missing += 1
                continue
            title = (row.get("title") or "").strip()
            multi = (row.get("title_multi_objects") or "").strip()
            if not title and not multi:
                continue
            seen += 1
            item = {"src": str(src), "title": title, "multi": multi}
            if len(reservoir) < max_images:
                reservoir.append(item)
            else:
                j = rng.randint(0, seen - 1)
                if j < max_images:
                    reservoir[j] = item
            # Plenty of hits — stop scanning the multi-million-row CSV early.
            if seen >= max_images * 20 and len(reservoir) >= max_images:
                break

    if not reservoir:
        raise FileNotFoundError(
            f"No SkyScript images under {root} matched the CSV "
            f"(missing path hits={missing:,}). Extract image zips or use "
            f"--allow-rsicd-fallback."
        )

    print(
        f"  SkyScript: reservoir {len(reservoir):,} paths "
        f"(scanned matches={seen:,}, unresolved={missing:,})",
        flush=True,
    )
    staging_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str, list[str], str, str]] = []
    for i, row in enumerate(reservoir):
        dest = str(staging_dir / f"sky_{i:06d}.jpg")
        caps = [c for c in (row["title"], row["multi"]) if c]
        jobs.append((row["src"], dest, caps, "SkyScript", DOMAIN_SATELLITE))
    # Use module-level default workers via parallel helper; caller can pass later.
    staged = _parallel_stage_path_jobs(jobs, image_size=image_size, workers=workers)
    print(f"  staged {len(staged):,} SkyScript images", flush=True)
    yield from staged


def iter_rsicd_staged(
    *,
    max_images: int,
    seed: int,
    staging_dir: Path,
    image_size: int,
    workers: int = DEFAULT_STAGE_WORKERS,
    local_dir: Path = DEFAULT_RSICD_LOCAL_DIR,
) -> Iterator[dict[str, Any]]:
    """Stage RSICD from a local on-disk mirror (download once if missing)."""
    ds = ensure_local_hf_dataset(local_dir, RSICD_HF_ID, split="train")
    staging_dir.mkdir(parents=True, exist_ok=True)
    n = len(ds)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:max_images]

    print(
        f"RSICD local mirror: selecting {len(indices):,}/{n:,} images "
        f"(workers={workers})",
        flush=True,
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def work(rank_i: int, ds_idx: int) -> dict[str, Any] | None:
        dest = staging_dir / f"rsicd_{rank_i:06d}.jpg"
        try:
            row = ds[ds_idx]
            caps = row.get("captions") or []
            if not isinstance(caps, list) or not caps:
                return None
            caps = [str(c).strip() for c in caps if str(c).strip()]
            if _staging_jpeg_ready(dest, image_size=image_size):
                return {
                    "image_path": str(dest),
                    "captions": caps,
                    "source": RSICD_HF_ID,
                    "domain": DOMAIN_SATELLITE,
                }
            img = _pil_from_row_image(row["image"])
            _save_staging_jpeg(img, dest, image_size=image_size)
            return {
                "image_path": str(dest),
                "captions": caps,
                "source": RSICD_HF_ID,
                "domain": DOMAIN_SATELLITE,
            }
        except Exception as exc:
            print(f"  skip RSICD idx={ds_idx}: {exc}", flush=True)
            return None

    ready: list[dict[str, Any]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(work, rank, idx): rank
            for rank, idx in enumerate(indices)
        }
        for fut in as_completed(futs):
            item = fut.result()
            done += 1
            if item is not None:
                ready.append(item)
            if done == 1 or done % 50 == 0 or done == len(indices):
                print(f"    RSICD stage {done}/{len(indices)}", flush=True)

    print(f"  staged {len(ready):,} RSICD images", flush=True)
    yield from ready


# ---------------------------------------------------------------------------
# Caption diversity + queries (text only — safe to hold full list)
# ---------------------------------------------------------------------------

def _soft_unique_captions(captions: list[str]) -> list[str]:
    kept: list[str] = []
    for cap in captions:
        text = str(cap).strip()
        if not text:
            continue
        if any(captions_are_near_duplicate(text, k) for k in kept):
            continue
        kept.append(text)
    return kept


def _offline_diverse_captions(primary: str, domain: str) -> list[str]:
    """Offline fallback when API diversify is disabled.

    Secondary lines intentionally use different vocabulary skeletons so they
    pass near-duplicate checks (unlike 'in airport' vs 'at airport').
    Prefer ``openrouter_diversify_captions`` for real quality.
    """
    p = primary.strip().rstrip(".")
    if domain == DOMAIN_SATELLITE:
        return [
            p + ".",
            "Bird's-eye remote-sensing frame with terrain texture and man-made structures visible.",
            f"Geospatial overhead context; main subject of interest: {p[:80]}.",
        ]
    return [
        p + ".",
        "Ground-level photograph with clear foreground subject and background setting.",
        f"Natural-image scene summary focusing on layout and objects: {p[:80]}.",
    ]


def diversify_record_captions(
    items: list[dict[str, Any]],
    *,
    skip_api: bool,
    config_path: str | Path,
    parallelism: int = 32,
    min_count: int = 2,
) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    need_idx: list[int] = []
    for i, item in enumerate(items):
        soft = _soft_unique_captions(list(item.get("captions") or []))
        if caption_set_is_diverse(soft, min_count=min_count):
            item["captions"] = soft
        else:
            need_idx.append(i)

    if not need_idx:
        print("  caption diversity: all items already diverse", flush=True)
        return

    print(
        f"  caption diversity: {len(need_idx):,}/{len(items):,} items need rewrites",
        flush=True,
    )

    if skip_api:
        for i in need_idx:
            primary = (items[i].get("captions") or ["scene"])[0]
            raw = _offline_diverse_captions(primary, items[i]["domain"])
            try:
                items[i]["captions"] = normalize_captions(raw, min_count=min_count)
            except ValueError:
                # Guaranteed pair if filters still fight us.
                items[i]["captions"] = [primary.strip() or "scene", raw[1]]
        print(
            f"  caption diversity: offline rewrites on {len(need_idx):,} items",
            flush=True,
        )
        return

    config = load_openrouter_config(config_path)
    api_key, model = config["api_key"], config["model"]
    workers = min(max(1, parallelism), len(need_idx))

    def work(i: int) -> tuple[int, list[str]]:
        caps = openrouter_diversify_captions(
            list(items[i].get("captions") or []),
            api_key=api_key,
            model=model,
            domain=str(items[i].get("domain", DOMAIN_GENERAL)),
            min_count=min_count,
        )
        return i, caps

    done = 0
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, i) for i in need_idx]
        for fut in as_completed(futures):
            i, caps = fut.result()
            items[i]["captions"] = caps
            done += 1
            if done == 1 or done % 10 == 0 or done == len(futures):
                elapsed = max(time.monotonic() - start, 1e-6)
                print(
                    f"    diversified {done}/{len(futures)} ({done / elapsed:.1f}/s)",
                    flush=True,
                )


def attach_queries(
    items: list[dict[str, Any]],
    *,
    skip_generation: bool,
    config_path: str | Path,
    batch_size: int = 8,
    parallelism: int = 32,
) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if skip_generation:
        for item in items:
            caps = item["captions"]
            item["query"] = caps[1] if len(caps) > 1 else caps[0]
            item["unrelated_query"] = "red sports car on a racetrack at night"
        print(f"  skip-query-generation: filled {len(items):,} queries offline", flush=True)
        return

    if batch_size < 1 or parallelism < 1:
        raise ValueError("batch_size and parallelism must be >= 1")

    config = load_openrouter_config(config_path)
    api_key, model = config["api_key"], config["model"]
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    workers = min(parallelism, len(batches))
    print(
        f"  generating queries via OpenRouter ({model}) for {len(items):,} items "
        f"({len(batches)} API calls, batch_size={batch_size}, "
        f"parallel_workers={workers}) ...",
        flush=True,
    )

    def work(batch: list[dict[str, Any]]) -> int:
        captions = [b["captions"][0] for b in batch]
        if len(captions) == 1:
            results = [openrouter_generate_queries(
                captions[0], api_key=api_key, model=model
            )]
        else:
            results = openrouter_generate_queries_batch(
                captions, api_key=api_key, model=model
            )
        for item, res in zip(batch, results):
            item["query"] = res["related_query"]
            item["unrelated_query"] = res["unrelated_query"]
        return len(batch)

    done_batches = 0
    done_items = 0
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, b) for b in batches]
        for fut in as_completed(futures):
            n = fut.result()
            done_batches += 1
            done_items += n
            if (
                done_batches == 1
                or done_batches == len(futures)
                or done_batches % 2 == 0
            ):
                elapsed = max(time.monotonic() - start, 1e-6)
                print(
                    f"    query batches {done_batches}/{len(futures)} "
                    f"({done_items:,}/{len(items):,} items, "
                    f"{done_items / elapsed:.1f} items/s)",
                    flush=True,
                )


# ---------------------------------------------------------------------------
# Build staged metadata (images on disk)
# ---------------------------------------------------------------------------

def build_staged_records(
    *,
    n_general: int,
    n_satellite: int,
    seed: int,
    skyscript_root: Path,
    download_skyscript_csv: bool,
    allow_rsicd_fallback: bool,
    image_size: int,
    staging_dir: Path,
    stage_workers: int = DEFAULT_STAGE_WORKERS,
    coco_local_dir: Path = DEFAULT_COCO_LOCAL_DIR,
) -> list[dict[str, Any]]:
    # Reuse staging dir so existing JPEGs are not re-encoded.
    staging_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    gen_dir = staging_dir / "general"
    for i, item in enumerate(
        iter_coco_staged(
            max_images=n_general,
            seed=seed,
            staging_dir=gen_dir,
            image_size=image_size,
            workers=stage_workers,
            local_dir=coco_local_dir,
        )
    ):
        item = dict(item)
        item["id"] = f"general-{i:06d}"
        item["query"] = ""
        item["unrelated_query"] = ""
        records.append(item)
        gc.collect()

    sat_dir = staging_dir / "satellite"
    try:
        sat_iter = iter_skyscript_staged(
            root=skyscript_root,
            max_images=n_satellite,
            seed=seed + 1,
            download_csv=download_skyscript_csv,
            staging_dir=sat_dir,
            image_size=image_size,
            workers=stage_workers,
        )
        sat_items = list(sat_iter)
    except FileNotFoundError as exc:
        if not allow_rsicd_fallback:
            raise
        print(f"SkyScript unavailable ({exc}); falling back to RSICD", flush=True)
        sat_items = list(
            iter_rsicd_staged(
                max_images=n_satellite,
                seed=seed + 1,
                staging_dir=sat_dir,
                image_size=image_size,
                workers=stage_workers,
            )
        )

    for i, item in enumerate(sat_items):
        if i >= n_satellite:
            # drop extra staged files
            Path(item["image_path"]).unlink(missing_ok=True)
            continue
        item = dict(item)
        item["id"] = f"satellite-{i:06d}"
        item["query"] = ""
        item["unrelated_query"] = ""
        records.append(item)

    del sat_items
    gc.collect()

    rng = random.Random(seed)
    rng.shuffle(records)
    print(
        f"Staged {len(records):,} records on disk under {staging_dir} "
        f"(general={sum(1 for r in records if r['domain']==DOMAIN_GENERAL):,}, "
        f"satellite={sum(1 for r in records if r['domain']==DOMAIN_SATELLITE):,})",
        flush=True,
    )
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--total", type=int, default=DEFAULT_TOTAL,
                   help=f"Target dataset size (default {DEFAULT_TOTAL}).")
    p.add_argument("--preview", action="store_true",
                   help="Build only 100 items for quality checking.")
    p.add_argument("--preview-count", type=int, default=100)
    p.add_argument("--satellite-fraction", type=float,
                   default=DEFAULT_SATELLITE_FRACTION)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE,
                   help=f"Output square size (default {DEFAULT_IMAGE_SIZE}).")
    p.add_argument("--write-chunk", type=int, default=DEFAULT_WRITE_CHUNK,
                   help="Rows per parquet flush (default 64; lower = less RAM).")
    p.add_argument("--stage-workers", type=int, default=DEFAULT_STAGE_WORKERS,
                   help="Parallel workers for image resize/JPEG staging.")
    p.add_argument(
        "--coco-local-dir",
        type=Path,
        default=DEFAULT_COCO_LOCAL_DIR,
        help="On-disk MS-COCO mirror (download once, reuse forever).",
    )
    p.add_argument("--skyscript-root", type=Path, default=DEFAULT_SKYSCRIPT_ROOT)
    p.add_argument("--download-skyscript-csv", action="store_true")
    p.add_argument("--allow-rsicd-fallback", action="store_true",
                   help="If SkyScript images are missing, use arampacha/rsicd.")
    p.add_argument("--skip-query-generation", action="store_true",
                   help="Do not call OpenRouter; use alternate captions as queries.")
    p.add_argument(
        "--force-caption-diversify",
        action="store_true",
        help="Call OpenRouter to rewrite near-duplicate captions even with "
             "--skip-query-generation.",
    )
    p.add_argument("--openrouter-config", default=str(DEFAULT_OPENROUTER_CONFIG))
    p.add_argument("--query-batch-size", type=int, default=8)
    p.add_argument("--query-parallelism", type=int, default=32)
    p.add_argument("--no-sidecar-jpegs", action="store_true")
    p.add_argument(
        "--no-hf-arrow",
        action="store_true",
        help="Skip datasets.save_to_disk(hf/); keep parquet + metadata.jsonl only "
             "(lower peak RAM / disk; loaders still work).",
    )
    p.add_argument(
        "--keep-staging",
        action="store_true",
        help="Do not delete the temporary staging JPEG directory after export.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    total = args.preview_count if args.preview else args.total
    if total < 2:
        print("--total/--preview-count must be >= 2", file=sys.stderr)
        return 2
    n_sat = int(round(total * args.satellite_fraction))
    n_gen = total - n_sat
    staging_dir = args.output_dir / ".staging"
    print(
        f"Generating TriSearch dataset: total={total} "
        f"(general={n_gen}, satellite={n_sat}) -> {args.output_dir}\n"
        f"  staging={staging_dir} write_chunk={args.write_chunk}",
        flush=True,
    )

    records = build_staged_records(
        n_general=n_gen,
        n_satellite=n_sat,
        seed=args.seed,
        skyscript_root=args.skyscript_root,
        download_skyscript_csv=args.download_skyscript_csv,
        allow_rsicd_fallback=args.allow_rsicd_fallback or args.preview,
        image_size=args.image_size,
        staging_dir=staging_dir,
        stage_workers=args.stage_workers,
        coco_local_dir=args.coco_local_dir,
    )
    if not records:
        print("No records produced.", file=sys.stderr)
        return 1
    if len(records) < total:
        print(
            f"warning: only produced {len(records)}/{total} records",
            file=sys.stderr,
        )

    diversify_record_captions(
        records,
        skip_api=args.skip_query_generation and not args.force_caption_diversify,
        config_path=args.openrouter_config,
        parallelism=args.query_parallelism,
    )
    attach_queries(
        records,
        skip_generation=args.skip_query_generation,
        config_path=args.openrouter_config,
        batch_size=args.query_batch_size,
        parallelism=args.query_parallelism,
    )

    good: list[dict[str, Any]] = []
    for rec in records:
        try:
            # Validate text fields; image checked at export from path.
            if not Path(rec["image_path"]).is_file():
                raise FileNotFoundError(rec["image_path"])
            normalize_captions(rec["captions"], min_count=2)
            if not str(rec.get("query", "")).strip():
                raise ValueError("empty query")
            if not str(rec.get("unrelated_query", "")).strip():
                raise ValueError("empty unrelated_query")
            good.append(rec)
        except (ValueError, TypeError, FileNotFoundError) as exc:
            print(f"  dropping {rec.get('id')}: {exc}", file=sys.stderr)
            Path(rec.get("image_path", "")).unlink(missing_ok=True)
    records = good
    if not records:
        print("No valid records after caption diversify.", file=sys.stderr)
        return 1

    out = save_dataset_streaming(
        records,
        args.output_dir,
        chunk_size=args.write_chunk,
        write_sidecar_jpegs=not args.no_sidecar_jpegs,
        write_hf_arrow=not args.no_hf_arrow,
    )

    if not args.keep_staging and staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"  removed staging dir {staging_dir}", flush=True)

    print(f"Saved dataset to {out}")
    print(f"  hf load: datasets.load_from_disk({out / 'hf'!r})")
    print(f"  viewer:  python3 view_dataset.py --dataset-dir {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
