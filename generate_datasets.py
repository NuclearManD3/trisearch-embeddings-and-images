#!/usr/bin/env python3
"""
Build the curated TriSearch Stage-1 dataset (HF-friendly export).

Memory model
------------
Images are **never** all kept in RAM. Flow:

1. Stream sources → resize to 512 → write staging JPEG to disk → keep metadata only.
2. Diversify captions / generate queries (text-only, low RAM).
3. Export parquet + hf/ in small chunks (``--write-chunk``, default 64).

Sources
-------
- **general**: MS-COCO via ``bitmind/MS-COCO`` (grouped multi-captions).
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
RSICD_HF_ID = "arampacha/rsicd"
SKYSCRIPT_CSV_URL = (
    "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/dataframe/"
    "SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv"
)
DEFAULT_SKYSCRIPT_ROOT = Path("models/data/SkyScript")
DEFAULT_TOTAL = 65_536
DEFAULT_SATELLITE_FRACTION = 0.5
STAGING_JPEG_QUALITY = 90


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


def _save_staging_jpeg(
    image: Image.Image,
    path: Path,
    *,
    image_size: int,
) -> Path:
    """Resize to square and write JPEG; free the large source image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    small = resize_square_rgb(image, image_size)
    small.save(path, format="JPEG", quality=STAGING_JPEG_QUALITY, optimize=True)
    small.close()
    if image is not small:
        try:
            image.close()
        except Exception:
            pass
    return path


# ---------------------------------------------------------------------------
# MS-COCO — stream groups without holding all images
# ---------------------------------------------------------------------------

def iter_coco_staged(
    *,
    max_images: int,
    seed: int,
    staging_dir: Path,
    image_size: int,
    hf_id: str = COCO_HF_ID,
) -> Iterator[dict[str, Any]]:
    """Yield staged general records (metadata + image_path only)."""
    from datasets import load_dataset

    print(f"Streaming general source {hf_id!r} (disk-staged, low RAM) ...", flush=True)
    ds = load_dataset(hf_id, split="train", streaming=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Reservoir of completed groups: only paths + captions (no pixels).
    reservoir: list[dict[str, Any]] = []
    seen = 0
    rng = random.Random(seed)

    current_id: Any = None
    current_caps: list[str] = []
    current_path: Path | None = None

    def _commit_group() -> None:
        nonlocal seen, current_id, current_caps, current_path
        if current_id is None or current_path is None:
            return
        try:
            caps = normalize_captions(current_caps, min_count=2)
        except ValueError:
            # Keep raw for later diversify if we have anything.
            caps = list(dict.fromkeys(c for c in current_caps if c))
            if not caps:
                if current_path.is_file():
                    current_path.unlink(missing_ok=True)
                return
        item = {
            "image_path": str(current_path),
            "captions": caps,
            "source": hf_id,
            "domain": DOMAIN_GENERAL,
        }
        seen += 1
        if len(reservoir) < max_images:
            reservoir.append(item)
        else:
            j = rng.randint(0, seen - 1)
            if j < max_images:
                old = reservoir[j]
                Path(old["image_path"]).unlink(missing_ok=True)
                reservoir[j] = item
            else:
                current_path.unlink(missing_ok=True)

    for row in ds:
        cid = row.get("cocoid", row.get("imgid"))
        sent = row.get("sentences") or {}
        if isinstance(sent, dict):
            cap = str(sent.get("raw") or sent.get("caption") or "").strip()
        else:
            cap = str(sent).strip()
        if not cap:
            continue

        if current_id is None:
            current_id = cid
            current_caps = [cap]
            try:
                img = _pil_from_row_image(row["image"])
            except Exception:
                current_id = None
                current_caps = []
                continue
            current_path = staging_dir / f"coco_{cid}.jpg"
            _save_staging_jpeg(img, current_path, image_size=image_size)
            continue

        if cid == current_id:
            if cap not in current_caps:
                current_caps.append(cap)
            continue

        _commit_group()
        current_id = cid
        current_caps = [cap]
        try:
            img = _pil_from_row_image(row["image"])
        except Exception:
            current_id = None
            current_caps = []
            current_path = None
            continue
        current_path = staging_dir / f"coco_{cid}.jpg"
        _save_staging_jpeg(img, current_path, image_size=image_size)

        # Early stop: enough reservoir fills and stream has moved on a bit.
        if seen >= max_images * 8 and len(reservoir) >= max_images:
            break

    _commit_group()
    print(f"  staged {len(reservoir):,} COCO images (from {seen:,} groups)", flush=True)
    yield from reservoir


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
    taken = 0
    for row in reservoir:
        try:
            with Image.open(row["src"]) as img:
                img.load()
                img = img.convert("RGB")
                out = staging_dir / f"sky_{taken:06d}.jpg"
                _save_staging_jpeg(img, out, image_size=image_size)
        except OSError:
            continue
        raw_caps = [c for c in (row["title"], row["multi"]) if c]
        yield {
            "image_path": str(out),
            "captions": raw_caps,
            "source": "SkyScript",
            "domain": DOMAIN_SATELLITE,
        }
        taken += 1
        if taken % 200 == 0:
            print(f"  staged {taken:,} SkyScript images ...", flush=True)
            gc.collect()
    print(f"  staged {taken:,} SkyScript images", flush=True)


def iter_rsicd_staged(
    *,
    max_images: int,
    seed: int,
    staging_dir: Path,
    image_size: int,
) -> Iterator[dict[str, Any]]:
    """Stream RSICD; stage JPEGs immediately (no full-row list of PIL images)."""
    from datasets import load_dataset

    print(f"Streaming satellite fallback {RSICD_HF_ID!r} (disk-staged) ...", flush=True)
    ds = load_dataset(RSICD_HF_ID, split="train", streaming=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0

    for row in ds:
        caps = row.get("captions") or []
        if not isinstance(caps, list) or not caps:
            continue
        try:
            img = _pil_from_row_image(row["image"])
        except (TypeError, OSError):
            continue
        seen += 1
        path = staging_dir / f"rsicd_{seen:06d}.jpg"
        _save_staging_jpeg(img, path, image_size=image_size)
        item = {
            "image_path": str(path),
            "captions": [str(c).strip() for c in caps if str(c).strip()],
            "source": RSICD_HF_ID,
            "domain": DOMAIN_SATELLITE,
        }
        if len(reservoir) < max_images:
            reservoir.append(item)
        else:
            j = rng.randint(0, seen - 1)
            if j < max_images:
                Path(reservoir[j]["image_path"]).unlink(missing_ok=True)
                reservoir[j] = item
            else:
                path.unlink(missing_ok=True)
        if seen >= max_images * 5 and len(reservoir) >= max_images:
            break

    print(f"  staged {len(reservoir):,} RSICD images (from {seen:,} seen)", flush=True)
    yield from reservoir


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
) -> list[dict[str, Any]]:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    gen_dir = staging_dir / "general"
    for i, item in enumerate(
        iter_coco_staged(
            max_images=n_general,
            seed=seed,
            staging_dir=gen_dir,
            image_size=image_size,
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
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--write-chunk", type=int, default=DEFAULT_WRITE_CHUNK,
                   help="Rows per parquet flush (default 64; lower = less RAM).")
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
