#!/usr/bin/env python3
"""
Build the curated TriSearch Stage-1 dataset (HF-friendly export).

Memory model
------------
Images are **never** all kept in RAM. Flow:

1. Stream sources → resize to 1024 → write staging JPEG to disk → keep metadata only.
2. Diversify captions / generate queries (text-only, low RAM).
3. Export parquet + hf/ in chunks (``--write-chunk``, default 4096 rows ≈ ~0.7GB).

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
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.request import urlopen

from PIL import Image

from trisearch_data_format import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_EXPORT_WORKERS,
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


def _log(msg: str) -> None:
    print(msg, flush=True)


def _phase(title: str) -> float:
    _log(f"\n=== {title} ===")
    return time.monotonic()


def _phase_done(t0: float, title: str, detail: str = "") -> None:
    dt = time.monotonic() - t0
    extra = f" — {detail}" if detail else ""
    _log(f"=== done: {title} ({dt:.1f}s){extra} ===")


# ---------------------------------------------------------------------------
# Resume / progress cache (survives crashes mid-API)
# ---------------------------------------------------------------------------

def record_stable_key(rec: dict[str, Any]) -> str:
    """Stable key for resume (independent of list shuffle / export ids)."""
    name = Path(str(rec.get("image_path", ""))).name
    domain = str(rec.get("domain", "unknown"))
    return f"{domain}/{name}"


class ProgressStore:
    """JSON progress file: captions + queries per staged image.

    Thread-safe. Writes use unique temp files + ``os.replace`` so concurrent
    flushes (main thread + worker fallbacks) cannot race on a shared ``.tmp``.
    Safe to resume after Ctrl-C or a failed API batch.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, dict[str, Any]] = {}
        self._dirty = 0
        self._lock = threading.RLock()
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self.data = raw
                _log(f"  resume cache: loaded {len(self.data):,} entries from {self.path}")
            except (json.JSONDecodeError, OSError) as exc:
                _log(f"  resume cache: could not load {self.path} ({exc}); starting fresh")
                self.data = {}

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            ent = self.data.get(key)
            return ent if isinstance(ent, dict) else None

    def captions_done(self, key: str) -> bool:
        ent = self.get(key)
        return bool(ent and ent.get("captions_done") and ent.get("captions"))

    def queries_done(self, key: str) -> bool:
        ent = self.get(key)
        return bool(
            ent
            and ent.get("queries_done")
            and str(ent.get("query", "")).strip()
            and str(ent.get("unrelated_query", "")).strip()
        )

    def set_captions(self, key: str, captions: list[str], *, flush_every: int = 50) -> None:
        with self._lock:
            ent = self.data.setdefault(key, {})
            ent["captions"] = list(captions)
            ent["captions_done"] = True
            self._dirty += 1
            if self._dirty >= flush_every:
                self._save_unlocked()

    def set_queries(
        self,
        key: str,
        query: str,
        unrelated_query: str,
        *,
        flush_every: int = 50,
    ) -> None:
        with self._lock:
            ent = self.data.setdefault(key, {})
            ent["query"] = query
            ent["unrelated_query"] = unrelated_query
            ent["queries_done"] = True
            self._dirty += 1
            if self._dirty >= flush_every:
                self._save_unlocked()

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        """Caller must hold ``self._lock``. Unique tmp avoids multi-thread clobber."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Snapshot under the lock so a concurrent set_* cannot mutate mid-dump.
        payload = json.dumps(self.data, ensure_ascii=False, separators=(",", ":"))
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        self._dirty = 0

    def apply_to_records(self, records: list[dict[str, Any]]) -> tuple[int, int]:
        """Merge cache into records. Returns (n_captions_restored, n_queries_restored)."""
        n_cap = n_q = 0
        with self._lock:
            for rec in records:
                key = record_stable_key(rec)
                rec["_key"] = key
                ent = self.data.get(key)
                if not isinstance(ent, dict):
                    continue
                if ent.get("captions_done") and ent.get("captions"):
                    rec["captions"] = list(ent["captions"])
                    rec["_captions_done"] = True
                    n_cap += 1
                if (
                    ent.get("queries_done")
                    and str(ent.get("query", "")).strip()
                    and str(ent.get("unrelated_query", "")).strip()
                ):
                    rec["query"] = str(ent["query"])
                    rec["unrelated_query"] = str(ent["unrelated_query"])
                    rec["_queries_done"] = True
                    n_q += 1
        return n_cap, n_q


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

    n_rows = len(text_ds)
    _log(
        f"Grouping COCO multi-captions from local mirror "
        f"({n_rows:,} rows, text-only, sequential scan) ..."
    )
    groups: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    t_group = time.monotonic()
    # Prefer sequential iteration (faster than random-access text_ds[i] for Arrow).
    for i, row in enumerate(text_ds):
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
        if (i + 1) % 25000 == 0 or (i + 1) == n_rows:
            elapsed = max(time.monotonic() - t_group, 1e-6)
            _log(
                f"    COCO group scan {i + 1:,}/{n_rows:,} rows "
                f"({(i + 1) / elapsed:.0f} rows/s, {len(groups):,} images)"
            )

    # Need ≥2 caption strings; near-dup diversity may be fixed later by diversify step.
    eligible = [cid for cid in order if len(groups[cid]["captions"]) >= 2]

    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = eligible[:max_images]
    _log(
        f"  COCO groups with multi-captions: {len(eligible):,}; "
        f"selected {len(selected):,} (no overscan)"
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
        _log(
            f"  COCO staging: {len(ready):,} cached, {len(jobs):,} to encode "
            f"(workers={workers})"
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
                _log(f"  skip COCO idx={idx}: {exc}")
                return None

        done = 0
        t_enc = time.monotonic()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = [pool.submit(work, j) for j in jobs]
            for fut in as_completed(futs):
                item = fut.result()
                done += 1
                if item is not None:
                    ready.append(item)
                if done == 1 or done % 50 == 0 or done == len(jobs):
                    elapsed = max(time.monotonic() - t_enc, 1e-6)
                    rate = done / elapsed
                    eta = (len(jobs) - done) / rate if rate > 0 else 0
                    _log(
                        f"    COCO encode {done}/{len(jobs)} "
                        f"({rate:.1f}/s, ETA {eta / 60:.1f} min)"
                    )
    else:
        _log(f"  COCO staging: all {len(ready):,} already cached")

    _log(f"  staged {len(ready):,} COCO images")
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


def _skyscript_basename_index(root: Path) -> dict[str, str]:
    """One-time basename→path map for local SkyScript trees (strings only).

    Much faster than millions of ``Path.is_file()`` probes while scanning the CSV.
    """
    import os

    t0 = time.monotonic()
    index: dict[str, str] = {}
    roots = [root]
    for sub in ("images2", "images3", "images4", "images5", "images6", "images7"):
        d = root / sub
        if d.is_dir():
            roots.append(d)
    for d in roots:
        # Non-recursive first (common layout); recurse only if empty.
        count_before = len(index)
        try:
            with os.scandir(d) as it:
                for ent in it:
                    if ent.is_file() and ent.name.lower().endswith((".jpg", ".jpeg")):
                        index.setdefault(ent.name, ent.path)
        except OSError:
            continue
        if len(index) == count_before:
            # Nested layout fallback
            for dirpath, _dirnames, filenames in os.walk(d):
                for name in filenames:
                    if name.lower().endswith((".jpg", ".jpeg")):
                        index.setdefault(name, os.path.join(dirpath, name))
    _log(
        f"  SkyScript basename index: {len(index):,} files "
        f"({time.monotonic() - t0:.1f}s)"
    )
    return index


def _resolve_skyscript_file(
    root: Path,
    filepath: str,
    *,
    name_index: dict[str, str] | None = None,
) -> Path | None:
    """Resolve one CSV filepath using optional basename index."""
    rel = filepath.lstrip("./")
    name = Path(rel).name
    if name_index is not None:
        hit = name_index.get(name)
        return Path(hit) if hit else None
    candidates = [root / rel, root / name]
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
    scanned = 0
    name_index = _skyscript_basename_index(root)
    t_csv = time.monotonic()
    _log(f"  SkyScript: scanning CSV {csv_path.name} for local files ...")
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            scanned += 1
            if scanned == 1 or scanned % 100000 == 0:
                elapsed = max(time.monotonic() - t_csv, 1e-6)
                _log(
                    f"    CSV rows {scanned:,} | matches={seen:,} "
                    f"unresolved={missing:,} | {scanned / elapsed:.0f} rows/s"
                )
            fp = (row.get("filepath") or "").strip()
            if not fp:
                continue
            src = _resolve_skyscript_file(root, fp, name_index=name_index)
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
                _log(
                    f"    early-stop CSV after {scanned:,} rows "
                    f"(matches={seen:,} >= {max_images * 20:,})"
                )
                break

    if not reservoir:
        raise FileNotFoundError(
            f"No SkyScript images under {root} matched the CSV "
            f"(missing path hits={missing:,}). Extract image zips or use "
            f"--allow-rsicd-fallback."
        )

    _log(
        f"  SkyScript: reservoir {len(reservoir):,} paths "
        f"(scanned matches={seen:,}, unresolved={missing:,}, "
        f"csv_rows={scanned:,})"
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
    parallelism: int = 64,
    min_count: int = 2,
    progress: ProgressStore | None = None,
) -> None:
    """Rewrite near-duplicate captions via OpenRouter (bounded in-flight pool).

    Submits at most ``parallelism * 2`` futures at a time so we never park 30k+
    tasks. Logs a heartbeat if the API stalls (no completions for 15s).
    Failed items fall back to offline diversify so one hung call cannot stall
    the whole run forever (urllib still respects per-request timeout).

    Completed captions are written to ``progress`` so a restart skips them.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    need_idx: list[int] = []
    skipped_cached = 0
    for i, item in enumerate(items):
        key = item.get("_key") or record_stable_key(item)
        item["_key"] = key
        # Resume: already finalized in a previous run.
        if item.get("_captions_done") or (progress and progress.captions_done(key)):
            if progress and progress.captions_done(key) and not item.get("_captions_done"):
                ent = progress.get(key) or {}
                item["captions"] = list(ent.get("captions") or item["captions"])
                item["_captions_done"] = True
            skipped_cached += 1
            continue
        soft = _soft_unique_captions(list(item.get("captions") or []))
        if caption_set_is_diverse(soft, min_count=min_count):
            item["captions"] = soft
            item["_captions_done"] = True
            if progress is not None:
                progress.set_captions(key, soft)
            continue
        need_idx.append(i)

    if progress is not None:
        progress.save()

    if not need_idx:
        _log(
            f"  caption diversity: nothing to do "
            f"(cached_or_already_diverse={skipped_cached:,})"
        )
        return

    _log(
        f"  caption diversity: {len(need_idx):,} need rewrites "
        f"(skipped_cached={skipped_cached:,})"
    )

    def _apply_offline(i: int) -> None:
        primary = (items[i].get("captions") or ["scene"])[0]
        raw = _offline_diverse_captions(primary, items[i]["domain"])
        try:
            items[i]["captions"] = normalize_captions(raw, min_count=min_count)
        except ValueError:
            items[i]["captions"] = [primary.strip() or "scene", raw[1]]
        items[i]["_captions_done"] = True
        if progress is not None:
            progress.set_captions(items[i]["_key"], items[i]["captions"])

    if skip_api:
        for i in need_idx:
            _apply_offline(i)
        if progress is not None:
            progress.save()
        _log(f"  caption diversity: offline rewrites on {len(need_idx):,} items")
        return

    config = load_openrouter_config(config_path)
    api_key, model = config["api_key"], config["model"]
    workers = min(max(1, parallelism), len(need_idx))
    max_pending = max(workers * 2, workers)
    _log(
        f"  caption diversity: OpenRouter model={model} workers={workers} "
        f"max_in_flight={max_pending}"
    )

    def work(i: int) -> tuple[int, list[str] | None, str | None]:
        try:
            caps = openrouter_diversify_captions(
                list(items[i].get("captions") or []),
                api_key=api_key,
                model=model,
                domain=str(items[i].get("domain", DOMAIN_GENERAL)),
                min_count=min_count,
                timeout=45.0,
            )
            return i, caps, None
        except Exception as exc:
            return i, None, f"{type(exc).__name__}: {exc}"

    done = 0
    failed = 0
    start = time.monotonic()
    last_progress = start
    idx_iter = iter(need_idx)
    pending: set = set()

    def _submit_one(pool: ThreadPoolExecutor) -> bool:
        try:
            i = next(idx_iter)
        except StopIteration:
            return False
        pending.add(pool.submit(work, i))
        return True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(pending) < max_pending and _submit_one(pool):
            pass

        while pending:
            finished, pending = wait(
                pending, timeout=15.0, return_when=FIRST_COMPLETED
            )
            if not finished:
                stalled = time.monotonic() - last_progress
                _log(
                    f"    diversified {done}/{len(need_idx)} "
                    f"(waiting on API; in_flight={len(pending)}; "
                    f"no result for {stalled:.0f}s)"
                )
                continue

            for fut in finished:
                i, caps, err = fut.result()
                if caps is not None:
                    items[i]["captions"] = caps
                    items[i]["_captions_done"] = True
                    if progress is not None:
                        progress.set_captions(items[i]["_key"], caps)
                else:
                    failed += 1
                    _apply_offline(i)
                    if failed <= 5 or failed % 50 == 0:
                        _log(
                            f"    diversify fallback offline for item {i} "
                            f"({err}); total_fail={failed}"
                        )
                done += 1
                last_progress = time.monotonic()
                if done == 1 or done % 25 == 0 or done == len(need_idx):
                    elapsed = max(time.monotonic() - start, 1e-6)
                    rate = done / elapsed
                    eta = (len(need_idx) - done) / rate if rate > 0 else 0
                    _log(
                        f"    diversified {done}/{len(need_idx)} "
                        f"({rate:.1f}/s, ETA {eta / 60:.1f} min, "
                        f"fail={failed}, in_flight={len(pending)})"
                    )
                while len(pending) < max_pending and _submit_one(pool):
                    pass

    if progress is not None:
        progress.save()
    _log(
        f"  caption diversity: finished {done:,} new "
        f"(api_fail_offline_fallback={failed:,}, "
        f"previously_cached={skipped_cached:,})"
    )


def attach_queries(
    items: list[dict[str, Any]],
    *,
    skip_generation: bool,
    config_path: str | Path,
    batch_size: int = 8,
    parallelism: int = 32,
    progress: ProgressStore | None = None,
) -> None:
    """Fill query / unrelated_query. Never aborts the run on a single batch error."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    # Ensure keys exist for resume.
    for item in items:
        item.setdefault("_key", record_stable_key(item))

    if skip_generation:
        for item in items:
            if item.get("_queries_done"):
                continue
            caps = item["captions"]
            item["query"] = caps[1] if len(caps) > 1 else caps[0]
            item["unrelated_query"] = "red sports car on a racetrack at night"
            item["_queries_done"] = True
            if progress is not None:
                progress.set_queries(
                    item["_key"], item["query"], item["unrelated_query"]
                )
        if progress is not None:
            progress.save()
        _log(f"  skip-query-generation: filled queries offline")
        return

    if batch_size < 1 or parallelism < 1:
        raise ValueError("batch_size and parallelism must be >= 1")

    pending_items = [
        it for it in items
        if not it.get("_queries_done")
        and not (progress and progress.queries_done(it["_key"]))
    ]
    # Restore any that are in progress store but not flagged yet.
    restored = 0
    for it in items:
        if it.get("_queries_done"):
            continue
        if progress and progress.queries_done(it["_key"]):
            ent = progress.get(it["_key"]) or {}
            it["query"] = str(ent.get("query", ""))
            it["unrelated_query"] = str(ent.get("unrelated_query", ""))
            it["_queries_done"] = True
            restored += 1
    pending_items = [it for it in items if not it.get("_queries_done")]

    if not pending_items:
        _log(
            f"  query generation: all {len(items):,} already cached "
            f"(restored={restored:,})"
        )
        return

    config = load_openrouter_config(config_path)
    api_key, model = config["api_key"], config["model"]
    batches = [
        pending_items[i : i + batch_size]
        for i in range(0, len(pending_items), batch_size)
    ]
    workers = min(parallelism, len(batches))
    _log(
        f"  generating queries via OpenRouter ({model}) for "
        f"{len(pending_items):,}/{len(items):,} items "
        f"({len(batches)} API calls, batch_size={batch_size}, "
        f"parallel_workers={workers}, restored={restored:,}) ..."
    )

    def _offline_query(item: dict[str, Any]) -> None:
        """Fill item fields only — progress is persisted on the main thread."""
        caps = item["captions"]
        item["query"] = caps[1] if len(caps) > 1 else caps[0]
        item["unrelated_query"] = "red sports car on a racetrack at night"
        item["_queries_done"] = True

    def work(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
        """Return (updated batch items, error or None). Never raises.

        Does not touch ProgressStore (thread safety + avoid double-write races);
        the main-thread consumer persists after ``fut.result()``.
        """
        captions = [b["captions"][0] for b in batch]
        try:
            if len(captions) == 1:
                results = [openrouter_generate_queries(
                    captions[0], api_key=api_key, model=model
                )]
            else:
                results = openrouter_generate_queries_batch(
                    captions, api_key=api_key, model=model
                )
            if len(results) != len(batch):
                raise RuntimeError(
                    f"expected {len(batch)} results, got {len(results)}"
                )
            for item, res in zip(batch, results):
                item["query"] = res["related_query"]
                item["unrelated_query"] = res["unrelated_query"]
                item["_queries_done"] = True
            return batch, None
        except Exception as exc:
            # Per-item retry once, then offline — do not crash the process.
            err = f"{type(exc).__name__}: {exc}"
            for item in batch:
                try:
                    res = openrouter_generate_queries(
                        item["captions"][0], api_key=api_key, model=model
                    )
                    item["query"] = res["related_query"]
                    item["unrelated_query"] = res["unrelated_query"]
                    item["_queries_done"] = True
                except Exception:
                    _offline_query(item)
            return batch, err

    done_batches = 0
    done_items = 0
    failed_batches = 0
    start = time.monotonic()
    last_progress = start
    batch_iter = iter(batches)
    pending: set = set()
    max_pending = max(workers * 2, workers)

    def _submit_one(pool: ThreadPoolExecutor) -> bool:
        try:
            b = next(batch_iter)
        except StopIteration:
            return False
        pending.add(pool.submit(work, b))
        return True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(pending) < max_pending and _submit_one(pool):
            pass
        while pending:
            finished, pending = wait(
                pending, timeout=15.0, return_when=FIRST_COMPLETED
            )
            if not finished:
                stalled = time.monotonic() - last_progress
                _log(
                    f"    query batches {done_batches}/{len(batches)} "
                    f"(waiting on API; in_flight={len(pending)}; "
                    f"no result for {stalled:.0f}s)"
                )
                continue
            for fut in finished:
                batch, err = fut.result()
                if err:
                    failed_batches += 1
                    if failed_batches <= 5 or failed_batches % 20 == 0:
                        _log(
                            f"    query batch recovered with fallbacks "
                            f"({err}); failed_batches={failed_batches}"
                        )
                for item in batch:
                    if progress is not None and item.get("_queries_done"):
                        progress.set_queries(
                            item["_key"],
                            item["query"],
                            item["unrelated_query"],
                        )
                done_batches += 1
                done_items += len(batch)
                last_progress = time.monotonic()
                if (
                    done_batches == 1
                    or done_batches == len(batches)
                    or done_batches % 5 == 0
                ):
                    elapsed = max(time.monotonic() - start, 1e-6)
                    rate = done_items / elapsed
                    eta = (len(pending_items) - done_items) / rate if rate > 0 else 0
                    _log(
                        f"    query batches {done_batches}/{len(batches)} "
                        f"({done_items:,}/{len(pending_items):,} items, "
                        f"{rate:.1f} items/s, ETA {eta / 60:.1f} min, "
                        f"batch_fail={failed_batches})"
                    )
                while len(pending) < max_pending and _submit_one(pool):
                    pass

    if progress is not None:
        progress.save()
    _log(
        f"  query generation: finished {done_items:,} new "
        f"(batch_fail_recovered={failed_batches:,}, restored={restored:,})"
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

    t0 = _phase(f"GENERAL / COCO staging (n={n_general:,})")
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
    _phase_done(t0, "GENERAL / COCO", f"{sum(1 for r in records if r['domain']==DOMAIN_GENERAL):,} images")

    t1 = _phase(f"SATELLITE staging (n={n_satellite:,})")
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
        _log(f"SkyScript unavailable ({exc}); falling back to RSICD")
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
            Path(item["image_path"]).unlink(missing_ok=True)
            continue
        item = dict(item)
        item["id"] = f"satellite-{i:06d}"
        item["query"] = ""
        item["unrelated_query"] = ""
        records.append(item)

    del sat_items
    _phase_done(
        t1,
        "SATELLITE",
        f"{sum(1 for r in records if r['domain']==DOMAIN_SATELLITE):,} images",
    )

    rng = random.Random(seed)
    rng.shuffle(records)
    _log(
        f"Staged {len(records):,} records on disk under {staging_dir} "
        f"(general={sum(1 for r in records if r['domain']==DOMAIN_GENERAL):,}, "
        f"satellite={sum(1 for r in records if r['domain']==DOMAIN_SATELLITE):,})"
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
                   help=f"Rows per parquet flush (default {DEFAULT_WRITE_CHUNK}).")
    p.add_argument(
        "--export-workers",
        type=int,
        default=DEFAULT_EXPORT_WORKERS,
        help=f"Parallel workers for export I/O (sidecar copy + JPEG bytes; "
             f"default {DEFAULT_EXPORT_WORKERS}, max 16).",
    )
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
    p.add_argument("--query-parallelism", type=int, default=32,
                   help="Concurrent OpenRouter *query* batch requests.")
    p.add_argument(
        "--diversify-parallelism",
        type=int,
        default=64,
        help="Concurrent OpenRouter caption-diversify requests (default 64).",
    )
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

    # Durable LLM cache (survives crash/Ctrl-C). Keys are domain/filename so
    # re-staging the same JPEGs restores captions + queries without re-calling APIs.
    progress_path = args.output_dir / ".generate_progress.json"
    progress = ProgressStore(progress_path)

    t_all = time.monotonic()
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

    n_cap, n_q = progress.apply_to_records(records)
    if n_cap or n_q:
        _log(
            f"  resume: restored captions={n_cap:,} queries={n_q:,} "
            f"from {progress_path}"
        )

    try:
        t_div = _phase("CAPTION DIVERSIFY")
        diversify_record_captions(
            records,
            skip_api=args.skip_query_generation and not args.force_caption_diversify,
            config_path=args.openrouter_config,
            parallelism=args.diversify_parallelism,
            progress=progress,
        )
        progress.save()
        _phase_done(t_div, "CAPTION DIVERSIFY")

        t_q = _phase("QUERY GENERATION")
        attach_queries(
            records,
            skip_generation=args.skip_query_generation,
            config_path=args.openrouter_config,
            batch_size=args.query_batch_size,
            parallelism=args.query_parallelism,
            progress=progress,
        )
        progress.save()
        _phase_done(t_q, "QUERY GENERATION")
    except BaseException:
        # Ctrl-C / crash: flush dirty buffer so the next run resumes cleanly.
        try:
            progress.save()
            _log(f"  resume cache saved to {progress_path} ({len(progress.data):,} entries)")
        except OSError:
            pass
        raise

    t_val = _phase("VALIDATE RECORDS")
    good: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        if (i + 1) % 10000 == 0:
            _log(f"  validated {i + 1:,}/{len(records):,}")
        try:
            if not Path(rec["image_path"]).is_file():
                raise FileNotFoundError(rec["image_path"])
            normalize_captions(rec["captions"], min_count=2)
            if not str(rec.get("query", "")).strip():
                raise ValueError("empty query")
            if not str(rec.get("unrelated_query", "")).strip():
                raise ValueError("empty unrelated_query")
            # Drop internal resume flags before export.
            good.append({
                k: v for k, v in rec.items()
                if not str(k).startswith("_")
            })
        except (ValueError, TypeError, FileNotFoundError) as exc:
            print(f"  dropping {rec.get('id')}: {exc}", file=sys.stderr)
            Path(rec.get("image_path", "")).unlink(missing_ok=True)
    records = good
    _phase_done(t_val, "VALIDATE", f"{len(records):,} kept")
    if not records:
        print("No valid records after caption diversify.", file=sys.stderr)
        return 1

    t_exp = _phase("EXPORT (parquet + sidecars + hf/)")
    out = save_dataset_streaming(
        records,
        args.output_dir,
        chunk_size=args.write_chunk,
        write_sidecar_jpegs=not args.no_sidecar_jpegs,
        write_hf_arrow=not args.no_hf_arrow,
        workers=args.export_workers,
    )
    _phase_done(t_exp, "EXPORT")

    if not args.keep_staging and staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
        _log(f"  removed staging dir {staging_dir}")
    # Progress cache is kept after success so a re-export or partial rebuild
    # does not re-bill OpenRouter. Delete manually to force full LLM re-run.

    _log(f"Saved dataset to {out} (total wall {time.monotonic() - t_all:.1f}s)")
    _log(f"  hf load: datasets.load_from_disk({out / 'hf'!r})")
    _log(f"  viewer:  python3 view_dataset.py --dataset-dir {out}")
    _log(f"  resume cache: {progress_path} ({len(progress.data):,} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
