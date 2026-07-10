#!/usr/bin/env python3
"""Publish a curated TriSearch export to the Hugging Face Hub.

Workflow
--------
1. Validate local export (``metadata.jsonl``, images/parquet).
2. Optionally **sync parquet** from repaired ``metadata.jsonl`` + sidecar JPEGs
   so Hub text matches the post-repair source of truth.
3. Write a detailed dataset card (``README.md``), ``LICENSE``, ``dataset_info.json``.
4. Stage a clean upload tree (no progress caches / staging dirs).
5. Upload with **resumable classic git-LFS** (XET disabled by default).

Upload reliability
------------------
``upload_large_folder`` + XET often **stalls** on multi‑GB dataset pushes.
This script defaults to:

* ``HF_HUB_DISABLE_XET=1`` (classic LFS)
* one file per commit (progress + easier resume)
* skip remote files whose size already matches
* per-file retries with exponential backoff

Re-run the same command after a stall to continue.

Examples
--------
  # Card + validation only (no network upload)
  python3 publish_dataset.py --dry-run

  # Rebuild parquet + push (resumable)
  python3 publish_dataset.py \\
      --repo-id NuclearManD/trisearch-v1 \\
      --sync-parquet --public --yes

  # Private draft
  python3 publish_dataset.py --repo-id NuclearManD/trisearch-v1-private --private --yes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from trisearch_data_format import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_EXPORT_WORKERS,
    DEFAULT_IMAGE_SIZE,
    DATASET_FORMAT_VERSION,
    OFFICIAL_SPLIT_SEED,
    OFFICIAL_TEST_DENOM,
    apply_official_splits,
    save_dataset_streaming,
)
from trisearch_dataset_card import (
    DATASET_VERSION,
    DEFAULT_HF_REPO_HINT,
    collect_dataset_stats,
    write_dataset_card,
    write_license_file,
)
from trisearch_quality import load_metadata_rows, write_metadata_jsonl


def _log(msg: str) -> None:
    print(msg, flush=True)


def _dir_size_gb(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / 1e9
    for dp, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(dp) / name).stat().st_size
            except OSError:
                pass
    return total / 1e9


def validate_export(root: Path, *, check_images: int = 64) -> list[str]:
    """Return list of error strings (empty ⇒ OK)."""
    errors: list[str] = []
    meta = root / "metadata.jsonl"
    if not meta.is_file():
        errors.append(f"missing {meta}")
        return errors

    try:
        rows = load_metadata_rows(root)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"metadata load failed: {exc}")
        return errors
    if len(rows) < 2:
        errors.append(f"too few rows: {len(rows)}")

    missing_img = 0
    checked = 0
    for row in rows:
        if checked >= check_images:
            break
        rel = row.get("file_name") or f"images/{row['domain']}/{row['id']}.jpg"
        path = root / str(rel)
        checked += 1
        if not path.is_file():
            missing_img += 1
            if missing_img <= 5:
                errors.append(f"missing image: {rel}")
    if missing_img:
        errors.append(f"{missing_img}/{checked} sampled sidecars missing")

    # Require either parquet or sidecars for a publishable package
    parquet = list((root / "data").glob("train-*.parquet")) if (root / "data").is_dir() else []
    if not parquet and not (root / "images").is_dir():
        errors.append("need data/train-*.parquet and/or images/ for publication")

    empty_q = sum(1 for r in rows if not str(r.get("query") or "").strip())
    empty_u = sum(1 for r in rows if not str(r.get("unrelated_query") or "").strip())
    if empty_q:
        errors.append(f"{empty_q} rows with empty query")
    if empty_u:
        errors.append(f"{empty_u} rows with empty unrelated_query")

    return errors


def ensure_official_splits(root: Path, *, force: bool = False) -> dict[str, int]:
    """Assign train/test (test = 1/16 per domain), write metadata + splits.json."""
    rows = load_metadata_rows(root)
    counts = apply_official_splits(rows, force=force)
    write_metadata_jsonl(root / "metadata.jsonl", rows)

    train_ids = sorted(r["id"] for r in rows if r.get("split") == "train")
    test_ids = sorted(r["id"] for r in rows if r.get("split") == "test")
    manifest = {
        "dataset_version": DATASET_VERSION,
        "split_seed": OFFICIAL_SPLIT_SEED,
        "test_denom": OFFICIAL_TEST_DENOM,
        "test_fraction": 1.0 / OFFICIAL_TEST_DENOM,
        "rule": (
            f"Per domain: sort ids, shuffle with Random(f'{{seed}}:{{domain}}'), "
            f"first len//{OFFICIAL_TEST_DENOM} → test, rest → train. "
            f"seed={OFFICIAL_SPLIT_SEED}."
        ),
        "counts": counts,
        "counts_by_domain": {},
        "train_ids": train_ids,
        "test_ids": test_ids,
    }
    by_dom: dict[str, dict[str, int]] = {}
    for r in rows:
        d = str(r["domain"])
        sp = str(r["split"])
        by_dom.setdefault(d, {"train": 0, "test": 0})
        by_dom[d][sp] += 1
    manifest["counts_by_domain"] = by_dom
    (root / "splits.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _log(
        f"  official splits: train={counts['train']:,} test={counts['test']:,} "
        f"(1/{OFFICIAL_TEST_DENOM} per domain, seed={OFFICIAL_SPLIT_SEED}) "
        f"→ {root / 'splits.json'}"
    )
    return counts


def sync_parquet_from_metadata(
    root: Path,
    *,
    workers: int = DEFAULT_EXPORT_WORKERS,
    write_hf: bool = False,
) -> None:
    """Rebuild ``data/{{train,test}}-*.parquet`` from metadata + sidecars.

    Ensures official splits first. Keeps ``images/`` in place (temp dir export).
    """
    import shutil as sh

    ensure_official_splits(root, force=False)
    rows = load_metadata_rows(root)

    def _stage(row: dict[str, Any]) -> dict[str, Any]:
        rel = row.get("file_name") or f"images/{row['domain']}/{row['id']}.jpg"
        img_path = root / str(rel)
        if not img_path.is_file():
            raise FileNotFoundError(f"sidecar missing for {row['id']}: {img_path}")
        return {
            "id": row["id"],
            "domain": row["domain"],
            "source": row.get("source", ""),
            "captions": list(row["captions"]),
            "query": row["query"],
            "unrelated_query": row["unrelated_query"],
            "split": row.get("split", "train"),
            "image_path": str(img_path),
        }

    train_rows = [_stage(r) for r in rows if r.get("split") == "train"]
    test_rows = [_stage(r) for r in rows if r.get("split") == "test"]
    if not train_rows or not test_rows:
        raise RuntimeError(
            f"split export needs both sides; train={len(train_rows)} test={len(test_rows)}"
        )

    tmp = root / ".publish_sync_tmp"
    if tmp.exists():
        sh.rmtree(tmp)
    _log(
        f"  sync parquet: train={len(train_rows):,} test={len(test_rows):,} "
        f"(workers={workers}, hf={write_hf}) ..."
    )
    t0 = time.monotonic()
    # Train first (clears temp), then test into same temp data/
    save_dataset_streaming(
        train_rows,
        tmp,
        write_sidecar_jpegs=False,
        write_hf_arrow=False,
        workers=workers,
        split_name="train",
        clear_output=True,
    )
    save_dataset_streaming(
        test_rows,
        tmp,
        write_sidecar_jpegs=False,
        write_hf_arrow=write_hf,
        workers=workers,
        split_name="test",
        clear_output=False,
    )
    # Prefer full metadata with split from root (already written by ensure_*)
    data_src = tmp / "data"
    if not data_src.is_dir():
        raise RuntimeError("sync failed: no data/ in temp export")
    data_dst = root / "data"
    if data_dst.exists():
        sh.rmtree(data_dst)
    sh.move(str(data_src), str(data_dst))
    if write_hf:
        hf_src = tmp / "hf"
        if hf_src.is_dir():
            hf_dst = root / "hf"
            if hf_dst.exists():
                sh.rmtree(hf_dst)
            sh.move(str(hf_src), str(hf_dst))
    sh.rmtree(tmp, ignore_errors=True)
    n_train = len(list((root / "data").glob("train-*.parquet")))
    n_test = len(list((root / "data").glob("test-*.parquet")))
    _log(
        f"  sync parquet done in {time.monotonic() - t0:.1f}s "
        f"({_dir_size_gb(root / 'data'):.2f} GB, "
        f"train_shards={n_train}, test_shards={n_test})"
    )


def update_dataset_info(root: Path, stats: dict[str, Any]) -> Path:
    info = {
        "format_version": stats.get("format_version", DATASET_FORMAT_VERSION),
        "dataset_version": stats.get("dataset_version", DATASET_VERSION),
        "num_rows": stats["num_rows"],
        "image_size": stats.get("image_size", DEFAULT_IMAGE_SIZE),
        "domains": stats["domains"],
        "sources": stats["sources"],
        "fields": [
            "id",
            "domain",
            "source",
            "captions",
            "query",
            "unrelated_query",
            "image",
        ],
        "unique_queries": stats["unique_queries"],
        "unique_unrelated": stats["unique_unrelated"],
        "query_collision_rate": stats["query_collision_rate"],
        "unrelated_collision_rate": stats["unrelated_collision_rate"],
        "hf_disk_path": "hf",
        "parquet_glob": "data/{train,test}-*.parquet",
        "splits": stats.get("splits") or {
            "train": "data/train-*.parquet",
            "test": "data/test-*.parquet",
            "test_fraction": 1.0 / OFFICIAL_TEST_DENOM,
            "seed": OFFICIAL_SPLIT_SEED,
        },
        "release": DATASET_VERSION,
        "generated_on": stats.get("generated_on"),
    }
    path = root / "dataset_info.json"
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_staging_dir(
    root: Path,
    staging: Path,
    *,
    include_sidecars: bool,
    include_hf_arrow: bool,
    include_quality_report: bool,
) -> dict[str, Any]:
    """Create a clean folder for Hub upload (symlinks for large blobs)."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    manifest: dict[str, Any] = {"files": [], "symlinks": []}

    def _link_or_copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(src.resolve(), dst)
            manifest["symlinks"].append(str(dst.relative_to(staging)))
        except OSError:
            shutil.copy2(src, dst)
            manifest["files"].append(str(dst.relative_to(staging)))

    # Required small files
    for name in (
        "README.md",
        "LICENSE",
        "dataset_info.json",
        "metadata.jsonl",
        "splits.json",
    ):
        src = root / name
        if name == "splits.json" and not src.is_file():
            continue
        if not src.is_file():
            raise FileNotFoundError(f"staging requires {src}")
        _link_or_copy(src, staging / name)

    pub_stats = root / "publication_stats.json"
    if pub_stats.is_file():
        _link_or_copy(pub_stats, staging / "publication_stats.json")

    if include_quality_report:
        qr = root / "quality" / "quality_report.json"
        if qr.is_file():
            _link_or_copy(qr, staging / "quality_report.json")

    # Parquet shards (train + test)
    data_src = root / "data"
    train_pq = sorted(data_src.glob("train-*.parquet")) if data_src.is_dir() else []
    test_pq = sorted(data_src.glob("test-*.parquet")) if data_src.is_dir() else []
    # Ignore incomplete part files
    train_pq = [p for p in train_pq if "-part-" not in p.name]
    test_pq = [p for p in test_pq if "-part-" not in p.name]
    if not train_pq:
        raise FileNotFoundError(
            f"No train parquet shards under {data_src}; run with --sync-parquet first"
        )
    if not test_pq:
        raise FileNotFoundError(
            f"No test parquet shards under {data_src}; run with --sync-parquet first"
        )
    for p in train_pq + test_pq:
        _link_or_copy(p, staging / "data" / p.name)
    parquet = train_pq + test_pq

    if include_sidecars and (root / "images").is_dir():
        # Symlink whole tree is awkward; link domain dirs
        for domain_dir in sorted((root / "images").iterdir()):
            if domain_dir.is_dir():
                _link_or_copy(domain_dir, staging / "images" / domain_dir.name)

    if include_hf_arrow and (root / "hf").is_dir():
        _link_or_copy(root / "hf", staging / "hf")

    # .gitattributes for LFS-friendly large files (Hub handles parquet LFS)
    gitattributes = (
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
        "*.jpg filter=lfs diff=lfs merge=lfs -text\n"
        "*.jpeg filter=lfs diff=lfs merge=lfs -text\n"
        "*.arrow filter=lfs diff=lfs merge=lfs -text\n"
    )
    (staging / ".gitattributes").write_text(gitattributes, encoding="utf-8")
    manifest["files"].append(".gitattributes")

    manifest["parquet_shards"] = len(parquet)
    manifest["staging_gb_approx"] = round(_dir_size_gb(staging), 2)
    # Follow symlinks for size of targets
    manifest["payload_gb_approx"] = round(
        _dir_size_gb(root / "data")
        + (_dir_size_gb(root / "images") if include_sidecars else 0)
        + (_dir_size_gb(root / "hf") if include_hf_arrow else 0)
        + 0.05,
        2,
    )
    (staging / "UPLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _disable_hf_xet() -> None:
    """Force classic git-LFS uploads. XET often stalls on multi‑GB dataset pushes.

    Must patch both the env var and the already-imported constants flag
    (``HF_HUB_DISABLE_XET`` is read once at import time).
    """
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    # hf_transfer is download-oriented; leave off so it cannot interfere.
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    try:
        from huggingface_hub import constants as hf_constants

        hf_constants.HF_HUB_DISABLE_XET = True
    except Exception:  # noqa: BLE001
        pass


def _iter_staging_payload(staging: Path) -> list[tuple[str, Path, int]]:
    """Return ``(path_in_repo, local_resolved_path, size_bytes)`` for upload.

    Follows symlinks so staging can be lightweight. Skips internal manifests
    and accidental ``.cache/`` trees left by interrupted hub uploads.
    """
    skip_names = {"UPLOAD_MANIFEST.json"}
    skip_prefixes = (".cache/", ".git/")
    items: list[tuple[str, Path, int]] = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        if path.name in skip_names or path.name.startswith("."):
            # Keep .gitattributes; skip other dotfiles.
            if path.name != ".gitattributes":
                continue
        rel = path.relative_to(staging).as_posix()
        if rel.startswith(skip_prefixes) or "/.cache/" in f"/{rel}":
            continue
        try:
            real = path.resolve(strict=True)
        except OSError:
            _log(f"  skip broken symlink: {rel}")
            continue
        if not real.is_file():
            continue
        items.append((rel, real, real.stat().st_size))
    # Stable order: small metadata first, large parquets last (progress + resume).
    def _key(it: tuple[str, Path, int]) -> tuple[int, str]:
        rel, _, size = it
        is_big = 1 if size >= 8 * 1024 * 1024 or rel.startswith("data/") else 0
        return (is_big, rel)

    items.sort(key=_key)
    return items


def _remote_file_sizes(api: Any, repo_id: str) -> dict[str, int]:
    """Map path_in_repo → size for files already on the Hub (best-effort)."""
    out: dict[str, int] = {}
    try:
        # list_repo_tree is more reliable than list_repo_files for sizes
        for entry in api.list_repo_tree(
            repo_id, repo_type="dataset", recursive=True
        ):
            path = getattr(entry, "path", None)
            size = getattr(entry, "size", None)
            if path is not None and size is not None:
                out[str(path)] = int(size)
    except Exception as exc:  # noqa: BLE001
        _log(f"  warn: could not list remote files ({exc}); will re-upload all")
    return out


def _upload_one_with_retries(
    api: Any,
    *,
    local_path: Path,
    path_in_repo: str,
    repo_id: str,
    commit_message: str,
    max_attempts: int = 8,
) -> None:
    """Upload a single file via classic LFS; retry on transient network errors."""
    last: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )
            return
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 — network stack varies
            last = exc
            # Don't thrash on auth / not-found style errors
            msg = str(exc).lower()
            if any(
                s in msg
                for s in (
                    "401",
                    "403",
                    "unauthorized",
                    "forbidden",
                    "invalid username",
                    "invalid token",
                )
            ):
                raise
            wait = min(120.0, 2.0 ** min(attempt, 6)) + (0.1 * attempt)
            _log(
                f"    retry {attempt}/{max_attempts} for {path_in_repo} "
                f"after {type(exc).__name__}: {exc}  (sleep {wait:.0f}s)"
            )
            time.sleep(wait)
    assert last is not None
    raise RuntimeError(
        f"Failed to upload {path_in_repo} after {max_attempts} attempts"
    ) from last


def push_to_hub(
    staging: Path,
    *,
    repo_id: str,
    private: bool,
    token: str | None,
    commit_message: str,
    large_folder: bool,
    upload_workers: int = 1,
    max_attempts: int = 8,
) -> str:
    """Push staging tree to the Hub with stall-resistant defaults.

    Default path (**resumable LFS**):
      * disables XET (frequent multi‑GB stalls)
      * uploads one file per commit (classic git LFS)
      * skips remote files that already match local size
      * retries each file with exponential backoff

    ``large_folder=True`` keeps ``upload_large_folder`` as an opt-in (XET-heavy,
    can hang on the final commit for big datasets).
    """
    _disable_hf_xet()

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    _log(f"  ensure repo exists: {repo_id} (private={private})")
    _log("  upload transport: classic git-LFS (HF_HUB_DISABLE_XET=1)")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    _log(f"  uploading from {staging} ...")
    t0 = time.monotonic()

    if large_folder:
        _log(
            "  mode=upload_large_folder (opt-in; may stall on XET/commit — "
            "prefer default resumable mode)"
        )
        # Still force XET off; large_folder then uses LFS batch path.
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=str(staging),
            repo_type="dataset",
            num_workers=max(1, min(upload_workers, 4)),
            print_report=True,
            print_report_every=30,
        )
    else:
        payload = _iter_staging_payload(staging)
        total_bytes = sum(s for _, _, s in payload)
        _log(
            f"  mode=resumable per-file LFS: {len(payload)} files, "
            f"{total_bytes / 1e9:.2f} GB"
        )
        remote = _remote_file_sizes(api, repo_id)
        if remote:
            _log(f"  remote already has {len(remote)} file path(s)")

        todo: list[tuple[str, Path, int]] = []
        skipped = 0
        for rel, local, size in payload:
            rsz = remote.get(rel)
            if rsz is not None and rsz == size:
                skipped += 1
                continue
            todo.append((rel, local, size))
        _log(
            f"  skip {skipped} already-matching file(s); "
            f"upload {len(todo)} remaining"
        )

        done_bytes = 0
        todo_bytes = sum(s for _, _, s in todo)
        for i, (rel, local, size) in enumerate(todo, 1):
            msg = f"{commit_message} [{i}/{len(todo)}] {rel}"
            _log(
                f"  ({i}/{len(todo)}) {rel}  ({size / 1e6:.1f} MB) ..."
            )
            file_t0 = time.monotonic()
            _upload_one_with_retries(
                api,
                local_path=local,
                path_in_repo=rel,
                repo_id=repo_id,
                commit_message=msg,
                max_attempts=max_attempts,
            )
            dt = max(time.monotonic() - file_t0, 1e-6)
            done_bytes += size
            rate = size / dt / 1e6
            pct = 100.0 * done_bytes / todo_bytes if todo_bytes else 100.0
            eta = (
                (todo_bytes - done_bytes) / (done_bytes / max(time.monotonic() - t0, 1e-6))
                if done_bytes
                else 0
            )
            _log(
                f"    ok {rate:.1f} MB/s  overall {pct:.1f}%  "
                f"ETA {eta / 60:.1f} min"
            )

    url = f"https://huggingface.co/datasets/{repo_id}"
    _log(f"  upload finished in {time.monotonic() - t0:.1f}s → {url}")
    return url


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Local curated export (default {DEFAULT_DATASET_ROOT})",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_HF_REPO_HINT,
        help=f"Hub dataset repo id (default {DEFAULT_HF_REPO_HINT})",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create/update repo as private",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Create/update repo as public (default unless --private)",
    )
    p.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token (default: cached login / HF_TOKEN env)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate, write card/LICENSE/staging plan; do not upload",
    )
    p.add_argument(
        "--card-only",
        action="store_true",
        help="Only write README.md + LICENSE + dataset_info + stats; no staging/upload",
    )
    p.add_argument(
        "--sync-parquet",
        action="store_true",
        help="Rebuild data/*.parquet from metadata.jsonl + images/ before publish "
             "(recommended after repair_dataset.py)",
    )
    p.add_argument(
        "--sync-hf-arrow",
        action="store_true",
        help="With --sync-parquet, also rebuild hf/ Arrow export",
    )
    p.add_argument(
        "--include-sidecars",
        action="store_true",
        help="Also upload images/ sidecars (≈2× size; usually unnecessary)",
    )
    p.add_argument(
        "--include-hf-arrow",
        action="store_true",
        help="Also upload hf/ Arrow tree (usually unnecessary if parquet present)",
    )
    p.add_argument(
        "--no-quality-report",
        action="store_true",
        help="Do not ship quality_report.json",
    )
    p.add_argument(
        "--export-workers",
        type=int,
        default=DEFAULT_EXPORT_WORKERS,
        help="Workers for --sync-parquet",
    )
    p.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Clean staging path (default: <dataset>/.hf_publish_stage)",
    )
    p.add_argument(
        "--keep-staging",
        action="store_true",
        help="Do not delete staging dir after upload",
    )
    p.add_argument(
        "--large-folder",
        action="store_true",
        help="Use upload_large_folder (can stall on multi-GB / XET). "
             "Default is resumable per-file classic LFS instead.",
    )
    p.add_argument(
        "--no-large-folder",
        action="store_true",
        help=argparse.SUPPRESS,  # legacy alias: default is already non-large
    )
    p.add_argument(
        "--upload-workers",
        type=int,
        default=1,
        help="Parallelism for --large-folder only (default 1; keep low).",
    )
    p.add_argument(
        "--upload-retries",
        type=int,
        default=8,
        help="Per-file upload attempts with backoff (default 8).",
    )
    p.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Commit message prefix for uploads",
    )
    p.add_argument(
        "--check-images",
        type=int,
        default=128,
        help="How many sidecar paths to existence-check during validate",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation before upload",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.dataset_dir.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    private = bool(args.private)
    if args.public:
        private = False

    t_all = time.monotonic()
    _log(f"=== TriSearch dataset publish ===")
    _log(f"  local:   {root}")
    _log(f"  repo:    {args.repo_id}")
    _log(f"  version: {DATASET_VERSION}")
    _log(f"  mode:    {'dry-run' if args.dry_run else 'card-only' if args.card_only else 'upload'}")

    # 1) Validate
    _log("\n[1/5] Validate export")
    errors = validate_export(root, check_images=args.check_images)
    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        if any("missing metadata" in e or "metadata load" in e for e in errors):
            return 1
        # Hard fail on empty queries; soft warn on partial image sample issues
        hard = [e for e in errors if "empty query" in e or "empty unrelated" in e or "too few" in e]
        if hard:
            return 1
        _log("  validation warnings present; continuing")
    else:
        _log("  OK")

    # 2) Official splits always; optional parquet sync (text repair → hub snapshot)
    _log("\n[2/5] Official train/test splits + optional parquet sync")
    try:
        ensure_official_splits(root, force=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  split assign failed: {exc}", file=sys.stderr)
        return 1
    if args.sync_parquet:
        try:
            sync_parquet_from_metadata(
                root,
                workers=args.export_workers,
                write_hf=args.sync_hf_arrow,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  sync failed: {exc}", file=sys.stderr)
            return 1
    else:
        test_pq = list((root / "data").glob("test-*.parquet")) if (root / "data").is_dir() else []
        if not test_pq:
            _log(
                "  WARNING: no data/test-*.parquet yet — Hub package needs "
                "--sync-parquet before upload"
            )
        else:
            _log("  parquet sync skipped (existing train/test shards present)")

    # 3) Card + license + info
    _log("\n[3/5] Write dataset card + LICENSE + dataset_info")
    card_path, stats = write_dataset_card(root, repo_id=args.repo_id)
    lic_path = write_license_file(root)
    info_path = update_dataset_info(root, stats)
    _log(f"  README.md     → {card_path} ({card_path.stat().st_size:,} bytes)")
    _log(f"  LICENSE       → {lic_path}")
    _log(f"  dataset_info  → {info_path}")
    _log(f"  rows={stats['num_rows']:,}  domains={stats['domains']}")
    _log(
        f"  unique_q={stats['unique_queries']:,}  "
        f"unique_uq={stats['unique_unrelated']:,}"
    )

    if args.card_only:
        _log(f"\nCard-only done in {time.monotonic() - t_all:.1f}s")
        return 0

    # 4) Staging
    _log("\n[4/5] Build clean staging tree")
    staging = args.staging_dir or (root / ".hf_publish_stage")
    try:
        manifest = build_staging_dir(
            root,
            staging,
            include_sidecars=args.include_sidecars,
            include_hf_arrow=args.include_hf_arrow,
            include_quality_report=not args.no_quality_report,
        )
    except FileNotFoundError as exc:
        print(f"  staging failed: {exc}", file=sys.stderr)
        print("  Hint: run with --sync-parquet if data/ is missing or stale.", file=sys.stderr)
        return 1
    _log(
        f"  staging={staging}\n"
        f"  parquet_shards={manifest['parquet_shards']}  "
        f"payload≈{manifest['payload_gb_approx']} GB"
    )

    if args.dry_run:
        _log("\n[5/5] Upload — skipped (--dry-run)")
        _log("  Staging preserved for inspection:" if True else "")
        _log(f"  {staging}")
        _log(
            "\nTo upload for real:\n"
            f"  python3 publish_dataset.py --repo-id {args.repo_id} "
            f"--dataset-dir {root}"
            + (" --private" if private else " --public")
            + (" --sync-parquet" if args.sync_parquet else "")
            + " --yes"
        )
        _log(f"\nDry-run complete in {time.monotonic() - t_all:.1f}s")
        return 0

    # 5) Upload
    _log("\n[5/5] Upload to Hugging Face Hub")
    if not args.yes:
        print(
            f"\nAbout to upload ≈{manifest['payload_gb_approx']} GB to "
            f"https://huggingface.co/datasets/{args.repo_id}\n"
            f"  private={private}\n"
            "Re-run with --yes to confirm.\n",
            file=sys.stderr,
        )
        return 2

    token = args.token or os.environ.get("HF_TOKEN")
    commit = args.commit_message or (
        f"Publish {stats['dataset_name']} {stats['dataset_version']} "
        f"({stats['num_rows']} rows, preliminary)"
    )
    try:
        url = push_to_hub(
            staging,
            repo_id=args.repo_id,
            private=private,
            token=token,
            commit_message=commit,
            large_folder=bool(args.large_folder),
            upload_workers=args.upload_workers,
            max_attempts=max(1, args.upload_retries),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  upload failed: {exc}", file=sys.stderr)
        _log(
            f"  staging left at {staging}\n"
            "  Re-run the same command to resume — files already on the Hub "
            "with matching size are skipped."
        )
        return 1

    if not args.keep_staging and staging.exists():
        # Only remove if it is our default under dataset dir
        shutil.rmtree(staging, ignore_errors=True)
        _log(f"  removed staging {staging}")

    _log(f"\nPublished: {url}")
    _log(f"Total wall time {time.monotonic() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
