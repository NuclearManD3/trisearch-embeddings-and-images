#!/usr/bin/env python3
"""Publish a curated TriSearch export to the Hugging Face Hub.

Workflow
--------
1. Validate local export (``metadata.jsonl``, images/parquet).
2. Optionally **sync parquet** from repaired ``metadata.jsonl`` + sidecar JPEGs
   so Hub text matches the post-repair source of truth.
3. Write a detailed dataset card (``README.md``), ``LICENSE``, ``dataset_info.json``.
4. Stage a clean upload tree (no progress caches / staging dirs).
5. Upload with ``huggingface_hub`` (``upload_large_folder`` by default).

Examples
--------
  # Card + validation only (no network upload)
  python3 publish_dataset.py --dry-run

  # Write card into the export, rebuild parquet from metadata, then push
  python3 publish_dataset.py \\
      --repo-id NuclearManD/trisearch-v1 \\
      --sync-parquet \\
      --public

  # Private draft repo
  python3 publish_dataset.py --repo-id NuclearManD/trisearch-v1-private --private
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
    save_dataset_streaming,
)
from trisearch_dataset_card import (
    DATASET_VERSION,
    DEFAULT_HF_REPO_HINT,
    collect_dataset_stats,
    write_dataset_card,
    write_license_file,
)
from trisearch_quality import load_metadata_rows


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


def sync_parquet_from_metadata(
    root: Path,
    *,
    workers: int = DEFAULT_EXPORT_WORKERS,
    write_hf: bool = False,
) -> None:
    """Rebuild ``data/*.parquet`` from ``metadata.jsonl`` + sidecar JPEGs.

    Keeps ``images/`` and ``metadata.jsonl`` in place (writes via temp dir).
    """
    import shutil as sh

    rows = load_metadata_rows(root)
    staged: list[dict[str, Any]] = []
    for row in rows:
        rel = row.get("file_name") or f"images/{row['domain']}/{row['id']}.jpg"
        img_path = root / str(rel)
        if not img_path.is_file():
            raise FileNotFoundError(f"sidecar missing for {row['id']}: {img_path}")
        staged.append(
            {
                "id": row["id"],
                "domain": row["domain"],
                "source": row.get("source", ""),
                "captions": list(row["captions"]),
                "query": row["query"],
                "unrelated_query": row["unrelated_query"],
                "image_path": str(img_path),
            }
        )

    tmp = root / ".publish_sync_tmp"
    if tmp.exists():
        sh.rmtree(tmp)
    _log(
        f"  sync parquet: {len(staged):,} rows → temp export "
        f"(workers={workers}, hf={write_hf}) ..."
    )
    t0 = time.monotonic()
    save_dataset_streaming(
        staged,
        tmp,
        write_sidecar_jpegs=False,
        write_hf_arrow=write_hf,
        workers=workers,
    )
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
    for name in ("dataset_info.json",):
        src = tmp / name
        if src.is_file():
            sh.copy2(src, root / name)
    sh.rmtree(tmp, ignore_errors=True)
    _log(f"  sync parquet done in {time.monotonic() - t0:.1f}s "
         f"({_dir_size_gb(root / 'data'):.2f} GB)")


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
        "parquet_glob": "data/train-*.parquet",
        "release": "preliminary",
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
    for name in ("README.md", "LICENSE", "dataset_info.json", "metadata.jsonl"):
        src = root / name
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

    # Parquet shards
    data_src = root / "data"
    parquet = sorted(data_src.glob("train-*.parquet")) if data_src.is_dir() else []
    if not parquet:
        raise FileNotFoundError(
            f"No parquet shards under {data_src}; run with --sync-parquet first"
        )
    for p in parquet:
        _link_or_copy(p, staging / "data" / p.name)

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


def push_to_hub(
    staging: Path,
    *,
    repo_id: str,
    private: bool,
    token: str | None,
    commit_message: str,
    large_folder: bool,
) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    _log(f"  ensure repo exists: {repo_id} (private={private})")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    _log(f"  uploading from {staging} ...")
    t0 = time.monotonic()
    if large_folder:
        # Best for multi-GB multi-file dataset uploads
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=str(staging),
            repo_type="dataset",
            # num_workers default is fine
        )
    else:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(staging),
            repo_type="dataset",
            commit_message=commit_message,
            ignore_patterns=["UPLOAD_MANIFEST.json"],
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
        "--no-large-folder",
        action="store_true",
        help="Use upload_folder instead of upload_large_folder",
    )
    p.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Commit message for non-large-folder upload",
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

    # 2) Optional parquet sync (text repair → hub snapshot)
    if args.sync_parquet:
        _log("\n[2/5] Sync parquet from metadata + sidecars")
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
        _log("\n[2/5] Sync parquet — skipped (pass --sync-parquet after repairs)")

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
            large_folder=not args.no_large_folder,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  upload failed: {exc}", file=sys.stderr)
        _log(f"  staging left at {staging}")
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
