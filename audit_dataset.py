#!/usr/bin/env python3
"""Audit a curated TriSearch dataset (text QC; optional sidecar checks).

Read-only. Writes under ``<dataset>/quality/`` by default:

  quality_report.json   — corpus summary + top collisions + repair estimates
  flags.jsonl           — one line per flagged row (repair queue)
  flags_by_code/        — optional split files for targeted repair jobs

Does **not** load images into RAM for the main pass (metadata.jsonl only).

Examples
--------
  python3 audit_dataset.py
  python3 audit_dataset.py --dataset-dir models/data/trisearch-v1
  python3 audit_dataset.py --check-images --max-image-check 5000
  python3 audit_dataset.py --split-by-code
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from trisearch_data_format import DEFAULT_DATASET_ROOT
from trisearch_quality import (
    audit_rows,
    check_sidecar_images,
    load_metadata_rows,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Curated dataset root (default {DEFAULT_DATASET_ROOT})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write reports (default: <dataset-dir>/quality)",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Audit only first N rows")
    p.add_argument(
        "--query-freq-threshold",
        type=int,
        default=15,
        help="Flag queries that appear ≥ this many times (default 15)",
    )
    p.add_argument(
        "--unrelated-freq-threshold",
        type=int,
        default=100,
        help="Flag unrelated_query strings that appear ≥ this many times",
    )
    p.add_argument(
        "--query-caption-overlap",
        type=float,
        default=0.85,
        help="Flag when query token overlap with a caption ≥ this (default 0.85)",
    )
    p.add_argument(
        "--check-images",
        action="store_true",
        help="Also check sidecar JPEG existence / min size",
    )
    p.add_argument(
        "--max-image-check",
        type=int,
        default=None,
        help="Limit image existence checks (default: all rows when --check-images)",
    )
    p.add_argument(
        "--split-by-code",
        action="store_true",
        help="Also write quality/flags_by_code/<code>.jsonl for repair fan-out",
    )
    p.add_argument(
        "--top-examples",
        type=int,
        default=5,
        help="Print this many example rows per top flag code",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.dataset_dir
    out_dir = args.output_dir or (root / "quality")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    _log(f"Auditing {root} ...")
    try:
        rows = load_metadata_rows(root, max_rows=args.max_rows)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    _log(f"  loaded {len(rows):,} metadata rows in {time.monotonic() - t0:.1f}s")

    t1 = time.monotonic()
    flag_records, summary = audit_rows(
        rows,
        query_freq_threshold=args.query_freq_threshold,
        unrelated_freq_threshold=args.unrelated_freq_threshold,
        query_caption_overlap=args.query_caption_overlap,
    )
    _log(
        f"  flagged {summary['num_flagged']:,}/{summary['num_rows']:,} "
        f"({summary['pct_flagged']}%) in {time.monotonic() - t1:.1f}s"
    )

    if args.check_images:
        t2 = time.monotonic()
        img_stats = check_sidecar_images(
            root, rows, max_check=args.max_image_check
        )
        summary["sidecar_images"] = img_stats
        _log(
            f"  sidecars: checked={img_stats['checked']:,} "
            f"missing={img_stats['missing']:,} tiny={img_stats['tiny']:,} "
            f"({time.monotonic() - t2:.1f}s)"
        )

    report_path = out_dir / "quality_report.json"
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _log(f"  wrote {report_path}")

    flags_path = out_dir / "flags.jsonl"
    with open(flags_path, "w", encoding="utf-8") as fh:
        for rec in flag_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _log(f"  wrote {flags_path} ({len(flag_records):,} lines)")

    if args.split_by_code:
        by_code: dict[str, list[dict]] = defaultdict(list)
        for rec in flag_records:
            for code in rec["codes"]:
                by_code[code].append(rec)
        split_dir = out_dir / "flags_by_code"
        if split_dir.exists():
            for p in split_dir.glob("*.jsonl"):
                p.unlink()
        split_dir.mkdir(parents=True, exist_ok=True)
        for code, recs in sorted(by_code.items()):
            path = split_dir / f"{code}.jsonl"
            with open(path, "w", encoding="utf-8") as fh:
                for rec in recs:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _log(f"  wrote {len(by_code)} files under {split_dir}")

    # Human-readable console summary
    _log("\n=== QUALITY SUMMARY ===")
    _log(f"  rows:     {summary['num_rows']:,}")
    _log(f"  flagged:  {summary['num_flagged']:,} ({summary['pct_flagged']}%)")
    _log(f"  domains:  {summary['domains']}")
    _log(
        f"  unique queries: {summary['unique_queries']:,} "
        f"(collision rate {summary['query_collision_rate']})"
    )
    _log(
        f"  unique unrelated: {summary['unique_unrelated']:,} "
        f"(collision rate {summary['unrelated_collision_rate']})"
    )
    _log("  flag counts:")
    for code, cnt in summary["flag_counts"].items():
        _log(f"    {cnt:6,}  {code}")
    re_ = summary["repair_estimate"]
    _log("  repair estimate (field-level, non-destructive):")
    _log(f"    query rewrites ≈ {re_['likely_query_rewrites']:,}")
    _log(f"    unrelated rewrites ≈ {re_['likely_unrelated_rewrites']:,}")
    _log(f"    caption rewrites ≈ {re_['likely_caption_rewrites']:,}")

    if summary.get("top_unrelated"):
        _log("  top unrelated_query values:")
        for item in summary["top_unrelated"][:8]:
            _log(f"    {item['count']:5,}  {item['text'][:70]}")

    # Example rows per top codes
    n_ex = max(0, args.top_examples)
    if n_ex and flag_records:
        top_codes = list(summary["flag_counts"].keys())[:5]
        by_code_ex: dict[str, list] = defaultdict(list)
        for rec in flag_records:
            for c in rec["codes"]:
                if c in top_codes and len(by_code_ex[c]) < n_ex:
                    by_code_ex[c].append(rec["id"])
        _log("  example ids by flag:")
        for code in top_codes:
            ids = by_code_ex.get(code) or []
            _log(f"    {code}: {', '.join(ids)}")

    _log(f"\nDone in {time.monotonic() - t0:.1f}s → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
