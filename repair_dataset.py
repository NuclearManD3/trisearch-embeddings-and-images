#!/usr/bin/env python3
"""Repair / improve an existing TriSearch export in place (no full regen).

Mutates ``metadata.jsonl`` text fields only by default. Images stay untouched.

Pipeline
--------
1. Load ``metadata.jsonl`` (+ optional ``quality/flags.jsonl`` or fresh audit).
2. **Local unrelated_query repair** — large distractor bank, corpus novelty ($0).
3. **Local query repair** — strip boilerplate, keyword queries from captions.
4. **Optional LLM query repair** — OpenRouter only for rows local repair can't fix.
5. Atomic rewrite of ``metadata.jsonl``; optional parquet/hf rebuild from sidecars.
6. Re-audit → ``quality/`` updated.

Resume: ``.repair_progress.json`` under the dataset root.

Examples
--------
  # Free local pass (recommended first)
  python3 repair_dataset.py --local-only

  # Local + LLM for remaining bad queries
  python3 repair_dataset.py --query-parallelism 32

  # Dry-run / limit
  python3 repair_dataset.py --local-only --dry-run --max-repair 500
  python3 repair_dataset.py --codes offline_unrelated,generic_unrelated --local-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from trisearch_data_format import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_EXPORT_WORKERS,
    save_dataset_streaming,
)
from trisearch_dataset import (
    DEFAULT_OPENROUTER_CONFIG,
    load_openrouter_config,
    openrouter_diversify_captions,
    openrouter_repair_related_query,
)
from trisearch_quality import (
    _norm,
    assign_unrelated_from_bank,
    audit_rows,
    build_distractor_bank,
    is_generic_unrelated,
    load_metadata_rows,
    local_query_repair,
    needs_query_repair,
    needs_unrelated_repair,
    token_overlap_ratio,
    write_metadata_jsonl,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


class RepairProgress:
    """Thread-safe JSON progress for repaired field values."""

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
                _log(f"  repair progress: loaded {len(self.data):,} from {self.path}")
            except (json.JSONDecodeError, OSError) as exc:
                _log(f"  repair progress: load failed ({exc}); starting empty")
                self.data = {}

    def get(self, rid: str) -> dict[str, Any]:
        with self._lock:
            ent = self.data.get(rid)
            return dict(ent) if isinstance(ent, dict) else {}

    def set_field(
        self,
        rid: str,
        field: str,
        value: Any,
        *,
        how: str,
        flush_every: int = 100,
    ) -> None:
        with self._lock:
            ent = self.data.setdefault(rid, {})
            ent[field] = value
            ent[f"{field}_how"] = how
            ent["updated"] = time.time()
            self._dirty += 1
            if self._dirty >= flush_every:
                self._save_unlocked()

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
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


def _load_flags(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    codes_filter: set[str] | None,
    query_freq_threshold: int,
    unrelated_freq_threshold: int,
    query_caption_overlap: float,
) -> dict[str, list[str]]:
    """Return id → flag codes. Prefer quality/flags.jsonl; else live audit."""
    flags_path = root / "quality" / "flags.jsonl"
    id_to_codes: dict[str, list[str]] = {}

    if flags_path.is_file():
        _log(f"  loading flags from {flags_path}")
        with open(flags_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = str(rec.get("id", ""))
                codes = list(rec.get("codes") or [])
                if codes_filter:
                    codes = [c for c in codes if c in codes_filter]
                if codes:
                    id_to_codes[rid] = codes
        _log(f"  flag file: {len(id_to_codes):,} rows with selected codes")
    else:
        _log("  no quality/flags.jsonl; running live audit ...")
        flag_records, summary = audit_rows(
            rows,
            query_freq_threshold=query_freq_threshold,
            unrelated_freq_threshold=unrelated_freq_threshold,
            query_caption_overlap=query_caption_overlap,
        )
        _log(
            f"  live audit flagged {summary['num_flagged']:,}/{summary['num_rows']:,}"
        )
        for rec in flag_records:
            codes = list(rec.get("codes") or [])
            if codes_filter:
                codes = [c for c in codes if c in codes_filter]
            if codes:
                id_to_codes[str(rec["id"])] = codes

    return id_to_codes


def _apply_progress(rows: list[dict[str, Any]], progress: RepairProgress) -> int:
    n = 0
    for row in rows:
        ent = progress.get(str(row["id"]))
        if not ent:
            continue
        changed = False
        if "query" in ent and ent["query"]:
            row["query"] = str(ent["query"])
            changed = True
        if "unrelated_query" in ent and ent["unrelated_query"]:
            row["unrelated_query"] = str(ent["unrelated_query"])
            changed = True
        if "captions" in ent and ent["captions"]:
            row["captions"] = list(ent["captions"])
            changed = True
        if changed:
            n += 1
    return n


def repair_unrelated_local(
    rows: list[dict[str, Any]],
    id_to_codes: dict[str, list[str]],
    *,
    progress: RepairProgress,
    dry_run: bool,
    max_repair: int | None,
) -> dict[str, int]:
    stats = {"considered": 0, "repaired": 0, "skipped_ok": 0, "failed": 0}

    # Seed "used" with all current good unrelated values for novelty.
    used: set[str] = set()
    for row in rows:
        u = str(row.get("unrelated_query", "")).strip()
        if u and not is_generic_unrelated(u):
            used.add(_norm(u))

    bank = build_distractor_bank(80_000)
    cursor = [0]
    _log(f"  unrelated bank size={len(bank):,}, reserved_used={len(used):,}")

    for row in rows:
        rid = str(row["id"])
        codes = id_to_codes.get(rid, [])
        if not needs_unrelated_repair(codes, row):
            stats["skipped_ok"] += 1
            continue
        # Already repaired in progress?
        ent = progress.get(rid)
        if ent.get("unrelated_query") and not is_generic_unrelated(str(ent["unrelated_query"])):
            row["unrelated_query"] = str(ent["unrelated_query"])
            stats["repaired"] += 1
            continue

        stats["considered"] += 1
        if max_repair is not None and stats["repaired"] >= max_repair:
            break

        phrase = assign_unrelated_from_bank(
            row, bank=bank, used=used, bank_index=cursor
        )
        if phrase is None:
            stats["failed"] += 1
            continue
        if dry_run:
            stats["repaired"] += 1
            continue
        row["unrelated_query"] = phrase
        progress.set_field(rid, "unrelated_query", phrase, how="local_bank")
        stats["repaired"] += 1
        if stats["repaired"] % 5000 == 0:
            _log(f"    unrelated local repaired {stats['repaired']:,} ...")

    if not dry_run:
        progress.save()
    return stats


def repair_query_local(
    rows: list[dict[str, Any]],
    id_to_codes: dict[str, list[str]],
    *,
    progress: RepairProgress,
    dry_run: bool,
    max_repair: int | None,
    query_caption_overlap: float,
) -> dict[str, int]:
    stats = {"considered": 0, "repaired": 0, "needs_llm": 0, "skipped_ok": 0}

    for row in rows:
        rid = str(row["id"])
        codes = id_to_codes.get(rid, [])
        if not needs_query_repair(codes, row):
            stats["skipped_ok"] += 1
            continue
        # Frequency collisions need distinctive LLM paraphrases; keyword
        # extraction often re-collides across similar scenes.
        code_set = set(codes)
        only_freq = code_set <= {"duplicate_query_frequent"} or code_set == {
            "duplicate_query_frequent",
            "domain_style_mismatch",
        }
        if only_freq or (
            "duplicate_query_frequent" in code_set
            and not code_set & {
                "query_eq_caption",
                "query_near_caption",
                "query_boilerplate",
                "query_too_short",
                "empty_field",
            }
        ):
            stats["needs_llm"] += 1
            continue

        ent = progress.get(rid)
        if ent.get("query") and ent.get("query_how") in ("local_keyword", "llm", "local_strip"):
            # Validate still good enough
            q = str(ent["query"])
            caps = row.get("captions") or []
            if (
                q
                and caps
                and max(token_overlap_ratio(q, c) for c in caps) < query_caption_overlap
                and _norm(q) not in {_norm(c) for c in caps}
            ):
                row["query"] = q
                stats["repaired"] += 1
                continue

        stats["considered"] += 1
        if max_repair is not None and stats["repaired"] >= max_repair:
            break

        new_q = local_query_repair(
            row, codes=codes, query_caption_overlap=query_caption_overlap
        )
        if new_q is None:
            stats["needs_llm"] += 1
            continue
        if dry_run:
            stats["repaired"] += 1
            continue
        row["query"] = new_q
        how = "local_strip" if "query_boilerplate" in codes and len(codes) == 1 else "local_keyword"
        progress.set_field(rid, "query", new_q, how=how)
        stats["repaired"] += 1

    if not dry_run:
        progress.save()
    return stats


def _query_still_bad(
    row: dict[str, Any],
    codes: list[str],
    *,
    query_caption_overlap: float,
    llm_done: bool,
) -> bool:
    q = str(row.get("query", "")).strip()
    caps = [str(c) for c in (row.get("captions") or []) if str(c).strip()]
    if not q or len(q) < 8:
        return True
    if any(_norm(q) == _norm(c) for c in caps):
        return True
    if caps and max(token_overlap_ratio(q, c) for c in caps) >= query_caption_overlap:
        return True
    # Corpus collisions: local keyword often re-collides; require an LLM rewrite.
    if "duplicate_query_frequent" in codes and not llm_done:
        return True
    return False


def repair_query_llm(
    rows: list[dict[str, Any]],
    id_to_codes: dict[str, list[str]],
    *,
    progress: RepairProgress,
    dry_run: bool,
    max_repair: int | None,
    config_path: Path,
    parallelism: int,
    query_caption_overlap: float,
) -> dict[str, int]:
    stats = {"considered": 0, "repaired": 0, "failed": 0, "skipped": 0}

    pending: list[dict[str, Any]] = []
    for row in rows:
        rid = str(row["id"])
        codes = id_to_codes.get(rid, [])
        if not needs_query_repair(codes, row):
            continue
        ent = progress.get(rid)
        llm_done = ent.get("query_how") == "llm" and bool(ent.get("query"))
        if llm_done:
            # Re-check quality; keep if good, else re-queue.
            row["query"] = str(ent["query"])
            if not _query_still_bad(
                row,
                codes,
                query_caption_overlap=query_caption_overlap,
                llm_done=True,
            ):
                stats["repaired"] += 1
                continue
        if not _query_still_bad(
            row,
            codes,
            query_caption_overlap=query_caption_overlap,
            llm_done=False,
        ):
            stats["skipped"] += 1
            continue
        pending.append(row)

    if max_repair is not None:
        pending = pending[: max(0, max_repair)]
    stats["considered"] = len(pending)
    if not pending:
        _log("  LLM query: nothing pending")
        return stats

    if dry_run:
        _log(f"  LLM query dry-run: would call API for {len(pending):,} rows")
        stats["repaired"] = len(pending)
        return stats

    config = load_openrouter_config(config_path)
    api_key, model = config["api_key"], config["model"]
    workers = max(1, min(int(parallelism), 64))
    _log(
        f"  LLM query repair: {len(pending):,} rows, model={model}, workers={workers}"
    )

    def work(row: dict[str, Any]) -> tuple[str, str | None, str | None]:
        rid = str(row["id"])
        try:
            q = openrouter_repair_related_query(
                list(row.get("captions") or []),
                api_key=api_key,
                model=model,
                domain=str(row.get("domain", "general")),
                bad_query=str(row.get("query", "")),
            )
            return rid, q, None
        except Exception as exc:  # noqa: BLE001
            return rid, None, f"{type(exc).__name__}: {exc}"

    done = 0
    start = time.monotonic()
    id_to_row = {str(r["id"]): r for r in pending}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending_futs = {pool.submit(work, r) for r in pending}
        while pending_futs:
            finished, pending_futs = wait(
                pending_futs, timeout=15.0, return_when=FIRST_COMPLETED
            )
            if not finished:
                _log(
                    f"    LLM query {done}/{len(pending)} "
                    f"(waiting; in_flight={len(pending_futs)})"
                )
                continue
            for fut in finished:
                rid, q, err = fut.result()
                if q:
                    id_to_row[rid]["query"] = q
                    progress.set_field(rid, "query", q, how="llm", flush_every=50)
                    stats["repaired"] += 1
                else:
                    stats["failed"] += 1
                    if stats["failed"] <= 5 or stats["failed"] % 50 == 0:
                        _log(f"    LLM fail {rid}: {err}")
                done += 1
                if done == 1 or done % 100 == 0 or done == len(pending):
                    elapsed = max(time.monotonic() - start, 1e-6)
                    rate = done / elapsed
                    eta = (len(pending) - done) / rate if rate else 0
                    _log(
                        f"    LLM query {done}/{len(pending)} "
                        f"({rate:.1f}/s, ETA {eta / 60:.1f} min, fail={stats['failed']})"
                    )
    progress.save()
    return stats


def repair_captions_llm(
    rows: list[dict[str, Any]],
    id_to_codes: dict[str, list[str]],
    *,
    progress: RepairProgress,
    dry_run: bool,
    config_path: Path,
    parallelism: int,
) -> dict[str, int]:
    from trisearch_quality import CAPTION_REPAIR_CODES

    stats = {"considered": 0, "repaired": 0, "failed": 0}
    pending = [
        r
        for r in rows
        if set(id_to_codes.get(str(r["id"]), [])) & CAPTION_REPAIR_CODES
    ]
    stats["considered"] = len(pending)
    if not pending:
        return stats
    if dry_run:
        _log(f"  caption LLM dry-run: {len(pending)} rows")
        return stats

    config = load_openrouter_config(config_path)
    api_key, model = config["api_key"], config["model"]
    workers = max(1, min(parallelism, 32))
    _log(f"  caption LLM repair: {len(pending)} rows, workers={workers}")

    def work(row: dict[str, Any]) -> tuple[str, list[str] | None, str | None]:
        rid = str(row["id"])
        try:
            caps = openrouter_diversify_captions(
                list(row.get("captions") or []),
                api_key=api_key,
                model=model,
                domain=str(row.get("domain", "general")),
            )
            return rid, caps, None
        except Exception as exc:  # noqa: BLE001
            return rid, None, str(exc)

    id_to_row = {str(r["id"]): r for r in pending}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, pending))
    for rid, caps, err in results:
        if caps:
            id_to_row[rid]["captions"] = caps
            progress.set_field(rid, "captions", caps, how="llm")
            stats["repaired"] += 1
        else:
            stats["failed"] += 1
            _log(f"    caption fail {rid}: {err}")
    progress.save()
    return stats


def rebuild_export(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    workers: int,
    write_hf: bool,
) -> None:
    """Rebuild parquet (+ optional hf/) from sidecar JPEGs + repaired metadata."""
    staged: list[dict[str, Any]] = []
    for row in rows:
        rel = row.get("file_name") or f"images/{row['domain']}/{row['id']}.jpg"
        img_path = root / str(rel)
        if not img_path.is_file():
            raise FileNotFoundError(img_path)
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
    # save_dataset_streaming clears data/hf/images — we must NOT wipe images.
    # So write parquet+hf into a temp dir then move data/ and hf/ only.
    tmp_out = root / ".repair_export_tmp"
    if tmp_out.exists():
        import shutil

        shutil.rmtree(tmp_out)
    save_dataset_streaming(
        staged,
        tmp_out,
        write_sidecar_jpegs=False,
        write_hf_arrow=write_hf,
        workers=workers,
    )
    import shutil

    data_src = tmp_out / "data"
    if data_src.is_dir():
        data_dst = root / "data"
        if data_dst.exists():
            shutil.rmtree(data_dst)
        shutil.move(str(data_src), str(data_dst))
    if write_hf:
        hf_src = tmp_out / "hf"
        if hf_src.is_dir():
            hf_dst = root / "hf"
            if hf_dst.exists():
                shutil.rmtree(hf_dst)
            shutil.move(str(hf_src), str(hf_dst))
    # Refresh dataset_info from tmp if present
    for name in ("dataset_info.json", "README.md"):
        src = tmp_out / name
        if src.is_file():
            shutil.copy2(src, root / name)
    shutil.rmtree(tmp_out, ignore_errors=True)
    _log(f"  rebuilt data/ (+hf={write_hf}) under {root}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument(
        "--local-only",
        action="store_true",
        help="Skip OpenRouter; only bank/keyword repairs",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-repair", type=int, default=None, help="Cap rows per phase")
    p.add_argument(
        "--codes",
        type=str,
        default="",
        help="Comma-separated flag codes to repair (default: all from flags file)",
    )
    p.add_argument("--skip-unrelated", action="store_true")
    p.add_argument("--skip-query", action="store_true")
    p.add_argument("--skip-captions", action="store_true")
    p.add_argument("--openrouter-config", type=Path, default=DEFAULT_OPENROUTER_CONFIG)
    p.add_argument("--query-parallelism", type=int, default=32)
    p.add_argument("--query-freq-threshold", type=int, default=15)
    p.add_argument("--unrelated-freq-threshold", type=int, default=100)
    p.add_argument("--query-caption-overlap", type=float, default=0.85)
    p.add_argument(
        "--rebuild-export",
        action="store_true",
        help="Rebuild data/*.parquet (and hf/ unless --no-hf) from sidecars",
    )
    p.add_argument("--no-hf", action="store_true")
    p.add_argument("--export-workers", type=int, default=DEFAULT_EXPORT_WORKERS)
    p.add_argument(
        "--no-reaudit",
        action="store_true",
        help="Skip post-repair audit_dataset write",
    )
    p.add_argument(
        "--fresh-progress",
        action="store_true",
        help="Ignore .repair_progress.json",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.dataset_dir
    meta_path = root / "metadata.jsonl"
    if not meta_path.is_file():
        print(f"No metadata.jsonl at {root}", file=sys.stderr)
        return 1

    t0 = time.monotonic()
    _log(f"Repairing dataset at {root}")
    rows = load_metadata_rows(root)
    _log(f"  loaded {len(rows):,} rows")

    progress_path = root / ".repair_progress.json"
    if args.fresh_progress and progress_path.is_file():
        progress_path.unlink()
        _log("  cleared repair progress")
    progress = RepairProgress(progress_path)
    n_applied = _apply_progress(rows, progress)
    if n_applied:
        _log(f"  applied prior progress to {n_applied:,} rows")

    codes_filter = None
    if args.codes.strip():
        codes_filter = {c.strip() for c in args.codes.split(",") if c.strip()}

    id_to_codes = _load_flags(
        root,
        rows,
        codes_filter=codes_filter,
        query_freq_threshold=args.query_freq_threshold,
        unrelated_freq_threshold=args.unrelated_freq_threshold,
        query_caption_overlap=args.query_caption_overlap,
    )
    # Rows that need work even without flag file (empty fields)
    for row in rows:
        rid = str(row["id"])
        codes = set(id_to_codes.get(rid, []))
        if not str(row.get("query", "")).strip():
            codes.add("empty_field")
        if not str(row.get("unrelated_query", "")).strip():
            codes.add("empty_field")
        if codes:
            id_to_codes[rid] = sorted(codes)

    _log(f"  repair queue: {len(id_to_codes):,} rows with flags")

    try:
        if not args.skip_unrelated:
            _log("\n=== LOCAL unrelated_query ===")
            st = repair_unrelated_local(
                rows,
                id_to_codes,
                progress=progress,
                dry_run=args.dry_run,
                max_repair=args.max_repair,
            )
            _log(f"  unrelated: {st}")

        if not args.skip_query:
            _log("\n=== LOCAL query ===")
            st = repair_query_local(
                rows,
                id_to_codes,
                progress=progress,
                dry_run=args.dry_run,
                max_repair=args.max_repair,
                query_caption_overlap=args.query_caption_overlap,
            )
            _log(f"  query local: {st}")

            if not args.local_only:
                _log("\n=== LLM query ===")
                st = repair_query_llm(
                    rows,
                    id_to_codes,
                    progress=progress,
                    dry_run=args.dry_run,
                    max_repair=args.max_repair,
                    config_path=args.openrouter_config,
                    parallelism=args.query_parallelism,
                    query_caption_overlap=args.query_caption_overlap,
                )
                _log(f"  query LLM: {st}")
            else:
                _log("  skipping LLM query (--local-only)")

        if not args.skip_captions and not args.local_only:
            _log("\n=== LLM captions (if needed) ===")
            st = repair_captions_llm(
                rows,
                id_to_codes,
                progress=progress,
                dry_run=args.dry_run,
                config_path=args.openrouter_config,
                parallelism=min(args.query_parallelism, 16),
            )
            _log(f"  captions: {st}")
    except BaseException:
        try:
            progress.save()
            _log(f"  progress saved to {progress_path}")
        except OSError:
            pass
        raise

    if args.dry_run:
        _log("\nDry-run: not writing metadata.jsonl")
        return 0

    _log("\n=== WRITE metadata.jsonl ===")
    write_metadata_jsonl(meta_path, rows)
    _log(f"  wrote {meta_path} ({len(rows):,} rows)")
    progress.save()

    if args.rebuild_export:
        _log("\n=== REBUILD parquet/hf ===")
        rebuild_export(
            root,
            rows,
            workers=args.export_workers,
            write_hf=not args.no_hf,
        )

    if not args.no_reaudit:
        _log("\n=== RE-AUDIT ===")
        flag_records, summary = audit_rows(
            rows,
            query_freq_threshold=args.query_freq_threshold,
            unrelated_freq_threshold=args.unrelated_freq_threshold,
            query_caption_overlap=args.query_caption_overlap,
        )
        qdir = root / "quality"
        qdir.mkdir(parents=True, exist_ok=True)
        (qdir / "quality_report.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with open(qdir / "flags.jsonl", "w", encoding="utf-8") as fh:
            for rec in flag_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _log(
            f"  post-repair: flagged {summary['num_flagged']:,}/{summary['num_rows']:,} "
            f"({summary['pct_flagged']}%)"
        )
        _log(f"  unique unrelated: {summary['unique_unrelated']:,} "
             f"(collision {summary['unrelated_collision_rate']})")
        _log(f"  unique queries: {summary['unique_queries']:,} "
             f"(collision {summary['query_collision_rate']})")
        _log("  flag counts:")
        for code, cnt in summary["flag_counts"].items():
            _log(f"    {cnt:6,}  {code}")

    _log(f"\nDone in {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
