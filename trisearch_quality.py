"""Text-level quality checks for curated TriSearch datasets.

Used by ``audit_dataset.py`` (read-only report) and later by repair scripts.
All checks are CPU-only and operate on metadata fields — no model downloads.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from trisearch_data_format import (
    DOMAIN_GENERAL,
    DOMAIN_SATELLITE,
    VALID_DOMAINS,
    caption_set_is_diverse,
    caption_tokens,
    captions_are_near_duplicate,
)

# Offline / soft-fail fingerprints written by generate_datasets.py
OFFLINE_UNRELATED = "red sports car on a racetrack at night"

# High-frequency template buckets seen in real exports (topic labels, not scenes)
GENERIC_UNRELATED_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"^underwater(\s+sea)?\s+creatures?",
        r"^space exploration(\s+images?)?$",
        r"^mountain landscapes?$",
        r"^fashion trends?$",
        r"^food recipes?$",
        r"^historical (artifacts?|architecture)$",
        r"^wildlife in africa$",
        r"^night sky stars?$",
        r"^photos? of cats",
        r"^pictures? of sunsets?",
        r"^images? of snow",
    )
)

QUERY_BOILERPLATE = re.compile(
    r"^(image of|photo of|picture of|a photo of|an image of)\s+",
    re.I,
)
AERIALISH = re.compile(
    r"\b(aerial|overhead|satellite|from above|nadir|top[- ]down)\b",
    re.I,
)

# Severity: repair priority (higher = fix first)
SEVERITY = {
    "empty_field": 100,
    "bad_domain": 100,
    "captions_not_diverse": 90,
    "offline_unrelated": 85,
    "query_eq_caption": 80,
    "query_near_caption": 70,
    "generic_unrelated": 65,
    "unrelated_eq_query": 90,
    "unrelated_eq_caption": 85,
    "query_too_short": 60,
    "query_too_long": 40,
    "query_boilerplate": 45,
    "caption_too_short": 55,
    "domain_style_mismatch": 35,
    "duplicate_query_frequent": 50,
    "duplicate_unrelated_frequent": 55,
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower()).rstrip(".")


def token_overlap_ratio(a: str, b: str) -> float:
    """Fraction of tokens in ``a`` that also appear in ``b``."""
    ta, tb = caption_tokens(a), caption_tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def is_generic_unrelated(text: str) -> bool:
    t = _norm(text)
    if not t:
        return True
    if t == _norm(OFFLINE_UNRELATED):
        return True
    return any(p.search(t) for p in GENERIC_UNRELATED_PATTERNS)


def flag_row(
    row: dict[str, Any],
    *,
    query_freq: dict[str, int] | None = None,
    unrelated_freq: dict[str, int] | None = None,
    query_freq_threshold: int = 15,
    unrelated_freq_threshold: int = 100,
    query_caption_overlap: float = 0.85,
    min_query_len: int = 8,
    max_query_len: int = 120,
    min_caption_len: int = 12,
) -> list[dict[str, Any]]:
    """Return a list of flag dicts ``{code, severity, detail}`` for one row."""
    flags: list[dict[str, Any]] = []

    def add(code: str, detail: str) -> None:
        flags.append(
            {
                "code": code,
                "severity": SEVERITY.get(code, 10),
                "detail": detail,
            }
        )

    rid = str(row.get("id", "?"))
    domain = str(row.get("domain", ""))
    captions = [str(c).strip() for c in (row.get("captions") or []) if str(c).strip()]
    query = str(row.get("query", "")).strip()
    unrelated = str(row.get("unrelated_query", "")).strip()
    source = str(row.get("source", "")).strip()

    if domain not in VALID_DOMAINS:
        add("bad_domain", f"domain={domain!r}")
    if not source:
        add("empty_field", "source empty")
    if not query:
        add("empty_field", "query empty")
    if not unrelated:
        add("empty_field", "unrelated_query empty")
    if len(captions) < 2:
        add("empty_field", f"only {len(captions)} caption(s)")

    if captions and not caption_set_is_diverse(captions, min_count=2):
        add("captions_not_diverse", "caption set fails Jaccard diversity")
    else:
        for i in range(len(captions)):
            for j in range(i + 1, len(captions)):
                if captions_are_near_duplicate(captions[i], captions[j]):
                    add(
                        "captions_not_diverse",
                        f"near-dup pair [{i},{j}]",
                    )
                    break
            else:
                continue
            break

    for i, c in enumerate(captions):
        if len(c) < min_caption_len:
            add("caption_too_short", f"captions[{i}] len={len(c)}")

    if query:
        if len(query) < min_query_len:
            add("query_too_short", f"len={len(query)}")
        if len(query) > max_query_len:
            add("query_too_long", f"len={len(query)}")
        if QUERY_BOILERPLATE.search(query):
            add("query_boilerplate", "starts with Image/Photo of…")
        qn = _norm(query)
        for i, c in enumerate(captions):
            cn = _norm(c)
            if qn == cn:
                add("query_eq_caption", f"exact match captions[{i}]")
                break
            ov = token_overlap_ratio(query, c)
            if ov >= query_caption_overlap:
                add(
                    "query_near_caption",
                    f"token_overlap={ov:.2f} with captions[{i}]",
                )
                break

    if unrelated:
        if _norm(unrelated) == _norm(OFFLINE_UNRELATED):
            add("offline_unrelated", "offline fallback fingerprint")
        elif is_generic_unrelated(unrelated):
            add("generic_unrelated", unrelated[:80])
        if query and _norm(unrelated) == _norm(query):
            add("unrelated_eq_query", "unrelated_query == query")
        for i, c in enumerate(captions):
            if _norm(unrelated) == _norm(c):
                add("unrelated_eq_caption", f"equals captions[{i}]")
                break

    if domain == DOMAIN_GENERAL and query and AERIALISH.search(query):
        add("domain_style_mismatch", "general domain but aerial-ish query")
    if domain == DOMAIN_SATELLITE and query and not AERIALISH.search(query):
        # Soft: many sat queries omit aerial words but are fine; only flag if
        # captions also lack aerial cues (likely domain-wrong query style).
        cap_blob = " ".join(captions)
        if not AERIALISH.search(cap_blob) and not AERIALISH.search(query):
            pass  # don't spam — sat captions often say "Aerial view" already

    if query_freq is not None and query:
        f = query_freq.get(_norm(query), 0)
        if f >= query_freq_threshold:
            add(
                "duplicate_query_frequent",
                f"query used {f} times in corpus",
            )
    if unrelated_freq is not None and unrelated:
        f = unrelated_freq.get(_norm(unrelated), 0)
        if f >= unrelated_freq_threshold:
            add(
                "duplicate_unrelated_frequent",
                f"unrelated used {f} times in corpus",
            )

    return flags


def corpus_frequencies(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    q_freq: Counter[str] = Counter()
    u_freq: Counter[str] = Counter()
    for row in rows:
        q = _norm(str(row.get("query", "")))
        u = _norm(str(row.get("unrelated_query", "")))
        if q:
            q_freq[q] += 1
        if u:
            u_freq[u] += 1
    return dict(q_freq), dict(u_freq)


def audit_rows(
    rows: Sequence[dict[str, Any]],
    *,
    query_freq_threshold: int = 15,
    unrelated_freq_threshold: int = 100,
    query_caption_overlap: float = 0.85,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit all rows. Returns (flag_records, summary_report).

    Each flag_record::
        {"id", "domain", "source", "flags": [...], "max_severity", "needs_repair": bool}
    """
    q_freq, u_freq = corpus_frequencies(rows)
    flag_records: list[dict[str, Any]] = []
    code_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    n_flagged = 0

    for row in rows:
        domain = str(row.get("domain", "unknown"))
        domain_counts[domain] += 1
        flags = flag_row(
            row,
            query_freq=q_freq,
            unrelated_freq=u_freq,
            query_freq_threshold=query_freq_threshold,
            unrelated_freq_threshold=unrelated_freq_threshold,
            query_caption_overlap=query_caption_overlap,
        )
        for fl in flags:
            code_counts[fl["code"]] += 1
        if flags:
            n_flagged += 1
            max_sev = max(f["severity"] for f in flags)
            flag_records.append(
                {
                    "id": str(row.get("id", "")),
                    "domain": domain,
                    "source": str(row.get("source", "")),
                    "flags": flags,
                    "max_severity": max_sev,
                    "needs_repair": True,
                    "codes": sorted({f["code"] for f in flags}),
                }
            )

    # Repair buckets: which fields to rewrite
    repair_captions = sum(
        1
        for r in flag_records
        if any(
            c in r["codes"]
            for c in ("captions_not_diverse", "caption_too_short", "empty_field")
        )
        and any(f["code"].startswith("caption") or f["code"] == "empty_field" for f in r["flags"])
    )
    repair_query = sum(
        1
        for r in flag_records
        if any(
            c in r["codes"]
            for c in (
                "query_eq_caption",
                "query_near_caption",
                "query_too_short",
                "query_boilerplate",
                "duplicate_query_frequent",
                "empty_field",
            )
        )
    )
    repair_unrelated = sum(
        1
        for r in flag_records
        if any(
            c in r["codes"]
            for c in (
                "offline_unrelated",
                "generic_unrelated",
                "unrelated_eq_query",
                "unrelated_eq_caption",
                "duplicate_unrelated_frequent",
                "empty_field",
            )
        )
    )

    n = len(rows)
    unique_q = len(q_freq)
    unique_u = len(u_freq)
    summary: dict[str, Any] = {
        "num_rows": n,
        "num_flagged": n_flagged,
        "pct_flagged": round(100.0 * n_flagged / n, 2) if n else 0.0,
        "domains": dict(domain_counts),
        "flag_counts": dict(code_counts.most_common()),
        "unique_queries": unique_q,
        "unique_unrelated": unique_u,
        "query_collision_rate": round(1.0 - unique_q / n, 4) if n else 0.0,
        "unrelated_collision_rate": round(1.0 - unique_u / n, 4) if n else 0.0,
        "top_queries": [
            {"text": t, "count": c} for t, c in Counter(q_freq).most_common(15)
        ],
        "top_unrelated": [
            {"text": t, "count": c} for t, c in Counter(u_freq).most_common(15)
        ],
        "repair_estimate": {
            "rows_needing_any_repair": n_flagged,
            "likely_query_rewrites": repair_query,
            "likely_unrelated_rewrites": repair_unrelated,
            "likely_caption_rewrites": repair_captions,
            "note": (
                "Repair scripts should rewrite only flagged fields; "
                "images/staging stay untouched."
            ),
        },
        "thresholds": {
            "query_freq_threshold": query_freq_threshold,
            "unrelated_freq_threshold": unrelated_freq_threshold,
            "query_caption_overlap": query_caption_overlap,
        },
    }
    # Highest severity first for repair queues
    flag_records.sort(key=lambda r: (-r["max_severity"], r["id"]))
    return flag_records, summary


def load_metadata_rows(
    dataset_dir: str | Path,
    *,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Load text metadata only (no images) from ``metadata.jsonl``."""
    import json
    from pathlib import Path

    root = Path(dataset_dir)
    meta_path = root / "metadata.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"No metadata.jsonl under {root}; run generate_datasets export first"
        )
    rows: list[dict[str, Any]] = []
    with open(meta_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            rec = {
                "id": str(meta["id"]),
                "domain": str(meta["domain"]),
                "source": str(meta.get("source", "")),
                "captions": list(meta.get("captions") or []),
                "query": str(meta.get("query", "")),
                "unrelated_query": str(meta.get("unrelated_query", "")),
                "file_name": str(meta.get("file_name", "")),
            }
            if meta.get("split"):
                rec["split"] = str(meta["split"])
            rows.append(rec)
    return rows


def check_sidecar_images(
    dataset_dir: str | Path,
    rows: Sequence[dict[str, Any]],
    *,
    max_check: int | None = None,
) -> dict[str, Any]:
    """Optional existence / size checks on sidecar JPEGs (no full decode)."""
    root = Path(dataset_dir)
    missing = 0
    tiny = 0
    checked = 0
    examples: list[str] = []
    for row in rows:
        if max_check is not None and checked >= max_check:
            break
        rel = row.get("file_name") or f"images/{row['domain']}/{row['id']}.jpg"
        path = root / str(rel)
        checked += 1
        if not path.is_file():
            missing += 1
            if len(examples) < 10:
                examples.append(f"missing:{rel}")
            continue
        sz = path.stat().st_size
        if sz < 1024:
            tiny += 1
            if len(examples) < 10:
                examples.append(f"tiny:{rel}:{sz}B")
    return {
        "checked": checked,
        "missing": missing,
        "tiny": tiny,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Repair helpers (deterministic / local; no API)
# ---------------------------------------------------------------------------

QUERY_REPAIR_CODES = frozenset({
    "query_eq_caption",
    "query_near_caption",
    "query_too_short",
    "query_too_long",
    "query_boilerplate",
    "duplicate_query_frequent",
    "empty_field",
})
UNRELATED_REPAIR_CODES = frozenset({
    "offline_unrelated",
    "generic_unrelated",
    "unrelated_eq_query",
    "unrelated_eq_caption",
    "duplicate_unrelated_frequent",
    "empty_field",
})
CAPTION_REPAIR_CODES = frozenset({
    "captions_not_diverse",
    "caption_too_short",
})

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with",
    "from", "by", "is", "are", "was", "were", "be", "this", "that", "these",
    "those", "its", "their", "his", "her", "as", "into", "near", "over",
    "under", "above", "below", "some", "many", "few", "very", "just",
})


def fields_to_repair(codes: Sequence[str]) -> set[str]:
    """Map flag codes → field names: query / unrelated_query / captions."""
    code_set = set(codes)
    fields: set[str] = set()
    if code_set & QUERY_REPAIR_CODES:
        # empty_field alone is ambiguous — only force query if other query codes
        # or empty query will be handled by the row content check in the repairer.
        if code_set & (QUERY_REPAIR_CODES - {"empty_field"}) or "empty_field" in code_set:
            fields.add("query")
    if code_set & UNRELATED_REPAIR_CODES:
        fields.add("unrelated_query")
    if code_set & CAPTION_REPAIR_CODES:
        fields.add("captions")
    return fields


def strip_query_boilerplate(query: str) -> str:
    q = str(query).strip()
    q = QUERY_BOILERPLATE.sub("", q).strip()
    # Drop trailing period for search-style brevity.
    return q.rstrip(".")


def keyword_search_query(
    captions: Sequence[str],
    *,
    max_words: int = 8,
    domain: str = DOMAIN_GENERAL,
) -> str:
    """Build a cheap search-style query from caption content words."""
    # Prefer later captions when present (often more varied after diversify).
    texts = [str(c) for c in captions if str(c).strip()]
    if not texts:
        return "photo scene"
    ordered = list(reversed(texts))  # try non-primary first
    ordered.append(texts[0])
    best = ""
    best_ov = 1.0
    for text in ordered:
        words = [
            w
            for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in _STOPWORDS and len(w) > 2
        ]
        # Dedup preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for w in words:
            if w not in seen:
                seen.add(w)
                uniq.append(w)
        candidate = " ".join(uniq[:max_words]).strip()
        if not candidate:
            continue
        if domain == DOMAIN_SATELLITE and not AERIALISH.search(candidate):
            if "aerial" not in seen:
                candidate = f"aerial {candidate}"
        ov = max(token_overlap_ratio(candidate, c) for c in texts)
        if ov < best_ov:
            best_ov = ov
            best = candidate
        if ov < 0.75:
            return candidate
    return best or "photo scene details"


def local_query_repair(
    row: dict[str, Any],
    *,
    codes: Sequence[str],
    query_caption_overlap: float = 0.85,
) -> str | None:
    """Try to fix query without API. Returns new query or None if needs LLM."""
    code_set = set(codes)
    query = str(row.get("query", "")).strip()
    captions = [str(c) for c in (row.get("captions") or []) if str(c).strip()]
    domain = str(row.get("domain", DOMAIN_GENERAL))

    if not query or "query_boilerplate" in code_set:
        query = strip_query_boilerplate(query) if query else ""

    needs_rewrite = bool(
        code_set
        & {
            "query_eq_caption",
            "query_near_caption",
            "query_too_short",
            "duplicate_query_frequent",
            "empty_field",
        }
    ) or not query

    if not needs_rewrite:
        stripped = strip_query_boilerplate(query)
        if stripped != query:
            return stripped
        return None

    candidate = keyword_search_query(captions, domain=domain)
    if not candidate or len(candidate) < 8:
        return None
    # Reject if still too close to every caption.
    if captions and max(token_overlap_ratio(candidate, c) for c in captions) >= query_caption_overlap:
        # Last resort: take rarest words across captions
        return None
    if _norm(candidate) == _norm(query):
        return None
    return candidate


def build_distractor_bank(n: int = 80_000, *, seed: int = 42) -> list[str]:
    """Large bank of specific scene-level distractor search phrases (local, free)."""
    import itertools
    import random

    subjects = [
        "vintage bicycle", "ceramic teapot", "red fire hydrant", "wooden chess set",
        "glass aquarium", "brass trombone", "leather suitcase", "clay pottery wheel",
        "neon diner sign", "steam locomotive", "hot air balloon", "kite festival",
        "sushi platter", "waffle iron", "sewing machine", "violin case",
        "skateboard ramp", "lighthouse tower", "cactus garden", "ice skating rink",
        "ferris wheel", "harbor crane", "windmill farm", "bamboo forest path",
        "coral reef diver", "desert dune buggy", "maple syrup shack", "bakery storefront",
        "robot vacuum", "origami crane display", "stained glass window", "marble fountain",
        "cargo freighter", "gondola ride", "ski chalet", "lavender field",
        "street mural", "farmers market stall", "observatory dome", "carousel horses",
        "blacksmith forge", "pottery kiln", "hang glider launch", "beehive boxes",
        "subway platform", "clock tower", "pagoda roof", "coral aquarium tank",
        "typewriter desk", "film projector", "camping tent", "kayak on lake",
        "solar panel array", "greenhouse tomatoes", "orchard ladder", "barn owl nest",
        "cobblestone alley", "rope bridge", "volcano crater rim", "tide pool crabs",
        "jazz trumpet solo", "ballet pointe shoes", "rugby scrum", "pottery glaze jars",
        "macarons box", "espresso machine", "yarn knitting basket", "board game night",
        "model train layout", "koi pond bridge", "sandcastle contest", "aurora cabin",
    ]
    settings = [
        "at dusk", "in morning fog", "under string lights", "after rainfall",
        "in winter snow", "beside a canal", "on a rooftop", "in a sunlit studio",
        "near a cliff edge", "inside a warehouse", "along a riverbank",
        "in autumn leaves", "on cobblestones", "under a glass dome",
        "in golden hour light", "beside cherry blossoms", "on a ferry deck",
        "in a crowded bazaar", "at a quiet lakeshore", "on a mountain trail",
        "in neon city rain", "beside sandstone cliffs", "in a library aisle",
        "on a wooden pier", "in a greenhouse aisle", "at a night market",
    ]
    styles = [
        "close-up photo", "wide angle shot", "candid street photo",
        "travel snapshot", "documentary photo", "detail macro shot",
        "handheld photo", "tripod long exposure", "smartphone photo",
        "film grain photo", "color photograph", "natural light photo",
    ]

    rng = random.Random(seed)
    phrases: list[str] = []
    # Combinatorial generation with shuffle for uniqueness
    combos = list(itertools.product(subjects, settings, styles))
    rng.shuffle(combos)
    for subj, setting, style in combos:
        phrases.append(f"{style} of {subj} {setting}")
        if len(phrases) >= n:
            break
    # Extra numeric variants if still short
    i = 0
    while len(phrases) < n:
        subj = subjects[i % len(subjects)]
        setting = settings[(i * 3) % len(settings)]
        style = styles[(i * 7) % len(styles)]
        phrases.append(f"{style}: {subj} {setting} view {i % 97}")
        i += 1
    # Dedup preserve order
    out: list[str] = []
    seen: set[str] = set()
    for p in phrases:
        k = _norm(p)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
        if len(out) >= n:
            break
    return out


def row_text_blocklist(row: dict[str, Any]) -> set[str]:
    """Normalized strings that a new unrelated_query must not equal."""
    banned = {_norm(str(row.get("query", "")))}
    for c in row.get("captions") or []:
        banned.add(_norm(str(c)))
    banned.discard("")
    return banned


def assign_unrelated_from_bank(
    row: dict[str, Any],
    *,
    bank: Sequence[str],
    used: set[str],
    bank_index: list[int],
) -> str | None:
    """Pick next unused bank phrase not conflicting with the row. Mutates used.

    ``bank_index`` is a one-element list holding the next bank cursor.
    """
    banned = row_text_blocklist(row)
    n = len(bank)
    if n == 0:
        return None
    start = bank_index[0] % n
    for offset in range(n):
        idx = (start + offset) % n
        phrase = bank[idx]
        key = _norm(phrase)
        if key in used or key in banned or is_generic_unrelated(phrase):
            continue
        if len(phrase) < 12:
            continue
        used.add(key)
        bank_index[0] = idx + 1
        return phrase
    return None


def needs_unrelated_repair(codes: Sequence[str], row: dict[str, Any]) -> bool:
    if set(codes) & (UNRELATED_REPAIR_CODES - {"empty_field"}):
        return True
    if not str(row.get("unrelated_query", "")).strip():
        return True
    if is_generic_unrelated(str(row.get("unrelated_query", ""))):
        return True
    return False


def needs_query_repair(codes: Sequence[str], row: dict[str, Any]) -> bool:
    if set(codes) & (QUERY_REPAIR_CODES - {"empty_field"}):
        return True
    if not str(row.get("query", "")).strip():
        return True
    return False


def write_metadata_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Atomic rewrite of metadata.jsonl (preserves known fields)."""
    import json
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                domain = str(row["domain"])
                rid = str(row["id"])
                file_name = str(
                    row.get("file_name") or f"images/{domain}/{rid}.jpg"
                )
                rec = {
                    "file_name": file_name,
                    "id": rid,
                    "domain": domain,
                    "source": str(row.get("source", "")),
                    "captions": list(row.get("captions") or []),
                    "query": str(row.get("query", "")),
                    "unrelated_query": str(row.get("unrelated_query", "")),
                }
                if row.get("split"):
                    rec["split"] = str(row["split"])
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
