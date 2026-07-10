"""HuggingFace dataset card generator for TriSearch curated exports.

Produces a detailed README.md (YAML front-matter + Markdown body) from live
``metadata.jsonl`` / ``dataset_info.json`` stats. Used by ``publish_dataset.py``
and can be called standalone.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from trisearch_data_format import (
    DATASET_FORMAT_VERSION,
    DEFAULT_DATASET_ROOT,
    DEFAULT_IMAGE_SIZE,
)
from trisearch_quality import (
    _norm,
    corpus_frequencies,
    is_generic_unrelated,
    load_metadata_rows,
)

# Preliminary public release identity
DATASET_NAME = "TriSearch-v1"
DATASET_VERSION = "0.1.0-preliminary"
DEFAULT_HF_REPO_HINT = "NuclearManD/trisearch-v1"


def collect_dataset_stats(
    dataset_dir: str | Path,
    *,
    sample_seed: int = 42,
    n_examples: int = 4,
) -> dict[str, Any]:
    """Compute publication stats from an on-disk curated export."""
    root = Path(dataset_dir)
    rows = load_metadata_rows(root)
    if not rows:
        raise ValueError(f"No rows in {root / 'metadata.jsonl'}")

    domains: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    n_caps: list[int] = []
    cap_lens: list[int] = []
    q_lens: list[int] = []
    uq_lens: list[int] = []
    for r in rows:
        domains[str(r["domain"])] += 1
        sources[str(r.get("source") or "unknown")] += 1
        caps = list(r.get("captions") or [])
        n_caps.append(len(caps))
        cap_lens.extend(len(c) for c in caps)
        q_lens.append(len(str(r.get("query") or "")))
        uq_lens.append(len(str(r.get("unrelated_query") or "")))

    q_freq, u_freq = corpus_frequencies(rows)
    generic_uq = sum(
        1 for r in rows if is_generic_unrelated(str(r.get("unrelated_query") or ""))
    )

    quality_path = root / "quality" / "quality_report.json"
    quality: dict[str, Any] | None = None
    if quality_path.is_file():
        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            quality = None

    info_path = root / "dataset_info.json"
    info: dict[str, Any] = {}
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            info = {}

    parquet_files = sorted((root / "data").glob("train-*.parquet")) if (root / "data").is_dir() else []
    has_images = (root / "images").is_dir()
    has_hf = (root / "hf").is_dir()
    has_meta = (root / "metadata.jsonl").is_file()

    rng = random.Random(sample_seed)
    # Stratified-ish: prefer one general + one satellite when possible
    by_dom: dict[str, list[dict[str, Any]]] = {"general": [], "satellite": []}
    for r in rows:
        d = str(r["domain"])
        if d in by_dom and len(by_dom[d]) < 200:
            by_dom[d].append(r)
    examples: list[dict[str, Any]] = []
    for d in ("general", "satellite"):
        pool = by_dom.get(d) or []
        if pool:
            examples.append(rng.choice(pool))
    while len(examples) < n_examples and rows:
        pick = rng.choice(rows)
        if pick["id"] not in {e["id"] for e in examples}:
            examples.append(pick)

    def _ex(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": r["id"],
            "domain": r["domain"],
            "source": r.get("source", ""),
            "captions": list(r.get("captions") or [])[:3],
            "query": r.get("query", ""),
            "unrelated_query": r.get("unrelated_query", ""),
        }

    n = len(rows)
    return {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "format_version": info.get("format_version", DATASET_FORMAT_VERSION),
        "image_size": info.get("image_size", DEFAULT_IMAGE_SIZE),
        "num_rows": n,
        "domains": dict(domains),
        "sources": dict(sources),
        "captions_per_image": {
            "min": min(n_caps),
            "max": max(n_caps),
            "mean": round(statistics.mean(n_caps), 2),
        },
        "caption_char_len": {
            "mean": round(statistics.mean(cap_lens), 1) if cap_lens else 0,
            "p10": sorted(cap_lens)[len(cap_lens) // 10] if cap_lens else 0,
            "p90": sorted(cap_lens)[9 * len(cap_lens) // 10] if cap_lens else 0,
        },
        "query_char_len": {
            "mean": round(statistics.mean(q_lens), 1) if q_lens else 0,
            "p10": sorted(q_lens)[len(q_lens) // 10] if q_lens else 0,
            "p90": sorted(q_lens)[9 * len(q_lens) // 10] if q_lens else 0,
        },
        "unique_queries": len(q_freq),
        "unique_unrelated": len(u_freq),
        "query_collision_rate": round(1.0 - len(q_freq) / n, 4) if n else 0.0,
        "unrelated_collision_rate": round(1.0 - len(u_freq) / n, 4) if n else 0.0,
        "generic_unrelated_count": generic_uq,
        "quality": quality,
        "layout": {
            "metadata_jsonl": has_meta,
            "parquet_shards": len(parquet_files),
            "sidecar_images": has_images,
            "hf_arrow": has_hf,
        },
        "examples": [_ex(e) for e in examples],
        "generated_on": date.today().isoformat(),
        "root": str(root.resolve()),
    }


def _yaml_front_matter(
    stats: dict[str, Any],
    *,
    repo_id: str | None,
    pretty_name: str,
) -> str:
    n = int(stats["num_rows"])
    if n < 1_000:
        size_cat = "n<1K"
    elif n < 10_000:
        size_cat = "1K<n<10K"
    elif n < 100_000:
        size_cat = "10K<n<100K"
    else:
        size_cat = "100K<n<1M"

    tags = [
        "trisearch",
        "multimodal",
        "image-text",
        "contrastive-learning",
        "retrieval",
        "remote-sensing",
        "satellite",
        "coco",
        "skyscript",
        "matryoshka",
        "preliminary",
    ]
    # HuggingFace dataset card YAML (see huggingface.co/docs/hub/datasets-cards)
    lines = [
        "---",
        f"pretty_name: \"{pretty_name}\"",
        "license: other",
        "license_name: composite-upstream",
        "license_link: LICENSE",
        "language:",
        "  - en",
        "task_categories:",
        "  - image-to-text",
        "  - text-to-image",
        "  - image-classification",
        "  - feature-extraction",
        "task_ids:",
        "  - multi-class-image-classification",
        "  - multi-label-image-classification",
        f"size_categories:",
        f"  - {size_cat}",
        "annotations_creators:",
        "  - found",
        "  - machine-generated",
        "language_creators:",
        "  - found",
        "  - machine-generated",
        "multilinguality:",
        "  - monolingual",
        "source_datasets:",
        "  - extended|coco",
        "  - original",
        "paperswithcode_id: null",
        "tags:",
    ]
    for t in tags:
        lines.append(f"  - {t}")
    lines.extend(
        [
            "dataset_info:",
            f"  dataset_name: {stats['dataset_name']}",
            f"  version: {stats['dataset_version']}",
            f"  format_version: {stats['format_version']}",
            f"  num_examples: {n}",
            f"  image_size: {stats['image_size']}",
            "configs:",
            "  - config_name: default",
            "    data_files:",
            "      - split: train",
            "        path: data/train-*.parquet",
            "    default: true",
        ]
    )
    if repo_id:
        lines.append(f"# hub_repo: {repo_id}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _fmt_domains(domains: dict[str, int]) -> str:
    parts = [f"**{k}**: {v:,}" for k, v in sorted(domains.items())]
    return ", ".join(parts)


def _fmt_sources(sources: dict[str, int]) -> str:
    rows = [
        f"| `{k}` | {v:,} | {100.0 * v / max(sum(sources.values()), 1):.1f}% |"
        for k, v in sorted(sources.items(), key=lambda kv: -kv[1])
    ]
    return "\n".join(
        [
            "| Upstream source | Count | Share |",
            "|-----------------|------:|------:|",
            *rows,
        ]
    )


def _examples_md(examples: Sequence[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, ex in enumerate(examples, 1):
        caps = "\n".join(f"  - {c}" for c in ex.get("captions") or [])
        blocks.append(
            f"### Example {i} — `{ex['id']}` ({ex['domain']}, source: `{ex.get('source','')}`)\n\n"
            f"**Captions:**\n{caps}\n\n"
            f"**Query:** {ex.get('query', '')}\n\n"
            f"**Unrelated query:** {ex.get('unrelated_query', '')}\n"
        )
    return "\n".join(blocks)


def render_dataset_card(
    stats: dict[str, Any],
    *,
    repo_id: str | None = None,
    pretty_name: str | None = None,
    include_examples: bool = True,
) -> str:
    """Render full HF dataset card markdown (front-matter + body)."""
    pretty = pretty_name or f"{stats['dataset_name']} ({stats['dataset_version']})"
    n = int(stats["num_rows"])
    img = int(stats["image_size"])
    q = stats.get("quality") or {}
    layout = stats.get("layout") or {}

    flagged_line = ""
    if q:
        flagged_line = (
            f"- **Automated QC (text)**: {q.get('num_flagged', '?')} / "
            f"{q.get('num_rows', n)} rows still soft-flagged "
            f"({q.get('pct_flagged', '?')}%) after repair pass — "
            f"see Quality section.\n"
        )

    body = f"""# {pretty}

> **Preliminary public data release** for the TriSearch multimodal training stack.
> This is an **initial curated corpus** (v{stats['dataset_version']}) intended for
> research on joint image–text embedding spaces (contrastive + query-style text).
> Expect schema stability with possible additive fields and quality improvements
> in later versions. Not a final production benchmark.

## Summary

| | |
|--|--|
| **Version** | `{stats['dataset_version']}` (format v{stats['format_version']}) |
| **Examples** | **{n:,}** image–text records |
| **Image size** | **{img}×{img}** RGB JPEG (center-crop after short-side scale) |
| **Domains** | {_fmt_domains(stats['domains'])} |
| **Captions / image** | {stats['captions_per_image']['min']}–{stats['captions_per_image']['max']} (mean {stats['captions_per_image']['mean']}) |
| **Unique queries** | {stats['unique_queries']:,} (collision rate {stats['query_collision_rate']}) |
| **Unique unrelated queries** | {stats['unique_unrelated']:,} (collision rate {stats['unrelated_collision_rate']}) |
| **Language** | English |
| **Card generated** | {stats.get('generated_on', '')} |

{flagged_line}
**Why this dataset exists.** TriSearch trains dual towers (vision + text) into a
shared Matryoshka embedding space for cross-modal retrieval and alignment.
Stage‑1 training needs:

1. Real images (not synthetic placeholders)
2. Multiple **diverse** captions per image
3. A **search-style query** that should retrieve the image
4. An **unrelated query** (distractor) for contrastive / ranking supervision
5. Explicit **domain** labels for balanced satellite vs general mixes

This release packages those fields in a HuggingFace-friendly layout.

## Domain mix

Balanced **50 / 50** design:

- **general** — everyday photographs (MS-COCO via `bitmind/MS-COCO`)
- **satellite** — overhead / remote-sensing scenes (SkyScript)

{_fmt_sources(stats['sources'])}

## Dataset structure

Each row is one training example:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Stable unique id (`general-XXXXXX` / `satellite-XXXXXX`) |
| `domain` | `string` | `general` or `satellite` |
| `source` | `string` | Upstream corpus id |
| `image` | `Image` | RGB, {img}×{img}, JPEG-encoded in parquet |
| `captions` | `list[string]` | ≥2 diverse natural-language captions for the **same** image |
| `query` | `string` | Short search-style query expected to match this image |
| `unrelated_query` | `string` | Search-style distractor on a **different** topic |

### On-disk / Hub layout

```text
{repo_id or '<repo>'}/
  README.md                 # this card
  LICENSE                   # composite upstream notice
  dataset_info.json         # machine-readable stats
  metadata.jsonl            # text fields + relative image paths (no pixels)
  data/
    train-XXXXX-of-YYYYY.parquet   # full rows including embedded images
  quality_report.json       # optional automated QC snapshot (if shipped)
```

Local exports used during development may also include `images/` sidecars and an
Arrow `hf/` tree; the **Hub package prioritizes parquet** for `load_dataset`.

| Artifact | Present in local export |
|----------|-------------------------|
| `metadata.jsonl` | {layout.get('metadata_jsonl')} |
| parquet shards | {layout.get('parquet_shards')} |
| sidecar `images/` | {layout.get('sidecar_images')} |
| `hf/` Arrow | {layout.get('hf_arrow')} |

## How the data was built

### 1. Sampling & staging

- **General:** MS-COCO images from a local mirror of [`bitmind/MS-COCO`](https://huggingface.co/datasets/bitmind/MS-COCO),
  resized to {img}×{img} (scale short side, then center-crop), stored as high-quality JPEG.
- **Satellite:** SkyScript overhead imagery + language-polished captions (CSV index + local image zips),
  same resize policy.
- Target size **{n:,}** with fixed seed for reproducibility of the mix.

### 2. Caption diversity

Source corpora sometimes list near-duplicate captions (punctuation / preposition
swaps). We:

- Drop near-duplicates via token Jaccard filtering
- Call an LLM (OpenRouter) only when a caption set is still insufficiently diverse
- Offline fallback rewrites when the API fails (so pipelines never hang)

### 3. Query & distractor generation

- **`query`**: search-like phrase a person might type to find a similar image
  (encouraged to avoid copying the caption verbatim)
- **`unrelated_query`**: a different visual topic for negative / ranking signals

Post-export **repair** pass (local distractor bank + targeted LLM rewrites)
improved uniqueness of distractors and reduced caption-copy queries. Residual
soft flags remain in the QC report (preliminary release).

### 4. Quality control

Automated **text** QC (`audit_dataset.py`) flags empty fields, near-duplicate
captions, offline fallback fingerprints, generic distractor templates,
query≈caption overlap, and high-frequency collisions.

| QC metric (latest local audit) | Value |
|--------------------------------|------:|
| Rows soft-flagged | {q.get('num_flagged', 'n/a')} |
| % soft-flagged | {q.get('pct_flagged', 'n/a')} |
| Unique queries | {stats['unique_queries']:,} |
| Unique unrelated | {stats['unique_unrelated']:,} |
| Generic-template unrelated count | {stats['generic_unrelated_count']:,} |

**Not yet in this release:** full CLIP/SigLIP alignment filtering, human rating
of every row, or hard near-duplicate image dedup across sources. Treat as a
**strong training prior**, not a frozen leaderboard set.

## Intended uses

**Suitable for**

- Contrastive image–text pretraining / continued alignment (e.g. dual encoders)
- Training with **query-style** text (not only captions)
- Multi-positive setups (multiple captions + query per image)
- Domain-balanced experiments (satellite vs natural images)
- Retrieval evaluation prototypes (image↔text)

**Not suitable for (without further work)**

- Safety / toxicity benchmarking
- Geographic or demographic fairness claims
- Medical or high-stakes remote-sensing decisions
- Claiming SOTA retrieval numbers without an independent test split

## Out-of-scope uses

- Do not present this preliminary mix as “the” standard RS or COCO replacement
- Do not ignore upstream licenses when redistributing derivatives
- Do not use machine-generated queries as if they were human search logs

## Loading

### From the Hub

```python
from datasets import load_dataset

ds = load_dataset("{repo_id or DEFAULT_HF_REPO_HINT}", split="train")
print(ds)
row = ds[0]
print(row["id"], row["domain"], row["query"])
row["image"]  # PIL.Image.Image, {img}x{img}
print(row["captions"])
print(row["unrelated_query"])
```

Streaming (lower peak disk):

```python
ds = load_dataset("{repo_id or DEFAULT_HF_REPO_HINT}", split="train", streaming=True)
for row in ds.take(3):
    print(row["id"], row["query"])
```

### From a local export

```python
from datasets import load_dataset, load_from_disk

# Parquet shards (embedded images)
ds = load_dataset("parquet", data_files="data/train-*.parquet", split="train")

# Or Arrow export if present
# ds = load_from_disk("hf")

# Text-only index (no pixels) for fast filtering
import json
with open("metadata.jsonl") as f:
    meta = [json.loads(line) for line in f]
```

### With the TriSearch training stack

```python
from trisearch_dataset import load_curated_training_rows

rows = load_curated_training_rows("models/data/trisearch-v1", seed=42)
# each row: image, caption(s), related_query, unrelated_query, domain, ...
```

## Text length & style notes

| Field | Approx. length (chars) |
|-------|------------------------:|
| Caption mean | {stats['caption_char_len']['mean']} (p10={stats['caption_char_len']['p10']}, p90={stats['caption_char_len']['p90']}) |
| Query mean | {stats['query_char_len']['mean']} (p10={stats['query_char_len']['p10']}, p90={stats['query_char_len']['p90']}) |

Queries are intentionally **short and search-like**. Captions are fuller scene
descriptions. Unrelated queries are scene-level phrases (not single topic words
where repair succeeded).

## Splits

This preliminary release ships a single **`train`** split only.

There is **no official validation/test split** yet. If you report numbers:

1. Hold out your own random or domain-stratified subset, or
2. Wait for a later release with frozen eval ids

Recommended ad-hoc holdout:

```python
ds = ds.train_test_split(test_size=0.02, seed=42)
train, smoke_test = ds["train"], ds["test"]
```

## Data fields (detailed)

### `captions`

- Length ≥ 2 after diversity filtering
- Mean ≈ {stats['captions_per_image']['mean']} strings per image
- Should describe the **same** visual content with different focus
  (layout vs objects vs context)

### `query`

- Positive retrieval text for the image
- Prefer non-verbatim paraphrases of captions
- May still partially overlap caption tokens on a minority of rows (see QC)

### `unrelated_query`

- Distractor for contrastive / ranking losses
- Should **not** match the image content
- Post-repair uniqueness is much higher than raw LLM batch output; residual
  medium-frequency templates may remain under the soft threshold

### `domain` / `source`

- Use `domain` for sampling ratios (e.g. 50% satellite in Stage‑1)
- `source` traces upstream provenance for license and debugging

## Known limitations (preliminary)

1. **Machine-authored queries/distractors** — not real user search logs; style
   reflects the generator model and repair prompts.
2. **Soft QC residue** — a few percent of rows may still show high query–caption
   token overlap or mild query collisions.
3. **No multimodal alignment gate** in v0.1 — we do not drop rows by CLIP score yet.
4. **Image near-duplicates** across COCO/SkyScript are not exhaustively purged.
5. **English only.**
6. **Satellite geography / sensor metadata** are not included as structured fields.
7. **Parquet vs sidecar text** — Hub package is built from the curated export;
   always prefer the published parquet as the release snapshot.

## Ethical considerations

- **Upstream imagery** may depict people, places, and infrastructure. Follow the
  licenses and privacy expectations of COCO and SkyScript.
- **Remote-sensing** images can be sensitive in some jurisdictions; this set is
  for research embedding models, not operational ISR.
- **LLM rewrites** can introduce factual drift relative to the pixels; captions
  remain the primary grounded text, with queries as auxiliary retrieval text.

## Licensing

**Composite / other.** This packaging does not re-license upstream assets.

You must comply with:

| Component | Notes |
|-----------|--------|
| [MS-COCO](https://cocodataset.org/#termsofuse) / `bitmind/MS-COCO` images & annotations | COCO terms; annotations typically CC-BY 4.0; images from Flickr with varying original licenses |
| [SkyScript](https://github.com/wangzhecheng/SkyScript) | Follow SkyScript / source imagery terms for remote-sensing data |
| Machine-generated captions/queries in this packaging | Provided for research use with the dataset; no trademark license granted |
| TriSearch packaging metadata (`id`, `domain`, repair text) | Research use; attribution appreciated |

See `LICENSE` in the repository root of this dataset for the short composite notice.

If you are unsure whether your use is allowed, contact the upstream dataset
maintainers **and** do not treat this preliminary HF mirror as legal advice.

## Citation

If you use this dataset, please cite the upstream sources and note the TriSearch
preliminary packaging:

```bibtex
@misc{{trisearch-v1-preliminary,
  title  = {{TriSearch-v1: Preliminary Multimodal Image-Text Training Corpus}},
  author = {{TriSearch contributors}},
  year   = {{2026}},
  note   = {{Preliminary release {stats['dataset_version']}. Built on MS-COCO and SkyScript.}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id or DEFAULT_HF_REPO_HINT}}}}}
}}
```

Also cite MS-COCO and SkyScript as required by their authors:

```bibtex
@inproceedings{{lin2014coco,
  title     = {{Microsoft COCO: Common Objects in Context}},
  author    = {{Lin, Tsung-Yi and others}},
  booktitle = {{ECCV}},
  year      = {{2014}}
}}

@article{{skyscript,
  title  = {{SkyScript: A Large and Semantically Diverse Vision-Language Dataset for Remote Sensing}},
  author = {{Wang, Zhecheng and others}},
  year   = {{2023}},
  note   = {{See SkyScript project page / paper for the canonical citation.}}
}}
```

## Changelog

### `{stats['dataset_version']}` — preliminary

- Initial public packaging of {n:,} curated rows
- 50/50 general (COCO) / satellite (SkyScript)
- {img}×{img} images, multi-caption, query + unrelated_query
- Diversity filtering + LLM assist + post-hoc repair for distractors/queries
- Automated text QC report
- HF parquet layout for `load_dataset`

## Maintenance

- **Status:** preliminary / research
- **Contact:** open an issue or discussion on the Hub repo
- **Reproducibility:** generation and repair scripts live in the TriSearch code
  repository (`generate_datasets.py`, `audit_dataset.py`, `repair_dataset.py`,
  `publish_dataset.py`)

## Card metadata

```
format_version: {stats['format_version']}
dataset_version: {stats['dataset_version']}
num_rows: {n}
image_size: {img}
generated_on: {stats.get('generated_on')}
```
"""

    if include_examples and stats.get("examples"):
        body += "\n## Illustrative examples (text only)\n\n"
        body += "Random stratified samples from the export (images omitted in the card):\n\n"
        body += _examples_md(stats["examples"])
        body += "\n"

    return _yaml_front_matter(stats, repo_id=repo_id, pretty_name=pretty) + body


def write_dataset_card(
    dataset_dir: str | Path,
    *,
    repo_id: str | None = None,
    output_path: str | Path | None = None,
    also_write_stats_json: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Collect stats, render card, write README.md (and optional stats JSON)."""
    root = Path(dataset_dir)
    stats = collect_dataset_stats(root)
    card = render_dataset_card(stats, repo_id=repo_id)
    out = Path(output_path) if output_path else root / "README.md"
    out.write_text(card, encoding="utf-8")
    if also_write_stats_json:
        stats_path = root / "publication_stats.json"
        # Don't embed full path secrets-style absolute roots in public file optionally
        public_stats = dict(stats)
        public_stats.pop("root", None)
        stats_path.write_text(
            json.dumps(public_stats, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return out, stats


COMPOSITE_LICENSE_TEXT = """TriSearch-v1 preliminary dataset — composite license notice
================================================================

This dataset is a **packaging** of third-party images and annotations with
additional machine-generated text fields (captions rewrites, search queries,
unrelated queries) and metadata (ids, domain labels).

1. Images and original annotations remain subject to their **upstream** licenses
   and terms of use:

   - MS-COCO / bitmind/MS-COCO — see https://cocodataset.org/#termsofuse
     and the Hugging Face dataset card for bitmind/MS-COCO.
   - SkyScript — see the SkyScript project / paper and any terms attached to
     the imagery distribution you obtained.

2. You must not remove upstream attribution or copyright notices.

3. The TriSearch-specific packaging (selection, resize to square JPEG, domain
   tags, diversified captions, query / unrelated_query fields, quality repair)
   is provided for **research and non-commercial experimentation** unless you
   obtain a separate grant. No warranties.

4. This notice is not legal advice. If your use case is commercial,
   safety-critical, or redistribution-heavy, review upstream licenses carefully
   and seek counsel as needed.

5. THE DATASET IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""


def write_license_file(dataset_dir: str | Path) -> Path:
    path = Path(dataset_dir) / "LICENSE"
    path.write_text(COMPOSITE_LICENSE_TEXT, encoding="utf-8")
    return path
