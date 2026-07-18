#!/usr/bin/env python3
"""
Dataset loading and manipulation for TriSearch training and evaluation.

All loaders require real image–caption data from HuggingFace datasets, local
JSONL files, or on-disk image paths. There is no synthetic or placeholder data.
"""

from __future__ import annotations

import io
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from PIL import Image
from torch.utils.data import Dataset

# Stage 1 defaults (training_plan.md §2–3)
# Published curated corpus on the Hub (preferred). Local export is optional cache.
DEFAULT_TRISEARCH_HF_DATASET = "NuclearManD/trisearch-dataset-64k-v0.0.1"
DEFAULT_CURATED_DATASET_DIR = Path("models/data/trisearch-v1")
DEFAULT_SATELLITE_DATASET = "SkyScript"
DEFAULT_SATELLITE_SPLIT = "train"
# Parquet/streaming-native general captions (HuggingFaceM4/COCO uses a deprecated script).
DEFAULT_GENERAL_DATASET = "bitmind/MS-COCO"
DEFAULT_GENERAL_SPLIT = "train"
DEFAULT_GENERAL_CAPTION_COLUMN = "caption_0"

# Evaluation / indexing defaults — same curated corpus as training (not Flickr/COCO).
VERIFICATION_SAMPLE_COUNT = 4

CHATEARTHNET_HF_REPO = DEFAULT_SATELLITE_DATASET
CHATEARTHNET_RGB_ZIP = "s2_rgb_images.zip"
DEFAULT_CHATEARTHNET_CACHE_DIR = Path("models/data/ChatEarthNet")
DEFAULT_CHATEARTHNET_IMAGE_ROOT = DEFAULT_CHATEARTHNET_CACHE_DIR / "s2_rgb_images"

DEFAULT_OPENROUTER_CONFIG = Path("config.yml")
DEFAULT_QUERY_CACHE_PATH = Path("models/data/stage1_query_cache.jsonl")
QUERY_CACHE_CAPTION_KEY = "caption"
QUERY_CACHE_RELATED_KEY = "related_query"
QUERY_CACHE_UNRELATED_KEY = "unrelated_query"
OPENROUTER_QUERY_BATCH_SIZE = 8
OPENROUTER_QUERY_PARALLELISM = 32
OPENROUTER_MAX_ATTEMPTS = 4


def is_image_path_reference(value: Any) -> bool:
    """True when ``value`` is a filename/path string, not an embedded image."""
    return isinstance(value, str) and not Path(value).is_file()


def _find_png_directory(base: Path, sample_name: str) -> Path | None:
    """Return a directory under ``base`` that contains ``sample_name``."""
    direct = base / sample_name
    if direct.is_file():
        return base
    matches = list(base.rglob(sample_name))
    if matches:
        return matches[0].parent
    return None


def download_chatearthnet_rgb_images(
    cache_dir: Path | str | None = None,
) -> Path:
    """Download and extract ChatEarthNet RGB PNGs from the HF dataset repo."""
    import zipfile

    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir or DEFAULT_CHATEARTHNET_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / ".s2_rgb_images_extracted"
    if marker.is_file():
        root = _find_png_directory(cache_dir, "2815_4942_patch00.png")
        if root is not None:
            return root

    print(
        f"Downloading {CHATEARTHNET_RGB_ZIP} from {CHATEARTHNET_HF_REPO} "
        f"(~13 GB; one-time cache under {cache_dir}) ..."
    )
    zip_path = hf_hub_download(
        repo_id=CHATEARTHNET_HF_REPO,
        repo_type="dataset",
        filename=CHATEARTHNET_RGB_ZIP,
        local_dir=str(cache_dir),
    )
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(cache_dir)
    marker.write_text("ok\n", encoding="utf-8")

    root = _find_png_directory(cache_dir, "2815_4942_patch00.png")
    if root is None:
        raise RuntimeError(
            f"Extracted {CHATEARTHNET_RGB_ZIP} under {cache_dir} but could not "
            "find PNG files. Check the archive layout."
        )
    print(f"ChatEarthNet images ready at {root}")
    return root


def resolve_chatearthnet_image_root(
    rows: list[dict[str, Any]],
    *,
    explicit_root: str | None = None,
    download_if_missing: bool = False,
) -> str | None:
    """Resolve PNG directory for path-based ChatEarthNet caption rows."""
    if not rows or not is_image_path_reference(rows[0].get("image")):
        return explicit_root

    sample_name = Path(str(rows[0]["image"])).name
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    candidates.append(DEFAULT_CHATEARTHNET_IMAGE_ROOT)
    candidates.append(DEFAULT_CHATEARTHNET_CACHE_DIR)

    for base in candidates:
        found = _find_png_directory(base, sample_name)
        if found is not None:
            print(f"Using ChatEarthNet image root: {found}")
            return str(found)

    if download_if_missing:
        return str(download_chatearthnet_rgb_images())

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"ChatEarthNet rows reference PNG filenames (e.g. {sample_name!r}) but no "
        f"image files were found. Searched: {tried}. "
        f"Pass --satellite-image-root /path/to/pngs, or "
        f"--download-satellite-images to fetch {CHATEARTHNET_RGB_ZIP} from HuggingFace."
    )


def validate_image_rows(
    rows: list[dict[str, Any]],
    image_root: str | None,
    image_column: str = "image",
    *,
    sample_count: int = 8,
    label: str = "training",
) -> None:
    """Fail fast when image files cannot be loaded (before model init)."""
    if not rows:
        raise ValueError(f"No {label} rows to validate.")
    indices = list(range(min(sample_count, len(rows))))
    if len(rows) > sample_count:
        rng = random.Random(0)
        indices = sorted(rng.sample(range(len(rows)), sample_count))

    failures: list[str] = []
    for idx in indices:
        row = rows[idx]
        try:
            load_pil_image(row[image_column], image_root=image_root)
        except (FileNotFoundError, TypeError, OSError) as exc:
            failures.append(f"  row {idx}: {exc}")
    if failures:
        root_hint = f" (image_root={image_root!r})" if image_root else ""
        raise FileNotFoundError(
            f"Cannot load {len(failures)}/{len(indices)} sample {label} images"
            f"{root_hint}:\n" + "\n".join(failures)
        )
    print(
        f"Validated {len(indices)} sample {label} images"
        f"{f' under {image_root}' if image_root else ''}."
    )


def load_pil_image(value: Any, image_root: str | None = None) -> Image.Image:
    """Load a PIL RGB image from a path, bytes, HF dict, or in-memory image."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "bytes" in value:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        path = Path(value)
        if not path.is_file() and image_root:
            path = Path(image_root) / value
        if path.is_file():
            return Image.open(path).convert("RGB")
        raise FileNotFoundError(f"Image not found: {value}")
    raise TypeError(f"Unsupported image value type: {type(value)}")


def normalize_training_text(value: Any) -> str:
    """Canonical text form for all training strings: strip + lowercase.

    Applied at load/stream time in this module so demos/trainers need no
    special casing. Empty after strip stays empty.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip().lower()


def pick_caption(row: dict[str, Any], caption_column: str) -> str:
    return normalize_training_text(row.get(caption_column, ""))


def caption_from_row(row: dict[str, Any], caption_column: str | None) -> str:
    if caption_column:
        return pick_caption(row, caption_column)
    for key in ("caption", "caption_0", "text", "sentence", "sentences"):
        if key in row and row[key]:
            return normalize_training_text(row[key])
    return ""


@dataclass
class DataSourceConfig:
    dataset: str
    split: str = "train"
    image_column: str = "image"
    caption_column: str = "caption"
    image_root: str | None = None
    max_samples: int | None = None


def stream_hf_rows(
    dataset: str,
    split: str = "train",
    *,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load rows via HF streaming (no dataset scripts, no ``trust_remote_code``)."""
    from datasets import load_dataset

    print(
        f"Loading dataset {dataset!r} (split={split}, streaming"
        f"{f', max={max_samples}' if max_samples is not None else ''}) ..."
    )
    ds = load_dataset(dataset, split=split, streaming=True)
    if max_samples is not None:
        rows = reservoir_sample_stream(iter(ds), count=max_samples, seed=seed)
    else:
        rows = [dict(row) for row in ds]
    print(f"  -> {len(rows):,} rows")
    return rows


def load_hf_rows(config: DataSourceConfig, *, seed: int = 42) -> list[dict[str, Any]]:
    return stream_hf_rows(
        config.dataset,
        config.split,
        max_samples=config.max_samples,
        seed=seed,
    )


def load_jsonl_rows(path: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    print(f"Loaded {len(rows):,} rows from {path}")
    return rows


def normalize_rows(
    rows: list[dict[str, Any]],
    image_column: str,
    caption_column: str,
) -> list[dict[str, Any]]:
    """Map heterogeneous HF rows to unified ``image`` + ``caption`` keys."""
    normalized: list[dict[str, Any]] = []
    for row in rows:
        caption = pick_caption(row, caption_column)
        if not caption:
            raise ValueError(
                f"Row missing caption in column {caption_column!r}: "
                f"keys={sorted(row)}"
            )
        normalized.append({
            "image": row[image_column],
            "caption": caption,
        })
    return normalized


def build_mixed_dataset(
    satellite_rows: list[dict[str, Any]],
    general_rows: list[dict[str, Any]],
    satellite_fraction: float,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Mix satellite and general rows per the stage-1 curriculum ratio."""
    if not satellite_rows and not general_rows:
        raise ValueError("No training rows available.")
    if not satellite_rows or not general_rows:
        return satellite_rows or general_rows

    total = min(len(satellite_rows), len(general_rows)) * 2
    n_sat = int(total * satellite_fraction)
    n_gen = total - n_sat
    rng = random.Random(seed)
    sat = rng.sample(satellite_rows, min(n_sat, len(satellite_rows)))
    gen = rng.sample(general_rows, min(n_gen, len(general_rows)))

    mixed: list[dict[str, Any]] = []
    sat_i = gen_i = 0
    sat_every = max(1, round(1.0 / satellite_fraction)) if satellite_fraction > 0 else 10**9
    while len(mixed) < total and (sat_i < len(sat) or gen_i < len(gen)):
        if sat_i < len(sat) and (len(mixed) % sat_every == 0 or gen_i >= len(gen)):
            mixed.append(sat[sat_i])
            sat_i += 1
        elif gen_i < len(gen):
            mixed.append(gen[gen_i])
            gen_i += 1
        else:
            break
    rng.shuffle(mixed)
    print(
        f"Mixed dataset: {len(mixed):,} rows "
        f"({satellite_fraction:.0%} satellite target)"
    )
    return mixed


def curated_dataset_available(path: str | Path | None = None) -> bool:
    """True if a *local* curated export exists (parquet / hf / metadata+images)."""
    root = Path(path or DEFAULT_CURATED_DATASET_DIR)
    if not root.is_dir():
        return False
    if (root / "hf").is_dir():
        return True
    if any((root / "data").glob("*.parquet")):
        return True
    if (root / "metadata.jsonl").is_file() and (root / "images").is_dir():
        return True
    return False


def _normalize_trisearch_fields(rec: dict[str, Any]) -> dict[str, Any]:
    """Text fields only — **never** decode the image here (RAM rule).

    All human-language fields are lowercased for training consistency.
    """
    raw_caps = list(rec.get("captions") or [])
    if not raw_caps and rec.get("caption"):
        raw_caps = [rec["caption"]]
    captions = [normalize_training_text(c) for c in raw_caps if normalize_training_text(c)]
    if not captions:
        raise ValueError(f"row {rec.get('id')!r} has no captions")
    primary = captions[0]
    related = normalize_training_text(
        rec.get(QUERY_CACHE_RELATED_KEY) or rec.get("query") or ""
    )
    if not related and len(captions) > 1:
        related = captions[1]
    unrelated = normalize_training_text(
        rec.get(QUERY_CACHE_UNRELATED_KEY) or rec.get("unrelated_query") or ""
    )
    return {
        # Keep raw image handle (path / HF Image / bytes dict); decode in getitem.
        "image": rec.get("image"),
        "caption": primary,
        "captions": captions,
        "domain": str(rec.get("domain") or "general"),
        "source": str(rec.get("source") or "trisearch"),
        "id": str(rec.get("id") or ""),
        QUERY_CACHE_RELATED_KEY: related,
        QUERY_CACHE_UNRELATED_KEY: unrelated,
    }


def _domain_balanced_indices(
    domains: Sequence[str],
    *,
    max_samples: int | None,
    seed: int,
    satellite_fraction: float | None,
) -> list[int] | None:
    """Return subset indices or None to keep all. Never touches images."""
    n = len(domains)
    if max_samples is None and (
        satellite_fraction is None or not (0.0 < float(satellite_fraction) < 1.0)
    ):
        return None
    rng = random.Random(seed)
    if satellite_fraction is not None and 0.0 < satellite_fraction < 1.0:
        sat = [i for i, d in enumerate(domains) if str(d) == "satellite"]
        gen = [i for i, d in enumerate(domains) if str(d) == "general"]
        if sat and gen:
            if max_samples is not None:
                n_sat = int(round(max_samples * satellite_fraction))
                n_gen = max_samples - n_sat
            else:
                n_sat, n_gen = len(sat), len(gen)
            rng.shuffle(sat)
            rng.shuffle(gen)
            out = sat[:n_sat] + gen[:n_gen]
            rng.shuffle(out)
            return out
    if max_samples is not None and max_samples < n:
        return rng.sample(range(n), max_samples)
    return None


class TriSearchMapDataset(Dataset):
    """Lazy curated TriSearch view: HF Arrow/parquet mmap or local sidecars.

    **Never** holds a full list of decoded PIL images. ``__getitem__`` decodes
    one example. Relies on the Hugging Face datasets cache on disk.
    """

    def __init__(
        self,
        source: Any,
        *,
        indices: list[int] | None = None,
        image_root: str | None = None,
        backend: str = "hf",
        queries_ready: bool = True,
        label: str = "trisearch",
    ):
        self._source = source
        self._indices = indices
        self.image_root = image_root
        self.backend = backend  # "hf" | "local_lazy" | "list"
        self.queries_ready = queries_ready
        self.label = label

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self._source)

    def _raw_at(self, i: int) -> dict[str, Any]:
        j = self._indices[i] if self._indices is not None else i
        if self.backend == "local_lazy":
            # LazyTriSearchDataset
            return self._source[j]
        if self.backend == "list":
            return dict(self._source[j])
        # HF datasets.Dataset row
        row = self._source[int(j)]
        return dict(row)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = _normalize_trisearch_fields(self._raw_at(idx))
        # Decode single image only now.
        rec["image"] = load_pil_image(rec["image"], self.image_root)
        if isinstance(rec["image"], Image.Image):
            rec["image"] = rec["image"].convert("RGB").copy()
        return rec

    def meta(self, idx: int) -> dict[str, Any]:
        """Text fields only (no image decode)."""
        rec = _normalize_trisearch_fields(self._raw_at(idx))
        rec.pop("image", None)
        return rec


def open_trisearch_map_dataset(
    *,
    hf_dataset: str | None = DEFAULT_TRISEARCH_HF_DATASET,
    dataset_dir: str | Path | None = None,
    prefer_local: bool = False,
    split: str = "train",
    max_samples: int | None = None,
    seed: int = 42,
    satellite_fraction: float | None = None,
    revision: str | None = None,
) -> TriSearchMapDataset:
    """Open curated TriSearch as a **lazy** map dataset (Hub HF cache or local).

    Full splits stay on disk. Optional ``max_samples`` only selects indices.
    """
    from datasets import load_dataset

    if split not in ("train", "test", "all"):
        raise ValueError(f"split must be train|test|all, got {split!r}")

    root = Path(dataset_dir or DEFAULT_CURATED_DATASET_DIR)
    if prefer_local and curated_dataset_available(root):
        from trisearch_data_format import apply_official_splits, open_lazy_dataset
        from trisearch_quality import load_metadata_rows

        lazy = open_lazy_dataset(root, max_samples=None, image_cache_size=0)
        indices: list[int] | None = None
        domains: list[str]
        try:
            meta_rows = load_metadata_rows(root)
            apply_official_splits(meta_rows, force=False)
            if len(meta_rows) == len(lazy) and split in ("train", "test"):
                indices = [
                    i for i, r in enumerate(meta_rows) if r.get("split") == split
                ]
                domains = [str(meta_rows[i]["domain"]) for i in indices]
            elif len(meta_rows) == len(lazy):
                domains = [str(r.get("domain", "general")) for r in meta_rows]
            else:
                domains = [
                    str(lazy.meta(i).get("domain", "general"))
                    for i in range(len(lazy))
                ]
        except Exception:
            domains = [
                str(lazy.meta(i).get("domain", "general")) for i in range(len(lazy))
            ]
        sub = _domain_balanced_indices(
            domains,
            max_samples=max_samples,
            seed=seed,
            satellite_fraction=satellite_fraction,
        )
        if indices is not None and sub is not None:
            indices = [indices[j] for j in sub]
        elif sub is not None:
            indices = sub

        print(
            f"Opened local curated map dataset {root} "
            f"(n={len(indices) if indices is not None else len(lazy)}, split={split})",
            flush=True,
        )
        return TriSearchMapDataset(
            lazy,
            indices=indices,
            backend="local_lazy",
            queries_ready=True,
            label=str(root),
        )

    if not hf_dataset:
        raise FileNotFoundError(
            f"No local curated data at {root} and no hf_dataset id. "
            f"Default Hub id: {DEFAULT_TRISEARCH_HF_DATASET}"
        )

    print(
        f"Opening TriSearch Hub map dataset {hf_dataset!r} "
        f"(split={split}, max_samples={max_samples}) — lazy, HF cache only",
        flush=True,
    )
    if split == "all":
        from datasets import concatenate_datasets

        ds = concatenate_datasets(
            [
                load_dataset(hf_dataset, split=sp, revision=revision)
                for sp in ("train", "test")
            ]
        )
    else:
        ds = load_dataset(hf_dataset, split=split, revision=revision)

    # Domain column only (strings) — not images.
    domains = list(ds["domain"]) if "domain" in ds.column_names else ["general"] * len(ds)
    indices = _domain_balanced_indices(
        domains,
        max_samples=max_samples,
        seed=seed,
        satellite_fraction=satellite_fraction,
    )
    n = len(indices) if indices is not None else len(ds)
    print(
        f"  map dataset ready: {n:,} rows (images decode on access; disk=HF cache)",
        flush=True,
    )
    return TriSearchMapDataset(
        ds,
        indices=indices,
        backend="hf",
        queries_ready=True,
        label=hf_dataset,
    )


def load_trisearch_hub_rows(
    dataset_id: str = DEFAULT_TRISEARCH_HF_DATASET,
    *,
    split: str = "train",
    max_samples: int | None = None,
    seed: int = 42,
    satellite_fraction: float | None = None,
    revision: str | None = None,
    materialize: bool | None = None,
) -> TriSearchMapDataset | list[dict[str, Any]]:
    """Open Hub curated data **lazily** (default).

    Set ``materialize=True`` only for tiny samples (demo/tests). Full splits
    always return :class:`TriSearchMapDataset`.
    """
    ds = open_trisearch_map_dataset(
        hf_dataset=dataset_id,
        prefer_local=False,
        split=split,
        max_samples=max_samples,
        seed=seed,
        satellite_fraction=satellite_fraction,
        revision=revision,
    )
    # Materialize only small samples explicitly requested.
    if materialize is None:
        materialize = max_samples is not None and max_samples <= 512
    if materialize:
        if max_samples is None or max_samples > 512:
            raise RuntimeError(
                "Refusing to materialize a large curated split into a Python list "
                "(never load multi-GB datasets into RAM). Use the map dataset."
            )
        return [ds[i] for i in range(len(ds))]
    return ds


def load_curated_training_rows(
    dataset_dir: str | Path | None = None,
    *,
    hf_dataset: str | None = DEFAULT_TRISEARCH_HF_DATASET,
    max_samples: int | None = None,
    seed: int = 42,
    satellite_fraction: float | None = None,
    split: str = "train",
    prefer_local: bool = False,
    revision: str | None = None,
    materialize: bool | None = None,
) -> TriSearchMapDataset | list[dict[str, Any]]:
    """Open curated TriSearch for training (lazy map dataset by default).

    Preference: Hub ``hf_dataset`` unless ``prefer_local`` and a local export exist.
    """
    ds = open_trisearch_map_dataset(
        hf_dataset=hf_dataset,
        dataset_dir=dataset_dir,
        prefer_local=prefer_local,
        split=split,
        max_samples=max_samples,
        seed=seed,
        satellite_fraction=satellite_fraction,
        revision=revision,
    )
    if materialize is None:
        materialize = max_samples is not None and max_samples <= 512
    if materialize:
        if max_samples is None or max_samples > 512:
            raise RuntimeError(
                "Refusing to materialize a large curated split into RAM. "
                "Use the returned map dataset / ImageCaptionDataset directly."
            )
        return [ds[i] for i in range(len(ds))]
    return ds


def load_stage1_training_rows(
    *,
    data_jsonl: str | None = None,
    curated_dataset_dir: str | None = None,
    hf_dataset: str | None = DEFAULT_TRISEARCH_HF_DATASET,
    prefer_curated: bool = True,
    prefer_local_curated: bool = False,
    curated_split: str = "train",
    image_root: str | None = None,
    satellite_dataset: str = DEFAULT_SATELLITE_DATASET,
    satellite_split: str = DEFAULT_SATELLITE_SPLIT,
    satellite_image_column: str = "image",
    satellite_caption_column: str = "caption",
    satellite_image_root: str | None = None,
    general_dataset: str = DEFAULT_GENERAL_DATASET,
    general_split: str = DEFAULT_GENERAL_SPLIT,
    general_image_column: str = "image",
    general_caption_column: str = DEFAULT_GENERAL_CAPTION_COLUMN,
    satellite_fraction: float = 0.5,
    max_satellite_samples: int | None = None,
    max_general_samples: int | None = None,
    seed: int = 42,
    download_satellite_images: bool = False,
) -> tuple[list[dict[str, Any]], str, str, str | None]:
    """Return (rows, image_column, caption_column, image_root) for stage-1 training.

    Preference order:
      1. ``--data-jsonl`` local JSONL
      2. Curated TriSearch (local export if present, else Hub ``hf_dataset``)
      3. Legacy HF satellite + general mix
    """
    if data_jsonl:
        rows = load_jsonl_rows(data_jsonl, max_samples=None)
        effective_root = image_root or satellite_image_root
        validate_image_rows(rows, effective_root, label="JSONL")
        return rows, "image", "caption", effective_root

    curated_path = Path(curated_dataset_dir or DEFAULT_CURATED_DATASET_DIR)
    if prefer_curated:
        max_total = None
        if max_satellite_samples is not None or max_general_samples is not None:
            max_total = (max_satellite_samples or 0) + (max_general_samples or 0)
            if max_total <= 0:
                max_total = None
        try:
            rows = open_trisearch_map_dataset(
                hf_dataset=hf_dataset,
                dataset_dir=curated_path,
                prefer_local=prefer_local_curated,
                split=curated_split,
                max_samples=max_total,
                seed=seed,
                satellite_fraction=satellite_fraction,
            )
            return rows, "image", "caption", None
        except Exception as exc:  # noqa: BLE001
            print(
                f"Curated TriSearch load failed ({exc}); "
                f"falling back to legacy HF mix.",
                flush=True,
            )

    explicit_sat_root = satellite_image_root or image_root
    # Legacy ChatEarthNet path only when that dataset is explicitly requested.
    satellite_rows = normalize_rows(
        load_hf_rows(
            DataSourceConfig(
                dataset=satellite_dataset,
                split=satellite_split,
                image_column=satellite_image_column,
                caption_column=satellite_caption_column,
                image_root=explicit_sat_root,
                max_samples=max_satellite_samples,
            ),
            seed=seed,
        ),
        image_column=satellite_image_column,
        caption_column=satellite_caption_column,
    )
    if (
        satellite_dataset == "JessicaYuan/ChatEarthNet"
        and satellite_rows
    ):
        explicit_sat_root = resolve_chatearthnet_image_root(
            satellite_rows,
            explicit_root=explicit_sat_root,
            download_if_missing=download_satellite_images,
        )
        validate_image_rows(
            satellite_rows,
            explicit_sat_root,
            label="satellite",
        )
    else:
        validate_image_rows(satellite_rows, explicit_sat_root, label="satellite")

    general_rows = normalize_rows(
        load_hf_rows(
            DataSourceConfig(
                dataset=general_dataset,
                split=general_split,
                image_column=general_image_column,
                caption_column=general_caption_column,
                max_samples=max_general_samples,
            ),
            seed=seed,
        ),
        image_column=general_image_column,
        caption_column=general_caption_column,
    )
    validate_image_rows(general_rows, image_root=None, label="general")

    mixed = build_mixed_dataset(
        satellite_rows, general_rows, satellite_fraction, seed=seed
    )
    return mixed, "image", "caption", explicit_sat_root


def reservoir_sample_stream(
    dataset_iter: Iterator[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    for i, row in enumerate(dataset_iter):
        row = dict(row)
        if i < count:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < count:
                reservoir[j] = row
    return reservoir


def load_dataset_samples(
    dataset: str,
    split: str,
    count: int,
    seed: int,
    image_column: str = "image",
    caption_column: str | None = None,
    image_root: str | None = None,
) -> list[dict[str, Any]]:
    """Sample ``count`` real image–caption pairs from a HuggingFace dataset."""
    from datasets import load_dataset

    print(f"Sampling {count:,} rows from {dataset!r} (split={split}) ...")
    ds = load_dataset(dataset, split=split, streaming=True)
    raw_rows = reservoir_sample_stream(iter(ds), count=count, seed=seed)
    print(f"  -> collected {len(raw_rows):,} rows")

    samples: list[dict[str, Any]] = []
    skipped = 0
    for row in raw_rows:
        try:
            image = load_pil_image(row[image_column], image_root=image_root)
            caption = caption_from_row(row, caption_column)
            if not caption:
                raise ValueError(f"row has no caption (column={caption_column!r})")
            samples.append({"image": image, "caption": caption})
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            skipped += 1
            if skipped <= 3:
                print(f"  skipped row: {exc}")
    if skipped:
        print(f"  skipped {skipped:,} rows without loadable image/caption pairs")
    if not samples:
        raise RuntimeError(
            f"No images could be loaded from {dataset!r}. "
            "For ChatEarthNet, pass --image-root to the directory of PNG files."
        )
    return samples


def load_stage1_demo_samples(
    count: int,
    seed: int = 42,
    *,
    curated_dataset_dir: str | Path | None = None,
    hf_dataset: str | None = DEFAULT_TRISEARCH_HF_DATASET,
    curated_split: str = "train",
    prefer_local_curated: bool = False,
    satellite_fraction: float | None = 0.5,
) -> TriSearchMapDataset:
    """Open a **lazy** TriSearch map over ``count`` indices (no full PIL list).

    Prefer :func:`open_trisearch_map_dataset` / demo ``build_from_map`` — this
    helper remains for callers that want a sized map sample.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    return open_trisearch_map_dataset(
        hf_dataset=hf_dataset,
        dataset_dir=curated_dataset_dir,
        prefer_local=prefer_local_curated,
        split=curated_split,
        max_samples=count,
        seed=seed,
        satellite_fraction=satellite_fraction,
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except ImportError:
        data: dict[str, Any] = {}
        section: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":") and ":" not in line[:-1]:
                section = line[:-1]
                data.setdefault(section, {})
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if section:
                data[section][key] = value
            else:
                data[key] = value
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


_OPENROUTER_CONFIG_CACHE: dict[str, dict[str, str]] = {}


def load_openrouter_config(config_path: str | Path = DEFAULT_OPENROUTER_CONFIG) -> dict[str, str]:
    """Load OpenRouter API settings from ``config.yml`` (cached per path)."""
    path = Path(config_path).resolve()
    key = str(path)
    cached = _OPENROUTER_CONFIG_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    if not path.is_file():
        raise FileNotFoundError(
            f"OpenRouter config not found at {path}. "
            "Create config.yml with openrouter.api_key and openrouter.model."
        )
    data = _load_yaml_mapping(path)
    section = data.get("openrouter", data)
    if not isinstance(section, dict):
        raise ValueError(f"Missing openrouter section in {path}")
    api_key = str(section.get("api_key", "")).strip()
    model = str(section.get("model", "")).strip()
    if not api_key or not model:
        raise ValueError(
            f"openrouter.api_key and openrouter.model are required in {path}"
        )
    result = {"api_key": api_key, "model": model}
    _OPENROUTER_CONFIG_CACHE[key] = result
    return dict(result)


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text.strip()
    for part in text.split("```"):
        chunk = part.strip()
        if chunk.startswith("json"):
            chunk = chunk[4:].strip()
        if chunk.startswith("{") or chunk.startswith("["):
            return chunk
    return text.strip()


def _extract_json_value(text: str) -> Any:
    text = _strip_code_fences(text)
    decoder = json.JSONDecoder()
    for opener in ("[", "{"):
        start = 0
        while start < len(text):
            idx = text.find(opener, start)
            if idx < 0:
                break
            try:
                value, _end = decoder.raw_decode(text[idx:])
                return value
            except json.JSONDecodeError:
                start = idx + 1
    match = re.search(
        r'\{[^{}]*"related_query"\s*:\s*"[^"]*"[^{}]*"unrelated_query"\s*:\s*"[^"]*"[^{}]*\}',
        text,
        flags=re.DOTALL,
    )
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No parseable JSON in model response: {text[:300]!r}")


def _normalize_query_entry(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object, got {type(payload)!r}")
    related = normalize_training_text(payload.get("related_query", ""))
    unrelated = normalize_training_text(payload.get("unrelated_query", ""))
    if not related or not unrelated:
        raise ValueError(f"Missing query fields in {payload!r}")
    return {
        QUERY_CACHE_RELATED_KEY: related,
        QUERY_CACHE_UNRELATED_KEY: unrelated,
    }


def _parse_query_json(content: str) -> dict[str, str]:
    payload = _extract_json_value(content)
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(
                f"Expected one query object, got array of length {len(payload)}"
            )
        payload = payload[0]
    return _normalize_query_entry(payload)


def _parse_query_batch_json(content: str, expected_count: int) -> list[dict[str, str]]:
    payload = _extract_json_value(content)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array, got {type(payload)!r}")
    if len(payload) != expected_count:
        raise ValueError(
            f"Expected {expected_count} query objects, got {len(payload)}"
        )
    return [_normalize_query_entry(item) for item in payload]


def _openrouter_chat_completion(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float = 120.0,
) -> str:
    import urllib.error
    import urllib.request

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter HTTP {exc.code}: {detail[:500]}"
        ) from exc
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {payload!r}")
    return str(choices[0].get("message", {}).get("content", "")).strip()


QUERY_OBJECT_PROMPT = (
    '{"thinking": "...", "synonyms": "...", "related_query": "...", "unrelated_query": "..."}\n'
    "Try to come up with queries a human would use in a google search for similar images.\n"
    "thinking: thoughts on what the caption is fundamentally describing.\n"
    "synonyms: words not in the caption which describe the same thing.\n"
    "related_query: what search someone would type to find an image matching the caption.  DO NOT USE THE SAME WORDS AS THE ORIGINAL CAPTION.  Use terms that are things someone could be interested in searching for.\n"
    "unrelated_query: a completely different topic; must not relate to the caption.  Must be a search looking for images, cannot be non-image topic.\n"
)

CAPTION_DIVERSIFY_PROMPT = (
    '{"captions": ["...", "...", "..."]}\n'
    "Write 2 or 3 DISTINCT English captions for the SAME image, based on the source text.\n"
    "Rules:\n"
    "- Captions must be meaningfully different (not synonym swaps of one sentence).\n"
    "- Vary focus: (1) overall scene, (2) notable objects/vehicles/structures, "
    "(3) layout/geometry/environment if relevant.\n"
    "- Do NOT reuse the same skeleton with only preposition/casing changes "
    "(e.g. 'planes parked in an airport' vs 'planes parked at an airport' is invalid).\n"
    "- Prefer concrete visual wording; keep each under 20 words.\n"
    "- For satellite/aerial scenes, use overhead/remote-sensing language when natural.\n"
)


def _parse_caption_list(content: str, *, min_count: int = 2) -> list[str]:
    from trisearch_data_format import normalize_captions

    payload = _extract_json_value(content)
    if isinstance(payload, dict):
        caps = payload.get("captions", payload.get("caption"))
    else:
        caps = payload
    if isinstance(caps, str):
        caps = [caps]
    if not isinstance(caps, list):
        raise ValueError(f"Expected captions list, got {type(caps)!r}")
    return normalize_captions(caps, min_count=min_count)


def openrouter_diversify_captions(
    source_captions: list[str],
    *,
    api_key: str,
    model: str,
    domain: str = "general",
    timeout: float = 45.0,
    max_attempts: int = OPENROUTER_MAX_ATTEMPTS,
    min_count: int = 2,
) -> list[str]:
    """Rewrite near-duplicate source captions into 2–3 genuinely varied ones."""
    joined = " | ".join(str(c).strip() for c in source_captions if str(c).strip())
    prompt = (
        "You help build a multimodal training dataset.\n"
        "Return ONLY one JSON object and nothing else:\n"
        f"{CAPTION_DIVERSIFY_PROMPT}"
        f"Domain: {domain}\n"
        f"Source text: {joined}\n"
    )
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            content = _openrouter_chat_completion(
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_tokens=220,
                timeout=timeout,
            )
            return _parse_caption_list(content, min_count=min_count)
        except (ValueError, json.JSONDecodeError, RuntimeError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(
        f"Failed to diversify captions for {joined[:100]!r} "
        f"after {max_attempts} attempts"
    ) from last_error


def openrouter_generate_queries(
    caption: str,
    *,
    api_key: str,
    model: str,
    timeout: float = 120.0,
    max_attempts: int = OPENROUTER_MAX_ATTEMPTS,
) -> dict[str, str]:
    """Ask OpenRouter for a related search query and an unrelated distractor."""
    prompt = (
        "You help train a text embedding model.\n"
        "Return ONLY one JSON object and nothing else:\n"
        f"{QUERY_OBJECT_PROMPT}"
        f"Caption: {caption}"
    )
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            content = _openrouter_chat_completion(
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_tokens=1024,
                timeout=timeout,
            )
            return _parse_query_json(content)
        except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(
        f"Failed to generate queries for caption {caption[:80]!r} "
        f"after {max_attempts} attempts"
    ) from last_error


def openrouter_repair_related_query(
    captions: list[str],
    *,
    api_key: str,
    model: str,
    domain: str = "general",
    bad_query: str = "",
    timeout: float = 45.0,
    max_attempts: int = OPENROUTER_MAX_ATTEMPTS,
) -> str:
    """Rewrite only the related search query (cheap single-field repair)."""
    joined = " | ".join(str(c).strip() for c in captions if str(c).strip())
    prompt = (
        "You help clean a multimodal retrieval training set.\n"
        "Return ONLY one JSON object: {\"related_query\": \"...\"}\n"
        "Write a short image-search query (3–10 words) a human would type.\n"
        "Rules:\n"
        "- MUST use different wording than the captions (synonyms / paraphrase).\n"
        "- At most 2 content words may overlap with any single caption.\n"
        "- No leading 'image of' / 'photo of' / 'picture of'.\n"
        "- Prefer concrete nouns and scene cues; no full sentences.\n"
        f"- Domain: {domain} "
        f"({'use aerial/overhead language when natural' if domain == 'satellite' else 'ground-level photo'}).\n"
        f"Captions: {joined}\n"
        f"Bad query to replace: {bad_query}\n"
    )
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            content = _openrouter_chat_completion(
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_tokens=80,
                timeout=timeout,
            )
            payload = _extract_json_value(content)
            if isinstance(payload, dict):
                q = str(
                    payload.get("related_query")
                    or payload.get("query")
                    or ""
                ).strip()
            elif isinstance(payload, str):
                q = payload.strip()
            else:
                raise ValueError(f"Unexpected payload type {type(payload)}")
            if len(q) < 4:
                raise ValueError(f"query too short: {q!r}")
            return q.rstrip(".")
        except (ValueError, json.JSONDecodeError, RuntimeError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(
        f"Failed to repair query for {joined[:80]!r} after {max_attempts} attempts"
    ) from last_error


def openrouter_generate_queries_batch(
    captions: list[str],
    *,
    api_key: str,
    model: str,
    timeout: float = 180.0,
    max_attempts: int = OPENROUTER_MAX_ATTEMPTS,
) -> list[dict[str, str]]:
    """Generate related/unrelated queries for multiple captions in one call."""
    if not captions:
        return []
    if len(captions) == 1:
        return [openrouter_generate_queries(
            captions[0], api_key=api_key, model=model, timeout=timeout
        )]

    numbered = "\n".join(f"{i + 1}. {caption}" for i, caption in enumerate(captions))
    prompt = (
        "You help train a text embedding model.\n"
        f"Return ONLY a JSON array of exactly {len(captions)} objects and nothing else.\n"
        "Each object must be:\n"
        f"{QUERY_OBJECT_PROMPT}"
        f"Captions:\n{numbered}"
    )
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            content = _openrouter_chat_completion(
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_tokens=32768,
                timeout=timeout,
            )
            return _parse_query_batch_json(content, len(captions))
        except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(
        f"Failed batch query generation for {len(captions)} captions "
        f"after {max_attempts} attempts"
    ) from last_error


def load_query_cache(path: str | Path) -> dict[str, dict[str, str]]:
    cache_path = Path(path)
    cache: dict[str, dict[str, str]] = {}
    if not cache_path.is_file():
        return cache
    with open(cache_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            caption = str(row.get(QUERY_CACHE_CAPTION_KEY, "")).strip()
            if not caption:
                continue
            cache[caption] = {
                QUERY_CACHE_RELATED_KEY: str(
                    row.get(QUERY_CACHE_RELATED_KEY, "")
                ).strip(),
                QUERY_CACHE_UNRELATED_KEY: str(
                    row.get(QUERY_CACHE_UNRELATED_KEY, "")
                ).strip(),
            }
    return cache


def save_query_cache(path: str | Path, cache: dict[str, dict[str, str]]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        for caption in sorted(cache):
            entry = cache[caption]
            fh.write(json.dumps({
                QUERY_CACHE_CAPTION_KEY: caption,
                QUERY_CACHE_RELATED_KEY: entry[QUERY_CACHE_RELATED_KEY],
                QUERY_CACHE_UNRELATED_KEY: entry[QUERY_CACHE_UNRELATED_KEY],
            }, ensure_ascii=False) + "\n")


def _generate_query_batch_with_fallback(
    captions: list[str],
    *,
    api_key: str,
    model: str,
) -> list[dict[str, str]]:
    try:
        return openrouter_generate_queries_batch(
            captions,
            api_key=api_key,
            model=model,
        )
    except RuntimeError:
        return [
            openrouter_generate_queries(
                caption,
                api_key=api_key,
                model=model,
            )
            for caption in captions
        ]


def append_query_cache_entry(
    path: str | Path,
    caption: str,
    entry: dict[str, str],
) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            QUERY_CACHE_CAPTION_KEY: caption,
            QUERY_CACHE_RELATED_KEY: entry[QUERY_CACHE_RELATED_KEY],
            QUERY_CACHE_UNRELATED_KEY: entry[QUERY_CACHE_UNRELATED_KEY],
        }, ensure_ascii=False) + "\n")


def enrich_rows_with_text_queries(
    rows: list[dict[str, Any]] | TriSearchMapDataset | Dataset,
    *,
    config_path: str | Path = DEFAULT_OPENROUTER_CONFIG,
    cache_path: str | Path = DEFAULT_QUERY_CACHE_PATH,
    max_new_queries: int | None = None,
    skip_generation: bool = False,
    caption_column: str = "caption",
    query_batch_size: int = OPENROUTER_QUERY_BATCH_SIZE,
    query_parallelism: int = OPENROUTER_QUERY_PARALLELISM,
) -> list[dict[str, Any]] | TriSearchMapDataset | Dataset:
    """Attach related/unrelated search queries (cached on disk).

    Curated map datasets already ship queries — returned unchanged (no full scan
    of images). List rows without queries are enriched via OpenRouter/cache.
    """
    if isinstance(rows, TriSearchMapDataset) and rows.queries_ready:
        print(
            f"All {len(rows):,} curated rows already have text queries "
            f"({rows.label}); skipping OpenRouter (lazy map dataset)."
        )
        return rows

    if not isinstance(rows, list):
        # Unknown map-like source without query guarantee — refuse full materialize.
        raise TypeError(
            "enrich_rows_with_text_queries needs a list of rows or a "
            "TriSearchMapDataset with queries_ready=True"
        )

    already: list[dict[str, Any]] = []
    need: list[dict[str, Any]] = []
    for row in rows:
        related = str(row.get(QUERY_CACHE_RELATED_KEY, "")).strip()
        unrelated = str(row.get(QUERY_CACHE_UNRELATED_KEY, "")).strip()
        if related and unrelated:
            already.append(row)
        else:
            need.append(row)
    if not need:
        print(
            f"All {len(already):,} rows already have text queries "
            f"(curated dataset); skipping OpenRouter."
        )
        return rows

    cache = load_query_cache(cache_path)

    unique_captions: list[str] = []
    seen: set[str] = set()
    for row in need:
        caption = pick_caption(row, caption_column)
        if caption and caption not in seen:
            seen.add(caption)
            unique_captions.append(caption)

    missing = [c for c in unique_captions if c not in cache]
    if max_new_queries is not None:
        missing = missing[: max_new_queries]

    def _row_can_offline_query(row: dict[str, Any]) -> bool:
        caption = pick_caption(row, caption_column)
        extras = row.get("captions")
        if not isinstance(extras, (list, tuple)):
            return False
        return any(str(t).strip() and str(t).strip() != caption for t in extras)

    if missing and skip_generation:
        unresolved = [
            c for c in missing
            if not any(
                pick_caption(r, caption_column) == c and _row_can_offline_query(r)
                for r in need
            )
        ]
        if unresolved:
            raise RuntimeError(
                f"{len(unresolved)} captions lack cached queries at {cache_path} "
                f"and have no alternate multi-captions. "
                "Run without --skip-query-generation or pre-fill the cache."
            )
        missing = []

    if missing:
        if query_batch_size < 1:
            raise ValueError("--query-batch-size must be >= 1")
        if query_parallelism < 1:
            raise ValueError("--query-parallelism must be >= 1")

        config = load_openrouter_config(config_path)
        batches = [
            missing[i : i + query_batch_size]
            for i in range(0, len(missing), query_batch_size)
        ]
        print(
            f"Generating text queries for {len(missing):,} captions via OpenRouter "
            f"({config['model']}, batch={query_batch_size}, "
            f"parallel={query_parallelism}, {len(batches):,} API calls) ..."
        )

        generated = 0
        cache_lock = threading.Lock()
        start_time = time.monotonic()

        def _store_batch(batch: list[str], entries: list[dict[str, str]]) -> int:
            nonlocal generated
            with cache_lock:
                for caption, entry in zip(batch, entries):
                    cache[caption] = entry
                    append_query_cache_entry(cache_path, caption, entry)
                    generated += 1
                    if (
                        generated == 1
                        or generated % 100 == 0
                        or generated == len(missing)
                    ):
                        elapsed = time.monotonic() - start_time
                        rate = generated / max(elapsed, 1e-6)
                        print(
                            f"  generated {generated:,}/{len(missing):,} "
                            f"({rate:.1f} captions/s)"
                        )
                return generated

        with ThreadPoolExecutor(max_workers=query_parallelism) as executor:
            futures = {
                executor.submit(
                    _generate_query_batch_with_fallback,
                    batch,
                    api_key=config["api_key"],
                    model=config["model"],
                ): batch
                for batch in batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                entries = future.result()
                if len(entries) != len(batch):
                    raise RuntimeError(
                        f"Query batch size mismatch for {len(batch)} captions "
                        f"(got {len(entries)} entries)"
                    )
                _store_batch(batch, entries)

        elapsed = time.monotonic() - start_time
        print(
            f"Saved query cache to {cache_path} ({generated:,} new entries, "
            f"{elapsed:.1f}s)"
        )

    enriched: list[dict[str, Any]] = list(already)
    for row in need:
        caption = pick_caption(row, caption_column)
        if caption not in cache:
            # Multi-caption rows can use another caption as related offline.
            extras = row.get("captions")
            alt = ""
            if isinstance(extras, (list, tuple)):
                for text in extras:
                    text = str(text).strip()
                    if text and text != caption:
                        alt = text
                        break
            if alt and skip_generation:
                enriched.append({
                    **row,
                    QUERY_CACHE_RELATED_KEY: normalize_training_text(alt),
                    QUERY_CACHE_UNRELATED_KEY: normalize_training_text(
                        row.get(
                            QUERY_CACHE_UNRELATED_KEY,
                            "red sports car on a racetrack at night",
                        )
                    ),
                })
                continue
            raise KeyError(
                f"No cached queries for caption {caption!r}. "
                f"Cache at {cache_path} may be incomplete."
            )
        entry = cache[caption]
        enriched.append({
            **row,
            QUERY_CACHE_RELATED_KEY: normalize_training_text(
                entry[QUERY_CACHE_RELATED_KEY]
            ),
            QUERY_CACHE_UNRELATED_KEY: normalize_training_text(
                entry[QUERY_CACHE_UNRELATED_KEY]
            ),
        })
    print(
        f"Attached text queries to {len(enriched):,} rows "
        f"({len(already):,} pre-filled, {len(unique_captions):,} unique captions)."
    )
    return enriched


def load_verification_samples(
    count: int = VERIFICATION_SAMPLE_COUNT,
    seed: int = 42,
    *,
    hf_dataset: str = DEFAULT_TRISEARCH_HF_DATASET,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Small TriSearch curated slice for post-training checkpoint verification.

    Uses the project dataset only (not Flickr/COCO). Bounded materialize.
    """
    if count > 64:
        raise ValueError("verification sample count must be <= 64")
    rows = load_curated_training_rows(
        hf_dataset=hf_dataset,
        max_samples=count,
        seed=seed,
        satellite_fraction=0.5,
        split=split,
        prefer_local=False,
        materialize=True,
    )
    assert isinstance(rows, list)
    print(
        f"Verification samples: {len(rows):,} from {hf_dataset} (split={split})",
        flush=True,
    )
    return rows


def _tokenize_text(tokenizer, text: str, max_text_length: int) -> dict[str, Any]:
    # Final training choke point: never feed mixed-case strings to the tokenizer.
    text = normalize_training_text(text)
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_text_length,
        padding="max_length",
    )


# ---------------------------------------------------------------------------
# Streamed text–text pairs (≤ queue_size in RAM)
# ---------------------------------------------------------------------------
# AllNLI alone is narrow (caption-style NLI). Default mix prefers broad Q–A:
#   GooAQ (~3M Google questions) + Natural Questions (Wikipedia) + light AllNLI.

DEFAULT_PARAPHRASE_DATASET = "sentence-transformers/gooaq"
DEFAULT_PARAPHRASE_CONFIG = "pair"
DEFAULT_PARAPHRASE_SPLIT = "train"
DEFAULT_PARAPHRASE_QUEUE_SIZE = 1000
DEFAULT_PARAPHRASE_REFILL_SIZE = 1000
DEFAULT_PARAPHRASE_SHUFFLE_BUFFER = 10_000
DEFAULT_MAX_TEXTS_PER_IMAGE = 4

# Weighted multi-source default: broad concepts >> pure NLI.
# Format: (hf_dataset, config, relative_weight)
DEFAULT_PARAPHRASE_SOURCES: tuple[tuple[str, str, float], ...] = (
    ("sentence-transformers/gooaq", "pair", 0.70),
    ("sentence-transformers/natural-questions", "pair", 0.20),
    ("sentence-transformers/all-nli", "triplet", 0.10),
)


@dataclass(frozen=True)
class TextPairSample:
    """One paraphrase/NLI training triple (negative optional)."""

    anchor: str
    positive: str
    negative: str | None = None


def _first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            return normalize_training_text(row[key])
    return ""


def parse_text_pair_row(row: dict[str, Any], *, config: str = "triplet") -> TextPairSample | None:
    """Map a HF embedding-pair row to :class:`TextPairSample`.

    Supports:
      - ``triplet``: anchor/positive/negative (AllNLI)
      - ``pair``: anchor/positive, question/answer, query/answer (GooAQ, NQ, …)
      - ``pair-class``: premise/hypothesis/label (entailment only)
    """
    cfg = (config or "triplet").lower()
    if cfg == "pair-class":
        a = _first_text(row, "premise", "sentence1", "anchor")
        b = _first_text(row, "hypothesis", "sentence2", "positive")
        label = row.get("label")
        # pair-class: 0=entailment, 1=neutral, 2=contradiction (ST convention)
        if label is not None and int(label) != 0:
            return None
        if not a or not b or a == b:
            return None
        return TextPairSample(anchor=a, positive=b, negative=None)
    if cfg == "pair":
        a = _first_text(
            row, "question", "query", "anchor", "sentence1", "premise", "text"
        )
        b = _first_text(
            row, "answer", "positive", "sentence2", "hypothesis", "passage", "context"
        )
        if not a or not b or a == b:
            return None
        return TextPairSample(anchor=a, positive=b, negative=None)
    # triplet (default)
    a = _first_text(row, "anchor", "premise", "question", "query", "sentence1")
    b = _first_text(row, "positive", "hypothesis", "answer", "sentence2")
    c = _first_text(row, "negative")
    if not a or not b or a == b:
        return None
    neg = c if c and c != a and c != b else None
    return TextPairSample(anchor=a, positive=b, negative=neg)


def parse_paraphrase_sources_spec(spec: str | None) -> list[tuple[str, str, float]]:
    """Parse ``dataset:config:weight,dataset:config:weight,...`` or empty → defaults.

    Shorthand: ``dataset`` alone uses config ``pair`` and weight 1.0.
    ``dataset:config`` uses weight 1.0.
    """
    if not spec or not str(spec).strip() or str(spec).strip().lower() in ("mix", "default"):
        return [(d, c, float(w)) for d, c, w in DEFAULT_PARAPHRASE_SOURCES]
    out: list[tuple[str, str, float]] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        bits = [b.strip() for b in part.split(":")]
        if len(bits) == 1:
            out.append((bits[0], "pair", 1.0))
        elif len(bits) == 2:
            # dataset:config  OR  dataset:weight
            try:
                w = float(bits[1])
                out.append((bits[0], "pair", w))
            except ValueError:
                out.append((bits[0], bits[1], 1.0))
        else:
            try:
                w = float(bits[2])
            except ValueError:
                w = 1.0
            out.append((bits[0], bits[1], w))
    if not out:
        return [(d, c, float(w)) for d, c, w in DEFAULT_PARAPHRASE_SOURCES]
    return out


def build_positive_texts_for_image(
    row: dict[str, Any],
    *,
    caption: str,
    related_query: str = "",
    caption_column: str = "caption",
    max_texts: int = DEFAULT_MAX_TEXTS_PER_IMAGE,
) -> list[str]:
    """Deduped texts that should be positives for this image (captions + query)."""
    max_n = max(1, int(max_texts))
    out: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        t = normalize_training_text(text)
        if not t or t in seen:
            return
        seen.add(t)
        out.append(t)

    _add(caption)
    extras = row.get("captions")
    if isinstance(extras, (list, tuple)):
        for item in extras:
            if len(out) >= max_n:
                break
            _add(item)
    # Also try caption_column siblings if captions list missing
    if len(out) < max_n and not isinstance(extras, (list, tuple)):
        for key in ("caption_0", "caption_1", "caption_2", "text"):
            if key == caption_column:
                continue
            if key in row and len(out) < max_n:
                _add(str(row.get(key, "")))
    if related_query and len(out) < max_n:
        _add(related_query)
    return out[:max_n]


class StreamingTextPairQueue:
    """In-memory queue of ≤ ``queue_size`` text pairs; refill from HF stream(s).

    Does **not** materialize full datasets. Supports a **weighted multi-source**
    mix (e.g. GooAQ + Natural Questions + AllNLI) so refills draw broad topics
    rather than a single narrow corpus. Pairs are **removed** on ``pop_batch``.
    """

    def __init__(
        self,
        *,
        dataset_name: str = DEFAULT_PARAPHRASE_DATASET,
        config: str = DEFAULT_PARAPHRASE_CONFIG,
        split: str = DEFAULT_PARAPHRASE_SPLIT,
        queue_size: int = DEFAULT_PARAPHRASE_QUEUE_SIZE,
        refill_size: int = DEFAULT_PARAPHRASE_REFILL_SIZE,
        shuffle_buffer: int = DEFAULT_PARAPHRASE_SHUFFLE_BUFFER,
        seed: int = 42,
        row_iterator: Iterator[dict[str, Any]] | None = None,
        sources: Sequence[tuple[str, str, float]] | None = None,
        use_default_mix: bool = False,
    ):
        if sources is not None:
            self.sources = [(d, c, float(w)) for d, c, w in sources if float(w) > 0]
        elif use_default_mix:
            self.sources = [
                (d, c, float(w)) for d, c, w in DEFAULT_PARAPHRASE_SOURCES
            ]
        else:
            self.sources = [(dataset_name, config, 1.0)]
        if not self.sources:
            self.sources = [(DEFAULT_PARAPHRASE_DATASET, DEFAULT_PARAPHRASE_CONFIG, 1.0)]

        # Back-compat single-source fields (first / only source).
        self.dataset_name = self.sources[0][0]
        self.config = self.sources[0][1]
        self.split = split
        self.queue_size = max(1, int(queue_size))
        self.refill_size = max(1, int(refill_size))
        self.shuffle_buffer = max(100, int(shuffle_buffer))
        self.seed = int(seed)
        self._queue: list[TextPairSample] = []
        # Per-source streaming state: (iterator | None, config_str)
        self._iters: list[Iterator[dict[str, Any]] | None] = [None] * len(self.sources)
        self._source_configs = [c for _, c, _ in self.sources]
        self._owns_stream = row_iterator is None
        # Optional single injected iterator (tests) — always uses sources[0] config.
        self._test_iter: Iterator[dict[str, Any]] | None = row_iterator
        self._refill_count = 0
        self._rng = random.Random(self.seed)
        weights = [max(0.0, float(w)) for _, _, w in self.sources]
        total_w = sum(weights) or 1.0
        self._weights = [w / total_w for w in weights]

    def __len__(self) -> int:
        return len(self._queue)

    @property
    def refill_count(self) -> int:
        return self._refill_count

    def describe(self) -> str:
        parts = [
            f"{d}/{c}@{w:.2f}" for (d, c, _), w in zip(self.sources, self._weights)
        ]
        return " + ".join(parts)

    def _open_stream(self, source_index: int) -> Iterator[dict[str, Any]]:
        from datasets import load_dataset

        name, cfg, _ = self.sources[source_index]
        # No trust_remote_code — parquet-native ST datasets only.
        ds = load_dataset(
            name,
            cfg,
            split=self.split,
            streaming=True,
        )
        seed = self.seed + self._refill_count * 9973 + source_index * 131
        try:
            ds = ds.shuffle(seed=seed, buffer_size=self.shuffle_buffer)
        except Exception:
            pass
        return iter(ds)

    def _pick_source(self) -> int:
        if len(self.sources) == 1:
            return 0
        r = self._rng.random()
        acc = 0.0
        for i, w in enumerate(self._weights):
            acc += w
            if r <= acc:
                return i
        return len(self.sources) - 1

    def _next_row(self, source_index: int) -> tuple[dict[str, Any] | None, str]:
        """Return (row, config) from the chosen source."""
        cfg = self._source_configs[source_index]
        if self._test_iter is not None and source_index == 0:
            try:
                return dict(next(self._test_iter)), cfg
            except StopIteration:
                return None, cfg

        it = self._iters[source_index]
        if it is None:
            if not self._owns_stream:
                return None, cfg
            it = self._open_stream(source_index)
            self._iters[source_index] = it
        try:
            return dict(next(it)), cfg
        except StopIteration:
            if not self._owns_stream:
                return None, cfg
            it = self._open_stream(source_index)
            self._iters[source_index] = it
            try:
                return dict(next(it)), cfg
            except StopIteration:
                return None, cfg

    def refill(self) -> int:
        """Pull up to ``refill_size`` valid pairs into the queue (cap at queue_size)."""
        added = 0
        attempts = 0
        max_attempts = self.refill_size * 25
        while (
            len(self._queue) < self.queue_size
            and added < self.refill_size
            and attempts < max_attempts
        ):
            attempts += 1
            src_i = self._pick_source()
            row, cfg = self._next_row(src_i)
            if row is None:
                # Try other sources before giving up this attempt.
                if len(self.sources) > 1:
                    for j in range(len(self.sources)):
                        if j == src_i:
                            continue
                        row, cfg = self._next_row(j)
                        if row is not None:
                            break
                if row is None:
                    break
            sample = parse_text_pair_row(row, config=cfg)
            if sample is None:
                continue
            self._queue.append(sample)
            added += 1
        self._refill_count += 1
        return added

    def pop_batch(self, n: int) -> list[TextPairSample]:
        """Remove and return ``n`` random pairs from the queue (refilling as needed)."""
        n = max(1, int(n))
        if len(self._queue) < n:
            self.refill()
        if len(self._queue) < n:
            self.refill()
        if len(self._queue) < n:
            raise RuntimeError(
                f"StreamingTextPairQueue could only collect {len(self._queue)} "
                f"pairs (need {n}) from {self.describe()}. "
                "Check network / dataset availability."
            )
        idxs = self._rng.sample(range(len(self._queue)), n)
        batch = [self._queue[i] for i in idxs]
        for i in sorted(idxs, reverse=True):
            del self._queue[i]
        return batch


class ImageCaptionDataset(Dataset):
    """PyTorch dataset over **lazy** or small row sources.

    ``rows`` may be a :class:`TriSearchMapDataset`, any Sequence with
    ``__getitem__`` returning a record dict, or a short list. Images are
    decoded only inside ``__getitem__`` (never preloaded for full corpora).
    """

    def __init__(
        self,
        rows: Any,
        image_processor,
        tokenizer,
        image_column: str = "image",
        caption_column: str = "caption",
        image_root: str | None = None,
        max_text_length: int = 512,
        with_text_queries: bool = False,
        related_query_column: str = QUERY_CACHE_RELATED_KEY,
        unrelated_query_column: str = QUERY_CACHE_UNRELATED_KEY,
        use_extra_captions_as_related: bool = True,
        max_texts_per_image: int = DEFAULT_MAX_TEXTS_PER_IMAGE,
    ):
        self.rows = rows
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_column = image_column
        self.caption_column = caption_column
        self.image_root = image_root
        self.max_text_length = max_text_length
        self.with_text_queries = with_text_queries
        self.related_query_column = related_query_column
        self.unrelated_query_column = unrelated_query_column
        self.use_extra_captions_as_related = use_extra_captions_as_related
        self.max_texts_per_image = max(1, int(max_texts_per_image))

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_related_query(self, row: dict[str, Any], caption: str) -> str:
        """Search query, else another caption for multi-caption rows (lowercase)."""
        related = normalize_training_text(row.get(self.related_query_column, ""))
        if related:
            return related
        if self.use_extra_captions_as_related:
            extras = row.get("captions")
            if isinstance(extras, (list, tuple)):
                for text in extras:
                    text = normalize_training_text(text)
                    if text and text != caption:
                        return text
        return ""

    def _fetch_row(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        if isinstance(row, dict):
            return row
        raise TypeError(f"Row {idx} is not a dict: {type(row)}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._fetch_row(idx)
        image = load_pil_image(row[self.image_column], self.image_root)
        # Always lowercase at the training-stream boundary (even if row was raw HF).
        caption = normalize_training_text(pick_caption(row, self.caption_column))

        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"][0]
        text = _tokenize_text(self.tokenizer, caption, self.max_text_length)
        sample = {
            "pixel_values": pixel_values,
            "input_ids": text["input_ids"][0],
            "attention_mask": text["attention_mask"][0],
            # Raw caption string for multi-positive (false-negative) masking.
            "caption": caption,
        }
        related = ""
        if self.with_text_queries:
            related = self._resolve_related_query(row, caption)
            unrelated = normalize_training_text(
                row.get(self.unrelated_query_column, "")
            )
            if not related or not unrelated:
                raise ValueError(
                    f"Row {idx} is missing text queries "
                    f"({self.related_query_column!r}, "
                    f"{self.unrelated_query_column!r})."
                )
            related_text = _tokenize_text(
                self.tokenizer, related, self.max_text_length
            )
            unrelated_text = _tokenize_text(
                self.tokenizer, unrelated, self.max_text_length
            )
            sample["query_input_ids"] = related_text["input_ids"][0]
            sample["query_attention_mask"] = related_text["attention_mask"][0]
            sample["unrelated_input_ids"] = unrelated_text["input_ids"][0]
            sample["unrelated_attention_mask"] = unrelated_text["attention_mask"][0]
            # Raw query string for multi-positive (false-negative) masking on query tasks.
            sample["related_query"] = related

        positive_texts = build_positive_texts_for_image(
            row,
            caption=caption,
            related_query=related,
            caption_column=self.caption_column,
            max_texts=self.max_texts_per_image,
        )
        pos_ids = []
        pos_masks = []
        for ptext in positive_texts:
            ptok = _tokenize_text(self.tokenizer, ptext, self.max_text_length)
            pos_ids.append(ptok["input_ids"][0])
            pos_masks.append(ptok["attention_mask"][0])
        sample["positive_input_ids"] = pos_ids
        sample["positive_attention_mask"] = pos_masks
        sample["positive_texts"] = positive_texts
        return sample


class Stage1Collator:
    def __init__(self, pad_token_id: int, with_text_queries: bool = False):
        self.pad_token_id = pad_token_id
        self.with_text_queries = with_text_queries

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch = {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        }
        if all("caption" in f for f in features):
            batch["captions"] = [f["caption"] for f in features]
        # Flatten multi-text positives: T texts for B images, text_image_ids maps T→B.
        if all("positive_input_ids" in f for f in features):
            flat_ids: list[Any] = []
            flat_masks: list[Any] = []
            text_image_ids: list[int] = []
            flat_texts: list[str] = []
            for img_i, f in enumerate(features):
                ids_list = f["positive_input_ids"]
                mask_list = f["positive_attention_mask"]
                texts = f.get("positive_texts") or [f.get("caption", "")] * len(ids_list)
                for j, (ids, mask) in enumerate(zip(ids_list, mask_list)):
                    flat_ids.append(ids)
                    flat_masks.append(mask)
                    text_image_ids.append(img_i)
                    if j < len(texts):
                        flat_texts.append(texts[j])
                    else:
                        flat_texts.append("")
            batch["all_input_ids"] = torch.stack(flat_ids)
            batch["all_attention_mask"] = torch.stack(flat_masks)
            batch["text_image_ids"] = torch.tensor(text_image_ids, dtype=torch.long)
            batch["all_positive_texts"] = flat_texts
        if self.with_text_queries:
            batch["query_input_ids"] = torch.stack(
                [f["query_input_ids"] for f in features]
            )
            batch["query_attention_mask"] = torch.stack(
                [f["query_attention_mask"] for f in features]
            )
            batch["unrelated_input_ids"] = torch.stack(
                [f["unrelated_input_ids"] for f in features]
            )
            batch["unrelated_attention_mask"] = torch.stack(
                [f["unrelated_attention_mask"] for f in features]
            )
            if all("related_query" in f for f in features):
                batch["related_queries"] = [f["related_query"] for f in features]
        return batch
