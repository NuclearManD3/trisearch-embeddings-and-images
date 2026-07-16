#!/usr/bin/env python3
"""Stage 2: train MMDiT to reconstruct images from frozen vision embeddings.

Pipeline
--------
1. **Precompute** patch embeddings (+ VAE latents) with **two SigLIP copies**
   (one per GPU), writing per-sample float16 cache files as we go (resume-safe).
2. **Unload** vision models (free VRAM for the generator).
3. **Train** full MMDiT split across both GPUs (pipeline parallel on transformer
   blocks) with conditioning loaded **one micro-batch at a time** from disk.
   Default optimizer is ``adamw8bit`` (moments in VRAM). Host RSS target ≤ ~6GB;
   large mostly-static tensors live on disk (embed cache), not in process memory.
"""

from __future__ import annotations

import json
import math
import resource
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trisearch_dataset import open_trisearch_map_dataset
from trisearch_models.inference import (
    CONDITIONING_HEADS_FILE,
    EMBED_DIM,
    MMDIT_DIR,
    MMDiTGenerator,
    SiglipEmbedder,
    matryoshka_normalize,
    prepare_stage2_condition_tokens,
    resolve_model_dir,
)
from trisearch_models.training import resolve_inference_checkpoint

DEFAULT_STAGE2_DIR = "models/trained/stage2"
DEFAULT_EMBED_CACHE_DIR = "models/data/stage2_embed_cache"
DEFAULT_EMBED_DROPOUT = 0.20
DEFAULT_MERGE_PROB = 0.05
DEFAULT_MAX_COND_TOKENS = 64
DEFAULT_IMAGE_SIZE = 512
CONFIG_FILE = "stage2_config.json"
TRAINING_STATE_FILE = "training_state.pt"
CACHE_META_FILE = "meta.json"
# Host RAM budget: never keep full-model Adam moments in process RSS.
# Moments live either in VRAM (adamw8bit) or as numpy.memmap under optim_dir/.
DEFAULT_OPTIM_DIR_NAME = "optim_disk"
DEFAULT_OPTIMIZER = "adamw8bit"
# Soft host-RAM target for non-mmap allocations (GB). Soft guard only.
HOST_RAM_SOFT_LIMIT_GB = 6.0


def _host_rss_gb() -> float:
    """Current process RSS in GiB (Linux: /proc; fallback: resource)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # kB
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    # ru_maxrss is kB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _log_host_rss(label: str) -> None:
    rss = _host_rss_gb()
    print(f"  [host RSS {rss:.2f} GiB] {label}")
    if rss > HOST_RAM_SOFT_LIMIT_GB + 0.25:
        print(
            f"  warning: host RSS {rss:.2f} GiB exceeds soft limit "
            f"{HOST_RAM_SOFT_LIMIT_GB:.1f} GiB — prefer adamw8bit + disk cache"
        )


# ---------------------------------------------------------------------------
# Vision encode / VAE helpers
# ---------------------------------------------------------------------------


def load_frozen_vision(
    *,
    vision_phase: int = 1,
    checkpoint_dir: str | None = None,
    device: str = "cuda:0",
) -> SiglipEmbedder:
    """Load Stage-1 (or seed) vision embedder and freeze all params."""
    if checkpoint_dir:
        root = Path(checkpoint_dir)
        vision = SiglipEmbedder(
            model_dir=str(root / "vision_model"),
            phase=max(vision_phase, 1),
            projection_path=str(root / "projection_heads.pt"),
            device=device,
        )
    else:
        try:
            root = resolve_inference_checkpoint(phase=vision_phase)
            vision = SiglipEmbedder(
                model_dir=str(root / "vision_model"),
                phase=max(vision_phase, 1),
                projection_path=str(root / "projection_heads.pt"),
                device=device,
            )
        except FileNotFoundError:
            vision = SiglipEmbedder(phase=vision_phase, device=device)

    vision.model.eval()
    vision.projection.eval()
    for p in vision.model.parameters():
        p.requires_grad_(False)
    for p in vision.projection.parameters():
        p.requires_grad_(False)
    return vision


@torch.no_grad()
def encode_vision_patches(
    vision: SiglipEmbedder,
    images: list,
) -> torch.Tensor:
    """Full-grid Matryoshka patch embeddings ``(B, P, D)`` float16 on CPU."""
    inputs = vision.processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(vision.device, non_blocking=True)
    hidden = vision.model(pixel_values=pixel_values).last_hidden_state
    hidden = hidden.to(dtype=vision.projection.weight.dtype)
    projected = vision.projection(hidden)
    # Matryoshka normalize in float32 for stability, store as float16 on disk.
    out = matryoshka_normalize(projected.float()).detach().half().cpu().contiguous()
    return out


def pil_to_vae_tensor(image, size: int) -> torch.Tensor:
    """PIL RGB → ``(1, 3, size, size)`` in ``[-1, 1]`` on CPU."""
    from PIL import Image as PILImage
    import torchvision.transforms.functional as tvf

    if not isinstance(image, PILImage.Image):
        from trisearch_dataset import load_pil_image

        image = load_pil_image(image)
    image = image.convert("RGB")
    # BILINEAR is much faster than default antialias path for 512² precompute.
    image = tvf.resize(
        image,
        [size, size],
        interpolation=tvf.InterpolationMode.BILINEAR,
        antialias=True,
    )
    t = tvf.to_tensor(image)  # [0,1]
    t = t * 2.0 - 1.0
    return t.unsqueeze(0)


def _load_map_rows(map_ds, indices: list[int]) -> list[tuple[int, object, str, str]]:
    """Decode map rows on CPU: ``(idx, pil, caption, record_id)``."""
    out = []
    for i in indices:
        row = map_ds[i]
        cap = str(row.get("caption") or (row.get("captions") or [""])[0])
        rid = str((row.get("id") if isinstance(row, dict) else "") or "")
        out.append((int(i), row["image"], cap, rid))
    return out


def _sample_path(cache_dir: Path, image_id: int) -> Path:
    """Primary on-disk sample path (safetensors; compact float16)."""
    return Path(cache_dir) / "samples" / f"{int(image_id):08d}.safetensors"


def _sample_meta_path(cache_dir: Path, image_id: int) -> Path:
    return Path(cache_dir) / "samples" / f"{int(image_id):08d}.json"


def _legacy_sample_path(cache_dir: Path, image_id: int) -> Path:
    return Path(cache_dir) / "samples" / f"{int(image_id):08d}.pt"


def _save_sample_record(cache_dir: Path | str, rec: dict[str, Any]) -> None:
    """Atomic float16 write via safetensors (+ tiny JSON for caption/id).

    ``torch.save`` of half tensors still stores ~4× larger storages; safetensors
    keeps embeddings+latents at true float16 size (~2MB/sample).
    """
    from safetensors.torch import save_file

    cache_dir = Path(cache_dir)
    image_id = int(rec["image_id"])
    (cache_dir / "samples").mkdir(parents=True, exist_ok=True)
    emb = rec["embeddings"]
    lat = rec["latents"]
    if not torch.is_tensor(emb) or not torch.is_tensor(lat):
        raise TypeError("embeddings/latents must be tensors")
    emb = emb.detach().half().contiguous().cpu()
    lat = lat.detach().half().contiguous().cpu()
    st_path = _sample_path(cache_dir, image_id)
    meta_path = _sample_meta_path(cache_dir, image_id)
    tmp_st = st_path.with_suffix(".safetensors.tmp")
    tmp_meta = meta_path.with_suffix(".json.tmp")
    save_file({"embeddings": emb, "latents": lat}, str(tmp_st))
    tmp_st.replace(st_path)
    meta = {
        "image_id": image_id,
        "caption": rec.get("caption", ""),
        "record_id": rec.get("record_id", ""),
    }
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False) + "\n")
    tmp_meta.replace(meta_path)
    legacy = _legacy_sample_path(cache_dir, image_id)
    if legacy.is_file():
        legacy.unlink()


# ---------------------------------------------------------------------------
# Dual-GPU embedding precompute (resume-safe)
# ---------------------------------------------------------------------------


def _load_cache_meta(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / CACHE_META_FILE
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _save_cache_meta(cache_dir: Path, meta: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / CACHE_META_FILE).write_text(json.dumps(meta, indent=2) + "\n")


def list_cached_sample_ids(cache_dir: Path) -> set[int]:
    samples = cache_dir / "samples"
    if not samples.is_dir():
        return set()
    ids: set[int] = set()
    for p in samples.glob("*.safetensors"):
        try:
            ids.add(int(p.stem))
        except ValueError:
            continue
    for p in samples.glob("*.pt"):
        try:
            ids.add(int(p.stem))
        except ValueError:
            continue
    return ids


def describe_stage2_cache(cache_dir: Path | str) -> dict[str, Any]:
    """Return cache size + meta for logging / safety checks."""
    cache_dir = Path(cache_dir)
    ids = list_cached_sample_ids(cache_dir)
    meta = _load_cache_meta(cache_dir)
    return {
        "cache_dir": str(cache_dir),
        "n_samples": len(ids),
        "meta": meta,
        "max_samples_meta": meta.get("max_samples"),
        "n_total_meta": meta.get("n_total"),
        "complete": bool(meta.get("complete")),
    }


def assert_stage2_train_cache_ok(
    cache_dir: Path | str,
    *,
    max_steps: int | None,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    allow_tiny_cache: bool = False,
    min_samples_warn: int = 256,
    min_samples_hard: int = 16,
) -> dict[str, Any]:
    """Fail or warn when the embed cache is a smoke-sized subset.

    Stage-2 easily *memorizes* a handful of latents if the cache only has
    e.g. 8 images (common after ``--max-samples 8`` smokes). With batch=1 and
    accum=8, 500 optimizer steps ≈ 500 full passes over those 8 images.
    """
    info = describe_stage2_cache(cache_dir)
    n = int(info["n_samples"])
    if n == 0:
        raise RuntimeError(f"Empty Stage-2 embed cache at {cache_dir}")

    bs = max(1, int(batch_size))
    accum = max(1, int(gradient_accumulation_steps))
    images_per_step = bs * accum
    steps = int(max_steps) if max_steps and max_steps > 0 else None
    presentations = (steps * images_per_step) if steps else None
    epochs = (presentations / n) if presentations else None

    meta_ms = info.get("max_samples_meta")
    print(f"  embed cache     : {info['cache_dir']}")
    print(f"  unique samples  : {n:,}" + (f" (meta max_samples={meta_ms})" if meta_ms else ""))
    if info.get("n_total_meta") is not None and int(info["n_total_meta"]) != n:
        print(
            f"  warning: meta n_total={info['n_total_meta']} but only {n} files on disk "
            f"— cache may be incomplete (re-run precompute without --skip-precompute)"
        )
    if epochs is not None:
        print(
            f"  planned coverage: ~{epochs:.1f} epochs "
            f"({presentations:,} image presentations / {n:,} unique)"
        )

    if n < min_samples_hard and not allow_tiny_cache:
        raise RuntimeError(
            f"Stage-2 embed cache has only {n} unique samples under {cache_dir}. "
            f"Training will memorize them (roads/rooms/flags in fixed layouts). "
            f"Precompute the full set, e.g.:\n"
            f"  python3 train_stage2.py --max-steps 0 --precompute-only "
            f"--embed-cache-dir {cache_dir}\n"
            f"(omit --max-samples; use a fresh cache dir if this one was a smoke). "
            f"Pass --allow-tiny-cache only for deliberate overfit smokes."
        )
    if n < min_samples_warn:
        print(
            f"  WARNING: only {n} unique images — expect near-exact memorization. "
            f"Use a full precompute (no --max-samples) for real Stage-2 training."
        )
    if epochs is not None and epochs > 50 and n < 1000:
        print(
            f"  WARNING: ~{epochs:.0f} epochs over {n} images is heavy overfit. "
            f"Increase the cache and/or reduce --max-steps."
        )
    info["epochs_planned"] = epochs
    info["presentations_planned"] = presentations
    return info


def precompute_stage2_embeddings(
    *,
    args: Any,
    cache_dir: Path | str,
    devices: list[str],
    batch_size: int = 4,
    image_size: int | None = None,
    rebuild: bool = False,
) -> Path:
    """Encode all map images with dual vision models; cache emb + VAE latents.

    Layout (fast path)
    ------------------
    * Each GPU owns **SigLIP + VAE** and runs a full shard end-to-end
      (decode → vision → VAE) with **no barrier** between GPUs.
    * Previously the VAE lived only on GPU1, so after dual vision GPU0 sat
      idle while GPU1 ran VAE for the whole wave (~4–5 GB extra VRAM on GPU1
      and lower GPU0 util).
    * Disk writes are async (writer thread pool) so the next GPU wave can
      start while safetensors flush.

    Resume: skip ids that already have ``samples/{id}.safetensors``.
    """
    import gc

    from diffusers import AutoencoderKL

    from trisearch_models.inference import MMDIT_VAE_ID

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "samples").mkdir(parents=True, exist_ok=True)
    image_size = int(image_size or getattr(args, "image_size", DEFAULT_IMAGE_SIZE))

    map_ds = open_trisearch_map_dataset(
        hf_dataset=args.hf_dataset,
        dataset_dir=getattr(args, "curated_dataset_dir", None),
        prefer_local=getattr(args, "prefer_local_curated", False),
        split=getattr(args, "curated_split", "train"),
        max_samples=getattr(args, "max_samples", None),
        seed=getattr(args, "seed", 42),
        satellite_fraction=getattr(args, "satellite_fraction", 0.5),
    )
    n_total = len(map_ds)
    meta = {
        "hf_dataset": args.hf_dataset,
        "split": getattr(args, "curated_split", "train"),
        "max_samples": getattr(args, "max_samples", None),
        "seed": getattr(args, "seed", 42),
        "satellite_fraction": getattr(args, "satellite_fraction", 0.5),
        "vision_phase": getattr(args, "vision_phase", 1),
        "vision_checkpoint_dir": getattr(args, "vision_checkpoint_dir", None),
        "image_size": image_size,
        "n_total": n_total,
        "embed_dim": EMBED_DIM,
        "dtype": "float16",
    }
    existing_meta = _load_cache_meta(cache_dir)
    if existing_meta and not rebuild:
        for key in ("hf_dataset", "split", "seed", "image_size", "vision_phase"):
            if key in existing_meta and existing_meta[key] != meta[key]:
                raise RuntimeError(
                    f"Embed cache at {cache_dir} was built with {key}="
                    f"{existing_meta[key]!r}, but current run has {meta[key]!r}. "
                    f"Use --rebuild-embed-cache or a different --embed-cache-dir."
                )
        old_ms = existing_meta.get("max_samples")
        new_ms = meta.get("max_samples")
        if old_ms is not None and new_ms is None and existing_meta.get("n_total", 0) < 1000:
            print(
                f"  note: cache was built with max_samples={old_ms}; "
                f"this run has no cap — will encode remaining ids up to full split."
            )
        if (
            old_ms is not None
            and new_ms is not None
            and int(old_ms) != int(new_ms)
            and int(new_ms) < int(old_ms)
        ):
            print(
                f"  note: cache max_samples={old_ms} → requested {new_ms}; "
                f"using existing files (ids may exceed the new cap)."
            )
    if rebuild:
        for p in (cache_dir / "samples").glob("*"):
            if p.suffix in (".pt", ".safetensors", ".json", ".tmp") or p.name.endswith(
                ".tmp"
            ):
                p.unlink()
        done: set[int] = set()
    else:
        done = list_cached_sample_ids(cache_dir)

    _save_cache_meta(cache_dir, {**meta, "n_done": len(done)})

    pending = [i for i in range(n_total) if i not in done]
    print(
        f"Stage-2 embed precompute: {len(done):,}/{n_total:,} cached, "
        f"{len(pending):,} remaining → {cache_dir}"
    )
    if not pending:
        print("  embed cache complete; skipping encode.")
        return cache_dir

    if len(devices) < 2:
        devices = [devices[0], devices[0]]
    dual = devices[0] != devices[1]
    print(
        f"  loading vision+VAE on {devices[0]}"
        + (f" and {devices[1]} (independent pipelines)" if dual else "")
        + " ..."
    )
    vision_a = load_frozen_vision(
        vision_phase=args.vision_phase,
        checkpoint_dir=args.vision_checkpoint_dir,
        device=devices[0],
    )
    vision_b = (
        load_frozen_vision(
            vision_phase=args.vision_phase,
            checkpoint_dir=args.vision_checkpoint_dir,
            device=devices[1],
        )
        if dual
        else vision_a
    )

    def _load_vae(device: str | torch.device):
        v = AutoencoderKL.from_pretrained(
            MMDIT_VAE_ID, subfolder="vae", torch_dtype=torch.float16
        )
        v = v.to(device).eval()
        for p in v.parameters():
            p.requires_grad_(False)
        return v

    vae_a = _load_vae(devices[0])
    vae_b = _load_vae(devices[1]) if dual else vae_a
    shift = getattr(vae_a.config, "shift_factor", 0.0) or 0.0
    scaling = float(vae_a.config.scaling_factor)
    _log_host_rss("after dual vision+VAE load")
    if torch.cuda.is_available():
        for dev in ({devices[0], devices[1]} if dual else {devices[0]}):
            d = torch.device(dev)
            if d.type == "cuda":
                print(
                    f"  VRAM {dev}: "
                    f"{torch.cuda.memory_allocated(d) / 1024**3:.2f} GiB alloc"
                )

    def _encode_vae_on(
        vae: torch.nn.Module, device: torch.device, pil_images: list
    ) -> torch.Tensor:
        """``(B, C, H, W)`` float16 latents on CPU."""
        # Parallel CPU resize — often the bottleneck when GPU waits.
        if len(pil_images) >= 4:
            with ThreadPoolExecutor(max_workers=min(8, len(pil_images))) as rp:
                tensors = list(rp.map(lambda im: pil_to_vae_tensor(im, image_size), pil_images))
            xs = torch.cat(tensors, dim=0)
        else:
            xs = torch.cat(
                [pil_to_vae_tensor(im, image_size) for im in pil_images], dim=0
            )
        xs = xs.to(device=device, dtype=torch.float16, non_blocking=True)
        with torch.no_grad():
            lat = vae.encode(xs).latent_dist.sample()
            lat = (lat - shift) * scaling
        out = lat.detach().half().cpu().contiguous()
        del xs, lat
        return out

    def _pipeline_worker(
        vision: SiglipEmbedder,
        vae: torch.nn.Module,
        device: str,
        rows: list[tuple[int, object, str, str]],
    ) -> list[dict[str, Any]]:
        """Per-GPU path on **prefetched** rows: SigLIP → VAE → CPU records."""
        if not rows:
            return []
        if str(device).startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        indices = [r[0] for r in rows]
        images = [r[1] for r in rows]
        captions = [r[2] for r in rows]
        record_ids = [r[3] for r in rows]
        with torch.no_grad():
            embs = encode_vision_patches(vision, images)  # (B,P,D) f16 cpu
            lats = _encode_vae_on(vae, torch.device(device), images)
        # No hard synchronize — next wave / writers can overlap CUDA completion.
        out: list[dict[str, Any]] = []
        for j, idx in enumerate(indices):
            out.append(
                {
                    "image_id": int(idx),
                    "caption": captions[j],
                    "embeddings": embs[j].contiguous(),
                    "latents": lats[j].contiguous(),
                    "record_id": record_ids[j],
                }
            )
        return out

    batch_size = max(1, int(batch_size))
    wave = batch_size * (2 if dual else 1)
    pbar = tqdm(total=len(pending), desc="Precompute embeds", unit="img")
    write_futs: list = []
    meta_every = max(wave * 4, 32)

    def _drain_writes(*, force: bool = False) -> None:
        nonlocal write_futs
        if not write_futs:
            return
        if not force and len(write_futs) < meta_every:
            return
        for fut in write_futs:
            fut.result()
        write_futs = []
        _save_cache_meta(cache_dir, {**meta, "n_done": len(done)})

    def _split_chunk(chunk: list[int]) -> tuple[list[int], list[int]]:
        if not dual:
            return chunk, []
        mid = len(chunk) // 2
        return chunk[:mid], chunk[mid:]

    with ThreadPoolExecutor(max_workers=2 if dual else 1) as gpu_pool, ThreadPoolExecutor(
        max_workers=4
    ) as io_pool, ThreadPoolExecutor(max_workers=2) as cpu_pool:
        # Prefetch first wave on CPU while... (nothing yet); then overlap.
        def _prefetch(chunk: list[int]):
            left_i, right_i = _split_chunk(chunk)
            if dual:
                fl = cpu_pool.submit(_load_map_rows, map_ds, left_i)
                fr = cpu_pool.submit(_load_map_rows, map_ds, right_i)
                return fl.result(), fr.result()
            return _load_map_rows(map_ds, left_i), []

        waves = [pending[s : s + wave] for s in range(0, len(pending), wave)]
        pref_left, pref_right = _prefetch(waves[0]) if waves else ([], [])

        for wi, chunk in enumerate(waves):
            # Kick off prefetch of *next* wave while GPUs run this one.
            next_pref = None
            if wi + 1 < len(waves):
                next_pref = cpu_pool.submit(_prefetch, waves[wi + 1])

            if dual:
                fut_a = gpu_pool.submit(
                    _pipeline_worker, vision_a, vae_a, devices[0], pref_left
                )
                fut_b = gpu_pool.submit(
                    _pipeline_worker, vision_b, vae_b, devices[1], pref_right
                )
                records = fut_a.result() + fut_b.result()
            else:
                records = _pipeline_worker(vision_a, vae_a, devices[0], pref_left)

            for rec in records:
                idx = int(rec["image_id"])
                write_futs.append(io_pool.submit(_save_sample_record, cache_dir, rec))
                done.add(idx)
                pbar.update(1)
            _drain_writes(force=False)
            del records

            if next_pref is not None:
                pref_left, pref_right = next_pref.result()
            else:
                pref_left, pref_right = [], []

        _drain_writes(force=True)
    pbar.close()

    # Unload vision + VAE before training loads MMDiT (free VRAM + host).
    del vision_a, vision_b, vae_a, vae_b, map_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    _log_host_rss("after precompute unload")
    print(f"  precompute done: {len(done):,}/{n_total:,} → unloaded embedders")
    _save_cache_meta(cache_dir, {**meta, "n_done": len(done), "complete": True})
    return cache_dir


class Stage2EmbedCacheDataset(Dataset):
    """Map-style dataset over precomputed per-sample cache files.

    Only path strings live in RAM; each ``__getitem__`` loads one float16
    sample from disk (never materialize the full cache into host memory).
    Prefers ``.safetensors`` (+ ``.json`` meta); falls back to legacy ``.pt``.
    """

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        samples = self.cache_dir / "samples"
        if not samples.is_dir():
            raise FileNotFoundError(f"No samples/ under {self.cache_dir}")
        st = {p.stem: p for p in samples.glob("*.safetensors")}
        pt = {p.stem: p for p in samples.glob("*.pt")}
        stems = sorted(set(st) | set(pt), key=lambda s: int(s) if s.isdigit() else s)
        if not stems:
            raise FileNotFoundError(f"Empty embed cache at {self.cache_dir}")
        self._entries: list[tuple[str, Path]] = []
        for stem in stems:
            if stem in st:
                self._entries.append(("st", st[stem]))
            else:
                self._entries.append(("pt", pt[stem]))

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        kind, path = self._entries[idx]
        if kind == "pt":
            return torch.load(path, map_location="cpu", weights_only=False)
        from safetensors.torch import load_file

        tensors = load_file(str(path), device="cpu")
        meta_path = path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
        return {
            "image_id": int(meta.get("image_id", path.stem)),
            "caption": meta.get("caption", ""),
            "record_id": meta.get("record_id", ""),
            "embeddings": tensors["embeddings"],
            "latents": tensors["latents"],
        }


def collate_stage2_cache(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack embeddings/latents as float16 (cast to compute dtype on GPU).

    Keeps host peak for a micro-batch tiny (~2MB/sample f16).
    """
    max_p = max(int(b["embeddings"].shape[0]) for b in batch)
    d = int(batch[0]["embeddings"].shape[-1])
    bsz = len(batch)
    embs = torch.zeros(bsz, max_p, d, dtype=torch.float16)
    lats = torch.stack([b["latents"].half() for b in batch], dim=0)
    ids = []
    captions = []
    for i, b in enumerate(batch):
        e = b["embeddings"].half()
        embs[i, : e.shape[0]] = e
        if e.shape[0] < max_p:
            embs[i, e.shape[0] :] = e[-1:]
        ids.append(int(b["image_id"]))
        captions.append(b.get("caption", ""))
    return {
        "embeddings": embs,
        "latents": lats,
        "image_ids": ids,
        "captions": captions,
    }


def build_stage2_cache_dataloader(args: Any, cache_dir: Path | str) -> DataLoader:
    """Build DataLoader over the embed cache.

    Default ``num_workers=2`` prefetches the next batches off the main thread
    (main-thread-only loading was pegging one Python core and starving GPUs).
    Keep workers low: each worker process costs host RAM.
    """
    ds = Stage2EmbedCacheDataset(cache_dir)
    nw = int(getattr(args, "dataloader_workers", 2) or 0)
    nw = max(0, min(nw, 4))
    kwargs: dict[str, Any] = dict(
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        drop_last=True,
        collate_fn=collate_stage2_cache,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


# ---------------------------------------------------------------------------
# Pipeline-parallel MMDiT (layer split across 2 GPUs)
# ---------------------------------------------------------------------------


def apply_pipeline_parallel(
    transformer: torch.nn.Module,
    device0: str | torch.device,
    device1: str | torch.device,
    *,
    split_at: int | None = None,
) -> int:
    """Place early blocks on ``device0``, late blocks + output on ``device1``.

    Returns the split index (first block index on device1).
    """
    d0 = torch.device(device0)
    d1 = torch.device(device1)
    n = len(transformer.transformer_blocks)
    split = int(split_at) if split_at is not None else n // 2
    split = max(1, min(split, n - 1)) if n >= 2 else 0

    transformer.pos_embed.to(d0)
    transformer.time_text_embed.to(d0)
    transformer.context_embedder.to(d0)
    for i, block in enumerate(transformer.transformer_blocks):
        block.to(d0 if i < split else d1)
    transformer.norm_out.to(d1)
    transformer.proj_out.to(d1)
    if hasattr(transformer, "image_proj"):
        transformer.image_proj.to(d0)

    transformer._pp_split_at = split
    transformer._pp_device0 = d0
    transformer._pp_device1 = d1

    # Install dual-device forward (preserves gradient checkpointing when enabled).
    _install_pipeline_forward(transformer)
    print(
        f"  pipeline parallel: blocks[0:{split}] on {d0}, "
        f"blocks[{split}:{n}] + head on {d1}"
    )
    return split


def _install_pipeline_forward(transformer: torch.nn.Module) -> None:
    """Replace forward with a device-aware pipeline that moves tensors at split."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: list = None,
        joint_attention_kwargs: dict | None = None,
        return_dict: bool = True,
        skip_layers: list[int] | None = None,
    ):
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        d0 = self._pp_device0
        d1 = self._pp_device1
        split = int(self._pp_split_at)

        height, width = hidden_states.shape[-2:]
        hidden_states = hidden_states.to(d0)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(d0)
        if pooled_projections is not None:
            pooled_projections = pooled_projections.to(d0)
        if timestep is not None:
            timestep = timestep.to(d0)

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in (
            joint_attention_kwargs or {}
        ):
            # Not used in Stage 2; keep API compatible if present.
            pass

        for index_block, block in enumerate(self.transformer_blocks):
            if index_block == split:
                hidden_states = hidden_states.to(d1)
                encoder_hidden_states = encoder_hidden_states.to(d1)
                temb = temb.to(d1)

            is_skip = skip_layers is not None and index_block in skip_layers
            if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                )
            elif not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if (
                block_controlnet_hidden_states is not None
                and getattr(block, "context_pre_only", False) is False
            ):
                interval_control = len(self.transformer_blocks) / len(
                    block_controlnet_hidden_states
                )
                hidden_states = (
                    hidden_states
                    + block_controlnet_hidden_states[
                        int(index_block / interval_control)
                    ].to(hidden_states.device)
                )

        # Ensure head device
        hidden_states = hidden_states.to(d1)
        temb = temb.to(d1)
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size
        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )
        )
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    # Bind as method
    import types

    transformer.forward = types.MethodType(forward, transformer)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def _save_transformer_sharded(transformer: torch.nn.Module, mmdit_dir: Path) -> None:
    """Save multi-device transformer as several safetensor shards (low host peak).

    Writes shards of ≤512MB host so we never hold a full 4GB CPU state dict.
    """
    import gc

    from safetensors.torch import save_file

    mmdit_dir.mkdir(parents=True, exist_ok=True)
    transformer.config.save_pretrained(mmdit_dir)

    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_idx = 0
    weight_map: dict[str, str] = {}
    max_shard = 512 * 1024 * 1024  # 512 MiB host at a time

    def _flush():
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        fname = f"diffusion_pytorch_model-{shard_idx:05d}-of-shards.safetensors"
        save_file(shard, str(mmdit_dir / fname))
        for k in shard:
            weight_map[k] = fname
        shard = {}
        shard_bytes = 0
        shard_idx += 1
        gc.collect()

    for name, tensor in transformer.state_dict().items():
        cpu_t = tensor.detach().to("cpu", copy=True).contiguous()
        nbytes = cpu_t.numel() * cpu_t.element_size()
        if shard and shard_bytes + nbytes > max_shard:
            _flush()
        shard[name] = cpu_t
        shard_bytes += nbytes
        del cpu_t
    _flush()

    # Single-file rename if only one shard (HF-friendly).
    if shard_idx == 1:
        src = mmdit_dir / "diffusion_pytorch_model-00000-of-shards.safetensors"
        dst = mmdit_dir / "diffusion_pytorch_model.safetensors"
        if src.is_file():
            src.replace(dst)
            weight_map = {k: "diffusion_pytorch_model.safetensors" for k in weight_map}
    else:
        index = {
            "metadata": {"total_size": sum(
                (mmdit_dir / f).stat().st_size
                for f in set(weight_map.values())
                if (mmdit_dir / f).is_file()
            )},
            "weight_map": weight_map,
        }
        (mmdit_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )
    gc.collect()


def save_stage2_checkpoint(
    root: Path,
    generator: MMDiTGenerator,
    args: Any,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    import gc
    import shutil

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    mmdit_dir = root / "mmdit"
    mmdit_dir.mkdir(parents=True, exist_ok=True)

    # Do NOT move the full module tree to CPU (spikes host RAM). Save sharded.
    try:
        generator.transformer.save_pretrained(mmdit_dir, safe_serialization=True)
    except Exception:
        _save_transformer_sharded(generator.transformer, mmdit_dir)

    # Adapters are small (~tens of MB).
    seq_cpu = {k: v.detach().cpu() for k, v in generator.embed_to_seq.state_dict().items()}
    pool_cpu = {
        k: v.detach().cpu() for k, v in generator.embed_to_pool.state_dict().items()
    }
    torch.save(
        {"embed_to_seq": seq_cpu, "embed_to_pool": pool_cpu},
        root / CONDITIONING_HEADS_FILE,
    )
    del seq_cpu, pool_cpu
    gc.collect()

    cfg = {
        "step": step,
        "embed_dropout": getattr(args, "embed_dropout", DEFAULT_EMBED_DROPOUT),
        "merge_prob": getattr(args, "merge_prob", DEFAULT_MERGE_PROB),
        "max_cond_tokens": getattr(args, "max_cond_tokens", DEFAULT_MAX_COND_TOKENS),
        "image_size": getattr(args, "image_size", DEFAULT_IMAGE_SIZE),
        "learning_rate": getattr(args, "learning_rate", None),
        "vision_phase": getattr(args, "vision_phase", 1),
        "full_pretrain": True,
        "optimizer": getattr(args, "optimizer", DEFAULT_OPTIMIZER),
        "pipeline_parallel": bool(getattr(generator, "_pp_devices", None)),
        "embed_cache_dir": getattr(args, "embed_cache_dir", DEFAULT_EMBED_CACHE_DIR),
    }
    (root / CONFIG_FILE).write_text(json.dumps(cfg, indent=2) + "\n")
    if optimizer is not None:
        # Never pickle full Adam state (multi-GB host spike). Metadata only.
        opt_meta = {"step": step, "optimizer_class": type(optimizer).__name__}
        if hasattr(optimizer, "state_dir"):
            opt_meta["state_dir"] = str(optimizer.state_dir)
        else:
            opt_meta["note"] = (
                "optimizer state not pickled (adamw8bit lives in VRAM; "
                "disk_adamw moments live under optim_disk/)"
            )
        torch.save(opt_meta, root / TRAINING_STATE_FILE)
    hist = root / "history" / f"step-{step}"
    if step > 0:
        hist.mkdir(parents=True, exist_ok=True)
        if (hist / "mmdit").exists():
            shutil.rmtree(hist / "mmdit")
        shutil.copytree(mmdit_dir, hist / "mmdit")
        shutil.copy2(root / CONDITIONING_HEADS_FILE, hist / CONDITIONING_HEADS_FILE)
        (hist / CONFIG_FILE).write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Saved Stage-2 checkpoint to {root} (step {step})")


def load_stage2_training_state(root: Path, optimizer: torch.optim.Optimizer) -> int:
    path = Path(root) / TRAINING_STATE_FILE
    if not path.is_file():
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    try:
        optimizer.load_state_dict(state["optimizer"])
    except Exception as exc:
        print(f"  warning: could not restore optimizer state ({exc}); continuing")
    return int(state.get("step", 0))


# ---------------------------------------------------------------------------
# Training loop (from precomputed cache)
# ---------------------------------------------------------------------------


def run_stage2_training(
    *,
    generator: MMDiTGenerator,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: Any,
    start_step: int = 0,
) -> int:
    """Train full MMDiT from precomputed embeddings/latents (no live vision)."""
    if not any(p.requires_grad for p in generator.transformer.parameters()):
        generator.freeze_non_stage2()
    generator.train()

    max_steps = int(args.max_steps) if args.max_steps > 0 else None
    accum = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
    embed_drop = float(getattr(args, "embed_dropout", DEFAULT_EMBED_DROPOUT))
    merge_prob = float(getattr(args, "merge_prob", DEFAULT_MERGE_PROB))
    max_cond = int(getattr(args, "max_cond_tokens", DEFAULT_MAX_COND_TOKENS))
    log_every = max(1, int(getattr(args, "logging_steps", 10)))
    save_every = max(1, int(getattr(args, "save_steps", 250)))
    trained_dir = Path(getattr(args, "trained_dir", DEFAULT_STAGE2_DIR))

    # Primary device = first pipeline device (or generator.device).
    primary = getattr(generator.transformer, "_pp_device0", None) or generator.device

    global_step = start_step
    micro = 0
    log_loss = None  # GPU tensor running sum; sync only on log
    log_n = 0
    optimizer.zero_grad(set_to_none=True)

    if max_steps is not None and start_step >= max_steps:
        print(f"Already at step {start_step} (max_steps={max_steps}); nothing to train.")
        return start_step

    # Latents are precomputed — drop VAE entirely (saves ~0.5–2 GiB host RSS).
    if getattr(generator, "vae", None) is not None:
        try:
            generator.vae.to("cpu")
        except Exception:
            pass
        del generator.vae
        generator.vae = None
        import gc

        gc.collect()
    # Tokenizer unused in cache train path.
    if getattr(generator, "tokenizer", None) is not None:
        generator.tokenizer = None
    _log_host_rss("train loop start")

    # Cache trainable param list once (avoids re-walk every clip step).
    trainable = [p for p in generator.trainable_parameters() if p.requires_grad]
    max_grad_norm = float(getattr(args, "max_grad_norm", 1.0))
    # Warm FM schedule once (avoids set_timesteps every micro-step).
    if hasattr(generator, "_ensure_train_schedule"):
        generator._ensure_train_schedule(primary)

    while True:
        for batch in dataloader:
            # H2D async — pin_memory from DataLoader helps when CUDA.
            embeddings = batch["embeddings"].to(
                device=primary, dtype=torch.float32, non_blocking=True
            )
            cond = prepare_stage2_condition_tokens(
                embeddings,
                shuffle=True,
                drop_prob=embed_drop,
                merge_prob=merge_prob,
                max_tokens=max_cond,
                training=True,
            )
            clean_latents = batch["latents"].to(
                device=primary, dtype=generator.compute_dtype, non_blocking=True
            )

            cond_shape = tuple(cond.shape)
            # No host metrics each step — .item()/.cpu() serializes the GPU.
            loss, _, _ = generator.forward_train(
                clean_latents, cond, return_metrics=False
            )
            (loss / accum).backward()
            log_loss = loss.detach() if log_loss is None else (log_loss + loss.detach())
            log_n += 1
            micro += 1
            last_cond_shape = cond_shape
            del loss, clean_latents, embeddings, cond

            if micro % accum != 0:
                continue

            # Do NOT empty_cache every step — that forces realloc thrash and
            # starves the GPU. Only free between rare checkpoint saves.
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step == 1 or global_step % log_every == 0:
                rss = _host_rss_gb()
                mean_loss = float(log_loss.float().item()) / max(log_n, 1)
                print(
                    f"step {global_step:5d} | loss {mean_loss:.4f} | "
                    f"cond_tokens {last_cond_shape} | host {rss:.2f}GiB"
                )
                log_loss = None
                log_n = 0

            if global_step % save_every == 0:
                save_stage2_checkpoint(
                    trained_dir, generator, args, global_step, optimizer
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if max_steps is not None and global_step >= max_steps:
                save_stage2_checkpoint(
                    trained_dir, generator, args, global_step, optimizer
                )
                return global_step

        if max_steps is None:
            break

    save_stage2_checkpoint(trained_dir, generator, args, global_step, optimizer)
    return global_step


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


class DiskOffloadAdamW(torch.optim.Optimizer):
    """AdamW with moments as **open numpy.memmap** files (host RSS stays small).

    Prefer ``adamw8bit`` for speed (moments in VRAM). This class is the low-host
    fallback when VRAM cannot hold 8-bit moments.

    Holding fp32 moments for ~2B params in process memory is ~16GB. Memmap keeps
    them on disk; mmaps stay open across steps (no open/close thrash). Updates
    run on GPU for the active parameter, streaming moments through a single
    reused host buffer so we never allocate multi-GB host tensors.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        state_dir: str | Path = "./optim_disk",
    ):
        import numpy as np

        self._np = np
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._param_keys: dict[int, str] = {}
        self._counter = 0
        # Keep open memmaps: key -> (m1, m2)
        self._open: dict[str, tuple[Any, Any]] = {}
        self._step_n: dict[str, int] = {}

    def _key_for(self, p: torch.nn.Parameter) -> str:
        pid = id(p)
        if pid not in self._param_keys:
            self._param_keys[pid] = f"p{self._counter:06d}"
            self._counter += 1
        return self._param_keys[pid]

    def _ensure_mmaps(self, key: str, shape: tuple[int, ...]):
        if key in self._open:
            return self._open[key]
        np = self._np
        n = int(math.prod(shape))
        paths = []
        maps = []
        for suffix in ("m1", "m2"):
            path = self.state_dir / f"{key}_{suffix}.dat"
            if not path.is_file() or path.stat().st_size != n * 4:
                mm = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
                mm[:] = 0
                mm.flush()
                del mm
            paths.append(path)
            maps.append(np.memmap(path, dtype=np.float32, mode="r+", shape=shape))
        self._open[key] = (maps[0], maps[1])
        step_path = self.state_dir / f"{key}_step.txt"
        if key not in self._step_n:
            self._step_n[key] = (
                int(step_path.read_text()) if step_path.is_file() else 0
            )
        return self._open[key]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        np = self._np
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                key = self._key_for(p)
                shape = tuple(p.shape)
                m1_mm, m2_mm = self._ensure_mmaps(key, shape)
                # Stream: host→GPU for moments, Adam on device, write back.
                # Reuse flat views; one param at a time keeps host peak low.
                device = p.device
                g = p.grad.detach().float()
                p_f = p.detach().float()
                m1 = torch.from_numpy(np.asarray(m1_mm)).to(device, non_blocking=True)
                m2 = torch.from_numpy(np.asarray(m2_mm)).to(device, non_blocking=True)
                step_n = self._step_n[key] + 1
                self._step_n[key] = step_n
                if wd != 0.0:
                    p_f = p_f.mul(1.0 - lr * wd)
                m1.mul_(beta1).add_(g, alpha=1.0 - beta1)
                m2.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                bias1 = 1.0 - beta1**step_n
                bias2 = 1.0 - beta2**step_n
                step_size = lr / bias1
                denom = m2.div(bias2).sqrt_().add_(eps)
                p_f.addcdiv_(m1, denom, value=-step_size)
                # Write moments back (copy_ into memmap via numpy view).
                m1_mm[:] = m1.detach().cpu().numpy()
                m2_mm[:] = m2.detach().cpu().numpy()
                p.copy_(p_f.to(dtype=p.dtype))
                p.grad = None
                del g, p_f, m1, m2, denom
        # Step counters: flush every call is fine (tiny files); skip if empty.
        if self._step_n:
            # Batch-write only keys we touched would need a dirty set; full
            # rewrite of counters is cheap vs moment I/O.
            for key, n in self._step_n.items():
                (self.state_dir / f"{key}_step.txt").write_text(str(n))
        return loss


CPUOffloadAdamW = DiskOffloadAdamW  # back-compat alias


def build_stage2_optimizer(
    generator: MMDiTGenerator,
    learning_rate: float,
    weight_decay: float = 0.01,
    optimizer_name: str = DEFAULT_OPTIMIZER,
    state_dir: str | Path | None = None,
):
    """Build optimizer for full Stage-2 pretraining.

    Default ``adamw8bit``: 8-bit Adam moments in **VRAM** (fast; use with
    pipeline parallel so weights+moments fit on 2×12GB).
    ``disk_adamw``: moments as open memmaps (slow, host RSS ≪ full model).
    """
    params = [p for p in generator.trainable_parameters() if p.requires_grad]
    name = (optimizer_name or DEFAULT_OPTIMIZER).lower()
    if name in ("adamw8bit", "adam8bit", "bnb"):
        import bitsandbytes as bnb

        print(
            f"  optimizer adamw8bit: {sum(p.numel() for p in params)/1e6:.1f}M "
            f"params; moments in VRAM"
        )
        return bnb.optim.AdamW8bit(params, lr=learning_rate, weight_decay=weight_decay)
    if name in ("disk_adamw", "cpu_adamw", "cpu-adamw", "offload_adamw"):
        if state_dir is None:
            state_dir = Path(DEFAULT_STAGE2_DIR) / DEFAULT_OPTIM_DIR_NAME
        print(
            f"  optimizer disk_adamw: moments under {state_dir} "
            f"(slow; prefer adamw8bit if VRAM allows)"
        )
        return DiskOffloadAdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            state_dir=state_dir,
        )
    if name in ("adamw", "adam"):
        print(
            "  warning: full AdamW keeps fp32 moments on host (~8 bytes/param) — "
            "may exceed 6GB RSS"
        )
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=learning_rate, momentum=0.0, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer {optimizer_name!r}")


def setup_pipeline_generator(
    *,
    model_dir: str,
    conditioning_path: str | None,
    device0: str,
    device1: str,
) -> MMDiTGenerator:
    """Load MMDiT, enable full train, split transformer across two GPUs.

    Strategy (fastest on 2×12GB for full pretrain):
    * Load weights with ``low_cpu_mem_usage`` then **pipeline-place** blocks
      (early on GPU0, late on GPU1) so neither GPU holds the full model.
    * VAE + unused text path stay on CPU (latents come from disk cache).
    * Prefer ``adamw8bit`` moments in VRAM with the split weights.
    """
    import gc

    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers import SD3Transformer2DModel
    from transformers import AutoTokenizer

    from trisearch_models.inference import (
        EMBED_DIM,
        MMDIT_VAE_ID,
        QWEN_TOKENIZER_ID,
        CONDITIONING_HEADS_FILE as _CHF,
    )

    d0 = torch.device(device0)
    d1 = torch.device(device1)
    compute_dtype = (
        torch.bfloat16
        if str(device0).startswith("cuda") and torch.cuda.is_bf16_supported()
        else (torch.float16 if str(device0).startswith("cuda") else torch.float32)
    )

    print(f"  loading MMDiT (low_cpu_mem) then pipeline-placing on {d0}+{d1} ...")
    # Build a thin shell without MMDiTGenerator.__init__ full .to(device0),
    # which would peak at the full model on one GPU + host.
    gen = object.__new__(MMDiTGenerator)
    gen.device = d0
    gen.embed_dim = EMBED_DIM
    gen.phase = 2 if conditioning_path else 0
    gen.model_dir = model_dir
    gen.compute_dtype = compute_dtype

    transformer = SD3Transformer2DModel.from_pretrained(
        model_dir,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    # Place blocks immediately (no full-module .to one device).
    apply_pipeline_parallel(transformer, device0, device1)
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    # Ensure no leftover CPU parameter storage (`.to` can leave host refs until GC).
    cpu_left = sum(
        p.numel() * p.element_size()
        for p in transformer.parameters()
        if p.device.type == "cpu"
    )
    if cpu_left:
        print(f"  warning: {cpu_left / 1e9:.2f}GB transformer params still on CPU")
    gen.transformer = transformer
    del transformer
    gc.collect()
    _log_host_rss("after pipeline-place transformer")

    gen.scheduler = FlowMatchEulerDiscreteScheduler()
    gen.scheduler.set_timesteps(gen.scheduler.config.num_train_timesteps)
    # Tokenizer only needed for text generate demos — keep tiny/lazy. Stage-2
    # train never uses the text path; avoid allocating vocab×joint_dim (~2GB).
    gen.tokenizer = AutoTokenizer.from_pretrained(QWEN_TOKENIZER_ID)

    # VAE on CPU only (train uses cached latents). float16 weights keep host small.
    gen.vae = AutoencoderKL.from_pretrained(
        MMDIT_VAE_ID, subfolder="vae", torch_dtype=torch.float16
    )
    gen.vae = gen.vae.to("cpu").eval()
    for p in gen.vae.parameters():
        p.requires_grad_(False)
    gen.vae_scale_factor = 2 ** (len(gen.vae.config.block_out_channels) - 1)

    cfg = gen.transformer.config
    gen.in_channels = cfg.in_channels
    gen.joint_dim = cfg.joint_attention_dim
    gen.pooled_dim = cfg.pooled_projection_dim
    gen.sample_size = int(getattr(cfg, "sample_size", 64) or 64)

    import torch.nn as nn

    # Stub text path: 1-row embedding (not trained, not used in stage-2 recon).
    # Full Qwen vocab Embedding(vocab, joint_dim) alone is ~2+ GiB host.
    gen.token_embedding = nn.Embedding(1, gen.joint_dim)
    gen.text_pool = nn.Linear(gen.joint_dim, gen.pooled_dim)
    gen.embed_to_seq = nn.Linear(EMBED_DIM, gen.joint_dim).to(
        device=d0, dtype=compute_dtype
    )
    gen.embed_to_pool = nn.Linear(EMBED_DIM, gen.pooled_dim).to(
        device=d0, dtype=compute_dtype
    )

    if conditioning_path is None:
        parent = Path(model_dir).parent
        cand = parent / _CHF
        if cand.is_file():
            conditioning_path = str(cand)
        elif (Path(model_dir) / _CHF).is_file():
            conditioning_path = str(Path(model_dir) / _CHF)
    if conditioning_path:
        gen.load_conditioning_heads(conditioning_path)

    gen.freeze_non_stage2()
    gen.embed_to_seq.to(d0)
    gen.embed_to_pool.to(d0)
    gen._pp_devices = (device0, device1)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_host_rss("pipeline generator ready")
    if torch.cuda.is_available():
        for dev in (d0, d1):
            if dev.type == "cuda":
                alloc = torch.cuda.memory_allocated(dev) / (1024**3)
                reserved = torch.cuda.memory_reserved(dev) / (1024**3)
                print(
                    f"  VRAM cuda:{dev.index}: alloc {alloc:.2f}GiB "
                    f"reserved {reserved:.2f}GiB"
                )
    return gen


def verify_stage2_checkpoint(
    trained_dir: str | Path,
    *,
    vision_phase: int = 1,
    device: str | None = None,
) -> None:
    """Load stage2 weights and run one finite train step + generate."""
    trained_dir = Path(trained_dir)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    gen = MMDiTGenerator(
        model_dir=str(trained_dir / "mmdit")
        if (trained_dir / "mmdit").is_dir()
        else resolve_model_dir(0, "mmdit"),
        phase=2 if (trained_dir / "mmdit").is_dir() else 0,
        device=device,
        conditioning_path=str(trained_dir / CONDITIONING_HEADS_FILE)
        if (trained_dir / CONDITIONING_HEADS_FILE).is_file()
        else None,
    )
    gen.train()
    b, p, d = 1, 8, EMBED_DIM
    emb = F.normalize(torch.randn(b, p, d, device=device), dim=-1)
    # Use actual latent size from a tiny random if sample_size large
    lat_h = min(int(gen.sample_size), 32)
    lat = torch.randn(b, gen.in_channels, lat_h, lat_h, device=device)
    loss, _, _ = gen.forward_train(lat, emb)
    if not math.isfinite(float(loss)):
        raise RuntimeError(f"Stage-2 verify non-finite loss {loss}")
    gen.eval()
    img = gen.generate(
        embeddings=emb[0],
        height=lat_h * gen.vae_scale_factor,
        width=lat_h * gen.vae_scale_factor,
        num_inference_steps=2,
        seed=0,
        shuffle_embeddings=True,
    )
    print(f"Stage-2 verify OK: loss={float(loss):.4f} generated={img.size}")
