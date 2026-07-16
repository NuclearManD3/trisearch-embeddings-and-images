#!/usr/bin/env python3
"""
Stage 2: full MMDiT pretraining from frozen SigLIP patch embeddings.

Flow
----
1. **Precompute** embeddings (+ VAE latents) with **two vision copies**
   (one per GPU), writing float16 ``samples/{id}.safetensors`` as each wave
   finishes (resume-safe). Runs in a **subprocess** so host RSS is released
   before train. Large static tensors stay on disk — not host RAM.
2. **Unload** vision models (child exits; parent gets a clean process).
3. **Train** full MMDiT with transformer blocks **pipeline-split** across both
   GPUs. Default ``adamw8bit`` keeps Adam moments in VRAM (fast). Host RSS
   soft-target ≤ ~6GB; use ``disk_adamw`` only if VRAM cannot fit moments.

Run::

  python3 train_stage2.py --max-steps 1000 --batch-size 1
  python3 train_stage2.py --max-steps 4 --max-samples 8 --batch-size 1 --fresh
  python3 train_stage2.py --skip-precompute   # use existing embed cache only
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from pathlib import Path

import torch

from trisearch_dataset import DEFAULT_TRISEARCH_HF_DATASET
from trisearch_models import gpu_device
from trisearch_models.inference import CONDITIONING_HEADS_FILE, MMDIT_DIR, resolve_model_dir
from trisearch_models.stage2 import (
    DEFAULT_EMBED_CACHE_DIR,
    DEFAULT_EMBED_DROPOUT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_COND_TOKENS,
    DEFAULT_MERGE_PROB,
    DEFAULT_STAGE2_DIR,
    assert_stage2_train_cache_ok,
    build_stage2_cache_dataloader,
    build_stage2_optimizer,
    load_stage2_training_state,
    list_cached_sample_ids,
    precompute_stage2_embeddings,
    run_stage2_training,
    setup_pipeline_generator,
    verify_stage2_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-dataset", default=DEFAULT_TRISEARCH_HF_DATASET)
    p.add_argument("--curated-dataset-dir", default=None)
    p.add_argument("--prefer-local-curated", action="store_true")
    p.add_argument(
        "--curated-split", default="train", choices=("train", "test", "all")
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--satellite-fraction", type=float, default=0.5)

    p.add_argument("--vision-phase", type=int, default=1)
    p.add_argument(
        "--vision-checkpoint-dir",
        default=None,
        help="Optional Stage-1 root (vision_model + projection_heads.pt).",
    )
    p.add_argument(
        "--mmdit-seed-dir",
        default=None,
        help=f"Seed MMDiT dir (default {MMDIT_DIR}).",
    )
    p.add_argument("--trained-dir", default=DEFAULT_STAGE2_DIR)
    p.add_argument("--fresh", action="store_true", help="Ignore stage2 training resume.")

    p.add_argument(
        "--embed-cache-dir",
        default=DEFAULT_EMBED_CACHE_DIR,
        help="Directory for precomputed patch embeddings + VAE latents.",
    )
    p.add_argument(
        "--rebuild-embed-cache",
        action="store_true",
        help="Delete cached samples and re-encode from scratch.",
    )
    p.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Do not run precompute (require an existing complete cache).",
    )
    p.add_argument(
        "--precompute-batch-size",
        type=int,
        default=12,
        help="Images per GPU per dual-GPU precompute wave (default 12). "
        "Each GPU runs SigLIP+VAE independently; try 16–24 if VRAM allows "
        "(~6–7GB used of 12GB at batch 8).",
    )

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--dataloader-workers",
        type=int,
        default=2,
        help="DataLoader workers for embed-cache prefetch (0=main thread only). "
        "2 is a good default; each worker costs host RAM.",
    )
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--optimizer",
        choices=("adamw8bit", "disk_adamw", "cpu_adamw", "adamw", "sgd"),
        default="adamw8bit",
        help="Default adamw8bit: moments in VRAM (fast; use with pipeline split). "
        "disk_adamw: moments on disk memmap (slow, low host RSS).",
    )
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=250)

    p.add_argument(
        "--embed-dropout",
        type=float,
        default=DEFAULT_EMBED_DROPOUT,
        help=f"Token dropout on patch embeddings (default {DEFAULT_EMBED_DROPOUT}).",
    )
    p.add_argument(
        "--merge-prob",
        type=float,
        default=DEFAULT_MERGE_PROB,
        help=f"Prob. of collapsing all tokens to one mean (default {DEFAULT_MERGE_PROB}).",
    )
    p.add_argument(
        "--max-cond-tokens",
        type=int,
        default=DEFAULT_MAX_COND_TOKENS,
        help="After shuffle, keep at most this many patch tokens "
        f"(default {DEFAULT_MAX_COND_TOKENS}).",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"VAE latent spatial size target (default {DEFAULT_IMAGE_SIZE}; ×16).",
    )

    p.add_argument("--gpu0", type=int, default=0)
    p.add_argument("--gpu1", type=int, default=1)
    p.add_argument(
        "--device",
        default=None,
        help="Force single device (disables dual-GPU precompute + pipeline split).",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-train checkpoint verification.",
    )
    p.add_argument(
        "--precompute-only",
        action="store_true",
        help="Only run embed precompute then exit (child process so train starts "
        "with a clean host RSS).",
    )
    p.add_argument(
        "--inprocess-precompute",
        action="store_true",
        help="Run precompute in this process (default: spawn a child so host "
        "RSS from SigLIP is fully released before MMDiT load).",
    )
    p.add_argument(
        "--allow-tiny-cache",
        action="store_true",
        help="Allow training when the embed cache has <16 unique images "
        "(deliberate overfit smokes only). Default: refuse so Stage-2 does "
        "not silently memorize a 4–8 image smoke set.",
    )
    return p.parse_args()


def _spawn_precompute_subprocess(argv: list[str]) -> None:
    """Re-exec this script with --precompute-only in a child process.

    Dual SigLIP + VAE leave multi-GB host RSS even after del/gc; exiting the
    child returns that memory to the OS before train loads MMDiT.
    """
    script = str(Path(__file__).resolve())
    skip = {
        "--precompute-only",
        "--inprocess-precompute",
        "--skip-precompute",
        "--skip-verify",
    }
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in skip:
            i += 1
            continue
        cleaned.append(a)
        i += 1
    cmd = [sys.executable, script, *cleaned, "--precompute-only"]
    print("  precompute via subprocess (clean host RSS for train) ...")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"precompute subprocess failed with code {proc.returncode}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.image_size % 16 != 0:
        raise SystemExit("--image-size must be a multiple of 16")

    if args.device:
        dev0 = dev1 = args.device
        dual = False
    elif torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        dev0 = str(gpu_device(args.gpu0))
        dev1 = str(gpu_device(args.gpu1))
        dual = True
    elif torch.cuda.is_available():
        dev0 = dev1 = str(gpu_device(0))
        dual = False
    else:
        dev0 = dev1 = "cpu"
        dual = False

    print("--- Stage 2: full MMDiT pretrain (precomputed embeds) ---")
    print(f"  devices       : {dev0}" + (f" + {dev1} (dual)" if dual else ""))
    print(f"  embed cache   : {args.embed_cache_dir}")
    print(f"  embed dropout : {args.embed_dropout}")
    print(f"  merge prob    : {args.merge_prob}")
    print(f"  max cond toks : {args.max_cond_tokens}")
    print(f"  image size    : {args.image_size}")
    print(f"  trained dir   : {args.trained_dir}")

    cache_dir = Path(args.embed_cache_dir)

    # ----- 1) Dual-GPU precompute (resume-safe) -----
    if args.precompute_only:
        precompute_stage2_embeddings(
            args=args,
            cache_dir=cache_dir,
            devices=[dev0, dev1] if dual else [dev0],
            batch_size=args.precompute_batch_size,
            image_size=args.image_size,
            rebuild=args.rebuild_embed_cache,
        )
        print("  --precompute-only: done; exiting.")
        return

    if not args.skip_precompute:
        if args.inprocess_precompute:
            precompute_stage2_embeddings(
                args=args,
                cache_dir=cache_dir,
                devices=[dev0, dev1] if dual else [dev0],
                batch_size=args.precompute_batch_size,
                image_size=args.image_size,
                rebuild=args.rebuild_embed_cache,
            )
        else:
            _spawn_precompute_subprocess(sys.argv[1:])
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        n = len(list_cached_sample_ids(cache_dir))
        if n == 0:
            raise SystemExit(
                f"--skip-precompute but no samples under {cache_dir}/samples/"
            )
        print(f"  using existing embed cache ({n:,} samples)")

    # ----- 2) Load pipeline-split generator (vision already unloaded) -----
    trained = Path(args.trained_dir)
    if not args.fresh and (trained / "mmdit").is_dir():
        mmdit_dir = str(trained / "mmdit")
        cond_path = (
            str(trained / CONDITIONING_HEADS_FILE)
            if (trained / CONDITIONING_HEADS_FILE).is_file()
            else None
        )
        print(f"  resume mmdit  : {mmdit_dir}")
    else:
        mmdit_dir = args.mmdit_seed_dir or resolve_model_dir(0, "mmdit")
        cond_path = None
        print(f"  seed mmdit    : {mmdit_dir}")

    if dual:
        generator = setup_pipeline_generator(
            model_dir=mmdit_dir,
            conditioning_path=cond_path,
            device0=dev0,
            device1=dev1,
        )
    else:
        from trisearch_models import MMDiTGenerator

        generator = MMDiTGenerator(
            model_dir=mmdit_dir,
            phase=2 if cond_path else 0,
            device=dev0,
            conditioning_path=cond_path,
        )
        generator.freeze_non_stage2()
        generator.vae.to("cpu")

    n_train = sum(
        p.numel() for p in generator.trainable_parameters() if p.requires_grad
    )
    print(f"  trainable     : {n_train / 1e6:.2f}M params (full MMDiT, no LoRA)")

    optim_dir = Path(args.trained_dir) / "optim_disk"
    optimizer = build_stage2_optimizer(
        generator,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        state_dir=optim_dir,
    )
    print(f"  optimizer     : {args.optimizer} (state_dir={optim_dir})")
    start_step = 0
    if not args.fresh and (trained / "training_state.pt").is_file():
        start_step = load_stage2_training_state(trained, optimizer)
        print(f"  resume step   : {start_step}")

    loader = build_stage2_cache_dataloader(args, cache_dir)
    assert_stage2_train_cache_ok(
        cache_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        allow_tiny_cache=bool(args.allow_tiny_cache),
    )
    print(
        f"  micro-batch   : {args.batch_size} × accum {args.gradient_accumulation_steps}"
    )

    final = run_stage2_training(
        generator=generator,
        dataloader=loader,
        optimizer=optimizer,
        args=args,
        start_step=start_step,
    )
    print(f"Training finished at step {final}")

    if not args.skip_verify:
        verify_stage2_checkpoint(
            args.trained_dir, vision_phase=args.vision_phase, device=dev0
        )


if __name__ == "__main__":
    main()
