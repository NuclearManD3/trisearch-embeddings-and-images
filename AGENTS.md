# AGENTS.md — TriSearch project rules

Instructions for humans and coding agents working in this repository.

## Mission

Build a split multimodal system (SigLIP vision embedder, Qwen3-MoE text embedder,
MMDiT generator) in a shared 1024-dim Matryoshka space. See `training_plan.md`
for the full multi-stage plan.

## Non-negotiable rules

### 0. Never load a full multi‑gig dataset into RAM

**Etched in stone. Do not violate.**

- **Never** materialize an entire multi‑GB image corpus as a Python `list` of
  PIL images (or equivalent).
- **Never** invent a second on-disk cache of training images/models when the
  Hugging Face Hub / `datasets` / transformers cache already holds them.
- Training and demos must use **map-style / lazy** access: `ds[i]` decodes
  **one** example on demand (HF Arrow/parquet mmap or local sidecars).
- Small samples only (`max_samples` / demo `count` ≪ full size) may be held as
  short lists. Full `train` (~10 GB class) must stay on disk via HF cache.
- Prefer `load_dataset("org/name", split=...)` + index select; do **not**
  `for row in ds: rows.append(decode(row))` over the full split.

### 1. No dataset scripts or remote code

- **Never** pass `trust_remote_code=True` to `load_dataset` or model loaders.
- **Never** rely on HuggingFace **dataset loading scripts** (removed in current
  `datasets` — they fail with “Dataset scripts are no longer supported”).
- Prefer Parquet-native Hub repos and the official curated dataset
  (`NuclearManD/trisearch-dataset-64k-v0.0.1`). Streaming is for **bounded**
  samples only — not a full-pass over 60k+ rows into a list.
- **Runtime data is TriSearch curated only** (train / demo / view / verify).
  COCO, SkyScript, Flickr, ChatEarthNet appear **only** in
  `generate_datasets.py` (and optional `--no-curated-dataset` emergency path).

### 2. No fake data

- **Never** add synthetic, placeholder, or randomly generated training data.
- **Never** default to demo/synthetic datasets when real data is available.
- Training and verification must use **real** image–caption pairs from:
  - HuggingFace datasets (e.g. ChatEarthNet, COCO, Flickr8k), or
  - Local JSONL with paths to real images.
- Dataset loading lives in `trisearch_dataset.py`. Extend that module; do not
  inline ad-hoc loaders in scripts.

### 3. Verify before you report done

- Do not claim a change works without running it.
- Training scripts must end with `verify_trained_checkpoint()` (loads weights,
  runs a real-data forward pass, checks finite loss).
- Inference must load 8-bit trained checkpoints via **seed shell +
  `load_state_dict`** (see `trisearch_models.loading` / `load_*_backbone`).
  There must be **no** `UNEXPECTED` keys in the load report for trained weights
  (implemented in `trisearch_models.inference`: `load_siglip_backbone`,
  `load_qwen_backbone`).
- Demos and runners must be executed, not just described.
- **UI / Gradio tools must be smoke-tested after any change** — not only
  imported. At minimum:
  - Launch (or call `get_api_info()` / first-request path) and confirm no
    traceback; for long-running servers, start them, hit the app once if
    practical, then stop.
  - Dataset tools: run against a real on-disk path when present
    (e.g. `python3 view_dataset.py --dataset-dir models/data/trisearch-v1`
    after a preview export exists).
  - Unit tests alone are not enough for Gradio event wiring (State/Number
    handlers can pass imports and still crash in the browser).
- If GPUs are unavailable, say so explicitly and run what you can (imports,
  dataset parsing, CPU-only paths).

### 4. Training precision and loading

- Stage-1 **trains** vision and text towers in **full bf16/fp16** (all
  parameters have `requires_grad`) via Unsloth `full_finetuning=True` for
  text; SigLIP loads without weight quantization.
- Optimizer is **AdamW8bit** (8-bit moments only — not weight quant).
- **Inference** may still load older **8-bit** checkpoints (bnb) with the
  seed-shell + `load_state_dict` pattern. New training saves are float
  safetensors (no `.SCB`).
- Resuming a **legacy 8-bit** train checkpoint dequantizes int8+SCB into the
  float shell (same architecture required).

### 5. Architecture and module layout

Keep shared code in modules, not duplicated in scripts:

| Module | Responsibility |
|--------|----------------|
| `trisearch_models/` | Model constants, inference embedders, 8-bit loading, training losses, `Stage1AlignmentModel`, Stage-2 MMDiT recon, checkpoint I/O, optimizer helpers |
| `trisearch_dataset.py` | HF/JSONL loading, mixing, `ImageCaptionDataset`, sampling for eval/index |
| `trisearch_data_format.py` | Curated TriSearch dataset schema, 1024px resize, HF export/load |
| `trisearch_demo_index.py` | Shared demo embedding cache/index (stage-1 search + stage-2 recon) |
| `generate_datasets.py` | Build curated Stage-1 dataset (COCO + SkyScript/RSICD, queries) |
| `view_dataset.py` | Gradio browser for curated dataset |
| `train_stage1.py` | Stage-1 CLI only — argparse + `main()` |
| `train_stage2.py` | Stage-2 CLI — freeze vision, train MMDiT recon from patch embeddings |
| `run_*.py` | Thin smoke/runner scripts |
| `demo_image_search.py` | Gradio retrieval UI over a real HF dataset sample |
| `demo_stage2_recon.py` | Gradio caption search + original \| generated recon (shuffled cond) |
| `demo_stage2_text2img.py` | Gradio text query → Qwen embeds → Stage-2 multi-image generation |

Scripts should import from these modules. If you add logic used in more than
one place, move it to the appropriate module first.

### 6. Dual-GPU layout

- Default: vision on `cuda:0`, text on `cuda:1`.
- Contrastive loss is computed on the vision GPU; text activations are moved
  across the bus inside `Stage1AlignmentModel`.

### 7. Training phases

- **Phase 0**: untrained seed weights under `models/{siglip-vision,qwen3-moe,mmdit}/`.
- **Phase 1+**: trained checkpoints under `models/trained/stage{N}/`.
- Runner scripts accept `--phase {0-5}` via `trisearch_models.resolve_model_dir`.

### 8. Stage 1 data mix

Per `training_plan.md`, stage 1 uses a **50/50 satellite/general** mix unless
overridden. Defaults:

- Satellite: `JessicaYuan/ChatEarthNet`
- General: `jxie/flickr8k` (embedded PIL images; caption column `caption_0`)

ChatEarthNet stores **filename paths** in the `image` column. Before training
starts, `load_stage1_training_rows()` resolves PNGs and validates sample loads:

- `--satellite-image-root /path/to/pngs`, or
- `--download-satellite-images` (caches `s2_rgb_images.zip` under
  `models/data/ChatEarthNet/`), or
- PNGs already present in `models/data/ChatEarthNet/s2_rgb_images/`

Training must **fail fast** during dataset load — not at the first batch step.

Use `--data-jsonl` for custom local corpora.

### 9. Contrastive training constraints

- `--batch-size` must be **>= 2** (in-batch negatives).
- Use `drop_last=True` on the training DataLoader.
- Projection heads are freshly initialized (Xavier); they need a higher LR
  (`--projection-learning-rate`, default `1e-4`).

## Environment

```bash
export UNSLOTH_COMPILE_DISABLE=1   # required for 8-bit SigLIP
export TRANSFORMERS_VERBOSITY=error  # optional, reduces HF noise
```

Hardware: tested on 2× RTX 3060 12GB.

## Common commands

```bash
# Stage 1 training (defaults: B=4×accum8, geo=0.4, heat_sparse=0, bank clear 250)
python3 train_stage1.py --fresh --max-steps 5000

# Smoke train (small Hub sample)
python3 train_stage1.py --max-steps 4 --max-satellite-samples 8 --max-general-samples 8 \
  --skip-query-generation --batch-size 2

# Resume from models/trained/stage1/ (default when not --fresh)
python3 train_stage1.py --max-steps 5000

# Image search demo (TriSearch curated only; lazy map — batch_size caps image RAM)
python3 demo_image_search.py --phase 1 --count 1000 --batch-size 4 --rebuild-index
# Use newest history/step-* training snapshot instead of completed stage1/
python3 demo_image_search.py --latest-checkpoint --count 200 --rebuild-index

# Dataset viewer (Hub curated; optional local export)
python3 view_dataset.py --max-load 64
python3 view_dataset.py --prefer-local --dataset-dir models/data/trisearch-v1

# Runner smoke tests
python3 run_siglip.py --phase 1
python3 run_qwen3.py --phase 1

# Stage 2: dual-GPU embed precompute (disk cache) + pipeline-parallel MMDiT train
# Host RSS soft-target ~6GB; default adamw8bit puts moments in VRAM.
# IMPORTANT: do not train on a smoke cache (--max-samples 8). That memorizes ~8 images.
# Precompute full (or large) cache first; train refuses caches with <16 samples unless --allow-tiny-cache.
python3 train_stage2.py --precompute-only --embed-cache-dir models/data/stage2_embed_cache
python3 train_stage2.py --max-steps 10000 --batch-size 1 --skip-precompute \
  --embed-cache-dir models/data/stage2_embed_cache
python3 train_stage2.py --max-steps 2 --max-samples 8 --batch-size 1 --fresh --skip-verify \
  --allow-tiny-cache  # deliberate overfit smoke only

# Stage-2 recon demo (shared embedding cache with demo_image_search)
python3 demo_stage2_recon.py --count 100 --generator-dir models/trained/stage2

# Stage-2 text→image (Qwen token embeds → MMDiT; multi-seed gallery)
python3 demo_stage2_text2img.py --generator-dir models/trained/stage2
python3 demo_stage2_text2img.py --smoke  # one tiny gen + api check, then exit

# Dataset / stage-2 unit tests
python3 -m unittest tests.test_dataset_pipeline -v
python3 -m unittest tests.test_stage2 -v
```

## Checklist before submitting work

- [ ] No `trust_remote_code` and no dataset loading scripts
- [ ] No synthetic/fake/demo training data introduced
- [ ] Shared logic in `trisearch_models` or `trisearch_dataset`, not copy-pasted
- [ ] Training uses full-precision towers + AdamW8bit; inference 8-bit uses seed shell
- [ ] Training or inference actually run and output captured
- [ ] Gradio UIs smoke-tested (`get_api_info` + launch/HTTP if applicable)
- [ ] `view_dataset.py` exercised on a real dataset dir when one exists
- [ ] Checkpoint verification passes (finite loss, no load warnings)
- [ ] Imports work: `python3 -c "from trisearch_models import ...; from trisearch_dataset import ..."`

## What not to do

- Do not commit `unsloth_compiled_cache/`, large checkpoints, or demo indexes
  unless explicitly asked.
- Do not re-introduce weight-only 8-bit training (int8 `requires_grad` is not
  supported; towers will not update).
- Do not add markdown docs the user did not ask for (this file is the
  exception — it was requested).