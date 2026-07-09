# AGENTS.md — TriSearch project rules

Instructions for humans and coding agents working in this repository.

## Mission

Build a split multimodal system (SigLIP vision embedder, Qwen3-MoE text embedder,
MMDiT generator) in a shared 1024-dim Matryoshka space. See `training_plan.md`
for the full multi-stage plan.

## Non-negotiable rules

### 1. No dataset scripts or remote code

- **Never** pass `trust_remote_code=True` to `load_dataset` or model loaders.
- **Never** rely on HuggingFace **dataset loading scripts** (removed in current
  `datasets` — they fail with “Dataset scripts are no longer supported”).
- Load HF data with **`streaming=True`** and materialize rows in
  `trisearch_dataset.stream_hf_rows()`.
- Prefer Parquet-native dataset repos. Defaults: ChatEarthNet (satellite,
  path-based images — requires `--satellite-image-root`), Flickr8k (general).

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

### 4. 8-bit training and loading

- Vision and text towers are trained and loaded in **8-bit** (bitsandbytes).
- Text tower goes through **Unsloth** (`FastLanguageModel`, AdamW8bit).
- Trained checkpoints are saved as `model.safetensors` + `config.json` per
  tower, plus `projection_heads.pt` at the stage root.
- **Do not** load trained 8-bit weights with `from_pretrained(trained_dir)`
  directly — that drops `.SCB` / `.weight_format` scales. Always use the
  seed-shell pattern in `trisearch_models`.

### 5. Architecture and module layout

Keep shared code in modules, not duplicated in scripts:

| Module | Responsibility |
|--------|----------------|
| `trisearch_models/` | Model constants, inference embedders, 8-bit loading, training losses, `Stage1AlignmentModel`, checkpoint I/O, optimizer helpers |
| `trisearch_dataset.py` | HF/JSONL loading, mixing, `ImageCaptionDataset`, sampling for eval/index |
| `trisearch_data_format.py` | Curated TriSearch dataset schema, 1024px resize, HF export/load |
| `generate_datasets.py` | Build curated Stage-1 dataset (COCO + SkyScript/RSICD, queries) |
| `view_dataset.py` | Gradio browser for curated dataset |
| `train_stage1.py` | Stage-1 CLI only — argparse + `main()` |
| `run_*.py` | Thin smoke/runner scripts |
| `demo_image_search.py` | Gradio retrieval UI over a real HF dataset sample |

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
# Stage 1 training (real data only)
python3 train_stage1.py --max-steps 10000 --batch-size 4

# Resume from models/trained/stage1/ (default)
python3 train_stage1.py --max-steps 20000

# Fresh run from seeds
python3 train_stage1.py --fresh --max-steps 10000

# Image search demo (curated mix or Flickr/COCO)
python3 demo_image_search.py --phase 1 --count 100 --rebuild-index

# Curated dataset (preview then inspect)
python3 generate_datasets.py --preview --skip-query-generation --allow-rsicd-fallback
python3 view_dataset.py --dataset-dir models/data/trisearch-v1

# Runner smoke tests
python3 run_siglip.py --phase 1
python3 run_qwen3.py --phase 1

# Dataset pipeline unit tests
python3 -m unittest tests.test_dataset_pipeline -v
```

## Checklist before submitting work

- [ ] No `trust_remote_code` and no dataset loading scripts
- [ ] No synthetic/fake/demo training data introduced
- [ ] Shared logic in `trisearch_models` or `trisearch_dataset`, not copy-pasted
- [ ] 8-bit load path uses seed shell + `load_state_dict` for trained weights
- [ ] Training or inference actually run and output captured
- [ ] Gradio UIs smoke-tested (`get_api_info` + launch/HTTP if applicable)
- [ ] `view_dataset.py` exercised on a real dataset dir when one exists
- [ ] Checkpoint verification passes (finite loss, no load warnings)
- [ ] Imports work: `python3 -c "from trisearch_models import ...; from trisearch_dataset import ..."`

## What not to do

- Do not commit `unsloth_compiled_cache/`, large checkpoints, or demo indexes
  unless explicitly asked.
- Do not dequantize 8-bit weights to fp32 for “convenience”.
- Do not add markdown docs the user did not ask for (this file is the
  exception — it was requested).