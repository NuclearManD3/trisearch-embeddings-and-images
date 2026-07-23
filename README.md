# TriSearch

**Split multimodal retrieval and generation** in a shared **1024-dimensional Matryoshka** embedding space.

| Component | Architecture | Role |
|-----------|--------------|------|
| **Vision embedder** | SigLIP-style (~1B) | Image → multi-vector (late-interaction) Matryoshka tokens |
| **Text embedder** | Qwen3-MoE (~1B) | Caption / query → multi-vector Matryoshka tokens |
| **Image generator** | MMDiT / DiT-style (~2B) | Embedding → image (reconstruction / text→image paths) |

Vision and text are trained into the **same** space with **ColBERT-style late interaction** (MaxSim), multi-caption positives, query tasks, memory-bank hard negatives, Matryoshka prefixes, and streamed text–text semantic training. Satellite and general imagery are first-class; the default Stage-1 path uses the curated **TriSearch 64k** corpus.

- **Author:** [NuclearManD](https://huggingface.co/NuclearManD) (aka NuclearManD3)
- **License:** [Apache License 2.0](LICENSE)
- **Curated data:** [`NuclearManD/trisearch-dataset-64k-v0.0.1`](https://huggingface.co/datasets/NuclearManD/trisearch-dataset-64k-v0.0.1)
- **Long-horizon plan:** [`training_plan.md`](training_plan.md)
- **Agent / contributor rules:** [`AGENTS.md`](AGENTS.md)

---

## Table of contents

1. [Why this design](#why-this-design)
2. [Repository layout](#repository-layout)
3. [Requirements](#requirements)
4. [Quick start](#quick-start)
5. [Data](#data)
6. [Models and checkpoints](#models-and-checkpoints)
7. [Stage-1 training](#stage-1-training)
8. [Stage-2 generator](#stage-2-generator)
9. [Inference and demos](#inference-and-demos)
10. [Important loading rules](#important-loading-rules)
11. [Tests](#tests)
12. [License and third-party data](#license-and-third-party-data)

---

## Why this design

- **Unified space:** retrieval and generation share one embedding geometry, not a bolted-on CLIP head + separate diffusion UNet vocabulary.
- **Combined Diffusion Model:** allows training on just image data for reinforcement learning, allowing use of much more training datasets which may not be labelled, and also garuntees that embeddings from the image embedder encode the image data - not just a description of it.
- **Late interaction:** multi-token MaxSim (hard or soft) instead of a single global cosine — better for fine-grained satellite and scene detail.
- **Matryoshka:** prefixes of the 1024-d vectors remain useful when truncated (e.g. 64 / 128 / 256 / 512).
- **Split towers:** vision and text live on separate GPUs by default; encodes can **overlap** on dual CUDA streams.
  - Allows training on two RTX 3060s

---

## Repository layout

```text
trisearch_models/          # Shared models, losses, checkpoints, Stage-2
  inference.py             # Embedders, loaders, phase resolution, heatmaps
  training.py              # Stage-1 alignment model, banks, train loop
  stage2.py                # MMDiT recon / conditioning helpers
trisearch_dataset.py       # Curated + HF loaders, multi-text, paraphrase queue
trisearch_data_format.py   # Curated schema / export
trisearch_demo_index.py    # Demo embedding index cache
train_stage1.py            # Stage-1 CLI
train_stage2.py            # Stage-2 CLI (freeze vision, train generator)
run_siglip.py / run_qwen3.py / run_mmdit.py
demo_image_search.py       # Gradio image search (real curated sample)
demo_stage2_recon.py
demo_stage2_text2img.py
generate_datasets.py       # Build curated dataset from source corpora
view_dataset.py            # Gradio dataset browser
model_seeding.py           # Architecture-aware seed from parent SigLIP / Qwen
create_*.py                # Seed model creation entrypoints
tests/                     # Unit tests (losses, math, datasets, stage2)
```

Large weights and caches live under `models/` (gitignored). Do not commit checkpoints, `unsloth_compiled_cache/`, or API keys.

---

## Requirements

### Software

- **Python 3.12+** recommended
- **PyTorch** with CUDA
- **Transformers**, **datasets**, **bitsandbytes**, **Unsloth**, **safetensors**, **Gradio** (demos), etc.

Typical env flags used in this project:

```bash
export UNSLOTH_COMPILE_DISABLE=1   # required for 8-bit SigLIP paths / Unsloth quirks
export TRANSFORMERS_VERBOSITY=error
```

### Hardware (tested)

| Setup | Notes |
|-------|--------|
| **2× NVIDIA RTX 3060 12GB** | Primary test platform: vision on `cuda:0`, text on `cuda:1` |
| Single GPU | Scripts clamp device indices; Stage-1 dual-stream overlap is disabled when both towers share one device |
| CUDA | Required for trained 8-bit inference and full dual-tower training |

Stage-1 training uses **full bf16/fp16** towers with **AdamW8bit**.

---

## Quick start

### 1. Seed models (once)

Architecture-aware transfer from parent SigLIP / Qwen into TriSearch-sized towers:

```bash
python3 create_siglip_model.py
python3 create_qwen3_moe_model.py
python3 create_mmdit_model.py   # generator seed for Stage 2+
```

Seeds are written under `models/siglip-vision/`, `models/qwen3-moe/`, `models/mmdit/`.

### 2. Smoke-test runners

```bash
# Latest trained checkpoint across stages (preferred), or seeds if none
python3 run_qwen3.py
python3 run_siglip.py --phase 1
```

### 3. Stage-1 train (curated 64k)

```bash
python3 train_stage1.py --max-steps 10000 --batch-size 4

# Resume from models/trained/stage1/ (default)
python3 train_stage1.py --max-steps 20000

# Fresh from seeds
python3 train_stage1.py --fresh --max-steps 10000
```

See the contents of `train_stage_1_cmd.sh` to see what commands I used that worked well.

### 4. Image search demo

```bash
python3 demo_image_search.py --phase 1 --count 100 --rebuild-index
# Prefer newest history/step-* snapshot:
python3 demo_image_search.py --latest-checkpoint --count 200 --rebuild-index
```

---

## Data

### Primary: curated TriSearch 64k

Default training and demos load:

**[`NuclearManD/trisearch-dataset-64k-v0.0.1`](https://huggingface.co/datasets/NuclearManD/trisearch-dataset-64k-v0.0.1)**

- Real image–caption pairs (general + satellite mix)
- Multiple captions / search queries where available
- Lazy / map-style access via Hugging Face `datasets` (no full-RAM decode)

Optional local export path: `models/data/trisearch-v1` (`--prefer-local-curated`).

```bash
# Browse a local export (if present)
python3 view_dataset.py --dataset-dir models/data/trisearch-v1
```

Build/regenerate curated data from source corpora with `generate_datasets.py` / `publish_dataset.py` (see those scripts for flags). Runtime training does **not** require OpenRouter if the curated set already includes queries.

### Text–text paraphrase mix (Stage-1)

In addition to image–text pairs, Stage-1 trains a **same-meaning paraphrase** loss from a **1k in-RAM queue** refilled from Hugging Face streams (never full materialization). Defaults are **paraphrase-only** (not query–answer retrieval):

| Source | Config | Weight | Role |
|--------|--------|--------|------|
| [`sentence-transformers/all-nli`](https://huggingface.co/datasets/sentence-transformers/all-nli) | `triplet` | 0.30 | Entailment positives + contradiction hard negatives (multi-genre) |
| [`humarin/chatgpt-paraphrases`](https://huggingface.co/datasets/humarin/chatgpt-paraphrases) | `default` | 0.25 | Explicit multi-paraphrase rewrites (broad topics) |
| [`sentence-transformers/coco-captions`](https://huggingface.co/datasets/sentence-transformers/coco-captions) | `pair` | 0.15 | Same-image caption pairs (everyday visual language) |
| [`sentence-transformers/flickr30k-captions`](https://huggingface.co/datasets/sentence-transformers/flickr30k-captions) | `pair` | 0.10 | Same-image caption pairs |
| [`sentence-transformers/quora-duplicates`](https://huggingface.co/datasets/sentence-transformers/quora-duplicates) | `pair` | 0.10 | Same intent, different wording |
| [`sentence-transformers/simple-wiki`](https://huggingface.co/datasets/sentence-transformers/simple-wiki) | `pair` | 0.05 | Wikipedia ↔ simple English (science, history, …) |
| [`sentence-transformers/sentence-compression`](https://huggingface.co/datasets/sentence-transformers/sentence-compression) | `pair` | 0.05 | Long ↔ short same meaning |

Negatives: **in-batch** (+ memory bank) always; **explicit hard negatives** mainly from AllNLI triplets.

```bash
# Default paraphrase mix
python3 train_stage1.py --paraphrase-dataset mix ...

# Custom mix
python3 train_stage1.py --paraphrase-sources \
  "sentence-transformers/all-nli:triplet:0.4,humarin/chatgpt-paraphrases:default:0.3,sentence-transformers/coco-captions:pair:0.3"

# Disable
python3 train_stage1.py --no-paraphrase ...
```

### Emergency / non-curated mixes

`--no-curated-dataset` falls back to other HF sources (e.g. satellite + general). Prefer the curated Hub set. Path-based satellite images may need `--satellite-image-root` or `--download-satellite-images`.

### Custom local data

```bash
python3 train_stage1.py --data-jsonl /path/to/pairs.jsonl ...
```

JSONL should follow the project schema (image paths + captions / queries). See `trisearch_data_format.py`.

---

## Models and checkpoints

### Phases

| Phase | Meaning | Typical path |
|-------|---------|----------------|
| **0** | Untrained seeds | `models/{siglip-vision,qwen3-moe,mmdit}/` |
| **1+** | Trained stages | `models/trained/stage{N}/` |

### Stage-1 checkpoint layout

```text
models/trained/stage1/
  vision_model/          # config + model.safetensors [+ preprocessor]
  text_model/            # config + model.safetensors + tokenizer files
  projection_heads.pt    # vision_projection + text_projection
  training_state.pt      # optimizer + global step
  stage1_config.json
  history/step-N/        # periodic snapshots (same layout)
```

`run_qwen3.py` defaults to the **newest valid checkpoint across stages** (stage5 → stage1, including `history/step-*` by mtime). Override with `--phase`, `--checkpoint-dir`, or `--model-dir`.

---

## Stage-1 training

### Objective (high level)

Multi-task mean of retrieval terms on shared embeddings:

1. **Caption(s) ↔ image** — multi-positive InfoNCE: every caption and related query for a sample is a positive for that image (capped by `--max-texts-per-image`, default 4)
2. **Query ↔ image** — search-style related query
3. **Query → caption** — with unrelated distractors
4. **Matryoshka** prefix CE on the same projected tokens
5. **Paraphrase / Q–A** text–text InfoNCE (streamed mix)
6. Optional **geometry** (anti-cone) and **heatmap sparsity** regularizers
7. Optional **gap hinge** on score margin for ranking

Scoring: soft or hard **MaxSim**, optional **score centering**, vision patch L2 keep / merge, dual-GPU **encode overlap** (vision stream ‖ fused text stream).

### Common commands

```bash
# Standard resume
python3 train_stage1.py --max-steps 10000 --batch-size 4

# Larger effective batch
python3 train_stage1.py --batch-size 32 --gradient-accumulation-steps 1 ...

# Ranking-oriented knobs (examples)
python3 train_stage1.py \
  --gap-loss-weight 1.5 --gap-margin 1.5 \
  --hard-bank-negatives 32 --bank-random-k 16 --bank-fn-margin 1.0 \
  --embedding-geo-weight 0.3 \
  --vision-merge-tokens 64
```

`--batch-size` must be **≥ 2** (in-batch negatives). Training DataLoader uses `drop_last=True`.

### Query enrichment (optional)

If rows lack related/unrelated queries, Stage-1 can call OpenRouter via a **local** `config.yml` (gitignored):

```yaml
openrouter:
  api_key: YOUR_KEY
  model: mistralai/mistral-nemo   # or another chat model
```

Not required when using the curated dataset with queries already present.

---

## Stage-2 generator

Freeze the vision embedder (or use a trained Stage-1 vision tower) and train MMDiT to **reconstruct** (and related paths) from vision embeddings.

```bash
python3 train_stage2.py --help
python3 demo_stage2_recon.py --help
python3 demo_stage2_text2img.py --help
```

See `trisearch_models/stage2.py` and `training_plan.md` Stage 2 for the intended freeze / recon / cycle design.

---

## Inference and demos

| Script | Purpose |
|--------|---------|
| `run_qwen3.py` | Interactive text embedder + MaxSim among previous queries |
| `run_siglip.py` | Interactive vision embedder smoke test |
| `run_mmdit.py` | Generator smoke test |
| `demo_image_search.py` | Gradio retrieval over a real curated sample (+ patch heatmaps) |
| `demo_stage2_recon.py` | Reconstruction demo |
| `demo_stage2_text2img.py` | Text-conditioned generation demo |

Heatmaps use **peak-relative** exponential contrast by default so high absolute cosine among non-matches stays cold; only near-peak patches heat up.

---

## Important loading rules

1. **Do not** load trained 8-bit tower weights with bare `from_pretrained(trained_dir)` if the checkpoint stores BnB int8 + `.SCB` scales — scales are dropped. Use project helpers: **seed shell + `load_state_dict`** (`load_siglip_backbone` / `load_qwen_backbone` in `trisearch_models`).
2. **New Stage-1 saves** may be full-precision safetensors; loaders still support legacy 8-bit via dequant into a float shell when resuming or serving.
3. **Unsloth** text MoE may store fused expert keys; reload through the project/Unsloth path, not a naive plain Transformers MoE load.
4. Never pass **`trust_remote_code=True`** for datasets in this project.
5. Never dump an entire multi-GB split into RAM as decoded images.

---

## Tests

```bash
python3 -m pytest tests/ -q
```

Key coverage: Stage-1 contrastive / multi-positive / bank / geo math, paraphrase queue helpers, Stage-2 unit tests, dataset pipeline tests.

---

## License and third-party data

- **Code:** Apache License 2.0 — see [LICENSE](LICENSE). Copyright 2026 NuclearManD.
- **Weights you train:** your responsibility under the licenses of base models (SigLIP, Qwen, Diffusers/MMDiT seeds) and of any datasets used.
- **Datasets:** respect Hub / original dataset licenses (TriSearch curated set, GooAQ, Natural Questions, AllNLI, COCO, SkyScript, etc.).

### Related docs

| File | Contents |
|------|----------|
| [`training_plan.md`](training_plan.md) | Multi-stage roadmap (alignment → generator → joint → RL → satellite) |
| [`AGENTS.md`](AGENTS.md) | Hard rules for contributors and coding agents |
| [`RELEASE_NOTES.md`](RELEASE_NOTES.md) | Changelog / checkpoint format notes |

### Citation

If you use TriSearch in academic work, please cite this repository.
