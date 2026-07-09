# Release notes

## 0.0.1 — 2026-07-09

Initial public packaging of TriSearch Stage-1 tooling (SigLIP vision + Qwen3-MoE
text embedders in a shared 1024-dim Matryoshka space, with MMDiT generator seeds
and runners).

### Bug fixes

- **Training mode was forced off after backbone load.** `load_siglip_backbone` /
  `load_qwen_backbone` always called `model.eval()` even with `for_training=True`,
  undoing Unsloth `for_training()` and leaving dropout / train-time behaviour
  inactive until a later recursive `train()`. Loaders now leave models in
  `train()` when requested and `eval()` only for inference.
- **`--max-steps` counted extra steps on resume.** Resume used
  `max_steps + start_step`, so a second run with `--max-steps 10000` after
  reaching step 10000 would train to 20000. It now matches the HuggingFace
  Trainer convention: `--max-steps` is a **total** global-step budget.
- **Text–text Matryoshka loss never reset in the training log.** After each log
  line the code assigned `log_text_matryoshka = 0.0` (undefined name) instead of
  clearing `log_text_text_matryoshka`, so the logged `text_text_m` metric drifted
  after the first logging interval.
- **Checkpoints omitted HuggingFace `quantization_config`.** Saves copied the
  full-precision seed `config.json`, so 8-bit `model.safetensors` (with `.SCB` /
  `.weight_format`) were not described as BitsAndBytes Int8. External tools
  (Transformers, Unsloth Studio, Hub UI) could not recognize the format.
  Saves now write a cleaned, JSON-serializable `quantization_config` and correct
  `dtype`.
- **Unsloth config broke `save_pretrained`.** Unsloth injects a non-serializable
  `get_loading_attributes` lambda into `BitsAndBytesConfig`; checkpoint writing
  now strips callables and always emits valid `config.json` + safetensors.
- **Checkpoints lacked tokenizer / image-processor files.** Text towers had no
  `tokenizer*.json` / `generation_config.json`; vision towers had no
  `preprocessor_config.json`. Stage-1 saves (and existing on-disk stage-1 trees)
  now ship a standard HF directory layout so external programs can resolve the
  repo without separate baseline IDs.
- **Runner scripts defaulted to CPU.** `run_siglip.py`, `run_qwen3.py`, and
  `run_mmdit.py` never set a device, so phase ≥ 1 (bitsandbytes 8-bit) failed or
  was unusable. They now default to `cuda:0` when CUDA is available and accept
  `--device`.
- **Demo GPU indices were not clamped.** `demo_image_search.py` only clamped the
  text GPU; an out-of-range `--vision-gpu` could crash on single-GPU hosts.
  Both indices clamp safely. Training `gpu_device()` also falls back when the
  default text GPU index is past `device_count`.
- **OpenRouter config required even with a full query cache.**
  `enrich_rows_with_text_queries` always loaded `config.yml`; training with a
  complete cache (or offline) no longer needs API credentials.
- **MMDiT / Diffusers weight detection only looked for `model.safetensors`.**
  Seed and Diffusers checkpoints use `diffusion_pytorch_model.safetensors`.
  Phase resolution and availability checks now accept HF and Diffusers weight
  filenames (and sharded index files).
- **`MMDiTGenerator` did not normalize device strings** through the same CUDA
  validation path as the embedders.
- **Projection / demo caches used bare `torch.load`.** Projection heads load
  with `weights_only=True` when supported; demo index loads keep
  `weights_only=False` (JPEG payloads) but tolerate older PyTorch signatures.

### Checkpoint format (Stage 1)

Each save under `models/trained/stage1/` now looks like:

```text
models/trained/stage1/
  vision_model/
    config.json              # includes quantization_config for 8-bit
    model.safetensors        # BnB Int8 weights + SCB scales
    preprocessor_config.json # image size matched to tower (e.g. 540)
  text_model/
    config.json              # includes quantization_config for 8-bit
    model.safetensors        # Unsloth/BnB trained weights
    tokenizer_config.json
    tokenizer.json
    generation_config.json
    chat_template.jinja      # when provided by the tokenizer
  projection_heads.pt        # vision_projection + text_projection
  training_state.pt          # optimizer + global_step
  stage1_config.json         # CLI args snapshot
  history/step-N/            # periodic snapshots (same layout)
```

**Loading note (unchanged project rule):** trained 8-bit weights must be applied
via a seed shell + `load_state_dict` (`load_siglip_backbone` /
`load_qwen_backbone`). Do not `from_pretrained(trained_dir)` alone for the
8-bit towers — BnB scale keys (`.SCB`) are dropped that way.

**Text tower note:** after Unsloth load, MoE experts may be stored in fused
form (`experts.gate_up_proj` / `experts.down_proj`). That is the on-disk format
produced by Stage-1 training and is reloaded correctly through Unsloth + the
project loaders. Plain Transformers `Qwen3MoeForCausalLM.from_pretrained` on
the trained text dir may not map fused expert keys without Unsloth.

### Compatibility

- Verified: save writes JSON-valid configs with `quantization_config.load_in_8bit`.
- Verified: `for_training=True` leaves models in train mode; inference leaves them in eval.
- Verified: trained Stage-1 vision and text reload via seed-shell loaders with no unexpected keys.
- Existing Stage-1 trees under `models/trained/stage1/` (including `history/`)
  were backfilled with `quantization_config`, tokenizers, and preprocessors.

### Known limitations

- Stage 2–5 (MMDiT training, full multi-stage curriculum) are not part of this
  release; MMDiT runners still use resized seed weights.
- Dual-GPU layout (vision `cuda:0`, text `cuda:1`) remains the preferred
  training setup; single-GPU falls back to one device for both towers.
