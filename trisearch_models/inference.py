#!/usr/bin/env python3
"""
Utility library that wraps the three resized models under ``models/`` in small,
easy-to-use classes for the runner (smoke-test) scripts:

- ``SiglipEmbedder``  : image  -> list of Matryoshka (1024-dim) patch embeddings.
- ``Qwen3MoeEmbedder``: text   -> list of Matryoshka (1024-dim) token embeddings,
                        with optional merging of similar consecutive embeddings.
- ``MMDiTGenerator``  : text *or* embeddings -> a generated image. Text is
                        tokenized like a normal text-to-image pipeline; external
                        embeddings are pushed through a small transform stage
                        that maps them to the transformer's conditioning shape.

All embeddings are Matryoshka embeddings of ``EMBED_DIM`` (1024) dimensions:
they are L2-normalized so that a prefix of any length is itself a usable
embedding, and so that a dot product equals cosine similarity.

Phase 0 uses resized seed weights; phase >= 1 loads trained 8-bit checkpoints
from ``models/trained/stage{N}/``. These wrappers are shared by runner scripts,
training verification, and retrieval demos.
"""

import warnings

import torch
import torch.nn.functional as F
from torch import nn

warnings.filterwarnings("ignore")

# Every embedding this project produces is a 1024-dim Matryoshka embedding.
EMBED_DIM = 1024

# Defaults: where each model lives on disk and which checkpoint supplies the
# (light) tokenizer / image-processor that the model dir itself does not carry.
SIGLIP_DIR = "models/siglip-vision"
QWEN_DIR = "models/qwen3-moe"
MMDIT_DIR = "models/mmdit"
TRAINED_ROOT = "models/trained"

SIGLIP_PROCESSOR_ID = "google/siglip-so400m-patch14-384"
QWEN_TOKENIZER_ID = "Qwen/Qwen3-1.7B"

# Training phases 0 (untrained seeds) through 5 (final phase, not yet complete).
MIN_TRAINING_PHASE = 0
MAX_TRAINING_PHASE = 5

_COMPONENT_KEYS = {
    "siglip": "vision_model",
    "qwen": "text_model",
    "mmdit": "mmdit",
}
_SEED_DIRS = {
    "siglip": SIGLIP_DIR,
    "qwen": QWEN_DIR,
    "mmdit": MMDIT_DIR,
}
_PROJECTION_STATE_KEYS = {
    "siglip": "vision_projection",
    "qwen": "text_projection",
}
# The MMDiT model dir holds only the transformer; the VAE that decodes its
# latents back to full-resolution pixels comes from the SD3 baseline.
MMDIT_VAE_ID = "v2ray/stable-diffusion-3-medium-diffusers"


# ============================ SHARED HELPERS ============================

def matryoshka_normalize(embeddings, dim=None):
    """L2-normalize embeddings (optionally truncated to a Matryoshka prefix).

    embeddings : tensor of shape (..., D).
    dim        : if given, keep only the first ``dim`` components before
                 normalizing (a Matryoshka prefix). Defaults to the full width.
    """
    if dim is not None:
        embeddings = embeddings[..., :dim]
    return F.normalize(embeddings, p=2, dim=-1)


def late_interaction_score(query_embeddings, doc_embeddings):
    """ColBERT-style MaxSim late-interaction score between two token sets.

    query_embeddings / doc_embeddings : (n, D) / (m, D) tensors of normalized
    embeddings. For each query token we take its best-matching document token
    (max over docs), then **mean** over query tokens (mean-MaxSim).

    Mean (not sum) keeps scores in roughly ``[-1, 1]`` regardless of query
    length, which stabilizes contrastive CE and makes scores comparable across
    captions of different lengths. Ranking of docs for a *fixed* query is
    identical to sum-MaxSim.
    """
    q = _stack(query_embeddings).to(torch.float32)
    d = _stack(doc_embeddings).to(torch.float32)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if d.ndim == 1:
        d = d.unsqueeze(0)
    if q.numel() == 0 or d.numel() == 0:
        return 0.0
    sim = q @ d.T  # (n, m); dot product == cosine because inputs are normalized.
    return sim.max(dim=1).values.mean().item()


def _stack(embeddings):
    """Turn a list of 1-D embeddings into a single (n, D) tensor."""
    if isinstance(embeddings, torch.Tensor):
        return embeddings
    return torch.stack([torch.as_tensor(e, dtype=torch.float32) for e in embeddings])


def patch_query_affinity(
    query_embeddings,
    patch_embeddings,
    *,
    reduce: str = "max",
) -> torch.Tensor:
    """Per-patch affinity to a multi-token query (for spatial heatmaps).

    Parameters
    ----------
    query_embeddings
        List or ``(n_q, D)`` of L2-normalized query token embeddings.
    patch_embeddings
        List or ``(n_p, D)`` of L2-normalized image patch embeddings (raster
        order from the vision tower).
    reduce
        How to collapse the query dimension:
        * ``"max"`` (default) — for each patch, best cosine over query tokens
          (answers “how well does any query part hit this region?”).
        * ``"mean"`` — mean cosine over query tokens.

    Returns
    -------
    ``(n_p,)`` float32 tensor of cosine affinities in roughly ``[-1, 1]``.
    """
    q = _stack(query_embeddings).to(torch.float32)
    p = _stack(patch_embeddings).to(torch.float32)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if p.ndim == 1:
        p = p.unsqueeze(0)
    if q.numel() == 0 or p.numel() == 0:
        return torch.zeros(p.shape[0] if p.ndim > 0 else 0, dtype=torch.float32)
    sim = q @ p.T  # (n_q, n_p)
    reduce = (reduce or "max").lower()
    if reduce == "mean":
        return sim.mean(dim=0)
    if reduce != "max":
        raise ValueError(f"reduce must be 'max' or 'mean', got {reduce!r}")
    return sim.max(dim=0).values


def infer_patch_grid(num_patches: int, grid_hw: tuple[int, int] | None = None) -> tuple[int, int]:
    """Resolve ``(H, W)`` patch grid for a flat raster of ``num_patches`` tokens."""
    if grid_hw is not None:
        gh, gw = int(grid_hw[0]), int(grid_hw[1])
        if gh * gw != num_patches:
            raise ValueError(
                f"grid_hw={grid_hw} has {gh * gw} cells but num_patches={num_patches}"
            )
        return gh, gw
    side = int(round(num_patches ** 0.5))
    if side * side != num_patches:
        raise ValueError(
            f"Cannot infer square patch grid from num_patches={num_patches}; "
            "pass grid_hw=(H, W) explicitly."
        )
    return side, side


def patch_affinity_grid(
    query_embeddings,
    patch_embeddings,
    *,
    grid_hw: tuple[int, int] | None = None,
    reduce: str = "max",
) -> torch.Tensor:
    """``patch_query_affinity`` reshaped to ``(grid_h, grid_w)``."""
    scores = patch_query_affinity(
        query_embeddings, patch_embeddings, reduce=reduce
    )
    gh, gw = infer_patch_grid(int(scores.numel()), grid_hw=grid_hw)
    return scores.reshape(gh, gw)


def _jet_colormap_rgb(t: float) -> tuple[int, int, int]:
    """Classic jet-like RGB for ``t`` in ``[0, 1]`` (no matplotlib dependency)."""
    t = max(0.0, min(1.0, float(t)))
    # Piecewise linear approximation of jet.
    if t < 0.25:
        r, g, b = 0.0, 4.0 * t, 1.0
    elif t < 0.5:
        r, g, b = 0.0, 1.0, 1.0 - 4.0 * (t - 0.25)
    elif t < 0.75:
        r, g, b = 4.0 * (t - 0.5), 1.0, 0.0
    else:
        r, g, b = 1.0, 1.0 - 4.0 * (t - 0.75), 0.0
    return (
        int(max(0, min(255, round(r * 255)))),
        int(max(0, min(255, round(g * 255)))),
        int(max(0, min(255, round(b * 255)))),
    )


def overlay_patch_heatmap(
    image,
    patch_scores,
    *,
    grid_hw: tuple[int, int] | None = None,
    alpha: float = 0.5,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
):
    """Overlay a patch-affinity heatmap on a PIL image (center-crop square space).

    ``patch_scores`` is a flat ``(n_p,)`` or ``(H, W)`` tensor of affinities.
    Scores are robustly normalized with percentiles then mapped with a jet
    colormap and alpha-blended onto ``image``.

    Returns a new RGB ``PIL.Image``.
    """
    from PIL import Image as PILImage

    if not isinstance(image, PILImage.Image):
        raise TypeError(f"image must be PIL.Image, got {type(image)}")
    scores = torch.as_tensor(patch_scores, dtype=torch.float32).detach().cpu()
    if scores.ndim == 1:
        gh, gw = infer_patch_grid(int(scores.numel()), grid_hw=grid_hw)
        grid = scores.reshape(gh, gw)
    elif scores.ndim == 2:
        grid = scores
        if grid_hw is not None and tuple(grid.shape) != (
            int(grid_hw[0]),
            int(grid_hw[1]),
        ):
            raise ValueError(
                f"2-D patch_scores shape {tuple(grid.shape)} != grid_hw={grid_hw}"
            )
    else:
        raise ValueError(f"patch_scores must be 1-D or 2-D, got shape {tuple(scores.shape)}")

    flat = grid.reshape(-1)
    if flat.numel() == 0:
        return image.convert("RGB")
    lo = float(torch.quantile(flat, percentile_low / 100.0))
    hi = float(torch.quantile(flat, percentile_high / 100.0))
    if hi <= lo:
        lo = float(flat.min())
        hi = float(flat.max())
        if hi <= lo:
            hi = lo + 1e-6
    norm = ((grid - lo) / (hi - lo)).clamp(0.0, 1.0)

    gh, gw = int(norm.shape[0]), int(norm.shape[1])
    heat = PILImage.new("RGB", (gw, gh))
    px = heat.load()
    for y in range(gh):
        for x in range(gw):
            px[x, y] = _jet_colormap_rgb(float(norm[y, x]))

    base = image.convert("RGB")
    heat_up = heat.resize(base.size, PILImage.Resampling.BILINEAR)
    a = max(0.0, min(1.0, float(alpha)))
    return PILImage.blend(base, heat_up, a)


def _valid_config_path(path):
    import json
    from pathlib import Path

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            json.load(fh)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def trained_dir_for_phase(phase):
    """Return the checkpoint root for a training phase (phase >= 1)."""
    from pathlib import Path

    if phase < 1:
        raise ValueError(f"trained_dir_for_phase expects phase >= 1, got {phase}")
    return Path(TRAINED_ROOT) / f"stage{phase}"


def _component_has_weights(component_dir) -> bool:
    """True when a HF/transformers or Diffusers weight file is present."""
    from pathlib import Path

    component_dir = Path(component_dir)
    for name in (
        "model.safetensors",
        "pytorch_model.bin",
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
    ):
        if (component_dir / name).is_file():
            return True
    # Sharded HF checkpoints.
    if (component_dir / "model.safetensors.index.json").is_file():
        return True
    if (component_dir / "pytorch_model.bin.index.json").is_file():
        return True
    return False


def phase_checkpoint_available(phase, component=None):
    """True when the requested training-phase checkpoint exists on disk."""
    from pathlib import Path

    if phase == 0:
        return True
    if not MIN_TRAINING_PHASE <= phase <= MAX_TRAINING_PHASE:
        return False

    root = trained_dir_for_phase(phase)
    if component is None:
        for key in ("siglip", "qwen"):
            comp = root / _COMPONENT_KEYS[key]
            if not _component_has_weights(comp) or not _valid_config_path(comp / "config.json"):
                return False
        projection = root / "projection_heads.pt"
        return projection.is_file()

    if component == "mmdit":
        mmdit_dir = root / _COMPONENT_KEYS["mmdit"]
        return (
            _valid_config_path(mmdit_dir / "config.json")
            and _component_has_weights(mmdit_dir)
        )

    comp = root / _COMPONENT_KEYS[component]
    if not _component_has_weights(comp) or not _valid_config_path(comp / "config.json"):
        return False
    if component in _PROJECTION_STATE_KEYS:
        return (root / "projection_heads.pt").is_file()
    return True


def resolve_model_dir(phase, component, *, require_trained=False):
    """Return the on-disk model directory for ``component`` at ``phase``.

    phase 0 always uses the resized seed weights. Higher phases load from
    ``models/trained/stage{N}/`` when that checkpoint exists. Components that
    were not trained yet (e.g. MMDiT before stage 2) fall back to the seed dir
    unless ``require_trained=True``.
    """
    from pathlib import Path

    if not MIN_TRAINING_PHASE <= phase <= MAX_TRAINING_PHASE:
        raise ValueError(
            f"phase must be between {MIN_TRAINING_PHASE} and "
            f"{MAX_TRAINING_PHASE}, got {phase}"
        )
    if phase == 0:
        return _SEED_DIRS[component]

    trained_component = trained_dir_for_phase(phase) / _COMPONENT_KEYS[component]
    if (
        trained_component.is_dir()
        and _valid_config_path(trained_component / "config.json")
        and _component_has_weights(trained_component)
    ):
        return str(trained_component)

    if require_trained or component in _PROJECTION_STATE_KEYS:
        raise FileNotFoundError(
            f"No trained {component} checkpoint for phase {phase} at "
            f"{trained_component}. Train stage {phase} first or use --phase 0."
        )

    seed_dir = _SEED_DIRS[component]
    return seed_dir


def resolve_projection_path(phase, component):
    """Return projection-head checkpoint path for embedders, or None at phase 0."""
    if phase == 0 or component not in _PROJECTION_STATE_KEYS:
        return None
    root = trained_dir_for_phase(phase)
    path = root / "projection_heads.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"No projection heads for phase {phase} at {path}. "
            f"Train stage {phase} first or use --phase 0."
        )
    return str(path)


def describe_phase(phase, component):
    """Short human-readable summary of what will be loaded."""
    if phase == 0:
        return f"phase 0 (untrained seed): {_SEED_DIRS[component]}"
    if component in _PROJECTION_STATE_KEYS:
        if phase_checkpoint_available(phase, component):
            root = trained_dir_for_phase(phase)
            comp = root / _COMPONENT_KEYS[component]
            return (
                f"phase {phase} (trained, 8-bit): {comp} "
                f"+ projection_heads from {root}"
            )
        return (
            f"phase {phase} (checkpoint missing at "
            f"{trained_dir_for_phase(phase) / _COMPONENT_KEYS[component]})"
        )
    if phase_checkpoint_available(phase, component):
        return (
            f"phase {phase} (trained): "
            f"{trained_dir_for_phase(phase) / _COMPONENT_KEYS[component]}"
        )
    return (
        f"phase {phase} (no trained {component} yet; using seed): "
        f"{_SEED_DIRS[component]}"
    )


def _filter_projection_state_dict(module, state):
    """Drop checkpoint keys the module does not own (e.g. legacy ``bias``)."""
    allowed = set(module.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in allowed}
    dropped = sorted(set(state.keys()) - allowed)
    if dropped:
        print(f"  projection load: ignored keys {dropped}")
    return filtered


def _load_projection_head(projection, path, state_key, device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if state_key not in state:
        raise KeyError(
            f"{path} is missing {state_key!r}; expected keys "
            f"{sorted(state)}"
        )
    filtered = _filter_projection_state_dict(projection, state[state_key])
    projection.load_state_dict(filtered, strict=True)


def _torch_device(device) -> torch.device:
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device} requested but CUDA is unavailable.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    return device


def _device_map_for(device: torch.device):
    device = _torch_device(device)
    if device.type == "cuda":
        return {"": device.index}
    return "cpu"


def _resolve_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def checkpoint_is_quantized(model_dir: str) -> bool:
    """True when ``model_dir/model.safetensors`` stores bitsandbytes 8-bit weights."""
    from pathlib import Path

    from safetensors import safe_open

    weights = Path(model_dir) / "model.safetensors"
    if not weights.is_file():
        return False
    with safe_open(str(weights), framework="pt") as handle:
        for key in handle.keys():
            if key.endswith(".SCB"):
                return True
    return False


def should_load_in_8bit(phase: int, model_dir: str) -> bool:
    """Trained checkpoints are saved as 8-bit; seeds stay full precision."""
    if phase >= 1:
        return True
    return checkpoint_is_quantized(model_dir)


def _load_trained_8bit_state(model: nn.Module, trained_dir: str, label: str) -> None:
    """Apply a trained 8-bit ``state_dict`` onto an already-quantized model shell."""
    from pathlib import Path

    from safetensors.torch import load_file

    weights_path = Path(trained_dir) / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing trained weights at {weights_path}")
    state = load_file(str(weights_path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load trained {label} weights from {weights_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )


def dequantize_bnb_int8_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert a bitsandbytes LLM.int8 safetensors dict to float weights.

    Pairs ``*.weight`` (int8) with sibling ``*.SCB`` (per-output absmax scales).
    Other keys (norms, embeds already float) pass through. Meta keys
    (``.SCB``, ``.weight_format``, ``.CB``) are dropped.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.endswith((".SCB", ".weight_format", ".CB")):
            continue
        if key.endswith(".weight"):
            scb_key = key[: -len(".weight")] + ".SCB"
            if scb_key in state and value is not None:
                w = value.detach().float()
                scb = state[scb_key].detach().float()
                # LLM.int8: SCB is (out_features,) scale for each row.
                if scb.numel() == w.shape[0]:
                    w = w * scb.reshape(w.shape[0], *([1] * (w.ndim - 1)))
                else:
                    # Fallback: broadcast if shapes align another way.
                    w = w * scb.view(-1, *([1] * (w.ndim - 1)))[: w.shape[0]]
                out[key] = w
                continue
        out[key] = value
    return out


def enable_full_parameter_training(model: nn.Module) -> int:
    """Set ``requires_grad=True`` on every float/complex parameter. Returns count."""
    n = 0
    for param in model.parameters():
        if param.is_floating_point() or param.is_complex():
            param.requires_grad = True
            n += param.numel()
    return n


def load_siglip_backbone(
    model_dir: str,
    device: torch.device,
    *,
    load_in_8bit: bool = False,
    seed_dir: str | None = None,
    for_training: bool = False,
    compute_dtype: torch.dtype | None = None,
):
    """Load SigLIP vision backbone.

    Training (``for_training=True``) always loads **full precision** weights
    (bf16/fp16) so every parameter can receive gradients. Inference may still
    use 8-bit for memory. Legacy 8-bit trained checkpoints are dequantized into
    the float shell when resuming for training.
    """
    from pathlib import Path

    from safetensors.torch import load_file
    from transformers import BitsAndBytesConfig, SiglipVisionModel

    device = _torch_device(device)
    dtype = compute_dtype or _resolve_compute_dtype()

    # ----- Full-precision training path (all params optimizable) -----
    if for_training:
        quantized = checkpoint_is_quantized(model_dir)
        load_dir = (seed_dir or SIGLIP_DIR) if quantized else model_dir
        model = SiglipVisionModel.from_pretrained(
            load_dir,
            torch_dtype=dtype,
            device_map=_device_map_for(device),
        )
        if quantized:
            weights_path = Path(model_dir) / "model.safetensors"
            if weights_path.is_file():
                raw = load_file(str(weights_path))
                dequant = dequantize_bnb_int8_state_dict(raw)
                missing, unexpected = model.load_state_dict(dequant, strict=False)
                # Ignore missing/unexpected scale keys; flag shape mismatches loudly.
                bad_m = [k for k in missing if not k.endswith((".SCB", ".weight_format"))]
                bad_u = [
                    k
                    for k in unexpected
                    if not k.endswith((".SCB", ".weight_format", ".CB"))
                ]
                if bad_m or bad_u:
                    raise RuntimeError(
                        f"Failed to dequant-load trained SigLIP from {weights_path}: "
                        f"missing={bad_m[:20]}, unexpected={bad_u[:20]}"
                    )
                print(
                    f"  dequantized 8-bit SigLIP weights from {model_dir} "
                    f"into float shell ({load_dir})"
                )
        # Ensure every floating weight shares compute_dtype (no float32 embed vs bf16 linear).
        model = model.to(device=device, dtype=dtype)
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        model.train()
        n = enable_full_parameter_training(model)
        print(f"  SigLIP full-precision training: {n:,} trainable float params")
        return model

    # ----- Inference / 8-bit paths (unchanged for demos) -----
    if load_in_8bit and checkpoint_is_quantized(model_dir):
        init_dir = seed_dir or SIGLIP_DIR
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = SiglipVisionModel.from_pretrained(
            init_dir,
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map=_device_map_for(device),
        )
        _load_trained_8bit_state(model, model_dir, "SigLIP")
        model.eval()
        return model

    if load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = SiglipVisionModel.from_pretrained(
            model_dir,
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map=_device_map_for(device),
        )
        model.eval()
        return model

    model = SiglipVisionModel.from_pretrained(model_dir, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model


def load_qwen_backbone(
    model_dir: str,
    device: torch.device,
    *,
    load_in_8bit: bool = False,
    seed_dir: str | None = None,
    tokenizer_id: str = QWEN_TOKENIZER_ID,
    max_seq_length: int = 512,
    for_training: bool = False,
    compute_dtype: torch.dtype | None = None,
):
    """Load Qwen3-MoE text backbone.

    Training uses Unsloth ``full_finetuning=True`` (fp16/bf16 weights, no weight
    quant) so AdamW8bit can update every parameter. Inference may still load
    8-bit. Legacy 8-bit trained checkpoints are dequantized when resuming train.
    """
    from pathlib import Path

    from safetensors.torch import load_file

    device = _torch_device(device)
    dtype = compute_dtype or _resolve_compute_dtype()

    # ----- Full-precision training (all params optimizable) -----
    if for_training:
        import os

        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        from unsloth import FastLanguageModel

        quantized = checkpoint_is_quantized(model_dir)
        load_name = (seed_dir or QWEN_DIR) if quantized else model_dir
        # full_finetuning=True forces Unsloth to disable 4/8-bit weight quant.
        from_pretrained_kwargs = {
            "model_name": load_name,
            "max_seq_length": max_seq_length,
            "dtype": dtype,
            "full_finetuning": True,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "load_in_16bit": False,
            "device_map": _device_map_for(device),
            "tokenizer_name": tokenizer_id,
            "fix_tokenizer": False,
            "use_gradient_checkpointing": "unsloth",
        }
        model, _ = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)
        if quantized:
            weights_path = Path(model_dir) / "model.safetensors"
            if weights_path.is_file():
                raw = load_file(str(weights_path))
                dequant = dequantize_bnb_int8_state_dict(raw)
                missing, unexpected = model.load_state_dict(dequant, strict=False)
                bad_m = [
                    k
                    for k in missing
                    if not k.endswith((".SCB", ".weight_format", ".CB"))
                ]
                bad_u = [
                    k
                    for k in unexpected
                    if not k.endswith((".SCB", ".weight_format", ".CB"))
                ]
                if bad_m or bad_u:
                    raise RuntimeError(
                        f"Failed to dequant-load trained Qwen from {weights_path}: "
                        f"missing={bad_m[:20]}, unexpected={bad_u[:20]}"
                    )
                print(
                    f"  dequantized 8-bit Qwen weights from {model_dir} "
                    f"into float shell ({load_name})"
                )
        model = FastLanguageModel.for_training(model)
        # Unsloth may leave embeddings in float32 while linears are bf16; unify.
        model = model.to(dtype=dtype)
        model.train()
        n = enable_full_parameter_training(model)
        print(f"  Qwen full-precision training: {n:,} trainable float params")
        return model

    # ----- Inference / 8-bit paths -----
    if load_in_8bit and checkpoint_is_quantized(model_dir):
        import os

        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        from unsloth import FastLanguageModel

        init_dir = seed_dir or QWEN_DIR
        from_pretrained_kwargs = {
            "model_name": init_dir,
            "max_seq_length": max_seq_length,
            "dtype": dtype,
            "full_finetuning": False,
            "load_in_4bit": False,
            "load_in_8bit": True,
            "load_in_16bit": False,
            "device_map": _device_map_for(device),
            "tokenizer_name": tokenizer_id,
            "fix_tokenizer": False,
        }
        model, _ = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)
        _load_trained_8bit_state(model, model_dir, "Qwen3-MoE")
        model = FastLanguageModel.for_inference(model)
        model.eval()
        return model

    if load_in_8bit:
        import os

        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        from unsloth import FastLanguageModel

        from_pretrained_kwargs = {
            "model_name": model_dir,
            "max_seq_length": max_seq_length,
            "dtype": dtype,
            "full_finetuning": False,
            "load_in_4bit": False,
            "load_in_8bit": True,
            "load_in_16bit": False,
            "device_map": _device_map_for(device),
            "tokenizer_name": tokenizer_id,
            "fix_tokenizer": False,
        }
        model, _ = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)
        model = FastLanguageModel.for_inference(model)
        model.eval()
        return model

    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_dir, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model


def default_inference_device(prefer_index: int = 0) -> str:
    """Pick a CUDA device when available (required for bitsandbytes 8-bit)."""
    if torch.cuda.is_available():
        index = prefer_index if prefer_index < torch.cuda.device_count() else 0
        return f"cuda:{index}"
    return "cpu"


def _text_backbone(model: nn.Module) -> nn.Module:
    return getattr(model, "model", model)


# ============================ SIGLIP (IMAGE) ============================

class SiglipEmbedder:
    """Vision embedder: takes a PIL ``Image`` and returns patch embeddings.

    The SigLIP vision tower produces one hidden vector per image patch; each is
    projected to the shared 1024-dim Matryoshka space and normalized. The list
    of per-patch embeddings is what a late-interaction retriever consumes.
    """

    def __init__(self, model_dir=None, phase=0, projection_path=None,
                 processor_id=SIGLIP_PROCESSOR_ID, embed_dim=EMBED_DIM,
                 device="cpu"):
        from transformers import AutoImageProcessor

        if model_dir is None:
            model_dir = resolve_model_dir(phase, "siglip")
        if projection_path is None:
            projection_path = resolve_projection_path(phase, "siglip")

        device = _torch_device(device)
        load_in_8bit = should_load_in_8bit(phase, model_dir)
        self.load_in_8bit = load_in_8bit
        self.embed_dim = embed_dim
        self.phase = phase
        self.model_dir = model_dir
        self.model = load_siglip_backbone(
            model_dir,
            device,
            load_in_8bit=load_in_8bit,
            seed_dir=SIGLIP_DIR if load_in_8bit else None,
        )
        self.device = _model_device(self.model)
        # Prefer a colocated preprocessor (HF layout); fall back to baseline id.
        from pathlib import Path

        local_proc = Path(model_dir) / "preprocessor_config.json"
        proc_source = str(model_dir) if local_proc.is_file() else processor_id
        self.processor = AutoImageProcessor.from_pretrained(proc_source)
        # The processor may come from the 384px baseline; our resized tower
        # expects `image_size` px (e.g. 540) -> keep the patch grid consistent
        # with the model's learned position embeddings.
        target = self.model.config.image_size
        self.processor.size = {"height": target, "width": target}
        hidden = self.model.config.hidden_size
        proj_dtype = _resolve_compute_dtype() if load_in_8bit else torch.float32
        # Projection head to the shared Matryoshka embedding space.
        self.projection = nn.Linear(
            hidden, embed_dim, bias=False, device=self.device, dtype=proj_dtype
        )
        self.projection.eval()
        if projection_path:
            _load_projection_head(
                self.projection, projection_path, "vision_projection", self.device
            )

    @torch.no_grad()
    def embed_image(self, image, matryoshka_dim=None):
        """Return a list of 1024-dim patch embeddings for ``image`` (PIL.Image)."""
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        out = self.model(pixel_values=pixel_values)
        # (1, num_patches, hidden) -> project every patch token.
        tokens = out.last_hidden_state[0].to(dtype=self.projection.weight.dtype)
        projected = self.projection(tokens)
        normed = matryoshka_normalize(projected, dim=matryoshka_dim)
        return [row for row in normed]


# ============================ QWEN3-MOE (TEXT) ============================

class Qwen3MoeEmbedder:
    """Text embedder: takes a string and returns token embeddings.

    Every token's final hidden state is projected to the 1024-dim Matryoshka
    space and normalized. Consecutive embeddings whose cosine similarity is at
    least ``merge_threshold`` are merged (averaged then re-normalized), which
    collapses redundant tokens for cheaper late-interaction search.
    """

    def __init__(self, model_dir=None, phase=0, projection_path=None,
                 tokenizer_id=QWEN_TOKENIZER_ID, embed_dim=EMBED_DIM,
                 max_seq_length=512, device="cpu"):
        from transformers import AutoTokenizer

        if model_dir is None:
            model_dir = resolve_model_dir(phase, "qwen")
        if projection_path is None:
            projection_path = resolve_projection_path(phase, "qwen")

        device = _torch_device(device)
        load_in_8bit = should_load_in_8bit(phase, model_dir)
        self.load_in_8bit = load_in_8bit
        self.embed_dim = embed_dim
        self.phase = phase
        self.model_dir = model_dir
        # Prefer a colocated tokenizer (HF layout) when the checkpoint ships one.
        from pathlib import Path

        local_tok = Path(model_dir) / "tokenizer_config.json"
        effective_tokenizer_id = str(model_dir) if local_tok.is_file() else tokenizer_id
        self.model = load_qwen_backbone(
            model_dir,
            device,
            load_in_8bit=load_in_8bit,
            seed_dir=QWEN_DIR if load_in_8bit else None,
            tokenizer_id=effective_tokenizer_id,
            max_seq_length=max_seq_length,
        )
        self.device = _model_device(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(effective_tokenizer_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        hidden = self.model.config.hidden_size
        proj_dtype = _resolve_compute_dtype() if load_in_8bit else torch.float32
        self.projection = nn.Linear(
            hidden, embed_dim, bias=False, device=self.device, dtype=proj_dtype
        )
        self.projection.eval()
        if projection_path:
            _load_projection_head(
                self.projection, projection_path, "text_projection", self.device
            )

    @torch.no_grad()
    def embed_text(self, text, merge_threshold=1.0, matryoshka_dim=None):
        """Return a list of 1024-dim token embeddings for ``text``.

        merge_threshold : minimum cosine similarity for two *consecutive*
                          embeddings to be merged into one. ``1.0`` merges only
                          (near-)identical neighbours; lower values merge more.
        """
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = _text_backbone(self.model)(**enc)
        tokens = out.last_hidden_state[0].to(dtype=self.projection.weight.dtype)
        projected = self.projection(tokens)
        normed = matryoshka_normalize(projected, dim=matryoshka_dim)
        merged = self._merge_similar(normed, merge_threshold)
        return [row for row in merged]

    @staticmethod
    def _merge_similar(embeddings, merge_threshold):
        """Greedily merge consecutive embeddings with cosine sim >= threshold."""
        if merge_threshold is None or embeddings.shape[0] <= 1:
            return embeddings
        groups = [embeddings[0:1]]
        for i in range(1, embeddings.shape[0]):
            current = embeddings[i]
            rep = matryoshka_normalize(groups[-1].mean(dim=0))
            if torch.dot(rep, current).item() >= merge_threshold:
                groups[-1] = torch.cat([groups[-1], current.unsqueeze(0)], dim=0)
            else:
                groups.append(current.unsqueeze(0))
        merged = torch.stack([matryoshka_normalize(g.mean(dim=0)) for g in groups])
        return merged


# ============================ CONDITIONING TRANSFORMS ============================

def shuffle_token_embeddings(tokens: torch.Tensor):
    """Randomly permute tokens along the sequence axis.

    Accepts ``(N, D)`` or ``(B, N, D)``. Returns ``(shuffled, perm)`` where
    ``perm`` indexes the original order (``shuffled = tokens[..., perm, :]`` for
    2-D, or per-batch perms stacked for 3-D).
    """
    if tokens.ndim == 2:
        n = tokens.shape[0]
        perm = torch.randperm(n, device=tokens.device)
        return tokens[perm], perm
    if tokens.ndim == 3:
        b, n, _ = tokens.shape
        perms = []
        outs = []
        for i in range(b):
            perm = torch.randperm(n, device=tokens.device)
            perms.append(perm)
            outs.append(tokens[i, perm])
        return torch.stack(outs, dim=0), torch.stack(perms, dim=0)
    raise ValueError(f"tokens must be 2-D or 3-D, got shape {tuple(tokens.shape)}")


def embedding_token_dropout(
    tokens: torch.Tensor,
    drop_prob: float = 0.2,
    *,
    training: bool = True,
) -> torch.Tensor:
    """Drop a random fraction of tokens (keep ≥1). No-op if not training or p≤0.

    ``tokens`` is ``(N, D)`` or ``(B, N, D)``. For batched input, each sample
    is dropped independently and results are left-padded by repeating the first
    kept token so the batch stays rectangular (generator needs fixed seq len).
    Prefer ``(N, D)`` per-sample application when seq lengths may diverge.
    """
    if not training or drop_prob is None or float(drop_prob) <= 0.0:
        return tokens
    prob = min(float(drop_prob), 1.0 - 1e-6)
    if tokens.ndim == 2:
        n = tokens.shape[0]
        k = max(1, int(round(n * (1.0 - prob))))
        k = min(k, n)
        if k >= n:
            return tokens
        idx = torch.randperm(n, device=tokens.device)[:k]
        idx, _ = idx.sort()
        return tokens[idx]
    if tokens.ndim == 3:
        b, n, d = tokens.shape
        k = max(1, int(round(n * (1.0 - prob))))
        k = min(k, n)
        if k >= n:
            return tokens
        out = tokens.new_empty(b, k, d)
        for i in range(b):
            idx = torch.randperm(n, device=tokens.device)[:k]
            idx, _ = idx.sort()
            out[i] = tokens[i, idx]
        return out
    raise ValueError(f"tokens must be 2-D or 3-D, got shape {tuple(tokens.shape)}")


def maybe_merge_embeddings_to_one(
    tokens: torch.Tensor,
    merge_prob: float = 0.05,
    *,
    training: bool = True,
) -> torch.Tensor:
    """With probability ``merge_prob``, collapse all tokens to one mean vector.

    Mean is L2-renormalized (Matryoshka-safe). One coin flip for the whole
    tensor (batch stays rectangular). Stage 2 uses this so the generator
    sometimes sees a heavily compressed condition (single token).
    """
    if not training or merge_prob is None or float(merge_prob) <= 0.0:
        return tokens
    if torch.rand(1).item() >= float(merge_prob):
        return tokens
    if tokens.ndim == 2:
        mean = tokens.mean(dim=0, keepdim=True)
        return F.normalize(mean.float(), dim=-1).to(dtype=tokens.dtype)
    if tokens.ndim == 3:
        mean = tokens.mean(dim=1, keepdim=True)  # (B, 1, D)
        return F.normalize(mean.float(), dim=-1).to(dtype=tokens.dtype)
    raise ValueError(f"tokens must be 2-D or 3-D, got shape {tuple(tokens.shape)}")


def prepare_stage2_condition_tokens(
    tokens: torch.Tensor,
    *,
    shuffle: bool = True,
    drop_prob: float = 0.2,
    merge_prob: float = 0.05,
    max_tokens: int = 64,
    training: bool = True,
) -> torch.Tensor:
    """Shuffle → cap length → dropout → occasional full merge (Stage 2 recipe).

    ``max_tokens`` keeps joint-attention VRAM bounded (full SigLIP grids are
    ~900 patches; SD3 MMDiT cannot hold that on 12GB with Adam states).
    After shuffle, taking the first ``max_tokens`` is a uniform random subset.
    """
    x = tokens
    if shuffle:
        x, _ = shuffle_token_embeddings(x)
    # Cap sequence length (post-shuffle subset).
    cap = int(max_tokens) if max_tokens is not None else 0
    if cap > 0:
        if x.ndim == 2 and x.shape[0] > cap:
            x = x[:cap]
        elif x.ndim == 3 and x.shape[1] > cap:
            x = x[:, :cap]
    x = embedding_token_dropout(x, drop_prob=drop_prob, training=training)
    x = maybe_merge_embeddings_to_one(x, merge_prob=merge_prob, training=training)
    return x


# ============================ MMDiT (IMAGE GENERATION) ============================

CONDITIONING_HEADS_FILE = "conditioning_heads.pt"


class MMDiTGenerator:
    """SD3-style MMDiT image generator wrapper.

    Two ways to condition generation:
      * ``generate(text=...)``       -- the text is tokenized like a normal
        text-to-image pipeline and turned into conditioning embeddings.
      * ``generate(embeddings=...)`` -- externally supplied 1024-dim Matryoshka
        embeddings are pushed through a small transform stage that maps them to
        the transformer's conditioning shapes.

    Stage 2 trains the transformer + embed adapters to reconstruct images from
    vision patch embeddings (see ``forward_train`` / ``train_stage2.py``).

    The transformer works in the VAE *latent* space (16 channels, 8x smaller
    than the image). The latents it produces are decoded back to full-resolution
    pixels with the SD3 VAE.
    """

    def __init__(
        self,
        model_dir=None,
        phase=0,
        tokenizer_id=QWEN_TOKENIZER_ID,
        vae_id=MMDIT_VAE_ID,
        embed_dim=EMBED_DIM,
        device="cpu",
        conditioning_path=None,
    ):
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from diffusers.models.transformers import SD3Transformer2DModel
        from transformers import AutoTokenizer
        from pathlib import Path

        if model_dir is None:
            model_dir = resolve_model_dir(phase, "mmdit")

        device = _torch_device(device)
        self.device = device
        self.embed_dim = embed_dim
        self.phase = phase
        self.model_dir = model_dir
        # bf16 saves VRAM on 12GB cards; fall back to fp32 on CPU.
        self.compute_dtype = (
            torch.bfloat16
            if str(device).startswith("cuda") and torch.cuda.is_bf16_supported()
            else (torch.float16 if str(device).startswith("cuda") else torch.float32)
        )
        self.transformer = SD3Transformer2DModel.from_pretrained(
            model_dir,
            torch_dtype=self.compute_dtype,
            low_cpu_mem_usage=True,
        )
        self.transformer = self.transformer.to(device)
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        # Full noise schedule for training (1000 steps).
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

        self.vae = AutoencoderKL.from_pretrained(
            vae_id, subfolder="vae", torch_dtype=self.compute_dtype
        )
        self.vae = self.vae.to(device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        cfg = self.transformer.config
        self.in_channels = cfg.in_channels
        self.joint_dim = cfg.joint_attention_dim
        self.pooled_dim = cfg.pooled_projection_dim
        self.sample_size = int(getattr(cfg, "sample_size", 64) or 64)

        vocab = len(self.tokenizer)
        self.token_embedding = nn.Embedding(vocab, self.joint_dim).to(device)
        self.text_pool = nn.Linear(self.joint_dim, self.pooled_dim).to(device)
        self.embed_to_seq = nn.Linear(embed_dim, self.joint_dim).to(
            device=device, dtype=self.compute_dtype
        )
        self.embed_to_pool = nn.Linear(embed_dim, self.pooled_dim).to(
            device=device, dtype=self.compute_dtype
        )

        # Optional Stage-2 conditioning checkpoint next to model_dir or explicit.
        if conditioning_path is None:
            parent = Path(model_dir).parent
            cand = parent / CONDITIONING_HEADS_FILE
            if cand.is_file():
                conditioning_path = str(cand)
            elif (Path(model_dir) / CONDITIONING_HEADS_FILE).is_file():
                conditioning_path = str(Path(model_dir) / CONDITIONING_HEADS_FILE)
        if conditioning_path:
            self.load_conditioning_heads(conditioning_path)

        self.eval()  # default inference mode; train_stage2 calls train()

    def train(self, mode: bool = True):
        """Train transformer + embed adapters; VAE stays frozen eval."""
        self.transformer.train(mode)
        self.embed_to_seq.train(mode)
        self.embed_to_pool.train(mode)
        # Text path unused in stage2 recon; keep eval.
        self.token_embedding.eval()
        self.text_pool.eval()
        self.vae.eval()
        return self

    def eval(self):
        self.transformer.eval()
        self.embed_to_seq.eval()
        self.embed_to_pool.eval()
        self.token_embedding.eval()
        self.text_pool.eval()
        self.vae.eval()
        return self

    def trainable_parameters(self):
        for p in self.transformer.parameters():
            if p.requires_grad:
                yield p
        yield from self.embed_to_seq.parameters()
        yield from self.embed_to_pool.parameters()

    def freeze_non_stage2(self):
        """Freeze VAE + unused text path; **full** train transformer + embed adapters.

        Stage 2 is full pretraining of the generator (not LoRA). Optimizer state
        is kept off-GPU in ``build_stage2_optimizer`` so 12GB dual-GPU setups fit.
        """
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.token_embedding.parameters():
            p.requires_grad_(False)
        for p in self.text_pool.parameters():
            p.requires_grad_(False)
        for p in self.transformer.parameters():
            p.requires_grad_(True)
        for p in self.embed_to_seq.parameters():
            p.requires_grad_(True)
        for p in self.embed_to_pool.parameters():
            p.requires_grad_(True)

    def save_conditioning_heads(self, path: str):
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embed_to_seq": self.embed_to_seq.state_dict(),
                "embed_to_pool": self.embed_to_pool.state_dict(),
            },
            path,
        )

    def load_conditioning_heads(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.embed_to_seq.load_state_dict(state["embed_to_seq"])
        self.embed_to_pool.load_state_dict(state["embed_to_pool"])
        self.embed_to_seq.to(self.device)
        self.embed_to_pool.to(self.device)

    def _encode_text(self, text):
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=77
        )
        ids = enc["input_ids"].to(self.device)
        seq = self.token_embedding(ids)
        pooled = self.text_pool(seq.mean(dim=1))
        return seq, pooled

    def _encode_embeddings(self, embeddings):
        """Map Matryoshka tokens → (encoder_hidden_states, pooled). Grad-enabled."""
        embs = _stack(embeddings).to(device=self.device, dtype=self.embed_to_seq.weight.dtype)
        if embs.ndim == 2:
            embs = embs.unsqueeze(0)
        seq = self.embed_to_seq(embs)
        pooled = self.embed_to_pool(embs.mean(dim=1))
        return seq, pooled

    @torch.no_grad()
    def encode_images_to_latents(self, images_n11: torch.Tensor) -> torch.Tensor:
        """VAE-encode images in ``[-1, 1]`` to scaled latents (training targets).

        Runs on whatever device the VAE currently lives on (CPU offload supported).
        """
        vae_param = next(self.vae.parameters())
        images_n11 = images_n11.to(device=vae_param.device, dtype=vae_param.dtype)
        posterior = self.vae.encode(images_n11).latent_dist
        latents = posterior.sample()
        shift = getattr(self.vae.config, "shift_factor", 0.0) or 0.0
        latents = (latents - shift) * self.vae.config.scaling_factor
        return latents

    def _ensure_train_schedule(self, device) -> int:
        """Cache the full FM discrete schedule on ``device`` (once per device)."""
        num = int(self.scheduler.config.num_train_timesteps)
        key = (num, str(device))
        cached = getattr(self, "_train_schedule_key", None)
        if cached != key or not hasattr(self, "_train_timesteps"):
            self.scheduler.set_timesteps(num, device=device)
            self._train_timesteps = self.scheduler.timesteps
            self._train_sigmas = self.scheduler.sigmas
            self._train_schedule_key = key
        return num

    def forward_train(
        self,
        clean_latents: torch.Tensor,
        embeddings: torch.Tensor,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        *,
        return_metrics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Flow-matching training step toward ``clean_latents`` (image latents).

        Returns ``(loss, model_pred, metrics)``.
        Target is rectified-flow velocity ``noise - clean_latents`` with
        ``z_t = (1-σ) * clean + σ * noise`` (diffusers FlowMatch convention).

        Metrics stay on-device (no host sync). Callers should ``.item()`` only
        when logging — every-step ``.cpu()`` serializes the GPU.
        """
        b = clean_latents.shape[0]
        device = clean_latents.device
        if noise is None:
            noise = torch.randn_like(clean_latents)
        num = self._ensure_train_schedule(device)
        if timesteps is None:
            indices = torch.randint(0, num, (b,), device=device)
            timesteps = self._train_timesteps[indices]
            sigmas = self._train_sigmas[indices].to(dtype=clean_latents.dtype)
        else:
            # Map provided timesteps to nearest schedule entry (rare path).
            diffs = (
                self._train_timesteps.unsqueeze(0) - timesteps.unsqueeze(1)
            ).abs()
            j = diffs.argmin(dim=1)
            sigmas = self._train_sigmas[j].to(dtype=clean_latents.dtype)
            timesteps = self._train_timesteps[j]

        while sigmas.ndim < clean_latents.ndim:
            sigmas = sigmas.unsqueeze(-1)

        noisy = (1.0 - sigmas) * clean_latents + sigmas * noise
        target = noise - clean_latents

        encoder_hidden_states, pooled = self._encode_embeddings(embeddings)
        dtype = getattr(self, "compute_dtype", None) or next(
            self.transformer.parameters()
        ).dtype
        noisy = noisy.to(dtype=dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=dtype)
        pooled = pooled.to(dtype=dtype)

        model_pred = self.transformer(
            hidden_states=noisy,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            timestep=timesteps,
            return_dict=False,
        )[0]
        # Pipeline-parallel: output may live on a different GPU than inputs.
        target = target.to(device=model_pred.device, dtype=model_pred.dtype)
        loss = F.mse_loss(model_pred.float(), target.float())
        if return_metrics:
            metrics = {
                "flow_mse": loss.detach(),
                "sigma_mean": sigmas.detach().float().mean(),
            }
        else:
            metrics = {}
        return loss, model_pred, metrics

    @torch.no_grad()
    def generate(
        self,
        text=None,
        embeddings=None,
        height=None,
        width=None,
        num_inference_steps=4,
        seed=0,
        shuffle_embeddings: bool = False,
    ):
        """Run the denoising loop and return a PIL image.

        Provide exactly one of ``text`` or ``embeddings``.
        ``height`` / ``width`` default to ``sample_size * vae_scale_factor``.
        """
        if (text is None) == (embeddings is None):
            raise ValueError("Provide exactly one of `text` or `embeddings`.")

        if height is None:
            height = self.sample_size * self.vae_scale_factor
        if width is None:
            width = self.sample_size * self.vae_scale_factor

        patch = self.transformer.config.patch_size
        divisor = self.vae_scale_factor * patch
        if height % divisor or width % divisor:
            raise ValueError(
                f"height/width must be multiples of {divisor} "
                f"(vae_scale_factor {self.vae_scale_factor} x patch_size {patch}); "
                f"got {height}x{width}."
            )

        if text is not None:
            encoder_hidden_states, pooled = self._encode_text(text)
        else:
            embs = embeddings
            if shuffle_embeddings:
                stacked = _stack(embs)
                if stacked.ndim == 1:
                    stacked = stacked.unsqueeze(0)
                stacked, _ = shuffle_token_embeddings(stacked)
                embs = stacked
            encoder_hidden_states, pooled = self._encode_embeddings(embs)

        dtype = next(self.transformer.parameters()).dtype
        encoder_hidden_states = encoder_hidden_states.to(dtype=dtype)
        pooled = pooled.to(dtype=dtype)

        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(
            1,
            self.in_channels,
            latent_h,
            latent_w,
            generator=generator,
        ).to(device=self.device, dtype=dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        for t in self.scheduler.timesteps:
            timestep = t.expand(latents.shape[0]).to(self.device)
            noise_pred = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled,
                timestep=timestep,
                return_dict=False,
            )[0]
            latents = self.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]

        return self._latents_to_image(latents)

    @torch.no_grad()
    def _latents_to_image(self, latents):
        """Decode latents to a full-resolution RGB image with the SD3 VAE."""
        import numpy as np
        from PIL import Image

        cfg = self.vae.config
        latents = latents / cfg.scaling_factor + getattr(cfg, "shift_factor", 0.0)
        latents = latents.to(dtype=next(self.vae.parameters()).dtype)
        decoded = self.vae.decode(latents, return_dict=False)[0]
        x = (decoded[0] / 2 + 0.5).clamp(0, 1)
        array = (x * 255.0).round().to(torch.uint8)
        array = array.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return Image.fromarray(array, mode="RGB")


# ============================ QUERY STORE ============================

class LateInteractionStore:
    """Tiny in-memory store mapping a query label to its token embeddings.

    Used by the runner scripts to keep a history of previous queries and, for
    each new query, report the most similar prior queries by MaxSim score.
    """

    def __init__(self):
        self.entries = []  # list of (label, embeddings tensor)

    def most_similar(self, embeddings, top_k=2):
        """Return the ``top_k`` prior (label, score) pairs, highest score first."""
        scored = [(label, late_interaction_score(embeddings, stored))
                  for label, stored in self.entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def add(self, label, embeddings):
        self.entries.append((label, _stack(embeddings)))
