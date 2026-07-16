#!/usr/bin/env python3
"""Low-level tensor ops for architecture-aware weight seeding."""

from __future__ import annotations

import math
import re
from typing import Any

import torch
import torch.nn.functional as F


def numel_of(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def init_range_for(tensor: torch.Tensor, default: float = 0.02) -> float:
    """Fan-in scaled noise for expanded weight regions."""
    if tensor.ndim >= 2:
        fan_in = int(tensor.shape[-1])
        return min(default, 1.0 / math.sqrt(max(fan_in, 1)))
    return default


@torch.no_grad()
def fill_new_region(
    dst: torch.Tensor,
    src_shape: tuple[int, ...],
    *,
    init_range: float | None = None,
    generator: torch.Generator | None = None,
) -> None:
    """Fill cells of ``dst`` outside the leading ``src_shape`` overlap with noise."""
    if dst.shape == src_shape:
        return
    rng = init_range if init_range is not None else init_range_for(dst)
    # Build a mask of the overlap, then noise the rest.
    slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src_shape))
    noise = torch.empty_like(dst)
    if generator is not None and dst.device.type == "cpu":
        noise.normal_(0.0, rng, generator=generator)
    else:
        noise.normal_(0.0, rng)
    # Preserve overlap; only write outside.
    full = noise
    full[slices] = dst[slices]
    dst.copy_(full)


@torch.no_grad()
def copy_exact(dst: torch.Tensor, src: torch.Tensor) -> str:
    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
    return "exact"


@torch.no_grad()
def copy_leading_overlap(
    dst: torch.Tensor,
    src: torch.Tensor,
    *,
    init_new: bool = True,
    init_range: float | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """Copy leading min-shape region; optionally re-init the rest."""
    if tuple(dst.shape) == tuple(src.shape):
        return copy_exact(dst, src)
    if dst.ndim != src.ndim:
        return "shape_mismatch"
    slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
    if init_new:
        fill_new_region(
            dst, tuple(src.shape), init_range=init_range, generator=generator
        )
    dst[slices].copy_(src[slices].to(device=dst.device, dtype=dst.dtype))
    return "arch_aware" if any(a != b for a, b in zip(dst.shape, src.shape)) else "exact"


@torch.no_grad()
def copy_linear_width(
    dst: torch.Tensor,
    src: torch.Tensor,
    *,
    init_range: float | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """Width expand/shrink for Linear weights/biases (any rank via leading slice)."""
    return copy_leading_overlap(
        dst, src, init_new=True, init_range=init_range, generator=generator
    )


@torch.no_grad()
def copy_heads_out_in(
    dst: torch.Tensor,
    src: torch.Tensor,
    *,
    head_dim: int,
    init_range: float | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """Copy attention projections laid out as ``[n_heads * head_dim, hidden]``.

    Transfers whole head blocks along the out-feature axis, then width on in-dim.
    """
    if dst.ndim != 2 or src.ndim != 2:
        return copy_linear_width(dst, src, init_range=init_range, generator=generator)
    if head_dim <= 0:
        return copy_linear_width(dst, src, init_range=init_range, generator=generator)

    d_out, d_in = dst.shape
    s_out, s_in = src.shape
    if d_out % head_dim != 0 or s_out % head_dim != 0:
        return copy_linear_width(dst, src, init_range=init_range, generator=generator)

    n_dst = d_out // head_dim
    n_src = s_out // head_dim
    n_copy = min(n_dst, n_src)
    in_copy = min(d_in, s_in)

    rng = init_range if init_range is not None else init_range_for(dst)
    # Start from small noise so unused heads/in-dims are not zeros.
    if generator is not None and dst.device.type == "cpu":
        dst.normal_(0.0, rng, generator=generator)
    else:
        dst.normal_(0.0, rng)

    for h in range(n_copy):
        dr = slice(h * head_dim, (h + 1) * head_dim)
        sr = slice(h * head_dim, (h + 1) * head_dim)
        dst[dr, :in_copy].copy_(src[sr, :in_copy].to(dtype=dst.dtype, device=dst.device))

    # Extra destination heads: clone source head 0 + noise.
    for h in range(n_copy, n_dst):
        dr = slice(h * head_dim, (h + 1) * head_dim)
        base = src[:head_dim, :in_copy].to(dtype=dst.dtype, device=dst.device)
        noise = torch.empty_like(dst[dr, :in_copy])
        noise.normal_(0.0, rng * 0.5)
        dst[dr, :in_copy].copy_(base + noise)

    return "arch_aware" if (n_dst != n_src or d_in != s_in) else "exact"


@torch.no_grad()
def resize_conv2d_spatial(
    dst: torch.Tensor,
    src: torch.Tensor,
    *,
    mode: str = "bilinear",
    init_range: float | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """Resize Conv2d weight ``(out, in, kH, kW)`` spatially, then channel-expand."""
    if dst.ndim != 4 or src.ndim != 4:
        return copy_linear_width(dst, src, init_range=init_range, generator=generator)

    d_o, d_i, d_h, d_w = dst.shape
    s_o, s_i, s_h, s_w = src.shape
    # Resize spatial of source to target kernel size.
    w = src.float()
    if (s_h, s_w) != (d_h, d_w):
        w = F.interpolate(w, size=(d_h, d_w), mode=mode, align_corners=False)
    w = w.to(dtype=dst.dtype, device=dst.device)

    rng = init_range if init_range is not None else init_range_for(dst)
    if generator is not None and dst.device.type == "cpu":
        dst.normal_(0.0, rng, generator=generator)
    else:
        dst.normal_(0.0, rng)

    o = min(d_o, s_o)
    i = min(d_i, s_i)
    dst[:o, :i].copy_(w[:o, :i])
    if d_o == s_o and d_i == s_i and (s_h, s_w) == (d_h, d_w):
        return "exact"
    return "arch_aware"


@torch.no_grad()
def resize_pos_embed_2d(
    dst: torch.Tensor,
    src: torch.Tensor,
    *,
    mode: str = "bilinear",
    init_range: float | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """Interpolate 2D position embeddings ``(N, D)`` on a square (or near-square) grid."""
    if dst.ndim != 2 or src.ndim != 2:
        return copy_linear_width(dst, src, init_range=init_range, generator=generator)

    n_d, d_d = dst.shape
    n_s, d_s = src.shape

    def _grid_hw(n: int) -> tuple[int, int]:
        g = int(round(math.sqrt(n)))
        if g * g == n:
            return g, g
        # Near-square factors
        for h in range(g, 0, -1):
            if n % h == 0:
                return h, n // h
        return 1, n

    hs, ws = _grid_hw(n_s)
    hd, wd = _grid_hw(n_d)
    if hs * ws != n_s or hd * wd != n_d:
        return copy_linear_width(dst, src, init_range=init_range, generator=generator)

    # (1, D, H, W) for interpolate on spatial dims
    grid = src.float().T.reshape(1, d_s, hs, ws)
    if (hs, ws) != (hd, wd):
        # Interpolate each channel; if D changes we only use min channels first.
        grid = F.interpolate(grid, size=(hd, wd), mode=mode, align_corners=False)
    # (N, D_s)
    flat = grid.reshape(d_s, hd * wd).T.contiguous()

    rng = init_range if init_range is not None else init_range_for(dst)
    if generator is not None and dst.device.type == "cpu":
        dst.normal_(0.0, rng, generator=generator)
    else:
        dst.normal_(0.0, rng)

    d_copy = min(d_d, d_s)
    n_copy = min(n_d, flat.shape[0])
    dst[:n_copy, :d_copy].copy_(
        flat[:n_copy, :d_copy].to(dtype=dst.dtype, device=dst.device)
    )
    if n_d == n_s and d_d == d_s and (hs, ws) == (hd, wd):
        return "exact"
    return "arch_aware"


# ---------------------------------------------------------------------------
# MoE pack / unpack
# ---------------------------------------------------------------------------

_EXPERT_UNPACKED_RE = re.compile(
    r"^(?P<pre>.*\.mlp\.experts)\.(?P<idx>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_EXPERT_PACKED_GATE_UP = re.compile(
    r"^(?P<pre>.*\.mlp\.experts)\.gate_up_proj$"
)
_EXPERT_PACKED_DOWN = re.compile(
    r"^(?P<pre>.*\.mlp\.experts)\.down_proj$"
)


@torch.no_grad()
def unpack_moe_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Expand packed MoE expert tensors into per-expert gate/up/down weights.

    Packed layouts (transformers 4.x/5.x Qwen MoE):
      - ``experts.gate_up_proj``: ``(E, 2*I, H)`` or ``(E, H, 2*I)``
      - ``experts.down_proj``:   ``(E, H, I)`` or ``(E, I, H)``
    """
    out: dict[str, torch.Tensor] = {}
    skip: set[str] = set()

    # Detect packed keys
    packed_gate_up: dict[str, torch.Tensor] = {}
    packed_down: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        m_gu = _EXPERT_PACKED_GATE_UP.match(k)
        m_d = _EXPERT_PACKED_DOWN.match(k)
        if m_gu and v.ndim == 3:
            packed_gate_up[m_gu.group("pre")] = v
            skip.add(k)
        elif m_d and v.ndim == 3:
            packed_down[m_d.group("pre")] = v
            skip.add(k)

    for k, v in state.items():
        if k not in skip:
            out[k] = v

    for pre, gu in packed_gate_up.items():
        # Prefer (E, 2I, H): out features in dim 1
        if gu.shape[1] >= gu.shape[2] or gu.shape[1] % 2 == 0:
            # (E, 2I, H)
            e, two_i, h = gu.shape
            if two_i % 2 != 0:
                # try (E, H, 2I)
                e, h, two_i = gu.shape
                if two_i % 2 != 0:
                    continue
                inter = two_i // 2
                for i in range(e):
                    out[f"{pre}.{i}.gate_proj.weight"] = gu[i, :, :inter].T.contiguous()
                    out[f"{pre}.{i}.up_proj.weight"] = gu[i, :, inter:].T.contiguous()
            else:
                inter = two_i // 2
                for i in range(e):
                    out[f"{pre}.{i}.gate_proj.weight"] = gu[i, :inter, :].contiguous()
                    out[f"{pre}.{i}.up_proj.weight"] = gu[i, inter:, :].contiguous()
        else:
            e, h, two_i = gu.shape
            if two_i % 2 != 0:
                continue
            inter = two_i // 2
            for i in range(e):
                out[f"{pre}.{i}.gate_proj.weight"] = gu[i, :, :inter].T.contiguous()
                out[f"{pre}.{i}.up_proj.weight"] = gu[i, :, inter:].T.contiguous()

    for pre, down in packed_down.items():
        # (E, H, I) → down_proj weight (H, I) each? Linear down is (H, I) if y = x @ W.T
        # HF down_proj.weight is (hidden, inter) = (H, I)
        if down.shape[1] >= down.shape[2]:
            # (E, H, I)
            e, h, inter = down.shape
            for i in range(e):
                out[f"{pre}.{i}.down_proj.weight"] = down[i].contiguous()
        else:
            # (E, I, H) → transpose to (H, I)
            e, inter, h = down.shape
            for i in range(e):
                out[f"{pre}.{i}.down_proj.weight"] = down[i].T.contiguous()

    return out


def list_expert_indices(state: dict[str, torch.Tensor], layer_prefix: str) -> list[int]:
    """Return sorted expert indices for ``...layers.L.mlp.experts``."""
    idxs: set[int] = set()
    for k in state:
        if not k.startswith(layer_prefix):
            continue
        m = _EXPERT_UNPACKED_RE.match(k)
        if m and m.group("pre") == layer_prefix.rstrip("."):
            idxs.add(int(m.group("idx")))
        # also handle prefix without trailing issues
        m2 = re.match(
            re.escape(layer_prefix) + r"\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$",
            k,
        )
        if m2:
            idxs.add(int(m2.group(1)))
    return sorted(idxs)


def expert_prefixes(state: dict[str, torch.Tensor]) -> list[str]:
    """Unique ``...mlp.experts`` prefixes that have unpacked experts."""
    pres: set[str] = set()
    for k in state:
        m = _EXPERT_UNPACKED_RE.match(k)
        if m:
            pres.add(m.group("pre"))
    return sorted(pres)


@torch.no_grad()
def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.float().reshape(-1)
    y = b.float().reshape(-1)
    if x.numel() != y.numel() or x.numel() == 0:
        return 0.0
    return float(F.cosine_similarity(x, y, dim=0).item())


@torch.no_grad()
def ensure_expert_diversity(
    state: dict[str, torch.Tensor],
    *,
    max_cos: float = 0.3,
    init_range: float = 0.02,
    max_rounds: int = 5,
    generator: torch.Generator | None = None,
) -> list[str]:
    """Re-init near-duplicate experts until pairwise cos < max_cos.

    Returns list of warning strings.
    """
    warnings: list[str] = []
    for pre in expert_prefixes(state):
        idxs = []
        for k in state:
            m = _EXPERT_UNPACKED_RE.match(k)
            if m and m.group("pre") == pre and m.group("proj") == "gate_proj":
                idxs.append(int(m.group("idx")))
        idxs = sorted(set(idxs))
        if len(idxs) < 2:
            continue
        for _round in range(max_rounds):
            fixed = False
            for i_pos, i in enumerate(idxs):
                wi = state.get(f"{pre}.{i}.gate_proj.weight")
                if wi is None:
                    continue
                for j in idxs[i_pos + 1 :]:
                    wj = state.get(f"{pre}.{j}.gate_proj.weight")
                    if wj is None:
                        continue
                    c = cosine_flat(wi, wj)
                    if c > max_cos:
                        # Re-init expert j fully.
                        for proj in ("gate_proj", "up_proj", "down_proj"):
                            key = f"{pre}.{j}.{proj}.weight"
                            if key not in state:
                                continue
                            t = state[key]
                            rng = init_range_for(t, init_range)
                            if generator is not None and t.device.type == "cpu":
                                t.normal_(0.0, rng, generator=generator)
                            else:
                                t.normal_(0.0, rng)
                        warnings.append(
                            f"reinit {pre}.{j} (cos={c:.3f} vs expert {i})"
                        )
                        fixed = True
            if not fixed:
                break
    return warnings


def classify_param_name(name: str) -> str:
    n = name.lower()
    if "patch_embed" in n or "patch_embedding" in n:
        return "patch"
    if "position_embedding" in n or "pos_embed" in n:
        return "pos"
    if "expert" in n:
        return "experts"
    if "self_attn" in n or ".attn." in n or "attention" in n:
        return "attn"
    if "mlp" in n or "ff." in n or "feed_forward" in n:
        return "mlp"
    if "embed" in n or "lm_head" in n or "wte" in n:
        return "embed"
    if "norm" in n or "layernorm" in n:
        return "norm"
    if "time" in n or "timestep" in n:
        return "time"
    if "context" in n or "caption" in n:
        return "context"
    return "other"


def layer_index(name: str) -> int | None:
    for pat in (
        r"encoder\.layers\.(\d+)",
        r"layers\.(\d+)",
        r"transformer_blocks\.(\d+)",
    ):
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def is_attention_out_proj(name: str) -> bool:
    return any(
        s in name
        for s in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "out_proj",
            "to_q",
            "to_k",
            "to_v",
            "to_out",
        )
    )


def empty_report() -> dict[str, Any]:
    return {
        "exact": 0,
        "arch_aware": 0,
        "partial_fallback": 0,
        "fresh": 0,
        "shape_mismatch": 0,
        "total_target": 0,
        "param_mass_source": 0,
        "param_mass_fresh": 0,
        "by_component": {},
        "warnings": [],
        "copied_keys": [],
        "fresh_keys": [],
    }


def bump(
    report: dict[str, Any],
    kind: str,
    name: str,
    shape: tuple[int, ...],
    *,
    source_mass: int = 0,
    fresh_mass: int | None = None,
) -> None:
    report[kind] = report.get(kind, 0) + 1
    te = numel_of(shape)
    if fresh_mass is None:
        if kind in ("exact", "arch_aware", "partial_fallback"):
            # source_mass may be partial
            sm = source_mass if source_mass > 0 else te
            fm = max(te - sm, 0)
        else:
            sm, fm = 0, te
    else:
        sm, fm = source_mass, fresh_mass
    report["param_mass_source"] += sm
    report["param_mass_fresh"] += fm
    comp = classify_param_name(name)
    bucket = report["by_component"].setdefault(
        comp, {"exact": 0, "arch_aware": 0, "partial_fallback": 0, "fresh": 0}
    )
    bucket[kind] = bucket.get(kind, 0) + 1
    if kind == "fresh":
        report["fresh_keys"].append(name)
    else:
        report["copied_keys"].append(name)
