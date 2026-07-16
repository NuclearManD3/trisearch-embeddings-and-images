#!/usr/bin/env python3
"""
Shared helpers for the three model-creation scripts.

Architecture-aware seeding preserves pretrained weights when shapes change:
patch/pos resize (SigLIP), head-block attention copies, MoE pack/unpack,
depth clone-last, width expand with noise, and expert diversity checks.

Public API
----------
- ``select_candidate`` / ``prompt_seed_model`` / ``create_seeded_model``
- ``seed_parameters(target, source, family=...)`` → report dict
- ``report_seeding`` / ``write_seed_report``
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Literal

import torch

from model_seeding_ops import (
    bump,
    classify_param_name,
    copy_exact,
    copy_heads_out_in,
    copy_leading_overlap,
    copy_linear_width,
    empty_report,
    ensure_expert_diversity,
    expert_prefixes,
    fill_new_region,
    init_range_for,
    is_attention_out_proj,
    layer_index,
    numel_of,
    resize_conv2d_spatial,
    resize_pos_embed_2d,
    unpack_moe_state_dict,
)

Family = Literal["auto", "generic", "siglip", "qwen_moe", "mmdit"]

# Default weight-seed sources (not necessarily the same as design baselines).
# No official Qwen3 MoE exists in the ~1.1–1.6B band; Qwen3-1.7B (~1.7B dense)
# is the closest sensible official checkpoint for arch-aware dense→MoE transfer.
DEFAULT_QWEN_SEED_ID = "Qwen/Qwen3-1.7B"


def select_candidate(candidate_cols, preset=None):
    """Return the column dict chosen by the user."""
    n = len(candidate_cols)
    if n == 0:
        raise SystemExit("No buildable candidates were found for this target.")

    if preset is not None:
        idx = int(preset)
    else:
        print(
            f"\nSelect a candidate configuration [1-{n}] "
            f"(the columns 'cand 1' .. 'cand {n}' above):"
        )
        while True:
            raw = input(f"Candidate number [1-{n}] (default 1): ").strip()
            if raw == "":
                idx = 1
                break
            try:
                idx = int(raw)
            except ValueError:
                print("  Please enter a number.")
                continue
            if 1 <= idx <= n:
                break
            print(f"  Out of range, choose between 1 and {n}.")

    if not (1 <= idx <= n):
        raise SystemExit(f"Candidate {idx} is out of range (1-{n}).")

    chosen = candidate_cols[idx - 1]
    print(
        f"Selected {chosen['header']} "
        f"({chosen['params']:,} params, {chosen['params'] / 1e9:.4f}B)."
    )
    return chosen


def prompt_seed_model(preset=None, default=None):
    """Return the HuggingFace id / local path of the model to seed weights from."""
    if preset is not None:
        seed_id = preset
    else:
        suffix = f" (default {default})" if default else ""
        seed_id = input(
            f"\nModel to download and seed parameters from{suffix}: "
        ).strip()
        if seed_id == "" and default:
            seed_id = default
    if not seed_id:
        raise SystemExit("A seed model id is required.")
    print(f"Seeding parameters from: {seed_id}")
    return seed_id


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_seed_report(output_dir: str | Path, report: dict[str, Any]) -> Path:
    path = Path(output_dir) / "seed_report.json"
    # Make JSON-safe (no tensors).
    safe = {
        k: v
        for k, v in report.items()
        if k not in ("copied_keys", "fresh_keys") or True
    }
    # Cap key lists for readability
    if len(safe.get("copied_keys", [])) > 200:
        safe["copied_keys"] = safe["copied_keys"][:200] + ["...truncated..."]
    if len(safe.get("fresh_keys", [])) > 200:
        safe["fresh_keys"] = safe["fresh_keys"][:200] + ["...truncated..."]
    path.write_text(json.dumps(safe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def report_seeding(stats: dict[str, Any]) -> None:
    print("\nParameter seeding summary:")
    print(f"  copied exactly       : {stats.get('exact', 0)}")
    print(f"  arch-aware transfer  : {stats.get('arch_aware', 0)}")
    print(f"  partial fallback     : {stats.get('partial_fallback', 0)}")
    print(f"  incompatible shapes  : {stats.get('shape_mismatch', 0)}")
    print(f"  freshly initialized  : {stats.get('fresh', 0)}")
    print(f"  total target tensors : {stats.get('total_target', 0)}")
    src_m = float(stats.get("param_mass_source", 0))
    fr_m = float(stats.get("param_mass_fresh", 0))
    tot = src_m + fr_m
    if tot > 0:
        print(
            f"  param mass from src  : {src_m / 1e9:.3f}B "
            f"({100.0 * src_m / tot:.1f}%)"
        )
        print(f"  param mass fresh     : {fr_m / 1e9:.3f}B ({100.0 * fr_m / tot:.1f}%)")
    if stats.get("warnings"):
        print(f"  warnings ({len(stats['warnings'])}):")
        for w in stats["warnings"][:12]:
            print(f"    - {w}")
        if len(stats["warnings"]) > 12:
            print(f"    ... +{len(stats['warnings']) - 12} more")
    if stats.get("by_component"):
        print("  by component:")
        for comp, bucket in sorted(stats["by_component"].items()):
            print(f"    {comp:10s} {bucket}")


def detect_family(target_model: torch.nn.Module) -> Family:
    name = type(target_model).__name__.lower()
    cfg = getattr(target_model, "config", None)
    model_type = str(getattr(cfg, "model_type", "") or "").lower()
    if "siglip" in name or "siglip" in model_type:
        return "siglip"
    if "qwen" in name or "qwen" in model_type:
        return "qwen_moe"
    if "sd3" in name or "transformer2d" in name or "mmdit" in name:
        return "mmdit"
    cls = getattr(cfg, "_class_name", "") or ""
    if "SD3" in cls:
        return "mmdit"
    return "generic"


def _infer_head_dim(target_model: torch.nn.Module, source_model: torch.nn.Module) -> int:
    for m in (target_model, source_model):
        cfg = getattr(m, "config", None)
        if cfg is None:
            continue
        hd = getattr(cfg, "head_dim", None)
        if hd:
            return int(hd)
        hidden = getattr(cfg, "hidden_size", None)
        heads = getattr(cfg, "num_attention_heads", None)
        if hidden and heads and int(hidden) % int(heads) == 0:
            return int(hidden) // int(heads)
        # SD3
        ahd = getattr(cfg, "attention_head_dim", None)
        if ahd:
            return int(ahd)
    return 64


def _source_max_layer(source_state: dict[str, torch.Tensor]) -> int:
    mx = -1
    for k in source_state:
        li = layer_index(k)
        if li is not None:
            mx = max(mx, li)
    return mx


def _target_max_layer(target_state: dict[str, torch.Tensor]) -> int:
    return _source_max_layer(target_state)


def _clone_block_keys(
    target_state: dict[str, torch.Tensor],
    donor_layer: int,
    new_layer: int,
) -> list[tuple[str, str]]:
    """Map target keys in ``new_layer`` to donor keys in ``donor_layer``."""
    pairs = []
    for name in target_state:
        li = layer_index(name)
        if li != new_layer:
            continue
        donor = re.sub(
            rf"(encoder\.layers|layers|transformer_blocks)\.{new_layer}\.",
            rf"\1.{donor_layer}.",
            name,
        )
        if donor in target_state or True:
            pairs.append((name, donor))
    return pairs


@torch.no_grad()
def _transfer_one(
    dst: torch.Tensor,
    src: torch.Tensor,
    name: str,
    *,
    family: Family,
    head_dim: int,
    init_range: float,
    generator: torch.Generator | None,
) -> tuple[str, int]:
    """Transfer one tensor; return (kind, source_mass_elements)."""
    if tuple(dst.shape) == tuple(src.shape):
        copy_exact(dst, src)
        return "exact", numel_of(tuple(dst.shape))

    if dst.ndim != src.ndim:
        return "shape_mismatch", 0

    # --- family-specific high-value remaps ---
    lname = name.lower()

    # SigLIP / vision patch conv
    if family == "siglip" and "patch_embedding.weight" in name and dst.ndim == 4:
        kind = resize_conv2d_spatial(
            dst, src, mode="bilinear", init_range=init_range, generator=generator
        )
        sm = numel_of(tuple(min(a, b) for a, b in zip(dst.shape, src.shape)))
        # rough: after resize source contributes full spatial of min channels
        sm = min(dst.shape[0], src.shape[0]) * min(dst.shape[1], src.shape[1]) * dst.shape[2] * dst.shape[3]
        return kind, sm

    if family == "siglip" and "patch_embedding.bias" in name:
        kind = copy_linear_width(dst, src, init_range=init_range, generator=generator)
        sm = min(numel_of(tuple(dst.shape)), numel_of(tuple(src.shape)))
        return kind, sm

    # Position embeddings (SigLIP token pos or MMDiT pos_embed.pos_embed)
    if (
        "position_embedding.weight" in name
        or name.endswith("pos_embed.pos_embed")
        or (family == "mmdit" and name.endswith("pos_embed") and dst.ndim == 2)
    ):
        kind = resize_pos_embed_2d(
            dst, src, mode="bilinear", init_range=init_range, generator=generator
        )
        sm = min(dst.shape[0], src.shape[0]) * min(dst.shape[-1], src.shape[-1])
        return kind, sm

    # Attention projections — head-aware when 2D
    if is_attention_out_proj(name) and dst.ndim == 2 and src.ndim == 2:
        kind = copy_heads_out_in(
            dst,
            src,
            head_dim=head_dim,
            init_range=init_range,
            generator=generator,
        )
        n_heads_d = dst.shape[0] // head_dim if head_dim and dst.shape[0] % head_dim == 0 else 0
        n_heads_s = src.shape[0] // head_dim if head_dim and src.shape[0] % head_dim == 0 else 0
        if n_heads_d and n_heads_s:
            sm = min(n_heads_d, n_heads_s) * head_dim * min(dst.shape[1], src.shape[1])
        else:
            sm = numel_of(tuple(min(a, b) for a, b in zip(dst.shape, src.shape)))
        return kind, sm

    # Generic width / leading overlap with re-init of new region
    kind = copy_leading_overlap(
        dst, src, init_new=True, init_range=init_range, generator=generator
    )
    if kind == "shape_mismatch":
        return kind, 0
    sm = numel_of(tuple(min(a, b) for a, b in zip(dst.shape, src.shape)))
    # rename generic partial to arch_aware when we re-inited
    if kind == "arch_aware":
        return "arch_aware", sm
    if any(a != b for a, b in zip(dst.shape, src.shape)):
        return "arch_aware", sm
    return kind, sm


def _apply_dense_mlp_to_experts(
    target_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    report: dict[str, Any],
    *,
    init_range: float,
    generator: torch.Generator | None,
) -> None:
    """When source is dense, copy layer MLP into each MoE expert with noise."""
    # source: model.layers.L.mlp.{gate,up,down}_proj.weight
    # target: model.layers.L.mlp.experts.i.{gate,up,down}_proj.weight
    dense_mlp = re.compile(
        r"^(?P<pre>.*\.layers\.(?P<L>\d+)\.mlp)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
    )
    src_mlp: dict[tuple[str, str], torch.Tensor] = {}
    for k, v in source_state.items():
        m = dense_mlp.match(k)
        if m:
            src_mlp[(m.group("pre"), m.group("proj"))] = v

    expert_re = re.compile(
        r"^(?P<pre>.*\.layers\.(?P<L>\d+)\.mlp)\.experts\.(?P<i>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
    )
    for name, dst in target_state.items():
        m = expert_re.match(name)
        if not m:
            continue
        if name in report.get("_handled", set()):
            continue
        src = src_mlp.get((m.group("pre"), m.group("proj")))
        if src is None:
            continue
        kind, sm = _transfer_one(
            dst,
            src,
            name,
            family="qwen_moe",
            head_dim=64,
            init_range=init_range,
            generator=generator,
        )
        # Add expert-index-dependent noise so experts are not clones.
        noise_scale = init_range_for(dst, init_range) * 0.35
        noise = torch.empty_like(dst)
        noise.normal_(0.0, noise_scale)
        # Keep transferred overlap dominant: only add noise on full tensor lightly.
        dst.add_(noise)
        te = numel_of(tuple(dst.shape))
        if kind == "shape_mismatch":
            bump(report, "shape_mismatch", name, tuple(dst.shape))
        else:
            # Count as arch_aware dense→expert
            if kind == "exact":
                kind = "arch_aware"
            bump(report, kind, name, tuple(dst.shape), source_mass=sm, fresh_mass=te - sm)
        report.setdefault("_handled", set()).add(name)


@torch.no_grad()
def _fill_new_layers_from_donor(
    target_state: dict[str, torch.Tensor],
    report: dict[str, Any],
    *,
    src_max_layer: int,
    tgt_max_layer: int,
    init_range: float,
    generator: torch.Generator | None,
) -> None:
    """Clone last source-aligned layer into deeper target layers; zero residual outs."""
    if src_max_layer < 0 or tgt_max_layer <= src_max_layer:
        return
    donor = src_max_layer
    for new_l in range(src_max_layer + 1, tgt_max_layer + 1):
        for name, dst in target_state.items():
            if layer_index(name) != new_l:
                continue
            if name in report.get("_handled", set()):
                continue
            donor_name = re.sub(
                rf"(encoder\.layers|layers|transformer_blocks)\.{new_l}\.",
                rf"\1.{donor}.",
                name,
            )
            donor_t = target_state.get(donor_name)
            if donor_t is None or tuple(donor_t.shape) != tuple(dst.shape):
                # fresh already from HF init
                te = numel_of(tuple(dst.shape))
                bump(report, "fresh", name, tuple(dst.shape), source_mass=0, fresh_mass=te)
                report.setdefault("_handled", set()).add(name)
                continue
            copy_exact(dst, donor_t)
            # Zero residual path projections so the new block starts near identity.
            if any(
                s in name
                for s in (
                    "o_proj",
                    "out_proj",
                    "down_proj",
                    "fc2",
                    "proj_out",
                    "to_out",
                )
            ):
                dst.mul_(0.0)
            te = numel_of(tuple(dst.shape))
            bump(report, "arch_aware", name, tuple(dst.shape), source_mass=te, fresh_mass=0)
            report.setdefault("_handled", set()).add(name)
        report["warnings"].append(
            f"new layer {new_l} cloned from layer {donor} (residual outs zeroed where matched)"
        )


@torch.no_grad()
def seed_parameters(
    target_model: torch.nn.Module,
    source_model: torch.nn.Module,
    *,
    family: Family = "auto",
    init_range: float = 0.02,
    expert_max_cos: float = 0.3,
    strict_experts: bool = False,
    generator: torch.Generator | None = None,
    seed_model_id: str | None = None,
) -> dict[str, Any]:
    """Architecture-aware weight transfer from ``source_model`` into ``target_model``."""
    if family == "auto":
        family = detect_family(target_model)

    report = empty_report()
    report["family"] = family
    report["seed_model_id"] = seed_model_id
    report["policies"] = {
        "init_range": init_range,
        "expert_max_cos": expert_max_cos,
        "strict_experts": strict_experts,
    }
    report["_handled"] = set()

    source_state = dict(source_model.state_dict())
    # MoE: unpack packed experts so names match per-expert targets.
    if family in ("qwen_moe", "auto", "generic"):
        source_state = unpack_moe_state_dict(source_state)

    target_state = target_model.state_dict()
    report["total_target"] = len(target_state)
    head_dim = _infer_head_dim(target_model, source_model)
    report["policies"]["head_dim"] = head_dim

    src_max = _source_max_layer(source_state)
    tgt_max = _target_max_layer(target_state)

    # --- Pass 1: name-matched transfer ---
    for name, tparam in target_state.items():
        sparam = source_state.get(name)
        if sparam is None:
            continue
        kind, sm = _transfer_one(
            tparam,
            sparam,
            name,
            family=family if family != "auto" else detect_family(target_model),
            head_dim=head_dim,
            init_range=init_range,
            generator=generator,
        )
        te = numel_of(tuple(tparam.shape))
        if kind == "shape_mismatch":
            bump(report, "shape_mismatch", name, tuple(tparam.shape))
        else:
            bump(
                report,
                kind if kind != "partial_fallback" else "partial_fallback",
                name,
                tuple(tparam.shape),
                source_mass=sm,
                fresh_mass=max(te - sm, 0),
            )
        report["_handled"].add(name)

    # --- Pass 2: dense→MoE expert broadcast (qwen) ---
    if family == "qwen_moe":
        # Only if we still have unhandled experts and source has dense MLP.
        unhandled_experts = [
            n
            for n in target_state
            if "experts." in n and n not in report["_handled"]
        ]
        if unhandled_experts:
            _apply_dense_mlp_to_experts(
                target_state,
                source_state,
                report,
                init_range=init_range,
                generator=generator,
            )

    # --- Pass 3: deeper layers ---
    _fill_new_layers_from_donor(
        target_state,
        report,
        src_max_layer=src_max,
        tgt_max_layer=tgt_max,
        init_range=init_range,
        generator=generator,
    )

    # --- Pass 4: remaining = fresh (already HF-init); just account ---
    for name, tparam in target_state.items():
        if name in report["_handled"]:
            continue
        te = numel_of(tuple(tparam.shape))
        bump(report, "fresh", name, tuple(tparam.shape), source_mass=0, fresh_mass=te)
        report["_handled"].add(name)

    # Load back into module
    target_model.load_state_dict(target_state)

    # Expert diversity on live state
    if family == "qwen_moe" or any("experts." in k for k in target_state):
        live = dict(target_model.state_dict())
        warnings = ensure_expert_diversity(
            live,
            max_cos=expert_max_cos,
            init_range=init_range,
            generator=generator,
        )
        if warnings:
            report["warnings"].extend(warnings)
            target_model.load_state_dict(live)
            # re-check
            live2 = dict(target_model.state_dict())
            from model_seeding_ops import cosine_flat

            bad = []
            for pre in expert_prefixes(live2):
                idxs = []
                for k in live2:
                    if k.startswith(pre + ".") and k.endswith(".gate_proj.weight"):
                        mid = k[len(pre) + 1 : -len(".gate_proj.weight")]
                        if mid.isdigit():
                            idxs.append(int(mid))
                idxs = sorted(set(idxs))
                for i, a in enumerate(idxs):
                    wa = live2.get(f"{pre}.{a}.gate_proj.weight")
                    if wa is None:
                        continue
                    for b in idxs[i + 1 :]:
                        wb = live2.get(f"{pre}.{b}.gate_proj.weight")
                        if wb is None:
                            continue
                        c = cosine_flat(wa, wb)
                        if c > expert_max_cos:
                            bad.append((pre, a, b, c))
            if bad:
                msg = (
                    f"expert diversity still violated for {len(bad)} pairs "
                    f"(e.g. {bad[0]})"
                )
                report["warnings"].append(msg)
                if strict_experts:
                    raise RuntimeError(msg)

    # Cleanup internal
    report.pop("_handled", None)
    report["validation"] = {
        "ok": not any("still violated" in w for w in report.get("warnings", [])),
        "family": family,
    }
    return report


# Back-compat alias: old call signature seed_parameters(target, source)
def seed_parameters_legacy(target_model, source_model):
    return seed_parameters(target_model, source_model, family="generic")


def create_seeded_model(
    search_fn,
    source_loader: Callable[[str], torch.nn.Module],
    output_dir: str,
    default_seed=None,
    candidate=None,
    seed_model=None,
    search_kwargs=None,
    *,
    family: Family = "auto",
    init_range: float = 0.02,
    expert_max_cos: float = 0.3,
    strict_experts: bool = False,
    torch_seed: int | None = 42,
):
    """End-to-end: search → pick cand → load source → arch-aware seed → save."""
    search_kwargs = dict(search_kwargs or {})
    baseline_col, candidate_cols = search_fn(**search_kwargs)

    chosen = select_candidate(candidate_cols, preset=candidate)
    seed_id = prompt_seed_model(preset=seed_model, default=default_seed)

    print(
        "\nBuilding the target model from the selected configuration "
        "(this materializes real weights) ..."
    )
    target = chosen["build_fn"](chosen["cfg_obj"])
    print(
        f"Target model built: {sum(p.numel() for p in target.parameters()):,} params."
    )

    print(
        "Loading the source model to seed parameters from (this downloads "
        "the checkpoint) ..."
    )
    source = source_loader(seed_id)

    gen = None
    if torch_seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(torch_seed))

    fam = family
    if fam == "auto":
        fam = detect_family(target)

    print(f"Architecture-aware seeding (family={fam}) ...")
    stats = seed_parameters(
        target,
        source,
        family=fam,
        init_range=init_range,
        expert_max_cos=expert_max_cos,
        strict_experts=strict_experts,
        generator=gen,
        seed_model_id=seed_id,
    )
    # Attach config snapshot
    cfg = getattr(target, "config", None)
    if cfg is not None:
        try:
            stats["target_config"] = dict(cfg.to_dict()) if hasattr(cfg, "to_dict") else dict(cfg)
        except Exception:
            stats["target_config"] = {"class": type(cfg).__name__}

    report_seeding(stats)

    ensure_output_dir(output_dir)
    target.save_pretrained(output_dir)
    report_path = write_seed_report(output_dir, stats)
    print(f"\nSaved seeded model to: {output_dir}")
    print(f"Seed report: {report_path}")
    return target, stats
