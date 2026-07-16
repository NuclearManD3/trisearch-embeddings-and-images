"""Unit tests for architecture-aware model seeding."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from model_seeding import detect_family, seed_parameters
from model_seeding_ops import (
    copy_heads_out_in,
    copy_leading_overlap,
    cosine_flat,
    ensure_expert_diversity,
    resize_conv2d_spatial,
    resize_pos_embed_2d,
    unpack_moe_state_dict,
)


class TestResizeOps(unittest.TestCase):
    def test_conv_spatial_resize_preserves_channel_overlap(self):
        src = torch.randn(8, 3, 14, 14)
        dst = torch.zeros(12, 3, 18, 18)
        kind = resize_conv2d_spatial(dst, src, mode="bilinear")
        self.assertEqual(kind, "arch_aware")
        # Not a pure corner copy of 14x14 into 18x18 (interpolated values differ
        # from raw src at overlapping spatial coords in general).
        self.assertEqual(dst.shape, (12, 3, 18, 18))
        self.assertTrue(dst[:8].abs().sum() > 0)
        self.assertTrue(dst[8:].abs().sum() > 0)  # new channels noisily init

    def test_pos_embed_2d_grid(self):
        # 4x4 grid, D=8 → 5x5 grid, D=10
        src = torch.randn(16, 8)
        dst = torch.zeros(25, 10)
        kind = resize_pos_embed_2d(dst, src, mode="bilinear")
        self.assertEqual(kind, "arch_aware")
        self.assertTrue(dst[:25, :8].abs().sum() > 0)

    def test_head_copy_keeps_whole_heads(self):
        head_dim = 4
        src = torch.randn(3 * head_dim, 10)  # 3 heads
        dst = torch.zeros(5 * head_dim, 12)  # 5 heads, wider in
        kind = copy_heads_out_in(dst, src, head_dim=head_dim)
        self.assertEqual(kind, "arch_aware")
        for h in range(3):
            self.assertTrue(
                torch.allclose(
                    dst[h * head_dim : (h + 1) * head_dim, :10],
                    src[h * head_dim : (h + 1) * head_dim, :10],
                )
            )


class TestMoEUnpack(unittest.TestCase):
    def test_unpack_gate_up_and_down(self):
        e, inter, h = 4, 6, 8
        gu = torch.randn(e, 2 * inter, h)
        down = torch.randn(e, h, inter)
        packed = {
            "model.layers.0.mlp.experts.gate_up_proj": gu,
            "model.layers.0.mlp.experts.down_proj": down,
            "model.layers.0.self_attn.q_proj.weight": torch.randn(h, h),
        }
        out = unpack_moe_state_dict(packed)
        self.assertIn("model.layers.0.mlp.experts.0.gate_proj.weight", out)
        self.assertIn("model.layers.0.mlp.experts.3.up_proj.weight", out)
        self.assertIn("model.layers.0.mlp.experts.1.down_proj.weight", out)
        self.assertTrue(
            torch.allclose(
                out["model.layers.0.mlp.experts.2.gate_proj.weight"],
                gu[2, :inter, :],
            )
        )
        self.assertTrue(
            torch.allclose(
                out["model.layers.0.mlp.experts.2.down_proj.weight"],
                down[2],
            )
        )
        self.assertIn("model.layers.0.self_attn.q_proj.weight", out)

    def test_expert_diversity_breaks_clones(self):
        h, inter = 16, 8
        base = torch.randn(inter, h)
        state = {}
        for i in range(4):
            # Nearly identical experts
            state[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = (
                base + 1e-6 * torch.randn(inter, h)
            )
            state[f"model.layers.0.mlp.experts.{i}.up_proj.weight"] = torch.randn(
                inter, h
            )
            state[f"model.layers.0.mlp.experts.{i}.down_proj.weight"] = torch.randn(
                h, inter
            )
        # Confirm clones are highly similar
        self.assertGreater(
            cosine_flat(
                state["model.layers.0.mlp.experts.0.gate_proj.weight"],
                state["model.layers.0.mlp.experts.1.gate_proj.weight"],
            ),
            0.9,
        )
        warnings = ensure_expert_diversity(state, max_cos=0.3, init_range=0.05)
        self.assertTrue(warnings)
        c = cosine_flat(
            state["model.layers.0.mlp.experts.0.gate_proj.weight"],
            state["model.layers.0.mlp.experts.1.gate_proj.weight"],
        )
        self.assertLess(c, 0.3)


class TinySiglipLike(nn.Module):
    def __init__(self, hidden=32, layers=2, patches=16, patch=4, ch=3):
        super().__init__()
        self.config = type(
            "C",
            (),
            {
                "model_type": "siglip_vision_model",
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": 4,
                "image_size": int(math.sqrt(patches)) * patch,
                "patch_size": patch,
            },
        )()
        self.patch = nn.Conv2d(ch, hidden, kernel_size=patch, stride=patch)
        self.pos = nn.Embedding(patches, hidden)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "q": nn.Linear(hidden, hidden),
                        "mlp": nn.Linear(hidden, hidden),
                    }
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x):
        return x


def _siglip_state(m: TinySiglipLike) -> dict:
    # Flatten names a bit like HF
    sd = {}
    sd["vision_model.embeddings.patch_embedding.weight"] = m.patch.weight.data
    sd["vision_model.embeddings.patch_embedding.bias"] = m.patch.bias.data
    sd["vision_model.embeddings.position_embedding.weight"] = m.pos.weight.data
    for i, layer in enumerate(m.layers):
        sd[f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = layer[
            "q"
        ].weight.data
        sd[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = layer["mlp"].weight.data
    return sd


class TinySiglipWrapper(nn.Module):
    """Minimal module whose state_dict keys look like SigLIP."""

    def __init__(self, hidden=32, layers=2, n_pos=16, patch=4):
        super().__init__()
        self.config = type(
            "C",
            (),
            {
                "model_type": "siglip_vision_model",
                "hidden_size": hidden,
                "num_attention_heads": 4,
                "head_dim": hidden // 4,
            },
        )()
        self.vision_model = nn.Module()
        self.vision_model.embeddings = nn.Module()
        self.vision_model.embeddings.patch_embedding = nn.Conv2d(
            3, hidden, kernel_size=patch, stride=patch
        )
        self.vision_model.embeddings.position_embedding = nn.Embedding(n_pos, hidden)
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.ModuleDict(
                            {"q_proj": nn.Linear(hidden, hidden, bias=False)}
                        ),
                        "mlp": nn.ModuleDict(
                            {"fc1": nn.Linear(hidden, hidden, bias=False)}
                        ),
                    }
                )
                for _ in range(layers)
            ]
        )
        # Register nested ModuleDicts properly for state_dict
        for i, layer in enumerate(self.vision_model.encoder.layers):
            # re-wrap as Module
            block = nn.Module()
            sa = nn.Module()
            sa.q_proj = layer["self_attn"]["q_proj"]
            block.self_attn = sa
            mlp = nn.Module()
            mlp.fc1 = layer["mlp"]["fc1"]
            block.mlp = mlp
            self.vision_model.encoder.layers[i] = block


class TestSeedParametersIntegration(unittest.TestCase):
    def test_detect_family(self):
        m = TinySiglipWrapper()
        self.assertEqual(detect_family(m), "siglip")

    def test_siglip_seed_resizes_patch_and_grows_depth(self):
        # Source: smaller hidden, fewer layers, smaller patch/pos
        src = TinySiglipWrapper(hidden=16, layers=1, n_pos=9, patch=2)  # 3x3 grid
        tgt = TinySiglipWrapper(hidden=24, layers=3, n_pos=16, patch=4)  # 4x4
        # Use fixed nonzero source weights
        with torch.no_grad():
            src.vision_model.embeddings.patch_embedding.weight.fill_(0.5)
            src.vision_model.embeddings.position_embedding.weight.normal_(0, 0.1)
            for layer in src.vision_model.encoder.layers:
                layer.self_attn.q_proj.weight.normal_(0, 0.1)
                layer.mlp.fc1.weight.normal_(0, 0.1)

        report = seed_parameters(tgt, src, family="siglip", init_range=0.02)
        self.assertGreater(report["arch_aware"] + report["exact"], 0)
        self.assertIn(report["family"], ("siglip",))
        # Layer 2 should have been handled (clone or fresh)
        self.assertEqual(
            report["exact"] + report["arch_aware"] + report["fresh"] + report.get(
                "partial_fallback", 0
            )
            + report.get("shape_mismatch", 0),
            report["total_target"],
        )

    def test_exact_copy_when_shapes_match(self):
        src = TinySiglipWrapper(hidden=16, layers=2, n_pos=16, patch=4)
        tgt = TinySiglipWrapper(hidden=16, layers=2, n_pos=16, patch=4)
        with torch.no_grad():
            for p in src.parameters():
                p.normal_(0, 0.05)
        report = seed_parameters(tgt, src, family="siglip")
        self.assertGreater(report["exact"], 0)
        # Spot-check one tensor
        self.assertTrue(
            torch.allclose(
                tgt.vision_model.encoder.layers[0].self_attn.q_proj.weight,
                src.vision_model.encoder.layers[0].self_attn.q_proj.weight,
            )
        )


class TinyQwenMoE(nn.Module):
    def __init__(self, hidden=32, layers=2, experts=4, inter=16, heads=4, head_dim=8):
        super().__init__()
        self.config = type(
            "C",
            (),
            {
                "model_type": "qwen3_moe",
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "head_dim": head_dim,
                "num_local_experts": experts,
            },
        )()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(100, hidden)
        self.model.layers = nn.ModuleList()
        for _ in range(layers):
            layer = nn.Module()
            sa = nn.Module()
            sa.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
            sa.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
            layer.self_attn = sa
            mlp = nn.Module()
            experts_mod = nn.ModuleList()
            for _e in range(experts):
                ex = nn.Module()
                ex.gate_proj = nn.Linear(hidden, inter, bias=False)
                ex.up_proj = nn.Linear(hidden, inter, bias=False)
                ex.down_proj = nn.Linear(inter, hidden, bias=False)
                experts_mod.append(ex)
            mlp.experts = experts_mod
            mlp.gate = nn.Linear(hidden, experts, bias=False)
            layer.mlp = mlp
            self.model.layers.append(layer)


class TinyDenseQwen(nn.Module):
    def __init__(self, hidden=32, layers=2, inter=16, heads=4, head_dim=8):
        super().__init__()
        self.config = type(
            "C",
            (),
            {
                "model_type": "qwen3",
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "head_dim": head_dim,
            },
        )()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(100, hidden)
        self.model.layers = nn.ModuleList()
        for _ in range(layers):
            layer = nn.Module()
            sa = nn.Module()
            sa.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
            sa.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
            layer.self_attn = sa
            mlp = nn.Module()
            mlp.gate_proj = nn.Linear(hidden, inter, bias=False)
            mlp.up_proj = nn.Linear(hidden, inter, bias=False)
            mlp.down_proj = nn.Linear(inter, hidden, bias=False)
            layer.mlp = mlp
            self.model.layers.append(layer)


class TestQwenDenseToMoE(unittest.TestCase):
    def test_dense_to_moe_seeds_experts_with_diversity(self):
        src = TinyDenseQwen(hidden=32, layers=2, inter=16, heads=4, head_dim=8)
        tgt = TinyQwenMoE(
            hidden=32, layers=2, experts=4, inter=16, heads=4, head_dim=8
        )
        with torch.no_grad():
            for p in src.parameters():
                p.normal_(0, 0.05)
        report = seed_parameters(
            tgt, src, family="qwen_moe", init_range=0.05, expert_max_cos=0.35
        )
        self.assertGreater(
            report["exact"] + report["arch_aware"], 0, msg=str(report)
        )
        # Experts should not be near-clones after diversity pass
        e0 = tgt.model.layers[0].mlp.experts[0].gate_proj.weight
        e1 = tgt.model.layers[0].mlp.experts[1].gate_proj.weight
        self.assertLess(cosine_flat(e0, e1), 0.35)
        self.assertTrue(report["validation"]["ok"])


class TestCopyLeading(unittest.TestCase):
    def test_overlap_exact(self):
        src = torch.randn(4, 6)
        dst = torch.zeros(7, 9)
        copy_leading_overlap(dst, src, init_new=True, init_range=0.01)
        self.assertTrue(torch.allclose(dst[:4, :6], src))


if __name__ == "__main__":
    unittest.main()
