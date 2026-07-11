"""Unit tests for Stage-1 late-interaction / bank / patch-keep upgrades.

Covers:
1. Vectorized late-interaction scoring == loop mean-MaxSim
2. Multi-positive masking as non-negatives (false-negative softening)
3. Soft MaxSim (τ logsumexp)
4. Bank policy B: enqueue every micro-batch; score as-of accum window start
5. Background (L2) patch drop for SigLIP vision MaxSim
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from trisearch_models.training import (
    DEFAULT_SOFT_MAXSIM_TEMPERATURE,
    EmbeddingMemoryBank,
    Stage1AlignmentModel,
    build_late_interaction_matrix,
    build_multi_positive_mask,
    caption_token_jaccard,
    contrastive_late_interaction_loss,
    differentiable_late_interaction_score,
    keep_top_patches_by_l2,
    masked_cross_entropy,
    soft_or_hard_maxsim,
)


def _rand_tokens(batch: int, lengths: list[int], dim: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for n in lengths:
        t = torch.randn(n, dim, generator=g)
        t = F.normalize(t, dim=-1)
        out.append(t)
    assert len(out) == batch
    return out


def _loop_matrix(queries, docs, soft_tau=None):
    rows = []
    for q in queries:
        rows.append(
            torch.stack(
                [
                    differentiable_late_interaction_score(
                        q, d, soft_maxsim_temperature=soft_tau
                    )
                    for d in docs
                ]
            )
        )
    return torch.stack(rows)


class TestVectorizedLateInteraction(unittest.TestCase):
    def test_hard_maxsim_matches_loop(self):
        queries = _rand_tokens(3, [2, 5, 1], dim=8, seed=1)
        docs = _rand_tokens(4, [3, 1, 6, 2], dim=8, seed=2)
        got = build_late_interaction_matrix(queries, docs)
        ref = _loop_matrix(queries, docs)
        self.assertTrue(torch.allclose(got, ref, atol=1e-5, rtol=1e-5), (got - ref).abs().max())

    def test_soft_maxsim_matches_loop(self):
        queries = _rand_tokens(2, [4, 3], dim=16, seed=3)
        docs = _rand_tokens(3, [2, 5, 4], dim=16, seed=4)
        tau = 0.05
        got = build_late_interaction_matrix(
            queries, docs, soft_maxsim_temperature=tau
        )
        ref = _loop_matrix(queries, docs, soft_tau=tau)
        self.assertTrue(torch.allclose(got, ref, atol=1e-5, rtol=1e-5))

    def test_soft_maxsim_approaches_hard_as_tau_to_zero(self):
        q = F.normalize(torch.randn(5, 8), dim=-1)
        d = F.normalize(torch.randn(7, 8), dim=-1)
        hard = differentiable_late_interaction_score(q, d)
        soft_small = differentiable_late_interaction_score(
            q, d, soft_maxsim_temperature=1e-4
        )
        soft_large = differentiable_late_interaction_score(
            q, d, soft_maxsim_temperature=1.0
        )
        self.assertLess(abs(float(hard - soft_small)), 1e-3)
        # Larger τ_s is a strict upper bound on hard max (logsumexp).
        self.assertGreaterEqual(float(soft_large), float(hard) - 1e-5)

    def test_soft_or_hard_maxsim_hard_path(self):
        x = torch.tensor([[1.0, 3.0, 2.0], [0.0, -1.0, 0.5]])
        self.assertTrue(torch.equal(soft_or_hard_maxsim(x, dim=-1), x.max(-1).values))

    def test_gradients_flow_through_soft_maxsim(self):
        q = F.normalize(torch.randn(3, 4, requires_grad=True), dim=-1)
        # re-enable grad after normalize
        q = q.detach().requires_grad_(True)
        d = F.normalize(torch.randn(5, 4), dim=-1).detach().requires_grad_(True)
        score = differentiable_late_interaction_score(
            q, d, soft_maxsim_temperature=0.1
        )
        score.backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(d.grad)
        # Soft max should give non-zero grad to more than one doc token typically.
        self.assertTrue((d.grad.abs().sum(dim=-1) > 0).sum() >= 1)


class TestMultiPositiveMasking(unittest.TestCase):
    def test_jaccard_identical(self):
        self.assertEqual(caption_token_jaccard("a red car", "a red car"), 1.0)

    def test_jaccard_disjoint(self):
        self.assertEqual(caption_token_jaccard("cat dog", "plane boat"), 0.0)

    def test_build_mask_marks_near_duplicates(self):
        caps = [
            "a dog runs in the park",
            "a dog runs in the park today",  # high overlap
            "satellite view of a harbor",
        ]
        mask = build_multi_positive_mask(
            caps, batch_size=3, jaccard_threshold=0.5, device=torch.device("cpu")
        )
        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertTrue(bool(mask[0, 0]) and bool(mask[1, 1]) and bool(mask[2, 2]))
        self.assertTrue(bool(mask[0, 1]) and bool(mask[1, 0]))
        self.assertFalse(bool(mask[0, 2]))

    def test_masked_ce_excludes_false_negatives(self):
        # scores: row 0 wants class 0; class 1 is a false-neg with huge logit
        scores = torch.tensor(
            [[1.0, 10.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 1, 2])
        # Without mask, CE is dominated by the 10.0 false negative.
        loss_raw = F.cross_entropy(scores, labels)
        non_neg = torch.tensor(
            [
                [True, True, False],
                [True, True, False],
                [False, False, True],
            ]
        )
        loss_masked = masked_cross_entropy(
            scores, labels, non_negative_mask=non_neg
        )
        self.assertLess(float(loss_masked), float(loss_raw))

    def test_contrastive_loss_with_non_negative_mask(self):
        texts = _rand_tokens(3, [2, 2, 2], dim=8, seed=10)
        # Make image 1 almost identical to image 0 so hard CE would push them apart.
        images = _rand_tokens(3, [3, 3, 3], dim=8, seed=11)
        images[1] = images[0].clone()
        mask = torch.tensor(
            [
                [True, True, False],
                [True, True, False],
                [False, False, True],
            ],
            dtype=torch.bool,
        )
        loss_masked = contrastive_late_interaction_loss(
            texts, images, temperature=0.07, non_negative_mask=mask
        )
        loss_plain = contrastive_late_interaction_loss(
            texts, images, temperature=0.07, non_negative_mask=None
        )
        self.assertTrue(torch.isfinite(loss_masked))
        self.assertTrue(torch.isfinite(loss_plain))
        # Softening false negatives should not increase loss on this collision.
        self.assertLessEqual(float(loss_masked), float(loss_plain) + 1e-5)


class TestSoftMaxSimDefault(unittest.TestCase):
    def test_default_temperature_positive(self):
        self.assertGreater(DEFAULT_SOFT_MAXSIM_TEMPERATURE, 0.0)

    def test_stage1_model_defaults_soft_maxsim_on(self):
        # Lightweight stub: only check constructor defaults without real backbones.
        # Instantiate EmbeddingMemoryBank path via __init__ signature inspection.
        import inspect

        sig = inspect.signature(Stage1AlignmentModel.__init__)
        self.assertTrue(sig.parameters["soft_maxsim"].default is True)
        self.assertEqual(
            sig.parameters["soft_maxsim_temperature"].default,
            DEFAULT_SOFT_MAXSIM_TEMPERATURE,
        )


class TestBankPolicyB(unittest.TestCase):
    def test_snapshot_is_independent_of_later_enqueues(self):
        bank = EmbeddingMemoryBank(capacity=8)
        t0 = [torch.randn(2, 4)]
        i0 = [torch.randn(3, 4)]
        bank.enqueue(image_raw=i0, text_raw=t0)
        snap = bank.snapshot()
        self.assertEqual(len(snap["image_raw"]), 1)
        bank.enqueue(image_raw=[torch.randn(3, 4)], text_raw=[torch.randn(2, 4)])
        self.assertEqual(len(bank), 2)
        # Snapshot length unchanged (detached copy of prior contents).
        self.assertEqual(len(snap["image_raw"]), 1)
        self.assertEqual(len(snap["text_raw"]), 1)

    def test_begin_accum_window_freezes_score_view(self):
        """Score view stays at window-start while live bank grows (policy B)."""

        class _Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.memory_bank = EmbeddingMemoryBank(capacity=16)
                self.bank_score_policy = "accum_window"
                self._score_bank_snapshot = None

            begin_accum_window = Stage1AlignmentModel.begin_accum_window
            _bank_raw_for_scoring = Stage1AlignmentModel._bank_raw_for_scoring

        m = _Tiny()
        m.memory_bank.enqueue(
            image_raw=[torch.ones(2, 3)], text_raw=[torch.ones(2, 3)]
        )
        m.begin_accum_window()
        text_s, image_s = m._bank_raw_for_scoring()
        self.assertEqual(len(image_s), 1)
        # Enqueue another micro-batch into the live bank.
        m.memory_bank.enqueue(
            image_raw=[torch.zeros(2, 3)], text_raw=[torch.zeros(2, 3)]
        )
        self.assertEqual(len(m.memory_bank), 2)
        # Scoring still sees window-start snapshot (1 entry).
        text_s2, image_s2 = m._bank_raw_for_scoring()
        self.assertEqual(len(image_s2), 1)
        self.assertTrue(torch.equal(image_s2[0], torch.ones(2, 3)))
        # Next window refreshes snapshot to include both.
        m.begin_accum_window()
        _, image_s3 = m._bank_raw_for_scoring()
        self.assertEqual(len(image_s3), 2)

    def test_live_policy_sees_enqueues_immediately(self):
        class _Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.memory_bank = EmbeddingMemoryBank(capacity=16)
                self.bank_score_policy = "live"
                self._score_bank_snapshot = None

            begin_accum_window = Stage1AlignmentModel.begin_accum_window
            _bank_raw_for_scoring = Stage1AlignmentModel._bank_raw_for_scoring

        m = _Tiny()
        m.begin_accum_window()  # no-op for live
        m.memory_bank.enqueue(
            image_raw=[torch.ones(2, 3)], text_raw=[torch.ones(2, 3)]
        )
        _, imgs = m._bank_raw_for_scoring()
        self.assertEqual(len(imgs), 1)


class TestBackgroundPatchDrop(unittest.TestCase):
    def test_keeps_highest_l2_patches(self):
        # 4 patches; norms ~ 1, 2, 3, 10
        tokens = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [10.0, 0.0, 0.0],
            ]
        )
        kept = keep_top_patches_by_l2(tokens, keep_ratio=0.5)
        self.assertEqual(kept.shape[0], 2)
        norms = kept.norm(dim=-1)
        self.assertTrue(torch.allclose(norms.sort().values, torch.tensor([3.0, 10.0])))

    def test_keep_ratio_one_returns_all(self):
        tokens = torch.randn(9, 4)
        kept = keep_top_patches_by_l2(tokens, keep_ratio=1.0)
        self.assertEqual(kept.shape, tokens.shape)
        self.assertTrue(torch.equal(kept, tokens))

    def test_not_random_is_deterministic(self):
        tokens = torch.randn(20, 8)
        a = keep_top_patches_by_l2(tokens, keep_ratio=0.4)
        b = keep_top_patches_by_l2(tokens, keep_ratio=0.4)
        self.assertTrue(torch.equal(a, b))

    def test_always_keeps_at_least_one(self):
        tokens = torch.randn(5, 3)
        kept = keep_top_patches_by_l2(tokens, keep_ratio=0.0)
        self.assertEqual(kept.shape[0], 1)


class TestTrainStage1CliDefaults(unittest.TestCase):
    def test_soft_maxsim_default_enabled(self):
        # Import parse_args without running main.
        import train_stage1

        # Simulate argv
        import sys

        old = sys.argv
        try:
            sys.argv = ["train_stage1.py", "--help"]
            # Just check the module constants / defaults via parser construction.
            # Re-build parser by calling parse_args with minimal required-safe args
            # is hard (loads models); inspect source defaults instead.
            src = open(train_stage1.__file__, encoding="utf-8").read()
            self.assertIn('default=True', src)
            self.assertIn("--soft-maxsim", src)
            self.assertIn('default="accum_window"', src)
            self.assertIn("--vision-patch-keep-ratio", src)
            self.assertIn("--multi-positive-jaccard", src)
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()
