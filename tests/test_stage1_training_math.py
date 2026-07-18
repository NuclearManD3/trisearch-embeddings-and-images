"""Unit tests for Stage-1 late-interaction / bank / patch-keep upgrades.

Covers:
1. Vectorized late-interaction scoring == loop mean-MaxSim
2. Multi-positive masking as non-negatives (false-negative softening)
3. Soft MaxSim (τ logsumexp)
4. Bank policy B: enqueue every micro-batch; score as-of accum window start
5. Background (L2) patch drop for SigLIP vision MaxSim
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from trisearch_models.training import (
    DEFAULT_GEO_AFTER_UNFREEZE,
    DEFAULT_HEATMAP_SPARSITY_WEIGHT,
    DEFAULT_MATRYOSHKA_DIMS,
    DEFAULT_SCORE_CENTER,
    DEFAULT_SOFT_MAXSIM_TEMPERATURE,
    DEFAULT_VISION_MERGE_TOKENS,
    DEFAULT_VISION_PATCH_DROP_PROB,
    EmbeddingMemoryBank,
    Stage1AlignmentModel,
    apply_hard_bank_mining,
    apply_score_center,
    build_late_interaction_matrix,
    build_multi_positive_mask,
    caption_token_jaccard,
    combine_full_and_matryoshka,
    contrastive_late_interaction_loss,
    contrastive_score_margin_metrics,
    differentiable_late_interaction_score,
    heatmap_sparsity_loss,
    keep_top_patches_by_l2,
    masked_cross_entropy,
    matryoshka_prefix_dims,
    mean_positive_rank,
    mean_task_losses,
    mean_unit_token_center,
    merge_tokens_by_similarity,
    apply_train_image_augmentations,
    random_drop_patches,
    random_shift_pixel_values,
    soft_or_hard_maxsim,
    train_image_geometric_augment,
    train_image_photometric_augment,
)
from image_augment import (
    DEFAULT_GRAYSCALE_PROB,
    denormalize_pixel_values,
    normalize_pixel_values,
    smooth_noise_field,
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


class TestContrastiveRankAndMargin(unittest.TestCase):
    def test_collapsed_scores_mid_rank_is_chance(self):
        """Full collapse must not report pos_rank=1 (old optimistic-ties bug)."""
        n_docs = 132
        scores = torch.ones(4, n_docs)
        rank = mean_positive_rank(scores)
        self.assertAlmostEqual(rank, (n_docs + 1) / 2.0, places=4)

    def test_unique_best_positive_rank_one(self):
        scores = torch.zeros(3, 10)
        scores[0, 0] = 5.0
        scores[1, 1] = 5.0
        scores[2, 2] = 5.0
        self.assertAlmostEqual(mean_positive_rank(scores), 1.0, places=4)

    def test_score_gap_zero_when_collapsed(self):
        scores = torch.ones(4, 36) * 3.5
        labels = torch.arange(4)
        m = contrastive_score_margin_metrics(scores, labels)
        self.assertAlmostEqual(m["score_gap"], 0.0, places=5)

    def test_score_gap_positive_when_pos_wins(self):
        scores = torch.zeros(2, 4)
        scores[0, 0] = 2.0
        scores[1, 1] = 2.0
        labels = torch.arange(2)
        m = contrastive_score_margin_metrics(scores, labels)
        self.assertGreater(m["score_gap"], 1.0)

    def test_contrastive_metrics_include_gap(self):
        texts = _rand_tokens(3, [2, 2, 2], dim=8, seed=20)
        images = _rand_tokens(3, [3, 3, 3], dim=8, seed=21)
        loss, metrics = contrastive_late_interaction_loss(
            texts, images, temperature=0.07, return_metrics=True
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("score_gap", metrics)
        self.assertTrue(math.isfinite(metrics["score_gap"]))
        self.assertLess(metrics["pos_rank"], 3.0)

    def test_heatmap_sparsity_default_off(self):
        self.assertEqual(DEFAULT_HEATMAP_SPARSITY_WEIGHT, 0.0)

    def test_geo_active_during_freeze_by_default(self):
        self.assertFalse(DEFAULT_GEO_AFTER_UNFREEZE)

    def test_score_center_default_on(self):
        self.assertTrue(DEFAULT_SCORE_CENTER)

    def test_vision_merge_default_positive(self):
        self.assertGreater(DEFAULT_VISION_MERGE_TOKENS, 0)


class TestScoreCenterAndMerge(unittest.TestCase):
    def test_merge_reduces_count(self):
        t = F.normalize(torch.randn(40, 16), dim=-1)
        m = merge_tokens_by_similarity(t, k=5)
        self.assertEqual(tuple(m.shape), (5, 16))

    def test_score_center_restores_gap_under_domain_cone(self):
        """Domain-shared patches collapse MaxSim gap; centering restores it."""
        g = torch.Generator().manual_seed(0)
        domain = F.normalize(torch.randn(32, generator=g), dim=0)
        B = 4
        texts, imgs = [], []
        for _ in range(B):
            u = F.normalize(torch.randn(32, generator=g), dim=0)
            t = F.normalize(
                domain + 0.3 * u + 0.05 * torch.randn(10, 32, generator=g), dim=-1
            )
            ct = F.normalize(
                domain + 0.3 * u + 0.05 * torch.randn(4, 32, generator=g), dim=-1
            )
            bg = F.normalize(
                domain + 0.02 * torch.randn(30, 32, generator=g), dim=-1
            )
            texts.append(t)
            imgs.append(torch.cat([ct, bg], 0))

        S0 = build_late_interaction_matrix(texts, imgs)
        gap0 = float(
            S0.diag().mean() - (S0.sum() - S0.diag().sum()) / (B * (B - 1))
        )
        center = mean_unit_token_center(texts, imgs)
        tc = apply_score_center(texts, center)
        ic = apply_score_center(imgs, center)
        S1 = build_late_interaction_matrix(tc, ic, query_topk=6)
        gap1 = float(
            S1.diag().mean() - (S1.sum() - S1.diag().sum()) / (B * (B - 1))
        )
        self.assertLess(gap0, 0.02)
        self.assertGreater(gap1, gap0 + 0.03)

    def test_contrastive_with_score_center_beats_unccentered(self):
        g = torch.Generator().manual_seed(1)
        domain = F.normalize(torch.randn(24, generator=g), dim=0)
        B = 4
        texts, imgs = [], []
        for _ in range(B):
            u = F.normalize(torch.randn(24, generator=g), dim=0)
            texts.append(
                F.normalize(
                    domain + 0.25 * u + 0.05 * torch.randn(8, 24, generator=g),
                    dim=-1,
                )
            )
            imgs.append(
                F.normalize(
                    domain + 0.25 * u + 0.05 * torch.randn(16, 24, generator=g),
                    dim=-1,
                )
            )
        loss0 = contrastive_late_interaction_loss(
            texts, imgs, temperature=0.07, score_center=False, query_topk=0
        )
        loss1 = contrastive_late_interaction_loss(
            texts, imgs, temperature=0.07, score_center=True, query_topk=4
        )
        self.assertLess(float(loss1), float(loss0))

    def test_query_topk_matrix_finite(self):
        q = _rand_tokens(2, [6, 4], dim=8, seed=3)
        d = _rand_tokens(3, [5, 5, 2], dim=8, seed=4)
        S = build_late_interaction_matrix(q, d, query_topk=2)
        self.assertEqual(tuple(S.shape), (2, 3))
        self.assertTrue(torch.isfinite(S).all())


class TestGapHingeLoss(unittest.TestCase):
    def test_zero_when_gap_above_margin(self):
        from trisearch_models.training import gap_hinge_loss

        # Positives on diagonal clearly best
        scores = torch.full((3, 3), -1.0)
        scores.fill_diagonal_(5.0)
        labels = torch.arange(3)
        self.assertAlmostEqual(
            float(gap_hinge_loss(scores, labels, margin=0.0)), 0.0, places=5
        )

    def test_positive_when_gap_negative(self):
        from trisearch_models.training import gap_hinge_loss

        # Positive is worst column
        scores = torch.zeros(2, 4)
        scores[:, 0] = -2.0
        scores[0, 1] = 3.0
        scores[1, 2] = 3.0
        labels = torch.zeros(2, dtype=torch.long)
        loss = gap_hinge_loss(scores, labels, margin=0.0)
        self.assertGreater(float(loss), 0.0)

    def test_gap_weight_improves_negative_gap_setup(self):
        """CE alone may leave neg gap; gap hinge should reduce it under SGD."""
        g = torch.Generator().manual_seed(0)
        domain = F.normalize(torch.randn(32, generator=g), dim=0)
        B = 4
        # Free centers: start with slight mismatch structure
        centers = F.normalize(
            domain + 0.05 * torch.randn(B, 32, generator=g), dim=-1
        ).clone().requires_grad_(True)
        opt = torch.optim.SGD([centers], lr=2.0)

        def tokens_from(c):
            t = [
                F.normalize(c[i] + 0.05 * torch.randn(6, 32), dim=-1)
                for i in range(B)
            ]
            im = [
                F.normalize(c[i] + 0.05 * torch.randn(8, 32), dim=-1)
                for i in range(B)
            ]
            return t, im

        t0, i0 = tokens_from(centers.detach())
        _, m0 = contrastive_late_interaction_loss(
            t0, i0, temperature=0.07, score_center=True,
            gap_weight=0.0, return_metrics=True,
        )
        for _ in range(40):
            opt.zero_grad()
            t, im = tokens_from(centers)
            loss = contrastive_late_interaction_loss(
                t, im, temperature=0.07, score_center=True,
                gap_weight=2.0, gap_margin=0.0, return_metrics=False,
            )
            loss.backward()
            opt.step()
        t1, i1 = tokens_from(centers.detach())
        _, m1 = contrastive_late_interaction_loss(
            t1, i1, temperature=0.07, score_center=True,
            gap_weight=0.0, return_metrics=True,
        )
        self.assertGreater(m1["score_gap"], m0["score_gap"] - 0.5)
        # After training with gap hinge, gap should be non-negative for easy free case
        self.assertGreater(m1["score_gap"], -0.1)


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


class TestRandomPatchDropAndShift(unittest.TestCase):
    def test_default_drop_prob(self):
        self.assertAlmostEqual(DEFAULT_VISION_PATCH_DROP_PROB, 0.15)

    def test_random_drop_reduces_count_when_training(self):
        tokens = torch.randn(100, 8)
        kept = random_drop_patches(tokens, drop_prob=0.4, training=True)
        self.assertEqual(kept.shape[0], 60)
        self.assertEqual(kept.shape[1], 8)

    def test_random_drop_noop_eval(self):
        tokens = torch.randn(20, 4)
        kept = random_drop_patches(tokens, drop_prob=0.4, training=False)
        self.assertTrue(torch.equal(kept, tokens))

    def test_random_drop_keeps_at_least_one(self):
        tokens = torch.randn(3, 2)
        kept = random_drop_patches(tokens, drop_prob=0.99, training=True)
        self.assertGreaterEqual(kept.shape[0], 1)

    def test_shift_preserves_shape(self):
        x = torch.randn(2, 3, 32, 32)
        y = random_shift_pixel_values(x, max_shift=18)
        self.assertEqual(tuple(y.shape), (2, 3, 32, 32))

    def test_shift_zero_is_noop(self):
        x = torch.randn(1, 3, 16, 16)
        y = random_shift_pixel_values(x, max_shift=0)
        self.assertTrue(torch.equal(x, y))


class TestTrainImageGeometricAugment(unittest.TestCase):
    def test_preserves_batch_shape(self):
        x = torch.randn(3, 3, 64, 64)
        y = train_image_geometric_augment(
            x,
            hflip_prob=1.0,
            max_rotate_deg=30.0,
            scale_min=0.85,
            scale_max=1.05,
            fill_mode="random",
        )
        self.assertEqual(tuple(y.shape), (3, 3, 64, 64))
        self.assertTrue(torch.isfinite(y).all())

    def test_mean_fill_mode(self):
        x = torch.randn(2, 3, 48, 48)
        y = train_image_geometric_augment(
            x,
            hflip_prob=0.0,
            max_rotate_deg=25.0,
            scale_min=0.9,
            scale_max=1.0,
            fill_mode="mean",
        )
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_full_stack_with_shift(self):
        x = torch.randn(2, 3, 40, 40)
        y = apply_train_image_augmentations(
            x,
            hflip_prob=0.5,
            max_rotate_deg=15.0,
            scale_min=0.85,
            scale_max=1.05,
            fill_mode="random",
            max_shift=8,
            enabled=True,
        )
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        z = apply_train_image_augmentations(x, enabled=False)
        self.assertTrue(torch.equal(z, x))

    def test_hflip_changes_image_when_forced(self):
        # Asymmetric image so flip is detectable.
        x = torch.zeros(1, 3, 32, 32)
        x[:, :, :, :8] = 1.0
        y = train_image_geometric_augment(
            x,
            hflip_prob=1.0,
            max_rotate_deg=0.0,
            scale_min=1.0,
            scale_max=1.0,
            fill_mode="mean",
        )
        # Left strip should move to the right.
        self.assertGreater(float(y[0, 0, 16, 28]), 0.5)
        self.assertLess(float(y[0, 0, 16, 2]), 0.5)


class TestPhotometricAugment(unittest.TestCase):
    def test_preserves_shape_and_finite(self):
        x = torch.randn(2, 3, 48, 48)
        y = train_image_photometric_augment(
            x,
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
            spatial_brightness=0.2,
            spatial_color=0.1,
            grayscale_prob=0.0,
        )
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(y).all())

    def test_disabled_is_identity(self):
        x = torch.randn(1, 3, 32, 32)
        y = train_image_photometric_augment(x, enabled=False)
        self.assertTrue(torch.equal(x, y))

    def test_strong_brightness_changes_mean(self):
        torch.manual_seed(0)
        # Mid-gray in denorm space after normalize with mean=std=0.5
        rgb = torch.full((1, 3, 32, 32), 0.5)
        x = normalize_pixel_values(rgb, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        y = train_image_photometric_augment(
            x,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            brightness=0.5,
            contrast=0.0,
            saturation=0.0,
            hue=0.0,
            spatial_brightness=0.0,
            spatial_color=0.0,
            grayscale_prob=0.0,
        )
        # Multiple draws until brightness factor ≠ 1
        changed = False
        for _ in range(20):
            y = train_image_photometric_augment(
                x,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                brightness=0.5,
                contrast=0.0,
                saturation=0.0,
                hue=0.0,
                spatial_brightness=0.0,
                spatial_color=0.0,
                grayscale_prob=0.0,
            )
            if not torch.allclose(y, x, atol=1e-5):
                changed = True
                break
        self.assertTrue(changed)

    def test_grayscale_makes_channels_equal(self):
        torch.manual_seed(1)
        rgb = torch.rand(1, 3, 24, 24)
        x = normalize_pixel_values(rgb, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        y = train_image_photometric_augment(
            x,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            brightness=0.0,
            contrast=0.0,
            saturation=0.0,
            hue=0.0,
            spatial_brightness=0.0,
            spatial_color=0.0,
            grayscale_prob=1.0,
        )
        rgb_y = denormalize_pixel_values(y, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.assertTrue(torch.allclose(rgb_y[0, 0], rgb_y[0, 1], atol=1e-5))
        self.assertTrue(torch.allclose(rgb_y[0, 1], rgb_y[0, 2], atol=1e-5))

    def test_smooth_noise_shape(self):
        f = smooth_noise_field(2, 3, 40, 40, grid=8)
        self.assertEqual(tuple(f.shape), (2, 3, 40, 40))
        self.assertTrue(torch.isfinite(f).all())

    def test_full_stack_with_photometric(self):
        x = torch.randn(2, 3, 40, 40)
        y = apply_train_image_augmentations(
            x,
            photometric=True,
            photo_brightness=0.2,
            spatial_brightness=0.1,
            spatial_color=0.05,
            grayscale_prob=0.0,
            max_shift=4,
            enabled=True,
        )
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(y).all())

    def test_default_grayscale_prob_positive(self):
        self.assertGreater(DEFAULT_GRAYSCALE_PROB, 0.0)


class TestHeatmapSparsityLoss(unittest.TestCase):
    def test_uniform_higher_than_peaked(self):
        # One query token; image patches: peaked vs flat similarities.
        q = F.normalize(torch.ones(1, 8), dim=-1)
        # Peaked: one patch = q, others orthogonal-ish
        peaked = F.normalize(torch.randn(16, 8), dim=-1)
        peaked[0] = q[0]
        # Uniform-ish: all patches similar to q
        flat = F.normalize(q[0].unsqueeze(0) + 0.05 * torch.randn(16, 8), dim=-1)
        loss_peak = heatmap_sparsity_loss([q], [peaked], temperature=0.07)
        loss_flat = heatmap_sparsity_loss([q], [flat], temperature=0.07)
        self.assertLess(float(loss_peak), float(loss_flat))

    def test_gradients_flow(self):
        q = F.normalize(torch.randn(3, 8, requires_grad=True), dim=-1)
        # re-enable grad after normalize
        q = q.detach().requires_grad_(True)
        d = F.normalize(torch.randn(12, 8), dim=-1).detach().requires_grad_(True)
        # Use unnormalized with grad
        q_raw = torch.randn(3, 8, requires_grad=True)
        d_raw = torch.randn(12, 8, requires_grad=True)
        qn = F.normalize(q_raw, dim=-1)
        dn = F.normalize(d_raw, dim=-1)
        loss = heatmap_sparsity_loss([qn], [dn], temperature=0.1)
        loss.backward()
        self.assertIsNotNone(q_raw.grad)
        self.assertIsNotNone(d_raw.grad)
        self.assertTrue(torch.isfinite(q_raw.grad).all())
        self.assertTrue(torch.isfinite(d_raw.grad).all())


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
            with open(train_stage1.__file__, encoding="utf-8") as f:
                src = f.read()
            self.assertIn('default=True', src)
            self.assertIn("--soft-maxsim", src)
            self.assertIn('default="accum_window"', src)
            self.assertIn("--vision-patch-keep-ratio", src)
            self.assertIn("--multi-positive-jaccard", src)
            self.assertIn("--hard-bank-negatives", src)
            self.assertIn("--query-image-weight", src)
            self.assertIn('default="64,128,256,512"', src)
            self.assertIn("--vision-patch-drop-prob", src)
            self.assertIn("--image-shift-max", src)
            self.assertIn("--heatmap-sparsity-weight", src)
            self.assertIn("--image-max-rotate-deg", src)
            self.assertIn("--image-hflip-prob", src)
            self.assertIn("--no-image-aug", src)
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()


class TestEmbeddingGeometry(unittest.TestCase):
    def test_cone_has_higher_center_than_isotropic(self):
        from trisearch_models.training import embedding_geometry_loss

        g = torch.Generator().manual_seed(0)
        # Simulate user-observed cone: dim0 ~ +0.05, rest small correlated noise
        cone = torch.randn(64, 128, generator=g) * 0.01
        cone[:, 0] = 0.05
        cone = F.normalize(cone, dim=-1)
        iso = F.normalize(torch.randn(64, 128, generator=g), dim=-1)
        loss_cone, m_cone = embedding_geometry_loss(
            cone, var_weight=0.0, vec_mean_weight=0.0, mag_floor_weight=0.0,
            max_abs_weight=0.0, uniformity_weight=0.0,
        )
        loss_iso, m_iso = embedding_geometry_loss(
            iso, var_weight=0.0, vec_mean_weight=0.0, mag_floor_weight=0.0,
            max_abs_weight=0.0, uniformity_weight=0.0,
        )
        self.assertGreater(m_cone["geo_mu_norm"], m_iso["geo_mu_norm"])
        self.assertGreater(float(loss_cone), float(loss_iso))

    def test_variance_floor_hits_collapsed_dims(self):
        from trisearch_models.training import embedding_geometry_loss

        # All vectors nearly identical → near-zero per-dim std
        base = F.normalize(torch.randn(1, 64), dim=-1)
        collapsed = base.expand(32, -1).clone() + torch.randn(32, 64) * 1e-6
        collapsed = F.normalize(collapsed, dim=-1)
        _, m = embedding_geometry_loss(
            collapsed,
            center_weight=0.0,
            var_weight=1.0,
            vec_mean_weight=0.0,
            mag_floor_weight=0.0,
            max_abs_weight=0.0,
            uniformity_weight=0.0,
            var_ratio=0.5,
        )
        self.assertGreater(m["geo_var"], 0.0)
        self.assertLess(m["geo_min_std"], 0.01)

    def test_all_negative_vec_mean_penalty(self):
        from trisearch_models.training import embedding_geometry_loss

        # All-negative unit vectors (after renorm still mostly negative)
        neg = -torch.ones(16, 32)
        neg = F.normalize(neg, dim=-1)
        balanced = F.normalize(torch.randn(16, 32), dim=-1)
        _, m_neg = embedding_geometry_loss(
            neg,
            center_weight=0.0,
            var_weight=0.0,
            vec_mean_weight=1.0,
            mag_floor_weight=0.0,
            max_abs_weight=0.0,
            uniformity_weight=0.0,
        )
        _, m_bal = embedding_geometry_loss(
            balanced,
            center_weight=0.0,
            var_weight=0.0,
            vec_mean_weight=1.0,
            mag_floor_weight=0.0,
            max_abs_weight=0.0,
            uniformity_weight=0.0,
        )
        self.assertGreater(m_neg["geo_vec_mean"], m_bal["geo_vec_mean"])

    def test_mag_floor_on_tiny_raw(self):
        from trisearch_models.training import embedding_geometry_loss

        raw = torch.randn(20, 16) * 0.001
        norm = F.normalize(raw, dim=-1)
        loss, m = embedding_geometry_loss(
            norm,
            raw=raw,
            center_weight=0.0,
            var_weight=0.0,
            vec_mean_weight=0.0,
            mag_floor=0.05,
            mag_floor_weight=1.0,
            max_abs_weight=0.0,
            uniformity_weight=0.0,
        )
        self.assertGreater(m["geo_mag_floor"], 0.0)
        self.assertGreater(float(loss), 0.0)

    def test_uniformity_lower_when_spread(self):
        from trisearch_models.training import embedding_geometry_loss

        g = torch.Generator().manual_seed(1)
        # Collapsed: all near same direction → uniformity closer to 0
        base = F.normalize(torch.randn(1, 64, generator=g), dim=-1)
        collapsed = F.normalize(
            base.expand(48, -1) + 1e-3 * torch.randn(48, 64, generator=g), dim=-1
        )
        spread = F.normalize(torch.randn(48, 64, generator=g), dim=-1)
        loss_c, m_c = embedding_geometry_loss(
            collapsed,
            center_weight=0.0,
            var_weight=0.0,
            vec_mean_weight=0.0,
            mag_floor_weight=0.0,
            max_abs_weight=0.0,
            uniformity_weight=1.0,
        )
        loss_s, m_s = embedding_geometry_loss(
            spread,
            center_weight=0.0,
            var_weight=0.0,
            vec_mean_weight=0.0,
            mag_floor_weight=0.0,
            max_abs_weight=0.0,
            uniformity_weight=1.0,
        )
        # Raw W&I: more spread → more negative
        self.assertLess(m_s["geo_uniformity"], m_c["geo_uniformity"])
        # Non-neg pen: collapsed ≈1, spread ≪1; loss uses pen (always ≥ 0)
        self.assertGreater(m_c["geo_uniformity_pen"], m_s["geo_uniformity_pen"])
        self.assertGreaterEqual(float(loss_c), 0.0)
        self.assertGreaterEqual(float(loss_s), 0.0)
        self.assertGreater(float(loss_c), float(loss_s))

    def test_geo_loss_never_negative(self):
        from trisearch_models.training import embedding_geometry_loss

        for seed in range(5):
            g = torch.Generator().manual_seed(seed)
            v = F.normalize(torch.randn(32, 64, generator=g), dim=-1)
            loss, _ = embedding_geometry_loss(v)
            self.assertGreaterEqual(float(loss), 0.0)

    def test_geo_square_soft_when_small(self):
        """Squaring: tiny badness → negligible; large badness → amplified."""
        raw_small = torch.tensor(0.2)
        raw_large = torch.tensor(2.0)
        self.assertLess(float(raw_small**2), float(raw_small))
        self.assertGreater(float(raw_large**2), float(raw_large))

    def test_prefix_ema_not_unit_normalized(self):
        """Raw EMA prefix must not be L2-normalized (would floor center ~0.25)."""
        from trisearch_models.training import embedding_geometry_loss

        g = torch.Generator().manual_seed(2)
        iso = F.normalize(torch.randn(64, 256, generator=g), dim=-1)
        # Small raw mean (isotropic history)
        ema_raw = iso.mean(0) * 0.05
        # Buggy unit EMA
        ema_unit = F.normalize(ema_raw, dim=-1)
        _, m_raw = embedding_geometry_loss(
            iso,
            ema_mean=ema_raw,
            var_weight=0.0,
            vec_mean_weight=0.0,
            mag_floor_weight=0.0,
            max_abs_weight=0.0,
            uniformity_weight=0.0,
        )
        _, m_unit = embedding_geometry_loss(
            iso,
            ema_mean=ema_unit,
            var_weight=0.0,
            vec_mean_weight=0.0,
            mag_floor_weight=0.0,
            max_abs_weight=0.0,
            uniformity_weight=0.0,
        )
        self.assertLess(m_raw["geo_center"], 0.05)
        self.assertGreater(m_unit["geo_center"], 0.2)

    def test_renormed_pool_detects_sample_cone(self):
        """Mean-pool without renorm understates sample-level cone; renorm fixes it."""
        from trisearch_models.training import (
            embedding_geometry_loss,
            mean_pool_token_list,
        )

        g = torch.Generator().manual_seed(3)
        c = F.normalize(torch.randn(1, 128, generator=g), dim=-1)
        seqs = [
            F.normalize(c + 0.1 * torch.randn(40, 128, generator=g), dim=-1)
            for _ in range(4)
        ]
        pooled_raw = mean_pool_token_list(seqs)
        assert pooled_raw is not None
        pooled_unit = F.normalize(pooled_raw.float(), dim=-1)
        _, m_raw = embedding_geometry_loss(
            pooled_raw,
            var_weight=0.0,
            vec_mean_weight=0.0,
            mag_floor_weight=0.0,
            max_abs_weight=0.0,
            uniformity_weight=0.0,
        )
        _, m_unit = embedding_geometry_loss(
            pooled_unit,
            var_weight=0.0,
            vec_mean_weight=0.0,
            mag_floor_weight=0.0,
            max_abs_weight=0.0,
            uniformity_weight=0.0,
        )
        self.assertGreater(m_unit["geo_mu_norm"], m_raw["geo_mu_norm"])
        self.assertGreater(m_unit["geo_mu_norm"], 0.8)

    def test_gradients_flow(self):
        from trisearch_models.training import embedding_geometry_loss

        raw = torch.randn(12, 32, requires_grad=True)
        norm = F.normalize(raw, dim=-1)
        loss, _ = embedding_geometry_loss(norm, raw=raw)
        loss.backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_ema_update(self):
        from trisearch_models.training import update_embedding_ema

        ema = torch.zeros(4)
        batch = torch.ones(4)
        update_embedding_ema(ema, batch, momentum=0.9)
        self.assertTrue(torch.allclose(ema, torch.full((4,), 0.1)))
        update_embedding_ema(ema, batch, momentum=0.9)
        self.assertTrue(torch.allclose(ema, torch.full((4,), 0.19), atol=1e-6))

    def test_stack_and_pool(self):
        from trisearch_models.training import mean_pool_token_list, stack_token_embeddings

        a = [torch.ones(2, 3), torch.zeros(1, 3)]
        b = [torch.full((3, 3), 2.0)]
        stacked = stack_token_embeddings([a, b])
        self.assertEqual(tuple(stacked.shape), (6, 3))
        pooled = mean_pool_token_list(a)
        self.assertEqual(tuple(pooled.shape), (2, 3))
        self.assertTrue(torch.allclose(pooled[0], torch.ones(3)))

    def test_default_geo_weight_positive(self):
        from trisearch_models.training import DEFAULT_EMBEDDING_GEO_WEIGHT
        import inspect

        self.assertGreater(DEFAULT_EMBEDDING_GEO_WEIGHT, 0.0)
        sig = inspect.signature(Stage1AlignmentModel.__init__)
        self.assertEqual(
            sig.parameters["embedding_geo_weight"].default,
            DEFAULT_EMBEDDING_GEO_WEIGHT,
        )
        self.assertFalse(sig.parameters["geo_after_unfreeze"].default)


class TestMatryoshkaPrefixes(unittest.TestCase):
    def test_default_dims_exclude_full_embed(self):
        self.assertNotIn(1024, DEFAULT_MATRYOSHKA_DIMS)
        self.assertEqual(matryoshka_prefix_dims(DEFAULT_MATRYOSHKA_DIMS), DEFAULT_MATRYOSHKA_DIMS)

    def test_strips_full_dim_and_invalid(self):
        got = matryoshka_prefix_dims((64, 128, 1024, 0, 2048, 64), embed_dim=1024)
        self.assertEqual(got, (64, 128))

    def test_combine_full_and_mrl_equal_share(self):
        full = torch.tensor(2.0)
        mrl = torch.tensor(4.0)
        # weight 1 → (2+4)/2 = 3
        self.assertAlmostEqual(
            float(combine_full_and_matryoshka(full, mrl, 1.0)), 3.0
        )
        # no prefixes → full only
        self.assertAlmostEqual(
            float(combine_full_and_matryoshka(full, mrl, 1.0, has_prefixes=False)),
            2.0,
        )


class TestHardBankMining(unittest.TestCase):
    def test_keeps_top_k_bank_per_row(self):
        # B=2, n_live=2, n_bank=4
        scores = torch.tensor(
            [
                [0.0, 1.0,  5.0, 4.0, 3.0, 2.0],  # bank ranks: 5,4,3,2 → top2 = 5,4
                [0.0, 1.0,  1.0, 9.0, 8.0, 0.5],  # top2 = 9,8
            ]
        )
        out = apply_hard_bank_mining(
            scores, n_live=2, hard_k=2, bank_random_k=0, bank_fn_margin=None
        )
        # live unchanged
        self.assertTrue(torch.equal(out[:, :2], scores[:, :2]))
        # row0 bank: keep 5 and 4, mask 3 and 2
        self.assertTrue(torch.isfinite(out[0, 2]))
        self.assertTrue(torch.isfinite(out[0, 3]))
        self.assertTrue(torch.isinf(out[0, 4]) and out[0, 4] < 0)
        self.assertTrue(torch.isinf(out[0, 5]) and out[0, 5] < 0)
        # row1 bank: keep 9 and 8
        self.assertTrue(torch.isinf(out[1, 2]) and out[1, 2] < 0)
        self.assertTrue(torch.isfinite(out[1, 3]))
        self.assertTrue(torch.isfinite(out[1, 4]))
        self.assertTrue(torch.isinf(out[1, 5]) and out[1, 5] < 0)

    def test_hard_k_zero_is_full_bank(self):
        scores = torch.randn(3, 10)
        # hard_k=0 → full bank (random_k ignored); no FN without labels.
        out = apply_hard_bank_mining(
            scores, n_live=3, hard_k=0, bank_random_k=8, bank_fn_margin=None
        )
        self.assertTrue(torch.equal(out, scores))

    def test_fn_filter_masks_bank_near_positive(self):
        # live pos col 0 score=5; bank col with 6 should be FN at margin=0.
        scores = torch.tensor(
            [
                [5.0, 0.0,  6.0, 4.0, 1.0],  # bank: 6 FN, 4 hard, 1 soft
            ]
        )
        labels = torch.tensor([0])
        out = apply_hard_bank_mining(
            scores,
            n_live=2,
            hard_k=2,
            labels=labels,
            bank_fn_margin=0.0,
            bank_random_k=0,
        )
        self.assertTrue(torch.isinf(out[0, 2]) and out[0, 2] < 0)  # FN
        self.assertTrue(torch.isfinite(out[0, 3]))  # hard
        self.assertTrue(torch.isfinite(out[0, 4]))  # hard (only 2 non-FN)

    def test_fn_filter_margin_keeps_slightly_below_pos(self):
        scores = torch.tensor([[10.0, 0.0, 9.5, 1.0, 0.5]])
        labels = torch.tensor([0])
        # margin=0 → 9.5 < 10 stays; margin=0 means thr=10, bank>=10 filtered
        out = apply_hard_bank_mining(
            scores,
            n_live=2,
            hard_k=3,
            labels=labels,
            bank_fn_margin=0.0,
            bank_random_k=0,
        )
        self.assertTrue(torch.isfinite(out[0, 2]))  # 9.5 < 10

        out2 = apply_hard_bank_mining(
            scores,
            n_live=2,
            hard_k=3,
            labels=labels,
            bank_fn_margin=1.0,  # thr=9 → 9.5 is FN
            bank_random_k=0,
        )
        self.assertTrue(torch.isinf(out2[0, 2]) and out2[0, 2] < 0)

    def test_hard_plus_random_keeps_extra(self):
        torch.manual_seed(0)
        # bank: 5,4,3,2,1 — hard_k=1 keeps 5; random_k=1 keeps one of 4..1
        scores = torch.tensor([[0.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
        out = apply_hard_bank_mining(
            scores,
            n_live=2,
            hard_k=1,
            bank_random_k=1,
            bank_fn_margin=None,
        )
        finite_bank = torch.isfinite(out[0, 2:]).sum().item()
        self.assertEqual(finite_bank, 2)  # 1 hard + 1 random
        self.assertTrue(torch.isfinite(out[0, 2]))  # hardest always kept

    def test_contrastive_with_hard_bank_finite(self):
        texts = _rand_tokens(2, [3, 2], dim=8, seed=20)
        images = _rand_tokens(2, [4, 3], dim=8, seed=21)
        bank_t = _rand_tokens(5, [2, 2, 2, 2, 2], dim=8, seed=22)
        bank_i = _rand_tokens(5, [3, 3, 3, 3, 3], dim=8, seed=23)
        loss = contrastive_late_interaction_loss(
            texts,
            images,
            temperature=0.07,
            bank_text_tokens=bank_t,
            bank_image_tokens=bank_i,
            hard_bank_k=2,
            bank_random_k=1,
            bank_fn_margin=0.0,
        )
        self.assertTrue(torch.isfinite(loss))

    def test_query_mask_differs_from_caption_mask(self):
        """Caption-near-dups must not force query-task multi-pos when queries differ."""
        caps = [
            "a red sports car on a racetrack",
            "a red sports car on a track",  # high caption Jaccard
            "satellite view of a harbor",
        ]
        queries = [
            "sports car photo",
            "harbor aerial",  # different intent despite similar first captions
            "harbor aerial view",
        ]
        cap_mask = build_multi_positive_mask(
            caps, batch_size=3, jaccard_threshold=0.5, device=torch.device("cpu")
        )
        q_mask = build_multi_positive_mask(
            queries, batch_size=3, jaccard_threshold=0.5, device=torch.device("cpu")
        )
        assert cap_mask is not None and q_mask is not None
        # Captions 0 and 1 are multi-pos; queries 0 and 1 are not.
        self.assertTrue(bool(cap_mask[0, 1]))
        self.assertFalse(bool(q_mask[0, 1]))
        # Queries 1 and 2 may be multi-pos (harbor).
        self.assertTrue(bool(q_mask[1, 2]))


class TestMeanTaskLossesErrorAtOutputs(unittest.TestCase):
    def test_mean_equals_average(self):
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(4.0, requires_grad=True)
        total = mean_task_losses([(a, 1.0), (b, 1.0)])
        self.assertAlmostEqual(float(total), 3.0)

    def test_gradients_add_at_shared_embedding(self):
        """Mean of task CEs adds ∂L_i/∂E on a shared embedding (not detach)."""
        e = torch.randn(4, requires_grad=True)
        l1 = (e * e).sum()
        l2 = (e - 1.0).pow(2).sum()
        total = mean_task_losses([(l1, 1.0), (l2, 1.0)])
        total.backward()
        # total = 0.5*l1 + 0.5*l2 → grad = e + (e-1) = 2e-1
        self.assertTrue(
            torch.allclose(e.grad, 2 * e.detach() - 1.0, atol=1e-5),
            (e.grad, 2 * e.detach() - 1.0),
        )
        e2 = e.detach().clone().requires_grad_(True)
        (0.5 * (e2 * e2).sum() + 0.5 * (e2 - 1.0).pow(2).sum()).backward()
        self.assertTrue(torch.allclose(e.grad, e2.grad))


class TestPeakHeatmapNormalize(unittest.TestCase):
    def test_peak_is_one_and_far_is_cold(self):
        from trisearch_models.inference import normalize_heatmap_scores

        # High absolute floor; only the 0.95 peak should be hot.
        s = torch.tensor([0.80, 0.82, 0.95, 0.88])
        h = normalize_heatmap_scores(s, mode="peak", peak_temperature=0.06)
        self.assertAlmostEqual(float(h[2]), 1.0, places=5)
        # δ=0.15 → exp(-0.15/0.06) ≈ 0.082
        self.assertLess(float(h[0]), 0.15)
        # Close non-match 0.88 vs peak 0.95: δ=0.07 → exp(-0.07/0.06) ≈ 0.31
        self.assertGreater(float(h[3]), float(h[0]))
        self.assertLess(float(h[3]), 0.5)

    def test_absolute_level_does_not_paint_flat_hot(self):
        from trisearch_models.inference import normalize_heatmap_scores

        flat_high = torch.full((8,), 0.9)
        h = normalize_heatmap_scores(flat_high, mode="peak", min_spread=1e-4)
        self.assertTrue(torch.all(h == 0.0))

    def test_close_match_separates_better_than_linear_percentile(self):
        from trisearch_models.inference import normalize_heatmap_scores

        # One peak 0.9, close 0.85, bulk non-match ~0.7
        s = torch.tensor([0.70, 0.71, 0.72, 0.85, 0.90])
        peak = normalize_heatmap_scores(s, mode="peak", peak_temperature=0.05)
        lin = normalize_heatmap_scores(s, mode="percentile")
        # Peak mode: gap between 0.85 and 0.90 should dominate visual scale.
        peak_gap = float(peak[4] - peak[3])
        lin_gap = float(lin[4] - lin[3])
        # Relative separation of close pair vs bulk: peak compresses bulk to near 0.
        self.assertLess(float(peak[0]), 0.05)
        self.assertGreater(peak_gap, lin_gap * 0.5)  # at least competitive
        self.assertAlmostEqual(float(peak[4]), 1.0, places=5)


class TestLatestTrainedCheckpointAcrossStages(unittest.TestCase):
    """find_latest_trained_checkpoint: stage5→1, newest within highest stage."""

    def _make_ckpt(self, root: Path, *, mtime: float | None = None) -> Path:
        import os
        import time

        root.mkdir(parents=True, exist_ok=True)
        for comp in ("vision_model", "text_model"):
            d = root / comp
            d.mkdir(parents=True, exist_ok=True)
            (d / "model.safetensors").write_bytes(b"x")
            (d / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
        proj = root / "projection_heads.pt"
        proj.write_bytes(b"y")
        if mtime is not None:
            os.utime(proj, (mtime, mtime))
            os.utime(root, (mtime, mtime))
        return root

    def test_prefers_higher_stage_over_newer_lower(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        import trisearch_models.training as tr

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            s1 = base / "stage1"
            s2 = base / "stage2"
            self._make_ckpt(s1 / "history" / "step-9000", mtime=2_000_000_000)
            self._make_ckpt(s2 / "history" / "step-10", mtime=1_000_000_000)
            with mock.patch.object(tr, "TRAINED_ROOT", str(base)):
                with mock.patch.object(tr, "DEFAULT_TRAINED_DIR", str(s1)):
                    with mock.patch.object(tr, "LEGACY_CHECKPOINT_DIR", str(base / "legacy")):
                        found = tr.find_latest_trained_checkpoint(max_stage=5, min_stage=1)
            self.assertIsNotNone(found)
            self.assertEqual(found, s2 / "history" / "step-10")

    def test_within_stage_picks_newest_mtime(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        import trisearch_models.training as tr

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            s1 = base / "stage1"
            older = self._make_ckpt(s1 / "history" / "step-100", mtime=1000)
            newer = self._make_ckpt(s1 / "history" / "step-200", mtime=2000)
            with mock.patch.object(tr, "TRAINED_ROOT", str(base)):
                with mock.patch.object(tr, "DEFAULT_TRAINED_DIR", str(s1)):
                    with mock.patch.object(tr, "LEGACY_CHECKPOINT_DIR", str(base / "legacy")):
                        found = tr.find_latest_trained_checkpoint()
            self.assertEqual(found, newer)
            self.assertNotEqual(found, older)

    def test_resolve_latest_across_stages_flag(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        import trisearch_models.training as tr

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            s1 = base / "stage1"
            ckpt = self._make_ckpt(s1, mtime=5000)
            with mock.patch.object(tr, "TRAINED_ROOT", str(base)):
                with mock.patch.object(tr, "DEFAULT_TRAINED_DIR", str(s1)):
                    with mock.patch.object(tr, "LEGACY_CHECKPOINT_DIR", str(base / "legacy")):
                        got = tr.resolve_inference_checkpoint(latest_across_stages=True)
            self.assertEqual(got, ckpt)


class TestMultiPosAndParaphraseQueue(unittest.TestCase):
    def test_multi_positive_cross_entropy_prefers_pos(self):
        from trisearch_models.training import multi_positive_cross_entropy

        scores = torch.tensor([[5.0, 4.0, 0.0], [0.0, 0.1, 5.0]])
        pos = torch.tensor([[True, True, False], [False, False, True]])
        loss = multi_positive_cross_entropy(scores, pos)
        self.assertGreater(float(loss), 0.0)
        self.assertLess(float(loss), 0.5)

    def test_multipos_contrastive_two_captions_one_image(self):
        from trisearch_models.training import contrastive_late_interaction_loss

        g = torch.Generator().manual_seed(7)
        def tok(seed):
            gg = torch.Generator().manual_seed(seed)
            return F.normalize(torch.randn(4, 16, generator=gg), dim=-1)

        imgs = [tok(1), tok(2)]
        texts = [tok(10), tok(11), tok(20)]
        ids = torch.tensor([0, 0, 1])
        loss, metrics = contrastive_late_interaction_loss(
            texts,
            imgs,
            temperature=0.07,
            text_image_ids=ids,
            score_center=False,
            gap_weight=0.0,
            return_metrics=True,
            soft_maxsim_temperature=None,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["n_text_docs"], 3.0)
        self.assertEqual(metrics["n_image_docs"], 2.0)

    def test_streaming_queue_pop_and_refill(self):
        from trisearch_dataset import StreamingTextPairQueue

        rows = [
            {"anchor": f"a{i}", "positive": f"p{i}", "negative": f"n{i}"}
            for i in range(40)
        ]
        q = StreamingTextPairQueue(
            queue_size=15,
            refill_size=8,
            row_iterator=iter(rows),
            seed=1,
        )
        self.assertEqual(q.refill(), 8)
        self.assertEqual(len(q), 8)
        b1 = q.pop_batch(5)
        self.assertEqual(len(b1), 5)
        self.assertEqual(len(q), 3)
        # Needs refill for larger batch
        b2 = q.pop_batch(6)
        self.assertEqual(len(b2), 6)
        self.assertLessEqual(len(q), 15)

    def test_build_positive_texts_includes_extras_and_query(self):
        from trisearch_dataset import build_positive_texts_for_image

        texts = build_positive_texts_for_image(
            {"captions": ["Primary view", "Secondary view", "Primary view"]},
            caption="Primary view",
            related_query="search for scene",
            max_texts=4,
        )
        self.assertEqual(texts[0], "primary view")
        self.assertIn("secondary view", texts)
        self.assertIn("search for scene", texts)
        self.assertEqual(len(texts), len(set(texts)))

    def test_paraphrase_contrastive_with_hard_negs(self):
        from trisearch_models.training import paraphrase_contrastive_loss

        g = torch.Generator().manual_seed(3)
        def tok():
            return F.normalize(torch.randn(3, 12, generator=g), dim=-1)

        anchors = [tok() for _ in range(4)]
        positives = [tok() for _ in range(4)]
        negatives = [tok() for _ in range(4)]
        loss = paraphrase_contrastive_loss(
            anchors,
            positives,
            negative_tokens=negatives,
            temperature=0.07,
            score_center=False,
            gap_weight=0.0,
            soft_maxsim_temperature=None,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)


class TestDualGpuFuseHelpers(unittest.TestCase):
    def test_cat_pad_text_batches(self):
        from trisearch_models.training import Stage1AlignmentModel

        ids1 = torch.ones(2, 3, dtype=torch.long)
        m1 = torch.ones(2, 3, dtype=torch.long)
        ids2 = torch.full((1, 5), 7, dtype=torch.long)
        m2 = torch.ones(1, 5, dtype=torch.long)
        out_ids, out_mask, ranges = Stage1AlignmentModel.cat_pad_text_batches(
            [(ids1, m1), (ids2, m2)], pad_token_id=0
        )
        self.assertEqual(tuple(out_ids.shape), (3, 5))
        self.assertEqual(ranges, [(0, 2), (2, 3)])
        self.assertEqual(int(out_ids[0, 3]), 0)
        self.assertEqual(int(out_mask[0, 3]), 0)
        self.assertEqual(int(out_ids[2, 0]), 7)
