"""Unit tests for Stage-2 conditioning transforms and demo index helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from trisearch_demo_index import (
    CACHE_VERSION,
    IndexedImage,
    ImageSearchIndex,
)
from trisearch_models.inference import (
    embedding_token_dropout,
    maybe_merge_embeddings_to_one,
    prepare_stage2_condition_tokens,
    shuffle_token_embeddings,
)


class TestShuffleAndDropout(unittest.TestCase):
    def test_shuffle_is_permutation(self):
        x = torch.arange(12, dtype=torch.float32).view(4, 3)
        y, perm = shuffle_token_embeddings(x)
        self.assertEqual(tuple(y.shape), (4, 3))
        self.assertEqual(len(perm), 4)
        self.assertEqual(sorted(perm.tolist()), [0, 1, 2, 3])
        self.assertTrue(torch.equal(y, x[perm]))

    def test_shuffle_batch(self):
        x = torch.randn(2, 5, 8)
        y, perms = shuffle_token_embeddings(x)
        self.assertEqual(tuple(y.shape), (2, 5, 8))
        self.assertEqual(tuple(perms.shape), (2, 5))

    def test_dropout_reduces(self):
        x = torch.randn(100, 4)
        y = embedding_token_dropout(x, drop_prob=0.4, training=True)
        self.assertEqual(y.shape[0], 60)
        z = embedding_token_dropout(x, drop_prob=0.4, training=False)
        self.assertTrue(torch.equal(z, x))

    def test_merge_collapses_to_one(self):
        torch.manual_seed(0)
        x = F.normalize(torch.randn(16, 8), dim=-1)
        # Force merge by high prob
        y = maybe_merge_embeddings_to_one(x, merge_prob=1.0, training=True)
        self.assertEqual(tuple(y.shape), (1, 8))
        # Norm ~1
        self.assertAlmostEqual(float(y.norm()), 1.0, places=4)

    def test_merge_batch(self):
        x = F.normalize(torch.randn(3, 10, 8), dim=-1)
        y = maybe_merge_embeddings_to_one(x, merge_prob=1.0, training=True)
        self.assertEqual(tuple(y.shape), (3, 1, 8))

    def test_prepare_pipeline_shapes(self):
        x = F.normalize(torch.randn(20, 8), dim=-1)
        y = prepare_stage2_condition_tokens(
            x,
            shuffle=True,
            drop_prob=0.5,
            merge_prob=0.0,
            max_tokens=20,
            training=True,
        )
        self.assertEqual(y.shape[0], 10)
        z = prepare_stage2_condition_tokens(
            x,
            shuffle=True,
            drop_prob=0.0,
            merge_prob=1.0,
            max_tokens=20,
            training=True,
        )
        self.assertEqual(tuple(z.shape), (1, 8))
        capped = prepare_stage2_condition_tokens(
            F.normalize(torch.randn(100, 8), dim=-1),
            shuffle=True,
            drop_prob=0.0,
            merge_prob=0.0,
            max_tokens=16,
            training=True,
        )
        self.assertEqual(capped.shape[0], 16)


class TestCaptionSearchAndCache(unittest.TestCase):
    def _index(self) -> ImageSearchIndex:
        entries = [
            IndexedImage(0, "a red barn in a green field", torch.randn(4, 8), "a"),
            IndexedImage(1, "blue ocean waves at sunset", torch.randn(4, 8), "b"),
            IndexedImage(2, "dense forest canopy above", torch.randn(4, 8), "c"),
        ]
        return ImageSearchIndex(entries)

    def test_plaintext_substring(self):
        idx = self._index()
        hits = idx.search_captions_plaintext("barn", top_k=2)
        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("barn", hits[0][1].caption)

    def test_plaintext_token_overlap(self):
        idx = self._index()
        hits = idx.search_captions_plaintext("ocean sunset", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("ocean", hits[0][1].caption)

    def test_cache_roundtrip(self):
        idx = self._index()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "idx.pt"
            idx.save(path, meta={"phase": 1, "dataset": "test"})
            loaded, meta = ImageSearchIndex.load(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(meta["phase"], 1)
            self.assertEqual(int(torch.load(path, weights_only=False)["version"]), CACHE_VERSION)
            self.assertEqual(loaded.entries[0].caption, idx.entries[0].caption)


if __name__ == "__main__":
    unittest.main()
