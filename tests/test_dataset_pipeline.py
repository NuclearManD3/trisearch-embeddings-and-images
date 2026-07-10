#!/usr/bin/env python3
"""Unit tests for curated dataset format, generation helpers, and training load."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trisearch_data_format import (  # noqa: E402
    DEFAULT_IMAGE_SIZE,
    DOMAIN_GENERAL,
    DOMAIN_SATELLITE,
    caption_set_is_diverse,
    captions_are_near_duplicate,
    load_dataset_records,
    normalize_captions,
    open_lazy_dataset,
    resize_square_rgb,
    save_dataset,
    validate_record,
)
from trisearch_dataset import (  # noqa: E402
    QUERY_CACHE_RELATED_KEY,
    QUERY_CACHE_UNRELATED_KEY,
    ImageCaptionDataset,
    curated_dataset_available,
    enrich_rows_with_text_queries,
    load_curated_training_rows,
    load_stage1_training_rows,
)


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (640, 480)) -> Image.Image:
    return Image.new("RGB", size, color)


class TestResizeAndCaptions(unittest.TestCase):
    def test_resize_square(self):
        img = resize_square_rgb(_solid((10, 20, 30), (800, 400)), DEFAULT_IMAGE_SIZE)
        self.assertEqual(img.size, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
        self.assertEqual(img.mode, "RGB")

    def test_resize_scale_short_side_then_crop(self):
        """Landscape: scale min side to target, then center-crop (not crop-first)."""
        size = 512  # smaller target so the synthetic stripe test stays light
        # Left=red, middle=green, right=blue stripes on 900×300.
        src = Image.new("RGB", (900, 300))
        for x in range(900):
            if x < 300:
                c = (255, 0, 0)
            elif x < 600:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
            for y in range(300):
                src.putpixel((x, y), c)
        out = resize_square_rgb(src, size)
        self.assertEqual(out.size, (size, size))
        # After scale: 1536×512; center crop keeps the green band.
        r, g, b = out.getpixel((size // 2, size // 2))
        self.assertGreater(g, 200)  # green dominant
        self.assertLess(r, 80)
        self.assertLess(b, 80)

    def test_resize_upscales_small_images(self):
        out = resize_square_rgb(_solid((1, 2, 3), (100, 80)), 512)
        self.assertEqual(out.size, (512, 512))

    def test_normalize_captions_dedupe(self):
        caps = normalize_captions(["A dog.", "a dog.", "A cat."], min_count=2)
        self.assertEqual(len(caps), 2)

    def test_normalize_captions_too_few(self):
        with self.assertRaises(ValueError):
            normalize_captions(["only one"], min_count=2)

    def test_rsicd_near_duplicates_rejected(self):
        """RSICD-style captions must not count as multi-caption diversity."""
        raw = [
            "some planes are parked in an airport.",
            "Some planes are parked at an airport.",
            "some planes are parked in an airport .",
        ]
        self.assertTrue(captions_are_near_duplicate(raw[0], raw[1]))
        self.assertTrue(captions_are_near_duplicate(raw[0], raw[2]))
        self.assertFalse(caption_set_is_diverse(raw, min_count=2))
        with self.assertRaises(ValueError):
            normalize_captions(raw, min_count=2)

    def test_diverse_captions_kept(self):
        raw = [
            "Many aircraft parked beside a terminal building.",
            "Airport runway with taxiing jet near hangars.",
        ]
        caps = normalize_captions(raw, min_count=2)
        self.assertEqual(len(caps), 2)


class TestSaveLoadRoundtrip(unittest.TestCase):
    def test_roundtrip(self):
        records = []
        for i, (domain, color) in enumerate([
            (DOMAIN_GENERAL, (255, 0, 0)),
            (DOMAIN_SATELLITE, (0, 128, 0)),
            (DOMAIN_GENERAL, (0, 0, 255)),
            (DOMAIN_SATELLITE, (128, 128, 0)),
        ]):
            img = resize_square_rgb(_solid(color, (600, 500)))
            rec = {
                "id": f"{domain}-{i:03d}",
                "domain": domain,
                "source": "unit-test",
                "captions": [f"caption alpha {i}", f"caption beta {i}"],
                "query": f"search for scene {i}",
                "unrelated_query": "blue sailboat on the ocean",
                "image": img,
            }
            validate_record(rec)
            records.append(rec)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ds"
            save_dataset(records, out, write_sidecar_jpegs=True)
            self.assertTrue((out / "dataset_info.json").is_file())
            self.assertTrue((out / "hf").is_dir())
            self.assertTrue((out / "metadata.jsonl").is_file())
            self.assertTrue(curated_dataset_available(out))

            info = json.loads((out / "dataset_info.json").read_text())
            self.assertEqual(info["num_rows"], 4)
            self.assertEqual(info["image_size"], DEFAULT_IMAGE_SIZE)

            loaded = load_dataset_records(out)
            self.assertEqual(len(loaded), 4)
            self.assertEqual(
                loaded[0]["image"].size, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
            )
            self.assertGreaterEqual(len(loaded[0]["captions"]), 2)

            lazy = open_lazy_dataset(out, image_cache_size=2)
            self.assertEqual(len(lazy), 4)
            # meta-only path must not require image decode
            m0 = lazy.meta(0)
            self.assertNotIn("image", m0)
            self.assertEqual(m0["id"], loaded[0]["id"])
            rec0 = lazy[0]
            self.assertEqual(
                rec0["image"].size, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
            )
            # LRU: re-fetch same index hits cache; overflow drops oldest
            _ = lazy[1]
            _ = lazy[2]
            self.assertEqual(len(lazy._image_cache), 2)
            _ = lazy[0]
            self.assertIn(0, lazy._image_cache)

            train_rows = load_curated_training_rows(out, seed=0)
            self.assertEqual(len(train_rows), 4)
            self.assertIn(QUERY_CACHE_RELATED_KEY, train_rows[0])
            self.assertTrue(train_rows[0][QUERY_CACHE_RELATED_KEY])

            mixed, col_i, col_c, root = load_stage1_training_rows(
                curated_dataset_dir=str(out),
                prefer_curated=True,
                seed=0,
            )
            self.assertEqual(col_i, "image")
            self.assertEqual(col_c, "caption")
            self.assertIsNone(root)
            self.assertEqual(len(mixed), 4)

            # enrich should no-op OpenRouter for pre-filled queries
            enriched = enrich_rows_with_text_queries(mixed, skip_generation=True)
            self.assertEqual(len(enriched), 4)
            self.assertTrue(enriched[0][QUERY_CACHE_RELATED_KEY])


class TestImageCaptionMultiCaption(unittest.TestCase):
    def test_extra_caption_as_related(self):
        class FakeTok:
            pad_token_id = 0

            def __call__(self, text, **kwargs):
                import torch

                # fixed length for stacking
                ids = [1, 2, 3]
                return {
                    "input_ids": torch.tensor([ids]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }

        class FakeProc:
            def __call__(self, images, return_tensors="pt"):
                import torch

                return {"pixel_values": torch.zeros(1, 3, 32, 32)}

        rows = [{
            "image": resize_square_rgb(_solid((1, 2, 3))),
            "caption": "primary caption about dogs",
            "captions": [
                "primary caption about dogs",
                "two pets playing outside",
            ],
            QUERY_CACHE_RELATED_KEY: "",
            QUERY_CACHE_UNRELATED_KEY: "skyscraper at dusk",
        }]
        ds = ImageCaptionDataset(
            rows,
            image_processor=FakeProc(),
            tokenizer=FakeTok(),
            with_text_queries=True,
        )
        sample = ds[0]
        self.assertIn("query_input_ids", sample)
        self.assertIn("unrelated_input_ids", sample)


class TestGenerateHelpers(unittest.TestCase):
    def test_attach_queries_offline(self):
        from generate_datasets import attach_queries

        items = [{
            "captions": ["a red barn in a field", "farm building countryside"],
            "query": "",
            "unrelated_query": "",
        }]
        attach_queries(items, skip_generation=True, config_path="missing.yml")
        self.assertEqual(items[0]["query"], "farm building countryside")
        self.assertTrue(items[0]["unrelated_query"])

    def test_diversify_offline_fixes_rsicd_clones(self):
        from generate_datasets import diversify_record_captions

        items = [{
            "id": "satellite-0",
            "domain": DOMAIN_SATELLITE,
            "captions": [
                "some planes are parked in an airport.",
                "Some planes are parked at an airport.",
                "some planes are parked in an airport .",
            ],
        }]
        diversify_record_captions(
            items, skip_api=True, config_path="missing.yml"
        )
        self.assertTrue(
            caption_set_is_diverse(items[0]["captions"], min_count=2),
            items[0]["captions"],
        )

    def test_progress_store_concurrent_saves(self):
        """Reproduce the FileNotFoundError race on shared .tmp under many writers."""
        import tempfile
        import threading
        from pathlib import Path

        from generate_datasets import ProgressStore

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".generate_progress.json"
            store = ProgressStore(path)
            errors: list[BaseException] = []
            n_threads = 16
            n_each = 40

            def worker(tid: int) -> None:
                try:
                    for i in range(n_each):
                        key = f"general/coco_{tid}_{i}.jpg"
                        store.set_queries(
                            key,
                            f"query {tid} {i}",
                            f"unrelated {tid} {i}",
                            flush_every=3,  # force frequent concurrent saves
                        )
                        if i % 7 == 0:
                            store.save()
                except BaseException as exc:  # noqa: BLE001 — collect for assert
                    errors.append(exc)

            threads = [
                threading.Thread(target=worker, args=(t,)) for t in range(n_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"concurrent save failures: {errors!r}")
            store.save()
            self.assertTrue(path.is_file())
            reloaded = ProgressStore(path)
            self.assertEqual(len(reloaded.data), n_threads * n_each)
            for tid in range(n_threads):
                for i in range(n_each):
                    key = f"general/coco_{tid}_{i}.jpg"
                    self.assertTrue(reloaded.queries_done(key), key)

    def test_progress_resume_skips_diversify_and_queries(self):
        """Second run must not re-LLM items already cached after a mid-run crash."""
        import tempfile
        from pathlib import Path

        from generate_datasets import (
            ProgressStore,
            attach_queries,
            diversify_record_captions,
            record_stable_key,
        )

        with tempfile.TemporaryDirectory() as td:
            progress_path = Path(td) / ".generate_progress.json"
            # Simulate run 1: diversify finished, queries partially done, then crash.
            prog1 = ProgressStore(progress_path)
            items_run1 = [
                {
                    "id": "general-0",
                    "domain": DOMAIN_GENERAL,
                    "image_path": str(Path(td) / "general" / "coco_1.jpg"),
                    "captions": [
                        "a red barn in a green field under blue sky",
                        "farm building with silo in countryside landscape",
                    ],
                    "query": "",
                    "unrelated_query": "",
                },
                {
                    "id": "satellite-0",
                    "domain": DOMAIN_SATELLITE,
                    "image_path": str(Path(td) / "satellite" / "sky_000001.jpg"),
                    "captions": [
                        "aerial view of airport runway with parked aircraft",
                        "satellite image of terminal buildings and taxiways",
                    ],
                    "query": "",
                    "unrelated_query": "",
                },
            ]
            for it in items_run1:
                it["_key"] = record_stable_key(it)
            prog1.set_captions(
                items_run1[0]["_key"], items_run1[0]["captions"]
            )
            prog1.set_captions(
                items_run1[1]["_key"], items_run1[1]["captions"]
            )
            prog1.set_queries(
                items_run1[0]["_key"],
                "red barn countryside photo",
                "underwater coral reef fish",
            )
            prog1.save()

            # Run 2: reload progress, apply, skip cached work.
            prog2 = ProgressStore(progress_path)
            items_run2 = [
                {
                    "id": "general-0",
                    "domain": DOMAIN_GENERAL,
                    "image_path": str(Path(td) / "general" / "coco_1.jpg"),
                    # Near-dup source captions — would trigger diversify without cache.
                    "captions": [
                        "some planes are parked in an airport.",
                        "Some planes are parked at an airport.",
                    ],
                    "query": "",
                    "unrelated_query": "",
                },
                {
                    "id": "satellite-0",
                    "domain": DOMAIN_SATELLITE,
                    "image_path": str(Path(td) / "satellite" / "sky_000001.jpg"),
                    "captions": [
                        "some planes are parked in an airport.",
                        "Some planes are parked at an airport.",
                    ],
                    "query": "",
                    "unrelated_query": "",
                },
            ]
            n_cap, n_q = prog2.apply_to_records(items_run2)
            self.assertEqual(n_cap, 2)
            self.assertEqual(n_q, 1)
            # Cached diverse captions restored (not the near-dup seeds above).
            self.assertIn("red barn", items_run2[0]["captions"][0])
            self.assertIn("aerial view", items_run2[1]["captions"][0])
            self.assertEqual(items_run2[0]["query"], "red barn countryside photo")

            diversify_record_captions(
                items_run2,
                skip_api=True,
                config_path="missing.yml",
                progress=prog2,
            )
            # Must keep restored captions, not offline-rewrite the near-dups.
            self.assertIn("red barn", items_run2[0]["captions"][0])
            self.assertIn("aerial view", items_run2[1]["captions"][0])

            attach_queries(
                items_run2,
                skip_generation=True,
                config_path="missing.yml",
                progress=prog2,
            )
            # Item 0 was cached; item 1 gets offline fill.
            self.assertEqual(items_run2[0]["query"], "red barn countryside photo")
            self.assertEqual(
                items_run2[0]["unrelated_query"], "underwater coral reef fish"
            )
            self.assertTrue(items_run2[1]["query"])
            self.assertTrue(prog2.queries_done(items_run2[1]["_key"]))


class TestQualityAudit(unittest.TestCase):
    def test_flag_row_offline_and_query_eq_caption(self):
        from trisearch_quality import OFFLINE_UNRELATED, flag_row

        row = {
            "id": "general-0",
            "domain": DOMAIN_GENERAL,
            "source": "unit",
            "captions": [
                "a red barn in a green field under blue sky",
                "farm building with silo in countryside landscape",
            ],
            "query": "a red barn in a green field under blue sky",
            "unrelated_query": OFFLINE_UNRELATED,
        }
        codes = {f["code"] for f in flag_row(row)}
        self.assertIn("query_eq_caption", codes)
        self.assertIn("offline_unrelated", codes)

    def test_audit_rows_writes_repair_estimate(self):
        from trisearch_quality import audit_rows

        rows = [
            {
                "id": f"general-{i}",
                "domain": DOMAIN_GENERAL,
                "source": "unit",
                "captions": [f"scene alpha {i} unique", f"scene beta {i} different objects"],
                "query": "same query for many images",
                "unrelated_query": "underwater sea creatures",
            }
            for i in range(20)
        ]
        flags, summary = audit_rows(
            rows, query_freq_threshold=10, unrelated_freq_threshold=10
        )
        self.assertGreater(summary["num_flagged"], 0)
        self.assertIn("repair_estimate", summary)
        self.assertTrue(any("duplicate_query_frequent" in r["codes"] for r in flags))

    def test_local_unrelated_and_query_repair(self):
        from trisearch_quality import (
            OFFLINE_UNRELATED,
            assign_unrelated_from_bank,
            build_distractor_bank,
            is_generic_unrelated,
            local_query_repair,
            write_metadata_jsonl,
        )

        bank = build_distractor_bank(500)
        self.assertGreater(len(bank), 100)
        used: set[str] = set()
        cursor = [0]
        row = {
            "id": "general-1",
            "domain": DOMAIN_GENERAL,
            "source": "unit",
            "captions": [
                "a pizza with vegetables and lemon wedge on a plate",
                "quiche style dish with greens beside a fork",
            ],
            "query": "Image of a pizza with vegetables and lemon wedge on a plate",
            "unrelated_query": OFFLINE_UNRELATED,
        }
        uq = assign_unrelated_from_bank(
            row, bank=bank, used=used, bank_index=cursor
        )
        self.assertIsNotNone(uq)
        self.assertFalse(is_generic_unrelated(uq))
        q = local_query_repair(
            row,
            codes=["query_boilerplate", "query_near_caption"],
            query_caption_overlap=0.85,
        )
        self.assertIsNotNone(q)
        self.assertFalse(str(q).lower().startswith("image of"))

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "metadata.jsonl"
            row2 = dict(row)
            row2["query"] = q
            row2["unrelated_query"] = uq
            row2["file_name"] = "images/general/general-1.jpg"
            write_metadata_jsonl(path, [row2])
            text = path.read_text(encoding="utf-8")
            self.assertIn(q, text)
            self.assertIn(uq, text)


class TestOfficialSplits(unittest.TestCase):
    def test_assign_one_sixteenth_stratified(self):
        from trisearch_data_format import assign_official_splits, apply_official_splits

        rows = []
        for d, prefix, n in (("general", "g", 32), ("satellite", "s", 32)):
            for i in range(n):
                rows.append({"id": f"{prefix}-{i:04d}", "domain": d})
        mapping = assign_official_splits(rows, seed=42, test_denom=16)
        # 32//16 = 2 test per domain
        self.assertEqual(sum(1 for v in mapping.values() if v == "test"), 4)
        self.assertEqual(sum(1 for v in mapping.values() if v == "train"), 60)
        # Deterministic
        mapping2 = assign_official_splits(rows, seed=42, test_denom=16)
        self.assertEqual(mapping, mapping2)
        apply_official_splits(rows, force=True)
        self.assertTrue(all(r["split"] in ("train", "test") for r in rows))


class TestDatasetCard(unittest.TestCase):
    def test_render_card_has_front_matter_and_sections(self):
        from trisearch_dataset_card import render_dataset_card

        stats = {
            "dataset_name": "TriSearch-v1",
            "dataset_version": "0.0.1",
            "format_version": 1,
            "image_size": 1024,
            "num_rows": 100,
            "domains": {"general": 50, "satellite": 50},
            "sources": {"bitmind/MS-COCO": 50, "SkyScript": 50},
            "captions_per_image": {"min": 2, "max": 4, "mean": 3.0},
            "caption_char_len": {"mean": 40.0, "p10": 20, "p90": 60},
            "query_char_len": {"mean": 25.0, "p10": 12, "p90": 40},
            "unique_queries": 90,
            "unique_unrelated": 80,
            "query_collision_rate": 0.1,
            "unrelated_collision_rate": 0.2,
            "generic_unrelated_count": 0,
            "quality": {"num_flagged": 5, "num_rows": 100, "pct_flagged": 5.0},
            "splits": {
                "train": 94,
                "test": 6,
                "test_denom": 16,
                "test_fraction": 1 / 16,
                "seed": 42,
                "by_domain": {
                    "general": {"train": 47, "test": 3},
                    "satellite": {"train": 47, "test": 3},
                },
            },
            "layout": {
                "metadata_jsonl": True,
                "splits_json": True,
                "train_parquet_shards": 2,
                "test_parquet_shards": 1,
                "parquet_shards": 3,
                "sidecar_images": True,
                "hf_arrow": False,
            },
            "examples": [
                {
                    "id": "general-0",
                    "domain": "general",
                    "source": "unit",
                    "captions": ["a", "b"],
                    "query": "q",
                    "unrelated_query": "u",
                }
            ],
            "generated_on": "2026-07-09",
        }
        card = render_dataset_card(stats, repo_id="org/trisearch-v1")
        self.assertTrue(card.startswith("---"))
        self.assertIn("pretty_name:", card)
        self.assertIn("load_dataset", card)
        self.assertIn("0.0.1", card)
        self.assertIn("split: test", card)
        self.assertIn("Official splits", card)
        self.assertNotIn("no official validation/test split", card.lower())
        self.assertIn("SkyScript", card)
        self.assertIn("unrelated_query", card)
        self.assertIn("composite", card.lower())


class TestViewDataset(unittest.TestCase):
    def test_build_viewer_api_info(self):
        """Gradio 5 crashes on some State wiring; catch that in CI."""
        from view_dataset import build_viewer

        records = []
        for i in range(3):
            img = resize_square_rgb(_solid((i * 40, 80, 120)))
            records.append({
                "id": f"t-{i}",
                "domain": DOMAIN_GENERAL if i % 2 == 0 else DOMAIN_SATELLITE,
                "source": "unit-test",
                "captions": [f"cap a {i}", f"cap b {i}"],
                "query": f"query {i}",
                "unrelated_query": "unrelated sailboat photo",
                "image": img,
            })
        demo = build_viewer(records)
        # This is the call path that failed in the browser (get_api_info).
        info = demo.get_api_info()
        self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
