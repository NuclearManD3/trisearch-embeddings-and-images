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
        img = resize_square_rgb(_solid((10, 20, 30), (800, 400)), 512)
        self.assertEqual(img.size, (512, 512))
        self.assertEqual(img.mode, "RGB")

    def test_resize_scale_short_side_then_crop(self):
        """Landscape: scale min side to 512, then center-crop (not crop-first)."""
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
        out = resize_square_rgb(src, 512)
        self.assertEqual(out.size, (512, 512))
        # After scale: 1536×512; center crop keeps the green band.
        r, g, b = out.getpixel((256, 256))
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
            self.assertEqual(loaded[0]["image"].size, (512, 512))
            self.assertGreaterEqual(len(loaded[0]["captions"]), 2)

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
