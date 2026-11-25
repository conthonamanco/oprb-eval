"""Adapter logic that can be checked without the heavy libraries or any weights."""

import importlib
import os
import random
import sys
import tempfile
import tomllib
import types
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from oprb_eval import models, pilot
from oprb_eval.adapters.doctr_words import page_text
from oprb_eval.data import Page
from oprb_eval.adapters.vlm import LOOP_WINDOW, cut_at_end, is_looping

REPO = Path(__file__).resolve().parents[1]


class RegistryTest(unittest.TestCase):
    def test_every_configured_adapter_exists(self):
        for name in ("models.toml", "smoke.toml"):
            configs = tomllib.loads((REPO / "configs" / name).read_text(encoding="utf-8"))
            for config in configs.values():
                module, _, cls = models.ADAPTERS[config["adapter"]].partition(":")
                imported = importlib.import_module(f"oprb_eval.adapters.{module}")
                self.assertTrue(hasattr(imported, cls))

    def test_heavy_libraries_are_not_imported_with_the_adapters(self):
        for name in ("transformers", "doctr", "rapidocr", "tesserocr"):
            self.assertNotIn(name, sys.modules)


class LoopTest(unittest.TestCase):
    def test_detects_short_and_long_periods(self):
        for period in (1, 7, 64):
            tokens = list(range(1000, 1100)) + list(range(period)) * (LOOP_WINDOW // period + 1)
            self.assertTrue(is_looping(tokens), period)

    def test_ignores_normal_text_and_short_repeats(self):
        self.assertFalse(is_looping(list(range(LOOP_WINDOW * 2))))
        self.assertFalse(is_looping([5] * (LOOP_WINDOW - 1)))
        self.assertFalse(is_looping([5] * (LOOP_WINDOW - 1) + [6]))
        self.assertFalse(is_looping(list(range(65)) * 10))

    def test_cut_at_end(self):
        self.assertEqual(cut_at_end([4, 5, 2, 0, 0], {2, 0}), ([4, 5], True))
        self.assertEqual(cut_at_end([4, 5], {2, 0}), ([4, 5], False))


class DoctrTest(unittest.TestCase):
    def test_relative_polygons_are_scaled_to_pixels_and_ordered(self):
        def word(text, x, y):
            x, y, w, h = x / 1000, y / 500, 60 / 1000, 12 / 500
            return SimpleNamespace(value=text, geometry=((x, y), (x + w, y), (x + w, y + h), (x, y + h)))

        words = [word("third", 10, 60), word("second", 75, 20), word("first", 10, 20)]
        page = SimpleNamespace(blocks=[SimpleNamespace(lines=[SimpleNamespace(words=words)])])
        self.assertEqual(page_text(page, 1000, 500), "first second\nthird")

    def test_straight_boxes_and_empty_pages(self):
        word = SimpleNamespace(value="x", geometry=((0.1, 0.1), (0.2, 0.15)))
        page = SimpleNamespace(blocks=[SimpleNamespace(lines=[SimpleNamespace(words=[word])])])
        self.assertEqual(page_text(page, 100, 100), "x")
        self.assertEqual(page_text(SimpleNamespace(blocks=[]), 100, 100), "")


class FakeEngines(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"OPRB_CACHE": tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.configs = tomllib.loads((REPO / "configs" / "models.toml").read_text(encoding="utf-8"))

    def test_ppocr_parameters_and_empty_result(self):
        seen = {}
        box = [[0, 0], [50, 0], [50, 10], [0, 10]]
        low = [[x, y + 30] for x, y in box]
        results = {
            "two.jpg": SimpleNamespace(txts=("b", "a"), boxes=[low, box]),
            "none.jpg": SimpleNamespace(txts=None, boxes=None),
            "boxes_only.jpg": SimpleNamespace(boxes=[box]),
        }

        class RapidOCR:
            def __init__(self, params):
                seen.update(params)

            def __call__(self, path):
                return results[Path(path).name]

        fake = types.ModuleType("rapidocr")
        fake.RapidOCR = RapidOCR
        fake.EngineType = Enum("EngineType", {"ONNXRUNTIME": "onnxruntime"})
        fake.OCRVersion = Enum("OCRVersion", {"V5": "PP-OCRv5"})
        fake.LangDet = Enum("LangDet", {"CH": "ch"})
        fake.LangRec = Enum("LangRec", {"EN": "en"})
        fake.ModelType = Enum("ModelType", {"MOBILE": "mobile"})
        with mock.patch.dict(sys.modules, {"rapidocr": fake}), mock.patch.dict(os.environ, {"OPRB_THREADS": "1"}):
            model = models.build(self.configs["ppocrv5_mobile_en"])
            texts = [p.text for p in model.transcribe([Path(n) for n in results])]
        self.assertEqual(texts, ["a\nb", "", ""])
        self.assertEqual(seen["Rec.lang_type"], fake.LangRec.EN)
        self.assertEqual(seen["EngineConfig.onnxruntime.intra_op_num_threads"], 1)
        self.assertFalse(seen["EngineConfig.onnxruntime.use_cuda"])

    def test_old_tesseract_is_refused(self):
        fake = types.ModuleType("tesserocr")
        fake.tesseract_version = lambda: "tesseract 4.1.1\n leptonica-1.82"
        with mock.patch.dict(sys.modules, {"tesserocr": fake}):
            with self.assertRaisesRegex(SystemExit, "Tesseract 5 is required"):
                models.build(self.configs["tesseract"])


class PilotTest(unittest.TestCase):
    config = {"adapter": "constant", "text": "The cat sat"}

    def setUp(self):
        # The ground-truth file does not exist: the pilot must never open it.
        self.pages = [
            Page("text", str(i), Path(f"{i}.jpg"), Path("no_such_gt.txt"), "Black_Ink", "0p5pct")
            for i in range(2)
        ]

    def test_variants_are_parsed(self):
        self.assertEqual(
            pilot.parse_variant("scale=2,deskew=true,dtype=float16"),
            {"scale": 2, "deskew": True, "dtype": "float16"},
        )
        self.assertEqual(pilot.parse_variant(""), {})
        with self.assertRaises(SystemExit):
            pilot.parse_variant("scale")

    def test_agreement_is_measured_against_the_first_variant(self):
        with mock.patch("builtins.print"):
            results = pilot.compare(self.config, self.pages, [], [1, 2])
        self.assertEqual([r["batch_size"] for r in results], [1, 2])
        self.assertEqual([r["agreement"] for r in results], [1.0, 1.0])
        self.assertEqual(pilot.agreement(["a b", ""], ["a", "x"]), 2 / 3)
        self.assertIsNone(pilot.agreement([""], ["x"]))

    def test_a_failing_variant_does_not_stop_the_others(self):
        def build(config):
            if config["dtype"] == "bad":
                raise RuntimeError("NaN")
            return models.build(config)

        with mock.patch("builtins.print"), mock.patch.object(pilot, "build", build):
            results = pilot.compare(self.config, self.pages, ["dtype=bad", "dtype=good"], [1])
        self.assertIn("NaN", results[0]["error"])
        self.assertEqual(results[1]["agreement"], 1.0)


class PreprocessTest(unittest.TestCase):
    def test_deskew_and_upscale(self):
        try:
            import numpy  # noqa: F401
            from PIL import Image, ImageDraw
        except ImportError:
            self.skipTest("numpy or Pillow is not installed")
        from oprb_eval.preprocess import deskew, skew_angle, upscale, upscale_longest_edge

        page = Image.new("RGB", (800, 1030), "white")
        draw = ImageDraw.Draw(page)
        rng = random.Random(0)
        for y in range(120, 900, 16):
            x = 100
            while x < 680:
                w = rng.randint(12, 60)
                draw.rectangle((x, y, x + w, y + 8), fill="black")
                x += w + 6
        draw.ellipse((700, 300, 780, 380), fill="black")  # ink blot
        for tilt in (-5, -2.5, 0, 1.3, 5):
            turned = page.rotate(-tilt, resample=Image.BICUBIC, expand=True, fillcolor="white")
            self.assertAlmostEqual(skew_angle(turned), tilt, delta=0.25)
            self.assertAlmostEqual(skew_angle(deskew(turned)), 0, delta=0.25)
        self.assertEqual(skew_angle(Image.new("RGB", (50, 50), "white")), 0.0)
        self.assertEqual(upscale(page, 2).size, (1600, 2060))
        self.assertEqual(max(upscale_longest_edge(page, 1540).size), 1540)
        self.assertEqual(upscale_longest_edge(page, 500).size, page.size)


if __name__ == "__main__":
    unittest.main()
