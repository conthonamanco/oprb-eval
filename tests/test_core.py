"""Standard-library tests: python -m unittest discover -s tests"""

import csv
import io
import json
import random
import tarfile
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from oprb_eval import cli, data, runner, score
from oprb_eval.metrics import edit_distance, score_page
from oprb_eval.models import Prediction
from oprb_eval.text import clean_text, prediction_text, reference_text

REPO = Path(__file__).resolve().parents[1]
BOX = "\t".join(["0"] * 8)


def naive_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def write_page(root: Path, category: str, stem: str, words: str):
    for kind in ("img", "gt"):
        (root / category / kind).mkdir(parents=True, exist_ok=True)
    (root / category / "img" / f"{stem}.jpg").write_bytes(b"not a real image")
    rows = "".join(f"{w}\t{BOX}\tFont\tparagraph\n" for w in words.split())
    (root / category / "gt" / f"{stem}.txt").write_text(rows, encoding="utf-8")


def make_dataset(root: Path, per_stratum: int = 3):
    for category in ("text", "nontext"):
        for kind in ("Black_Ink", "Through_Dust"):
            for density in ("0p5pct", "1p0pct"):
                for i in range(per_stratum):
                    stem = f"{kind}__{density}__{i}.tar_x_{i}_ori"
                    write_page(root, category, stem, "The cat sat")


class FakeModel:
    """Fails on any batch that has a page whose name contains `bad`."""

    def __init__(self, bad=None):
        self.bad, self.batches = bad, []

    def transcribe(self, image_paths):
        self.batches.append(len(image_paths))
        if self.bad and any(self.bad in p.stem for p in image_paths):
            raise RuntimeError("boom")
        return [Prediction(f"the cat sat {p.stem}", 3, False) for p in image_paths]


class MetricsTest(unittest.TestCase):
    def test_edit_distance_matches_naive(self):
        rng = random.Random(0)
        for _ in range(500):
            a = "".join(rng.choices("abc ", k=rng.randint(0, 90)))
            b = "".join(rng.choices("abc ", k=rng.randint(0, 90)))
            self.assertEqual(edit_distance(a, b), naive_distance(a, b), (a, b))
            a, b = a.split(), b.split()
            self.assertEqual(edit_distance(a, b), naive_distance(a, b), (a, b))

    def test_notebook_toy_example(self):
        s = score_page("the cat sat", "the cat")
        self.assertAlmostEqual(s.wer, 1 / 3)
        self.assertAlmostEqual(s.cer, 4 / 11)
        self.assertAlmostEqual(s.precision, 1.0)
        self.assertAlmostEqual(s.recall, 2 / 3)
        self.assertAlmostEqual(s.f1, 0.8)

    def test_empty_hypothesis_and_insertions(self):
        s = score_page("a b", "")
        self.assertEqual((s.cer, s.wer, s.precision, s.recall, s.f1), (1.0, 1.0, 0.0, 0.0, 0.0))
        self.assertGreater(score_page("a b", "a b " + "x " * 10).wer, 1.0)

    def test_empty_reference_rejected(self):
        with self.assertRaises(ValueError):
            score_page("", "x")

    def test_word_order_only_affects_cer_wer(self):
        s = score_page("one two three four", "three four one two")
        self.assertEqual(s.f1, 1.0)
        self.assertGreater(s.wer, 0)

    def test_matches_jiwer(self):
        try:
            import jiwer
        except ImportError:
            self.skipTest("jiwer is not installed")
        rng = random.Random(1)
        for _ in range(200):
            ref = clean_text(" ".join(rng.choices(["ab", "c", "de", "ﬁ", "é"], k=rng.randint(1, 40))))
            hyp = clean_text(" ".join(rng.choices(["ab", "c", "de", "x"], k=rng.randint(0, 40))))
            s = score_page(ref, hyp)
            self.assertAlmostEqual(s.cer, jiwer.cer(ref, hyp))
            self.assertAlmostEqual(s.wer, jiwer.wer(ref, hyp))


class TextTest(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("  The\n\tﬁne  CAT "), "the fine cat")

    def test_pseudo_tokens(self):
        tokens = ["A", "##LTLine##", "b", "##LTFigure##"]
        self.assertEqual(reference_text(tokens, "notebook"), "a ##ltline## b ##ltfigure##")
        self.assertEqual(reference_text(tokens, "normalized"), "a b")

    def test_markup_removed_only_in_normalized(self):
        raw = "# Title\n\n**bold** text\n| a | b |\n|---|---|\n![fig](x.png)\n<td>cell</td> x &lt; y"
        self.assertEqual(prediction_text(raw, "normalized"), "title bold text | a | b | cell x < y")
        self.assertIn("# title", prediction_text(raw, "notebook"))

    def test_maths_is_not_treated_as_a_tag(self):
        text = "a < b and c > d, 2 * 3"
        self.assertEqual(prediction_text(text, "normalized"), text)


class DataTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_parse_stem(self):
        self.assertEqual(
            data.parse_stem("White_Whitener__2p0pct__105.tar_1804.06143.gz_MassiveMIMO_10_ori"),
            ("White_Whitener", "2p0pct"),
        )
        self.assertEqual(data.parse_stem("odd-name"), ("odd-name", "unknown"))

    def test_read_gt_skips_malformed_rows_and_keeps_quotes(self):
        path = self.root / "a.txt"
        rows = [f'﻿"Hi\t{BOX}\tF\tparagraph', "short\trow", f"\t{BOX}\tF\tx", f"you\t{BOX}\tF\tx"]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.assertEqual(data.read_gt_tokens(path), ['"Hi', "you"])

    def test_unpaired_files_are_an_error(self):
        make_dataset(self.root, 1)
        next((self.root / "text" / "gt").glob("*.txt")).unlink()
        with self.assertRaises(ValueError):
            data.collect_pages(self.root)

    def test_stratified_sample(self):
        make_dataset(self.root)
        pages = data.collect_pages(self.root)
        self.assertEqual(len(pages), 24)
        small, large = data.stratified_sample(pages, 4), data.stratified_sample(pages, 9)
        self.assertEqual(len(small), 8)
        self.assertEqual(len({(p.category, p.occlusion_type, p.density) for p in small}), 8)
        self.assertLess({p.key for p in small}, {p.key for p in large})
        self.assertEqual(small, data.stratified_sample(list(reversed(pages)), 4))
        self.assertEqual(len(data.stratified_sample(pages, 999)), 24)

    def test_extract_flat_ignores_archive_paths(self):
        archive = self.root / "x.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name in ("../../evil.jpg", "img/ok.jpg", "img/skip.txt"):
                info = tarfile.TarInfo(name)
                info.size = 3
                tar.addfile(info, io.BytesIO(b"abc"))
        out = self.root / "out" / "img"
        self.assertEqual(data.extract_flat(archive, out, ".jpg"), 2)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["evil.jpg", "ok.jpg"])
        (out / "ok.jpg").write_bytes(b"a")  # interrupted extraction
        data.extract_flat(archive, out, ".jpg")
        self.assertEqual((out / "ok.jpg").read_bytes(), b"abc")


class RunnerTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        make_dataset(self.root / "data", 1)
        self.pages = data.collect_pages(self.root / "data")
        self.out = self.root / "results" / "m"

    def run_model(self, model, config=None, **options):
        config = config or {"adapter": "x"}
        with redirect_stdout(io.StringIO()):
            return runner.run(lambda: model, "m", config, self.pages, self.out, **options)

    def test_same_stem_in_both_categories_is_not_skipped(self):
        self.assertEqual(self.run_model(FakeModel())["ok"], 8)

    def test_failed_pages_are_retried_and_good_ones_are_not(self):
        counts = self.run_model(FakeModel(bad="Black_Ink__0p5pct"))
        self.assertEqual((counts["ok"], counts["error"]), (6, 2))
        counts = self.run_model(FakeModel())
        self.assertEqual((counts["skipped"], counts["ok"], counts["error"]), (6, 2, 0))

    def test_one_bad_page_does_not_fail_its_batch(self):
        counts = self.run_model(FakeModel(bad="Black_Ink__0p5pct"), batch_size=3)
        self.assertEqual((counts["ok"], counts["error"]), (6, 2))

    def test_batching_gives_the_same_records(self):
        self.run_model(FakeModel(), batch_size=3)
        batched = {k: r["text"] for k, r in runner.load_predictions(self.out).items()}
        self.out = self.root / "results" / "single"
        self.run_model(FakeModel())
        single = {k: r["text"] for k, r in runner.load_predictions(self.out).items()}
        self.assertEqual(batched, single)
        for key, text in single.items():
            self.assertTrue(text.endswith(key.split("/", 1)[1]))

    def test_wrong_number_of_predictions_is_an_error(self):
        class Short:
            def transcribe(self, image_paths):
                return []

        self.assertEqual(self.run_model(Short(), batch_size=4)["error"], 8)

    def test_shards_cover_every_page_once_and_resume_across_layouts(self):
        self.assertEqual(self.run_model(FakeModel(), shard=(0, 2))["ok"], 4)
        self.assertEqual(len(runner.load_predictions(self.out)), 4)
        self.assertEqual(self.run_model(FakeModel(), shard=(1, 2))["ok"], 4)
        self.assertEqual(len(runner.load_predictions(self.out)), 8)
        self.assertEqual(self.run_model(FakeModel())["skipped"], 8)

    def test_success_beats_failure_from_a_file_read_later(self):
        self.run_model(FakeModel(bad="Black_Ink"))  # predictions.jsonl is read after shard files
        self.run_model(FakeModel(), shard=(0, 2))
        self.run_model(FakeModel(), shard=(1, 2))
        records = runner.load_predictions(self.out)
        self.assertEqual({r["status"] for r in records.values()}, {"ok"})

    def test_model_is_not_loaded_when_nothing_is_left(self):
        self.run_model(FakeModel())

        def explode():
            raise AssertionError("model was loaded")

        counts = runner.run(explode, "m", {"adapter": "x"}, self.pages, self.out)
        self.assertEqual(counts["skipped"], 8)

    def test_changed_config_is_refused_before_inference(self):
        self.run_model(FakeModel())
        model = FakeModel()
        with self.assertRaises(SystemExit):
            self.run_model(model, {"adapter": "x", "prompt": "new"})
        self.assertEqual(model.batches, [])

    def test_resume_after_torn_write(self):
        self.run_model(FakeModel())
        path = runner.predictions_file(self.out, (0, 1))
        path.write_bytes(path.read_bytes()[:-20])
        self.assertEqual(len(runner.load_predictions(self.out)), 7)
        self.assertEqual(self.run_model(FakeModel())["ok"], 1)
        self.assertEqual(len(runner.load_predictions(self.out)), 8)

    def test_predictions_contain_no_ground_truth(self):
        self.run_model(FakeModel())
        allowed = {"key", "session", "status", "output_tokens", "truncated", "text"}
        for record in runner.load_predictions(self.out).values():
            self.assertEqual(set(record), allowed)

    def test_each_session_records_its_environment(self):
        self.run_model(FakeModel())
        (line,) = (self.out / runner.SESSIONS).read_text(encoding="utf-8").splitlines()
        expected = {"session", "batch_size", "gpus", "packages", "python", "commit"}
        self.assertLessEqual(expected, set(json.loads(line)))

    def test_failures_count_as_empty_transcripts(self):
        class Perfect(FakeModel):
            def transcribe(self, image_paths):
                return [Prediction("The cat sat") for _ in super().transcribe(image_paths)]

        self.run_model(Perfect(bad="Black_Ink__0p5pct"))
        records = runner.load_predictions(self.out)
        rows, empty = score.score_pages(self.pages, records, "notebook")
        total = score.aggregate(rows, ())[0]
        self.assertEqual((total["n"], total["n_failed"], empty), (8, 2, 0))
        self.assertAlmostEqual(total["cer_mean"], 2 / 8)
        self.assertAlmostEqual(total["cer_micro"], 2 / 8)
        self.assertEqual(total["cer_median"], 0)

    def test_latency_is_one_image_at_a_time_and_excludes_failures(self):
        model = FakeModel(bad="Black_Ink__0p5pct")
        report = runner.measure_latency(
            lambda: model, "m", {"adapter": "x"}, self.pages, self.out, warmup=2
        )
        self.assertEqual(model.batches, [1] * 10)
        self.assertEqual((report["n"], report["n_failed"]), (6, 2))
        self.assertFalse(list(self.out.glob("predictions*")))


class CliTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        make_dataset(self.root / "data")
        self.model = ["--model", "smoke"]
        self.config = ["--config", str(REPO / "configs" / "smoke.toml")]

    def cli(self, command, *args):
        paths = ["--data", str(self.root / "data"), "--results", str(self.root / "results")]
        with redirect_stdout(io.StringIO()) as out:
            cli.main([command, *paths, *args])
        return out.getvalue()

    def test_end_to_end(self):
        with mock.patch.dict(data.EXPECTED, {"text": 12, "nontext": 12}):
            self.cli("stats")
            self.cli("run", *self.model, *self.config, "--sample", "4")
            with self.assertRaises(SystemExit):  # an incomplete run must not be scored
                self.cli("score", *self.model)
            self.assertIn("[notebook/sample4] n=8", self.cli("score", *self.model, "--sample", "4"))
            for shard in ("0/2", "1/2"):
                self.cli("run", *self.model, *self.config, "--shard", shard, "--batch-size", "5")
            self.assertIn("[notebook/full] n=24", self.cli("score", *self.model))
            self.assertIn("n=4", self.cli("latency", *self.model, *self.config, "--sample", "2"))
        path = self.root / "results" / "smoke" / "summary_notebook_full_type.csv"
        with path.open(encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 4)

    def test_incomplete_dataset_is_refused(self):
        options = {"stats": [], "run": [*self.model, *self.config], "score": self.model}
        for command, args in options.items():
            with self.assertRaisesRegex(SystemExit, "expected 6000"):
                self.cli(command, *args)

    def test_bad_shards_are_rejected(self):
        for shard in ("2/2", "x", "-1/2"):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                self.cli("run", *self.model, "--shard", shard)


class ConfigTest(unittest.TestCase):
    def test_exactly_five_models_with_pinned_weights(self):
        configs = tomllib.loads((REPO / "configs" / "models.toml").read_text(encoding="utf-8"))
        self.assertEqual(len(configs), 5)
        self.assertEqual({c["family"] for c in configs.values()}, {"ocr", "ocr-vlm", "vlm"})
        for name, config in configs.items():
            if "hf_id" in config:
                self.assertRegex(config["revision"], r"^[0-9a-f]{40}$", name)


if __name__ == "__main__":
    unittest.main()
