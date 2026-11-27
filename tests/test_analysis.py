"""Standard-library tests of the report tables: python -m unittest discover -s tests"""

import random
import tempfile
import unittest
from pathlib import Path

from oprb_eval import analysis, score

TYPES = ("Black_Ink", "Sim")


def page(i, category, cer, f1, ref_chars=100, hyp_words=10, layout="1col"):
    return {
        "key": f"{category}/p{i:04d}",
        "category": category,
        "occlusion_type": TYPES[i % 2],
        "density": "none" if i % 2 else "1p0pct",
        "layout": layout,
        "status": "ok",
        "truncated": False,
        "cer": cer,
        "wer": cer,
        "precision": f1,
        "recall": f1,
        "f1": f1,
        "char_errors": round(cer * ref_chars),
        "word_errors": 0,
        "ref_chars": ref_chars,
        "ref_words": 10,
        "hyp_words": hyp_words,
    }


def write_model(root, model, rows):
    for protocol in score.PROTOCOLS:
        score.write_csv(root / model / f"per_image_{protocol}_t.csv", rows)


class TableTests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        categories = ["text"] * 40 + ["nontext"] * 20
        base = [rng.random() * 0.5 for _ in categories]
        # Model "good" is better than "bad" by 0.1 CER on every page; "same" equals "bad".
        self.models = ("good", "bad", "same")
        rows = {
            "good": [page(i, c, b, 1 - b) for i, (c, b) in enumerate(zip(categories, base))],
            "bad": [page(i, c, b + 0.1, 0.9 - b) for i, (c, b) in enumerate(zip(categories, base))],
        }
        rows["same"] = [dict(r) for r in rows["bad"]]
        for model, model_rows in rows.items():
            # Write in a shuffled order: loading must line pages up by key.
            shuffled = model_rows[:]
            rng.shuffle(shuffled)
            write_model(self.root, model, shuffled)
        self.table = analysis.load_all(self.root, "t", self.models)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rows_are_typed_and_aligned(self):
        rows = self.table["columns"]
        self.assertEqual([r["key"] for r in rows["good"]], [r["key"] for r in rows["bad"]])
        self.assertIsInstance(rows["good"][0]["char_errors"], int)
        self.assertIs(rows["good"][0]["truncated"], False)

    def test_models_on_different_pages_are_refused(self):
        rows = [page(i, "text", 0.1, 0.9) for i in range(5)]
        write_model(self.root, "short", rows)
        with self.assertRaises(ValueError):
            analysis.load_all(self.root, "t", ("good", "short"))

    def test_paired_difference(self):
        rows = analysis.paired(self.table)
        cer = next(r for r in rows if r["protocol"] == "columns" and r["metric"] == "cer"
                   and (r["model_a"], r["model_b"]) == ("good", "bad"))
        self.assertAlmostEqual(cer["diff"], -0.1)
        # The difference is the same on every page, so the interval has no width.
        self.assertAlmostEqual(cer["diff_low"], -0.1)
        self.assertAlmostEqual(cer["diff_high"], -0.1)
        self.assertTrue(cer["excludes_zero"])
        self.assertEqual((cer["share_a_better"], cer["share_b_better"]), (1.0, 0.0))
        tie = next(r for r in rows if r["protocol"] == "columns" and r["metric"] == "f1"
                   and (r["model_a"], r["model_b"]) == ("bad", "same"))
        self.assertEqual((tie["diff"], tie["excludes_zero"], tie["share_a_better"]), (0, False, 0))

    def test_micro_cer_weights_pages_by_length(self):
        rows = [page(0, "text", 0.1, 0.9, ref_chars=1000), page(1, "text", 50.0, 0.0, ref_chars=2)]
        self.assertAlmostEqual(analysis.micro_cer(rows), (100 + 100) / 1002)
        self.assertGreater(sum(r["cer"] for r in rows) / 2, 25)

    def test_overall_ranks(self):
        rows = [r for r in analysis.overall(self.table) if r["protocol"] == "columns"]
        by_model = {r["model"]: r for r in rows}
        self.assertEqual(by_model["good"]["rank_cer"], 1)
        self.assertEqual(by_model["good"]["rank_f1"], 1)
        self.assertEqual(by_model["good"]["rank_cer_micro"], 1)

    def test_category_gap_and_layouts(self):
        rows = analysis.category_gap(self.table)
        gap = next(r for r in rows if r["protocol"] == "columns" and r["model"] == "good" and r["layout"] == "all")
        self.assertEqual((gap["n_text"], gap["n_nontext"]), (40, 20))
        self.assertAlmostEqual(gap["cer_gap"], gap["cer_nontext"] - gap["cer_text"])
        self.assertLessEqual(gap["cer_gap_low"], gap["cer_gap"])
        self.assertLessEqual(gap["cer_gap"], gap["cer_gap_high"])
        two = next(r for r in rows if r["protocol"] == "columns" and r["model"] == "good" and r["layout"] == "2col")
        self.assertEqual((two["n_text"], two["n_nontext"]), (0, 0))

    def test_groups_and_spread(self):
        groups = analysis.by_group(self.table, "occlusion_type")
        cells = [r for r in groups if r["protocol"] == "columns" and r["category"] == "text"]
        self.assertEqual(sum(r["n"] for r in cells if r["model"] == "good"), 40)
        self.assertEqual({r["rank_cer"] for r in cells if r["model"] == "good"}, {1})
        spread = analysis.spread(groups, "occlusion_type")
        one = next(r for r in spread if (r["protocol"], r["category"], r["model"]) == ("columns", "text", "good"))
        self.assertAlmostEqual(one["range"], one["best_value"] - one["worst_value"])
        self.assertIn(one["best"], TYPES)

    def test_reference_lengths_count_every_page(self):
        rows = [r for r in analysis.reference_lengths(self.table) if r["protocol"] == "columns"]
        self.assertEqual(sum(r["pages"] for r in rows), 60)

    def test_qualitative_picks_are_reproducible(self):
        picks = analysis.qualitative_picks(self.table)
        self.assertEqual(picks, analysis.qualitative_picks(self.table))
        rules = {(p["rule"], p["model"]) for p in picks}
        self.assertIn(("shared_median", "all"), rules)
        self.assertEqual(len(picks), 1 + 4 * len(self.models))

    def test_quantile_row_breaks_ties_by_key(self):
        rows = [{"key": k, "f1": 0.5} for k in ("c", "a", "b")]
        self.assertEqual(analysis.quantile_row(rows, "f1", 0.0)["key"], "a")
        self.assertEqual(analysis.quantile_row(rows, "f1", 1.0)["key"], "c")

    def test_run_writes_every_table(self):
        out = self.root / "analysis"
        tables = analysis.run(self.root, "t", out, self.models)
        for name in tables:
            self.assertTrue((out / f"{name}_t.csv").exists(), name)


if __name__ == "__main__":
    unittest.main()
