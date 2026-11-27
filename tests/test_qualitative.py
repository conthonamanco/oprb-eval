"""Standard-library tests of the qualitative excerpts: python -m unittest discover -s tests"""

import json
import tempfile
import unittest
from pathlib import Path

from oprb_eval import qualitative
from oprb_eval.data import GTToken, Page


def token(text, x, y, w=40, h=10):
    return GTToken(text, x, y, x + w, y + h)


def one_column(lines):
    """Each line a list of words, lines 20 px apart. Word positions shift from line to line, as in
    real text, so the gaps between words never line up into what looks like a column gutter."""
    return [token(word, 50 + 44 * i + (17 * row) % 40, 100 + 20 * row)
            for row, words in enumerate(lines) for i, word in enumerate(words)]


def gt_file(path, tokens):
    rows = [f"{t.text}\t{t.x0:.0f}\t{t.y0:.0f}\t{t.x1:.0f}\t{t.y1:.0f}\t0\t0\t0\tfont\tparagraph" for t in tokens]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class LineTests(unittest.TestCase):
    def test_one_column_lines_and_placeholders(self):
        tokens = one_column([["a", "b", "c"], ["##LTLine##"], ["d", "e"]])
        lines = qualitative.gt_lines(tokens)
        self.assertEqual([[t.text for t in line] for line in lines], [["a", "b", "c"], ["d", "e"]])

    def test_two_columns_are_read_one_after_the_other(self):
        # File order runs along whole printed lines: left half, then right half of each line.
        tokens = []
        for row in range(12):
            y = 100 + 20 * row
            tokens += [token(f"L{row}a", 50, y, 150), token(f"L{row}b", 210, y, 150),
                       token(f"R{row}a", 450, y, 150), token(f"R{row}b", 610, y, 150)]
        lines = qualitative.gt_lines(tokens)
        texts = [" ".join(t.text for t in line) for line in lines]
        self.assertEqual(texts[0], "L0a L0b")
        self.assertEqual(texts[11], "L11a L11b")
        self.assertEqual(texts[12], "R0a R0b")
        self.assertEqual(len(texts), 24)

    def test_windows_group_lines(self):
        lines = qualitative.gt_lines(one_column([["a"], ["b"], ["c"], ["d"]]))
        groups = qualitative.windows(lines, size=3)
        self.assertEqual([[t.text for t in w] for w in groups], [["a", "b", "c"], ["d"]])


class SpanTests(unittest.TestCase):
    def test_exact_span_is_found(self):
        reference = "the quick brown fox".split()
        hypothesis = "header text the quick brown fox jumps over".split()
        start, end, errors = qualitative.best_span(reference, hypothesis)
        self.assertEqual((hypothesis[start:end], errors), (reference, 0))

    def test_span_with_a_missing_word(self):
        reference = "one two three four five".split()
        hypothesis = "zero one two four five six".split()
        _, _, errors = qualitative.best_span(reference, hypothesis)
        self.assertEqual(errors, 1)

    def test_empty_output(self):
        self.assertEqual(qualitative.best_span(["a", "b"], []), (0, 0, 2))

    def test_box_scale(self):
        self.assertEqual(qualitative.box_scale(990, 1700, 2200), 2.2)
        self.assertEqual(qualitative.box_scale(1500, 1700, 2200), 1.0)
        self.assertEqual(qualitative.box_scale(900, 800, 1000), 1.0)


class BuildTests(unittest.TestCase):
    def test_rules_pick_the_documented_window(self):
        lines = [["alpha", "beta", "gamma"], ["delta", "epsilon", "zeta"], ["eta", "theta", "iota"],
                 ["kappa", "lambda", "mu"], ["nu", "xi", "omicron"], ["pi", "rho", "sigma"]]
        reference = " ".join(" ".join(line) for line in lines)
        # Model "good" reads everything; "bad" loses the second window (lines 4-6).
        outputs = {"good": reference, "bad": " ".join(" ".join(line) for line in lines[:3])}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_file(root / "p.txt", one_column(lines))
            page = Page("text", "Black_Ink__1p0pct__x", root / "p.jpg", root / "p.txt", "Black_Ink", "1p0pct")
            predictions = {m: {page.key: {"status": "ok", "text": t, "truncated": False}} for m, t in outputs.items()}
            picks = [
                {"rule": "shared_median", "model": "all", "key": page.key, "value": "0.5"},
                {"rule": "median_f1", "model": "good", "key": page.key, "value": "1.0"},
                {"rule": "most_missing_words", "model": "bad", "key": page.key, "value": "0.5"},
            ]
            examples = qualitative.build(picks, {page.key: page}, predictions, root / "out", images=False)
            saved = json.loads((root / "out" / "examples.json").read_text(encoding="utf-8"))
        shared = examples[0]["excerpt"]
        self.assertEqual(shared["reference"], "kappa lambda mu nu xi omicron pi rho sigma")
        self.assertEqual(shared["models"]["good"]["word_errors"], 0)
        self.assertEqual(shared["models"]["bad"]["word_errors"], 9)
        # With no errors anywhere for "good", the earliest window wins the tie.
        self.assertEqual(examples[1]["excerpt"]["reference"], "alpha beta gamma delta epsilon zeta eta theta iota")
        self.assertEqual(examples[2]["output_head"], outputs["bad"])
        self.assertEqual(examples[2]["output_words"], {"good": 18, "bad": 9})
        self.assertEqual(len(saved), 3)


if __name__ == "__main__":
    unittest.main()
