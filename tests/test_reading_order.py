import math
import random
import unittest

from oprb_eval.reading_order import page_angle, to_text

HEIGHT, LINE, CHAR, SPACE = 12, 18, 7, 5


def words_at(x, y, text, level="word"):
    """Axis-aligned quads for a line of text that starts at (x, y)."""
    units = text.split() if level == "word" else [text]
    out = []
    for unit in units:
        w = CHAR * len(unit)
        out.append((unit, [(x, y), (x + w, y), (x + w, y + HEIGHT), (x, y + HEIGHT)]))
        x += w + SPACE
    return out


def rotate(items, degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return [(t, [(x * c - y * s, x * s + y * c) for x, y in quad]) for t, quad in items]


def paragraph(name, lines):
    return [f"{name}{i} alpha beta gamma delta epsilon zeta" for i in range(lines)]


def render(items, degrees=0, seed=0):
    items = rotate(items, degrees)
    random.Random(seed).shuffle(items)
    return to_text([t for t, _ in items], [q for _, q in items])


class ReadingOrderTest(unittest.TestCase):
    def check(self, build, expected):
        for level in ("word", "line"):
            for degrees in (-5, -2, 0, 3, 5):
                with self.subTest(level=level, degrees=degrees):
                    self.assertEqual(render(build(level), degrees).split(), expected.split())

    def test_single_column(self):
        lines = paragraph("p", 30)

        def build(level):
            return [w for i, t in enumerate(lines) for w in words_at(80, 100 + LINE * i, t, level)]

        self.check(build, " ".join(lines))

    def test_two_columns_with_title_and_footer(self):
        left, right = paragraph("l", 30), paragraph("r", 30)
        title = "a long title that clearly spans the gutter between both of the columns here"

        def build(level):
            items = words_at(120, 60, title, level)
            for i, t in enumerate(left):
                items += words_at(60, 120 + LINE * i, t, level)
            for i, t in enumerate(right):  # baselines of the columns do not line up
                items += words_at(400, 124 + LINE * i, t, level)
            return items + words_at(375, 120 + LINE * 33, "17", level)

        self.check(build, " ".join([title, *left, *right, "17"]))

    def test_equation_number_stays_on_its_line(self):
        lines = paragraph("p", 12)

        def build(level):
            items = [w for i, t in enumerate(lines) for w in words_at(80, 100 + LINE * i, t, level)]
            return items + words_at(120, 400, "x = y + z", level) + words_at(350, 400, "(1)", level)

        self.check(build, " ".join([*lines, "x = y + z (1)"]))

    def test_angle_ignores_direction_and_short_boxes(self):
        items = rotate([w for i in range(20) for w in words_at(50, 30 * i, "somewhat long words", "word")], 4)
        quads = [q for _, q in items]
        self.assertAlmostEqual(math.degrees(page_angle(quads)), 4, places=3)
        turned = [q[2:] + q[:2] for q in quads]
        self.assertAlmostEqual(math.degrees(page_angle(turned)), 4, places=3)

    def test_every_word_appears_exactly_once(self):
        rng = random.Random(3)
        items = []
        for i in range(400):
            items += words_at(rng.uniform(0, 800), rng.uniform(0, 1000), f"w{i}")
        text = render(items, 5)
        self.assertEqual(sorted(text.split()), sorted(f"w{i}" for i in range(400)))

    def test_empty(self):
        self.assertEqual(to_text([], []), "")
        self.assertEqual(to_text([" "], [[(0, 0), (1, 0), (1, 1), (0, 1)]]), "")
        self.assertEqual(to_text(["a"], [[(0, 0), (1, 0), (1, 1), (0, 1)]]), "a")


if __name__ == "__main__":
    unittest.main()
