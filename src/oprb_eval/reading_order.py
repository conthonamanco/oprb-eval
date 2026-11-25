"""Turn detected text boxes into running text.

Test pages are rotated by up to 5 degrees, so boxes are deskewed first. Words are then
joined into line fragments, one- or two-column layout is detected, and fragments are
read top to bottom within each column.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

Point = tuple[float, float]

JOIN_GAP = 1.2  # largest gap inside a line, in line heights
MIN_OVERLAP = 0.5  # vertical overlap that puts two boxes on the same line
GUTTER = 0.012  # width of the empty strip between columns, relative to the text width


@dataclass
class Box:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def _long_edge(quad: Sequence[Point]) -> tuple[float, float, float]:
    """(angle, length, aspect ratio) of the longer side of a quadrilateral."""
    a = (quad[1][0] - quad[0][0], quad[1][1] - quad[0][1])
    b = (quad[2][0] - quad[1][0], quad[2][1] - quad[1][1])
    long, short = sorted((a, b), key=lambda v: math.hypot(*v), reverse=True)
    length = math.hypot(*long)
    angle = math.atan2(long[1], long[0])
    angle -= math.pi * round(angle / math.pi)  # direction of travel does not matter
    return angle, length, length / max(math.hypot(*short), 1e-9)


def page_angle(quads: Sequence[Sequence[Point]]) -> float:
    """Length-weighted median angle of clearly elongated, roughly horizontal boxes."""
    edges = [_long_edge(q) for q in quads]
    wide = sorted((a, n) for a, n, ratio in edges if ratio >= 2 and abs(a) < math.pi / 6)
    half, total = sum(n for _, n in wide) / 2, 0.0
    for angle, length in wide:
        total += length
        if total >= half:
            return angle
    return 0.0


def _deskew(texts: Sequence[str], quads: Sequence[Sequence[Point]]) -> list[Box]:
    angle = page_angle(quads)
    cos, sin = math.cos(-angle), math.sin(-angle)
    boxes = []
    for text, quad in zip(texts, quads, strict=True):
        xs = [x * cos - y * sin for x, y in quad]
        ys = [x * sin + y * cos for x, y in quad]
        boxes.append(Box(text, min(xs), min(ys), max(xs), max(ys)))
    return boxes


def _same_line(a: Box, b: Box) -> bool:
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    return overlap >= MIN_OVERLAP * min(a.height, b.height)


def _join_fragments(boxes: list[Box]) -> list[Box]:
    """Chain each box to its nearest right-hand neighbour on the same line."""
    size = statistics.median(b.height for b in boxes) or 1.0
    bins: dict[int, list[int]] = {}
    for i, box in enumerate(boxes):
        for k in range(int(box.y0 // size), int(box.y1 // size) + 1):
            bins.setdefault(k, []).append(i)

    links = []
    for i, box in enumerate(boxes):
        near = {j for k in range(int(box.y0 // size), int(box.y1 // size) + 1) for j in bins[k]}
        for j in near:
            other = boxes[j]
            gap = other.x0 - box.x1
            limit = JOIN_GAP * min(box.height, other.height)
            if other.x0 > box.cx and gap <= limit and _same_line(box, other):
                links.append((gap, i, j))
    following: dict[int, int] = {}
    preceded: set[int] = set()
    for _, i, j in sorted(links, key=lambda link: (abs(link[0]), link[1], link[2])):
        if i not in following and j not in preceded:
            following[i] = j
            preceded.add(j)

    fragments = []
    for i in range(len(boxes)):
        if i in preceded:
            continue
        chain = [boxes[i]]
        while i in following:
            i = following[i]
            chain.append(boxes[i])
        fragments.append(
            Box(
                " ".join(b.text for b in chain),
                min(b.x0 for b in chain),
                min(b.y0 for b in chain),
                max(b.x1 for b in chain),
                max(b.y1 for b in chain),
            )
        )
    return fragments


def _find_gutter(fragments: list[Box]) -> float | None:
    """Centre of the gutter of a two-column page, or None for a single column."""
    left, right = min(f.x0 for f in fragments), max(f.x1 for f in fragments)
    width = right - left
    half = GUTTER * width / 2
    best = None
    for step in range(81):  # the middle 40% of the text width
        x = left + width * (0.30 + 0.40 * step / 80)
        crossing = sum(f.x1 - f.x0 for f in fragments if f.x0 < x + half and f.x1 > x - half)
        candidate = (crossing, abs(x - (left + right) / 2), x)
        best = min(best, candidate) if best else candidate
    crossing, _, x = best
    total = sum(f.x1 - f.x0 for f in fragments)
    on_left = sum(f.x1 - f.x0 for f in fragments if f.x1 <= x - half)
    on_right = sum(f.x1 - f.x0 for f in fragments if f.x0 >= x + half)
    two_columns = crossing <= 0.4 * total and min(on_left, on_right) >= 0.15 * total
    return x if two_columns else None


def _rows(fragments: list[Box]) -> list[str]:
    """Read one column: group fragments into rows, then left to right inside a row."""
    rows: list[list[Box]] = []
    for fragment in sorted(fragments, key=lambda f: f.cy):
        if rows and _same_line(rows[-1][0], fragment):
            rows[-1].append(fragment)
        else:
            rows.append([fragment])
    return [" ".join(f.text for f in sorted(row, key=lambda f: f.x0)) for row in rows]


def to_text(texts: Sequence[str], quads: Sequence[Sequence[Point]]) -> str:
    """`quads[i]` holds the four corners of `texts[i]`, in pixels and in cyclic order."""
    kept = [(t.strip(), q) for t, q in zip(texts, quads, strict=True) if t.strip()]
    if not kept:
        return ""
    fragments = _join_fragments(_deskew([t for t, _ in kept], [q for _, q in kept]))
    gutter = _find_gutter(fragments)
    if gutter is None:
        return "\n".join(_rows(fragments))

    # Lines that cross the gutter (title, wide figures) cut the page into bands. Within a
    # band the left column is read before the right one.
    spanning = sorted((f for f in fragments if f.x0 < gutter < f.x1), key=lambda f: f.cy)
    lines: list[str] = []
    rest = [f for f in fragments if not f.x0 < gutter < f.x1]
    previous = -math.inf
    for limit, cut in [*((f.cy, f) for f in spanning), (math.inf, None)]:
        band = [f for f in rest if previous <= f.cy < limit]
        lines += _rows([f for f in band if f.cx < gutter])
        lines += _rows([f for f in band if f.cx >= gutter])
        if cut is not None:
            lines.append(cut.text)
        previous = limit
    return "\n".join(lines)
