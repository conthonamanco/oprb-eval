"""CER, WER and bag-of-words P/R/F1, with the sample notebook's definitions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass


def edit_distance(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """Levenshtein distance (Myers/Hyyro bit-parallel, on Python's big integers)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    bits: dict[Hashable, int] = {}
    for i, symbol in enumerate(a):
        bits[symbol] = bits.get(symbol, 0) | (1 << i)
    mask = (1 << len(a)) - 1
    last = 1 << (len(a) - 1)
    pv, mv, score = mask, 0, len(a)
    for symbol in b:
        eq = bits.get(symbol, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & last:
            score += 1
        if mh & last:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
    return score


@dataclass(frozen=True)
class PageScore:
    cer: float
    wer: float
    precision: float
    recall: float
    f1: float
    char_errors: int
    word_errors: int
    ref_chars: int
    ref_words: int
    hyp_words: int


def score_page(reference: str, hypothesis: str) -> PageScore:
    """Both strings must already be cleaned. CER and WER can exceed 1.0."""
    if not reference:
        raise ValueError("Empty reference: CER/WER are undefined.")
    ref_words, hyp_words = reference.split(), hypothesis.split()
    char_errors = edit_distance(reference, hypothesis)
    word_errors = edit_distance(ref_words, hyp_words)
    matched = sum((Counter(ref_words) & Counter(hyp_words)).values())
    precision = matched / len(hyp_words) if hyp_words else 0.0
    recall = matched / len(ref_words)
    f1 = 2 * precision * recall / (precision + recall) if matched else 0.0
    return PageScore(
        cer=char_errors / len(reference),
        wer=word_errors / len(ref_words),
        precision=precision,
        recall=recall,
        f1=f1,
        char_errors=char_errors,
        word_errors=word_errors,
        ref_chars=len(reference),
        ref_words=len(ref_words),
        hyp_words=len(hyp_words),
    )
