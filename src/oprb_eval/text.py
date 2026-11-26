"""Text normalisation for the two scoring protocols."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Sequence

from . import reading_order
from .data import GTToken

# Layout placeholders that DocBank stores as tokens, e.g. ##LTLine##, ##LTFigure##.
PSEUDO_TOKEN = re.compile(r"^##LT\w+##$")

# Only known markup tags are removed, so maths such as "a < b" survives.
_TAGS = (
    r"table|thead|tbody|tfoot|tr|td|th|caption|br|hr|p|div|span|sup|sub|b|i|u|em|strong|img|"
    r"h[1-6]|ul|ol|li|a|code|doctag|text|section_header_level_\d|page_header|page_footer|"
    r"footnote|formula|picture|otsl|list_item|ordered_list|unordered_list|loc_\d+"
)
_HTML_TAG = re.compile(rf"</?(?:{_TAGS})(?:\s[^<>]*)?/?>", re.IGNORECASE)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_FENCE = re.compile(r"^[ \t]*```\w*[ \t]*$", re.MULTILINE)
_MD_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)
_MD_TABLE_RULE = re.compile(r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t|:-]*$", re.MULTILINE)
_MD_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")

# LaTeX, which the VLMs write for formulas (the LightOnOCR card documents Markdown with LaTeX
# maths). The ground truth holds the characters printed on the page, so a command that prints
# one symbol becomes that symbol, other commands are dropped and their arguments kept.
_LATEX_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠", "approx": "≈",
    "sim": "∼", "simeq": "≃", "equiv": "≡", "propto": "∝", "pm": "±", "mp": "∓",
    "times": "×", "cdot": "·", "div": "÷", "infty": "∞", "partial": "∂", "nabla": "∇",
    "sum": "∑", "prod": "∏", "int": "∫", "oint": "∮", "sqrt": "√", "in": "∈", "notin": "∉",
    "subset": "⊂", "subseteq": "⊆", "cup": "∪", "cap": "∩", "forall": "∀", "exists": "∃",
    "rightarrow": "→", "to": "→", "leftarrow": "←", "Rightarrow": "⇒", "leftrightarrow": "↔",
    "ll": "≪", "gg": "≫", "prime": "′", "circ": "∘", "ldots": "…", "cdots": "⋯", "dots": "…",
    "langle": "⟨", "rangle": "⟩", "hbar": "ħ", "ell": "ℓ", "dagger": "†", "star": "⋆",
}
_LATEX_ENV = re.compile(r"\\(?:begin|end)\{[^{}]*\}")
_LATEX_DELIMITER = re.compile(r"\$\$?|\\[()\[\]]")
_LATEX_COMMAND = re.compile(r"\\([A-Za-z]+)")
_LATEX_SPACE = re.compile(r"\\\\|\\[,;:! ]|(?<=\s)&(?=\s)")
_LATEX_SCRIPT = re.compile(r"[{}]|[_^](?=[{\w])")


def strip_latex(text: str) -> str:
    """`$L_{k'} \\geq L_k$` -> `Lk' ≥ Lk`.

    Text with no backslash and no `$` is returned as it is, so OCR output that never uses
    LaTeX keeps its braces and underscores.
    """
    if "\\" not in text and "$" not in text:
        return text
    text = _LATEX_ENV.sub(" ", text)
    text = _LATEX_DELIMITER.sub("", text)
    text = _LATEX_SPACE.sub(" ", text)
    text = _LATEX_COMMAND.sub(lambda m: _LATEX_SYMBOLS.get(m.group(1), " "), text)
    return _LATEX_SCRIPT.sub("", text)


def clean_text(text: str) -> str:
    """NFKC, collapse whitespace, casefold. Identical to the sample notebook."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def strip_markup(text: str) -> str:
    """Remove output-format markup (HTML, DocTags, Markdown) but keep the recognised text."""
    text = _HTML_TAG.sub(" ", text)
    text = html.unescape(text)
    text = _MD_IMAGE.sub(" ", text)
    text = _MD_FENCE.sub("", text)
    text = _MD_TABLE_RULE.sub("", text)
    text = _MD_HEADING.sub("", text)
    return strip_latex(_MD_BOLD.sub(r"\1", text))


def _boxes(gt: Sequence[GTToken]) -> tuple[list[str], list[list[tuple[float, float]]]]:
    kept = [t for t in gt if not PSEUDO_TOKEN.match(t.text)]
    quads = [[(t.x0, t.y0), (t.x1, t.y0), (t.x1, t.y1), (t.x0, t.y1)] for t in kept]
    return [t.text for t in kept], quads


def column_count(gt: Sequence[GTToken]) -> int:
    """1 or 2 text columns on the page, from the GT boxes (0 if it has no text)."""
    return reading_order.column_count(*_boxes(gt))


def reference_text(gt: Sequence[GTToken | str], protocol: str) -> str:
    """Reference transcript of one page under `protocol`.

    notebook   GT tokens in file order, as the sample notebook joins them.
    normalized Same order, without the ##LT…## layout placeholders.
    columns    Same tokens as normalized, re-read column by column from their boxes. File
               order alternates between the columns of a two-column page, so a model that
               reads one column and then the other gets a large CER there although every
               word is right. This uses the reading-order code of the PP-OCR and docTR
               adapters, which may favour those two models slightly.
    """
    tokens = [t if isinstance(t, str) else t.text for t in gt]
    if protocol == "notebook":
        return clean_text(" ".join(tokens))
    if protocol == "normalized":
        return clean_text(" ".join(t for t in tokens if not PSEUDO_TOKEN.match(t)))
    if protocol == "columns":
        if any(isinstance(t, str) for t in gt):
            raise TypeError("The columns protocol needs GT boxes: pass read_gt() rows.")
        return clean_text(reading_order.to_text(*_boxes(gt)))
    raise ValueError(f"Unknown protocol: {protocol}")


def prediction_text(prediction: str, protocol: str) -> str:
    if protocol == "notebook":
        return clean_text(prediction)
    if protocol in ("normalized", "columns"):
        return clean_text(strip_markup(prediction))
    raise ValueError(f"Unknown protocol: {protocol}")
