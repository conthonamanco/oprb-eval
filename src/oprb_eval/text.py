"""Text normalisation for the two scoring protocols."""

from __future__ import annotations

import html
import re
import unicodedata

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
    return _MD_BOLD.sub(r"\1", text)


def reference_text(tokens: list[str], protocol: str) -> str:
    if protocol == "notebook":
        return clean_text(" ".join(tokens))
    if protocol == "normalized":
        return clean_text(" ".join(t for t in tokens if not PSEUDO_TOKEN.match(t)))
    raise ValueError(f"Unknown protocol: {protocol}")


def prediction_text(prediction: str, protocol: str) -> str:
    if protocol == "notebook":
        return clean_text(prediction)
    if protocol == "normalized":
        return clean_text(strip_markup(prediction))
    raise ValueError(f"Unknown protocol: {protocol}")
