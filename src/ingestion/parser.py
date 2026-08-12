"""PDF text extraction.

Uses pypdf, chosen by measurement rather than reputation: it is one of two
parsers that keep a table row intact on this document, and the faster of the
two. See notebooks/00_parser_comparison.ipynb for the evidence.

Note the limitation recorded there: pypdf returns table *content*, not table
*structure*. A figure arrives on the correct line but carries no marker naming
its column, so the prompt must bind it via the header text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pypdf

PAGE_MARKER = "--- page {page} ---"

# Reads back what PAGE_MARKER writes, so the two cannot drift apart.
_MARKER_PATTERN = re.compile(r"--- page (\d+) ---")


class ParserError(Exception):
    """Raised when a document cannot be read or a page request is invalid."""


def extract_pages(pdf_path: Path | str, pages: list[int]) -> str:
    """Return the text of the given 1-indexed pages, in the order requested.

    Pages are 1-indexed to match the page citations in the requirements, and
    each is prefixed with a marker so extracted values can be traced back to
    their source page.
    """
    if not pages:
        raise ParserError("No pages requested; pass at least one page number.")

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise ParserError(f"PDF not found: {pdf_path}")

    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as exc:  # pypdf raises a variety of parse errors
        raise ParserError(f"Could not read {pdf_path}: {exc}") from exc

    total = len(reader.pages)
    out_of_range = [page for page in pages if not 1 <= page <= total]
    if out_of_range:
        raise ParserError(
            f"Page(s) {out_of_range} out of range: {pdf_path.name} has {total} pages."
        )

    sections = []
    for page in pages:
        text = reader.pages[page - 1].extract_text() or "" #pypdf is 0 indexed
        sections.append(f"{PAGE_MARKER.format(page=page)}\n{text}")

    return "\n\n".join(sections)


def _collapse(text: str) -> str:
    """Collapse whitespace, so pypdf's irregular spacing does not defeat a match."""
    return re.sub(r"\s+", " ", text).strip().lower()


def page_of_quote(quote: str, page_text: str, claimed: int) -> int:
    """The page whose section contains `quote`, or `claimed` if none does.

    A model can read the right sentence and record the wrong page: related
    figures appear on several pages, and a value that is correct but cited to
    the wrong page cannot be verified by a reader following the citation.

    The quote is required to be verbatim, so which page it came from is a
    lookup rather than a judgement - the same split as Part 2, where the model
    classifies a date and a tool checks the arithmetic.

    Falls back to the claimed page when the quote is paraphrased and matches
    nothing. Correcting it is not possible, and dropping the figure would lose a
    value that may be right; `check_traceability` still reports it.
    """
    if not quote.strip():
        return claimed

    sections = _MARKER_PATTERN.split(page_text)
    wanted = _collapse(quote)

    # split() yields [before, page, body, page, body, ...] - pair them up.
    for page, body in zip(sections[1::2], sections[2::2], strict=True):
        if wanted in _collapse(body):
            return int(page)

    return claimed
