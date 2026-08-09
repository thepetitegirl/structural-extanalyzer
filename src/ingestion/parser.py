"""PDF text extraction.

Uses pypdf, chosen by measurement rather than reputation: it is one of two
parsers that keep a table row intact on this document, and the faster of the
two. See notebooks/00_parser_comparison.ipynb for the evidence.

Note the limitation recorded there: pypdf returns table *content*, not table
*structure*. A figure arrives on the correct line but carries no marker naming
its column, so the prompt must bind it via the header text.
"""

from __future__ import annotations

from pathlib import Path

import pypdf

PAGE_MARKER = "--- page {page} ---"


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
        text = reader.pages[page - 1].extract_text() or ""
        sections.append(f"{PAGE_MARKER.format(page=page)}\n{text}")

    return "\n\n".join(sections)
