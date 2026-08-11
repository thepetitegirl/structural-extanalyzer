"""Unit tests for src.ingestion.parser."""

import pytest

from src.ingestion.parser import PAGE_MARKER, ParserError, extract_pages


def _build_pdf(page_texts: list[str]) -> bytes:
    """Assemble a minimal PDF containing one line of text per page.

    Written as raw PDF objects rather than through a builder library: the tests
    must not depend on the real document, which is gitignored, and a hand-built
    file keeps the fixture independent of any parser's own writer API.
    """
    objects: dict[int, bytes] = {}
    count = len(page_texts)
    kids = " ".join(f"{4 + i * 2} 0 R" for i in range(count))

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode()
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for index, text in enumerate(page_texts):
        page_id, content_id = 4 + index * 2, 5 + index * 2
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
        objects[content_id] = b"<< /Length %d >>\nstream\n%s\nendstream" % (
            len(stream),
            stream,
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + objects[number] + b"\nendobj\n"

    last = max(objects)
    xref_offset = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (last + 1)
    for number in range(1, last + 1):
        out += b"%010d 00000 n \n" % offsets.get(number, 0)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        last + 1,
        xref_offset,
    )
    return bytes(out)


@pytest.fixture
def sample_pdf(tmp_path):
    """Write a three-page PDF reading 'Page one/two/three content'."""
    path = tmp_path / "sample.pdf"
    path.write_bytes(
        _build_pdf(["Page one content", "Page two content", "Page three content"])
    )
    return path


def test_extracts_single_page(sample_pdf):
    """A one-page request returns that page's text."""
    assert "Page two" in extract_pages(sample_pdf, [2])


def test_pages_are_one_indexed(sample_pdf):
    """Page 1 is the first page, matching how the document numbers them."""
    assert "Page one" in extract_pages(sample_pdf, [1])


def test_extracts_multiple_pages_in_order(sample_pdf):
    """Requested pages appear in the order given."""
    text = extract_pages(sample_pdf, [1, 3])

    assert text.index("Page one") < text.index("Page three")
    assert "Page two" not in text


def test_marker_records_page_number(sample_pdf):
    """Each page is prefixed with a marker naming its number, for provenance."""
    text = extract_pages(sample_pdf, [2])

    assert PAGE_MARKER.format(page=2) in text


def test_missing_file_raises(tmp_path):
    """A missing PDF raises ParserError rather than a bare OSError."""
    with pytest.raises(ParserError, match="not found"):
        extract_pages(tmp_path / "absent.pdf", [1])


def test_out_of_range_page_raises(sample_pdf):
    """Requesting a page beyond the document names the valid range."""
    with pytest.raises(ParserError, match="3 pages"):
        extract_pages(sample_pdf, [99])


def test_empty_page_list_raises(sample_pdf):
    """An empty page list is a caller error, not a silent empty result."""
    with pytest.raises(ParserError, match="No pages"):
        extract_pages(sample_pdf, [])
