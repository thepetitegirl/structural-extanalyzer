"""Unit tests for src.ingestion.download.

No test reaches the network: the fetch function is stubbed.
"""

import pytest

from src.ingestion.download import DownloadError, ensure_pdf


def test_returns_existing_file_without_fetching(tmp_path):
    """An already-downloaded file is used as-is."""
    target = tmp_path / "doc.pdf"
    target.write_bytes(b"%PDF-1.4 existing")

    called = False

    def fetch(url):
        nonlocal called
        called = True
        return b"downloaded"

    result = ensure_pdf("https://example.invalid/doc.pdf", target, fetch=fetch)

    assert result == target
    assert not called, "should not re-download a file that is already present"


def test_downloads_when_absent(tmp_path):
    """A missing file is fetched and written to the target path."""
    target = tmp_path / "nested" / "doc.pdf"

    result = ensure_pdf(
        "https://example.invalid/doc.pdf", target, fetch=lambda url: b"%PDF-1.4 new"
    )

    assert result.read_bytes() == b"%PDF-1.4 new"


def test_creates_parent_directory(tmp_path):
    """The target directory is created if it does not exist."""
    target = tmp_path / "data" / "doc.pdf"

    ensure_pdf("https://example.invalid/doc.pdf", target, fetch=lambda url: b"%PDF-x")

    assert target.parent.is_dir()


def test_rejects_a_response_that_is_not_a_pdf(tmp_path):
    """A response without a PDF header raises rather than being saved.

    A redirect to an HTML error page would otherwise be written to disk and
    fail later as an unreadable PDF.
    """
    target = tmp_path / "doc.pdf"

    with pytest.raises(DownloadError, match="not a PDF"):
        ensure_pdf(
            "https://example.invalid/doc.pdf",
            target,
            fetch=lambda url: b"<!DOCTYPE html><html>404</html>",
        )

    assert not target.exists(), "a bad response must not be left on disk"


def test_missing_file_and_no_url_raises(tmp_path):
    """Without a URL there is nothing to fall back on, so the error says so."""
    with pytest.raises(DownloadError, match="no pdf_url"):
        ensure_pdf(None, tmp_path / "absent.pdf", fetch=lambda url: b"%PDF")


def test_fetch_failure_is_reported_with_the_url(tmp_path):
    """A network failure names the URL that could not be reached."""

    def failing_fetch(url):
        raise OSError("connection refused")

    with pytest.raises(DownloadError, match="example.invalid"):
        ensure_pdf(
            "https://example.invalid/doc.pdf",
            tmp_path / "doc.pdf",
            fetch=failing_fetch,
        )
