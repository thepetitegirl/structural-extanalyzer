"""Unit tests for src.ingestion.download.

No test reaches the network: the fetch function is stubbed.
"""

import pytest

from src.ingestion.download import DownloadError, ensure_pdf, local_path

URL = "https://example.invalid/reports/fy2024_analysis.pdf"


def test_local_path_takes_the_filename_from_the_url(tmp_path):
    """The cache location is derived from the URL, not configured separately."""
    assert local_path(URL, tmp_path).name == "fy2024_analysis.pdf"


def test_local_path_decodes_percent_escapes(tmp_path):
    """A URL-encoded filename is written out readably."""
    encoded = "https://example.invalid/my%20report.pdf"

    assert local_path(encoded, tmp_path).name == "my report.pdf"


def test_url_without_a_filename_raises(tmp_path):
    """A URL with no filename gives nothing to save as."""
    with pytest.raises(DownloadError, match="filename"):
        local_path("https://example.invalid/", tmp_path)


def test_downloads_when_absent(tmp_path):
    """A missing file is fetched and written to the derived path."""
    result = ensure_pdf(URL, fetch=lambda url: b"%PDF-1.4 new", data_dir=tmp_path)

    assert result.read_bytes() == b"%PDF-1.4 new"
    assert result.name == "fy2024_analysis.pdf"


def test_existing_file_is_reused_without_fetching(tmp_path):
    """A cached document is used as-is, so it is downloaded at most once."""
    target = tmp_path / "fy2024_analysis.pdf"
    target.write_bytes(b"%PDF-1.4 existing")

    called = False

    def fetch(url):
        nonlocal called
        called = True
        return b"downloaded"

    result = ensure_pdf(URL, fetch=fetch, data_dir=tmp_path)

    assert result.read_bytes() == b"%PDF-1.4 existing"
    assert not called, "should not re-download a document already cached"


def test_creates_the_data_directory(tmp_path):
    """data/ is gitignored, so a fresh clone will not have it."""
    target_dir = tmp_path / "data"

    ensure_pdf(URL, fetch=lambda url: b"%PDF-x", data_dir=target_dir)

    assert target_dir.is_dir()


def test_missing_url_raises(tmp_path):
    """Without a URL there is nothing to fetch, so the error says so."""
    with pytest.raises(DownloadError, match="pdf_url"):
        ensure_pdf("", fetch=lambda url: b"%PDF", data_dir=tmp_path)


def test_fetch_failure_is_reported_with_the_url(tmp_path):
    """A network failure names the URL that could not be reached."""

    def failing_fetch(url):
        raise OSError("connection refused")

    with pytest.raises(DownloadError, match="example.invalid"):
        ensure_pdf(URL, fetch=failing_fetch, data_dir=tmp_path)
