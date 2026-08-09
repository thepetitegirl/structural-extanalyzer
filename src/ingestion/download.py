"""Fetching the source document.

`data/` is gitignored, so a fresh clone has no PDF. The URL lives in config.yml
and the file is downloaded once, then reused - a clone needs only the config,
not the document.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Callable

# A PDF always starts with this. Checking it catches the common failure where a
# redirect returns an HTML error page that would otherwise be saved as a .pdf
# and fail much later as an unreadable document.
PDF_MAGIC = b"%PDF"

USER_AGENT = "structural-extanalyzer"


class DownloadError(Exception):
    """Raised when the source document cannot be obtained."""


def _fetch(url: str) -> bytes:
    """Retrieve a URL and return its body."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def ensure_pdf(
    url: str | None,
    path: Path | str,
    fetch: Callable[[str], bytes] = _fetch,
) -> Path:
    """Return a local path to the PDF, downloading it if not already present."""
    path = Path(path)

    if path.is_file():
        return path

    if not url:
        raise DownloadError(
            f"{path} does not exist and no pdf_url is configured. "
            "Add pdf_url to config.yml or place the file at that path."
        )

    try:
        payload = fetch(url)
    except Exception as exc:
        raise DownloadError(f"Could not download {url}: {exc}") from exc

    if not payload.startswith(PDF_MAGIC):
        raise DownloadError(
            f"Response from {url} is not a PDF "
            f"(starts with {payload[:20]!r}). The link may have moved."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)

    return path
