"""Fetching the source document.

Only the URL is configured. The file is saved under `data/` using the filename
from the URL, and reused on later runs - so a fresh clone needs nothing but the
config, and no run downloads the same document twice.
"""

from __future__ import annotations

import tomllib
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlparse

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _user_agent() -> str:
    """Identify this client to the server it fetches from.

    Both name and version come from pyproject.toml, read at run time, so
    neither is written out here and neither can drift from the project's own
    metadata.
    """
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    return f"{project['name']}/{project['version']}"


class DownloadError(Exception):
    """Raised when the source document cannot be obtained."""


def _fetch(url: str) -> bytes:
    """Retrieve a URL and return its body."""
    request = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def local_path(url: str, data_dir: Path | str = DATA_DIR) -> Path:
    """Where a URL's document is cached, taken from the URL's filename."""
    name = Path(unquote(urlparse(url).path)).name
    if not name:
        raise DownloadError(f"Cannot determine a filename from {url}.")
    return Path(data_dir) / name


def ensure_pdf(
    url: str,
    fetch: Callable[[str], bytes] = _fetch,
    data_dir: Path | str = DATA_DIR,
) -> Path:
    """Return a local path to the document, downloading it if not already there.

    The cached file is used as-is when present, so the document is fetched at
    most once however many times the pipeline runs.
    """
    if not url:
        raise DownloadError("No pdf_url is configured. Add one to config.yml.")

    path = local_path(url, data_dir)
    if path.is_file():
        return path

    try:
        payload = fetch(url)
    except Exception as exc:
        raise DownloadError(f"Could not download {url}: {exc}") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)

    return path
