"""Saving each part's answer to disk, shared by all three.

The printed output scrolls away with the terminal, and re-running to see it
again costs API budget for a result already obtained. Each part therefore
writes its answer to `results/`, which is committed: the answers can be read
from the repository without running anything.

One helper rather than three, because the three parts produce different shapes
- a Pydantic model, a list of dicts, a dataclass trace - and only the
serialisation differs, not the writing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Relative to the repository root, so a clone writes inside itself rather than
# to wherever an absolute path happened to point on one machine.
RESULTS_DIR = Path("results")


def _plain(value: Any) -> Any:
    """Convert Pydantic models and dataclasses into JSON-serialisable data.

    Applied recursively: a trace is a dataclass holding lists of models, so
    converting only the top level would leave models nested inside it.
    """
    if isinstance(value, BaseModel):
        return value.model_dump()

    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))

    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]

    return value


def save_json(payload: Any, path: Path | str) -> Path:
    """Write `payload` to `path` as JSON, returning where it went.

    Any missing parent directories are created, so a fresh clone with no
    `results/` still saves. An existing file is replaced: a re-run supersedes
    the previous answer rather than accumulating alongside it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2) + "\n")

    return path
