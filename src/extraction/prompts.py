"""Prompt loading, shared by all three parts.

Prompt text lives in `prompts/*.yaml` so it can be revised without touching
Python. This module holds only the loading logic.

Every prompt has a `system` section and a `human` section, and nothing else.
The split matters: models weight system instructions as standing rules and the
human message as the request, so the page bindings and conventions belong in
`system` while the document text and the ask belong in `human`.

**Every exchange is assumed to be a single turn** - one call, one answer. There
is no `assistant` section because no prompt continues a conversation; each call
carries everything the model needs. Where a run makes several calls, as the
Part 3 supervisor does, each is independent and state is threaded through the
graph rather than through message history.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"

REQUIRED_SECTIONS = ("system", "human")


class PromptError(Exception):
    """Raised when a prompt file is missing or malformed."""


def load_prompt(name: str, prompt_dir: Path | str = PROMPT_DIR) -> ChatPromptTemplate:
    """Load `<name>.yaml` from the prompt directory as a ChatPromptTemplate."""
    path = Path(prompt_dir) / f"{name}.yaml"
    if not path.is_file():
        raise PromptError(f"Prompt file not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}

    missing = [section for section in REQUIRED_SECTIONS if not data.get(section)]
    if missing:
        raise PromptError(f"Prompt {path} is missing section(s): {', '.join(missing)}")

    return ChatPromptTemplate.from_messages(
        [("system", data["system"]), ("human", data["human"])]
    )
