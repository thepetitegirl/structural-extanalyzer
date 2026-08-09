"""Prompt loading.

Prompt text lives in `prompts/*.yaml` so it can be revised without touching
Python. This module holds only the loading logic.
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
