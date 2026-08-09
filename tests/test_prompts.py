"""Unit tests for src.extraction.prompts."""

import pytest
import yaml
from langchain_core.prompts import ChatPromptTemplate

from src.extraction.prompts import PromptError, load_prompt


@pytest.fixture
def prompt_dir(tmp_path):
    """Write a minimal prompt file and return its directory."""
    (tmp_path / "demo.yaml").write_text(
        yaml.safe_dump(
            {
                "system": "You extract figures for {target_year}.",
                "human": "{page_text}\n\nExtract the fields.",
            }
        )
    )
    return tmp_path


def test_returns_chat_prompt_template(prompt_dir):
    """A prompt file loads as a ChatPromptTemplate."""
    assert isinstance(load_prompt("demo", prompt_dir), ChatPromptTemplate)


def test_exposes_declared_variables(prompt_dir):
    """Template variables in the YAML are exposed for formatting."""
    prompt = load_prompt("demo", prompt_dir)

    assert set(prompt.input_variables) == {"target_year", "page_text"}


def test_formats_with_supplied_values(prompt_dir):
    """Formatting substitutes values into both system and human messages."""
    messages = load_prompt("demo", prompt_dir).format_messages(
        target_year="FY2023", page_text="--- page 5 ---"
    )

    assert "FY2023" in messages[0].content
    assert "--- page 5 ---" in messages[1].content


def test_missing_file_raises(tmp_path):
    """A missing prompt file raises PromptError naming the prompt."""
    with pytest.raises(PromptError, match="absent"):
        load_prompt("absent", tmp_path)


def test_missing_required_section_raises(tmp_path):
    """A prompt file without a human section is rejected."""
    (tmp_path / "partial.yaml").write_text(yaml.safe_dump({"system": "only system"}))

    with pytest.raises(PromptError, match="human"):
        load_prompt("partial", tmp_path)


def test_real_extraction_prompt_loads():
    """The committed extraction prompt parses and declares its variables."""
    prompt = load_prompt("extraction")

    assert set(prompt.input_variables) == {
        "target_year",
        "page_text",
        "page_cit",
        "page_cit_yoy",
        "page_top_ups",
        "page_taxes",
        "page_fiscal",
    }


def test_page_numbers_are_not_hardcoded():
    """Field pages come from config, so the prompt text must not name them."""
    prompt = load_prompt("extraction")
    human = prompt.messages[1].prompt.template

    assert "PAGE 5 ONLY" not in human
    assert "PAGE 20 ONLY" not in human
