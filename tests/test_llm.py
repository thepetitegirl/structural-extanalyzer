"""Unit tests for src.llm."""

import dataclasses

import pytest

from src.config import Config, ConfigError
from src.llm import UnsupportedProviderError, get_chat_model


def _config(**overrides) -> Config:
    """A config carrying the settings the factory reads."""
    base = Config(
        pdf_url="https://example.invalid/doc.pdf",
        pages=[5],
        provider="groq",
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_retries=6,
    )
    return dataclasses.replace(base, **overrides)


def test_settings_come_from_config(monkeypatch):
    """Model, temperature and retries are read from config, not hardcoded."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    model = get_chat_model(_config(model="llama-3.3-70b-versatile", max_retries=3))

    assert model.model_name == "llama-3.3-70b-versatile"
    assert model.max_retries == 3


def test_temperature_zero_keeps_extraction_deterministic(monkeypatch):
    """Temperature comes from config, where it is 0 for reproducibility."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    model = get_chat_model(_config())

    # ChatGroq clamps an exact 0 to a near-zero epsilon the API accepts.
    assert model.temperature <= 1e-6


def test_overrides_take_precedence(monkeypatch):
    """A caller can vary one setting without editing config."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    model = get_chat_model(_config(), model="llama-3.1-8b-instant", max_retries=1)

    assert model.max_retries == 1


def test_missing_api_key_fails_before_the_model_is_built(monkeypatch):
    """A missing credential is reported here, not as an auth error mid-run."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        get_chat_model(_config())


def test_unknown_provider_raises(monkeypatch):
    """A provider with no factory branch fails loudly, naming what is supported."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    with pytest.raises(UnsupportedProviderError, match="groq"):
        get_chat_model(_config(provider="ollama"))
