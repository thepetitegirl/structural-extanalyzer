"""Unit tests for src.config."""

import pytest
import yaml

from src.config import ConfigError, load_config


@pytest.fixture
def config_file(tmp_path):
    """Write a minimal valid config.yml and return its path."""
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "pdf_path": "data/example.pdf",
                "pages": [5, 6, 8, 20],
                "provider": "google",
                "model": "gemini-2.0-flash",
            }
        )
    )
    return path


def test_loads_values_from_yaml(config_file):
    """Values in config.yml are exposed as attributes."""
    config = load_config(config_file)

    assert config.pdf_path == "data/example.pdf"
    assert config.pages == [5, 6, 8, 20]
    assert config.provider == "google"
    assert config.model == "gemini-2.0-flash"


def test_missing_file_raises(tmp_path):
    """A missing config file raises ConfigError, not FileNotFoundError."""
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yml")


def test_missing_required_key_raises(tmp_path):
    """A config lacking a required key names the key it is missing."""
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump({"pdf_path": "data/example.pdf"}))

    with pytest.raises(ConfigError, match="pages"):
        load_config(path)


def test_no_secrets_in_config(config_file):
    """Config carries no credentials; keys come from the environment only."""
    config = load_config(config_file)

    assert not any("key" in field.lower() for field in config.__dataclass_fields__)


def test_api_key_read_from_environment(config_file, monkeypatch):
    """api_key() reads the provider's key from the environment."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-value")
    config = load_config(config_file)

    assert config.api_key() == "test-key-value"


def test_missing_api_key_raises(config_file, monkeypatch):
    """A missing API key raises rather than passing None downstream."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config = load_config(config_file)

    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        config.api_key()


def test_ollama_needs_no_api_key(tmp_path, monkeypatch):
    """Ollama runs locally, so api_key() returns None instead of raising."""
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "pdf_path": "data/example.pdf",
                "pages": [5],
                "provider": "ollama",
                "model": "qwen3:8b",
            }
        )
    )
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    assert load_config(path).api_key() is None
