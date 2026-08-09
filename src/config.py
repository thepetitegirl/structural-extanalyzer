"""Configuration loading.

Non-secret settings live in `config.yml` and are committed. Credentials live in
`.env` and are read from the environment, never from the YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path("config.yml")

REQUIRED_KEYS = ("pdf_path", "pages", "provider", "model")

# Providers that authenticate with an API key, and the variable holding it.
# Ollama is absent because it runs locally and needs no credential.
API_KEY_VARS = {
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
}


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class Config:
    """Settings for one extraction run."""

    pdf_path: str
    pages: list[int]
    provider: str
    model: str
    temperature: float = 0.0
    # Field name -> the page(s) that field must be read from. Injected into the
    # prompt so a page is defined here and nowhere else.
    field_pages: dict[str, int | list[int]] = field(default_factory=dict)
    # Part 2: pages holding the dates to normalise and classify.
    date_pages: list[int] = field(default_factory=list)

    def page_for(self, field_name: str) -> str:
        """Render a field's bound page(s) for use in a prompt."""
        pages = self.field_pages.get(field_name)
        if pages is None:
            raise ConfigError(f"No page configured for field {field_name!r}.")
        if isinstance(pages, int):
            return str(pages)
        return " AND ".join(str(page) for page in pages)

    def api_key(self) -> str | None:
        """Return the provider's API key, or None if it needs no credential.

        Raises ConfigError when a key is required but absent, so the failure
        surfaces here rather than as an opaque auth error mid-request.
        """
        var = API_KEY_VARS.get(self.provider)
        if var is None:
            return None

        key = os.environ.get(var)
        if not key:
            raise ConfigError(
                f"{var} is not set. Copy .env.example to .env and add your key."
            )
        return key


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Load settings from a YAML file, with .env loaded into the environment."""
    load_dotenv()

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ConfigError(f"Config {path} is missing required key(s): {', '.join(missing)}")

    pages = list(data["pages"])
    field_pages = data.get("field_pages") or {}

    # A field bound to a page that is never extracted would silently produce a
    # wrong answer, so catch the mismatch here rather than at inference time.
    for name, cited in field_pages.items():
        cited_pages = [cited] if isinstance(cited, int) else list(cited)
        missing = [page for page in cited_pages if page not in pages]
        if missing:
            raise ConfigError(
                f"Field {name!r} cites page(s) {missing}, which are not in "
                f"pages {pages}. Add them to `pages` or correct `field_pages`."
            )

    return Config(
        pdf_path=data["pdf_path"],
        pages=pages,
        provider=data["provider"],
        model=data["model"],
        temperature=float(data.get("temperature", 0.0)),
        field_pages=field_pages,
        date_pages=list(data.get("date_pages", [])),
    )
