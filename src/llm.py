"""Chat model factory, shared by all three parts.

Groq is used as the model provider: it does not require downloading models onto
local hardware, and it supports Pydantic schemas for structured output, where
`langchain-huggingface` raises NotImplementedError.

Every setting comes from configuration - provider, model, temperature and retry
count from `config.yml`, the API key from the environment via `.env`. Nothing
about how the model behaves is written out here, so changing it never means
editing Python.

Kept as a factory rather than an inline constructor so the model is built in one
place and all three parts share it.
"""

from __future__ import annotations

from typing import Any

SUPPORTED_PROVIDERS = ("groq",)


class UnsupportedProviderError(Exception):
    """Raised when the configured provider has no factory branch."""


def get_chat_model(config, **overrides: Any):
    """Build the chat model described by `config`.

    `config.api_key()` is called first, so a missing credential fails here with
    a message naming the variable rather than as an opaque auth error partway
    through a run.

    `overrides` let a caller vary one setting without editing config - the model
    comparison notebook uses it to try a different model - and are unused by the
    pipeline itself.
    """
    config.api_key()

    settings: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "max_retries": config.max_retries,
    }
    settings.update(overrides)

    if config.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(**settings)

    raise UnsupportedProviderError(
        f"Unknown provider {config.provider!r}. "
        f"Supported: {', '.join(SUPPORTED_PROVIDERS)}."
    )
