"""Chat model factory.

Groq only. Other providers were evaluated and rejected: gpt-oss cannot call
tools at either size, local models via Ollama fold units into values and are
slow, and langchain-huggingface does not support Pydantic schemas for function
calling. See notebooks/02_extraction.ipynb for the measurements.

Kept as a factory rather than an inline constructor so the model is built in one
place and Parts 2 and 3 share it.
"""

from __future__ import annotations

from typing import Any

SUPPORTED_PROVIDERS = ("groq",)


class UnsupportedProviderError(Exception):
    """Raised when the configured provider has no factory branch."""


def get_chat_model(provider: str, model: str, **kwargs: Any):
    """Return a chat model for the given provider.

    Temperature defaults to 0: extraction should be reproducible, and there is
    no value in sampling variety when copying figures out of a document.
    """
    kwargs.setdefault("temperature", 0)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, **kwargs)

    raise UnsupportedProviderError(
        f"Unknown provider {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}."
    )
