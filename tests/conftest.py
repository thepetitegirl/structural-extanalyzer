"""Shared test fixtures for the supervisor graph.

A graph run makes up to six model calls, so the single-result stub used in
Parts 1 and 2 is not enough. `ScriptedModel` replays a queued sequence and
records every prompt it was given - which is also how a routing decision is
asserted without a live model.
"""

import pytest
from langchain_core.runnables import Runnable

from src.config import Config


class ScriptedModel(Runnable):
    """Replays queued results in order, recording every prompt received."""

    def __init__(self, results):
        self.results = list(results)
        self.received = []
        self.schemas = []
        self.methods = []

    def with_structured_output(self, schema, **kwargs):
        """Record the requested schema and method, and return self."""
        self.schemas.append(schema)
        self.methods.append(kwargs.get("method"))
        return self

    def invoke(self, input, config=None, **kwargs):
        """Record the rendered prompt and return the next scripted result."""
        self.received.append(input)
        if not self.results:
            raise AssertionError(
                f"ScriptedModel ran out of results after {len(self.received)} calls. "
                "The graph made more calls than the test scripted."
            )
        return self.results.pop(0)

    @property
    def prompts(self) -> list[str]:
        """Every prompt rendered as text, for asserting what a node was shown."""
        return [
            received.to_string() if hasattr(received, "to_string") else str(received)
            for received in self.received
        ]


@pytest.fixture
def config():
    """A config with the real page bindings, but no filesystem dependency."""
    return Config(
        pdf_url="https://example.invalid/doc.pdf",
        pages=[5, 6, 8, 20],
        provider="groq",
        model="llama-3.1-8b-instant",
        agent_pages={"revenue": [9, 13, 15], "expenditure": [16, 18, 20]},
        max_turns=4,
        expenditure_hints=["fund", "spend", "expenditure", "top-up", "transfer"],
    )


@pytest.fixture
def fake_pages(monkeypatch):
    """Replace page extraction with a recorder.

    Returns the list that records each (path, pages) call, so a test can assert
    which pages an agent actually asked for.
    """
    calls = []

    def _extract(pdf_path, pages):
        calls.append((str(pdf_path), list(pages)))
        return "\n\n".join(f"--- page {page} ---\ntext of page {page}" for page in pages)

    monkeypatch.setattr("src.agents.base.extract_pages", _extract)
    return calls
