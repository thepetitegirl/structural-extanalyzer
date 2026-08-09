"""Unit tests for src.extraction.date_reasoning.

The model is stubbed. These check the verification layer - that a wrong
classification is caught rather than accepted - which is the point of the
module.
"""

import pytest
from langchain_core.runnables import Runnable

from src.extraction.date_reasoning import (
    DateClassification,
    DateClassifications,
    check,
    classify,
    to_output,
)
from src.tools.date_tool import DateStatus


def _classified(status, iso="2024-02-16"):
    """Build a model answer with the given status."""
    return DateClassifications(
        dates=[
            DateClassification(
                original_text="Distributed on Budget Day: 16 February 2024",
                normalized_date=iso,
                comparison="2024-02-16 is after 2024-01-01",
                status=status,
            )
        ]
    )


class StubModel(Runnable):
    """Returns a canned classification, recording the prompt it was given."""

    def __init__(self, result):
        self.result = result
        self.received = None

    def with_structured_output(self, schema, **kwargs):
        """Record the schema and return self."""
        self.schema = schema
        return self

    def invoke(self, input, config=None, **kwargs):
        """Record the prompt and return the canned result."""
        self.received = input
        return self.result


def test_correct_classification_agrees_with_the_tool():
    """A right answer is marked as agreeing."""
    results = check(_classified(DateStatus.UPCOMING))

    assert results[0]["agrees"]
    assert results[0]["model_status"] == "Upcoming"


def test_wrong_classification_is_caught():
    """A confident wrong answer is flagged, not accepted.

    This is the guard against hallucination: the model says Expired for a date
    after the reference, and the tool disagrees.
    """
    results = check(_classified(DateStatus.EXPIRED))

    assert not results[0]["agrees"]
    assert results[0]["model_status"] == "Expired"
    assert results[0]["tool_status"] == "Upcoming"


def test_output_matches_the_required_shape():
    """Each entry carries original_text, normalized_date and status."""
    entry = to_output(_classified(DateStatus.UPCOMING))[0]

    assert set(entry) >= {"original_text", "normalized_date", "status"}


def test_reasoning_is_reported():
    """The model's comparison travels with the answer, so it can be read."""
    entry = to_output(_classified(DateStatus.UPCOMING))[0]

    assert "2024-01-01" in entry["reasoning"]


def test_expired_date_verifies():
    """A date before the reference is Expired and agrees with the tool."""
    results = check(_classified(DateStatus.EXPIRED, iso="2008-02-15"))

    assert results[0]["agrees"]


def test_reference_date_itself_is_ongoing():
    """A date equal to the reference is Ongoing under the strict reading."""
    results = check(_classified(DateStatus.ONGOING, iso="2024-01-01"))

    assert results[0]["agrees"]


def test_model_sees_only_the_normalised_dates():
    """The document is not passed to the model, so it cannot invent a date."""
    model = StubModel(_classified(DateStatus.UPCOMING))

    classify(
        [
            {
                "original_text": "Distributed on Budget Day: 16 February 2024",
                "normalized_date": "2024-02-16",
            }
        ],
        model=model,
    )

    rendered = str(model.received)
    assert "2024-02-16" in rendered
    assert "Operating Revenue" not in rendered


def test_reference_date_reaches_the_prompt():
    """The reference is stated to the model rather than left implicit."""
    model = StubModel(_classified(DateStatus.UPCOMING))

    classify(
        [{"original_text": "x", "normalized_date": "2024-02-16"}],
        model=model,
        reference="2024-01-01",
    )

    assert "2024-01-01" in str(model.received)
