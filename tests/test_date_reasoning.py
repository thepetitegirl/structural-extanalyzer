"""Unit tests for src.extraction.date_reasoning.

The model is stubbed. These check the verification layer - that a wrong
classification is caught rather than accepted - which is the point of the
module.
"""

import json

import pytest
from langchain_core.runnables import Runnable

from src.extraction.date_reasoning import (
    DateClassification,
    DateClassifications,
    check,
    classify,
    save_output,
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


def test_save_output_writes_the_answer_as_json(tmp_path):
    """The reported answer is written to disk as well as printed."""
    path = tmp_path / "dates.json"

    save_output(to_output(_classified(DateStatus.UPCOMING)), path)

    assert json.loads(path.read_text())[0]["normalized_date"] == "2024-02-16"


def test_save_output_creates_missing_directories(tmp_path):
    """A fresh clone has no results directory, so saving makes one."""
    path = tmp_path / "results" / "dates.json"

    save_output(to_output(_classified(DateStatus.UPCOMING)), path)

    assert path.is_file()


def test_save_output_returns_the_path(tmp_path):
    """The caller is told where the file went, so it can report it."""
    path = tmp_path / "dates.json"

    assert save_output(to_output(_classified(DateStatus.UPCOMING)), path) == path


def _classified_period(status, start="2023-12-31", end="2024-02-01"):
    """Build a model answer for a span rather than a single date."""
    return DateClassifications(
        dates=[
            DateClassification(
                original_text="a period running from 31 December 2023 to 1 February 2024",
                normalized_date=None,
                start_date=start,
                end_date=end,
                comparison="2024-01-01 falls between 2023-12-31 and 2024-02-01",
                status=status,
            )
        ]
    )


def test_period_enclosing_the_reference_is_ongoing():
    """A span containing the reference is Ongoing - what a point cannot be.

    Nothing in this document is classified as a period, but the branch exists
    so the definition of Ongoing is implemented rather than only described.
    """
    results = check(_classified_period(DateStatus.ONGOING))

    assert results[0]["agrees"]
    assert results[0]["tool_status"] == "Ongoing"


def test_period_is_verified_with_the_span_rule():
    """A span is checked by classify_period, not the single-date rule."""
    results = check(_classified_period(DateStatus.EXPIRED, "2022-01-01", "2023-01-01"))

    assert results[0]["tool_status"] == "Expired"


def test_wrong_period_classification_is_caught():
    """A wrong span answer is flagged, just as a wrong single date is."""
    results = check(_classified_period(DateStatus.EXPIRED))

    assert not results[0]["agrees"]


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


def test_unparseable_date_is_excluded_from_the_prompt():
    """A date with no ISO form is skipped, not shown as the string None."""
    model = StubModel(_classified(DateStatus.UPCOMING))

    classify(
        [
            {"original_text": "16 Feb. 2024", "normalized_date": None},
            {"original_text": "Budget Day", "normalized_date": "2024-02-16"},
        ],
        model=model,
    )

    assert "None" not in str(model.received)


def test_all_dates_unparseable_raises():
    """With nothing to classify, the failure is loud rather than a model call."""
    with pytest.raises(ValueError, match="parseable"):
        classify(
            [{"original_text": "sometime", "normalized_date": None}],
            model=StubModel(None),
        )
