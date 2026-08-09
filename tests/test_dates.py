"""Unit tests for src.extraction.dates.

The model is stubbed: no test reaches the network.
"""

import pytest
from langchain_core.runnables import Runnable

from src.extraction.dates import (
    DateFinding,
    DocumentDates,
    normalize_dates,
    normalized_with_context,
)

PAGE_VARS = {"page_distribution": "1", "page_estate_duty": "36"}


@pytest.fixture
def found_dates():
    """What the model returns: dates as written, not normalised."""
    return DocumentDates(
        distribution=DateFinding(
            original_text="Distributed on Budget Day: 16 February 2024",
            date_as_written="16 February 2024",
            page=1,
        ),
        estate_duty=DateFinding(
            original_text="Estate Duty does not apply to a person who dies after 15 February 2008.",
            date_as_written="15 February 2008",
            page=36,
        ),
    )


class StubModel(Runnable):
    """Returns a canned DocumentDates, recording what it was asked."""

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


def test_step_one_returns_a_list_of_iso_dates(found_dates):
    """Step 1's output is a list of normalised dates, nothing more."""
    assert normalize_dates(found_dates) == ["2024-02-16", "2008-02-15"]


def test_normalization_keeps_document_order(found_dates):
    """Dates come back in the order the document presents them."""
    assert normalize_dates(found_dates)[0] == "2024-02-16"


def test_context_pairs_each_date_with_its_source(found_dates):
    """Step 2 needs the original wording, so it travels with the date."""
    entries = normalized_with_context(found_dates)

    assert entries[0]["original_text"].startswith("Distributed on Budget Day")
    assert entries[0]["normalized_date"] == "2024-02-16"
    assert entries[0]["page"] == 1


def test_unparseable_date_is_reported_not_guessed(found_dates):
    """A date the tool cannot read yields None rather than a fabricated one."""
    found_dates.distribution.date_as_written = "sometime next year"

    assert normalize_dates(found_dates)[0] is None
