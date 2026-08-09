"""Unit tests for src.tools.date_tool.

Normalisation and classification are pure functions, so these need no model.
"""

import pytest

from src.tools.date_tool import (
    DateStatus,
    classify_date,
    classify_period,
    normalize_date,
)


def test_normalizes_long_form_date():
    """A date written as '16 February 2024' becomes 2024-02-16."""
    assert normalize_date.invoke({"text": "16 February 2024"}) == "2024-02-16"


def test_normalizes_date_inside_a_sentence():
    """The date is found within surrounding words."""
    result = normalize_date.invoke(
        {"text": "Distributed on Budget Day: 16 February 2024"}
    )

    assert result == "2024-02-16"


def test_normalizes_abbreviated_month():
    """Abbreviated month names are accepted."""
    assert normalize_date.invoke({"text": "15 Feb 2008"}) == "2008-02-15"


def test_normalizes_month_first_form():
    """A month-first date is accepted."""
    assert normalize_date.invoke({"text": "February 16, 2024"}) == "2024-02-16"


def test_passes_through_iso_input():
    """A date already in ISO form is returned unchanged."""
    assert normalize_date.invoke({"text": "2024-02-16"}) == "2024-02-16"


def test_unparseable_text_returns_none():
    """Text with no date returns None rather than guessing."""
    assert normalize_date.invoke({"text": "no date here"}) is None


def test_past_date_is_expired():
    """A date before the reference date has passed."""
    status = classify_date.invoke(
        {"iso_date": "2008-02-15", "reference": "2024-01-01"}
    )

    assert status == DateStatus.EXPIRED


def test_future_date_is_upcoming():
    """A date after the reference date is still to come."""
    status = classify_date.invoke(
        {"iso_date": "2024-02-16", "reference": "2024-01-01"}
    )

    assert status == DateStatus.UPCOMING


def test_reference_date_itself_is_ongoing():
    """A date equal to the reference is active, not past or future."""
    status = classify_date.invoke(
        {"iso_date": "2024-01-01", "reference": "2024-01-01"}
    )

    assert status == DateStatus.ONGOING


def test_classification_is_relative_to_supplied_reference():
    """The reference date is a parameter, not today's date.

    The requirement fixes it at 2024-01-01, so a date in 2024 must classify as
    upcoming even though that date is now in the past.
    """
    assert (
        classify_date.invoke({"iso_date": "2024-02-16", "reference": "2024-01-01"})
        == DateStatus.UPCOMING
    )
    assert (
        classify_date.invoke({"iso_date": "2024-02-16", "reference": "2025-01-01"})
        == DateStatus.EXPIRED
    )


def test_period_spanning_the_reference_is_ongoing():
    """A period that has started and not finished is active."""
    status = classify_period.invoke(
        {"start": "2024-04-01", "end": "2025-03-31", "reference": "2024-06-01"}
    )

    assert status == DateStatus.ONGOING


def test_period_still_ongoing_late_in_its_run():
    """Ongoing does not require the reference to be near the start.

    A period is active for its whole span, not just when it begins.
    """
    status = classify_period.invoke(
        {"start": "2024-04-01", "end": "2025-03-31", "reference": "2025-03-30"}
    )

    assert status == DateStatus.ONGOING


def test_period_boundaries_are_inclusive():
    """The first and last days of a period are part of it."""
    for reference in ("2024-04-01", "2025-03-31"):
        status = classify_period.invoke(
            {"start": "2024-04-01", "end": "2025-03-31", "reference": reference}
        )
        assert status == DateStatus.ONGOING, f"{reference} should be inside the period"


def test_finished_period_is_expired():
    """A period whose end has passed is expired."""
    status = classify_period.invoke(
        {"start": "2022-04-01", "end": "2023-03-31", "reference": "2024-01-01"}
    )

    assert status == DateStatus.EXPIRED


def test_period_not_yet_started_is_upcoming():
    """A period beginning after the reference has not started.

    FY2024 runs 1 April 2024 to 31 March 2025, so against 2024-01-01 it is
    upcoming - it has not begun yet.
    """
    status = classify_period.invoke(
        {"start": "2024-04-01", "end": "2025-03-31", "reference": "2024-01-01"}
    )

    assert status == DateStatus.UPCOMING


def test_period_ending_before_it_starts_raises():
    """A reversed period is a data error, not something to classify."""
    with pytest.raises(ValueError, match="before it starts"):
        classify_period.invoke(
            {"start": "2025-03-31", "end": "2024-04-01", "reference": "2024-01-01"}
        )


def test_invalid_iso_date_raises():
    """A malformed date is an error, not a silent misclassification."""
    with pytest.raises(ValueError):
        classify_date.invoke({"iso_date": "not-a-date", "reference": "2024-01-01"})


def test_tools_are_langchain_tools():
    """Both are decorated tools, so an agent can bind them."""
    assert normalize_date.name == "normalize_date"
    assert classify_date.name == "classify_date"
    assert normalize_date.description
    assert classify_date.description
