"""Unit tests for src.evaluation.

Scoring is pure comparison, so these run without a model or a network call.
"""

import pytest
import yaml

from src.evaluation import load_expected, score_result
from src.extraction.schemas import ExtractionResult, Money, Percentage, TaxList


def _result(cit_value=28.4, cit_page=5, taxes=None):
    """Build a result, correct by default, with selected fields overridable."""
    return ExtractionResult(
        corporate_income_tax=Money(
            value=cit_value, unit="billion", page=cit_page, quote="q"
        ),
        corporate_income_tax_yoy=Percentage(value=17.0, page=5, quote="q"),
        total_top_ups=Money(value=20352.0, unit="million", page=20, quote="q"),
        operating_revenue_taxes=TaxList(
            names=taxes
            or [
                "Corporate Income Tax",
                "Other Taxes",
                "Vehicle Quota Premiums",
                "Personal Income Tax",
                "Assets Taxes",
                "Betting Taxes",
                "Goods and Services Tax",
            ],
            pages=[5, 6],
            quote="q",
        ),
        fiscal_position=Money(value=-3.57, unit="billion", page=8, quote="q"),
    )


@pytest.fixture
def expected():
    """The committed expected values."""
    return load_expected()


def test_correct_result_passes_every_check(expected):
    """A fully correct result scores all checks as passed."""
    report = score_result(_result(), expected)

    assert report.passed
    assert all(check.passed for check in report.checks)


def test_wrong_value_fails(expected):
    """A wrong figure is reported as a failure."""
    report = score_result(_result(cit_value=28.03), expected)

    assert not report.passed
    failed = [c for c in report.checks if not c.passed]
    assert any("corporate_income_tax" == c.field for c in failed)


def test_right_value_from_wrong_page_fails(expected):
    """A correct number read from the wrong page still fails.

    This is the check that catches an instruction being ignored: 28.4 is right,
    but if it came from page 8 the model was not doing what it was told.
    """
    report = score_result(_result(cit_page=8), expected)

    assert not report.passed
    cit = next(c for c in report.checks if c.field == "corporate_income_tax")
    assert not cit.passed
    assert "page" in cit.detail.lower()


def test_tolerance_allows_rounding(expected):
    """Values within tolerance pass, so 28.400001 is not a failure."""
    assert score_result(_result(cit_value=28.4001), expected).passed


def test_short_tax_list_fails(expected):
    """A truncated tax list fails the minimum-count check."""
    report = score_result(_result(taxes=["Corporate Income Tax"]), expected)

    taxes = next(c for c in report.checks if c.field == "operating_revenue_taxes")
    assert not taxes.passed


def test_tax_list_missing_required_member_fails(expected):
    """A list of the right length but missing a required tax fails."""
    names = [f"Tax {n}" for n in range(7)]
    report = score_result(_result(taxes=names), expected)

    taxes = next(c for c in report.checks if c.field == "operating_revenue_taxes")
    assert not taxes.passed
    assert "Corporate Income Tax" in taxes.detail


def test_over_long_tax_list_fails(expected):
    """Extra names mean an uncited page was read, so a longer list fails.

    Pages 5-6 name seven taxes. Withholding Tax and Stamp Duty appear only in
    Table 1.1 on page 8, so their presence proves the page binding was ignored.
    """
    names = [
        "Corporate Income Tax",
        "Other Taxes",
        "Vehicle Quota Premiums",
        "Personal Income Tax",
        "Assets Taxes",
        "Betting Taxes",
        "Goods and Services Tax",
        "Withholding Tax",
        "Stamp Duty",
        "Motor Vehicle Taxes",
        "Statutory Boards' Contributions",
        "Fees and Charges",
    ]
    report = score_result(_result(taxes=names), expected)

    taxes = next(c for c in report.checks if c.field == "operating_revenue_taxes")
    assert not taxes.passed
    assert "at most" in taxes.detail


def test_expected_file_covers_every_result_field(expected):
    """Every field in the schema has an expected value, so none is unscored.

    `dates` is excluded: it holds the Part 2 expectations, which are scored by
    score_dates rather than against ExtractionResult.
    """
    part_one = set(expected) - {"dates"}

    assert set(ExtractionResult.model_fields) == part_one


def test_summary_counts_passes(expected):
    """The report summarises how many checks passed."""
    report = score_result(_result(), expected)

    assert report.summary().startswith("5/5")
