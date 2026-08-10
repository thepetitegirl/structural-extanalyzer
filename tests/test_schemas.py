"""Unit tests for src.extraction.schemas."""

import pytest
from pydantic import ValidationError

from src.extraction.schemas import ExtractionResult, Money, Percentage, TaxList


def test_money_carries_value_page_and_quote():
    """A numeric field records where its value came from."""
    field = Money(value=28.38, unit="billion", page=8, quote="Corporate Income Tax 28.38")

    assert field.value == 28.38
    assert field.unit == "billion"
    assert field.page == 8
    assert "28.38" in field.quote


def test_money_rejects_unknown_unit():
    """Units are constrained, so a million/billion mix-up fails loudly."""
    with pytest.raises(ValidationError):
        Money(value=20352.0, unit="thousand", page=20, quote="Total 20,352")


def test_money_accepts_negative_value():
    """Deficits are negative; parenthesised source figures map to a minus sign."""
    field = Money(
        value=-3.57, unit="billion", page=8, quote="OVERALL FISCAL POSITION (3.57)"
    )

    assert field.value == -3.57


def test_percentage_accepts_negative():
    """A year-on-year fall is a negative percentage, not an error."""
    assert Percentage(value=-1.2, page=16, quote="(1.2)").value == -1.2


def test_quote_must_not_be_empty():
    """An empty quote defeats the purpose of provenance, so it is rejected."""
    with pytest.raises(ValidationError):
        Percentage(value=17.0, page=5, quote="   ")


def test_page_must_be_positive():
    """Pages are 1-indexed, so zero or negative page numbers are invalid."""
    with pytest.raises(ValidationError):
        Percentage(value=17.0, page=0, quote="17.0%")


def test_tax_list_holds_names_with_provenance():
    """The tax list records its source pages alongside the names."""
    taxes = TaxList(
        names=["Corporate Income Tax", "Personal Income Tax"],
        pages=[5, 6],
        quote="higher collections from Corporate Income Tax",
    )

    assert len(taxes.names) == 2
    assert taxes.pages == [5, 6]


def test_tax_list_rejects_empty():
    """An empty tax list means extraction failed and should not validate."""
    with pytest.raises(ValidationError):
        TaxList(names=[], pages=[5], quote="none found")


def test_extraction_result_holds_all_five_fields(sample_result):
    """The top-level result carries the five required fields."""
    assert sample_result.corporate_income_tax.value == 28.38
    assert sample_result.corporate_income_tax_yoy.value == 17.0
    assert sample_result.total_top_ups.value == 20352.0
    assert sample_result.operating_revenue_taxes.names
    assert sample_result.fiscal_position.value == -3.57


def test_extraction_result_rejects_missing_field():
    """Every field is required; a partial extraction fails validation."""
    with pytest.raises(ValidationError):
        ExtractionResult(
            corporate_income_tax=Money(value=28.38, unit="billion", page=8, quote="x")
        )


@pytest.fixture
def sample_result():
    """A fully populated result, using the document's known values."""
    return ExtractionResult(
        corporate_income_tax=Money(
            value=28.38,
            unit="billion",
            page=8,
            quote="Corporate Income Tax 23.07 24.26 28.38",
        ),
        corporate_income_tax_yoy=Percentage(
            value=17.0, page=5, quote="$4.1 billion (17.0%) higher"
        ),
        total_top_ups=Money(value=20352.0, unit="million", page=20, quote="Total 20,352"),
        operating_revenue_taxes=TaxList(
            names=["Corporate Income Tax", "Goods and Services Tax"],
            pages=[5, 6],
            quote="higher collections from Corporate Income Tax",
        ),
        fiscal_position=Money(
            value=-3.57,
            unit="billion",
            page=8,
            quote="OVERALL FISCAL POSITION 1.72 (0.35) (3.57)",
        ),
    )
