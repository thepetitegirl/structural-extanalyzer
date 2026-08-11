"""Part 1: Pydantic models describing the extraction output.

Every field carries provenance - the page it came from and the text it was read
from. This is a correctness measure, not decoration: several target terms appear
repeatedly in the source document with different values (Corporate Income Tax
appears on seven pages), so a bare number cannot be verified. The quote lets a
reader confirm the model read the intended row and column.

Units are explicit rather than normalised. The source states top-ups in $million
and every other figure in $billion; silently converting would hide a mismatch,
so the unit travels with the value.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Unit = Literal["million", "billion"]


class _Cited(BaseModel):
    """Base for any value that must be traceable to its source text."""

    quote: str = Field(
        description="Verbatim text from the document containing this value.",
    )

    @field_validator("quote")
    @classmethod
    def quote_must_not_be_blank(cls, value: str) -> str:
        """Reject blank quotes; provenance is the point of the field."""
        if not value.strip():
            raise ValueError("quote must not be empty")
        return value


class Money(_Cited):
    """A monetary amount, with its unit and source."""

    value: float = Field(
        description="The amount. Negative for deficits, including figures the "
        "document shows in parentheses, e.g. (3.57) means -3.57.",
    )
    unit: Unit = Field(description="Unit as stated in the document: million or billion.")
    page: int = Field(gt=0, description="1-indexed page the value was read from.")


class Percentage(_Cited):
    """A percentage, with its source."""

    value: float = Field(
        description="The percentage. Negative for a decrease, including figures "
        "the document shows in parentheses, e.g. (1.2) means -1.2.",
    )
    page: int = Field(gt=0, description="1-indexed page the value was read from.")


class TaxList(_Cited):
    """Named taxes, with the pages they were listed on."""

    names: list[str] = Field(
        min_length=1,
        description="Tax names exactly as written in the document.",
    )
    pages: list[int] = Field(description="1-indexed pages the names were read from.")


class ExtractionResult(BaseModel):
    """The five fields extracted from the budget document."""

    corporate_income_tax: Money = Field(
        description="Corporate Income Tax collections for the target year."
    )
    corporate_income_tax_yoy: Percentage = Field(
        description="Year-on-year percentage change in Corporate Income Tax."
    )
    total_top_ups: Money = Field(
        description="Total top-ups to Endowment and Trust Funds."
    )
    operating_revenue_taxes: TaxList = Field(
        description="Taxes named in the Operating Revenue section."
    )
    fiscal_position: Money = Field(
        description="Overall Fiscal Position for the target year."
    )
