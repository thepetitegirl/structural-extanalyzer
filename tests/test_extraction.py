"""Unit tests for src.extraction.extractor and src.llm.

The model is always a stub: no test may reach the network or spend API budget.
"""

import pytest
from langchain_core.runnables import Runnable

from src.extraction.extractor import build_chain, extract
from src.extraction.schemas import ExtractionResult, Money, Percentage, TaxList
from src.llm import UnsupportedProviderError, get_chat_model

PAGE_VARS = {
    "page_cit": "5",
    "page_cit_yoy": "5",
    "page_top_ups": "20",
    "page_taxes": "5 AND 6",
    "page_fiscal": "8",
}


@pytest.fixture
def expected_result():
    """A result carrying the document's known values."""
    return ExtractionResult(
        corporate_income_tax=Money(
            value=28.4, unit="billion", page=5, quote="revised to $28.4 billion"
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
            value=-3.57, unit="billion", page=8, quote="OVERALL FISCAL POSITION (3.57)"
        ),
    )


class StubModel(Runnable):
    """Stands in for a chat model, recording what it was asked.

    Subclasses Runnable so it composes with `|` exactly as a real model does.
    """

    def __init__(self, result):
        self.result = result
        self.received = None
        self.schema = None

    def with_structured_output(self, schema, **kwargs):
        """Record the requested schema and return self."""
        self.schema = schema
        return self

    def invoke(self, input, config=None, **kwargs):
        """Record the rendered prompt and return the canned result."""
        self.received = input
        return self.result


def test_chain_returns_validated_result(expected_result):
    """The chain yields an ExtractionResult, not raw text."""
    chain = build_chain(StubModel(expected_result))

    result = chain.invoke(
        {"page_text": "--- page 5 ---", "target_year": "FY2023", **PAGE_VARS}
    )

    assert isinstance(result, ExtractionResult)
    assert result.corporate_income_tax.value == 28.4


def test_chain_passes_page_text_to_model(expected_result):
    """Page text reaches the model rather than being dropped."""
    model = StubModel(expected_result)

    build_chain(model).invoke(
        {
            "page_text": "--- page 8 ---\nOVERALL FISCAL POSITION",
            "target_year": "FY2023",
            **PAGE_VARS,
        }
    )

    rendered = str(model.received)
    assert "OVERALL FISCAL POSITION" in rendered


def test_chain_states_target_year(expected_result):
    """The target year is stated in the prompt, not left implicit."""
    model = StubModel(expected_result)

    build_chain(model).invoke({"page_text": "text", "target_year": "FY2023", **PAGE_VARS})

    assert "FY2023" in str(model.received)


def test_chain_requests_the_result_schema(expected_result):
    """Structured output is bound to ExtractionResult."""
    model = StubModel(expected_result)

    build_chain(model).invoke({"page_text": "text", "target_year": "FY2023", **PAGE_VARS})

    assert model.schema is ExtractionResult


def test_extract_reads_configured_pages(tmp_path, expected_result, monkeypatch):
    """extract() pulls the configured pages and returns a validated result."""
    monkeypatch.setattr(
        "src.extraction.extractor.extract_pages",
        lambda path, pages: f"--- page {pages[0]} ---",
    )
    model = StubModel(expected_result)

    result = extract("any.pdf", [5], model=model)

    assert result.fiscal_position.value == -3.57


def test_unsupported_provider_raises():
    """An unknown provider names the ones that are supported."""
    with pytest.raises(UnsupportedProviderError, match="groq"):
        get_chat_model("nonexistent-provider", "some-model")
