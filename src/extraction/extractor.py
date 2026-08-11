"""Part 1: the extraction chain - page text in, ExtractionResult out."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.evaluation import score_result
from src.extraction.prompts import load_prompt
from src.extraction.schemas import ExtractionResult
from src.ingestion.download import ensure_pdf
from src.ingestion.parser import extract_pages
from src.llm import get_chat_model

DEFAULT_TARGET_YEAR = "Revised FY2023"


def build_chain(model, prompt_name: str = "extraction"):
    """Compose prompt, model and schema into a runnable chain."""
    prompt = load_prompt(prompt_name)
    return prompt | model.with_structured_output(ExtractionResult)


def page_variables(config) -> dict[str, str]:
    """Map each field's configured page(s) to its prompt variable.

    Keeps the binding in config.yml rather than in the prompt text, so changing
    a page in one place changes it everywhere.
    """
    return {
        "page_cit": config.pages_for_field("corporate_income_tax"),
        "page_cit_yoy": config.pages_for_field("corporate_income_tax_yoy"),
        "page_top_ups": config.pages_for_field("total_top_ups"),
        "page_taxes": config.pages_for_field("operating_revenue_taxes"),
        "page_fiscal": config.pages_for_field("fiscal_position"),
    }


def extract(
    pdf_path: Path | str,
    pages: list[int],
    model=None,
    target_year: str = DEFAULT_TARGET_YEAR,
    config=None,
) -> ExtractionResult:
    """Extract the five fields from the given pages of a document.

    A model may be supplied directly; otherwise one is built from config.
    """
    if config is None:
        config = load_config()

    if model is None:
        model = get_chat_model(config)

    page_text = extract_pages(pdf_path, pages)
    chain = build_chain(model)
    return chain.invoke(
        {"page_text": page_text, "target_year": target_year, **page_variables(config)}
    )


def main() -> None:
    """Run extraction using config.yml, print the result, and score it.

    Scoring against known values is what makes prompt iteration measurable:
    edit prompts/extraction.yaml, re-run, and read the table.
    """
    config = load_config()
    pdf_path = ensure_pdf(config.pdf_url)
    result = extract(pdf_path, config.pages, config=config)

    print(json.dumps(result.model_dump(), indent=2))
    print()
    print(f"{config.provider}/{config.model}  temperature={config.temperature}")
    print(score_result(result).table())


if __name__ == "__main__":
    main()
