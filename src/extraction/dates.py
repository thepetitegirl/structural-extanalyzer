"""Finding dates in the document, then normalising and classifying them.

The work is split deliberately:

  - the model FINDS dates and quotes them as written;
  - tools NORMALISE and CLASSIFY them.

Date arithmetic is deterministic, so a tool does it. Asking a model to compare
dates invites a plausible wrong answer, and there is no way to tell from the
output that it guessed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import load_config
from src.extraction.prompts import load_prompt
from src.ingestion.download import ensure_pdf
from src.ingestion.parser import extract_pages
from src.llm import get_chat_model
from src.tools.date_tool import normalize_date

REFERENCE_DATE = "2024-01-01"


class DateFinding(BaseModel):
    """A date located in the document, recorded as the document writes it."""

    original_text: str = Field(
        description="The sentence or phrase containing the date, verbatim."
    )
    date_as_written: str = Field(
        description="The date exactly as written, e.g. '16 February 2024'. "
        "Do not convert it to another format."
    )
    page: int = Field(gt=0, description="1-indexed page the date was read from.")


class DocumentDates(BaseModel):
    """The dates required from the document."""

    distribution: DateFinding = Field(
        description="The date the document was distributed, from the title page."
    )
    estate_duty: DateFinding = Field(
        description="The date named in the glossary entry for Estate Duty."
    )


async def normalize_dates_mcp(found: DocumentDates) -> list[str | None]:
    """Normalise via the local MCP server rather than an in-process call.

    Same result as `normalize_dates`; the tool runs in a subprocess and is
    reached over the protocol. This is the path the requirement asks for.
    """
    from src.tools.mcp_client import normalize_dates_via_mcp

    return await normalize_dates_via_mcp(
        [finding.date_as_written for finding in (found.distribution, found.estate_duty)]
    )


def normalize_dates(found: DocumentDates) -> list[str | None]:
    """Convert each found date to ISO format.

    This is the output of step 1: a list of normalised dates, in the order the
    document presents them. Classification is a separate step, done by an LLM
    reasoning over this list.

    A date the tool cannot parse yields None rather than a fabricated value.
    """
    return [
        normalize_date.invoke({"text": finding.date_as_written})
        for finding in (found.distribution, found.estate_duty)
    ]


def normalized_with_context(found: DocumentDates) -> list[dict]:
    """Normalised dates paired with the text they came from.

    Step 2 needs the original wording to report alongside its classification,
    and the page lets a reader check the value against the document.
    """
    return [
        {
            "original_text": finding.original_text,
            "normalized_date": normalize_date.invoke(
                {"text": finding.date_as_written}
            ),
            "page": finding.page,
        }
        for finding in (found.distribution, found.estate_duty)
    ]


def find_dates(
    pdf_path: Path | str, pages: list[int], model=None, config=None
) -> DocumentDates:
    """Locate the required dates on the given pages."""
    if config is None:
        config = load_config()

    if model is None:
        config.api_key()
        model = get_chat_model(
            config.provider, config.model, temperature=config.temperature
        )

    page_text = extract_pages(pdf_path, pages)
    chain = load_prompt("dates") | model.with_structured_output(DocumentDates)

    return chain.invoke(
        {
            "page_text": page_text,
            "page_distribution": str(pages[0]),
            "page_estate_duty": str(pages[-1]),
        }
    )


def main() -> None:
    """Find the document's dates and normalise them; print the list.

    Normalisation goes through the local MCP server, which is the path the
    requirement asks for. If the server cannot be reached, the same tool is
    called in-process instead - the decorator fallback the requirement allows.
    """
    config = load_config()

    pdf_path = ensure_pdf(config.pdf_url, config.pdf_path)
    found = find_dates(pdf_path, config.date_pages, config=config)

    try:
        normalised = asyncio.run(normalize_dates_mcp(found))
        route = "local MCP server (stdio)"
    except Exception as exc:
        normalised = normalize_dates(found)
        route = f"in-process @tool - MCP unavailable: {exc}"

    print(json.dumps(normalised, indent=2))
    print(f"\nnormalised via: {route}")


if __name__ == "__main__":
    main()
