"""Part 2, step 1: locating dates in the document and normalising them to ISO.

Responsibilities are divided by the nature of each task. The model locates each
date and quotes it as written, since a date may be phrased in several ways and
recognising one in prose requires judgement. `normalize_date` then parses that
text into ISO format, which is deterministic and therefore belongs in a tool.

Classification is step 2 and lives in `date_reasoning.py`. It is performed by
the LLM rather than by a tool: reasoning is the point, so
`classify_date` verifies the model's conclusion afterwards rather than
supplying it.

**Date discovery is assumed to succeed.** Each required date is stated
explicitly on its cited page, and no secondary method is attempted if the model
does not return one. There is no pattern match, retry, or alternative model.
Failures are surfaced rather than absorbed: an absent date fails schema
validation, and a date `normalize_date` cannot parse is reported as skipped.
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
    """The dates to be extracted, named individually.

    **This schema is specific to one document.** Both the field names and their
    descriptions describe where these two dates live in this publication, and a
    different source would need the class rewritten rather than reconfigured.

    Named fields are chosen over a generic `list[DateFinding]` because they make
    absence detectable: if the model returns only one date, validation fails
    here. A list would accept whatever it found, and a missing date would look
    like a short list rather than an error. The descriptions also carry into the
    prompt, so "from the title page" is doing work rather than documenting.

    The alternative - building the schema at run time from configuration -
    would generalise it, at the cost of static typing on the result and of
    validation that can name what is missing. For a fixed set of dates that
    trade is not worth making, but it is the change a second document would
    require.
    """

    distribution: DateFinding = Field(
        description="The date the document was distributed, from the title page."
    )
    estate_duty: DateFinding = Field(
        description="The date named in the glossary entry for Estate Duty."
    )


def _findings(found: DocumentDates) -> list[DateFinding]:
    """Each date in the order the schema declares it.

    Read from the schema rather than named here, so adding a date means
    changing `DocumentDates` alone rather than every function that walks it.
    """
    return [getattr(found, name) for name in DocumentDates.model_fields]


async def normalize_dates_mcp(found: DocumentDates) -> list[str | None]:
    """Normalise the found dates. **Primary path: over MCP.**

    Sends each date's written form to the MCP server, which runs as a
    subprocess and is reached over the protocol rather than called as a Python
    function. `normalize_dates` below is the same operation without the
    protocol, kept as the fallback; both end at `normalize_date` in
    date_tool.py, so the results are identical and only the route differs.

    Async because the MCP client is async-only - see mcp_client.py. Callers in
    synchronous code wrap this in `asyncio.run`.
    """
    from src.tools.mcp_client import normalize_dates_via_mcp

    return await normalize_dates_via_mcp(
        [finding.date_as_written for finding in _findings(found)]
    )


def normalize_dates(found: DocumentDates) -> list[str | None]:
    """Convert each found date to ISO format. **Fallback path: in-process.**

    Calls `normalize_date` directly instead of going through the MCP server,
    for when the subprocess cannot be started - `main` below falls back to this
    and reports which route it used. The result is the same as
    `normalize_dates_mcp`; this one needs no subprocess and is therefore
    synchronous.

    Either way this is the output of step 1: a list of normalised dates, in the
    order the document presents them. Classification is a separate step, done
    by an LLM reasoning over this list.

    A date the tool cannot parse yields None rather than a fabricated value.
    """
    return [
        normalize_date.invoke({"text": finding.date_as_written})
        for finding in _findings(found)
    ]


def normalized_with_context(found: DocumentDates) -> list[dict]:
    """Normalised dates paired with the text they came from.

    Step 2 needs the original wording to report alongside its classification,
    and the page lets a reader check the value against the document.
    """
    return [
        {
            "original_text": finding.original_text,
            "normalized_date": normalize_date.invoke({"text": finding.date_as_written}),
            "page": finding.page,
        }
        for finding in _findings(found)
    ]


def find_dates(pdf_path: Path | str, model=None, config=None) -> DocumentDates:
    """Locate each date on the page config binds it to.

    Pages come from the `date_pages` map by name, so a date is never read from
    whichever page happens to sit at a given position in a list.
    """
    if config is None:
        config = load_config()

    if model is None:
        model = get_chat_model(config)

    page_text = extract_pages(pdf_path, config.date_page_numbers)
    chain = load_prompt("dates") | model.with_structured_output(DocumentDates) #match schema

    # One prompt variable per date, named for the schema field it binds.
    pages = {
        f"page_{name}": str(config.page_for_date(name))
        for name in DocumentDates.model_fields
    }

    return chain.invoke({"page_text": page_text, **pages})


def main() -> None:
    """Find the document's dates and normalise them; print the list.

    Normalisation goes through the local MCP server. If the server cannot be
    reached, the same tool is called in-process instead.
    """
    config = load_config()

    pdf_path = ensure_pdf(config.pdf_url)
    found = find_dates(pdf_path, config=config)

    try: #MCP
        normalised = asyncio.run(normalize_dates_mcp(found))
        route = "local MCP server (stdio)"
    except Exception as exc: #falls back to tool
        normalised = normalize_dates(found)
        route = f"in-process @tool - MCP unavailable: {exc}"

    print(json.dumps(normalised, indent=2))
    print(f"\nnormalised via: {route}")


if __name__ == "__main__":
    main()
