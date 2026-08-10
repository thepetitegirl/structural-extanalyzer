"""Step 2: an LLM classifies each normalised date against a reference date.

The requirement asks for LLM reasoning here rather than a comparison operator,
so the model produces the answer. That invites a confident wrong answer, since
a status label carries no evidence of how it was reached.

Three things guard against that:

  - the model sees only the normalised dates, not the document, so it cannot
    introduce a date that was never extracted;
  - the prompt requires it to state the comparison it made, so faulty reasoning
    is visible rather than hidden behind a label;
  - every answer is checked against `classify_date`, which computes the same
    result deterministically. A disagreement is reported, not silently
    accepted.

The checker does not overrule the model - the LLM's answer is what the
requirement asks for, so it is what gets reported, with the disagreement
attached.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from src.config import load_config
from src.evaluation import score_dates
from src.extraction.dates import find_dates, normalized_with_context
from src.extraction.prompts import load_prompt
from src.ingestion.download import ensure_pdf
from src.llm import get_chat_model
from src.tools.date_tool import DateStatus, classify_date

REFERENCE_DATE = "2024-01-01"


class DateClassification(BaseModel):
    """One date, classified by the model."""

    original_text: str = Field(
        description="The text the date was found in, copied from the input."
    )
    normalized_date: str = Field(description="The ISO date, copied from the input.")
    comparison: str = Field(
        description="The comparison made against the reference date, e.g. "
        "'2008 is earlier than 2024'. State this before deciding the status."
    )
    status: DateStatus = Field(
        description="Expired if before the reference, Upcoming if after, "
        "Ongoing if it is the reference date itself."
    )


class DateClassifications(BaseModel):
    """The classified dates."""

    dates: list[DateClassification] = Field(
        min_length=1, description="One entry per date given."
    )


def to_output(classified: DateClassifications) -> list[dict]:
    """Return the classified dates.

    The three keys the requirement specifies, plus the comparison the model
    made. The reasoning is not required, but a bare status carries no evidence
    of how it was reached - and a wrong classification is indistinguishable
    from a right one without it.
    """
    return [
        {
            "original_text": entry.original_text,
            "normalized_date": entry.normalized_date,
            "status": str(entry.status),
            "reasoning": entry.comparison,
        }
        for entry in classified.dates
    ]


def check(classified: DateClassifications, reference: str = REFERENCE_DATE) -> list[dict]:
    """Compare each classification against the deterministic tool.

    Kept separate from the answer: the requirement's output is three keys, so
    this is diagnostic rather than part of the result. A disagreement means the
    model reached a conclusion the arithmetic does not support.
    """
    checked = []

    for entry in classified.dates:
        expected = classify_date.invoke(
            {"iso_date": entry.normalized_date, "reference": reference}
        )
        checked.append(
            {
                "normalized_date": entry.normalized_date,
                "model_status": str(entry.status),
                "tool_status": str(expected),
                "reasoning": entry.comparison,
                "agrees": entry.status == expected,
            }
        )

    return checked


def classify(
    dates: list[dict], model=None, reference: str = REFERENCE_DATE, config=None
) -> DateClassifications:
    """Have the model classify each normalised date against the reference."""
    if config is None:
        config = load_config()

    if model is None:
        model = get_chat_model(config)

    # Only the normalised dates and their source text reach the model. Without
    # the document it cannot introduce a date that was never extracted. A date
    # the tool could not parse has no ISO form to classify, so it is excluded
    # rather than shown as the literal "None"; main() reports the skip.
    usable = [entry for entry in dates if entry["normalized_date"] is not None]
    if not usable:
        raise ValueError("No parseable dates to classify.")

    listing = "\n".join(
        f"- {entry['normalized_date']}  (from: {entry['original_text']})"
        for entry in usable
    )

    chain = load_prompt("date_reasoning") | model.with_structured_output(
        DateClassifications
    )

    return chain.invoke({"dates": listing, "reference": reference})


def main() -> None:
    """Find, normalise, classify and verify the document's dates."""
    config = load_config()

    pdf_path = ensure_pdf(config.pdf_url)
    found = find_dates(pdf_path, config.date_pages, config=config)
    normalised = normalized_with_context(found)

    for entry in normalised:
        if entry["normalized_date"] is None:
            print(f"WARNING: could not normalise {entry['original_text']!r}; skipped.")

    classified = classify(normalised, config=config)

    print(json.dumps(to_output(classified), indent=2))

    print()
    print(score_dates(to_output(classified)).table())

    # Diagnostic, not part of the answer: does the arithmetic agree?
    checked = check(classified)
    disagreements = [entry for entry in checked if not entry["agrees"]]

    if disagreements:
        print(
            f"\nWARNING: {len(disagreements)} classification(s) disagree "
            "with the deterministic check:"
        )
        for entry in disagreements:
            print(
                f"  {entry['normalized_date']}: model said "
                f"{entry['model_status']}, tool computes {entry['tool_status']}"
            )
    else:
        print(f"\nAll {len(checked)} classifications agree with the deterministic check.")


if __name__ == "__main__":
    main()
