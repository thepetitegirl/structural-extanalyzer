"""Date normalisation and classification, exposed as LangChain tools.

Both are deterministic Python. The model decides *when* to call them and on what
text; it never does the date arithmetic itself, which is the point - a model
asked to compare dates can get it wrong, and a comparison operator cannot.

The reference date is a parameter rather than today's date. The requirement
fixes it at 2024-01-01, so 2024-02-16 must classify as upcoming even though that
date has since passed.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import StrEnum

from langchain_core.tools import tool

# Tried in order. The document writes dates as "16 February 2024", but a tool
# meant for reuse should accept the obvious variants.
DATE_FORMATS = (
    "%d %B %Y",  # 16 February 2024
    "%d %b %Y",  # 16 Feb 2024
    "%B %d, %Y",  # February 16, 2024
    "%b %d, %Y",  # Feb 16, 2024
    "%B %d %Y",  # February 16 2024
    "%Y-%m-%d",  # already ISO
    "%d/%m/%Y",  # 16/02/2024
)

# Matches a date inside a longer sentence, e.g.
# "Distributed on Budget Day: 16 February 2024".
DATE_PATTERNS = (
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
    r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}",
    r"\d{4}-\d{2}-\d{2}",
    r"\d{1,2}/\d{1,2}/\d{4}",
)


class DateStatus(StrEnum):
    """How a date stands relative to a reference date."""

    EXPIRED = "Expired"
    UPCOMING = "Upcoming"
    ONGOING = "Ongoing"


def _parse(text: str) -> date | None:
    """Parse a bare date string, trying each accepted format."""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


@tool
def normalize_date(text: str) -> str | None:
    """Convert a date written in prose into ISO format (YYYY-MM-DD).

    Accepts a bare date ("16 February 2024") or a date inside a sentence
    ("Distributed on Budget Day: 16 February 2024"). Returns None when no date
    can be found, rather than guessing.
    """
    if (parsed := _parse(text)) is not None:
        return parsed.isoformat()

    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text):
            if (parsed := _parse(match.group(0))) is not None:
                return parsed.isoformat()

    return None


@tool
def classify_period(
    start: str, end: str, reference: str = "2024-01-01"
) -> DateStatus:
    """Classify a date range against a reference date.

    A period is Ongoing when it has started and has not yet finished - that is,
    when the reference falls anywhere between start and end inclusive. It is
    Expired once the end has passed, and Upcoming before the start.

    This is what makes Ongoing meaningful: a single date can only coincide with
    the reference, but a period can genuinely be active around it.
    """
    try:
        first = date.fromisoformat(start)
        last = date.fromisoformat(end)
        against = date.fromisoformat(reference)
    except ValueError as exc:
        raise ValueError(f"Expected ISO dates (YYYY-MM-DD): {exc}") from exc

    if last < first:
        raise ValueError(f"Period ends ({end}) before it starts ({start}).")

    if last < against:
        return DateStatus.EXPIRED
    if first > against:
        return DateStatus.UPCOMING
    return DateStatus.ONGOING


@tool
def classify_date(iso_date: str, reference: str = "2024-01-01") -> DateStatus:
    """Classify an ISO date against a reference date.

    Returns Expired if it falls before the reference, Upcoming if after, and
    Ongoing if it is the reference date itself - a period that is currently
    active rather than past or future.

    The reference defaults to 2024-01-01 as the requirement specifies. It is a
    parameter, not today's date, so results stay stable over time.
    """
    try:
        target = date.fromisoformat(iso_date)
        against = date.fromisoformat(reference)
    except ValueError as exc:
        raise ValueError(f"Expected ISO dates (YYYY-MM-DD): {exc}") from exc

    if target < against:
        return DateStatus.EXPIRED
    if target > against:
        return DateStatus.UPCOMING
    return DateStatus.ONGOING
