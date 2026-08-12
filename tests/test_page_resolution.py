"""Resolving a figure's page from the quote it was read from.

An agent can read the right sentence and record the wrong page number: both
pages are in its set, the value is correct, and nothing in the value indicates
the attribution is wrong. Only the quote does. Since the quote must be verbatim,
the page is derivable rather than a matter of judgement - so it is derived here
rather than trusted from the model.
"""

from src.ingestion.parser import page_of_quote

PAGES = (
    "--- page 13 ---\n"
    "A basic deficit of $6.1 billion is estimated for FY2024. After factoring "
    "in Top-ups to Endowment and Trust Funds of $20.4 billion, NIRC of "
    "$23.5 billion, the position is a surplus.\n"
    "Estimated FY2024 Operating Revenue is $108.6 billion (15.1% of GDP).\n"
    "\n"
    "--- page 15 ---\n"
    "2.4 Net Investment Returns Contribution\n"
    "Estimated FY2024 NIRC is $23.5 billion, which is $0.6 billion higher.\n"
)


def test_finds_the_page_holding_the_quote():
    """A quote is attributed to the marker section that contains it."""
    assert page_of_quote("Estimated FY2024 NIRC is $23.5 billion", PAGES, 13) == 15


def test_leaves_a_correct_page_alone():
    """A quote already attributed correctly keeps its page."""
    quote = "Estimated FY2024 Operating Revenue is $108.6 billion"
    assert page_of_quote(quote, PAGES, 13) == 13


def test_falls_back_to_the_claimed_page_when_the_quote_is_not_found():
    """A paraphrased quote is left as the model recorded it.

    Correcting it is impossible, and discarding the figure would lose a value
    that may well be right. `check_traceability` still fails it, which is the
    behaviour that should surface a paraphrase.
    """
    assert page_of_quote("NIRC is roughly 23 billion dollars", PAGES, 13) == 13


def test_matching_ignores_whitespace_differences():
    """Pypdf spaces text irregularly, so matching normalises whitespace."""
    assert page_of_quote("Estimated  FY2024\nNIRC is $23.5 billion", PAGES, 13) == 15


def test_an_empty_quote_keeps_the_claimed_page():
    """Nothing to match on, so there is nothing to correct."""
    assert page_of_quote("", PAGES, 13) == 13


def test_text_without_markers_keeps_the_claimed_page():
    """Page text is always marked up; absent markers, there is nothing to read."""
    assert page_of_quote("some text", "unmarked page body", 13) == 13
