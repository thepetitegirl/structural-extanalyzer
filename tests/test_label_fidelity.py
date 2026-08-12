"""Whether the answer describes each figure the way the finding labelled it.

Synthesis writes prose over findings it cannot verify, and the four existing
checks all read provenance - value, unit, page, routing. None reads the words
around the number, so a figure can be relabelled and still pass everything.

A live run did exactly that twice: "Operating Revenue" became "total government
revenue", and NIRC was expanded to "Non-Interest Revenue" rather than Net
Investment Returns Contribution. Both kept the correct value, unit and page.
"""

from src.graph.evaluation import check_labels
from src.graph.state import Figure, Finding
from src.graph.trace import Trace


def _trace(answer, label="Estimated FY2024 Operating Revenue", page=13):
    """A trace whose single finding carries one labelled figure."""
    return Trace(
        query="q",
        findings=[
            Finding(
                agent="revenue_agent",
                sub_task="t",
                summary="s",
                figures=[
                    Figure(
                        value=108.6,
                        unit="billion",
                        page=page,
                        label=label,
                        quote="Estimated FY2024 Operating Revenue is $108.6 billion",
                    )
                ],
                pages_read=[9, 13, 15],
            )
        ],
        answer=answer,
    )


def test_passes_when_the_answer_uses_the_findings_wording():
    """The label appears near the citation, so the figure is described as found."""
    trace = _trace("Operating Revenue is estimated at $108.6 billion (p.13).")

    assert check_labels(trace).passed


def test_catches_a_figure_described_as_something_broader():
    """Operating Revenue is not total revenue - the document separates them."""
    trace = _trace("The estimated total government revenue is $108.6 billion (p.13).")

    check = check_labels(trace)

    assert not check.passed
    assert "p.13" in check.detail


def test_catches_an_acronym_expanded_wrongly():
    """NIRC is Net Investment Returns Contribution, not Non-Interest Revenue."""
    trace = _trace(
        "The estimated Non-Interest Revenue (NIR) is $23.5 billion (p.15).",
        label="Estimated FY2024 NIRC",
        page=15,
    )

    assert not check_labels(trace).passed


def test_a_partial_label_match_is_enough():
    """The answer may shorten a label, so long as it keeps its distinctive words."""
    trace = _trace("Operating Revenue: $108.6 billion (p.13).")

    assert check_labels(trace).passed


def test_matching_ignores_case():
    """Sentence case in prose should not fail a title-cased label."""
    trace = _trace("The operating revenue is $108.6 billion (p.13).")

    assert check_labels(trace).passed


def test_an_uncited_answer_is_not_scored():
    """A declined query cites nothing, so there is nothing to compare."""
    trace = Trace(query="q", findings=[], answer="This cannot be answered.")

    assert check_labels(trace).passed
