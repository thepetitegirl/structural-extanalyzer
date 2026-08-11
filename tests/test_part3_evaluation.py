"""Deterministic checks on an open-ended answer.

Part 3's queries have no single correct wording, so prose cannot be scored.
Four things can be:

  - did the supervisor route to the agents the query needed;
  - are the required figures present with the right unit;
  - does every quote actually appear on the page it cites;
  - did any agent cite a page outside its own set.

Together these verify the answer is grounded and correctly routed, which is
what matters. Whether it reads well is left to the reader.
"""

import pytest

from src.graph.evaluation import (
    check_figures,
    check_page_discipline,
    check_routing,
    check_traceability,
    score_part3,
)
from src.graph.state import Decision, Figure, Finding
from src.graph.trace import Trace


def _finding(agent, pages, figures):
    """One agent's report."""
    return Finding(
        agent=agent,
        sub_task="find it",
        summary="summary",
        figures=figures,
        pages_read=pages,
    )


def _figure(value, unit, page, label, quote):
    """One cited figure."""
    return Figure(value=value, unit=unit, page=page, label=label, quote=quote)


@pytest.fixture
def good_trace():
    """A trace with correct routing, figures and citations."""
    return Trace(
        query="What are the revenue streams, and how is the Fund supported?",
        decisions=[
            Decision(
                turn=1,
                reasoning="revenue first",
                chose="revenue_agent",
                routed_to="revenue_agent",
            ),
            Decision(
                turn=2,
                reasoning="then spending",
                chose="expenditure_agent",
                routed_to="expenditure_agent",
            ),
            Decision(turn=3, reasoning="done", chose="synthesis", routed_to="synthesis"),
        ],
        findings=[
            _finding(
                "revenue_agent",
                [9, 13, 15],
                [
                    _figure(
                        108.6,
                        "billion",
                        13,
                        "Operating Revenue",
                        "Estimated FY2024 Operating Revenue is $108.6 billion",
                    )
                ],
            ),
            _finding(
                "expenditure_agent",
                [16, 18, 20],
                [
                    _figure(
                        5.0,
                        "billion",
                        18,
                        "Future Energy Fund",
                        "Future Energy Fund with an initial injection of $5.0 billion",
                    )
                ],
            ),
        ],
        answer="Revenue is $108.6 billion (p.13); the Fund receives $5.0 billion (p.18). "
        "The document does not identify a specific revenue stream as funding it.",
    )


def test_routing_passes_when_expected_agents_ran(good_trace):
    """Both agents were invoked, as the query required."""
    check = check_routing(good_trace, expected=["revenue_agent", "expenditure_agent"])

    assert check.passed


def test_routing_fails_when_an_agent_was_missed(good_trace):
    """A two-part query answered by one agent is a routing failure."""
    trace = Trace(
        query=good_trace.query,
        decisions=good_trace.decisions,
        findings=good_trace.findings[:1],
        answer=good_trace.answer,
    )

    check = check_routing(trace, expected=["revenue_agent", "expenditure_agent"])

    assert not check.passed
    assert "expenditure_agent" in check.detail


def test_routing_fails_when_an_agent_ran_unnecessarily(good_trace):
    """An out-of-scope query that invoked an agent has over-delegated."""
    check = check_routing(good_trace, expected=[])

    assert not check.passed


def test_figures_pass_when_present_with_the_right_unit(good_trace):
    """The required values appear, correctly scaled."""
    check = check_figures(good_trace, required=[{"value": 5.0, "unit": "billion"}])

    assert check.passed


def test_figures_fail_on_a_thousandfold_unit_error(good_trace):
    """5000 labelled billions is the 1000x error Part 1 documented.

    The value looks plausible and the unit is valid, so nothing but this check
    would catch it.
    """
    trace = Trace(
        query=good_trace.query,
        decisions=good_trace.decisions,
        findings=[
            _finding(
                "expenditure_agent",
                [16, 18, 20],
                [
                    _figure(
                        5000,
                        "billion",
                        20,
                        "Future Energy Fund",
                        "Future Energy Fund 5,000",
                    )
                ],
            )
        ],
        answer="",
    )

    check = check_figures(trace, required=[{"value": 5.0, "unit": "billion"}])

    assert not check.passed


def test_figures_accept_either_valid_unit_form(good_trace):
    """5.0 billion and 5,000 million are the same amount, both correct.

    Which is right depends on the page cited, so an alternatives list passes
    when any one of them is present.
    """
    trace = Trace(
        query=good_trace.query,
        decisions=good_trace.decisions,
        findings=[
            _finding(
                "expenditure_agent",
                [16, 18, 20],
                [
                    _figure(
                        5000,
                        "million",
                        20,
                        "Future Energy Fund",
                        "Future Energy Fund 5,000",
                    )
                ],
            )
        ],
        answer="",
    )

    check = check_figures(
        trace,
        required=[],
        any_of=[[{"value": 5.0, "unit": "billion"}, {"value": 5000, "unit": "million"}]],
    )

    assert check.passed


def test_traceability_passes_when_quotes_are_on_their_pages(good_trace):
    """Each quote is found in the text of the page it cites."""
    pages = {
        13: "Estimated FY2024 Operating Revenue is $108.6 billion (15.1% of GDP).",
        18: "The Government will establish the Future Energy Fund with an "
        "initial injection of $5.0 billion.",
    }

    check = check_traceability(good_trace, page_text=pages)

    assert check.passed


def test_traceability_fails_when_a_quote_is_not_on_its_page(good_trace):
    """A quote attributed to the wrong page is caught.

    The model can read the right sentence and cite the wrong page - the value
    is then correct but unverifiable, which this check treats as a failure.
    """
    pages = {
        13: "Estimated FY2024 Operating Revenue is $108.6 billion.",
        18: "Unrelated text.",
    }

    check = check_traceability(good_trace, page_text=pages)

    assert not check.passed
    assert "p.18" in check.detail


def test_traceability_ignores_whitespace_differences(good_trace):
    """Pypdf spaces text irregularly, so matching is whitespace-normalised."""
    pages = {
        13: "Estimated  FY2024\nOperating Revenue is\n$108.6 billion",
        18: "Future Energy   Fund with an initial\ninjection of $5.0 billion",
    }

    assert check_traceability(good_trace, page_text=pages).passed


def test_page_discipline_passes_when_agents_stay_in_scope(good_trace, config):
    """Every figure came from a page its agent was given."""
    assert check_page_discipline(good_trace, config).passed


def test_page_discipline_fails_on_a_page_outside_the_set(config):
    """A figure from an unread page is fabricated, full stop."""
    trace = Trace(
        query="q",
        findings=[
            _finding(
                "expenditure_agent",
                [16, 18, 20],
                [_figure(108.6, "billion", 13, "Operating Revenue", "on page 13")],
            )
        ],
    )

    check = check_page_discipline(trace, config)

    assert not check.passed
    assert "13" in check.detail


def test_score_part3_reports_every_check(good_trace, config):
    """The report covers all four checks in one table."""
    pages = {
        13: "Estimated FY2024 Operating Revenue is $108.6 billion",
        18: "Future Energy Fund with an initial injection of $5.0 billion",
    }

    report = score_part3(
        good_trace,
        expected={
            "routed_to": ["revenue_agent", "expenditure_agent"],
            "figures": [{"value": 5.0, "unit": "billion"}],
        },
        config=config,
        page_text=pages,
    )

    assert len(report.checks) == 4
    assert report.passed


def test_declined_query_scores_as_correctly_routed(config):
    """Declining an out-of-scope query is a pass, not an absence of routing."""
    trace = Trace(
        query="What is the capital of France?",
        decisions=[
            Decision(
                turn=1,
                reasoning="not a budget question",
                chose="out_of_scope",
                routed_to="out_of_scope",
            )
        ],
        findings=[],
        answer="This question cannot be answered from the document.",
        declined=True,
    )

    report = score_part3(trace, expected={"routed_to": []}, config=config, page_text={})

    assert report.passed
