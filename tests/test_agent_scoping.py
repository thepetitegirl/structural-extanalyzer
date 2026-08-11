"""Each agent reads only its own pages.

These are the highest-value tests in Part 3. Page scoping is what makes the two
agents genuinely complementary - if the expenditure agent could read page 13 it
could answer the revenue half of the query alone, and the collaboration would
be theatre.

Scoping is also the main cost control: the whole document is ~14.8k tokens
against ~1.6k for an agent's page set.
"""

import pytest

from src.agents.base import run_agent
from src.graph.state import Figure, Finding


def _report(agent="revenue_agent", pages=(13,)):
    """A plausible agent response."""
    return Finding(
        agent=agent,
        sub_task="Identify revenue streams",
        summary="Operating Revenue is $108.6 billion.",
        figures=[
            Figure(
                value=108.6,
                unit="billion",
                page=13,
                label="Operating Revenue",
                quote="Estimated FY2024 Operating Revenue is $108.6 billion",
            )
        ],
        pages_read=list(pages),
    )


def test_revenue_agent_reads_only_its_configured_pages(config, fake_pages, request):
    """The revenue agent asks for exactly the pages config gives it."""
    from tests.conftest import ScriptedModel

    model = ScriptedModel([_report()])

    run_agent("revenue", "any.pdf", "Identify revenue streams", model, config)

    assert fake_pages[0][1] == [9, 13, 15]


def test_expenditure_agent_reads_only_its_configured_pages(config, fake_pages):
    """The expenditure agent asks for exactly its own pages."""
    from tests.conftest import ScriptedModel

    model = ScriptedModel([_report(agent="expenditure_agent", pages=(18,))])

    run_agent("expenditure", "any.pdf", "Find the fund", model, config)

    assert fake_pages[0][1] == [16, 18, 20]


def test_expenditure_agent_never_sees_page_13(config, fake_pages):
    """Page 13 is revenue-only, and that is what forces collaboration.

    It carries both the revenue total and the top-ups sentence. If the
    expenditure agent could read it, it could answer the combined query alone.
    """
    from tests.conftest import ScriptedModel

    model = ScriptedModel([_report(agent="expenditure_agent", pages=(18,))])

    run_agent("expenditure", "any.pdf", "Find the fund", model, config)

    assert 13 not in fake_pages[0][1]
    assert "page 13" not in model.prompts[0]


def test_revenue_agent_never_sees_the_fund_pages(config, fake_pages):
    """The Future Energy Fund pages are expenditure-only, symmetrically."""
    from tests.conftest import ScriptedModel

    model = ScriptedModel([_report()])

    run_agent("revenue", "any.pdf", "Identify revenue streams", model, config)

    assert not {16, 18, 20} & set(fake_pages[0][1])


def test_prompt_stays_within_its_character_budget(config, fake_pages):
    """A widened page set fails loudly rather than quietly costing tokens.

    Without this, adding pages "to be safe" would silently multiply the cost of
    every agent call, and nothing in the output would show it.
    """
    from tests.conftest import ScriptedModel

    model = ScriptedModel([_report()])

    run_agent("revenue", "any.pdf", "Identify revenue streams", model, config)

    assert len(model.prompts[0]) < config.prompt_character_budget


def test_finding_records_the_pages_that_were_read(config, fake_pages):
    """The finding reports its own page set, so the trace can check it."""
    from tests.conftest import ScriptedModel

    model = ScriptedModel([_report()])

    finding = run_agent("revenue", "any.pdf", "Identify revenue streams", model, config)

    assert finding.pages_read == [9, 13, 15]


def test_sub_task_reaches_the_agent(config, fake_pages):
    """The supervisor's brief is what the agent is asked, not the raw query."""
    from tests.conftest import ScriptedModel

    model = ScriptedModel([_report()])

    run_agent("revenue", "any.pdf", "Identify FY2024 revenue streams", model, config)

    assert "Identify FY2024 revenue streams" in model.prompts[0]


def test_unknown_agent_raises(config, fake_pages):
    """An agent with no configured pages is a configuration error."""
    from tests.conftest import ScriptedModel

    with pytest.raises(Exception, match="nonexistent"):
        run_agent("nonexistent", "any.pdf", "task", ScriptedModel([]), config)
