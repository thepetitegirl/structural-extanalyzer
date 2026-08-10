"""Unit tests for the two specialist agents.

`test_agent_scoping.py` covers which pages an agent may read. These cover what
each agent does with them: that it is given its own prompt, reports findings
under its own name, and carries provenance through unchanged.

The model is stubbed throughout.
"""

from src.agents.base import AgentReport, run_agent
from src.agents.expenditure_agent import NODE_NAME as EXPENDITURE_NODE
from src.agents.expenditure_agent import expenditure_node
from src.agents.revenue_agent import NODE_NAME as REVENUE_NODE
from src.agents.revenue_agent import revenue_node
from src.graph.state import Figure


def _report(figures=None, summary="Operating Revenue is $108.6 billion."):
    """What an agent's model call returns."""
    return AgentReport(
        summary=summary,
        figures=figures
        if figures is not None
        else [
            Figure(
                value=108.6,
                unit="billion",
                page=13,
                label="Operating Revenue",
                quote="Estimated FY2024 Operating Revenue is $108.6 billion",
            )
        ],
    )


def _model(*reports):
    """A model returning the given reports in order."""
    from tests.conftest import ScriptedModel

    return ScriptedModel(list(reports))


# --- revenue agent ---------------------------------------------------------


def test_revenue_node_names_itself_in_the_finding(config, fake_pages):
    """The finding says which agent produced it, so the trace can group them."""
    update = revenue_node(
        {"query": "q", "sub_task": "Identify revenue streams"},
        _model(_report()),
        config,
        "any.pdf",
    )

    assert update["findings"][0].agent == REVENUE_NODE


def test_revenue_node_marks_itself_visited(config, fake_pages):
    """Visiting is recorded so the supervisor will not send it twice."""
    update = revenue_node(
        {"query": "q", "sub_task": "t"}, _model(_report()), config, "any.pdf"
    )

    assert update["visited"] == [REVENUE_NODE]


def test_revenue_agent_uses_the_revenue_prompt(config, fake_pages):
    """Each agent gets its own prompt, not a shared generic one."""
    model = _model(_report())

    revenue_node({"query": "q", "sub_task": "t"}, model, config, "any.pdf")

    assert "revenue" in model.prompts[0].lower()
    assert "Total Expenditure" not in model.prompts[0]


def test_revenue_agent_falls_back_to_the_query_without_a_sub_task(config, fake_pages):
    """A missing brief is not a crash; the raw query is used instead."""
    model = _model(_report())

    revenue_node({"query": "What are the revenue streams?"}, model, config, "any.pdf")

    assert "What are the revenue streams?" in model.prompts[0]


# --- expenditure agent -----------------------------------------------------


def test_expenditure_node_names_itself_in_the_finding(config, fake_pages):
    """The expenditure agent reports under its own name."""
    update = expenditure_node(
        {"query": "q", "sub_task": "Find the fund"},
        _model(_report(summary="The Future Energy Fund receives $5.0 billion.")),
        config,
        "any.pdf",
    )

    assert update["findings"][0].agent == EXPENDITURE_NODE


def test_expenditure_agent_uses_the_expenditure_prompt(config, fake_pages):
    """Its prompt covers spending, and warns about the unit trap."""
    model = _model(_report())

    expenditure_node({"query": "q", "sub_task": "t"}, model, config, "any.pdf")

    prompt = model.prompts[0]
    assert "spending" in prompt.lower()
    assert "million" in prompt.lower()


def test_expenditure_prompt_warns_about_scale(config, fake_pages):
    """The 1000x error is named explicitly.

    The Future Energy Fund appears on this agent's pages as 5.00 (billion) and
    5,000 (million). Reporting the second as billions overstates it a
    thousandfold, and nothing in the number reveals the error.
    """
    model = _model(_report())

    expenditure_node({"query": "q", "sub_task": "t"}, model, config, "any.pdf")

    assert "thousandfold" in model.prompts[0].lower()


# --- shared behaviour ------------------------------------------------------


def test_figures_pass_through_unchanged(config, fake_pages):
    """An agent does not alter the values its model reported.

    Provenance only means something if the finding carries exactly what was
    read - a unit quietly normalised here would defeat the traceability check.
    """
    figure = Figure(
        value=5000,
        unit="million",
        page=20,
        label="Future Energy Fund",
        quote="Future Energy Fund 5,000",
    )

    finding = run_agent("expenditure", "any.pdf", "t", _model(_report([figure])), config)

    assert finding.figures[0].value == 5000
    assert finding.figures[0].unit == "million"


def test_agent_reporting_no_figures_is_valid(config, fake_pages):
    """An honest empty answer beats a fabricated figure.

    Both prompts tell the agent to say so when its pages do not cover the
    sub-task, so a finding with no figures must be representable.
    """
    finding = run_agent(
        "revenue",
        "any.pdf",
        "How much went to the Future Energy Fund?",
        _model(_report([], summary="These pages do not cover fund allocations.")),
        config,
    )

    assert finding.figures == []
    assert "do not cover" in finding.summary


def test_sub_task_is_recorded_on_the_finding(config, fake_pages):
    """The trace shows what each agent was asked, not just what it answered."""
    finding = run_agent(
        "revenue", "any.pdf", "Identify FY2024 revenue streams", _model(_report()), config
    )

    assert finding.sub_task == "Identify FY2024 revenue streams"


def test_both_agents_share_one_implementation(config, fake_pages):
    """The agents differ by config and prompt, not by duplicated logic.

    If they diverge, this is where it shows: both wrappers call run_agent, so a
    change to one cannot silently miss the other.
    """
    revenue_update = revenue_node(
        {"query": "q", "sub_task": "t"}, _model(_report()), config, "any.pdf"
    )
    expenditure_update = expenditure_node(
        {"query": "q", "sub_task": "t"}, _model(_report()), config, "any.pdf"
    )

    assert set(revenue_update) == set(expenditure_update) == {"findings", "visited"}
