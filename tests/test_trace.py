"""The supervisor's decision trace.

The requirement names this a deliverable, so it is tested like one. What
separates a decision trace from a route log is that it records why a route was
taken, and when the system overruled the model - both are asserted here.

No model is involved: a Trace is built by hand and rendered.
"""

from src.graph.state import Decision, Figure, Finding, NodeCost
from src.graph.trace import Trace


def _decision(turn, chose, routed_to=None, overridden=None, reasoning="because"):
    """One supervisor turn."""
    return Decision(
        turn=turn,
        reasoning=reasoning,
        chose=chose,
        routed_to=routed_to or chose,
        overridden=overridden,
        sub_task="do the thing",
    )


def _finding(agent, page, value, label):
    """One agent's report."""
    return Finding(
        agent=agent,
        sub_task="find it",
        summary=f"{label} is {value}.",
        figures=[
            Figure(
                value=value,
                unit="billion",
                page=page,
                label=label,
                quote=f"{label} is ${value} billion",
            )
        ],
        pages_read=[page],
    )


def _trace(**overrides):
    """A complete two-agent trace."""
    defaults = dict(
        query="What are the revenue streams, and how is the Fund supported?",
        decisions=[
            _decision(1, "revenue_agent", reasoning="First part asks about revenue"),
            _decision(2, "expenditure_agent", reasoning="The Fund is spending"),
            _decision(3, "synthesis", reasoning="Both parts covered"),
        ],
        findings=[
            _finding("revenue_agent", 13, 108.6, "Operating Revenue"),
            _finding("expenditure_agent", 18, 5.0, "Future Energy Fund"),
        ],
        answer="Revenue is $108.6 billion (p.13). The Fund receives $5.0 billion (p.18).",
        costs=[
            NodeCost(node="supervisor", input_tokens=350, output_tokens=120, seconds=1.1)
        ],
    )
    defaults.update(overrides)
    return Trace(**defaults)


def test_every_routing_decision_is_recorded():
    """A three-hop run yields three decisions."""
    assert len(_trace().decisions) == 3


def test_table_shows_the_stated_reasoning():
    """Why the supervisor routed is visible, not just where."""
    rendered = _trace().table()

    assert "First part asks about revenue" in rendered


def test_table_shows_an_override():
    """When a guard overrules the model, both choices appear.

    A forced route presented as a decision would misrepresent the system.
    """
    trace = _trace(
        decisions=[
            _decision(1, "revenue_agent"),
            _decision(
                2,
                "revenue_agent",
                routed_to="synthesis",
                overridden="revenue_agent has already reported",
            ),
        ]
    )

    rendered = trace.table()
    assert "revenue_agent" in rendered
    assert "synthesis" in rendered
    assert "already reported" in rendered


def test_summary_counts_decisions_agents_and_figures():
    """One line stating what happened."""
    summary = _trace().summary()

    assert "3 decision" in summary
    assert "2 agent" in summary
    assert "2 figure" in summary


def test_summary_reports_overrides():
    """An overridden run says so in its summary."""
    trace = _trace(
        decisions=[
            _decision(
                1, "synthesis", routed_to="revenue_agent", overridden="no findings"
            ),
        ]
    )

    assert "1 override" in trace.summary()


def test_agents_invoked_lists_only_agents():
    """Synthesis is a node but not an agent."""
    assert _trace().agents_invoked == ["revenue_agent", "expenditure_agent"]


def test_citations_carry_page_value_and_quote():
    """Every figure is traceable back to its source text."""
    citations = _trace().citations()

    assert len(citations) == 2
    assert all(citation.page and citation.quote for citation in citations)


def test_render_includes_query_answer_and_citations():
    """The full rendering is self-contained."""
    rendered = _trace().render()

    assert "revenue streams" in rendered
    assert "108.6" in rendered
    assert "p.18" in rendered


def test_costs_render_with_a_total():
    """Token and time costs are reported, given the daily quota."""
    rendered = _trace().costs_table()

    assert "supervisor" in rendered
    assert "350" in rendered
    assert "total" in rendered.lower()


def test_trace_without_costs_still_renders():
    """Cost capture is optional; its absence must not break the trace."""
    assert _trace(costs=[]).render()


def test_declined_query_renders_without_findings():
    """An out-of-scope query has a trace too: one decision, no agents."""
    trace = _trace(
        decisions=[_decision(1, "out_of_scope", reasoning="Not a budget question")],
        findings=[],
        answer="The document does not cover this.",
    )

    rendered = trace.render()
    assert "out_of_scope" in rendered
    assert trace.agents_invoked == []
