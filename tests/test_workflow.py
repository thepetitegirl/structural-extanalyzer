"""The supervisor graph, end to end with a scripted model.

Every path a demo query can take is scripted here, so routing is asserted
without spending budget. The scripts double as documentation of what each
query is expected to do.
"""

import pytest
from langchain_core.messages import AIMessage

from src.agents.base import AgentReport
from src.agents.supervisor import RouteDecision
from src.graph.state import Figure
from src.graph.workflow import DECLINE_MESSAGE, build_graph, run_query


def _figure(value, page, label, unit="billion"):
    """A figure an agent might report."""
    return Figure(
        value=value,
        unit=unit,
        page=page,
        label=label,
        quote=f"{label} is ${value} {unit}",
    )


def _revenue_report():
    """What the revenue agent returns."""
    return AgentReport(
        summary="Operating Revenue is $108.6 billion; NIRC adds $23.5 billion.",
        figures=[
            _figure(108.6, 13, "Operating Revenue"),
            _figure(23.5, 15, "NIRC"),
        ],
    )


def _expenditure_report():
    """What the expenditure agent returns."""
    return AgentReport(
        summary="The Future Energy Fund receives an initial injection of $5.0 billion.",
        figures=[_figure(5.0, 18, "Future Energy Fund")],
    )


REQUIRED_QUERY = (
    "What are the key government revenue streams, and how will the Budget "
    "for the Future Energy Fund be supported?"
)


@pytest.fixture
def both_agents_script():
    """The required query: revenue, then expenditure, then synthesis."""
    return [
        RouteDecision(
            reasoning="Query has two parts; the first asks which streams fund the Budget.",
            next="revenue_agent",
            sub_task="Identify FY2024 revenue streams",
        ),
        _revenue_report(),
        RouteDecision(
            reasoning="Revenue established. The Fund is a spending item.",
            next="expenditure_agent",
            sub_task="Find the Future Energy Fund allocation",
        ),
        _expenditure_report(),
        RouteDecision(
            reasoning="Both parts covered; no gap left.",
            next="synthesis",
            sub_task="",
        ),
        AIMessage(
            content="Revenue is $108.6 billion (p.13). The Fund gets $5.0 billion (p.18)."
        ),
    ]


def _run(script, config, fake_pages, query=REQUIRED_QUERY):
    """Run the graph with a scripted model."""
    from tests.conftest import ScriptedModel

    model = ScriptedModel(script)
    graph = build_graph(model, config, "any.pdf")
    return graph.invoke({"query": query}), model


def test_both_agents_run_for_the_required_query(config, fake_pages, both_agents_script):
    """The required query invokes revenue and expenditure, in that order."""
    final, _ = _run(both_agents_script, config, fake_pages)

    assert final["visited"] == ["revenue_agent", "expenditure_agent"]


def test_findings_accumulate_rather_than_overwrite(
    config, fake_pages, both_agents_script
):
    """Both agents' findings survive to synthesis.

    Guards the append-only reducer: without it the second agent's finding
    would replace the first, and half the answer would vanish silently.
    """
    final, _ = _run(both_agents_script, config, fake_pages)

    assert len(final["findings"]) == 2


def test_every_decision_is_recorded(config, fake_pages, both_agents_script):
    """Three supervisor turns produce three trace entries."""
    final, _ = _run(both_agents_script, config, fake_pages)

    assert len(final["decisions"]) == 3


def test_supervisor_sees_the_first_agents_findings(
    config, fake_pages, both_agents_script
):
    """Turn 2 is a decision made in light of turn 1's result.

    This is what makes the agents collaborative rather than independent: the
    second brief can be written knowing what the first agent found.
    """
    _, model = _run(both_agents_script, config, fake_pages)

    second_supervisor_prompt = model.prompts[2]
    assert "Operating Revenue is $108.6 billion" in second_supervisor_prompt


def test_synthesis_receives_findings_not_page_text(
    config, fake_pages, both_agents_script
):
    """Synthesis sees what agents reported, never the document.

    This is both the cost guarantee and what makes traceability meaningful: it
    cannot introduce a figure no agent cited.
    """
    _, model = _run(both_agents_script, config, fake_pages)

    synthesis_prompt = model.prompts[-1]
    assert "Operating Revenue" in synthesis_prompt
    assert "--- page" not in synthesis_prompt


def test_single_agent_query_skips_the_other(config, fake_pages):
    """A revenue-only query never invokes the expenditure agent."""
    script = [
        RouteDecision(
            reasoning="Purely a revenue question.",
            next="revenue_agent",
            sub_task="Identify revenue streams",
        ),
        _revenue_report(),
        RouteDecision(reasoning="Answered.", next="synthesis", sub_task=""),
        AIMessage(content="Operating Revenue is $108.6 billion (p.13)."),
    ]

    final, _ = _run(script, config, fake_pages, "What are the revenue streams?")

    assert final["visited"] == ["revenue_agent"]


def test_out_of_scope_query_invokes_no_agent(config, fake_pages):
    """A question the document cannot answer is declined in one turn."""
    script = [
        RouteDecision(
            reasoning="Geography, not a budget question.",
            next="out_of_scope",
            sub_task="",
        )
    ]

    final, _ = _run(script, config, fake_pages, "What is the capital of France?")

    assert final.get("visited", []) == []
    assert final["declined"]
    assert final["answer"] == DECLINE_MESSAGE


def test_decline_does_not_call_the_model_again(config, fake_pages):
    """Declining is a fixed message, not a model call.

    The model knows the capital of France. Letting it answer would produce
    exactly the ungrounded output the rest of the system prevents.
    """
    script = [
        RouteDecision(
            reasoning="Not a budget question.", next="out_of_scope", sub_task=""
        )
    ]

    _, model = _run(script, config, fake_pages, "What is the capital of France?")

    assert len(model.received) == 1


def test_nonsense_query_is_declined(config, fake_pages):
    """Unparseable input is declined rather than answered."""
    script = [
        RouteDecision(reasoning="Not a question.", next="out_of_scope", sub_task="")
    ]

    final, _ = _run(script, config, fake_pages, "asdfgh qwerty")

    assert final["declined"]


def test_repeat_agent_choice_is_overridden_to_synthesis(config, fake_pages):
    """A model looping on one agent is stopped, and the trace says so."""
    script = [
        RouteDecision(reasoning="Revenue first.", next="revenue_agent", sub_task="a"),
        _revenue_report(),
        RouteDecision(reasoning="More revenue.", next="revenue_agent", sub_task="b"),
        AIMessage(content="Operating Revenue is $108.6 billion (p.13)."),
    ]

    final, _ = _run(script, config, fake_pages, "What are the revenue streams?")

    assert final["visited"] == ["revenue_agent"]
    assert final["decisions"][-1].was_overridden


def test_graph_has_the_expected_nodes(config):
    """Catches a typo'd node name without running anything."""
    from tests.conftest import ScriptedModel

    graph = build_graph(ScriptedModel([]), config, "any.pdf")
    nodes = set(graph.get_graph().nodes)

    assert {
        "supervisor",
        "revenue_agent",
        "expenditure_agent",
        "synthesis",
        "decline",
    } <= nodes


def test_costs_are_recorded_per_node(config, fake_pages, both_agents_script):
    """Every node reports what it cost, for the daily quota."""
    final, _ = _run(both_agents_script, config, fake_pages)

    nodes = {cost.node for cost in final["costs"]}
    assert "supervisor" in nodes
    assert "revenue_agent" in nodes


def test_run_query_returns_a_populated_trace(config, fake_pages, both_agents_script):
    """run_query wraps the graph result in a Trace."""
    from tests.conftest import ScriptedModel

    trace = run_query(
        REQUIRED_QUERY,
        model=ScriptedModel(both_agents_script),
        config=config,
        pdf_path="any.pdf",
    )

    assert trace.agents_invoked == ["revenue_agent", "expenditure_agent"]
    assert len(trace.citations()) == 3
    assert trace.answer
