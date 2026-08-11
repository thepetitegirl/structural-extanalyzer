"""Supervisor routing and its guards.

The model chooses; deterministic code prevents non-termination. These tests
cover both, and in particular that an override is recorded rather than
silently applied - a forced route must never be presented as a decision.
"""

from src.agents.supervisor import RouteDecision, decide, route
from src.graph.state import Figure, Finding
from tests.conftest import ScriptedModel


def _finding(agent="revenue_agent"):
    """A finding from an agent that has already reported."""
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
        pages_read=[9, 13, 15],
    )


def _decision(next_node, reasoning="because", sub_task="do the thing"):
    """What the model returns."""
    return RouteDecision(reasoning=reasoning, next=next_node, sub_task=sub_task)


def test_routes_where_the_model_chose(config):
    """With no guard triggered, the model's choice stands."""
    decision, record = route(
        _decision("revenue_agent"), visited=[], findings=[], turn=1, config=config
    )

    assert decision == "revenue_agent"
    assert not record.was_overridden


def test_records_the_models_reasoning(config):
    """The stated rationale survives into the trace.

    Without it, a two-agent graph routing correctly is indistinguishable from
    a coin flip that happened to land well.
    """
    _, record = route(
        _decision("revenue_agent", reasoning="The query asks which streams fund it"),
        visited=[],
        findings=[],
        turn=1,
        config=config,
    )

    assert "streams fund it" in record.reasoning


def test_will_not_revisit_an_agent(config):
    """An agent that has reported is not sent again.

    Its pages have not changed, so a second pass would read identical text.
    """
    decision, record = route(
        _decision("revenue_agent"),
        visited=["revenue_agent"],
        findings=[_finding()],
        turn=2,
        config=config,
    )

    assert decision == "synthesis"
    assert record.was_overridden
    assert record.chose == "revenue_agent"
    assert record.routed_to == "synthesis"


def test_stops_at_max_turns(config):
    """The turn cap forces synthesis, so the graph provably terminates."""
    decision, record = route(
        _decision("expenditure_agent"),
        visited=["revenue_agent"],
        findings=[_finding()],
        turn=config.max_turns,
        config=config,
    )

    assert decision == "synthesis"
    assert "turn" in (record.overridden or "").lower()


def test_will_not_synthesise_with_no_findings(config):
    """Synthesising nothing would produce an ungrounded answer."""
    decision, record = route(
        _decision("synthesis"), visited=[], findings=[], turn=1, config=config
    )

    assert decision in ("revenue_agent", "expenditure_agent")
    assert record.was_overridden


def test_fallback_prefers_expenditure_when_the_query_hints_at_spending(config):
    """A hint word in the query steers which agent the guard falls back to."""
    decision, _ = route(
        _decision("synthesis"),
        visited=[],
        findings=[],
        turn=1,
        config=config,
        query="How will the Future Energy Fund be supported?",
    )

    assert decision == "expenditure_agent"


def test_fallback_prefers_revenue_without_a_spending_hint(config):
    """With no hint word the fallback goes to revenue."""
    decision, _ = route(
        _decision("synthesis"),
        visited=[],
        findings=[],
        turn=1,
        config=config,
        query="What are the key government income streams?",
    )

    assert decision == "revenue_agent"


def test_out_of_scope_is_not_overridden_by_the_empty_findings_guard(config):
    """Declining a query is a valid first decision, not a failure to gather.

    The empty-findings guard exists to stop premature synthesis; it must not
    force an agent to run on a question the document cannot answer.
    """
    decision, record = route(
        _decision("out_of_scope"), visited=[], findings=[], turn=1, config=config
    )

    assert decision == "out_of_scope"
    assert not record.was_overridden


def test_second_agent_is_reachable_after_the_first(config):
    """The unvisited agent can still be routed to."""
    decision, record = route(
        _decision("expenditure_agent"),
        visited=["revenue_agent"],
        findings=[_finding()],
        turn=2,
        config=config,
    )

    assert decision == "expenditure_agent"
    assert not record.was_overridden


def test_override_names_the_guard_that_fired(config):
    """The trace says which rule intervened, not merely that one did."""
    _, record = route(
        _decision("revenue_agent"),
        visited=["revenue_agent"],
        findings=[_finding()],
        turn=2,
        config=config,
    )

    assert record.overridden
    assert "already" in record.overridden.lower()


def test_turn_number_is_recorded(config):
    """Each decision knows which turn it was."""
    _, record = route(
        _decision("revenue_agent"), visited=[], findings=[], turn=3, config=config
    )

    assert record.turn == 3


def test_sub_task_is_carried_into_the_record(config):
    """The brief handed to the agent is part of the trace."""
    _, record = route(
        _decision("revenue_agent", sub_task="Identify FY2024 revenue streams"),
        visited=[],
        findings=[],
        turn=1,
        config=config,
    )

    assert record.sub_task == "Identify FY2024 revenue streams"


def test_decide_fills_empty_slots_with_placeholders(config):
    """On the first turn the prompt says so, rather than showing empty slots."""
    model = ScriptedModel([_decision("revenue_agent")])

    decide({"query": "What are the revenue streams?"}, model, config)

    assert "Agents already consulted: none" in model.prompts[0]
    assert "(none yet)" in model.prompts[0]


def test_decide_numbers_the_turn_from_decisions_so_far(config):
    """Two decisions recorded means this is turn three."""
    model = ScriptedModel([_decision("synthesis")])
    state = {
        "query": "q",
        "findings": [_finding()],
        "visited": ["revenue_agent"],
        "decisions": ["turn 1", "turn 2"],
    }

    _, record = decide(state, model, config)

    assert record.turn == 3


def test_decide_requests_route_decisions_in_json_mode(config):
    """The schema is RouteDecision and the method json_mode.

    json_mode is load-bearing: Groq's parser rejects the tool-call wrapper
    over a long reasoning field, so losing it breaks live runs only.
    """
    model = ScriptedModel([_decision("revenue_agent")])

    decide({"query": "q"}, model, config)

    assert model.schemas == [RouteDecision]
    assert model.methods == ["json_mode"]
