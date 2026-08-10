"""The supervisor: decides which agent works next, or that the work is done.

The model chooses; deterministic code guards the choice. This is the same split
as Part 2, where the LLM classifies a date and a tool checks the arithmetic -
judgement where judgement is needed, code where there is one right answer.

Three guards, all of which record themselves in the trace when they fire:

  1. no agent runs twice - its pages have not changed, so a second pass would
     read identical text;
  2. a hard turn cap, so the graph provably terminates;
  3. no synthesis before any agent has reported, which would produce an
     ungrounded answer.

A guarded route is never presented as a decision: `Decision` keeps both what
the model chose and what actually ran.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.extraction.prompts import load_prompt
from src.graph.state import AGENTS, Decision, Finding, Route

# Which agent to fall back to when the model tries to synthesise with nothing
# in hand. Keyed by a word that suggests the subject.
_EXPENDITURE_HINTS = ("fund", "spend", "expenditure", "top-up", "top up", "transfer")


class RouteDecision(BaseModel):
    """The supervisor's choice for one turn.

    `reasoning` is declared before `next` so the model states why before it
    commits. With two agents and an obviously two-part query, a coin flip looks
    defensible about half the time - the rationale is what distinguishes a
    decision from a guess.
    """

    reasoning: str = Field(
        description="Why this route. Name the part of the query being addressed "
        "and why that agent covers it. State this before choosing."
    )
    next: Route = Field(
        description="revenue_agent, expenditure_agent, synthesis, or "
        "out_of_scope if the document cannot answer the query at all."
    )
    sub_task: str = Field(
        default="",
        description="The specific question for the chosen agent. Empty when "
        "routing to synthesis or out_of_scope.",
    )


def _fallback_agent(query: str, visited: list[str]) -> str:
    """Pick an agent when the model tried to synthesise with no findings."""
    remaining = [agent for agent in AGENTS if agent not in visited]
    if not remaining:
        return "synthesis"

    lowered = query.lower()
    if any(hint in lowered for hint in _EXPENDITURE_HINTS):
        preferred = "expenditure_agent"
    else:
        preferred = "revenue_agent"

    return preferred if preferred in remaining else remaining[0]


def route(
    decision: RouteDecision,
    visited: list[str],
    findings: list[Finding],
    turn: int,
    config,
    query: str = "",
) -> tuple[str, Decision]:
    """Apply the guards to the model's choice.

    Returns the node to run and the trace record for this turn.
    """
    chose = decision.next
    routed_to = chose
    overridden = None

    if chose in visited:
        routed_to = "synthesis"
        overridden = f"{chose} has already reported; its pages have not changed"

    elif turn >= config.max_turns and chose in AGENTS:
        routed_to = "synthesis"
        overridden = f"turn cap of {config.max_turns} reached"

    elif chose == "synthesis" and not findings:
        routed_to = _fallback_agent(query, visited)
        overridden = "nothing to synthesise; no agent has reported yet"

    record = Decision(
        turn=turn,
        reasoning=decision.reasoning,
        chose=chose,
        routed_to=routed_to,
        overridden=overridden,
        sub_task=decision.sub_task,
    )

    return routed_to, record


def decide(state, model, config) -> tuple[str, Decision]:
    """Ask the model where to go next, then guard its answer."""
    findings = state.get("findings", [])
    visited = state.get("visited", [])
    turn = len(state.get("decisions", [])) + 1

    prompt = load_prompt("supervisor")
    # JSON mode rather than tool-calling. `reasoning` asks for a paragraph
    # before the choice, and over that length the model's tool-call wrapper
    # drifts from the format Groq's parser accepts - it emitted
    # "<function=RouteDecision> {...}" where the parser wanted no space, and
    # the request was rejected despite the decision itself being correct.
    # JSON mode has no wrapper to misparse; Pydantic still validates.
    chain = prompt | model.with_structured_output(RouteDecision, method="json_mode")

    decision = chain.invoke(
        {
            "query": state["query"],
            "visited": ", ".join(visited) or "none",
            "findings": _render_findings(findings),
            "revenue_pages": _render_pages(config, "revenue"),
            "expenditure_pages": _render_pages(config, "expenditure"),
        }
    )

    return route(decision, visited, findings, turn, config, state["query"])


def _render_pages(config, agent: str) -> str:
    """Render an agent's pages for the prompt."""
    return ", ".join(str(page) for page in config.pages_for_agent(agent))


def _render_findings(findings: list[Finding]) -> str:
    """Render findings for the supervisor, summaries only.

    Figures are omitted deliberately: the supervisor decides who works next and
    does not need the numbers, so sending them would cost tokens on every turn
    for no decision it could make differently.
    """
    if not findings:
        return "(none yet)"

    return "\n\n".join(
        f"{finding.agent} (pages {finding.pages_read}):\n{finding.summary}"
        for finding in findings
    )
