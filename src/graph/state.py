"""State passed between nodes of the supervisor graph.

Three design decisions worth naming:

**Findings, decisions and visited use append-only reducers.** A node returns
`{"findings": [one]}` and LangGraph appends it. No node can erase another's
work, so the trace is complete by construction rather than by discipline.

**Page text never enters state.** Each agent loads its pages inside its own
node and discards them there, passing on a structured `Finding`. A message list
would accumulate the document and re-send it on every hop, which is the single
largest avoidable cost in the graph.

**Figures reuse Part 1's provenance rules.** `Figure` inherits the non-blank
quote validator and the million/billion unit constraint from
`src/extraction/schemas.py`, so "every number is traceable to a page" has one
implementation across all three parts.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from src.extraction.schemas import _Cited

# Part 1's fields were all monetary, so its Unit allows only million and
# billion. An agent reading the revenue breakdown chart on page 9 finds shares
# ("Corporate Income Tax 27.2%"), which are legitimate figures with no monetary
# unit - so percent is added here rather than widening Part 1's constraint,
# where it would weaken a check that is doing real work.
AgentUnit = Literal["million", "billion", "percent"]

# The nodes a supervisor may route to. `out_of_scope` lets it decline a query
# the document cannot answer rather than being forced to pick an agent.
Route = Literal["revenue_agent", "expenditure_agent", "synthesis", "out_of_scope"]

AGENTS = ("revenue_agent", "expenditure_agent")


class Figure(_Cited):
    """A figure an agent read, with the unit and page it came from."""

    value: float = Field(
        description="The amount, as written. Negative for deficits, including "
        "figures the document shows in parentheses."
    )
    unit: AgentUnit = Field(
        description="Unit as the citing page states it: million, billion, or "
        "percent for a share. Do not convert between them."
    )
    page: int = Field(gt=0, description="1-indexed page the figure was read from.")
    label: str = Field(
        description="What the figure measures, e.g. 'Future Energy Fund top-up'."
    )


class Finding(BaseModel):
    """One agent's report on the sub-task it was given."""

    agent: str = Field(description="Which agent produced this.")
    sub_task: str = Field(description="The question the supervisor handed over.")
    summary: str = Field(
        description="What the agent found, in prose, grounded in its pages."
    )
    figures: list[Figure] = Field(
        default_factory=list, description="Every figure cited, with provenance."
    )
    pages_read: list[int] = Field(
        default_factory=list, description="The pages this agent was given."
    )


class Decision(BaseModel):
    """One supervisor turn: what it chose, what ran, and why.

    `chose` and `routed_to` differ when a guard overrides the model. Recording
    both is what makes this a decision trace rather than a route log - a forced
    route is never presented as a judgement.
    """

    turn: int = Field(gt=0)
    reasoning: str = Field(
        description="Why this route. Stated before the choice is made."
    )
    chose: str = Field(description="What the model picked.")
    routed_to: str = Field(description="What actually ran.")
    overridden: str | None = Field(
        default=None, description="The guard that fired, if the model was overruled."
    )
    sub_task: str = Field(default="", description="The brief handed to the agent.")

    @property
    def was_overridden(self) -> bool:
        """True when a guard changed the model's choice."""
        return self.overridden is not None


class NodeCost(BaseModel):
    """How long one node call took.

    Only elapsed time is recorded: structured output hides the raw response,
    so per-call token usage is not visible without extra plumbing that the
    trace does not need.
    """

    node: str
    seconds: float = 0.0


class SupervisorState(TypedDict, total=False):
    """State threaded through the graph.

    Annotated fields accumulate; the rest are last-write-wins.
    """

    query: str
    findings: Annotated[list[Finding], operator.add]
    decisions: Annotated[list[Decision], operator.add]
    visited: Annotated[list[str], operator.add]
    costs: Annotated[list[NodeCost], operator.add]
    # Routing target and the brief for the agent about to run.
    next: str
    sub_task: str
    # Set by synthesis (or by an out-of-scope decline).
    answer: str
    declined: bool
