"""Part 3: deterministic checks on an open-ended answer.

Parts 1 and 2 score against exact values. Part 3's queries have no single
correct wording, so the prose cannot be scored - but four things can be, and
together they establish that the answer is grounded and correctly routed.

Reuses `Check` and `Report` from `src/evaluation.py`, so all three parts render
their results the same way.

What is deliberately not scored: how well the answer reads. Any automated proxy
- keyword presence, length - would be a poor measure that invites optimising
against it. The trace is supplied so a reader can judge for themselves.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.evaluation import TOLERANCE, Check, Report
from src.graph.trace import Trace

DEMO_QUERIES_PATH = (
    Path(__file__).resolve().parents[2] / "expectations" / "demo_queries.yaml"
)


def load_demo_queries(path: Path | str = DEMO_QUERIES_PATH) -> list[dict]:
    """Load the demo queries that are enabled.

    Kept in YAML so a query can be added, reworded or skipped without touching
    code - and so a run can be trimmed when the daily token budget is tight.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    return [query for query in data.get("queries", []) if query.get("enabled", True)]


def demo_query(query_id: str, path: Path | str = DEMO_QUERIES_PATH) -> str:
    """The wording of one demo query, looked up by id.

    Lets a caller name the query it wants - `demo_query("required")` - without
    repeating the text, so the wording lives in the YAML alone. Disabled
    queries are still found: skipping one in a demo run should not stop it
    being referred to by name.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}

    for query in data.get("queries", []):
        if query.get("id") == query_id:
            return query["query"]

    known = ", ".join(q.get("id", "?") for q in data.get("queries", []))
    raise KeyError(f"No demo query with id {query_id!r}. Known ids: {known}.")


def _normalise(text: str) -> str:
    """Collapse whitespace, so pypdf's irregular spacing does not defeat a match."""
    return re.sub(r"\s+", " ", text).strip().lower()


def check_routing(trace: Trace, expected: list[str]) -> Check:
    """Did the supervisor invoke the agents this query needed?

    Binary and read straight from the trace. Catches both an agent that should
    have run and did not, and one that ran unnecessarily.
    """
    invoked = trace.agents_invoked
    missing = [agent for agent in expected if agent not in invoked]
    extra = [agent for agent in invoked if agent not in expected]

    problems = []
    if missing:
        problems.append(f"never invoked: {', '.join(missing)}")
    if extra:
        problems.append(f"invoked unnecessarily: {', '.join(extra)}")

    if problems:
        return Check("routing", False, "; ".join(problems))

    return Check(
        "routing", True, f"invoked {', '.join(invoked) or 'no agent (declined)'}"
    )


def _matches(figure, wanted: dict) -> bool:
    """True when a figure has the wanted value and unit."""
    if abs(figure.value - wanted["value"]) > TOLERANCE:
        return False
    return figure.unit == wanted["unit"]


def check_figures(
    trace: Trace, required: list[dict], any_of: list[list[dict]] | None = None
) -> Check:
    """Are the required figures present, with the unit their page states?

    `required` must all appear. Each group in `any_of` needs one member - the
    Future Energy Fund is correct as 5.0 billion or 5,000 million depending on
    which page was cited, and both readings should pass.

    The check that earns its place: a value of 5000 labelled billions is a
    thousandfold error, and it looks entirely plausible in isolation.
    """
    citations = trace.citations()
    problems = []

    for wanted in required:
        if not any(_matches(figure, wanted) for figure in citations):
            problems.append(f"missing {wanted['value']} {wanted['unit']}")

    for group in any_of or []:
        if not any(_matches(figure, option) for figure in citations for option in group):
            options = " or ".join(
                f"{option['value']} {option['unit']}" for option in group
            )
            problems.append(f"missing {options}")

    if problems:
        return Check("figures", False, "; ".join(problems))

    total = len(required) + len(any_of or [])
    return Check("figures", True, f"{total} required figure(s) present, units correct")


def check_traceability(trace: Trace, page_text: dict[int, str]) -> Check:
    """Does every quote appear on the page it cites?

    Stronger than trusting the model's page number: it can read the right
    sentence and attribute it wrongly, leaving a correct value that cannot be
    verified. Matching is whitespace-normalised because pypdf spaces text
    irregularly.
    """
    unverified = []

    for citation in trace.citations():
        text = page_text.get(citation.page)
        if text is None:
            unverified.append(f"p.{citation.page} (not supplied)")
        elif _normalise(citation.quote) not in _normalise(text):
            unverified.append(f"p.{citation.page} ({citation.label})")

    if unverified:
        return Check(
            "traceability",
            False,
            f"{len(unverified)} quote(s) not found on their cited page: "
            + ", ".join(unverified),
        )

    total = len(trace.citations())
    return Check(
        "traceability", True, f"{total}/{total} quotes found on their cited page"
    )


def check_page_discipline(trace: Trace, config) -> Check:
    """Did any agent cite a page outside its own set?

    A figure from an unread page cannot have been read, so it was fabricated.
    """
    problems = []

    for finding in trace.findings:
        agent_key = finding.agent.removesuffix("_agent")
        try:
            allowed = set(config.pages_for_agent(agent_key))
        except Exception:
            problems.append(f"{finding.agent}: no configured page set")
            continue

        cited = {figure.page for figure in finding.figures}
        outside = sorted(cited - allowed)
        if outside:
            problems.append(
                f"{finding.agent} cited page(s) {outside}, outside {sorted(allowed)}"
            )

    if problems:
        return Check("page discipline", False, "; ".join(problems))

    return Check("page discipline", True, "every figure came from an agent's own pages")


def score_part3(
    trace: Trace, expected: dict, config, page_text: dict[int, str] | None = None
) -> Report:
    """Part 3: run all four checks against one query's trace."""
    return Report(
        [
            check_routing(trace, expected.get("routed_to", [])),
            check_figures(
                trace, expected.get("figures", []), expected.get("figures_any_of")
            ),
            check_traceability(trace, page_text or {}),
            check_page_discipline(trace, config),
        ]
    )
