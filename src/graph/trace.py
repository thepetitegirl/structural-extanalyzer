"""Rendering the supervisor's decision trace.

The requirement asks for the supervisor's decision-making *process*, not a log
of which functions ran. Four things separate the two, and each is a constraint
on what gets recorded rather than a formatting choice:

  - why it routed there: `Decision.reasoning`, stated before the choice;
  - that it was a choice: `chose` alongside `routed_to`;
  - when the system overruled it: `overridden` names the guard;
  - what each agent contributed: one `Finding` each, with pages and figures.

Rendering mirrors `Report.table()` in `src/evaluation.py`, so Part 3's output
reads like Parts 1 and 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.graph.state import AGENTS, Decision, Finding, NodeCost

# Groq's free tier allows this many tokens per model per day. Reported
# alongside a run's cost so the figure has a denominator.
DAILY_TOKEN_QUOTA = 100_000


@dataclass(frozen=True)
class Citation:
    """One figure, with everything needed to check it against the document."""

    label: str
    value: float
    unit: str
    page: int
    quote: str
    agent: str


@dataclass(frozen=True)
class Trace:
    """A complete record of one query."""

    query: str
    decisions: list[Decision] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    answer: str = ""
    costs: list[NodeCost] = field(default_factory=list)
    declined: bool = False

    @property
    def agents_invoked(self) -> list[str]:
        """The specialist agents that ran, in order."""
        return [finding.agent for finding in self.findings if finding.agent in AGENTS]

    @property
    def overrides(self) -> list[Decision]:
        """Turns where a guard overruled the model."""
        return [decision for decision in self.decisions if decision.was_overridden]

    def citations(self) -> list[Citation]:
        """Every figure any agent reported, flattened."""
        return [
            Citation(
                label=figure.label,
                value=figure.value,
                unit=figure.unit,
                page=figure.page,
                quote=figure.quote,
                agent=finding.agent,
            )
            for finding in self.findings
            for figure in finding.figures
        ]

    def summary(self) -> str:
        """One line stating what happened."""
        def plural(count: int, noun: str, suffix: str = "") -> str:
            """Render a count with its noun pluralised."""
            return f"{count} {noun}{'s' if count != 1 else ''}{suffix}"

        return ", ".join(
            [
                plural(len(self.decisions), "decision"),
                plural(len(self.agents_invoked), "agent", " invoked"),
                plural(len(self.overrides), "override"),
                plural(len(self.citations()), "figure", " cited"),
            ]
        )

    def table(self) -> str:
        """The routing table: what was chosen, what ran, and why."""
        lines = [
            f"{'turn':<5} {'chose':<18} {'routed to':<18} why",
            "-" * 92,
        ]

        for decision in self.decisions:
            reason = decision.reasoning.strip().replace("\n", " ")
            lines.append(
                f"{decision.turn:<5} {decision.chose:<18} {decision.routed_to:<18} {reason[:48]}"
            )
            if decision.was_overridden:
                lines.append(
                    f"{'':<5} {'':<18} {'OVERRIDDEN:':<18} {decision.overridden}"
                )

        return "\n".join(lines)

    def findings_table(self) -> str:
        """What each agent read and reported."""
        if not self.findings:
            return "(no agent was invoked)"

        lines = [f"{'agent':<20} {'pages':<16} figures"]
        for finding in self.findings:
            pages = ", ".join(str(page) for page in finding.pages_read)
            lines.append(f"{finding.agent:<20} {pages:<16} {len(finding.figures)}")

        return "\n".join(lines)

    def citations_table(self) -> str:
        """Every figure with the page and text it came from."""
        citations = self.citations()
        if not citations:
            return "(no figures cited)"

        lines = []
        for citation in citations:
            quote = citation.quote.strip().replace("\n", " ")
            lines.append(
                f"  {citation.value:>10,.2f} {citation.unit:<8} p.{citation.page:<4} "
                f"{citation.label}"
            )
            lines.append(f'{"":>13}{"":>9}{"":>6} "{quote[:64]}"')

        return "\n".join(lines)

    def costs_table(self) -> str:
        """What the run cost, against the daily quota."""
        if not self.costs:
            return "(costs not recorded)"

        lines = [f"{'node':<20} {'in':>8} {'out':>8} {'seconds':>9}"]
        total_in = total_out = 0
        total_seconds = 0.0

        for cost in self.costs:
            lines.append(
                f"{cost.node:<20} {cost.input_tokens:>8,} {cost.output_tokens:>8,} "
                f"{cost.seconds:>9.2f}"
            )
            total_in += cost.input_tokens
            total_out += cost.output_tokens
            total_seconds += cost.seconds

        share = (total_in + total_out) / DAILY_TOKEN_QUOTA * 100
        lines.append("-" * 48)
        lines.append(
            f"{'total':<20} {total_in:>8,} {total_out:>8,} {total_seconds:>9.2f}"
            f"   ({share:.1f}% of daily quota)"
        )

        return "\n".join(lines)

    def render(self) -> str:
        """The full trace, as shown in the notebook."""
        sections = [
            f"QUERY: {self.query}",
            "",
            "SUPERVISOR DECISIONS",
            self.table(),
            "",
            "AGENT FINDINGS",
            self.findings_table(),
            "",
            "CITATIONS",
            self.citations_table(),
            "",
            "ANSWER",
            self.answer or "(none)",
            "",
            "NODE COSTS",
            self.costs_table(),
            "",
            self.summary(),
        ]
        return "\n".join(sections)
