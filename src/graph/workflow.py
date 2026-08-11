"""Part 3: the supervisor graph.

    START -> supervisor -> revenue_agent ------+
                 |  |                          |
                 |  +-> expenditure_agent ------+
                 |                              |
                 +-> synthesis -> END           |
                 |      ^                       |
                 |      +-----------------------+
                 +-> decline -> END

Both agents route unconditionally back to the supervisor, which is the only
node that decides. A fixed chain would have no decision to trace, and the
a decision trace needs decisions to record - so the loop is
what makes the deliverable possible, not an embellishment.

Re-entering after each agent also lets the second agent's brief be written in
light of the first's findings, which is what makes the collaboration real
rather than two independent lookups stapled together.
"""

from __future__ import annotations

import time
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from src.agents.expenditure_agent import expenditure_node
from src.agents.revenue_agent import revenue_node
from src.agents.supervisor import decide
from src.config import load_config
from src.extraction.prompts import load_prompt
from src.graph.evaluation import demo_query
from src.graph.state import NodeCost, SupervisorState
from src.graph.trace import Trace
from src.ingestion.download import ensure_pdf
from src.llm import get_chat_model
from src.results import RESULTS_DIR, save_json

RESULTS_PATH = RESULTS_DIR / "supervisor.json"

# Which demo query `main` answers. The wording itself lives in
# expectations/demo_queries.yaml alongside what it is expected to demonstrate.
REQUIRED_QUERY_ID = "required"



def _timed(node: str, fn):
    """Run a node, returning its result and what it cost."""
    started = time.perf_counter()
    result = fn()
    return result, NodeCost(node=node, seconds=round(time.perf_counter() - started, 3))


def build_graph(model, config, pdf_path: Path | str):
    """Compile the supervisor graph.

    The model, config and document path are bound here rather than carried in
    state: they do not change during a run, and keeping them out of state keeps
    the trace to what actually varies.
    """

    def supervisor(state: SupervisorState) -> dict:
        """Decide who works next, and record why."""
        (next_node, record), cost = _timed(
            "supervisor", lambda: decide(state, model, config)
        )
        return {
            "next": next_node,
            "sub_task": record.sub_task,
            "decisions": [record],
            "costs": [cost],
        }

    def revenue(state: SupervisorState) -> dict:
        """Run the revenue specialist."""
        update, cost = _timed(
            "revenue_agent", lambda: revenue_node(state, model, config, pdf_path)
        )
        return {**update, "costs": [cost]}

    def expenditure(state: SupervisorState) -> dict:
        """Run the expenditure specialist."""
        update, cost = _timed(
            "expenditure_agent",
            lambda: expenditure_node(state, model, config, pdf_path),
        )
        return {**update, "costs": [cost]}

    def synthesis(state: SupervisorState) -> dict:
        """Combine the findings into an answer.

        Sees findings, never the document: it cannot introduce a figure no
        agent reported, which is what makes the traceability check meaningful.
        """
        prompt = load_prompt("synthesis")
        chain = prompt | model

        rendered = "\n\n".join(
            f"{finding.agent} (pages {finding.pages_read}):\n"
            f"{finding.summary}\n"
            + "\n".join(
                f"  - {figure.label}: {figure.value} {figure.unit} "
                f'(p.{figure.page}) "{figure.quote}"'
                for figure in finding.figures
            )
            for finding in state.get("findings", [])
        )

        response, cost = _timed(
            "synthesis",
            lambda: chain.invoke({"query": state["query"], "findings": rendered}),
        )

        return {
            "answer": getattr(response, "content", str(response)),
            "costs": [cost],
        }

    def decline(state: SupervisorState) -> dict:
        """Answer a query the document cannot address.

        Deliberately a fixed message rather than a model call: the model may
        well know the answer, and letting it reply would produce exactly the
        ungrounded output the rest of the system prevents. The wording names
        what the document covers, so it lives in config.yml with the other
        document-specific settings.
        """
        return {"answer": config.decline_message, "declined": True}

    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor", supervisor)
    graph.add_node("revenue_agent", revenue)
    graph.add_node("expenditure_agent", expenditure)
    graph.add_node("synthesis", synthesis)
    graph.add_node("decline", decline)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "revenue_agent": "revenue_agent",
            "expenditure_agent": "expenditure_agent",
            "synthesis": "synthesis",
            "out_of_scope": "decline",
        },
    )

    # Agents always return to the supervisor, which decides again with their
    # findings in hand. This is the loop.
    graph.add_edge("revenue_agent", "supervisor")
    graph.add_edge("expenditure_agent", "supervisor")

    graph.add_edge("synthesis", END)
    graph.add_edge("decline", END)

    return graph.compile()


def run_query(query: str, model=None, config=None, pdf_path=None) -> Trace:
    """Answer one query, returning the full trace."""
    if config is None:
        config = load_config()

    if model is None:
        model = get_chat_model(config)

    if pdf_path is None:
        pdf_path = ensure_pdf(config.pdf_url)

    graph = build_graph(model, config, pdf_path)
    final = graph.invoke({"query": query})

    return Trace(
        query=query,
        decisions=final.get("decisions", []),
        findings=final.get("findings", []),
        answer=final.get("answer", ""),
        costs=final.get("costs", []),
        declined=final.get("declined", False),
    )


def stream_trace(query: str, model=None, config=None, pdf_path=None):
    """Run a query, printing each node's update as it happens.

    Same run as `run_query`, different view: this shows the graph deciding in
    real time, where the returned Trace shows the record afterwards.
    """
    if config is None:
        config = load_config()

    if model is None:
        model = get_chat_model(config)

    if pdf_path is None:
        pdf_path = ensure_pdf(config.pdf_url)

    graph = build_graph(model, config, pdf_path)

    for update in graph.stream({"query": query}, stream_mode="updates"):
        for node, changes in update.items():
            if node == "supervisor":
                decision = changes["decisions"][-1]
                arrow = (
                    f"{decision.chose} -> {decision.routed_to}"
                    if decision.was_overridden
                    else decision.routed_to
                )
                print(f"[supervisor] turn {decision.turn}: {arrow}")
                print(f"             {decision.reasoning}")
            elif node in ("revenue_agent", "expenditure_agent"):
                finding = changes["findings"][-1]
                print(
                    f"[{node}] read pages {finding.pages_read}, "
                    f"{len(finding.figures)} figures"
                )
            elif node in ("synthesis", "decline"):
                print(f"[{node}] answered")


def main() -> None:
    """Answer the two-part query, print the trace, and save it.

    The wording comes from `expectations/demo_queries.yaml`, where the notebook's
    queries already live, so it is written once rather than here as well.
    """
    trace = run_query(demo_query(REQUIRED_QUERY_ID))

    print(trace.render())
    print(f"\nsaved to: {save_json(trace, RESULTS_PATH)}")


if __name__ == "__main__":
    main()
