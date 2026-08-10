"""The revenue specialist node.

Thin by design: the work is in `base.run_agent`, so the two agents cannot drift
apart. What distinguishes this agent is its prompt and its page set, both named
in config rather than here.
"""

from __future__ import annotations

from src.agents.base import run_agent

AGENT_KEY = "revenue"
NODE_NAME = "revenue_agent"


def revenue_node(state, model, config, pdf_path):
    """Run the revenue agent and append its finding to state."""
    finding = run_agent(
        AGENT_KEY, pdf_path, state.get("sub_task") or state["query"], model, config
    )

    return {"findings": [finding], "visited": [NODE_NAME]}
