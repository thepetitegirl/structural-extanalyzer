"""Part 3: the expenditure specialist node.

Thin by design, mirroring `revenue_agent`. Its pages carry the Future Energy
Fund in three forms - 5.00 (billion), $5.0 billion, and 5,000 (million) - which
is why its prompt spends more words on units than the revenue agent's.
"""

from __future__ import annotations

from src.agents.base import run_agent

AGENT_KEY = "expenditure"
NODE_NAME = "expenditure_agent"


def expenditure_node(state, model, config, pdf_path):
    """Run the expenditure agent and append its finding to state."""
    finding = run_agent(
        AGENT_KEY, pdf_path, state.get("sub_task") or state["query"], model, config
    )

    return {"findings": [finding], "visited": [NODE_NAME]}
