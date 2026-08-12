"""Part 3: shared machinery for the specialist agents.

The two agents differ only in their name, their pages and their prompt. Writing
them as two near-identical files would invite drift, so the work lives here and
each agent module is a thin wrapper - the same reason `mcp_server.py` imports
from `date_tool.py` rather than reimplementing it.

Page text is loaded here and discarded when the node returns. It never enters
graph state, which is what keeps a multi-hop run affordable.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from src.extraction.prompts import load_prompt
from src.graph.state import Figure, Finding
from src.ingestion.parser import extract_pages, page_of_quote


class AgentReport(BaseModel):
    """What a specialist returns before it is wrapped with its provenance."""

    summary: str = Field(
        description="What the pages say about the sub-task, in prose. Ground "
        "every claim in the text; say plainly if the pages do not cover it."
    )
    figures: list[Figure] = Field(
        default_factory=list,
        description="Every figure cited, each with its value, unit as the page "
        "states it, page number, and the text it was read from.",
    )


def run_agent(
    agent: str,
    pdf_path: Path | str,
    sub_task: str,
    model,
    config,
) -> Finding:
    """Run one specialist agent over its configured pages.

    `agent` is the config key ("revenue"), and the returned finding names the
    node ("revenue_agent") so the trace and the graph agree.
    """
    pages = config.pages_for_agent(agent)
    page_text = extract_pages(pdf_path, pages)

    prompt = load_prompt(f"{agent}_agent")
    # JSON mode rather than tool-calling, for the same reason as the
    # supervisor: over a long generation the model's tool-call wrapper drifts
    # from the format Groq's parser accepts, and the request is rejected even
    # when the content is right. JSON mode has no wrapper to misparse.
    chain = prompt | model.with_structured_output(AgentReport, method="json_mode")

    report = chain.invoke({"page_text": page_text, "sub_task": sub_task})

    return Finding(
        agent=f"{agent}_agent",
        sub_task=sub_task,
        summary=report.summary,
        figures=[_attribute(figure, page_text) for figure in report.figures],
        pages_read=pages,
    )


def _attribute(figure: Figure, page_text: str) -> Figure:
    """Set a figure's page to the one whose text contains its quote.

    The prompt already instructs the model to read the marker above the
    sentence it is quoting, and it still misattributes: related figures appear
    on several pages, and a value correct on the wrong page reads as plausible.
    Because the quote must be verbatim, the page follows from it - so it is
    resolved here rather than left to the model to get right.
    """
    resolved = page_of_quote(figure.quote, page_text, figure.page)

    if resolved == figure.page:
        return figure

    # Keep what the model claimed, so the correction shows in the trace rather
    # than replacing the citation silently.
    return figure.model_copy(update={"page": resolved, "claimed_page": figure.page})
