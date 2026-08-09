"""Local MCP server exposing the date tools.

Runs over stdio: a client launches this as a subprocess and they exchange
JSON-RPC on stdin/stdout. Nothing is networked and nothing is deployed - the
server exists only for the lifetime of the client that started it.

The tool logic is imported from date_tool.py rather than reimplemented, so this
module is a transport and the existing unit tests still cover the behaviour.

Run directly to serve on stdio:

    uv run python -m src.tools.mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.tools.date_tool import classify_date as _classify
from src.tools.date_tool import normalize_date as _normalize

mcp = FastMCP("date-tools")


@mcp.tool()
def normalize_date(text: str) -> str | None:
    """Convert a date written in prose into ISO format (YYYY-MM-DD).

    Accepts a bare date ("16 February 2024") or one inside a sentence
    ("Distributed on Budget Day: 16 February 2024"). Returns null when no date
    can be found, rather than guessing.
    """
    return _normalize.invoke({"text": text})


@mcp.tool()
def classify_date(iso_date: str, reference: str = "2024-01-01") -> str:
    """Classify an ISO date against a reference date.

    Returns "Expired" if the date falls before the reference, "Upcoming" if
    after, and "Ongoing" if it is the reference date itself.

    The reference is a parameter rather than today's date, so results stay
    stable over time.
    """
    return str(_classify.invoke({"iso_date": iso_date, "reference": reference}))


def main() -> None:
    """Serve the date tools over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
