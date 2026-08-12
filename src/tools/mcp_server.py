"""Part 2: local MCP server exposing the date normalisation tool.

Runs over stdio: a client launches this as a subprocess and they exchange
JSON-RPC on stdin (its input) and stdout (its output). Nothing is networked and
nothing is deployed - the server exists only for the lifetime of the client
that started it.

**This module is transport, not logic.** The tool below is a one-line wrapper
around the real implementation in date_tool.py. Nothing is reimplemented here,
so the unit tests on date_tool.py still cover the behaviour this server serves.

Because stdout carries the protocol, nothing here may print: stray output would
corrupt the message stream the client is reading.

Run directly to serve on stdio:

    uv run python -m src.tools.mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

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


def main() -> None:
    """Serve the date tool over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
