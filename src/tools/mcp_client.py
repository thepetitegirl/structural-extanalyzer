"""Part 2: client for the local MCP date server.

Launches `src.tools.mcp_server` as a subprocess and calls its tools over stdio.
This is the path the design calls for: the tool runs in a separate process
and is reached over the protocol, rather than being called as a Python
function.

**This module owns the awkward part** - starting the process, exchanging
JSON-RPC over the pipes, and unwrapping the replies - so callers get back a
plain list of ISO date strings and never touch the protocol themselves.

Everything here is async because the MCP SDK's client side is async-only
(`stdio_client` and `ClientSession` are both async context managers). It is not
async for concurrency: the dates are normalised one after another in a single
session. Synchronous callers bridge the gap with `asyncio.run`, as
`dates.main` does.

The `@tool` decorators in date_tool.py remain available as a fallback and are what the unit
tests exercise directly. Both reach the same implementation - only the
transport differs.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parents[2]

SERVER_MODULE = "src.tools.mcp_server"


def server_parameters() -> StdioServerParameters:
    """Describe how to launch the local server.

    Uses the running interpreter so the subprocess shares this environment,
    rather than depending on a particular launcher being on PATH.
    """
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", SERVER_MODULE],
        cwd=str(REPO_ROOT),
    )


def _tool_text(result) -> str | None:
    """Pull the text payload out of an MCP tool result.

    A server-side tool exception arrives as a normal result with `isError`
    set, not as a raised exception - returning its message as if it were a
    value would hand the caller an error string where a date was expected.
    """
    if result.isError:
        detail = result.content[0].text if result.content else "no detail"
        raise RuntimeError(f"MCP tool call failed: {detail}")

    if not result.content:
        return None

    text = result.content[0].text

    # The server returns null for a date it cannot parse; MCP delivers that as
    # the string "null".
    if text in ("null", ""):
        return None

    return text


async def normalize_dates_via_mcp(texts: list[str]) -> list[str | None]:
    """Normalise each text to ISO format, using the MCP server.

    One session handles every date: starting a subprocess per call would cost
    more than the work itself.
    """
    if not texts:
        return []

    async with stdio_client(server_parameters()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            results = []
            for text in texts:
                result = await session.call_tool("normalize_date", {"text": text})
                results.append(_tool_text(result))

            return results


async def list_tools() -> list[dict]:
    """Return the tools the server exposes, for inspection."""
    async with stdio_client(server_parameters()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()

            return [
                {"name": tool.name, "description": (tool.description or "").strip()}
                for tool in listing.tools
            ]


async def main() -> None:
    """Show what the server exposes, as a manual check."""
    for tool in await list_tools():
        print(f"{tool['name']}: {tool['description'].splitlines()[0]}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
