"""Unit tests for src.tools.mcp_server.

These exercise the server's tool functions directly rather than over a stdio
session: the protocol wiring is the MCP library's responsibility, and a
subprocess per test would be slow without testing our code any harder.

No model and no network is involved.
"""

import pytest

from src.tools import mcp_server
from src.tools.date_tool import normalize_date


def test_server_is_named():
    """The server has a name, which is what a client sees on connecting."""
    assert mcp_server.mcp.name == "date-tools"


@pytest.mark.asyncio
async def test_only_normalisation_is_exposed():
    """Normalisation is the one step that runs over the protocol.

    Classification is verified in-process by `date_reasoning.check`, so
    exposing it here would advertise a route nothing calls.
    """
    tools = await mcp_server.mcp.list_tools()

    assert {tool.name for tool in tools} == {"normalize_date"}


@pytest.mark.asyncio
async def test_exposed_tools_have_descriptions():
    """Each tool carries a description, which is how a model knows to call it."""
    for tool in await mcp_server.mcp.list_tools():
        assert tool.description, f"{tool.name} has no description"


def test_normalize_matches_the_underlying_tool():
    """The MCP wrapper returns what the @tool version returns.

    The server imports the tool rather than reimplementing it, so this guards
    against the two paths drifting apart.
    """
    text = "Distributed on Budget Day: 16 February 2024"

    assert mcp_server.normalize_date(text) == normalize_date.invoke({"text": text})


def test_unparseable_date_returns_none():
    """A date the tool cannot read yields null over the protocol, not a guess."""
    assert mcp_server.normalize_date("no date here") is None
