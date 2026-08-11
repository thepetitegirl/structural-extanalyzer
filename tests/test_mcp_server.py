"""Unit tests for src.tools.mcp_server.

These exercise the server's tool functions directly rather than over a stdio
session: the protocol wiring is the MCP library's responsibility, and a
subprocess per test would be slow without testing our code any harder.

No model and no network is involved.
"""

import pytest

from src.tools import mcp_server
from src.tools.date_tool import classify_date, normalize_date


def test_server_is_named():
    """The server has a name, which is what a client sees on connecting."""
    assert mcp_server.mcp.name == "date-tools"


@pytest.mark.asyncio
async def test_both_tools_are_registered():
    """Both date tools are exposed to clients."""
    tools = await mcp_server.mcp.list_tools()

    assert {tool.name for tool in tools} == {"normalize_date", "classify_date"}


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


def test_classify_matches_the_underlying_tool():
    """The MCP wrapper agrees with the @tool version."""
    result = mcp_server.classify_date("2008-02-15", "2024-01-01")

    assert result == str(
        classify_date.invoke({"iso_date": "2008-02-15", "reference": "2024-01-01"})
    )


def test_classify_returns_a_plain_string():
    """The status crosses the protocol as a string, not an enum."""
    result = mcp_server.classify_date("2024-02-16", "2024-01-01")

    assert isinstance(result, str)
    assert result == "Upcoming"


def test_reference_date_defaults_to_the_required_value():
    """Omitting the reference uses the fixed default of 2024-01-01."""
    assert mcp_server.classify_date("2024-02-16") == "Upcoming"


def test_unparseable_date_returns_none():
    """A date the tool cannot read yields null over the protocol, not a guess."""
    assert mcp_server.normalize_date("no date here") is None
