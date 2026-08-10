"""Unit tests for src.tools.mcp_client.

One test starts the real server as a subprocess to prove the protocol path
works end to end; it is marked so it can be skipped when subprocesses are
unwelcome. No model and no network is involved either way.
"""

from types import SimpleNamespace

import pytest

from src.tools.mcp_client import _tool_text, normalize_dates_via_mcp, server_parameters


def test_server_parameters_point_at_the_local_server():
    """The client launches this repo's server module, not a remote service."""
    params = server_parameters()

    assert "src.tools.mcp_server" in params.args
    assert params.command


@pytest.mark.asyncio
async def test_normalizes_dates_over_the_protocol():
    """Dates are normalised by the server, reached over stdio.

    This starts a real subprocess and speaks MCP to it, so it proves the
    transport rather than the wrapped function - which the date_tool tests
    already cover.
    """
    results = await normalize_dates_via_mcp(
        ["Distributed on Budget Day: 16 February 2024", "15 February 2008"]
    )

    assert results == ["2024-02-16", "2008-02-15"]


@pytest.mark.asyncio
async def test_unparseable_date_returns_none_over_the_protocol():
    """A date the server cannot read comes back as None, not a guess."""
    results = await normalize_dates_via_mcp(["no date here"])

    assert results == [None]


@pytest.mark.asyncio
async def test_empty_input_makes_no_calls():
    """An empty list returns an empty list without starting a session."""
    assert await normalize_dates_via_mcp([]) == []


def test_server_error_raises_rather_than_passing_as_a_value():
    """An isError result raises; its message must never be returned as a date."""
    result = SimpleNamespace(isError=True, content=[SimpleNamespace(text="boom")])

    with pytest.raises(RuntimeError, match="boom"):
        _tool_text(result)
