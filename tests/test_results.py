"""Unit tests for src.results.

Saving is pure file I/O, so these need no model and no network.
"""

import json

from src.results import RESULTS_DIR, save_json


def test_writes_json_to_the_given_path(tmp_path):
    """The payload is written as JSON at the path asked for."""
    path = save_json({"a": 1}, tmp_path / "out.json")

    assert json.loads(path.read_text()) == {"a": 1}


def test_creates_missing_directories(tmp_path):
    """A fresh clone has no results directory, so saving makes one."""
    path = save_json([1, 2], tmp_path / "nested" / "deep" / "out.json")

    assert path.is_file()


def test_returns_the_path(tmp_path):
    """The caller is told where the file went, so it can report it."""
    target = tmp_path / "out.json"

    assert save_json({"a": 1}, target) == target


def test_overwrites_an_existing_file(tmp_path):
    """A re-run replaces the previous answer rather than appending to it."""
    target = tmp_path / "out.json"
    save_json({"first": True}, target)

    save_json({"second": True}, target)

    assert json.loads(target.read_text()) == {"second": True}


def test_ends_with_a_newline(tmp_path):
    """A trailing newline keeps the file well-formed for git and editors."""
    path = save_json({"a": 1}, tmp_path / "out.json")

    assert path.read_text().endswith("\n")


def test_serialises_pydantic_models(tmp_path):
    """A result carrying Pydantic models is written without manual dumping."""
    from src.graph.state import NodeCost

    path = save_json({"costs": [NodeCost(node="supervisor", seconds=1.5)]},
                     tmp_path / "out.json")

    assert json.loads(path.read_text())["costs"][0]["node"] == "supervisor"


def test_results_dir_is_a_relative_path():
    """Results are written inside the repository, not to an absolute path."""
    assert not RESULTS_DIR.is_absolute()
