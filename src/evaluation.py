"""Scoring an extraction against known-correct values.

Separate from the unit tests, which mock the model. This compares a real
extraction against `evaluation/expected.yaml`.

A value alone is not enough to score against - 28.38 and 28.03 are both
plausible answers for Corporate Income Tax, differing only in which page they
came from. Each check therefore verifies the page as well, so a right number
read from the wrong page is reported as a failure rather than a pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

EXPECTED_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "expected.yaml"

# Figures are quoted to two decimals in the source, so anything closer than
# half a hundredth is a rounding artefact rather than a different answer.
TOLERANCE = 0.005


class EvaluationError(Exception):
    """Raised when expected values cannot be loaded."""


@dataclass(frozen=True)
class Check:
    """The outcome of scoring one field."""

    field: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Report:
    """The outcome of scoring a whole extraction."""

    checks: list[Check]

    @property
    def passed(self) -> bool:
        """True when every check passed."""
        return all(check.passed for check in self.checks)

    def summary(self) -> str:
        """A one-line count of passing checks."""
        passing = sum(1 for check in self.checks if check.passed)
        return f"{passing}/{len(self.checks)} checks passed"

    def table(self) -> str:
        """A readable per-field breakdown."""
        lines = [f"{'field':28s} {'result':6s} detail", "-" * 78]
        for check in self.checks:
            lines.append(
                f"{check.field:28s} {'Pass' if check.passed else 'FAIL':6s} {check.detail}"
            )
        lines.append("")
        lines.append(self.summary())
        return "\n".join(lines)


def expected_dates(path: Path | str = EXPECTED_PATH) -> dict:
    """Return the Part 2 date expectations."""
    return load_expected(path).get("dates", {})


def score_dates(results: list[dict], expected: dict | None = None) -> Report:
    """Score normalised, classified dates against the known answers.

    `results` is the output of date_reasoning.to_output(): one entry per date
    with original_text, normalized_date and status.
    """
    if expected is None:
        expected = expected_dates()

    checks = []
    for (name, want), got in zip(expected.items(), results, strict=False):
        problems = []

        if got.get("normalized_date") != want["normalized"]:
            problems.append(
                f"normalised {got.get('normalized_date')} != {want['normalized']}"
            )

        if got.get("status") != want["status"]:
            problems.append(f"status {got.get('status')} != {want['status']}")

        if problems:
            checks.append(Check(name, False, "; ".join(problems)))
        else:
            checks.append(Check(name, True, f"{want['normalized']} ({want['status']})"))

    return Report(checks)


def load_expected(path: Path | str = EXPECTED_PATH) -> dict:
    """Load the known-correct values."""
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"Expected values not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _score_numeric(name: str, got, want: dict) -> Check:
    """Score one numeric field on value, page, and unit."""
    problems = []

    if abs(got.value - want["value"]) > TOLERANCE:
        problems.append(f"value {got.value} != {want['value']}")

    if got.page != want["page"]:
        problems.append(f"page {got.page} != {want['page']}")

    want_unit = want.get("unit")
    if want_unit and getattr(got, "unit", None) != want_unit:
        problems.append(f"unit {getattr(got, 'unit', None)} != {want_unit}")

    if problems:
        return Check(name, False, "; ".join(problems))
    return Check(name, True, f"{got.value} (page {got.page})")


def _score_tax_list(name: str, got, want: dict) -> Check:
    """Score the tax list on count, required members, and source pages."""
    problems = []

    minimum = want.get("min_count", 1)
    if len(got.names) < minimum:
        problems.append(f"{len(got.names)} names, expected at least {minimum}")

    # An over-long list is a failure, not a bonus: the extra names can only have
    # come from a page this field is not bound to.
    maximum = want.get("max_count")
    if maximum is not None and len(got.names) > maximum:
        problems.append(
            f"{len(got.names)} names, expected at most {maximum} - "
            "extra names indicate an uncited page was read"
        )

    missing = [tax for tax in want.get("must_include", []) if tax not in got.names]
    if missing:
        problems.append(f"missing: {', '.join(missing)}")

    want_pages = set(want.get("pages", []))
    if want_pages and not set(got.pages) <= want_pages:
        unexpected = sorted(set(got.pages) - want_pages)
        problems.append(f"read from unexpected page(s): {unexpected}")

    if problems:
        return Check(name, False, "; ".join(problems))
    return Check(name, True, f"{len(got.names)} names (pages {sorted(set(got.pages))})")


def score_result(result, expected: dict | None = None) -> Report:
    """Score an ExtractionResult against the known-correct values."""
    if expected is None:
        expected = load_expected()

    # Part 2's dates live under their own key and are scored separately.
    fields = {k: v for k, v in expected.items() if k != "dates"}

    checks = []
    for name, want in fields.items():
        got = getattr(result, name, None)
        if got is None:
            checks.append(Check(name, False, "field absent from result"))
        elif hasattr(got, "names"):
            checks.append(_score_tax_list(name, got, want))
        else:
            checks.append(_score_numeric(name, got, want))

    return Report(checks)
