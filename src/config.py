"""Configuration loading, shared by all three parts.

Non-secret settings live in `config.yml` and are committed. Credentials live in
`.env` and are read from the environment, never from the YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path("config.yml")

REQUIRED_KEYS = ("pdf_url", "pages", "provider", "model")

# Part 2's reference date, used when config.yml does not name one. The date
# tools carry the same literal as an inert signature default; this is the value
# the application actually classifies against.
DEFAULT_REFERENCE_DATE = "2024-01-01"

# Part 3's reply to a query the document cannot answer, used when config.yml
# does not give one. Fixed text rather than a model call: the model may well
# know the answer, and letting it reply would produce exactly the ungrounded
# output the rest of the system prevents.
DEFAULT_DECLINE_MESSAGE = (
    "This question cannot be answered from the document. It covers Singapore "
    "government revenue and expenditure for FY2024 - Operating Revenue and its "
    "components, Net Investment Returns Contribution, Total Expenditure, "
    "Special Transfers, and top-ups to Endowment and Trust Funds."
)

# Providers that authenticate with an API key, and the variable holding it.
# A provider absent from this map needs no credential, so api_key() returns
# None for it rather than raising.
API_KEY_VARS = {
    "groq": "GROQ_API_KEY",
}


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class Config:
    """Settings for one extraction run."""

    pdf_url: str
    pages: list[int]
    provider: str
    model: str
    temperature: float = 0.0
    # Retries for a rate-limited request; see config.yml for why 6.
    max_retries: int = 6
    # Field name -> the page(s) that field must be read from. Injected into the
    # prompt so a page is defined here and nowhere else.
    field_pages: dict[str, int | list[int]] = field(default_factory=dict)
    # Part 2: pages holding the dates to normalise and classify.
    date_pages: dict[str, int] = field(default_factory=dict)
    # Part 2: the date every extracted date is classified against. Fixed rather
    # than today's date, so a 2024 date stays Upcoming as the real date recedes.
    reference_date: str = DEFAULT_REFERENCE_DATE
    # Part 3: the pages each specialist agent may read. Deliberately narrow -
    # an agent able to read the whole document will find a plausible figure on
    # a page it was not asked about, and nothing in the number says so.
    agent_pages: dict[str, list[int]] = field(default_factory=dict)
    # Part 3: hard cap on supervisor turns, so the graph provably terminates.
    max_turns: int = 4
    # Part 3: what the graph replies when the document cannot answer at all.
    decline_message: str = DEFAULT_DECLINE_MESSAGE
    # Part 3: words that suggest a query is about spending rather than revenue,
    # used only to pick a fallback agent when the model tries to synthesise
    # with nothing in hand. A hint list that misses is not a wrong answer - the
    # other agent still runs on a later turn.
    expenditure_hints: list[str] = field(default_factory=list)
    # Part 3: ceiling on an agent's rendered prompt. An agent's pages should be
    # a few thousand characters, so a prompt near the whole document means a
    # page set was widened by mistake - which multiplies the cost of every call
    # without visibly changing the output.
    prompt_character_budget: int = 12_000

    def pages_for_agent(self, agent: str) -> list[int]:
        """Part 3: the pages a specialist agent may read.

        Returns the page numbers themselves, since the agent passes them to
        `extract_pages`. Compare `pages_for_field`, which renders pages for a
        prompt instead.
        """
        pages = self.agent_pages.get(agent)
        if not pages:
            raise ConfigError(
                f"No pages configured for agent {agent!r}. "
                f"Known agents: {', '.join(sorted(self.agent_pages)) or 'none'}."
            )
        return pages

    def page_for_date(self, name: str) -> int:
        """Part 2: the page a named date must be read from.

        Bound by name rather than by position in a list, so adding a date
        cannot silently shift which page another one is read from.
        """
        page = self.date_pages.get(name)
        if page is None:
            raise ConfigError(
                f"No page configured for date {name!r}. "
                f"Known dates: {', '.join(sorted(self.date_pages)) or 'none'}."
            )
        return page

    @property
    def date_page_numbers(self) -> list[int]:
        """Every page a date is bound to, for extraction."""
        return sorted(set(self.date_pages.values()))

    def pages_for_field(self, field_name: str) -> str:
        """Part 1: a field's bound page(s), rendered for the prompt.

        Returns a string rather than numbers - "5", or "5 AND 6" - because it
        is injected directly into the extraction prompt as the page the field
        must be read from.
        """
        pages = self.field_pages.get(field_name)
        if pages is None:
            raise ConfigError(f"No page configured for field {field_name!r}.")
        if isinstance(pages, int):
            return str(pages)
        return " AND ".join(str(page) for page in pages)

    def api_key(self) -> str | None:
        """Return the provider's API key, or None if it needs no credential.

        Raises ConfigError when a key is required but absent, so the failure
        surfaces here rather than as an opaque auth error mid-request.
        """
        var = API_KEY_VARS.get(self.provider)
        if var is None:
            return None

        key = os.environ.get(var)
        if not key:
            raise ConfigError(
                f"{var} is not set. Copy .env.example to .env and add your key."
            )
        return key


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Load settings from a YAML file, with .env loaded into the environment."""
    load_dotenv()

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ConfigError(
            f"Config {path} is missing required key(s): {', '.join(missing)}"
        )

    pages = list(data["pages"])
    field_pages = data.get("field_pages") or {}

    # A field bound to a page that is never extracted would silently produce a
    # wrong answer, so catch the mismatch here rather than at inference time.
    for name, cited in field_pages.items():
        cited_pages = [cited] if isinstance(cited, int) else list(cited)
        missing = [page for page in cited_pages if page not in pages]
        if missing:
            raise ConfigError(
                f"Field {name!r} cites page(s) {missing}, which are not in "
                f"pages {pages}. Add them to `pages` or correct `field_pages`."
            )

    # Agent page sets are NOT validated against `pages`: that is Part 1's
    # extraction set, and the agents legitimately read elsewhere.
    agent_pages = data.get("agent_pages") or {}
    for agent, agent_page_list in agent_pages.items():
        if not agent_page_list:
            raise ConfigError(f"Agent {agent!r} has no pages configured.")
        invalid = [page for page in agent_page_list if page < 1]
        if invalid:
            raise ConfigError(
                f"Agent {agent!r} cites non-positive page(s) {invalid}; "
                "pages are 1-indexed."
            )

    # Turn one can only route to an agent, so a cap below 2 would force
    # synthesis with no findings - the very state guard 3 exists to prevent.
    max_turns = int(data.get("max_turns", 4))
    if max_turns < 2:
        raise ConfigError(f"max_turns must be at least 2, got {max_turns}.")

    # The reference is compared against ISO dates, so a value the tools cannot
    # parse would surface as a ValueError mid-classification instead of here.
    reference_date = str(data.get("reference_date", DEFAULT_REFERENCE_DATE))
    try:
        date.fromisoformat(reference_date)
    except ValueError as exc:
        raise ConfigError(
            f"reference_date must be an ISO date (YYYY-MM-DD), got "
            f"{reference_date!r}."
        ) from exc

    # A budget of zero or less would fail every agent, so it is a configuration
    # error rather than a limit that happens never to be satisfiable.
    prompt_character_budget = int(data.get("prompt_character_budget", 12_000))
    if prompt_character_budget < 1:
        raise ConfigError(
            f"prompt_character_budget must be positive, got "
            f"{prompt_character_budget}."
        )

    # Declining with an empty string would answer nothing at all, so an blank
    # message is a configuration error rather than a silent non-answer.
    decline_message = str(data.get("decline_message") or DEFAULT_DECLINE_MESSAGE)
    if not decline_message.strip():
        raise ConfigError("decline_message must not be empty.")

    return Config(
        pdf_url=data["pdf_url"],
        pages=pages,
        provider=data["provider"],
        model=data["model"],
        temperature=float(data.get("temperature", 0.0)),
        max_retries=int(data.get("max_retries", 6)),
        field_pages=field_pages,
        date_pages=dict(data.get("date_pages") or {}),
        reference_date=reference_date,
        agent_pages={k: list(v) for k, v in agent_pages.items()},
        max_turns=max_turns,
        decline_message=decline_message,
        prompt_character_budget=prompt_character_budget,
        # Lowercased here because the query is lowercased before matching, so a
        # capitalised hint in the YAML would silently never fire.
        expenditure_hints=[
            str(hint).lower() for hint in (data.get("expenditure_hints") or [])
        ],
    )
