# Dependencies

Nine runtime dependencies, pinned in `pyproject.toml` and installed with
`uv sync`. Why each is used:

| Dependency | Why |
|---|---|
| `langchain`, `langchain-groq` | Schema-bound calls via `with_structured_output`, so the model returns a validated object rather than prose |
| `langgraph` | Conditional edges and accumulating state for Part 3's supervisor loop |
| `mcp` | Part 2's requirement - FastMCP server over stdio |
| `pydantic` | Schemas as contracts; field descriptions carry into the prompt |
| `pypdf` | Chosen on measured row integrity and speed - see [Part 1](../README.md#part-1-extraction) |
| `pyyaml` | Config, expectations and queries outside code |
| `python-dotenv` | `GROQ_API_KEY` from a gitignored `.env` |

Dev-only: `pytest`, `pytest-asyncio`, `ruff`.
