# Developing a multi-agent analyzer and document extractor

Extracting structured information from an unstructured document - the Singapore
MOF *Analysis of Revenue and Expenditure FY2024* - in three parts:

* Document extraction and prompting via LangChain
* Tool calling via a local MCP server
* Multi-agent supervisor via the LangGraph framework

## The three parts

```mermaid
flowchart LR
    PDF[("source data<br/>unstructured PDF")] --> PARSE["pypdf parser"]

    subgraph P1[" "]
        direction TB
        P1T["<b>PART 1</b><br/>extraction"]
        P1T --> P1P["prompt engineering"]
        P1P --> P1L["LLM + schema"]
        P1L --> P1R["five fields<br/>value + unit + page + quote"]
        P1R --> P1S["score_result()"]
    end

    subgraph P2[" "]
        direction TB
        P2T["<b>PART 2</b><br/>dates and tools"]
        P2T --> P2F["LLM finds dates as written"]
        P2F --> P2N["normalize_date<br/>via local MCP server"]
        P2N --> P2C["LLM classifies vs 2024-01-01"]
        P2C --> P2V["classify_date verifies"]
    end

    subgraph P3[" "]
        direction TB
        P3T["<b>PART 3</b><br/>supervisor"]
        P3T --> P3S{"supervisor<br/>routes each turn"}
        P3S -->|revenue| P3R["revenue agent"]
        P3S -->|expenditure| P3E["expenditure agent"]
        P3R -.->|finding| P3S
        P3E -.->|finding| P3S
        P3S -->|done| P3Y["synthesis + trace"]
        P3S -->|out of scope| P3D["decline"]
    end

    PARSE --> P1T
    PARSE --> P2T
    PARSE --> P3T

    CFG["config.yml"] -.-> PARSE

    classDef shared fill:#e8e8e8,stroke:#666,color:#000
    classDef title fill:#ffffff,stroke:#333,stroke-width:2px,color:#000
    classDef part1 fill:#dbeafe,stroke:#2563eb,color:#000
    classDef part2 fill:#dcfce7,stroke:#16a34a,color:#000
    classDef part3 fill:#fef3c7,stroke:#d97706,color:#000
    classDef check fill:#fae8ff,stroke:#a21caf,color:#000

    class PDF,PARSE,CFG shared
    class P1T,P2T,P3T title
    class P1P,P1L,P1R part1
    class P2F,P2N,P2C part2
    class P3S,P3R,P3E,P3Y,P3D part3
    class P1S,P2V check

    style P1 fill:#f8fbff,stroke:#2563eb
    style P2 fill:#f7fdf9,stroke:#16a34a
    style P3 fill:#fffdf5,stroke:#d97706
```

Grey is shared infrastructure, blue Part 1, green Part 2, amber Part 3. The two
purple nodes are the deterministic checks - `score_result()` compares against
known values, `classify_date` verifies what the LLM concluded.

No database and no vector store: the page holding each answer is known, so
retrieval is already solved. The model reads the supplied text and copies
values into typed fields - it does not compute or recall.

## Quick start

```bash
uv sync
cp .env.example .env          # add GROQ_API_KEY (free, no card)

uv run pytest                 # 185 tests, no key or network needed
uv run ruff check .

# Part 1 - extraction
uv run python -m src.extraction.extractor      # five fields, scored

# Part 2 - dates and tools
uv run python -m src.extraction.dates          # dates found and normalised
uv run python -m src.extraction.date_reasoning # classified vs 2024-01-01, checked
uv run python -m src.tools.mcp_client          # smoke check: list the MCP server's tools

# Part 3 - supervisor
uv run python -m src.graph.workflow            # the required query, with trace
```

The source PDF is downloaded from the URL in `config.yml` on first use and
cached in `data/`, which is gitignored - a fresh clone needs only the config.

Notebooks carry the evidence for each decision:

| Notebook | Shows |
|---|---|
| `00_parser_comparison.ipynb` | Why pypdf, measured against PyMuPDF and pdfplumber |
| `01_value_exploration.ipynb` | What the correct answers are, derived from the document |
| `02_extraction.ipynb` | Model selection and two prompt versions, scored |
| `03_dates.ipynb` | MCP server, normalisation, classification with a check |
| `04_supervisor.ipynb` | The graph, seven demo queries, and the decision trace |

## Part 1: extraction

All five fields extract correctly from their cited pages. As seen below, the fields are of different nature such as floats and list of strings.

| Field | Value | Unit | Page | Note |
|---|---|---|---|---|
| Corporate Income Tax | 28.4 | billion | 5 | Prose - stated in a sentence, so context disambiguates it |
| Year-on-year change | 17.0 | % | 5 | Prose - same sentence as the amount, stated not calculated |
| Total top-ups | 20,352 | million | 20 | **Table row** - a "Total" line under a ($ million) heading |
| Taxes in Operating Revenue | 7 names | - | 5-6 | Prose - names spread across several paragraphs |
| Overall Fiscal Position | -3.57 | billion | 8 | **Table row** - one figure per year column, with negative value in parentheses |

Two of the five come off table rows, which is a core criterion in choosing the most appropriate parser, as shown below. 

### Parser: pypdf

Measured in [`notebooks/00_parser_comparison.ipynb`](notebooks/00_parser_comparison.ipynb).

**The parsing unit is the page.** The requirements cite pages - "page 5", "page
20" - so pages are the natural unit, and `extract_pages()` returns only those
requested, each prefixed with a `--- page N ---` marker. Those markers are what
make the page binding enforceable: the prompt can say PAGE 5 ONLY because the
model can see where page 5 ends.

Chunking by paragraph or token count would break that, leaving one
undifferentiated block with no way to tell which page a figure came from.

Five parsers were considered: `pypdf`, `PyMuPDF`, `pdfplumber`, `Docling` and `OCR`. Two were ruled out on inspection as Docling's ML layout analysis solves a problem a born-digital PDF does not have, and OCR needs raster images, of which this document has none. The remaining three were measured through **row integrity** and **latency**.

As seen above in extraction, since two out of five fields are obtained from table rows, it is important to ensure that row integrity is maintained so that the information can be obtained accurately.

> **Row integrity** means the table row survives extraction as one line - the
> label still attached to its figures. The Corporate Income Tax row on page 8
> reads `Corporate Income Tax 23.07 24.26 28.38 23.0 17.0` in the document.

What each parser returns for that row:

| Parser | Text Output for the Corporate Income Tax row | Result |
|---|---|:---:|
| `pypdf` | `Corporate Income Tax 23.07 24.26 28.38 23.0 17.0` | Pass |
| `pdfplumber` | `Corporate Income Tax 23.07 24.26 28.38 23.0 17.0` | Pass |
| `PyMuPDF` | `Corporate Income Tax` - figures detached | **Fail** |

Note: For `PyMuPDF`, the figures become detached from their label, so the model may attach them to the wrong row and return a wrong answer with no sign anything went wrong.

Note 2: `pdfplumber` is the only parser with a feature called table extractor. However, it detected no tables on page 8, because detection is line-based and this table has no ruling lines. Since the other two parsers have no table extractor at all, no parser yields a cell grid here.

In addition, **latency** is also considered as a secondary criterion to ensure that the information can be parsed fast to facilitate a more efficient extraction. 

| Parser | Row integrity | Speed (4 pp.) | Verdict |
|---|:---:|---:|---|
| **pypdf** | Pass | 450 ms | **Chosen** - correct, and the faster of the two |
| pdfplumber | Pass | 823 ms | Rejected - correct but 1.8x slower |
| PyMuPDF | **Fail** | 52 ms | Rejected - fastest, but wrong |
| Docling | not tested | - | Ruled out - no scanned pages |
| OCR | not tested | - | Ruled out - zero raster images |

**Correctness decides; speed only breaks ties.** Based on both criteria, `pypdf` is chosen as the parser due to its accuracy in parsing the table content and is 1.8x faster than `pdfplumber`.


### Prompt engineering

Every prompt is a YAML file with a `system` section and a `human` section, so
wording can be revised without touching Python. The split is deliberate: models
weight system instructions as standing rules and the human message as the
request, so the page bindings and conventions live in `system` while the
document text and the ask live in `human`.

**Every exchange is a single turn** - one call, one answer. No prompt continues
a conversation, so there is no `assistant` section; each call carries
everything the model needs. Where a run makes several calls, as the Part 3
supervisor does, each is independent and state is threaded through the graph
rather than through message history.

The first prompt got three of five fields right. It named the correct page for
every field and still read the wrong one twice, because it treated the page
citations as preferences rather than constraints. Both versions are kept
(`prompts/extraction.yaml` and `extraction_v1.yaml`) and run side by side in
the notebook.

Wording proved load-bearing: changing "read both pages to the end" to "read
those pages to the end" changed which taxes were returned, reproducibly at
temperature 0.

## Part 2: dates and tools

| Date | Page | Normalised | Status vs 2024-01-01 |
|---|---|---|---|
| Distribution | 1 | 2024-02-16 | Upcoming |
| Estate Duty | 36 | 2008-02-15 | Expired |

The split is deliberate: the model finds dates in prose because phrasing
varies, and a tool parses them because that has one right answer. The
requirement asks for LLM *reasoning* on the classification, so the model
decides and `classify_date` verifies afterwards - detection rather than
prevention, since prevention would mean not asking the model at all.

`src/tools/mcp_server.py` exposes both tools over stdio and is what the
pipeline uses. The `@tool` decorators remain as an automatic fallback; both
share one implementation, so the server is a transport rather than a second
copy.

## Part 3: multi-agent supervisor

Agents route unconditionally back to the supervisor, which is the only node
that decides. A fixed chain would have no decision to trace, and the trace is
what the requirement asks for.

**Seven demo queries, seven correct routes** - single-agent both ways, two
agents collaborating, and two queries declined without invoking either.
Queries live in `evaluation/demo_queries.yaml` so one can be added or disabled
without touching code.

**Page 13 is scoped to revenue only, deliberately.** It carries both the
revenue total and the top-ups sentence, so keeping it out of the expenditure
set means neither agent can answer the combined query alone. The collaboration
is structural, and asserted in the tests.

**The supervisor's choice is guarded.** Three deterministic rules - no agent
twice, a turn cap, no synthesis before any finding - and each records itself in
the trace when it fires, so a forced route is never presented as a decision.

### Scoring an open-ended answer

Prose has no single correct wording, so it is not scored. Four things are:

| Check | Catches |
|---|---|
| Routing | An agent that should have run and did not, or one that ran needlessly |
| Figures | A wrong value, or the right value with the wrong unit |
| Traceability | A quote that does not appear on the page it cites |
| Page discipline | A figure from a page the agent was never given |

## Assumptions

**Only the cited pages are read.** A correctness measure, not an optimisation.
"Corporate Income Tax" appears eight times across seven pages:

| Page | Value | What it is |
|---|---|---|
| 5 | **28.4** | Revised FY2023, prose - the answer |
| 8 | 28.38 | same figure, table precision |
| 8 | 23.07 / 24.26 | FY2022 actual / FY2023 estimated |
| 9 | 27.2% | share of Operating Revenue |
| 16 | 28.03 | Estimated FY2024 |
| 26 | 28,380 / 28,029 | same figures in $million |
| 27 | 3.9% | share of GDP |

All plausible. They differ by year, unit and kind, and nothing in the number
says which. Restricting the input eliminates seven wrong answers before the
model reads anything.

**The tax list counts revenue lines, not their constituents.** Pages 5-6 name
seven taxes as Operating Revenue components. Four more appear inside the
description of one of them - "Other Taxes, which include the Foreign Worker
Levy, Water Conservation Tax, Land Betterment Charge, and Annual Tonnage Tax"
(p.5) - and are not counted separately, since the document presents them as
what Other Taxes consists of rather than as revenue lines in their own right.
Counting them would give 11.

**Each field is read from the page cited for it,** not from wherever the figure
is most precise. Page 5 says "$28.4 billion"; page 8's table says 28.38. The
citation decides.

**Units are recorded, not converted.** Page 20 states $million while the other
pages state $billion. Normalising would hide a 1000x error.

**Target year is Revised FY2023,** except top-ups, which page 20 states for
FY2024. That follows from the page citations rather than a consistent year
choice, so no single target year is correct for all five fields.

**"Latest Actual Fiscal Position" is read as the latest available figure** —
the Revised FY2023 deficit, -3.57. Table 1.1's strictly *Actual* column is
FY2022 (1.72, a surplus); that reading would break with every other field,
which the spec labels 2024 and the cited pages state as FY2023. The rejected
alternative is noted here rather than silently dropped.

**Dates are explicit calendar dates.** Open-ended expressions - "till present",
"with immediate effect" - are out of scope; `normalize_date` returns None
rather than inventing a boundary.

**The reference date is fixed at 2024-01-01,** not today, so results stay
stable over time.

**The document does not identify a revenue stream funding the Future Energy
Fund,** and the answer says so. Government revenue is not earmarked to
particular funds; naming one would be wrong however fluent.

**Structured output uses JSON mode rather than tool-calling.** Over a long
generation the tool-call wrapper drifts from the format Groq's parser accepts,
and the request is rejected even when the content is correct. JSON mode also
works across model families where tool-calling support varies.

**Temperature is 0** throughout, so the same input yields the same output.

**Everything is document-specific.** The parser verdict, page citations, unit
conventions and the two-fiscal-year structure are particular to this
publication. Re-run the notebooks for any new source.

## Model and provider

`llama-3.1-8b-instant` on Groq, used by every part. Constraint: free tier, no
card - GPT-4 and Claude are excluded on cost, not merit.

| | Ollama (local) | Groq | Gemini |
|---|---|---|---|
| Disk / RAM | 2.0-5.2 GB / 2.5-5.6 GB | none | none |
| Rate limit | **none** | 100k tokens/day, 6k/min | tight free tier |
| Correct value | **No** - returned 28400000000.0 | Yes | - |

Neither free hosted tier supports sustained development. Groq's daily allowance
was exhausted in one session of prompt iteration: an extraction call is ~3.2k
tokens and a supervisor query ~7k. Local models have no limit but fold the unit
into the value, failing silently.

Groq was chosen because a hard failure is recoverable where a wrong number is
not, but this is a project constraint rather than a clean win.

## Known limitations

- **pypdf reads table content, not table structure.** `28.38` arrives on the
  right line but carries no marker placing it in the *Revised FY2023* column;
  the prompt binds it via the header. No fallback parser helps.
- **Chart values are drawn as shapes, not stored as data,** so no parser or OCR
  recovers them. Read the corresponding table instead.
- **One citation in Part 3 is unverifiable.** The revenue agent reports NIRC at
  $23.5 billion citing page 13, but quotes page 15's wording. Both pages state
  the figure, and the agent blended them. The value is right; the citation
  cannot be checked, and the traceability check catches it. A prompt fix was
  attempted and did not work.
- **Two agents make routing hard to distinguish from luck.** With an obviously
  two-part query, a coin flip is defensible about half the time. The
  supervisor's stated reasoning is the only evidence of judgement, which is why
  it is a required field rather than optional.
- **Synthesis cannot check itself.** It sees findings rather than the document,
  so a wrong inference drawn from two correct findings would pass every check.
- **Page scoping does work a real system would need retrieval for.** At 37
  pages the page holding each answer is known, so retrieval is solved by the
  requirement rather than by the system.

## Layout

```
config.yml                     # pdf url, page bindings, model, agent pages
.env                           # GROQ_API_KEY (gitignored; see .env.example)
prompts/                       # every prompt, edited without touching Python
  extraction.yaml              #   part 1, the version in use
  extraction_v1.yaml           #   part 1, first attempt - kept for comparison
  dates.yaml                   #   part 2: locate dates on pp.1, 36
  date_reasoning.yaml          #   part 2: classify vs the fixed reference
  supervisor.yaml              #   part 3: routing, reasoning before choice
  revenue_agent.yaml           #   part 3: revenue specialist
  expenditure_agent.yaml       #   part 3: expenditure specialist
  synthesis.yaml               #   part 3: forbids inventing a revenue link
evaluation/
  expected.yaml                # known-correct values for parts 1 and 2
  demo_queries.yaml            # part 3 queries and expected routing
src/
  config.py                    # config.yml + .env -> one settings object
  llm.py                       # get_chat_model(provider, model), Groq only
  evaluation.py                # Check/Report scoring for parts 1 and 2
  ingestion/
    download.py                # ensure_pdf(): fetch once, cache in data/
    parser.py                  # extract_pages(): pypdf, --- page N --- markers
  extraction/
    schemas.py                 # part 1 fields, each with page provenance
    prompts.py                 # load_prompt(name) -> ChatPromptTemplate
    extractor.py               # part 1 chain and entry point
    dates.py                   # part 2: find dates, normalise via the tool
    date_reasoning.py          # part 2: LLM classifies, checker verifies
  tools/
    date_tool.py               # normalize_date, classify_date as @tool
    mcp_server.py              # the same tools over stdio via FastMCP
    mcp_client.py              # MCP session helpers and tool listing
  agents/
    base.py                    # AgentReport, run_agent()
    revenue_agent.py           # thin wrapper over base, pages 9, 13, 15
    expenditure_agent.py       # thin wrapper over base, pages 16, 18, 20
    supervisor.py              # RouteDecision, guards, route()
  graph/
    state.py                   # SupervisorState, Finding, Decision, NodeCost
    trace.py                   # Trace: table(), summary(), render()
    workflow.py                # build_graph(), run_query(), stream_trace()
    evaluation.py              # score_part3() and the four checks
notebooks/                     # evidence for each decision (table above)
tests/                         # 185 tests; model stubbed, no network
```
