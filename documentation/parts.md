# Parts 1-3: design and implementation

What was built for each part, what was measured, and the assumptions each
rests on. Overview, API and results are in the [README](../README.md).

## Part 1: extraction

Part 1 involves extracting five fields from their cited pages. As seen below, the fields consist different data types such as floats and list of strings.

| Field | Value | Unit | Page | Note |
|---|---|---|---|---|
| Corporate Income Tax | 28.4 | billion | 5 | Prose: stated in a sentence, so context disambiguates it |
| Year-on-year change | 17.0 | % | 5 | Prose: same sentence as the amount, stated not calculated |
| Total top-ups | 20,352 | million | 20 | **Table row**: a "Total" line under a ($ million) heading |
| Taxes in Operating Revenue | 7 names | - | 5-6 | Prose: names spread across several paragraphs |
| Overall Fiscal Position | -3.57 | billion | 8 | **Table row** : one figure per year column, with negative value in parentheses |

Two of the five are of table rows, which is a core criterion in choosing the most appropriate parser, as shown below. 

### 1.0 LLM and provider

The LLM used for extraction `llama-3.1-8b-instant`, served through Groq, at `temperature: 0`. All settings can be found in `config.yml`.

| Why Groq | |
|---|---|
| Free, no card | The constraint that excluded GPT-4 and Claude on cost, not merit |
| Nothing downloaded | Ollama needs 2-5 GB of disk and RAM per model |
| Structured output | Supports Pydantic schemas; `langchain-huggingface` raises `NotImplementedError` |
| Correct values | A local model returned `28400000000.0`, folding the unit into the value |
| The trade | 100k tokens/day - which is why answers are committed to `results/` rather than regenerated |

**Consideration 1: Temperature 0 is what makes prompt iteration measurable.** The model takes
the most likely token each time, so the same pages yield the same figures, and
a changed result can be attributed to the prompt rather than to sampling. To make it reproducible, temperature 0 is used.

### 1.1 Parser: pypdf

**Consideration: Page is used as the parsing unit.** Each target field is identified by the page
it appears on - "page 5", "page 20" - so pages are chosen as the natural unit, and [`extract_pages()`](../src/ingestion/parser.py) returns only those
requested, each prefixed with a `--- page N ---` marker. Those markers are what
make the page binding enforceable: the prompt can say PAGE 5 ONLY because the
model can see where page 5 ends.

Chunking by paragraph or token count would break that, leaving one
undifferentiated block with no way to tell which page a figure came from.

Five parsers were considered. Docling and OCR were ruled out on inspection -
Docling's ML layout analysis solves a problem a born-digital PDF does not have,
and OCR needs raster images, of which this document has none. The remaining
three were measured on **row integrity** and **latency**.

> **Row integrity** means the table row survives extraction as one line, the
> label still attached to its figures. The Corporate Income Tax row on page 8
> reads `Corporate Income Tax 23.07 24.26 28.38 23.0 17.0` in the document.

| Parser | Row integrity | Speed (4 pp.) | Verdict |
|---|:---:|---:|---|
| **pypdf** | Pass | 450 ms | **Chosen** - correct, and the faster of the two that work |
| pdfplumber | Pass | 823 ms | Rejected - correct but 1.8x slower |
| PyMuPDF | **Fail** - returns `Corporate Income Tax` with the figures detached | 52 ms | Rejected - fastest, but wrong |
| Docling | not tested | - | Ruled out - no scanned pages |
| OCR | not tested | - | Ruled out - zero raster images |

**Correctness decides; speed only breaks ties.** Two of the five target fields
are table rows, so a parser that detaches a label from its figures cannot be
used however fast it is.

### 1.1a Extracting structured information

The schema decides what shape the answer comes back in. Without it the model
returns a sentence to be parsed afterwards, and parsing free text is where a
wrong figure slips through unnoticed. The five fields are Pydantic models in
`src/extraction/schemas.py`, bound with `with_structured_output(ExtractionResult)`,
so the model fills a typed object rather than writing prose.

| In the schema | Why |
|---|---|
| `value: float` | A figure arrives as a number, so `28.4` cannot come back as "about $28 billion" |
| `unit: Literal["million", "billion"]` | Units as the page states them, never converted - p.20 says $million and everything else $billion, and normalising silently would hide a mismatch |
| `page: int = Field(gt=0)` | Every value carries where it was read from |
| `quote: str` | The verbatim text it was read from, with a validator rejecting a blank one |

Page and quote are what make the answer checkable: "Corporate Income Tax"
appears on several pages with different values, so a bare number cannot be
verified. The schema also rejects bad output rather than passing it on - a
missing field, a page of 0, an unrecognised unit or an empty quote all raise
instead of returning something that looks fine.

### 1.2 Prompt engineering
The next part of the task involves extracting the relevant information through prompt engineering in Large Language Model (LLM).
The prompt template can be found at `prompts/extraction.yaml`, which is a configurable file that is separated from the core architecture. Users can first fill in the page numbers and document text to be queried at `config.yml`, where the configurations will be adapted into the prompt template. The prompt template can also be edited. 

Configurations that are filled in `config.yml` 

| Filled in | From | Reaches the prompt as |
|---|---|---|
| Which page each field is bound to | `field_pages` in `config.yml`, via `page_variables()` | `{page_cit}`, `{page_top_ups}` - one placeholder per field |
| The text of those pages | `extract_pages()`, each page prefixed `--- page N ---` | `{page_text}` |

> Note: both the page number and the markers must be present. A page number
> binds nothing if the model cannot see where that page starts, and the markers
> bind nothing if the prompt never names a page.


**Consideration 1: Guardrails in the extraction prompt.** It is important to ensure that the LLM is grounded with the extraction with no hallucination of the extraction of numerical data. As such, the following areas are included as guardrails in the prompt 

| Guardrail | Wording | Closes |
|---|---|---|
| Scope binding | "PAGE 5 ONLY", one page named per field | A right value read from a page the field was never bound to |
| Copy, do not compute | "Do NOT calculate anything. Not a difference, not a percentage, not a sum" | A derived figure that is arithmetically sound and appears nowhere in the document |
| No prior knowledge | "Do NOT use prior knowledge of Singapore budgets" | The model answering from training rather than from the text supplied |
| Cite or omit | "If you are about to write a number, first find it verbatim in the text. If you cannot point to it, you must not write it" | An uncheckable number - every field carries the page and quote it came from |

**Consideration 2: Constraints are stated as limits, not preferences.** A rule
the model can read as advice gets treated as advice, so each is written as a
hard bound and scoped to the page it governs:

- **Bound to a page.** "Corporate Income Tax is on page 5" is a hint; "PAGE 5
  ONLY" is a limit. The prompt also names the trap - the top-ups total appears
  on p.8 as 24.32 billion, so it says to take p.20's 20,352 million rather than
  the first plausible match.
- **Scoped, not general.** *Prefer prose over tables* applies only on the cited
  page. Left general it competes with the citation and returns p.5's prose
  (-3.6) where the fiscal position is p.8's table figure (-3.57).
- **Withheld where stating it would give the answer.** The tax list is
  constrained to prose on the cited pages, but the expected count is left out -
  a scorer cannot check an answer the prompt supplied.

### 1.3 Assumptions

**Only the cited pages are read.** A correctness measure, not an optimisation.For instance, "Corporate Income Tax" appears eight times across seven pages:

| Page | Value | What it is |
|---|---|---|
| 5 | **28.4** | Revised FY2023, prose - the answer |
| 8 | 28.38 | same figure, table precision |
| 8 | 23.07 / 24.26 | FY2022 actual / FY2023 estimated |
| 9 | 27.2% | share of Operating Revenue |
| 16 | 28.03 | Estimated FY2024 |
| 26 | 28,380 / 28,029 | same figures in $million |
| 27 | 3.9% | share of GDP |

Restricting the input eliminates seven wrong answers before the
model reads anything.

**The tax list counts revenue lines, not their constituents.** Pages 5-6 name
seven taxes as Operating Revenue components. Four more appear inside the
description of one of them - "Other Taxes, which include the Foreign Worker
Levy, Water Conservation Tax, Land Betterment Charge, and Annual Tonnage Tax"
(p.5) - and are not counted separately, since the document presents them as
what Other Taxes consists of rather than as revenue lines in their own right.


## Part 2: dates and tools

Part 2 extends from part 1 by including MCP tools and LLM reasoning. In this part, date related information from two fields are extracted, normalized into ISO format through MCP tool and LLM is used for reasoning by comparing against a reference date `2024-01-01`. The comparison and fields are shown below

| Status | Means | For a single date | For a period |
|---|---|---|---|
| Expired | Already passed | Falls before the reference | Its end falls before the reference |
| Upcoming | Still to come | Falls after the reference | Its start falls after the reference |
| Ongoing | A period currently active | Only when it *is* the reference - a point has no span to be inside | The reference falls within it, inclusive |

| Field | Page | Normalised | Status vs 2024-01-01 |
|---|---|---|---|
| Distribution | 1 | 2024-02-16 | Upcoming |
| Estate Duty | 36 | 2008-02-15 | Expired |

### 2.1 How the dates are extracted

| # | Step | Who | What happens |
|---|---|---|---|
| 1 | Select pages | code | `date_pages` binds each date to a page - 1 and 36 - and `extract_pages()` returns only those |
| 2 | Extracts date | model | Fills `DocumentDates`: the sentence verbatim, the date **exactly as written**, and the page |
| 3 | Normalise | tool | [`normalize_date`](../src/tools/date_tool.py) turns "16 February 2024" into `2024-02-16`, called over the local MCP server |
| 4 | Classify | model | LLM classifies extracted data against `reference_date` into Expired / Upcoming / Ongoing |
| 5 | Verify | tool | `classify_date` (deterministic Python tool) recomputes the status and flags any disagreement |

**Only step 3 crosses the MCP protocol,** and `normalize_date` is the only tool
the server exposes. It runs over the server as a subprocess, with the
in-process `@tool` as a fallback; `main` reports which route it used.

Built with **FastMCP**, where `@mcp.tool()` builds a tool's schema from its
signature and docstring. Transport is **stdio**: `mcp_client` runs
[`mcp_server`](../src/tools/mcp_server.py) as a subprocess, sending requests to its
stdin and reading replies from its stdout, which is why nothing in the server
prints.


The answer is written to `results/dates.json`:

```json
[
  {
    "original_text": "Distributed on Budget Day: 16 February 2024",
    "normalized_date": "2024-02-16",
    "status": "Upcoming",
    "reasoning": "2024 is the same year as 2024, February is earlier than January, 16 is earlier than 1"
  },
  ...
]
```

Both statuses are correct and both were verified by `classify_date`. The
`reasoning` on the first is not - the model compared the fields in the wrong
direction and arrived at the right answer anyway. It is recorded rather than
cleaned up, and is the clearest argument for step 5: the status is trusted
because a tool recomputed it, not because the explanation reads well.

`reasoning` is used to record the comparison the model made, so the status can be
audited rather than taken on trust. Without it, a wrong classification looks
exactly like a right one.


**Consideration 1: the LLM is grounded in the document's own words.** The model
never converts a date and the tool never interprets one. Step 2 hands over the
wording as the document writes it, so `normalize_date` parses that rather than
the model's idea of it - a full `strptime` match first, then a regex to pull a
date out of a longer sentence. Anything unparseable returns `None`, never a
guess.

**Consideration 2: a failure is surfaced, not absorbed.** Both dates are stated
plainly on their cited pages, so no retry or fallback search runs if the model
misses one. A missing date fails schema validation; a date `normalize_date`
cannot parse is reported as skipped rather than classified.

### 2.2 Assumptions

**Dates are explicit calendar dates.** Open-ended expressions - "till present",
"with immediate effect" - are out of scope; `normalize_date` returns None
rather than inventing a boundary.

**Dates are written in full.** The document spells them out - "16 February
2024" - as a government publication does, so that is the format the tool is
built for. Three further shapes are accepted due to the needs of the document which are:

| Shape | Example | |
|---|---|---|
| Day month year | 16 February 2024 | The document's own form |
| Month day, year | February 16, 2024 | Accepted |
| ISO | 2024-02-16 | Accepted |
| Slashed, day first | 16/02/2024 | Accepted, read day first - a US-style 02/16/2024 is misread rather than rejected |
| Abbreviated month | 16 Feb 2024 | Accepted |
| Ordinals, partial or non-English dates | 16th February 2024; "2024"; "16 Février" | Not parsed |


## Part 3: multi-agent supervisor

Part 3 focuses on handling complex queries such as *"What are the key revenue streams, and how will
the Future Energy Fund be supported?"*, where the question spans across two subjects sitting on different
pages.

A multi-agent system is built using a Langgaraph architecture which consists of a supervisor that routes and two
specialist agents (revenue agent and expenditure agent) each bound to their own pages. There are also two terminal nodes, which are
synthesis, which writes the answer, and decline, for a query the document
cannot address.

A run of the query above generally takes three turns:

| Turn | Potential routes | Definition |
|---|---|---|
| 1 | `revenue_agent` | Specialises in identifying and extracting information on revenue, from pp. 9, 13, 15 |
| 2 | `expenditure_agent` | Specialises in finding and analysing information on government spending, from pp. 16, 18, 20 |
| 3 | `synthesis` | Writes the answer from the two findings, citing the figures each agent reported |

> Note 1: a query covering one subject only takes two turns - one agent, then
> synthesis. The turn count follows from how many subjects the question spans,
> but synthesis always runs, since it is what turns findings into an answer.

> Note 2: six pages covering the information the query needs are fixed in
> `config.yml`, three per agent. Narrowing what each agent can see is what keeps
> a plausible figure from the wrong page out of reach.


**How routing works**

**Reasoning** Every turn, including
after an agent has reported, the model is given the query, which agents have
already been consulted, their summaries so far, and the page ranges each agent
covers. It returns a `RouteDecision`: a stated rationale, the chosen route, and
the sub-task for that agent. `reasoning` is declared before `next` in the
schema, so the justification is produced before the choice rather than fitted
to it afterwards.

**Deterministic check through a guard.** The model's answer is not routed
directly. `route()` in [supervisor.py](../src/agents/supervisor.py) screens it in
plain Python - a list membership test and an integer comparison, no model
involved - so the graph terminates, never answers from nothing, and never
discards what it gathered:

| Condition | Forces |
|---|---|
| The chosen agent has already reported | `synthesis` |
| `max_turns` reached | `synthesis` |
| `synthesis` chosen with no findings | An agent, so there is something to synthesise |
| `out_of_scope` chosen with findings present | `synthesis` |

When it fires, the trace keeps both the choice and the override, so a forced route never reads as the model's decision. 

### 3.2 Inside one agent turn

What an agent does with a sub-task, taken from the saved run in
`results/supervisor.json`. The supervisor routed to `revenue_agent` with the
sub-task *"list the key government revenue streams"*:

| Step | What happens |
|---|---|
| 1 | [`extract_pages()`](../src/ingestion/parser.py) loads specific pages for a specific scope, pp. 9, 13, 15 only, each prefixed `--- page N ---` |
| 2 | The page text and the sub-task fill the agent's prompt - the document never enters graph state |
| 3 | The model returns an `AgentReport`: a prose summary plus a list of figures, each with value, unit, page and the quote it was read from |
| 4 | [`page_of_quote`](../src/ingestion/parser.py) resets each figure's page to the marker section its quote actually falls under |
| 5 | The report becomes a `Finding`, tagged with the agent's name and the pages it read, and appended to state |

Step 4 earns its place: the model recorded NIRC as p.13 while quoting p.15's
sentence, and both pages state the figure, so the value gives no sign anything
is wrong. Only the quote does, and since it must be verbatim, the page follows
from it.

The finding returns to the supervisor, not to the other agent - each agent sees
only its own pages and its own sub-task. The supervisor then writes the next
brief: here it routed to the expenditure agent for the second part of the query,
which repeats the same five steps over pp. 16, 18, 20.

### 3.3 Demo queries

Seven queries exercise the routes the graph can take - one agent, both agents,
and declined. They are configured in `expectations/demo_queries.yaml`, and the
scores for each are under [Results](../README.md#results).

**Consideration 1: Choice of specific multiple pages.** The six pages are
shortlisted from the query and fixed in `config.yml`. It spans across revenue and expenditure, which sit in
different sections, and each subject then spans pages of its own such as the total,
its composition and the supporting detail are all stated separately.

| Agent | Pages | What is on them |
|---|---|---|
| `revenue_agent` | 9, 13, 15 | Revenue breakdown, the FY2024 narrative, NIRC |
| `expenditure_agent` | 16, 18, 20 | Table 2.1, the top-ups prose, Table 2.4 |

As the information is disjoint and spans pages, the split into a narrow range is also what
gives each agent a subject it can answer and the other cannot.

> Note: page 13 carries both the revenue total and the top-ups sentence, and is
> assigned to `revenue_agent` alone. That keeps the combined query out of reach
> of either agent on its own.

**Consideration 2: Restricting the input is a correctness measure** rather than
an optimisation. Several target terms recur across the document with different
values as seen in part 1, so an agent granted the full text will surface a figure that is
plausible in isolation and wrong in context - with nothing in the value itself
to indicate which.

Fixing the page sets in configuration, rather than letting an agent retrieve
what it judges relevant, is also what makes the citations checkable:
`check_page_discipline` compares every cited page against the agent's
configured set, and a figure from outside it was never read. Under retrieval any
page would be legitimate, so the check would have nothing to assert.


### 3.4 Evaluation

As the generated answer is free text, there is no single correct wording to
score it against. Five deterministic checks, written in Python rather than
judged by a model, are applied to the run instead. Each is binary (pass or fail), and a query
passes only when all five do.

| Check | Catches | Compared against |
|---|---|---|
| Routing | An agent that should have run and did not, or one that ran needlessly | Expected agents in `demo_queries.yaml` |
| Figures | A wrong value, or the right value with the wrong unit | Required values in `demo_queries.yaml`, within a 0.005 tolerance |
| Traceability | A quote that does not appear on the page it cites | The text of the page it cites |
| Page discipline | A figure from a page the agent was never given | The agent's page set in `config.yml` |
| Labels | A figure renamed on its way into the answer | The label the finding gave it |

> Note: the last three compare against the document, the configuration and the
> findings rather than declared expectations, so they hold for any query.

### 3.5 Assumptions

**Each agent runs at most once per query.** One pass over an agent's pages is
assumed to be enough, which keeps the graph provably terminating and the cost
bounded at one model call per agent. What is given up is a second call under a
*different* sub-task, which might surface something the first was not asked for.

**Each query is answered from scratch, with no LangGraph memory.** Out of scope
here: every query is self-contained, so no checkpointer and no `thread_id`.
`SupervisorState` carries findings and decisions between nodes within one run
and is discarded when it returns - short-term state, not memory. Persisting it
would break the guard, which reads `visited` as "who has reported on this
query".

**Part 3 uses JSON mode rather than tool-calling.** The supervisor and both
agents pass `method="json_mode"`; Parts 1 and 2 use LangChain's default, which
is tool-calling. The difference is generation length. Part 3 asks for a stated
rationale before the choice, and over that length the tool-call wrapper drifts
from the format Groq's parser accepts, which leads to the request being rejected even when the
content is correct. Parts 1 and 2 emit short structured objects and never hit
it, so they are left on the default rather than changed for symmetry.

**The model quotes accurately but may cite the wrong page.** It is assumed to be
grounded in the text it was given, so the quotes it returns are verbatim. The
page attached to them is less reliable as related figures appear on several
pages, and the model can read one and record another.
[`page_of_quote`](../src/ingestion/parser.py) replaces the model's page with the
one whose text contains the quote. Where no match is found, the model's page is
kept and the traceability check reports it.
