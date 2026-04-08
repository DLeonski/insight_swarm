# Insight Swarm v2 — Design Specification

**Date:** 2026-04-08
**Status:** Approved
**Goal:** Redesign Insight Swarm as a pure pandas multi-agent orchestration tool with a web UI — no SQL, no database connectors, no terminal required.

---

## 1. Overview

Insight Swarm v2 is a locally-running web app. The user drags a CSV or XLSX file onto a browser page, optionally types a question, and receives a professionally formatted HTML report produced by a swarm of locally-running LLM agents. No SQL is generated. No technical knowledge is required.

**What changes from v1:**
- CLI replaced by a FastAPI web server with drag-and-drop UI
- SQL pipeline (sql_writer, sql_retry, executor) deleted entirely
- DuckDB as input source deleted
- Data slicing for hypothesis testing done by an LLM agent that writes and executes pandas code
- Web search results interpreted by a dedicated LLM agent before reaching the evaluator
- Hypothesis testing restructured as an Evaluator-Optimizer loop

---

## 2. Web Interface

`python -m insight_swarm` starts a FastAPI server on `http://localhost:8000` and opens it in the default browser automatically.

### 2.1 Single-page UI (`static/index.html`)

Fields:
- **File drop zone** — drag-and-drop or click-to-browse, accepts `.csv` and `.xlsx` only
- **Question** (optional) — "What do you want to know?" If empty, defaults to `"Analyse this dataset and identify the most important business insights."`
- **Industry** (optional) — e.g. "retail"
- **Context** (optional) — e.g. "US market, 2014-2016"
- **Run Analysis** button

On click: file is uploaded via POST to `/run`. The page switches to a live progress view fed by Server-Sent Events (SSE) from `/progress/{run_id}`. Each LangGraph node emits a progress event on completion. When `report_node` finishes, a **View Report** button appears linking to the saved HTML file.

### 2.2 Server (`server.py`)

- FastAPI app
- `POST /run` — accepts multipart file upload + form fields, saves file to `uploads/`, starts LangGraph graph in a background thread, returns `run_id`
- `GET /progress/{run_id}` — SSE stream; each node calls a shared `emit(run_id, message)` helper on completion
- `GET /reports/{filename}` — serves generated HTML report files
- On startup: calls `webbrowser.open("http://localhost:8000")`

---

## 3. LangGraph Architecture

### 3.1 Graph (CSV / XLSX)

```
START
  ↓
schema_node       — load file with pandas, compute full data summary
  ↓
analyst_node      — interprets summary, writes initial analysis
  ↓
proposer_node     — generates ONE hypothesis per call
  ↓ conditional: internal or external?
  ├─ internal ──→ data_slicer_node   — deepseek-coder:6.7b writes + runs pandas code
  │               ↓                    returns list[dict] result
  │             evaluator_node       — reads hypothesis + data slice
  │               ↓ conditional
  │               ├─ data_request (evaluator_rounds < 3) → data_slicer_node
  │               └─ verdict → store on hypothesis, back to proposer
  │
  └─ external ──→ web_search_node    — DuckDuckGo fetch only (no LLM)
                  ↓
                web_analyst_node     — llama3.1:8b filters snippets, extracts relevant context
                  ↓
                evaluator_node       — reads hypothesis + filtered web context → verdict
                  ↓
                back to proposer
  ↓ (3 hypotheses done OR cycle_count >= max_cycles)
narrator_node     — executive summary
  ↓
report_node       — HTML + JSON log
  ↓
END
```

### 3.2 Conditional Edges

| After node | Condition | Routes to |
|---|---|---|
| `schema_node` | always | `analyst_node` |
| `analyst_node` | always | `proposer_node` |
| `proposer_node` | hypothesis type == "internal" | `data_slicer_node` |
| `proposer_node` | hypothesis type == "external" | `web_search_node` |
| `proposer_node` | all hypotheses done OR cycle_count >= max_cycles | `narrator_node` |
| `evaluator_node` | outputs data_request AND evaluator_rounds < 3 | `data_slicer_node` |
| `evaluator_node` | outputs verdict | `proposer_node` |
| `web_analyst_node` | always | `evaluator_node` |

---

## 4. Shared State

```python
class Hypothesis(TypedDict):
    text: str
    type: Literal["internal", "external"]
    data_slice: Optional[list[dict]]   # last pandas result used by evaluator
    web_context: Optional[str]         # filtered web summary from web_analyst
    verdict: Literal["confirmed", "rejected", "inconclusive"]
    evidence: str                       # specific data or text the evaluator cited

class LogEntry(TypedDict):
    node: str
    timestamp: str
    duration_ms: int
    summary: str
    detail: Any

class InsightState(TypedDict):
    # Input
    question: str
    industry: Optional[str]
    context: Optional[str]
    max_cycles: int

    # File
    source_file: str        # path to uploaded file
    source_type: str        # "csv" | "xlsx"
    df_path: str            # same as source_file; kept for clarity in data_slicer

    # Schema & summary
    schema: dict            # {col: dtype} flat dict of the primary table
    data_summary: list[dict]  # full precomputed groupby summary from schema_node

    # Analysis
    analyses: list[str]

    # Hypotheses
    hypotheses: list[Hypothesis]          # completed hypotheses with verdicts
    current_hypothesis: Optional[Hypothesis]  # hypothesis currently being evaluated
    evaluator_rounds: int                 # resets to 0 per hypothesis
    cycle_count: int

    # Evaluator ↔ Data Slicer exchange
    data_request: Optional[dict]          # {group_by: [str], metric: str, filter?: {str: Any}}
    current_data_slice: list[dict]        # result from data_slicer for current round

    # Output
    narrative: str
    run_log: list[LogEntry]
```

**Removed from v1 state:** `db_path`, `current_sql`, `sql_error`, `sql_retry_count`, `query_context`, `current_query_goal`, `sql_results`

---

## 5. Agent Nodes

### schema_node
- Reads CSV (UTF-8 with latin-1 fallback) or XLSX (all sheets, use first sheet for summary)
- Computes `data_summary`: for every categorical column × every numeric column, compute grouped sum and mean. Also compute overall `describe()`.
- Stores flat `schema` dict: `{column_name: dtype_string}` — used by data_slicer to know available columns.
- Emits SSE: `"Schema loaded: N rows, M columns"`

### analyst_node
**Model:** `llama3.1:8b`

Receives: `data_summary` as markdown table, `question`, `industry`, `context`.
Writes 2-3 sentence interpretation focusing on patterns, anomalies, and business implications.
Output appended to `analyses`.
Emits SSE: `"Initial analysis complete"`

### proposer_node
**Model:** `llama3.1:8b`

On first call: generates hypothesis 1 from analyst text.
On subsequent calls: generates next hypothesis informed by all previous verdicts and evidence — the proposer learns from what the evaluator found.
Generates max 3 hypotheses total. Each tagged `internal` (testable from the dataset) or `external` (requires world knowledge).
When 3 hypotheses are done or `cycle_count >= max_cycles`: routes to narrator.
Emits SSE: `"Hypothesis N/3: <text>"`

### data_slicer_node
**Model:** `deepseek-coder:6.7b`

Receives: hypothesis text + `schema` (column names + types) + `data_request` (if set by evaluator) or generates a default slice for the hypothesis.

Prompt instructs the model to write a single pandas script that:
- Loads the DataFrame (provided as variable `df` in the exec environment)
- Performs the requested groupby / filter
- Stores result in variable `result` as a list of dicts

Execution:
```python
local_env = {"df": df, "pd": pd}
exec(generated_code, local_env)
result = local_env["result"]
```

On exception: retry once with error message in prompt. If second attempt fails: `current_data_slice = []`.
Stores result in `current_data_slice`.
Emits SSE: `"Data slice computed"`

### evaluator_node
**Model:** `llama3.1:8b`

Receives: hypothesis text + `current_data_slice` (as markdown table) OR `web_context` (for external hypotheses).

Output is one of:
1. `{"verdict": "confirmed"|"rejected"|"inconclusive", "evidence": "<specific text/numbers>"}` — evaluation complete
2. `{"data_request": {"group_by": ["col1", "col2"], "metric": "sales", "filter": {"region": "West"}}}` — needs a different slice

If output is a data_request AND `evaluator_rounds < 3`: increment `evaluator_rounds`, store `data_request`, route back to `data_slicer_node`.
If output is a data_request AND `evaluator_rounds >= 3`: force verdict `inconclusive`, evidence = "Max data requests reached".
If output is verdict: store on `current_hypothesis`, append to `hypotheses`, increment `cycle_count`, reset `evaluator_rounds = 0`, route to `proposer_node`.

Parser: extract JSON from LLM response using regex `\{.*\}` with `re.DOTALL`. Fall back to `inconclusive` if parsing fails.
Emits SSE: `"H{n} verdict: {verdict}"`

### web_search_node
*(No LLM)*

Builds query: first 10 words of hypothesis + industry + years from context + `"market statistics trends report"`.
Fetches top 10 DuckDuckGo results.
Stores raw results in state for `web_analyst_node`.
Emits SSE: `"Web search: {n} results fetched"`

### web_analyst_node
**Model:** `llama3.1:8b`

Receives: raw DuckDuckGo snippets + hypothesis text + industry + context.
Task: filter out irrelevant results (local business listings, unrelated pages), extract only what is pertinent to the hypothesis, return a concise summary with source titles.
If nothing relevant found: returns `"No relevant external data found."`
Output stored in `current_hypothesis.web_context`.
Emits SSE: `"Web results filtered and interpreted"`

### narrator_node
**Model:** `llama3.1:8b`

Receives: question, industry, context, all `analyses`, all `hypotheses` with verdicts and evidence.
Writes executive summary in management consulting style (2-4 paragraphs). Notes any inconclusive findings and why.
Emits SSE: `"Executive summary written"`

### report_node
*(No LLM — Jinja2 + Plotly)*

- Renders self-contained HTML using Jinja2 template
- One card per hypothesis: verdict badge (green/red/yellow), evidence text, Plotly chart if data slice exists
- Collapsible sections showing the pandas code generated by data_slicer (for transparency)
- Saves `reports/report_YYYYMMDD_HHMMSS.html`
- Saves `logs/run_YYYYMMDD_HHMMSS.json`
- Emits SSE: `"done:{filename}"` — triggers View Report button in UI

---

## 6. Models & Dependencies

| Role | Model |
|---|---|
| Code generation (data slicer) | `deepseek-coder:6.7b` |
| Analysis, reasoning, narrative | `llama3.1:8b` |

**Dependencies (updated):**
```
langgraph
langchain
langchain-community
langchain-ollama
fastapi
uvicorn
python-multipart      # FastAPI file uploads
pandas
openpyxl
plotly
jinja2
duckduckgo-search
tabulate
```

**Removed:** `duckdb`

---

## 7. Project Structure

```
insight_swarm/
├── __main__.py           — starts FastAPI server, opens browser
├── server.py             — FastAPI app, /run, /progress/{id}, /reports/{file}
├── graph.py              — LangGraph graph definition (updated)
├── state.py              — InsightState, Hypothesis, LogEntry (updated)
├── nodes/
│   ├── schema.py         — updated: pandas only
│   ├── analyst.py        — updated: reads data_summary
│   ├── proposer.py       — new: replaces hypothesis.py
│   ├── data_slicer.py    — new: deepseek-coder writes + runs pandas code
│   ├── evaluator.py      — new: verdict or data_request output
│   ├── web_search.py     — updated: fetch only
│   ├── web_analyst.py    — updated: LLM filters raw results
│   ├── narrator.py       — unchanged
│   └── report.py         — updated: shows pandas code, not SQL
├── prompts/
│   ├── analyst.txt
│   ├── proposer.txt      — new
│   ├── data_slicer.txt   — new
│   ├── evaluator.txt     — new
│   ├── web_analyst.txt   — updated
│   └── narrator.txt
├── templates/
│   └── report.html.j2
├── static/
│   └── index.html        — new: drag-and-drop web UI
├── uploads/              — gitignored
├── reports/              — gitignored
├── logs/                 — gitignored
└── data/
    └── superstore.csv
```

**Deleted files:** `nodes/sql_writer.py`, `nodes/sql_retry.py`, `nodes/executor.py`

---

## 8. Error Handling

| Scenario | Behaviour |
|---|---|
| data_slicer generates broken code (attempt 1) | Retry with error message in prompt |
| data_slicer generates broken code (attempt 2) | `current_data_slice = []`, evaluator marks inconclusive |
| web_analyst finds nothing relevant | Sets web_context = "No relevant external data found" |
| DuckDuckGo rate-limited or fails | web_context = None, evaluator marks inconclusive |
| Evaluator requests data 3 times | Force inconclusive, evidence = "Max data requests reached" |
| Ollama model not available | Exit before graph with `ollama pull` instructions |
| Unsupported file type | SSE error event: "Supported formats: .csv, .xlsx" |
| File parse error | SSE error event with specific message |
| max_cycles reached | Force exit to narrator, noted in narrative |

---

## 9. Out of Scope

- Authentication or multi-user support
- PostgreSQL, MySQL, or other database connectors
- Persistent conversation / follow-up questions after report
- Fine-tuning or custom model training
- Streaming LLM output to the browser (progress is per-node, not per-token)
