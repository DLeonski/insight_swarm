# Insight Swarm — Design Specification

**Date:** 2026-04-08  
**Status:** Approved  
**Goal:** Multi-agent business analysis system — "replace 2 hours of analyst work with one prompt"

---

## 1. Overview

Insight Swarm is a CLI tool that takes a business question and a data file, then orchestrates a swarm of LLM agents to write SQL, analyse results, generate and test hypotheses (including web search for external context), and produce a self-contained HTML report with charts.

**Primary goals:**
- Portfolio piece demonstrating LangGraph multi-agent orchestration
- Learning project for understanding cyclic graphs, conditional edges, and local LLM integration

---

## 2. CLI Interface

```bash
python -m insight_swarm \
  --file sales.csv \          # or --db mydata.duckdb
  --question "Why did sales drop in Q3?" \
  --industry "retail" \       # optional — enriches agent prompts
  --context "US market, 2022-2023, B2C" \  # optional — additional context
  --max-cycles 5              # optional — safety guard, default 5
```

**Rules:**
- Either `--file` (CSV or XLSX, auto-loaded into DuckDB) or `--db` (existing DuckDB file) is required.
- `--industry` and `--context` are optional. Without them, agents work from schema alone.
- `--max-cycles` caps the hypothesis testing loop to prevent infinite cycles.

---

## 3. LangGraph Architecture

### 3.1 Graph Structure

The graph is **cyclic** with conditional edges. The hypothesis testing loop repeats until all hypotheses are tested or `max_cycles` is reached.

```
START (question, file/db, industry?, context?, max_cycles)
  ↓
[ schema_node ]         — discovers tables, columns, types, row counts
  ↓
[ sql_writer_node ]     — deepseek-coder:6.7b generates initial query
  ↓
[ executor_node ]       — runs SQL against DuckDB, returns DataFrame
  ↓ conditional: valid SQL?
  ├─ error & retries < 2  →  [ sql_retry_node ]  →  executor_node
  ├─ error & retries ≥ 2  →  mark query failed, continue
  ↓ success
  ↓ conditional: sufficient data? (threshold: ≥ 10 rows)
  ├─ insufficient  ──────────────────────────────────────────────┐
  ↓ sufficient                                                   │
[ analyst_node ]        — llama3.1:8b interprets DataFrame        │
  ↓                                                              │
[ hypothesis_node ]     — llama3.1:8b generates hypotheses        │
                          each tagged: type = "internal"|"external"│
  ↓ conditional: next hypothesis type + cycle_count < max_cycles? │
  ├─ external  ────────────────────────────────────────────────┐ │
  ├─ cycle_count ≥ max_cycles  →  narrator_node (force exit)   │ │
  ↓ internal                                                   │ │
  sql_writer_node → executor_node → insufficient? ─────────────┤ │
                                                               ↓ ↓
                                              [ web_search_node ]
                                              — DuckDuckGo search
                                              — query built from hypothesis
                                                + industry + context
                                                               ↓
                                              [ web_analyst_node ]
                                              — llama3.1:8b merges
                                                SQL results (if any)
                                                + web context
                                               ↓ [all hypotheses done]
[ narrator_node ]       — llama3.1:8b writes executive summary
  ↓
[ report_node ]         — generates HTML + run_TIMESTAMP.json
  ↓
END  — opens report in browser
```

### 3.2 Conditional Edges

| Edge | Condition | Routes to |
|---|---|---|
| After `executor_node` | `sql_error` is set and `sql_retry_count < 2` | `sql_retry_node` |
| After `executor_node` | `sql_error` is set and `sql_retry_count >= 2` | mark query as failed in log, set `sql_results` entry to `None`, route to `analyst_node` (initial query) or `hypothesis_node` (hypothesis query) |
| After `executor_node` | rows returned `< 10` | `web_search_node` |
| After `executor_node` | rows returned `>= 10` | `analyst_node` |
| After `hypothesis_node` | `cycle_count >= max_cycles` | `narrator_node` (force exit) |
| After `hypothesis_node` | next hypothesis type is `"external"` | `web_search_node` |
| After `hypothesis_node` | next hypothesis type is `"internal"` | `sql_writer_node` |
| After `hypothesis_node` | all hypotheses tested | `narrator_node` |
| After `web_analyst_node` | more hypotheses pending | back to `hypothesis_node` |
| After `web_analyst_node` | all done | `narrator_node` |

---

## 4. Shared State

```python
class Hypothesis(TypedDict):
    text: str
    type: Literal["internal", "external"]
    sql_result: Optional[DataFrame]
    web_context: Optional[str]
    verdict: Literal["confirmed", "rejected", "inconclusive"]

class LogEntry(TypedDict):
    node: str
    timestamp: str
    duration_ms: int
    summary: str          # short human-readable description
    detail: Any           # node-specific payload (SQL, hypotheses, etc.)

class InsightState(TypedDict):
    # Input
    question: str
    industry: Optional[str]
    context: Optional[str]
    max_cycles: int

    # Schema
    schema: dict          # {table_name: [{column, type}]}

    # SQL execution
    current_query_goal: str
    current_sql: Optional[str]
    sql_error: Optional[str]
    sql_retry_count: int  # reset to 0 per hypothesis
    sql_results: list[DataFrame]

    # Analysis
    analyses: list[str]

    # Hypotheses
    hypotheses: list[Hypothesis]
    current_hypothesis_idx: int
    cycle_count: int

    # Output
    narrative: str
    run_log: list[LogEntry]
```

---

## 5. Agent Nodes

### schema_node
Handles file loading based on extension before running schema discovery:

| Input | Strategy |
|---|---|
| `.csv` | `duckdb.read_csv_auto(file)` — native, no extra dependencies |
| `.xlsx` | `pandas.read_excel(file, sheet_name=None)` → each sheet registered as a separate DuckDB table via `con.register(sheet_name, df)` |
| `.duckdb` | `duckdb.connect(file)` — direct connection |

After loading:
- Runs `SHOW TABLES` and `DESCRIBE <table>` for each table.
- Records table names, column names/types, and `SELECT COUNT(*)` per table.
- Writes schema dict to state.

### sql_writer_node
**Model:** `deepseek-coder:6.7b` via `langchain_community.llms.Ollama`

System prompt includes: schema, question, industry, context, current query goal.  
If `sql_error` is set (retry path): includes the error message and previous SQL so the model can correct it.  
Output: raw SQL string.

### sql_retry_node
- Increments `sql_retry_count`.
- Sets `current_query_goal` to include the error context.
- Routes back to `executor_node`.

### executor_node
- Runs `current_sql` against DuckDB.
- On success: appends DataFrame to `sql_results`, clears `sql_error` and `sql_retry_count`.
- On error: sets `sql_error` with the DuckDB exception message.
- Checks row count for the sufficiency conditional.

### analyst_node
**Model:** `llama3.1:8b`

Receives: schema, question, industry, context, latest DataFrame (as markdown table, truncated to 50 rows).  
Output: natural language interpretation appended to `analyses`.

### hypothesis_node
**Model:** `llama3.1:8b`

On first call: generates 3 hypotheses from the initial analysis.  
On subsequent calls: advances `current_hypothesis_idx` and resets `sql_retry_count` to 0 (each hypothesis gets a fresh retry budget). Does not regenerate the hypothesis list.  
Each hypothesis is tagged `internal` (testable via SQL against the dataset) or `external` (requires knowledge outside the dataset — holidays, market events, competitor actions).  
Force-exits to narrator if `cycle_count >= max_cycles`.

### web_search_node
- Builds a search query from: hypothesis text + industry + context + question keywords.
- Runs DuckDuckGo search via `duckduckgo-search` package (top 5 results).
- Stores raw snippets in `current_hypothesis.web_context`.

### web_analyst_node
**Model:** `llama3.1:8b`

Merges available SQL results (if any) with web context snippets.  
Produces a verdict (`confirmed`, `rejected`, `inconclusive`) and a short explanation.  
Appended to the current hypothesis and to `analyses`.

### narrator_node
**Model:** `llama3.1:8b`

Receives: question, industry, context, all analyses, all hypotheses with verdicts.  
Writes an executive summary in the style of a management consulting report (2–4 paragraphs).

### report_node
- Renders self-contained HTML using a Jinja2 template.
- Embeds Plotly charts (one per confirmed/rejected hypothesis with data).
- Includes collapsible SQL sections for each query.
- Saves `reports/report_YYYYMMDD_HHMMSS.html`.
- Saves `logs/run_YYYYMMDD_HHMMSS.json` with full `run_log`.
- Opens the HTML file in the default browser (`webbrowser.open`).

---

## 6. Models & Dependencies

| Role | Model | Why |
|---|---|---|
| SQL generation | `deepseek-coder:6.7b` | Best local model for SQL |
| Analysis, reasoning, narrative | `llama3.1:8b` | Strong reasoning, good instruction following |

All models accessed via `langchain_community.llms.Ollama`. Swapping to OpenAI requires only changing the LLM constructor — prompts and node logic remain unchanged.

**Key dependencies:**
```
langgraph
langchain
langchain-community
duckdb
pandas
openpyxl   # pandas XLSX backend
plotly
jinja2
duckduckgo-search
ollama  # for pulling models
```

---

## 7. Project Structure

```
insight_swarm/
├── __main__.py           # CLI entrypoint (argparse)
├── graph.py              # LangGraph graph definition and compilation
├── state.py              # InsightState, Hypothesis, LogEntry TypedDicts
├── nodes/
│   ├── schema.py
│   ├── sql_writer.py
│   ├── sql_retry.py
│   ├── executor.py
│   ├── analyst.py
│   ├── hypothesis.py
│   ├── web_search.py
│   ├── web_analyst.py
│   ├── narrator.py
│   └── report.py
├── prompts/
│   ├── sql_writer.txt
│   ├── analyst.txt
│   ├── hypothesis.txt
│   ├── web_analyst.txt
│   └── narrator.txt
├── templates/
│   └── report.html.j2    # Jinja2 report template
├── reports/              # generated HTML reports (gitignored)
├── logs/                 # generated JSON run logs (gitignored)
└── data/
    └── superstore.csv    # sample dataset for demo
```

---

## 8. Terminal Output (live progress)

Each node prints a status line on completion:

```
✓ Schema discovered: 3 tables, 21 columns
→ SQL Writer: generating initial query...
✓ Query executed: 847 rows
→ Analyst: interpreting results...
✓ Hypotheses generated: 3 (2 internal, 1 external)
→ [H1/3 internal] SQL Writer: testing "Seasonal effect in West region"...
✓ H1 confirmed
→ [H2/3 internal] SQL Writer: testing "Furniture category decline"...
✓ H2 confirmed
→ [H3/3 external] Web search: "New competitor Q3 2023 retail US"...
✓ H3 inconclusive (no data found)
→ Narrator: writing executive summary...
✓ Report saved: reports/report_20260408_143021.html
✓ Opening in browser...
```

---

## 9. Error Handling

| Scenario | Behaviour |
|---|---|
| SQL syntax error (retry < 2) | Retry with error message in prompt |
| SQL syntax error (retry ≥ 2) | Mark query as failed, log, continue |
| DuckDB file not found | Exit with clear error message before graph starts |
| CSV parse error | Exit with clear error message before graph starts |
| XLSX parse error | Exit with clear error message listing sheet names if readable |
| Unsupported file extension | Exit with message: "Supported formats: .csv, .xlsx, .duckdb" |
| Ollama model not available | Exit with message listing `ollama pull` commands needed |
| DuckDuckGo rate limit | Catch exception, set `web_context = None`, mark `inconclusive` |
| `max_cycles` reached | Force exit to narrator with note in narrative |

---

## 10. Output Files

### HTML Report (`reports/report_YYYYMMDD_HHMMSS.html`)
- Self-contained (all CSS, JS, Plotly embedded inline)
- Sections: header, executive summary, findings (one card per hypothesis), collapsible SQL queries
- Hypothesis cards show: verdict badge, supporting chart (Plotly), key finding text

### JSON Run Log (`logs/run_YYYYMMDD_HHMMSS.json`)
```json
{
  "question": "Why did sales drop in Q3?",
  "industry": "retail",
  "context": "US market, 2022-2023",
  "total_duration_ms": 34821,
  "hypotheses_tested": 3,
  "sql_retries_total": 1,
  "cycle_count": 3,
  "steps": [
    {
      "node": "sql_writer",
      "timestamp": "2026-04-08T14:30:01Z",
      "duration_ms": 2341,
      "retries": 0,
      "output": "SELECT ..."
    }
  ]
}
```

---

## 11. Out of Scope

- Web UI or streaming interface (CLI only)
- Authentication or multi-user support
- Persistent conversation / follow-up questions
- Fine-tuning or custom model training
- Support for databases other than DuckDB (PostgreSQL, MySQL, etc.)
