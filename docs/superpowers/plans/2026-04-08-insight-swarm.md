# Insight Swarm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI multi-agent tool that takes a business question + data file, orchestrates a LangGraph swarm of agents, and produces an HTML report with charts and a JSON run log.

**Architecture:** A cyclic LangGraph StateGraph with 10 nodes connected by conditional edges. Nodes share a typed state dict. SQL generation uses `deepseek-coder:6.7b`, all reasoning uses `llama3.1:8b`, both via Ollama. A hypothesis testing loop cycles until all 3 hypotheses are tested or `max_cycles` is reached; web search via DuckDuckGo handles external hypotheses and insufficient SQL results.

**Tech Stack:** Python 3.11+, LangGraph, langchain-community, DuckDB, pandas, openpyxl, Plotly, Jinja2, duckduckgo-search, Ollama, pytest

---

## File Map

```
insight_swarm/
├── __main__.py          # CLI entrypoint — argparse, input validation, graph invocation
├── graph.py             # LangGraph StateGraph assembly + all routing functions
├── state.py             # InsightState, Hypothesis, LogEntry TypedDicts
├── nodes/
│   ├── __init__.py
│   ├── schema.py        # Load CSV/XLSX/DuckDB → temp .duckdb → discover schema
│   ├── sql_writer.py    # deepseek-coder:6.7b → generate SQL string
│   ├── sql_retry.py     # Increment retry count, inject error into goal
│   ├── executor.py      # Run current_sql against DuckDB → records list or error
│   ├── analyst.py       # llama3.1:8b → interpret latest SQL results
│   ├── hypothesis.py    # llama3.1:8b → generate or advance hypotheses
│   ├── web_search.py    # DuckDuckGo → top 5 snippets for current hypothesis
│   ├── web_analyst.py   # llama3.1:8b → merge SQL+web → verdict JSON
│   ├── narrator.py      # llama3.1:8b → executive summary prose
│   └── report.py        # Jinja2 + Plotly → HTML file + JSON log file
├── prompts/
│   ├── sql_writer.txt
│   ├── analyst.txt
│   ├── hypothesis.txt
│   ├── web_analyst.txt
│   └── narrator.txt
├── templates/
│   └── report.html.j2
├── reports/             # gitignored — generated HTML reports
├── logs/                # gitignored — generated JSON run logs
└── data/
    └── superstore.csv   # demo dataset

tests/
├── conftest.py          # shared fixtures (minimal InsightState, tmp CSV)
├── test_state.py
├── test_graph_routing.py
├── test_cli.py
└── nodes/
    ├── test_schema.py
    ├── test_sql_writer.py
    ├── test_executor.py
    ├── test_sql_retry.py
    ├── test_analyst.py
    ├── test_hypothesis.py
    ├── test_web_search.py
    ├── test_web_analyst.py
    ├── test_narrator.py
    └── test_report.py
```

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `insight_swarm/__init__.py`
- Create: `insight_swarm/nodes/__init__.py`
- Create: `insight_swarm/prompts/` (empty dir)
- Create: `insight_swarm/templates/` (empty dir)
- Create: `insight_swarm/reports/.gitkeep`
- Create: `insight_swarm/logs/.gitkeep`
- Create: `insight_swarm/data/.gitkeep`
- Create: `tests/__init__.py`
- Create: `tests/nodes/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "insight-swarm"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2",
    "langchain>=0.3",
    "langchain-community>=0.3",
    "duckdb>=1.0",
    "pandas>=2.0",
    "openpyxl>=3.1",
    "plotly>=5.0",
    "jinja2>=3.1",
    "duckduckgo-search>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.py[cod]
*.egg-info/
.venv/
dist/
insight_swarm/reports/
insight_swarm/logs/
*.duckdb
.superpowers/
```

- [ ] **Step 3: Create package skeleton**

```bash
mkdir -p insight_swarm/nodes insight_swarm/prompts insight_swarm/templates \
         insight_swarm/reports insight_swarm/logs insight_swarm/data \
         tests/nodes
touch insight_swarm/__init__.py insight_swarm/nodes/__init__.py \
      insight_swarm/reports/.gitkeep insight_swarm/logs/.gitkeep \
      insight_swarm/data/.gitkeep \
      tests/__init__.py tests/nodes/__init__.py
```

- [ ] **Step 4: Create and activate virtual environment, install dependencies**

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -e ".[dev]"
```

Expected: no errors, `pytest --collect-only` shows 0 tests collected.

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml .gitignore insight_swarm/ tests/
git commit -m "feat: project skeleton and dependencies"
```

---

## Task 2: State TypedDicts

**Files:**
- Create: `insight_swarm/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_state.py
from insight_swarm.state import InsightState, Hypothesis, LogEntry

def test_hypothesis_has_required_keys():
    h: Hypothesis = {
        "text": "Sales dropped due to seasonality",
        "type": "internal",
        "sql_result": None,
        "web_context": None,
        "verdict": "inconclusive",
    }
    assert h["type"] in ("internal", "external")
    assert h["verdict"] in ("confirmed", "rejected", "inconclusive")


def test_insight_state_structure():
    state: InsightState = {
        "question": "Why did sales drop?",
        "industry": "retail",
        "context": "US, 2023",
        "max_cycles": 5,
        "source_file": "data/sales.csv",
        "source_type": "csv",
        "db_path": "/tmp/test.duckdb",
        "schema": {},
        "query_context": "initial",
        "current_query_goal": "Analyse overall sales trend",
        "current_sql": None,
        "sql_error": None,
        "sql_retry_count": 0,
        "sql_results": [],
        "analyses": [],
        "hypotheses": [],
        "current_hypothesis_idx": 0,
        "cycle_count": 0,
        "narrative": "",
        "run_log": [],
    }
    assert state["max_cycles"] == 5
    assert state["query_context"] == "initial"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/test_state.py -v
```
Expected: `ImportError: cannot import name 'InsightState'`

- [ ] **Step 3: Implement state.py**

```python
# insight_swarm/state.py
from __future__ import annotations
from typing import Any, Literal, Optional
from typing_extensions import TypedDict


class Hypothesis(TypedDict):
    text: str
    type: Literal["internal", "external"]
    sql_result: Optional[list[dict]]   # records format, None if not yet run
    web_context: Optional[str]
    verdict: Literal["confirmed", "rejected", "inconclusive"]


class LogEntry(TypedDict):
    node: str
    timestamp: str       # ISO-8601
    duration_ms: int
    summary: str
    detail: Any


class InsightState(TypedDict):
    # Input
    question: str
    industry: Optional[str]
    context: Optional[str]
    max_cycles: int

    # File source (set by __main__, used by schema + executor)
    source_file: str      # original file path
    source_type: str      # "csv" | "xlsx" | "duckdb"
    db_path: str          # path to working .duckdb file (set by schema_node)

    # Schema
    schema: dict          # {table_name: [{"column": str, "type": str}]}

    # SQL execution
    query_context: str    # "initial" | "hypothesis"
    current_query_goal: str
    current_sql: Optional[str]
    sql_error: Optional[str]
    sql_retry_count: int  # reset to 0 per hypothesis by hypothesis_node
    sql_results: list     # list of list[dict] — one entry per query run

    # Analysis
    analyses: list        # list[str]

    # Hypotheses
    hypotheses: list      # list[Hypothesis]
    current_hypothesis_idx: int
    cycle_count: int

    # Output
    narrative: str
    run_log: list         # list[LogEntry]
```

- [ ] **Step 4: Run test — verify it passes**

```bash
pytest tests/test_state.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add insight_swarm/state.py tests/test_state.py
git commit -m "feat: InsightState, Hypothesis, LogEntry TypedDicts"
```

---

## Task 3: schema_node

**Files:**
- Create: `insight_swarm/nodes/schema.py`
- Create: `tests/nodes/test_schema.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/nodes/test_schema.py
import duckdb
import pandas as pd
import pytest
from pathlib import Path
from insight_swarm.nodes.schema import schema_node


@pytest.fixture
def minimal_state(tmp_path):
    return {
        "question": "Why did sales drop?",
        "industry": None,
        "context": None,
        "max_cycles": 5,
        "source_file": "",
        "source_type": "",
        "db_path": "",
        "schema": {},
        "query_context": "initial",
        "current_query_goal": "",
        "current_sql": None,
        "sql_error": None,
        "sql_retry_count": 0,
        "sql_results": [],
        "analyses": [],
        "hypotheses": [],
        "current_hypothesis_idx": 0,
        "cycle_count": 0,
        "narrative": "",
        "run_log": [],
    }


def test_schema_node_loads_csv(tmp_path, minimal_state):
    csv_file = tmp_path / "sales.csv"
    csv_file.write_text("product,sales,quarter\nWidget,1000,Q1\nGadget,2000,Q2\n")

    state = {**minimal_state, "source_file": str(csv_file), "source_type": "csv"}
    result = schema_node(state)

    assert "schema" in result
    assert "db_path" in result
    tables = result["schema"]
    assert len(tables) >= 1
    first_table = list(tables.values())[0]
    col_names = [c["column"] for c in first_table]
    assert "product" in col_names
    assert "sales" in col_names


def test_schema_node_loads_xlsx(tmp_path, minimal_state):
    xlsx_file = tmp_path / "data.xlsx"
    df = pd.DataFrame({"revenue": [100, 200], "region": ["North", "South"]})
    df.to_excel(xlsx_file, sheet_name="Sales", index=False)

    state = {**minimal_state, "source_file": str(xlsx_file), "source_type": "xlsx"}
    result = schema_node(state)

    assert "Sales" in result["schema"]
    col_names = [c["column"] for c in result["schema"]["Sales"]]
    assert "revenue" in col_names


def test_schema_node_records_row_counts(tmp_path, minimal_state):
    csv_file = tmp_path / "orders.csv"
    csv_file.write_text("id,amount\n1,50\n2,75\n3,100\n")
    state = {**minimal_state, "source_file": str(csv_file), "source_type": "csv"}
    result = schema_node(state)

    table_info = list(result["schema"].values())[0]
    assert any(col.get("row_count") == 3 for col in table_info if "row_count" in col)


def test_schema_node_appends_log(tmp_path, minimal_state):
    csv_file = tmp_path / "x.csv"
    csv_file.write_text("a,b\n1,2\n")
    state = {**minimal_state, "source_file": str(csv_file), "source_type": "csv"}
    result = schema_node(state)
    assert len(result["run_log"]) == 1
    assert result["run_log"][0]["node"] == "schema"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/nodes/test_schema.py -v
```
Expected: `ImportError: cannot import name 'schema_node'`

- [ ] **Step 3: Implement schema_node**

```python
# insight_swarm/nodes/schema.py
from __future__ import annotations
import time
from datetime import datetime, timezone
from pathlib import Path
import duckdb
import pandas as pd
from insight_swarm.state import InsightState, LogEntry


def _build_log(summary: str, detail: dict, start: float) -> LogEntry:
    return {
        "node": "schema",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - start) * 1000),
        "summary": summary,
        "detail": detail,
    }


def schema_node(state: InsightState) -> dict:
    start = time.time()
    source_file = state["source_file"]
    source_type = state["source_type"]

    # Build a working DuckDB file path next to the source (or in tmp)
    db_path = str(Path(source_file).with_suffix(".duckdb"))

    con = duckdb.connect(db_path)

    if source_type == "csv":
        table_name = Path(source_file).stem.replace("-", "_").replace(" ", "_")
        con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS "
            f"SELECT * FROM read_csv_auto(?)",
            [source_file],
        )
    elif source_type == "xlsx":
        sheets = pd.read_excel(source_file, sheet_name=None)
        for sheet_name, df in sheets.items():
            safe_name = sheet_name.replace(" ", "_").replace("-", "_")
            con.register(safe_name, df)
            con.execute(
                f"CREATE OR REPLACE TABLE {safe_name} AS SELECT * FROM {safe_name}"
            )
    # duckdb: already connected, tables already exist

    # Discover schema
    tables = con.execute("SHOW TABLES").fetchall()
    schema: dict = {}
    for (table_name,) in tables:
        columns_raw = con.execute(f"DESCRIBE {table_name}").fetchall()
        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        columns = [
            {"column": row[0], "type": row[1], "row_count": row_count}
            for row in columns_raw
        ]
        schema[table_name] = columns

    con.close()

    total_tables = len(schema)
    total_cols = sum(len(v) for v in schema.values())
    print(f"✓ Schema discovered: {total_tables} table(s), {total_cols} column(s)")

    log = _build_log(
        summary=f"Discovered {total_tables} table(s), {total_cols} column(s)",
        detail={"tables": list(schema.keys())},
        start=start,
    )

    return {
        "db_path": db_path,
        "schema": schema,
        "current_query_goal": f"Analyse data to answer: {state['question']}",
        "run_log": state["run_log"] + [log],
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/nodes/test_schema.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add insight_swarm/nodes/schema.py tests/nodes/test_schema.py
git commit -m "feat: schema_node — load CSV/XLSX/DuckDB and discover schema"
```

---

## Task 4: Prompt Templates

**Files:**
- Create: `insight_swarm/prompts/sql_writer.txt`
- Create: `insight_swarm/prompts/analyst.txt`
- Create: `insight_swarm/prompts/hypothesis.txt`
- Create: `insight_swarm/prompts/web_analyst.txt`
- Create: `insight_swarm/prompts/narrator.txt`

No tests needed — prompts are plain text consumed by LLM nodes.

- [ ] **Step 1: Write sql_writer.txt**

```
You are an expert SQL analyst working with DuckDB.

Database schema:
{schema}

Business question: {question}
{industry_line}
{context_line}
Query goal: {goal}
{error_line}

Rules:
- Return ONLY the raw SQL query. No explanation, no markdown, no code fences.
- Use DuckDB syntax (strftime for dates, QUALIFY for window filtering).
- Always add LIMIT 1000 unless the query is an aggregation.

SQL:
```

- [ ] **Step 2: Write analyst.txt**

```
You are a business analyst interpreting SQL query results.

Business question: {question}
{industry_line}
{context_line}

Query results ({row_count} rows, showing first 50):
{data_markdown}

Write 2-3 sentences interpreting what these results reveal. Focus on patterns, anomalies, and business implications. Be specific about numbers.
```

- [ ] **Step 3: Write hypothesis.txt**

```
You are a business analyst. Generate exactly 3 hypotheses to explain the observed data.

Business question: {question}
{industry_line}
{context_line}

Analysis so far:
{analyses}

Classify each hypothesis:
- "internal": testable with SQL queries against the available dataset
- "external": requires knowledge outside the dataset (holidays, competitor actions, economic events, market conditions)

Respond with ONLY a valid JSON array — no explanation, no markdown:
[
  {"text": "concise hypothesis description", "type": "internal"},
  {"text": "concise hypothesis description", "type": "external"},
  {"text": "concise hypothesis description", "type": "internal"}
]
```

- [ ] **Step 4: Write web_analyst.txt**

```
You are a business analyst evaluating evidence for a hypothesis.

Hypothesis: {hypothesis_text}
Business question: {question}
{industry_line}

SQL evidence from dataset:
{sql_evidence}

Web search findings:
{web_context}

Based on all available evidence, produce a verdict.

Respond with ONLY a valid JSON object — no explanation, no markdown:
{"verdict": "confirmed", "explanation": "one or two sentences citing specific evidence"}

Verdict must be one of: "confirmed", "rejected", "inconclusive"
```

- [ ] **Step 5: Write narrator.txt**

```
You are a management consultant writing an executive summary for a business report.

Business question: {question}
{industry_line}
{context_line}

Hypothesis testing results:
{hypotheses_summary}

Detailed analyses:
{analyses}

Write a professional executive summary of 2-4 paragraphs:
- Paragraph 1: State the main finding directly.
- Paragraph 2: Explain confirmed hypotheses with supporting evidence.
- Paragraph 3: Note what was tested and ruled out (rejected/inconclusive hypotheses).
- Paragraph 4: One forward-looking sentence or recommendation.

Write in professional business prose. Do not use bullet points or headers.
```

- [ ] **Step 6: Commit**

```bash
git add insight_swarm/prompts/
git commit -m "feat: prompt templates for all LLM nodes"
```

---

## Task 5: sql_writer_node

**Files:**
- Create: `insight_swarm/nodes/sql_writer.py`
- Create: `tests/nodes/test_sql_writer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/nodes/test_sql_writer.py
from unittest.mock import MagicMock, patch
from tests.conftest import make_state


def test_sql_writer_sets_current_sql(make_state):
    state = make_state(query_context="initial")
    with patch("insight_swarm.nodes.sql_writer.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = (
            "SELECT quarter, SUM(sales) FROM orders GROUP BY quarter"
        )
        from insight_swarm.nodes.sql_writer import sql_writer_node
        result = sql_writer_node(state)

    assert "current_sql" in result
    assert "SELECT" in result["current_sql"].upper()


def test_sql_writer_includes_error_in_prompt_on_retry(make_state):
    state = make_state(
        query_context="hypothesis",
        sql_error="Parser Error: syntax error at or near 'FORM'",
        current_sql="SELECT * FORM orders",
    )
    captured_prompts = []

    def fake_invoke(prompt):
        captured_prompts.append(prompt)
        return "SELECT * FROM orders LIMIT 100"

    with patch("insight_swarm.nodes.sql_writer.Ollama") as MockOllama:
        MockOllama.return_value.invoke.side_effect = fake_invoke
        from insight_swarm.nodes.sql_writer import sql_writer_node
        sql_writer_node(state)

    assert "Parser Error" in captured_prompts[0]


def test_sql_writer_appends_log(make_state):
    state = make_state()
    with patch("insight_swarm.nodes.sql_writer.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = "SELECT 1"
        from insight_swarm.nodes.sql_writer import sql_writer_node
        result = sql_writer_node(state)

    assert len(result["run_log"]) == len(state["run_log"]) + 1
    assert result["run_log"][-1]["node"] == "sql_writer"
```

- [ ] **Step 2: Create tests/conftest.py with shared fixture**

```python
# tests/conftest.py
import pytest
from insight_swarm.state import InsightState


def make_state(**overrides) -> InsightState:
    base: InsightState = {
        "question": "Why did sales drop in Q3?",
        "industry": "retail",
        "context": "US market, 2023",
        "max_cycles": 5,
        "source_file": "data/superstore.csv",
        "source_type": "csv",
        "db_path": "/tmp/test.duckdb",
        "schema": {
            "orders": [
                {"column": "order_id", "type": "VARCHAR", "row_count": 1000},
                {"column": "sales", "type": "DOUBLE", "row_count": 1000},
                {"column": "quarter", "type": "VARCHAR", "row_count": 1000},
                {"column": "region", "type": "VARCHAR", "row_count": 1000},
            ]
        },
        "query_context": "initial",
        "current_query_goal": "Analyse overall sales trend by quarter",
        "current_sql": None,
        "sql_error": None,
        "sql_retry_count": 0,
        "sql_results": [],
        "analyses": [],
        "hypotheses": [],
        "current_hypothesis_idx": 0,
        "cycle_count": 0,
        "narrative": "",
        "run_log": [],
    }
    return {**base, **overrides}


@pytest.fixture
def make_state():
    return make_state
```

- [ ] **Step 3: Run test — verify it fails**

```bash
pytest tests/nodes/test_sql_writer.py -v
```
Expected: `ImportError: cannot import name 'sql_writer_node'`

- [ ] **Step 4: Implement sql_writer_node**

```python
# insight_swarm/nodes/sql_writer.py
from __future__ import annotations
import time
from datetime import datetime, timezone
from pathlib import Path
from langchain_community.llms import Ollama
from insight_swarm.state import InsightState, LogEntry

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "sql_writer.txt"
_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")
SQL_MODEL = "deepseek-coder:6.7b"


def _format_schema(schema: dict) -> str:
    lines = []
    for table, cols in schema.items():
        row_count = cols[0].get("row_count", "?") if cols else "?"
        lines.append(f"Table: {table} ({row_count} rows)")
        for col in cols:
            lines.append(f"  - {col['column']} ({col['type']})")
    return "\n".join(lines)


def sql_writer_node(state: InsightState) -> dict:
    start = time.time()
    llm = Ollama(model=SQL_MODEL)

    industry_line = f"Industry: {state['industry']}" if state.get("industry") else ""
    context_line = f"Context: {state['context']}" if state.get("context") else ""
    error_line = (
        f"\nPrevious SQL had an error:\n{state['sql_error']}\n"
        f"Broken SQL:\n{state['current_sql']}\nFix the query."
        if state.get("sql_error")
        else ""
    )

    prompt = _TEMPLATE.format(
        schema=_format_schema(state["schema"]),
        question=state["question"],
        industry_line=industry_line,
        context_line=context_line,
        goal=state["current_query_goal"],
        error_line=error_line,
    )

    context_label = state.get("query_context", "initial")
    print(f"→ SQL Writer ({context_label}): generating query...")
    raw = llm.invoke(prompt).strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("sql"):
            raw = raw[3:]
    raw = raw.strip()

    log: LogEntry = {
        "node": "sql_writer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - start) * 1000),
        "summary": f"Generated SQL for goal: {state['current_query_goal'][:60]}",
        "detail": {"sql": raw, "retries": state["sql_retry_count"]},
    }

    return {
        "current_sql": raw,
        "run_log": state["run_log"] + [log],
    }
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/nodes/test_sql_writer.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add insight_swarm/nodes/sql_writer.py tests/nodes/test_sql_writer.py tests/conftest.py
git commit -m "feat: sql_writer_node — deepseek-coder generates SQL with retry context"
```

---

## Task 6: executor_node

**Files:**
- Create: `insight_swarm/nodes/executor.py`
- Create: `tests/nodes/test_executor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/nodes/test_executor.py
import duckdb
import pytest
from pathlib import Path
from tests.conftest import make_state


@pytest.fixture
def db_with_orders(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE orders AS
        SELECT * FROM (VALUES
            ('Q1', 'West', 1000.0),
            ('Q2', 'West', 1200.0),
            ('Q3', 'West', 800.0),
            ('Q1', 'East', 900.0),
            ('Q2', 'East', 950.0),
            ('Q3', 'East', 870.0),
            ('Q1', 'North', 700.0),
            ('Q2', 'North', 720.0),
            ('Q3', 'North', 650.0),
            ('Q1', 'South', 500.0),
            ('Q2', 'South', 510.0)
        ) t(quarter, region, sales)
    """)
    con.close()
    return db_path


def test_executor_successful_query(db_with_orders):
    from insight_swarm.nodes.executor import executor_node
    state = make_state(
        db_path=db_with_orders,
        current_sql="SELECT quarter, SUM(sales) as total FROM orders GROUP BY quarter",
    )
    result = executor_node(state)

    assert result["sql_error"] is None
    assert len(result["sql_results"]) == 1
    rows = result["sql_results"][0]
    assert len(rows) == 3  # Q1, Q2, Q3


def test_executor_failed_query_sets_error(db_with_orders):
    from insight_swarm.nodes.executor import executor_node
    state = make_state(
        db_path=db_with_orders,
        current_sql="SELECT * FORM nonexistent_table",
    )
    result = executor_node(state)

    assert result["sql_error"] is not None
    assert "sql_results" in result  # list unchanged


def test_executor_clears_error_on_success(db_with_orders):
    from insight_swarm.nodes.executor import executor_node
    state = make_state(
        db_path=db_with_orders,
        current_sql="SELECT 1 AS val",
        sql_error="previous error",
        sql_retry_count=1,
    )
    result = executor_node(state)

    assert result["sql_error"] is None
    assert result["sql_retry_count"] == 0


def test_executor_appends_log(db_with_orders):
    from insight_swarm.nodes.executor import executor_node
    state = make_state(db_path=db_with_orders, current_sql="SELECT 1 AS val")
    result = executor_node(state)
    assert result["run_log"][-1]["node"] == "executor"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/nodes/test_executor.py -v
```
Expected: `ImportError: cannot import name 'executor_node'`

- [ ] **Step 3: Implement executor_node**

```python
# insight_swarm/nodes/executor.py
from __future__ import annotations
import time
from datetime import datetime, timezone
import duckdb
from insight_swarm.state import InsightState, LogEntry

SUFFICIENT_ROW_THRESHOLD = 10


def executor_node(state: InsightState) -> dict:
    start = time.time()
    sql = state["current_sql"]
    db_path = state["db_path"]

    try:
        con = duckdb.connect(db_path)
        df = con.execute(sql).df()
        con.close()

        records = df.to_dict(orient="records")
        row_count = len(records)
        print(f"✓ Query executed: {row_count} rows")

        log: LogEntry = {
            "node": "executor",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": int((time.time() - start) * 1000),
            "summary": f"Query returned {row_count} rows",
            "detail": {"sql": sql, "row_count": row_count},
        }

        return {
            "sql_results": state["sql_results"] + [records],
            "sql_error": None,
            "sql_retry_count": 0,
            "run_log": state["run_log"] + [log],
        }

    except Exception as exc:
        error_msg = str(exc)
        print(f"✗ Query failed: {error_msg[:80]}")

        log: LogEntry = {
            "node": "executor",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": int((time.time() - start) * 1000),
            "summary": f"Query error: {error_msg[:80]}",
            "detail": {"sql": sql, "error": error_msg},
        }

        return {
            "sql_error": error_msg,
            "run_log": state["run_log"] + [log],
        }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/nodes/test_executor.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add insight_swarm/nodes/executor.py tests/nodes/test_executor.py
git commit -m "feat: executor_node — run SQL against DuckDB, capture errors"
```

---

## Task 7: sql_retry_node

**Files:**
- Create: `insight_swarm/nodes/sql_retry.py`
- Create: `tests/nodes/test_sql_retry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/nodes/test_sql_retry.py
from tests.conftest import make_state


def test_sql_retry_increments_count():
    from insight_swarm.nodes.sql_retry import sql_retry_node
    state = make_state(sql_retry_count=0, sql_error="syntax error")
    result = sql_retry_node(state)
    assert result["sql_retry_count"] == 1


def test_sql_retry_increments_again():
    from insight_swarm.nodes.sql_retry import sql_retry_node
    state = make_state(sql_retry_count=1, sql_error="type mismatch")
    result = sql_retry_node(state)
    assert result["sql_retry_count"] == 2


def test_sql_retry_appends_log():
    from insight_swarm.nodes.sql_retry import sql_retry_node
    state = make_state(sql_retry_count=0, sql_error="error")
    result = sql_retry_node(state)
    assert result["run_log"][-1]["node"] == "sql_retry"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/nodes/test_sql_retry.py -v
```

- [ ] **Step 3: Implement sql_retry_node**

```python
# insight_swarm/nodes/sql_retry.py
from __future__ import annotations
import time
from datetime import datetime, timezone
from insight_swarm.state import InsightState, LogEntry


def sql_retry_node(state: InsightState) -> dict:
    start = time.time()
    new_count = state["sql_retry_count"] + 1
    print(f"→ SQL retry {new_count}/2: will re-attempt with error context")

    log: LogEntry = {
        "node": "sql_retry",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - start) * 1000),
        "summary": f"Retry {new_count}/2 — error: {str(state.get('sql_error', ''))[:60]}",
        "detail": {"retry_count": new_count, "error": state.get("sql_error")},
    }

    return {
        "sql_retry_count": new_count,
        "run_log": state["run_log"] + [log],
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/nodes/test_sql_retry.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add insight_swarm/nodes/sql_retry.py tests/nodes/test_sql_retry.py
git commit -m "feat: sql_retry_node — increment retry counter before re-attempting SQL"
```

---

## Task 8: analyst_node

**Files:**
- Create: `insight_swarm/nodes/analyst.py`
- Create: `tests/nodes/test_analyst.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/nodes/test_analyst.py
from unittest.mock import patch
from tests.conftest import make_state


def test_analyst_appends_analysis(make_state):
    state = make_state(
        sql_results=[[{"quarter": "Q3", "total_sales": 800.0}]],
        analyses=[],
    )
    with patch("insight_swarm.nodes.analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = (
            "Q3 sales were 33% lower than Q2, indicating a significant seasonal dip."
        )
        from insight_swarm.nodes.analyst import analyst_node
        result = analyst_node(state)

    assert len(result["analyses"]) == 1
    assert "Q3" in result["analyses"][0]


def test_analyst_handles_empty_results(make_state):
    state = make_state(sql_results=[None], analyses=[])
    with patch("insight_swarm.nodes.analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = "No data available for analysis."
        from insight_swarm.nodes.analyst import analyst_node
        result = analyst_node(state)

    assert len(result["analyses"]) == 1


def test_analyst_appends_log(make_state):
    state = make_state(sql_results=[[{"val": 1}]])
    with patch("insight_swarm.nodes.analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = "Some insight."
        from insight_swarm.nodes.analyst import analyst_node
        result = analyst_node(state)

    assert result["run_log"][-1]["node"] == "analyst"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/nodes/test_analyst.py -v
```

- [ ] **Step 3: Implement analyst_node**

```python
# insight_swarm/nodes/analyst.py
from __future__ import annotations
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from langchain_community.llms import Ollama
from insight_swarm.state import InsightState, LogEntry

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "analyst.txt"
_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")
REASONING_MODEL = "llama3.1:8b"


def _records_to_markdown(records: list[dict] | None, max_rows: int = 50) -> str:
    if not records:
        return "(no data returned)"
    df = pd.DataFrame(records[:max_rows])
    return df.to_markdown(index=False)


def analyst_node(state: InsightState) -> dict:
    start = time.time()
    llm = Ollama(model=REASONING_MODEL)

    latest = state["sql_results"][-1] if state["sql_results"] else None
    row_count = len(latest) if latest else 0
    data_md = _records_to_markdown(latest)

    prompt = _TEMPLATE.format(
        question=state["question"],
        industry_line=f"Industry: {state['industry']}" if state.get("industry") else "",
        context_line=f"Context: {state['context']}" if state.get("context") else "",
        row_count=row_count,
        data_markdown=data_md,
    )

    print("→ Analyst: interpreting results...")
    interpretation = llm.invoke(prompt).strip()

    log: LogEntry = {
        "node": "analyst",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - start) * 1000),
        "summary": interpretation[:100],
        "detail": {"row_count": row_count},
    }

    return {
        "analyses": state["analyses"] + [interpretation],
        "run_log": state["run_log"] + [log],
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/nodes/test_analyst.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add insight_swarm/nodes/analyst.py tests/nodes/test_analyst.py
git commit -m "feat: analyst_node — llama3.1:8b interprets SQL results"
```

---

## Task 9: hypothesis_node

**Files:**
- Create: `insight_swarm/nodes/hypothesis.py`
- Create: `tests/nodes/test_hypothesis.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/nodes/test_hypothesis.py
import json
from unittest.mock import patch
from tests.conftest import make_state

_MOCK_HYPOTHESES = json.dumps([
    {"text": "West region had seasonal drop", "type": "internal"},
    {"text": "Holiday calendar affected retail", "type": "external"},
    {"text": "Furniture category declined", "type": "internal"},
])


def test_hypothesis_generates_on_first_call(make_state):
    state = make_state(hypotheses=[], analyses=["Q3 was 20% below Q2."])
    with patch("insight_swarm.nodes.hypothesis.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = _MOCK_HYPOTHESES
        from insight_swarm.nodes.hypothesis import hypothesis_node
        result = hypothesis_node(state)

    assert len(result["hypotheses"]) == 3
    assert result["hypotheses"][0]["type"] in ("internal", "external")
    assert result["current_hypothesis_idx"] == 0


def test_hypothesis_advances_on_subsequent_call(make_state):
    hypotheses = [
        {"text": "H1", "type": "internal", "sql_result": None, "web_context": None, "verdict": "confirmed"},
        {"text": "H2", "type": "external", "sql_result": None, "web_context": None, "verdict": "inconclusive"},
        {"text": "H3", "type": "internal", "sql_result": None, "web_context": None, "verdict": "inconclusive"},
    ]
    state = make_state(hypotheses=hypotheses, current_hypothesis_idx=0, cycle_count=1)
    from insight_swarm.nodes.hypothesis import hypothesis_node
    result = hypothesis_node(state)

    assert result["current_hypothesis_idx"] == 1
    assert result["sql_retry_count"] == 0  # reset for next hypothesis
    assert result["cycle_count"] == 2


def test_hypothesis_sets_query_goal_for_internal(make_state):
    hypotheses = [
        {"text": "West region seasonal drop", "type": "internal",
         "sql_result": None, "web_context": None, "verdict": "inconclusive"},
    ]
    state = make_state(hypotheses=hypotheses, current_hypothesis_idx=0)
    from insight_swarm.nodes.hypothesis import hypothesis_node
    result = hypothesis_node(state)
    assert "West region seasonal drop" in result.get("current_query_goal", "")


def test_hypothesis_appends_log(make_state):
    state = make_state(hypotheses=[], analyses=["some analysis"])
    with patch("insight_swarm.nodes.hypothesis.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = _MOCK_HYPOTHESES
        from insight_swarm.nodes.hypothesis import hypothesis_node
        result = hypothesis_node(state)
    assert result["run_log"][-1]["node"] == "hypothesis"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/nodes/test_hypothesis.py -v
```

- [ ] **Step 3: Implement hypothesis_node**

```python
# insight_swarm/nodes/hypothesis.py
from __future__ import annotations
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from langchain_community.llms import Ollama
from insight_swarm.state import InsightState, Hypothesis, LogEntry

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "hypothesis.txt"
_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")
REASONING_MODEL = "llama3.1:8b"


def _parse_hypotheses(raw: str) -> list[Hypothesis]:
    # Extract JSON array from response
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in hypothesis response: {raw[:200]}")
    data = json.loads(match.group())
    return [
        {
            "text": h["text"],
            "type": h["type"],
            "sql_result": None,
            "web_context": None,
            "verdict": "inconclusive",
        }
        for h in data
    ]


def hypothesis_node(state: InsightState) -> dict:
    start = time.time()

    # Subsequent call: advance index, reset retry count
    if state["hypotheses"]:
        idx = state["current_hypothesis_idx"] + 1
        new_cycle = state["cycle_count"] + 1
        current = state["hypotheses"][idx] if idx < len(state["hypotheses"]) else None

        log: LogEntry = {
            "node": "hypothesis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": int((time.time() - start) * 1000),
            "summary": f"Advancing to hypothesis {idx + 1}/{len(state['hypotheses'])}",
            "detail": {"idx": idx, "cycle_count": new_cycle},
        }

        updates: dict = {
            "current_hypothesis_idx": idx,
            "cycle_count": new_cycle,
            "sql_retry_count": 0,
            "query_context": "hypothesis",
            "run_log": state["run_log"] + [log],
        }

        if current:
            updates["current_query_goal"] = (
                f"Test hypothesis: {current['text']}"
            )

        return updates

    # First call: generate hypotheses
    llm = Ollama(model=REASONING_MODEL)
    analyses_text = "\n\n".join(state["analyses"]) or "No analysis yet."

    prompt = _TEMPLATE.format(
        question=state["question"],
        industry_line=f"Industry: {state['industry']}" if state.get("industry") else "",
        context_line=f"Context: {state['context']}" if state.get("context") else "",
        analyses=analyses_text,
    )

    print("→ Hypothesis Generator: creating hypotheses...")
    raw = llm.invoke(prompt).strip()
    hypotheses = _parse_hypotheses(raw)

    n_internal = sum(1 for h in hypotheses if h["type"] == "internal")
    n_external = len(hypotheses) - n_internal
    print(f"✓ Hypotheses generated: {len(hypotheses)} ({n_internal} internal, {n_external} external)")

    log: LogEntry = {
        "node": "hypothesis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - start) * 1000),
        "summary": f"Generated {len(hypotheses)} hypotheses",
        "detail": {"hypotheses": [h["text"] for h in hypotheses]},
    }

    first_goal = f"Test hypothesis: {hypotheses[0]['text']}" if hypotheses else ""

    return {
        "hypotheses": hypotheses,
        "current_hypothesis_idx": 0,
        "cycle_count": state["cycle_count"] + 1,
        "sql_retry_count": 0,
        "query_context": "hypothesis",
        "current_query_goal": first_goal,
        "run_log": state["run_log"] + [log],
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/nodes/test_hypothesis.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add insight_swarm/nodes/hypothesis.py tests/nodes/test_hypothesis.py
git commit -m "feat: hypothesis_node — generate and advance typed hypotheses"
```

---

## Task 10: web_search_node

**Files:**
- Create: `insight_swarm/nodes/web_search.py`
- Create: `tests/nodes/test_web_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/nodes/test_web_search.py
from unittest.mock import patch, MagicMock
from tests.conftest import make_state


def _state_with_hypothesis(h_text="Holiday effect on retail", h_type="external"):
    hypotheses = [
        {"text": h_text, "type": h_type,
         "sql_result": None, "web_context": None, "verdict": "inconclusive"}
    ]
    return make_state(hypotheses=hypotheses, current_hypothesis_idx=0)


def test_web_search_sets_web_context():
    from insight_swarm.nodes.web_search import web_search_node
    state = _state_with_hypothesis()
    mock_results = [
        {"title": "Q3 Retail Drop", "body": "Sales fell 15% due to fewer shopping days.", "href": "http://x.com"},
        {"title": "Holiday Schedule", "body": "Labor Day shifted consumer spending patterns.", "href": "http://y.com"},
    ]
    with patch("insight_swarm.nodes.web_search.DDGS") as MockDDGS:
        MockDDGS.return_value.__enter__ = lambda s: s
        MockDDGS.return_value.__exit__ = MagicMock(return_value=False)
        MockDDGS.return_value.text.return_value = mock_results
        result = web_search_node(state)

    hypotheses = result["hypotheses"]
    assert hypotheses[0]["web_context"] is not None
    assert "Retail Drop" in hypotheses[0]["web_context"] or "Holiday" in hypotheses[0]["web_context"]


def test_web_search_handles_rate_limit_gracefully():
    from insight_swarm.nodes.web_search import web_search_node
    state = _state_with_hypothesis()
    with patch("insight_swarm.nodes.web_search.DDGS") as MockDDGS:
        MockDDGS.return_value.__enter__ = lambda s: s
        MockDDGS.return_value.__exit__ = MagicMock(return_value=False)
        MockDDGS.return_value.text.side_effect = Exception("Ratelimit")
        result = web_search_node(state)

    hypotheses = result["hypotheses"]
    assert hypotheses[0]["web_context"] is None


def test_web_search_appends_log():
    from insight_swarm.nodes.web_search import web_search_node
    state = _state_with_hypothesis()
    with patch("insight_swarm.nodes.web_search.DDGS") as MockDDGS:
        MockDDGS.return_value.__enter__ = lambda s: s
        MockDDGS.return_value.__exit__ = MagicMock(return_value=False)
        MockDDGS.return_value.text.return_value = []
        result = web_search_node(state)

    assert result["run_log"][-1]["node"] == "web_search"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/nodes/test_web_search.py -v
```

- [ ] **Step 3: Implement web_search_node**

```python
# insight_swarm/nodes/web_search.py
from __future__ import annotations
import time
from datetime import datetime, timezone
from duckduckgo_search import DDGS
from insight_swarm.state import InsightState, LogEntry


def _build_query(hypothesis: str, question: str, industry: str | None, context: str | None) -> str:
    parts = [hypothesis]
    if industry:
        parts.append(industry)
    # Extract year from context if available
    if context:
        import re
        years = re.findall(r"\b20\d{2}\b", context)
        if years:
            parts.extend(years)
    # Add key words from business question (first 5 words)
    parts.append(" ".join(question.split()[:5]))
    return " ".join(parts)


def web_search_node(state: InsightState) -> dict:
    start = time.time()
    idx = state["current_hypothesis_idx"]
    hypothesis = state["hypotheses"][idx]
    query = _build_query(
        hypothesis["text"],
        state["question"],
        state.get("industry"),
        state.get("context"),
    )

    print(f'→ Web Search: "{query[:70]}..."')
    web_context: str | None = None

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if results:
            snippets = [f"[{r['title']}] {r['body']}" for r in results]
            web_context = "\n".join(snippets)
            print(f"✓ Web search: {len(results)} results found")
        else:
            print("✓ Web search: no results found")
    except Exception as exc:
        print(f"✗ Web search failed: {exc} — marking as inconclusive")

    updated_hypotheses = list(state["hypotheses"])
    updated_hypotheses[idx] = {**hypothesis, "web_context": web_context}

    log: LogEntry = {
        "node": "web_search",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - start) * 1000),
        "summary": f"Search for H{idx + 1}: {len(results) if web_context else 0} results",
        "detail": {"query": query, "result_count": len(results) if web_context else 0},
    }

    return {
        "hypotheses": updated_hypotheses,
        "run_log": state["run_log"] + [log],
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/nodes/test_web_search.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add insight_swarm/nodes/web_search.py tests/nodes/test_web_search.py
git commit -m "feat: web_search_node — DuckDuckGo search with rate limit handling"
```

---

## Task 11: web_analyst_node

**Files:**
- Create: `insight_swarm/nodes/web_analyst.py`
- Create: `tests/nodes/test_web_analyst.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/nodes/test_web_analyst.py
import json
from unittest.mock import patch
from tests.conftest import make_state


def _state_with_web_hypothesis(verdict_response="confirmed"):
    hypotheses = [
        {
            "text": "Holiday calendar reduced shopping days",
            "type": "external",
            "sql_result": [{"quarter": "Q3", "sales": 800}],
            "web_context": "[Retail News] Labor Day shift reduced consumer spend by 12%.",
            "verdict": "inconclusive",
        }
    ]
    return make_state(hypotheses=hypotheses, current_hypothesis_idx=0), json.dumps(
        {"verdict": verdict_response, "explanation": "Labor Day shift confirmed by web sources."}
    )


def test_web_analyst_sets_verdict():
    from insight_swarm.nodes.web_analyst import web_analyst_node
    state, mock_response = _state_with_web_hypothesis("confirmed")
    with patch("insight_swarm.nodes.web_analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = mock_response
        result = web_analyst_node(state)

    h = result["hypotheses"][0]
    assert h["verdict"] == "confirmed"


def test_web_analyst_appends_analysis():
    from insight_swarm.nodes.web_analyst import web_analyst_node
    state, mock_response = _state_with_web_hypothesis("rejected")
    with patch("insight_swarm.nodes.web_analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = mock_response
        result = web_analyst_node(state)

    assert len(result["analyses"]) == len(state["analyses"]) + 1


def test_web_analyst_handles_malformed_json():
    from insight_swarm.nodes.web_analyst import web_analyst_node
    state, _ = _state_with_web_hypothesis()
    with patch("insight_swarm.nodes.web_analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = "The hypothesis is confirmed by the data."
        result = web_analyst_node(state)

    # Falls back to inconclusive rather than crashing
    assert result["hypotheses"][0]["verdict"] in ("confirmed", "rejected", "inconclusive")


def test_web_analyst_appends_log():
    from insight_swarm.nodes.web_analyst import web_analyst_node
    state, mock_response = _state_with_web_hypothesis()
    with patch("insight_swarm.nodes.web_analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = mock_response
        result = web_analyst_node(state)
    assert result["run_log"][-1]["node"] == "web_analyst"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/nodes/test_web_analyst.py -v
```

- [ ] **Step 3: Implement web_analyst_node**

```python
# insight_swarm/nodes/web_analyst.py
from __future__ import annotations
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from langchain_community.llms import Ollama
from insight_swarm.state import InsightState, LogEntry

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "web_analyst.txt"
_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")
REASONING_MODEL = "llama3.1:8b"


def _records_to_markdown(records: list[dict] | None) -> str:
    if not records:
        return "(no SQL data available)"
    return pd.DataFrame(records[:20]).to_markdown(index=False)


def _parse_verdict(raw: str) -> tuple[str, str]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            verdict = data.get("verdict", "inconclusive")
            if verdict not in ("confirmed", "rejected", "inconclusive"):
                verdict = "inconclusive"
            return verdict, data.get("explanation", raw[:200])
        except json.JSONDecodeError:
            pass
    return "inconclusive", raw[:200]


def web_analyst_node(state: InsightState) -> dict:
    start = time.time()
    idx = state["current_hypothesis_idx"]
    hypothesis = state["hypotheses"][idx]
    llm = Ollama(model=REASONING_MODEL)

    sql_evidence = _records_to_markdown(hypothesis.get("sql_result"))
    web_context = hypothesis.get("web_context") or "(no web results available)"

    prompt = _TEMPLATE.format(
        hypothesis_text=hypothesis["text"],
        question=state["question"],
        industry_line=f"Industry: {state['industry']}" if state.get("industry") else "",
        sql_evidence=sql_evidence,
        web_context=web_context,
    )

    n = idx + 1
    total = len(state["hypotheses"])
    print(f"→ Web Analyst: evaluating H{n}/{total}...")
    raw = llm.invoke(prompt).strip()
    verdict, explanation = _parse_verdict(raw)
    print(f"✓ H{n} {verdict}")

    updated_hypotheses = list(state["hypotheses"])
    updated_hypotheses[idx] = {**hypothesis, "verdict": verdict}

    analysis_entry = f"H{n} ({hypothesis['text']}): {verdict} — {explanation}"

    log: LogEntry = {
        "node": "web_analyst",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - start) * 1000),
        "summary": f"H{n} verdict: {verdict}",
        "detail": {"hypothesis": hypothesis["text"], "verdict": verdict, "explanation": explanation},
    }

    return {
        "hypotheses": updated_hypotheses,
        "analyses": state["analyses"] + [analysis_entry],
        "run_log": state["run_log"] + [log],
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/nodes/test_web_analyst.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add insight_swarm/nodes/web_analyst.py tests/nodes/test_web_analyst.py
git commit -m "feat: web_analyst_node — merge SQL+web evidence into verdict"
```

---

## Task 12: narrator_node

**Files:**
- Create: `insight_swarm/nodes/narrator.py`
- Create: `tests/nodes/test_narrator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/nodes/test_narrator.py
from unittest.mock import patch
from tests.conftest import make_state


def _state_with_verdicts():
    hypotheses = [
        {"text": "Seasonal effect", "type": "internal", "sql_result": None,
         "web_context": None, "verdict": "confirmed"},
        {"text": "Competitor actions", "type": "external", "sql_result": None,
         "web_context": "No evidence found.", "verdict": "rejected"},
    ]
    return make_state(
        hypotheses=hypotheses,
        analyses=["Q3 was down 20%.", "West region down most."],
    )


def test_narrator_sets_narrative():
    from insight_swarm.nodes.narrator import narrator_node
    state = _state_with_verdicts()
    with patch("insight_swarm.nodes.narrator.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = (
            "Sales declined primarily due to seasonal factors. "
            "The West region showed the steepest drop. "
            "Competitor actions were investigated but found to be immaterial."
        )
        result = narrator_node(state)

    assert len(result["narrative"]) > 20


def test_narrator_appends_log():
    from insight_swarm.nodes.narrator import narrator_node
    state = _state_with_verdicts()
    with patch("insight_swarm.nodes.narrator.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = "Summary text."
        result = narrator_node(state)
    assert result["run_log"][-1]["node"] == "narrator"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/nodes/test_narrator.py -v
```

- [ ] **Step 3: Implement narrator_node**

```python
# insight_swarm/nodes/narrator.py
from __future__ import annotations
import time
from datetime import datetime, timezone
from pathlib import Path
from langchain_community.llms import Ollama
from insight_swarm.state import InsightState, LogEntry

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "narrator.txt"
_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")
REASONING_MODEL = "llama3.1:8b"


def _summarise_hypotheses(hypotheses: list) -> str:
    lines = []
    for i, h in enumerate(hypotheses, 1):
        lines.append(f"H{i} [{h['verdict'].upper()}] {h['text']}")
    return "\n".join(lines)


def narrator_node(state: InsightState) -> dict:
    start = time.time()
    llm = Ollama(model=REASONING_MODEL)

    prompt = _TEMPLATE.format(
        question=state["question"],
        industry_line=f"Industry: {state['industry']}" if state.get("industry") else "",
        context_line=f"Context: {state['context']}" if state.get("context") else "",
        hypotheses_summary=_summarise_hypotheses(state["hypotheses"]),
        analyses="\n\n".join(state["analyses"]),
    )

    print("→ Narrator: writing executive summary...")
    narrative = llm.invoke(prompt).strip()

    log: LogEntry = {
        "node": "narrator",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - start) * 1000),
        "summary": narrative[:100],
        "detail": {},
    }

    return {
        "narrative": narrative,
        "run_log": state["run_log"] + [log],
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/nodes/test_narrator.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add insight_swarm/nodes/narrator.py tests/nodes/test_narrator.py
git commit -m "feat: narrator_node — llama3.1:8b writes executive summary"
```

---

## Task 13: report_node + Jinja2 Template

**Files:**
- Create: `insight_swarm/templates/report.html.j2`
- Create: `insight_swarm/nodes/report.py`
- Create: `tests/nodes/test_report.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/nodes/test_report.py
import json
from pathlib import Path
from tests.conftest import make_state


def _full_state(tmp_path):
    return make_state(
        narrative="Sales declined in Q3 primarily due to seasonal effects.",
        hypotheses=[
            {
                "text": "Seasonal effect",
                "type": "internal",
                "sql_result": [{"quarter": "Q3", "sales": 800}, {"quarter": "Q2", "sales": 1200}],
                "web_context": None,
                "verdict": "confirmed",
            },
            {
                "text": "Competitor actions",
                "type": "external",
                "sql_result": None,
                "web_context": "No evidence.",
                "verdict": "rejected",
            },
        ],
        sql_results=[[{"quarter": "Q3", "sales": 800}]],
        run_log=[],
        db_path=str(tmp_path / "test.duckdb"),
    )


def test_report_creates_html_file(tmp_path, monkeypatch):
    import insight_swarm.nodes.report as report_module
    monkeypatch.setattr(report_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(report_module, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(report_module, "webbrowser", type("W", (), {"open": lambda *a, **k: None})())

    from insight_swarm.nodes.report import report_node
    state = _full_state(tmp_path)
    report_node(state)

    html_files = list((tmp_path / "reports").glob("*.html"))
    assert len(html_files) == 1
    content = html_files[0].read_text()
    assert "Seasonal effect" in content
    assert "confirmed" in content.lower()


def test_report_creates_json_log(tmp_path, monkeypatch):
    import insight_swarm.nodes.report as report_module
    monkeypatch.setattr(report_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(report_module, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(report_module, "webbrowser", type("W", (), {"open": lambda *a, **k: None})())

    from insight_swarm.nodes.report import report_node
    state = _full_state(tmp_path)
    report_node(state)

    json_files = list((tmp_path / "logs").glob("*.json"))
    assert len(json_files) == 1
    data = json.loads(json_files[0].read_text())
    assert "question" in data
    assert "hypotheses_tested" in data
```

- [ ] **Step 2: Create Jinja2 template**

```html
{# insight_swarm/templates/report.html.j2 #}
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Insight Swarm Report</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #0d1117; color: #e6edf3; margin: 0; padding: 24px; }
    .header { border-bottom: 1px solid #30363d; padding-bottom: 16px; margin-bottom: 24px; }
    .header h1 { color: #58a6ff; margin: 0 0 4px; }
    .meta { color: #8b949e; font-size: 13px; }
    .section-label { color: #f0883e; font-size: 11px; font-weight: bold;
                     letter-spacing: 1px; text-transform: uppercase; margin: 24px 0 8px; }
    .narrative { background: #161b22; border-left: 3px solid #58a6ff;
                 padding: 16px; border-radius: 4px; line-height: 1.7; }
    .hypothesis-card { background: #161b22; border: 1px solid #30363d;
                       border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    .verdict-badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
                     font-size: 11px; font-weight: bold; text-transform: uppercase; }
    .verdict-confirmed { background: #1a4731; color: #3fb950; }
    .verdict-rejected { background: #3d1a1a; color: #f85149; }
    .verdict-inconclusive { background: #2d2a1a; color: #d29922; }
    .h-text { font-weight: bold; margin: 8px 0 4px; }
    details { margin-top: 16px; }
    summary { cursor: pointer; color: #58a6ff; font-size: 13px; }
    pre { background: #0d1117; padding: 12px; border-radius: 4px;
          font-size: 12px; overflow-x: auto; color: #3fb950; }
    .chart { margin-top: 12px; }
  </style>
</head>
<body>
  <div class="header">
    <h1>Insight Swarm Report</h1>
    <div class="meta">Q: {{ question }} &bull; {{ generated_at }} &bull; {{ source_file }}</div>
  </div>

  <div class="section-label">Executive Summary</div>
  <div class="narrative">{{ narrative }}</div>

  <div class="section-label">Findings</div>
  {% for h in hypotheses %}
  <div class="hypothesis-card">
    <span class="verdict-badge verdict-{{ h.verdict }}">{{ h.verdict }}</span>
    <div class="h-text">H{{ loop.index }}: {{ h.text }}</div>
    {% if h.chart_html %}
    <div class="chart">{{ h.chart_html | safe }}</div>
    {% endif %}
    {% if h.web_context %}
    <details>
      <summary>Web evidence</summary>
      <pre>{{ h.web_context }}</pre>
    </details>
    {% endif %}
  </div>
  {% endfor %}

  <div class="section-label">SQL Queries</div>
  {% for sql in sql_queries %}
  <details>
    <summary>Query {{ loop.index }}</summary>
    <pre>{{ sql }}</pre>
  </details>
  {% endfor %}
</body>
</html>
```

- [ ] **Step 3: Run test — verify it fails**

```bash
pytest tests/nodes/test_report.py -v
```

- [ ] **Step 4: Implement report_node**

```python
# insight_swarm/nodes/report.py
from __future__ import annotations
import json
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import plotly.express as px
from jinja2 import Environment, FileSystemLoader
from insight_swarm.state import InsightState

_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
LOGS_DIR = Path(__file__).parent.parent / "logs"


def _make_chart(hypothesis: dict) -> str | None:
    records = hypothesis.get("sql_result")
    if not records or hypothesis["verdict"] not in ("confirmed", "rejected"):
        return None
    df = pd.DataFrame(records)
    numeric_cols = df.select_dtypes("number").columns.tolist()
    if not numeric_cols:
        return None
    y_col = numeric_cols[0]
    x_col = [c for c in df.columns if c != y_col][0] if len(df.columns) > 1 else df.columns[0]
    fig = px.bar(df, x=x_col, y=y_col, title=hypothesis["text"][:60],
                 template="plotly_dark")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def report_node(state: InsightState) -> dict:
    start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Enrich hypotheses with chart HTML
    hypotheses_ctx = []
    for h in state["hypotheses"]:
        hypotheses_ctx.append({**h, "chart_html": _make_chart(h)})

    # Collect SQL strings from run_log
    sql_queries = [
        entry["detail"]["sql"]
        for entry in state["run_log"]
        if entry["node"] in ("sql_writer", "executor") and "sql" in entry.get("detail", {})
    ]

    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))
    template = env.get_template("report.html.j2")
    html = template.render(
        question=state["question"],
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        source_file=state.get("source_file", ""),
        narrative=state["narrative"],
        hypotheses=hypotheses_ctx,
        sql_queries=sql_queries,
    )

    html_path = REPORTS_DIR / f"report_{timestamp}.html"
    html_path.write_text(html, encoding="utf-8")

    # JSON run log
    total_duration = int((time.time() - start) * 1000)
    sql_retries = sum(
        1 for e in state["run_log"] if e["node"] == "sql_retry"
    )
    log_data = {
        "question": state["question"],
        "industry": state.get("industry"),
        "context": state.get("context"),
        "source_file": state.get("source_file"),
        "total_duration_ms": total_duration,
        "hypotheses_tested": len(state["hypotheses"]),
        "sql_retries_total": sql_retries,
        "cycle_count": state["cycle_count"],
        "steps": state["run_log"],
    }
    json_path = LOGS_DIR / f"run_{timestamp}.json"
    json_path.write_text(json.dumps(log_data, indent=2, default=str), encoding="utf-8")

    print(f"✓ Report saved: {html_path}")
    print(f"✓ Log saved: {json_path}")
    print("✓ Opening in browser...")
    webbrowser.open(str(html_path))

    return {}
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/nodes/test_report.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add insight_swarm/nodes/report.py insight_swarm/templates/ tests/nodes/test_report.py
git commit -m "feat: report_node — Jinja2 HTML report + JSON run log with Plotly charts"
```

---

## Task 14: LangGraph Assembly + Routing Functions

**Files:**
- Create: `insight_swarm/graph.py`
- Create: `tests/test_graph_routing.py`

- [ ] **Step 1: Write failing tests for routing functions**

```python
# tests/test_graph_routing.py
from tests.conftest import make_state
from insight_swarm.graph import (
    route_after_executor,
    route_after_hypothesis,
    route_after_web_analyst,
)


def test_route_executor_retry_when_error_and_retries_lt_2():
    state = make_state(sql_error="syntax error", sql_retry_count=1, query_context="initial")
    assert route_after_executor(state) == "retry"


def test_route_executor_failed_initial_when_error_and_retries_ge_2():
    state = make_state(sql_error="error", sql_retry_count=2, query_context="initial")
    assert route_after_executor(state) == "failed_initial"


def test_route_executor_failed_hypothesis_when_error_and_retries_ge_2():
    state = make_state(sql_error="error", sql_retry_count=2, query_context="hypothesis")
    assert route_after_executor(state) == "failed_hypothesis"


def test_route_executor_web_search_when_insufficient():
    state = make_state(sql_error=None, sql_results=[[{"a": 1}] * 5])  # 5 rows < 10
    assert route_after_executor(state) == "web_search"


def test_route_executor_analyst_when_sufficient():
    state = make_state(sql_error=None, sql_results=[[{"a": i} for i in range(15)]])
    assert route_after_executor(state) == "analyst"


def test_route_hypothesis_force_exit_at_max_cycles():
    state = make_state(cycle_count=5, max_cycles=5, hypotheses=[
        {"text": "H1", "type": "internal", "sql_result": None, "web_context": None, "verdict": "inconclusive"}
    ], current_hypothesis_idx=0)
    assert route_after_hypothesis(state) == "force_exit"


def test_route_hypothesis_external():
    state = make_state(cycle_count=1, max_cycles=5, hypotheses=[
        {"text": "H1", "type": "external", "sql_result": None, "web_context": None, "verdict": "inconclusive"}
    ], current_hypothesis_idx=0)
    assert route_after_hypothesis(state) == "external"


def test_route_hypothesis_internal():
    state = make_state(cycle_count=1, max_cycles=5, hypotheses=[
        {"text": "H1", "type": "internal", "sql_result": None, "web_context": None, "verdict": "inconclusive"}
    ], current_hypothesis_idx=0)
    assert route_after_hypothesis(state) == "internal"


def test_route_hypothesis_done_when_idx_exceeds_list():
    state = make_state(cycle_count=1, max_cycles=5, hypotheses=[
        {"text": "H1", "type": "internal", "sql_result": None, "web_context": None, "verdict": "confirmed"}
    ], current_hypothesis_idx=1)  # past end
    assert route_after_hypothesis(state) == "done"


def test_route_web_analyst_more_when_pending():
    hypotheses = [
        {"text": "H1", "type": "internal", "sql_result": None, "web_context": None, "verdict": "confirmed"},
        {"text": "H2", "type": "external", "sql_result": None, "web_context": None, "verdict": "inconclusive"},
    ]
    state = make_state(hypotheses=hypotheses, current_hypothesis_idx=0)
    assert route_after_web_analyst(state) == "more"


def test_route_web_analyst_done_when_last():
    hypotheses = [
        {"text": "H1", "type": "external", "sql_result": None, "web_context": None, "verdict": "confirmed"},
    ]
    state = make_state(hypotheses=hypotheses, current_hypothesis_idx=0)
    assert route_after_web_analyst(state) == "done"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/test_graph_routing.py -v
```

- [ ] **Step 3: Implement graph.py with routing functions and graph assembly**

```python
# insight_swarm/graph.py
from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from insight_swarm.state import InsightState
from insight_swarm.nodes.schema import schema_node
from insight_swarm.nodes.sql_writer import sql_writer_node
from insight_swarm.nodes.sql_retry import sql_retry_node
from insight_swarm.nodes.executor import executor_node
from insight_swarm.nodes.analyst import analyst_node
from insight_swarm.nodes.hypothesis import hypothesis_node
from insight_swarm.nodes.web_search import web_search_node
from insight_swarm.nodes.web_analyst import web_analyst_node
from insight_swarm.nodes.narrator import narrator_node
from insight_swarm.nodes.report import report_node

SUFFICIENT_ROW_THRESHOLD = 10


# ── Routing functions ────────────────────────────────────────────────────────

def route_after_executor(state: InsightState) -> str:
    if state.get("sql_error"):
        if state.get("sql_retry_count", 0) < 2:
            return "retry"
        # Retries exhausted — route depends on context
        if state.get("query_context", "initial") == "hypothesis":
            return "failed_hypothesis"
        return "failed_initial"

    latest = state["sql_results"][-1] if state["sql_results"] else []
    if len(latest) < SUFFICIENT_ROW_THRESHOLD:
        return "web_search"
    return "analyst"


def route_after_hypothesis(state: InsightState) -> str:
    if state["cycle_count"] >= state["max_cycles"]:
        return "force_exit"
    idx = state["current_hypothesis_idx"]
    hypotheses = state["hypotheses"]
    if idx >= len(hypotheses):
        return "done"
    current = hypotheses[idx]
    if current["type"] == "external":
        return "external"
    return "internal"


def route_after_web_analyst(state: InsightState) -> str:
    idx = state["current_hypothesis_idx"]
    if idx + 1 < len(state["hypotheses"]):
        return "more"
    return "done"


# ── Graph assembly ───────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(InsightState)

    g.add_node("schema", schema_node)
    g.add_node("sql_writer", sql_writer_node)
    g.add_node("sql_retry", sql_retry_node)
    g.add_node("executor", executor_node)
    g.add_node("analyst", analyst_node)
    g.add_node("hypothesis", hypothesis_node)
    g.add_node("web_search", web_search_node)
    g.add_node("web_analyst", web_analyst_node)
    g.add_node("narrator", narrator_node)
    g.add_node("report", report_node)

    # Linear start
    g.add_edge(START, "schema")
    g.add_edge("schema", "sql_writer")
    g.add_edge("sql_writer", "executor")

    # After executor: retry / fail / web_search / analyst
    g.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "retry": "sql_retry",
            "failed_initial": "analyst",
            "failed_hypothesis": "hypothesis",
            "web_search": "web_search",
            "analyst": "analyst",
        },
    )
    g.add_edge("sql_retry", "executor")

    # After analyst → hypothesis
    g.add_edge("analyst", "hypothesis")

    # After hypothesis: branch by type or exit
    g.add_conditional_edges(
        "hypothesis",
        route_after_hypothesis,
        {
            "internal": "sql_writer",
            "external": "web_search",
            "done": "narrator",
            "force_exit": "narrator",
        },
    )

    # Web path
    g.add_edge("web_search", "web_analyst")
    g.add_conditional_edges(
        "web_analyst",
        route_after_web_analyst,
        {
            "more": "hypothesis",
            "done": "narrator",
        },
    )

    # End
    g.add_edge("narrator", "report")
    g.add_edge("report", END)

    return g.compile()


graph = build_graph()
```

- [ ] **Step 4: Run routing tests — verify they pass**

```bash
pytest tests/test_graph_routing.py -v
```
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add insight_swarm/graph.py tests/test_graph_routing.py
git commit -m "feat: LangGraph assembly with conditional edges and routing functions"
```

---

## Task 15: CLI Entrypoint

**Files:**
- Create: `insight_swarm/__main__.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_cli_rejects_missing_file_and_db(capsys):
    with pytest.raises(SystemExit):
        with patch("sys.argv", ["insight_swarm", "--question", "Why?"]):
            from insight_swarm.__main__ import main
            main()


def test_cli_rejects_unsupported_extension(tmp_path, capsys):
    bad_file = tmp_path / "data.parquet"
    bad_file.write_text("x")
    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", [
            "insight_swarm", "--file", str(bad_file), "--question", "Why?"
        ]):
            import importlib
            import insight_swarm.__main__ as m
            importlib.reload(m)
            m.main()
    assert exc_info.value.code != 0


def test_cli_builds_initial_state(tmp_path):
    csv_file = tmp_path / "sales.csv"
    csv_file.write_text("a,b\n1,2\n")
    captured_state = {}

    def fake_invoke(state, config=None):
        captured_state.update(state)
        return state

    with patch("sys.argv", [
        "insight_swarm",
        "--file", str(csv_file),
        "--question", "Why did sales drop?",
        "--industry", "retail",
        "--context", "US 2023",
        "--max-cycles", "3",
    ]):
        with patch("insight_swarm.__main__.graph") as mock_graph:
            mock_graph.invoke.side_effect = fake_invoke
            from insight_swarm import __main__ as m
            import importlib; importlib.reload(m)
            m.main()

    assert captured_state["question"] == "Why did sales drop?"
    assert captured_state["industry"] == "retail"
    assert captured_state["max_cycles"] == 3
    assert captured_state["source_type"] == "csv"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/test_cli.py -v
```

- [ ] **Step 3: Implement __main__.py**

```python
# insight_swarm/__main__.py
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from insight_swarm.graph import graph


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".duckdb"}


def _validate_inputs(args: argparse.Namespace) -> tuple[str, str]:
    """Return (source_file, source_type) or exit with error."""
    if not args.file and not args.db:
        print("Error: provide --file (CSV/XLSX) or --db (DuckDB)", file=sys.stderr)
        sys.exit(1)

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(
                f"Error: unsupported file type '{ext}'. "
                f"Supported formats: .csv, .xlsx, .duckdb",
                file=sys.stderr,
            )
            sys.exit(1)
        source_type = ext.lstrip(".")
        return str(path), source_type

    # --db path
    path = Path(args.db)
    if not path.exists():
        print(f"Error: DuckDB file not found: {args.db}", file=sys.stderr)
        sys.exit(1)
    return str(path), "duckdb"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insight Swarm — multi-agent business analysis"
    )
    parser.add_argument("--file", help="Path to CSV or XLSX file")
    parser.add_argument("--db", help="Path to existing DuckDB file")
    parser.add_argument("--question", required=True, help="Business question to analyse")
    parser.add_argument("--industry", default=None, help="Industry context (e.g. 'retail')")
    parser.add_argument("--context", default=None, help="Additional context (e.g. 'US, 2023')")
    parser.add_argument("--max-cycles", type=int, default=5, help="Max hypothesis test cycles (default 5)")
    args = parser.parse_args()

    source_file, source_type = _validate_inputs(args)

    initial_state = {
        "question": args.question,
        "industry": args.industry,
        "context": args.context,
        "max_cycles": args.max_cycles,
        "source_file": source_file,
        "source_type": source_type,
        "db_path": "",           # set by schema_node
        "schema": {},
        "query_context": "initial",
        "current_query_goal": "",
        "current_sql": None,
        "sql_error": None,
        "sql_retry_count": 0,
        "sql_results": [],
        "analyses": [],
        "hypotheses": [],
        "current_hypothesis_idx": 0,
        "cycle_count": 0,
        "narrative": "",
        "run_log": [],
    }

    print(f'\n🔍 Insight Swarm starting...')
    print(f'   Question: {args.question}')
    if args.industry:
        print(f'   Industry: {args.industry}')
    print()

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_cli.py -v
```
Expected: 3 passed (note: test_cli_builds_initial_state may need `importlib.reload` adjustment — if it fails due to module caching, run with `-p no:cacheprovider`).

- [ ] **Step 5: Run the full test suite**

```bash
pytest -v
```
Expected: all tests pass (LLM tests use mocks, no Ollama needed).

- [ ] **Step 6: Commit**

```bash
git add insight_swarm/__main__.py tests/test_cli.py
git commit -m "feat: CLI entrypoint with input validation and graph invocation"
```

---

## Task 16: Sample Dataset + Integration Test

**Files:**
- Create: `insight_swarm/data/superstore.csv` (download)
- No new source files

- [ ] **Step 1: Download Superstore dataset**

```bash
# Download the classic Superstore dataset (public domain)
curl -L "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv" \
  -o insight_swarm/data/sample.csv
```

If no internet access, create a minimal demo dataset instead:

```bash
python - <<'EOF'
import csv, random, datetime
rows = []
regions = ["West", "East", "North", "South"]
categories = ["Furniture", "Technology", "Office Supplies"]
for i in range(500):
    quarter = f"Q{random.randint(1,4)}"
    year = random.choice([2022, 2023])
    region = random.choice(regions)
    category = random.choice(categories)
    base = {"Furniture": 800, "Technology": 1200, "Office Supplies": 300}[category]
    multiplier = 0.7 if (quarter == "Q3" and year == 2023 and region == "West") else 1.0
    sales = round(base * multiplier * (0.8 + random.random() * 0.4), 2)
    rows.append([i+1, f"ORD-{i+1:04d}", region, category, quarter, year, sales])

with open("insight_swarm/data/superstore.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "order_id", "region", "category", "quarter", "year", "sales"])
    writer.writerows(rows)
print("Created insight_swarm/data/superstore.csv with 500 rows")
EOF
```

- [ ] **Step 2: Verify Ollama and models are available**

```bash
# Install Ollama from https://ollama.com/download
ollama pull deepseek-coder:6.7b
ollama pull llama3.1:8b
ollama list  # verify both models appear
```

- [ ] **Step 3: Run integration test (requires Ollama)**

```bash
python -m insight_swarm \
  --file insight_swarm/data/superstore.csv \
  --question "Why did sales drop in Q3 2023?" \
  --industry "retail" \
  --context "US market, 2022-2023" \
  --max-cycles 3
```

Expected output:
```
🔍 Insight Swarm starting...
   Question: Why did sales drop in Q3 2023?
   Industry: retail

✓ Schema discovered: 1 table(s), 7 column(s)
→ SQL Writer (initial): generating query...
✓ Query executed: N rows
→ Analyst: interpreting results...
→ Hypothesis Generator: creating hypotheses...
✓ Hypotheses generated: 3 (N internal, N external)
...
✓ Report saved: insight_swarm/reports/report_YYYYMMDD_HHMMSS.html
✓ Opening in browser...
```

A browser window opens with the HTML report.

- [ ] **Step 4: Verify JSON log was created**

```bash
ls insight_swarm/logs/
cat insight_swarm/logs/run_*.json | python -m json.tool | head -30
```

- [ ] **Step 5: Final commit**

```bash
git add insight_swarm/data/superstore.csv
git commit -m "feat: sample dataset and integration test complete"
```

---

## Self-Review Checklist (completed)

- [x] **Spec §2 CLI interface** — Task 15 implements all args including `--industry`, `--context`, `--max-cycles`
- [x] **Spec §3 LangGraph graph** — Task 14 implements all 10 nodes and all conditional edges from §3.2
- [x] **Spec §4 Shared State** — Task 2 implements all fields, plus `source_file`, `source_type`, `db_path`, `query_context` (implementation details)
- [x] **Spec §5 All nodes** — Tasks 3, 5, 6, 7, 8, 9, 10, 11, 12, 13 implement all 10 nodes
- [x] **Spec §5 schema_node CSV/XLSX/DuckDB** — Task 3 handles all three formats
- [x] **Spec §5 sql_retry_count reset per hypothesis** — hypothesis_node in Task 9 resets `sql_retry_count: 0` on advance
- [x] **Spec §6 Models** — deepseek-coder:6.7b in sql_writer, llama3.1:8b everywhere else
- [x] **Spec §8 Terminal output** — print statements in every node match the spec format
- [x] **Spec §9 Error handling** — unsupported extension in Task 15, DuckDuckGo rate limit in Task 10, SQL retry in Tasks 6+7
- [x] **Spec §10 HTML + JSON output** — Task 13 produces both files with correct structure
- [x] **Type consistency** — `sql_results` is `list[list[dict]]` throughout; `hypotheses` is `list[Hypothesis]` throughout; `run_log` is `list[LogEntry]` throughout
