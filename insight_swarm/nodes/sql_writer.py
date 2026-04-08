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
