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
