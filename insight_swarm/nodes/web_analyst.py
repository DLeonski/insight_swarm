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

    industry_line = f"Industry: {state['industry']}" if state.get("industry") else ""
    prompt = _TEMPLATE.format(
        hypothesis_text=hypothesis["text"],
        question=state["question"],
        industry_line=industry_line,
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
