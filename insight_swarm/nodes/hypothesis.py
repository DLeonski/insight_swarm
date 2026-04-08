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

        goal_hypothesis = current if current else state["hypotheses"][state["current_hypothesis_idx"]]
        updates["current_query_goal"] = f"Test hypothesis: {goal_hypothesis['text']}"

        return updates

    # First call: generate hypotheses
    llm = Ollama(model=REASONING_MODEL)
    analyses_text = "\n\n".join(state["analyses"]) or "No analysis yet."

    prompt = (
        _TEMPLATE
        .replace("{question}", state["question"])
        .replace("{industry_line}", f"Industry: {state['industry']}" if state.get("industry") else "")
        .replace("{context_line}", f"Context: {state['context']}" if state.get("context") else "")
        .replace("{analyses}", analyses_text)
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
