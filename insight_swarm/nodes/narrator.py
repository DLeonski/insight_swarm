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
