from __future__ import annotations
import re
import time
from datetime import datetime, timezone
from duckduckgo_search import DDGS
from insight_swarm.state import InsightState, LogEntry


def _build_query(hypothesis: str, question: str, industry: str | None, context: str | None) -> str:
    parts = [hypothesis]
    if industry:
        parts.append(industry)
    if context:
        years = re.findall(r"\b20\d{2}\b", context)
        if years:
            parts.extend(years)
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
    results = []

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
        "summary": f"Search for H{idx + 1}: {len(results)} results",
        "detail": {"query": query, "result_count": len(results)},
    }

    return {
        "hypotheses": updated_hypotheses,
        "run_log": state["run_log"] + [log],
    }
