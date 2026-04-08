from __future__ import annotations
import re
import time
from datetime import datetime, timezone
from duckduckgo_search import DDGS
from insight_swarm.state import InsightState, LogEntry

# Signals that indicate a result is a local/commercial listing, not business research
_SPAM_SIGNALS = frozenset([
    "menu", "restaurant", "taco", "burger", "pizza", "sandwich", "coffee",
    "near me", "hours of operation", "order online", "delivery", "takeout",
    "participating locations", "while supplies last", "find a location",
])


def _build_query(hypothesis: str, question: str, industry: str | None, context: str | None) -> str:
    # Use only the first 10 words of the hypothesis to keep the query focused
    core = " ".join(hypothesis.split()[:10])
    parts = [core]
    if industry:
        parts.append(industry)
    if context:
        years = re.findall(r"\b20\d{2}\b", context)
        parts.extend(years[:2])
    # Bias toward credible business/research sources
    parts.append("market statistics trends report")
    return " ".join(parts)


def _is_relevant(result: dict) -> bool:
    text = (result.get("title", "") + " " + result.get("body", "")).lower()
    return not any(signal in text for signal in _SPAM_SIGNALS)


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

    print(f'→ Web Search: "{query[:80]}"')
    web_context: str | None = None
    kept: list[dict] = []

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=10))
        kept = [r for r in raw_results if _is_relevant(r)][:5]
        if kept:
            snippets = [f"[{r['title']}] {r['body']}" for r in kept]
            web_context = "\n".join(snippets)
            print(f"✓ Web search: {len(kept)} relevant results (filtered from {len(raw_results)})")
        else:
            print(f"✓ Web search: no relevant results ({len(raw_results)} filtered out as spam)")
    except Exception as exc:
        print(f"✗ Web search failed: {exc} — marking as inconclusive")

    updated_hypotheses = list(state["hypotheses"])
    updated_hypotheses[idx] = {**hypothesis, "web_context": web_context}

    log: LogEntry = {
        "node": "web_search",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - start) * 1000),
        "summary": f"Search for H{idx + 1}: {len(kept)} relevant results",
        "detail": {"query": query, "result_count": len(kept)},
    }

    return {
        "hypotheses": updated_hypotheses,
        "run_log": state["run_log"] + [log],
    }
