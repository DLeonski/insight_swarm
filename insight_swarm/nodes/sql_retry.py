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
