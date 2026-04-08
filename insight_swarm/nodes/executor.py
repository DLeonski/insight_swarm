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
            "sql_results": state["sql_results"],
            "run_log": state["run_log"] + [log],
        }
