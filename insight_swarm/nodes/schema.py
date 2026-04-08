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
    print(f"Schema discovered: {total_tables} table(s), {total_cols} column(s)")

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
