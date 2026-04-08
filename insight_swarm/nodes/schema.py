from __future__ import annotations
import re
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


def _compute_stats(df: pd.DataFrame) -> list[dict]:
    """Compute a human-readable summary of the DataFrame as a list of records."""
    records: list[dict] = []
    records.append({"metric": "Overview", "value": f"{len(df):,} rows, {len(df.columns)} columns"})

    # Detect a year or date column for temporal aggregation
    year_col: str | None = None
    df_work = df.copy()
    for col in df_work.columns:
        if re.search(r"year", col, re.IGNORECASE):
            year_col = col
            break
        if re.search(r"date", col, re.IGNORECASE) and df_work[col].dtype == object:
            try:
                parsed = pd.to_datetime(df_work[col], errors="coerce")
                if parsed.notna().sum() > len(df_work) * 0.5:
                    df_work["_year"] = parsed.dt.year
                    year_col = "_year"
                    break
            except Exception:
                pass

    num_cols = [c for c in df_work.select_dtypes(include="number").columns
                if c != year_col and not c.startswith("_")]

    if year_col:
        for ncol in num_cols[:4]:
            agg = df_work.groupby(year_col)[ncol].sum()
            agg_str = " | ".join(f"{int(yr)}: {val:,.0f}" for yr, val in agg.items())
            records.append({"metric": f"{ncol} by year", "value": agg_str})
    else:
        for col in num_cols[:6]:
            s = df_work[col]
            records.append({
                "metric": col,
                "value": f"sum={s.sum():,.0f}  mean={s.mean():,.1f}  min={s.min():,.1f}  max={s.max():,.1f}",
            })

    for col in df_work.select_dtypes(include="object").columns[:6]:
        top = df_work[col].value_counts().head(5)
        records.append({
            "metric": col,
            "value": " | ".join(f"{v} ({c})" for v, c in top.items()),
        })

    return records


def schema_node(state: InsightState) -> dict:
    start = time.time()
    source_file = state["source_file"]
    source_type = state["source_type"]

    # Build a working DuckDB file path next to the source
    db_path = str(Path(source_file).with_suffix(".duckdb"))
    con = duckdb.connect(db_path)

    initial_sql_results: list = []

    if source_type == "csv":
        table_name = Path(source_file).stem.replace("-", "_").replace(" ", "_")
        table_name = re.sub(r"_+", "_", table_name).strip("_")
        # Try UTF-8 first; fall back to latin-1 for Excel/Tableau-generated CSVs
        try:
            df = pd.read_csv(source_file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(source_file, encoding="latin-1")
        con.register("_csv_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _csv_tmp")
        con.unregister("_csv_tmp")
        initial_sql_results = [_compute_stats(df)]
        print(f"✓ CSV loaded: {len(df):,} rows — pandas stats computed, skipping SQL")

    elif source_type == "xlsx":
        sheets = pd.read_excel(source_file, sheet_name=None)
        first_df: pd.DataFrame | None = None
        for sheet_name, df in sheets.items():
            safe_name = re.sub(r"_+", "_", sheet_name.replace(" ", "_").replace("-", "_")).strip("_")
            con.register(safe_name, df)
            con.execute(f"CREATE OR REPLACE TABLE {safe_name} AS SELECT * FROM {safe_name}")
            if first_df is None:
                first_df = df
        if first_df is not None:
            initial_sql_results = [_compute_stats(first_df)]
        print(f"✓ XLSX loaded: {len(sheets)} sheet(s) — pandas stats computed, skipping SQL")

    # duckdb: already connected, tables already exist

    # Discover schema
    tables = con.execute("SHOW TABLES").fetchall()
    schema: dict = {}
    for (tname,) in tables:
        columns_raw = con.execute(f"DESCRIBE {tname}").fetchall()
        row_count = con.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
        schema[tname] = [
            {"column": row[0], "type": row[1], "row_count": row_count}
            for row in columns_raw
        ]
    con.close()

    total_tables = len(schema)
    total_cols = sum(len(v) for v in schema.values())
    print(f"✓ Schema discovered: {total_tables} table(s), {total_cols} column(s)")

    log = _build_log(
        summary=f"Discovered {total_tables} table(s), {total_cols} column(s)",
        detail={"tables": list(schema.keys())},
        start=start,
    )

    result: dict = {
        "db_path": db_path,
        "schema": schema,
        "current_query_goal": f"Analyse data to answer: {state['question']}",
        "run_log": state["run_log"] + [log],
    }
    if initial_sql_results:
        result["sql_results"] = initial_sql_results

    return result
