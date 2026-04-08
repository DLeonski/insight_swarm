from __future__ import annotations
from typing import Any, Literal, Optional
from typing_extensions import TypedDict


class Hypothesis(TypedDict):
    text: str
    type: Literal["internal", "external"]
    sql_result: Optional[list[dict]]   # records format, None if not yet run
    web_context: Optional[str]
    verdict: Literal["confirmed", "rejected", "inconclusive"]


class LogEntry(TypedDict):
    node: str
    timestamp: str       # ISO-8601
    duration_ms: int
    summary: str
    detail: Any


class InsightState(TypedDict):
    # Input
    question: str
    industry: Optional[str]
    context: Optional[str]
    max_cycles: int

    # File source (set by __main__, used by schema + executor)
    source_file: str      # original file path
    source_type: str      # "csv" | "xlsx" | "duckdb"
    db_path: str          # path to working .duckdb file (set by schema_node)

    # Schema
    schema: dict          # {table_name: [{"column": str, "type": str}]}

    # SQL execution
    query_context: str    # "initial" | "hypothesis"
    current_query_goal: str
    current_sql: Optional[str]
    sql_error: Optional[str]
    sql_retry_count: int  # reset to 0 per hypothesis by hypothesis_node
    sql_results: list     # list of list[dict] — one entry per query run

    # Analysis
    analyses: list        # list[str]

    # Hypotheses
    hypotheses: list      # list[Hypothesis]
    current_hypothesis_idx: int
    cycle_count: int

    # Output
    narrative: str
    run_log: list         # list[LogEntry]
