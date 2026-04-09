from __future__ import annotations
from typing import Any, Literal, Optional
from typing_extensions import TypedDict


class Hypothesis(TypedDict):
    text: str
    type: Literal["internal", "external"]
    data_slice: Optional[list[dict]]   # last pandas result used by evaluator
    web_context: Optional[str]         # filtered summary from web_analyst
    verdict: Literal["confirmed", "rejected", "inconclusive"]
    evidence: str                      # specific text/numbers cited by evaluator


class LogEntry(TypedDict):
    node: str
    timestamp: str       # ISO-8601
    duration_ms: int
    summary: str
    detail: Any


class InsightState(TypedDict):
    # Run identity (for SSE progress)
    run_id: str

    # Input
    question: str
    industry: Optional[str]
    context: Optional[str]
    max_cycles: int

    # File
    source_file: str      # path to uploaded file
    source_type: str      # "csv" | "xlsx"

    # Schema & summary (set by schema_node)
    schema: dict          # {column_name: dtype_string} flat dict
    data_summary: list[dict]  # precomputed groupby summary rows

    # Analysis
    analyses: list        # list[str]

    # Hypotheses
    hypotheses: list      # list[Hypothesis] — completed with verdicts
    current_hypothesis: Optional[dict]   # Hypothesis being evaluated right now
    evaluator_rounds: int                # resets to 0 per hypothesis
    cycle_count: int

    # Evaluator ↔ Data Slicer exchange
    data_request: Optional[dict]     # {group_by: [str], metric: str, filter?: {str: Any}}
    current_data_slice: list         # list[dict] result from data_slicer

    # Output
    narrative: str
    run_log: list         # list[LogEntry]
