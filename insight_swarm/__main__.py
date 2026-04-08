from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Placeholder import - will be satisfied when graph.py is committed
try:
    from insight_swarm.graph import graph
except ImportError:
    graph = None


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".duckdb"}


def _validate_inputs(args: argparse.Namespace) -> tuple[str, str]:
    """Return (source_file, source_type) or exit with error."""
    if not args.file and not args.db:
        print("Error: provide --file (CSV/XLSX) or --db (DuckDB)", file=sys.stderr)
        sys.exit(1)

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(
                f"Error: unsupported file type '{ext}'. "
                f"Supported formats: .csv, .xlsx, .duckdb",
                file=sys.stderr,
            )
            sys.exit(1)
        source_type = ext.lstrip(".")
        return str(path), source_type

    # --db path
    path = Path(args.db)
    if not path.exists():
        print(f"Error: DuckDB file not found: {args.db}", file=sys.stderr)
        sys.exit(1)
    return str(path), "duckdb"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insight Swarm — multi-agent business analysis"
    )
    parser.add_argument("--file", help="Path to CSV or XLSX file")
    parser.add_argument("--db", help="Path to existing DuckDB file")
    parser.add_argument("--question", required=True, help="Business question to analyse")
    parser.add_argument("--industry", default=None, help="Industry context (e.g. 'retail')")
    parser.add_argument("--context", default=None, help="Additional context (e.g. 'US, 2023')")
    parser.add_argument("--max-cycles", type=int, default=5, help="Max hypothesis test cycles (default 5)")
    args = parser.parse_args()

    source_file, source_type = _validate_inputs(args)

    initial_state = {
        "question": args.question,
        "industry": args.industry,
        "context": args.context,
        "max_cycles": args.max_cycles,
        "source_file": source_file,
        "source_type": source_type,
        "db_path": "",
        "schema": {},
        "query_context": "initial",
        "current_query_goal": "",
        "current_sql": None,
        "sql_error": None,
        "sql_retry_count": 0,
        "sql_results": [],
        "analyses": [],
        "hypotheses": [],
        "current_hypothesis_idx": 0,
        "cycle_count": 0,
        "narrative": "",
        "run_log": [],
    }

    print(f'\nInsight Swarm starting...')
    print(f'   Question: {args.question}')
    if args.industry:
        print(f'   Industry: {args.industry}')
    print()

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
