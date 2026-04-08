# tests/conftest.py
import pytest
from insight_swarm.state import InsightState


def make_state(**overrides) -> InsightState:
    base: InsightState = {
        "question": "Why did sales drop in Q3?",
        "industry": "retail",
        "context": "US market, 2023",
        "max_cycles": 5,
        "source_file": "data/superstore.csv",
        "source_type": "csv",
        "db_path": "/tmp/test.duckdb",
        "schema": {
            "orders": [
                {"column": "order_id", "type": "VARCHAR", "row_count": 1000},
                {"column": "sales", "type": "DOUBLE", "row_count": 1000},
                {"column": "quarter", "type": "VARCHAR", "row_count": 1000},
                {"column": "region", "type": "VARCHAR", "row_count": 1000},
            ]
        },
        "query_context": "initial",
        "current_query_goal": "Analyse overall sales trend by quarter",
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
    return {**base, **overrides}


@pytest.fixture(name="make_state")
def make_state_fixture():
    return make_state
