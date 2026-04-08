import duckdb
import pytest
from pathlib import Path


def make_state(**overrides):
    base = {
        "question": "Why did sales drop in Q3?",
        "industry": "retail",
        "context": "US market, 2023",
        "max_cycles": 5,
        "source_file": "data/superstore.csv",
        "source_type": "csv",
        "db_path": "",
        "schema": {},
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


@pytest.fixture
def db_with_orders(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE orders AS
        SELECT * FROM (VALUES
            ('Q1', 'West', 1000.0),
            ('Q2', 'West', 1200.0),
            ('Q3', 'West', 800.0),
            ('Q1', 'East', 900.0),
            ('Q2', 'East', 950.0),
            ('Q3', 'East', 870.0),
            ('Q1', 'North', 700.0),
            ('Q2', 'North', 720.0),
            ('Q3', 'North', 650.0),
            ('Q1', 'South', 500.0),
            ('Q2', 'South', 510.0)
        ) t(quarter, region, sales)
    """)
    con.close()
    return db_path


def test_executor_successful_query(db_with_orders):
    from insight_swarm.nodes.executor import executor_node
    state = make_state(
        db_path=db_with_orders,
        current_sql="SELECT quarter, SUM(sales) as total FROM orders GROUP BY quarter",
    )
    result = executor_node(state)

    assert result["sql_error"] is None
    assert len(result["sql_results"]) == 1
    rows = result["sql_results"][0]
    assert len(rows) == 3  # Q1, Q2, Q3


def test_executor_failed_query_sets_error(db_with_orders):
    from insight_swarm.nodes.executor import executor_node
    state = make_state(
        db_path=db_with_orders,
        current_sql="SELECT * FORM nonexistent_table",
    )
    result = executor_node(state)

    assert result["sql_error"] is not None
    assert "sql_results" in result  # list unchanged


def test_executor_clears_error_on_success(db_with_orders):
    from insight_swarm.nodes.executor import executor_node
    state = make_state(
        db_path=db_with_orders,
        current_sql="SELECT 1 AS val",
        sql_error="previous error",
        sql_retry_count=1,
    )
    result = executor_node(state)

    assert result["sql_error"] is None
    assert result["sql_retry_count"] == 0


def test_executor_appends_log(db_with_orders):
    from insight_swarm.nodes.executor import executor_node
    state = make_state(db_path=db_with_orders, current_sql="SELECT 1 AS val")
    result = executor_node(state)
    assert result["run_log"][-1]["node"] == "executor"
