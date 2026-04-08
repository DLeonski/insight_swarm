def make_state(**overrides):
    base = {
        "question": "Why did sales drop in Q3?",
        "industry": "retail",
        "context": "US market, 2023",
        "max_cycles": 5,
        "source_file": "data/superstore.csv",
        "source_type": "csv",
        "db_path": "/tmp/test.duckdb",
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


def test_sql_retry_increments_count():
    from insight_swarm.nodes.sql_retry import sql_retry_node
    state = make_state(sql_retry_count=0, sql_error="syntax error")
    result = sql_retry_node(state)
    assert result["sql_retry_count"] == 1


def test_sql_retry_increments_again():
    from insight_swarm.nodes.sql_retry import sql_retry_node
    state = make_state(sql_retry_count=1, sql_error="type mismatch")
    result = sql_retry_node(state)
    assert result["sql_retry_count"] == 2


def test_sql_retry_appends_log():
    from insight_swarm.nodes.sql_retry import sql_retry_node
    state = make_state(sql_retry_count=0, sql_error="error")
    result = sql_retry_node(state)
    assert result["run_log"][-1]["node"] == "sql_retry"
