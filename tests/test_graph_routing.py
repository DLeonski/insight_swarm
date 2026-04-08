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


def test_route_executor_retry_when_error_and_retries_lt_2():
    from insight_swarm.graph import route_after_executor
    state = make_state(sql_error="syntax error", sql_retry_count=1, query_context="initial")
    assert route_after_executor(state) == "retry"


def test_route_executor_failed_initial_when_error_and_retries_ge_2():
    from insight_swarm.graph import route_after_executor
    state = make_state(sql_error="error", sql_retry_count=2, query_context="initial")
    assert route_after_executor(state) == "failed_initial"


def test_route_executor_failed_hypothesis_when_error_and_retries_ge_2():
    from insight_swarm.graph import route_after_executor
    state = make_state(sql_error="error", sql_retry_count=2, query_context="hypothesis")
    assert route_after_executor(state) == "failed_hypothesis"


def test_route_executor_web_search_when_insufficient():
    from insight_swarm.graph import route_after_executor
    state = make_state(sql_error=None, sql_results=[[{"a": 1}] * 5])  # 5 rows < 10
    assert route_after_executor(state) == "web_search"


def test_route_executor_analyst_when_sufficient():
    from insight_swarm.graph import route_after_executor
    state = make_state(sql_error=None, sql_results=[[{"a": i} for i in range(15)]])
    assert route_after_executor(state) == "analyst"


def test_route_hypothesis_force_exit_at_max_cycles():
    from insight_swarm.graph import route_after_hypothesis
    state = make_state(cycle_count=5, max_cycles=5, hypotheses=[
        {"text": "H1", "type": "internal", "sql_result": None, "web_context": None, "verdict": "inconclusive"}
    ], current_hypothesis_idx=0)
    assert route_after_hypothesis(state) == "force_exit"


def test_route_hypothesis_external():
    from insight_swarm.graph import route_after_hypothesis
    state = make_state(cycle_count=1, max_cycles=5, hypotheses=[
        {"text": "H1", "type": "external", "sql_result": None, "web_context": None, "verdict": "inconclusive"}
    ], current_hypothesis_idx=0)
    assert route_after_hypothesis(state) == "external"


def test_route_hypothesis_internal():
    from insight_swarm.graph import route_after_hypothesis
    state = make_state(cycle_count=1, max_cycles=5, hypotheses=[
        {"text": "H1", "type": "internal", "sql_result": None, "web_context": None, "verdict": "inconclusive"}
    ], current_hypothesis_idx=0)
    assert route_after_hypothesis(state) == "internal"


def test_route_hypothesis_done_when_idx_exceeds_list():
    from insight_swarm.graph import route_after_hypothesis
    state = make_state(cycle_count=1, max_cycles=5, hypotheses=[
        {"text": "H1", "type": "internal", "sql_result": None, "web_context": None, "verdict": "confirmed"}
    ], current_hypothesis_idx=1)  # past end
    assert route_after_hypothesis(state) == "done"


def test_route_web_analyst_more_when_pending():
    from insight_swarm.graph import route_after_web_analyst
    hypotheses = [
        {"text": "H1", "type": "internal", "sql_result": None, "web_context": None, "verdict": "confirmed"},
        {"text": "H2", "type": "external", "sql_result": None, "web_context": None, "verdict": "inconclusive"},
    ]
    state = make_state(hypotheses=hypotheses, current_hypothesis_idx=0)
    assert route_after_web_analyst(state) == "more"


def test_route_web_analyst_done_when_last():
    from insight_swarm.graph import route_after_web_analyst
    hypotheses = [
        {"text": "H1", "type": "external", "sql_result": None, "web_context": None, "verdict": "confirmed"},
    ]
    state = make_state(hypotheses=hypotheses, current_hypothesis_idx=0)
    assert route_after_web_analyst(state) == "done"
