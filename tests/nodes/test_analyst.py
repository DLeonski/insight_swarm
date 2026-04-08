from unittest.mock import patch


def make_state(**overrides):
    base = {
        "question": "Why did sales drop in Q3?",
        "industry": "retail",
        "context": "US market, 2023",
        "max_cycles": 5,
        "source_file": "data/superstore.csv",
        "source_type": "csv",
        "db_path": "/tmp/test.duckdb",
        "schema": {"orders": [{"column": "sales", "type": "DOUBLE", "row_count": 100}]},
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


def test_analyst_appends_analysis():
    state = make_state(
        sql_results=[[{"quarter": "Q3", "total_sales": 800.0}]],
        analyses=[],
    )
    with patch("insight_swarm.nodes.analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = (
            "Q3 sales were 33% lower than Q2, indicating a significant seasonal dip."
        )
        from insight_swarm.nodes.analyst import analyst_node
        result = analyst_node(state)

    assert len(result["analyses"]) == 1
    assert "Q3" in result["analyses"][0]


def test_analyst_handles_empty_results():
    state = make_state(sql_results=[None], analyses=[])
    with patch("insight_swarm.nodes.analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = "No data available for analysis."
        from insight_swarm.nodes.analyst import analyst_node
        result = analyst_node(state)

    assert len(result["analyses"]) == 1


def test_analyst_appends_log():
    state = make_state(sql_results=[[{"val": 1}]])
    with patch("insight_swarm.nodes.analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = "Some insight."
        from insight_swarm.nodes.analyst import analyst_node
        result = analyst_node(state)

    assert result["run_log"][-1]["node"] == "analyst"
