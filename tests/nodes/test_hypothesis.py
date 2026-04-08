import json
from unittest.mock import patch

_MOCK_HYPOTHESES = json.dumps([
    {"text": "West region had seasonal drop", "type": "internal"},
    {"text": "Holiday calendar affected retail", "type": "external"},
    {"text": "Furniture category declined", "type": "internal"},
])


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


def test_hypothesis_generates_on_first_call():
    state = make_state(hypotheses=[], analyses=["Q3 was 20% below Q2."])
    with patch("insight_swarm.nodes.hypothesis.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = _MOCK_HYPOTHESES
        from insight_swarm.nodes.hypothesis import hypothesis_node
        result = hypothesis_node(state)

    assert len(result["hypotheses"]) == 3
    assert result["hypotheses"][0]["type"] in ("internal", "external")
    assert result["current_hypothesis_idx"] == 0


def test_hypothesis_advances_on_subsequent_call():
    hypotheses = [
        {"text": "H1", "type": "internal", "sql_result": None, "web_context": None, "verdict": "confirmed"},
        {"text": "H2", "type": "external", "sql_result": None, "web_context": None, "verdict": "inconclusive"},
        {"text": "H3", "type": "internal", "sql_result": None, "web_context": None, "verdict": "inconclusive"},
    ]
    state = make_state(hypotheses=hypotheses, current_hypothesis_idx=0, cycle_count=1)
    from insight_swarm.nodes.hypothesis import hypothesis_node
    result = hypothesis_node(state)

    assert result["current_hypothesis_idx"] == 1
    assert result["sql_retry_count"] == 0  # reset for next hypothesis
    assert result["cycle_count"] == 2


def test_hypothesis_sets_query_goal_for_internal():
    hypotheses = [
        {"text": "West region seasonal drop", "type": "internal",
         "sql_result": None, "web_context": None, "verdict": "inconclusive"},
    ]
    state = make_state(hypotheses=hypotheses, current_hypothesis_idx=0)
    from insight_swarm.nodes.hypothesis import hypothesis_node
    result = hypothesis_node(state)
    assert "West region seasonal drop" in result.get("current_query_goal", "")


def test_hypothesis_appends_log():
    state = make_state(hypotheses=[], analyses=["some analysis"])
    with patch("insight_swarm.nodes.hypothesis.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = _MOCK_HYPOTHESES
        from insight_swarm.nodes.hypothesis import hypothesis_node
        result = hypothesis_node(state)
    assert result["run_log"][-1]["node"] == "hypothesis"
