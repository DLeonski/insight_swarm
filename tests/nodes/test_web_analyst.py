import json
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
        "schema": {},
        "query_context": "hypothesis",
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
    return {**base, **overrides}


def _state_with_web_hypothesis(verdict_response="confirmed"):
    hypotheses = [
        {
            "text": "Holiday calendar reduced shopping days",
            "type": "external",
            "sql_result": [{"quarter": "Q3", "sales": 800}],
            "web_context": "[Retail News] Labor Day shift reduced consumer spend by 12%.",
            "verdict": "inconclusive",
        }
    ]
    return make_state(hypotheses=hypotheses, current_hypothesis_idx=0), json.dumps(
        {"verdict": verdict_response, "explanation": "Labor Day shift confirmed by web sources."}
    )


def test_web_analyst_sets_verdict():
    from insight_swarm.nodes.web_analyst import web_analyst_node
    state, mock_response = _state_with_web_hypothesis("confirmed")
    with patch("insight_swarm.nodes.web_analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = mock_response
        result = web_analyst_node(state)

    h = result["hypotheses"][0]
    assert h["verdict"] == "confirmed"


def test_web_analyst_appends_analysis():
    from insight_swarm.nodes.web_analyst import web_analyst_node
    state, mock_response = _state_with_web_hypothesis("rejected")
    with patch("insight_swarm.nodes.web_analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = mock_response
        result = web_analyst_node(state)

    assert len(result["analyses"]) == len(state["analyses"]) + 1


def test_web_analyst_handles_malformed_json():
    from insight_swarm.nodes.web_analyst import web_analyst_node
    state, _ = _state_with_web_hypothesis()
    with patch("insight_swarm.nodes.web_analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = "The hypothesis is confirmed by the data."
        result = web_analyst_node(state)

    assert result["hypotheses"][0]["verdict"] in ("confirmed", "rejected", "inconclusive")


def test_web_analyst_appends_log():
    from insight_swarm.nodes.web_analyst import web_analyst_node
    state, mock_response = _state_with_web_hypothesis()
    with patch("insight_swarm.nodes.web_analyst.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = mock_response
        result = web_analyst_node(state)
    assert result["run_log"][-1]["node"] == "web_analyst"
