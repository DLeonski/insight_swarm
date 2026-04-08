from unittest.mock import patch, MagicMock


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


def _state_with_hypothesis(h_text="Holiday effect on retail", h_type="external"):
    hypotheses = [
        {"text": h_text, "type": h_type,
         "sql_result": None, "web_context": None, "verdict": "inconclusive"}
    ]
    return make_state(hypotheses=hypotheses, current_hypothesis_idx=0)


def test_web_search_sets_web_context():
    from insight_swarm.nodes.web_search import web_search_node
    state = _state_with_hypothesis()
    mock_results = [
        {"title": "Q3 Retail Drop", "body": "Sales fell 15% due to fewer shopping days.", "href": "http://x.com"},
        {"title": "Holiday Schedule", "body": "Labor Day shifted consumer spending patterns.", "href": "http://y.com"},
    ]
    with patch("insight_swarm.nodes.web_search.DDGS") as MockDDGS:
        MockDDGS.return_value.__enter__ = lambda s: s
        MockDDGS.return_value.__exit__ = MagicMock(return_value=False)
        MockDDGS.return_value.text.return_value = mock_results
        result = web_search_node(state)

    hypotheses = result["hypotheses"]
    assert hypotheses[0]["web_context"] is not None
    assert "Retail Drop" in hypotheses[0]["web_context"] or "Holiday" in hypotheses[0]["web_context"]


def test_web_search_handles_rate_limit_gracefully():
    from insight_swarm.nodes.web_search import web_search_node
    state = _state_with_hypothesis()
    with patch("insight_swarm.nodes.web_search.DDGS") as MockDDGS:
        MockDDGS.return_value.__enter__ = lambda s: s
        MockDDGS.return_value.__exit__ = MagicMock(return_value=False)
        MockDDGS.return_value.text.side_effect = Exception("Ratelimit")
        result = web_search_node(state)

    hypotheses = result["hypotheses"]
    assert hypotheses[0]["web_context"] is None


def test_web_search_appends_log():
    from insight_swarm.nodes.web_search import web_search_node
    state = _state_with_hypothesis()
    with patch("insight_swarm.nodes.web_search.DDGS") as MockDDGS:
        MockDDGS.return_value.__enter__ = lambda s: s
        MockDDGS.return_value.__exit__ = MagicMock(return_value=False)
        MockDDGS.return_value.text.return_value = []
        result = web_search_node(state)

    assert result["run_log"][-1]["node"] == "web_search"
