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
        "analyses": ["Q3 was down 20%.", "West region down most."],
        "hypotheses": [
            {"text": "Seasonal effect", "type": "internal", "sql_result": None,
             "web_context": None, "verdict": "confirmed"},
            {"text": "Competitor actions", "type": "external", "sql_result": None,
             "web_context": "No evidence found.", "verdict": "rejected"},
        ],
        "current_hypothesis_idx": 0,
        "cycle_count": 2,
        "narrative": "",
        "run_log": [],
    }
    return {**base, **overrides}


def test_narrator_sets_narrative():
    from insight_swarm.nodes.narrator import narrator_node
    state = make_state()
    with patch("insight_swarm.nodes.narrator.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = (
            "Sales declined primarily due to seasonal factors. "
            "The West region showed the steepest drop. "
            "Competitor actions were investigated but found to be immaterial."
        )
        result = narrator_node(state)

    assert len(result["narrative"]) > 20


def test_narrator_appends_log():
    from insight_swarm.nodes.narrator import narrator_node
    state = make_state()
    with patch("insight_swarm.nodes.narrator.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = "Summary text."
        result = narrator_node(state)
    assert result["run_log"][-1]["node"] == "narrator"
