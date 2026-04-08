from insight_swarm.state import InsightState, Hypothesis, LogEntry

def test_hypothesis_has_required_keys():
    h: Hypothesis = {
        "text": "Sales dropped due to seasonality",
        "type": "internal",
        "sql_result": None,
        "web_context": None,
        "verdict": "inconclusive",
    }
    assert h["type"] in ("internal", "external")
    assert h["verdict"] in ("confirmed", "rejected", "inconclusive")


def test_insight_state_structure():
    state: InsightState = {
        "question": "Why did sales drop?",
        "industry": "retail",
        "context": "US, 2023",
        "max_cycles": 5,
        "source_file": "data/sales.csv",
        "source_type": "csv",
        "db_path": "/tmp/test.duckdb",
        "schema": {},
        "query_context": "initial",
        "current_query_goal": "Analyse overall sales trend",
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
    assert state["max_cycles"] == 5
    assert state["query_context"] == "initial"
