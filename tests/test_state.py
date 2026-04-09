from insight_swarm.state import InsightState, Hypothesis, LogEntry


def test_hypothesis_has_evidence_field():
    h: Hypothesis = {
        "text": "Sales dropped in West",
        "type": "internal",
        "data_slice": None,
        "web_context": None,
        "verdict": "inconclusive",
        "evidence": "",
    }
    assert h["evidence"] == ""


def test_insight_state_has_data_summary():
    from tests.conftest import make_state
    state = make_state()
    assert "data_summary" in state
    assert "current_hypothesis" in state
    assert "evaluator_rounds" in state
    assert "data_request" in state
    assert "current_data_slice" in state
    assert "run_id" in state


def test_insight_state_missing_sql_fields():
    from tests.conftest import make_state
    state = make_state()
    assert "sql_error" not in state
    assert "sql_retry_count" not in state
    assert "current_sql" not in state
    assert "db_path" not in state
