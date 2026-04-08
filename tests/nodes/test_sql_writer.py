# tests/nodes/test_sql_writer.py
from unittest.mock import MagicMock, patch
from tests.conftest import make_state


def test_sql_writer_sets_current_sql():
    state = make_state(query_context="initial")
    with patch("insight_swarm.nodes.sql_writer.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = (
            "SELECT quarter, SUM(sales) FROM orders GROUP BY quarter"
        )
        from insight_swarm.nodes.sql_writer import sql_writer_node
        result = sql_writer_node(state)

    assert "current_sql" in result
    assert "SELECT" in result["current_sql"].upper()


def test_sql_writer_includes_error_in_prompt_on_retry():
    state = make_state(
        query_context="hypothesis",
        sql_error="Parser Error: syntax error at or near 'FORM'",
        current_sql="SELECT * FORM orders",
    )
    captured_prompts = []

    def fake_invoke(prompt):
        captured_prompts.append(prompt)
        return "SELECT * FROM orders LIMIT 100"

    with patch("insight_swarm.nodes.sql_writer.Ollama") as MockOllama:
        MockOllama.return_value.invoke.side_effect = fake_invoke
        from insight_swarm.nodes.sql_writer import sql_writer_node
        sql_writer_node(state)

    assert "Parser Error" in captured_prompts[0]


def test_sql_writer_appends_log():
    state = make_state()
    with patch("insight_swarm.nodes.sql_writer.Ollama") as MockOllama:
        MockOllama.return_value.invoke.return_value = "SELECT 1"
        from insight_swarm.nodes.sql_writer import sql_writer_node
        result = sql_writer_node(state)

    assert len(result["run_log"]) == len(state["run_log"]) + 1
    assert result["run_log"][-1]["node"] == "sql_writer"
