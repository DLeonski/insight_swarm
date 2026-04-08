import json
from pathlib import Path


def make_state(tmp_path, **overrides):
    base = {
        "question": "Why did sales drop in Q3?",
        "industry": "retail",
        "context": "US market, 2023",
        "max_cycles": 5,
        "source_file": "data/superstore.csv",
        "source_type": "csv",
        "db_path": str(tmp_path / "test.duckdb"),
        "schema": {},
        "query_context": "hypothesis",
        "current_query_goal": "",
        "current_sql": None,
        "sql_error": None,
        "sql_retry_count": 0,
        "sql_results": [[{"quarter": "Q3", "sales": 800}]],
        "analyses": ["Q3 was down 20%."],
        "hypotheses": [
            {
                "text": "Seasonal effect",
                "type": "internal",
                "sql_result": [{"quarter": "Q3", "sales": 800}, {"quarter": "Q2", "sales": 1200}],
                "web_context": None,
                "verdict": "confirmed",
            },
            {
                "text": "Competitor actions",
                "type": "external",
                "sql_result": None,
                "web_context": "No evidence.",
                "verdict": "rejected",
            },
        ],
        "current_hypothesis_idx": 1,
        "cycle_count": 2,
        "narrative": "Sales declined in Q3 primarily due to seasonal effects.",
        "run_log": [],
    }
    return {**base, **overrides}


def test_report_creates_html_file(tmp_path, monkeypatch):
    import insight_swarm.nodes.report as report_module
    monkeypatch.setattr(report_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(report_module, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(report_module, "webbrowser", type("W", (), {"open": staticmethod(lambda *a, **k: None)})())

    from insight_swarm.nodes.report import report_node
    state = make_state(tmp_path)
    report_node(state)

    html_files = list((tmp_path / "reports").glob("*.html"))
    assert len(html_files) == 1
    content = html_files[0].read_text()
    assert "Seasonal effect" in content
    assert "confirmed" in content.lower()


def test_report_creates_json_log(tmp_path, monkeypatch):
    import insight_swarm.nodes.report as report_module
    monkeypatch.setattr(report_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(report_module, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(report_module, "webbrowser", type("W", (), {"open": staticmethod(lambda *a, **k: None)})())

    from insight_swarm.nodes.report import report_node
    state = make_state(tmp_path)
    report_node(state)

    json_files = list((tmp_path / "logs").glob("*.json"))
    assert len(json_files) == 1
    data = json.loads(json_files[0].read_text())
    assert "question" in data
    assert "hypotheses_tested" in data
