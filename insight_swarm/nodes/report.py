from __future__ import annotations
import json
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import plotly.express as px
from jinja2 import Environment, FileSystemLoader
from insight_swarm.state import InsightState

_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
LOGS_DIR = Path(__file__).parent.parent / "logs"


def _make_chart(hypothesis: dict) -> str | None:
    records = hypothesis.get("sql_result")
    if not records or hypothesis["verdict"] not in ("confirmed", "rejected"):
        return None
    df = pd.DataFrame(records)
    numeric_cols = df.select_dtypes("number").columns.tolist()
    if not numeric_cols:
        return None
    y_col = numeric_cols[0]
    x_col = [c for c in df.columns if c != y_col][0] if len(df.columns) > 1 else df.columns[0]
    fig = px.bar(df, x=x_col, y=y_col, title=hypothesis["text"][:60],
                 template="plotly_dark")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def report_node(state: InsightState) -> dict:
    start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Enrich hypotheses with chart HTML
    hypotheses_ctx = []
    for h in state["hypotheses"]:
        hypotheses_ctx.append({**h, "chart_html": _make_chart(h)})

    # Collect SQL strings from run_log
    sql_queries = [
        entry["detail"]["sql"]
        for entry in state["run_log"]
        if entry["node"] in ("sql_writer", "executor") and "sql" in entry.get("detail", {})
    ]

    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))
    template = env.get_template("report.html.j2")
    html = template.render(
        question=state["question"],
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        source_file=state.get("source_file", ""),
        narrative=state["narrative"],
        hypotheses=hypotheses_ctx,
        sql_queries=sql_queries,
    )

    html_path = REPORTS_DIR / f"report_{timestamp}.html"
    html_path.write_text(html, encoding="utf-8")

    # JSON run log
    total_duration = int((time.time() - start) * 1000)
    sql_retries = sum(1 for e in state["run_log"] if e["node"] == "sql_retry")
    log_data = {
        "question": state["question"],
        "industry": state.get("industry"),
        "context": state.get("context"),
        "source_file": state.get("source_file"),
        "total_duration_ms": total_duration,
        "hypotheses_tested": len(state["hypotheses"]),
        "sql_retries_total": sql_retries,
        "cycle_count": state["cycle_count"],
        "steps": state["run_log"],
    }
    json_path = LOGS_DIR / f"run_{timestamp}.json"
    json_path.write_text(json.dumps(log_data, indent=2, default=str), encoding="utf-8")

    print(f"Report saved: {html_path}")
    print(f"Log saved: {json_path}")
    print("Opening in browser...")
    webbrowser.open(str(html_path))

    return {}
