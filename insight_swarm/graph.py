from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from insight_swarm.state import InsightState
from insight_swarm.nodes.schema import schema_node
from insight_swarm.nodes.sql_writer import sql_writer_node
from insight_swarm.nodes.sql_retry import sql_retry_node
from insight_swarm.nodes.executor import executor_node
from insight_swarm.nodes.analyst import analyst_node
from insight_swarm.nodes.hypothesis import hypothesis_node
from insight_swarm.nodes.web_search import web_search_node
from insight_swarm.nodes.web_analyst import web_analyst_node
from insight_swarm.nodes.narrator import narrator_node
from insight_swarm.nodes.report import report_node

SUFFICIENT_ROW_THRESHOLD = 10


# ── Routing functions ────────────────────────────────────────────────────────

def route_after_executor(state: InsightState) -> str:
    if state.get("sql_error"):
        if state.get("sql_retry_count", 0) < 2:
            return "retry"
        if state.get("query_context", "initial") == "hypothesis":
            return "failed_hypothesis"
        return "failed_initial"

    latest = state["sql_results"][-1] if state["sql_results"] else []
    if len(latest) < SUFFICIENT_ROW_THRESHOLD:
        return "web_search"
    return "analyst"


def route_after_hypothesis(state: InsightState) -> str:
    if state["cycle_count"] >= state["max_cycles"]:
        return "force_exit"
    idx = state["current_hypothesis_idx"]
    hypotheses = state["hypotheses"]
    if idx >= len(hypotheses):
        return "done"
    current = hypotheses[idx]
    if current["type"] == "external":
        return "external"
    return "internal"


def route_after_web_analyst(state: InsightState) -> str:
    idx = state["current_hypothesis_idx"]
    if idx + 1 < len(state["hypotheses"]):
        return "more"
    return "done"


# ── Graph assembly ───────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(InsightState)

    g.add_node("schema", schema_node)
    g.add_node("sql_writer", sql_writer_node)
    g.add_node("sql_retry", sql_retry_node)
    g.add_node("executor", executor_node)
    g.add_node("analyst", analyst_node)
    g.add_node("hypothesis", hypothesis_node)
    g.add_node("web_search", web_search_node)
    g.add_node("web_analyst", web_analyst_node)
    g.add_node("narrator", narrator_node)
    g.add_node("report", report_node)

    # Linear start
    g.add_edge(START, "schema")
    g.add_edge("schema", "sql_writer")
    g.add_edge("sql_writer", "executor")

    # After executor: retry / fail / web_search / analyst
    g.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "retry": "sql_retry",
            "failed_initial": "analyst",
            "failed_hypothesis": "hypothesis",
            "web_search": "web_search",
            "analyst": "analyst",
        },
    )
    g.add_edge("sql_retry", "executor")

    # After analyst → hypothesis
    g.add_edge("analyst", "hypothesis")

    # After hypothesis: branch by type or exit
    g.add_conditional_edges(
        "hypothesis",
        route_after_hypothesis,
        {
            "internal": "sql_writer",
            "external": "web_search",
            "done": "narrator",
            "force_exit": "narrator",
        },
    )

    # Web path
    g.add_edge("web_search", "web_analyst")
    g.add_conditional_edges(
        "web_analyst",
        route_after_web_analyst,
        {
            "more": "hypothesis",
            "done": "narrator",
        },
    )

    # End
    g.add_edge("narrator", "report")
    g.add_edge("report", END)

    return g.compile()


graph = build_graph()
