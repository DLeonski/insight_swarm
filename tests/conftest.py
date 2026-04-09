import pytest
from insight_swarm.state import InsightState


def make_state(**overrides) -> InsightState:
    base: InsightState = {
        "run_id": "test-run-001",
        "question": "Why did sales drop in Q3?",
        "industry": "retail",
        "context": "US market, 2023",
        "max_cycles": 5,
        "source_file": "insight_swarm/data/superstore.csv",
        "source_type": "csv",
        "schema": {
            "Order ID": "object",
            "Sales": "float64",
            "Profit": "float64",
            "Region": "object",
            "Category": "object",
            "Order Date": "object",
        },
        "data_summary": [
            {"metric": "Overview", "value": "500 rows, 7 columns"},
            {"metric": "Sales by year", "value": "2022: 120000 | 2023: 98000"},
        ],
        "analyses": [],
        "hypotheses": [],
        "current_hypothesis": None,
        "evaluator_rounds": 0,
        "cycle_count": 0,
        "data_request": None,
        "current_data_slice": [],
        "narrative": "",
        "run_log": [],
    }
    return {**base, **overrides}


@pytest.fixture(name="make_state")
def make_state_fixture():
    return make_state
