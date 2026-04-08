import duckdb
import pandas as pd
import pytest
from pathlib import Path
from insight_swarm.nodes.schema import schema_node


@pytest.fixture
def minimal_state(tmp_path):
    return {
        "question": "Why did sales drop?",
        "industry": None,
        "context": None,
        "max_cycles": 5,
        "source_file": "",
        "source_type": "",
        "db_path": "",
        "schema": {},
        "query_context": "initial",
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


def test_schema_node_loads_csv(tmp_path, minimal_state):
    csv_file = tmp_path / "sales.csv"
    csv_file.write_text("product,sales,quarter\nWidget,1000,Q1\nGadget,2000,Q2\n")

    state = {**minimal_state, "source_file": str(csv_file), "source_type": "csv"}
    result = schema_node(state)

    assert "schema" in result
    assert "db_path" in result
    tables = result["schema"]
    assert len(tables) >= 1
    first_table = list(tables.values())[0]
    col_names = [c["column"] for c in first_table]
    assert "product" in col_names
    assert "sales" in col_names


def test_schema_node_loads_xlsx(tmp_path, minimal_state):
    xlsx_file = tmp_path / "data.xlsx"
    df = pd.DataFrame({"revenue": [100, 200], "region": ["North", "South"]})
    df.to_excel(xlsx_file, sheet_name="Sales", index=False)

    state = {**minimal_state, "source_file": str(xlsx_file), "source_type": "xlsx"}
    result = schema_node(state)

    assert "Sales" in result["schema"]
    col_names = [c["column"] for c in result["schema"]["Sales"]]
    assert "revenue" in col_names


def test_schema_node_records_row_counts(tmp_path, minimal_state):
    csv_file = tmp_path / "orders.csv"
    csv_file.write_text("id,amount\n1,50\n2,75\n3,100\n")
    state = {**minimal_state, "source_file": str(csv_file), "source_type": "csv"}
    result = schema_node(state)

    table_info = list(result["schema"].values())[0]
    assert any(col.get("row_count") == 3 for col in table_info if "row_count" in col)


def test_schema_node_appends_log(tmp_path, minimal_state):
    csv_file = tmp_path / "x.csv"
    csv_file.write_text("a,b\n1,2\n")
    state = {**minimal_state, "source_file": str(csv_file), "source_type": "csv"}
    result = schema_node(state)
    assert len(result["run_log"]) == 1
    assert result["run_log"][0]["node"] == "schema"
