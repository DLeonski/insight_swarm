import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_cli_rejects_missing_file_and_db(capsys):
    with pytest.raises(SystemExit):
        with patch("sys.argv", ["insight_swarm", "--question", "Why?"]):
            import importlib
            import insight_swarm.__main__ as m
            importlib.reload(m)
            m.main()


def test_cli_rejects_unsupported_extension(tmp_path):
    bad_file = tmp_path / "data.parquet"
    bad_file.write_text("x")
    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", [
            "insight_swarm", "--file", str(bad_file), "--question", "Why?"
        ]):
            import importlib
            import insight_swarm.__main__ as m
            importlib.reload(m)
            m.main()
    assert exc_info.value.code != 0


def test_cli_builds_initial_state(tmp_path):
    csv_file = tmp_path / "sales.csv"
    csv_file.write_text("a,b\n1,2\n")
    captured_state = {}

    def fake_invoke(state, config=None):
        captured_state.update(state)
        return state

    # Mock graph before importing __main__
    with patch("sys.argv", [
        "insight_swarm",
        "--file", str(csv_file),
        "--question", "Why did sales drop?",
        "--industry", "retail",
        "--context", "US 2023",
        "--max-cycles", "3",
    ]):
        import sys
        import importlib

        # Create a mock graph module
        mock_graph_obj = MagicMock()
        mock_graph_obj.invoke.side_effect = fake_invoke

        # Patch sys.modules to inject our mock
        with patch.dict("sys.modules", {"insight_swarm.graph": MagicMock(graph=mock_graph_obj)}):
            # Remove cached __main__ if it exists
            if "insight_swarm.__main__" in sys.modules:
                del sys.modules["insight_swarm.__main__"]

            import insight_swarm.__main__ as m
            m.graph = mock_graph_obj
            m.main()

    assert captured_state["question"] == "Why did sales drop?"
    assert captured_state["industry"] == "retail"
    assert captured_state["max_cycles"] == 3
    assert captured_state["source_type"] == "csv"
