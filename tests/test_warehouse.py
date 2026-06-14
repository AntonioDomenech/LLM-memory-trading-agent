from pathlib import Path

from agent_benchmark.warehouse.store import Warehouse
from agent_benchmark.warehouse.universe import CONTEXT_SYMBOLS, STOCK_SYMBOLS
from agent_benchmark.warehouse.validate import validate_warehouse


def test_balanced_universe_counts():
    assert len(STOCK_SYMBOLS) == 50
    assert "AAPL" in STOCK_SYMBOLS
    assert "^VIX" in CONTEXT_SYMBOLS


def test_warehouse_bootstrap_calendar_and_symbols(tmp_path):
    wh = Warehouse(tmp_path / "warehouse.duckdb")
    try:
        result = wh.bootstrap("2000-01-01", "2000-01-10")
        assert result["calendar_rows"] == 10
        assert result["symbols"] == len(STOCK_SYMBOLS) + len(CONTEXT_SYMBOLS)
        status = wh.status()
        assert status["tables"]["calendar_daily"] == 10
        assert status["tables"]["symbols"] == 69
    finally:
        wh.close()


def test_validate_empty_warehouse_reports_symbols(tmp_path):
    wh = Warehouse(tmp_path / "warehouse.duckdb")
    try:
        wh.bootstrap("2000-01-01", "2000-01-03")
        result = validate_warehouse(wh)
        assert result["calendar_rows"] == 3
        assert result["symbols"] == 69
    finally:
        wh.close()
