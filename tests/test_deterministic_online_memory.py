from __future__ import annotations

import pandas as pd
import pytest

from agent_benchmark.deterministic_memory import DeterministicMarketMemory
from agent_benchmark.schemas import BenchmarkConfig
from agent_benchmark.warehouse.store import Warehouse


def test_deterministic_cases_label_the_actual_next_open_holding_window(tmp_path):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    dates = pd.bdate_range("2024-01-02", periods=30).date.astype(str).tolist()
    try:
        for index, day in enumerate(dates):
            open_price = 100.0
            if index == 22:
                open_price = 110.0
            warehouse.conn.execute(
                """
                INSERT OR REPLACE INTO asset_daily
                    (date, symbol, market_open, listed, ohlcv_available, open, high, low,
                     close, adj_close, volume, return_1d, source, source_error)
                VALUES (CAST(? AS DATE), 'AAPL', true, true, true, ?, 111, 99, 100, 100,
                        1000, 0, 'test', '')
                """,
                [day, open_price],
            )

        config = BenchmarkConfig(
            symbol="AAPL",
            train_start=dates[20],
            train_end=dates[-1],
            historical_price_basis="adjusted",
        )
        memory = DeterministicMarketMemory(warehouse)
        cases = memory._load_cases(config, ["AAPL"])
        case = cases[cases["date"] == dates[20]].iloc[0]
        item = memory._row_to_memory(case, 1.0)
        lessons = memory.online_lessons(config, ["AAPL"], horizon_days=1)
    finally:
        warehouse.close()

    assert case["fill_date"] == dates[21]
    assert case["outcome_date_1d"] == dates[22]
    assert case["outcome_1d"] == pytest.approx(0.10)
    assert item["metadata"]["execution_timing"] == "decision_close_next_adjusted_open_to_open"
    assert item["metadata"]["fill_date"] == dates[21]
    lesson = next(value for value in lessons if value.snapshot.decision_timestamp[:10] == dates[20])
    assert lesson.entry_timestamp[:10] == dates[21]
    assert lesson.outcome_timestamp[:10] == dates[22]
    assert lesson.outcomes.long.gross_return == pytest.approx(0.10)


def test_adjusted_feature_state_is_not_corrupted_by_a_split(tmp_path):
    warehouse = Warehouse(tmp_path / "split.duckdb")
    dates = pd.bdate_range("2020-07-01", periods=35).date.astype(str).tolist()
    try:
        for index, day in enumerate(dates):
            raw = 100.0 if index < 15 else 25.0
            adjusted = 25.0
            warehouse.conn.execute(
                """
                INSERT INTO asset_daily VALUES
                    (CAST(? AS DATE), 'AAPL', true, true, true, ?, ?, ?, ?, ?, 1000, 0, 'test', '')
                """,
                [day, raw, raw, raw, raw, adjusted],
            )
        config = BenchmarkConfig(
            symbol="AAPL",
            train_start=dates[20],
            train_end=dates[-1],
            historical_price_basis="adjusted",
        )
        cases = DeterministicMarketMemory(warehouse)._load_cases(config, ["AAPL"])
        row = cases[cases["date"] == dates[20]].iloc[0]
    finally:
        warehouse.close()

    assert row["return_20d"] == pytest.approx(0.0)
    assert row["volatility_20d"] == pytest.approx(0.0)


def test_prompt_aggregate_does_not_count_overlapping_forward_windows():
    rows = pd.DataFrame(
        [
            {"fill_date": "2024-01-02", "outcome_date_20d": "2024-01-30", "_distance": 0.1},
            {"fill_date": "2024-01-10", "outcome_date_20d": "2024-02-07", "_distance": 0.2},
            {"fill_date": "2024-02-08", "outcome_date_20d": "2024-03-07", "_distance": 0.3},
        ]
    )

    selected = DeterministicMarketMemory._non_overlapping_rows(rows, 10, horizon="20d")

    assert selected.index.tolist() == [0, 2]
