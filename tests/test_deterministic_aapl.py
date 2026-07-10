from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.deterministic_aapl import (
    CostAssumptions,
    EvaluationPeriod,
    StrategySpec,
    build_target_exposure,
    canonical_market_frame,
    compare_ledgers,
    historical_selection_audit,
    load_or_download_market_frame,
    market_data_sha256,
    performance_metrics,
    simulate_period,
    terminal_close_sensitivity,
)


def market_frame(periods: int = 260, start: str = "2023-01-02") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=periods)
    aapl_close = np.linspace(100.0, 200.0, periods)
    spy_close = np.linspace(300.0, 450.0, periods)
    return pd.DataFrame(
        {
            "aapl_open": aapl_close * 0.99,
            "aapl_close": aapl_close,
            "aapl_adj_close": aapl_close,
            "spy_adj_close": spy_close,
        },
        index=index,
    )


def test_canonical_frame_builds_adjusted_open_and_rejects_duplicates():
    frame = market_frame(5)
    canonical = canonical_market_frame(frame)
    assert canonical.iloc[0]["aapl_adj_open"] == pytest.approx(frame.iloc[0]["aapl_open"])
    duplicated = pd.concat([frame, frame.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        canonical_market_frame(duplicated)


def test_strategy_rejects_appl_typo():
    with pytest.raises(ValueError, match="AAPL only"):
        StrategySpec(symbol="APPL").validate()


def test_trend_regime_target_uses_complete_lookbacks():
    frame = market_frame(260)
    spec = StrategySpec(aapl_sma_days=150, spy_sma_days=200)
    target = build_target_exposure(frame, spec)
    assert target.iloc[:199].isna().all()
    assert target.iloc[200] == pytest.approx(1.10)


def test_signal_is_shifted_one_session_before_next_open_fill():
    frame = market_frame(6)
    target = pd.Series([0.0, 1.0, 0.0, 1.0, 1.0, 1.0], index=frame.index)
    period = EvaluationPeriod(
        "toy",
        frame.index[1].date().isoformat(),
        frame.index[-1].date().isoformat(),
    )
    ledger = simulate_period(
        frame,
        target,
        period,
        CostAssumptions(slippage_bps=0.0, annual_margin_rate=0.0),
        max_exposure=1.0,
    )
    assert ledger.loc[0, "decision_date"] == frame.index[0].date().isoformat()
    assert ledger.loc[0, "fill_date"] == frame.index[1].date().isoformat()
    assert ledger.loc[0, "target_exposure"] == 0.0
    assert ledger.loc[1, "target_exposure"] == 1.0


def test_buy_hold_uses_exact_entry_slippage_without_unintended_margin():
    frame = market_frame(5)
    target = pd.Series(1.0, index=frame.index)
    period = EvaluationPeriod(
        "toy",
        frame.index[1].date().isoformat(),
        frame.index[-1].date().isoformat(),
    )
    costs = CostAssumptions(slippage_bps=5.0, annual_margin_rate=0.08)
    ledger = simulate_period(frame, target, period, costs, max_exposure=1.0)
    first_price = canonical_market_frame(frame).loc[frame.index[1], "aapl_adj_open"]
    expected_shares = 1000.0 / (first_price * 1.0005)
    assert ledger.loc[0, "shares"] == pytest.approx(expected_shares)
    assert ledger.loc[0, "cash"] == pytest.approx(0.0, abs=1e-9)
    assert ledger["margin_interest"].sum() == 0.0
    assert ledger["new_exposure_after_fill"].max() <= 1.0 + 1e-9


def test_declared_margin_target_has_exact_exposure_and_financing():
    frame = market_frame(8)
    target = pd.Series(1.10, index=frame.index)
    period = EvaluationPeriod(
        "toy",
        frame.index[1].date().isoformat(),
        frame.index[-1].date().isoformat(),
    )
    ledger = simulate_period(
        frame,
        target,
        period,
        CostAssumptions(slippage_bps=5.0, annual_margin_rate=0.08),
        max_exposure=1.10,
    )
    assert ledger.loc[0, "cash"] < 0
    assert ledger.loc[0, "new_exposure_after_fill"] == pytest.approx(1.10, abs=1e-9)
    assert ledger.loc[1:, "margin_interest"].sum() > 0
    assert ledger["new_exposure_after_fill"].max() <= 1.10 + 1e-8


def test_identical_strategy_and_benchmark_never_pass_from_rounding():
    frame = market_frame(8)
    target = pd.Series(1.0, index=frame.index)
    period = EvaluationPeriod(
        "toy",
        frame.index[1].date().isoformat(),
        frame.index[-1].date().isoformat(),
    )
    ledger = simulate_period(frame, target, period, CostAssumptions(), max_exposure=1.0)
    comparison = compare_ledgers(ledger, ledger.copy())
    assert comparison["excess_return_vs_aapl_buy_hold"] == 0.0
    assert comparison["requested_success"] is False


def test_performance_metrics_compound_months_back_to_total_return():
    frame = market_frame(50, start="2024-01-02")
    target = pd.Series(1.0, index=frame.index)
    period = EvaluationPeriod(
        "toy",
        frame.index[1].date().isoformat(),
        frame.index[-1].date().isoformat(),
    )
    ledger = simulate_period(
        frame,
        target,
        period,
        CostAssumptions(slippage_bps=0.0, annual_margin_rate=0.0),
        max_exposure=1.0,
    )
    metrics = performance_metrics(ledger)
    compounded = math.prod(1.0 + value for value in metrics["monthly_returns"].values()) - 1.0
    assert compounded == pytest.approx(metrics["total_return"])
    assert metrics["best_daily_return"] >= metrics["worst_daily_return"]
    assert metrics["maximum_exposure"] <= 1.0 + 1e-9


def test_market_hash_is_stable_to_column_input_order():
    frame = market_frame(10)
    reordered = frame[["spy_adj_close", "aapl_adj_close", "aapl_close", "aapl_open"]]
    assert market_data_sha256(frame) == market_data_sha256(reordered)


def test_historical_selection_audit_rejects_final_period_data():
    frame = market_frame(6000, start="2000-01-03")
    with pytest.raises(ValueError, match="end before"):
        historical_selection_audit(
            frame,
            StrategySpec(),
            CostAssumptions(),
            first_year=2020,
            last_year=2024,
        )


def test_cache_accepts_weekend_start_and_honors_requested_end(tmp_path, monkeypatch):
    frame = market_frame(10, start="1999-01-04")
    cache = tmp_path / "market.csv"
    frame.reset_index(names="date").to_csv(cache, index=False, float_format="%.17g")

    def unexpected_download(*args, **kwargs):
        raise AssertionError("a complete cache must not be refreshed")

    monkeypatch.setattr(
        "agent_benchmark.deterministic_aapl.download_market_frame",
        unexpected_download,
    )
    loaded, origin = load_or_download_market_frame(
        cache,
        start="1999-01-01",
        end_inclusive=frame.index[4].date().isoformat(),
    )
    assert origin == "cache"
    assert loaded.index.min() == frame.index.min()
    assert loaded.index.max() == frame.index[4]
    assert len(loaded) == 5


def test_terminal_close_sensitivity_captures_last_session_move():
    frame = market_frame(5)
    frame.loc[frame.index[-1], "aapl_close"] = frame.loc[frame.index[-1], "aapl_open"] * 2.0
    frame.loc[frame.index[-1], "aapl_adj_close"] = frame.loc[frame.index[-1], "aapl_close"]
    target = pd.Series(1.0, index=frame.index)
    period = EvaluationPeriod(
        "toy",
        frame.index[1].date().isoformat(),
        frame.index[-1].date().isoformat(),
    )
    ledger = simulate_period(
        frame,
        target,
        period,
        CostAssumptions(slippage_bps=0.0, annual_margin_rate=0.0),
        max_exposure=1.0,
    )
    sensitivity = terminal_close_sensitivity(ledger, frame)
    assert sensitivity["terminal_total_return"] > performance_metrics(ledger)["total_return"]
    assert sensitivity["last_session_open_to_close_return"] == pytest.approx(1.0)


def test_metrics_report_pre_rebalance_leverage_spike():
    frame = market_frame(5)
    # A leveraged long position becomes more leveraged after a large opening gap down.
    frame.loc[frame.index[2]:, "aapl_open"] *= 0.5
    frame.loc[frame.index[2]:, "aapl_close"] *= 0.5
    frame.loc[frame.index[2]:, "aapl_adj_close"] *= 0.5
    target = pd.Series(1.10, index=frame.index)
    period = EvaluationPeriod(
        "toy",
        frame.index[1].date().isoformat(),
        frame.index[-1].date().isoformat(),
    )
    ledger = simulate_period(
        frame,
        target,
        period,
        CostAssumptions(slippage_bps=0.0, annual_margin_rate=0.0),
        max_exposure=1.10,
    )
    metrics = performance_metrics(ledger)
    assert metrics["maximum_post_fill_exposure"] == pytest.approx(1.10)
    assert metrics["maximum_holding_exposure_for_return"] > 1.10
    assert metrics["maximum_exposure"] == metrics["maximum_holding_exposure_for_return"]


def test_market_snapshot_17_digit_round_trip_preserves_hash(tmp_path):
    frame = canonical_market_frame(market_frame(10))
    path = tmp_path / "snapshot.csv"
    frame.reset_index(names="date").to_csv(path, index=False, float_format="%.17g")
    assert market_data_sha256(frame) == market_data_sha256(pd.read_csv(path))
