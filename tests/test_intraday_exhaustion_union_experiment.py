from __future__ import annotations

import hashlib
import math

import pandas as pd
import pytest

from agent_benchmark.intraday_exhaustion_union_experiment import (
    _decision_dates_sha256,
    _episode_rows,
    apply_development_gates,
    simulate_intraday_only,
)


def _frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2004-12-31",
            "2005-01-03",
            "2005-01-04",
            "2005-01-05",
            "2005-01-06",
        ]
    )
    return pd.DataFrame(
        {
            "aapl_open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "aapl_close": [100.0, 100.0, 90.0, 100.0, 100.0],
            "aapl_adj_close": [100.0, 100.0, 90.0, 100.0, 100.0],
            "spy_adj_close": [100.0] * 5,
            "qqq_adj_close": [100.0] * 5,
        },
        index=dates,
    )


def _signal(frame: pd.DataFrame) -> pd.Series:
    signal = pd.Series(False, index=frame.index, dtype=bool)
    signal.loc[pd.Timestamp("2005-01-03")] = True
    return signal


def test_intraday_ledger_sells_at_open_and_buys_unconditionally_at_close():
    frame = _frame()
    ledger, events = simulate_intraday_only(
        frame,
        _signal(frame),
        cost_bps=10.0,
        end=pd.Timestamp("2005-01-06"),
    )

    cash_day = ledger.loc[ledger["fill_date"] == "2005-01-04"].iloc[0]
    assert cash_day["target_exposure"] == 0.0
    assert cash_day["new_exposure_after_fill"] == 0.0
    assert cash_day["shares"] == 0.0
    assert cash_day["cash"] > 0.0

    same_day = events.loc[events["event_date"] == "2005-01-04"]
    assert same_day["event_time"].tolist() == ["open", "close"]
    assert same_day["side"].tolist() == ["sell", "buy"]
    assert same_day.iloc[1]["cash_after"] == 0.0
    assert same_day.iloc[1]["shares_after"] > 0.0
    assert ledger["cash"].min() >= 0.0
    assert ledger["shares"].min() >= 0.0
    assert ledger["new_exposure_after_fill"].between(0.0, 1.0).all()


def test_intraday_cost_and_edge_formulas_reconcile():
    frame = _frame()
    cost = 0.001
    ledger, events = simulate_intraday_only(
        frame,
        _signal(frame),
        cost_bps=10.0,
        end=pd.Timestamp("2005-01-06"),
    )
    initial_shares = 1000.0 / (100.0 * (1.0 + cost))
    sale_cash = initial_shares * 100.0 * (1.0 - cost)
    expected_close_shares = sale_cash / (90.0 * (1.0 + cost))
    close_buy = events.loc[
        (events["event_date"] == "2005-01-04")
        & (events["event_time"] == "close")
    ].iloc[0]
    assert close_buy["shares_after"] == pytest.approx(expected_close_shares)
    next_open = ledger.loc[ledger["fill_date"] == "2005-01-05"].iloc[0]
    assert next_open["equity"] == pytest.approx(expected_close_shares * 100.0)

    episodes = _episode_rows(
        frame, _signal(frame), cost_bps=10.0, mode="intraday"
    )
    expected_edge = math.log(100.0 / 90.0) + math.log(
        (1.0 - cost) / (1.0 + cost)
    )
    assert episodes.iloc[0]["net_active_log_edge"] == pytest.approx(expected_edge)


def test_open_to_open_control_uses_following_open_not_same_day_close():
    frame = _frame()
    intraday = _episode_rows(
        frame, _signal(frame), cost_bps=5.0, mode="intraday"
    )
    control = _episode_rows(
        frame, _signal(frame), cost_bps=5.0, mode="open_to_open"
    )
    assert intraday.iloc[0]["buyback_date"] == "2005-01-04"
    assert control.iloc[0]["buyback_date"] == "2005-01-05"
    assert intraday.iloc[0]["raw_active_log_edge"] == pytest.approx(
        math.log(100.0 / 90.0)
    )
    assert control.iloc[0]["raw_active_log_edge"] == pytest.approx(0.0)


def test_decision_date_identity_has_a_frozen_unambiguous_encoding():
    index = pd.to_datetime(["2005-01-03", "2005-01-04", "2005-01-05"])
    signal = pd.Series([True, False, True], index=index, dtype=bool)
    expected = hashlib.sha256(b"2005-01-03\n2005-01-05\n").hexdigest()
    assert _decision_dates_sha256(signal) == expected


def _passing_metrics() -> dict:
    policy = {
        "total_active_log_edge": 1.0,
        "relative_ending_wealth": 1.5,
        "positive_year_count": 10,
        "positive_fold_count": 6,
        "edge_after_best_year_removed": 0.5,
        "maximum_positive_year_share": 0.2,
        "cash_episode_count": 121,
        "cash_episode_win_rate": 0.60,
        "mean_cash_episode_edge": 0.01,
        "median_cash_episode_edge": 0.005,
        "maximum_positive_episode_share": 0.1,
        "negative_aapl_year_aggregate_edge": 0.3,
        "negative_aapl_year_positive_count": 3,
        "no_leverage_proof": {"passed": True},
    }
    incremental = {
        "total_incremental_active_log_edge": 0.1,
        "positive_incremental_fold_count": 5,
        "incremental_after_best_fold_removed": 0.04,
    }
    return {
        "base_5bps": {
            "candidate": dict(policy),
            "candidate_vs_union": dict(incremental),
        },
        "stress_10bps": {
            "candidate": dict(policy),
            "candidate_vs_union": dict(incremental),
        },
        "integrity": {
            "candidate_and_union_signal_dates_identical": True,
            "physical_later_rows_opened": False,
            "network_calls": 0,
            "api_calls": 0,
            "llm_calls": 0,
            "broker_actions": 0,
            "real_money_actions": 0,
        },
    }


def test_gates_require_trading_improvement_not_runtime_bookkeeping():
    metrics = _passing_metrics()
    assert apply_development_gates(metrics)["passed"] is True
    metrics["stress_10bps"]["candidate_vs_union"][
        "total_incremental_active_log_edge"
    ] = 0.0
    report = apply_development_gates(metrics)
    assert report["passed"] is False
    assert "candidate_beats_union_by_0001_10bps" in report["failures"]

