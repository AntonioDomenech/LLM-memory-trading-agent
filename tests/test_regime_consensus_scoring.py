from __future__ import annotations

import copy
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.deterministic_aapl import CostAssumptions, EvaluationPeriod
from agent_benchmark.regime_consensus_scoring import (
    REGIME_CONSENSUS_GATES,
    RegimeConsensusScoringError,
    apply_regime_consensus_gates,
    evaluate_regime_consensus_ablation,
    evaluate_regime_consensus_development,
    score_regime_consensus_metrics,
)
from agent_benchmark.unleveraged_aapl import simulate_unleveraged_period


NEGATIVE_BUY_HOLD_YEARS = {2008, 2011, 2015, 2018}
DEVELOPMENT_PERIOD = EvaluationPeriod(
    "development", "2005-01-01", "2018-12-31"
)


def _market_and_target() -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.bdate_range("2004-12-31", "2018-12-31")
    fill_dates = dates[
        (dates >= pd.Timestamp(DEVELOPMENT_PERIOD.start))
        & (dates <= pd.Timestamp(DEVELOPMENT_PERIOD.end))
    ]
    executed_target = np.ones(len(fill_dates), dtype=float)
    reentry_positions: list[int] = []
    for year in range(2005, 2019):
        positions = np.flatnonzero(fill_dates.year == year)
        for offset in (60, 140):
            start = int(positions[offset])
            executed_target[start : start + 2] = 0.0
            reentry_positions.append(start + 2)

    # Each CASH episode spans a deliberately adverse two-session AAPL move.
    # Positive years still rise overall; four years are genuinely negative.
    log_return = np.asarray(
        [0.0001 if date.year in NEGATIVE_BUY_HOLD_YEARS else 0.0007 for date in dates],
        dtype=float,
    )
    for position in reentry_positions:
        log_return[dates.get_loc(fill_dates[position])] -= 0.04
    price = 100.0 * np.exp(np.cumsum(log_return))
    frame = pd.DataFrame(
        {
            "aapl_open": price,
            "aapl_close": price,
            "aapl_adj_close": price,
            "spy_adj_close": 100.0 * np.exp(np.arange(len(dates)) * 0.0002),
        },
        index=dates,
    )

    target_at_close = pd.Series(1.0, index=dates, name="target_exposure")
    for fill_date, target in zip(fill_dates, executed_target, strict=True):
        target_at_close.iloc[dates.get_loc(fill_date) - 1] = target
    return frame, target_at_close


@lru_cache(maxsize=2)
def _cached_inputs(cost_bps: float) -> tuple[object, ...]:
    frame, target_at_close = _market_and_target()
    costs = CostAssumptions(slippage_bps=cost_bps, annual_margin_rate=0.0)
    strategy = simulate_unleveraged_period(
        frame, target_at_close, DEVELOPMENT_PERIOD, costs
    )
    buy_hold = simulate_unleveraged_period(
        frame,
        pd.Series(1.0, index=frame.index, name="target_exposure"),
        DEVELOPMENT_PERIOD,
        costs,
    )
    executed_target = strategy["target_exposure"].to_numpy(dtype=float)
    oof_cash_target = 1.0 - executed_target
    oof_dates = pd.DatetimeIndex(pd.to_datetime(strategy["decision_date"]))
    positions = np.arange(len(strategy))
    actual_edge = np.where(positions % 3 == 0, 0.01, -0.005).astype(float)
    label = (actual_edge > 0.0).astype(float)
    probability = np.where(label == 1.0, 0.85, 0.15)
    prevalence = np.full(len(strategy), 0.5, dtype=float)
    predicted_edge = actual_edge + 0.0001
    causal_mean_edge = np.zeros(len(strategy), dtype=float)
    return (
        strategy,
        buy_hold,
        executed_target,
        probability,
        label,
        prevalence,
        predicted_edge,
        actual_edge,
        causal_mean_edge,
        oof_cash_target,
        oof_dates,
    )


def _inputs(cost_bps: float = 10.0) -> tuple[object, ...]:
    return copy.deepcopy(_cached_inputs(float(cost_bps)))


def _score(inputs: tuple[object, ...], *, cost_bps: float = 10.0):
    return score_regime_consensus_metrics(*inputs, cost_bps=cost_bps)


def test_passing_score_uses_complete_continuous_cash_round_trips() -> None:
    result = evaluate_regime_consensus_development(*_inputs(), cost_bps=10.0)
    metrics = result["metrics"]

    assert result["gates"]["passed"] is True
    assert result["gates"]["base_gates_passed"] is True
    assert metrics["scoring_contract"] == (
        "aapl-one-session-regime-consensus-development-v1"
    )
    assert metrics["oof_cash_signals"] == 56
    assert metrics["cash_days"] == 56
    assert metrics["completed_cash_episodes"] == 28
    assert metrics["cash_episodes"] == 28
    assert metrics["completed_cash_episode_wins"] == 28
    assert metrics["completed_cash_episode_win_rate"] == 1.0
    assert metrics["completed_cash_episode_mean_active_log_edge"] > 0.03
    assert metrics["brier_relative_improvement"] > 0.02
    assert metrics["expected_edge_mae_relative_improvement"] > 0.01
    assert metrics["cost_bps_bound_to_both_ledgers"] is True
    assert metrics["oof_cash_target_bound_to_executed_ledger"] is True

    first = metrics["completed_cash_episode_records"][0]
    assert first["cash_sessions"] == 2
    assert first["realized_active_log_edge"] == pytest.approx(
        np.log(first["sell_fill_price"] / first["buy_fill_price"])
    )
    assert first["ledger_active_log_edge"] == pytest.approx(
        first["realized_active_log_edge"]
    )
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_episode_quality_is_not_the_sum_of_isolated_daily_labels() -> None:
    inputs = list(_inputs())
    cash = np.asarray(inputs[9], dtype=float)
    actual = np.asarray(inputs[7], dtype=float).copy()
    # Make every isolated one-session target on a CASH day negative while the
    # exact multi-day sell/re-entry fills still define profitable episodes.
    actual[cash == 1.0] = -0.02
    label = (actual > 0.0).astype(float)
    inputs[3] = np.where(label == 1.0, 0.85, 0.15)
    inputs[4] = label
    inputs[6] = actual + 0.0001
    inputs[7] = actual

    metrics = _score(tuple(inputs))

    assert np.sum(actual[cash == 1.0]) < 0.0
    assert metrics["completed_cash_episode_wins"] == 28
    assert metrics["completed_cash_episode_mean_active_log_edge"] > 0.03


def test_first_and_final_fills_must_be_long() -> None:
    inputs = list(_inputs())
    target = np.asarray(inputs[2], dtype=float).copy()
    target[0] = 0.0
    inputs[2] = target
    with pytest.raises(RegimeConsensusScoringError, match="First and final"):
        _score(tuple(inputs))

    inputs = list(_inputs())
    target = np.asarray(inputs[2], dtype=float).copy()
    target[-1] = 0.0
    inputs[2] = target
    with pytest.raises(RegimeConsensusScoringError, match="First and final"):
        _score(tuple(inputs))


def test_oof_cash_is_bound_by_decision_date_and_must_be_binary() -> None:
    inputs = list(_inputs())
    cash = np.asarray(inputs[9], dtype=float).copy()
    cash[10] = 1.0 - cash[10]
    inputs[9] = cash
    with pytest.raises(RegimeConsensusScoringError, match="does not match the executed"):
        _score(tuple(inputs))

    inputs = list(_inputs())
    cash = np.asarray(inputs[9], dtype=float).copy()
    cash[10] = 0.5
    inputs[9] = cash
    with pytest.raises(RegimeConsensusScoringError, match="exactly 0 or 1"):
        _score(tuple(inputs))

    inputs = list(_inputs())
    dates = pd.DatetimeIndex(inputs[10]).to_numpy(copy=True)
    dates[10] = dates[9]
    inputs[10] = dates
    with pytest.raises(RegimeConsensusScoringError, match="unique chronological"):
        _score(tuple(inputs))


def test_cost_fills_cash_equity_and_returns_are_fail_closed() -> None:
    with pytest.raises(RegimeConsensusScoringError, match="declared cost_bps"):
        _score(_inputs(10.0), cost_bps=5.0)

    inputs = list(_inputs())
    strategy = inputs[0].copy()
    trade_row = int(np.flatnonzero(strategy["trade_executed"].to_numpy())[1])
    strategy.loc[trade_row, "fill_price"] += 0.01
    inputs[0] = strategy
    with pytest.raises(RegimeConsensusScoringError, match="fill prices"):
        _score(tuple(inputs))

    inputs = list(_inputs())
    strategy = inputs[0].copy()
    strategy.loc[100, "equity"] += 0.01
    inputs[0] = strategy
    with pytest.raises(
        RegimeConsensusScoringError,
        match="equity does not reconcile|daily_return does not reconcile",
    ):
        _score(tuple(inputs))


def test_predictive_thresholds_are_inclusive_except_episode_mean() -> None:
    metrics = _score(_inputs())
    metrics["brier_relative_improvement"] = REGIME_CONSENSUS_GATES[
        "minimum_brier_relative_improvement"
    ]
    metrics["expected_edge_mae_relative_improvement"] = REGIME_CONSENSUS_GATES[
        "minimum_expected_edge_mae_relative_improvement"
    ]
    metrics["completed_cash_episode_win_rate"] = REGIME_CONSENSUS_GATES[
        "minimum_completed_cash_episode_win_rate"
    ]
    gates = apply_regime_consensus_gates(metrics)
    assert gates["checks"]["brier_improves_causal_prevalence_by_at_least_2pct"]
    assert gates["checks"][
        "expected_edge_mae_improves_causal_mean_by_at_least_1pct"
    ]
    assert gates["checks"]["completed_cash_episode_win_rate_at_least_55pct"]
    assert gates["passed"] is True

    metrics["completed_cash_episode_mean_active_log_edge"] = 0.0
    gates = apply_regime_consensus_gates(metrics)
    assert gates["checks"][
        "completed_cash_episode_mean_edge_strictly_positive"
    ] is False
    assert gates["passed"] is False


def test_trailing_unmatured_targets_are_allowed_but_internal_gaps_fail() -> None:
    inputs = list(_inputs())
    label = np.asarray(inputs[4], dtype=float).copy()
    actual = np.asarray(inputs[7], dtype=float).copy()
    label[-3:] = np.nan
    actual[-3:] = np.nan
    inputs[4] = label
    inputs[7] = actual
    metrics = _score(tuple(inputs))
    assert metrics["unmatured_trailing_oof_rows"] == 3
    assert metrics["mature_oof_rows"] == len(label) - 3

    inputs = list(_inputs())
    actual = np.asarray(inputs[7], dtype=float).copy()
    actual[100] = np.nan
    inputs[7] = actual
    with pytest.raises(RegimeConsensusScoringError, match="trailing suffix"):
        _score(tuple(inputs))


def _worse_aapl_only(metrics: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(metrics)
    result["brier_score"] = float(metrics["brier_score"]) + 0.01
    result["expected_edge_mae"] = float(metrics["expected_edge_mae"]) + 0.001
    result["total_active_log_edge"] = float(metrics["total_active_log_edge"]) - 0.10
    result["minimum_fold_active_log_edge"] = (
        float(metrics["minimum_fold_active_log_edge"]) - 0.01
    )
    return result


def test_paired_ablation_requires_full_improvement_at_both_costs() -> None:
    full_5 = _score(_inputs(5.0), cost_bps=5.0)
    full_10 = _score(_inputs(10.0), cost_bps=10.0)
    aapl_5 = _worse_aapl_only(full_5)
    aapl_10 = _worse_aapl_only(full_10)

    result = evaluate_regime_consensus_ablation(
        full_5, aapl_5, full_10, aapl_10
    )
    assert result["passed"] is True
    assert result["comparisons"]["5bps"]["passed"] is True
    assert result["comparisons"]["10bps"]["passed"] is True
    assert result["checks"][
        "full_brier_strictly_lower_than_aapl_only_5bps"
    ]
    assert result["checks"][
        "full_total_active_edge_strictly_greater_than_aapl_only_10bps"
    ]
    json.dumps(result, sort_keys=True, allow_nan=False)

    tied = copy.deepcopy(aapl_10)
    tied["total_active_log_edge"] = full_10["total_active_log_edge"]
    failed = evaluate_regime_consensus_ablation(full_5, aapl_5, full_10, tied)
    assert failed["checks"][
        "full_total_active_edge_strictly_greater_than_aapl_only_10bps"
    ] is False
    assert failed["passed"] is False


def test_paired_ablation_fails_mismatched_support_and_wrong_cost() -> None:
    full_5 = _score(_inputs(5.0), cost_bps=5.0)
    full_10 = _score(_inputs(10.0), cost_bps=10.0)
    aapl_5 = _worse_aapl_only(full_5)
    aapl_10 = _worse_aapl_only(full_10)
    aapl_5["predictive_evaluation_support_sha256"] = "0" * 64

    result = evaluate_regime_consensus_ablation(
        full_5, aapl_5, full_10, aapl_10
    )
    assert result["checks"]["paired_predictive_support_matches_5bps"] is False
    assert result["passed"] is False

    wrong_cost = copy.deepcopy(full_5)
    wrong_cost["declared_cost_bps"] = 10.0
    with pytest.raises(RegimeConsensusScoringError, match="required 5bps cost"):
        evaluate_regime_consensus_ablation(
            wrong_cost, _worse_aapl_only(wrong_cost), full_10, aapl_10
        )
