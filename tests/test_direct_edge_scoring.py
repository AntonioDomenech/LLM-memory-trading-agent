from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.direct_edge_scoring import (
    DIRECT_EDGE_GATES,
    DirectEdgeScoringError,
    apply_direct_edge_gates,
    evaluate_direct_edge_development,
    score_direct_edge_metrics,
)


NEGATIVE_BUY_HOLD_YEARS = (2008, 2011, 2015, 2018)


def _add_execution_proof(ledger: pd.DataFrame, *, cost_bps: float = 10.0) -> pd.DataFrame:
    result = ledger.copy()
    shares = result["shares"].to_numpy(dtype=float)
    delta = np.diff(np.r_[0.0, shares])
    reference = np.full(len(result), 100.0, dtype=float)
    rate = cost_bps / 10_000.0
    fill = reference.copy()
    fill[delta > 1e-12] *= 1.0 + rate
    fill[delta < -1e-12] *= 1.0 - rate
    slippage = np.abs(delta) * np.abs(fill - reference)
    daily_return = result["daily_return"].to_numpy(dtype=float)
    equity = 1000.0 * np.cumprod(1.0 + daily_return)
    result["signed_share_delta"] = delta
    result["reference_price"] = reference
    result["fill_price"] = fill
    result["slippage"] = slippage
    result["fees"] = np.zeros(len(result), dtype=float)
    result["trade_executed"] = np.abs(delta) > 1e-12
    result["equity"] = equity
    result["equity_before_fill"] = np.r_[1000.0, equity[:-1]]
    return result


def _synthetic_inputs() -> tuple[object, ...]:
    dates = pd.bdate_range("2005-01-03", "2018-12-31")
    count = len(dates)
    oof_cash_target = np.zeros(count, dtype=float)
    oof_cash_block_start = np.zeros(count, dtype=float)
    for start in range(100, 2100, 100):
        oof_cash_target[start : start + 5] = 1.0
        oof_cash_block_start[start] = 1.0
    executed_target = 1.0 - oof_cash_target
    executed_cash = executed_target == 0.0
    entered = executed_cash & ~np.r_[False, executed_cash[:-1]]
    exited = ~executed_cash & np.r_[False, executed_cash[:-1]]

    benchmark_log = np.full(count, 0.00008, dtype=float)
    for year in NEGATIVE_BUY_HOLD_YEARS:
        benchmark_log[dates.year == year] = -0.0002
    benchmark_return = np.expm1(benchmark_log)
    strategy_return = np.expm1(benchmark_log + 0.0001)
    strategy = _add_execution_proof(pd.DataFrame(
        {
            "fill_date": dates.date.astype(str),
            "decision_date": dates.date.astype(str),
            "target_exposure": executed_target,
            "new_exposure_after_fill": executed_target,
            "holding_exposure_for_return": executed_target,
            "cash": np.where(executed_cash, 1000.0, 0.0),
            "shares": np.where(executed_cash, 0.0, 10.0),
            "margin_interest": np.zeros(count),
            "trade_executed": entered | exited,
            "daily_return": strategy_return,
        }
    ))
    buy_hold = _add_execution_proof(pd.DataFrame(
        {
            "fill_date": dates.date.astype(str),
            "target_exposure": np.ones(count),
            "new_exposure_after_fill": np.ones(count),
            "holding_exposure_for_return": np.ones(count),
            "cash": np.zeros(count),
            "shares": np.full(count, 10.0),
            "margin_interest": np.zeros(count),
            "trade_executed": np.r_[True, np.zeros(count - 1, dtype=bool)],
            "daily_return": benchmark_return,
        }
    ))

    positions = np.arange(count)
    actual_edge = np.where(positions % 4 == 0, 0.01, -0.005)
    cash_label = (actual_edge > 0.0).astype(float)
    probability = np.where(cash_label == 1.0, 0.9, 0.1)
    baseline_probability = np.full(count, 0.5)
    predicted_edge = actual_edge + 0.0005
    baseline_edge = np.zeros(count)
    return (
        strategy,
        buy_hold,
        executed_target,
        probability,
        cash_label,
        baseline_probability,
        predicted_edge,
        actual_edge,
        baseline_edge,
        oof_cash_target,
        oof_cash_block_start,
        dates,
    )


def _score(inputs: tuple[object, ...]) -> dict[str, object]:
    return score_direct_edge_metrics(*inputs, cost_bps=10.0)


def _bind_strategy_to_oof_cash(inputs: list[object]) -> None:
    cash_target = np.asarray(inputs[9], dtype=float)
    executed_target = 1.0 - cash_target
    strategy = inputs[0].copy()
    strategy["target_exposure"] = executed_target
    strategy["new_exposure_after_fill"] = executed_target
    strategy["holding_exposure_for_return"] = executed_target
    strategy["cash"] = np.where(cash_target == 1.0, 1000.0, 0.0)
    strategy["shares"] = np.where(cash_target == 1.0, 0.0, 10.0)
    strategy = _add_execution_proof(strategy)
    inputs[0] = strategy
    inputs[2] = executed_target


def test_passing_wrapper_preserves_base_gates_and_adds_direct_metrics() -> None:
    inputs = _synthetic_inputs()
    result = evaluate_direct_edge_development(*inputs, cost_bps=10.0)
    metrics = result["metrics"]
    gates = result["gates"]

    assert gates["passed"] is True
    assert gates["base_gates_passed"] is True
    assert metrics["scoring_contract"] == "aapl-direct-cash-edge-development-v1"
    assert metrics["underlying_scoring_contract"] == (
        "aapl-downside-ensemble-development-v1"
    )
    assert metrics["brier_score"] < metrics[
        "causal_training_climatology_brier_score"
    ]
    assert metrics["strategy_no_leverage_proof"]["passed"] is True
    assert metrics["buy_hold_no_leverage_proof"]["passed"] is True
    assert metrics["expected_edge_mae"] == pytest.approx(0.0005)
    assert metrics["expected_edge_mae_relative_improvement"] > 0.01
    assert metrics["oof_cash_episode_starts"] == 20
    assert metrics["episode_start_realized_edge_mean_10bps"] == pytest.approx(0.01)
    assert metrics["episode_start_realized_edge_win_rate_10bps"] == 1.0
    assert set(gates["contract"]["base_development_gates"])
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_mae_is_against_causal_baseline_tie_fails_and_one_percent_passes() -> None:
    inputs = list(_synthetic_inputs())
    actual = np.asarray(inputs[7], dtype=float)
    baseline = np.asarray(inputs[8], dtype=float)
    baseline_mae = float(np.mean(np.abs(actual - baseline)))

    inputs[6] = baseline.copy()
    tied_metrics = _score(tuple(inputs))
    assert tied_metrics["expected_edge_mae"] == pytest.approx(baseline_mae)
    assert tied_metrics["causal_training_mean_edge_mae"] == pytest.approx(baseline_mae)
    assert tied_metrics["expected_edge_mae_relative_improvement"] == pytest.approx(0.0)
    tied_gate = apply_direct_edge_gates(tied_metrics)
    assert (
        tied_gate["checks"][
            "expected_edge_mae_improves_causal_mean_by_at_least_1pct"
        ]
        is False
    )

    # A constant absolute error equal to 99% of baseline MAE gives exactly a
    # 1% relative improvement, which is inclusive by contract.
    inputs[6] = actual + 0.99 * baseline_mae
    threshold_metrics = _score(tuple(inputs))
    assert threshold_metrics["expected_edge_mae"] == pytest.approx(
        0.99 * baseline_mae
    )
    assert threshold_metrics["expected_edge_mae_relative_improvement"] == pytest.approx(
        DIRECT_EDGE_GATES["minimum_expected_edge_mae_relative_improvement"]
    )
    assert (
        apply_direct_edge_gates(threshold_metrics)["checks"][
            "expected_edge_mae_improves_causal_mean_by_at_least_1pct"
        ]
        is True
    )


def test_episode_start_mean_and_win_rate_use_only_episode_start_edges() -> None:
    inputs = list(_synthetic_inputs())
    actual = np.asarray(inputs[7], dtype=float).copy()
    cash_target = np.zeros(len(actual), dtype=float)
    starts = (10, 30, 50, 70)
    for start in starts:
        cash_target[start : start + 5] = 1.0
    actual[list(starts)] = [-0.01, -0.01, 0.01, 0.03]
    label = (actual > 0.0).astype(float)
    inputs[3] = np.where(label == 1.0, 0.9, 0.1)
    inputs[4] = label
    inputs[6] = actual + 0.0005
    inputs[7] = actual
    inputs[9] = cash_target
    block_start = np.zeros(len(actual), dtype=float)
    block_start[list(starts)] = 1.0
    inputs[10] = block_start
    _bind_strategy_to_oof_cash(inputs)

    metrics = _score(tuple(inputs))
    gates = apply_direct_edge_gates(metrics)

    assert metrics["mature_oof_cash_episode_starts"] == 4
    assert metrics["episode_start_realized_edge_mean_10bps"] == pytest.approx(0.005)
    assert metrics["episode_start_realized_edge_wins_10bps"] == 2
    assert metrics["episode_start_realized_edge_win_rate_10bps"] == 0.5
    assert gates["checks"]["episode_start_realized_edge_mean_strictly_positive"]
    assert (
        gates["checks"][
            "episode_start_realized_edge_win_rate_strictly_above_half"
        ]
        is False
    )
    assert gates["passed"] is False


def test_trailing_immature_labels_are_allowed_but_internal_gaps_are_rejected() -> None:
    inputs = list(_synthetic_inputs())
    label = np.asarray(inputs[4], dtype=float).copy()
    actual = np.asarray(inputs[7], dtype=float).copy()
    cash_target = np.asarray(inputs[9], dtype=float).copy()
    label[-6:] = np.nan
    actual[-6:] = np.nan
    cash_target[-3:] = 1.0
    block_start = np.asarray(inputs[10], dtype=float).copy()
    block_start[-3] = 1.0
    inputs[4] = label
    inputs[7] = actual
    inputs[9] = cash_target
    inputs[10] = block_start
    _bind_strategy_to_oof_cash(inputs)

    metrics = _score(tuple(inputs))
    assert metrics["unmatured_trailing_oof_rows"] == 6
    assert metrics["unmatured_trailing_label_rows"] == 6
    assert metrics["unmatured_oof_cash_episode_starts"] == 1

    internal = list(_synthetic_inputs())
    internal_actual = np.asarray(internal[7], dtype=float).copy()
    internal_actual[100] = np.nan
    internal[7] = internal_actual
    with pytest.raises(DirectEdgeScoringError, match="trailing suffix"):
        _score(tuple(internal))

    mismatched = list(_synthetic_inputs())
    mismatched_label = np.asarray(mismatched[4], dtype=float).copy()
    mismatched_label[-6:] = np.nan
    mismatched[4] = mismatched_label
    with pytest.raises(DirectEdgeScoringError, match="same maturity mask"):
        _score(tuple(mismatched))


def test_alignment_nonfinite_binary_consistency_and_base_fail_closed() -> None:
    inputs = list(_synthetic_inputs())
    inputs[6] = np.asarray(inputs[6], dtype=float)[:-1]
    with pytest.raises(DirectEdgeScoringError, match="must align"):
        _score(tuple(inputs))

    inputs = list(_synthetic_inputs())
    prediction = np.asarray(inputs[6], dtype=float).copy()
    prediction[10] = np.inf
    inputs[6] = prediction
    with pytest.raises(DirectEdgeScoringError, match="finite"):
        _score(tuple(inputs))

    inputs = list(_synthetic_inputs())
    label = np.asarray(inputs[4], dtype=float).copy()
    label[10] = 1.0 - label[10]
    inputs[4] = label
    with pytest.raises(DirectEdgeScoringError, match="exactly equal"):
        _score(tuple(inputs))

    inputs = list(_synthetic_inputs())
    strategy = inputs[0].copy()
    strategy.loc[0, "cash"] = -1e-12
    inputs[0] = strategy
    with pytest.raises(DirectEdgeScoringError, match="strict no-leverage"):
        _score(tuple(inputs))

    inputs = list(_synthetic_inputs())
    mismatched_cash = np.asarray(inputs[9], dtype=float).copy()
    mismatched_cash[10:15] = 1.0
    mismatched_start = np.asarray(inputs[10], dtype=float).copy()
    mismatched_start[10] = 1.0
    inputs[9] = mismatched_cash
    inputs[10] = mismatched_start
    with pytest.raises(DirectEdgeScoringError, match="does not match the executed"):
        _score(tuple(inputs))


def test_adjacent_cash_blocks_are_scored_as_two_predictions() -> None:
    inputs = list(_synthetic_inputs())
    cash = np.zeros_like(np.asarray(inputs[9], dtype=float))
    starts = np.zeros_like(cash)
    cash[20:30] = 1.0
    starts[[20, 25]] = 1.0
    actual = np.asarray(inputs[7], dtype=float).copy()
    actual[[20, 25]] = [0.01, -0.005]
    label = (actual > 0.0).astype(float)
    inputs[3] = np.where(label == 1.0, 0.9, 0.1)
    inputs[4] = label
    inputs[6] = actual + 0.0005
    inputs[7] = actual
    inputs[9] = cash
    inputs[10] = starts
    _bind_strategy_to_oof_cash(inputs)

    metrics = _score(tuple(inputs))

    assert metrics["oof_cash_episode_starts"] == 2
    assert metrics["oof_cash_block_starts"] == 2
    assert metrics["episode_start_realized_edge_wins_10bps"] == 1
    assert metrics["episode_start_realized_edge_win_rate_10bps"] == 0.5


def test_cash_block_starts_and_declared_cost_are_fail_closed() -> None:
    inputs = list(_synthetic_inputs())
    missing_start = np.asarray(inputs[10], dtype=float).copy()
    missing_start[100] = 0.0
    inputs[10] = missing_start
    with pytest.raises(DirectEdgeScoringError, match="not exactly reconstructed"):
        _score(tuple(inputs))

    inputs = _synthetic_inputs()
    with pytest.raises(DirectEdgeScoringError, match="declared cost_bps"):
        score_direct_edge_metrics(*inputs, cost_bps=5.0)

    corrupted = list(_synthetic_inputs())
    strategy = corrupted[0].copy()
    strategy.loc[0, "daily_return"] += 0.001
    corrupted[0] = strategy
    with pytest.raises(DirectEdgeScoringError, match="reconcile to equity"):
        _score(tuple(corrupted))


def test_direct_wrapper_cannot_pass_when_any_base_gate_fails() -> None:
    passing = _score(_synthetic_inputs())
    assert apply_direct_edge_gates(passing)["passed"] is True

    failed = copy.deepcopy(passing)
    failed["total_active_log_edge"] = 0.0
    gates = apply_direct_edge_gates(failed)

    assert gates["checks"]["material_total_active_log_edge"] is False
    assert gates["base_gates_passed"] is False
    assert (
        gates["checks"][
            "expected_edge_mae_improves_causal_mean_by_at_least_1pct"
        ]
        is True
    )
    assert gates["passed"] is False
