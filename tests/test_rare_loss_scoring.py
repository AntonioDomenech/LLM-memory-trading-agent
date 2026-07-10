from __future__ import annotations

import copy
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.deterministic_aapl import CostAssumptions, EvaluationPeriod
from agent_benchmark.rare_loss_scoring import (
    CLIPPED_EDGE_BOUND,
    RARE_LOSS_GATES,
    RareLossScoringError,
    apply_rare_loss_gates,
    evaluate_rare_loss_ablation,
    evaluate_rare_loss_development,
    score_rare_loss_metrics,
)
from agent_benchmark.unleveraged_aapl import simulate_unleveraged_period


NEGATIVE_BUY_HOLD_YEARS = {2008, 2011, 2015, 2018}
DEVELOPMENT_PERIOD = EvaluationPeriod(
    "rare_loss_development", "2005-01-01", "2018-12-31"
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

    log_return = np.asarray(
        [
            0.0001 if date.year in NEGATIVE_BUY_HOLD_YEARS else 0.0007
            for date in dates
        ],
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
    cash = 1.0 - executed_target
    dates = pd.DatetimeIndex(pd.to_datetime(strategy["decision_date"]))
    positions = np.arange(len(strategy))

    actual_edge = np.where(positions % 3 == 0, 0.01, -0.005).astype(float)
    ordinary_label = (actual_edge > 0.0).astype(float)
    ordinary_probability = np.where(ordinary_label == 1.0, 0.85, 0.15)
    ordinary_prevalence = np.full(len(strategy), 0.5, dtype=float)
    predicted_edge = actual_edge + 0.0001
    causal_mean_edge = np.zeros(len(strategy), dtype=float)

    # Exactly one of each episode's two CASH rows is a severe event. This gives
    # 50% precision against a causal 10% prevalence while retaining ordinary
    # non-signal rows for a meaningful severe-event Brier score.
    severe_label = np.zeros(len(strategy), dtype=float)
    starts = np.flatnonzero((cash == 1.0) & np.r_[True, cash[:-1] == 0.0])
    severe_label[starts] = 1.0
    severe_probability = np.where(severe_label == 1.0, 0.80, 0.02)
    severe_prevalence = np.full(len(strategy), 0.10, dtype=float)
    return (
        strategy,
        buy_hold,
        executed_target,
        ordinary_probability,
        ordinary_label,
        ordinary_prevalence,
        predicted_edge,
        actual_edge,
        causal_mean_edge,
        severe_probability,
        severe_label,
        severe_prevalence,
        cash,
        dates,
    )


def _inputs(cost_bps: float = 10.0) -> tuple[object, ...]:
    return copy.deepcopy(_cached_inputs(float(cost_bps)))


def _score(cost_bps: float = 10.0) -> dict[str, object]:
    return score_rare_loss_metrics(*_inputs(cost_bps), cost_bps=cost_bps)


def test_passing_score_adds_severe_brier_precision_and_episode_contract() -> None:
    result = evaluate_rare_loss_development(*_inputs(), cost_bps=10.0)
    metrics = result["metrics"]
    gates = result["gates"]

    assert metrics["scoring_contract"] == (
        "aapl-one-session-rare-loss-forest-development-v1"
    )
    assert metrics["underlying_scoring_contract"] == (
        "aapl-one-session-regime-consensus-development-v1"
    )
    assert metrics["mature_severe_oof_rows"] == metrics["mature_oof_rows"]
    assert metrics["mature_cash_signal_count"] == 56
    assert metrics["severe_event_precision_at_cash_signals"] == pytest.approx(0.5)
    assert metrics["causal_severe_prevalence_at_cash_signals"] == pytest.approx(0.1)
    assert metrics["severe_event_precision_lift"] == pytest.approx(5.0)
    assert metrics["severe_brier_relative_improvement"] > 0.05
    assert metrics["completed_cash_episodes"] == 28
    assert gates["checks"][
        "severe_brier_improves_causal_prevalence_by_at_least_5pct"
    ]
    assert gates["checks"]["severe_event_precision_at_least_25pct"]
    assert gates["checks"][
        "severe_event_precision_at_least_twice_causal_prevalence"
    ]
    assert gates["checks"]["at_least_20_completed_cash_episodes"]
    assert gates["passed"] is True
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_rare_loss_gate_boundaries_are_inclusive_and_19_episodes_fail() -> None:
    metrics = _score()
    metrics["severe_brier_relative_improvement"] = RARE_LOSS_GATES[
        "minimum_severe_brier_relative_improvement"
    ]
    metrics["severe_event_precision_at_cash_signals"] = RARE_LOSS_GATES[
        "minimum_severe_event_precision"
    ]
    metrics["severe_event_precision_lift"] = RARE_LOSS_GATES[
        "minimum_severe_event_precision_lift"
    ]
    metrics["completed_cash_episodes"] = RARE_LOSS_GATES[
        "minimum_completed_cash_episodes"
    ]
    gates = apply_rare_loss_gates(metrics)
    assert gates["passed"] is True

    metrics["completed_cash_episodes"] = 19
    gates = apply_rare_loss_gates(metrics)
    assert gates["checks"]["at_least_20_completed_cash_episodes"] is False
    assert gates["passed"] is False


@pytest.mark.parametrize(
    ("metric", "threshold", "check"),
    (
        (
            "severe_brier_relative_improvement",
            0.05,
            "severe_brier_improves_causal_prevalence_by_at_least_5pct",
        ),
        (
            "severe_event_precision_at_cash_signals",
            0.25,
            "severe_event_precision_at_least_25pct",
        ),
        (
            "severe_event_precision_lift",
            2.0,
            "severe_event_precision_at_least_twice_causal_prevalence",
        ),
    ),
)
def test_each_severe_predictive_gate_fails_below_its_frozen_threshold(
    metric: str,
    threshold: float,
    check: str,
) -> None:
    metrics = _score()
    metrics[metric] = np.nextafter(threshold, -np.inf)
    gates = apply_rare_loss_gates(metrics)
    assert gates["checks"][check] is False
    assert gates["passed"] is False


def test_severe_and_ordinary_labels_require_the_same_trailing_maturity() -> None:
    inputs = list(_inputs())
    ordinary = np.asarray(inputs[4], dtype=float).copy()
    edge = np.asarray(inputs[7], dtype=float).copy()
    severe = np.asarray(inputs[10], dtype=float).copy()
    ordinary[-3:] = np.nan
    edge[-3:] = np.nan
    severe[-3:] = np.nan
    inputs[4] = ordinary
    inputs[7] = edge
    inputs[10] = severe
    metrics = score_rare_loss_metrics(*tuple(inputs), cost_bps=10.0)
    assert metrics["unmatured_trailing_oof_rows"] == 3
    assert metrics["unmatured_trailing_severe_oof_rows"] == 3

    mismatch = list(_inputs())
    ordinary = np.asarray(mismatch[4], dtype=float).copy()
    edge = np.asarray(mismatch[7], dtype=float).copy()
    severe = np.asarray(mismatch[10], dtype=float).copy()
    ordinary[-3:] = np.nan
    edge[-3:] = np.nan
    severe[-2:] = np.nan
    mismatch[4] = ordinary
    mismatch[7] = edge
    mismatch[10] = severe
    with pytest.raises(RareLossScoringError, match="share maturity support"):
        score_rare_loss_metrics(*tuple(mismatch), cost_bps=10.0)

    internal_gap = list(_inputs())
    severe = np.asarray(internal_gap[10], dtype=float).copy()
    severe[100] = np.nan
    internal_gap[10] = severe
    with pytest.raises(RareLossScoringError, match="only as a suffix"):
        score_rare_loss_metrics(*tuple(internal_gap), cost_bps=10.0)


@pytest.mark.parametrize("value", (-CLIPPED_EDGE_BOUND, CLIPPED_EDGE_BOUND))
def test_predicted_and_actual_edge_accept_the_exact_clip_boundaries(
    value: float,
) -> None:
    inputs = list(_inputs())
    position = 100
    ordinary = np.asarray(inputs[4], dtype=float).copy()
    probability = np.asarray(inputs[3], dtype=float).copy()
    predicted = np.asarray(inputs[6], dtype=float).copy()
    actual = np.asarray(inputs[7], dtype=float).copy()
    actual[position] = value
    predicted[position] = value
    ordinary[position] = float(value > 0.0)
    probability[position] = 0.85 if ordinary[position] == 1.0 else 0.15
    inputs[3] = probability
    inputs[4] = ordinary
    inputs[6] = predicted
    inputs[7] = actual

    metrics = score_rare_loss_metrics(*tuple(inputs), cost_bps=10.0)
    assert metrics["expected_edge_target_semantics"].endswith(
        "clipped_to_plus_minus_8pct"
    )


@pytest.mark.parametrize(
    ("input_position", "name"),
    ((6, "predicted_edge_10bps"), (7, "actual_edge_10bps")),
)
@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_predicted_and_actual_edge_refuse_values_outside_eight_percent(
    input_position: int,
    name: str,
    sign: float,
) -> None:
    inputs = list(_inputs())
    values = np.asarray(inputs[input_position], dtype=float).copy()
    values[100] = sign * np.nextafter(CLIPPED_EDGE_BOUND, np.inf)
    inputs[input_position] = values
    with pytest.raises(RareLossScoringError, match=rf"{name}.*clipped"):
        score_rare_loss_metrics(*tuple(inputs), cost_bps=10.0)


def _worse_core(metrics: dict[str, object]) -> dict[str, object]:
    core = copy.deepcopy(metrics)
    core["brier_score"] = float(metrics["brier_score"]) + 0.01
    core["severe_brier_score"] = float(metrics["severe_brier_score"]) + 0.01
    core["total_active_log_edge"] = float(metrics["total_active_log_edge"]) - 0.10
    core["minimum_fold_active_log_edge"] = (
        float(metrics["minimum_fold_active_log_edge"]) - 0.01
    )
    return core


def test_strict_full_vs_core_ablation_passes_all_checks_at_both_costs() -> None:
    full_5 = _score(5.0)
    full_10 = _score(10.0)
    core_5 = _worse_core(full_5)
    core_10 = _worse_core(full_10)

    result = evaluate_rare_loss_ablation(full_5, core_5, full_10, core_10)
    assert result["passed"] is True
    for cost in ("5bps", "10bps"):
        comparison = result["comparisons"][cost]
        assert comparison["passed"] is True
        assert result["checks"][f"paired_support_matches_{cost}"]
        assert result["checks"][f"full_ordinary_brier_strictly_lower_{cost}"]
        assert result["checks"][f"full_severe_brier_strictly_lower_{cost}"]
        assert result["checks"][f"full_total_edge_strictly_higher_{cost}"]
        assert result["checks"][f"full_weakest_fold_strictly_higher_{cost}"]
        assert result["checks"][f"full_rare_loss_gates_pass_{cost}"]
    json.dumps(result, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("cost", ("5bps", "10bps"))
@pytest.mark.parametrize(
    ("metric", "check"),
    (
        ("brier_score", "full_ordinary_brier_strictly_lower"),
        ("severe_brier_score", "full_severe_brier_strictly_lower"),
        ("total_active_log_edge", "full_total_edge_strictly_higher"),
        ("minimum_fold_active_log_edge", "full_weakest_fold_strictly_higher"),
    ),
)
def test_each_strict_ablation_comparison_must_win_at_each_cost(
    cost: str,
    metric: str,
    check: str,
) -> None:
    full_5 = _score(5.0)
    full_10 = _score(10.0)
    core_5 = _worse_core(full_5)
    core_10 = _worse_core(full_10)
    full = full_5 if cost == "5bps" else full_10
    core = core_5 if cost == "5bps" else core_10
    core[metric] = full[metric]

    result = evaluate_rare_loss_ablation(full_5, core_5, full_10, core_10)
    assert result["checks"][f"{check}_{cost}"] is False
    assert result["comparisons"][cost]["passed"] is False
    assert result["passed"] is False


def test_ablation_rejects_mismatched_ordinary_or_severe_support() -> None:
    full_5 = _score(5.0)
    full_10 = _score(10.0)
    core_5 = _worse_core(full_5)
    core_10 = _worse_core(full_10)
    core_5["severe_predictive_support_sha256"] = "0" * 64
    result = evaluate_rare_loss_ablation(full_5, core_5, full_10, core_10)
    assert result["checks"]["paired_support_matches_5bps"] is False
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("field", "check_suffix"),
    (
        (
            "predictive_evaluation_support_sha256",
            "ordinary_support_bound_across_costs",
        ),
        ("severe_predictive_support_sha256", "severe_support_bound_across_costs"),
        ("mature_oof_rows", "ordinary_support_bound_across_costs"),
        ("mature_severe_oof_rows", "severe_support_bound_across_costs"),
    ),
)
def test_cross_cost_support_mismatch_fails_even_when_full_core_pairing_matches(
    field: str,
    check_suffix: str,
) -> None:
    full_5 = _score(5.0)
    full_10 = _score(10.0)
    core_5 = _worse_core(full_5)
    core_10 = _worse_core(full_10)
    if field.endswith("sha256"):
        replacement: object = "f" * 64
    else:
        replacement = int(full_10[field]) - 1
    # Keep full/core identical inside the 10-bps pair. Only cross-cost binding
    # differs, so this specifically exercises the new fail-closed contract.
    full_10[field] = replacement
    core_10[field] = replacement

    result = evaluate_rare_loss_ablation(full_5, core_5, full_10, core_10)
    assert result["checks"]["paired_support_matches_10bps"] is True
    assert result["cross_cost_predictive_binding"][
        f"full_{check_suffix}"
    ] is False
    assert result["cross_cost_predictive_binding"][
        f"core_{check_suffix}"
    ] is False
    assert result["passed"] is False


@pytest.mark.parametrize("model", ("full", "core"))
@pytest.mark.parametrize(
    ("field", "check_suffix"),
    (
        ("brier_score", "ordinary_brier_bound_across_costs"),
        ("severe_brier_score", "severe_brier_bound_across_costs"),
        ("expected_edge_mae", "edge_mae_bound_across_costs"),
    ),
)
def test_cross_cost_brier_and_mae_mismatch_fail_for_each_model(
    model: str,
    field: str,
    check_suffix: str,
) -> None:
    full_5 = _score(5.0)
    full_10 = _score(10.0)
    core_5 = _worse_core(full_5)
    core_10 = _worse_core(full_10)
    changed = full_10 if model == "full" else core_10
    changed[field] = float(changed[field]) + 1e-6

    result = evaluate_rare_loss_ablation(full_5, core_5, full_10, core_10)
    assert result["cross_cost_predictive_binding"][
        f"{model}_{check_suffix}"
    ] is False
    assert result["checks"][f"{model}_{check_suffix}"] is False
    assert result["passed"] is False


def test_cross_cost_prediction_and_cash_hash_mismatch_fails_closed() -> None:
    full_5 = _score(5.0)
    full_10 = _score(10.0)
    core_5 = _worse_core(full_5)
    core_10 = _worse_core(full_10)
    assert full_5["prediction_and_cash_target_sha256"] == full_10[
        "prediction_and_cash_target_sha256"
    ]
    assert core_5["prediction_and_cash_target_sha256"] == core_10[
        "prediction_and_cash_target_sha256"
    ]

    # Keep the 10-bps full/core pair mutually consistent. This isolates the
    # fact that it is not the same predictive/CASH object used at 5 bps.
    replacement = "a" * 64
    full_10["prediction_and_cash_target_sha256"] = replacement
    core_10["prediction_and_cash_target_sha256"] = replacement
    result = evaluate_rare_loss_ablation(full_5, core_5, full_10, core_10)

    for model in ("full", "core"):
        key = f"{model}_predictions_and_cash_target_bound_across_costs"
        assert result["cross_cost_predictive_binding"][key] is False
        assert result["checks"][key] is False
    assert result["passed"] is False

    core_5 = _worse_core(full_5)
    core_5["predictive_evaluation_support_sha256"] = "1" * 64
    result = evaluate_rare_loss_ablation(full_5, core_5, full_10, core_10)
    assert result["checks"]["paired_support_matches_5bps"] is False
    assert result["passed"] is False
