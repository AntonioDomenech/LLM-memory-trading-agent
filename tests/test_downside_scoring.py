import copy
import json

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.downside_scoring import (
    ACTIVE_EDGE_WIN_TOLERANCE,
    DEVELOPMENT_FOLDS,
    DEVELOPMENT_GATES,
    DownsideScoringError,
    apply_downside_gates,
    evaluate_downside_development,
    month_end_rolling_win_rate,
    score_downside_metrics,
)


NEGATIVE_BUY_HOLD_YEARS = (2008, 2011, 2015, 2018)


def _synthetic_inputs(
    *,
    active_log_increment: float = 0.0001,
    negative_buy_hold_years: tuple[int, ...] = NEGATIVE_BUY_HOLD_YEARS,
):
    dates = pd.bdate_range("2005-01-03", "2018-12-31")
    count = len(dates)
    target = np.ones(count, dtype=float)
    # Twenty disjoint three-session episodes: 60 cash days, below 2% of rows.
    for start in range(100, 2100, 100):
        target[start : start + 3] = 0.0
    cash = target == 0.0
    entered_cash = cash & ~np.r_[False, cash[:-1]]
    exited_cash = ~cash & np.r_[False, cash[:-1]]

    benchmark_log = np.full(count, 0.00008, dtype=float)
    for year in negative_buy_hold_years:
        benchmark_log[dates.year == year] = -0.0002
    benchmark_return = np.expm1(benchmark_log)
    strategy_return = np.expm1(benchmark_log + active_log_increment)

    strategy = pd.DataFrame(
        {
            "fill_date": dates.date.astype(str),
            "target_exposure": target,
            "new_exposure_after_fill": target,
            "holding_exposure_for_return": target,
            "cash": np.where(cash, 1000.0, 0.0),
            "shares": np.where(cash, 0.0, 10.0),
            "margin_interest": np.zeros(count),
            "trade_executed": entered_cash | exited_cash,
            "daily_return": strategy_return,
        }
    )
    buy_hold = pd.DataFrame(
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
    )

    labels = (np.arange(count) % 11 == 0).astype(float)
    probability = np.where(labels == 1.0, 0.9, 0.1)
    causal_climatology = np.full(count, 0.5)
    return strategy, buy_hold, target, probability, labels, causal_climatology


def _score(inputs, *, cost_bps: float = 10.0):
    strategy, buy_hold, target, probability, labels, climatology = inputs
    return score_downside_metrics(
        strategy,
        buy_hold,
        target,
        probability,
        labels,
        climatology,
        cost_bps=cost_bps,
    )


def _replace_strategy_target(strategy: pd.DataFrame, target: np.ndarray) -> None:
    cash = target == 0.0
    strategy["target_exposure"] = target
    strategy["new_exposure_after_fill"] = target
    strategy["holding_exposure_for_return"] = target
    strategy["cash"] = np.where(cash, 1000.0, 0.0)
    strategy["shares"] = np.where(cash, 0.0, 10.0)


def test_passing_score_covers_all_folds_downside_gates_and_json_contract():
    inputs = _synthetic_inputs()
    strategy, buy_hold, target, probability, labels, climatology = inputs

    result = evaluate_downside_development(
        strategy,
        buy_hold,
        target,
        probability,
        labels,
        climatology,
        cost_bps=10.0,
    )
    metrics = result["metrics"]

    assert result["gates"]["passed"] is True
    assert metrics["caller_supplied_cost_bps"] == 10.0
    assert list(metrics["fold_active_log_edges"]) == [item[0] for item in DEVELOPMENT_FOLDS]
    assert metrics["positive_folds"] == 7
    assert metrics["rolling_252_session_month_end_win_rate"] == 1.0
    assert metrics["rolling_756_session_month_end_win_rate"] == 1.0
    assert metrics["cash_days"] == 60
    assert metrics["cash_episodes"] == 20
    assert metrics["negative_buy_hold_years"] == [
        str(year) for year in NEGATIVE_BUY_HOLD_YEARS
    ]
    assert metrics["negative_buy_hold_year_win_rate"] == 1.0
    assert metrics["aggregate_active_log_edge_in_negative_buy_hold_years"] > 0.01
    assert metrics["brier_score"] < metrics["causal_training_climatology_brier_score"]
    assert metrics["strategy_no_leverage_proof"]["proof_tolerance"] == 0.0
    assert metrics["buy_hold_no_leverage_proof"]["all_long_buy_hold_target"] is True
    # This is also the artifact-facing guarantee that no NumPy scalar or NaN
    # survives the scorer.
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_gate_threshold_ties_pass_brier_tie_fails_and_dust_is_not_a_win():
    metrics = _score(_synthetic_inputs())
    threshold_metrics = copy.deepcopy(metrics)
    threshold_metrics.update(
        {
            "total_active_log_edge": DEVELOPMENT_GATES[
                "minimum_total_active_log_edge"
            ],
            "median_annual_active_log_edge": DEVELOPMENT_GATES[
                "minimum_median_annual_active_log_edge"
            ],
            "active_log_edge_without_best_year": DEVELOPMENT_GATES[
                "minimum_active_log_edge_without_best_year"
            ],
            "rolling_252_session_month_end_win_rate": DEVELOPMENT_GATES[
                "minimum_252_session_month_end_win_rate"
            ],
            "rolling_756_session_month_end_win_rate": DEVELOPMENT_GATES[
                "minimum_756_session_month_end_win_rate"
            ],
            "annual_win_rate": DEVELOPMENT_GATES["minimum_annual_win_rate"],
            "positive_folds": DEVELOPMENT_GATES["minimum_positive_folds"],
            "cash_days": DEVELOPMENT_GATES["minimum_cash_days"],
            "cash_episodes": DEVELOPMENT_GATES["minimum_cash_episodes"],
            "cash_day_rate": DEVELOPMENT_GATES["maximum_cash_day_rate"],
            "largest_positive_year_share": DEVELOPMENT_GATES[
                "maximum_largest_positive_year_share"
            ],
            "aggregate_active_log_edge_in_negative_buy_hold_years": DEVELOPMENT_GATES[
                "minimum_negative_buy_hold_year_active_log_edge"
            ],
            "negative_buy_hold_year_win_rate": DEVELOPMENT_GATES[
                "minimum_negative_buy_hold_year_win_rate"
            ],
        }
    )
    assert apply_downside_gates(threshold_metrics)["passed"] is True

    threshold_metrics["causal_training_climatology_brier_score"] = threshold_metrics[
        "brier_score"
    ]
    tied = apply_downside_gates(threshold_metrics)
    assert tied["checks"]["brier_beats_causal_training_climatology"] is False
    assert tied["passed"] is False

    threshold_metrics["causal_training_climatology_brier_score"] = (
        threshold_metrics["brier_score"] + ACTIVE_EDGE_WIN_TOLERANCE / 2.0
    )
    dust_improvement = apply_downside_gates(threshold_metrics)
    assert (
        dust_improvement["checks"]["brier_beats_causal_training_climatology"]
        is False
    )

    index = pd.bdate_range("2010-01-01", periods=800)
    dust = pd.Series(0.0, index=index)
    dust.iloc[300] = ACTIVE_EDGE_WIN_TOLERANCE / 10.0
    win_rate, observations = month_end_rolling_win_rate(dust, 252)
    assert observations > 0
    assert win_rate == 0.0


@pytest.mark.parametrize(
    ("ledger_name", "column", "bad_value", "message"),
    [
        ("strategy", "new_exposure_after_fill", 1.0 + 1e-12, "strict no-leverage"),
        ("strategy", "cash", -1e-12, "strict no-leverage"),
        ("buy_hold", "holding_exposure_for_return", 1.0 + 1e-12, "strict no-leverage"),
        ("buy_hold", "cash", -1e-12, "strict no-leverage"),
    ],
)
def test_no_leverage_is_exact_for_strategy_and_buy_hold(
    ledger_name, column, bad_value, message
):
    inputs = list(_synthetic_inputs())
    ledger_index = 0 if ledger_name == "strategy" else 1
    inputs[ledger_index] = inputs[ledger_index].copy()
    inputs[ledger_index].loc[0, column] = bad_value

    with pytest.raises(DownsideScoringError, match=message):
        _score(tuple(inputs))


def test_binary_target_must_match_strategy_and_buy_hold_must_be_all_long():
    inputs = list(_synthetic_inputs())
    nonbinary = inputs[2].copy()
    nonbinary[10] = 0.5
    inputs[2] = nonbinary
    with pytest.raises(DownsideScoringError, match="exactly 0 or 1"):
        _score(tuple(inputs))

    inputs = list(_synthetic_inputs())
    mismatched = inputs[2].copy()
    mismatched[10] = 1.0 - mismatched[10]
    inputs[2] = mismatched
    with pytest.raises(DownsideScoringError, match="does not exactly match"):
        _score(tuple(inputs))

    inputs = list(_synthetic_inputs())
    buy_hold = inputs[1].copy()
    buy_hold.loc[
        10,
        ["target_exposure", "new_exposure_after_fill", "holding_exposure_for_return"],
    ] = 0.0
    buy_hold.loc[10, "cash"] = 1000.0
    buy_hold.loc[10, "shares"] = 0.0
    inputs[1] = buy_hold
    with pytest.raises(DownsideScoringError, match="100% AAPL"):
        _score(tuple(inputs))


def test_negative_buy_hold_gate_fails_when_no_negative_year_exists():
    metrics = _score(_synthetic_inputs(negative_buy_hold_years=()))
    gates = apply_downside_gates(metrics)

    assert metrics["negative_buy_hold_years"] == []
    assert metrics["aggregate_active_log_edge_in_negative_buy_hold_years"] == 0.0
    assert metrics["negative_buy_hold_year_win_rate"] is None
    assert gates["checks"]["negative_buy_hold_aggregate_edge"] is False
    assert gates["checks"]["negative_buy_hold_year_win_rate"] is False
    assert gates["passed"] is False


def test_cash_activity_counts_initial_and_separated_episodes_exactly():
    inputs = list(_synthetic_inputs())
    strategy = inputs[0].copy()
    target = np.ones(len(strategy), dtype=float)
    target[0:3] = 0.0
    target[10:12] = 0.0
    target[20] = 0.0
    _replace_strategy_target(strategy, target)
    inputs[0] = strategy
    inputs[2] = target

    metrics = _score(tuple(inputs))

    assert metrics["cash_days"] == 6
    assert metrics["cash_episodes"] == 3
    assert apply_downside_gates(metrics)["checks"]["minimum_cash_days"] is False


def test_brier_is_strict_and_allows_only_a_trailing_unmatured_label_suffix():
    inputs = list(_synthetic_inputs())
    labels = inputs[4].copy()
    labels[-6:] = np.nan
    inputs[4] = labels
    metrics = _score(tuple(inputs))
    assert metrics["brier_rows"] == len(labels) - 6
    assert metrics["unmatured_trailing_label_rows"] == 6

    tied = list(_synthetic_inputs())
    tied[3] = tied[5].copy()
    tied_metrics = _score(tuple(tied))
    tied_gates = apply_downside_gates(tied_metrics)
    assert tied_metrics["brier_score"] == tied_metrics[
        "causal_training_climatology_brier_score"
    ]
    assert tied_gates["checks"]["brier_beats_causal_training_climatology"] is False

    internal_gap = list(_synthetic_inputs())
    gap_labels = internal_gap[4].copy()
    gap_labels[100] = np.nan
    internal_gap[4] = gap_labels
    with pytest.raises(DownsideScoringError, match="trailing suffix"):
        _score(tuple(internal_gap))


def test_session_alignment_returns_and_cost_fail_closed():
    inputs = list(_synthetic_inputs())
    buy_hold = inputs[1].copy()
    buy_hold.loc[10, "fill_date"] = "2006-01-01"
    inputs[1] = buy_hold
    with pytest.raises(DownsideScoringError, match="strictly increasing|same fill sessions"):
        _score(tuple(inputs))

    inputs = list(_synthetic_inputs())
    strategy = inputs[0].copy()
    strategy.loc[10, "daily_return"] = np.nan
    inputs[0] = strategy
    with pytest.raises(DownsideScoringError, match="non-finite"):
        _score(tuple(inputs))

    with pytest.raises(DownsideScoringError, match="at least 0.0"):
        _score(_synthetic_inputs(), cost_bps=-0.01)
