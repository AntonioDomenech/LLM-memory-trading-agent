"""Deterministic development scoring for the frozen downside ensemble.

This module deliberately knows nothing about the walk-forward implementation.
Callers provide two already-simulated, same-session ledgers, the binary target
that is represented in the strategy ledger, and aligned out-of-fold
probability diagnostics.  The ledgers must cover fill years 2005 through 2018.

All return comparisons use active log increments.  A positive year, fold, or
rolling window must exceed ``ACTIVE_EDGE_WIN_TOLERANCE`` so floating-point dust
cannot turn a tie into a win.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .unleveraged_aapl import assert_unleveraged_ledger


DEVELOPMENT_FIRST_YEAR = 2005
DEVELOPMENT_LAST_YEAR = 2018
DEVELOPMENT_YEARS = tuple(range(DEVELOPMENT_FIRST_YEAR, DEVELOPMENT_LAST_YEAR + 1))
DEVELOPMENT_FOLDS: tuple[tuple[str, int, int], ...] = (
    ("2005_2006", 2005, 2006),
    ("2007_2008", 2007, 2008),
    ("2009_2010", 2009, 2010),
    ("2011_2012", 2011, 2012),
    ("2013_2014", 2013, 2014),
    ("2015_2016", 2015, 2016),
    ("2017_2018", 2017, 2018),
)

ACTIVE_EDGE_WIN_TOLERANCE = 1e-12

DEVELOPMENT_GATES: Mapping[str, float | int] = {
    "minimum_total_active_log_edge": 0.02,
    "minimum_median_annual_active_log_edge": 0.0005,
    "minimum_active_log_edge_without_best_year": 0.005,
    "minimum_252_session_month_end_win_rate": 0.60,
    "minimum_756_session_month_end_win_rate": 0.70,
    "minimum_annual_win_rate": 0.55,
    "minimum_positive_folds": 5,
    "minimum_cash_days": 30,
    "minimum_cash_episodes": 12,
    "maximum_cash_day_rate": 0.20,
    "maximum_largest_positive_year_share": 0.45,
    "minimum_negative_buy_hold_year_active_log_edge": 0.01,
    "minimum_negative_buy_hold_year_win_rate": 0.60,
}


class DownsideScoringError(ValueError):
    """Raised when a scoring input violates the sealed development contract."""


def _finite_float(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise DownsideScoringError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DownsideScoringError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise DownsideScoringError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise DownsideScoringError(f"{name} must be at least {minimum}")
    return result


def _one_dimensional_numeric(values: Sequence[float], *, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DownsideScoringError(f"{name} must be numeric") from exc
    if result.ndim != 1:
        raise DownsideScoringError(f"{name} must be one-dimensional")
    if not len(result):
        raise DownsideScoringError(f"{name} cannot be empty")
    return result


def _strict_binary(values: Sequence[float], *, name: str) -> np.ndarray:
    result = _one_dimensional_numeric(values, name=name)
    if not np.isfinite(result).all():
        raise DownsideScoringError(f"{name} must contain only finite binary values")
    if not np.isin(result, (0.0, 1.0)).all():
        raise DownsideScoringError(f"{name} must contain exactly 0 or 1")
    return result


def _fill_dates(ledger: pd.DataFrame, *, name: str) -> pd.DatetimeIndex:
    if "fill_date" not in ledger.columns:
        raise DownsideScoringError(f"{name} ledger lacks fill_date")
    try:
        parsed = pd.to_datetime(ledger["fill_date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise DownsideScoringError(f"{name} ledger has an invalid fill_date") from exc
    dates = pd.DatetimeIndex(parsed)
    if dates.tz is not None:
        raise DownsideScoringError(f"{name} fill dates must be timezone-naive sessions")
    if dates.hasnans:
        raise DownsideScoringError(f"{name} ledger has a missing fill_date")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise DownsideScoringError(
            f"{name} fill dates must be unique and strictly increasing"
        )
    if not dates.equals(dates.normalize()):
        raise DownsideScoringError(f"{name} fill dates must not contain a time component")
    observed_years = tuple(sorted(set(int(value) for value in dates.year)))
    if observed_years != DEVELOPMENT_YEARS:
        raise DownsideScoringError(
            "Ledgers must contain every fill year from 2005 through 2018 and no others"
        )
    return dates


def _strict_no_leverage_proof(
    ledger: pd.DataFrame,
    *,
    name: str,
    expected_target: np.ndarray | None,
    require_all_long: bool,
) -> dict[str, Any]:
    """Use the shared proof with zero numerical tolerance, then bind targets."""

    try:
        proof = assert_unleveraged_ledger(ledger, tolerance=0.0)
    except (ValueError, RuntimeError) as exc:
        raise DownsideScoringError(f"{name} ledger failed strict no-leverage proof: {exc}") from exc

    ledger_target = _strict_binary(
        ledger["target_exposure"].to_numpy(),
        name=f"{name} ledger target_exposure",
    )
    if expected_target is not None and not np.array_equal(ledger_target, expected_target):
        raise DownsideScoringError(
            "Binary decision target does not exactly match strategy target_exposure"
        )
    if require_all_long and not np.array_equal(
        ledger_target, np.ones(len(ledger_target), dtype=float)
    ):
        raise DownsideScoringError(
            "Same-session buy-and-hold benchmark must target 100% AAPL on every row"
        )
    # ``assert_unleveraged_ledger`` has already checked every proof column with
    # tolerance zero.  These explicit booleans make the serialized claim easy
    # for an artifact verifier to audit without reinterpreting tolerances.
    return {
        **proof,
        "proof_tolerance": 0.0,
        "exact_binary_targets": True,
        "all_long_buy_hold_target": bool(require_all_long),
    }


def _active_log_series(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    strategy_dates = _fill_dates(strategy, name="strategy")
    benchmark_dates = _fill_dates(benchmark, name="buy_hold")
    if not strategy_dates.equals(benchmark_dates):
        raise DownsideScoringError(
            "Strategy and buy-and-hold ledgers must contain the same fill sessions"
        )
    for ledger, name in ((strategy, "strategy"), (benchmark, "buy_hold")):
        if "daily_return" not in ledger.columns:
            raise DownsideScoringError(f"{name} ledger lacks daily_return")
    strategy_returns = pd.to_numeric(
        strategy["daily_return"], errors="raise"
    ).to_numpy(dtype=float)
    benchmark_returns = pd.to_numeric(
        benchmark["daily_return"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(strategy_returns).all() or not np.isfinite(
        benchmark_returns
    ).all():
        raise DownsideScoringError("A ledger contains a non-finite daily return")
    if np.any(strategy_returns <= -1.0) or np.any(benchmark_returns <= -1.0):
        raise DownsideScoringError("A daily return is invalid for logarithmic scoring")
    active_log = pd.Series(
        np.log1p(strategy_returns) - np.log1p(benchmark_returns),
        index=strategy_dates,
        name="active_log_increment",
        dtype=float,
    )
    benchmark_log = pd.Series(
        np.log1p(benchmark_returns),
        index=benchmark_dates,
        name="buy_hold_log_increment",
        dtype=float,
    )
    return active_log, benchmark_log


def month_end_rolling_win_rate(
    active_log: pd.Series,
    sessions: int,
) -> tuple[float | None, int]:
    """Return the last observable rolling-window result in each calendar month."""

    if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < 1:
        raise DownsideScoringError("sessions must be a positive integer")
    numeric = pd.to_numeric(active_log, errors="raise").astype(float)
    if not isinstance(numeric.index, pd.DatetimeIndex):
        raise DownsideScoringError("active_log must use a DatetimeIndex")
    if not np.isfinite(numeric.to_numpy()).all():
        raise DownsideScoringError("active_log contains a non-finite value")
    rolling = numeric.rolling(sessions, min_periods=sessions).sum()
    month_end = rolling.groupby(rolling.index.to_period("M")).tail(1).dropna()
    if month_end.empty:
        return None, 0
    return (
        float((month_end > ACTIVE_EDGE_WIN_TOLERANCE).mean()),
        int(len(month_end)),
    )


def _brier_metrics(
    downside_probability: Sequence[float],
    downside_label: Sequence[float],
    causal_training_climatology_probability: Sequence[float],
) -> dict[str, Any]:
    predicted = _one_dimensional_numeric(
        downside_probability, name="downside_probability"
    )
    labels = _one_dimensional_numeric(downside_label, name="downside_label")
    baseline = _one_dimensional_numeric(
        causal_training_climatology_probability,
        name="causal_training_climatology_probability",
    )
    if not (len(predicted) == len(labels) == len(baseline)):
        raise DownsideScoringError(
            "Brier inputs must contain the same number of out-of-fold rows"
        )
    if not np.isfinite(predicted).all() or not np.isfinite(baseline).all():
        raise DownsideScoringError("Brier probabilities must be finite on every row")
    if ((predicted < 0.0) | (predicted > 1.0)).any() or (
        (baseline < 0.0) | (baseline > 1.0)
    ).any():
        raise DownsideScoringError("Brier probabilities must lie in [0, 1]")
    if np.isinf(labels).any():
        raise DownsideScoringError("downside_label cannot contain infinity")
    eligible = np.isfinite(labels)
    if not eligible.any():
        raise DownsideScoringError("At least one causally mature downside label is required")
    missing_positions = np.flatnonzero(~eligible)
    if len(missing_positions):
        first_missing = int(missing_positions[0])
        if eligible[first_missing:].any():
            raise DownsideScoringError(
                "Missing downside labels are allowed only as an unmatured trailing suffix"
            )
    eligible_labels = labels[eligible]
    if not np.isin(eligible_labels, (0.0, 1.0)).all():
        raise DownsideScoringError("Mature downside labels must contain exactly 0 or 1")
    model_brier = float(np.mean(np.square(predicted[eligible] - eligible_labels)))
    baseline_brier = float(np.mean(np.square(baseline[eligible] - eligible_labels)))
    return {
        "brier_score": model_brier,
        "causal_training_climatology_brier_score": baseline_brier,
        "brier_improvement": float(baseline_brier - model_brier),
        "brier_rows": int(eligible.sum()),
        "unmatured_trailing_label_rows": int((~eligible).sum()),
    }


def score_downside_metrics(
    strategy: pd.DataFrame,
    buy_hold: pd.DataFrame,
    decision_target: Sequence[float],
    downside_probability: Sequence[float],
    downside_label: Sequence[float],
    causal_training_climatology_probability: Sequence[float],
    *,
    cost_bps: float,
) -> dict[str, Any]:
    """Score one continuous 2005-2018 strategy against same-session AAPL.

    ``decision_target`` is the executed target aligned one-for-one with the
    strategy ledger rows.  Brier inputs are a separate aligned out-of-fold
    sequence; a trailing NaN label suffix is permitted for observations whose
    forward outcome has not matured inside the bounded snapshot.
    """

    supplied_cost = _finite_float(cost_bps, name="cost_bps", minimum=0.0)
    target = _strict_binary(decision_target, name="decision_target")
    if len(strategy) != len(buy_hold) or len(target) != len(strategy):
        raise DownsideScoringError(
            "Strategy, buy-and-hold, and decision target must have equal lengths"
        )
    strategy_proof = _strict_no_leverage_proof(
        strategy,
        name="strategy",
        expected_target=target,
        require_all_long=False,
    )
    buy_hold_proof = _strict_no_leverage_proof(
        buy_hold,
        name="buy_hold",
        expected_target=None,
        require_all_long=True,
    )
    active_log, buy_hold_log = _active_log_series(strategy, buy_hold)

    annual_active_series = active_log.groupby(active_log.index.year).sum()
    annual_buy_hold_log_series = buy_hold_log.groupby(buy_hold_log.index.year).sum()
    annual_active = {
        str(year): float(annual_active_series.get(year, 0.0))
        for year in DEVELOPMENT_YEARS
    }
    annual_buy_hold_log = {
        str(year): float(annual_buy_hold_log_series.get(year, 0.0))
        for year in DEVELOPMENT_YEARS
    }
    annual_buy_hold_return = {
        year: float(math.expm1(value)) for year, value in annual_buy_hold_log.items()
    }
    annual_values = np.asarray(list(annual_active.values()), dtype=float)
    total_active_log_edge = float(active_log.sum())
    best_year = float(annual_values.max())
    positive_annual = annual_values[annual_values > ACTIVE_EDGE_WIN_TOLERANCE]
    positive_sum = float(positive_annual.sum()) if len(positive_annual) else 0.0
    largest_positive_share = (
        float(positive_annual.max() / positive_sum) if positive_sum > 0.0 else None
    )

    fold_edges: dict[str, float] = {}
    fold_session_counts: dict[str, int] = {}
    for fold_name, first_year, last_year in DEVELOPMENT_FOLDS:
        mask = (active_log.index.year >= first_year) & (
            active_log.index.year <= last_year
        )
        if not mask.any():
            raise DownsideScoringError(f"Development fold {fold_name} has no sessions")
        fold_edges[fold_name] = float(active_log.loc[mask].sum())
        fold_session_counts[fold_name] = int(mask.sum())

    rolling_252, rolling_252_count = month_end_rolling_win_rate(active_log, 252)
    rolling_756, rolling_756_count = month_end_rolling_win_rate(active_log, 756)

    cash = target == 0.0
    prior_cash = np.r_[False, cash[:-1]]
    cash_episodes = cash & ~prior_cash

    negative_years = [
        str(year)
        for year in DEVELOPMENT_YEARS
        if annual_buy_hold_log[str(year)] < -ACTIVE_EDGE_WIN_TOLERANCE
    ]
    negative_year_edges = {year: annual_active[year] for year in negative_years}
    negative_aggregate = float(sum(negative_year_edges.values()))
    negative_win_rate = (
        float(
            np.mean(
                np.asarray(list(negative_year_edges.values()), dtype=float)
                > ACTIVE_EDGE_WIN_TOLERANCE
            )
        )
        if negative_year_edges
        else None
    )

    brier = _brier_metrics(
        downside_probability,
        downside_label,
        causal_training_climatology_probability,
    )
    metrics: dict[str, Any] = {
        "scoring_contract": "aapl-downside-ensemble-development-v1",
        "caller_supplied_cost_bps": supplied_cost,
        "active_edge_win_tolerance": ACTIVE_EDGE_WIN_TOLERANCE,
        "first_fill_year": DEVELOPMENT_FIRST_YEAR,
        "last_fill_year": DEVELOPMENT_LAST_YEAR,
        "sessions": int(len(active_log)),
        "total_active_log_edge": total_active_log_edge,
        "relative_wealth_vs_buy_hold": float(math.expm1(total_active_log_edge)),
        "annual_active_log_edges": annual_active,
        "annual_buy_hold_log_returns": annual_buy_hold_log,
        "annual_buy_hold_returns": annual_buy_hold_return,
        "annual_win_rate": float(
            np.mean(annual_values > ACTIVE_EDGE_WIN_TOLERANCE)
        ),
        "median_annual_active_log_edge": float(np.median(annual_values)),
        "best_annual_active_log_edge": best_year,
        "active_log_edge_without_best_year": float(total_active_log_edge - best_year),
        "largest_positive_year_share": largest_positive_share,
        "fold_active_log_edges": fold_edges,
        "fold_session_counts": fold_session_counts,
        "positive_folds": int(
            sum(value > ACTIVE_EDGE_WIN_TOLERANCE for value in fold_edges.values())
        ),
        "minimum_fold_active_log_edge": float(min(fold_edges.values())),
        "rolling_252_session_month_end_win_rate": rolling_252,
        "rolling_252_session_month_end_observations": rolling_252_count,
        "rolling_756_session_month_end_win_rate": rolling_756,
        "rolling_756_session_month_end_observations": rolling_756_count,
        "cash_days": int(cash.sum()),
        "cash_episodes": int(cash_episodes.sum()),
        "cash_day_rate": float(cash.mean()),
        "negative_buy_hold_years": negative_years,
        "negative_buy_hold_year_active_log_edges": negative_year_edges,
        "aggregate_active_log_edge_in_negative_buy_hold_years": negative_aggregate,
        "negative_buy_hold_year_win_rate": negative_win_rate,
        "strategy_no_leverage_proof": strategy_proof,
        "buy_hold_no_leverage_proof": buy_hold_proof,
        "exact_binary_decision_target": True,
        **brier,
    }
    # Fail here rather than allowing a NaN/NumPy scalar to leak into an
    # artifact that only fails when its manifest is serialized later.
    try:
        json.dumps(metrics, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise DownsideScoringError("Metrics are not canonical JSON-compatible") from exc
    return metrics


def apply_downside_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the complete frozen development gate contract by name."""

    contract = DEVELOPMENT_GATES
    largest_share = metrics.get("largest_positive_year_share")
    negative_win_rate = metrics.get("negative_buy_hold_year_win_rate")
    checks = {
        "material_total_active_log_edge": (
            float(metrics["total_active_log_edge"])
            >= contract["minimum_total_active_log_edge"]
        ),
        "material_median_annual_edge": (
            float(metrics["median_annual_active_log_edge"])
            >= contract["minimum_median_annual_active_log_edge"]
        ),
        "material_without_best_year": (
            float(metrics["active_log_edge_without_best_year"])
            >= contract["minimum_active_log_edge_without_best_year"]
        ),
        "rolling_252_win_rate": (
            metrics.get("rolling_252_session_month_end_win_rate") is not None
            and float(metrics["rolling_252_session_month_end_win_rate"])
            >= contract["minimum_252_session_month_end_win_rate"]
        ),
        "rolling_756_win_rate": (
            metrics.get("rolling_756_session_month_end_win_rate") is not None
            and float(metrics["rolling_756_session_month_end_win_rate"])
            >= contract["minimum_756_session_month_end_win_rate"]
        ),
        "annual_win_rate": (
            float(metrics["annual_win_rate"])
            >= contract["minimum_annual_win_rate"]
        ),
        "positive_fold_count": (
            int(metrics["positive_folds"]) >= contract["minimum_positive_folds"]
        ),
        "minimum_cash_days": (
            int(metrics["cash_days"]) >= contract["minimum_cash_days"]
        ),
        "minimum_cash_episodes": (
            int(metrics["cash_episodes"]) >= contract["minimum_cash_episodes"]
        ),
        "maximum_cash_day_rate": (
            float(metrics["cash_day_rate"]) <= contract["maximum_cash_day_rate"]
        ),
        "largest_positive_year_share": (
            largest_share is not None
            and float(largest_share) <= contract["maximum_largest_positive_year_share"]
        ),
        "negative_buy_hold_aggregate_edge": (
            float(metrics["aggregate_active_log_edge_in_negative_buy_hold_years"])
            >= contract["minimum_negative_buy_hold_year_active_log_edge"]
        ),
        "negative_buy_hold_year_win_rate": (
            negative_win_rate is not None
            and float(negative_win_rate)
            >= contract["minimum_negative_buy_hold_year_win_rate"]
        ),
        "brier_beats_causal_training_climatology": (
            float(metrics["brier_score"])
            < float(metrics["causal_training_climatology_brier_score"])
            - ACTIVE_EDGE_WIN_TOLERANCE
        ),
        "strict_no_leverage_strategy": bool(
            (metrics.get("strategy_no_leverage_proof") or {}).get("passed")
        ),
        "strict_no_leverage_buy_hold": bool(
            (metrics.get("buy_hold_no_leverage_proof") or {}).get("passed")
        ),
        "exact_binary_decision_target": bool(
            metrics.get("exact_binary_decision_target")
        ),
    }
    gate = {
        "contract": dict(contract),
        "checks": {name: bool(value) for name, value in checks.items()},
        "passed": bool(all(checks.values())),
    }
    json.dumps(gate, sort_keys=True, allow_nan=False)
    return gate


def evaluate_downside_development(
    strategy: pd.DataFrame,
    buy_hold: pd.DataFrame,
    decision_target: Sequence[float],
    downside_probability: Sequence[float],
    downside_label: Sequence[float],
    causal_training_climatology_probability: Sequence[float],
    *,
    cost_bps: float,
) -> dict[str, Any]:
    """Return JSON-friendly development metrics and their named gate results."""

    metrics = score_downside_metrics(
        strategy,
        buy_hold,
        decision_target,
        downside_probability,
        downside_label,
        causal_training_climatology_probability,
        cost_bps=cost_bps,
    )
    result = {
        "metrics": metrics,
        "gates": apply_downside_gates(metrics),
    }
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


__all__ = [
    "ACTIVE_EDGE_WIN_TOLERANCE",
    "DEVELOPMENT_FIRST_YEAR",
    "DEVELOPMENT_FOLDS",
    "DEVELOPMENT_GATES",
    "DEVELOPMENT_LAST_YEAR",
    "DownsideScoringError",
    "apply_downside_gates",
    "evaluate_downside_development",
    "month_end_rolling_win_rate",
    "score_downside_metrics",
]
