"""Development scoring for the direct five-session AAPL CASH-edge model.

The ledger and robustness contract is intentionally inherited from
``downside_scoring``.  Only the predictive semantics change: the probability
is the chance that CASH beats LONG after 10 bps of slippage on each trade leg,
and the continuous prediction is that same realized active log edge.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .downside_scoring import (
    ACTIVE_EDGE_WIN_TOLERANCE,
    DEVELOPMENT_GATES,
    DownsideScoringError,
    apply_downside_gates,
    score_downside_metrics,
)


DIRECT_EDGE_GATES: Mapping[str, float] = {
    "minimum_expected_edge_mae_relative_improvement": 0.01,
    "minimum_episode_start_realized_edge_mean_exclusive": 0.0,
    "minimum_episode_start_realized_edge_win_rate_exclusive": 0.50,
}

_CASH_BLOCK_ROWS = 5
_TRADE_TOLERANCE = 1e-12
_NUMERIC_ABSOLUTE_TOLERANCE = 1e-10
_NUMERIC_RELATIVE_TOLERANCE = 1e-12


class DirectEdgeScoringError(DownsideScoringError):
    """Raised when direct-edge predictive inputs violate their contract."""


def _numeric_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DirectEdgeScoringError(f"{name} must be numeric") from exc
    if result.ndim != 1:
        raise DirectEdgeScoringError(f"{name} must be one-dimensional")
    if not len(result):
        raise DirectEdgeScoringError(f"{name} cannot be empty")
    return result


def _finite_probability(values: Sequence[float], *, name: str) -> np.ndarray:
    result = _numeric_vector(values, name=name)
    if not np.isfinite(result).all():
        raise DirectEdgeScoringError(f"{name} must be finite on every OOF row")
    if ((result < 0.0) | (result > 1.0)).any():
        raise DirectEdgeScoringError(f"{name} must lie in [0, 1]")
    return result


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    result = _numeric_vector(values, name=name)
    if not np.isfinite(result).all():
        raise DirectEdgeScoringError(f"{name} must be finite on every OOF row")
    return result


def _strict_binary(values: Sequence[float], *, name: str) -> np.ndarray:
    result = _finite_vector(values, name=name)
    if not np.isin(result, (0.0, 1.0)).all():
        raise DirectEdgeScoringError(f"{name} must contain exactly 0 or 1")
    return result


def _mature_suffix_mask(values: np.ndarray, *, name: str) -> np.ndarray:
    if np.isinf(values).any():
        raise DirectEdgeScoringError(f"{name} cannot contain infinity")
    mature = np.isfinite(values)
    if not mature.any():
        raise DirectEdgeScoringError(f"{name} requires at least one mature row")
    missing = np.flatnonzero(~mature)
    if len(missing) and mature[int(missing[0]) :].any():
        raise DirectEdgeScoringError(
            f"Missing {name} values are allowed only as an unmatured trailing suffix"
        )
    return mature


def _bind_oof_cash_to_executed_ledger(
    strategy: pd.DataFrame,
    oof_cash_target: np.ndarray,
    oof_decision_dates: Sequence[object],
) -> dict[str, int | bool]:
    """Prove that prediction-side CASH signals are the executed strategy.

    The final OOF decision can fall after the last fill in a bounded period.
    Such unmatched decisions are allowed only as one chronological trailing
    suffix after the ledger's final decision; every overlapping decision must
    equal ``1 - target_exposure`` exactly.
    """

    if "decision_date" not in strategy.columns:
        raise DirectEdgeScoringError(
            "strategy ledger lacks decision_date for OOF target binding"
        )
    try:
        oof_dates = pd.DatetimeIndex(pd.to_datetime(oof_decision_dates, errors="raise"))
        ledger_dates = pd.DatetimeIndex(
            pd.to_datetime(strategy["decision_date"], errors="raise")
        )
    except (TypeError, ValueError) as exc:
        raise DirectEdgeScoringError(
            "OOF and ledger decision dates must be valid dates"
        ) from exc
    if oof_dates.tz is not None or ledger_dates.tz is not None:
        raise DirectEdgeScoringError("OOF and ledger decision dates must be timezone-naive")
    oof_dates = oof_dates.normalize()
    ledger_dates = ledger_dates.normalize()
    if len(oof_dates) != len(oof_cash_target):
        raise DirectEdgeScoringError(
            "oof_decision_dates must align with oof_cash_target"
        )
    if (
        oof_dates.has_duplicates
        or not oof_dates.is_monotonic_increasing
        or ledger_dates.has_duplicates
        or not ledger_dates.is_monotonic_increasing
    ):
        raise DirectEdgeScoringError(
            "OOF and ledger decision dates must be unique and chronological"
        )

    ledger_target = _strict_binary(
        strategy["target_exposure"].to_numpy(),
        name="strategy ledger target_exposure",
    )
    executed_cash_by_date = {
        timestamp: 1.0 - float(target)
        for timestamp, target in zip(ledger_dates, ledger_target, strict=True)
    }
    matched = np.asarray(
        [timestamp in executed_cash_by_date for timestamp in oof_dates],
        dtype=bool,
    )
    if not matched.any():
        raise DirectEdgeScoringError(
            "No OOF decision date overlaps the executed strategy ledger"
        )
    missing_positions = np.flatnonzero(~matched)
    if len(missing_positions):
        first_missing = int(missing_positions[0])
        if matched[first_missing:].any() or not bool(
            (oof_dates[~matched] > ledger_dates.max()).all()
        ):
            raise DirectEdgeScoringError(
                "Unmatched OOF decisions are allowed only as a trailing suffix "
                "after the final executed ledger decision"
            )
    expected = np.asarray(
        [executed_cash_by_date[timestamp] for timestamp in oof_dates[matched]],
        dtype=float,
    )
    if not np.array_equal(oof_cash_target[matched], expected):
        raise DirectEdgeScoringError(
            "OOF CASH target does not match the executed strategy by decision date"
        )
    return {
        "oof_executed_decision_matches": int(matched.sum()),
        "oof_unexecuted_trailing_decisions": int((~matched).sum()),
        "oof_cash_target_bound_to_executed_ledger": True,
    }


def _bind_ledger_cost_bps(
    ledger: pd.DataFrame,
    *,
    name: str,
    cost_bps: float,
) -> dict[str, Any]:
    """Verify the caller's cost label against fills, slippage, and returns."""

    if isinstance(cost_bps, bool):
        raise DirectEdgeScoringError("cost_bps must be a finite number")
    try:
        bps = float(cost_bps)
    except (TypeError, ValueError) as exc:
        raise DirectEdgeScoringError("cost_bps must be a finite number") from exc
    if not math.isfinite(bps) or bps < 0.0 or bps >= 10_000.0:
        raise DirectEdgeScoringError(
            "cost_bps must be finite and lie in [0, 10,000)"
        )

    required = {
        "signed_share_delta",
        "reference_price",
        "fill_price",
        "slippage",
        "fees",
        "trade_executed",
        "shares",
        "equity",
        "equity_before_fill",
        "daily_return",
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise DirectEdgeScoringError(
            f"{name} ledger lacks exact cost proof columns: {missing}"
        )

    numeric: dict[str, np.ndarray] = {}
    for column in sorted(required.difference({"trade_executed"})):
        try:
            values = pd.to_numeric(ledger[column], errors="raise").to_numpy(
                dtype=float
            )
        except (TypeError, ValueError) as exc:
            raise DirectEdgeScoringError(
                f"{name} ledger {column} must be numeric"
            ) from exc
        if not np.isfinite(values).all():
            raise DirectEdgeScoringError(
                f"{name} ledger {column} must be finite"
            )
        numeric[column] = values

    reference = numeric["reference_price"]
    fill = numeric["fill_price"]
    delta = numeric["signed_share_delta"]
    if (reference <= 0.0).any() or (fill <= 0.0).any():
        raise DirectEdgeScoringError(
            f"{name} ledger contains a non-positive execution price"
        )
    trade = np.abs(delta) > _TRADE_TOLERANCE
    raw_trade = ledger["trade_executed"].to_numpy()
    if any(not isinstance(value, (bool, np.bool_)) for value in raw_trade):
        raise DirectEdgeScoringError(
            f"{name} ledger trade_executed must contain booleans"
        )
    if not np.array_equal(raw_trade.astype(bool), trade):
        raise DirectEdgeScoringError(
            f"{name} ledger trade_executed does not match signed_share_delta"
        )

    rate = bps / 10_000.0
    expected_fill = reference.copy()
    expected_fill[delta > _TRADE_TOLERANCE] = (
        reference[delta > _TRADE_TOLERANCE] * (1.0 + rate)
    )
    expected_fill[delta < -_TRADE_TOLERANCE] = (
        reference[delta < -_TRADE_TOLERANCE] * (1.0 - rate)
    )
    if not np.allclose(
        fill,
        expected_fill,
        rtol=_NUMERIC_RELATIVE_TOLERANCE,
        atol=_NUMERIC_ABSOLUTE_TOLERANCE,
    ):
        raise DirectEdgeScoringError(
            f"{name} ledger fill prices do not prove the declared cost_bps"
        )
    expected_slippage = np.abs(delta) * np.abs(expected_fill - reference)
    if not np.allclose(
        numeric["slippage"],
        expected_slippage,
        rtol=_NUMERIC_RELATIVE_TOLERANCE,
        atol=_NUMERIC_ABSOLUTE_TOLERANCE,
    ):
        raise DirectEdgeScoringError(
            f"{name} ledger slippage does not match fills and share deltas"
        )
    if not np.allclose(
        numeric["fees"],
        0.0,
        rtol=0.0,
        atol=_NUMERIC_ABSOLUTE_TOLERANCE,
    ):
        raise DirectEdgeScoringError(
            f"{name} ledger must use the declared zero-commission contract"
        )

    expected_delta = np.diff(np.r_[0.0, numeric["shares"]])
    if not np.allclose(
        delta,
        expected_delta,
        rtol=_NUMERIC_RELATIVE_TOLERANCE,
        atol=_NUMERIC_ABSOLUTE_TOLERANCE,
    ):
        raise DirectEdgeScoringError(
            f"{name} ledger share deltas do not reconcile to holdings"
        )

    equity = numeric["equity"]
    if (equity <= 0.0).any() or numeric["equity_before_fill"][0] <= 0.0:
        raise DirectEdgeScoringError(
            f"{name} ledger contains non-positive equity"
        )
    expected_return = np.empty(len(equity), dtype=float)
    expected_return[0] = equity[0] / numeric["equity_before_fill"][0] - 1.0
    expected_return[1:] = equity[1:] / equity[:-1] - 1.0
    if not np.allclose(
        numeric["daily_return"],
        expected_return,
        rtol=_NUMERIC_RELATIVE_TOLERANCE,
        atol=_NUMERIC_ABSOLUTE_TOLERANCE,
    ):
        raise DirectEdgeScoringError(
            f"{name} ledger daily_return does not reconcile to equity"
        )

    return {
        "declared_cost_bps": bps,
        "cost_bps_bound_to_fills": True,
        "zero_commission_proof": True,
        "trade_rows": int(trade.sum()),
        "total_verified_slippage": float(numeric["slippage"].sum()),
        "daily_returns_reconciled_to_equity": True,
    }


def _validate_cash_block_starts(
    cash_target: np.ndarray,
    cash_block_start: Sequence[float],
) -> np.ndarray:
    starts = _strict_binary(cash_block_start, name="oof_cash_block_start")
    if len(starts) != len(cash_target):
        raise DirectEdgeScoringError(
            "oof_cash_block_start must align with oof_cash_target"
        )
    rebuilt = np.zeros(len(cash_target), dtype=float)
    active_until = 0
    for position in np.flatnonzero(starts == 1.0):
        position = int(position)
        if position < active_until:
            raise DirectEdgeScoringError(
                "OOF CASH block starts cannot overlap an active five-row block"
            )
        active_until = min(position + _CASH_BLOCK_ROWS, len(rebuilt))
        rebuilt[position:active_until] = 1.0
    if not np.array_equal(rebuilt, cash_target):
        raise DirectEdgeScoringError(
            "OOF CASH target is not exactly reconstructed by its five-row block starts"
        )
    return starts


def _predictive_metrics(
    cash_win_probability: Sequence[float],
    cash_win_label: Sequence[float],
    causal_training_mean_probability: Sequence[float],
    predicted_edge_10bps: Sequence[float],
    actual_edge_10bps: Sequence[float],
    causal_training_mean_edge: Sequence[float],
    oof_cash_target: Sequence[float],
    oof_cash_block_start: Sequence[float],
) -> dict[str, Any]:
    probability = _finite_probability(
        cash_win_probability,
        name="cash_win_probability",
    )
    label = _numeric_vector(cash_win_label, name="cash_win_label")
    baseline_probability = _finite_probability(
        causal_training_mean_probability,
        name="causal_training_mean_probability",
    )
    prediction = _finite_vector(
        predicted_edge_10bps,
        name="predicted_edge_10bps",
    )
    actual = _numeric_vector(actual_edge_10bps, name="actual_edge_10bps")
    baseline_edge = _finite_vector(
        causal_training_mean_edge,
        name="causal_training_mean_edge",
    )
    cash_target = _strict_binary(oof_cash_target, name="oof_cash_target")
    block_start = _validate_cash_block_starts(
        cash_target,
        oof_cash_block_start,
    )

    lengths = {
        len(probability),
        len(label),
        len(baseline_probability),
        len(prediction),
        len(actual),
        len(baseline_edge),
        len(cash_target),
        len(block_start),
    }
    if len(lengths) != 1:
        raise DirectEdgeScoringError(
            "All probability, edge, label, baseline, and OOF CASH inputs must align"
        )

    label_mature = _mature_suffix_mask(label, name="cash_win_label")
    edge_mature = _mature_suffix_mask(actual, name="actual_edge_10bps")
    if not np.array_equal(label_mature, edge_mature):
        raise DirectEdgeScoringError(
            "Binary and continuous direct-edge labels must have the same maturity mask"
        )
    mature_label = label[label_mature]
    if not np.isin(mature_label, (0.0, 1.0)).all():
        raise DirectEdgeScoringError(
            "Mature cash_win_label values must contain exactly 0 or 1"
        )
    implied_label = (actual[edge_mature] > 0.0).astype(float)
    if not np.array_equal(mature_label, implied_label):
        raise DirectEdgeScoringError(
            "cash_win_label must exactly equal actual_edge_10bps > 0"
        )

    model_mae = float(np.mean(np.abs(prediction[edge_mature] - actual[edge_mature])))
    baseline_mae = float(
        np.mean(np.abs(baseline_edge[edge_mature] - actual[edge_mature]))
    )
    absolute_improvement = float(baseline_mae - model_mae)
    relative_improvement = (
        float(absolute_improvement / baseline_mae)
        if baseline_mae > 0.0
        else None
    )

    cash = cash_target == 1.0
    episode_start = block_start == 1.0
    mature_episode_start = episode_start & edge_mature
    episode_edges = actual[mature_episode_start]
    episode_mean = float(np.mean(episode_edges)) if len(episode_edges) else None
    episode_wins = int(
        np.sum(episode_edges > ACTIVE_EDGE_WIN_TOLERANCE)
    )
    episode_win_rate = (
        float(episode_wins / len(episode_edges)) if len(episode_edges) else None
    )

    # Probability scoring itself is delegated to the base scorer.  These
    # values bind all other diagnostics to the same aligned OOF population.
    return {
        "predictive_target_semantics": "cash_beats_long_after_10bps_per_leg",
        "expected_edge_target_semantics": "cash_active_log_edge_10bps",
        "oof_rows": int(len(actual)),
        "mature_oof_rows": int(edge_mature.sum()),
        "unmatured_trailing_oof_rows": int((~edge_mature).sum()),
        "expected_edge_mae": model_mae,
        "causal_training_mean_edge_mae": baseline_mae,
        "expected_edge_mae_absolute_improvement": absolute_improvement,
        "expected_edge_mae_relative_improvement": relative_improvement,
        "oof_cash_days": int(cash.sum()),
        "oof_cash_episode_starts": int(episode_start.sum()),
        "oof_cash_block_starts": int(episode_start.sum()),
        "mature_oof_cash_episode_starts": int(mature_episode_start.sum()),
        "unmatured_oof_cash_episode_starts": int(
            (episode_start & ~edge_mature).sum()
        ),
        "episode_start_realized_edge_mean_10bps": episode_mean,
        "episode_start_realized_edge_wins_10bps": episode_wins,
        "episode_start_realized_edge_win_rate_10bps": episode_win_rate,
        "exact_binary_oof_cash_target": True,
        "exact_five_row_cash_block_reconstruction": True,
        "exact_binary_edge_label_consistency": True,
    }


def score_direct_edge_metrics(
    strategy: pd.DataFrame,
    buy_hold: pd.DataFrame,
    executed_binary_target: Sequence[float],
    cash_win_probability: Sequence[float],
    cash_win_label: Sequence[float],
    causal_training_mean_probability: Sequence[float],
    predicted_edge_10bps: Sequence[float],
    actual_edge_10bps: Sequence[float],
    causal_training_mean_edge: Sequence[float],
    oof_cash_target: Sequence[float],
    oof_cash_block_start: Sequence[float],
    oof_decision_dates: Sequence[object],
    *,
    cost_bps: float,
) -> dict[str, Any]:
    """Score one direct-edge candidate over the sealed 2005-2018 period.

    ``executed_binary_target`` is LONG=1/CASH=0 and binds to the strategy
    ledger.  ``oof_cash_target`` is the prediction-side inverse convention,
    CASH=1/LONG=0, aligned with all OOF probability and edge arrays.  The OOF
    population need not have the same length as the fill ledger, because a
    close decision executes at the following session's open.
    """

    try:
        metrics = score_downside_metrics(
            strategy,
            buy_hold,
            executed_binary_target,
            cash_win_probability,
            cash_win_label,
            causal_training_mean_probability,
            cost_bps=cost_bps,
        )
    except DownsideScoringError as exc:
        raise DirectEdgeScoringError(str(exc)) from exc

    predictive = _predictive_metrics(
        cash_win_probability,
        cash_win_label,
        causal_training_mean_probability,
        predicted_edge_10bps,
        actual_edge_10bps,
        causal_training_mean_edge,
        oof_cash_target,
        oof_cash_block_start,
    )
    cash = _strict_binary(oof_cash_target, name="oof_cash_target")
    execution_binding = _bind_oof_cash_to_executed_ledger(
        strategy,
        cash,
        oof_decision_dates,
    )
    cost_binding = {
        "strategy": _bind_ledger_cost_bps(
            strategy,
            name="strategy",
            cost_bps=cost_bps,
        ),
        "buy_hold": _bind_ledger_cost_bps(
            buy_hold,
            name="buy_hold",
            cost_bps=cost_bps,
        ),
    }
    metrics["underlying_scoring_contract"] = metrics["scoring_contract"]
    metrics["scoring_contract"] = "aapl-direct-cash-edge-development-v1"
    metrics.update(predictive)
    metrics.update(execution_binding)
    metrics["exact_cost_binding"] = cost_binding
    metrics["cost_bps_bound_to_both_ledgers"] = True
    try:
        json.dumps(metrics, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise DirectEdgeScoringError("Metrics are not canonical JSON-compatible") from exc
    return metrics


def apply_direct_edge_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply every base gate plus the direct expected-edge quality gates."""

    base = apply_downside_gates(metrics)
    improvement = metrics.get("expected_edge_mae_relative_improvement")
    episode_mean = metrics.get("episode_start_realized_edge_mean_10bps")
    episode_win_rate = metrics.get("episode_start_realized_edge_win_rate_10bps")
    direct_checks = {
        "expected_edge_mae_improves_causal_mean_by_at_least_1pct": (
            improvement is not None
            and math.isfinite(float(improvement))
            and float(improvement)
            >= DIRECT_EDGE_GATES[
                "minimum_expected_edge_mae_relative_improvement"
            ]
        ),
        "episode_start_realized_edge_mean_strictly_positive": (
            episode_mean is not None
            and math.isfinite(float(episode_mean))
            and float(episode_mean) > ACTIVE_EDGE_WIN_TOLERANCE
        ),
        "episode_start_realized_edge_win_rate_strictly_above_half": (
            episode_win_rate is not None
            and math.isfinite(float(episode_win_rate))
            and float(episode_win_rate)
            > DIRECT_EDGE_GATES[
                "minimum_episode_start_realized_edge_win_rate_exclusive"
            ]
        ),
        "exact_binary_oof_cash_target": bool(
            metrics.get("exact_binary_oof_cash_target")
        ),
        "exact_binary_edge_label_consistency": bool(
            metrics.get("exact_binary_edge_label_consistency")
        ),
        "oof_cash_target_bound_to_executed_ledger": bool(
            metrics.get("oof_cash_target_bound_to_executed_ledger")
        ),
        "exact_five_row_cash_block_reconstruction": bool(
            metrics.get("exact_five_row_cash_block_reconstruction")
        ),
        "cost_bps_bound_to_both_ledgers": bool(
            metrics.get("cost_bps_bound_to_both_ledgers")
        ),
    }
    checks = {
        **{name: bool(value) for name, value in base["checks"].items()},
        **{name: bool(value) for name, value in direct_checks.items()},
    }
    gate = {
        "contract": {
            "base_development_gates": dict(DEVELOPMENT_GATES),
            **dict(DIRECT_EDGE_GATES),
        },
        "base_gates_passed": bool(base["passed"]),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    json.dumps(gate, sort_keys=True, allow_nan=False)
    return gate


def evaluate_direct_edge_development(
    strategy: pd.DataFrame,
    buy_hold: pd.DataFrame,
    executed_binary_target: Sequence[float],
    cash_win_probability: Sequence[float],
    cash_win_label: Sequence[float],
    causal_training_mean_probability: Sequence[float],
    predicted_edge_10bps: Sequence[float],
    actual_edge_10bps: Sequence[float],
    causal_training_mean_edge: Sequence[float],
    oof_cash_target: Sequence[float],
    oof_cash_block_start: Sequence[float],
    oof_decision_dates: Sequence[object],
    *,
    cost_bps: float,
) -> dict[str, Any]:
    """Return JSON-safe direct-edge metrics and their named gate results."""

    metrics = score_direct_edge_metrics(
        strategy,
        buy_hold,
        executed_binary_target,
        cash_win_probability,
        cash_win_label,
        causal_training_mean_probability,
        predicted_edge_10bps,
        actual_edge_10bps,
        causal_training_mean_edge,
        oof_cash_target,
        oof_cash_block_start,
        oof_decision_dates,
        cost_bps=cost_bps,
    )
    result = {"metrics": metrics, "gates": apply_direct_edge_gates(metrics)}
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


__all__ = [
    "DIRECT_EDGE_GATES",
    "DirectEdgeScoringError",
    "apply_direct_edge_gates",
    "evaluate_direct_edge_development",
    "score_direct_edge_metrics",
]
