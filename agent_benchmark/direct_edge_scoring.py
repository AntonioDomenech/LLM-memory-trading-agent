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


def _predictive_metrics(
    cash_win_probability: Sequence[float],
    cash_win_label: Sequence[float],
    causal_training_mean_probability: Sequence[float],
    predicted_edge_10bps: Sequence[float],
    actual_edge_10bps: Sequence[float],
    causal_training_mean_edge: Sequence[float],
    oof_cash_target: Sequence[float],
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

    lengths = {
        len(probability),
        len(label),
        len(baseline_probability),
        len(prediction),
        len(actual),
        len(baseline_edge),
        len(cash_target),
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
    episode_start = cash & ~np.r_[False, cash[:-1]]
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
        "mature_oof_cash_episode_starts": int(mature_episode_start.sum()),
        "unmatured_oof_cash_episode_starts": int(
            (episode_start & ~edge_mature).sum()
        ),
        "episode_start_realized_edge_mean_10bps": episode_mean,
        "episode_start_realized_edge_wins_10bps": episode_wins,
        "episode_start_realized_edge_win_rate_10bps": episode_win_rate,
        "exact_binary_oof_cash_target": True,
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
    )
    metrics["underlying_scoring_contract"] = metrics["scoring_contract"]
    metrics["scoring_contract"] = "aapl-direct-cash-edge-development-v1"
    metrics.update(predictive)
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
