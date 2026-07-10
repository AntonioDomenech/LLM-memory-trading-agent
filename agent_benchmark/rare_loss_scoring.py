"""Sealed scoring for the frozen one-session rare-loss forest.

The existing one-session scorer owns ledger reconciliation, no-leverage proof,
ordinary CASH-win calibration, expected-edge calibration, and the broad
2005--2018 economic gates.  This module adds the independently predeclared
severe-loss target, the stricter completed-episode minimum, and the exact
full-sentiment versus core-price ablation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .downside_scoring import ACTIVE_EDGE_WIN_TOLERANCE
from .regime_consensus_scoring import (
    RegimeConsensusScoringError,
    apply_regime_consensus_gates,
    score_regime_consensus_metrics,
)


RARE_LOSS_GATES: Mapping[str, float | int] = {
    "minimum_severe_brier_relative_improvement": 0.05,
    "minimum_severe_event_precision": 0.25,
    "minimum_severe_event_precision_lift": 2.0,
    "minimum_completed_cash_episodes": 20,
}
CLIPPED_EDGE_BOUND = 0.08

REQUIRED_ABLATION_COST_BPS = (5.0, 10.0)


class RareLossScoringError(RegimeConsensusScoringError):
    """Raised when a rare-loss scoring input violates the frozen contract."""


def _numeric_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RareLossScoringError(f"{name} must be numeric") from exc
    if result.ndim != 1 or not len(result):
        raise RareLossScoringError(f"{name} must be a non-empty one-dimensional vector")
    if np.isinf(result).any():
        raise RareLossScoringError(f"{name} cannot contain infinity")
    return result


def _probability(values: Sequence[float], *, name: str) -> np.ndarray:
    result = _numeric_vector(values, name=name)
    if not np.isfinite(result).all() or ((result < 0.0) | (result > 1.0)).any():
        raise RareLossScoringError(f"{name} must contain finite probabilities")
    return result


def _binary(values: Sequence[float], *, name: str) -> np.ndarray:
    result = _numeric_vector(values, name=name)
    if not np.isfinite(result).all() or not np.isin(result, (0.0, 1.0)).all():
        raise RareLossScoringError(f"{name} must contain exact binary values")
    return result


def _dates(values: Sequence[object]) -> pd.DatetimeIndex:
    try:
        result = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise RareLossScoringError("oof_decision_dates must contain valid dates") from exc
    if result.tz is not None:
        raise RareLossScoringError("oof_decision_dates must be timezone-naive")
    result = result.normalize()
    if result.hasnans or result.has_duplicates or not result.is_monotonic_increasing:
        raise RareLossScoringError("oof_decision_dates must be unique and chronological")
    return result


def _mature_binary(values: Sequence[float], *, name: str) -> tuple[np.ndarray, np.ndarray]:
    result = _numeric_vector(values, name=name)
    mature = np.isfinite(result)
    if not mature.any():
        raise RareLossScoringError(f"{name} has no mature rows")
    missing = np.flatnonzero(~mature)
    if len(missing) and mature[int(missing[0]) :].any():
        raise RareLossScoringError(f"Missing {name} values are allowed only as a suffix")
    if not np.isin(result[mature], (0.0, 1.0)).all():
        raise RareLossScoringError(f"Mature {name} values must be binary")
    return result, mature


def _severe_support_hash(
    dates: pd.DatetimeIndex,
    mature: np.ndarray,
    labels: np.ndarray,
    baselines: np.ndarray,
) -> str:
    rows = [
        "|".join(
            (
                dates[position].date().isoformat(),
                float(labels[position]).hex(),
                float(baselines[position]).hex(),
            )
        )
        for position in np.flatnonzero(mature)
    ]
    return hashlib.sha256("\n".join(rows).encode("ascii")).hexdigest()


def _severe_metrics(
    severe_probability: Sequence[float],
    severe_label: Sequence[float],
    causal_severe_prevalence: Sequence[float],
    oof_cash_target: Sequence[float],
    oof_decision_dates: Sequence[object],
) -> dict[str, Any]:
    probability = _probability(severe_probability, name="severe_probability")
    label, mature = _mature_binary(severe_label, name="severe_label")
    baseline = _probability(
        causal_severe_prevalence, name="causal_severe_prevalence"
    )
    cash = _binary(oof_cash_target, name="oof_cash_target")
    dates = _dates(oof_decision_dates)
    if len({len(probability), len(label), len(baseline), len(cash), len(dates)}) != 1:
        raise RareLossScoringError("Severe-event inputs must align one-for-one")

    actual = label[mature]
    model_brier = float(np.mean(np.square(probability[mature] - actual)))
    baseline_brier = float(np.mean(np.square(baseline[mature] - actual)))
    absolute = float(baseline_brier - model_brier)
    relative = float(absolute / baseline_brier) if baseline_brier > 0.0 else None

    mature_cash = mature & (cash == 1.0)
    signal_count = int(mature_cash.sum())
    precision = float(label[mature_cash].mean()) if signal_count else None
    causal_at_signals = float(baseline[mature_cash].mean()) if signal_count else None
    lift = (
        float(precision / causal_at_signals)
        if precision is not None and causal_at_signals is not None and causal_at_signals > 0.0
        else None
    )
    return {
        "severe_target_semantics": "next_open_to_following_open_aapl_simple_return_lte_minus_2_5pct",
        "severe_brier_score": model_brier,
        "causal_severe_prevalence_brier_score": baseline_brier,
        "severe_brier_absolute_improvement": absolute,
        "severe_brier_relative_improvement": relative,
        "mature_severe_oof_rows": int(mature.sum()),
        "unmatured_trailing_severe_oof_rows": int((~mature).sum()),
        "mature_cash_signal_count": signal_count,
        "severe_event_precision_at_cash_signals": precision,
        "causal_severe_prevalence_at_cash_signals": causal_at_signals,
        "severe_event_precision_lift": lift,
        "severe_predictive_support_sha256": _severe_support_hash(
            dates, mature, label, baseline
        ),
    }


def score_rare_loss_metrics(
    strategy: pd.DataFrame,
    buy_hold: pd.DataFrame,
    executed_binary_target: Sequence[float],
    ordinary_cash_win_probability_10bps: Sequence[float],
    ordinary_cash_win_label_10bps: Sequence[float],
    causal_ordinary_prevalence: Sequence[float],
    predicted_edge_10bps: Sequence[float],
    actual_edge_10bps: Sequence[float],
    causal_training_mean_edge_10bps: Sequence[float],
    severe_probability: Sequence[float],
    severe_label: Sequence[float],
    causal_severe_prevalence: Sequence[float],
    oof_cash_target: Sequence[float],
    oof_decision_dates: Sequence[object],
    *,
    cost_bps: float,
) -> dict[str, Any]:
    """Return fully reconciled economics plus both predictive targets."""

    predicted_clipped = _numeric_vector(
        predicted_edge_10bps, name="predicted_edge_10bps"
    )
    actual_clipped = _numeric_vector(actual_edge_10bps, name="actual_edge_10bps")
    if not np.isfinite(predicted_clipped).all() or (
        np.abs(predicted_clipped) > CLIPPED_EDGE_BOUND
    ).any():
        raise RareLossScoringError(
            "predicted_edge_10bps must be finite and clipped to +/-8%"
        )
    finite_actual = np.isfinite(actual_clipped)
    if (np.abs(actual_clipped[finite_actual]) > CLIPPED_EDGE_BOUND).any():
        raise RareLossScoringError("actual_edge_10bps must be clipped to +/-8%")

    try:
        metrics = score_regime_consensus_metrics(
            strategy,
            buy_hold,
            executed_binary_target,
            ordinary_cash_win_probability_10bps,
            ordinary_cash_win_label_10bps,
            causal_ordinary_prevalence,
            predicted_edge_10bps,
            actual_edge_10bps,
            causal_training_mean_edge_10bps,
            oof_cash_target,
            oof_decision_dates,
            cost_bps=cost_bps,
        )
    except RegimeConsensusScoringError as exc:
        raise RareLossScoringError(str(exc)) from exc
    severe = _severe_metrics(
        severe_probability,
        severe_label,
        causal_severe_prevalence,
        oof_cash_target,
        oof_decision_dates,
    )
    if int(severe["mature_severe_oof_rows"]) != int(metrics["mature_oof_rows"]):
        raise RareLossScoringError("Severe and ordinary labels must share maturity support")
    metrics["underlying_scoring_contract"] = metrics["scoring_contract"]
    metrics["scoring_contract"] = "aapl-one-session-rare-loss-forest-development-v1"
    metrics["expected_edge_target_semantics"] = (
        "one_session_cash_active_log_edge_after_10bps_round_trip_clipped_to_plus_minus_8pct"
    )
    metrics.update(severe)
    ordinary_probability = _probability(
        ordinary_cash_win_probability_10bps,
        name="ordinary_cash_win_probability_10bps",
    )
    severe_probability_array = _probability(
        severe_probability, name="severe_probability"
    )
    cash_array = _binary(oof_cash_target, name="oof_cash_target")
    decision_dates = _dates(oof_decision_dates)
    if len(
        {
            len(ordinary_probability),
            len(severe_probability_array),
            len(predicted_clipped),
            len(cash_array),
            len(decision_dates),
        }
    ) != 1:
        raise RareLossScoringError("Prediction-binding inputs must align")
    prediction_rows = [
        "|".join(
            (
                decision_dates[position].date().isoformat(),
                float(ordinary_probability[position]).hex(),
                float(severe_probability_array[position]).hex(),
                float(predicted_clipped[position]).hex(),
                float(cash_array[position]).hex(),
            )
        )
        for position in range(len(decision_dates))
    ]
    metrics["prediction_and_cash_target_sha256"] = hashlib.sha256(
        "\n".join(prediction_rows).encode("ascii")
    ).hexdigest()
    json.dumps(metrics, sort_keys=True, allow_nan=False)
    return metrics


def apply_rare_loss_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply all inherited gates and the four rare-loss additions."""

    inherited = apply_regime_consensus_gates(metrics)
    severe_brier = metrics.get("severe_brier_relative_improvement")
    precision = metrics.get("severe_event_precision_at_cash_signals")
    lift = metrics.get("severe_event_precision_lift")
    checks = {
        **{name: bool(value) for name, value in inherited["checks"].items()},
        "severe_brier_improves_causal_prevalence_by_at_least_5pct": bool(
            severe_brier is not None
            and math.isfinite(float(severe_brier))
            and float(severe_brier)
            >= RARE_LOSS_GATES["minimum_severe_brier_relative_improvement"]
        ),
        "severe_event_precision_at_least_25pct": bool(
            precision is not None
            and math.isfinite(float(precision))
            and float(precision) >= RARE_LOSS_GATES["minimum_severe_event_precision"]
        ),
        "severe_event_precision_at_least_twice_causal_prevalence": bool(
            lift is not None
            and math.isfinite(float(lift))
            and float(lift) >= RARE_LOSS_GATES["minimum_severe_event_precision_lift"]
        ),
        "at_least_20_completed_cash_episodes": bool(
            int(metrics.get("completed_cash_episodes", -1))
            >= RARE_LOSS_GATES["minimum_completed_cash_episodes"]
        ),
    }
    result = {
        "contract": {
            "inherited_one_session_gates": inherited["contract"],
            **dict(RARE_LOSS_GATES),
        },
        "inherited_gates_passed": bool(inherited["passed"]),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def evaluate_rare_loss_development(*args: Any, **kwargs: Any) -> dict[str, Any]:
    metrics = score_rare_loss_metrics(*args, **kwargs)
    result = {"metrics": metrics, "gates": apply_rare_loss_gates(metrics)}
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def _metrics(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RareLossScoringError(f"{name} must be a mapping")
    nested = value.get("metrics")
    result = nested if isinstance(nested, Mapping) else value
    if result.get("scoring_contract") != "aapl-one-session-rare-loss-forest-development-v1":
        raise RareLossScoringError(f"{name} uses the wrong scoring contract")
    return result


def _finite_metric(metrics: Mapping[str, Any], key: str, *, name: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool):
        raise RareLossScoringError(f"{name} {key} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RareLossScoringError(f"{name} {key} must be finite") from exc
    if not math.isfinite(result):
        raise RareLossScoringError(f"{name} {key} must be finite")
    return result


def evaluate_rare_loss_ablation(
    full_5bps: Mapping[str, Any],
    core_5bps: Mapping[str, Any],
    full_10bps: Mapping[str, Any],
    core_10bps: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the sentiment model to beat core on every frozen comparison."""

    pairs = {
        "5bps": (5.0, _metrics(full_5bps, name="full_5bps"), _metrics(core_5bps, name="core_5bps")),
        "10bps": (10.0, _metrics(full_10bps, name="full_10bps"), _metrics(core_10bps, name="core_10bps")),
    }
    checks: dict[str, bool] = {}
    comparisons: dict[str, Any] = {}
    for label, (expected_cost, full, core) in pairs.items():
        if _finite_metric(full, "declared_cost_bps", name=f"full_{label}") != expected_cost:
            raise RareLossScoringError(f"full_{label} has the wrong cost")
        if _finite_metric(core, "declared_cost_bps", name=f"core_{label}") != expected_cost:
            raise RareLossScoringError(f"core_{label} has the wrong cost")
        ordinary_full = _finite_metric(full, "brier_score", name=f"full_{label}")
        ordinary_core = _finite_metric(core, "brier_score", name=f"core_{label}")
        severe_full = _finite_metric(full, "severe_brier_score", name=f"full_{label}")
        severe_core = _finite_metric(core, "severe_brier_score", name=f"core_{label}")
        total_full = _finite_metric(full, "total_active_log_edge", name=f"full_{label}")
        total_core = _finite_metric(core, "total_active_log_edge", name=f"core_{label}")
        weakest_full = _finite_metric(full, "minimum_fold_active_log_edge", name=f"full_{label}")
        weakest_core = _finite_metric(core, "minimum_fold_active_log_edge", name=f"core_{label}")
        support = bool(
            full.get("predictive_evaluation_support_sha256")
            == core.get("predictive_evaluation_support_sha256")
            and full.get("severe_predictive_support_sha256")
            == core.get("severe_predictive_support_sha256")
            and int(full.get("mature_oof_rows", -1))
            == int(core.get("mature_oof_rows", -2))
        )
        pair_checks = {
            f"paired_support_matches_{label}": support,
            f"full_ordinary_brier_strictly_lower_{label}": ordinary_full < ordinary_core - ACTIVE_EDGE_WIN_TOLERANCE,
            f"full_severe_brier_strictly_lower_{label}": severe_full < severe_core - ACTIVE_EDGE_WIN_TOLERANCE,
            f"full_total_edge_strictly_higher_{label}": total_full > total_core + ACTIVE_EDGE_WIN_TOLERANCE,
            f"full_weakest_fold_strictly_higher_{label}": weakest_full > weakest_core + ACTIVE_EDGE_WIN_TOLERANCE,
            f"full_rare_loss_gates_pass_{label}": bool(apply_rare_loss_gates(full)["passed"]),
        }
        checks.update(pair_checks)
        comparisons[label] = {
            "declared_cost_bps": expected_cost,
            "full_ordinary_brier": ordinary_full,
            "core_ordinary_brier": ordinary_core,
            "full_severe_brier": severe_full,
            "core_severe_brier": severe_core,
            "full_total_active_log_edge": total_full,
            "core_total_active_log_edge": total_core,
            "full_minimum_fold_active_log_edge": weakest_full,
            "core_minimum_fold_active_log_edge": weakest_core,
            "checks": pair_checks,
            "passed": bool(all(pair_checks.values())),
        }

    # Predictive evaluation is cost-invariant and must be the same object
    # bound into both economic ledgers.  Without these cross-cost checks a
    # caller could silently score different OOF forecasts at 5 and 10 bps.
    full_5 = pairs["5bps"][1]
    full_10 = pairs["10bps"][1]
    core_5 = pairs["5bps"][2]
    core_10 = pairs["10bps"][2]
    cross_cost: dict[str, bool] = {}
    for model_name, low_cost, high_cost in (
        ("full", full_5, full_10),
        ("core", core_5, core_10),
    ):
        cross_cost.update(
            {
                f"{model_name}_ordinary_support_bound_across_costs": bool(
                    low_cost.get("predictive_evaluation_support_sha256")
                    == high_cost.get("predictive_evaluation_support_sha256")
                    and int(low_cost.get("mature_oof_rows", -1))
                    == int(high_cost.get("mature_oof_rows", -2))
                ),
                f"{model_name}_severe_support_bound_across_costs": bool(
                    low_cost.get("severe_predictive_support_sha256")
                    == high_cost.get("severe_predictive_support_sha256")
                    and int(low_cost.get("mature_severe_oof_rows", -1))
                    == int(high_cost.get("mature_severe_oof_rows", -2))
                ),
                f"{model_name}_ordinary_brier_bound_across_costs": bool(
                    _finite_metric(low_cost, "brier_score", name=f"{model_name}_5bps")
                    == _finite_metric(high_cost, "brier_score", name=f"{model_name}_10bps")
                ),
                f"{model_name}_severe_brier_bound_across_costs": bool(
                    _finite_metric(
                        low_cost, "severe_brier_score", name=f"{model_name}_5bps"
                    )
                    == _finite_metric(
                        high_cost, "severe_brier_score", name=f"{model_name}_10bps"
                    )
                ),
                f"{model_name}_edge_mae_bound_across_costs": bool(
                    _finite_metric(
                        low_cost, "expected_edge_mae", name=f"{model_name}_5bps"
                    )
                    == _finite_metric(
                        high_cost, "expected_edge_mae", name=f"{model_name}_10bps"
                    )
                ),
                f"{model_name}_predictions_and_cash_target_bound_across_costs": bool(
                    low_cost.get("prediction_and_cash_target_sha256")
                    == high_cost.get("prediction_and_cash_target_sha256")
                    and bool(low_cost.get("prediction_and_cash_target_sha256"))
                ),
            }
        )
    checks.update(cross_cost)
    result = {
        "contract": {
            "name": "aapl-rare-loss-full-sentiment-vs-core-ablation-v1",
            "required_cost_bps": list(REQUIRED_ABLATION_COST_BPS),
            "strict_tolerance": ACTIVE_EDGE_WIN_TOLERANCE,
        },
        "comparisons": comparisons,
        "cross_cost_predictive_binding": cross_cost,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


__all__ = [
    "CLIPPED_EDGE_BOUND",
    "RARE_LOSS_GATES",
    "RareLossScoringError",
    "apply_rare_loss_gates",
    "evaluate_rare_loss_ablation",
    "evaluate_rare_loss_development",
    "score_rare_loss_metrics",
]
