"""Sealed development scoring for the one-session regime consensus model.

The economic contract is inherited from :mod:`downside_scoring`: every
candidate is a continuous, unleveraged LONG/CASH account over the same AAPL
fills from 2005 through 2018.  This module adds the one-session predictive
contract, exact execution-cost binding, completed-CASH-episode diagnostics,
and the paired full-model versus AAPL-only ablation required at both 5 and 10
basis points.

Predictions are evaluated one decision at a time.  Trading outcomes are not:
consecutive CASH decisions form one continuous episode, and that episode is
scored once from its actual sell fill to its actual LONG re-entry fill.
"""

from __future__ import annotations

import hashlib
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


REGIME_CONSENSUS_GATES: Mapping[str, float] = {
    "minimum_brier_relative_improvement": 0.02,
    "minimum_expected_edge_mae_relative_improvement": 0.01,
    "minimum_completed_cash_episode_mean_edge_exclusive": 0.0,
    "minimum_completed_cash_episode_win_rate": 0.55,
}

REQUIRED_ABLATION_COST_BPS = (5.0, 10.0)

_TRADE_TOLERANCE = 1e-12
_NUMERIC_ABSOLUTE_TOLERANCE = 1e-10
_NUMERIC_RELATIVE_TOLERANCE = 1e-12


class RegimeConsensusScoringError(DownsideScoringError):
    """Raised when a regime-consensus scoring input violates its contract."""


def _numeric_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RegimeConsensusScoringError(f"{name} must be numeric") from exc
    if result.ndim != 1:
        raise RegimeConsensusScoringError(f"{name} must be one-dimensional")
    if not len(result):
        raise RegimeConsensusScoringError(f"{name} cannot be empty")
    return result


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    result = _numeric_vector(values, name=name)
    if not np.isfinite(result).all():
        raise RegimeConsensusScoringError(f"{name} must be finite on every OOF row")
    return result


def _finite_probability(values: Sequence[float], *, name: str) -> np.ndarray:
    result = _finite_vector(values, name=name)
    if ((result < 0.0) | (result > 1.0)).any():
        raise RegimeConsensusScoringError(f"{name} must lie in [0, 1]")
    return result


def _strict_binary(values: Sequence[float], *, name: str) -> np.ndarray:
    result = _finite_vector(values, name=name)
    if not np.isin(result, (0.0, 1.0)).all():
        raise RegimeConsensusScoringError(f"{name} must contain exactly 0 or 1")
    return result


def _mature_suffix_mask(values: np.ndarray, *, name: str) -> np.ndarray:
    if np.isinf(values).any():
        raise RegimeConsensusScoringError(f"{name} cannot contain infinity")
    mature = np.isfinite(values)
    if not mature.any():
        raise RegimeConsensusScoringError(f"{name} requires at least one mature row")
    missing = np.flatnonzero(~mature)
    if len(missing) and mature[int(missing[0]) :].any():
        raise RegimeConsensusScoringError(
            f"Missing {name} values are allowed only as an unmatured trailing suffix"
        )
    return mature


def _normalized_dates(values: Sequence[object], *, name: str) -> pd.DatetimeIndex:
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise RegimeConsensusScoringError(f"{name} must contain valid dates") from exc
    if dates.tz is not None:
        raise RegimeConsensusScoringError(f"{name} must be timezone-naive")
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise RegimeConsensusScoringError(
            f"{name} must contain unique chronological dates"
        )
    dates = dates.normalize()
    if dates.has_duplicates:
        raise RegimeConsensusScoringError(
            f"{name} must remain unique after date normalization"
        )
    return dates


def _bind_oof_cash_to_executed_ledger(
    strategy: pd.DataFrame,
    oof_cash_target: np.ndarray,
    oof_decision_dates: Sequence[object],
) -> dict[str, int | bool]:
    """Bind prediction-side CASH=1 decisions to executed LONG=1 targets."""

    if "decision_date" not in strategy.columns:
        raise RegimeConsensusScoringError(
            "strategy ledger lacks decision_date for OOF target binding"
        )
    oof_dates = _normalized_dates(oof_decision_dates, name="oof_decision_dates")
    ledger_dates = _normalized_dates(
        strategy["decision_date"].tolist(), name="strategy decision_date"
    )
    if len(oof_dates) != len(oof_cash_target):
        raise RegimeConsensusScoringError(
            "oof_decision_dates must align with oof_cash_target"
        )

    ledger_target = _strict_binary(
        strategy["target_exposure"].to_numpy(),
        name="strategy ledger target_exposure",
    )
    executed_cash = {
        timestamp: 1.0 - float(target)
        for timestamp, target in zip(ledger_dates, ledger_target, strict=True)
    }
    matched = np.asarray([timestamp in executed_cash for timestamp in oof_dates])
    if not matched.any():
        raise RegimeConsensusScoringError(
            "No OOF decision date overlaps the executed strategy ledger"
        )
    missing = np.flatnonzero(~matched)
    if len(missing):
        first_missing = int(missing[0])
        if matched[first_missing:].any() or not bool(
            (oof_dates[~matched] > ledger_dates.max()).all()
        ):
            raise RegimeConsensusScoringError(
                "Unmatched OOF decisions are allowed only as a trailing suffix "
                "after the final executed ledger decision"
            )
    expected = np.asarray(
        [executed_cash[timestamp] for timestamp in oof_dates[matched]], dtype=float
    )
    if not np.array_equal(oof_cash_target[matched], expected):
        raise RegimeConsensusScoringError(
            "OOF CASH target does not match the executed strategy by decision date"
        )
    return {
        "oof_executed_decision_matches": int(matched.sum()),
        "oof_unexecuted_trailing_decisions": int((~matched).sum()),
        "oof_cash_target_bound_to_executed_ledger": True,
    }


def _finite_cost_bps(cost_bps: float) -> float:
    if isinstance(cost_bps, bool):
        raise RegimeConsensusScoringError("cost_bps must be a finite number")
    try:
        bps = float(cost_bps)
    except (TypeError, ValueError) as exc:
        raise RegimeConsensusScoringError("cost_bps must be a finite number") from exc
    if not math.isfinite(bps) or bps < 0.0 or bps >= 10_000.0:
        raise RegimeConsensusScoringError(
            "cost_bps must be finite and lie in [0, 10,000)"
        )
    return bps


def _bind_ledger_cost_bps(
    ledger: pd.DataFrame,
    *,
    name: str,
    cost_bps: float,
) -> dict[str, Any]:
    """Prove declared slippage from fills and reconcile the whole cash ledger."""

    bps = _finite_cost_bps(cost_bps)
    required = {
        "cash",
        "daily_return",
        "equity",
        "equity_before_fill",
        "fees",
        "fill_price",
        "margin_interest",
        "reference_price",
        "shares",
        "signed_share_delta",
        "slippage",
        "trade_executed",
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise RegimeConsensusScoringError(
            f"{name} ledger lacks exact cost proof columns: {missing}"
        )

    numeric: dict[str, np.ndarray] = {}
    for column in sorted(required.difference({"trade_executed"})):
        try:
            values = pd.to_numeric(ledger[column], errors="raise").to_numpy(
                dtype=float
            )
        except (TypeError, ValueError) as exc:
            raise RegimeConsensusScoringError(
                f"{name} ledger {column} must be numeric"
            ) from exc
        if not np.isfinite(values).all():
            raise RegimeConsensusScoringError(
                f"{name} ledger {column} must be finite"
            )
        numeric[column] = values

    reference = numeric["reference_price"]
    fill = numeric["fill_price"]
    delta = numeric["signed_share_delta"]
    shares = numeric["shares"]
    cash = numeric["cash"]
    fees = numeric["fees"]
    margin_interest = numeric["margin_interest"]
    if (reference <= 0.0).any() or (fill <= 0.0).any():
        raise RegimeConsensusScoringError(
            f"{name} ledger contains a non-positive execution price"
        )
    if not np.allclose(fees, 0.0, rtol=0.0, atol=_NUMERIC_ABSOLUTE_TOLERANCE):
        raise RegimeConsensusScoringError(
            f"{name} ledger must use the declared zero-commission contract"
        )
    if not np.allclose(
        margin_interest, 0.0, rtol=0.0, atol=_NUMERIC_ABSOLUTE_TOLERANCE
    ):
        raise RegimeConsensusScoringError(
            f"{name} ledger must use the declared zero-margin-interest contract"
        )

    trade = np.abs(delta) > _TRADE_TOLERANCE
    raw_trade = ledger["trade_executed"].to_numpy()
    if any(not isinstance(value, (bool, np.bool_)) for value in raw_trade):
        raise RegimeConsensusScoringError(
            f"{name} ledger trade_executed must contain booleans"
        )
    if not np.array_equal(raw_trade.astype(bool), trade):
        raise RegimeConsensusScoringError(
            f"{name} ledger trade_executed does not match signed_share_delta"
        )

    rate = bps / 10_000.0
    expected_fill = reference.copy()
    expected_fill[delta > _TRADE_TOLERANCE] *= 1.0 + rate
    expected_fill[delta < -_TRADE_TOLERANCE] *= 1.0 - rate
    if not np.allclose(
        fill,
        expected_fill,
        rtol=_NUMERIC_RELATIVE_TOLERANCE,
        atol=_NUMERIC_ABSOLUTE_TOLERANCE,
    ):
        raise RegimeConsensusScoringError(
            f"{name} ledger fill prices do not prove the declared cost_bps"
        )
    expected_slippage = np.abs(delta) * np.abs(expected_fill - reference)
    if not np.allclose(
        numeric["slippage"],
        expected_slippage,
        rtol=_NUMERIC_RELATIVE_TOLERANCE,
        atol=_NUMERIC_ABSOLUTE_TOLERANCE,
    ):
        raise RegimeConsensusScoringError(
            f"{name} ledger slippage does not match fills and share deltas"
        )

    expected_delta = np.diff(np.r_[0.0, shares])
    if not np.allclose(
        delta,
        expected_delta,
        rtol=_NUMERIC_RELATIVE_TOLERANCE,
        atol=_NUMERIC_ABSOLUTE_TOLERANCE,
    ):
        raise RegimeConsensusScoringError(
            f"{name} ledger share deltas do not reconcile to holdings"
        )

    equity_before = numeric["equity_before_fill"]
    equity = numeric["equity"]
    if (equity_before <= 0.0).any() or (equity <= 0.0).any():
        raise RegimeConsensusScoringError(f"{name} ledger contains non-positive equity")
    expected_equity = cash + shares * reference
    if not np.allclose(
        equity,
        expected_equity,
        rtol=_NUMERIC_RELATIVE_TOLERANCE,
        atol=_NUMERIC_ABSOLUTE_TOLERANCE,
    ):
        raise RegimeConsensusScoringError(
            f"{name} ledger equity does not reconcile to cash and marked holdings"
        )

    previous_cash = np.r_[equity_before[0], cash[:-1]]
    previous_shares = np.r_[0.0, shares[:-1]]
    expected_before = previous_cash + previous_shares * reference - margin_interest
    if not np.allclose(
        equity_before,
        expected_before,
        rtol=_NUMERIC_RELATIVE_TOLERANCE,
        atol=_NUMERIC_ABSOLUTE_TOLERANCE,
    ):
        raise RegimeConsensusScoringError(
            f"{name} ledger equity_before_fill does not reconcile chronologically"
        )
    expected_cash = previous_cash - margin_interest - delta * fill - fees
    if not np.allclose(
        cash,
        expected_cash,
        rtol=_NUMERIC_RELATIVE_TOLERANCE,
        atol=_NUMERIC_ABSOLUTE_TOLERANCE,
    ):
        raise RegimeConsensusScoringError(
            f"{name} ledger cash does not reconcile to its declared fills"
        )

    expected_return = np.empty(len(equity), dtype=float)
    expected_return[0] = equity[0] / equity_before[0] - 1.0
    expected_return[1:] = equity[1:] / equity[:-1] - 1.0
    if not np.allclose(
        numeric["daily_return"],
        expected_return,
        rtol=_NUMERIC_RELATIVE_TOLERANCE,
        atol=_NUMERIC_ABSOLUTE_TOLERANCE,
    ):
        raise RegimeConsensusScoringError(
            f"{name} ledger daily_return does not reconcile to equity"
        )

    return {
        "declared_cost_bps": bps,
        "cost_bps_bound_to_fills": True,
        "zero_commission_proof": True,
        "zero_margin_interest_proof": True,
        "cash_and_equity_reconciled": True,
        "daily_returns_reconciled_to_equity": True,
        "trade_rows": int(trade.sum()),
        "total_verified_slippage": float(numeric["slippage"].sum()),
        "numeric_absolute_tolerance": _NUMERIC_ABSOLUTE_TOLERANCE,
        "numeric_relative_tolerance": _NUMERIC_RELATIVE_TOLERANCE,
    }


def _completed_cash_episode_metrics(
    strategy: pd.DataFrame,
    buy_hold: pd.DataFrame,
    executed_target: np.ndarray,
    *,
    cost_bps: float,
) -> dict[str, Any]:
    """Score each continuous CASH run once from sell fill to re-entry fill."""

    if executed_target[0] != 1.0 or executed_target[-1] != 1.0:
        raise RegimeConsensusScoringError(
            "First and final development fills must both be LONG"
        )
    cash = executed_target == 0.0
    starts = np.flatnonzero(cash & ~np.r_[False, cash[:-1]])
    reentries = np.flatnonzero(~cash & np.r_[False, cash[:-1]])
    if len(starts) != len(reentries):  # boundary LONG checks make this defensive.
        raise RegimeConsensusScoringError(
            "Every CASH episode must have exactly one LONG re-entry"
        )

    delta = pd.to_numeric(strategy["signed_share_delta"], errors="raise").to_numpy(
        dtype=float
    )
    reference = pd.to_numeric(
        strategy["reference_price"], errors="raise"
    ).to_numpy(dtype=float)
    fill = pd.to_numeric(strategy["fill_price"], errors="raise").to_numpy(dtype=float)
    strategy_return = pd.to_numeric(
        strategy["daily_return"], errors="raise"
    ).to_numpy(dtype=float)
    benchmark_return = pd.to_numeric(
        buy_hold["daily_return"], errors="raise"
    ).to_numpy(dtype=float)
    active_log = np.log1p(strategy_return) - np.log1p(benchmark_return)
    fill_dates = _normalized_dates(strategy["fill_date"].tolist(), name="fill_date")
    decision_dates = _normalized_dates(
        strategy["decision_date"].tolist(), name="strategy decision_date"
    )

    records: list[dict[str, Any]] = []
    for episode_number, (start, reentry) in enumerate(
        zip(starts, reentries, strict=True), start=1
    ):
        start = int(start)
        reentry = int(reentry)
        if reentry <= start:
            raise RegimeConsensusScoringError(
                "A CASH re-entry must occur after its episode entry"
            )
        if delta[start] >= -_TRADE_TOLERANCE:
            raise RegimeConsensusScoringError(
                "A CASH episode entry must contain the exact AAPL sell fill"
            )
        if delta[reentry] <= _TRADE_TOLERANCE:
            raise RegimeConsensusScoringError(
                "A CASH episode re-entry must contain the exact AAPL buy fill"
            )
        if np.any(np.abs(delta[start + 1 : reentry]) > _TRADE_TOLERANCE):
            raise RegimeConsensusScoringError(
                "A continuous CASH episode cannot contain an intermediate trade"
            )

        fill_edge = float(math.log(fill[start] / fill[reentry]))
        rate = float(cost_bps) / 10_000.0
        reference_open_edge = float(
            math.log((1.0 - rate) / (1.0 + rate))
            - math.log(reference[reentry] / reference[start])
        )
        if not math.isclose(
            fill_edge,
            reference_open_edge,
            rel_tol=_NUMERIC_RELATIVE_TOLERANCE,
            abs_tol=_NUMERIC_ABSOLUTE_TOLERANCE,
        ):
            raise RegimeConsensusScoringError(
                "Completed CASH episode edge does not reconcile to its exact "
                "entry/re-entry reference opens and one round-trip cost"
            )
        ledger_edge = float(active_log[start : reentry + 1].sum())
        if not math.isclose(
            fill_edge,
            ledger_edge,
            rel_tol=_NUMERIC_RELATIVE_TOLERANCE,
            abs_tol=_NUMERIC_ABSOLUTE_TOLERANCE,
        ):
            raise RegimeConsensusScoringError(
                "Completed CASH episode edge does not reconcile between exact "
                "entry/re-entry fills and the active-return ledger"
            )
        records.append(
            {
                "episode": episode_number,
                "cash_entry_decision_date": decision_dates[start].date().isoformat(),
                "cash_entry_fill_date": fill_dates[start].date().isoformat(),
                "long_reentry_fill_date": fill_dates[reentry].date().isoformat(),
                "cash_sessions": int(reentry - start),
                "cash_entry_reference_open": float(reference[start]),
                "long_reentry_reference_open": float(reference[reentry]),
                "sell_fill_price": float(fill[start]),
                "buy_fill_price": float(fill[reentry]),
                "realized_active_log_edge": fill_edge,
                "reference_open_round_trip_active_log_edge": reference_open_edge,
                "ledger_active_log_edge": ledger_edge,
                "win": bool(fill_edge > ACTIVE_EDGE_WIN_TOLERANCE),
            }
        )

    edges = np.asarray(
        [record["realized_active_log_edge"] for record in records], dtype=float
    )
    wins = int(np.sum(edges > ACTIVE_EDGE_WIN_TOLERANCE)) if len(edges) else 0
    return {
        "completed_cash_episodes": int(len(records)),
        "completed_cash_episode_wins": wins,
        "completed_cash_episode_mean_active_log_edge": (
            float(edges.mean()) if len(edges) else None
        ),
        "completed_cash_episode_win_rate": (
            float(wins / len(edges)) if len(edges) else None
        ),
        "completed_cash_episode_records": records,
        "episode_scoring_cost_bps": float(cost_bps),
        "episode_scoring_semantics": (
            "one_continuous_cash_run_scored_once_from_sell_fill_to_long_reentry_fill"
        ),
        "first_development_fill_long": True,
        "final_development_fill_long": True,
        "all_cash_episodes_completed": True,
        "episode_edges_reconciled_to_active_ledger": True,
    }


def _support_sha256(
    dates: pd.DatetimeIndex,
    mature: np.ndarray,
    label: np.ndarray,
    actual_edge: np.ndarray,
    baseline_probability: np.ndarray,
    baseline_edge: np.ndarray,
) -> str:
    rows = [
        "|".join(
            (
                dates[position].date().isoformat(),
                float(label[position]).hex(),
                float(actual_edge[position]).hex(),
                float(baseline_probability[position]).hex(),
                float(baseline_edge[position]).hex(),
            )
        )
        for position in np.flatnonzero(mature)
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _predictive_metrics(
    cash_win_probability_10bps: Sequence[float],
    cash_win_label_10bps: Sequence[float],
    causal_prevalence_probability: Sequence[float],
    predicted_edge_10bps: Sequence[float],
    actual_edge_10bps: Sequence[float],
    causal_training_mean_edge_10bps: Sequence[float],
    oof_cash_target: Sequence[float],
    oof_decision_dates: Sequence[object],
) -> tuple[dict[str, Any], np.ndarray]:
    probability = _finite_probability(
        cash_win_probability_10bps, name="cash_win_probability_10bps"
    )
    label = _numeric_vector(cash_win_label_10bps, name="cash_win_label_10bps")
    baseline_probability = _finite_probability(
        causal_prevalence_probability, name="causal_prevalence_probability"
    )
    prediction = _finite_vector(predicted_edge_10bps, name="predicted_edge_10bps")
    actual = _numeric_vector(actual_edge_10bps, name="actual_edge_10bps")
    baseline_edge = _finite_vector(
        causal_training_mean_edge_10bps,
        name="causal_training_mean_edge_10bps",
    )
    cash_target = _strict_binary(oof_cash_target, name="oof_cash_target")
    dates = _normalized_dates(oof_decision_dates, name="oof_decision_dates")
    lengths = {
        len(probability),
        len(label),
        len(baseline_probability),
        len(prediction),
        len(actual),
        len(baseline_edge),
        len(cash_target),
        len(dates),
    }
    if len(lengths) != 1:
        raise RegimeConsensusScoringError(
            "All probability, edge, label, date, baseline, and OOF CASH inputs must align"
        )

    label_mature = _mature_suffix_mask(label, name="cash_win_label_10bps")
    edge_mature = _mature_suffix_mask(actual, name="actual_edge_10bps")
    if not np.array_equal(label_mature, edge_mature):
        raise RegimeConsensusScoringError(
            "Binary and continuous one-session labels must have the same maturity mask"
        )
    mature_label = label[label_mature]
    if not np.isin(mature_label, (0.0, 1.0)).all():
        raise RegimeConsensusScoringError(
            "Mature cash_win_label_10bps values must contain exactly 0 or 1"
        )
    if not np.array_equal(
        mature_label, (actual[edge_mature] > 0.0).astype(float)
    ):
        raise RegimeConsensusScoringError(
            "cash_win_label_10bps must exactly equal actual_edge_10bps > 0"
        )

    model_brier = float(np.mean(np.square(probability[label_mature] - mature_label)))
    baseline_brier = float(
        np.mean(np.square(baseline_probability[label_mature] - mature_label))
    )
    brier_absolute = float(baseline_brier - model_brier)
    brier_relative = (
        float(brier_absolute / baseline_brier) if baseline_brier > 0.0 else None
    )
    model_mae = float(np.mean(np.abs(prediction[edge_mature] - actual[edge_mature])))
    baseline_mae = float(
        np.mean(np.abs(baseline_edge[edge_mature] - actual[edge_mature]))
    )
    mae_absolute = float(baseline_mae - model_mae)
    mae_relative = float(mae_absolute / baseline_mae) if baseline_mae > 0.0 else None
    return (
        {
            "predictive_target_semantics": (
                "one_session_cash_beats_long_after_10bps_round_trip"
            ),
            "expected_edge_target_semantics": (
                "one_session_cash_active_log_edge_after_10bps_round_trip"
            ),
            "oof_rows": int(len(actual)),
            "mature_oof_rows": int(edge_mature.sum()),
            "unmatured_trailing_oof_rows": int((~edge_mature).sum()),
            "brier_score": model_brier,
            "causal_prevalence_brier_score": baseline_brier,
            "brier_absolute_improvement": brier_absolute,
            "brier_relative_improvement": brier_relative,
            "expected_edge_mae": model_mae,
            "causal_training_mean_edge_mae": baseline_mae,
            "expected_edge_mae_absolute_improvement": mae_absolute,
            "expected_edge_mae_relative_improvement": mae_relative,
            "oof_cash_signals": int(cash_target.sum()),
            "predictive_evaluation_support_sha256": _support_sha256(
                dates,
                edge_mature,
                label,
                actual,
                baseline_probability,
                baseline_edge,
            ),
            "exact_binary_edge_label_consistency": True,
            "exact_binary_oof_cash_target": True,
            "causal_prevalence_baseline_declared": True,
        },
        cash_target,
    )


def score_regime_consensus_metrics(
    strategy: pd.DataFrame,
    buy_hold: pd.DataFrame,
    executed_binary_target: Sequence[float],
    cash_win_probability_10bps: Sequence[float],
    cash_win_label_10bps: Sequence[float],
    causal_prevalence_probability: Sequence[float],
    predicted_edge_10bps: Sequence[float],
    actual_edge_10bps: Sequence[float],
    causal_training_mean_edge_10bps: Sequence[float],
    oof_cash_target: Sequence[float],
    oof_decision_dates: Sequence[object],
    *,
    cost_bps: float,
) -> dict[str, Any]:
    """Score one frozen regime-consensus candidate over 2005 through 2018."""

    supplied_cost = _finite_cost_bps(cost_bps)
    target = _strict_binary(executed_binary_target, name="executed_binary_target")
    if len(target) != len(strategy):
        raise RegimeConsensusScoringError(
            "executed_binary_target must align one-for-one with strategy fills"
        )
    if target[0] != 1.0 or target[-1] != 1.0:
        raise RegimeConsensusScoringError(
            "First and final development fills must both be LONG"
        )

    predictive, cash_target = _predictive_metrics(
        cash_win_probability_10bps,
        cash_win_label_10bps,
        causal_prevalence_probability,
        predicted_edge_10bps,
        actual_edge_10bps,
        causal_training_mean_edge_10bps,
        oof_cash_target,
        oof_decision_dates,
    )
    try:
        metrics = score_downside_metrics(
            strategy,
            buy_hold,
            target,
            cash_win_probability_10bps,
            cash_win_label_10bps,
            causal_prevalence_probability,
            cost_bps=supplied_cost,
        )
    except DownsideScoringError as exc:
        raise RegimeConsensusScoringError(str(exc)) from exc

    cost_binding = {
        "strategy": _bind_ledger_cost_bps(
            strategy, name="strategy", cost_bps=supplied_cost
        ),
        "buy_hold": _bind_ledger_cost_bps(
            buy_hold, name="buy_hold", cost_bps=supplied_cost
        ),
    }
    episodes = _completed_cash_episode_metrics(
        strategy,
        buy_hold,
        target,
        cost_bps=supplied_cost,
    )
    execution_binding = _bind_oof_cash_to_executed_ledger(
        strategy, cash_target, oof_decision_dates
    )

    metrics["underlying_scoring_contract"] = metrics["scoring_contract"]
    metrics["scoring_contract"] = (
        "aapl-one-session-regime-consensus-development-v1"
    )
    # Replace the base scorer's absolute Brier diagnostics with the explicit
    # causal-prevalence relative-improvement contract.
    metrics.update(predictive)
    metrics.update(episodes)
    metrics.update(execution_binding)
    metrics["exact_cost_binding"] = cost_binding
    metrics["cost_bps_bound_to_both_ledgers"] = True
    metrics["declared_cost_bps"] = supplied_cost
    if int(metrics["cash_episodes"]) != int(metrics["completed_cash_episodes"]):
        raise RegimeConsensusScoringError(
            "Base cash-episode count does not match completed round-trip episodes"
        )
    try:
        json.dumps(metrics, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise RegimeConsensusScoringError(
            "Regime-consensus metrics are not canonical JSON-compatible"
        ) from exc
    return metrics


def apply_regime_consensus_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply inherited economic gates and all one-session predictive gates."""

    base = apply_downside_gates(metrics)
    brier_improvement = metrics.get("brier_relative_improvement")
    mae_improvement = metrics.get("expected_edge_mae_relative_improvement")
    episode_mean = metrics.get("completed_cash_episode_mean_active_log_edge")
    episode_win_rate = metrics.get("completed_cash_episode_win_rate")
    regime_checks = {
        "brier_improves_causal_prevalence_by_at_least_2pct": (
            brier_improvement is not None
            and math.isfinite(float(brier_improvement))
            and float(brier_improvement)
            >= REGIME_CONSENSUS_GATES["minimum_brier_relative_improvement"]
        ),
        "expected_edge_mae_improves_causal_mean_by_at_least_1pct": (
            mae_improvement is not None
            and math.isfinite(float(mae_improvement))
            and float(mae_improvement)
            >= REGIME_CONSENSUS_GATES[
                "minimum_expected_edge_mae_relative_improvement"
            ]
        ),
        "completed_cash_episode_mean_edge_strictly_positive": (
            episode_mean is not None
            and math.isfinite(float(episode_mean))
            and float(episode_mean) > ACTIVE_EDGE_WIN_TOLERANCE
        ),
        "completed_cash_episode_win_rate_at_least_55pct": (
            episode_win_rate is not None
            and math.isfinite(float(episode_win_rate))
            and float(episode_win_rate)
            >= REGIME_CONSENSUS_GATES[
                "minimum_completed_cash_episode_win_rate"
            ]
        ),
        "first_development_fill_long": bool(
            metrics.get("first_development_fill_long")
        ),
        "final_development_fill_long": bool(
            metrics.get("final_development_fill_long")
        ),
        "all_cash_episodes_completed": bool(metrics.get("all_cash_episodes_completed")),
        "episode_edges_reconciled_to_active_ledger": bool(
            metrics.get("episode_edges_reconciled_to_active_ledger")
        ),
        "exact_binary_edge_label_consistency": bool(
            metrics.get("exact_binary_edge_label_consistency")
        ),
        "oof_cash_target_bound_to_executed_ledger": bool(
            metrics.get("oof_cash_target_bound_to_executed_ledger")
        ),
        "cost_bps_bound_to_both_ledgers": bool(
            metrics.get("cost_bps_bound_to_both_ledgers")
        ),
    }
    checks = {
        **{name: bool(value) for name, value in base["checks"].items()},
        **{name: bool(value) for name, value in regime_checks.items()},
    }
    gate = {
        "contract": {
            "base_development_gates": dict(DEVELOPMENT_GATES),
            **dict(REGIME_CONSENSUS_GATES),
        },
        "base_gates_passed": bool(base["passed"]),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    json.dumps(gate, sort_keys=True, allow_nan=False)
    return gate


def evaluate_regime_consensus_development(
    strategy: pd.DataFrame,
    buy_hold: pd.DataFrame,
    executed_binary_target: Sequence[float],
    cash_win_probability_10bps: Sequence[float],
    cash_win_label_10bps: Sequence[float],
    causal_prevalence_probability: Sequence[float],
    predicted_edge_10bps: Sequence[float],
    actual_edge_10bps: Sequence[float],
    causal_training_mean_edge_10bps: Sequence[float],
    oof_cash_target: Sequence[float],
    oof_decision_dates: Sequence[object],
    *,
    cost_bps: float,
) -> dict[str, Any]:
    """Return JSON-safe metrics and named gates for one declared cost."""

    metrics = score_regime_consensus_metrics(
        strategy,
        buy_hold,
        executed_binary_target,
        cash_win_probability_10bps,
        cash_win_label_10bps,
        causal_prevalence_probability,
        predicted_edge_10bps,
        actual_edge_10bps,
        causal_training_mean_edge_10bps,
        oof_cash_target,
        oof_decision_dates,
        cost_bps=cost_bps,
    )
    result = {"metrics": metrics, "gates": apply_regime_consensus_gates(metrics)}
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def _metrics_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegimeConsensusScoringError(f"{name} must be a metrics mapping")
    nested = value.get("metrics")
    metrics = nested if isinstance(nested, Mapping) else value
    if metrics.get("scoring_contract") != (
        "aapl-one-session-regime-consensus-development-v1"
    ):
        raise RegimeConsensusScoringError(
            f"{name} does not use the regime-consensus scoring contract"
        )
    return metrics


def _metric_float(metrics: Mapping[str, Any], key: str, *, name: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool):
        raise RegimeConsensusScoringError(f"{name} {key} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RegimeConsensusScoringError(
            f"{name} {key} must be a finite number"
        ) from exc
    if not math.isfinite(result):
        raise RegimeConsensusScoringError(f"{name} {key} must be a finite number")
    return result


def evaluate_regime_consensus_ablation(
    full_5bps: Mapping[str, Any],
    aapl_only_5bps: Mapping[str, Any],
    full_10bps: Mapping[str, Any],
    aapl_only_10bps: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the paired full-versus-AAPL-only contract at both required costs."""

    pairs = {
        "5bps": (
            5.0,
            _metrics_mapping(full_5bps, name="full_5bps"),
            _metrics_mapping(aapl_only_5bps, name="aapl_only_5bps"),
        ),
        "10bps": (
            10.0,
            _metrics_mapping(full_10bps, name="full_10bps"),
            _metrics_mapping(aapl_only_10bps, name="aapl_only_10bps"),
        ),
    }
    checks: dict[str, bool] = {}
    comparisons: dict[str, Any] = {}
    for label, (expected_cost, full, aapl_only) in pairs.items():
        for model_name, metrics in (("full", full), ("aapl_only", aapl_only)):
            declared = _finite_cost_bps(metrics.get("declared_cost_bps", -1.0))
            if declared != expected_cost:
                raise RegimeConsensusScoringError(
                    f"{model_name}_{label} does not prove the required {expected_cost:g}bps cost"
                )

        support_match = bool(
            full.get("predictive_evaluation_support_sha256")
            and full.get("predictive_evaluation_support_sha256")
            == aapl_only.get("predictive_evaluation_support_sha256")
            and int(full.get("mature_oof_rows", -1))
            == int(aapl_only.get("mature_oof_rows", -2))
        )
        full_brier = _metric_float(full, "brier_score", name=f"full_{label}")
        aapl_brier = _metric_float(
            aapl_only, "brier_score", name=f"aapl_only_{label}"
        )
        full_mae = _metric_float(full, "expected_edge_mae", name=f"full_{label}")
        aapl_mae = _metric_float(
            aapl_only, "expected_edge_mae", name=f"aapl_only_{label}"
        )
        full_total_edge = _metric_float(
            full, "total_active_log_edge", name=f"full_{label}"
        )
        aapl_total_edge = _metric_float(
            aapl_only, "total_active_log_edge", name=f"aapl_only_{label}"
        )
        full_weakest_fold = _metric_float(
            full, "minimum_fold_active_log_edge", name=f"full_{label}"
        )
        aapl_weakest_fold = _metric_float(
            aapl_only,
            "minimum_fold_active_log_edge",
            name=f"aapl_only_{label}",
        )

        pair_checks = {
            f"paired_predictive_support_matches_{label}": support_match,
            f"both_models_bind_exact_declared_cost_{label}": bool(
                full.get("cost_bps_bound_to_both_ledgers")
                and aapl_only.get("cost_bps_bound_to_both_ledgers")
            ),
            f"full_brier_strictly_lower_than_aapl_only_{label}": (
                full_brier < aapl_brier - ACTIVE_EDGE_WIN_TOLERANCE
            ),
            f"full_edge_mae_no_worse_than_aapl_only_{label}": (
                full_mae <= aapl_mae + ACTIVE_EDGE_WIN_TOLERANCE
            ),
            f"full_total_active_edge_strictly_greater_than_aapl_only_{label}": (
                full_total_edge
                > aapl_total_edge + ACTIVE_EDGE_WIN_TOLERANCE
            ),
            f"full_weakest_fold_no_worse_than_aapl_only_{label}": (
                full_weakest_fold
                >= aapl_weakest_fold - ACTIVE_EDGE_WIN_TOLERANCE
            ),
            f"full_regime_consensus_gates_pass_{label}": bool(
                apply_regime_consensus_gates(full)["passed"]
            ),
        }
        checks.update(pair_checks)
        comparisons[label] = {
            "declared_cost_bps": expected_cost,
            "full_brier_score": full_brier,
            "aapl_only_brier_score": aapl_brier,
            "full_expected_edge_mae": full_mae,
            "aapl_only_expected_edge_mae": aapl_mae,
            "full_total_active_log_edge": full_total_edge,
            "aapl_only_total_active_log_edge": aapl_total_edge,
            "full_minimum_fold_active_log_edge": full_weakest_fold,
            "aapl_only_minimum_fold_active_log_edge": aapl_weakest_fold,
            "predictive_evaluation_support_sha256": full.get(
                "predictive_evaluation_support_sha256"
            ),
            "checks": pair_checks,
            "passed": bool(all(pair_checks.values())),
        }

    result = {
        "contract": {
            "name": "aapl-regime-consensus-full-vs-aapl-only-ablation-v1",
            "required_cost_bps": list(REQUIRED_ABLATION_COST_BPS),
            "strict_comparison_tolerance": ACTIVE_EDGE_WIN_TOLERANCE,
            "requirements": {
                "full_brier": "strictly_lower",
                "full_expected_edge_mae": "no_worse",
                "full_total_active_log_edge": "strictly_greater",
                "full_minimum_fold_active_log_edge": "no_worse",
                "full_regime_consensus_gates": "pass",
            },
        },
        "comparisons": comparisons,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


__all__ = [
    "REGIME_CONSENSUS_GATES",
    "REQUIRED_ABLATION_COST_BPS",
    "RegimeConsensusScoringError",
    "apply_regime_consensus_gates",
    "evaluate_regime_consensus_ablation",
    "evaluate_regime_consensus_development",
    "score_regime_consensus_metrics",
]
