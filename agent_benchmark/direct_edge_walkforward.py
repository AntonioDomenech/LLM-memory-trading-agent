"""Frozen pre-2019 walk-forward policy for the direct CASH-edge model.

The module consumes a point-in-time frame; it never downloads data or chooses
an evaluation-period winner.  One price-only and, when possible, one
market-sentiment model are fitted before each fixed two-year decision block.
Every fitted model and preprocessing state then remains unchanged for the
whole block.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd

from .direct_edge_features import (
    DIRECT_EDGE_LABEL_COLUMNS,
    MARKET_SENTIMENT_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    price_only_chronological_training_mask,
    sentiment_chronological_training_mask,
)
from .direct_edge_gam import DirectEdgeGAM


DEVELOPMENT_LAST_DECISION: Final[pd.Timestamp] = pd.Timestamp("2018-12-31")
CASH_EPISODE_DECISION_ROWS: Final[int] = 5
MODEL_FAMILIES: Final[tuple[str, str]] = ("price_only", "market_sentiment")


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    first_year: int
    last_year: int

    def contains(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.year >= self.first_year) & (index.year <= self.last_year)


@dataclass(frozen=True)
class DirectEdgeCandidate:
    candidate_id: str
    probability_gate: float
    expected_edge_gate: float


WALK_FORWARD_FOLDS: Final[tuple[WalkForwardFold, ...]] = tuple(
    WalkForwardFold(f"{year}-{year + 1}", year, year + 1)
    for year in range(2005, 2019, 2)
)

DIRECT_EDGE_CANDIDATES: Final[tuple[DirectEdgeCandidate, ...]] = (
    DirectEdgeCandidate("p50_e0", 0.50, 0.0),
    DirectEdgeCandidate("p55_e0", 0.55, 0.0),
    DirectEdgeCandidate("p50_e25", 0.50, 0.0025),
    DirectEdgeCandidate("p55_e25", 0.55, 0.0025),
)


_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        *PRICE_FEATURE_COLUMNS,
        *MARKET_SENTIMENT_FEATURE_COLUMNS,
        *DIRECT_EDGE_LABEL_COLUMNS,
        "price_features_ready",
        "sentiment_features_ready",
        "sentiment_price_only_fallback",
    }
)


def candidate_cash_target_column(model_family: str, candidate_id: str) -> str:
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown direct-edge model family: {model_family!r}")
    if candidate_id not in {item.candidate_id for item in DIRECT_EDGE_CANDIDATES}:
        raise ValueError(f"Unknown frozen direct-edge candidate: {candidate_id!r}")
    return f"cash_target_{model_family}_{candidate_id}"


def five_session_cash_target(
    triggers: Sequence[object] | pd.Series | np.ndarray,
) -> np.ndarray:
    """Turn entry triggers into one continuous non-overlapping CASH state."""

    raw = np.asarray(triggers, dtype=object)
    if raw.ndim != 1:
        raise ValueError("CASH triggers must be one-dimensional")
    normalized = np.zeros(len(raw), dtype=bool)
    for position, value in enumerate(raw):
        if value is None or value is pd.NA:
            continue
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            pass
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("CASH triggers must contain only booleans or missing values")
        normalized[position] = bool(value)

    target = np.zeros(len(raw), dtype=np.int8)
    remaining = 0
    for position, trigger in enumerate(normalized):
        if remaining == 0 and trigger:
            remaining = CASH_EPISODE_DECISION_ROWS
        if remaining:
            target[position] = 1
            remaining -= 1
    return target


def _canonical_frame(feature_label_frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(feature_label_frame, pd.DataFrame):
        raise TypeError("feature_label_frame must be a pandas DataFrame")
    if feature_label_frame.empty:
        raise ValueError("feature_label_frame cannot be empty")
    missing = sorted(_REQUIRED_COLUMNS.difference(feature_label_frame.columns))
    if missing:
        raise ValueError(f"feature_label_frame is missing required columns: {missing}")

    try:
        index = pd.DatetimeIndex(pd.to_datetime(feature_label_frame.index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("feature_label_frame index must contain valid dates") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates:
        raise ValueError("feature_label_frame contains duplicate decision dates")
    if not index.is_monotonic_increasing:
        raise ValueError("feature_label_frame must be in chronological order")
    if bool((index > DEVELOPMENT_LAST_DECISION).any()):
        first = index[index > DEVELOPMENT_LAST_DECISION][0]
        raise ValueError(
            "Pre-2019 direct-edge walk-forward refuses post-2018 decision rows; "
            f"first forbidden date is {first.date().isoformat()}"
        )

    for name in (
        "price_features_ready",
        "sentiment_features_ready",
        "sentiment_price_only_fallback",
    ):
        values = feature_label_frame[name]
        if values.isna().any() or any(
            not isinstance(value, (bool, np.bool_)) for value in values.to_numpy()
        ):
            raise ValueError(f"{name} must contain non-missing booleans")

    result = feature_label_frame.copy()
    result.index = index
    result.index.name = "decision_date"
    return result


def _finite_rows(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    try:
        values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Direct-edge model inputs must be numeric") from exc
    return np.isfinite(values).all(axis=1)


def _sha256_lines(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _fit_metadata(training: pd.DataFrame) -> dict[str, str]:
    maturity = pd.to_datetime(training["label_maturity_date"], errors="raise")
    binary = training["cash_beats_long_10bps"].to_numpy(dtype=float)
    edge = training["cash_active_log_edge_10bps"].to_numpy(dtype=float)
    return {
        "training_start_date": training.index.min().date().isoformat(),
        "training_end_date": training.index.max().date().isoformat(),
        "maximum_label_maturity_date": maturity.max().date().isoformat(),
        "training_dates_sha256": _sha256_lines(
            [timestamp.date().isoformat() for timestamp in training.index]
        ),
        "binary_target_sha256": _sha256_lines(
            [float(value).hex() for value in binary]
        ),
        "edge_target_sha256": _sha256_lines(
            [float(value).hex() for value in edge]
        ),
    }


def _fit_model(
    frame: pd.DataFrame,
    *,
    family: str,
    fold: WalkForwardFold,
    first_decision: pd.Timestamp,
) -> tuple[DirectEdgeGAM, pd.DataFrame]:
    if family == "price_only":
        mask = price_only_chronological_training_mask(
            frame, as_of_date=first_decision
        )
        feature_names = PRICE_FEATURE_COLUMNS
    elif family == "market_sentiment":
        mask = sentiment_chronological_training_mask(
            frame, as_of_date=first_decision
        )
        feature_names = PRICE_FEATURE_COLUMNS + MARKET_SENTIMENT_FEATURE_COLUMNS
    else:  # pragma: no cover - internal invariant
        raise RuntimeError(f"Unexpected model family: {family}")

    training = frame.loc[mask].copy()
    if training.empty:
        raise ValueError(f"No eligible {family} training rows for fold {fold.fold_id}")
    model = DirectEdgeGAM().fit(
        training.loc[:, list(feature_names)],
        training["cash_beats_long_10bps"].to_numpy(dtype=float),
        training["cash_active_log_edge_10bps"].to_numpy(dtype=float),
        feature_names=feature_names,
        fit_metadata=_fit_metadata(training),
    )
    return model, training


def _model_hash(model: DirectEdgeGAM) -> str:
    value = model.model_sha256
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("Direct-edge model returned an invalid SHA-256")
    return value


def _predict_components(model: DirectEdgeGAM, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw = model.predict_components(features)
    if not isinstance(raw, Mapping):
        raise ValueError("Direct-edge prediction components must be a mapping")
    probability = np.asarray(raw["cash_beats_long_probability"], dtype=float)
    edge = np.asarray(raw["expected_edge_10bps"], dtype=float)
    if probability.shape != (len(features),) or edge.shape != (len(features),):
        raise ValueError("Direct-edge prediction components have invalid shapes")
    if (
        not np.isfinite(probability).all()
        or not np.isfinite(edge).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
    ):
        raise ValueError("Direct-edge prediction components are invalid")
    return probability, edge


def _write_training_metadata(
    output: pd.DataFrame,
    *,
    prefix: str,
    model_hash: str,
    training: pd.DataFrame,
) -> None:
    binary = training["cash_beats_long_10bps"].to_numpy(dtype=float)
    edge = training["cash_active_log_edge_10bps"].to_numpy(dtype=float)
    maturity = pd.to_datetime(training["label_maturity_date"], errors="raise")
    output[f"{prefix}_model_sha256"] = model_hash
    output[f"{prefix}_training_count"] = int(len(training))
    output[f"{prefix}_training_positive_count"] = int(binary.sum())
    output[f"{prefix}_training_max_decision_date"] = training.index.max()
    output[f"{prefix}_training_max_label_maturity_date"] = maturity.max()
    output[f"{prefix}_baseline_cash_win_probability"] = float(binary.mean())
    output[f"{prefix}_baseline_mean_net_edge"] = float(edge.mean())


def _write_price_metadata(
    output: pd.DataFrame,
    model: DirectEdgeGAM,
    training: pd.DataFrame,
) -> None:
    _write_training_metadata(
        output,
        prefix="price_only",
        model_hash=_model_hash(model),
        training=training,
    )


def _predict_fold(frame: pd.DataFrame, fold: WalkForwardFold) -> pd.DataFrame:
    in_fold = fold.contains(frame.index)
    if not bool(in_fold.any()):
        raise ValueError(f"No decision rows are available for fixed fold {fold.fold_id}")
    fill = frame.loc[in_fold].copy()
    first_decision = fill.index[0]

    price_model, price_training = _fit_model(
        frame,
        family="price_only",
        fold=fold,
        first_decision=first_decision,
    )
    price_hash = _model_hash(price_model)

    sentiment_model: DirectEdgeGAM | None = None
    sentiment_training = pd.DataFrame()
    sentiment_failure = ""
    try:
        sentiment_model, sentiment_training = _fit_model(
            frame,
            family="market_sentiment",
            fold=fold,
            first_decision=first_decision,
        )
        sentiment_hash: str | None = _model_hash(sentiment_model)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        sentiment_hash = None
        sentiment_failure = f"{type(exc).__name__}: {exc}"

    sentiment_mask = sentiment_chronological_training_mask(
        frame, as_of_date=first_decision
    )
    sentiment_attempted = frame.loc[sentiment_mask]

    output = pd.DataFrame(index=fill.index)
    output.index.name = "decision_date"
    output["fold_id"] = fold.fold_id
    output["fold_first_decision_date"] = first_decision
    output["fold_last_decision_date"] = fill.index[-1]
    _write_price_metadata(output, price_model, price_training)

    price_ready = (
        fill["price_features_ready"].to_numpy(dtype=bool)
        & _finite_rows(fill, PRICE_FEATURE_COLUMNS)
    )
    price_probability = np.full(len(fill), np.nan, dtype=float)
    price_edge = np.full(len(fill), np.nan, dtype=float)
    if price_ready.any():
        probability, edge = _predict_components(
            price_model, fill.loc[price_ready, list(PRICE_FEATURE_COLUMNS)]
        )
        price_probability[price_ready] = probability
        price_edge[price_ready] = edge

    sentiment_probability = price_probability.copy()
    sentiment_edge = price_edge.copy()
    used_fallback = np.ones(len(fill), dtype=bool)
    whole_fold_fallback = sentiment_model is None

    if sentiment_model is not None:
        sentiment_columns = PRICE_FEATURE_COLUMNS + MARKET_SENTIMENT_FEATURE_COLUMNS
        sentiment_ready = (
            price_ready
            & fill["sentiment_features_ready"].to_numpy(dtype=bool)
            & ~fill["sentiment_price_only_fallback"].to_numpy(dtype=bool)
            & _finite_rows(fill, MARKET_SENTIMENT_FEATURE_COLUMNS)
        )
        try:
            if sentiment_ready.any():
                probability, edge = _predict_components(
                    sentiment_model,
                    fill.loc[sentiment_ready, list(sentiment_columns)],
                )
                sentiment_probability[sentiment_ready] = probability
                sentiment_edge[sentiment_ready] = edge
                used_fallback[sentiment_ready] = False
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            # A failed fixed model cannot be repaired or partially substituted
            # inside a fill block: conservatively route the entire block to the
            # already-frozen price model.
            sentiment_failure = f"{type(exc).__name__}: {exc}"
            whole_fold_fallback = True
            sentiment_model = None
            sentiment_hash = None
            sentiment_probability = price_probability.copy()
            sentiment_edge = price_edge.copy()
            used_fallback[:] = True

    output["price_only_cash_win_probability"] = price_probability
    output["price_only_expected_net_edge"] = price_edge
    output["market_sentiment_cash_win_probability"] = sentiment_probability
    output["market_sentiment_expected_net_edge"] = sentiment_edge
    output["market_sentiment_used_price_fallback"] = used_fallback
    output["market_sentiment_whole_fold_fallback"] = whole_fold_fallback
    output["market_sentiment_fallback_reason"] = sentiment_failure
    output["market_sentiment_fitted_model_sha256"] = sentiment_hash or ""
    # This family-level hash always names the effective frozen model.  When a
    # sentiment fit is unavailable, the effective whole-fold model is exactly
    # the already-fitted price model.
    output["market_sentiment_model_sha256"] = sentiment_hash or price_hash
    output["market_sentiment_prediction_model_sha256"] = np.where(
        used_fallback,
        price_hash,
        sentiment_hash or price_hash,
    )
    output["market_sentiment_attempted_training_count"] = int(
        len(sentiment_attempted)
    )
    output["market_sentiment_attempted_training_max_decision_date"] = (
        sentiment_attempted.index.max() if len(sentiment_attempted) else pd.NaT
    )
    attempted_maturity = pd.to_datetime(
        sentiment_attempted["label_maturity_date"], errors="raise"
    )
    output["market_sentiment_attempted_training_max_label_maturity_date"] = (
        attempted_maturity.max() if len(attempted_maturity) else pd.NaT
    )

    if sentiment_model is None:
        for suffix in (
            "training_count",
            "training_positive_count",
            "training_max_decision_date",
            "training_max_label_maturity_date",
        ):
            output[f"market_sentiment_{suffix}"] = output[
                f"price_only_{suffix}"
            ]
        output["market_sentiment_baseline_cash_win_probability"] = output[
            "price_only_baseline_cash_win_probability"
        ]
        output["market_sentiment_baseline_mean_net_edge"] = output[
            "price_only_baseline_mean_net_edge"
        ]
    else:
        binary = sentiment_training["cash_beats_long_10bps"].to_numpy(dtype=float)
        edge = sentiment_training["cash_active_log_edge_10bps"].to_numpy(dtype=float)
        maturity = pd.to_datetime(
            sentiment_training["label_maturity_date"], errors="raise"
        )
        output["market_sentiment_training_count"] = int(len(sentiment_training))
        output["market_sentiment_training_positive_count"] = int(binary.sum())
        output["market_sentiment_training_max_decision_date"] = (
            sentiment_training.index.max()
        )
        output["market_sentiment_training_max_label_maturity_date"] = maturity.max()
        sentiment_probability_baseline = float(binary.mean())
        sentiment_edge_baseline = float(edge.mean())
        output["market_sentiment_baseline_cash_win_probability"] = np.where(
            used_fallback,
            output["price_only_baseline_cash_win_probability"],
            sentiment_probability_baseline,
        )
        output["market_sentiment_baseline_mean_net_edge"] = np.where(
            used_fallback,
            output["price_only_baseline_mean_net_edge"],
            sentiment_edge_baseline,
        )

    output["price_features_ready"] = price_ready
    for name in DIRECT_EDGE_LABEL_COLUMNS:
        if name in fill.columns:
            output[name] = fill[name]
    return output


def build_pre2019_direct_edge_predictions(
    feature_label_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return deterministic causal predictions and eight frozen candidates."""

    frame = _canonical_frame(feature_label_frame)
    output = pd.concat(
        [_predict_fold(frame, fold) for fold in WALK_FORWARD_FOLDS],
        axis=0,
    ).sort_index(kind="mergesort")
    if output.index.has_duplicates:
        raise RuntimeError("Fixed direct-edge walk-forward folds overlap")

    for family in MODEL_FAMILIES:
        probability = output[f"{family}_cash_win_probability"].to_numpy(dtype=float)
        edge = output[f"{family}_expected_net_edge"].to_numpy(dtype=float)
        finite = np.isfinite(probability) & np.isfinite(edge)
        for candidate in DIRECT_EDGE_CANDIDATES:
            trigger = (
                finite
                & (probability >= candidate.probability_gate)
                & (edge >= candidate.expected_edge_gate)
            )
            output[
                candidate_cash_target_column(family, candidate.candidate_id)
            ] = five_session_cash_target(trigger)
    return output


__all__ = [
    "CASH_EPISODE_DECISION_ROWS",
    "DEVELOPMENT_LAST_DECISION",
    "DIRECT_EDGE_CANDIDATES",
    "MODEL_FAMILIES",
    "WALK_FORWARD_FOLDS",
    "DirectEdgeCandidate",
    "WalkForwardFold",
    "build_pre2019_direct_edge_predictions",
    "candidate_cash_target_column",
    "five_session_cash_target",
]
