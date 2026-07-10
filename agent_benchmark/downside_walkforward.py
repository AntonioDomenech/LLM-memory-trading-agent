"""Frozen pre-2019 walk-forward plumbing for the downside-forest family.

This module deliberately does not download data, select a candidate, score a
ledger, or inspect any decision after 2018.  It consumes the point-in-time
frame produced by :mod:`agent_benchmark.downside_features`, fits one price-only
and one price-plus-CFTC model *before* each fixed two-year fill block, and then
keeps both fitted states unchanged for the whole block.

The augmented prediction is a conservative effective prediction.  On rows
explicitly marked ``cftc_price_only_fallback`` it is copied bit-for-bit from
the price-only prediction.  Missing CFTC values on a row that claims to be
ready are not imputed and cannot trigger CASH.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import pandas as pd

from .downside_features import (
    CFTC_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    chronological_training_mask,
)
from .downside_forest import DownsideForest, ForestConfig, LABEL_LOG_RETURN_CUTOFF


DEVELOPMENT_LAST_DECISION: Final[pd.Timestamp] = pd.Timestamp("2018-12-31")
# Absolute probabilities are not comparable across expanding folds with
# different historical crash prevalence.  Each gate is instead a fixed
# multiple of the causal training prevalence stored by that fold's model.
RISK_MULTIPLE_GATES: Final[tuple[float, ...]] = (1.10, 1.20, 1.30, 1.40)
CASH_EPISODE_DECISION_ROWS: Final[int] = 5
MODEL_FAMILIES: Final[tuple[str, str]] = ("price_only", "price_cftc")


@dataclass(frozen=True)
class WalkForwardFold:
    """One immutable two-year out-of-fold prediction block."""

    fold_id: str
    first_year: int
    last_year: int

    def contains(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.year >= self.first_year) & (index.year <= self.last_year)


WALK_FORWARD_FOLDS: Final[tuple[WalkForwardFold, ...]] = tuple(
    WalkForwardFold(f"{first_year}-{first_year + 1}", first_year, first_year + 1)
    for first_year in range(2005, 2019, 2)
)


_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        *PRICE_FEATURE_COLUMNS,
        *CFTC_FEATURE_COLUMNS,
        "cftc_price_only_fallback",
        "label_maturity_date",
        "crash_label_5session",
        "aapl_forward_log_return_5",
    }
)


def candidate_cash_target_column(model_family: str, risk_multiple: float) -> str:
    """Return the stable column name for one of the eight frozen candidates."""

    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {model_family!r}")
    gate = float(risk_multiple)
    if not any(gate == expected for expected in RISK_MULTIPLE_GATES):
        raise ValueError(f"Risk-multiple gate is not frozen: {risk_multiple!r}")
    return f"cash_target_{model_family}_r{int(round(gate * 100)):03d}"


def five_session_cash_target(
    triggers: Sequence[object] | pd.Series | np.ndarray,
) -> np.ndarray:
    """Convert causal entry triggers into non-overlapping five-row episodes.

    A trigger starts CASH on that decision row and the following four decision
    rows.  Triggers observed while already in CASH are ignored.  Missing values
    are false.  A final episode may be truncated only by the supplied array's
    boundary; no look-ahead is used to suppress it.
    """

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

    target = np.zeros(len(normalized), dtype=np.int8)
    rows_remaining = 0
    for position, trigger in enumerate(normalized):
        if rows_remaining == 0 and trigger:
            rows_remaining = CASH_EPISODE_DECISION_ROWS
        if rows_remaining:
            target[position] = 1
            rows_remaining -= 1
    return target


def _canonical_frame(feature_label_frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(feature_label_frame, pd.DataFrame):
        raise TypeError("feature_label_frame must be a pandas DataFrame")
    missing = sorted(_REQUIRED_COLUMNS.difference(feature_label_frame.columns))
    if missing:
        raise ValueError(f"feature_label_frame is missing required columns: {missing}")
    if feature_label_frame.empty:
        raise ValueError("feature_label_frame cannot be empty")

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
            "Pre-2019 walk-forward refuses post-2018 decision rows; "
            f"first forbidden date is {first.date().isoformat()}"
        )

    fallback = feature_label_frame["cftc_price_only_fallback"]
    if fallback.isna().any() or any(
        not isinstance(value, (bool, np.bool_)) for value in fallback.to_numpy()
    ):
        raise ValueError("cftc_price_only_fallback must contain non-missing booleans")

    result = feature_label_frame.copy()
    result.index = index
    result.index.name = "decision_date"
    return result


def _finite_rows(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    try:
        matrix = frame.loc[:, list(columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Model feature and return columns must be numeric") from exc
    return np.isfinite(matrix).all(axis=1)


def _fit_fold_model(
    frame: pd.DataFrame,
    *,
    first_decision: pd.Timestamp,
    fold: WalkForwardFold,
    model_family: str,
) -> tuple[DownsideForest | None, pd.DataFrame]:
    require_cftc = model_family == "price_cftc"
    mask = chronological_training_mask(
        frame,
        as_of_date=first_decision,
        require_cftc=require_cftc,
        strictly_before_as_of=True,
    )
    mask &= pd.Series(
        np.isfinite(
            pd.to_numeric(
                frame["aapl_forward_log_return_5"], errors="coerce"
            ).to_numpy(dtype=float)
        ),
        index=frame.index,
    )
    training = frame.loc[mask].copy()
    feature_names = (
        PRICE_FEATURE_COLUMNS
        if model_family == "price_only"
        else PRICE_FEATURE_COLUMNS + CFTC_FEATURE_COLUMNS
    )
    namespace = (
        "downside-walkforward-v1|"
        f"{model_family}|{fold.fold_id}|strictly-before-{first_decision.date().isoformat()}"
    )
    minimum_bootstrap_rows = 2 * ForestConfig().min_samples_leaf
    bootstrap_rows = int(
        math.ceil(len(training) * ForestConfig().bootstrap_fraction)
    )
    if require_cftc and bootstrap_rows < minimum_bootstrap_rows:
        # Early VIX/CFTC history can legitimately be too sparse for the fixed
        # forest.  The augmented policy then becomes the exact price policy
        # for the whole fold.  Never weaken min_samples_leaf or manufacture
        # sentiment values merely to force a fit.
        return None, training
    try:
        model = DownsideForest().fit(
            training.loc[:, list(feature_names)],
            training["aapl_forward_log_return_5"].to_numpy(dtype=float),
            namespace=namespace,
        )
    except ValueError as exc:
        raise ValueError(
            f"Cannot fit {model_family} model for fold {fold.fold_id}: {exc}"
        ) from exc
    return model, training


def _training_metadata(
    output: pd.DataFrame,
    *,
    prefix: str,
    model: DownsideForest | None,
    training: pd.DataFrame,
) -> None:
    maturity = pd.to_datetime(training["label_maturity_date"], errors="raise")
    output[f"{prefix}_model_status"] = (
        "fitted" if model is not None else "insufficient_training_price_fallback"
    )
    output[f"{prefix}_model_sha256"] = (
        model.model_sha256 if model is not None else None
    )
    output[f"{prefix}_training_count"] = int(len(training))
    output[f"{prefix}_training_positive_count"] = (
        int(model.training_positive_count)
        if model is not None
        else int(
            (
                pd.to_numeric(
                    training["aapl_forward_log_return_5"], errors="raise"
                ).to_numpy(dtype=float)
                <= LABEL_LOG_RETURN_CUTOFF
            ).sum()
        )
    )
    output[f"{prefix}_training_max_decision_date"] = training.index.max()
    output[f"{prefix}_training_max_label_maturity_date"] = maturity.max()
    output[f"{prefix}_baseline_downside_probability"] = (
        float(model.global_prevalence) if model is not None else np.nan
    )
    output[f"{prefix}_baseline_mean_clipped_return"] = (
        float(model.global_mean_clipped_return) if model is not None else np.nan
    )


def _predict_fold(
    frame: pd.DataFrame,
    fold: WalkForwardFold,
) -> pd.DataFrame:
    fold_mask = fold.contains(frame.index)
    if not bool(fold_mask.any()):
        raise ValueError(f"No decision rows are available for fixed fold {fold.fold_id}")
    fill = frame.loc[fold_mask].copy()
    first_decision = fill.index[0]
    price_model, price_training = _fit_fold_model(
        frame,
        first_decision=first_decision,
        fold=fold,
        model_family="price_only",
    )
    cftc_model, cftc_training = _fit_fold_model(
        frame,
        first_decision=first_decision,
        fold=fold,
        model_family="price_cftc",
    )

    output = pd.DataFrame(index=fill.index)
    output.index.name = "decision_date"
    output["fold_id"] = fold.fold_id
    output["fold_first_decision_date"] = first_decision
    output["fold_last_decision_date"] = fill.index[-1]
    _training_metadata(
        output,
        prefix="price_only",
        model=price_model,
        training=price_training,
    )
    _training_metadata(
        output,
        prefix="price_cftc",
        model=cftc_model,
        training=cftc_training,
    )

    price_ready = _finite_rows(fill, PRICE_FEATURE_COLUMNS)
    price_probability = np.full(len(fill), np.nan, dtype=float)
    price_mean = np.full(len(fill), np.nan, dtype=float)
    if price_ready.any():
        components = price_model.predict_components(
            fill.loc[price_ready, list(PRICE_FEATURE_COLUMNS)]
        )
        price_probability[price_ready] = components["downside_probability"]
        price_mean[price_ready] = components["mean_clipped_return"]

    data_fallback = fill["cftc_price_only_fallback"].to_numpy(dtype=bool)
    cftc_finite = _finite_rows(fill, CFTC_FEATURE_COLUMNS)
    augmented_probability = np.full(len(fill), np.nan, dtype=float)
    augmented_mean = np.full(len(fill), np.nan, dtype=float)
    if cftc_model is None:
        effective_fallback = np.ones(len(fill), dtype=bool)
        identity_fallback = price_ready
        fallback_reason = np.full(
            len(fill), "insufficient_fold_training", dtype=object
        )
    else:
        effective_fallback = data_fallback.copy()
        identity_fallback = price_ready & data_fallback
        fallback_reason = np.where(
            data_fallback, "cftc_data_unavailable", "none"
        ).astype(object)
        augmented_ready = price_ready & ~data_fallback & cftc_finite
        if augmented_ready.any():
            augmented_columns = PRICE_FEATURE_COLUMNS + CFTC_FEATURE_COLUMNS
            components = cftc_model.predict_components(
                fill.loc[augmented_ready, list(augmented_columns)]
            )
            augmented_probability[augmented_ready] = components[
                "downside_probability"
            ]
            augmented_mean[augmented_ready] = components["mean_clipped_return"]

    # Every fallback is an identity operation, never zero/mean imputation.
    augmented_probability[identity_fallback] = price_probability[identity_fallback]
    augmented_mean[identity_fallback] = price_mean[identity_fallback]

    price_baseline = np.full(
        len(fill), float(price_model.global_prevalence), dtype=float
    )
    augmented_baseline = np.full(
        len(fill),
        (
            float(cftc_model.global_prevalence)
            if cftc_model is not None
            else float(price_model.global_prevalence)
        ),
        dtype=float,
    )
    # The predictive baseline must follow the same causal route as the
    # effective prediction.  On an explicit CFTC fallback row the augmented
    # candidate is literally the price model, so its climatology is too.
    effective_augmented_baseline = augmented_baseline.copy()
    effective_augmented_baseline[identity_fallback] = price_baseline[
        identity_fallback
    ]

    output["price_only_downside_probability"] = price_probability
    output["price_only_predicted_mean_clipped_return"] = price_mean
    output["price_cftc_downside_probability"] = augmented_probability
    output["price_cftc_predicted_mean_clipped_return"] = augmented_mean
    output["price_cftc_effective_baseline_downside_probability"] = (
        effective_augmented_baseline
    )
    output["price_cftc_used_fallback"] = effective_fallback
    output["price_cftc_fallback_reason"] = fallback_reason
    output["cftc_data_price_only_fallback"] = data_fallback
    output["price_features_ready"] = price_ready
    output["cftc_features_finite"] = cftc_finite

    # Keep outcome fields for later scoring.  They are never passed to predict.
    for name in (
        "label_entry_date",
        "label_maturity_date",
        "aapl_forward_simple_return_5",
        "aapl_forward_log_return_5",
        "crash_label_5session",
        "cash_active_log_edge_5bps",
        "cash_active_log_edge_10bps",
    ):
        if name in fill.columns:
            output[name] = fill[name]
    return output


def build_pre2019_walkforward_predictions(
    feature_label_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Fit the 14 frozen fold models and return causal 2005-2018 predictions.

    The input may contain earlier rows for training, but any decision row after
    2018 is rejected.  Each model uses only rows whose label maturity date is
    *strictly before* its fill block's first decision date.
    """

    frame = _canonical_frame(feature_label_frame)
    blocks = [_predict_fold(frame, fold) for fold in WALK_FORWARD_FOLDS]
    output = pd.concat(blocks, axis=0).sort_index(kind="mergesort")
    if output.index.has_duplicates:
        raise RuntimeError("Fixed walk-forward folds unexpectedly overlap")

    for family in MODEL_FAMILIES:
        probability = output[f"{family}_downside_probability"].to_numpy(dtype=float)
        baseline_column = (
            "price_only_baseline_downside_probability"
            if family == "price_only"
            else "price_cftc_effective_baseline_downside_probability"
        )
        baseline = output[baseline_column].to_numpy(dtype=float)
        finite = (
            np.isfinite(probability)
            & np.isfinite(baseline)
            & (baseline > 0.0)
        )
        risk_multiple = np.full(len(output), np.nan, dtype=float)
        risk_multiple[finite] = probability[finite] / baseline[finite]
        output[f"{family}_downside_risk_multiple"] = risk_multiple
        for gate in RISK_MULTIPLE_GATES:
            trigger = finite & (risk_multiple >= gate)
            output[candidate_cash_target_column(family, gate)] = (
                five_session_cash_target(trigger)
            )
    return output


__all__ = [
    "CASH_EPISODE_DECISION_ROWS",
    "DEVELOPMENT_LAST_DECISION",
    "MODEL_FAMILIES",
    "RISK_MULTIPLE_GATES",
    "WALK_FORWARD_FOLDS",
    "WalkForwardFold",
    "build_pre2019_walkforward_predictions",
    "candidate_cash_target_column",
    "five_session_cash_target",
]
