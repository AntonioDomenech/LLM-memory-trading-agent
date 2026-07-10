"""Causal one-session features and targets for the frozen rare-loss model.

Every feature is known after the completed AAPL close on decision session
``t``.  A CASH decision made then sells AAPL at adjusted open ``t+1`` and buys
it back at adjusted open ``t+2``.  The target therefore matures only at the
second future open.  Missing rows are retained; this module never compacts a
calendar or fills market/context observations.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from .direct_edge_features import build_market_sentiment_features
from .downside_features import (
    build_downside_price_features,
    canonical_downside_price_frame,
)


LABEL_ENTRY_OFFSET = 1
LABEL_MATURITY_OFFSET = 2
ONE_SESSION_HORIZON = 1
SEVERE_SIMPLE_RETURN_THRESHOLD = -0.025
CASH_EDGE_SLIPPAGE_BPS = 10.0
REGRESSION_EDGE_CLIP = 0.08

CORE_FEATURE_COLUMNS: tuple[str, ...] = (
    "aapl_lr_1",
    "aapl_lr_5",
    "aapl_lr_20",
    "aapl_intraday_lr",
    "aapl_gap_lr",
    "aapl_downside_rv_20",
    "aapl_drawdown_60",
    "aapl_drawdown_252",
    "aapl_minus_qqq_lr_5",
    "aapl_minus_qqq_lr_20",
    "qqq_lr_5",
    "qqq_lr_20",
    "qqq_lr_60",
    "qqq_rv_20",
    "qqq_minus_spy_lr_20",
)

SENTIMENT_FEATURE_COLUMNS: tuple[str, ...] = (
    "iwm_lr_5",
    "iwm_lr_20",
    "qqq_minus_iwm_lr_20",
    "vix_z_252",
    "vix_lr_5",
)

RARE_LOSS_FEATURE_COLUMNS: tuple[str, ...] = (
    *CORE_FEATURE_COLUMNS,
    *SENTIMENT_FEATURE_COLUMNS,
)

RARE_LOSS_LABEL_COLUMNS: tuple[str, ...] = (
    "label_entry_date",
    "label_maturity_date",
    "label_entry_adjusted_open",
    "label_exit_adjusted_open",
    "aapl_forward_simple_return_1",
    "aapl_forward_log_return_1",
    "severe_loss_label_1session",
    "cash_active_log_edge_10bps",
    "cash_beats_long_10bps",
    "cash_active_log_edge_10bps_clipped",
)

DateLike = str | date | datetime | pd.Timestamp


def _normalize_timestamp(value: DateLike, *, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid date") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} cannot be missing")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _normalized_decision_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("feature_label_frame index must contain valid dates") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates:
        raise ValueError("feature_label_frame contains duplicate decision dates")
    if not index.is_monotonic_increasing:
        raise ValueError("feature_label_frame must be chronological")
    return index


def _cash_round_trip_log_factor(slippage_bps: float) -> float:
    rate = float(slippage_bps) / 10_000.0
    if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
        raise ValueError("slippage_bps must be finite and between 0 and 10,000")
    return math.log1p(-rate) - math.log1p(rate)


def _nullable_binary(values: pd.Series, valid: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="Int8")
    result.loc[valid] = values.loc[valid].astype(np.int8)
    return result


def _numeric_vector(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.to_numpy(dtype=float, na_value=np.nan)


def build_rare_loss_features(
    price_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build the frozen 15 AAPL-core plus five sentiment features.

    Price returns, drawdowns, and rolling volatilities include information only
    through close ``t``.  IWM and VIX are aligned by the exact AAPL session
    date by the existing audited context builder.  No TNX field is selected.
    """

    prices = canonical_downside_price_frame(price_frame)
    price_source = build_downside_price_features(prices)
    core = price_source.loc[:, list(CORE_FEATURE_COLUMNS)]
    sentiment_source = build_market_sentiment_features(prices, context_frame)
    if not core.index.equals(sentiment_source.index):
        raise ValueError("Price and sentiment feature dates do not align")
    sentiment = sentiment_source.loc[:, list(SENTIMENT_FEATURE_COLUMNS)]
    features = pd.concat([core, sentiment], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.loc[:, list(RARE_LOSS_FEATURE_COLUMNS)].astype(float)


def build_one_session_rare_loss_labels(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Build severe-loss, ordinary CASH-win, and clipped edge targets."""

    prices = canonical_downside_price_frame(price_frame)
    opens = prices["aapl_adj_open"]
    dates = pd.Series(prices.index, index=prices.index, dtype="datetime64[ns]")
    entry_date = dates.shift(-LABEL_ENTRY_OFFSET)
    maturity_date = dates.shift(-LABEL_MATURITY_OFFSET)
    entry_open = opens.shift(-LABEL_ENTRY_OFFSET)
    exit_open = opens.shift(-LABEL_MATURITY_OFFSET)
    entry_valid = entry_date.notna() & entry_open.notna()
    outcome_valid = (
        entry_valid
        & maturity_date.notna()
        & exit_open.notna()
    )

    simple_return = pd.Series(np.nan, index=prices.index, dtype=float)
    log_return = pd.Series(np.nan, index=prices.index, dtype=float)
    simple_return.loc[outcome_valid] = (
        exit_open.loc[outcome_valid] / entry_open.loc[outcome_valid] - 1.0
    )
    log_return.loc[outcome_valid] = np.log(
        exit_open.loc[outcome_valid] / entry_open.loc[outcome_valid]
    )
    active_edge = _cash_round_trip_log_factor(CASH_EDGE_SLIPPAGE_BPS) - log_return
    clipped_edge = active_edge.clip(-REGRESSION_EDGE_CLIP, REGRESSION_EDGE_CLIP)

    labels = pd.DataFrame(
        {
            "label_entry_date": entry_date.where(entry_valid),
            "label_maturity_date": maturity_date.where(outcome_valid),
            "label_entry_adjusted_open": entry_open.where(entry_valid),
            "label_exit_adjusted_open": exit_open.where(outcome_valid),
            "aapl_forward_simple_return_1": simple_return,
            "aapl_forward_log_return_1": log_return,
            "severe_loss_label_1session": _nullable_binary(
                simple_return <= SEVERE_SIMPLE_RETURN_THRESHOLD,
                outcome_valid,
            ),
            "cash_active_log_edge_10bps": active_edge.where(outcome_valid),
            "cash_beats_long_10bps": _nullable_binary(
                active_edge > 0.0,
                outcome_valid,
            ),
            "cash_active_log_edge_10bps_clipped": clipped_edge.where(
                outcome_valid
            ),
        },
        index=prices.index,
    )
    return labels.loc[:, list(RARE_LOSS_LABEL_COLUMNS)]


def rare_loss_feature_readiness(feature_frame: pd.DataFrame) -> pd.Series:
    """Return exact all-20-feature readiness; partial fallback is forbidden."""

    missing = sorted(set(RARE_LOSS_FEATURE_COLUMNS).difference(feature_frame.columns))
    if missing:
        raise ValueError(f"feature_frame is missing required columns: {missing}")
    ready = np.isfinite(
        feature_frame.loc[:, list(RARE_LOSS_FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all(axis=1)
    return pd.Series(ready, index=feature_frame.index, dtype=bool, name="features_ready")


def entry_fill_years(feature_label_frame: pd.DataFrame) -> pd.Series:
    if "label_entry_date" not in feature_label_frame.columns:
        raise ValueError("feature_label_frame is missing label_entry_date")
    years = pd.to_datetime(
        feature_label_frame["label_entry_date"], errors="coerce"
    ).dt.year.astype("Int16")
    years.name = "entry_fill_year"
    return years


def entry_fill_year_mask(
    feature_label_frame: pd.DataFrame,
    *,
    first_year: int,
    last_year: int,
) -> pd.Series:
    if isinstance(first_year, bool) or not isinstance(first_year, int):
        raise ValueError("first_year must be an integer")
    if isinstance(last_year, bool) or not isinstance(last_year, int):
        raise ValueError("last_year must be an integer")
    if last_year < first_year:
        raise ValueError("last_year must be greater than or equal to first_year")
    selected = entry_fill_years(feature_label_frame).between(
        first_year, last_year
    ).fillna(False).astype(bool)
    selected.name = "entry_fill_year_selected"
    return selected


def build_rare_loss_feature_label_frame(
    price_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Combine frozen features, one-session targets, and readiness audits."""

    features = build_rare_loss_features(price_frame, context_frame)
    labels = build_one_session_rare_loss_labels(price_frame)
    if not features.index.equals(labels.index):
        raise ValueError("Feature and label dates do not align")
    result = pd.concat([features, labels], axis=1)
    result["features_ready"] = rare_loss_feature_readiness(result)
    result["label_available"] = (
        result["label_entry_date"].notna()
        & result["label_maturity_date"].notna()
        & result["label_entry_adjusted_open"].notna()
        & result["label_exit_adjusted_open"].notna()
        & result["aapl_forward_simple_return_1"].notna()
        & result["severe_loss_label_1session"].notna()
        & result["cash_active_log_edge_10bps"].notna()
        & result["cash_beats_long_10bps"].notna()
        & result["cash_active_log_edge_10bps_clipped"].notna()
    )
    result["training_ready"] = (
        result["features_ready"] & result["label_available"]
    )
    result["entry_fill_year"] = entry_fill_years(result)
    return result


def strict_pre_fold_training_mask(
    feature_label_frame: pd.DataFrame,
    *,
    fold_first_decision_date: DateLike,
) -> pd.Series:
    """Select complete rows whose outcomes matured strictly before a fold."""

    required = {
        *RARE_LOSS_FEATURE_COLUMNS,
        *RARE_LOSS_LABEL_COLUMNS,
        "features_ready",
        "label_available",
    }
    missing = sorted(required.difference(feature_label_frame.columns))
    if missing:
        raise ValueError(f"feature_label_frame is missing required columns: {missing}")
    index = _normalized_decision_index(feature_label_frame)
    boundary = _normalize_timestamp(
        fold_first_decision_date,
        field_name="fold_first_decision_date",
    )
    derived_ready = rare_loss_feature_readiness(feature_label_frame)
    actual_ready = feature_label_frame["features_ready"]
    if actual_ready.isna().any() or not pd.api.types.is_bool_dtype(actual_ready.dtype):
        raise ValueError("features_ready must contain non-missing booleans")
    if not actual_ready.astype(bool).equals(derived_ready):
        raise ValueError("features_ready is inconsistent with feature values")
    label_available = feature_label_frame["label_available"]
    if label_available.isna().any() or not pd.api.types.is_bool_dtype(
        label_available.dtype
    ):
        raise ValueError("label_available must contain non-missing booleans")
    entry = pd.to_datetime(
        feature_label_frame["label_entry_date"], errors="coerce"
    )
    maturity = pd.to_datetime(
        feature_label_frame["label_maturity_date"], errors="coerce"
    )
    entry_open = _numeric_vector(
        feature_label_frame["label_entry_adjusted_open"]
    )
    exit_open = _numeric_vector(
        feature_label_frame["label_exit_adjusted_open"]
    )
    simple_return = _numeric_vector(
        feature_label_frame["aapl_forward_simple_return_1"]
    )
    log_return = _numeric_vector(
        feature_label_frame["aapl_forward_log_return_1"]
    )
    severe = _numeric_vector(feature_label_frame["severe_loss_label_1session"])
    active_edge = _numeric_vector(
        feature_label_frame["cash_active_log_edge_10bps"]
    )
    ordinary = _numeric_vector(feature_label_frame["cash_beats_long_10bps"])
    clipped = _numeric_vector(
        feature_label_frame["cash_active_log_edge_10bps_clipped"]
    )
    derived_label_available = (
        entry.notna().to_numpy()
        & maturity.notna().to_numpy()
        & np.isfinite(entry_open)
        & (entry_open > 0.0)
        & np.isfinite(exit_open)
        & (exit_open > 0.0)
        & np.isfinite(simple_return)
        & np.isfinite(log_return)
        & np.isfinite(severe)
        & np.isfinite(active_edge)
        & np.isfinite(ordinary)
        & np.isfinite(clipped)
    )
    if not np.array_equal(
        label_available.to_numpy(dtype=bool), derived_label_available
    ):
        raise ValueError("label_available is inconsistent with target values")
    if not np.isin(severe[derived_label_available], (0.0, 1.0)).all() or not np.isin(
        ordinary[derived_label_available], (0.0, 1.0)
    ).all():
        raise ValueError("Rare-loss binary targets must be exact zero/one values")
    decision_values = index.to_numpy(dtype="datetime64[ns]")
    entry_values = entry.to_numpy(dtype="datetime64[ns]")
    maturity_values = maturity.to_numpy(dtype="datetime64[ns]")
    if not (
        (entry_values[derived_label_available] > decision_values[derived_label_available]).all()
        and (
            maturity_values[derived_label_available]
            > entry_values[derived_label_available]
        ).all()
    ):
        raise ValueError("Target dates do not follow decision < entry < maturity")
    expected_simple = (
        exit_open[derived_label_available] / entry_open[derived_label_available] - 1.0
    )
    expected_log = np.log(
        exit_open[derived_label_available] / entry_open[derived_label_available]
    )
    if not np.allclose(
        simple_return[derived_label_available],
        expected_simple,
        rtol=0.0,
        atol=1e-12,
    ) or not np.allclose(
        log_return[derived_label_available],
        expected_log,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Forward returns are inconsistent with adjusted opens")
    if not np.array_equal(
        severe[derived_label_available],
        (simple_return[derived_label_available] <= SEVERE_SIMPLE_RETURN_THRESHOLD).astype(
            float
        ),
    ):
        raise ValueError("severe-loss target is inconsistent with simple return")
    if not np.array_equal(
        ordinary[derived_label_available],
        (active_edge[derived_label_available] > 0.0).astype(float),
    ):
        raise ValueError("ordinary CASH target is inconsistent with active edge")
    expected_edge = (
        _cash_round_trip_log_factor(CASH_EDGE_SLIPPAGE_BPS)
        - log_return[derived_label_available]
    )
    if not np.allclose(
        active_edge[derived_label_available],
        expected_edge,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("active edge is inconsistent with the 10bps contract")
    if not np.array_equal(
        clipped[derived_label_available],
        np.clip(
            active_edge[derived_label_available],
            -REGRESSION_EDGE_CLIP,
            REGRESSION_EDGE_CLIP,
        ),
    ):
        raise ValueError("clipped regression target is inconsistent with active edge")
    selected = (
        (index < boundary)
        & maturity.notna().to_numpy()
        & (maturity.to_numpy(dtype="datetime64[ns]") < boundary.to_datetime64())
        & derived_ready.to_numpy(dtype=bool)
        & derived_label_available
    )
    return pd.Series(
        selected,
        index=feature_label_frame.index,
        dtype=bool,
        name="strict_pre_fold_training_eligible",
    )


__all__ = [
    "CASH_EDGE_SLIPPAGE_BPS",
    "CORE_FEATURE_COLUMNS",
    "LABEL_ENTRY_OFFSET",
    "LABEL_MATURITY_OFFSET",
    "ONE_SESSION_HORIZON",
    "RARE_LOSS_FEATURE_COLUMNS",
    "RARE_LOSS_LABEL_COLUMNS",
    "REGRESSION_EDGE_CLIP",
    "SENTIMENT_FEATURE_COLUMNS",
    "SEVERE_SIMPLE_RETURN_THRESHOLD",
    "build_one_session_rare_loss_labels",
    "build_rare_loss_feature_label_frame",
    "build_rare_loss_features",
    "entry_fill_year_mask",
    "entry_fill_years",
    "rare_loss_feature_readiness",
    "strict_pre_fold_training_mask",
]
