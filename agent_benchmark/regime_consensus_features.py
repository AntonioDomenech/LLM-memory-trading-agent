"""Frozen completed-close inputs and one-session labels for regime consensus.

The feature contract is deliberately small and fixed.  Three four-feature
heads describe AAPL, the broad market, and market sentiment respectively.
Every value is known after the completed close of decision session ``t`` and
is reused from the already-audited price and market-sentiment implementations.
There is no forward filling, interpolation, or partial-head fallback.

The isolated CASH outcome for decision ``t`` sells AAPL at adjusted open
``t+1`` and buys it back at adjusted open ``t+2``.  It is therefore mature at
the latter open.  Strict pre-fold masks identify causal label and per-head
eligibility without compacting away missing rows that must split HMM
sequences.
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

HEAD_NAMES: tuple[str, ...] = (
    "aapl_state",
    "market_state",
    "sentiment_state",
)

AAPL_STATE_FEATURE_COLUMNS: tuple[str, ...] = (
    "aapl_lr_5",
    "aapl_lr_20",
    "aapl_drawdown_60",
    "aapl_downside_rv_20",
)
MARKET_STATE_FEATURE_COLUMNS: tuple[str, ...] = (
    "qqq_lr_20",
    "qqq_lr_60",
    "iwm_lr_20",
    "spy_lr_20",
)
SENTIMENT_STATE_FEATURE_COLUMNS: tuple[str, ...] = (
    "vix_z_252",
    "vix_lr_5",
    "qqq_minus_iwm_lr_20",
    "qqq_rv_20",
)

# These tuples are positionally aligned with HEAD_NAMES.  Positive orientation
# means a larger feature value represents more stress; negative orientation
# means a smaller value represents more stress.
HEAD_FEATURE_COLUMNS: tuple[tuple[str, ...], ...] = (
    AAPL_STATE_FEATURE_COLUMNS,
    MARKET_STATE_FEATURE_COLUMNS,
    SENTIMENT_STATE_FEATURE_COLUMNS,
)
HEAD_ORIENTATIONS: tuple[tuple[int, ...], ...] = (
    (-1, -1, -1, +1),
    (-1, -1, -1, -1),
    (+1, +1, +1, +1),
)

REGIME_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for columns in HEAD_FEATURE_COLUMNS for column in columns
)
HEAD_READINESS_COLUMNS: tuple[str, ...] = tuple(
    f"{head_name}_ready" for head_name in HEAD_NAMES
)
ALL_HEADS_READY_COLUMN = "all_three_heads_ready"

REGIME_LABEL_COLUMNS: tuple[str, ...] = (
    "label_entry_date",
    "label_maturity_date",
    "label_entry_adjusted_open",
    "label_exit_adjusted_open",
    "aapl_forward_log_return_1",
    "cash_active_log_edge_5bps",
    "cash_active_log_edge_10bps",
    "cash_beats_long_5bps",
    "cash_beats_long_10bps",
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
    return index


def _cash_round_trip_log_factor(slippage_bps: float) -> float:
    rate = float(slippage_bps) / 10_000.0
    if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
        raise ValueError("slippage_bps must be finite and between 0 and 10,000")
    # CASH sells at mid*(1-rate), then buys back at mid*(1+rate).
    return math.log1p(-rate) - math.log1p(rate)


def _nullable_binary(values: pd.Series, valid: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="Int8")
    result.loc[valid] = values.loc[valid].astype(np.int8)
    return result


def build_regime_consensus_features(
    price_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return the exact twelve completed-close inputs in frozen order.

    The price fields are selected from :func:`build_downside_price_features`;
    IWM and VIX fields are selected from
    :func:`build_market_sentiment_features`.  That implementation aligns
    context by exact AAPL session date and deliberately leaves missing values
    as NaN.  This function does not add TNX, CFTC, news, dates, or raw prices.
    """

    prices = canonical_downside_price_frame(price_frame)
    price_features = build_downside_price_features(prices)
    sentiment_features = build_market_sentiment_features(prices, context_frame)
    if not price_features.index.equals(sentiment_features.index):
        raise ValueError("Price and market-sentiment feature dates do not align")

    sources = pd.concat([price_features, sentiment_features], axis=1)
    features = sources.loc[:, list(REGIME_FEATURE_COLUMNS)].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.astype(float)


def regime_head_readiness(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-head and all-three readiness without partial fallback."""

    missing = sorted(set(REGIME_FEATURE_COLUMNS).difference(feature_frame.columns))
    if missing:
        raise ValueError(f"feature_frame is missing required columns: {missing}")

    readiness = pd.DataFrame(index=feature_frame.index)
    for head_name, feature_columns in zip(
        HEAD_NAMES,
        HEAD_FEATURE_COLUMNS,
        strict=True,
    ):
        values = feature_frame.loc[:, list(feature_columns)].to_numpy(dtype=float)
        readiness[f"{head_name}_ready"] = np.isfinite(values).all(axis=1)
    readiness[ALL_HEADS_READY_COLUMN] = readiness.loc[
        :, list(HEAD_READINESS_COLUMNS)
    ].all(axis=1)
    return readiness.astype(bool)


def build_one_session_regime_labels(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Build isolated next-open-to-following-open CASH outcomes at 5/10 bps."""

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

    forward_log_return = pd.Series(np.nan, index=prices.index, dtype=float)
    forward_log_return.loc[outcome_valid] = np.log(
        exit_open.loc[outcome_valid] / entry_open.loc[outcome_valid]
    )
    edge_5 = _cash_round_trip_log_factor(5.0) - forward_log_return
    edge_10 = _cash_round_trip_log_factor(10.0) - forward_log_return

    labels = pd.DataFrame(
        {
            "label_entry_date": entry_date.where(entry_valid),
            "label_maturity_date": maturity_date.where(outcome_valid),
            "label_entry_adjusted_open": entry_open.where(entry_valid),
            "label_exit_adjusted_open": exit_open.where(outcome_valid),
            "aapl_forward_log_return_1": forward_log_return,
            "cash_active_log_edge_5bps": edge_5.where(outcome_valid),
            "cash_active_log_edge_10bps": edge_10.where(outcome_valid),
            "cash_beats_long_5bps": _nullable_binary(
                edge_5 > 0.0,
                outcome_valid,
            ),
            "cash_beats_long_10bps": _nullable_binary(
                edge_10 > 0.0,
                outcome_valid,
            ),
        },
        index=prices.index,
    )
    return labels.loc[:, list(REGIME_LABEL_COLUMNS)]


def entry_fill_years(feature_label_frame: pd.DataFrame) -> pd.Series:
    """Return nullable calendar years from actual entry-fill dates."""

    if "label_entry_date" not in feature_label_frame.columns:
        raise ValueError("feature_label_frame is missing label_entry_date")
    entries = pd.to_datetime(
        feature_label_frame["label_entry_date"],
        errors="coerce",
    )
    result = entries.dt.year.astype("Int16")
    result.name = "entry_fill_year"
    return result


def entry_fill_year_mask(
    feature_label_frame: pd.DataFrame,
    *,
    first_year: int,
    last_year: int,
) -> pd.Series:
    """Select labels whose actual entry fill falls in an inclusive year range."""

    if isinstance(first_year, bool) or not isinstance(first_year, int):
        raise ValueError("first_year must be an integer")
    if isinstance(last_year, bool) or not isinstance(last_year, int):
        raise ValueError("last_year must be an integer")
    if last_year < first_year:
        raise ValueError("last_year must be greater than or equal to first_year")
    years = entry_fill_years(feature_label_frame)
    selected = years.between(first_year, last_year).fillna(False).astype(bool)
    selected.name = "entry_fill_year_selected"
    return selected


def build_regime_consensus_feature_label_frame(
    price_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Combine the frozen twelve inputs, one-session labels, and readiness."""

    features = build_regime_consensus_features(price_frame, context_frame)
    labels = build_one_session_regime_labels(price_frame)
    if not features.index.equals(labels.index):
        raise ValueError("Feature and label dates do not align")
    readiness = regime_head_readiness(features)
    result = pd.concat([features, labels, readiness], axis=1)
    result["label_available"] = (
        result["label_maturity_date"].notna()
        & result["cash_active_log_edge_5bps"].notna()
        & result["cash_active_log_edge_10bps"].notna()
        & result["cash_beats_long_5bps"].notna()
        & result["cash_beats_long_10bps"].notna()
    )
    result["entry_fill_year"] = entry_fill_years(result)
    return result


def strict_pre_fold_label_mask(
    feature_label_frame: pd.DataFrame,
    *,
    fold_first_decision_date: DateLike,
) -> pd.Series:
    """Admit every outcome that matured strictly before a fold.

    This is the causal sample for the predictive baseline.  It deliberately
    imposes no feature-readiness condition.
    """

    required = {
        "label_maturity_date",
        "cash_active_log_edge_10bps",
        "cash_beats_long_10bps",
    }
    missing = sorted(required.difference(feature_label_frame.columns))
    if missing:
        raise ValueError(f"feature_label_frame is missing required columns: {missing}")

    decision_index = _normalized_decision_index(feature_label_frame)
    boundary = _normalize_timestamp(
        fold_first_decision_date,
        field_name="fold_first_decision_date",
    )
    maturity = pd.to_datetime(
        feature_label_frame["label_maturity_date"],
        errors="coerce",
    )

    eligible = (
        (decision_index < boundary)
        & maturity.notna().to_numpy()
        & (maturity.to_numpy(dtype="datetime64[ns]") < boundary.to_datetime64())
        & feature_label_frame["cash_active_log_edge_10bps"].notna().to_numpy()
        & feature_label_frame["cash_beats_long_10bps"].notna().to_numpy()
    )
    return pd.Series(
        eligible,
        index=feature_label_frame.index,
        dtype=bool,
        name="strict_pre_fold_label_eligible",
    )


def strict_pre_fold_head_training_mask(
    feature_label_frame: pd.DataFrame,
    *,
    fold_first_decision_date: DateLike,
    head_name: str,
) -> pd.Series:
    """Add one head's own four-feature readiness to the mature-label mask.

    Each HMM head trains on its own complete observations.  Missing inputs for
    another head do not remove this head's eligible case.  This mask is for
    count and eligibility audits: an HMM caller must still pass the full
    chronological array, including NaN rows, so gaps split sequences instead
    of being bridged.  Requiring all three heads is reserved for an actual
    trading decision.
    """

    if head_name not in HEAD_NAMES:
        raise ValueError(f"head_name must be one of {HEAD_NAMES}")
    position = HEAD_NAMES.index(head_name)
    feature_columns = HEAD_FEATURE_COLUMNS[position]
    readiness_column = HEAD_READINESS_COLUMNS[position]
    required = {*feature_columns, readiness_column}
    missing = sorted(required.difference(feature_label_frame.columns))
    if missing:
        raise ValueError(f"feature_label_frame is missing required columns: {missing}")

    values = feature_label_frame.loc[:, list(feature_columns)].to_numpy(dtype=float)
    derived_ready = pd.Series(
        np.isfinite(values).all(axis=1),
        index=feature_label_frame.index,
        dtype=bool,
        name=readiness_column,
    )
    actual = feature_label_frame[readiness_column]
    if actual.isna().any():
        raise ValueError(f"{readiness_column} cannot contain missing values")
    if not pd.api.types.is_bool_dtype(actual.dtype):
        raise ValueError(f"{readiness_column} must be boolean")
    actual_boolean = actual.astype(bool)
    if not actual_boolean.equals(derived_ready):
        raise ValueError(
            f"{readiness_column} is inconsistent with its feature inputs"
        )

    mask = strict_pre_fold_label_mask(
        feature_label_frame,
        fold_first_decision_date=fold_first_decision_date,
    )
    mask &= derived_ready
    mask.name = f"strict_pre_fold_{head_name}_eligible"
    return mask


def strict_pre_fold_training_mask(
    feature_label_frame: pd.DataFrame,
    *,
    fold_first_decision_date: DateLike,
    head_name: str,
) -> pd.Series:
    """Compatibility spelling for :func:`strict_pre_fold_head_training_mask`."""

    return strict_pre_fold_head_training_mask(
        feature_label_frame,
        fold_first_decision_date=fold_first_decision_date,
        head_name=head_name,
    )


__all__ = [
    "ALL_HEADS_READY_COLUMN",
    "AAPL_STATE_FEATURE_COLUMNS",
    "HEAD_FEATURE_COLUMNS",
    "HEAD_NAMES",
    "HEAD_ORIENTATIONS",
    "HEAD_READINESS_COLUMNS",
    "LABEL_ENTRY_OFFSET",
    "LABEL_MATURITY_OFFSET",
    "MARKET_STATE_FEATURE_COLUMNS",
    "ONE_SESSION_HORIZON",
    "REGIME_FEATURE_COLUMNS",
    "REGIME_LABEL_COLUMNS",
    "SENTIMENT_STATE_FEATURE_COLUMNS",
    "build_one_session_regime_labels",
    "build_regime_consensus_feature_label_frame",
    "build_regime_consensus_features",
    "entry_fill_year_mask",
    "entry_fill_years",
    "regime_head_readiness",
    "strict_pre_fold_head_training_mask",
    "strict_pre_fold_label_mask",
    "strict_pre_fold_training_mask",
]
