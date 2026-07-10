"""Point-in-time features and direct CASH-edge labels for AAPL.

This module is deliberately pure: it performs no downloads, fitting, or
evaluation-period filtering.  Each row represents a decision after the
completed close of AAPL session ``t``.  A hypothetical CASH episode sells at
the adjusted AAPL open at ``t+1`` and buys back at the adjusted open at
``t+6``.  The resulting target therefore cannot enter a training set until
the open at ``t+6`` has occurred.

The 24 price-only fields are imported from :mod:`downside_features`, so the
two model families share one exact implementation.  Market-sentiment context
is aligned by exact AAPL session date.  It is never forward-filled or replaced
with zero: a missing/nonfinite IWM, VIX, or TNX observation produces NaN
features and an explicit price-only fallback.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Sequence

import numpy as np
import pandas as pd

from .downside_features import (
    PRICE_FEATURE_COLUMNS,
    build_downside_price_features,
    canonical_downside_price_frame,
)


LABEL_ENTRY_OFFSET = 1
LABEL_MATURITY_OFFSET = 6
FIVE_SESSION_HORIZON = 5

MARKET_SENTIMENT_FEATURE_COLUMNS: tuple[str, ...] = (
    "spy_lr_5",
    "spy_drawdown_60",
    "qqq_drawdown_60",
    "iwm_lr_5",
    "iwm_lr_20",
    "iwm_lr_60",
    "iwm_rv_20",
    "qqq_minus_iwm_lr_20",
    "vix_log_level",
    "vix_lr_5",
    "vix_lr_20",
    "vix_z_252",
    "tnx_level",
    "tnx_delta_5",
    "tnx_delta_20",
)

DIRECT_EDGE_LABEL_COLUMNS: tuple[str, ...] = (
    "label_entry_date",
    "label_maturity_date",
    "label_entry_adjusted_open",
    "label_exit_adjusted_open",
    "aapl_forward_log_return_5",
    "cash_active_log_edge_5bps",
    "cash_active_log_edge_10bps",
    "cash_beats_long_10bps",
)

CONTEXT_VALUE_COLUMNS: tuple[str, ...] = (
    "iwm_adj_close",
    "vix_close",
    "tnx_close",
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


def _date_index(frame: pd.DataFrame, *, allow_duplicates: bool) -> pd.DataFrame:
    data = frame.copy()
    date_column = _find_column(data.columns, ("date", "datetime", "timestamp"))
    if date_column is not None:
        data[date_column] = pd.to_datetime(data[date_column], errors="raise")
        data = data.set_index(date_column)
    try:
        index = pd.DatetimeIndex(pd.to_datetime(data.index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Context data index must contain valid dates") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    data.index = index.normalize()
    data = data.sort_index(kind="stable")
    if not allow_duplicates and data.index.has_duplicates:
        raise ValueError("Wide context data contains duplicate dates")
    return data


def _column_key(value: object) -> str:
    text = str(value).strip().lower().replace("^", "")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _find_column(
    columns: Sequence[object],
    aliases: Sequence[str],
) -> object | None:
    keyed = {_column_key(column): column for column in columns}
    for alias in aliases:
        if _column_key(alias) in keyed:
            return keyed[_column_key(alias)]
    return None


def _positive_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    raw = numeric.to_numpy(dtype=float)
    return numeric.where(np.isfinite(raw) & (raw > 0.0))


def _canonical_long_context(frame: pd.DataFrame) -> pd.DataFrame:
    data = _date_index(frame, allow_duplicates=True)
    symbol_column = _find_column(data.columns, ("symbol", "ticker"))
    if symbol_column is None:
        raise ValueError("Long context data must contain a symbol column")

    close_column = _find_column(data.columns, ("close", "price"))
    adjusted_column = _find_column(
        data.columns,
        ("adj_close", "adjusted_close", "adjclose", "adjustedclose"),
    )
    if close_column is None and adjusted_column is None:
        raise ValueError("Long context data must contain close or adjusted close")

    symbols = (
        data[symbol_column]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "", regex=False)
    )
    canonical_symbol = symbols.map(
        {
            "IWM": "IWM",
            "^IWM": "IWM",
            "VIX": "VIX",
            "^VIX": "VIX",
            "TNX": "TNX",
            "^TNX": "TNX",
        }
    )
    relevant = canonical_symbol.notna()
    data = data.loc[relevant].copy()
    canonical_symbol = canonical_symbol.loc[relevant]
    if data.empty:
        return pd.DataFrame(columns=CONTEXT_VALUE_COLUMNS, dtype=float)

    pairs = pd.MultiIndex.from_arrays([data.index, canonical_symbol])
    if pairs.has_duplicates:
        raise ValueError("Long context data contains duplicate date/symbol rows")

    close = (
        _positive_numeric(data[close_column])
        if close_column is not None
        else pd.Series(np.nan, index=data.index, dtype=float)
    )
    adjusted = (
        _positive_numeric(data[adjusted_column])
        if adjusted_column is not None
        else pd.Series(np.nan, index=data.index, dtype=float)
    )
    # Adjusted close is required for the tradable IWM ETF when provided.  VIX
    # and TNX are index/yield levels, so their ordinary close is preferred.
    values = pd.Series(np.nan, index=data.index, dtype=float)
    iwm_rows = canonical_symbol.eq("IWM").to_numpy()
    level_rows = ~iwm_rows
    values.iloc[np.flatnonzero(iwm_rows)] = (
        adjusted.where(adjusted.notna(), close).to_numpy()[iwm_rows]
    )
    values.iloc[np.flatnonzero(level_rows)] = (
        close.where(close.notna(), adjusted).to_numpy()[level_rows]
    )

    tidy = pd.DataFrame(
        {
            "date": data.index,
            "symbol": canonical_symbol.to_numpy(),
            "value": values.to_numpy(),
        }
    )
    wide = tidy.pivot(index="date", columns="symbol", values="value")
    result = pd.DataFrame(index=pd.DatetimeIndex(wide.index))
    result["iwm_adj_close"] = wide.get("IWM", np.nan)
    result["vix_close"] = wide.get("VIX", np.nan)
    result["tnx_close"] = wide.get("TNX", np.nan)
    return result.loc[:, list(CONTEXT_VALUE_COLUMNS)].astype(float)


def _flatten_multiindex_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    flattened = frame.copy()
    flattened.columns = tuple(
        "_".join(str(part) for part in column if str(part) not in ("", "None"))
        for column in flattened.columns
    )
    return flattened


def _canonical_wide_context(frame: pd.DataFrame) -> pd.DataFrame:
    data = _date_index(_flatten_multiindex_columns(frame), allow_duplicates=False)
    aliases = {
        "iwm_adj_close": (
            "iwm_adj_close",
            "iwm_adjusted_close",
            "adj_close_iwm",
            "adjusted_close_iwm",
            "iwm_close",
            "close_iwm",
            "IWM",
        ),
        "vix_close": (
            "vix_close",
            "close_vix",
            "vix_adj_close",
            "adj_close_vix",
            "^VIX",
            "VIX",
        ),
        "tnx_close": (
            "tnx_close",
            "close_tnx",
            "tnx_adj_close",
            "adj_close_tnx",
            "^TNX",
            "TNX",
        ),
    }
    result = pd.DataFrame(index=data.index)
    for output, candidates in aliases.items():
        source = _find_column(data.columns, candidates)
        result[output] = (
            _positive_numeric(data[source])
            if source is not None
            else pd.Series(np.nan, index=data.index, dtype=float)
        )
    return result.loc[:, list(CONTEXT_VALUE_COLUMNS)].astype(float)


def canonical_market_sentiment_context(
    context_frame: pd.DataFrame,
    decision_dates: Sequence[DateLike] | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return exact-date IWM/VIX/TNX values on the AAPL session index.

    ``context_frame`` may be long (``date, symbol, close/adj_close``) or wide.
    Context-only dates are discarded and missing AAPL-session observations are
    retained as NaN.  No as-of merge, forward fill, interpolation, or zero
    imputation is performed.
    """

    normalized = [
        _normalize_timestamp(value, field_name="decision_date")
        for value in decision_dates
    ]
    index = pd.DatetimeIndex(normalized)
    if index.has_duplicates:
        raise ValueError("decision_dates contains duplicates")

    flattened = _flatten_multiindex_columns(context_frame.copy())
    is_long = _find_column(flattened.columns, ("symbol", "ticker")) is not None
    context = (
        _canonical_long_context(flattened)
        if is_long
        else _canonical_wide_context(flattened)
    )
    result = context.reindex(index)
    result.index.name = None
    return result.loc[:, list(CONTEXT_VALUE_COLUMNS)].astype(float)


def _log_return(prices: pd.Series, sessions: int) -> pd.Series:
    return np.log(prices / prices.shift(sessions))


def _realized_volatility(one_session_log_return: pd.Series, sessions: int) -> pd.Series:
    return (
        one_session_log_return.rolling(sessions, min_periods=sessions)
        .std(ddof=1)
        .mul(math.sqrt(252.0))
    )


def build_market_sentiment_features(
    price_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build the exact 15 completed-close market-sentiment features.

    VIX's 252-session z-score standardizes log(VIX) over the inclusive trailing
    window with sample standard deviation (``ddof=1``).  TNX deltas are raw,
    absolute yield-index differences rather than percentage returns.
    """

    prices = canonical_downside_price_frame(price_frame)
    context = canonical_market_sentiment_context(context_frame, prices.index)
    spy = prices["spy_adj_close"]
    qqq = prices["qqq_adj_close"]
    iwm = context["iwm_adj_close"]
    vix = context["vix_close"]
    tnx = context["tnx_close"]

    qqq_lr_20 = _log_return(qqq, 20)
    iwm_lr_1 = _log_return(iwm, 1)
    iwm_lr_20 = _log_return(iwm, 20)
    vix_log = np.log(vix)
    vix_mean = vix_log.rolling(252, min_periods=252).mean()
    vix_std = vix_log.rolling(252, min_periods=252).std(ddof=1)

    features = pd.DataFrame(
        {
            "spy_lr_5": _log_return(spy, 5),
            "spy_drawdown_60": spy / spy.rolling(60, min_periods=60).max() - 1.0,
            "qqq_drawdown_60": qqq / qqq.rolling(60, min_periods=60).max() - 1.0,
            "iwm_lr_5": _log_return(iwm, 5),
            "iwm_lr_20": iwm_lr_20,
            "iwm_lr_60": _log_return(iwm, 60),
            "iwm_rv_20": _realized_volatility(iwm_lr_1, 20),
            "qqq_minus_iwm_lr_20": qqq_lr_20 - iwm_lr_20,
            "vix_log_level": vix_log,
            "vix_lr_5": _log_return(vix, 5),
            "vix_lr_20": _log_return(vix, 20),
            "vix_z_252": (vix_log - vix_mean) / vix_std,
            "tnx_level": tnx,
            "tnx_delta_5": tnx - tnx.shift(5),
            "tnx_delta_20": tnx - tnx.shift(20),
        },
        index=prices.index,
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.loc[:, list(MARKET_SENTIMENT_FEATURE_COLUMNS)].astype(float)


def _cash_round_trip_log_factor(slippage_bps: float) -> float:
    rate = float(slippage_bps) / 10_000.0
    if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
        raise ValueError("slippage_bps must be finite and between 0 and 10,000")
    return math.log((1.0 - rate) / (1.0 + rate))


def build_five_session_direct_edge_labels(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Build the direct five-session CASH edge target at 10 bps per leg."""

    prices = canonical_downside_price_frame(price_frame)
    opens = prices["aapl_adj_open"]
    dates = pd.Series(prices.index, index=prices.index, dtype="datetime64[ns]")
    entry_date = dates.shift(-LABEL_ENTRY_OFFSET)
    maturity_date = dates.shift(-LABEL_MATURITY_OFFSET)
    entry_open = opens.shift(-LABEL_ENTRY_OFFSET)
    exit_open = opens.shift(-LABEL_MATURITY_OFFSET)

    future_opens_valid = pd.concat(
        [
            opens.shift(-offset).notna().rename(f"open_plus_{offset}")
            for offset in range(LABEL_ENTRY_OFFSET, LABEL_MATURITY_OFFSET + 1)
        ],
        axis=1,
    ).all(axis=1)
    valid = future_opens_valid & entry_date.notna() & maturity_date.notna()

    forward_log_return = pd.Series(np.nan, index=prices.index, dtype=float)
    forward_log_return.loc[valid] = np.log(
        exit_open.loc[valid] / entry_open.loc[valid]
    )
    edge_5 = _cash_round_trip_log_factor(5.0) - forward_log_return
    edge_10 = _cash_round_trip_log_factor(10.0) - forward_log_return
    binary = pd.Series(pd.NA, index=prices.index, dtype="Int8")
    binary.loc[valid] = (edge_10.loc[valid] > 0.0).astype(np.int8)

    labels = pd.DataFrame(
        {
            "label_entry_date": entry_date.where(valid),
            "label_maturity_date": maturity_date.where(valid),
            "label_entry_adjusted_open": entry_open.where(valid),
            "label_exit_adjusted_open": exit_open.where(valid),
            "aapl_forward_log_return_5": forward_log_return,
            "cash_active_log_edge_5bps": edge_5.where(valid),
            "cash_active_log_edge_10bps": edge_10.where(valid),
            "cash_beats_long_10bps": binary,
        },
        index=prices.index,
    )
    return labels.loc[:, list(DIRECT_EDGE_LABEL_COLUMNS)]


def build_direct_edge_feature_label_frame(
    price_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Combine direct-edge labels with price-only and sentiment feature sets."""

    prices = canonical_downside_price_frame(price_frame)
    price_features = build_downside_price_features(prices)
    sentiment = build_market_sentiment_features(prices, context_frame)
    labels = build_five_session_direct_edge_labels(prices)
    context = canonical_market_sentiment_context(context_frame, prices.index)
    result = pd.concat([price_features, sentiment, labels], axis=1)

    price_ready = np.isfinite(
        result.loc[:, list(PRICE_FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all(axis=1)
    sentiment_ready = np.isfinite(
        result.loc[:, list(MARKET_SENTIMENT_FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all(axis=1)
    context_current_ready = np.isfinite(
        context.loc[:, list(CONTEXT_VALUE_COLUMNS)].to_numpy(dtype=float)
    ).all(axis=1)
    result["price_features_ready"] = price_ready
    result["sentiment_features_ready"] = sentiment_ready
    result["sentiment_price_only_fallback"] = ~sentiment_ready
    result["sentiment_status"] = np.where(
        sentiment_ready,
        "ready",
        np.where(context_current_ready, "insufficient_history", "missing_context"),
    )
    result["label_available"] = (
        result["label_maturity_date"].notna()
        & result["cash_active_log_edge_10bps"].notna()
        & result["cash_beats_long_10bps"].notna()
    )
    return result


def _strict_before_training_mask(
    feature_label_frame: pd.DataFrame,
    *,
    as_of_date: DateLike,
    require_sentiment: bool,
) -> pd.Series:
    feature_columns = list(PRICE_FEATURE_COLUMNS)
    required = {
        *feature_columns,
        "label_maturity_date",
        "cash_active_log_edge_10bps",
        "cash_beats_long_10bps",
    }
    if require_sentiment:
        feature_columns.extend(MARKET_SENTIMENT_FEATURE_COLUMNS)
        required.update(MARKET_SENTIMENT_FEATURE_COLUMNS)
        required.add("sentiment_price_only_fallback")
    missing = sorted(required.difference(feature_label_frame.columns))
    if missing:
        raise ValueError(f"feature_label_frame is missing required columns: {missing}")

    try:
        decision_index = pd.DatetimeIndex(
            pd.to_datetime(feature_label_frame.index, errors="raise")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "feature_label_frame index must contain valid decision dates"
        ) from exc
    if decision_index.tz is not None:
        decision_index = decision_index.tz_localize(None)
    decision_index = decision_index.normalize()
    if decision_index.has_duplicates:
        raise ValueError("feature_label_frame contains duplicate decision dates")

    boundary = _normalize_timestamp(as_of_date, field_name="as_of_date")
    maturity = pd.to_datetime(
        feature_label_frame["label_maturity_date"], errors="coerce"
    )
    features_ready = np.isfinite(
        feature_label_frame.loc[:, feature_columns].to_numpy(dtype=float)
    ).all(axis=1)
    eligible = (
        (decision_index < boundary)
        & maturity.notna().to_numpy()
        & (maturity.to_numpy(dtype="datetime64[ns]") < boundary.to_datetime64())
        & feature_label_frame["cash_active_log_edge_10bps"].notna().to_numpy()
        & feature_label_frame["cash_beats_long_10bps"].notna().to_numpy()
        & features_ready
    )
    if require_sentiment:
        eligible &= ~feature_label_frame["sentiment_price_only_fallback"].astype(
            bool
        ).to_numpy()
    return pd.Series(
        eligible,
        index=feature_label_frame.index,
        dtype=bool,
        name="chronologically_eligible",
    )


def price_only_chronological_training_mask(
    feature_label_frame: pd.DataFrame,
    *,
    as_of_date: DateLike,
) -> pd.Series:
    """Select price-only cases whose outcomes matured strictly before a fold."""

    return _strict_before_training_mask(
        feature_label_frame,
        as_of_date=as_of_date,
        require_sentiment=False,
    )


def sentiment_chronological_training_mask(
    feature_label_frame: pd.DataFrame,
    *,
    as_of_date: DateLike,
) -> pd.Series:
    """Select sentiment cases whose outcomes matured strictly before a fold."""

    return _strict_before_training_mask(
        feature_label_frame,
        as_of_date=as_of_date,
        require_sentiment=True,
    )


__all__ = [
    "CONTEXT_VALUE_COLUMNS",
    "DIRECT_EDGE_LABEL_COLUMNS",
    "FIVE_SESSION_HORIZON",
    "LABEL_ENTRY_OFFSET",
    "LABEL_MATURITY_OFFSET",
    "MARKET_SENTIMENT_FEATURE_COLUMNS",
    "PRICE_FEATURE_COLUMNS",
    "build_direct_edge_feature_label_frame",
    "build_five_session_direct_edge_labels",
    "build_market_sentiment_features",
    "canonical_market_sentiment_context",
    "price_only_chronological_training_mask",
    "sentiment_chronological_training_mask",
]
