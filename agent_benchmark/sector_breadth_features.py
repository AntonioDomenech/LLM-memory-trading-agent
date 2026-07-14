"""Point-in-time sector-breadth features for five-session AAPL decisions.

Every feature row is known only after the completed 4:15 p.m. ET VIX daily
value for session ``t``.  The direct-edge label, reused from
:mod:`direct_edge_features`, enters at the AAPL adjusted open at ``t+1`` and
matures at the adjusted open at ``t+6``.  Context is aligned by exact session
date and is never forward-filled.

The Parquet loader is deliberately narrow: callers must provide explicit
inclusive bounds, DuckDB projects only four allowlisted source columns, and
only the original nine Select Sector SPDRs, IWM, and ^VIX can reach Python.
It performs no download, model fitting, or evaluation-period selection.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from .direct_edge_features import (
    DIRECT_EDGE_LABEL_COLUMNS,
    build_five_session_direct_edge_labels,
)
from .downside_features import (
    PRICE_FEATURE_COLUMNS,
    build_downside_price_features,
    canonical_downside_price_frame,
)


DateLike = str | date | datetime | pd.Timestamp

_WINDOWS: Final[tuple[int, ...]] = (1, 5, 20, 60)
_SECTOR_SYMBOLS: Final[tuple[str, ...]] = (
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
)
_DEFENSIVE_SYMBOLS: Final[tuple[str, ...]] = ("XLP", "XLU", "XLV")
_CYCLICAL_SYMBOLS: Final[tuple[str, ...]] = (
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLY",
)
_CONTEXT_SYMBOLS: Final[tuple[str, ...]] = (*_SECTOR_SYMBOLS, "IWM", "^VIX")
_CONTEXT_COLUMN_BY_SYMBOL: Final[dict[str, str]] = {
    **{symbol: f"{symbol.lower()}_adj_close" for symbol in _SECTOR_SYMBOLS},
    "IWM": "iwm_adj_close",
    "^VIX": "vix_close",
}
_CONTEXT_VALUE_COLUMNS: Final[tuple[str, ...]] = tuple(
    _CONTEXT_COLUMN_BY_SYMBOL[symbol] for symbol in _CONTEXT_SYMBOLS
)
_FULL_PANEL_START: Final[pd.Timestamp] = pd.Timestamp("2000-05-26")
_SECTOR_VALUE_COLUMNS: Final[tuple[str, ...]] = tuple(
    _CONTEXT_COLUMN_BY_SYMBOL[symbol] for symbol in _SECTOR_SYMBOLS
)


def _window_feature_names(window: int) -> tuple[str, ...]:
    names = (
        f"sector_participation_{window}",
        f"sector_median_lr_{window}",
        f"sector_dispersion_lr_{window}",
        f"defensive_minus_cyclical_participation_{window}",
        f"defensive_minus_cyclical_median_lr_{window}",
        f"iwm_minus_spy_lr_{window}",
        f"aapl_vs_xlk_residual_lr_{window}",
    )
    # The canonical downside controls already contain the exact same 5- and
    # 20-session AAPL-minus-QQQ log returns.  Keep only the nonduplicated 1/60
    # additions in this incremental context vector.
    if window in {1, 60}:
        return (*names, f"aapl_vs_qqq_residual_lr_{window}")
    return names


# This is the complete new context vector.  Keep both membership and order
# fixed: downstream walk-forward manifests bind directly to this tuple.
SECTOR_BREADTH_FEATURE_COLUMNS: tuple[str, ...] = (
    *(
        name
        for window in _WINDOWS
        for name in _window_feature_names(window)
    ),
    "vix_log_level",
    "vix_lr_1",
    "vix_lr_5",
    "vix_lr_20",
    "vix_lr_60",
    "vix_z_252",
)


_SECTOR_CONTEXT_PARQUET_QUERY: Final[str] = """
SELECT
    CAST(date AS DATE) AS date,
    upper(trim(CAST(symbol AS VARCHAR))) AS symbol,
    CAST(close AS DOUBLE) AS close,
    CAST(adj_close AS DOUBLE) AS adj_close
FROM read_parquet(?)
WHERE CAST(date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
  AND market_open IS TRUE
  AND listed IS TRUE
  AND ohlcv_available IS TRUE
  AND upper(trim(CAST(symbol AS VARCHAR))) IN (
      'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY',
      'IWM', '^VIX'
  )
ORDER BY date, symbol
""".strip()


def _normalize_date(value: DateLike, *, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid date") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} cannot be missing")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _positive_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    array = numeric.to_numpy(dtype=float)
    return numeric.where(np.isfinite(array) & (array > 0.0))


def _canonical_context_index(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="raise")
        data = data.set_index("date")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(data.index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Sector context index must contain valid dates") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    data.index = index.normalize()
    data = data.sort_index(kind="stable")
    if data.index.has_duplicates:
        raise ValueError("Sector context contains duplicate dates")
    return data


def _canonical_wide_context(frame: pd.DataFrame) -> pd.DataFrame:
    data = _canonical_context_index(frame)
    missing = sorted(set(_CONTEXT_VALUE_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Sector context is missing required columns: {missing}")
    cleaned = pd.DataFrame(index=data.index)
    for name in _CONTEXT_VALUE_COLUMNS:
        cleaned[name] = _positive_numeric(data[name])
    return cleaned.loc[:, list(_CONTEXT_VALUE_COLUMNS)]


def load_sector_context_parquet(
    path: str | Path,
    *,
    start: DateLike,
    end: DateLike,
) -> pd.DataFrame:
    """Load one complete, bounded close panel through a fixed DuckDB query.

    The result has one row per returned session and exactly eleven wide value
    columns.  Each returned session must contain each allowlisted symbol once.
    Tradable ETFs require finite positive adjusted closes; ^VIX requires a
    finite positive ordinary close.  Rows outside ``[start, end]`` cannot pass.
    """

    lower = _normalize_date(start, field_name="start")
    upper = _normalize_date(end, field_name="end")
    if lower > upper:
        raise ValueError("start must be on or before end")
    if lower < _FULL_PANEL_START:
        raise ValueError(
            "start cannot precede the fixed 2000-05-26 complete-panel boundary"
        )
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Sector context Parquet does not exist: {source}")

    try:
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            frame = connection.execute(
                _SECTOR_CONTEXT_PARQUET_QUERY,
                [str(source), lower.date().isoformat(), upper.date().isoformat()],
            ).fetchdf()
        finally:
            connection.close()
    except Exception as exc:
        raise ValueError("Could not execute bounded sector context query") from exc

    expected_columns = ["date", "symbol", "close", "adj_close"]
    if list(frame.columns) != expected_columns:
        raise ValueError("Bounded sector query returned unexpected columns")
    if frame.empty:
        raise ValueError("Bounded sector query returned no eligible rows")

    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise")).normalize()
    symbols = frame["symbol"].astype(str)
    if (dates < lower).any() or (dates > upper).any():
        raise ValueError("Bounded sector query returned data outside its inclusive bounds")
    if not symbols.isin(_CONTEXT_SYMBOLS).all():
        raise ValueError("Bounded sector query returned a non-allowlisted symbol")
    pairs = pd.MultiIndex.from_arrays([dates, symbols])
    if pairs.has_duplicates:
        raise ValueError("Sector context contains duplicate date/symbol rows")

    observed = pd.DataFrame({"date": dates, "symbol": symbols})
    expected_symbol_set = set(_CONTEXT_SYMBOLS)
    for session, group in observed.groupby("date", sort=False):
        present = set(group["symbol"])
        if present != expected_symbol_set:
            missing = sorted(expected_symbol_set - present)
            extra = sorted(present - expected_symbol_set)
            raise ValueError(
                f"Sector context session {session.date().isoformat()} has an incomplete "
                f"symbol panel; missing={missing}, extra={extra}"
            )

    values = pd.Series(np.nan, index=frame.index, dtype=float)
    vix_mask = symbols.eq("^VIX").to_numpy()
    values.loc[vix_mask] = pd.to_numeric(
        frame.loc[vix_mask, "close"], errors="coerce"
    ).to_numpy(dtype=float)
    values.loc[~vix_mask] = pd.to_numeric(
        frame.loc[~vix_mask, "adj_close"], errors="coerce"
    ).to_numpy(dtype=float)
    value_array = values.to_numpy(dtype=float)
    if not (np.isfinite(value_array) & (value_array > 0.0)).all():
        raise ValueError("Sector context contains a missing, nonfinite, or nonpositive close")

    tidy = pd.DataFrame(
        {"date": dates, "symbol": symbols.to_numpy(), "value": value_array}
    )
    pivot = tidy.pivot(index="date", columns="symbol", values="value")
    result = pd.DataFrame(index=pd.DatetimeIndex(pivot.index))
    for symbol in _CONTEXT_SYMBOLS:
        result[_CONTEXT_COLUMN_BY_SYMBOL[symbol]] = pivot[symbol]
    result.index.name = None
    return result.loc[:, list(_CONTEXT_VALUE_COLUMNS)].astype(float)


def _log_return(values: pd.Series, sessions: int) -> pd.Series:
    return np.log(values / values.shift(sessions))


def _complete_cross_section_stat(
    returns: pd.DataFrame,
    *,
    statistic: str,
) -> pd.Series:
    complete = returns.notna().all(axis=1)
    if statistic == "participation":
        result = returns.gt(0.0).sum(axis=1).div(float(returns.shape[1]))
    elif statistic == "median":
        result = returns.median(axis=1, skipna=False)
    elif statistic == "dispersion":
        result = returns.std(axis=1, ddof=0, skipna=False)
    else:  # pragma: no cover - all calls are fixed above
        raise AssertionError(f"Unknown cross-section statistic: {statistic}")
    return result.where(complete)


def _build_sector_breadth_features(
    prices: pd.DataFrame,
    context_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed_context = _canonical_context_index(context_frame)
    outside = indexed_context.index[~indexed_context.index.isin(prices.index)]
    if len(outside):
        raise ValueError(
            "Sector context contains a date outside the authorized price index: "
            f"{outside[0].date().isoformat()}"
        )
    # The boundary check intentionally precedes numeric coercion, so even an
    # invalid value on an unauthorized future row is rejected as out of scope
    # rather than inspected or silently discarded.
    context = _canonical_wide_context(indexed_context).reindex(prices.index)
    sectors = context.loc[:, list(_SECTOR_VALUE_COLUMNS)]
    aapl = prices["aapl_adj_close"]
    spy = prices["spy_adj_close"]
    qqq = prices["qqq_adj_close"]
    iwm = context["iwm_adj_close"]
    xlk = context["xlk_adj_close"]

    defensive_columns = [
        _CONTEXT_COLUMN_BY_SYMBOL[symbol] for symbol in _DEFENSIVE_SYMBOLS
    ]
    cyclical_columns = [
        _CONTEXT_COLUMN_BY_SYMBOL[symbol] for symbol in _CYCLICAL_SYMBOLS
    ]
    features: dict[str, pd.Series] = {}
    for window in _WINDOWS:
        sector_returns = np.log(sectors / sectors.shift(window))
        defensive = sector_returns.loc[:, defensive_columns]
        cyclical = sector_returns.loc[:, cyclical_columns]
        features[f"sector_participation_{window}"] = _complete_cross_section_stat(
            sector_returns, statistic="participation"
        )
        features[f"sector_median_lr_{window}"] = _complete_cross_section_stat(
            sector_returns, statistic="median"
        )
        features[f"sector_dispersion_lr_{window}"] = _complete_cross_section_stat(
            sector_returns, statistic="dispersion"
        )
        features[f"defensive_minus_cyclical_participation_{window}"] = (
            _complete_cross_section_stat(defensive, statistic="participation")
            - _complete_cross_section_stat(cyclical, statistic="participation")
        )
        features[f"defensive_minus_cyclical_median_lr_{window}"] = (
            _complete_cross_section_stat(defensive, statistic="median")
            - _complete_cross_section_stat(cyclical, statistic="median")
        )
        features[f"iwm_minus_spy_lr_{window}"] = _log_return(
            iwm, window
        ) - _log_return(spy, window)
        features[f"aapl_vs_xlk_residual_lr_{window}"] = _log_return(
            aapl, window
        ) - _log_return(xlk, window)
        if window in {1, 60}:
            features[f"aapl_vs_qqq_residual_lr_{window}"] = _log_return(
                aapl, window
            ) - _log_return(qqq, window)

    vix_log = np.log(context["vix_close"])
    vix_mean = vix_log.rolling(252, min_periods=252).mean()
    vix_std = vix_log.rolling(252, min_periods=252).std(ddof=1)
    features.update(
        {
            "vix_log_level": vix_log,
            "vix_lr_1": _log_return(context["vix_close"], 1),
            "vix_lr_5": _log_return(context["vix_close"], 5),
            "vix_lr_20": _log_return(context["vix_close"], 20),
            "vix_lr_60": _log_return(context["vix_close"], 60),
            "vix_z_252": (vix_log - vix_mean) / vix_std,
        }
    )
    result = pd.DataFrame(features, index=prices.index)
    result = result.replace([np.inf, -np.inf], np.nan)
    return (
        result.loc[:, list(SECTOR_BREADTH_FEATURE_COLUMNS)].astype(float),
        context,
    )


def build_sector_breadth_feature_label_frame(
    price_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Combine post-4:15-close-t features and t+1-to-t+6 labels."""

    prices = canonical_downside_price_frame(price_frame)
    price_features = build_downside_price_features(prices)
    breadth, context = _build_sector_breadth_features(prices, context_frame)
    labels = build_five_session_direct_edge_labels(prices)
    result = pd.concat([price_features, breadth, labels], axis=1)

    price_ready = np.isfinite(
        result.loc[:, list(PRICE_FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all(axis=1)
    breadth_ready = np.isfinite(
        result.loc[:, list(SECTOR_BREADTH_FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all(axis=1)
    current_context_ready = np.isfinite(
        context.loc[:, list(_CONTEXT_VALUE_COLUMNS)].to_numpy(dtype=float)
    ).all(axis=1)
    result["price_features_ready"] = price_ready
    result["sector_breadth_features_ready"] = breadth_ready
    result["sector_breadth_price_only_fallback"] = ~breadth_ready
    result["sector_breadth_status"] = np.where(
        breadth_ready,
        "ready",
        np.where(current_context_ready, "insufficient_history", "missing_context"),
    )
    result["label_available"] = (
        result["label_maturity_date"].notna()
        & result["cash_active_log_edge_10bps"].notna()
        & result["cash_beats_long_10bps"].notna()
    )
    ordered = [
        *PRICE_FEATURE_COLUMNS,
        *SECTOR_BREADTH_FEATURE_COLUMNS,
        *DIRECT_EDGE_LABEL_COLUMNS,
        "price_features_ready",
        "sector_breadth_features_ready",
        "sector_breadth_price_only_fallback",
        "sector_breadth_status",
        "label_available",
    ]
    return result.loc[:, ordered]


__all__ = [
    "SECTOR_BREADTH_FEATURE_COLUMNS",
    "build_sector_breadth_feature_label_frame",
    "load_sector_context_parquet",
]
