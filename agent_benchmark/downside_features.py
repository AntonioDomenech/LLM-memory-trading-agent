"""Point-in-time inputs for the frozen five-session AAPL downside model.

The module is deliberately limited to deterministic feature and label
construction.  It does not download data, fit a scaler/model, inspect an
evaluation period, or update state.  A row is a decision made after the
completed market close at session ``t``.  Its hypothetical trade fills at the
adjusted AAPL open at ``t+1`` and its five-session outcome becomes knowable at
the adjusted open at ``t+6``.

The optional CFTC fields preserve the same conservative point-in-time contract
as :mod:`agent_benchmark.cftc_cot_policy`: a complete three-market release is
usable only on/after its declared availability date, the current release is
excluded from its own 52-release reference window, and observations older than
14 calendar days cause an explicit price-only fallback.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .cftc_cot_policy import COTWeeklyRecord, REQUIRED_MARKETS


FIVE_SESSION_HORIZON = 5
LABEL_ENTRY_OFFSET = 1
LABEL_MATURITY_OFFSET = 6
CRASH_SIMPLE_RETURN_THRESHOLD = -0.04
CFTC_LOOKBACK_RELEASES = 52
CFTC_MAX_RELEASE_AGE_DAYS = 14

REQUIRED_PRICE_COLUMNS: tuple[str, ...] = (
    "aapl_open",
    "aapl_close",
    "aapl_adj_close",
    "spy_adj_close",
    "qqq_adj_close",
)

# This is the complete price-only model vector.  Keep its order stable so the
# fitted model and its immutable manifest can bind to one unambiguous schema.
PRICE_FEATURE_COLUMNS: tuple[str, ...] = (
    "aapl_lr_1",
    "aapl_lr_5",
    "aapl_lr_20",
    "aapl_lr_60",
    "aapl_lr_252",
    "aapl_intraday_lr",
    "aapl_gap_lr",
    "aapl_rv_20",
    "aapl_rv_60",
    "aapl_downside_rv_20",
    "aapl_drawdown_60",
    "aapl_drawdown_252",
    "aapl_sma_distance_20",
    "aapl_sma_distance_200",
    "qqq_lr_5",
    "qqq_lr_20",
    "qqq_lr_60",
    "spy_lr_20",
    "spy_lr_60",
    "aapl_minus_qqq_lr_5",
    "aapl_minus_qqq_lr_20",
    "qqq_minus_spy_lr_20",
    "qqq_rv_20",
    "spy_rv_20",
)

CFTC_FEATURE_COLUMNS: tuple[str, ...] = (
    "cot_nasdaq_net_share_z52",
    "cot_nasdaq_minus_sp500_net_share_z52",
    "cot_vix_net_share_z52",
)

LABEL_COLUMNS: tuple[str, ...] = (
    "label_entry_date",
    "label_maturity_date",
    "label_entry_adjusted_open",
    "label_exit_adjusted_open",
    "aapl_forward_simple_return_5",
    "aapl_forward_log_return_5",
    "crash_label_5session",
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


def canonical_downside_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a date-indexed price frame without dropping incomplete sessions.

    Invalid or missing individual prices become ``NaN``.  Preserving the row is
    important: silently dropping it would change what ``t+6`` means and could
    turn a data-quality gap into an apparently valid label.
    """

    data = frame.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="raise")
        data = data.set_index("date")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(data.index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Market data index must contain valid dates") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    data.index = index
    data = data.sort_index()
    if data.index.has_duplicates:
        raise ValueError("Market data contains duplicate decision dates")

    missing = [name for name in REQUIRED_PRICE_COLUMNS if name not in data.columns]
    if missing:
        raise ValueError(f"Market data is missing required columns: {missing}")

    cleaned = pd.DataFrame(index=data.index)
    for name in REQUIRED_PRICE_COLUMNS:
        values = pd.to_numeric(data[name], errors="coerce").astype(float)
        valid = np.isfinite(values.to_numpy(dtype=float)) & (values.to_numpy(dtype=float) > 0)
        cleaned[name] = values.where(valid)

    cleaned["aapl_adj_open"] = (
        cleaned["aapl_open"]
        * cleaned["aapl_adj_close"]
        / cleaned["aapl_close"]
    )
    adjusted_open = cleaned["aapl_adj_open"].to_numpy(dtype=float)
    cleaned["aapl_adj_open"] = cleaned["aapl_adj_open"].where(
        np.isfinite(adjusted_open) & (adjusted_open > 0)
    )
    return cleaned[
        [
            "aapl_open",
            "aapl_close",
            "aapl_adj_close",
            "aapl_adj_open",
            "spy_adj_close",
            "qqq_adj_close",
        ]
    ]


def _log_return(prices: pd.Series, sessions: int) -> pd.Series:
    return np.log(prices / prices.shift(sessions))


def _realized_volatility(one_session_log_return: pd.Series, sessions: int) -> pd.Series:
    return (
        one_session_log_return.rolling(sessions, min_periods=sessions)
        .std(ddof=1)
        .mul(math.sqrt(252.0))
    )


def build_downside_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the exact 24 scale-free price features known at close ``t``."""

    data = canonical_downside_price_frame(frame)
    aapl = data["aapl_adj_close"]
    qqq = data["qqq_adj_close"]
    spy = data["spy_adj_close"]

    aapl_lr_1 = _log_return(aapl, 1)
    aapl_lr_5 = _log_return(aapl, 5)
    aapl_lr_20 = _log_return(aapl, 20)
    qqq_lr_5 = _log_return(qqq, 5)
    qqq_lr_20 = _log_return(qqq, 20)
    spy_lr_20 = _log_return(spy, 20)
    qqq_lr_1 = _log_return(qqq, 1)
    spy_lr_1 = _log_return(spy, 1)

    downside_square = aapl_lr_1.clip(upper=0.0).pow(2)
    features = pd.DataFrame(
        {
            "aapl_lr_1": aapl_lr_1,
            "aapl_lr_5": aapl_lr_5,
            "aapl_lr_20": aapl_lr_20,
            "aapl_lr_60": _log_return(aapl, 60),
            "aapl_lr_252": _log_return(aapl, 252),
            "aapl_intraday_lr": np.log(data["aapl_close"] / data["aapl_open"]),
            "aapl_gap_lr": np.log(
                data["aapl_adj_open"] / data["aapl_adj_close"].shift(1)
            ),
            "aapl_rv_20": _realized_volatility(aapl_lr_1, 20),
            "aapl_rv_60": _realized_volatility(aapl_lr_1, 60),
            "aapl_downside_rv_20": np.sqrt(
                252.0
                * downside_square.rolling(20, min_periods=20).mean()
            ),
            "aapl_drawdown_60": (
                aapl / aapl.rolling(60, min_periods=60).max() - 1.0
            ),
            "aapl_drawdown_252": (
                aapl / aapl.rolling(252, min_periods=252).max() - 1.0
            ),
            "aapl_sma_distance_20": (
                aapl / aapl.rolling(20, min_periods=20).mean() - 1.0
            ),
            "aapl_sma_distance_200": (
                aapl / aapl.rolling(200, min_periods=200).mean() - 1.0
            ),
            "qqq_lr_5": qqq_lr_5,
            "qqq_lr_20": qqq_lr_20,
            "qqq_lr_60": _log_return(qqq, 60),
            "spy_lr_20": spy_lr_20,
            "spy_lr_60": _log_return(spy, 60),
            "aapl_minus_qqq_lr_5": aapl_lr_5 - qqq_lr_5,
            "aapl_minus_qqq_lr_20": aapl_lr_20 - qqq_lr_20,
            "qqq_minus_spy_lr_20": qqq_lr_20 - spy_lr_20,
            "qqq_rv_20": _realized_volatility(qqq_lr_1, 20),
            "spy_rv_20": _realized_volatility(spy_lr_1, 20),
        },
        index=data.index,
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.loc[:, list(PRICE_FEATURE_COLUMNS)].astype(float)


def _cash_round_trip_log_factor(slippage_bps: float) -> float:
    rate = float(slippage_bps) / 10_000.0
    if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
        raise ValueError("slippage_bps must be finite and between 0 and 10,000")
    # Sell at mid*(1-rate), later buy at mid*(1+rate).
    return math.log1p(-rate) - math.log1p(rate)


def _nullable_binary(values: pd.Series, valid: pd.Series) -> pd.Series:
    output = pd.Series(pd.NA, index=values.index, dtype="Int8")
    output.loc[valid] = values.loc[valid].astype(np.int8)
    return output


def build_five_session_downside_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Build next-open five-session labels without exposing an immature row.

    For decision row ``i`` the entry is adjusted open ``i+1`` and the outcome
    is adjusted open ``i+6``.  The latter is both the exit/re-entry price and
    the first moment the label can enter a chronological training set.
    """

    data = canonical_downside_price_frame(frame)
    opens = data["aapl_adj_open"]
    date_values = pd.Series(data.index, index=data.index, dtype="datetime64[ns]")
    entry_date = date_values.shift(-LABEL_ENTRY_OFFSET)
    maturity_date = date_values.shift(-LABEL_MATURITY_OFFSET)
    entry_open = opens.shift(-LABEL_ENTRY_OFFSET)
    exit_open = opens.shift(-LABEL_MATURITY_OFFSET)

    # Every fill in the five-session interval must exist.  Checking only the
    # two endpoints would silently bridge a corrupt/missing interior session.
    future_open_validity = pd.concat(
        [
            (opens.shift(-offset).notna() & (opens.shift(-offset) > 0)).rename(
                f"open_plus_{offset}"
            )
            for offset in range(LABEL_ENTRY_OFFSET, LABEL_MATURITY_OFFSET + 1)
        ],
        axis=1,
    )
    valid = (
        future_open_validity.all(axis=1)
        & entry_date.notna()
        & maturity_date.notna()
    )

    simple_return = pd.Series(np.nan, index=data.index, dtype=float)
    simple_return.loc[valid] = exit_open.loc[valid] / entry_open.loc[valid] - 1.0
    log_return = pd.Series(np.nan, index=data.index, dtype=float)
    log_return.loc[valid] = np.log(exit_open.loc[valid] / entry_open.loc[valid])

    crash = _nullable_binary(
        simple_return <= CRASH_SIMPLE_RETURN_THRESHOLD,
        valid,
    )
    edge_5 = _cash_round_trip_log_factor(5.0) - log_return
    edge_10 = _cash_round_trip_log_factor(10.0) - log_return
    cash_5 = _nullable_binary(edge_5 > 0.0, valid)
    cash_10 = _nullable_binary(edge_10 > 0.0, valid)

    labels = pd.DataFrame(
        {
            "label_entry_date": entry_date.where(valid),
            "label_maturity_date": maturity_date.where(valid),
            "label_entry_adjusted_open": entry_open.where(valid),
            "label_exit_adjusted_open": exit_open.where(valid),
            "aapl_forward_simple_return_5": simple_return,
            "aapl_forward_log_return_5": log_return,
            "crash_label_5session": crash,
            "cash_active_log_edge_5bps": edge_5.where(valid),
            "cash_active_log_edge_10bps": edge_10.where(valid),
            "cash_beats_long_5bps": cash_5,
            "cash_beats_long_10bps": cash_10,
        },
        index=data.index,
    )
    return labels.loc[:, list(LABEL_COLUMNS)]


@dataclass(frozen=True)
class _CompleteCOTRelease:
    report_date: date
    availability_date: date
    nasdaq_net_share: float
    sp500_net_share: float
    vix_net_share: float

    @property
    def nasdaq_minus_sp500_net_share(self) -> float:
        return self.nasdaq_net_share - self.sp500_net_share


def _normalize_cot_records(
    records: Iterable[COTWeeklyRecord | Mapping[str, Any]],
) -> tuple[COTWeeklyRecord, ...]:
    normalized = tuple(
        item if isinstance(item, COTWeeklyRecord) else COTWeeklyRecord.from_mapping(item)
        for item in records
    )
    seen: set[tuple[date, str]] = set()
    for item in normalized:
        key = (item.report_date, item.market)
        if key in seen:
            raise ValueError(
                "duplicate CFTC input for "
                f"report_date={item.report_date.isoformat()} market={item.market}"
            )
        seen.add(key)
    return normalized


def _complete_cot_releases(
    records: Iterable[COTWeeklyRecord | Mapping[str, Any]],
) -> tuple[_CompleteCOTRelease, ...]:
    grouped: dict[date, dict[str, COTWeeklyRecord]] = {}
    for item in _normalize_cot_records(records):
        grouped.setdefault(item.report_date, {})[item.market] = item

    complete: list[_CompleteCOTRelease] = []
    for report_date, by_market in grouped.items():
        if any(market not in by_market for market in REQUIRED_MARKETS):
            continue
        availability = max(by_market[market].availability_date for market in REQUIRED_MARKETS)
        complete.append(
            _CompleteCOTRelease(
                report_date=report_date,
                availability_date=availability,
                nasdaq_net_share=by_market["NASDAQ"].net_share,
                sp500_net_share=by_market["SP500"].net_share,
                vix_net_share=by_market["VIX"].net_share,
            )
        )
    return tuple(sorted(complete, key=lambda item: (item.availability_date, item.report_date)))


def _z_score(current: float, history: Sequence[float]) -> float | None:
    values = np.asarray(history, dtype=float)
    if len(values) != CFTC_LOOKBACK_RELEASES or not np.isfinite(values).all():
        return None
    standard_deviation = float(np.std(values, ddof=0))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        return None
    result = (float(current) - float(np.mean(values))) / standard_deviation
    return float(result) if math.isfinite(result) else None


def build_cftc_continuous_features(
    decision_dates: Sequence[DateLike] | pd.DatetimeIndex,
    records: Iterable[COTWeeklyRecord | Mapping[str, Any]] = (),
) -> pd.DataFrame:
    """Build three causal CFTC z-scores plus explicit fallback provenance."""

    normalized_dates = [
        _normalize_timestamp(value, field_name="decision_date") for value in decision_dates
    ]
    index = pd.DatetimeIndex(normalized_dates)
    if index.has_duplicates:
        raise ValueError("decision_dates contains duplicates")

    complete = _complete_cot_releases(records)
    availability_dates = [item.availability_date for item in complete]
    rows: list[dict[str, Any]] = []
    for timestamp in index:
        decision_date = timestamp.date()
        available_count = bisect_right(availability_dates, decision_date)
        row: dict[str, Any] = {
            "cot_nasdaq_net_share_z52": np.nan,
            "cot_nasdaq_minus_sp500_net_share_z52": np.nan,
            "cot_vix_net_share_z52": np.nan,
            "cftc_price_only_fallback": True,
            "cftc_status": "no_complete_release",
            "cftc_current_report_date": pd.NaT,
            "cftc_current_availability_date": pd.NaT,
            "cftc_release_age_days": np.nan,
            "cftc_complete_releases_available": available_count,
            "cftc_prior_complete_releases": 0,
        }
        if available_count == 0:
            rows.append(row)
            continue

        current = complete[available_count - 1]
        age = (decision_date - current.availability_date).days
        row.update(
            {
                "cftc_current_report_date": pd.Timestamp(current.report_date),
                "cftc_current_availability_date": pd.Timestamp(
                    current.availability_date
                ),
                "cftc_release_age_days": age,
            }
        )
        if age > CFTC_MAX_RELEASE_AGE_DAYS:
            row["cftc_status"] = "stale_release"
            rows.append(row)
            continue

        # Exclude every release with the current availability timestamp.  This
        # is stricter than merely slicing off the current report and protects
        # against delayed batches released together.
        prior = tuple(
            item
            for item in complete[:available_count]
            if item.availability_date < current.availability_date
        )
        row["cftc_prior_complete_releases"] = len(prior)
        if len(prior) < CFTC_LOOKBACK_RELEASES:
            row["cftc_status"] = "insufficient_history"
            rows.append(row)
            continue

        window = prior[-CFTC_LOOKBACK_RELEASES:]
        z_values = (
            _z_score(
                current.nasdaq_net_share,
                [item.nasdaq_net_share for item in window],
            ),
            _z_score(
                current.nasdaq_minus_sp500_net_share,
                [item.nasdaq_minus_sp500_net_share for item in window],
            ),
            _z_score(
                current.vix_net_share,
                [item.vix_net_share for item in window],
            ),
        )
        if any(value is None for value in z_values):
            row["cftc_status"] = "zero_variance"
            rows.append(row)
            continue

        row.update(dict(zip(CFTC_FEATURE_COLUMNS, z_values, strict=True)))
        row["cftc_price_only_fallback"] = False
        row["cftc_status"] = "ready"
        rows.append(row)

    result = pd.DataFrame(rows, index=index)
    result.index.name = None
    for name in CFTC_FEATURE_COLUMNS:
        result[name] = pd.to_numeric(result[name], errors="coerce").astype(float)
    result["cftc_price_only_fallback"] = result["cftc_price_only_fallback"].astype(bool)
    return result


def build_downside_feature_label_frame(
    price_frame: pd.DataFrame,
    *,
    cot_records: Iterable[COTWeeklyRecord | Mapping[str, Any]] = (),
) -> pd.DataFrame:
    """Combine features, labels, readiness, and point-in-time CFTC metadata."""

    data = canonical_downside_price_frame(price_frame)
    features = build_downside_price_features(data)
    labels = build_five_session_downside_labels(data)
    cftc = build_cftc_continuous_features(data.index, cot_records)
    result = pd.concat([features, labels, cftc], axis=1)
    result["price_features_ready"] = np.isfinite(
        result.loc[:, list(PRICE_FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all(axis=1)
    result["label_available"] = (
        result["label_maturity_date"].notna()
        & result["crash_label_5session"].notna()
    )
    return result


def chronological_training_mask(
    feature_label_frame: pd.DataFrame,
    *,
    as_of_date: DateLike,
    require_cftc: bool = False,
    strictly_before_as_of: bool = False,
) -> pd.Series:
    """Select only labels that were mature by the specified decision close.

    Since the label for row ``i`` matures at open ``i+6``, this rule naturally
    purges the five decision rows immediately preceding a chronological fold
    boundary.  Set ``strictly_before_as_of`` for a model frozen before a whole
    validation block: labels maturing at the validation block's first open are
    then excluded too.  Rows after ``as_of_date`` are always excluded even if a
    malformed caller supplies an impossible maturity timestamp.
    """

    required = {
        *PRICE_FEATURE_COLUMNS,
        "label_maturity_date",
        "crash_label_5session",
    }
    if require_cftc:
        required.update(CFTC_FEATURE_COLUMNS)
        required.add("cftc_price_only_fallback")
    missing = sorted(required.difference(feature_label_frame.columns))
    if missing:
        raise ValueError(f"feature_label_frame is missing required columns: {missing}")

    try:
        decision_index = pd.DatetimeIndex(
            pd.to_datetime(feature_label_frame.index, errors="raise")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("feature_label_frame index must contain valid decision dates") from exc
    if decision_index.tz is not None:
        decision_index = decision_index.tz_localize(None)
    decision_index = decision_index.normalize()
    if decision_index.has_duplicates:
        raise ValueError("feature_label_frame contains duplicate decision dates")

    as_of = _normalize_timestamp(as_of_date, field_name="as_of_date")
    maturity = pd.to_datetime(
        feature_label_frame["label_maturity_date"], errors="coerce"
    )
    price_ready = np.isfinite(
        feature_label_frame.loc[:, list(PRICE_FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all(axis=1)
    if not isinstance(strictly_before_as_of, bool):
        raise ValueError("strictly_before_as_of must be boolean")
    if strictly_before_as_of:
        decision_is_eligible = decision_index < as_of
        maturity_is_eligible = (
            maturity.to_numpy(dtype="datetime64[ns]") < as_of.to_datetime64()
        )
    else:
        decision_is_eligible = decision_index <= as_of
        maturity_is_eligible = (
            maturity.to_numpy(dtype="datetime64[ns]") <= as_of.to_datetime64()
        )
    mask = pd.Series(
        decision_is_eligible
        & maturity.notna().to_numpy()
        & maturity_is_eligible
        & feature_label_frame["crash_label_5session"].notna().to_numpy()
        & price_ready,
        index=feature_label_frame.index,
        dtype=bool,
        name="chronologically_eligible",
    )
    if require_cftc:
        cftc_ready = (
            ~feature_label_frame["cftc_price_only_fallback"].astype(bool)
            & np.isfinite(
                feature_label_frame.loc[:, list(CFTC_FEATURE_COLUMNS)].to_numpy(
                    dtype=float
                )
            ).all(axis=1)
        )
        mask &= cftc_ready
    return mask


def chronologically_eligible_cases(
    feature_label_frame: pd.DataFrame,
    *,
    as_of_date: DateLike,
    require_cftc: bool = False,
    strictly_before_as_of: bool = False,
) -> pd.DataFrame:
    """Return a defensive copy of the causally eligible training rows."""

    mask = chronological_training_mask(
        feature_label_frame,
        as_of_date=as_of_date,
        require_cftc=require_cftc,
        strictly_before_as_of=strictly_before_as_of,
    )
    return feature_label_frame.loc[mask].copy()


__all__ = [
    "CFTC_FEATURE_COLUMNS",
    "CFTC_LOOKBACK_RELEASES",
    "CFTC_MAX_RELEASE_AGE_DAYS",
    "CRASH_SIMPLE_RETURN_THRESHOLD",
    "FIVE_SESSION_HORIZON",
    "LABEL_COLUMNS",
    "LABEL_ENTRY_OFFSET",
    "LABEL_MATURITY_OFFSET",
    "PRICE_FEATURE_COLUMNS",
    "REQUIRED_PRICE_COLUMNS",
    "build_cftc_continuous_features",
    "build_downside_feature_label_frame",
    "build_downside_price_features",
    "build_five_session_downside_labels",
    "canonical_downside_price_frame",
    "chronological_training_mask",
    "chronologically_eligible_cases",
]
