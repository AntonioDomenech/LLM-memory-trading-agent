"""Pure deterministic replay for the causal contextual expert aggregator.

This module deliberately owns no data acquisition, account simulation, news,
LLM, API, or network behavior.  Callers supply an already-authorized canonical
daily frame.  The only project dependencies are the fixed signal arithmetic
and the sequential aggregation model.

The replay boundary is intentionally stronger than a convenient backtest
wrapper:

* fixed signals are built from the original fixed-signal implementation;
* tail-only ``stage_outcome_available`` fields never enter the opportunity
  stream;
* every accepted union opportunity is passed to the model, independently of
  the model action;
* checkpoints bind a bounded audit tail while continuation requires and
  verifies the complete canonical prefix before invoking the fixed builder on
  ``prefix + later``; and
* rolling canonical hashes make full replay and checkpoint continuation
  exactly comparable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .chronological_exhaustion_expert import build_fixed_expert_signals
from .contextual_expert_aggregation import (
    ABLATION_MODES,
    CAUSAL_ONLINE_MODE,
    EXPERT_NAMES,
    FROZEN_CUTOFF_MODE,
    FULL_MODE,
    GLOBAL_ONLY_MODE,
    LEARNING_MODES,
    LIFETIME_ONLY_MODE,
    SCALE_NAMES,
    SLOW_LOOKBACK,
    ContextualExpertAggregator,
    MarketSession,
)


REPLAY_SCHEMA_VERSION = 2
FEATURE_LOOKBACK_SESSIONS = 126

RUNTIME_SEGMENT_FIELDS = (
    "first_session_date",
    "source_offset",
    "runtime",
    "prior_segment_terminal_date",
    "prior_segment_terminal_count",
    "parent_checkpoint_digest",
)

CANONICAL_MARKET_COLUMNS = (
    "aapl_open",
    "aapl_close",
    "aapl_adj_close",
    "aapl_adj_open",
    "spy_adj_close",
    "qqq_adj_close",
)
SIGNAL_INPUT_COLUMNS = (
    "aapl_open",
    "aapl_close",
    "aapl_adj_close",
    "spy_adj_close",
    "qqq_adj_close",
)

# These are the fixed builder's close-time fields that do not depend on the
# physical end of a supplied stage.  In particular, none of the
# ``*_pending_stage_outcome``, ``stage_outcome_available``, or builder target
# columns are admitted here.
FIXED_CAUSAL_SIGNAL_COLUMNS = (
    "aapl_intraday_return",
    "contextual_prior_intraday_percentile",
    "contextual_spy_return_10",
    "contextual_qqq_return_10",
    "contextual_ready",
    "contextual_raw_signal",
    "contextual_virtual_signal",
    "contextual_virtual_signal_blocked",
    "weak_trend_prior_intraday_percentile",
    "weak_trend_spy_return_20",
    "weak_trend_qqq_return_20",
    "weak_trend_aapl_sma_20",
    "weak_trend_ready",
    "weak_trend_raw_signal",
    "weak_trend_virtual_signal",
    "weak_trend_virtual_signal_blocked",
    "unfiltered_union_candidate_signal",
    "unfiltered_union_signal",
    "unfiltered_union_signal_blocked",
)
COMPARATOR_TARGET_COLUMNS = (
    "fixed_always_long_target_exposure",
    "fixed_union_cash_target_exposure",
    "fixed_contextual_only_target_exposure",
    "fixed_weak_trend_only_target_exposure",
)
FIXED_FEATURE_COLUMNS = (
    *CANONICAL_MARKET_COLUMNS,
    *FIXED_CAUSAL_SIGNAL_COLUMNS,
    *COMPARATOR_TARGET_COLUMNS,
)

COOLDOWN_SIGNAL_COLUMNS = (
    "contextual_raw_signal",
    "contextual_virtual_signal",
    "weak_trend_raw_signal",
    "weak_trend_virtual_signal",
    "unfiltered_union_candidate_signal",
    "unfiltered_union_signal",
)
COOLDOWN_PREDECESSOR_KEYS = (
    "contextual_virtual_signal",
    "weak_trend_virtual_signal",
    "unfiltered_union_signal",
)
OPPORTUNITY_SERIES = (
    "contextual_virtual_signal",
    "weak_trend_virtual_signal",
    "unfiltered_union_candidate_signal",
    "unfiltered_union_signal",
)

ONLINE_FULL_ARM = "online_full"
FROZEN_2018_ARM = "frozen_2018"
GLOBAL_ONLY_ARM = "global_only"
LIFETIME_ONLY_ARM = "lifetime_only"
CONFIRMATION_ARM_ORDER = (
    ONLINE_FULL_ARM,
    FROZEN_2018_ARM,
    GLOBAL_ONLY_ARM,
    LIFETIME_ONLY_ARM,
)


def _forecast_columns() -> tuple[str, ...]:
    columns = [
        "aapl_adj_open",
        "contextual_virtual_signal",
        "weak_trend_virtual_signal",
        "unfiltered_union_candidate_signal",
        "canonical_union_opportunity",
        *COMPARATOR_TARGET_COLUMNS,
        "market_state",
        "Q",
        "cash_score",
        "action",
        "learner_target_exposure",
    ]
    columns.extend(f"advice__{name}" for name in EXPERT_NAMES)
    columns.extend(f"aggregate_weight__{name}" for name in EXPERT_NAMES)
    for scale in SCALE_NAMES:
        columns.append(f"scale_active__{scale}")
        columns.extend(f"weight__{scale}__{name}" for name in EXPERT_NAMES)
        columns.append(f"rho__{scale}")
    columns.extend(
        [
            "matured_on_close_count",
            "matured_signal_date",
            "matured_signal_market_state",
            "matured_entry_adjusted_open",
            "matured_exit_adjusted_open",
            "matured_net_cash_log_edge_10bps",
            "matured_reward_range",
            "matured_admitted",
        ]
    )
    columns.extend(f"matured_advice__{name}" for name in EXPERT_NAMES)
    columns.extend(f"matured_reward__{name}" for name in EXPERT_NAMES)
    columns.extend(
        [
            "processed_session_count",
            "matured_lesson_count",
            "admitted_lesson_count",
            "pending_lesson_count",
            "opportunity_count_to_date",
        ]
    )
    return tuple(columns)


FORECAST_COLUMNS = _forecast_columns()
MATURED_EVENT_COLUMNS = (
    "signal_date",
    "maturity_date",
    "signal_market_state",
    "entry_adjusted_open",
    "exit_adjusted_open",
    "net_cash_log_edge_10bps",
    "reward_range",
    "admitted",
    *(f"advice__{name}" for name in EXPERT_NAMES),
    *(f"reward__{name}" for name in EXPERT_NAMES),
)
MATURED_EVENT_STRING_COLUMNS = (
    "signal_date",
    "maturity_date",
    "signal_market_state",
)
MATURED_EVENT_FLOAT_COLUMNS = (
    "entry_adjusted_open",
    "exit_adjusted_open",
    "net_cash_log_edge_10bps",
    "reward_range",
    *(f"reward__{name}" for name in EXPERT_NAMES),
)
MATURED_EVENT_BOOL_COLUMNS = (
    "admitted",
    *(f"advice__{name}" for name in EXPERT_NAMES),
)


def _strict_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != expected:
        raise ValueError(f"{name} has missing or unexpected fields")


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _strict_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _iso_session_date(value: Any, *, name: str) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a calendar date") from exc
    if pd.isna(timestamp) or timestamp.tz is not None or timestamp != timestamp.normalize():
        raise ValueError(f"{name} must be a timezone-naive normalized date")
    return timestamp.date().isoformat()


def _canonical_runtime_payload(
    value: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    _strict_keys(
        value,
        {"learning_mode", "frozen_cutoff", "ablation_mode"},
        name=name,
    )
    model = ContextualExpertAggregator(
        learning_mode=value["learning_mode"],
        frozen_cutoff=value["frozen_cutoff"],
        ablation_mode=value["ablation_mode"],
    )
    return {
        "learning_mode": model.learning_mode,
        "frozen_cutoff": model.frozen_cutoff,
        "ablation_mode": model.ablation_mode,
    }


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest") from exc
    if raw.hex() != value:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _normalize_for_hash(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return {"$missing": True}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return {"$timestamp": _iso_session_date(value, name="timestamp")}
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"$float": "nan"}
        if not math.isfinite(value):
            raise ValueError("canonical hashes reject infinite values")
        return {"$float": value.hex()}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical mapping keys must be strings")
        return {
            key: _normalize_for_hash(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _normalize_for_hash(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _initial_chain(domain: str, columns: Sequence[str]) -> str:
    return _sha256_hex({"domain": domain, "columns": list(columns)})


def _extend_frame_chain(
    prior_digest: str,
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> str:
    digest = bytes.fromhex(_require_sha256(prior_digest, name="prior digest"))
    if tuple(frame.columns) != tuple(columns):
        raise ValueError("frame columns do not match the canonical hash schema")
    for row_index, row in frame.iterrows():
        record = {
            "date": _iso_session_date(row_index, name="frame index"),
            "values": [row[column] for column in columns],
        }
        digest = hashlib.sha256(digest + _canonical_bytes(record)).digest()
    return digest.hex()


def _extend_event_chain(
    prior_digest: str,
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> str:
    digest = bytes.fromhex(_require_sha256(prior_digest, name="prior digest"))
    if tuple(frame.columns) != tuple(columns):
        raise ValueError("event columns do not match the canonical hash schema")
    for _, row in frame.iterrows():
        record = {column: row[column] for column in columns}
        digest = hashlib.sha256(digest + _canonical_bytes(record)).digest()
    return digest.hex()


def _empty_indexed_frame(columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(columns))
    frame.index = pd.DatetimeIndex([], name="date")
    return frame


def _matured_event_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(records), columns=MATURED_EVENT_COLUMNS)
    for column in MATURED_EVENT_STRING_COLUMNS:
        frame[column] = frame[column].astype(object)
    for column in MATURED_EVENT_FLOAT_COLUMNS:
        frame[column] = frame[column].astype("float64")
    for column in MATURED_EVENT_BOOL_COLUMNS:
        frame[column] = frame[column].astype(bool)
    return frame


def canonical_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate without sorting, dropping, filling, or mutating the input."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("market frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("market frame must not be empty")
    if "date" in frame.columns:
        raise ValueError("market frame must use a DatetimeIndex, not a date column")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("market frame must use a DatetimeIndex")
    index = frame.index
    if index.tz is not None:
        raise ValueError("market frame index must be timezone-naive")
    if index.has_duplicates:
        raise ValueError("market frame must not contain duplicate sessions")
    if not index.is_monotonic_increasing:
        raise ValueError("market frame must be strictly chronological")
    if not all(value == value.normalize() for value in index):
        raise ValueError("market frame sessions must be normalized dates")
    observed_columns = tuple(frame.columns)
    if observed_columns not in (SIGNAL_INPUT_COLUMNS, CANONICAL_MARKET_COLUMNS):
        raise ValueError(
            "market frame columns must be exactly the ordered five physical "
            "columns or ordered six canonical columns"
        )

    clean = pd.DataFrame(index=pd.DatetimeIndex(index.copy(), name="date"))
    for column in SIGNAL_INPUT_COLUMNS:
        if not frame[column].map(
            lambda value: isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError(f"{column} must contain real non-boolean values")
        try:
            values = pd.to_numeric(frame[column], errors="raise").astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{column} must be numeric") from exc
        array = values.to_numpy(dtype=float)
        if not np.isfinite(array).all() or (array <= 0.0).any():
            raise ValueError(f"{column} must be finite and strictly positive")
        if (array < np.finfo(float).tiny).any():
            raise ValueError(f"{column} must not contain subnormal values")
        clean[column] = array

    derived = (
        clean["aapl_open"]
        * clean["aapl_adj_close"]
        / clean["aapl_close"]
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(derived).all()
        or (derived <= 0.0).any()
        or (derived < np.finfo(float).tiny).any()
    ):
        raise ValueError("adjusted AAPL open could not be constructed")
    if "aapl_adj_open" in frame.columns:
        if not frame["aapl_adj_open"].map(
            lambda value: isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError(
                "aapl_adj_open must contain real non-boolean values"
            )
        try:
            supplied = pd.to_numeric(
                frame["aapl_adj_open"], errors="raise"
            ).to_numpy(dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("aapl_adj_open must be numeric") from exc
        if (
            not np.isfinite(supplied).all()
            or (supplied <= 0.0).any()
            or not np.array_equal(
                supplied.astype("float64", copy=False).view("uint64"),
                derived.astype("float64", copy=False).view("uint64"),
            )
        ):
            raise ValueError(
                "aapl_adj_open is not bit-exactly equal to open/close adjustment"
            )
    clean["aapl_adj_open"] = derived
    return clean.loc[:, CANONICAL_MARKET_COLUMNS].copy()


def _strict_boolean_series(series: pd.Series, *, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    if not series.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{name} must contain booleans only")
    return series.astype(bool)


def _seeded_canonicalize(raw: pd.Series, predecessor_accepted: bool) -> pd.Series:
    predecessor = _strict_bool(
        predecessor_accepted, name="cooldown predecessor"
    )
    values = _strict_boolean_series(raw, name=str(raw.name)).to_numpy(dtype=bool)
    accepted = np.zeros(len(values), dtype=bool)
    previous = predecessor
    for position, active in enumerate(values):
        current = bool(active and not previous)
        accepted[position] = current
        previous = current
    return pd.Series(accepted, index=raw.index, name=raw.name, dtype=bool)


def _causal_builder_frame(canonical: pd.DataFrame) -> pd.DataFrame:
    built = build_fixed_expert_signals(canonical.loc[:, SIGNAL_INPUT_COLUMNS])
    if not isinstance(built, pd.DataFrame):
        raise TypeError("fixed signal builder must return a DataFrame")
    if not built.index.equals(canonical.index):
        raise ValueError("fixed signal builder changed the canonical session index")
    missing = sorted(set(FIXED_CAUSAL_SIGNAL_COLUMNS).difference(built.columns))
    if missing:
        raise ValueError(f"fixed signal builder is missing columns: {missing}")
    result = built.loc[:, FIXED_CAUSAL_SIGNAL_COLUMNS].copy()
    boolean_columns = (
        "contextual_ready",
        "contextual_raw_signal",
        "contextual_virtual_signal",
        "contextual_virtual_signal_blocked",
        "weak_trend_ready",
        "weak_trend_raw_signal",
        "weak_trend_virtual_signal",
        "weak_trend_virtual_signal_blocked",
        "unfiltered_union_candidate_signal",
        "unfiltered_union_signal",
        "unfiltered_union_signal_blocked",
    )
    for column in boolean_columns:
        result[column] = _strict_boolean_series(result[column], name=column)
    numeric = result.drop(columns=list(boolean_columns)).to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("fixed causal features must not contain infinite values")
    return result


def _apply_comparator_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    opportunity = result["unfiltered_union_signal"].to_numpy(dtype=bool)
    contextual = (
        opportunity
        & result["contextual_virtual_signal"].to_numpy(dtype=bool)
    )
    weak = (
        opportunity
        & result["weak_trend_virtual_signal"].to_numpy(dtype=bool)
    )
    result["fixed_always_long_target_exposure"] = 1.0
    result["fixed_union_cash_target_exposure"] = np.where(
        opportunity, 0.0, 1.0
    )
    result["fixed_contextual_only_target_exposure"] = np.where(
        contextual, 0.0, 1.0
    )
    result["fixed_weak_trend_only_target_exposure"] = np.where(
        weak, 0.0, 1.0
    )
    return result


def _signal_tail_to_frame(
    records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dates: list[pd.Timestamp] = []
    for position, raw in enumerate(records):
        expected = {"date", *COOLDOWN_SIGNAL_COLUMNS}
        _strict_keys(raw, expected, name=f"signal_cooldown_tail[{position}]")
        dates.append(pd.Timestamp(_iso_session_date(raw["date"], name="tail date")))
        rows.append(
            {
                column: _strict_bool(raw[column], name=f"tail[{column}]")
                for column in COOLDOWN_SIGNAL_COLUMNS
            }
        )
    frame = pd.DataFrame(rows, columns=COOLDOWN_SIGNAL_COLUMNS)
    frame.index = pd.DatetimeIndex(dates, name="date")
    return frame


def _market_tail_to_frame(
    records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    if not records:
        return _empty_indexed_frame(CANONICAL_MARKET_COLUMNS)
    rows: list[dict[str, Any]] = []
    dates: list[pd.Timestamp] = []
    for position, raw in enumerate(records):
        expected = {"date", *CANONICAL_MARKET_COLUMNS}
        _strict_keys(raw, expected, name=f"market_feature_tail[{position}]")
        dates.append(pd.Timestamp(_iso_session_date(raw["date"], name="tail date")))
        rows.append({column: raw[column] for column in CANONICAL_MARKET_COLUMNS})
    candidate = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))
    return canonical_market_frame(candidate)


def _records_from_frame(frame: pd.DataFrame) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        record = {"date": _iso_session_date(row_index, name="frame date")}
        for column in frame.columns:
            value = row[column]
            if isinstance(value, np.generic):
                value = value.item()
            record[column] = value
        records.append(record)
    return tuple(records)


def _validate_cooldown_tail(
    tail: pd.DataFrame, predecessor: Mapping[str, Any]
) -> None:
    _strict_keys(
        predecessor,
        set(COOLDOWN_PREDECESSOR_KEYS),
        name="cooldown_predecessor",
    )
    contextual = _seeded_canonicalize(
        tail["contextual_raw_signal"],
        _strict_bool(
            predecessor["contextual_virtual_signal"],
            name="contextual predecessor",
        ),
    )
    weak = _seeded_canonicalize(
        tail["weak_trend_raw_signal"],
        _strict_bool(
            predecessor["weak_trend_virtual_signal"],
            name="weak predecessor",
        ),
    )
    candidate = (contextual | weak).astype(bool)
    union = _seeded_canonicalize(
        candidate,
        _strict_bool(
            predecessor["unfiltered_union_signal"],
            name="union predecessor",
        ),
    )
    expected = {
        "contextual_virtual_signal": contextual,
        "weak_trend_virtual_signal": weak,
        "unfiltered_union_candidate_signal": candidate,
        "unfiltered_union_signal": union,
    }
    for column, values in expected.items():
        if not tail[column].equals(values.rename(column)):
            raise ValueError(f"signal cooldown tail violates {column} recurrence")


@dataclass(frozen=True)
class ReplayCheckpoint:
    """Strict, self-hashed replay envelope for deterministic continuation."""

    schema_version: int
    checkpoint_date: str
    source_start_date: str
    source_session_count: int
    runtime_segments: tuple[Mapping[str, Any], ...]
    fixed_feature_columns: tuple[str, ...]
    forecast_columns: tuple[str, ...]
    matured_event_columns: tuple[str, ...]
    model_payload: Mapping[str, Any]
    pending_lessons: tuple[Mapping[str, Any], ...]
    market_feature_tail: tuple[Mapping[str, Any], ...]
    signal_cooldown_tail: tuple[Mapping[str, Any], ...]
    cooldown_predecessor: Mapping[str, bool]
    opportunity_counts: Mapping[str, int]
    opportunity_hashes: Mapping[str, str]
    market_prefix_sha256: str
    fixed_feature_prefix_sha256: str
    forecast_prefix_sha256: str
    matured_event_prefix_sha256: str
    model_state_sha256: str
    pending_state_sha256: str
    market_history_tail_sha256: str
    signal_cooldown_tail_sha256: str
    digest_sha256: str

    def _without_digest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_date": self.checkpoint_date,
            "source_start_date": self.source_start_date,
            "source_session_count": self.source_session_count,
            "runtime_segments": copy.deepcopy(list(self.runtime_segments)),
            "fixed_feature_columns": list(self.fixed_feature_columns),
            "forecast_columns": list(self.forecast_columns),
            "matured_event_columns": list(self.matured_event_columns),
            "model_payload": copy.deepcopy(dict(self.model_payload)),
            "pending_lessons": copy.deepcopy(list(self.pending_lessons)),
            "market_feature_tail": copy.deepcopy(list(self.market_feature_tail)),
            "signal_cooldown_tail": copy.deepcopy(list(self.signal_cooldown_tail)),
            "cooldown_predecessor": dict(self.cooldown_predecessor),
            "opportunity_counts": dict(self.opportunity_counts),
            "opportunity_hashes": dict(self.opportunity_hashes),
            "market_prefix_sha256": self.market_prefix_sha256,
            "fixed_feature_prefix_sha256": self.fixed_feature_prefix_sha256,
            "forecast_prefix_sha256": self.forecast_prefix_sha256,
            "matured_event_prefix_sha256": self.matured_event_prefix_sha256,
            "model_state_sha256": self.model_state_sha256,
            "pending_state_sha256": self.pending_state_sha256,
            "market_history_tail_sha256": self.market_history_tail_sha256,
            "signal_cooldown_tail_sha256": self.signal_cooldown_tail_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._without_digest()
        payload["digest_sha256"] = self.digest_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayCheckpoint":
        expected = {
            "schema_version",
            "checkpoint_date",
            "source_start_date",
            "source_session_count",
            "runtime_segments",
            "fixed_feature_columns",
            "forecast_columns",
            "matured_event_columns",
            "model_payload",
            "pending_lessons",
            "market_feature_tail",
            "signal_cooldown_tail",
            "cooldown_predecessor",
            "opportunity_counts",
            "opportunity_hashes",
            "market_prefix_sha256",
            "fixed_feature_prefix_sha256",
            "forecast_prefix_sha256",
            "matured_event_prefix_sha256",
            "model_state_sha256",
            "pending_state_sha256",
            "market_history_tail_sha256",
            "signal_cooldown_tail_sha256",
            "digest_sha256",
        }
        _strict_keys(payload, expected, name="replay checkpoint")
        sequence_fields = (
            "fixed_feature_columns",
            "forecast_columns",
            "matured_event_columns",
            "runtime_segments",
            "pending_lessons",
            "market_feature_tail",
            "signal_cooldown_tail",
        )
        for field in sequence_fields:
            if not isinstance(payload[field], (list, tuple)):
                raise ValueError(f"checkpoint {field} must be a sequence")
        checkpoint = cls(
            schema_version=payload["schema_version"],
            checkpoint_date=payload["checkpoint_date"],
            source_start_date=payload["source_start_date"],
            source_session_count=payload["source_session_count"],
            runtime_segments=tuple(copy.deepcopy(payload["runtime_segments"])),
            fixed_feature_columns=tuple(payload["fixed_feature_columns"]),
            forecast_columns=tuple(payload["forecast_columns"]),
            matured_event_columns=tuple(payload["matured_event_columns"]),
            model_payload=copy.deepcopy(payload["model_payload"]),
            pending_lessons=tuple(copy.deepcopy(payload["pending_lessons"])),
            market_feature_tail=tuple(copy.deepcopy(payload["market_feature_tail"])),
            signal_cooldown_tail=tuple(
                copy.deepcopy(payload["signal_cooldown_tail"])
            ),
            cooldown_predecessor=copy.deepcopy(payload["cooldown_predecessor"]),
            opportunity_counts=copy.deepcopy(payload["opportunity_counts"]),
            opportunity_hashes=copy.deepcopy(payload["opportunity_hashes"]),
            market_prefix_sha256=payload["market_prefix_sha256"],
            fixed_feature_prefix_sha256=payload["fixed_feature_prefix_sha256"],
            forecast_prefix_sha256=payload["forecast_prefix_sha256"],
            matured_event_prefix_sha256=payload["matured_event_prefix_sha256"],
            model_state_sha256=payload["model_state_sha256"],
            pending_state_sha256=payload["pending_state_sha256"],
            market_history_tail_sha256=payload["market_history_tail_sha256"],
            signal_cooldown_tail_sha256=payload[
                "signal_cooldown_tail_sha256"
            ],
            digest_sha256=payload["digest_sha256"],
        )
        checkpoint.validate()
        return checkpoint

    def validate(self) -> None:
        if (
            isinstance(self.schema_version, (bool, np.bool_))
            or not isinstance(self.schema_version, (int, np.integer))
            or int(self.schema_version) != REPLAY_SCHEMA_VERSION
        ):
            raise ValueError("unsupported replay checkpoint schema_version")
        checkpoint_date = _iso_session_date(
            self.checkpoint_date, name="checkpoint_date"
        )
        start_date = _iso_session_date(
            self.source_start_date, name="source_start_date"
        )
        count = _strict_nonnegative_int(
            self.source_session_count, name="source_session_count"
        )
        if count == 0 or start_date > checkpoint_date:
            raise ValueError("checkpoint source bounds are invalid")
        if self.fixed_feature_columns != FIXED_FEATURE_COLUMNS:
            raise ValueError("checkpoint fixed feature schema is incompatible")
        if self.forecast_columns != FORECAST_COLUMNS:
            raise ValueError("checkpoint forecast schema is incompatible")
        if self.matured_event_columns != MATURED_EVENT_COLUMNS:
            raise ValueError("checkpoint matured-event schema is incompatible")

        model = ContextualExpertAggregator.from_dict(self.model_payload)
        model_payload = model.to_dict()
        state = model_payload["state"]
        if state["processed_session_count"] != count:
            raise ValueError("checkpoint model/session counts disagree")
        if state["last_session_date"] != checkpoint_date:
            raise ValueError("checkpoint model/date bounds disagree")
        observed_pending = [copy.deepcopy(dict(row)) for row in self.pending_lessons]
        if observed_pending != state["pending_lessons"]:
            raise ValueError("checkpoint pending state disagrees with model state")

        if not isinstance(self.runtime_segments, tuple) or not self.runtime_segments:
            raise ValueError("checkpoint runtime_segments must be a nonempty tuple")
        previous_offset = -1
        previous_first_date: str | None = None
        previous_runtime: dict[str, Any] | None = None
        for position, raw_segment in enumerate(self.runtime_segments):
            _strict_keys(
                raw_segment,
                set(RUNTIME_SEGMENT_FIELDS),
                name=f"runtime_segments[{position}]",
            )
            first_date = _iso_session_date(
                raw_segment["first_session_date"],
                name=f"runtime_segments[{position}].first_session_date",
            )
            offset = _strict_nonnegative_int(
                raw_segment["source_offset"],
                name=f"runtime_segments[{position}].source_offset",
            )
            if offset >= count:
                raise ValueError("runtime segment source_offset exceeds source")
            if first_date > checkpoint_date:
                raise ValueError("runtime segment begins after checkpoint_date")
            runtime = _canonical_runtime_payload(
                raw_segment["runtime"],
                name=f"runtime_segments[{position}].runtime",
            )
            if dict(raw_segment["runtime"]) != runtime:
                raise ValueError("runtime segment runtime is not canonical")
            prior_count = _strict_nonnegative_int(
                raw_segment["prior_segment_terminal_count"],
                name=(
                    f"runtime_segments[{position}]."
                    "prior_segment_terminal_count"
                ),
            )
            prior_date_raw = raw_segment["prior_segment_terminal_date"]
            parent_digest = raw_segment["parent_checkpoint_digest"]
            if position == 0:
                if (
                    offset != 0
                    or first_date != start_date
                    or prior_date_raw is not None
                    or prior_count != 0
                    or parent_digest is not None
                ):
                    raise ValueError("first runtime segment has invalid lineage")
            else:
                if offset <= previous_offset:
                    raise ValueError("runtime segments must have increasing offsets")
                if first_date <= str(previous_first_date):
                    raise ValueError("runtime segments must have increasing dates")
                if prior_count != offset:
                    raise ValueError(
                        "runtime segment prior terminal count must equal source_offset"
                    )
                if prior_date_raw is None:
                    raise ValueError("runtime transition requires prior terminal date")
                prior_date = _iso_session_date(
                    prior_date_raw,
                    name=(
                        f"runtime_segments[{position}]."
                        "prior_segment_terminal_date"
                    ),
                )
                if prior_date >= first_date or prior_date < str(previous_first_date):
                    raise ValueError("runtime segment prior terminal date is invalid")
                _require_sha256(
                    parent_digest,
                    name=(
                        f"runtime_segments[{position}]."
                        "parent_checkpoint_digest"
                    ),
                )
                if runtime == previous_runtime:
                    raise ValueError("redundant unchanged runtime segment is forbidden")
            previous_offset = offset
            previous_first_date = first_date
            previous_runtime = runtime
        if previous_runtime != model_payload["runtime"]:
            raise ValueError("terminal runtime segment disagrees with model runtime")

        market_tail = _market_tail_to_frame(self.market_feature_tail)
        signal_tail = _signal_tail_to_frame(self.signal_cooldown_tail)
        expected_tail_length = min(count, FEATURE_LOOKBACK_SESSIONS)
        if len(market_tail) != expected_tail_length or len(signal_tail) != expected_tail_length:
            raise ValueError("checkpoint feature/cooldown tail length is invalid")
        if not market_tail.index.equals(signal_tail.index):
            raise ValueError("checkpoint market and signal tails are misaligned")
        if market_tail.index[-1].date().isoformat() != checkpoint_date:
            raise ValueError("checkpoint tails do not end at checkpoint_date")
        if count == len(market_tail) and market_tail.index[0].date().isoformat() != start_date:
            raise ValueError("complete checkpoint tail does not begin at source_start_date")
        _validate_cooldown_tail(signal_tail, self.cooldown_predecessor)

        _strict_keys(
            self.opportunity_counts,
            set(OPPORTUNITY_SERIES),
            name="opportunity_counts",
        )
        _strict_keys(
            self.opportunity_hashes,
            set(OPPORTUNITY_SERIES),
            name="opportunity_hashes",
        )
        for name in OPPORTUNITY_SERIES:
            observed_count = _strict_nonnegative_int(
                self.opportunity_counts[name], name=f"opportunity_counts[{name}]"
            )
            if observed_count > count:
                raise ValueError("opportunity count exceeds source session count")
            _require_sha256(
                self.opportunity_hashes[name],
                name=f"opportunity_hashes[{name}]",
            )
        if (
            self.opportunity_counts["unfiltered_union_signal"]
            > self.opportunity_counts["unfiltered_union_candidate_signal"]
        ):
            raise ValueError("accepted union count exceeds candidate count")

        digest_fields = (
            "market_prefix_sha256",
            "fixed_feature_prefix_sha256",
            "forecast_prefix_sha256",
            "matured_event_prefix_sha256",
            "model_state_sha256",
            "pending_state_sha256",
            "market_history_tail_sha256",
            "signal_cooldown_tail_sha256",
            "digest_sha256",
        )
        for field in digest_fields:
            _require_sha256(getattr(self, field), name=field)
        if self.model_state_sha256 != _sha256_hex(state):
            raise ValueError("checkpoint model-state digest mismatch")
        if self.pending_state_sha256 != _sha256_hex(state["pending_lessons"]):
            raise ValueError("checkpoint pending-state digest mismatch")
        if self.market_history_tail_sha256 != _sha256_hex(
            state["market_history"]
        ):
            raise ValueError("checkpoint model-history digest mismatch")
        if self.signal_cooldown_tail_sha256 != _sha256_hex(
            list(self.signal_cooldown_tail)
        ):
            raise ValueError("checkpoint cooldown-tail digest mismatch")

        history = state["market_history"]
        expected_history_length = min(count, SLOW_LOOKBACK + 2)
        if len(history) != expected_history_length:
            raise ValueError("checkpoint model-history length is inconsistent")
        market_suffix = market_tail.iloc[-expected_history_length:]
        signal_suffix = signal_tail.iloc[-expected_history_length:]
        for position, history_row in enumerate(history):
            market_row = market_suffix.iloc[position]
            signal_row = signal_suffix.iloc[position]
            expected = {
                "session_date": market_suffix.index[position].date().isoformat(),
                "aapl_open": float(market_row["aapl_open"]),
                "aapl_close": float(market_row["aapl_close"]),
                "aapl_adj_close": float(market_row["aapl_adj_close"]),
                "aapl_adj_open": float(market_row["aapl_adj_open"]),
                "spy_adj_close": float(market_row["spy_adj_close"]),
                "qqq_adj_close": float(market_row["qqq_adj_close"]),
                "contextual_signal": bool(
                    signal_row["contextual_virtual_signal"]
                ),
                "weak_trend_signal": bool(
                    signal_row["weak_trend_virtual_signal"]
                ),
                "canonical_union_opportunity": bool(
                    signal_row["unfiltered_union_signal"]
                ),
            }
            if history_row != expected:
                raise ValueError("checkpoint replay tail disagrees with model history")

        expected_digest = _sha256_hex(self._without_digest())
        if self.digest_sha256 != expected_digest:
            raise ValueError("checkpoint envelope digest mismatch")


@dataclass(frozen=True)
class ReplayDiagnostics:
    processed_rows: int
    accepted_opportunities: int
    cash_actions: int
    matured_events: int
    admitted_events: int
    ending_pending_lessons: int
    fixed_feature_sha256: str
    forecast_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class ReplayResult:
    opportunity_frame: pd.DataFrame
    forecast: pd.DataFrame
    matured_lessons: pd.DataFrame
    checkpoint: ReplayCheckpoint
    diagnostics: ReplayDiagnostics

    def __post_init__(self) -> None:
        if tuple(self.opportunity_frame.columns) != FIXED_FEATURE_COLUMNS:
            raise ValueError("result opportunity frame has an incompatible schema")
        if tuple(self.forecast.columns) != FORECAST_COLUMNS:
            raise ValueError("result forecast has an incompatible schema")
        if tuple(self.matured_lessons.columns) != MATURED_EVENT_COLUMNS:
            raise ValueError("result matured-event frame has an incompatible schema")
        if not self.opportunity_frame.index.equals(self.forecast.index):
            raise ValueError("result opportunity and forecast indexes are misaligned")


@dataclass(frozen=True)
class ReplayEqualityProof:
    fixed_features_equal: bool
    forecasts_equal: bool
    matured_events_equal: bool
    model_state_equal: bool
    pending_state_equal: bool
    cooldown_tail_equal: bool
    prefix_hashes_equal: bool
    checkpoint_envelope_equal: bool
    runtime_equal: bool
    checkpoint_digest_equal: bool
    passed: bool

    def require(self) -> None:
        if not self.passed:
            raise ValueError("deterministic replay equality proof failed")


@dataclass(frozen=True)
class ConfirmationForks:
    checkpoint_sha256: str
    arms: Mapping[str, ReplayResult]

    def __post_init__(self) -> None:
        if tuple(self.arms) != CONFIRMATION_ARM_ORDER:
            raise ValueError("confirmation arms are missing or reordered")


@dataclass(frozen=True)
class _VerifiedPrefix:
    canonical_market: pd.DataFrame
    regenerated: ReplayResult


def _verify_complete_historical_prefix(
    historical_prefix: pd.DataFrame,
    checkpoint: ReplayCheckpoint,
) -> _VerifiedPrefix:
    """Replay and bind the complete prefix before any suffix is inspected."""

    checkpoint.validate()
    canonical_prefix = canonical_market_frame(historical_prefix)
    fixed_features = build_fixed_opportunity_frame(canonical_prefix)
    regenerated: ReplayResult | None = None
    segments = checkpoint.runtime_segments
    for position, segment in enumerate(segments):
        start = int(segment["source_offset"])
        stop = (
            int(segments[position + 1]["source_offset"])
            if position + 1 < len(segments)
            else len(canonical_prefix)
        )
        if start >= stop or stop > len(canonical_prefix):
            raise ValueError("runtime segment offsets do not partition prefix")
        first_date = canonical_prefix.index[start].date().isoformat()
        if first_date != segment["first_session_date"]:
            raise ValueError("runtime segment first session does not match prefix")
        previous_checkpoint = None if regenerated is None else regenerated.checkpoint
        if position:
            if previous_checkpoint is None:
                raise ValueError("runtime transition is missing regenerated parent")
            prior_date = canonical_prefix.index[start - 1].date().isoformat()
            if (
                segment["prior_segment_terminal_date"] != prior_date
                or segment["prior_segment_terminal_count"] != start
                or segment["prior_segment_terminal_date"]
                != previous_checkpoint.checkpoint_date
                or segment["prior_segment_terminal_count"]
                != previous_checkpoint.source_session_count
                or segment["parent_checkpoint_digest"]
                != previous_checkpoint.digest_sha256
            ):
                raise ValueError(
                    "runtime transition does not match regenerated parent checkpoint"
                )
        runtime = segment["runtime"]
        regenerated = _run_canonical_replay(
            canonical_prefix.iloc[start:stop].copy(),
            fixed_features.iloc[start:stop].copy(),
            previous=previous_checkpoint,
            learning_mode=runtime["learning_mode"],
            frozen_cutoff=runtime["frozen_cutoff"],
            ablation_mode=runtime["ablation_mode"],
        )
        regenerated.checkpoint.validate()
        expected_lineage = tuple(
            copy.deepcopy(list(checkpoint.runtime_segments[: position + 1]))
        )
        if regenerated.checkpoint.runtime_segments != expected_lineage:
            raise ValueError("regenerated runtime lineage differs from checkpoint")
    if regenerated is None:
        raise ValueError("historical prefix has no runtime segment")
    # Validate both envelopes independently immediately before exact equality.
    checkpoint.validate()
    regenerated.checkpoint.validate()
    if regenerated.checkpoint.to_dict() != checkpoint.to_dict():
        raise ValueError(
            "historical prefix replay does not reproduce the exact checkpoint envelope"
        )
    return _VerifiedPrefix(
        canonical_market=canonical_prefix,
        regenerated=regenerated,
    )


def _build_continuation_opportunity_frame(
    canonical_later: pd.DataFrame,
    checkpoint: ReplayCheckpoint,
    canonical_prefix: pd.DataFrame,
) -> pd.DataFrame:
    if len(canonical_prefix) != checkpoint.source_session_count:
        raise ValueError("historical prefix length disagrees with checkpoint")
    if canonical_prefix.index[0].date().isoformat() != checkpoint.source_start_date:
        raise ValueError("historical prefix start disagrees with checkpoint")
    if canonical_prefix.index[-1].date().isoformat() != checkpoint.checkpoint_date:
        raise ValueError("historical prefix end disagrees with checkpoint")
    if canonical_later.index[0] <= canonical_prefix.index[-1]:
        raise ValueError("continuation must begin strictly after checkpoint_date")

    observed_market_hash = _extend_frame_chain(
        _initial_chain("canonical_market_v1", CANONICAL_MARKET_COLUMNS),
        canonical_prefix,
        columns=CANONICAL_MARKET_COLUMNS,
    )
    if observed_market_hash != checkpoint.market_prefix_sha256:
        raise ValueError("historical prefix market hash disagrees with checkpoint")

    # Invoke the original fixed builder once over the exact same canonical
    # prefix that produced the checkpoint plus the new rows.  Slicing happens
    # only after all rolling numerics and both cooldowns have been computed.
    # This preserves even pandas' accumulator-level floating-point bytes.
    combined_market = pd.concat([canonical_prefix, canonical_later], axis=0)
    combined_built = _causal_builder_frame(combined_market)
    combined_features = pd.concat([combined_market, combined_built], axis=1)
    combined_features = _apply_comparator_targets(combined_features).loc[
        :, FIXED_FEATURE_COLUMNS
    ]
    prefix_features = combined_features.iloc[: len(canonical_prefix)].copy()

    observed_feature_hash = _extend_frame_chain(
        _initial_chain("fixed_features_v1", FIXED_FEATURE_COLUMNS),
        prefix_features,
        columns=FIXED_FEATURE_COLUMNS,
    )
    if observed_feature_hash != checkpoint.fixed_feature_prefix_sha256:
        raise ValueError("historical prefix fixed-feature hash disagrees with checkpoint")

    observed_counts = {
        name: int(prefix_features[name].sum()) for name in OPPORTUNITY_SERIES
    }
    if observed_counts != dict(checkpoint.opportunity_counts):
        raise ValueError("historical prefix opportunity counts disagree with checkpoint")
    observed_hashes: dict[str, str] = {}
    for name in OPPORTUNITY_SERIES:
        observed_hashes[name] = _extend_frame_chain(
            _initial_chain(f"opportunity_series_v1:{name}", (name,)),
            prefix_features.loc[:, [name]],
            columns=(name,),
        )
    if observed_hashes != dict(checkpoint.opportunity_hashes):
        raise ValueError("historical prefix opportunity hashes disagree with checkpoint")

    market_tail, signal_tail, predecessor = _tail_components(
        previous=None,
        canonical=canonical_prefix,
        features=prefix_features,
    )
    if market_tail != checkpoint.market_feature_tail:
        raise ValueError("historical prefix market tail disagrees with checkpoint")
    if signal_tail != checkpoint.signal_cooldown_tail:
        raise ValueError("historical prefix signal/cooldown tail disagrees with checkpoint")
    if predecessor != dict(checkpoint.cooldown_predecessor):
        raise ValueError("historical prefix cooldown predecessor disagrees with checkpoint")

    return combined_features.iloc[len(canonical_prefix) :].copy()


def build_fixed_opportunity_frame(
    frame: pd.DataFrame,
    *,
    checkpoint: ReplayCheckpoint | Mapping[str, Any] | None = None,
    historical_prefix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the exact causal fixed-feature and accepted-opportunity frame."""

    if checkpoint is None:
        if historical_prefix is not None:
            raise ValueError("historical_prefix is valid only for checkpoint continuation")
        canonical = canonical_market_frame(frame)
        built = _causal_builder_frame(canonical)
        result = pd.concat([canonical, built], axis=1)
        result = _apply_comparator_targets(result)
        return result.loc[:, FIXED_FEATURE_COLUMNS].copy()
    if historical_prefix is None:
        raise ValueError("checkpoint continuation requires the complete historical_prefix")
    parsed = (
        checkpoint
        if isinstance(checkpoint, ReplayCheckpoint)
        else ReplayCheckpoint.from_dict(checkpoint)
    )
    verified = _verify_complete_historical_prefix(historical_prefix, parsed)
    # The suffix is intentionally not canonicalized until prefix replay and
    # exact envelope equality have succeeded.
    canonical = canonical_market_frame(frame)
    return _build_continuation_opportunity_frame(
        canonical,
        parsed,
        verified.canonical_market,
    )


# Concise public alias used by runner-side orchestration.
build_opportunity_frame = build_fixed_opportunity_frame


def _matured_record(
    lesson: Any,
    *,
    entry_adjusted_open: float,
    exit_adjusted_open: float,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "signal_date": lesson.signal_date,
        "maturity_date": lesson.maturity_date,
        "signal_market_state": lesson.market_state or "unknown",
        "entry_adjusted_open": float(entry_adjusted_open),
        "exit_adjusted_open": float(exit_adjusted_open),
        "net_cash_log_edge_10bps": float(lesson.net_cash_log_edge_10bps),
        "reward_range": float(lesson.reward_range),
        "admitted": bool(lesson.admitted),
    }
    record.update(
        {
            f"advice__{name}": bool(lesson.expert_cash_advice[name])
            for name in EXPERT_NAMES
        }
    )
    record.update(
        {
            f"reward__{name}": float(lesson.expert_rewards[name])
            for name in EXPERT_NAMES
        }
    )
    return record


def _decision_row(
    *,
    decision: Any,
    feature_row: pd.Series,
    model: ContextualExpertAggregator,
    matured_records: Sequence[Mapping[str, Any]],
    opportunity_count_to_date: int,
) -> dict[str, Any]:
    if len(matured_records) > 1:
        raise ValueError("canonical cooldown permits at most one maturity per close")
    row: dict[str, Any] = {
        "aapl_adj_open": float(decision.adjusted_open),
        "contextual_virtual_signal": bool(
            feature_row["contextual_virtual_signal"]
        ),
        "weak_trend_virtual_signal": bool(
            feature_row["weak_trend_virtual_signal"]
        ),
        "unfiltered_union_candidate_signal": bool(
            feature_row["unfiltered_union_candidate_signal"]
        ),
        "canonical_union_opportunity": bool(decision.opportunity),
        **{
            column: float(feature_row[column])
            for column in COMPARATOR_TARGET_COLUMNS
        },
        "market_state": decision.market_state,
        "Q": float(decision.cash_score),
        "cash_score": float(decision.cash_score),
        "action": decision.action,
        "learner_target_exposure": 0.0 if decision.action == "CASH" else 1.0,
    }
    row.update(
        {
            f"advice__{name}": bool(decision.expert_cash_advice[name])
            for name in EXPERT_NAMES
        }
    )
    row.update(
        {
            f"aggregate_weight__{name}": float(
                decision.aggregate_expert_weights[name]
            )
            for name in EXPERT_NAMES
        }
    )
    for scale in SCALE_NAMES:
        active = scale in decision.scale_expert_weights
        row[f"scale_active__{scale}"] = active
        for name in EXPERT_NAMES:
            row[f"weight__{scale}__{name}"] = (
                float(decision.scale_expert_weights[scale][name])
                if active
                else np.nan
            )
        row[f"rho__{scale}"] = (
            float(decision.scale_state_pooling[scale]) if active else np.nan
        )

    row.update(
        {
            "matured_on_close_count": len(matured_records),
            "matured_signal_date": "",
            "matured_signal_market_state": "",
            "matured_entry_adjusted_open": np.nan,
            "matured_exit_adjusted_open": np.nan,
            "matured_net_cash_log_edge_10bps": np.nan,
            "matured_reward_range": np.nan,
            "matured_admitted": False,
        }
    )
    row.update({f"matured_advice__{name}": False for name in EXPERT_NAMES})
    row.update({f"matured_reward__{name}": np.nan for name in EXPERT_NAMES})
    if matured_records:
        event = matured_records[0]
        row.update(
            {
                "matured_signal_date": event["signal_date"],
                "matured_signal_market_state": event["signal_market_state"],
                "matured_entry_adjusted_open": event["entry_adjusted_open"],
                "matured_exit_adjusted_open": event["exit_adjusted_open"],
                "matured_net_cash_log_edge_10bps": event[
                    "net_cash_log_edge_10bps"
                ],
                "matured_reward_range": event["reward_range"],
                "matured_admitted": event["admitted"],
            }
        )
        row.update(
            {
                f"matured_advice__{name}": event[f"advice__{name}"]
                for name in EXPERT_NAMES
            }
        )
        row.update(
            {
                f"matured_reward__{name}": event[f"reward__{name}"]
                for name in EXPERT_NAMES
            }
        )
    state = model.state_dict()
    row.update(
        {
            "processed_session_count": state["processed_session_count"],
            "matured_lesson_count": state["matured_lesson_count"],
            "admitted_lesson_count": state["admitted_lesson_count"],
            "pending_lesson_count": decision.pending_lesson_count,
            "opportunity_count_to_date": opportunity_count_to_date,
        }
    )
    return {column: row[column] for column in FORECAST_COLUMNS}


def _tail_components(
    *,
    previous: ReplayCheckpoint | None,
    canonical: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, bool],
]:
    if previous is None:
        old_market = _empty_indexed_frame(CANONICAL_MARKET_COLUMNS)
        old_signals = _empty_indexed_frame(COOLDOWN_SIGNAL_COLUMNS)
        old_predecessor = {name: False for name in COOLDOWN_PREDECESSOR_KEYS}
    else:
        old_market = _market_tail_to_frame(previous.market_feature_tail)
        old_signals = _signal_tail_to_frame(previous.signal_cooldown_tail)
        old_predecessor = dict(previous.cooldown_predecessor)
    new_signals = features.loc[:, COOLDOWN_SIGNAL_COLUMNS]
    combined_market = (
        canonical.copy()
        if old_market.empty
        else pd.concat([old_market, canonical])
    )
    combined_signals = (
        new_signals.copy()
        if old_signals.empty
        else pd.concat([old_signals, new_signals])
    )
    for column in COOLDOWN_SIGNAL_COLUMNS:
        combined_signals[column] = _strict_boolean_series(
            combined_signals[column], name=column
        )
    if not combined_market.index.equals(combined_signals.index):
        raise ValueError("market and cooldown tail construction diverged")
    cut = max(0, len(combined_market) - FEATURE_LOOKBACK_SESSIONS)
    if cut:
        predecessor_row = combined_signals.iloc[cut - 1]
        predecessor = {
            name: bool(predecessor_row[name]) for name in COOLDOWN_PREDECESSOR_KEYS
        }
    else:
        predecessor = old_predecessor
    market_tail = combined_market.iloc[cut:].copy()
    signal_tail = combined_signals.iloc[cut:].copy()
    _validate_cooldown_tail(signal_tail, predecessor)
    return (
        _records_from_frame(market_tail),
        _records_from_frame(signal_tail),
        predecessor,
    )


def _make_checkpoint(
    *,
    previous: ReplayCheckpoint | None,
    canonical: pd.DataFrame,
    features: pd.DataFrame,
    forecast: pd.DataFrame,
    matured_events: pd.DataFrame,
    model: ContextualExpertAggregator,
) -> ReplayCheckpoint:
    if previous is None:
        source_start = canonical.index[0].date().isoformat()
        source_count = 0
        market_hash = _initial_chain("canonical_market_v1", CANONICAL_MARKET_COLUMNS)
        feature_hash = _initial_chain("fixed_features_v1", FIXED_FEATURE_COLUMNS)
        forecast_hash = _initial_chain("aggregation_forecast_v1", FORECAST_COLUMNS)
        matured_hash = _initial_chain("matured_events_v1", MATURED_EVENT_COLUMNS)
        counts = {name: 0 for name in OPPORTUNITY_SERIES}
        opportunity_hashes = {
            name: _initial_chain(f"opportunity_series_v1:{name}", (name,))
            for name in OPPORTUNITY_SERIES
        }
    else:
        source_start = previous.source_start_date
        source_count = previous.source_session_count
        market_hash = previous.market_prefix_sha256
        feature_hash = previous.fixed_feature_prefix_sha256
        forecast_hash = previous.forecast_prefix_sha256
        matured_hash = previous.matured_event_prefix_sha256
        counts = dict(previous.opportunity_counts)
        opportunity_hashes = dict(previous.opportunity_hashes)

    market_hash = _extend_frame_chain(
        market_hash, canonical, columns=CANONICAL_MARKET_COLUMNS
    )
    feature_hash = _extend_frame_chain(
        feature_hash, features, columns=FIXED_FEATURE_COLUMNS
    )
    forecast_hash = _extend_frame_chain(
        forecast_hash, forecast, columns=FORECAST_COLUMNS
    )
    matured_hash = _extend_event_chain(
        matured_hash, matured_events, columns=MATURED_EVENT_COLUMNS
    )
    for name in OPPORTUNITY_SERIES:
        counts[name] += int(features[name].sum())
        one = features.loc[:, [name]].copy()
        opportunity_hashes[name] = _extend_frame_chain(
            opportunity_hashes[name], one, columns=(name,)
        )

    market_tail, signal_tail, predecessor = _tail_components(
        previous=previous,
        canonical=canonical,
        features=features,
    )
    model_payload = model.to_dict()
    state = model_payload["state"]
    pending = tuple(copy.deepcopy(state["pending_lessons"]))
    current_runtime = copy.deepcopy(model_payload["runtime"])
    if previous is None:
        runtime_segments: list[dict[str, Any]] = [
            {
                "first_session_date": canonical.index[0].date().isoformat(),
                "source_offset": 0,
                "runtime": current_runtime,
                "prior_segment_terminal_date": None,
                "prior_segment_terminal_count": 0,
                "parent_checkpoint_digest": None,
            }
        ]
    else:
        runtime_segments = copy.deepcopy(list(previous.runtime_segments))
        if current_runtime != previous.model_payload["runtime"]:
            runtime_segments.append(
                {
                    "first_session_date": canonical.index[0].date().isoformat(),
                    "source_offset": previous.source_session_count,
                    "runtime": current_runtime,
                    "prior_segment_terminal_date": previous.checkpoint_date,
                    "prior_segment_terminal_count": previous.source_session_count,
                    "parent_checkpoint_digest": previous.digest_sha256,
                }
            )
    without_digest = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "checkpoint_date": canonical.index[-1].date().isoformat(),
        "source_start_date": source_start,
        "source_session_count": source_count + len(canonical),
        "runtime_segments": runtime_segments,
        "fixed_feature_columns": list(FIXED_FEATURE_COLUMNS),
        "forecast_columns": list(FORECAST_COLUMNS),
        "matured_event_columns": list(MATURED_EVENT_COLUMNS),
        "model_payload": model_payload,
        "pending_lessons": list(pending),
        "market_feature_tail": list(market_tail),
        "signal_cooldown_tail": list(signal_tail),
        "cooldown_predecessor": predecessor,
        "opportunity_counts": counts,
        "opportunity_hashes": opportunity_hashes,
        "market_prefix_sha256": market_hash,
        "fixed_feature_prefix_sha256": feature_hash,
        "forecast_prefix_sha256": forecast_hash,
        "matured_event_prefix_sha256": matured_hash,
        "model_state_sha256": _sha256_hex(state),
        "pending_state_sha256": _sha256_hex(state["pending_lessons"]),
        "market_history_tail_sha256": _sha256_hex(state["market_history"]),
        "signal_cooldown_tail_sha256": _sha256_hex(list(signal_tail)),
    }
    payload = dict(without_digest)
    payload["digest_sha256"] = _sha256_hex(without_digest)
    return ReplayCheckpoint.from_dict(payload)


def _selected_runtime(
    *,
    previous: ReplayCheckpoint | None,
    learning_mode: str | None,
    frozen_cutoff: Any,
    ablation_mode: str | None,
) -> tuple[str, Any, str]:
    if previous is None:
        selected_learning = (
            CAUSAL_ONLINE_MODE if learning_mode is None else learning_mode
        )
        selected_ablation = FULL_MODE if ablation_mode is None else ablation_mode
    else:
        runtime = previous.model_payload["runtime"]
        selected_learning = (
            runtime["learning_mode"] if learning_mode is None else learning_mode
        )
        selected_ablation = (
            runtime["ablation_mode"] if ablation_mode is None else ablation_mode
        )
        if frozen_cutoff is None and selected_learning == FROZEN_CUTOFF_MODE:
            frozen_cutoff = runtime["frozen_cutoff"]
    if selected_learning not in LEARNING_MODES:
        raise ValueError(f"unsupported learning_mode: {selected_learning!r}")
    if selected_ablation not in ABLATION_MODES:
        raise ValueError(f"unsupported ablation_mode: {selected_ablation!r}")
    if selected_learning == CAUSAL_ONLINE_MODE:
        if frozen_cutoff is not None:
            raise ValueError("causal_online mode must not receive frozen_cutoff")
        selected_cutoff = None
    else:
        if frozen_cutoff is None:
            raise ValueError("frozen_cutoff mode requires frozen_cutoff")
        selected_cutoff = _iso_session_date(frozen_cutoff, name="frozen_cutoff")
    return selected_learning, selected_cutoff, selected_ablation


def _run_canonical_replay(
    canonical: pd.DataFrame,
    features: pd.DataFrame,
    *,
    previous: ReplayCheckpoint | None,
    learning_mode: str,
    frozen_cutoff: Any,
    ablation_mode: str,
) -> ReplayResult:
    if not canonical.index.equals(features.index):
        raise ValueError("canonical market and feature rows are misaligned")
    if tuple(features.columns) != FIXED_FEATURE_COLUMNS:
        raise ValueError("fixed features have an incompatible schema")
    if previous is None:
        model = ContextualExpertAggregator(
            learning_mode=learning_mode,
            frozen_cutoff=frozen_cutoff,
            ablation_mode=ablation_mode,
        )
        base_opportunity_count = 0
    else:
        previous.validate()
        model = ContextualExpertAggregator.from_dict(
            previous.model_payload,
            learning_mode=learning_mode,
            frozen_cutoff=frozen_cutoff,
            ablation_mode=ablation_mode,
        )
        base_opportunity_count = previous.opportunity_counts[
            "unfiltered_union_signal"
        ]

    forecast_records: list[dict[str, Any]] = []
    matured_records: list[dict[str, Any]] = []
    opportunity_count = int(base_opportunity_count)
    for row_index, market_row in canonical.iterrows():
        feature_row = features.loc[row_index]
        pending_before = {
            lesson.signal_date: lesson
            for lesson in model.pending_lessons
            if lesson.sessions_until_maturity == 1
        }
        session = MarketSession(
            session_date=row_index.date().isoformat(),
            aapl_open=float(market_row["aapl_open"]),
            aapl_close=float(market_row["aapl_close"]),
            aapl_adj_close=float(market_row["aapl_adj_close"]),
            aapl_adj_open=float(market_row["aapl_adj_open"]),
            spy_adj_close=float(market_row["spy_adj_close"]),
            qqq_adj_close=float(market_row["qqq_adj_close"]),
            contextual_signal=bool(feature_row["contextual_virtual_signal"]),
            weak_trend_signal=bool(feature_row["weak_trend_virtual_signal"]),
            canonical_union_opportunity=bool(
                feature_row["unfiltered_union_signal"]
            ),
        )
        decision = model.process_session(session)
        current_matured: list[dict[str, Any]] = []
        for lesson in decision.matured_lessons:
            pending = pending_before.get(lesson.signal_date)
            if pending is None or pending.entry_adjusted_open is None:
                raise ValueError("matured lesson is absent from pre-close pending state")
            event = _matured_record(
                lesson,
                entry_adjusted_open=float(pending.entry_adjusted_open),
                exit_adjusted_open=float(decision.adjusted_open),
            )
            current_matured.append(event)
            matured_records.append(event)
        if decision.opportunity:
            opportunity_count += 1
        forecast_records.append(
            _decision_row(
                decision=decision,
                feature_row=feature_row,
                model=model,
                matured_records=current_matured,
                opportunity_count_to_date=opportunity_count,
            )
        )

    forecast = pd.DataFrame(
        forecast_records,
        index=pd.DatetimeIndex(canonical.index.copy(), name="date"),
        columns=FORECAST_COLUMNS,
    )
    matured = _matured_event_frame(matured_records)
    checkpoint_result = _make_checkpoint(
        previous=previous,
        canonical=canonical,
        features=features,
        forecast=forecast,
        matured_events=matured,
        model=model,
    )
    diagnostics = ReplayDiagnostics(
        processed_rows=len(canonical),
        accepted_opportunities=int(features["unfiltered_union_signal"].sum()),
        cash_actions=int((forecast["action"] == "CASH").sum()),
        matured_events=len(matured),
        admitted_events=int(matured["admitted"].sum()) if len(matured) else 0,
        ending_pending_lessons=len(checkpoint_result.pending_lessons),
        fixed_feature_sha256=checkpoint_result.fixed_feature_prefix_sha256,
        forecast_sha256=checkpoint_result.forecast_prefix_sha256,
        checkpoint_sha256=checkpoint_result.digest_sha256,
    )
    return ReplayResult(
        opportunity_frame=features,
        forecast=forecast,
        matured_lessons=matured,
        checkpoint=checkpoint_result,
        diagnostics=diagnostics,
    )


def replay(
    frame: pd.DataFrame,
    *,
    checkpoint: ReplayCheckpoint | Mapping[str, Any] | None = None,
    historical_prefix: pd.DataFrame | None = None,
    learning_mode: str | None = None,
    frozen_cutoff: Any = None,
    ablation_mode: str | None = None,
) -> ReplayResult:
    """Replay from empty state or continue only after exact prefix replay."""

    if checkpoint is None:
        if historical_prefix is not None:
            raise ValueError(
                "historical_prefix is valid only for checkpoint continuation"
            )
        selected_learning, selected_cutoff, selected_ablation = _selected_runtime(
            previous=None,
            learning_mode=learning_mode,
            frozen_cutoff=frozen_cutoff,
            ablation_mode=ablation_mode,
        )
        canonical = canonical_market_frame(frame)
        features = build_fixed_opportunity_frame(canonical)
        return _run_canonical_replay(
            canonical,
            features,
            previous=None,
            learning_mode=selected_learning,
            frozen_cutoff=selected_cutoff,
            ablation_mode=selected_ablation,
        )

    parsed = (
        checkpoint
        if isinstance(checkpoint, ReplayCheckpoint)
        else ReplayCheckpoint.from_dict(checkpoint)
    )
    if historical_prefix is None:
        raise ValueError(
            "checkpoint continuation requires the complete historical_prefix"
        )
    verified = _verify_complete_historical_prefix(historical_prefix, parsed)
    selected_learning, selected_cutoff, selected_ablation = _selected_runtime(
        previous=parsed,
        learning_mode=learning_mode,
        frozen_cutoff=frozen_cutoff,
        ablation_mode=ablation_mode,
    )
    # No suffix property, value, index, or column is touched before the exact
    # prefix replay above succeeds.
    canonical = canonical_market_frame(frame)
    if canonical.index[0] <= pd.Timestamp(parsed.checkpoint_date):
        raise ValueError("continuation must contain later sessions only")
    features = _build_continuation_opportunity_frame(
        canonical,
        parsed,
        verified.canonical_market,
    )
    return _run_canonical_replay(
        canonical,
        features,
        previous=parsed,
        learning_mode=selected_learning,
        frozen_cutoff=selected_cutoff,
        ablation_mode=selected_ablation,
    )


def replay_from_empty(
    frame: pd.DataFrame,
    *,
    learning_mode: str | None = None,
    frozen_cutoff: Any = None,
    ablation_mode: str | None = None,
) -> ReplayResult:
    return replay(
        frame,
        learning_mode=learning_mode,
        frozen_cutoff=frozen_cutoff,
        ablation_mode=ablation_mode,
    )


def continue_from_checkpoint(
    later_frame: pd.DataFrame,
    checkpoint: ReplayCheckpoint | Mapping[str, Any],
    *,
    historical_prefix: pd.DataFrame,
    learning_mode: str | None = None,
    frozen_cutoff: Any = None,
    ablation_mode: str | None = None,
) -> ReplayResult:
    return replay(
        later_frame,
        checkpoint=checkpoint,
        historical_prefix=historical_prefix,
        learning_mode=learning_mode,
        frozen_cutoff=frozen_cutoff,
        ablation_mode=ablation_mode,
    )


replay_from_checkpoint = continue_from_checkpoint


def fork_confirmation_arms(
    later_frame: pd.DataFrame,
    checkpoint: ReplayCheckpoint | Mapping[str, Any],
    *,
    historical_prefix: pd.DataFrame,
    cutoff: Any = "2018-12-31",
) -> ConfirmationForks:
    parsed = (
        checkpoint
        if isinstance(checkpoint, ReplayCheckpoint)
        else ReplayCheckpoint.from_dict(checkpoint)
    )
    parsed.validate()
    cutoff_iso = _iso_session_date(cutoff, name="confirmation cutoff")
    if parsed.checkpoint_date != cutoff_iso:
        raise ValueError("confirmation must fork the exact cutoff checkpoint")
    runtime = parsed.model_payload["runtime"]
    if runtime != {
        "learning_mode": CAUSAL_ONLINE_MODE,
        "frozen_cutoff": None,
        "ablation_mode": FULL_MODE,
    }:
        raise ValueError("confirmation checkpoint must be causal-online full mode")
    verified = _verify_complete_historical_prefix(historical_prefix, parsed)
    # The suffix is touched only after the one shared exact prefix replay.
    canonical_later = canonical_market_frame(later_frame)
    if canonical_later.index[0] <= pd.Timestamp(parsed.checkpoint_date):
        raise ValueError("continuation must contain later sessions only")
    features = _build_continuation_opportunity_frame(
        canonical_later,
        parsed,
        verified.canonical_market,
    )
    arms: dict[str, ReplayResult] = {}
    arms[ONLINE_FULL_ARM] = _run_canonical_replay(
        canonical_later,
        features,
        previous=parsed,
        learning_mode=CAUSAL_ONLINE_MODE,
        frozen_cutoff=None,
        ablation_mode=FULL_MODE,
    )
    arms[FROZEN_2018_ARM] = _run_canonical_replay(
        canonical_later,
        features,
        previous=parsed,
        learning_mode=FROZEN_CUTOFF_MODE,
        frozen_cutoff=cutoff_iso,
        ablation_mode=FULL_MODE,
    )
    arms[GLOBAL_ONLY_ARM] = _run_canonical_replay(
        canonical_later,
        features,
        previous=parsed,
        learning_mode=CAUSAL_ONLINE_MODE,
        frozen_cutoff=None,
        ablation_mode=GLOBAL_ONLY_MODE,
    )
    arms[LIFETIME_ONLY_ARM] = _run_canonical_replay(
        canonical_later,
        features,
        previous=parsed,
        learning_mode=CAUSAL_ONLINE_MODE,
        frozen_cutoff=None,
        ablation_mode=LIFETIME_ONLY_MODE,
    )
    online_state = arms[ONLINE_FULL_ARM].checkpoint.model_payload["state"]
    for name in (GLOBAL_ONLY_ARM, LIFETIME_ONLY_ARM):
        if arms[name].checkpoint.model_payload["state"] != online_state:
            raise ValueError("online ablation forks changed action-independent learning")
    return ConfirmationForks(
        checkpoint_sha256=parsed.digest_sha256,
        arms=arms,
    )


def _canonical_event_frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if tuple(left.columns) != MATURED_EVENT_COLUMNS:
        return False
    if tuple(right.columns) != MATURED_EVENT_COLUMNS or len(left) != len(right):
        return False
    left_records = [
        {column: row[column] for column in MATURED_EVENT_COLUMNS}
        for _, row in left.iterrows()
    ]
    right_records = [
        {column: row[column] for column in MATURED_EVENT_COLUMNS}
        for _, row in right.iterrows()
    ]
    return _canonical_bytes(left_records) == _canonical_bytes(right_records)


def _proof(left: ReplayResult, right: ReplayResult) -> ReplayEqualityProof:
    checkpoints = (left.checkpoint, right.checkpoint)
    checkpoints[0].validate()
    checkpoints[1].validate()
    fixed_equal = left.opportunity_frame.equals(right.opportunity_frame)
    forecast_equal = left.forecast.equals(right.forecast)
    matured_equal = _canonical_event_frames_equal(
        left.matured_lessons, right.matured_lessons
    )
    state_equal = (
        checkpoints[0].model_payload["state"]
        == checkpoints[1].model_payload["state"]
    )
    pending_equal = checkpoints[0].pending_lessons == checkpoints[1].pending_lessons
    cooldown_equal = (
        checkpoints[0].signal_cooldown_tail
        == checkpoints[1].signal_cooldown_tail
        and checkpoints[0].cooldown_predecessor
        == checkpoints[1].cooldown_predecessor
    )
    hashes_equal = all(
        getattr(checkpoints[0], name) == getattr(checkpoints[1], name)
        for name in (
            "market_prefix_sha256",
            "fixed_feature_prefix_sha256",
            "forecast_prefix_sha256",
            "matured_event_prefix_sha256",
        )
    ) and checkpoints[0].opportunity_hashes == checkpoints[1].opportunity_hashes
    runtime_equal = (
        checkpoints[0].model_payload["runtime"]
        == checkpoints[1].model_payload["runtime"]
    )
    digest_equal = checkpoints[0].digest_sha256 == checkpoints[1].digest_sha256
    envelope_equal = checkpoints[0].to_dict() == checkpoints[1].to_dict()
    passed = all(
        (
            fixed_equal,
            forecast_equal,
            matured_equal,
            state_equal,
            pending_equal,
            cooldown_equal,
            hashes_equal,
            runtime_equal,
            digest_equal,
            envelope_equal,
        )
    )
    return ReplayEqualityProof(
        fixed_features_equal=fixed_equal,
        forecasts_equal=forecast_equal,
        matured_events_equal=matured_equal,
        model_state_equal=state_equal,
        pending_state_equal=pending_equal,
        cooldown_tail_equal=cooldown_equal,
        prefix_hashes_equal=hashes_equal,
        checkpoint_envelope_equal=envelope_equal,
        runtime_equal=runtime_equal,
        checkpoint_digest_equal=digest_equal,
        passed=passed,
    )


def prove_replay_prefix(
    sealed_prefix: ReplayResult,
    regenerated_prefix: ReplayResult,
) -> ReplayEqualityProof:
    """Prove two physically truncated replays end at the same exact prefix."""

    sealed_prefix.checkpoint.validate()
    regenerated_prefix.checkpoint.validate()
    if sealed_prefix.checkpoint.checkpoint_date != regenerated_prefix.checkpoint.checkpoint_date:
        raise ValueError("prefix proof requires the same physical cutoff")
    if sealed_prefix.checkpoint.source_session_count != regenerated_prefix.checkpoint.source_session_count:
        raise ValueError("prefix proof requires the same source session count")
    return _proof(sealed_prefix, regenerated_prefix)


def verify_resume_equivalence(
    full_history: ReplayResult,
    resumed_suffix: ReplayResult,
) -> ReplayEqualityProof:
    """Compare a suffix continuation with the same rows from full replay."""

    full_history.checkpoint.validate()
    resumed_suffix.checkpoint.validate()
    if resumed_suffix.forecast.empty:
        raise ValueError("resumed suffix must not be empty")
    suffix_index = resumed_suffix.forecast.index
    try:
        full_features = full_history.opportunity_frame.loc[suffix_index]
        full_forecast = full_history.forecast.loc[suffix_index]
    except KeyError as exc:
        raise ValueError("full replay does not contain the resumed suffix") from exc
    full_matured = full_history.matured_lessons
    if len(full_matured):
        start = suffix_index[0].date().isoformat()
        full_matured = full_matured.loc[
            full_matured["maturity_date"] >= start
        ].reset_index(drop=True)
    resumed_matured = resumed_suffix.matured_lessons.reset_index(drop=True)
    fixed_equal = full_features.equals(resumed_suffix.opportunity_frame)
    forecast_equal = full_forecast.equals(resumed_suffix.forecast)
    matured_equal = _canonical_event_frames_equal(full_matured, resumed_matured)
    full_checkpoint = full_history.checkpoint
    resumed_checkpoint = resumed_suffix.checkpoint
    state_equal = (
        full_checkpoint.model_payload["state"]
        == resumed_checkpoint.model_payload["state"]
    )
    pending_equal = full_checkpoint.pending_lessons == resumed_checkpoint.pending_lessons
    cooldown_equal = (
        full_checkpoint.signal_cooldown_tail
        == resumed_checkpoint.signal_cooldown_tail
        and full_checkpoint.cooldown_predecessor
        == resumed_checkpoint.cooldown_predecessor
    )
    hashes_equal = all(
        getattr(full_checkpoint, name) == getattr(resumed_checkpoint, name)
        for name in (
            "market_prefix_sha256",
            "fixed_feature_prefix_sha256",
            "forecast_prefix_sha256",
            "matured_event_prefix_sha256",
        )
    ) and full_checkpoint.opportunity_hashes == resumed_checkpoint.opportunity_hashes
    runtime_equal = (
        full_checkpoint.model_payload["runtime"]
        == resumed_checkpoint.model_payload["runtime"]
    )
    digest_equal = full_checkpoint.digest_sha256 == resumed_checkpoint.digest_sha256
    envelope_equal = full_checkpoint.to_dict() == resumed_checkpoint.to_dict()
    passed = all(
        (
            fixed_equal,
            forecast_equal,
            matured_equal,
            state_equal,
            pending_equal,
            cooldown_equal,
            hashes_equal,
            runtime_equal,
            digest_equal,
            envelope_equal,
        )
    )
    return ReplayEqualityProof(
        fixed_features_equal=fixed_equal,
        forecasts_equal=forecast_equal,
        matured_events_equal=matured_equal,
        model_state_equal=state_equal,
        pending_state_equal=pending_equal,
        cooldown_tail_equal=cooldown_equal,
        prefix_hashes_equal=hashes_equal,
        checkpoint_envelope_equal=envelope_equal,
        runtime_equal=runtime_equal,
        checkpoint_digest_equal=digest_equal,
        passed=passed,
    )


prove_resume_equivalence = verify_resume_equivalence


__all__ = [
    "CANONICAL_MARKET_COLUMNS",
    "COMPARATOR_TARGET_COLUMNS",
    "CONFIRMATION_ARM_ORDER",
    "FEATURE_LOOKBACK_SESSIONS",
    "FIXED_FEATURE_COLUMNS",
    "FORECAST_COLUMNS",
    "FROZEN_2018_ARM",
    "GLOBAL_ONLY_ARM",
    "LIFETIME_ONLY_ARM",
    "MATURED_EVENT_COLUMNS",
    "ONLINE_FULL_ARM",
    "ConfirmationForks",
    "ReplayCheckpoint",
    "ReplayDiagnostics",
    "ReplayEqualityProof",
    "ReplayResult",
    "build_fixed_opportunity_frame",
    "build_opportunity_frame",
    "canonical_market_frame",
    "continue_from_checkpoint",
    "fork_confirmation_arms",
    "prove_replay_prefix",
    "prove_resume_equivalence",
    "replay",
    "replay_from_checkpoint",
    "replay_from_empty",
    "verify_resume_equivalence",
]
