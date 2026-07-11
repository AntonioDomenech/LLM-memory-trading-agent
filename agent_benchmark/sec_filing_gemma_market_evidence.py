"""Pure, exact market-data evidence for the SEC-filing Gemma experiment.

This module performs no filesystem, network, dataframe, or model I/O.  It
canonicalizes an already materialized AAPL market frame, binds the exact local
source artifacts and source-window slices, and creates decision-time prefixes
that cannot contain a row after the completed decision-session close.

All market numbers are serialized with :meth:`float.hex`.  Decimal rendering
is deliberately forbidden: even a one-ULP change must change the row chain and
the externally pinned manifest identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from datetime import date
import hashlib
import hmac
import json
import math
import re
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_contract import (
    LABEL_MATURITY_OFFSET,
    MARKET_HISTORY_START,
    MARKET_LOOKBACK_SESSIONS,
    STAGE_ORDER,
    STAGE_WINDOWS,
    SecFilingGemmaContractError,
    canonical_market_session_calendar,
    market_session_calendar_sha256,
)
from agent_benchmark.sec_session_calendar import (
    EXPECTED_MARKET_HISTORY_SESSIONS,
    MARKET_HISTORY_CALENDAR_ID,
)


MARKET_SYMBOLS: Final[tuple[str, ...]] = (
    "AAPL",
    "SPY",
    "QQQ",
    "IWM",
    "VIX",
    "TNX",
)
CONTEXT_SYMBOLS: Final[tuple[str, ...]] = MARKET_SYMBOLS[1:]
MARKET_FIELDS: Final[tuple[str, ...]] = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)
PRICE_FIELDS: Final[tuple[str, ...]] = MARKET_FIELDS[:-1]
DERIVED_MARKET_FIELDS: Final[tuple[str, ...]] = ("adjusted_open",)
CANONICAL_MARKET_FIELDS: Final[tuple[str, ...]] = (
    *MARKET_FIELDS,
    *DERIVED_MARKET_FIELDS,
)
MARKET_LOOKBACK_ROW_COUNT: Final[int] = MARKET_LOOKBACK_SESSIONS + 1

MARKET_SOURCE_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-market-source-v1"
)
MARKET_STAGE_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-market-stage-v1"
)
MARKET_ROW_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-market-row-v1"
MARKET_PREFIX_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-market-prefix-v1"
MARKET_SOURCE_FAMILY: Final[str] = "sealed-local-adjusted-ohlcv-snapshot-v1"
MARKET_CUTOFF_RULE: Final[str] = "completed_decision_session_close_inclusive"
ADJUSTED_OPEN_DERIVATION: Final[str] = (
    "python_binary64_open_times_adjusted_close_div_close_left_associative_v1"
)
FILL_SESSION_OFFSET: Final[int] = 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_EVENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_RAW_OBSERVATION_KEYS = {"available", *MARKET_FIELDS}
_CANONICAL_OBSERVATION_KEYS = {
    "available",
    *(f"{name}_hex" for name in CANONICAL_MARKET_FIELDS),
}
_ROW_CHAIN_GENESIS_SHA256: Final[str] = hashlib.sha256(
    b"aapl-sec-gemma-market-row-chain-v1\x00"
).hexdigest()


class MarketEvidenceError(SecFilingGemmaContractError):
    """Raised when market evidence is incomplete, non-canonical, or unbound."""


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MarketEvidenceError("Market evidence must be finite canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise MarketEvidenceError(f"{location} must be a string-keyed mapping")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    observed = set(value)
    if observed != expected:
        raise MarketEvidenceError(
            f"Invalid {location} keys; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MarketEvidenceError(f"{location} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MarketEvidenceError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _iso_date(value: Any, location: str) -> date:
    if not isinstance(value, str):
        raise MarketEvidenceError(f"{location} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise MarketEvidenceError(f"{location} must be a canonical ISO date") from exc
    if parsed.isoformat() != value:
        raise MarketEvidenceError(f"{location} must use YYYY-MM-DD form")
    return parsed


def encode_float_hex(value: Any, location: str = "market value") -> str:
    """Return the sole permitted lossless representation of a market float."""

    if isinstance(value, bool) or not isinstance(value, float):
        raise MarketEvidenceError(f"{location} must be a float, not bool or decimal text")
    if not math.isfinite(value):
        raise MarketEvidenceError(f"{location} must be finite")
    return value.hex()


def decode_float_hex(value: Any, location: str = "market value") -> float:
    """Decode and verify a canonical, finite ``float.hex`` string."""

    if not isinstance(value, str):
        raise MarketEvidenceError(f"{location} must be canonical float.hex text")
    try:
        decoded = float.fromhex(value)
    except ValueError as exc:
        raise MarketEvidenceError(f"{location} is not float.hex text") from exc
    if not math.isfinite(decoded) or decoded.hex() != value:
        raise MarketEvidenceError(f"{location} is not canonical finite float.hex text")
    return decoded


def derive_adjusted_open_hex(
    *,
    open_hex: Any,
    close_hex: Any,
    adjusted_close_hex: Any,
    location: str = "market observation",
) -> str:
    """Derive adjusted open with the one frozen binary-float operation order.

    The caller never supplies adjusted open.  Execution and label ledgers use
    exactly ``open * adjusted_close / close`` in that order, encoded without a
    decimal round trip.
    """

    open_value = decode_float_hex(open_hex, f"{location}.open_hex")
    close_value = decode_float_hex(close_hex, f"{location}.close_hex")
    adjusted_close = decode_float_hex(
        adjusted_close_hex, f"{location}.adjusted_close_hex"
    )
    derived = open_value * adjusted_close / close_value
    if not math.isfinite(derived) or derived <= 0.0:
        raise MarketEvidenceError(f"{location}.adjusted_open is outside the permitted range")
    return derived.hex()


def _stage_market_window(stage: str) -> tuple[str, str]:
    if stage not in STAGE_ORDER:
        raise MarketEvidenceError("Market artifact stage is invalid")
    return MARKET_HISTORY_START, STAGE_WINDOWS[stage][1]


def _exact_calendar(session_dates: Sequence[str]) -> tuple[str, ...]:
    try:
        return canonical_market_session_calendar(session_dates)
    except SecFilingGemmaContractError as exc:
        raise MarketEvidenceError(str(exc)) from exc


def _window_sessions(stage: str, calendar: Sequence[str]) -> tuple[str, ...]:
    start, end = _stage_market_window(stage)
    return tuple(value for value in calendar if start <= value <= end)


def _normalize_available_sessions(
    values: Any,
    *,
    symbol: str,
    expected_window_sessions: tuple[str, ...],
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MarketEvidenceError(f"{symbol} available_sessions must be a sequence")
    normalized = [
        _iso_date(value, f"{symbol}.available_sessions[{index}]").isoformat()
        for index, value in enumerate(values)
    ]
    if normalized != sorted(normalized) or len(normalized) != len(set(normalized)):
        raise MarketEvidenceError(
            f"{symbol} available sessions must be sorted and unique"
        )
    permitted = set(expected_window_sessions)
    if any(value not in permitted for value in normalized):
        raise MarketEvidenceError(f"{symbol} has a source row outside the exact window")
    if symbol == "AAPL" and tuple(normalized) != expected_window_sessions:
        raise MarketEvidenceError("AAPL source must preserve every exact market session")
    return normalized


def build_market_source_manifest(
    *,
    artifact_stage: str,
    source_artifacts: Mapping[str, Mapping[str, Any]],
    session_dates: Sequence[str] = EXPECTED_MARKET_HISTORY_SESSIONS,
) -> dict[str, Any]:
    """Bind exact local source files and their exact cumulative stage slices.

    Each symbol specification has exactly five keys: ``artifact_sha256``,
    ``artifact_bytes``, ``window_sha256``, ``window_bytes``, and
    ``available_sessions``.  The latter is the complete set of source rows in
    the frozen cumulative stage window.  It is retained in the manifest so a
    context row cannot silently disappear behind a count.
    """

    calendar = _exact_calendar(session_dates)
    window_start, window_end = _stage_market_window(artifact_stage)
    expected_sessions = _window_sessions(artifact_stage, calendar)
    sources_value = _expect_mapping(source_artifacts, "source_artifacts")
    if set(sources_value) != set(MARKET_SYMBOLS):
        raise MarketEvidenceError("Source artifacts must contain exactly all six symbols")

    sources: list[dict[str, Any]] = []
    for symbol in MARKET_SYMBOLS:
        source = _expect_mapping(sources_value[symbol], f"source_artifacts.{symbol}")
        _expect_keys(
            source,
            {
                "artifact_sha256",
                "artifact_bytes",
                "window_sha256",
                "window_bytes",
                "available_sessions",
            },
            f"source_artifacts.{symbol}",
        )
        available = _normalize_available_sessions(
            source["available_sessions"],
            symbol=symbol,
            expected_window_sessions=expected_sessions,
        )
        sources.append(
            {
                "symbol": symbol,
                "artifact_sha256": _sha256(
                    source["artifact_sha256"], f"{symbol}.artifact_sha256"
                ),
                "artifact_bytes": _strict_int(
                    source["artifact_bytes"], f"{symbol}.artifact_bytes", minimum=1
                ),
                "window_sha256": _sha256(
                    source["window_sha256"], f"{symbol}.window_sha256"
                ),
                "window_bytes": _strict_int(
                    source["window_bytes"], f"{symbol}.window_bytes", minimum=1
                ),
                "available_session_count": len(available),
                "available_first_session": available[0] if available else None,
                "available_last_session": available[-1] if available else None,
                "available_sessions_sha256": _canonical_sha256(available),
                "available_sessions": available,
            }
        )
    body = {
        "schema_version": MARKET_SOURCE_MANIFEST_SCHEMA_VERSION,
        "source_family": MARKET_SOURCE_FAMILY,
        "artifact_stage": artifact_stage,
        "calendar_id": MARKET_HISTORY_CALENDAR_ID,
        "calendar_sha256": market_session_calendar_sha256(calendar),
        "window_start": window_start,
        "window_end": window_end,
        "window_session_count": len(expected_sessions),
        "window_sessions_sha256": _canonical_sha256(list(expected_sessions)),
        "sources": sources,
    }
    return {**body, "source_manifest_sha256": _canonical_sha256(body)}


def _source_specs_from_manifest(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sources = manifest.get("sources")
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise MarketEvidenceError("sources must be an ordered sequence")
    if len(sources) != len(MARKET_SYMBOLS):
        raise MarketEvidenceError("sources must contain exactly all six symbols")
    result: dict[str, dict[str, Any]] = {}
    for index, symbol in enumerate(MARKET_SYMBOLS):
        source = _expect_mapping(sources[index], f"sources[{index}]")
        _expect_keys(
            source,
            {
                "symbol",
                "artifact_sha256",
                "artifact_bytes",
                "window_sha256",
                "window_bytes",
                "available_session_count",
                "available_first_session",
                "available_last_session",
                "available_sessions_sha256",
                "available_sessions",
            },
            f"sources[{index}]",
        )
        if source["symbol"] != symbol:
            raise MarketEvidenceError("Market sources are missing, duplicated, or reordered")
        result[symbol] = dict(source)
    return result


def _validate_source_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_artifact_stage: str,
    expected_source_manifest_sha256: str,
    session_dates: Sequence[str],
    expected_source_artifact_sha256s: Mapping[str, str] | None = None,
    expected_source_window_sha256s: Mapping[str, str] | None = None,
) -> str:
    value = _expect_mapping(manifest, "market source manifest")
    _expect_keys(
        value,
        {
            "schema_version",
            "source_family",
            "artifact_stage",
            "calendar_id",
            "calendar_sha256",
            "window_start",
            "window_end",
            "window_session_count",
            "window_sessions_sha256",
            "sources",
            "source_manifest_sha256",
        },
        "market source manifest",
    )
    if value["schema_version"] != MARKET_SOURCE_MANIFEST_SCHEMA_VERSION:
        raise MarketEvidenceError("Market source schema changed")
    if value["source_family"] != MARKET_SOURCE_FAMILY:
        raise MarketEvidenceError("Market source family changed")
    if value["artifact_stage"] != expected_artifact_stage:
        raise MarketEvidenceError("Market source stage does not match the expected stage")
    calendar = _exact_calendar(session_dates)
    start, end = _stage_market_window(expected_artifact_stage)
    expected_sessions = _window_sessions(expected_artifact_stage, calendar)
    if (
        value["calendar_id"] != MARKET_HISTORY_CALENDAR_ID
        or value["calendar_sha256"] != market_session_calendar_sha256(calendar)
        or value["window_start"] != start
        or value["window_end"] != end
        or value["window_session_count"] != len(expected_sessions)
        or value["window_sessions_sha256"]
        != _canonical_sha256(list(expected_sessions))
    ):
        raise MarketEvidenceError("Market source calendar or exact stage window changed")

    source_specs = _source_specs_from_manifest(value)
    for symbol in MARKET_SYMBOLS:
        source = source_specs[symbol]
        _sha256(source["artifact_sha256"], f"{symbol}.artifact_sha256")
        _strict_int(source["artifact_bytes"], f"{symbol}.artifact_bytes", minimum=1)
        _sha256(source["window_sha256"], f"{symbol}.window_sha256")
        _strict_int(source["window_bytes"], f"{symbol}.window_bytes", minimum=1)
        available = _normalize_available_sessions(
            source["available_sessions"],
            symbol=symbol,
            expected_window_sessions=expected_sessions,
        )
        if (
            source["available_session_count"] != len(available)
            or source["available_first_session"]
            != (available[0] if available else None)
            or source["available_last_session"]
            != (available[-1] if available else None)
            or source["available_sessions_sha256"] != _canonical_sha256(available)
        ):
            raise MarketEvidenceError(f"{symbol} source availability evidence is inconsistent")

    if expected_source_artifact_sha256s is not None:
        expected = _expect_mapping(
            expected_source_artifact_sha256s,
            "expected_source_artifact_sha256s",
        )
        if set(expected) != set(MARKET_SYMBOLS) or any(
            source_specs[symbol]["artifact_sha256"]
            != _sha256(expected[symbol], f"expected artifact {symbol}")
            for symbol in MARKET_SYMBOLS
        ):
            raise MarketEvidenceError("Market source artifact hashes are not externally pinned")
    if expected_source_window_sha256s is not None:
        expected = _expect_mapping(
            expected_source_window_sha256s,
            "expected_source_window_sha256s",
        )
        if set(expected) != set(MARKET_SYMBOLS) or any(
            source_specs[symbol]["window_sha256"]
            != _sha256(expected[symbol], f"expected window {symbol}")
            for symbol in MARKET_SYMBOLS
        ):
            raise MarketEvidenceError("Market source window hashes are not externally pinned")

    body = {key: value[key] for key in value if key != "source_manifest_sha256"}
    calculated = _canonical_sha256(body)
    observed = _sha256(value["source_manifest_sha256"], "source_manifest_sha256")
    if not hmac.compare_digest(observed, calculated):
        raise MarketEvidenceError("Market source manifest hash is not canonical")
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_source_manifest_sha256,
            "expected_source_manifest_sha256",
        ),
    ):
        raise MarketEvidenceError("Market source manifest is not externally pinned")
    return observed


def validate_market_source_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_artifact_stage: str,
    expected_source_manifest_sha256: str,
    expected_source_artifact_sha256s: Mapping[str, str],
    expected_source_window_sha256s: Mapping[str, str],
    session_dates: Sequence[str] = EXPECTED_MARKET_HISTORY_SESSIONS,
) -> str:
    """Validate source identity, stage/window, and two external hash pins."""

    return _validate_source_manifest(
        manifest,
        expected_artifact_stage=expected_artifact_stage,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        session_dates=session_dates,
        expected_source_artifact_sha256s=expected_source_artifact_sha256s,
        expected_source_window_sha256s=expected_source_window_sha256s,
    )


def _validate_market_values(values: Mapping[str, float], location: str) -> None:
    if any(values[name] <= 0.0 for name in PRICE_FIELDS):
        raise MarketEvidenceError(f"{location} prices must be positive")
    if values["volume"] < 0.0:
        raise MarketEvidenceError(f"{location} volume cannot be negative")
    if values["high"] < max(values["open"], values["low"], values["close"]):
        raise MarketEvidenceError(f"{location} high is inconsistent with OHLC")
    if values["low"] > min(values["open"], values["high"], values["close"]):
        raise MarketEvidenceError(f"{location} low is inconsistent with OHLC")


def _canonicalize_raw_observation(value: Any, location: str) -> dict[str, Any]:
    observed = _expect_mapping(value, location)
    _expect_keys(observed, _RAW_OBSERVATION_KEYS, location)
    available = observed["available"]
    if not isinstance(available, bool):
        raise MarketEvidenceError(f"{location}.available must be a boolean")
    if not available:
        if any(observed[name] is not None for name in MARKET_FIELDS):
            raise MarketEvidenceError(
                f"{location} unavailable row must use explicit nulls for every field"
            )
        return {
            "available": False,
            **{f"{name}_hex": None for name in CANONICAL_MARKET_FIELDS},
        }
    numeric = {
        name: float.fromhex(encode_float_hex(observed[name], f"{location}.{name}"))
        for name in MARKET_FIELDS
    }
    _validate_market_values(numeric, location)
    encoded = {
        "available": True,
        **{
            f"{name}_hex": encode_float_hex(observed[name], f"{location}.{name}")
            for name in MARKET_FIELDS
        },
    }
    encoded["adjusted_open_hex"] = derive_adjusted_open_hex(
        open_hex=encoded["open_hex"],
        close_hex=encoded["close_hex"],
        adjusted_close_hex=encoded["adjusted_close_hex"],
        location=location,
    )
    return encoded


def _validate_canonical_observation(value: Any, location: str) -> dict[str, Any]:
    observed = _expect_mapping(value, location)
    _expect_keys(observed, _CANONICAL_OBSERVATION_KEYS, location)
    available = observed["available"]
    if not isinstance(available, bool):
        raise MarketEvidenceError(f"{location}.available must be a boolean")
    if not available:
        if any(
            observed[f"{name}_hex"] is not None
            for name in CANONICAL_MARKET_FIELDS
        ):
            raise MarketEvidenceError(
                f"{location} unavailable row must retain explicit nulls"
            )
        return dict(observed)
    numeric = {
        name: decode_float_hex(observed[f"{name}_hex"], f"{location}.{name}_hex")
        for name in MARKET_FIELDS
    }
    _validate_market_values(numeric, location)
    expected_adjusted_open = derive_adjusted_open_hex(
        open_hex=observed["open_hex"],
        close_hex=observed["close_hex"],
        adjusted_close_hex=observed["adjusted_close_hex"],
        location=location,
    )
    adjusted_open = observed["adjusted_open_hex"]
    decode_float_hex(adjusted_open, f"{location}.adjusted_open_hex")
    if adjusted_open != expected_adjusted_open:
        raise MarketEvidenceError(
            f"{location}.adjusted_open_hex is not the exact derived adjusted open"
        )
    return dict(observed)


def _canonicalize_raw_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_sessions: tuple[str, ...],
) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise MarketEvidenceError("Market rows must be an ordered sequence")
    if len(rows) != len(expected_sessions):
        raise MarketEvidenceError(
            "Market rows are missing or contain extra/duplicate AAPL sessions"
        )
    canonical: list[dict[str, Any]] = []
    previous_hash = _ROW_CHAIN_GENESIS_SHA256
    for index, expected_session in enumerate(expected_sessions):
        row = _expect_mapping(rows[index], f"rows[{index}]")
        _expect_keys(row, {"session", "observations"}, f"rows[{index}]")
        session = _iso_date(row["session"], f"rows[{index}].session").isoformat()
        if session != expected_session:
            raise MarketEvidenceError(
                "AAPL sessions are missing, duplicated, reordered, or outside the window"
            )
        observations = _expect_mapping(
            row["observations"], f"rows[{index}].observations"
        )
        if set(observations) != set(MARKET_SYMBOLS):
            raise MarketEvidenceError(
                "Every AAPL session must explicitly contain all context symbols"
            )
        canonical_observations = {
            symbol: _canonicalize_raw_observation(
                observations[symbol], f"rows[{index}].observations.{symbol}"
            )
            for symbol in MARKET_SYMBOLS
        }
        if canonical_observations["AAPL"]["available"] is not True:
            raise MarketEvidenceError("AAPL must be available on every exact market session")
        body = {
            "schema_version": MARKET_ROW_SCHEMA_VERSION,
            "row_index": index,
            "session": session,
            "observations": canonical_observations,
            "previous_row_sha256": previous_hash,
        }
        row_hash = _canonical_sha256(body)
        canonical.append({**body, "row_sha256": row_hash})
        previous_hash = row_hash
    return canonical


def _validate_canonical_rows(
    rows: Any,
    *,
    expected_sessions: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise MarketEvidenceError("Canonical market rows must be an ordered sequence")
    if len(rows) != len(expected_sessions):
        raise MarketEvidenceError(
            "Canonical market rows are missing or contain extra/duplicate AAPL sessions"
        )
    normalized: list[dict[str, Any]] = []
    available_by_symbol: dict[str, list[str]] = {symbol: [] for symbol in MARKET_SYMBOLS}
    previous_hash = _ROW_CHAIN_GENESIS_SHA256
    expected_keys = {
        "schema_version",
        "row_index",
        "session",
        "observations",
        "previous_row_sha256",
        "row_sha256",
    }
    for index, expected_session in enumerate(expected_sessions):
        row = _expect_mapping(rows[index], f"rows[{index}]")
        _expect_keys(row, expected_keys, f"rows[{index}]")
        if row["schema_version"] != MARKET_ROW_SCHEMA_VERSION:
            raise MarketEvidenceError("Market row schema changed")
        if _strict_int(row["row_index"], f"rows[{index}].row_index") != index:
            raise MarketEvidenceError("Market row indices are missing, duplicated, or reordered")
        session = _iso_date(row["session"], f"rows[{index}].session").isoformat()
        if session != expected_session:
            raise MarketEvidenceError(
                "AAPL sessions are missing, duplicated, reordered, or outside the window"
            )
        observations = _expect_mapping(
            row["observations"], f"rows[{index}].observations"
        )
        if set(observations) != set(MARKET_SYMBOLS):
            raise MarketEvidenceError(
                "Every AAPL session must explicitly contain all context symbols"
            )
        canonical_observations = {
            symbol: _validate_canonical_observation(
                observations[symbol], f"rows[{index}].observations.{symbol}"
            )
            for symbol in MARKET_SYMBOLS
        }
        if canonical_observations["AAPL"]["available"] is not True:
            raise MarketEvidenceError("AAPL must be available on every exact market session")
        for symbol in MARKET_SYMBOLS:
            if canonical_observations[symbol]["available"]:
                available_by_symbol[symbol].append(session)
        if row["previous_row_sha256"] != previous_hash:
            raise MarketEvidenceError("Market row hash chain is broken or reordered")
        body = {
            "schema_version": MARKET_ROW_SCHEMA_VERSION,
            "row_index": index,
            "session": session,
            "observations": canonical_observations,
            "previous_row_sha256": previous_hash,
        }
        calculated = _canonical_sha256(body)
        if not hmac.compare_digest(
            calculated, _sha256(row["row_sha256"], f"rows[{index}].row_sha256")
        ):
            raise MarketEvidenceError(
                "Market row hash changed, including a sub-decimal float mutation"
            )
        normalized.append({**body, "row_sha256": calculated})
        previous_hash = calculated
    return normalized, available_by_symbol


def _stage_body(
    *,
    artifact_stage: str,
    source_manifest_sha256: str,
    calendar_sha256: str,
    window_start: str,
    window_end: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    frames = {
        symbol: [
            {"session": row["session"], "observation": row["observations"][symbol]}
            for row in rows
        ]
        for symbol in MARKET_SYMBOLS
    }
    row_hashes = [row["row_sha256"] for row in rows]
    return {
        "schema_version": MARKET_STAGE_MANIFEST_SCHEMA_VERSION,
        "artifact_stage": artifact_stage,
        "calendar_id": MARKET_HISTORY_CALENDAR_ID,
        "calendar_sha256": calendar_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "market_window_start": window_start,
        "market_window_end": window_end,
        "score_window_start": STAGE_WINDOWS[artifact_stage][0],
        "score_window_end": STAGE_WINDOWS[artifact_stage][1],
        "row_schema_version": MARKET_ROW_SCHEMA_VERSION,
        "numeric_encoding": "python_float_hex_exact_finite",
        "adjusted_open_derivation": ADJUSTED_OPEN_DERIVATION,
        "missing_context_encoding": "available_false_and_all_value_hex_null",
        "source_value_reconciliation": (
            "authoritative_stage_verifier_must_parse_exact_source_window_bytes"
        ),
        "row_chain_genesis_sha256": _ROW_CHAIN_GENESIS_SHA256,
        "row_count": len(rows),
        "first_session": rows[0]["session"],
        "last_session": rows[-1]["session"],
        "row_chain_tip_sha256": row_hashes[-1],
        "row_hashes_sha256": _canonical_sha256(row_hashes),
        "symbol_frame_sha256s": {
            symbol: _canonical_sha256(frames[symbol]) for symbol in MARKET_SYMBOLS
        },
        "rows": rows,
    }


def _reconcile_rows_to_sources(
    *,
    available_by_symbol: Mapping[str, list[str]],
    source_manifest: Mapping[str, Any],
) -> None:
    specs = _source_specs_from_manifest(source_manifest)
    for symbol in MARKET_SYMBOLS:
        available = available_by_symbol[symbol]
        source = specs[symbol]
        if (
            available != source["available_sessions"]
            or len(available) != source["available_session_count"]
            or _canonical_sha256(available) != source["available_sessions_sha256"]
        ):
            raise MarketEvidenceError(
                f"{symbol} source rows were silently dropped, inserted, or shifted"
            )


def build_market_stage_manifest(
    *,
    artifact_stage: str,
    source_manifest: Mapping[str, Any],
    expected_source_manifest_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    session_dates: Sequence[str] = EXPECTED_MARKET_HISTORY_SESSIONS,
) -> dict[str, Any]:
    """Build a complete cumulative stage frame and its per-session hash chain."""

    calendar = _exact_calendar(session_dates)
    source_hash = _validate_source_manifest(
        source_manifest,
        expected_artifact_stage=artifact_stage,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        session_dates=calendar,
    )
    expected_sessions = _window_sessions(artifact_stage, calendar)
    canonical_rows = _canonicalize_raw_rows(rows, expected_sessions=expected_sessions)
    _, availability = _validate_canonical_rows(
        canonical_rows, expected_sessions=expected_sessions
    )
    _reconcile_rows_to_sources(
        available_by_symbol=availability, source_manifest=source_manifest
    )
    window_start, window_end = _stage_market_window(artifact_stage)
    body = _stage_body(
        artifact_stage=artifact_stage,
        source_manifest_sha256=source_hash,
        calendar_sha256=market_session_calendar_sha256(calendar),
        window_start=window_start,
        window_end=window_end,
        rows=canonical_rows,
    )
    return {**body, "market_stage_manifest_sha256": _canonical_sha256(body)}


def validate_market_stage_manifest(
    manifest: Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any],
    expected_artifact_stage: str,
    expected_source_manifest_sha256: str,
    expected_market_stage_manifest_sha256: str,
    session_dates: Sequence[str] = EXPECTED_MARKET_HISTORY_SESSIONS,
) -> str:
    """Replay the frame, source coverage, row chain, and external hash pins.

    This pure validator cannot derive OHLCV values from opaque source-file
    hashes.  The authoritative stage verifier must parse the exact pinned
    source-window bytes and compare every canonical value before promotion.
    """

    value = _expect_mapping(manifest, "market stage manifest")
    expected_keys = {
        "schema_version",
        "artifact_stage",
        "calendar_id",
        "calendar_sha256",
        "source_manifest_sha256",
        "market_window_start",
        "market_window_end",
        "score_window_start",
        "score_window_end",
        "row_schema_version",
        "numeric_encoding",
        "adjusted_open_derivation",
        "missing_context_encoding",
        "source_value_reconciliation",
        "row_chain_genesis_sha256",
        "row_count",
        "first_session",
        "last_session",
        "row_chain_tip_sha256",
        "row_hashes_sha256",
        "symbol_frame_sha256s",
        "rows",
        "market_stage_manifest_sha256",
    }
    _expect_keys(value, expected_keys, "market stage manifest")
    if value["schema_version"] != MARKET_STAGE_MANIFEST_SCHEMA_VERSION:
        raise MarketEvidenceError("Market stage schema changed")
    if value["artifact_stage"] != expected_artifact_stage:
        raise MarketEvidenceError("Market stage does not match the expected stage")
    calendar = _exact_calendar(session_dates)
    source_hash = _validate_source_manifest(
        source_manifest,
        expected_artifact_stage=expected_artifact_stage,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        session_dates=calendar,
    )
    expected_sessions = _window_sessions(expected_artifact_stage, calendar)
    rows, availability = _validate_canonical_rows(
        value["rows"], expected_sessions=expected_sessions
    )
    _reconcile_rows_to_sources(
        available_by_symbol=availability, source_manifest=source_manifest
    )
    start, end = _stage_market_window(expected_artifact_stage)
    body = _stage_body(
        artifact_stage=expected_artifact_stage,
        source_manifest_sha256=source_hash,
        calendar_sha256=market_session_calendar_sha256(calendar),
        window_start=start,
        window_end=end,
        rows=rows,
    )
    if any(value[key] != body[key] for key in body):
        raise MarketEvidenceError("Market stage summary, window, or row identities changed")
    calculated = _canonical_sha256(body)
    observed = _sha256(
        value["market_stage_manifest_sha256"], "market_stage_manifest_sha256"
    )
    if not hmac.compare_digest(observed, calculated):
        raise MarketEvidenceError("Market stage manifest hash is not canonical")
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_market_stage_manifest_sha256,
            "expected_market_stage_manifest_sha256",
        ),
    ):
        raise MarketEvidenceError("Market stage manifest is not externally pinned")
    return observed


def _offset_session(
    session: str,
    offset: int,
    *,
    calendar: tuple[str, ...],
    location: str,
) -> str:
    try:
        index = calendar.index(session)
    except ValueError as exc:
        raise MarketEvidenceError(f"{location} is not in the exact contract calendar") from exc
    target = index + offset
    if target >= len(calendar):
        raise MarketEvidenceError(
            f"{location} lacks its exact +{offset} contract-calendar session"
        )
    return calendar[target]


def _prefix_chain_identity(*, row_count: int, row_chain_tip_sha256: str) -> str:
    return _canonical_sha256(
        {
            "domain": "aapl-sec-gemma-market-prefix-chain-v1",
            "row_chain_genesis_sha256": _ROW_CHAIN_GENESIS_SHA256,
            "full_prefix_row_count": row_count,
            "full_prefix_row_chain_tip_sha256": row_chain_tip_sha256,
        }
    )


def _lookback_slice_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_sha256(
        {
            "domain": "aapl-sec-gemma-market-lookback-slice-v1",
            "first_row_index": rows[0]["row_index"],
            "last_row_index": rows[-1]["row_index"],
            "row_count": len(rows),
            "start_parent_row_sha256": rows[0]["previous_row_sha256"],
            "row_hashes_sha256": _canonical_sha256(
                [row["row_sha256"] for row in rows]
            ),
            "row_chain_tip_sha256": rows[-1]["row_sha256"],
        }
    )


def _prefix_body(
    *,
    stage_manifest: Mapping[str, Any],
    decision_event_id: str,
    decision_session: str,
    calendar: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(decision_event_id, str) or _EVENT_ID_RE.fullmatch(
        decision_event_id
    ) is None:
        raise MarketEvidenceError("decision_event_id is not canonical")
    stage = stage_manifest["artifact_stage"]
    score_start, score_end = STAGE_WINDOWS[stage]
    if not (score_start <= decision_session <= score_end):
        raise MarketEvidenceError("Decision session is outside the exact stage score window")
    sessions = [row["session"] for row in stage_manifest["rows"]]
    try:
        decision_index = sessions.index(decision_session)
    except ValueError as exc:
        raise MarketEvidenceError("Decision session is not an exact AAPL session") from exc
    full_prefix_row_count = decision_index + 1
    if full_prefix_row_count < MARKET_LOOKBACK_ROW_COUNT:
        raise MarketEvidenceError(
            "Decision session lacks the exact 252-session market prehistory"
        )
    lookback_start = full_prefix_row_count - MARKET_LOOKBACK_ROW_COUNT
    # Only t-252 through t is materialized.  The row-chain tip still commits to
    # every row from the 1998 genesis through t, without copying that history
    # into every quarterly prediction receipt.  Deep-copy the compact slice so
    # caller mutation cannot rewrite the authoritative stage object in memory.
    lookback_rows = copy.deepcopy(
        list(stage_manifest["rows"][lookback_start:full_prefix_row_count])
    )
    full_prefix_tip = stage_manifest["rows"][decision_index]["row_sha256"]
    fill = _offset_session(
        decision_session,
        FILL_SESSION_OFFSET,
        calendar=calendar,
        location="decision_session",
    )
    maturity = _offset_session(
        decision_session,
        LABEL_MATURITY_OFFSET,
        calendar=calendar,
        location="decision_session",
    )
    return {
        "schema_version": MARKET_PREFIX_SCHEMA_VERSION,
        "artifact_stage": stage,
        "decision_event_id": decision_event_id,
        "market_stage_manifest_sha256": stage_manifest[
            "market_stage_manifest_sha256"
        ],
        "source_manifest_sha256": stage_manifest["source_manifest_sha256"],
        "calendar_id": MARKET_HISTORY_CALENDAR_ID,
        "calendar_sha256": stage_manifest["calendar_sha256"],
        "market_cutoff_rule": MARKET_CUTOFF_RULE,
        "decision_session": decision_session,
        "market_cutoff_session": decision_session,
        "fill_session_offset": FILL_SESSION_OFFSET,
        "fill_session": fill,
        "label_maturity_session_offset": LABEL_MATURITY_OFFSET,
        "label_maturity_session": maturity,
        "full_prefix_first_session": stage_manifest["rows"][0]["session"],
        "full_prefix_last_session": decision_session,
        "full_prefix_row_count": full_prefix_row_count,
        "full_prefix_row_chain_tip_sha256": full_prefix_tip,
        "full_prefix_chain_identity_sha256": _prefix_chain_identity(
            row_count=full_prefix_row_count,
            row_chain_tip_sha256=full_prefix_tip,
        ),
        "maximum_feature_lookback_sessions": MARKET_LOOKBACK_SESSIONS,
        "lookback_start_row_index": lookback_rows[0]["row_index"],
        "lookback_end_row_index": lookback_rows[-1]["row_index"],
        "lookback_first_session": lookback_rows[0]["session"],
        "lookback_last_session": lookback_rows[-1]["session"],
        "lookback_row_count": len(lookback_rows),
        "lookback_start_parent_row_sha256": lookback_rows[0][
            "previous_row_sha256"
        ],
        "lookback_row_chain_tip_sha256": lookback_rows[-1]["row_sha256"],
        "lookback_row_hashes_sha256": _canonical_sha256(
            [row["row_sha256"] for row in lookback_rows]
        ),
        "lookback_rows_sha256": _canonical_sha256(lookback_rows),
        "lookback_slice_identity_sha256": _lookback_slice_identity(lookback_rows),
        "lookback_rows": lookback_rows,
    }


def build_decision_market_prefix(
    *,
    stage_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    expected_artifact_stage: str,
    expected_source_manifest_sha256: str,
    expected_market_stage_manifest_sha256: str,
    decision_event_id: str,
    decision_session: str,
    session_dates: Sequence[str] = EXPECTED_MARKET_HISTORY_SESSIONS,
) -> dict[str, Any]:
    """Create the only market payload a decision at session ``t`` may see."""

    calendar = _exact_calendar(session_dates)
    validate_market_stage_manifest(
        stage_manifest,
        source_manifest=source_manifest,
        expected_artifact_stage=expected_artifact_stage,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_market_stage_manifest_sha256=expected_market_stage_manifest_sha256,
        session_dates=calendar,
    )
    decision = _iso_date(decision_session, "decision_session").isoformat()
    body = _prefix_body(
        stage_manifest=stage_manifest,
        decision_event_id=decision_event_id,
        decision_session=decision,
        calendar=calendar,
    )
    return {**body, "market_prefix_sha256": _canonical_sha256(body)}


def validate_decision_market_prefix(
    prefix: Mapping[str, Any],
    *,
    stage_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    expected_artifact_stage: str,
    expected_source_manifest_sha256: str,
    expected_market_stage_manifest_sha256: str,
    expected_decision_event_id: str,
    expected_decision_session: str,
    expected_market_prefix_sha256: str,
    session_dates: Sequence[str] = EXPECTED_MARKET_HISTORY_SESSIONS,
) -> str:
    """Reject future rows and replay the exact fill/maturity calendar binding."""

    value = _expect_mapping(prefix, "decision market prefix")
    expected_keys = {
        "schema_version",
        "artifact_stage",
        "decision_event_id",
        "market_stage_manifest_sha256",
        "source_manifest_sha256",
        "calendar_id",
        "calendar_sha256",
        "market_cutoff_rule",
        "decision_session",
        "market_cutoff_session",
        "fill_session_offset",
        "fill_session",
        "label_maturity_session_offset",
        "label_maturity_session",
        "full_prefix_first_session",
        "full_prefix_last_session",
        "full_prefix_row_count",
        "full_prefix_row_chain_tip_sha256",
        "full_prefix_chain_identity_sha256",
        "maximum_feature_lookback_sessions",
        "lookback_start_row_index",
        "lookback_end_row_index",
        "lookback_first_session",
        "lookback_last_session",
        "lookback_row_count",
        "lookback_start_parent_row_sha256",
        "lookback_row_chain_tip_sha256",
        "lookback_row_hashes_sha256",
        "lookback_rows_sha256",
        "lookback_slice_identity_sha256",
        "lookback_rows",
        "market_prefix_sha256",
    }
    _expect_keys(value, expected_keys, "decision market prefix")
    if value["schema_version"] != MARKET_PREFIX_SCHEMA_VERSION:
        raise MarketEvidenceError("Decision market prefix schema changed")
    calendar = _exact_calendar(session_dates)
    validate_market_stage_manifest(
        stage_manifest,
        source_manifest=source_manifest,
        expected_artifact_stage=expected_artifact_stage,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_market_stage_manifest_sha256=expected_market_stage_manifest_sha256,
        session_dates=calendar,
    )
    expected_decision = _iso_date(
        expected_decision_session, "expected_decision_session"
    ).isoformat()
    if value["decision_event_id"] != expected_decision_event_id:
        raise MarketEvidenceError("Decision event identity changed")
    if value["decision_session"] != expected_decision:
        raise MarketEvidenceError("Decision session changed")
    expected_body = _prefix_body(
        stage_manifest=stage_manifest,
        decision_event_id=expected_decision_event_id,
        decision_session=expected_decision,
        calendar=calendar,
    )
    prefix_rows = value["lookback_rows"]
    if isinstance(prefix_rows, Sequence) and not isinstance(prefix_rows, (str, bytes)):
        if prefix_rows and isinstance(prefix_rows[-1], Mapping):
            last = prefix_rows[-1].get("session")
            if isinstance(last, str) and last > expected_decision:
                raise MarketEvidenceError("Decision market prefix contains a later row")
    if any(value[key] != expected_body[key] for key in expected_body):
        raise MarketEvidenceError(
            "Decision market prefix is incomplete, reordered, mutated, or contains future rows"
        )
    calculated = _canonical_sha256(expected_body)
    observed = _sha256(value["market_prefix_sha256"], "market_prefix_sha256")
    if not hmac.compare_digest(observed, calculated):
        raise MarketEvidenceError("Decision market prefix hash is not canonical")
    if not hmac.compare_digest(
        observed,
        _sha256(expected_market_prefix_sha256, "expected_market_prefix_sha256"),
    ):
        raise MarketEvidenceError("Decision market prefix is not externally pinned")
    return observed


def validate_market_stage_extension(
    *,
    prior_stage_manifest: Mapping[str, Any],
    prior_source_manifest: Mapping[str, Any],
    expected_prior_stage: str,
    expected_prior_source_manifest_sha256: str,
    expected_prior_market_stage_manifest_sha256: str,
    extended_stage_manifest: Mapping[str, Any],
    extended_source_manifest: Mapping[str, Any],
    expected_extended_stage: str,
    expected_extended_source_manifest_sha256: str,
    expected_extended_market_stage_manifest_sha256: str,
    session_dates: Sequence[str] = EXPECTED_MARKET_HISTORY_SESSIONS,
) -> str:
    """Prove that a later stage only appends rows to the earlier row chain."""

    calendar = _exact_calendar(session_dates)
    validate_market_stage_manifest(
        prior_stage_manifest,
        source_manifest=prior_source_manifest,
        expected_artifact_stage=expected_prior_stage,
        expected_source_manifest_sha256=expected_prior_source_manifest_sha256,
        expected_market_stage_manifest_sha256=expected_prior_market_stage_manifest_sha256,
        session_dates=calendar,
    )
    validate_market_stage_manifest(
        extended_stage_manifest,
        source_manifest=extended_source_manifest,
        expected_artifact_stage=expected_extended_stage,
        expected_source_manifest_sha256=expected_extended_source_manifest_sha256,
        expected_market_stage_manifest_sha256=expected_extended_market_stage_manifest_sha256,
        session_dates=calendar,
    )
    if STAGE_ORDER.index(expected_extended_stage) != STAGE_ORDER.index(
        expected_prior_stage
    ) + 1:
        raise MarketEvidenceError("Market stage extension must advance exactly once")
    prior_rows = prior_stage_manifest["rows"]
    extended_rows = extended_stage_manifest["rows"]
    if len(extended_rows) <= len(prior_rows) or extended_rows[: len(prior_rows)] != prior_rows:
        raise MarketEvidenceError(
            "Later market stage rewrote or failed to preserve the complete row-chain prefix"
        )
    if extended_rows[len(prior_rows)]["previous_row_sha256"] != prior_stage_manifest[
        "row_chain_tip_sha256"
    ]:
        raise MarketEvidenceError("Later market stage is not chained to the prior tip")
    return prior_stage_manifest["row_chain_tip_sha256"]


__all__ = [
    "ADJUSTED_OPEN_DERIVATION",
    "CANONICAL_MARKET_FIELDS",
    "CONTEXT_SYMBOLS",
    "DERIVED_MARKET_FIELDS",
    "FILL_SESSION_OFFSET",
    "MARKET_CUTOFF_RULE",
    "MARKET_FIELDS",
    "MARKET_LOOKBACK_ROW_COUNT",
    "MARKET_PREFIX_SCHEMA_VERSION",
    "MARKET_ROW_SCHEMA_VERSION",
    "MARKET_SOURCE_FAMILY",
    "MARKET_SOURCE_MANIFEST_SCHEMA_VERSION",
    "MARKET_STAGE_MANIFEST_SCHEMA_VERSION",
    "MARKET_SYMBOLS",
    "MarketEvidenceError",
    "build_decision_market_prefix",
    "build_market_source_manifest",
    "build_market_stage_manifest",
    "decode_float_hex",
    "derive_adjusted_open_hex",
    "encode_float_hex",
    "validate_decision_market_prefix",
    "validate_market_source_manifest",
    "validate_market_stage_extension",
    "validate_market_stage_manifest",
]
