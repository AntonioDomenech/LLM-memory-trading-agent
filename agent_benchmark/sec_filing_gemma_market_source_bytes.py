"""Exact byte reconciliation for SEC/Gemma market-source snapshots.

The structural market-evidence module deliberately cannot prove that its
canonical rows came from the sealed source bytes.  This module closes that
boundary with one frozen canonical snapshot format.  It is pure: callers pass
detached bytes, and every byte, row, float, session, source manifest, and stage
row is replayed before a reconciliation receipt is returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import re
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_contract import canonical_sha256
from agent_benchmark.sec_filing_gemma_market_evidence import (
    MARKET_FIELDS,
    MARKET_SYMBOLS,
    MarketEvidenceError,
    decode_float_hex,
    encode_float_hex,
    validate_market_source_manifest,
    validate_market_stage_manifest,
)
from agent_benchmark.sec_session_calendar import EXPECTED_MARKET_HISTORY_SESSIONS


MARKET_SOURCE_SNAPSHOT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-market-source-snapshot-v1"
)
MARKET_SOURCE_RECONCILIATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-market-source-reconciliation-v1"
)
MAX_MARKET_SOURCE_SNAPSHOT_BYTES: Final[int] = 512 * 1024 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ROW_KEYS = {"session", *(f"{field}_hex" for field in MARKET_FIELDS)}


class MarketSourceBytesError(MarketEvidenceError):
    """Raised when sealed market bytes cannot produce the canonical stage."""


def _detach_plain_json(value: Any, location: str) -> Any:
    """Copy only built-in JSON containers; reject switching Mapping views."""

    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str or key in result:
                raise MarketSourceBytesError(
                    f"{location} must have unique built-in string keys"
                )
            result[key] = _detach_plain_json(child, f"{location}.{key}")
        return result
    if type(value) is list:
        return [
            _detach_plain_json(child, f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    if value is None or type(value) in {str, bool, int, float}:
        return value
    raise MarketSourceBytesError(
        f"{location} must be detached built-in finite JSON"
    )


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MarketSourceBytesError(f"{location} must be a lowercase SHA-256")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    if not all(isinstance(key, str) for key in value) or set(value) != expected:
        raise MarketSourceBytesError(f"{location} has unexpected or missing keys")


def _reject_duplicate_pairs(location: str):
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MarketSourceBytesError(
                    f"{location} contains a duplicate JSON key"
                )
            result[key] = value
        return result

    return hook


def _strict_json_bytes(payload: bytes, location: str) -> Mapping[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError(f"{location} must be exact bytes")
    if not payload or len(payload) > MAX_MARKET_SOURCE_SNAPSHOT_BYTES:
        raise MarketSourceBytesError(f"{location} is empty or oversized")
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs(location),
            parse_constant=lambda token: (_ for _ in ()).throw(
                MarketSourceBytesError(
                    f"{location} contains a non-finite JSON constant: {token}"
                )
            ),
        )
    except MarketSourceBytesError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MarketSourceBytesError(f"{location} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise MarketSourceBytesError(f"{location} must contain one JSON object")
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_snapshot_body(
    *, symbol: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if symbol not in MARKET_SYMBOLS:
        raise MarketSourceBytesError("Market snapshot symbol is not frozen")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise MarketSourceBytesError("Market snapshot rows must be a sequence")
    normalized: list[dict[str, str]] = []
    previous: str | None = None
    permitted_sessions = set(EXPECTED_MARKET_HISTORY_SESSIONS)
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise MarketSourceBytesError(f"rows[{index}] must be an object")
        _expect_keys(raw, _ROW_KEYS, f"rows[{index}]")
        session = raw["session"]
        if not isinstance(session, str) or session not in permitted_sessions:
            raise MarketSourceBytesError(
                f"rows[{index}] session is outside the frozen market calendar"
            )
        if previous is not None and session <= previous:
            raise MarketSourceBytesError(
                "Market snapshot sessions are duplicated or reordered"
            )
        previous = session
        row: dict[str, str] = {"session": session}
        for field in MARKET_FIELDS:
            encoded = raw[f"{field}_hex"]
            decoded = decode_float_hex(encoded, f"rows[{index}].{field}_hex")
            canonical = encode_float_hex(decoded, f"rows[{index}].{field}")
            if encoded != canonical:
                raise MarketSourceBytesError(
                    f"rows[{index}].{field}_hex is not canonical float.hex"
                )
            row[f"{field}_hex"] = canonical
        normalized.append(row)
    body = {
        "schema_version": MARKET_SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "symbol": symbol,
        "row_count": len(normalized),
        "first_session": normalized[0]["session"] if normalized else None,
        "last_session": normalized[-1]["session"] if normalized else None,
        "rows_sha256": canonical_sha256(normalized),
        "rows": normalized,
    }
    return body


def build_market_source_snapshot_bytes(
    *, symbol: str, rows: Sequence[Mapping[str, Any]]
) -> bytes:
    """Build the only accepted canonical local market-source byte format."""

    body = _canonical_snapshot_body(symbol=symbol, rows=rows)
    snapshot = {**body, "snapshot_sha256": canonical_sha256(body)}
    return _canonical_bytes(snapshot)


def _parse_snapshot(payload: bytes, *, expected_symbol: str, location: str) -> dict[str, Any]:
    value = _strict_json_bytes(payload, location)
    _expect_keys(
        value,
        {
            "schema_version",
            "symbol",
            "row_count",
            "first_session",
            "last_session",
            "rows_sha256",
            "rows",
            "snapshot_sha256",
        },
        location,
    )
    if value["schema_version"] != MARKET_SOURCE_SNAPSHOT_SCHEMA_VERSION:
        raise MarketSourceBytesError(f"{location} schema changed")
    if value["symbol"] != expected_symbol:
        raise MarketSourceBytesError(f"{location} symbol changed")
    rows = value["rows"]
    body = _canonical_snapshot_body(symbol=expected_symbol, rows=rows)
    if any(value[key] != body[key] for key in body):
        raise MarketSourceBytesError(f"{location} summary or row identity changed")
    observed = _sha256(value["snapshot_sha256"], f"{location}.snapshot_sha256")
    if not hmac.compare_digest(observed, canonical_sha256(body)):
        raise MarketSourceBytesError(f"{location} self-hash changed")
    expected_bytes = _canonical_bytes({**body, "snapshot_sha256": observed})
    if payload != expected_bytes:
        raise MarketSourceBytesError(f"{location} bytes are not canonical")
    return {**body, "snapshot_sha256": observed}


def _source_specs(source_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    sources = source_manifest.get("sources")
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise MarketSourceBytesError("Market source manifest sources are invalid")
    result: dict[str, Mapping[str, Any]] = {}
    for index, symbol in enumerate(MARKET_SYMBOLS):
        if index >= len(sources) or not isinstance(sources[index], Mapping):
            raise MarketSourceBytesError("Market source manifest is incomplete")
        item = sources[index]
        if item.get("symbol") != symbol:
            raise MarketSourceBytesError("Market source manifest order changed")
        result[symbol] = item
    if len(sources) != len(MARKET_SYMBOLS):
        raise MarketSourceBytesError("Market source manifest has extra symbols")
    return result


def validate_market_source_bytes_against_stage(
    *,
    artifact_bytes_by_symbol: Mapping[str, bytes],
    window_bytes_by_symbol: Mapping[str, bytes],
    source_manifest: Mapping[str, Any],
    stage_manifest: Mapping[str, Any],
    expected_artifact_stage: str,
    expected_source_manifest_sha256: str,
    expected_market_stage_manifest_sha256: str,
) -> dict[str, Any]:
    """Rebuild every stage OHLCV value from exact externally pinned bytes."""

    source_value = _detach_plain_json(source_manifest, "market source manifest")
    stage_value = _detach_plain_json(stage_manifest, "market stage manifest")
    if type(artifact_bytes_by_symbol) is not dict:
        raise MarketSourceBytesError("Artifact byte mapping must be a detached plain dict")
    if type(window_bytes_by_symbol) is not dict:
        raise MarketSourceBytesError("Window byte mapping must be a detached plain dict")
    if set(artifact_bytes_by_symbol) != set(MARKET_SYMBOLS):
        raise MarketSourceBytesError("Artifact bytes must contain exactly six symbols")
    if set(window_bytes_by_symbol) != set(MARKET_SYMBOLS):
        raise MarketSourceBytesError("Window bytes must contain exactly six symbols")
    artifact_inputs = {
        symbol: artifact_bytes_by_symbol[symbol] for symbol in MARKET_SYMBOLS
    }
    window_inputs = {
        symbol: window_bytes_by_symbol[symbol] for symbol in MARKET_SYMBOLS
    }
    specs = _source_specs(source_value)
    artifact_hashes = {
        symbol: _sha256(specs[symbol]["artifact_sha256"], f"{symbol}.artifact_sha256")
        for symbol in MARKET_SYMBOLS
    }
    window_hashes = {
        symbol: _sha256(specs[symbol]["window_sha256"], f"{symbol}.window_sha256")
        for symbol in MARKET_SYMBOLS
    }
    source_hash = validate_market_source_manifest(
        source_value,
        expected_artifact_stage=expected_artifact_stage,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_source_artifact_sha256s=artifact_hashes,
        expected_source_window_sha256s=window_hashes,
    )
    stage_hash = validate_market_stage_manifest(
        stage_value,
        source_manifest=source_value,
        expected_artifact_stage=expected_artifact_stage,
        expected_source_manifest_sha256=source_hash,
        expected_market_stage_manifest_sha256=expected_market_stage_manifest_sha256,
    )
    stage_rows = stage_value["rows"]
    if isinstance(stage_rows, (str, bytes)) or not isinstance(stage_rows, Sequence):
        raise MarketSourceBytesError("Market stage rows are invalid")
    start = source_value["window_start"]
    end = source_value["window_end"]
    reconciled_counts: dict[str, int] = {}
    artifact_snapshot_hashes: dict[str, str] = {}
    window_snapshot_hashes: dict[str, str] = {}

    for symbol in MARKET_SYMBOLS:
        artifact_bytes = artifact_inputs[symbol]
        window_bytes = window_inputs[symbol]
        spec = specs[symbol]
        if (
            hashlib.sha256(artifact_bytes).hexdigest() != artifact_hashes[symbol]
            or len(artifact_bytes) != spec["artifact_bytes"]
        ):
            raise MarketSourceBytesError(f"{symbol} artifact bytes do not match the manifest")
        if (
            hashlib.sha256(window_bytes).hexdigest() != window_hashes[symbol]
            or len(window_bytes) != spec["window_bytes"]
        ):
            raise MarketSourceBytesError(f"{symbol} window bytes do not match the manifest")
        artifact = _parse_snapshot(
            artifact_bytes,
            expected_symbol=symbol,
            location=f"{symbol} artifact snapshot",
        )
        window = _parse_snapshot(
            window_bytes,
            expected_symbol=symbol,
            location=f"{symbol} window snapshot",
        )
        if any(row["session"] > end for row in artifact["rows"]):
            raise MarketSourceBytesError(
                f"{symbol} artifact contains a future row beyond the stage cutoff"
            )
        derived_rows = [
            row for row in artifact["rows"] if start <= row["session"] <= end
        ]
        derived_window_bytes = build_market_source_snapshot_bytes(
            symbol=symbol,
            rows=derived_rows,
        )
        if derived_window_bytes != window_bytes or window["rows"] != derived_rows:
            raise MarketSourceBytesError(
                f"{symbol} window is not the exact cumulative stage slice"
            )
        available_sessions = [row["session"] for row in derived_rows]
        if available_sessions != spec["available_sessions"]:
            raise MarketSourceBytesError(
                f"{symbol} byte rows do not match source availability evidence"
            )
        expected_rows: list[dict[str, str]] = []
        for stage_row in stage_rows:
            observation = stage_row["observations"][symbol]
            if observation["available"] is True:
                expected_rows.append(
                    {
                        "session": stage_row["session"],
                        **{
                            f"{field}_hex": observation[f"{field}_hex"]
                            for field in MARKET_FIELDS
                        },
                    }
                )
        if derived_rows != expected_rows:
            raise MarketSourceBytesError(
                f"{symbol} stage OHLCV values do not reconcile to source bytes"
            )
        reconciled_counts[symbol] = len(derived_rows)
        artifact_snapshot_hashes[symbol] = artifact["snapshot_sha256"]
        window_snapshot_hashes[symbol] = window["snapshot_sha256"]

    body = {
        "schema_version": MARKET_SOURCE_RECONCILIATION_SCHEMA_VERSION,
        "artifact_stage": expected_artifact_stage,
        "source_manifest_sha256": source_hash,
        "market_stage_manifest_sha256": stage_hash,
        "artifact_snapshot_sha256s": artifact_snapshot_hashes,
        "window_snapshot_sha256s": window_snapshot_hashes,
        "reconciled_row_counts": reconciled_counts,
        "value_reconciliation": "all_available_ohlcv_fields_exact_float_hex",
    }
    return {**body, "reconciliation_sha256": canonical_sha256(body)}


__all__ = [
    "MARKET_SOURCE_RECONCILIATION_SCHEMA_VERSION",
    "MARKET_SOURCE_SNAPSHOT_SCHEMA_VERSION",
    "MAX_MARKET_SOURCE_SNAPSHOT_BYTES",
    "MarketSourceBytesError",
    "build_market_source_snapshot_bytes",
    "validate_market_source_bytes_against_stage",
]
