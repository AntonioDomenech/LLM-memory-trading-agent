"""Pure decision-time features for the SEC/Gemma online risk overlay.

Raw market rows and extractor output may be converted only into unpersisted
numeric components.  A complete authoritative feature row, including its
canonical hash, can be minted only by the owned public entry points after they
replay externally pinned market, universe, and extraction evidence.

The three feature arms are deliberately explicit:

``semantic``
    Six market values, four filing-meaning values, extraction quality, and form.
``no_filing_meaning``
    The same row with only the four filing-meaning values zeroed.  Extraction
    quality is preserved, so invalid model output cannot masquerade as meaning.
``no_gemma_channel``
    A diagnostic with all five Gemma-derived values zeroed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from datetime import date, datetime
import hashlib
import hmac
import math
import re
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_contract import (
    ADVERSE_FLAG_NAMES,
    DIMENSION_NAMES,
    DOCUMENT_QUALITIES,
    HORIZON_SESSIONS as SEC_HORIZON_SESSIONS,
    MAX_SENTENCES,
    STAGE_ORDER,
    STAGE_WINDOWS,
    build_contract_manifest as build_sec_contract_manifest,
    validate_corpus_universe_manifest,
    validate_extractor_output,
    validate_stage_content_manifest,
)
from agent_benchmark.sec_filing_gemma_market_evidence import (
    CANONICAL_MARKET_FIELDS,
    MARKET_LOOKBACK_ROW_COUNT,
    MARKET_PREFIX_SCHEMA_VERSION,
    MARKET_ROW_SCHEMA_VERSION,
    MARKET_SYMBOLS,
    decode_float_hex,
    derive_adjusted_open_hex,
    market_session_calendar_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    CONTROL_FEATURES,
    FEATURES,
    MARKET_FEATURES,
    MEANING_FEATURES,
    QUALITY_FEATURE,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_market_verifier import (
    MARKET_UNAVAILABLE_PROOF_SCHEMA_VERSION,
    MARKET_UNAVAILABLE_REASONS,
    validate_market_unavailable_event_proof,
)
from agent_benchmark.sec_session_calendar import (
    EXPECTED_MARKET_HISTORY_SESSIONS,
    EXPECTED_SESSIONS,
    MARKET_HISTORY_CALENDAR_ID,
)


FEATURE_ROW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-feature-row-v1"
)
MARKET_RESULT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-market-features-v1"
)
MARKET_PREFIX_PROOF_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-validated-market-prefix-proof-v1"
)
UNIVERSE_EVENT_PROOF_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-validated-universe-event-proof-v1"
)
EXTRACTION_EVENT_PROOF_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-validated-extraction-event-proof-v1"
)
REQUIRED_PREFIX_ROWS: Final[int] = 253
REQUIRED_MARKET_SYMBOLS: Final[tuple[str, ...]] = (
    "AAPL",
    "QQQ",
    "SPY",
    "IWM",
    "VIX",
)
CURRENT_IMPACT_ENCODING: Final[dict[str, float]] = {
    "favorable": -1.0,
    "neutral": 0.0,
    "unfavorable": 1.0,
    "mixed": 0.0,
    "not_stated": 0.0,
}
CHANGE_ENCODING: Final[dict[str, float]] = {
    "improving": -1.0,
    "stable": 0.0,
    "deteriorating": 1.0,
    "mixed": 0.0,
    "not_comparable": 0.0,
    "not_stated": 0.0,
}
SEMANTIC_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "commercial_deterioration": (
        "demand",
        "pricing_power",
        "supply_chain",
    ),
    "financial_deterioration": (
        "gross_margin",
        "operating_cost_pressure",
        "capital_allocation",
        "liquidity",
    ),
    "risk_outlook_deterioration": (
        "forward_guidance",
        "legal_regulatory",
        "management_uncertainty",
    ),
}
DOCUMENT_QUALITY_RISK: Final[dict[str, float]] = {
    "usable": 0.0,
    "thin": 0.5,
    "unusable": 1.0,
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ACCESSION_RE = re.compile(r"0000320193-[0-9]{2}-[0-9]{6}\Z")
_ACCEPTANCE_RE = re.compile(r"[0-9]{14}\Z")
_SENTENCE_ID_RE = re.compile(r"[CP][0-9]{4}\Z")
_MARKET_OBSERVATION_KEYS = {
    "available",
    *(f"{name}_hex" for name in CANONICAL_MARKET_FIELDS),
}
_MARKET_ROW_KEYS = {
    "schema_version",
    "row_index",
    "session",
    "observations",
    "previous_row_sha256",
    "row_sha256",
}
_PREFIX_KEYS = {
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
_UNIVERSE_RECORD_KEYS = {
    "accession_number",
    "subject_cik",
    "form",
    "acceptance_datetime",
    "filing_date",
    "filing_date_change",
    "primary_document",
    "source_record_sha256",
    "availability_session",
    "artifact_stage",
}
_CONTENT_RECORD_KEYS = {
    "accession_number",
    "form",
    "availability_session",
    "primary_document",
    "primary_document_sha256",
    "normalized_text_sha256",
    "primary_document_bytes",
    "normalized_text_bytes",
}


class SecGemmaOnlineRiskOverlayFeatureError(ValueError):
    """Raised when decision-time evidence cannot be authenticated exactly."""


def _proof_keys(
    value: Mapping[str, Any],
    expected: set[str],
    location: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"Invalid {location} keys; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _optional_sha256(value: Any, location: str) -> str | None:
    return None if value is None else _sha256(value, location)


def _iso_date(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} must be a canonical ISO date"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} must be a canonical ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} must use YYYY-MM-DD form"
        )
    return value


def _strict_positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} must be an integer >= 1"
        )
    return value


def _offset_session(session: str, offset: int) -> str:
    try:
        position = EXPECTED_MARKET_HISTORY_SESSIONS.index(session)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Session is outside the exact frozen trading calendar"
        ) from exc
    target = position + offset
    if target < 0 or target >= len(EXPECTED_MARKET_HISTORY_SESSIONS):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Frozen calendar does not cover the offset"
        )
    return EXPECTED_MARKET_HISTORY_SESSIONS[target]


def _validate_proof_observation(
    value: Any,
    location: str,
) -> dict[str, Any]:
    observation = _mapping(value, location)
    _proof_keys(observation, _MARKET_OBSERVATION_KEYS, location)
    available = observation["available"]
    if not isinstance(available, bool):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location}.available must be boolean"
        )
    if not available:
        if any(
            observation[f"{name}_hex"] is not None
            for name in CANONICAL_MARKET_FIELDS
        ):
            raise SecGemmaOnlineRiskOverlayFeatureError(
                f"{location} unavailable observation must use explicit nulls"
            )
        return dict(observation)

    numeric = {
        name: decode_float_hex(
            observation[f"{name}_hex"], f"{location}.{name}"
        )
        for name in (
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        )
    }
    if any(
        numeric[name] <= 0.0
        for name in ("open", "high", "low", "close", "adjusted_close")
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} prices must be positive"
        )
    if numeric["volume"] < 0.0:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} volume cannot be negative"
        )
    if numeric["high"] < max(
        numeric["open"], numeric["low"], numeric["close"]
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} high is inconsistent with OHLC"
        )
    if numeric["low"] > min(
        numeric["open"], numeric["high"], numeric["close"]
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} low is inconsistent with OHLC"
        )
    expected_adjusted_open = derive_adjusted_open_hex(
        open_hex=observation["open_hex"],
        close_hex=observation["close_hex"],
        adjusted_close_hex=observation["adjusted_close_hex"],
        location=location,
    )
    adjusted_open = observation["adjusted_open_hex"]
    decode_float_hex(adjusted_open, f"{location}.adjusted_open")
    if adjusted_open != expected_adjusted_open:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location}.adjusted_open is not the frozen derived value"
        )
    return dict(observation)


def _validate_proof_market_row(
    value: Any,
    *,
    expected_index: int,
    expected_session: str,
    expected_parent_sha256: str,
    location: str,
) -> dict[str, Any]:
    row = _mapping(value, location)
    _proof_keys(row, _MARKET_ROW_KEYS, location)
    if row["schema_version"] != MARKET_ROW_SCHEMA_VERSION:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market row schema changed"
        )
    if (
        isinstance(row["row_index"], bool)
        or not isinstance(row["row_index"], int)
        or row["row_index"] != expected_index
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market row index changed or is not contiguous"
        )
    session = _iso_date(row["session"], f"{location}.session")
    if session != expected_session:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market rows are missing, shifted, or reordered"
        )
    if row["previous_row_sha256"] != expected_parent_sha256:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market row hash chain is broken"
        )
    observations = _mapping(
        row["observations"], f"{location}.observations"
    )
    if set(observations) != set(MARKET_SYMBOLS):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market row must contain exactly all six symbols"
        )
    normalized_observations = {
        symbol: _validate_proof_observation(
            observations[symbol], f"{location}.observations.{symbol}"
        )
        for symbol in MARKET_SYMBOLS
    }
    if normalized_observations["AAPL"]["available"] is not True:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "AAPL must be present in every market row"
        )
    body = {
        "schema_version": MARKET_ROW_SCHEMA_VERSION,
        "row_index": expected_index,
        "session": session,
        "observations": normalized_observations,
        "previous_row_sha256": expected_parent_sha256,
    }
    calculated = canonical_sha256(body)
    if not hmac.compare_digest(
        calculated,
        _sha256(row["row_sha256"], f"{location}.row_sha256"),
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market row checksum does not reconcile"
        )
    return {**body, "row_sha256": calculated}


def _validate_compact_market_prefix(
    prefix: Mapping[str, Any],
    *,
    expected_market_prefix_sha256: str,
) -> dict[str, Any]:
    value = _mapping(prefix, "decision market prefix")
    _proof_keys(value, _PREFIX_KEYS, "decision market prefix")
    if value["schema_version"] != MARKET_PREFIX_SCHEMA_VERSION:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision market prefix schema changed"
        )
    stage = value["artifact_stage"]
    if stage not in STAGE_ORDER:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision market prefix stage is invalid"
        )
    decision = _iso_date(value["decision_session"], "decision_session")
    if not (STAGE_WINDOWS[stage][0] <= decision <= STAGE_WINDOWS[stage][1]):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision session is outside its stage"
        )
    if (
        value["market_cutoff_rule"]
        != "completed_decision_session_close_inclusive"
        or value["market_cutoff_session"] != decision
        or value["full_prefix_last_session"] != decision
        or value["lookback_last_session"] != decision
        or value["fill_session_offset"] != 1
        or value["fill_session"] != _offset_session(decision, 1)
        or value["label_maturity_session_offset"]
        != SEC_HORIZON_SESSIONS + 1
        or value["label_maturity_session"]
        != _offset_session(decision, SEC_HORIZON_SESSIONS + 1)
        or value["calendar_id"] != MARKET_HISTORY_CALENDAR_ID
        or value["maximum_feature_lookback_sessions"] != 252
        or value["lookback_row_count"] != MARKET_LOOKBACK_ROW_COUNT
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision market timing or lookback changed"
        )
    _sha256(
        value["market_stage_manifest_sha256"],
        "market_stage_manifest_sha256",
    )
    _sha256(value["source_manifest_sha256"], "source_manifest_sha256")
    if value["calendar_sha256"] != market_session_calendar_sha256(
        EXPECTED_MARKET_HISTORY_SESSIONS
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision prefix calendar binding changed"
        )
    try:
        decision_position = EXPECTED_MARKET_HISTORY_SESSIONS.index(decision)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision session is outside the exact market calendar"
        ) from exc
    expected_sessions = EXPECTED_MARKET_HISTORY_SESSIONS[
        decision_position - 252 : decision_position + 1
    ]
    if len(expected_sessions) != MARKET_LOOKBACK_ROW_COUNT:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision lacks the exact market prehistory"
        )
    expected_start_index = decision_position - 252
    if (
        value["lookback_start_row_index"] != expected_start_index
        or value["lookback_end_row_index"] != decision_position
        or value["lookback_first_session"] != expected_sessions[0]
        or value["full_prefix_first_session"]
        != EXPECTED_MARKET_HISTORY_SESSIONS[0]
        or value["full_prefix_row_count"] != decision_position + 1
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision market prefix indices changed"
        )
    rows_value = value["lookback_rows"]
    if isinstance(rows_value, (str, bytes)) or not isinstance(
        rows_value, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision lookback rows must be a sequence"
        )
    if len(rows_value) != MARKET_LOOKBACK_ROW_COUNT:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision lookback must contain exactly 253 rows"
        )
    parent = _sha256(
        value["lookback_start_parent_row_sha256"],
        "lookback_start_parent_row_sha256",
    )
    rows: list[dict[str, Any]] = []
    for offset, expected_session in enumerate(expected_sessions):
        row = _validate_proof_market_row(
            rows_value[offset],
            expected_index=expected_start_index + offset,
            expected_session=expected_session,
            expected_parent_sha256=parent,
            location=f"lookback_rows[{offset}]",
        )
        rows.append(row)
        parent = row["row_sha256"]
    row_hashes = [row["row_sha256"] for row in rows]
    frozen_genesis = hashlib.sha256(
        b"aapl-sec-gemma-market-row-chain-v1\x00"
    ).hexdigest()
    prefix_chain_identity = canonical_sha256(
        {
            "domain": "aapl-sec-gemma-market-prefix-chain-v1",
            "row_chain_genesis_sha256": frozen_genesis,
            "full_prefix_row_count": value["full_prefix_row_count"],
            "full_prefix_row_chain_tip_sha256": parent,
        }
    )
    slice_identity = canonical_sha256(
        {
            "domain": "aapl-sec-gemma-market-lookback-slice-v1",
            "first_row_index": rows[0]["row_index"],
            "last_row_index": rows[-1]["row_index"],
            "row_count": len(rows),
            "start_parent_row_sha256": rows[0]["previous_row_sha256"],
            "row_hashes_sha256": canonical_sha256(row_hashes),
            "row_chain_tip_sha256": parent,
        }
    )
    if (
        value["lookback_row_chain_tip_sha256"] != parent
        or value["full_prefix_row_chain_tip_sha256"] != parent
        or value["lookback_row_hashes_sha256"] != canonical_sha256(row_hashes)
        or value["lookback_rows_sha256"] != canonical_sha256(rows)
        or value["lookback_slice_identity_sha256"] != slice_identity
        or value["full_prefix_chain_identity_sha256"]
        != prefix_chain_identity
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision prefix chain identities do not reconcile"
        )
    body = {
        key: value[key]
        for key in value
        if key != "market_prefix_sha256"
    }
    calculated = canonical_sha256(body)
    observed = _sha256(
        value["market_prefix_sha256"], "market_prefix_sha256"
    )
    if not hmac.compare_digest(observed, calculated):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision market prefix checksum changed"
        )
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_market_prefix_sha256,
            "expected_market_prefix_sha256",
        ),
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision market prefix is not externally pinned"
        )
    return {**body, "lookback_rows": rows, "market_prefix_sha256": observed}


def validate_market_prefix_proof(
    proof: Mapping[str, Any],
    *,
    prefix: Mapping[str, Any],
    expected_market_prefix_proof_sha256: str,
) -> dict[str, Any]:
    """Validate an externally pinned compact market-prefix proof."""

    value = _mapping(proof, "market prefix proof")
    expected_keys = {
        "schema_version",
        "artifact_stage",
        "decision_event_id",
        "decision_session",
        "market_prefix_sha256",
        "market_stage_manifest_sha256",
        "source_manifest_sha256",
        "calendar_sha256",
        "full_prefix_chain_identity_sha256",
        "lookback_rows_sha256",
        "lookback_row_count",
        "validation_semantics",
        "authoritative_source_byte_reconciliation_required",
        "market_prefix_proof_sha256",
    }
    _proof_keys(value, expected_keys, "market prefix proof")
    compact = _validate_compact_market_prefix(
        prefix,
        expected_market_prefix_sha256=value["market_prefix_sha256"],
    )
    expected_fields = {
        "schema_version": MARKET_PREFIX_PROOF_SCHEMA_VERSION,
        "artifact_stage": compact["artifact_stage"],
        "decision_event_id": compact["decision_event_id"],
        "decision_session": compact["decision_session"],
        "market_prefix_sha256": compact["market_prefix_sha256"],
        "market_stage_manifest_sha256": compact[
            "market_stage_manifest_sha256"
        ],
        "source_manifest_sha256": compact["source_manifest_sha256"],
        "calendar_sha256": compact["calendar_sha256"],
        "full_prefix_chain_identity_sha256": compact[
            "full_prefix_chain_identity_sha256"
        ],
        "lookback_rows_sha256": compact["lookback_rows_sha256"],
        "lookback_row_count": MARKET_LOOKBACK_ROW_COUNT,
        "validation_semantics": (
            "structural_stage_source_hash_replay_then_compact_prefix_isolation"
        ),
        "authoritative_source_byte_reconciliation_required": True,
    }
    if any(
        value[key] != expected
        for key, expected in expected_fields.items()
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market prefix proof differs from its prefix"
        )
    body = {
        key: value[key]
        for key in value
        if key != "market_prefix_proof_sha256"
    }
    calculated = canonical_sha256(body)
    observed = _sha256(
        value["market_prefix_proof_sha256"],
        "market_prefix_proof_sha256",
    )
    if not hmac.compare_digest(observed, calculated):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market prefix proof checksum changed"
        )
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_market_prefix_proof_sha256,
            "expected_market_prefix_proof_sha256",
        ),
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market prefix proof is not externally pinned"
        )
    return {**dict(value), "compact_prefix": compact}


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} must be a mapping"
        )
    if not all(isinstance(key, str) for key in value):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} must be a string-keyed mapping"
        )
    return value


def _normalize_universe_record(
    value: Any,
    location: str,
) -> dict[str, Any]:
    record = _mapping(value, location)
    _proof_keys(record, _UNIVERSE_RECORD_KEYS, location)
    accession = record["accession_number"]
    if (
        not isinstance(accession, str)
        or _ACCESSION_RE.fullmatch(accession) is None
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} is not an Apple accession"
        )
    if (
        record["subject_cik"] != "0000320193"
        or record["form"] not in {"10-K", "10-Q"}
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} is not an eligible Apple filing"
        )
    acceptance = record["acceptance_datetime"]
    if acceptance is not None:
        if (
            not isinstance(acceptance, str)
            or _ACCEPTANCE_RE.fullmatch(acceptance) is None
        ):
            raise SecGemmaOnlineRiskOverlayFeatureError(
                f"{location} acceptance timestamp is invalid"
            )
        try:
            datetime.strptime(acceptance, "%Y%m%d%H%M%S")
        except ValueError as exc:
            raise SecGemmaOnlineRiskOverlayFeatureError(
                f"{location} acceptance timestamp is invalid"
            ) from exc
    filing_date = _iso_date(
        record["filing_date"], f"{location}.filing_date"
    )
    change = record["filing_date_change"]
    if change is not None:
        change = _iso_date(change, f"{location}.filing_date_change")
    availability = _iso_date(
        record["availability_session"],
        f"{location}.availability_session",
    )
    if availability not in EXPECTED_SESSIONS:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} availability is not a frozen session"
        )
    stage = record["artifact_stage"]
    if stage not in STAGE_ORDER or not (
        STAGE_WINDOWS[stage][0] <= availability <= STAGE_WINDOWS[stage][1]
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} stage is inconsistent"
        )
    primary = record["primary_document"]
    if (
        not isinstance(primary, str)
        or not primary
        or "/" in primary
        or "\\" in primary
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} primary document is invalid"
        )
    return {
        "accession_number": accession,
        "subject_cik": "0000320193",
        "form": record["form"],
        "acceptance_datetime": acceptance,
        "filing_date": filing_date,
        "filing_date_change": change,
        "primary_document": primary,
        "source_record_sha256": _sha256(
            record["source_record_sha256"],
            f"{location}.source_record_sha256",
        ),
        "availability_session": availability,
        "artifact_stage": stage,
    }


def _normalize_content_record(
    value: Any,
    location: str,
) -> dict[str, Any]:
    record = _mapping(value, location)
    _proof_keys(record, _CONTENT_RECORD_KEYS, location)
    accession = record["accession_number"]
    if (
        not isinstance(accession, str)
        or _ACCESSION_RE.fullmatch(accession) is None
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} accession is invalid"
        )
    form = record["form"]
    if form not in {"10-K", "10-Q"}:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} form is invalid"
        )
    availability = _iso_date(
        record["availability_session"],
        f"{location}.availability_session",
    )
    if availability not in EXPECTED_SESSIONS:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} availability is not a frozen session"
        )
    primary = record["primary_document"]
    if (
        not isinstance(primary, str)
        or not primary
        or "/" in primary
        or "\\" in primary
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} primary document is invalid"
        )
    return {
        "accession_number": accession,
        "form": form,
        "availability_session": availability,
        "primary_document": primary,
        "primary_document_sha256": _sha256(
            record["primary_document_sha256"],
            f"{location}.primary_document_sha256",
        ),
        "normalized_text_sha256": _sha256(
            record["normalized_text_sha256"],
            f"{location}.normalized_text_sha256",
        ),
        "primary_document_bytes": _strict_positive_int(
            record["primary_document_bytes"],
            f"{location}.primary_document_bytes",
        ),
        "normalized_text_bytes": _strict_positive_int(
            record["normalized_text_bytes"],
            f"{location}.normalized_text_bytes",
        ),
    }


def build_validated_universe_event_proof(
    *,
    universe_manifest: Mapping[str, Any],
    expected_corpus_universe_sha256: str,
    current_accession_number: str,
    content_manifests_by_stage: Mapping[str, Mapping[str, Any]],
    expected_content_manifest_sha256s: Mapping[str, str],
    session_dates: Sequence[str] = EXPECTED_SESSIONS,
) -> dict[str, Any]:
    """Mint the legacy-compatible compact event proof without NumPy/pandas."""

    universe_hash = validate_corpus_universe_manifest(
        universe_manifest,
        session_dates=session_dates,
        expected_universe_sha256=expected_corpus_universe_sha256,
        require_complete_coverage=True,
    )
    records = list(universe_manifest["records"])
    current = next(
        (
            record
            for record in records
            if record["accession_number"] == current_accession_number
        ),
        None,
    )
    if current is None:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Current event is absent from the universe"
        )
    current = _normalize_universe_record(
        current, "current universe record"
    )
    before = sorted(
        (
            _normalize_universe_record(
                record, "prior universe record"
            )
            for record in records
            if record["form"] == current["form"]
            and (
                record["availability_session"],
                record["accession_number"],
            )
            < (
                current["availability_session"],
                current["accession_number"],
            )
        ),
        key=lambda record: (
            record["availability_session"],
            record["accession_number"],
        ),
    )
    prior = before[-1] if before else None
    needed_stages = {current["artifact_stage"]}
    if prior is not None:
        needed_stages.add(prior["artifact_stage"])
    manifests = _mapping(
        content_manifests_by_stage, "content_manifests_by_stage"
    )
    expected_hashes = _mapping(
        expected_content_manifest_sha256s,
        "expected_content_manifest_sha256s",
    )
    _proof_keys(manifests, needed_stages, "content_manifests_by_stage")
    _proof_keys(
        expected_hashes,
        needed_stages,
        "expected_content_manifest_sha256s",
    )
    content_by_accession: dict[str, Mapping[str, Any]] = {}
    normalized_content_hashes: dict[str, str] = {}
    for stage in STAGE_ORDER:
        if stage not in needed_stages:
            continue
        manifest = manifests[stage]
        content_hash = validate_stage_content_manifest(
            manifest,
            universe_manifest=universe_manifest,
            expected_content_manifest_sha256=expected_hashes[stage],
        )
        normalized_content_hashes[stage] = content_hash
        for document in manifest["documents"]:
            content_by_accession[document["accession_number"]] = document
    current_content = content_by_accession.get(
        current["accession_number"]
    )
    prior_content = (
        None
        if prior is None
        else content_by_accession.get(prior["accession_number"])
    )
    if current_content is None or (
        prior is not None and prior_content is None
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Universe event content is absent from its validated stage "
            "manifest"
        )
    body = {
        "schema_version": UNIVERSE_EVENT_PROOF_SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(
            build_sec_contract_manifest()
        ),
        "corpus_universe_sha256": universe_hash,
        "current_record": copy.deepcopy(current),
        "current_record_sha256": canonical_sha256(current),
        "current_content_record": copy.deepcopy(dict(current_content)),
        "current_content_record_sha256": canonical_sha256(current_content),
        "current_filing_sha256": current_content[
            "normalized_text_sha256"
        ],
        "prior_same_form_record": copy.deepcopy(prior),
        "prior_same_form_record_sha256": (
            None if prior is None else canonical_sha256(prior)
        ),
        "prior_same_form_content_record": (
            None
            if prior_content is None
            else copy.deepcopy(dict(prior_content))
        ),
        "prior_same_form_content_record_sha256": (
            None
            if prior_content is None
            else canonical_sha256(prior_content)
        ),
        "prior_same_form_filing_sha256": (
            None
            if prior_content is None
            else prior_content["normalized_text_sha256"]
        ),
        "content_manifest_sha256s_by_stage": normalized_content_hashes,
        "prior_selection": (
            "immediate_predecessor_same_form_by_availability_session_and_"
            "accession"
        ),
    }
    return {
        **body,
        "universe_event_proof_sha256": canonical_sha256(body),
    }


def validate_universe_event_proof(
    proof: Mapping[str, Any],
    *,
    expected_universe_event_proof_sha256: str,
) -> dict[str, Any]:
    """Validate the compact universe/content proof used by a worker."""

    value = _mapping(proof, "universe event proof")
    expected_keys = {
        "schema_version",
        "contract_sha256",
        "corpus_universe_sha256",
        "current_record",
        "current_record_sha256",
        "current_content_record",
        "current_content_record_sha256",
        "current_filing_sha256",
        "prior_same_form_record",
        "prior_same_form_record_sha256",
        "prior_same_form_content_record",
        "prior_same_form_content_record_sha256",
        "prior_same_form_filing_sha256",
        "content_manifest_sha256s_by_stage",
        "prior_selection",
        "universe_event_proof_sha256",
    }
    _proof_keys(value, expected_keys, "universe event proof")
    if (
        value["schema_version"] != UNIVERSE_EVENT_PROOF_SCHEMA_VERSION
        or value["contract_sha256"]
        != canonical_sha256(build_sec_contract_manifest())
        or value["prior_selection"]
        != (
            "immediate_predecessor_same_form_by_availability_session_and_"
            "accession"
        )
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Universe event proof semantics changed"
        )
    _sha256(value["corpus_universe_sha256"], "corpus_universe_sha256")
    current = _normalize_universe_record(
        value["current_record"], "current record"
    )
    if value["current_record_sha256"] != canonical_sha256(current):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Current universe record hash changed"
        )
    current_content = _normalize_content_record(
        value["current_content_record"], "current content record"
    )
    if (
        current_content["accession_number"]
        != current["accession_number"]
        or current_content["form"] != current["form"]
        or current_content["availability_session"]
        != current["availability_session"]
        or current_content["primary_document"]
        != current["primary_document"]
        or value["current_content_record_sha256"]
        != canonical_sha256(current_content)
        or value["current_filing_sha256"]
        != current_content["normalized_text_sha256"]
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Current content proof does not reconcile"
        )
    _sha256(value["current_filing_sha256"], "current_filing_sha256")
    prior_value = value["prior_same_form_record"]
    prior_content: dict[str, Any] | None = None
    if prior_value is None:
        if any(
            value[name] is not None
            for name in (
                "prior_same_form_record_sha256",
                "prior_same_form_content_record",
                "prior_same_form_content_record_sha256",
                "prior_same_form_filing_sha256",
            )
        ):
            raise SecGemmaOnlineRiskOverlayFeatureError(
                "First same-form event claims prior evidence"
            )
        prior = None
    else:
        prior = _normalize_universe_record(
            prior_value, "prior same-form record"
        )
        prior_content = _normalize_content_record(
            value["prior_same_form_content_record"],
            "prior same-form content record",
        )
        if (
            prior["form"] != current["form"]
            or (
                prior["availability_session"],
                prior["accession_number"],
            )
            >= (
                current["availability_session"],
                current["accession_number"],
            )
            or value["prior_same_form_record_sha256"]
            != canonical_sha256(prior)
            or prior_content["accession_number"]
            != prior["accession_number"]
            or prior_content["form"] != prior["form"]
            or prior_content["availability_session"]
            != prior["availability_session"]
            or prior_content["primary_document"]
            != prior["primary_document"]
            or value["prior_same_form_content_record_sha256"]
            != canonical_sha256(prior_content)
            or value["prior_same_form_filing_sha256"]
            != prior_content["normalized_text_sha256"]
        ):
            raise SecGemmaOnlineRiskOverlayFeatureError(
                "Prior same-form content proof is inconsistent"
            )
        _sha256(
            value["prior_same_form_filing_sha256"],
            "prior_same_form_filing_sha256",
        )
    stage_hashes = _mapping(
        value["content_manifest_sha256s_by_stage"],
        "content_manifest_sha256s_by_stage",
    )
    needed_stages = {current["artifact_stage"]}
    if prior is not None:
        needed_stages.add(prior["artifact_stage"])
    _proof_keys(
        stage_hashes,
        needed_stages,
        "content_manifest_sha256s_by_stage",
    )
    for stage, digest in stage_hashes.items():
        _sha256(digest, f"content_manifest_sha256s_by_stage.{stage}")
    body = {
        key: value[key]
        for key in value
        if key != "universe_event_proof_sha256"
    }
    calculated = canonical_sha256(body)
    observed = _sha256(
        value["universe_event_proof_sha256"],
        "universe_event_proof_sha256",
    )
    if not hmac.compare_digest(observed, calculated):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Universe event proof checksum changed"
        )
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_universe_event_proof_sha256,
            "expected_universe_event_proof_sha256",
        ),
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Universe event proof is not externally pinned"
        )
    return {
        **dict(value),
        "current_record": current,
        "current_content_record": current_content,
        "prior_same_form_record": prior,
        "prior_same_form_content_record": (
            None if prior is None else prior_content
        ),
    }


def _normalize_supplied_sentence_ids(
    value: Any,
    location: str,
) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} must be a sequence"
        )
    sentence_ids = list(value)
    if len(sentence_ids) > MAX_SENTENCES or any(
        not isinstance(item, str)
        or _SENTENCE_ID_RE.fullmatch(item) is None
        for item in sentence_ids
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} contains invalid sentence ids"
        )
    if len(sentence_ids) != len(set(sentence_ids)):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} sentence ids must be unique"
        )
    if sentence_ids:
        current_count = sum(
            item.startswith("C") for item in sentence_ids
        )
        prior_count = len(sentence_ids) - current_count
        if current_count == 0:
            raise SecGemmaOnlineRiskOverlayFeatureError(
                f"{location} requires current-filing sentence ids"
            )
        expected = [
            f"C{index:04d}" for index in range(1, current_count + 1)
        ]
        expected += [
            f"P{index:04d}" for index in range(1, prior_count + 1)
        ]
        if sentence_ids != expected:
            raise SecGemmaOnlineRiskOverlayFeatureError(
                f"{location} sentence ids must be contiguous and ordered"
            )
    return sentence_ids


def validate_extraction_event_proof(
    proof: Mapping[str, Any],
    *,
    universe_event_proof: Mapping[str, Any],
    expected_universe_event_proof_sha256: str,
    expected_extraction_event_proof_sha256: str,
) -> dict[str, Any]:
    """Validate exact extraction status, semantics, and event binding."""

    universe = validate_universe_event_proof(
        universe_event_proof,
        expected_universe_event_proof_sha256=(
            expected_universe_event_proof_sha256
        ),
    )
    value = _mapping(proof, "extraction event proof")
    expected_keys = {
        "schema_version",
        "contract_sha256",
        "universe_event_proof_sha256",
        "accession_number",
        "form",
        "decision_session",
        "current_filing_sha256",
        "prior_same_form_filing_sha256",
        "extraction_status",
        "extraction_evidence_authenticated",
        "extraction_evidence_sha256",
        "extraction_output_sha256",
        "extraction_output_canonical_sha256",
        "supplied_sentence_ids",
        "supplied_sentence_ids_sha256",
        "document_quality",
        "validated_output",
        "validation_scope",
        "authoritative_request_and_ollama_receipt_replay_required",
        "extraction_event_proof_sha256",
    }
    _proof_keys(value, expected_keys, "extraction event proof")
    current = universe["current_record"]
    if (
        value["schema_version"] != EXTRACTION_EVENT_PROOF_SCHEMA_VERSION
        or value["contract_sha256"]
        != canonical_sha256(build_sec_contract_manifest())
        or value["universe_event_proof_sha256"]
        != universe["universe_event_proof_sha256"]
        or value["accession_number"] != current["accession_number"]
        or value["form"] != current["form"]
        or value["decision_session"] != current["availability_session"]
        or value["current_filing_sha256"]
        != universe["current_filing_sha256"]
        or value["prior_same_form_filing_sha256"]
        != universe["prior_same_form_filing_sha256"]
        or value["validation_scope"]
        != "external_pin_exact_output_bytes_hash_schema_and_event_binding"
        or value[
            "authoritative_request_and_ollama_receipt_replay_required"
        ]
        is not True
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Extraction proof is bound to another event"
        )
    status = value["extraction_status"]
    if status not in {"valid", "invalid", "unavailable"}:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Extraction proof status is invalid"
        )
    authenticated = value["extraction_evidence_authenticated"]
    if not isinstance(authenticated, bool):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Extraction authentication must be boolean"
        )
    evidence_hash = _optional_sha256(
        value["extraction_evidence_sha256"],
        "extraction_evidence_sha256",
    )
    output_hash = _optional_sha256(
        value["extraction_output_sha256"],
        "extraction_output_sha256",
    )
    canonical_output_hash = _optional_sha256(
        value["extraction_output_canonical_sha256"],
        "extraction_output_canonical_sha256",
    )
    if not isinstance(value["supplied_sentence_ids"], list):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Extraction proof sentence ids must be a list"
        )
    sentence_ids = _normalize_supplied_sentence_ids(
        value["supplied_sentence_ids"],
        "extraction proof supplied_sentence_ids",
    )
    if value["supplied_sentence_ids_sha256"] != canonical_sha256(
        sentence_ids
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Extraction proof sentence ids are invalid"
        )
    if sentence_ids:
        has_prior_ids = any(
            item.startswith("P") for item in sentence_ids
        )
        has_prior_filing = (
            universe["prior_same_form_filing_sha256"] is not None
        )
        if has_prior_ids != has_prior_filing:
            raise SecGemmaOnlineRiskOverlayFeatureError(
                "Extraction proof sentence ids do not reconcile with prior "
                "filing"
            )
    output = value["validated_output"]
    quality = value["document_quality"]
    if authenticated:
        if (
            evidence_hash is None
            or status == "unavailable"
            or not sentence_ids
        ):
            raise SecGemmaOnlineRiskOverlayFeatureError(
                "Authenticated extraction evidence has an impossible status"
            )
        if status == "invalid":
            if any(
                item is not None
                for item in (
                    output,
                    quality,
                    output_hash,
                    canonical_output_hash,
                )
            ):
                raise SecGemmaOnlineRiskOverlayFeatureError(
                    "Sealed invalid extraction cannot claim valid output"
                )
        else:
            output_mapping = _mapping(
                output, "validated extractor output"
            )
            if (
                output_hash is None
                or canonical_output_hash is None
                or quality not in DOCUMENT_QUALITIES
                or canonical_output_hash
                != canonical_sha256(output_mapping)
                or output_mapping.get("document_quality") != quality
            ):
                raise SecGemmaOnlineRiskOverlayFeatureError(
                    "Validated extraction output identities are incomplete"
                )
            validate_extractor_output(
                output_mapping,
                supplied_sentence_ids=sentence_ids,
            )
    elif any(
        item is not None
        for item in (
            output,
            quality,
            output_hash,
            canonical_output_hash,
        )
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Unauthenticated extraction cannot expose semantic output"
        )
    body = {
        key: value[key]
        for key in value
        if key != "extraction_event_proof_sha256"
    }
    calculated = canonical_sha256(body)
    observed = _sha256(
        value["extraction_event_proof_sha256"],
        "extraction_event_proof_sha256",
    )
    if not hmac.compare_digest(observed, calculated):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Extraction event proof checksum changed"
        )
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_extraction_event_proof_sha256,
            "expected_extraction_event_proof_sha256",
        ),
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Extraction event proof is not externally pinned"
        )
    return dict(value)


def _canonical_float_hex(value: Any, location: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} must be a finite number"
        )
    number = float(value)
    if not math.isfinite(number):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            f"{location} must be a finite number"
        )
    return number.hex()


def _decode_positive_float_hex(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        number = float.fromhex(value)
    except ValueError:
        return None
    if not math.isfinite(number) or number <= 0.0 or number.hex() != value:
        return None
    return number


def _unavailable_market_components(
    reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "available": False,
        "values": {name: None for name in MARKET_FEATURES},
        "unavailable_reasons": sorted(set(reasons)),
    }


def _required_market_positions() -> dict[str, tuple[int, ...]]:
    return {
        "AAPL": tuple(range(REQUIRED_PREFIX_ROWS - 63, REQUIRED_PREFIX_ROWS)),
        "QQQ": (REQUIRED_PREFIX_ROWS - 21, REQUIRED_PREFIX_ROWS - 1),
        "SPY": (REQUIRED_PREFIX_ROWS - 21, REQUIRED_PREFIX_ROWS - 1),
        "IWM": (REQUIRED_PREFIX_ROWS - 21, REQUIRED_PREFIX_ROWS - 1),
        "VIX": (REQUIRED_PREFIX_ROWS - 21, REQUIRED_PREFIX_ROWS - 1),
    }


def calculate_market_feature_components(
    prefix_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Calculate the six market numbers without creating a persisted row.

    Required missing, duplicate, noncanonical, nonfinite, or nonpositive values
    return explicit unavailable numeric components.  They are never imputed.
    This function deliberately returns no schema identity or canonical hash.
    """

    if isinstance(prefix_rows, (str, bytes)) or not isinstance(
        prefix_rows, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market prefix rows must be a sequence"
        )
    rows = list(prefix_rows)
    if len(rows) != REQUIRED_PREFIX_ROWS:
        return _unavailable_market_components(
            [f"required_prefix_row_count:{len(rows)}"]
        )

    sessions: list[str] = []
    normalized_rows: list[Mapping[str, Any]] = []
    structural_reasons: list[str] = []
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            structural_reasons.append(f"row_not_mapping:{index}")
            continue
        session = raw_row.get("session")
        observations = raw_row.get("observations")
        if not isinstance(session, str):
            structural_reasons.append(f"session_invalid:{index}")
        else:
            sessions.append(session)
        if not isinstance(observations, Mapping):
            structural_reasons.append(f"observations_invalid:{index}")
        normalized_rows.append(raw_row)
    if structural_reasons:
        return _unavailable_market_components(structural_reasons)
    if sessions != sorted(sessions):
        return _unavailable_market_components(
            ["market_sessions_not_increasing"]
        )
    duplicates = sorted(
        session for session in set(sessions) if sessions.count(session) > 1
    )
    if duplicates:
        return _unavailable_market_components(
            [f"duplicate_session:{session}" for session in duplicates]
        )

    required_positions = _required_market_positions()
    values: dict[tuple[str, int], float] = {}
    missing: list[str] = []
    for symbol, positions in required_positions.items():
        for position in positions:
            row = normalized_rows[position]
            session = str(row["session"])
            observations = _mapping(
                row["observations"], f"market row {position}.observations"
            )
            observation = observations.get(symbol)
            if not isinstance(observation, Mapping):
                missing.append(f"{symbol}@{session}:missing_observation")
                continue
            if observation.get("available") is not True:
                missing.append(f"{symbol}@{session}:unavailable")
                continue
            adjusted_close = _decode_positive_float_hex(
                observation.get("adjusted_close_hex")
            )
            if adjusted_close is None:
                missing.append(
                    f"{symbol}@{session}:invalid_adjusted_close"
                )
                continue
            values[(symbol, position)] = adjusted_close
    if missing:
        return _unavailable_market_components(missing)

    t = REQUIRED_PREFIX_ROWS - 1
    t20 = t - 20
    try:
        aapl_log_return = math.log(
            values[("AAPL", t)] / values[("AAPL", t20)]
        )
        qqq_log_return = math.log(
            values[("QQQ", t)] / values[("QQQ", t20)]
        )
        drawdown_window = [
            values[("AAPL", position)] for position in range(t - 62, t + 1)
        ]
        drawdown = values[("AAPL", t)] / max(drawdown_window) - 1.0
        aapl_returns = [
            math.log(
                values[("AAPL", position)]
                / values[("AAPL", position - 1)]
            )
            for position in range(t - 19, t + 1)
        ]
        mean_return = sum(aapl_returns) / len(aapl_returns)
        sample_variance = sum(
            (item - mean_return) ** 2 for item in aapl_returns
        ) / (len(aapl_returns) - 1)
        realized_volatility = math.sqrt(sample_variance) * math.sqrt(252.0)
        feature_values = {
            "aapl_minus_qqq_log_return_20": (
                aapl_log_return - qqq_log_return
            ),
            "aapl_drawdown_63": drawdown,
            "aapl_realized_volatility_20": realized_volatility,
            "spy_log_return_20": math.log(
                values[("SPY", t)] / values[("SPY", t20)]
            ),
            "iwm_log_return_20": math.log(
                values[("IWM", t)] / values[("IWM", t20)]
            ),
            "vix_log_change_20": math.log(
                values[("VIX", t)] / values[("VIX", t20)]
            ),
        }
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market feature arithmetic is non-finite"
        ) from exc
    if any(not math.isfinite(value) for value in feature_values.values()):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market feature arithmetic is non-finite"
        )
    if tuple(feature_values) != MARKET_FEATURES:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market feature order differs from preregistration"
        )
    return {
        "available": True,
        "values": feature_values,
        "unavailable_reasons": [],
    }


def _validate_extractor_mapping(output: Mapping[str, Any]) -> None:
    if output.get("document_quality") not in DOCUMENT_QUALITIES:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Validated extractor output has an invalid document quality"
        )
    dimensions = _mapping(output.get("dimensions"), "extractor dimensions")
    flags = _mapping(output.get("flags"), "extractor flags")
    if set(dimensions) != set(DIMENSION_NAMES):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Validated extractor dimensions changed"
        )
    if not set(ADVERSE_FLAG_NAMES).issubset(flags):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Validated extractor adverse flags changed"
        )
    for name in DIMENSION_NAMES:
        dimension = _mapping(dimensions[name], f"dimension {name}")
        if dimension.get("current_impact") not in CURRENT_IMPACT_ENCODING:
            raise SecGemmaOnlineRiskOverlayFeatureError(
                f"Dimension {name} current impact changed"
            )
        if dimension.get("change_vs_prior") not in CHANGE_ENCODING:
            raise SecGemmaOnlineRiskOverlayFeatureError(
                f"Dimension {name} comparative change changed"
            )
    for name in ADVERSE_FLAG_NAMES:
        present = _mapping(flags[name], f"flag {name}").get("present")
        if not isinstance(present, bool):
            raise SecGemmaOnlineRiskOverlayFeatureError(
                f"Flag {name} presence must be boolean"
            )


def calculate_semantic_feature_components(
    *,
    extraction_status: str,
    extraction_authenticated: bool,
    document_quality: str | None,
    validated_output: Mapping[str, Any] | None,
    has_prior_same_form: bool,
) -> dict[str, Any]:
    """Calculate filing meaning without creating a persisted feature row."""

    if not isinstance(extraction_authenticated, bool):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Extraction authentication must be boolean"
        )
    if not isinstance(has_prior_same_form, bool):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Prior same-form availability must be boolean"
        )
    if extraction_status not in {"valid", "invalid", "unavailable"}:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Extraction status is invalid"
        )
    neutral = {name: 0.0 for name in MEANING_FEATURES}
    if not extraction_authenticated:
        if validated_output is not None or document_quality is not None:
            raise SecGemmaOnlineRiskOverlayFeatureError(
                "Unauthenticated extraction cannot expose semantic output"
            )
        return {
            "extraction_available": False,
            "schema_valid_extraction": False,
            "meaning_values": neutral,
            "semantic_quality_risk": 1.0,
        }
    if extraction_status == "unavailable":
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Authenticated extraction cannot have unavailable status"
        )
    if extraction_status == "invalid":
        if validated_output is not None or document_quality is not None:
            raise SecGemmaOnlineRiskOverlayFeatureError(
                "Invalid extraction cannot expose semantic output"
            )
        return {
            "extraction_available": True,
            "schema_valid_extraction": False,
            "meaning_values": neutral,
            "semantic_quality_risk": 1.0,
        }
    if validated_output is None:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Authenticated valid extraction is missing its output"
        )

    _validate_extractor_mapping(validated_output)
    quality = validated_output["document_quality"]
    if quality != document_quality:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Extractor document quality differs from authenticated proof"
        )
    if quality == "unusable":
        return {
            "extraction_available": True,
            "schema_valid_extraction": True,
            "meaning_values": neutral,
            "semantic_quality_risk": 1.0,
        }

    dimensions = validated_output["dimensions"]
    if not has_prior_same_form and any(
        dimensions[name]["change_vs_prior"] != "not_comparable"
        for name in DIMENSION_NAMES
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "First same-form filing must mark every comparative change "
            "not_comparable"
        )
    dimension_scores: dict[str, float] = {}
    for name in DIMENSION_NAMES:
        dimension = dimensions[name]
        current_component = CURRENT_IMPACT_ENCODING[
            dimension["current_impact"]
        ]
        change_component = (
            0.0
            if not has_prior_same_form
            else CHANGE_ENCODING[dimension["change_vs_prior"]]
        )
        dimension_scores[name] = (
            current_component + change_component
        ) / 2.0
    meaning: dict[str, float] = {}
    for output_name, members in SEMANTIC_GROUPS.items():
        meaning[output_name] = sum(
            dimension_scores[name] for name in members
        ) / len(members)
    flags = validated_output["flags"]
    meaning["adverse_flag_fraction"] = (
        sum(bool(flags[name]["present"]) for name in ADVERSE_FLAG_NAMES)
        / len(ADVERSE_FLAG_NAMES)
    )
    if tuple(meaning) != MEANING_FEATURES:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Filing-meaning feature order differs from preregistration"
        )
    return {
        "extraction_available": True,
        "schema_valid_extraction": True,
        "meaning_values": meaning,
        "semantic_quality_risk": DOCUMENT_QUALITY_RISK[quality],
    }


def _market_result_from_components(
    components: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(components, "market feature components")
    if set(value) != {"available", "values", "unavailable_reasons"}:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market feature component keys changed"
        )
    available = value["available"]
    if not isinstance(available, bool):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market feature availability must be boolean"
        )
    values = _mapping(value["values"], "market component values")
    if set(values) != set(MARKET_FEATURES):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market component value names changed"
        )
    reasons = value["unavailable_reasons"]
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in reasons)
        or reasons != sorted(set(reasons))
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market component unavailable reasons changed"
        )
    if available:
        if reasons or any(values[name] is None for name in MARKET_FEATURES):
            raise SecGemmaOnlineRiskOverlayFeatureError(
                "Available market components are incomplete"
            )
        values_hex = {
            name: _canonical_float_hex(values[name], name)
            for name in MARKET_FEATURES
        }
    else:
        if not reasons or any(
            values[name] is not None for name in MARKET_FEATURES
        ):
            raise SecGemmaOnlineRiskOverlayFeatureError(
                "Unavailable market components expose values"
            )
        values_hex = {name: None for name in MARKET_FEATURES}
    body = {
        "schema_version": MARKET_RESULT_SCHEMA_VERSION,
        "available": available,
        "feature_names": list(MARKET_FEATURES),
        "values_hex": values_hex,
        "unavailable_reasons": list(reasons),
        "unavailable_reasons_sha256": canonical_sha256(reasons),
    }
    return {**body, "market_result_sha256": canonical_sha256(body)}


def _market_result_from_anomalies(
    anomalies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reasons = [
        (
            f"{anomaly['symbol']}@{anomaly['session']}:"
            f"{anomaly['field']}:{anomaly['reason']}"
        )
        for anomaly in anomalies
    ]
    return _market_result_from_components(
        _unavailable_market_components(reasons)
    )


def _vector_hex(
    market_values_hex: Mapping[str, str | None],
    meaning: Mapping[str, float],
    quality: float,
    form_10k: float,
    *,
    zero_meaning: bool,
    zero_quality: bool,
) -> list[str]:
    values: dict[str, float] = {
        name: float.fromhex(str(market_values_hex[name]))
        for name in MARKET_FEATURES
    }
    values.update(
        {
            name: 0.0 if zero_meaning else float(meaning[name])
            for name in MEANING_FEATURES
        }
    )
    values[QUALITY_FEATURE] = 0.0 if zero_quality else quality
    values[CONTROL_FEATURES[0]] = form_10k
    if tuple(values) != FEATURES:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Feature vector order differs from preregistration"
        )
    return [
        _canonical_float_hex(values[name], name) for name in FEATURES
    ]


def _validate_row_identity(
    *,
    accession_number: Any,
    form: Any,
    decision_session: Any,
    acceptance_datetime: Any,
    artifact_stage: Any,
) -> None:
    if not isinstance(accession_number, str) or not accession_number:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Accession number is invalid"
        )
    if form not in {"10-K", "10-Q"}:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Filing form is invalid"
        )
    if not isinstance(decision_session, str) or not decision_session:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Decision session is invalid"
        )
    if acceptance_datetime is not None and (
        not isinstance(acceptance_datetime, str)
        or len(acceptance_datetime) != 14
        or not acceptance_datetime.isascii()
        or not acceptance_datetime.isdigit()
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Acceptance timestamp is invalid"
        )
    if not isinstance(artifact_stage, str) or not artifact_stage:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Artifact stage is invalid"
        )


def _feature_payload_components(
    *,
    form: str,
    market: Mapping[str, Any],
    extraction_status: str,
    extraction_authenticated: bool,
    document_quality: str | None,
    validated_output: Mapping[str, Any] | None,
    has_prior_same_form: bool,
) -> dict[str, Any]:
    semantic = calculate_semantic_feature_components(
        extraction_status=extraction_status,
        extraction_authenticated=extraction_authenticated,
        document_quality=document_quality,
        validated_output=validated_output,
        has_prior_same_form=has_prior_same_form,
    )
    extraction_available = semantic["extraction_available"]
    schema_valid = semantic["schema_valid_extraction"]
    meaning = semantic["meaning_values"]
    quality = semantic["semantic_quality_risk"]
    unavailable_reasons: list[str] = []
    if not market["available"]:
        unavailable_reasons.append("market_unavailable")
    if not extraction_available:
        unavailable_reasons.append(
            "missing_or_unauthenticated_extraction_evidence"
        )
    prediction_available = not unavailable_reasons
    meaning_nonzero = schema_valid and any(
        abs(value) > 1e-12 for value in meaning.values()
    )
    form_10k = float(form == "10-K")
    semantic_vector: list[str] | None = None
    no_meaning_vector: list[str] | None = None
    no_gemma_vector: list[str] | None = None
    if prediction_available:
        semantic_vector = _vector_hex(
            market["values_hex"],
            meaning,
            quality,
            form_10k,
            zero_meaning=False,
            zero_quality=False,
        )
        no_meaning_vector = _vector_hex(
            market["values_hex"],
            meaning,
            quality,
            form_10k,
            zero_meaning=True,
            zero_quality=False,
        )
        no_gemma_vector = _vector_hex(
            market["values_hex"],
            meaning,
            quality,
            form_10k,
            zero_meaning=True,
            zero_quality=True,
        )
    return {
        "extraction_status": extraction_status,
        "extraction_authenticated": extraction_authenticated,
        "schema_valid_extraction": schema_valid,
        "document_quality": document_quality,
        "market_available": market["available"],
        "prediction_available": prediction_available,
        "fit_eligible": prediction_available,
        "meaning_nonzero": meaning_nonzero,
        "unavailable_reasons": unavailable_reasons,
        "unavailable_reasons_sha256": canonical_sha256(
            unavailable_reasons
        ),
        "market_result": copy.deepcopy(dict(market)),
        "market_result_sha256": market["market_result_sha256"],
        "meaning_feature_names": list(MEANING_FEATURES),
        "meaning_values_hex": {
            name: _canonical_float_hex(meaning[name], name)
            for name in MEANING_FEATURES
        },
        "semantic_quality_risk_hex": _canonical_float_hex(
            quality, QUALITY_FEATURE
        ),
        "form_10k_hex": _canonical_float_hex(
            form_10k, CONTROL_FEATURES[0]
        ),
        "feature_names": list(FEATURES),
        "semantic_values_hex": semantic_vector,
        "no_filing_meaning_values_hex": no_meaning_vector,
        "no_gemma_channel_values_hex": no_gemma_vector,
    }


def _mint_feature_row_from_source_bound_components(
    *,
    accession_number: str,
    form: str,
    decision_session: str,
    acceptance_datetime: str | None,
    artifact_stage: str,
    market_lookback_rows: Sequence[Mapping[str, Any]],
    extraction_status: str,
    extraction_authenticated: bool,
    document_quality: str | None,
    validated_output: Mapping[str, Any] | None,
    has_prior_same_form: bool,
    upstream_bindings: Mapping[str, Any],
    market_unavailable_authenticated: bool = False,
) -> dict[str, Any]:
    """Mint one row from already authenticated, source-bound components.

    This entry point performs no I/O and grants no authority by itself.  It is
    intended for the reviewed production worker after the opaque acquisition
    vault has released a cutoff-safe market slice, a validated universe proof,
    and one sealed extractor result.  The older proof-replay entry point below
    delegates here after validating its complete public evidence chain.
    """

    _validate_row_identity(
        accession_number=accession_number,
        form=form,
        decision_session=decision_session,
        acceptance_datetime=acceptance_datetime,
        artifact_stage=artifact_stage,
    )
    if not isinstance(has_prior_same_form, bool):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Prior same-form availability must be Boolean"
        )
    if not isinstance(market_unavailable_authenticated, bool):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market-unavailable authentication must be Boolean"
        )
    bindings = copy.deepcopy(dict(_mapping(
        upstream_bindings, "feature upstream bindings"
    )))
    market_components = calculate_market_feature_components(
        market_lookback_rows
    )
    if (
        not market_components["available"]
        and not market_unavailable_authenticated
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Authenticated market anomaly requires the unavailable evidence "
            "entry point"
        )
    market = _market_result_from_components(market_components)
    payload = _feature_payload_components(
        form=form,
        market=market,
        extraction_status=extraction_status,
        extraction_authenticated=extraction_authenticated,
        document_quality=document_quality,
        validated_output=validated_output,
        has_prior_same_form=has_prior_same_form,
    )
    body = {
        "schema_version": FEATURE_ROW_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "accession_number": accession_number,
        "form": form,
        "decision_session": decision_session,
        "acceptance_datetime": acceptance_datetime,
        "artifact_stage": artifact_stage,
        **payload,
        "upstream_bindings": bindings,
        "upstream_bindings_sha256": canonical_sha256(bindings),
    }
    return {**body, "feature_row_sha256": canonical_sha256(body)}


def build_sec_gemma_online_risk_overlay_feature_row(
    *,
    market_prefix: Mapping[str, Any],
    market_prefix_proof: Mapping[str, Any],
    expected_market_prefix_proof_sha256: str,
    universe_event_proof: Mapping[str, Any],
    expected_universe_event_proof_sha256: str,
    extraction_event_proof: Mapping[str, Any],
    expected_extraction_event_proof_sha256: str,
) -> dict[str, Any]:
    """Replay the pinned normal evidence chain and mint one feature row."""

    market_proof = validate_market_prefix_proof(
        market_prefix_proof,
        prefix=market_prefix,
        expected_market_prefix_proof_sha256=(
            expected_market_prefix_proof_sha256
        ),
    )
    prefix = market_proof["compact_prefix"]
    universe = validate_universe_event_proof(
        universe_event_proof,
        expected_universe_event_proof_sha256=(
            expected_universe_event_proof_sha256
        ),
    )
    extraction = validate_extraction_event_proof(
        extraction_event_proof,
        universe_event_proof=universe_event_proof,
        expected_universe_event_proof_sha256=(
            expected_universe_event_proof_sha256
        ),
        expected_extraction_event_proof_sha256=(
            expected_extraction_event_proof_sha256
        ),
    )
    current = universe["current_record"]
    if (
        current["accession_number"] != prefix["decision_event_id"]
        or current["availability_session"] != prefix["decision_session"]
        or current["artifact_stage"] != prefix["artifact_stage"]
        or extraction["accession_number"] != current["accession_number"]
        or extraction["decision_session"] != current["availability_session"]
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Market, universe, and extraction proofs refer to different events"
        )
    bindings = {
        "market_prefix_sha256": prefix["market_prefix_sha256"],
        "market_prefix_proof_sha256": market_proof[
            "market_prefix_proof_sha256"
        ],
        "market_prefix_chain_identity_sha256": prefix[
            "full_prefix_chain_identity_sha256"
        ],
        "universe_event_proof_sha256": universe[
            "universe_event_proof_sha256"
        ],
        "current_universe_record_sha256": universe[
            "current_record_sha256"
        ],
        "current_filing_sha256": universe["current_filing_sha256"],
        "prior_same_form_filing_sha256": universe[
            "prior_same_form_filing_sha256"
        ],
        "extraction_event_proof_sha256": extraction[
            "extraction_event_proof_sha256"
        ],
        "extraction_evidence_sha256": extraction[
            "extraction_evidence_sha256"
        ],
        "extraction_output_sha256": extraction[
            "extraction_output_sha256"
        ],
        "extraction_output_canonical_sha256": extraction[
            "extraction_output_canonical_sha256"
        ],
    }
    return _mint_feature_row_from_source_bound_components(
        accession_number=current["accession_number"],
        form=current["form"],
        decision_session=current["availability_session"],
        acceptance_datetime=current["acceptance_datetime"],
        artifact_stage=current["artifact_stage"],
        market_lookback_rows=prefix["lookback_rows"],
        extraction_status=extraction["extraction_status"],
        extraction_authenticated=extraction[
            "extraction_evidence_authenticated"
        ],
        document_quality=extraction["document_quality"],
        validated_output=extraction["validated_output"],
        has_prior_same_form=universe["prior_same_form_record"] is not None,
        upstream_bindings=bindings,
    )


def build_sec_gemma_online_risk_overlay_unavailable_feature_row(
    *,
    market_source_manifest: Mapping[str, Any],
    expected_market_source_manifest_sha256: str,
    market_acquisition_receipt: Mapping[str, Any],
    expected_market_acquisition_receipt_sha256: str,
    market_verifier_receipt: Mapping[str, Any],
    expected_market_verifier_receipt_sha256: str,
    market_unavailable_event_proof: Mapping[str, Any],
    expected_market_unavailable_event_proof_sha256: str,
    universe_event_proof: Mapping[str, Any],
    expected_universe_event_proof_sha256: str,
    extraction_event_proof: Mapping[str, Any],
    expected_extraction_event_proof_sha256: str,
) -> dict[str, Any]:
    """Replay a market-anomaly chain and mint an audit-only feature row."""

    unavailable = validate_market_unavailable_event_proof(
        market_unavailable_event_proof,
        source_manifest=market_source_manifest,
        expected_source_manifest_sha256=(
            expected_market_source_manifest_sha256
        ),
        acquisition_receipt=market_acquisition_receipt,
        expected_acquisition_receipt_sha256=(
            expected_market_acquisition_receipt_sha256
        ),
        verifier_receipt=market_verifier_receipt,
        expected_verifier_receipt_sha256=(
            expected_market_verifier_receipt_sha256
        ),
        expected_market_unavailable_event_proof_sha256=(
            expected_market_unavailable_event_proof_sha256
        ),
    )
    universe = validate_universe_event_proof(
        universe_event_proof,
        expected_universe_event_proof_sha256=(
            expected_universe_event_proof_sha256
        ),
    )
    extraction = validate_extraction_event_proof(
        extraction_event_proof,
        universe_event_proof=universe_event_proof,
        expected_universe_event_proof_sha256=(
            expected_universe_event_proof_sha256
        ),
        expected_extraction_event_proof_sha256=(
            expected_extraction_event_proof_sha256
        ),
    )
    current = universe["current_record"]
    if (
        unavailable["accession_number"] != current["accession_number"]
        or unavailable["decision_session"] != current["availability_session"]
        or unavailable["artifact_stage"] != current["artifact_stage"]
        or extraction["accession_number"] != current["accession_number"]
        or extraction["decision_session"] != current["availability_session"]
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Unavailable market proof refers to another filing event"
        )
    _validate_row_identity(
        accession_number=current["accession_number"],
        form=current["form"],
        decision_session=current["availability_session"],
        acceptance_datetime=current["acceptance_datetime"],
        artifact_stage=current["artifact_stage"],
    )
    market = _market_result_from_anomalies(unavailable["anomalies"])
    payload = _feature_payload_components(
        form=current["form"],
        market=market,
        extraction_status=extraction["extraction_status"],
        extraction_authenticated=extraction[
            "extraction_evidence_authenticated"
        ],
        document_quality=extraction["document_quality"],
        validated_output=extraction["validated_output"],
        has_prior_same_form=universe["prior_same_form_record"] is not None,
    )
    bindings = {
        "market_unavailable_event_proof_sha256": unavailable[
            "market_unavailable_event_proof_sha256"
        ],
        "market_source_manifest_sha256": unavailable[
            "source_manifest_sha256"
        ],
        "market_acquisition_receipt_sha256": unavailable[
            "acquisition_receipt_sha256"
        ],
        "market_verifier_receipt_sha256": unavailable[
            "verifier_receipt_sha256"
        ],
        "market_anomalies_sha256": unavailable["anomalies_sha256"],
        "universe_event_proof_sha256": universe[
            "universe_event_proof_sha256"
        ],
        "current_universe_record_sha256": universe[
            "current_record_sha256"
        ],
        "current_filing_sha256": universe["current_filing_sha256"],
        "prior_same_form_filing_sha256": universe[
            "prior_same_form_filing_sha256"
        ],
        "extraction_event_proof_sha256": extraction[
            "extraction_event_proof_sha256"
        ],
        "extraction_evidence_sha256": extraction[
            "extraction_evidence_sha256"
        ],
        "extraction_output_sha256": extraction[
            "extraction_output_sha256"
        ],
        "extraction_output_canonical_sha256": extraction[
            "extraction_output_canonical_sha256"
        ],
    }
    body = {
        "schema_version": FEATURE_ROW_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "accession_number": current["accession_number"],
        "form": current["form"],
        "decision_session": current["availability_session"],
        "acceptance_datetime": current["acceptance_datetime"],
        "artifact_stage": current["artifact_stage"],
        **payload,
        "upstream_bindings": copy.deepcopy(bindings),
        "upstream_bindings_sha256": canonical_sha256(bindings),
    }
    return {**body, "feature_row_sha256": canonical_sha256(body)}


def validate_sec_gemma_online_risk_overlay_feature_row(
    row: Mapping[str, Any],
    *,
    expected_feature_row_sha256: str,
    market_prefix: Mapping[str, Any],
    market_prefix_proof: Mapping[str, Any],
    expected_market_prefix_proof_sha256: str,
    universe_event_proof: Mapping[str, Any],
    expected_universe_event_proof_sha256: str,
    extraction_event_proof: Mapping[str, Any],
    expected_extraction_event_proof_sha256: str,
) -> str:
    """Rebuild one normal feature row and require byte-level identity."""

    if not isinstance(row, Mapping):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Feature row must be a mapping"
        )
    rebuilt = build_sec_gemma_online_risk_overlay_feature_row(
        market_prefix=market_prefix,
        market_prefix_proof=market_prefix_proof,
        expected_market_prefix_proof_sha256=(
            expected_market_prefix_proof_sha256
        ),
        universe_event_proof=universe_event_proof,
        expected_universe_event_proof_sha256=(
            expected_universe_event_proof_sha256
        ),
        extraction_event_proof=extraction_event_proof,
        expected_extraction_event_proof_sha256=(
            expected_extraction_event_proof_sha256
        ),
    )
    if dict(row) != rebuilt:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Feature row differs from authenticated proof replay"
        )
    observed = row.get("feature_row_sha256")
    if not isinstance(observed, str) or not hmac.compare_digest(
        observed, expected_feature_row_sha256
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Feature row is not externally pinned"
        )
    return observed


def validate_sec_gemma_online_risk_overlay_unavailable_feature_row(
    row: Mapping[str, Any],
    *,
    expected_feature_row_sha256: str,
    market_source_manifest: Mapping[str, Any],
    expected_market_source_manifest_sha256: str,
    market_acquisition_receipt: Mapping[str, Any],
    expected_market_acquisition_receipt_sha256: str,
    market_verifier_receipt: Mapping[str, Any],
    expected_market_verifier_receipt_sha256: str,
    market_unavailable_event_proof: Mapping[str, Any],
    expected_market_unavailable_event_proof_sha256: str,
    universe_event_proof: Mapping[str, Any],
    expected_universe_event_proof_sha256: str,
    extraction_event_proof: Mapping[str, Any],
    expected_extraction_event_proof_sha256: str,
) -> str:
    """Replay the unavailable chain and require byte-level row identity."""

    if not isinstance(row, Mapping):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Feature row must be a mapping"
        )
    rebuilt = build_sec_gemma_online_risk_overlay_unavailable_feature_row(
        market_source_manifest=market_source_manifest,
        expected_market_source_manifest_sha256=(
            expected_market_source_manifest_sha256
        ),
        market_acquisition_receipt=market_acquisition_receipt,
        expected_market_acquisition_receipt_sha256=(
            expected_market_acquisition_receipt_sha256
        ),
        market_verifier_receipt=market_verifier_receipt,
        expected_market_verifier_receipt_sha256=(
            expected_market_verifier_receipt_sha256
        ),
        market_unavailable_event_proof=market_unavailable_event_proof,
        expected_market_unavailable_event_proof_sha256=(
            expected_market_unavailable_event_proof_sha256
        ),
        universe_event_proof=universe_event_proof,
        expected_universe_event_proof_sha256=(
            expected_universe_event_proof_sha256
        ),
        extraction_event_proof=extraction_event_proof,
        expected_extraction_event_proof_sha256=(
            expected_extraction_event_proof_sha256
        ),
    )
    if dict(row) != rebuilt:
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Unavailable feature row differs from authenticated proof replay"
        )
    observed = row.get("feature_row_sha256")
    if not isinstance(observed, str) or not hmac.compare_digest(
        observed, expected_feature_row_sha256
    ):
        raise SecGemmaOnlineRiskOverlayFeatureError(
            "Unavailable feature row is not externally pinned"
        )
    return observed


__all__ = [
    "CHANGE_ENCODING",
    "CURRENT_IMPACT_ENCODING",
    "DOCUMENT_QUALITY_RISK",
    "FEATURE_ROW_SCHEMA_VERSION",
    "MARKET_RESULT_SCHEMA_VERSION",
    "MARKET_UNAVAILABLE_PROOF_SCHEMA_VERSION",
    "MARKET_UNAVAILABLE_REASONS",
    "REQUIRED_MARKET_SYMBOLS",
    "REQUIRED_PREFIX_ROWS",
    "SEMANTIC_GROUPS",
    "SecGemmaOnlineRiskOverlayFeatureError",
    "build_validated_universe_event_proof",
    "build_sec_gemma_online_risk_overlay_feature_row",
    "build_sec_gemma_online_risk_overlay_unavailable_feature_row",
    "calculate_market_feature_components",
    "calculate_semantic_feature_components",
    "validate_extraction_event_proof",
    "validate_market_prefix_proof",
    "validate_sec_gemma_online_risk_overlay_feature_row",
    "validate_sec_gemma_online_risk_overlay_unavailable_feature_row",
    "validate_universe_event_proof",
]
