"""Frozen feature and isolated label rows for the SEC/Gemma experiment.

The feature path consumes only one already sealed 253-row market prefix ending
at decision close ``t``.  It cannot receive an outcome row.  The label path is
a separate callable which derives the exact chained rows ``t+1`` through
``t+21`` from a validated stage frame and is never called by the feature
builder.

All persisted numbers use canonical :meth:`float.hex` text.  Missing market
history is represented explicitly; it is never filled, interpolated, or
backfilled.  Every eligible filing therefore has an audit row even when its
feature vector is unavailable to the learner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from datetime import date, datetime
import hashlib
import hmac
import json
import math
import re
from typing import Any, Final

import numpy as np
import pandas as pd

from agent_benchmark.direct_edge_features import (
    MARKET_SENTIMENT_FEATURE_COLUMNS,
    build_market_sentiment_features,
)
from agent_benchmark.downside_features import (
    PRICE_FEATURE_COLUMNS,
    build_downside_price_features,
)
from agent_benchmark.sec_filing_gemma_contract import (
    ACTIVE_EDGE_TOLERANCE,
    ADVERSE_FLAG_NAMES,
    CONTRACT_VERSION,
    DIMENSION_NAMES,
    DOCUMENT_QUALITIES,
    FLAG_NAMES,
    HORIZON_SESSIONS,
    MAX_SENTENCES,
    STAGE_ORDER,
    STAGE_WINDOWS,
    SecFilingGemmaContractError,
    build_contract_manifest,
    canonical_sha256,
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
    validate_decision_market_prefix,
    validate_market_stage_manifest,
)
from agent_benchmark.sec_session_calendar import (
    EXPECTED_MARKET_HISTORY_SESSIONS,
    EXPECTED_SESSIONS,
    MARKET_HISTORY_CALENDAR_ID,
)


FEATURE_ROW_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-feature-row-v1"
OWNED_DEVELOPMENT_FEATURE_INPUTS_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-feature-inputs-v1"
)
OWNED_DEVELOPMENT_FEATURE_BATCH_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-feature-batch-v1"
)
OWNED_DEVELOPMENT_LABEL_PROJECTION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-label-projection-v1"
)
OWNED_DEVELOPMENT_LABEL_BATCH_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-label-batch-v1"
)
DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-feature-assembly-plan-v1"
)
DEVELOPMENT_LABEL_ASSEMBLY_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-label-assembly-plan-v1"
)
MARKET_FEATURE_ROW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-market-feature-row-v1"
)
LABEL_EVIDENCE_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-20-session-label-v1"
MARKET_PREFIX_PROOF_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-validated-market-prefix-proof-v1"
)
UNIVERSE_EVENT_PROOF_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-validated-universe-event-proof-v1"
)
EXTRACTION_EVENT_PROOF_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-validated-extraction-event-proof-v1"
)

FILING_CALENDAR_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "form_is_10k",
    "sessions_since_prior_same_form",
    "semantic_output_unavailable",
)
SEMANTIC_AGGREGATE_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "commercial_current_mean",
    "commercial_change_mean",
    "financial_current_mean",
    "financial_change_mean",
    "risk_and_outlook_current_mean",
    "risk_and_outlook_change_mean",
    "adverse_flag_count",
    "management_transition_flag",
    "stated_dimension_fraction",
    "comparable_dimension_fraction",
    "document_usable",
    "document_thin",
)
SEMANTIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *PRICE_FEATURE_COLUMNS,
    *MARKET_SENTIMENT_FEATURE_COLUMNS,
    *FILING_CALENDAR_FEATURE_COLUMNS,
    *SEMANTIC_AGGREGATE_FEATURE_COLUMNS,
)
# The ablation has the identical schema and missingness indicators.  Only the
# 12 semantic-content values are replaced by exact zeros.
ABLATION_FEATURE_COLUMNS: Final[tuple[str, ...]] = SEMANTIC_FEATURE_COLUMNS

DIMENSION_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "commercial": ("demand", "pricing_power", "gross_margin"),
    "financial": (
        "operating_cost_pressure",
        "capital_allocation",
        "liquidity",
    ),
    "risk_and_outlook": (
        "forward_guidance",
        "supply_chain",
        "legal_regulatory",
        "management_uncertainty",
    ),
}
CURRENT_IMPACT_ENCODING: Final[dict[str, float]] = {
    "favorable": 1.0,
    "neutral": 0.0,
    "unfavorable": -1.0,
    "mixed": 0.0,
    "not_stated": 0.0,
}
COMPARATIVE_CHANGE_ENCODING: Final[dict[str, float]] = {
    "improving": 1.0,
    "stable": 0.0,
    "deteriorating": -1.0,
    "mixed": 0.0,
    "not_comparable": 0.0,
    "not_stated": 0.0,
}
EXTRACTION_STATUSES: Final[frozenset[str]] = frozenset(
    {"valid", "invalid", "unavailable"}
)
FEATURE_UNAVAILABLE_REASONS: Final[frozenset[str]] = frozenset(
    {
        "missing_required_market_history",
        "missing_or_unauthenticated_extraction_evidence",
    }
)
TRANSACTION_COST_BPS: Final[tuple[int, int]] = (5, 10)
LABEL_ENTRY_OFFSET: Final[int] = 1
LABEL_MATURITY_OFFSET: Final[int] = HORIZON_SESSIONS + 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ACCESSION_RE = re.compile(r"[0-9]{10}-[0-9]{2}-[0-9]{6}\Z")
_ACCEPTANCE_RE = re.compile(r"[0-9]{14}\Z")
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
_SENTENCE_ID_RE = re.compile(r"[CP][0-9]{4}\Z")
_MODEL_EVENT_PLAN_ITEM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "event_ordinal",
        "accession_number",
        "form",
        "availability_session",
        "sec_document_ordinal",
    }
)
_DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "plan_kind",
        "artifact_stage",
        "development_root_scope_sha256",
        "development_content_root_plan_sha256",
        "candidate_sha256",
        "candidate_design_sha256",
        "corpus_universe_sha256",
        "development_cutoff_session",
        "start_consumed_request_count",
        "development_sec_execution_claim_sha256",
        "development_sec_reader_receipt_sha256",
        "development_market_execution_claim_sha256",
        "development_market_reader_receipt_sha256",
        "development_market_acquisition_receipt_sha256",
        "development_market_acquisition_bundle_sha256",
        "development_market_acquisition_validation_sha256",
        "development_market_source_manifest_sha256",
        "development_market_stage_manifest_sha256",
        "development_market_source_reconciliation_sha256",
        "development_market_byte_index_sha256",
        "development_model_execution_claim_sha256",
        "development_model_reader_receipt_sha256",
        "event_count",
        "event_plan",
        "event_plan_sha256",
        "execution_source_hashes_sha256",
        "canonical_market_rows_required",
        "raw_market_output_permitted",
        "normalized_filing_text_output_permitted",
        "model_transport_envelope_output_permitted",
        "feature_rows_output_permitted",
        "outcome_access_permitted",
        "label_access_permitted",
        "training_membership_access_permitted",
        "learner_fit_permitted",
        "prediction_access_permitted",
        "holdout_access_permitted",
        "ledger_mutation_permitted",
        "stage_promotion_permitted",
        "feature_assembly_plan_sha256",
    }
)
_FEATURE_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "accession_number",
        "form",
        "artifact_stage",
        "decision_session",
        "extraction_status",
        "extraction_evidence_authenticated",
        "document_quality",
        "semantic_available",
        "market_available",
        "prediction_available",
        "fit_eligible",
        "unavailable_reasons",
        "missing_market_observations",
        "missing_market_observations_sha256",
        "bindings",
        "bindings_sha256",
        "market_feature_row_sha256",
        "price_regime_features_hex",
        "market_sentiment_features_hex",
        "filing_calendar_features_hex",
        "semantic_aggregate_features_hex",
        "semantic_feature_names",
        "semantic_feature_schema_sha256",
        "semantic_feature_values_hex",
        "ablation_feature_names",
        "ablation_feature_schema_sha256",
        "ablation_feature_values_hex",
        "feature_row_sha256",
    }
)
_FEATURE_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "contract_sha256",
        "corpus_universe_sha256",
        "universe_event_proof_sha256",
        "current_universe_record_sha256",
        "prior_same_form_universe_record_sha256",
        "current_source_record_sha256",
        "prior_same_form_source_record_sha256",
        "current_filing_sha256",
        "prior_same_form_filing_sha256",
        "extraction_event_proof_sha256",
        "extraction_identity_sha256",
        "extraction_evidence_sha256",
        "extraction_output_sha256",
        "extraction_output_canonical_sha256",
        "market_prefix_sha256",
        "market_prefix_proof_sha256",
        "market_stage_manifest_sha256",
        "source_manifest_sha256",
        "market_prefix_chain_identity_sha256",
    }
)
_OWNED_DEVELOPMENT_FEATURE_BATCH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "development_root_scope_sha256",
        "feature_assembly_plan_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "development_sec_reader_receipt_sha256",
        "development_market_reader_receipt_sha256",
        "development_model_reader_receipt_sha256",
        "event_count",
        "event_plan_sha256",
        "feature_row_schema_version",
        "feature_row_sha256s",
        "feature_rows_sha256",
        "feature_rows",
        "labels_included",
        "outcomes_included",
        "post_decision_market_rows_included",
        "training_membership_included",
        "learner_fit_authorized",
        "stage_promotion_authorized",
        "production_authorized",
        "feature_batch_sha256",
    }
)
_DEVELOPMENT_LABEL_MATURITY_ITEM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "event_ordinal",
        "accession_number",
        "form",
        "decision_session",
        "sec_document_ordinal",
        "label_maturity_session",
        "matured_by_development_cutoff",
    }
)
_DEVELOPMENT_LABEL_AUDIT_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "event_ordinal",
        "accession_number",
        "decision_session",
        "feature_row_sha256",
        "label_maturity_session",
        "matured_by_development_cutoff",
        "label_evidence_sha256",
    }
)
_COMPACT_ADJUSTED_OPEN_PATH_ITEM_KEYS: Final[frozenset[str]] = frozenset(
    {"session", "adjusted_open_hex", "source_market_row_sha256"}
)
_LABEL_EVIDENCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "accession_number",
        "decision_session",
        "entry_session",
        "exit_session",
        "label_maturity_session",
        "horizon_sessions",
        "entry_session_offset",
        "label_maturity_session_offset",
        "market_prefix_sha256",
        "market_prefix_proof_sha256",
        "market_stage_manifest_sha256",
        "source_manifest_sha256",
        "feature_row_sha256",
        "market_feature_row_sha256",
        "extraction_identity_sha256",
        "future_market_rows_sha256",
        "future_market_row_count",
        "future_market_row_chain_tip_sha256",
        "entry_source_market_row_sha256",
        "exit_source_market_row_sha256",
        "adjusted_open_path",
        "adjusted_open_path_sha256",
        "entry_adjusted_open_hex",
        "exit_adjusted_open_hex",
        "aapl_forward_log_return_20_hex",
        "cash_round_trip_log_cost_5bps_hex",
        "cash_round_trip_log_cost_10bps_hex",
        "cash_active_log_edge_5bps_hex",
        "cash_active_log_edge_10bps_hex",
        "cash_beats_long_5bps",
        "cash_beats_long_10bps",
        "binary_comparison_tolerance_hex",
        "label_evidence_sha256",
    }
)
_OWNED_DEVELOPMENT_LABEL_BATCH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "development_root_scope_sha256",
        "label_assembly_plan_sha256",
        "source_feature_assembly_plan_sha256",
        "source_feature_batch_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "development_market_reader_receipt_sha256",
        "development_cutoff_session",
        "event_count",
        "matured_label_count",
        "unmatured_event_count",
        "maturity_audit_rows",
        "maturity_audit_rows_sha256",
        "label_evidence_schema_version",
        "label_evidence_sha256s",
        "label_evidence_rows",
        "label_evidence_rows_sha256",
        "development_labels_included",
        "development_outcomes_included",
        "compact_adjusted_open_paths_included",
        "full_market_rows_included",
        "post_cutoff_market_data_included",
        "training_membership_included",
        "learner_fit_authorized",
        "prediction_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
        "label_batch_sha256",
    }
)


class SecFilingGemmaFeatureError(SecFilingGemmaContractError):
    """Raised when feature or label evidence is noncanonical or unbound."""


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise SecFilingGemmaFeatureError(f"{location} must be a string-keyed mapping")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SecFilingGemmaFeatureError(
            f"Invalid {location} keys; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaFeatureError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _optional_sha256(value: Any, location: str) -> str | None:
    return None if value is None else _sha256(value, location)


def _iso_date(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise SecFilingGemmaFeatureError(f"{location} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecFilingGemmaFeatureError(
            f"{location} must be a canonical ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise SecFilingGemmaFeatureError(f"{location} must use YYYY-MM-DD form")
    return value


def _strict_positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SecFilingGemmaFeatureError(f"{location} must be an integer >= 1")
    return value


def _float_hex(value: Any, location: str = "feature value") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise SecFilingGemmaFeatureError(f"{location} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise SecFilingGemmaFeatureError(f"{location} must be finite")
    return number.hex()


def _offset_session(session: str, offset: int, *, market_calendar: bool) -> str:
    calendar = (
        EXPECTED_MARKET_HISTORY_SESSIONS if market_calendar else EXPECTED_SESSIONS
    )
    try:
        position = calendar.index(session)
    except ValueError as exc:
        raise SecFilingGemmaFeatureError(
            "Session is outside the exact frozen trading calendar"
        ) from exc
    target = position + offset
    if target < 0 or target >= len(calendar):
        raise SecFilingGemmaFeatureError("Frozen calendar does not cover the offset")
    return calendar[target]


def _validate_observation(value: Any, location: str) -> dict[str, Any]:
    observation = _expect_mapping(value, location)
    _expect_keys(observation, _MARKET_OBSERVATION_KEYS, location)
    available = observation["available"]
    if not isinstance(available, bool):
        raise SecFilingGemmaFeatureError(f"{location}.available must be boolean")
    if not available:
        if any(
            observation[f"{name}_hex"] is not None
            for name in CANONICAL_MARKET_FIELDS
        ):
            raise SecFilingGemmaFeatureError(
                f"{location} unavailable observation must use explicit nulls"
            )
        return dict(observation)

    numeric = {
        name: decode_float_hex(observation[f"{name}_hex"], f"{location}.{name}")
        for name in ("open", "high", "low", "close", "adjusted_close", "volume")
    }
    if any(numeric[name] <= 0.0 for name in ("open", "high", "low", "close", "adjusted_close")):
        raise SecFilingGemmaFeatureError(f"{location} prices must be positive")
    if numeric["volume"] < 0.0:
        raise SecFilingGemmaFeatureError(f"{location} volume cannot be negative")
    if numeric["high"] < max(numeric["open"], numeric["low"], numeric["close"]):
        raise SecFilingGemmaFeatureError(f"{location} high is inconsistent with OHLC")
    if numeric["low"] > min(numeric["open"], numeric["high"], numeric["close"]):
        raise SecFilingGemmaFeatureError(f"{location} low is inconsistent with OHLC")
    expected_adjusted_open = derive_adjusted_open_hex(
        open_hex=observation["open_hex"],
        close_hex=observation["close_hex"],
        adjusted_close_hex=observation["adjusted_close_hex"],
        location=location,
    )
    adjusted_open = observation["adjusted_open_hex"]
    decode_float_hex(adjusted_open, f"{location}.adjusted_open")
    if adjusted_open != expected_adjusted_open:
        raise SecFilingGemmaFeatureError(
            f"{location}.adjusted_open is not the frozen derived value"
        )
    return dict(observation)


def _validate_market_row(
    value: Any,
    *,
    expected_index: int,
    expected_session: str,
    expected_parent_sha256: str,
    location: str,
) -> dict[str, Any]:
    row = _expect_mapping(value, location)
    _expect_keys(row, _MARKET_ROW_KEYS, location)
    if row["schema_version"] != MARKET_ROW_SCHEMA_VERSION:
        raise SecFilingGemmaFeatureError("Market row schema changed")
    if (
        isinstance(row["row_index"], bool)
        or not isinstance(row["row_index"], int)
        or row["row_index"] != expected_index
    ):
        raise SecFilingGemmaFeatureError("Market row index changed or is not contiguous")
    session = _iso_date(row["session"], f"{location}.session")
    if session != expected_session:
        raise SecFilingGemmaFeatureError("Market rows are missing, shifted, or reordered")
    if row["previous_row_sha256"] != expected_parent_sha256:
        raise SecFilingGemmaFeatureError("Market row hash chain is broken")
    observations = _expect_mapping(row["observations"], f"{location}.observations")
    if set(observations) != set(MARKET_SYMBOLS):
        raise SecFilingGemmaFeatureError("Market row must contain exactly all six symbols")
    normalized_observations = {
        symbol: _validate_observation(
            observations[symbol], f"{location}.observations.{symbol}"
        )
        for symbol in MARKET_SYMBOLS
    }
    if normalized_observations["AAPL"]["available"] is not True:
        raise SecFilingGemmaFeatureError("AAPL must be present in every market row")
    body = {
        "schema_version": MARKET_ROW_SCHEMA_VERSION,
        "row_index": expected_index,
        "session": session,
        "observations": normalized_observations,
        "previous_row_sha256": expected_parent_sha256,
    }
    calculated = canonical_sha256(body)
    if not hmac.compare_digest(
        calculated, _sha256(row["row_sha256"], f"{location}.row_sha256")
    ):
        raise SecFilingGemmaFeatureError("Market row checksum does not reconcile")
    return {**body, "row_sha256": calculated}


def _validate_compact_market_prefix(
    prefix: Mapping[str, Any], *, expected_market_prefix_sha256: str
) -> dict[str, Any]:
    value = _expect_mapping(prefix, "decision market prefix")
    _expect_keys(value, _PREFIX_KEYS, "decision market prefix")
    if value["schema_version"] != MARKET_PREFIX_SCHEMA_VERSION:
        raise SecFilingGemmaFeatureError("Decision market prefix schema changed")
    stage = value["artifact_stage"]
    if stage not in STAGE_ORDER:
        raise SecFilingGemmaFeatureError("Decision market prefix stage is invalid")
    decision = _iso_date(value["decision_session"], "decision_session")
    if not (STAGE_WINDOWS[stage][0] <= decision <= STAGE_WINDOWS[stage][1]):
        raise SecFilingGemmaFeatureError("Decision session is outside its stage")
    if (
        value["market_cutoff_rule"] != "completed_decision_session_close_inclusive"
        or value["market_cutoff_session"] != decision
        or value["full_prefix_last_session"] != decision
        or value["lookback_last_session"] != decision
        or value["fill_session_offset"] != LABEL_ENTRY_OFFSET
        or value["fill_session"] != _offset_session(decision, 1, market_calendar=True)
        or value["label_maturity_session_offset"] != LABEL_MATURITY_OFFSET
        or value["label_maturity_session"]
        != _offset_session(decision, LABEL_MATURITY_OFFSET, market_calendar=True)
        or value["calendar_id"] != MARKET_HISTORY_CALENDAR_ID
        or value["maximum_feature_lookback_sessions"] != 252
        or value["lookback_row_count"] != MARKET_LOOKBACK_ROW_COUNT
    ):
        raise SecFilingGemmaFeatureError("Decision market timing or lookback changed")
    _sha256(value["market_stage_manifest_sha256"], "market_stage_manifest_sha256")
    _sha256(value["source_manifest_sha256"], "source_manifest_sha256")
    if value["calendar_sha256"] != market_session_calendar_sha256(
        EXPECTED_MARKET_HISTORY_SESSIONS
    ):
        raise SecFilingGemmaFeatureError("Decision prefix calendar binding changed")

    try:
        decision_position = EXPECTED_MARKET_HISTORY_SESSIONS.index(decision)
    except ValueError as exc:
        raise SecFilingGemmaFeatureError(
            "Decision session is outside the exact market calendar"
        ) from exc
    expected_sessions = EXPECTED_MARKET_HISTORY_SESSIONS[
        decision_position - 252 : decision_position + 1
    ]
    if len(expected_sessions) != MARKET_LOOKBACK_ROW_COUNT:
        raise SecFilingGemmaFeatureError("Decision lacks the exact market prehistory")
    expected_start_index = decision_position - 252
    if (
        value["lookback_start_row_index"] != expected_start_index
        or value["lookback_end_row_index"] != decision_position
        or value["lookback_first_session"] != expected_sessions[0]
        or value["full_prefix_first_session"] != EXPECTED_MARKET_HISTORY_SESSIONS[0]
        or value["full_prefix_row_count"] != decision_position + 1
    ):
        raise SecFilingGemmaFeatureError("Decision market prefix indices changed")
    rows_value = value["lookback_rows"]
    if isinstance(rows_value, (str, bytes)) or not isinstance(rows_value, Sequence):
        raise SecFilingGemmaFeatureError("Decision lookback rows must be a sequence")
    if len(rows_value) != MARKET_LOOKBACK_ROW_COUNT:
        raise SecFilingGemmaFeatureError("Decision lookback must contain exactly 253 rows")
    parent = _sha256(
        value["lookback_start_parent_row_sha256"],
        "lookback_start_parent_row_sha256",
    )
    rows: list[dict[str, Any]] = []
    for offset, expected_session in enumerate(expected_sessions):
        row = _validate_market_row(
            rows_value[offset],
            expected_index=expected_start_index + offset,
            expected_session=expected_session,
            expected_parent_sha256=parent,
            location=f"lookback_rows[{offset}]",
        )
        rows.append(row)
        parent = row["row_sha256"]
    row_hashes = [row["row_sha256"] for row in rows]
    # MARKET_PREFIX_SCHEMA_VERSION intentionally does not carry the genesis as
    # a field.  Rebuild its identity with the frozen market-evidence genesis.
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
        or value["full_prefix_chain_identity_sha256"] != prefix_chain_identity
    ):
        raise SecFilingGemmaFeatureError("Decision prefix chain identities do not reconcile")
    body = {key: value[key] for key in value if key != "market_prefix_sha256"}
    calculated = canonical_sha256(body)
    observed = _sha256(value["market_prefix_sha256"], "market_prefix_sha256")
    if not hmac.compare_digest(observed, calculated):
        raise SecFilingGemmaFeatureError("Decision market prefix checksum changed")
    if not hmac.compare_digest(
        observed,
        _sha256(expected_market_prefix_sha256, "expected_market_prefix_sha256"),
    ):
        raise SecFilingGemmaFeatureError("Decision market prefix is not externally pinned")
    return {**body, "lookback_rows": rows, "market_prefix_sha256": observed}


def build_validated_market_prefix_proof(
    *,
    prefix: Mapping[str, Any],
    stage_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    expected_artifact_stage: str,
    expected_source_manifest_sha256: str,
    expected_market_stage_manifest_sha256: str,
    expected_decision_event_id: str,
    expected_decision_session: str,
    expected_market_prefix_sha256: str,
    session_dates: Sequence[str] = EXPECTED_MARKET_HISTORY_SESSIONS,
) -> dict[str, Any]:
    """Mint a compact proof only after the authoritative market replay passes.

    The effectful orchestration layer may see the full cumulative stage.  The
    downstream feature worker receives this proof and the 253-row prefix only,
    so it never receives a future market row.
    """

    validated_hash = validate_decision_market_prefix(
        prefix,
        stage_manifest=stage_manifest,
        source_manifest=source_manifest,
        expected_artifact_stage=expected_artifact_stage,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_market_stage_manifest_sha256=expected_market_stage_manifest_sha256,
        expected_decision_event_id=expected_decision_event_id,
        expected_decision_session=expected_decision_session,
        expected_market_prefix_sha256=expected_market_prefix_sha256,
        session_dates=session_dates,
    )
    compact = _validate_compact_market_prefix(
        prefix, expected_market_prefix_sha256=validated_hash
    )
    body = {
        "schema_version": MARKET_PREFIX_PROOF_SCHEMA_VERSION,
        "artifact_stage": compact["artifact_stage"],
        "decision_event_id": compact["decision_event_id"],
        "decision_session": compact["decision_session"],
        "market_prefix_sha256": compact["market_prefix_sha256"],
        "market_stage_manifest_sha256": compact["market_stage_manifest_sha256"],
        "source_manifest_sha256": compact["source_manifest_sha256"],
        "calendar_sha256": compact["calendar_sha256"],
        "full_prefix_chain_identity_sha256": compact[
            "full_prefix_chain_identity_sha256"
        ],
        "lookback_rows_sha256": compact["lookback_rows_sha256"],
        "lookback_row_count": compact["lookback_row_count"],
        "validation_semantics": (
            "structural_stage_source_hash_replay_then_compact_prefix_isolation"
        ),
        "authoritative_source_byte_reconciliation_required": True,
    }
    return {**body, "market_prefix_proof_sha256": canonical_sha256(body)}


def validate_market_prefix_proof(
    proof: Mapping[str, Any],
    *,
    prefix: Mapping[str, Any],
    expected_market_prefix_proof_sha256: str,
) -> dict[str, Any]:
    """Validate an externally pinned proof without opening the full stage."""

    value = _expect_mapping(proof, "market prefix proof")
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
    _expect_keys(value, expected_keys, "market prefix proof")
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
        "market_stage_manifest_sha256": compact["market_stage_manifest_sha256"],
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
    if any(value[key] != expected for key, expected in expected_fields.items()):
        raise SecFilingGemmaFeatureError("Market prefix proof differs from its prefix")
    body = {key: value[key] for key in value if key != "market_prefix_proof_sha256"}
    calculated = canonical_sha256(body)
    observed = _sha256(
        value["market_prefix_proof_sha256"], "market_prefix_proof_sha256"
    )
    if not hmac.compare_digest(observed, calculated):
        raise SecFilingGemmaFeatureError("Market prefix proof checksum changed")
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_market_prefix_proof_sha256,
            "expected_market_prefix_proof_sha256",
        ),
    ):
        raise SecFilingGemmaFeatureError("Market prefix proof is not externally pinned")
    return {**dict(value), "compact_prefix": compact}


def _normalize_universe_record(value: Any, location: str) -> dict[str, Any]:
    record = _expect_mapping(value, location)
    _expect_keys(record, _UNIVERSE_RECORD_KEYS, location)
    accession = record["accession_number"]
    if not isinstance(accession, str) or _ACCESSION_RE.fullmatch(accession) is None:
        raise SecFilingGemmaFeatureError(f"{location} is not an Apple accession")
    if record["subject_cik"] != "0000320193" or record["form"] not in {"10-K", "10-Q"}:
        raise SecFilingGemmaFeatureError(f"{location} is not an eligible Apple filing")
    acceptance = record["acceptance_datetime"]
    if acceptance is not None:
        if not isinstance(acceptance, str) or _ACCEPTANCE_RE.fullmatch(acceptance) is None:
            raise SecFilingGemmaFeatureError(f"{location} acceptance timestamp is invalid")
        try:
            datetime.strptime(acceptance, "%Y%m%d%H%M%S")
        except ValueError as exc:
            raise SecFilingGemmaFeatureError(
                f"{location} acceptance timestamp is invalid"
            ) from exc
    filing_date = _iso_date(record["filing_date"], f"{location}.filing_date")
    change = record["filing_date_change"]
    if change is not None:
        change = _iso_date(change, f"{location}.filing_date_change")
    availability = _iso_date(
        record["availability_session"], f"{location}.availability_session"
    )
    if availability not in EXPECTED_SESSIONS:
        raise SecFilingGemmaFeatureError(f"{location} availability is not a frozen session")
    stage = record["artifact_stage"]
    if stage not in STAGE_ORDER or not (
        STAGE_WINDOWS[stage][0] <= availability <= STAGE_WINDOWS[stage][1]
    ):
        raise SecFilingGemmaFeatureError(f"{location} stage is inconsistent")
    primary = record["primary_document"]
    if not isinstance(primary, str) or not primary or "/" in primary or "\\" in primary:
        raise SecFilingGemmaFeatureError(f"{location} primary document is invalid")
    return {
        "accession_number": accession,
        "subject_cik": "0000320193",
        "form": record["form"],
        "acceptance_datetime": acceptance,
        "filing_date": filing_date,
        "filing_date_change": change,
        "primary_document": primary,
        "source_record_sha256": _sha256(
            record["source_record_sha256"], f"{location}.source_record_sha256"
        ),
        "availability_session": availability,
        "artifact_stage": stage,
    }


def _normalize_content_record(value: Any, location: str) -> dict[str, Any]:
    """Validate the exact compact content-record shape exposed to a worker."""

    record = _expect_mapping(value, location)
    _expect_keys(record, _CONTENT_RECORD_KEYS, location)
    accession = record["accession_number"]
    if not isinstance(accession, str) or _ACCESSION_RE.fullmatch(accession) is None:
        raise SecFilingGemmaFeatureError(f"{location} accession is invalid")
    form = record["form"]
    if form not in {"10-K", "10-Q"}:
        raise SecFilingGemmaFeatureError(f"{location} form is invalid")
    availability = _iso_date(
        record["availability_session"], f"{location}.availability_session"
    )
    if availability not in EXPECTED_SESSIONS:
        raise SecFilingGemmaFeatureError(
            f"{location} availability is not a frozen session"
        )
    primary = record["primary_document"]
    if not isinstance(primary, str) or not primary or "/" in primary or "\\" in primary:
        raise SecFilingGemmaFeatureError(f"{location} primary document is invalid")
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
            record["primary_document_bytes"], f"{location}.primary_document_bytes"
        ),
        "normalized_text_bytes": _strict_positive_int(
            record["normalized_text_bytes"], f"{location}.normalized_text_bytes"
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
    """Isolate one exact universe event and its immediate same-form prior."""

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
        raise SecFilingGemmaFeatureError("Current event is absent from the universe")
    current = _normalize_universe_record(current, "current universe record")
    before = sorted(
        (
            _normalize_universe_record(record, "prior universe record")
            for record in records
            if record["form"] == current["form"]
            and (record["availability_session"], record["accession_number"])
            < (current["availability_session"], current["accession_number"])
        ),
        key=lambda record: (record["availability_session"], record["accession_number"]),
    )
    prior = before[-1] if before else None
    needed_stages = {current["artifact_stage"]}
    if prior is not None:
        needed_stages.add(prior["artifact_stage"])
    manifests = _expect_mapping(
        content_manifests_by_stage, "content_manifests_by_stage"
    )
    expected_hashes = _expect_mapping(
        expected_content_manifest_sha256s,
        "expected_content_manifest_sha256s",
    )
    _expect_keys(manifests, needed_stages, "content_manifests_by_stage")
    _expect_keys(
        expected_hashes, needed_stages, "expected_content_manifest_sha256s"
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
    current_content = content_by_accession.get(current["accession_number"])
    prior_content = (
        None
        if prior is None
        else content_by_accession.get(prior["accession_number"])
    )
    if current_content is None or (prior is not None and prior_content is None):
        raise SecFilingGemmaFeatureError(
            "Universe event content is absent from its validated stage manifest"
        )
    body = {
        "schema_version": UNIVERSE_EVENT_PROOF_SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "corpus_universe_sha256": universe_hash,
        "current_record": copy.deepcopy(current),
        "current_record_sha256": canonical_sha256(current),
        "current_content_record": copy.deepcopy(dict(current_content)),
        "current_content_record_sha256": canonical_sha256(current_content),
        "current_filing_sha256": current_content["normalized_text_sha256"],
        "prior_same_form_record": copy.deepcopy(prior),
        "prior_same_form_record_sha256": (
            None if prior is None else canonical_sha256(prior)
        ),
        "prior_same_form_content_record": (
            None if prior_content is None else copy.deepcopy(dict(prior_content))
        ),
        "prior_same_form_content_record_sha256": (
            None if prior_content is None else canonical_sha256(prior_content)
        ),
        "prior_same_form_filing_sha256": (
            None if prior_content is None else prior_content["normalized_text_sha256"]
        ),
        "content_manifest_sha256s_by_stage": normalized_content_hashes,
        "prior_selection": (
            "immediate_predecessor_same_form_by_availability_session_and_accession"
        ),
    }
    return {**body, "universe_event_proof_sha256": canonical_sha256(body)}


def validate_universe_event_proof(
    proof: Mapping[str, Any], *, expected_universe_event_proof_sha256: str
) -> dict[str, Any]:
    """Validate the sealed compact universe/content proof used by a worker."""

    value = _expect_mapping(proof, "universe event proof")
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
    _expect_keys(value, expected_keys, "universe event proof")
    if (
        value["schema_version"] != UNIVERSE_EVENT_PROOF_SCHEMA_VERSION
        or value["contract_sha256"] != canonical_sha256(build_contract_manifest())
        or value["prior_selection"]
        != "immediate_predecessor_same_form_by_availability_session_and_accession"
    ):
        raise SecFilingGemmaFeatureError("Universe event proof semantics changed")
    _sha256(value["corpus_universe_sha256"], "corpus_universe_sha256")
    current = _normalize_universe_record(value["current_record"], "current record")
    if value["current_record_sha256"] != canonical_sha256(current):
        raise SecFilingGemmaFeatureError("Current universe record hash changed")
    current_content = _normalize_content_record(
        value["current_content_record"], "current content record"
    )
    if (
        current_content["accession_number"] != current["accession_number"]
        or current_content["form"] != current["form"]
        or current_content["availability_session"] != current["availability_session"]
        or current_content["primary_document"] != current["primary_document"]
        or value["current_content_record_sha256"]
        != canonical_sha256(current_content)
        or value["current_filing_sha256"]
        != current_content["normalized_text_sha256"]
    ):
        raise SecFilingGemmaFeatureError("Current content proof does not reconcile")
    _sha256(value["current_filing_sha256"], "current_filing_sha256")
    prior_value = value["prior_same_form_record"]
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
            raise SecFilingGemmaFeatureError("First same-form event claims prior evidence")
        prior = None
    else:
        prior = _normalize_universe_record(prior_value, "prior same-form record")
        prior_content = _normalize_content_record(
            value["prior_same_form_content_record"],
            "prior same-form content record",
        )
        if (
            prior["form"] != current["form"]
            or (prior["availability_session"], prior["accession_number"])
            >= (current["availability_session"], current["accession_number"])
            or value["prior_same_form_record_sha256"] != canonical_sha256(prior)
            or prior_content["accession_number"] != prior["accession_number"]
            or prior_content["form"] != prior["form"]
            or prior_content["availability_session"] != prior["availability_session"]
            or prior_content["primary_document"] != prior["primary_document"]
            or value["prior_same_form_content_record_sha256"]
            != canonical_sha256(prior_content)
            or value["prior_same_form_filing_sha256"]
            != prior_content["normalized_text_sha256"]
        ):
            raise SecFilingGemmaFeatureError("Prior same-form content proof is inconsistent")
        _sha256(
            value["prior_same_form_filing_sha256"],
            "prior_same_form_filing_sha256",
        )
    stage_hashes = _expect_mapping(
        value["content_manifest_sha256s_by_stage"],
        "content_manifest_sha256s_by_stage",
    )
    needed_stages = {current["artifact_stage"]}
    if prior is not None:
        needed_stages.add(prior["artifact_stage"])
    _expect_keys(stage_hashes, needed_stages, "content_manifest_sha256s_by_stage")
    for stage, digest in stage_hashes.items():
        _sha256(digest, f"content_manifest_sha256s_by_stage.{stage}")
    body = {key: value[key] for key in value if key != "universe_event_proof_sha256"}
    calculated = canonical_sha256(body)
    observed = _sha256(
        value["universe_event_proof_sha256"], "universe_event_proof_sha256"
    )
    if not hmac.compare_digest(observed, calculated):
        raise SecFilingGemmaFeatureError("Universe event proof checksum changed")
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_universe_event_proof_sha256,
            "expected_universe_event_proof_sha256",
        ),
    ):
        raise SecFilingGemmaFeatureError("Universe event proof is not externally pinned")
    return {
        **dict(value),
        "current_record": current,
        "current_content_record": current_content,
        "prior_same_form_record": prior,
        "prior_same_form_content_record": (
            None if prior is None else prior_content
        ),
    }


def _strict_json_object(payload: bytes) -> Mapping[str, Any]:
    if not isinstance(payload, bytes) or not payload:
        raise SecFilingGemmaFeatureError("Extractor output bytes are missing")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SecFilingGemmaFeatureError("Extractor output is not UTF-8") from exc
    if text.startswith("\ufeff"):
        raise SecFilingGemmaFeatureError("Extractor output cannot contain a BOM")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise SecFilingGemmaFeatureError("Extractor output has duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SecFilingGemmaFeatureError(
                    f"Extractor output contains invalid constant {token}"
                )
            ),
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise SecFilingGemmaFeatureError("Extractor output is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise SecFilingGemmaFeatureError("Extractor output must be an object")
    return value


def _neutral_semantics() -> dict[str, float]:
    return {name: 0.0 for name in SEMANTIC_AGGREGATE_FEATURE_COLUMNS}


def _semantic_aggregates(output: Mapping[str, Any] | None) -> dict[str, float]:
    if output is None or output["document_quality"] == "unusable":
        return _neutral_semantics()
    dimensions = output["dimensions"]
    flags = output["flags"]
    result: dict[str, float] = {}
    for group_name, group_members in DIMENSION_GROUPS.items():
        current_values = [
            CURRENT_IMPACT_ENCODING[dimensions[name]["current_impact"]]
            for name in group_members
        ]
        change_values = [
            COMPARATIVE_CHANGE_ENCODING[dimensions[name]["change_vs_prior"]]
            for name in group_members
        ]
        result[f"{group_name}_current_mean"] = sum(current_values) / len(group_members)
        result[f"{group_name}_change_mean"] = sum(change_values) / len(group_members)
    result["adverse_flag_count"] = float(
        sum(bool(flags[name]["present"]) for name in ADVERSE_FLAG_NAMES)
    )
    result["management_transition_flag"] = float(
        bool(flags["management_transition"]["present"])
    )
    result["stated_dimension_fraction"] = sum(
        dimensions[name]["current_impact"] != "not_stated" for name in DIMENSION_NAMES
    ) / len(DIMENSION_NAMES)
    result["comparable_dimension_fraction"] = sum(
        dimensions[name]["change_vs_prior"] not in {"not_stated", "not_comparable"}
        for name in DIMENSION_NAMES
    ) / len(DIMENSION_NAMES)
    result["document_usable"] = float(output["document_quality"] == "usable")
    result["document_thin"] = float(output["document_quality"] == "thin")
    if tuple(result) != SEMANTIC_AGGREGATE_FEATURE_COLUMNS:
        raise SecFilingGemmaFeatureError("Semantic aggregate order changed")
    return result


def _normalize_supplied_sentence_ids(value: Any, location: str) -> list[str]:
    """Require the exact contiguous C#### then P#### request-id ordering."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SecFilingGemmaFeatureError(f"{location} must be a sequence")
    sentence_ids = list(value)
    if len(sentence_ids) > MAX_SENTENCES or any(
        not isinstance(item, str) or _SENTENCE_ID_RE.fullmatch(item) is None
        for item in sentence_ids
    ):
        raise SecFilingGemmaFeatureError(f"{location} contains invalid sentence ids")
    if len(sentence_ids) != len(set(sentence_ids)):
        raise SecFilingGemmaFeatureError(f"{location} sentence ids must be unique")
    if sentence_ids:
        current_count = sum(item.startswith("C") for item in sentence_ids)
        prior_count = len(sentence_ids) - current_count
        if current_count == 0:
            raise SecFilingGemmaFeatureError(
                f"{location} requires current-filing sentence ids"
            )
        expected = [f"C{index:04d}" for index in range(1, current_count + 1)]
        expected += [f"P{index:04d}" for index in range(1, prior_count + 1)]
        if sentence_ids != expected:
            raise SecFilingGemmaFeatureError(
                f"{location} sentence ids must be contiguous and ordered"
            )
    return sentence_ids


def _authenticate_extraction(
    *,
    extraction_status: str,
    extraction_evidence_sha256: str | None,
    expected_extraction_evidence_sha256: str | None,
    extractor_output: Mapping[str, Any] | None,
    extractor_output_bytes: bytes | None,
    expected_extraction_output_sha256: str | None,
    expected_extraction_output_canonical_sha256: str | None,
    supplied_sentence_ids: Sequence[str],
) -> tuple[bool, Mapping[str, Any] | None, str | None, str | None, str | None]:
    supplied_sentence_ids = _normalize_supplied_sentence_ids(
        supplied_sentence_ids, "supplied_sentence_ids"
    )
    if extraction_status not in EXTRACTION_STATUSES:
        raise SecFilingGemmaFeatureError("Extraction status is invalid")
    evidence_hash: str | None = None
    try:
        evidence_hash = _sha256(extraction_evidence_sha256, "extraction_evidence_sha256")
        if not hmac.compare_digest(
            evidence_hash,
            _sha256(
                expected_extraction_evidence_sha256,
                "expected_extraction_evidence_sha256",
            ),
        ):
            raise SecFilingGemmaFeatureError("Extraction evidence is not externally pinned")
        if extraction_status == "unavailable":
            return False, None, evidence_hash, None, None
        if extraction_status == "invalid":
            if (
                extractor_output is not None
                or extractor_output_bytes is not None
                or expected_extraction_output_sha256 is not None
                or expected_extraction_output_canonical_sha256 is not None
            ):
                raise SecFilingGemmaFeatureError(
                    "A sealed invalid extraction cannot carry model output"
                )
            return True, None, evidence_hash, None, None
        if extractor_output is None or extractor_output_bytes is None:
            raise SecFilingGemmaFeatureError("Validated extraction output is missing")
        parsed = _strict_json_object(extractor_output_bytes)
        if dict(parsed) != dict(extractor_output):
            raise SecFilingGemmaFeatureError(
                "Extractor output mapping differs from its exact bytes"
            )
        output_hash = hashlib.sha256(extractor_output_bytes).hexdigest()
        if not hmac.compare_digest(
            output_hash,
            _sha256(
                expected_extraction_output_sha256,
                "expected_extraction_output_sha256",
            ),
        ):
            raise SecFilingGemmaFeatureError("Extractor output bytes are not pinned")
        canonical_hash = canonical_sha256(parsed)
        if not hmac.compare_digest(
            canonical_hash,
            _sha256(
                expected_extraction_output_canonical_sha256,
                "expected_extraction_output_canonical_sha256",
            ),
        ):
            raise SecFilingGemmaFeatureError("Extractor output semantics are not pinned")
        validate_extractor_output(parsed, supplied_sentence_ids=supplied_sentence_ids)
        return True, parsed, evidence_hash, output_hash, canonical_hash
    except (SecFilingGemmaContractError, TypeError, ValueError):
        # Missing or unauthenticated extraction evidence is an auditable row,
        # but it cannot be presented to either learner.
        return False, None, evidence_hash, None, None


def build_validated_extraction_event_proof(
    *,
    universe_event_proof: Mapping[str, Any],
    expected_universe_event_proof_sha256: str,
    extraction_status: str,
    extraction_evidence_sha256: str | None,
    expected_extraction_evidence_sha256: str | None,
    extractor_output: Mapping[str, Any] | None,
    extractor_output_bytes: bytes | None,
    expected_extraction_output_sha256: str | None,
    expected_extraction_output_canonical_sha256: str | None,
    supplied_sentence_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Bind validated extraction evidence to one exact universe/content event."""

    universe = validate_universe_event_proof(
        universe_event_proof,
        expected_universe_event_proof_sha256=expected_universe_event_proof_sha256,
    )
    sentence_ids = _normalize_supplied_sentence_ids(
        supplied_sentence_ids, "supplied_sentence_ids"
    )
    if sentence_ids:
        has_prior_ids = any(item.startswith("P") for item in sentence_ids)
        has_prior_filing = universe["prior_same_form_filing_sha256"] is not None
        if has_prior_ids != has_prior_filing:
            raise SecFilingGemmaFeatureError(
                "Supplied sentence ids do not reconcile with prior filing evidence"
            )
    authenticated, output, evidence_hash, output_hash, canonical_output_hash = (
        _authenticate_extraction(
            extraction_status=extraction_status,
            extraction_evidence_sha256=extraction_evidence_sha256,
            expected_extraction_evidence_sha256=expected_extraction_evidence_sha256,
            extractor_output=extractor_output,
            extractor_output_bytes=extractor_output_bytes,
            expected_extraction_output_sha256=expected_extraction_output_sha256,
            expected_extraction_output_canonical_sha256=(
                expected_extraction_output_canonical_sha256
            ),
            supplied_sentence_ids=sentence_ids,
        )
    )
    if authenticated and not sentence_ids:
        raise SecFilingGemmaFeatureError(
            "Authenticated extraction evidence requires canonical request sentence ids"
        )
    current = universe["current_record"]
    body = {
        "schema_version": EXTRACTION_EVENT_PROOF_SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "universe_event_proof_sha256": universe[
            "universe_event_proof_sha256"
        ],
        "accession_number": current["accession_number"],
        "form": current["form"],
        "decision_session": current["availability_session"],
        "current_filing_sha256": universe["current_filing_sha256"],
        "prior_same_form_filing_sha256": universe[
            "prior_same_form_filing_sha256"
        ],
        "extraction_status": extraction_status,
        "extraction_evidence_authenticated": authenticated,
        "extraction_evidence_sha256": evidence_hash,
        "extraction_output_sha256": output_hash,
        "extraction_output_canonical_sha256": canonical_output_hash,
        "supplied_sentence_ids": sentence_ids,
        "supplied_sentence_ids_sha256": canonical_sha256(sentence_ids),
        "document_quality": None if output is None else output["document_quality"],
        "validated_output": None if output is None else copy.deepcopy(dict(output)),
        "validation_scope": (
            "external_pin_exact_output_bytes_hash_schema_and_event_binding"
        ),
        "authoritative_request_and_ollama_receipt_replay_required": True,
    }
    return {**body, "extraction_event_proof_sha256": canonical_sha256(body)}


def validate_extraction_event_proof(
    proof: Mapping[str, Any],
    *,
    universe_event_proof: Mapping[str, Any],
    expected_universe_event_proof_sha256: str,
    expected_extraction_event_proof_sha256: str,
) -> dict[str, Any]:
    """Validate extraction/event association and its exact status truth table."""

    universe = validate_universe_event_proof(
        universe_event_proof,
        expected_universe_event_proof_sha256=expected_universe_event_proof_sha256,
    )
    value = _expect_mapping(proof, "extraction event proof")
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
    _expect_keys(value, expected_keys, "extraction event proof")
    current = universe["current_record"]
    if (
        value["schema_version"] != EXTRACTION_EVENT_PROOF_SCHEMA_VERSION
        or value["contract_sha256"] != canonical_sha256(build_contract_manifest())
        or value["universe_event_proof_sha256"]
        != universe["universe_event_proof_sha256"]
        or value["accession_number"] != current["accession_number"]
        or value["form"] != current["form"]
        or value["decision_session"] != current["availability_session"]
        or value["current_filing_sha256"] != universe["current_filing_sha256"]
        or value["prior_same_form_filing_sha256"]
        != universe["prior_same_form_filing_sha256"]
        or value["validation_scope"]
        != "external_pin_exact_output_bytes_hash_schema_and_event_binding"
        or value["authoritative_request_and_ollama_receipt_replay_required"] is not True
    ):
        raise SecFilingGemmaFeatureError("Extraction proof is bound to another event")
    status = value["extraction_status"]
    if status not in EXTRACTION_STATUSES:
        raise SecFilingGemmaFeatureError("Extraction proof status is invalid")
    authenticated = value["extraction_evidence_authenticated"]
    if not isinstance(authenticated, bool):
        raise SecFilingGemmaFeatureError("Extraction authentication must be boolean")
    evidence_hash = _optional_sha256(
        value["extraction_evidence_sha256"], "extraction_evidence_sha256"
    )
    output_hash = _optional_sha256(
        value["extraction_output_sha256"], "extraction_output_sha256"
    )
    canonical_output_hash = _optional_sha256(
        value["extraction_output_canonical_sha256"],
        "extraction_output_canonical_sha256",
    )
    if not isinstance(value["supplied_sentence_ids"], list):
        raise SecFilingGemmaFeatureError("Extraction proof sentence ids must be a list")
    sentence_ids = _normalize_supplied_sentence_ids(
        value["supplied_sentence_ids"], "extraction proof supplied_sentence_ids"
    )
    if value["supplied_sentence_ids_sha256"] != canonical_sha256(sentence_ids):
        raise SecFilingGemmaFeatureError("Extraction proof sentence ids are invalid")
    if sentence_ids:
        has_prior_ids = any(item.startswith("P") for item in sentence_ids)
        has_prior_filing = universe["prior_same_form_filing_sha256"] is not None
        if has_prior_ids != has_prior_filing:
            raise SecFilingGemmaFeatureError(
                "Extraction proof sentence ids do not reconcile with prior filing"
            )
    output = value["validated_output"]
    quality = value["document_quality"]
    if authenticated:
        if evidence_hash is None or status == "unavailable" or not sentence_ids:
            raise SecFilingGemmaFeatureError(
                "Authenticated extraction evidence has an impossible status"
            )
        if status == "invalid":
            if any(
                item is not None
                for item in (output, quality, output_hash, canonical_output_hash)
            ):
                raise SecFilingGemmaFeatureError(
                    "Sealed invalid extraction cannot claim valid output"
                )
        else:
            output_mapping = _expect_mapping(output, "validated extractor output")
            if (
                output_hash is None
                or canonical_output_hash is None
                or quality not in DOCUMENT_QUALITIES
                or canonical_output_hash != canonical_sha256(output_mapping)
                or output_mapping.get("document_quality") != quality
            ):
                raise SecFilingGemmaFeatureError(
                    "Validated extraction output identities are incomplete"
                )
            validate_extractor_output(
                output_mapping, supplied_sentence_ids=sentence_ids
            )
    else:
        if any(
            item is not None
            for item in (output, quality, output_hash, canonical_output_hash)
        ):
            raise SecFilingGemmaFeatureError(
                "Unauthenticated extraction cannot expose semantic output"
            )
    body = {key: value[key] for key in value if key != "extraction_event_proof_sha256"}
    calculated = canonical_sha256(body)
    observed = _sha256(
        value["extraction_event_proof_sha256"],
        "extraction_event_proof_sha256",
    )
    if not hmac.compare_digest(observed, calculated):
        raise SecFilingGemmaFeatureError("Extraction event proof checksum changed")
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_extraction_event_proof_sha256,
            "expected_extraction_event_proof_sha256",
        ),
    ):
        raise SecFilingGemmaFeatureError("Extraction event proof is not externally pinned")
    return dict(value)


def _frames_from_prefix_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []

    def field(row: Mapping[str, Any], symbol: str, name: str) -> float:
        observation = row["observations"][symbol]
        if not observation["available"]:
            return math.nan
        return decode_float_hex(
            observation[f"{name}_hex"], f"{row['session']}.{symbol}.{name}"
        )

    for row in rows:
        price_rows.append(
            {
                "date": row["session"],
                "aapl_open": field(row, "AAPL", "open"),
                "aapl_close": field(row, "AAPL", "close"),
                "aapl_adj_close": field(row, "AAPL", "adjusted_close"),
                "spy_adj_close": field(row, "SPY", "adjusted_close"),
                "qqq_adj_close": field(row, "QQQ", "adjusted_close"),
            }
        )
        context_rows.append(
            {
                "date": row["session"],
                "iwm_adj_close": field(row, "IWM", "adjusted_close"),
                "vix_close": field(row, "VIX", "close"),
                "tnx_close": field(row, "TNX", "close"),
            }
        )
    return pd.DataFrame(price_rows), pd.DataFrame(context_rows)


def _encode_feature_map(
    names: Sequence[str], values: Mapping[str, Any]
) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        value = float(values[name])
        result[name] = value.hex() if math.isfinite(value) else None
    return result


def _extraction_causal_identity_sha256(
    *, universe: Mapping[str, Any], extraction: Mapping[str, Any]
) -> str:
    """Hash only the event-local extraction identity used by predictions.

    The validated universe and extraction proofs retain their complete corpus,
    content-stage, and receipt provenance.  Those full-proof hashes are
    intentionally excluded here: extending a corpus with a later filing must
    not rewrite an earlier prediction identity.  The projection still binds
    every event-local source/content record plus the authenticated request,
    receipt, and exact output identities needed to prevent substitution.
    """

    current_record = universe["current_record"]
    current_content = universe["current_content_record"]
    prior_record = universe["prior_same_form_record"]
    prior_content = universe["prior_same_form_content_record"]
    body = {
        "identity_domain": "aapl-sec-gemma-extraction-causal-identity-v1",
        "accession_number": current_record["accession_number"],
        "form": current_record["form"],
        "decision_session": current_record["availability_session"],
        "current_universe_record_sha256": canonical_sha256(current_record),
        "current_source_record_sha256": current_record["source_record_sha256"],
        "current_content_record_sha256": canonical_sha256(current_content),
        "current_primary_document_sha256": current_content[
            "primary_document_sha256"
        ],
        "current_filing_sha256": current_content["normalized_text_sha256"],
        "prior_same_form_universe_record_sha256": (
            None if prior_record is None else canonical_sha256(prior_record)
        ),
        "prior_same_form_source_record_sha256": (
            None if prior_record is None else prior_record["source_record_sha256"]
        ),
        "prior_same_form_content_record_sha256": (
            None if prior_content is None else canonical_sha256(prior_content)
        ),
        "prior_same_form_primary_document_sha256": (
            None if prior_content is None else prior_content["primary_document_sha256"]
        ),
        "prior_same_form_filing_sha256": (
            None if prior_content is None else prior_content["normalized_text_sha256"]
        ),
        "extraction_status": extraction["extraction_status"],
        "extraction_evidence_authenticated": extraction[
            "extraction_evidence_authenticated"
        ],
        # The authoritative stage verifier must replay this pinned evidence to
        # the exact redacted request and Ollama call receipt before minting the
        # extraction proof.  Keeping it in the causal projection prevents a
        # different request/receipt from being substituted for the same filing.
        "extraction_request_and_receipt_evidence_sha256": extraction[
            "extraction_evidence_sha256"
        ],
        "extraction_output_sha256": extraction["extraction_output_sha256"],
        "extraction_output_canonical_sha256": extraction[
            "extraction_output_canonical_sha256"
        ],
        "supplied_sentence_ids_sha256": extraction[
            "supplied_sentence_ids_sha256"
        ],
        "document_quality": extraction["document_quality"],
    }
    return canonical_sha256(body)


def build_sec_filing_gemma_feature_row(
    *,
    market_prefix: Mapping[str, Any],
    market_prefix_proof: Mapping[str, Any],
    expected_market_prefix_proof_sha256: str,
    universe_event_proof: Mapping[str, Any],
    expected_universe_event_proof_sha256: str,
    extraction_event_proof: Mapping[str, Any],
    expected_extraction_event_proof_sha256: str,
) -> dict[str, Any]:
    """Build one auditable decision-close feature row without outcome access."""

    market_proof = validate_market_prefix_proof(
        market_prefix_proof,
        prefix=market_prefix,
        expected_market_prefix_proof_sha256=expected_market_prefix_proof_sha256,
    )
    prefix = market_proof["compact_prefix"]
    universe = validate_universe_event_proof(
        universe_event_proof,
        expected_universe_event_proof_sha256=expected_universe_event_proof_sha256,
    )
    current = universe["current_record"]
    prior = universe["prior_same_form_record"]
    extraction = validate_extraction_event_proof(
        extraction_event_proof,
        universe_event_proof=universe_event_proof,
        expected_universe_event_proof_sha256=expected_universe_event_proof_sha256,
        expected_extraction_event_proof_sha256=(
            expected_extraction_event_proof_sha256
        ),
    )
    if (
        current["accession_number"] != prefix["decision_event_id"]
        or current["availability_session"] != prefix["decision_session"]
        or current["artifact_stage"] != prefix["artifact_stage"]
    ):
        raise SecFilingGemmaFeatureError("Market prefix and universe event differ")
    extraction_status = extraction["extraction_status"]
    extraction_authenticated = extraction["extraction_evidence_authenticated"]
    validated_output = extraction["validated_output"]
    evidence_hash = extraction["extraction_evidence_sha256"]
    output_hash = extraction["extraction_output_sha256"]
    canonical_output_hash = extraction["extraction_output_canonical_sha256"]
    semantic_output_unavailable = not (
        extraction_authenticated and extraction_status == "valid"
    )

    rows = prefix["lookback_rows"]
    missing_market_observations = [
        f"{symbol}@{row['session']}"
        for row in rows
        for symbol in MARKET_SYMBOLS
        if not row["observations"][symbol]["available"]
    ]
    market_available = not missing_market_observations
    price_frame, context_frame = _frames_from_prefix_rows(rows)
    price = build_downside_price_features(price_frame).iloc[-1]
    sentiment = build_market_sentiment_features(price_frame, context_frame).iloc[-1]
    price_map = _encode_feature_map(PRICE_FEATURE_COLUMNS, price)
    sentiment_map = _encode_feature_map(MARKET_SENTIMENT_FEATURE_COLUMNS, sentiment)
    if any(value is None for value in (*price_map.values(), *sentiment_map.values())):
        market_available = False

    current_position = EXPECTED_SESSIONS.index(current["availability_session"])
    sessions_since_prior = (
        0
        if prior is None
        else current_position - EXPECTED_SESSIONS.index(prior["availability_session"])
    )
    calendar_values = {
        "form_is_10k": float(current["form"] == "10-K"),
        "sessions_since_prior_same_form": float(sessions_since_prior),
        "semantic_output_unavailable": float(semantic_output_unavailable),
    }
    semantic_values = _semantic_aggregates(
        validated_output if extraction_authenticated and extraction_status == "valid" else None
    )
    calendar_map = {
        name: _float_hex(calendar_values[name], name)
        for name in FILING_CALENDAR_FEATURE_COLUMNS
    }
    semantic_map = {
        name: _float_hex(semantic_values[name], name)
        for name in SEMANTIC_AGGREGATE_FEATURE_COLUMNS
    }
    unavailable_reasons: list[str] = []
    if not market_available:
        unavailable_reasons.append("missing_required_market_history")
    if not extraction_authenticated:
        unavailable_reasons.append("missing_or_unauthenticated_extraction_evidence")
    prediction_available = not unavailable_reasons
    semantic_available = bool(
        extraction_authenticated
        and extraction_status == "valid"
        and validated_output is not None
        and validated_output["document_quality"] in {"usable", "thin"}
    )

    semantic_vector_values: list[str | None] = [
        *(price_map[name] for name in PRICE_FEATURE_COLUMNS),
        *(sentiment_map[name] for name in MARKET_SENTIMENT_FEATURE_COLUMNS),
        *(calendar_map[name] for name in FILING_CALENDAR_FEATURE_COLUMNS),
        *(semantic_map[name] for name in SEMANTIC_AGGREGATE_FEATURE_COLUMNS),
    ]
    ablation_vector_values: list[str | None] = [
        *(price_map[name] for name in PRICE_FEATURE_COLUMNS),
        *(sentiment_map[name] for name in MARKET_SENTIMENT_FEATURE_COLUMNS),
        *(calendar_map[name] for name in FILING_CALENDAR_FEATURE_COLUMNS),
        *(_float_hex(0.0) for _ in SEMANTIC_AGGREGATE_FEATURE_COLUMNS),
    ]
    if not prediction_available:
        semantic_vector: list[str] | None = None
        ablation_vector: list[str] | None = None
    else:
        if any(value is None for value in semantic_vector_values + ablation_vector_values):
            raise SecFilingGemmaFeatureError("Available feature vector contains nulls")
        semantic_vector = [str(value) for value in semantic_vector_values]
        ablation_vector = [str(value) for value in ablation_vector_values]

    current_record_hash = universe["current_record_sha256"]
    prior_record_hash = universe["prior_same_form_record_sha256"]
    current_filing_hash = universe["current_filing_sha256"]
    prior_filing_hash = universe["prior_same_form_filing_sha256"]
    extraction_identity_sha256 = _extraction_causal_identity_sha256(
        universe=universe,
        extraction=extraction,
    )
    bindings = {
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "corpus_universe_sha256": universe["corpus_universe_sha256"],
        "universe_event_proof_sha256": universe[
            "universe_event_proof_sha256"
        ],
        "current_universe_record_sha256": current_record_hash,
        "prior_same_form_universe_record_sha256": prior_record_hash,
        "current_source_record_sha256": current["source_record_sha256"],
        "prior_same_form_source_record_sha256": (
            None if prior is None else prior["source_record_sha256"]
        ),
        "current_filing_sha256": current_filing_hash,
        "prior_same_form_filing_sha256": prior_filing_hash,
        "extraction_event_proof_sha256": extraction[
            "extraction_event_proof_sha256"
        ],
        "extraction_identity_sha256": extraction_identity_sha256,
        "extraction_evidence_sha256": evidence_hash,
        "extraction_output_sha256": output_hash,
        "extraction_output_canonical_sha256": canonical_output_hash,
        "market_prefix_sha256": prefix["market_prefix_sha256"],
        "market_prefix_proof_sha256": market_proof[
            "market_prefix_proof_sha256"
        ],
        "market_stage_manifest_sha256": prefix["market_stage_manifest_sha256"],
        "source_manifest_sha256": prefix["source_manifest_sha256"],
        "market_prefix_chain_identity_sha256": prefix[
            "full_prefix_chain_identity_sha256"
        ],
    }
    market_feature_body = {
        "schema_version": MARKET_FEATURE_ROW_SCHEMA_VERSION,
        "accession_number": current["accession_number"],
        "decision_session": current["availability_session"],
        "market_prefix_chain_identity_sha256": prefix[
            "full_prefix_chain_identity_sha256"
        ],
        "market_available": market_available,
        "missing_market_observations": missing_market_observations,
        "missing_market_observations_sha256": canonical_sha256(
            missing_market_observations
        ),
        "price_regime_features_hex": price_map,
        "market_sentiment_features_hex": sentiment_map,
    }
    market_feature_row_sha256 = canonical_sha256(market_feature_body)
    body = {
        "schema_version": FEATURE_ROW_SCHEMA_VERSION,
        "accession_number": current["accession_number"],
        "form": current["form"],
        "artifact_stage": current["artifact_stage"],
        "decision_session": current["availability_session"],
        "extraction_status": extraction_status,
        "extraction_evidence_authenticated": extraction_authenticated,
        "document_quality": (
            None if validated_output is None else validated_output["document_quality"]
        ),
        "semantic_available": semantic_available,
        "market_available": market_available,
        "prediction_available": prediction_available,
        "fit_eligible": prediction_available,
        "unavailable_reasons": unavailable_reasons,
        "missing_market_observations": missing_market_observations,
        "missing_market_observations_sha256": canonical_sha256(
            missing_market_observations
        ),
        "bindings": bindings,
        "bindings_sha256": canonical_sha256(bindings),
        "market_feature_row_sha256": market_feature_row_sha256,
        "price_regime_features_hex": price_map,
        "market_sentiment_features_hex": sentiment_map,
        "filing_calendar_features_hex": calendar_map,
        "semantic_aggregate_features_hex": semantic_map,
        "semantic_feature_names": list(SEMANTIC_FEATURE_COLUMNS),
        "semantic_feature_schema_sha256": canonical_sha256(
            list(SEMANTIC_FEATURE_COLUMNS)
        ),
        "semantic_feature_values_hex": semantic_vector,
        "ablation_feature_names": list(ABLATION_FEATURE_COLUMNS),
        "ablation_feature_schema_sha256": canonical_sha256(
            list(ABLATION_FEATURE_COLUMNS)
        ),
        "ablation_feature_values_hex": ablation_vector,
    }
    return {**body, "feature_row_sha256": canonical_sha256(body)}


def validate_sec_filing_gemma_feature_row(
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
    """Rebuild from all compact upstream proofs and require exact equality."""

    value = _expect_mapping(row, "feature row")
    rebuilt = build_sec_filing_gemma_feature_row(
        market_prefix=market_prefix,
        market_prefix_proof=market_prefix_proof,
        expected_market_prefix_proof_sha256=expected_market_prefix_proof_sha256,
        universe_event_proof=universe_event_proof,
        expected_universe_event_proof_sha256=expected_universe_event_proof_sha256,
        extraction_event_proof=extraction_event_proof,
        expected_extraction_event_proof_sha256=(
            expected_extraction_event_proof_sha256
        ),
    )
    if dict(value) != rebuilt:
        raise SecFilingGemmaFeatureError(
            "Feature row differs from the authoritative compact-proof replay"
        )
    observed = _sha256(value.get("feature_row_sha256"), "feature_row_sha256")
    if not hmac.compare_digest(
        observed, _sha256(expected_feature_row_sha256, "expected_feature_row_sha256")
    ):
        raise SecFilingGemmaFeatureError("Feature row is not externally pinned")
    return observed


def _validated_development_feature_assembly_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the feature-only subset of the authorization-owned plan."""

    value = _expect_mapping(plan, "development feature assembly plan")
    _expect_keys(
        value,
        set(_DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_KEYS),
        "development feature assembly plan",
    )
    if (
        value["schema_version"]
        != DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["plan_kind"] != "request_free_development_feature_assembly"
        or value["artifact_stage"] != "development"
        or value["development_cutoff_session"] != STAGE_WINDOWS["development"][1]
    ):
        raise SecFilingGemmaFeatureError(
            "Development feature assembly plan identity changed"
        )
    hash_fields = (
        "development_root_scope_sha256",
        "development_content_root_plan_sha256",
        "candidate_sha256",
        "candidate_design_sha256",
        "corpus_universe_sha256",
        "development_sec_execution_claim_sha256",
        "development_sec_reader_receipt_sha256",
        "development_market_execution_claim_sha256",
        "development_market_reader_receipt_sha256",
        "development_market_acquisition_receipt_sha256",
        "development_market_acquisition_bundle_sha256",
        "development_market_acquisition_validation_sha256",
        "development_market_source_manifest_sha256",
        "development_market_stage_manifest_sha256",
        "development_market_source_reconciliation_sha256",
        "development_market_byte_index_sha256",
        "development_model_execution_claim_sha256",
        "development_model_reader_receipt_sha256",
        "event_plan_sha256",
        "execution_source_hashes_sha256",
        "feature_assembly_plan_sha256",
    )
    for field in hash_fields:
        _sha256(value[field], f"development feature assembly plan.{field}")
    consumed_count = value["start_consumed_request_count"]
    if (
        isinstance(consumed_count, bool)
        or not isinstance(consumed_count, int)
        or consumed_count != 0
    ):
        raise SecFilingGemmaFeatureError(
            "Development feature assembly plan consumed count is invalid"
        )
    event_count = _strict_positive_int(
        value["event_count"], "development feature assembly plan.event_count"
    )
    raw_event_plan = value["event_plan"]
    if type(raw_event_plan) is not list or len(raw_event_plan) != event_count:
        raise SecFilingGemmaFeatureError(
            "Development feature assembly event plan count changed"
        )
    event_plan: list[dict[str, Any]] = []
    observed_document_ordinals: list[int] = []
    observed_accessions: list[str] = []
    for ordinal, raw_item in enumerate(raw_event_plan, start=1):
        item = _expect_mapping(
            raw_item, f"development feature assembly event plan {ordinal}"
        )
        _expect_keys(
            item,
            set(_MODEL_EVENT_PLAN_ITEM_KEYS),
            f"development feature assembly event plan {ordinal}",
        )
        accession = item["accession_number"]
        availability = _iso_date(
            item["availability_session"],
            f"development feature assembly event plan {ordinal}.availability_session",
        )
        document_ordinal = item["sec_document_ordinal"]
        if (
            isinstance(item["event_ordinal"], bool)
            or not isinstance(item["event_ordinal"], int)
            or item["event_ordinal"] != ordinal
            or not isinstance(accession, str)
            or _ACCESSION_RE.fullmatch(accession) is None
            or item["form"] not in {"10-K", "10-Q"}
            or not (
                STAGE_WINDOWS["development"][0]
                <= availability
                <= STAGE_WINDOWS["development"][1]
            )
            or isinstance(document_ordinal, bool)
            or not isinstance(document_ordinal, int)
            or document_ordinal < 1
        ):
            raise SecFilingGemmaFeatureError(
                "Development feature assembly event plan item is invalid"
            )
        event_plan.append(dict(item))
        observed_document_ordinals.append(document_ordinal)
        observed_accessions.append(accession)
    if len(observed_accessions) != len(set(observed_accessions)):
        raise SecFilingGemmaFeatureError(
            "Development feature assembly event-plan accessions are duplicated"
        )
    if event_plan != sorted(
        event_plan,
        key=lambda item: (item["availability_session"], item["accession_number"]),
    ) or sorted(observed_document_ordinals) != list(
        range(1, event_count + 1)
    ):
        raise SecFilingGemmaFeatureError(
            "Development feature assembly event order is not canonical"
        )
    if canonical_sha256(event_plan) != value["event_plan_sha256"]:
        raise SecFilingGemmaFeatureError(
            "Development feature assembly event-plan checksum changed"
        )
    expected_flags = {
        "canonical_market_rows_required": True,
        "raw_market_output_permitted": False,
        "normalized_filing_text_output_permitted": False,
        "model_transport_envelope_output_permitted": False,
        "feature_rows_output_permitted": True,
        "outcome_access_permitted": False,
        "label_access_permitted": False,
        "training_membership_access_permitted": False,
        "learner_fit_permitted": False,
        "prediction_access_permitted": False,
        "holdout_access_permitted": False,
        "ledger_mutation_permitted": False,
        "stage_promotion_permitted": False,
    }
    if any(value[field] is not expected for field, expected in expected_flags.items()):
        raise SecFilingGemmaFeatureError(
            "Development feature assembly plan crossed its feature-only authority"
        )
    body = {
        key: value[key]
        for key in value
        if key != "feature_assembly_plan_sha256"
    }
    if canonical_sha256(body) != value["feature_assembly_plan_sha256"]:
        raise SecFilingGemmaFeatureError(
            "Development feature assembly plan checksum changed"
        )
    return copy.deepcopy(dict(value))


def _validated_feature_hex_mapping(
    value: Any,
    *,
    names: Sequence[str],
    location: str,
) -> dict[str, str | None]:
    mapping = _expect_mapping(value, location)
    _expect_keys(mapping, set(names), location)
    result: dict[str, str | None] = {}
    for name in names:
        raw = mapping[name]
        if raw is None:
            result[name] = None
            continue
        if not isinstance(raw, str):
            raise SecFilingGemmaFeatureError(f"{location}.{name} is not float.hex text")
        try:
            number = float.fromhex(raw)
        except ValueError as exc:
            raise SecFilingGemmaFeatureError(
                f"{location}.{name} is not float.hex text"
            ) from exc
        if not math.isfinite(number) or number.hex() != raw:
            raise SecFilingGemmaFeatureError(
                f"{location}.{name} is not canonical finite float.hex text"
            )
        result[name] = raw
    return result


def _validated_batch_feature_row(
    row: Mapping[str, Any],
    *,
    event_plan_item: Mapping[str, Any],
    feature_assembly_plan: Mapping[str, Any],
) -> dict[str, Any]:
    value = _expect_mapping(row, "owned development feature row")
    _expect_keys(value, set(_FEATURE_ROW_KEYS), "owned development feature row")
    if (
        value["schema_version"] != FEATURE_ROW_SCHEMA_VERSION
        or value["accession_number"] != event_plan_item["accession_number"]
        or value["form"] != event_plan_item["form"]
        or value["artifact_stage"] != "development"
        or value["decision_session"] != event_plan_item["availability_session"]
        or value["extraction_status"] not in EXTRACTION_STATUSES
        or value["document_quality"] not in {None, *DOCUMENT_QUALITIES}
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development feature row crossed its event identity"
        )
    boolean_fields = (
        "extraction_evidence_authenticated",
        "semantic_available",
        "market_available",
        "prediction_available",
        "fit_eligible",
    )
    if any(type(value[field]) is not bool for field in boolean_fields):
        raise SecFilingGemmaFeatureError(
            "Owned development feature availability flags are invalid"
        )
    expected_semantic_available = bool(
        value["extraction_evidence_authenticated"]
        and value["extraction_status"] == "valid"
        and value["document_quality"] in {"usable", "thin"}
    )
    if value["semantic_available"] is not expected_semantic_available:
        raise SecFilingGemmaFeatureError(
            "Owned development semantic availability changed"
        )
    missing = value["missing_market_observations"]
    reasons = value["unavailable_reasons"]
    if (
        type(missing) is not list
        or any(type(item) is not str for item in missing)
        or len(missing) != len(set(missing))
        or type(reasons) is not list
        or any(reason not in FEATURE_UNAVAILABLE_REASONS for reason in reasons)
        or len(reasons) != len(set(reasons))
        or value["missing_market_observations_sha256"] != canonical_sha256(missing)
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development feature missingness evidence changed"
        )
    expected_reasons: list[str] = []
    if not value["market_available"]:
        expected_reasons.append("missing_required_market_history")
    if not value["extraction_evidence_authenticated"]:
        expected_reasons.append("missing_or_unauthenticated_extraction_evidence")
    if (
        reasons != expected_reasons
        or value["prediction_available"] is not (not expected_reasons)
        or value["fit_eligible"] is not value["prediction_available"]
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development feature readiness changed"
        )

    bindings = _expect_mapping(value["bindings"], "owned development feature bindings")
    _expect_keys(bindings, set(_FEATURE_BINDING_KEYS), "owned development feature bindings")
    required_binding_hashes = (
        "contract_sha256",
        "corpus_universe_sha256",
        "universe_event_proof_sha256",
        "current_universe_record_sha256",
        "current_source_record_sha256",
        "current_filing_sha256",
        "extraction_event_proof_sha256",
        "extraction_identity_sha256",
        "market_prefix_sha256",
        "market_prefix_proof_sha256",
        "market_stage_manifest_sha256",
        "source_manifest_sha256",
        "market_prefix_chain_identity_sha256",
    )
    optional_binding_hashes = (
        "prior_same_form_universe_record_sha256",
        "prior_same_form_source_record_sha256",
        "prior_same_form_filing_sha256",
        "extraction_evidence_sha256",
        "extraction_output_sha256",
        "extraction_output_canonical_sha256",
    )
    for field in required_binding_hashes:
        _sha256(bindings[field], f"owned development feature bindings.{field}")
    for field in optional_binding_hashes:
        _optional_sha256(
            bindings[field], f"owned development feature bindings.{field}"
        )
    if (
        bindings["contract_sha256"] != canonical_sha256(build_contract_manifest())
        or bindings["corpus_universe_sha256"]
        != feature_assembly_plan["corpus_universe_sha256"]
        or bindings["market_stage_manifest_sha256"]
        != feature_assembly_plan["development_market_stage_manifest_sha256"]
        or bindings["source_manifest_sha256"]
        != feature_assembly_plan["development_market_source_manifest_sha256"]
        or value["bindings_sha256"] != canonical_sha256(bindings)
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development feature bindings crossed the assembly plan"
        )

    price = _validated_feature_hex_mapping(
        value["price_regime_features_hex"],
        names=PRICE_FEATURE_COLUMNS,
        location="owned development price features",
    )
    sentiment = _validated_feature_hex_mapping(
        value["market_sentiment_features_hex"],
        names=MARKET_SENTIMENT_FEATURE_COLUMNS,
        location="owned development sentiment features",
    )
    calendar = _validated_feature_hex_mapping(
        value["filing_calendar_features_hex"],
        names=FILING_CALENDAR_FEATURE_COLUMNS,
        location="owned development calendar features",
    )
    semantic = _validated_feature_hex_mapping(
        value["semantic_aggregate_features_hex"],
        names=SEMANTIC_AGGREGATE_FEATURE_COLUMNS,
        location="owned development semantic features",
    )
    expected_market_available = not missing and all(
        item is not None for item in (*price.values(), *sentiment.values())
    )
    if value["market_available"] is not expected_market_available:
        raise SecFilingGemmaFeatureError(
            "Owned development market feature availability changed"
        )
    if (
        value["semantic_feature_names"] != list(SEMANTIC_FEATURE_COLUMNS)
        or value["ablation_feature_names"] != list(ABLATION_FEATURE_COLUMNS)
        or value["semantic_feature_schema_sha256"]
        != canonical_sha256(list(SEMANTIC_FEATURE_COLUMNS))
        or value["ablation_feature_schema_sha256"]
        != canonical_sha256(list(ABLATION_FEATURE_COLUMNS))
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development feature schema changed"
        )
    expected_semantic_vector = [
        *(price[name] for name in PRICE_FEATURE_COLUMNS),
        *(sentiment[name] for name in MARKET_SENTIMENT_FEATURE_COLUMNS),
        *(calendar[name] for name in FILING_CALENDAR_FEATURE_COLUMNS),
        *(semantic[name] for name in SEMANTIC_AGGREGATE_FEATURE_COLUMNS),
    ]
    expected_ablation_vector = [
        *(price[name] for name in PRICE_FEATURE_COLUMNS),
        *(sentiment[name] for name in MARKET_SENTIMENT_FEATURE_COLUMNS),
        *(calendar[name] for name in FILING_CALENDAR_FEATURE_COLUMNS),
        *(float(0.0).hex() for _ in SEMANTIC_AGGREGATE_FEATURE_COLUMNS),
    ]
    if value["prediction_available"]:
        if (
            any(item is None for item in expected_semantic_vector)
            or value["semantic_feature_values_hex"] != expected_semantic_vector
            or value["ablation_feature_values_hex"] != expected_ablation_vector
        ):
            raise SecFilingGemmaFeatureError(
                "Owned development available feature vector changed"
            )
    elif (
        value["semantic_feature_values_hex"] is not None
        or value["ablation_feature_values_hex"] is not None
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development unavailable feature row exposed a vector"
        )

    market_feature_body = {
        "schema_version": MARKET_FEATURE_ROW_SCHEMA_VERSION,
        "accession_number": value["accession_number"],
        "decision_session": value["decision_session"],
        "market_prefix_chain_identity_sha256": bindings[
            "market_prefix_chain_identity_sha256"
        ],
        "market_available": value["market_available"],
        "missing_market_observations": missing,
        "missing_market_observations_sha256": value[
            "missing_market_observations_sha256"
        ],
        "price_regime_features_hex": price,
        "market_sentiment_features_hex": sentiment,
    }
    if value["market_feature_row_sha256"] != canonical_sha256(market_feature_body):
        raise SecFilingGemmaFeatureError(
            "Owned development market feature-row checksum changed"
        )
    body = {key: value[key] for key in value if key != "feature_row_sha256"}
    observed = _sha256(value["feature_row_sha256"], "feature_row_sha256")
    if observed != canonical_sha256(body):
        raise SecFilingGemmaFeatureError(
            "Owned development feature-row checksum changed"
        )
    return copy.deepcopy(dict(value))


def build_owned_development_feature_batch(
    *,
    feature_assembly_plan: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a self-hashed, explicitly non-authorizing development feature batch."""

    plan = _validated_development_feature_assembly_plan(feature_assembly_plan)
    if isinstance(feature_rows, (str, bytes)) or not isinstance(feature_rows, Sequence):
        raise SecFilingGemmaFeatureError("Owned development feature rows must be a sequence")
    rows = list(feature_rows)
    if len(rows) != plan["event_count"]:
        raise SecFilingGemmaFeatureError(
            "Owned development feature-row count differs from the assembly plan"
        )
    validated_rows = [
        _validated_batch_feature_row(
            row,
            event_plan_item=event_plan_item,
            feature_assembly_plan=plan,
        )
        for row, event_plan_item in zip(rows, plan["event_plan"], strict=True)
    ]
    row_hashes = [row["feature_row_sha256"] for row in validated_rows]
    body = {
        "schema_version": OWNED_DEVELOPMENT_FEATURE_BATCH_SCHEMA_VERSION,
        "development_root_scope_sha256": plan[
            "development_root_scope_sha256"
        ],
        "feature_assembly_plan_sha256": plan["feature_assembly_plan_sha256"],
        "candidate_sha256": plan["candidate_sha256"],
        "corpus_universe_sha256": plan["corpus_universe_sha256"],
        "development_sec_reader_receipt_sha256": plan[
            "development_sec_reader_receipt_sha256"
        ],
        "development_market_reader_receipt_sha256": plan[
            "development_market_reader_receipt_sha256"
        ],
        "development_model_reader_receipt_sha256": plan[
            "development_model_reader_receipt_sha256"
        ],
        "event_count": plan["event_count"],
        "event_plan_sha256": plan["event_plan_sha256"],
        "feature_row_schema_version": FEATURE_ROW_SCHEMA_VERSION,
        "feature_row_sha256s": row_hashes,
        "feature_rows_sha256": canonical_sha256(validated_rows),
        "feature_rows": validated_rows,
        "labels_included": False,
        "outcomes_included": False,
        "post_decision_market_rows_included": False,
        "training_membership_included": False,
        "learner_fit_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    return {**body, "feature_batch_sha256": canonical_sha256(body)}


def validate_owned_development_feature_batch(
    batch: Mapping[str, Any],
    *,
    feature_assembly_plan: Mapping[str, Any],
    expected_feature_assembly_plan_sha256: str,
    expected_feature_batch_sha256: str,
) -> str:
    """Rebuild an owned development feature batch and require both external pins."""

    value = _expect_mapping(batch, "owned development feature batch")
    _expect_keys(
        value,
        set(_OWNED_DEVELOPMENT_FEATURE_BATCH_KEYS),
        "owned development feature batch",
    )
    if type(value["event_count"]) is not int or value["event_count"] < 1:
        raise SecFilingGemmaFeatureError(
            "Owned development feature batch event_count must be an exact positive integer"
        )
    expected_flags = {
        "labels_included": False,
        "outcomes_included": False,
        "post_decision_market_rows_included": False,
        "training_membership_included": False,
        "learner_fit_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    if any(type(value[field]) is not bool for field in expected_flags):
        raise SecFilingGemmaFeatureError(
            "Owned development feature batch authority flags must be exact booleans"
        )
    if any(value[field] is not expected for field, expected in expected_flags.items()):
        raise SecFilingGemmaFeatureError(
            "Owned development feature batch crossed its non-authorizing boundary"
        )
    observed = _sha256(value["feature_batch_sha256"], "feature_batch_sha256")
    supplied_body = {
        key: value[key] for key in value if key != "feature_batch_sha256"
    }
    if canonical_sha256(supplied_body) != observed:
        raise SecFilingGemmaFeatureError(
            "Owned development feature batch checksum changed"
        )
    plan = _validated_development_feature_assembly_plan(feature_assembly_plan)
    expected_plan_hash = _sha256(
        expected_feature_assembly_plan_sha256,
        "expected_feature_assembly_plan_sha256",
    )
    if plan["feature_assembly_plan_sha256"] != expected_plan_hash:
        raise SecFilingGemmaFeatureError(
            "Owned development feature assembly plan is not externally pinned"
        )
    rows = value["feature_rows"]
    if type(rows) is not list:
        raise SecFilingGemmaFeatureError(
            "Owned development feature batch rows must be an exact list"
        )
    rebuilt = build_owned_development_feature_batch(
        feature_assembly_plan=plan,
        feature_rows=rows,
    )
    if dict(value) != rebuilt:
        raise SecFilingGemmaFeatureError(
            "Owned development feature batch differs from exact replay"
        )
    if observed != _sha256(
        expected_feature_batch_sha256, "expected_feature_batch_sha256"
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development feature batch is not externally pinned"
        )
    return observed


def _validated_development_label_assembly_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the authorization-owned development label plan exactly."""

    value = _expect_mapping(plan, "development label assembly plan")
    try:
        from agent_benchmark.sec_filing_gemma_stage_authorization import (
            validate_development_label_assembly_plan,
        )

        observed = validate_development_label_assembly_plan(
            value,
            expected_label_assembly_plan_sha256=value.get(
                "label_assembly_plan_sha256"
            ),
        )
    except Exception:
        raise SecFilingGemmaFeatureError(
            "Development label assembly plan failed exact authorization replay"
        ) from None
    if (
        value.get("schema_version")
        != DEVELOPMENT_LABEL_ASSEMBLY_PLAN_SCHEMA_VERSION
        or observed != value.get("label_assembly_plan_sha256")
    ):
        raise SecFilingGemmaFeatureError(
            "Development label assembly plan identity changed"
        )
    return copy.deepcopy(dict(value))


def _positive_float_hex(value: Any, location: str) -> float:
    if type(value) is not str:
        raise SecFilingGemmaFeatureError(
            f"{location} must be canonical positive float.hex text"
        )
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise SecFilingGemmaFeatureError(
            f"{location} must be canonical positive float.hex text"
        ) from exc
    if not math.isfinite(number) or number <= 0.0 or number.hex() != value:
        raise SecFilingGemmaFeatureError(
            f"{location} must be canonical positive float.hex text"
        )
    return number


def _validated_compact_development_label_evidence(
    row: Mapping[str, Any],
    *,
    maturity_item: Mapping[str, Any],
    audit_row: Mapping[str, Any],
    feature_row: Mapping[str, Any],
    label_assembly_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one compact t+1..t+21 label without accepting full market rows."""

    value = _expect_mapping(row, "owned development label evidence")
    _expect_keys(
        value,
        set(_LABEL_EVIDENCE_KEYS),
        "owned development label evidence",
    )
    if maturity_item["matured_by_development_cutoff"] is not True:
        raise SecFilingGemmaFeatureError(
            "An unmatured development event cannot carry label evidence"
        )
    decision_session = maturity_item["decision_session"]
    try:
        decision_index = EXPECTED_MARKET_HISTORY_SESSIONS.index(decision_session)
    except ValueError:
        raise SecFilingGemmaFeatureError(
            "Development label decision session is outside the frozen calendar"
        ) from None
    expected_sessions = list(
        EXPECTED_MARKET_HISTORY_SESSIONS[
            decision_index + LABEL_ENTRY_OFFSET :
            decision_index + LABEL_MATURITY_OFFSET + 1
        ]
    )
    if (
        len(expected_sessions) != LABEL_MATURITY_OFFSET
        or expected_sessions[-1] != maturity_item["label_maturity_session"]
        or expected_sessions[-1] > label_assembly_plan["development_cutoff_session"]
    ):
        raise SecFilingGemmaFeatureError(
            "Development label horizon crossed its frozen maturity boundary"
        )

    bindings = _expect_mapping(
        feature_row.get("bindings"), "owned development label feature bindings"
    )
    source_feature_plan = label_assembly_plan["source_feature_assembly_plan"]
    identity_matches = (
        value["schema_version"] == LABEL_EVIDENCE_SCHEMA_VERSION
        and value["accession_number"] == maturity_item["accession_number"]
        and value["accession_number"] == audit_row["accession_number"]
        and value["decision_session"] == decision_session
        and value["decision_session"] == audit_row["decision_session"]
        and value["entry_session"] == expected_sessions[0]
        and value["exit_session"] == expected_sessions[-1]
        and value["label_maturity_session"] == expected_sessions[-1]
        and type(value["horizon_sessions"]) is int
        and value["horizon_sessions"] == HORIZON_SESSIONS
        and type(value["entry_session_offset"]) is int
        and value["entry_session_offset"] == LABEL_ENTRY_OFFSET
        and type(value["label_maturity_session_offset"]) is int
        and value["label_maturity_session_offset"] == LABEL_MATURITY_OFFSET
        and value["feature_row_sha256"] == feature_row["feature_row_sha256"]
        and value["feature_row_sha256"] == audit_row["feature_row_sha256"]
        and value["market_feature_row_sha256"]
        == feature_row["market_feature_row_sha256"]
        and value["extraction_identity_sha256"]
        == bindings.get("extraction_identity_sha256")
        and value["market_prefix_sha256"]
        == bindings.get("market_prefix_sha256")
        and value["market_prefix_proof_sha256"]
        == bindings.get("market_prefix_proof_sha256")
        and value["market_stage_manifest_sha256"]
        == bindings.get("market_stage_manifest_sha256")
        and value["market_stage_manifest_sha256"]
        == source_feature_plan["development_market_stage_manifest_sha256"]
        and value["source_manifest_sha256"]
        == bindings.get("source_manifest_sha256")
        and value["source_manifest_sha256"]
        == source_feature_plan["development_market_source_manifest_sha256"]
    )
    if not identity_matches:
        raise SecFilingGemmaFeatureError(
            "Owned development label evidence crossed its event or feature binding"
        )
    hash_fields = (
        "market_prefix_sha256",
        "market_prefix_proof_sha256",
        "market_stage_manifest_sha256",
        "source_manifest_sha256",
        "feature_row_sha256",
        "market_feature_row_sha256",
        "extraction_identity_sha256",
        "future_market_rows_sha256",
        "future_market_row_chain_tip_sha256",
        "entry_source_market_row_sha256",
        "exit_source_market_row_sha256",
        "adjusted_open_path_sha256",
        "label_evidence_sha256",
    )
    for field in hash_fields:
        _sha256(value[field], f"owned development label evidence.{field}")

    path = value["adjusted_open_path"]
    if type(path) is not list or len(path) != LABEL_MATURITY_OFFSET:
        raise SecFilingGemmaFeatureError(
            "Owned development adjusted-open path must contain exactly 21 sessions"
        )
    normalized_path: list[dict[str, str]] = []
    for index, (raw_item, expected_session) in enumerate(
        zip(path, expected_sessions, strict=True)
    ):
        item = _expect_mapping(
            raw_item, f"owned development adjusted-open path {index}"
        )
        _expect_keys(
            item,
            set(_COMPACT_ADJUSTED_OPEN_PATH_ITEM_KEYS),
            f"owned development adjusted-open path {index}",
        )
        if item["session"] != expected_session:
            raise SecFilingGemmaFeatureError(
                "Owned development adjusted-open path sessions changed"
            )
        _positive_float_hex(
            item["adjusted_open_hex"],
            f"owned development adjusted-open path {index}.adjusted_open_hex",
        )
        _sha256(
            item["source_market_row_sha256"],
            f"owned development adjusted-open path {index}.source_market_row_sha256",
        )
        normalized_path.append(dict(item))
    if (
        type(value["future_market_row_count"]) is not int
        or value["future_market_row_count"] != LABEL_MATURITY_OFFSET
        or value["adjusted_open_path_sha256"] != canonical_sha256(normalized_path)
        or value["future_market_row_chain_tip_sha256"]
        != normalized_path[-1]["source_market_row_sha256"]
        or value["entry_source_market_row_sha256"]
        != normalized_path[0]["source_market_row_sha256"]
        or value["exit_source_market_row_sha256"]
        != normalized_path[-1]["source_market_row_sha256"]
        or value["entry_adjusted_open_hex"]
        != normalized_path[0]["adjusted_open_hex"]
        or value["exit_adjusted_open_hex"]
        != normalized_path[-1]["adjusted_open_hex"]
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development adjusted-open path identity changed"
        )
    source_row_hashes = [
        item["source_market_row_sha256"] for item in normalized_path
    ]
    if len(source_row_hashes) != len(set(source_row_hashes)):
        raise SecFilingGemmaFeatureError(
            "Owned development adjusted-open path repeats a market row"
        )

    entry = _positive_float_hex(
        value["entry_adjusted_open_hex"], "entry_adjusted_open_hex"
    )
    exit_value = _positive_float_hex(
        value["exit_adjusted_open_hex"], "exit_adjusted_open_hex"
    )
    forward_return = math.log(exit_value / entry)
    cost_5 = _cash_round_trip_log_factor(5)
    cost_10 = _cash_round_trip_log_factor(10)
    edge_5 = cost_5 - forward_return
    edge_10 = cost_10 - forward_return
    if not all(
        math.isfinite(number)
        for number in (forward_return, cost_5, cost_10, edge_5, edge_10)
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development label arithmetic is non-finite"
        )
    expected_hex = {
        "aapl_forward_log_return_20_hex": forward_return.hex(),
        "cash_round_trip_log_cost_5bps_hex": cost_5.hex(),
        "cash_round_trip_log_cost_10bps_hex": cost_10.hex(),
        "cash_active_log_edge_5bps_hex": edge_5.hex(),
        "cash_active_log_edge_10bps_hex": edge_10.hex(),
        "binary_comparison_tolerance_hex": float(ACTIVE_EDGE_TOLERANCE).hex(),
    }
    if any(value[field] != expected for field, expected in expected_hex.items()):
        raise SecFilingGemmaFeatureError(
            "Owned development label arithmetic changed"
        )
    if (
        type(value["cash_beats_long_5bps"]) is not bool
        or type(value["cash_beats_long_10bps"]) is not bool
        or value["cash_beats_long_5bps"] is not (edge_5 > ACTIVE_EDGE_TOLERANCE)
        or value["cash_beats_long_10bps"] is not (edge_10 > ACTIVE_EDGE_TOLERANCE)
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development binary label boundary changed"
        )
    body = {key: value[key] for key in value if key != "label_evidence_sha256"}
    observed = _sha256(
        value["label_evidence_sha256"], "label_evidence_sha256"
    )
    if (
        observed != canonical_sha256(body)
        or observed != audit_row["label_evidence_sha256"]
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development label evidence checksum changed"
        )
    return copy.deepcopy(dict(value))


def build_owned_development_label_batch(
    *,
    label_assembly_plan: Mapping[str, Any],
    source_feature_batch: Mapping[str, Any],
    maturity_audit_rows: Sequence[Mapping[str, Any]],
    label_evidence_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the development-only label batch without authorizing learner use."""

    plan = _validated_development_label_assembly_plan(label_assembly_plan)
    source_feature_plan = plan["source_feature_assembly_plan"]
    source_batch = _expect_mapping(
        source_feature_batch, "owned development source feature batch"
    )
    try:
        validate_owned_development_feature_batch(
            source_batch,
            feature_assembly_plan=source_feature_plan,
            expected_feature_assembly_plan_sha256=plan[
                "source_feature_assembly_plan_sha256"
            ],
            expected_feature_batch_sha256=source_batch.get(
                "feature_batch_sha256"
            ),
        )
    except Exception:
        raise SecFilingGemmaFeatureError(
            "Development label source feature batch failed exact replay"
        ) from None
    if (
        source_batch["development_root_scope_sha256"]
        != plan["development_root_scope_sha256"]
        or source_batch["feature_assembly_plan_sha256"]
        != plan["source_feature_assembly_plan_sha256"]
        or source_batch["event_count"] != plan["event_count"]
        or source_batch["event_plan_sha256"]
        != source_feature_plan["event_plan_sha256"]
    ):
        raise SecFilingGemmaFeatureError(
            "Development label source feature batch crossed its assembly plan"
        )
    if (
        isinstance(maturity_audit_rows, (str, bytes))
        or not isinstance(maturity_audit_rows, Sequence)
        or isinstance(label_evidence_rows, (str, bytes))
        or not isinstance(label_evidence_rows, Sequence)
    ):
        raise SecFilingGemmaFeatureError(
            "Development label audit and evidence rows must be sequences"
        )
    audits = list(maturity_audit_rows)
    labels = list(label_evidence_rows)
    maturity_plan = plan["maturity_plan"]
    feature_rows = source_batch["feature_rows"]
    if (
        len(audits) != plan["event_count"]
        or len(audits) != len(maturity_plan)
        or len(audits) != len(feature_rows)
        or len(labels) != plan["matured_event_count"]
    ):
        raise SecFilingGemmaFeatureError(
            "Development label audit or evidence count changed"
        )

    normalized_audits: list[dict[str, Any]] = []
    for ordinal, (raw_audit, maturity_item, feature_row) in enumerate(
        zip(audits, maturity_plan, feature_rows, strict=True), start=1
    ):
        audit = _expect_mapping(
            raw_audit, f"owned development label maturity audit {ordinal}"
        )
        _expect_keys(
            audit,
            set(_DEVELOPMENT_LABEL_AUDIT_ROW_KEYS),
            f"owned development label maturity audit {ordinal}",
        )
        matured = maturity_item["matured_by_development_cutoff"]
        if (
            type(audit["event_ordinal"]) is not int
            or audit["event_ordinal"] != ordinal
            or audit["event_ordinal"] != maturity_item["event_ordinal"]
            or audit["accession_number"] != maturity_item["accession_number"]
            or audit["accession_number"] != feature_row["accession_number"]
            or audit["decision_session"] != maturity_item["decision_session"]
            or audit["decision_session"] != feature_row["decision_session"]
            or audit["feature_row_sha256"] != feature_row["feature_row_sha256"]
            or audit["label_maturity_session"]
            != maturity_item["label_maturity_session"]
            or type(audit["matured_by_development_cutoff"]) is not bool
            or audit["matured_by_development_cutoff"] is not matured
        ):
            raise SecFilingGemmaFeatureError(
                "Development label maturity audit crossed or reordered an event"
            )
        _sha256(audit["feature_row_sha256"], "feature_row_sha256")
        if matured:
            _sha256(
                audit["label_evidence_sha256"], "label_evidence_sha256"
            )
        elif audit["label_evidence_sha256"] is not None:
            raise SecFilingGemmaFeatureError(
                "An unmatured development event exposed label evidence"
            )
        normalized_audits.append(copy.deepcopy(dict(audit)))

    matured_triplets = [
        (maturity_item, audit, feature_row)
        for maturity_item, audit, feature_row in zip(
            maturity_plan, normalized_audits, feature_rows, strict=True
        )
        if maturity_item["matured_by_development_cutoff"]
    ]
    normalized_labels = [
        _validated_compact_development_label_evidence(
            raw_label,
            maturity_item=maturity_item,
            audit_row=audit,
            feature_row=feature_row,
            label_assembly_plan=plan,
        )
        for raw_label, (maturity_item, audit, feature_row) in zip(
            labels, matured_triplets, strict=True
        )
    ]
    label_hashes = [row["label_evidence_sha256"] for row in normalized_labels]
    body = {
        "schema_version": OWNED_DEVELOPMENT_LABEL_BATCH_SCHEMA_VERSION,
        "development_root_scope_sha256": plan[
            "development_root_scope_sha256"
        ],
        "label_assembly_plan_sha256": plan["label_assembly_plan_sha256"],
        "source_feature_assembly_plan_sha256": plan[
            "source_feature_assembly_plan_sha256"
        ],
        "source_feature_batch_sha256": source_batch["feature_batch_sha256"],
        "candidate_sha256": source_feature_plan["candidate_sha256"],
        "corpus_universe_sha256": source_feature_plan[
            "corpus_universe_sha256"
        ],
        "development_market_reader_receipt_sha256": source_feature_plan[
            "development_market_reader_receipt_sha256"
        ],
        "development_cutoff_session": plan["development_cutoff_session"],
        "event_count": plan["event_count"],
        "matured_label_count": plan["matured_event_count"],
        "unmatured_event_count": plan["unmatured_event_count"],
        "maturity_audit_rows": normalized_audits,
        "maturity_audit_rows_sha256": canonical_sha256(normalized_audits),
        "label_evidence_schema_version": LABEL_EVIDENCE_SCHEMA_VERSION,
        "label_evidence_sha256s": label_hashes,
        "label_evidence_rows": normalized_labels,
        "label_evidence_rows_sha256": canonical_sha256(normalized_labels),
        "development_labels_included": True,
        "development_outcomes_included": True,
        "compact_adjusted_open_paths_included": True,
        "full_market_rows_included": False,
        "post_cutoff_market_data_included": False,
        "training_membership_included": False,
        "learner_fit_authorized": False,
        "prediction_authorized": False,
        "holdout_access_authorized": False,
        "ledger_mutation_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    return {**body, "label_batch_sha256": canonical_sha256(body)}


def validate_owned_development_label_batch(
    batch: Mapping[str, Any],
    *,
    label_assembly_plan: Mapping[str, Any],
    expected_label_assembly_plan_sha256: str,
    source_feature_batch: Mapping[str, Any],
    expected_source_feature_batch_sha256: str,
    expected_label_batch_sha256: str,
) -> str:
    """Rebuild the compact label batch and require all three external pins."""

    value = _expect_mapping(batch, "owned development label batch")
    _expect_keys(
        value,
        set(_OWNED_DEVELOPMENT_LABEL_BATCH_KEYS),
        "owned development label batch",
    )
    for field in (
        "event_count",
        "matured_label_count",
        "unmatured_event_count",
    ):
        if type(value[field]) is not int or value[field] < 0:
            raise SecFilingGemmaFeatureError(
                f"Owned development label batch {field} must be an exact nonnegative integer"
            )
    expected_flags = {
        "development_labels_included": True,
        "development_outcomes_included": True,
        "compact_adjusted_open_paths_included": True,
        "full_market_rows_included": False,
        "post_cutoff_market_data_included": False,
        "training_membership_included": False,
        "learner_fit_authorized": False,
        "prediction_authorized": False,
        "holdout_access_authorized": False,
        "ledger_mutation_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    if any(type(value[field]) is not bool for field in expected_flags):
        raise SecFilingGemmaFeatureError(
            "Owned development label batch authority flags must be exact booleans"
        )
    if any(value[field] is not expected for field, expected in expected_flags.items()):
        raise SecFilingGemmaFeatureError(
            "Owned development label batch crossed its non-authorizing boundary"
        )
    observed = _sha256(value["label_batch_sha256"], "label_batch_sha256")
    supplied_body = {
        key: value[key] for key in value if key != "label_batch_sha256"
    }
    if canonical_sha256(supplied_body) != observed:
        raise SecFilingGemmaFeatureError(
            "Owned development label batch checksum changed"
        )
    plan = _validated_development_label_assembly_plan(label_assembly_plan)
    if plan["label_assembly_plan_sha256"] != _sha256(
        expected_label_assembly_plan_sha256,
        "expected_label_assembly_plan_sha256",
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development label assembly plan is not externally pinned"
        )
    source_batch = _expect_mapping(
        source_feature_batch, "owned development source feature batch"
    )
    if source_batch.get("feature_batch_sha256") != _sha256(
        expected_source_feature_batch_sha256,
        "expected_source_feature_batch_sha256",
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development source feature batch is not externally pinned"
        )
    rebuilt = build_owned_development_label_batch(
        label_assembly_plan=plan,
        source_feature_batch=source_batch,
        maturity_audit_rows=value["maturity_audit_rows"],
        label_evidence_rows=value["label_evidence_rows"],
    )
    if dict(value) != rebuilt:
        raise SecFilingGemmaFeatureError(
            "Owned development label batch differs from exact replay"
        )
    if observed != _sha256(
        expected_label_batch_sha256, "expected_label_batch_sha256"
    ):
        raise SecFilingGemmaFeatureError(
            "Owned development label batch is not externally pinned"
        )
    return observed


def _cash_round_trip_log_factor(cost_bps: int) -> float:
    rate = cost_bps / 10_000.0
    return math.log1p(-rate) - math.log1p(rate)


def build_twenty_session_label_evidence(
    *,
    market_prefix: Mapping[str, Any],
    market_prefix_proof: Mapping[str, Any],
    expected_market_prefix_proof_sha256: str,
    stage_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    expected_artifact_stage: str,
    expected_source_manifest_sha256: str,
    expected_market_stage_manifest_sha256: str,
    feature_row: Mapping[str, Any],
    expected_feature_row_sha256: str,
    universe_event_proof: Mapping[str, Any],
    expected_universe_event_proof_sha256: str,
    extraction_event_proof: Mapping[str, Any],
    expected_extraction_event_proof_sha256: str,
    session_dates: Sequence[str] = EXPECTED_MARKET_HISTORY_SESSIONS,
) -> dict[str, Any]:
    """Derive t+1..t+21 only from the authoritative validated stage frame."""

    market_proof = validate_market_prefix_proof(
        market_prefix_proof,
        prefix=market_prefix,
        expected_market_prefix_proof_sha256=expected_market_prefix_proof_sha256,
    )
    prefix = market_proof["compact_prefix"]
    feature_hash = validate_sec_filing_gemma_feature_row(
        feature_row,
        expected_feature_row_sha256=expected_feature_row_sha256,
        market_prefix=market_prefix,
        market_prefix_proof=market_prefix_proof,
        expected_market_prefix_proof_sha256=expected_market_prefix_proof_sha256,
        universe_event_proof=universe_event_proof,
        expected_universe_event_proof_sha256=expected_universe_event_proof_sha256,
        extraction_event_proof=extraction_event_proof,
        expected_extraction_event_proof_sha256=(
            expected_extraction_event_proof_sha256
        ),
    )
    bindings = feature_row["bindings"]
    if (
        feature_row["accession_number"] != prefix["decision_event_id"]
        or feature_row["decision_session"] != prefix["decision_session"]
        or bindings["market_prefix_sha256"] != prefix["market_prefix_sha256"]
        or bindings["market_prefix_proof_sha256"]
        != market_proof["market_prefix_proof_sha256"]
        or bindings["market_stage_manifest_sha256"]
        != expected_market_stage_manifest_sha256
        or bindings["source_manifest_sha256"] != expected_source_manifest_sha256
    ):
        raise SecFilingGemmaFeatureError(
            "Label feature row is not bound to the validated market event"
        )
    validate_market_stage_manifest(
        stage_manifest,
        source_manifest=source_manifest,
        expected_artifact_stage=expected_artifact_stage,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_market_stage_manifest_sha256=expected_market_stage_manifest_sha256,
        session_dates=session_dates,
    )
    if (
        expected_artifact_stage != prefix["artifact_stage"]
        or stage_manifest["market_stage_manifest_sha256"]
        != prefix["market_stage_manifest_sha256"]
        or source_manifest["source_manifest_sha256"]
        != prefix["source_manifest_sha256"]
    ):
        raise SecFilingGemmaFeatureError(
            "Label stage/source differs from the prediction-time proof"
        )
    decision = prefix["decision_session"]
    expected_sessions = tuple(
        _offset_session(decision, offset, market_calendar=True)
        for offset in range(1, LABEL_MATURITY_OFFSET + 1)
    )
    rows_by_session = {
        row["session"]: row for row in stage_manifest["rows"]
    }
    future_market_rows = [
        rows_by_session.get(session) for session in expected_sessions
    ]
    if any(row is None for row in future_market_rows):
        raise SecFilingGemmaFeatureError(
            "Validated market stage does not cover the complete label horizon"
        )
    parent = prefix["full_prefix_row_chain_tip_sha256"]
    first_index = prefix["lookback_end_row_index"] + 1
    rows: list[dict[str, Any]] = []
    for index, expected_session in enumerate(expected_sessions):
        row = _validate_market_row(
            future_market_rows[index],
            expected_index=first_index + index,
            expected_session=expected_session,
            expected_parent_sha256=parent,
            location=f"future_market_rows[{index}]",
        )
        rows.append(row)
        parent = row["row_sha256"]
    future_rows_hash = canonical_sha256(rows)

    open_path: list[dict[str, str]] = []
    for row in rows:
        observation = row["observations"]["AAPL"]
        if not observation["available"]:
            raise SecFilingGemmaFeatureError("AAPL label path contains a missing open")
        adjusted_open_hex = observation["adjusted_open_hex"]
        adjusted_open = decode_float_hex(
            adjusted_open_hex, f"{row['session']}.AAPL.adjusted_open"
        )
        if adjusted_open <= 0.0:
            raise SecFilingGemmaFeatureError("AAPL label path contains an invalid open")
        open_path.append(
            {
                "session": row["session"],
                "adjusted_open_hex": adjusted_open_hex,
                "source_market_row_sha256": row["row_sha256"],
            }
        )
    entry = decode_float_hex(open_path[0]["adjusted_open_hex"], "entry adjusted open")
    exit_value = decode_float_hex(
        open_path[-1]["adjusted_open_hex"], "exit adjusted open"
    )
    long_log_return = math.log(exit_value / entry)
    edges = {
        cost: _cash_round_trip_log_factor(cost) - long_log_return
        for cost in TRANSACTION_COST_BPS
    }
    body = {
        "schema_version": LABEL_EVIDENCE_SCHEMA_VERSION,
        "accession_number": prefix["decision_event_id"],
        "decision_session": decision,
        "entry_session": expected_sessions[0],
        "exit_session": expected_sessions[-1],
        "label_maturity_session": expected_sessions[-1],
        "horizon_sessions": HORIZON_SESSIONS,
        "entry_session_offset": LABEL_ENTRY_OFFSET,
        "label_maturity_session_offset": LABEL_MATURITY_OFFSET,
        "market_prefix_sha256": prefix["market_prefix_sha256"],
        "market_prefix_proof_sha256": market_proof[
            "market_prefix_proof_sha256"
        ],
        "market_stage_manifest_sha256": stage_manifest[
            "market_stage_manifest_sha256"
        ],
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "feature_row_sha256": feature_hash,
        "market_feature_row_sha256": feature_row["market_feature_row_sha256"],
        "extraction_identity_sha256": bindings["extraction_identity_sha256"],
        "future_market_rows_sha256": future_rows_hash,
        "future_market_row_count": len(rows),
        "future_market_row_chain_tip_sha256": rows[-1]["row_sha256"],
        "entry_source_market_row_sha256": rows[0]["row_sha256"],
        "exit_source_market_row_sha256": rows[-1]["row_sha256"],
        "adjusted_open_path": open_path,
        "adjusted_open_path_sha256": canonical_sha256(open_path),
        "entry_adjusted_open_hex": open_path[0]["adjusted_open_hex"],
        "exit_adjusted_open_hex": open_path[-1]["adjusted_open_hex"],
        "aapl_forward_log_return_20_hex": _float_hex(long_log_return),
        "cash_round_trip_log_cost_5bps_hex": _float_hex(
            _cash_round_trip_log_factor(5)
        ),
        "cash_round_trip_log_cost_10bps_hex": _float_hex(
            _cash_round_trip_log_factor(10)
        ),
        "cash_active_log_edge_5bps_hex": _float_hex(edges[5]),
        "cash_active_log_edge_10bps_hex": _float_hex(edges[10]),
        "cash_beats_long_5bps": edges[5] > ACTIVE_EDGE_TOLERANCE,
        "cash_beats_long_10bps": edges[10] > ACTIVE_EDGE_TOLERANCE,
        "binary_comparison_tolerance_hex": _float_hex(ACTIVE_EDGE_TOLERANCE),
    }
    return {**body, "label_evidence_sha256": canonical_sha256(body)}


def validate_twenty_session_label_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_label_evidence_sha256: str,
    market_prefix: Mapping[str, Any],
    market_prefix_proof: Mapping[str, Any],
    expected_market_prefix_proof_sha256: str,
    stage_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    expected_artifact_stage: str,
    expected_source_manifest_sha256: str,
    expected_market_stage_manifest_sha256: str,
    feature_row: Mapping[str, Any],
    expected_feature_row_sha256: str,
    universe_event_proof: Mapping[str, Any],
    expected_universe_event_proof_sha256: str,
    extraction_event_proof: Mapping[str, Any],
    expected_extraction_event_proof_sha256: str,
    session_dates: Sequence[str] = EXPECTED_MARKET_HISTORY_SESSIONS,
) -> str:
    """Rebuild the full label from pinned sources and require byte semantics."""

    value = _expect_mapping(evidence, "label evidence")
    rebuilt = build_twenty_session_label_evidence(
        market_prefix=market_prefix,
        market_prefix_proof=market_prefix_proof,
        expected_market_prefix_proof_sha256=expected_market_prefix_proof_sha256,
        stage_manifest=stage_manifest,
        source_manifest=source_manifest,
        expected_artifact_stage=expected_artifact_stage,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_market_stage_manifest_sha256=(
            expected_market_stage_manifest_sha256
        ),
        feature_row=feature_row,
        expected_feature_row_sha256=expected_feature_row_sha256,
        universe_event_proof=universe_event_proof,
        expected_universe_event_proof_sha256=expected_universe_event_proof_sha256,
        extraction_event_proof=extraction_event_proof,
        expected_extraction_event_proof_sha256=(
            expected_extraction_event_proof_sha256
        ),
        session_dates=session_dates,
    )
    if dict(value) != rebuilt:
        raise SecFilingGemmaFeatureError(
            "Label evidence differs from the authoritative stage replay"
        )
    observed = _sha256(value.get("label_evidence_sha256"), "label_evidence_sha256")
    if not hmac.compare_digest(
        observed,
        _sha256(expected_label_evidence_sha256, "expected_label_evidence_sha256"),
    ):
        raise SecFilingGemmaFeatureError("Label evidence is not externally pinned")
    return observed


__all__ = [
    "ABLATION_FEATURE_COLUMNS",
    "COMPARATIVE_CHANGE_ENCODING",
    "CURRENT_IMPACT_ENCODING",
    "DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_SCHEMA_VERSION",
    "DEVELOPMENT_LABEL_ASSEMBLY_PLAN_SCHEMA_VERSION",
    "DIMENSION_GROUPS",
    "EXTRACTION_STATUSES",
    "FEATURE_ROW_SCHEMA_VERSION",
    "FEATURE_UNAVAILABLE_REASONS",
    "FILING_CALENDAR_FEATURE_COLUMNS",
    "LABEL_ENTRY_OFFSET",
    "LABEL_EVIDENCE_SCHEMA_VERSION",
    "LABEL_MATURITY_OFFSET",
    "MARKET_FEATURE_ROW_SCHEMA_VERSION",
    "MARKET_PREFIX_PROOF_SCHEMA_VERSION",
    "OWNED_DEVELOPMENT_FEATURE_BATCH_SCHEMA_VERSION",
    "OWNED_DEVELOPMENT_FEATURE_INPUTS_SCHEMA_VERSION",
    "OWNED_DEVELOPMENT_LABEL_BATCH_SCHEMA_VERSION",
    "OWNED_DEVELOPMENT_LABEL_PROJECTION_SCHEMA_VERSION",
    "SEMANTIC_AGGREGATE_FEATURE_COLUMNS",
    "SEMANTIC_FEATURE_COLUMNS",
    "UNIVERSE_EVENT_PROOF_SCHEMA_VERSION",
    "EXTRACTION_EVENT_PROOF_SCHEMA_VERSION",
    "SecFilingGemmaFeatureError",
    "build_owned_development_feature_batch",
    "build_owned_development_label_batch",
    "build_sec_filing_gemma_feature_row",
    "build_twenty_session_label_evidence",
    "build_validated_extraction_event_proof",
    "build_validated_market_prefix_proof",
    "build_validated_universe_event_proof",
    "validate_extraction_event_proof",
    "validate_market_prefix_proof",
    "validate_owned_development_feature_batch",
    "validate_owned_development_label_batch",
    "validate_sec_filing_gemma_feature_row",
    "validate_twenty_session_label_evidence",
    "validate_universe_event_proof",
]
