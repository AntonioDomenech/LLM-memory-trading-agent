"""Pure, fail-closed contract for the frozen SEC-filing Gemma experiment.

This module deliberately performs no filesystem, network, market-data, SEC,
or model I/O.  It defines the experiment before any semantic filing text or
post-2018 outcome is opened and validates the receipts produced by later,
effectful runners.
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

from agent_benchmark.sec_session_calendar import (
    CALENDAR_ID as AUTHORITATIVE_CALENDAR_ID,
    EXPECTED_SESSIONS as AUTHORITATIVE_SESSION_DATES,
    EXPECTED_MARKET_HISTORY_SESSIONS as AUTHORITATIVE_MARKET_SESSION_DATES,
    LEGACY_CALENDAR_ID as LEGACY_AUTHORITATIVE_CALENDAR_ID,
    LEGACY_EXPECTED_SESSIONS as LEGACY_AUTHORITATIVE_SESSION_DATES,
    MARKET_HISTORY_CALENDAR_ID as AUTHORITATIVE_MARKET_CALENDAR_ID,
    MARKET_HISTORY_CALENDAR_START as AUTHORITATIVE_MARKET_CALENDAR_START,
)


CONTRACT_VERSION: Final[str] = "aapl-sec-filing-gemma-v1"
CANDIDATE_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-candidate-v1"
STAGE_RECEIPT_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-stage-receipt-v1"
EXTRACTOR_SCHEMA_VERSION: Final[str] = "sec-filing-extractor-v1"
EXTRACTOR_REQUEST_VERSION: Final[str] = "issuer-relative-grounded-request-v1"
PREPROCESSOR_VERSION: Final[str] = "issuer-relative-period-grounded-sentences-v1"
LIVE_LESSON_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-live-lesson-v1"
UNIVERSE_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-corpus-universe-v1"
CONTENT_MANIFEST_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-stage-content-v1"
CALENDAR_SOURCE_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "aapl-nyse-calendar-source-evidence-v1"
)

CALENDAR_SOURCE_URLS: Final[dict[str, str]] = {
    "nyse_hours_calendars": "https://www.nyse.com/markets/hours-calendars",
    "nyse_2025_calendar_pdf": (
        "https://www.nyse.com/publicdocs/ICE_NYSE_2025_Yearly_Trading_Calendar.pdf"
    ),
    "nyse_2026_calendar_pdf": (
        "https://www.nyse.com/publicdocs/nyse/ICE_NYSE_2026_Yearly_Trading_Calendar.pdf"
    ),
    "carter_closure_notice": (
        "https://www.nyse.com/publicdocs/nyse/markets/american-options/"
        "rule-interpretations/2025/National_Day_of_Mourning_20250102.pdf"
    ),
}

MAX_RUNTIME_SECONDS: Final[int] = 3_600
MAX_SEC_SECONDS: Final[int] = 720
MAX_MODEL_SECONDS: Final[int] = 2_160
MAX_FIT_SECONDS: Final[int] = 480
MAX_SEC_REQUESTS: Final[int] = 1_000
MAX_SEC_BYTES: Final[int] = 1_610_612_736  # 1.5 GiB
MAX_INPUT_BYTES: Final[int] = 20_000
MAX_SENTENCES: Final[int] = 72
MAX_SENTENCE_CHARACTERS: Final[int] = 220
HORIZON_SESSIONS: Final[int] = 20
LABEL_MATURITY_OFFSET: Final[int] = HORIZON_SESSIONS + 1
ACTIVE_EDGE_TOLERANCE: Final[float] = 1e-12
BRIER_TARGET_COST_BPS: Final[int] = 10
MARKET_HISTORY_START: Final[str] = AUTHORITATIVE_MARKET_CALENDAR_START.isoformat()
MARKET_LOOKBACK_SESSIONS: Final[int] = 252

DEVELOPMENT_FOLD_SPECS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("fold_1", "2004-12-31", "2005-01-03", "2007-12-31"),
    ("fold_2", "2007-12-31", "2008-01-02", "2010-12-31"),
    ("fold_3", "2010-12-31", "2011-01-03", "2013-12-31"),
    ("fold_4", "2013-12-31", "2014-01-02", "2016-12-30"),
    ("fold_5", "2016-12-30", "2017-01-03", "2018-12-31"),
)

STAGE_ORDER: Final[tuple[str, ...]] = (
    "development",
    "intermediate",
    "final",
)
STAGE_WINDOWS: Final[dict[str, tuple[str, str]]] = {
    "development": ("2000-01-01", "2018-12-31"),
    "intermediate": ("2019-01-01", "2023-12-31"),
    "final": ("2024-01-01", "2026-07-09"),
}
STAGE_MODEL_CALL_CAPS: Final[dict[str, int]] = {
    "development": 80,
    "intermediate": 20,
    "final": 12,
}

DIMENSION_NAMES: Final[tuple[str, ...]] = (
    "demand",
    "pricing_power",
    "gross_margin",
    "operating_cost_pressure",
    "capital_allocation",
    "liquidity",
    "forward_guidance",
    "supply_chain",
    "legal_regulatory",
    "management_uncertainty",
)
FLAG_NAMES: Final[tuple[str, ...]] = (
    "new_material_risk",
    "guidance_withdrawn",
    "liquidity_stress",
    "restructuring_or_impairment",
    "internal_control_weakness",
    "management_transition",
)
ADVERSE_FLAG_NAMES: Final[tuple[str, ...]] = FLAG_NAMES[:-1]
CURRENT_IMPACTS: Final[frozenset[str]] = frozenset(
    {"favorable", "neutral", "unfavorable", "mixed", "not_stated"}
)
COMPARATIVE_CHANGES: Final[frozenset[str]] = frozenset(
    {
        "improving",
        "stable",
        "deteriorating",
        "mixed",
        "not_comparable",
        "not_stated",
    }
)
DOCUMENT_QUALITIES: Final[frozenset[str]] = frozenset(
    {"usable", "thin", "unusable"}
)
CANDIDATE_IDS: Final[tuple[str, ...]] = (
    "p50_e0",
    "p55_e0",
    "p50_e25",
    "p55_e25",
)
REQUIRED_SOURCE_HASHES: Final[tuple[str, ...]] = (
    "artifact_sealer",
    "calendar",
    "contract",
    "extractor",
    "extractor_prompt",
    "extractor_schema",
    "learner",
    "ledger",
    "market_features",
    "preprocessor",
    "reveal_registry",
    "runner",
    "scorer",
    "sec_acquirer",
    "sec_audit_verifier",
    "sec_corpus_selector",
    "stage_verifier",
)
REQUIRED_STAGE_VERIFIER_CHECKS: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            "artifact_replay",
            "candidate_identity",
            "chronology",
            "gate_replay",
            "ledger_replay",
            "no_leverage",
            "prediction_replay",
            "prerequisite_evidence_identity",
            "registry_identity",
            "request_identity",
            "runtime_budget",
            "source_identity",
            "stage_access_identity",
            "stage_identity",
            "zero_cost",
        }
    )
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_ACCESSION_RE = re.compile(r"[0-9]{10}-[0-9]{2}-[0-9]{6}\Z")
_SENTENCE_ID_RE = re.compile(r"[CP][0-9]{4}\Z")
_ACCEPTANCE_RE = re.compile(r"[0-9]{14}\Z")
_ATTEMPT_RE = re.compile(r"aapl-sec-filing-gemma-v1-attempt-[0-9]{3}\Z")
MANDATORY_IDENTITY_TERMS: Final[tuple[str, ...]] = (
    "aapl",
    "apple",
    "app store",
    "apple music",
    "apple pay",
    "apple tv",
    "apple watch",
    "applecare",
    "airpods",
    "beats",
    "cupertino",
    "greater china",
    "homepod",
    "icloud",
    "iphone",
    "ipad",
    "ipados",
    "ipod",
    "ios",
    "itunes",
    "mac",
    "macos",
    "macbook",
    "safari",
    "siri",
    "tim cook",
    "vision pro",
    "xcode",
    "0000320193",
)
EXTRACTOR_SYSTEM_PROMPT: Final[str] = (
    "Extract only evidence-grounded relative business conditions from the anonymized "
    "current periodic filing and its optional anonymized prior same-form filing. Use "
    "only supplied C and P sentence identifiers. Return exactly the required JSON "
    "schema. Do not infer or name the issuer, date, security, price, return, forecast, "
    "benchmark, or trading action."
)
_SPELLED_ABSOLUTE_TERMS = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "trillion",
        "dollar",
        "dollars",
        "euro",
        "euros",
        "yen",
        "pound",
        "pounds",
        "percent",
        "percentage",
    }
)
_SPELLED_DATE_TERMS = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
        "twenty-first",
        "twenty-second",
        "twenty-third",
        "twenty-fourth",
        "twenty-fifth",
        "twenty-sixth",
        "twenty-seventh",
        "twenty-eighth",
        "twenty-ninth",
        "thirtieth",
        "thirty-first",
    }
)


class SecFilingGemmaContractError(ValueError):
    """Raised when an experiment artifact violates the frozen contract."""


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SecFilingGemmaContractError(f"{location} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise SecFilingGemmaContractError(f"{location} keys must be strings")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SecFilingGemmaContractError(
            f"Invalid {location} keys; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _strict_bool(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise SecFilingGemmaContractError(f"{location} must be a boolean")
    return value


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SecFilingGemmaContractError(
            f"{location} must be an integer >= {minimum}"
        )
    return value


def _finite_number(value: Any, location: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SecFilingGemmaContractError(f"{location} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise SecFilingGemmaContractError(f"{location} is outside the permitted range")
    return number


def _iso_date(value: Any, location: str) -> date:
    if not isinstance(value, str):
        raise SecFilingGemmaContractError(f"{location} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecFilingGemmaContractError(f"{location} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise SecFilingGemmaContractError(
            f"{location} must use canonical YYYY-MM-DD form"
        )
    return parsed


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaContractError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _commit(value: Any, location: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise SecFilingGemmaContractError(
            f"{location} must be a full lowercase Git object id"
        )
    return value


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value with one deterministic, finite encoding."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaContractError(
            "Contract values must be finite JSON values"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_text(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaContractError("Model payload must be finite JSON") from exc


def build_extractor_json_schema() -> dict[str, Any]:
    dimension_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["current_impact", "change_vs_prior", "evidence_sentence_ids"],
        "properties": {
            "current_impact": {"type": "string", "enum": sorted(CURRENT_IMPACTS)},
            "change_vs_prior": {"type": "string", "enum": sorted(COMPARATIVE_CHANGES)},
            "evidence_sentence_ids": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[CP][0-9]{4}$"},
                "uniqueItems": True,
            },
        },
    }
    flag_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["present", "evidence_sentence_ids"],
        "properties": {
            "present": {"type": "boolean"},
            "evidence_sentence_ids": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[CP][0-9]{4}$"},
                "uniqueItems": True,
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "document_quality", "dimensions", "flags"],
        "properties": {
            "schema_version": {"const": EXTRACTOR_SCHEMA_VERSION},
            "document_quality": {
                "type": "string",
                "enum": sorted(DOCUMENT_QUALITIES),
            },
            "dimensions": {
                "type": "object",
                "additionalProperties": False,
                "required": list(DIMENSION_NAMES),
                "properties": {
                    name: copy.deepcopy(dimension_schema) for name in DIMENSION_NAMES
                },
            },
            "flags": {
                "type": "object",
                "additionalProperties": False,
                "required": list(FLAG_NAMES),
                "properties": {name: copy.deepcopy(flag_schema) for name in FLAG_NAMES},
            },
        },
    }


def build_extractor_model_payload(sentences: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Build the only object that the dedicated Ollama client may serialize."""

    canonical_sentences = [dict(sentence) for sentence in sentences]
    user_content = _canonical_json_text({"sentences": canonical_sentences})
    return {
        "model": "gemma4:12b",
        "messages": [
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "format": build_extractor_json_schema(),
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_ctx": 6_144,
            "num_predict": 512,
        },
    }


def build_redacted_input_manifest(
    *,
    artifact_stage: str,
    accession_number: str,
    corpus_universe_sha256: str,
    model_payload_sha256: str,
    universe_manifest: Mapping[str, Any],
    stage_content_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if artifact_stage not in STAGE_ORDER:
        raise SecFilingGemmaContractError("Redacted-input artifact stage is invalid")
    universe_hash = _sha256(corpus_universe_sha256, "corpus_universe_sha256")
    if universe_manifest.get("universe_sha256") != universe_hash:
        raise SecFilingGemmaContractError("Redacted-input manifest universe is not bound")
    if (
        stage_content_manifest.get("artifact_stage") != artifact_stage
        or stage_content_manifest.get("corpus_universe_sha256") != universe_hash
    ):
        raise SecFilingGemmaContractError("Redacted inputs are not bound to stage content")
    content_hash = _sha256(
        stage_content_manifest.get("content_manifest_sha256"),
        "content_manifest_sha256",
    )
    validate_stage_content_manifest(
        stage_content_manifest,
        universe_manifest=universe_manifest,
        expected_content_manifest_sha256=content_hash,
    )
    stage_accessions = {
        document["accession_number"]
        for document in stage_content_manifest["documents"]
    }
    if accession_number not in stage_accessions:
        raise SecFilingGemmaContractError(
            "Redacted-input event accession is outside its authorized stage"
        )
    payload_hash = _sha256(model_payload_sha256, "model_payload_sha256")
    body = {
        "schema_version": "aapl-sec-gemma-redacted-input-event-v2",
        "artifact_stage": artifact_stage,
        "accession_number": accession_number,
        "corpus_universe_sha256": universe_hash,
        "content_manifest_sha256": content_hash,
        "model_payload_sha256": payload_hash,
        "isolation_scope": "one_current_filing_and_its_immediate_prior_only",
    }
    return {**body, "redacted_input_manifest_sha256": canonical_sha256(body)}


def validate_redacted_input_manifest(
    manifest: Mapping[str, Any],
    *,
    universe_manifest: Mapping[str, Any],
    stage_content_manifest: Mapping[str, Any],
    expected_manifest_sha256: str,
) -> str:
    value = _expect_mapping(manifest, "redacted-input manifest")
    _expect_keys(
        value,
        {
            "schema_version",
            "artifact_stage",
            "accession_number",
            "corpus_universe_sha256",
            "content_manifest_sha256",
            "model_payload_sha256",
            "isolation_scope",
            "redacted_input_manifest_sha256",
        },
        "redacted-input manifest",
    )
    rebuilt = build_redacted_input_manifest(
        artifact_stage=value["artifact_stage"],
        accession_number=value["accession_number"],
        corpus_universe_sha256=value["corpus_universe_sha256"],
        model_payload_sha256=value["model_payload_sha256"],
        universe_manifest=universe_manifest,
        stage_content_manifest=stage_content_manifest,
    )
    if value != rebuilt:
        raise SecFilingGemmaContractError("Redacted-input manifest is not canonical")
    observed = _sha256(
        value["redacted_input_manifest_sha256"],
        "redacted_input_manifest_sha256",
    )
    if not hmac.compare_digest(
        observed, _sha256(expected_manifest_sha256, "expected_manifest_sha256")
    ):
        raise SecFilingGemmaContractError("Redacted-input manifest is not externally pinned")
    return observed


def build_contract_manifest() -> dict[str, Any]:
    """Return a fresh copy of the immutable experiment definition."""

    manifest: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "objective": {
            "symbol": "AAPL",
            "benchmark": "same_ledger_aapl_buy_and_hold",
            "initial_capital_usd": 1_000,
            "positions": ["LONG", "CASH"],
            "allowed_target_exposures": [0, 1],
            "leverage_allowed": False,
            "shorting_allowed": False,
            "negative_cash_allowed": False,
            "paid_apis_allowed": False,
        },
        "chronology": {
            "development": {
                "first_availability_session": "2000-01-01",
                "last_availability_session": "2018-12-31",
                "maximum_label_maturity_session": "2018-12-31",
                "purpose": "design_and_frozen_grid_selection",
            },
            "intermediate": {
                "first_availability_session": "2019-01-01",
                "last_availability_session": "2023-12-31",
                "maximum_refit_label_maturity_session": "2023-12-31",
                "reveal_count": 1,
                "purpose": "single_confirmation_then_predeclared_refit",
            },
            "final": {
                "first_availability_session": "2024-01-01",
                "last_availability_session": "2026-07-09",
                "periods_revealed_together": ["2024", "2025", "2026_ytd"],
                "training_permitted": False,
                "partial_period_reveal_permitted": False,
            },
            "availability_rule": (
                "first complete AAPL session strictly after the latest defensible "
                "SEC acceptance, filing, or filing-date-change date"
            ),
            "decision_time": "after_completed_availability_session_close",
            "execution_time": "next_adjusted_open",
            "calendar_required_through": "2026-07-10",
            "calendar_id": AUTHORITATIVE_CALENDAR_ID,
            "calendar_semantics": "exact_nyse_trading_session_dates_not_hours",
            "calendar_source_evidence_required": True,
            "calendar_source_semantic_reconciliation_required": True,
            "calendar_authoritative_source_urls": copy.deepcopy(
                CALENDAR_SOURCE_URLS
            ),
        },
        "corpus": {
            "subject_cik": "0000320193",
            "forms": ["10-K", "10-Q"],
            "amendments_included": False,
            "selection": "complete_metadata_eligible_universe_no_sampling",
            "periodic_document": "submissions_primary_document_only",
            "split_storage": "physically_separate_checksum_bound_artifacts",
            "minimum_development_filings": 72,
            "minimum_intermediate_filings": 19,
            "completed_year_minimum": {
                "total": 3,
                "10-K": 1,
                "10-Q": 2,
            },
            "model_call_caps": dict(STAGE_MODEL_CALL_CAPS),
            "audit_is_prerequisite_not_corpus": True,
            "minimum_exact_acceptance_timestamp_rate": 0.95,
            "minimum_usable_document_rate_overall": 0.95,
            "minimum_usable_document_rate_each_stage": 0.90,
            "availability_must_be_recomputed_from_pinned_calendar": True,
            "authoritative_catalog_and_universe_hash_required": True,
        },
        "extractor": {
            "schema_version": EXTRACTOR_SCHEMA_VERSION,
            "request_version": EXTRACTOR_REQUEST_VERSION,
            "preprocessor_version": PREPROCESSOR_VERSION,
            "one_call_per_filing": True,
            "prior_same_form_in_same_prompt": True,
            "maximum_utf8_bytes": MAX_INPUT_BYTES,
            "maximum_sentences": MAX_SENTENCES,
            "maximum_sentence_characters": MAX_SENTENCE_CHARACTERS,
            "dimensions": list(DIMENSION_NAMES),
            "flags": list(FLAG_NAMES),
            "free_text_allowed": False,
            "prices_returns_outcomes_or_actions_allowed": False,
            "invalid_output_policy": "unavailable_no_retry_no_repair_no_imputation",
            "evidence_required": True,
            "metadata_envelope_sent_to_model": False,
            "model_payload": "exact_system_user_schema_and_generation_payload_only",
            "per_event_stage_bound_redacted_input_manifest_required": True,
            "preprocessor_process_scope": (
                "fresh_worker_receives_only_current_and_immediate_prior_text"
            ),
        },
        "model": {
            "role": "grounded_structured_text_extractor_only",
            "name": "gemma4:12b",
            "endpoint": "http://127.0.0.1:11434/api/chat",
            "transport": {
                "trust_env": False,
                "proxies": False,
                "follow_redirects": False,
                "pull": False,
                "stream": False,
                "think": False,
                "temperature": 0,
                "seed": 0,
                "num_ctx": 6_144,
                "num_predict": 512,
                "connect_timeout_seconds": 2,
                "read_timeout_seconds": 30,
                "retries": 0,
                "repair_attempts": 0,
            },
            "parametric_contamination_risk": "unresolved_but_bounded",
            "direct_trading_authority": False,
            "runtime_template_system_parameters_details_and_version_hash_required": True,
            "per_call_client_receipt_trust": (
                "unattested_until_stage_runner_replay"
            ),
            "runtime_identity_guard": (
                "candidate_pins_checked_immediately_before_and_after_complete_"
                "stage_extraction_batch"
            ),
        },
        "predictor": {
            "decision_owner": "deterministic_regularized_two_head_model_v1",
            "target": "cash_vs_aapl_active_log_edge_after_10bps",
            "target_horizon_sessions": HORIZON_SESSIONS,
            "cash_episode_sessions": HORIZON_SESSIONS,
            "signal_during_active_episode": "ignored_no_extension",
            "market_context": ["SPY", "QQQ", "IWM", "VIX", "TNX"],
            "market_history_start": MARKET_HISTORY_START,
            "market_history_calendar_id": AUTHORITATIVE_MARKET_CALENDAR_ID,
            "maximum_market_lookback_sessions": MARKET_LOOKBACK_SESSIONS,
            "market_context_timing": "completed_decision_session_close_only",
            "prediction_timeline": {
                "decision_session": "filing_availability_session",
                "market_feature_cutoff": "decision_session_adjusted_close",
                "fill_session_offset": 1,
                "cash_exit_and_label_maturity_session_offset": LABEL_MATURITY_OFFSET,
                "label_eligible_for_prediction": (
                    "only_if_maturity_session_strictly_precedes_decision_session"
                ),
                "missing_prehistory": "prediction_unavailable_no_backfill",
            },
            "price_regime_features": [
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
            ],
            "market_sentiment_features": [
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
            ],
            "filing_calendar_features": [
                "form_is_10k",
                "sessions_since_prior_same_form",
                "semantic_output_unavailable",
            ],
            "feature_semantics": {
                "sessions_since_prior_same_form": (
                    "difference_between_zero_based_indices_in_the_exact_2000_onward_"
                    "authoritative_session_calendar_first_same_form_is_zero"
                ),
                "group_mean_denominator": "fixed_declared_group_size_not_stated_is_zero",
                "adverse_flag_names": list(ADVERSE_FLAG_NAMES),
                "adverse_flag_count_excludes_management_transition": True,
                "management_transition_is_separate_feature": True,
                "stated_dimension_fraction": (
                    "current_impact_not_not_stated_count_divided_by_10"
                ),
                "comparable_dimension_fraction": (
                    "change_not_not_stated_or_not_comparable_count_divided_by_10"
                ),
                "document_usable": "one_only_for_validated_quality_usable",
                "document_thin": "one_only_for_validated_quality_thin",
                "valid_unusable_output": (
                    "neutral_semantics_both_document_quality_indicators_zero"
                ),
                "invalid_model_output": (
                    "neutral_semantics_semantic_output_unavailable_one_no_retry"
                ),
                "missing_or_unauthenticated_extraction_evidence": (
                    "prediction_unavailable_integrity_failure"
                ),
                "ablation_missingness": (
                    "identical_semantic_output_unavailable_indicator_semantic_"
                    "content_features_zero"
                ),
                "required_market_support": (
                    "all_six_symbols_complete_through_decision_close_with_exact_"
                    "253_session_materialized_slice"
                ),
                "label_window": "entry_t_plus_1_adjusted_open_exit_t_plus_21_adjusted_open",
                "label_cost_application": (
                    "declared_cost_rate_on_each_of_entry_and_exit_position_changing_fills"
                ),
                "causal_event_identities": {
                    "market_prefix_chain_identity_sha256": (
                        "row_chain_genesis_count_and_tip_through_decision_session_only"
                    ),
                    "market_feature_row_sha256": (
                        "event_session_causal_market_prefix_identity_complete_support_"
                        "missingness_and_exact_market_feature_values_only"
                    ),
                    "extraction_identity_sha256": (
                        "event_local_current_and_prior_filing_extraction_status_"
                        "authentication_evidence_output_and_sentence_identities_only"
                    ),
                    "full_stage_provenance_hashes_are_separate": True,
                },
            },
            "semantic_encoding": {
                "current_impact": {
                    "favorable": 1,
                    "neutral": 0,
                    "unfavorable": -1,
                    "mixed": 0,
                    "not_stated": 0,
                },
                "change_vs_prior": {
                    "improving": 1,
                    "stable": 0,
                    "deteriorating": -1,
                    "mixed": 0,
                    "not_comparable": 0,
                    "not_stated": 0,
                },
                "dimension_groups": {
                    "commercial": ["demand", "pricing_power", "gross_margin"],
                    "financial": [
                        "operating_cost_pressure",
                        "capital_allocation",
                        "liquidity",
                    ],
                    "risk_and_outlook": [
                        "forward_guidance",
                        "supply_chain",
                        "legal_regulatory",
                        "management_uncertainty",
                    ],
                },
                "downstream_features": [
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
                ],
                "flag_true": 1,
                "flag_false": 0,
            },
            "learner_configuration": {
                "feature_selection": False,
                "interactions": False,
                "scaling": "training_median_mad_clip_4_scale_floor_1e-6",
                "probability_head": "ridge_logistic_lambda_0.1_max_iter_50_tol_1e-10",
                "edge_head": "ridge_huber_lambda_0.1_delta_1.5_max_iter_50_tol_1e-10",
                "frozen_numerical_constants": {
                    "raw_mad_multiplier": 1.4826,
                    "raw_scale_floor": 1e-6,
                    "raw_z_clip": 4.0,
                    "ridge_lambda": 0.1,
                    "logistic_max_iterations": 50,
                    "logistic_tolerance": 1e-10,
                    "newton_line_search_max_steps": 50,
                    "newton_armijo_constant": 1e-4,
                    "logistic_curvature_floor": 1e-15,
                    "huber_delta": 1.5,
                    "huber_max_iterations": 50,
                    "huber_tolerance": 1e-10,
                    "target_mad_multiplier": 1.4826,
                    "target_scale_floor": 1e-6,
                    "edge_clip_lower": -0.5,
                    "edge_clip_upper": 0.5,
                },
                "intercept_regularized": False,
                "edge_target_preprocessing": (
                    "clip_to_closed_interval_minus_0.5_plus_0.5_then_training_"
                    "median_mad_scale_1.4826_floor_1e-6"
                ),
                "logistic_initialization": "intercept_logit_training_prevalence_others_zero",
                "logistic_solver": (
                    "newton_armijo_backtracking_stop_on_max_abs_gradient_or_"
                    "coefficient_change"
                ),
                "huber_initialization": "intercept_training_target_median_others_zero",
                "huber_solver": "irls_stop_on_max_abs_coefficient_change",
                "nonfinite_or_missing_required_market_feature": "prediction_unavailable",
                "fit_order": (
                    "fixed_once_per_development_fold_from_all_and_only_labels_whose_"
                    "maturity_session_strictly_precedes_the_fold_test_first_session"
                ),
                "development_fold_state": (
                    "semantic_ablation_scalers_heads_and_climatology_fit_once_at_fold_"
                    "start_and_held_byte_identical_through_the_complete_test_window"
                ),
                "post_selection_refit": (
                    "fit_once_from_all_and_only_development_labels_matured_by_2018_12_31"
                ),
            },
            "semantic_predictor": "gemma_dimensions_flags_plus_market_context",
            "ablation_predictor": "filing_calendar_plus_market_context_semantics_neutral",
            "candidate_grid": [
                {
                    "candidate_id": "p50_e0",
                    "probability_gate": 0.50,
                    "expected_edge_gate": 0.0,
                },
                {
                    "candidate_id": "p55_e0",
                    "probability_gate": 0.55,
                    "expected_edge_gate": 0.0,
                },
                {
                    "candidate_id": "p50_e25",
                    "probability_gate": 0.50,
                    "expected_edge_gate": 0.0025,
                },
                {
                    "candidate_id": "p55_e25",
                    "probability_gate": 0.55,
                    "expected_edge_gate": 0.0025,
                },
            ],
            "candidate_selection": (
                "pass_all_development_gates_then_lowest_10bps_brier_then_highest_"
                "10bps_active_edge_then_frozen_candidate_order"
            ),
            "candidate_gate_comparison": "probability_gte_and_expected_edge_gte",
            "binary_cash_win_target": {
                "cost_bps": BRIER_TARGET_COST_BPS,
                "positive_if": (
                    "cash_active_log_edge_strictly_above_comparison_tolerance"
                ),
                "comparison_tolerance": ACTIVE_EDGE_TOLERANCE,
                "used_by": [
                    "probability_head",
                    "brier_score",
                    "causal_climatology",
                    "episode_win_rate",
                    "live_lessons",
                ],
            },
            "development_folds": [
                {
                    "fold_id": fold_id,
                    "train_label_maturity_through": train_through,
                    "test_first_session": test_first,
                    "test_last_session": test_last,
                    "state_updates_inside_test_window": False,
                }
                for fold_id, train_through, test_first, test_last in DEVELOPMENT_FOLD_SPECS
            ],
        },
        "gates": {
            "development_both_5bps_and_10bps": {
                "minimum_total_active_log_edge": 0.02,
                "minimum_median_annual_active_log_edge": 0.0005,
                "minimum_edge_without_best_year": 0.005,
                "minimum_252_session_month_end_win_rate": 0.60,
                "minimum_756_session_month_end_win_rate": 0.70,
                "minimum_annual_win_rate": 0.55,
                "minimum_positive_folds": 4,
                "fold_count": 5,
                "minimum_cash_days": 30,
                "minimum_cash_episodes": 12,
                "maximum_cash_rate": 0.20,
                "maximum_largest_positive_year_share": 0.45,
                "minimum_negative_buy_hold_year_edge": 0.01,
                "minimum_negative_buy_hold_year_win_rate": 0.60,
                "minimum_relative_brier_improvement_vs_causal_climatology": 0.02,
                "minimum_10bps_edge_advantage_vs_ablation": 0.01,
                "minimum_relative_brier_improvement_vs_ablation": 0.01,
                "minimum_fold_edge_difference_vs_ablation": 0.0,
            },
            "intermediate_5bps": {
                "minimum_total_active_log_edge": 0.01,
                "minimum_winning_years": 3,
                "year_count": 5,
                "minimum_median_annual_edge": 0.0,
                "minimum_edge_without_best_year_exclusive": 0.0,
                "minimum_252_session_month_end_win_rate": 0.60,
                "minimum_cash_days": 10,
                "minimum_cash_episodes": 5,
                "maximum_cash_rate": 0.20,
                "maximum_largest_positive_year_share": 0.60,
                "minimum_negative_buy_hold_year_edge_exclusive": 0.0,
                "minimum_relative_brier_improvement_vs_climatology": 0.01,
                "minimum_edge_advantage_vs_ablation": 0.0025,
            },
            "intermediate_10bps": {
                "minimum_total_active_log_edge_exclusive": 0.0,
                "minimum_winning_years": 3,
                "minimum_edge_without_best_year": 0.0,
                "minimum_mean_episode_edge_exclusive": 0.0,
                "must_beat_ablation": True,
            },
            "final": {
                "periods": ["2024", "2025", "2026_ytd"],
                "minimum_each_period_5bps_active_log_edge": 0.004987541511039074,
                "minimum_continuous_5bps_active_log_edge": 0.01980262729617973,
                "minimum_each_period_10bps_active_log_edge_exclusive": 0.0,
                "minimum_continuous_10bps_active_log_edge_exclusive": 0.0,
                "terminal_conventions": ["adjusted_open", "terminal_adjusted_close"],
                "minimum_episode_count_each_period": 1,
                "minimum_total_episode_count": 6,
                "minimum_episode_win_rate": 0.55,
                "minimum_mean_episode_edge_10bps_exclusive": 0.0,
                "maximum_single_episode_positive_edge_share": 0.50,
                "maximum_drawdown_disadvantage_percentage_points": 1.0,
                "minimum_continuous_edge_advantage_vs_ablation_exclusive": 0.0,
                "minimum_periods_beating_ablation": 2,
            },
        },
        "execution": {
            "decision_after_close": True,
            "fill": "next_adjusted_open",
            "terminal_valuation": ["adjusted_open", "terminal_adjusted_close"],
            "transaction_costs_bps": [5, 10],
            "cost_application": (
                "slippage_on_each_position-changing_fill_zero_commission_zero_margin"
            ),
            "same_sessions_and_prices_as_benchmark": True,
            "binary_target_only": True,
            "require_existing_unleveraged_ledger_proof": True,
            "last_cutoff_decision_without_in_period_fill": "recorded_not_scored",
            "open_episode_at_cutoff": "terminal_valued_and_never_used_as_training",
            "period_attribution": "sum_daily_active_log_increments_on_one_continuous_ledger",
            "brier_rows": "only_predictions_with_full_horizon_matured_by_score_cutoff",
            "comparison_tolerance": ACTIVE_EDGE_TOLERANCE,
        },
        "scoring_semantics": {
            "prediction_invariance": (
                "future_only_stage_extensions_cannot_change_earlier_causal_evidence_"
                "identities_feature_values_probabilities_edges_or_actions"
            ),
            "brier_target": (
                "cash_beats_aapl_after_10bps_strictly_above_1e_12"
            ),
            "causal_climatology": (
                "beta_1_1_smoothed_cash_win_rate_from_the_exact_frozen_fold_training_set"
            ),
            "fold_prediction_state": (
                "one_state_per_declared_fold_fit_before_test_and_immutable_inside_test"
            ),
            "same_fold_test_outcomes_can_update_state": False,
            "semantic_ablation_and_climatology_training_support_identical": True,
            "fold_score_rows": "only_labels_matured_by_the_phase_cutoff",
            "annual_win": "active_log_edge_strictly_above_1e-12",
            "rolling_win": "month_end_sum_strictly_above_1e-12",
            "episode_positive_share_denominator": "sum_of_strictly_positive_episode_edges",
            "zero_positive_episode_denominator": "gate_fails",
            "drawdown": "maximum_peak_to_trough_decline_in_total_wealth",
            "cross_year_episode": "state_continues_and_daily_edge_is_attributed_by_session_year",
            "intermediate_order": (
                "sequential_prefix_only_predictions_from_through_2018_state_then_score_once_"
                "then_refit_through_2023"
            ),
            "final_order": (
                "sequential_prefix_only_prediction_and_action_sealed_before_its_own_future_"
                "return_frozen_state_labels_scores_and_gates_quarantined_joint_report_after_cutoff"
            ),
        },
        "holdout_governance": {
            "evidence_class": "approach_specific_reused_historical_holdout",
            "repository_wide_reveal_registry_required": True,
            "one_registered_attempt_id_per_candidate": True,
            "historical_final_reveal_count_lower_bound_required": True,
            "known_hash_identities_plus_unattributed_count_not_exhaustive": True,
            "candidate_binds_predecessor_registry_snapshot": True,
            "appended_entry_binds_candidate_then_new_tip_and_count_are_externally_pinned": True,
            "winner_selection_across_final_attempts_forbidden": True,
            "globally_pristine_claim_allowed": False,
            "prospective_paper_trading_required_for_pristine_evidence": True,
            "required_semantic_prerequisite_checks": list(
                REQUIRED_STAGE_VERIFIER_CHECKS
            ),
        },
        "runtime": {
            "hard_total_seconds": MAX_RUNTIME_SECONDS,
            "sec_acquisition_seconds": MAX_SEC_SECONDS,
            "gemma_extraction_seconds": MAX_MODEL_SECONDS,
            "fit_simulation_sealing_seconds": MAX_FIT_SECONDS,
            "sec_request_cap": MAX_SEC_REQUESTS,
            "sec_byte_cap": MAX_SEC_BYTES,
            "sec_rate_limit_requests_per_second": 2,
            "partial_run_can_pass": False,
            "stage_specific_runtime_receipts_required": True,
            "final_all_stage_summary_is_not_stage_authorization": True,
            "development_latency_preflight": {
                "filings": 5,
                "maximum_length_only": True,
                "project_complete_run_before_later_stage_access": True,
            },
        },
        "live_continuation": {
            "separate_from_primary_frozen_final": True,
            "lesson_horizon_sessions": HORIZON_SESSIONS,
            "lesson_must_mature_before_ingestion": True,
            "lesson_first_use": "strictly_after_ingestion_session",
            "append_only_chronological_lessons": True,
            "cumulative_prediction_binding_ledger_required": True,
            "cross_batch_lesson_accession_or_prediction_replay_forbidden": True,
            "refit_timing": (
                "before_each_new_filing_decision_using_all_and_only_matured_lessons"
            ),
            "gemma_prompt_schema_weights_and_action_thresholds_immutable": True,
            "downstream_state_transition_hash_chain_required": True,
            "calendar_updates": "append_only_extension_preserving_pinned_historical_prefix",
        },
    }
    return copy.deepcopy(manifest)


def validate_contract_manifest(manifest: Mapping[str, Any]) -> str:
    observed = _expect_mapping(manifest, "contract manifest")
    expected = build_contract_manifest()
    if canonical_sha256(observed) != canonical_sha256(expected) or observed != expected:
        raise SecFilingGemmaContractError(
            "Contract manifest does not match the frozen experiment definition"
        )
    return canonical_sha256(expected)


def _candidate_body(
    *,
    model_digest: str,
    ollama_runtime_fingerprint_sha256: str,
    sec_audit_checksums_json_sha256: str,
    sec_catalog_artifact_sha256: str,
    sec_audit_source_commit: str,
    calendar_source_evidence_sha256: str,
    calendar_sessions_sha256: str,
    corpus_universe_sha256: str,
    corpus_universe_semantic_sha256: str,
    identity_lexicon_sha256: str,
    predecessor_reveal_registry_sha256: str,
    holdout_attempt_id: str,
    experiment_source_commit: str,
    source_tree_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    digest = _sha256(model_digest, "model_digest")
    runtime_fingerprint = _sha256(
        ollama_runtime_fingerprint_sha256,
        "ollama_runtime_fingerprint_sha256",
    )
    audit_hash = _sha256(
        sec_audit_checksums_json_sha256,
        "sec_audit_checksums_json_sha256",
    )
    catalog_hash = _sha256(
        sec_catalog_artifact_sha256,
        "sec_catalog_artifact_sha256",
    )
    source_commit = _commit(sec_audit_source_commit, "sec_audit_source_commit")
    calendar_evidence_hash = _sha256(
        calendar_source_evidence_sha256,
        "calendar_source_evidence_sha256",
    )
    calendar_sessions_hash = _sha256(
        calendar_sessions_sha256,
        "calendar_sessions_sha256",
    )
    if calendar_sessions_hash != canonical_sha256(
        list(AUTHORITATIVE_SESSION_DATES)
    ):
        raise SecFilingGemmaContractError(
            "Candidate calendar sessions are not the frozen authoritative sequence"
        )
    universe_hash = _sha256(corpus_universe_sha256, "corpus_universe_sha256")
    universe_semantic_hash = _sha256(
        corpus_universe_semantic_sha256,
        "corpus_universe_semantic_sha256",
    )
    lexicon_hash = _sha256(identity_lexicon_sha256, "identity_lexicon_sha256")
    predecessor_registry_hash = _sha256(
        predecessor_reveal_registry_sha256,
        "predecessor_reveal_registry_sha256",
    )
    if not isinstance(holdout_attempt_id, str) or _ATTEMPT_RE.fullmatch(holdout_attempt_id) is None:
        raise SecFilingGemmaContractError("holdout_attempt_id is not canonical")
    experiment_commit = _commit(experiment_source_commit, "experiment_source_commit")
    tree_hash = _sha256(source_tree_sha256, "source_tree_sha256")
    sources = _expect_mapping(source_hashes, "source_hashes")
    _expect_keys(sources, set(REQUIRED_SOURCE_HASHES), "source_hashes")
    normalized_sources = {
        key: _sha256(sources[key], f"source_hashes.{key}")
        for key in REQUIRED_SOURCE_HASHES
    }
    contract = build_contract_manifest()
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": canonical_sha256(contract),
        "model": {
            "name": contract["model"]["name"],
            "digest": digest,
            "runtime_fingerprint_sha256": runtime_fingerprint,
            "endpoint": contract["model"]["endpoint"],
            "transport": copy.deepcopy(contract["model"]["transport"]),
        },
        "bindings": {
            "sec_audit_checksums_json_sha256": audit_hash,
            "sec_catalog_artifact_sha256": catalog_hash,
            "sec_audit_source_commit": source_commit,
            "calendar_source_evidence_sha256": calendar_evidence_hash,
            "calendar_sessions_sha256": calendar_sessions_hash,
            "corpus_universe_sha256": universe_hash,
            "corpus_universe_semantic_sha256": universe_semantic_hash,
            "identity_lexicon_sha256": lexicon_hash,
            "predecessor_reveal_registry_sha256": predecessor_registry_hash,
            "holdout_attempt_id": holdout_attempt_id,
            "experiment_source_commit": experiment_commit,
            "source_tree_sha256": tree_hash,
            "source_hashes": normalized_sources,
        },
    }


def build_candidate_manifest(
    *,
    model_digest: str,
    ollama_runtime_fingerprint_sha256: str,
    sec_audit_checksums_json_sha256: str,
    sec_catalog_artifact_sha256: str,
    sec_audit_source_commit: str,
    calendar_source_evidence_sha256: str,
    calendar_sessions_sha256: str,
    corpus_universe_sha256: str,
    corpus_universe_semantic_sha256: str,
    identity_lexicon_sha256: str,
    predecessor_reveal_registry_sha256: str,
    holdout_attempt_id: str,
    experiment_source_commit: str,
    source_tree_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Bind the frozen design to exact audit, model, calendar, and code bytes."""

    body = _candidate_body(
        model_digest=model_digest,
        ollama_runtime_fingerprint_sha256=ollama_runtime_fingerprint_sha256,
        sec_audit_checksums_json_sha256=sec_audit_checksums_json_sha256,
        sec_catalog_artifact_sha256=sec_catalog_artifact_sha256,
        sec_audit_source_commit=sec_audit_source_commit,
        calendar_source_evidence_sha256=calendar_source_evidence_sha256,
        calendar_sessions_sha256=calendar_sessions_sha256,
        corpus_universe_sha256=corpus_universe_sha256,
        corpus_universe_semantic_sha256=corpus_universe_semantic_sha256,
        identity_lexicon_sha256=identity_lexicon_sha256,
        predecessor_reveal_registry_sha256=predecessor_reveal_registry_sha256,
        holdout_attempt_id=holdout_attempt_id,
        experiment_source_commit=experiment_source_commit,
        source_tree_sha256=source_tree_sha256,
        source_hashes=source_hashes,
    )
    return {**body, "candidate_sha256": canonical_sha256(body)}


def validate_candidate_manifest(
    manifest: Mapping[str, Any], *, expected_candidate_sha256: str | None = None
) -> str:
    value = _expect_mapping(manifest, "candidate manifest")
    _expect_keys(
        value,
        {
            "schema_version",
            "contract_version",
            "contract_sha256",
            "model",
            "bindings",
            "candidate_sha256",
        },
        "candidate manifest",
    )
    model = _expect_mapping(value["model"], "candidate model")
    bindings = _expect_mapping(value["bindings"], "candidate bindings")
    _expect_keys(
        model,
        {"name", "digest", "runtime_fingerprint_sha256", "endpoint", "transport"},
        "candidate model",
    )
    _expect_keys(
        bindings,
        {
            "sec_audit_checksums_json_sha256",
            "sec_catalog_artifact_sha256",
            "sec_audit_source_commit",
            "calendar_source_evidence_sha256",
            "calendar_sessions_sha256",
            "corpus_universe_sha256",
            "corpus_universe_semantic_sha256",
            "identity_lexicon_sha256",
            "predecessor_reveal_registry_sha256",
            "holdout_attempt_id",
            "experiment_source_commit",
            "source_tree_sha256",
            "source_hashes",
        },
        "candidate bindings",
    )
    rebuilt = build_candidate_manifest(
        model_digest=model["digest"],
        ollama_runtime_fingerprint_sha256=model["runtime_fingerprint_sha256"],
        sec_audit_checksums_json_sha256=bindings[
            "sec_audit_checksums_json_sha256"
        ],
        sec_catalog_artifact_sha256=bindings["sec_catalog_artifact_sha256"],
        sec_audit_source_commit=bindings["sec_audit_source_commit"],
        calendar_source_evidence_sha256=bindings[
            "calendar_source_evidence_sha256"
        ],
        calendar_sessions_sha256=bindings["calendar_sessions_sha256"],
        corpus_universe_sha256=bindings["corpus_universe_sha256"],
        corpus_universe_semantic_sha256=bindings[
            "corpus_universe_semantic_sha256"
        ],
        identity_lexicon_sha256=bindings["identity_lexicon_sha256"],
        predecessor_reveal_registry_sha256=bindings[
            "predecessor_reveal_registry_sha256"
        ],
        holdout_attempt_id=bindings["holdout_attempt_id"],
        experiment_source_commit=bindings["experiment_source_commit"],
        source_tree_sha256=bindings["source_tree_sha256"],
        source_hashes=_expect_mapping(bindings["source_hashes"], "source_hashes"),
    )
    if value != rebuilt:
        raise SecFilingGemmaContractError(
            "Candidate manifest differs from its canonical frozen construction"
        )
    candidate_hash = _sha256(value["candidate_sha256"], "candidate_sha256")
    if expected_candidate_sha256 is not None:
        expected_hash = _sha256(
            expected_candidate_sha256, "expected_candidate_sha256"
        )
        if not hmac.compare_digest(candidate_hash, expected_hash):
            raise SecFilingGemmaContractError(
                "Candidate manifest does not match the externally pinned hash"
            )
    return candidate_hash


def _validate_candidate_universe_provenance(
    candidate_manifest: Mapping[str, Any],
    universe_manifest: Mapping[str, Any],
) -> None:
    """Require the candidate's SEC catalog and calendar to be the universe's."""

    bindings = _expect_mapping(candidate_manifest["bindings"], "candidate bindings")
    universe = _expect_mapping(universe_manifest, "corpus universe manifest")
    if universe.get("calendar_artifact_sha256") != bindings[
        "calendar_source_evidence_sha256"
    ]:
        raise SecFilingGemmaContractError(
            "Corpus universe calendar is not the candidate-bound calendar artifact"
        )
    if universe.get("calendar_sessions_sha256") != bindings[
        "calendar_sessions_sha256"
    ]:
        raise SecFilingGemmaContractError(
            "Corpus universe sessions are not the candidate-bound calendar sequence"
        )
    if universe.get("catalog_artifact_sha256") != bindings[
        "sec_catalog_artifact_sha256"
    ]:
        raise SecFilingGemmaContractError(
            "Corpus universe catalog is not the candidate-bound SEC catalog artifact"
        )
    if universe.get("universe_semantic_sha256") != bindings[
        "corpus_universe_semantic_sha256"
    ]:
        raise SecFilingGemmaContractError(
            "Corpus universe semantics are not the candidate-bound design universe"
        )


def _stage_receipt_body(
    *,
    stage: str,
    candidate_sha256: str,
    corpus_manifest_sha256: str,
    extraction_artifact_sha256: str,
    input_model_state_sha256: str | None,
    output_model_state_sha256: str,
    selected_candidate_id: str,
    passed: bool,
    parent_receipt_sha256: str | None,
    design_change_count: int,
    partial_results_exposed: bool,
) -> dict[str, Any]:
    if stage not in STAGE_ORDER:
        raise SecFilingGemmaContractError(f"Unknown stage: {stage!r}")
    candidate_hash = _sha256(candidate_sha256, "candidate_sha256")
    corpus_hash = _sha256(corpus_manifest_sha256, "corpus_manifest_sha256")
    extraction_hash = _sha256(
        extraction_artifact_sha256, "extraction_artifact_sha256"
    )
    output_state = _sha256(output_model_state_sha256, "output_model_state_sha256")
    if input_model_state_sha256 is None:
        input_state = None
    else:
        input_state = _sha256(
            input_model_state_sha256, "input_model_state_sha256"
        )
    if selected_candidate_id not in CANDIDATE_IDS:
        raise SecFilingGemmaContractError("Unknown frozen candidate id")
    passed_value = _strict_bool(passed, "passed")
    changes = _strict_int(design_change_count, "design_change_count")
    partial = _strict_bool(partial_results_exposed, "partial_results_exposed")
    if changes != 0:
        raise SecFilingGemmaContractError("Design changes after freezing are forbidden")
    if stage == "development":
        if parent_receipt_sha256 is not None or input_state is not None:
            raise SecFilingGemmaContractError(
                "Development cannot have a parent receipt or input model state"
            )
        parent_hash = None
        refit_policy = "select_from_frozen_grid_through_2018"
    else:
        if parent_receipt_sha256 is None or input_state is None:
            raise SecFilingGemmaContractError(
                f"{stage} requires its parent receipt and input model state"
            )
        parent_hash = _sha256(parent_receipt_sha256, "parent_receipt_sha256")
        refit_policy = (
            "single_predeclared_refit_through_2023"
            if stage == "intermediate"
            else "no_refit_or_online_update"
        )
    if stage == "final":
        if input_state != output_state:
            raise SecFilingGemmaContractError(
                "The primary final test cannot update its model state"
            )
        if partial:
            raise SecFilingGemmaContractError(
                "Final-year partial results cannot be exposed"
            )
    if stage == "intermediate" and partial:
        raise SecFilingGemmaContractError(
            "The one-shot intermediate result cannot be exposed partially"
        )
    return {
        "schema_version": STAGE_RECEIPT_SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "stage": stage,
        "candidate_sha256": candidate_hash,
        "corpus_manifest_sha256": corpus_hash,
        "extraction_artifact_sha256": extraction_hash,
        "input_model_state_sha256": input_state,
        "output_model_state_sha256": output_state,
        "selected_candidate_id": selected_candidate_id,
        "passed": passed_value,
        "parent_receipt_sha256": parent_hash,
        "design_change_count": changes,
        "partial_results_exposed": partial,
        "refit_policy": refit_policy,
    }


def _build_structural_stage_receipt(**kwargs: Any) -> dict[str, Any]:
    """Build an internal shape fixture, never an access authorization."""

    body = _stage_receipt_body(**kwargs)
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _validate_structural_stage_receipt(receipt: Mapping[str, Any]) -> str:
    value = _expect_mapping(receipt, "stage receipt")
    expected_keys = {
        "schema_version",
        "contract_sha256",
        "stage",
        "candidate_sha256",
        "corpus_manifest_sha256",
        "extraction_artifact_sha256",
        "input_model_state_sha256",
        "output_model_state_sha256",
        "selected_candidate_id",
        "passed",
        "parent_receipt_sha256",
        "design_change_count",
        "partial_results_exposed",
        "refit_policy",
        "receipt_sha256",
    }
    _expect_keys(value, expected_keys, "stage receipt")
    rebuilt = _build_structural_stage_receipt(
        stage=value["stage"],
        candidate_sha256=value["candidate_sha256"],
        corpus_manifest_sha256=value["corpus_manifest_sha256"],
        extraction_artifact_sha256=value["extraction_artifact_sha256"],
        input_model_state_sha256=value["input_model_state_sha256"],
        output_model_state_sha256=value["output_model_state_sha256"],
        selected_candidate_id=value["selected_candidate_id"],
        passed=value["passed"],
        parent_receipt_sha256=value["parent_receipt_sha256"],
        design_change_count=value["design_change_count"],
        partial_results_exposed=value["partial_results_exposed"],
    )
    if value != rebuilt:
        raise SecFilingGemmaContractError("Stage receipt is not canonical")
    return _sha256(value["receipt_sha256"], "receipt_sha256")


def _validate_structural_stage_transition(
    parent: Mapping[str, Any], child: Mapping[str, Any]
) -> None:
    parent_hash = _validate_structural_stage_receipt(parent)
    _validate_structural_stage_receipt(child)
    parent_stage = parent["stage"]
    child_stage = child["stage"]
    if STAGE_ORDER.index(child_stage) != STAGE_ORDER.index(parent_stage) + 1:
        raise SecFilingGemmaContractError("Stages must advance exactly once in order")
    if parent["passed"] is not True:
        raise SecFilingGemmaContractError("A failed stage cannot authorize another reveal")
    if child["parent_receipt_sha256"] != parent_hash:
        raise SecFilingGemmaContractError("Child does not bind the exact parent receipt")
    if child["candidate_sha256"] != parent["candidate_sha256"]:
        raise SecFilingGemmaContractError("Candidate changed between stages")
    if child["selected_candidate_id"] != parent["selected_candidate_id"]:
        raise SecFilingGemmaContractError("Selected candidate changed between stages")
    if child["input_model_state_sha256"] != parent["output_model_state_sha256"]:
        raise SecFilingGemmaContractError("Child did not consume the exact parent state")
    if child["corpus_manifest_sha256"] == parent["corpus_manifest_sha256"]:
        raise SecFilingGemmaContractError("Stage corpus artifacts must be physically distinct")
    if child["extraction_artifact_sha256"] == parent["extraction_artifact_sha256"]:
        raise SecFilingGemmaContractError(
            "Stage extraction artifacts must be physically distinct"
        )


def authorize_stage_access(
    *,
    stage: str,
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
    prior_receipts: Sequence[Mapping[str, Any]],
) -> str:
    """Fail closed until authoritative prediction/ledger proof exists.

    A self-authored Boolean or checksum is not sufficient to unlock a split.
    The later effectful runner must verify a sealed prediction artifact created
    before outcome access, recompute every score/gate from bound ledgers, run
    the independent no-leverage proof, and bind runtime evidence.  Until that
    verifier exists, no stage transition is authorized.
    """

    del stage, candidate_manifest, expected_candidate_sha256, prior_receipts
    raise SecFilingGemmaContractError(
        "Stage access is disabled until the authoritative sealed-evidence "
        "verifier is implemented"
    )


def _canonical_session_sequence(session_dates: Sequence[str]) -> tuple[str, ...]:
    if isinstance(session_dates, (str, bytes)) or not isinstance(session_dates, Sequence):
        raise SecFilingGemmaContractError("Session calendar must be a sequence")
    parsed = [_iso_date(value, f"session_dates[{index}]") for index, value in enumerate(session_dates)]
    if not parsed or parsed != sorted(parsed) or len(parsed) != len(set(parsed)):
        raise SecFilingGemmaContractError("Session calendar must be nonempty, unique, and sorted")
    if any(value.weekday() >= 5 for value in parsed):
        raise SecFilingGemmaContractError("Session calendar cannot contain weekend dates")
    return tuple(value.isoformat() for value in parsed)


def canonical_session_calendar(session_dates: Sequence[str]) -> tuple[str, ...]:
    canonical = _canonical_session_sequence(session_dates)
    if canonical != AUTHORITATIVE_SESSION_DATES:
        raise SecFilingGemmaContractError(
            "Session calendar is not the exact frozen authoritative NYSE sequence"
        )
    return canonical


def session_calendar_sha256(session_dates: Sequence[str]) -> str:
    return canonical_sha256(list(canonical_session_calendar(session_dates)))


def canonical_market_session_calendar(
    session_dates: Sequence[str],
) -> tuple[str, ...]:
    canonical = _canonical_session_sequence(session_dates)
    if canonical != AUTHORITATIVE_MARKET_SESSION_DATES:
        raise SecFilingGemmaContractError(
            "Market-feature calendar is not the exact frozen authoritative NYSE sequence"
        )
    return canonical


def market_session_calendar_sha256(session_dates: Sequence[str]) -> str:
    return canonical_sha256(list(canonical_market_session_calendar(session_dates)))


def build_calendar_extension_manifest(
    *,
    prior_session_dates: Sequence[str],
    extended_session_dates: Sequence[str],
    expected_prior_sessions_sha256: str,
) -> dict[str, Any]:
    prior = _canonical_session_sequence(prior_session_dates)
    extended = canonical_session_calendar(extended_session_dates)
    if prior != LEGACY_AUTHORITATIVE_SESSION_DATES:
        raise SecFilingGemmaContractError(
            "Prior calendar is not the exact immutable legacy NYSE sequence"
        )
    prior_hash = canonical_sha256(list(prior))
    if not hmac.compare_digest(
        prior_hash,
        _sha256(expected_prior_sessions_sha256, "expected_prior_sessions_sha256"),
    ):
        raise SecFilingGemmaContractError("Prior calendar is not externally pinned")
    if len(extended) <= len(prior) or extended[: len(prior)] != prior:
        raise SecFilingGemmaContractError(
            "Calendar extension must preserve the complete historical prefix"
        )
    body = {
        "schema_version": "aapl-session-calendar-extension-v2",
        "prior_calendar_id": LEGACY_AUTHORITATIVE_CALENDAR_ID,
        "extended_calendar_id": AUTHORITATIVE_CALENDAR_ID,
        "hash_serialization": "canonical_json_utf8_sha256_bare_hex",
        "prior_sessions_sha256": prior_hash,
        "prior_session_count": len(prior),
        "prior_last_session": prior[-1],
        "extended_sessions_sha256": canonical_sha256(list(extended)),
        "extended_session_count": len(extended),
        "extended_last_session": extended[-1],
        "appended_sessions": list(extended[len(prior) :]),
    }
    return {**body, "extension_manifest_sha256": canonical_sha256(body)}


def validate_calendar_extension_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_prior_sessions_sha256: str,
    expected_extension_manifest_sha256: str,
) -> str:
    """Validate the one frozen v1-to-v2 authoritative calendar extension."""

    observed = _expect_mapping(manifest, "calendar extension manifest")
    expected = build_calendar_extension_manifest(
        prior_session_dates=LEGACY_AUTHORITATIVE_SESSION_DATES,
        extended_session_dates=AUTHORITATIVE_SESSION_DATES,
        expected_prior_sessions_sha256=expected_prior_sessions_sha256,
    )
    _expect_keys(observed, set(expected), "calendar extension manifest")
    expected_hash = _sha256(
        expected_extension_manifest_sha256,
        "expected_extension_manifest_sha256",
    )
    observed_hash = _sha256(
        observed["extension_manifest_sha256"],
        "calendar extension manifest extension_manifest_sha256",
    )
    if (
        observed != expected
        or canonical_sha256(
            {
                key: observed[key]
                for key in observed
                if key != "extension_manifest_sha256"
            }
        )
        != expected["extension_manifest_sha256"]
        or not hmac.compare_digest(
            observed_hash, expected_hash
        )
    ):
        raise SecFilingGemmaContractError(
            "Calendar extension is noncanonical or not externally pinned"
        )
    return expected["extension_manifest_sha256"]


def build_calendar_source_evidence_manifest(
    *,
    retrieved_at_utc: str,
    source_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind exact official NYSE source bytes to all frozen calendar sequences."""

    if not isinstance(retrieved_at_utc, str) or not retrieved_at_utc.endswith("Z"):
        raise SecFilingGemmaContractError(
            "Calendar source retrieval time must be canonical UTC"
        )
    try:
        parsed = datetime.fromisoformat(retrieved_at_utc[:-1] + "+00:00")
    except ValueError as exc:
        raise SecFilingGemmaContractError(
            "Calendar source retrieval time must be canonical UTC"
        ) from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != retrieved_at_utc:
        raise SecFilingGemmaContractError(
            "Calendar source retrieval time must have whole-second UTC form"
        )
    records = _expect_mapping(source_records, "calendar source records")
    _expect_keys(records, set(CALENDAR_SOURCE_URLS), "calendar source records")
    normalized_records: dict[str, dict[str, Any]] = {}
    for name, expected_url in CALENDAR_SOURCE_URLS.items():
        record = _expect_mapping(records[name], f"calendar source {name}")
        _expect_keys(record, {"url", "content_sha256", "byte_count"}, f"calendar source {name}")
        if record["url"] != expected_url:
            raise SecFilingGemmaContractError(
                "Calendar evidence did not use the frozen official source URL"
            )
        normalized_records[name] = {
            "url": expected_url,
            "content_sha256": _sha256(
                record["content_sha256"], f"calendar source {name} content_sha256"
            ),
            "byte_count": _strict_int(
                record["byte_count"], f"calendar source {name} byte_count", minimum=1
            ),
        }
    legacy_hash = canonical_sha256(list(LEGACY_AUTHORITATIVE_SESSION_DATES))
    current_hash = canonical_sha256(list(AUTHORITATIVE_SESSION_DATES))
    market_hash = canonical_sha256(list(AUTHORITATIVE_MARKET_SESSION_DATES))
    extension = build_calendar_extension_manifest(
        prior_session_dates=LEGACY_AUTHORITATIVE_SESSION_DATES,
        extended_session_dates=AUTHORITATIVE_SESSION_DATES,
        expected_prior_sessions_sha256=legacy_hash,
    )
    body = {
        "schema_version": CALENDAR_SOURCE_EVIDENCE_SCHEMA_VERSION,
        "retrieved_at_utc": retrieved_at_utc,
        "source_records": normalized_records,
        "legacy_calendar_id": LEGACY_AUTHORITATIVE_CALENDAR_ID,
        "legacy_sessions_sha256": legacy_hash,
        "legacy_session_count": len(LEGACY_AUTHORITATIVE_SESSION_DATES),
        "current_calendar_id": AUTHORITATIVE_CALENDAR_ID,
        "current_sessions_sha256": current_hash,
        "current_session_count": len(AUTHORITATIVE_SESSION_DATES),
        "market_history_calendar_id": AUTHORITATIVE_MARKET_CALENDAR_ID,
        "market_history_sessions_sha256": market_hash,
        "market_history_session_count": len(AUTHORITATIVE_MARKET_SESSION_DATES),
        "extension_manifest_sha256": extension["extension_manifest_sha256"],
        "calendar_semantics": "trading_session_dates_not_intraday_hours",
        "pre_2024_independent_reference_sessions_sha256": (
            "af8ca7c0efc6c1e39df40eeb0dce47482077704041befbf12c6c188f2c706369"
        ),
        "pre_2024_independent_reference_session_count": 6_037,
        "semantic_reconciliation_required_before_candidate_use": True,
        "authorizes_corpus_or_model_access": False,
    }
    return {**body, "calendar_source_evidence_sha256": canonical_sha256(body)}


def validate_calendar_source_evidence_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_calendar_source_evidence_sha256: str,
) -> str:
    observed = _expect_mapping(manifest, "calendar source evidence")
    expected_keys = {
        "schema_version",
        "retrieved_at_utc",
        "source_records",
        "legacy_calendar_id",
        "legacy_sessions_sha256",
        "legacy_session_count",
        "current_calendar_id",
        "current_sessions_sha256",
        "current_session_count",
        "market_history_calendar_id",
        "market_history_sessions_sha256",
        "market_history_session_count",
        "extension_manifest_sha256",
        "calendar_semantics",
        "pre_2024_independent_reference_sessions_sha256",
        "pre_2024_independent_reference_session_count",
        "semantic_reconciliation_required_before_candidate_use",
        "authorizes_corpus_or_model_access",
        "calendar_source_evidence_sha256",
    }
    _expect_keys(observed, expected_keys, "calendar source evidence")
    rebuilt = build_calendar_source_evidence_manifest(
        retrieved_at_utc=observed["retrieved_at_utc"],
        source_records=_expect_mapping(
            observed["source_records"], "calendar source records"
        ),
    )
    observed_hash = _sha256(
        observed["calendar_source_evidence_sha256"],
        "calendar_source_evidence_sha256",
    )
    expected_hash = _sha256(
        expected_calendar_source_evidence_sha256,
        "expected_calendar_source_evidence_sha256",
    )
    if (
        observed != rebuilt
        or not hmac.compare_digest(observed_hash, expected_hash)
        or not hmac.compare_digest(
            observed_hash, rebuilt["calendar_source_evidence_sha256"]
        )
    ):
        raise SecFilingGemmaContractError(
            "Calendar source evidence is noncanonical or not externally pinned"
        )
    return observed_hash


def _next_session_after(value: date, sessions: Sequence[str]) -> date:
    for raw in sessions:
        candidate = _iso_date(raw, "session calendar")
        if candidate > value:
            return candidate
    raise SecFilingGemmaContractError("Pinned calendar has no later availability session")


def _session_offset(value: date, offset: int, sessions: Sequence[str]) -> date:
    canonical = canonical_session_calendar(sessions)
    iso_value = value.isoformat()
    try:
        position = canonical.index(iso_value)
    except ValueError as exc:
        raise SecFilingGemmaContractError("Claimed date is not a pinned market session") from exc
    target = position + offset
    if target >= len(canonical):
        raise SecFilingGemmaContractError("Pinned calendar does not cover label maturity")
    return _iso_date(canonical[target], "offset market session")


def _stage_for_session(value: date) -> str:
    for stage in STAGE_ORDER:
        first = _iso_date(STAGE_WINDOWS[stage][0], f"{stage} first session")
        last = _iso_date(STAGE_WINDOWS[stage][1], f"{stage} last session")
        if first <= value <= last:
            return stage
    raise SecFilingGemmaContractError("Availability is outside the frozen corpus universe")


def build_corpus_universe_manifest(
    *,
    catalog_artifact_sha256: str,
    calendar_artifact_sha256: str,
    catalog_total_record_count: int,
    catalog_eligible_record_count: int,
    session_dates: Sequence[str],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive the immutable eligible corpus from official-source evidence.

    The effectful acquirer is responsible for proving that the bound catalog
    artifact contains every current and referenced historical Submissions
    record.  This pure function then refuses sampling and recomputes every
    conservative availability session from the pinned calendar.
    """

    catalog_hash = _sha256(catalog_artifact_sha256, "catalog_artifact_sha256")
    calendar_hash = _sha256(calendar_artifact_sha256, "calendar_artifact_sha256")
    total_count = _strict_int(catalog_total_record_count, "catalog_total_record_count", minimum=1)
    eligible_count = _strict_int(
        catalog_eligible_record_count, "catalog_eligible_record_count", minimum=1
    )
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise SecFilingGemmaContractError("Universe records must be a sequence")
    if eligible_count != len(records) or total_count < eligible_count:
        raise SecFilingGemmaContractError(
            "Bound catalog eligible count must equal the complete supplied universe"
        )
    sessions = canonical_session_calendar(session_dates)
    expected_keys = {
        "accession_number",
        "subject_cik",
        "form",
        "acceptance_datetime",
        "filing_date",
        "filing_date_change",
        "primary_document",
        "source_record_sha256",
    }
    normalized: list[dict[str, Any]] = []
    accessions: set[str] = set()
    exact_acceptance_count = 0
    for index, raw in enumerate(records):
        record = _expect_mapping(raw, f"universe records[{index}]")
        _expect_keys(record, expected_keys, f"universe records[{index}]")
        accession = record["accession_number"]
        if (
            not isinstance(accession, str)
            or _ACCESSION_RE.fullmatch(accession) is None
            or not accession.startswith("0000320193-")
            or accession in accessions
        ):
            raise SecFilingGemmaContractError("Universe accession is not a unique Apple accession")
        accessions.add(accession)
        if record["subject_cik"] != "0000320193":
            raise SecFilingGemmaContractError("Universe record subject CIK is not Apple")
        if record["form"] not in {"10-K", "10-Q"}:
            raise SecFilingGemmaContractError("Universe contains an ineligible or amended form")
        filing_date = _iso_date(record["filing_date"], "filing_date")
        change_raw = record["filing_date_change"]
        change_date = None if change_raw is None else _iso_date(change_raw, "filing_date_change")
        acceptance_raw = record["acceptance_datetime"]
        acceptance_date: date | None = None
        if acceptance_raw is not None:
            if not isinstance(acceptance_raw, str) or _ACCEPTANCE_RE.fullmatch(acceptance_raw) is None:
                raise SecFilingGemmaContractError("Acceptance timestamp must be 14 digits or null")
            try:
                acceptance_date = datetime.strptime(acceptance_raw, "%Y%m%d%H%M%S").date()
            except ValueError as exc:
                raise SecFilingGemmaContractError("Acceptance timestamp is invalid") from exc
            exact_acceptance_count += 1
        evidence_dates = [filing_date]
        if acceptance_date is not None:
            evidence_dates.append(acceptance_date)
        if change_date is not None:
            evidence_dates.append(change_date)
        availability = _next_session_after(max(evidence_dates), sessions)
        stage = _stage_for_session(availability)
        primary_document = record["primary_document"]
        if (
            not isinstance(primary_document, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", primary_document) is None
        ):
            raise SecFilingGemmaContractError("Primary document must be a safe basename")
        normalized.append(
            {
                "accession_number": accession,
                "subject_cik": "0000320193",
                "form": record["form"],
                "acceptance_datetime": acceptance_raw,
                "filing_date": filing_date.isoformat(),
                "filing_date_change": None if change_date is None else change_date.isoformat(),
                "primary_document": primary_document,
                "source_record_sha256": _sha256(
                    record["source_record_sha256"], "source_record_sha256"
                ),
                "availability_session": availability.isoformat(),
                "artifact_stage": stage,
            }
        )
    normalized.sort(key=lambda item: (item["availability_session"], item["accession_number"]))
    stage_counts = {
        stage: sum(record["artifact_stage"] == stage for record in normalized)
        for stage in STAGE_ORDER
    }
    for stage, count in stage_counts.items():
        if count > STAGE_MODEL_CALL_CAPS[stage]:
            raise SecFilingGemmaContractError(f"{stage} exceeds its complete-universe call cap")
    acceptance_rate = exact_acceptance_count / len(normalized)
    if acceptance_rate < 0.95:
        raise SecFilingGemmaContractError("Exact SGML acceptance coverage is below 95 percent")
    body = {
        "schema_version": UNIVERSE_SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "catalog_artifact_sha256": catalog_hash,
        "calendar_artifact_sha256": calendar_hash,
        "calendar_sessions_sha256": canonical_sha256(list(sessions)),
        "catalog_total_record_count": total_count,
        "catalog_eligible_record_count": eligible_count,
        "exact_acceptance_timestamp_count": exact_acceptance_count,
        "exact_acceptance_timestamp_rate": acceptance_rate,
        "stage_counts": stage_counts,
        "records": normalized,
    }
    semantic_records = [
        {
            "accession_number": record["accession_number"],
            "form": record["form"],
            "availability_session": record["availability_session"],
            "artifact_stage": record["artifact_stage"],
        }
        for record in normalized
    ]
    semantic_body = {
        "schema_version": "aapl-sec-gemma-corpus-universe-semantic-v1",
        "contract_sha256": body["contract_sha256"],
        "calendar_sessions_sha256": body["calendar_sessions_sha256"],
        "behavioral_records": semantic_records,
    }
    body["universe_semantic_sha256"] = canonical_sha256(semantic_body)
    return {**body, "universe_sha256": canonical_sha256(body)}


def validate_corpus_universe_manifest(
    manifest: Mapping[str, Any],
    *,
    session_dates: Sequence[str],
    expected_universe_sha256: str | None = None,
    require_complete_coverage: bool = True,
) -> str:
    value = _expect_mapping(manifest, "corpus universe manifest")
    _expect_keys(
        value,
        {
            "schema_version",
            "contract_sha256",
            "catalog_artifact_sha256",
            "calendar_artifact_sha256",
            "calendar_sessions_sha256",
            "catalog_total_record_count",
            "catalog_eligible_record_count",
            "exact_acceptance_timestamp_count",
            "exact_acceptance_timestamp_rate",
            "stage_counts",
            "records",
            "universe_semantic_sha256",
            "universe_sha256",
        },
        "corpus universe manifest",
    )
    source_records: list[dict[str, Any]] = []
    for raw in value["records"]:
        record = _expect_mapping(raw, "universe record")
        source_records.append(
            {
                key: record[key]
                for key in (
                    "accession_number",
                    "subject_cik",
                    "form",
                    "acceptance_datetime",
                    "filing_date",
                    "filing_date_change",
                    "primary_document",
                    "source_record_sha256",
                )
            }
        )
    rebuilt = build_corpus_universe_manifest(
        catalog_artifact_sha256=value["catalog_artifact_sha256"],
        calendar_artifact_sha256=value["calendar_artifact_sha256"],
        catalog_total_record_count=value["catalog_total_record_count"],
        catalog_eligible_record_count=value["catalog_eligible_record_count"],
        session_dates=session_dates,
        records=source_records,
    )
    if value != rebuilt:
        raise SecFilingGemmaContractError("Corpus universe manifest is not canonical")
    universe_hash = _sha256(value["universe_sha256"], "universe_sha256")
    if expected_universe_sha256 is not None and not hmac.compare_digest(
        universe_hash,
        _sha256(expected_universe_sha256, "expected_universe_sha256"),
    ):
        raise SecFilingGemmaContractError("Corpus universe does not match its external pin")
    if require_complete_coverage:
        validate_complete_corpus(value)
    return universe_hash


def validate_complete_corpus(universe_manifest: Mapping[str, Any]) -> None:
    value = _expect_mapping(universe_manifest, "corpus universe manifest")
    records = value.get("records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise SecFilingGemmaContractError("Corpus universe records are missing")
    stage_counts = value.get("stage_counts")
    if not isinstance(stage_counts, Mapping):
        raise SecFilingGemmaContractError("Corpus universe stage counts are missing")
    if stage_counts.get("development", 0) < 72:
        raise SecFilingGemmaContractError("Development corpus contains fewer than 72 filings")
    if stage_counts.get("intermediate", 0) < 19:
        raise SecFilingGemmaContractError("Intermediate corpus contains fewer than 19 filings")
    for year in range(2000, 2026):
        yearly = [
            record
            for record in records
            if _iso_date(record["availability_session"], "availability_session").year == year
        ]
        k_count = sum(record["form"] == "10-K" for record in yearly)
        q_count = sum(record["form"] == "10-Q" for record in yearly)
        if len(yearly) < 3 or k_count < 1 or q_count < 2:
            raise SecFilingGemmaContractError(
                f"Completed year {year} lacks frozen periodic filing coverage"
            )


def build_stage_content_manifest(
    *,
    artifact_stage: str,
    corpus_universe_sha256: str,
    documents: Sequence[Mapping[str, Any]],
    universe_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind only one authorized stage's filing bytes and normalized text."""

    if artifact_stage not in STAGE_ORDER:
        raise SecFilingGemmaContractError("Stage content artifact has an invalid stage")
    universe_hash = _sha256(corpus_universe_sha256, "corpus_universe_sha256")
    if universe_manifest.get("universe_sha256") != universe_hash:
        raise SecFilingGemmaContractError("Stage content universe is not bound")
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise SecFilingGemmaContractError("Stage content documents must be a sequence")
    universe_records = {
        record["accession_number"]: record
        for record in universe_manifest["records"]
        if record["artifact_stage"] == artifact_stage
    }
    expected_keys = {
        "accession_number",
        "primary_document_sha256",
        "normalized_text_sha256",
        "primary_document_bytes",
        "normalized_text_bytes",
    }
    normalized: list[dict[str, Any]] = []
    observed: set[str] = set()
    for index, raw in enumerate(documents):
        document = _expect_mapping(raw, f"stage content documents[{index}]")
        _expect_keys(document, expected_keys, f"stage content documents[{index}]")
        accession = document["accession_number"]
        if accession not in universe_records or accession in observed:
            raise SecFilingGemmaContractError(
                "Stage content must contain each authorized accession exactly once"
            )
        observed.add(accession)
        universe_record = universe_records[accession]
        normalized.append(
            {
                "accession_number": accession,
                "form": universe_record["form"],
                "availability_session": universe_record["availability_session"],
                "primary_document": universe_record["primary_document"],
                "primary_document_sha256": _sha256(
                    document["primary_document_sha256"], "primary_document_sha256"
                ),
                "normalized_text_sha256": _sha256(
                    document["normalized_text_sha256"], "normalized_text_sha256"
                ),
                "primary_document_bytes": _strict_int(
                    document["primary_document_bytes"],
                    "primary_document_bytes",
                    minimum=1,
                ),
                "normalized_text_bytes": _strict_int(
                    document["normalized_text_bytes"],
                    "normalized_text_bytes",
                    minimum=1,
                ),
            }
        )
    if observed != set(universe_records):
        raise SecFilingGemmaContractError(
            "Stage content omits one or more metadata-eligible accessions"
        )
    normalized.sort(key=lambda item: (item["availability_session"], item["accession_number"]))
    body = {
        "schema_version": CONTENT_MANIFEST_SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "artifact_stage": artifact_stage,
        "corpus_universe_sha256": universe_hash,
        "document_count": len(normalized),
        "documents": normalized,
    }
    return {**body, "content_manifest_sha256": canonical_sha256(body)}


def validate_stage_content_manifest(
    manifest: Mapping[str, Any],
    *,
    universe_manifest: Mapping[str, Any],
    expected_content_manifest_sha256: str,
) -> str:
    value = _expect_mapping(manifest, "stage content manifest")
    _expect_keys(
        value,
        {
            "schema_version",
            "contract_sha256",
            "artifact_stage",
            "corpus_universe_sha256",
            "document_count",
            "documents",
            "content_manifest_sha256",
        },
        "stage content manifest",
    )
    source_documents = [
        {
            key: document[key]
            for key in (
                "accession_number",
                "primary_document_sha256",
                "normalized_text_sha256",
                "primary_document_bytes",
                "normalized_text_bytes",
            )
        }
        for document in value["documents"]
    ]
    rebuilt = build_stage_content_manifest(
        artifact_stage=value["artifact_stage"],
        corpus_universe_sha256=value["corpus_universe_sha256"],
        documents=source_documents,
        universe_manifest=universe_manifest,
    )
    if value != rebuilt:
        raise SecFilingGemmaContractError("Stage content manifest is not canonical")
    observed = _sha256(
        value["content_manifest_sha256"],
        "content_manifest_sha256",
    )
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_content_manifest_sha256,
            "expected_content_manifest_sha256",
        ),
    ):
        raise SecFilingGemmaContractError("Stage content manifest is not externally pinned")
    return observed


def validate_extractor_request(
    request: Mapping[str, Any],
    *,
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
    universe_manifest: Mapping[str, Any],
    content_manifests_by_stage: Mapping[str, Mapping[str, Any]],
    expected_content_manifest_sha256s: Mapping[str, str],
    session_dates: Sequence[str],
    forbidden_identity_terms: Sequence[str],
    redacted_input_manifest: Mapping[str, Any],
    expected_redacted_input_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate a metadata envelope and return only the safe Ollama payload."""

    candidate_hash = validate_candidate_manifest(
        candidate_manifest, expected_candidate_sha256=expected_candidate_sha256
    )
    bindings = candidate_manifest["bindings"]
    value = _expect_mapping(request, "extractor metadata envelope")
    _expect_keys(
        value,
        {
            "request_version",
            "preprocessor_version",
            "corpus_universe_sha256",
            "identity_lexicon_sha256",
            "redacted_input_manifest_sha256",
            "stage",
            "current_accession_number",
            "current_form",
            "current_availability_session",
            "current_filing_sha256",
            "prior_accession_number",
            "prior_availability_session",
            "prior_same_form_filing_sha256",
            "model_payload",
            "model_payload_sha256",
            "redaction_report",
        },
        "extractor metadata envelope",
    )
    if value["request_version"] != EXTRACTOR_REQUEST_VERSION:
        raise SecFilingGemmaContractError("Extractor request version changed")
    if value["preprocessor_version"] != PREPROCESSOR_VERSION:
        raise SecFilingGemmaContractError("Sentence preprocessor changed")
    universe_hash = validate_corpus_universe_manifest(
        universe_manifest,
        session_dates=session_dates,
        expected_universe_sha256=bindings["corpus_universe_sha256"],
    )
    _validate_candidate_universe_provenance(candidate_manifest, universe_manifest)
    if value["corpus_universe_sha256"] != universe_hash:
        raise SecFilingGemmaContractError("Extractor request is not bound to the corpus universe")
    if isinstance(forbidden_identity_terms, (str, bytes)) or not isinstance(
        forbidden_identity_terms, Sequence
    ):
        raise SecFilingGemmaContractError("Identity lexicon must be a sequence")
    normalized_lexicon = sorted(
        {
            term.strip().casefold()
            for term in forbidden_identity_terms
            if isinstance(term, str) and term.strip()
        }
    )
    if len(normalized_lexicon) != len(forbidden_identity_terms):
        raise SecFilingGemmaContractError("Identity lexicon must contain unique nonblank strings")
    if not set(MANDATORY_IDENTITY_TERMS).issubset(normalized_lexicon):
        raise SecFilingGemmaContractError(
            "Identity lexicon omits a mandatory issuer identity term"
        )
    lexicon_hash = canonical_sha256(normalized_lexicon)
    if not hmac.compare_digest(
        lexicon_hash,
        bindings["identity_lexicon_sha256"],
    ) or value["identity_lexicon_sha256"] != lexicon_hash:
        raise SecFilingGemmaContractError("Extractor identity lexicon is not externally pinned")
    records = list(universe_manifest["records"])
    current_accession = value["current_accession_number"]
    current = next(
        (record for record in records if record["accession_number"] == current_accession),
        None,
    )
    if current is None:
        raise SecFilingGemmaContractError("Extractor current filing is absent from the universe")
    if value["stage"] != current["artifact_stage"]:
        raise SecFilingGemmaContractError("Extractor current filing crossed a stage boundary")
    allowed_stages = set(STAGE_ORDER[: STAGE_ORDER.index(current["artifact_stage"]) + 1])
    manifests = _expect_mapping(content_manifests_by_stage, "content_manifests_by_stage")
    expected_content_hashes = _expect_mapping(
        expected_content_manifest_sha256s,
        "expected_content_manifest_sha256s",
    )
    _expect_keys(manifests, allowed_stages, "content_manifests_by_stage")
    _expect_keys(
        expected_content_hashes,
        allowed_stages,
        "expected_content_manifest_sha256s",
    )
    content_by_accession: dict[str, Mapping[str, Any]] = {}
    for stage in STAGE_ORDER:
        if stage not in allowed_stages:
            continue
        manifest = _expect_mapping(manifests[stage], f"content manifest {stage}")
        if manifest.get("artifact_stage") != stage:
            raise SecFilingGemmaContractError("Content manifest stage key is inconsistent")
        validate_stage_content_manifest(
            manifest,
            universe_manifest=universe_manifest,
            expected_content_manifest_sha256=expected_content_hashes[stage],
        )
        for document in manifest["documents"]:
            content_by_accession[document["accession_number"]] = document
    current_content = content_by_accession.get(current_accession)
    if current_content is None:
        raise SecFilingGemmaContractError("Extractor current filing content is unavailable")
    redacted_manifest = _expect_mapping(
        redacted_input_manifest,
        "redacted-input manifest",
    )
    if redacted_manifest.get("artifact_stage") != current["artifact_stage"]:
        raise SecFilingGemmaContractError(
            "Extractor redacted-input artifact crossed a stage boundary"
        )
    if redacted_manifest.get("accession_number") != current_accession:
        raise SecFilingGemmaContractError(
            "Extractor redacted-input artifact belongs to another filing event"
        )
    redacted_manifest_hash = validate_redacted_input_manifest(
        redacted_manifest,
        universe_manifest=universe_manifest,
        stage_content_manifest=manifests[current["artifact_stage"]],
        expected_manifest_sha256=expected_redacted_input_manifest_sha256,
    )
    if value["redacted_input_manifest_sha256"] != redacted_manifest_hash:
        raise SecFilingGemmaContractError(
            "Extractor envelope is not bound to the redacted-input manifest"
        )
    if value["current_form"] != current["form"]:
        raise SecFilingGemmaContractError("Extractor current form differs from the universe")
    if value["current_availability_session"] != current["availability_session"]:
        raise SecFilingGemmaContractError("Extractor availability differs from the universe")
    if value["current_filing_sha256"] != current_content["normalized_text_sha256"]:
        raise SecFilingGemmaContractError("Extractor current text hash differs from the universe")
    same_form_before = [
        record
        for record in records
        if record["form"] == current["form"]
        and (
            record["availability_session"],
            record["accession_number"],
        )
        < (current["availability_session"], current["accession_number"])
    ]
    expected_prior = same_form_before[-1] if same_form_before else None
    prior_accession = value["prior_accession_number"]
    prior_availability = value["prior_availability_session"]
    prior_hash = value["prior_same_form_filing_sha256"]
    if expected_prior is None:
        if any(item is not None for item in (prior_accession, prior_availability, prior_hash)):
            raise SecFilingGemmaContractError("First same-form filing cannot claim a prior filing")
    else:
        prior_content = content_by_accession.get(expected_prior["accession_number"])
        if prior_content is None:
            raise SecFilingGemmaContractError(
                "Extractor prior same-form content is unavailable"
            )
        if (
            prior_accession != expected_prior["accession_number"]
            or prior_availability != expected_prior["availability_session"]
            or prior_hash != prior_content["normalized_text_sha256"]
        ):
            raise SecFilingGemmaContractError(
                "Extractor prior filing is not the immediately preceding same-form filing"
            )
        if prior_hash == value["current_filing_sha256"]:
            raise SecFilingGemmaContractError("Current filing cannot be its own prior evidence")
    model_payload = _expect_mapping(value["model_payload"], "model_payload")
    messages = model_payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise SecFilingGemmaContractError("Model payload must contain exactly two messages")
    user_message = _expect_mapping(messages[1], "model user message")
    if set(user_message) != {"role", "content"} or user_message.get("role") != "user":
        raise SecFilingGemmaContractError("Model user message is not canonical")
    try:
        user_content = json.loads(user_message["content"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise SecFilingGemmaContractError("Model user content is not canonical JSON") from exc
    user_mapping = _expect_mapping(user_content, "model user content")
    _expect_keys(user_mapping, {"sentences"}, "model user content")
    if user_message["content"] != _canonical_json_text(user_mapping):
        raise SecFilingGemmaContractError("Model user content is not canonically encoded")
    sentences = user_mapping["sentences"]
    if isinstance(sentences, (str, bytes)) or not isinstance(sentences, Sequence):
        raise SecFilingGemmaContractError("sentences must be a sequence")
    if not 1 <= len(sentences) <= MAX_SENTENCES:
        raise SecFilingGemmaContractError("Extractor sentence count is outside the cap")
    ids: list[str] = []
    texts: list[str] = []
    for index, raw in enumerate(sentences):
        sentence = _expect_mapping(raw, f"sentences[{index}]")
        _expect_keys(sentence, {"id", "text"}, f"sentences[{index}]")
        sentence_id = sentence["id"]
        text = sentence["text"]
        if not isinstance(sentence_id, str) or _SENTENCE_ID_RE.fullmatch(sentence_id) is None:
            raise SecFilingGemmaContractError("Sentence ids must be C#### or P####")
        if not isinstance(text, str) or not text.strip():
            raise SecFilingGemmaContractError("Redacted sentence text cannot be blank")
        if text != text.strip() or len(text) > MAX_SENTENCE_CHARACTERS:
            raise SecFilingGemmaContractError("Redacted sentence text is not canonical")
        if not text.isascii():
            raise SecFilingGemmaContractError(
                "Redacted sentence text must be canonical ASCII"
            )
        lowered = text.casefold()
        if any(
            re.search(
                rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
                lowered,
            )
            is not None
            for term in normalized_lexicon
        ):
            raise SecFilingGemmaContractError("Issuer identity survived redaction")
        if re.search(r"[0-9$%]", text):
            raise SecFilingGemmaContractError("Absolute numeric information survived redaction")
        ordered_words = re.findall(r"[a-z]+", lowered)
        words = set(ordered_words)
        if words.intersection(_SPELLED_ABSOLUTE_TERMS):
            raise SecFilingGemmaContractError(
                "Spelled absolute numeric information survived redaction"
            )
        unambiguous_calendar_words = {
            "january",
            "february",
            "march",
            "april",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        }
        if words.intersection(unambiguous_calendar_words):
            raise SecFilingGemmaContractError("Calendar date language survived redaction")
        for position, word in enumerate(ordered_words):
            if word != "may":
                continue
            previous_word = ordered_words[position - 1] if position else None
            next_word = (
                ordered_words[position + 1]
                if position + 1 < len(ordered_words)
                else None
            )
            if previous_word in {
                "by",
                "during",
                "from",
                "in",
                "on",
                "since",
                "through",
                "until",
            } or next_word in _SPELLED_DATE_TERMS:
                raise SecFilingGemmaContractError(
                    "Calendar date language survived redaction"
                )
        month_positions = [
            index for index, word in enumerate(ordered_words) if word == "may"
        ]
        ordinal_positions = [
            index
            for index, word in enumerate(ordered_words)
            if word in _SPELLED_DATE_TERMS
            and word not in unambiguous_calendar_words | {"may"}
        ]
        if any(abs(month - ordinal) <= 3 for month in month_positions for ordinal in ordinal_positions):
            raise SecFilingGemmaContractError("Spelled date information survived redaction")
        ids.append(sentence_id)
        texts.append(text)
    if len(ids) != len(set(ids)):
        raise SecFilingGemmaContractError("Sentence ids must be unique")
    current_ids = [item for item in ids if item.startswith("C")]
    prior_ids = [item for item in ids if item.startswith("P")]
    if not current_ids:
        raise SecFilingGemmaContractError("Every request requires current-filing evidence")
    expected_ids = [f"C{index:04d}" for index in range(1, len(current_ids) + 1)]
    expected_ids += [f"P{index:04d}" for index in range(1, len(prior_ids) + 1)]
    if ids != expected_ids:
        raise SecFilingGemmaContractError("Sentence ids must be contiguous and ordered")
    if (prior_hash is None) != (not prior_ids):
        raise SecFilingGemmaContractError(
            "Prior filing hash and prior sentence evidence must agree"
        )
    expected_payload = build_extractor_model_payload(
        [_expect_mapping(sentence, "model sentence") for sentence in sentences]
    )
    if model_payload != expected_payload:
        raise SecFilingGemmaContractError(
            "Model payload contains metadata or differs from the exact safe payload"
        )
    payload_hash = canonical_sha256(model_payload)
    if value["model_payload_sha256"] != payload_hash:
        raise SecFilingGemmaContractError("Model payload hash does not reconcile")
    expected_payload_hash = redacted_manifest["model_payload_sha256"]
    if payload_hash != expected_payload_hash:
        raise SecFilingGemmaContractError(
            "Model payload does not match the externally pinned redacted-input manifest"
        )
    byte_count = len("\n".join(texts).encode("utf-8"))
    if byte_count > MAX_INPUT_BYTES:
        raise SecFilingGemmaContractError("Extractor request exceeds the UTF-8 byte cap")
    report = _expect_mapping(value["redaction_report"], "redaction_report")
    _expect_keys(
        report,
        {
            "identity_matches_remaining",
            "numeric_matches_remaining",
            "date_matches_remaining",
            "forbidden_context_fields",
            "sentence_count",
            "utf8_bytes",
            "maximum_sentence_characters",
        },
        "redaction_report",
    )
    for field in (
        "identity_matches_remaining",
        "numeric_matches_remaining",
        "date_matches_remaining",
        "forbidden_context_fields",
    ):
        if _strict_int(report[field], f"redaction_report.{field}") != 0:
            raise SecFilingGemmaContractError("Redaction report contains a failed gate")
    if _strict_int(report["sentence_count"], "redaction_report.sentence_count") != len(ids):
        raise SecFilingGemmaContractError("Redaction sentence count does not reconcile")
    if _strict_int(report["utf8_bytes"], "redaction_report.utf8_bytes") != byte_count:
        raise SecFilingGemmaContractError("Redaction byte count does not reconcile")
    maximum = max(len(text) for text in texts)
    if _strict_int(
        report["maximum_sentence_characters"],
        "redaction_report.maximum_sentence_characters",
    ) != maximum:
        raise SecFilingGemmaContractError("Redaction character count does not reconcile")
    return {
        "candidate_sha256": candidate_hash,
        "sentence_ids": tuple(ids),
        "model_payload": copy.deepcopy(expected_payload),
        "model_payload_sha256": payload_hash,
    }


def validate_extractor_output(
    output: Mapping[str, Any], *, supplied_sentence_ids: Sequence[str]
) -> None:
    value = _expect_mapping(output, "extractor output")
    _expect_keys(
        value,
        {"schema_version", "document_quality", "dimensions", "flags"},
        "extractor output",
    )
    if value["schema_version"] != EXTRACTOR_SCHEMA_VERSION:
        raise SecFilingGemmaContractError("Extractor schema version changed")
    if value["document_quality"] not in DOCUMENT_QUALITIES:
        raise SecFilingGemmaContractError("Invalid document quality")
    supplied = set(supplied_sentence_ids)
    if len(supplied) != len(supplied_sentence_ids) or any(
        not isinstance(item, str) or _SENTENCE_ID_RE.fullmatch(item) is None
        for item in supplied
    ):
        raise SecFilingGemmaContractError("Invalid supplied sentence ids")
    dimensions = _expect_mapping(value["dimensions"], "dimensions")
    flags = _expect_mapping(value["flags"], "flags")
    _expect_keys(dimensions, set(DIMENSION_NAMES), "dimensions")
    _expect_keys(flags, set(FLAG_NAMES), "flags")
    unusable = value["document_quality"] == "unusable"
    for name in DIMENSION_NAMES:
        item = _expect_mapping(dimensions[name], f"dimensions.{name}")
        _expect_keys(
            item,
            {"current_impact", "change_vs_prior", "evidence_sentence_ids"},
            f"dimensions.{name}",
        )
        current = item["current_impact"]
        change = item["change_vs_prior"]
        if current not in CURRENT_IMPACTS or change not in COMPARATIVE_CHANGES:
            raise SecFilingGemmaContractError(f"Invalid semantic enum in {name}")
        evidence = item["evidence_sentence_ids"]
        if not isinstance(evidence, list) or len(evidence) != len(set(evidence)):
            raise SecFilingGemmaContractError(f"Invalid evidence list in {name}")
        if any(item_id not in supplied for item_id in evidence):
            raise SecFilingGemmaContractError(f"Unknown evidence id in {name}")
        has_current = any(item_id.startswith("C") for item_id in evidence)
        has_prior = any(item_id.startswith("P") for item_id in evidence)
        if current != "not_stated" and not has_current:
            raise SecFilingGemmaContractError(f"Current impact in {name} lacks evidence")
        if change not in {"not_stated", "not_comparable"} and not (
            has_current and has_prior
        ):
            raise SecFilingGemmaContractError(
                f"Comparative change in {name} lacks current and prior evidence"
            )
        if current == "not_stated" and change in {"not_stated", "not_comparable"} and evidence:
            raise SecFilingGemmaContractError(f"Unstated dimension {name} cited evidence")
        if unusable and (
            current != "not_stated" or change != "not_stated" or evidence
        ):
            raise SecFilingGemmaContractError("Unusable documents cannot assert semantics")
    for name in FLAG_NAMES:
        item = _expect_mapping(flags[name], f"flags.{name}")
        _expect_keys(item, {"present", "evidence_sentence_ids"}, f"flags.{name}")
        present = _strict_bool(item["present"], f"flags.{name}.present")
        evidence = item["evidence_sentence_ids"]
        if not isinstance(evidence, list) or len(evidence) != len(set(evidence)):
            raise SecFilingGemmaContractError(f"Invalid evidence list in flag {name}")
        if any(item_id not in supplied for item_id in evidence):
            raise SecFilingGemmaContractError(f"Unknown flag evidence id in {name}")
        if present and not any(item_id.startswith("C") for item_id in evidence):
            raise SecFilingGemmaContractError(f"True flag {name} lacks current evidence")
        if not present and evidence:
            raise SecFilingGemmaContractError(f"False flag {name} cited evidence")
        if unusable and present:
            raise SecFilingGemmaContractError("Unusable documents cannot assert flags")


def validate_stage_extraction_coverage(
    stage: str,
    qualities_by_accession: Mapping[str, str],
    *,
    universe_manifest: Mapping[str, Any],
) -> None:
    if stage not in STAGE_ORDER:
        raise SecFilingGemmaContractError("Extraction stage is invalid")
    qualities = _expect_mapping(qualities_by_accession, "stage qualities_by_accession")
    expected_accessions = {
        record["accession_number"]
        for record in universe_manifest["records"]
        if record["artifact_stage"] == stage
    }
    _expect_keys(qualities, expected_accessions, "stage qualities_by_accession")
    allowed = DOCUMENT_QUALITIES | {"invalid"}
    if any(quality not in allowed for quality in qualities.values()):
        raise SecFilingGemmaContractError("Extraction coverage contains an invalid quality")
    available = sum(quality in {"usable", "thin"} for quality in qualities.values())
    if available / len(expected_accessions) < 0.90:
        raise SecFilingGemmaContractError(
            f"{stage} extraction availability is below 90 percent"
        )


def validate_final_extraction_coverage(
    qualities_by_accession: Mapping[str, str],
    *,
    universe_manifest: Mapping[str, Any],
) -> None:
    qualities = _expect_mapping(qualities_by_accession, "qualities_by_accession")
    records = list(universe_manifest["records"])
    expected_accessions = {record["accession_number"] for record in records}
    _expect_keys(qualities, expected_accessions, "qualities_by_accession")
    allowed = DOCUMENT_QUALITIES | {"invalid"}
    if any(quality not in allowed for quality in qualities.values()):
        raise SecFilingGemmaContractError("Extraction coverage contains an invalid quality")
    available = {
        accession
        for accession, quality in qualities.items()
        if quality in {"usable", "thin"}
    }
    for stage in STAGE_ORDER:
        stage_accessions = {
            record["accession_number"]
            for record in records
            if record["artifact_stage"] == stage
        }
        if stage_accessions and len(stage_accessions & available) / len(stage_accessions) < 0.90:
            raise SecFilingGemmaContractError(
                f"{stage} extraction availability is below 90 percent"
            )
    if len(available) / len(records) < 0.95:
        raise SecFilingGemmaContractError("Overall extraction availability is below 95 percent")


def validate_training_rows(
    phase: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
    universe_manifest: Mapping[str, Any],
    expected_universe_sha256: str,
    extraction_artifact_sha256: str,
    market_data_manifest_sha256: str,
    label_ledger_sha256: str,
    session_dates: Sequence[str],
    expected_calendar_sessions_sha256: str,
) -> str:
    cutoffs = {
        "development_fit": date(2018, 12, 31),
        "pre_final_refit": date(2023, 12, 31),
    }
    if phase not in cutoffs:
        raise SecFilingGemmaContractError("Training is forbidden in the requested phase")
    candidate_hash = validate_candidate_manifest(
        candidate_manifest, expected_candidate_sha256=expected_candidate_sha256
    )
    universe_hash = validate_corpus_universe_manifest(
        universe_manifest,
        session_dates=session_dates,
        expected_universe_sha256=expected_universe_sha256,
    )
    _validate_candidate_universe_provenance(candidate_manifest, universe_manifest)
    if universe_hash != candidate_manifest["bindings"]["corpus_universe_sha256"]:
        raise SecFilingGemmaContractError("Training universe is not candidate-bound")
    extraction_hash = _sha256(
        extraction_artifact_sha256, "extraction_artifact_sha256"
    )
    market_hash = _sha256(market_data_manifest_sha256, "market_data_manifest_sha256")
    label_hash = _sha256(label_ledger_sha256, "label_ledger_sha256")
    observed_calendar_hash = session_calendar_sha256(session_dates)
    if not hmac.compare_digest(
        observed_calendar_hash,
        _sha256(expected_calendar_sessions_sha256, "expected_calendar_sessions_sha256"),
    ):
        raise SecFilingGemmaContractError("Training calendar is not externally pinned")
    cutoff = cutoffs[phase]
    expected_keys = {
        "accession_number",
        "candidate_sha256",
        "corpus_universe_sha256",
        "artifact_stage",
        "feature_availability_session",
        "label_maturity_session",
        "horizon_sessions",
        "feature_row_sha256",
        "extraction_identity_sha256",
        "market_prefix_chain_identity_sha256",
        "market_feature_row_sha256",
        "label_evidence_sha256",
        "semantic_available",
        "market_available",
        "prediction_available",
        "fit_eligible",
    }
    observed_ids: set[str] = set()
    previous: tuple[date, str] | None = None
    permitted_stages = (
        {"development"}
        if phase == "development_fit"
        else {"development", "intermediate"}
    )
    universe_by_accession = {
        record["accession_number"]: record for record in universe_manifest["records"]
    }
    expected_accessions: set[str] = set()
    for accession, record in universe_by_accession.items():
        if record["artifact_stage"] not in permitted_stages:
            continue
        feature_date = _iso_date(record["availability_session"], "availability_session")
        maturity_date = _session_offset(
            feature_date, LABEL_MATURITY_OFFSET, session_dates
        )
        if maturity_date <= cutoff:
            expected_accessions.add(accession)
    for index, raw in enumerate(rows):
        row = _expect_mapping(raw, f"training_rows[{index}]")
        _expect_keys(row, expected_keys, f"training_rows[{index}]")
        accession = row["accession_number"]
        if accession not in expected_accessions or accession in observed_ids:
            raise SecFilingGemmaContractError(
                "Training rows must be the exact unique matured filing-event set"
            )
        observed_ids.add(accession)
        universe_record = universe_by_accession[accession]
        if row["candidate_sha256"] != candidate_hash:
            raise SecFilingGemmaContractError("Training row is not candidate-bound")
        if row["corpus_universe_sha256"] != universe_hash:
            raise SecFilingGemmaContractError("Training row is not universe-bound")
        if row["artifact_stage"] != universe_record["artifact_stage"]:
            raise SecFilingGemmaContractError("Training row crossed a corpus stage")
        feature_date = _iso_date(
            row["feature_availability_session"], "feature_availability_session"
        )
        if feature_date.isoformat() != universe_record["availability_session"]:
            raise SecFilingGemmaContractError("Training feature date differs from the filing")
        maturity_date = _iso_date(row["label_maturity_session"], "label_maturity_session")
        required_maturity = _session_offset(
            feature_date, LABEL_MATURITY_OFFSET, session_dates
        )
        if maturity_date != required_maturity:
            raise SecFilingGemmaContractError(
                "Training label maturity is not the exact 20-session outcome"
            )
        if feature_date < date(2000, 1, 1):
            raise SecFilingGemmaContractError("Training feature predates the frozen universe")
        if maturity_date > cutoff:
            raise SecFilingGemmaContractError("Training row uses an outcome beyond its cutoff")
        if _strict_int(row["horizon_sessions"], "horizon_sessions") != HORIZON_SESSIONS:
            raise SecFilingGemmaContractError("Training horizon changed")
        _sha256(row["feature_row_sha256"], "feature_row_sha256")
        _sha256(row["extraction_identity_sha256"], "extraction_identity_sha256")
        _sha256(
            row["market_prefix_chain_identity_sha256"],
            "market_prefix_chain_identity_sha256",
        )
        _sha256(row["market_feature_row_sha256"], "market_feature_row_sha256")
        _sha256(row["label_evidence_sha256"], "label_evidence_sha256")
        semantic_available = _strict_bool(
            row["semantic_available"], "semantic_available"
        )
        market_available = _strict_bool(
            row["market_available"], "market_available"
        )
        prediction_available = _strict_bool(
            row["prediction_available"], "prediction_available"
        )
        fit_eligible = _strict_bool(row["fit_eligible"], "fit_eligible")
        if fit_eligible != prediction_available:
            raise SecFilingGemmaContractError(
                "Learner support must equal prediction-available support"
            )
        if prediction_available and not market_available:
            raise SecFilingGemmaContractError(
                "Prediction availability requires complete market support"
            )
        if semantic_available and not prediction_available:
            raise SecFilingGemmaContractError(
                "Semantic availability cannot survive an unavailable prediction row"
            )
        sort_key = (feature_date, accession)
        if previous is not None and sort_key <= previous:
            raise SecFilingGemmaContractError("Training rows must be chronological")
        previous = sort_key
    if observed_ids != expected_accessions:
        raise SecFilingGemmaContractError(
            "Training rows omit one or more matured eligible filing events"
        )
    return canonical_sha256(
        {
            "phase": phase,
            "candidate_sha256": candidate_hash,
            "corpus_universe_sha256": universe_hash,
            "extraction_artifact_sha256": extraction_hash,
            "market_data_manifest_sha256": market_hash,
            "label_ledger_sha256": label_hash,
            "rows": list(rows),
        }
    )


def validate_final_runtime_summary(
    evidence: Mapping[str, Any],
    *,
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
    universe_manifest: Mapping[str, Any],
    expected_universe_sha256: str,
    session_dates: Sequence[str],
) -> None:
    candidate_hash = validate_candidate_manifest(
        candidate_manifest, expected_candidate_sha256=expected_candidate_sha256
    )
    universe_hash = validate_corpus_universe_manifest(
        universe_manifest,
        session_dates=session_dates,
        expected_universe_sha256=expected_universe_sha256,
    )
    _validate_candidate_universe_provenance(candidate_manifest, universe_manifest)
    if universe_hash != candidate_manifest["bindings"]["corpus_universe_sha256"]:
        raise SecFilingGemmaContractError("Runtime corpus universe is not candidate-bound")
    if (
        universe_manifest["calendar_artifact_sha256"]
        != candidate_manifest["bindings"]["calendar_source_evidence_sha256"]
    ):
        raise SecFilingGemmaContractError("Runtime calendar is not candidate-bound")
    value = _expect_mapping(evidence, "runtime evidence")
    _expect_keys(
        value,
        {
            "complete",
            "partial",
            "candidate_sha256",
            "corpus_universe_sha256",
            "total_elapsed_seconds",
            "projected_total_seconds",
            "sec_acquisition_seconds",
            "gemma_extraction_seconds",
            "fit_simulation_sealing_seconds",
            "sec_requests",
            "sec_bytes",
            "sec_hosts",
            "model_calls_by_stage",
            "called_accessions_by_stage",
            "model_call_ledger_sha256",
            "artifact_sha256s",
            "market_data_sha256s_by_stage",
            "model_endpoint",
            "model_name",
            "model_digest",
            "model_runtime_fingerprint_sha256",
            "model_hosts",
            "paid_api_calls",
            "estimated_cost_usd",
            "pull_attempts",
            "retries",
            "repair_attempts",
        },
        "runtime evidence",
    )
    if not _strict_bool(value["complete"], "complete") or _strict_bool(
        value["partial"], "partial"
    ):
        raise SecFilingGemmaContractError("Partial or incomplete runs cannot pass")
    if value["candidate_sha256"] != candidate_hash:
        raise SecFilingGemmaContractError("Runtime evidence is not candidate-bound")
    if value["corpus_universe_sha256"] != universe_hash:
        raise SecFilingGemmaContractError("Runtime evidence is not corpus-bound")
    limits = (
        ("total_elapsed_seconds", MAX_RUNTIME_SECONDS),
        ("projected_total_seconds", MAX_RUNTIME_SECONDS),
        ("sec_acquisition_seconds", MAX_SEC_SECONDS),
        ("gemma_extraction_seconds", MAX_MODEL_SECONDS),
        ("fit_simulation_sealing_seconds", MAX_FIT_SECONDS),
    )
    for field, maximum in limits:
        if _finite_number(value[field], field, minimum=0.0) > maximum:
            raise SecFilingGemmaContractError(f"{field} exceeds its frozen budget")
    phase_sum = sum(
        _finite_number(value[field], field, minimum=0.0)
        for field in (
            "sec_acquisition_seconds",
            "gemma_extraction_seconds",
            "fit_simulation_sealing_seconds",
        )
    )
    if _finite_number(value["total_elapsed_seconds"], "total_elapsed_seconds") + 1e-9 < phase_sum:
        raise SecFilingGemmaContractError("Runtime phase seconds exceed total elapsed time")
    if _finite_number(value["projected_total_seconds"], "projected_total_seconds") + 1e-9 < phase_sum:
        raise SecFilingGemmaContractError("Runtime phase seconds exceed projected time")
    sec_requests = _strict_int(value["sec_requests"], "sec_requests", minimum=1)
    if sec_requests > MAX_SEC_REQUESTS:
        raise SecFilingGemmaContractError("SEC request cap exceeded")
    if _strict_int(value["sec_bytes"], "sec_bytes") > MAX_SEC_BYTES:
        raise SecFilingGemmaContractError("SEC byte cap exceeded")
    if value["sec_hosts"] not in (["data.sec.gov", "www.sec.gov"], ["www.sec.gov", "data.sec.gov"]):
        raise SecFilingGemmaContractError("Only official SEC hosts are permitted")
    calls = _expect_mapping(value["model_calls_by_stage"], "model_calls_by_stage")
    _expect_keys(calls, set(STAGE_ORDER), "model_calls_by_stage")
    called = _expect_mapping(value["called_accessions_by_stage"], "called_accessions_by_stage")
    _expect_keys(called, set(STAGE_ORDER), "called_accessions_by_stage")
    for stage in STAGE_ORDER:
        expected_accessions = sorted(
            record["accession_number"]
            for record in universe_manifest["records"]
            if record["artifact_stage"] == stage
        )
        if called[stage] != expected_accessions:
            raise SecFilingGemmaContractError(
                f"{stage} call ledger does not cover every eligible filing exactly once"
            )
        if _strict_int(calls[stage], f"model_calls_by_stage.{stage}") != len(
            expected_accessions
        ):
            raise SecFilingGemmaContractError(
                f"{stage} model-call count does not reconcile to the universe"
            )
    _sha256(value["model_call_ledger_sha256"], "model_call_ledger_sha256")
    artifacts = _expect_mapping(value["artifact_sha256s"], "artifact_sha256s")
    _expect_keys(
        artifacts,
        {
            "sec_corpus",
            "development_extraction",
            "intermediate_extraction",
            "final_extraction",
            "fit_simulation_evaluation",
        },
        "artifact_sha256s",
    )
    for name, digest in artifacts.items():
        _sha256(digest, f"artifact_sha256s.{name}")
    if len(set(artifacts.values())) != len(artifacts):
        raise SecFilingGemmaContractError(
            "Runtime stage artifacts must be physically distinct"
        )
    market_data = _expect_mapping(
        value["market_data_sha256s_by_stage"], "market_data_sha256s_by_stage"
    )
    _expect_keys(market_data, set(STAGE_ORDER), "market_data_sha256s_by_stage")
    required_market_keys = {"AAPL", "SPY", "QQQ", "IWM", "VIX", "TNX", "canonical_frame"}
    observed_market_hashes: list[str] = []
    for stage in STAGE_ORDER:
        stage_hashes = _expect_mapping(
            market_data[stage], f"market_data_sha256s_by_stage.{stage}"
        )
        _expect_keys(
            stage_hashes,
            required_market_keys,
            f"market_data_sha256s_by_stage.{stage}",
        )
        for name, digest in stage_hashes.items():
            observed_market_hashes.append(
                _sha256(digest, f"market_data_sha256s_by_stage.{stage}.{name}")
            )
    if len(set(observed_market_hashes)) != len(observed_market_hashes):
        raise SecFilingGemmaContractError(
            "Runtime market artifacts must be distinct stage-bounded slices"
        )
    if value["model_endpoint"] != "http://127.0.0.1:11434/api/chat":
        raise SecFilingGemmaContractError("Model endpoint is not the frozen loopback endpoint")
    if value["model_name"] != "gemma4:12b":
        raise SecFilingGemmaContractError("Model name changed")
    if value["model_digest"] != candidate_manifest["model"]["digest"]:
        raise SecFilingGemmaContractError("Runtime model digest differs from the candidate")
    if (
        value["model_runtime_fingerprint_sha256"]
        != candidate_manifest["model"]["runtime_fingerprint_sha256"]
    ):
        raise SecFilingGemmaContractError("Ollama runtime fingerprint differs from the candidate")
    if value["model_hosts"] != ["127.0.0.1"]:
        raise SecFilingGemmaContractError("Model traffic must remain on IPv4 loopback")
    for field in ("paid_api_calls", "pull_attempts", "retries", "repair_attempts"):
        if _strict_int(value[field], field) != 0:
            raise SecFilingGemmaContractError(f"{field} must remain zero")
    if _finite_number(value["estimated_cost_usd"], "estimated_cost_usd", minimum=0.0) != 0.0:
        raise SecFilingGemmaContractError("Paid API cost must remain exactly zero")


def validate_execution_policy(policy: Mapping[str, Any]) -> None:
    expected = build_contract_manifest()["execution"]
    value = _expect_mapping(policy, "execution policy")
    if value != expected:
        raise SecFilingGemmaContractError("Execution or no-leverage policy changed")


def validate_live_lessons(
    lessons: Sequence[Mapping[str, Any]],
    *,
    as_of_session: str,
    session_dates: Sequence[str],
    expected_calendar_sessions_sha256: str,
    expected_prior_tip_sha256: str | None,
    expected_prior_count: int,
    expected_prior_model_state_sha256: str,
    expected_prior_last_sort_key: Sequence[str] | None,
    prior_lesson_bindings: Sequence[Mapping[str, Any]],
    expected_prior_lesson_bindings_sha256: str,
    eligible_lesson_bindings: Sequence[Mapping[str, Any]],
    expected_eligible_lesson_bindings_sha256: str,
) -> dict[str, Any]:
    as_of = _iso_date(as_of_session, "as_of_session")
    observed_calendar_hash = session_calendar_sha256(session_dates)
    if not hmac.compare_digest(
        observed_calendar_hash,
        _sha256(expected_calendar_sessions_sha256, "expected_calendar_sessions_sha256"),
    ):
        raise SecFilingGemmaContractError("Live lesson calendar is not externally pinned")
    _session_offset(as_of, 0, session_dates)
    prior_count = _strict_int(expected_prior_count, "expected_prior_count")
    if prior_count == 0:
        if expected_prior_tip_sha256 is not None:
            raise SecFilingGemmaContractError("Empty live history cannot have a prior tip")
        if expected_prior_last_sort_key is not None:
            raise SecFilingGemmaContractError("Empty live history cannot have a prior sort key")
        prior_tip = None
        prior_sort_key: tuple[date, date, str] | None = None
    else:
        prior_tip = _sha256(expected_prior_tip_sha256, "expected_prior_tip_sha256")
        if (
            isinstance(expected_prior_last_sort_key, (str, bytes))
            or not isinstance(expected_prior_last_sort_key, Sequence)
            or len(expected_prior_last_sort_key) != 3
            or not isinstance(expected_prior_last_sort_key[2], str)
            or not expected_prior_last_sort_key[2]
        ):
            raise SecFilingGemmaContractError(
                "Nonempty live history requires its externally pinned last sort key"
            )
        prior_sort_key = (
            _iso_date(expected_prior_last_sort_key[0], "prior last ingestion session"),
            _iso_date(expected_prior_last_sort_key[1], "prior last decision session"),
            expected_prior_last_sort_key[2],
        )
    prior_state = _sha256(
        expected_prior_model_state_sha256, "expected_prior_model_state_sha256"
    )
    binding_keys = {
        "lesson_id",
        "accession_number",
        "decision_session",
        "feature_sha256",
        "prediction_receipt_sha256",
        "market_data_manifest_sha256",
        "strategy_ledger_slice_sha256",
        "benchmark_ledger_slice_sha256",
    }

    def normalize_bindings(
        records: Sequence[Mapping[str, Any]], location: str
    ) -> list[dict[str, Any]]:
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise SecFilingGemmaContractError(f"{location} must be a sequence")
        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(records):
            record = _expect_mapping(raw, f"{location}[{index}]")
            _expect_keys(record, binding_keys, f"{location}[{index}]")
            lesson_id = record["lesson_id"]
            if not isinstance(lesson_id, str) or not lesson_id:
                raise SecFilingGemmaContractError(
                    f"{location} lesson ids must be nonblank strings"
                )
            accession = record["accession_number"]
            if (
                not isinstance(accession, str)
                or _ACCESSION_RE.fullmatch(accession) is None
                or not accession.startswith("0000320193-")
            ):
                raise SecFilingGemmaContractError(
                    f"{location} contains a non-Apple accession"
                )
            decision = _iso_date(record["decision_session"], "decision_session")
            _session_offset(decision, 0, session_dates)
            normalized.append(
                {
                    "lesson_id": lesson_id,
                    "accession_number": accession,
                    "decision_session": decision.isoformat(),
                    **{
                        field: _sha256(record[field], field)
                        for field in binding_keys
                        if field.endswith("_sha256")
                    },
                }
            )
        normalized.sort(key=lambda item: item["lesson_id"])
        if len({item["lesson_id"] for item in normalized}) != len(normalized):
            raise SecFilingGemmaContractError(f"{location} lesson ids are duplicated")
        if len({item["accession_number"] for item in normalized}) != len(normalized):
            raise SecFilingGemmaContractError(f"{location} accessions are duplicated")
        if len({item["prediction_receipt_sha256"] for item in normalized}) != len(
            normalized
        ):
            raise SecFilingGemmaContractError(
                f"{location} prediction receipts are duplicated"
            )
        return normalized

    normalized_prior_bindings = normalize_bindings(
        prior_lesson_bindings,
        "prior_lesson_bindings",
    )
    if len(normalized_prior_bindings) != prior_count:
        raise SecFilingGemmaContractError(
            "Prior live lesson bindings do not reconcile to the prior count"
        )
    prior_bindings_hash = canonical_sha256(normalized_prior_bindings)
    if not hmac.compare_digest(
        prior_bindings_hash,
        _sha256(
            expected_prior_lesson_bindings_sha256,
            "expected_prior_lesson_bindings_sha256",
        ),
    ):
        raise SecFilingGemmaContractError(
            "Prior live lesson bindings are not externally pinned"
        )
    normalized_eligible = normalize_bindings(
        eligible_lesson_bindings,
        "eligible_lesson_bindings",
    )
    prior_ids = {item["lesson_id"] for item in normalized_prior_bindings}
    prior_accessions = {
        item["accession_number"] for item in normalized_prior_bindings
    }
    prior_receipts = {
        item["prediction_receipt_sha256"] for item in normalized_prior_bindings
    }
    if (
        prior_ids.intersection(item["lesson_id"] for item in normalized_eligible)
        or prior_accessions.intersection(
            item["accession_number"] for item in normalized_eligible
        )
        or prior_receipts.intersection(
            item["prediction_receipt_sha256"] for item in normalized_eligible
        )
    ):
        raise SecFilingGemmaContractError(
            "Live extension replays a prior lesson, accession, or prediction receipt"
        )
    eligible_hash = canonical_sha256(normalized_eligible)
    if not hmac.compare_digest(
        eligible_hash,
        _sha256(
            expected_eligible_lesson_bindings_sha256,
            "expected_eligible_lesson_bindings_sha256",
        ),
    ):
        raise SecFilingGemmaContractError(
            "Eligible live lesson bindings are not externally pinned"
        )
    eligible_by_id = {item["lesson_id"]: item for item in normalized_eligible}
    expected_keys = {
        "schema_version",
        "sequence_number",
        "lesson_id",
        "accession_number",
        "decision_session",
        "label_maturity_session",
        "ingested_session",
        "first_eligible_decision_session",
        "horizon_sessions",
        "feature_sha256",
        "prediction_receipt_sha256",
        "market_data_manifest_sha256",
        "strategy_ledger_slice_sha256",
        "benchmark_ledger_slice_sha256",
        "cash_beats_long_10bps",
        "cash_active_log_edge_10bps",
        "parent_lesson_sha256",
        "input_model_state_sha256",
        "output_model_state_sha256",
        "lesson_sha256",
    }
    previous_sort_key = prior_sort_key
    previous_lesson_hash = prior_tip
    previous_output_state = prior_state
    lesson_ids: set[str] = set()
    for index, raw in enumerate(lessons):
        lesson = _expect_mapping(raw, f"lessons[{index}]")
        _expect_keys(lesson, expected_keys, f"lessons[{index}]")
        if lesson["schema_version"] != LIVE_LESSON_SCHEMA_VERSION:
            raise SecFilingGemmaContractError("Live lesson schema changed")
        if _strict_int(lesson["sequence_number"], "sequence_number", minimum=1) != (
            prior_count + index + 1
        ):
            raise SecFilingGemmaContractError("Live lesson sequence does not extend the prior ledger")
        lesson_id = lesson["lesson_id"]
        if (
            not isinstance(lesson_id, str)
            or not lesson_id
            or lesson_id in lesson_ids
            or lesson_id in prior_ids
        ):
            raise SecFilingGemmaContractError("Live lesson ids must be unique strings")
        lesson_ids.add(lesson_id)
        accession = lesson["accession_number"]
        if (
            not isinstance(accession, str)
            or _ACCESSION_RE.fullmatch(accession) is None
            or not accession.startswith("0000320193-")
        ):
            raise SecFilingGemmaContractError("Live lesson is not bound to an Apple filing")
        decision = _iso_date(lesson["decision_session"], "decision_session")
        maturity = _iso_date(lesson["label_maturity_session"], "label_maturity_session")
        ingested = _iso_date(lesson["ingested_session"], "ingested_session")
        first_use = _iso_date(
            lesson["first_eligible_decision_session"],
            "first_eligible_decision_session",
        )
        binding = eligible_by_id.get(lesson_id)
        if binding is None or any(
            lesson[field] != binding[field]
            for field in binding_keys
            if field != "lesson_id"
        ):
            raise SecFilingGemmaContractError(
                "Live lesson does not match its externally sealed prediction binding"
            )
        required_maturity = _session_offset(
            decision, LABEL_MATURITY_OFFSET, session_dates
        )
        if maturity != required_maturity:
            raise SecFilingGemmaContractError(
                "Live lesson maturity is not the exact 20-session outcome"
            )
        _session_offset(ingested, 0, session_dates)
        required_first_use = _session_offset(ingested, 1, session_dates)
        if not (maturity <= ingested and first_use == required_first_use):
            raise SecFilingGemmaContractError(
                "A live lesson was used before its outcome causally matured"
            )
        if ingested > as_of:
            raise SecFilingGemmaContractError("Live lesson comes from the future")
        sort_key = (ingested, decision, lesson_id)
        if previous_sort_key is not None and sort_key <= previous_sort_key:
            raise SecFilingGemmaContractError("Live lessons must be append-only chronological")
        previous_sort_key = sort_key
        if _strict_int(lesson["horizon_sessions"], "horizon_sessions") != HORIZON_SESSIONS:
            raise SecFilingGemmaContractError("Live lesson horizon changed")
        _sha256(lesson["feature_sha256"], "feature_sha256")
        for field in (
            "prediction_receipt_sha256",
            "market_data_manifest_sha256",
            "strategy_ledger_slice_sha256",
            "benchmark_ledger_slice_sha256",
        ):
            _sha256(lesson[field], field)
        beats = _strict_bool(lesson["cash_beats_long_10bps"], "cash_beats_long_10bps")
        edge = _finite_number(
            lesson["cash_active_log_edge_10bps"], "cash_active_log_edge_10bps"
        )
        if beats != (edge > ACTIVE_EDGE_TOLERANCE):
            raise SecFilingGemmaContractError(
                "Live lesson Boolean does not reconcile to its 10-bps edge"
            )
        parent_hash = lesson["parent_lesson_sha256"]
        if parent_hash != previous_lesson_hash:
            raise SecFilingGemmaContractError("Live lesson parent hash chain is broken")
        input_state = _sha256(
            lesson["input_model_state_sha256"], "input_model_state_sha256"
        )
        output_state = _sha256(
            lesson["output_model_state_sha256"], "output_model_state_sha256"
        )
        if input_state != previous_output_state or output_state == input_state:
            raise SecFilingGemmaContractError("Live downstream model-state chain is broken")
        body = {key: lesson[key] for key in expected_keys if key != "lesson_sha256"}
        lesson_hash = canonical_sha256(body)
        if not hmac.compare_digest(
            lesson_hash, _sha256(lesson["lesson_sha256"], "lesson_sha256")
        ):
            raise SecFilingGemmaContractError("Live lesson hash is not canonical")
        previous_lesson_hash = lesson_hash
        previous_output_state = output_state
    if sorted(lesson_ids) != sorted(eligible_by_id):
        raise SecFilingGemmaContractError(
            "Live extension does not contain all and only externally eligible lessons"
        )
    combined_bindings = sorted(
        normalized_prior_bindings + normalized_eligible,
        key=lambda item: item["lesson_id"],
    )
    return {
        "lesson_count": prior_count + len(lessons),
        "tip_sha256": previous_lesson_hash,
        "model_state_sha256": previous_output_state,
        "prior_lesson_bindings_sha256": prior_bindings_hash,
        "eligible_lesson_bindings_sha256": eligible_hash,
        "lesson_bindings": combined_bindings,
        "lesson_bindings_sha256": canonical_sha256(combined_bindings),
        "last_sort_key": (
            None
            if previous_sort_key is None
            else [
                previous_sort_key[0].isoformat(),
                previous_sort_key[1].isoformat(),
                previous_sort_key[2],
            ]
        ),
    }


__all__ = [
    "ACTIVE_EDGE_TOLERANCE",
    "ADVERSE_FLAG_NAMES",
    "BRIER_TARGET_COST_BPS",
    "CALENDAR_SOURCE_EVIDENCE_SCHEMA_VERSION",
    "CALENDAR_SOURCE_URLS",
    "CANDIDATE_IDS",
    "CONTENT_MANIFEST_SCHEMA_VERSION",
    "CONTRACT_VERSION",
    "DIMENSION_NAMES",
    "DEVELOPMENT_FOLD_SPECS",
    "EXTRACTOR_REQUEST_VERSION",
    "EXTRACTOR_SCHEMA_VERSION",
    "FLAG_NAMES",
    "HORIZON_SESSIONS",
    "LABEL_MATURITY_OFFSET",
    "LIVE_LESSON_SCHEMA_VERSION",
    "MANDATORY_IDENTITY_TERMS",
    "MARKET_HISTORY_START",
    "MARKET_LOOKBACK_SESSIONS",
    "MAX_RUNTIME_SECONDS",
    "PREPROCESSOR_VERSION",
    "REQUIRED_SOURCE_HASHES",
    "REQUIRED_STAGE_VERIFIER_CHECKS",
    "STAGE_MODEL_CALL_CAPS",
    "STAGE_ORDER",
    "STAGE_WINDOWS",
    "UNIVERSE_SCHEMA_VERSION",
    "SecFilingGemmaContractError",
    "authorize_stage_access",
    "build_calendar_extension_manifest",
    "build_calendar_source_evidence_manifest",
    "build_candidate_manifest",
    "build_contract_manifest",
    "build_corpus_universe_manifest",
    "build_extractor_json_schema",
    "build_extractor_model_payload",
    "build_redacted_input_manifest",
    "build_stage_content_manifest",
    "canonical_session_calendar",
    "canonical_market_session_calendar",
    "canonical_sha256",
    "validate_candidate_manifest",
    "validate_calendar_extension_manifest",
    "validate_calendar_source_evidence_manifest",
    "validate_complete_corpus",
    "validate_contract_manifest",
    "validate_corpus_universe_manifest",
    "validate_execution_policy",
    "validate_final_extraction_coverage",
    "validate_final_runtime_summary",
    "validate_extractor_output",
    "validate_extractor_request",
    "validate_live_lessons",
    "validate_redacted_input_manifest",
    "validate_stage_extraction_coverage",
    "validate_stage_content_manifest",
    "validate_training_rows",
    "market_session_calendar_sha256",
    "session_calendar_sha256",
]
