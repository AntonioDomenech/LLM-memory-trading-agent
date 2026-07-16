"""Locked append-only state for the SEC/Gemma online overlay v2.1.

Production state has one contract-keyed path below the repository's ignored
``data`` directory.  A separate fsynced hash-chain records each database tip,
including a consumption tombstone before any effect capability is returned.
Reopening a consumed attempt can only seal it as terminal-indeterminate.

This protects against ordinary retries, copies, rollbacks, crashes, and
cooperating concurrent processes.  No design using only writable local files
can defeat a machine owner who maliciously rewrites both the database and the
entire anchor history; durable external publication is still required for that
stronger threat model.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import threading
from typing import Any, Final

from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
    ATTEMPT_IDS,
    ATTEMPT_KIND_BY_ID,
    ATTEMPT_ORDINAL_BY_ID,
    CONSUMED,
    DEVELOPMENT_ACQUISITION,
    MINIMUM_PASS_RECORD_COUNTS,
    PLANNED,
    PREREQUISITE_ATTEMPT_BY_ID,
    REPORT_RECORD_TABLES,
    TERMINAL_FAIL,
    TERMINAL_INDETERMINATE,
    TERMINAL_PASS,
    TERMINAL_STATUSES,
    SecGemmaOnlineRiskOverlayAttemptError,
    VerifiedAcquisitionTerminalEvidence,
    VerifiedScoredTerminalEvidence,
    build_attempt_transition,
    build_implementation_manifest,
    current_attempt_status,
    is_verified_acquisition_terminal_evidence,
    is_verified_scored_terminal_evidence,
    store_record_receipt_material,
    validate_attempt_history,
    validate_attempt_plan,
    validate_acquisition_terminal_evidence,
    validate_implementation_manifest,
    validate_scored_terminal_evidence,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    ACQUISITION_TERMINAL_EVIDENCE_FIELDS,
    ACQUISITION_VALIDATION_CHECKS,
    ACQUISITION_VALIDATION_FIELDS,
    CONFIRMATION_ATTEMPT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_ACQUISITION_ID,
    DEVELOPMENT_ATTEMPT_ID,
    EXTERNAL_PUBLICATION_FIELDS,
    FINAL_ATTEMPT_ID,
    MAX_MARKET_SECONDS,
    MAX_MARKET_REQUESTS_PER_STAGE,
    MAX_SEC_BYTES,
    MAX_SEC_REQUESTS,
    SCORED_TERMINAL_EVIDENCE_FIELDS,
    canonical_json_bytes,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_features import (
    FEATURE_ROW_SCHEMA_VERSION,
)
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
    SecGemmaOnlineRiskOverlaySourceVerificationError,
    verify_live_source_tree,
)


STORE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-store-v1"
)
JOURNAL_ENTRY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-journal-entry-v1"
)
ANCHOR_ENTRY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-anchor-entry-v1"
)
APPLICATION_ID: Final[int] = 0x53474F52
USER_VERSION: Final[int] = 2

STATE_RELATIVE_DIRECTORY: Final[Path] = (
    Path("data")
    / "sec_gemma_online_risk_overlay_v2_1"
    / CONTRACT_SHA256
)
ANCHOR_RELATIVE_DIRECTORY: Final[Path] = (
    Path("data") / "sec_gemma_online_risk_overlay_v2_1_anchors"
)
DATABASE_FILENAME: Final[str] = "state.sqlite3"
ANCHOR_FILENAME: Final[str] = f"{CONTRACT_SHA256}.anchor.jsonl"
LOCK_FILENAME: Final[str] = f"{CONTRACT_SHA256}.lock"
MAX_RECORD_PAYLOAD_BYTES: Final[int] = 8 * 1024 * 1024

ATTEMPTS_TABLE: Final[str] = "attempts"
RECORD_TABLES: Final[tuple[str, ...]] = REPORT_RECORD_TABLES
ALL_TYPED_TABLES: Final[tuple[str, ...]] = (
    ATTEMPTS_TABLE,
    *RECORD_TABLES,
)
_ALL_TABLES: Final[frozenset[str]] = frozenset(
    {"meta", "journal", *ALL_TYPED_TABLES}
)
_META_KEYS: Final[frozenset[str]] = frozenset(
    {
        "store_schema_version",
        "implementation_manifest",
        "store_instance_id",
        "journal_genesis_sha256",
    }
)
_IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_STORE_ID_RE = re.compile(r"[0-9a-f]{64}\Z")
_CAPABILITY_SENTINEL = object()
_ACTIVE_LOCK_PATHS: set[Path] = set()
_ACTIVE_LOCK_PATHS_GUARD = threading.Lock()
_PREDECESSOR_FEATURE_ATTEMPTS: Final[
    dict[str, tuple[tuple[str, str], ...]]
] = {
    "development": (),
    "confirmation": (("development", DEVELOPMENT_ATTEMPT_ID),),
    "final": (
        ("development", DEVELOPMENT_ATTEMPT_ID),
        ("confirmation", CONFIRMATION_ATTEMPT_ID),
    ),
}
_FEATURE_STAGE_WINDOWS: Final[dict[str, tuple[str, str]]] = {
    "development": ("2000-01-01", "2018-12-31"),
    "confirmation": ("2019-01-01", "2023-12-31"),
}
_ACQUISITION_RECOVERY_SCOPE: Final[
    dict[str, tuple[str, str]]
] = {
    DEVELOPMENT_ACQUISITION_ID: ("development", "development_acquisition"),
    CONFIRMATION_ATTEMPT_ID: ("confirmation", "confirmation"),
    FINAL_ATTEMPT_ID: ("final", "final"),
}
_ACQUISITION_VALIDATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-acquisition-validation-v1"
)
_ACQUISITION_VALIDATION_VERIFIER_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-acquisition-validator-v1"
)
_ACQUISITION_PUBLIC_SUMMARY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-public-summary-v1"
)
_ACQUISITION_REQUEST_ACCOUNTING_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-request-accounting-v1"
)
_PHASE_OUTPUT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-phase-output-v1"
)
_ACQUISITION_PHASE_EVIDENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "command",
        "phase",
        "phase_output_sha256",
        "payload_sha256",
        "phase_receipt",
        "verified_acquisition_report",
        "public_summary",
        "request_accounting",
    }
)
_ACQUISITION_PUBLIC_SUMMARY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "stage",
        "attempt_id",
        "attempt_kind",
        "acquisition_plan_sha256",
        "predecessor_bundle_sha256",
        "bundle_sha256",
        "manifest_sha256",
        "private_index_sha256",
        "sec_catalog_source_count",
        "sec_primary_document_count",
        "market_response_count",
        "model_request_count",
        "model_slice_sha256",
        "stage_slice_sha256",
        "market_elapsed_seconds_hex",
        "total_raw_byte_count",
        "complete_batch",
        "quarantine_only",
        "production_authority",
        "public_summary_sha256",
    }
)
_ACQUISITION_REQUEST_ACCOUNTING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "stage",
        "attempt_id",
        "sec_request_count",
        "market_request_count",
        "sec_bytes",
        "market_bytes",
        "network_request_count",
        "retry_count",
        "redirect_count",
        "market_elapsed_seconds_hex",
        "accounting_sha256",
    }
)
_PHASE_RECEIPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "phase",
        "phase_output_sha256",
        "payload_sha256",
        "counters",
        "elapsed_seconds_hex",
        "market_elapsed_seconds_hex",
        "deadline_monotonic_hex",
        "phase_receipt_sha256",
    }
)
_PHASE_COUNTER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "sec_request_count",
        "market_request_count",
        "model_call_count",
        "retry_count",
        "fallback_count",
        "model_pull_count",
        "paid_api_call_count",
    }
)

_ACQUISITION_TABLE_EFFECTS: Final[dict[str, frozenset[str]]] = {
    "evidence": frozenset(
        {
            "official_sec_network",
            "market_network",
            "deterministic_private_quarantine",
        }
    ),
    "artifacts": frozenset({"deterministic_private_quarantine"}),
}
_SCORING_TABLE_EFFECTS: Final[dict[str, frozenset[str]]] = {
    "evidence": frozenset(
        {
            "official_sec_network",
            "market_network",
            "ollama_runtime_identity",
            "gemma_batch",
            "canonical_market_value_read",
            "chronological_replay",
            "joint_report_seal",
        }
    ),
    "features": frozenset(
        {
            "gemma_batch",
            "canonical_market_value_read",
            "chronological_replay",
        }
    ),
    "predictions": frozenset({"chronological_replay"}),
    "lessons": frozenset({"chronological_replay"}),
    "ledgers": frozenset({"chronological_replay"}),
    "artifacts": frozenset({"joint_report_seal"}),
}
_ANCHOR_EVENTS: Final[frozenset[str]] = frozenset(
    {
        "store_created",
        "checkpoint",
        "consume_intent",
        "consume_committed",
        "terminal_intent",
        "terminal_committed",
        "recovered_checkpoint",
        "recovered_indeterminate",
    }
)


class SecGemmaOnlineRiskOverlayStoreError(RuntimeError):
    """The state path, source binding, journal, or requested write is invalid."""


class SecGemmaOnlineRiskOverlayStoreConflict(
    SecGemmaOnlineRiskOverlayStoreError
):
    """An immutable identity, attempt, lock, or anchor conflicts."""


@dataclass(frozen=True, slots=True)
class StoreRecordReceipt:
    table: str
    identity: str
    attempt_id: str
    payload_sha256: str
    journal_sequence: int
    journal_entry_sha256: str


class EffectCapability:
    """Opaque authority for one consumed attempt in one live store instance."""

    __slots__ = (
        "_attempt_id",
        "_allowed_effects",
        "_receipt",
        "_store_instance_id",
        "_store_nonce",
        "_transition_sha256",
        "_sentinel",
    )

    def __init__(
        self,
        *,
        attempt_id: str,
        allowed_effects: tuple[str, ...],
        receipt: StoreRecordReceipt,
        store_instance_id: str,
        store_nonce: str,
        transition_sha256: str,
        _sentinel: object,
    ) -> None:
        if _sentinel is not _CAPABILITY_SENTINEL:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Effect capabilities can only be issued by a consumed store"
            )
        self._attempt_id = attempt_id
        self._allowed_effects = allowed_effects
        self._receipt = receipt
        self._store_instance_id = store_instance_id
        self._store_nonce = store_nonce
        self._transition_sha256 = transition_sha256
        self._sentinel = _sentinel

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    @property
    def allowed_effects(self) -> tuple[str, ...]:
        return self._allowed_effects

    @property
    def consumption_receipt(self) -> StoreRecordReceipt:
        return self._receipt

    def __repr__(self) -> str:
        return f"EffectCapability(attempt_id={self._attempt_id!r})"


def _sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} must be a lowercase SHA-256"
        )
    return value


def _identity(value: Any, location: str) -> str:
    if type(value) is not str or _IDENTITY_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} must be a canonical bounded identity"
        )
    return value


def _detach_json(value: Any, location: str) -> Any:
    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str or key in result:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    f"{location} must have unique built-in string keys"
                )
            result[key] = _detach_json(child, f"{location}.{key}")
        return result
    if type(value) is list:
        return [
            _detach_json(child, f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    if value is None or type(value) in {str, bool, int, float}:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise SecGemmaOnlineRiskOverlayStoreError(
                f"{location} contains non-finite JSON"
            ) from exc
        return value
    raise SecGemmaOnlineRiskOverlayStoreError(
        f"{location} must contain only exact built-in JSON"
    )


def _canonical_payload(value: Any, location: str) -> tuple[Any, bytes, str]:
    detached = _detach_json(value, location)
    payload = canonical_json_bytes(detached)
    return detached, payload, hashlib.sha256(payload).hexdigest()


def _reject_duplicate_pairs(
    location: str,
):
    def build(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    f"{location} contains a duplicate JSON key"
                )
            result[key] = value
        return result

    return build


def _strict_json_bytes(payload: bytes, location: str) -> Any:
    if type(payload) is not bytes:
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} must be exact bytes"
        )
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs(location),
            parse_constant=lambda token: (_ for _ in ()).throw(
                SecGemmaOnlineRiskOverlayStoreError(
                    f"{location} contains non-finite JSON token {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} is not strict UTF-8 JSON"
        ) from exc
    detached, canonical, _ = _canonical_payload(value, location)
    if canonical != payload:
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} is not canonical JSON"
        )
    return detached


def _validated_predecessor_feature_row(
    value: Any,
    *,
    expected_stage: str,
    location: str,
) -> tuple[dict[str, Any], tuple[str, str, str]]:
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} must be one exact mapping"
        )
    row = _detach_json(value, location)
    feature_hash = row.get("feature_row_sha256")
    decision_session = row.get("decision_session")
    acceptance = row.get("acceptance_datetime")
    accession = row.get("accession_number")
    form = row.get("form")
    bindings = row.get("upstream_bindings")
    bindings_hash = row.get("upstream_bindings_sha256")
    first, last = _FEATURE_STAGE_WINDOWS[expected_stage]
    if (
        row.get("schema_version") != FEATURE_ROW_SCHEMA_VERSION
        or row.get("contract_version") != CONTRACT_VERSION
        or row.get("contract_sha256") != CONTRACT_SHA256
        or row.get("artifact_stage") != expected_stage
        or type(feature_hash) is not str
        or _SHA256_RE.fullmatch(feature_hash) is None
        or type(decision_session) is not str
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", decision_session) is None
        or not first <= decision_session <= last
        or type(accession) is not str
        or not accession
        or len(accession) > 64
        or form not in {"10-K", "10-Q"}
        or (
            acceptance is not None
            and (
                type(acceptance) is not str
                or len(acceptance) != 14
                or not acceptance.isascii()
                or not acceptance.isdigit()
            )
        )
        or type(bindings) is not dict
        or type(bindings_hash) is not str
        or _SHA256_RE.fullmatch(bindings_hash) is None
        or bindings_hash != canonical_sha256(bindings)
    ):
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} is not one exact stage-bound feature row"
        )
    body = {
        key: child
        for key, child in row.items()
        if key != "feature_row_sha256"
    }
    if feature_hash != canonical_sha256(body):
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} self-hash changed"
        )
    return row, (decision_session, acceptance or "", accession)


def _validated_self_hashed_mapping(
    value: Any,
    *,
    expected_fields: frozenset[str] | set[str],
    hash_field: str,
    location: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} must be one exact mapping"
        )
    observed = _detach_json(value, location)
    digest = observed.get(hash_field)
    if (
        set(observed) != set(expected_fields)
        or type(digest) is not str
        or _SHA256_RE.fullmatch(digest) is None
        or digest
        != canonical_sha256(
            {
                key: child
                for key, child in observed.items()
                if key != hash_field
            }
        )
    ):
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} schema or self-hash changed"
        )
    return observed


def _validated_float_hex(value: Any, location: str) -> float:
    if type(value) is not str:
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} must be canonical hexadecimal float text"
        )
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} must be canonical hexadecimal float text"
        ) from exc
    if (
        not math.isfinite(parsed)
        or parsed < 0.0
        or parsed.hex() != value
    ):
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} must be one nonnegative finite canonical float"
        )
    return parsed


def _validated_acquisition_phase_material(
    value: Any,
    *,
    expected_stage: str,
    expected_attempt_id: str,
    expected_attempt_kind: str,
    expected_command: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayStoreError(
            "Acquisition phase evidence must be one exact mapping"
        )
    observed = _detach_json(value, "acquisition phase evidence")
    if (
        set(observed) != _ACQUISITION_PHASE_EVIDENCE_FIELDS
        or observed.get("command") != expected_command
        or observed.get("phase") != "acquisition"
    ):
        raise SecGemmaOnlineRiskOverlayStoreError(
            "Acquisition phase evidence identity changed"
        )

    report = _validated_self_hashed_mapping(
        observed["verified_acquisition_report"],
        expected_fields=set(ACQUISITION_VALIDATION_FIELDS),
        hash_field="validation_sha256",
        location="verified acquisition report",
    )
    checks = report.get("checks")
    predecessors = report.get("predecessor_chain_bundle_sha256s")
    expected_predecessor_count = {
        "development": 0,
        "confirmation": 1,
        "final": 2,
    }[expected_stage]
    if (
        report.get("schema_version")
        != _ACQUISITION_VALIDATION_SCHEMA_VERSION
        or report.get("verifier_id")
        != _ACQUISITION_VALIDATION_VERIFIER_ID
        or report.get("verdict") != "pass"
        or report.get("stage") != expected_stage
        or report.get("attempt_id") != expected_attempt_id
        or report.get("attempt_kind") != expected_attempt_kind
        or type(checks) is not dict
        or set(checks) != set(ACQUISITION_VALIDATION_CHECKS)
        or any(
            type(digest) is not str
            or _SHA256_RE.fullmatch(digest) is None
            for digest in checks.values()
        )
        or report.get("check_set_sha256") != canonical_sha256(checks)
        or type(predecessors) is not list
        or len(predecessors) != expected_predecessor_count
        or len(set(predecessors)) != len(predecessors)
        or any(
            type(digest) is not str
            or _SHA256_RE.fullmatch(digest) is None
            for digest in predecessors
        )
    ):
        raise SecGemmaOnlineRiskOverlayStoreError(
            "Verified acquisition report is not exact stage-bound evidence"
        )
    for field in (
        "acquisition_plan_sha256",
        "bundle_sha256",
        "manifest_sha256",
        "private_index_sha256",
        "check_set_sha256",
    ):
        _sha256(report.get(field), f"verified acquisition {field}")

    summary = _validated_self_hashed_mapping(
        observed["public_summary"],
        expected_fields=_ACQUISITION_PUBLIC_SUMMARY_FIELDS,
        hash_field="public_summary_sha256",
        location="acquisition public summary",
    )
    predecessor_bundle = (
        predecessors[-1] if predecessors else None
    )
    count_fields = (
        "sec_catalog_source_count",
        "sec_primary_document_count",
        "market_response_count",
        "model_request_count",
        "total_raw_byte_count",
    )
    if (
        summary.get("schema_version")
        != _ACQUISITION_PUBLIC_SUMMARY_SCHEMA_VERSION
        or summary.get("stage") != expected_stage
        or summary.get("attempt_id") != expected_attempt_id
        or summary.get("attempt_kind") != expected_attempt_kind
        or summary.get("acquisition_plan_sha256")
        != report["acquisition_plan_sha256"]
        or summary.get("predecessor_bundle_sha256")
        != predecessor_bundle
        or summary.get("bundle_sha256") != report["bundle_sha256"]
        or summary.get("manifest_sha256") != report["manifest_sha256"]
        or summary.get("private_index_sha256")
        != report["private_index_sha256"]
        or any(
            type(summary.get(field)) is not int
            or summary[field] < 0
            for field in count_fields
        )
        or summary["model_request_count"] < 1
        or summary.get("complete_batch") is not True
        or summary.get("quarantine_only") is not True
        or summary.get("production_authority") is not True
    ):
        raise SecGemmaOnlineRiskOverlayStoreError(
            "Acquisition public summary is incomplete or crossed"
        )
    _sha256(
        summary.get("model_slice_sha256"),
        "acquisition public summary model slice",
    )
    _sha256(
        summary.get("stage_slice_sha256"),
        "acquisition public summary stage slice",
    )

    accounting = _validated_self_hashed_mapping(
        observed["request_accounting"],
        expected_fields=_ACQUISITION_REQUEST_ACCOUNTING_FIELDS,
        hash_field="accounting_sha256",
        location="acquisition request accounting",
    )
    integer_fields = (
        "sec_request_count",
        "market_request_count",
        "sec_bytes",
        "market_bytes",
        "network_request_count",
        "retry_count",
        "redirect_count",
    )
    if (
        accounting.get("schema_version")
        != _ACQUISITION_REQUEST_ACCOUNTING_SCHEMA_VERSION
        or accounting.get("stage") != expected_stage
        or accounting.get("attempt_id") != expected_attempt_id
        or any(
            type(accounting.get(field)) is not int
            or accounting[field] < 0
            for field in integer_fields
        )
        or not 1
        <= accounting["sec_request_count"]
        <= MAX_SEC_REQUESTS
        or accounting["market_request_count"]
        != MAX_MARKET_REQUESTS_PER_STAGE
        or accounting["sec_bytes"] > MAX_SEC_BYTES
        or accounting["network_request_count"]
        != accounting["sec_request_count"]
        + accounting["market_request_count"]
        or accounting["retry_count"] != 0
        or accounting["redirect_count"] != 0
        or summary["market_response_count"]
        != accounting["market_request_count"]
        or summary.get("market_elapsed_seconds_hex")
        != accounting.get("market_elapsed_seconds_hex")
        or summary["total_raw_byte_count"]
        != accounting["sec_bytes"] + accounting["market_bytes"]
    ):
        raise SecGemmaOnlineRiskOverlayStoreError(
            "Acquisition request accounting is incomplete or crossed"
        )
    market_elapsed = _validated_float_hex(
        accounting.get("market_elapsed_seconds_hex"),
        "acquisition market elapsed seconds",
    )
    if market_elapsed > float(MAX_MARKET_SECONDS):
        raise SecGemmaOnlineRiskOverlayStoreError(
            "Acquisition market elapsed time exceeds its frozen cap"
        )

    safe_payload = {
        "verified_acquisition_report": report,
        "public_summary": summary,
        "request_accounting": accounting,
    }
    payload_hash = canonical_sha256(safe_payload)
    if (
        observed.get("payload_sha256") != payload_hash
        or type(observed.get("phase_output_sha256")) is not str
        or _SHA256_RE.fullmatch(observed["phase_output_sha256"]) is None
    ):
        raise SecGemmaOnlineRiskOverlayStoreError(
            "Acquisition phase payload hash changed"
        )

    receipt = _validated_self_hashed_mapping(
        observed["phase_receipt"],
        expected_fields=_PHASE_RECEIPT_FIELDS,
        hash_field="phase_receipt_sha256",
        location="acquisition phase receipt",
    )
    counters = receipt.get("counters")
    if (
        receipt.get("phase") != "acquisition"
        or receipt.get("phase_output_sha256")
        != observed["phase_output_sha256"]
        or receipt.get("payload_sha256") != payload_hash
        or receipt.get("market_elapsed_seconds_hex")
        != accounting["market_elapsed_seconds_hex"]
        or type(counters) is not dict
        or set(counters) != _PHASE_COUNTER_KEYS
        or any(
            type(count) is not int or count < 0
            for count in counters.values()
        )
        or counters["sec_request_count"]
        != accounting["sec_request_count"]
        or counters["market_request_count"]
        != accounting["market_request_count"]
        or counters["model_call_count"] != 0
        or any(
            counters[field] != 0
            for field in (
                "retry_count",
                "fallback_count",
                "model_pull_count",
                "paid_api_call_count",
            )
        )
    ):
        raise SecGemmaOnlineRiskOverlayStoreError(
            "Acquisition phase receipt counters changed"
        )
    _validated_float_hex(
        receipt.get("elapsed_seconds_hex"),
        "acquisition elapsed seconds",
    )
    deadline = _validated_float_hex(
        receipt.get("deadline_monotonic_hex"),
        "acquisition deadline",
    )
    if deadline <= 0.0:
        raise SecGemmaOnlineRiskOverlayStoreError(
            "Acquisition deadline must be positive"
        )
    phase_body = {
        "schema_version": _PHASE_OUTPUT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "command": expected_command,
        "phase": "acquisition",
        "counters": counters,
        "payload": safe_payload,
        "payload_sha256": payload_hash,
    }
    if observed["phase_output_sha256"] != canonical_sha256(phase_body):
        raise SecGemmaOnlineRiskOverlayStoreError(
            "Acquisition phase output commitment changed"
        )
    return safe_payload, observed


def _is_reparse(details: os.stat_result) -> bool:
    attributes = getattr(details, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def _real_directory(path: Path, location: str) -> None:
    try:
        details = path.lstat()
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or path.resolve(strict=True) != path
    ):
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} must be one canonical real directory"
        )


def _real_file(path: Path, location: str) -> None:
    try:
        details = path.lstat()
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} is unavailable"
        ) from exc
    if (
        not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or details.st_nlink != 1
        or path.resolve(strict=True) != path
    ):
        raise SecGemmaOnlineRiskOverlayStoreError(
            f"{location} must be one canonical non-hard-linked file"
        )


def _repository_root(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise SecGemmaOnlineRiskOverlayStoreError(
            "Repository root must be an absolute pathlib.Path"
        )
    _real_directory(value, "Repository root")
    _real_directory(value / ".git", "Repository .git directory")
    return value


def _ensure_real_directory_path(repo_root: Path, relative: Path) -> Path:
    current = repo_root
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Contract state directory cannot be created"
            ) from exc
        _real_directory(current, "Contract state directory")
    return current


class _LifetimeFileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any = None

    def acquire(self) -> None:
        with _ACTIVE_LOCK_PATHS_GUARD:
            if self.path in _ACTIVE_LOCK_PATHS:
                raise SecGemmaOnlineRiskOverlayStoreConflict(
                    "Contract state is already open in this process"
                )
            _ACTIVE_LOCK_PATHS.add(self.path)
        try:
            handle = open(self.path, "a+b", buffering=0)
            if self.path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            _real_file(self.path, "State lock file")
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._handle = handle
        except Exception as exc:
            with _ACTIVE_LOCK_PATHS_GUARD:
                _ACTIVE_LOCK_PATHS.discard(self.path)
            try:
                handle.close()
            except (OSError, UnboundLocalError):
                pass
            if isinstance(exc, SecGemmaOnlineRiskOverlayStoreError):
                raise
            raise SecGemmaOnlineRiskOverlayStoreConflict(
                "Contract state is locked by another process"
            ) from exc

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None
            with _ACTIVE_LOCK_PATHS_GUARD:
                _ACTIVE_LOCK_PATHS.discard(self.path)


def _meta_sql() -> str:
    return """
        CREATE TABLE meta (
            key TEXT PRIMARY KEY CHECK (
                key IN (
                    'store_schema_version',
                    'implementation_manifest',
                    'store_instance_id',
                    'journal_genesis_sha256'
                )
            ),
            value_json BLOB NOT NULL,
            value_sha256 TEXT NOT NULL
        ) STRICT
    """


def _journal_sql() -> str:
    allowed_tables = ",".join(repr(value) for value in ALL_TYPED_TABLES)
    return f"""
        CREATE TABLE journal (
            sequence INTEGER PRIMARY KEY,
            record_table TEXT NOT NULL CHECK (
                record_table IN ({allowed_tables})
            ),
            identity TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            previous_entry_sha256 TEXT NOT NULL,
            entry_sha256 TEXT NOT NULL UNIQUE,
            UNIQUE (record_table, identity)
        ) STRICT
    """


def _record_table_sql(table: str) -> str:
    return f"""
        CREATE TABLE {table} (
            identity TEXT PRIMARY KEY,
            attempt_id TEXT NOT NULL,
            payload_json BLOB NOT NULL,
            payload_sha256 TEXT NOT NULL,
            journal_sequence INTEGER NOT NULL UNIQUE,
            journal_entry_sha256 TEXT NOT NULL UNIQUE,
            FOREIGN KEY (journal_sequence) REFERENCES journal(sequence)
        ) STRICT
    """


def _attempts_sql() -> str:
    statuses = ",".join(
        repr(value)
        for value in (PLANNED, CONSUMED, *sorted(TERMINAL_STATUSES))
    )
    return f"""
        CREATE TABLE attempts (
            identity TEXT PRIMARY KEY,
            attempt_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ({statuses})),
            attempt_plan_json BLOB NOT NULL,
            attempt_plan_sha256 TEXT NOT NULL,
            payload_json BLOB NOT NULL,
            payload_sha256 TEXT NOT NULL,
            journal_sequence INTEGER NOT NULL UNIQUE,
            journal_entry_sha256 TEXT NOT NULL UNIQUE,
            UNIQUE (attempt_id, status),
            FOREIGN KEY (journal_sequence) REFERENCES journal(sequence)
        ) STRICT
    """


def _immutable_trigger_sql(table: str, operation: str) -> str:
    trigger = f"{table}_forbid_{operation.lower()}"
    return f"""
        CREATE TRIGGER {trigger}
        BEFORE {operation} ON {table}
        BEGIN
            SELECT RAISE(ABORT, '{table} is append-only');
        END
    """


def _normalized_schema_sql(value: str) -> str:
    return " ".join(value.split())


def _journal_genesis_sha256(
    implementation_manifest: Mapping[str, Any],
    store_instance_id: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": STORE_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "implementation_manifest_sha256": implementation_manifest[
                "implementation_manifest_sha256"
            ],
            "store_instance_id": store_instance_id,
        }
    )


def _anchor_genesis_sha256(
    implementation_manifest: Mapping[str, Any],
    store_instance_id: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": ANCHOR_ENTRY_SCHEMA_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "implementation_manifest_sha256": implementation_manifest[
                "implementation_manifest_sha256"
            ],
            "store_instance_id": store_instance_id,
        }
    )


class SecGemmaOnlineRiskOverlayStore:
    """One canonical, exclusively locked store for one implementation."""

    def __init__(
        self,
        repo_root: Path,
        *,
        implementation_manifest: Mapping[str, Any],
    ) -> None:
        self._repo_root = _repository_root(repo_root)
        self._state_directory = _ensure_real_directory_path(
            self._repo_root, STATE_RELATIVE_DIRECTORY
        )
        self._anchor_directory = _ensure_real_directory_path(
            self._repo_root, ANCHOR_RELATIVE_DIRECTORY
        )
        self._path = self._state_directory / DATABASE_FILENAME
        self._anchor_path = self._anchor_directory / ANCHOR_FILENAME
        self._lock_path = self._anchor_directory / LOCK_FILENAME
        self._lock = _LifetimeFileLock(self._lock_path)
        self._lock.acquire()
        self._connection: sqlite3.Connection | None = None
        self._active_capabilities: dict[str, EffectCapability] = {}
        self._store_nonce = secrets.token_hex(32)
        self._poisoned = False
        try:
            self._implementation_manifest = validate_implementation_manifest(
                implementation_manifest
            )
            database_exists = self._path.exists()
            anchor_exists = self._anchor_path.exists()
            if database_exists != anchor_exists:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Database and external anchor must exist as one pair"
                )
            if database_exists:
                _real_file(self._path, "State database")
                _real_file(self._anchor_path, "State anchor")
            self._connection = sqlite3.connect(
                os.fspath(self._path),
                timeout=5.0,
                isolation_level=None,
            )
            self._connection.row_factory = sqlite3.Row
            self._configure_connection()
            if not database_exists:
                self._create_schema()
                self._create_anchor()
            self._verify_schema()
            self._verify_meta()
            database = self._verify_database_chain()
            anchor = self._verify_anchor_chain()
            self._reconcile_reopen(database, anchor)
            self.verify_chain()
        except Exception:
            self._close_resources(recover=False)
            raise

    def __enter__(self) -> "SecGemmaOnlineRiskOverlayStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def anchor_path(self) -> Path:
        return self._anchor_path

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    def _require_open(self) -> sqlite3.Connection:
        if self._connection is None:
            raise SecGemmaOnlineRiskOverlayStoreError("Store is closed")
        if self._poisoned:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Store is poisoned after an incomplete anchored mutation"
            )
        return self._connection

    def _close_resources(self, *, recover: bool) -> None:
        connection = self._connection
        pending_error: BaseException | None = None
        try:
            if recover and connection is not None and not self._poisoned:
                for capability in list(self._active_capabilities.values()):
                    self.finish_attempt(
                        capability,
                        terminal_status=TERMINAL_INDETERMINATE,
                    )
        except BaseException as exc:
            pending_error = exc
        finally:
            self._active_capabilities.clear()
            if connection is not None:
                try:
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    pass
                try:
                    connection.close()
                except BaseException as exc:
                    if pending_error is None:
                        pending_error = exc
                finally:
                    self._connection = None
                try:
                    self._validate_sidecars()
                except BaseException as exc:
                    if pending_error is None:
                        pending_error = exc
            try:
                self._lock.release()
            except BaseException as exc:
                if pending_error is None:
                    pending_error = exc
        if pending_error is not None:
            raise pending_error

    def close(self) -> None:
        self._close_resources(recover=True)

    def _configure_connection(self) -> None:
        connection = self._require_open()
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA recursive_triggers = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA synchronous = FULL")
        mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(mode).casefold() != "wal":
            raise SecGemmaOnlineRiskOverlayStoreError(
                "SQLite WAL mode could not be enabled"
            )
        locking = connection.execute(
            "PRAGMA locking_mode = EXCLUSIVE"
        ).fetchone()[0]
        if str(locking).casefold() != "exclusive":
            raise SecGemmaOnlineRiskOverlayStoreError(
                "SQLite exclusive locking could not be enabled"
            )
        self._validate_sidecars()

    def _validate_sidecars(self) -> None:
        for suffix in ("-wal", "-shm"):
            path = Path(f"{self._path}{suffix}")
            if path.exists():
                _real_file(path, f"SQLite {suffix} sidecar")
                if path.parent != self._state_directory:
                    raise SecGemmaOnlineRiskOverlayStoreError(
                        "SQLite sidecar escaped the contract state directory"
                    )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._require_open()
        began = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            began = True
            yield connection
            connection.execute("COMMIT")
            began = False
        except Exception:
            if began and connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def _create_schema(self) -> None:
        manifest = self._implementation_manifest
        store_id = secrets.token_hex(32)
        self._store_instance_id = store_id
        genesis = _journal_genesis_sha256(manifest, store_id)
        with self._transaction() as connection:
            connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {USER_VERSION}")
            connection.execute(_meta_sql())
            connection.execute(_journal_sql())
            connection.execute(_attempts_sql())
            for table in RECORD_TABLES:
                connection.execute(_record_table_sql(table))
            for table in ("meta", "journal", *ALL_TYPED_TABLES):
                connection.execute(_immutable_trigger_sql(table, "UPDATE"))
                connection.execute(_immutable_trigger_sql(table, "DELETE"))
            meta_values = {
                "store_schema_version": STORE_SCHEMA_VERSION,
                "implementation_manifest": manifest,
                "store_instance_id": store_id,
                "journal_genesis_sha256": genesis,
            }
            for key, value in meta_values.items():
                payload = canonical_json_bytes(value)
                connection.execute(
                    "INSERT INTO meta(key,value_json,value_sha256) VALUES(?,?,?)",
                    (key, payload, hashlib.sha256(payload).hexdigest()),
                )
        _real_file(self._path, "State database")

    def _verify_schema(self) -> None:
        connection = self._require_open()
        application_id = connection.execute(
            "PRAGMA application_id"
        ).fetchone()[0]
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if application_id != APPLICATION_ID or user_version != USER_VERSION:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "SQLite application or schema version changed"
            )
        expected_schema: dict[tuple[str, str], tuple[str, str]] = {
            ("table", "meta"): ("meta", _normalized_schema_sql(_meta_sql())),
            ("table", "journal"): (
                "journal",
                _normalized_schema_sql(_journal_sql()),
            ),
            ("table", ATTEMPTS_TABLE): (
                ATTEMPTS_TABLE,
                _normalized_schema_sql(_attempts_sql()),
            ),
        }
        expected_schema.update(
            {
                ("table", table): (
                    table,
                    _normalized_schema_sql(_record_table_sql(table)),
                )
                for table in RECORD_TABLES
            }
        )
        expected_schema.update(
            {
                ("trigger", f"{table}_forbid_{operation.lower()}"): (
                    table,
                    _normalized_schema_sql(
                        _immutable_trigger_sql(table, operation)
                    ),
                )
                for table in ("meta", "journal", *ALL_TYPED_TABLES)
                for operation in ("UPDATE", "DELETE")
            }
        )
        observed: dict[tuple[str, str], tuple[str, str]] = {}
        for row in connection.execute(
            """
            SELECT type,name,tbl_name,sql FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
              AND type IN ('table','trigger','view','index')
            """
        ):
            if type(row["sql"]) is not str:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Store schema contains a non-canonical SQL object"
                )
            observed[(row["type"], row["name"])] = (
                row["tbl_name"],
                _normalized_schema_sql(row["sql"]),
            )
        if observed != expected_schema:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Store SQL schema changed"
            )
        if {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table'"
            )
        } != set(_ALL_TABLES):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Store table inventory changed"
            )
        if [row[0] for row in connection.execute(
            "PRAGMA integrity_check"
        ).fetchall()] != ["ok"]:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "SQLite integrity check failed"
            )
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "SQLite foreign-key check failed"
            )

    def _meta_value(self, key: str) -> Any:
        row = self._require_open().execute(
            "SELECT value_json,value_sha256 FROM meta WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            raise SecGemmaOnlineRiskOverlayStoreError(
                f"Store metadata {key} is missing"
            )
        payload = bytes(row["value_json"])
        if hashlib.sha256(payload).hexdigest() != row["value_sha256"]:
            raise SecGemmaOnlineRiskOverlayStoreError(
                f"Store metadata {key} hash changed"
            )
        return _strict_json_bytes(payload, f"store metadata {key}")

    def _verify_meta(self) -> None:
        keys = {
            row[0] for row in self._require_open().execute("SELECT key FROM meta")
        }
        if keys != set(_META_KEYS):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Store metadata keys changed"
            )
        if self._meta_value("store_schema_version") != STORE_SCHEMA_VERSION:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Store schema metadata changed"
            )
        if self._meta_value(
            "implementation_manifest"
        ) != self._implementation_manifest:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Store is bound to another implementation manifest"
            )
        store_id = self._meta_value("store_instance_id")
        if type(store_id) is not str or _STORE_ID_RE.fullmatch(store_id) is None:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Store instance id is invalid"
            )
        self._store_instance_id = store_id
        genesis = _sha256(
            self._meta_value("journal_genesis_sha256"), "journal genesis"
        )
        if genesis != _journal_genesis_sha256(
            self._implementation_manifest, store_id
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Journal genesis differs from the store binding"
            )

    def _create_anchor(self) -> None:
        try:
            descriptor = os.open(
                self._anchor_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            os.close(descriptor)
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "External anchor cannot be created exactly once"
            ) from exc
        _real_file(self._anchor_path, "State anchor")
        self._append_anchor(
            "store_created",
            self._verify_database_chain(),
            attempt_id=None,
            event_payload={},
        )

    def _anchor_entries(self) -> list[dict[str, Any]]:
        _real_file(self._anchor_path, "State anchor")
        payload = self._anchor_path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "External anchor has a partial final record"
            )
        lines = payload.splitlines()
        if not lines:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "External anchor is empty"
            )
        previous = _anchor_genesis_sha256(
            self._implementation_manifest, self._store_instance_id
        )
        entries: list[dict[str, Any]] = []
        expected_keys = {
            "schema_version",
            "sequence",
            "event",
            "contract_sha256",
            "implementation_manifest_sha256",
            "store_instance_id",
            "database_journal_sequence",
            "database_journal_tip_sha256",
            "attempt_id",
            "event_payload",
            "previous_anchor_sha256",
            "anchor_entry_sha256",
        }
        for sequence, raw_line in enumerate(lines, start=1):
            value = _strict_json_bytes(
                bytes(raw_line), f"anchor entry {sequence}"
            )
            if type(value) is not dict or set(value) != expected_keys:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "External anchor entry shape changed"
                )
            digest = _sha256(
                value["anchor_entry_sha256"],
                "anchor entry sha256",
            )
            body = {
                key: value[key]
                for key in value
                if key != "anchor_entry_sha256"
            }
            if (
                value["schema_version"] != ANCHOR_ENTRY_SCHEMA_VERSION
                or value["sequence"] != sequence
                or value["event"] not in _ANCHOR_EVENTS
                or value["contract_sha256"] != CONTRACT_SHA256
                or value["implementation_manifest_sha256"]
                != self._implementation_manifest[
                    "implementation_manifest_sha256"
                ]
                or value["store_instance_id"] != self._store_instance_id
                or type(value["database_journal_sequence"]) is not int
                or value["database_journal_sequence"] < 0
                or value["previous_anchor_sha256"] != previous
                or canonical_sha256(body) != digest
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "External anchor chain is inconsistent"
                )
            _sha256(
                value["database_journal_tip_sha256"],
                "anchor database tip",
            )
            if value["attempt_id"] is not None and (
                type(value["attempt_id"]) is not str
                or value["attempt_id"] not in ATTEMPT_KIND_BY_ID
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "External anchor names an unknown attempt"
                )
            _detach_json(value["event_payload"], "anchor event payload")
            entries.append(value)
            previous = digest
        if entries[0]["event"] != "store_created":
            raise SecGemmaOnlineRiskOverlayStoreError(
                "External anchor lacks its creation record"
            )
        return entries

    def _verify_anchor_chain(self) -> dict[str, Any]:
        entries = self._anchor_entries()
        latest = entries[-1]
        return {
            "entry_count": len(entries),
            "tip_sha256": latest["anchor_entry_sha256"],
            "latest": latest,
        }

    def _append_anchor(
        self,
        event: str,
        database_snapshot: Mapping[str, Any],
        *,
        attempt_id: str | None,
        event_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if event not in _ANCHOR_EVENTS:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Unknown external anchor event"
            )
        if self._anchor_path.stat().st_size:
            entries = self._anchor_entries()
            sequence = len(entries) + 1
            previous = entries[-1]["anchor_entry_sha256"]
        else:
            sequence = 1
            previous = _anchor_genesis_sha256(
                self._implementation_manifest, self._store_instance_id
            )
        body = {
            "schema_version": ANCHOR_ENTRY_SCHEMA_VERSION,
            "sequence": sequence,
            "event": event,
            "contract_sha256": CONTRACT_SHA256,
            "implementation_manifest_sha256": self._implementation_manifest[
                "implementation_manifest_sha256"
            ],
            "store_instance_id": self._store_instance_id,
            "database_journal_sequence": database_snapshot[
                "journal_entry_count"
            ],
            "database_journal_tip_sha256": database_snapshot[
                "journal_tip_sha256"
            ],
            "attempt_id": attempt_id,
            "event_payload": _detach_json(
                dict(event_payload), "anchor event payload"
            ),
            "previous_anchor_sha256": previous,
        }
        entry = {**body, "anchor_entry_sha256": canonical_sha256(body)}
        serialized = canonical_json_bytes(entry) + b"\n"
        try:
            with open(self._anchor_path, "ab", buffering=0) as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            self._poisoned = True
            raise SecGemmaOnlineRiskOverlayStoreError(
                "External anchor append failed"
            ) from exc
        return entry

    def _attempt_rows(self, attempt_id: str) -> list[sqlite3.Row]:
        return list(
            self._require_open().execute(
                """
                SELECT * FROM attempts
                WHERE attempt_id=? ORDER BY journal_sequence
                """,
                (attempt_id,),
            )
        )

    def attempt_history(self, attempt_id: str) -> list[dict[str, Any]]:
        if attempt_id not in ATTEMPT_KIND_BY_ID:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Unknown preregistered attempt id"
            )
        return [
            _strict_json_bytes(
                bytes(row["payload_json"]),
                f"attempt transition {row['identity']}",
            )
            for row in self._attempt_rows(attempt_id)
        ]

    def _attempt_plan_from_rows(
        self, rows: Sequence[sqlite3.Row]
    ) -> dict[str, Any]:
        if not rows:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Attempt has no registered plan"
            )
        first = bytes(rows[0]["attempt_plan_json"])
        plan = _strict_json_bytes(first, "stored attempt plan")
        for row in rows:
            if (
                bytes(row["attempt_plan_json"]) != first
                or row["attempt_plan_sha256"] != plan["attempt_plan_sha256"]
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Attempt transitions do not share one immutable plan"
                )
        try:
            return validate_attempt_plan(
                plan,
                implementation_manifest=self._implementation_manifest,
            )
        except SecGemmaOnlineRiskOverlayAttemptError as exc:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Stored attempt plan is invalid"
            ) from exc

    def _current_status(self, attempt_id: str) -> str | None:
        rows = self._attempt_rows(attempt_id)
        return None if not rows else rows[-1]["status"]

    def _journal_tip_tx(
        self, connection: sqlite3.Connection
    ) -> tuple[int, str]:
        row = connection.execute(
            """
            SELECT sequence,entry_sha256
            FROM journal ORDER BY sequence DESC LIMIT 1
            """
        ).fetchone()
        if row is None:
            return 0, _sha256(
                self._meta_value("journal_genesis_sha256"),
                "journal genesis",
            )
        return int(row["sequence"]), _sha256(
            row["entry_sha256"], "journal tip"
        )

    def _append_journal_and_typed_tx(
        self,
        connection: sqlite3.Connection,
        *,
        table: str,
        identity: str,
        attempt_id: str,
        payload_bytes: bytes,
        payload_hash: str,
        status: str | None = None,
        attempt_plan_bytes: bytes | None = None,
        attempt_plan_hash: str | None = None,
    ) -> StoreRecordReceipt:
        last_sequence, prior_hash = self._journal_tip_tx(connection)
        sequence = last_sequence + 1
        entry_body = {
            "schema_version": JOURNAL_ENTRY_SCHEMA_VERSION,
            "sequence": sequence,
            "record_table": table,
            "identity": identity,
            "attempt_id": attempt_id,
            "payload_sha256": payload_hash,
            "previous_entry_sha256": prior_hash,
        }
        entry_hash = canonical_sha256(entry_body)
        connection.execute(
            """
            INSERT INTO journal(
                sequence,record_table,identity,attempt_id,payload_sha256,
                previous_entry_sha256,entry_sha256
            ) VALUES(?,?,?,?,?,?,?)
            """,
            (
                sequence,
                table,
                identity,
                attempt_id,
                payload_hash,
                prior_hash,
                entry_hash,
            ),
        )
        if table == ATTEMPTS_TABLE:
            if (
                status is None
                or attempt_plan_bytes is None
                or attempt_plan_hash is None
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Attempt append lacks its plan or status"
                )
            connection.execute(
                """
                INSERT INTO attempts(
                    identity,attempt_id,status,attempt_plan_json,
                    attempt_plan_sha256,payload_json,payload_sha256,
                    journal_sequence,journal_entry_sha256
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    identity,
                    attempt_id,
                    status,
                    attempt_plan_bytes,
                    attempt_plan_hash,
                    payload_bytes,
                    payload_hash,
                    sequence,
                    entry_hash,
                ),
            )
        else:
            connection.execute(
                f"""
                INSERT INTO {table}(
                    identity,attempt_id,payload_json,payload_sha256,
                    journal_sequence,journal_entry_sha256
                ) VALUES(?,?,?,?,?,?)
                """,
                (
                    identity,
                    attempt_id,
                    payload_bytes,
                    payload_hash,
                    sequence,
                    entry_hash,
                ),
            )
        return StoreRecordReceipt(
            table=table,
            identity=identity,
            attempt_id=attempt_id,
            payload_sha256=payload_hash,
            journal_sequence=sequence,
            journal_entry_sha256=entry_hash,
        )

    def _append_attempt_transition_tx(
        self,
        connection: sqlite3.Connection,
        *,
        plan: Mapping[str, Any],
        transition: Mapping[str, Any],
    ) -> StoreRecordReceipt:
        _, transition_bytes, transition_hash = _canonical_payload(
            transition, "attempt transition"
        )
        return self._append_journal_and_typed_tx(
            connection,
            table=ATTEMPTS_TABLE,
            identity=(
                f"{transition['attempt_id']}:"
                f"{transition['sequence_number']}"
            ),
            attempt_id=transition["attempt_id"],
            payload_bytes=transition_bytes,
            payload_hash=transition_hash,
            status=transition["status"],
            attempt_plan_bytes=canonical_json_bytes(plan),
            attempt_plan_hash=plan["attempt_plan_sha256"],
        )

    def register_attempt(
        self, attempt_plan: Mapping[str, Any]
    ) -> StoreRecordReceipt:
        try:
            plan = validate_attempt_plan(
                attempt_plan,
                implementation_manifest=self._implementation_manifest,
            )
        except SecGemmaOnlineRiskOverlayAttemptError as exc:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Attempt plan is invalid"
            ) from exc
        attempt_id = plan["attempt_id"]
        ordinal = ATTEMPT_ORDINAL_BY_ID[attempt_id]
        with self._transaction() as connection:
            rows = self._attempt_rows(attempt_id)
            if rows:
                stored_plan = self._attempt_plan_from_rows(rows)
                if len(rows) == 1 and rows[0]["status"] == PLANNED:
                    if stored_plan == plan:
                        return self._receipt_from_row(ATTEMPTS_TABLE, rows[0])
                raise SecGemmaOnlineRiskOverlayStoreConflict(
                    "Attempt was already started or has a different plan"
                )
            present = [
                fixed_id
                for fixed_id in ATTEMPT_IDS
                if self._attempt_rows(fixed_id)
            ]
            if present != list(ATTEMPT_IDS[: ordinal - 1]):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Attempt registration skipped or reordered the sequence"
                )
            predecessor_id = PREREQUISITE_ATTEMPT_BY_ID[attempt_id]
            if predecessor_id is not None:
                predecessor_rows = self._attempt_rows(predecessor_id)
                predecessor_plan = self._attempt_plan_from_rows(
                    predecessor_rows
                )
                predecessor_history = self.attempt_history(predecessor_id)
                if current_attempt_status(
                    attempt_plan=predecessor_plan,
                    implementation_manifest=self._implementation_manifest,
                    transitions=predecessor_history,
                ) != TERMINAL_PASS:
                    raise SecGemmaOnlineRiskOverlayStoreError(
                        "Later attempt requires predecessor terminal pass"
                    )
                if (
                    plan["prerequisite_terminal_transition_sha256"]
                    != predecessor_history[-1]["transition_sha256"]
                ):
                    raise SecGemmaOnlineRiskOverlayStoreError(
                        "Attempt plan lost its exact predecessor binding"
                    )
            transition = build_attempt_transition(
                attempt_plan=plan,
                implementation_manifest=self._implementation_manifest,
                status=PLANNED,
            )
            receipt = self._append_attempt_transition_tx(
                connection, plan=plan, transition=transition
            )
        self._append_anchor(
            "checkpoint",
            self._verify_database_chain(),
            attempt_id=attempt_id,
            event_payload={"status": PLANNED},
        )
        return receipt

    def _live_reverify_implementation(self) -> None:
        try:
            verified = verify_live_source_tree(self._repo_root)
            observed = build_implementation_manifest(
                verified_sources=verified
            )
        except (
            SecGemmaOnlineRiskOverlaySourceVerificationError,
            SecGemmaOnlineRiskOverlayAttemptError,
        ) as exc:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Live Git/source verification failed before consumption"
            ) from exc
        if observed != self._implementation_manifest:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Live Git/source evidence differs from the store manifest"
            )

    def consume_attempt(self, attempt_id: str) -> EffectCapability:
        if attempt_id not in ATTEMPT_KIND_BY_ID:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Unknown preregistered attempt id"
            )
        self._live_reverify_implementation()
        rows = self._attempt_rows(attempt_id)
        if not rows or rows[-1]["status"] != PLANNED:
            raise SecGemmaOnlineRiskOverlayStoreConflict(
                "Attempt is absent, consumed, terminal, or already retried"
            )
        plan = self._attempt_plan_from_rows(rows)
        history = self.attempt_history(attempt_id)
        transition = build_attempt_transition(
            attempt_plan=plan,
            implementation_manifest=self._implementation_manifest,
            status=CONSUMED,
            prior_transition=history[-1],
        )
        self._append_anchor(
            "consume_intent",
            self._verify_database_chain(),
            attempt_id=attempt_id,
            event_payload={
                "attempt_plan_sha256": plan["attempt_plan_sha256"],
                "transition_sha256": transition["transition_sha256"],
            },
        )
        try:
            with self._transaction() as connection:
                receipt = self._append_attempt_transition_tx(
                    connection, plan=plan, transition=transition
                )
            snapshot = self._verify_database_chain()
            self._append_anchor(
                "consume_committed",
                snapshot,
                attempt_id=attempt_id,
                event_payload={
                    "attempt_plan_sha256": plan["attempt_plan_sha256"],
                    "transition_sha256": transition["transition_sha256"],
                },
            )
        except Exception:
            self._poisoned = True
            raise
        capability = EffectCapability(
            attempt_id=attempt_id,
            allowed_effects=tuple(plan["allowed_effects"]),
            receipt=receipt,
            store_instance_id=self._store_instance_id,
            store_nonce=self._store_nonce,
            transition_sha256=transition["transition_sha256"],
            _sentinel=_CAPABILITY_SENTINEL,
        )
        self._active_capabilities[attempt_id] = capability
        return capability

    def _authorize_capability(
        self, capability: EffectCapability
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if (
            not isinstance(capability, EffectCapability)
            or getattr(capability, "_sentinel", None)
            is not _CAPABILITY_SENTINEL
            or capability._store_instance_id != self._store_instance_id
            or capability._store_nonce != self._store_nonce
            or self._active_capabilities.get(capability.attempt_id)
            is not capability
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Effect capability is forged, stale, foreign, or consumed"
            )
        rows = self._attempt_rows(capability.attempt_id)
        plan = self._attempt_plan_from_rows(rows)
        history = self.attempt_history(capability.attempt_id)
        if (
            rows[-1]["status"] != CONSUMED
            or history[-1]["transition_sha256"]
            != capability._transition_sha256
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Effect capability no longer names the consumed transition"
            )
        return plan, history

    def authorize_effect(
        self, capability: EffectCapability, effect: str
    ) -> None:
        plan, _ = self._authorize_capability(capability)
        if type(effect) is not str or effect not in plan["allowed_effects"]:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Effect is not permitted by this exact attempt plan"
            )

    def _table_effects(
        self, attempt_kind: str, table: str
    ) -> frozenset[str]:
        mapping = (
            _ACQUISITION_TABLE_EFFECTS
            if attempt_kind == DEVELOPMENT_ACQUISITION
            else _SCORING_TABLE_EFFECTS
        )
        allowed = mapping.get(table)
        if allowed is None:
            raise SecGemmaOnlineRiskOverlayStoreError(
                f"{table} is confidentially forbidden for this attempt stage"
            )
        return allowed

    def _append_record(
        self,
        table: str,
        *,
        capability: EffectCapability,
        effect: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> StoreRecordReceipt:
        plan, _ = self._authorize_capability(capability)
        self.authorize_effect(capability, effect)
        if effect not in self._table_effects(plan["attempt_kind"], table):
            raise SecGemmaOnlineRiskOverlayStoreError(
                f"{effect} cannot write the confidential {table} table"
            )
        if type(payload) is not dict:
            raise SecGemmaOnlineRiskOverlayStoreError(
                f"{table} payload must be one exact built-in mapping"
            )
        fixed_identity = _identity(identity, f"{table} identity")
        _, payload_bytes, payload_hash = _canonical_payload(
            payload, f"{table} payload"
        )
        if len(payload_bytes) > MAX_RECORD_PAYLOAD_BYTES:
            raise SecGemmaOnlineRiskOverlayStoreError(
                f"{table} payload exceeds the fixed local evidence limit"
            )
        with self._transaction() as connection:
            existing = connection.execute(
                f"SELECT * FROM {table} WHERE identity=?", (fixed_identity,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["attempt_id"] == capability.attempt_id
                    and bytes(existing["payload_json"]) == payload_bytes
                    and existing["payload_sha256"] == payload_hash
                ):
                    return self._receipt_from_row(table, existing)
                raise SecGemmaOnlineRiskOverlayStoreConflict(
                    f"{table} identity already binds different immutable bytes"
                )
            receipt = self._append_journal_and_typed_tx(
                connection,
                table=table,
                identity=fixed_identity,
                attempt_id=capability.attempt_id,
                payload_bytes=payload_bytes,
                payload_hash=payload_hash,
            )
        try:
            self._append_anchor(
                "checkpoint",
                self._verify_database_chain(),
                attempt_id=capability.attempt_id,
                event_payload={
                    "effect": effect,
                    "record_table": table,
                    "identity": fixed_identity,
                },
            )
        except Exception:
            self._poisoned = True
            raise
        return receipt

    def append_evidence(
        self,
        *,
        capability: EffectCapability,
        effect: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> StoreRecordReceipt:
        return self._append_record(
            "evidence",
            capability=capability,
            effect=effect,
            identity=identity,
            payload=payload,
        )

    def append_feature(
        self,
        *,
        capability: EffectCapability,
        effect: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> StoreRecordReceipt:
        return self._append_record(
            "features",
            capability=capability,
            effect=effect,
            identity=identity,
            payload=payload,
        )

    def append_prediction(
        self,
        *,
        capability: EffectCapability,
        effect: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> StoreRecordReceipt:
        return self._append_record(
            "predictions",
            capability=capability,
            effect=effect,
            identity=identity,
            payload=payload,
        )

    def append_lesson(
        self,
        *,
        capability: EffectCapability,
        effect: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> StoreRecordReceipt:
        return self._append_record(
            "lessons",
            capability=capability,
            effect=effect,
            identity=identity,
            payload=payload,
        )

    def append_ledger(
        self,
        *,
        capability: EffectCapability,
        effect: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> StoreRecordReceipt:
        return self._append_record(
            "ledgers",
            capability=capability,
            effect=effect,
            identity=identity,
            payload=payload,
        )

    def append_artifact(
        self,
        *,
        capability: EffectCapability,
        effect: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> StoreRecordReceipt:
        return self._append_record(
            "artifacts",
            capability=capability,
            effect=effect,
            identity=identity,
            payload=payload,
        )

    def _record_commitment(self, attempt_id: str) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        connection = self._require_open()
        for table in RECORD_TABLES:
            rows = list(
                connection.execute(
                    f"""
                    SELECT identity,payload_sha256,journal_sequence,
                           journal_entry_sha256
                    FROM {table} WHERE attempt_id=?
                    ORDER BY journal_sequence
                    """,
                    (attempt_id,),
                )
            )
            counts[table] = len(rows)
            records.extend(
                {
                    "table": table,
                    "identity": row["identity"],
                    "payload_sha256": row["payload_sha256"],
                    "journal_sequence": row["journal_sequence"],
                    "journal_entry_sha256": row[
                        "journal_entry_sha256"
                    ],
                }
                for row in rows
            )
        records.sort(key=lambda item: item["journal_sequence"])
        return {
            "record_counts": counts,
            "record_commitment_sha256": canonical_sha256(records),
        }

    def terminal_evidence_material(
        self, capability: EffectCapability
    ) -> dict[str, Any]:
        plan, _ = self._authorize_capability(capability)
        commitment = self._record_commitment(capability.attempt_id)
        return {
            "attempt_id": capability.attempt_id,
            "attempt_plan_sha256": plan["attempt_plan_sha256"],
            **commitment,
        }

    def _artifact_payload_for_receipt(
        self,
        receipt: StoreRecordReceipt,
    ) -> dict[str, Any]:
        if receipt.table != "artifacts":
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Terminal evidence receipt must name the artifacts table"
            )
        actual = self.record_receipt(receipt.table, receipt.identity)
        if actual != receipt:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Terminal evidence receipt differs from the current store record"
            )
        row = self._require_open().execute(
            """
            SELECT payload_json,payload_sha256
            FROM artifacts WHERE identity=?
            """,
            (receipt.identity,),
        ).fetchone()
        if row is None:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Terminal evidence artifact does not exist"
            )
        payload_bytes = bytes(row["payload_json"])
        if hashlib.sha256(payload_bytes).hexdigest() != row["payload_sha256"]:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Terminal evidence artifact payload hash changed"
            )
        payload = _strict_json_bytes(
            payload_bytes,
            f"terminal evidence artifact {receipt.identity}",
        )
        if type(payload) is not dict:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Terminal evidence artifact must be one exact mapping"
            )
        return payload

    def _validate_terminal_evidence(
        self,
        capability: EffectCapability,
        plan: Mapping[str, Any],
        *,
        terminal_status: str,
        verified_terminal_evidence: (
            VerifiedAcquisitionTerminalEvidence
            | VerifiedScoredTerminalEvidence
            | None
        ),
    ) -> dict[str, Any] | None:
        if terminal_status == TERMINAL_INDETERMINATE:
            if verified_terminal_evidence is not None:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Terminal-indeterminate cannot bind semantic evidence"
                )
            return None
        if plan["attempt_kind"] == DEVELOPMENT_ACQUISITION:
            if terminal_status != TERMINAL_PASS:
                if verified_terminal_evidence is not None:
                    raise SecGemmaOnlineRiskOverlayStoreError(
                        "Failed acquisition cannot release terminal evidence"
                    )
                return None
            if not is_verified_acquisition_terminal_evidence(
                verified_terminal_evidence
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Acquisition pass requires opaque verifier-issued terminal evidence"
                )
            assert (
                type(verified_terminal_evidence)
                is VerifiedAcquisitionTerminalEvidence
            )
            validator = validate_acquisition_terminal_evidence
        else:
            if terminal_status == TERMINAL_PASS:
                if not is_verified_scored_terminal_evidence(
                    verified_terminal_evidence
                ):
                    raise SecGemmaOnlineRiskOverlayStoreError(
                        "Scored pass requires opaque verifier-issued terminal evidence"
                    )
            elif terminal_status == TERMINAL_FAIL:
                if not is_verified_scored_terminal_evidence(
                    verified_terminal_evidence
                ):
                    raise SecGemmaOnlineRiskOverlayStoreError(
                        "Terminal-fail requires opaque scored failed-gate evidence"
                    )
            else:  # pragma: no cover - guarded by TERMINAL_STATUSES
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Unsupported terminal evidence status"
                )
            assert type(verified_terminal_evidence) is VerifiedScoredTerminalEvidence
            validator = validate_scored_terminal_evidence
        try:
            evidence = validator(
                verified_terminal_evidence,
                implementation_manifest=self._implementation_manifest,
                attempt_plan=plan,
            )
        except SecGemmaOnlineRiskOverlayAttemptError as exc:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Verified terminal evidence is invalid"
            ) from exc
        if evidence["terminal_status"] != terminal_status:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Terminal evidence status differs from the requested transition"
            )
        if (
            terminal_status == TERMINAL_PASS
            and evidence["verdict"] != "pass"
        ) or (
            terminal_status == TERMINAL_FAIL
            and evidence["verdict"] != "failed_gate"
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Terminal evidence verdict differs from its transition"
            )
        receipt = verified_terminal_evidence.artifact_receipt
        try:
            receipt_material = store_record_receipt_material(receipt)
        except SecGemmaOnlineRiskOverlayAttemptError as exc:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Terminal evidence lost its exact store receipt"
            ) from exc
        artifact_payload = self._artifact_payload_for_receipt(receipt)
        if artifact_payload != verified_terminal_evidence.artifact_payload:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Terminal evidence artifact differs from current stored bytes"
            )
        material = self.terminal_evidence_material(capability)
        if (
            evidence["attempt_id"] != material["attempt_id"]
            or evidence["attempt_plan_sha256"]
            != material["attempt_plan_sha256"]
            or evidence["record_commitment_sha256"]
            != material["record_commitment_sha256"]
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Terminal evidence does not commit the current post-consumption records"
            )
        if (
            type(verified_terminal_evidence)
            is VerifiedScoredTerminalEvidence
            and evidence["record_counts"] != material["record_counts"]
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Scored terminal evidence record counts changed"
            )
        minimums = MINIMUM_PASS_RECORD_COUNTS[capability.attempt_id]
        for table, minimum in minimums.items():
            count = material["record_counts"][table]
            if count < minimum:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    f"Verified terminal evidence lacks required {table} records"
                )
        if plan["attempt_kind"] == DEVELOPMENT_ACQUISITION:
            if any(
                material["record_counts"][table] != 0
                for table in ("features", "predictions", "lessons", "ledgers")
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Development acquisition contains forbidden semantic records"
                )
        publication = verified_terminal_evidence.external_publication.publication
        return {
            "terminal_evidence": evidence,
            "external_publication": publication,
            "artifact_receipt": receipt_material,
        }

    def finish_attempt(
        self,
        capability: EffectCapability,
        *,
        terminal_status: str,
        verified_terminal_evidence: (
            VerifiedAcquisitionTerminalEvidence
            | VerifiedScoredTerminalEvidence
            | None
        ) = None,
    ) -> StoreRecordReceipt:
        if terminal_status not in TERMINAL_STATUSES:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Attempt finish status must be terminal"
            )
        plan, history = self._authorize_capability(capability)
        terminal_binding = self._validate_terminal_evidence(
            capability,
            plan,
            terminal_status=terminal_status,
            verified_terminal_evidence=verified_terminal_evidence,
        )
        transition = build_attempt_transition(
            attempt_plan=plan,
            implementation_manifest=self._implementation_manifest,
            status=terminal_status,
            prior_transition=history[-1],
        )
        event_payload: dict[str, Any] = {
            "terminal_status": terminal_status,
            "transition_sha256": transition["transition_sha256"],
        }
        if terminal_binding is not None:
            event_payload.update(terminal_binding)
        self._append_anchor(
            "terminal_intent",
            self._verify_database_chain(),
            attempt_id=capability.attempt_id,
            event_payload=event_payload,
        )
        try:
            with self._transaction() as connection:
                receipt = self._append_attempt_transition_tx(
                    connection, plan=plan, transition=transition
                )
            self._append_anchor(
                "terminal_committed",
                self._verify_database_chain(),
                attempt_id=capability.attempt_id,
                event_payload=event_payload,
            )
        except Exception:
            self._poisoned = True
            raise
        self._active_capabilities.pop(capability.attempt_id, None)
        return receipt

    def _recover_to_indeterminate(self, attempt_id: str) -> None:
        rows = self._attempt_rows(attempt_id)
        if not rows:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Recovery attempt is not registered"
            )
        plan = self._attempt_plan_from_rows(rows)
        history = self.attempt_history(attempt_id)
        transitions: list[dict[str, Any]] = []
        if rows[-1]["status"] == PLANNED:
            consumed = build_attempt_transition(
                attempt_plan=plan,
                implementation_manifest=self._implementation_manifest,
                status=CONSUMED,
                prior_transition=history[-1],
            )
            transitions.append(consumed)
            history.append(consumed)
        elif rows[-1]["status"] != CONSUMED:
            return
        terminal = build_attempt_transition(
            attempt_plan=plan,
            implementation_manifest=self._implementation_manifest,
            status=TERMINAL_INDETERMINATE,
            prior_transition=history[-1],
        )
        transitions.append(terminal)
        with self._transaction() as connection:
            for transition in transitions:
                self._append_attempt_transition_tx(
                    connection, plan=plan, transition=transition
                )
        self._append_anchor(
            "recovered_indeterminate",
            self._verify_database_chain(),
            attempt_id=attempt_id,
            event_payload={
                "terminal_status": TERMINAL_INDETERMINATE,
                "transition_sha256": terminal["transition_sha256"],
            },
        )

    def _reconcile_reopen(
        self,
        database: Mapping[str, Any],
        anchor: Mapping[str, Any],
    ) -> None:
        latest = anchor["latest"]
        anchored_sequence = latest["database_journal_sequence"]
        database_sequence = database["journal_entry_count"]
        if anchored_sequence > database_sequence:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Database rolled back behind the external anchor"
            )
        if (
            anchored_sequence == database_sequence
            and latest["database_journal_tip_sha256"]
            != database["journal_tip_sha256"]
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Database tip differs from the external anchor"
            )
        current_consumed = [
            attempt_id
            for attempt_id, status in database["attempt_states"].items()
            if status == CONSUMED
        ]
        if len(current_consumed) > 1:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Multiple consumed attempts cannot coexist"
            )
        if latest["event"] == "consume_intent":
            attempt_id = latest["attempt_id"]
            if attempt_id is None:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Consumption intent lacks its attempt"
                )
            status = database["attempt_states"].get(attempt_id)
            if status in {PLANNED, CONSUMED}:
                self._recover_to_indeterminate(attempt_id)
                return
        if current_consumed:
            self._recover_to_indeterminate(current_consumed[0])
            return
        if anchored_sequence < database_sequence:
            new_attempt_rows = list(
                self._require_open().execute(
                    """
                    SELECT attempt_id,status,payload_json
                    FROM attempts
                    WHERE journal_sequence > ?
                    ORDER BY journal_sequence
                    """,
                    (anchored_sequence,),
                )
            )
            if (
                latest["event"] == "terminal_intent"
                and len(new_attempt_rows) == 1
                and new_attempt_rows[0]["attempt_id"] == latest["attempt_id"]
                and new_attempt_rows[0]["status"]
                == latest["event_payload"].get("terminal_status")
            ):
                transition = _strict_json_bytes(
                    bytes(new_attempt_rows[0]["payload_json"]),
                    "recovered terminal transition",
                )
                if (
                    transition["transition_sha256"]
                    != latest["event_payload"].get("transition_sha256")
                ):
                    raise SecGemmaOnlineRiskOverlayStoreError(
                        "Recovered terminal transition differs from its intent"
                    )
                self._append_anchor(
                    "terminal_committed",
                    database,
                    attempt_id=latest["attempt_id"],
                    event_payload=latest["event_payload"],
                )
                return
            terminal_passes = {
                row["attempt_id"]
                for row in new_attempt_rows
                if row["status"] == TERMINAL_PASS
            }
            if terminal_passes:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Unanchored terminal pass cannot be recovered"
                )
            self._append_anchor(
                "recovered_checkpoint",
                database,
                attempt_id=latest["attempt_id"],
                event_payload={"prior_event": latest["event"]},
            )

    def _receipt_from_row(
        self, table: str, row: sqlite3.Row
    ) -> StoreRecordReceipt:
        return StoreRecordReceipt(
            table=table,
            identity=row["identity"],
            attempt_id=row["attempt_id"],
            payload_sha256=row["payload_sha256"],
            journal_sequence=row["journal_sequence"],
            journal_entry_sha256=row["journal_entry_sha256"],
        )

    def record_receipt(self, table: str, identity: str) -> StoreRecordReceipt:
        if table not in ALL_TYPED_TABLES:
            raise SecGemmaOnlineRiskOverlayStoreError("Unknown typed table")
        fixed_identity = _identity(identity, f"{table} identity")
        row = self._require_open().execute(
            f"SELECT * FROM {table} WHERE identity=?", (fixed_identity,)
        ).fetchone()
        if row is None:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Store record does not exist"
            )
        return self._receipt_from_row(table, row)

    def predecessor_feature_rows(self, stage: str) -> list[dict[str, Any]]:
        """Return exact feature rows from terminal-passed predecessor stages.

        This deliberately exposes no acquisition bytes, model requests, or
        current-stage records.  The returned rows are detached canonical JSON
        values whose store, journal, terminal anchor, stage, self-hash, and
        chronological order have all been revalidated.
        """

        if type(stage) is not str or stage not in _PREDECESSOR_FEATURE_ATTEMPTS:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Feature predecessor stage is not preregistered"
            )
        snapshot = self.verify_chain()
        required = _PREDECESSOR_FEATURE_ATTEMPTS[stage]
        if not required:
            return []

        detached_rows: list[dict[str, Any]] = []
        seen_accessions: set[str] = set()
        seen_hashes: set[str] = set()
        prior_key: tuple[str, str, str] | None = None
        connection = self._require_open()
        for expected_stage, attempt_id in required:
            if snapshot["attempt_states"].get(attempt_id) != TERMINAL_PASS:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Feature predecessor is absent or did not terminal-pass"
                )
            history = self.attempt_history(attempt_id)
            if (
                len(history) != 3
                or [item["status"] for item in history]
                != [PLANNED, CONSUMED, TERMINAL_PASS]
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Feature predecessor transition chain is not exact"
                )
            binding = self.terminal_anchor_binding(attempt_id)
            evidence = binding["terminal_evidence"]
            receipt = binding["artifact_receipt"]
            publication = binding["external_publication"]
            commitment = self._record_commitment(attempt_id)
            if (
                evidence.get("terminal_status") != TERMINAL_PASS
                or evidence.get("stage") != expected_stage
                or evidence.get("attempt_id") != attempt_id
                or evidence.get("record_commitment_sha256")
                != commitment["record_commitment_sha256"]
                or evidence.get("record_counts") != commitment["record_counts"]
                or receipt.get("attempt_id") != attempt_id
                or publication.get("attempt_id") != attempt_id
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Feature predecessor terminal anchor lost its exact binding"
                )
            rows = list(
                connection.execute(
                    """
                    SELECT
                        f.identity,
                        f.attempt_id,
                        f.payload_json,
                        f.payload_sha256,
                        f.journal_sequence,
                        f.journal_entry_sha256,
                        j.record_table AS journal_record_table,
                        j.identity AS journal_identity,
                        j.attempt_id AS journal_attempt_id,
                        j.payload_sha256 AS journal_payload_sha256,
                        j.entry_sha256 AS journal_entry_sha256_exact
                    FROM features AS f
                    JOIN journal AS j
                      ON j.sequence=f.journal_sequence
                    WHERE f.attempt_id=?
                    ORDER BY f.journal_sequence
                    """,
                    (attempt_id,),
                )
            )
            if (
                len(rows) != commitment["record_counts"]["features"]
                or not rows
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Feature predecessor has an incomplete persisted row set"
                )
            for ordinal, stored in enumerate(rows, start=1):
                payload = bytes(stored["payload_json"])
                payload_hash = hashlib.sha256(payload).hexdigest()
                if (
                    stored["attempt_id"] != attempt_id
                    or stored["journal_record_table"] != "features"
                    or stored["journal_identity"] != stored["identity"]
                    or stored["journal_attempt_id"] != attempt_id
                    or stored["journal_payload_sha256"]
                    != stored["payload_sha256"]
                    or stored["journal_entry_sha256_exact"]
                    != stored["journal_entry_sha256"]
                    or payload_hash != stored["payload_sha256"]
                ):
                    raise SecGemmaOnlineRiskOverlayStoreError(
                        "Feature predecessor row lost its journal binding"
                    )
                value = _strict_json_bytes(
                    payload,
                    (
                        f"{expected_stage} predecessor feature row "
                        f"{ordinal}"
                    ),
                )
                row, order_key = _validated_predecessor_feature_row(
                    value,
                    expected_stage=expected_stage,
                    location=(
                        f"{expected_stage} predecessor feature row "
                        f"{ordinal}"
                    ),
                )
                feature_hash = row["feature_row_sha256"]
                accession = row["accession_number"]
                if (
                    stored["identity"] != f"feature:{feature_hash}"
                    or feature_hash in seen_hashes
                    or accession in seen_accessions
                    or (prior_key is not None and order_key <= prior_key)
                ):
                    raise SecGemmaOnlineRiskOverlayStoreError(
                        "Feature predecessor chronology is duplicated or reordered"
                    )
                seen_hashes.add(feature_hash)
                seen_accessions.add(accession)
                prior_key = order_key
                detached_rows.append(row)
        return detached_rows

    def terminal_acquisition_phase_evidence(
        self,
        attempt_id: str,
    ) -> dict[str, Any]:
        """Return only safe acquisition evidence from one terminal-pass attempt.

        Raw provider bytes, request bodies, model requests, semantic outputs,
        and generic record access are intentionally unavailable through this
        reader.  Every returned mapping is detached after the database journal,
        external anchor, terminal artifact, publication, phase receipt, and
        canonical phase-output commitment have been revalidated.
        """

        if (
            type(attempt_id) is not str
            or attempt_id not in _ACQUISITION_RECOVERY_SCOPE
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Acquisition recovery attempt is not preregistered"
            )
        expected_stage, expected_command = (
            _ACQUISITION_RECOVERY_SCOPE[attempt_id]
        )
        expected_attempt_kind = ATTEMPT_KIND_BY_ID[attempt_id]
        snapshot = self.verify_chain()
        if snapshot["attempt_states"].get(attempt_id) != TERMINAL_PASS:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Acquisition recovery attempt did not terminal-pass"
            )
        history = self.attempt_history(attempt_id)
        if (
            len(history) != 3
            or [item["status"] for item in history]
            != [PLANNED, CONSUMED, TERMINAL_PASS]
            or history[-1].get("attempt_id") != attempt_id
            or history[-1].get("attempt_kind") != expected_attempt_kind
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Acquisition recovery transition chain is not exact"
            )

        binding = self.terminal_anchor_binding(attempt_id)
        raw_terminal = binding["terminal_evidence"]
        terminal_fields = (
            set(ACQUISITION_TERMINAL_EVIDENCE_FIELDS)
            if attempt_id == DEVELOPMENT_ACQUISITION_ID
            else set(SCORED_TERMINAL_EVIDENCE_FIELDS)
        )
        terminal = _validated_self_hashed_mapping(
            raw_terminal,
            expected_fields=terminal_fields,
            hash_field="terminal_evidence_sha256",
            location="acquisition recovery terminal evidence",
        )
        commitment = self._record_commitment(attempt_id)
        if (
            terminal.get("verdict") != "pass"
            or terminal.get("terminal_status") != TERMINAL_PASS
            or terminal.get("stage") != expected_stage
            or terminal.get("attempt_id") != attempt_id
            or terminal.get("attempt_kind") != expected_attempt_kind
            or terminal.get("attempt_plan_sha256")
            != history[-1].get("attempt_plan_sha256")
            or terminal.get("record_commitment_sha256")
            != commitment["record_commitment_sha256"]
            or (
                attempt_id != DEVELOPMENT_ACQUISITION_ID
                and terminal.get("record_counts")
                != commitment["record_counts"]
            )
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Acquisition recovery terminal evidence is not current"
            )

        receipt_material = binding["artifact_receipt"]
        receipt_fields = {
            "table",
            "identity",
            "attempt_id",
            "payload_sha256",
            "journal_sequence",
            "journal_entry_sha256",
        }
        if (
            type(receipt_material) is not dict
            or set(receipt_material) != receipt_fields
            or receipt_material.get("table") != "artifacts"
            or receipt_material.get("attempt_id") != attempt_id
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Acquisition recovery artifact receipt changed"
            )
        try:
            receipt = StoreRecordReceipt(**receipt_material)
        except TypeError as exc:  # pragma: no cover - exact fields above
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Acquisition recovery artifact receipt changed"
            ) from exc
        artifact = self._artifact_payload_for_receipt(receipt)
        artifact_hash = canonical_sha256(artifact)
        receipt_hash_field = (
            "acquisition_artifact_receipt_sha256"
            if attempt_id == DEVELOPMENT_ACQUISITION_ID
            else "joint_artifact_receipt_sha256"
        )
        terminal_artifact_hash = (
            terminal.get("acquisition_validation_sha256")
            if attempt_id == DEVELOPMENT_ACQUISITION_ID
            else terminal.get("joint_stage_report_sha256")
        )
        if (
            receipt.payload_sha256 != artifact_hash
            or terminal.get(receipt_hash_field)
            != canonical_sha256(receipt_material)
            or type(terminal_artifact_hash) is not str
            or _SHA256_RE.fullmatch(terminal_artifact_hash) is None
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Acquisition recovery artifact binding changed"
            )
        if attempt_id != DEVELOPMENT_ACQUISITION_ID:
            joint_hash = artifact.get("joint_stage_report_sha256")
            joint_body = {
                key: child
                for key, child in artifact.items()
                if key != "joint_stage_report_sha256"
            }
            if (
                artifact.get("metric_stage") != expected_stage
                or artifact.get("attempt_id") != attempt_id
                or artifact.get("attempt_plan_sha256")
                != history[-1].get("attempt_plan_sha256")
                or type(joint_hash) is not str
                or _SHA256_RE.fullmatch(joint_hash) is None
                or joint_hash != canonical_sha256(joint_body)
                or joint_hash != terminal_artifact_hash
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Acquisition recovery joint artifact changed"
                )

        publication = _validated_self_hashed_mapping(
            binding["external_publication"],
            expected_fields=set(EXTERNAL_PUBLICATION_FIELDS),
            hash_field="publication_sha256",
            location="acquisition recovery external publication",
        )
        if (
            publication.get("contract_version") != CONTRACT_VERSION
            or publication.get("contract_sha256") != CONTRACT_SHA256
            or publication.get("attempt_id") != attempt_id
            or publication.get("terminal_status") != TERMINAL_PASS
            or publication.get("artifact_sha256") != terminal_artifact_hash
            or publication.get("external_cost_usd") != 0
            or terminal.get("external_publication_sha256")
            != publication["publication_sha256"]
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Acquisition recovery publication changed"
            )

        identity = f"phase:{attempt_id}:acquisition"
        rows = list(
            self._require_open().execute(
                """
                SELECT
                    e.identity,
                    e.attempt_id,
                    e.payload_json,
                    e.payload_sha256,
                    e.journal_sequence,
                    e.journal_entry_sha256,
                    j.record_table AS journal_record_table,
                    j.identity AS journal_identity,
                    j.attempt_id AS journal_attempt_id,
                    j.payload_sha256 AS journal_payload_sha256,
                    j.entry_sha256 AS journal_entry_sha256_exact
                FROM evidence AS e
                JOIN journal AS j
                  ON j.sequence=e.journal_sequence
                WHERE e.identity=? AND e.attempt_id=?
                """,
                (identity, attempt_id),
            )
        )
        if len(rows) != 1:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Acquisition recovery lacks one exact phase evidence row"
            )
        stored = rows[0]
        payload_bytes = bytes(stored["payload_json"])
        if (
            stored["identity"] != identity
            or stored["attempt_id"] != attempt_id
            or stored["journal_record_table"] != "evidence"
            or stored["journal_identity"] != identity
            or stored["journal_attempt_id"] != attempt_id
            or stored["journal_payload_sha256"]
            != stored["payload_sha256"]
            or stored["journal_entry_sha256_exact"]
            != stored["journal_entry_sha256"]
            or hashlib.sha256(payload_bytes).hexdigest()
            != stored["payload_sha256"]
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Acquisition recovery phase row lost its journal binding"
            )
        phase_value = _strict_json_bytes(
            payload_bytes,
            "acquisition recovery phase evidence",
        )
        safe_payload, _ = _validated_acquisition_phase_material(
            phase_value,
            expected_stage=expected_stage,
            expected_attempt_id=attempt_id,
            expected_attempt_kind=expected_attempt_kind,
            expected_command=expected_command,
        )

        checkpoints = [
            entry
            for entry in self._anchor_entries()
            if entry["event"] == "checkpoint"
            and entry["attempt_id"] == attempt_id
            and entry["event_payload"].get("record_table") == "evidence"
            and entry["event_payload"].get("identity") == identity
        ]
        if (
            len(checkpoints) != 1
            or checkpoints[0]["event_payload"].get("effect")
            != "official_sec_network"
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Acquisition recovery phase row lost its effect anchor"
            )

        report = safe_payload["verified_acquisition_report"]
        if attempt_id == DEVELOPMENT_ACQUISITION_ID:
            if (
                artifact != report
                or terminal["acquisition_validation_sha256"]
                != report["validation_sha256"]
                or terminal["bundle_sha256"] != report["bundle_sha256"]
                or terminal["manifest_sha256"]
                != report["manifest_sha256"]
                or terminal["private_index_sha256"]
                != report["private_index_sha256"]
                or terminal["check_set_sha256"]
                != report["check_set_sha256"]
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Development acquisition artifact crossed its phase evidence"
                )
        else:
            predecessor_attempts = (
                (DEVELOPMENT_ACQUISITION_ID,)
                if attempt_id == CONFIRMATION_ATTEMPT_ID
                else (
                    DEVELOPMENT_ACQUISITION_ID,
                    CONFIRMATION_ATTEMPT_ID,
                )
            )
            predecessor_bundles = [
                self.terminal_acquisition_phase_evidence(predecessor)[
                    "verified_acquisition_report"
                ]["bundle_sha256"]
                for predecessor in predecessor_attempts
            ]
            if (
                report["predecessor_chain_bundle_sha256s"]
                != predecessor_bundles
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Acquisition recovery predecessor bundle chain changed"
                )
        return _detach_json(
            safe_payload,
            "detached acquisition recovery material",
        )

    def terminal_anchor_binding(self, attempt_id: str) -> dict[str, Any]:
        """Read one exact evidence-bearing committed terminal anchor."""

        if attempt_id not in ATTEMPT_KIND_BY_ID:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Unknown preregistered attempt id"
            )
        matches = [
            entry
            for entry in self._anchor_entries()
            if entry["event"] == "terminal_committed"
            and entry["attempt_id"] == attempt_id
        ]
        if len(matches) != 1:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Attempt lacks one exact committed terminal anchor"
            )
        payload = _detach_json(
            matches[0]["event_payload"],
            "committed terminal anchor payload",
        )
        required = {
            "terminal_status",
            "transition_sha256",
            "terminal_evidence",
            "external_publication",
            "artifact_receipt",
        }
        if type(payload) is not dict or set(payload) != required:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Committed terminal anchor has no releasable evidence binding"
            )
        history = self.attempt_history(attempt_id)
        evidence = payload["terminal_evidence"]
        publication = payload["external_publication"]
        receipt = payload["artifact_receipt"]
        if (
            not history
            or history[-1]["status"] != payload["terminal_status"]
            or history[-1]["transition_sha256"]
            != payload["transition_sha256"]
            or type(evidence) is not dict
            or evidence.get("attempt_id") != attempt_id
            or evidence.get("terminal_status") != payload["terminal_status"]
            or type(publication) is not dict
            or evidence.get("external_publication_sha256")
            != publication.get("publication_sha256")
            or type(receipt) is not dict
            or evidence.get(
                "acquisition_artifact_receipt_sha256",
                evidence.get("joint_artifact_receipt_sha256"),
            )
            != canonical_sha256(receipt)
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Committed terminal anchor evidence binding is inconsistent"
            )
        return {
            "terminal_evidence": evidence,
            "external_publication": publication,
            "artifact_receipt": receipt,
        }

    def _verify_database_chain(self) -> dict[str, Any]:
        self._verify_schema()
        self._verify_meta()
        connection = self._require_open()
        expected_prior = _sha256(
            self._meta_value("journal_genesis_sha256"), "journal genesis"
        )
        journal_rows = list(
            connection.execute("SELECT * FROM journal ORDER BY sequence")
        )
        typed_seen: set[tuple[str, str]] = set()
        semantic_record_sequences: list[tuple[str, int]] = []
        for expected_sequence, row in enumerate(journal_rows, start=1):
            if row["sequence"] != expected_sequence:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Journal sequence is duplicated, reordered, or skipped"
                )
            table = row["record_table"]
            if table not in ALL_TYPED_TABLES:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Journal names an unknown typed table"
                )
            identity = _identity(row["identity"], "journal identity")
            attempt_id = row["attempt_id"]
            if attempt_id not in ATTEMPT_KIND_BY_ID:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Journal names an unknown attempt"
                )
            payload_hash = _sha256(
                row["payload_sha256"], "journal payload hash"
            )
            prior = _sha256(
                row["previous_entry_sha256"], "journal previous hash"
            )
            entry_body = {
                "schema_version": JOURNAL_ENTRY_SCHEMA_VERSION,
                "sequence": expected_sequence,
                "record_table": table,
                "identity": identity,
                "attempt_id": attempt_id,
                "payload_sha256": payload_hash,
                "previous_entry_sha256": prior,
            }
            expected_entry = canonical_sha256(entry_body)
            if (
                prior != expected_prior
                or row["entry_sha256"] != expected_entry
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Journal chain forked, changed, or was reordered"
                )
            typed = connection.execute(
                f"SELECT * FROM {table} WHERE identity=?", (identity,)
            ).fetchone()
            if typed is None:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Journal entry lacks its typed record"
                )
            payload_bytes = bytes(typed["payload_json"])
            payload_value = _strict_json_bytes(
                payload_bytes, f"{table} payload {identity}"
            )
            if (
                typed["attempt_id"] != attempt_id
                or typed["payload_sha256"] != payload_hash
                or hashlib.sha256(payload_bytes).hexdigest() != payload_hash
                or typed["journal_sequence"] != expected_sequence
                or typed["journal_entry_sha256"] != expected_entry
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Typed record does not reconcile to its journal entry"
                )
            if table == ATTEMPTS_TABLE and (
                type(payload_value) is not dict
                or typed["status"] != payload_value.get("status")
                or attempt_id != payload_value.get("attempt_id")
                or identity
                != f"{attempt_id}:{payload_value.get('sequence_number')}"
                or typed["attempt_plan_sha256"]
                != payload_value.get("attempt_plan_sha256")
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Attempt row metadata differs from its transition payload"
                )
            if table != ATTEMPTS_TABLE:
                semantic_record_sequences.append(
                    (attempt_id, expected_sequence)
                )
            typed_seen.add((table, identity))
            expected_prior = expected_entry
        all_typed = {
            (table, row["identity"])
            for table in ALL_TYPED_TABLES
            for row in connection.execute(f"SELECT identity FROM {table}")
        }
        if typed_seen != all_typed:
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Typed records and journal membership differ"
            )
        present_attempts: list[str] = []
        attempt_states: dict[str, str] = {}
        attempt_windows: dict[str, tuple[int, int | None]] = {}
        for attempt_id in ATTEMPT_IDS:
            rows = self._attempt_rows(attempt_id)
            if not rows:
                continue
            present_attempts.append(attempt_id)
            plan = self._attempt_plan_from_rows(rows)
            history = self.attempt_history(attempt_id)
            try:
                validated = validate_attempt_history(
                    attempt_plan=plan,
                    implementation_manifest=self._implementation_manifest,
                    transitions=history,
                )
            except SecGemmaOnlineRiskOverlayAttemptError as exc:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Stored attempt transition chain is invalid"
                ) from exc
            attempt_states[attempt_id] = validated[-1]["status"]
            consumed_sequences = [
                row["journal_sequence"]
                for row in rows
                if row["status"] == CONSUMED
            ]
            terminal_sequences = [
                row["journal_sequence"]
                for row in rows
                if row["status"] in TERMINAL_STATUSES
            ]
            if len(consumed_sequences) > 1 or len(terminal_sequences) > 1:
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Attempt has duplicate consumed or terminal transitions"
                )
            if consumed_sequences:
                attempt_windows[attempt_id] = (
                    consumed_sequences[0],
                    terminal_sequences[0] if terminal_sequences else None,
                )
            predecessor_id = PREREQUISITE_ATTEMPT_BY_ID[attempt_id]
            if predecessor_id is not None:
                predecessor = self.attempt_history(predecessor_id)
                if (
                    not predecessor
                    or predecessor[-1]["status"] != TERMINAL_PASS
                    or plan["prerequisite_terminal_transition_sha256"]
                    != predecessor[-1]["transition_sha256"]
                ):
                    raise SecGemmaOnlineRiskOverlayStoreError(
                        "Stored attempt lost its predecessor pass binding"
                    )
        if present_attempts != list(ATTEMPT_IDS[: len(present_attempts)]):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Stored attempts are not a contiguous fixed prefix"
            )
        for attempt_id, sequence in semantic_record_sequences:
            window = attempt_windows.get(attempt_id)
            if (
                window is None
                or sequence <= window[0]
                or (window[1] is not None and sequence >= window[1])
            ):
                raise SecGemmaOnlineRiskOverlayStoreError(
                    "Semantic record lies outside its consumed attempt window"
                )
        counts = {
            table: connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            for table in ALL_TYPED_TABLES
        }
        return {
            "schema_version": STORE_SCHEMA_VERSION,
            "implementation_manifest_sha256": self._implementation_manifest[
                "implementation_manifest_sha256"
            ],
            "store_instance_id": self._store_instance_id,
            "journal_entry_count": len(journal_rows),
            "journal_tip_sha256": expected_prior,
            "attempt_states": attempt_states,
            "table_counts": counts,
            "chain_valid": True,
        }

    def verify_chain(self) -> dict[str, Any]:
        database = self._verify_database_chain()
        anchor = self._verify_anchor_chain()
        latest = anchor["latest"]
        if (
            latest["database_journal_sequence"]
            != database["journal_entry_count"]
            or latest["database_journal_tip_sha256"]
            != database["journal_tip_sha256"]
        ):
            raise SecGemmaOnlineRiskOverlayStoreError(
                "Database and external anchor tips differ"
            )
        return {
            **database,
            "anchor_entry_count": anchor["entry_count"],
            "anchor_tip_sha256": anchor["tip_sha256"],
            "anchor_valid": True,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return integrity metadata without semantic payloads."""

        return self.verify_chain()


__all__ = [
    "ALL_TYPED_TABLES",
    "ANCHOR_ENTRY_SCHEMA_VERSION",
    "ANCHOR_FILENAME",
    "ANCHOR_RELATIVE_DIRECTORY",
    "APPLICATION_ID",
    "DATABASE_FILENAME",
    "EffectCapability",
    "JOURNAL_ENTRY_SCHEMA_VERSION",
    "LOCK_FILENAME",
    "MAX_RECORD_PAYLOAD_BYTES",
    "RECORD_TABLES",
    "STATE_RELATIVE_DIRECTORY",
    "STORE_SCHEMA_VERSION",
    "SecGemmaOnlineRiskOverlayStore",
    "SecGemmaOnlineRiskOverlayStoreConflict",
    "SecGemmaOnlineRiskOverlayStoreError",
    "StoreRecordReceipt",
]
