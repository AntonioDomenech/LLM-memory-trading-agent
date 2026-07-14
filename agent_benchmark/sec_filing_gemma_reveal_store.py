"""Effectful, local persistence for SEC/Gemma reveal governance.

The sibling :mod:`sec_filing_gemma_reveal_registry` module deliberately has no
I/O.  This module is its narrow effectful boundary.  It authenticates the
checked-in genesis evidence, keeps one authoritative registry/pin pair, applies
registry appends with compare-and-swap semantics, and consumes stage reveal
requests exactly once only after the fixed semantic prerequisite verifier
returns a strongly bound result.

No model runtime or network service is used by this module.  Its owned carry-in
helper may re-read and copy only normalized filing bytes that a terminal SEC
reader receipt already binds.  Its market finalizer may re-read quarantined
provider bytes and canonical snapshots, but cannot fetch them or accept market
rows from a caller.  The anchor files contain governance receipts, while the
adjacent fixed ``stage_outputs`` namespace contains their replayed bytes.
Registration and intermediate access never count as a final-period touch;
successful final-request consumption increments the separate actual-final-
touch counter exactly once.
"""

from __future__ import annotations

from collections.abc import Mapping
import base64
import copy
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import subprocess
import sys
import time
from types import MappingProxyType
from typing import Any, Final
import uuid

from agent_benchmark.sec_filing_gemma_contract import (
    CANONICAL_IDENTITY_LEXICON,
    CONTRACT_VERSION,
    EXTRACTOR_REQUEST_VERSION,
    PREPROCESSOR_VERSION,
    REQUIRED_STAGE_VERIFIER_CHECKS,
    SecFilingGemmaContractError,
    build_stage_content_manifest,
    canonical_sha256,
    validate_redacted_input_manifest,
    validate_extractor_request,
    validate_stage_content_manifest,
)
from agent_benchmark.sec_filing_gemma_ollama import (
    INVALID_ATTEMPT_STATUS,
    VALID_ATTEMPT_STATUS,
    SecFilingGemmaOllamaError,
    validate_ollama_model_attempt_receipt,
    validate_ollama_runtime_probe_receipt,
    validate_runtime_identity_guard,
)
from agent_benchmark.sec_filing_gemma_preprocessor import (
    SecFilingGemmaPreprocessorError,
    validate_owned_preprocessing_receipt,
    validate_preprocessed_event,
)
from agent_benchmark.sec_filing_gemma_corpus import (
    SecFilingGemmaCorpusError,
    _validate_persisted_authenticated_stage_access_batch,
)
from agent_benchmark.sec_filing_gemma_market_acquirer import (
    MARKET_ACQUISITION_BUNDLE_SCHEMA_VERSION,
    MARKET_ACQUISITION_SCHEMA_VERSION,
    YAHOO_MAX_RESPONSE_BYTES,
    YAHOO_MAX_TOTAL_RESPONSE_BYTES,
    _consume_owned_market_transport_capability,
    build_development_market_acquisition_plan,
    validate_development_market_acquisition_bundle,
)
from agent_benchmark.sec_filing_gemma_features import (
    OWNED_DEVELOPMENT_FEATURE_INPUTS_SCHEMA_VERSION,
    OWNED_DEVELOPMENT_LABEL_PROJECTION_SCHEMA_VERSION,
    SecFilingGemmaFeatureError,
    build_owned_development_feature_batch,
    build_owned_development_label_batch,
    build_sec_filing_gemma_feature_row,
    build_twenty_session_label_evidence,
    build_validated_extraction_event_proof,
    build_validated_market_prefix_proof,
    build_validated_universe_event_proof,
    validate_owned_development_feature_batch,
    validate_owned_development_label_batch,
    validate_twenty_session_label_evidence,
)
from agent_benchmark.sec_filing_gemma_training_membership import (
    OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_PROJECTION_SCHEMA_VERSION,
    SecFilingGemmaTrainingMembershipError,
    build_owned_development_training_membership_batch,
    validate_owned_development_training_membership_batch,
)
from agent_benchmark.sec_filing_gemma_learner_fit import (
    OWNED_DEVELOPMENT_OOF_LEARNER_FIT_PROJECTION_SCHEMA_VERSION,
    SecFilingGemmaLearnerFitError,
    build_owned_development_oof_learner_fit_batch,
    derive_development_oof_learner_fit_input_specs,
    validate_owned_development_oof_learner_fit_batch,
)
from agent_benchmark.sec_filing_gemma_learner_prediction import (
    OWNED_DEVELOPMENT_OOF_PREDICTION_PROJECTION_SCHEMA_VERSION,
    SecFilingGemmaLearnerPredictionError,
    derive_development_oof_prediction_feature_batch,
    derive_development_oof_prediction_fold_model_bundle,
    derive_development_oof_prediction_fold_model_specs,
    derive_development_oof_prediction_input_specs,
)
from agent_benchmark.sec_filing_gemma_market_evidence import (
    MARKET_SYMBOLS,
    build_decision_market_prefix,
)
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND,
    HISTORICAL_REGISTRY_CANONICAL_SHA256,
    REVEAL_REQUEST_SCHEMA_VERSION,
    SecFilingGemmaRevealRegistryError,
    build_initial_reveal_registry,
    derive_registry_pin,
    historical_reveal_declaration,
    validate_registry_pin_transition,
    validate_reveal_registry,
    validate_single_candidate_reveal_request,
)
from agent_benchmark.sec_filing_gemma_stage_verifier import (
    STAGE_EVIDENCE_KEYS,
    STAGE_EVIDENCE_SCHEMA_VERSION,
    STAGE_AUDIT_RECEIPT_SCHEMA_VERSION,
    authoritative_prerequisite_validator,
    detach_untrusted_stage_json,
    preflight_untrusted_stage_json,
)
from agent_benchmark.sec_filing_gemma_stage_access import (
    DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID,
    SecFilingGemmaStageAccessError,
    build_development_content_root_plan,
    validate_development_content_root_plan,
    validate_prior_same_form_carry_in_scope,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS
from agent_benchmark.sec_filing_gemma_stage_authorization import (
    CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION,
    MODEL_EXECUTION_SOURCE_ROLES,
    MARKET_EXECUTION_SOURCE_ROLES,
    SEC_EXECUTION_RESOLVED_SOURCE_PATHS,
    SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID,
    STAGE_EVIDENCE_OUTPUT_COMPONENT_ID,
    STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH,
    SecFilingGemmaStageAuthorizationError,
    authenticate_reveal_store_trusted_stage_content_pin,
    build_development_oof_learner_fit_plan,
    build_development_oof_prediction_plan,
    build_development_label_assembly_plan,
    build_development_training_membership_assembly_plan,
    build_development_model_execution_abort,
    build_development_model_execution_claim,
    build_development_model_reader_receipt,
    build_development_feature_assembly_plan,
    build_development_market_execution_abort,
    build_development_market_execution_claim,
    build_development_market_reader_receipt,
    build_development_sec_execution_abort,
    build_development_sec_execution_claim,
    build_development_sec_reader_receipt,
    build_development_root_carry_in_reader_receipt,
    build_stage_carry_in_reader_receipt,
    build_stage_model_execution_abort,
    build_stage_model_execution_claim,
    build_stage_model_reader_receipt,
    build_reveal_store_current_tip_anchor,
    build_consumed_stage_authorization_grant,
    build_consumed_stage_output_receipt,
    build_stage_sec_execution_abort,
    build_stage_sec_execution_claim,
    build_stage_sec_reader_receipt,
    derive_consumed_stage_store_state_pin,
    derive_reveal_store_trusted_stage_content_pin,
    validate_consumed_stage_authorization_grant,
    validate_consumed_stage_output_receipt,
    validate_development_root_carry_in_reader_receipt,
    validate_development_feature_assembly_plan,
    validate_development_label_assembly_plan,
    validate_development_oof_learner_fit_plan,
    validate_development_oof_prediction_plan,
    validate_development_training_membership_assembly_plan,
    validate_reveal_store_current_tip_anchor,
    validate_reveal_store_current_tip_anchor_structure,
    validate_reveal_store_current_tip_anchor_transition,
    validate_stage_carry_in_reader_receipt,
    validate_trusted_stage_content_authentication_receipt,
    _sec_component_plan_from_bundle,
)


STORE_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-reveal-store-v1"
STORE_ANCHOR_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-reveal-store-anchor-v1"
CONSUMPTION_LEDGER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-request-ledger-v1"
)
CONSUMPTION_ENTRY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-request-entry-v1"
)
SEMANTIC_PREREQUISITE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-semantic-prerequisite-validation-v1"
)
INITIAL_PIN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-initial-external-registry-pin-v1"
)
STATE_FILENAME: Final[str] = "sec_gemma_reveal_store.json"
CURRENT_TIP_ANCHOR_FILENAME: Final[str] = (
    "sec_gemma_reveal_store_current_tip.json"
)
RESTORE_PENDING_FILENAME: Final[str] = (
    "sec_gemma_reveal_store_restore_pending.json"
)
LOCK_FILENAME: Final[str] = ".sec_gemma_reveal_store.lock"
DEVELOPMENT_SEC_EXECUTION_LOCKS_DIRECTORY_NAME: Final[str] = (
    ".development-sec-execution-locks"
)
MODEL_EXECUTION_LOCK_FILENAME: Final[str] = ".model-execution.lock"
MARKET_EXECUTION_LOCK_FILENAME: Final[str] = ".market-execution.lock"
STAGE_OUTPUTS_DIRECTORY_NAME: Final[str] = "stage_outputs"
SEC_STAGE_COMPONENT_DIRECTORY_NAME: Final[str] = "sec"
SEC_BATCH_COMPLETE_MARKER_FILENAME: Final[str] = "complete.json"
SEC_BATCH_COMPLETE_MARKER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-sec-batch-complete-v1"
)
DEVELOPMENT_SEC_ROOT_COMPLETE_MARKER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-sec-root-complete-v1"
)
STAGE_EVIDENCE_COMPONENT_DIRECTORY_NAME: Final[str] = "stage_evidence"
STAGE_EVIDENCE_FILENAME: Final[str] = "stage_evidence.json"
STAGE_EVIDENCE_COMPLETE_MARKER_FILENAME: Final[str] = "complete.json"
STAGE_EVIDENCE_COMPLETE_MARKER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-evidence-complete-v1"
)
_STAGE_EVIDENCE_COMPLETE_MARKER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "request_sha256",
        "claim_sha256",
        "sec_reader_receipt_sha256",
        "component_id",
        "relative_path",
        "byte_count",
        "document_sha256",
        "stage_evidence_sha256",
        "marker_sha256",
    }
)
PRIOR_SAME_FORM_CARRY_IN_COMPONENT_DIRECTORY_NAME: Final[str] = (
    "prior_same_form_carry_in"
)
PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_FILENAME: Final[str] = "complete.json"
PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-prior-same-form-carry-in-complete-v1"
)
PRIOR_SAME_FORM_CARRY_IN_COMPONENT_ID: Final[str] = (
    "owned_prior_same_form_normalized_text_carry_in"
)
DEVELOPMENT_ROOT_CARRY_IN_COMPONENT_DIRECTORY_NAME: Final[str] = (
    "development_root_prior_same_form_carry_in"
)
DEVELOPMENT_ROOT_CARRY_IN_COMPLETE_MARKER_FILENAME: Final[str] = "complete.json"
DEVELOPMENT_ROOT_CARRY_IN_COMPLETE_MARKER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-root-prior-same-form-carry-in-complete-v1"
)
DEVELOPMENT_ROOT_CARRY_IN_COMPONENT_ID: Final[str] = (
    "owned_development_root_prior_same_form_normalized_text_carry_in"
)
MODEL_EXTRACTION_COMPONENT_DIRECTORY_NAME: Final[str] = "model_extraction"
MODEL_EXTRACTION_COMPLETE_MARKER_FILENAME: Final[str] = "complete.json"
STAGE_MODEL_EXTRACTION_COMPLETE_MARKER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-stage-model-batch-complete-v1"
)
DEVELOPMENT_MODEL_EXTRACTION_COMPLETE_MARKER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-model-batch-complete-v1"
)
MARKET_SOURCE_COMPONENT_DIRECTORY_NAME: Final[str] = "market_source"
DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME: Final[str] = "complete.json"
DEVELOPMENT_MARKET_COMPLETE_MARKER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-market-batch-complete-v1"
)
DEVELOPMENT_MARKET_COMPONENT_ID: Final[str] = (
    "owned_development_market_evidence_batch"
)
MAX_MARKET_BATCH_FILES: Final[int] = 23
MAX_MARKET_BATCH_FILE_BYTES: Final[int] = 128 * 1024 * 1024
MAX_MARKET_BATCH_TOTAL_BYTES: Final[int] = 768 * 1024 * 1024
_DEVELOPMENT_MARKET_COMPLETE_MARKER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "development_root_scope_sha256",
        "claim_sha256",
        "market_acquisition_plan_sha256",
        "acquisition_receipt_sha256",
        "acquisition_bundle_sha256",
        "acquisition_validation_sha256",
        "source_manifest_sha256",
        "market_stage_manifest_sha256",
        "source_reconciliation_sha256",
        "market_component_id",
        "byte_index",
        "byte_index_sha256",
        "byte_count_total",
        "marker_sha256",
    }
)
MODEL_CALL_INTENT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-model-call-intent-v1"
)
OWNED_MODEL_ATTEMPT_TRANSPORT_MODE: Final[str] = (
    "owned_hardened_loopback_session_requires_stage_attestation"
)
OWNED_RUNTIME_PROBE_TRANSPORT_MODE: Final[str] = (
    "owned_hardened_loopback_runtime_probe_unattested"
)
MAX_PRIOR_SAME_FORM_CARRY_IN_RECORDS: Final[int] = 16
MAX_PRIOR_SAME_FORM_CARRY_IN_TOTAL_BYTES: Final[int] = 256 * 1024 * 1024
_PRIOR_SAME_FORM_CARRY_IN_MARKER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "request_sha256",
        "claim_sha256",
        "sec_reader_receipt_sha256",
        "parent_request_sha256",
        "parent_claim_sha256",
        "parent_sec_reader_receipt_sha256",
        "parent_stage_output_receipt_sha256",
        "parent_stage_evidence_sha256",
        "parent_stage_evidence_document_sha256",
        "parent_stage_evidence_complete_marker_sha256",
        "component_id",
        "carry_in_records_sha256",
        "prerequisite_content_manifest_sha256",
        "byte_index",
        "byte_index_sha256",
        "byte_count_total",
        "marker_sha256",
    }
)
_DEVELOPMENT_ROOT_CARRY_IN_MARKER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "request_sha256",
        "claim_sha256",
        "sec_reader_receipt_sha256",
        "stage_access_manifest_sha256",
        "trusted_stage_content_pin_sha256",
        "prerequisite_stage_evidence_sha256",
        "development_root_scope_sha256",
        "development_claim_sha256",
        "development_sec_reader_receipt_sha256",
        "development_content_root_plan_sha256",
        "development_content_manifest_sha256",
        "development_root_complete_marker_sha256",
        "corpus_universe_sha256",
        "component_id",
        "carry_in_records_sha256",
        "prerequisite_content_manifest_sha256",
        "byte_index",
        "byte_index_sha256",
        "byte_count_total",
        "marker_sha256",
    }
)
MAX_STAGE_EVIDENCE_FILE_BYTES: Final[int] = 256 * 1024 * 1024
MAX_SEC_BATCH_FILES: Final[int] = 4_096
MAX_SEC_BATCH_FILE_BYTES: Final[int] = 128 * 1024 * 1024
MAX_SEC_BATCH_TOTAL_BYTES: Final[int] = 1536 * 1024 * 1024
INITIAL_PIN_RELATIVE_PATH: Final[str] = (
    "docs/protocol_evidence/sec_gemma_reveal_registry_initial_pin.json"
)
REQUIRED_SEMANTIC_CHECKS: Final[tuple[str, ...]] = (
    REQUIRED_STAGE_VERIFIER_CHECKS
)
AUTHORITATIVE_VALIDATOR_ID: Final[str] = (
    "aapl-sec-gemma-authoritative-stage-verifier-v1"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TAGGED_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
CURRENT_TIP_PENDING_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-reveal-store-current-tip-pending-v1"
)
RESTORE_PENDING_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-reveal-store-restore-pending-v1"
)
AUTHORITATIVE_STAGE_PROMOTION_ENABLED: Final[bool] = False
AUTHENTICATED_STORE_VERIFIER_CONTEXT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-authenticated-store-verifier-context-v1"
)
PARENT_CONSUMPTION_BINDING_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-parent-consumption-binding-v2"
)
MAX_STATE_FILE_BYTES: Final[int] = 16 * 1024 * 1024
MAX_CURRENT_TIP_ANCHOR_FILE_BYTES: Final[int] = 64 * 1024 * 1024
MAX_TRACKED_ANCHOR_FILE_BYTES: Final[int] = 8 * 1024 * 1024
MAX_RESTORE_PENDING_FILE_BYTES: Final[int] = 128 * 1024 * 1024
_TEMP_RE = re.compile(
    rf"\.(?:{re.escape(STATE_FILENAME)}|"
    rf"{re.escape(CURRENT_TIP_ANCHOR_FILENAME)}|"
    rf"{re.escape(RESTORE_PENDING_FILENAME)})\.[0-9a-f]{{32}}\.tmp\Z"
)
_WINDOWS_REPARSE_POINT = 0x400
_BINARY = getattr(os, "O_BINARY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "registry_sha256",
        "registry_tip_sha256",
        "registered_entry_count",
        "historical_final_reveal_count_lower_bound",
        "stage",
        "stage_access_manifest_sha256",
        "prerequisite_stage",
        "prerequisite_stage_evidence_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "request_scope",
        "authorizes_outcome_access",
        "effectful_atomic_single_use_consumption_required",
        "cross_attempt_comparison_permitted",
        "cross_attempt_winner_selection_permitted",
        "globally_pristine_claim",
        "request_sha256",
    }
)


class SecFilingGemmaRevealStoreError(ValueError):
    """The local reveal store failed closed on unsafe or inconsistent state."""


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise SecFilingGemmaRevealStoreError(f"{location} must be a JSON object")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SecFilingGemmaRevealStoreError(
            f"Invalid {location} keys; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be an integer >= {minimum}"
        )
    return value


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _tagged_sha256(value: Any, location: str) -> str:
    if type(value) is not str or _TAGGED_SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be a tagged lowercase SHA-256 digest"
        )
    return value


def _safe_id(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise SecFilingGemmaRevealStoreError(f"{location} is not a safe identity")
    return value


def _semantic_checks(value: Any, location: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be the exact frozen verifier checklist"
        )
    normalized = tuple(value)
    if normalized != REQUIRED_SEMANTIC_CHECKS:
        raise SecFilingGemmaRevealStoreError(
            f"{location} does not match the exact frozen verifier checklist"
        )
    return normalized


def _same_digest(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


def _json_value_copy(value: Any, location: str) -> Any:
    """Copy one finite JSON value and reject non-JSON or duplicate-key hazards."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be a finite JSON value"
        ) from exc


def _exact_builtin_json_copy(value: Any, location: str) -> Any:
    """Detach an authorization-critical caller value without invoking hooks.

    Only exact built-in containers and scalar types are traversed.  In
    particular, a ``Mapping`` implementation or a ``dict`` subclass is
    rejected before any caller-controlled iterator, item accessor, encoder,
    comparison, or representation method can execute.
    """

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SecFilingGemmaRevealStoreError(
                f"{location} contains a non-finite float"
            )
        return value
    if type(value) is list:
        return [
            _exact_builtin_json_copy(child, f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    if type(value) is tuple:
        return [
            _exact_builtin_json_copy(child, f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    if type(value) is dict:
        detached: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise SecFilingGemmaRevealStoreError(
                    f"{location} contains a non-string key"
                )
            detached[key] = _exact_builtin_json_copy(
                child, f"{location}.{key}"
            )
        return detached
    raise SecFilingGemmaRevealStoreError(
        f"{location} must contain only exact built-in JSON values"
    )


def _exact_caller_dict(value: Any, location: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be an exact built-in dict"
        )
    detached = _exact_builtin_json_copy(value, location)
    if type(detached) is not dict:  # pragma: no cover - guaranteed above
        raise SecFilingGemmaRevealStoreError(f"{location} must be an object")
    return detached


def _strict_json_bytes(payload: bytes, location: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SecFilingGemmaRevealStoreError(
                    f"{location} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise SecFilingGemmaRevealStoreError(
            f"{location} contains non-finite JSON constant {value}"
        )

    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except SecFilingGemmaRevealStoreError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise SecFilingGemmaRevealStoreError(
            f"{location} is not strict UTF-8 JSON"
        ) from exc


def _encoded_state(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store state is not finite JSON"
        ) from exc


def _is_reparse(details: os.stat_result) -> bool:
    return bool(
        getattr(details, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    )


def _validate_real_directory(path: Path, location: str) -> os.stat_result:
    try:
        details = path.lstat()
    except FileNotFoundError as exc:
        raise SecFilingGemmaRevealStoreError(f"{location} does not exist") from exc
    if stat.S_ISLNK(details.st_mode) or _is_reparse(details):
        raise SecFilingGemmaRevealStoreError(
            f"{location} cannot be a link or reparse point"
        )
    if not stat.S_ISDIR(details.st_mode):
        raise SecFilingGemmaRevealStoreError(f"{location} must be a directory")
    return details


def _secure_directory(path: Path, *, create: bool, location: str) -> Path:
    raw = os.fspath(path)
    absolute = Path(os.path.abspath(raw))
    windows = PureWindowsPath(str(absolute))
    if str(absolute).startswith(("\\\\", "//")) or windows.drive.startswith("\\"):
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be on a local filesystem"
        )
    if not absolute.anchor:
        raise SecFilingGemmaRevealStoreError(f"{location} must be absolute")

    anchor = Path(absolute.anchor)
    _validate_real_directory(anchor, f"{location} filesystem root")
    current = anchor
    for component in absolute.relative_to(anchor).parts:
        if component in {"", ".", ".."}:
            raise SecFilingGemmaRevealStoreError(
                f"{location} contains an unsafe path component"
            )
        current = current / component
        try:
            _validate_real_directory(current, location)
        except SecFilingGemmaRevealStoreError as exc:
            if not create or current.exists() or current.is_symlink():
                raise
            try:
                current.mkdir()
            except FileExistsError:
                pass
            _validate_real_directory(current, location)
    return absolute


def _validate_regular_details(details: os.stat_result, location: str) -> None:
    if (
        stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
    ):
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be a single-link regular file, never a link or "
            "reparse point"
        )


def _read_regular_bytes(
    path: Path,
    location: str,
    *,
    max_bytes: int = MAX_TRACKED_ANCHOR_FILE_BYTES,
) -> bytes:
    if type(max_bytes) is not int or max_bytes < 1:
        raise SecFilingGemmaRevealStoreError("Regular-file read cap is invalid")
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise SecFilingGemmaRevealStoreError(f"{location} does not exist") from exc
    _validate_regular_details(before, location)
    if before.st_size > max_bytes:
        raise SecFilingGemmaRevealStoreError(
            f"{location} exceeds the {max_bytes}-byte safety limit"
        )
    flags = os.O_RDONLY | _BINARY | _NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SecFilingGemmaRevealStoreError(
            f"Could not securely open {location}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        _validate_regular_details(opened, location)
        if opened.st_size > max_bytes:
            raise SecFilingGemmaRevealStoreError(
                f"{location} exceeds the {max_bytes}-byte safety limit"
            )
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise SecFilingGemmaRevealStoreError(
                f"{location} identity changed while opening"
            )
        chunks: list[bytes] = []
        observed_size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - observed_size))
            if not chunk:
                break
            observed_size += len(chunk)
            if observed_size > max_bytes:
                raise SecFilingGemmaRevealStoreError(
                    f"{location} exceeded the {max_bytes}-byte safety limit while reading"
                )
            chunks.append(chunk)
        after = path.lstat()
        _validate_regular_details(after, location)
        if (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino):
            raise SecFilingGemmaRevealStoreError(
                f"{location} identity changed while reading"
            )
        if after.st_size > max_bytes or after.st_size != observed_size:
            raise SecFilingGemmaRevealStoreError(
                f"{location} size changed or exceeded its limit while reading"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _create_or_replay_regular_bytes(
    path: Path,
    payload: bytes,
    location: str,
    *,
    max_bytes: int,
    recover_interrupted_precommit: bool = False,
) -> bytes:
    """Create one owned file exactly once, or replay its identical bytes."""

    if type(payload) is not bytes or not payload:
        raise SecFilingGemmaRevealStoreError(
            f"{location} payload must be non-empty exact bytes"
        )
    if type(max_bytes) is not int or max_bytes < 1 or len(payload) > max_bytes:
        raise SecFilingGemmaRevealStoreError(
            f"{location} payload exceeds its fixed safety limit"
        )
    try:
        existing_details = path.lstat()
    except FileNotFoundError:
        existing_details = None
    if type(recover_interrupted_precommit) is not bool:
        raise SecFilingGemmaRevealStoreError(
            f"{location} recovery policy must be exact"
        )
    if existing_details is not None:
        _validate_regular_details(existing_details, location)
        existing = _read_regular_bytes(path, location, max_bytes=max_bytes)
        if existing != payload:
            if not recover_interrupted_precommit:
                raise SecFilingGemmaRevealStoreError(
                    f"{location} differs from the exact owned replay"
                )
            path.unlink()
            _fsync_directory(path.parent)
        else:
            return existing

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY | _NOFOLLOW
    descriptor: int | None = None
    completed = False
    created_here = False
    try:
        descriptor = os.open(path, flags, 0o600)
        created_here = True
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise SecFilingGemmaRevealStoreError(
                    f"Could not finish writing {location}"
                )
            remaining = remaining[written:]
        os.fsync(descriptor)
        details = os.fstat(descriptor)
        _validate_regular_details(details, location)
        if details.st_size != len(payload):
            raise SecFilingGemmaRevealStoreError(
                f"{location} size differs from its exact owned bytes"
            )
        completed = True
    except FileExistsError:
        # Another owner cannot legitimately race while the store lock is held,
        # but fail by replaying rather than ever replacing an existing path.
        existing = _read_regular_bytes(path, location, max_bytes=max_bytes)
        if existing != payload:
            raise SecFilingGemmaRevealStoreError(
                f"{location} raced with different durable bytes"
            )
        return existing
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created_here and not completed:
            try:
                details = path.lstat()
            except FileNotFoundError:
                details = None
            if details is not None:
                try:
                    _validate_regular_details(details, location)
                    path.unlink()
                except Exception:
                    pass
    _fsync_directory(path.parent)
    observed = _read_regular_bytes(path, location, max_bytes=max_bytes)
    if observed != payload:
        raise SecFilingGemmaRevealStoreError(
            f"{location} changed after its create-new write"
        )
    return observed


def _execution_source_hashes(repository_root: Path) -> dict[str, str]:
    """Hash every resolved candidate source and reject loaded-path substitution."""

    root = repository_root.resolve(strict=True)
    observed: dict[str, str] = {}
    for role, relative_path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS:
        expected_path = repository_root / Path(relative_path)
        try:
            resolved_path = expected_path.resolve(strict=True)
            resolved_path.relative_to(root)
        except Exception:
            raise SecFilingGemmaRevealStoreError(
                "SEC execution source path escaped its canonical repository"
            ) from None
        module_name = (
            "agent_benchmark"
            if relative_path == "agent_benchmark/__init__.py"
            else relative_path[:-3].replace("/", ".")
        )
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            loaded_file = getattr(loaded, "__file__", None)
            try:
                loaded_path = Path(loaded_file).resolve(strict=True)
            except Exception:
                raise SecFilingGemmaRevealStoreError(
                    "Loaded SEC execution module has no canonical source path"
                ) from None
            if loaded_path != resolved_path:
                raise SecFilingGemmaRevealStoreError(
                    "Loaded SEC execution module differs from its repository source"
                )
        payload = _read_regular_bytes(
            expected_path,
            f"SEC execution source {role}",
            max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
        )
        observed[role] = hashlib.sha256(payload).hexdigest()
    return observed


def _model_execution_source_hashes(repository_root: Path) -> dict[str, str]:
    """Return only the exact candidate sources that may execute Gemma."""

    all_sources = _execution_source_hashes(repository_root)
    try:
        observed = {role: all_sources[role] for role in MODEL_EXECUTION_SOURCE_ROLES}
    except KeyError as exc:
        raise SecFilingGemmaRevealStoreError(
            "A required model execution source has no canonical repository path"
        ) from exc
    if set(observed) != set(MODEL_EXECUTION_SOURCE_ROLES):
        raise SecFilingGemmaRevealStoreError(
            "Model execution source roles changed"
        )
    return observed


def _market_execution_source_hashes(repository_root: Path) -> dict[str, str]:
    """Return the exact candidate sources authorized for market acquisition."""

    all_sources = _execution_source_hashes(repository_root)
    try:
        observed = {
            role: all_sources[role] for role in MARKET_EXECUTION_SOURCE_ROLES
        }
    except KeyError as exc:
        raise SecFilingGemmaRevealStoreError(
            "A required market execution source has no canonical repository path"
        ) from exc
    if set(observed) != set(MARKET_EXECUTION_SOURCE_ROLES):
        raise SecFilingGemmaRevealStoreError(
            "Market execution source roles changed"
        )
    return observed


def _validated_development_root_plan_for_store(
    authenticated_store_snapshot: Mapping[str, Any],
    development_content_root_plan: Any,
    *,
    require_latest_candidate: bool,
) -> dict[str, Any]:
    """Rebuild one request-free root plan from its registered candidate.

    The embedded universe is evidence, not caller selection authority.  Every
    URL, budget, and scope field is deterministically reconstructed from the
    exact candidate entry already authenticated by the reveal-store state.
    """

    state = _expect_mapping(
        authenticated_store_snapshot,
        "development-root authenticated store snapshot",
    )
    plan = _exact_caller_dict(
        development_content_root_plan,
        "development content root plan",
    )
    root_scope = _expect_mapping(
        plan.get("root_scope"),
        "development content root plan scope",
    )
    candidate_hash = _sha256(
        root_scope.get("candidate_sha256"),
        "development content root candidate hash",
    )
    registry = _expect_mapping(
        state.get("latest_registry"),
        "development-root latest registry",
    )
    entries = registry.get("entries")
    if type(entries) is not list or not entries:
        raise SecFilingGemmaRevealStoreError(
            "Development-content root requires a registered candidate"
        )
    matching = [
        entry
        for entry in entries
        if type(entry) is dict and entry.get("candidate_sha256") == candidate_hash
    ]
    if len(matching) != 1:
        raise SecFilingGemmaRevealStoreError(
            "Development-content root candidate is not uniquely registered"
        )
    entry = matching[0]
    if require_latest_candidate and entry != entries[-1]:
        raise SecFilingGemmaRevealStoreError(
            "Development-content root must belong to the latest registered candidate"
        )
    candidate = _expect_mapping(
        entry.get("candidate_manifest"),
        "development-root registered candidate manifest",
    )
    bindings = _expect_mapping(
        candidate.get("bindings"),
        "development-root candidate bindings",
    )
    universe = _expect_mapping(
        plan.get("corpus_universe_manifest"),
        "development-root embedded corpus universe",
    )
    try:
        expected = build_development_content_root_plan(
            candidate_manifest=candidate,
            expected_candidate_sha256=entry["candidate_sha256"],
            expected_candidate_design_sha256=entry["candidate_design_sha256"],
            expected_attempt_id=entry["attempt_id"],
            base_corpus_universe_sha256=bindings["corpus_universe_sha256"],
            corpus_universe_manifest=universe,
            session_calendar_sha256=bindings["calendar_sessions_sha256"],
        )
        validate_development_content_root_plan(
            plan,
            expected_development_content_root_plan_sha256=plan[
                "development_content_root_plan_sha256"
            ],
            candidate_manifest=candidate,
            expected_candidate_sha256=entry["candidate_sha256"],
            expected_candidate_design_sha256=entry["candidate_design_sha256"],
            expected_attempt_id=entry["attempt_id"],
            base_corpus_universe_sha256=bindings["corpus_universe_sha256"],
            corpus_universe_manifest=universe,
            session_calendar_sha256=bindings["calendar_sessions_sha256"],
        )
    except (KeyError, SecFilingGemmaStageAccessError) as exc:
        raise SecFilingGemmaRevealStoreError(
            "Development-content root plan is not the exact registered-candidate construction"
        ) from exc
    if plan != expected:
        raise SecFilingGemmaRevealStoreError(
            "Development-content root plan changed after deterministic reconstruction"
        )
    return expected


def _read_owned_sec_indexed_payloads(
    component_directory: Path,
    raw_index: Any,
    *,
    location: str,
    maximum_files: int = MAX_SEC_BATCH_FILES,
    maximum_file_bytes: int = MAX_SEC_BATCH_FILE_BYTES,
    maximum_total_bytes: int = MAX_SEC_BATCH_TOTAL_BYTES,
    complete_marker_filename: str = SEC_BATCH_COMPLETE_MARKER_FILENAME,
    maximum_file_bytes_by_name: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Rehash one flat create-new SEC payload directory exactly."""

    secured = _secure_directory(
        component_directory,
        create=False,
        location=location,
    )
    if (
        type(raw_index) is not list
        or not raw_index
        or len(raw_index) + 1 > maximum_files
    ):
        raise SecFilingGemmaRevealStoreError(
            f"{location} complete marker has no byte index"
        )
    observed_names: list[str] = []
    with os.scandir(secured) as entries:
        for item in entries:
            observed_names.append(item.name)
            if len(observed_names) > maximum_files:
                raise SecFilingGemmaRevealStoreError(
                    f"{location} contains too many files"
                )
    if len({name.casefold() for name in observed_names}) != len(observed_names):
        raise SecFilingGemmaRevealStoreError(
            f"{location} contains a case-colliding file"
        )
    byte_index: list[dict[str, Any]] = []
    payloads_by_name: dict[str, bytes] = {}
    total_bytes = 0
    expected_names = {complete_marker_filename}
    for ordinal, raw_item in enumerate(raw_index, start=1):
        if type(raw_item) is not dict or set(raw_item) != {
            "ordinal",
            "logical_id",
            "relative_path",
            "byte_count",
            "sha256",
        }:
            raise SecFilingGemmaRevealStoreError(
                f"{location} byte-index item is not exact"
            )
        relative = raw_item["relative_path"]
        byte_count = raw_item["byte_count"]
        file_cap = maximum_file_bytes
        if maximum_file_bytes_by_name is not None:
            file_cap = maximum_file_bytes_by_name.get(relative, file_cap)
        if (
            isinstance(raw_item["ordinal"], bool)
            or type(raw_item["ordinal"]) is not int
            or raw_item["ordinal"] != ordinal
            or type(raw_item["logical_id"]) is not str
            or not raw_item["logical_id"]
            or type(relative) is not str
            or not relative
            or "/" in relative
            or "\\" in relative
            or relative in {".", "..", complete_marker_filename}
            or isinstance(byte_count, bool)
            or type(byte_count) is not int
            or byte_count < 1
            or type(file_cap) is not int
            or file_cap < 1
            or byte_count > file_cap
        ):
            raise SecFilingGemmaRevealStoreError(
                f"{location} byte-index path is unsafe"
            )
        _sha256(raw_item["sha256"], f"{location} byte-index hash")
        if relative in payloads_by_name:
            raise SecFilingGemmaRevealStoreError(
                f"{location} byte-index path is duplicated"
            )
        payload = _read_regular_bytes(
            secured / relative,
            f"{location} byte {relative}",
            max_bytes=byte_count,
        )
        total_bytes += len(payload)
        if total_bytes > maximum_total_bytes:
            raise SecFilingGemmaRevealStoreError(
                f"{location} exceeds its total byte limit"
            )
        observed = {
            "ordinal": ordinal,
            "logical_id": raw_item["logical_id"],
            "relative_path": relative,
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if observed != raw_item:
            raise SecFilingGemmaRevealStoreError(
                f"{location} bytes differ from their complete marker"
            )
        byte_index.append(observed)
        payloads_by_name[relative] = payload
        expected_names.add(relative)
    if set(observed_names) != expected_names:
        raise SecFilingGemmaRevealStoreError(
            f"{location} has missing or extra durable bytes"
        )
    return byte_index, payloads_by_name


def _expected_market_byte_layout() -> list[tuple[str, str]]:
    layout: list[tuple[str, str]] = []
    for symbol in MARKET_SYMBOLS:
        layout.append(
            (f"raw-response-{symbol}", f"raw-response-{symbol}.json")
        )
    for symbol in MARKET_SYMBOLS:
        layout.append((f"artifact-{symbol}", f"artifact-{symbol}.json"))
    for symbol in MARKET_SYMBOLS:
        layout.append((f"window-{symbol}", f"window-{symbol}.json"))
    layout.extend(
        (
            ("source-manifest", "source-manifest.json"),
            ("stage-manifest", "stage-manifest.json"),
            ("reconciliation-receipt", "reconciliation-receipt.json"),
            ("acquisition-receipt", "acquisition-receipt.json"),
        )
    )
    if len(layout) + 1 != MAX_MARKET_BATCH_FILES:
        raise SecFilingGemmaRevealStoreError(
            "Owned development market layout count changed"
        )
    return layout


def _prevalidate_owned_market_byte_index(
    secured: Path,
    marker: Mapping[str, Any],
    *,
    marker_byte_count: int,
) -> list[dict[str, Any]]:
    """Reject an oversized or non-exact market component before payload reads."""

    raw_index = marker.get("byte_index")
    expected_layout = _expected_market_byte_layout()
    if type(raw_index) is not list or len(raw_index) != len(expected_layout):
        raise SecFilingGemmaRevealStoreError(
            "Owned development market byte index lacks its exact entry count"
        )
    byte_index: list[dict[str, Any]] = []
    byte_count_total = 0
    raw_response_total = 0
    exact_keys = {
        "ordinal",
        "logical_id",
        "relative_path",
        "byte_count",
        "sha256",
    }
    for ordinal, (raw_item, expected_item) in enumerate(
        zip(raw_index, expected_layout, strict=True),
        start=1,
    ):
        if type(raw_item) is not dict or set(raw_item) != exact_keys:
            raise SecFilingGemmaRevealStoreError(
                "Owned development market byte-index item is not exact"
            )
        logical_id, relative_path = expected_item
        byte_count = raw_item.get("byte_count")
        exact_file_cap = (
            YAHOO_MAX_RESPONSE_BYTES
            if ordinal <= len(MARKET_SYMBOLS)
            else MAX_MARKET_BATCH_FILE_BYTES
        )
        if (
            isinstance(raw_item.get("ordinal"), bool)
            or type(raw_item.get("ordinal")) is not int
            or raw_item.get("ordinal") != ordinal
            or raw_item.get("logical_id") != logical_id
            or raw_item.get("relative_path") != relative_path
            or isinstance(byte_count, bool)
            or type(byte_count) is not int
            or byte_count < 1
            or byte_count > exact_file_cap
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned development market byte index differs from its exact layout"
            )
        _sha256(raw_item.get("sha256"), "owned development market byte hash")
        byte_count_total += byte_count
        if byte_count_total > MAX_MARKET_BATCH_TOTAL_BYTES:
            raise SecFilingGemmaRevealStoreError(
                "Owned development market byte index exceeds its total byte limit"
            )
        if ordinal <= len(MARKET_SYMBOLS):
            raw_response_total += byte_count
            if raw_response_total > YAHOO_MAX_TOTAL_RESPONSE_BYTES:
                raise SecFilingGemmaRevealStoreError(
                    "Owned development market raw-response index exceeds its aggregate "
                    "byte ceiling"
                )
        try:
            file_details = (secured / relative_path).lstat()
        except FileNotFoundError as exc:
            raise SecFilingGemmaRevealStoreError(
                "Owned development market byte-index file does not exist"
            ) from exc
        _validate_regular_details(
            file_details,
            f"owned development market byte {relative_path}",
        )
        if file_details.st_size != byte_count:
            raise SecFilingGemmaRevealStoreError(
                "Owned development market file size differs from its byte index"
            )
        byte_index.append(dict(raw_item))

    if (
        isinstance(marker.get("byte_count_total"), bool)
        or type(marker.get("byte_count_total")) is not int
        or marker.get("byte_index_sha256") != canonical_sha256(byte_index)
        or marker.get("byte_count_total") != byte_count_total
        or isinstance(marker_byte_count, bool)
        or type(marker_byte_count) is not int
        or marker_byte_count < 1
        or marker_byte_count + byte_count_total > MAX_MARKET_BATCH_TOTAL_BYTES
    ):
        raise SecFilingGemmaRevealStoreError(
            "Owned development market byte index is inconsistent"
        )
    observed_names: list[str] = []
    with os.scandir(secured) as entries:
        for item in entries:
            observed_names.append(item.name)
            if len(observed_names) > MAX_MARKET_BATCH_FILES:
                raise SecFilingGemmaRevealStoreError(
                    "Owned development market component contains too many files"
                )
    expected_names = {
        DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME,
        *(relative_path for _logical_id, relative_path in expected_layout),
    }
    if (
        len(observed_names) != MAX_MARKET_BATCH_FILES
        or len({name.casefold() for name in observed_names}) != len(observed_names)
        or set(observed_names) != expected_names
    ):
        raise SecFilingGemmaRevealStoreError(
            "Owned development market component lacks its exact flat layout"
        )
    return byte_index


_MODEL_EVENT_FILE_ITEMS: Final[tuple[tuple[str, str], ...]] = (
    ("preprocessed-event", "preprocessed_event.json"),
    ("preprocessing-receipt", "preprocessing_receipt.json"),
    ("redacted-input-manifest", "redacted_input_manifest.json"),
    ("call-intent", "call_intent.json"),
    ("model-attempt-receipt", "model_attempt_receipt.json"),
)


def _expected_model_byte_layout(event_count: int) -> list[tuple[str, str]]:
    if isinstance(event_count, bool) or type(event_count) is not int or event_count < 1:
        raise SecFilingGemmaRevealStoreError(
            "Owned model claim has no positive exact event count"
        )
    layout: list[tuple[str, str]] = [
        ("pre-runtime-probe", "pre_runtime_probe.json")
    ]
    for event_ordinal in range(1, event_count + 1):
        event_name = f"event-{event_ordinal:06d}"
        event_directory = f"events/{event_ordinal:06d}"
        layout.extend(
            (
                (f"{event_name}-{logical_suffix}", f"{event_directory}/{filename}")
                for logical_suffix, filename in _MODEL_EVENT_FILE_ITEMS
            )
        )
    layout.extend(
        (
            ("post-runtime-probe", "post_runtime_probe.json"),
            ("runtime-guard", "runtime_guard.json"),
        )
    )
    if len(layout) > MAX_SEC_BATCH_FILES:
        raise SecFilingGemmaRevealStoreError(
            "Owned model event layout exceeds its fixed file limit"
        )
    return layout


def _build_owned_model_call_intent(
    *,
    claim: Mapping[str, Any],
    event: Mapping[str, Any],
    preprocessed_event_sha256: str,
    owned_preprocessing_receipt_sha256: str,
    redacted_input_manifest_sha256: str,
    model_payload_sha256: str,
    runtime_probe_receipt_sha256: str,
    runtime_evidence_sha256: str,
) -> dict[str, Any]:
    """Build the create-new intent that must precede one irreversible model call."""

    claim_value = _exact_caller_dict(dict(claim), "owned model call-intent claim")
    event_value = _exact_caller_dict(dict(event), "owned model call-intent event")
    if set(event_value) != {
        "event_ordinal",
        "accession_number",
        "form",
        "availability_session",
        "sec_document_ordinal",
    }:
        raise SecFilingGemmaRevealStoreError(
            "Owned model call-intent event keys changed"
        )
    event_ordinal = _strict_int(
        event_value["event_ordinal"],
        "owned model call-intent event ordinal",
        minimum=1,
    )
    if event_ordinal > claim_value.get("event_count", 0):
        raise SecFilingGemmaRevealStoreError(
            "Owned model call-intent event exceeds the claim"
        )
    stage = claim_value.get("authorized_stage")
    if stage == "development":
        scope_kind = "development_root"
        scope_hash = _sha256(
            claim_value.get("development_root_scope_sha256"),
            "owned model call-intent development scope",
        )
    elif stage in {"intermediate", "final"}:
        scope_kind = "stage_request"
        scope_hash = _sha256(
            claim_value.get("request_sha256"),
            "owned model call-intent stage scope",
        )
    else:
        raise SecFilingGemmaRevealStoreError(
            "Owned model call-intent claim stage changed"
        )
    body = {
        "schema_version": MODEL_CALL_INTENT_SCHEMA_VERSION,
        "scope_kind": scope_kind,
        "scope_sha256": scope_hash,
        "claim_sha256": _sha256(
            claim_value.get("claim_sha256"),
            "owned model call-intent claim hash",
        ),
        "candidate_sha256": _sha256(
            claim_value.get("candidate_sha256"),
            "owned model call-intent candidate hash",
        ),
        "stage": stage,
        "event_ordinal": event_ordinal,
        "call_sequence": event_ordinal,
        "accession_number": event_value["accession_number"],
        "form": event_value["form"],
        "availability_session": event_value["availability_session"],
        "sec_document_ordinal": event_value["sec_document_ordinal"],
        "preprocessed_event_sha256": _sha256(
            preprocessed_event_sha256,
            "owned model call-intent preprocessed event hash",
        ),
        "owned_preprocessing_receipt_sha256": _sha256(
            owned_preprocessing_receipt_sha256,
            "owned model call-intent preprocessing receipt hash",
        ),
        "redacted_input_manifest_sha256": _sha256(
            redacted_input_manifest_sha256,
            "owned model call-intent redacted manifest hash",
        ),
        "model_payload_sha256": _sha256(
            model_payload_sha256,
            "owned model call-intent model payload hash",
        ),
        "runtime_probe_receipt_sha256": _sha256(
            runtime_probe_receipt_sha256,
            "owned model call-intent runtime probe receipt hash",
        ),
        "runtime_evidence_sha256": _sha256(
            runtime_evidence_sha256,
            "owned model call-intent runtime evidence hash",
        ),
        "model_name": claim_value.get("model_name"),
        "model_digest": _sha256(
            claim_value.get("model_digest"),
            "owned model call-intent model digest",
        ),
        "runtime_fingerprint_sha256": _sha256(
            claim_value.get("runtime_fingerprint_sha256"),
            "owned model call-intent runtime fingerprint",
        ),
        "model_transport_sha256": _sha256(
            claim_value.get("model_transport_sha256"),
            "owned model call-intent transport hash",
        ),
        "transport_mode": OWNED_MODEL_ATTEMPT_TRANSPORT_MODE,
        "external_effect_started": False,
        "effect_retry_permitted_after_intent": False,
    }
    return {**body, "call_intent_sha256": canonical_sha256(body)}


def _read_owned_model_indexed_payloads(
    component_directory: Path,
    raw_index: Any,
    *,
    event_count: int,
    location: str,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Rehash the exact hierarchical model artifact layout without traversal."""

    secured = _secure_directory(
        component_directory,
        create=False,
        location=location,
    )
    expected_layout = _expected_model_byte_layout(event_count)
    if type(raw_index) is not list or len(raw_index) != len(expected_layout):
        raise SecFilingGemmaRevealStoreError(
            f"{location} byte index differs from its fixed event layout"
        )
    top_level_names = [item.name for item in secured.iterdir()]
    expected_top_level = {
        "pre_runtime_probe.json",
        "post_runtime_probe.json",
        "runtime_guard.json",
        MODEL_EXTRACTION_COMPLETE_MARKER_FILENAME,
        "events",
    }
    if (
        set(top_level_names) != expected_top_level
        or len(top_level_names) != len(expected_top_level)
        or len({name.casefold() for name in top_level_names})
        != len(top_level_names)
    ):
        raise SecFilingGemmaRevealStoreError(
            f"{location} has missing, extra, or case-colliding paths"
        )
    events_directory = _secure_directory(
        secured / "events",
        create=False,
        location=f"{location} events directory",
    )
    expected_event_directories = {
        f"{ordinal:06d}" for ordinal in range(1, event_count + 1)
    }
    observed_event_directories = [item.name for item in events_directory.iterdir()]
    if (
        set(observed_event_directories) != expected_event_directories
        or len(observed_event_directories) != len(expected_event_directories)
        or len({name.casefold() for name in observed_event_directories})
        != len(observed_event_directories)
    ):
        raise SecFilingGemmaRevealStoreError(
            f"{location} event directories differ from the claim"
        )
    expected_event_files = {filename for _suffix, filename in _MODEL_EVENT_FILE_ITEMS}
    event_directories: dict[str, Path] = {}
    for event_name in sorted(expected_event_directories):
        event_directory = _secure_directory(
            events_directory / event_name,
            create=False,
            location=f"{location} event {event_name}",
        )
        observed_names = [item.name for item in event_directory.iterdir()]
        if (
            set(observed_names) != expected_event_files
            or len(observed_names) != len(expected_event_files)
            or len({name.casefold() for name in observed_names})
            != len(observed_names)
        ):
            raise SecFilingGemmaRevealStoreError(
                f"{location} event {event_name} differs from its fixed artifacts"
            )
        event_directories[event_name] = event_directory
    byte_index: list[dict[str, Any]] = []
    payloads_by_name: dict[str, bytes] = {}
    total_bytes = 0
    for ordinal, ((logical_id, relative_path), raw_item) in enumerate(
        zip(expected_layout, raw_index),
        start=1,
    ):
        if type(raw_item) is not dict or set(raw_item) != {
            "ordinal",
            "logical_id",
            "relative_path",
            "byte_count",
            "sha256",
        }:
            raise SecFilingGemmaRevealStoreError(
                f"{location} byte-index item is not exact"
            )
        if "/" in relative_path:
            _events, event_name, filename = relative_path.split("/")
            path = event_directories[event_name] / filename
        else:
            path = secured / relative_path
        payload = _read_regular_bytes(
            path,
            f"{location} byte {relative_path}",
            max_bytes=MAX_SEC_BATCH_FILE_BYTES,
        )
        total_bytes += len(payload)
        if total_bytes > MAX_SEC_BATCH_TOTAL_BYTES:
            raise SecFilingGemmaRevealStoreError(
                f"{location} exceeds its total byte limit"
            )
        parsed = _strict_json_bytes(payload, f"{location} byte {relative_path}")
        if type(parsed) is not dict or payload != _encoded_state(parsed):
            raise SecFilingGemmaRevealStoreError(
                f"{location} byte {relative_path} is not canonical JSON"
            )
        observed = {
            "ordinal": ordinal,
            "logical_id": logical_id,
            "relative_path": relative_path,
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if observed != raw_item:
            raise SecFilingGemmaRevealStoreError(
                f"{location} bytes differ from their complete marker"
            )
        byte_index.append(observed)
        payloads_by_name[relative_path] = payload
    return byte_index, payloads_by_name


def _git(repo_root: Path, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SecFilingGemmaRevealStoreError(
            "Could not authenticate reveal-store anchors through local Git"
        ) from exc
    if completed.returncode != 0:
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store anchors must exist in the current Git HEAD"
        )
    return completed.stdout


def _tracked_head_bytes(repo_root: Path, relative_name: str) -> bytes:
    relative = PurePosixPath(relative_name)
    windows = PureWindowsPath(relative_name)
    if (
        relative.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in relative.parts
        or len(relative.parts) < 1
    ):
        raise SecFilingGemmaRevealStoreError("Tracked anchor path is unsafe")

    top_level_raw = _git(repo_root, "rev-parse", "--show-toplevel").strip()
    try:
        top_level = Path(os.path.abspath(os.fsdecode(top_level_raw)))
    except UnicodeError as exc:
        raise SecFilingGemmaRevealStoreError(
            "Git repository root is not a valid local path"
        ) from exc
    if os.path.normcase(str(top_level)) != os.path.normcase(str(repo_root)):
        raise SecFilingGemmaRevealStoreError(
            "repository_root must be the exact Git worktree root"
        )

    blob = _git(repo_root, "cat-file", "blob", f"HEAD:{relative.as_posix()}")
    worktree_path = repo_root.joinpath(*relative.parts)
    observed = _read_regular_bytes(
        worktree_path, f"tracked anchor {relative.as_posix()}"
    )
    # Git may materialize tracked JSON with platform line endings.  Authenticate
    # the strict JSON value rather than requiring byte-identical checkout EOLs.
    try:
        observed_value = _strict_json_bytes(
            observed, f"tracked anchor {relative.as_posix()}"
        )
        head_value = _strict_json_bytes(
            blob, f"Git HEAD anchor {relative.as_posix()}"
        )
        same_value = _same_digest(
            canonical_sha256(observed_value), canonical_sha256(head_value)
        )
    except SecFilingGemmaContractError as exc:
        raise SecFilingGemmaRevealStoreError(
            f"Tracked anchor {relative.as_posix()} is not finite canonical JSON"
        ) from exc
    if not same_value:
        raise SecFilingGemmaRevealStoreError(
            f"Tracked anchor {relative.as_posix()} does not match Git HEAD"
        )
    return blob


def _load_tracked_anchor(repo_root: Path) -> dict[str, Any]:
    pin_payload = _strict_json_bytes(
        _tracked_head_bytes(repo_root, INITIAL_PIN_RELATIVE_PATH),
        "initial registry-pin artifact",
    )
    pin = _expect_mapping(pin_payload, "initial registry-pin artifact")
    _expect_keys(
        pin,
        {
            "schema_version",
            "contract_version",
            "migration_source_canonical_sha256",
            "registry_pin",
        },
        "initial registry-pin artifact",
    )
    if pin["schema_version"] != INITIAL_PIN_SCHEMA_VERSION:
        raise SecFilingGemmaRevealStoreError("Unknown initial registry-pin schema")
    if pin["contract_version"] != CONTRACT_VERSION:
        raise SecFilingGemmaRevealStoreError(
            "Initial registry pin belongs to another experiment contract"
        )

    declaration = historical_reveal_declaration()
    migration_name = declaration["migration_source"]
    if not isinstance(migration_name, str):
        raise SecFilingGemmaRevealStoreError("Migration source path is invalid")
    migration_payload = _strict_json_bytes(
        _tracked_head_bytes(repo_root, migration_name),
        "historical migration snapshot",
    )
    try:
        migration_hash = canonical_sha256(migration_payload)
    except SecFilingGemmaContractError as exc:
        raise SecFilingGemmaRevealStoreError(
            "Historical migration snapshot is not canonical finite JSON"
        ) from exc
    expected_migration_hash = declaration["migration_source_canonical_sha256"]
    if (
        not isinstance(expected_migration_hash, str)
        or not _same_digest(expected_migration_hash, HISTORICAL_REGISTRY_CANONICAL_SHA256)
        or not _same_digest(migration_hash, expected_migration_hash)
        or pin["migration_source_canonical_sha256"] != expected_migration_hash
    ):
        raise SecFilingGemmaRevealStoreError(
            "Tracked migration snapshot canonical hash does not match the frozen anchor"
        )

    initial_registry = build_initial_reveal_registry()
    expected_pin = derive_registry_pin(initial_registry)
    if pin["registry_pin"] != expected_pin:
        raise SecFilingGemmaRevealStoreError(
            "Tracked initial pin does not authenticate the deterministic genesis registry"
        )
    pin_copy = _json_value_copy(dict(pin), "initial registry-pin artifact")
    return {
        "schema_version": STORE_ANCHOR_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "initial_pin_relative_path": INITIAL_PIN_RELATIVE_PATH,
        "initial_pin_artifact_canonical_sha256": canonical_sha256(pin_copy),
        "migration_source": migration_name,
        "migration_source_canonical_sha256": migration_hash,
        "initial_registry_sha256": initial_registry["registry_sha256"],
        "initial_registry_pin": expected_pin,
        "historical_final_reveal_count_lower_bound": (
            HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND
        ),
    }


class _ExclusiveFileLock:
    def __init__(self, path: Path, *, timeout_seconds: float) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.descriptor: int | None = None

    def __enter__(self) -> "_ExclusiveFileLock":
        flags = os.O_RDWR | os.O_CREAT | _BINARY | _NOFOLLOW
        before: os.stat_result | None = None
        try:
            before = self.path.lstat()
            _validate_regular_details(before, "reveal-store lock")
        except FileNotFoundError:
            pass
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise SecFilingGemmaRevealStoreError(
                "Could not securely open the reveal-store lock"
            ) from exc
        try:
            details = os.fstat(descriptor)
            _validate_regular_details(details, "reveal-store lock")
            after = self.path.lstat()
            _validate_regular_details(after, "reveal-store lock")
            if (
                (details.st_dev, details.st_ino) != (after.st_dev, after.st_ino)
                or before is not None
                and (before.st_dev, before.st_ino)
                != (details.st_dev, details.st_ino)
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Reveal-store lock identity changed while opening"
                )
            if details.st_size == 0:
                os.write(descriptor, b"0")
                os.fsync(descriptor)
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt

                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise SecFilingGemmaRevealStoreError(
                            "Timed out acquiring the exclusive reveal-store lock"
                        ) from exc
                    time.sleep(0.025)
            self.descriptor = descriptor
            return self
        except Exception:
            os.close(descriptor)
            raise

    def __exit__(self, *args: Any) -> None:
        descriptor, self.descriptor = self.descriptor, None
        if descriptor is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | _BINARY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        # Windows commonly refuses directory fsync through the CRT.  The file
        # itself is always fsynced before replace; directory fsync is best effort.
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY | _NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise SecFilingGemmaRevealStoreError(
                    "Could not finish writing reveal-store temporary state"
                )
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None

        if path.exists() or path.is_symlink():
            _validate_regular_details(path.lstat(), "authoritative reveal-store state")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            details = temporary.lstat()
        except FileNotFoundError:
            return
        _validate_regular_details(details, "reveal-store temporary state")
        temporary.unlink()


def _cleanup_interrupted_temporaries(directory: Path) -> None:
    for entry in os.scandir(directory):
        if _TEMP_RE.fullmatch(entry.name) is None:
            continue
        path = Path(entry.path)
        details = path.lstat()
        _validate_regular_details(details, "interrupted reveal-store temporary state")
        path.unlink()
    _fsync_directory(directory)


def _consumption_genesis(anchor: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": "aapl-sec-gemma-consumed-request-genesis-v1",
            "contract_version": CONTRACT_VERSION,
            "store_anchor_sha256": canonical_sha256(anchor),
        }
    )


def _consumption_ledger(
    entries: list[dict[str, Any]], *, tip_sha256: str, anchor: Mapping[str, Any]
) -> dict[str, Any]:
    actual_final_touch_count = sum(
        1 for entry in entries if entry["stage"] == "final"
    )
    body = {
        "schema_version": CONSUMPTION_LEDGER_SCHEMA_VERSION,
        "entries": copy.deepcopy(entries),
        "chain": {
            "genesis_tip_sha256": _consumption_genesis(anchor),
            "tip_sha256": tip_sha256,
            "consumed_request_count": len(entries),
            "actual_final_touch_count": actual_final_touch_count,
            "historical_final_reveal_count_lower_bound": (
                HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND
            ),
            "repository_final_touch_count_lower_bound": (
                HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND
                + actual_final_touch_count
            ),
        },
    }
    return {**body, "ledger_sha256": canonical_sha256(body)}


def _initial_consumption_ledger(anchor: Mapping[str, Any]) -> dict[str, Any]:
    genesis = _consumption_genesis(anchor)
    return _consumption_ledger([], tip_sha256=genesis, anchor=anchor)


def _state_snapshot(
    *,
    anchor: Mapping[str, Any],
    latest_registry: Mapping[str, Any],
    latest_pin: Mapping[str, Any],
    consumption_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": STORE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "anchor": copy.deepcopy(dict(anchor)),
        "latest_registry": copy.deepcopy(dict(latest_registry)),
        "latest_registry_pin": copy.deepcopy(dict(latest_pin)),
        "consumption_ledger": copy.deepcopy(dict(consumption_ledger)),
    }
    return {**body, "state_sha256": canonical_sha256(body)}


def _pending_current_tip_document(
    *,
    prior_tip_anchor: Mapping[str, Any] | None,
    next_tip_anchor: Mapping[str, Any],
    authenticated_store_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prior = (
        None
        if prior_tip_anchor is None
        else _exact_caller_dict(dict(prior_tip_anchor), "prior current-tip anchor")
    )
    next_anchor = _exact_caller_dict(
        dict(next_tip_anchor), "next current-tip anchor"
    )
    if prior is None:
        validated_next = validate_reveal_store_current_tip_anchor_structure(
            next_anchor
        )
        if (
            validated_next["revision"] != 0
            or validated_next["previous_tip_anchor_sha256"] is not None
        ):
            raise SecFilingGemmaRevealStoreError(
                "Only the deterministic genesis may omit a prior current-tip anchor"
            )
    else:
        try:
            prior, validated_next = validate_reveal_store_current_tip_anchor_transition(
                prior,
                next_anchor,
                authenticated_store_snapshot=authenticated_store_snapshot,
            )
        except SecFilingGemmaStageAuthorizationError as exc:
            raise SecFilingGemmaRevealStoreError(
                "Current-tip transaction is not a valid monotonic CAS transition"
            ) from exc
    body = {
        "schema_version": CURRENT_TIP_PENDING_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "prior_tip_anchor": prior,
        "next_tip_anchor": validated_next,
    }
    return {**body, "pending_sha256": canonical_sha256(body)}


def _validated_pending_current_tip_document(
    raw: Any,
    *,
    authenticated_store_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        detached = detach_untrusted_stage_json(
            raw, "pending current-tip transaction"
        )
    except Exception as exc:
        raise SecFilingGemmaRevealStoreError(
            "Pending current-tip transaction exceeds fixed allocation bounds"
        ) from exc
    value = _expect_mapping(detached, "pending current-tip transaction")
    _expect_keys(
        value,
        {
            "schema_version",
            "contract_version",
            "prior_tip_anchor",
            "next_tip_anchor",
            "pending_sha256",
        },
        "pending current-tip transaction",
    )
    if (
        value["schema_version"] != CURRENT_TIP_PENDING_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
    ):
        raise SecFilingGemmaRevealStoreError(
            "Pending current-tip transaction schema or contract changed"
        )
    observed = _sha256(value["pending_sha256"], "pending current-tip hash")
    body = {key: value[key] for key in value if key != "pending_sha256"}
    if not _same_digest(observed, canonical_sha256(body)):
        raise SecFilingGemmaRevealStoreError(
            "Pending current-tip transaction hash is inconsistent"
        )
    try:
        prior_raw = value["prior_tip_anchor"]
        if prior_raw is None:
            next_anchor = validate_reveal_store_current_tip_anchor_structure(
                value["next_tip_anchor"]
            )
            if (
                next_anchor["revision"] != 0
                or next_anchor["previous_tip_anchor_sha256"] is not None
            ):
                raise SecFilingGemmaStageAuthorizationError(
                    "Non-genesis pending anchor has no predecessor"
                )
            prior = None
        else:
            prior, next_anchor = validate_reveal_store_current_tip_anchor_transition(
                prior_raw,
                value["next_tip_anchor"],
                authenticated_store_snapshot=authenticated_store_snapshot,
            )
    except SecFilingGemmaStageAuthorizationError as exc:
        raise SecFilingGemmaRevealStoreError(
            "Pending current-tip transaction is stale, forked, or invalid"
        ) from exc
    return {
        **body,
        "prior_tip_anchor": prior,
        "next_tip_anchor": next_anchor,
        "pending_sha256": observed,
    }


def _restore_pending_document(
    *,
    state_bytes: bytes,
    tip_anchor_bytes: bytes,
) -> dict[str, Any]:
    if (
        type(state_bytes) is not bytes
        or not state_bytes
        or len(state_bytes) > MAX_STATE_FILE_BYTES
        or type(tip_anchor_bytes) is not bytes
        or not tip_anchor_bytes
        or len(tip_anchor_bytes) > MAX_CURRENT_TIP_ANCHOR_FILE_BYTES
    ):
        raise SecFilingGemmaRevealStoreError(
            "Restore transaction target bytes exceed their safety limits"
        )
    body = {
        "schema_version": RESTORE_PENDING_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "state_bytes_base64": base64.b64encode(state_bytes).decode("ascii"),
        "state_bytes_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "state_byte_count": len(state_bytes),
        "tip_anchor_bytes_base64": base64.b64encode(tip_anchor_bytes).decode(
            "ascii"
        ),
        "tip_anchor_bytes_sha256": hashlib.sha256(
            tip_anchor_bytes
        ).hexdigest(),
        "tip_anchor_byte_count": len(tip_anchor_bytes),
    }
    document = {**body, "restore_pending_sha256": canonical_sha256(body)}
    if len(_encoded_state(document)) > MAX_RESTORE_PENDING_FILE_BYTES:
        raise SecFilingGemmaRevealStoreError(
            "Restore transaction document exceeds its safety limit"
        )
    return document


def _decode_restore_bytes(
    encoded: Any,
    *,
    expected_sha256: Any,
    expected_byte_count: Any,
    maximum_bytes: int,
    location: str,
) -> bytes:
    if type(encoded) is not str or not encoded:
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be nonempty canonical Base64"
        )
    expected_hash = _sha256(expected_sha256, f"{location} hash")
    byte_count = _strict_int(
        expected_byte_count, f"{location} byte count", minimum=1
    )
    if byte_count > maximum_bytes:
        raise SecFilingGemmaRevealStoreError(
            f"{location} exceeds its safety limit"
        )
    try:
        payload = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise SecFilingGemmaRevealStoreError(
            f"{location} is not strict Base64"
        ) from exc
    if (
        len(payload) != byte_count
        or base64.b64encode(payload).decode("ascii") != encoded
        or not _same_digest(hashlib.sha256(payload).hexdigest(), expected_hash)
    ):
        raise SecFilingGemmaRevealStoreError(
            f"{location} bytes do not match their exact recovery pin"
        )
    return payload


def _validated_restore_pending_document(
    raw: Any,
    *,
    tracked_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        detached = detach_untrusted_stage_json(
            raw, "reveal-store restore transaction"
        )
    except Exception as exc:
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store restore transaction exceeds fixed allocation bounds"
        ) from exc
    value = _expect_mapping(detached, "reveal-store restore transaction")
    _expect_keys(
        value,
        {
            "schema_version",
            "contract_version",
            "state_bytes_base64",
            "state_bytes_sha256",
            "state_byte_count",
            "tip_anchor_bytes_base64",
            "tip_anchor_bytes_sha256",
            "tip_anchor_byte_count",
            "restore_pending_sha256",
        },
        "reveal-store restore transaction",
    )
    if (
        value["schema_version"] != RESTORE_PENDING_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
    ):
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store restore transaction schema or contract changed"
        )
    observed = _sha256(
        value["restore_pending_sha256"], "restore transaction self-hash"
    )
    body = {
        key: value[key]
        for key in value
        if key != "restore_pending_sha256"
    }
    if not _same_digest(observed, canonical_sha256(body)):
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store restore transaction self-hash is inconsistent"
        )
    state_bytes = _decode_restore_bytes(
        value["state_bytes_base64"],
        expected_sha256=value["state_bytes_sha256"],
        expected_byte_count=value["state_byte_count"],
        maximum_bytes=MAX_STATE_FILE_BYTES,
        location="restore target state",
    )
    tip_bytes = _decode_restore_bytes(
        value["tip_anchor_bytes_base64"],
        expected_sha256=value["tip_anchor_bytes_sha256"],
        expected_byte_count=value["tip_anchor_byte_count"],
        maximum_bytes=MAX_CURRENT_TIP_ANCHOR_FILE_BYTES,
        location="restore target current-tip anchor",
    )
    try:
        target_state = detach_untrusted_stage_json(
            _strict_json_bytes(state_bytes, "restore target state"),
            "restore target state",
        )
        target_tip = detach_untrusted_stage_json(
            _strict_json_bytes(tip_bytes, "restore target current-tip anchor"),
            "restore target current-tip anchor",
        )
    except Exception as exc:
        raise SecFilingGemmaRevealStoreError(
            "Restore targets exceed fixed allocation bounds"
        ) from exc
    state = _validate_state(
        _expect_mapping(target_state, "restore target state"),
        expected_anchor=tracked_anchor,
    )
    tip_mapping = _expect_mapping(
        target_tip,
        "restore target current-tip anchor",
    )
    try:
        tip = validate_reveal_store_current_tip_anchor(state, tip_mapping)
    except SecFilingGemmaStageAuthorizationError as exc:
        raise SecFilingGemmaRevealStoreError(
            "Restore target tip does not authenticate its target state"
        ) from exc
    return {
        "state": state,
        "tip_anchor": tip,
        "state_bytes": state_bytes,
        "tip_anchor_bytes": tip_bytes,
        "restore_pending_sha256": observed,
    }


def _state_bytes_match_tip_anchor(
    state_bytes: bytes, state: Mapping[str, Any], tip_anchor: Mapping[str, Any]
) -> bool:
    return (
        tip_anchor["state_sha256"] == state.get("state_sha256")
        and tip_anchor["state_snapshot_byte_count"] == len(state_bytes)
        and _same_digest(
            tip_anchor["state_snapshot_bytes_sha256"],
            hashlib.sha256(state_bytes).hexdigest(),
        )
    )


def _authorization_bundle(
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    expected_new_consumption_entry_sha256: str,
) -> dict[str, Any]:
    snapshot = _exact_caller_dict(
        dict(authenticated_store_snapshot),
        "authenticated post-consumption store snapshot",
    )
    state_bytes = _encoded_state(snapshot)
    store_pin = derive_consumed_stage_store_state_pin(snapshot)
    grant = build_consumed_stage_authorization_grant(
        authenticated_store_snapshot=snapshot,
        external_store_state_pin=store_pin,
        expected_new_consumption_entry_sha256=(
            expected_new_consumption_entry_sha256
        ),
    )
    if (
        grant["store_snapshot_bytes_sha256"]
        != hashlib.sha256(state_bytes).hexdigest()
        or grant["store_snapshot_byte_count"] != len(state_bytes)
    ):
        raise SecFilingGemmaRevealStoreError(
            "Authorization grant does not bind the exact post-consumption bytes"
        )
    body = {
        "schema_version": CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION,
        "authenticated_store_snapshot": snapshot,
        "store_state_pin": store_pin,
        "authorization_grant": grant,
    }
    return {**body, "bundle_sha256": canonical_sha256(body)}


def _authenticated_store_verifier_context(
    *,
    trusted_stage_content_pin: Mapping[str, Any],
    trusted_stage_content_authentication: Mapping[str, Any],
    parent_consumption_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pin = _exact_caller_dict(
        dict(trusted_stage_content_pin),
        "authenticated verifier trusted content pin",
    )
    authentication = _exact_caller_dict(
        dict(trusted_stage_content_authentication),
        "authenticated verifier trusted content receipt",
    )
    parent = (
        None
        if parent_consumption_binding is None
        else _exact_caller_dict(
            dict(parent_consumption_binding),
            "authenticated verifier parent-consumption binding",
        )
    )
    body = {
        "schema_version": AUTHENTICATED_STORE_VERIFIER_CONTEXT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "context_kind": "reveal_store_authenticated_preconsumption_context",
        "trusted_stage_content_pin": pin,
        "trusted_stage_content_authentication": authentication,
        "parent_consumption_binding": parent,
    }
    return {**body, "authenticated_store_context_sha256": canonical_sha256(body)}


def _request_body_hash(request: Mapping[str, Any]) -> str:
    _expect_keys(request, set(_REQUEST_KEYS), "stored reveal request")
    observed = _sha256(request["request_sha256"], "stored request_sha256")
    body = {key: request[key] for key in request if key != "request_sha256"}
    expected = canonical_sha256(body)
    if not _same_digest(observed, expected):
        raise SecFilingGemmaRevealStoreError(
            "Stored reveal request is not its canonical hashed body"
        )
    if (
        request["schema_version"] != REVEAL_REQUEST_SCHEMA_VERSION
        or request["contract_version"] != CONTRACT_VERSION
        or request["stage"] not in {"intermediate", "final"}
        or request["authorizes_outcome_access"] is not False
        or request["effectful_atomic_single_use_consumption_required"] is not True
        or request["cross_attempt_comparison_permitted"] is not False
        or request["cross_attempt_winner_selection_permitted"] is not False
        or request["globally_pristine_claim"] is not False
    ):
        raise SecFilingGemmaRevealStoreError(
            "Stored request changed its non-authorizing single-candidate semantics"
        )
    return expected


@dataclass(frozen=True)
class SemanticPrerequisiteValidation:
    """Strongly bound result returned by an independent semantic validator.

    A truthy value or ad-hoc mapping is intentionally insufficient.  The
    caller must separately pin ``validator_id`` and ``validator_source_sha256``;
    the store verifies both and all stage/candidate/evidence bindings.
    """

    validator_id: str
    validator_source_sha256: str
    prerequisite_stage: str
    prerequisite_stage_evidence_sha256: str
    attempt_id: str
    candidate_sha256: str
    candidate_design_sha256: str
    registry_entry_sha256: str
    request_sha256: str
    requested_stage: str
    stage_access_manifest_sha256: str
    registry_sha256: str
    registry_tip_sha256: str
    semantic_checks: tuple[str, ...]
    semantic_receipt: Mapping[str, Any]

    @classmethod
    def success(
        cls,
        expected_context: Mapping[str, Any],
        *,
        validator_id: str,
        validator_source_sha256: str,
        semantic_checks: tuple[str, ...],
        semantic_receipt: Mapping[str, Any],
    ) -> "SemanticPrerequisiteValidation":
        context = _expect_mapping(expected_context, "prerequisite context")
        required = {
            "prerequisite_stage",
            "prerequisite_stage_evidence_sha256",
            "attempt_id",
            "candidate_sha256",
            "candidate_design_sha256",
            "registry_entry_sha256",
            "request_sha256",
            "stage",
            "stage_access_manifest_sha256",
            "registry_sha256",
            "registry_tip_sha256",
        }
        if not required <= set(context):
            raise SecFilingGemmaRevealStoreError(
                "Prerequisite context is missing immutable request bindings"
            )
        return cls(
            validator_id=validator_id,
            validator_source_sha256=validator_source_sha256,
            prerequisite_stage=context["prerequisite_stage"],
            prerequisite_stage_evidence_sha256=context[
                "prerequisite_stage_evidence_sha256"
            ],
            attempt_id=context["attempt_id"],
            candidate_sha256=context["candidate_sha256"],
            candidate_design_sha256=context["candidate_design_sha256"],
            registry_entry_sha256=context["registry_entry_sha256"],
            request_sha256=context["request_sha256"],
            requested_stage=context["stage"],
            stage_access_manifest_sha256=context[
                "stage_access_manifest_sha256"
            ],
            registry_sha256=context["registry_sha256"],
            registry_tip_sha256=context["registry_tip_sha256"],
            semantic_checks=semantic_checks,
            semantic_receipt=semantic_receipt,
        )

    def to_dict(self) -> dict[str, Any]:
        validator_id = _safe_id(self.validator_id, "semantic validator_id")
        source_hash = _sha256(
            self.validator_source_sha256, "semantic validator_source_sha256"
        )
        prerequisite_stage = self.prerequisite_stage
        if prerequisite_stage not in {"development", "intermediate"}:
            raise SecFilingGemmaRevealStoreError(
                "Semantic prerequisite stage is invalid"
            )
        evidence_hash = _sha256(
            self.prerequisite_stage_evidence_sha256,
            "semantic prerequisite_stage_evidence_sha256",
        )
        attempt_id = _safe_id(self.attempt_id, "semantic attempt_id")
        candidate_hash = _sha256(self.candidate_sha256, "semantic candidate_sha256")
        candidate_design_hash = _sha256(
            self.candidate_design_sha256,
            "semantic candidate_design_sha256",
        )
        entry_hash = _sha256(
            self.registry_entry_sha256, "semantic registry_entry_sha256"
        )
        request_hash = _sha256(self.request_sha256, "semantic request_sha256")
        if self.requested_stage not in {"intermediate", "final"}:
            raise SecFilingGemmaRevealStoreError(
                "Semantic requested stage is invalid"
            )
        access_hash = _sha256(
            self.stage_access_manifest_sha256,
            "semantic stage_access_manifest_sha256",
        )
        registry_hash = _sha256(
            self.registry_sha256, "semantic registry_sha256"
        )
        registry_tip = _sha256(
            self.registry_tip_sha256, "semantic registry_tip_sha256"
        )
        semantic_checks = _semantic_checks(
            self.semantic_checks, "Semantic checks"
        )
        receipt = _json_value_copy(
            dict(_expect_mapping(self.semantic_receipt, "semantic receipt")),
            "semantic receipt",
        )
        if not receipt:
            raise SecFilingGemmaRevealStoreError("Semantic receipt cannot be empty")
        body = {
            "schema_version": SEMANTIC_PREREQUISITE_SCHEMA_VERSION,
            "validation_kind": "independent_semantic_prerequisite_replay",
            "validator_id": validator_id,
            "validator_source_sha256": source_hash,
            "prerequisite_stage": prerequisite_stage,
            "prerequisite_stage_evidence_sha256": evidence_hash,
            "attempt_id": attempt_id,
            "candidate_sha256": candidate_hash,
            "candidate_design_sha256": candidate_design_hash,
            "registry_entry_sha256": entry_hash,
            "request_sha256": request_hash,
            "requested_stage": self.requested_stage,
            "stage_access_manifest_sha256": access_hash,
            "registry_sha256": registry_hash,
            "registry_tip_sha256": registry_tip,
            "semantic_checks": list(semantic_checks),
            "semantic_receipt": receipt,
            "semantic_receipt_sha256": canonical_sha256(receipt),
            "semantic_validation_completed": True,
            "authorizes_outcome_access": False,
        }
        return {**body, "result_sha256": canonical_sha256(body)}


def _validate_semantic_result(
    result: Any,
    *,
    expected_context: Mapping[str, Any],
    expected_validator_id: str,
    expected_validator_source_sha256: str,
) -> dict[str, Any]:
    if type(result) is not SemanticPrerequisiteValidation:
        raise SecFilingGemmaRevealStoreError(
            "Prerequisite verifier must return SemanticPrerequisiteValidation, "
            "not an arbitrary truthy result"
        )
    value = result.to_dict()
    if (
        value["validator_id"] != _safe_id(
            expected_validator_id, "expected semantic validator_id"
        )
        or not _same_digest(
            value["validator_source_sha256"],
            _sha256(
                expected_validator_source_sha256,
                "expected semantic validator_source_sha256",
            ),
        )
    ):
        raise SecFilingGemmaRevealStoreError(
            "Prerequisite result is not from the independently pinned validator"
        )
    for key in (
        "prerequisite_stage",
        "prerequisite_stage_evidence_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "request_sha256",
        "stage_access_manifest_sha256",
        "registry_sha256",
        "registry_tip_sha256",
    ):
        if value[key] != expected_context[key]:
            raise SecFilingGemmaRevealStoreError(
                f"Semantic prerequisite result is not bound to request field {key}"
            )
    semantic_receipt = _expect_mapping(
        value["semantic_receipt"],
        "semantic prerequisite audit receipt",
    )
    for receipt_key, context_key in (
        (
            "trusted_stage_content_pin_sha256",
            "trusted_stage_content_pin_sha256",
        ),
        (
            "trusted_stage_content_authentication_receipt_sha256",
            "trusted_stage_content_authentication_receipt_sha256",
        ),
        (
            "parent_consumption_binding_sha256",
            "parent_consumption_binding_sha256",
        ),
        (
            "authenticated_store_context_sha256",
            "authenticated_store_context_sha256",
        ),
    ):
        if semantic_receipt.get(receipt_key) != expected_context.get(context_key):
            raise SecFilingGemmaRevealStoreError(
                f"Semantic prerequisite audit lost store binding {receipt_key}"
            )
    if value["requested_stage"] != expected_context["stage"]:
        raise SecFilingGemmaRevealStoreError(
            "Semantic prerequisite result is not bound to the requested stage"
        )
    _semantic_checks(value["semantic_checks"], "Semantic prerequisite checks")
    body = {key: value[key] for key in value if key != "result_sha256"}
    if canonical_sha256(body) != value["result_sha256"]:
        raise SecFilingGemmaRevealStoreError(
            "Semantic prerequisite result hash is inconsistent"
        )
    return value


def _expected_context_for_consumed_request(
    request: Mapping[str, Any],
    *,
    trusted_stage_content_pin_sha256: str,
    trusted_stage_content_authentication_receipt_sha256: str,
    parent_consumption_binding_sha256: str | None,
    authenticated_store_context_sha256: str,
) -> dict[str, Any]:
    return {
        "prerequisite_stage": request["prerequisite_stage"],
        "prerequisite_stage_evidence_sha256": request[
            "prerequisite_stage_evidence_sha256"
        ],
        "attempt_id": request["attempt_id"],
        "candidate_sha256": request["candidate_sha256"],
        "candidate_design_sha256": request["candidate_design_sha256"],
        "registry_entry_sha256": request["registry_entry_sha256"],
        "request_sha256": request["request_sha256"],
        "stage": request["stage"],
        "stage_access_manifest_sha256": request[
            "stage_access_manifest_sha256"
        ],
        "registry_sha256": request["registry_sha256"],
        "registry_tip_sha256": request["registry_tip_sha256"],
        "trusted_stage_content_pin_sha256": trusted_stage_content_pin_sha256,
        "trusted_stage_content_authentication_receipt_sha256": (
            trusted_stage_content_authentication_receipt_sha256
        ),
        "parent_consumption_binding_sha256": parent_consumption_binding_sha256,
        "authenticated_store_context_sha256": authenticated_store_context_sha256,
    }


def _locate_parent_intermediate_predecessor(
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    child_request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state = _expect_mapping(
        authenticated_store_snapshot,
        "parent predecessor store snapshot",
    )
    current_tip = _expect_mapping(
        independent_current_tip_anchor,
        "parent predecessor current tip",
    )
    if child_request["stage"] != "final" or child_request[
        "prerequisite_stage"
    ] != "intermediate":
        raise SecFilingGemmaRevealStoreError(
            "Only a final request may require an intermediate predecessor"
        )
    matching = [
        entry
        for entry in state["consumption_ledger"]["entries"]
        if entry["stage"] == "intermediate"
        and entry["attempt_id"] == child_request["attempt_id"]
        and entry["candidate_sha256"] == child_request["candidate_sha256"]
        and entry["registry_entry_sha256"]
        == child_request["registry_entry_sha256"]
        and entry["request"]["candidate_design_sha256"]
        == child_request["candidate_design_sha256"]
        and entry["request"]["registry_sha256"]
        == child_request["registry_sha256"]
        and entry["request"]["registry_tip_sha256"]
        == child_request["registry_tip_sha256"]
    ]
    if len(matching) != 1:
        raise SecFilingGemmaRevealStoreError(
            "Final request requires exactly one intermediate predecessor"
        )
    entry = matching[0]
    ledger = state["consumption_ledger"]
    if (
        entry != ledger["entries"][-1]
        or entry["entry_sha256"] != ledger["chain"]["tip_sha256"]
    ):
        raise SecFilingGemmaRevealStoreError(
            "Final request predecessor is not the exact consumption-ledger tip"
        )
    bundle = current_tip["authorization_bundles"].get(entry["request_sha256"])
    if bundle is None:
        raise SecFilingGemmaRevealStoreError(
            "Final request predecessor lacks its persisted authorization bundle"
        )
    output_receipt = current_tip["consumed_stage_output_receipts"].get(
        entry["request_sha256"]
    )
    if output_receipt is None:
        raise SecFilingGemmaRevealStoreError(
            "Final request predecessor lacks its persisted first-output receipt"
        )
    return entry, bundle, output_receipt


def _build_parent_consumption_binding(
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    child_request: Mapping[str, Any],
    parent_entry: Mapping[str, Any],
    parent_bundle: Mapping[str, Any],
    parent_output_receipt: Mapping[str, Any],
    prerequisite_stage_evidence: Mapping[str, Any],
    parent_output_binding: Mapping[str, Any],
) -> dict[str, Any]:
    state = _expect_mapping(
        authenticated_store_snapshot,
        "parent binding store snapshot",
    )
    current_tip = _expect_mapping(
        independent_current_tip_anchor,
        "parent binding current tip",
    )
    parent = _expect_mapping(parent_entry, "parent consumption entry")
    bundle = _expect_mapping(parent_bundle, "parent authorization bundle")
    output_receipt = _expect_mapping(
        parent_output_receipt,
        "parent consumed-stage output receipt",
    )
    parent_request = _expect_mapping(parent["request"], "parent reveal request")
    persisted_bundle = current_tip["authorization_bundles"].get(
        parent_request["request_sha256"]
    )
    if persisted_bundle != bundle:
        raise SecFilingGemmaRevealStoreError(
            "Parent authorization bundle is not the exact current-tip bundle"
        )
    if current_tip["consumed_stage_output_receipts"].get(
        parent_request["request_sha256"]
    ) != output_receipt:
        raise SecFilingGemmaRevealStoreError(
            "Parent output receipt is not the exact current-tip receipt"
        )
    parent_access = _expect_mapping(
        parent["stage_access_manifest"],
        "parent stage-access manifest",
    )
    parent_validation = _expect_mapping(
        parent["prerequisite_validation"],
        "parent prerequisite validation",
    )
    parent_audit = _expect_mapping(
        parent_validation["semantic_receipt"],
        "parent stage audit receipt",
    )
    parent_pin = current_tip["trusted_stage_content_pins"].get(
        parent_request["request_sha256"]
    )
    if parent_pin is None:
        raise SecFilingGemmaRevealStoreError(
            "Final request predecessor lacks its persisted trusted content pin"
        )
    if (
        parent_audit.get("schema_version") != STAGE_AUDIT_RECEIPT_SCHEMA_VERSION
        or parent_audit.get("prerequisite_stage") != "development"
        or parent_audit.get("requested_stage") != "intermediate"
        or parent_audit.get("stage_evidence_sha256")
        != parent_request["prerequisite_stage_evidence_sha256"]
        or parent_audit.get("candidate_sha256")
        != parent_request["candidate_sha256"]
        or parent_audit.get("trusted_stage_content_pin_sha256")
        != parent_pin.get("pin_sha256")
        or parent_audit.get("parent_consumption_binding_sha256") is not None
        or parent_audit.get("authorizes_outcome_access") is not False
    ):
        raise SecFilingGemmaRevealStoreError(
            "Parent audit receipt lost its exact stage, evidence, candidate, or pin binding"
        )
    audit_hash = _sha256(
        parent_audit.get("audit_receipt_sha256"),
        "parent audit receipt hash",
    )
    audit_body = {
        key: parent_audit[key]
        for key in parent_audit
        if key != "audit_receipt_sha256"
    }
    if not _same_digest(audit_hash, canonical_sha256(audit_body)):
        raise SecFilingGemmaRevealStoreError(
            "Parent audit receipt self-hash is inconsistent"
        )
    parent_authentication = validate_trusted_stage_content_authentication_receipt(
        parent_audit.get("trusted_stage_content_authentication")
    )
    if (
        parent_audit.get(
            "trusted_stage_content_authentication_receipt_sha256"
        )
        != parent_authentication["authentication_receipt_sha256"]
        or parent_validation["semantic_receipt_sha256"]
        != canonical_sha256(parent_audit)
    ):
        raise SecFilingGemmaRevealStoreError(
            "Parent audit receipt lost its authentication or semantic-receipt binding"
        )
    parent_store_context = _authenticated_store_verifier_context(
        trusted_stage_content_pin=parent_pin,
        trusted_stage_content_authentication=parent_authentication,
        parent_consumption_binding=None,
    )
    if (
        parent_store_context["authenticated_store_context_sha256"]
        != parent_audit.get("authenticated_store_context_sha256")
    ):
        raise SecFilingGemmaRevealStoreError(
            "Parent audit receipt does not bind its reconstructed store context"
        )
    parent_expected_context = _expected_context_for_consumed_request(
        parent_request,
        trusted_stage_content_pin_sha256=parent_pin["pin_sha256"],
        trusted_stage_content_authentication_receipt_sha256=parent_authentication[
            "authentication_receipt_sha256"
        ],
        parent_consumption_binding_sha256=None,
        authenticated_store_context_sha256=parent_store_context[
            "authenticated_store_context_sha256"
        ],
    )
    try:
        validate_consumed_stage_authorization_grant(
            bundle["authorization_grant"],
            authenticated_store_snapshot=state,
            external_store_state_pin=bundle["store_state_pin"],
            independent_current_tip_anchor=current_tip,
            expected_consumption_entry_sha256=parent["entry_sha256"],
            expected_request_sha256=parent_request["request_sha256"],
            expected_candidate_sha256=parent_request["candidate_sha256"],
            expected_stage="intermediate",
            expected_prerequisite_stage_evidence_sha256=parent_request[
                "prerequisite_stage_evidence_sha256"
            ],
            expected_stage_access_manifest_sha256=parent_request[
                "stage_access_manifest_sha256"
            ],
            expected_output_namespace=parent_access["output"]["namespace"],
        )
        output_binding = _expect_mapping(
            parent_output_binding,
            "durable parent stage-evidence output binding",
        )
        if output_binding["output_stage_evidence_sha256"] != child_request[
            "prerequisite_stage_evidence_sha256"
        ]:
            raise SecFilingGemmaStageAuthorizationError(
                "Parent output receipt differs from the child prerequisite evidence"
            )
        validate_consumed_stage_output_receipt(
            output_receipt,
            authenticated_store_snapshot=state,
            independent_current_tip_anchor=current_tip,
            authorization_bundle=bundle,
            **output_binding,
        )
    except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
        raise SecFilingGemmaRevealStoreError(
            "Parent authorization bundle is not valid at the current tip"
        ) from exc
    for request_field in (
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "registry_sha256",
        "registry_tip_sha256",
    ):
        if parent_request[request_field] != child_request[request_field]:
            raise SecFilingGemmaRevealStoreError(
                f"Parent predecessor crossed child identity {request_field}"
            )
    child_section = {
        field: child_request[field]
        for field in (
            "request_sha256",
            "stage",
            "prerequisite_stage",
            "prerequisite_stage_evidence_sha256",
            "stage_access_manifest_sha256",
            "attempt_id",
            "candidate_sha256",
            "candidate_design_sha256",
            "registry_entry_sha256",
            "registry_sha256",
            "registry_tip_sha256",
        )
    }
    parent_section = {
        "entry_sha256": parent["entry_sha256"],
        "sequence": parent["sequence"],
        "request_sha256": parent_request["request_sha256"],
        "stage": parent_request["stage"],
        "prerequisite_stage": parent_request["prerequisite_stage"],
        "prerequisite_stage_evidence_sha256": parent_request[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": parent_request[
            "stage_access_manifest_sha256"
        ],
        "expected_context_sha256": canonical_sha256(parent_expected_context),
        "attempt_id": parent_request["attempt_id"],
        "candidate_sha256": parent_request["candidate_sha256"],
        "candidate_design_sha256": parent_request["candidate_design_sha256"],
        "registry_entry_sha256": parent_request["registry_entry_sha256"],
        "registry_sha256": parent_request["registry_sha256"],
        "registry_tip_sha256": parent_request["registry_tip_sha256"],
        "prerequisite_validation_result_sha256": parent_validation[
            "result_sha256"
        ],
        "semantic_receipt_sha256": parent_validation[
            "semantic_receipt_sha256"
        ],
        "audit_receipt": parent_audit,
        "audit_receipt_sha256": audit_hash,
        "trusted_stage_content_pin": parent_pin,
        "trusted_stage_content_pin_sha256": parent_pin["pin_sha256"],
        "authorization_bundle_sha256": bundle["bundle_sha256"],
        "authorization_grant_sha256": bundle["authorization_grant"][
            "authorization_grant_sha256"
        ],
        "store_pin_sha256": bundle["store_state_pin"]["store_pin_sha256"],
        "consumed_stage_output_receipt": output_receipt,
        "consumed_stage_output_receipt_sha256": output_receipt[
            "output_receipt_sha256"
        ],
    }
    authenticated_tip = {
        "store_state_sha256": state["state_sha256"],
        "state_snapshot_bytes_sha256": current_tip[
            "state_snapshot_bytes_sha256"
        ],
        "state_snapshot_byte_count": current_tip["state_snapshot_byte_count"],
        "consumption_ledger_sha256": state["consumption_ledger"][
            "ledger_sha256"
        ],
        "consumption_ledger_tip_sha256": state["consumption_ledger"]["chain"][
            "tip_sha256"
        ],
        "consumed_request_count": state["consumption_ledger"]["chain"][
            "consumed_request_count"
        ],
        "current_tip_anchor_sha256": current_tip["tip_anchor_sha256"],
        "current_tip_revision": current_tip["revision"],
        "consumed_stage_output_receipts": current_tip[
            "consumed_stage_output_receipts"
        ],
        "consumed_stage_output_receipts_sha256": canonical_sha256(
            current_tip["consumed_stage_output_receipts"]
        ),
        "stage_sec_execution_claims": current_tip[
            "stage_sec_execution_claims"
        ],
        "stage_sec_execution_claims_sha256": canonical_sha256(
            current_tip["stage_sec_execution_claims"]
        ),
        "stage_sec_reader_receipts": current_tip[
            "stage_sec_reader_receipts"
        ],
        "stage_sec_reader_receipts_sha256": canonical_sha256(
            current_tip["stage_sec_reader_receipts"]
        ),
    }
    body = {
        "schema_version": PARENT_CONSUMPTION_BINDING_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "binding_kind": "exact_prior_intermediate_consumption_and_grant",
        "child_request": child_section,
        "parent_consumption": parent_section,
        "authenticated_preconsumption_tip": authenticated_tip,
    }
    return {
        **body,
        "parent_consumption_binding_sha256": canonical_sha256(body),
    }


def _stage_evidence_sha256(evidence: Mapping[str, Any]) -> str:
    value = _json_value_copy(
        dict(_expect_mapping(evidence, "prerequisite stage evidence")),
        "prerequisite stage evidence",
    )
    if "stage_evidence_sha256" not in value:
        return canonical_sha256(value)
    observed = _sha256(
        value["stage_evidence_sha256"], "stage evidence self-hash"
    )
    body = {key: value[key] for key in value if key != "stage_evidence_sha256"}
    expected = canonical_sha256(body)
    if not _same_digest(observed, expected):
        raise SecFilingGemmaRevealStoreError(
            "Prerequisite stage evidence self-hash is inconsistent"
        )
    return expected


def _stage_evidence_output_binding(
    evidence: Mapping[str, Any],
    *,
    authorization_grant: Mapping[str, Any],
) -> dict[str, Any]:
    value = _json_value_copy(
        dict(_expect_mapping(evidence, "consumed-stage output evidence")),
        "consumed-stage output evidence",
    )
    grant = _expect_mapping(
        authorization_grant,
        "consumed-stage output authorization grant",
    )
    schema_version = _safe_id(
        value.get("schema_version"),
        "consumed-stage output evidence schema version",
    )
    prerequisite_stage = value.get("prerequisite_stage", value.get("stage"))
    prerequisite_stage = _safe_id(
        prerequisite_stage,
        "consumed-stage output evidence prerequisite stage",
    )
    candidate_manifest = value.get("candidate_manifest")
    candidate_sha256 = (
        candidate_manifest.get("candidate_sha256")
        if type(candidate_manifest) is dict
        else value.get("candidate_sha256")
    )
    candidate_hash = _sha256(
        candidate_sha256,
        "consumed-stage output evidence candidate hash",
    )
    parent_hash = value.get("parent_stage_evidence_sha256")
    if parent_hash is None:
        # Compact store tests and pre-run diagnostics may use a reduced
        # envelope.  The receipt still binds the only parent evidence hash
        # authorized by the exact grant; the production v3 evidence schema
        # carries this field explicitly and the authoritative verifier checks it.
        parent_hash = grant.get("prerequisite_stage_evidence_sha256")
    parent_hash = _sha256(
        parent_hash,
        "consumed-stage output evidence parent hash",
    )
    evidence_hash = _stage_evidence_sha256(value)
    try:
        canonical_bytes = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - detached above
        raise SecFilingGemmaRevealStoreError(
            "Consumed-stage output evidence is not canonical finite JSON"
        ) from exc
    if (
        prerequisite_stage != grant.get("stage")
        or parent_hash != grant.get("prerequisite_stage_evidence_sha256")
        or candidate_hash != grant.get("candidate_sha256")
    ):
        raise SecFilingGemmaRevealStoreError(
            "Consumed-stage output evidence crossed its authorization grant"
        )
    return {
        "output_stage_evidence_schema_version": schema_version,
        "output_stage_evidence_sha256": evidence_hash,
        "output_stage_evidence_document_sha256": hashlib.sha256(
            canonical_bytes
        ).hexdigest(),
        "output_stage_evidence_canonical_byte_count": len(canonical_bytes),
        "output_stage_evidence_prerequisite_stage": prerequisite_stage,
        "output_parent_stage_evidence_sha256": parent_hash,
        "output_candidate_sha256": candidate_hash,
    }


def _stage_access_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    value = _json_value_copy(
        dict(_expect_mapping(manifest, "stage access manifest")),
        "stage access manifest",
    )
    if "stage_access_manifest_sha256" not in value:
        raise SecFilingGemmaRevealStoreError(
            "Stage access manifest must carry its canonical self-hash"
        )
    observed = _sha256(
        value["stage_access_manifest_sha256"],
        "stage access manifest self-hash",
    )
    body = {
        key: value[key]
        for key in value
        if key != "stage_access_manifest_sha256"
    }
    expected = canonical_sha256(body)
    if not _same_digest(observed, expected):
        raise SecFilingGemmaRevealStoreError(
            "Stage access manifest self-hash is inconsistent"
        )
    return expected


def _validate_consumption_ledger(
    ledger: Mapping[str, Any], *, anchor: Mapping[str, Any]
) -> dict[str, Any]:
    value = _expect_mapping(ledger, "consumption ledger")
    _expect_keys(
        value,
        {"schema_version", "entries", "chain", "ledger_sha256"},
        "consumption ledger",
    )
    if value["schema_version"] != CONSUMPTION_LEDGER_SCHEMA_VERSION:
        raise SecFilingGemmaRevealStoreError("Unknown consumption-ledger schema")
    if not isinstance(value["entries"], list):
        raise SecFilingGemmaRevealStoreError("Consumption entries must be a list")

    entries: list[dict[str, Any]] = []
    seen_requests: set[str] = set()
    seen_stages: set[tuple[str, str]] = set()
    current_tip = _consumption_genesis(anchor)
    final_count = 0
    intermediate_candidates: set[tuple[str, str, str]] = set()
    entry_keys = {
        "schema_version",
        "sequence",
        "request_sha256",
        "request",
        "stage_access_manifest",
        "stage",
        "attempt_id",
        "candidate_sha256",
        "registry_entry_sha256",
        "prerequisite_validation",
        "prior_tip_sha256",
        "final_touch_delta",
        "cumulative_actual_final_touch_count",
        "entry_sha256",
    }
    validation_keys = {
        "schema_version",
        "validation_kind",
        "validator_id",
        "validator_source_sha256",
        "prerequisite_stage",
        "prerequisite_stage_evidence_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "request_sha256",
        "requested_stage",
        "stage_access_manifest_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "semantic_checks",
        "semantic_receipt",
        "semantic_receipt_sha256",
        "semantic_validation_completed",
        "authorizes_outcome_access",
        "result_sha256",
    }

    for sequence, raw_entry in enumerate(value["entries"], start=1):
        entry = _expect_mapping(raw_entry, f"consumption entry {sequence}")
        _expect_keys(entry, entry_keys, f"consumption entry {sequence}")
        if (
            entry["schema_version"] != CONSUMPTION_ENTRY_SCHEMA_VERSION
            or _strict_int(entry["sequence"], "consumption sequence", minimum=1)
            != sequence
        ):
            raise SecFilingGemmaRevealStoreError(
                "Consumption entry sequence or schema is invalid"
            )
        request = _expect_mapping(entry["request"], "consumed reveal request")
        request_hash = _request_body_hash(request)
        if entry["request_sha256"] != request_hash:
            raise SecFilingGemmaRevealStoreError(
                "Consumption entry does not bind its exact request"
            )
        access_manifest = _expect_mapping(
            entry["stage_access_manifest"], "consumed stage access manifest"
        )
        if (
            _stage_access_manifest_sha256(access_manifest)
            != request["stage_access_manifest_sha256"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Consumption entry does not bind its exact stage access manifest"
            )
        stage = entry["stage"]
        attempt_id = entry["attempt_id"]
        candidate_hash = entry["candidate_sha256"]
        registry_entry_hash = entry["registry_entry_sha256"]
        if (
            stage != request["stage"]
            or attempt_id != request["attempt_id"]
            or candidate_hash != request["candidate_sha256"]
            or registry_entry_hash != request["registry_entry_sha256"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Consumption identity is not bound to the stored request"
            )
        _safe_id(attempt_id, "consumption attempt_id")
        _sha256(candidate_hash, "consumption candidate_sha256")
        _sha256(registry_entry_hash, "consumption registry_entry_sha256")
        if request_hash in seen_requests or (attempt_id, stage) in seen_stages:
            raise SecFilingGemmaRevealStoreError(
                "A reveal request or candidate stage was consumed more than once"
            )

        validation = _expect_mapping(
            entry["prerequisite_validation"], "stored prerequisite validation"
        )
        _expect_keys(validation, validation_keys, "stored prerequisite validation")
        validation_body = {
            key: validation[key] for key in validation if key != "result_sha256"
        }
        semantic_receipt = _expect_mapping(
            validation["semantic_receipt"], "stored semantic receipt"
        )
        if (
            validation["schema_version"] != SEMANTIC_PREREQUISITE_SCHEMA_VERSION
            or validation["validation_kind"]
            != "independent_semantic_prerequisite_replay"
            or validation["semantic_validation_completed"] is not True
            or validation["authorizes_outcome_access"] is not False
            or canonical_sha256(validation_body) != validation["result_sha256"]
            or not semantic_receipt
            or canonical_sha256(semantic_receipt)
            != validation["semantic_receipt_sha256"]
            or _semantic_checks(
                validation["semantic_checks"], "Stored semantic checks"
            )
            != REQUIRED_SEMANTIC_CHECKS
        ):
            raise SecFilingGemmaRevealStoreError(
                "Stored prerequisite validation is not canonical semantic evidence"
            )
        _safe_id(validation["validator_id"], "stored semantic validator_id")
        _sha256(
            validation["validator_source_sha256"],
            "stored semantic validator_source_sha256",
        )
        _sha256(
            validation["candidate_design_sha256"],
            "stored semantic candidate_design_sha256",
        )
        _sha256(validation["request_sha256"], "stored semantic request_sha256")
        _sha256(
            validation["stage_access_manifest_sha256"],
            "stored semantic stage_access_manifest_sha256",
        )
        _sha256(
            validation["registry_sha256"], "stored semantic registry_sha256"
        )
        _sha256(
            validation["registry_tip_sha256"],
            "stored semantic registry_tip_sha256",
        )
        for request_key, validation_key in (
            ("prerequisite_stage", "prerequisite_stage"),
            (
                "prerequisite_stage_evidence_sha256",
                "prerequisite_stage_evidence_sha256",
            ),
            ("attempt_id", "attempt_id"),
            ("candidate_sha256", "candidate_sha256"),
            ("candidate_design_sha256", "candidate_design_sha256"),
            ("registry_entry_sha256", "registry_entry_sha256"),
            ("request_sha256", "request_sha256"),
            ("stage_access_manifest_sha256", "stage_access_manifest_sha256"),
            ("registry_sha256", "registry_sha256"),
            ("registry_tip_sha256", "registry_tip_sha256"),
        ):
            if request[request_key] != validation[validation_key]:
                raise SecFilingGemmaRevealStoreError(
                    "Stored semantic validation lost its request binding"
                )
        if validation["requested_stage"] != request["stage"]:
            raise SecFilingGemmaRevealStoreError(
                "Stored semantic validation lost its requested-stage binding"
            )

        prior_tip = _sha256(entry["prior_tip_sha256"], "consumption prior tip")
        if not _same_digest(prior_tip, current_tip):
            raise SecFilingGemmaRevealStoreError(
                "Consumed-request hash chain is broken or rolled back"
            )
        delta = _strict_int(entry["final_touch_delta"], "final touch delta")
        if delta != (1 if stage == "final" else 0):
            raise SecFilingGemmaRevealStoreError(
                "Only actual final-request consumption may increment final touches"
            )
        candidate_key = (attempt_id, candidate_hash, registry_entry_hash)
        if stage == "final" and candidate_key not in intermediate_candidates:
            raise SecFilingGemmaRevealStoreError(
                "Final request cannot precede intermediate request consumption"
            )
        final_count += delta
        if entry["cumulative_actual_final_touch_count"] != final_count:
            raise SecFilingGemmaRevealStoreError(
                "Cumulative actual final-touch count is inconsistent"
            )
        body = {key: entry[key] for key in entry if key != "entry_sha256"}
        expected_entry_hash = canonical_sha256(body)
        if entry["entry_sha256"] != expected_entry_hash:
            raise SecFilingGemmaRevealStoreError(
                "Consumed-request entry hash is inconsistent"
            )

        normalized = _json_value_copy(dict(entry), "consumption entry")
        entries.append(normalized)
        seen_requests.add(request_hash)
        seen_stages.add((attempt_id, stage))
        if stage == "intermediate":
            intermediate_candidates.add(candidate_key)
        current_tip = expected_entry_hash

    expected = _consumption_ledger(entries, tip_sha256=current_tip, anchor=anchor)
    if value != expected:
        raise SecFilingGemmaRevealStoreError(
            "Consumption ledger count, tip, or final-touch total is inconsistent"
        )
    return expected


def _validate_state(
    state: Mapping[str, Any], *, expected_anchor: Mapping[str, Any]
) -> dict[str, Any]:
    value = _expect_mapping(state, "reveal-store state")
    _expect_keys(
        value,
        {
            "schema_version",
            "contract_version",
            "anchor",
            "latest_registry",
            "latest_registry_pin",
            "consumption_ledger",
            "state_sha256",
        },
        "reveal-store state",
    )
    if (
        value["schema_version"] != STORE_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
    ):
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store state belongs to another schema or contract"
        )
    anchor = _expect_mapping(value["anchor"], "stored anchor")
    if anchor != expected_anchor:
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store state does not bind the tracked genesis anchors"
        )
    registry = _expect_mapping(value["latest_registry"], "latest registry")
    pin = _expect_mapping(value["latest_registry_pin"], "latest registry pin")
    try:
        validate_reveal_registry(registry, external_pin=pin)
    except SecFilingGemmaRevealRegistryError as exc:
        raise SecFilingGemmaRevealStoreError(
            "Authoritative registry and pin are stale, forked, rolled back, or invalid"
        ) from exc
    ledger = _validate_consumption_ledger(
        _expect_mapping(value["consumption_ledger"], "consumption ledger"),
        anchor=anchor,
    )
    latest_entries = registry["entries"]
    for consumed in ledger["entries"]:
        request = consumed["request"]
        registered_count = _strict_int(
            request["registered_entry_count"],
            "consumed request registered_entry_count",
            minimum=1,
        )
        if registered_count > len(latest_entries):
            raise SecFilingGemmaRevealStoreError(
                "Latest registry was rolled back behind a consumed request"
            )
        registered_entry = latest_entries[registered_count - 1]
        if (
            registered_entry["entry_sha256"]
            != request["registry_entry_sha256"]
            or registered_entry["candidate_sha256"]
            != request["candidate_sha256"]
            or registered_entry["attempt_id"] != request["attempt_id"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Consumed request is not an ancestor of the latest registry"
            )
        prefix = copy.deepcopy(dict(registry))
        prefix["entries"] = copy.deepcopy(latest_entries[:registered_count])
        prefix["chain"]["tip_sha256"] = registered_entry["entry_sha256"]
        prefix["chain"]["registered_entry_count"] = registered_count
        prefix_body = {
            key: prefix[key] for key in prefix if key != "registry_sha256"
        }
        prefix["registry_sha256"] = canonical_sha256(prefix_body)
        try:
            prefix_pin = derive_registry_pin(prefix)
        except SecFilingGemmaRevealRegistryError as exc:
            raise SecFilingGemmaRevealStoreError(
                "Consumed request registry ancestry is not canonical"
            ) from exc
        if (
            request["registry_sha256"] != prefix["registry_sha256"]
            or request["registry_tip_sha256"] != prefix_pin["tip_sha256"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Consumed request does not bind its exact registry ancestor"
            )
    expected = _state_snapshot(
        anchor=anchor,
        latest_registry=registry,
        latest_pin=pin,
        consumption_ledger=ledger,
    )
    if value != expected:
        raise SecFilingGemmaRevealStoreError(
            "Authoritative reveal-store state hash is inconsistent"
        )
    return expected


class SecFilingGemmaRevealStore:
    """Atomic local store for registry appends and one-shot reveal requests."""

    def __init__(
        self,
        *,
        repository_root: Path,
        store_directory: Path,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        if isinstance(lock_timeout_seconds, bool) or not isinstance(
            lock_timeout_seconds, (int, float)
        ):
            raise TypeError("lock_timeout_seconds must be a positive number")
        if not (0.0 < float(lock_timeout_seconds) <= 60.0):
            raise SecFilingGemmaRevealStoreError(
                "lock_timeout_seconds must be in (0, 60]"
            )
        self._repository_root = _secure_directory(
            Path(repository_root), create=False, location="repository_root"
        )
        self._store_directory = Path(
            os.path.abspath(os.fspath(store_directory))
        )
        self._lock_timeout_seconds = float(lock_timeout_seconds)

    @property
    def repository_root(self) -> Path:
        return self._repository_root

    @property
    def store_directory(self) -> Path:
        return self._store_directory

    @property
    def lock_timeout_seconds(self) -> float:
        return self._lock_timeout_seconds

    @property
    def state_path(self) -> Path:
        return self.store_directory / STATE_FILENAME

    @property
    def current_tip_anchor_path(self) -> Path:
        return self.store_directory / CURRENT_TIP_ANCHOR_FILENAME

    @property
    def restore_pending_path(self) -> Path:
        return self.store_directory / RESTORE_PENDING_FILENAME

    @property
    def lock_path(self) -> Path:
        return self.store_directory / LOCK_FILENAME

    def _locked(self) -> _ExclusiveFileLock:
        secured = _secure_directory(
            self._store_directory,
            create=True,
            location="reveal-store directory",
        )
        if secured != self._store_directory:
            raise SecFilingGemmaRevealStoreError(
                "Reveal-store directory identity changed"
            )
        return _ExclusiveFileLock(
            self.lock_path, timeout_seconds=self._lock_timeout_seconds
        )

    def _owned_development_sec_root_execution_lock(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> _ExclusiveFileLock:
        """Serialize one root runner without holding the reveal-store CAS lock."""

        scope_hash = _sha256(
            development_root_scope_sha256,
            "development SEC execution-lock root scope hash",
        )
        store_directory = _secure_directory(
            self._store_directory,
            create=True,
            location="reveal-store directory",
        )
        locks_directory = _secure_directory(
            store_directory / DEVELOPMENT_SEC_EXECUTION_LOCKS_DIRECTORY_NAME,
            create=True,
            location="development SEC execution-lock directory",
        )
        return _ExclusiveFileLock(
            locks_directory / f"{scope_hash}.lock",
            timeout_seconds=self._lock_timeout_seconds,
        )

    def _owned_model_execution_lock(self) -> _ExclusiveFileLock:
        """Serialize the one globally active model effect across all lifecycles."""

        store_directory = _secure_directory(
            self._store_directory,
            create=True,
            location="reveal-store directory",
        )
        return _ExclusiveFileLock(
            store_directory / MODEL_EXECUTION_LOCK_FILENAME,
            timeout_seconds=self._lock_timeout_seconds,
        )

    def _owned_market_execution_lock(self) -> _ExclusiveFileLock:
        """Serialize the one development market effect outside the CAS lock."""

        store_directory = _secure_directory(
            self._store_directory,
            create=True,
            location="reveal-store directory",
        )
        return _ExclusiveFileLock(
            store_directory / MARKET_EXECUTION_LOCK_FILENAME,
            timeout_seconds=self._lock_timeout_seconds,
        )

    def _recover_pending_restore_locked(
        self, tracked_anchor: Mapping[str, Any]
    ) -> None:
        path = self.restore_pending_path
        if not path.exists() and not path.is_symlink():
            return
        payload = _read_regular_bytes(
            path,
            "pending reveal-store restore transaction",
            max_bytes=MAX_RESTORE_PENDING_FILE_BYTES,
        )
        parsed = _strict_json_bytes(
            payload, "pending reveal-store restore transaction"
        )
        recovery = _validated_restore_pending_document(
            parsed, tracked_anchor=tracked_anchor
        )
        # The recovery file remains authoritative until both target files are
        # exact. A stop after either replace simply replays this idempotently.
        _atomic_replace(self.state_path, recovery["state_bytes"])
        _atomic_replace(
            self.current_tip_anchor_path, recovery["tip_anchor_bytes"]
        )
        if (
            _read_regular_bytes(
                self.state_path,
                "recovered reveal-store state",
                max_bytes=MAX_STATE_FILE_BYTES,
            )
            != recovery["state_bytes"]
            or _read_regular_bytes(
                self.current_tip_anchor_path,
                "recovered reveal-store current-tip anchor",
                max_bytes=MAX_CURRENT_TIP_ANCHOR_FILE_BYTES,
            )
            != recovery["tip_anchor_bytes"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Recovered reveal-store files differ from the restore transaction"
            )
        try:
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(
                metadata.st_mode
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Restore transaction path changed file type"
                )
            path.unlink()
        except OSError as exc:
            raise SecFilingGemmaRevealStoreError(
                "Completed restore transaction could not be retired"
            ) from exc
        if path.exists() or path.is_symlink():
            raise SecFilingGemmaRevealStoreError(
                "Completed restore transaction remains present"
            )

    def _read_state_and_tip_locked(
        self,
        tracked_anchor: Mapping[str, Any],
        *,
        recover_pending_restore: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
        if recover_pending_restore:
            self._recover_pending_restore_locked(tracked_anchor)
        try:
            state_payload = _read_regular_bytes(
                self.state_path,
                "authoritative reveal-store state",
                max_bytes=MAX_STATE_FILE_BYTES,
            )
        except SecFilingGemmaRevealStoreError as exc:
            if not self.state_path.exists() and not self.state_path.is_symlink():
                raise SecFilingGemmaRevealStoreError(
                    "Reveal store is not initialized"
                ) from exc
            raise
        state_parsed = _strict_json_bytes(
            state_payload, "authoritative reveal-store state"
        )
        try:
            state_parsed = detach_untrusted_stage_json(
                state_parsed, "authoritative reveal-store state"
            )
        except Exception as exc:
            raise SecFilingGemmaRevealStoreError(
                "Authoritative reveal-store state exceeds fixed allocation bounds"
            ) from exc
        state = _validate_state(
            _expect_mapping(state_parsed, "authoritative reveal-store state"),
            expected_anchor=tracked_anchor,
        )
        try:
            tip_payload = _read_regular_bytes(
                self.current_tip_anchor_path,
                "independent reveal-store current-tip anchor",
                max_bytes=MAX_CURRENT_TIP_ANCHOR_FILE_BYTES,
            )
        except SecFilingGemmaRevealStoreError as exc:
            if (
                not self.current_tip_anchor_path.exists()
                and not self.current_tip_anchor_path.is_symlink()
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Independent reveal-store current-tip anchor is missing"
                ) from exc
            raise
        tip_parsed = _strict_json_bytes(
            tip_payload, "independent reveal-store current-tip anchor"
        )
        tip_mapping = _expect_mapping(
            tip_parsed, "independent reveal-store current-tip anchor"
        )
        if tip_mapping.get("schema_version") == CURRENT_TIP_PENDING_SCHEMA_VERSION:
            pending = _validated_pending_current_tip_document(
                tip_mapping,
                authenticated_store_snapshot=state,
            )
            prior = pending["prior_tip_anchor"]
            next_anchor = pending["next_tip_anchor"]
            if _state_bytes_match_tip_anchor(state_payload, state, next_anchor):
                resolved = next_anchor
            elif prior is not None and _state_bytes_match_tip_anchor(
                state_payload, state, prior
            ):
                resolved = prior
            else:
                raise SecFilingGemmaRevealStoreError(
                    "Interrupted current-tip transaction matches neither its prior nor next state"
                )
            try:
                resolved = validate_reveal_store_current_tip_anchor(
                    state, resolved
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Interrupted current-tip transaction cannot authenticate its state"
                ) from exc
            _atomic_replace(
                self.current_tip_anchor_path,
                _encoded_state(resolved),
            )
            tip_payload = _read_regular_bytes(
                self.current_tip_anchor_path,
                "resolved reveal-store current-tip anchor",
                max_bytes=MAX_CURRENT_TIP_ANCHOR_FILE_BYTES,
            )
            if tip_payload != _encoded_state(resolved):
                raise SecFilingGemmaRevealStoreError(
                    "Resolved current-tip anchor bytes are inconsistent"
                )
            tip = resolved
        else:
            try:
                tip = validate_reveal_store_current_tip_anchor(
                    state, tip_mapping
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Independent current-tip anchor does not authenticate the latest state"
                ) from exc
        return state, tip, state_payload, tip_payload

    def _read_state_locked(self, anchor: Mapping[str, Any]) -> dict[str, Any]:
        state, _tip, _state_bytes, _tip_bytes = self._read_state_and_tip_locked(
            anchor
        )
        return state

    def _commit_state_and_tip_locked(
        self,
        *,
        tracked_anchor: Mapping[str, Any],
        prior_tip_anchor: Mapping[str, Any],
        next_state: Mapping[str, Any],
        authorization_bundle: Mapping[str, Any] | None = None,
        trusted_stage_content_pin: Mapping[str, Any] | None = None,
        consumed_stage_output_receipt: Mapping[str, Any] | None = None,
        stage_sec_execution_claim: Mapping[str, Any] | None = None,
        stage_sec_reader_receipt: Mapping[str, Any] | None = None,
        stage_sec_execution_abort: Mapping[str, Any] | None = None,
        stage_carry_in_reader_receipt: Mapping[str, Any] | None = None,
        development_root_carry_in_reader_receipt: Mapping[str, Any] | None = None,
        development_sec_execution_claim: Mapping[str, Any] | None = None,
        development_sec_reader_receipt: Mapping[str, Any] | None = None,
        development_sec_execution_abort: Mapping[str, Any] | None = None,
        development_market_execution_claim: Mapping[str, Any] | None = None,
        development_market_reader_receipt: Mapping[str, Any] | None = None,
        development_market_execution_abort: Mapping[str, Any] | None = None,
        stage_model_execution_claim: Mapping[str, Any] | None = None,
        stage_model_reader_receipt: Mapping[str, Any] | None = None,
        stage_model_execution_abort: Mapping[str, Any] | None = None,
        development_model_execution_claim: Mapping[str, Any] | None = None,
        development_model_reader_receipt: Mapping[str, Any] | None = None,
        development_model_execution_abort: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        validated_state = _validate_state(
            _expect_mapping(next_state, "next reveal-store state"),
            expected_anchor=tracked_anchor,
        )
        bundles = copy.deepcopy(prior_tip_anchor["authorization_bundles"])
        pins = copy.deepcopy(prior_tip_anchor["trusted_stage_content_pins"])
        output_receipts = copy.deepcopy(
            prior_tip_anchor["consumed_stage_output_receipts"]
        )
        sec_claims = copy.deepcopy(prior_tip_anchor["stage_sec_execution_claims"])
        sec_reader_receipts = copy.deepcopy(
            prior_tip_anchor["stage_sec_reader_receipts"]
        )
        sec_aborts = copy.deepcopy(prior_tip_anchor["stage_sec_execution_aborts"])
        carry_in_receipts = copy.deepcopy(
            prior_tip_anchor["stage_carry_in_reader_receipts"]
        )
        development_root_carry_in_receipts = copy.deepcopy(
            prior_tip_anchor["development_root_carry_in_reader_receipts"]
        )
        development_sec_claims = copy.deepcopy(
            prior_tip_anchor["development_sec_execution_claims"]
        )
        development_sec_reader_receipts = copy.deepcopy(
            prior_tip_anchor["development_sec_reader_receipts"]
        )
        development_sec_aborts = copy.deepcopy(
            prior_tip_anchor["development_sec_execution_aborts"]
        )
        development_market_claims = copy.deepcopy(
            prior_tip_anchor["development_market_execution_claims"]
        )
        development_market_reader_receipts = copy.deepcopy(
            prior_tip_anchor["development_market_reader_receipts"]
        )
        development_market_aborts = copy.deepcopy(
            prior_tip_anchor["development_market_execution_aborts"]
        )
        model_claims = copy.deepcopy(
            prior_tip_anchor["stage_model_execution_claims"]
        )
        model_reader_receipts = copy.deepcopy(
            prior_tip_anchor["stage_model_reader_receipts"]
        )
        model_aborts = copy.deepcopy(
            prior_tip_anchor["stage_model_execution_aborts"]
        )
        development_model_claims = copy.deepcopy(
            prior_tip_anchor["development_model_execution_claims"]
        )
        development_model_reader_receipts = copy.deepcopy(
            prior_tip_anchor["development_model_reader_receipts"]
        )
        development_model_aborts = copy.deepcopy(
            prior_tip_anchor["development_model_execution_aborts"]
        )
        transition_payload_count = sum(
            value is not None
            for value in (
                authorization_bundle,
                trusted_stage_content_pin,
                consumed_stage_output_receipt,
                stage_sec_execution_claim,
                stage_sec_reader_receipt,
                stage_sec_execution_abort,
                stage_carry_in_reader_receipt,
                development_root_carry_in_reader_receipt,
                development_sec_execution_claim,
                development_sec_reader_receipt,
                development_sec_execution_abort,
                development_market_execution_claim,
                development_market_reader_receipt,
                development_market_execution_abort,
                stage_model_execution_claim,
                stage_model_reader_receipt,
                stage_model_execution_abort,
                development_model_execution_claim,
                development_model_reader_receipt,
                development_model_execution_abort,
            )
        )
        if transition_payload_count > 1:
            raise SecFilingGemmaRevealStoreError(
                "Trusted pin, authorization bundle, output receipt, and SEC execution artifacts require separate transitions"
            )
        if trusted_stage_content_pin is not None:
            pin = _exact_caller_dict(
                trusted_stage_content_pin,
                "trusted stage-content pin",
            )
            request_hash = _sha256(
                pin.get("request_sha256"),
                "trusted stage-content pin request hash",
            )
            if request_hash in pins and pins[request_hash] != pin:
                raise SecFilingGemmaRevealStoreError(
                    "A persisted trusted stage-content pin cannot be replaced"
                )
            pins[request_hash] = pin
        if authorization_bundle is not None:
            bundle = _exact_caller_dict(
                authorization_bundle, "authorization bundle"
            )
            request_hash = _sha256(
                bundle.get("authorization_grant", {}).get("request_sha256"),
                "authorization bundle request hash",
            )
            if request_hash in bundles and bundles[request_hash] != bundle:
                raise SecFilingGemmaRevealStoreError(
                    "A persisted authorization bundle cannot be replaced"
                )
            bundles[request_hash] = bundle
        if consumed_stage_output_receipt is not None:
            output_receipt = _exact_caller_dict(
                consumed_stage_output_receipt,
                "consumed-stage output receipt",
            )
            request_hash = _sha256(
                output_receipt.get("request_sha256"),
                "consumed-stage output receipt request hash",
            )
            if (
                request_hash in output_receipts
                and output_receipts[request_hash] != output_receipt
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted consumed-stage output receipt cannot be replaced"
                )
            output_receipts[request_hash] = output_receipt
        if stage_sec_execution_claim is not None:
            sec_claim = _exact_caller_dict(
                stage_sec_execution_claim,
                "stage SEC execution claim",
            )
            request_hash = _sha256(
                sec_claim.get("request_sha256"),
                "stage SEC execution claim request hash",
            )
            if request_hash in sec_claims and sec_claims[request_hash] != sec_claim:
                raise SecFilingGemmaRevealStoreError(
                    "A persisted stage SEC execution claim cannot be replaced"
                )
            sec_claims[request_hash] = sec_claim
        if stage_sec_reader_receipt is not None:
            sec_receipt = _exact_caller_dict(
                stage_sec_reader_receipt,
                "stage SEC reader receipt",
            )
            request_hash = _sha256(
                sec_receipt.get("request_sha256"),
                "stage SEC reader receipt request hash",
            )
            if (
                request_hash in sec_reader_receipts
                and sec_reader_receipts[request_hash] != sec_receipt
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted stage SEC reader receipt cannot be replaced"
                )
            sec_reader_receipts[request_hash] = sec_receipt
        if stage_sec_execution_abort is not None:
            sec_abort = _exact_caller_dict(
                stage_sec_execution_abort,
                "stage SEC execution abort",
            )
            request_hash = _sha256(
                sec_abort.get("request_sha256"),
                "stage SEC execution abort request hash",
            )
            if request_hash in sec_aborts and sec_aborts[request_hash] != sec_abort:
                raise SecFilingGemmaRevealStoreError(
                    "A persisted stage SEC execution abort cannot be replaced"
                )
            sec_aborts[request_hash] = sec_abort
        if stage_carry_in_reader_receipt is not None:
            carry_receipt = _exact_caller_dict(
                stage_carry_in_reader_receipt,
                "stage carry-in reader receipt",
            )
            request_hash = _sha256(
                carry_receipt.get("request_sha256"),
                "stage carry-in reader receipt request hash",
            )
            if (
                request_hash in carry_in_receipts
                and carry_in_receipts[request_hash] != carry_receipt
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted stage carry-in reader receipt cannot be replaced"
                )
            carry_in_receipts[request_hash] = carry_receipt
        if development_root_carry_in_reader_receipt is not None:
            development_carry_receipt = _exact_caller_dict(
                development_root_carry_in_reader_receipt,
                "development-root carry-in reader receipt",
            )
            request_hash = _sha256(
                development_carry_receipt.get("request_sha256"),
                "development-root carry-in reader receipt request hash",
            )
            if (
                request_hash in development_root_carry_in_receipts
                and development_root_carry_in_receipts[request_hash]
                != development_carry_receipt
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted development-root carry-in receipt cannot be replaced"
                )
            development_root_carry_in_receipts[request_hash] = (
                development_carry_receipt
            )
        if development_sec_execution_claim is not None:
            development_claim = _exact_caller_dict(
                development_sec_execution_claim,
                "development SEC execution claim",
            )
            root_scope_hash = _sha256(
                development_claim.get("development_root_scope_sha256"),
                "development SEC execution claim root scope hash",
            )
            if (
                root_scope_hash in development_sec_claims
                and development_sec_claims[root_scope_hash] != development_claim
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted development SEC execution claim cannot be replaced"
                )
            development_sec_claims[root_scope_hash] = development_claim
        if development_sec_reader_receipt is not None:
            development_receipt = _exact_caller_dict(
                development_sec_reader_receipt,
                "development SEC reader receipt",
            )
            root_scope_hash = _sha256(
                development_receipt.get("development_root_scope_sha256"),
                "development SEC reader receipt root scope hash",
            )
            if (
                root_scope_hash in development_sec_reader_receipts
                and development_sec_reader_receipts[root_scope_hash]
                != development_receipt
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted development SEC reader receipt cannot be replaced"
                )
            development_sec_reader_receipts[root_scope_hash] = development_receipt
        if development_sec_execution_abort is not None:
            development_abort = _exact_caller_dict(
                development_sec_execution_abort,
                "development SEC execution abort",
            )
            root_scope_hash = _sha256(
                development_abort.get("development_root_scope_sha256"),
                "development SEC execution abort root scope hash",
            )
            if (
                root_scope_hash in development_sec_aborts
                and development_sec_aborts[root_scope_hash] != development_abort
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted development SEC execution abort cannot be replaced"
                )
            development_sec_aborts[root_scope_hash] = development_abort
        if development_market_execution_claim is not None:
            market_claim = _exact_caller_dict(
                development_market_execution_claim,
                "development market execution claim",
            )
            root_scope_hash = _sha256(
                market_claim.get("development_root_scope_sha256"),
                "development market execution claim root scope hash",
            )
            if (
                root_scope_hash in development_market_claims
                and development_market_claims[root_scope_hash] != market_claim
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted development market execution claim cannot be replaced"
                )
            development_market_claims[root_scope_hash] = market_claim
        if development_market_reader_receipt is not None:
            market_receipt = _exact_caller_dict(
                development_market_reader_receipt,
                "development market reader receipt",
            )
            root_scope_hash = _sha256(
                market_receipt.get("development_root_scope_sha256"),
                "development market reader receipt root scope hash",
            )
            if (
                root_scope_hash in development_market_reader_receipts
                and development_market_reader_receipts[root_scope_hash]
                != market_receipt
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted development market reader receipt cannot be replaced"
                )
            development_market_reader_receipts[root_scope_hash] = market_receipt
        if development_market_execution_abort is not None:
            market_abort = _exact_caller_dict(
                development_market_execution_abort,
                "development market execution abort",
            )
            root_scope_hash = _sha256(
                market_abort.get("development_root_scope_sha256"),
                "development market execution abort root scope hash",
            )
            if (
                root_scope_hash in development_market_aborts
                and development_market_aborts[root_scope_hash] != market_abort
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted development market execution abort cannot be replaced"
                )
            development_market_aborts[root_scope_hash] = market_abort
        if stage_model_execution_claim is not None:
            model_claim = _exact_caller_dict(
                stage_model_execution_claim,
                "stage model execution claim",
            )
            request_hash = _sha256(
                model_claim.get("request_sha256"),
                "stage model execution claim request hash",
            )
            if request_hash in model_claims and model_claims[request_hash] != model_claim:
                raise SecFilingGemmaRevealStoreError(
                    "A persisted stage model execution claim cannot be replaced"
                )
            model_claims[request_hash] = model_claim
        if stage_model_reader_receipt is not None:
            model_receipt = _exact_caller_dict(
                stage_model_reader_receipt,
                "stage model reader receipt",
            )
            request_hash = _sha256(
                model_receipt.get("request_sha256"),
                "stage model reader receipt request hash",
            )
            if (
                request_hash in model_reader_receipts
                and model_reader_receipts[request_hash] != model_receipt
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted stage model reader receipt cannot be replaced"
                )
            model_reader_receipts[request_hash] = model_receipt
        if stage_model_execution_abort is not None:
            model_abort = _exact_caller_dict(
                stage_model_execution_abort,
                "stage model execution abort",
            )
            request_hash = _sha256(
                model_abort.get("request_sha256"),
                "stage model execution abort request hash",
            )
            if request_hash in model_aborts and model_aborts[request_hash] != model_abort:
                raise SecFilingGemmaRevealStoreError(
                    "A persisted stage model execution abort cannot be replaced"
                )
            model_aborts[request_hash] = model_abort
        if development_model_execution_claim is not None:
            development_model_claim = _exact_caller_dict(
                development_model_execution_claim,
                "development model execution claim",
            )
            root_scope_hash = _sha256(
                development_model_claim.get("development_root_scope_sha256"),
                "development model execution claim root scope hash",
            )
            if (
                root_scope_hash in development_model_claims
                and development_model_claims[root_scope_hash]
                != development_model_claim
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted development model execution claim cannot be replaced"
                )
            development_model_claims[root_scope_hash] = development_model_claim
        if development_model_reader_receipt is not None:
            development_model_receipt = _exact_caller_dict(
                development_model_reader_receipt,
                "development model reader receipt",
            )
            root_scope_hash = _sha256(
                development_model_receipt.get("development_root_scope_sha256"),
                "development model reader receipt root scope hash",
            )
            if (
                root_scope_hash in development_model_reader_receipts
                and development_model_reader_receipts[root_scope_hash]
                != development_model_receipt
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted development model reader receipt cannot be replaced"
                )
            development_model_reader_receipts[root_scope_hash] = (
                development_model_receipt
            )
        if development_model_execution_abort is not None:
            development_model_abort = _exact_caller_dict(
                development_model_execution_abort,
                "development model execution abort",
            )
            root_scope_hash = _sha256(
                development_model_abort.get("development_root_scope_sha256"),
                "development model execution abort root scope hash",
            )
            if (
                root_scope_hash in development_model_aborts
                and development_model_aborts[root_scope_hash]
                != development_model_abort
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted development model execution abort cannot be replaced"
                )
            development_model_aborts[root_scope_hash] = development_model_abort
        try:
            next_tip = build_reveal_store_current_tip_anchor(
                validated_state,
                revision=prior_tip_anchor["revision"] + 1,
                previous_tip_anchor_sha256=prior_tip_anchor["tip_anchor_sha256"],
                authorization_bundles=bundles,
                trusted_stage_content_pins=pins,
                consumed_stage_output_receipts=output_receipts,
                stage_sec_execution_claims=sec_claims,
                stage_sec_reader_receipts=sec_reader_receipts,
                stage_sec_execution_aborts=sec_aborts,
                stage_carry_in_reader_receipts=carry_in_receipts,
                development_root_carry_in_reader_receipts=(
                    development_root_carry_in_receipts
                ),
                development_sec_execution_claims=development_sec_claims,
                development_sec_reader_receipts=development_sec_reader_receipts,
                development_sec_execution_aborts=development_sec_aborts,
                development_market_execution_claims=development_market_claims,
                development_market_reader_receipts=(
                    development_market_reader_receipts
                ),
                development_market_execution_aborts=development_market_aborts,
                stage_model_execution_claims=model_claims,
                stage_model_reader_receipts=model_reader_receipts,
                stage_model_execution_aborts=model_aborts,
                development_model_execution_claims=development_model_claims,
                development_model_reader_receipts=(
                    development_model_reader_receipts
                ),
                development_model_execution_aborts=development_model_aborts,
            )
        except SecFilingGemmaStageAuthorizationError as exc:
            raise SecFilingGemmaRevealStoreError(
                "Could not build the next independent current-tip anchor"
            ) from exc
        pending = _pending_current_tip_document(
            prior_tip_anchor=prior_tip_anchor,
            next_tip_anchor=next_tip,
            authenticated_store_snapshot=validated_state,
        )
        state_bytes = _encoded_state(validated_state)
        pending_bytes = _encoded_state(pending)
        next_tip_bytes = _encoded_state(next_tip)
        if len(state_bytes) > MAX_STATE_FILE_BYTES:
            raise SecFilingGemmaRevealStoreError(
                "Next reveal-store state exceeds its safety limit"
            )
        if max(len(pending_bytes), len(next_tip_bytes)) > MAX_CURRENT_TIP_ANCHOR_FILE_BYTES:
            raise SecFilingGemmaRevealStoreError(
                "Next current-tip transaction exceeds its safety limit"
            )

        # The pending anchor is a write-ahead CAS record.  A crash before the
        # state replace resolves to ``prior``; a crash after it resolves to
        # ``next``.  Thus the consumed entry and exact persisted bundle become
        # visible as one logical transaction despite using two regular files.
        _atomic_replace(self.current_tip_anchor_path, pending_bytes)
        _atomic_replace(self.state_path, state_bytes)
        _atomic_replace(self.current_tip_anchor_path, next_tip_bytes)
        committed_state, committed_tip, _state_bytes, _tip_bytes = (
            self._read_state_and_tip_locked(tracked_anchor)
        )
        if committed_state != validated_state or committed_tip != next_tip:
            raise SecFilingGemmaRevealStoreError(
                "Committed state/current-tip transaction differs from its CAS target"
            )
        return committed_state, committed_tip

    def initialize(self) -> dict[str, Any]:
        """Create genesis state once, or validate and return existing state."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            anchor = _load_tracked_anchor(self.repository_root)
            self._recover_pending_restore_locked(anchor)
            state_exists = self.state_path.exists() or self.state_path.is_symlink()
            tip_exists = (
                self.current_tip_anchor_path.exists()
                or self.current_tip_anchor_path.is_symlink()
            )
            if state_exists and tip_exists:
                return self._read_state_locked(anchor)
            registry = build_initial_reveal_registry()
            pin = derive_registry_pin(registry)
            state = _state_snapshot(
                anchor=anchor,
                latest_registry=registry,
                latest_pin=pin,
                consumption_ledger=_initial_consumption_ledger(anchor),
            )
            try:
                tip = build_reveal_store_current_tip_anchor(
                    state,
                    revision=0,
                    previous_tip_anchor_sha256=None,
                    authorization_bundles={},
                    trusted_stage_content_pins={},
                    consumed_stage_output_receipts={},
                    stage_sec_execution_claims={},
                    stage_sec_reader_receipts={},
                    stage_sec_execution_aborts={},
                    development_root_carry_in_reader_receipts={},
                    development_sec_execution_claims={},
                    development_sec_reader_receipts={},
                    development_sec_execution_aborts={},
                    development_market_execution_claims={},
                    development_market_reader_receipts={},
                    development_market_execution_aborts={},
                    stage_model_execution_claims={},
                    stage_model_reader_receipts={},
                    stage_model_execution_aborts={},
                    development_model_execution_claims={},
                    development_model_reader_receipts={},
                    development_model_execution_aborts={},
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Could not build the genesis current-tip anchor"
                ) from exc
            pending = _pending_current_tip_document(
                prior_tip_anchor=None,
                next_tip_anchor=tip,
            )
            if state_exists and not tip_exists:
                raise SecFilingGemmaRevealStoreError(
                    "Reveal store has state without its genesis write-ahead anchor"
                )
            if tip_exists and not state_exists:
                pending_bytes = _read_regular_bytes(
                    self.current_tip_anchor_path,
                    "interrupted genesis current-tip transaction",
                    max_bytes=MAX_CURRENT_TIP_ANCHOR_FILE_BYTES,
                )
                parsed = _strict_json_bytes(
                    pending_bytes, "interrupted genesis current-tip transaction"
                )
                observed_pending = _validated_pending_current_tip_document(
                    parsed
                )
                if observed_pending != pending:
                    raise SecFilingGemmaRevealStoreError(
                        "Incomplete reveal-store pair is not the deterministic genesis transaction"
                    )
                _atomic_replace(self.state_path, _encoded_state(state))
                _atomic_replace(
                    self.current_tip_anchor_path, _encoded_state(tip)
                )
                return self._read_state_locked(anchor)
            _atomic_replace(self.current_tip_anchor_path, _encoded_state(pending))
            _atomic_replace(self.state_path, _encoded_state(state))
            _atomic_replace(self.current_tip_anchor_path, _encoded_state(tip))
            return self._read_state_locked(anchor)

    def load(self) -> dict[str, Any]:
        """Load and fully authenticate the latest state without mutating it."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            anchor = _load_tracked_anchor(self.repository_root)
            return self._read_state_locked(anchor)

    def load_current_tip_anchor(self) -> dict[str, Any]:
        """Load the independent latest-tip proof for downstream validation."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            _state, tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            return copy.deepcopy(tip)

    def claim_owned_development_sec_root_execution(
        self,
        *,
        development_content_root_plan: Mapping[str, Any],
        sec_user_agent_sha256: str,
    ) -> dict[str, Any]:
        """Claim the complete pre-reveal development corpus exactly once."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            caller_plan = _exact_caller_dict(
                development_content_root_plan,
                "development content root plan",
            )
            root_scope_hash = _sha256(
                caller_plan.get("development_root_scope_sha256"),
                "development content root scope hash",
            )
            user_agent_hash = _tagged_sha256(
                sec_user_agent_sha256,
                "development SEC execution User-Agent hash",
            )
            existing_claim = current_tip[
                "development_sec_execution_claims"
            ].get(root_scope_hash)
            if existing_claim is not None:
                exact_plan = _validated_development_root_plan_for_store(
                    current,
                    caller_plan,
                    require_latest_candidate=False,
                )
                if (
                    existing_claim.get("sec_user_agent_sha256")
                    != user_agent_hash
                ):
                    raise SecFilingGemmaRevealStoreError(
                        "Development SEC execution contact differs from its durable claim"
                    )
                if existing_claim.get("development_content_root_plan") != exact_plan:
                    raise SecFilingGemmaRevealStoreError(
                        "Development SEC execution plan differs from its durable claim"
                    )
                return {
                    "claim": copy.deepcopy(existing_claim),
                    "created": False,
                    "reader_receipt": copy.deepcopy(
                        current_tip["development_sec_reader_receipts"].get(
                            root_scope_hash
                        )
                    ),
                    "abort": copy.deepcopy(
                        current_tip["development_sec_execution_aborts"].get(
                            root_scope_hash
                        )
                    ),
                }
            exact_plan = _validated_development_root_plan_for_store(
                current,
                caller_plan,
                require_latest_candidate=True,
            )
            execution_sources = _execution_source_hashes(self.repository_root)
            try:
                claim = build_development_sec_execution_claim(
                    current,
                    development_content_root_plan=exact_plan,
                    independent_current_tip_anchor=current_tip,
                    execution_source_hashes=execution_sources,
                    sec_user_agent_sha256=user_agent_hash,
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Could not claim the exact development SEC content root"
                ) from exc
            committed_state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                development_sec_execution_claim=claim,
            )
            if committed_state != current:
                raise SecFilingGemmaRevealStoreError(
                    "Development SEC execution claim changed reveal-store state"
                )
            persisted = committed_tip["development_sec_execution_claims"].get(
                root_scope_hash
            )
            if persisted != claim:
                raise SecFilingGemmaRevealStoreError(
                    "Committed development SEC claim differs from its CAS target"
                )
            return {
                "claim": copy.deepcopy(persisted),
                "created": True,
                "reader_receipt": None,
                "abort": None,
            }

    def abort_owned_development_sec_root_execution(
        self,
        *,
        development_root_scope_sha256: str,
        reason: str,
    ) -> dict[str, Any]:
        """Terminally close an indeterminate development-root SEC claim."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            root_scope_hash = _sha256(
                development_root_scope_sha256,
                "development SEC abort root scope hash",
            )
            claim = current_tip["development_sec_execution_claims"].get(
                root_scope_hash
            )
            if type(claim) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Development SEC execution cannot abort before its durable claim"
                )
            if root_scope_hash in current_tip["development_sec_reader_receipts"]:
                raise SecFilingGemmaRevealStoreError(
                    "Completed development SEC execution cannot be aborted"
                )
            existing = current_tip["development_sec_execution_aborts"].get(
                root_scope_hash
            )
            try:
                abort = build_development_sec_execution_abort(
                    claim,
                    reason=reason,
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Development SEC execution abort is not canonical"
                ) from exc
            if existing is not None:
                if existing != abort:
                    raise SecFilingGemmaRevealStoreError(
                        "Development SEC execution already has another terminal abort"
                    )
                return copy.deepcopy(existing)
            _state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                development_sec_execution_abort=abort,
            )
            persisted = committed_tip["development_sec_execution_aborts"].get(
                root_scope_hash
            )
            if persisted != abort:
                raise SecFilingGemmaRevealStoreError(
                    "Committed development SEC abort differs from its CAS target"
                )
            return copy.deepcopy(persisted)

    def _record_owned_development_sec_root_reader_output(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Replay the immutable development corpus and append its receipt."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            root_scope_hash = _sha256(
                development_root_scope_sha256,
                "development SEC reader root scope hash",
            )
            claim = current_tip["development_sec_execution_claims"].get(
                root_scope_hash
            )
            if type(claim) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Development SEC reader output lacks its durable execution claim"
                )
            if root_scope_hash in current_tip["development_sec_execution_aborts"]:
                raise SecFilingGemmaRevealStoreError(
                    "Aborted development SEC execution cannot publish reader bytes"
                )
            existing = current_tip["development_sec_reader_receipts"].get(
                root_scope_hash
            )
            self._revalidate_authorized_sec_execution_sources(claim)
            plan = _validated_development_root_plan_for_store(
                current,
                claim.get("development_content_root_plan"),
                require_latest_candidate=False,
            )
            if (
                plan["development_root_scope_sha256"] != root_scope_hash
                or plan["development_content_root_plan_sha256"]
                != claim.get("development_content_root_plan_sha256")
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development SEC reader crossed its claimed root plan"
                )
            component_directory = (
                self.store_directory
                / STAGE_OUTPUTS_DIRECTORY_NAME
                / claim["claim_sha256"]
                / SEC_STAGE_COMPONENT_DIRECTORY_NAME
            )
            secured = _secure_directory(
                component_directory,
                create=False,
                location="owned development SEC root directory",
            )
            marker_path = secured / SEC_BATCH_COMPLETE_MARKER_FILENAME
            marker_bytes = _read_regular_bytes(
                marker_path,
                "owned development SEC root complete marker",
                max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
            )
            marker = _strict_json_bytes(
                marker_bytes,
                "owned development SEC root complete marker",
            )
            expected_marker_keys = {
                "schema_version",
                "development_root_scope_sha256",
                "claim_sha256",
                "candidate_sha256",
                "corpus_universe_sha256",
                "development_content_root_plan_sha256",
                "component_id",
                "development_content_manifest_sha256",
                "byte_index",
                "byte_index_sha256",
                "marker_sha256",
            }
            if type(marker) is not dict or set(marker) != expected_marker_keys:
                raise SecFilingGemmaRevealStoreError(
                    "Owned development SEC root marker keys changed"
                )
            marker_body = {
                key: marker[key] for key in marker if key != "marker_sha256"
            }
            if (
                marker["schema_version"]
                != DEVELOPMENT_SEC_ROOT_COMPLETE_MARKER_SCHEMA_VERSION
                or marker["development_root_scope_sha256"] != root_scope_hash
                or marker["claim_sha256"] != claim["claim_sha256"]
                or marker["candidate_sha256"] != claim["candidate_sha256"]
                or marker["corpus_universe_sha256"]
                != claim["corpus_universe_sha256"]
                or marker["development_content_root_plan_sha256"]
                != claim["development_content_root_plan_sha256"]
                or marker["component_id"]
                != DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID
                or marker["marker_sha256"] != canonical_sha256(marker_body)
                or marker_bytes != _encoded_state(marker)
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned development SEC root marker is not canonical or claim-bound"
                )
            byte_index, payloads_by_name = _read_owned_sec_indexed_payloads(
                secured,
                marker["byte_index"],
                location="owned development SEC root directory",
            )
            if marker["byte_index_sha256"] != canonical_sha256(byte_index):
                raise SecFilingGemmaRevealStoreError(
                    "Owned development SEC root byte index is not canonical"
                )
            documents_plan = plan["sec_access_plan"]["documents"]
            expected_layout: list[tuple[str, str]] = []
            for document_ordinal in range(1, len(documents_plan) + 1):
                prefix = f"document-{document_ordinal:04d}"
                expected_layout.extend(
                    (
                        (f"{prefix}-raw", f"{prefix}.raw"),
                        (f"{prefix}-normalized", f"{prefix}.normalized.txt"),
                    )
                )
            expected_layout.extend(
                (
                    ("request-receipts-json", "request-receipts.json"),
                    ("byte-manifest-json", "byte-manifest.json"),
                    ("corpus-universe-json", "corpus-universe.json"),
                    (
                        "development-content-manifest-json",
                        "development-content-manifest.json",
                    ),
                )
            )
            observed_layout = [
                (item["logical_id"], item["relative_path"])
                for item in byte_index
            ]
            if observed_layout != expected_layout:
                raise SecFilingGemmaRevealStoreError(
                    "Owned development SEC root lacks its exact evidence layout"
                )
            budgets = plan["budgets"]
            try:
                _validate_persisted_authenticated_stage_access_batch(
                    authenticated_document_plan=documents_plan,
                    raw_documents=tuple(
                        payloads_by_name[f"document-{ordinal:04d}.raw"]
                        for ordinal in range(1, len(documents_plan) + 1)
                    ),
                    normalized_documents=tuple(
                        payloads_by_name[
                            f"document-{ordinal:04d}.normalized.txt"
                        ]
                        for ordinal in range(1, len(documents_plan) + 1)
                    ),
                    request_receipts_json=payloads_by_name[
                        "request-receipts.json"
                    ],
                    byte_manifest_json=payloads_by_name["byte-manifest.json"],
                    expected_max_requests=budgets["max_sec_requests"],
                    expected_max_bytes=budgets["max_raw_batch_bytes"],
                    expected_max_seconds=float(
                        budgets["max_sec_acquisition_seconds"]
                    ),
                    expected_user_agent_sha256=claim["sec_user_agent_sha256"],
                )
            except (
                KeyError,
                SecFilingGemmaCorpusError,
                TypeError,
                ValueError,
            ) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Owned development SEC root bytes fail independent semantic replay"
                ) from exc
            universe_bytes = payloads_by_name["corpus-universe.json"]
            universe = _strict_json_bytes(
                universe_bytes,
                "owned development SEC root corpus universe",
            )
            if (
                type(universe) is not dict
                or universe != plan["corpus_universe_manifest"]
                or universe_bytes != _encoded_state(universe)
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned development SEC root universe differs from its claim"
                )
            actual_content_documents = []
            for ordinal, document_plan in enumerate(documents_plan, start=1):
                raw_document = payloads_by_name[f"document-{ordinal:04d}.raw"]
                normalized_document = payloads_by_name[
                    f"document-{ordinal:04d}.normalized.txt"
                ]
                actual_content_documents.append(
                    {
                        "accession_number": document_plan["accession_number"],
                        "primary_document_sha256": hashlib.sha256(
                            raw_document
                        ).hexdigest(),
                        "normalized_text_sha256": hashlib.sha256(
                            normalized_document
                        ).hexdigest(),
                        "primary_document_bytes": len(raw_document),
                        "normalized_text_bytes": len(normalized_document),
                    }
                )
            try:
                expected_content_manifest = build_stage_content_manifest(
                    artifact_stage="development",
                    corpus_universe_sha256=plan["corpus_provenance"][
                        "corpus_universe_sha256"
                    ],
                    documents=actual_content_documents,
                    universe_manifest=universe,
                )
                content_manifest_bytes = payloads_by_name[
                    "development-content-manifest.json"
                ]
                content_manifest = _strict_json_bytes(
                    content_manifest_bytes,
                    "owned development SEC root content manifest",
                )
                validate_stage_content_manifest(
                    content_manifest,
                    universe_manifest=universe,
                    expected_content_manifest_sha256=marker[
                        "development_content_manifest_sha256"
                    ],
                )
            except (
                KeyError,
                SecFilingGemmaContractError,
                TypeError,
                ValueError,
            ) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Owned development SEC root content manifest does not replay"
                ) from exc
            if (
                type(content_manifest) is not dict
                or content_manifest != expected_content_manifest
                or content_manifest_bytes != _encoded_state(content_manifest)
                or marker["development_content_manifest_sha256"]
                != expected_content_manifest["content_manifest_sha256"]
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned development SEC root content manifest differs from exact bytes"
                )
            try:
                self._revalidate_authorized_sec_execution_sources(claim)
                receipt = build_development_sec_reader_receipt(
                    claim,
                    content_manifest_sha256=expected_content_manifest[
                        "content_manifest_sha256"
                    ],
                    byte_index=byte_index,
                    complete_marker_sha256=hashlib.sha256(marker_bytes).hexdigest(),
                )
            except (
                SecFilingGemmaRevealStoreError,
                SecFilingGemmaStageAuthorizationError,
            ) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Owned development SEC reader receipt could not be finalized"
                ) from exc
            if existing is not None:
                if existing != receipt:
                    raise SecFilingGemmaRevealStoreError(
                        "Persisted development SEC receipt differs from replayed bytes"
                    )
                return copy.deepcopy(existing)
            _state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                development_sec_reader_receipt=receipt,
            )
            persisted = committed_tip["development_sec_reader_receipts"].get(
                root_scope_hash
            )
            if persisted != receipt:
                raise SecFilingGemmaRevealStoreError(
                    "Committed development SEC receipt differs from rehashed bytes"
                )
            return copy.deepcopy(persisted)

    def claim_owned_development_market_execution(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Atomically claim the fixed six-call development market acquisition."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            scope_hash = _sha256(
                development_root_scope_sha256,
                "development market execution root scope hash",
            )
            existing_claim = current_tip[
                "development_market_execution_claims"
            ].get(scope_hash)
            if existing_claim is not None:
                return {
                    "claim": copy.deepcopy(existing_claim),
                    "created": False,
                    "reader_receipt": copy.deepcopy(
                        current_tip["development_market_reader_receipts"].get(
                            scope_hash
                        )
                    ),
                    "abort": copy.deepcopy(
                        current_tip["development_market_execution_aborts"].get(
                            scope_hash
                        )
                    ),
                }
            try:
                claim = build_development_market_execution_claim(
                    current,
                    development_root_scope_sha256=scope_hash,
                    independent_current_tip_anchor=current_tip,
                    market_acquisition_plan=(
                        build_development_market_acquisition_plan()
                    ),
                    execution_source_hashes=_market_execution_source_hashes(
                        self.repository_root
                    ),
                )
            except Exception as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Could not claim the exact development market batch"
                ) from exc
            committed_state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                development_market_execution_claim=claim,
            )
            if committed_state != current:
                raise SecFilingGemmaRevealStoreError(
                    "Development market claim changed reveal-store state"
                )
            persisted = committed_tip[
                "development_market_execution_claims"
            ].get(scope_hash)
            if persisted != claim:
                raise SecFilingGemmaRevealStoreError(
                    "Committed development market claim differs from its CAS target"
                )
            return {
                "claim": copy.deepcopy(persisted),
                "created": True,
                "reader_receipt": None,
                "abort": None,
            }

    def abort_owned_development_market_execution(
        self,
        *,
        development_root_scope_sha256: str,
        reason: str,
    ) -> dict[str, Any]:
        """Terminally close an indeterminate market claim without retrying I/O."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            scope_hash = _sha256(
                development_root_scope_sha256,
                "development market abort root scope hash",
            )
            claim = current_tip["development_market_execution_claims"].get(
                scope_hash
            )
            if type(claim) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Development market execution cannot abort before its claim"
                )
            if scope_hash in current_tip["development_market_reader_receipts"]:
                raise SecFilingGemmaRevealStoreError(
                    "Completed development market execution cannot be aborted"
                )
            existing = current_tip["development_market_execution_aborts"].get(
                scope_hash
            )
            try:
                abort = build_development_market_execution_abort(
                    claim,
                    reason=reason,
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Development market execution abort is not canonical"
                ) from exc
            if existing is not None:
                if existing != abort:
                    raise SecFilingGemmaRevealStoreError(
                        "Development market execution already has another terminal abort"
                    )
                return copy.deepcopy(existing)
            _state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                development_market_execution_abort=abort,
            )
            persisted = committed_tip[
                "development_market_execution_aborts"
            ].get(scope_hash)
            if persisted != abort:
                raise SecFilingGemmaRevealStoreError(
                    "Committed development market abort differs from its CAS target"
                )
            return copy.deepcopy(persisted)

    def _revalidate_authorized_market_execution_sources(
        self,
        claim: Mapping[str, Any],
    ) -> None:
        """Reject source or acquisition-plan substitution after market claim."""

        if type(claim) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "Market execution source revalidation requires an exact claim"
            )
        if _market_execution_source_hashes(self.repository_root) != claim.get(
            "execution_source_hashes"
        ):
            raise SecFilingGemmaRevealStoreError(
                "Market execution sources changed after the durable claim"
            )
        if build_development_market_acquisition_plan() != claim.get(
            "market_acquisition_plan"
        ):
            raise SecFilingGemmaRevealStoreError(
                "Market acquisition plan changed after the durable claim"
            )

    def _replay_owned_development_market_component_locked(
        self,
        *,
        claim: Mapping[str, Any],
        development_root_scope_sha256: str,
        expected_reader_receipt: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Fully replay one owned market component without exposing source bytes."""

        claim_value = _expect_mapping(
            claim,
            "owned development market replay claim",
        )
        scope_hash = _sha256(
            development_root_scope_sha256,
            "owned development market replay root scope hash",
        )
        if claim_value.get("development_root_scope_sha256") != scope_hash:
            raise SecFilingGemmaRevealStoreError(
                "Owned development market replay crossed its root scope"
            )
        if expected_reader_receipt is not None and type(expected_reader_receipt) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "Owned development market replay reader receipt is not exact"
            )
        self._revalidate_authorized_market_execution_sources(claim_value)
        claim_hash = _sha256(
            claim_value.get("claim_sha256"),
            "development market replay claim hash",
        )
        component_directory = (
            self.store_directory
            / STAGE_OUTPUTS_DIRECTORY_NAME
            / claim_hash
            / MARKET_SOURCE_COMPONENT_DIRECTORY_NAME
        )
        secured = _secure_directory(
            component_directory,
            create=False,
            location="owned development market component directory",
        )
        directory_before = _validate_real_directory(
            secured,
            "owned development market component directory",
        )
        marker_path = secured / DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME
        marker_bytes = _read_regular_bytes(
            marker_path,
            "owned development market complete marker",
            max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
        )
        marker = _strict_json_bytes(
            marker_bytes,
            "owned development market complete marker",
        )
        if type(marker) is not dict or set(marker) != set(
            _DEVELOPMENT_MARKET_COMPLETE_MARKER_KEYS
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned development market marker keys changed"
            )
        marker_body = {
            key: value for key, value in marker.items() if key != "marker_sha256"
        }
        if (
            marker.get("schema_version")
            != DEVELOPMENT_MARKET_COMPLETE_MARKER_SCHEMA_VERSION
            or marker.get("development_root_scope_sha256") != scope_hash
            or marker.get("claim_sha256") != claim_hash
            or marker.get("market_acquisition_plan_sha256")
            != claim_value.get("market_acquisition_plan_sha256")
            or marker.get("market_component_id") != DEVELOPMENT_MARKET_COMPONENT_ID
            or marker.get("marker_sha256") != canonical_sha256(marker_body)
            or marker_bytes != _encoded_state(marker)
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned development market marker is not canonical or claim-bound"
            )
        prevalidated_byte_index = _prevalidate_owned_market_byte_index(
            secured,
            marker,
            marker_byte_count=len(marker_bytes),
        )
        market_file_caps = {
            f"raw-response-{symbol}.json": YAHOO_MAX_RESPONSE_BYTES
            for symbol in MARKET_SYMBOLS
        }
        byte_index, payloads = _read_owned_sec_indexed_payloads(
            secured,
            prevalidated_byte_index,
            location="owned development market component directory",
            maximum_files=MAX_MARKET_BATCH_FILES,
            maximum_file_bytes=MAX_MARKET_BATCH_FILE_BYTES,
            maximum_total_bytes=MAX_MARKET_BATCH_TOTAL_BYTES,
            complete_marker_filename=DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME,
            maximum_file_bytes_by_name=market_file_caps,
        )
        if (
            len(byte_index) + 1 != MAX_MARKET_BATCH_FILES
            or marker.get("byte_index_sha256") != canonical_sha256(byte_index)
            or marker.get("byte_count_total")
            != sum(item["byte_count"] for item in byte_index)
            or marker["byte_count_total"] > MAX_MARKET_BATCH_TOTAL_BYTES
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned development market byte index is inconsistent"
            )
        expected_layout = _expected_market_byte_layout()
        observed_layout = [
            (item["logical_id"], item["relative_path"]) for item in byte_index
        ]
        if observed_layout != expected_layout:
            raise SecFilingGemmaRevealStoreError(
                "Owned development market component lacks its exact layout"
            )

        raw_bytes = {
            symbol: payloads[f"raw-response-{symbol}.json"]
            for symbol in MARKET_SYMBOLS
        }
        if sum(len(raw_bytes[symbol]) for symbol in MARKET_SYMBOLS) > (
            YAHOO_MAX_TOTAL_RESPONSE_BYTES
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned development market raw responses exceed their aggregate "
                "byte ceiling"
            )
        artifact_bytes = {
            symbol: payloads[f"artifact-{symbol}.json"] for symbol in MARKET_SYMBOLS
        }
        window_bytes = {
            symbol: payloads[f"window-{symbol}.json"] for symbol in MARKET_SYMBOLS
        }
        canonical_documents: dict[str, dict[str, Any]] = {}
        for filename in (
            "source-manifest.json",
            "stage-manifest.json",
            "reconciliation-receipt.json",
            "acquisition-receipt.json",
        ):
            parsed = _strict_json_bytes(
                payloads[filename],
                f"owned development market {filename}",
            )
            if type(parsed) is not dict or payloads[filename] != _encoded_state(parsed):
                raise SecFilingGemmaRevealStoreError(
                    f"Owned development market {filename} is not canonical"
                )
            canonical_documents[filename] = parsed
        source_manifest = canonical_documents["source-manifest.json"]
        stage_manifest = canonical_documents["stage-manifest.json"]
        reconciliation = canonical_documents["reconciliation-receipt.json"]
        acquisition_receipt = canonical_documents["acquisition-receipt.json"]
        policy = acquisition_receipt.get("transport_policy")
        if (
            type(policy) is not dict
            or policy.get("trusted_production_transport") is not True
            or policy.get("authorizes_production_use") is not False
            or policy.get("fresh_network_provenance_claimed") is not False
            or policy.get("network_occurrence_is_only_locally_observed") is not True
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned development market transport claims are not honest"
            )
        raw_hashes = {
            symbol: hashlib.sha256(raw_bytes[symbol]).hexdigest()
            for symbol in MARKET_SYMBOLS
        }
        artifact_hashes = {
            symbol: hashlib.sha256(artifact_bytes[symbol]).hexdigest()
            for symbol in MARKET_SYMBOLS
        }
        window_hashes = {
            symbol: hashlib.sha256(window_bytes[symbol]).hexdigest()
            for symbol in MARKET_SYMBOLS
        }
        compact_body = {
            "schema_version": MARKET_ACQUISITION_BUNDLE_SCHEMA_VERSION,
            "artifact_stage": "development",
            "acquisition_receipt_sha256": acquisition_receipt.get(
                "acquisition_receipt_sha256"
            ),
            "acquisition_plan_sha256": claim_value[
                "market_acquisition_plan_sha256"
            ],
            "source_manifest_sha256": source_manifest.get("source_manifest_sha256"),
            "market_stage_manifest_sha256": stage_manifest.get(
                "market_stage_manifest_sha256"
            ),
            "source_reconciliation_sha256": reconciliation.get(
                "reconciliation_sha256"
            ),
            "raw_response_sha256s": raw_hashes,
            "artifact_sha256s": artifact_hashes,
            "window_sha256s": window_hashes,
        }
        bundle_hash = canonical_sha256(compact_body)
        bundle = {
            **compact_body,
            "bundle_sha256": bundle_hash,
            "raw_response_bytes_by_symbol": raw_bytes,
            "artifact_bytes_by_symbol": artifact_bytes,
            "window_bytes_by_symbol": window_bytes,
            "source_manifest": source_manifest,
            "stage_manifest": stage_manifest,
            "reconciliation_receipt": reconciliation,
            "acquisition_receipt": acquisition_receipt,
        }
        try:
            validation = validate_development_market_acquisition_bundle(
                bundle,
                expected_acquisition_plan_sha256=claim_value[
                    "market_acquisition_plan_sha256"
                ],
                expected_acquisition_receipt_sha256=acquisition_receipt[
                    "acquisition_receipt_sha256"
                ],
                expected_bundle_sha256=bundle_hash,
            )
        except Exception as exc:
            raise SecFilingGemmaRevealStoreError(
                "Owned development market bytes fail semantic replay"
            ) from exc
        validation_hash = validation.get("validation_sha256")
        if (
            marker.get("acquisition_receipt_sha256")
            != acquisition_receipt.get("acquisition_receipt_sha256")
            or marker.get("acquisition_bundle_sha256") != bundle_hash
            or marker.get("acquisition_validation_sha256") != validation_hash
            or marker.get("source_manifest_sha256")
            != source_manifest.get("source_manifest_sha256")
            or marker.get("market_stage_manifest_sha256")
            != stage_manifest.get("market_stage_manifest_sha256")
            or marker.get("source_reconciliation_sha256")
            != reconciliation.get("reconciliation_sha256")
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned development market marker differs from semantic replay"
            )

        byte_index_after, payloads_after = _read_owned_sec_indexed_payloads(
            secured,
            prevalidated_byte_index,
            location="owned development market component directory",
            maximum_files=MAX_MARKET_BATCH_FILES,
            maximum_file_bytes=MAX_MARKET_BATCH_FILE_BYTES,
            maximum_total_bytes=MAX_MARKET_BATCH_TOTAL_BYTES,
            complete_marker_filename=DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME,
            maximum_file_bytes_by_name=market_file_caps,
        )
        marker_after = _read_regular_bytes(
            marker_path,
            "owned development market complete marker",
            max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
        )
        directory_after = _validate_real_directory(
            secured,
            "owned development market component directory",
        )
        if (
            marker_after != marker_bytes
            or byte_index_after != byte_index
            or payloads_after != payloads
            or (directory_before.st_dev, directory_before.st_ino)
            != (directory_after.st_dev, directory_after.st_ino)
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned development market component changed during replay"
            )
        self._revalidate_authorized_market_execution_sources(claim_value)
        try:
            receipt = build_development_market_reader_receipt(
                claim_value,
                acquisition_receipt_sha256=acquisition_receipt[
                    "acquisition_receipt_sha256"
                ],
                acquisition_bundle_sha256=bundle_hash,
                acquisition_validation_sha256=validation_hash,
                source_manifest_sha256=source_manifest["source_manifest_sha256"],
                market_stage_manifest_sha256=stage_manifest[
                    "market_stage_manifest_sha256"
                ],
                source_reconciliation_sha256=reconciliation[
                    "reconciliation_sha256"
                ],
                raw_response_sha256s=raw_hashes,
                artifact_sha256s=artifact_hashes,
                window_sha256s=window_hashes,
                byte_index=byte_index,
                complete_marker_sha256=hashlib.sha256(marker_bytes).hexdigest(),
                owned_transport_attested_by_store=True,
            )
        except Exception as exc:
            raise SecFilingGemmaRevealStoreError(
                "Owned development market reader receipt could not be finalized"
            ) from exc
        if expected_reader_receipt is not None and receipt != expected_reader_receipt:
            raise SecFilingGemmaRevealStoreError(
                "Persisted development market receipt differs from replayed bytes"
            )
        return {
            "reader_receipt": copy.deepcopy(receipt),
            "source_manifest": copy.deepcopy(source_manifest),
            "stage_manifest": copy.deepcopy(stage_manifest),
        }

    def _record_owned_development_market_reader_output(
        self,
        *,
        development_root_scope_sha256: str,
        owned_transport_capability: object | None = None,
    ) -> dict[str, Any]:
        """Reparse the fixed raw market component and commit its exact receipt."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            scope_hash = _sha256(
                development_root_scope_sha256,
                "development market reader root scope hash",
            )
            claim = current_tip["development_market_execution_claims"].get(
                scope_hash
            )
            if type(claim) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Development market reader lacks its durable claim"
                )
            if scope_hash in current_tip["development_market_execution_aborts"]:
                raise SecFilingGemmaRevealStoreError(
                    "Aborted development market execution has no readable output"
                )
            existing = current_tip["development_market_reader_receipts"].get(
                scope_hash
            )
            if existing is not None and owned_transport_capability is not None:
                raise SecFilingGemmaRevealStoreError(
                    "Committed development market receipts replay without capabilities"
                )
            replay = self._replay_owned_development_market_component_locked(
                claim=claim,
                development_root_scope_sha256=scope_hash,
                expected_reader_receipt=existing,
            )
            receipt = replay["reader_receipt"]
            if existing is not None:
                return copy.deepcopy(existing)
            try:
                _consume_owned_market_transport_capability(
                    owned_transport_capability,
                    development_root_scope_sha256=scope_hash,
                    claim_sha256=claim["claim_sha256"],
                    acquisition_plan_sha256=claim[
                        "market_acquisition_plan_sha256"
                    ],
                    acquisition_receipt_sha256=receipt[
                        "acquisition_receipt_sha256"
                    ],
                    bundle_sha256=receipt["acquisition_bundle_sha256"],
                    validation_sha256=receipt["acquisition_validation_sha256"],
                    source_manifest_sha256=receipt["source_manifest_sha256"],
                    market_stage_manifest_sha256=receipt[
                        "market_stage_manifest_sha256"
                    ],
                    source_reconciliation_sha256=receipt[
                        "source_reconciliation_sha256"
                    ],
                )
            except Exception as exc:
                raise SecFilingGemmaRevealStoreError(
                    "A new development market receipt requires the matching "
                    "same-execution owned Yahoo transport capability"
                ) from exc
            _state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                development_market_reader_receipt=receipt,
            )
            persisted = committed_tip[
                "development_market_reader_receipts"
            ].get(scope_hash)
            if persisted != receipt:
                raise SecFilingGemmaRevealStoreError(
                    "Committed development market receipt differs from rehashed bytes"
                )
            return copy.deepcopy(persisted)

    def claim_owned_development_model_execution(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Atomically claim the exact request-free development Gemma batch."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            scope_hash = _sha256(
                development_root_scope_sha256,
                "development model execution root scope hash",
            )
            existing_claim = current_tip[
                "development_model_execution_claims"
            ].get(scope_hash)
            if existing_claim is not None:
                return {
                    "claim": copy.deepcopy(existing_claim),
                    "created": False,
                    "reader_receipt": copy.deepcopy(
                        current_tip["development_model_reader_receipts"].get(
                            scope_hash
                        )
                    ),
                    "abort": copy.deepcopy(
                        current_tip["development_model_execution_aborts"].get(
                            scope_hash
                        )
                    ),
                }
            try:
                claim = build_development_model_execution_claim(
                    current,
                    development_root_scope_sha256=scope_hash,
                    independent_current_tip_anchor=current_tip,
                    execution_source_hashes=_model_execution_source_hashes(
                        self.repository_root
                    ),
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Could not claim the exact development model batch"
                ) from exc
            committed_state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                development_model_execution_claim=claim,
            )
            if committed_state != current:
                raise SecFilingGemmaRevealStoreError(
                    "Development model claim changed reveal-store state"
                )
            persisted = committed_tip["development_model_execution_claims"].get(
                scope_hash
            )
            if persisted != claim:
                raise SecFilingGemmaRevealStoreError(
                    "Committed development model claim differs from its CAS target"
                )
            return {
                "claim": copy.deepcopy(persisted),
                "created": True,
                "reader_receipt": None,
                "abort": None,
            }

    def abort_owned_development_model_execution(
        self,
        *,
        development_root_scope_sha256: str,
        reason: str,
    ) -> dict[str, Any]:
        """Terminally close an indeterminate development model effect."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            scope_hash = _sha256(
                development_root_scope_sha256,
                "development model abort root scope hash",
            )
            claim = current_tip["development_model_execution_claims"].get(
                scope_hash
            )
            if type(claim) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Development model execution cannot abort before its durable claim"
                )
            if scope_hash in current_tip["development_model_reader_receipts"]:
                raise SecFilingGemmaRevealStoreError(
                    "Completed development model execution cannot be aborted"
                )
            existing = current_tip["development_model_execution_aborts"].get(
                scope_hash
            )
            try:
                abort = build_development_model_execution_abort(
                    claim,
                    reason=reason,
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Development model execution abort is not canonical"
                ) from exc
            if existing is not None:
                if existing != abort:
                    raise SecFilingGemmaRevealStoreError(
                        "Development model execution already has another terminal abort"
                    )
                return copy.deepcopy(existing)
            _state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                development_model_execution_abort=abort,
            )
            persisted = committed_tip["development_model_execution_aborts"].get(
                scope_hash
            )
            if persisted != abort:
                raise SecFilingGemmaRevealStoreError(
                    "Committed development model abort differs from its CAS target"
                )
            return copy.deepcopy(persisted)

    def claim_authorized_model_stage_execution(
        self,
        *,
        request_sha256: str,
    ) -> dict[str, Any]:
        """Atomically claim one exact intermediate/final Gemma batch."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            request_hash = _sha256(
                request_sha256,
                "authorized model execution request hash",
            )
            existing_claim = current_tip["stage_model_execution_claims"].get(
                request_hash
            )
            if existing_claim is not None:
                return {
                    "claim": copy.deepcopy(existing_claim),
                    "created": False,
                    "reader_receipt": copy.deepcopy(
                        current_tip["stage_model_reader_receipts"].get(
                            request_hash
                        )
                    ),
                    "abort": copy.deepcopy(
                        current_tip["stage_model_execution_aborts"].get(
                            request_hash
                        )
                    ),
                }
            bundle = current_tip["authorization_bundles"].get(request_hash)
            if type(bundle) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Authorized model execution lacks its current grant bundle"
                )
            try:
                claim = build_stage_model_execution_claim(
                    bundle,
                    independent_current_tip_anchor=current_tip,
                    execution_source_hashes=_model_execution_source_hashes(
                        self.repository_root
                    ),
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Could not claim the exact current model stage grant"
                ) from exc
            committed_state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                stage_model_execution_claim=claim,
            )
            if committed_state != current:
                raise SecFilingGemmaRevealStoreError(
                    "Stage model execution claim changed reveal-store state"
                )
            persisted = committed_tip["stage_model_execution_claims"].get(
                request_hash
            )
            if persisted != claim:
                raise SecFilingGemmaRevealStoreError(
                    "Committed stage model claim differs from its CAS target"
                )
            return {
                "claim": copy.deepcopy(persisted),
                "created": True,
                "reader_receipt": None,
                "abort": None,
            }

    def abort_authorized_model_stage_execution(
        self,
        *,
        request_sha256: str,
        reason: str,
    ) -> dict[str, Any]:
        """Terminally close an indeterminate stage model effect."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            request_hash = _sha256(request_sha256, "stage model abort request hash")
            claim = current_tip["stage_model_execution_claims"].get(request_hash)
            if type(claim) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Stage model execution cannot abort before its durable claim"
                )
            if request_hash in current_tip["stage_model_reader_receipts"]:
                raise SecFilingGemmaRevealStoreError(
                    "Completed stage model execution cannot be aborted"
                )
            existing = current_tip["stage_model_execution_aborts"].get(
                request_hash
            )
            try:
                abort = build_stage_model_execution_abort(claim, reason=reason)
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Stage model execution abort is not canonical"
                ) from exc
            if existing is not None:
                if existing != abort:
                    raise SecFilingGemmaRevealStoreError(
                        "Stage model execution already has another terminal abort"
                    )
                return copy.deepcopy(existing)
            _state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                stage_model_execution_abort=abort,
            )
            persisted = committed_tip["stage_model_execution_aborts"].get(
                request_hash
            )
            if persisted != abort:
                raise SecFilingGemmaRevealStoreError(
                    "Committed stage model abort differs from its CAS target"
                )
            return copy.deepcopy(persisted)

    def _revalidate_authorized_model_execution_sources(
        self,
        claim: Mapping[str, Any],
    ) -> None:
        """Reject repository-source substitution after a model claim was minted."""

        if type(claim) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "Model execution source revalidation requires an exact claim"
            )
        if _model_execution_source_hashes(self.repository_root) != claim.get(
            "execution_source_hashes"
        ):
            raise SecFilingGemmaRevealStoreError(
                "Model execution sources changed after the durable claim"
            )

    def _read_owned_model_component_locked(
        self,
        *,
        claim: Mapping[str, Any],
        lifecycle_kind: str,
        lifecycle_sha256: str,
        marker_schema_version: str,
    ) -> tuple[list[dict[str, Any]], dict[str, bytes], bytes]:
        """Rehash one fixed model component and its marker under the store lock."""

        if type(claim) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "Owned model output lacks its exact execution claim"
            )
        lifecycle_hash = _sha256(
            lifecycle_sha256,
            "owned model component lifecycle hash",
        )
        if lifecycle_kind not in {"development_root_scope", "stage_request"}:
            raise SecFilingGemmaRevealStoreError(
                "Owned model component lifecycle kind changed"
            )
        self._revalidate_authorized_model_execution_sources(claim)
        component_directory = (
            self.store_directory
            / STAGE_OUTPUTS_DIRECTORY_NAME
            / claim["claim_sha256"]
            / MODEL_EXTRACTION_COMPONENT_DIRECTORY_NAME
        )
        secured = _secure_directory(
            component_directory,
            create=False,
            location="owned model extraction component directory",
        )
        directory_before = _validate_real_directory(
            secured,
            "owned model extraction component directory",
        )
        marker_path = secured / MODEL_EXTRACTION_COMPLETE_MARKER_FILENAME
        marker_bytes = _read_regular_bytes(
            marker_path,
            "owned model extraction complete marker",
            max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
        )
        marker = _strict_json_bytes(
            marker_bytes,
            "owned model extraction complete marker",
        )
        expected_marker_keys = {
            "schema_version",
            "lifecycle_kind",
            "lifecycle_sha256",
            "claim_sha256",
            "candidate_sha256",
            "authorized_stage",
            "output_namespace",
            "model_component_id",
            "event_count",
            "event_plan_sha256",
            "identity_lexicon_sha256",
            "execution_source_hashes_sha256",
            "model_name",
            "model_digest",
            "runtime_fingerprint_sha256",
            "model_transport_sha256",
            "model_runtime_limits_sha256",
            "byte_index",
            "byte_index_sha256",
            "byte_count_total",
            "marker_sha256",
        }
        marker_body = (
            {key: marker[key] for key in marker if key != "marker_sha256"}
            if type(marker) is dict
            else {}
        )
        claim_bindings = {
            "claim_sha256": claim.get("claim_sha256"),
            "candidate_sha256": claim.get("candidate_sha256"),
            "authorized_stage": claim.get("authorized_stage"),
            "output_namespace": claim.get("output_namespace"),
            "model_component_id": claim.get("model_component_id"),
            "event_count": claim.get("event_count"),
            "event_plan_sha256": claim.get("event_plan_sha256"),
            "identity_lexicon_sha256": claim.get("identity_lexicon_sha256"),
            "execution_source_hashes_sha256": claim.get(
                "execution_source_hashes_sha256"
            ),
            "model_name": claim.get("model_name"),
            "model_digest": claim.get("model_digest"),
            "runtime_fingerprint_sha256": claim.get(
                "runtime_fingerprint_sha256"
            ),
            "model_transport_sha256": claim.get("model_transport_sha256"),
            "model_runtime_limits_sha256": claim.get(
                "model_runtime_limits_sha256"
            ),
        }
        if (
            type(marker) is not dict
            or set(marker) != expected_marker_keys
            or marker["schema_version"] != marker_schema_version
            or marker["lifecycle_kind"] != lifecycle_kind
            or marker["lifecycle_sha256"] != lifecycle_hash
            or any(marker.get(key) != value for key, value in claim_bindings.items())
            or marker["marker_sha256"] != canonical_sha256(marker_body)
            or marker_bytes != _encoded_state(marker)
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned model complete marker is not canonical or claim-bound"
            )
        byte_index, payloads = _read_owned_model_indexed_payloads(
            secured,
            marker["byte_index"],
            event_count=claim["event_count"],
            location="owned model extraction component directory",
        )
        if (
            marker["byte_index_sha256"] != canonical_sha256(byte_index)
            or marker["byte_count_total"]
            != sum(item["byte_count"] for item in byte_index)
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned model byte index is not exact"
            )
        marker_after = _read_regular_bytes(
            marker_path,
            "owned model extraction closure marker",
            max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
        )
        directory_after = _validate_real_directory(
            secured,
            "owned model extraction component directory",
        )
        if (
            marker_after != marker_bytes
            or (directory_before.st_dev, directory_before.st_ino)
            != (directory_after.st_dev, directory_after.st_ino)
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned model component changed during exact replay"
            )
        self._revalidate_authorized_model_execution_sources(claim)
        return byte_index, payloads, marker_bytes

    def _load_owned_model_event_inputs_locked(
        self,
        *,
        authenticated_store_snapshot: Mapping[str, Any],
        independent_current_tip_anchor: Mapping[str, Any],
        claim: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Derive current/prior text only from terminal store-owned ancestry."""

        current = _expect_mapping(
            authenticated_store_snapshot,
            "owned model input store snapshot",
        )
        current_tip = _expect_mapping(
            independent_current_tip_anchor,
            "owned model input current tip",
        )
        claim_value = _expect_mapping(claim, "owned model input claim")
        stage = claim_value.get("authorized_stage")
        if stage == "development":
            scope_hash = _sha256(
                claim_value.get("development_root_scope_sha256"),
                "owned development model input scope",
            )
            if (
                current_tip["development_model_execution_claims"].get(scope_hash)
                != claim_value
                or scope_hash in current_tip["development_model_execution_aborts"]
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development model input claim is not active exact ancestry"
                )
            sec_claim = current_tip["development_sec_execution_claims"].get(
                scope_hash
            )
            sec_reader = current_tip["development_sec_reader_receipts"].get(
                scope_hash
            )
            if (
                type(sec_claim) is not dict
                or type(sec_reader) is not dict
                or sec_claim.get("claim_sha256")
                != claim_value.get("development_sec_execution_claim_sha256")
                or sec_reader.get("receipt_sha256")
                != claim_value.get("development_sec_reader_receipt_sha256")
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development model input lost its terminal SEC root"
                )
            root_source = self._read_owned_development_root_carry_source_locked(
                authenticated_store_snapshot=current,
                independent_current_tip_anchor=current_tip,
                development_sec_execution_claim=sec_claim,
                development_sec_reader_receipt=sec_reader,
            )
            universe = root_source["universe"]
            content_manifest = root_source["content_manifest"]
            content_manifests_by_stage = {
                "development": copy.deepcopy(content_manifest)
            }
            payloads_by_name = root_source["payloads_by_name"]
            byte_rows = {
                row["relative_path"]: row for row in root_source["byte_index"]
            }
            sec_reader_hash = sec_reader["receipt_sha256"]
            carry_reader_hash = None
            initial_prior_by_form: dict[str, dict[str, Any]] = {}
            lifecycle_kind = "development_root"
            lifecycle_hash = scope_hash
        elif stage in {"intermediate", "final"}:
            request_hash = _sha256(
                claim_value.get("request_sha256"),
                "owned stage model input request",
            )
            if (
                current_tip["stage_model_execution_claims"].get(request_hash)
                != claim_value
                or request_hash in current_tip["stage_model_execution_aborts"]
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Stage model input claim is not active exact ancestry"
                )
            bundle = current_tip["authorization_bundles"].get(request_hash)
            sec_claim = current_tip["stage_sec_execution_claims"].get(request_hash)
            sec_reader = current_tip["stage_sec_reader_receipts"].get(request_hash)
            root_scope_hash = claim_value.get("development_root_scope_sha256")
            root_claim = current_tip["development_sec_execution_claims"].get(
                root_scope_hash
            )
            root_reader = current_tip["development_sec_reader_receipts"].get(
                root_scope_hash
            )
            if any(
                type(value) is not dict
                for value in (bundle, sec_claim, sec_reader, root_claim, root_reader)
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Stage model input lost terminal SEC or development ancestry"
                )
            if (
                sec_claim["claim_sha256"]
                != claim_value.get("stage_sec_execution_claim_sha256")
                or sec_reader["receipt_sha256"]
                != claim_value.get("stage_sec_reader_receipt_sha256")
                or root_claim["claim_sha256"]
                != claim_value.get("development_sec_execution_claim_sha256")
                or root_reader["receipt_sha256"]
                != claim_value.get("development_sec_reader_receipt_sha256")
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Stage model input crossed its claim-bound SEC ancestry"
                )
            root_source = self._read_owned_development_root_carry_source_locked(
                authenticated_store_snapshot=current,
                independent_current_tip_anchor=current_tip,
                development_sec_execution_claim=root_claim,
                development_sec_reader_receipt=root_reader,
            )
            universe = root_source["universe"]
            root_content_manifest = root_source["content_manifest"]
            sec_replay = self._replay_owned_sec_reader_bytes_locked(
                claim=sec_claim,
                reader_receipt=sec_reader,
            )
            payloads_by_name = sec_replay["payloads_by_name"]
            byte_rows = {
                row["relative_path"]: row for row in sec_replay["byte_index"]
            }
            try:
                _exact_bundle, _grant, component_plan = (
                    _sec_component_plan_from_bundle(bundle)
                )
                content_documents = []
                for document_ordinal, document in enumerate(
                    component_plan["sec_access_plan"]["documents"],
                    start=1,
                ):
                    raw_payload = payloads_by_name[
                        f"document-{document_ordinal:04d}.raw"
                    ]
                    normalized_payload = payloads_by_name[
                        f"document-{document_ordinal:04d}.normalized.txt"
                    ]
                    content_documents.append(
                        {
                            "accession_number": document["accession_number"],
                            "primary_document_sha256": hashlib.sha256(
                                raw_payload
                            ).hexdigest(),
                            "normalized_text_sha256": hashlib.sha256(
                                normalized_payload
                            ).hexdigest(),
                            "primary_document_bytes": len(raw_payload),
                            "normalized_text_bytes": len(normalized_payload),
                        }
                    )
                content_manifest = build_stage_content_manifest(
                    artifact_stage=stage,
                    corpus_universe_sha256=universe["universe_sha256"],
                    documents=content_documents,
                    universe_manifest=universe,
                )
            except (
                KeyError,
                SecFilingGemmaContractError,
                SecFilingGemmaStageAuthorizationError,
            ) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Stage model content manifest cannot be rebuilt from SEC bytes"
                ) from exc
            content_manifests_by_stage = {
                "development": copy.deepcopy(root_content_manifest)
            }
            if stage == "intermediate":
                content_manifests_by_stage["intermediate"] = copy.deepcopy(
                    content_manifest
                )
            if stage == "intermediate":
                carry_receipt = current_tip[
                    "development_root_carry_in_reader_receipts"
                ].get(request_hash)
                carry_directory_name = (
                    DEVELOPMENT_ROOT_CARRY_IN_COMPONENT_DIRECTORY_NAME
                )
                carry_provenance = "development_root_carry_in"
            else:
                carry_receipt = current_tip["stage_carry_in_reader_receipts"].get(
                    request_hash
                )
                carry_directory_name = PRIOR_SAME_FORM_CARRY_IN_COMPONENT_DIRECTORY_NAME
                carry_provenance = "stage_carry_in"
            if (
                type(carry_receipt) is not dict
                or carry_receipt.get("receipt_sha256")
                != claim_value.get("carry_in_reader_receipt_sha256")
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Stage model input lost its exact carry-in receipt"
                )
            carry_directory = (
                self.store_directory
                / STAGE_OUTPUTS_DIRECTORY_NAME
                / sec_claim["claim_sha256"]
                / carry_directory_name
            )
            carry_marker_path = (
                carry_directory / MODEL_EXTRACTION_COMPLETE_MARKER_FILENAME
            )
            carry_marker_bytes = _read_regular_bytes(
                carry_marker_path,
                "owned model carry-in complete marker",
                max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
            )
            carry_marker = _strict_json_bytes(
                carry_marker_bytes,
                "owned model carry-in complete marker",
            )
            carry_index = carry_receipt.get("carry_in_byte_index")
            if (
                type(carry_marker) is not dict
                or carry_marker_bytes != _encoded_state(carry_marker)
                or carry_marker.get("byte_index") != carry_index
                or hashlib.sha256(carry_marker_bytes).hexdigest()
                != carry_receipt.get("carry_in_complete_marker_sha256")
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Stage model carry-in marker differs from its receipt"
                )
            replayed_carry_index, carry_payloads = _read_owned_sec_indexed_payloads(
                carry_directory,
                carry_index,
                location="owned model carry-in component directory",
            )
            if replayed_carry_index != carry_index:
                raise SecFilingGemmaRevealStoreError(
                    "Stage model carry-in byte index changed"
                )
            try:
                if stage == "intermediate":
                    validate_development_root_carry_in_reader_receipt(
                        carry_receipt,
                        authenticated_store_snapshot=current,
                        independent_current_tip_anchor=current_tip,
                        carry_in_byte_index=replayed_carry_index,
                        carry_in_complete_marker_sha256=hashlib.sha256(
                            carry_marker_bytes
                        ).hexdigest(),
                        reader_source_sha256=carry_receipt["reader_source_sha256"],
                    )
                else:
                    validate_stage_carry_in_reader_receipt(
                        carry_receipt,
                        authenticated_store_snapshot=current,
                        independent_current_tip_anchor=current_tip,
                        carry_in_byte_index=replayed_carry_index,
                        carry_in_complete_marker_sha256=hashlib.sha256(
                            carry_marker_bytes
                        ).hexdigest(),
                        reader_source_sha256=carry_receipt["reader_source_sha256"],
                    )
            except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Stage model carry-in receipt failed exact replay"
                ) from exc
            if stage == "final":
                parent_request_hash = carry_receipt.get("parent_request_sha256")
                parent_bundle = current_tip["authorization_bundles"].get(
                    parent_request_hash
                )
                if type(parent_bundle) is not dict:
                    raise SecFilingGemmaRevealStoreError(
                        "Final model input lost its parent authorization bundle"
                    )
                parent_evidence, _parent_binding = (
                    self._read_owned_stage_evidence_output_locked(
                        authenticated_store_snapshot=current,
                        independent_current_tip_anchor=current_tip,
                        authorization_bundle=parent_bundle,
                        request_sha256=parent_request_hash,
                        require_current_store_state=False,
                    )
                )
                parent_content = parent_evidence.get(
                    "prerequisite_content_manifest"
                )
                if (
                    type(parent_content) is not dict
                    or parent_content.get("artifact_stage") != "intermediate"
                ):
                    raise SecFilingGemmaRevealStoreError(
                        "Final model input parent content manifest changed"
                    )
                content_manifests_by_stage["intermediate"] = copy.deepcopy(
                    parent_content
                )
                content_manifests_by_stage["final"] = copy.deepcopy(
                    content_manifest
                )
            initial_prior_by_form = {}
            records = carry_receipt.get("carry_in_records")
            if type(records) is not list or len(records) != len(replayed_carry_index):
                raise SecFilingGemmaRevealStoreError(
                    "Stage model carry-in records differ from their bytes"
                )
            for record, row in zip(records, replayed_carry_index):
                if type(record) is not dict or record.get("form") not in {"10-K", "10-Q"}:
                    raise SecFilingGemmaRevealStoreError(
                        "Stage model carry-in record identity changed"
                    )
                payload = carry_payloads[row["relative_path"]]
                initial_prior_by_form[record["form"]] = {
                    "text": payload,
                    "source": {
                        "relative_path": row["relative_path"],
                        "byte_count": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    },
                    "provenance": carry_provenance,
                    "accession_number": record["accession_number"],
                    "availability_session": record["availability_session"],
                }
            if set(initial_prior_by_form) != {"10-K", "10-Q"}:
                raise SecFilingGemmaRevealStoreError(
                    "Stage model carry-in lacks one exact prior per filing form"
                )
            sec_reader_hash = sec_reader["receipt_sha256"]
            carry_reader_hash = carry_receipt["receipt_sha256"]
            lifecycle_kind = "stage_request"
            lifecycle_hash = request_hash
        else:
            raise SecFilingGemmaRevealStoreError(
                "Owned model input claim stage changed"
            )

        events: list[dict[str, Any]] = []
        prior_by_form = dict(initial_prior_by_form)
        for event in claim_value.get("event_plan", []):
            if type(event) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Owned model event plan item is not exact"
                )
            document_ordinal = event.get("sec_document_ordinal")
            relative_path = f"document-{document_ordinal:04d}.normalized.txt"
            row = byte_rows.get(relative_path)
            payload = payloads_by_name.get(relative_path)
            if (
                type(row) is not dict
                or type(payload) is not bytes
                or row.get("sha256") != hashlib.sha256(payload).hexdigest()
                or row.get("byte_count") != len(payload)
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned model current filing bytes differ from the event plan"
                )
            prior = prior_by_form.get(event["form"])
            events.append(
                {
                    "event": copy.deepcopy(event),
                    "current_normalized_text": payload,
                    "current_normalized_source": {
                        "relative_path": relative_path,
                        "byte_count": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    },
                    "prior_same_form_normalized_text": (
                        None if prior is None else prior["text"]
                    ),
                    "prior_same_form_normalized_source": (
                        None if prior is None else copy.deepcopy(prior["source"])
                    ),
                    "prior_accession_number": (
                        None if prior is None else prior["accession_number"]
                    ),
                    "prior_availability_session": (
                        None if prior is None else prior["availability_session"]
                    ),
                    "prior_provenance_kind": (
                        None if prior is None else prior["provenance"]
                    ),
                    "sec_reader_receipt_sha256": sec_reader_hash,
                    "carry_in_reader_receipt_sha256": carry_reader_hash,
                }
            )
            prior_by_form[event["form"]] = {
                "text": payload,
                "source": {
                    "relative_path": relative_path,
                    "byte_count": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "provenance": "same_scope_document",
                "accession_number": event["accession_number"],
                "availability_session": event["availability_session"],
            }
        if (
            len(events) != claim_value.get("event_count")
            or [item["event"] for item in events] != claim_value.get("event_plan")
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned model inputs differ from the chronological event plan"
            )
        candidates = [
            entry.get("candidate_manifest")
            for entry in current["latest_registry"]["entries"]
            if entry.get("candidate_sha256") == claim_value["candidate_sha256"]
        ]
        if (
            len(candidates) != 1
            or type(candidates[0]) is not dict
            or candidates[0].get("candidate_sha256")
            != claim_value["candidate_sha256"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned model inputs lack one exact registered candidate"
            )
        return {
            "claim": copy.deepcopy(claim_value),
            "scope_kind": lifecycle_kind,
            "scope_sha256": lifecycle_hash,
            "candidate_manifest": copy.deepcopy(candidates[0]),
            "corpus_universe_manifest": copy.deepcopy(universe),
            "content_manifests_by_stage": content_manifests_by_stage,
            "session_dates": list(EXPECTED_SESSIONS),
            "sec_reader_receipt_sha256": sec_reader_hash,
            "carry_in_reader_receipt_sha256": carry_reader_hash,
            "events": events,
        }

    def _load_owned_development_model_event_inputs(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Load exact development model inputs by their sole lifecycle hash."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            scope_hash = _sha256(
                development_root_scope_sha256,
                "development model input root scope hash",
            )
            claim = current_tip["development_model_execution_claims"].get(
                scope_hash
            )
            return self._load_owned_model_event_inputs_locked(
                authenticated_store_snapshot=current,
                independent_current_tip_anchor=current_tip,
                claim=claim,
            )

    def _load_authorized_model_stage_event_inputs(
        self,
        *,
        request_sha256: str,
    ) -> dict[str, Any]:
        """Load exact stage model inputs by their sole lifecycle hash."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            request_hash = _sha256(
                request_sha256,
                "stage model input request hash",
            )
            claim = current_tip["stage_model_execution_claims"].get(request_hash)
            return self._load_owned_model_event_inputs_locked(
                authenticated_store_snapshot=current,
                independent_current_tip_anchor=current_tip,
                claim=claim,
            )

    def _validate_owned_model_component_semantics_locked(
        self,
        *,
        authenticated_store_snapshot: Mapping[str, Any],
        independent_current_tip_anchor: Mapping[str, Any],
        claim: Mapping[str, Any],
        payloads_by_name: Mapping[str, bytes],
        model_reader_receipt_sha256: str | None = None,
    ) -> list[dict[str, Any]]:
        """Independently replay every persisted model artifact before terminal CAS."""

        claim_value = _expect_mapping(claim, "owned model semantic claim")
        loaded = self._load_owned_model_event_inputs_locked(
            authenticated_store_snapshot=authenticated_store_snapshot,
            independent_current_tip_anchor=independent_current_tip_anchor,
            claim=claim_value,
        )
        if set(loaded) != {
            "claim",
            "scope_kind",
            "scope_sha256",
            "candidate_manifest",
            "corpus_universe_manifest",
            "content_manifests_by_stage",
            "session_dates",
            "sec_reader_receipt_sha256",
            "carry_in_reader_receipt_sha256",
            "events",
        } or loaded["claim"] != claim_value:
            raise SecFilingGemmaRevealStoreError(
                "Owned model semantic input bundle is not exact"
            )
        candidate = loaded["candidate_manifest"]
        universe = loaded["corpus_universe_manifest"]
        content_by_stage = loaded["content_manifests_by_stage"]
        session_dates = loaded["session_dates"]
        events = loaded["events"]
        stage = claim_value["authorized_stage"]
        before_manifest = _strict_json_bytes(
            payloads_by_name["pre_runtime_probe.json"],
            "owned model pre-runtime probe",
        )
        try:
            before_probe = validate_ollama_runtime_probe_receipt(
                before_manifest,
                expected_model_digest=claim_value["model_digest"],
                expected_runtime_fingerprint_sha256=claim_value[
                    "runtime_fingerprint_sha256"
                ],
                expected_transport_mode=OWNED_RUNTIME_PROBE_TRANSPORT_MODE,
                expected_probe_receipt_sha256=before_manifest.get(
                    "receipt_sha256"
                ),
            )
        except (AttributeError, KeyError, SecFilingGemmaOllamaError) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Owned model pre-runtime probe failed exact replay"
            ) from exc
        before_identity = before_probe.pinned_runtime_identity()
        attempt_hashes: list[str] = []
        projection_inputs: list[dict[str, Any]] = []
        elapsed_nanoseconds = 0
        for event_item in events:
            if type(event_item) is not dict or set(event_item) != {
                "event",
                "current_normalized_text",
                "current_normalized_source",
                "prior_same_form_normalized_text",
                "prior_same_form_normalized_source",
                "prior_accession_number",
                "prior_availability_session",
                "prior_provenance_kind",
                "sec_reader_receipt_sha256",
                "carry_in_reader_receipt_sha256",
            }:
                raise SecFilingGemmaRevealStoreError(
                    "Owned model semantic event input is not exact"
                )
            event = event_item["event"]
            ordinal = event["event_ordinal"]
            prefix = f"events/{ordinal:06d}"
            try:
                current_bytes = event_item["current_normalized_text"]
                prior_bytes = event_item["prior_same_form_normalized_text"]
                current_text = current_bytes.decode("utf-8")
                prior_text = (
                    None if prior_bytes is None else prior_bytes.decode("utf-8")
                )
            except (AttributeError, UnicodeDecodeError) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Owned model normalized inputs are not exact UTF-8"
                ) from exc
            if current_text.encode("utf-8") != current_bytes or (
                prior_text is not None and prior_text.encode("utf-8") != prior_bytes
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned model normalized inputs are not canonical UTF-8"
                )
            preprocessed = _strict_json_bytes(
                payloads_by_name[f"{prefix}/preprocessed_event.json"],
                f"owned model preprocessed event {ordinal}",
            )
            preprocessing_receipt = _strict_json_bytes(
                payloads_by_name[f"{prefix}/preprocessing_receipt.json"],
                f"owned model preprocessing receipt {ordinal}",
            )
            redacted_manifest = _strict_json_bytes(
                payloads_by_name[f"{prefix}/redacted_input_manifest.json"],
                f"owned model redacted manifest {ordinal}",
            )
            call_intent = _strict_json_bytes(
                payloads_by_name[f"{prefix}/call_intent.json"],
                f"owned model call intent {ordinal}",
            )
            attempt_manifest = _strict_json_bytes(
                payloads_by_name[f"{prefix}/model_attempt_receipt.json"],
                f"owned model attempt receipt {ordinal}",
            )
            try:
                validate_preprocessed_event(
                    preprocessed,
                    current_normalized_text=current_text,
                    prior_same_form_normalized_text=prior_text,
                    identity_lexicon=CANONICAL_IDENTITY_LEXICON,
                )
                validate_owned_preprocessing_receipt(
                    preprocessing_receipt,
                    scope_kind=loaded["scope_kind"],
                    scope_sha256=loaded["scope_sha256"],
                    candidate_sha256=claim_value["candidate_sha256"],
                    model_execution_claim_sha256=claim_value["claim_sha256"],
                    sec_reader_receipt_sha256=loaded[
                        "sec_reader_receipt_sha256"
                    ],
                    carry_in_reader_receipt_sha256=loaded[
                        "carry_in_reader_receipt_sha256"
                    ],
                    stage=stage,
                    event_ordinal=ordinal,
                    accession_number=event["accession_number"],
                    form=event["form"],
                    current_normalized_source=event_item[
                        "current_normalized_source"
                    ],
                    prior_same_form_normalized_source=event_item[
                        "prior_same_form_normalized_source"
                    ],
                    prior_provenance_kind=event_item["prior_provenance_kind"],
                    preprocessor_source_sha256=claim_value[
                        "execution_source_hashes"
                    ]["preprocessor"],
                    preprocessed_event=preprocessed,
                    current_normalized_text=current_text,
                    prior_same_form_normalized_text=prior_text,
                )
                validate_redacted_input_manifest(
                    redacted_manifest,
                    universe_manifest=universe,
                    stage_content_manifest=content_by_stage[stage],
                    expected_manifest_sha256=redacted_manifest[
                        "redacted_input_manifest_sha256"
                    ],
                    expected_preprocessed_event_sha256=preprocessed[
                        "preprocessed_event_sha256"
                    ],
                    expected_owned_preprocessing_receipt_sha256=(
                        preprocessing_receipt["receipt_sha256"]
                    ),
                    expected_sec_reader_receipt_sha256=loaded[
                        "sec_reader_receipt_sha256"
                    ],
                    expected_carry_in_reader_receipt_sha256=loaded[
                        "carry_in_reader_receipt_sha256"
                    ],
                )
                extractor_request = {
                    "request_version": EXTRACTOR_REQUEST_VERSION,
                    "preprocessor_version": PREPROCESSOR_VERSION,
                    "corpus_universe_sha256": claim_value[
                        "corpus_universe_sha256"
                    ],
                    "identity_lexicon_sha256": claim_value[
                        "identity_lexicon_sha256"
                    ],
                    "redacted_input_manifest_sha256": redacted_manifest[
                        "redacted_input_manifest_sha256"
                    ],
                    "stage": stage,
                    "current_accession_number": event["accession_number"],
                    "current_form": event["form"],
                    "current_availability_session": event[
                        "availability_session"
                    ],
                    "current_filing_sha256": event_item[
                        "current_normalized_source"
                    ]["sha256"],
                    "prior_accession_number": event_item[
                        "prior_accession_number"
                    ],
                    "prior_availability_session": event_item[
                        "prior_availability_session"
                    ],
                    "prior_same_form_filing_sha256": (
                        None
                        if event_item["prior_same_form_normalized_source"] is None
                        else event_item["prior_same_form_normalized_source"][
                            "sha256"
                        ]
                    ),
                    "model_payload": preprocessed["model_payload"],
                    "model_payload_sha256": preprocessed[
                        "model_payload_sha256"
                    ],
                    "redaction_report": preprocessed["redaction_report"],
                }
                validated_request = validate_extractor_request(
                    extractor_request,
                    candidate_manifest=candidate,
                    expected_candidate_sha256=claim_value["candidate_sha256"],
                    universe_manifest=universe,
                    content_manifests_by_stage=content_by_stage,
                    expected_content_manifest_sha256s={
                        item_stage: manifest["content_manifest_sha256"]
                        for item_stage, manifest in content_by_stage.items()
                    },
                    session_dates=session_dates,
                    forbidden_identity_terms=CANONICAL_IDENTITY_LEXICON,
                    redacted_input_manifest=redacted_manifest,
                    expected_redacted_input_manifest_sha256=(
                        redacted_manifest["redacted_input_manifest_sha256"]
                    ),
                    expected_preprocessed_event_sha256=preprocessed[
                        "preprocessed_event_sha256"
                    ],
                    expected_owned_preprocessing_receipt_sha256=(
                        preprocessing_receipt["receipt_sha256"]
                    ),
                    expected_sec_reader_receipt_sha256=loaded[
                        "sec_reader_receipt_sha256"
                    ],
                    expected_carry_in_reader_receipt_sha256=loaded[
                        "carry_in_reader_receipt_sha256"
                    ],
                )
                expected_intent = _build_owned_model_call_intent(
                    claim=claim_value,
                    event=event,
                    preprocessed_event_sha256=preprocessed[
                        "preprocessed_event_sha256"
                    ],
                    owned_preprocessing_receipt_sha256=(
                        preprocessing_receipt["receipt_sha256"]
                    ),
                    redacted_input_manifest_sha256=redacted_manifest[
                        "redacted_input_manifest_sha256"
                    ],
                    model_payload_sha256=preprocessed["model_payload_sha256"],
                    runtime_probe_receipt_sha256=before_manifest[
                        "receipt_sha256"
                    ],
                    runtime_evidence_sha256=before_identity.evidence_sha256,
                )
                if call_intent != expected_intent:
                    raise SecFilingGemmaRevealStoreError(
                        "Owned model call intent is not the exact pre-call plan"
                    )
                attempt = validate_ollama_model_attempt_receipt(
                    attempt_manifest,
                    expected_candidate_sha256=claim_value["candidate_sha256"],
                    expected_model_payload_sha256=preprocessed[
                        "model_payload_sha256"
                    ],
                    expected_sentence_ids=validated_request["sentence_ids"],
                    expected_runtime_evidence_sha256=(
                        before_identity.evidence_sha256
                    ),
                    expected_model_digest=claim_value["model_digest"],
                    expected_runtime_fingerprint_sha256=claim_value[
                        "runtime_fingerprint_sha256"
                    ],
                    expected_transport_mode=OWNED_MODEL_ATTEMPT_TRANSPORT_MODE,
                )
            except (
                KeyError,
                SecFilingGemmaContractError,
                SecFilingGemmaOllamaError,
                SecFilingGemmaPreprocessorError,
            ) as exc:
                raise SecFilingGemmaRevealStoreError(
                    f"Owned model event {ordinal} failed complete semantic replay"
                ) from exc
            attempt_hashes.append(attempt.receipt_sha256)
            projection_inputs.append(
                {
                    "event": copy.deepcopy(event),
                    "preprocessed_event_sha256": preprocessed[
                        "preprocessed_event_sha256"
                    ],
                    "owned_preprocessing_receipt_sha256": preprocessing_receipt[
                        "receipt_sha256"
                    ],
                    "redacted_input_manifest_sha256": redacted_manifest[
                        "redacted_input_manifest_sha256"
                    ],
                    "call_intent_sha256": call_intent["call_intent_sha256"],
                    "model_attempt_receipt_sha256": attempt.receipt_sha256,
                    "attempt": attempt,
                }
            )
            elapsed_nanoseconds += attempt.elapsed_nanoseconds
        maximum_seconds = claim_value["model_runtime_limits"].get(
            "maximum_model_seconds"
        )
        if (
            type(maximum_seconds) not in {int, float}
            or isinstance(maximum_seconds, bool)
            or maximum_seconds <= 0
            or elapsed_nanoseconds > int(float(maximum_seconds) * 1_000_000_000)
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned model attempts exceeded their claim-bound time ceiling"
            )
        after_manifest = _strict_json_bytes(
            payloads_by_name["post_runtime_probe.json"],
            "owned model post-runtime probe",
        )
        runtime_guard = _strict_json_bytes(
            payloads_by_name["runtime_guard.json"],
            "owned model runtime guard",
        )
        try:
            after_probe = validate_ollama_runtime_probe_receipt(
                after_manifest,
                expected_model_digest=claim_value["model_digest"],
                expected_runtime_fingerprint_sha256=claim_value[
                    "runtime_fingerprint_sha256"
                ],
                expected_transport_mode=OWNED_RUNTIME_PROBE_TRANSPORT_MODE,
                expected_probe_receipt_sha256=after_manifest.get(
                    "receipt_sha256"
                ),
            )
            validate_runtime_identity_guard(
                runtime_guard,
                before_evidence=before_probe.runtime_evidence(),
                after_evidence=after_probe.runtime_evidence(),
                expected_model_digest=claim_value["model_digest"],
                expected_runtime_fingerprint_sha256=claim_value[
                    "runtime_fingerprint_sha256"
                ],
                stage=stage,
                model_call_receipt_sha256s=attempt_hashes,
                expected_runtime_guard_sha256=runtime_guard.get(
                    "runtime_guard_sha256"
                ),
            )
        except (AttributeError, KeyError, SecFilingGemmaOllamaError) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Owned model post-runtime guard failed exact replay"
            ) from exc
        self._revalidate_authorized_model_execution_sources(claim_value)
        if model_reader_receipt_sha256 is None:
            return []
        if stage != "development":
            raise SecFilingGemmaRevealStoreError(
                "Only terminal development model evidence can project feature inputs"
            )
        reader_hash = _sha256(
            model_reader_receipt_sha256,
            "owned development feature model reader receipt hash",
        )
        runtime_guard_hash = _sha256(
            runtime_guard.get("runtime_guard_sha256"),
            "owned development feature runtime guard hash",
        )
        projections: list[dict[str, Any]] = []
        try:
            for item in projection_inputs:
                event = item["event"]
                universe_proof = build_validated_universe_event_proof(
                    universe_manifest=universe,
                    expected_corpus_universe_sha256=claim_value[
                        "corpus_universe_sha256"
                    ],
                    current_accession_number=event["accession_number"],
                    content_manifests_by_stage=content_by_stage,
                    expected_content_manifest_sha256s={
                        item_stage: manifest["content_manifest_sha256"]
                        for item_stage, manifest in content_by_stage.items()
                    },
                    session_dates=session_dates,
                )
                attempt = item["attempt"]
                evidence_body = {
                    "identity_domain": (
                        "aapl-sec-gemma-owned-extraction-evidence-v1"
                    ),
                    "model_execution_claim_sha256": claim_value["claim_sha256"],
                    "model_reader_receipt_sha256": reader_hash,
                    "event_ordinal": event["event_ordinal"],
                    "accession_number": event["accession_number"],
                    "form": event["form"],
                    "decision_session": event["availability_session"],
                    "preprocessed_event_sha256": item[
                        "preprocessed_event_sha256"
                    ],
                    "owned_preprocessing_receipt_sha256": item[
                        "owned_preprocessing_receipt_sha256"
                    ],
                    "redacted_input_manifest_sha256": item[
                        "redacted_input_manifest_sha256"
                    ],
                    "call_intent_sha256": item["call_intent_sha256"],
                    "model_attempt_receipt_sha256": item[
                        "model_attempt_receipt_sha256"
                    ],
                    "runtime_guard_sha256": runtime_guard_hash,
                }
                evidence_hash = canonical_sha256(evidence_body)
                if attempt.attempt_status == VALID_ATTEMPT_STATUS:
                    extraction_status = "valid"
                    extractor_output = attempt.validated_extractor_output()
                    extractor_output_bytes = attempt.extractor_output_bytes
                    output_hash = attempt.extractor_output_sha256
                    canonical_output_hash = (
                        attempt.extractor_output_canonical_sha256
                    )
                elif attempt.attempt_status == INVALID_ATTEMPT_STATUS:
                    extraction_status = "invalid"
                    extractor_output = None
                    extractor_output_bytes = None
                    output_hash = None
                    canonical_output_hash = None
                else:
                    raise SecFilingGemmaRevealStoreError(
                        "Owned model attempt status cannot project feature evidence"
                    )
                extraction_proof = build_validated_extraction_event_proof(
                    universe_event_proof=universe_proof,
                    expected_universe_event_proof_sha256=universe_proof[
                        "universe_event_proof_sha256"
                    ],
                    extraction_status=extraction_status,
                    extraction_evidence_sha256=evidence_hash,
                    expected_extraction_evidence_sha256=evidence_hash,
                    extractor_output=extractor_output,
                    extractor_output_bytes=extractor_output_bytes,
                    expected_extraction_output_sha256=output_hash,
                    expected_extraction_output_canonical_sha256=(
                        canonical_output_hash
                    ),
                    supplied_sentence_ids=list(attempt.sentence_ids),
                )
                projections.append(
                    {
                        "event_ordinal": event["event_ordinal"],
                        "event_plan_item": copy.deepcopy(event),
                        "universe_event_proof": universe_proof,
                        "extraction_event_proof": extraction_proof,
                    }
                )
        except (
            KeyError,
            SecFilingGemmaContractError,
            SecFilingGemmaOllamaError,
            TypeError,
            ValueError,
        ) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Owned model feature projection failed exact semantic replay"
            ) from exc
        if (
            len(projections) != claim_value.get("event_count")
            or [item["event_plan_item"] for item in projections]
            != claim_value.get("event_plan")
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned model feature projections differ from the event plan"
            )
        return projections

    def _load_owned_development_feature_inputs(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Project causal development feature inputs from terminal owned evidence."""

        with self._locked():
            return self._load_owned_development_feature_inputs_locked(
                development_root_scope_sha256=development_root_scope_sha256,
            )

    def _load_owned_development_feature_inputs_locked(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Project feature inputs while the caller holds the reveal-store lock."""

        def project_locked() -> dict[str, Any]:
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            scope_hash = _sha256(
                development_root_scope_sha256,
                "owned development feature input root scope hash",
            )
            if current_tip.get("consumed_request_count") != 0:
                raise SecFilingGemmaRevealStoreError(
                    "Development feature inputs require zero consumed reveals"
                )
            active_effect_maps = (
                (
                    "stage SEC",
                    "stage_sec_execution_claims",
                    "stage_sec_reader_receipts",
                    "stage_sec_execution_aborts",
                ),
                (
                    "development SEC",
                    "development_sec_execution_claims",
                    "development_sec_reader_receipts",
                    "development_sec_execution_aborts",
                ),
                (
                    "development market",
                    "development_market_execution_claims",
                    "development_market_reader_receipts",
                    "development_market_execution_aborts",
                ),
                (
                    "stage model",
                    "stage_model_execution_claims",
                    "stage_model_reader_receipts",
                    "stage_model_execution_aborts",
                ),
                (
                    "development model",
                    "development_model_execution_claims",
                    "development_model_reader_receipts",
                    "development_model_execution_aborts",
                ),
            )
            for effect_name, claim_map, reader_map, abort_map in active_effect_maps:
                active = set(current_tip[claim_map]) - set(
                    current_tip[reader_map]
                ) - set(current_tip[abort_map])
                if active:
                    raise SecFilingGemmaRevealStoreError(
                        f"Development feature inputs found an active {effect_name} effect"
                    )

            sec_claim = current_tip["development_sec_execution_claims"].get(
                scope_hash
            )
            sec_reader = current_tip["development_sec_reader_receipts"].get(
                scope_hash
            )
            market_claim = current_tip[
                "development_market_execution_claims"
            ].get(scope_hash)
            market_reader = current_tip[
                "development_market_reader_receipts"
            ].get(scope_hash)
            model_claim = current_tip[
                "development_model_execution_claims"
            ].get(scope_hash)
            model_reader = current_tip[
                "development_model_reader_receipts"
            ].get(scope_hash)
            terminal_items = (
                sec_claim,
                sec_reader,
                market_claim,
                market_reader,
                model_claim,
                model_reader,
            )
            if (
                scope_hash in current_tip["development_sec_execution_aborts"]
                or scope_hash
                in current_tip["development_market_execution_aborts"]
                or scope_hash
                in current_tip["development_model_execution_aborts"]
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development feature inputs cannot use an aborted effect"
                )
            if any(type(item) is not dict for item in terminal_items):
                raise SecFilingGemmaRevealStoreError(
                    "Development feature inputs require terminal SEC, market, and model readers"
                )
            if any(
                item.get("development_root_scope_sha256") != scope_hash
                for item in terminal_items
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development feature inputs crossed their root scope"
                )
            try:
                feature_plan = build_development_feature_assembly_plan(
                    current,
                    development_root_scope_sha256=scope_hash,
                    independent_current_tip_anchor=current_tip,
                )
                validate_development_feature_assembly_plan(
                    feature_plan,
                    expected_feature_assembly_plan_sha256=feature_plan[
                        "feature_assembly_plan_sha256"
                    ],
                )
            except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Development feature assembly plan failed exact store replay"
                ) from exc
            if (
                feature_plan["development_root_scope_sha256"] != scope_hash
                or feature_plan["start_consumed_request_count"] != 0
                or feature_plan["development_sec_execution_claim_sha256"]
                != sec_claim["claim_sha256"]
                or feature_plan["development_sec_reader_receipt_sha256"]
                != sec_reader["receipt_sha256"]
                or feature_plan["development_market_execution_claim_sha256"]
                != market_claim["claim_sha256"]
                or feature_plan["development_market_reader_receipt_sha256"]
                != market_reader["receipt_sha256"]
                or feature_plan["development_model_execution_claim_sha256"]
                != model_claim["claim_sha256"]
                or feature_plan["development_model_reader_receipt_sha256"]
                != model_reader["receipt_sha256"]
                or feature_plan["event_plan"] != model_claim["event_plan"]
                or feature_plan["event_plan_sha256"]
                != model_claim["event_plan_sha256"]
                or feature_plan["event_count"] != model_claim["event_count"]
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development feature assembly plan crossed terminal ancestry"
                )

            sec_replay = self._read_owned_development_root_carry_source_locked(
                authenticated_store_snapshot=current,
                independent_current_tip_anchor=current_tip,
                development_sec_execution_claim=sec_claim,
                development_sec_reader_receipt=sec_reader,
            )
            market_replay = self._replay_owned_development_market_component_locked(
                claim=market_claim,
                development_root_scope_sha256=scope_hash,
                expected_reader_receipt=market_reader,
            )
            source_manifest = market_replay["source_manifest"]
            stage_manifest = market_replay["stage_manifest"]
            if (
                source_manifest.get("source_manifest_sha256")
                != feature_plan["development_market_source_manifest_sha256"]
                or stage_manifest.get("market_stage_manifest_sha256")
                != feature_plan["development_market_stage_manifest_sha256"]
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development feature market evidence crossed its plan"
                )

            model_index, model_payloads, model_marker = (
                self._read_owned_model_component_locked(
                    claim=model_claim,
                    lifecycle_kind="development_root_scope",
                    lifecycle_sha256=scope_hash,
                    marker_schema_version=(
                        DEVELOPMENT_MODEL_EXTRACTION_COMPLETE_MARKER_SCHEMA_VERSION
                    ),
                )
            )
            try:
                expected_model_reader = build_development_model_reader_receipt(
                    model_claim,
                    byte_index=model_index,
                    complete_marker_sha256=hashlib.sha256(model_marker).hexdigest(),
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Development feature model reader failed exact replay"
                ) from exc
            if expected_model_reader != model_reader:
                raise SecFilingGemmaRevealStoreError(
                    "Development feature model bytes differ from their terminal reader"
                )
            model_projections = (
                self._validate_owned_model_component_semantics_locked(
                    authenticated_store_snapshot=current,
                    independent_current_tip_anchor=current_tip,
                    claim=model_claim,
                    payloads_by_name=model_payloads,
                    model_reader_receipt_sha256=model_reader["receipt_sha256"],
                )
            )
            closure_model_index, closure_model_payloads, closure_model_marker = (
                self._read_owned_model_component_locked(
                    claim=model_claim,
                    lifecycle_kind="development_root_scope",
                    lifecycle_sha256=scope_hash,
                    marker_schema_version=(
                        DEVELOPMENT_MODEL_EXTRACTION_COMPLETE_MARKER_SCHEMA_VERSION
                    ),
                )
            )
            if (
                closure_model_index != model_index
                or closure_model_payloads != model_payloads
                or closure_model_marker != model_marker
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development feature model component changed during projection"
                )

            events: list[dict[str, Any]] = []
            if len(model_projections) != feature_plan["event_count"]:
                raise SecFilingGemmaRevealStoreError(
                    "Development feature model projection count changed"
                )
            for ordinal, (event_plan_item, model_projection) in enumerate(
                zip(
                    feature_plan["event_plan"],
                    model_projections,
                    strict=True,
                ),
                start=1,
            ):
                if (
                    event_plan_item.get("event_ordinal") != ordinal
                    or model_projection.get("event_ordinal") != ordinal
                    or model_projection.get("event_plan_item") != event_plan_item
                ):
                    raise SecFilingGemmaRevealStoreError(
                        "Development feature event order changed"
                    )
                try:
                    market_prefix = build_decision_market_prefix(
                        stage_manifest=stage_manifest,
                        source_manifest=source_manifest,
                        expected_artifact_stage="development",
                        expected_source_manifest_sha256=feature_plan[
                            "development_market_source_manifest_sha256"
                        ],
                        expected_market_stage_manifest_sha256=feature_plan[
                            "development_market_stage_manifest_sha256"
                        ],
                        decision_event_id=event_plan_item["accession_number"],
                        decision_session=event_plan_item[
                            "availability_session"
                        ],
                    )
                    market_prefix_proof = build_validated_market_prefix_proof(
                        prefix=market_prefix,
                        stage_manifest=stage_manifest,
                        source_manifest=source_manifest,
                        expected_artifact_stage="development",
                        expected_source_manifest_sha256=feature_plan[
                            "development_market_source_manifest_sha256"
                        ],
                        expected_market_stage_manifest_sha256=feature_plan[
                            "development_market_stage_manifest_sha256"
                        ],
                        expected_decision_event_id=event_plan_item[
                            "accession_number"
                        ],
                        expected_decision_session=event_plan_item[
                            "availability_session"
                        ],
                        expected_market_prefix_sha256=market_prefix[
                            "market_prefix_sha256"
                        ],
                    )
                except (KeyError, SecFilingGemmaContractError) as exc:
                    raise SecFilingGemmaRevealStoreError(
                        f"Development feature event {ordinal} market prefix failed replay"
                    ) from exc
                events.append(
                    {
                        "event_ordinal": ordinal,
                        "event_plan_item": copy.deepcopy(event_plan_item),
                        "market_prefix": market_prefix,
                        "market_prefix_proof": market_prefix_proof,
                        "universe_event_proof": copy.deepcopy(
                            model_projection["universe_event_proof"]
                        ),
                        "extraction_event_proof": copy.deepcopy(
                            model_projection["extraction_event_proof"]
                        ),
                    }
                )

            closure_sec = self._read_owned_development_root_carry_source_locked(
                authenticated_store_snapshot=current,
                independent_current_tip_anchor=current_tip,
                development_sec_execution_claim=sec_claim,
                development_sec_reader_receipt=sec_reader,
            )
            closure_market = (
                self._replay_owned_development_market_component_locked(
                    claim=market_claim,
                    development_root_scope_sha256=scope_hash,
                    expected_reader_receipt=market_reader,
                )
            )
            final_model_index, final_model_payloads, final_model_marker = (
                self._read_owned_model_component_locked(
                    claim=model_claim,
                    lifecycle_kind="development_root_scope",
                    lifecycle_sha256=scope_hash,
                    marker_schema_version=(
                        DEVELOPMENT_MODEL_EXTRACTION_COMPLETE_MARKER_SCHEMA_VERSION
                    ),
                )
            )
            if (
                closure_sec != sec_replay
                or closure_market != market_replay
                or final_model_index != model_index
                or final_model_payloads != model_payloads
                or final_model_marker != model_marker
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development feature ancestry changed during assembly"
                )
            self._revalidate_authorized_model_execution_sources(model_claim)
            body = {
                "schema_version": (
                    OWNED_DEVELOPMENT_FEATURE_INPUTS_SCHEMA_VERSION
                ),
                "feature_assembly_plan": copy.deepcopy(feature_plan),
                "events": events,
            }
            detached = _exact_builtin_json_copy(
                body,
                "owned development feature inputs",
            )
            if type(detached) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Owned development feature inputs are not exact JSON"
                )
            return {
                **detached,
                "feature_inputs_sha256": canonical_sha256(detached),
            }

        return project_locked()

    def _build_owned_development_feature_batch_from_inputs_locked(
        self,
        *,
        feature_inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Rebuild the exact safe feature batch from one locked store projection."""

        if type(feature_inputs) is not dict or set(feature_inputs) != {
            "schema_version",
            "feature_assembly_plan",
            "events",
            "feature_inputs_sha256",
        }:
            raise SecFilingGemmaRevealStoreError(
                "Development label source feature inputs are not exact"
            )
        feature_plan = feature_inputs["feature_assembly_plan"]
        events = feature_inputs["events"]
        if (
            feature_inputs["schema_version"]
            != OWNED_DEVELOPMENT_FEATURE_INPUTS_SCHEMA_VERSION
            or type(feature_plan) is not dict
            or type(events) is not list
            or len(events) != feature_plan.get("event_count")
            or feature_inputs["feature_inputs_sha256"]
            != canonical_sha256(
                {
                    key: feature_inputs[key]
                    for key in feature_inputs
                    if key != "feature_inputs_sha256"
                }
            )
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development label source feature event count changed"
            )
        feature_rows: list[dict[str, Any]] = []
        for ordinal, event in enumerate(events, start=1):
            try:
                if (
                    type(event) is not dict
                    or event.get("event_ordinal") != ordinal
                    or event.get("event_plan_item")
                    != feature_plan["event_plan"][ordinal - 1]
                ):
                    raise SecFilingGemmaRevealStoreError(
                        "Development label source feature event order changed"
                    )
                prefix_proof = event["market_prefix_proof"]
                universe_proof = event["universe_event_proof"]
                extraction_proof = event["extraction_event_proof"]
                feature_rows.append(
                    build_sec_filing_gemma_feature_row(
                        market_prefix=event["market_prefix"],
                        market_prefix_proof=prefix_proof,
                        expected_market_prefix_proof_sha256=prefix_proof[
                            "market_prefix_proof_sha256"
                        ],
                        universe_event_proof=universe_proof,
                        expected_universe_event_proof_sha256=universe_proof[
                            "universe_event_proof_sha256"
                        ],
                        extraction_event_proof=extraction_proof,
                        expected_extraction_event_proof_sha256=extraction_proof[
                            "extraction_event_proof_sha256"
                        ],
                    )
                )
            except SecFilingGemmaRevealStoreError:
                raise
            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                SecFilingGemmaContractError,
                SecFilingGemmaFeatureError,
            ) as exc:
                accession = (
                    event.get("event_plan_item", {}).get("accession_number")
                    if type(event) is dict
                    and type(event.get("event_plan_item")) is dict
                    else "unknown"
                )
                raise SecFilingGemmaRevealStoreError(
                    "Development label source feature event "
                    f"{ordinal} ({accession}) failed causal replay"
                ) from exc
        try:
            batch = build_owned_development_feature_batch(
                feature_assembly_plan=feature_plan,
                feature_rows=feature_rows,
            )
            validate_owned_development_feature_batch(
                batch,
                feature_assembly_plan=feature_plan,
                expected_feature_assembly_plan_sha256=feature_plan[
                    "feature_assembly_plan_sha256"
                ],
                expected_feature_batch_sha256=batch["feature_batch_sha256"],
            )
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            SecFilingGemmaContractError,
            SecFilingGemmaFeatureError,
        ) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Development label source feature batch failed exact replay"
            ) from exc
        return batch

    def _load_owned_development_label_projection(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Derive only development labels matured by the frozen cutoff."""

        with self._locked():
            return self._load_owned_development_label_projection_locked(
                development_root_scope_sha256=development_root_scope_sha256,
            )

    def _load_owned_development_label_projection_locked(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Derive development labels while the caller holds the store lock."""

        def project_locked() -> dict[str, Any]:
            scope_hash = _sha256(
                development_root_scope_sha256,
                "owned development label projection root scope hash",
            )
            feature_inputs = self._load_owned_development_feature_inputs_locked(
                development_root_scope_sha256=scope_hash,
            )
            source_feature_batch = (
                self._build_owned_development_feature_batch_from_inputs_locked(
                    feature_inputs=feature_inputs,
                )
            )
            source_feature_plan = feature_inputs["feature_assembly_plan"]

            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            try:
                label_plan = build_development_label_assembly_plan(
                    current,
                    development_root_scope_sha256=scope_hash,
                    source_feature_assembly_plan=source_feature_plan,
                    independent_current_tip_anchor=current_tip,
                )
                validate_development_label_assembly_plan(
                    label_plan,
                    expected_label_assembly_plan_sha256=label_plan[
                        "label_assembly_plan_sha256"
                    ],
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                SecFilingGemmaStageAuthorizationError,
            ) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Development label assembly plan failed exact store replay"
                ) from exc
            if (
                label_plan.get("development_root_scope_sha256") != scope_hash
                or label_plan.get("start_consumed_request_count") != 0
                or label_plan.get("source_feature_assembly_plan")
                != source_feature_plan
                or label_plan.get("source_feature_assembly_plan_sha256")
                != source_feature_plan["feature_assembly_plan_sha256"]
                or label_plan.get("event_count")
                != source_feature_batch["event_count"]
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development label assembly plan crossed its source feature batch"
                )

            market_claim = current_tip[
                "development_market_execution_claims"
            ].get(scope_hash)
            market_reader = current_tip[
                "development_market_reader_receipts"
            ].get(scope_hash)
            model_claim = current_tip[
                "development_model_execution_claims"
            ].get(scope_hash)
            if (
                type(market_claim) is not dict
                or type(market_reader) is not dict
                or type(model_claim) is not dict
                or scope_hash
                in current_tip["development_market_execution_aborts"]
                or market_claim.get("development_root_scope_sha256") != scope_hash
                or market_reader.get("development_root_scope_sha256") != scope_hash
                or market_reader.get("claim_sha256")
                != market_claim.get("claim_sha256")
                or model_claim.get("development_root_scope_sha256") != scope_hash
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development label projection lacks terminal same-root market/model ancestry"
                )
            market_replay = self._replay_owned_development_market_component_locked(
                claim=market_claim,
                development_root_scope_sha256=scope_hash,
                expected_reader_receipt=market_reader,
            )
            source_manifest = market_replay["source_manifest"]
            stage_manifest = market_replay["stage_manifest"]
            if (
                source_manifest.get("source_manifest_sha256")
                != source_feature_plan["development_market_source_manifest_sha256"]
                or stage_manifest.get("market_stage_manifest_sha256")
                != source_feature_plan["development_market_stage_manifest_sha256"]
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development label market evidence crossed its source feature plan"
                )

            events = feature_inputs["events"]
            feature_rows = source_feature_batch["feature_rows"]
            maturity_plan = label_plan["maturity_plan"]
            if not (
                type(events) is list
                and type(feature_rows) is list
                and type(maturity_plan) is list
                and len(events) == len(feature_rows) == len(maturity_plan)
                == label_plan["event_count"]
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development label event, feature, and maturity counts differ"
                )

            maturity_audit_rows: list[dict[str, Any]] = []
            label_evidence_rows: list[dict[str, Any]] = []
            maturity_item_keys = {
                "event_ordinal",
                "accession_number",
                "form",
                "decision_session",
                "sec_document_ordinal",
                "label_maturity_session",
                "matured_by_development_cutoff",
            }
            for ordinal, (event, feature_row, maturity_item) in enumerate(
                zip(events, feature_rows, maturity_plan, strict=True),
                start=1,
            ):
                event_plan_item = (
                    event.get("event_plan_item") if type(event) is dict else None
                )
                if (
                    type(event_plan_item) is not dict
                    or type(feature_row) is not dict
                    or type(maturity_item) is not dict
                    or set(maturity_item) != maturity_item_keys
                    or maturity_item.get("event_ordinal") != ordinal
                    or event.get("event_ordinal") != ordinal
                    or event_plan_item.get("event_ordinal") != ordinal
                    or maturity_item.get("accession_number")
                    != event_plan_item.get("accession_number")
                    or maturity_item.get("form") != event_plan_item.get("form")
                    or maturity_item.get("decision_session")
                    != event_plan_item.get("availability_session")
                    or maturity_item.get("sec_document_ordinal")
                    != event_plan_item.get("sec_document_ordinal")
                    or feature_row.get("accession_number")
                    != maturity_item.get("accession_number")
                    or feature_row.get("decision_session")
                    != maturity_item.get("decision_session")
                    or type(maturity_item.get("matured_by_development_cutoff"))
                    is not bool
                ):
                    raise SecFilingGemmaRevealStoreError(
                        f"Development label event {ordinal} crossed its maturity plan"
                    )

                evidence_hash: str | None = None
                if maturity_item["matured_by_development_cutoff"]:
                    prefix_proof = event["market_prefix_proof"]
                    universe_proof = event["universe_event_proof"]
                    extraction_proof = event["extraction_event_proof"]
                    label_kwargs = {
                        "market_prefix": event["market_prefix"],
                        "market_prefix_proof": prefix_proof,
                        "expected_market_prefix_proof_sha256": prefix_proof[
                            "market_prefix_proof_sha256"
                        ],
                        "stage_manifest": stage_manifest,
                        "source_manifest": source_manifest,
                        "expected_artifact_stage": "development",
                        "expected_source_manifest_sha256": source_feature_plan[
                            "development_market_source_manifest_sha256"
                        ],
                        "expected_market_stage_manifest_sha256": source_feature_plan[
                            "development_market_stage_manifest_sha256"
                        ],
                        "feature_row": feature_row,
                        "expected_feature_row_sha256": feature_row[
                            "feature_row_sha256"
                        ],
                        "universe_event_proof": universe_proof,
                        "expected_universe_event_proof_sha256": universe_proof[
                            "universe_event_proof_sha256"
                        ],
                        "extraction_event_proof": extraction_proof,
                        "expected_extraction_event_proof_sha256": extraction_proof[
                            "extraction_event_proof_sha256"
                        ],
                    }
                    context = (
                        f"event {ordinal} ({maturity_item['accession_number']}) "
                        f"decision={maturity_item['decision_session']} "
                        f"maturity={maturity_item['label_maturity_session']}"
                    )
                    try:
                        evidence = build_twenty_session_label_evidence(
                            **label_kwargs,
                        )
                        validate_twenty_session_label_evidence(
                            evidence,
                            expected_label_evidence_sha256=evidence[
                                "label_evidence_sha256"
                            ],
                            **label_kwargs,
                        )
                    except (
                        KeyError,
                        IndexError,
                        TypeError,
                        ValueError,
                        SecFilingGemmaContractError,
                        SecFilingGemmaFeatureError,
                    ) as exc:
                        raise SecFilingGemmaRevealStoreError(
                            "Development label target path failed exact replay for "
                            f"{context}: {exc}"
                        ) from exc
                    if (
                        evidence.get("accession_number")
                        != maturity_item["accession_number"]
                        or evidence.get("decision_session")
                        != maturity_item["decision_session"]
                        or evidence.get("label_maturity_session")
                        != maturity_item["label_maturity_session"]
                        or evidence.get("feature_row_sha256")
                        != feature_row["feature_row_sha256"]
                    ):
                        raise SecFilingGemmaRevealStoreError(
                            "Development label evidence crossed its maturity item for "
                            f"{context}"
                        )
                    evidence_hash = _sha256(
                        evidence.get("label_evidence_sha256"),
                        f"development label evidence {ordinal} hash",
                    )
                    label_evidence_rows.append(copy.deepcopy(evidence))

                maturity_audit_rows.append(
                    {
                        "event_ordinal": ordinal,
                        "accession_number": maturity_item["accession_number"],
                        "decision_session": maturity_item["decision_session"],
                        "feature_row_sha256": feature_row["feature_row_sha256"],
                        "label_maturity_session": maturity_item[
                            "label_maturity_session"
                        ],
                        "matured_by_development_cutoff": maturity_item[
                            "matured_by_development_cutoff"
                        ],
                        "label_evidence_sha256": evidence_hash,
                    }
                )

            if (
                len(label_evidence_rows) != label_plan["matured_event_count"]
                or len(maturity_audit_rows) != label_plan["event_count"]
                or sum(
                    row["label_evidence_sha256"] is None
                    for row in maturity_audit_rows
                )
                != label_plan["unmatured_event_count"]
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development label maturity audit counts changed"
                )

            closure_feature_inputs = (
                self._load_owned_development_feature_inputs_locked(
                    development_root_scope_sha256=scope_hash,
                )
            )
            closure_feature_batch = (
                self._build_owned_development_feature_batch_from_inputs_locked(
                    feature_inputs=closure_feature_inputs,
                )
            )
            closure_market_replay = (
                self._replay_owned_development_market_component_locked(
                    claim=market_claim,
                    development_root_scope_sha256=scope_hash,
                    expected_reader_receipt=market_reader,
                )
            )
            if (
                closure_feature_inputs != feature_inputs
                or closure_feature_batch != source_feature_batch
                or closure_market_replay != market_replay
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development label source ancestry changed during projection"
                )
            self._revalidate_authorized_market_execution_sources(market_claim)
            self._revalidate_authorized_model_execution_sources(model_claim)

            body = {
                "schema_version": (
                    OWNED_DEVELOPMENT_LABEL_PROJECTION_SCHEMA_VERSION
                ),
                "label_assembly_plan": copy.deepcopy(label_plan),
                "source_feature_batch": copy.deepcopy(source_feature_batch),
                "maturity_audit_rows": maturity_audit_rows,
                "label_evidence_rows": label_evidence_rows,
            }
            detached = _exact_builtin_json_copy(
                body,
                "owned development label projection",
            )
            if type(detached) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Owned development label projection is not exact JSON"
                )
            return {
                **detached,
                "label_projection_sha256": canonical_sha256(detached),
            }

        return project_locked()

    def _build_owned_development_label_batch_from_projection_locked(
        self,
        *,
        label_projection: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Rebuild the exact compact label batch from one locked projection."""

        if type(label_projection) is not dict or set(label_projection) != {
            "schema_version",
            "label_assembly_plan",
            "source_feature_batch",
            "maturity_audit_rows",
            "label_evidence_rows",
            "label_projection_sha256",
        }:
            raise SecFilingGemmaRevealStoreError(
                "Development membership source label projection is not exact"
            )
        projection = _exact_builtin_json_copy(
            label_projection,
            "development membership source label projection",
        )
        if type(projection) is not dict:  # pragma: no cover - guaranteed above
            raise SecFilingGemmaRevealStoreError(
                "Development membership source label projection is not an object"
            )
        projection_hash = projection["label_projection_sha256"]
        projection_body = {
            key: projection[key]
            for key in projection
            if key != "label_projection_sha256"
        }
        if (
            projection["schema_version"]
            != OWNED_DEVELOPMENT_LABEL_PROJECTION_SCHEMA_VERSION
            or not _same_digest(
                _sha256(
                    projection_hash,
                    "development membership source label projection hash",
                ),
                canonical_sha256(projection_body),
            )
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development membership source label projection checksum changed"
            )
        label_plan = projection["label_assembly_plan"]
        source_feature_batch = projection["source_feature_batch"]
        try:
            if type(label_plan) is not dict or type(source_feature_batch) is not dict:
                raise SecFilingGemmaFeatureError(
                    "Development membership source plans and batches must be objects"
                )
            validate_development_label_assembly_plan(
                label_plan,
                expected_label_assembly_plan_sha256=label_plan[
                    "label_assembly_plan_sha256"
                ],
            )
            batch = build_owned_development_label_batch(
                label_assembly_plan=label_plan,
                source_feature_batch=source_feature_batch,
                maturity_audit_rows=projection["maturity_audit_rows"],
                label_evidence_rows=projection["label_evidence_rows"],
            )
            validate_owned_development_label_batch(
                batch,
                label_assembly_plan=label_plan,
                expected_label_assembly_plan_sha256=label_plan[
                    "label_assembly_plan_sha256"
                ],
                source_feature_batch=source_feature_batch,
                expected_source_feature_batch_sha256=source_feature_batch[
                    "feature_batch_sha256"
                ],
                expected_label_batch_sha256=batch["label_batch_sha256"],
            )
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            SecFilingGemmaContractError,
            SecFilingGemmaFeatureError,
            SecFilingGemmaStageAuthorizationError,
        ) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Development membership source label batch failed exact replay"
            ) from exc
        return batch

    def _load_owned_development_training_membership_projection(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Project the exact safe sources for development training membership."""

        with self._locked():
            return (
                self._load_owned_development_training_membership_projection_locked(
                    development_root_scope_sha256=development_root_scope_sha256,
                )
            )

    def _load_owned_development_training_membership_projection_locked(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Project membership sources while the caller holds the store lock."""

        scope_hash = _sha256(
            development_root_scope_sha256,
            "owned development training membership projection root scope hash",
        )
        tracked_anchor = _load_tracked_anchor(self.repository_root)
        current, current_tip, state_bytes, tip_bytes = (
            self._read_state_and_tip_locked(tracked_anchor)
        )
        raw_label_projection = (
            self._load_owned_development_label_projection_locked(
                development_root_scope_sha256=scope_hash,
            )
        )
        label_projection = _exact_builtin_json_copy(
            raw_label_projection,
            "owned development training membership source label projection",
        )
        if type(label_projection) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "Development training membership source label projection is not exact"
            )
        source_label_batch = (
            self._build_owned_development_label_batch_from_projection_locked(
                label_projection=label_projection,
            )
        )
        label_plan = label_projection["label_assembly_plan"]
        source_feature_batch = label_projection["source_feature_batch"]

        try:
            training_membership_plan = (
                build_development_training_membership_assembly_plan(
                    current,
                    development_root_scope_sha256=scope_hash,
                    source_label_assembly_plan=label_plan,
                    independent_current_tip_anchor=current_tip,
                )
            )
            validate_development_training_membership_assembly_plan(
                training_membership_plan,
                expected_training_membership_assembly_plan_sha256=(
                    training_membership_plan[
                        "training_membership_assembly_plan_sha256"
                    ]
                ),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            SecFilingGemmaStageAuthorizationError,
        ) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Development training membership assembly plan failed exact store replay"
            ) from exc

        source_feature_plan = label_plan.get("source_feature_assembly_plan")
        lineage_matches = (
            type(training_membership_plan) is dict
            and type(label_plan) is dict
            and type(source_feature_plan) is dict
            and type(source_feature_batch) is dict
            and type(source_label_batch) is dict
            and training_membership_plan.get("development_root_scope_sha256")
            == scope_hash
            and training_membership_plan.get("start_consumed_request_count") == 0
            and training_membership_plan.get("source_label_assembly_plan")
            == label_plan
            and training_membership_plan.get(
                "source_label_assembly_plan_sha256"
            )
            == label_plan.get("label_assembly_plan_sha256")
            and training_membership_plan.get(
                "source_feature_assembly_plan_sha256"
            )
            == source_feature_plan.get("feature_assembly_plan_sha256")
            and training_membership_plan.get("candidate_sha256")
            == source_feature_plan.get("candidate_sha256")
            == source_feature_batch.get("candidate_sha256")
            == source_label_batch.get("candidate_sha256")
            and training_membership_plan.get("corpus_universe_sha256")
            == source_feature_plan.get("corpus_universe_sha256")
            == source_feature_batch.get("corpus_universe_sha256")
            == source_label_batch.get("corpus_universe_sha256")
            and source_feature_batch.get("development_root_scope_sha256")
            == scope_hash
            and source_label_batch.get("development_root_scope_sha256")
            == scope_hash
            and source_feature_batch.get("feature_assembly_plan_sha256")
            == source_feature_plan.get("feature_assembly_plan_sha256")
            == label_plan.get("source_feature_assembly_plan_sha256")
            == source_label_batch.get("source_feature_assembly_plan_sha256")
            and source_label_batch.get("source_feature_batch_sha256")
            == source_feature_batch.get("feature_batch_sha256")
            and source_label_batch.get("label_assembly_plan_sha256")
            == label_plan.get("label_assembly_plan_sha256")
            and training_membership_plan.get("development_cutoff_session")
            == label_plan.get("development_cutoff_session")
            == source_label_batch.get("development_cutoff_session")
            and training_membership_plan.get("event_count")
            == label_plan.get("event_count")
            == source_feature_batch.get("event_count")
            == source_label_batch.get("event_count")
            and training_membership_plan.get("matured_event_count")
            == label_plan.get("matured_event_count")
            == source_label_batch.get("matured_label_count")
            and training_membership_plan.get("unmatured_event_count")
            == label_plan.get("unmatured_event_count")
            == source_label_batch.get("unmatured_event_count")
        )
        if not lineage_matches:
            raise SecFilingGemmaRevealStoreError(
                "Development training membership crossed its feature or label ancestry"
            )

        body = {
            "schema_version": (
                OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_PROJECTION_SCHEMA_VERSION
            ),
            "training_membership_assembly_plan": copy.deepcopy(
                training_membership_plan
            ),
            "source_feature_batch": copy.deepcopy(source_feature_batch),
            "source_label_batch": copy.deepcopy(source_label_batch),
        }
        detached = _exact_builtin_json_copy(
            body,
            "owned development training membership projection",
        )
        if type(detached) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "Owned development training membership projection is not exact JSON"
            )
        (
            closure_current,
            closure_tip,
            closure_state_bytes,
            closure_tip_bytes,
        ) = self._read_state_and_tip_locked(tracked_anchor)
        if (
            closure_state_bytes != state_bytes
            or closure_tip_bytes != tip_bytes
            or closure_current != current
            or closure_tip != current_tip
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development training membership authorization ancestry changed during projection"
            )
        return {
            **detached,
            "membership_projection_sha256": canonical_sha256(detached),
        }

    def _build_owned_development_training_membership_batch_from_projection_locked(
        self,
        *,
        training_membership_projection: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Rebuild the exact compact membership batch from one locked projection."""

        expected_keys = {
            "schema_version",
            "training_membership_assembly_plan",
            "source_feature_batch",
            "source_label_batch",
            "membership_projection_sha256",
        }
        if (
            type(training_membership_projection) is not dict
            or set(training_membership_projection) != expected_keys
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development learner-fit source membership projection is not exact"
            )
        projection = _exact_builtin_json_copy(
            training_membership_projection,
            "development learner-fit source membership projection",
        )
        if type(projection) is not dict:  # pragma: no cover - guaranteed above
            raise SecFilingGemmaRevealStoreError(
                "Development learner-fit source membership projection is not an object"
            )
        projection_hash = projection["membership_projection_sha256"]
        projection_body = {
            key: projection[key]
            for key in projection
            if key != "membership_projection_sha256"
        }
        if (
            projection["schema_version"]
            != OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_PROJECTION_SCHEMA_VERSION
            or not _same_digest(
                _sha256(
                    projection_hash,
                    "development learner-fit source membership projection hash",
                ),
                canonical_sha256(projection_body),
            )
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development learner-fit source membership projection checksum changed"
            )

        membership_plan = projection["training_membership_assembly_plan"]
        source_feature_batch = projection["source_feature_batch"]
        source_label_batch = projection["source_label_batch"]
        try:
            if (
                type(membership_plan) is not dict
                or type(source_feature_batch) is not dict
                or type(source_label_batch) is not dict
            ):
                raise SecFilingGemmaTrainingMembershipError(
                    "Development learner-fit membership sources must be objects"
                )
            validate_development_training_membership_assembly_plan(
                membership_plan,
                expected_training_membership_assembly_plan_sha256=(
                    membership_plan["training_membership_assembly_plan_sha256"]
                ),
            )
            batch = build_owned_development_training_membership_batch(
                training_membership_assembly_plan=membership_plan,
                source_feature_batch=source_feature_batch,
                source_label_batch=source_label_batch,
            )
            validate_owned_development_training_membership_batch(
                batch,
                training_membership_assembly_plan=membership_plan,
                expected_training_membership_assembly_plan_sha256=(
                    membership_plan["training_membership_assembly_plan_sha256"]
                ),
                source_feature_batch=source_feature_batch,
                expected_source_feature_batch_sha256=source_feature_batch[
                    "feature_batch_sha256"
                ],
                source_label_batch=source_label_batch,
                expected_source_label_batch_sha256=source_label_batch[
                    "label_batch_sha256"
                ],
                expected_training_membership_batch_sha256=batch[
                    "training_membership_batch_sha256"
                ],
            )
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            SecFilingGemmaContractError,
            SecFilingGemmaStageAuthorizationError,
            SecFilingGemmaTrainingMembershipError,
        ) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Development learner-fit source membership batch failed exact replay"
            ) from exc
        return batch

    def _load_owned_development_oof_learner_fit_projection(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Project only the owned inputs authorized for ten OOF learner fits."""

        with self._locked():
            return self._load_owned_development_oof_learner_fit_projection_locked(
                development_root_scope_sha256=development_root_scope_sha256,
            )

    def _load_owned_development_oof_learner_fit_projection_locked(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Project OOF fit inputs while the caller holds the store lock."""

        scope_hash = _sha256(
            development_root_scope_sha256,
            "owned development OOF learner-fit projection root scope hash",
        )
        tracked_anchor = _load_tracked_anchor(self.repository_root)
        current, current_tip, state_bytes, tip_bytes = (
            self._read_state_and_tip_locked(tracked_anchor)
        )
        raw_membership_projection = (
            self._load_owned_development_training_membership_projection_locked(
                development_root_scope_sha256=scope_hash,
            )
        )
        membership_projection = _exact_builtin_json_copy(
            raw_membership_projection,
            "owned development OOF learner-fit source membership projection",
        )
        if type(membership_projection) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "Development OOF learner-fit source membership projection is not exact"
            )
        membership_batch = (
            self._build_owned_development_training_membership_batch_from_projection_locked(
                training_membership_projection=membership_projection,
            )
        )
        membership_plan = membership_projection[
            "training_membership_assembly_plan"
        ]
        try:
            fit_input_specs = derive_development_oof_learner_fit_input_specs(
                membership_batch
            )
            learner_fit_plan = build_development_oof_learner_fit_plan(
                current,
                development_root_scope_sha256=scope_hash,
                source_training_membership_assembly_plan=membership_plan,
                source_training_membership_projection_sha256=(
                    membership_projection["membership_projection_sha256"]
                ),
                source_training_membership_batch_sha256=membership_batch[
                    "training_membership_batch_sha256"
                ],
                fit_input_specs=fit_input_specs,
                independent_current_tip_anchor=current_tip,
            )
            validate_development_oof_learner_fit_plan(
                learner_fit_plan,
                expected_development_oof_learner_fit_plan_sha256=(
                    learner_fit_plan[
                        "development_oof_learner_fit_plan_sha256"
                    ]
                ),
            )
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            SecFilingGemmaContractError,
            SecFilingGemmaLearnerFitError,
            SecFilingGemmaStageAuthorizationError,
        ) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Development OOF learner-fit plan failed exact store replay"
            ) from exc

        source_view_ids = membership_batch.get("training_view_ids")
        variant_specs = membership_plan.get("model_variant_specs")
        variant_ids = (
            [item.get("variant_id") for item in variant_specs]
            if type(variant_specs) is list
            and all(type(item) is dict for item in variant_specs)
            else None
        )
        deferred = learner_fit_plan.get("deferred_training_views")
        lineage_matches = (
            type(learner_fit_plan) is dict
            and type(membership_plan) is dict
            and type(membership_batch) is dict
            and type(fit_input_specs) is list
            and type(source_view_ids) is list
            and len(source_view_ids) == 6
            and learner_fit_plan.get("development_root_scope_sha256")
            == scope_hash
            == membership_plan.get("development_root_scope_sha256")
            == membership_batch.get("development_root_scope_sha256")
            and learner_fit_plan.get("start_consumed_request_count") == 0
            == membership_plan.get("start_consumed_request_count")
            and learner_fit_plan.get(
                "source_training_membership_assembly_plan"
            )
            == membership_plan
            and learner_fit_plan.get(
                "source_training_membership_assembly_plan_sha256"
            )
            == membership_plan.get("training_membership_assembly_plan_sha256")
            == membership_batch.get(
                "training_membership_assembly_plan_sha256"
            )
            and learner_fit_plan.get(
                "source_training_membership_projection_sha256"
            )
            == membership_projection.get("membership_projection_sha256")
            and learner_fit_plan.get(
                "source_training_membership_batch_sha256"
            )
            == membership_batch.get("training_membership_batch_sha256")
            and learner_fit_plan.get("candidate_sha256")
            == membership_plan.get("candidate_sha256")
            == membership_batch.get("candidate_sha256")
            and learner_fit_plan.get("corpus_universe_sha256")
            == membership_plan.get("corpus_universe_sha256")
            == membership_batch.get("corpus_universe_sha256")
            and learner_fit_plan.get("calendar_sessions_sha256")
            == membership_plan.get("calendar_sessions_sha256")
            == membership_batch.get("calendar_sessions_sha256")
            and learner_fit_plan.get("development_cutoff_session")
            == membership_plan.get("development_cutoff_session")
            == membership_batch.get("development_cutoff_session")
            and membership_plan.get("event_count")
            == membership_batch.get("event_count")
            and membership_plan.get("matured_event_count")
            == membership_batch.get("matured_label_count")
            and membership_plan.get("unmatured_event_count")
            == membership_batch.get("unmatured_event_count")
            and learner_fit_plan.get("source_training_view_count")
            == membership_plan.get("membership_view_count")
            == membership_batch.get("training_view_count")
            == len(source_view_ids)
            and learner_fit_plan.get("source_training_view_ids")
            == source_view_ids
            and learner_fit_plan.get("source_training_view_specs_sha256")
            == membership_plan.get("membership_view_specs_sha256")
            == membership_batch.get("membership_view_specs_sha256")
            and learner_fit_plan.get("authorized_training_view_count") == 5
            and learner_fit_plan.get("authorized_training_view_ids")
            == source_view_ids[:5]
            and learner_fit_plan.get("deferred_training_view_count") == 1
            and type(deferred) is list
            and len(deferred) == 1
            and type(deferred[0]) is dict
            and deferred[0].get("source_training_view_ordinal") == 6
            and deferred[0].get("training_view_id") == source_view_ids[5]
            and deferred[0].get("reason")
            == (
                "requires_passed_development_ranking_receipt_and_frozen_"
                "candidate_selection"
            )
            and learner_fit_plan.get("deferred_training_views_sha256")
            == canonical_sha256(deferred)
            and type(variant_ids) is list
            and learner_fit_plan.get("model_variant_count") == 2
            == membership_plan.get("model_variant_count")
            == len(variant_ids)
            and learner_fit_plan.get("model_variant_ids") == variant_ids
            and membership_plan.get("model_variant_specs_sha256")
            == membership_batch.get("model_variant_specs_sha256")
            and learner_fit_plan.get("learner_fit_input_count")
            == learner_fit_plan.get("learner_state_output_count")
            == len(fit_input_specs)
            == 10
            and learner_fit_plan.get("learner_fit_input_specs")
            == fit_input_specs
            and learner_fit_plan.get("learner_fit_input_specs_sha256")
            == canonical_sha256(fit_input_specs)
        )
        if not lineage_matches:
            raise SecFilingGemmaRevealStoreError(
                "Development OOF learner-fit crossed its membership ancestry"
            )

        body = {
            "schema_version": (
                OWNED_DEVELOPMENT_OOF_LEARNER_FIT_PROJECTION_SCHEMA_VERSION
            ),
            "development_oof_learner_fit_plan": copy.deepcopy(
                learner_fit_plan
            ),
            "source_training_membership_batch": copy.deepcopy(
                membership_batch
            ),
        }
        detached = _exact_builtin_json_copy(
            body,
            "owned development OOF learner-fit projection",
        )
        if type(detached) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "Owned development OOF learner-fit projection is not exact JSON"
            )
        (
            closure_current,
            closure_tip,
            closure_state_bytes,
            closure_tip_bytes,
        ) = self._read_state_and_tip_locked(tracked_anchor)
        if (
            closure_state_bytes != state_bytes
            or closure_tip_bytes != tip_bytes
            or closure_current != current
            or closure_tip != current_tip
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development OOF learner-fit authorization ancestry changed during projection"
            )
        return {
            **detached,
            "learner_fit_projection_sha256": canonical_sha256(detached),
        }

    def _load_owned_development_oof_prediction_projection(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Project only the frozen OOF states and their causal feature rows."""

        with self._locked():
            return self._load_owned_development_oof_prediction_projection_locked(
                development_root_scope_sha256=development_root_scope_sha256,
            )

    def _load_owned_development_oof_prediction_projection_locked(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Project OOF prediction inputs while the caller holds the store lock."""

        scope_hash = _sha256(
            development_root_scope_sha256,
            "owned development OOF prediction projection root scope hash",
        )
        tracked_anchor = _load_tracked_anchor(self.repository_root)
        current, current_tip, state_bytes, tip_bytes = (
            self._read_state_and_tip_locked(tracked_anchor)
        )

        raw_membership_projection = (
            self._load_owned_development_training_membership_projection_locked(
                development_root_scope_sha256=scope_hash,
            )
        )
        membership_projection = _exact_builtin_json_copy(
            raw_membership_projection,
            "owned development OOF prediction source membership projection",
        )
        if type(membership_projection) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "Development OOF prediction source membership projection is not exact"
            )
        membership_batch = (
            self._build_owned_development_training_membership_batch_from_projection_locked(
                training_membership_projection=membership_projection,
            )
        )
        membership_plan = membership_projection.get(
            "training_membership_assembly_plan"
        )
        source_feature_batch = membership_projection.get("source_feature_batch")

        try:
            if (
                type(membership_plan) is not dict
                or type(source_feature_batch) is not dict
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development OOF prediction membership sources are not exact objects"
                )
            fit_input_specs = derive_development_oof_learner_fit_input_specs(
                membership_batch
            )
            learner_fit_plan = build_development_oof_learner_fit_plan(
                current,
                development_root_scope_sha256=scope_hash,
                source_training_membership_assembly_plan=membership_plan,
                source_training_membership_projection_sha256=(
                    membership_projection["membership_projection_sha256"]
                ),
                source_training_membership_batch_sha256=membership_batch[
                    "training_membership_batch_sha256"
                ],
                fit_input_specs=fit_input_specs,
                independent_current_tip_anchor=current_tip,
            )
            validate_development_oof_learner_fit_plan(
                learner_fit_plan,
                expected_development_oof_learner_fit_plan_sha256=learner_fit_plan[
                    "development_oof_learner_fit_plan_sha256"
                ],
            )
            learner_fit_projection_body = {
                "schema_version": (
                    OWNED_DEVELOPMENT_OOF_LEARNER_FIT_PROJECTION_SCHEMA_VERSION
                ),
                "development_oof_learner_fit_plan": copy.deepcopy(
                    learner_fit_plan
                ),
                "source_training_membership_batch": copy.deepcopy(
                    membership_batch
                ),
            }
            learner_fit_projection_sha256 = canonical_sha256(
                learner_fit_projection_body
            )
            learner_fit_batch = build_owned_development_oof_learner_fit_batch(
                development_oof_learner_fit_plan=learner_fit_plan,
                expected_development_oof_learner_fit_plan_sha256=learner_fit_plan[
                    "development_oof_learner_fit_plan_sha256"
                ],
                source_training_membership_batch=membership_batch,
                expected_source_training_membership_batch_sha256=membership_batch[
                    "training_membership_batch_sha256"
                ],
            )
            validate_owned_development_oof_learner_fit_batch(
                learner_fit_batch,
                development_oof_learner_fit_plan=learner_fit_plan,
                expected_development_oof_learner_fit_plan_sha256=learner_fit_plan[
                    "development_oof_learner_fit_plan_sha256"
                ],
                source_training_membership_batch=membership_batch,
                expected_source_training_membership_batch_sha256=membership_batch[
                    "training_membership_batch_sha256"
                ],
                expected_learner_fit_batch_sha256=learner_fit_batch[
                    "learner_fit_batch_sha256"
                ],
            )

            prediction_fold_model_bundle = (
                derive_development_oof_prediction_fold_model_bundle(
                    learner_fit_batch
                )
            )
            prediction_feature_batch = (
                derive_development_oof_prediction_feature_batch(
                    source_feature_batch
                )
            )
            prediction_fold_model_specs = (
                derive_development_oof_prediction_fold_model_specs(
                    prediction_fold_model_bundle
                )
            )
            prediction_input_specs = (
                derive_development_oof_prediction_input_specs(
                    prediction_feature_batch
                )
            )
            prediction_plan = build_development_oof_prediction_plan(
                current,
                development_root_scope_sha256=scope_hash,
                source_development_oof_learner_fit_plan=learner_fit_plan,
                source_development_oof_learner_fit_projection_sha256=(
                    learner_fit_projection_sha256
                ),
                source_development_oof_learner_fit_batch_sha256=(
                    learner_fit_batch["learner_fit_batch_sha256"]
                ),
                prediction_fold_model_bundle_sha256=(
                    prediction_fold_model_bundle[
                        "prediction_fold_model_bundle_sha256"
                    ]
                ),
                prediction_fold_model_specs=prediction_fold_model_specs,
                source_feature_batch_sha256=source_feature_batch[
                    "feature_batch_sha256"
                ],
                prediction_feature_batch_sha256=prediction_feature_batch[
                    "prediction_feature_batch_sha256"
                ],
                prediction_input_specs=prediction_input_specs,
                independent_current_tip_anchor=current_tip,
            )
            validate_development_oof_prediction_plan(
                prediction_plan,
                expected_development_oof_prediction_plan_sha256=prediction_plan[
                    "development_oof_prediction_plan_sha256"
                ],
            )
        except SecFilingGemmaRevealStoreError:
            raise
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            SecFilingGemmaContractError,
            SecFilingGemmaLearnerFitError,
            SecFilingGemmaLearnerPredictionError,
            SecFilingGemmaStageAuthorizationError,
        ) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Development OOF prediction sources failed exact replay"
            ) from exc

        fit_plan_hash = learner_fit_plan.get(
            "development_oof_learner_fit_plan_sha256"
        )
        fit_batch_hash = learner_fit_batch.get("learner_fit_batch_sha256")
        feature_batch_hash = source_feature_batch.get("feature_batch_sha256")
        fold_bundle_hash = prediction_fold_model_bundle.get(
            "prediction_fold_model_bundle_sha256"
        )
        prediction_feature_hash = prediction_feature_batch.get(
            "prediction_feature_batch_sha256"
        )
        lineage_matches = (
            prediction_plan.get("development_root_scope_sha256") == scope_hash
            == learner_fit_plan.get("development_root_scope_sha256")
            == learner_fit_batch.get("development_root_scope_sha256")
            == source_feature_batch.get("development_root_scope_sha256")
            and prediction_plan.get("start_consumed_request_count") == 0
            == learner_fit_plan.get("start_consumed_request_count")
            and prediction_plan.get(
                "source_development_oof_learner_fit_plan_sha256"
            )
            == fit_plan_hash
            and prediction_plan.get(
                "source_development_oof_learner_fit_projection_sha256"
            )
            == learner_fit_projection_sha256
            and prediction_plan.get(
                "source_development_oof_learner_fit_batch_sha256"
            )
            == fit_batch_hash
            == prediction_fold_model_bundle.get(
                "source_learner_fit_batch_sha256"
            )
            and prediction_plan.get("prediction_fold_model_bundle_sha256")
            == fold_bundle_hash
            and prediction_plan.get("source_feature_batch_sha256")
            == feature_batch_hash
            == prediction_feature_batch.get("source_feature_batch_sha256")
            and prediction_plan.get("prediction_feature_batch_sha256")
            == prediction_feature_hash
            and prediction_plan.get("prediction_fold_model_specs")
            == prediction_fold_model_specs
            and prediction_plan.get("prediction_fold_model_specs_sha256")
            == canonical_sha256(prediction_fold_model_specs)
            and prediction_plan.get("prediction_input_specs")
            == prediction_input_specs
            and prediction_plan.get("prediction_input_specs_sha256")
            == canonical_sha256(prediction_input_specs)
        )
        for field in (
            "contract_sha256",
            "candidate_sha256",
            "corpus_universe_sha256",
        ):
            lineage_matches = lineage_matches and (
                prediction_plan.get(field)
                == learner_fit_plan.get(field)
                == learner_fit_batch.get(field)
                == prediction_fold_model_bundle.get(field)
                == prediction_feature_batch.get(field)
            )
        lineage_matches = lineage_matches and (
            prediction_plan.get("candidate_sha256")
            == source_feature_batch.get("candidate_sha256")
            and prediction_plan.get("corpus_universe_sha256")
            == source_feature_batch.get("corpus_universe_sha256")
        )
        for field in (
            "calendar_sessions_sha256",
            "development_cutoff_session",
        ):
            lineage_matches = lineage_matches and (
                prediction_plan.get(field)
                == learner_fit_plan.get(field)
                == learner_fit_batch.get(field)
                == prediction_fold_model_bundle.get(field)
            )
        source_feature_plan_hash = source_feature_batch.get(
            "feature_assembly_plan_sha256"
        )
        lineage_matches = lineage_matches and (
            membership_plan.get("source_feature_assembly_plan_sha256")
            == membership_batch.get("source_feature_assembly_plan_sha256")
            == source_feature_plan_hash
            == prediction_plan.get("source_feature_assembly_plan_sha256")
        )
        authorized_fold_ids = [
            "fold_1",
            "fold_2",
            "fold_3",
            "fold_4",
            "fold_5",
        ]
        available_input_count = sum(
            item.get("prediction_available") is True
            for item in prediction_input_specs
            if type(item) is dict
        )
        unavailable_input_count = sum(
            item.get("prediction_available") is False
            for item in prediction_input_specs
            if type(item) is dict
        )
        lineage_matches = lineage_matches and (
            prediction_plan.get("source_training_membership_assembly_plan_sha256")
            == learner_fit_plan.get(
                "source_training_membership_assembly_plan_sha256"
            )
            == membership_batch.get("training_membership_assembly_plan_sha256")
            and prediction_plan.get("source_training_membership_projection_sha256")
            == learner_fit_plan.get(
                "source_training_membership_projection_sha256"
            )
            == membership_projection.get("membership_projection_sha256")
            and prediction_plan.get("source_training_membership_batch_sha256")
            == learner_fit_plan.get("source_training_membership_batch_sha256")
            == membership_batch.get("training_membership_batch_sha256")
            and prediction_plan.get("source_event_count")
            == source_feature_batch.get("event_count")
            == prediction_feature_batch.get("source_event_count")
            and prediction_plan.get("authorized_fold_count") == 5
            == prediction_plan.get("prediction_fold_model_count")
            == prediction_fold_model_bundle.get("fold_model_count")
            == prediction_feature_batch.get("prediction_fold_count")
            == len(prediction_fold_model_specs)
            and prediction_plan.get("authorized_fold_ids")
            == authorized_fold_ids
            == prediction_fold_model_bundle.get("fold_ids")
            == prediction_feature_batch.get("prediction_fold_ids")
            and prediction_plan.get("model_variant_count") == 2
            == prediction_fold_model_bundle.get("model_variant_count")
            and prediction_plan.get("model_variant_ids")
            == ["semantic", "ablation"]
            == prediction_fold_model_bundle.get("model_variant_ids")
            and prediction_plan.get("learner_state_count") == 10
            and prediction_plan.get("learner_model_type")
            == prediction_fold_model_bundle.get("learner_model_type")
            and prediction_plan.get("learner_state_schema_version")
            == prediction_fold_model_bundle.get("learner_state_schema_version")
            and prediction_plan.get("learner_config_sha256")
            == learner_fit_plan.get("learner_config_sha256")
            and prediction_plan.get("feature_schema_sha256")
            == prediction_fold_model_bundle.get("feature_schema_sha256")
            == prediction_feature_batch.get("feature_schema_sha256")
            and prediction_plan.get("prediction_input_count")
            == prediction_feature_batch.get("prediction_event_count")
            == len(prediction_input_specs)
            and prediction_plan.get("available_prediction_input_count")
            == available_input_count
            and prediction_plan.get("unavailable_prediction_input_count")
            == unavailable_input_count
            and available_input_count + unavailable_input_count
            == len(prediction_input_specs)
        )
        if not lineage_matches:
            raise SecFilingGemmaRevealStoreError(
                "Development OOF prediction crossed its owned ancestry"
            )

        body = {
            "schema_version": (
                OWNED_DEVELOPMENT_OOF_PREDICTION_PROJECTION_SCHEMA_VERSION
            ),
            "development_oof_prediction_plan": copy.deepcopy(prediction_plan),
            "prediction_fold_model_bundle": copy.deepcopy(
                prediction_fold_model_bundle
            ),
            "prediction_feature_batch": copy.deepcopy(prediction_feature_batch),
        }
        detached = _exact_builtin_json_copy(
            body,
            "owned development OOF prediction projection",
        )
        if type(detached) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "Owned development OOF prediction projection is not exact JSON"
            )
        forbidden_projection_keys = {
            "source_development_oof_learner_fit_plan",
            "source_training_membership_assembly_plan",
            "source_training_membership_batch",
            "training_set_membership",
            "semantic_training_features_hex",
            "ablation_training_features_hex",
            "training_binary_targets",
            "training_edge_targets_hex",
            "source_feature_batch",
            "source_label_batch",
            "learner_fit_views",
            "learner_fit_records",
            "deferred_training_views",
        }
        forbidden_projection_strings = {
            "intermediate_frozen_through_2018",
            "2019-01-01",
            "2023-12-31",
        }
        stack: list[Any] = [detached]
        while stack:
            value = stack.pop()
            if type(value) is dict:
                if forbidden_projection_keys.intersection(value):
                    raise SecFilingGemmaRevealStoreError(
                        "Owned development OOF prediction projection exposed a private source"
                    )
                stack.extend(value.values())
            elif type(value) is list:
                stack.extend(value)
            elif type(value) is str and value in forbidden_projection_strings:
                raise SecFilingGemmaRevealStoreError(
                    "Owned development OOF prediction projection exposed a deferred window"
                )
        (
            closure_current,
            closure_tip,
            closure_state_bytes,
            closure_tip_bytes,
        ) = self._read_state_and_tip_locked(tracked_anchor)
        if (
            closure_state_bytes != state_bytes
            or closure_tip_bytes != tip_bytes
            or closure_current != current
            or closure_tip != current_tip
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development OOF prediction authorization ancestry changed during projection"
            )
        return {
            **detached,
            "prediction_projection_sha256": canonical_sha256(detached),
        }

    def _record_owned_development_model_reader_output(
        self,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        """Rehash and terminally bind one development model component."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            scope_hash = _sha256(
                development_root_scope_sha256,
                "development model reader root scope hash",
            )
            claim = current_tip["development_model_execution_claims"].get(
                scope_hash
            )
            if type(claim) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Development model output lacks its durable execution claim"
                )
            if scope_hash in current_tip["development_model_execution_aborts"]:
                raise SecFilingGemmaRevealStoreError(
                    "Aborted development model execution cannot publish bytes"
                )
            byte_index, payloads_by_name, marker_bytes = (
                self._read_owned_model_component_locked(
                claim=claim,
                lifecycle_kind="development_root_scope",
                lifecycle_sha256=scope_hash,
                marker_schema_version=(
                    DEVELOPMENT_MODEL_EXTRACTION_COMPLETE_MARKER_SCHEMA_VERSION
                ),
                )
            )
            self._validate_owned_model_component_semantics_locked(
                authenticated_store_snapshot=current,
                independent_current_tip_anchor=current_tip,
                claim=claim,
                payloads_by_name=payloads_by_name,
            )
            closure_index, closure_payloads, closure_marker = (
                self._read_owned_model_component_locked(
                    claim=claim,
                    lifecycle_kind="development_root_scope",
                    lifecycle_sha256=scope_hash,
                    marker_schema_version=(
                        DEVELOPMENT_MODEL_EXTRACTION_COMPLETE_MARKER_SCHEMA_VERSION
                    ),
                )
            )
            if (
                closure_index != byte_index
                or closure_payloads != payloads_by_name
                or closure_marker != marker_bytes
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development model component changed during semantic replay"
                )
            try:
                receipt = build_development_model_reader_receipt(
                    claim,
                    byte_index=byte_index,
                    complete_marker_sha256=hashlib.sha256(marker_bytes).hexdigest(),
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Development model reader receipt could not be finalized"
                ) from exc
            existing = current_tip["development_model_reader_receipts"].get(
                scope_hash
            )
            if existing is not None:
                if existing != receipt:
                    raise SecFilingGemmaRevealStoreError(
                        "Persisted development model receipt differs from replayed bytes"
                    )
                return copy.deepcopy(existing)
            _state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                development_model_reader_receipt=receipt,
            )
            persisted = committed_tip["development_model_reader_receipts"].get(
                scope_hash
            )
            if persisted != receipt:
                raise SecFilingGemmaRevealStoreError(
                    "Committed development model receipt differs from rehashed bytes"
                )
            return copy.deepcopy(persisted)

    def _record_authorized_model_stage_reader_output(
        self,
        *,
        request_sha256: str,
    ) -> dict[str, Any]:
        """Rehash and terminally bind one authorized stage model component."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            request_hash = _sha256(
                request_sha256,
                "stage model reader request hash",
            )
            claim = current_tip["stage_model_execution_claims"].get(request_hash)
            if type(claim) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Stage model output lacks its durable execution claim"
                )
            if request_hash in current_tip["stage_model_execution_aborts"]:
                raise SecFilingGemmaRevealStoreError(
                    "Aborted stage model execution cannot publish bytes"
                )
            byte_index, payloads_by_name, marker_bytes = (
                self._read_owned_model_component_locked(
                claim=claim,
                lifecycle_kind="stage_request",
                lifecycle_sha256=request_hash,
                marker_schema_version=(
                    STAGE_MODEL_EXTRACTION_COMPLETE_MARKER_SCHEMA_VERSION
                ),
                )
            )
            self._validate_owned_model_component_semantics_locked(
                authenticated_store_snapshot=current,
                independent_current_tip_anchor=current_tip,
                claim=claim,
                payloads_by_name=payloads_by_name,
            )
            closure_index, closure_payloads, closure_marker = (
                self._read_owned_model_component_locked(
                    claim=claim,
                    lifecycle_kind="stage_request",
                    lifecycle_sha256=request_hash,
                    marker_schema_version=(
                        STAGE_MODEL_EXTRACTION_COMPLETE_MARKER_SCHEMA_VERSION
                    ),
                )
            )
            if (
                closure_index != byte_index
                or closure_payloads != payloads_by_name
                or closure_marker != marker_bytes
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Stage model component changed during semantic replay"
                )
            try:
                receipt = build_stage_model_reader_receipt(
                    claim,
                    byte_index=byte_index,
                    complete_marker_sha256=hashlib.sha256(marker_bytes).hexdigest(),
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Stage model reader receipt could not be finalized"
                ) from exc
            existing = current_tip["stage_model_reader_receipts"].get(
                request_hash
            )
            if existing is not None:
                if existing != receipt:
                    raise SecFilingGemmaRevealStoreError(
                        "Persisted stage model receipt differs from replayed bytes"
                    )
                return copy.deepcopy(existing)
            _state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                stage_model_reader_receipt=receipt,
            )
            persisted = committed_tip["stage_model_reader_receipts"].get(
                request_hash
            )
            if persisted != receipt:
                raise SecFilingGemmaRevealStoreError(
                    "Committed stage model receipt differs from rehashed bytes"
                )
            return copy.deepcopy(persisted)

    def claim_authorized_sec_stage_execution(
        self,
        *,
        request_sha256: str,
        sec_user_agent_sha256: str,
    ) -> dict[str, Any]:
        """Atomically claim one exact current grant before any SEC reader I/O.

        A recovered active claim is deliberately returned with ``created=False``;
        callers must not repeat the external effect because the prior process may
        have completed it without persisting its durable byte receipt.
        """

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            request_hash = _sha256(
                request_sha256,
                "authorized SEC execution request hash",
            )
            user_agent_hash = _tagged_sha256(
                sec_user_agent_sha256,
                "authorized SEC execution User-Agent hash",
            )
            existing_claim = current_tip["stage_sec_execution_claims"].get(
                request_hash
            )
            if existing_claim is not None:
                if existing_claim.get("sec_user_agent_sha256") != user_agent_hash:
                    raise SecFilingGemmaRevealStoreError(
                        "Authorized SEC execution contact differs from its durable claim"
                    )
                return {
                    "claim": copy.deepcopy(existing_claim),
                    "created": False,
                    "reader_receipt": copy.deepcopy(
                        current_tip["stage_sec_reader_receipts"].get(request_hash)
                    ),
                    "abort": copy.deepcopy(
                        current_tip["stage_sec_execution_aborts"].get(request_hash)
                    ),
                }
            bundle = current_tip["authorization_bundles"].get(request_hash)
            if type(bundle) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Authorized SEC execution lacks its current grant bundle"
                )
            execution_sources = _execution_source_hashes(self.repository_root)
            try:
                claim = build_stage_sec_execution_claim(
                    bundle,
                    independent_current_tip_anchor=current_tip,
                    execution_source_hashes=execution_sources,
                    sec_user_agent_sha256=user_agent_hash,
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Could not claim the exact current SEC stage grant"
                ) from exc
            committed_state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                stage_sec_execution_claim=claim,
            )
            if committed_state != current:
                raise SecFilingGemmaRevealStoreError(
                    "SEC execution claim changed reveal-store state"
                )
            persisted = committed_tip["stage_sec_execution_claims"].get(
                request_hash
            )
            if persisted != claim:
                raise SecFilingGemmaRevealStoreError(
                    "Committed SEC execution claim differs from its CAS target"
                )
            return {
                "claim": copy.deepcopy(persisted),
                "created": True,
                "reader_receipt": None,
                "abort": None,
            }

    def _revalidate_authorized_sec_execution_sources(
        self,
        claim: Mapping[str, Any],
    ) -> None:
        """Reject disk-source substitution after an execution claim was minted."""

        if type(claim) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "SEC execution source revalidation requires an exact claim"
            )
        if _execution_source_hashes(self.repository_root) != claim.get(
            "execution_source_hashes"
        ):
            raise SecFilingGemmaRevealStoreError(
                "SEC execution sources changed after the durable claim"
            )

    def abort_authorized_sec_stage_execution(
        self,
        *,
        request_sha256: str,
        reason: str,
    ) -> dict[str, Any]:
        """Terminally close an indeterminate SEC claim without repeating I/O."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            request_hash = _sha256(request_sha256, "SEC abort request hash")
            claim = current_tip["stage_sec_execution_claims"].get(request_hash)
            if type(claim) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "SEC execution cannot abort before its durable claim"
                )
            if request_hash in current_tip["stage_sec_reader_receipts"]:
                raise SecFilingGemmaRevealStoreError(
                    "Completed SEC execution cannot be aborted"
                )
            existing = current_tip["stage_sec_execution_aborts"].get(request_hash)
            try:
                abort = build_stage_sec_execution_abort(claim, reason=reason)
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "SEC execution abort is not canonical"
                ) from exc
            if existing is not None:
                if existing != abort:
                    raise SecFilingGemmaRevealStoreError(
                        "SEC execution already has another terminal abort"
                    )
                return copy.deepcopy(existing)
            _state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                stage_sec_execution_abort=abort,
            )
            persisted = committed_tip["stage_sec_execution_aborts"].get(
                request_hash
            )
            if persisted != abort:
                raise SecFilingGemmaRevealStoreError(
                    "Committed SEC execution abort differs from its CAS target"
                )
            return copy.deepcopy(persisted)

    def _record_authorized_sec_stage_reader_output(
        self,
        *,
        request_sha256: str,
    ) -> dict[str, Any]:
        """Re-read owned durable SEC bytes and append their exact receipt once."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            request_hash = _sha256(request_sha256, "SEC reader output request hash")
            claim = current_tip["stage_sec_execution_claims"].get(request_hash)
            if type(claim) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "SEC reader output lacks its durable execution claim"
                )
            if request_hash in current_tip["stage_sec_execution_aborts"]:
                raise SecFilingGemmaRevealStoreError(
                    "Aborted SEC execution cannot publish reader bytes"
                )
            existing = current_tip["stage_sec_reader_receipts"].get(request_hash)
            self._revalidate_authorized_sec_execution_sources(claim)
            component_directory = (
                self.store_directory
                / STAGE_OUTPUTS_DIRECTORY_NAME
                / claim["claim_sha256"]
                / SEC_STAGE_COMPONENT_DIRECTORY_NAME
            )
            secured = _secure_directory(
                component_directory,
                create=False,
                location="owned SEC stage batch directory",
            )
            marker_path = secured / SEC_BATCH_COMPLETE_MARKER_FILENAME
            marker_bytes = _read_regular_bytes(
                marker_path,
                "owned SEC stage batch complete marker",
                max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
            )
            marker = _strict_json_bytes(
                marker_bytes,
                "owned SEC stage batch complete marker",
            )
            if type(marker) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC stage batch complete marker must be an object"
                )
            expected_marker_keys = {
                "schema_version",
                "request_sha256",
                "claim_sha256",
                "component_id",
                "byte_index",
                "byte_index_sha256",
                "marker_sha256",
            }
            if set(marker) != expected_marker_keys:
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC stage batch marker keys changed"
                )
            marker_body = {
                key: marker[key] for key in marker if key != "marker_sha256"
            }
            if (
                marker["schema_version"]
                != SEC_BATCH_COMPLETE_MARKER_SCHEMA_VERSION
                or marker["request_sha256"] != request_hash
                or marker["claim_sha256"] != claim["claim_sha256"]
                or marker["component_id"]
                != SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID
                or marker["marker_sha256"] != canonical_sha256(marker_body)
                or marker_bytes != _encoded_state(marker)
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC stage batch marker is not canonical or claim-bound"
                )
            raw_index = marker["byte_index"]
            if type(raw_index) is not list or not raw_index:
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC stage batch marker has no byte index"
                )
            observed_names = [item.name for item in secured.iterdir()]
            if len(observed_names) > MAX_SEC_BATCH_FILES:
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC stage batch contains too many files"
                )
            if len({name.casefold() for name in observed_names}) != len(observed_names):
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC stage batch contains a case-colliding file"
                )
            byte_index: list[dict[str, Any]] = []
            payloads_by_name: dict[str, bytes] = {}
            total_bytes = 0
            expected_names = {SEC_BATCH_COMPLETE_MARKER_FILENAME}
            for ordinal, raw_item in enumerate(raw_index, start=1):
                if type(raw_item) is not dict or set(raw_item) != {
                    "ordinal",
                    "logical_id",
                    "relative_path",
                    "byte_count",
                    "sha256",
                }:
                    raise SecFilingGemmaRevealStoreError(
                        "Owned SEC stage byte-index item is not exact"
                    )
                relative = raw_item["relative_path"]
                if (
                    type(relative) is not str
                    or not relative
                    or "/" in relative
                    or "\\" in relative
                    or relative in {".", "..", SEC_BATCH_COMPLETE_MARKER_FILENAME}
                ):
                    raise SecFilingGemmaRevealStoreError(
                        "Owned SEC stage byte-index path is unsafe"
                    )
                payload = _read_regular_bytes(
                    secured / relative,
                    f"owned SEC stage byte {relative}",
                    max_bytes=MAX_SEC_BATCH_FILE_BYTES,
                )
                total_bytes += len(payload)
                if total_bytes > MAX_SEC_BATCH_TOTAL_BYTES:
                    raise SecFilingGemmaRevealStoreError(
                        "Owned SEC stage byte bundle exceeds its total limit"
                    )
                observed = {
                    "ordinal": ordinal,
                    "logical_id": raw_item["logical_id"],
                    "relative_path": relative,
                    "byte_count": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                if observed != raw_item:
                    raise SecFilingGemmaRevealStoreError(
                        "Owned SEC stage bytes differ from their complete marker"
                    )
                byte_index.append(observed)
                payloads_by_name[relative] = payload
                expected_names.add(relative)
            if (
                set(observed_names) != expected_names
                or marker["byte_index_sha256"] != canonical_sha256(byte_index)
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC stage directory has missing, extra, or reordered bytes"
                )
            bundle = current_tip["authorization_bundles"].get(request_hash)
            if type(bundle) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC stage batch lost its authorization bundle"
                )
            try:
                _exact_bundle, _grant, component_plan = (
                    _sec_component_plan_from_bundle(bundle)
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC stage batch cannot recover its component plan"
                ) from exc
            if canonical_sha256(component_plan) != claim["sec_component_plan_sha256"]:
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC stage batch crossed its claimed component plan"
                )
            sec_plan = component_plan["sec_access_plan"]
            documents_plan = sec_plan["documents"]
            expected_layout: list[tuple[str, str]] = []
            for document_ordinal in range(1, len(documents_plan) + 1):
                prefix = f"document-{document_ordinal:04d}"
                expected_layout.extend(
                    (
                        (f"{prefix}-raw", f"{prefix}.raw"),
                        (
                            f"{prefix}-normalized",
                            f"{prefix}.normalized.txt",
                        ),
                    )
                )
            expected_layout.extend(
                (
                    ("request-receipts-json", "request-receipts.json"),
                    ("byte-manifest-json", "byte-manifest.json"),
                )
            )
            observed_layout = [
                (item["logical_id"], item["relative_path"])
                for item in byte_index
            ]
            if observed_layout != expected_layout:
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC stage batch does not contain the exact granted evidence layout"
                )
            try:
                _validate_persisted_authenticated_stage_access_batch(
                    authenticated_document_plan=documents_plan,
                    raw_documents=tuple(
                        payloads_by_name[f"document-{ordinal:04d}.raw"]
                        for ordinal in range(1, len(documents_plan) + 1)
                    ),
                    normalized_documents=tuple(
                        payloads_by_name[
                            f"document-{ordinal:04d}.normalized.txt"
                        ]
                        for ordinal in range(1, len(documents_plan) + 1)
                    ),
                    request_receipts_json=payloads_by_name["request-receipts.json"],
                    byte_manifest_json=payloads_by_name["byte-manifest.json"],
                    expected_max_requests=component_plan["max_sec_requests"],
                    expected_max_bytes=component_plan["max_sec_response_bytes"],
                    expected_max_seconds=float(
                        component_plan["max_sec_acquisition_seconds"]
                    ),
                    expected_user_agent_sha256=claim["sec_user_agent_sha256"],
                )
            except (
                KeyError,
                SecFilingGemmaCorpusError,
                TypeError,
                ValueError,
            ) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC stage batch bytes fail independent semantic replay"
                ) from exc
            try:
                self._revalidate_authorized_sec_execution_sources(claim)
            except SecFilingGemmaRevealStoreError:
                raise SecFilingGemmaRevealStoreError(
                    "SEC execution sources changed before reader receipt finalization"
                ) from None
            try:
                receipt = build_stage_sec_reader_receipt(
                    claim,
                    byte_index=byte_index,
                    complete_marker_sha256=hashlib.sha256(marker_bytes).hexdigest(),
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC stage reader receipt could not be built"
                ) from exc
            if existing is not None:
                if existing != receipt:
                    raise SecFilingGemmaRevealStoreError(
                        "Persisted SEC reader receipt differs from replayed durable bytes"
                    )
                return copy.deepcopy(existing)
            _state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                stage_sec_reader_receipt=receipt,
            )
            persisted = committed_tip["stage_sec_reader_receipts"].get(
                request_hash
            )
            if persisted != receipt:
                raise SecFilingGemmaRevealStoreError(
                    "Committed SEC reader receipt differs from rehashed durable bytes"
                )
            return copy.deepcopy(persisted)

    def _replay_owned_sec_reader_bytes_locked(
        self,
        *,
        claim: Mapping[str, Any],
        reader_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Rehash one terminal SEC batch under the caller's existing lock."""

        claim_value = _expect_mapping(claim, "owned SEC closure claim")
        reader = _expect_mapping(
            reader_receipt,
            "owned SEC closure reader receipt",
        )
        if (
            reader.get("request_sha256") != claim_value.get("request_sha256")
            or reader.get("claim_sha256") != claim_value.get("claim_sha256")
            or reader.get("sec_component_id")
            != SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned SEC closure crossed its terminal reader ancestry"
            )
        self._revalidate_authorized_sec_execution_sources(claim_value)
        component_directory = (
            self.store_directory
            / STAGE_OUTPUTS_DIRECTORY_NAME
            / claim_value["claim_sha256"]
            / SEC_STAGE_COMPONENT_DIRECTORY_NAME
        )
        secured = _secure_directory(
            component_directory,
            create=False,
            location="owned SEC closure component directory",
        )
        directory_before = _validate_real_directory(
            secured,
            "owned SEC closure component directory",
        )
        byte_index = reader.get("byte_index")
        if (
            type(byte_index) is not list
            or not byte_index
            or reader.get("byte_index_sha256") != canonical_sha256(byte_index)
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned SEC closure reader byte index is not exact"
            )
        expected_names = {
            item.get("relative_path") for item in byte_index if type(item) is dict
        } | {SEC_BATCH_COMPLETE_MARKER_FILENAME}
        if None in expected_names or len(expected_names) != len(byte_index) + 1:
            raise SecFilingGemmaRevealStoreError(
                "Owned SEC closure reader byte index repeats a path"
            )
        names_before = [item.name for item in secured.iterdir()]
        if (
            len(names_before) != len(expected_names)
            or len({name.casefold() for name in names_before}) != len(names_before)
            or set(names_before) != expected_names
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned SEC closure directory has missing, extra, or case-colliding files"
            )
        marker_bytes = _read_regular_bytes(
            secured / SEC_BATCH_COMPLETE_MARKER_FILENAME,
            "owned SEC closure complete marker",
            max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
        )
        marker = _strict_json_bytes(
            marker_bytes,
            "owned SEC closure complete marker",
        )
        marker_keys = {
            "schema_version",
            "request_sha256",
            "claim_sha256",
            "component_id",
            "byte_index",
            "byte_index_sha256",
            "marker_sha256",
        }
        if type(marker) is not dict or set(marker) != marker_keys:
            raise SecFilingGemmaRevealStoreError(
                "Owned SEC closure complete marker keys changed"
            )
        marker_body = {
            key: marker[key] for key in marker if key != "marker_sha256"
        }
        if (
            marker.get("schema_version") != SEC_BATCH_COMPLETE_MARKER_SCHEMA_VERSION
            or marker.get("request_sha256") != claim_value["request_sha256"]
            or marker.get("claim_sha256") != claim_value["claim_sha256"]
            or marker.get("component_id") != SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID
            or marker.get("byte_index") != byte_index
            or marker.get("byte_index_sha256") != reader["byte_index_sha256"]
            or marker.get("marker_sha256") != canonical_sha256(marker_body)
            or marker_bytes != _encoded_state(marker)
            or hashlib.sha256(marker_bytes).hexdigest()
            != reader.get("complete_marker_sha256")
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned SEC closure marker differs from its reader receipt"
            )
        total_bytes = 0
        payloads: dict[str, bytes] = {}
        for ordinal, item in enumerate(byte_index, start=1):
            if (
                type(item) is not dict
                or set(item)
                != {"ordinal", "logical_id", "relative_path", "byte_count", "sha256"}
                or item.get("ordinal") != ordinal
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC closure byte-index item is not exact"
                )
            relative_path = item["relative_path"]
            if (
                type(relative_path) is not str
                or not relative_path
                or "/" in relative_path
                or "\\" in relative_path
                or relative_path in {".", "..", SEC_BATCH_COMPLETE_MARKER_FILENAME}
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC closure byte-index path is unsafe"
                )
            payload = _read_regular_bytes(
                secured / relative_path,
                f"owned SEC closure byte {relative_path}",
                max_bytes=MAX_SEC_BATCH_FILE_BYTES,
            )
            total_bytes += len(payload)
            if (
                total_bytes > MAX_SEC_BATCH_TOTAL_BYTES
                or item.get("byte_count") != len(payload)
                or item.get("sha256") != hashlib.sha256(payload).hexdigest()
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned SEC closure bytes differ from their reader receipt"
                )
            payloads[relative_path] = payload
        if total_bytes != reader.get("byte_count_total"):
            raise SecFilingGemmaRevealStoreError(
                "Owned SEC closure byte total differs from its reader receipt"
            )
        self._revalidate_authorized_sec_execution_sources(claim_value)
        if (
            _read_regular_bytes(
                secured / SEC_BATCH_COMPLETE_MARKER_FILENAME,
                "owned SEC closure marker replay",
                max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
            )
            != marker_bytes
            or any(
                _read_regular_bytes(
                    secured / relative_path,
                    f"owned SEC closure replay {relative_path}",
                    max_bytes=MAX_SEC_BATCH_FILE_BYTES,
                )
                != payload
                for relative_path, payload in payloads.items()
            )
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned SEC closure bytes changed during replay"
            )
        directory_after = _validate_real_directory(
            secured,
            "owned SEC closure component directory",
        )
        names_after = [item.name for item in secured.iterdir()]
        if (
            (directory_before.st_dev, directory_before.st_ino)
            != (directory_after.st_dev, directory_after.st_ino)
            or len(names_after) != len(expected_names)
            or len({name.casefold() for name in names_after}) != len(names_after)
            or set(names_after) != expected_names
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned SEC closure directory changed during replay"
            )
        return {
            "byte_index": copy.deepcopy(byte_index),
            "payloads_by_name": copy.deepcopy(payloads),
            "complete_marker_sha256": hashlib.sha256(marker_bytes).hexdigest(),
        }

    def _locate_owned_intermediate_development_root_carry_in_ancestry_locked(
        self,
        *,
        authenticated_store_snapshot: Mapping[str, Any],
        independent_current_tip_anchor: Mapping[str, Any],
        request_sha256: str,
        require_terminal_child_reader: bool = True,
        require_terminal_development_reader: bool = True,
    ) -> dict[str, Any]:
        """Resolve one intermediate child and its request-free root ancestry."""

        state = _expect_mapping(
            authenticated_store_snapshot,
            "development-root carry-in authenticated store snapshot",
        )
        tip = _expect_mapping(
            independent_current_tip_anchor,
            "development-root carry-in independent current tip",
        )
        request_hash = _sha256(
            request_sha256,
            "development-root carry-in request hash",
        )
        entries = state["consumption_ledger"]["entries"]
        if type(entries) is not list or not entries:
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in requires a consumed intermediate request"
            )
        child_entry = entries[-1]
        child_request = _expect_mapping(
            child_entry.get("request"),
            "development-root carry-in intermediate request",
        )
        if (
            child_entry.get("request_sha256") != request_hash
            or child_request.get("request_sha256") != request_hash
            or child_entry.get("entry_sha256")
            != state["consumption_ledger"]["chain"]["tip_sha256"]
            or child_entry.get("stage") != "intermediate"
            or child_request.get("stage") != "intermediate"
            or child_request.get("prerequisite_stage") != "development"
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in request is not the exact intermediate ledger tip"
            )
        child_bundle = tip["authorization_bundles"].get(request_hash)
        child_claim = tip["stage_sec_execution_claims"].get(request_hash)
        child_reader = tip["stage_sec_reader_receipts"].get(request_hash)
        trusted_pin = tip["trusted_stage_content_pins"].get(request_hash)
        if (
            request_hash in tip["consumed_stage_output_receipts"]
            and request_hash
            not in tip["development_root_carry_in_reader_receipts"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in cannot be added after child stage output"
            )
        required_child = (child_bundle, child_claim, trusted_pin)
        if any(type(value) is not dict for value in required_child) or (
            require_terminal_child_reader and type(child_reader) is not dict
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in lacks exact intermediate ancestry"
            )
        if (
            request_hash in tip["stage_sec_execution_aborts"]
            or child_bundle["authorization_grant"].get("request_sha256")
            != request_hash
            or child_claim.get("request_sha256") != request_hash
            or (
                child_reader is not None
                and (
                    type(child_reader) is not dict
                    or child_reader.get("claim_sha256")
                    != child_claim.get("claim_sha256")
                )
            )
            or trusted_pin.get("request_sha256") != request_hash
            or trusted_pin.get("prerequisite_stage") != "development"
            or trusted_pin.get("requested_stage") != "intermediate"
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in intermediate ancestry crossed its request"
            )

        matching_roots: list[tuple[str, dict[str, Any]]] = []
        for raw_scope_hash, raw_claim in tip[
            "development_sec_execution_claims"
        ].items():
            if type(raw_claim) is not dict:
                continue
            if all(
                raw_claim.get(field) == child_request.get(field)
                for field in (
                    "attempt_id",
                    "candidate_sha256",
                    "candidate_design_sha256",
                    "registry_entry_sha256",
                    "registry_sha256",
                    "registry_tip_sha256",
                )
            ):
                matching_roots.append((raw_scope_hash, raw_claim))
        if len(matching_roots) != 1:
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in requires one exact candidate-bound root"
            )
        root_scope_hash, development_claim = matching_roots[0]
        root_scope_hash = _sha256(
            root_scope_hash,
            "development-root carry-in root scope hash",
        )
        development_reader = tip["development_sec_reader_receipts"].get(
            root_scope_hash
        )
        if require_terminal_development_reader and type(development_reader) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in lacks its terminal root reader receipt"
            )
        if (
            root_scope_hash in tip["development_sec_execution_aborts"]
            or development_claim.get("development_root_scope_sha256")
            != root_scope_hash
            or (
                development_reader is not None
                and (
                    type(development_reader) is not dict
                    or development_reader.get("claim_sha256")
                    != development_claim.get("claim_sha256")
                )
            )
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in root ancestry is not terminal and exact"
            )
        root_start_count = development_claim.get("start_consumed_request_count")
        child_sequence = child_entry.get("sequence")
        if (
            type(root_start_count) is not int
            or type(child_sequence) is not int
            or root_start_count < 0
            or root_start_count >= child_sequence
            or root_start_count > len(entries)
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in root was not claimed before its child"
            )
        prefix_tip = (
            state["consumption_ledger"]["chain"]["genesis_tip_sha256"]
            if root_start_count == 0
            else entries[root_start_count - 1]["entry_sha256"]
        )
        if (
            development_claim.get("start_consumption_ledger_tip_sha256")
            != prefix_tip
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in root start tip is not the child ledger prefix"
            )
        child_access = _expect_mapping(
            child_entry.get("stage_access_manifest"),
            "development-root carry-in stage-access manifest",
        )
        evidence_pin = _expect_mapping(
            child_access.get("prerequisite_evidence_pin"),
            "development-root carry-in prerequisite evidence pin",
        )
        carry_scope = _expect_mapping(
            child_access.get("prior_same_form_carry_in"),
            "development-root carry-in persisted scope",
        )
        root_content_hash = (
            development_reader.get("content_manifest_sha256")
            if type(development_reader) is dict
            else None
        )
        if (
            child_access.get("stage_access_manifest_sha256")
            != child_request.get("stage_access_manifest_sha256")
            or trusted_pin.get("stage_access_manifest_sha256")
            != child_request.get("stage_access_manifest_sha256")
            or trusted_pin.get("prerequisite_stage_evidence_sha256")
            != child_request.get("prerequisite_stage_evidence_sha256")
            or evidence_pin.get("stage") != "development"
            or evidence_pin.get("content_manifest_sha256")
            != trusted_pin.get("content_manifest_sha256")
            or carry_scope.get("prerequisite_content_manifest_sha256")
            != trusted_pin.get("content_manifest_sha256")
            or (
                root_content_hash is not None
                and root_content_hash != trusted_pin.get("content_manifest_sha256")
            )
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in content identity crossed its trusted pin"
            )
        return {
            "child_entry": copy.deepcopy(child_entry),
            "child_request": copy.deepcopy(child_request),
            "child_bundle": copy.deepcopy(child_bundle),
            "child_claim": copy.deepcopy(child_claim),
            "child_reader": copy.deepcopy(child_reader),
            "trusted_pin": copy.deepcopy(trusted_pin),
            "development_root_scope_sha256": root_scope_hash,
            "development_claim": copy.deepcopy(development_claim),
            "development_reader": copy.deepcopy(development_reader),
        }

    def _owned_intermediate_development_root_scope_sha256(
        self,
        *,
        request_sha256: str,
    ) -> str:
        """Resolve the root before replaying either terminal reader."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            ancestry = (
                self._locate_owned_intermediate_development_root_carry_in_ancestry_locked(
                    authenticated_store_snapshot=current,
                    independent_current_tip_anchor=current_tip,
                    request_sha256=request_sha256,
                    require_terminal_child_reader=False,
                    require_terminal_development_reader=False,
                )
            )
            return ancestry["development_root_scope_sha256"]

    def _locate_owned_final_carry_in_ancestry_locked(
        self,
        *,
        authenticated_store_snapshot: Mapping[str, Any],
        independent_current_tip_anchor: Mapping[str, Any],
        request_sha256: str,
        require_terminal_child_reader: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Resolve a final child and its immediately preceding parent internally."""

        state = _expect_mapping(
            authenticated_store_snapshot,
            "owned carry-in authenticated store snapshot",
        )
        tip = _expect_mapping(
            independent_current_tip_anchor,
            "owned carry-in independent current tip",
        )
        request_hash = _sha256(request_sha256, "owned carry-in request hash")
        if type(require_terminal_child_reader) is not bool:
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in child-reader policy must be exact"
            )
        entries = state["consumption_ledger"]["entries"]
        if type(entries) is not list or len(entries) < 2:
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in requires a consumed final child and intermediate parent"
            )
        child_entry = entries[-1]
        parent_entry = entries[-2]
        child_request = _expect_mapping(
            child_entry.get("request"),
            "owned carry-in final child request",
        )
        parent_request = _expect_mapping(
            parent_entry.get("request"),
            "owned carry-in intermediate parent request",
        )
        if (
            child_entry.get("request_sha256") != request_hash
            or child_request.get("request_sha256") != request_hash
            or child_entry.get("entry_sha256")
            != state["consumption_ledger"]["chain"]["tip_sha256"]
            or child_request.get("stage") != "final"
            or child_request.get("prerequisite_stage") != "intermediate"
            or parent_entry.get("stage") != "intermediate"
            or parent_request.get("stage") != "intermediate"
            or parent_request.get("prerequisite_stage") != "development"
            or child_entry.get("sequence") != parent_entry.get("sequence") + 1
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in request is not the exact final ledger tip after its parent"
            )
        for field in (
            "attempt_id",
            "candidate_sha256",
            "candidate_design_sha256",
            "registry_entry_sha256",
            "registry_sha256",
            "registry_tip_sha256",
        ):
            if child_request.get(field) != parent_request.get(field):
                raise SecFilingGemmaRevealStoreError(
                    f"Owned carry-in crossed parent/child identity {field}"
                )
        parent_hash = _sha256(
            parent_request.get("request_sha256"),
            "owned carry-in parent request hash",
        )
        child_bundle = tip["authorization_bundles"].get(request_hash)
        parent_bundle = tip["authorization_bundles"].get(parent_hash)
        child_claim = tip["stage_sec_execution_claims"].get(request_hash)
        parent_claim = tip["stage_sec_execution_claims"].get(parent_hash)
        child_reader = tip["stage_sec_reader_receipts"].get(request_hash)
        parent_reader = tip["stage_sec_reader_receipts"].get(parent_hash)
        parent_output = tip["consumed_stage_output_receipts"].get(parent_hash)
        required_values = (
            child_bundle,
            parent_bundle,
            child_claim,
            parent_claim,
            parent_reader,
            parent_output,
        )
        if any(type(value) is not dict for value in required_values) or (
            require_terminal_child_reader and type(child_reader) is not dict
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in lacks exact child or parent durable ancestry"
            )
        if (
            request_hash in tip["stage_sec_execution_aborts"]
            or parent_hash in tip["stage_sec_execution_aborts"]
            or child_bundle["authorization_grant"].get("request_sha256")
            != request_hash
            or parent_bundle["authorization_grant"].get("request_sha256")
            != parent_hash
            or child_claim.get("request_sha256") != request_hash
            or parent_claim.get("request_sha256") != parent_hash
            or (
                child_reader is not None
                and (
                    type(child_reader) is not dict
                    or child_reader.get("claim_sha256")
                    != child_claim.get("claim_sha256")
                )
            )
            or parent_reader.get("claim_sha256")
            != parent_claim.get("claim_sha256")
            or parent_output.get("request_sha256") != parent_hash
            or parent_output.get("output_stage_evidence_sha256")
            != child_request.get("prerequisite_stage_evidence_sha256")
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in ancestry crossed its exact final parent boundary"
            )
        return {
            "child_entry": copy.deepcopy(child_entry),
            "child_request": copy.deepcopy(child_request),
            "child_bundle": copy.deepcopy(child_bundle),
            "child_claim": copy.deepcopy(child_claim),
            "child_reader": copy.deepcopy(child_reader),
            "parent_entry": copy.deepcopy(parent_entry),
            "parent_request": copy.deepcopy(parent_request),
            "parent_bundle": copy.deepcopy(parent_bundle),
            "parent_claim": copy.deepcopy(parent_claim),
            "parent_reader": copy.deepcopy(parent_reader),
            "parent_output_receipt": copy.deepcopy(parent_output),
        }

    def _owned_final_carry_in_parent_request_sha256(
        self,
        *,
        request_sha256: str,
    ) -> str:
        """Resolve the parent hash before replaying its reader outside the lock."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            ancestry = self._locate_owned_final_carry_in_ancestry_locked(
                authenticated_store_snapshot=current,
                independent_current_tip_anchor=current_tip,
                request_sha256=request_sha256,
                require_terminal_child_reader=False,
            )
            return ancestry["parent_request"]["request_sha256"]

    def _read_owned_stage_evidence_output_locked(
        self,
        *,
        authenticated_store_snapshot: Mapping[str, Any],
        independent_current_tip_anchor: Mapping[str, Any],
        authorization_bundle: Mapping[str, Any],
        request_sha256: str,
        require_current_store_state: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Replay one fixed durable stage-evidence document under the store lock."""

        current = _expect_mapping(
            authenticated_store_snapshot,
            "owned stage-evidence authenticated store snapshot",
        )
        current_tip = _expect_mapping(
            independent_current_tip_anchor,
            "owned stage-evidence independent current tip",
        )
        request_hash = _sha256(
            request_sha256,
            "owned stage-evidence request hash",
        )
        bundle = _expect_mapping(
            authorization_bundle,
            "owned stage-evidence authorization bundle",
        )
        if current_tip["authorization_bundles"].get(request_hash) != bundle:
            raise SecFilingGemmaRevealStoreError(
                "Owned stage evidence lost its exact authorization bundle"
            )
        grant = _expect_mapping(
            bundle.get("authorization_grant"),
            "owned stage-evidence authorization grant",
        )
        claim = current_tip["stage_sec_execution_claims"].get(request_hash)
        sec_reader = current_tip["stage_sec_reader_receipts"].get(request_hash)
        if type(claim) is not dict or type(sec_reader) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "Owned stage evidence requires its exact SEC claim and reader receipt"
            )
        if request_hash in current_tip["stage_sec_execution_aborts"]:
            raise SecFilingGemmaRevealStoreError(
                "Aborted SEC execution cannot publish stage evidence"
            )
        if (
            sec_reader.get("claim_sha256") != claim.get("claim_sha256")
            or sec_reader.get("request_sha256") != request_hash
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned stage evidence SEC ancestry is inconsistent"
            )
        if type(require_current_store_state) is not bool:
            raise SecFilingGemmaRevealStoreError(
                "Owned stage-evidence current-state policy must be exact"
            )
        if require_current_store_state:
            try:
                validate_consumed_stage_authorization_grant(
                    grant,
                    authenticated_store_snapshot=current,
                    external_store_state_pin=bundle["store_state_pin"],
                    independent_current_tip_anchor=current_tip,
                    expected_consumption_entry_sha256=grant[
                        "consumption_entry_sha256"
                    ],
                    expected_request_sha256=request_hash,
                    expected_candidate_sha256=grant["candidate_sha256"],
                    expected_stage=grant["stage"],
                    expected_prerequisite_stage_evidence_sha256=grant[
                        "prerequisite_stage_evidence_sha256"
                    ],
                    expected_stage_access_manifest_sha256=grant[
                        "stage_access_manifest_sha256"
                    ],
                    expected_output_namespace=grant["output_namespace"],
                )
            except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Owned stage evidence lacks an exact current grant"
                ) from exc
        self._revalidate_authorized_sec_execution_sources(claim)

        component_directory = (
            self.store_directory
            / STAGE_OUTPUTS_DIRECTORY_NAME
            / claim["claim_sha256"]
            / STAGE_EVIDENCE_COMPONENT_DIRECTORY_NAME
        )
        secured = _secure_directory(
            component_directory,
            create=False,
            location="owned stage-evidence directory",
        )
        directory_before = _validate_real_directory(
            secured,
            "owned stage-evidence directory",
        )
        observed_names = [item.name for item in secured.iterdir()]
        expected_names = {
            STAGE_EVIDENCE_FILENAME,
            STAGE_EVIDENCE_COMPLETE_MARKER_FILENAME,
        }
        if (
            len(observed_names) != len(expected_names)
            or len({name.casefold() for name in observed_names})
            != len(observed_names)
            or set(observed_names) != expected_names
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned stage-evidence directory has missing, extra, or case-colliding files"
            )

        evidence_bytes = _read_regular_bytes(
            secured / STAGE_EVIDENCE_FILENAME,
            "owned stage-evidence document",
            max_bytes=MAX_STAGE_EVIDENCE_FILE_BYTES,
        )
        evidence = _strict_json_bytes(
            evidence_bytes,
            "owned stage-evidence document",
        )
        if type(evidence) is not dict:
            raise SecFilingGemmaRevealStoreError(
                "Owned stage-evidence document must be an object"
            )
        try:
            preflight_untrusted_stage_json(
                evidence,
                "owned stage-evidence document",
            )
            detached_evidence = detach_untrusted_stage_json(
                evidence,
                "owned stage-evidence document",
            )
        except Exception as exc:
            raise SecFilingGemmaRevealStoreError(
                "Owned stage-evidence document exceeds fixed allocation bounds"
            ) from exc
        if (
            type(detached_evidence) is not dict
            or set(detached_evidence) != STAGE_EVIDENCE_KEYS
            or detached_evidence.get("schema_version")
            != STAGE_EVIDENCE_SCHEMA_VERSION
            or type(detached_evidence.get("parent_stage_evidence_sha256"))
            is not str
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned stage-evidence document is not the exact v3 envelope"
            )
        try:
            canonical_evidence_bytes = json.dumps(
                detached_evidence,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        except (RecursionError, TypeError, ValueError) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Owned stage-evidence document is not finite canonical JSON"
            ) from exc
        if evidence_bytes != canonical_evidence_bytes:
            raise SecFilingGemmaRevealStoreError(
                "Owned stage-evidence bytes are not exact compact canonical JSON"
            )
        binding = _stage_evidence_output_binding(
            detached_evidence,
            authorization_grant=grant,
        )
        if (
            binding["output_stage_evidence_schema_version"]
            != STAGE_EVIDENCE_SCHEMA_VERSION
            or binding["output_stage_evidence_document_sha256"]
            != hashlib.sha256(evidence_bytes).hexdigest()
            or binding["output_stage_evidence_canonical_byte_count"]
            != len(evidence_bytes)
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned stage-evidence binding differs from its durable bytes"
            )

        marker_bytes = _read_regular_bytes(
            secured / STAGE_EVIDENCE_COMPLETE_MARKER_FILENAME,
            "owned stage-evidence complete marker",
            max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
        )
        marker = _strict_json_bytes(
            marker_bytes,
            "owned stage-evidence complete marker",
        )
        if type(marker) is not dict or set(marker) != _STAGE_EVIDENCE_COMPLETE_MARKER_KEYS:
            raise SecFilingGemmaRevealStoreError(
                "Owned stage-evidence complete marker keys changed"
            )
        marker_body = {
            key: marker[key] for key in marker if key != "marker_sha256"
        }
        if (
            marker.get("schema_version")
            != STAGE_EVIDENCE_COMPLETE_MARKER_SCHEMA_VERSION
            or marker.get("request_sha256") != request_hash
            or marker.get("claim_sha256") != claim["claim_sha256"]
            or marker.get("sec_reader_receipt_sha256")
            != sec_reader["receipt_sha256"]
            or marker.get("component_id") != STAGE_EVIDENCE_OUTPUT_COMPONENT_ID
            or marker.get("relative_path") != STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH
            or type(marker.get("byte_count")) is not int
            or marker.get("byte_count") != len(evidence_bytes)
            or marker.get("document_sha256")
            != binding["output_stage_evidence_document_sha256"]
            or marker.get("stage_evidence_sha256")
            != binding["output_stage_evidence_sha256"]
            or marker.get("marker_sha256") != canonical_sha256(marker_body)
            or marker_bytes != _encoded_state(marker)
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned stage-evidence marker is not canonical or ancestry-bound"
            )
        self._revalidate_authorized_sec_execution_sources(claim)
        if (
            _read_regular_bytes(
                secured / STAGE_EVIDENCE_FILENAME,
                "owned stage-evidence document closure replay",
                max_bytes=MAX_STAGE_EVIDENCE_FILE_BYTES,
            )
            != evidence_bytes
            or _read_regular_bytes(
                secured / STAGE_EVIDENCE_COMPLETE_MARKER_FILENAME,
                "owned stage-evidence complete marker closure replay",
                max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
            )
            != marker_bytes
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned stage-evidence bytes changed during complete replay"
            )
        directory_after = _validate_real_directory(
            secured,
            "owned stage-evidence directory",
        )
        observed_names_after = [item.name for item in secured.iterdir()]
        if (
            (directory_before.st_dev, directory_before.st_ino)
            != (directory_after.st_dev, directory_after.st_ino)
            or len(observed_names_after) != len(expected_names)
            or len({name.casefold() for name in observed_names_after})
            != len(observed_names_after)
            or set(observed_names_after) != expected_names
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned stage-evidence directory changed while replaying bytes"
            )
        self._revalidate_authorized_sec_execution_sources(claim)
        return (
            copy.deepcopy(detached_evidence),
            {
                **binding,
                "output_stage_evidence_complete_marker_sha256": hashlib.sha256(
                    marker_bytes
                ).hexdigest(),
            },
        )

    def _read_owned_development_root_carry_source_locked(
        self,
        *,
        authenticated_store_snapshot: Mapping[str, Any],
        independent_current_tip_anchor: Mapping[str, Any],
        development_sec_execution_claim: Mapping[str, Any],
        development_sec_reader_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Replay the full root directory before selecting any carry-in bytes."""

        current = _expect_mapping(
            authenticated_store_snapshot,
            "development-root carry source store snapshot",
        )
        current_tip = _expect_mapping(
            independent_current_tip_anchor,
            "development-root carry source current tip",
        )
        claim = _expect_mapping(
            development_sec_execution_claim,
            "development-root carry source claim",
        )
        reader = _expect_mapping(
            development_sec_reader_receipt,
            "development-root carry source reader receipt",
        )
        root_scope_hash = _sha256(
            claim.get("development_root_scope_sha256"),
            "development-root carry source scope hash",
        )
        if (
            current_tip["development_sec_execution_claims"].get(root_scope_hash)
            != claim
            or current_tip["development_sec_reader_receipts"].get(root_scope_hash)
            != reader
            or root_scope_hash in current_tip["development_sec_execution_aborts"]
            or reader.get("claim_sha256") != claim.get("claim_sha256")
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry source is not exact current-tip ancestry"
            )
        self._revalidate_authorized_sec_execution_sources(claim)
        plan = _validated_development_root_plan_for_store(
            current,
            claim.get("development_content_root_plan"),
            require_latest_candidate=False,
        )
        if (
            plan["development_root_scope_sha256"] != root_scope_hash
            or plan["development_content_root_plan_sha256"]
            != claim.get("development_content_root_plan_sha256")
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry source crossed its root plan"
            )
        component_directory = (
            self.store_directory
            / STAGE_OUTPUTS_DIRECTORY_NAME
            / claim["claim_sha256"]
            / SEC_STAGE_COMPONENT_DIRECTORY_NAME
        )
        secured = _secure_directory(
            component_directory,
            create=False,
            location="development-root carry source directory",
        )
        directory_before = _validate_real_directory(
            secured,
            "development-root carry source directory",
        )
        marker_path = secured / SEC_BATCH_COMPLETE_MARKER_FILENAME
        marker_bytes = _read_regular_bytes(
            marker_path,
            "development-root carry source complete marker",
            max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
        )
        marker = _strict_json_bytes(
            marker_bytes,
            "development-root carry source complete marker",
        )
        expected_marker_keys = {
            "schema_version",
            "development_root_scope_sha256",
            "claim_sha256",
            "candidate_sha256",
            "corpus_universe_sha256",
            "development_content_root_plan_sha256",
            "component_id",
            "development_content_manifest_sha256",
            "byte_index",
            "byte_index_sha256",
            "marker_sha256",
        }
        if type(marker) is not dict or set(marker) != expected_marker_keys:
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry source marker keys changed"
            )
        marker_body = {
            key: marker[key] for key in marker if key != "marker_sha256"
        }
        if (
            marker["schema_version"]
            != DEVELOPMENT_SEC_ROOT_COMPLETE_MARKER_SCHEMA_VERSION
            or marker["development_root_scope_sha256"] != root_scope_hash
            or marker["claim_sha256"] != claim["claim_sha256"]
            or marker["candidate_sha256"] != claim["candidate_sha256"]
            or marker["corpus_universe_sha256"]
            != claim["corpus_universe_sha256"]
            or marker["development_content_root_plan_sha256"]
            != claim["development_content_root_plan_sha256"]
            or marker["component_id"] != DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID
            or marker["marker_sha256"] != canonical_sha256(marker_body)
            or marker_bytes != _encoded_state(marker)
            or hashlib.sha256(marker_bytes).hexdigest()
            != reader.get("complete_marker_sha256")
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry source marker is not exact ancestry"
            )
        byte_index, payloads_by_name = _read_owned_sec_indexed_payloads(
            secured,
            marker["byte_index"],
            location="development-root carry source directory",
        )
        if (
            marker["byte_index_sha256"] != canonical_sha256(byte_index)
            or reader.get("byte_index") != byte_index
            or reader.get("byte_index_sha256") != canonical_sha256(byte_index)
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry source byte index crossed its receipt"
            )
        documents_plan = plan["sec_access_plan"]["documents"]
        expected_layout: list[tuple[str, str]] = []
        for document_ordinal in range(1, len(documents_plan) + 1):
            prefix = f"document-{document_ordinal:04d}"
            expected_layout.extend(
                (
                    (f"{prefix}-raw", f"{prefix}.raw"),
                    (f"{prefix}-normalized", f"{prefix}.normalized.txt"),
                )
            )
        expected_layout.extend(
            (
                ("request-receipts-json", "request-receipts.json"),
                ("byte-manifest-json", "byte-manifest.json"),
                ("corpus-universe-json", "corpus-universe.json"),
                (
                    "development-content-manifest-json",
                    "development-content-manifest.json",
                ),
            )
        )
        if [
            (item["logical_id"], item["relative_path"]) for item in byte_index
        ] != expected_layout:
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry source layout changed"
            )
        universe_bytes = payloads_by_name["corpus-universe.json"]
        universe = _strict_json_bytes(
            universe_bytes,
            "development-root carry source corpus universe",
        )
        content_manifest_bytes = payloads_by_name[
            "development-content-manifest.json"
        ]
        content_manifest = _strict_json_bytes(
            content_manifest_bytes,
            "development-root carry source content manifest",
        )
        try:
            validate_stage_content_manifest(
                content_manifest,
                universe_manifest=universe,
                expected_content_manifest_sha256=reader[
                    "content_manifest_sha256"
                ],
            )
        except (
            KeyError,
            SecFilingGemmaContractError,
            TypeError,
            ValueError,
        ) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry source content manifest does not replay"
            ) from exc
        if (
            type(universe) is not dict
            or universe != plan["corpus_universe_manifest"]
            or universe_bytes != _encoded_state(universe)
            or type(content_manifest) is not dict
            or content_manifest_bytes != _encoded_state(content_manifest)
            or marker["development_content_manifest_sha256"]
            != content_manifest["content_manifest_sha256"]
            or reader["content_manifest_sha256"]
            != content_manifest["content_manifest_sha256"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry source universe or content changed"
            )
        directory_after = _validate_real_directory(
            secured,
            "development-root carry source directory",
        )
        observed_names = [item.name for item in secured.iterdir()]
        expected_names = {
            item["relative_path"] for item in byte_index
        } | {SEC_BATCH_COMPLETE_MARKER_FILENAME}
        if (
            (directory_before.st_dev, directory_before.st_ino)
            != (directory_after.st_dev, directory_after.st_ino)
            or len(observed_names) != len(expected_names)
            or len({name.casefold() for name in observed_names})
            != len(observed_names)
            or set(observed_names) != expected_names
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry source directory changed during replay"
            )
        return {
            "plan": copy.deepcopy(plan),
            "universe": copy.deepcopy(universe),
            "content_manifest": copy.deepcopy(content_manifest),
            "byte_index": copy.deepcopy(byte_index),
            "payloads_by_name": copy.deepcopy(payloads_by_name),
            "complete_marker_sha256": hashlib.sha256(marker_bytes).hexdigest(),
        }

    def _read_and_seal_owned_development_root_carry_in_locked(
        self,
        *,
        authenticated_store_snapshot: Mapping[str, Any],
        independent_current_tip_anchor: Mapping[str, Any],
        request_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Copy only the two child-authorized normalized root documents."""

        current = _expect_mapping(
            authenticated_store_snapshot,
            "development-root carry-in authenticated store snapshot",
        )
        current_tip = _expect_mapping(
            independent_current_tip_anchor,
            "development-root carry-in independent current tip",
        )
        ancestry = (
            self._locate_owned_intermediate_development_root_carry_in_ancestry_locked(
                authenticated_store_snapshot=current,
                independent_current_tip_anchor=current_tip,
                request_sha256=request_sha256,
            )
        )
        child_entry = ancestry["child_entry"]
        child_request = ancestry["child_request"]
        child_claim = ancestry["child_claim"]
        child_reader = ancestry["child_reader"]
        trusted_pin = ancestry["trusted_pin"]
        development_claim = ancestry["development_claim"]
        development_reader = ancestry["development_reader"]
        source = self._read_owned_development_root_carry_source_locked(
            authenticated_store_snapshot=current,
            independent_current_tip_anchor=current_tip,
            development_sec_execution_claim=development_claim,
            development_sec_reader_receipt=development_reader,
        )
        child_access = _expect_mapping(
            child_entry.get("stage_access_manifest"),
            "development-root carry-in child stage access",
        )
        raw_carry_scope = _expect_mapping(
            child_access.get("prior_same_form_carry_in"),
            "development-root prior same-form carry-in scope",
        )
        scope = _expect_mapping(
            child_access.get("scope"),
            "development-root carry-in child access scope",
        )
        content_hash = _sha256(
            source["content_manifest"].get("content_manifest_sha256"),
            "development-root carry-in prerequisite content hash",
        )
        try:
            derived_records = validate_prior_same_form_carry_in_scope(
                child_access,
                corpus_universe_manifest=source["universe"],
                prerequisite_content_manifest=source["content_manifest"],
                expected_prerequisite_stage="development",
                expected_requested_stage="intermediate",
                expected_prerequisite_stage_evidence_sha256=child_request[
                    "prerequisite_stage_evidence_sha256"
                ],
            )
        except (KeyError, SecFilingGemmaStageAccessError, TypeError) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in scope cannot be rederived from root bytes"
            ) from exc
        if (
            len(derived_records) != 2
            or [record.get("form") for record in derived_records]
            != ["10-K", "10-Q"]
            or raw_carry_scope.get("records") != derived_records
            or raw_carry_scope.get("records_sha256")
            != canonical_sha256(derived_records)
            or raw_carry_scope.get("prerequisite_content_manifest_sha256")
            != content_hash
            or trusted_pin.get("content_manifest_sha256") != content_hash
            or development_reader.get("content_manifest_sha256") != content_hash
            or scope.get("authorized_stage") != "intermediate"
            or scope.get("prerequisite_stage_data_scope")
            != (
                "sealed_evidence_identity_plus_exact_read_only_prior_same_form_"
                "normalized_text_carry_in"
            )
            or scope.get("general_cross_stage_access_permitted") is not False
            or scope.get("exact_prior_same_form_carry_in_read_permitted") is not True
            or scope.get("future_stage_access_permitted") is not False
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in scope differs from exact root derivation"
            )

        root_documents = source["plan"]["sec_access_plan"]["documents"]
        document_ordinal_by_accession = {
            document["accession_number"]: ordinal
            for ordinal, document in enumerate(root_documents, start=1)
        }
        if len(document_ordinal_by_accession) != len(root_documents):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in source repeats an accession"
            )
        source_rows_by_path = {
            row["relative_path"]: row for row in source["byte_index"]
        }
        content_by_accession = {
            document["accession_number"]: document
            for document in source["content_manifest"]["documents"]
        }
        selected: list[tuple[dict[str, Any], bytes, str]] = []
        total_bytes = 0
        for carry_ordinal, record in enumerate(derived_records, start=1):
            source_ordinal = document_ordinal_by_accession.get(
                record["accession_number"]
            )
            content_record = content_by_accession.get(record["accession_number"])
            if type(source_ordinal) is not int or type(content_record) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Development-root carry-in accession is absent from root evidence"
                )
            source_relative_path = (
                f"document-{source_ordinal:04d}.normalized.txt"
            )
            source_row = source_rows_by_path.get(source_relative_path)
            payload = source["payloads_by_name"].get(source_relative_path)
            if type(source_row) is not dict or type(payload) is not bytes:
                raise SecFilingGemmaRevealStoreError(
                    "Development-root carry-in normalized source is unavailable"
                )
            try:
                if payload.decode("utf-8").encode("utf-8") != payload:
                    raise UnicodeError
            except UnicodeError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Development-root carry-in normalized text is not exact UTF-8"
                ) from exc
            payload_hash = hashlib.sha256(payload).hexdigest()
            total_bytes += len(payload)
            if (
                total_bytes > MAX_PRIOR_SAME_FORM_CARRY_IN_TOTAL_BYTES
                or source_row.get("logical_id")
                != f"document-{source_ordinal:04d}-normalized"
                or source_row.get("byte_count") != len(payload)
                or source_row.get("sha256") != payload_hash
                or content_record.get("normalized_text_sha256") != payload_hash
                or content_record.get("normalized_text_bytes") != len(payload)
                or canonical_sha256(content_record)
                != record.get("content_record_sha256")
                or record.get("normalized_text_sha256") != payload_hash
                or record.get("normalized_text_bytes") != len(payload)
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development-root carry-in bytes crossed their sealed record"
                )
            destination_relative_path = (
                f"carry-in-{carry_ordinal:04d}.normalized.txt"
            )
            row = {
                "ordinal": carry_ordinal,
                "logical_id": f"carry-in-{carry_ordinal:04d}-normalized",
                "relative_path": destination_relative_path,
                "byte_count": len(payload),
                "sha256": payload_hash,
            }
            selected.append((row, payload, source_relative_path))

        child_claim_directory = (
            self.store_directory
            / STAGE_OUTPUTS_DIRECTORY_NAME
            / child_claim["claim_sha256"]
        )
        secured_child_claim_directory = _secure_directory(
            child_claim_directory,
            create=False,
            location="development-root carry-in intermediate claim directory",
        )
        component_directory = (
            secured_child_claim_directory
            / DEVELOPMENT_ROOT_CARRY_IN_COMPONENT_DIRECTORY_NAME
        )
        receipt_present = (
            child_request["request_sha256"]
            in current_tip["development_root_carry_in_reader_receipts"]
        )
        if not receipt_present:
            try:
                component_directory.mkdir(exist_ok=False)
                _fsync_directory(secured_child_claim_directory)
            except FileExistsError:
                pass
        secured_component = _secure_directory(
            component_directory,
            create=False,
            location="owned development-root carry-in component directory",
        )
        component_directory_before = _validate_real_directory(
            secured_component,
            "owned development-root carry-in component directory",
        )
        byte_index = [row for row, _payload, _source in selected]
        marker_body = {
            "schema_version": (
                DEVELOPMENT_ROOT_CARRY_IN_COMPLETE_MARKER_SCHEMA_VERSION
            ),
            "request_sha256": child_request["request_sha256"],
            "claim_sha256": child_claim["claim_sha256"],
            "sec_reader_receipt_sha256": child_reader["receipt_sha256"],
            "stage_access_manifest_sha256": child_request[
                "stage_access_manifest_sha256"
            ],
            "trusted_stage_content_pin_sha256": trusted_pin["pin_sha256"],
            "prerequisite_stage_evidence_sha256": child_request[
                "prerequisite_stage_evidence_sha256"
            ],
            "development_root_scope_sha256": ancestry[
                "development_root_scope_sha256"
            ],
            "development_claim_sha256": development_claim["claim_sha256"],
            "development_sec_reader_receipt_sha256": development_reader[
                "receipt_sha256"
            ],
            "development_content_root_plan_sha256": development_claim[
                "development_content_root_plan_sha256"
            ],
            "development_content_manifest_sha256": content_hash,
            "development_root_complete_marker_sha256": development_reader[
                "complete_marker_sha256"
            ],
            "corpus_universe_sha256": development_claim[
                "corpus_universe_sha256"
            ],
            "component_id": DEVELOPMENT_ROOT_CARRY_IN_COMPONENT_ID,
            "carry_in_records_sha256": canonical_sha256(derived_records),
            "prerequisite_content_manifest_sha256": content_hash,
            "byte_index": byte_index,
            "byte_index_sha256": canonical_sha256(byte_index),
            "byte_count_total": total_bytes,
        }
        marker = {**marker_body, "marker_sha256": canonical_sha256(marker_body)}
        marker_bytes = _encoded_state(marker)
        expected_names = {
            row["relative_path"] for row, _payload, _source in selected
        } | {DEVELOPMENT_ROOT_CARRY_IN_COMPLETE_MARKER_FILENAME}
        observed_names_before = [item.name for item in secured_component.iterdir()]
        marker_present = (
            DEVELOPMENT_ROOT_CARRY_IN_COMPLETE_MARKER_FILENAME
            in observed_names_before
        )
        if marker_present:
            marker_path = (
                secured_component
                / DEVELOPMENT_ROOT_CARRY_IN_COMPLETE_MARKER_FILENAME
            )
            structurally_committed_marker = False
            try:
                existing_marker_bytes = _read_regular_bytes(
                    marker_path,
                    "owned development-root carry-in pre-existing marker",
                    max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
                )
                existing_marker = _strict_json_bytes(
                    existing_marker_bytes,
                    "owned development-root carry-in pre-existing marker",
                )
                if type(existing_marker) is dict:
                    existing_marker_body = {
                        key: existing_marker[key]
                        for key in existing_marker
                        if key != "marker_sha256"
                    }
                    structurally_committed_marker = (
                        set(existing_marker)
                        == _DEVELOPMENT_ROOT_CARRY_IN_MARKER_KEYS
                        and existing_marker.get("schema_version")
                        == DEVELOPMENT_ROOT_CARRY_IN_COMPLETE_MARKER_SCHEMA_VERSION
                        and existing_marker.get("marker_sha256")
                        == canonical_sha256(existing_marker_body)
                        and existing_marker_bytes == _encoded_state(existing_marker)
                    )
            except (KeyError, SecFilingGemmaRevealStoreError):
                structurally_committed_marker = False
            if not structurally_committed_marker:
                if receipt_present:
                    raise SecFilingGemmaRevealStoreError(
                        "Persisted development-root carry-in receipt lost its marker"
                    )
                _validate_regular_details(
                    marker_path.lstat(),
                    "interrupted development-root carry-in marker",
                )
                marker_path.unlink()
                _fsync_directory(secured_component)
                observed_names_before = [
                    name
                    for name in observed_names_before
                    if name != DEVELOPMENT_ROOT_CARRY_IN_COMPLETE_MARKER_FILENAME
                ]
                marker_present = False
        if (
            len({name.casefold() for name in observed_names_before})
            != len(observed_names_before)
            or not set(observed_names_before).issubset(expected_names)
            or (
                (marker_present or receipt_present)
                and set(observed_names_before) != expected_names
            )
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in component has an extra or colliding file"
            )
        recover_interrupted_precommit = not marker_present and not receipt_present
        for row, payload, _source_relative_path in selected:
            _create_or_replay_regular_bytes(
                secured_component / row["relative_path"],
                payload,
                f"development-root carry-in normalized text {row['ordinal']}",
                max_bytes=MAX_SEC_BATCH_FILE_BYTES,
                recover_interrupted_precommit=recover_interrupted_precommit,
            )
        _create_or_replay_regular_bytes(
            secured_component / DEVELOPMENT_ROOT_CARRY_IN_COMPLETE_MARKER_FILENAME,
            marker_bytes,
            "development-root carry-in complete marker",
            max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
        )
        replayed_marker = _strict_json_bytes(
            _read_regular_bytes(
                secured_component
                / DEVELOPMENT_ROOT_CARRY_IN_COMPLETE_MARKER_FILENAME,
                "development-root carry-in complete marker replay",
                max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
            ),
            "development-root carry-in complete marker replay",
        )
        if (
            type(replayed_marker) is not dict
            or set(replayed_marker) != _DEVELOPMENT_ROOT_CARRY_IN_MARKER_KEYS
            or replayed_marker != marker
            or _encoded_state(replayed_marker) != marker_bytes
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in marker is not exact canonical ancestry"
            )
        for row, payload, _source_relative_path in selected:
            if (
                _read_regular_bytes(
                    secured_component / row["relative_path"],
                    f"development-root carry-in closure text {row['ordinal']}",
                    max_bytes=MAX_SEC_BATCH_FILE_BYTES,
                )
                != payload
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development-root carry-in bytes changed during closure replay"
                )
        closure_source = self._read_owned_development_root_carry_source_locked(
            authenticated_store_snapshot=current,
            independent_current_tip_anchor=current_tip,
            development_sec_execution_claim=development_claim,
            development_sec_reader_receipt=development_reader,
        )
        component_directory_after = _validate_real_directory(
            secured_component,
            "owned development-root carry-in component directory",
        )
        observed_names_after = [item.name for item in secured_component.iterdir()]
        if (
            closure_source != source
            or (
                component_directory_before.st_dev,
                component_directory_before.st_ino,
            )
            != (
                component_directory_after.st_dev,
                component_directory_after.st_ino,
            )
            or len(observed_names_after) != len(expected_names)
            or len({name.casefold() for name in observed_names_after})
            != len(observed_names_after)
            or set(observed_names_after) != expected_names
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in source or destination changed during replay"
            )
        self._revalidate_authorized_sec_execution_sources(child_claim)
        self._revalidate_authorized_sec_execution_sources(development_claim)
        self._replay_owned_sec_reader_bytes_locked(
            claim=child_claim,
            reader_receipt=child_reader,
        )
        reader_source_sha256 = child_claim["execution_source_hashes"][
            "reveal_store"
        ]
        if (
            development_claim["execution_source_hashes"].get("reveal_store")
            != reader_source_sha256
        ):
            raise SecFilingGemmaRevealStoreError(
                "Development-root carry-in source closures differ"
            )
        binding = {
            "carry_in_records": copy.deepcopy(derived_records),
            "carry_in_byte_index": copy.deepcopy(byte_index),
            "carry_in_complete_marker_sha256": hashlib.sha256(
                marker_bytes
            ).hexdigest(),
            "reader_source_sha256": reader_source_sha256,
        }
        return ancestry, binding

    def _read_and_seal_owned_final_carry_in_locked(
        self,
        *,
        authenticated_store_snapshot: Mapping[str, Any],
        independent_current_tip_anchor: Mapping[str, Any],
        request_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Replay a final parent's SEC text and seal only its authorized subset."""

        current = _expect_mapping(
            authenticated_store_snapshot,
            "owned carry-in authenticated store snapshot",
        )
        current_tip = _expect_mapping(
            independent_current_tip_anchor,
            "owned carry-in independent current tip",
        )
        ancestry = self._locate_owned_final_carry_in_ancestry_locked(
            authenticated_store_snapshot=current,
            independent_current_tip_anchor=current_tip,
            request_sha256=request_sha256,
        )
        child_request = ancestry["child_request"]
        child_entry = ancestry["child_entry"]
        child_claim = ancestry["child_claim"]
        child_reader = ancestry["child_reader"]
        parent_request = ancestry["parent_request"]
        parent_bundle = ancestry["parent_bundle"]
        parent_claim = ancestry["parent_claim"]
        parent_reader = ancestry["parent_reader"]
        parent_output = ancestry["parent_output_receipt"]
        self._replay_owned_sec_reader_bytes_locked(
            claim=child_claim,
            reader_receipt=child_reader,
        )
        self._replay_owned_sec_reader_bytes_locked(
            claim=parent_claim,
            reader_receipt=parent_reader,
        )
        parent_evidence, parent_evidence_binding = (
            self._read_owned_stage_evidence_output_locked(
                authenticated_store_snapshot=current,
                independent_current_tip_anchor=current_tip,
                authorization_bundle=parent_bundle,
                request_sha256=parent_request["request_sha256"],
                require_current_store_state=False,
            )
        )
        try:
            expected_parent_output = build_consumed_stage_output_receipt(
                parent_bundle,
                stage_sec_execution_claim=parent_claim,
                stage_sec_reader_receipt=parent_reader,
                **parent_evidence_binding,
            )
        except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in parent output receipt failed durable replay"
            ) from exc
        if parent_output != expected_parent_output:
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in parent output receipt differs from durable replay"
            )

        child_access = _expect_mapping(
            child_entry.get("stage_access_manifest"),
            "owned carry-in child stage-access manifest",
        )
        raw_carry_scope = _expect_mapping(
            child_access.get("prior_same_form_carry_in"),
            "owned prior same-form carry-in scope",
        )
        scope = _expect_mapping(
            child_access.get("scope"),
            "owned carry-in child access scope",
        )
        parent_content = _expect_mapping(
            parent_evidence.get("prerequisite_content_manifest"),
            "owned carry-in parent content manifest",
        )
        parent_universe = _expect_mapping(
            parent_evidence.get("corpus_universe_manifest"),
            "owned carry-in parent corpus universe",
        )
        content_hash = _sha256(
            parent_content.get("content_manifest_sha256"),
            "owned carry-in prerequisite content manifest hash",
        )
        try:
            derived_records = validate_prior_same_form_carry_in_scope(
                child_access,
                corpus_universe_manifest=parent_universe,
                prerequisite_content_manifest=parent_content,
                expected_prerequisite_stage=child_request[
                    "prerequisite_stage"
                ],
                expected_requested_stage=child_request["stage"],
                expected_prerequisite_stage_evidence_sha256=child_request[
                    "prerequisite_stage_evidence_sha256"
                ],
            )
        except (KeyError, SecFilingGemmaStageAccessError, TypeError) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in scope cannot be rederived from durable parent evidence"
            ) from exc
        if (
            not 0 < len(derived_records) <= MAX_PRIOR_SAME_FORM_CARRY_IN_RECORDS
            or raw_carry_scope.get("records") != derived_records
            or scope.get("authorized_stage") != "final"
            or scope.get("prerequisite_stage_data_scope")
            != (
                "sealed_evidence_identity_plus_exact_read_only_prior_same_form_"
                "normalized_text_carry_in"
            )
            or scope.get("general_cross_stage_access_permitted") is not False
            or scope.get("exact_prior_same_form_carry_in_read_permitted")
            is not True
            or scope.get("future_stage_access_permitted") is not False
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in persisted scope differs from its exact durable derivation"
            )

        try:
            _exact_parent_bundle, _parent_grant, parent_component_plan = (
                _sec_component_plan_from_bundle(parent_bundle)
            )
        except SecFilingGemmaStageAuthorizationError as exc:
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in parent SEC plan is no longer exact"
            ) from exc
        parent_documents = parent_component_plan["sec_access_plan"]["documents"]
        parent_document_ordinals = {
            document["accession_number"]: ordinal
            for ordinal, document in enumerate(parent_documents, start=1)
        }
        reader_rows_by_path = {
            row["relative_path"]: row for row in parent_reader["byte_index"]
        }
        if len(parent_document_ordinals) != len(parent_documents):
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in parent SEC plan repeats an accession"
            )
        self._revalidate_authorized_sec_execution_sources(parent_claim)
        self._revalidate_authorized_sec_execution_sources(child_claim)
        parent_component_directory = (
            self.store_directory
            / STAGE_OUTPUTS_DIRECTORY_NAME
            / parent_claim["claim_sha256"]
            / SEC_STAGE_COMPONENT_DIRECTORY_NAME
        )
        secured_parent_component = _secure_directory(
            parent_component_directory,
            create=False,
            location="owned carry-in parent SEC component directory",
        )
        parent_directory_before = _validate_real_directory(
            secured_parent_component,
            "owned carry-in parent SEC component directory",
        )
        expected_parent_names = {
            row["relative_path"] for row in parent_reader["byte_index"]
        } | {SEC_BATCH_COMPLETE_MARKER_FILENAME}
        observed_parent_names_before = [
            item.name for item in secured_parent_component.iterdir()
        ]
        if (
            len(observed_parent_names_before) != len(expected_parent_names)
            or len({name.casefold() for name in observed_parent_names_before})
            != len(observed_parent_names_before)
            or set(observed_parent_names_before) != expected_parent_names
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in parent SEC directory has missing, extra, or case-colliding files"
            )

        selected: list[tuple[dict[str, Any], bytes, str]] = []
        total_bytes = 0
        for carry_ordinal, record in enumerate(derived_records, start=1):
            parent_document_ordinal = parent_document_ordinals.get(
                record["accession_number"]
            )
            if type(parent_document_ordinal) is not int:
                raise SecFilingGemmaRevealStoreError(
                    "Owned carry-in accession is absent from the exact parent SEC plan"
                )
            source_relative_path = (
                f"document-{parent_document_ordinal:04d}.normalized.txt"
            )
            reader_row = reader_rows_by_path.get(source_relative_path)
            if type(reader_row) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Owned carry-in normalized source is absent from the parent reader receipt"
                )
            payload = _read_regular_bytes(
                secured_parent_component / source_relative_path,
                f"owned carry-in parent normalized text {carry_ordinal}",
                max_bytes=MAX_SEC_BATCH_FILE_BYTES,
            )
            try:
                if payload.decode("utf-8").encode("utf-8") != payload:
                    raise UnicodeError
            except UnicodeError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Owned carry-in parent normalized text is not exact UTF-8"
                ) from exc
            total_bytes += len(payload)
            payload_hash = hashlib.sha256(payload).hexdigest()
            if (
                total_bytes > MAX_PRIOR_SAME_FORM_CARRY_IN_TOTAL_BYTES
                or reader_row.get("logical_id")
                != f"document-{parent_document_ordinal:04d}-normalized"
                or reader_row.get("byte_count") != len(payload)
                or reader_row.get("sha256") != payload_hash
                or record["normalized_text_bytes"] != len(payload)
                or record["normalized_text_sha256"] != payload_hash
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned carry-in parent normalized bytes crossed their sealed record"
                )
            destination_relative_path = (
                f"carry-in-{carry_ordinal:04d}.normalized.txt"
            )
            row = {
                "ordinal": carry_ordinal,
                "logical_id": f"carry-in-{carry_ordinal:04d}-normalized",
                "relative_path": destination_relative_path,
                "byte_count": len(payload),
                "sha256": payload_hash,
            }
            selected.append((row, payload, source_relative_path))

        child_claim_directory = (
            self.store_directory
            / STAGE_OUTPUTS_DIRECTORY_NAME
            / child_claim["claim_sha256"]
        )
        secured_child_claim_directory = _secure_directory(
            child_claim_directory,
            create=False,
            location="owned carry-in child claim directory",
        )
        component_directory = (
            secured_child_claim_directory
            / PRIOR_SAME_FORM_CARRY_IN_COMPONENT_DIRECTORY_NAME
        )
        try:
            component_directory.mkdir(exist_ok=False)
            _fsync_directory(secured_child_claim_directory)
        except FileExistsError:
            pass
        secured_component = _secure_directory(
            component_directory,
            create=False,
            location="owned prior same-form carry-in component directory",
        )
        component_directory_before = _validate_real_directory(
            secured_component,
            "owned prior same-form carry-in component directory",
        )
        byte_index = [row for row, _payload, _source in selected]
        marker_body = {
            "schema_version": (
                PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_SCHEMA_VERSION
            ),
            "request_sha256": child_request["request_sha256"],
            "claim_sha256": child_claim["claim_sha256"],
            "sec_reader_receipt_sha256": child_reader["receipt_sha256"],
            "parent_request_sha256": parent_request["request_sha256"],
            "parent_claim_sha256": parent_claim["claim_sha256"],
            "parent_sec_reader_receipt_sha256": parent_reader["receipt_sha256"],
            "parent_stage_output_receipt_sha256": parent_output[
                "output_receipt_sha256"
            ],
            "parent_stage_evidence_sha256": parent_evidence_binding[
                "output_stage_evidence_sha256"
            ],
            "parent_stage_evidence_document_sha256": parent_evidence_binding[
                "output_stage_evidence_document_sha256"
            ],
            "parent_stage_evidence_complete_marker_sha256": (
                parent_evidence_binding[
                    "output_stage_evidence_complete_marker_sha256"
                ]
            ),
            "component_id": PRIOR_SAME_FORM_CARRY_IN_COMPONENT_ID,
            "carry_in_records_sha256": raw_carry_scope["records_sha256"],
            "prerequisite_content_manifest_sha256": content_hash,
            "byte_index": byte_index,
            "byte_index_sha256": canonical_sha256(byte_index),
            "byte_count_total": total_bytes,
        }
        marker = {**marker_body, "marker_sha256": canonical_sha256(marker_body)}
        marker_bytes = _encoded_state(marker)
        expected_names = {
            row["relative_path"] for row, _payload, _source in selected
        } | {PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_FILENAME}
        observed_names_before = [item.name for item in secured_component.iterdir()]
        marker_present = (
            PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_FILENAME
            in observed_names_before
        )
        receipt_present = (
            child_request["request_sha256"]
            in current_tip["stage_carry_in_reader_receipts"]
        )
        if marker_present:
            marker_path = (
                secured_component
                / PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_FILENAME
            )
            structurally_committed_marker = False
            try:
                existing_marker_bytes = _read_regular_bytes(
                    marker_path,
                    "owned carry-in pre-existing complete marker",
                    max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
                )
                existing_marker = _strict_json_bytes(
                    existing_marker_bytes,
                    "owned carry-in pre-existing complete marker",
                )
                if type(existing_marker) is dict:
                    existing_marker_body = {
                        key: existing_marker[key]
                        for key in existing_marker
                        if key != "marker_sha256"
                    }
                    structurally_committed_marker = (
                        set(existing_marker)
                        == _PRIOR_SAME_FORM_CARRY_IN_MARKER_KEYS
                        and existing_marker.get("schema_version")
                        == PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_SCHEMA_VERSION
                        and existing_marker.get("marker_sha256")
                        == canonical_sha256(existing_marker_body)
                        and existing_marker_bytes == _encoded_state(existing_marker)
                    )
            except (KeyError, SecFilingGemmaRevealStoreError):
                structurally_committed_marker = False
            if not structurally_committed_marker:
                if receipt_present:
                    raise SecFilingGemmaRevealStoreError(
                        "Persisted carry-in receipt lost its complete marker"
                    )
                _validate_regular_details(
                    marker_path.lstat(),
                    "interrupted owned carry-in complete marker",
                )
                marker_path.unlink()
                _fsync_directory(secured_component)
                observed_names_before = [
                    name
                    for name in observed_names_before
                    if name != PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_FILENAME
                ]
                marker_present = False
        if (
            len({name.casefold() for name in observed_names_before})
            != len(observed_names_before)
            or not set(observed_names_before).issubset(expected_names)
            or (
                (marker_present or receipt_present)
                and set(observed_names_before) != expected_names
            )
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in component has an extra or case-colliding file"
            )
        recover_interrupted_precommit = not marker_present and not receipt_present
        for row, payload, _source_relative_path in selected:
            _create_or_replay_regular_bytes(
                secured_component / row["relative_path"],
                payload,
                f"owned carry-in normalized text {row['ordinal']}",
                max_bytes=MAX_SEC_BATCH_FILE_BYTES,
                recover_interrupted_precommit=recover_interrupted_precommit,
            )
        _create_or_replay_regular_bytes(
            secured_component / PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_FILENAME,
            marker_bytes,
            "owned carry-in complete marker",
            max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
        )
        replayed_marker = _strict_json_bytes(
            _read_regular_bytes(
                secured_component
                / PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_FILENAME,
                "owned carry-in complete marker replay",
                max_bytes=MAX_TRACKED_ANCHOR_FILE_BYTES,
            ),
            "owned carry-in complete marker replay",
        )
        if (
            type(replayed_marker) is not dict
            or set(replayed_marker) != _PRIOR_SAME_FORM_CARRY_IN_MARKER_KEYS
            or replayed_marker != marker
            or _encoded_state(replayed_marker) != marker_bytes
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in complete marker is not exact canonical ancestry"
            )
        for row, payload, source_relative_path in selected:
            if (
                _read_regular_bytes(
                    secured_parent_component / source_relative_path,
                    f"owned carry-in parent closure text {row['ordinal']}",
                    max_bytes=MAX_SEC_BATCH_FILE_BYTES,
                )
                != payload
                or _read_regular_bytes(
                    secured_component / row["relative_path"],
                    f"owned carry-in child closure text {row['ordinal']}",
                    max_bytes=MAX_SEC_BATCH_FILE_BYTES,
                )
                != payload
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned carry-in bytes changed during closure replay"
                )
        parent_directory_after = _validate_real_directory(
            secured_parent_component,
            "owned carry-in parent SEC component directory",
        )
        component_directory_after = _validate_real_directory(
            secured_component,
            "owned prior same-form carry-in component directory",
        )
        observed_parent_names_after = [
            item.name for item in secured_parent_component.iterdir()
        ]
        observed_names_after = [item.name for item in secured_component.iterdir()]
        if (
            (parent_directory_before.st_dev, parent_directory_before.st_ino)
            != (parent_directory_after.st_dev, parent_directory_after.st_ino)
            or (
                component_directory_before.st_dev,
                component_directory_before.st_ino,
            )
            != (
                component_directory_after.st_dev,
                component_directory_after.st_ino,
            )
            or len(observed_parent_names_after) != len(expected_parent_names)
            or len({name.casefold() for name in observed_parent_names_after})
            != len(observed_parent_names_after)
            or set(observed_parent_names_after) != expected_parent_names
            or len(observed_names_after) != len(expected_names)
            or len({name.casefold() for name in observed_names_after})
            != len(observed_names_after)
            or set(observed_names_after) != expected_names
        ):
            raise SecFilingGemmaRevealStoreError(
                "Owned carry-in directory changed during closure replay"
            )
        self._revalidate_authorized_sec_execution_sources(parent_claim)
        self._revalidate_authorized_sec_execution_sources(child_claim)
        self._replay_owned_sec_reader_bytes_locked(
            claim=child_claim,
            reader_receipt=child_reader,
        )
        self._replay_owned_sec_reader_bytes_locked(
            claim=parent_claim,
            reader_receipt=parent_reader,
        )
        binding = {
            "carry_in_records": copy.deepcopy(derived_records),
            "carry_in_byte_index": copy.deepcopy(byte_index),
            "carry_in_complete_marker_sha256": hashlib.sha256(
                marker_bytes
            ).hexdigest(),
            "reader_source_sha256": child_claim["execution_source_hashes"][
                "reveal_store"
            ],
            "parent_stage_evidence_document_sha256": parent_evidence_binding[
                "output_stage_evidence_document_sha256"
            ],
            "parent_stage_evidence_complete_marker_sha256": (
                parent_evidence_binding[
                    "output_stage_evidence_complete_marker_sha256"
                ]
            ),
        }
        return ancestry, binding

    def _locate_owned_parent_intermediate_predecessor_locked(
        self,
        *,
        authenticated_store_snapshot: Mapping[str, Any],
        independent_current_tip_anchor: Mapping[str, Any],
        child_request: Mapping[str, Any],
        prerequisite_stage_evidence: Mapping[str, Any],
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Locate a parent and replace caller evidence with its durable replay."""

        entry, bundle, output_receipt = _locate_parent_intermediate_predecessor(
            authenticated_store_snapshot=authenticated_store_snapshot,
            independent_current_tip_anchor=independent_current_tip_anchor,
            child_request=child_request,
        )
        parent_request = _expect_mapping(
            entry.get("request"),
            "owned parent intermediate request",
        )
        durable_evidence, output_binding = (
            self._read_owned_stage_evidence_output_locked(
                authenticated_store_snapshot=authenticated_store_snapshot,
                independent_current_tip_anchor=independent_current_tip_anchor,
                authorization_bundle=bundle,
                request_sha256=parent_request["request_sha256"],
            )
        )
        caller_evidence = _json_value_copy(
            dict(
                _expect_mapping(
                    prerequisite_stage_evidence,
                    "caller parent stage evidence",
                )
            ),
            "caller parent stage evidence",
        )
        if (
            caller_evidence != durable_evidence
            or output_binding["output_stage_evidence_sha256"]
            != child_request["prerequisite_stage_evidence_sha256"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Final request prerequisite differs from the durable parent output"
            )
        try:
            validate_consumed_stage_output_receipt(
                output_receipt,
                authenticated_store_snapshot=authenticated_store_snapshot,
                independent_current_tip_anchor=independent_current_tip_anchor,
                authorization_bundle=bundle,
                **output_binding,
            )
        except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
            raise SecFilingGemmaRevealStoreError(
                "Final request predecessor output receipt is not exact at the current tip"
            ) from exc
        return (
            entry,
            bundle,
            output_receipt,
            durable_evidence,
            output_binding,
        )

    def compare_and_swap_append(
        self,
        *,
        transition: Mapping[str, Any],
        appended_registry: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically apply exactly one child registry transition.

        The transition's expected prior pin must equal the independently loaded
        authoritative pin.  A stale, forked, skipped, or rollback transition is
        rejected without changing the store.
        """

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(anchor)
            )
            # Authenticate and capture both files before touching caller data.
            if type(transition) is not dict or type(appended_registry) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Registry CAS inputs must be exact built-in dicts"
                )
            try:
                detached_cas = detach_untrusted_stage_json(
                    {
                        "transition": transition,
                        "appended_registry": appended_registry,
                    },
                    "registry CAS caller bundle",
                )
            except Exception as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Registry CAS inputs exceed fixed allocation bounds"
                ) from exc
            transition_value = detached_cas["transition"]
            appended_value = detached_cas["appended_registry"]
            try:
                validate_registry_pin_transition(
                    transition_value,
                    prior_registry=current["latest_registry"],
                    external_prior_pin=current["latest_registry_pin"],
                    appended_registry=appended_value,
                )
                next_pin = derive_registry_pin(appended_value)
            except SecFilingGemmaRevealRegistryError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Registry compare-and-swap rejected a stale, forked, or rollback transition"
                ) from exc
            next_state = _state_snapshot(
                anchor=anchor,
                latest_registry=appended_value,
                latest_pin=next_pin,
                consumption_ledger=current["consumption_ledger"],
            )
            committed, _next_tip = self._commit_state_and_tip_locked(
                tracked_anchor=anchor,
                prior_tip_anchor=current_tip,
                next_state=next_state,
            )
            return committed

    def consume_request(
        self,
        request: Mapping[str, Any],
        *,
        candidate_manifest: Mapping[str, Any],
        stage: str,
        stage_access_manifest: Mapping[str, Any],
        prerequisite_stage_evidence: Mapping[str, Any],
        _issue_authorization_grant: bool = False,
    ) -> dict[str, Any]:
        """Validate and atomically consume one non-authorizing stage request.

        The fixed candidate-bound verifier executes while the exclusive lock is
        held. It receives bounded detached copies of the prerequisite evidence,
        the actual self-hashed stage-access manifest, and the expected request
        context plus a reveal-store-authenticated trusted-content pin and, for
        final requests, the exact prior intermediate consumption/grant binding.
        It must return :class:`SemanticPrerequisiteValidation`.  Failure,
        replay, verifier mutation of the state file, or any binding mismatch
        leaves no consumption entry. A successfully precommitted trusted pin is
        deliberately retained across verifier failure and reused on an exact
        retry; it cannot authorize access by itself.

        There is deliberately no caller-supplied validator parameter. Until
        the fixed verifier can complete every frozen check, it raises and this
        method cannot consume a request.
        """
        with self._locked():
            locked_repository_root = self._repository_root
            locked_store_directory = self._store_directory
            locked_state_path = self.state_path
            locked_tip_anchor_path = self.current_tip_anchor_path
            locked_restore_pending_path = self.restore_pending_path
            locked_lock_path = self.lock_path
            _cleanup_interrupted_temporaries(self.store_directory)
            anchor = _load_tracked_anchor(self.repository_root)
            (
                current,
                current_tip,
                original_state_bytes,
                original_tip_anchor_bytes,
            ) = self._read_state_and_tip_locked(anchor)
            # Both independent files are authenticated and captured before any
            # authorization-critical caller object is examined.
            if type(_issue_authorization_grant) is not bool:
                raise SecFilingGemmaRevealStoreError(
                    "Internal grant-issuance flag must be an exact boolean"
                )
            caller_objects = {
                "request": request,
                "candidate_manifest": candidate_manifest,
                "stage_access_manifest": stage_access_manifest,
                "prerequisite_stage_evidence": prerequisite_stage_evidence,
            }
            if any(type(item) is not dict for item in caller_objects.values()):
                raise SecFilingGemmaRevealStoreError(
                    "Authorization-critical inputs must be exact built-in dicts"
                )
            # One shared no-copy budget rejects an already-oversized bundle
            # before allocation.  The bounded detacher then rechecks every
            # limit while copying, so concurrent mutation cannot insert an
            # unchecked deep or oversized value between check and copy.
            try:
                preflight_untrusted_stage_json(
                    caller_objects, "reveal request caller bundle"
                )
                detached_objects = detach_untrusted_stage_json(
                    caller_objects, "reveal request caller bundle"
                )
            except Exception as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Reveal request inputs exceed the fixed verifier allocation bounds"
                ) from exc
            if type(detached_objects) is not dict:  # pragma: no cover - fixed wrapper
                raise SecFilingGemmaRevealStoreError(
                    "Reveal request inputs could not be detached"
                )
            request_value = detached_objects["request"]
            candidate_value = detached_objects["candidate_manifest"]
            if type(stage) is not str:
                raise SecFilingGemmaRevealStoreError(
                    "requested stage must be an exact built-in string"
                )
            access_manifest = detached_objects["stage_access_manifest"]
            access_hash = _stage_access_manifest_sha256(access_manifest)
            evidence = detached_objects["prerequisite_stage_evidence"]
            evidence_hash = _stage_evidence_sha256(evidence)
            try:
                request_hash = validate_single_candidate_reveal_request(
                    request_value,
                    registry=current["latest_registry"],
                    external_pin=current["latest_registry_pin"],
                    candidate_manifest=candidate_value,
                    stage=stage,
                    stage_access_manifest_sha256=access_hash,
                    prerequisite_stage_evidence_sha256=evidence_hash,
                )
            except SecFilingGemmaRevealRegistryError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Reveal request is altered, stale, non-authorizing, or not current-tip bound"
                ) from exc
            ledger = current["consumption_ledger"]
            matching_entries = [
                entry
                for entry in ledger["entries"]
                if (
                entry["request_sha256"] == request_hash
                or (
                    entry["attempt_id"] == request_value["attempt_id"]
                    and entry["stage"] == stage
                )
                )
            ]
            if matching_entries:
                if _issue_authorization_grant:
                    existing_entry = matching_entries[0]
                    existing_bundle = current_tip["authorization_bundles"].get(
                        request_hash
                    )
                    if (
                        existing_entry["request_sha256"] == request_hash
                        and existing_entry["entry_sha256"]
                        == ledger["chain"]["tip_sha256"]
                        and existing_bundle is not None
                        and existing_bundle["authenticated_store_snapshot"]
                        == current
                    ):
                        try:
                            validate_consumed_stage_authorization_grant(
                                existing_bundle["authorization_grant"],
                                authenticated_store_snapshot=current,
                                external_store_state_pin=existing_bundle[
                                    "store_state_pin"
                                ],
                                independent_current_tip_anchor=current_tip,
                                expected_consumption_entry_sha256=existing_entry[
                                    "entry_sha256"
                                ],
                                expected_request_sha256=request_hash,
                                expected_candidate_sha256=request_value[
                                    "candidate_sha256"
                                ],
                                expected_stage=stage,
                                expected_prerequisite_stage_evidence_sha256=(
                                    evidence_hash
                                ),
                                expected_stage_access_manifest_sha256=access_hash,
                                expected_output_namespace=access_manifest["output"][
                                    "namespace"
                                ],
                            )
                        except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
                            raise SecFilingGemmaRevealStoreError(
                                "Persisted retry bundle failed current-tip authentication"
                            ) from exc
                        return copy.deepcopy(existing_bundle)
                raise SecFilingGemmaRevealStoreError(
                    "Reveal request or candidate stage was already consumed"
                )
            parent_entry: dict[str, Any] | None = None
            parent_bundle: dict[str, Any] | None = None
            parent_output_receipt: dict[str, Any] | None = None
            parent_durable_evidence: dict[str, Any] | None = None
            parent_output_binding: dict[str, Any] | None = None
            if stage == "final":
                # Perform this exact predecessor/grant check before the monotonic
                # final-request pin precommit. A missing or ungranted parent must
                # leave no final-stage authorization material behind.
                (
                    parent_entry,
                    parent_bundle,
                    parent_output_receipt,
                    parent_durable_evidence,
                    parent_output_binding,
                ) = self._locate_owned_parent_intermediate_predecessor_locked(
                    authenticated_store_snapshot=current,
                    independent_current_tip_anchor=current_tip,
                    child_request=request_value,
                    prerequisite_stage_evidence=evidence,
                )

            try:
                trusted_content_pin = derive_reveal_store_trusted_stage_content_pin(
                    current,
                    reveal_request=request_value,
                    stage_access_manifest=access_manifest,
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Reveal store could not derive the exact trusted stage-content pin"
                ) from exc
            persisted_pin = current_tip["trusted_stage_content_pins"].get(
                request_hash
            )
            if persisted_pin is None:
                try:
                    current, current_tip = self._commit_state_and_tip_locked(
                        tracked_anchor=anchor,
                        prior_tip_anchor=current_tip,
                        next_state=current,
                        trusted_stage_content_pin=trusted_content_pin,
                    )
                except SecFilingGemmaStageAuthorizationError as exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Trusted stage-content pin precommit failed closed"
                    ) from exc
                (
                    current,
                    current_tip,
                    original_state_bytes,
                    original_tip_anchor_bytes,
                ) = self._read_state_and_tip_locked(anchor)
                persisted_pin = current_tip["trusted_stage_content_pins"].get(
                    request_hash
                )
            if persisted_pin != trusted_content_pin:
                raise SecFilingGemmaRevealStoreError(
                    "Current tip contains another trusted pin for this request"
                )
            try:
                trusted_content_authentication = (
                    authenticate_reveal_store_trusted_stage_content_pin(
                        current,
                        current_tip,
                        reveal_request=request_value,
                        stage_access_manifest=access_manifest,
                    )
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Persisted trusted stage-content pin failed current-tip authentication"
                ) from exc
            parent_consumption_binding: dict[str, Any] | None = None
            if stage == "final":
                # Reload the exact parent objects from the post-pin current tip.
                # The pin-only revision is monotonic and state-preserving, but
                # this avoids carrying any pre-transition authorization object
                # into the verifier context.
                (
                    parent_entry,
                    parent_bundle,
                    parent_output_receipt,
                    parent_durable_evidence,
                    parent_output_binding,
                ) = self._locate_owned_parent_intermediate_predecessor_locked(
                    authenticated_store_snapshot=current,
                    independent_current_tip_anchor=current_tip,
                    child_request=request_value,
                    prerequisite_stage_evidence=evidence,
                )
                if parent_durable_evidence is None or parent_output_binding is None:
                    raise SecFilingGemmaRevealStoreError(
                        "Final request lost its durable parent evidence"
                    )
                evidence = parent_durable_evidence
                parent_consumption_binding = _build_parent_consumption_binding(
                    authenticated_store_snapshot=current,
                    independent_current_tip_anchor=current_tip,
                    child_request=request_value,
                    parent_entry=parent_entry,
                    parent_bundle=parent_bundle,
                    parent_output_receipt=parent_output_receipt,
                    prerequisite_stage_evidence=evidence,
                    parent_output_binding=parent_output_binding,
                )
            authenticated_store_context = _authenticated_store_verifier_context(
                trusted_stage_content_pin=trusted_content_pin,
                trusted_stage_content_authentication=trusted_content_authentication,
                parent_consumption_binding=parent_consumption_binding,
            )

            context_dict = _expected_context_for_consumed_request(
                request_value,
                trusted_stage_content_pin_sha256=trusted_content_pin["pin_sha256"],
                trusted_stage_content_authentication_receipt_sha256=(
                    trusted_content_authentication["authentication_receipt_sha256"]
                ),
                parent_consumption_binding_sha256=(
                    None
                    if parent_consumption_binding is None
                    else parent_consumption_binding[
                        "parent_consumption_binding_sha256"
                    ]
                ),
                authenticated_store_context_sha256=authenticated_store_context[
                    "authenticated_store_context_sha256"
                ],
            )
            context = MappingProxyType(context_dict)
            promotion_gate_before = AUTHORITATIVE_STAGE_PROMOTION_ENABLED
            restore_document = _restore_pending_document(
                state_bytes=original_state_bytes,
                tip_anchor_bytes=original_tip_anchor_bytes,
            )
            restore_document_bytes = _encoded_state(restore_document)
            # Persist the authenticated original pair before effectful verifier
            # code runs. A hard process stop after any verifier-side mutation
            # is therefore repaired on the next locked load.
            _atomic_replace(
                locked_restore_pending_path,
                restore_document_bytes,
            )
            try:
                try:
                    verifier_inputs = detach_untrusted_stage_json(
                        {
                            "evidence": evidence,
                            "stage_access_manifest": access_manifest,
                            "expected_context": context_dict,
                            "authenticated_store_context": authenticated_store_context,
                        },
                        "fixed verifier input bundle",
                    )
                    try:
                        result = authoritative_prerequisite_validator(
                            verifier_inputs["evidence"],
                            verifier_inputs["stage_access_manifest"],
                            verifier_inputs["expected_context"],
                            authenticated_store_context=verifier_inputs[
                                "authenticated_store_context"
                            ],
                        )
                    finally:
                        promotion_gate_changed = (
                            AUTHORITATIVE_STAGE_PROMOTION_ENABLED
                            is not promotion_gate_before
                        )
                        globals()["AUTHORITATIVE_STAGE_PROMOTION_ENABLED"] = (
                            promotion_gate_before
                        )
                except SecFilingGemmaRevealStoreError:
                    raise
                except Exception as exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Fixed semantic prerequisite verifier failed; request was not consumed"
                    ) from exc
                if (
                    self._repository_root != locked_repository_root
                    or self._store_directory != locked_store_directory
                    or self.state_path != locked_state_path
                    or self.current_tip_anchor_path != locked_tip_anchor_path
                    or self.restore_pending_path != locked_restore_pending_path
                    or self.lock_path != locked_lock_path
                ):
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store paths changed during prerequisite validation"
                    )
                validation = _validate_semantic_result(
                    result,
                    expected_context=context,
                    expected_validator_id=AUTHORITATIVE_VALIDATOR_ID,
                    expected_validator_source_sha256=candidate_value["bindings"][
                        "source_hashes"
                    ]["stage_verifier"],
                )
                if promotion_gate_before is not True:
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store stage promotion remains independently disabled"
                    )
                if promotion_gate_changed:
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store stage promotion gate changed during validation"
                    )
                if (
                    self._repository_root != locked_repository_root
                    or self._store_directory != locked_store_directory
                    or self.state_path != locked_state_path
                    or self.current_tip_anchor_path != locked_tip_anchor_path
                    or self.restore_pending_path != locked_restore_pending_path
                    or self.lock_path != locked_lock_path
                ):
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store paths changed while validating the verifier result"
                    )
                # The verifier is effectful Python. Check exact bytes before
                # parsing so even a rehashed or differently encoded mutation is
                # rejected rather than silently replaced by the next state.
                try:
                    (
                        after_verifier,
                        after_verifier_tip,
                        after_verifier_bytes,
                        after_verifier_tip_bytes,
                    ) = self._read_state_and_tip_locked(
                        anchor,
                        recover_pending_restore=False,
                    )
                except SecFilingGemmaRevealStoreError as exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store state was damaged during prerequisite "
                        "validation; the prior authenticated state will be restored"
                    ) from exc
                if (
                    after_verifier_bytes != original_state_bytes
                    or after_verifier_tip_bytes != original_tip_anchor_bytes
                    or after_verifier != current
                    or after_verifier_tip != current_tip
                ):
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store state changed during prerequisite validation; "
                        "the prior authenticated state will be restored"
                    )
                persisted_restore = _read_regular_bytes(
                    locked_restore_pending_path,
                    "pre-verifier restore transaction",
                    max_bytes=MAX_RESTORE_PENDING_FILE_BYTES,
                )
                if persisted_restore != restore_document_bytes:
                    raise SecFilingGemmaRevealStoreError(
                        "Pre-verifier restore transaction changed during validation"
                    )
                locked_restore_pending_path.unlink()
                if (
                    locked_restore_pending_path.exists()
                    or locked_restore_pending_path.is_symlink()
                ):
                    raise SecFilingGemmaRevealStoreError(
                        "Pre-verifier restore transaction could not be retired"
                    )
            except BaseException:
                # Cleanup has finally-like semantics: verifier exceptions,
                # invalid semantic results, path redirection, and state-damage
                # errors cannot escape while mutated authoritative bytes remain.
                self._repository_root = locked_repository_root
                self._store_directory = locked_store_directory
                try:
                    restore_document = _restore_pending_document(
                        state_bytes=original_state_bytes,
                        tip_anchor_bytes=original_tip_anchor_bytes,
                    )
                    _atomic_replace(
                        locked_restore_pending_path,
                        _encoded_state(restore_document),
                    )
                    self._recover_pending_restore_locked(anchor)
                    restored_state_bytes = _read_regular_bytes(
                        locked_state_path,
                        "restored authoritative reveal-store state",
                        max_bytes=MAX_STATE_FILE_BYTES,
                    )
                    restored_tip_bytes = _read_regular_bytes(
                        locked_tip_anchor_path,
                        "restored independent current-tip anchor",
                        max_bytes=MAX_CURRENT_TIP_ANCHOR_FILE_BYTES,
                    )
                    if (
                        restored_state_bytes != original_state_bytes
                        or restored_tip_bytes != original_tip_anchor_bytes
                    ):
                        raise SecFilingGemmaRevealStoreError(
                            "Restored reveal-store bytes do not match the authenticated original"
                        )
                except BaseException as restore_exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store state could not be restored after prerequisite validation"
                    ) from restore_exc
                raise

            entries = copy.deepcopy(ledger["entries"])
            final_delta = 1 if stage == "final" else 0
            cumulative_final = (
                ledger["chain"]["actual_final_touch_count"] + final_delta
            )
            body = {
                "schema_version": CONSUMPTION_ENTRY_SCHEMA_VERSION,
                "sequence": len(entries) + 1,
                "request_sha256": request_hash,
                "request": request_value,
                "stage_access_manifest": access_manifest,
                "stage": stage,
                "attempt_id": request_value["attempt_id"],
                "candidate_sha256": request_value["candidate_sha256"],
                "registry_entry_sha256": request_value["registry_entry_sha256"],
                "prerequisite_validation": validation,
                "prior_tip_sha256": ledger["chain"]["tip_sha256"],
                "final_touch_delta": final_delta,
                "cumulative_actual_final_touch_count": cumulative_final,
            }
            entry = {**body, "entry_sha256": canonical_sha256(body)}
            entries.append(entry)
            next_ledger = _consumption_ledger(
                entries, tip_sha256=entry["entry_sha256"], anchor=anchor
            )
            next_state = _state_snapshot(
                anchor=anchor,
                latest_registry=current["latest_registry"],
                latest_pin=current["latest_registry_pin"],
                consumption_ledger=next_ledger,
            )
            bundle: dict[str, Any] | None = None
            if _issue_authorization_grant:
                try:
                    bundle = _authorization_bundle(
                        authenticated_store_snapshot=next_state,
                        expected_new_consumption_entry_sha256=entry[
                            "entry_sha256"
                        ],
                    )
                except SecFilingGemmaStageAuthorizationError as exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Authenticated consumption could not produce an exact authorization grant"
                    ) from exc

            # This transaction is intentionally not rolled back in an outer
            # exception handler.  If the process stops after the state replace,
            # the pending CAS anchor makes the exact state+bundle recoverable;
            # retrying the same request returns that stored bundle rather than
            # consuming a second entry.
            try:
                authenticated_next_state, authenticated_next_tip = (
                    self._commit_state_and_tip_locked(
                        tracked_anchor=anchor,
                        prior_tip_anchor=current_tip,
                        next_state=next_state,
                        authorization_bundle=bundle,
                    )
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "State/current-tip authorization transaction failed closed"
                ) from exc
            if bundle is None:
                return authenticated_next_state
            persisted = authenticated_next_tip["authorization_bundles"].get(
                request_hash
            )
            if persisted != bundle:
                raise SecFilingGemmaRevealStoreError(
                    "Committed current tip did not preserve the exact authorization bundle"
                )
            try:
                validate_consumed_stage_authorization_grant(
                    persisted["authorization_grant"],
                    authenticated_store_snapshot=authenticated_next_state,
                    external_store_state_pin=persisted["store_state_pin"],
                    independent_current_tip_anchor=authenticated_next_tip,
                    expected_consumption_entry_sha256=entry["entry_sha256"],
                    expected_request_sha256=request_hash,
                    expected_candidate_sha256=request_value["candidate_sha256"],
                    expected_stage=stage,
                    expected_prerequisite_stage_evidence_sha256=evidence_hash,
                    expected_stage_access_manifest_sha256=access_hash,
                    expected_output_namespace=access_manifest["output"]["namespace"],
                )
            except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Committed authorization bundle failed independent current-tip validation"
                ) from exc
            return copy.deepcopy(persisted)

    def _record_owned_development_root_carry_in_reader_output(
        self,
        *,
        request_sha256: str,
    ) -> dict[str, Any]:
        """Copy and receipt the intermediate child's exact root carry-in."""

        request_hash = _sha256(
            request_sha256,
            "owned development-root carry-in reader request hash",
        )
        root_scope_hash = (
            self._owned_intermediate_development_root_scope_sha256(
                request_sha256=request_hash
            )
        )
        replayed_development_reader = (
            self._record_owned_development_sec_root_reader_output(
                development_root_scope_sha256=root_scope_hash
            )
        )
        replayed_child_reader = self._record_authorized_sec_stage_reader_output(
            request_sha256=request_hash,
        )
        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            ancestry, binding = (
                self._read_and_seal_owned_development_root_carry_in_locked(
                    authenticated_store_snapshot=current,
                    independent_current_tip_anchor=current_tip,
                    request_sha256=request_hash,
                )
            )
            if (
                ancestry["child_reader"] != replayed_child_reader
                or ancestry["development_reader"]
                != replayed_development_reader
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Development-root carry-in ancestry changed after reader replay"
                )
            try:
                receipt = build_development_root_carry_in_reader_receipt(
                    ancestry["child_bundle"],
                    stage_sec_execution_claim=ancestry["child_claim"],
                    stage_sec_reader_receipt=ancestry["child_reader"],
                    development_sec_execution_claim=ancestry[
                        "development_claim"
                    ],
                    development_sec_reader_receipt=ancestry[
                        "development_reader"
                    ],
                    carry_in_byte_index=binding["carry_in_byte_index"],
                    carry_in_complete_marker_sha256=binding[
                        "carry_in_complete_marker_sha256"
                    ],
                    reader_source_sha256=binding["reader_source_sha256"],
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Development-root carry-in output is not exact authorized ancestry"
                ) from exc
            closure_ancestry, closure_binding = (
                self._read_and_seal_owned_development_root_carry_in_locked(
                    authenticated_store_snapshot=current,
                    independent_current_tip_anchor=current_tip,
                    request_sha256=request_hash,
                )
            )
            if closure_ancestry != ancestry or closure_binding != binding:
                raise SecFilingGemmaRevealStoreError(
                    "Development-root carry-in ancestry changed before receipt CAS"
                )
            existing = current_tip[
                "development_root_carry_in_reader_receipts"
            ].get(request_hash)
            if existing is not None:
                if existing != receipt:
                    raise SecFilingGemmaRevealStoreError(
                        "Intermediate child already has another root carry-in receipt"
                    )
                try:
                    validate_development_root_carry_in_reader_receipt(
                        existing,
                        authenticated_store_snapshot=current,
                        independent_current_tip_anchor=current_tip,
                        carry_in_byte_index=binding["carry_in_byte_index"],
                        carry_in_complete_marker_sha256=binding[
                            "carry_in_complete_marker_sha256"
                        ],
                        reader_source_sha256=binding["reader_source_sha256"],
                    )
                except SecFilingGemmaStageAuthorizationError as exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Persisted development-root carry-in receipt failed replay"
                    ) from exc
                return copy.deepcopy(existing)
            committed_state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                development_root_carry_in_reader_receipt=receipt,
            )
            persisted = committed_tip[
                "development_root_carry_in_reader_receipts"
            ].get(request_hash)
            try:
                validate_development_root_carry_in_reader_receipt(
                    persisted,
                    authenticated_store_snapshot=committed_state,
                    independent_current_tip_anchor=committed_tip,
                    carry_in_byte_index=binding["carry_in_byte_index"],
                    carry_in_complete_marker_sha256=binding[
                        "carry_in_complete_marker_sha256"
                    ],
                    reader_source_sha256=binding["reader_source_sha256"],
                )
            except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Committed development-root carry-in receipt failed validation"
                ) from exc
            return copy.deepcopy(persisted)

    def _record_owned_stage_carry_in_reader_output(
        self,
        *,
        request_sha256: str,
    ) -> dict[str, Any]:
        """Copy and receipt the exact final child's authorized parent texts."""

        request_hash = _sha256(
            request_sha256,
            "owned carry-in reader request hash",
        )
        # Reject a non-final or non-lineal request read-only before either SEC
        # replayer can append a terminal reader receipt.
        parent_request_hash = self._owned_final_carry_in_parent_request_sha256(
            request_sha256=request_hash,
        )
        # Both SEC replayers own their own transactions and locks.  Replaying
        # them here avoids recursive locking while ensuring that finalization
        # starts from exact terminal reader receipts, never caller ancestry.
        replayed_child_reader = self._record_authorized_sec_stage_reader_output(
            request_sha256=request_hash,
        )
        replayed_parent_reader = self._record_authorized_sec_stage_reader_output(
            request_sha256=parent_request_hash,
        )
        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            ancestry, binding = self._read_and_seal_owned_final_carry_in_locked(
                authenticated_store_snapshot=current,
                independent_current_tip_anchor=current_tip,
                request_sha256=request_hash,
            )
            if (
                ancestry["child_reader"] != replayed_child_reader
                or ancestry["parent_reader"] != replayed_parent_reader
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned carry-in reader ancestry changed after SEC replay"
                )
            try:
                receipt = build_stage_carry_in_reader_receipt(
                    ancestry["child_bundle"],
                    stage_sec_execution_claim=ancestry["child_claim"],
                    stage_sec_reader_receipt=ancestry["child_reader"],
                    parent_authorization_bundle=ancestry["parent_bundle"],
                    parent_stage_sec_execution_claim=ancestry["parent_claim"],
                    parent_stage_sec_reader_receipt=ancestry["parent_reader"],
                    parent_consumed_stage_output_receipt=ancestry[
                        "parent_output_receipt"
                    ],
                    carry_in_byte_index=binding["carry_in_byte_index"],
                    carry_in_complete_marker_sha256=binding[
                        "carry_in_complete_marker_sha256"
                    ],
                    reader_source_sha256=binding["reader_source_sha256"],
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Owned carry-in reader output is not exact authorized ancestry"
                ) from exc
            closure_ancestry, closure_binding = (
                self._read_and_seal_owned_final_carry_in_locked(
                    authenticated_store_snapshot=current,
                    independent_current_tip_anchor=current_tip,
                    request_sha256=request_hash,
                )
            )
            if closure_ancestry != ancestry or closure_binding != binding:
                raise SecFilingGemmaRevealStoreError(
                    "Owned carry-in ancestry changed before receipt CAS"
                )
            existing = current_tip["stage_carry_in_reader_receipts"].get(
                request_hash
            )
            if existing is not None:
                if existing != receipt:
                    raise SecFilingGemmaRevealStoreError(
                        "Final child already has a different carry-in reader receipt"
                    )
                try:
                    validate_stage_carry_in_reader_receipt(
                        existing,
                        authenticated_store_snapshot=current,
                        independent_current_tip_anchor=current_tip,
                        carry_in_byte_index=binding["carry_in_byte_index"],
                        carry_in_complete_marker_sha256=binding[
                            "carry_in_complete_marker_sha256"
                        ],
                        reader_source_sha256=binding["reader_source_sha256"],
                    )
                except SecFilingGemmaStageAuthorizationError as exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Persisted carry-in reader receipt failed durable replay"
                    ) from exc
                return copy.deepcopy(existing)
            committed_state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                stage_carry_in_reader_receipt=receipt,
            )
            persisted = committed_tip["stage_carry_in_reader_receipts"].get(
                request_hash
            )
            try:
                validate_stage_carry_in_reader_receipt(
                    persisted,
                    authenticated_store_snapshot=committed_state,
                    independent_current_tip_anchor=committed_tip,
                    carry_in_byte_index=binding["carry_in_byte_index"],
                    carry_in_complete_marker_sha256=binding[
                        "carry_in_complete_marker_sha256"
                    ],
                    reader_source_sha256=binding["reader_source_sha256"],
                )
            except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Committed carry-in reader receipt failed durable validation"
                ) from exc
            return copy.deepcopy(persisted)

    def _record_owned_stage_evidence_output(
        self,
        *,
        request_sha256: str,
    ) -> dict[str, Any]:
        """Replay and persist one fixed-path, store-recomputed stage output."""

        # Finalize or replay the durable SEC batch before taking this method's
        # lock.  That helper owns its own transaction and must never be invoked
        # recursively under the same non-reentrant store lock.
        replayed_sec_reader = self._record_authorized_sec_stage_reader_output(
            request_sha256=request_sha256,
        )
        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            request_hash = _sha256(
                request_sha256,
                "owned stage-evidence output request hash",
            )
            bundle = current_tip["authorization_bundles"].get(request_hash)
            claim = current_tip["stage_sec_execution_claims"].get(request_hash)
            sec_reader = current_tip["stage_sec_reader_receipts"].get(
                request_hash
            )
            if (
                type(bundle) is not dict
                or type(claim) is not dict
                or type(sec_reader) is not dict
                or sec_reader != replayed_sec_reader
                or request_hash in current_tip["stage_sec_execution_aborts"]
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Owned stage evidence lacks exact terminal SEC ancestry"
                )
            _durable_evidence, binding = self._read_owned_stage_evidence_output_locked(
                authenticated_store_snapshot=current,
                independent_current_tip_anchor=current_tip,
                authorization_bundle=bundle,
                request_sha256=request_hash,
            )
            try:
                receipt = build_consumed_stage_output_receipt(
                    bundle,
                    stage_sec_execution_claim=claim,
                    stage_sec_reader_receipt=sec_reader,
                    **binding,
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Owned stage evidence is not authorized by its exact SEC ancestry"
                ) from exc
            existing = current_tip["consumed_stage_output_receipts"].get(
                request_hash
            )
            if existing is not None:
                if existing != receipt:
                    raise SecFilingGemmaRevealStoreError(
                        "Consumed grant already has a different durable first output"
                    )
                try:
                    validate_consumed_stage_output_receipt(
                        existing,
                        authenticated_store_snapshot=current,
                        independent_current_tip_anchor=current_tip,
                        authorization_bundle=bundle,
                        **binding,
                    )
                except SecFilingGemmaStageAuthorizationError as exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Persisted durable stage-output receipt is invalid"
                    ) from exc
                return copy.deepcopy(existing)
            committed_state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                consumed_stage_output_receipt=receipt,
            )
            persisted = committed_tip["consumed_stage_output_receipts"].get(
                request_hash
            )
            try:
                validate_consumed_stage_output_receipt(
                    persisted,
                    authenticated_store_snapshot=committed_state,
                    independent_current_tip_anchor=committed_tip,
                    authorization_bundle=bundle,
                    **binding,
                )
            except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Committed durable stage-output receipt failed validation"
                ) from exc
            return copy.deepcopy(persisted)

    def consume_request_and_issue_authorization_grant(
        self,
        request: Mapping[str, Any],
        *,
        candidate_manifest: Mapping[str, Any],
        stage: str,
        stage_access_manifest: Mapping[str, Any],
        prerequisite_stage_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Consume one request and return its exact post-consumption grant bundle.

        The grant is derived for the newly appended ledger tip, and its exact
        bundle is committed in the separate monotonic current-tip anchor as the
        same recoverable transaction as the consumed state.  The bundled pin
        is not a trust root: a downstream API must independently load the
        store's current-tip anchor and pass it to grant validation.  The fixed
        production verifier still raises, so production cannot reach grant
        issuance while any prerequisite check remains unsupported.
        """

        return self.consume_request(
            request,
            candidate_manifest=candidate_manifest,
            stage=stage,
            stage_access_manifest=stage_access_manifest,
            prerequisite_stage_evidence=prerequisite_stage_evidence,
            _issue_authorization_grant=True,
        )


__all__ = [
    "AUTHORITATIVE_STAGE_PROMOTION_ENABLED",
    "AUTHORITATIVE_VALIDATOR_ID",
    "CONSUMPTION_ENTRY_SCHEMA_VERSION",
    "CONSUMPTION_LEDGER_SCHEMA_VERSION",
    "CURRENT_TIP_ANCHOR_FILENAME",
    "CURRENT_TIP_PENDING_SCHEMA_VERSION",
    "INITIAL_PIN_RELATIVE_PATH",
    "LOCK_FILENAME",
    "MAX_CURRENT_TIP_ANCHOR_FILE_BYTES",
    "MAX_RESTORE_PENDING_FILE_BYTES",
    "MAX_STATE_FILE_BYTES",
    "MAX_STAGE_EVIDENCE_FILE_BYTES",
    "REQUIRED_SEMANTIC_CHECKS",
    "SEMANTIC_PREREQUISITE_SCHEMA_VERSION",
    "RESTORE_PENDING_FILENAME",
    "RESTORE_PENDING_SCHEMA_VERSION",
    "STATE_FILENAME",
    "STAGE_EVIDENCE_COMPLETE_MARKER_FILENAME",
    "STAGE_EVIDENCE_COMPLETE_MARKER_SCHEMA_VERSION",
    "STAGE_EVIDENCE_COMPONENT_DIRECTORY_NAME",
    "STAGE_EVIDENCE_FILENAME",
    "STAGE_OUTPUTS_DIRECTORY_NAME",
    "STORE_SCHEMA_VERSION",
    "SecFilingGemmaRevealStore",
    "SecFilingGemmaRevealStoreError",
    "SemanticPrerequisiteValidation",
]
