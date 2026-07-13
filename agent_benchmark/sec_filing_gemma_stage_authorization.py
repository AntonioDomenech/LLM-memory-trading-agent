"""Pure consumed-stage authorization grants for downstream SEC/Gemma APIs.

The effectful reveal store remains the only component allowed to consume a
request.  This module can derive a detached pin from an already authenticated
post-consumption store snapshot, mint a grant for that snapshot's exact ledger
tip, and validate the grant later against a separately supplied snapshot/pin
plus the independently loaded monotonic current-tip anchor.

Pin derivation is not external authentication.  A downstream caller must load
the current-tip anchor independently of the bundle; an old snapshot and its old
bundled pin are deliberately rejected after any newer store transition.
No function here performs filesystem, network, model, market, SEC, or outcome
I/O, and the compact grant contains no market values, labels, scores, or returns.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import json
import re
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_contract import (
    CONTRACT_VERSION,
    REQUIRED_STAGE_VERIFIER_CHECKS,
    build_stage_content_manifest,
    canonical_sha256,
    validate_candidate_manifest,
)
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    REVEAL_REQUEST_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_stage_access import (
    DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID,
    DEVELOPMENT_CONTENT_ROOT_PLAN_SCHEMA_VERSION,
    STAGE_ACCESS_MANIFEST_SCHEMA_VERSION,
    validate_development_content_root_plan,
    validate_prior_same_form_carry_in_scope,
)
from agent_benchmark.sec_filing_gemma_stage_verifier import (
    STAGE_EVIDENCE_SCHEMA_VERSION,
    detach_untrusted_stage_json,
)
from agent_benchmark.sec_filing_gemma_source_identity import (
    CANONICAL_SOURCE_ROLE_PATHS,
)


STORE_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-reveal-store-v1"
CONSUMPTION_LEDGER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-request-ledger-v1"
)
CONSUMPTION_ENTRY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-request-entry-v1"
)
SEMANTIC_PREREQUISITE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-semantic-prerequisite-validation-v1"
)
CONSUMED_STAGE_STORE_PIN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-stage-store-pin-v1"
)
CONSUMED_STAGE_AUTHORIZATION_GRANT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-stage-authorization-grant-v1"
)
CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-stage-authorization-bundle-v1"
)
CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-stage-output-receipt-v2"
)
STAGE_SEC_EXECUTION_CLAIM_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-sec-execution-claim-v2"
)
STAGE_SEC_READER_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-sec-reader-receipt-v2"
)
STAGE_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-carry-in-reader-receipt-v1"
)
STAGE_SEC_EXECUTION_ABORT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-sec-execution-abort-v1"
)
DEVELOPMENT_SEC_EXECUTION_CLAIM_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-sec-execution-claim-v1"
)
DEVELOPMENT_SEC_READER_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-sec-reader-receipt-v1"
)
DEVELOPMENT_SEC_EXECUTION_ABORT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-sec-execution-abort-v1"
)
DEVELOPMENT_ROOT_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-root-carry-in-reader-receipt-v1"
)
REVEAL_STORE_CURRENT_TIP_ANCHOR_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-reveal-store-current-tip-anchor-v7"
)
TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-trusted-stage-content-pin-v2"
)
TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-trusted-stage-content-authentication-v1"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TAGGED_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_OUTPUT_NAMESPACE_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_AAPL_ACCESSION_RE = re.compile(r"0000320193-[0-9]{2}-[0-9]{6}\Z")
STAGE_RUNNER_REPOSITORY_PATH: Final[str] = (
    "agent_benchmark/sec_filing_gemma_stage_runner.py"
)
SEC_CORPUS_REPOSITORY_PATH: Final[str] = (
    "agent_benchmark/sec_filing_gemma_corpus.py"
)
SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID: Final[str] = "sec_stage_document_batch"
STAGE_EVIDENCE_OUTPUT_COMPONENT_ID: Final[str] = "owned_stage_evidence_document"
STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH: Final[str] = "stage_evidence.json"
# A Latin-1 source byte can expand to at most two UTF-8 bytes.  Keeping the
# complete raw batch at 64 MiB therefore guarantees that each normalized file
# remains inside the reveal store's fixed 128 MiB per-file ceiling, while the
# raw and normalized batch plus canonical JSON stays far below its total cap.
OWNED_SEC_RAW_BATCH_MAX_BYTES: Final[int] = 64 * 1024 * 1024
SEC_EXECUTION_RESOLVED_SOURCE_PATHS: Final[tuple[tuple[str, str], ...]] = tuple(
    (role, path)
    for role, path in CANONICAL_SOURCE_ROLE_PATHS.items()
    if path is not None
)
_STAGE_PREREQUISITES: Final[dict[str, str]] = {
    "intermediate": "development",
    "final": "intermediate",
}
_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "anchor",
        "latest_registry",
        "latest_registry_pin",
        "consumption_ledger",
        "state_sha256",
    }
)
_LEDGER_KEYS: Final[frozenset[str]] = frozenset(
    {"schema_version", "entries", "chain", "ledger_sha256"}
)
_LEDGER_CHAIN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "genesis_tip_sha256",
        "tip_sha256",
        "consumed_request_count",
        "actual_final_touch_count",
        "historical_final_reveal_count_lower_bound",
        "repository_final_touch_count_lower_bound",
    }
)
_ENTRY_KEYS: Final[frozenset[str]] = frozenset(
    {
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
)
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
_VALIDATION_KEYS: Final[frozenset[str]] = frozenset(
    {
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
)
_STORE_PIN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "store_state_sha256",
        "store_snapshot_bytes_sha256",
        "store_snapshot_byte_count",
        "store_anchor_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "consumption_ledger_sha256",
        "consumption_ledger_tip_sha256",
        "consumed_request_count",
        "store_pin_sha256",
    }
)
_GRANT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "authorization_kind",
        "consumption_entry_sha256",
        "consumption_entry_sequence",
        "request_sha256",
        "attempt_id",
        "candidate_sha256",
        "registry_entry_sha256",
        "prerequisite_stage",
        "stage",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "output_namespace",
        "store_state_sha256",
        "store_snapshot_bytes_sha256",
        "store_snapshot_byte_count",
        "store_pin_sha256",
        "consumption_ledger_sha256",
        "consumption_ledger_tip_sha256",
        "consumed_stage_access_authorized",
        "authorization_scope",
        "outcomes_included",
        "market_values_included",
        "cross_stage_access_permitted",
        "grant_reuse_across_store_state_tips_permitted",
        "authorization_grant_sha256",
    }
)
_BUNDLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "authenticated_store_snapshot",
        "store_state_pin",
        "authorization_grant",
        "bundle_sha256",
    }
)
_STAGE_OUTPUT_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "receipt_kind",
        "consumption_entry_sha256",
        "consumption_entry_sequence",
        "request_sha256",
        "attempt_id",
        "candidate_sha256",
        "registry_entry_sha256",
        "input_prerequisite_stage",
        "output_stage",
        "input_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "output_namespace",
        "authorization_bundle_sha256",
        "authorization_grant_sha256",
        "sec_execution_claim_sha256",
        "sec_reader_receipt_sha256",
        "grant_store_state_sha256",
        "grant_consumption_ledger_sha256",
        "grant_consumption_ledger_tip_sha256",
        "output_kind",
        "output_stage_evidence_schema_version",
        "output_stage_evidence_component_id",
        "output_stage_evidence_relative_path",
        "output_stage_evidence_sha256",
        "output_stage_evidence_document_sha256",
        "output_stage_evidence_complete_marker_sha256",
        "output_stage_evidence_canonical_byte_count",
        "output_stage_evidence_prerequisite_stage",
        "output_parent_stage_evidence_sha256",
        "output_candidate_sha256",
        "output_stage_evidence_recomputed_by_store",
        "fresh_stage_evidence_provenance_claimed",
        "cross_stage_output_permitted",
        "grant_reuse_for_different_output_permitted",
        "output_receipt_sha256",
    }
)
_STAGE_SEC_EXECUTION_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "claim_kind",
        "request_sha256",
        "consumption_entry_sha256",
        "consumption_entry_sequence",
        "attempt_id",
        "candidate_sha256",
        "registry_entry_sha256",
        "input_prerequisite_stage",
        "authorized_stage",
        "input_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "output_namespace",
        "authorization_bundle_sha256",
        "authorization_grant_sha256",
        "grant_store_state_sha256",
        "grant_consumption_ledger_sha256",
        "grant_consumption_ledger_tip_sha256",
        "start_current_tip_anchor_sha256",
        "runner_repository_path",
        "runner_source_sha256",
        "sec_corpus_repository_path",
        "sec_corpus_source_sha256",
        "execution_source_hashes",
        "execution_source_hashes_sha256",
        "execution_source_role_count",
        "sec_user_agent_sha256",
        "sec_component_id",
        "sec_component_plan_sha256",
        "effect_may_be_repeated_after_indeterminate_crash",
        "claim_sha256",
    }
)
_STAGE_SEC_READER_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "receipt_kind",
        "request_sha256",
        "claim_sha256",
        "authorized_stage",
        "candidate_sha256",
        "output_namespace",
        "authorization_bundle_sha256",
        "authorization_grant_sha256",
        "sec_component_id",
        "sec_component_plan_sha256",
        "runner_source_sha256",
        "sec_corpus_source_sha256",
        "execution_source_hashes_sha256",
        "execution_source_role_count",
        "sec_user_agent_sha256",
        "byte_index",
        "byte_index_sha256",
        "byte_count_total",
        "complete_marker_sha256",
        "fresh_network_provenance_claimed",
        "reader_output_recomputed_by_store",
        "receipt_sha256",
    }
)
_STAGE_CARRY_IN_READER_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "receipt_kind",
        "request_sha256",
        "consumption_entry_sha256",
        "consumption_entry_sequence",
        "attempt_id",
        "candidate_sha256",
        "registry_entry_sha256",
        "input_prerequisite_stage",
        "authorized_stage",
        "input_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "output_namespace",
        "authorization_bundle_sha256",
        "authorization_grant_sha256",
        "grant_store_state_sha256",
        "grant_consumption_ledger_sha256",
        "grant_consumption_ledger_tip_sha256",
        "child_sec_execution_claim_sha256",
        "child_sec_reader_receipt_sha256",
        "parent_request_sha256",
        "parent_sec_execution_claim_sha256",
        "parent_sec_reader_receipt_sha256",
        "parent_consumed_stage_output_receipt_sha256",
        "parent_content_manifest_sha256",
        "parent_stage_artifact_sha256",
        "parent_external_seal_receipt_sha256",
        "parent_stage_evidence_sha256",
        "parent_stage_evidence_document_sha256",
        "parent_stage_evidence_complete_marker_sha256",
        "selection_policy",
        "artifact_scope",
        "carry_in_record_count",
        "carry_in_records_sha256",
        "carry_in_records",
        "carry_in_byte_index",
        "carry_in_byte_index_sha256",
        "carry_in_byte_count_total",
        "carry_in_complete_marker_sha256",
        "reader_source_sha256",
        "reader_output_recomputed_by_store",
        "network_refetch_permitted",
        "write_permitted",
        "general_cross_stage_access_permitted",
        "fresh_carry_in_provenance_claimed",
        "receipt_sha256",
    }
)
_STAGE_SEC_EXECUTION_ABORT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "abort_kind",
        "request_sha256",
        "claim_sha256",
        "authorized_stage",
        "candidate_sha256",
        "output_namespace",
        "reason",
        "external_effect_retry_permitted",
        "abort_sha256",
    }
)
_DEVELOPMENT_CONTENT_ROOT_PLAN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "root_scope",
        "development_root_scope_sha256",
        "corpus_provenance",
        "corpus_universe_manifest",
        "sec_access_plan",
        "budgets",
        "output",
        "scope",
        "authorization_semantics",
        "development_content_root_plan_sha256",
    }
)
_DEVELOPMENT_SEC_EXECUTION_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "claim_kind",
        "development_root_scope_sha256",
        "development_content_root_plan_sha256",
        "development_content_root_plan",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "registered_entry_count",
        "corpus_universe_sha256",
        "corpus_universe_semantic_sha256",
        "start_current_tip_anchor_sha256",
        "start_state_sha256",
        "start_consumption_ledger_sha256",
        "start_consumption_ledger_tip_sha256",
        "start_consumed_request_count",
        "output_namespace",
        "output_write_mode",
        "sec_component_id",
        "runner_repository_path",
        "runner_source_sha256",
        "sec_corpus_repository_path",
        "sec_corpus_source_sha256",
        "execution_source_hashes",
        "execution_source_hashes_sha256",
        "execution_source_role_count",
        "sec_user_agent_sha256",
        "authorizes_outcome_access",
        "market_access_permitted",
        "model_access_permitted",
        "future_stage_access_permitted",
        "reveal_request_consumption_permitted",
        "consumption_ledger_mutation_permitted",
        "effect_may_be_repeated_after_indeterminate_crash",
        "claim_sha256",
    }
)
_DEVELOPMENT_SEC_READER_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "receipt_kind",
        "development_root_scope_sha256",
        "claim_sha256",
        "development_content_root_plan_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "corpus_universe_sha256",
        "corpus_universe_semantic_sha256",
        "output_namespace",
        "sec_component_id",
        "runner_source_sha256",
        "sec_corpus_source_sha256",
        "execution_source_hashes_sha256",
        "execution_source_role_count",
        "sec_user_agent_sha256",
        "content_manifest_sha256",
        "byte_index",
        "byte_index_sha256",
        "byte_count_total",
        "complete_marker_sha256",
        "fresh_network_provenance_claimed",
        "reader_output_recomputed_by_store",
        "receipt_sha256",
    }
)
_DEVELOPMENT_SEC_EXECUTION_ABORT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "abort_kind",
        "development_root_scope_sha256",
        "claim_sha256",
        "development_content_root_plan_sha256",
        "attempt_id",
        "candidate_sha256",
        "output_namespace",
        "sec_component_id",
        "reason",
        "external_effect_retry_permitted",
        "abort_sha256",
    }
)
_DEVELOPMENT_ROOT_CARRY_IN_READER_RECEIPT_KEYS: Final[frozenset[str]] = (
    frozenset(
        {
            "schema_version",
            "contract_version",
            "receipt_kind",
            "request_sha256",
            "consumption_entry_sha256",
            "consumption_entry_sequence",
            "attempt_id",
            "candidate_sha256",
            "candidate_design_sha256",
            "registry_entry_sha256",
            "registry_sha256",
            "registry_tip_sha256",
            "registered_entry_count",
            "input_prerequisite_stage",
            "authorized_stage",
            "input_stage_evidence_sha256",
            "stage_access_manifest_sha256",
            "output_namespace",
            "authorization_bundle_sha256",
            "authorization_grant_sha256",
            "grant_store_state_sha256",
            "grant_consumption_ledger_sha256",
            "grant_consumption_ledger_tip_sha256",
            "child_sec_execution_claim_sha256",
            "child_sec_reader_receipt_sha256",
            "development_root_scope_sha256",
            "development_sec_execution_claim_sha256",
            "development_sec_reader_receipt_sha256",
            "development_content_root_plan_sha256",
            "development_content_manifest_sha256",
            "corpus_universe_sha256",
            "corpus_universe_semantic_sha256",
            "prerequisite_stage_artifact_sha256",
            "prerequisite_external_seal_receipt_sha256",
            "selection_policy",
            "artifact_scope",
            "carry_in_record_count",
            "carry_in_records_sha256",
            "carry_in_records",
            "carry_in_byte_index",
            "carry_in_byte_index_sha256",
            "carry_in_byte_count_total",
            "carry_in_complete_marker_sha256",
            "reader_source_sha256",
            "reader_output_recomputed_by_store",
            "network_refetch_permitted",
            "write_permitted",
            "general_cross_stage_access_permitted",
            "fresh_carry_in_provenance_claimed",
            "receipt_sha256",
        }
    )
)
_CURRENT_TIP_ANCHOR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "revision",
        "previous_tip_anchor_sha256",
        "state_sha256",
        "state_snapshot_bytes_sha256",
        "state_snapshot_byte_count",
        "registry_sha256",
        "registry_tip_sha256",
        "consumption_ledger_sha256",
        "consumption_ledger_tip_sha256",
        "consumed_request_count",
        "trusted_stage_content_pins",
        "authorization_bundles",
        "consumed_stage_output_receipts",
        "stage_sec_execution_claims",
        "stage_sec_reader_receipts",
        "stage_carry_in_reader_receipts",
        "stage_sec_execution_aborts",
        "development_sec_execution_claims",
        "development_sec_reader_receipts",
        "development_sec_execution_aborts",
        "development_root_carry_in_reader_receipts",
        "tip_anchor_sha256",
    }
)
_TRUSTED_STAGE_CONTENT_PIN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "request_sha256",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "prerequisite_stage",
        "requested_stage",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "content_manifest_sha256",
        "stage_artifact_sha256",
        "external_seal_receipt_sha256",
        "trusted_store_state_sha256",
        "pin_sha256",
    }
)
_TRUSTED_STAGE_CONTENT_AUTHENTICATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "authentication_kind",
        "request_sha256",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "prerequisite_stage",
        "requested_stage",
        "trusted_stage_content_pin_sha256",
        "trusted_store_state_sha256",
        "trusted_current_tip_anchor_sha256",
        "trusted_current_tip_revision",
        "trusted_stage_content_pins_sha256",
        "authentication_receipt_sha256",
    }
)


class SecFilingGemmaStageAuthorizationError(ValueError):
    """A consumed-stage grant or its externally supplied state proof is invalid."""


def _plain(value: Any, location: str) -> Any:
    """Return one detached exact-JSON value under fixed allocation bounds."""

    try:
        return detach_untrusted_stage_json(value, location)
    except Exception as exc:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must contain exact built-in values within the "
            "bounded exact-JSON authorization limits"
        ) from exc


def _mapping(value: Any, location: str) -> dict[str, Any]:
    # Authorization-critical inputs must never execute caller-controlled
    # ``Mapping`` methods while they are being detached.  Exact built-in
    # containers have no overridable iteration/item hooks.
    if type(value) is not dict:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must be an exact built-in dict"
        )
    detached = _plain(value, location)
    if type(detached) is not dict:
        raise SecFilingGemmaStageAuthorizationError(f"{location} must be an object")
    return detached


def _expect_keys(value: Mapping[str, Any], expected: frozenset[str], location: str) -> None:
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} keys changed; missing={missing}, extra={extra}"
        )


def _sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _tagged_sha256(value: Any, location: str) -> str:
    if type(value) is not str or _TAGGED_SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must be a tagged lowercase SHA-256 digest"
        )
    return value


def _safe_id(value: Any, location: str) -> str:
    if type(value) is not str or _SAFE_ID_RE.fullmatch(value) is None:
        raise SecFilingGemmaStageAuthorizationError(f"{location} is invalid")
    return value


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must be an integer at least {minimum}"
        )
    return value


def _self_hash(value: Mapping[str, Any], field: str, location: str) -> str:
    observed = _sha256(value.get(field), f"{location} {field}")
    body = {key: value[key] for key in value if key != field}
    expected = canonical_sha256(body)
    if not hmac.compare_digest(observed, expected):
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} self-hash is inconsistent"
        )
    return observed


def _encoded_store_snapshot(value: Mapping[str, Any]) -> bytes:
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
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageAuthorizationError(
            "Authenticated store snapshot cannot be encoded exactly"
        ) from exc


def _consumption_genesis(anchor: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": "aapl-sec-gemma-consumed-request-genesis-v1",
            "contract_version": CONTRACT_VERSION,
            "store_anchor_sha256": canonical_sha256(anchor),
        }
    )


def _validated_entry(
    raw: Any, *, expected_sequence: int, expected_prior_tip_sha256: str
) -> dict[str, Any]:
    entry = _mapping(raw, f"consumption entry {expected_sequence}")
    _expect_keys(entry, _ENTRY_KEYS, f"consumption entry {expected_sequence}")
    if (
        entry["schema_version"] != CONSUMPTION_ENTRY_SCHEMA_VERSION
        or _strict_int(entry["sequence"], "consumption entry sequence", minimum=1)
        != expected_sequence
        or entry["prior_tip_sha256"] != expected_prior_tip_sha256
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumption entry sequence, schema, or prior tip changed"
        )
    stage = entry["stage"]
    if stage not in _STAGE_PREREQUISITES:
        raise SecFilingGemmaStageAuthorizationError("Consumption entry stage is invalid")
    _safe_id(entry["attempt_id"], "consumption attempt id")
    _sha256(entry["request_sha256"], "consumption request hash")
    _sha256(entry["candidate_sha256"], "consumption candidate hash")
    _sha256(entry["registry_entry_sha256"], "consumption registry entry hash")
    _self_hash(entry, "entry_sha256", f"consumption entry {expected_sequence}")
    return entry


def _validated_ledger(
    raw: Any, *, anchor: Mapping[str, Any]
) -> dict[str, Any]:
    ledger = _mapping(raw, "consumption ledger")
    _expect_keys(ledger, _LEDGER_KEYS, "consumption ledger")
    if ledger["schema_version"] != CONSUMPTION_LEDGER_SCHEMA_VERSION:
        raise SecFilingGemmaStageAuthorizationError("Consumption ledger schema changed")
    entries_raw = ledger["entries"]
    if type(entries_raw) is not list:
        raise SecFilingGemmaStageAuthorizationError("Consumption entries must be a list")
    chain = _mapping(ledger["chain"], "consumption ledger chain")
    _expect_keys(chain, _LEDGER_CHAIN_KEYS, "consumption ledger chain")
    genesis = _consumption_genesis(anchor)
    if chain["genesis_tip_sha256"] != genesis:
        raise SecFilingGemmaStageAuthorizationError("Consumption genesis changed")
    current_tip = genesis
    entries: list[dict[str, Any]] = []
    seen_requests: set[str] = set()
    seen_attempt_stages: set[tuple[str, str]] = set()
    intermediate_candidates: set[tuple[str, str, str]] = set()
    final_count = 0
    for sequence, raw_entry in enumerate(entries_raw, start=1):
        entry = _validated_entry(
            raw_entry,
            expected_sequence=sequence,
            expected_prior_tip_sha256=current_tip,
        )
        request_hash = entry["request_sha256"]
        attempt_stage = (entry["attempt_id"], entry["stage"])
        if request_hash in seen_requests or attempt_stage in seen_attempt_stages:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumption request or candidate stage was replayed"
            )
        candidate_key = (
            entry["attempt_id"],
            entry["candidate_sha256"],
            entry["registry_entry_sha256"],
        )
        if entry["stage"] == "final" and candidate_key not in intermediate_candidates:
            raise SecFilingGemmaStageAuthorizationError(
                "Final consumption lacks its exact intermediate predecessor"
            )
        expected_delta = 1 if entry["stage"] == "final" else 0
        if entry["final_touch_delta"] != expected_delta:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumption final-touch delta changed"
            )
        final_count += expected_delta
        if entry["cumulative_actual_final_touch_count"] != final_count:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumption cumulative final-touch count changed"
            )
        entries.append(entry)
        seen_requests.add(request_hash)
        seen_attempt_stages.add(attempt_stage)
        if entry["stage"] == "intermediate":
            intermediate_candidates.add(candidate_key)
        current_tip = entry["entry_sha256"]
    count = _strict_int(chain["consumed_request_count"], "consumed request count")
    actual_final = _strict_int(chain["actual_final_touch_count"], "actual final count")
    historical_final = _strict_int(
        chain["historical_final_reveal_count_lower_bound"],
        "historical final lower bound",
    )
    repository_final = _strict_int(
        chain["repository_final_touch_count_lower_bound"],
        "repository final lower bound",
    )
    if (
        count != len(entries)
        or actual_final != final_count
        or repository_final != historical_final + final_count
        or chain["tip_sha256"] != current_tip
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumption ledger count, tip, or final-touch totals changed"
        )
    _self_hash(ledger, "ledger_sha256", "consumption ledger")
    return ledger


def _validated_store_snapshot(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _mapping(raw, "authenticated reveal-store snapshot")
    _expect_keys(state, _STATE_KEYS, "authenticated reveal-store snapshot")
    if (
        state["schema_version"] != STORE_SCHEMA_VERSION
        or state["contract_version"] != CONTRACT_VERSION
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Reveal-store snapshot schema or contract changed"
        )
    anchor = _mapping(state["anchor"], "store anchor")
    registry = _mapping(state["latest_registry"], "latest registry")
    registry_pin = _mapping(state["latest_registry_pin"], "latest registry pin")
    registry_hash = _sha256(registry.get("registry_sha256"), "latest registry hash")
    if registry_pin.get("registry_sha256") != registry_hash:
        raise SecFilingGemmaStageAuthorizationError(
            "Latest registry pin belongs to another registry"
        )
    registry_tip = _sha256(registry_pin.get("tip_sha256"), "latest registry tip")
    registry_chain = _mapping(registry.get("chain"), "latest registry chain")
    if (
        registry_chain.get("tip_sha256") != registry_tip
        or registry_chain.get("registered_entry_count")
        != registry_pin.get("registered_entry_count")
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Latest registry chain and pin disagree"
        )
    ledger = _validated_ledger(state["consumption_ledger"], anchor=anchor)
    _self_hash(state, "state_sha256", "authenticated reveal-store snapshot")
    return state, ledger


def derive_consumed_stage_store_state_pin(
    authenticated_store_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a pin that must be persisted through a separate trust path."""

    state, ledger = _validated_store_snapshot(authenticated_store_snapshot)
    registry = state["latest_registry"]
    registry_pin = state["latest_registry_pin"]
    chain = ledger["chain"]
    snapshot_bytes = _encoded_store_snapshot(state)
    body = {
        "schema_version": CONSUMED_STAGE_STORE_PIN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "store_state_sha256": state["state_sha256"],
        "store_snapshot_bytes_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "store_snapshot_byte_count": len(snapshot_bytes),
        "store_anchor_sha256": canonical_sha256(state["anchor"]),
        "registry_sha256": registry["registry_sha256"],
        "registry_tip_sha256": registry_pin["tip_sha256"],
        "consumption_ledger_sha256": ledger["ledger_sha256"],
        "consumption_ledger_tip_sha256": chain["tip_sha256"],
        "consumed_request_count": chain["consumed_request_count"],
    }
    return {**body, "store_pin_sha256": canonical_sha256(body)}


def validate_consumed_stage_store_state_pin(
    authenticated_store_snapshot: Mapping[str, Any],
    external_store_state_pin: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an authenticated snapshot against a separately supplied pin."""

    observed = _mapping(external_store_state_pin, "external store-state pin")
    _expect_keys(observed, _STORE_PIN_KEYS, "external store-state pin")
    _self_hash(observed, "store_pin_sha256", "external store-state pin")
    expected = derive_consumed_stage_store_state_pin(authenticated_store_snapshot)
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "External store-state pin does not authenticate this exact snapshot"
        )
    return observed


def _validated_trusted_stage_content_pin(
    raw: Any,
    *,
    expected_request_sha256: str | None = None,
) -> dict[str, Any]:
    pin = _mapping(raw, "trusted stage-content pin")
    _expect_keys(
        pin,
        _TRUSTED_STAGE_CONTENT_PIN_KEYS,
        "trusted stage-content pin",
    )
    if (
        pin["schema_version"] != TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION
        or pin["contract_version"] != CONTRACT_VERSION
        or _STAGE_PREREQUISITES.get(pin["requested_stage"])
        != pin["prerequisite_stage"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Trusted stage-content pin schema, contract, or stage transition changed"
        )
    _safe_id(pin["attempt_id"], "trusted content attempt id")
    for field in (
        "request_sha256",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "content_manifest_sha256",
        "stage_artifact_sha256",
        "external_seal_receipt_sha256",
        "trusted_store_state_sha256",
    ):
        _sha256(pin[field], f"trusted content {field}")
    pin_hash = _self_hash(pin, "pin_sha256", "trusted stage-content pin")
    if (
        expected_request_sha256 is not None
        and pin["request_sha256"]
        != _sha256(expected_request_sha256, "expected trusted-content request hash")
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Trusted stage-content pin is stored under another request"
        )
    pin["pin_sha256"] = pin_hash
    return pin


def _validated_trusted_stage_content_pins(
    raw: Any,
) -> dict[str, dict[str, Any]]:
    pins = _mapping(raw, "current-tip trusted stage-content pins")
    validated: dict[str, dict[str, Any]] = {}
    for request_sha256, raw_pin in pins.items():
        request_hash = _sha256(
            request_sha256,
            "current-tip trusted stage-content pin key",
        )
        validated[request_hash] = _validated_trusted_stage_content_pin(
            raw_pin,
            expected_request_sha256=request_hash,
        )
    return validated


def _validated_trusted_content_request_and_access(
    reveal_request: Mapping[str, Any],
    stage_access_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    request = _mapping(reveal_request, "trusted-content reveal request")
    _expect_keys(request, _REQUEST_KEYS, "trusted-content reveal request")
    request_hash = _self_hash(
        request,
        "request_sha256",
        "trusted-content reveal request",
    )
    requested_stage = request["stage"]
    prerequisite_stage = request["prerequisite_stage"]
    if (
        request["schema_version"] != REVEAL_REQUEST_SCHEMA_VERSION
        or request["contract_version"] != CONTRACT_VERSION
        or _STAGE_PREREQUISITES.get(requested_stage) != prerequisite_stage
        or request["authorizes_outcome_access"] is not False
        or request["effectful_atomic_single_use_consumption_required"] is not True
        or request["cross_attempt_comparison_permitted"] is not False
        or request["cross_attempt_winner_selection_permitted"] is not False
        or request["globally_pristine_claim"] is not False
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Trusted-content request changed its frozen non-authorizing semantics"
        )
    _safe_id(request["attempt_id"], "trusted-content request attempt id")
    for field in (
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
    ):
        _sha256(request[field], f"trusted-content request {field}")

    access = _mapping(
        stage_access_manifest,
        "trusted-content stage-access manifest",
    )
    access_hash = _self_hash(
        access,
        "stage_access_manifest_sha256",
        "trusted-content stage-access manifest",
    )
    transition = _mapping(
        access.get("transition"),
        "trusted-content stage-access transition",
    )
    candidate = _mapping(
        access.get("candidate"),
        "trusted-content stage-access candidate",
    )
    evidence_pin = _mapping(
        access.get("prerequisite_evidence_pin"),
        "trusted-content prerequisite evidence pin",
    )
    _expect_keys(
        evidence_pin,
        frozenset(
            {
                "stage",
                "content_manifest_sha256",
                "stage_artifact_sha256",
                "external_seal_receipt_sha256",
            }
        ),
        "trusted-content prerequisite evidence pin",
    )
    if (
        access.get("schema_version") != STAGE_ACCESS_MANIFEST_SCHEMA_VERSION
        or access.get("contract_version") != CONTRACT_VERSION
        or access_hash != request["stage_access_manifest_sha256"]
        or transition.get("prerequisite_stage") != prerequisite_stage
        or transition.get("requested_stage") != requested_stage
        or transition.get("single_use_consumption_required") is not True
        or transition.get("stage_reuse_permitted") is not False
        or candidate.get("attempt_id") != request["attempt_id"]
        or candidate.get("candidate_sha256") != request["candidate_sha256"]
        or candidate.get("candidate_design_sha256")
        != request["candidate_design_sha256"]
        or evidence_pin.get("stage") != prerequisite_stage
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Trusted-content stage access crossed its request or stage boundary"
        )
    for field in (
        "content_manifest_sha256",
        "stage_artifact_sha256",
        "external_seal_receipt_sha256",
    ):
        _sha256(evidence_pin[field], f"trusted-content prerequisite {field}")
    request["request_sha256"] = request_hash
    access["stage_access_manifest_sha256"] = access_hash
    return request, access, evidence_pin


def derive_reveal_store_trusted_stage_content_pin(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    reveal_request: Mapping[str, Any],
    stage_access_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the exact pre-consumption content pin for one validated request."""

    state, _ledger = _validated_store_snapshot(authenticated_store_snapshot)
    request, access, evidence_pin = _validated_trusted_content_request_and_access(
        reveal_request,
        stage_access_manifest,
    )
    body = {
        "schema_version": TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "request_sha256": request["request_sha256"],
        "prerequisite_stage_evidence_sha256": request[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": access[
            "stage_access_manifest_sha256"
        ],
        "prerequisite_stage": request["prerequisite_stage"],
        "requested_stage": request["stage"],
        "attempt_id": request["attempt_id"],
        "candidate_sha256": request["candidate_sha256"],
        "candidate_design_sha256": request["candidate_design_sha256"],
        "registry_entry_sha256": request["registry_entry_sha256"],
        "content_manifest_sha256": evidence_pin["content_manifest_sha256"],
        "stage_artifact_sha256": evidence_pin["stage_artifact_sha256"],
        "external_seal_receipt_sha256": evidence_pin[
            "external_seal_receipt_sha256"
        ],
        "trusted_store_state_sha256": state["state_sha256"],
    }
    return {**body, "pin_sha256": canonical_sha256(body)}


def authenticate_reveal_store_trusted_stage_content_pin(
    authenticated_store_snapshot: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    *,
    reveal_request: Mapping[str, Any],
    stage_access_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate exact pin membership through the independently loaded tip."""

    state, _ledger = _validated_store_snapshot(authenticated_store_snapshot)
    current_tip = validate_reveal_store_current_tip_anchor(
        state,
        independent_current_tip_anchor,
    )
    expected_pin = derive_reveal_store_trusted_stage_content_pin(
        state,
        reveal_request=reveal_request,
        stage_access_manifest=stage_access_manifest,
    )
    request_hash = expected_pin["request_sha256"]
    persisted = current_tip["trusted_stage_content_pins"].get(request_hash)
    if persisted != expected_pin:
        raise SecFilingGemmaStageAuthorizationError(
            "Trusted stage-content pin is not exactly persisted at the current tip"
        )
    body = {
        "schema_version": TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "authentication_kind": (
            "reveal_store_current_tip_persisted_trusted_stage_content_pin"
        ),
        "request_sha256": request_hash,
        "prerequisite_stage_evidence_sha256": expected_pin[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": expected_pin[
            "stage_access_manifest_sha256"
        ],
        "prerequisite_stage": expected_pin["prerequisite_stage"],
        "requested_stage": expected_pin["requested_stage"],
        "trusted_stage_content_pin_sha256": expected_pin["pin_sha256"],
        "trusted_store_state_sha256": state["state_sha256"],
        "trusted_current_tip_anchor_sha256": current_tip[
            "tip_anchor_sha256"
        ],
        "trusted_current_tip_revision": current_tip["revision"],
        "trusted_stage_content_pins_sha256": canonical_sha256(
            current_tip["trusted_stage_content_pins"]
        ),
    }
    return {
        **body,
        "authentication_receipt_sha256": canonical_sha256(body),
    }


def validate_trusted_stage_content_authentication_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(receipt, "trusted stage-content authentication receipt")
    _expect_keys(
        value,
        _TRUSTED_STAGE_CONTENT_AUTHENTICATION_KEYS,
        "trusted stage-content authentication receipt",
    )
    if (
        value["schema_version"]
        != TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["authentication_kind"]
        != "reveal_store_current_tip_persisted_trusted_stage_content_pin"
        or _STAGE_PREREQUISITES.get(value["requested_stage"])
        != value["prerequisite_stage"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Trusted stage-content authentication semantics changed"
        )
    for field in (
        "request_sha256",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "trusted_stage_content_pin_sha256",
        "trusted_store_state_sha256",
        "trusted_current_tip_anchor_sha256",
        "trusted_stage_content_pins_sha256",
    ):
        _sha256(value[field], f"trusted content authentication {field}")
    _strict_int(
        value["trusted_current_tip_revision"],
        "trusted content current-tip revision",
    )
    _self_hash(
        value,
        "authentication_receipt_sha256",
        "trusted stage-content authentication receipt",
    )
    return value


def _validated_authorization_bundles(raw: Any) -> dict[str, dict[str, Any]]:
    bundles = _mapping(raw, "current-tip authorization bundles")
    validated: dict[str, dict[str, Any]] = {}
    for request_sha256, raw_bundle in bundles.items():
        request_hash = _sha256(
            request_sha256, "current-tip authorization-bundle key"
        )
        bundle = _mapping(
            raw_bundle,
            f"current-tip authorization bundle {request_hash}",
        )
        _expect_keys(
            bundle,
            _BUNDLE_KEYS,
            f"current-tip authorization bundle {request_hash}",
        )
        if (
            bundle["schema_version"]
            != CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Current-tip authorization bundle schema changed"
            )
        _self_hash(
            bundle,
            "bundle_sha256",
            f"current-tip authorization bundle {request_hash}",
        )
        snapshot = _mapping(
            bundle["authenticated_store_snapshot"],
            "current-tip bundled store snapshot",
        )
        pin = _mapping(
            bundle["store_state_pin"],
            "current-tip bundled store-state pin",
        )
        grant = _mapping(
            bundle["authorization_grant"],
            "current-tip bundled authorization grant",
        )
        if grant.get("request_sha256") != request_hash:
            raise SecFilingGemmaStageAuthorizationError(
                "Current-tip authorization bundle is stored under another request"
            )
        # Validate all detached component self-hashes, then reconstruct the
        # exact grant from the bundled snapshot and pin.
        _validated_store_snapshot(snapshot)
        validate_consumed_stage_store_state_pin(snapshot, pin)
        _expect_keys(grant, _GRANT_KEYS, "current-tip bundled authorization grant")
        _self_hash(
            grant,
            "authorization_grant_sha256",
            "current-tip bundled authorization grant",
        )
        expected_grant = build_consumed_stage_authorization_grant(
            authenticated_store_snapshot=snapshot,
            external_store_state_pin=pin,
            expected_new_consumption_entry_sha256=_sha256(
                grant.get("consumption_entry_sha256"),
                "current-tip bundled consumption entry hash",
            ),
        )
        if grant != expected_grant:
            raise SecFilingGemmaStageAuthorizationError(
                "Current-tip authorization bundle grant is not exact"
            )
        validated[request_hash] = bundle
    return validated


def _validated_consumed_stage_output_receipts(
    raw: Any,
    *,
    authorization_bundles: Mapping[str, Any],
    sec_execution_claims: Mapping[str, Any],
    sec_reader_receipts: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    receipts = _mapping(raw, "current-tip consumed-stage output receipts")
    bundles = _mapping(
        authorization_bundles,
        "current-tip authorization bundles for output receipts",
    )
    claims = _mapping(
        sec_execution_claims,
        "current-tip SEC claims for output receipts",
    )
    readers = _mapping(
        sec_reader_receipts,
        "current-tip SEC reader receipts for output receipts",
    )
    validated: dict[str, dict[str, Any]] = {}
    for request_sha256, raw_receipt in receipts.items():
        request_hash = _sha256(
            request_sha256,
            "current-tip consumed-stage output-receipt key",
        )
        receipt = _mapping(
            raw_receipt,
            f"current-tip consumed-stage output receipt {request_hash}",
        )
        _expect_keys(
            receipt,
            _STAGE_OUTPUT_RECEIPT_KEYS,
            f"current-tip consumed-stage output receipt {request_hash}",
        )
        if (
            receipt["schema_version"]
            != CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION
            or receipt["contract_version"] != CONTRACT_VERSION
            or receipt["receipt_kind"]
            != "first_output_for_exact_consumed_stage_grant"
            or receipt["output_kind"] != "next_stage_evidence"
            or receipt["output_stage_evidence_schema_version"]
            != STAGE_EVIDENCE_SCHEMA_VERSION
            or receipt["output_stage_evidence_component_id"]
            != STAGE_EVIDENCE_OUTPUT_COMPONENT_ID
            or receipt["output_stage_evidence_relative_path"]
            != STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH
            or receipt["output_stage_evidence_recomputed_by_store"] is not True
            or receipt["fresh_stage_evidence_provenance_claimed"] is not False
            or receipt["cross_stage_output_permitted"] is not False
            or receipt["grant_reuse_for_different_output_permitted"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt semantics changed"
            )
        _self_hash(
            receipt,
            "output_receipt_sha256",
            f"current-tip consumed-stage output receipt {request_hash}",
        )
        for field in (
            "consumption_entry_sha256",
            "request_sha256",
            "candidate_sha256",
            "registry_entry_sha256",
            "input_stage_evidence_sha256",
            "stage_access_manifest_sha256",
            "authorization_bundle_sha256",
            "authorization_grant_sha256",
            "sec_execution_claim_sha256",
            "sec_reader_receipt_sha256",
            "grant_store_state_sha256",
            "grant_consumption_ledger_sha256",
            "grant_consumption_ledger_tip_sha256",
            "output_stage_evidence_sha256",
            "output_stage_evidence_document_sha256",
            "output_stage_evidence_complete_marker_sha256",
            "output_parent_stage_evidence_sha256",
            "output_candidate_sha256",
        ):
            _sha256(receipt[field], f"consumed-stage output receipt {field}")
        _strict_int(
            receipt["consumption_entry_sequence"],
            "consumed-stage output receipt entry sequence",
            minimum=1,
        )
        _strict_int(
            receipt["output_stage_evidence_canonical_byte_count"],
            "consumed-stage output receipt evidence byte count",
            minimum=2,
        )
        for field in (
            "attempt_id",
            "input_prerequisite_stage",
            "output_stage",
            "output_stage_evidence_schema_version",
            "output_stage_evidence_prerequisite_stage",
        ):
            _safe_id(receipt[field], f"consumed-stage output receipt {field}")
        namespace = receipt["output_namespace"]
        if (
            type(namespace) is not str
            or _OUTPUT_NAMESPACE_RE.fullmatch(namespace) is None
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt namespace is invalid"
            )
        if receipt["request_sha256"] != request_hash:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt is stored under another request"
            )
        bundle = bundles.get(request_hash)
        if type(bundle) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt lacks its persisted authorization bundle"
            )
        grant = _mapping(
            bundle.get("authorization_grant"),
            "consumed-stage output receipt authorization grant",
        )
        sec_claim = claims.get(request_hash)
        sec_reader = readers.get(request_hash)
        if type(sec_claim) is not dict or type(sec_reader) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt lacks its SEC execution ancestry"
            )
        expected_bindings = {
            "consumption_entry_sha256": grant.get("consumption_entry_sha256"),
            "consumption_entry_sequence": grant.get("consumption_entry_sequence"),
            "request_sha256": grant.get("request_sha256"),
            "attempt_id": grant.get("attempt_id"),
            "candidate_sha256": grant.get("candidate_sha256"),
            "registry_entry_sha256": grant.get("registry_entry_sha256"),
            "input_prerequisite_stage": grant.get("prerequisite_stage"),
            "output_stage": grant.get("stage"),
            "input_stage_evidence_sha256": grant.get(
                "prerequisite_stage_evidence_sha256"
            ),
            "stage_access_manifest_sha256": grant.get(
                "stage_access_manifest_sha256"
            ),
            "output_namespace": grant.get("output_namespace"),
            "authorization_bundle_sha256": bundle.get("bundle_sha256"),
            "authorization_grant_sha256": grant.get(
                "authorization_grant_sha256"
            ),
            "sec_execution_claim_sha256": sec_claim.get("claim_sha256"),
            "sec_reader_receipt_sha256": sec_reader.get("receipt_sha256"),
            "grant_store_state_sha256": grant.get("store_state_sha256"),
            "grant_consumption_ledger_sha256": grant.get(
                "consumption_ledger_sha256"
            ),
            "grant_consumption_ledger_tip_sha256": grant.get(
                "consumption_ledger_tip_sha256"
            ),
            "output_stage_evidence_prerequisite_stage": grant.get("stage"),
            "output_parent_stage_evidence_sha256": grant.get(
                "prerequisite_stage_evidence_sha256"
            ),
            "output_candidate_sha256": grant.get("candidate_sha256"),
        }
        if any(receipt[field] != expected for field, expected in expected_bindings.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt crossed its grant or output boundary"
            )
        validated[request_hash] = receipt
    return validated


def _validated_sec_byte_index(raw: Any) -> list[dict[str, Any]]:
    if type(raw) is not list or not raw:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC reader byte index must be a non-empty exact list"
        )
    validated: list[dict[str, Any]] = []
    logical_ids: set[str] = set()
    relative_paths: set[str] = set()
    for ordinal, raw_item in enumerate(raw, start=1):
        item = _mapping(raw_item, f"SEC reader byte index item {ordinal}")
        _expect_keys(
            item,
            {"ordinal", "logical_id", "relative_path", "byte_count", "sha256"},
            f"SEC reader byte index item {ordinal}",
        )
        if _strict_int(item["ordinal"], "SEC byte ordinal", minimum=1) != ordinal:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader byte index ordinals are not contiguous"
            )
        logical_id = _safe_id(item["logical_id"], "SEC byte logical id")
        path = item["relative_path"]
        if (
            type(path) is not str
            or not path
            or len(path) > 240
            or "\\" in path
            or path.startswith("/")
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or any(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", part) is None
                for part in path.split("/")
            )
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader byte index contains an unsafe relative path"
            )
        if logical_id in logical_ids or path.casefold() in relative_paths:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader byte index contains a duplicate or case-colliding item"
            )
        logical_ids.add(logical_id)
        relative_paths.add(path.casefold())
        validated.append(
            {
                "ordinal": ordinal,
                "logical_id": logical_id,
                "relative_path": path,
                "byte_count": _strict_int(
                    item["byte_count"], "SEC byte count", minimum=1
                ),
                "sha256": _sha256(item["sha256"], "SEC byte SHA-256"),
            }
        )
    return validated


def _validated_carry_in_records(
    raw: Any,
    *,
    expected_stage: str,
    expected_content_manifest_sha256: str,
) -> list[dict[str, Any]]:
    if type(raw) is not list or len(raw) != 2:
        raise SecFilingGemmaStageAuthorizationError(
            "Final carry-in must contain exactly one 10-K and one 10-Q"
        )
    if expected_stage != "intermediate":
        raise SecFilingGemmaStageAuthorizationError(
            "Only intermediate-to-final carry-in is supported"
        )
    content_hash = _sha256(
        expected_content_manifest_sha256,
        "final carry-in parent content-manifest hash",
    )
    expected_keys = {
        "accession_number",
        "form",
        "availability_session",
        "artifact_stage",
        "source_record_sha256",
        "normalized_text_sha256",
        "normalized_text_bytes",
        "content_record_sha256",
        "content_manifest_sha256",
    }
    validated: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw):
        record = _mapping(raw_record, f"final carry-in record {index}")
        _expect_keys(record, expected_keys, f"final carry-in record {index}")
        accession = _safe_id(
            record["accession_number"],
            f"final carry-in accession {index}",
        )
        if _AAPL_ACCESSION_RE.fullmatch(accession) is None:
            raise SecFilingGemmaStageAuthorizationError(
                "Final carry-in contains a non-Apple accession"
            )
        form = record["form"]
        if form not in {"10-K", "10-Q"}:
            raise SecFilingGemmaStageAuthorizationError(
                "Final carry-in form is outside the frozen periodic-report scope"
            )
        availability = _safe_id(
            record["availability_session"],
            f"final carry-in availability session {index}",
        )
        artifact_stage = _safe_id(
            record["artifact_stage"],
            f"final carry-in artifact stage {index}",
        )
        if artifact_stage != expected_stage:
            raise SecFilingGemmaStageAuthorizationError(
                "Final carry-in crossed its intermediate parent stage"
            )
        record_content_hash = _sha256(
            record["content_manifest_sha256"],
            f"final carry-in content-manifest hash {index}",
        )
        if record_content_hash != content_hash:
            raise SecFilingGemmaStageAuthorizationError(
                "Final carry-in record crossed its parent content manifest"
            )
        validated.append(
            {
                "accession_number": accession,
                "form": form,
                "availability_session": availability,
                "artifact_stage": artifact_stage,
                "source_record_sha256": _sha256(
                    record["source_record_sha256"],
                    f"final carry-in source-record hash {index}",
                ),
                "normalized_text_sha256": _sha256(
                    record["normalized_text_sha256"],
                    f"final carry-in normalized-text hash {index}",
                ),
                "normalized_text_bytes": _strict_int(
                    record["normalized_text_bytes"],
                    f"final carry-in normalized-text byte count {index}",
                    minimum=1,
                ),
                "content_record_sha256": _sha256(
                    record["content_record_sha256"],
                    f"final carry-in content-record hash {index}",
                ),
                "content_manifest_sha256": record_content_hash,
            }
        )
    if [record["form"] for record in validated] != ["10-K", "10-Q"]:
        raise SecFilingGemmaStageAuthorizationError(
            "Final carry-in records are not in the exact frozen form order"
        )
    if len({record["accession_number"] for record in validated}) != 2:
        raise SecFilingGemmaStageAuthorizationError(
            "Final carry-in accessions are duplicated"
        )
    return validated


def _validated_stage_sec_execution_claims(
    raw: Any,
    *,
    authorization_bundles: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    claims = _mapping(raw, "current-tip SEC execution claims")
    bundles = _mapping(
        authorization_bundles,
        "current-tip authorization bundles for SEC execution claims",
    )
    validated: dict[str, dict[str, Any]] = {}
    for request_sha256, raw_claim in claims.items():
        request_hash = _sha256(request_sha256, "SEC execution claim map key")
        claim = _mapping(raw_claim, f"SEC execution claim {request_hash}")
        _expect_keys(
            claim,
            _STAGE_SEC_EXECUTION_CLAIM_KEYS,
            f"SEC execution claim {request_hash}",
        )
        if (
            claim["schema_version"] != STAGE_SEC_EXECUTION_CLAIM_SCHEMA_VERSION
            or claim["contract_version"] != CONTRACT_VERSION
            or claim["claim_kind"] != "owned_sec_stage_document_batch"
            or claim["runner_repository_path"] != STAGE_RUNNER_REPOSITORY_PATH
            or claim["sec_corpus_repository_path"] != SEC_CORPUS_REPOSITORY_PATH
            or claim["sec_component_id"]
            != SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID
            or claim["effect_may_be_repeated_after_indeterminate_crash"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim semantics changed"
            )
        _self_hash(claim, "claim_sha256", "SEC execution claim")
        raw_execution_sources = _mapping(
            claim["execution_source_hashes"],
            "SEC execution claim source hashes",
        )
        expected_source_roles = {
            role for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
        }
        if set(raw_execution_sources) != expected_source_roles:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim does not bind the complete resolved source set"
            )
        execution_sources = {
            role: _sha256(
                raw_execution_sources[role],
                f"SEC execution claim source hash {role}",
            )
            for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
        }
        if (
            claim["execution_source_hashes"] != execution_sources
            or claim["execution_source_hashes_sha256"]
            != canonical_sha256(execution_sources)
            or claim["execution_source_role_count"] != len(execution_sources)
            or claim["runner_source_sha256"] != execution_sources["runner"]
            or claim["sec_corpus_source_sha256"]
            != execution_sources["sec_corpus_selector"]
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim source closure is inconsistent"
            )
        for field in (
            "request_sha256",
            "consumption_entry_sha256",
            "candidate_sha256",
            "registry_entry_sha256",
            "input_stage_evidence_sha256",
            "stage_access_manifest_sha256",
            "authorization_bundle_sha256",
            "authorization_grant_sha256",
            "grant_store_state_sha256",
            "grant_consumption_ledger_sha256",
            "grant_consumption_ledger_tip_sha256",
            "start_current_tip_anchor_sha256",
            "runner_source_sha256",
            "sec_corpus_source_sha256",
            "execution_source_hashes_sha256",
            "sec_component_plan_sha256",
        ):
            _sha256(claim[field], f"SEC execution claim {field}")
        _tagged_sha256(
            claim["sec_user_agent_sha256"],
            "SEC execution claim sec_user_agent_sha256",
        )
        _strict_int(
            claim["consumption_entry_sequence"],
            "SEC execution claim entry sequence",
            minimum=1,
        )
        _strict_int(
            claim["execution_source_role_count"],
            "SEC execution claim source-role count",
            minimum=1,
        )
        for field in (
            "attempt_id",
            "input_prerequisite_stage",
            "authorized_stage",
        ):
            _safe_id(claim[field], f"SEC execution claim {field}")
        namespace = claim["output_namespace"]
        if (
            type(namespace) is not str
            or _OUTPUT_NAMESPACE_RE.fullmatch(namespace) is None
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim namespace is invalid"
            )
        if claim["request_sha256"] != request_hash:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim is stored under another request"
            )
        bundle = bundles.get(request_hash)
        if type(bundle) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim lacks its persisted grant bundle"
            )
        grant = _mapping(
            bundle.get("authorization_grant"),
            "SEC execution claim grant",
        )
        expected = {
            "request_sha256": grant.get("request_sha256"),
            "consumption_entry_sha256": grant.get("consumption_entry_sha256"),
            "consumption_entry_sequence": grant.get("consumption_entry_sequence"),
            "attempt_id": grant.get("attempt_id"),
            "candidate_sha256": grant.get("candidate_sha256"),
            "registry_entry_sha256": grant.get("registry_entry_sha256"),
            "input_prerequisite_stage": grant.get("prerequisite_stage"),
            "authorized_stage": grant.get("stage"),
            "input_stage_evidence_sha256": grant.get(
                "prerequisite_stage_evidence_sha256"
            ),
            "stage_access_manifest_sha256": grant.get(
                "stage_access_manifest_sha256"
            ),
            "output_namespace": grant.get("output_namespace"),
            "authorization_bundle_sha256": bundle.get("bundle_sha256"),
            "authorization_grant_sha256": grant.get(
                "authorization_grant_sha256"
            ),
            "grant_store_state_sha256": grant.get("store_state_sha256"),
            "grant_consumption_ledger_sha256": grant.get(
                "consumption_ledger_sha256"
            ),
            "grant_consumption_ledger_tip_sha256": grant.get(
                "consumption_ledger_tip_sha256"
            ),
        }
        if any(claim[field] != value for field, value in expected.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim crossed its persisted grant"
            )
        validated[request_hash] = claim
    return validated


def _validated_stage_sec_reader_receipts(
    raw: Any,
    *,
    claims: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    receipts = _mapping(raw, "current-tip SEC reader receipts")
    validated: dict[str, dict[str, Any]] = {}
    for request_sha256, raw_receipt in receipts.items():
        request_hash = _sha256(request_sha256, "SEC reader receipt map key")
        receipt = _mapping(raw_receipt, f"SEC reader receipt {request_hash}")
        _expect_keys(
            receipt,
            _STAGE_SEC_READER_RECEIPT_KEYS,
            f"SEC reader receipt {request_hash}",
        )
        if (
            receipt["schema_version"] != STAGE_SEC_READER_RECEIPT_SCHEMA_VERSION
            or receipt["contract_version"] != CONTRACT_VERSION
            or receipt["receipt_kind"]
            != "store_rehashed_owned_sec_stage_document_batch"
            or receipt["sec_component_id"]
            != SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID
            or receipt["fresh_network_provenance_claimed"] is not False
            or receipt["reader_output_recomputed_by_store"] is not True
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader receipt semantics changed"
            )
        _self_hash(receipt, "receipt_sha256", "SEC reader receipt")
        byte_index = _validated_sec_byte_index(receipt["byte_index"])
        if (
            receipt["byte_index"] != byte_index
            or receipt["byte_index_sha256"] != canonical_sha256(byte_index)
            or receipt["byte_count_total"]
            != sum(item["byte_count"] for item in byte_index)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader receipt byte index is inconsistent"
            )
        _strict_int(
            receipt["byte_count_total"],
            "SEC reader receipt total byte count",
            minimum=1,
        )
        _strict_int(
            receipt["execution_source_role_count"],
            "SEC reader receipt source-role count",
            minimum=1,
        )
        for field in (
            "request_sha256",
            "claim_sha256",
            "candidate_sha256",
            "authorization_bundle_sha256",
            "authorization_grant_sha256",
            "sec_component_plan_sha256",
            "runner_source_sha256",
            "sec_corpus_source_sha256",
            "execution_source_hashes_sha256",
            "byte_index_sha256",
            "complete_marker_sha256",
        ):
            _sha256(receipt[field], f"SEC reader receipt {field}")
        _tagged_sha256(
            receipt["sec_user_agent_sha256"],
            "SEC reader receipt sec_user_agent_sha256",
        )
        claim = claims.get(request_hash)
        if type(claim) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader receipt lacks its execution claim"
            )
        expected = {
            "request_sha256": request_hash,
            "claim_sha256": claim.get("claim_sha256"),
            "authorized_stage": claim.get("authorized_stage"),
            "candidate_sha256": claim.get("candidate_sha256"),
            "output_namespace": claim.get("output_namespace"),
            "authorization_bundle_sha256": claim.get(
                "authorization_bundle_sha256"
            ),
            "authorization_grant_sha256": claim.get(
                "authorization_grant_sha256"
            ),
            "sec_component_id": claim.get("sec_component_id"),
            "sec_component_plan_sha256": claim.get(
                "sec_component_plan_sha256"
            ),
            "runner_source_sha256": claim.get("runner_source_sha256"),
            "sec_corpus_source_sha256": claim.get("sec_corpus_source_sha256"),
            "execution_source_hashes_sha256": claim.get(
                "execution_source_hashes_sha256"
            ),
            "execution_source_role_count": claim.get(
                "execution_source_role_count"
            ),
            "sec_user_agent_sha256": claim.get("sec_user_agent_sha256"),
        }
        if any(receipt[field] != value for field, value in expected.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader receipt crossed its execution claim"
            )
        validated[request_hash] = receipt
    return validated


def _validated_stage_carry_in_reader_receipts(
    raw: Any,
    *,
    authorization_bundles: Mapping[str, Any],
    sec_execution_claims: Mapping[str, Any],
    sec_reader_receipts: Mapping[str, Any],
    consumed_stage_output_receipts: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    receipts = _mapping(raw, "current-tip stage carry-in reader receipts")
    validated: dict[str, dict[str, Any]] = {}
    for request_sha256, raw_receipt in receipts.items():
        request_hash = _sha256(request_sha256, "stage carry-in receipt map key")
        receipt = _mapping(
            raw_receipt,
            f"stage carry-in reader receipt {request_hash}",
        )
        _expect_keys(
            receipt,
            _STAGE_CARRY_IN_READER_RECEIPT_KEYS,
            f"stage carry-in reader receipt {request_hash}",
        )
        if (
            receipt["schema_version"]
            != STAGE_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION
            or receipt["contract_version"] != CONTRACT_VERSION
            or receipt["receipt_kind"]
            != "store_rehashed_final_child_prior_same_form_carry_in"
            or receipt["reader_output_recomputed_by_store"] is not True
            or receipt["network_refetch_permitted"] is not False
            or receipt["write_permitted"] is not False
            or receipt["general_cross_stage_access_permitted"] is not False
            or receipt["fresh_carry_in_provenance_claimed"] is not False
            or receipt["request_sha256"] != request_hash
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage carry-in reader receipt semantics changed"
            )
        _self_hash(receipt, "receipt_sha256", "stage carry-in reader receipt")
        parent_request_hash = _sha256(
            receipt["parent_request_sha256"],
            "stage carry-in parent request hash",
        )
        child_bundle = authorization_bundles.get(request_hash)
        parent_bundle = authorization_bundles.get(parent_request_hash)
        child_claim = sec_execution_claims.get(request_hash)
        parent_claim = sec_execution_claims.get(parent_request_hash)
        child_reader = sec_reader_receipts.get(request_hash)
        parent_reader = sec_reader_receipts.get(parent_request_hash)
        parent_output = consumed_stage_output_receipts.get(parent_request_hash)
        if any(
            type(value) is not dict
            for value in (
                child_bundle,
                parent_bundle,
                child_claim,
                parent_claim,
                child_reader,
                parent_reader,
                parent_output,
            )
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage carry-in receipt lacks exact child or parent ancestry"
            )
        expected = build_stage_carry_in_reader_receipt(
            child_bundle,
            stage_sec_execution_claim=child_claim,
            stage_sec_reader_receipt=child_reader,
            parent_authorization_bundle=parent_bundle,
            parent_stage_sec_execution_claim=parent_claim,
            parent_stage_sec_reader_receipt=parent_reader,
            parent_consumed_stage_output_receipt=parent_output,
            carry_in_byte_index=receipt["carry_in_byte_index"],
            carry_in_complete_marker_sha256=receipt[
                "carry_in_complete_marker_sha256"
            ],
            reader_source_sha256=receipt["reader_source_sha256"],
        )
        if receipt != expected:
            raise SecFilingGemmaStageAuthorizationError(
                "Stage carry-in reader receipt differs from its exact ancestry"
            )
        validated[request_hash] = receipt
    return validated


def _validated_stage_sec_execution_aborts(
    raw: Any,
    *,
    claims: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    aborts = _mapping(raw, "current-tip SEC execution aborts")
    validated: dict[str, dict[str, Any]] = {}
    for request_sha256, raw_abort in aborts.items():
        request_hash = _sha256(request_sha256, "SEC execution abort map key")
        abort = _mapping(raw_abort, f"SEC execution abort {request_hash}")
        _expect_keys(
            abort,
            _STAGE_SEC_EXECUTION_ABORT_KEYS,
            f"SEC execution abort {request_hash}",
        )
        if (
            abort["schema_version"] != STAGE_SEC_EXECUTION_ABORT_SCHEMA_VERSION
            or abort["contract_version"] != CONTRACT_VERSION
            or abort["abort_kind"] != "indeterminate_owned_sec_stage_execution"
            or abort["reason"]
            not in {
                "claim_recovered_without_terminal_receipt",
                "external_effect_failed_or_completion_unknown",
                "durable_output_verification_failed",
            }
            or abort["external_effect_retry_permitted"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution abort semantics changed"
            )
        _self_hash(abort, "abort_sha256", "SEC execution abort")
        for field in (
            "request_sha256",
            "claim_sha256",
            "candidate_sha256",
        ):
            _sha256(abort[field], f"SEC execution abort {field}")
        claim = claims.get(request_hash)
        if type(claim) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution abort lacks its execution claim"
            )
        expected = {
            "request_sha256": request_hash,
            "claim_sha256": claim.get("claim_sha256"),
            "authorized_stage": claim.get("authorized_stage"),
            "candidate_sha256": claim.get("candidate_sha256"),
            "output_namespace": claim.get("output_namespace"),
        }
        if any(abort[field] != value for field, value in expected.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution abort crossed its claim"
            )
        validated[request_hash] = abort
    return validated


def _validated_development_content_root_plan(raw: Any) -> dict[str, Any]:
    plan = _mapping(raw, "development-content root plan")
    _expect_keys(
        plan,
        _DEVELOPMENT_CONTENT_ROOT_PLAN_KEYS,
        "development-content root plan",
    )
    if (
        plan["schema_version"] != DEVELOPMENT_CONTENT_ROOT_PLAN_SCHEMA_VERSION
        or plan["contract_version"] != CONTRACT_VERSION
        or plan["authorization_semantics"]
        != "non_authorizing_request_free_plan_until_owned_development_root_claim"
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-content root plan semantics changed"
        )
    _sha256(plan["contract_sha256"], "development-content root contract hash")
    plan_hash = _self_hash(
        plan,
        "development_content_root_plan_sha256",
        "development-content root plan",
    )
    root_scope = _mapping(
        plan["root_scope"],
        "development-content root scope",
    )
    _expect_keys(
        root_scope,
        frozenset(
            {
                "scope_kind",
                "artifact_stage",
                "candidate_sha256",
                "candidate_design_sha256",
                "attempt_id",
                "corpus_universe_sha256",
                "corpus_universe_semantic_sha256",
                "document_count",
                "accessions_sha256",
                "official_urls_sha256",
                "output_namespace",
                "component_id",
            }
        ),
        "development-content root scope",
    )
    if (
        root_scope["scope_kind"]
        != "request_free_candidate_bound_complete_development_content_root"
        or root_scope["artifact_stage"] != "development"
        or root_scope["component_id"] != DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-content root scope semantics changed"
        )
    scope_hash = _sha256(
        plan["development_root_scope_sha256"],
        "development root scope hash",
    )
    if not hmac.compare_digest(scope_hash, canonical_sha256(root_scope)):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-content root scope self-hash is inconsistent"
        )
    _safe_id(root_scope["attempt_id"], "development root attempt id")
    for field in (
        "candidate_sha256",
        "candidate_design_sha256",
        "corpus_universe_sha256",
        "corpus_universe_semantic_sha256",
        "accessions_sha256",
        "official_urls_sha256",
    ):
        _sha256(root_scope[field], f"development root scope {field}")
    document_count = _strict_int(
        root_scope["document_count"],
        "development root document count",
        minimum=1,
    )
    namespace = root_scope["output_namespace"]
    if (
        type(namespace) is not str
        or _OUTPUT_NAMESPACE_RE.fullmatch(namespace) is None
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-content root output namespace is invalid"
        )

    provenance = _mapping(
        plan["corpus_provenance"],
        "development-content root corpus provenance",
    )
    _expect_keys(
        provenance,
        frozenset(
            {
                "identity_role",
                "corpus_universe_sha256",
                "corpus_universe_semantic_sha256",
                "sec_catalog_artifact_sha256",
                "calendar_source_evidence_sha256",
                "session_calendar_sha256",
                "complete_coverage_required",
                "caller_selection_permitted",
            }
        ),
        "development-content root corpus provenance",
    )
    if (
        provenance["identity_role"]
        != "immutable_candidate_bound_complete_universe"
        or provenance["complete_coverage_required"] is not True
        or provenance["caller_selection_permitted"] is not False
        or provenance["corpus_universe_sha256"]
        != root_scope["corpus_universe_sha256"]
        or provenance["corpus_universe_semantic_sha256"]
        != root_scope["corpus_universe_semantic_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-content root corpus provenance changed"
        )
    for field in (
        "corpus_universe_sha256",
        "corpus_universe_semantic_sha256",
        "sec_catalog_artifact_sha256",
        "calendar_source_evidence_sha256",
        "session_calendar_sha256",
    ):
        _sha256(provenance[field], f"development root provenance {field}")
    universe = _mapping(
        plan["corpus_universe_manifest"],
        "development-content root corpus universe",
    )
    if (
        universe.get("universe_sha256") != root_scope["corpus_universe_sha256"]
        or universe.get("universe_semantic_sha256")
        != root_scope["corpus_universe_semantic_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-content root universe crossed its scope"
        )

    sec_plan = _mapping(
        plan["sec_access_plan"],
        "development-content root SEC plan",
    )
    if (
        sec_plan.get("artifact_stage") != "development"
        or sec_plan.get("selection_policy")
        != "all_and_only_development_stage_universe_primary_documents"
        or sec_plan.get("method") != "GET"
        or sec_plan.get("network_scope") != "official_sec_https_only"
        or sec_plan.get("redirects_permitted") is not False
        or sec_plan.get("retries_permitted") is not False
        or sec_plan.get("cache_substitution_permitted") is not False
        or sec_plan.get("document_count") != document_count
        or sec_plan.get("accessions_sha256")
        != root_scope["accessions_sha256"]
        or sec_plan.get("official_urls_sha256")
        != root_scope["official_urls_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-content root SEC plan crossed its exact scope"
        )
    if type(sec_plan.get("documents")) is not list or len(
        sec_plan["documents"]
    ) != document_count:
        raise SecFilingGemmaStageAuthorizationError(
            "Development-content root SEC plan document count changed"
        )

    budgets = _mapping(plan["budgets"], "development-content root budgets")
    if (
        budgets.get("max_sec_requests") != document_count
        or budgets.get("max_redirects") != 0
        or budgets.get("max_retries") != 0
        or budgets.get("max_paid_api_calls") != 0
        or budgets.get("max_estimated_cost_usd") != 0.0
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-content root budgets changed their zero-cost scope"
        )
    _strict_int(
        budgets.get("max_raw_batch_bytes"),
        "development root raw batch cap",
        minimum=1,
    )

    output = _mapping(plan["output"], "development-content root output")
    _expect_keys(
        output,
        frozenset(
            {
                "namespace",
                "component_id",
                "write_mode",
                "existing_namespace_reuse_permitted",
            }
        ),
        "development-content root output",
    )
    if (
        output["namespace"] != namespace
        or output["component_id"] != DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID
        or output["write_mode"] != "create_new_exclusive"
        or output["existing_namespace_reuse_permitted"] is not False
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-content root output semantics changed"
        )

    scope = _mapping(plan["scope"], "development-content root authorization scope")
    _expect_keys(
        scope,
        frozenset(
            {
                "authorized_artifact_stage",
                "authorized_availability_window",
                "prohibited_artifact_stages",
                "pre_reveal_training_input",
                "reveal_request_required",
                "reveal_request_consumption_permitted",
                "outcome_access_permitted",
                "market_access_permitted",
                "model_access_permitted",
                "future_stage_access_permitted",
                "consumption_ledger_mutation_permitted",
            }
        ),
        "development-content root authorization scope",
    )
    window = _mapping(
        scope["authorized_availability_window"],
        "development-content root availability window",
    )
    if (
        window != {"first_session": "2000-01-01", "last_session": "2018-12-31"}
        or scope["authorized_artifact_stage"] != "development"
        or scope["prohibited_artifact_stages"] != ["intermediate", "final"]
        or scope["pre_reveal_training_input"] is not True
        or scope["reveal_request_required"] is not False
        or any(
            scope[field] is not False
            for field in (
                "reveal_request_consumption_permitted",
                "outcome_access_permitted",
                "market_access_permitted",
                "model_access_permitted",
                "future_stage_access_permitted",
                "consumption_ledger_mutation_permitted",
            )
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-content root authorization scope changed"
        )
    plan["development_content_root_plan_sha256"] = plan_hash
    return plan


def _development_registered_candidate(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    require_latest: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state, _ledger = _validated_store_snapshot(authenticated_store_snapshot)
    registry = _mapping(
        state["latest_registry"],
        "development root candidate registry",
    )
    entries = registry.get("entries")
    if type(entries) is not list or not entries:
        raise SecFilingGemmaStageAuthorizationError(
            "Development root requires one latest registered candidate"
        )
    registered_count = _strict_int(
        state["latest_registry_pin"].get("registered_entry_count"),
        "development root registered entry count",
        minimum=1,
    )
    if len(entries) != registered_count:
        raise SecFilingGemmaStageAuthorizationError(
            "Development root registry entry count is inconsistent"
        )
    root_scope = _mapping(plan["root_scope"], "development root plan scope")
    candidate_hash = _sha256(
        root_scope["candidate_sha256"],
        "development root candidate hash",
    )
    candidate_matches: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for index, raw_entry in enumerate(entries):
        entry = _mapping(
            raw_entry,
            f"development root registry entry {index + 1}",
        )
        candidate = entry.get("candidate_manifest")
        if (
            entry.get("candidate_sha256") == candidate_hash
            and entry.get("candidate_design_sha256")
            == root_scope["candidate_design_sha256"]
            and entry.get("attempt_id") == root_scope["attempt_id"]
            and type(candidate) is dict
        ):
            candidate_matches.append(
                (
                    index,
                    entry,
                    _mapping(
                        candidate,
                        "development root registered candidate manifest",
                    ),
                )
            )
    if len(candidate_matches) != 1:
        raise SecFilingGemmaStageAuthorizationError(
            "Development root plan lacks exactly one registered candidate"
        )
    entry_index, entry, candidate = candidate_matches[0]
    if require_latest and entry_index != len(entries) - 1:
        raise SecFilingGemmaStageAuthorizationError(
            "Development root plan is not bound to the latest registry candidate"
        )
    try:
        validated_candidate_hash = validate_candidate_manifest(
            candidate,
            expected_candidate_sha256=candidate_hash,
        )
    except Exception as exc:
        raise SecFilingGemmaStageAuthorizationError(
            "Development root candidate manifest is not canonical"
        ) from exc
    expected_identity = {
        "candidate_sha256": candidate_hash,
        "candidate_design_sha256": root_scope["candidate_design_sha256"],
        "attempt_id": root_scope["attempt_id"],
    }
    if (
        validated_candidate_hash != candidate_hash
        or any(entry.get(field) != value for field, value in expected_identity.items())
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development root plan crossed its registered candidate identity"
        )
    _sha256(entry.get("entry_sha256"), "development root registry entry hash")
    try:
        validate_development_content_root_plan(
            plan,
            expected_development_content_root_plan_sha256=plan[
                "development_content_root_plan_sha256"
            ],
            candidate_manifest=candidate,
            expected_candidate_sha256=candidate_hash,
            expected_candidate_design_sha256=root_scope[
                "candidate_design_sha256"
            ],
            expected_attempt_id=root_scope["attempt_id"],
            base_corpus_universe_sha256=candidate["bindings"][
                "corpus_universe_sha256"
            ],
            corpus_universe_manifest=plan["corpus_universe_manifest"],
            session_calendar_sha256=plan["corpus_provenance"][
                "session_calendar_sha256"
            ],
        )
    except Exception as exc:
        raise SecFilingGemmaStageAuthorizationError(
            "Development root plan failed its exact candidate-bound validation"
        ) from exc
    return state, entry, candidate


def _validated_development_sec_execution_claims(
    raw: Any,
    *,
    authenticated_store_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    claims = _mapping(raw, "current-tip development SEC execution claims")
    validated: dict[str, dict[str, Any]] = {}
    for raw_scope_sha256, raw_claim in claims.items():
        scope_hash = _sha256(
            raw_scope_sha256,
            "development SEC execution claim map key",
        )
        claim = _mapping(
            raw_claim,
            f"development SEC execution claim {scope_hash}",
        )
        _expect_keys(
            claim,
            _DEVELOPMENT_SEC_EXECUTION_CLAIM_KEYS,
            f"development SEC execution claim {scope_hash}",
        )
        if (
            claim["schema_version"]
            != DEVELOPMENT_SEC_EXECUTION_CLAIM_SCHEMA_VERSION
            or claim["contract_version"] != CONTRACT_VERSION
            or claim["claim_kind"]
            != "owned_development_sec_content_root"
            or claim["runner_repository_path"] != STAGE_RUNNER_REPOSITORY_PATH
            or claim["sec_corpus_repository_path"]
            != SEC_CORPUS_REPOSITORY_PATH
            or claim["sec_component_id"]
            != DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID
            or claim["output_write_mode"] != "create_new_exclusive"
            or claim["authorizes_outcome_access"] is not False
            or claim["market_access_permitted"] is not False
            or claim["model_access_permitted"] is not False
            or claim["future_stage_access_permitted"] is not False
            or claim["reveal_request_consumption_permitted"] is not False
            or claim["consumption_ledger_mutation_permitted"] is not False
            or claim["effect_may_be_repeated_after_indeterminate_crash"]
            is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC execution claim semantics changed"
            )
        _self_hash(claim, "claim_sha256", "development SEC execution claim")
        plan = _validated_development_content_root_plan(
            claim["development_content_root_plan"]
        )
        root_scope = plan["root_scope"]
        expected_plan_bindings = {
            "development_root_scope_sha256": scope_hash,
            "development_content_root_plan_sha256": plan[
                "development_content_root_plan_sha256"
            ],
            "attempt_id": root_scope["attempt_id"],
            "candidate_sha256": root_scope["candidate_sha256"],
            "candidate_design_sha256": root_scope[
                "candidate_design_sha256"
            ],
            "corpus_universe_sha256": root_scope["corpus_universe_sha256"],
            "corpus_universe_semantic_sha256": root_scope[
                "corpus_universe_semantic_sha256"
            ],
            "output_namespace": root_scope["output_namespace"],
            "sec_component_id": root_scope["component_id"],
        }
        if (
            plan["development_root_scope_sha256"] != scope_hash
            or any(
                claim[field] != value
                for field, value in expected_plan_bindings.items()
            )
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC execution claim crossed its exact root plan"
            )
        for field in (
            "development_root_scope_sha256",
            "development_content_root_plan_sha256",
            "candidate_sha256",
            "candidate_design_sha256",
            "registry_entry_sha256",
            "registry_sha256",
            "registry_tip_sha256",
            "corpus_universe_sha256",
            "corpus_universe_semantic_sha256",
            "start_current_tip_anchor_sha256",
            "start_state_sha256",
            "start_consumption_ledger_sha256",
            "start_consumption_ledger_tip_sha256",
            "runner_source_sha256",
            "sec_corpus_source_sha256",
            "execution_source_hashes_sha256",
        ):
            _sha256(claim[field], f"development SEC execution claim {field}")
        _safe_id(claim["attempt_id"], "development SEC execution attempt id")
        _strict_int(
            claim["registered_entry_count"],
            "development SEC execution registered entry count",
            minimum=1,
        )
        _strict_int(
            claim["start_consumed_request_count"],
            "development SEC execution consumed request count",
        )
        _tagged_sha256(
            claim["sec_user_agent_sha256"],
            "development SEC execution User-Agent hash",
        )
        raw_sources = _mapping(
            claim["execution_source_hashes"],
            "development SEC execution source hashes",
        )
        expected_roles = {role for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS}
        if set(raw_sources) != expected_roles:
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC execution source closure is incomplete"
            )
        sources = {
            role: _sha256(
                raw_sources[role],
                f"development SEC execution source hash {role}",
            )
            for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
        }
        if (
            claim["execution_source_hashes"] != sources
            or claim["execution_source_hashes_sha256"]
            != canonical_sha256(sources)
            or claim["execution_source_role_count"] != len(sources)
            or claim["runner_source_sha256"] != sources["runner"]
            or claim["sec_corpus_source_sha256"]
            != sources["sec_corpus_selector"]
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC execution source closure is inconsistent"
            )
        if authenticated_store_snapshot is not None:
            state, registry_entry, candidate = _development_registered_candidate(
                authenticated_store_snapshot,
                plan=plan,
                require_latest=False,
            )
            candidate_sources = _mapping(
                _mapping(
                    candidate.get("bindings"),
                    "development root candidate bindings",
                ).get("source_hashes"),
                "development root candidate source hashes",
            )
            if (
                claim["registry_entry_sha256"]
                != registry_entry["entry_sha256"]
            ) or any(
                candidate_sources.get(role) != source_hash
                for role, source_hash in sources.items()
            ):
                raise SecFilingGemmaStageAuthorizationError(
                    "Development SEC execution claim crossed its registered candidate"
                )
        validated[scope_hash] = claim
    return validated


def _validated_development_sec_reader_receipts(
    raw: Any,
    *,
    claims: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    receipts = _mapping(raw, "current-tip development SEC reader receipts")
    validated: dict[str, dict[str, Any]] = {}
    for raw_scope_sha256, raw_receipt in receipts.items():
        scope_hash = _sha256(
            raw_scope_sha256,
            "development SEC reader receipt map key",
        )
        receipt = _mapping(
            raw_receipt,
            f"development SEC reader receipt {scope_hash}",
        )
        _expect_keys(
            receipt,
            _DEVELOPMENT_SEC_READER_RECEIPT_KEYS,
            f"development SEC reader receipt {scope_hash}",
        )
        if (
            receipt["schema_version"]
            != DEVELOPMENT_SEC_READER_RECEIPT_SCHEMA_VERSION
            or receipt["contract_version"] != CONTRACT_VERSION
            or receipt["receipt_kind"]
            != "store_rehashed_owned_development_sec_content_root"
            or receipt["fresh_network_provenance_claimed"] is not False
            or receipt["reader_output_recomputed_by_store"] is not True
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC reader receipt semantics changed"
            )
        _self_hash(receipt, "receipt_sha256", "development SEC reader receipt")
        index = _validated_sec_byte_index(receipt["byte_index"])
        if (
            receipt["byte_index"] != index
            or receipt["byte_index_sha256"] != canonical_sha256(index)
            or receipt["byte_count_total"]
            != sum(item["byte_count"] for item in index)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC reader receipt byte index is inconsistent"
            )
        for field in (
            "development_root_scope_sha256",
            "claim_sha256",
            "development_content_root_plan_sha256",
            "candidate_sha256",
            "candidate_design_sha256",
            "registry_entry_sha256",
            "corpus_universe_sha256",
            "corpus_universe_semantic_sha256",
            "runner_source_sha256",
            "sec_corpus_source_sha256",
            "execution_source_hashes_sha256",
            "content_manifest_sha256",
            "byte_index_sha256",
            "complete_marker_sha256",
        ):
            _sha256(receipt[field], f"development SEC reader receipt {field}")
        _tagged_sha256(
            receipt["sec_user_agent_sha256"],
            "development SEC reader receipt User-Agent hash",
        )
        claim = claims.get(scope_hash)
        if type(claim) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC reader receipt lacks its execution claim"
            )
        expected = {
            "development_root_scope_sha256": scope_hash,
            "claim_sha256": claim.get("claim_sha256"),
            "development_content_root_plan_sha256": claim.get(
                "development_content_root_plan_sha256"
            ),
            "attempt_id": claim.get("attempt_id"),
            "candidate_sha256": claim.get("candidate_sha256"),
            "candidate_design_sha256": claim.get("candidate_design_sha256"),
            "registry_entry_sha256": claim.get("registry_entry_sha256"),
            "corpus_universe_sha256": claim.get("corpus_universe_sha256"),
            "corpus_universe_semantic_sha256": claim.get(
                "corpus_universe_semantic_sha256"
            ),
            "output_namespace": claim.get("output_namespace"),
            "sec_component_id": claim.get("sec_component_id"),
            "runner_source_sha256": claim.get("runner_source_sha256"),
            "sec_corpus_source_sha256": claim.get("sec_corpus_source_sha256"),
            "execution_source_hashes_sha256": claim.get(
                "execution_source_hashes_sha256"
            ),
            "execution_source_role_count": claim.get(
                "execution_source_role_count"
            ),
            "sec_user_agent_sha256": claim.get("sec_user_agent_sha256"),
        }
        if any(receipt[field] != value for field, value in expected.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC reader receipt crossed its execution claim"
            )
        validated[scope_hash] = receipt
    return validated


def _validated_development_sec_execution_aborts(
    raw: Any,
    *,
    claims: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    aborts = _mapping(raw, "current-tip development SEC execution aborts")
    validated: dict[str, dict[str, Any]] = {}
    for raw_scope_sha256, raw_abort in aborts.items():
        scope_hash = _sha256(
            raw_scope_sha256,
            "development SEC execution abort map key",
        )
        abort = _mapping(
            raw_abort,
            f"development SEC execution abort {scope_hash}",
        )
        _expect_keys(
            abort,
            _DEVELOPMENT_SEC_EXECUTION_ABORT_KEYS,
            f"development SEC execution abort {scope_hash}",
        )
        if (
            abort["schema_version"]
            != DEVELOPMENT_SEC_EXECUTION_ABORT_SCHEMA_VERSION
            or abort["contract_version"] != CONTRACT_VERSION
            or abort["abort_kind"]
            != "indeterminate_owned_development_sec_content_root"
            or abort["reason"]
            not in {
                "claim_recovered_without_terminal_receipt",
                "external_effect_failed_or_completion_unknown",
                "durable_output_verification_failed",
            }
            or abort["external_effect_retry_permitted"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC execution abort semantics changed"
            )
        _self_hash(abort, "abort_sha256", "development SEC execution abort")
        for field in (
            "development_root_scope_sha256",
            "claim_sha256",
            "development_content_root_plan_sha256",
            "candidate_sha256",
        ):
            _sha256(abort[field], f"development SEC execution abort {field}")
        claim = claims.get(scope_hash)
        if type(claim) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC execution abort lacks its execution claim"
            )
        expected = {
            "development_root_scope_sha256": scope_hash,
            "claim_sha256": claim.get("claim_sha256"),
            "development_content_root_plan_sha256": claim.get(
                "development_content_root_plan_sha256"
            ),
            "attempt_id": claim.get("attempt_id"),
            "candidate_sha256": claim.get("candidate_sha256"),
            "output_namespace": claim.get("output_namespace"),
            "sec_component_id": claim.get("sec_component_id"),
        }
        if any(abort[field] != value for field, value in expected.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC execution abort crossed its claim"
            )
        validated[scope_hash] = abort
    return validated


def _validated_development_root_carry_in_reader_receipts(
    raw: Any,
    *,
    authorization_bundles: Mapping[str, Any],
    sec_execution_claims: Mapping[str, Any],
    sec_reader_receipts: Mapping[str, Any],
    development_sec_execution_claims: Mapping[str, Any],
    development_sec_reader_receipts: Mapping[str, Any],
    development_sec_execution_aborts: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    receipts = _mapping(
        raw,
        "current-tip development-root carry-in reader receipts",
    )
    validated: dict[str, dict[str, Any]] = {}
    for raw_request_sha256, raw_receipt in receipts.items():
        request_hash = _sha256(
            raw_request_sha256,
            "development-root carry-in receipt map key",
        )
        receipt = _mapping(
            raw_receipt,
            f"development-root carry-in reader receipt {request_hash}",
        )
        _expect_keys(
            receipt,
            _DEVELOPMENT_ROOT_CARRY_IN_READER_RECEIPT_KEYS,
            f"development-root carry-in reader receipt {request_hash}",
        )
        if (
            receipt["schema_version"]
            != DEVELOPMENT_ROOT_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION
            or receipt["contract_version"] != CONTRACT_VERSION
            or receipt["receipt_kind"]
            != "store_rehashed_intermediate_child_development_root_carry_in"
            or receipt["request_sha256"] != request_hash
            or receipt["authorized_stage"] != "intermediate"
            or receipt["input_prerequisite_stage"] != "development"
            or receipt["reader_output_recomputed_by_store"] is not True
            or receipt["network_refetch_permitted"] is not False
            or receipt["write_permitted"] is not False
            or receipt["general_cross_stage_access_permitted"] is not False
            or receipt["fresh_carry_in_provenance_claimed"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development-root carry-in reader receipt semantics changed"
            )
        _self_hash(
            receipt,
            "receipt_sha256",
            "development-root carry-in reader receipt",
        )
        root_scope_hash = _sha256(
            receipt["development_root_scope_sha256"],
            "development-root carry-in root scope hash",
        )
        child_bundle = authorization_bundles.get(request_hash)
        child_claim = sec_execution_claims.get(request_hash)
        child_reader = sec_reader_receipts.get(request_hash)
        root_claim = development_sec_execution_claims.get(root_scope_hash)
        root_reader = development_sec_reader_receipts.get(root_scope_hash)
        if root_scope_hash in development_sec_execution_aborts:
            raise SecFilingGemmaStageAuthorizationError(
                "Development-root carry-in cannot descend from an aborted root"
            )
        if any(
            type(value) is not dict
            for value in (
                child_bundle,
                child_claim,
                child_reader,
                root_claim,
                root_reader,
            )
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development-root carry-in receipt lost its persisted ancestry"
            )
        expected = build_development_root_carry_in_reader_receipt(
            child_bundle,
            stage_sec_execution_claim=child_claim,
            stage_sec_reader_receipt=child_reader,
            development_sec_execution_claim=root_claim,
            development_sec_reader_receipt=root_reader,
            carry_in_byte_index=receipt["carry_in_byte_index"],
            carry_in_complete_marker_sha256=receipt[
                "carry_in_complete_marker_sha256"
            ],
            reader_source_sha256=receipt["reader_source_sha256"],
        )
        if receipt != expected:
            raise SecFilingGemmaStageAuthorizationError(
                "Development-root carry-in receipt differs from its exact ancestry"
            )
        validated[request_hash] = receipt
    return validated


def validate_reveal_store_current_tip_anchor_structure(
    current_tip_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a detached stable tip anchor without choosing a store state.

    This structural form is used by the effectful store while resolving an
    interrupted two-file transaction.  It does *not* authenticate a snapshot;
    callers authorizing downstream access must use
    :func:`validate_reveal_store_current_tip_anchor` instead.
    """

    anchor = _mapping(current_tip_anchor, "independent current-tip anchor")
    _expect_keys(anchor, _CURRENT_TIP_ANCHOR_KEYS, "independent current-tip anchor")
    if (
        anchor["schema_version"]
        != REVEAL_STORE_CURRENT_TIP_ANCHOR_SCHEMA_VERSION
        or anchor["contract_version"] != CONTRACT_VERSION
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Independent current-tip anchor schema or contract changed"
        )
    revision = _strict_int(anchor["revision"], "current-tip revision")
    previous = anchor["previous_tip_anchor_sha256"]
    if revision == 0:
        if previous is not None:
            raise SecFilingGemmaStageAuthorizationError(
                "Genesis current-tip anchor cannot have a predecessor"
            )
    else:
        _sha256(previous, "previous current-tip anchor hash")
    for field in (
        "state_sha256",
        "state_snapshot_bytes_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "consumption_ledger_sha256",
        "consumption_ledger_tip_sha256",
    ):
        _sha256(anchor[field], f"current-tip {field}")
    _strict_int(
        anchor["state_snapshot_byte_count"],
        "current-tip state snapshot byte count",
        minimum=1,
    )
    _strict_int(
        anchor["consumed_request_count"],
        "current-tip consumed request count",
    )
    anchor["trusted_stage_content_pins"] = _validated_trusted_stage_content_pins(
        anchor["trusted_stage_content_pins"]
    )
    anchor["authorization_bundles"] = _validated_authorization_bundles(
        anchor["authorization_bundles"]
    )
    anchor["stage_sec_execution_claims"] = _validated_stage_sec_execution_claims(
        anchor["stage_sec_execution_claims"],
        authorization_bundles=anchor["authorization_bundles"],
    )
    anchor["stage_sec_reader_receipts"] = _validated_stage_sec_reader_receipts(
        anchor["stage_sec_reader_receipts"],
        claims=anchor["stage_sec_execution_claims"],
    )
    anchor["stage_sec_execution_aborts"] = _validated_stage_sec_execution_aborts(
        anchor["stage_sec_execution_aborts"],
        claims=anchor["stage_sec_execution_claims"],
    )
    anchor["development_sec_execution_claims"] = (
        _validated_development_sec_execution_claims(
            anchor["development_sec_execution_claims"]
        )
    )
    anchor["development_sec_reader_receipts"] = (
        _validated_development_sec_reader_receipts(
            anchor["development_sec_reader_receipts"],
            claims=anchor["development_sec_execution_claims"],
        )
    )
    anchor["development_sec_execution_aborts"] = (
        _validated_development_sec_execution_aborts(
            anchor["development_sec_execution_aborts"],
            claims=anchor["development_sec_execution_claims"],
        )
    )
    anchor["consumed_stage_output_receipts"] = (
        _validated_consumed_stage_output_receipts(
            anchor["consumed_stage_output_receipts"],
            authorization_bundles=anchor["authorization_bundles"],
            sec_execution_claims=anchor["stage_sec_execution_claims"],
            sec_reader_receipts=anchor["stage_sec_reader_receipts"],
        )
    )
    anchor["stage_carry_in_reader_receipts"] = (
        _validated_stage_carry_in_reader_receipts(
            anchor["stage_carry_in_reader_receipts"],
            authorization_bundles=anchor["authorization_bundles"],
            sec_execution_claims=anchor["stage_sec_execution_claims"],
            sec_reader_receipts=anchor["stage_sec_reader_receipts"],
            consumed_stage_output_receipts=anchor[
                "consumed_stage_output_receipts"
            ],
        )
    )
    anchor["development_root_carry_in_reader_receipts"] = (
        _validated_development_root_carry_in_reader_receipts(
            anchor["development_root_carry_in_reader_receipts"],
            authorization_bundles=anchor["authorization_bundles"],
            sec_execution_claims=anchor["stage_sec_execution_claims"],
            sec_reader_receipts=anchor["stage_sec_reader_receipts"],
            development_sec_execution_claims=anchor[
                "development_sec_execution_claims"
            ],
            development_sec_reader_receipts=anchor[
                "development_sec_reader_receipts"
            ],
            development_sec_execution_aborts=anchor[
                "development_sec_execution_aborts"
            ],
        )
    )
    if set(anchor["stage_carry_in_reader_receipts"]) & set(
        anchor["development_root_carry_in_reader_receipts"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Final and development-root carry-in receipts cannot share a request"
        )
    if set(anchor["stage_sec_reader_receipts"]) & set(
        anchor["stage_sec_execution_aborts"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution cannot be both completed and aborted"
        )
    if set(anchor["development_sec_reader_receipts"]) & set(
        anchor["development_sec_execution_aborts"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development SEC execution cannot be both completed and aborted"
        )
    active_stage_claims = set(anchor["stage_sec_execution_claims"]) - set(
        anchor["stage_sec_reader_receipts"]
    ) - set(anchor["stage_sec_execution_aborts"])
    active_development_claims = set(
        anchor["development_sec_execution_claims"]
    ) - set(anchor["development_sec_reader_receipts"]) - set(
        anchor["development_sec_execution_aborts"]
    )
    if len(active_stage_claims) + len(active_development_claims) > 1:
        raise SecFilingGemmaStageAuthorizationError(
            "At most one SEC execution claim may be globally active"
        )
    _self_hash(anchor, "tip_anchor_sha256", "independent current-tip anchor")
    return anchor


def build_reveal_store_current_tip_anchor(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    revision: int,
    previous_tip_anchor_sha256: str | None,
    authorization_bundles: Mapping[str, Any],
    trusted_stage_content_pins: Mapping[str, Any] | None = None,
    consumed_stage_output_receipts: Mapping[str, Any] | None = None,
    stage_carry_in_reader_receipts: Mapping[str, Any] | None = None,
    stage_sec_execution_claims: Mapping[str, Any] | None = None,
    stage_sec_reader_receipts: Mapping[str, Any] | None = None,
    stage_sec_execution_aborts: Mapping[str, Any] | None = None,
    development_sec_execution_claims: Mapping[str, Any] | None = None,
    development_sec_reader_receipts: Mapping[str, Any] | None = None,
    development_sec_execution_aborts: Mapping[str, Any] | None = None,
    development_root_carry_in_reader_receipts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the separately persisted CAS anchor for one exact store state."""

    state, ledger = _validated_store_snapshot(authenticated_store_snapshot)
    current_revision = _strict_int(revision, "current-tip revision")
    if current_revision == 0:
        if previous_tip_anchor_sha256 is not None:
            raise SecFilingGemmaStageAuthorizationError(
                "Genesis current-tip anchor cannot have a predecessor"
            )
    else:
        _sha256(
            previous_tip_anchor_sha256,
            "previous current-tip anchor hash",
        )
    bundles = _validated_authorization_bundles(authorization_bundles)
    pins = _validated_trusted_stage_content_pins(
        {} if trusted_stage_content_pins is None else trusted_stage_content_pins
    )
    sec_claims = _validated_stage_sec_execution_claims(
        {} if stage_sec_execution_claims is None else stage_sec_execution_claims,
        authorization_bundles=bundles,
    )
    sec_receipts = _validated_stage_sec_reader_receipts(
        {} if stage_sec_reader_receipts is None else stage_sec_reader_receipts,
        claims=sec_claims,
    )
    sec_aborts = _validated_stage_sec_execution_aborts(
        {} if stage_sec_execution_aborts is None else stage_sec_execution_aborts,
        claims=sec_claims,
    )
    if set(sec_receipts) & set(sec_aborts):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution cannot be both completed and aborted"
        )
    development_claims = _validated_development_sec_execution_claims(
        (
            {}
            if development_sec_execution_claims is None
            else development_sec_execution_claims
        ),
        authenticated_store_snapshot=state,
    )
    development_receipts = _validated_development_sec_reader_receipts(
        (
            {}
            if development_sec_reader_receipts is None
            else development_sec_reader_receipts
        ),
        claims=development_claims,
    )
    development_aborts = _validated_development_sec_execution_aborts(
        (
            {}
            if development_sec_execution_aborts is None
            else development_sec_execution_aborts
        ),
        claims=development_claims,
    )
    if set(development_receipts) & set(development_aborts):
        raise SecFilingGemmaStageAuthorizationError(
            "Development SEC execution cannot be both completed and aborted"
        )
    active_stage_claim_count = len(
        set(sec_claims) - set(sec_receipts) - set(sec_aborts)
    )
    active_development_claim_count = len(
        set(development_claims)
        - set(development_receipts)
        - set(development_aborts)
    )
    if active_stage_claim_count + active_development_claim_count > 1:
        raise SecFilingGemmaStageAuthorizationError(
            "At most one SEC execution claim may be globally active"
        )
    output_receipts = _validated_consumed_stage_output_receipts(
        (
            {}
            if consumed_stage_output_receipts is None
            else consumed_stage_output_receipts
        ),
        authorization_bundles=bundles,
        sec_execution_claims=sec_claims,
        sec_reader_receipts=sec_receipts,
    )
    carry_in_receipts = _validated_stage_carry_in_reader_receipts(
        (
            {}
            if stage_carry_in_reader_receipts is None
            else stage_carry_in_reader_receipts
        ),
        authorization_bundles=bundles,
        sec_execution_claims=sec_claims,
        sec_reader_receipts=sec_receipts,
        consumed_stage_output_receipts=output_receipts,
    )
    development_root_carry_in_receipts = (
        _validated_development_root_carry_in_reader_receipts(
            (
                {}
                if development_root_carry_in_reader_receipts is None
                else development_root_carry_in_reader_receipts
            ),
            authorization_bundles=bundles,
            sec_execution_claims=sec_claims,
            sec_reader_receipts=sec_receipts,
            development_sec_execution_claims=development_claims,
            development_sec_reader_receipts=development_receipts,
            development_sec_execution_aborts=development_aborts,
        )
    )
    if set(carry_in_receipts) & set(development_root_carry_in_receipts):
        raise SecFilingGemmaStageAuthorizationError(
            "Final and development-root carry-in receipts cannot share a request"
        )
    state_bytes = _encoded_store_snapshot(state)
    body = {
        "schema_version": REVEAL_STORE_CURRENT_TIP_ANCHOR_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "revision": current_revision,
        "previous_tip_anchor_sha256": previous_tip_anchor_sha256,
        "state_sha256": state["state_sha256"],
        "state_snapshot_bytes_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "state_snapshot_byte_count": len(state_bytes),
        "registry_sha256": state["latest_registry"]["registry_sha256"],
        "registry_tip_sha256": state["latest_registry_pin"]["tip_sha256"],
        "consumption_ledger_sha256": ledger["ledger_sha256"],
        "consumption_ledger_tip_sha256": ledger["chain"]["tip_sha256"],
        "consumed_request_count": ledger["chain"]["consumed_request_count"],
        "trusted_stage_content_pins": pins,
        "authorization_bundles": bundles,
        "consumed_stage_output_receipts": output_receipts,
        "stage_carry_in_reader_receipts": carry_in_receipts,
        "stage_sec_execution_claims": sec_claims,
        "stage_sec_reader_receipts": sec_receipts,
        "stage_sec_execution_aborts": sec_aborts,
        "development_sec_execution_claims": development_claims,
        "development_sec_reader_receipts": development_receipts,
        "development_sec_execution_aborts": development_aborts,
        "development_root_carry_in_reader_receipts": (
            development_root_carry_in_receipts
        ),
    }
    return {**body, "tip_anchor_sha256": canonical_sha256(body)}


def validate_reveal_store_current_tip_anchor_transition(
    prior_tip_anchor: Mapping[str, Any],
    next_tip_anchor: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one monotonic, bundle-preserving CAS-anchor transition."""

    prior = validate_reveal_store_current_tip_anchor_structure(prior_tip_anchor)
    next_anchor = validate_reveal_store_current_tip_anchor_structure(next_tip_anchor)
    if (
        next_anchor["revision"] != prior["revision"] + 1
        or next_anchor["previous_tip_anchor_sha256"]
        != prior["tip_anchor_sha256"]
        or next_anchor["consumed_request_count"]
        - prior["consumed_request_count"]
        not in {0, 1}
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip anchor transition is stale, non-monotonic, or forked"
        )
    prior_bundles = prior["authorization_bundles"]
    next_bundles = next_anchor["authorization_bundles"]
    if any(next_bundles.get(key) != value for key, value in prior_bundles.items()):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip anchor transition removed or changed a persisted bundle"
        )
    bundle_delta = len(next_bundles) - len(prior_bundles)
    consumption_delta = (
        next_anchor["consumed_request_count"]
        - prior["consumed_request_count"]
    )
    if bundle_delta not in {0, 1} or bundle_delta > consumption_delta:
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition may append at most one authorization bundle"
        )
    prior_pins = prior["trusted_stage_content_pins"]
    next_pins = next_anchor["trusted_stage_content_pins"]
    if any(next_pins.get(key) != value for key, value in prior_pins.items()):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip anchor transition removed or changed a trusted content pin"
        )
    pin_delta = len(next_pins) - len(prior_pins)
    if pin_delta not in {0, 1}:
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition may append at most one trusted content pin"
        )
    if pin_delta:
        if consumption_delta != 0 or bundle_delta != 0:
            raise SecFilingGemmaStageAuthorizationError(
                "Trusted content pin append must be a dedicated tip-only transition"
            )
        immutable_state_fields = (
            "state_sha256",
            "state_snapshot_bytes_sha256",
            "state_snapshot_byte_count",
            "registry_sha256",
            "registry_tip_sha256",
            "consumption_ledger_sha256",
            "consumption_ledger_tip_sha256",
            "consumed_request_count",
        )
        if any(next_anchor[field] != prior[field] for field in immutable_state_fields):
            raise SecFilingGemmaStageAuthorizationError(
                "Trusted content pin append changed the authenticated store state"
            )
        new_request_hash = next(iter(set(next_pins) - set(prior_pins)))
        if next_pins[new_request_hash]["trusted_store_state_sha256"] != prior[
            "state_sha256"
        ]:
            raise SecFilingGemmaStageAuthorizationError(
                "Trusted content pin does not bind the pre-consumption state"
            )
    elif consumption_delta == 1 and set(next_pins) != set(prior_pins):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumption transition changed trusted content pin membership"
        )
    prior_outputs = prior["consumed_stage_output_receipts"]
    next_outputs = next_anchor["consumed_stage_output_receipts"]
    if any(next_outputs.get(key) != value for key, value in prior_outputs.items()):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip anchor transition removed or changed a consumed-stage output receipt"
        )
    output_delta = len(next_outputs) - len(prior_outputs)
    if output_delta not in {0, 1}:
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition may append at most one consumed-stage output receipt"
        )
    if output_delta:
        if consumption_delta != 0 or bundle_delta != 0 or pin_delta != 0:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt append must be a dedicated tip-only transition"
            )
        immutable_state_fields = (
            "state_sha256",
            "state_snapshot_bytes_sha256",
            "state_snapshot_byte_count",
            "registry_sha256",
            "registry_tip_sha256",
            "consumption_ledger_sha256",
            "consumption_ledger_tip_sha256",
            "consumed_request_count",
        )
        if any(next_anchor[field] != prior[field] for field in immutable_state_fields):
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt append changed the authenticated store state"
            )
        new_request_hash = next(iter(set(next_outputs) - set(prior_outputs)))
        if new_request_hash not in prior_bundles:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt was appended before its grant bundle"
            )
        new_receipt = next_outputs[new_request_hash]
        current_grant_bindings = {
            "grant_store_state_sha256": prior["state_sha256"],
            "grant_consumption_ledger_sha256": prior[
                "consumption_ledger_sha256"
            ],
            "grant_consumption_ledger_tip_sha256": prior[
                "consumption_ledger_tip_sha256"
            ],
            "consumption_entry_sha256": prior[
                "consumption_ledger_tip_sha256"
            ],
            "consumption_entry_sequence": prior["consumed_request_count"],
        }
        if any(
            new_receipt[field] != expected
            for field, expected in current_grant_bindings.items()
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt does not bind the exact current grant tip"
            )
    elif set(next_outputs) != set(prior_outputs):
        raise SecFilingGemmaStageAuthorizationError(
            "Non-output transition changed consumed-stage output receipt membership"
        )

    prior_carry_ins = prior["stage_carry_in_reader_receipts"]
    next_carry_ins = next_anchor["stage_carry_in_reader_receipts"]
    if any(
        next_carry_ins.get(key) != value
        for key, value in prior_carry_ins.items()
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition removed or changed a stage carry-in reader receipt"
        )
    carry_in_delta = len(next_carry_ins) - len(prior_carry_ins)
    if carry_in_delta not in {0, 1}:
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition may append at most one stage carry-in reader receipt"
        )
    if carry_in_delta:
        sec_maps_unchanged = all(
            next_anchor[name] == prior[name]
            for name in (
                "stage_sec_execution_claims",
                "stage_sec_reader_receipts",
                "stage_sec_execution_aborts",
            )
        )
        if (
            consumption_delta != 0
            or bundle_delta != 0
            or pin_delta != 0
            or output_delta != 0
            or not sec_maps_unchanged
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage carry-in receipt append must be a dedicated tip-only transition"
            )
        immutable_state_fields = (
            "state_sha256",
            "state_snapshot_bytes_sha256",
            "state_snapshot_byte_count",
            "registry_sha256",
            "registry_tip_sha256",
            "consumption_ledger_sha256",
            "consumption_ledger_tip_sha256",
            "consumed_request_count",
        )
        if any(next_anchor[field] != prior[field] for field in immutable_state_fields):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage carry-in receipt append changed the authenticated store state"
            )
        new_request_hash = next(iter(set(next_carry_ins) - set(prior_carry_ins)))
        receipt = next_carry_ins[new_request_hash]
        current_bindings = {
            "grant_store_state_sha256": prior["state_sha256"],
            "grant_consumption_ledger_sha256": prior[
                "consumption_ledger_sha256"
            ],
            "grant_consumption_ledger_tip_sha256": prior[
                "consumption_ledger_tip_sha256"
            ],
            "consumption_entry_sha256": prior[
                "consumption_ledger_tip_sha256"
            ],
            "consumption_entry_sequence": prior["consumed_request_count"],
        }
        if new_request_hash not in prior_bundles or any(
            receipt[field] != expected
            for field, expected in current_bindings.items()
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage carry-in receipt does not bind the exact current final grant tip"
            )
    elif set(next_carry_ins) != set(prior_carry_ins):
        raise SecFilingGemmaStageAuthorizationError(
            "Non-carry-in transition changed stage carry-in receipt membership"
        )

    prior_development_root_carry_ins = prior[
        "development_root_carry_in_reader_receipts"
    ]
    next_development_root_carry_ins = next_anchor[
        "development_root_carry_in_reader_receipts"
    ]
    if any(
        next_development_root_carry_ins.get(key) != value
        for key, value in prior_development_root_carry_ins.items()
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition removed or changed a development-root carry-in receipt"
        )
    development_root_carry_in_delta = len(
        next_development_root_carry_ins
    ) - len(prior_development_root_carry_ins)
    if development_root_carry_in_delta not in {0, 1}:
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition may append at most one development-root carry-in receipt"
        )
    if development_root_carry_in_delta:
        execution_maps_unchanged = all(
            next_anchor[name] == prior[name]
            for name in (
                "stage_sec_execution_claims",
                "stage_sec_reader_receipts",
                "stage_sec_execution_aborts",
                "development_sec_execution_claims",
                "development_sec_reader_receipts",
                "development_sec_execution_aborts",
            )
        )
        if (
            consumption_delta
            or bundle_delta
            or pin_delta
            or output_delta
            or carry_in_delta
            or not execution_maps_unchanged
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development-root carry-in append must be a dedicated tip-only transition"
            )
        immutable_state_fields = (
            "state_sha256",
            "state_snapshot_bytes_sha256",
            "state_snapshot_byte_count",
            "registry_sha256",
            "registry_tip_sha256",
            "consumption_ledger_sha256",
            "consumption_ledger_tip_sha256",
            "consumed_request_count",
        )
        if any(next_anchor[field] != prior[field] for field in immutable_state_fields):
            raise SecFilingGemmaStageAuthorizationError(
                "Development-root carry-in append changed authenticated store state"
            )
        request_hash = next(
            iter(
                set(next_development_root_carry_ins)
                - set(prior_development_root_carry_ins)
            )
        )
        receipt = next_development_root_carry_ins[request_hash]
        root_scope_hash = receipt["development_root_scope_sha256"]
        current_bindings = {
            "grant_store_state_sha256": prior["state_sha256"],
            "grant_consumption_ledger_sha256": prior[
                "consumption_ledger_sha256"
            ],
            "grant_consumption_ledger_tip_sha256": prior[
                "consumption_ledger_tip_sha256"
            ],
            "consumption_entry_sha256": prior[
                "consumption_ledger_tip_sha256"
            ],
            "consumption_entry_sequence": prior["consumed_request_count"],
        }
        if request_hash in prior_outputs:
            raise SecFilingGemmaStageAuthorizationError(
                "Development-root carry-in receipt must precede the same request's consumed-stage output receipt"
            )
        if (
            request_hash not in prior_bundles
            or request_hash not in prior["stage_sec_execution_claims"]
            or request_hash not in prior["stage_sec_reader_receipts"]
            or root_scope_hash not in prior["development_sec_execution_claims"]
            or root_scope_hash not in prior["development_sec_reader_receipts"]
            or root_scope_hash in prior["development_sec_execution_aborts"]
            or any(
                receipt[field] != expected
                for field, expected in current_bindings.items()
            )
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development-root carry-in does not bind prior terminal ancestry and the exact current grant tip"
            )
    elif set(next_development_root_carry_ins) != set(
        prior_development_root_carry_ins
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Non-carry-in transition changed development-root carry-in membership"
        )

    prior_development_claims = prior["development_sec_execution_claims"]
    next_development_claims = next_anchor["development_sec_execution_claims"]
    prior_development_receipts = prior["development_sec_reader_receipts"]
    next_development_receipts = next_anchor["development_sec_reader_receipts"]
    prior_development_aborts = prior["development_sec_execution_aborts"]
    next_development_aborts = next_anchor["development_sec_execution_aborts"]
    for prior_map, next_map, label in (
        (
            prior_development_claims,
            next_development_claims,
            "development SEC execution claim",
        ),
        (
            prior_development_receipts,
            next_development_receipts,
            "development SEC reader receipt",
        ),
        (
            prior_development_aborts,
            next_development_aborts,
            "development SEC execution abort",
        ),
    ):
        if any(next_map.get(key) != value for key, value in prior_map.items()):
            raise SecFilingGemmaStageAuthorizationError(
                f"Current-tip transition removed or changed a persisted {label}"
            )
    development_claim_delta = len(next_development_claims) - len(
        prior_development_claims
    )
    development_reader_delta = len(next_development_receipts) - len(
        prior_development_receipts
    )
    development_abort_delta = len(next_development_aborts) - len(
        prior_development_aborts
    )
    if any(
        delta not in {0, 1}
        for delta in (
            development_claim_delta,
            development_reader_delta,
            development_abort_delta,
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition may append at most one development SEC execution artifact"
        )
    development_delta_count = (
        development_claim_delta
        + development_reader_delta
        + development_abort_delta
    )
    if development_delta_count > 1:
        raise SecFilingGemmaStageAuthorizationError(
            "Development SEC claim, reader receipt, and abort require separate transitions"
        )

    prior_sec_claims = prior["stage_sec_execution_claims"]
    next_sec_claims = next_anchor["stage_sec_execution_claims"]
    prior_sec_receipts = prior["stage_sec_reader_receipts"]
    next_sec_receipts = next_anchor["stage_sec_reader_receipts"]
    prior_sec_aborts = prior["stage_sec_execution_aborts"]
    next_sec_aborts = next_anchor["stage_sec_execution_aborts"]
    for prior_map, next_map, label in (
        (prior_sec_claims, next_sec_claims, "SEC execution claim"),
        (prior_sec_receipts, next_sec_receipts, "SEC reader receipt"),
        (prior_sec_aborts, next_sec_aborts, "SEC execution abort"),
    ):
        if any(next_map.get(key) != value for key, value in prior_map.items()):
            raise SecFilingGemmaStageAuthorizationError(
                f"Current-tip transition removed or changed a persisted {label}"
            )
    claim_delta = len(next_sec_claims) - len(prior_sec_claims)
    reader_delta = len(next_sec_receipts) - len(prior_sec_receipts)
    abort_delta = len(next_sec_aborts) - len(prior_sec_aborts)
    if any(delta not in {0, 1} for delta in (claim_delta, reader_delta, abort_delta)):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition may append at most one SEC execution artifact"
        )
    sec_delta_count = claim_delta + reader_delta + abort_delta
    if sec_delta_count > 1:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC claim, reader receipt, and abort require separate transitions"
        )
    prior_active = set(prior_sec_claims) - set(prior_sec_receipts) - set(
        prior_sec_aborts
    )
    if prior_active:
        active_request = next(iter(prior_active))
        terminal_request: str | None = None
        if reader_delta:
            terminal_request = next(
                iter(set(next_sec_receipts) - set(prior_sec_receipts))
            )
        elif abort_delta:
            terminal_request = next(iter(set(next_sec_aborts) - set(prior_sec_aborts)))
        if (
            claim_delta
            or sec_delta_count != 1
            or terminal_request != active_request
            or consumption_delta
            or bundle_delta
            or pin_delta
            or output_delta
            or carry_in_delta
            or development_root_carry_in_delta
            or development_delta_count
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Active SEC execution claim blocks every transition except its exact terminal receipt"
            )
    if sec_delta_count:
        if (
            consumption_delta
            or bundle_delta
            or pin_delta
            or output_delta
            or carry_in_delta
            or development_root_carry_in_delta
            or development_delta_count
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution artifact append must be a dedicated tip-only transition"
            )
        immutable_state_fields = (
            "state_sha256",
            "state_snapshot_bytes_sha256",
            "state_snapshot_byte_count",
            "registry_sha256",
            "registry_tip_sha256",
            "consumption_ledger_sha256",
            "consumption_ledger_tip_sha256",
            "consumed_request_count",
        )
        if any(next_anchor[field] != prior[field] for field in immutable_state_fields):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution artifact append changed authenticated store state"
            )
    if claim_delta:
        request_hash = next(iter(set(next_sec_claims) - set(prior_sec_claims)))
        claim = next_sec_claims[request_hash]
        current_bindings = {
            "start_current_tip_anchor_sha256": prior["tip_anchor_sha256"],
            "grant_store_state_sha256": prior["state_sha256"],
            "grant_consumption_ledger_sha256": prior[
                "consumption_ledger_sha256"
            ],
            "grant_consumption_ledger_tip_sha256": prior[
                "consumption_ledger_tip_sha256"
            ],
            "consumption_entry_sha256": prior[
                "consumption_ledger_tip_sha256"
            ],
            "consumption_entry_sequence": prior["consumed_request_count"],
        }
        if request_hash not in prior_bundles or any(
            claim[field] != expected
            for field, expected in current_bindings.items()
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim does not bind the exact current grant tip"
            )
        expected_claim = build_stage_sec_execution_claim(
            prior_bundles[request_hash],
            independent_current_tip_anchor=prior,
            execution_source_hashes=claim["execution_source_hashes"],
            sec_user_agent_sha256=claim["sec_user_agent_sha256"],
        )
        if claim != expected_claim:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim differs from the exact authenticated component plan"
            )
    if not sec_delta_count:
        if (
            set(next_sec_claims) != set(prior_sec_claims)
            or set(next_sec_receipts) != set(prior_sec_receipts)
            or set(next_sec_aborts) != set(prior_sec_aborts)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Non-SEC transition changed SEC execution membership"
            )

    prior_active_development = set(prior_development_claims) - set(
        prior_development_receipts
    ) - set(prior_development_aborts)
    if prior_active_development:
        active_scope = next(iter(prior_active_development))
        terminal_scope: str | None = None
        if development_reader_delta:
            terminal_scope = next(
                iter(
                    set(next_development_receipts)
                    - set(prior_development_receipts)
                )
            )
        elif development_abort_delta:
            terminal_scope = next(
                iter(
                    set(next_development_aborts)
                    - set(prior_development_aborts)
                )
            )
        if (
            development_claim_delta
            or development_delta_count != 1
            or terminal_scope != active_scope
            or consumption_delta
            or bundle_delta
            or pin_delta
            or output_delta
            or carry_in_delta
            or development_root_carry_in_delta
            or sec_delta_count
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Active development SEC execution claim blocks every transition except its exact terminal receipt"
            )
    if development_delta_count:
        if (
            consumption_delta
            or bundle_delta
            or pin_delta
            or output_delta
            or carry_in_delta
            or development_root_carry_in_delta
            or sec_delta_count
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC execution artifact append must be a dedicated tip-only transition"
            )
        immutable_state_fields = (
            "state_sha256",
            "state_snapshot_bytes_sha256",
            "state_snapshot_byte_count",
            "registry_sha256",
            "registry_tip_sha256",
            "consumption_ledger_sha256",
            "consumption_ledger_tip_sha256",
            "consumed_request_count",
        )
        if any(next_anchor[field] != prior[field] for field in immutable_state_fields):
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC execution artifact append changed authenticated store state"
            )
    if development_claim_delta:
        scope_hash = next(
            iter(set(next_development_claims) - set(prior_development_claims))
        )
        claim = next_development_claims[scope_hash]
        current_bindings = {
            "development_root_scope_sha256": scope_hash,
            "start_current_tip_anchor_sha256": prior["tip_anchor_sha256"],
            "start_state_sha256": prior["state_sha256"],
            "start_consumption_ledger_sha256": prior[
                "consumption_ledger_sha256"
            ],
            "start_consumption_ledger_tip_sha256": prior[
                "consumption_ledger_tip_sha256"
            ],
            "start_consumed_request_count": prior["consumed_request_count"],
            "registry_sha256": prior["registry_sha256"],
            "registry_tip_sha256": prior["registry_tip_sha256"],
        }
        if any(
            claim[field] != expected
            for field, expected in current_bindings.items()
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development SEC execution claim does not bind the exact current registry and ledger tip"
            )
    if not development_delta_count:
        if (
            set(next_development_claims) != set(prior_development_claims)
            or set(next_development_receipts)
            != set(prior_development_receipts)
            or set(next_development_aborts) != set(prior_development_aborts)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Non-development transition changed development SEC execution membership"
            )
    return prior, next_anchor


def validate_reveal_store_current_tip_anchor(
    authenticated_store_snapshot: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a snapshot against the independently loaded latest tip."""

    state, _ledger = _validated_store_snapshot(authenticated_store_snapshot)
    observed = validate_reveal_store_current_tip_anchor_structure(
        independent_current_tip_anchor
    )
    expected = build_reveal_store_current_tip_anchor(
        state,
        revision=observed["revision"],
        previous_tip_anchor_sha256=observed["previous_tip_anchor_sha256"],
        authorization_bundles=observed["authorization_bundles"],
        trusted_stage_content_pins=observed["trusted_stage_content_pins"],
        consumed_stage_output_receipts=observed[
            "consumed_stage_output_receipts"
        ],
        stage_carry_in_reader_receipts=observed[
            "stage_carry_in_reader_receipts"
        ],
        stage_sec_execution_claims=observed["stage_sec_execution_claims"],
        stage_sec_reader_receipts=observed["stage_sec_reader_receipts"],
        stage_sec_execution_aborts=observed["stage_sec_execution_aborts"],
        development_sec_execution_claims=observed[
            "development_sec_execution_claims"
        ],
        development_sec_reader_receipts=observed[
            "development_sec_reader_receipts"
        ],
        development_sec_execution_aborts=observed[
            "development_sec_execution_aborts"
        ],
        development_root_carry_in_reader_receipts=observed[
            "development_root_carry_in_reader_receipts"
        ],
    )
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Independent current-tip anchor does not authenticate this exact latest state"
        )
    return observed


def _validated_latest_consumption(
    state: Mapping[str, Any], *, expected_entry_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ledger = state["consumption_ledger"]
    entries = ledger["entries"]
    expected_entry = _sha256(expected_entry_sha256, "expected consumption entry hash")
    if not entries:
        raise SecFilingGemmaStageAuthorizationError(
            "No consumed stage exists from which to mint a grant"
        )
    entry = entries[-1]
    if (
        entry["entry_sha256"] != expected_entry
        or ledger["chain"]["tip_sha256"] != expected_entry
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Grant entry is not the exact newly appended consumption-ledger tip"
        )
    request = _mapping(entry["request"], "consumed reveal request")
    _expect_keys(request, _REQUEST_KEYS, "consumed reveal request")
    request_hash = _self_hash(request, "request_sha256", "consumed reveal request")
    stage = request["stage"]
    prerequisite = request["prerequisite_stage"]
    if (
        request["schema_version"] != REVEAL_REQUEST_SCHEMA_VERSION
        or request["contract_version"] != CONTRACT_VERSION
        or _STAGE_PREREQUISITES.get(stage) != prerequisite
        or request["authorizes_outcome_access"] is not False
        or request["effectful_atomic_single_use_consumption_required"] is not True
        or request["cross_attempt_comparison_permitted"] is not False
        or request["cross_attempt_winner_selection_permitted"] is not False
        or request["globally_pristine_claim"] is not False
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumed request changed its frozen non-authorizing semantics"
        )
    for field in (
        "registry_sha256",
        "registry_tip_sha256",
        "stage_access_manifest_sha256",
        "prerequisite_stage_evidence_sha256",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
    ):
        _sha256(request[field], f"consumed request {field}")
    _safe_id(request["attempt_id"], "consumed request attempt id")
    if (
        entry["request_sha256"] != request_hash
        or entry["stage"] != stage
        or entry["attempt_id"] != request["attempt_id"]
        or entry["candidate_sha256"] != request["candidate_sha256"]
        or entry["registry_entry_sha256"] != request["registry_entry_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumption entry is not bound to its exact request"
        )

    access = _mapping(entry["stage_access_manifest"], "consumed stage-access manifest")
    access_hash = _self_hash(
        access,
        "stage_access_manifest_sha256",
        "consumed stage-access manifest",
    )
    transition = _mapping(access.get("transition"), "stage-access transition")
    candidate = _mapping(access.get("candidate"), "stage-access candidate")
    output = _mapping(access.get("output"), "stage-access output")
    scope = _mapping(access.get("scope"), "stage-access scope")
    _expect_keys(
        output,
        frozenset(
            {
                "namespace",
                "write_mode",
                "existing_namespace_reuse_permitted",
                "cross_stage_write_permitted",
            }
        ),
        "stage-access output",
    )
    expected_namespace = f"aapl-sec-gemma-{request['attempt_id']}-{stage}"
    namespace = output["namespace"]
    if (
        access.get("schema_version") != STAGE_ACCESS_MANIFEST_SCHEMA_VERSION
        or access.get("contract_version") != CONTRACT_VERSION
        or access_hash != request["stage_access_manifest_sha256"]
        or transition.get("prerequisite_stage") != prerequisite
        or transition.get("requested_stage") != stage
        or transition.get("single_use_consumption_required") is not True
        or transition.get("stage_reuse_permitted") is not False
        or candidate.get("candidate_sha256") != request["candidate_sha256"]
        or candidate.get("candidate_design_sha256")
        != request["candidate_design_sha256"]
        or candidate.get("attempt_id") != request["attempt_id"]
        or not isinstance(namespace, str)
        or _OUTPUT_NAMESPACE_RE.fullmatch(namespace) is None
        or namespace != expected_namespace
        or output["write_mode"] != "create_new_exclusive"
        or output["existing_namespace_reuse_permitted"] is not False
        or output["cross_stage_write_permitted"] is not False
        or scope.get("authorized_stage") != stage
        or scope.get("future_stage_access_permitted") is not False
        or scope.get("outcome_access_before_atomic_request_consumption_permitted")
        is not False
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumed stage-access manifest crossed an identity or namespace boundary"
        )

    validation = _mapping(
        entry["prerequisite_validation"], "consumed prerequisite validation"
    )
    _expect_keys(validation, _VALIDATION_KEYS, "consumed prerequisite validation")
    _self_hash(validation, "result_sha256", "consumed prerequisite validation")
    semantic_receipt = _mapping(
        validation["semantic_receipt"], "consumed semantic receipt"
    )
    checks = validation["semantic_checks"]
    if (
        validation["schema_version"] != SEMANTIC_PREREQUISITE_SCHEMA_VERSION
        or validation["validation_kind"]
        != "independent_semantic_prerequisite_replay"
        or validation["semantic_validation_completed"] is not True
        or validation["authorizes_outcome_access"] is not False
        or checks != list(REQUIRED_STAGE_VERIFIER_CHECKS)
        or not semantic_receipt
        or canonical_sha256(semantic_receipt)
        != validation["semantic_receipt_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumed prerequisite validation is not exact semantic evidence"
        )
    for request_field, validation_field in (
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
        if request[request_field] != validation[validation_field]:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed prerequisite validation lost its request binding"
            )
    if validation["requested_stage"] != stage:
        raise SecFilingGemmaStageAuthorizationError(
            "Consumed prerequisite validation crossed a stage"
        )
    return entry, request, access


def build_consumed_stage_authorization_grant(
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    external_store_state_pin: Mapping[str, Any],
    expected_new_consumption_entry_sha256: str,
) -> dict[str, Any]:
    """Mint a compact grant for the exact authenticated consumption-ledger tip."""

    state, ledger = _validated_store_snapshot(authenticated_store_snapshot)
    pin = validate_consumed_stage_store_state_pin(state, external_store_state_pin)
    entry, request, access = _validated_latest_consumption(
        state,
        expected_entry_sha256=expected_new_consumption_entry_sha256,
    )
    body = {
        "schema_version": CONSUMED_STAGE_AUTHORIZATION_GRANT_SCHEMA_VERSION,
        "authorization_kind": "exact_consumed_stage_output_namespace_access",
        "consumption_entry_sha256": entry["entry_sha256"],
        "consumption_entry_sequence": entry["sequence"],
        "request_sha256": request["request_sha256"],
        "attempt_id": request["attempt_id"],
        "candidate_sha256": request["candidate_sha256"],
        "registry_entry_sha256": request["registry_entry_sha256"],
        "prerequisite_stage": request["prerequisite_stage"],
        "stage": request["stage"],
        "prerequisite_stage_evidence_sha256": request[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": request["stage_access_manifest_sha256"],
        "output_namespace": access["output"]["namespace"],
        "store_state_sha256": state["state_sha256"],
        "store_snapshot_bytes_sha256": pin["store_snapshot_bytes_sha256"],
        "store_snapshot_byte_count": pin["store_snapshot_byte_count"],
        "store_pin_sha256": pin["store_pin_sha256"],
        "consumption_ledger_sha256": ledger["ledger_sha256"],
        "consumption_ledger_tip_sha256": ledger["chain"]["tip_sha256"],
        "consumed_stage_access_authorized": True,
        "authorization_scope": "exact_consumed_stage_output_namespace_only",
        "outcomes_included": False,
        "market_values_included": False,
        "cross_stage_access_permitted": False,
        "grant_reuse_across_store_state_tips_permitted": False,
    }
    return {**body, "authorization_grant_sha256": canonical_sha256(body)}


def validate_consumed_stage_authorization_grant(
    grant: Mapping[str, Any],
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    external_store_state_pin: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    expected_consumption_entry_sha256: str,
    expected_request_sha256: str,
    expected_candidate_sha256: str,
    expected_stage: str,
    expected_prerequisite_stage_evidence_sha256: str,
    expected_stage_access_manifest_sha256: str,
    expected_output_namespace: str,
) -> str:
    """Require an exact *current-tip* persisted grant and return its hash.

    The bundled state pin is deliberately insufficient as a trust root: an
    old snapshot and its old pin still agree with one another.  Every
    downstream authorization therefore also needs the independently loaded
    latest CAS anchor, and the exact reconstructed bundle must be present in
    that anchor.
    """

    observed = _mapping(grant, "consumed-stage authorization grant")
    _expect_keys(observed, _GRANT_KEYS, "consumed-stage authorization grant")
    grant_hash = _self_hash(
        observed,
        "authorization_grant_sha256",
        "consumed-stage authorization grant",
    )
    expected = build_consumed_stage_authorization_grant(
        authenticated_store_snapshot=authenticated_store_snapshot,
        external_store_state_pin=external_store_state_pin,
        expected_new_consumption_entry_sha256=expected_consumption_entry_sha256,
    )
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Authorization grant differs from the authenticated consumption entry"
        )
    current_anchor = validate_reveal_store_current_tip_anchor(
        authenticated_store_snapshot,
        independent_current_tip_anchor,
    )
    bundle_body = {
        "schema_version": CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION,
        "authenticated_store_snapshot": _mapping(
            authenticated_store_snapshot,
            "authenticated reveal-store snapshot",
        ),
        "store_state_pin": _mapping(
            external_store_state_pin,
            "external store-state pin",
        ),
        "authorization_grant": observed,
    }
    expected_bundle = {
        **bundle_body,
        "bundle_sha256": canonical_sha256(bundle_body),
    }
    if (
        current_anchor["authorization_bundles"].get(observed["request_sha256"])
        != expected_bundle
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Authorization grant bundle is not persisted at the independent current tip"
        )
    bindings = {
        "consumption_entry_sha256": _sha256(
            expected_consumption_entry_sha256, "expected consumption entry hash"
        ),
        "request_sha256": _sha256(expected_request_sha256, "expected request hash"),
        "candidate_sha256": _sha256(
            expected_candidate_sha256, "expected candidate hash"
        ),
        "stage": expected_stage,
        "prerequisite_stage_evidence_sha256": _sha256(
            expected_prerequisite_stage_evidence_sha256,
            "expected prerequisite-stage evidence hash",
        ),
        "stage_access_manifest_sha256": _sha256(
            expected_stage_access_manifest_sha256,
            "expected stage-access manifest hash",
        ),
        "output_namespace": expected_output_namespace,
    }
    if type(expected_stage) is not str or expected_stage not in _STAGE_PREREQUISITES:
        raise SecFilingGemmaStageAuthorizationError("Expected grant stage is invalid")
    if (
        type(expected_output_namespace) is not str
        or _OUTPUT_NAMESPACE_RE.fullmatch(expected_output_namespace) is None
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Expected output namespace is invalid"
        )
    for key, expected_value in bindings.items():
        if observed[key] != expected_value:
            raise SecFilingGemmaStageAuthorizationError(
                f"Authorization grant is not bound to expected {key}"
            )
    if (
        observed["consumed_stage_access_authorized"] is not True
        or observed["outcomes_included"] is not False
        or observed["market_values_included"] is not False
        or observed["cross_stage_access_permitted"] is not False
        or observed["grant_reuse_across_store_state_tips_permitted"] is not False
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Authorization grant changed its narrow no-outcome scope"
        )
    return grant_hash


def _sec_component_plan_from_bundle(
    authorization_bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_bundle = _mapping(authorization_bundle, "SEC execution authorization bundle")
    raw_grant = _mapping(
        raw_bundle.get("authorization_grant"),
        "SEC execution authorization grant",
    )
    request_hash = _sha256(
        raw_grant.get("request_sha256"),
        "SEC execution authorization request hash",
    )
    bundle = _validated_authorization_bundles({request_hash: raw_bundle})[
        request_hash
    ]
    grant = bundle["authorization_grant"]
    snapshot = bundle["authenticated_store_snapshot"]
    entry, _request, _access = _validated_latest_consumption(
        snapshot,
        expected_entry_sha256=grant["consumption_entry_sha256"],
    )
    access = _mapping(entry["stage_access_manifest"], "SEC execution stage access")
    access_body = {
        key: access[key]
        for key in access
        if key != "stage_access_manifest_sha256"
    }
    if (
        access.get("stage_access_manifest_sha256")
        != grant["stage_access_manifest_sha256"]
        or canonical_sha256(access_body) != grant["stage_access_manifest_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution stage-access manifest lost its self-hash"
        )
    sec_plan = _mapping(access.get("sec_access_plan"), "SEC execution access plan")
    _expect_keys(
        sec_plan,
        {
            "selection_policy",
            "method",
            "network_scope",
            "redirects_permitted",
            "retries_permitted",
            "cache_substitution_permitted",
            "document_count",
            "accessions_sha256",
            "official_urls_sha256",
            "documents",
        },
        "SEC execution access plan",
    )
    if (
        sec_plan["selection_policy"]
        != "all_and_only_requested_stage_universe_primary_documents"
        or sec_plan["method"] != "GET"
        or sec_plan["network_scope"] != "official_sec_https_only"
        or sec_plan["redirects_permitted"] is not False
        or sec_plan["retries_permitted"] is not False
        or sec_plan["cache_substitution_permitted"] is not False
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution access plan permits an unsafe transport"
        )
    documents = sec_plan["documents"]
    if type(documents) is not list or not documents:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution access plan has no exact document list"
        )
    normalized_documents: list[dict[str, str]] = []
    for index, raw_document in enumerate(documents):
        document = _mapping(raw_document, f"SEC execution document {index}")
        _expect_keys(
            document,
            {"accession_number", "official_url"},
            f"SEC execution document {index}",
        )
        accession = _safe_id(
            document["accession_number"], f"SEC execution accession {index}"
        )
        url = document["official_url"]
        expected_prefix = (
            "https://www.sec.gov/Archives/edgar/data/320193/"
            f"{accession.replace('-', '')}/"
        )
        if (
            _AAPL_ACCESSION_RE.fullmatch(accession) is None
            or type(url) is not str
            or not url.startswith(expected_prefix)
            or len(url) <= len(expected_prefix)
            or any(token in url for token in ("?", "#", "\\", ".."))
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution document URL is outside the fixed Apple archive"
            )
        normalized_documents.append(
            {"accession_number": accession, "official_url": url}
        )
    accessions = [document["accession_number"] for document in normalized_documents]
    official_urls = [document["official_url"] for document in normalized_documents]
    if (
        sec_plan["documents"] != normalized_documents
        or sec_plan["document_count"] != len(normalized_documents)
        or accessions != sorted(accessions)
        or official_urls != sorted(official_urls)
        or len(accessions) != len(set(accessions))
        or len(official_urls) != len(set(official_urls))
        or sec_plan["accessions_sha256"] != canonical_sha256(accessions)
        or sec_plan["official_urls_sha256"] != canonical_sha256(official_urls)
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution document plan is not canonical"
        )
    budgets = _mapping(access.get("budgets"), "SEC execution budgets")
    authorized_max_response_bytes = _strict_int(
        budgets.get("max_sec_response_bytes"),
        "SEC execution byte budget",
        minimum=1,
    )
    component_plan = {
        "authorized_stage": grant["stage"],
        "sec_access_plan": sec_plan,
        "max_sec_requests": _strict_int(
            budgets.get("max_sec_requests"),
            "SEC execution request budget",
            minimum=1,
        ),
        "authorized_max_sec_response_bytes": authorized_max_response_bytes,
        "owned_sec_raw_batch_max_bytes": OWNED_SEC_RAW_BATCH_MAX_BYTES,
        "max_sec_response_bytes": min(
            authorized_max_response_bytes,
            OWNED_SEC_RAW_BATCH_MAX_BYTES,
        ),
        "max_sec_acquisition_seconds": budgets.get(
            "max_sec_acquisition_seconds"
        ),
    }
    if component_plan["max_sec_requests"] != len(normalized_documents):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution request budget differs from its document plan"
        )
    return bundle, grant, component_plan


def build_development_sec_execution_claim(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    development_content_root_plan: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    execution_source_hashes: Mapping[str, Any],
    sec_user_agent_sha256: str,
) -> dict[str, Any]:
    """Claim one request-free development SEC root at the exact store tip."""

    plan = _validated_development_content_root_plan(
        development_content_root_plan
    )
    state, registry_entry, candidate = _development_registered_candidate(
        authenticated_store_snapshot,
        plan=plan,
        require_latest=True,
    )
    current_tip = validate_reveal_store_current_tip_anchor(
        state,
        independent_current_tip_anchor,
    )
    root_scope = plan["root_scope"]
    raw_sources = _mapping(
        execution_source_hashes,
        "owned development SEC execution source hashes",
    )
    expected_roles = {role for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS}
    if set(raw_sources) != expected_roles:
        raise SecFilingGemmaStageAuthorizationError(
            "Owned development SEC execution source closure is incomplete"
        )
    sources = {
        role: _sha256(
            raw_sources[role],
            f"owned development SEC execution source hash {role}",
        )
        for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
    }
    candidate_sources = _mapping(
        _mapping(
            candidate.get("bindings"),
            "development root candidate bindings",
        ).get("source_hashes"),
        "development root candidate source hashes",
    )
    if any(
        candidate_sources.get(role) != source_hash
        for role, source_hash in sources.items()
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Owned development SEC execution bytes differ from the registered candidate"
        )
    user_agent_hash = _tagged_sha256(
        sec_user_agent_sha256,
        "owned development SEC execution User-Agent hash",
    )
    registry_pin = state["latest_registry_pin"]
    ledger = state["consumption_ledger"]
    body = {
        "schema_version": DEVELOPMENT_SEC_EXECUTION_CLAIM_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "claim_kind": "owned_development_sec_content_root",
        "development_root_scope_sha256": plan[
            "development_root_scope_sha256"
        ],
        "development_content_root_plan_sha256": plan[
            "development_content_root_plan_sha256"
        ],
        "development_content_root_plan": plan,
        "attempt_id": root_scope["attempt_id"],
        "candidate_sha256": root_scope["candidate_sha256"],
        "candidate_design_sha256": root_scope["candidate_design_sha256"],
        "registry_entry_sha256": registry_entry["entry_sha256"],
        "registry_sha256": state["latest_registry"]["registry_sha256"],
        "registry_tip_sha256": registry_pin["tip_sha256"],
        "registered_entry_count": registry_pin["registered_entry_count"],
        "corpus_universe_sha256": root_scope["corpus_universe_sha256"],
        "corpus_universe_semantic_sha256": root_scope[
            "corpus_universe_semantic_sha256"
        ],
        "start_current_tip_anchor_sha256": current_tip["tip_anchor_sha256"],
        "start_state_sha256": state["state_sha256"],
        "start_consumption_ledger_sha256": ledger["ledger_sha256"],
        "start_consumption_ledger_tip_sha256": ledger["chain"]["tip_sha256"],
        "start_consumed_request_count": ledger["chain"][
            "consumed_request_count"
        ],
        "output_namespace": root_scope["output_namespace"],
        "output_write_mode": plan["output"]["write_mode"],
        "sec_component_id": root_scope["component_id"],
        "runner_repository_path": STAGE_RUNNER_REPOSITORY_PATH,
        "runner_source_sha256": sources["runner"],
        "sec_corpus_repository_path": SEC_CORPUS_REPOSITORY_PATH,
        "sec_corpus_source_sha256": sources["sec_corpus_selector"],
        "execution_source_hashes": sources,
        "execution_source_hashes_sha256": canonical_sha256(sources),
        "execution_source_role_count": len(sources),
        "sec_user_agent_sha256": user_agent_hash,
        "authorizes_outcome_access": False,
        "market_access_permitted": False,
        "model_access_permitted": False,
        "future_stage_access_permitted": False,
        "reveal_request_consumption_permitted": False,
        "consumption_ledger_mutation_permitted": False,
        "effect_may_be_repeated_after_indeterminate_crash": False,
    }
    return {**body, "claim_sha256": canonical_sha256(body)}


def build_development_sec_reader_receipt(
    claim: Mapping[str, Any],
    *,
    content_manifest_sha256: str,
    byte_index: list[dict[str, Any]],
    complete_marker_sha256: str,
) -> dict[str, Any]:
    """Bind store-rehashed root bytes and content manifest to one claim."""

    claim_value = _mapping(claim, "development SEC reader receipt claim")
    scope_hash = _sha256(
        claim_value.get("development_root_scope_sha256"),
        "development SEC reader receipt scope hash",
    )
    claim_value = _validated_development_sec_execution_claims(
        {scope_hash: claim_value}
    )[scope_hash]
    index = _validated_sec_byte_index(byte_index)
    body = {
        "schema_version": DEVELOPMENT_SEC_READER_RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "receipt_kind": "store_rehashed_owned_development_sec_content_root",
        "development_root_scope_sha256": scope_hash,
        "claim_sha256": claim_value["claim_sha256"],
        "development_content_root_plan_sha256": claim_value[
            "development_content_root_plan_sha256"
        ],
        "attempt_id": claim_value["attempt_id"],
        "candidate_sha256": claim_value["candidate_sha256"],
        "candidate_design_sha256": claim_value["candidate_design_sha256"],
        "registry_entry_sha256": claim_value["registry_entry_sha256"],
        "corpus_universe_sha256": claim_value["corpus_universe_sha256"],
        "corpus_universe_semantic_sha256": claim_value[
            "corpus_universe_semantic_sha256"
        ],
        "output_namespace": claim_value["output_namespace"],
        "sec_component_id": claim_value["sec_component_id"],
        "runner_source_sha256": claim_value["runner_source_sha256"],
        "sec_corpus_source_sha256": claim_value["sec_corpus_source_sha256"],
        "execution_source_hashes_sha256": claim_value[
            "execution_source_hashes_sha256"
        ],
        "execution_source_role_count": claim_value[
            "execution_source_role_count"
        ],
        "sec_user_agent_sha256": claim_value["sec_user_agent_sha256"],
        "content_manifest_sha256": _sha256(
            content_manifest_sha256,
            "development SEC content-manifest hash",
        ),
        "byte_index": index,
        "byte_index_sha256": canonical_sha256(index),
        "byte_count_total": sum(item["byte_count"] for item in index),
        "complete_marker_sha256": _sha256(
            complete_marker_sha256,
            "development SEC complete marker hash",
        ),
        "fresh_network_provenance_claimed": False,
        "reader_output_recomputed_by_store": True,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def build_development_sec_execution_abort(
    claim: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Terminally refuse retry after an indeterminate development SEC effect."""

    claim_value = _mapping(claim, "development SEC execution abort claim")
    scope_hash = _sha256(
        claim_value.get("development_root_scope_sha256"),
        "development SEC execution abort scope hash",
    )
    claim_value = _validated_development_sec_execution_claims(
        {scope_hash: claim_value}
    )[scope_hash]
    if reason not in {
        "claim_recovered_without_terminal_receipt",
        "external_effect_failed_or_completion_unknown",
        "durable_output_verification_failed",
    }:
        raise SecFilingGemmaStageAuthorizationError(
            "Development SEC execution abort reason is not canonical"
        )
    body = {
        "schema_version": DEVELOPMENT_SEC_EXECUTION_ABORT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "abort_kind": "indeterminate_owned_development_sec_content_root",
        "development_root_scope_sha256": scope_hash,
        "claim_sha256": claim_value["claim_sha256"],
        "development_content_root_plan_sha256": claim_value[
            "development_content_root_plan_sha256"
        ],
        "attempt_id": claim_value["attempt_id"],
        "candidate_sha256": claim_value["candidate_sha256"],
        "output_namespace": claim_value["output_namespace"],
        "sec_component_id": claim_value["sec_component_id"],
        "reason": reason,
        "external_effect_retry_permitted": False,
    }
    return {**body, "abort_sha256": canonical_sha256(body)}


def build_stage_sec_execution_claim(
    authorization_bundle: Mapping[str, Any],
    *,
    independent_current_tip_anchor: Mapping[str, Any],
    execution_source_hashes: Mapping[str, Any],
    sec_user_agent_sha256: str,
) -> dict[str, Any]:
    """Claim the exact latest grant before the owned SEC reader may run."""

    bundle, grant, component_plan = _sec_component_plan_from_bundle(
        authorization_bundle
    )
    registry = _mapping(
        bundle["authenticated_store_snapshot"].get("latest_registry"),
        "SEC execution candidate registry",
    )
    raw_entries = registry.get("entries")
    if type(raw_entries) is not list:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution candidate registry has no exact entries"
        )
    candidate_matches: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_entries):
        registry_entry = _mapping(
            raw_entry,
            f"SEC execution candidate registry entry {index}",
        )
        candidate = registry_entry.get("candidate_manifest")
        if (
            registry_entry.get("entry_sha256") == grant["registry_entry_sha256"]
            and type(candidate) is dict
            and candidate.get("candidate_sha256") == grant["candidate_sha256"]
        ):
            candidate_matches.append(candidate)
    if len(candidate_matches) != 1:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution grant lacks exactly one registry-pinned candidate"
        )
    candidate = candidate_matches[0]
    try:
        validated_candidate_hash = validate_candidate_manifest(
            candidate,
            expected_candidate_sha256=grant["candidate_sha256"],
        )
    except Exception as exc:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution candidate source pins are not canonical"
        ) from exc
    if validated_candidate_hash != grant["candidate_sha256"]:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution candidate identity changed"
        )
    candidate_sources = _mapping(
        _mapping(candidate.get("bindings"), "SEC execution candidate bindings").get(
            "source_hashes"
        ),
        "SEC execution candidate source hashes",
    )
    raw_execution_sources = _mapping(
        execution_source_hashes,
        "owned SEC execution source hashes",
    )
    expected_source_roles = {
        role for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
    }
    if set(raw_execution_sources) != expected_source_roles:
        raise SecFilingGemmaStageAuthorizationError(
            "Owned SEC execution source closure is incomplete"
        )
    execution_sources = {
        role: _sha256(
            raw_execution_sources[role],
            f"owned SEC execution source hash {role}",
        )
        for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
    }
    if any(
        candidate_sources.get(role) != source_hash
        for role, source_hash in execution_sources.items()
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Owned SEC execution bytes differ from the registered candidate"
        )
    runner_source_hash = execution_sources["runner"]
    corpus_source_hash = execution_sources["sec_corpus_selector"]
    user_agent_hash = _tagged_sha256(
        sec_user_agent_sha256,
        "owned SEC execution User-Agent hash",
    )
    current_tip = validate_reveal_store_current_tip_anchor(
        bundle["authenticated_store_snapshot"],
        independent_current_tip_anchor,
    )
    request_hash = grant["request_sha256"]
    if current_tip["authorization_bundles"].get(request_hash) != bundle:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution grant bundle is not exact at the current tip"
        )
    validate_consumed_stage_authorization_grant(
        grant,
        authenticated_store_snapshot=bundle["authenticated_store_snapshot"],
        external_store_state_pin=bundle["store_state_pin"],
        independent_current_tip_anchor=current_tip,
        expected_consumption_entry_sha256=grant["consumption_entry_sha256"],
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
    body = {
        "schema_version": STAGE_SEC_EXECUTION_CLAIM_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "claim_kind": "owned_sec_stage_document_batch",
        "request_sha256": request_hash,
        "consumption_entry_sha256": grant["consumption_entry_sha256"],
        "consumption_entry_sequence": grant["consumption_entry_sequence"],
        "attempt_id": grant["attempt_id"],
        "candidate_sha256": grant["candidate_sha256"],
        "registry_entry_sha256": grant["registry_entry_sha256"],
        "input_prerequisite_stage": grant["prerequisite_stage"],
        "authorized_stage": grant["stage"],
        "input_stage_evidence_sha256": grant[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": grant["stage_access_manifest_sha256"],
        "output_namespace": grant["output_namespace"],
        "authorization_bundle_sha256": bundle["bundle_sha256"],
        "authorization_grant_sha256": grant["authorization_grant_sha256"],
        "grant_store_state_sha256": grant["store_state_sha256"],
        "grant_consumption_ledger_sha256": grant[
            "consumption_ledger_sha256"
        ],
        "grant_consumption_ledger_tip_sha256": grant[
            "consumption_ledger_tip_sha256"
        ],
        "start_current_tip_anchor_sha256": current_tip["tip_anchor_sha256"],
        "runner_repository_path": STAGE_RUNNER_REPOSITORY_PATH,
        "runner_source_sha256": runner_source_hash,
        "sec_corpus_repository_path": SEC_CORPUS_REPOSITORY_PATH,
        "sec_corpus_source_sha256": corpus_source_hash,
        "execution_source_hashes": execution_sources,
        "execution_source_hashes_sha256": canonical_sha256(execution_sources),
        "execution_source_role_count": len(execution_sources),
        "sec_user_agent_sha256": user_agent_hash,
        "sec_component_id": SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID,
        "sec_component_plan_sha256": canonical_sha256(component_plan),
        "effect_may_be_repeated_after_indeterminate_crash": False,
    }
    return {**body, "claim_sha256": canonical_sha256(body)}


def build_stage_sec_reader_receipt(
    claim: Mapping[str, Any],
    *,
    byte_index: list[dict[str, Any]],
    complete_marker_sha256: str,
) -> dict[str, Any]:
    """Bind store-rehashed durable SEC bytes to one execution claim."""

    claim_value = _mapping(claim, "SEC reader receipt claim")
    _expect_keys(
        claim_value,
        _STAGE_SEC_EXECUTION_CLAIM_KEYS,
        "SEC reader receipt claim",
    )
    _self_hash(claim_value, "claim_sha256", "SEC reader receipt claim")
    index = _validated_sec_byte_index(byte_index)
    body = {
        "schema_version": STAGE_SEC_READER_RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "receipt_kind": "store_rehashed_owned_sec_stage_document_batch",
        "request_sha256": claim_value["request_sha256"],
        "claim_sha256": claim_value["claim_sha256"],
        "authorized_stage": claim_value["authorized_stage"],
        "candidate_sha256": claim_value["candidate_sha256"],
        "output_namespace": claim_value["output_namespace"],
        "authorization_bundle_sha256": claim_value[
            "authorization_bundle_sha256"
        ],
        "authorization_grant_sha256": claim_value[
            "authorization_grant_sha256"
        ],
        "sec_component_id": claim_value["sec_component_id"],
        "sec_component_plan_sha256": claim_value[
            "sec_component_plan_sha256"
        ],
        "runner_source_sha256": claim_value["runner_source_sha256"],
        "sec_corpus_source_sha256": claim_value["sec_corpus_source_sha256"],
        "execution_source_hashes_sha256": claim_value[
            "execution_source_hashes_sha256"
        ],
        "execution_source_role_count": claim_value[
            "execution_source_role_count"
        ],
        "sec_user_agent_sha256": claim_value["sec_user_agent_sha256"],
        "byte_index": index,
        "byte_index_sha256": canonical_sha256(index),
        "byte_count_total": sum(item["byte_count"] for item in index),
        "complete_marker_sha256": _sha256(
            complete_marker_sha256, "SEC complete marker hash"
        ),
        "fresh_network_provenance_claimed": False,
        "reader_output_recomputed_by_store": True,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def build_stage_sec_execution_abort(
    claim: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Terminally refuse any retry after a possibly executed SEC effect."""

    claim_value = _mapping(claim, "SEC execution abort claim")
    _expect_keys(
        claim_value,
        _STAGE_SEC_EXECUTION_CLAIM_KEYS,
        "SEC execution abort claim",
    )
    _self_hash(claim_value, "claim_sha256", "SEC execution abort claim")
    if reason not in {
        "claim_recovered_without_terminal_receipt",
        "external_effect_failed_or_completion_unknown",
        "durable_output_verification_failed",
    }:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution abort reason is not canonical"
        )
    body = {
        "schema_version": STAGE_SEC_EXECUTION_ABORT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "abort_kind": "indeterminate_owned_sec_stage_execution",
        "request_sha256": claim_value["request_sha256"],
        "claim_sha256": claim_value["claim_sha256"],
        "authorized_stage": claim_value["authorized_stage"],
        "candidate_sha256": claim_value["candidate_sha256"],
        "output_namespace": claim_value["output_namespace"],
        "reason": reason,
        "external_effect_retry_permitted": False,
    }
    return {**body, "abort_sha256": canonical_sha256(body)}


def _reconstruct_development_content_manifest_from_root(
    development_sec_execution_claim: Mapping[str, Any],
    development_sec_reader_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the development content identity from only persisted root ancestry."""

    root_claim = _mapping(
        development_sec_execution_claim,
        "development-root carry-in execution claim",
    )
    root_reader = _mapping(
        development_sec_reader_receipt,
        "development-root carry-in reader receipt",
    )
    plan = _mapping(
        root_claim.get("development_content_root_plan"),
        "development-root carry-in content plan",
    )
    sec_plan = _mapping(
        plan.get("sec_access_plan"),
        "development-root carry-in SEC plan",
    )
    documents_plan = sec_plan.get("documents")
    if type(documents_plan) is not list or not documents_plan:
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in has no exact root document plan"
        )
    root_index = _validated_sec_byte_index(root_reader.get("byte_index"))
    expected_layout: list[tuple[str, str]] = []
    for ordinal in range(1, len(documents_plan) + 1):
        prefix = f"document-{ordinal:04d}"
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
        (item["logical_id"], item["relative_path"]) for item in root_index
    ] != expected_layout:
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in ancestry has an inexact root byte layout"
        )
    rows_by_logical_id = {item["logical_id"]: item for item in root_index}
    content_documents: list[dict[str, Any]] = []
    for ordinal, raw_document in enumerate(documents_plan, start=1):
        document = _mapping(
            raw_document,
            f"development-root carry-in document {ordinal}",
        )
        raw_row = rows_by_logical_id[f"document-{ordinal:04d}-raw"]
        normalized_row = rows_by_logical_id[
            f"document-{ordinal:04d}-normalized"
        ]
        content_documents.append(
            {
                "accession_number": document.get("accession_number"),
                "primary_document_sha256": raw_row["sha256"],
                "normalized_text_sha256": normalized_row["sha256"],
                "primary_document_bytes": raw_row["byte_count"],
                "normalized_text_bytes": normalized_row["byte_count"],
            }
        )
    universe = _mapping(
        plan.get("corpus_universe_manifest"),
        "development-root carry-in corpus universe",
    )
    try:
        content_manifest = build_stage_content_manifest(
            artifact_stage="development",
            corpus_universe_sha256=root_claim["corpus_universe_sha256"],
            documents=content_documents,
            universe_manifest=universe,
        )
    except Exception as exc:
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in content manifest cannot be reconstructed"
        ) from exc
    if (
        content_manifest["content_manifest_sha256"]
        != root_reader.get("content_manifest_sha256")
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in content identity differs from its root receipt"
        )
    for logical_id, value, label in (
        (
            "corpus-universe-json",
            universe,
            "development-root carry-in universe bytes",
        ),
        (
            "development-content-manifest-json",
            content_manifest,
            "development-root carry-in content-manifest bytes",
        ),
    ):
        encoded = _encoded_store_snapshot(value)
        row = rows_by_logical_id[logical_id]
        if (
            row["byte_count"] != len(encoded)
            or row["sha256"] != hashlib.sha256(encoded).hexdigest()
        ):
            raise SecFilingGemmaStageAuthorizationError(
                f"{label} differ from the persisted root reader index"
            )
    return content_manifest


def build_development_root_carry_in_reader_receipt(
    authorization_bundle: Mapping[str, Any],
    *,
    stage_sec_execution_claim: Mapping[str, Any],
    stage_sec_reader_receipt: Mapping[str, Any],
    development_sec_execution_claim: Mapping[str, Any],
    development_sec_reader_receipt: Mapping[str, Any],
    carry_in_byte_index: list[dict[str, Any]],
    carry_in_complete_marker_sha256: str,
    reader_source_sha256: str,
) -> dict[str, Any]:
    """Bind an intermediate request's exact two-text read from its development root."""

    raw_bundle = _mapping(
        authorization_bundle,
        "development-root carry-in child authorization bundle",
    )
    raw_grant = _mapping(
        raw_bundle.get("authorization_grant"),
        "development-root carry-in child authorization grant",
    )
    request_hash = _sha256(
        raw_grant.get("request_sha256"),
        "development-root carry-in child request hash",
    )
    bundles = _validated_authorization_bundles({request_hash: raw_bundle})
    child_bundle = bundles[request_hash]
    child_grant = child_bundle["authorization_grant"]
    child_claim = _validated_stage_sec_execution_claims(
        {request_hash: stage_sec_execution_claim},
        authorization_bundles=bundles,
    )[request_hash]
    child_reader = _validated_stage_sec_reader_receipts(
        {request_hash: stage_sec_reader_receipt},
        claims={request_hash: child_claim},
    )[request_hash]
    root_claim_input = _mapping(
        development_sec_execution_claim,
        "development-root carry-in root claim",
    )
    root_scope_hash = _sha256(
        root_claim_input.get("development_root_scope_sha256"),
        "development-root carry-in root scope hash",
    )
    root_claim = _validated_development_sec_execution_claims(
        {root_scope_hash: root_claim_input},
        authenticated_store_snapshot=child_bundle["authenticated_store_snapshot"],
    )[root_scope_hash]
    root_reader = _validated_development_sec_reader_receipts(
        {root_scope_hash: development_sec_reader_receipt},
        claims={root_scope_hash: root_claim},
    )[root_scope_hash]
    child_entry, child_request, child_access = _validated_latest_consumption(
        child_bundle["authenticated_store_snapshot"],
        expected_entry_sha256=child_grant["consumption_entry_sha256"],
    )
    if (
        child_grant["stage"] != "intermediate"
        or child_grant["prerequisite_stage"] != "development"
        or child_request["stage"] != "intermediate"
        or child_request["prerequisite_stage"] != "development"
        or child_entry["stage"] != "intermediate"
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in is restricted to development-to-intermediate"
        )
    identity_bindings = {
        "attempt_id": root_claim["attempt_id"],
        "candidate_sha256": root_claim["candidate_sha256"],
        "candidate_design_sha256": root_claim["candidate_design_sha256"],
        "registry_entry_sha256": root_claim["registry_entry_sha256"],
        "registry_sha256": root_claim["registry_sha256"],
        "registry_tip_sha256": root_claim["registry_tip_sha256"],
        "registered_entry_count": root_claim["registered_entry_count"],
    }
    if any(
        child_request.get(field) != expected
        for field, expected in identity_bindings.items()
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in child crossed its registered root candidate"
        )
    child_access_candidate = _mapping(
        child_access.get("candidate"),
        "development-root carry-in child candidate",
    )
    if any(
        child_access_candidate.get(field) != identity_bindings[field]
        for field in ("attempt_id", "candidate_sha256", "candidate_design_sha256")
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in stage access crossed its root candidate"
        )
    child_ledger = child_bundle["authenticated_store_snapshot"][
        "consumption_ledger"
    ]
    child_entries = child_ledger["entries"]
    root_start_count = _strict_int(
        root_claim["start_consumed_request_count"],
        "development-root carry-in root start consumption count",
    )
    child_sequence = _strict_int(
        child_grant["consumption_entry_sequence"],
        "development-root carry-in child consumption sequence",
        minimum=1,
    )
    if root_start_count >= child_sequence or root_start_count > len(child_entries):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in root did not predate child consumption"
        )
    expected_root_start_tip = (
        child_ledger["chain"]["genesis_tip_sha256"]
        if root_start_count == 0
        else child_entries[root_start_count - 1]["entry_sha256"]
    )
    if root_claim["start_consumption_ledger_tip_sha256"] != expected_root_start_tip:
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in root is not bound to the child's exact earlier ledger prefix"
        )
    if child_claim["execution_source_hashes"] != root_claim[
        "execution_source_hashes"
    ]:
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in child changed its registered execution bytes"
        )
    root_plan = root_claim["development_content_root_plan"]
    root_provenance = root_plan["corpus_provenance"]
    child_provenance = _mapping(
        child_access.get("corpus_provenance"),
        "development-root carry-in child corpus provenance",
    )
    frozen_base = _mapping(
        child_provenance.get("frozen_base_universe"),
        "development-root carry-in child frozen universe",
    )
    expected_frozen_base = {
        "corpus_universe_sha256": root_claim["corpus_universe_sha256"],
        "corpus_universe_semantic_sha256": root_claim[
            "corpus_universe_semantic_sha256"
        ],
        "sec_catalog_artifact_sha256": root_provenance[
            "sec_catalog_artifact_sha256"
        ],
        "calendar_source_evidence_sha256": root_provenance[
            "calendar_source_evidence_sha256"
        ],
        "session_calendar_sha256": root_provenance["session_calendar_sha256"],
    }
    if any(
        frozen_base.get(field) != expected
        for field, expected in expected_frozen_base.items()
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in child changed its frozen corpus identity"
        )
    content_manifest = _reconstruct_development_content_manifest_from_root(
        root_claim,
        root_reader,
    )
    evidence_pin = _mapping(
        child_access.get("prerequisite_evidence_pin"),
        "development-root carry-in prerequisite evidence pin",
    )
    _expect_keys(
        evidence_pin,
        {
            "stage",
            "content_manifest_sha256",
            "stage_artifact_sha256",
            "external_seal_receipt_sha256",
        },
        "development-root carry-in prerequisite evidence pin",
    )
    if (
        evidence_pin["stage"] != "development"
        or evidence_pin["content_manifest_sha256"]
        != content_manifest["content_manifest_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in prerequisite pin differs from the root content"
        )
    try:
        records = validate_prior_same_form_carry_in_scope(
            child_access,
            corpus_universe_manifest=root_plan["corpus_universe_manifest"],
            prerequisite_content_manifest=content_manifest,
            expected_prerequisite_stage="development",
            expected_requested_stage="intermediate",
            expected_prerequisite_stage_evidence_sha256=child_grant[
                "prerequisite_stage_evidence_sha256"
            ],
        )
    except Exception as exc:
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in scope is not the exact canonical two-text read"
        ) from exc
    if len(records) != 2 or [record.get("form") for record in records] != [
        "10-K",
        "10-Q",
    ]:
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in must contain exactly 10-K then 10-Q"
        )
    carry_plan = _mapping(
        child_access.get("prior_same_form_carry_in"),
        "development-root carry-in access plan",
    )
    documents_plan = root_plan["sec_access_plan"]["documents"]
    source_ordinal_by_accession = {
        document["accession_number"]: ordinal
        for ordinal, document in enumerate(documents_plan, start=1)
    }
    root_rows_by_logical_id = {
        item["logical_id"]: item for item in root_reader["byte_index"]
    }
    expected_copy_index: list[dict[str, Any]] = []
    for copy_ordinal, record in enumerate(records, start=1):
        source_ordinal = source_ordinal_by_accession.get(record["accession_number"])
        source_row = root_rows_by_logical_id.get(
            (
                ""
                if type(source_ordinal) is not int
                else f"document-{source_ordinal:04d}-normalized"
            )
        )
        if (
            type(source_ordinal) is not int
            or type(source_row) is not dict
            or source_row.get("relative_path")
            != f"document-{source_ordinal:04d}.normalized.txt"
            or source_row.get("sha256") != record["normalized_text_sha256"]
            or source_row.get("byte_count") != record["normalized_text_bytes"]
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development-root carry-in bytes differ from the persisted root"
            )
        expected_copy_index.append(
            {
                "ordinal": copy_ordinal,
                "logical_id": f"carry-in-{copy_ordinal:04d}-normalized",
                "relative_path": f"carry-in-{copy_ordinal:04d}.normalized.txt",
                "byte_count": source_row["byte_count"],
                "sha256": source_row["sha256"],
            }
        )
    copied_index = _validated_sec_byte_index(carry_in_byte_index)
    if copied_index != expected_copy_index:
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in copied-byte index differs from its root bytes"
        )
    marker_hash = _sha256(
        carry_in_complete_marker_sha256,
        "development-root carry-in complete-marker hash",
    )
    source_hash = _sha256(
        reader_source_sha256,
        "development-root carry-in reader source hash",
    )
    if source_hash != child_claim["execution_source_hashes"].get("reveal_store"):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in reader source differs from child execution bytes"
        )
    body = {
        "schema_version": DEVELOPMENT_ROOT_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "receipt_kind": (
            "store_rehashed_intermediate_child_development_root_carry_in"
        ),
        "request_sha256": request_hash,
        "consumption_entry_sha256": child_grant["consumption_entry_sha256"],
        "consumption_entry_sequence": child_grant["consumption_entry_sequence"],
        "attempt_id": child_grant["attempt_id"],
        "candidate_sha256": child_grant["candidate_sha256"],
        "candidate_design_sha256": child_request["candidate_design_sha256"],
        "registry_entry_sha256": child_grant["registry_entry_sha256"],
        "registry_sha256": child_request["registry_sha256"],
        "registry_tip_sha256": child_request["registry_tip_sha256"],
        "registered_entry_count": child_request["registered_entry_count"],
        "input_prerequisite_stage": child_grant["prerequisite_stage"],
        "authorized_stage": child_grant["stage"],
        "input_stage_evidence_sha256": child_grant[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": child_grant[
            "stage_access_manifest_sha256"
        ],
        "output_namespace": child_grant["output_namespace"],
        "authorization_bundle_sha256": child_bundle["bundle_sha256"],
        "authorization_grant_sha256": child_grant["authorization_grant_sha256"],
        "grant_store_state_sha256": child_grant["store_state_sha256"],
        "grant_consumption_ledger_sha256": child_grant[
            "consumption_ledger_sha256"
        ],
        "grant_consumption_ledger_tip_sha256": child_grant[
            "consumption_ledger_tip_sha256"
        ],
        "child_sec_execution_claim_sha256": child_claim["claim_sha256"],
        "child_sec_reader_receipt_sha256": child_reader["receipt_sha256"],
        "development_root_scope_sha256": root_scope_hash,
        "development_sec_execution_claim_sha256": root_claim["claim_sha256"],
        "development_sec_reader_receipt_sha256": root_reader["receipt_sha256"],
        "development_content_root_plan_sha256": root_claim[
            "development_content_root_plan_sha256"
        ],
        "development_content_manifest_sha256": content_manifest[
            "content_manifest_sha256"
        ],
        "corpus_universe_sha256": root_claim["corpus_universe_sha256"],
        "corpus_universe_semantic_sha256": root_claim[
            "corpus_universe_semantic_sha256"
        ],
        "prerequisite_stage_artifact_sha256": evidence_pin[
            "stage_artifact_sha256"
        ],
        "prerequisite_external_seal_receipt_sha256": evidence_pin[
            "external_seal_receipt_sha256"
        ],
        "selection_policy": carry_plan["selection_policy"],
        "artifact_scope": carry_plan["artifact_scope"],
        "carry_in_record_count": len(records),
        "carry_in_records_sha256": canonical_sha256(records),
        "carry_in_records": records,
        "carry_in_byte_index": copied_index,
        "carry_in_byte_index_sha256": canonical_sha256(copied_index),
        "carry_in_byte_count_total": sum(
            item["byte_count"] for item in copied_index
        ),
        "carry_in_complete_marker_sha256": marker_hash,
        "reader_source_sha256": source_hash,
        "reader_output_recomputed_by_store": True,
        "network_refetch_permitted": False,
        "write_permitted": False,
        "general_cross_stage_access_permitted": False,
        "fresh_carry_in_provenance_claimed": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def validate_development_root_carry_in_reader_receipt(
    receipt: Mapping[str, Any],
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    carry_in_byte_index: list[dict[str, Any]],
    carry_in_complete_marker_sha256: str,
    reader_source_sha256: str,
) -> str:
    """Require current-tip root ancestry and revalidate the copied development bytes."""

    observed = _mapping(receipt, "development-root carry-in reader receipt")
    _expect_keys(
        observed,
        _DEVELOPMENT_ROOT_CARRY_IN_READER_RECEIPT_KEYS,
        "development-root carry-in reader receipt",
    )
    observed_hash = _self_hash(
        observed,
        "receipt_sha256",
        "development-root carry-in reader receipt",
    )
    current_tip = validate_reveal_store_current_tip_anchor(
        authenticated_store_snapshot,
        independent_current_tip_anchor,
    )
    request_hash = _sha256(
        observed.get("request_sha256"),
        "development-root carry-in child request hash",
    )
    root_scope_hash = _sha256(
        observed.get("development_root_scope_sha256"),
        "development-root carry-in root scope hash",
    )
    if current_tip["development_root_carry_in_reader_receipts"].get(
        request_hash
    ) != observed:
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in receipt is not persisted at the current tip"
        )
    if root_scope_hash in current_tip["development_sec_execution_aborts"]:
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in cannot descend from an aborted root"
        )
    child_bundle = current_tip["authorization_bundles"].get(request_hash)
    child_claim = current_tip["stage_sec_execution_claims"].get(request_hash)
    child_reader = current_tip["stage_sec_reader_receipts"].get(request_hash)
    root_claim = current_tip["development_sec_execution_claims"].get(
        root_scope_hash
    )
    root_reader = current_tip["development_sec_reader_receipts"].get(
        root_scope_hash
    )
    if any(
        type(value) is not dict
        for value in (
            child_bundle,
            child_claim,
            child_reader,
            root_claim,
            root_reader,
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in receipt lost its persisted ancestry"
        )
    expected = build_development_root_carry_in_reader_receipt(
        child_bundle,
        stage_sec_execution_claim=child_claim,
        stage_sec_reader_receipt=child_reader,
        development_sec_execution_claim=root_claim,
        development_sec_reader_receipt=root_reader,
        carry_in_byte_index=carry_in_byte_index,
        carry_in_complete_marker_sha256=carry_in_complete_marker_sha256,
        reader_source_sha256=reader_source_sha256,
    )
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Development-root carry-in receipt differs from revalidated durable inputs"
        )
    return observed_hash


def build_stage_carry_in_reader_receipt(
    authorization_bundle: Mapping[str, Any],
    *,
    stage_sec_execution_claim: Mapping[str, Any],
    stage_sec_reader_receipt: Mapping[str, Any],
    parent_authorization_bundle: Mapping[str, Any],
    parent_stage_sec_execution_claim: Mapping[str, Any],
    parent_stage_sec_reader_receipt: Mapping[str, Any],
    parent_consumed_stage_output_receipt: Mapping[str, Any],
    carry_in_byte_index: list[dict[str, Any]],
    carry_in_complete_marker_sha256: str,
    reader_source_sha256: str,
) -> dict[str, Any]:
    """Bind one exact final-child copy of the two parent carry-in texts."""

    raw_child_bundle = _mapping(
        authorization_bundle,
        "final carry-in child authorization bundle",
    )
    raw_child_grant = _mapping(
        raw_child_bundle.get("authorization_grant"),
        "final carry-in child authorization grant",
    )
    child_request_hash = _sha256(
        raw_child_grant.get("request_sha256"),
        "final carry-in child request hash",
    )
    raw_parent_bundle = _mapping(
        parent_authorization_bundle,
        "final carry-in parent authorization bundle",
    )
    raw_parent_grant = _mapping(
        raw_parent_bundle.get("authorization_grant"),
        "final carry-in parent authorization grant",
    )
    parent_request_hash = _sha256(
        raw_parent_grant.get("request_sha256"),
        "final carry-in parent request hash",
    )
    if child_request_hash == parent_request_hash:
        raise SecFilingGemmaStageAuthorizationError(
            "Final carry-in child cannot reuse its parent request"
        )
    bundles = _validated_authorization_bundles(
        {
            child_request_hash: raw_child_bundle,
            parent_request_hash: raw_parent_bundle,
        }
    )
    child_bundle = bundles[child_request_hash]
    parent_bundle = bundles[parent_request_hash]
    child_grant = child_bundle["authorization_grant"]
    parent_grant = parent_bundle["authorization_grant"]
    claims = _validated_stage_sec_execution_claims(
        {
            child_request_hash: stage_sec_execution_claim,
            parent_request_hash: parent_stage_sec_execution_claim,
        },
        authorization_bundles=bundles,
    )
    readers = _validated_stage_sec_reader_receipts(
        {
            child_request_hash: stage_sec_reader_receipt,
            parent_request_hash: parent_stage_sec_reader_receipt,
        },
        claims=claims,
    )
    outputs = _validated_consumed_stage_output_receipts(
        {parent_request_hash: parent_consumed_stage_output_receipt},
        authorization_bundles=bundles,
        sec_execution_claims=claims,
        sec_reader_receipts=readers,
    )
    child_claim = claims[child_request_hash]
    child_reader = readers[child_request_hash]
    parent_claim = claims[parent_request_hash]
    parent_reader = readers[parent_request_hash]
    parent_output = outputs[parent_request_hash]
    child_entry, child_request, child_access = _validated_latest_consumption(
        child_bundle["authenticated_store_snapshot"],
        expected_entry_sha256=child_grant["consumption_entry_sha256"],
    )
    parent_entry, parent_request, _parent_access = _validated_latest_consumption(
        parent_bundle["authenticated_store_snapshot"],
        expected_entry_sha256=parent_grant["consumption_entry_sha256"],
    )
    child_entries = child_bundle["authenticated_store_snapshot"][
        "consumption_ledger"
    ]["entries"]
    if (
        len(child_entries) < 2
        or child_entries[-2] != parent_entry
        or child_entry["sequence"] != parent_entry["sequence"] + 1
        or child_entry["prior_tip_sha256"] != parent_entry["entry_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Final carry-in parent is not the exact immediate consumed predecessor"
        )
    if (
        child_grant["stage"] != "final"
        or child_grant["prerequisite_stage"] != "intermediate"
        or parent_grant["stage"] != "intermediate"
        or parent_grant["prerequisite_stage"] != "development"
        or child_request["stage"] != "final"
        or parent_request["stage"] != "intermediate"
        or child_entry["stage"] != "final"
        or parent_entry["stage"] != "intermediate"
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Carry-in receipt is restricted to one intermediate-to-final transition"
        )
    for field in (
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "registry_sha256",
        "registry_tip_sha256",
    ):
        if child_request[field] != parent_request[field]:
            raise SecFilingGemmaStageAuthorizationError(
                "Final carry-in parent and child changed candidate identity"
            )
    if (
        child_grant["prerequisite_stage_evidence_sha256"]
        != parent_output["output_stage_evidence_sha256"]
        or parent_output["output_stage"] != "intermediate"
        or parent_output["output_candidate_sha256"]
        != child_grant["candidate_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Final carry-in child does not consume the exact parent stage evidence"
        )

    evidence_pin = _mapping(
        child_access.get("prerequisite_evidence_pin"),
        "final carry-in parent evidence pin",
    )
    _expect_keys(
        evidence_pin,
        {
            "stage",
            "content_manifest_sha256",
            "stage_artifact_sha256",
            "external_seal_receipt_sha256",
        },
        "final carry-in parent evidence pin",
    )
    carry_plan = _mapping(
        child_access.get("prior_same_form_carry_in"),
        "final carry-in access plan",
    )
    _expect_keys(
        carry_plan,
        {
            "selection_policy",
            "artifact_scope",
            "network_refetch_permitted",
            "write_permitted",
            "bound_by_prerequisite_stage_evidence_sha256",
            "prerequisite_content_manifest_sha256",
            "record_count",
            "records_sha256",
            "records",
        },
        "final carry-in access plan",
    )
    scope = _mapping(child_access.get("scope"), "final carry-in access scope")
    if (
        evidence_pin["stage"] != "intermediate"
        or carry_plan["selection_policy"]
        != "latest_prerequisite_stage_filing_of_each_requested_stage_form"
        or carry_plan["artifact_scope"] != "sealed_normalized_text_only"
        or carry_plan["network_refetch_permitted"] is not False
        or carry_plan["write_permitted"] is not False
        or carry_plan["bound_by_prerequisite_stage_evidence_sha256"]
        != child_grant["prerequisite_stage_evidence_sha256"]
        or carry_plan["prerequisite_content_manifest_sha256"]
        != evidence_pin["content_manifest_sha256"]
        or scope.get("general_cross_stage_access_permitted") is not False
        or scope.get("exact_prior_same_form_carry_in_read_permitted") is not True
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Final carry-in access plan changed its exact read-only scope"
        )
    records = _validated_carry_in_records(
        carry_plan["records"],
        expected_stage="intermediate",
        expected_content_manifest_sha256=evidence_pin["content_manifest_sha256"],
    )
    if (
        carry_plan["records"] != records
        or carry_plan["record_count"] != len(records)
        or carry_plan["records_sha256"] != canonical_sha256(records)
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Final carry-in record set differs from its stage-access plan"
        )

    _exact_parent_bundle, _exact_parent_grant, parent_component_plan = (
        _sec_component_plan_from_bundle(parent_bundle)
    )
    parent_documents = parent_component_plan["sec_access_plan"]["documents"]
    document_ordinal_by_accession = {
        document["accession_number"]: ordinal
        for ordinal, document in enumerate(parent_documents, start=1)
    }
    parent_bytes_by_logical_id = {
        item["logical_id"]: item for item in parent_reader["byte_index"]
    }
    expected_copy_index: list[dict[str, Any]] = []
    for copy_ordinal, record in enumerate(records, start=1):
        source_ordinal = document_ordinal_by_accession.get(record["accession_number"])
        if source_ordinal is None:
            raise SecFilingGemmaStageAuthorizationError(
                "Final carry-in accession is absent from the parent SEC plan"
            )
        source_item = parent_bytes_by_logical_id.get(
            f"document-{source_ordinal:04d}-normalized"
        )
        if (
            type(source_item) is not dict
            or source_item.get("relative_path")
            != f"document-{source_ordinal:04d}.normalized.txt"
            or source_item.get("sha256") != record["normalized_text_sha256"]
            or source_item.get("byte_count") != record["normalized_text_bytes"]
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Final carry-in bytes differ from the parent SEC reader receipt"
            )
        expected_copy_index.append(
            {
                "ordinal": copy_ordinal,
                "logical_id": f"carry-in-{copy_ordinal:04d}-normalized",
                "relative_path": f"carry-in-{copy_ordinal:04d}.normalized.txt",
                "byte_count": source_item["byte_count"],
                "sha256": source_item["sha256"],
            }
        )
    copied_index = _validated_sec_byte_index(carry_in_byte_index)
    if copied_index != expected_copy_index:
        raise SecFilingGemmaStageAuthorizationError(
            "Final carry-in copied-byte index differs from its parent bytes"
        )
    marker_hash = _sha256(
        carry_in_complete_marker_sha256,
        "final carry-in complete-marker hash",
    )
    source_hash = _sha256(reader_source_sha256, "final carry-in reader source hash")
    if source_hash != child_claim["execution_source_hashes"].get("reveal_store"):
        raise SecFilingGemmaStageAuthorizationError(
            "Final carry-in reader source differs from the child execution closure"
        )
    body = {
        "schema_version": STAGE_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "receipt_kind": "store_rehashed_final_child_prior_same_form_carry_in",
        "request_sha256": child_request_hash,
        "consumption_entry_sha256": child_grant["consumption_entry_sha256"],
        "consumption_entry_sequence": child_grant["consumption_entry_sequence"],
        "attempt_id": child_grant["attempt_id"],
        "candidate_sha256": child_grant["candidate_sha256"],
        "registry_entry_sha256": child_grant["registry_entry_sha256"],
        "input_prerequisite_stage": child_grant["prerequisite_stage"],
        "authorized_stage": child_grant["stage"],
        "input_stage_evidence_sha256": child_grant[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": child_grant[
            "stage_access_manifest_sha256"
        ],
        "output_namespace": child_grant["output_namespace"],
        "authorization_bundle_sha256": child_bundle["bundle_sha256"],
        "authorization_grant_sha256": child_grant[
            "authorization_grant_sha256"
        ],
        "grant_store_state_sha256": child_grant["store_state_sha256"],
        "grant_consumption_ledger_sha256": child_grant[
            "consumption_ledger_sha256"
        ],
        "grant_consumption_ledger_tip_sha256": child_grant[
            "consumption_ledger_tip_sha256"
        ],
        "child_sec_execution_claim_sha256": child_claim["claim_sha256"],
        "child_sec_reader_receipt_sha256": child_reader["receipt_sha256"],
        "parent_request_sha256": parent_request_hash,
        "parent_sec_execution_claim_sha256": parent_claim["claim_sha256"],
        "parent_sec_reader_receipt_sha256": parent_reader["receipt_sha256"],
        "parent_consumed_stage_output_receipt_sha256": parent_output[
            "output_receipt_sha256"
        ],
        "parent_content_manifest_sha256": evidence_pin[
            "content_manifest_sha256"
        ],
        "parent_stage_artifact_sha256": evidence_pin["stage_artifact_sha256"],
        "parent_external_seal_receipt_sha256": evidence_pin[
            "external_seal_receipt_sha256"
        ],
        "parent_stage_evidence_sha256": parent_output[
            "output_stage_evidence_sha256"
        ],
        "parent_stage_evidence_document_sha256": parent_output[
            "output_stage_evidence_document_sha256"
        ],
        "parent_stage_evidence_complete_marker_sha256": parent_output[
            "output_stage_evidence_complete_marker_sha256"
        ],
        "selection_policy": carry_plan["selection_policy"],
        "artifact_scope": carry_plan["artifact_scope"],
        "carry_in_record_count": len(records),
        "carry_in_records_sha256": canonical_sha256(records),
        "carry_in_records": records,
        "carry_in_byte_index": copied_index,
        "carry_in_byte_index_sha256": canonical_sha256(copied_index),
        "carry_in_byte_count_total": sum(
            item["byte_count"] for item in copied_index
        ),
        "carry_in_complete_marker_sha256": marker_hash,
        "reader_source_sha256": source_hash,
        "reader_output_recomputed_by_store": True,
        "network_refetch_permitted": False,
        "write_permitted": False,
        "general_cross_stage_access_permitted": False,
        "fresh_carry_in_provenance_claimed": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def validate_stage_carry_in_reader_receipt(
    receipt: Mapping[str, Any],
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    carry_in_byte_index: list[dict[str, Any]],
    carry_in_complete_marker_sha256: str,
    reader_source_sha256: str,
) -> str:
    """Require exact current-tip membership and revalidated copied-byte inputs."""

    observed = _mapping(receipt, "stage carry-in reader receipt")
    _expect_keys(
        observed,
        _STAGE_CARRY_IN_READER_RECEIPT_KEYS,
        "stage carry-in reader receipt",
    )
    observed_hash = _self_hash(
        observed,
        "receipt_sha256",
        "stage carry-in reader receipt",
    )
    current_tip = validate_reveal_store_current_tip_anchor(
        authenticated_store_snapshot,
        independent_current_tip_anchor,
    )
    child_request_hash = _sha256(
        observed.get("request_sha256"),
        "stage carry-in child request hash",
    )
    parent_request_hash = _sha256(
        observed.get("parent_request_sha256"),
        "stage carry-in parent request hash",
    )
    if current_tip["stage_carry_in_reader_receipts"].get(
        child_request_hash
    ) != observed:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage carry-in reader receipt is not persisted at the current tip"
        )
    child_bundle = current_tip["authorization_bundles"].get(child_request_hash)
    parent_bundle = current_tip["authorization_bundles"].get(parent_request_hash)
    child_claim = current_tip["stage_sec_execution_claims"].get(
        child_request_hash
    )
    parent_claim = current_tip["stage_sec_execution_claims"].get(
        parent_request_hash
    )
    child_reader = current_tip["stage_sec_reader_receipts"].get(
        child_request_hash
    )
    parent_reader = current_tip["stage_sec_reader_receipts"].get(
        parent_request_hash
    )
    parent_output = current_tip["consumed_stage_output_receipts"].get(
        parent_request_hash
    )
    if any(
        type(value) is not dict
        for value in (
            child_bundle,
            parent_bundle,
            child_claim,
            parent_claim,
            child_reader,
            parent_reader,
            parent_output,
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Stage carry-in reader receipt lost its persisted ancestry"
        )
    expected = build_stage_carry_in_reader_receipt(
        child_bundle,
        stage_sec_execution_claim=child_claim,
        stage_sec_reader_receipt=child_reader,
        parent_authorization_bundle=parent_bundle,
        parent_stage_sec_execution_claim=parent_claim,
        parent_stage_sec_reader_receipt=parent_reader,
        parent_consumed_stage_output_receipt=parent_output,
        carry_in_byte_index=carry_in_byte_index,
        carry_in_complete_marker_sha256=carry_in_complete_marker_sha256,
        reader_source_sha256=reader_source_sha256,
    )
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage carry-in reader receipt differs from revalidated durable inputs"
        )
    return observed_hash


def build_consumed_stage_output_receipt(
    authorization_bundle: Mapping[str, Any],
    *,
    stage_sec_execution_claim: Mapping[str, Any],
    stage_sec_reader_receipt: Mapping[str, Any],
    output_stage_evidence_schema_version: str,
    output_stage_evidence_sha256: str,
    output_stage_evidence_document_sha256: str,
    output_stage_evidence_complete_marker_sha256: str,
    output_stage_evidence_canonical_byte_count: int,
    output_stage_evidence_prerequisite_stage: str,
    output_parent_stage_evidence_sha256: str,
    output_candidate_sha256: str,
) -> dict[str, Any]:
    """Bind the first recorded next-stage evidence document to one consumed grant.

    This pure builder does not authorize I/O.  The effectful reveal store must
    first validate the bundle at its independently loaded current tip and then
    persist the resulting receipt with an append-only CAS transition.  The
    receipt binds store-recomputed durable evidence to the exact SEC claim and
    reader receipt, but deliberately does not claim fresh end-to-end evidence
    provenance until the remaining owned components exist.
    """

    raw_bundle = _mapping(authorization_bundle, "stage-output authorization bundle")
    raw_grant = _mapping(
        raw_bundle.get("authorization_grant"),
        "stage-output authorization grant",
    )
    request_hash = _sha256(
        raw_grant.get("request_sha256"),
        "stage-output authorization request hash",
    )
    bundle = _validated_authorization_bundles({request_hash: raw_bundle})[
        request_hash
    ]
    grant = bundle["authorization_grant"]
    claim = _validated_stage_sec_execution_claims(
        {request_hash: stage_sec_execution_claim},
        authorization_bundles={request_hash: bundle},
    )[request_hash]
    reader = _validated_stage_sec_reader_receipts(
        {request_hash: stage_sec_reader_receipt},
        claims={request_hash: claim},
    )[request_hash]
    schema_version = _safe_id(
        output_stage_evidence_schema_version,
        "output stage-evidence schema version",
    )
    if schema_version != STAGE_EVIDENCE_SCHEMA_VERSION:
        raise SecFilingGemmaStageAuthorizationError(
            "Output stage evidence is not the exact durable v3 schema"
        )
    evidence_hash = _sha256(
        output_stage_evidence_sha256,
        "output stage-evidence hash",
    )
    document_hash = _sha256(
        output_stage_evidence_document_sha256,
        "output stage-evidence document hash",
    )
    marker_hash = _sha256(
        output_stage_evidence_complete_marker_sha256,
        "output stage-evidence complete-marker hash",
    )
    sec_claim_hash = claim["claim_sha256"]
    sec_reader_hash = reader["receipt_sha256"]
    byte_count = _strict_int(
        output_stage_evidence_canonical_byte_count,
        "output stage-evidence canonical byte count",
        minimum=2,
    )
    output_prerequisite = _safe_id(
        output_stage_evidence_prerequisite_stage,
        "output stage-evidence prerequisite stage",
    )
    output_parent_hash = _sha256(
        output_parent_stage_evidence_sha256,
        "output parent stage-evidence hash",
    )
    output_candidate_hash = _sha256(
        output_candidate_sha256,
        "output candidate hash",
    )
    if (
        output_prerequisite != grant["stage"]
        or output_parent_hash != grant["prerequisite_stage_evidence_sha256"]
        or output_candidate_hash != grant["candidate_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Stage-output evidence crossed its consumed grant boundary"
        )
    body = {
        "schema_version": CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "receipt_kind": "first_output_for_exact_consumed_stage_grant",
        "consumption_entry_sha256": grant["consumption_entry_sha256"],
        "consumption_entry_sequence": grant["consumption_entry_sequence"],
        "request_sha256": grant["request_sha256"],
        "attempt_id": grant["attempt_id"],
        "candidate_sha256": grant["candidate_sha256"],
        "registry_entry_sha256": grant["registry_entry_sha256"],
        "input_prerequisite_stage": grant["prerequisite_stage"],
        "output_stage": grant["stage"],
        "input_stage_evidence_sha256": grant[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": grant["stage_access_manifest_sha256"],
        "output_namespace": grant["output_namespace"],
        "authorization_bundle_sha256": bundle["bundle_sha256"],
        "authorization_grant_sha256": grant["authorization_grant_sha256"],
        "sec_execution_claim_sha256": sec_claim_hash,
        "sec_reader_receipt_sha256": sec_reader_hash,
        "grant_store_state_sha256": grant["store_state_sha256"],
        "grant_consumption_ledger_sha256": grant[
            "consumption_ledger_sha256"
        ],
        "grant_consumption_ledger_tip_sha256": grant[
            "consumption_ledger_tip_sha256"
        ],
        "output_kind": "next_stage_evidence",
        "output_stage_evidence_schema_version": schema_version,
        "output_stage_evidence_component_id": STAGE_EVIDENCE_OUTPUT_COMPONENT_ID,
        "output_stage_evidence_relative_path": STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH,
        "output_stage_evidence_sha256": evidence_hash,
        "output_stage_evidence_document_sha256": document_hash,
        "output_stage_evidence_complete_marker_sha256": marker_hash,
        "output_stage_evidence_canonical_byte_count": byte_count,
        "output_stage_evidence_prerequisite_stage": output_prerequisite,
        "output_parent_stage_evidence_sha256": output_parent_hash,
        "output_candidate_sha256": output_candidate_hash,
        "output_stage_evidence_recomputed_by_store": True,
        "fresh_stage_evidence_provenance_claimed": False,
        "cross_stage_output_permitted": False,
        "grant_reuse_for_different_output_permitted": False,
    }
    return {**body, "output_receipt_sha256": canonical_sha256(body)}


def validate_consumed_stage_output_receipt(
    receipt: Mapping[str, Any],
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    authorization_bundle: Mapping[str, Any],
    output_stage_evidence_schema_version: str,
    output_stage_evidence_sha256: str,
    output_stage_evidence_document_sha256: str,
    output_stage_evidence_complete_marker_sha256: str,
    output_stage_evidence_canonical_byte_count: int,
    output_stage_evidence_prerequisite_stage: str,
    output_parent_stage_evidence_sha256: str,
    output_candidate_sha256: str,
) -> str:
    """Require exact current-tip membership for one first-output receipt."""

    bundle = _mapping(authorization_bundle, "stage-output authorization bundle")
    grant = _mapping(
        bundle.get("authorization_grant"),
        "stage-output authorization grant",
    )
    current_tip = validate_reveal_store_current_tip_anchor(
        authenticated_store_snapshot,
        independent_current_tip_anchor,
    )
    request_hash = _sha256(
        grant.get("request_sha256"),
        "stage-output authorization request hash",
    )
    sec_claim = current_tip["stage_sec_execution_claims"].get(request_hash)
    sec_reader = current_tip["stage_sec_reader_receipts"].get(request_hash)
    if (
        type(sec_claim) is not dict
        or type(sec_reader) is not dict
        or sec_reader.get("claim_sha256") != sec_claim.get("claim_sha256")
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Stage-output receipt lacks its exact SEC claim and reader ancestry"
        )
    if current_tip["authorization_bundles"].get(request_hash) != bundle:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage-output authorization bundle is not persisted at the current tip"
        )
    validate_consumed_stage_authorization_grant(
        grant,
        authenticated_store_snapshot=authenticated_store_snapshot,
        external_store_state_pin=bundle["store_state_pin"],
        independent_current_tip_anchor=current_tip,
        expected_consumption_entry_sha256=grant["consumption_entry_sha256"],
        expected_request_sha256=request_hash,
        expected_candidate_sha256=output_candidate_sha256,
        expected_stage=output_stage_evidence_prerequisite_stage,
        expected_prerequisite_stage_evidence_sha256=(
            output_parent_stage_evidence_sha256
        ),
        expected_stage_access_manifest_sha256=grant[
            "stage_access_manifest_sha256"
        ],
        expected_output_namespace=grant["output_namespace"],
    )
    expected = build_consumed_stage_output_receipt(
        bundle,
        stage_sec_execution_claim=sec_claim,
        stage_sec_reader_receipt=sec_reader,
        output_stage_evidence_schema_version=output_stage_evidence_schema_version,
        output_stage_evidence_sha256=output_stage_evidence_sha256,
        output_stage_evidence_document_sha256=(
            output_stage_evidence_document_sha256
        ),
        output_stage_evidence_complete_marker_sha256=(
            output_stage_evidence_complete_marker_sha256
        ),
        output_stage_evidence_canonical_byte_count=(
            output_stage_evidence_canonical_byte_count
        ),
        output_stage_evidence_prerequisite_stage=(
            output_stage_evidence_prerequisite_stage
        ),
        output_parent_stage_evidence_sha256=output_parent_stage_evidence_sha256,
        output_candidate_sha256=output_candidate_sha256,
    )
    observed = _mapping(receipt, "consumed-stage output receipt")
    _expect_keys(observed, _STAGE_OUTPUT_RECEIPT_KEYS, "consumed-stage output receipt")
    _self_hash(
        observed,
        "output_receipt_sha256",
        "consumed-stage output receipt",
    )
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Consumed-stage output receipt differs from the exact grant or evidence"
        )
    if current_tip["consumed_stage_output_receipts"].get(request_hash) != observed:
        raise SecFilingGemmaStageAuthorizationError(
            "Consumed-stage output receipt is not persisted at the current tip"
        )
    return observed["output_receipt_sha256"]


__all__ = [
    "CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION",
    "CONSUMED_STAGE_AUTHORIZATION_GRANT_SCHEMA_VERSION",
    "CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION",
    "CONSUMED_STAGE_STORE_PIN_SCHEMA_VERSION",
    "DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID",
    "DEVELOPMENT_CONTENT_ROOT_PLAN_SCHEMA_VERSION",
    "DEVELOPMENT_SEC_EXECUTION_ABORT_SCHEMA_VERSION",
    "DEVELOPMENT_SEC_EXECUTION_CLAIM_SCHEMA_VERSION",
    "DEVELOPMENT_SEC_READER_RECEIPT_SCHEMA_VERSION",
    "DEVELOPMENT_ROOT_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION",
    "REVEAL_STORE_CURRENT_TIP_ANCHOR_SCHEMA_VERSION",
    "OWNED_SEC_RAW_BATCH_MAX_BYTES",
    "SEC_EXECUTION_RESOLVED_SOURCE_PATHS",
    "SEC_CORPUS_REPOSITORY_PATH",
    "SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID",
    "STAGE_EVIDENCE_OUTPUT_COMPONENT_ID",
    "STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH",
    "STAGE_RUNNER_REPOSITORY_PATH",
    "STAGE_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION",
    "STAGE_SEC_EXECUTION_ABORT_SCHEMA_VERSION",
    "STAGE_SEC_EXECUTION_CLAIM_SCHEMA_VERSION",
    "STAGE_SEC_READER_RECEIPT_SCHEMA_VERSION",
    "TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION",
    "TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION",
    "SecFilingGemmaStageAuthorizationError",
    "authenticate_reveal_store_trusted_stage_content_pin",
    "build_consumed_stage_authorization_grant",
    "build_consumed_stage_output_receipt",
    "build_development_sec_execution_abort",
    "build_development_sec_execution_claim",
    "build_development_sec_reader_receipt",
    "build_development_root_carry_in_reader_receipt",
    "build_stage_carry_in_reader_receipt",
    "build_stage_sec_execution_abort",
    "build_stage_sec_execution_claim",
    "build_stage_sec_reader_receipt",
    "build_reveal_store_current_tip_anchor",
    "derive_consumed_stage_store_state_pin",
    "derive_reveal_store_trusted_stage_content_pin",
    "validate_consumed_stage_authorization_grant",
    "validate_consumed_stage_output_receipt",
    "validate_consumed_stage_store_state_pin",
    "validate_development_root_carry_in_reader_receipt",
    "validate_reveal_store_current_tip_anchor",
    "validate_reveal_store_current_tip_anchor_structure",
    "validate_reveal_store_current_tip_anchor_transition",
    "validate_stage_carry_in_reader_receipt",
    "validate_trusted_stage_content_authentication_receipt",
]
