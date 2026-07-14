"""Pure consumed-stage authorization grants for downstream SEC/Gemma APIs.

The effectful reveal store remains the only component allowed to consume a
request.  This module can derive a detached pin from an already authenticated
post-consumption store snapshot, mint a grant for that snapshot's exact ledger
tip, and validate the grant later against a separately supplied snapshot/pin
plus the independently loaded monotonic current-tip anchor.

Pin derivation is not external authentication.  A downstream caller must load
the current-tip anchor independently of the bundle; an old snapshot and its old
bundled pin are deliberately rejected after any newer store transition.
No function here performs network, model, market-provider, SEC, or outcome I/O.
Market authority validation only rebuilds the reviewed local source-hash plan;
the compact grant contains no market values, labels, scores, or returns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from datetime import date
import hashlib
import hmac
import json
import math
import re
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_contract import (
    BRIER_TARGET_COST_BPS,
    CANDIDATE_IDS,
    CANONICAL_IDENTITY_LEXICON_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_POLICY_SESSION_DATES,
    DEVELOPMENT_FOLD_SPECS,
    HORIZON_SESSIONS,
    LABEL_MATURITY_OFFSET,
    MAX_INPUT_BYTES,
    MAX_MODEL_SECONDS,
    MAX_SENTENCE_CHARACTERS,
    MAX_SENTENCES,
    REQUIRED_STAGE_VERIFIER_CHECKS,
    STAGE_WINDOWS,
    STAGE_MODEL_CALL_CAPS,
    build_contract_manifest,
    build_stage_content_manifest,
    canonical_sha256,
    development_policy_session_calendar_sha256,
    market_session_calendar_sha256,
    validate_candidate_manifest,
)
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    REVEAL_REQUEST_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_market_acquirer import (
    YAHOO_MAX_RESPONSE_BYTES,
    YAHOO_REQUEST_COUNT,
    YAHOO_REQUEST_TIMEOUT_SECONDS,
    build_development_market_acquisition_plan,
)
from agent_benchmark.sec_filing_gemma_market_evidence import (
    MARKET_FIELDS,
    MARKET_SOURCE_FAMILY,
    MARKET_SYMBOLS,
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
from agent_benchmark.sec_filing_gemma_prediction_evidence import (
    PREDICTION_PREFIX_SCHEMA_VERSION,
    PREDICTION_ROW_SCHEMA_VERSION,
    _PREDICTION_ROW_KEYS as _AUTHORITATIVE_PREDICTION_ROW_KEYS,
)
from agent_benchmark.sec_filing_gemma_learner import (
    MODEL_TYPE as LEARNER_MODEL_TYPE,
    STATE_SCHEMA_VERSION as LEARNER_STATE_SCHEMA_VERSION,
    SecFilingGemmaLearnerConfig,
)
from agent_benchmark.sec_session_calendar import EXPECTED_MARKET_HISTORY_SESSIONS


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
DEVELOPMENT_MARKET_EXECUTION_CLAIM_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-market-execution-claim-v1"
)
DEVELOPMENT_MARKET_READER_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-market-reader-receipt-v2"
)
DEVELOPMENT_MARKET_EXECUTION_ABORT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-market-execution-abort-v1"
)
DEVELOPMENT_ROOT_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-root-carry-in-reader-receipt-v1"
)
STAGE_MODEL_EXECUTION_CLAIM_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-model-execution-claim-v3"
)
STAGE_MODEL_READER_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-model-reader-receipt-v3"
)
STAGE_MODEL_EXECUTION_ABORT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-model-execution-abort-v1"
)
DEVELOPMENT_MODEL_EXECUTION_CLAIM_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-model-execution-claim-v3"
)
DEVELOPMENT_MODEL_READER_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-model-reader-receipt-v3"
)
DEVELOPMENT_MODEL_EXECUTION_ABORT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-model-execution-abort-v1"
)
DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-feature-assembly-plan-v1"
)
DEVELOPMENT_LABEL_ASSEMBLY_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-label-assembly-plan-v1"
)
DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-training-membership-assembly-plan-v1"
)
DEVELOPMENT_OOF_LEARNER_FIT_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-oof-learner-fit-plan-v1"
)
DEVELOPMENT_OOF_PREDICTION_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-oof-prediction-plan-v1"
)
DEVELOPMENT_POLICY_REPLAY_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-policy-replay-plan-v1"
)
DEVELOPMENT_POLICY_PREFIX_SEAL_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-policy-prefix-seal-plan-v1"
)
DEVELOPMENT_POLICY_PREFIX_SEAL_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-policy-prefix-seal-receipt-v1"
)
DEVELOPMENT_POLICY_PREFIX_EXTERNAL_PIN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-policy-prefix-external-pin-v1"
)
REVEAL_STORE_CURRENT_TIP_ANCHOR_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-reveal-store-current-tip-anchor-v10"
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
_ISO_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
STAGE_RUNNER_REPOSITORY_PATH: Final[str] = (
    "agent_benchmark/sec_filing_gemma_stage_runner.py"
)
SEC_CORPUS_REPOSITORY_PATH: Final[str] = (
    "agent_benchmark/sec_filing_gemma_corpus.py"
)
SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID: Final[str] = "sec_stage_document_batch"
STAGE_MODEL_BATCH_COMPONENT_ID: Final[str] = "owned_stage_gemma_model_batch"
DEVELOPMENT_MARKET_BATCH_COMPONENT_ID: Final[str] = (
    "owned_development_market_evidence_batch"
)
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
# The owned model path executes through store/registry validation, SEC/carry
# replay, the authoritative calendar, stage access/verifier ancestry, and the
# local extractor.  Bind the complete currently resolved candidate source tree
# instead of a hand-maintained subset so an omitted helper cannot change model
# inputs while the execution claim still appears source-identical.  The two
# deliberately unresolved ledger role remains excluded and fails closed at its
# own later ownership gate.
MODEL_EXECUTION_SOURCE_ROLES: Final[tuple[str, ...]] = tuple(
    role for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
)
MARKET_EXECUTION_SOURCE_ROLES: Final[tuple[str, ...]] = tuple(
    role for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
)
_MARKET_ACQUISITION_SOURCE_PATH_ROLES: Final[tuple[tuple[str, str], ...]] = (
    ("agent_benchmark/sec_filing_gemma_market_acquirer.py", "market_acquirer"),
    ("agent_benchmark/sec_filing_gemma_market_evidence.py", "market_evidence"),
    (
        "agent_benchmark/sec_filing_gemma_market_source_bytes.py",
        "market_source_bytes",
    ),
    ("agent_benchmark/sec_filing_gemma_contract.py", "contract"),
    ("agent_benchmark/sec_session_calendar.py", "calendar"),
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
_STAGE_MODEL_EXECUTION_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
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
        "stage_sec_execution_claim_sha256",
        "stage_sec_reader_receipt_sha256",
        "development_root_scope_sha256",
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
        "corpus_universe_sha256",
        "carry_in_kind",
        "carry_in_reader_receipt_sha256",
        "sec_document_count",
        "sec_acquisition_accession_order",
        "sec_acquisition_accession_order_sha256",
        "event_count",
        "event_plan",
        "event_plan_sha256",
        "identity_lexicon_sha256",
        "candidate_source_hashes_sha256",
        "execution_source_hashes",
        "execution_source_hashes_sha256",
        "execution_source_role_count",
        "model_name",
        "model_digest",
        "runtime_fingerprint_sha256",
        "model_transport_sha256",
        "model_runtime_limits",
        "model_runtime_limits_sha256",
        "model_component_id",
        "owned_local_model_execution_required",
        "caller_supplied_path_permitted",
        "filing_text_included",
        "market_access_permitted",
        "outcome_access_permitted",
        "future_stage_access_permitted",
        "paid_api_access_permitted",
        "external_network_access_permitted",
        "effect_may_be_repeated_after_indeterminate_crash",
        "claim_sha256",
    }
)
_STAGE_MODEL_READER_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "receipt_kind",
        "request_sha256",
        "claim_sha256",
        "authorized_stage",
        "candidate_sha256",
        "output_namespace",
        "stage_sec_execution_claim_sha256",
        "stage_sec_reader_receipt_sha256",
        "development_root_scope_sha256",
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
        "corpus_universe_sha256",
        "carry_in_kind",
        "carry_in_reader_receipt_sha256",
        "sec_document_count",
        "sec_acquisition_accession_order_sha256",
        "event_count",
        "event_plan_sha256",
        "identity_lexicon_sha256",
        "execution_source_hashes_sha256",
        "execution_source_role_count",
        "model_name",
        "model_digest",
        "runtime_fingerprint_sha256",
        "model_transport_sha256",
        "model_runtime_limits_sha256",
        "model_component_id",
        "byte_index",
        "byte_index_sha256",
        "byte_count_total",
        "complete_marker_sha256",
        "fresh_model_provenance_claimed",
        "reader_output_recomputed_by_store",
        "receipt_sha256",
    }
)
_STAGE_MODEL_EXECUTION_ABORT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "abort_kind",
        "request_sha256",
        "claim_sha256",
        "authorized_stage",
        "candidate_sha256",
        "output_namespace",
        "model_component_id",
        "reason",
        "external_effect_retry_permitted",
        "abort_sha256",
    }
)
_DEVELOPMENT_MODEL_EXECUTION_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "claim_kind",
        "development_root_scope_sha256",
        "development_content_root_plan_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "corpus_universe_sha256",
        "corpus_universe_semantic_sha256",
        "authorized_stage",
        "output_namespace",
        "start_current_tip_anchor_sha256",
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
        "sec_document_count",
        "sec_acquisition_accession_order",
        "sec_acquisition_accession_order_sha256",
        "event_count",
        "event_plan",
        "event_plan_sha256",
        "identity_lexicon_sha256",
        "candidate_source_hashes_sha256",
        "execution_source_hashes",
        "execution_source_hashes_sha256",
        "execution_source_role_count",
        "model_name",
        "model_digest",
        "runtime_fingerprint_sha256",
        "model_transport_sha256",
        "model_runtime_limits",
        "model_runtime_limits_sha256",
        "model_component_id",
        "carry_in_required",
        "owned_local_model_execution_required",
        "caller_supplied_path_permitted",
        "filing_text_included",
        "market_access_permitted",
        "outcome_access_permitted",
        "future_stage_access_permitted",
        "paid_api_access_permitted",
        "external_network_access_permitted",
        "reveal_request_consumption_permitted",
        "consumption_ledger_mutation_permitted",
        "effect_may_be_repeated_after_indeterminate_crash",
        "claim_sha256",
    }
)
_DEVELOPMENT_MODEL_READER_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "receipt_kind",
        "development_root_scope_sha256",
        "claim_sha256",
        "development_content_root_plan_sha256",
        "authorized_stage",
        "candidate_sha256",
        "output_namespace",
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
        "sec_document_count",
        "sec_acquisition_accession_order_sha256",
        "event_count",
        "event_plan_sha256",
        "identity_lexicon_sha256",
        "execution_source_hashes_sha256",
        "execution_source_role_count",
        "model_name",
        "model_digest",
        "runtime_fingerprint_sha256",
        "model_transport_sha256",
        "model_runtime_limits_sha256",
        "model_component_id",
        "byte_index",
        "byte_index_sha256",
        "byte_count_total",
        "complete_marker_sha256",
        "fresh_model_provenance_claimed",
        "reader_output_recomputed_by_store",
        "receipt_sha256",
    }
)
_DEVELOPMENT_MODEL_EXECUTION_ABORT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "abort_kind",
        "development_root_scope_sha256",
        "claim_sha256",
        "development_content_root_plan_sha256",
        "authorized_stage",
        "candidate_sha256",
        "output_namespace",
        "model_component_id",
        "reason",
        "external_effect_retry_permitted",
        "abort_sha256",
    }
)
_DEVELOPMENT_MARKET_EXECUTION_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "claim_kind",
        "development_root_scope_sha256",
        "development_content_root_plan_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "registered_entry_count",
        "corpus_universe_sha256",
        "corpus_universe_semantic_sha256",
        "authorized_stage",
        "output_namespace",
        "start_current_tip_anchor_sha256",
        "start_state_sha256",
        "start_consumption_ledger_sha256",
        "start_consumption_ledger_tip_sha256",
        "start_consumed_request_count",
        "development_sec_execution_claim_sha256",
        "development_sec_reader_receipt_sha256",
        "market_acquisition_plan",
        "market_acquisition_plan_sha256",
        "candidate_source_hashes_sha256",
        "execution_source_hashes",
        "execution_source_hashes_sha256",
        "execution_source_role_count",
        "market_component_id",
        "source_family",
        "market_symbols",
        "market_fields",
        "fixed_request_count",
        "response_byte_ceiling_per_symbol",
        "timeout_seconds_per_symbol",
        "owned_market_execution_required",
        "caller_supplied_path_permitted",
        "caller_supplied_bytes_permitted",
        "market_access_permitted",
        "outcome_access_permitted",
        "future_stage_access_permitted",
        "paid_api_access_permitted",
        "external_network_access_permitted",
        "reveal_request_consumption_permitted",
        "consumption_ledger_mutation_permitted",
        "effect_may_be_repeated_after_indeterminate_crash",
        "claim_sha256",
    }
)
_DEVELOPMENT_MARKET_READER_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "receipt_kind",
        "development_root_scope_sha256",
        "claim_sha256",
        "development_content_root_plan_sha256",
        "authorized_stage",
        "candidate_sha256",
        "output_namespace",
        "development_sec_execution_claim_sha256",
        "development_sec_reader_receipt_sha256",
        "market_acquisition_plan_sha256",
        "execution_source_hashes_sha256",
        "execution_source_role_count",
        "market_component_id",
        "acquisition_receipt_sha256",
        "acquisition_bundle_sha256",
        "acquisition_validation_sha256",
        "source_manifest_sha256",
        "market_stage_manifest_sha256",
        "source_reconciliation_sha256",
        "raw_response_sha256s",
        "artifact_sha256s",
        "window_sha256s",
        "byte_index",
        "byte_index_sha256",
        "byte_count_total",
        "complete_marker_sha256",
        "fresh_network_provenance_claimed",
        "provider_response_normalization_replayed_by_store",
        "owned_transport_attested_by_store",
        "reader_output_recomputed_by_store",
        "receipt_sha256",
    }
)
_DEVELOPMENT_MARKET_EXECUTION_ABORT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "abort_kind",
        "development_root_scope_sha256",
        "claim_sha256",
        "development_content_root_plan_sha256",
        "authorized_stage",
        "candidate_sha256",
        "output_namespace",
        "market_component_id",
        "reason",
        "external_effect_retry_permitted",
        "abort_sha256",
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
_DEVELOPMENT_LABEL_ASSEMBLY_PLAN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "plan_kind",
        "artifact_stage",
        "development_root_scope_sha256",
        "start_consumed_request_count",
        "source_feature_assembly_plan",
        "source_feature_assembly_plan_sha256",
        "calendar_sessions_sha256",
        "development_cutoff_session",
        "label_horizon_sessions",
        "label_entry_session_offset",
        "label_maturity_session_offset",
        "maturity_rule",
        "event_count",
        "maturity_plan",
        "maturity_plan_sha256",
        "matured_event_count",
        "unmatured_event_count",
        "canonical_market_rows_required",
        "development_outcome_derivation_permitted",
        "development_label_rows_output_permitted",
        "post_cutoff_market_access_permitted",
        "raw_market_output_permitted",
        "normalized_filing_text_output_permitted",
        "model_transport_envelope_output_permitted",
        "training_membership_access_permitted",
        "learner_fit_permitted",
        "prediction_access_permitted",
        "holdout_access_permitted",
        "ledger_mutation_permitted",
        "stage_promotion_permitted",
        "production_permitted",
        "label_assembly_plan_sha256",
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
_DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_KEYS: Final[
    frozenset[str]
] = frozenset(
    {
        "schema_version",
        "contract_version",
        "plan_kind",
        "artifact_stage",
        "development_root_scope_sha256",
        "start_consumed_request_count",
        "source_label_assembly_plan",
        "source_label_assembly_plan_sha256",
        "source_feature_assembly_plan_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "development_cutoff_session",
        "label_horizon_sessions",
        "label_entry_session_offset",
        "label_maturity_session_offset",
        "event_count",
        "matured_event_count",
        "unmatured_event_count",
        "membership_view_count",
        "membership_view_specs",
        "membership_view_specs_sha256",
        "model_variant_count",
        "model_variant_specs",
        "model_variant_specs_sha256",
        "membership_maturity_rule",
        "feature_eligibility_rule",
        "membership_order_rule",
        "shared_variant_support_rule",
        "target_cost_bps",
        "binary_target_field",
        "binary_target_encoding",
        "edge_target_field",
        "minimum_training_row_count",
        "both_binary_classes_required",
        "semantic_ablation_feature_matrices_must_differ",
        "canonical_source_feature_rows_required",
        "canonical_source_label_rows_required",
        "source_feature_rows_access_permitted",
        "source_label_rows_access_permitted",
        "development_outcome_values_access_permitted",
        "development_outcome_derivation_permitted",
        "training_membership_access_permitted",
        "training_membership_rows_output_permitted",
        "training_feature_matrices_output_permitted",
        "training_target_vectors_output_permitted",
        "outcome_based_membership_filtering_permitted",
        "row_rebalancing_permitted",
        "post_cutoff_market_access_permitted",
        "raw_market_output_permitted",
        "compact_adjusted_open_paths_output_permitted",
        "normalized_filing_text_output_permitted",
        "model_transport_envelope_output_permitted",
        "learner_fit_permitted",
        "prediction_access_permitted",
        "holdout_access_permitted",
        "ledger_mutation_permitted",
        "stage_promotion_permitted",
        "production_permitted",
        "training_membership_assembly_plan_sha256",
    }
)
_DEVELOPMENT_TRAINING_MEMBERSHIP_VIEW_SPEC_KEYS: Final[frozenset[str]] = (
    frozenset(
        {
            "view_ordinal",
            "training_view_id",
            "view_kind",
            "training_source_stage",
            "prediction_stage",
            "train_label_maturity_through",
            "prediction_window_first_date",
            "prediction_window_last_date",
            "state_updates_inside_prediction_window",
        }
    )
)
_DEVELOPMENT_TRAINING_MEMBERSHIP_VARIANT_SPEC_KEYS: Final[frozenset[str]] = (
    frozenset(
        {
            "variant_ordinal",
            "variant_id",
            "feature_names_field",
            "feature_values_field",
        }
    )
)
_DEVELOPMENT_LABEL_ENTRY_SESSION_OFFSET: Final[int] = 1
_DEVELOPMENT_LABEL_MATURITY_RULE: Final[str] = (
    "t_plus_21_session_lte_development_cutoff_inclusive"
)
_DEVELOPMENT_TRAINING_MEMBERSHIP_MATURITY_RULE: Final[str] = (
    "source_label_matured_by_development_cutoff_true_and_"
    "label_maturity_session_lte_train_cutoff_and_"
    "strictly_precedes_prediction_window_first_date"
)
_DEVELOPMENT_TRAINING_FEATURE_ELIGIBILITY_RULE: Final[str] = (
    "include_iff_fit_eligible_and_prediction_available_are_exact_true"
)
_DEVELOPMENT_TRAINING_MEMBERSHIP_ORDER_RULE: Final[str] = (
    "source_feature_event_plan_order_filtered_without_reordering"
)
_DEVELOPMENT_TRAINING_SHARED_VARIANT_SUPPORT_RULE: Final[str] = (
    "semantic_and_ablation_use_identical_ordered_membership_and_target_vectors"
)
_DEVELOPMENT_TRAINING_BINARY_TARGET_FIELD: Final[str] = (
    "cash_beats_long_10bps"
)
_DEVELOPMENT_TRAINING_BINARY_TARGET_ENCODING: Final[str] = (
    "false_to_0_true_to_1"
)
_DEVELOPMENT_TRAINING_EDGE_TARGET_FIELD: Final[str] = (
    "cash_active_log_edge_10bps_hex"
)
_DEVELOPMENT_TRAINING_MINIMUM_ROW_COUNT: Final[int] = 2
_DEVELOPMENT_OOF_LEARNER_FIT_INPUT_SPEC_KEYS: Final[frozenset[str]] = (
    frozenset(
        {
            "fit_ordinal",
            "source_training_view_ordinal",
            "training_view_id",
            "source_training_view_sha256",
            "head_variant",
            "training_row_count",
            "training_positive_count",
            "training_set_membership_sha256",
            "training_feature_matrix_sha256",
            "training_binary_target_sha256",
            "training_edge_target_sha256",
            "learner_input_context_sha256",
            "fit_metadata_template_sha256",
            "feature_schema_sha256",
            "train_label_maturity_through",
            "maximum_training_label_maturity_session",
            "fit_input_spec_sha256",
        }
    )
)
_DEVELOPMENT_OOF_DEFERRED_TRAINING_VIEW_KEYS: Final[frozenset[str]] = (
    frozenset(
        {
            "source_training_view_ordinal",
            "training_view_id",
            "reason",
        }
    )
)
_DEVELOPMENT_OOF_LEARNER_FIT_PLAN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "plan_kind",
        "artifact_stage",
        "development_root_scope_sha256",
        "start_consumed_request_count",
        "source_training_membership_assembly_plan",
        "source_training_membership_assembly_plan_sha256",
        "source_training_membership_projection_sha256",
        "source_training_membership_batch_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "development_cutoff_session",
        "source_training_view_count",
        "source_training_view_ids",
        "source_training_view_specs_sha256",
        "authorized_training_view_count",
        "authorized_training_view_ids",
        "deferred_training_view_count",
        "deferred_training_views",
        "deferred_training_views_sha256",
        "model_variant_count",
        "model_variant_ids",
        "learner_fit_input_count",
        "learner_fit_input_specs",
        "learner_fit_input_specs_sha256",
        "learner_state_output_count",
        "learner_model_type",
        "learner_state_schema_version",
        "learner_config",
        "learner_config_sha256",
        "fit_order_rule",
        "maximum_fit_seconds",
        "canonical_source_training_membership_batch_required",
        "source_training_membership_batch_access_permitted",
        "authorized_training_feature_matrices_access_permitted",
        "authorized_training_target_vectors_access_permitted",
        "deterministic_learner_fit_permitted",
        "learner_state_output_permitted",
        "compact_fit_audit_output_permitted",
        "deterministic_refit_validation_permitted",
        "source_feature_batch_access_permitted",
        "source_label_batch_access_permitted",
        "development_outcome_derivation_permitted",
        "training_membership_derivation_permitted",
        "training_membership_mutation_permitted",
        "outcome_based_membership_filtering_permitted",
        "row_rebalancing_permitted",
        "deferred_training_view_fit_permitted",
        "hyperparameter_change_permitted",
        "solver_retry_permitted",
        "feature_selection_permitted",
        "model_transport_access_permitted",
        "network_access_permitted",
        "prediction_access_permitted",
        "candidate_selection_permitted",
        "threshold_action_access_permitted",
        "holdout_access_permitted",
        "ledger_mutation_permitted",
        "stage_promotion_permitted",
        "production_permitted",
        "development_oof_learner_fit_plan_sha256",
    }
)
_DEVELOPMENT_OOF_AUTHORIZED_TRAINING_VIEW_IDS: Final[tuple[str, ...]] = (
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "fold_5",
)
_DEVELOPMENT_OOF_MODEL_VARIANT_IDS: Final[tuple[str, ...]] = (
    "semantic",
    "ablation",
)
_DEVELOPMENT_OOF_DEFERRED_TRAINING_VIEW_ID: Final[str] = (
    "intermediate_frozen_through_2018"
)
_DEVELOPMENT_OOF_DEFERRED_REASON: Final[str] = (
    "requires_passed_development_ranking_receipt_and_frozen_candidate_selection"
)
_DEVELOPMENT_OOF_FIT_ORDER_RULE: Final[str] = (
    "view_ordinal_ascending_then_semantic_then_ablation_exactly_once"
)
_DEVELOPMENT_OOF_MAXIMUM_FIT_SECONDS: Final[int] = 60
_DEVELOPMENT_OOF_PREDICTION_FOLD_MODEL_SPEC_KEYS: Final[frozenset[str]] = (
    frozenset(
        {
            "fold_ordinal",
            "fold_id",
            "prediction_window_first_date",
            "prediction_window_last_date",
            "source_learner_fit_view_sha256",
            "prediction_fold_context_sha256",
            "semantic_fit_record_sha256",
            "semantic_learner_state_sha256",
            "ablation_fit_record_sha256",
            "ablation_learner_state_sha256",
            "feature_schema_sha256",
            "prediction_fold_model_sha256",
            "prediction_fold_model_spec_sha256",
        }
    )
)
_DEVELOPMENT_OOF_PREDICTION_INPUT_SPEC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "prediction_ordinal",
        "source_event_ordinal",
        "fold_ordinal",
        "fold_id",
        "decision_session",
        "accession_number",
        "source_feature_row_sha256",
        "event_binding_sha256",
        "prediction_available",
        "unavailable_reason",
        "semantic_feature_schema_sha256",
        "semantic_feature_values_sha256",
        "ablation_feature_schema_sha256",
        "ablation_feature_values_sha256",
        "prediction_feature_input_sha256",
        "prediction_input_spec_sha256",
    }
)
_DEVELOPMENT_OOF_PREDICTION_PLAN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "plan_kind",
        "artifact_stage",
        "development_root_scope_sha256",
        "start_consumed_request_count",
        "source_development_oof_learner_fit_plan_sha256",
        "source_development_oof_learner_fit_projection_sha256",
        "source_development_oof_learner_fit_batch_sha256",
        "source_training_membership_assembly_plan_sha256",
        "source_training_membership_projection_sha256",
        "source_training_membership_batch_sha256",
        "source_feature_assembly_plan_sha256",
        "source_feature_batch_sha256",
        "prediction_feature_batch_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "development_cutoff_session",
        "source_event_count",
        "authorized_fold_count",
        "authorized_fold_ids",
        "prediction_fold_model_count",
        "prediction_fold_model_bundle_sha256",
        "prediction_fold_model_specs",
        "prediction_fold_model_specs_sha256",
        "model_variant_count",
        "model_variant_ids",
        "learner_state_count",
        "learner_model_type",
        "learner_state_schema_version",
        "learner_config_sha256",
        "feature_schema_sha256",
        "prediction_input_count",
        "available_prediction_input_count",
        "unavailable_prediction_input_count",
        "prediction_input_specs",
        "prediction_input_specs_sha256",
        "prediction_population_rule",
        "prediction_input_order_rule",
        "fold_state_usage_rule",
        "maximum_prediction_calls",
        "maximum_prediction_seconds",
        "canonical_prediction_fold_model_bundle_required",
        "canonical_prediction_feature_batch_required",
        "authorized_learner_state_access_permitted",
        "authorized_prediction_feature_row_access_permitted",
        "learner_state_deserialization_permitted",
        "deterministic_numeric_prediction_permitted",
        "raw_prediction_component_output_permitted",
        "unavailable_prediction_output_permitted",
        "compact_prediction_audit_output_permitted",
        "source_development_oof_learner_fit_batch_access_permitted",
        "source_training_membership_batch_access_permitted",
        "training_membership_rows_access_permitted",
        "training_feature_matrices_access_permitted",
        "training_target_vectors_access_permitted",
        "source_feature_batch_access_permitted",
        "source_label_batch_access_permitted",
        "label_access_permitted",
        "outcome_access_permitted",
        "post_decision_market_data_access_permitted",
        "post_2018_data_access_permitted",
        "deferred_training_view_access_permitted",
        "deferred_training_view_state_access_permitted",
        "learner_fit_permitted",
        "learner_state_update_permitted",
        "online_learning_permitted",
        "feature_mutation_permitted",
        "row_drop_permitted",
        "row_reordering_permitted",
        "prediction_retry_permitted",
        "model_transport_access_permitted",
        "network_access_permitted",
        "threshold_action_access_permitted",
        "candidate_selection_permitted",
        "policy_state_transition_permitted",
        "prediction_sealing_permitted",
        "label_release_permitted",
        "holdout_access_permitted",
        "ledger_mutation_permitted",
        "stage_promotion_permitted",
        "production_permitted",
        "development_oof_prediction_plan_sha256",
    }
)
_DEVELOPMENT_POLICY_REPLAY_PLAN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "plan_kind",
        "artifact_stage",
        "development_root_scope_sha256",
        "start_store_state_bytes_sha256",
        "start_store_state_sha256",
        "start_current_tip_anchor_bytes_sha256",
        "start_current_tip_anchor_sha256",
        "start_current_tip_revision",
        "start_consumed_request_count",
        "source_development_oof_prediction_plan_sha256",
        "source_development_oof_prediction_projection_sha256",
        "source_development_oof_prediction_batch_sha256",
        "source_raw_prediction_rows_sha256",
        "source_raw_prediction_tip_sha256",
        "source_raw_prediction_row_count",
        "policy_replay_input_count",
        "policy_replay_input_specs",
        "policy_replay_input_specs_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "development_cutoff_session",
        "candidate_count",
        "candidate_ids",
        "candidate_threshold_specs",
        "candidate_threshold_specs_sha256",
        "model_variant_count",
        "model_variant_ids",
        "candidate_gate_comparison_rule",
        "cash_episode_sessions",
        "cash_episode_rule",
        "unavailable_prediction_rule",
        "policy_replay_order_rule",
        "source_development_oof_prediction_batch_access_permitted",
        "raw_prediction_components_access_permitted",
        "threshold_evaluation_permitted",
        "policy_state_transition_permitted",
        "compact_policy_replay_output_permitted",
        "numeric_prediction_permitted",
        "learner_state_access_permitted",
        "source_feature_batch_access_permitted",
        "source_label_batch_access_permitted",
        "label_access_permitted",
        "outcome_access_permitted",
        "post_decision_market_data_access_permitted",
        "post_2018_data_access_permitted",
        "deferred_training_view_access_permitted",
        "deferred_training_view_state_access_permitted",
        "learner_fit_permitted",
        "learner_refit_permitted",
        "learner_state_update_permitted",
        "online_learning_permitted",
        "candidate_selection_permitted",
        "scoring_permitted",
        "prediction_sealing_permitted",
        "policy_replay_sealing_permitted",
        "label_release_permitted",
        "holdout_access_permitted",
        "model_transport_access_permitted",
        "network_access_permitted",
        "raw_prediction_mutation_permitted",
        "row_drop_permitted",
        "row_reordering_permitted",
        "policy_retry_permitted",
        "ledger_mutation_permitted",
        "stage_promotion_permitted",
        "production_permitted",
        "development_policy_replay_plan_sha256",
    }
)
_DEVELOPMENT_POLICY_PREFIX_SEAL_INPUT_SPEC_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-policy-prefix-seal-input-spec-v1"
)
_PREDICTION_ARTIFACT_STORE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-prediction-artifact-store-v1"
)
_PREDICTION_ARTIFACT_EXTERNAL_PIN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-prediction-artifact-external-pin-v1"
)
_PREDICTION_ARTIFACT_ENCODING: Final[str] = (
    "canonical-json-utf8-sorted-keys-v1"
)
_DEVELOPMENT_POLICY_PREFIX_SEAL_NAMESPACE_KIND: Final[str] = (
    "owned-development-policy-prefix-seals-v1"
)
_DEVELOPMENT_POLICY_PREFIX_SEAL_BATCH_RULE: Final[str] = (
    "all_authorized_cumulative_prefixes_exactly_once_in_one_atomic_compare_and_swap"
)
_DEVELOPMENT_POLICY_PREFIX_SEAL_RECOVERY_RULE: Final[str] = (
    "only_exact_same_plan_and_bytes_may_replay_after_crash_no_alternate_repair"
)
_DEVELOPMENT_POLICY_PREFIX_EXTERNAL_PIN_RULE: Final[str] = (
    "exact_receipt_and_full_chain_pin_bytes_must_be_retained_outside_mutable_seal_namespace"
)
_OWNED_DEVELOPMENT_POLICY_REPLAY_BATCH_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-policy-replay-batch-v2"
)
_DEVELOPMENT_POLICY_PREFIX_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "initial_event_sequence_sha256",
        "event_sequence_sha256",
        "row_count",
        "genesis_sha256",
        "parent_prefix_sha256",
        "parent_tip_sha256",
        "appended_row_sha256",
        "tip_sha256",
        "rows_sha256",
        "rows",
        "prediction_prefix_sha256",
    }
)
_DEVELOPMENT_POLICY_EVENT_BINDING_KEYS: Final[tuple[str, ...]] = (
    "accession_number",
    "form",
    "stage",
    "decision_session",
    "extraction_identity_sha256",
    "market_prefix_chain_identity_sha256",
    "market_feature_row_sha256",
    "fold_id",
)
_DEVELOPMENT_POLICY_REPLAY_BATCH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "artifact_stage",
        "development_root_scope_sha256",
        "development_policy_replay_plan_sha256",
        "source_prediction_projection_sha256",
        "source_prediction_batch_schema_version",
        "source_prediction_batch_sha256",
        "source_raw_prediction_rows_sha256",
        "source_raw_prediction_tip_sha256",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "source_market_calendar_sessions_sha256",
        "policy_calendar_sessions_sha256",
        "development_cutoff_session",
        "source_prediction_event_count",
        "available_prediction_count",
        "unavailable_prediction_count",
        "candidate_count",
        "candidate_ids",
        "model_variant_count",
        "model_variant_ids",
        "candidate_grid",
        "candidate_grid_sha256",
        "threshold_comparison_rule",
        "cash_episode_rule",
        "unavailable_prediction_rule",
        "policy_replay_order_rule",
        "policy_replay_input_count",
        "policy_replay_input_specs_sha256",
        "policy_prefix",
        "policy_prefix_sha256",
        "binding_count",
        "raw_to_policy_bindings",
        "raw_to_policy_bindings_sha256",
        "prediction_components_equal_source",
        "policy_prefix_included",
        "raw_to_policy_bindings_included",
        "threshold_evaluation_authorized",
        "policy_transition_authorized",
        "source_feature_rows_included",
        "learner_states_included",
        "labels_included",
        "outcomes_included",
        "market_prices_included",
        "post_2018_prediction_rows_included",
        "post_2018_market_or_outcome_data_included",
        "calendar_schedule_after_cutoff_used_only_for_open_episode_dates",
        "numeric_prediction_authorized",
        "learner_fit_authorized",
        "model_transport_authorized",
        "network_access_authorized",
        "scoring_authorized",
        "candidate_selection_authorized",
        "ranking_authorized",
        "sealing_authorized",
        "refit_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
        "policy_replay_batch_sha256",
    }
)
_DEVELOPMENT_POLICY_PREFIX_SEAL_INPUT_SPEC_KEYS: Final[frozenset[str]] = (
    frozenset(
        {
            "schema_version",
            "seal_ordinal",
            "policy_sequence_number",
            "decision_session",
            "accession_number",
            "prediction_row_sha256",
            "parent_prediction_prefix_sha256",
            "parent_prediction_tip_sha256",
            "prediction_prefix_sha256",
            "prediction_tip_sha256",
            "prediction_rows_sha256",
            "artifact_sha256",
            "artifact_size_bytes",
            "policy_prefix_seal_input_spec_sha256",
        }
    )
)
_DEVELOPMENT_POLICY_ARTIFACT_GENESIS_PIN_KEYS: Final[frozenset[str]] = (
    frozenset(
        {
            "schema_version",
            "candidate_sha256",
            "stage",
            "sealed_artifact_count",
            "last_prediction_sequence_number",
            "prediction_prefix_sha256",
            "prediction_tip_sha256",
            "prediction_rows_sha256",
            "artifact_sha256",
            "seal_tip_sha256",
            "external_pin_sha256",
        }
    )
)
_DEVELOPMENT_POLICY_PREFIX_SEAL_PLAN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "plan_kind",
        "artifact_stage",
        "development_root_scope_sha256",
        "start_store_state_bytes_sha256",
        "start_store_state_sha256",
        "start_current_tip_anchor_bytes_sha256",
        "start_current_tip_anchor_sha256",
        "start_current_tip_revision",
        "start_consumed_request_count",
        "source_development_policy_replay_plan_sha256",
        "source_development_policy_replay_projection_sha256",
        "source_development_policy_replay_batch_sha256",
        "source_development_oof_prediction_plan_sha256",
        "source_development_oof_prediction_projection_sha256",
        "source_development_oof_prediction_batch_sha256",
        "source_raw_prediction_rows_sha256",
        "source_raw_prediction_tip_sha256",
        "source_raw_prediction_row_count",
        "source_raw_to_policy_bindings_sha256",
        "source_identity_receipt_sha256",
        "source_role_bindings_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "source_market_calendar_sessions_sha256",
        "policy_calendar_sessions_sha256",
        "development_cutoff_session",
        "policy_prefix_count",
        "final_policy_prefix_sha256",
        "final_policy_tip_sha256",
        "final_policy_rows_sha256",
        "policy_prefix_seal_input_count",
        "policy_prefix_seal_input_specs",
        "policy_prefix_seal_input_specs_sha256",
        "artifact_encoding",
        "artifact_store_schema_version",
        "artifact_sealer_genesis_pin_schema_version",
        "artifact_sealer_genesis_pin_bytes_sha256",
        "artifact_sealer_genesis_pin_sha256",
        "artifact_sealer_genesis_seal_tip_sha256",
        "seal_store_namespace_kind",
        "seal_store_namespace_sha256",
        "authorized_seal_receipt_schema_version",
        "authorized_external_pin_schema_version",
        "seal_batch_rule",
        "seal_recovery_rule",
        "external_pin_rule",
        "source_development_policy_replay_batch_access_permitted",
        "deterministic_policy_replay_validation_permitted",
        "cumulative_policy_prefix_materialization_permitted",
        "canonical_artifact_encoding_permitted",
        "fixed_namespace_artifact_store_write_permitted",
        "atomic_batch_compare_and_swap_permitted",
        "exact_idempotent_recovery_permitted",
        "seal_receipt_emission_permitted",
        "external_pin_emission_permitted",
        "raw_prediction_mutation_permitted",
        "policy_row_drop_permitted",
        "policy_row_reordering_permitted",
        "alternate_policy_retry_permitted",
        "source_feature_batch_access_permitted",
        "source_label_batch_access_permitted",
        "label_access_permitted",
        "outcome_access_permitted",
        "post_decision_market_data_access_permitted",
        "post_2018_data_access_permitted",
        "numeric_prediction_permitted",
        "learner_state_access_permitted",
        "learner_fit_permitted",
        "learner_refit_permitted",
        "learner_state_update_permitted",
        "new_threshold_evaluation_permitted",
        "new_policy_state_transition_permitted",
        "candidate_selection_permitted",
        "scoring_permitted",
        "ranking_permitted",
        "holdout_access_permitted",
        "model_transport_access_permitted",
        "network_access_permitted",
        "mutable_store_pin_discovery_as_external_permitted",
        "ledger_mutation_permitted",
        "stage_promotion_permitted",
        "production_permitted",
        "development_policy_prefix_seal_plan_sha256",
    }
)
_DEVELOPMENT_POLICY_REPLAY_MODEL_VARIANT_IDS: Final[tuple[str, ...]] = (
    "semantic",
    "ablation",
)
_DEVELOPMENT_POLICY_REPLAY_GATE_COMPARISON_RULE: Final[str] = (
    "probability_gte_and_expected_edge_gte"
)
_DEVELOPMENT_POLICY_REPLAY_CASH_EPISODE_RULE: Final[str] = (
    "accepted_after_close_fill_t_plus_1_exit_t_plus_21_fixed_20_session_"
    "cash_episode_never_extend_scheduled_or_active_episode"
)
_DEVELOPMENT_POLICY_REPLAY_UNAVAILABLE_RULE: Final[str] = (
    "unavailable_prediction_starts_no_new_cash_episode_"
    "existing_episode_keeps_original_exit"
)
_DEVELOPMENT_POLICY_REPLAY_ORDER_RULE: Final[str] = (
    "source_raw_prediction_ordinal_ascending_exactly_once"
)
_DEVELOPMENT_OOF_PREDICTION_UNAVAILABLE_REASONS: Final[frozenset[str]] = (
    frozenset(
        {
            "missing_required_market_features",
            "missing_required_extraction_features",
            "missing_required_market_and_extraction_features",
        }
    )
)
_DEVELOPMENT_OOF_PREDICTION_POPULATION_RULE: Final[str] = (
    "all_and_only_source_feature_events_in_frozen_2005_2018_fold_windows_"
    "in_source_order"
)
_DEVELOPMENT_OOF_PREDICTION_INPUT_ORDER_RULE: Final[str] = (
    "prediction_ordinal_and_source_event_ordinal_ascending_with_unique_"
    "strictly_increasing_decision_sessions"
)
_DEVELOPMENT_OOF_PREDICTION_FOLD_STATE_USAGE_RULE: Final[str] = (
    "exact_current_fold_semantic_then_ablation_states_no_cross_fold_or_state_update"
)
_DEVELOPMENT_OOF_MAXIMUM_PREDICTION_SECONDS: Final[int] = 60
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
        "development_market_execution_claims",
        "development_market_reader_receipts",
        "development_market_execution_aborts",
        "development_root_carry_in_reader_receipts",
        "stage_model_execution_claims",
        "stage_model_reader_receipts",
        "stage_model_execution_aborts",
        "development_model_execution_claims",
        "development_model_reader_receipts",
        "development_model_execution_aborts",
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


def _validated_development_market_acquisition_plan(
    raw: Any,
    *,
    execution_source_hashes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = _mapping(raw, "development market acquisition plan")
    _expect_keys(
        plan,
        frozenset(
            {
                "schema_version",
                "provider_family",
                "artifact_stage",
                "request_window",
                "requests",
                "transport_authority",
                "normalization_authority",
                "source_code_sha256s",
                "acquisition_plan_sha256",
            }
        ),
        "development market acquisition plan",
    )
    try:
        expected = _mapping(
            build_development_market_acquisition_plan(),
            "owned development market acquisition plan",
        )
    except Exception as exc:
        raise SecFilingGemmaStageAuthorizationError(
            "Owned development market acquisition plan cannot be reconstructed"
        ) from exc
    if plan != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Development market acquisition plan changed its exact authority"
        )
    source_code_hashes = _mapping(
        plan["source_code_sha256s"],
        "development market acquisition source hashes",
    )
    expected_source_paths = {path for path, _role in _MARKET_ACQUISITION_SOURCE_PATH_ROLES}
    if set(source_code_hashes) != expected_source_paths:
        raise SecFilingGemmaStageAuthorizationError(
            "Development market acquisition source closure is incomplete"
        )
    normalized_source_hashes = {
        path: _sha256(
            source_code_hashes[path],
            f"development market acquisition source {path}",
        )
        for path, _role in _MARKET_ACQUISITION_SOURCE_PATH_ROLES
    }
    if plan["source_code_sha256s"] != normalized_source_hashes:
        raise SecFilingGemmaStageAuthorizationError(
            "Development market acquisition source hashes are not canonical"
        )
    if execution_source_hashes is not None:
        execution_sources = _mapping(
            execution_source_hashes,
            "development market execution source hashes for acquisition plan",
        )
        if any(
            execution_sources.get(role) != normalized_source_hashes[path]
            for path, role in _MARKET_ACQUISITION_SOURCE_PATH_ROLES
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development market acquisition plan differs from its execution source closure"
            )
    return plan


def _validated_market_symbol_sha256s(raw: Any, location: str) -> dict[str, str]:
    values = _mapping(raw, location)
    if set(values) != set(MARKET_SYMBOLS):
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must contain exactly the frozen market symbols"
        )
    return {
        symbol: _sha256(values[symbol], f"{location}.{symbol}")
        for symbol in MARKET_SYMBOLS
    }


def _expected_development_market_byte_layout() -> tuple[tuple[str, str], ...]:
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
    if len(layout) != len(MARKET_SYMBOLS) * 3 + 4:
        raise SecFilingGemmaStageAuthorizationError(
            "Development market byte layout count changed"
        )
    return tuple(layout)


def _validated_development_market_byte_index(
    raw: Any,
    *,
    raw_response_sha256s: Mapping[str, str],
    artifact_sha256s: Mapping[str, str],
    window_sha256s: Mapping[str, str],
) -> list[dict[str, Any]]:
    index = _validated_sec_byte_index(raw)
    expected_layout = _expected_development_market_byte_layout()
    observed_layout = tuple(
        (item["logical_id"], item["relative_path"]) for item in index
    )
    logical_ids = [item["logical_id"].casefold() for item in index]
    paths = [item["relative_path"].casefold() for item in index]
    if (
        observed_layout != expected_layout
        or len(logical_ids) != len(set(logical_ids))
        or len(paths) != len(set(paths))
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development market byte index changed its exact ordered layout"
        )

    expected_hashes: dict[str, str] = {}
    for symbol in MARKET_SYMBOLS:
        expected_hashes[f"raw-response-{symbol}.json"] = raw_response_sha256s[
            symbol
        ]
        expected_hashes[f"artifact-{symbol}.json"] = artifact_sha256s[symbol]
        expected_hashes[f"window-{symbol}.json"] = window_sha256s[symbol]
    if any(
        item["sha256"] != expected_hashes[item["relative_path"]]
        for item in index
        if item["relative_path"] in expected_hashes
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development market byte index crossed its symbol payload hashes"
        )
    return index


_DEVELOPMENT_MARKET_MODEL_BINDING_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("development_market_acquisition_receipt_sha256", "acquisition_receipt_sha256"),
    ("development_market_acquisition_bundle_sha256", "acquisition_bundle_sha256"),
    (
        "development_market_acquisition_validation_sha256",
        "acquisition_validation_sha256",
    ),
    ("development_market_source_manifest_sha256", "source_manifest_sha256"),
    (
        "development_market_stage_manifest_sha256",
        "market_stage_manifest_sha256",
    ),
    (
        "development_market_source_reconciliation_sha256",
        "source_reconciliation_sha256",
    ),
    ("development_market_byte_index_sha256", "byte_index_sha256"),
)


def _development_market_model_bindings(
    *,
    development_root_scope_sha256: str,
    development_market_execution_claims: Mapping[str, Any],
    development_market_reader_receipts: Mapping[str, Any],
    development_market_execution_aborts: Mapping[str, Any],
    location: str,
) -> dict[str, str]:
    scope_hash = _sha256(
        development_root_scope_sha256,
        f"{location} development root scope hash",
    )
    market_claim = development_market_execution_claims.get(scope_hash)
    market_reader = development_market_reader_receipts.get(scope_hash)
    if (
        type(market_claim) is not dict
        or type(market_reader) is not dict
        or scope_hash in development_market_execution_aborts
        or market_reader.get("development_root_scope_sha256") != scope_hash
        or market_reader.get("claim_sha256") != market_claim.get("claim_sha256")
        or market_reader.get("owned_transport_attested_by_store") is not True
    ):
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} requires exactly one successful, non-aborted, "
            "store-attested development market claim and reader for its root"
        )
    bindings = {
        "development_market_execution_claim_sha256": _sha256(
            market_claim.get("claim_sha256"),
            f"{location} development market claim hash",
        ),
        "development_market_reader_receipt_sha256": _sha256(
            market_reader.get("receipt_sha256"),
            f"{location} development market reader receipt hash",
        ),
    }
    for model_field, reader_field in _DEVELOPMENT_MARKET_MODEL_BINDING_FIELDS:
        bindings[model_field] = _sha256(
            market_reader.get(reader_field),
            f"{location} {reader_field}",
        )
    return bindings


def _validated_development_market_execution_claims(
    raw: Any,
    *,
    development_sec_execution_claims: Mapping[str, Any],
    development_sec_reader_receipts: Mapping[str, Any],
    development_sec_execution_aborts: Mapping[str, Any],
    authenticated_store_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    claims = _mapping(raw, "current-tip development market execution claims")
    validated: dict[str, dict[str, Any]] = {}
    for raw_scope_sha256, raw_claim in claims.items():
        scope_hash = _sha256(raw_scope_sha256, "development market claim map key")
        claim = _mapping(raw_claim, f"development market execution claim {scope_hash}")
        _expect_keys(
            claim,
            _DEVELOPMENT_MARKET_EXECUTION_CLAIM_KEYS,
            f"development market execution claim {scope_hash}",
        )
        if (
            claim["schema_version"] != DEVELOPMENT_MARKET_EXECUTION_CLAIM_SCHEMA_VERSION
            or claim["contract_version"] != CONTRACT_VERSION
            or claim["claim_kind"] != "owned_development_market_evidence_batch"
            or claim["authorized_stage"] != "development"
            or claim["market_component_id"] != DEVELOPMENT_MARKET_BATCH_COMPONENT_ID
            or claim["source_family"] != MARKET_SOURCE_FAMILY
            or claim["market_symbols"] != list(MARKET_SYMBOLS)
            or claim["market_fields"] != list(MARKET_FIELDS)
            or claim["fixed_request_count"] != YAHOO_REQUEST_COUNT
            or claim["response_byte_ceiling_per_symbol"] != YAHOO_MAX_RESPONSE_BYTES
            or claim["timeout_seconds_per_symbol"] != YAHOO_REQUEST_TIMEOUT_SECONDS
            or claim["owned_market_execution_required"] is not True
            or claim["caller_supplied_path_permitted"] is not False
            or claim["caller_supplied_bytes_permitted"] is not False
            or claim["market_access_permitted"] is not True
            or claim["outcome_access_permitted"] is not False
            or claim["future_stage_access_permitted"] is not False
            or claim["paid_api_access_permitted"] is not False
            or claim["external_network_access_permitted"] is not True
            or claim["reveal_request_consumption_permitted"] is not False
            or claim["consumption_ledger_mutation_permitted"] is not False
            or claim["effect_may_be_repeated_after_indeterminate_crash"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development market execution claim semantics changed"
            )
        _self_hash(claim, "claim_sha256", "development market execution claim")
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
            "development_sec_execution_claim_sha256",
            "development_sec_reader_receipt_sha256",
            "market_acquisition_plan_sha256",
            "candidate_source_hashes_sha256",
            "execution_source_hashes_sha256",
        ):
            _sha256(claim[field], f"development market claim {field}")
        _safe_id(claim["attempt_id"], "development market attempt id")
        _strict_int(claim["registered_entry_count"], "development market registry count", minimum=1)
        _strict_int(claim["start_consumed_request_count"], "development market consumed request count")
        if claim["development_root_scope_sha256"] != scope_hash:
            raise SecFilingGemmaStageAuthorizationError(
                "Development market claim crossed its root scope"
            )
        sec_claim = development_sec_execution_claims.get(scope_hash)
        sec_reader = development_sec_reader_receipts.get(scope_hash)
        if (
            type(sec_claim) is not dict
            or type(sec_reader) is not dict
            or scope_hash in development_sec_execution_aborts
            or claim["development_sec_execution_claim_sha256"]
            != sec_claim.get("claim_sha256")
            or claim["development_sec_reader_receipt_sha256"]
            != sec_reader.get("receipt_sha256")
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development market claim lacks exact terminal development SEC ancestry"
            )
        expected_ancestry = {
            "development_content_root_plan_sha256": sec_claim.get(
                "development_content_root_plan_sha256"
            ),
            "attempt_id": sec_claim.get("attempt_id"),
            "candidate_sha256": sec_claim.get("candidate_sha256"),
            "candidate_design_sha256": sec_claim.get("candidate_design_sha256"),
            "registry_entry_sha256": sec_claim.get("registry_entry_sha256"),
            "registry_sha256": sec_claim.get("registry_sha256"),
            "registry_tip_sha256": sec_claim.get("registry_tip_sha256"),
            "registered_entry_count": sec_claim.get("registered_entry_count"),
            "corpus_universe_sha256": sec_claim.get("corpus_universe_sha256"),
            "corpus_universe_semantic_sha256": sec_claim.get(
                "corpus_universe_semantic_sha256"
            ),
            "output_namespace": sec_claim.get("output_namespace"),
        }
        if any(claim[field] != value for field, value in expected_ancestry.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "Development market claim crossed its development SEC root"
            )
        sources_raw = _mapping(
            claim["execution_source_hashes"],
            "development market execution source hashes",
        )
        if set(sources_raw) != set(MARKET_EXECUTION_SOURCE_ROLES):
            raise SecFilingGemmaStageAuthorizationError(
                "Development market execution source closure is incomplete"
            )
        sources = {
            role: _sha256(sources_raw[role], f"development market source {role}")
            for role in MARKET_EXECUTION_SOURCE_ROLES
        }
        if (
            claim["execution_source_hashes"] != sources
            or claim["execution_source_hashes_sha256"] != canonical_sha256(sources)
            or claim["execution_source_role_count"] != len(sources)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development market execution source closure is inconsistent"
            )
        plan = _validated_development_market_acquisition_plan(
            claim["market_acquisition_plan"],
            execution_source_hashes=sources,
        )
        if claim["market_acquisition_plan_sha256"] != plan["acquisition_plan_sha256"]:
            raise SecFilingGemmaStageAuthorizationError(
                "Development market claim changed its acquisition-plan identity"
            )
        if authenticated_store_snapshot is not None:
            state, registry_entry, candidate = _development_registered_candidate(
                authenticated_store_snapshot,
                plan=sec_claim["development_content_root_plan"],
                require_latest=False,
            )
            candidate_sources = _mapping(
                candidate["bindings"]["source_hashes"],
                "development market candidate source hashes",
            )
            if (
                claim["registry_entry_sha256"] != registry_entry["entry_sha256"]
                or claim["candidate_source_hashes_sha256"]
                != canonical_sha256(candidate_sources)
                or any(candidate_sources.get(role) != digest for role, digest in sources.items())
            ):
                raise SecFilingGemmaStageAuthorizationError(
                    "Development market claim crossed its registered candidate"
                )
        validated[scope_hash] = claim
    return validated


def _validated_development_market_reader_receipts(
    raw: Any,
    *,
    claims: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    receipts = _mapping(raw, "current-tip development market reader receipts")
    validated: dict[str, dict[str, Any]] = {}
    for raw_scope_sha256, raw_receipt in receipts.items():
        scope_hash = _sha256(raw_scope_sha256, "development market reader map key")
        receipt = _mapping(raw_receipt, f"development market reader receipt {scope_hash}")
        _expect_keys(
            receipt,
            _DEVELOPMENT_MARKET_READER_RECEIPT_KEYS,
            f"development market reader receipt {scope_hash}",
        )
        if (
            receipt["schema_version"] != DEVELOPMENT_MARKET_READER_RECEIPT_SCHEMA_VERSION
            or receipt["contract_version"] != CONTRACT_VERSION
            or receipt["receipt_kind"]
            != "store_rehashed_owned_development_market_evidence_batch"
            or receipt["authorized_stage"] != "development"
            or receipt["market_component_id"] != DEVELOPMENT_MARKET_BATCH_COMPONENT_ID
            or receipt["fresh_network_provenance_claimed"] is not False
            or receipt["provider_response_normalization_replayed_by_store"]
            is not True
            or receipt["owned_transport_attested_by_store"] is not True
            or receipt["reader_output_recomputed_by_store"] is not True
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development market reader receipt semantics changed"
            )
        _self_hash(receipt, "receipt_sha256", "development market reader receipt")
        raw_hashes = _validated_market_symbol_sha256s(
            receipt["raw_response_sha256s"], "development market raw response hashes"
        )
        artifact_hashes = _validated_market_symbol_sha256s(
            receipt["artifact_sha256s"], "development market artifact hashes"
        )
        window_hashes = _validated_market_symbol_sha256s(
            receipt["window_sha256s"], "development market window hashes"
        )
        index = _validated_development_market_byte_index(
            receipt["byte_index"],
            raw_response_sha256s=raw_hashes,
            artifact_sha256s=artifact_hashes,
            window_sha256s=window_hashes,
        )
        if (
            receipt["byte_index"] != index
            or receipt["byte_index_sha256"] != canonical_sha256(index)
            or receipt["byte_count_total"] != sum(item["byte_count"] for item in index)
            or receipt["raw_response_sha256s"] != raw_hashes
            or receipt["artifact_sha256s"] != artifact_hashes
            or receipt["window_sha256s"] != window_hashes
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development market reader receipt evidence is inconsistent"
            )
        for field in (
            "development_root_scope_sha256",
            "claim_sha256",
            "development_content_root_plan_sha256",
            "candidate_sha256",
            "development_sec_execution_claim_sha256",
            "development_sec_reader_receipt_sha256",
            "market_acquisition_plan_sha256",
            "execution_source_hashes_sha256",
            "acquisition_receipt_sha256",
            "acquisition_bundle_sha256",
            "acquisition_validation_sha256",
            "source_manifest_sha256",
            "market_stage_manifest_sha256",
            "source_reconciliation_sha256",
            "byte_index_sha256",
            "complete_marker_sha256",
        ):
            _sha256(receipt[field], f"development market reader {field}")
        claim = claims.get(scope_hash)
        expected = {
            "development_root_scope_sha256": scope_hash,
            "claim_sha256": claim.get("claim_sha256") if type(claim) is dict else None,
            "development_content_root_plan_sha256": claim.get(
                "development_content_root_plan_sha256"
            ) if type(claim) is dict else None,
            "authorized_stage": "development",
            "candidate_sha256": claim.get("candidate_sha256") if type(claim) is dict else None,
            "output_namespace": claim.get("output_namespace") if type(claim) is dict else None,
            "development_sec_execution_claim_sha256": claim.get(
                "development_sec_execution_claim_sha256"
            ) if type(claim) is dict else None,
            "development_sec_reader_receipt_sha256": claim.get(
                "development_sec_reader_receipt_sha256"
            ) if type(claim) is dict else None,
            "market_acquisition_plan_sha256": claim.get(
                "market_acquisition_plan_sha256"
            ) if type(claim) is dict else None,
            "execution_source_hashes_sha256": claim.get(
                "execution_source_hashes_sha256"
            ) if type(claim) is dict else None,
            "execution_source_role_count": claim.get(
                "execution_source_role_count"
            ) if type(claim) is dict else None,
            "market_component_id": DEVELOPMENT_MARKET_BATCH_COMPONENT_ID,
        }
        if type(claim) is not dict or any(
            receipt[field] != value for field, value in expected.items()
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development market reader receipt crossed its execution claim"
            )
        validated[scope_hash] = receipt
    return validated


def _validated_development_market_execution_aborts(
    raw: Any,
    *,
    claims: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    aborts = _mapping(raw, "current-tip development market execution aborts")
    validated: dict[str, dict[str, Any]] = {}
    for raw_scope_sha256, raw_abort in aborts.items():
        scope_hash = _sha256(raw_scope_sha256, "development market abort map key")
        abort = _mapping(raw_abort, f"development market execution abort {scope_hash}")
        _expect_keys(
            abort,
            _DEVELOPMENT_MARKET_EXECUTION_ABORT_KEYS,
            f"development market execution abort {scope_hash}",
        )
        if (
            abort["schema_version"] != DEVELOPMENT_MARKET_EXECUTION_ABORT_SCHEMA_VERSION
            or abort["contract_version"] != CONTRACT_VERSION
            or abort["abort_kind"] != "indeterminate_owned_development_market_evidence_batch"
            or abort["authorized_stage"] != "development"
            or abort["market_component_id"] != DEVELOPMENT_MARKET_BATCH_COMPONENT_ID
            or abort["reason"] not in {
                "claim_recovered_without_terminal_receipt",
                "external_effect_failed_or_completion_unknown",
                "durable_output_verification_failed",
            }
            or abort["external_effect_retry_permitted"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development market execution abort semantics changed"
            )
        _self_hash(abort, "abort_sha256", "development market execution abort")
        claim = claims.get(scope_hash)
        expected = {
            "development_root_scope_sha256": scope_hash,
            "claim_sha256": claim.get("claim_sha256") if type(claim) is dict else None,
            "development_content_root_plan_sha256": claim.get(
                "development_content_root_plan_sha256"
            ) if type(claim) is dict else None,
            "authorized_stage": "development",
            "candidate_sha256": claim.get("candidate_sha256") if type(claim) is dict else None,
            "output_namespace": claim.get("output_namespace") if type(claim) is dict else None,
            "market_component_id": DEVELOPMENT_MARKET_BATCH_COMPONENT_ID,
        }
        if type(claim) is not dict or any(
            abort[field] != value for field, value in expected.items()
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development market execution abort crossed its claim"
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


def _validated_model_accession_order(raw: Any, *, location: str) -> list[str]:
    if type(raw) is not list or not raw:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must be a non-empty exact list"
        )
    accessions = [
        _safe_id(value, f"{location} item {index}")
        for index, value in enumerate(raw)
    ]
    if (
        any(_AAPL_ACCESSION_RE.fullmatch(value) is None for value in accessions)
        or accessions != sorted(accessions)
        or len(accessions) != len(set(accessions))
    ):
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} is not the exact canonical Apple accession order"
        )
    return accessions


def _model_event_plan(
    *,
    universe_manifest: Mapping[str, Any],
    stage: str,
    sec_acquisition_accession_order: list[str],
) -> list[dict[str, Any]]:
    if stage not in {"development", "intermediate", "final"}:
        raise SecFilingGemmaStageAuthorizationError(
            "Owned model event-plan stage is invalid"
        )
    universe = _mapping(universe_manifest, "owned model corpus universe")
    records = universe.get("records")
    if type(records) is not list:
        raise SecFilingGemmaStageAuthorizationError(
            "Owned model corpus universe has no exact records"
        )
    acquisition_order = _validated_model_accession_order(
        sec_acquisition_accession_order,
        location="owned model SEC acquisition accession order",
    )
    sec_ordinals = {
        accession: ordinal
        for ordinal, accession in enumerate(acquisition_order, start=1)
    }
    stage_records: list[dict[str, str]] = []
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"owned model universe record {index}")
        if record.get("artifact_stage") != stage:
            continue
        accession = _safe_id(
            record.get("accession_number"),
            f"owned model universe accession {index}",
        )
        form = record.get("form")
        availability = record.get("availability_session")
        if (
            _AAPL_ACCESSION_RE.fullmatch(accession) is None
            or form not in {"10-K", "10-Q"}
            or type(availability) is not str
            or _ISO_DATE_RE.fullmatch(availability) is None
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Owned model universe event identity is invalid"
            )
        stage_records.append(
            {
                "accession_number": accession,
                "form": form,
                "availability_session": availability,
            }
        )
    stage_records.sort(
        key=lambda record: (
            record["availability_session"],
            record["accession_number"],
        )
    )
    if (
        len(stage_records) != len(acquisition_order)
        or {record["accession_number"] for record in stage_records}
        != set(acquisition_order)
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Owned model event plan differs from the exact stage SEC document set"
        )
    return [
        {
            "event_ordinal": event_ordinal,
            "accession_number": record["accession_number"],
            "form": record["form"],
            "availability_session": record["availability_session"],
            "sec_document_ordinal": sec_ordinals[record["accession_number"]],
        }
        for event_ordinal, record in enumerate(stage_records, start=1)
    ]


def _validated_model_event_plan(
    raw: Any,
    *,
    universe_manifest: Mapping[str, Any],
    stage: str,
    sec_acquisition_accession_order: list[str],
) -> list[dict[str, Any]]:
    if type(raw) is not list:
        raise SecFilingGemmaStageAuthorizationError(
            "Owned model event plan must be an exact list"
        )
    observed = _plain(raw, "owned model event plan")
    expected = _model_event_plan(
        universe_manifest=universe_manifest,
        stage=stage,
        sec_acquisition_accession_order=sec_acquisition_accession_order,
    )
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Owned model event plan is not the canonical chronological stage plan"
        )
    return expected


def _validated_model_execution_source_hashes(
    raw: Any,
    *,
    location: str,
) -> dict[str, str]:
    sources = _mapping(raw, location)
    if set(sources) != set(MODEL_EXECUTION_SOURCE_ROLES):
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} does not bind the exact owned model source closure"
        )
    return {
        role: _sha256(sources[role], f"{location} {role}")
        for role in MODEL_EXECUTION_SOURCE_ROLES
    }


def _expected_model_runtime_limits(
    *,
    stage: str,
    accession_count: int,
) -> dict[str, Any]:
    if stage not in STAGE_MODEL_CALL_CAPS:
        raise SecFilingGemmaStageAuthorizationError(
            "Owned model execution stage is invalid"
        )
    count = _strict_int(
        accession_count,
        "owned model execution accession count",
        minimum=1,
    )
    call_cap = STAGE_MODEL_CALL_CAPS[stage]
    if count > call_cap:
        raise SecFilingGemmaStageAuthorizationError(
            "Owned model execution exceeds its frozen stage call cap"
        )
    return {
        "model_call_count": count,
        "model_call_cap": call_cap,
        "maximum_model_seconds": MAX_MODEL_SECONDS,
        "maximum_input_utf8_bytes": MAX_INPUT_BYTES,
        "maximum_sentences": MAX_SENTENCES,
        "maximum_sentence_characters": MAX_SENTENCE_CHARACTERS,
        "connect_timeout_seconds": 2,
        "read_timeout_seconds": 30,
        "redirects": 0,
        "retries": 0,
        "pull_attempts": 0,
        "repair_attempts": 0,
        "streaming": False,
        "thinking": False,
    }


def _validated_model_runtime_limits(
    raw: Any,
    *,
    stage: str,
    accession_count: int,
    location: str,
) -> dict[str, Any]:
    limits = _mapping(raw, location)
    expected = _expected_model_runtime_limits(
        stage=stage,
        accession_count=accession_count,
    )
    if limits != expected:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} changed the frozen local-model limits"
        )
    return limits


def _stage_model_candidate_context(
    authorization_bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    bundle, grant, component_plan = _sec_component_plan_from_bundle(
        authorization_bundle
    )
    registry = _mapping(
        bundle["authenticated_store_snapshot"].get("latest_registry"),
        "stage model candidate registry",
    )
    raw_entries = registry.get("entries")
    if type(raw_entries) is not list:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model candidate registry has no exact entries"
        )
    matches: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_entries):
        entry = _mapping(raw_entry, f"stage model registry entry {index}")
        candidate = entry.get("candidate_manifest")
        if (
            entry.get("entry_sha256") == grant["registry_entry_sha256"]
            and type(candidate) is dict
            and candidate.get("candidate_sha256") == grant["candidate_sha256"]
        ):
            matches.append(_mapping(candidate, "stage model candidate manifest"))
    if len(matches) != 1:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model grant lacks exactly one registry-pinned candidate"
        )
    candidate = matches[0]
    try:
        validate_candidate_manifest(
            candidate,
            expected_candidate_sha256=grant["candidate_sha256"],
        )
    except Exception as exc:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model candidate manifest is not canonical"
        ) from exc
    accessions = [
        document["accession_number"]
        for document in component_plan["sec_access_plan"]["documents"]
    ]
    return bundle, grant, candidate, accessions


def _stage_model_development_root_context(
    *,
    current_tip: Mapping[str, Any],
    bundle: Mapping[str, Any],
    grant: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    matches: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    candidate_bindings = _mapping(
        candidate.get("bindings"),
        "stage model candidate bindings for development root",
    )
    candidate_sources = _mapping(
        candidate_bindings.get("source_hashes"),
        "stage model candidate source hashes for development root",
    )
    snapshot = _mapping(
        bundle.get("authenticated_store_snapshot"),
        "stage model bundled snapshot for development root",
    )
    registry = _mapping(
        snapshot.get("latest_registry"),
        "stage model registry for development root",
    )
    registry_pin = _mapping(
        snapshot.get("latest_registry_pin"),
        "stage model registry pin for development root",
    )
    claims = _mapping(
        current_tip.get("development_sec_execution_claims"),
        "stage model development SEC claims",
    )
    readers = _mapping(
        current_tip.get("development_sec_reader_receipts"),
        "stage model development SEC readers",
    )
    aborts = _mapping(
        current_tip.get("development_sec_execution_aborts"),
        "stage model development SEC aborts",
    )
    for scope_hash, root_claim in claims.items():
        root_reader = readers.get(scope_hash)
        if (
            type(root_claim) is not dict
            or type(root_reader) is not dict
            or scope_hash in aborts
            or root_claim.get("candidate_sha256") != grant["candidate_sha256"]
            or root_claim.get("registry_entry_sha256")
            != grant["registry_entry_sha256"]
            or root_claim.get("registry_sha256") != registry.get("registry_sha256")
            or root_claim.get("registry_tip_sha256") != registry_pin.get("tip_sha256")
            or root_claim.get("corpus_universe_sha256")
            != candidate_bindings.get("corpus_universe_sha256")
            or root_claim.get("corpus_universe_semantic_sha256")
            != candidate_bindings.get("corpus_universe_semantic_sha256")
        ):
            continue
        plan = _validated_development_content_root_plan(
            root_claim.get("development_content_root_plan")
        )
        root_sources = _mapping(
            root_claim.get("execution_source_hashes"),
            "stage model development-root execution source hashes",
        )
        if (
            plan["development_root_scope_sha256"] != scope_hash
            or root_reader.get("claim_sha256") != root_claim.get("claim_sha256")
            or any(
                candidate_sources.get(role) != value
                for role, value in root_sources.items()
            )
        ):
            continue
        universe = _mapping(
            plan.get("corpus_universe_manifest"),
            "stage model development-root universe",
        )
        if (
            universe.get("universe_sha256")
            != candidate_bindings.get("corpus_universe_sha256")
            or universe.get("universe_semantic_sha256")
            != candidate_bindings.get("corpus_universe_semantic_sha256")
        ):
            continue
        matches.append((root_claim, root_reader, universe))
    if len(matches) != 1:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model execution requires exactly one terminal matching development SEC root"
        )
    return matches[0]


def _validated_stage_model_execution_claims(
    raw: Any,
    *,
    authorization_bundles: Mapping[str, Any],
    sec_execution_claims: Mapping[str, Any],
    sec_reader_receipts: Mapping[str, Any],
    sec_execution_aborts: Mapping[str, Any],
    stage_carry_in_reader_receipts: Mapping[str, Any],
    development_root_carry_in_reader_receipts: Mapping[str, Any],
    development_sec_execution_claims: Mapping[str, Any],
    development_sec_reader_receipts: Mapping[str, Any],
    development_sec_execution_aborts: Mapping[str, Any],
    development_market_execution_claims: Mapping[str, Any],
    development_market_reader_receipts: Mapping[str, Any],
    development_market_execution_aborts: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    claims = _mapping(raw, "current-tip stage model execution claims")
    validated: dict[str, dict[str, Any]] = {}
    for raw_request_sha256, raw_claim in claims.items():
        request_hash = _sha256(
            raw_request_sha256,
            "stage model execution claim map key",
        )
        claim = _mapping(
            raw_claim,
            f"stage model execution claim {request_hash}",
        )
        _expect_keys(
            claim,
            _STAGE_MODEL_EXECUTION_CLAIM_KEYS,
            f"stage model execution claim {request_hash}",
        )
        if (
            claim["schema_version"] != STAGE_MODEL_EXECUTION_CLAIM_SCHEMA_VERSION
            or claim["contract_version"] != CONTRACT_VERSION
            or claim["claim_kind"] != "owned_stage_gemma_model_batch"
            or claim["request_sha256"] != request_hash
            or claim["model_component_id"] != STAGE_MODEL_BATCH_COMPONENT_ID
            or claim["owned_local_model_execution_required"] is not True
            or claim["caller_supplied_path_permitted"] is not False
            or claim["filing_text_included"] is not False
            or claim["market_access_permitted"] is not False
            or claim["outcome_access_permitted"] is not False
            or claim["future_stage_access_permitted"] is not False
            or claim["paid_api_access_permitted"] is not False
            or claim["external_network_access_permitted"] is not False
            or claim["effect_may_be_repeated_after_indeterminate_crash"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution claim semantics changed"
            )
        _self_hash(claim, "claim_sha256", "stage model execution claim")
        stage = claim["authorized_stage"]
        if stage not in _STAGE_PREREQUISITES:
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution claim stage is invalid"
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
            "stage_sec_execution_claim_sha256",
            "stage_sec_reader_receipt_sha256",
            "development_root_scope_sha256",
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
            "corpus_universe_sha256",
            "carry_in_reader_receipt_sha256",
            "sec_acquisition_accession_order_sha256",
            "event_plan_sha256",
            "identity_lexicon_sha256",
            "candidate_source_hashes_sha256",
            "execution_source_hashes_sha256",
            "model_digest",
            "runtime_fingerprint_sha256",
            "model_transport_sha256",
            "model_runtime_limits_sha256",
        ):
            _sha256(claim[field], f"stage model execution claim {field}")
        for field in ("attempt_id", "input_prerequisite_stage", "model_name"):
            _safe_id(claim[field], f"stage model execution claim {field}")
        if claim["input_prerequisite_stage"] != _STAGE_PREREQUISITES[stage]:
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution claim crossed its prerequisite stage"
            )
        namespace = claim["output_namespace"]
        if type(namespace) is not str or _OUTPUT_NAMESPACE_RE.fullmatch(namespace) is None:
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution claim namespace is invalid"
            )
        _strict_int(
            claim["consumption_entry_sequence"],
            "stage model execution entry sequence",
            minimum=1,
        )
        acquisition_accessions = _validated_model_accession_order(
            claim["sec_acquisition_accession_order"],
            location="stage model execution SEC acquisition order",
        )
        if (
            claim["sec_document_count"] != len(acquisition_accessions)
            or claim["sec_acquisition_accession_order_sha256"]
            != canonical_sha256(acquisition_accessions)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution SEC acquisition binding is inconsistent"
            )
        sources = _validated_model_execution_source_hashes(
            claim["execution_source_hashes"],
            location="stage model execution source hashes",
        )
        if (
            claim["execution_source_hashes"] != sources
            or claim["execution_source_hashes_sha256"] != canonical_sha256(sources)
            or claim["execution_source_role_count"] != len(sources)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution source closure is inconsistent"
            )
        limits = _validated_model_runtime_limits(
            claim["model_runtime_limits"],
            stage=stage,
            accession_count=len(acquisition_accessions),
            location="stage model runtime limits",
        )
        if claim["model_runtime_limits_sha256"] != canonical_sha256(limits):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model runtime-limit hash is inconsistent"
            )
        bundle = authorization_bundles.get(request_hash)
        sec_claim = sec_execution_claims.get(request_hash)
        sec_reader = sec_reader_receipts.get(request_hash)
        if (
            type(bundle) is not dict
            or type(sec_claim) is not dict
            or type(sec_reader) is not dict
            or request_hash in sec_execution_aborts
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution claim lacks terminal SEC ancestry"
            )
        final_carry = stage_carry_in_reader_receipts.get(request_hash)
        root_carry = development_root_carry_in_reader_receipts.get(request_hash)
        if stage == "intermediate":
            carry = root_carry
            carry_kind = "development_root_carry_in"
            other_carry = final_carry
        else:
            carry = final_carry
            carry_kind = "stage_carry_in"
            other_carry = root_carry
        if type(carry) is not dict or other_carry is not None:
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution claim requires exactly one stage-correct carry receipt"
            )
        canonical_bundle, grant, candidate, planned_accessions = (
            _stage_model_candidate_context(bundle)
        )
        root_claim, root_reader, universe = _stage_model_development_root_context(
            current_tip={
                "development_sec_execution_claims": (
                    development_sec_execution_claims
                ),
                "development_sec_reader_receipts": (
                    development_sec_reader_receipts
                ),
                "development_sec_execution_aborts": (
                    development_sec_execution_aborts
                ),
            },
            bundle=canonical_bundle,
            grant=grant,
            candidate=candidate,
        )
        market_bindings = _development_market_model_bindings(
            development_root_scope_sha256=root_claim[
                "development_root_scope_sha256"
            ],
            development_market_execution_claims=(
                development_market_execution_claims
            ),
            development_market_reader_receipts=(
                development_market_reader_receipts
            ),
            development_market_execution_aborts=(
                development_market_execution_aborts
            ),
            location="Stage model execution claim",
        )
        event_plan = _validated_model_event_plan(
            claim["event_plan"],
            universe_manifest=universe,
            stage=stage,
            sec_acquisition_accession_order=planned_accessions,
        )
        if (
            claim["event_count"] != len(event_plan)
            or claim["event_plan_sha256"] != canonical_sha256(event_plan)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution event-plan binding is inconsistent"
            )
        candidate_sources = _mapping(
            _mapping(candidate["bindings"], "stage model candidate bindings")[
                "source_hashes"
            ],
            "stage model candidate source hashes",
        )
        if (
            claim["identity_lexicon_sha256"]
            != CANONICAL_IDENTITY_LEXICON_SHA256
            or candidate["bindings"]["identity_lexicon_sha256"]
            != CANONICAL_IDENTITY_LEXICON_SHA256
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution changed the canonical identity lexicon"
            )
        model = _mapping(candidate["model"], "stage model candidate model")
        expected = {
            "request_sha256": grant["request_sha256"],
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
            "authorization_bundle_sha256": canonical_bundle["bundle_sha256"],
            "authorization_grant_sha256": grant["authorization_grant_sha256"],
            "grant_store_state_sha256": grant["store_state_sha256"],
            "grant_consumption_ledger_sha256": grant["consumption_ledger_sha256"],
            "grant_consumption_ledger_tip_sha256": grant[
                "consumption_ledger_tip_sha256"
            ],
            "stage_sec_execution_claim_sha256": sec_claim["claim_sha256"],
            "stage_sec_reader_receipt_sha256": sec_reader["receipt_sha256"],
            "development_root_scope_sha256": root_claim[
                "development_root_scope_sha256"
            ],
            "development_sec_execution_claim_sha256": root_claim[
                "claim_sha256"
            ],
            "development_sec_reader_receipt_sha256": root_reader[
                "receipt_sha256"
            ],
            **market_bindings,
            "corpus_universe_sha256": universe["universe_sha256"],
            "carry_in_kind": carry_kind,
            "carry_in_reader_receipt_sha256": carry["receipt_sha256"],
            "sec_document_count": len(planned_accessions),
            "sec_acquisition_accession_order": planned_accessions,
            "sec_acquisition_accession_order_sha256": canonical_sha256(
                planned_accessions
            ),
            "event_count": len(event_plan),
            "event_plan": event_plan,
            "event_plan_sha256": canonical_sha256(event_plan),
            "identity_lexicon_sha256": CANONICAL_IDENTITY_LEXICON_SHA256,
            "candidate_source_hashes_sha256": canonical_sha256(candidate_sources),
            "model_name": model["name"],
            "model_digest": model["digest"],
            "runtime_fingerprint_sha256": model[
                "runtime_fingerprint_sha256"
            ],
            "model_transport_sha256": canonical_sha256(model["transport"]),
        }
        if (
            any(claim[field] != value for field, value in expected.items())
            or any(candidate_sources.get(role) != value for role, value in sources.items())
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution claim crossed its candidate, SEC, carry, or grant ancestry"
            )
        validated[request_hash] = claim
    return validated


def _validated_stage_model_reader_receipts(
    raw: Any,
    *,
    claims: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    receipts = _mapping(raw, "current-tip stage model reader receipts")
    validated: dict[str, dict[str, Any]] = {}
    for raw_request_sha256, raw_receipt in receipts.items():
        request_hash = _sha256(raw_request_sha256, "stage model reader map key")
        receipt = _mapping(raw_receipt, f"stage model reader receipt {request_hash}")
        _expect_keys(
            receipt,
            _STAGE_MODEL_READER_RECEIPT_KEYS,
            f"stage model reader receipt {request_hash}",
        )
        if (
            receipt["schema_version"] != STAGE_MODEL_READER_RECEIPT_SCHEMA_VERSION
            or receipt["contract_version"] != CONTRACT_VERSION
            or receipt["receipt_kind"] != "store_rehashed_owned_stage_gemma_model_batch"
            or receipt["request_sha256"] != request_hash
            or receipt["model_component_id"] != STAGE_MODEL_BATCH_COMPONENT_ID
            or receipt["fresh_model_provenance_claimed"] is not False
            or receipt["reader_output_recomputed_by_store"] is not True
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model reader receipt semantics changed"
            )
        _self_hash(receipt, "receipt_sha256", "stage model reader receipt")
        index = _validated_sec_byte_index(receipt["byte_index"])
        if (
            receipt["byte_index"] != index
            or receipt["byte_index_sha256"] != canonical_sha256(index)
            or receipt["byte_count_total"]
            != sum(item["byte_count"] for item in index)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model reader byte index is inconsistent"
            )
        for field in (
            "request_sha256",
            "claim_sha256",
            "candidate_sha256",
            "stage_sec_execution_claim_sha256",
            "stage_sec_reader_receipt_sha256",
            "development_root_scope_sha256",
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
            "corpus_universe_sha256",
            "carry_in_reader_receipt_sha256",
            "sec_acquisition_accession_order_sha256",
            "event_plan_sha256",
            "identity_lexicon_sha256",
            "execution_source_hashes_sha256",
            "model_digest",
            "runtime_fingerprint_sha256",
            "model_transport_sha256",
            "model_runtime_limits_sha256",
            "byte_index_sha256",
            "complete_marker_sha256",
        ):
            _sha256(receipt[field], f"stage model reader receipt {field}")
        claim = claims.get(request_hash)
        if type(claim) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model reader receipt lacks its execution claim"
            )
        expected_fields = (
            "authorized_stage",
            "candidate_sha256",
            "output_namespace",
            "stage_sec_execution_claim_sha256",
            "stage_sec_reader_receipt_sha256",
            "development_root_scope_sha256",
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
            "corpus_universe_sha256",
            "carry_in_kind",
            "carry_in_reader_receipt_sha256",
            "sec_document_count",
            "sec_acquisition_accession_order_sha256",
            "event_count",
            "event_plan_sha256",
            "identity_lexicon_sha256",
            "execution_source_hashes_sha256",
            "execution_source_role_count",
            "model_name",
            "model_digest",
            "runtime_fingerprint_sha256",
            "model_transport_sha256",
            "model_runtime_limits_sha256",
            "model_component_id",
        )
        if (
            receipt["claim_sha256"] != claim["claim_sha256"]
            or any(receipt[field] != claim[field] for field in expected_fields)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model reader receipt crossed its execution claim"
            )
        validated[request_hash] = receipt
    return validated


def _validated_stage_model_execution_aborts(
    raw: Any,
    *,
    claims: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    aborts = _mapping(raw, "current-tip stage model execution aborts")
    validated: dict[str, dict[str, Any]] = {}
    for raw_request_sha256, raw_abort in aborts.items():
        request_hash = _sha256(raw_request_sha256, "stage model abort map key")
        abort = _mapping(raw_abort, f"stage model execution abort {request_hash}")
        _expect_keys(
            abort,
            _STAGE_MODEL_EXECUTION_ABORT_KEYS,
            f"stage model execution abort {request_hash}",
        )
        if (
            abort["schema_version"] != STAGE_MODEL_EXECUTION_ABORT_SCHEMA_VERSION
            or abort["contract_version"] != CONTRACT_VERSION
            or abort["abort_kind"] != "indeterminate_owned_stage_gemma_model_batch"
            or abort["reason"]
            not in {
                "claim_recovered_without_terminal_receipt",
                "external_effect_failed_or_completion_unknown",
                "durable_output_verification_failed",
            }
            or abort["external_effect_retry_permitted"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution abort semantics changed"
            )
        _self_hash(abort, "abort_sha256", "stage model execution abort")
        claim = claims.get(request_hash)
        if type(claim) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution abort lacks its claim"
            )
        expected = {
            "request_sha256": request_hash,
            "claim_sha256": claim["claim_sha256"],
            "authorized_stage": claim["authorized_stage"],
            "candidate_sha256": claim["candidate_sha256"],
            "output_namespace": claim["output_namespace"],
            "model_component_id": claim["model_component_id"],
        }
        if any(abort[field] != value for field, value in expected.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution abort crossed its claim"
            )
        validated[request_hash] = abort
    return validated


def _validated_development_model_execution_claims(
    raw: Any,
    *,
    development_sec_execution_claims: Mapping[str, Any],
    development_sec_reader_receipts: Mapping[str, Any],
    development_sec_execution_aborts: Mapping[str, Any],
    development_market_execution_claims: Mapping[str, Any],
    development_market_reader_receipts: Mapping[str, Any],
    development_market_execution_aborts: Mapping[str, Any],
    authenticated_store_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    claims = _mapping(raw, "current-tip development model execution claims")
    validated: dict[str, dict[str, Any]] = {}
    for raw_scope_sha256, raw_claim in claims.items():
        scope_hash = _sha256(
            raw_scope_sha256,
            "development model execution claim map key",
        )
        claim = _mapping(
            raw_claim,
            f"development model execution claim {scope_hash}",
        )
        _expect_keys(
            claim,
            _DEVELOPMENT_MODEL_EXECUTION_CLAIM_KEYS,
            f"development model execution claim {scope_hash}",
        )
        if (
            claim["schema_version"]
            != DEVELOPMENT_MODEL_EXECUTION_CLAIM_SCHEMA_VERSION
            or claim["contract_version"] != CONTRACT_VERSION
            or claim["claim_kind"] != "owned_development_gemma_model_batch"
            or claim["development_root_scope_sha256"] != scope_hash
            or claim["authorized_stage"] != "development"
            or claim["model_component_id"] != STAGE_MODEL_BATCH_COMPONENT_ID
            or claim["carry_in_required"] is not False
            or claim["owned_local_model_execution_required"] is not True
            or claim["caller_supplied_path_permitted"] is not False
            or claim["filing_text_included"] is not False
            or claim["market_access_permitted"] is not False
            or claim["outcome_access_permitted"] is not False
            or claim["future_stage_access_permitted"] is not False
            or claim["paid_api_access_permitted"] is not False
            or claim["external_network_access_permitted"] is not False
            or claim["reveal_request_consumption_permitted"] is not False
            or claim["consumption_ledger_mutation_permitted"] is not False
            or claim["effect_may_be_repeated_after_indeterminate_crash"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution claim semantics changed"
            )
        _self_hash(claim, "claim_sha256", "development model execution claim")
        _strict_int(
            claim["start_consumed_request_count"],
            "development model execution start consumed-request count",
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
            "sec_acquisition_accession_order_sha256",
            "event_plan_sha256",
            "identity_lexicon_sha256",
            "candidate_source_hashes_sha256",
            "execution_source_hashes_sha256",
            "model_digest",
            "runtime_fingerprint_sha256",
            "model_transport_sha256",
            "model_runtime_limits_sha256",
        ):
            _sha256(claim[field], f"development model execution claim {field}")
        for field in ("attempt_id", "model_name"):
            _safe_id(claim[field], f"development model execution claim {field}")
        namespace = claim["output_namespace"]
        if type(namespace) is not str or _OUTPUT_NAMESPACE_RE.fullmatch(namespace) is None:
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution namespace is invalid"
            )
        acquisition_accessions = _validated_model_accession_order(
            claim["sec_acquisition_accession_order"],
            location="development model execution SEC acquisition order",
        )
        if (
            claim["sec_document_count"] != len(acquisition_accessions)
            or claim["sec_acquisition_accession_order_sha256"]
            != canonical_sha256(acquisition_accessions)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution SEC acquisition binding is inconsistent"
            )
        if claim["identity_lexicon_sha256"] != CANONICAL_IDENTITY_LEXICON_SHA256:
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution changed the canonical identity lexicon"
            )
        sources = _validated_model_execution_source_hashes(
            claim["execution_source_hashes"],
            location="development model execution source hashes",
        )
        if (
            claim["execution_source_hashes"] != sources
            or claim["execution_source_hashes_sha256"] != canonical_sha256(sources)
            or claim["execution_source_role_count"] != len(sources)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution source closure is inconsistent"
            )
        limits = _validated_model_runtime_limits(
            claim["model_runtime_limits"],
            stage="development",
            accession_count=len(acquisition_accessions),
            location="development model runtime limits",
        )
        if claim["model_runtime_limits_sha256"] != canonical_sha256(limits):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model runtime-limit hash is inconsistent"
            )
        sec_claim = development_sec_execution_claims.get(scope_hash)
        sec_reader = development_sec_reader_receipts.get(scope_hash)
        if (
            type(sec_claim) is not dict
            or type(sec_reader) is not dict
            or scope_hash in development_sec_execution_aborts
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution claim lacks terminal development SEC ancestry"
            )
        market_bindings = _development_market_model_bindings(
            development_root_scope_sha256=scope_hash,
            development_market_execution_claims=(
                development_market_execution_claims
            ),
            development_market_reader_receipts=(
                development_market_reader_receipts
            ),
            development_market_execution_aborts=(
                development_market_execution_aborts
            ),
            location="Development model execution claim",
        )
        plan = _validated_development_content_root_plan(
            sec_claim["development_content_root_plan"]
        )
        root_scope = plan["root_scope"]
        planned_accessions = [
            document["accession_number"]
            for document in plan["sec_access_plan"]["documents"]
        ]
        planned_accessions = _validated_model_accession_order(
            planned_accessions,
            location="development model planned SEC acquisition order",
        )
        universe = _mapping(
            plan["corpus_universe_manifest"],
            "development model corpus universe",
        )
        event_plan = _validated_model_event_plan(
            claim["event_plan"],
            universe_manifest=universe,
            stage="development",
            sec_acquisition_accession_order=planned_accessions,
        )
        if (
            claim["event_count"] != len(event_plan)
            or claim["event_plan_sha256"] != canonical_sha256(event_plan)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution event-plan binding is inconsistent"
            )
        expected = {
            "development_root_scope_sha256": scope_hash,
            "development_content_root_plan_sha256": plan[
                "development_content_root_plan_sha256"
            ],
            "attempt_id": sec_claim["attempt_id"],
            "candidate_sha256": sec_claim["candidate_sha256"],
            "candidate_design_sha256": sec_claim["candidate_design_sha256"],
            "registry_entry_sha256": sec_claim["registry_entry_sha256"],
            "registry_sha256": sec_claim["registry_sha256"],
            "registry_tip_sha256": sec_claim["registry_tip_sha256"],
            "corpus_universe_sha256": sec_claim["corpus_universe_sha256"],
            "corpus_universe_semantic_sha256": sec_claim[
                "corpus_universe_semantic_sha256"
            ],
            "output_namespace": sec_claim["output_namespace"],
            "development_sec_execution_claim_sha256": sec_claim[
                "claim_sha256"
            ],
            "development_sec_reader_receipt_sha256": sec_reader[
                "receipt_sha256"
            ],
            **market_bindings,
            "start_consumed_request_count": sec_claim[
                "start_consumed_request_count"
            ],
            "sec_document_count": len(planned_accessions),
            "sec_acquisition_accession_order": planned_accessions,
            "sec_acquisition_accession_order_sha256": canonical_sha256(
                planned_accessions
            ),
            "event_count": len(event_plan),
            "event_plan": event_plan,
            "event_plan_sha256": canonical_sha256(event_plan),
            "identity_lexicon_sha256": CANONICAL_IDENTITY_LEXICON_SHA256,
        }
        if any(claim[field] != value for field, value in expected.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution claim crossed its SEC root ancestry"
            )
        sec_sources = _mapping(
            sec_claim["execution_source_hashes"],
            "development SEC source hashes for model execution",
        )
        if any(
            sec_sources.get(role) != value
            for role, value in sources.items()
            if role in sec_sources
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution bytes differ from its SEC root source closure"
            )
        for field in ("model_digest", "runtime_fingerprint_sha256"):
            _sha256(claim[field], f"development model candidate {field}")
        if authenticated_store_snapshot is not None:
            _state, _registry_entry, candidate = _development_registered_candidate(
                authenticated_store_snapshot,
                plan=plan,
                require_latest=False,
            )
            candidate_sources = _mapping(
                candidate["bindings"]["source_hashes"],
                "development model candidate source hashes",
            )
            model = _mapping(candidate["model"], "development model candidate model")
            candidate_expected = {
                "identity_lexicon_sha256": CANONICAL_IDENTITY_LEXICON_SHA256,
                "candidate_source_hashes_sha256": canonical_sha256(
                    candidate_sources
                ),
                "model_name": model["name"],
                "model_digest": model["digest"],
                "runtime_fingerprint_sha256": model[
                    "runtime_fingerprint_sha256"
                ],
                "model_transport_sha256": canonical_sha256(model["transport"]),
            }
            if (
                any(
                    claim[field] != value
                    for field, value in candidate_expected.items()
                )
                or any(
                    candidate_sources.get(role) != value
                    for role, value in sources.items()
                )
                or candidate["bindings"]["identity_lexicon_sha256"]
                != CANONICAL_IDENTITY_LEXICON_SHA256
            ):
                raise SecFilingGemmaStageAuthorizationError(
                    "Development model execution claim crossed its registered candidate"
                )
        validated[scope_hash] = claim
    return validated


def _validated_development_model_reader_receipts(
    raw: Any,
    *,
    claims: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    receipts = _mapping(raw, "current-tip development model reader receipts")
    validated: dict[str, dict[str, Any]] = {}
    for raw_scope_sha256, raw_receipt in receipts.items():
        scope_hash = _sha256(raw_scope_sha256, "development model reader map key")
        receipt = _mapping(
            raw_receipt,
            f"development model reader receipt {scope_hash}",
        )
        _expect_keys(
            receipt,
            _DEVELOPMENT_MODEL_READER_RECEIPT_KEYS,
            f"development model reader receipt {scope_hash}",
        )
        if (
            receipt["schema_version"]
            != DEVELOPMENT_MODEL_READER_RECEIPT_SCHEMA_VERSION
            or receipt["contract_version"] != CONTRACT_VERSION
            or receipt["receipt_kind"]
            != "store_rehashed_owned_development_gemma_model_batch"
            or receipt["development_root_scope_sha256"] != scope_hash
            or receipt["authorized_stage"] != "development"
            or receipt["model_component_id"] != STAGE_MODEL_BATCH_COMPONENT_ID
            or receipt["fresh_model_provenance_claimed"] is not False
            or receipt["reader_output_recomputed_by_store"] is not True
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model reader receipt semantics changed"
            )
        _self_hash(receipt, "receipt_sha256", "development model reader receipt")
        index = _validated_sec_byte_index(receipt["byte_index"])
        if (
            receipt["byte_index"] != index
            or receipt["byte_index_sha256"] != canonical_sha256(index)
            or receipt["byte_count_total"]
            != sum(item["byte_count"] for item in index)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model reader byte index is inconsistent"
            )
        for field in (
            "development_root_scope_sha256",
            "claim_sha256",
            "development_content_root_plan_sha256",
            "candidate_sha256",
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
            "sec_acquisition_accession_order_sha256",
            "event_plan_sha256",
            "identity_lexicon_sha256",
            "execution_source_hashes_sha256",
            "model_digest",
            "runtime_fingerprint_sha256",
            "model_transport_sha256",
            "model_runtime_limits_sha256",
            "byte_index_sha256",
            "complete_marker_sha256",
        ):
            _sha256(receipt[field], f"development model reader receipt {field}")
        claim = claims.get(scope_hash)
        if type(claim) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "Development model reader receipt lacks its execution claim"
            )
        expected_fields = (
            "development_content_root_plan_sha256",
            "authorized_stage",
            "candidate_sha256",
            "output_namespace",
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
            "sec_document_count",
            "sec_acquisition_accession_order_sha256",
            "event_count",
            "event_plan_sha256",
            "identity_lexicon_sha256",
            "execution_source_hashes_sha256",
            "execution_source_role_count",
            "model_name",
            "model_digest",
            "runtime_fingerprint_sha256",
            "model_transport_sha256",
            "model_runtime_limits_sha256",
            "model_component_id",
        )
        if (
            receipt["claim_sha256"] != claim["claim_sha256"]
            or any(receipt[field] != claim[field] for field in expected_fields)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model reader receipt crossed its execution claim"
            )
        validated[scope_hash] = receipt
    return validated


def _validated_development_model_execution_aborts(
    raw: Any,
    *,
    claims: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    aborts = _mapping(raw, "current-tip development model execution aborts")
    validated: dict[str, dict[str, Any]] = {}
    for raw_scope_sha256, raw_abort in aborts.items():
        scope_hash = _sha256(raw_scope_sha256, "development model abort map key")
        abort = _mapping(
            raw_abort,
            f"development model execution abort {scope_hash}",
        )
        _expect_keys(
            abort,
            _DEVELOPMENT_MODEL_EXECUTION_ABORT_KEYS,
            f"development model execution abort {scope_hash}",
        )
        if (
            abort["schema_version"]
            != DEVELOPMENT_MODEL_EXECUTION_ABORT_SCHEMA_VERSION
            or abort["contract_version"] != CONTRACT_VERSION
            or abort["abort_kind"]
            != "indeterminate_owned_development_gemma_model_batch"
            or abort["reason"]
            not in {
                "claim_recovered_without_terminal_receipt",
                "external_effect_failed_or_completion_unknown",
                "durable_output_verification_failed",
            }
            or abort["external_effect_retry_permitted"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution abort semantics changed"
            )
        _self_hash(abort, "abort_sha256", "development model execution abort")
        claim = claims.get(scope_hash)
        if type(claim) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution abort lacks its claim"
            )
        expected = {
            "development_root_scope_sha256": scope_hash,
            "claim_sha256": claim["claim_sha256"],
            "development_content_root_plan_sha256": claim[
                "development_content_root_plan_sha256"
            ],
            "authorized_stage": "development",
            "candidate_sha256": claim["candidate_sha256"],
            "output_namespace": claim["output_namespace"],
            "model_component_id": claim["model_component_id"],
        }
        if any(abort[field] != value for field, value in expected.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution abort crossed its claim"
            )
        validated[scope_hash] = abort
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
    anchor["development_market_execution_claims"] = (
        _validated_development_market_execution_claims(
            anchor["development_market_execution_claims"],
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
    anchor["development_market_reader_receipts"] = (
        _validated_development_market_reader_receipts(
            anchor["development_market_reader_receipts"],
            claims=anchor["development_market_execution_claims"],
        )
    )
    anchor["development_market_execution_aborts"] = (
        _validated_development_market_execution_aborts(
            anchor["development_market_execution_aborts"],
            claims=anchor["development_market_execution_claims"],
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
    anchor["stage_model_execution_claims"] = (
        _validated_stage_model_execution_claims(
            anchor["stage_model_execution_claims"],
            authorization_bundles=anchor["authorization_bundles"],
            sec_execution_claims=anchor["stage_sec_execution_claims"],
            sec_reader_receipts=anchor["stage_sec_reader_receipts"],
            sec_execution_aborts=anchor["stage_sec_execution_aborts"],
            stage_carry_in_reader_receipts=anchor[
                "stage_carry_in_reader_receipts"
            ],
            development_root_carry_in_reader_receipts=anchor[
                "development_root_carry_in_reader_receipts"
            ],
            development_sec_execution_claims=anchor[
                "development_sec_execution_claims"
            ],
            development_sec_reader_receipts=anchor[
                "development_sec_reader_receipts"
            ],
            development_sec_execution_aborts=anchor[
                "development_sec_execution_aborts"
            ],
            development_market_execution_claims=anchor[
                "development_market_execution_claims"
            ],
            development_market_reader_receipts=anchor[
                "development_market_reader_receipts"
            ],
            development_market_execution_aborts=anchor[
                "development_market_execution_aborts"
            ],
        )
    )
    anchor["stage_model_reader_receipts"] = (
        _validated_stage_model_reader_receipts(
            anchor["stage_model_reader_receipts"],
            claims=anchor["stage_model_execution_claims"],
        )
    )
    anchor["stage_model_execution_aborts"] = (
        _validated_stage_model_execution_aborts(
            anchor["stage_model_execution_aborts"],
            claims=anchor["stage_model_execution_claims"],
        )
    )
    anchor["development_model_execution_claims"] = (
        _validated_development_model_execution_claims(
            anchor["development_model_execution_claims"],
            development_sec_execution_claims=anchor[
                "development_sec_execution_claims"
            ],
            development_sec_reader_receipts=anchor[
                "development_sec_reader_receipts"
            ],
            development_sec_execution_aborts=anchor[
                "development_sec_execution_aborts"
            ],
            development_market_execution_claims=anchor[
                "development_market_execution_claims"
            ],
            development_market_reader_receipts=anchor[
                "development_market_reader_receipts"
            ],
            development_market_execution_aborts=anchor[
                "development_market_execution_aborts"
            ],
        )
    )
    anchor["development_model_reader_receipts"] = (
        _validated_development_model_reader_receipts(
            anchor["development_model_reader_receipts"],
            claims=anchor["development_model_execution_claims"],
        )
    )
    anchor["development_model_execution_aborts"] = (
        _validated_development_model_execution_aborts(
            anchor["development_model_execution_aborts"],
            claims=anchor["development_model_execution_claims"],
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
    if set(anchor["development_market_reader_receipts"]) & set(
        anchor["development_market_execution_aborts"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development market execution cannot be both completed and aborted"
        )
    if set(anchor["stage_model_reader_receipts"]) & set(
        anchor["stage_model_execution_aborts"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model execution cannot be both completed and aborted"
        )
    if set(anchor["development_model_reader_receipts"]) & set(
        anchor["development_model_execution_aborts"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development model execution cannot be both completed and aborted"
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
    active_stage_model_claims = set(
        anchor["stage_model_execution_claims"]
    ) - set(anchor["stage_model_reader_receipts"]) - set(
        anchor["stage_model_execution_aborts"]
    )
    active_development_model_claims = set(
        anchor["development_model_execution_claims"]
    ) - set(anchor["development_model_reader_receipts"]) - set(
        anchor["development_model_execution_aborts"]
    )
    if (
        len(active_stage_model_claims)
        + len(active_development_model_claims)
        > 1
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "At most one model execution claim may be globally active"
        )
    if (
        active_stage_model_claims or active_development_model_claims
    ) and (active_stage_claims or active_development_claims):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC and model effects cannot be globally active together"
        )
    active_development_market_claims = set(
        anchor["development_market_execution_claims"]
    ) - set(anchor["development_market_reader_receipts"]) - set(
        anchor["development_market_execution_aborts"]
    )
    if len(active_development_market_claims) > 1:
        raise SecFilingGemmaStageAuthorizationError(
            "At most one market execution claim may be globally active"
        )
    if active_development_market_claims and (
        active_stage_claims
        or active_development_claims
        or active_stage_model_claims
        or active_development_model_claims
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Market, SEC, and model effects cannot be globally active together"
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
    development_market_execution_claims: Mapping[str, Any] | None = None,
    development_market_reader_receipts: Mapping[str, Any] | None = None,
    development_market_execution_aborts: Mapping[str, Any] | None = None,
    development_root_carry_in_reader_receipts: Mapping[str, Any] | None = None,
    stage_model_execution_claims: Mapping[str, Any] | None = None,
    stage_model_reader_receipts: Mapping[str, Any] | None = None,
    stage_model_execution_aborts: Mapping[str, Any] | None = None,
    development_model_execution_claims: Mapping[str, Any] | None = None,
    development_model_reader_receipts: Mapping[str, Any] | None = None,
    development_model_execution_aborts: Mapping[str, Any] | None = None,
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
    development_market_claims = _validated_development_market_execution_claims(
        (
            {}
            if development_market_execution_claims is None
            else development_market_execution_claims
        ),
        development_sec_execution_claims=development_claims,
        development_sec_reader_receipts=development_receipts,
        development_sec_execution_aborts=development_aborts,
        authenticated_store_snapshot=state,
    )
    development_market_receipts = _validated_development_market_reader_receipts(
        (
            {}
            if development_market_reader_receipts is None
            else development_market_reader_receipts
        ),
        claims=development_market_claims,
    )
    development_market_aborts = _validated_development_market_execution_aborts(
        (
            {}
            if development_market_execution_aborts is None
            else development_market_execution_aborts
        ),
        claims=development_market_claims,
    )
    if set(development_market_receipts) & set(development_market_aborts):
        raise SecFilingGemmaStageAuthorizationError(
            "Development market execution cannot be both completed and aborted"
        )
    active_market_count = len(
        set(development_market_claims)
        - set(development_market_receipts)
        - set(development_market_aborts)
    )
    if active_market_count > 1:
        raise SecFilingGemmaStageAuthorizationError(
            "At most one market execution claim may be globally active"
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
    model_claims = _validated_stage_model_execution_claims(
        {} if stage_model_execution_claims is None else stage_model_execution_claims,
        authorization_bundles=bundles,
        sec_execution_claims=sec_claims,
        sec_reader_receipts=sec_receipts,
        sec_execution_aborts=sec_aborts,
        stage_carry_in_reader_receipts=carry_in_receipts,
        development_root_carry_in_reader_receipts=(
            development_root_carry_in_receipts
        ),
        development_sec_execution_claims=development_claims,
        development_sec_reader_receipts=development_receipts,
        development_sec_execution_aborts=development_aborts,
        development_market_execution_claims=development_market_claims,
        development_market_reader_receipts=development_market_receipts,
        development_market_execution_aborts=development_market_aborts,
    )
    model_receipts = _validated_stage_model_reader_receipts(
        {} if stage_model_reader_receipts is None else stage_model_reader_receipts,
        claims=model_claims,
    )
    model_aborts = _validated_stage_model_execution_aborts(
        {} if stage_model_execution_aborts is None else stage_model_execution_aborts,
        claims=model_claims,
    )
    if set(model_receipts) & set(model_aborts):
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model execution cannot be both completed and aborted"
        )
    development_model_claims = _validated_development_model_execution_claims(
        (
            {}
            if development_model_execution_claims is None
            else development_model_execution_claims
        ),
        development_sec_execution_claims=development_claims,
        development_sec_reader_receipts=development_receipts,
        development_sec_execution_aborts=development_aborts,
        development_market_execution_claims=development_market_claims,
        development_market_reader_receipts=development_market_receipts,
        development_market_execution_aborts=development_market_aborts,
        authenticated_store_snapshot=state,
    )
    development_model_receipts = _validated_development_model_reader_receipts(
        (
            {}
            if development_model_reader_receipts is None
            else development_model_reader_receipts
        ),
        claims=development_model_claims,
    )
    development_model_aborts = _validated_development_model_execution_aborts(
        (
            {}
            if development_model_execution_aborts is None
            else development_model_execution_aborts
        ),
        claims=development_model_claims,
    )
    if set(development_model_receipts) & set(development_model_aborts):
        raise SecFilingGemmaStageAuthorizationError(
            "Development model execution cannot be both completed and aborted"
        )
    active_model_count = len(
        set(model_claims) - set(model_receipts) - set(model_aborts)
    ) + len(
        set(development_model_claims)
        - set(development_model_receipts)
        - set(development_model_aborts)
    )
    if active_model_count > 1:
        raise SecFilingGemmaStageAuthorizationError(
            "At most one model execution claim may be globally active"
        )
    if active_model_count and (
        active_stage_claim_count + active_development_claim_count
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC and model effects cannot be globally active together"
        )
    if active_market_count and (
        active_model_count + active_stage_claim_count + active_development_claim_count
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Market, SEC, and model effects cannot be globally active together"
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
        "development_market_execution_claims": development_market_claims,
        "development_market_reader_receipts": development_market_receipts,
        "development_market_execution_aborts": development_market_aborts,
        "development_root_carry_in_reader_receipts": (
            development_root_carry_in_receipts
        ),
        "stage_model_execution_claims": model_claims,
        "stage_model_reader_receipts": model_receipts,
        "stage_model_execution_aborts": model_aborts,
        "development_model_execution_claims": development_model_claims,
        "development_model_reader_receipts": development_model_receipts,
        "development_model_execution_aborts": development_model_aborts,
    }
    return {**body, "tip_anchor_sha256": canonical_sha256(body)}


def validate_reveal_store_current_tip_anchor_transition(
    prior_tip_anchor: Mapping[str, Any],
    next_tip_anchor: Mapping[str, Any],
    *,
    authenticated_store_snapshot: Mapping[str, Any] | None = None,
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

    prior_market_claims = prior["development_market_execution_claims"]
    next_market_claims = next_anchor["development_market_execution_claims"]
    prior_market_receipts = prior["development_market_reader_receipts"]
    next_market_receipts = next_anchor["development_market_reader_receipts"]
    prior_market_aborts = prior["development_market_execution_aborts"]
    next_market_aborts = next_anchor["development_market_execution_aborts"]
    for prior_map, next_map, label in (
        (
            prior_market_claims,
            next_market_claims,
            "development market execution claim",
        ),
        (
            prior_market_receipts,
            next_market_receipts,
            "development market reader receipt",
        ),
        (
            prior_market_aborts,
            next_market_aborts,
            "development market execution abort",
        ),
    ):
        if any(next_map.get(key) != value for key, value in prior_map.items()):
            raise SecFilingGemmaStageAuthorizationError(
                f"Current-tip transition removed or changed a persisted {label}"
            )
    market_claim_delta = len(next_market_claims) - len(prior_market_claims)
    market_reader_delta = len(next_market_receipts) - len(prior_market_receipts)
    market_abort_delta = len(next_market_aborts) - len(prior_market_aborts)
    if any(
        delta not in {0, 1}
        for delta in (
            market_claim_delta,
            market_reader_delta,
            market_abort_delta,
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition may append at most one development market execution artifact"
        )
    market_delta_count = (
        market_claim_delta + market_reader_delta + market_abort_delta
    )
    if market_delta_count > 1:
        raise SecFilingGemmaStageAuthorizationError(
            "Development market claim, reader receipt, and abort require separate transitions"
        )
    non_market_map_names = (
        "trusted_stage_content_pins",
        "authorization_bundles",
        "consumed_stage_output_receipts",
        "stage_carry_in_reader_receipts",
        "stage_sec_execution_claims",
        "stage_sec_reader_receipts",
        "stage_sec_execution_aborts",
        "development_sec_execution_claims",
        "development_sec_reader_receipts",
        "development_sec_execution_aborts",
        "development_root_carry_in_reader_receipts",
        "stage_model_execution_claims",
        "stage_model_reader_receipts",
        "stage_model_execution_aborts",
        "development_model_execution_claims",
        "development_model_reader_receipts",
        "development_model_execution_aborts",
    )
    non_market_maps_unchanged = all(
        next_anchor[name] == prior[name] for name in non_market_map_names
    )
    prior_active_market = set(prior_market_claims) - set(
        prior_market_receipts
    ) - set(prior_market_aborts)
    if prior_active_market:
        active_scope = next(iter(prior_active_market))
        terminal_scope: str | None = None
        if market_reader_delta:
            terminal_scope = next(
                iter(set(next_market_receipts) - set(prior_market_receipts))
            )
        elif market_abort_delta:
            terminal_scope = next(
                iter(set(next_market_aborts) - set(prior_market_aborts))
            )
        if (
            market_claim_delta
            or market_delta_count != 1
            or terminal_scope != active_scope
            or not non_market_maps_unchanged
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Active development market execution claim blocks every transition except its exact terminal receipt"
            )
    if market_delta_count:
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
        if not non_market_maps_unchanged:
            raise SecFilingGemmaStageAuthorizationError(
                "Development market artifact append must be a dedicated tip-only transition"
            )
        if any(next_anchor[field] != prior[field] for field in immutable_state_fields):
            raise SecFilingGemmaStageAuthorizationError(
                "Development market artifact append changed authenticated store state"
            )
    if market_claim_delta:
        scope_hash = next(
            iter(set(next_market_claims) - set(prior_market_claims))
        )
        claim = next_market_claims[scope_hash]
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
        }
        if (
            scope_hash not in prior["development_sec_execution_claims"]
            or scope_hash not in prior["development_sec_reader_receipts"]
            or scope_hash in prior["development_sec_execution_aborts"]
            or any(
                claim[field] != expected
                for field, expected in current_bindings.items()
            )
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development market execution claim does not bind terminal SEC root ancestry at the exact current tip"
            )
        if authenticated_store_snapshot is None:
            raise SecFilingGemmaStageAuthorizationError(
                "Development market claim transition requires the authenticated prior store snapshot"
            )
        validate_reveal_store_current_tip_anchor(
            authenticated_store_snapshot,
            prior,
        )
        expected_claim = build_development_market_execution_claim(
            authenticated_store_snapshot,
            development_root_scope_sha256=scope_hash,
            independent_current_tip_anchor=prior,
            market_acquisition_plan=claim["market_acquisition_plan"],
            execution_source_hashes=claim["execution_source_hashes"],
        )
        if claim != expected_claim:
            raise SecFilingGemmaStageAuthorizationError(
                "Development market execution claim differs from its exact candidate-bound owned plan"
            )
    if market_reader_delta:
        scope_hash = next(
            iter(set(next_market_receipts) - set(prior_market_receipts))
        )
        receipt = next_market_receipts[scope_hash]
        claim = prior_market_claims.get(scope_hash)
        if type(claim) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "Development market reader receipt lacks its prior execution claim"
            )
        expected_receipt = build_development_market_reader_receipt(
            claim,
            acquisition_receipt_sha256=receipt["acquisition_receipt_sha256"],
            acquisition_bundle_sha256=receipt["acquisition_bundle_sha256"],
            acquisition_validation_sha256=receipt[
                "acquisition_validation_sha256"
            ],
            source_manifest_sha256=receipt["source_manifest_sha256"],
            market_stage_manifest_sha256=receipt[
                "market_stage_manifest_sha256"
            ],
            source_reconciliation_sha256=receipt[
                "source_reconciliation_sha256"
            ],
            raw_response_sha256s=receipt["raw_response_sha256s"],
            artifact_sha256s=receipt["artifact_sha256s"],
            window_sha256s=receipt["window_sha256s"],
            byte_index=receipt["byte_index"],
            complete_marker_sha256=receipt["complete_marker_sha256"],
            owned_transport_attested_by_store=receipt[
                "owned_transport_attested_by_store"
            ],
        )
        if receipt != expected_receipt:
            raise SecFilingGemmaStageAuthorizationError(
                "Development market reader receipt differs from its exact durable artifact closure"
            )
    if market_abort_delta:
        scope_hash = next(iter(set(next_market_aborts) - set(prior_market_aborts)))
        abort = next_market_aborts[scope_hash]
        claim = prior_market_claims.get(scope_hash)
        if type(claim) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "Development market abort lacks its prior execution claim"
            )
        expected_abort = build_development_market_execution_abort(
            claim,
            reason=abort["reason"],
        )
        if abort != expected_abort:
            raise SecFilingGemmaStageAuthorizationError(
                "Development market abort differs from its exact no-retry terminal"
            )
    if not market_delta_count and any(
        set(next_anchor[name]) != set(prior[name])
        for name in (
            "development_market_execution_claims",
            "development_market_reader_receipts",
            "development_market_execution_aborts",
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Non-market transition changed development market execution membership"
        )

    prior_model_claims = prior["stage_model_execution_claims"]
    next_model_claims = next_anchor["stage_model_execution_claims"]
    prior_model_receipts = prior["stage_model_reader_receipts"]
    next_model_receipts = next_anchor["stage_model_reader_receipts"]
    prior_model_aborts = prior["stage_model_execution_aborts"]
    next_model_aborts = next_anchor["stage_model_execution_aborts"]
    prior_development_model_claims = prior[
        "development_model_execution_claims"
    ]
    next_development_model_claims = next_anchor[
        "development_model_execution_claims"
    ]
    prior_development_model_receipts = prior[
        "development_model_reader_receipts"
    ]
    next_development_model_receipts = next_anchor[
        "development_model_reader_receipts"
    ]
    prior_development_model_aborts = prior[
        "development_model_execution_aborts"
    ]
    next_development_model_aborts = next_anchor[
        "development_model_execution_aborts"
    ]
    for prior_map, next_map, label in (
        (prior_model_claims, next_model_claims, "stage model execution claim"),
        (prior_model_receipts, next_model_receipts, "stage model reader receipt"),
        (prior_model_aborts, next_model_aborts, "stage model execution abort"),
        (
            prior_development_model_claims,
            next_development_model_claims,
            "development model execution claim",
        ),
        (
            prior_development_model_receipts,
            next_development_model_receipts,
            "development model reader receipt",
        ),
        (
            prior_development_model_aborts,
            next_development_model_aborts,
            "development model execution abort",
        ),
    ):
        if any(next_map.get(key) != value for key, value in prior_map.items()):
            raise SecFilingGemmaStageAuthorizationError(
                f"Current-tip transition removed or changed a persisted {label}"
            )
    model_claim_delta = len(next_model_claims) - len(prior_model_claims)
    model_reader_delta = len(next_model_receipts) - len(prior_model_receipts)
    model_abort_delta = len(next_model_aborts) - len(prior_model_aborts)
    development_model_claim_delta = len(
        next_development_model_claims
    ) - len(prior_development_model_claims)
    development_model_reader_delta = len(
        next_development_model_receipts
    ) - len(prior_development_model_receipts)
    development_model_abort_delta = len(
        next_development_model_aborts
    ) - len(prior_development_model_aborts)
    model_deltas = (
        model_claim_delta,
        model_reader_delta,
        model_abort_delta,
        development_model_claim_delta,
        development_model_reader_delta,
        development_model_abort_delta,
    )
    if any(delta not in {0, 1} for delta in model_deltas):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition may append at most one model execution artifact"
        )
    model_delta_count = sum(model_deltas)
    if model_delta_count > 1:
        raise SecFilingGemmaStageAuthorizationError(
            "Model claims, reader receipts, and aborts require separate global transitions"
        )
    prior_active_models = set(prior_model_claims) - set(
        prior_model_receipts
    ) - set(prior_model_aborts)
    prior_active_development_models = set(
        prior_development_model_claims
    ) - set(prior_development_model_receipts) - set(
        prior_development_model_aborts
    )
    non_model_execution_maps = (
        "stage_sec_execution_claims",
        "stage_sec_reader_receipts",
        "stage_sec_execution_aborts",
        "development_sec_execution_claims",
        "development_sec_reader_receipts",
        "development_sec_execution_aborts",
    )
    non_model_execution_maps_unchanged = all(
        next_anchor[name] == prior[name] for name in non_model_execution_maps
    )
    if prior_active_models or prior_active_development_models:
        if prior_active_models:
            active_key = next(iter(prior_active_models))
            terminal_key = None
            if model_reader_delta:
                terminal_key = next(
                    iter(set(next_model_receipts) - set(prior_model_receipts))
                )
            elif model_abort_delta:
                terminal_key = next(
                    iter(set(next_model_aborts) - set(prior_model_aborts))
                )
            correct_lifecycle_terminal = (
                terminal_key == active_key
                and not development_model_reader_delta
                and not development_model_abort_delta
            )
        else:
            active_key = next(iter(prior_active_development_models))
            terminal_key = None
            if development_model_reader_delta:
                terminal_key = next(
                    iter(
                        set(next_development_model_receipts)
                        - set(prior_development_model_receipts)
                    )
                )
            elif development_model_abort_delta:
                terminal_key = next(
                    iter(
                        set(next_development_model_aborts)
                        - set(prior_development_model_aborts)
                    )
                )
            correct_lifecycle_terminal = (
                terminal_key == active_key
                and not model_reader_delta
                and not model_abort_delta
            )
        if (
            model_claim_delta
            or development_model_claim_delta
            or model_delta_count != 1
            or not correct_lifecycle_terminal
            or consumption_delta
            or bundle_delta
            or pin_delta
            or output_delta
            or carry_in_delta
            or development_root_carry_in_delta
            or not non_model_execution_maps_unchanged
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Active model execution claim blocks every transition except its exact terminal receipt"
            )
    if model_delta_count:
        if (
            consumption_delta
            or bundle_delta
            or pin_delta
            or output_delta
            or carry_in_delta
            or development_root_carry_in_delta
            or not non_model_execution_maps_unchanged
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Model execution artifact append must be a dedicated tip-only transition"
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
                "Model execution artifact append changed authenticated store state"
            )
    if model_claim_delta:
        request_hash = next(
            iter(set(next_model_claims) - set(prior_model_claims))
        )
        claim = next_model_claims[request_hash]
        if (
            request_hash not in prior["authorization_bundles"]
            or request_hash not in prior["stage_sec_execution_claims"]
            or request_hash not in prior["stage_sec_reader_receipts"]
            or request_hash in prior["stage_sec_execution_aborts"]
            or request_hash in prior["consumed_stage_output_receipts"]
            or claim["start_current_tip_anchor_sha256"]
            != prior["tip_anchor_sha256"]
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution claim does not bind terminal pre-output ancestry at the exact current tip"
            )
        expected_claim = build_stage_model_execution_claim(
            prior["authorization_bundles"][request_hash],
            independent_current_tip_anchor=prior,
            execution_source_hashes=claim["execution_source_hashes"],
        )
        if claim != expected_claim:
            raise SecFilingGemmaStageAuthorizationError(
                "Stage model execution claim differs from its exact owned plan"
            )
    if development_model_claim_delta:
        scope_hash = next(
            iter(
                set(next_development_model_claims)
                - set(prior_development_model_claims)
            )
        )
        claim = next_development_model_claims[scope_hash]
        if (
            scope_hash not in prior["development_sec_execution_claims"]
            or scope_hash not in prior["development_sec_reader_receipts"]
            or scope_hash in prior["development_sec_execution_aborts"]
            or claim["start_current_tip_anchor_sha256"]
            != prior["tip_anchor_sha256"]
            or claim["start_consumed_request_count"]
            != prior["consumed_request_count"]
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution claim does not bind terminal SEC root ancestry at the exact current tip"
            )
        if authenticated_store_snapshot is None:
            raise SecFilingGemmaStageAuthorizationError(
                "Development model claim transition requires the authenticated prior store snapshot"
            )
        validate_reveal_store_current_tip_anchor(
            authenticated_store_snapshot,
            prior,
        )
        expected_claim = build_development_model_execution_claim(
            authenticated_store_snapshot,
            development_root_scope_sha256=scope_hash,
            independent_current_tip_anchor=prior,
            execution_source_hashes=claim["execution_source_hashes"],
        )
        if claim != expected_claim:
            raise SecFilingGemmaStageAuthorizationError(
                "Development model execution claim differs from its exact candidate-bound owned plan"
            )
    if not model_delta_count:
        model_map_names = (
            "stage_model_execution_claims",
            "stage_model_reader_receipts",
            "stage_model_execution_aborts",
            "development_model_execution_claims",
            "development_model_reader_receipts",
            "development_model_execution_aborts",
        )
        if any(
            set(next_anchor[name]) != set(prior[name])
            for name in model_map_names
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Non-model transition changed model execution membership"
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
        development_market_execution_claims=observed[
            "development_market_execution_claims"
        ],
        development_market_reader_receipts=observed[
            "development_market_reader_receipts"
        ],
        development_market_execution_aborts=observed[
            "development_market_execution_aborts"
        ],
        development_root_carry_in_reader_receipts=observed[
            "development_root_carry_in_reader_receipts"
        ],
        stage_model_execution_claims=observed[
            "stage_model_execution_claims"
        ],
        stage_model_reader_receipts=observed[
            "stage_model_reader_receipts"
        ],
        stage_model_execution_aborts=observed[
            "stage_model_execution_aborts"
        ],
        development_model_execution_claims=observed[
            "development_model_execution_claims"
        ],
        development_model_reader_receipts=observed[
            "development_model_reader_receipts"
        ],
        development_model_execution_aborts=observed[
            "development_model_execution_aborts"
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


def build_development_market_execution_claim(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    development_root_scope_sha256: str,
    independent_current_tip_anchor: Mapping[str, Any],
    market_acquisition_plan: Mapping[str, Any],
    execution_source_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    """Claim the six-call development market effect after its terminal SEC root."""

    state, ledger = _validated_store_snapshot(authenticated_store_snapshot)
    current_tip = validate_reveal_store_current_tip_anchor(
        state,
        independent_current_tip_anchor,
    )
    scope_hash = _sha256(
        development_root_scope_sha256,
        "development market root scope hash",
    )
    sec_claim = current_tip["development_sec_execution_claims"].get(scope_hash)
    sec_reader = current_tip["development_sec_reader_receipts"].get(scope_hash)
    if (
        type(sec_claim) is not dict
        or type(sec_reader) is not dict
        or scope_hash in current_tip["development_sec_execution_aborts"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development market execution requires one terminal development SEC reader"
        )
    if scope_hash in current_tip["development_market_execution_claims"]:
        raise SecFilingGemmaStageAuthorizationError(
            "Development market execution is already claimed"
        )
    if current_tip["consumed_request_count"] != sec_claim[
        "start_consumed_request_count"
    ]:
        raise SecFilingGemmaStageAuthorizationError(
            "Development market execution must precede reveal-request consumption"
        )

    active_sec = set(current_tip["stage_sec_execution_claims"]) - set(
        current_tip["stage_sec_reader_receipts"]
    ) - set(current_tip["stage_sec_execution_aborts"])
    active_development_sec = set(
        current_tip["development_sec_execution_claims"]
    ) - set(current_tip["development_sec_reader_receipts"]) - set(
        current_tip["development_sec_execution_aborts"]
    )
    active_model = set(current_tip["stage_model_execution_claims"]) - set(
        current_tip["stage_model_reader_receipts"]
    ) - set(current_tip["stage_model_execution_aborts"])
    active_development_model = set(
        current_tip["development_model_execution_claims"]
    ) - set(current_tip["development_model_reader_receipts"]) - set(
        current_tip["development_model_execution_aborts"]
    )
    active_market = set(
        current_tip["development_market_execution_claims"]
    ) - set(current_tip["development_market_reader_receipts"]) - set(
        current_tip["development_market_execution_aborts"]
    )
    if (
        active_sec
        or active_development_sec
        or active_model
        or active_development_model
        or active_market
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Another owned external effect is already active"
        )

    plan = _validated_development_market_acquisition_plan(
        market_acquisition_plan,
        execution_source_hashes=execution_source_hashes,
    )
    content_root_plan = _validated_development_content_root_plan(
        sec_claim["development_content_root_plan"]
    )
    _state, registry_entry, candidate = _development_registered_candidate(
        state,
        plan=content_root_plan,
        require_latest=False,
    )
    raw_sources = _mapping(
        execution_source_hashes,
        "owned development market execution source hashes",
    )
    if set(raw_sources) != set(MARKET_EXECUTION_SOURCE_ROLES):
        raise SecFilingGemmaStageAuthorizationError(
            "Owned development market execution source closure is incomplete"
        )
    sources = {
        role: _sha256(
            raw_sources[role],
            f"owned development market execution source hash {role}",
        )
        for role in MARKET_EXECUTION_SOURCE_ROLES
    }
    candidate_sources = _mapping(
        candidate["bindings"]["source_hashes"],
        "development market candidate source hashes",
    )
    if any(candidate_sources.get(role) != value for role, value in sources.items()):
        raise SecFilingGemmaStageAuthorizationError(
            "Owned development market execution bytes differ from the registered candidate"
        )
    if any(
        sources[role] != plan["source_code_sha256s"][path]
        for path, role in _MARKET_ACQUISITION_SOURCE_PATH_ROLES
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Owned development market plan differs from its candidate-bound source closure"
        )

    body = {
        "schema_version": DEVELOPMENT_MARKET_EXECUTION_CLAIM_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "claim_kind": "owned_development_market_evidence_batch",
        "development_root_scope_sha256": scope_hash,
        "development_content_root_plan_sha256": sec_claim[
            "development_content_root_plan_sha256"
        ],
        "attempt_id": sec_claim["attempt_id"],
        "candidate_sha256": sec_claim["candidate_sha256"],
        "candidate_design_sha256": sec_claim["candidate_design_sha256"],
        "registry_entry_sha256": registry_entry["entry_sha256"],
        "registry_sha256": sec_claim["registry_sha256"],
        "registry_tip_sha256": sec_claim["registry_tip_sha256"],
        "registered_entry_count": sec_claim["registered_entry_count"],
        "corpus_universe_sha256": sec_claim["corpus_universe_sha256"],
        "corpus_universe_semantic_sha256": sec_claim[
            "corpus_universe_semantic_sha256"
        ],
        "authorized_stage": "development",
        "output_namespace": sec_claim["output_namespace"],
        "start_current_tip_anchor_sha256": current_tip["tip_anchor_sha256"],
        "start_state_sha256": state["state_sha256"],
        "start_consumption_ledger_sha256": ledger["ledger_sha256"],
        "start_consumption_ledger_tip_sha256": ledger["chain"]["tip_sha256"],
        "start_consumed_request_count": ledger["chain"][
            "consumed_request_count"
        ],
        "development_sec_execution_claim_sha256": sec_claim["claim_sha256"],
        "development_sec_reader_receipt_sha256": sec_reader["receipt_sha256"],
        "market_acquisition_plan": plan,
        "market_acquisition_plan_sha256": plan["acquisition_plan_sha256"],
        "candidate_source_hashes_sha256": canonical_sha256(candidate_sources),
        "execution_source_hashes": sources,
        "execution_source_hashes_sha256": canonical_sha256(sources),
        "execution_source_role_count": len(sources),
        "market_component_id": DEVELOPMENT_MARKET_BATCH_COMPONENT_ID,
        "source_family": MARKET_SOURCE_FAMILY,
        "market_symbols": list(MARKET_SYMBOLS),
        "market_fields": list(MARKET_FIELDS),
        "fixed_request_count": YAHOO_REQUEST_COUNT,
        "response_byte_ceiling_per_symbol": YAHOO_MAX_RESPONSE_BYTES,
        "timeout_seconds_per_symbol": YAHOO_REQUEST_TIMEOUT_SECONDS,
        "owned_market_execution_required": True,
        "caller_supplied_path_permitted": False,
        "caller_supplied_bytes_permitted": False,
        "market_access_permitted": True,
        "outcome_access_permitted": False,
        "future_stage_access_permitted": False,
        "paid_api_access_permitted": False,
        "external_network_access_permitted": True,
        "reveal_request_consumption_permitted": False,
        "consumption_ledger_mutation_permitted": False,
        "effect_may_be_repeated_after_indeterminate_crash": False,
    }
    return {**body, "claim_sha256": canonical_sha256(body)}


def build_development_market_reader_receipt(
    claim: Mapping[str, Any],
    *,
    acquisition_receipt_sha256: str,
    acquisition_bundle_sha256: str,
    acquisition_validation_sha256: str,
    source_manifest_sha256: str,
    market_stage_manifest_sha256: str,
    source_reconciliation_sha256: str,
    raw_response_sha256s: Mapping[str, Any],
    artifact_sha256s: Mapping[str, Any],
    window_sha256s: Mapping[str, Any],
    byte_index: list[dict[str, Any]],
    complete_marker_sha256: str,
    owned_transport_attested_by_store: bool,
) -> dict[str, Any]:
    """Bind store-rehashed market artifacts and provenance to one claim."""

    claim_value = _mapping(claim, "development market reader receipt claim")
    scope_hash = _sha256(
        claim_value.get("development_root_scope_sha256"),
        "development market reader receipt scope hash",
    )
    _expect_keys(
        claim_value,
        _DEVELOPMENT_MARKET_EXECUTION_CLAIM_KEYS,
        "development market reader receipt claim",
    )
    _self_hash(
        claim_value,
        "claim_sha256",
        "development market reader receipt claim",
    )
    if owned_transport_attested_by_store is not True:
        raise SecFilingGemmaStageAuthorizationError(
            "Development market reader requires owned transport attestation"
        )
    raw_hashes = _validated_market_symbol_sha256s(
        raw_response_sha256s,
        "development market raw response hashes",
    )
    artifact_hashes = _validated_market_symbol_sha256s(
        artifact_sha256s,
        "development market artifact hashes",
    )
    window_hashes = _validated_market_symbol_sha256s(
        window_sha256s,
        "development market window hashes",
    )
    acquisition_receipt_hash = _sha256(
        acquisition_receipt_sha256,
        "development market acquisition receipt hash",
    )
    source_manifest_hash = _sha256(
        source_manifest_sha256,
        "development market source manifest hash",
    )
    market_stage_manifest_hash = _sha256(
        market_stage_manifest_sha256,
        "development market stage manifest hash",
    )
    source_reconciliation_hash = _sha256(
        source_reconciliation_sha256,
        "development market source reconciliation hash",
    )
    index = _validated_development_market_byte_index(
        byte_index,
        raw_response_sha256s=raw_hashes,
        artifact_sha256s=artifact_hashes,
        window_sha256s=window_hashes,
    )
    body = {
        "schema_version": DEVELOPMENT_MARKET_READER_RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "receipt_kind": "store_rehashed_owned_development_market_evidence_batch",
        "development_root_scope_sha256": scope_hash,
        "claim_sha256": claim_value["claim_sha256"],
        "development_content_root_plan_sha256": claim_value[
            "development_content_root_plan_sha256"
        ],
        "authorized_stage": "development",
        "candidate_sha256": claim_value["candidate_sha256"],
        "output_namespace": claim_value["output_namespace"],
        "development_sec_execution_claim_sha256": claim_value[
            "development_sec_execution_claim_sha256"
        ],
        "development_sec_reader_receipt_sha256": claim_value[
            "development_sec_reader_receipt_sha256"
        ],
        "market_acquisition_plan_sha256": claim_value[
            "market_acquisition_plan_sha256"
        ],
        "execution_source_hashes_sha256": claim_value[
            "execution_source_hashes_sha256"
        ],
        "execution_source_role_count": claim_value[
            "execution_source_role_count"
        ],
        "market_component_id": DEVELOPMENT_MARKET_BATCH_COMPONENT_ID,
        "acquisition_receipt_sha256": acquisition_receipt_hash,
        "acquisition_bundle_sha256": _sha256(
            acquisition_bundle_sha256,
            "development market acquisition bundle hash",
        ),
        "acquisition_validation_sha256": _sha256(
            acquisition_validation_sha256,
            "development market acquisition validation hash",
        ),
        "source_manifest_sha256": source_manifest_hash,
        "market_stage_manifest_sha256": market_stage_manifest_hash,
        "source_reconciliation_sha256": source_reconciliation_hash,
        "raw_response_sha256s": raw_hashes,
        "artifact_sha256s": artifact_hashes,
        "window_sha256s": window_hashes,
        "byte_index": index,
        "byte_index_sha256": canonical_sha256(index),
        "byte_count_total": sum(item["byte_count"] for item in index),
        "complete_marker_sha256": _sha256(
            complete_marker_sha256,
            "development market complete marker hash",
        ),
        "fresh_network_provenance_claimed": False,
        "provider_response_normalization_replayed_by_store": True,
        "owned_transport_attested_by_store": True,
        "reader_output_recomputed_by_store": True,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def build_development_market_execution_abort(
    claim: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Terminally refuse retry after an indeterminate market network effect."""

    claim_value = _mapping(claim, "development market execution abort claim")
    _expect_keys(
        claim_value,
        _DEVELOPMENT_MARKET_EXECUTION_CLAIM_KEYS,
        "development market execution abort claim",
    )
    _self_hash(
        claim_value,
        "claim_sha256",
        "development market execution abort claim",
    )
    if reason not in {
        "claim_recovered_without_terminal_receipt",
        "external_effect_failed_or_completion_unknown",
        "durable_output_verification_failed",
    }:
        raise SecFilingGemmaStageAuthorizationError(
            "Development market execution abort reason is not canonical"
        )
    body = {
        "schema_version": DEVELOPMENT_MARKET_EXECUTION_ABORT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "abort_kind": "indeterminate_owned_development_market_evidence_batch",
        "development_root_scope_sha256": claim_value[
            "development_root_scope_sha256"
        ],
        "claim_sha256": claim_value["claim_sha256"],
        "development_content_root_plan_sha256": claim_value[
            "development_content_root_plan_sha256"
        ],
        "authorized_stage": "development",
        "candidate_sha256": claim_value["candidate_sha256"],
        "output_namespace": claim_value["output_namespace"],
        "market_component_id": DEVELOPMENT_MARKET_BATCH_COMPONENT_ID,
        "reason": reason,
        "external_effect_retry_permitted": False,
    }
    return {**body, "abort_sha256": canonical_sha256(body)}


def validate_development_market_execution_claim(
    claim: Mapping[str, Any],
    *,
    independent_current_tip_anchor: Mapping[str, Any],
) -> str:
    """Require exact current-tip membership for one development market claim."""

    observed = _mapping(claim, "development market execution claim")
    current_tip = validate_reveal_store_current_tip_anchor_structure(
        independent_current_tip_anchor
    )
    scope_hash = _sha256(
        observed.get("development_root_scope_sha256"),
        "development market claim root scope hash",
    )
    if (
        current_tip["development_market_execution_claims"].get(scope_hash)
        != observed
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development market execution claim is not exact at the current tip"
        )
    return _self_hash(
        observed,
        "claim_sha256",
        "development market execution claim",
    )


def validate_development_market_reader_receipt(
    receipt: Mapping[str, Any],
    *,
    independent_current_tip_anchor: Mapping[str, Any],
    acquisition_receipt_sha256: str,
    acquisition_bundle_sha256: str,
    acquisition_validation_sha256: str,
    source_manifest_sha256: str,
    market_stage_manifest_sha256: str,
    source_reconciliation_sha256: str,
    raw_response_sha256s: Mapping[str, Any],
    artifact_sha256s: Mapping[str, Any],
    window_sha256s: Mapping[str, Any],
    byte_index: list[dict[str, Any]],
    complete_marker_sha256: str,
    owned_transport_attested_by_store: bool,
) -> str:
    """Require exact membership and rehashed development market artifacts."""

    observed = _mapping(receipt, "development market reader receipt")
    current_tip = validate_reveal_store_current_tip_anchor_structure(
        independent_current_tip_anchor
    )
    scope_hash = _sha256(
        observed.get("development_root_scope_sha256"),
        "development market reader root scope hash",
    )
    claim = current_tip["development_market_execution_claims"].get(scope_hash)
    if (
        type(claim) is not dict
        or current_tip["development_market_reader_receipts"].get(scope_hash)
        != observed
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development market reader receipt is not exact at the current tip"
        )
    expected = build_development_market_reader_receipt(
        claim,
        acquisition_receipt_sha256=acquisition_receipt_sha256,
        acquisition_bundle_sha256=acquisition_bundle_sha256,
        acquisition_validation_sha256=acquisition_validation_sha256,
        source_manifest_sha256=source_manifest_sha256,
        market_stage_manifest_sha256=market_stage_manifest_sha256,
        source_reconciliation_sha256=source_reconciliation_sha256,
        raw_response_sha256s=raw_response_sha256s,
        artifact_sha256s=artifact_sha256s,
        window_sha256s=window_sha256s,
        byte_index=byte_index,
        complete_marker_sha256=complete_marker_sha256,
        owned_transport_attested_by_store=owned_transport_attested_by_store,
    )
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Development market reader receipt differs from rehashed durable bytes"
        )
    return observed["receipt_sha256"]


def validate_development_market_execution_abort(
    abort: Mapping[str, Any],
    *,
    independent_current_tip_anchor: Mapping[str, Any],
) -> str:
    """Require exact current-tip membership for one development market abort."""

    observed = _mapping(abort, "development market execution abort")
    current_tip = validate_reveal_store_current_tip_anchor_structure(
        independent_current_tip_anchor
    )
    scope_hash = _sha256(
        observed.get("development_root_scope_sha256"),
        "development market abort root scope hash",
    )
    if (
        current_tip["development_market_execution_aborts"].get(scope_hash)
        != observed
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development market execution abort is not exact at the current tip"
        )
    return _self_hash(
        observed,
        "abort_sha256",
        "development market execution abort",
    )


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


def build_stage_model_execution_claim(
    authorization_bundle: Mapping[str, Any],
    *,
    independent_current_tip_anchor: Mapping[str, Any],
    execution_source_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    """Claim one exact intermediate/final owned Gemma batch at the current tip."""

    bundle, grant, candidate, accessions = _stage_model_candidate_context(
        authorization_bundle
    )
    current_tip = validate_reveal_store_current_tip_anchor(
        bundle["authenticated_store_snapshot"],
        independent_current_tip_anchor,
    )
    request_hash = grant["request_sha256"]
    if current_tip["authorization_bundles"].get(request_hash) != bundle:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model authorization bundle is not exact at the current tip"
        )
    stage = grant["stage"]
    if stage not in _STAGE_PREREQUISITES:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model execution supports only intermediate and final requests"
        )
    sec_claim = current_tip["stage_sec_execution_claims"].get(request_hash)
    sec_reader = current_tip["stage_sec_reader_receipts"].get(request_hash)
    if (
        type(sec_claim) is not dict
        or type(sec_reader) is not dict
        or request_hash in current_tip["stage_sec_execution_aborts"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model execution requires one terminal owned SEC reader"
        )
    final_carry = current_tip["stage_carry_in_reader_receipts"].get(request_hash)
    root_carry = current_tip[
        "development_root_carry_in_reader_receipts"
    ].get(request_hash)
    if stage == "intermediate":
        carry = root_carry
        other_carry = final_carry
        carry_kind = "development_root_carry_in"
    else:
        carry = final_carry
        other_carry = root_carry
        carry_kind = "stage_carry_in"
    if type(carry) is not dict or other_carry is not None:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model execution requires exactly one stage-correct carry receipt"
        )
    root_claim, root_reader, universe = _stage_model_development_root_context(
        current_tip=current_tip,
        bundle=bundle,
        grant=grant,
        candidate=candidate,
    )
    market_bindings = _development_market_model_bindings(
        development_root_scope_sha256=root_claim[
            "development_root_scope_sha256"
        ],
        development_market_execution_claims=current_tip[
            "development_market_execution_claims"
        ],
        development_market_reader_receipts=current_tip[
            "development_market_reader_receipts"
        ],
        development_market_execution_aborts=current_tip[
            "development_market_execution_aborts"
        ],
        location="Stage model execution",
    )
    event_plan = _model_event_plan(
        universe_manifest=universe,
        stage=stage,
        sec_acquisition_accession_order=accessions,
    )
    if request_hash in current_tip["consumed_stage_output_receipts"]:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model execution must precede the same request's stage output"
        )
    if request_hash in current_tip["stage_model_execution_claims"]:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model execution is already claimed"
        )
    active_sec = set(current_tip["stage_sec_execution_claims"]) - set(
        current_tip["stage_sec_reader_receipts"]
    ) - set(current_tip["stage_sec_execution_aborts"])
    active_development_sec = set(
        current_tip["development_sec_execution_claims"]
    ) - set(current_tip["development_sec_reader_receipts"]) - set(
        current_tip["development_sec_execution_aborts"]
    )
    active_model = set(current_tip["stage_model_execution_claims"]) - set(
        current_tip["stage_model_reader_receipts"]
    ) - set(current_tip["stage_model_execution_aborts"])
    active_development_model = set(
        current_tip["development_model_execution_claims"]
    ) - set(current_tip["development_model_reader_receipts"]) - set(
        current_tip["development_model_execution_aborts"]
    )
    active_market = set(
        current_tip["development_market_execution_claims"]
    ) - set(current_tip["development_market_reader_receipts"]) - set(
        current_tip["development_market_execution_aborts"]
    )
    if (
        active_sec
        or active_development_sec
        or active_model
        or active_development_model
        or active_market
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Another owned external effect is already active"
        )
    sources = _validated_model_execution_source_hashes(
        execution_source_hashes,
        location="owned stage model execution source hashes",
    )
    candidate_sources = _mapping(
        candidate["bindings"]["source_hashes"],
        "owned stage model candidate source hashes",
    )
    if (
        candidate["bindings"]["identity_lexicon_sha256"]
        != CANONICAL_IDENTITY_LEXICON_SHA256
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Owned stage model candidate changed the canonical identity lexicon"
        )
    if any(candidate_sources.get(role) != value for role, value in sources.items()):
        raise SecFilingGemmaStageAuthorizationError(
            "Owned stage model execution bytes differ from the registered candidate"
        )
    model = _mapping(candidate["model"], "owned stage model candidate model")
    limits = _expected_model_runtime_limits(
        stage=stage,
        accession_count=len(accessions),
    )
    body = {
        "schema_version": STAGE_MODEL_EXECUTION_CLAIM_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "claim_kind": "owned_stage_gemma_model_batch",
        "request_sha256": request_hash,
        "consumption_entry_sha256": grant["consumption_entry_sha256"],
        "consumption_entry_sequence": grant["consumption_entry_sequence"],
        "attempt_id": grant["attempt_id"],
        "candidate_sha256": grant["candidate_sha256"],
        "registry_entry_sha256": grant["registry_entry_sha256"],
        "input_prerequisite_stage": grant["prerequisite_stage"],
        "authorized_stage": stage,
        "input_stage_evidence_sha256": grant[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": grant["stage_access_manifest_sha256"],
        "output_namespace": grant["output_namespace"],
        "authorization_bundle_sha256": bundle["bundle_sha256"],
        "authorization_grant_sha256": grant["authorization_grant_sha256"],
        "grant_store_state_sha256": grant["store_state_sha256"],
        "grant_consumption_ledger_sha256": grant["consumption_ledger_sha256"],
        "grant_consumption_ledger_tip_sha256": grant[
            "consumption_ledger_tip_sha256"
        ],
        "start_current_tip_anchor_sha256": current_tip["tip_anchor_sha256"],
        "stage_sec_execution_claim_sha256": sec_claim["claim_sha256"],
        "stage_sec_reader_receipt_sha256": sec_reader["receipt_sha256"],
        "development_root_scope_sha256": root_claim[
            "development_root_scope_sha256"
        ],
        "development_sec_execution_claim_sha256": root_claim["claim_sha256"],
        "development_sec_reader_receipt_sha256": root_reader["receipt_sha256"],
        **market_bindings,
        "corpus_universe_sha256": universe["universe_sha256"],
        "carry_in_kind": carry_kind,
        "carry_in_reader_receipt_sha256": carry["receipt_sha256"],
        "sec_document_count": len(accessions),
        "sec_acquisition_accession_order": accessions,
        "sec_acquisition_accession_order_sha256": canonical_sha256(accessions),
        "event_count": len(event_plan),
        "event_plan": event_plan,
        "event_plan_sha256": canonical_sha256(event_plan),
        "identity_lexicon_sha256": CANONICAL_IDENTITY_LEXICON_SHA256,
        "candidate_source_hashes_sha256": canonical_sha256(candidate_sources),
        "execution_source_hashes": sources,
        "execution_source_hashes_sha256": canonical_sha256(sources),
        "execution_source_role_count": len(sources),
        "model_name": model["name"],
        "model_digest": model["digest"],
        "runtime_fingerprint_sha256": model["runtime_fingerprint_sha256"],
        "model_transport_sha256": canonical_sha256(model["transport"]),
        "model_runtime_limits": limits,
        "model_runtime_limits_sha256": canonical_sha256(limits),
        "model_component_id": STAGE_MODEL_BATCH_COMPONENT_ID,
        "owned_local_model_execution_required": True,
        "caller_supplied_path_permitted": False,
        "filing_text_included": False,
        "market_access_permitted": False,
        "outcome_access_permitted": False,
        "future_stage_access_permitted": False,
        "paid_api_access_permitted": False,
        "external_network_access_permitted": False,
        "effect_may_be_repeated_after_indeterminate_crash": False,
    }
    return {**body, "claim_sha256": canonical_sha256(body)}


def build_stage_model_reader_receipt(
    claim: Mapping[str, Any],
    *,
    byte_index: list[dict[str, Any]],
    complete_marker_sha256: str,
) -> dict[str, Any]:
    """Bind store-rehashed durable stage model artifacts to one claim."""

    claim_value = _mapping(claim, "stage model reader receipt claim")
    _expect_keys(
        claim_value,
        _STAGE_MODEL_EXECUTION_CLAIM_KEYS,
        "stage model reader receipt claim",
    )
    _self_hash(claim_value, "claim_sha256", "stage model reader receipt claim")
    index = _validated_sec_byte_index(byte_index)
    body = {
        "schema_version": STAGE_MODEL_READER_RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "receipt_kind": "store_rehashed_owned_stage_gemma_model_batch",
        "request_sha256": claim_value["request_sha256"],
        "claim_sha256": claim_value["claim_sha256"],
        "authorized_stage": claim_value["authorized_stage"],
        "candidate_sha256": claim_value["candidate_sha256"],
        "output_namespace": claim_value["output_namespace"],
        "stage_sec_execution_claim_sha256": claim_value[
            "stage_sec_execution_claim_sha256"
        ],
        "stage_sec_reader_receipt_sha256": claim_value[
            "stage_sec_reader_receipt_sha256"
        ],
        "development_root_scope_sha256": claim_value[
            "development_root_scope_sha256"
        ],
        "development_sec_execution_claim_sha256": claim_value[
            "development_sec_execution_claim_sha256"
        ],
        "development_sec_reader_receipt_sha256": claim_value[
            "development_sec_reader_receipt_sha256"
        ],
        "development_market_execution_claim_sha256": claim_value[
            "development_market_execution_claim_sha256"
        ],
        "development_market_reader_receipt_sha256": claim_value[
            "development_market_reader_receipt_sha256"
        ],
        "development_market_acquisition_receipt_sha256": claim_value[
            "development_market_acquisition_receipt_sha256"
        ],
        "development_market_acquisition_bundle_sha256": claim_value[
            "development_market_acquisition_bundle_sha256"
        ],
        "development_market_acquisition_validation_sha256": claim_value[
            "development_market_acquisition_validation_sha256"
        ],
        "development_market_source_manifest_sha256": claim_value[
            "development_market_source_manifest_sha256"
        ],
        "development_market_stage_manifest_sha256": claim_value[
            "development_market_stage_manifest_sha256"
        ],
        "development_market_source_reconciliation_sha256": claim_value[
            "development_market_source_reconciliation_sha256"
        ],
        "development_market_byte_index_sha256": claim_value[
            "development_market_byte_index_sha256"
        ],
        "corpus_universe_sha256": claim_value["corpus_universe_sha256"],
        "carry_in_kind": claim_value["carry_in_kind"],
        "carry_in_reader_receipt_sha256": claim_value[
            "carry_in_reader_receipt_sha256"
        ],
        "sec_document_count": claim_value["sec_document_count"],
        "sec_acquisition_accession_order_sha256": claim_value[
            "sec_acquisition_accession_order_sha256"
        ],
        "event_count": claim_value["event_count"],
        "event_plan_sha256": claim_value["event_plan_sha256"],
        "identity_lexicon_sha256": claim_value["identity_lexicon_sha256"],
        "execution_source_hashes_sha256": claim_value[
            "execution_source_hashes_sha256"
        ],
        "execution_source_role_count": claim_value[
            "execution_source_role_count"
        ],
        "model_name": claim_value["model_name"],
        "model_digest": claim_value["model_digest"],
        "runtime_fingerprint_sha256": claim_value[
            "runtime_fingerprint_sha256"
        ],
        "model_transport_sha256": claim_value["model_transport_sha256"],
        "model_runtime_limits_sha256": claim_value[
            "model_runtime_limits_sha256"
        ],
        "model_component_id": claim_value["model_component_id"],
        "byte_index": index,
        "byte_index_sha256": canonical_sha256(index),
        "byte_count_total": sum(item["byte_count"] for item in index),
        "complete_marker_sha256": _sha256(
            complete_marker_sha256,
            "stage model complete marker hash",
        ),
        "fresh_model_provenance_claimed": False,
        "reader_output_recomputed_by_store": True,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def build_stage_model_execution_abort(
    claim: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Terminally refuse retry after a possibly executed stage model effect."""

    claim_value = _mapping(claim, "stage model execution abort claim")
    _expect_keys(
        claim_value,
        _STAGE_MODEL_EXECUTION_CLAIM_KEYS,
        "stage model execution abort claim",
    )
    _self_hash(claim_value, "claim_sha256", "stage model execution abort claim")
    if reason not in {
        "claim_recovered_without_terminal_receipt",
        "external_effect_failed_or_completion_unknown",
        "durable_output_verification_failed",
    }:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model execution abort reason is not canonical"
        )
    body = {
        "schema_version": STAGE_MODEL_EXECUTION_ABORT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "abort_kind": "indeterminate_owned_stage_gemma_model_batch",
        "request_sha256": claim_value["request_sha256"],
        "claim_sha256": claim_value["claim_sha256"],
        "authorized_stage": claim_value["authorized_stage"],
        "candidate_sha256": claim_value["candidate_sha256"],
        "output_namespace": claim_value["output_namespace"],
        "model_component_id": claim_value["model_component_id"],
        "reason": reason,
        "external_effect_retry_permitted": False,
    }
    return {**body, "abort_sha256": canonical_sha256(body)}


def build_development_model_execution_claim(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    development_root_scope_sha256: str,
    independent_current_tip_anchor: Mapping[str, Any],
    execution_source_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    """Claim the request-free development Gemma batch after its SEC root reader."""

    state, _ledger = _validated_store_snapshot(authenticated_store_snapshot)
    current_tip = validate_reveal_store_current_tip_anchor(
        state,
        independent_current_tip_anchor,
    )
    scope_hash = _sha256(
        development_root_scope_sha256,
        "development model root scope hash",
    )
    sec_claim = current_tip["development_sec_execution_claims"].get(scope_hash)
    sec_reader = current_tip["development_sec_reader_receipts"].get(scope_hash)
    if (
        type(sec_claim) is not dict
        or type(sec_reader) is not dict
        or scope_hash in current_tip["development_sec_execution_aborts"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development model execution requires one terminal development SEC reader"
        )
    market_bindings = _development_market_model_bindings(
        development_root_scope_sha256=scope_hash,
        development_market_execution_claims=current_tip[
            "development_market_execution_claims"
        ],
        development_market_reader_receipts=current_tip[
            "development_market_reader_receipts"
        ],
        development_market_execution_aborts=current_tip[
            "development_market_execution_aborts"
        ],
        location="Development model execution",
    )
    if scope_hash in current_tip["development_model_execution_claims"]:
        raise SecFilingGemmaStageAuthorizationError(
            "Development model execution is already claimed"
        )
    if current_tip["consumed_request_count"] != sec_claim[
        "start_consumed_request_count"
    ]:
        raise SecFilingGemmaStageAuthorizationError(
            "Development model execution must precede reveal-request consumption"
        )
    active_sec = set(current_tip["stage_sec_execution_claims"]) - set(
        current_tip["stage_sec_reader_receipts"]
    ) - set(current_tip["stage_sec_execution_aborts"])
    active_development_sec = set(
        current_tip["development_sec_execution_claims"]
    ) - set(current_tip["development_sec_reader_receipts"]) - set(
        current_tip["development_sec_execution_aborts"]
    )
    active_model = set(current_tip["stage_model_execution_claims"]) - set(
        current_tip["stage_model_reader_receipts"]
    ) - set(current_tip["stage_model_execution_aborts"])
    active_development_model = set(
        current_tip["development_model_execution_claims"]
    ) - set(current_tip["development_model_reader_receipts"]) - set(
        current_tip["development_model_execution_aborts"]
    )
    active_market = set(
        current_tip["development_market_execution_claims"]
    ) - set(current_tip["development_market_reader_receipts"]) - set(
        current_tip["development_market_execution_aborts"]
    )
    if (
        active_sec
        or active_development_sec
        or active_model
        or active_development_model
        or active_market
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Another owned external effect is already active"
        )
    plan = _validated_development_content_root_plan(
        sec_claim["development_content_root_plan"]
    )
    _state, _registry_entry, candidate = _development_registered_candidate(
        state,
        plan=plan,
        require_latest=False,
    )
    candidate_sources = _mapping(
        candidate["bindings"]["source_hashes"],
        "development model candidate source hashes",
    )
    if (
        candidate["bindings"]["identity_lexicon_sha256"]
        != CANONICAL_IDENTITY_LEXICON_SHA256
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Owned development model candidate changed the canonical identity lexicon"
        )
    sources = _validated_model_execution_source_hashes(
        execution_source_hashes,
        location="owned development model execution source hashes",
    )
    if any(candidate_sources.get(role) != value for role, value in sources.items()):
        raise SecFilingGemmaStageAuthorizationError(
            "Owned development model execution bytes differ from the registered candidate"
        )
    sec_sources = _mapping(
        sec_claim["execution_source_hashes"],
        "development SEC source hashes for model execution",
    )
    if any(
        sec_sources.get(role) != value
        for role, value in sources.items()
        if role in sec_sources
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Owned development model bytes differ from the terminal SEC root"
        )
    root_scope = plan["root_scope"]
    accessions = [
        document["accession_number"]
        for document in plan["sec_access_plan"]["documents"]
    ]
    accessions = _validated_model_accession_order(
        accessions,
        location="development model planned accession order",
    )
    universe = _mapping(
        plan["corpus_universe_manifest"],
        "development model corpus universe",
    )
    event_plan = _model_event_plan(
        universe_manifest=universe,
        stage="development",
        sec_acquisition_accession_order=accessions,
    )
    model = _mapping(candidate["model"], "development model candidate model")
    limits = _expected_model_runtime_limits(
        stage="development",
        accession_count=len(accessions),
    )
    body = {
        "schema_version": DEVELOPMENT_MODEL_EXECUTION_CLAIM_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "claim_kind": "owned_development_gemma_model_batch",
        "development_root_scope_sha256": scope_hash,
        "development_content_root_plan_sha256": plan[
            "development_content_root_plan_sha256"
        ],
        "attempt_id": sec_claim["attempt_id"],
        "candidate_sha256": sec_claim["candidate_sha256"],
        "candidate_design_sha256": sec_claim["candidate_design_sha256"],
        "registry_entry_sha256": sec_claim["registry_entry_sha256"],
        "registry_sha256": sec_claim["registry_sha256"],
        "registry_tip_sha256": sec_claim["registry_tip_sha256"],
        "corpus_universe_sha256": root_scope["corpus_universe_sha256"],
        "corpus_universe_semantic_sha256": root_scope[
            "corpus_universe_semantic_sha256"
        ],
        "authorized_stage": "development",
        "output_namespace": sec_claim["output_namespace"],
        "start_current_tip_anchor_sha256": current_tip["tip_anchor_sha256"],
        "start_consumed_request_count": current_tip["consumed_request_count"],
        "development_sec_execution_claim_sha256": sec_claim["claim_sha256"],
        "development_sec_reader_receipt_sha256": sec_reader["receipt_sha256"],
        **market_bindings,
        "sec_document_count": len(accessions),
        "sec_acquisition_accession_order": accessions,
        "sec_acquisition_accession_order_sha256": canonical_sha256(accessions),
        "event_count": len(event_plan),
        "event_plan": event_plan,
        "event_plan_sha256": canonical_sha256(event_plan),
        "identity_lexicon_sha256": CANONICAL_IDENTITY_LEXICON_SHA256,
        "candidate_source_hashes_sha256": canonical_sha256(candidate_sources),
        "execution_source_hashes": sources,
        "execution_source_hashes_sha256": canonical_sha256(sources),
        "execution_source_role_count": len(sources),
        "model_name": model["name"],
        "model_digest": model["digest"],
        "runtime_fingerprint_sha256": model["runtime_fingerprint_sha256"],
        "model_transport_sha256": canonical_sha256(model["transport"]),
        "model_runtime_limits": limits,
        "model_runtime_limits_sha256": canonical_sha256(limits),
        "model_component_id": STAGE_MODEL_BATCH_COMPONENT_ID,
        "carry_in_required": False,
        "owned_local_model_execution_required": True,
        "caller_supplied_path_permitted": False,
        "filing_text_included": False,
        "market_access_permitted": False,
        "outcome_access_permitted": False,
        "future_stage_access_permitted": False,
        "paid_api_access_permitted": False,
        "external_network_access_permitted": False,
        "reveal_request_consumption_permitted": False,
        "consumption_ledger_mutation_permitted": False,
        "effect_may_be_repeated_after_indeterminate_crash": False,
    }
    return {**body, "claim_sha256": canonical_sha256(body)}


def build_development_model_reader_receipt(
    claim: Mapping[str, Any],
    *,
    byte_index: list[dict[str, Any]],
    complete_marker_sha256: str,
) -> dict[str, Any]:
    """Bind store-rehashed development model artifacts to one root claim."""

    claim_value = _mapping(claim, "development model reader receipt claim")
    _expect_keys(
        claim_value,
        _DEVELOPMENT_MODEL_EXECUTION_CLAIM_KEYS,
        "development model reader receipt claim",
    )
    _self_hash(
        claim_value,
        "claim_sha256",
        "development model reader receipt claim",
    )
    index = _validated_sec_byte_index(byte_index)
    body = {
        "schema_version": DEVELOPMENT_MODEL_READER_RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "receipt_kind": "store_rehashed_owned_development_gemma_model_batch",
        "development_root_scope_sha256": claim_value[
            "development_root_scope_sha256"
        ],
        "claim_sha256": claim_value["claim_sha256"],
        "development_content_root_plan_sha256": claim_value[
            "development_content_root_plan_sha256"
        ],
        "authorized_stage": "development",
        "candidate_sha256": claim_value["candidate_sha256"],
        "output_namespace": claim_value["output_namespace"],
        "development_sec_execution_claim_sha256": claim_value[
            "development_sec_execution_claim_sha256"
        ],
        "development_sec_reader_receipt_sha256": claim_value[
            "development_sec_reader_receipt_sha256"
        ],
        "development_market_execution_claim_sha256": claim_value[
            "development_market_execution_claim_sha256"
        ],
        "development_market_reader_receipt_sha256": claim_value[
            "development_market_reader_receipt_sha256"
        ],
        "development_market_acquisition_receipt_sha256": claim_value[
            "development_market_acquisition_receipt_sha256"
        ],
        "development_market_acquisition_bundle_sha256": claim_value[
            "development_market_acquisition_bundle_sha256"
        ],
        "development_market_acquisition_validation_sha256": claim_value[
            "development_market_acquisition_validation_sha256"
        ],
        "development_market_source_manifest_sha256": claim_value[
            "development_market_source_manifest_sha256"
        ],
        "development_market_stage_manifest_sha256": claim_value[
            "development_market_stage_manifest_sha256"
        ],
        "development_market_source_reconciliation_sha256": claim_value[
            "development_market_source_reconciliation_sha256"
        ],
        "development_market_byte_index_sha256": claim_value[
            "development_market_byte_index_sha256"
        ],
        "sec_document_count": claim_value["sec_document_count"],
        "sec_acquisition_accession_order_sha256": claim_value[
            "sec_acquisition_accession_order_sha256"
        ],
        "event_count": claim_value["event_count"],
        "event_plan_sha256": claim_value["event_plan_sha256"],
        "identity_lexicon_sha256": claim_value["identity_lexicon_sha256"],
        "execution_source_hashes_sha256": claim_value[
            "execution_source_hashes_sha256"
        ],
        "execution_source_role_count": claim_value[
            "execution_source_role_count"
        ],
        "model_name": claim_value["model_name"],
        "model_digest": claim_value["model_digest"],
        "runtime_fingerprint_sha256": claim_value[
            "runtime_fingerprint_sha256"
        ],
        "model_transport_sha256": claim_value["model_transport_sha256"],
        "model_runtime_limits_sha256": claim_value[
            "model_runtime_limits_sha256"
        ],
        "model_component_id": claim_value["model_component_id"],
        "byte_index": index,
        "byte_index_sha256": canonical_sha256(index),
        "byte_count_total": sum(item["byte_count"] for item in index),
        "complete_marker_sha256": _sha256(
            complete_marker_sha256,
            "development model complete marker hash",
        ),
        "fresh_model_provenance_claimed": False,
        "reader_output_recomputed_by_store": True,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def build_development_model_execution_abort(
    claim: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Terminally refuse retry after a possibly executed development model effect."""

    claim_value = _mapping(claim, "development model execution abort claim")
    _expect_keys(
        claim_value,
        _DEVELOPMENT_MODEL_EXECUTION_CLAIM_KEYS,
        "development model execution abort claim",
    )
    _self_hash(
        claim_value,
        "claim_sha256",
        "development model execution abort claim",
    )
    if reason not in {
        "claim_recovered_without_terminal_receipt",
        "external_effect_failed_or_completion_unknown",
        "durable_output_verification_failed",
    }:
        raise SecFilingGemmaStageAuthorizationError(
            "Development model execution abort reason is not canonical"
        )
    body = {
        "schema_version": DEVELOPMENT_MODEL_EXECUTION_ABORT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "abort_kind": "indeterminate_owned_development_gemma_model_batch",
        "development_root_scope_sha256": claim_value[
            "development_root_scope_sha256"
        ],
        "claim_sha256": claim_value["claim_sha256"],
        "development_content_root_plan_sha256": claim_value[
            "development_content_root_plan_sha256"
        ],
        "authorized_stage": "development",
        "candidate_sha256": claim_value["candidate_sha256"],
        "output_namespace": claim_value["output_namespace"],
        "model_component_id": claim_value["model_component_id"],
        "reason": reason,
        "external_effect_retry_permitted": False,
    }
    return {**body, "abort_sha256": canonical_sha256(body)}


def validate_stage_model_execution_claim(
    claim: Mapping[str, Any],
    *,
    independent_current_tip_anchor: Mapping[str, Any],
) -> str:
    """Require exact current-tip membership for one stage model claim."""

    observed = _mapping(claim, "stage model execution claim")
    current_tip = validate_reveal_store_current_tip_anchor_structure(
        independent_current_tip_anchor
    )
    request_hash = _sha256(
        observed.get("request_sha256"),
        "stage model execution claim request hash",
    )
    if current_tip["stage_model_execution_claims"].get(request_hash) != observed:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model execution claim is not exact at the current tip"
        )
    return _self_hash(observed, "claim_sha256", "stage model execution claim")


def validate_stage_model_reader_receipt(
    receipt: Mapping[str, Any],
    *,
    independent_current_tip_anchor: Mapping[str, Any],
    byte_index: list[dict[str, Any]],
    complete_marker_sha256: str,
) -> str:
    """Require exact current membership and rehashed stage model bytes."""

    observed = _mapping(receipt, "stage model reader receipt")
    current_tip = validate_reveal_store_current_tip_anchor_structure(
        independent_current_tip_anchor
    )
    request_hash = _sha256(
        observed.get("request_sha256"),
        "stage model reader receipt request hash",
    )
    claim = current_tip["stage_model_execution_claims"].get(request_hash)
    if (
        type(claim) is not dict
        or current_tip["stage_model_reader_receipts"].get(request_hash)
        != observed
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model reader receipt is not exact at the current tip"
        )
    expected = build_stage_model_reader_receipt(
        claim,
        byte_index=byte_index,
        complete_marker_sha256=complete_marker_sha256,
    )
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model reader receipt differs from rehashed durable bytes"
        )
    return observed["receipt_sha256"]


def validate_stage_model_execution_abort(
    abort: Mapping[str, Any],
    *,
    independent_current_tip_anchor: Mapping[str, Any],
) -> str:
    """Require exact current-tip membership for one stage model abort."""

    observed = _mapping(abort, "stage model execution abort")
    current_tip = validate_reveal_store_current_tip_anchor_structure(
        independent_current_tip_anchor
    )
    request_hash = _sha256(
        observed.get("request_sha256"),
        "stage model abort request hash",
    )
    if current_tip["stage_model_execution_aborts"].get(request_hash) != observed:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage model execution abort is not exact at the current tip"
        )
    return _self_hash(observed, "abort_sha256", "stage model execution abort")


def validate_development_model_execution_claim(
    claim: Mapping[str, Any],
    *,
    independent_current_tip_anchor: Mapping[str, Any],
) -> str:
    """Require exact current-tip membership for one development model claim."""

    observed = _mapping(claim, "development model execution claim")
    current_tip = validate_reveal_store_current_tip_anchor_structure(
        independent_current_tip_anchor
    )
    scope_hash = _sha256(
        observed.get("development_root_scope_sha256"),
        "development model claim root scope hash",
    )
    if (
        current_tip["development_model_execution_claims"].get(scope_hash)
        != observed
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development model execution claim is not exact at the current tip"
        )
    return _self_hash(
        observed,
        "claim_sha256",
        "development model execution claim",
    )


def validate_development_model_reader_receipt(
    receipt: Mapping[str, Any],
    *,
    independent_current_tip_anchor: Mapping[str, Any],
    byte_index: list[dict[str, Any]],
    complete_marker_sha256: str,
) -> str:
    """Require exact current membership and rehashed development model bytes."""

    observed = _mapping(receipt, "development model reader receipt")
    current_tip = validate_reveal_store_current_tip_anchor_structure(
        independent_current_tip_anchor
    )
    scope_hash = _sha256(
        observed.get("development_root_scope_sha256"),
        "development model reader root scope hash",
    )
    claim = current_tip["development_model_execution_claims"].get(scope_hash)
    if (
        type(claim) is not dict
        or current_tip["development_model_reader_receipts"].get(scope_hash)
        != observed
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development model reader receipt is not exact at the current tip"
        )
    expected = build_development_model_reader_receipt(
        claim,
        byte_index=byte_index,
        complete_marker_sha256=complete_marker_sha256,
    )
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Development model reader receipt differs from rehashed durable bytes"
        )
    return observed["receipt_sha256"]


def validate_development_model_execution_abort(
    abort: Mapping[str, Any],
    *,
    independent_current_tip_anchor: Mapping[str, Any],
) -> str:
    """Require exact current-tip membership for one development model abort."""

    observed = _mapping(abort, "development model execution abort")
    current_tip = validate_reveal_store_current_tip_anchor_structure(
        independent_current_tip_anchor
    )
    scope_hash = _sha256(
        observed.get("development_root_scope_sha256"),
        "development model abort root scope hash",
    )
    if (
        current_tip["development_model_execution_aborts"].get(scope_hash)
        != observed
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development model execution abort is not exact at the current tip"
        )
    return _self_hash(
        observed,
        "abort_sha256",
        "development model execution abort",
    )


def _validated_development_feature_event_plan(raw: Any) -> list[dict[str, Any]]:
    if type(raw) is not list or not raw:
        raise SecFilingGemmaStageAuthorizationError(
            "Development feature assembly event plan must be a non-empty exact list"
        )
    event_keys = frozenset(
        {
            "event_ordinal",
            "accession_number",
            "form",
            "availability_session",
            "sec_document_ordinal",
        }
    )
    stage_start, stage_end = STAGE_WINDOWS["development"]
    events: list[dict[str, Any]] = []
    accessions: set[str] = set()
    sec_document_ordinals: set[int] = set()
    for ordinal, raw_event in enumerate(raw, start=1):
        event = _mapping(
            raw_event,
            f"development feature assembly event {ordinal}",
        )
        _expect_keys(
            event,
            event_keys,
            f"development feature assembly event {ordinal}",
        )
        if (
            _strict_int(
                event["event_ordinal"],
                "development feature assembly event ordinal",
                minimum=1,
            )
            != ordinal
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development feature assembly event ordinals are not contiguous"
            )
        accession = _safe_id(
            event["accession_number"],
            "development feature assembly accession",
        )
        session = event["availability_session"]
        sec_document_ordinal = _strict_int(
            event["sec_document_ordinal"],
            "development feature assembly SEC document ordinal",
            minimum=1,
        )
        if (
            _AAPL_ACCESSION_RE.fullmatch(accession) is None
            or type(event["form"]) is not str
            or event["form"] not in {"10-K", "10-Q"}
            or type(session) is not str
            or _ISO_DATE_RE.fullmatch(session) is None
            or not (stage_start <= session <= stage_end)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development feature assembly event identity is invalid or outside development"
            )
        if accession in accessions or sec_document_ordinal in sec_document_ordinals:
            raise SecFilingGemmaStageAuthorizationError(
                "Development feature assembly events contain duplicate source identities"
            )
        accessions.add(accession)
        sec_document_ordinals.add(sec_document_ordinal)
        events.append(event)
    chronology = [
        (event["availability_session"], event["accession_number"])
        for event in events
    ]
    if chronology != sorted(chronology):
        raise SecFilingGemmaStageAuthorizationError(
            "Development feature assembly events are not in exact chronological order"
        )
    if sec_document_ordinals != set(range(1, len(events) + 1)):
        raise SecFilingGemmaStageAuthorizationError(
            "Development feature assembly SEC document ordinals are not an exact permutation"
        )
    return events


def build_development_feature_assembly_plan(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    development_root_scope_sha256: str,
    independent_current_tip_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a request-free, feature-only plan from one terminal development root."""

    authenticated_state, _ledger = _validated_store_snapshot(
        authenticated_store_snapshot
    )
    current_tip = validate_reveal_store_current_tip_anchor(
        authenticated_state,
        independent_current_tip_anchor,
    )
    scope_hash = _sha256(
        development_root_scope_sha256,
        "development feature assembly root scope hash",
    )
    sec_claim = current_tip["development_sec_execution_claims"].get(scope_hash)
    sec_reader = current_tip["development_sec_reader_receipts"].get(scope_hash)
    market_claim = current_tip["development_market_execution_claims"].get(
        scope_hash
    )
    market_reader = current_tip["development_market_reader_receipts"].get(
        scope_hash
    )
    model_claim = current_tip["development_model_execution_claims"].get(scope_hash)
    model_reader = current_tip["development_model_reader_receipts"].get(scope_hash)
    if (
        any(
            type(value) is not dict
            for value in (
                sec_claim,
                sec_reader,
                market_claim,
                market_reader,
                model_claim,
                model_reader,
            )
        )
        or scope_hash in current_tip["development_sec_execution_aborts"]
        or scope_hash in current_tip["development_market_execution_aborts"]
        or scope_hash in current_tip["development_model_execution_aborts"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development feature assembly requires terminal non-aborted "
            "SEC, market, and model ancestry"
        )

    active_lifecycles = (
        (
            current_tip["stage_sec_execution_claims"],
            current_tip["stage_sec_reader_receipts"],
            current_tip["stage_sec_execution_aborts"],
        ),
        (
            current_tip["development_sec_execution_claims"],
            current_tip["development_sec_reader_receipts"],
            current_tip["development_sec_execution_aborts"],
        ),
        (
            current_tip["development_market_execution_claims"],
            current_tip["development_market_reader_receipts"],
            current_tip["development_market_execution_aborts"],
        ),
        (
            current_tip["stage_model_execution_claims"],
            current_tip["stage_model_reader_receipts"],
            current_tip["stage_model_execution_aborts"],
        ),
        (
            current_tip["development_model_execution_claims"],
            current_tip["development_model_reader_receipts"],
            current_tip["development_model_execution_aborts"],
        ),
    )
    if any(
        set(claims) - set(receipts) - set(aborts)
        for claims, receipts, aborts in active_lifecycles
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development feature assembly requires no active owned external effect"
        )
    if (
        current_tip["consumed_request_count"] != 0
        or sec_claim["start_consumed_request_count"] != 0
        or market_claim["start_consumed_request_count"] != 0
        or model_claim["start_consumed_request_count"] != 0
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development feature assembly must precede every reveal-request consumption"
        )

    market_bindings = _development_market_model_bindings(
        development_root_scope_sha256=scope_hash,
        development_market_execution_claims=current_tip[
            "development_market_execution_claims"
        ],
        development_market_reader_receipts=current_tip[
            "development_market_reader_receipts"
        ],
        development_market_execution_aborts=current_tip[
            "development_market_execution_aborts"
        ],
        location="Development feature assembly",
    )
    exact_model_ancestry = {
        "development_root_scope_sha256": scope_hash,
        "development_content_root_plan_sha256": sec_claim[
            "development_content_root_plan_sha256"
        ],
        "candidate_sha256": sec_claim["candidate_sha256"],
        "candidate_design_sha256": sec_claim["candidate_design_sha256"],
        "corpus_universe_sha256": sec_claim["corpus_universe_sha256"],
        "development_sec_execution_claim_sha256": sec_claim["claim_sha256"],
        "development_sec_reader_receipt_sha256": sec_reader["receipt_sha256"],
        **market_bindings,
    }
    if any(
        model_claim[field] != expected
        for field, expected in exact_model_ancestry.items()
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development feature assembly model claim crossed its terminal root ancestry"
        )
    if (
        model_reader["development_root_scope_sha256"] != scope_hash
        or model_reader["claim_sha256"] != model_claim["claim_sha256"]
        or any(
            model_reader[field] != model_claim[field]
            for field in (
                "development_content_root_plan_sha256",
                "candidate_sha256",
                "development_sec_execution_claim_sha256",
                "development_sec_reader_receipt_sha256",
                *tuple(market_bindings),
                "event_count",
                "event_plan_sha256",
                "execution_source_hashes_sha256",
            )
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development feature assembly model reader crossed its execution claim"
        )

    content_root_plan = _validated_development_content_root_plan(
        sec_claim["development_content_root_plan"]
    )
    planned_accessions = _validated_model_accession_order(
        [
            document["accession_number"]
            for document in content_root_plan["sec_access_plan"]["documents"]
        ],
        location="development feature assembly SEC acquisition order",
    )
    event_plan = _validated_model_event_plan(
        model_claim["event_plan"],
        universe_manifest=content_root_plan["corpus_universe_manifest"],
        stage="development",
        sec_acquisition_accession_order=planned_accessions,
    )
    event_plan = _validated_development_feature_event_plan(event_plan)
    event_plan_hash = canonical_sha256(event_plan)
    if (
        model_claim["event_count"] != len(event_plan)
        or model_claim["event_plan_sha256"] != event_plan_hash
        or model_reader["event_count"] != len(event_plan)
        or model_reader["event_plan_sha256"] != event_plan_hash
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development feature assembly event plan crossed its terminal model batch"
        )

    body = {
        "schema_version": DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "plan_kind": "request_free_development_feature_assembly",
        "artifact_stage": "development",
        "development_root_scope_sha256": scope_hash,
        "development_content_root_plan_sha256": model_claim[
            "development_content_root_plan_sha256"
        ],
        "candidate_sha256": model_claim["candidate_sha256"],
        "candidate_design_sha256": model_claim["candidate_design_sha256"],
        "corpus_universe_sha256": model_claim["corpus_universe_sha256"],
        "development_cutoff_session": STAGE_WINDOWS["development"][1],
        "start_consumed_request_count": 0,
        "development_sec_execution_claim_sha256": sec_claim["claim_sha256"],
        "development_sec_reader_receipt_sha256": sec_reader["receipt_sha256"],
        **market_bindings,
        "development_model_execution_claim_sha256": model_claim["claim_sha256"],
        "development_model_reader_receipt_sha256": model_reader[
            "receipt_sha256"
        ],
        "event_count": len(event_plan),
        "event_plan": event_plan,
        "event_plan_sha256": event_plan_hash,
        "execution_source_hashes_sha256": model_claim[
            "execution_source_hashes_sha256"
        ],
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
    return {
        **body,
        "feature_assembly_plan_sha256": canonical_sha256(body),
    }


def validate_development_feature_assembly_plan(
    plan: Mapping[str, Any],
    *,
    expected_feature_assembly_plan_sha256: str,
) -> str:
    """Validate the exact detached capability and chronology of one feature plan."""

    value = _mapping(plan, "development feature assembly plan")
    _expect_keys(
        value,
        _DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_KEYS,
        "development feature assembly plan",
    )
    expected_capabilities = {
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
    start_consumed_request_count = _strict_int(
        value["start_consumed_request_count"],
        "development feature assembly start consumed-request count",
    )
    if (
        value["schema_version"]
        != DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["plan_kind"] != "request_free_development_feature_assembly"
        or value["artifact_stage"] != "development"
        or value["development_cutoff_session"]
        != STAGE_WINDOWS["development"][1]
        or start_consumed_request_count != 0
        or any(
            value[field] is not expected
            for field, expected in expected_capabilities.items()
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development feature assembly plan semantics or capability boundary changed"
        )
    for field in (
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
    ):
        _sha256(value[field], f"development feature assembly plan {field}")
    event_plan = _validated_development_feature_event_plan(value["event_plan"])
    if (
        _strict_int(
            value["event_count"],
            "development feature assembly event count",
            minimum=1,
        )
        != len(event_plan)
        or value["event_plan_sha256"] != canonical_sha256(event_plan)
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development feature assembly event plan count or hash changed"
        )
    observed = _self_hash(
        value,
        "feature_assembly_plan_sha256",
        "development feature assembly plan",
    )
    expected = _sha256(
        expected_feature_assembly_plan_sha256,
        "expected development feature assembly plan hash",
    )
    if not hmac.compare_digest(observed, expected):
        raise SecFilingGemmaStageAuthorizationError(
            "Development feature assembly plan is not externally pinned"
        )
    return observed


def _expected_development_label_maturity_plan(
    source_feature_assembly_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Derive the exact t+21 maturity boundary without opening market values."""

    events = _validated_development_feature_event_plan(
        source_feature_assembly_plan.get("event_plan")
    )
    calendar = EXPECTED_MARKET_HISTORY_SESSIONS
    session_positions = {session: index for index, session in enumerate(calendar)}
    cutoff = STAGE_WINDOWS["development"][1]
    maturity_plan: list[dict[str, Any]] = []
    for event in events:
        decision_session = event["availability_session"]
        decision_index = session_positions.get(decision_session)
        if decision_index is None:
            raise SecFilingGemmaStageAuthorizationError(
                "Development label decision session is outside the frozen market calendar"
            )
        maturity_index = decision_index + LABEL_MATURITY_OFFSET
        if maturity_index >= len(calendar):
            raise SecFilingGemmaStageAuthorizationError(
                "Development label maturity is outside the frozen market calendar"
            )
        maturity_session = calendar[maturity_index]
        maturity_plan.append(
            {
                "event_ordinal": event["event_ordinal"],
                "accession_number": event["accession_number"],
                "form": event["form"],
                "decision_session": decision_session,
                "sec_document_ordinal": event["sec_document_ordinal"],
                "label_maturity_session": maturity_session,
                "matured_by_development_cutoff": maturity_session <= cutoff,
            }
        )
    return maturity_plan


def _validated_development_label_maturity_plan(
    raw: Any,
    *,
    source_feature_assembly_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = _expected_development_label_maturity_plan(
        source_feature_assembly_plan
    )
    if type(raw) is not list or len(raw) != len(expected):
        raise SecFilingGemmaStageAuthorizationError(
            "Development label maturity plan count changed"
        )
    observed: list[dict[str, Any]] = []
    for ordinal, (raw_item, expected_item) in enumerate(
        zip(raw, expected, strict=True),
        start=1,
    ):
        item = _mapping(raw_item, f"development label maturity item {ordinal}")
        _expect_keys(
            item,
            _DEVELOPMENT_LABEL_MATURITY_ITEM_KEYS,
            f"development label maturity item {ordinal}",
        )
        _strict_int(
            item["event_ordinal"],
            f"development label maturity item {ordinal} event ordinal",
            minimum=1,
        )
        _safe_id(
            item["accession_number"],
            f"development label maturity item {ordinal} accession",
        )
        if (
            type(item["form"]) is not str
            or item["form"] not in {"10-K", "10-Q"}
            or type(item["decision_session"]) is not str
            or _ISO_DATE_RE.fullmatch(item["decision_session"]) is None
            or type(item["label_maturity_session"]) is not str
            or _ISO_DATE_RE.fullmatch(item["label_maturity_session"]) is None
            or type(item["matured_by_development_cutoff"]) is not bool
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development label maturity item identity or boundary type changed"
            )
        _strict_int(
            item["sec_document_ordinal"],
            f"development label maturity item {ordinal} SEC document ordinal",
            minimum=1,
        )
        if item != expected_item:
            raise SecFilingGemmaStageAuthorizationError(
                "Development label maturity item differs from the frozen calendar derivation"
            )
        observed.append(item)
    return observed


def build_development_label_assembly_plan(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    development_root_scope_sha256: str,
    source_feature_assembly_plan: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a request-free plan for development labels matured by the cutoff."""

    supplied_feature_plan = _mapping(
        source_feature_assembly_plan,
        "development label source feature assembly plan",
    )
    supplied_feature_hash = _sha256(
        supplied_feature_plan.get("feature_assembly_plan_sha256"),
        "development label source feature assembly plan hash",
    )
    validate_development_feature_assembly_plan(
        supplied_feature_plan,
        expected_feature_assembly_plan_sha256=supplied_feature_hash,
    )
    rebuilt_feature_plan = build_development_feature_assembly_plan(
        authenticated_store_snapshot,
        development_root_scope_sha256=development_root_scope_sha256,
        independent_current_tip_anchor=independent_current_tip_anchor,
    )
    if supplied_feature_plan != rebuilt_feature_plan:
        raise SecFilingGemmaStageAuthorizationError(
            "Development label source feature plan differs from the terminal store replay"
        )
    maturity_plan = _expected_development_label_maturity_plan(
        supplied_feature_plan
    )
    matured_event_count = sum(
        item["matured_by_development_cutoff"] for item in maturity_plan
    )
    body = {
        "schema_version": DEVELOPMENT_LABEL_ASSEMBLY_PLAN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "plan_kind": "request_free_development_label_assembly",
        "artifact_stage": "development",
        "development_root_scope_sha256": supplied_feature_plan[
            "development_root_scope_sha256"
        ],
        "start_consumed_request_count": supplied_feature_plan[
            "start_consumed_request_count"
        ],
        "source_feature_assembly_plan": supplied_feature_plan,
        "source_feature_assembly_plan_sha256": supplied_feature_hash,
        "calendar_sessions_sha256": market_session_calendar_sha256(
            EXPECTED_MARKET_HISTORY_SESSIONS
        ),
        "development_cutoff_session": STAGE_WINDOWS["development"][1],
        "label_horizon_sessions": HORIZON_SESSIONS,
        "label_entry_session_offset": _DEVELOPMENT_LABEL_ENTRY_SESSION_OFFSET,
        "label_maturity_session_offset": LABEL_MATURITY_OFFSET,
        "maturity_rule": _DEVELOPMENT_LABEL_MATURITY_RULE,
        "event_count": len(maturity_plan),
        "maturity_plan": maturity_plan,
        "maturity_plan_sha256": canonical_sha256(maturity_plan),
        "matured_event_count": matured_event_count,
        "unmatured_event_count": len(maturity_plan) - matured_event_count,
        "canonical_market_rows_required": True,
        "development_outcome_derivation_permitted": True,
        "development_label_rows_output_permitted": True,
        "post_cutoff_market_access_permitted": False,
        "raw_market_output_permitted": False,
        "normalized_filing_text_output_permitted": False,
        "model_transport_envelope_output_permitted": False,
        "training_membership_access_permitted": False,
        "learner_fit_permitted": False,
        "prediction_access_permitted": False,
        "holdout_access_permitted": False,
        "ledger_mutation_permitted": False,
        "stage_promotion_permitted": False,
        "production_permitted": False,
    }
    return {**body, "label_assembly_plan_sha256": canonical_sha256(body)}


def validate_development_label_assembly_plan(
    plan: Mapping[str, Any],
    *,
    expected_label_assembly_plan_sha256: str,
) -> str:
    """Validate the exact development-only maturity plan and narrow boundary."""

    value = _mapping(plan, "development label assembly plan")
    _expect_keys(
        value,
        _DEVELOPMENT_LABEL_ASSEMBLY_PLAN_KEYS,
        "development label assembly plan",
    )
    source_feature_plan = _mapping(
        value["source_feature_assembly_plan"],
        "development label source feature assembly plan",
    )
    source_feature_hash = _sha256(
        value["source_feature_assembly_plan_sha256"],
        "development label source feature assembly plan hash",
    )
    validate_development_feature_assembly_plan(
        source_feature_plan,
        expected_feature_assembly_plan_sha256=source_feature_hash,
    )
    capabilities = {
        "canonical_market_rows_required": True,
        "development_outcome_derivation_permitted": True,
        "development_label_rows_output_permitted": True,
        "post_cutoff_market_access_permitted": False,
        "raw_market_output_permitted": False,
        "normalized_filing_text_output_permitted": False,
        "model_transport_envelope_output_permitted": False,
        "training_membership_access_permitted": False,
        "learner_fit_permitted": False,
        "prediction_access_permitted": False,
        "holdout_access_permitted": False,
        "ledger_mutation_permitted": False,
        "stage_promotion_permitted": False,
        "production_permitted": False,
    }
    event_count = _strict_int(
        value["event_count"],
        "development label event count",
        minimum=1,
    )
    matured_count = _strict_int(
        value["matured_event_count"],
        "development label matured event count",
    )
    unmatured_count = _strict_int(
        value["unmatured_event_count"],
        "development label unmatured event count",
    )
    start_consumed_request_count = _strict_int(
        value["start_consumed_request_count"],
        "development label start consumed-request count",
    )
    label_horizon_sessions = _strict_int(
        value["label_horizon_sessions"],
        "development label horizon sessions",
        minimum=1,
    )
    label_entry_session_offset = _strict_int(
        value["label_entry_session_offset"],
        "development label entry session offset",
        minimum=1,
    )
    label_maturity_session_offset = _strict_int(
        value["label_maturity_session_offset"],
        "development label maturity session offset",
        minimum=1,
    )
    _sha256(
        value["calendar_sessions_sha256"],
        "development label calendar sessions hash",
    )
    _sha256(
        value["maturity_plan_sha256"],
        "development label maturity plan hash",
    )
    _sha256(
        value["development_root_scope_sha256"],
        "development label root scope hash",
    )
    if (
        value["schema_version"]
        != DEVELOPMENT_LABEL_ASSEMBLY_PLAN_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["plan_kind"] != "request_free_development_label_assembly"
        or value["artifact_stage"] != "development"
        or value["development_root_scope_sha256"]
        != source_feature_plan["development_root_scope_sha256"]
        or value["development_cutoff_session"]
        != STAGE_WINDOWS["development"][1]
        or start_consumed_request_count != 0
        or start_consumed_request_count
        != source_feature_plan["start_consumed_request_count"]
        or label_horizon_sessions != HORIZON_SESSIONS
        or label_entry_session_offset
        != _DEVELOPMENT_LABEL_ENTRY_SESSION_OFFSET
        or label_maturity_session_offset != LABEL_MATURITY_OFFSET
        or value["maturity_rule"] != _DEVELOPMENT_LABEL_MATURITY_RULE
        or value["calendar_sessions_sha256"]
        != market_session_calendar_sha256(EXPECTED_MARKET_HISTORY_SESSIONS)
        or event_count != source_feature_plan["event_count"]
        or matured_count + unmatured_count != event_count
        or any(
            value[field] is not expected
            for field, expected in capabilities.items()
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development label assembly semantics or capability boundary changed"
        )
    maturity_plan = _validated_development_label_maturity_plan(
        value["maturity_plan"],
        source_feature_assembly_plan=source_feature_plan,
    )
    expected_matured_count = sum(
        item["matured_by_development_cutoff"] for item in maturity_plan
    )
    if (
        value["maturity_plan_sha256"] != canonical_sha256(maturity_plan)
        or matured_count != expected_matured_count
        or unmatured_count != event_count - expected_matured_count
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development label maturity plan hash or counts changed"
        )
    observed = _self_hash(
        value,
        "label_assembly_plan_sha256",
        "development label assembly plan",
    )
    expected = _sha256(
        expected_label_assembly_plan_sha256,
        "expected development label assembly plan hash",
    )
    if not hmac.compare_digest(observed, expected):
        raise SecFilingGemmaStageAuthorizationError(
            "Development label assembly plan is not externally pinned"
        )
    return observed


def _expected_development_training_membership_view_specs() -> list[dict[str, Any]]:
    specs = [
        {
            "view_ordinal": ordinal,
            "training_view_id": fold_id,
            "view_kind": "development_out_of_fold",
            "training_source_stage": "development",
            "prediction_stage": "development",
            "train_label_maturity_through": train_through,
            "prediction_window_first_date": test_first,
            "prediction_window_last_date": test_last,
            "state_updates_inside_prediction_window": False,
        }
        for ordinal, (fold_id, train_through, test_first, test_last) in enumerate(
            DEVELOPMENT_FOLD_SPECS,
            start=1,
        )
    ]
    specs.append(
        {
            "view_ordinal": len(specs) + 1,
            "training_view_id": "intermediate_frozen_through_2018",
            "view_kind": "intermediate_frozen_refit",
            "training_source_stage": "development",
            "prediction_stage": "intermediate",
            "train_label_maturity_through": STAGE_WINDOWS["development"][1],
            "prediction_window_first_date": STAGE_WINDOWS["intermediate"][0],
            "prediction_window_last_date": STAGE_WINDOWS["intermediate"][1],
            "state_updates_inside_prediction_window": False,
        }
    )
    return specs


def _validated_development_training_membership_view_specs(
    raw: Any,
) -> list[dict[str, Any]]:
    expected = _expected_development_training_membership_view_specs()
    if type(raw) is not list or len(raw) != len(expected):
        raise SecFilingGemmaStageAuthorizationError(
            "Development training-membership view count changed"
        )
    observed: list[dict[str, Any]] = []
    for ordinal, (raw_item, expected_item) in enumerate(
        zip(raw, expected, strict=True),
        start=1,
    ):
        item = _mapping(
            raw_item,
            f"development training-membership view {ordinal}",
        )
        _expect_keys(
            item,
            _DEVELOPMENT_TRAINING_MEMBERSHIP_VIEW_SPEC_KEYS,
            f"development training-membership view {ordinal}",
        )
        if (
            _strict_int(
                item["view_ordinal"],
                f"development training-membership view {ordinal} ordinal",
                minimum=1,
            )
            != ordinal
            or type(item["state_updates_inside_prediction_window"]) is not bool
            or item != expected_item
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development training-membership view differs from the frozen contract"
            )
        observed.append(item)
    return observed


def _expected_development_training_membership_variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "variant_ordinal": 1,
            "variant_id": "semantic",
            "feature_names_field": "semantic_feature_names",
            "feature_values_field": "semantic_feature_values_hex",
        },
        {
            "variant_ordinal": 2,
            "variant_id": "ablation",
            "feature_names_field": "ablation_feature_names",
            "feature_values_field": "ablation_feature_values_hex",
        },
    ]


def _validated_development_training_membership_variant_specs(
    raw: Any,
) -> list[dict[str, Any]]:
    expected = _expected_development_training_membership_variant_specs()
    if type(raw) is not list or len(raw) != len(expected):
        raise SecFilingGemmaStageAuthorizationError(
            "Development training-membership variant count changed"
        )
    observed: list[dict[str, Any]] = []
    for ordinal, (raw_item, expected_item) in enumerate(
        zip(raw, expected, strict=True),
        start=1,
    ):
        item = _mapping(
            raw_item,
            f"development training-membership variant {ordinal}",
        )
        _expect_keys(
            item,
            _DEVELOPMENT_TRAINING_MEMBERSHIP_VARIANT_SPEC_KEYS,
            f"development training-membership variant {ordinal}",
        )
        if (
            _strict_int(
                item["variant_ordinal"],
                f"development training-membership variant {ordinal} ordinal",
                minimum=1,
            )
            != ordinal
            or item != expected_item
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development training-membership variant differs from the frozen contract"
            )
        observed.append(item)
    return observed


def _expected_development_training_membership_capabilities() -> dict[str, bool]:
    return {
        "canonical_source_feature_rows_required": True,
        "canonical_source_label_rows_required": True,
        "source_feature_rows_access_permitted": True,
        "source_label_rows_access_permitted": True,
        "development_outcome_values_access_permitted": True,
        "development_outcome_derivation_permitted": False,
        "training_membership_access_permitted": True,
        "training_membership_rows_output_permitted": True,
        "training_feature_matrices_output_permitted": True,
        "training_target_vectors_output_permitted": True,
        "outcome_based_membership_filtering_permitted": False,
        "row_rebalancing_permitted": False,
        "post_cutoff_market_access_permitted": False,
        "raw_market_output_permitted": False,
        "compact_adjusted_open_paths_output_permitted": False,
        "normalized_filing_text_output_permitted": False,
        "model_transport_envelope_output_permitted": False,
        "learner_fit_permitted": False,
        "prediction_access_permitted": False,
        "holdout_access_permitted": False,
        "ledger_mutation_permitted": False,
        "stage_promotion_permitted": False,
        "production_permitted": False,
    }


def build_development_training_membership_assembly_plan(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    development_root_scope_sha256: str,
    source_label_assembly_plan: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize only deterministic membership assembly from owned development rows."""

    supplied_label_plan = _mapping(
        source_label_assembly_plan,
        "development training-membership source label assembly plan",
    )
    supplied_label_hash = _sha256(
        supplied_label_plan.get("label_assembly_plan_sha256"),
        "development training-membership source label assembly plan hash",
    )
    validate_development_label_assembly_plan(
        supplied_label_plan,
        expected_label_assembly_plan_sha256=supplied_label_hash,
    )
    supplied_feature_plan = _mapping(
        supplied_label_plan["source_feature_assembly_plan"],
        "development training-membership source feature assembly plan",
    )
    supplied_feature_hash = _sha256(
        supplied_label_plan["source_feature_assembly_plan_sha256"],
        "development training-membership source feature assembly plan hash",
    )
    validate_development_feature_assembly_plan(
        supplied_feature_plan,
        expected_feature_assembly_plan_sha256=supplied_feature_hash,
    )
    rebuilt_label_plan = build_development_label_assembly_plan(
        authenticated_store_snapshot,
        development_root_scope_sha256=development_root_scope_sha256,
        source_feature_assembly_plan=supplied_feature_plan,
        independent_current_tip_anchor=independent_current_tip_anchor,
    )
    if supplied_label_plan != rebuilt_label_plan:
        raise SecFilingGemmaStageAuthorizationError(
            "Development training-membership source label plan differs from "
            "the terminal store replay"
        )

    view_specs = _expected_development_training_membership_view_specs()
    variant_specs = _expected_development_training_membership_variant_specs()
    body = {
        "schema_version": (
            DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_SCHEMA_VERSION
        ),
        "contract_version": CONTRACT_VERSION,
        "plan_kind": "request_free_development_training_membership_assembly",
        "artifact_stage": "development",
        "development_root_scope_sha256": supplied_label_plan[
            "development_root_scope_sha256"
        ],
        "start_consumed_request_count": supplied_label_plan[
            "start_consumed_request_count"
        ],
        "source_label_assembly_plan": supplied_label_plan,
        "source_label_assembly_plan_sha256": supplied_label_hash,
        "source_feature_assembly_plan_sha256": supplied_feature_hash,
        "candidate_sha256": supplied_feature_plan["candidate_sha256"],
        "corpus_universe_sha256": supplied_feature_plan[
            "corpus_universe_sha256"
        ],
        "calendar_sessions_sha256": supplied_label_plan[
            "calendar_sessions_sha256"
        ],
        "development_cutoff_session": supplied_label_plan[
            "development_cutoff_session"
        ],
        "label_horizon_sessions": supplied_label_plan[
            "label_horizon_sessions"
        ],
        "label_entry_session_offset": supplied_label_plan[
            "label_entry_session_offset"
        ],
        "label_maturity_session_offset": supplied_label_plan[
            "label_maturity_session_offset"
        ],
        "event_count": supplied_label_plan["event_count"],
        "matured_event_count": supplied_label_plan["matured_event_count"],
        "unmatured_event_count": supplied_label_plan["unmatured_event_count"],
        "membership_view_count": len(view_specs),
        "membership_view_specs": view_specs,
        "membership_view_specs_sha256": canonical_sha256(view_specs),
        "model_variant_count": len(variant_specs),
        "model_variant_specs": variant_specs,
        "model_variant_specs_sha256": canonical_sha256(variant_specs),
        "membership_maturity_rule": (
            _DEVELOPMENT_TRAINING_MEMBERSHIP_MATURITY_RULE
        ),
        "feature_eligibility_rule": (
            _DEVELOPMENT_TRAINING_FEATURE_ELIGIBILITY_RULE
        ),
        "membership_order_rule": _DEVELOPMENT_TRAINING_MEMBERSHIP_ORDER_RULE,
        "shared_variant_support_rule": (
            _DEVELOPMENT_TRAINING_SHARED_VARIANT_SUPPORT_RULE
        ),
        "target_cost_bps": BRIER_TARGET_COST_BPS,
        "binary_target_field": _DEVELOPMENT_TRAINING_BINARY_TARGET_FIELD,
        "binary_target_encoding": _DEVELOPMENT_TRAINING_BINARY_TARGET_ENCODING,
        "edge_target_field": _DEVELOPMENT_TRAINING_EDGE_TARGET_FIELD,
        "minimum_training_row_count": _DEVELOPMENT_TRAINING_MINIMUM_ROW_COUNT,
        "both_binary_classes_required": True,
        "semantic_ablation_feature_matrices_must_differ": True,
        **_expected_development_training_membership_capabilities(),
    }
    return {
        **body,
        "training_membership_assembly_plan_sha256": canonical_sha256(body),
    }


def validate_development_training_membership_assembly_plan(
    plan: Mapping[str, Any],
    *,
    expected_training_membership_assembly_plan_sha256: str,
) -> str:
    """Validate the exact non-fitting membership authority and frozen views."""

    value = _mapping(plan, "development training-membership assembly plan")
    _expect_keys(
        value,
        _DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_KEYS,
        "development training-membership assembly plan",
    )
    source_label_plan = _mapping(
        value["source_label_assembly_plan"],
        "development training-membership source label assembly plan",
    )
    source_label_hash = _sha256(
        value["source_label_assembly_plan_sha256"],
        "development training-membership source label assembly plan hash",
    )
    validate_development_label_assembly_plan(
        source_label_plan,
        expected_label_assembly_plan_sha256=source_label_hash,
    )
    source_feature_plan = _mapping(
        source_label_plan["source_feature_assembly_plan"],
        "development training-membership source feature assembly plan",
    )
    source_feature_hash = _sha256(
        value["source_feature_assembly_plan_sha256"],
        "development training-membership source feature assembly plan hash",
    )
    validate_development_feature_assembly_plan(
        source_feature_plan,
        expected_feature_assembly_plan_sha256=source_feature_hash,
    )
    view_specs = _validated_development_training_membership_view_specs(
        value["membership_view_specs"]
    )
    variant_specs = _validated_development_training_membership_variant_specs(
        value["model_variant_specs"]
    )
    capabilities = _expected_development_training_membership_capabilities()
    if any(type(value[field]) is not bool for field in capabilities):
        raise SecFilingGemmaStageAuthorizationError(
            "Development training-membership capabilities must be exact booleans"
        )
    label_horizon_sessions = _strict_int(
        value["label_horizon_sessions"],
        "development training-membership label horizon sessions",
        minimum=1,
    )
    label_entry_session_offset = _strict_int(
        value["label_entry_session_offset"],
        "development training-membership label entry session offset",
        minimum=1,
    )
    label_maturity_session_offset = _strict_int(
        value["label_maturity_session_offset"],
        "development training-membership label maturity session offset",
        minimum=1,
    )
    if (
        value["schema_version"]
        != DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["plan_kind"]
        != "request_free_development_training_membership_assembly"
        or value["artifact_stage"] != "development"
        or value["development_root_scope_sha256"]
        != source_label_plan["development_root_scope_sha256"]
        or value["source_label_assembly_plan_sha256"]
        != source_label_plan["label_assembly_plan_sha256"]
        or value["source_feature_assembly_plan_sha256"]
        != source_label_plan["source_feature_assembly_plan_sha256"]
        or value["candidate_sha256"] != source_feature_plan["candidate_sha256"]
        or value["corpus_universe_sha256"]
        != source_feature_plan["corpus_universe_sha256"]
        or value["calendar_sessions_sha256"]
        != source_label_plan["calendar_sessions_sha256"]
        or value["development_cutoff_session"]
        != source_label_plan["development_cutoff_session"]
        or label_horizon_sessions != source_label_plan["label_horizon_sessions"]
        or label_horizon_sessions != HORIZON_SESSIONS
        or label_entry_session_offset
        != source_label_plan["label_entry_session_offset"]
        or label_entry_session_offset != _DEVELOPMENT_LABEL_ENTRY_SESSION_OFFSET
        or label_maturity_session_offset
        != source_label_plan["label_maturity_session_offset"]
        or label_maturity_session_offset != LABEL_MATURITY_OFFSET
        or value["membership_maturity_rule"]
        != _DEVELOPMENT_TRAINING_MEMBERSHIP_MATURITY_RULE
        or value["feature_eligibility_rule"]
        != _DEVELOPMENT_TRAINING_FEATURE_ELIGIBILITY_RULE
        or value["membership_order_rule"]
        != _DEVELOPMENT_TRAINING_MEMBERSHIP_ORDER_RULE
        or value["shared_variant_support_rule"]
        != _DEVELOPMENT_TRAINING_SHARED_VARIANT_SUPPORT_RULE
        or value["binary_target_field"]
        != _DEVELOPMENT_TRAINING_BINARY_TARGET_FIELD
        or value["binary_target_encoding"]
        != _DEVELOPMENT_TRAINING_BINARY_TARGET_ENCODING
        or value["edge_target_field"] != _DEVELOPMENT_TRAINING_EDGE_TARGET_FIELD
        or type(value["both_binary_classes_required"]) is not bool
        or value["both_binary_classes_required"] is not True
        or type(value["semantic_ablation_feature_matrices_must_differ"])
        is not bool
        or value["semantic_ablation_feature_matrices_must_differ"] is not True
        or any(
            value[field] is not expected
            for field, expected in capabilities.items()
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development training-membership semantics or capability boundary changed"
        )

    start_count = _strict_int(
        value["start_consumed_request_count"],
        "development training-membership start consumed-request count",
    )
    event_count = _strict_int(
        value["event_count"],
        "development training-membership event count",
        minimum=1,
    )
    matured_count = _strict_int(
        value["matured_event_count"],
        "development training-membership matured event count",
    )
    unmatured_count = _strict_int(
        value["unmatured_event_count"],
        "development training-membership unmatured event count",
    )
    view_count = _strict_int(
        value["membership_view_count"],
        "development training-membership view count",
        minimum=1,
    )
    variant_count = _strict_int(
        value["model_variant_count"],
        "development training-membership variant count",
        minimum=1,
    )
    target_cost_bps = _strict_int(
        value["target_cost_bps"],
        "development training-membership target cost bps",
        minimum=1,
    )
    minimum_training_row_count = _strict_int(
        value["minimum_training_row_count"],
        "development training-membership minimum training row count",
        minimum=2,
    )
    for field in (
        "development_root_scope_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "membership_view_specs_sha256",
        "model_variant_specs_sha256",
    ):
        _sha256(value[field], f"development training-membership {field}")
    if (
        start_count != 0
        or start_count != source_label_plan["start_consumed_request_count"]
        or event_count != source_label_plan["event_count"]
        or matured_count != source_label_plan["matured_event_count"]
        or unmatured_count != source_label_plan["unmatured_event_count"]
        or matured_count + unmatured_count != event_count
        or view_count != len(view_specs)
        or view_count != len(DEVELOPMENT_FOLD_SPECS) + 1
        or value["membership_view_specs_sha256"]
        != canonical_sha256(view_specs)
        or variant_count != len(variant_specs)
        or variant_count != 2
        or value["model_variant_specs_sha256"]
        != canonical_sha256(variant_specs)
        or target_cost_bps != BRIER_TARGET_COST_BPS
        or minimum_training_row_count != _DEVELOPMENT_TRAINING_MINIMUM_ROW_COUNT
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development training-membership counts, hashes, or frozen targets changed"
        )
    observed = _self_hash(
        value,
        "training_membership_assembly_plan_sha256",
        "development training-membership assembly plan",
    )
    expected = _sha256(
        expected_training_membership_assembly_plan_sha256,
        "expected development training-membership assembly plan hash",
    )
    if not hmac.compare_digest(observed, expected):
        raise SecFilingGemmaStageAuthorizationError(
            "Development training-membership assembly plan is not externally pinned"
        )
    return observed


def _expected_development_oof_learner_config() -> dict[str, Any]:
    config = SecFilingGemmaLearnerConfig()
    config.validate()
    return asdict(config)


def _validated_development_oof_learner_config(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "development OOF learner config")
    expected = _expected_development_oof_learner_config()
    _expect_keys(
        value,
        frozenset(expected),
        "development OOF learner config",
    )
    for field, expected_value in expected.items():
        observed = value[field]
        if type(expected_value) is int:
            matches = type(observed) is int and observed == expected_value
        else:
            matches = (
                type(expected_value) is float
                and type(observed) is float
                and math.isfinite(observed)
                and observed.hex() == expected_value.hex()
            )
        if not matches:
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF learner config differs from the frozen learner"
            )
    return value


def _expected_development_oof_capabilities() -> dict[str, bool]:
    return {
        "canonical_source_training_membership_batch_required": True,
        "source_training_membership_batch_access_permitted": True,
        "authorized_training_feature_matrices_access_permitted": True,
        "authorized_training_target_vectors_access_permitted": True,
        "deterministic_learner_fit_permitted": True,
        "learner_state_output_permitted": True,
        "compact_fit_audit_output_permitted": True,
        "deterministic_refit_validation_permitted": True,
        "source_feature_batch_access_permitted": False,
        "source_label_batch_access_permitted": False,
        "development_outcome_derivation_permitted": False,
        "training_membership_derivation_permitted": False,
        "training_membership_mutation_permitted": False,
        "outcome_based_membership_filtering_permitted": False,
        "row_rebalancing_permitted": False,
        "deferred_training_view_fit_permitted": False,
        "hyperparameter_change_permitted": False,
        "solver_retry_permitted": False,
        "feature_selection_permitted": False,
        "model_transport_access_permitted": False,
        "network_access_permitted": False,
        "prediction_access_permitted": False,
        "candidate_selection_permitted": False,
        "threshold_action_access_permitted": False,
        "holdout_access_permitted": False,
        "ledger_mutation_permitted": False,
        "stage_promotion_permitted": False,
        "production_permitted": False,
    }


def _validated_development_oof_iso_date(value: Any, location: str) -> str:
    if type(value) is not str or _ISO_DATE_RE.fullmatch(value) is None:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must be a canonical ISO date"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must be a canonical ISO date"
        ) from None
    if parsed.isoformat() != value:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must be a canonical ISO date"
        )
    return value


def _validated_development_oof_fit_input_specs(
    raw: Any,
    *,
    source_view_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    authorized_views = source_view_specs[: len(_DEVELOPMENT_OOF_AUTHORIZED_TRAINING_VIEW_IDS)]
    expected_pairs = [
        (view, head_variant)
        for view in authorized_views
        for head_variant in _DEVELOPMENT_OOF_MODEL_VARIANT_IDS
    ]
    if type(raw) is not list or len(raw) != len(expected_pairs):
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF learner fit-input count changed"
        )
    observed: list[dict[str, Any]] = []
    for fit_ordinal, (raw_item, (view, expected_variant)) in enumerate(
        zip(raw, expected_pairs, strict=True),
        start=1,
    ):
        item = _mapping(
            raw_item,
            f"development OOF learner fit input {fit_ordinal}",
        )
        _expect_keys(
            item,
            _DEVELOPMENT_OOF_LEARNER_FIT_INPUT_SPEC_KEYS,
            f"development OOF learner fit input {fit_ordinal}",
        )
        source_view_ordinal = _strict_int(
            item["source_training_view_ordinal"],
            f"development OOF fit input {fit_ordinal} source view ordinal",
            minimum=1,
        )
        row_count = _strict_int(
            item["training_row_count"],
            f"development OOF fit input {fit_ordinal} row count",
            minimum=2,
        )
        positive_count = _strict_int(
            item["training_positive_count"],
            f"development OOF fit input {fit_ordinal} positive count",
            minimum=1,
        )
        for field in (
            "source_training_view_sha256",
            "training_set_membership_sha256",
            "training_feature_matrix_sha256",
            "training_binary_target_sha256",
            "training_edge_target_sha256",
            "learner_input_context_sha256",
            "fit_metadata_template_sha256",
            "feature_schema_sha256",
        ):
            _sha256(
                item[field],
                f"development OOF fit input {fit_ordinal} {field}",
            )
        cutoff = _validated_development_oof_iso_date(
            item["train_label_maturity_through"],
            f"development OOF fit input {fit_ordinal} train cutoff",
        )
        maximum_maturity = _validated_development_oof_iso_date(
            item["maximum_training_label_maturity_session"],
            f"development OOF fit input {fit_ordinal} maximum maturity",
        )
        if (
            _strict_int(
                item["fit_ordinal"],
                f"development OOF fit input {fit_ordinal} ordinal",
                minimum=1,
            )
            != fit_ordinal
            or source_view_ordinal != view["view_ordinal"]
            or item["training_view_id"] != view["training_view_id"]
            or item["head_variant"] != expected_variant
            or cutoff != view["train_label_maturity_through"]
            or maximum_maturity > cutoff
            or maximum_maturity >= view["prediction_window_first_date"]
            or positive_count >= row_count
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF learner fit input crossed its frozen view, "
                "variant, class, or maturity boundary"
            )
        _self_hash(
            item,
            "fit_input_spec_sha256",
            f"development OOF learner fit input {fit_ordinal}",
        )
        observed.append(item)

    shared_pair_fields = (
        "source_training_view_ordinal",
        "training_view_id",
        "source_training_view_sha256",
        "training_row_count",
        "training_positive_count",
        "training_set_membership_sha256",
        "training_binary_target_sha256",
        "training_edge_target_sha256",
        "learner_input_context_sha256",
        "feature_schema_sha256",
        "train_label_maturity_through",
        "maximum_training_label_maturity_session",
    )
    for pair_start in range(0, len(observed), 2):
        semantic = observed[pair_start]
        ablation = observed[pair_start + 1]
        if (
            any(semantic[field] != ablation[field] for field in shared_pair_fields)
            or hmac.compare_digest(
                semantic["training_feature_matrix_sha256"],
                ablation["training_feature_matrix_sha256"],
            )
            or hmac.compare_digest(
                semantic["fit_metadata_template_sha256"],
                ablation["fit_metadata_template_sha256"],
            )
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF semantic and ablation fit inputs do not share "
                "one exact causal training support"
            )
    return observed


def _expected_development_oof_deferred_views(
    source_view_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deferred = source_view_specs[-1]
    if deferred["training_view_id"] != _DEVELOPMENT_OOF_DEFERRED_TRAINING_VIEW_ID:
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF source does not end with the frozen deferred view"
        )
    return [
        {
            "source_training_view_ordinal": deferred["view_ordinal"],
            "training_view_id": _DEVELOPMENT_OOF_DEFERRED_TRAINING_VIEW_ID,
            "reason": _DEVELOPMENT_OOF_DEFERRED_REASON,
        }
    ]


def _validated_development_oof_deferred_views(
    raw: Any,
    *,
    source_view_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected = _expected_development_oof_deferred_views(source_view_specs)
    if type(raw) is not list or len(raw) != 1:
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF deferred training-view count changed"
        )
    item = _mapping(raw[0], "development OOF deferred training view")
    _expect_keys(
        item,
        _DEVELOPMENT_OOF_DEFERRED_TRAINING_VIEW_KEYS,
        "development OOF deferred training view",
    )
    if (
        _strict_int(
            item["source_training_view_ordinal"],
            "development OOF deferred source view ordinal",
            minimum=1,
        )
        != len(source_view_specs)
        or item != expected[0]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF deferred training view changed"
        )
    return [item]


def build_development_oof_learner_fit_plan(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    development_root_scope_sha256: str,
    source_training_membership_assembly_plan: Mapping[str, Any],
    source_training_membership_projection_sha256: str,
    source_training_membership_batch_sha256: str,
    fit_input_specs: Any,
    independent_current_tip_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize only ten fixed development OOF learner fits."""

    supplied_membership_plan = _mapping(
        source_training_membership_assembly_plan,
        "development OOF source training-membership assembly plan",
    )
    supplied_membership_plan_hash = _sha256(
        supplied_membership_plan.get(
            "training_membership_assembly_plan_sha256"
        ),
        "development OOF source training-membership assembly plan hash",
    )
    validate_development_training_membership_assembly_plan(
        supplied_membership_plan,
        expected_training_membership_assembly_plan_sha256=(
            supplied_membership_plan_hash
        ),
    )
    source_label_plan = _mapping(
        supplied_membership_plan["source_label_assembly_plan"],
        "development OOF source label assembly plan",
    )
    rebuilt_membership_plan = (
        build_development_training_membership_assembly_plan(
            authenticated_store_snapshot,
            development_root_scope_sha256=development_root_scope_sha256,
            source_label_assembly_plan=source_label_plan,
            independent_current_tip_anchor=independent_current_tip_anchor,
        )
    )
    if supplied_membership_plan != rebuilt_membership_plan:
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF source training-membership plan differs from "
            "the terminal store replay"
        )

    projection_hash = _sha256(
        source_training_membership_projection_sha256,
        "development OOF source training-membership projection hash",
    )
    batch_hash = _sha256(
        source_training_membership_batch_sha256,
        "development OOF source training-membership batch hash",
    )
    source_view_specs = _validated_development_training_membership_view_specs(
        supplied_membership_plan["membership_view_specs"]
    )
    source_variant_specs = (
        _validated_development_training_membership_variant_specs(
            supplied_membership_plan["model_variant_specs"]
        )
    )
    validated_fit_specs = _validated_development_oof_fit_input_specs(
        fit_input_specs,
        source_view_specs=source_view_specs,
    )
    source_view_ids = [item["training_view_id"] for item in source_view_specs]
    authorized_view_ids = list(_DEVELOPMENT_OOF_AUTHORIZED_TRAINING_VIEW_IDS)
    deferred_views = _expected_development_oof_deferred_views(source_view_specs)
    variant_ids = [item["variant_id"] for item in source_variant_specs]
    learner_config = _expected_development_oof_learner_config()
    body = {
        "schema_version": DEVELOPMENT_OOF_LEARNER_FIT_PLAN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "plan_kind": "request_free_development_oof_learner_fit",
        "artifact_stage": "development",
        "development_root_scope_sha256": supplied_membership_plan[
            "development_root_scope_sha256"
        ],
        "start_consumed_request_count": supplied_membership_plan[
            "start_consumed_request_count"
        ],
        "source_training_membership_assembly_plan": supplied_membership_plan,
        "source_training_membership_assembly_plan_sha256": (
            supplied_membership_plan_hash
        ),
        "source_training_membership_projection_sha256": projection_hash,
        "source_training_membership_batch_sha256": batch_hash,
        "candidate_sha256": supplied_membership_plan["candidate_sha256"],
        "corpus_universe_sha256": supplied_membership_plan[
            "corpus_universe_sha256"
        ],
        "calendar_sessions_sha256": supplied_membership_plan[
            "calendar_sessions_sha256"
        ],
        "development_cutoff_session": supplied_membership_plan[
            "development_cutoff_session"
        ],
        "source_training_view_count": len(source_view_ids),
        "source_training_view_ids": source_view_ids,
        "source_training_view_specs_sha256": supplied_membership_plan[
            "membership_view_specs_sha256"
        ],
        "authorized_training_view_count": len(authorized_view_ids),
        "authorized_training_view_ids": authorized_view_ids,
        "deferred_training_view_count": len(deferred_views),
        "deferred_training_views": deferred_views,
        "deferred_training_views_sha256": canonical_sha256(deferred_views),
        "model_variant_count": len(variant_ids),
        "model_variant_ids": variant_ids,
        "learner_fit_input_count": len(validated_fit_specs),
        "learner_fit_input_specs": validated_fit_specs,
        "learner_fit_input_specs_sha256": canonical_sha256(
            validated_fit_specs
        ),
        "learner_state_output_count": len(validated_fit_specs),
        "learner_model_type": LEARNER_MODEL_TYPE,
        "learner_state_schema_version": LEARNER_STATE_SCHEMA_VERSION,
        "learner_config": learner_config,
        "learner_config_sha256": canonical_sha256(learner_config),
        "fit_order_rule": _DEVELOPMENT_OOF_FIT_ORDER_RULE,
        "maximum_fit_seconds": _DEVELOPMENT_OOF_MAXIMUM_FIT_SECONDS,
        **_expected_development_oof_capabilities(),
    }
    return {
        **body,
        "development_oof_learner_fit_plan_sha256": canonical_sha256(body),
    }


def validate_development_oof_learner_fit_plan(
    plan: Mapping[str, Any],
    *,
    expected_development_oof_learner_fit_plan_sha256: str,
) -> str:
    """Validate the exact ten-fit OOF authority and its deferred sixth view."""

    value = _mapping(plan, "development OOF learner-fit plan")
    _expect_keys(
        value,
        _DEVELOPMENT_OOF_LEARNER_FIT_PLAN_KEYS,
        "development OOF learner-fit plan",
    )
    source_plan = _mapping(
        value["source_training_membership_assembly_plan"],
        "development OOF source training-membership assembly plan",
    )
    source_plan_hash = _sha256(
        value["source_training_membership_assembly_plan_sha256"],
        "development OOF source training-membership assembly plan hash",
    )
    validate_development_training_membership_assembly_plan(
        source_plan,
        expected_training_membership_assembly_plan_sha256=source_plan_hash,
    )
    source_view_specs = _validated_development_training_membership_view_specs(
        source_plan["membership_view_specs"]
    )
    source_variant_specs = (
        _validated_development_training_membership_variant_specs(
            source_plan["model_variant_specs"]
        )
    )
    fit_specs = _validated_development_oof_fit_input_specs(
        value["learner_fit_input_specs"],
        source_view_specs=source_view_specs,
    )
    deferred_views = _validated_development_oof_deferred_views(
        value["deferred_training_views"],
        source_view_specs=source_view_specs,
    )
    learner_config = _validated_development_oof_learner_config(
        value["learner_config"]
    )
    capabilities = _expected_development_oof_capabilities()
    if any(type(value[field]) is not bool for field in capabilities):
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF learner-fit capabilities must be exact booleans"
        )

    source_view_count = _strict_int(
        value["source_training_view_count"],
        "development OOF source training-view count",
        minimum=1,
    )
    authorized_view_count = _strict_int(
        value["authorized_training_view_count"],
        "development OOF authorized training-view count",
        minimum=1,
    )
    deferred_view_count = _strict_int(
        value["deferred_training_view_count"],
        "development OOF deferred training-view count",
        minimum=1,
    )
    variant_count = _strict_int(
        value["model_variant_count"],
        "development OOF model-variant count",
        minimum=1,
    )
    fit_input_count = _strict_int(
        value["learner_fit_input_count"],
        "development OOF learner fit-input count",
        minimum=1,
    )
    state_output_count = _strict_int(
        value["learner_state_output_count"],
        "development OOF learner state-output count",
        minimum=1,
    )
    state_schema_version = _strict_int(
        value["learner_state_schema_version"],
        "development OOF learner state schema version",
        minimum=1,
    )
    maximum_fit_seconds = _strict_int(
        value["maximum_fit_seconds"],
        "development OOF maximum fit seconds",
        minimum=1,
    )
    start_count = _strict_int(
        value["start_consumed_request_count"],
        "development OOF start consumed-request count",
    )
    for field in (
        "contract_sha256",
        "development_root_scope_sha256",
        "source_training_membership_projection_sha256",
        "source_training_membership_batch_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "source_training_view_specs_sha256",
        "deferred_training_views_sha256",
        "learner_fit_input_specs_sha256",
        "learner_config_sha256",
    ):
        _sha256(value[field], f"development OOF {field}")

    source_view_ids = [item["training_view_id"] for item in source_view_specs]
    authorized_view_ids = list(_DEVELOPMENT_OOF_AUTHORIZED_TRAINING_VIEW_IDS)
    variant_ids = [item["variant_id"] for item in source_variant_specs]
    if (
        value["schema_version"]
        != DEVELOPMENT_OOF_LEARNER_FIT_PLAN_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["contract_sha256"]
        != canonical_sha256(build_contract_manifest())
        or value["plan_kind"] != "request_free_development_oof_learner_fit"
        or value["artifact_stage"] != "development"
        or value["development_root_scope_sha256"]
        != source_plan["development_root_scope_sha256"]
        or start_count != 0
        or start_count != source_plan["start_consumed_request_count"]
        or value["source_training_membership_assembly_plan_sha256"]
        != source_plan["training_membership_assembly_plan_sha256"]
        or value["candidate_sha256"] != source_plan["candidate_sha256"]
        or value["corpus_universe_sha256"]
        != source_plan["corpus_universe_sha256"]
        or value["calendar_sessions_sha256"]
        != source_plan["calendar_sessions_sha256"]
        or value["development_cutoff_session"]
        != source_plan["development_cutoff_session"]
        or source_view_count != len(source_view_specs)
        or source_view_count != len(DEVELOPMENT_FOLD_SPECS) + 1
        or type(value["source_training_view_ids"]) is not list
        or value["source_training_view_ids"] != source_view_ids
        or value["source_training_view_specs_sha256"]
        != source_plan["membership_view_specs_sha256"]
        or authorized_view_count != len(authorized_view_ids)
        or type(value["authorized_training_view_ids"]) is not list
        or value["authorized_training_view_ids"] != authorized_view_ids
        or deferred_view_count != len(deferred_views)
        or deferred_view_count != 1
        or value["deferred_training_views_sha256"]
        != canonical_sha256(deferred_views)
        or variant_count != len(variant_ids)
        or variant_count != len(_DEVELOPMENT_OOF_MODEL_VARIANT_IDS)
        or type(value["model_variant_ids"]) is not list
        or value["model_variant_ids"] != variant_ids
        or variant_ids != list(_DEVELOPMENT_OOF_MODEL_VARIANT_IDS)
        or fit_input_count != len(fit_specs)
        or fit_input_count != len(authorized_view_ids) * len(variant_ids)
        or value["learner_fit_input_specs_sha256"]
        != canonical_sha256(fit_specs)
        or state_output_count != fit_input_count
        or value["learner_model_type"] != LEARNER_MODEL_TYPE
        or state_schema_version != LEARNER_STATE_SCHEMA_VERSION
        or value["learner_config_sha256"] != canonical_sha256(learner_config)
        or value["fit_order_rule"] != _DEVELOPMENT_OOF_FIT_ORDER_RULE
        or maximum_fit_seconds != _DEVELOPMENT_OOF_MAXIMUM_FIT_SECONDS
        or any(
            value[field] is not expected
            for field, expected in capabilities.items()
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF learner-fit identity, order, or capability boundary changed"
        )

    observed = _self_hash(
        value,
        "development_oof_learner_fit_plan_sha256",
        "development OOF learner-fit plan",
    )
    expected = _sha256(
        expected_development_oof_learner_fit_plan_sha256,
        "expected development OOF learner-fit plan hash",
    )
    if not hmac.compare_digest(observed, expected):
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF learner-fit plan is not externally pinned"
        )
    return observed


def _expected_development_oof_prediction_capabilities() -> dict[str, bool]:
    return {
        "canonical_prediction_fold_model_bundle_required": True,
        "canonical_prediction_feature_batch_required": True,
        "authorized_learner_state_access_permitted": True,
        "authorized_prediction_feature_row_access_permitted": True,
        "learner_state_deserialization_permitted": True,
        "deterministic_numeric_prediction_permitted": True,
        "raw_prediction_component_output_permitted": True,
        "unavailable_prediction_output_permitted": True,
        "compact_prediction_audit_output_permitted": True,
        "source_development_oof_learner_fit_batch_access_permitted": False,
        "source_training_membership_batch_access_permitted": False,
        "training_membership_rows_access_permitted": False,
        "training_feature_matrices_access_permitted": False,
        "training_target_vectors_access_permitted": False,
        "source_feature_batch_access_permitted": False,
        "source_label_batch_access_permitted": False,
        "label_access_permitted": False,
        "outcome_access_permitted": False,
        "post_decision_market_data_access_permitted": False,
        "post_2018_data_access_permitted": False,
        "deferred_training_view_access_permitted": False,
        "deferred_training_view_state_access_permitted": False,
        "learner_fit_permitted": False,
        "learner_state_update_permitted": False,
        "online_learning_permitted": False,
        "feature_mutation_permitted": False,
        "row_drop_permitted": False,
        "row_reordering_permitted": False,
        "prediction_retry_permitted": False,
        "model_transport_access_permitted": False,
        "network_access_permitted": False,
        "threshold_action_access_permitted": False,
        "candidate_selection_permitted": False,
        "policy_state_transition_permitted": False,
        "prediction_sealing_permitted": False,
        "label_release_permitted": False,
        "holdout_access_permitted": False,
        "ledger_mutation_permitted": False,
        "stage_promotion_permitted": False,
        "production_permitted": False,
    }


def _expected_development_policy_replay_capabilities() -> dict[str, bool]:
    """Return the exact threshold/policy-only Phase A authority boundary."""

    return {
        "source_development_oof_prediction_batch_access_permitted": True,
        "raw_prediction_components_access_permitted": True,
        "threshold_evaluation_permitted": True,
        "policy_state_transition_permitted": True,
        "compact_policy_replay_output_permitted": True,
        "numeric_prediction_permitted": False,
        "learner_state_access_permitted": False,
        "source_feature_batch_access_permitted": False,
        "source_label_batch_access_permitted": False,
        "label_access_permitted": False,
        "outcome_access_permitted": False,
        "post_decision_market_data_access_permitted": False,
        "post_2018_data_access_permitted": False,
        "deferred_training_view_access_permitted": False,
        "deferred_training_view_state_access_permitted": False,
        "learner_fit_permitted": False,
        "learner_refit_permitted": False,
        "learner_state_update_permitted": False,
        "online_learning_permitted": False,
        "candidate_selection_permitted": False,
        "scoring_permitted": False,
        "prediction_sealing_permitted": False,
        "policy_replay_sealing_permitted": False,
        "label_release_permitted": False,
        "holdout_access_permitted": False,
        "model_transport_access_permitted": False,
        "network_access_permitted": False,
        "raw_prediction_mutation_permitted": False,
        "row_drop_permitted": False,
        "row_reordering_permitted": False,
        "policy_retry_permitted": False,
        "ledger_mutation_permitted": False,
        "stage_promotion_permitted": False,
        "production_permitted": False,
    }


def _expected_development_policy_prefix_seal_capabilities() -> dict[str, bool]:
    """Return the exact materialize-and-seal-only authority boundary."""

    return {
        "source_development_policy_replay_batch_access_permitted": True,
        "deterministic_policy_replay_validation_permitted": True,
        "cumulative_policy_prefix_materialization_permitted": True,
        "canonical_artifact_encoding_permitted": True,
        "fixed_namespace_artifact_store_write_permitted": True,
        "atomic_batch_compare_and_swap_permitted": True,
        "exact_idempotent_recovery_permitted": True,
        "seal_receipt_emission_permitted": True,
        "external_pin_emission_permitted": True,
        "raw_prediction_mutation_permitted": False,
        "policy_row_drop_permitted": False,
        "policy_row_reordering_permitted": False,
        "alternate_policy_retry_permitted": False,
        "source_feature_batch_access_permitted": False,
        "source_label_batch_access_permitted": False,
        "label_access_permitted": False,
        "outcome_access_permitted": False,
        "post_decision_market_data_access_permitted": False,
        "post_2018_data_access_permitted": False,
        "numeric_prediction_permitted": False,
        "learner_state_access_permitted": False,
        "learner_fit_permitted": False,
        "learner_refit_permitted": False,
        "learner_state_update_permitted": False,
        "new_threshold_evaluation_permitted": False,
        "new_policy_state_transition_permitted": False,
        "candidate_selection_permitted": False,
        "scoring_permitted": False,
        "ranking_permitted": False,
        "holdout_access_permitted": False,
        "model_transport_access_permitted": False,
        "network_access_permitted": False,
        "mutable_store_pin_discovery_as_external_permitted": False,
        "ledger_mutation_permitted": False,
        "stage_promotion_permitted": False,
        "production_permitted": False,
    }


def _development_policy_replay_candidate_threshold_specs() -> list[dict[str, Any]]:
    manifest = build_contract_manifest()
    predictor = _mapping(
        manifest.get("predictor"),
        "development policy replay contract predictor",
    )
    raw_grid = predictor.get("candidate_grid")
    if type(raw_grid) is not list or len(raw_grid) != len(CANDIDATE_IDS):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy replay contract candidate grid changed"
        )
    result: list[dict[str, Any]] = []
    for ordinal, (raw, candidate_id) in enumerate(
        zip(raw_grid, CANDIDATE_IDS, strict=True),
        start=1,
    ):
        candidate = _mapping(
            raw,
            f"development policy replay contract candidate {ordinal}",
        )
        if set(candidate) != {
            "candidate_id",
            "probability_gate",
            "expected_edge_gate",
        } or candidate.get("candidate_id") != candidate_id:
            raise SecFilingGemmaStageAuthorizationError(
                "Development policy replay candidate identity or order changed"
            )
        probability_gate = candidate["probability_gate"]
        expected_edge_gate = candidate["expected_edge_gate"]
        if (
            type(probability_gate) not in {int, float}
            or type(expected_edge_gate) not in {int, float}
            or not math.isfinite(float(probability_gate))
            or not math.isfinite(float(expected_edge_gate))
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development policy replay candidate gates must be finite numbers"
            )
        body = {
            "candidate_ordinal": ordinal,
            "candidate_id": candidate_id,
            "probability_gate_hex": float(probability_gate).hex(),
            "expected_edge_gate_hex": float(expected_edge_gate).hex(),
        }
        result.append(
            {
                **body,
                "candidate_threshold_spec_sha256": canonical_sha256(body),
            }
        )
    if predictor.get("candidate_gate_comparison") != (
        _DEVELOPMENT_POLICY_REPLAY_GATE_COMPARISON_RULE
    ) or predictor.get("cash_episode_sessions") != HORIZON_SESSIONS:
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy replay contract threshold or episode rule changed"
        )
    return result


def _validated_development_policy_replay_input_specs(
    raw: Any,
) -> list[dict[str, Any]]:
    if type(raw) is not list:
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy replay input specs must be an exact list"
        )
    supplied = _plain(raw, "development policy replay input specs")
    if type(supplied) is not list or not supplied:
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy replay input specs cannot be empty"
        )
    expected_keys = frozenset(
        {
            "schema_version",
            "input_ordinal",
            "source_raw_prediction_row_sha256",
            "source_feature_row_sha256",
            "event_binding_sha256",
            "decision_session",
            "accession_number",
            "fold_id",
            "prediction_fold_context_sha256",
            "semantic_learner_state_sha256",
            "ablation_learner_state_sha256",
            "prediction_status",
            "unavailable_reason",
            "numerical_components_sha256",
            "policy_replay_input_spec_sha256",
        }
    )
    fold_windows = {
        fold_id: (first, last)
        for fold_id, _train_cutoff, first, last in DEVELOPMENT_FOLD_SPECS
    }
    result: list[dict[str, Any]] = []
    raw_row_hashes: set[str] = set()
    feature_row_hashes: set[str] = set()
    event_hashes: set[str] = set()
    sessions: set[str] = set()
    accessions: set[str] = set()
    prior_session: str | None = None
    for ordinal, raw_spec in enumerate(supplied, start=1):
        spec = _mapping(
            raw_spec,
            f"development policy replay input {ordinal}",
        )
        _expect_keys(
            spec,
            expected_keys,
            f"development policy replay input {ordinal}",
        )
        if (
            spec["schema_version"]
            != "aapl-sec-gemma-development-policy-replay-input-spec-v1"
            or _strict_int(
                spec["input_ordinal"],
                f"development policy replay input {ordinal} ordinal",
                minimum=1,
            )
            != ordinal
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development policy replay input schema or order changed"
            )
        for field in (
            "source_raw_prediction_row_sha256",
            "source_feature_row_sha256",
            "event_binding_sha256",
            "prediction_fold_context_sha256",
            "semantic_learner_state_sha256",
            "ablation_learner_state_sha256",
            "numerical_components_sha256",
        ):
            _sha256(
                spec[field],
                f"development policy replay input {ordinal} {field}",
            )
        session = _validated_development_oof_iso_date(
            spec["decision_session"],
            f"development policy replay input {ordinal} decision session",
        )
        fold_id = spec["fold_id"]
        window = fold_windows.get(fold_id)
        accession = spec["accession_number"]
        if (
            window is None
            or not window[0] <= session <= window[1]
            or prior_session is not None
            and session <= prior_session
            or type(accession) is not str
            or _AAPL_ACCESSION_RE.fullmatch(accession) is None
            or spec["semantic_learner_state_sha256"]
            == spec["ablation_learner_state_sha256"]
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development policy replay input crossed its frozen chronology, fold, or state pair"
            )
        status = spec["prediction_status"]
        if status == "available_pre_label":
            if spec["unavailable_reason"] is not None:
                raise SecFilingGemmaStageAuthorizationError(
                    "Available development policy replay input has an unavailable reason"
                )
        elif status == "unavailable_pre_label":
            if spec["unavailable_reason"] not in (
                _DEVELOPMENT_OOF_PREDICTION_UNAVAILABLE_REASONS
            ):
                raise SecFilingGemmaStageAuthorizationError(
                    "Unavailable development policy replay input changed its reason"
                )
        else:
            raise SecFilingGemmaStageAuthorizationError(
                "Development policy replay input status changed"
            )
        _self_hash(
            spec,
            "policy_replay_input_spec_sha256",
            f"development policy replay input {ordinal}",
        )
        raw_row_hashes.add(spec["source_raw_prediction_row_sha256"])
        feature_row_hashes.add(spec["source_feature_row_sha256"])
        event_hashes.add(spec["event_binding_sha256"])
        sessions.add(session)
        accessions.add(accession)
        prior_session = session
        result.append(spec)
    if not all(
        len(values) == len(result)
        for values in (
            raw_row_hashes,
            feature_row_hashes,
            event_hashes,
            sessions,
            accessions,
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy replay inputs duplicate a raw row, feature row, event, session, or accession"
        )
    return result


def _validated_development_oof_prediction_fold_model_specs(
    raw: Any,
    *,
    source_fit_plan: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if type(raw) is not list:
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF prediction fold-model specs must be an exact list"
        )
    supplied = _plain(raw, "development OOF prediction fold-model specs")
    if type(supplied) is not list or len(supplied) != len(DEVELOPMENT_FOLD_SPECS):
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF prediction fold-model count changed"
        )
    fit_specs: list[dict[str, Any]] | None = None
    if source_fit_plan is not None:
        raw_fit_specs = source_fit_plan["learner_fit_input_specs"]
        if (
            type(raw_fit_specs) is not list
            or len(raw_fit_specs) != 2 * len(DEVELOPMENT_FOLD_SPECS)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF prediction source fit-state count changed"
            )
        fit_specs = raw_fit_specs
        feature_schema_sha256 = _sha256(
            fit_specs[0]["feature_schema_sha256"],
            "development OOF prediction feature schema hash",
        )
        if any(
            spec.get("feature_schema_sha256") != feature_schema_sha256
            for spec in fit_specs
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF prediction source fit feature schemas differ"
            )
    else:
        first_spec = _mapping(
            supplied[0], "development OOF prediction first fold model"
        )
        feature_schema_sha256 = _sha256(
            first_spec.get("feature_schema_sha256"),
            "development OOF prediction feature schema hash",
        )

    result: list[dict[str, Any]] = []
    state_hashes: set[str] = set()
    fit_record_hashes: set[str] = set()
    fit_view_hashes: set[str] = set()
    fold_model_hashes: set[str] = set()
    for ordinal, (raw_spec, frozen_fold) in enumerate(
        zip(supplied, DEVELOPMENT_FOLD_SPECS, strict=True),
        start=1,
    ):
        spec = _mapping(raw_spec, f"development OOF prediction fold model {ordinal}")
        _expect_keys(
            spec,
            _DEVELOPMENT_OOF_PREDICTION_FOLD_MODEL_SPEC_KEYS,
            f"development OOF prediction fold model {ordinal}",
        )
        fold_id, _train_cutoff, first, last = frozen_fold
        semantic_fit_spec = (
            fit_specs[2 * (ordinal - 1)] if fit_specs is not None else None
        )
        ablation_fit_spec = (
            fit_specs[2 * (ordinal - 1) + 1]
            if fit_specs is not None
            else None
        )
        for field in (
            "source_learner_fit_view_sha256",
            "prediction_fold_context_sha256",
            "semantic_fit_record_sha256",
            "semantic_learner_state_sha256",
            "ablation_fit_record_sha256",
            "ablation_learner_state_sha256",
            "feature_schema_sha256",
            "prediction_fold_model_sha256",
        ):
            _sha256(
                spec[field],
                f"development OOF prediction fold model {ordinal} {field}",
            )
        _validated_development_oof_iso_date(
            spec["prediction_window_first_date"],
            f"development OOF prediction fold model {ordinal} first date",
        )
        _validated_development_oof_iso_date(
            spec["prediction_window_last_date"],
            f"development OOF prediction fold model {ordinal} last date",
        )
        if (
            _strict_int(
                spec["fold_ordinal"],
                f"development OOF prediction fold model {ordinal} ordinal",
                minimum=1,
            )
            != ordinal
            or spec["fold_id"] != fold_id
            or spec["prediction_window_first_date"] != first
            or spec["prediction_window_last_date"] != last
            or (
                semantic_fit_spec is not None
                and semantic_fit_spec.get("training_view_id") != fold_id
            )
            or (
                semantic_fit_spec is not None
                and semantic_fit_spec.get("head_variant") != "semantic"
            )
            or (
                ablation_fit_spec is not None
                and ablation_fit_spec.get("training_view_id") != fold_id
            )
            or (
                ablation_fit_spec is not None
                and ablation_fit_spec.get("head_variant") != "ablation"
            )
            or spec["feature_schema_sha256"] != feature_schema_sha256
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF prediction fold model crossed its frozen fold, "
                "variant order, window, or feature schema"
            )
        if (
            spec["semantic_learner_state_sha256"]
            == spec["ablation_learner_state_sha256"]
            or spec["semantic_fit_record_sha256"]
            == spec["ablation_fit_record_sha256"]
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF prediction semantic and ablation states must remain distinct"
            )
        _self_hash(
            spec,
            "prediction_fold_model_spec_sha256",
            f"development OOF prediction fold model {ordinal}",
        )
        state_hashes.update(
            (
                spec["semantic_learner_state_sha256"],
                spec["ablation_learner_state_sha256"],
            )
        )
        fit_record_hashes.update(
            (
                spec["semantic_fit_record_sha256"],
                spec["ablation_fit_record_sha256"],
            )
        )
        fit_view_hashes.add(spec["source_learner_fit_view_sha256"])
        fold_model_hashes.add(spec["prediction_fold_model_sha256"])
        result.append(spec)
    if (
        len(state_hashes) != 2 * len(DEVELOPMENT_FOLD_SPECS)
        or len(fit_record_hashes) != 2 * len(DEVELOPMENT_FOLD_SPECS)
        or len(fit_view_hashes) != len(DEVELOPMENT_FOLD_SPECS)
        or len(fold_model_hashes) != len(DEVELOPMENT_FOLD_SPECS)
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF prediction fold models duplicate a fit view, record, state, or bundle"
        )
    return result, feature_schema_sha256


def _development_oof_prediction_fold_for_session(
    session: str,
) -> tuple[int, tuple[str, str, str, str]] | None:
    for ordinal, fold in enumerate(DEVELOPMENT_FOLD_SPECS, start=1):
        if fold[2] <= session <= fold[3]:
            return ordinal, fold
    return None


def _validated_development_oof_prediction_input_specs(
    raw: Any,
    *,
    source_feature_events: list[dict[str, Any]] | None = None,
    feature_schema_sha256: str,
) -> list[dict[str, Any]]:
    if type(raw) is not list:
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF prediction input specs must be an exact list"
        )
    supplied = _plain(raw, "development OOF prediction input specs")
    if type(supplied) is not list:
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF prediction input specs must remain a list"
        )
    expected_events = None
    if source_feature_events is not None:
        expected_events = [
            event
            for event in source_feature_events
            if _development_oof_prediction_fold_for_session(
                event["availability_session"]
            )
            is not None
        ]
    if (
        not supplied
        or expected_events is not None
        and len(supplied) != len(expected_events)
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF prediction inputs omit or add a frozen 2005-2018 event"
        )

    result: list[dict[str, Any]] = []
    decision_sessions: set[str] = set()
    accessions: set[str] = set()
    feature_row_hashes: set[str] = set()
    event_binding_hashes: set[str] = set()
    prediction_feature_hashes: set[str] = set()
    previous_decision: str | None = None
    previous_source_event_ordinal: int | None = None
    for prediction_ordinal, raw_spec in enumerate(supplied, start=1):
        event = (
            expected_events[prediction_ordinal - 1]
            if expected_events is not None
            else None
        )
        spec = _mapping(
            raw_spec,
            f"development OOF prediction input {prediction_ordinal}",
        )
        _expect_keys(
            spec,
            _DEVELOPMENT_OOF_PREDICTION_INPUT_SPEC_KEYS,
            f"development OOF prediction input {prediction_ordinal}",
        )
        decision = _validated_development_oof_iso_date(
            spec["decision_session"],
            f"development OOF prediction input {prediction_ordinal} decision",
        )
        fold_match = _development_oof_prediction_fold_for_session(decision)
        if fold_match is None:
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF prediction input lies outside 2005-2018"
            )
        fold_ordinal, frozen_fold = fold_match
        fold_id = frozen_fold[0]
        source_event_ordinal = _strict_int(
            spec["source_event_ordinal"],
            f"development OOF prediction input {prediction_ordinal} source event ordinal",
            minimum=1,
        )
        if (
            _strict_int(
                spec["prediction_ordinal"],
                f"development OOF prediction input {prediction_ordinal} ordinal",
                minimum=1,
            )
            != prediction_ordinal
            or (
                event is not None
                and source_event_ordinal != event["event_ordinal"]
            )
            or _strict_int(
                spec["fold_ordinal"],
                f"development OOF prediction input {prediction_ordinal} fold ordinal",
                minimum=1,
            )
            != fold_ordinal
            or spec["fold_id"] != fold_id
            or (
                event is not None
                and decision != event["availability_session"]
            )
            or (
                event is not None
                and spec["accession_number"] != event["accession_number"]
            )
            or _AAPL_ACCESSION_RE.fullmatch(spec["accession_number"] or "") is None
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF prediction input crossed its source event, fold, or chronology"
            )
        if (
            previous_source_event_ordinal is not None
            and source_event_ordinal <= previous_source_event_ordinal
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF prediction source event ordinals must be unique and strictly increasing"
            )
        previous_source_event_ordinal = source_event_ordinal
        if (
            previous_decision is not None
            and decision <= previous_decision
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF prediction decision sessions must be unique and strictly increasing"
            )
        previous_decision = decision
        if decision in decision_sessions or spec["accession_number"] in accessions:
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF prediction decision session or accession is duplicated"
            )
        decision_sessions.add(decision)
        accessions.add(spec["accession_number"])

        for field in (
            "source_feature_row_sha256",
            "event_binding_sha256",
            "semantic_feature_schema_sha256",
            "ablation_feature_schema_sha256",
            "prediction_feature_input_sha256",
        ):
            _sha256(
                spec[field],
                f"development OOF prediction input {prediction_ordinal} {field}",
            )
        if (
            spec["semantic_feature_schema_sha256"] != feature_schema_sha256
            or spec["ablation_feature_schema_sha256"] != feature_schema_sha256
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF prediction input feature schema crossed its fold states"
            )
        if type(spec["prediction_available"]) is not bool:
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF prediction availability must be an exact boolean"
            )
        semantic_values_hash = spec["semantic_feature_values_sha256"]
        ablation_values_hash = spec["ablation_feature_values_sha256"]
        if spec["prediction_available"]:
            if spec["unavailable_reason"] is not None:
                raise SecFilingGemmaStageAuthorizationError(
                    "Available development OOF prediction input has an unavailable reason"
                )
            _sha256(
                semantic_values_hash,
                f"development OOF prediction input {prediction_ordinal} semantic values hash",
            )
            _sha256(
                ablation_values_hash,
                f"development OOF prediction input {prediction_ordinal} ablation values hash",
            )
        elif (
            spec["unavailable_reason"]
            not in _DEVELOPMENT_OOF_PREDICTION_UNAVAILABLE_REASONS
            or semantic_values_hash is not None
            or ablation_values_hash is not None
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Unavailable development OOF prediction input changed its frozen reason or exposed a vector hash"
            )
        _self_hash(
            spec,
            "prediction_input_spec_sha256",
            f"development OOF prediction input {prediction_ordinal}",
        )
        feature_row_hashes.add(spec["source_feature_row_sha256"])
        event_binding_hashes.add(spec["event_binding_sha256"])
        prediction_feature_hashes.add(spec["prediction_feature_input_sha256"])
        result.append(spec)
    if (
        len(feature_row_hashes) != len(result)
        or len(event_binding_hashes) != len(result)
        or len(prediction_feature_hashes) != len(result)
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF prediction inputs duplicate a feature row, event binding, or compact input"
        )
    return result


def build_development_oof_prediction_plan(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    development_root_scope_sha256: str,
    source_development_oof_learner_fit_plan: Mapping[str, Any],
    source_development_oof_learner_fit_projection_sha256: str,
    source_development_oof_learner_fit_batch_sha256: str,
    prediction_fold_model_bundle_sha256: str,
    prediction_fold_model_specs: Any,
    source_feature_batch_sha256: str,
    prediction_feature_batch_sha256: str,
    prediction_input_specs: Any,
    independent_current_tip_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize only causal numeric OOF inference from five frozen state pairs."""

    root_scope_hash = _sha256(
        development_root_scope_sha256,
        "development OOF prediction root scope hash",
    )
    source_fit_plan = _mapping(
        source_development_oof_learner_fit_plan,
        "development OOF prediction source learner-fit plan",
    )
    source_fit_plan_hash = _sha256(
        source_fit_plan.get("development_oof_learner_fit_plan_sha256"),
        "development OOF prediction source learner-fit plan hash",
    )
    validate_development_oof_learner_fit_plan(
        source_fit_plan,
        expected_development_oof_learner_fit_plan_sha256=source_fit_plan_hash,
    )
    membership_plan = _mapping(
        source_fit_plan["source_training_membership_assembly_plan"],
        "development OOF prediction source membership plan",
    )
    rebuilt_fit_plan = build_development_oof_learner_fit_plan(
        authenticated_store_snapshot,
        development_root_scope_sha256=root_scope_hash,
        source_training_membership_assembly_plan=membership_plan,
        source_training_membership_projection_sha256=source_fit_plan[
            "source_training_membership_projection_sha256"
        ],
        source_training_membership_batch_sha256=source_fit_plan[
            "source_training_membership_batch_sha256"
        ],
        fit_input_specs=source_fit_plan["learner_fit_input_specs"],
        independent_current_tip_anchor=independent_current_tip_anchor,
    )
    if source_fit_plan != rebuilt_fit_plan:
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF prediction source learner-fit plan differs from the terminal store replay"
        )
    label_plan = _mapping(
        membership_plan["source_label_assembly_plan"],
        "development OOF prediction source label plan",
    )
    feature_plan = _mapping(
        label_plan["source_feature_assembly_plan"],
        "development OOF prediction source feature plan",
    )
    source_feature_events = _validated_development_feature_event_plan(
        feature_plan["event_plan"]
    )
    fold_specs, feature_schema_sha256 = (
        _validated_development_oof_prediction_fold_model_specs(
            prediction_fold_model_specs,
            source_fit_plan=source_fit_plan,
        )
    )
    input_specs = _validated_development_oof_prediction_input_specs(
        prediction_input_specs,
        source_feature_events=source_feature_events,
        feature_schema_sha256=feature_schema_sha256,
    )
    available_count = sum(
        1 for spec in input_specs if spec["prediction_available"]
    )
    unavailable_count = len(input_specs) - available_count
    capabilities = _expected_development_oof_prediction_capabilities()
    body = {
        "schema_version": DEVELOPMENT_OOF_PREDICTION_PLAN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "plan_kind": "request_free_development_oof_prediction",
        "artifact_stage": "development",
        "development_root_scope_sha256": root_scope_hash,
        "start_consumed_request_count": source_fit_plan[
            "start_consumed_request_count"
        ],
        "source_development_oof_learner_fit_plan_sha256": source_fit_plan_hash,
        "source_development_oof_learner_fit_projection_sha256": _sha256(
            source_development_oof_learner_fit_projection_sha256,
            "development OOF prediction source learner-fit projection hash",
        ),
        "source_development_oof_learner_fit_batch_sha256": _sha256(
            source_development_oof_learner_fit_batch_sha256,
            "development OOF prediction source learner-fit batch hash",
        ),
        "source_training_membership_assembly_plan_sha256": source_fit_plan[
            "source_training_membership_assembly_plan_sha256"
        ],
        "source_training_membership_projection_sha256": source_fit_plan[
            "source_training_membership_projection_sha256"
        ],
        "source_training_membership_batch_sha256": source_fit_plan[
            "source_training_membership_batch_sha256"
        ],
        "source_feature_assembly_plan_sha256": feature_plan[
            "feature_assembly_plan_sha256"
        ],
        "source_feature_batch_sha256": _sha256(
            source_feature_batch_sha256,
            "development OOF prediction source feature batch hash",
        ),
        "prediction_feature_batch_sha256": _sha256(
            prediction_feature_batch_sha256,
            "development OOF prediction compact feature batch hash",
        ),
        "candidate_sha256": source_fit_plan["candidate_sha256"],
        "corpus_universe_sha256": source_fit_plan["corpus_universe_sha256"],
        "calendar_sessions_sha256": source_fit_plan["calendar_sessions_sha256"],
        "development_cutoff_session": source_fit_plan[
            "development_cutoff_session"
        ],
        "source_event_count": len(source_feature_events),
        "authorized_fold_count": len(DEVELOPMENT_FOLD_SPECS),
        "authorized_fold_ids": [fold[0] for fold in DEVELOPMENT_FOLD_SPECS],
        "prediction_fold_model_count": len(fold_specs),
        "prediction_fold_model_bundle_sha256": _sha256(
            prediction_fold_model_bundle_sha256,
            "development OOF prediction fold-model bundle hash",
        ),
        "prediction_fold_model_specs": fold_specs,
        "prediction_fold_model_specs_sha256": canonical_sha256(fold_specs),
        "model_variant_count": len(_DEVELOPMENT_OOF_MODEL_VARIANT_IDS),
        "model_variant_ids": list(_DEVELOPMENT_OOF_MODEL_VARIANT_IDS),
        "learner_state_count": 2 * len(DEVELOPMENT_FOLD_SPECS),
        "learner_model_type": LEARNER_MODEL_TYPE,
        "learner_state_schema_version": LEARNER_STATE_SCHEMA_VERSION,
        "learner_config_sha256": source_fit_plan["learner_config_sha256"],
        "feature_schema_sha256": feature_schema_sha256,
        "prediction_input_count": len(input_specs),
        "available_prediction_input_count": available_count,
        "unavailable_prediction_input_count": unavailable_count,
        "prediction_input_specs": input_specs,
        "prediction_input_specs_sha256": canonical_sha256(input_specs),
        "prediction_population_rule": (
            _DEVELOPMENT_OOF_PREDICTION_POPULATION_RULE
        ),
        "prediction_input_order_rule": (
            _DEVELOPMENT_OOF_PREDICTION_INPUT_ORDER_RULE
        ),
        "fold_state_usage_rule": (
            _DEVELOPMENT_OOF_PREDICTION_FOLD_STATE_USAGE_RULE
        ),
        "maximum_prediction_calls": 2 * available_count,
        "maximum_prediction_seconds": (
            _DEVELOPMENT_OOF_MAXIMUM_PREDICTION_SECONDS
        ),
        **capabilities,
    }
    return {
        **body,
        "development_oof_prediction_plan_sha256": canonical_sha256(body),
    }


def validate_development_oof_prediction_plan(
    plan: Mapping[str, Any],
    *,
    expected_development_oof_prediction_plan_sha256: str,
) -> str:
    """Validate the exact feature-only, state-frozen development OOF authority."""

    value = _mapping(plan, "development OOF prediction plan")
    _expect_keys(
        value,
        _DEVELOPMENT_OOF_PREDICTION_PLAN_KEYS,
        "development OOF prediction plan",
    )
    _sha256(
        value["source_development_oof_learner_fit_plan_sha256"],
        "development OOF prediction source learner-fit plan hash",
    )
    fold_specs, feature_schema_sha256 = (
        _validated_development_oof_prediction_fold_model_specs(
            value["prediction_fold_model_specs"],
        )
    )
    input_specs = _validated_development_oof_prediction_input_specs(
        value["prediction_input_specs"],
        feature_schema_sha256=feature_schema_sha256,
    )
    capabilities = _expected_development_oof_prediction_capabilities()
    if any(type(value[field]) is not bool for field in capabilities):
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF prediction capabilities must be exact booleans"
        )
    integer_fields = {
        "start_consumed_request_count": 0,
        "authorized_fold_count": len(DEVELOPMENT_FOLD_SPECS),
        "prediction_fold_model_count": len(fold_specs),
        "model_variant_count": len(_DEVELOPMENT_OOF_MODEL_VARIANT_IDS),
        "learner_state_count": 2 * len(DEVELOPMENT_FOLD_SPECS),
        "learner_state_schema_version": LEARNER_STATE_SCHEMA_VERSION,
        "prediction_input_count": len(input_specs),
        "available_prediction_input_count": sum(
            1 for spec in input_specs if spec["prediction_available"]
        ),
        "unavailable_prediction_input_count": sum(
            1 for spec in input_specs if not spec["prediction_available"]
        ),
        "maximum_prediction_calls": 2
        * sum(1 for spec in input_specs if spec["prediction_available"]),
        "maximum_prediction_seconds": (
            _DEVELOPMENT_OOF_MAXIMUM_PREDICTION_SECONDS
        ),
    }
    _strict_int(
        value["source_event_count"],
        "development OOF prediction source_event_count",
        minimum=len(input_specs),
    )
    for field, expected_value in integer_fields.items():
        if (
            _strict_int(value[field], f"development OOF prediction {field}")
            != expected_value
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development OOF prediction count or runtime boundary changed"
            )
    for field in (
        "contract_sha256",
        "development_root_scope_sha256",
        "source_development_oof_learner_fit_projection_sha256",
        "source_development_oof_learner_fit_batch_sha256",
        "source_training_membership_assembly_plan_sha256",
        "source_training_membership_projection_sha256",
        "source_training_membership_batch_sha256",
        "source_feature_assembly_plan_sha256",
        "source_feature_batch_sha256",
        "prediction_feature_batch_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "prediction_fold_model_bundle_sha256",
        "prediction_fold_model_specs_sha256",
        "learner_config_sha256",
        "feature_schema_sha256",
        "prediction_input_specs_sha256",
    ):
        _sha256(value[field], f"development OOF prediction {field}")
    expected_fold_ids = [fold[0] for fold in DEVELOPMENT_FOLD_SPECS]
    if (
        value["schema_version"]
        != DEVELOPMENT_OOF_PREDICTION_PLAN_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["contract_sha256"] != canonical_sha256(build_contract_manifest())
        or value["plan_kind"] != "request_free_development_oof_prediction"
        or value["artifact_stage"] != "development"
        or value["development_cutoff_session"]
        != STAGE_WINDOWS["development"][1]
        or type(value["authorized_fold_ids"]) is not list
        or value["authorized_fold_ids"] != expected_fold_ids
        or value["prediction_fold_model_specs_sha256"]
        != canonical_sha256(fold_specs)
        or type(value["model_variant_ids"]) is not list
        or value["model_variant_ids"]
        != list(_DEVELOPMENT_OOF_MODEL_VARIANT_IDS)
        or value["learner_model_type"] != LEARNER_MODEL_TYPE
        or value["feature_schema_sha256"] != feature_schema_sha256
        or value["prediction_input_specs_sha256"]
        != canonical_sha256(input_specs)
        or value["prediction_population_rule"]
        != _DEVELOPMENT_OOF_PREDICTION_POPULATION_RULE
        or value["prediction_input_order_rule"]
        != _DEVELOPMENT_OOF_PREDICTION_INPUT_ORDER_RULE
        or value["fold_state_usage_rule"]
        != _DEVELOPMENT_OOF_PREDICTION_FOLD_STATE_USAGE_RULE
        or any(
            value[field] is not expected
            for field, expected in capabilities.items()
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF prediction identity, population, order, or capability boundary changed"
        )
    observed = _self_hash(
        value,
        "development_oof_prediction_plan_sha256",
        "development OOF prediction plan",
    )
    expected = _sha256(
        expected_development_oof_prediction_plan_sha256,
        "expected development OOF prediction plan hash",
    )
    if not hmac.compare_digest(observed, expected):
        raise SecFilingGemmaStageAuthorizationError(
            "Development OOF prediction plan is not externally pinned"
        )
    return observed


def build_development_policy_replay_plan(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    development_root_scope_sha256: str,
    authenticated_store_state_bytes_sha256: str,
    source_development_oof_prediction_plan: Mapping[str, Any],
    source_development_oof_prediction_projection_sha256: str,
    source_development_oof_prediction_batch_sha256: str,
    source_raw_prediction_rows_sha256: str,
    source_raw_prediction_tip_sha256: str,
    source_raw_prediction_row_count: int,
    policy_replay_input_specs: Any,
    independent_current_tip_anchor: Mapping[str, Any],
    independent_current_tip_anchor_bytes_sha256: str,
) -> dict[str, Any]:
    """Authorize only deterministic development threshold and policy replay."""

    state, ledger = _validated_store_snapshot(authenticated_store_snapshot)
    current_tip = validate_reveal_store_current_tip_anchor(
        state,
        independent_current_tip_anchor,
    )
    state_bytes_hash = _sha256(
        authenticated_store_state_bytes_sha256,
        "development policy replay authenticated state bytes hash",
    )
    tip_bytes_hash = _sha256(
        independent_current_tip_anchor_bytes_sha256,
        "development policy replay current-tip bytes hash",
    )
    expected_state_bytes_hash = hashlib.sha256(
        _encoded_store_snapshot(state)
    ).hexdigest()
    expected_tip_bytes_hash = hashlib.sha256(
        _encoded_store_snapshot(current_tip)
    ).hexdigest()
    if (
        not hmac.compare_digest(state_bytes_hash, expected_state_bytes_hash)
        or not hmac.compare_digest(tip_bytes_hash, expected_tip_bytes_hash)
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy replay byte pins crossed their parsed state or tip"
        )

    consumed_count = _strict_int(
        ledger["chain"]["consumed_request_count"],
        "development policy replay consumed request count",
    )
    tip_consumed_count = _strict_int(
        current_tip["consumed_request_count"],
        "development policy replay current-tip consumed request count",
    )
    if consumed_count != 0 or tip_consumed_count != 0:
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy replay requires the untouched zero-consumption store"
        )

    root_scope_hash = _sha256(
        development_root_scope_sha256,
        "development policy replay root scope hash",
    )
    source_plan = _mapping(
        source_development_oof_prediction_plan,
        "development policy replay source prediction plan",
    )
    source_plan_hash = _sha256(
        source_plan.get("development_oof_prediction_plan_sha256"),
        "development policy replay source prediction plan hash",
    )
    validate_development_oof_prediction_plan(
        source_plan,
        expected_development_oof_prediction_plan_sha256=source_plan_hash,
    )
    if (
        source_plan["development_root_scope_sha256"] != root_scope_hash
        or source_plan["start_consumed_request_count"] != 0
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy replay source prediction plan crossed its root or store phase"
        )

    input_specs = _validated_development_policy_replay_input_specs(
        policy_replay_input_specs
    )
    raw_row_count = _strict_int(
        source_raw_prediction_row_count,
        "development policy replay source raw prediction row count",
        minimum=1,
    )
    if raw_row_count != len(input_specs):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy replay input population differs from its raw prediction population"
        )
    candidate_specs = _development_policy_replay_candidate_threshold_specs()
    capabilities = _expected_development_policy_replay_capabilities()
    body = {
        "schema_version": DEVELOPMENT_POLICY_REPLAY_PLAN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "plan_kind": "request_free_development_policy_replay",
        "artifact_stage": "development",
        "development_root_scope_sha256": root_scope_hash,
        "start_store_state_bytes_sha256": state_bytes_hash,
        "start_store_state_sha256": state["state_sha256"],
        "start_current_tip_anchor_bytes_sha256": tip_bytes_hash,
        "start_current_tip_anchor_sha256": current_tip["tip_anchor_sha256"],
        "start_current_tip_revision": current_tip["revision"],
        "start_consumed_request_count": consumed_count,
        "source_development_oof_prediction_plan_sha256": source_plan_hash,
        "source_development_oof_prediction_projection_sha256": _sha256(
            source_development_oof_prediction_projection_sha256,
            "development policy replay source prediction projection hash",
        ),
        "source_development_oof_prediction_batch_sha256": _sha256(
            source_development_oof_prediction_batch_sha256,
            "development policy replay source prediction batch hash",
        ),
        "source_raw_prediction_rows_sha256": _sha256(
            source_raw_prediction_rows_sha256,
            "development policy replay source raw prediction rows hash",
        ),
        "source_raw_prediction_tip_sha256": _sha256(
            source_raw_prediction_tip_sha256,
            "development policy replay source raw prediction tip hash",
        ),
        "source_raw_prediction_row_count": raw_row_count,
        "policy_replay_input_count": len(input_specs),
        "policy_replay_input_specs": input_specs,
        "policy_replay_input_specs_sha256": canonical_sha256(input_specs),
        "candidate_sha256": source_plan["candidate_sha256"],
        "corpus_universe_sha256": source_plan["corpus_universe_sha256"],
        "calendar_sessions_sha256": source_plan["calendar_sessions_sha256"],
        "development_cutoff_session": source_plan[
            "development_cutoff_session"
        ],
        "candidate_count": len(CANDIDATE_IDS),
        "candidate_ids": list(CANDIDATE_IDS),
        "candidate_threshold_specs": candidate_specs,
        "candidate_threshold_specs_sha256": canonical_sha256(candidate_specs),
        "model_variant_count": len(
            _DEVELOPMENT_POLICY_REPLAY_MODEL_VARIANT_IDS
        ),
        "model_variant_ids": list(
            _DEVELOPMENT_POLICY_REPLAY_MODEL_VARIANT_IDS
        ),
        "candidate_gate_comparison_rule": (
            _DEVELOPMENT_POLICY_REPLAY_GATE_COMPARISON_RULE
        ),
        "cash_episode_sessions": HORIZON_SESSIONS,
        "cash_episode_rule": _DEVELOPMENT_POLICY_REPLAY_CASH_EPISODE_RULE,
        "unavailable_prediction_rule": (
            _DEVELOPMENT_POLICY_REPLAY_UNAVAILABLE_RULE
        ),
        "policy_replay_order_rule": _DEVELOPMENT_POLICY_REPLAY_ORDER_RULE,
        **capabilities,
    }
    return {
        **body,
        "development_policy_replay_plan_sha256": canonical_sha256(body),
    }


def validate_development_policy_replay_plan(
    plan: Mapping[str, Any],
    *,
    expected_development_policy_replay_plan_sha256: str,
) -> str:
    """Validate one exact Phase A threshold/policy-only replay authority."""

    value = _mapping(plan, "development policy replay plan")
    _expect_keys(
        value,
        _DEVELOPMENT_POLICY_REPLAY_PLAN_KEYS,
        "development policy replay plan",
    )
    input_specs = _validated_development_policy_replay_input_specs(
        value["policy_replay_input_specs"]
    )
    candidate_specs = _development_policy_replay_candidate_threshold_specs()
    capabilities = _expected_development_policy_replay_capabilities()
    if any(type(value[field]) is not bool for field in capabilities):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy replay capabilities must be exact booleans"
        )
    for field in (
        "contract_sha256",
        "development_root_scope_sha256",
        "start_store_state_bytes_sha256",
        "start_store_state_sha256",
        "start_current_tip_anchor_bytes_sha256",
        "start_current_tip_anchor_sha256",
        "source_development_oof_prediction_plan_sha256",
        "source_development_oof_prediction_projection_sha256",
        "source_development_oof_prediction_batch_sha256",
        "source_raw_prediction_rows_sha256",
        "source_raw_prediction_tip_sha256",
        "policy_replay_input_specs_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "candidate_threshold_specs_sha256",
    ):
        _sha256(value[field], f"development policy replay {field}")
    integer_expectations = {
        "start_consumed_request_count": 0,
        "source_raw_prediction_row_count": len(input_specs),
        "policy_replay_input_count": len(input_specs),
        "candidate_count": len(CANDIDATE_IDS),
        "model_variant_count": len(
            _DEVELOPMENT_POLICY_REPLAY_MODEL_VARIANT_IDS
        ),
        "cash_episode_sessions": HORIZON_SESSIONS,
    }
    _strict_int(
        value["start_current_tip_revision"],
        "development policy replay start current-tip revision",
    )
    for field, expected in integer_expectations.items():
        if (
            _strict_int(value[field], f"development policy replay {field}")
            != expected
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development policy replay count or episode boundary changed"
            )
    if (
        value["schema_version"]
        != DEVELOPMENT_POLICY_REPLAY_PLAN_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["contract_sha256"]
        != canonical_sha256(build_contract_manifest())
        or value["plan_kind"]
        != "request_free_development_policy_replay"
        or value["artifact_stage"] != "development"
        or value["development_cutoff_session"]
        != STAGE_WINDOWS["development"][1]
        or type(value["candidate_ids"]) is not list
        or value["candidate_ids"] != list(CANDIDATE_IDS)
        or type(value["candidate_threshold_specs"]) is not list
        or value["candidate_threshold_specs"] != candidate_specs
        or value["candidate_threshold_specs_sha256"]
        != canonical_sha256(candidate_specs)
        or type(value["model_variant_ids"]) is not list
        or value["model_variant_ids"]
        != list(_DEVELOPMENT_POLICY_REPLAY_MODEL_VARIANT_IDS)
        or value["policy_replay_input_specs_sha256"]
        != canonical_sha256(input_specs)
        or value["candidate_gate_comparison_rule"]
        != _DEVELOPMENT_POLICY_REPLAY_GATE_COMPARISON_RULE
        or value["cash_episode_rule"]
        != _DEVELOPMENT_POLICY_REPLAY_CASH_EPISODE_RULE
        or value["unavailable_prediction_rule"]
        != _DEVELOPMENT_POLICY_REPLAY_UNAVAILABLE_RULE
        or value["policy_replay_order_rule"]
        != _DEVELOPMENT_POLICY_REPLAY_ORDER_RULE
        or any(
            value[field] is not expected
            for field, expected in capabilities.items()
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy replay identity, order, episode, or capability boundary changed"
        )
    observed = _self_hash(
        value,
        "development_policy_replay_plan_sha256",
        "development policy replay plan",
    )
    expected = _sha256(
        expected_development_policy_replay_plan_sha256,
        "expected development policy replay plan hash",
    )
    if not hmac.compare_digest(observed, expected):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy replay plan is not externally pinned"
        )
    return observed


def _canonical_compact_json_bytes(value: Any, location: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must be finite canonical JSON"
        ) from exc


def _development_policy_prediction_genesis(
    *,
    contract_sha256: str,
    candidate_sha256: str,
    corpus_universe_sha256: str,
    calendar_sessions_sha256: str,
    initial_event_sequence_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "domain": "aapl-sec-gemma-prediction-genesis-v3",
            "contract_sha256": contract_sha256,
            "candidate_sha256": candidate_sha256,
            "corpus_universe_sha256": corpus_universe_sha256,
            "calendar_sessions_sha256": calendar_sessions_sha256,
            "initial_event_sequence_sha256": initial_event_sequence_sha256,
        }
    )


def _development_policy_prefix_body(
    *,
    prefix: Mapping[str, Any],
    rows: list[dict[str, Any]],
    event_sequence_sha256: str,
    parent_prefix_sha256: str | None,
    parent_tip_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": PREDICTION_PREFIX_SCHEMA_VERSION,
        "contract_sha256": prefix["contract_sha256"],
        "candidate_sha256": prefix["candidate_sha256"],
        "corpus_universe_sha256": prefix["corpus_universe_sha256"],
        "calendar_sessions_sha256": prefix["calendar_sessions_sha256"],
        "initial_event_sequence_sha256": prefix[
            "initial_event_sequence_sha256"
        ],
        "event_sequence_sha256": event_sequence_sha256,
        "row_count": len(rows),
        "genesis_sha256": prefix["genesis_sha256"],
        "parent_prefix_sha256": parent_prefix_sha256,
        "parent_tip_sha256": parent_tip_sha256,
        "appended_row_sha256": rows[-1]["prediction_row_sha256"],
        "tip_sha256": rows[-1]["prediction_row_sha256"],
        "rows_sha256": canonical_sha256(rows),
        "rows": rows,
    }


def _validated_development_policy_prefix_seal_input_specs(
    raw: Any,
) -> list[dict[str, Any]]:
    if type(raw) is not list or not raw:
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy-prefix seal inputs must be a nonempty exact list"
        )
    specs: list[dict[str, Any]] = []
    prior_prefix: str | None = None
    prior_tip: str | None = None
    prior_key: tuple[str, str] | None = None
    observed_rows: set[str] = set()
    observed_prefixes: set[str] = set()
    observed_artifacts: set[str] = set()
    for ordinal, raw_spec in enumerate(raw, start=1):
        spec = _mapping(
            raw_spec, f"development policy-prefix seal input {ordinal}"
        )
        _expect_keys(
            spec,
            _DEVELOPMENT_POLICY_PREFIX_SEAL_INPUT_SPEC_KEYS,
            f"development policy-prefix seal input {ordinal}",
        )
        if (
            spec["schema_version"]
            != _DEVELOPMENT_POLICY_PREFIX_SEAL_INPUT_SPEC_SCHEMA_VERSION
            or _strict_int(
                spec["seal_ordinal"],
                f"development policy-prefix seal input {ordinal} ordinal",
                minimum=1,
            )
            != ordinal
            or _strict_int(
                spec["policy_sequence_number"],
                f"development policy-prefix seal input {ordinal} sequence",
                minimum=1,
            )
            != ordinal
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Development policy-prefix seal input order changed"
            )
        decision = _validated_development_oof_iso_date(
            spec["decision_session"],
            f"development policy-prefix seal input {ordinal} decision session",
        )
        accession = spec["accession_number"]
        if type(accession) is not str or _AAPL_ACCESSION_RE.fullmatch(accession) is None:
            raise SecFilingGemmaStageAuthorizationError(
                "Development policy-prefix seal input accession is invalid"
            )
        key = (decision, accession)
        if prior_key is not None and key <= prior_key:
            raise SecFilingGemmaStageAuthorizationError(
                "Development policy-prefix seal inputs are reordered or duplicated"
            )
        prior_key = key
        row_hash = _sha256(
            spec["prediction_row_sha256"],
            f"development policy-prefix seal input {ordinal} row",
        )
        parent_prefix = spec["parent_prediction_prefix_sha256"]
        if ordinal == 1:
            if parent_prefix is not None:
                raise SecFilingGemmaStageAuthorizationError(
                    "First development policy-prefix seal input cannot have a parent prefix"
                )
        else:
            parent_prefix = _sha256(
                parent_prefix,
                f"development policy-prefix seal input {ordinal} parent prefix",
            )
            if parent_prefix != prior_prefix:
                raise SecFilingGemmaStageAuthorizationError(
                    "Development policy-prefix seal input crossed its parent prefix"
                )
        parent_tip = _sha256(
            spec["parent_prediction_tip_sha256"],
            f"development policy-prefix seal input {ordinal} parent tip",
        )
        if ordinal > 1 and parent_tip != prior_tip:
            raise SecFilingGemmaStageAuthorizationError(
                "Development policy-prefix seal input crossed its parent tip"
            )
        prefix_hash = _sha256(
            spec["prediction_prefix_sha256"],
            f"development policy-prefix seal input {ordinal} prefix",
        )
        tip_hash = _sha256(
            spec["prediction_tip_sha256"],
            f"development policy-prefix seal input {ordinal} tip",
        )
        rows_hash = _sha256(
            spec["prediction_rows_sha256"],
            f"development policy-prefix seal input {ordinal} rows",
        )
        artifact_hash = _sha256(
            spec["artifact_sha256"],
            f"development policy-prefix seal input {ordinal} artifact",
        )
        if tip_hash != row_hash:
            raise SecFilingGemmaStageAuthorizationError(
                "Development policy-prefix seal input tip differs from its appended row"
            )
        _strict_int(
            spec["artifact_size_bytes"],
            f"development policy-prefix seal input {ordinal} artifact size",
            minimum=1,
        )
        _self_hash(
            spec,
            "policy_prefix_seal_input_spec_sha256",
            f"development policy-prefix seal input {ordinal}",
        )
        for identity, observed, label in (
            (row_hash, observed_rows, "row"),
            (prefix_hash, observed_prefixes, "prefix"),
            (artifact_hash, observed_artifacts, "artifact"),
        ):
            if identity in observed:
                raise SecFilingGemmaStageAuthorizationError(
                    f"Development policy-prefix seal inputs duplicate a {label}"
                )
            observed.add(identity)
        prior_prefix = prefix_hash
        prior_tip = tip_hash
        specs.append(spec)
    return specs


def _validated_development_policy_artifact_genesis_pin(
    raw: Any,
    *,
    expected_candidate_sha256: str,
    expected_pin_bytes_sha256: str,
) -> dict[str, Any]:
    pin = _mapping(raw, "development policy artifact-sealer genesis pin")
    _expect_keys(
        pin,
        _DEVELOPMENT_POLICY_ARTIFACT_GENESIS_PIN_KEYS,
        "development policy artifact-sealer genesis pin",
    )
    candidate = _sha256(
        expected_candidate_sha256,
        "development policy artifact-sealer expected candidate",
    )
    seal_tip = canonical_sha256(
        {
            "domain": "aapl-sec-gemma-prediction-artifact-seal-genesis-v1",
            "candidate_sha256": candidate,
            "stage": "development",
        }
    )
    body = {
        "schema_version": _PREDICTION_ARTIFACT_EXTERNAL_PIN_SCHEMA_VERSION,
        "candidate_sha256": candidate,
        "stage": "development",
        "sealed_artifact_count": 0,
        "last_prediction_sequence_number": 0,
        "prediction_prefix_sha256": None,
        "prediction_tip_sha256": None,
        "prediction_rows_sha256": None,
        "artifact_sha256": None,
        "seal_tip_sha256": seal_tip,
    }
    expected = {
        **body,
        "external_pin_sha256": canonical_sha256(body),
    }
    _strict_int(pin["sealed_artifact_count"], "genesis sealed artifact count")
    _strict_int(
        pin["last_prediction_sequence_number"],
        "genesis prediction sequence number",
    )
    if pin != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy artifact-sealer genesis pin changed identity"
        )
    encoded = _canonical_compact_json_bytes(pin, "artifact-sealer genesis pin")
    expected_bytes_hash = _sha256(
        expected_pin_bytes_sha256,
        "development policy artifact-sealer genesis pin bytes hash",
    )
    if not hmac.compare_digest(hashlib.sha256(encoded).hexdigest(), expected_bytes_hash):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy artifact-sealer genesis pin bytes changed"
        )
    return pin


def _validated_source_development_policy_replay_batch(
    raw: Any,
    *,
    source_policy_replay_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    batch = _mapping(raw, "source development policy replay batch")
    _expect_keys(
        batch,
        _DEVELOPMENT_POLICY_REPLAY_BATCH_KEYS,
        "source development policy replay batch",
    )
    if (
        batch["schema_version"]
        != _OWNED_DEVELOPMENT_POLICY_REPLAY_BATCH_SCHEMA_VERSION
        or batch["artifact_stage"] != "development"
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Source development policy replay batch schema or stage changed"
        )
    batch_hash = _self_hash(
        batch,
        "policy_replay_batch_sha256",
        "source development policy replay batch",
    )
    plan = source_policy_replay_plan
    expected_policy_calendar = development_policy_session_calendar_sha256(
        DEVELOPMENT_POLICY_SESSION_DATES
    )
    cross_bindings = {
        "development_root_scope_sha256": plan["development_root_scope_sha256"],
        "development_policy_replay_plan_sha256": plan[
            "development_policy_replay_plan_sha256"
        ],
        "source_prediction_projection_sha256": plan[
            "source_development_oof_prediction_projection_sha256"
        ],
        "source_prediction_batch_sha256": plan[
            "source_development_oof_prediction_batch_sha256"
        ],
        "source_raw_prediction_rows_sha256": plan[
            "source_raw_prediction_rows_sha256"
        ],
        "source_raw_prediction_tip_sha256": plan[
            "source_raw_prediction_tip_sha256"
        ],
        "contract_sha256": plan["contract_sha256"],
        "candidate_sha256": plan["candidate_sha256"],
        "corpus_universe_sha256": plan["corpus_universe_sha256"],
        "source_market_calendar_sessions_sha256": plan["calendar_sessions_sha256"],
        "policy_calendar_sessions_sha256": expected_policy_calendar,
        "development_cutoff_session": plan["development_cutoff_session"],
        "source_prediction_event_count": plan["source_raw_prediction_row_count"],
        "candidate_count": plan["candidate_count"],
        "candidate_ids": plan["candidate_ids"],
        "model_variant_count": plan["model_variant_count"],
        "model_variant_ids": plan["model_variant_ids"],
        "threshold_comparison_rule": plan["candidate_gate_comparison_rule"],
        "cash_episode_rule": plan["cash_episode_rule"],
        "unavailable_prediction_rule": plan["unavailable_prediction_rule"],
        "policy_replay_order_rule": plan["policy_replay_order_rule"],
        "policy_replay_input_count": plan["policy_replay_input_count"],
        "policy_replay_input_specs_sha256": plan[
            "policy_replay_input_specs_sha256"
        ],
    }
    if any(batch.get(field) != expected for field, expected in cross_bindings.items()):
        raise SecFilingGemmaStageAuthorizationError(
            "Source development policy replay batch crossed its plan ancestry"
        )
    event_count = _strict_int(
        batch["source_prediction_event_count"],
        "source development policy replay event count",
        minimum=1,
    )
    available = _strict_int(
        batch["available_prediction_count"],
        "source development policy replay available count",
    )
    unavailable = _strict_int(
        batch["unavailable_prediction_count"],
        "source development policy replay unavailable count",
    )
    bindings = batch["raw_to_policy_bindings"]
    if (
        type(bindings) is not list
        or available + unavailable != event_count
        or _strict_int(batch["binding_count"], "policy replay binding count")
        != event_count
        or len(bindings) != event_count
        or batch["raw_to_policy_bindings_sha256"] != canonical_sha256(bindings)
        or batch["candidate_grid_sha256"]
        != canonical_sha256(batch["candidate_grid"])
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Source development policy replay population or binding chain changed"
        )
    true_flags = (
        "prediction_components_equal_source",
        "policy_prefix_included",
        "raw_to_policy_bindings_included",
        "threshold_evaluation_authorized",
        "policy_transition_authorized",
        "calendar_schedule_after_cutoff_used_only_for_open_episode_dates",
    )
    false_flags = (
        "source_feature_rows_included",
        "learner_states_included",
        "labels_included",
        "outcomes_included",
        "market_prices_included",
        "post_2018_prediction_rows_included",
        "post_2018_market_or_outcome_data_included",
        "numeric_prediction_authorized",
        "learner_fit_authorized",
        "model_transport_authorized",
        "network_access_authorized",
        "scoring_authorized",
        "candidate_selection_authorized",
        "ranking_authorized",
        "sealing_authorized",
        "refit_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
    )
    if any(type(batch[field]) is not bool or batch[field] is not True for field in true_flags):
        raise SecFilingGemmaStageAuthorizationError(
            "Source development policy replay lost an outcome-free replay proof"
        )
    if any(type(batch[field]) is not bool or batch[field] is not False for field in false_flags):
        raise SecFilingGemmaStageAuthorizationError(
            "Source development policy replay crossed into sealing or outcome authority"
        )

    prefix = _mapping(batch["policy_prefix"], "source development policy prefix")
    _expect_keys(prefix, _DEVELOPMENT_POLICY_PREFIX_KEYS, "source development policy prefix")
    if (
        prefix["schema_version"] != PREDICTION_PREFIX_SCHEMA_VERSION
        or prefix["contract_sha256"] != batch["contract_sha256"]
        or prefix["candidate_sha256"] != batch["candidate_sha256"]
        or prefix["corpus_universe_sha256"] != batch["corpus_universe_sha256"]
        or prefix["calendar_sessions_sha256"] != expected_policy_calendar
        or batch["policy_prefix_sha256"] != prefix["prediction_prefix_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Source development policy prefix crossed its batch identity"
        )
    rows_raw = prefix["rows"]
    if type(rows_raw) is not list or len(rows_raw) != event_count:
        raise SecFilingGemmaStageAuthorizationError(
            "Source development policy prefix omits or adds a policy row"
        )
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    sessions = DEVELOPMENT_POLICY_SESSION_DATES
    session_index = {session: index for index, session in enumerate(sessions)}
    prior_prefix_hash: str | None = None
    prior_tip_hash: str | None = None
    previous_key: tuple[str, str] | None = None
    seal_specs: list[dict[str, Any]] = []
    for ordinal, raw_row in enumerate(rows_raw, start=1):
        row = _mapping(raw_row, f"source development policy row {ordinal}")
        _expect_keys(
            row,
            frozenset(_AUTHORITATIVE_PREDICTION_ROW_KEYS),
            f"source development policy row {ordinal}",
        )
        if (
            row["schema_version"] != PREDICTION_ROW_SCHEMA_VERSION
            or _strict_int(
                row["sequence_number"],
                f"source development policy row {ordinal} sequence",
                minimum=1,
            )
            != ordinal
            or row["stage"] != "development"
            or row["contract_sha256"] != prefix["contract_sha256"]
            or row["candidate_sha256"] != prefix["candidate_sha256"]
            or row["corpus_universe_sha256"] != prefix["corpus_universe_sha256"]
            or row["calendar_sessions_sha256"] != expected_policy_calendar
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Source development policy row identity or order changed"
            )
        decision = _validated_development_oof_iso_date(
            row["decision_session"],
            f"source development policy row {ordinal} decision session",
        )
        accession = row["accession_number"]
        if (
            type(accession) is not str
            or _AAPL_ACCESSION_RE.fullmatch(accession) is None
            or decision > STAGE_WINDOWS["development"][1]
            or decision not in session_index
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Source development policy row is outside the bounded population"
            )
        key = (decision, accession)
        if previous_key is not None and key <= previous_key:
            raise SecFilingGemmaStageAuthorizationError(
                "Source development policy rows are duplicated or reordered"
            )
        previous_key = key
        decision_index = session_index[decision]
        if decision_index + LABEL_MATURITY_OFFSET >= len(sessions):
            raise SecFilingGemmaStageAuthorizationError(
                "Development policy calendar does not cover a fixed episode"
            )
        if (
            row["market_feature_cutoff_session"] != decision
            or row["fill_session"] != sessions[decision_index + 1]
            or row["cash_exit_session"]
            != sessions[decision_index + LABEL_MATURITY_OFFSET]
            or row["label_maturity_session"]
            != sessions[decision_index + LABEL_MATURITY_OFFSET]
            or row["horizon_sessions"] != HORIZON_SESSIONS
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Source development policy row crossed its causal episode schedule"
            )
        event = {field: row[field] for field in _DEVELOPMENT_POLICY_EVENT_BINDING_KEYS}
        events.append(event)
        event_sequence_hash = canonical_sha256(events)
        if row["event_sequence_sha256"] != event_sequence_hash:
            raise SecFilingGemmaStageAuthorizationError(
                "Source development policy row crossed its event prefix"
            )
        if ordinal == 1:
            initial_event_hash = event_sequence_hash
            prediction_genesis = _development_policy_prediction_genesis(
                contract_sha256=prefix["contract_sha256"],
                candidate_sha256=prefix["candidate_sha256"],
                corpus_universe_sha256=prefix["corpus_universe_sha256"],
                calendar_sessions_sha256=expected_policy_calendar,
                initial_event_sequence_sha256=initial_event_hash,
            )
            if (
                prefix["initial_event_sequence_sha256"] != initial_event_hash
                or prefix["genesis_sha256"] != prediction_genesis
                or row["prior_prediction_prefix_sha256"] is not None
                or row["parent_prediction_sha256"] != prediction_genesis
            ):
                raise SecFilingGemmaStageAuthorizationError(
                    "Source development policy genesis changed"
                )
            parent_tip = prediction_genesis
        else:
            if (
                row["prior_prediction_prefix_sha256"] != prior_prefix_hash
                or row["parent_prediction_sha256"] != prior_tip_hash
            ):
                raise SecFilingGemmaStageAuthorizationError(
                    "Source development policy row crossed its cumulative parent"
                )
            assert prior_tip_hash is not None
            parent_tip = prior_tip_hash
        row_body = {
            field: row[field]
            for field in _AUTHORITATIVE_PREDICTION_ROW_KEYS
            if field != "prediction_row_sha256"
        }
        row_hash = canonical_sha256(row_body)
        if row["prediction_row_sha256"] != row_hash:
            raise SecFilingGemmaStageAuthorizationError(
                "Source development policy row hash is inconsistent"
            )
        rows.append(row)
        prefix_body = _development_policy_prefix_body(
            prefix=prefix,
            rows=rows,
            event_sequence_sha256=event_sequence_hash,
            parent_prefix_sha256=prior_prefix_hash,
            parent_tip_sha256=parent_tip,
        )
        cumulative_prefix = {
            **prefix_body,
            "prediction_prefix_sha256": canonical_sha256(prefix_body),
        }
        artifact_bytes = _canonical_compact_json_bytes(
            cumulative_prefix,
            f"development policy-prefix artifact {ordinal}",
        )
        spec_body = {
            "schema_version": (
                _DEVELOPMENT_POLICY_PREFIX_SEAL_INPUT_SPEC_SCHEMA_VERSION
            ),
            "seal_ordinal": ordinal,
            "policy_sequence_number": ordinal,
            "decision_session": decision,
            "accession_number": accession,
            "prediction_row_sha256": row_hash,
            "parent_prediction_prefix_sha256": prior_prefix_hash,
            "parent_prediction_tip_sha256": parent_tip,
            "prediction_prefix_sha256": cumulative_prefix[
                "prediction_prefix_sha256"
            ],
            "prediction_tip_sha256": row_hash,
            "prediction_rows_sha256": cumulative_prefix["rows_sha256"],
            "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "artifact_size_bytes": len(artifact_bytes),
        }
        seal_specs.append(
            {
                **spec_body,
                "policy_prefix_seal_input_spec_sha256": canonical_sha256(
                    spec_body
                ),
            }
        )
        prior_prefix_hash = cumulative_prefix["prediction_prefix_sha256"]
        prior_tip_hash = row_hash
    final_prefix_body = _development_policy_prefix_body(
        prefix=prefix,
        rows=rows,
        event_sequence_sha256=rows[-1]["event_sequence_sha256"],
        parent_prefix_sha256=(seal_specs[-2]["prediction_prefix_sha256"] if len(seal_specs) > 1 else None),
        parent_tip_sha256=(rows[-2]["prediction_row_sha256"] if len(rows) > 1 else prefix["genesis_sha256"]),
    )
    expected_final_prefix = {
        **final_prefix_body,
        "prediction_prefix_sha256": canonical_sha256(final_prefix_body),
    }
    if prefix != expected_final_prefix or prefix["rows_sha256"] != canonical_sha256(rows):
        raise SecFilingGemmaStageAuthorizationError(
            "Source development policy prefix differs from its exact cumulative replay"
        )
    _validated_development_policy_prefix_seal_input_specs(seal_specs)
    if batch_hash != batch["policy_replay_batch_sha256"]:
        raise AssertionError("validated policy replay batch hash changed")
    return batch, seal_specs


def _development_policy_prefix_seal_namespace_sha256(
    *,
    development_root_scope_sha256: str,
    source_development_policy_replay_plan_sha256: str,
    source_development_policy_replay_batch_sha256: str,
    final_policy_prefix_sha256: str,
    policy_prefix_seal_input_specs_sha256: str,
    artifact_sealer_genesis_pin_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "domain": _DEVELOPMENT_POLICY_PREFIX_SEAL_NAMESPACE_KIND,
            "development_root_scope_sha256": development_root_scope_sha256,
            "source_development_policy_replay_plan_sha256": (
                source_development_policy_replay_plan_sha256
            ),
            "source_development_policy_replay_batch_sha256": (
                source_development_policy_replay_batch_sha256
            ),
            "final_policy_prefix_sha256": final_policy_prefix_sha256,
            "policy_prefix_seal_input_specs_sha256": (
                policy_prefix_seal_input_specs_sha256
            ),
            "artifact_sealer_genesis_pin_sha256": (
                artifact_sealer_genesis_pin_sha256
            ),
        }
    )


def build_development_policy_prefix_seal_plan(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    development_root_scope_sha256: str,
    authenticated_store_state_bytes_sha256: str,
    source_development_policy_replay_plan: Mapping[str, Any],
    source_development_policy_replay_projection_sha256: str,
    source_development_policy_replay_batch: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    independent_current_tip_anchor_bytes_sha256: str,
    artifact_sealer_genesis_pin: Mapping[str, Any],
    artifact_sealer_genesis_pin_bytes_sha256: str,
    source_identity_receipt_sha256: str,
    source_role_bindings_sha256: str,
) -> dict[str, Any]:
    """Authorize only exact cumulative-prefix materialization and atomic sealing."""

    state, ledger = _validated_store_snapshot(authenticated_store_snapshot)
    current_tip = validate_reveal_store_current_tip_anchor(
        state, independent_current_tip_anchor
    )
    state_bytes_hash = _sha256(
        authenticated_store_state_bytes_sha256,
        "development policy-prefix seal authenticated state bytes hash",
    )
    tip_bytes_hash = _sha256(
        independent_current_tip_anchor_bytes_sha256,
        "development policy-prefix seal current-tip bytes hash",
    )
    if (
        not hmac.compare_digest(
            state_bytes_hash,
            hashlib.sha256(_encoded_store_snapshot(state)).hexdigest(),
        )
        or not hmac.compare_digest(
            tip_bytes_hash,
            hashlib.sha256(_encoded_store_snapshot(current_tip)).hexdigest(),
        )
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy-prefix seal byte pins crossed their parsed state or tip"
        )
    consumed_count = _strict_int(
        ledger["chain"]["consumed_request_count"],
        "development policy-prefix seal consumed request count",
    )
    if consumed_count != 0 or current_tip["consumed_request_count"] != 0:
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy-prefix sealing requires the untouched zero-consumption store"
        )
    root_scope_hash = _sha256(
        development_root_scope_sha256,
        "development policy-prefix seal root scope hash",
    )
    replay_plan = _mapping(
        source_development_policy_replay_plan,
        "source development policy replay plan",
    )
    replay_plan_hash = _sha256(
        replay_plan.get("development_policy_replay_plan_sha256"),
        "source development policy replay plan hash",
    )
    validate_development_policy_replay_plan(
        replay_plan,
        expected_development_policy_replay_plan_sha256=replay_plan_hash,
    )
    if (
        replay_plan["development_root_scope_sha256"] != root_scope_hash
        or replay_plan["start_store_state_bytes_sha256"] != state_bytes_hash
        or replay_plan["start_store_state_sha256"] != state["state_sha256"]
        or replay_plan["start_current_tip_anchor_bytes_sha256"] != tip_bytes_hash
        or replay_plan["start_current_tip_anchor_sha256"]
        != current_tip["tip_anchor_sha256"]
        or replay_plan["start_current_tip_revision"] != current_tip["revision"]
        or replay_plan["start_consumed_request_count"] != consumed_count
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy-prefix seal crossed its replay-plan store ancestry"
        )
    batch, seal_specs = _validated_source_development_policy_replay_batch(
        source_development_policy_replay_batch,
        source_policy_replay_plan=replay_plan,
    )
    prefix = batch["policy_prefix"]
    genesis_pin = _validated_development_policy_artifact_genesis_pin(
        artifact_sealer_genesis_pin,
        expected_candidate_sha256=batch["candidate_sha256"],
        expected_pin_bytes_sha256=artifact_sealer_genesis_pin_bytes_sha256,
    )
    specs_hash = canonical_sha256(seal_specs)
    namespace_hash = _development_policy_prefix_seal_namespace_sha256(
        development_root_scope_sha256=root_scope_hash,
        source_development_policy_replay_plan_sha256=replay_plan_hash,
        source_development_policy_replay_batch_sha256=batch[
            "policy_replay_batch_sha256"
        ],
        final_policy_prefix_sha256=prefix["prediction_prefix_sha256"],
        policy_prefix_seal_input_specs_sha256=specs_hash,
        artifact_sealer_genesis_pin_sha256=genesis_pin["external_pin_sha256"],
    )
    capabilities = _expected_development_policy_prefix_seal_capabilities()
    body = {
        "schema_version": DEVELOPMENT_POLICY_PREFIX_SEAL_PLAN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "plan_kind": "request_free_development_policy_prefix_seal",
        "artifact_stage": "development",
        "development_root_scope_sha256": root_scope_hash,
        "start_store_state_bytes_sha256": state_bytes_hash,
        "start_store_state_sha256": state["state_sha256"],
        "start_current_tip_anchor_bytes_sha256": tip_bytes_hash,
        "start_current_tip_anchor_sha256": current_tip["tip_anchor_sha256"],
        "start_current_tip_revision": current_tip["revision"],
        "start_consumed_request_count": consumed_count,
        "source_development_policy_replay_plan_sha256": replay_plan_hash,
        "source_development_policy_replay_projection_sha256": _sha256(
            source_development_policy_replay_projection_sha256,
            "source development policy replay projection hash",
        ),
        "source_development_policy_replay_batch_sha256": batch[
            "policy_replay_batch_sha256"
        ],
        "source_development_oof_prediction_plan_sha256": replay_plan[
            "source_development_oof_prediction_plan_sha256"
        ],
        "source_development_oof_prediction_projection_sha256": replay_plan[
            "source_development_oof_prediction_projection_sha256"
        ],
        "source_development_oof_prediction_batch_sha256": replay_plan[
            "source_development_oof_prediction_batch_sha256"
        ],
        "source_raw_prediction_rows_sha256": replay_plan[
            "source_raw_prediction_rows_sha256"
        ],
        "source_raw_prediction_tip_sha256": replay_plan[
            "source_raw_prediction_tip_sha256"
        ],
        "source_raw_prediction_row_count": replay_plan[
            "source_raw_prediction_row_count"
        ],
        "source_raw_to_policy_bindings_sha256": batch[
            "raw_to_policy_bindings_sha256"
        ],
        "source_identity_receipt_sha256": _sha256(
            source_identity_receipt_sha256,
            "development policy-prefix seal source identity receipt hash",
        ),
        "source_role_bindings_sha256": _sha256(
            source_role_bindings_sha256,
            "development policy-prefix seal source-role bindings hash",
        ),
        "candidate_sha256": batch["candidate_sha256"],
        "corpus_universe_sha256": batch["corpus_universe_sha256"],
        "source_market_calendar_sessions_sha256": batch[
            "source_market_calendar_sessions_sha256"
        ],
        "policy_calendar_sessions_sha256": batch[
            "policy_calendar_sessions_sha256"
        ],
        "development_cutoff_session": batch["development_cutoff_session"],
        "policy_prefix_count": len(seal_specs),
        "final_policy_prefix_sha256": prefix["prediction_prefix_sha256"],
        "final_policy_tip_sha256": prefix["tip_sha256"],
        "final_policy_rows_sha256": prefix["rows_sha256"],
        "policy_prefix_seal_input_count": len(seal_specs),
        "policy_prefix_seal_input_specs": seal_specs,
        "policy_prefix_seal_input_specs_sha256": specs_hash,
        "artifact_encoding": _PREDICTION_ARTIFACT_ENCODING,
        "artifact_store_schema_version": _PREDICTION_ARTIFACT_STORE_SCHEMA_VERSION,
        "artifact_sealer_genesis_pin_schema_version": genesis_pin[
            "schema_version"
        ],
        "artifact_sealer_genesis_pin_bytes_sha256": _sha256(
            artifact_sealer_genesis_pin_bytes_sha256,
            "development policy-prefix seal genesis pin bytes hash",
        ),
        "artifact_sealer_genesis_pin_sha256": genesis_pin[
            "external_pin_sha256"
        ],
        "artifact_sealer_genesis_seal_tip_sha256": genesis_pin[
            "seal_tip_sha256"
        ],
        "seal_store_namespace_kind": _DEVELOPMENT_POLICY_PREFIX_SEAL_NAMESPACE_KIND,
        "seal_store_namespace_sha256": namespace_hash,
        "authorized_seal_receipt_schema_version": (
            DEVELOPMENT_POLICY_PREFIX_SEAL_RECEIPT_SCHEMA_VERSION
        ),
        "authorized_external_pin_schema_version": (
            DEVELOPMENT_POLICY_PREFIX_EXTERNAL_PIN_SCHEMA_VERSION
        ),
        "seal_batch_rule": _DEVELOPMENT_POLICY_PREFIX_SEAL_BATCH_RULE,
        "seal_recovery_rule": _DEVELOPMENT_POLICY_PREFIX_SEAL_RECOVERY_RULE,
        "external_pin_rule": _DEVELOPMENT_POLICY_PREFIX_EXTERNAL_PIN_RULE,
        **capabilities,
    }
    return {
        **body,
        "development_policy_prefix_seal_plan_sha256": canonical_sha256(body),
    }


def validate_development_policy_prefix_seal_plan(
    plan: Mapping[str, Any],
    *,
    expected_development_policy_prefix_seal_plan_sha256: str,
) -> str:
    """Validate one exact seal-only plan without reading labels or outcomes."""

    value = _mapping(plan, "development policy-prefix seal plan")
    _expect_keys(
        value,
        _DEVELOPMENT_POLICY_PREFIX_SEAL_PLAN_KEYS,
        "development policy-prefix seal plan",
    )
    capabilities = _expected_development_policy_prefix_seal_capabilities()
    if any(type(value[field]) is not bool for field in capabilities):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy-prefix seal capabilities must be exact booleans"
        )
    for field in (
        "contract_sha256",
        "development_root_scope_sha256",
        "start_store_state_bytes_sha256",
        "start_store_state_sha256",
        "start_current_tip_anchor_bytes_sha256",
        "start_current_tip_anchor_sha256",
        "source_development_policy_replay_plan_sha256",
        "source_development_policy_replay_projection_sha256",
        "source_development_policy_replay_batch_sha256",
        "source_development_oof_prediction_plan_sha256",
        "source_development_oof_prediction_projection_sha256",
        "source_development_oof_prediction_batch_sha256",
        "source_raw_prediction_rows_sha256",
        "source_raw_prediction_tip_sha256",
        "source_raw_to_policy_bindings_sha256",
        "source_identity_receipt_sha256",
        "source_role_bindings_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "source_market_calendar_sessions_sha256",
        "policy_calendar_sessions_sha256",
        "final_policy_prefix_sha256",
        "final_policy_tip_sha256",
        "final_policy_rows_sha256",
        "policy_prefix_seal_input_specs_sha256",
        "artifact_sealer_genesis_pin_bytes_sha256",
        "artifact_sealer_genesis_pin_sha256",
        "artifact_sealer_genesis_seal_tip_sha256",
        "seal_store_namespace_sha256",
    ):
        _sha256(value[field], f"development policy-prefix seal {field}")
    _strict_int(
        value["start_current_tip_revision"],
        "development policy-prefix seal start current-tip revision",
    )
    specs = _validated_development_policy_prefix_seal_input_specs(
        value["policy_prefix_seal_input_specs"]
    )
    expected_count = len(specs)
    for field, expected in (
        ("start_consumed_request_count", 0),
        ("source_raw_prediction_row_count", expected_count),
        ("policy_prefix_count", expected_count),
        ("policy_prefix_seal_input_count", expected_count),
    ):
        if _strict_int(value[field], f"development policy-prefix seal {field}") != expected:
            raise SecFilingGemmaStageAuthorizationError(
                "Development policy-prefix seal count changed"
            )
    expected_policy_calendar = development_policy_session_calendar_sha256(
        DEVELOPMENT_POLICY_SESSION_DATES
    )
    last = specs[-1]
    genesis_seal_tip = canonical_sha256(
        {
            "domain": "aapl-sec-gemma-prediction-artifact-seal-genesis-v1",
            "candidate_sha256": value["candidate_sha256"],
            "stage": "development",
        }
    )
    genesis_body = {
        "schema_version": _PREDICTION_ARTIFACT_EXTERNAL_PIN_SCHEMA_VERSION,
        "candidate_sha256": value["candidate_sha256"],
        "stage": "development",
        "sealed_artifact_count": 0,
        "last_prediction_sequence_number": 0,
        "prediction_prefix_sha256": None,
        "prediction_tip_sha256": None,
        "prediction_rows_sha256": None,
        "artifact_sha256": None,
        "seal_tip_sha256": genesis_seal_tip,
    }
    genesis_pin = {
        **genesis_body,
        "external_pin_sha256": canonical_sha256(genesis_body),
    }
    genesis_pin_bytes_hash = hashlib.sha256(
        _canonical_compact_json_bytes(genesis_pin, "expected genesis pin")
    ).hexdigest()
    namespace_hash = _development_policy_prefix_seal_namespace_sha256(
        development_root_scope_sha256=value["development_root_scope_sha256"],
        source_development_policy_replay_plan_sha256=value[
            "source_development_policy_replay_plan_sha256"
        ],
        source_development_policy_replay_batch_sha256=value[
            "source_development_policy_replay_batch_sha256"
        ],
        final_policy_prefix_sha256=value["final_policy_prefix_sha256"],
        policy_prefix_seal_input_specs_sha256=value[
            "policy_prefix_seal_input_specs_sha256"
        ],
        artifact_sealer_genesis_pin_sha256=value[
            "artifact_sealer_genesis_pin_sha256"
        ],
    )
    if (
        value["schema_version"]
        != DEVELOPMENT_POLICY_PREFIX_SEAL_PLAN_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["contract_sha256"] != canonical_sha256(build_contract_manifest())
        or value["plan_kind"] != "request_free_development_policy_prefix_seal"
        or value["artifact_stage"] != "development"
        or value["development_cutoff_session"] != STAGE_WINDOWS["development"][1]
        or value["policy_calendar_sessions_sha256"] != expected_policy_calendar
        or value["policy_prefix_seal_input_specs_sha256"]
        != canonical_sha256(specs)
        or value["final_policy_prefix_sha256"]
        != last["prediction_prefix_sha256"]
        or value["final_policy_tip_sha256"] != last["prediction_tip_sha256"]
        or value["final_policy_rows_sha256"] != last["prediction_rows_sha256"]
        or value["artifact_encoding"] != _PREDICTION_ARTIFACT_ENCODING
        or value["artifact_store_schema_version"]
        != _PREDICTION_ARTIFACT_STORE_SCHEMA_VERSION
        or value["artifact_sealer_genesis_pin_schema_version"]
        != _PREDICTION_ARTIFACT_EXTERNAL_PIN_SCHEMA_VERSION
        or value["artifact_sealer_genesis_pin_bytes_sha256"]
        != genesis_pin_bytes_hash
        or value["artifact_sealer_genesis_pin_sha256"]
        != genesis_pin["external_pin_sha256"]
        or value["artifact_sealer_genesis_seal_tip_sha256"] != genesis_seal_tip
        or value["seal_store_namespace_kind"]
        != _DEVELOPMENT_POLICY_PREFIX_SEAL_NAMESPACE_KIND
        or value["seal_store_namespace_sha256"] != namespace_hash
        or value["authorized_seal_receipt_schema_version"]
        != DEVELOPMENT_POLICY_PREFIX_SEAL_RECEIPT_SCHEMA_VERSION
        or value["authorized_external_pin_schema_version"]
        != DEVELOPMENT_POLICY_PREFIX_EXTERNAL_PIN_SCHEMA_VERSION
        or value["seal_batch_rule"]
        != _DEVELOPMENT_POLICY_PREFIX_SEAL_BATCH_RULE
        or value["seal_recovery_rule"]
        != _DEVELOPMENT_POLICY_PREFIX_SEAL_RECOVERY_RULE
        or value["external_pin_rule"]
        != _DEVELOPMENT_POLICY_PREFIX_EXTERNAL_PIN_RULE
        or any(value[field] is not expected for field, expected in capabilities.items())
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy-prefix seal identity, ancestry, or authority changed"
        )
    observed = _self_hash(
        value,
        "development_policy_prefix_seal_plan_sha256",
        "development policy-prefix seal plan",
    )
    expected_hash = _sha256(
        expected_development_policy_prefix_seal_plan_sha256,
        "expected development policy-prefix seal plan hash",
    )
    if not hmac.compare_digest(observed, expected_hash):
        raise SecFilingGemmaStageAuthorizationError(
            "Development policy-prefix seal plan is not externally pinned"
        )
    return observed


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
    "DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_SCHEMA_VERSION",
    "DEVELOPMENT_LABEL_ASSEMBLY_PLAN_SCHEMA_VERSION",
    "DEVELOPMENT_OOF_LEARNER_FIT_PLAN_SCHEMA_VERSION",
    "DEVELOPMENT_OOF_PREDICTION_PLAN_SCHEMA_VERSION",
    "DEVELOPMENT_POLICY_REPLAY_PLAN_SCHEMA_VERSION",
    "DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_SCHEMA_VERSION",
    "DEVELOPMENT_MARKET_BATCH_COMPONENT_ID",
    "DEVELOPMENT_MARKET_EXECUTION_ABORT_SCHEMA_VERSION",
    "DEVELOPMENT_MARKET_EXECUTION_CLAIM_SCHEMA_VERSION",
    "DEVELOPMENT_MARKET_READER_RECEIPT_SCHEMA_VERSION",
    "DEVELOPMENT_SEC_EXECUTION_ABORT_SCHEMA_VERSION",
    "DEVELOPMENT_SEC_EXECUTION_CLAIM_SCHEMA_VERSION",
    "DEVELOPMENT_SEC_READER_RECEIPT_SCHEMA_VERSION",
    "DEVELOPMENT_MODEL_EXECUTION_ABORT_SCHEMA_VERSION",
    "DEVELOPMENT_MODEL_EXECUTION_CLAIM_SCHEMA_VERSION",
    "DEVELOPMENT_MODEL_READER_RECEIPT_SCHEMA_VERSION",
    "DEVELOPMENT_ROOT_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION",
    "MARKET_EXECUTION_SOURCE_ROLES",
    "MODEL_EXECUTION_SOURCE_ROLES",
    "REVEAL_STORE_CURRENT_TIP_ANCHOR_SCHEMA_VERSION",
    "OWNED_SEC_RAW_BATCH_MAX_BYTES",
    "SEC_EXECUTION_RESOLVED_SOURCE_PATHS",
    "SEC_CORPUS_REPOSITORY_PATH",
    "SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID",
    "STAGE_EVIDENCE_OUTPUT_COMPONENT_ID",
    "STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH",
    "STAGE_MODEL_BATCH_COMPONENT_ID",
    "STAGE_MODEL_EXECUTION_ABORT_SCHEMA_VERSION",
    "STAGE_MODEL_EXECUTION_CLAIM_SCHEMA_VERSION",
    "STAGE_MODEL_READER_RECEIPT_SCHEMA_VERSION",
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
    "build_development_market_execution_abort",
    "build_development_market_execution_claim",
    "build_development_market_reader_receipt",
    "build_development_feature_assembly_plan",
    "build_development_label_assembly_plan",
    "build_development_oof_learner_fit_plan",
    "build_development_oof_prediction_plan",
    "build_development_policy_replay_plan",
    "build_development_training_membership_assembly_plan",
    "build_development_sec_execution_abort",
    "build_development_sec_execution_claim",
    "build_development_sec_reader_receipt",
    "build_development_model_execution_abort",
    "build_development_model_execution_claim",
    "build_development_model_reader_receipt",
    "build_development_root_carry_in_reader_receipt",
    "build_stage_carry_in_reader_receipt",
    "build_stage_model_execution_abort",
    "build_stage_model_execution_claim",
    "build_stage_model_reader_receipt",
    "build_stage_sec_execution_abort",
    "build_stage_sec_execution_claim",
    "build_stage_sec_reader_receipt",
    "build_reveal_store_current_tip_anchor",
    "derive_consumed_stage_store_state_pin",
    "derive_reveal_store_trusted_stage_content_pin",
    "validate_consumed_stage_authorization_grant",
    "validate_consumed_stage_output_receipt",
    "validate_consumed_stage_store_state_pin",
    "validate_development_market_execution_abort",
    "validate_development_market_execution_claim",
    "validate_development_market_reader_receipt",
    "validate_development_feature_assembly_plan",
    "validate_development_label_assembly_plan",
    "validate_development_oof_learner_fit_plan",
    "validate_development_oof_prediction_plan",
    "validate_development_policy_replay_plan",
    "validate_development_training_membership_assembly_plan",
    "validate_development_root_carry_in_reader_receipt",
    "validate_development_model_execution_abort",
    "validate_development_model_execution_claim",
    "validate_development_model_reader_receipt",
    "validate_reveal_store_current_tip_anchor",
    "validate_reveal_store_current_tip_anchor_structure",
    "validate_reveal_store_current_tip_anchor_transition",
    "validate_stage_carry_in_reader_receipt",
    "validate_stage_model_execution_abort",
    "validate_stage_model_execution_claim",
    "validate_stage_model_reader_receipt",
    "validate_trusted_stage_content_authentication_receipt",
]
