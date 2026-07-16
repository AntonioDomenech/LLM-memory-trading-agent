"""Pure v2.2 preregistration for the SEC/Gemma online risk-overlay experiment.

This module performs no filesystem, network, SEC, market, model, or clock I/O.
The exact manifest is intentionally strict: changing any field creates a new
approach and requires a new branch before any semantic extraction or scoring.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_extractor_prompt import (
    EXTRACTOR_SYSTEM_PROMPT,
)
from agent_benchmark.sec_filing_gemma_extractor_schema import (
    EXTRACTOR_SCHEMA_VERSION,
    build_extractor_json_schema,
)


CONTRACT_VERSION: Final[str] = "aapl-sec-gemma-online-risk-overlay-v2-2"
BRANCH_NAME: Final[str] = "codex/aapl-sec-gemma-online-risk-overlay-v2-2"
BASELINE_POLICY_ID: Final[str] = "fixed-contextual-plus-weak-trend-union-v1"
BASELINE_SOURCE_FILE: Final[str] = (
    "agent_benchmark/chronological_exhaustion_expert.py"
)
BASELINE_SOURCE_SHA256: Final[str] = (
    "a30224763c9858aed905b76215c2c5a66eddd58f107182d751d8eb6a32688c6e"
)
MODEL_NAME: Final[str] = "gemma4:12b"
MODEL_MANIFEST_SHA256: Final[str] = (
    "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
)
MODEL_CONFIG_DIGEST: Final[str] = (
    "c805f5b265d8e695c44f4065dfc368206cd8026447604925fef8db57ee32ee23"
)
MODEL_LAYER_DIGESTS: Final[tuple[str, ...]] = (
    "1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606",
    "675ad6e68101ca9413ec806855c452362f0213f2dfc5800996b086fdb8119842",
    "0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594",
    "56380ca2ab89f1f68c283f4d50863c0bcab52ae3f1b9a88e4ab5617b176f71a3",
)
OLLAMA_VERSION: Final[str] = "0.32.0"
RUNTIME_FINGERPRINT_SHA256: Final[str] = (
    "816a7c1a6b1e87d083f8e0f85654f80ba0db124bf09fe8dedce960c8124e1c77"
)
FINAL_ATTEMPT_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-final-attempt-001"
)
CONFIRMATION_ATTEMPT_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-confirmation-attempt-001"
)
DEVELOPMENT_ATTEMPT_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-development-attempt-001"
)
DEVELOPMENT_ACQUISITION_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-development-acquisition-001"
)
POSITIVE_EDGE_TOLERANCE: Final[float] = 1e-12
HORIZON_SESSIONS: Final[int] = 20
LABEL_MATURITY_OFFSET: Final[int] = 21
PROBABILITY_GATE: Final[float] = 0.55
EXPECTED_EDGE_GATE: Final[float] = 0.0025
MINIMUM_TRAINING_ROWS: Final[int] = 20
MINIMUM_CLASS_ROWS: Final[int] = 4
MAX_TOTAL_RUNTIME_SECONDS: Final[int] = 3_600
MAX_SEC_SECONDS: Final[int] = 720
MAX_MODEL_SECONDS: Final[int] = 2_160
MAX_DETERMINISTIC_SECONDS: Final[int] = 480
MAX_SEC_REQUESTS: Final[int] = 1_000
MAX_SEC_BYTES: Final[int] = 1_610_612_736
MAX_SEC_REQUESTS_PER_SECOND: Final[int] = 2
MAX_MARKET_REQUESTS_PER_STAGE: Final[int] = 6
MAX_MARKET_SECONDS: Final[int] = 210
GOVERNANCE_CONTINGENCY_SECONDS: Final[int] = 239
PUBLICATION_INTENT_PREPARATION_SECONDS: Final[int] = 89
SUPERVISED_PUBLICATION_SECONDS: Final[int] = 90
PUBLICATION_FINALIZATION_PENDING_SECONDS: Final[int] = 60
MAX_PUBLICATION_RECOVERY_SECONDS: Final[int] = 300
PUBLICATION_RECOVERY_COMPLETION_GENESIS_SHA256: Final[str] = (
    "34e85583a21ae65da1a764dce38fff53d1dc1970fa274042b722f55461389f61"
)
PUBLICATION_RECOVERY_NO_REMOTE_OBSERVATION_SHA256: Final[str] = (
    "5f16fcac9dc256973faecb69ac46cea7548f84819af8e71c6e072e9ac44eb095"
)
PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256: Final[str] = (
    "1119c4e4995b1184490a7d9e9494287258a484c6b195fdcceeeab1eea08758e8"
)
PUBLICATION_NORMAL_OPERATION_SHA256: Final[str] = (
    "3ab7597f3ee2cc80514c4d822ed653c7c43f3f41596eff1ebf9556a84a2da431"
)
PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256: Final[str] = (
    "1abebdbf8ecf79863e8c8abebc1c10936795cc51f83a1dfd654b0348d27eb938"
)
PUBLICATION_NORMAL_NO_RECOVERY_COMPLETION_SHA256: Final[str] = (
    "65d1d544c19404d359d9c9edc45d32653e6c14419e0001f7eb79edfde0e96f85"
)
PUBLICATION_REMOTE_REF_ABSENT_SENTINEL: Final[str] = "REF_ABSENT"
PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL: Final[str] = "VALUE_MISSING"
PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL: Final[str] = "VALUE_MALFORMED"
PUBLICATION_REMOTE_READBACK_COMMAND_PROFILE_SHA256: Final[str] = (
    "985e00259f3bafc0c965d8cfeff4208c7623b6d5edf94b753627764d4cf5f66a"
)
PUBLICATION_PUSH_COMMAND_PROFILE_SHA256: Final[str] = (
    "be2dbeb69d602d1bbe13f41609e8717bf6d605d1bdff2a38486b5db5a38abc05"
)
EMPTY_BYTES_SHA256: Final[str] = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)

SOURCE_PINS: Final[dict[str, str]] = {
    "sec_filing_gemma_contract": (
        "d5a268da138862b510b0f12b139a04fa91a91d7deba4fe2b28666609c03318e4"
    ),
    "sec_filing_gemma_preprocessor": (
        "febdbaa2fa5f6ecd528fdc9642614b0f8fd8df79f0f9c7ea5c398d89d1be2544"
    ),
    "sec_filing_gemma_extractor_prompt": (
        "312d2aad202ece6b9e80f4508127f702fc965e92aa6bcb30217e4a62ec8e8d7d"
    ),
    "sec_filing_gemma_extractor_schema": (
        "ee9b5804a3816799cd26549b56ca852586d5effe57897c5d74cf06ce8db52c6e"
    ),
    "sec_filing_gemma_ollama": (
        "854fa9184658549ef72e6d618eab018d07161adddf0b8268f0f167dc655557d4"
    ),
    "sec_filing_gemma_corpus": (
        "74831feadcae050eee497da0a3405d65a5c4649a59830bf3f768ab4d35f9164b"
    ),
    "sec_audit_transport": (
        "52915b70c4e43d40987e69c2102241aa6926fdbe93cf688e2388ba01dcf29bd2"
    ),
    "sec_point_in_time": (
        "ccdfc7514fc94223fc9b4f1f57975ac17bcd0f8e5c2e82b10c9f970d11248854"
    ),
    "sec_filing_content": (
        "70c969c8eee82e82c0ea1a8a5e424178122fdba8ac1b12ed5a12e207177e10bf"
    ),
    "sec_session_calendar": (
        "9a463fa0453440dc777a850e7af934dfc0f2740243e7e8e05ff6ebc430262f71"
    ),
    "sec_filing_gemma_market_source_bytes": (
        "4cf885352769b4b48fd0a7c90c8cd50dbb4330ae6a1f144ff17aed6eb9b9030d"
    ),
    "sec_filing_gemma_market_acquirer": (
        "79ac6275fa5a10b2805a52af7748e81ca3a6a8582413a2077b43a448ab410bc0"
    ),
    "sec_filing_gemma_learner": (
        "b948820036454f76fb606068b54060916535f3721e29564340565c890113fc92"
    ),
    "chronological_exhaustion_expert": BASELINE_SOURCE_SHA256,
}
SOURCE_PIN_FILES: Final[dict[str, str]] = {
    "sec_filing_gemma_contract": "agent_benchmark/sec_filing_gemma_contract.py",
    "sec_filing_gemma_preprocessor": (
        "agent_benchmark/sec_filing_gemma_preprocessor.py"
    ),
    "sec_filing_gemma_extractor_prompt": (
        "agent_benchmark/sec_filing_gemma_extractor_prompt.py"
    ),
    "sec_filing_gemma_extractor_schema": (
        "agent_benchmark/sec_filing_gemma_extractor_schema.py"
    ),
    "sec_filing_gemma_ollama": "agent_benchmark/sec_filing_gemma_ollama.py",
    "sec_filing_gemma_corpus": "agent_benchmark/sec_filing_gemma_corpus.py",
    "sec_audit_transport": "agent_benchmark/sec_audit_transport.py",
    "sec_point_in_time": "agent_benchmark/sec_point_in_time.py",
    "sec_filing_content": "agent_benchmark/sec_filing_content.py",
    "sec_session_calendar": "agent_benchmark/sec_session_calendar.py",
    "sec_filing_gemma_market_source_bytes": (
        "agent_benchmark/sec_filing_gemma_market_source_bytes.py"
    ),
    "sec_filing_gemma_market_acquirer": (
        "agent_benchmark/sec_filing_gemma_market_acquirer.py"
    ),
    "sec_filing_gemma_learner": "agent_benchmark/sec_filing_gemma_learner.py",
    "chronological_exhaustion_expert": BASELINE_SOURCE_FILE,
}

NEW_SOURCE_FILES: Final[dict[str, str]] = {
    "acquisition": (
        "agent_benchmark/sec_gemma_online_risk_overlay_acquisition.py"
    ),
    "attempt": "agent_benchmark/sec_gemma_online_risk_overlay_attempt.py",
    "baseline": "agent_benchmark/sec_gemma_online_risk_overlay_baseline.py",
    "features": "agent_benchmark/sec_gemma_online_risk_overlay_features.py",
    "learner": "agent_benchmark/sec_gemma_online_risk_overlay_learner.py",
    "ledger": "agent_benchmark/sec_gemma_online_risk_overlay_ledger.py",
    "market_verifier": (
        "agent_benchmark/sec_gemma_online_risk_overlay_market_verifier.py"
    ),
    "metrics": "agent_benchmark/sec_gemma_online_risk_overlay_metrics.py",
    "no_leverage": (
        "agent_benchmark/sec_gemma_online_risk_overlay_no_leverage.py"
    ),
    "policy": "agent_benchmark/sec_gemma_online_risk_overlay_policy.py",
    "production": (
        "agent_benchmark/sec_gemma_online_risk_overlay_production.py"
    ),
    "publisher": (
        "agent_benchmark/sec_gemma_online_risk_overlay_publisher.py"
    ),
    "registry": "agent_benchmark/sec_gemma_online_risk_overlay_registry.py",
    "replay": "agent_benchmark/sec_gemma_online_risk_overlay_replay.py",
    "runner": "agent_benchmark/sec_gemma_online_risk_overlay_runner.py",
    "runtime": "agent_benchmark/sec_gemma_online_risk_overlay_runtime.py",
    "source_verifier": (
        "agent_benchmark/sec_gemma_online_risk_overlay_source_verifier.py"
    ),
    "store": "agent_benchmark/sec_gemma_online_risk_overlay_store.py",
    "vault": "agent_benchmark/sec_gemma_online_risk_overlay_vault.py",
}

ACQUISITION_VALIDATION_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "verifier_id",
    "verdict",
    "stage",
    "attempt_id",
    "attempt_kind",
    "acquisition_plan_sha256",
    "bundle_sha256",
    "manifest_sha256",
    "private_index_sha256",
    "predecessor_chain_bundle_sha256s",
    "checks",
    "check_set_sha256",
    "validation_sha256",
)
ACQUISITION_VALIDATION_CHECKS: Final[tuple[str, ...]] = (
    "exact_raw_bytes_replayed_sha256",
    "request_receipts_reconciled_sha256",
    "stage_and_attempt_scope_bound_sha256",
    "private_identity_digest_only_sha256",
    "market_prefix_continuity_replayed_sha256",
    "blinded_model_requests_replayed_sha256",
    "request_byte_retry_redirect_caps_reconciled_sha256",
)
ACQUISITION_TERMINAL_EVIDENCE_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "verifier_id",
    "verdict",
    "terminal_status",
    "stage",
    "attempt_id",
    "attempt_kind",
    "attempt_plan_sha256",
    "acquisition_validation_sha256",
    "bundle_sha256",
    "manifest_sha256",
    "private_index_sha256",
    "check_set_sha256",
    "record_commitment_sha256",
    "acquisition_artifact_receipt_sha256",
    "publication_intent_sha256",
    "publication_intent_store_receipt_sha256",
    "publication_receipt_sha256",
    "publication_receipt_store_receipt_sha256",
    "external_publication_sha256",
    "terminal_evidence_sha256",
)
SCORED_TERMINAL_EVIDENCE_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "verifier_id",
    "verdict",
    "terminal_status",
    "stage",
    "attempt_id",
    "attempt_kind",
    "attempt_plan_sha256",
    "stage_input_bundle_sha256",
    "deterministic_evaluation_sha256",
    "stage_metrics_input_sha256",
    "stage_metrics_sha256",
    "gate_report_sha256",
    "no_leverage_proofs_sha256",
    "joint_stage_report_sha256",
    "record_counts",
    "record_commitment_sha256",
    "joint_artifact_receipt_sha256",
    "gate_checks",
    "gate_check_set_sha256",
    "failed_gate_names",
    "publication_intent_sha256",
    "publication_intent_store_receipt_sha256",
    "publication_receipt_sha256",
    "publication_receipt_store_receipt_sha256",
    "external_publication_sha256",
    "terminal_evidence_sha256",
)
ACQUISITION_TERMINAL_RECONSTRUCTION_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "reconstruction_verifier_id",
    "contract_version",
    "contract_sha256",
    "implementation_manifest_sha256",
    "implementation_commit",
    "store_instance_id",
    "attempt_id",
    "attempt_kind",
    "attempt_plan_sha256",
    "stage",
    "terminal_status",
    "report_kind",
    "terminal_artifact_sha256",
    "terminal_artifact_store_receipt_sha256",
    "acquisition_validation_sha256",
    "bundle_sha256",
    "manifest_sha256",
    "private_index_sha256",
    "check_set_sha256",
    "record_counts",
    "record_commitment_sha256",
    "acquisition_artifact_receipt_sha256",
    "sealed_acquisition_phase_evidence_sha256",
    "sealed_vault_commitments_sha256",
    "terminal_reconstruction_material_sha256",
)
SCORED_TERMINAL_RECONSTRUCTION_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "reconstruction_verifier_id",
    "contract_version",
    "contract_sha256",
    "implementation_manifest_sha256",
    "implementation_commit",
    "store_instance_id",
    "attempt_id",
    "attempt_kind",
    "attempt_plan_sha256",
    "stage",
    "terminal_status",
    "report_kind",
    "terminal_artifact_sha256",
    "terminal_artifact_store_receipt_sha256",
    "stage_input_bundle_sha256",
    "deterministic_evaluation_sha256",
    "stage_metrics_input_sha256",
    "stage_metrics_sha256",
    "gate_report_sha256",
    "no_leverage_proofs_sha256",
    "joint_stage_report_sha256",
    "joint_artifact_receipt_sha256",
    "record_counts",
    "record_commitment_sha256",
    "gate_checks",
    "gate_check_set_sha256",
    "failed_gate_names",
    "terminal_reconstruction_material_sha256",
)
EXTERNAL_TAG_REF_TEMPLATE: Final[str] = (
    "refs/tags/sec-gemma-online-risk-overlay-v2-2/"
    "attempts/{attempt_id}/terminal"
)
FINAL_REGISTRY_TAG_REF_TEMPLATE: Final[str] = (
    "refs/tags/sec-gemma-online-risk-overlay-v2-2/"
    "registry/{attempt_id}/successor"
)
EXTERNAL_TAG_MESSAGE_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "contract_version",
    "contract_sha256",
    "implementation_commit",
    "attempt_id",
    "terminal_status",
    "report_kind",
    "artifact_sha256",
    "predecessor_publication_sha256",
    "external_cost_usd",
)
EXTERNAL_PUBLICATION_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "publisher_id",
    "contract_version",
    "contract_sha256",
    "implementation_commit",
    "attempt_id",
    "terminal_status",
    "report_kind",
    "artifact_sha256",
    "predecessor_publication_sha256",
    "tag_ref",
    "tag_target_commit",
    "tag_message_sha256",
    "remote_name",
    "remote_url",
    "remote_tag_object_sha1",
    "remote_peeled_commit",
    "external_cost_usd",
    "publication_sha256",
)
PUBLICATION_INTENT_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "intent_verifier_id",
    "contract_version",
    "contract_sha256",
    "implementation_manifest_sha256",
    "implementation_commit",
    "attempt_id",
    "attempt_kind",
    "attempt_plan_sha256",
    "terminal_status",
    "report_kind",
    "artifact_sha256",
    "artifact_store_receipt_sha256",
    "terminal_reconstruction_material_sha256",
    "terminal_reconstruction_material_store_receipt_sha256",
    "record_counts",
    "record_commitment_sha256",
    "store_instance_id",
    "store_journal_sequence",
    "store_journal_tip_sha256",
    "normal_attempt_elapsed_at_intent_prepare_hex",
    "predecessor_publication_sha256",
    "tag_ref",
    "tag_target_commit",
    "tag_message_sha256",
    "expected_tag_object_sha1",
    "expected_peeled_commit",
    "expected_publication_sha256",
    "remote_name",
    "remote_url",
    "intent_status",
    "research_effect_authority_invalidated",
    "semantic_result_release_blocked",
    "next_stage_authority_blocked",
    "external_cost_usd",
    "publication_intent_sha256",
)
PUBLICATION_RECEIPT_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "receipt_verifier_id",
    "contract_version",
    "contract_sha256",
    "implementation_manifest_sha256",
    "implementation_commit",
    "store_instance_id",
    "attempt_id",
    "publication_intent_sha256",
    "publication_intent_store_receipt_sha256",
    "publication_remote_observation_sha256",
    "recovery_invocation_completion_sha256",
    "pre_push_authorization_sha256",
    "external_publication_sha256",
    "remote_tag_object_sha1",
    "remote_peeled_commit",
    "pre_receipt_store_journal_sequence",
    "pre_receipt_store_journal_tip_sha256",
    "receipt_status",
    "publication_capability_invalidated",
    "terminalization_capability_required",
    "publication_receipt_sha256",
)
PUBLICATION_REMOTE_READBACK_EVIDENCE_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "evidence_verifier_id",
    "contract_version",
    "contract_sha256",
    "implementation_manifest_sha256",
    "implementation_commit",
    "store_instance_id",
    "store_session_nonce_sha256",
    "attempt_id",
    "publication_intent_sha256",
    "observation_operation_kind",
    "observation_operation_sha256",
    "worker_ownership_sha256",
    "observation_ordinal",
    "observation_phase",
    "prior_push_command_sha256",
    "tag_ref",
    "remote_name",
    "remote_url",
    "transport_isolation_profile_sha256",
    "isolated_transport_git_directory_manifest_sha256",
    "command_profile_id",
    "command_sequence_sha256",
    "process_exit_status",
    "process_exit_code",
    "stdout_byte_count",
    "stdout_sha256",
    "stderr_byte_count",
    "stderr_sha256",
    "transport_status",
    "ref_lookup_status",
    "raw_ref_object_value",
    "raw_peeled_value",
    "raw_tag_message_sha256",
    "remote_readback_evidence_sha256",
)
PUBLICATION_REMOTE_OBSERVATION_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "observation_verifier_id",
    "contract_version",
    "contract_sha256",
    "implementation_manifest_sha256",
    "implementation_commit",
    "store_instance_id",
    "store_session_nonce_sha256",
    "attempt_id",
    "publication_intent_sha256",
    "observation_operation_kind",
    "observation_operation_sha256",
    "worker_ownership_sha256",
    "observation_ordinal",
    "observation_phase",
    "prior_push_command_sha256",
    "tag_ref",
    "remote_name",
    "remote_url",
    "expected_tag_object_sha1",
    "expected_peeled_commit",
    "observed_ref_state",
    "observed_tag_object_sha1",
    "observed_peeled_commit",
    "observed_tag_message_sha256",
    "remote_readback_evidence_sha256",
    "pre_observation_store_journal_sequence",
    "pre_observation_store_journal_tip_sha256",
    "publication_remote_observation_sha256",
)
PUBLICATION_RECOVERY_INVOCATION_START_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "start_verifier_id",
    "contract_version",
    "contract_sha256",
    "implementation_manifest_sha256",
    "implementation_commit",
    "store_instance_id",
    "store_session_nonce_sha256",
    "attempt_id",
    "publication_intent_sha256",
    "invocation_ordinal",
    "prior_recovery_completion_sha256",
    "prior_cumulative_recovery_seconds",
    "start_status",
    "invocation_seconds_cap",
    "durable_pre_push_authorization_required",
    "pre_start_store_journal_sequence",
    "pre_start_store_journal_tip_sha256",
    "recovery_invocation_start_sha256",
)
PUBLICATION_PRE_PUSH_AUTHORIZATION_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "authorization_verifier_id",
    "contract_version",
    "contract_sha256",
    "implementation_manifest_sha256",
    "implementation_commit",
    "store_instance_id",
    "store_session_nonce_sha256",
    "attempt_id",
    "publication_intent_sha256",
    "authorization_operation_kind",
    "authorization_operation_sha256",
    "authorization_operation_ordinal",
    "worker_ownership_sha256",
    "pre_push_authorization_marker_key",
    "remote_observation_sha256",
    "tag_ref",
    "expected_tag_object_sha1",
    "authorization_status",
    "push_command_limit",
    "pre_push_authorization_sha256",
)
PUBLICATION_RECOVERY_INVOCATION_COMPLETION_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "completion_verifier_id",
    "contract_version",
    "contract_sha256",
    "implementation_manifest_sha256",
    "implementation_commit",
    "store_instance_id",
    "store_session_nonce_sha256",
    "attempt_id",
    "publication_intent_sha256",
    "recovery_invocation_start_sha256",
    "invocation_ordinal",
    "prior_recovery_completion_sha256",
    "pre_push_authorization_sha256",
    "completion_status",
    "outcome",
    "remote_observation_sha256",
    "push_command_count_upper_bound",
    "elapsed_seconds",
    "cumulative_recovery_seconds",
    "recovery_invocation_completion_sha256",
)
PUBLICATION_WORKER_OWNERSHIP_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "owner_verifier_id",
    "contract_version",
    "contract_sha256",
    "implementation_manifest_sha256",
    "implementation_commit",
    "store_instance_id",
    "store_session_nonce_sha256",
    "attempt_id",
    "publication_intent_sha256",
    "operation_kind",
    "operation_sha256",
    "owner_nonce_sha256",
    "owner_process_id",
    "owner_process_creation_filetime_hex",
    "job_object_name_sha256",
    "owner_mutex_name_sha256",
    "kill_on_parent_exit",
    "child_assignment_before_resume_required",
    "ownership_status",
    "worker_ownership_sha256",
)
PUBLICATION_WORKER_QUIESCENCE_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "quiescence_verifier_id",
    "contract_version",
    "contract_sha256",
    "implementation_manifest_sha256",
    "implementation_commit",
    "store_instance_id",
    "store_session_nonce_sha256",
    "attempt_id",
    "publication_intent_sha256",
    "worker_ownership_sha256",
    "verification_mode",
    "prior_owner_process_dead",
    "owner_mutex_unowned",
    "job_object_active_process_count",
    "recorded_git_ssh_processes_alive_count",
    "quiescence_status",
    "worker_quiescence_sha256",
)
TERMINALIZATION_CLAIM_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "claim_verifier_id",
    "contract_version",
    "contract_sha256",
    "implementation_manifest_sha256",
    "implementation_commit",
    "store_instance_id",
    "store_session_nonce_sha256",
    "attempt_id",
    "publication_intent_sha256",
    "publication_receipt_sha256",
    "terminalization_capability_nonce_sha256",
    "terminal_evidence_sha256",
    "terminal_status",
    "claim_status",
    "pre_claim_store_journal_sequence",
    "pre_claim_store_journal_tip_sha256",
    "terminalization_claim_sha256",
)
PUBLICATION_CONFLICT_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "contract_version",
    "contract_sha256",
    "implementation_manifest_sha256",
    "store_instance_id",
    "attempt_id",
    "publication_intent_sha256",
    "observation_operation_kind",
    "observation_operation_sha256",
    "tag_ref",
    "remote_observation_sha256",
    "conflict_reason",
    "poisoned",
    "prior_governance_record_sha256",
    "publication_conflict_sha256",
)
FINAL_REGISTRY_SUCCESSOR_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "contract_version",
    "contract_sha256",
    "branch",
    "implementation_commit",
    "predecessor_registry_sha256",
    "predecessor_registry_tip_sha256",
    "predecessor_reveal_count",
    "successor_ordinal",
    "final_attempt_id",
    "status",
    "successor_entry_sha256",
    "successor_registry_sha256",
)
FINAL_REGISTRY_AUTHORIZATION_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "verifier_id",
    "predecessor_registry_sha256",
    "predecessor_registry_tip_sha256",
    "predecessor_reveal_count",
    "successor_entry_sha256",
    "successor_registry_sha256",
    "external_publication_sha256",
    "final_attempt_id",
    "authorization_sha256",
)

DEVELOPMENT_BLOCKS: Final[tuple[tuple[str, str, str], ...]] = (
    ("block_1", "2005-01-03", "2007-12-31"),
    ("block_2", "2008-01-02", "2010-12-31"),
    ("block_3", "2011-01-03", "2013-12-31"),
    ("block_4", "2014-01-02", "2016-12-30"),
    ("block_5", "2017-01-03", "2018-12-31"),
)

MARKET_FEATURES: Final[tuple[str, ...]] = (
    "aapl_minus_qqq_log_return_20",
    "aapl_drawdown_63",
    "aapl_realized_volatility_20",
    "spy_log_return_20",
    "iwm_log_return_20",
    "vix_log_change_20",
)
MEANING_FEATURES: Final[tuple[str, ...]] = (
    "commercial_deterioration",
    "financial_deterioration",
    "risk_outlook_deterioration",
    "adverse_flag_fraction",
)
QUALITY_FEATURE: Final[str] = "semantic_quality_risk"
SEMANTIC_FEATURES: Final[tuple[str, ...]] = (
    *MEANING_FEATURES,
    QUALITY_FEATURE,
)
CONTROL_FEATURES: Final[tuple[str, ...]] = (
    "form_10k",
)
FEATURES: Final[tuple[str, ...]] = (
    *MARKET_FEATURES,
    *SEMANTIC_FEATURES,
    *CONTROL_FEATURES,
)


class SecGemmaOnlineRiskOverlayContractError(ValueError):
    """Raised when a purported contract differs from the preregistration."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical UTF-8 representation used for identity."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _prompt_sha256() -> str:
    return hashlib.sha256(EXTRACTOR_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def _schema_sha256() -> str:
    return canonical_sha256(build_extractor_json_schema())


def build_runtime_fingerprint_material() -> dict[str, Any]:
    """Return the exact local-model metadata whose literal hash is pinned."""

    return {
        "schema_version": "sec-gemma-v2-1-local-runtime-pin-v1",
        "model_name": MODEL_NAME,
        "ollama_version": OLLAMA_VERSION,
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "model_config_digest": MODEL_CONFIG_DIGEST,
        "model_layer_digests": list(MODEL_LAYER_DIGESTS),
        "version_response_sha256": (
            "2bd89ec9b983123a225f3df0381c737a45302bb7417e345bf9ef92304e4388cf"
        ),
        "show_semantic_sha256": (
            "5ccdf8b9a40bb762ea998dc9f5691ab2855d96833d60940b1d93686d08eb47e6"
        ),
        "show_semantic_excluded_keys": ["modified_at"],
        "model_info_sha256": (
            "d21c1c125758901fcea224a7cb9df1057aeba7ebb5177b82d6ba1a096d65fc7b"
        ),
    }


def build_contract_manifest() -> dict[str, Any]:
    """Build a detached copy of the exact frozen experiment contract."""

    manifest: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "branch": BRANCH_NAME,
        "status_at_preregistration": "unrun",
        "objective": {
            "asset": "AAPL",
            "benchmark": "same-ledger AAPL buy-and-hold",
            "allowed_target_exposures": [0, 1],
            "maximum_target_exposure": 1,
            "maximum_realized_exposure": 1,
            "shorting": False,
            "leverage": False,
            "borrowing": False,
            "negative_cash": False,
            "cash_interest": False,
            "paid_api_calls": 0,
        },
        "evidence_classification": {
            "globally_pristine": False,
            "reason": (
                "The repository has already inspected 2024 onward and the "
                "foundation model may contain pretraining knowledge through 2024; "
                "the inherited baseline also includes a retrospectively selected "
                "expert."
            ),
            "honest_claim": "candidate-specific chronological retrospective replay",
            "prospective_proof_required": True,
        },
        "data": {
            "sec": {
                "issuer_cik": "0000320193",
                "forms": ["10-K", "10-Q"],
                "amendments": False,
                "source": "official SEC submissions and archive primary documents",
                "corpus_stages": {
                    "development": ["2000-01-01", "2018-12-31"],
                    "intermediate_confirmation": ["2019-01-01", "2023-12-31"],
                    "final_live_style": ["2024-01-01", "2026-07-09"],
                },
                "availability_rule": (
                    "first complete NYSE session strictly after the latest "
                    "defensible acceptance, filing, or filing-change date"
                ),
                "minimum_development_corpus_filings_2000_2018": 72,
                "minimum_confirmation_filings": 19,
                "complete_metadata_eligible_universe_required": True,
                "detached_replay": {
                    "catalogue": (
                        "rebuild the exact official SEC catalogue, complete eligible "
                        "universe, source receipts, and hashes from the quarantined raw "
                        "official bytes with validate_detached_catalog_replay"
                    ),
                    "stage_content": (
                        "rebuild every selected primary-document byte artifact and "
                        "receipt with validate_detached_stage_content_replay"
                    ),
                    "universe_membership": (
                        "every selected filing accession, form, primary-document URL, "
                        "acceptance timestamp, and availability session must equal the "
                        "replayed official universe; caller-supplied metadata is never "
                        "an authority"
                    ),
                    "future_metadata_boundary": (
                        "a current SEC submissions response may contain records after "
                        "the active stage cutoff only inside the opaque acquisition "
                        "vault; no count, row, timestamp, accession, or derived value "
                        "from those records is released to scoring code"
                    ),
                },
            },
            "market": {
                "evidence_symbols": ["AAPL", "SPY", "QQQ", "IWM", "VIX", "TNX"],
                "feature_symbols": ["AAPL", "QQQ", "SPY", "IWM", "VIX"],
                "unused_but_bound_evidence_symbols": ["TNX"],
                "provider_symbol_mapping": {
                    "AAPL": "AAPL",
                    "SPY": "SPY",
                    "QQQ": "QQQ",
                    "IWM": "IWM",
                    "VIX": "^VIX",
                    "TNX": "^TNX",
                },
                "source_acquisition": {
                    "schema_version": (
                        "aapl-sec-gemma-online-overlay-yahoo-chart-v8-v1"
                    ),
                    "provider_family": (
                        "yahoo-finance-chart-v8-public-unauthenticated"
                    ),
                    "endpoint": (
                        "https://query1.finance.yahoo.com/v8/finance/chart"
                    ),
                    "transport": (
                        "owned HTTPS GET with normal certificate and hostname "
                        "verification; no proxy, cookie, authentication, redirect, "
                        "compression, alternate host, provider, or fallback"
                    ),
                    "fixed_user_agent": (
                        "LLM-memory-trading-agent/1.0 market-evidence "
                        "(no-auth; one-shot)"
                    ),
                    "request_order": ["AAPL", "SPY", "QQQ", "IWM", "VIX", "TNX"],
                    "request_count_each_stage": MAX_MARKET_REQUESTS_PER_STAGE,
                    "request_path": (
                        "endpoint + '/' + percent-encoded provider symbol"
                    ),
                    "query_items_in_order": [
                        ["period1", "stage_period1_utc"],
                        ["period2", "stage_period2_utc"],
                        ["interval", "1d"],
                        ["includePrePost", "false"],
                        ["includeAdjustedClose", "true"],
                        ["events", "div,splits"],
                    ],
                    "request_windows": {
                        "development": {
                            "start": "1998-01-01",
                            "end_exclusive": "2019-01-01",
                            "period1_utc": 883612800,
                            "period2_utc": 1546300800,
                            "last_eligible_session": "2018-12-31",
                        },
                        "confirmation": {
                            "start": "1998-01-01",
                            "end_exclusive": "2024-01-01",
                            "period1_utc": 883612800,
                            "period2_utc": 1704067200,
                            "last_eligible_session": "2023-12-29",
                        },
                        "final": {
                            "start": "1998-01-01",
                            "end_exclusive": "2026-07-11",
                            "period1_utc": 883612800,
                            "period2_utc": 1783728000,
                            "last_transport_session_quarantined": "2026-07-10",
                            "last_exposed_market_value_session": "2026-07-09",
                            "last_scored_fill_origin_decision_session": "2026-07-08",
                            "last_pending_prediction_session": "2026-07-09",
                        },
                    },
                    "provider_timezones": {
                        "AAPL": "America/New_York",
                        "SPY": "America/New_York",
                        "QQQ": "America/New_York",
                        "IWM": "America/New_York",
                        "VIX": "America/Chicago",
                        "TNX": "America/Chicago",
                    },
                    "timestamp_to_session": (
                        "convert each integer Unix timestamp to the frozen expected "
                        "provider timezone, require that its local date equals its UTC "
                        "date, then use that ISO date"
                    ),
                    "canonical_prefix_rule": (
                        "confirmation must reproduce every development canonical "
                        "session, explicit-absence marker, and float bit pattern; final "
                        "must reproduce the complete confirmation prefix identically. "
                        "Any back-adjustment, revision, omission, or addition inside a "
                        "sealed prefix terminally fails the branch"
                    ),
                    "private_raw_metadata_rule": (
                        "exact provider bytes enter a private quarantine because Yahoo "
                        "metadata can contain current quote fields; no raw metadata or "
                        "row beyond the stage value boundary is returned to the "
                        "experiment. The final 2026-07-10 transport row remains private"
                    ),
                    "opaque_vault_rule": (
                        "raw SEC and Yahoo bytes are owned by a durable capability-gated "
                        "quarantine process and are never returned as a public mapping "
                        "or object attribute; scoring receives only a canonical stage-"
                        "cutoff slice after its own attempt is durably consumed"
                    ),
                    "visibility_and_lock_rule": (
                        "development raw bytes may be acquired into quarantine before "
                        "the development lock but no price value may be exposed; "
                        "confirmation and final acquisition occur only after their "
                        "durable stage locks"
                    ),
                    "retry_and_selection_rule": (
                        "no HTTP retry and no alternate snapshot; the first complete "
                        "authenticated stage batch is sealed, while failed transport "
                        "receipts expose no values and cannot be used to choose a batch"
                    ),
                },
                "required_market_coverage": {
                    "aapl_exact_expected_session_counts": {
                        "development_through_2018_12_31": 5283,
                        "confirmation_through_2023_12_29": 6541,
                        "final_exposed_through_2026_07_09": 7172,
                        "final_transport_through_2026_07_10": 7173,
                    },
                    "context_first_accepted_session": {
                        "SPY": "1998-01-02",
                        "QQQ": "1999-03-10",
                        "IWM": "2000-05-26",
                        "VIX": "1998-01-02",
                        "TNX": "1998-01-02",
                    },
                    "context_allowed_missing_sessions": {
                        "SPY": [],
                        "QQQ": [],
                        "IWM": [],
                        "VIX": [],
                        "TNX": [
                            "1998-10-12",
                            "1998-11-11",
                            "1999-10-11",
                            "1999-11-11",
                            "2003-11-11",
                            "2005-10-10",
                            "2005-11-11",
                            "2006-10-09",
                            "2010-10-11",
                            "2016-11-11",
                        ],
                    },
                    "coverage_rule": (
                        "AAPL must contain every frozen expected session through the "
                        "stage boundary; each context symbol must contain every frozen "
                        "session from its first accepted session except only the listed "
                        "absences. A weekend, unexplained gap, missing last usable "
                        "session, or truncated 253-row tail terminally fails acquisition"
                    ),
                },
                "canonical_provider_fields": [
                    "raw_open",
                    "raw_high",
                    "raw_low",
                    "raw_close",
                    "raw_volume",
                    "adjusted_close",
                ],
                "ledger_price_fields": [
                    "raw_open",
                    "raw_close",
                    "adjusted_close",
                ],
                "ledger_price_validation": (
                    "on every exposed AAPL ledger session, raw_open, raw_close, and "
                    "adjusted_close must each exist exactly once and be finite and "
                    "strictly positive; adjusted_open = raw_open*adjusted_close/raw_close "
                    "must also be finite and strictly positive. Any failure terminally "
                    "fails the stage rather than making only one filing unavailable"
                ),
                "price_field": "adjusted_close",
                "feature_cutoff": "completed decision-session close",
                "fill": "next adjusted open",
                "horizon": {
                    "decision_close_index": "t",
                    "entry_open_index": "t+1",
                    "exit_open_index": "t+21",
                    "held_open_to_open_intervals": 20,
                    "label_maturity_session_index": "t+21",
                },
                "calendar": {
                    "calendar_id": "nyse_trading_session_dates_2000_01_01_2026_07_10_v2",
                    "calendar_dates_sha256": (
                        "e0550f12f98d7e0cf38d6797ae0d0e410bb3f9006793830ed75e0f753ccc5cf9"
                    ),
                    "market_calendar_id": (
                        "nyse_trading_session_dates_1998_01_01_2026_07_10_v1"
                    ),
                    "market_calendar_dates_sha256": (
                        "e9d37d63d158f8a3b6de58ef81970b27bcac9edb6a9c7b9e2f3ffe477d2f3032"
                    ),
                    "official_source_evidence_required_before_effects": True,
                },
            },
            "forbidden": [
                "GDELT event rows described as news",
                "future prices or outcomes in features",
                "post-decision filing revisions",
                "paid data or model APIs",
            ],
        },
        "gemma": {
            "role": "fixed evidence-grounded filing reader, never trader",
            "model_name": MODEL_NAME,
            "model_manifest_sha256": MODEL_MANIFEST_SHA256,
            "model_config_digest": MODEL_CONFIG_DIGEST,
            "model_layer_digests": list(MODEL_LAYER_DIGESTS),
            "ollama_version": OLLAMA_VERSION,
            "runtime_fingerprint_sha256": RUNTIME_FINGERPRINT_SHA256,
            "runtime_fingerprint_material": build_runtime_fingerprint_material(),
            "runtime_version_response_sha256": (
                "2bd89ec9b983123a225f3df0381c737a45302bb7417e345bf9ef92304e4388cf"
            ),
            "runtime_show_semantic_sha256": (
                "5ccdf8b9a40bb762ea998dc9f5691ab2855d96833d60940b1d93686d08eb47e6"
            ),
            "runtime_show_semantic_excluded_keys": ["modified_at"],
            "runtime_show_raw_sha256_diagnostic_only": (
                "5f56fb0fb2214ddcb9fa21c66aa31e37297f553e8758aeda5958f0f287d70893"
            ),
            "runtime_model_info_sha256": (
                "d21c1c125758901fcea224a7cb9df1057aeba7ebb5177b82d6ba1a096d65fc7b"
            ),
            "runtime_note": (
                "Gemma 4 exposes two active FROM blobs; v2.2 verifies the exact "
                "manifest and all four layer digests and replaces v2's unstable raw "
                "show-byte pin with one strict semantic show pin"
            ),
            "pre_call_identity_gate": (
                "before every semantic batch, hash the installed manifest bytes, "
                "verify the config and ordered layer digests, hash every layer's "
                "content, query strict version/show JSON, require the exact show key "
                "set and a string modified_at field, remove only modified_at, hash the "
                "canonical remaining show object, rebuild the fingerprint, and require "
                "semantic equality with these pins; the observed raw show hash remains "
                "diagnostic and whitespace, key order, or modified_at alone cannot fail"
            ),
            "endpoint": "http://127.0.0.1:11434/api/chat",
            "loopback_only": True,
            "model_pull": False,
            "temperature": 0,
            "seed": 0,
            "context_tokens": 6144,
            "output_tokens": 512,
            "retries": 0,
            "repairs": 0,
            "prompt_sha256": _prompt_sha256(),
            "schema_version": EXTRACTOR_SCHEMA_VERSION,
            "schema_sha256": _schema_sha256(),
            "input": (
                "issuer/date/market-blinded current and prior same-form filing "
                "sentences only"
            ),
            "forbidden_inputs": [
                "ticker or issuer identity",
                "exact dates",
                "market prices or returns",
                "labels, actions, scores, or benchmark results",
            ],
            "invalid_output_policy": "neutral semantics plus quality risk; no retry",
            "source_pins": {
                role: {"file": SOURCE_PIN_FILES[role], "sha256": digest}
                for role, digest in sorted(SOURCE_PINS.items())
            },
            "new_v2_2_sources": dict(sorted(NEW_SOURCE_FILES.items())),
            "source_inventory_rule": (
                "the implementation manifest must bind every exact new role/path plus "
                "every literal inherited source pin; dependency-closed local imports "
                "from production, acquisition, runtime, publisher, registry, vault, "
                "runner, and transports may not escape this inventory"
            ),
            "allowed_external_python_distributions": [
                "requests",
                "urllib3",
                "certifi",
                "charset-normalizer",
                "idna",
            ],
            "external_distribution_rule": (
                "if a listed distribution is imported by the production transport, "
                "the clean implementation receipt must bind its installed version, "
                "direct_url metadata when present, and SHA-256 of every imported module "
                "file before registration; no unlisted distribution may be imported"
            ),
        },
        "features": {
            "ordered_names": list(FEATURES),
            "count": len(FEATURES),
            "market_names": list(MARKET_FEATURES),
            "semantic_names": list(SEMANTIC_FEATURES),
            "meaning_names": list(MEANING_FEATURES),
            "quality_name": QUALITY_FEATURE,
            "control_names": list(CONTROL_FEATURES),
            "market_formulas": {
                "aapl_minus_qqq_log_return_20": (
                    "log(AAPL_adj_close[t]/AAPL_adj_close[t-20]) - "
                    "log(QQQ_adj_close[t]/QQQ_adj_close[t-20])"
                ),
                "aapl_drawdown_63": (
                    "AAPL_adj_close[t]/max(AAPL_adj_close[t-62:t inclusive])-1"
                ),
                "aapl_realized_volatility_20": (
                    "sample_std_ddof_1(log(AAPL_adj_close[i]/AAPL_adj_close[i-1]) "
                    "for i=t-19..t)*sqrt(252)"
                ),
                "spy_log_return_20": (
                    "log(SPY_adj_close[t]/SPY_adj_close[t-20])"
                ),
                "iwm_log_return_20": (
                    "log(IWM_adj_close[t]/IWM_adj_close[t-20])"
                ),
                "vix_log_change_20": (
                    "log(VIX_adj_close[t]/VIX_adj_close[t-20])"
                ),
            },
            "market_missingness": (
                "any absent, duplicate, nonfinite, or nonpositive required adjusted "
                "close makes the event unavailable; no imputation or row drop"
            ),
            "semantic_groups": {
                "commercial": ["demand", "pricing_power", "supply_chain"],
                "financial": [
                    "gross_margin",
                    "operating_cost_pressure",
                    "capital_allocation",
                    "liquidity",
                ],
                "risk_outlook": [
                    "forward_guidance",
                    "legal_regulatory",
                    "management_uncertainty",
                ],
            },
            "semantic_encoding": {
                "current_impact": {
                    "favorable": -1,
                    "neutral": 0,
                    "unfavorable": 1,
                    "mixed": 0,
                    "not_stated": 0,
                },
                "change_vs_prior": {
                    "improving": -1,
                    "stable": 0,
                    "deteriorating": 1,
                    "mixed": 0,
                    "not_comparable": 0,
                    "not_stated": 0,
                },
                "dimension_score": "mean(current_impact, change_vs_prior)",
                "group_score": "fixed-denominator mean of member dimension scores",
                "adverse_flags": [
                    "new_material_risk",
                    "guidance_withdrawn",
                    "liquidity_stress",
                    "restructuring_or_impairment",
                    "internal_control_weakness",
                ],
                "semantic_quality_risk": {
                    "usable": 0,
                    "thin": 0.5,
                    "unusable_or_invalid": 1,
                },
                "adverse_flag_fraction": "sum(present for five adverse flags)/5",
            },
            "no_filing_meaning_ablation": (
                "a separate causal arm with identical market, form, extraction-quality, "
                "event, and label rows; set only the four filing-meaning features to "
                "zero and preserve semantic_quality_risk exactly"
            ),
            "no_gemma_channel_diagnostic": (
                "a separately reported diagnostic arm that zeros all five Gemma-derived "
                "features; it cannot satisfy or rescue any filing-meaning success gate"
            ),
            "schema_valid_extraction_rate": (
                "authenticated exact-schema-valid Gemma outputs divided by every "
                "metadata-eligible event requiring a call; missing, unauthenticated, "
                "and schema-invalid outputs remain in the denominator"
            ),
            "extraction_coverage_stage_assignment": (
                "coverage and nonzero-meaning counts use only events whose filing "
                "decision session lies in that development, confirmation, or final "
                "stage; earlier cumulative rows cannot satisfy a later-stage gate"
            ),
            "nonzero_meaning_row": (
                "an authenticated schema-valid row with absolute value above 1e-12 in "
                "at least one of the four filing-meaning features"
            ),
        },
        "event_availability": {
            "eligible_universe_rule": (
                "every metadata-eligible filing receives exactly one chronological "
                "audit row; availability status may not remove or reorder the row"
            ),
            "missing_primary_document_before_attempt": (
                "failure of the acquisition prerequisite: no scored attempt begins "
                "until every eligible primary document and provenance record for the "
                "stage is authenticated and sealed"
            ),
            "first_same_form_filing": (
                "an authenticated current filing with no earlier same-form filing is "
                "available; prior-change fields are not_comparable and encode as zero, "
                "while current-impact fields and current extraction quality remain live"
            ),
            "authenticated_schema_invalid_output": (
                "available and trainable when market inputs exist: set the four meaning "
                "features to zero, semantic_quality_risk to 1, preserve form_10k, seal "
                "the invalid-response receipt, and do not retry or repair. The primary "
                "meaning arm and quality-preserving no-meaning ablation are therefore "
                "identical on this row"
            ),
            "missing_or_unauthenticated_model_output": (
                "event unavailable: seal an audit row, take no SEC-overlay action, make "
                "no prediction, and exclude it from learner membership, Brier support, "
                "episodes, and action-difference counts; no retry"
            ),
            "missing_or_unauthenticated_feature_provenance": (
                "event unavailable under the same exclusions; a whole-batch model or "
                "runtime identity failure terminally fails the consumed stage"
            ),
            "market_unavailable": (
                "event unavailable under the same exclusions when any required market "
                "input is absent, duplicate, nonfinite, or nonpositive; no imputation"
            ),
            "unavailable_label_rule": (
                "a counterfactual label may still mature and is retained as audit-only, "
                "but it can never train either semantic or ablation learner without the "
                "complete immutable decision-time feature row"
            ),
            "learner_unready": (
                "an otherwise available event is retained for later training but emits "
                "no fitted prediction and schedules no overlay; it is outside Brier and "
                "action-difference support until both heads satisfy readiness"
            ),
            "active_overlay_event": (
                "compute and seal its available prediction normally, then force the "
                "effective schedule flag false under non-overlap; its label still "
                "matures and can train future decisions"
            ),
            "undefined_or_nonfinite_metric": "terminal stage failure",
        },
        "learner": {
            "training_mode": "continuous expanding causal refit before each filing",
            "training_rows": (
                "all and only earlier filing labels with maturity_session <= "
                "current_decision_session; the t+21 exit open on the current "
                "session is known before that session's close"
            ),
            "same_session_matured_label_admission": "maturity_session <= decision_session",
            "counterfactual_lessons": True,
            "minimum_training_rows": MINIMUM_TRAINING_ROWS,
            "minimum_rows_per_binary_class": MINIMUM_CLASS_ROWS,
            "unready_action": "no SEC overlay; inherited baseline still trades",
            "label": (
                "counterfactual incremental 10bps log edge of activating the "
                "20-session overlay versus the fixed baseline-only ledger"
            ),
            "probability_head": (
                "robust-scaled ridge logistic for overlay beating fixed baseline"
            ),
            "edge_head": (
                "robust-scaled ridge Huber for incremental edge versus baseline"
            ),
            "learner_config": {
                "raw_mad_multiplier": 1.4826,
                "raw_scale_floor": 1e-6,
                "raw_z_clip": 4,
                "ridge_lambda": 0.1,
                "intercept_regularized": False,
                "logistic_initial_coefficients": (
                    "intercept=logit(training prevalence), all slopes=0"
                ),
                "logistic_max_iterations": 50,
                "logistic_tolerance": 1e-10,
                "newton_line_search": (
                    "Armijo constant 1e-4, halving from step 1"
                ),
                "newton_line_search_max_steps": 50,
                "huber_delta": 1.5,
                "huber_initial_coefficients": (
                    "intercept=median(standardized clipped edge), all slopes=0"
                ),
                "huber_max_iterations": 50,
                "huber_tolerance": 1e-10,
                "target_mad_multiplier": 1.4826,
                "target_scale_floor": 1e-6,
                "edge_clip": [-0.5, 0.5],
                "persisted_floats": "canonical float.hex",
            },
            "action_gate": {
                "probability_at_least": PROBABILITY_GATE,
                "expected_incremental_10bps_log_edge_at_least": EXPECTED_EDGE_GATE,
                "candidate_grid": False,
            },
        },
        "policy": {
            "baseline_policy_id": BASELINE_POLICY_ID,
            "baseline_source_file": BASELINE_SOURCE_FILE,
            "baseline_source_sha256": BASELINE_SOURCE_SHA256,
            "baseline_kind": "fixed raw expert union, not binary-regime selector",
            "baseline_signal_column": "unfiltered_union_signal",
            "baseline_forbidden_column": "unfiltered_union_target_exposure",
            "baseline_parameters": {
                "intraday_return": "AAPL_close/AAPL_open-1",
                "intraday_prior_percentile_lookback": 126,
                "contextual_percentile": 0.90,
                "contextual_spy_qqq_return_lookback": 10,
                "weak_trend_percentile": 0.925,
                "weak_trend_spy_qqq_return_lookback": 20,
                "weak_trend_aapl_sma_lookback": 20,
                "contextual_condition": (
                    "intraday above shifted prior percentile AND SPY and QQQ "
                    "lookback returns both below zero"
                ),
                "weak_trend_condition": (
                    "intraday above shifted prior percentile AND SPY and QQQ "
                    "lookback returns both below zero AND AAPL adjusted close "
                    "below its 20-session simple moving average"
                ),
                "composition": (
                    "canonicalize each expert's one-session signals; OR them; "
                    "canonicalize the union once over the continuous prefix"
                ),
            },
            "baseline_prefix_invariance": (
                "a completed-close raw union signal is persisted even when t+1 or "
                "t+2 lies beyond the current physical price prefix; appending rows "
                "may fill its pending action but may not rewrite the decision"
            ),
            "baseline_action": (
                "signal at close t schedules CASH at open t+1 and LONG at open t+2"
            ),
            "baseline_cooldown_independence": (
                "baseline canonicalization and cooldown evolve from baseline signals "
                "only; an SEC overlay never suppresses, resets, or creates a baseline "
                "signal"
            ),
            "overlay_horizon_sessions": HORIZON_SESSIONS,
            "overlay_entry": "next adjusted open after the filing decision close",
            "overlay_exit": "t+21 adjusted open after 20 open-to-open intervals",
            "active_overlay_extension": False,
            "overlapping_overlay": False,
            "ignored_signals_still_mature_as_lessons": True,
            "combined_cash": "baseline_cash OR active_sec_overlay",
            "overlap_execution": (
                "target exposure is recomputed at every open; a fill and its cost "
                "occur only when combined target exposure changes. A baseline signal "
                "inside an active overlay has no fill but still advances the independent "
                "baseline state. At overlay exit, remain CASH without a fill when the "
                "baseline is CASH at that open; otherwise buy AAPL once."
            ),
            "lesson_counterfactual": (
                "for every eligible filing, fork the exact baseline-only cash/share/"
                "cooldown state immediately before open t+1 into baseline-only and "
                "forced-overlay arms. Ignore all other SEC overlays in both arms, "
                "continue identical future baseline signals, apply actual changing-leg "
                "costs, and compare normalized wealth at open t+21."
            ),
            "costs_bps_per_changing_leg": [5, 10],
            "same_action_stream_at_both_costs": True,
        },
        "ledger": {
            "initial_capital_usd": 1000,
            "initial_exposure_before_genesis_open": 0,
            "genesis_target_for_strategy_and_benchmark": 1,
            "adjusted_open_formula": "raw_open * adjusted_close / raw_close",
            "held_long_return_factor": "adjusted_open[current]/adjusted_open[prior]",
            "held_cash_return_factor": 1,
            "buy_cost_factor": "1/(1+cost_bps/10000)",
            "sell_cost_factor": "1-cost_bps/10000",
            "buy_fill_arithmetic": (
                "shares = pre_fill_cash / (adjusted_open * "
                "(1+cost_bps/10000)); cash = 0"
            ),
            "sell_fill_arithmetic": (
                "cash = pre_fill_shares * adjusted_open * "
                "(1-cost_bps/10000); shares = 0"
            ),
            "fill_order": (
                "earn the prior target's open-to-open return first, then execute the "
                "current open's target-changing fill and cost"
            ),
            "same_prices_and_initial_purchase_as_benchmark": True,
            "stage_boundaries": (
                "carry exact cash, shares, exposure, prior adjusted open, pending fills, "
                "and active overlay; no synthetic trade or capitalization reset"
            ),
            "terminal_valuation": ["adjusted_open", "terminal_adjusted_close"],
            "adjusted_open_valuation": "cash + shares * adjusted_open",
            "terminal_adjusted_close_valuation": (
                "cash + shares * terminal_adjusted_close; no synthetic liquidation "
                "or terminal transaction cost"
            ),
            "realized_exposure_formula": (
                "0 when shares=0; otherwise shares*valuation_price/wealth = 1 within "
                "1e-12, with finite positive wealth and cash >= 0"
            ),
            "cash_interest": False,
            "fractional_shares": True,
            "debt": False,
        },
        "chronology": {
            "portfolio_genesis": "2000-01-03",
            "warmup_and_learning": ["2000-01-03", "2004-12-31"],
            "development_corpus": ["2000-01-01", "2018-12-31"],
            "development_qualification": ["2005-01-03", "2018-12-31"],
            "development_blocks": [
                {"id": item[0], "first": item[1], "last": item[2]}
                for item in DEVELOPMENT_BLOCKS
            ],
            "confirmation": ["2019-01-01", "2023-12-31"],
            "final_live_style_replay": ["2024-01-01", "2026-07-09"],
            "final_cutoff_rule": (
                "2026 YTD performance and both terminal valuations end on 2026-07-09. "
                "Only decisions through 2026-07-08 can create scored fills. A filing "
                "decision after the 2026-07-09 close is sealed with its pending t+1 "
                "action and lesson but contributes no 2026-07-10 price, fill, return, "
                "episode, or action difference to this audit"
            ),
            "updates_inside_every_period": (
                "permitted only after the complete 20-session outcome matures"
            ),
            "every_2025_decision_state": (
                "immediately before each filing, admit every earlier eligible label "
                "with maturity_session <= that filing's decision_session; no annual "
                "cutoff freezes learning"
            ),
            "same_session_event_order": (
                "on one decision session, first admit all newly matured labels once, "
                "then process filings by exact SEC acceptance timestamp ascending and "
                "accession ascending; no outcome can enter between same-session filings"
            ),
            "same_close_pending_overlay_rule": (
                "the first ordered filing that passes may reserve the next-open overlay; "
                "that scheduled overlay blocks every later same-close filing from "
                "scheduling or extending another overlay, although each still receives "
                "an audit prediction and later lesson"
            ),
            "controls": {
                "semantic_arm_continuity": (
                    "the full semantic, quality-preserving no-meaning, and all-five-zero "
                    "diagnostic arms each start at portfolio genesis with an independent "
                    "causal learner, cash/share account, baseline cooldown, pending "
                    "fills, and overlay non-overlap state; all are carried without reset "
                    "through every block and stage"
                ),
                "semantic_arm_labels": (
                    "all semantic arms receive the same matured counterfactual labels; "
                    "only their frozen decision-time feature transforms differ"
                ),
                "frozen_state_contents": (
                    "exact fitted coefficients, scalers, training membership and "
                    "counts; pending pre-fork labels remain audit-only and never refit "
                    "the frozen control after the fork"
                ),
                "fork_boundary_order": (
                    "fork immediately after all labels with maturity_session on or "
                    "before the preceding session have been admitted, and before any "
                    "label maturing on the boundary session or any boundary-session "
                    "filing decision is processed"
                ),
                "development": (
                    "at the start of each block's first session under fork_boundary_order, "
                    "fork the primary account, active/scheduled overlay, baseline state, "
                    "and learner state. The "
                    "control sees later causal features but never admits another label; "
                    "it is discarded after that block and never alters the primary."
                ),
                "confirmation": (
                    "fork the exact continuous primary account and through-2018 learner "
                    "state before any 2019-session admission or decision; preserve any "
                    "scheduled/active overlay; "
                    "never update the control during confirmation"
                ),
                "final": (
                    "fork the exact continuous primary account and through-2023 learner "
                    "state before any 2024-session admission or decision; preserve any "
                    "scheduled/active overlay; "
                    "never update the control during the final replay"
                ),
            },
            "primary_account_continuity": (
                "one account from genesis; no cash/share/position/model reset and no "
                "synthetic boundary fill"
            ),
            "no_reset_at_stage_or_year_boundary": True,
            "design_changes_after_development": False,
        },
        "metric_definitions": {
            "positive_tolerance": POSITIVE_EDGE_TOLERANCE,
            "positive_edge": "active log edge > 1e-12",
            "negative_edge": "active log edge < -1e-12",
            "tie": "absolute active log edge <= 1e-12",
            "reporting_interval_assignment": (
                "an adjusted-open return interval and any target-changing fill cost at "
                "its destination open belong to that destination session; the common "
                "genesis purchase belongs to 2000-01-03. The terminal-close factor, "
                "when reported, belongs to the final evidence session"
            ),
            "reporting_support": {
                "development_years": list(range(2005, 2019)),
                "confirmation_years": list(range(2019, 2024)),
                "final_periods": ["2024", "2025", "2026_ytd_through_2026-07-09"],
                "warmup_excluded_from_qualification_gates": list(range(2000, 2005)),
            },
            "adjusted_open_report_variant": (
                "normalize each continuous arm at the boundary immediately before the "
                "first destination-open interval assigned to the report; apply every "
                "open-to-open return and fill cost assigned through the report's final "
                "session open, with no reset or liquidation"
            ),
            "terminal_adjusted_close_report_variant": (
                "start from the adjusted-open report variant and append exactly one "
                "same-session factor at that report window's final date: adjusted_close/"
                "adjusted_open for an arm long after the final-open fill, or 1 for cash. "
                "Do not carry this diagnostic close valuation into the continuous "
                "account or the next report window and do not charge a liquidation cost"
            ),
            "maximum_drawdown": (
                "within the exact scored report window, begin with normalized wealth 1 "
                "immediately before its first assigned interval; sample wealth after "
                "each destination-open return and fill. For the terminal-close variant "
                "append only its one final-close observation. At each observation peak "
                "is the maximum wealth seen including the initial 1; MDD = min(wealth/"
                "running_peak - 1), so MDD is finite and nonpositive"
            ),
            "drawdown_comparison": (
                "at both 5bps and 10bps and under each terminal variant, strategy_MDD "
                ">= same-ledger_AAPL_MDD - 0.01 over the continuous final window"
            ),
            "active_log_edge": (
                "sum(log(strategy_net_return_factor) - "
                "log(same-ledger_AAPL_net_return_factor)) over exact open intervals"
            ),
            "incremental_log_edge": (
                "sum(log(left_policy_net_return_factor) - "
                "log(right_control_net_return_factor)) over identical intervals"
            ),
            "negative_aapl_year": (
                "same-ledger AAPL calendar-year log return < -1e-12; a strategy win "
                "requires calendar-year active log edge > 1e-12"
            ),
            "action_difference": (
                "one eligible filing where effective schedule_overlay booleans differ "
                "after readiness, threshold, and each arm's non-overlap state; exclude "
                "a final-cutoff prediction whose next-open action remains pending"
            ),
            "difference_block": (
                "one declared development block containing at least one action "
                "difference assigned by filing decision session; count each block once"
            ),
            "semantic_difference_year": (
                "one calendar year containing at least one full-semantic versus "
                "quality-preserving no-meaning action difference assigned by filing "
                "decision session; count each year once"
            ),
            "xor_interval": (
                "one maximal contiguous adjusted-open ledger interval where the two "
                "actual compared target exposures differ; its contribution is the sum "
                "of their daily net log-return difference including changing-leg costs"
            ),
            "complete_xor_interval": (
                "an XOR interval whose unequal exposure starts and returns to equality "
                "inside the scored physical window"
            ),
            "overlay_episode": (
                "one accepted semantic filing overlay from its t+1 entry open through "
                "its t+21 exit open; incremental contribution is combined-policy minus "
                "baseline-only net log return over that exact interval"
            ),
            "episode_win_rate": (
                "strictly positive complete contributions divided by all complete "
                "contributions; zero and negative contributions are non-wins, and an "
                "empty denominator is undefined and fails every dependent gate"
            ),
            "open_boundary_episode_rule": (
                "carried or still-open contributions enter aggregate ledger edge, but "
                "only episodes/intervals with both fill boundaries inside available "
                "data enter complete-count, win-rate, median, and concentration gates"
            ),
            "best_block_removal": (
                "total edge minus max(block edge); block edges partition the scored "
                "open intervals exactly, and a removed block is never refit or replayed"
            ),
            "best_episode_removal": (
                "total incremental edge minus the largest strictly positive complete "
                "episode or XOR contribution; absence of a positive complete item fails"
            ),
            "positive_concentration": (
                "largest strictly positive complete contribution divided by the sum "
                "of all strictly positive complete contributions"
            ),
            "brier_support": (
                "identical eligible filing decisions with available semantic and "
                "no-filing-meaning predictions and a t+21 label matured by the report "
                "cutoff; binary target is incremental overlay edge > 1e-12"
            ),
            "brier_relative_improvement": (
                "(ablation_brier - semantic_brier) / max(ablation_brier, 1e-12)"
            ),
            "comparison_fork_rule": (
                "event-label counterfactual arms fork immediately before that event's "
                "t+1 open; block/stage frozen-learning controls fork only at their exact "
                "declared chronology boundary. Semantic arms never fork after genesis, "
                "and reporting windows only normalize carried wealth rather than "
                "creating a new account"
            ),
        },
        "gates": {
            "development": {
                "combined_total_active_log_edge_10bps_at_least": 0.02,
                "combined_edge_without_best_block_10bps_at_least": 0.005,
                "positive_combined_blocks_10bps_at_least": 4,
                "annual_win_rate_10bps_at_least": 0.55,
                "negative_aapl_year_win_rate_10bps_at_least": 0.60,
                "complete_sec_overlay_episodes_at_least": 12,
                "overlay_episode_win_rate_10bps_at_least": 0.55,
                "overlay_median_edge_10bps_strictly_positive": True,
                "largest_positive_episode_share_10bps_at_most": 0.35,
                "schema_valid_extraction_rate_at_least": 0.90,
                "nonzero_filing_meaning_rows_at_least": 24,
                "incremental_vs_baseline_10bps_at_least": 0.005,
                "incremental_vs_baseline_without_best_block_10bps_strictly_positive": True,
                "online_vs_block_frozen_action_differences_at_least": 5,
                "online_vs_block_frozen_difference_blocks_at_least": 3,
                "online_vs_block_frozen_10bps_edge_strictly_positive": True,
                "semantic_vs_no_filing_meaning_action_differences_at_least": 5,
                "semantic_vs_no_filing_meaning_difference_blocks_at_least": 3,
                "semantic_vs_no_filing_meaning_complete_xor_intervals_at_least": 4,
                "semantic_vs_no_filing_meaning_10bps_edge_at_least": 0.005,
                "semantic_edge_without_best_xor_10bps_strictly_positive": True,
                "semantic_brier_relative_improvement_at_least": 0.01,
            },
            "confirmation": {
                "combined_active_log_edge_positive_at_5_and_10bps": True,
                "combined_positive_years_at_least_3_at_both_5_and_10bps": True,
                "schema_valid_extraction_rate_at_least": 0.90,
                "nonzero_filing_meaning_rows_at_least": 6,
                "incremental_vs_baseline_10bps_at_least": 0.0025,
                "incremental_without_best_episode_10bps_strictly_positive": True,
                "semantic_vs_no_filing_meaning_action_differences_at_least": 3,
                "semantic_difference_years_at_least": 2,
                "semantic_vs_no_filing_meaning_10bps_edge_at_least": 0.0025,
                "online_vs_frozen_action_differences_at_least": 2,
                "online_vs_frozen_10bps_edge_strictly_positive": True,
            },
            "final": {
                "active_edge_5bps_each_2024_2025_2026_ytd_at_least": 0.005,
                "continuous_active_edge_5bps_at_least": 0.02,
                "active_edge_positive_each_period_at_10bps": True,
                "schema_valid_extraction_rate_at_least": 0.90,
                "nonzero_filing_meaning_rows_at_least": 3,
                "incremental_vs_baseline_continuous_10bps_strictly_positive": True,
                "incremental_vs_baseline_positive_periods_10bps_at_least": 2,
                "semantic_vs_no_filing_meaning_action_differences_at_least": 2,
                "semantic_vs_no_filing_meaning_continuous_10bps_strictly_positive": True,
                "online_vs_frozen_continuous_10bps_strictly_positive": True,
                "post_2023_online_vs_frozen_action_differences_at_least": 3,
                "complete_sec_overlay_episodes_at_least": 6,
                "overlay_episode_win_rate_10bps_at_least": 0.55,
                "largest_positive_episode_share_10bps_at_most": 0.50,
                "max_drawdown_not_worse_than_aapl_by_more_than_at_5_and_10bps": 0.01,
                "terminal_valuation_methods_required": [
                    "adjusted_open",
                    "terminal_adjusted_close",
                ],
                "all_return_edge_and_drawdown_gates_pass_under_both_terminal_valuations": True,
                "undefined_metric_fails": True,
            },
            "zero_action_difference_is_rejection": True,
            "undefined_or_nonfinite_metric_fails_every_dependent_gate": True,
            "failure_blocks_next_stage": True,
        },
        "execution_integrity": {
            "production_authorities_required_before_registration": [
                "source-bound parent deadline guard",
                "reviewed exact SEC transport and acquisition adapter",
                "reviewed owned Yahoo transport",
                "durable opaque quarantine repository",
                "source-bound Gemma and deterministic stage executor",
                "non-force external Git-tag report publisher",
                "verified final-registry successor authorizer",
            ],
            "test_double_boundary": (
                "fake, injected, foreign, subclassed, or test-only transports, stores, "
                "executors, publishers, vaults, clocks, and registry authorizers may "
                "exercise local tests but can never register or consume a production "
                "attempt"
            ),
            "production_source_binding": (
                "every authority is an exact source-inventory role verified from the "
                "clean pushed implementation commit immediately before registration "
                "and again immediately before consumption"
            ),
            "opaque_types": {
                "acquisition_validation": "VerifiedAcquisitionReport",
                "acquisition_terminal": "VerifiedAcquisitionTerminalEvidence",
                "scored_terminal": "VerifiedScoredTerminalEvidence",
                "terminal_reconstruction_material": (
                    "VerifiedTerminalReconstructionMaterial"
                ),
                "terminal_reconstruction_material_receipt": (
                    "VerifiedTerminalReconstructionMaterialReceipt"
                ),
                "publication_intent": "VerifiedPublicationIntent",
                "external_publication": "VerifiedExternalPublication",
                "publication_remote_observation": (
                    "VerifiedDurablePublicationRemoteObservation"
                ),
                "publication_receipt": "VerifiedDurablePublicationReceipt",
                "publication_capability": "PublicationRecoveryCapability",
                "terminalization_capability": "TerminalizationCapability",
                "terminalization_claim": "VerifiedTerminalizationClaim",
                "publication_worker_ownership": (
                    "VerifiedPublicationWorkerOwnership"
                ),
                "publication_worker_quiescence": (
                    "VerifiedPublicationWorkerQuiescence"
                ),
                "final_registry": "VerifiedFinalRegistryAuthorization",
            },
            "cross_version_boundary": (
                "every experiment state schema, verifier ID, state namespace, receipt, "
                "attempt, and tag namespace is v2-2; the byte-identical inherited local "
                "runtime fingerprint alone retains its v2-1 schema label as immutable "
                "model-identity evidence, and every v2 or v2.1 experiment artifact is "
                "rejected rather than upgraded, translated, or re-emitted"
            ),
            "acquisition_terminal_pass": {
                "authority": (
                    "only the opaque VerifiedAcquisitionReport returned by a complete "
                    "detached replay of the currently sealed durable quarantine"
                ),
                "validation_fields": list(ACQUISITION_VALIDATION_FIELDS),
                "exact_checks": list(ACQUISITION_VALIDATION_CHECKS),
                "terminal_evidence_fields": list(
                    ACQUISITION_TERMINAL_EVIDENCE_FIELDS
                ),
                "digest_fields_are_strict_lowercase_sha256": True,
                "stale_report_after_vault_mutation_fails": True,
                "store_artifact_receipt_and_payload_hash_must_exist": True,
                "terminal_anchor_binds_entire_evidence_and_publication": True,
                "arbitrary_all_true_mapping_forbidden": True,
            },
            "scored_terminal_pass": {
                "authority": (
                    "only an opaque scored-stage verification rebuilt from the sealed "
                    "chronological replay, exact deterministic stage metrics, exact "
                    "literal stage gate report, independent no-leverage proofs, current "
                    "store record commitment, and sealed joint report"
                ),
                "exact_gate_key_set": (
                    "must equal the literal keys under gates[stage], with no missing, "
                    "extra, renamed, truncated, or caller-selected checks"
                ),
                "terminal_evidence_fields": list(
                    SCORED_TERMINAL_EVIDENCE_FIELDS
                ),
                "every_literal_gate_true": True,
                "joint_report_hash_and_store_artifact_receipt_bound": True,
                "terminal_anchor_binds_entire_evidence_and_publication": True,
                "arbitrary_all_true_mapping_forbidden": True,
            },
            "failed_scored_gate": (
                "persist the complete deterministic evaluation, stage metrics, gate "
                "report, no-leverage proofs, and joint report; append the exact joint "
                "artifact to the store, externally pin its hash, terminally fail the "
                "attempt, then release only that sealed diagnostic. Never convert it "
                "to pass, retry it, hide it, or advance to the next stage. finish_attempt "
                "must receive opaque VerifiedScoredTerminalEvidence and bind the joint "
                "artifact receipt, exact failed gate names, and publication receipt in "
                "the terminal anchor; invalid or partial evaluation releases nothing"
            ),
            "external_report_pin": {
                "transport": (
                    "one annotated Git tag pushed without force through the frozen "
                    "config-isolated literal remote URL"
                ),
                "git_object_format": "sha1",
                "ordinary_terminal_ref_template": EXTERNAL_TAG_REF_TEMPLATE,
                "final_registry_ref_template": (
                    FINAL_REGISTRY_TAG_REF_TEMPLATE
                ),
                "stable_ref_rule": (
                    "each ordinary attempt has exactly one stable terminal ref that is "
                    "independent of report kind, terminal status, and artifact hash, so "
                    "forked or copied stores cannot publish contradictory terminal "
                    "claims under distinct refs; the final-registry successor uses its "
                    "separate stable registry ref because the final ordinary result "
                    "shares the final attempt ID"
                ),
                "report_kinds": [
                    "acquisition_pass",
                    "scored_pass",
                    "scored_failed_gate",
                    "final_registry_successor",
                ],
                "tag_target": "the clean pushed implementation commit",
                "tag_message_fields": list(EXTERNAL_TAG_MESSAGE_FIELDS),
                "publication_fields": list(EXTERNAL_PUBLICATION_FIELDS),
                "opaque_receipt_type": "VerifiedExternalPublication",
                "remote_confirmation": (
                    "the exact remote annotated-tag object SHA-1 and peeled target are "
                    "read back and stored before the terminal transition"
                ),
                "bare_hash_or_string_return_forbidden": True,
                "deletion_force_or_reuse_forbidden": True,
            },
            "publication_worker_containment": {
                "scope": [
                    "normal_publication",
                    "publication_recovery",
                ],
                "os_mechanism": (
                    "every Git or SSH publication subprocess and descendant runs in a "
                    "dedicated Windows Job Object configured with "
                    "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; the source-bound parent is the "
                    "only process retaining the controlling job handle, every child is "
                    "created suspended, assigned before resume, and forbidden to break "
                    "away, so parent exit kills the complete tree"
                ),
                "normal_operation_sha256": PUBLICATION_NORMAL_OPERATION_SHA256,
                "ownership_record": {
                    "fields": list(PUBLICATION_WORKER_OWNERSHIP_FIELDS),
                    "journal_record_table": "publication_worker_ownerships",
                    "record_identity_fields": [
                        "store_instance_id",
                        "attempt_id",
                        "operation_sha256",
                        "owner_nonce_sha256",
                    ],
                    "prepare_anchor_event": "publication_worker_owner_prepare",
                    "committed_anchor_event": (
                        "publication_worker_owner_committed"
                    ),
                    "fixed_values": {
                        "schema_version": (
                            "sec-gemma-online-risk-overlay-v2-2-"
                            "publication-worker-ownership-v1"
                        ),
                        "owner_verifier_id": (
                            "sec-gemma-online-risk-overlay-v2-2-"
                            "publication-worker-owner-verifier-v1"
                        ),
                        "kill_on_parent_exit": True,
                        "child_assignment_before_resume_required": True,
                        "ownership_status": "claimed",
                    },
                    "operation_identity_rule": (
                        "normal publication uses the frozen normal-operation sentinel; "
                        "recovery publication uses its exact committed recovery start hash"
                    ),
                    "before_child_rule": (
                        "create the empty kill-on-close job and exclusive owner mutex, "
                        "then fsync and read back the exact self-hashed ownership row and "
                        "both anchors before creating any Git, SSH, or helper child"
                    ),
                    "exact_reconciliation": (
                        "the prepare anchor stores the exact canonical ownership row bytes; "
                        "no child may exist in a partial state, prepare without a row "
                        "appends only that row and committed anchor, exact row without the "
                        "committed anchor appends only that anchor, exact committed "
                        "ownership is idempotent, and every different or extra owner, "
                        "nonce, process identity, job, mutex, row, or anchor poisons"
                    ),
                },
                "quiescence_record": {
                    "fields": list(PUBLICATION_WORKER_QUIESCENCE_FIELDS),
                    "journal_record_table": "publication_worker_quiescences",
                    "record_identity_fields": [
                        "store_instance_id",
                        "attempt_id",
                        "worker_ownership_sha256",
                        "store_session_nonce_sha256",
                        "verification_mode",
                    ],
                    "prepare_anchor_event": (
                        "publication_worker_quiescence_prepare"
                    ),
                    "committed_anchor_event": (
                        "publication_worker_quiescence_committed"
                    ),
                    "fixed_values": {
                        "schema_version": (
                            "sec-gemma-online-risk-overlay-v2-2-"
                            "publication-worker-quiescence-v1"
                        ),
                        "quiescence_verifier_id": (
                            "sec-gemma-online-risk-overlay-v2-2-"
                            "publication-worker-quiescence-verifier-v1"
                        ),
                        "owner_mutex_unowned": True,
                        "job_object_active_process_count": 0,
                        "recorded_git_ssh_processes_alive_count": 0,
                        "quiescence_status": "verified_no_live_owner_or_worker",
                    },
                    "verification_modes": [
                        "same_session_clean_release",
                        "post_restart_prior_owner_dead",
                    ],
                    "post_restart_rule": (
                        "post-restart verification requires the exact prior owner PID plus "
                        "creation FILETIME identity to be dead, the exclusive owner mutex "
                        "to be unowned, the named job to be absent or report zero active "
                        "processes, and every recorded Git or SSH process identity to be "
                        "dead before this record can commit"
                    ),
                    "exact_reconciliation": (
                        "the prepare anchor stores the exact canonical quiescence row "
                        "bytes; partial exact states complete idempotently, while any live "
                        "owner, worker, mutex, job member, different observation, row, or "
                        "anchor fails closed and forbids capability issuance"
                    ),
                },
                "capability_gate": (
                    "no normal or recovery publication capability may be issued while an "
                    "ownership record lacks an exact quiescence successor; after restart, "
                    "a fresh capability additionally requires verification mode "
                    "post_restart_prior_owner_dead, so an old Git or SSH tree can never "
                    "overlap a new publication owner"
                ),
                "deadline_and_parent_exit_rule": (
                    "both normal and recovery supervisors use the same kill-on-close "
                    "containment; deadline handling calls TerminateJobObject, closes the "
                    "controlling handle, waits for zero active processes, and persists "
                    "quiescence, while unexpected parent exit closes the sole handle and "
                    "lets the operating system kill the complete tree"
                ),
            },
            "publication_transport_isolation": {
                "schema_version": (
                    "sec-gemma-online-risk-overlay-v2-2-"
                    "publication-transport-isolation-v1"
                ),
                "profile_id": (
                    "sec-gemma-online-risk-overlay-v2-2-"
                    "publication-transport-isolation-profile-v1"
                ),
                "isolated_git_directory_manifest_fields": [
                    "schema_version",
                    "profile_id",
                    "contract_version",
                    "contract_sha256",
                    "implementation_manifest_sha256",
                    "implementation_commit",
                    "store_instance_id",
                    "store_session_nonce_sha256",
                    "attempt_id",
                    "publication_intent_sha256",
                    "operation_kind",
                    "operation_sha256",
                    "isolated_git_directory_identity_sha256",
                    "local_config_entries_sha256",
                    "empty_hooks_directory_path",
                    "empty_hooks_directory_identity_sha256",
                    "empty_hooks_directory_listing_sha256",
                    "alternates_file_bytes_sha256",
                    "alternate_object_directory_identity_sha256",
                    "git_executable_path",
                    "git_executable_sha256",
                    "git_version_stdout_sha256",
                    "git_exec_path",
                    "git_exec_path_directory_manifest_sha256",
                    "git_remote_https_executable_path",
                    "git_remote_https_executable_sha256",
                    "credential_helper_config_value",
                    "credential_helper_executable_path",
                    "credential_helper_executable_sha256",
                    "credential_helper_version_stdout_sha256",
                    "command_interpreter_executable_path",
                    "command_interpreter_executable_sha256",
                    "transport_executable_closure_manifest_sha256",
                    "child_environment_policy_id",
                    "child_environment_sha256",
                    "path_lookup_forbidden",
                    "remote_url",
                    "remote_url_scheme",
                    "readback_command_profile_sha256",
                    "push_command_profile_sha256",
                    "pre_transport_store_journal_sequence",
                    "pre_transport_store_journal_tip_sha256",
                    "isolated_transport_git_directory_manifest_sha256",
                ],
                "journal_record_table": (
                    "publication_isolated_transport_git_directory_manifests"
                ),
                "prepare_anchor_event": "publication_transport_manifest_prepare",
                "committed_anchor_event": (
                    "publication_transport_manifest_committed"
                ),
                "local_config_exact_entries": [
                    ["core.repositoryformatversion", "0"],
                    ["core.bare", "true"],
                    [
                        "credential.helper",
                        (
                            "!\"C:/Program Files/Git/mingw64/bin/"
                            "git-credential-manager.exe\""
                        ),
                    ],
                    ["core.hooksPath", "{verified_empty_hooks_directory_path}"],
                    ["credential.useHttpPath", "true"],
                    ["http.followRedirects", "false"],
                    ["http.sslVerify", "true"],
                ],
                "local_config_allowlist_only": True,
                "forbidden_local_config_namespaces": [
                    "remote.*",
                    "url.*",
                    "include.*",
                    "includeIf.*",
                    "advice.*Hook*",
                    "http.proxy",
                    "http.noProxy",
                    "http.curloptResolve",
                    "http.*.extraheader",
                ],
                "child_environment_policy_id": (
                    "sec-gemma-online-risk-overlay-v2-2-"
                    "publication-exact-child-environment-v1"
                ),
                "child_environment_mode": "exact_allowlist_no_inheritance",
                "inherited_environment_forbidden": True,
                "child_environment_key_set": [
                    "SystemRoot",
                    "WINDIR",
                    "COMSPEC",
                    "PATH",
                    "PATHEXT",
                    "TEMP",
                    "TMP",
                    "USERPROFILE",
                    "LOCALAPPDATA",
                    "APPDATA",
                    "HOME",
                    "GIT_CONFIG_NOSYSTEM",
                    "GIT_CONFIG_SYSTEM",
                    "GIT_CONFIG_GLOBAL",
                    "GIT_CONFIG_COUNT",
                    "GIT_TERMINAL_PROMPT",
                    "GIT_ALLOW_PROTOCOL",
                    "GIT_PROTOCOL_FROM_USER",
                    "GIT_EXEC_PATH",
                    "GCM_INTERACTIVE",
                    "LANG",
                    "LC_ALL",
                ],
                "exact_child_environment": {
                    "SystemRoot": "{verified_SystemRoot}",
                    "WINDIR": "{verified_WINDIR}",
                    "COMSPEC": "{verified_COMSPEC}",
                    "PATH": "{verified_transport_PATH}",
                    "PATHEXT": "{verified_PATHEXT}",
                    "TEMP": "{verified_TEMP}",
                    "TMP": "{verified_TMP}",
                    "USERPROFILE": "{verified_USERPROFILE}",
                    "LOCALAPPDATA": "{verified_LOCALAPPDATA}",
                    "APPDATA": "{verified_APPDATA}",
                    "HOME": "{verified_HOME}",
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_SYSTEM": "NUL",
                    "GIT_CONFIG_GLOBAL": "NUL",
                    "GIT_CONFIG_COUNT": "0",
                    "GIT_TERMINAL_PROMPT": "0",
                    "GIT_ALLOW_PROTOCOL": "https",
                    "GIT_PROTOCOL_FROM_USER": "0",
                    "GIT_EXEC_PATH": "{verified_git_exec_path}",
                    "GCM_INTERACTIVE": "Never",
                    "LANG": "C",
                    "LC_ALL": "C",
                },
                "unlisted_environment_rule": (
                    "construct a new environment block from exactly the case-sensitive "
                    "child_environment_key_set and exact_child_environment mapping; never "
                    "copy or inherit os.environ, reject case-fold duplicates, and omit every "
                    "unlisted name, including GIT_CONFIG_PARAMETERS, GIT_DIR, "
                    "GIT_COMMON_DIR, GIT_WORK_TREE, GIT_OBJECT_DIRECTORY, "
                    "GIT_ALTERNATE_OBJECT_DIRECTORIES, GIT_INDEX_FILE, GIT_SSH, "
                    "GIT_SSH_COMMAND, GIT_PROXY_COMMAND, GIT_ASKPASS, SSH_ASKPASS, "
                    "GIT_SSL_NO_VERIFY, GIT_SSL_CAINFO, GIT_SSL_CAPATH, and every other "
                    "unlisted GIT_* or SSH_* variable; HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, "
                    "NO_PROXY, http_proxy, https_proxy, all_proxy, and no_proxy are also "
                    "unlisted and absent"
                ),
                "allowed_remote_url": (
                    "https://github.com/AntonioDomenech/"
                    "LLM-memory-trading-agent.git"
                ),
                "remote_url_scheme": "https",
                "path_lookup_forbidden": True,
                "remote_url_grammar": (
                    "the exact allowed HTTPS URL only: lowercase https scheme, github.com "
                    "host, no userinfo, explicit port, query, fragment, redirect, SCP-like "
                    "syntax, alternate protocol, or percent-encoded authority"
                ),
                "executable_identity_rule": (
                    "before each worker creation, re-hash and compare the verified absolute "
                    "Git executable, exact git-remote-https executable, Git exec-path "
                    "directory manifest, absolute credential helper, command interpreter "
                    "if used, and complete transport executable closure; path lookup is "
                    "forbidden for Git, the HTTPS helper, and credential helper, and any "
                    "changed, missing, additional, or unmanifested executable poisons "
                    "before process creation"
                ),
                "credential_helper_rule": (
                    "credential_helper_config_value is exactly the frozen shell snippet "
                    "!\"C:/Program Files/Git/mingw64/bin/"
                    "git-credential-manager.exe\"; the leading exclamation mark selects "
                    "Git's shell-snippet form, the whitespace-containing forward-slash "
                    "absolute path is double-quoted, no arguments or expansion syntax are "
                    "allowed, and the separately normalized executable plus pinned command "
                    "interpreter are hash-verified before process creation"
                ),
                "hook_disable_rule": (
                    "core.hooksPath is the manifest-bound absolute path of a dedicated "
                    "empty directory inside the verified isolated transport root; the "
                    "directory identity and canonical empty listing are hash-bound, it "
                    "contains no files, subdirectories, links, junctions, reparse points, "
                    "alternate data streams, or executable entries, and its exact empty "
                    "state plus parent chain is rechecked immediately before every Git "
                    "process; any change poisons before process creation, so pre-push and "
                    "all other repository hooks are impossible"
                ),
                "environment_identity_rule": (
                    "the exact child-environment keys and values are canonically hashed "
                    "into child_environment_sha256; a missing, additional, differently "
                    "cased, duplicate, or changed entry poisons before process creation"
                ),
                "object_access_rule": (
                    "the fresh isolated bare Git directory has no worktree or remotes and "
                    "one exact objects/info/alternates entry naming only the verified "
                    "source-bound implementation repository object directory; the exact "
                    "alternates bytes and both directory identities are manifest-bound, "
                    "and object access never imports source repository configuration"
                ),
                "literal_endpoint_rule": (
                    "every ls-remote and push argv substitutes the publication intent's "
                    "literal allowed HTTPS remote_url, never remote_name; protocol "
                    "allowlisting, disabled redirects, absent proxy variables, disabled "
                    "system and global config, and allowlist-only isolated local config "
                    "exclude remote, URL rewrite, include, pushurl, proxy, curl resolve, or "
                    "extra-header injection; any different effective endpoint evidence "
                    "poisons before observation or push authorization"
                ),
                "durability_rule": (
                    "the exact self-hashed isolated-directory manifest and its prepare and "
                    "committed anchors are fsynced and read back before any Git or SSH "
                    "worker is created; every observation evidence value and every push "
                    "command hash binds transport_isolation_profile_sha256 and "
                    "isolated_transport_git_directory_manifest_sha256"
                ),
                "exact_reconciliation": (
                    "the prepare anchor contains the exact canonical self-hashed manifest "
                    "row bytes; prepare without a row appends only that exact row and "
                    "committed anchor, exact row without committed anchor appends only "
                    "that anchor, exact committed manifest is idempotent, and every "
                    "different config entry, environment name or value, executable or "
                    "helper identity, hook path or empty-directory state, URL, object "
                    "alternate, profile hash, row, extra row, or anchor poisons before any "
                    "worker is created"
                ),
                "readback_command_profile_sha256": (
                    PUBLICATION_REMOTE_READBACK_COMMAND_PROFILE_SHA256
                ),
                "push_command_profile_sha256": (
                    PUBLICATION_PUSH_COMMAND_PROFILE_SHA256
                ),
            },
            "publication_intent": {
                "scope": (
                    "every acquisition-pass, scored-pass, and scored-failed-gate "
                    "ordinary terminal report before its first remote Git push"
                ),
                "fields": list(PUBLICATION_INTENT_FIELDS),
                "opaque_receipt_type": "VerifiedPublicationIntent",
                "fixed_intent_status": "publication_pending",
                "fixed_values": {
                    "schema_version": (
                        "sec-gemma-online-risk-overlay-v2-2-publication-intent-v1"
                    ),
                    "intent_verifier_id": (
                        "sec-gemma-online-risk-overlay-v2-2-publication-intent-verifier-v1"
                    ),
                    "intent_status": "publication_pending",
                    "research_effect_authority_invalidated": True,
                    "semantic_result_release_blocked": True,
                    "next_stage_authority_blocked": True,
                    "external_cost_usd": 0,
                },
                "canonical_tag_preparation_order": (
                    "after the terminal artifact and current store commitment are "
                    "sealed, construct and validate the complete canonical annotated-tag "
                    "message, exact local tag object SHA-1, peeled implementation commit, "
                    "one stable per-attempt ordinary terminal tag ref, frozen remote "
                    "name, and frozen remote URL before building the publication intent"
                ),
                "durability_before_remote_effect": (
                    "complete the exact publication_intent_prepare external anchor, then "
                    "append the exact self-hashed publication_intent journal row, then "
                    "complete the publication_intent_committed external anchor; fsync and "
                    "read back each database, anchor, and parent-directory boundary, issue "
                    "the opaque intent store receipt only from the committed state, and "
                    "only then permit the first remote push"
                ),
                "append_protocol": {
                    "prepare_anchor_event": "publication_intent_prepare",
                    "journal_record_table": "publication_intents",
                    "committed_anchor_event": "publication_intent_committed",
                    "prepare_anchor_frozen_payload": (
                        "before the database row, the fsynced prepare anchor stores the "
                        "complete exact canonical self-hashed intent row bytes, "
                        "publication_intent_sha256, and "
                        "normal_attempt_elapsed_at_intent_prepare_hex; recovery must use "
                        "those bytes and may not resample or reconstruct elapsed time"
                    ),
                    "order": [
                        "normal_elapsed_hex_frozen_in_exact_intent_bytes",
                        "prepare_anchor_fsynced",
                        "exact_journal_row_committed",
                        "committed_anchor_fsynced",
                        "opaque_store_receipt_issued",
                    ],
                    "exact_reconciliation": (
                        "prepare anchor without a row deterministically appends only its "
                        "exact intent row and committed anchor; exact row without the "
                        "committed anchor appends only that anchor; an exact fully "
                        "committed intent is idempotent; any missing prepare anchor, "
                        "different row, different hash, unexpected extra row, or different "
                        "anchor payload poisons and fails closed"
                    ),
                    "pending_elapsed_binding": (
                        "normal_attempt_elapsed_at_intent_prepare_hex is a lowercase exact "
                        "finite nonnegative IEEE-754 binary64 hexadecimal value, round-"
                        "trips without loss, and decodes strictly below 3600; it is inside "
                        "the self-hashed intent row and is frozen with that complete row in "
                        "publication_intent_prepare before the database append, then bound "
                        "unchanged by the row, committed anchor, and opaque intent store "
                        "receipt, so a crash after the row but before the committed anchor "
                        "never requires a monotonic-clock resample"
                    ),
                },
                "fixed_terminal_material": (
                    "the intent immutably fixes the attempt and plan, intended terminal "
                    "status, report kind, artifact hash and store receipt, record counts "
                    "and commitment, exact terminal reconstruction material hash and its "
                    "opaque append-only store receipt hash, store identity, exact pre-"
                    "intent committed journal sequence and tip, exact elapsed-at-prepare "
                    "hex, predecessor publication, canonical tag identity, expected "
                    "publication hash, remote identity, and zero cost"
                ),
                "noncircular_store_binding": (
                    "store_journal_sequence and store_journal_tip_sha256 identify the "
                    "exact committed database tip immediately before the intent append; "
                    "the intent's separate append-only store receipt is issued only after "
                    "that append and is therefore outside the self-hashed intent body, "
                    "publication-intent and publication-receipt governance records remain "
                    "journaled and externally anchored but are excluded from semantic "
                    "record_counts and record_commitment_sha256, "
                    "while both publication_intent_sha256 and "
                    "publication_intent_store_receipt_sha256 are required in the later "
                    "ordinary terminal evidence and terminal anchor"
                ),
                "live_monotonic_clock_or_deadline_field_forbidden": True,
                "authority_cutoff": (
                    "issuing the durable intent atomically destroys and invalidates the "
                    "attempt's capability and every SEC, Yahoo, Gemma, model, market-value, "
                    "feature, learner, replay, metric, gate, no-leverage, scoring, artifact-"
                    "mutation, and research effect authority; only the source-bound exact "
                    "publication-recovery capability remains"
                ),
                "pending_receipt_rule": (
                    "the verified intent is the durable no-semantic publication-pending "
                    "receipt; it exposes no extraction, prediction, action, metric, gate, "
                    "return, failure direction, or terminal artifact payload"
                ),
                "remote_push_without_intent_forbidden": True,
                "intent_mutation_or_replacement_forbidden": True,
            },
            "publication_pending_terminal_reconstruction": {
                "material_hash_field": "terminal_reconstruction_material_sha256",
                "material_store_receipt_hash_field": (
                    "terminal_reconstruction_material_store_receipt_sha256"
                ),
                "journal_record_table": "terminal_reconstruction_materials",
                "record_identity_fields": [
                    "store_instance_id",
                    "attempt_id",
                    "report_kind",
                ],
                "acquisition_schema_version": (
                    "sec-gemma-online-risk-overlay-v2-2-"
                    "acquisition-terminal-reconstruction-v1"
                ),
                "scored_schema_version": (
                    "sec-gemma-online-risk-overlay-v2-2-"
                    "scored-terminal-reconstruction-v1"
                ),
                "acquisition_fields": list(
                    ACQUISITION_TERMINAL_RECONSTRUCTION_FIELDS
                ),
                "scored_fields": list(SCORED_TERMINAL_RECONSTRUCTION_FIELDS),
                "opaque_material_type": (
                    "VerifiedTerminalReconstructionMaterial"
                ),
                "opaque_store_receipt_type": (
                    "VerifiedTerminalReconstructionMaterialReceipt"
                ),
                "append_protocol": {
                    "prepare_anchor_event": "terminal_reconstruction_prepare",
                    "committed_anchor_event": "terminal_reconstruction_committed",
                    "order": [
                        "terminal_artifact_and_source_evidence_sealed",
                        "exact_reconstruction_row_canonicalized_and_self_hashed",
                        "prepare_anchor_with_exact_row_bytes_fsynced",
                        "exact_journal_row_committed",
                        "committed_anchor_fsynced",
                        "opaque_store_receipt_issued",
                        "publication_intent_prepare_permitted",
                    ],
                    "exact_reconciliation": (
                        "the prepare anchor contains the exact canonical self-hashed row "
                        "bytes; prepare without a row appends only that row and its exact "
                        "committed anchor, an exact row without the committed anchor "
                        "appends only that anchor, a fully committed row is idempotent, "
                        "and every different identity, row, hash, receipt, extra row, or "
                        "anchor poisons and fails closed"
                    ),
                },
                "intent_binding": (
                    "publication intent is forbidden until the exact reconstruction row "
                    "and committed anchor are durable and its opaque append-only store "
                    "receipt exists; the intent binds both "
                    "terminal_reconstruction_material_sha256 and "
                    "terminal_reconstruction_material_store_receipt_sha256"
                ),
                "acquisition_rehydration": (
                    "validate the exact acquisition reconstruction row and receipt, then "
                    "read only its named projections of the intent-bound acquisition "
                    "terminal artifact and receipt, acquisition validation, bundle, "
                    "manifest, private index, check set, record commitment, sealed "
                    "acquisition phase evidence, and sealed vault commitments; issue a "
                    "fresh opaque VerifiedAcquisitionReport for terminalization without "
                    "any SEC or Yahoo request, raw-data mutation, new selection, or "
                    "exposure of quarantined values"
                ),
                "scored_reconstruction": (
                    "validate the exact scored reconstruction row and receipt, then read "
                    "only its named projections of the intent-bound sealed joint artifact "
                    "and receipt, stage input, deterministic evaluation, metric inputs and "
                    "outputs, gate report, no-leverage proofs, exact frozen gate checks, "
                    "failed-gate names, and record commitment; reissue terminal evidence "
                    "without chronological replay, metric calculation, gate evaluation, "
                    "ledger calculation, no-leverage recomputation, or scoring"
                ),
                "read_only_local_material_permitted": [
                    "exact reconstruction material row and store receipt",
                    "intent-bound terminal artifact payload",
                    "append-only artifact and governance receipts",
                    "sealed acquisition phase evidence",
                    "sealed vault commitments",
                    "frozen scored gate checks and failed-gate names",
                    "store and anchor commitments",
                ],
                "new_research_or_recomputation_forbidden": True,
            },
            "publication_remote_observation": {
                "fields": list(PUBLICATION_REMOTE_OBSERVATION_FIELDS),
                "opaque_observation_type": (
                    "VerifiedDurablePublicationRemoteObservation"
                ),
                "scope": [
                    "normal_publication",
                    "publication_recovery",
                ],
                "journal_record_table": "publication_remote_observations",
                "record_identity_fields": [
                    "store_instance_id",
                    "attempt_id",
                    "observation_operation_sha256",
                    "observation_ordinal",
                ],
                "prepare_anchor_event": "publication_remote_observation_prepare",
                "committed_anchor_event": (
                    "publication_remote_observation_committed"
                ),
                "fixed_values": {
                    "schema_version": (
                        "sec-gemma-online-risk-overlay-v2-2-"
                        "publication-remote-observation-v1"
                    ),
                    "observation_verifier_id": (
                        "sec-gemma-online-risk-overlay-v2-2-"
                        "publication-remote-observation-verifier-v1"
                    ),
                },
                "readback_evidence_fields": list(
                    PUBLICATION_REMOTE_READBACK_EVIDENCE_FIELDS
                ),
                "readback_evidence_fixed_values": {
                    "schema_version": (
                        "sec-gemma-online-risk-overlay-v2-2-"
                        "publication-remote-readback-evidence-v1"
                    ),
                    "evidence_verifier_id": (
                        "sec-gemma-online-risk-overlay-v2-2-"
                        "publication-remote-readback-evidence-verifier-v1"
                    ),
                    "command_profile_id": (
                        "sec-gemma-online-risk-overlay-v2-2-"
                        "publication-remote-readback-command-profile-v1"
                    ),
                },
                "readback_command_profile": {
                    "profile_template_sha256": (
                        PUBLICATION_REMOTE_READBACK_COMMAND_PROFILE_SHA256
                    ),
                    "executable": "{verified_git_executable_path}",
                    "working_directory": (
                        "verified_isolated_transport_git_directory"
                    ),
                    "argv_template": [
                        "ls-remote",
                        "--tags",
                        "{remote_url}",
                        "{tag_ref}",
                        "{tag_ref}^{}",
                    ],
                    "command_count": 1,
                    "stdin_bytes": 0,
                    "stdout_representation": "exact_unmodified_bytes",
                    "stderr_representation": "exact_unmodified_bytes",
                    "child_environment_policy_id": (
                        "sec-gemma-online-risk-overlay-v2-2-"
                        "publication-exact-child-environment-v1"
                    ),
                    "child_environment_mode": "exact_allowlist_no_inheritance",
                    "inherited_environment_forbidden": True,
                    "child_environment_key_set": [
                        "SystemRoot",
                        "WINDIR",
                        "COMSPEC",
                        "PATH",
                        "PATHEXT",
                        "TEMP",
                        "TMP",
                        "USERPROFILE",
                        "LOCALAPPDATA",
                        "APPDATA",
                        "HOME",
                        "GIT_CONFIG_NOSYSTEM",
                        "GIT_CONFIG_SYSTEM",
                        "GIT_CONFIG_GLOBAL",
                        "GIT_CONFIG_COUNT",
                        "GIT_TERMINAL_PROMPT",
                        "GIT_ALLOW_PROTOCOL",
                        "GIT_PROTOCOL_FROM_USER",
                        "GIT_EXEC_PATH",
                        "GCM_INTERACTIVE",
                        "LANG",
                        "LC_ALL",
                    ],
                    "exact_child_environment": {
                        "SystemRoot": "{verified_SystemRoot}",
                        "WINDIR": "{verified_WINDIR}",
                        "COMSPEC": "{verified_COMSPEC}",
                        "PATH": "{verified_transport_PATH}",
                        "PATHEXT": "{verified_PATHEXT}",
                        "TEMP": "{verified_TEMP}",
                        "TMP": "{verified_TMP}",
                        "USERPROFILE": "{verified_USERPROFILE}",
                        "LOCALAPPDATA": "{verified_LOCALAPPDATA}",
                        "APPDATA": "{verified_APPDATA}",
                        "HOME": "{verified_HOME}",
                        "GIT_CONFIG_NOSYSTEM": "1",
                        "GIT_CONFIG_SYSTEM": "NUL",
                        "GIT_CONFIG_GLOBAL": "NUL",
                        "GIT_CONFIG_COUNT": "0",
                        "GIT_TERMINAL_PROMPT": "0",
                        "GIT_ALLOW_PROTOCOL": "https",
                        "GIT_PROTOCOL_FROM_USER": "0",
                        "GIT_EXEC_PATH": "{verified_git_exec_path}",
                        "GCM_INTERACTIVE": "Never",
                        "LANG": "C",
                        "LC_ALL": "C",
                    },
                    "unlisted_environment_rule": (
                        "construct a new exact environment mapping and omit every unlisted "
                        "variable without copying the parent environment"
                    ),
                    "sequence_hash_rule": (
                        "command_sequence_sha256 is the canonical SHA-256 of this exact "
                        "profile after substituting the manifest-bound absolute Git path, "
                        "Git exec path, already frozen literal remote URL, and tag ref; no "
                        "alternate option order, executable, environment, command, text "
                        "decoding, or additional readback command is valid; the top-level "
                        "Git subprocess is created directly with shell disabled, and the "
                        "only permitted internal shell use is Git's exact manifest-bound "
                        "credential-helper snippet under the pinned interpreter"
                    ),
                },
                "readback_evidence_cross_field_matrix": {
                    "completed_absent": {
                        "process_exit_status": "exited",
                        "process_exit_code": 0,
                        "transport_status": "completed",
                        "ref_lookup_status": "absent",
                        "stdout_byte_count": 0,
                        "stdout_sha256": EMPTY_BYTES_SHA256,
                        "stderr_byte_count": 0,
                        "stderr_sha256": EMPTY_BYTES_SHA256,
                        "raw_ref_object_value": (
                            PUBLICATION_REMOTE_REF_ABSENT_SENTINEL
                        ),
                        "raw_peeled_value": (
                            PUBLICATION_REMOTE_REF_ABSENT_SENTINEL
                        ),
                        "raw_tag_message_sha256": (
                            PUBLICATION_REMOTE_REF_ABSENT_SENTINEL
                        ),
                        "observation_permitted": True,
                        "projected_observed_ref_state": "absent",
                    },
                    "completed_present_well_formed": {
                        "process_exit_status": "exited",
                        "process_exit_code": 0,
                        "transport_status": "completed",
                        "ref_lookup_status": "present",
                        "stdout_grammar": (
                            "one or two LF-terminated ASCII rows, each exactly lowercase "
                            "40-hex OID, TAB, and one requested ref; no duplicate or "
                            "unrequested ref"
                        ),
                        "stderr_byte_count": 0,
                        "stderr_sha256": EMPTY_BYTES_SHA256,
                        "raw_ref_object_value": "exact_lowercase_40_hex_oid",
                        "raw_peeled_value": (
                            "exact_lowercase_40_hex_oid_or_VALUE_MISSING"
                        ),
                        "raw_tag_message_sha256": (
                            PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
                        ),
                        "observation_permitted": True,
                        "projected_observed_ref_state": (
                            "exact_expected_if_object_and_peeled_match_else_conflicting"
                        ),
                    },
                    "completed_malformed_output": {
                        "process_exit_status": "exited",
                        "process_exit_code": 0,
                        "transport_status": "protocol_error",
                        "ref_lookup_status": "unknown",
                        "raw_ref_object_value": (
                            PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL
                        ),
                        "raw_peeled_value": (
                            PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL
                        ),
                        "raw_tag_message_sha256": (
                            PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL
                        ),
                        "observation_permitted": False,
                    },
                    "exited_nonzero": {
                        "process_exit_status": "exited",
                        "process_exit_code": "any_integer_except_zero",
                        "transport_status": "unavailable",
                        "ref_lookup_status": "unknown",
                        "raw_ref_object_value": (
                            PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
                        ),
                        "raw_peeled_value": (
                            PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
                        ),
                        "raw_tag_message_sha256": (
                            PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
                        ),
                        "observation_permitted": False,
                    },
                    "deadline": {
                        "process_exit_status": "deadline",
                        "process_exit_code": None,
                        "transport_status": "unavailable",
                        "ref_lookup_status": "unknown",
                        "raw_ref_object_value": (
                            PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
                        ),
                        "raw_peeled_value": (
                            PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
                        ),
                        "raw_tag_message_sha256": (
                            PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
                        ),
                        "observation_permitted": False,
                    },
                    "parent_interrupted": {
                        "process_exit_status": "parent_interrupted",
                        "process_exit_code": None,
                        "transport_status": "unavailable",
                        "ref_lookup_status": "unknown",
                        "raw_ref_object_value": (
                            PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
                        ),
                        "raw_peeled_value": (
                            PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
                        ),
                        "raw_tag_message_sha256": (
                            PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
                        ),
                        "observation_permitted": False,
                    },
                    "spawn_failed": {
                        "process_exit_status": "spawn_failed",
                        "process_exit_code": None,
                        "transport_status": "unavailable",
                        "ref_lookup_status": "unknown",
                        "raw_ref_object_value": (
                            PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
                        ),
                        "raw_peeled_value": (
                            PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
                        ),
                        "raw_tag_message_sha256": (
                            PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
                        ),
                        "observation_permitted": False,
                    },
                },
                "evidence_matrix_exhaustiveness_rule": (
                    "every readback attempt matches exactly one matrix row; no other "
                    "exit, transport, lookup, stdout, stderr, or raw-value combination "
                    "is valid, every nonzero or missing exit code maps to lookup unknown "
                    "and forbids an observation, and only completed_absent may project "
                    "REF_ABSENT and authorize an absent observation"
                ),
                "present_observation_projection_rule": (
                    "for completed_present_well_formed, observed object and peeled values "
                    "are the exact parsed lowercase OIDs or literal VALUE_MISSING for a "
                    "missing peeled row; exact_expected requires both expected OIDs and "
                    "projects the already frozen expected tag-message SHA-256 because the "
                    "annotated-tag object OID content-addresses those bytes, while every "
                    "other well-formed present result is conflicting and uses "
                    "VALUE_MISSING for an unobserved different tag message"
                ),
                "readback_process_exit_statuses": [
                    "exited",
                    "deadline",
                    "parent_interrupted",
                    "spawn_failed",
                ],
                "readback_transport_statuses": [
                    "completed",
                    "unavailable",
                    "protocol_error",
                ],
                "readback_ref_lookup_statuses": [
                    "absent",
                    "present",
                    "unknown",
                ],
                "literal_observed_value_sentinels": {
                    "ref_absent": PUBLICATION_REMOTE_REF_ABSENT_SENTINEL,
                    "value_missing": PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL,
                    "value_malformed": PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
                    "no_prior_push_command_sha256": (
                        PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256
                    ),
                },
                "observation_phases": [
                    "pre_push",
                    "post_push",
                ],
                "observed_ref_states": [
                    "absent",
                    "exact_expected",
                    "conflicting",
                ],
                "accepted_observation_operations": {
                    "normal_publication": {
                        "observation_operation_sha256": (
                            PUBLICATION_NORMAL_OPERATION_SHA256
                        ),
                        "required_recovery_start": False,
                    },
                    "publication_recovery": {
                        "observation_operation_sha256": (
                            "the exact committed recovery_invocation_start_sha256"
                        ),
                        "required_recovery_start": True,
                    },
                },
                "operation_binding_rule": (
                    "accept exactly normal_publication with the frozen normal-operation "
                    "sentinel or publication_recovery with the exact committed recovery "
                    "start hash; worker_ownership_sha256 must identify the exact committed "
                    "owner whose operation_sha256 matches, and recovery additionally "
                    "requires the exact committed start row and anchors"
                ),
                "allowed_observation_sequences": [
                    [],
                    [
                        {
                            "observation_ordinal": 1,
                            "observation_phase": "pre_push",
                            "observed_ref_state": "absent",
                        }
                    ],
                    [
                        {
                            "observation_ordinal": 1,
                            "observation_phase": "pre_push",
                            "observed_ref_state": "exact_expected",
                        }
                    ],
                    [
                        {
                            "observation_ordinal": 1,
                            "observation_phase": "pre_push",
                            "observed_ref_state": "conflicting",
                        }
                    ],
                    [
                        {
                            "observation_ordinal": 1,
                            "observation_phase": "pre_push",
                            "observed_ref_state": "absent",
                        },
                        {
                            "observation_ordinal": 2,
                            "observation_phase": "post_push",
                            "observed_ref_state": "absent",
                        },
                    ],
                    [
                        {
                            "observation_ordinal": 1,
                            "observation_phase": "pre_push",
                            "observed_ref_state": "absent",
                        },
                        {
                            "observation_ordinal": 2,
                            "observation_phase": "post_push",
                            "observed_ref_state": "exact_expected",
                        },
                    ],
                    [
                        {
                            "observation_ordinal": 1,
                            "observation_phase": "pre_push",
                            "observed_ref_state": "absent",
                        },
                        {
                            "observation_ordinal": 2,
                            "observation_phase": "post_push",
                            "observed_ref_state": "conflicting",
                        },
                    ],
                ],
                "observation_sequence_rule": (
                    "the empty sequence is valid only when no completed readback permits "
                    "an observation; otherwise the first and only pre_push observation "
                    "has ordinal one, exact or conflicting pre_push state terminates the "
                    "operation, and absent may be followed by at most one post_push "
                    "observation at ordinal two; no gap, duplicate, third observation, "
                    "second pre_push, first post_push, or other transition is valid"
                ),
                "push_phase_binding_rule": (
                    "every pre_push observation stores the frozen no-prior-push-command "
                    "sentinel; a post_push observation is valid only after the exact "
                    "committed one-push authorization for the same operation and one "
                    "non-force push command, stores the exact canonical push-command "
                    "SHA-256 in prior_push_command_sha256, and binds that command in the "
                    "exact readback evidence bytes"
                ),
                "classification_rule": (
                    "an observation row is permitted only when evidence says process "
                    "exit status exited with integer exit code zero, transport status "
                    "completed, and ref lookup status absent or present; absent requires "
                    "ref lookup absent and the "
                    "literal REF_ABSENT sentinel for all observed object fields, while "
                    "transport unavailable, protocol error, unknown lookup, deadline, "
                    "interruption, or spawn failure can never be encoded as absence or "
                    "produce an observation row; exact_expected requires "
                    "the expected annotated-tag object, peeled commit, and tag-message "
                    "hash; every existing malformed, missing, or different binding is "
                    "conflicting and records its exact normalized observed values or the "
                    "literal VALUE_MISSING or VALUE_MALFORMED sentinel"
                ),
                "evidence_binding_rule": (
                    "remote_readback_evidence_sha256 is the hash of exactly the listed "
                    "evidence fields excluding its self-hash; command sequence, exit "
                    "status and code, exact stdout and stderr byte counts and SHA-256 "
                    "hashes, transport and lookup statuses, raw normalized values, "
                    "operation identity, phase, ordinal, and prior push command are all "
                    "mandatory, and the observation is a deterministic projection of "
                    "that one evidence value"
                ),
                "transport_isolation_binding_rule": (
                    "transport_isolation_profile_sha256 is the canonical hash of the "
                    "frozen publication_transport_isolation block, and "
                    "isolated_transport_git_directory_manifest_sha256 identifies the "
                    "exact committed same-operation isolated directory; evidence is "
                    "invalid unless argv contains the intent's literal remote_url and "
                    "the effective endpoint, config isolation, object alternate, and "
                    "both command-profile hashes match that manifest"
                ),
                "append_protocol": {
                    "order": [
                        "exact_remote_readback_evidence_canonicalized",
                        "exact_observation_row_canonicalized_and_self_hashed",
                        "prepare_anchor_with_exact_evidence_and_row_bytes_fsynced",
                        "exact_journal_row_committed",
                        "committed_anchor_fsynced",
                        "opaque_observation_issued",
                    ],
                    "exact_reconciliation": (
                        "the prepare anchor contains the exact canonical remote-readback "
                        "evidence bytes and exact canonical self-hashed observation row "
                        "bytes; prepare without a row appends only that row and committed "
                        "anchor, an exact row without the committed anchor appends only "
                        "that anchor, a fully committed observation is idempotent, and "
                        "every different evidence payload, identity, ordinal, phase, "
                        "state, row, hash, extra row, or anchor poisons"
                    ),
                },
                "durability_gate": (
                    "no remote readback may authorize a push, publication receipt, "
                    "completion field, or conflict poison until its exact observation "
                    "row and prepare and committed anchors are fsynced and read back"
                ),
                "conflict_proof_boundary": (
                    "a conflict becomes proven only when the exact conflicting "
                    "observation is durably committed; if the parent dies before the "
                    "observation prepare anchor there is no durable proof to preserve, "
                    "while a committed conflicting observation deterministically "
                    "requires poison reconciliation before any new remote action even "
                    "if the foreign ref later disappears"
                ),
            },
            "publication_receipt": {
                "fields": list(PUBLICATION_RECEIPT_FIELDS),
                "opaque_receipt_type": "VerifiedDurablePublicationReceipt",
                "fixed_values": {
                    "schema_version": (
                        "sec-gemma-online-risk-overlay-v2-2-publication-receipt-v1"
                    ),
                    "receipt_verifier_id": (
                        "sec-gemma-online-risk-overlay-v2-2-publication-receipt-verifier-v1"
                    ),
                    "receipt_status": "publication_verified",
                    "publication_capability_invalidated": True,
                    "terminalization_capability_required": True,
                },
                "append_protocol": {
                    "prepare_anchor_event": "publication_receipt_prepare",
                    "journal_record_table": "publication_receipts",
                    "committed_anchor_event": "publication_receipt_committed",
                    "order": [
                        "verified_external_publication_observation_committed",
                        "matching_receipt_eligible_recovery_completion_committed_if_recovery",
                        "prepare_anchor_with_exact_row_bytes_fsynced",
                        "exact_journal_row_committed",
                        "committed_anchor_fsynced",
                        "opaque_store_receipt_issued",
                    ],
                    "exact_reconciliation": (
                        "the prepare anchor contains the exact canonical self-hashed "
                        "receipt row bytes; prepare without a row deterministically "
                        "appends only that exact row and committed anchor, exact row "
                        "without the committed anchor appends only that anchor, an exact "
                        "fully committed receipt is idempotent, and any different "
                        "observation, publication, row, hash, unexpected extra row, or "
                        "anchor payload poisons and fails closed"
                    ),
                },
                "observation_binding_rule": (
                    "publication_remote_observation_sha256 must identify the exact "
                    "durably committed exact_expected observation for the same intent, "
                    "stable ref, expected tag object, peeled commit, operation, and worker "
                    "owner; an in-memory or unanchored readback cannot create a receipt"
                ),
                "completion_and_authorization_binding_rule": (
                    "for normal_publication, recovery_invocation_completion_sha256 is the "
                    "frozen normal-no-recovery-completion sentinel; for "
                    "publication_recovery it is the exact committed completion for the "
                    "same operation and observation whose matrix row is receipt-eligible, "
                    "and that completion must exist before receipt prepare; "
                    "pre_push_authorization_sha256 is the no-authorization sentinel for "
                    "a pre_push exact_expected observation and the exact same-operation "
                    "authorization for a post_push exact_expected observation"
                ),
                "normal_no_recovery_completion_sha256": (
                    PUBLICATION_NORMAL_NO_RECOVERY_COMPLETION_SHA256
                ),
                "terminal_evidence_precondition": (
                    "ordinary acquisition or scored terminal evidence may be issued only "
                    "after the exact publication_receipt_committed anchor and opaque "
                    "publication receipt store receipt exist; terminal evidence and its "
                    "terminal anchor bind publication_receipt_sha256 and "
                    "publication_receipt_store_receipt_sha256"
                ),
                "receipt_committed_restart_rule": (
                    "if the durable publication receipt exists but terminal evidence or "
                    "terminal transition does not, restart skips all remote publication "
                    "and issues only a fresh terminalization capability"
                ),
            },
            "publication_recovery": {
                "eligibility": (
                    "only the exact durable publication intent, its exact append-only "
                    "store receipt, and a chain containing only whitelisted descendant "
                    "governance records may authorize recovery; eligibility is rebuilt "
                    "from durable bytes rather than any pre-restart opaque object"
                ),
                "whitelisted_descendant_governance_records": [
                    "publication_intent_prepare",
                    "publication_intent_committed",
                    "publication_transport_manifest_prepare",
                    "publication_transport_manifest_committed",
                    "publication_recovery_start_prepare",
                    "publication_recovery_start_committed",
                    "publication_pre_push_authorization_prepare",
                    "publication_pre_push_authorization_committed",
                    "publication_recovery_completion_prepare",
                    "publication_recovery_completion_committed",
                    "publication_worker_owner_prepare",
                    "publication_worker_owner_committed",
                    "publication_worker_quiescence_prepare",
                    "publication_worker_quiescence_committed",
                    "publication_remote_observation_prepare",
                    "publication_remote_observation_committed",
                    "publication_receipt_prepare",
                    "publication_receipt_committed",
                    "publication_conflict_prepare",
                    "publication_conflict_committed",
                    "terminalization_claim_prepare",
                    "terminalization_claim_committed",
                    "terminal_intent",
                    "terminal_committed",
                ],
                "unknown_descendant_record_poisons": True,
                "restart_boot_and_downtime_rule": (
                    "the same exact publish-or-recover operation may continue after "
                    "process restart, operating-system restart, machine boot, or arbitrary "
                    "downtime; elapsed downtime never reopens the effectful attempt and "
                    "never authorizes any research computation"
                ),
                "exact_publish_or_recover": (
                    "reconstruct the canonical tag object and expected publication from "
                    "the intent; each recovery invocation first performs and durably "
                    "commits an exact current remote observation and may issue at most "
                    "one push command, only when that observation proves the frozen ref "
                    "absent, and only for the same precomputed object without force; a "
                    "later invocation may attempt that same single push only after its "
                    "own new exact durable observation again proves the ref absent; "
                    "accept only the exact expected annotated-tag "
                    "object SHA-1, peeled implementation commit, message, ref, remote, "
                    "predecessor, artifact, terminal status, report kind, contract, "
                    "implementation, expected publication hash, and zero cost"
                ),
                "push_commands_per_invocation_at_most": 1,
                "conflicting_tag_rule": (
                    "any existing ref with a missing, malformed, or different object, "
                    "peeled target, message, or binding poisons the publication state and "
                    "fails closed; it is never deleted, force-updated, replaced, reused as "
                    "evidence, or converted to another terminal result"
                ),
                "poison_is_irreversible": True,
                "transient_failure_rule": (
                    "network, process, timeout, restart, boot, or unavailable-remote "
                    "failure without a proven conflicting tag preserves the exact durable "
                    "publication-pending intent for a later bounded recovery invocation"
                ),
                "pending_rule": (
                    "until exact remote readback and the normal opaque publication receipt "
                    "exist, the attempt remains publication-pending, releases no semantic "
                    "result, and provides no predecessor or next-stage authority"
                ),
                "store_reopen_precedence": (
                    "on every reopen, reconcile an exact durable publication intent and "
                    "then every partial or committed transport-isolation manifest before "
                    "the complete observation, authorization, poison, completion, and "
                    "receipt sweep, the generic consumed-attempt recovery rule, or any "
                    "capability issuance; an exact intent preserves publication-pending "
                    "across restart and boot, while a consumed attempt with no exact "
                    "durable intent follows the existing terminal-indeterminate recovery"
                ),
                "observation_reopen_precedence": {
                    "ordered_sweep": [
                        "reconcile_every_partial_transport_manifest_prepare_row_and_commit",
                        "validate_exact_transport_manifest_and_reject_extras",
                        "reconcile_every_partial_observation_prepare_row_and_commit",
                        "validate_exact_operation_sequence_and_reject_extras",
                        "reconcile_matching_authorization_and_worker_quiescence",
                        "materialize_irreversible_conflict_poison_when_required",
                        "materialize_terminal_observation_completion_when_required",
                        "materialize_exact_publication_receipt_when_eligible",
                        "only_then_consider_fresh_capability",
                    ],
                    "terminal_observation_definition": (
                        "pre_push exact_expected or conflicting, or ordinal-two post_push "
                        "absent, exact_expected, or conflicting; pre_push absent alone is "
                        "nonterminal"
                    ),
                    "first_terminal_observation_rule": (
                        "the first valid terminal observation in the frozen operation "
                        "sequence is authoritative; the sequence rules permit at most one, "
                        "and any second terminal observation, conflicting duplicate, "
                        "different row, or extra ordinal poisons rather than replacing it"
                    ),
                    "recovery_completion_reconstruction": (
                        "for a recovery terminal observation with no completion prepare "
                        "anchor, deterministically prepare and commit the unique matching "
                        "matrix completion before any receipt or fresh capability: "
                        "pre_push exact_expected maps to remote_exact_without_push, "
                        "pre_push conflicting maps to "
                        "remote_conflict_poisoned_without_push, post_push absent maps to "
                        "post_push_ref_absent, post_push exact_expected maps to "
                        "published_exact_after_push, and post_push conflicting maps to "
                        "post_push_conflict_poisoned; use the exact observation and "
                        "authorization hashes, push-command-count upper bound fixed by "
                        "phase, elapsed seconds "
                        "exactly 0x1.2c00000000000p+8, and cumulative seconds equal to "
                        "the correctly rounded binary64 sum of prior cumulative plus that "
                        "exact value; a conflict mapping waits for its exact matching "
                        "irreversible poison to commit before completion"
                    ),
                    "normal_terminal_observation_reconciliation": (
                        "normal exact_expected observation creates or reconciles the exact "
                        "receipt directly with the normal-no-recovery-completion sentinel; "
                        "normal conflicting observation creates or reconciles poison; "
                        "normal post_push absent remains pending; no case performs another "
                        "remote read before this local reconciliation"
                    ),
                    "pre_push_absent_crash_rule": (
                        "a lone pre_push absent observation is not durable authority for "
                        "a later operation; recovery interruption reconciliation closes "
                        "its old start conservatively, and any fresh operation must obtain "
                        "its own new durable pre_push observation"
                    ),
                    "capability_block": (
                        "receipt, poison, or fresh publication capability issuance is "
                        "forbidden until every transport manifest plus this full observation "
                        "sweep is complete; a committed terminal observation can never be "
                        "discarded, superseded by a new readback, or rewritten because the "
                        "remote ref later changes or disappears"
                    ),
                },
                "capability_state_machine": {
                    "effect_capability": (
                        "valid only before publication_intent_prepare and permanently "
                        "invalidated by publication_intent_committed"
                    ),
                    "publication_capability": (
                        "fresh opaque capability issued only from an exact committed "
                        "intent with no committed receipt or poison and no unresolved "
                        "publication worker ownership, partial or mismatched transport "
                        "manifest, or unreconciled observation, authorization, completion, "
                        "poison, or receipt state; permits only one bounded exact "
                        "publish-or-recover invocation and its governance records"
                    ),
                    "terminalization_capability": (
                        "fresh opaque capability issued only from an exact committed "
                        "publication receipt with no poison and no existing terminalization "
                        "claim; it is single-use, must be atomically claimed and invalidated "
                        "before the first terminal_intent anchor, and permits only read-only "
                        "terminal reconstruction, ordinary terminal evidence issuance, "
                        "the exact terminal transition, anchors, and sealed-result return"
                    ),
                    "restart_reissue": (
                        "every reopen validates the durable intent, intent store receipt, "
                        "whitelisted descendant chain, every partial or committed transport "
                        "manifest, and any publication receipt, then first proves every prior "
                        "publication worker owner quiescent and completes the ordered local "
                        "reconciliation sweep; it issues the appropriate fresh capability "
                        "bound to the exact store instance and a new store-session nonce only "
                        "when no terminalization claim exists; an existing claim enters exact "
                        "terminal recovery or idempotent sealed-result rehydration without a "
                        "new capability, and "
                        "every capability from an earlier process, store session, or nonce "
                        "is stale"
                    ),
                },
                "recovery_invocation_start_record": {
                    "fields": list(
                        PUBLICATION_RECOVERY_INVOCATION_START_FIELDS
                    ),
                    "journal_record_table": "publication_recovery_invocation_starts",
                    "record_identity_fields": [
                        "store_instance_id",
                        "attempt_id",
                        "invocation_ordinal",
                    ],
                    "prepare_anchor_event": "publication_recovery_start_prepare",
                    "committed_anchor_event": (
                        "publication_recovery_start_committed"
                    ),
                    "fixed_values": {
                        "schema_version": (
                            "sec-gemma-online-risk-overlay-v2-2-"
                            "publication-recovery-start-v1"
                        ),
                        "start_verifier_id": (
                            "sec-gemma-online-risk-overlay-v2-2-"
                            "publication-recovery-start-verifier-v1"
                        ),
                        "start_status": "started",
                        "invocation_seconds_cap": (
                            MAX_PUBLICATION_RECOVERY_SECONDS
                        ),
                        "durable_pre_push_authorization_required": True,
                    },
                    "first_prior_completion_sha256": (
                        PUBLICATION_RECOVERY_COMPLETION_GENESIS_SHA256
                    ),
                    "first_prior_cumulative_recovery_seconds": "0x0.0p+0",
                    "time_encoding_rule": (
                        "prior_cumulative_recovery_seconds is a lowercase exact finite "
                        "nonnegative IEEE-754 binary64 hexadecimal value that round-trips "
                        "without loss; invocation ordinal one uses exactly 0x0.0p+0 and "
                        "every later start copies the prior committed completion's exact "
                        "cumulative value byte for byte"
                    ),
                    "before_remote_action_rule": (
                        "the exact self-hashed start row and its prepare and committed "
                        "anchors must be fsynced and read back before any remote readback, "
                        "push authorization, or push; every field needed to reconstruct "
                        "the start is fixed before that first remote action"
                    ),
                    "exact_reconciliation": (
                        "the prepare anchor contains the exact canonical start row bytes; "
                        "prepare without a row appends only that row and committed anchor, "
                        "an exact row without the committed anchor appends only that "
                        "anchor, a fully committed start is idempotent, and every "
                        "different or extra row, hash, identity, nonce, or anchor poisons"
                    ),
                    "self_hashed_and_externally_anchored": True,
                },
                "pre_push_authorization_record": {
                    "fields": list(PUBLICATION_PRE_PUSH_AUTHORIZATION_FIELDS),
                    "journal_record_table": "publication_pre_push_authorizations",
                    "record_identity_fields": [
                        "store_instance_id",
                        "attempt_id",
                        "pre_push_authorization_marker_key",
                    ],
                    "prepare_anchor_event": (
                        "publication_pre_push_authorization_prepare"
                    ),
                    "committed_anchor_event": (
                        "publication_pre_push_authorization_committed"
                    ),
                    "fixed_values": {
                        "schema_version": (
                            "sec-gemma-online-risk-overlay-v2-2-"
                            "publication-pre-push-authorization-v1"
                        ),
                        "authorization_verifier_id": (
                            "sec-gemma-online-risk-overlay-v2-2-"
                            "publication-pre-push-authorization-verifier-v1"
                        ),
                        "authorization_status": "push_authorized_once",
                        "push_command_limit": 1,
                    },
                    "accepted_authorization_operations": {
                        "normal_publication": {
                            "authorization_operation_sha256": (
                                PUBLICATION_NORMAL_OPERATION_SHA256
                            ),
                            "authorization_operation_ordinal": 0,
                            "required_recovery_start": False,
                        },
                        "publication_recovery": {
                            "authorization_operation_sha256": (
                                "the exact committed recovery_invocation_start_sha256"
                            ),
                            "authorization_operation_ordinal": (
                                "the exact recovery invocation ordinal"
                            ),
                            "required_recovery_start": True,
                        },
                    },
                    "operation_binding_rule": (
                        "normal_publication binds the frozen normal-operation sentinel, "
                        "ordinal zero, and exact committed normal worker ownership; "
                        "publication_recovery binds its exact committed recovery start "
                        "hash and ordinal plus matching exact committed worker ownership; "
                        "worker_ownership_sha256 must name the owner whose operation hash "
                        "equals authorization_operation_sha256"
                    ),
                    "marker_key_derivation_rule": (
                        "only after the exact operation foundation and worker ownership "
                        "exist, derive pre_push_authorization_marker_key as the canonical "
                        "SHA-256 identity of schema version, store instance, store-session "
                        "nonce, attempt, publication intent, operation kind, operation "
                        "hash, operation ordinal, and worker ownership; for recovery the "
                        "operation foundation is the committed start, so the marker key "
                        "is never a recovery-start field and creates no hash cycle"
                    ),
                    "issuance_rule": (
                        "only a fresh exact durably committed pre_push remote observation "
                        "for the same normal or recovery operation proving the stable ref "
                        "absent may create this marker; remote_observation_sha256 binds "
                        "that observation, and the marker's exact self-hashed row and both "
                        "anchors must be fsynced and read back before the sole push command "
                        "may be issued; no normal or recovery push is legal without this "
                        "marker, and none is created for an exact existing ref, conflict, "
                        "unavailable remote, or unanchored observation"
                    ),
                    "push_command_profile": {
                        "profile_template_sha256": (
                            PUBLICATION_PUSH_COMMAND_PROFILE_SHA256
                        ),
                        "executable": "{verified_git_executable_path}",
                        "working_directory": (
                            "verified_isolated_transport_git_directory"
                        ),
                        "argv_template": [
                            "push",
                            "--porcelain",
                            "--no-verify",
                            "{remote_url}",
                            "{expected_tag_object_sha1}:{tag_ref}",
                        ],
                        "command_count": 1,
                        "force_forbidden": True,
                        "stdin_bytes": 0,
                        "stdout_representation": "exact_unmodified_bytes",
                        "stderr_representation": "exact_unmodified_bytes",
                        "child_environment_policy_id": (
                            "sec-gemma-online-risk-overlay-v2-2-"
                            "publication-exact-child-environment-v1"
                        ),
                        "child_environment_mode": "exact_allowlist_no_inheritance",
                        "inherited_environment_forbidden": True,
                        "child_environment_key_set": [
                            "SystemRoot",
                            "WINDIR",
                            "COMSPEC",
                            "PATH",
                            "PATHEXT",
                            "TEMP",
                            "TMP",
                            "USERPROFILE",
                            "LOCALAPPDATA",
                            "APPDATA",
                            "HOME",
                            "GIT_CONFIG_NOSYSTEM",
                            "GIT_CONFIG_SYSTEM",
                            "GIT_CONFIG_GLOBAL",
                            "GIT_CONFIG_COUNT",
                            "GIT_TERMINAL_PROMPT",
                            "GIT_ALLOW_PROTOCOL",
                            "GIT_PROTOCOL_FROM_USER",
                            "GIT_EXEC_PATH",
                            "GCM_INTERACTIVE",
                            "LANG",
                            "LC_ALL",
                        ],
                        "exact_child_environment": {
                            "SystemRoot": "{verified_SystemRoot}",
                            "WINDIR": "{verified_WINDIR}",
                            "COMSPEC": "{verified_COMSPEC}",
                            "PATH": "{verified_transport_PATH}",
                            "PATHEXT": "{verified_PATHEXT}",
                            "TEMP": "{verified_TEMP}",
                            "TMP": "{verified_TMP}",
                            "USERPROFILE": "{verified_USERPROFILE}",
                            "LOCALAPPDATA": "{verified_LOCALAPPDATA}",
                            "APPDATA": "{verified_APPDATA}",
                            "HOME": "{verified_HOME}",
                            "GIT_CONFIG_NOSYSTEM": "1",
                            "GIT_CONFIG_SYSTEM": "NUL",
                            "GIT_CONFIG_GLOBAL": "NUL",
                            "GIT_CONFIG_COUNT": "0",
                            "GIT_TERMINAL_PROMPT": "0",
                            "GIT_ALLOW_PROTOCOL": "https",
                            "GIT_PROTOCOL_FROM_USER": "0",
                            "GIT_EXEC_PATH": "{verified_git_exec_path}",
                            "GCM_INTERACTIVE": "Never",
                            "LANG": "C",
                            "LC_ALL": "C",
                        },
                        "unlisted_environment_rule": (
                            "construct a new exact environment mapping and omit every "
                            "unlisted variable without copying the parent environment"
                        ),
                    },
                    "push_command_evidence_rule": (
                        "the command runs only in the exact isolated directory bound by "
                        "the authorization's durable absent observation, substitutes the "
                        "intent's literal allowed HTTPS remote_url and expected tag object, "
                        "uses the manifest-bound absolute Git executable and child "
                        "environment, and uses no remote name, force, rewrite, redirect, "
                        "proxy, pushurl, repository hook, or alternate argv; --no-verify "
                        "forbids pre-push execution independently of the manifest-bound "
                        "empty hooks directory; the top-level Git subprocess is created "
                        "directly with shell disabled, and the only internal shell use "
                        "allowed is the exact manifest-bound credential-helper snippet "
                        "under the pinned interpreter; "
                        "prior_push_command_sha256 in any post_push observation is the "
                        "canonical hash of this resolved profile plus exact process exit, "
                        "stdout, and stderr evidence"
                    ),
                    "exact_reconciliation": (
                        "the prepare anchor contains the exact canonical authorization row "
                        "bytes; prepare without a row appends only that row and committed "
                        "anchor, an exact row without the committed anchor appends only "
                        "that anchor, a fully committed marker is idempotent and consumes "
                        "the operation's one-push authority, and every different or extra "
                        "row, observation, marker key, hash, or anchor poisons"
                    ),
                    "self_hashed_and_externally_anchored": True,
                },
                "recovery_invocation_completion_record": {
                    "fields": list(
                        PUBLICATION_RECOVERY_INVOCATION_COMPLETION_FIELDS
                    ),
                    "journal_record_table": (
                        "publication_recovery_invocation_completions"
                    ),
                    "record_identity_fields": [
                        "store_instance_id",
                        "attempt_id",
                        "invocation_ordinal",
                    ],
                    "prepare_anchor_event": (
                        "publication_recovery_completion_prepare"
                    ),
                    "committed_anchor_event": (
                        "publication_recovery_completion_committed"
                    ),
                    "fixed_values": {
                        "schema_version": (
                            "sec-gemma-online-risk-overlay-v2-2-"
                            "publication-recovery-completion-v1"
                        ),
                        "completion_verifier_id": (
                            "sec-gemma-online-risk-overlay-v2-2-"
                            "publication-recovery-completion-verifier-v1"
                        ),
                        "completion_status": "completed",
                    },
                    "outcome_cross_field_matrix": {
                        "remote_unavailable_before_observation": {
                            "outcome": "remote_unavailable_before_observation",
                            "pre_push_authorization_sha256": (
                                PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
                            ),
                            "remote_observation_sha256": (
                                PUBLICATION_RECOVERY_NO_REMOTE_OBSERVATION_SHA256
                            ),
                            "push_command_count_upper_bound": 0,
                            "conflict_poison_requirement": "forbidden",
                            "publication_receipt_eligible": False,
                        },
                        "remote_exact_without_push": {
                            "outcome": "remote_exact_without_push",
                            "pre_push_authorization_sha256": (
                                PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
                            ),
                            "remote_observation_sha256": (
                                "exact_expected_remote_observation_sha256"
                            ),
                            "push_command_count_upper_bound": 0,
                            "conflict_poison_requirement": "forbidden",
                            "publication_receipt_eligible": True,
                        },
                        "remote_conflict_poisoned_without_push": {
                            "outcome": (
                                "remote_conflict_poisoned_without_push"
                            ),
                            "pre_push_authorization_sha256": (
                                PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
                            ),
                            "remote_observation_sha256": (
                                "exact_conflicting_remote_observation_sha256"
                            ),
                            "push_command_count_upper_bound": 0,
                            "conflict_poison_requirement": (
                                "exact_committed_matching_conflict_poison"
                            ),
                            "publication_receipt_eligible": False,
                        },
                        "remote_absent_authorization_not_committed": {
                            "outcome": (
                                "remote_absent_authorization_not_committed"
                            ),
                            "pre_push_authorization_sha256": (
                                PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
                            ),
                            "remote_observation_sha256": (
                                "exact_absent_remote_observation_sha256"
                            ),
                            "push_command_count_upper_bound": 0,
                            "conflict_poison_requirement": "forbidden",
                            "publication_receipt_eligible": False,
                        },
                        "authorized_push_not_issued": {
                            "outcome": "authorized_push_not_issued",
                            "pre_push_authorization_sha256": (
                                "exact_committed_pre_push_authorization_sha256"
                            ),
                            "remote_observation_sha256": (
                                "exact_absent_remote_observation_sha256"
                            ),
                            "push_command_count_upper_bound": 0,
                            "conflict_poison_requirement": "forbidden",
                            "publication_receipt_eligible": False,
                        },
                        "push_issued_unconfirmed": {
                            "outcome": "push_issued_unconfirmed",
                            "pre_push_authorization_sha256": (
                                "exact_committed_pre_push_authorization_sha256"
                            ),
                            "remote_observation_sha256": (
                                "exact_absent_remote_observation_sha256"
                            ),
                            "push_command_count_upper_bound": 1,
                            "conflict_poison_requirement": "forbidden",
                            "publication_receipt_eligible": False,
                        },
                        "post_push_ref_absent": {
                            "outcome": "post_push_ref_absent",
                            "pre_push_authorization_sha256": (
                                "exact_committed_pre_push_authorization_sha256"
                            ),
                            "remote_observation_sha256": (
                                "exact_absent_remote_observation_sha256"
                            ),
                            "push_command_count_upper_bound": 1,
                            "conflict_poison_requirement": "forbidden",
                            "publication_receipt_eligible": False,
                        },
                        "published_exact_after_push": {
                            "outcome": "published_exact_after_push",
                            "pre_push_authorization_sha256": (
                                "exact_committed_pre_push_authorization_sha256"
                            ),
                            "remote_observation_sha256": (
                                "exact_expected_remote_observation_sha256"
                            ),
                            "push_command_count_upper_bound": 1,
                            "conflict_poison_requirement": "forbidden",
                            "publication_receipt_eligible": True,
                        },
                        "post_push_conflict_poisoned": {
                            "outcome": "post_push_conflict_poisoned",
                            "pre_push_authorization_sha256": (
                                "exact_committed_pre_push_authorization_sha256"
                            ),
                            "remote_observation_sha256": (
                                "exact_conflicting_remote_observation_sha256"
                            ),
                            "push_command_count_upper_bound": 1,
                            "conflict_poison_requirement": (
                                "exact_committed_matching_conflict_poison"
                            ),
                            "publication_receipt_eligible": False,
                        },
                        "interrupted_without_authorization": {
                            "outcome": "interrupted_before_completion",
                            "pre_push_authorization_sha256": (
                                PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
                            ),
                            "remote_observation_sha256": (
                                PUBLICATION_RECOVERY_NO_REMOTE_OBSERVATION_SHA256
                            ),
                            "push_command_count_upper_bound": 0,
                            "conflict_poison_requirement": (
                                "not_inferred_from_completion"
                            ),
                            "publication_receipt_eligible": False,
                        },
                        "interrupted_with_authorization": {
                            "outcome": "interrupted_before_completion",
                            "pre_push_authorization_sha256": (
                                "exact_committed_pre_push_authorization_sha256"
                            ),
                            "remote_observation_sha256": (
                                PUBLICATION_RECOVERY_NO_REMOTE_OBSERVATION_SHA256
                            ),
                            "push_command_count_upper_bound": 1,
                            "conflict_poison_requirement": (
                                "not_inferred_from_completion"
                            ),
                            "publication_receipt_eligible": False,
                        },
                    },
                    "outcome_observation_requirements": {
                        "remote_unavailable_before_observation": {
                            "observation_phase": None,
                            "observation_ordinal": None,
                            "observed_ref_state": None,
                            "required_sequence": [],
                        },
                        "remote_exact_without_push": {
                            "observation_phase": "pre_push",
                            "observation_ordinal": 1,
                            "observed_ref_state": "exact_expected",
                            "required_sequence": ["pre_push:exact_expected"],
                        },
                        "remote_conflict_poisoned_without_push": {
                            "observation_phase": "pre_push",
                            "observation_ordinal": 1,
                            "observed_ref_state": "conflicting",
                            "required_sequence": ["pre_push:conflicting"],
                        },
                        "remote_absent_authorization_not_committed": {
                            "observation_phase": "pre_push",
                            "observation_ordinal": 1,
                            "observed_ref_state": "absent",
                            "required_sequence": ["pre_push:absent"],
                        },
                        "authorized_push_not_issued": {
                            "observation_phase": "pre_push",
                            "observation_ordinal": 1,
                            "observed_ref_state": "absent",
                            "required_sequence": ["pre_push:absent"],
                        },
                        "push_issued_unconfirmed": {
                            "observation_phase": "pre_push",
                            "observation_ordinal": 1,
                            "observed_ref_state": "absent",
                            "required_sequence": ["pre_push:absent"],
                            "forbidden_committed_post_push_observation": True,
                        },
                        "post_push_ref_absent": {
                            "observation_phase": "post_push",
                            "observation_ordinal": 2,
                            "observed_ref_state": "absent",
                            "required_sequence": [
                                "pre_push:absent",
                                "post_push:absent",
                            ],
                        },
                        "published_exact_after_push": {
                            "observation_phase": "post_push",
                            "observation_ordinal": 2,
                            "observed_ref_state": "exact_expected",
                            "required_sequence": [
                                "pre_push:absent",
                                "post_push:exact_expected",
                            ],
                        },
                        "post_push_conflict_poisoned": {
                            "observation_phase": "post_push",
                            "observation_ordinal": 2,
                            "observed_ref_state": "conflicting",
                            "required_sequence": [
                                "pre_push:absent",
                                "post_push:conflicting",
                            ],
                        },
                        "interrupted_without_authorization": {
                            "observation_phase": None,
                            "observation_ordinal": None,
                            "observed_ref_state": None,
                            "required_sequence": (
                                "ignored_for_conservative_interruption_row"
                            ),
                        },
                        "interrupted_with_authorization": {
                            "observation_phase": None,
                            "observation_ordinal": None,
                            "observed_ref_state": None,
                            "required_sequence": (
                                "ignored_for_conservative_interruption_row"
                            ),
                        },
                    },
                    "outcome_observation_binding_rule": (
                        "each clean completion references the exact durable observation "
                        "at the phase, ordinal, state, and complete operation sequence "
                        "listed for its outcome; published_exact_after_push and "
                        "post_push_conflict_poisoned can never reference a pre_push "
                        "observation, push_issued_unconfirmed requires no committed "
                        "post_push observation, and no unlisted or extra observation is "
                        "compatible with completion"
                    ),
                    "outcome_matrix_rule": (
                        "every completion matches exactly one matrix row and no other "
                        "outcome spelling or cross-field combination is valid; exact "
                        "dynamic hashes must resolve to the invocation's committed "
                        "authorization, last exact remote observation, and matching "
                        "conflict poison where required"
                    ),
                    "clean_completion_observation_rule": (
                        "for a non-interrupted row, remote_observation_sha256 is the last "
                        "exact durably committed observation in that invocation, or the "
                        "frozen no-observation sentinel only when no exact observation "
                        "committed; an exact absent observation is retained when a later "
                        "post-push readback is unavailable or was never durably committed"
                    ),
                    "authorization_and_push_rule": (
                        "a clean completion with no committed authorization uses the "
                        "frozen no-authorization sentinel and push-command-count upper "
                        "bound zero; a clean "
                        "completion with a committed authorization uses its exact hash "
                        "and has upper bound zero only for authorized_push_not_issued, "
                        "otherwise one; for clean completion the upper bound equals the "
                        "known actual count, while interrupted completion may conservatively "
                        "overstate command issuance; values greater than one are invalid"
                    ),
                    "conflict_completion_rule": (
                        "each clean conflict outcome requires the already committed "
                        "irreversible conflict poison for the same operation and exact "
                        "remote observation before completion; every non-conflict clean "
                        "outcome forbids such poison"
                    ),
                    "after_action_rule": (
                        "only after the invocation's remote work ends, or deterministic "
                        "interruption reconciliation runs, may the exact outcome, final "
                        "observation, push-command-count upper bound, elapsed seconds, and "
                        "cumulative seconds be canonicalized into this separate self-hashed "
                        "completion row"
                    ),
                    "exact_reconciliation": (
                        "the prepare anchor contains the exact canonical completion row "
                        "bytes; prepare without a row appends only that exact row and "
                        "committed anchor, an exact row without the committed anchor "
                        "appends only that anchor, a fully committed completion is "
                        "idempotent, and every different or extra row, hash, count, time, "
                        "outcome, observation, or anchor poisons"
                    ),
                    "cumulative_time_rule": (
                        "the start ordinal and prior completion hash plus the completion "
                        "elapsed and cumulative seconds form one append-only chain that "
                        "survives restart and boot; cumulative time equals prior cumulative "
                        "time plus this elapsed time and never resets"
                    ),
                    "time_encoding_rule": (
                        "elapsed_seconds and cumulative_recovery_seconds are lowercase "
                        "exact finite nonnegative IEEE-754 binary64 hexadecimal values "
                        "that round-trip without loss; elapsed decodes at most 300, and "
                        "cumulative is the correctly rounded round-to-nearest-ties-to-even "
                        "binary64 sum of the start's prior cumulative value and elapsed"
                    ),
                    "self_hashed_and_externally_anchored": True,
                },
                "interrupted_invocation_reconciliation": {
                    "no_observation_sentinel_sha256": (
                        PUBLICATION_RECOVERY_NO_REMOTE_OBSERVATION_SHA256
                    ),
                    "no_pre_push_authorization_sentinel_sha256": (
                        PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
                    ),
                    "rule": (
                        "after reconciling every partial start, observation, pre-push "
                        "authorization, completion, poison, and receipt append, and only "
                        "when no terminal observation requires exact matrix completion "
                        "reconstruction, an exact committed start from an earlier stale "
                        "store session with no completion prepare anchor is closed by one "
                        "canonical completion: outcome interrupted_before_completion, "
                        "remote_observation_sha256 equal to the no-observation sentinel, "
                        "elapsed_seconds exactly 0x1.2c00000000000p+8, which decodes to "
                        "300, cumulative seconds equal to the correctly rounded binary64 "
                        "sum of prior cumulative plus that exact value, and "
                        "push_command_count_upper_bound equal to one if and "
                        "only if the exact durable committed pre-push authorization marker "
                        "for that recovery start exists, otherwise zero with the "
                        "no-authorization sentinel; this is explicitly a conservative "
                        "upper bound because a crash may follow marker commit but precede "
                        "command issuance, and it is never reported as a known actual "
                        "command count; a "
                        "completion prepare anchor always reconciles its already fixed "
                        "exact row and is never replaced by these interruption sentinels"
                    ),
                },
                "conflict_poison_record": {
                    "fields": list(PUBLICATION_CONFLICT_FIELDS),
                    "prepare_anchor_event": "publication_conflict_prepare",
                    "journal_record_table": "publication_conflicts",
                    "committed_anchor_event": "publication_conflict_committed",
                    "self_hashed_and_externally_anchored": True,
                    "accepted_observation_operations": {
                        "normal_publication": {
                            "observation_operation_sha256": (
                                PUBLICATION_NORMAL_OPERATION_SHA256
                            ),
                            "required_recovery_start": False,
                        },
                        "publication_recovery": {
                            "observation_operation_sha256": (
                                "the exact committed recovery_invocation_start_sha256"
                            ),
                            "required_recovery_start": True,
                        },
                    },
                    "operation_binding_rule": (
                        "accept exactly one operation pair: normal_publication with the "
                        "frozen normal-operation sentinel, or publication_recovery with "
                        "the exact committed recovery start hash; in both cases "
                        "prior_governance_record_sha256 is the exact committed worker "
                        "ownership record whose operation_sha256 equals "
                        "observation_operation_sha256, and recovery additionally requires "
                        "that exact committed start row and anchors"
                    ),
                    "proof_timing_rule": (
                        "the same operation that durably commits the exact conflicting "
                        "remote observation must durably prepare and commit this poison "
                        "before returning pending or releasing ownership; normal "
                        "publication never waits for a later recovery invocation to "
                        "preserve a proven conflict"
                    ),
                    "observation_precondition": (
                        "remote_observation_sha256 must identify an exact committed "
                        "conflicting publication-remote-observation row for the same "
                        "intent, operation, worker ownership, and ref; a remote readback "
                        "is not proven conflict evidence before that observation's "
                        "prepare and committed anchors exist"
                    ),
                    "exact_reconciliation": (
                        "the prepare anchor contains the exact canonical self-hashed "
                        "conflict row bytes; prepare without a row appends only that exact "
                        "row and committed anchor, exact row without committed anchor "
                        "appends only that anchor, exact committed poison is idempotent, "
                        "and every different observation, row, hash, extra row, or anchor "
                        "remains poison"
                    ),
                    "irreversibility_rule": (
                        "once committed, the poison remains authoritative even if the "
                        "foreign or conflicting remote tag is later deleted or changed; "
                        "no later record, store copy, or remote observation may remove it"
                    ),
                },
                "completion_rule": (
                    "after exact remote readback, issue the ordinary "
                    "VerifiedExternalPublication, durably commit the exact publication "
                    "receipt, invalidate publication authority, issue a fresh "
                    "single-use terminalization capability, reconstruct ordinary "
                    "acquisition or scored terminal evidence from sealed material, "
                    "atomically commit the exact terminalization claim and invalidate "
                    "that capability, then use the existing terminal_intent and "
                    "terminal_committed store sequence"
                ),
                "effect_scope": (
                    "recovery may inspect the sealed intent, exact intent-bound terminal "
                    "artifact payload, append-only receipts, sealed acquisition evidence "
                    "and vault commitments, store and anchor commitments, clean "
                    "implementation identity, local canonical tag object, frozen gate "
                    "material, and frozen Git remote only; new SEC, Yahoo, Ollama, Gemma, "
                    "model, market acquisition, feature, learner, replay, metric, gate, "
                    "ledger, no-leverage, and scoring work is forbidden"
                ),
                "each_invocation_seconds_at_most": (
                    MAX_PUBLICATION_RECOVERY_SECONDS
                ),
                "killable_supervised_process_required": True,
                "process_tree_termination_on_deadline_required": True,
                "zero_model_market_or_data_effects": True,
                "time_accounting": (
                    "report each recovery invocation and cumulative recovery elapsed time "
                    "separately; recovery time is not part of, and may never be described "
                    "as part of, the sub-hour effectful attempt"
                ),
            },
            "terminalization_claim": {
                "fields": list(TERMINALIZATION_CLAIM_FIELDS),
                "opaque_claim_type": "VerifiedTerminalizationClaim",
                "journal_record_table": "terminalization_claims",
                "record_identity_fields": [
                    "store_instance_id",
                    "attempt_id",
                ],
                "prepare_anchor_event": "terminalization_claim_prepare",
                "committed_anchor_event": "terminalization_claim_committed",
                "fixed_values": {
                    "schema_version": (
                        "sec-gemma-online-risk-overlay-v2-2-"
                        "terminalization-claim-v1"
                    ),
                    "claim_verifier_id": (
                        "sec-gemma-online-risk-overlay-v2-2-"
                        "terminalization-claim-verifier-v1"
                    ),
                    "claim_status": "claimed_single_use",
                },
                "atomic_single_use_rule": (
                    "under the store's exclusive transaction and unique attempt identity, "
                    "the first valid TerminalizationCapability freezes the exact terminal "
                    "evidence and status, appends this self-hashed claim through prepare, "
                    "row, and committed anchors, and is atomically invalidated by that "
                    "commit before terminal_intent may be appended"
                ),
                "exact_reconciliation": (
                    "the prepare anchor stores the exact canonical claim row bytes; "
                    "prepare without a row appends only that row and committed anchor, an "
                    "exact row without committed anchor appends only that anchor, exact "
                    "committed claim is idempotent, and every different or extra claim, "
                    "capability nonce, evidence, status, row, or anchor poisons"
                ),
                "concurrent_or_repeated_entry_rule": (
                    "a concurrent or repeated caller that observes any exact claim cannot "
                    "claim a capability or append a second terminal_intent; it must enter "
                    "the exact terminal recovery state machine, which appends the first "
                    "frozen terminal_intent exactly once when the committed claim exists "
                    "without one, then appends each later missing predeclared transition "
                    "or anchor at most once or performs read-only sealed-result "
                    "rehydration after full commit"
                ),
                "terminal_intent_precondition": (
                    "the exact committed terminalization claim and its frozen evidence "
                    "hash are mandatory inputs to the one terminal_intent anchor"
                ),
            },
            "terminal_completion_recovery": {
                "publication_intent_precludes_indeterminate": (
                    "after publication_intent_committed, no crash, timeout, restart, boot, "
                    "partial governance write, publication receipt, or terminalization "
                    "window may convert the attempt to terminal-indeterminate"
                ),
                "receipt_without_terminal_intent": (
                    "an exact committed publication receipt with no terminal_intent "
                    "rehydrates the sealed terminal material and issues fresh ordinary "
                    "terminal evidence; if no terminalization claim exists it issues and "
                    "atomically claims one fresh single-use terminalization capability "
                    "before beginning the exact terminal sequence, while an existing "
                    "claim enters the claim_without_terminal_intent window without "
                    "another capability or claim"
                ),
                "claim_without_terminal_intent": (
                    "if the exact terminalization_claim_committed anchor exists but no "
                    "terminal_intent exists, validate the claim, frozen terminal evidence "
                    "and status, publication intent and receipt, and reconstruction "
                    "bindings, then append the one exact predeclared terminal_intent "
                    "anchor exactly once without issuing another capability or claim; "
                    "after that append, continue through the ordinary "
                    "terminal_intent_without_database_transition window"
                ),
                "terminal_intent_without_database_transition": (
                    "if the exact terminal_intent anchor exists but its terminal database "
                    "transition is absent, validate the evidence, intent and publication "
                    "receipt bindings, append exactly that predeclared transition once, "
                    "then append the exact terminal_committed anchor"
                ),
                "database_transition_without_terminal_committed_anchor": (
                    "if the exact terminal database transition exists after the matching "
                    "terminal_intent but terminal_committed is absent, validate the row "
                    "and all bindings and append only the exact terminal_committed anchor"
                ),
                "fully_committed_idempotency": (
                    "an exact terminal_committed state performs no publication, capability "
                    "claim, terminal_intent, database transition, or new anchor; it may "
                    "return the identical sealed result only by read-only validation and "
                    "rehydration from the exact terminal anchor and database row, bound "
                    "terminal artifact and store receipt, reconstruction-material row and "
                    "receipt, terminal evidence, publication intent, and publication "
                    "receipt, with no new research computation"
                ),
                "mismatch_rule": (
                    "any different terminal evidence, transition, status, row, anchor, "
                    "intent, receipt, terminalization claim, capability nonce, or "
                    "unexpected descendant governance record poisons and fails closed "
                    "without becoming indeterminate"
                ),
            },
            "final_registry_authorization": {
                "successor_fields": list(FINAL_REGISTRY_SUCCESSOR_FIELDS),
                "authorization_fields": list(
                    FINAL_REGISTRY_AUTHORIZATION_FIELDS
                ),
                "successor_rule": (
                    "validate the frozen predecessor registry and tip, require successor "
                    "ordinal = predecessor reveal count + 1, append exactly this contract, "
                    "branch, clean implementation commit, final attempt ID, and status "
                    "registered_unrun, hash the successor entry and whole registry, then "
                    "externally publish that registry hash under the separate stable "
                    "final-registry ref before final registration"
                ),
                "tag_ref_template": FINAL_REGISTRY_TAG_REF_TEMPLATE,
                "attempt_plan_input": (
                    "only opaque VerifiedFinalRegistryAuthorization; an arbitrary "
                    "hexadecimal string, mapping, or publication receipt alone is never "
                    "authorization"
                ),
            },
            "check_name_rule": (
                "dedicated validators accept the exact frozen names even when longer "
                "than 64 characters; generic source-role syntax is not reused for gate "
                "or acquisition check names"
            ),
            "attempt_deadline_scope": (
                "the parent monotonic deadline starts before prerequisite final-registry "
                "authorization or any attempt registration, whichever is earlier, and "
                "ends only after source reverification, registration, consumption, all "
                "network/model/deterministic work, replay, metrics, gates, no-leverage "
                "proofs, and every research store write, then remains active until either "
                "the exact remote publication, terminal anchor, and sealed result complete "
                "and are returned, or the fsynced opaque no-semantic publication-pending "
                "receipt is explicitly returned; a live normal parent never stops its "
                "clock merely because the intent was committed, and a whole-host crash "
                "after intent preparation fixes the durable pending boundary at the exact "
                "elapsed hex frozen in the prepare anchor and intent row so that the first "
                "post-boot operation is reconciliation followed by a later recovery "
                "invocation; the 239-second normal-governance partition is fixed "
                "at 89 "
                "seconds for local canonical-tag and intent preparation, 90 seconds for "
                "supervised publication, and 60 seconds for finalization or pending "
                "handoff, and the complete normal interval must remain strictly below "
                "3600 seconds"
            ),
        },
        "runtime": {
            "scope": (
                "each effectful acquisition attempt and each scored stage attempt is "
                "independently strictly below 3600 seconds; no individual approach "
                "test may combine multiple attempts to evade this ceiling, and normal "
                "effectful timing ends only when the normal parent returns a complete "
                "sealed terminal result or explicitly returns the exact durable "
                "no-semantic publication-pending receipt"
            ),
            "cumulative_lifecycle_rule": (
                "report the sum of all acquisition and stage attempts separately; do "
                "not describe the multi-stage research lifecycle or any separately "
                "reported publication recovery time as a sub-hour run"
            ),
            "acquisition_seconds_at_most": MAX_SEC_SECONDS,
            "sec_seconds_at_most": MAX_SEC_SECONDS,
            "sec_requests_at_most": MAX_SEC_REQUESTS,
            "sec_transport_bytes_at_most": MAX_SEC_BYTES,
            "sec_requests_per_second_at_most": MAX_SEC_REQUESTS_PER_SECOND,
            "market_seconds_at_most_within_acquisition": MAX_MARKET_SECONDS,
            "market_requests_each_stage": MAX_MARKET_REQUESTS_PER_STAGE,
            "sec_plus_market_combined_seconds_at_most": MAX_SEC_SECONDS,
            "gemma_seconds_at_most": MAX_MODEL_SECONDS,
            "deterministic_seconds_at_most": MAX_DETERMINISTIC_SECONDS,
            "total_seconds_strictly_below": MAX_TOTAL_RUNTIME_SECONDS,
            "contingency_seconds": GOVERNANCE_CONTINGENCY_SECONDS,
            "maximum_phase_caps_plus_contingency_seconds": 3599,
            "normal_governance_partition": {
                "local_tag_and_intent_preparation_seconds": (
                    PUBLICATION_INTENT_PREPARATION_SECONDS
                ),
                "supervised_publication_seconds": (
                    SUPERVISED_PUBLICATION_SECONDS
                ),
                "terminal_finalization_or_pending_seconds": (
                    PUBLICATION_FINALIZATION_PENDING_SECONDS
                ),
                "partition_sum_seconds": GOVERNANCE_CONTINGENCY_SECONDS,
                "borrowing_between_partitions_forbidden": True,
                "remote_push_before_durable_intent_forbidden": True,
            },
            "publication_recovery": {
                "each_non_effectful_invocation_seconds_at_most": (
                    MAX_PUBLICATION_RECOVERY_SECONDS
                ),
                "separately_reported_from_effectful_attempt": True,
                "arbitrary_downtime_between_invocations_permitted": True,
                "research_effects_permitted": False,
                "model_calls_permitted": 0,
                "market_or_sec_requests_permitted": 0,
            },
            "normal_to_recovery_transition": (
                "the 89-second intent-preparation, 90-second supervised-publication, and "
                "60-second terminal-finalization-or-pending partitions all belong to the "
                "same original sub-hour normal invocation; only after that invocation "
                "returns the explicit pending receipt, or after a whole-host crash leaves "
                "the already committed intent as its durable pending boundary, may a new "
                "separately timed recovery invocation begin"
            ),
            "per_attempt_phase_budgets": {
                "development_acquisition": {
                    "acquisition_seconds_at_most": MAX_SEC_SECONDS,
                    "gemma_seconds_at_most": 0,
                    "deterministic_seconds_at_most": MAX_DETERMINISTIC_SECONDS,
                    "total_seconds_strictly_below": MAX_TOTAL_RUNTIME_SECONDS,
                },
                "development_scored": {
                    "acquisition_seconds_at_most": 0,
                    "gemma_seconds_at_most": MAX_MODEL_SECONDS,
                    "deterministic_seconds_at_most": MAX_DETERMINISTIC_SECONDS,
                    "total_seconds_strictly_below": MAX_TOTAL_RUNTIME_SECONDS,
                },
                "confirmation_scored": {
                    "acquisition_seconds_at_most": MAX_SEC_SECONDS,
                    "gemma_seconds_at_most": MAX_MODEL_SECONDS,
                    "deterministic_seconds_at_most": MAX_DETERMINISTIC_SECONDS,
                    "total_seconds_strictly_below": MAX_TOTAL_RUNTIME_SECONDS,
                },
                "final_scored": {
                    "acquisition_seconds_at_most": MAX_SEC_SECONDS,
                    "gemma_seconds_at_most": MAX_MODEL_SECONDS,
                    "deterministic_seconds_at_most": MAX_DETERMINISTIC_SECONDS,
                    "total_seconds_strictly_below": MAX_TOTAL_RUNTIME_SECONDS,
                },
            },
            "model_call_caps": {
                "development_2000_2018": 80,
                "confirmation_2019_2023": 20,
                "final_2024_2026_ytd": 12,
            },
            "latency_preflight": {
                "selection": (
                    "after deterministic preprocessing, select the five development "
                    "events with greatest canonical UTF-8 model-request byte length; "
                    "break ties by accession ascending"
                ),
                "execution_order": (
                    "those five execute first in descending request-byte length then "
                    "accession ascending; remaining events execute by availability "
                    "session then acceptance timestamp then accession"
                ),
                "calls_are_sealed_batch_calls_not_duplicates": True,
                "projection": (
                    "sum(first_five_elapsed_seconds) + remaining_call_count * "
                    "max(first_five_elapsed_seconds)"
                ),
                "projection_must_not_exceed_gemma_seconds": MAX_MODEL_SECONDS,
                "minimum_preflight_calls": 5,
            },
            "clock_and_process": (
                "an external parent process samples monotonic elapsed time before and "
                "after each phase and the complete current attempt, enforces shrinking "
                "subprocess deadlines, and terminates a worker on budget overrun"
            ),
            "failure_on_budget_overrun": True,
        },
        "stage_access": {
            "preregistration_binding": (
                "the first implementation commit must record the exact commit that "
                "contains this literal contract hash; every effectful attempt binds "
                "that commit, the implementation commit, clean-tree identity, and all "
                "new-v2.2 source hashes before access"
            ),
            "development_acquisition": {
                "attempt_id": DEVELOPMENT_ACQUISITION_ID,
                "one_shot": True,
                "consume_before_first_official_sec_or_market_network_request": True,
                "no_model_or_canonical_market_value_access": True,
                "private_quarantine_outputs_only": (
                    "exact raw bytes, authentication receipts, hashes, counts, and "
                    "deterministic blinded model requests; no semantic output, price "
                    "value, return, label, action, or score"
                ),
                "raw_quarantine_is_opaque": True,
                "detached_catalog_and_stage_content_replay_required": True,
                "exact_full_market_coverage_required": True,
                "verified_acquisition_report_and_external_pin_required_for_pass": True,
                "failed_or_pre_intent_indeterminate_acquisition_is_terminal": True,
                "post_intent_failure_remains_publication_pending": True,
            },
            "development": {
                "attempt_id": DEVELOPMENT_ATTEMPT_ID,
                "one_shot": True,
                "acquisition_pass_required": True,
                "consume_before_first_real_gemma_call_or_first_canonical_market_value_read": True,
                "latency_preflight_calls_occur_after_consumption": True,
                "source_bound_production_executor_required_before_consumption": True,
                "exact_scored_stage_verification_required_for_pass": True,
                "failed_gate_joint_report_must_be_pinned_and_released": True,
                "pre_intent_indeterminate_execution_is_terminal": True,
                "post_intent_failure_remains_publication_pending": True,
            },
            "confirmation": {
                "attempt_id": CONFIRMATION_ATTEMPT_ID,
                "one_shot": True,
                "consume_before_first_stage_network_request": True,
                "consume_before_first_2019_feature_or_outcome_read": True,
                "source_bound_production_executor_required_before_consumption": True,
                "exact_scored_stage_verification_required_for_pass": True,
                "failed_gate_joint_report_must_be_pinned_and_released": True,
                "pre_intent_indeterminate_execution_is_terminal": True,
                "post_intent_failure_remains_publication_pending": True,
                "development_pass_required": True,
            },
            "final": {
                "attempt_id": FINAL_ATTEMPT_ID,
                "one_shot": True,
                "consume_before_first_stage_network_request": True,
                "consume_before_first_2024_feature_or_outcome_read": True,
                "source_bound_production_executor_required_before_consumption": True,
                "exact_scored_stage_verification_required_for_pass": True,
                "failed_gate_joint_report_must_be_pinned_and_released": True,
                "pre_intent_indeterminate_execution_is_terminal": True,
                "post_intent_failure_remains_publication_pending": True,
                "confirmation_pass_required": True,
                "predecessor_registry_pin_file": (
                    "docs/protocol_evidence/sec_gemma_reveal_registry_initial_pin.json"
                ),
                "predecessor_registry_pin_file_sha256": (
                    "85fe468ddad10a7fa1d226ea769a65cf91a359047075b9567f967fe4f776fdde"
                ),
                "predecessor_registry_sha256": (
                    "5853fcfc8f9ddb651981cb4b1c9f9426e57b214898ab6f080c0485eb1b40fe61"
                ),
                "predecessor_registry_tip_sha256": (
                    "10b0ecce18437a9c0e7f03f27882a3d7ba8836c1bcc48a75c3a64d0623de879a"
                ),
                "historical_final_reveal_count_lower_bound": 10,
                "opaque_verified_successor_authorization_required": True,
                "register_and_externally_pin_before_consumption": True,
            },
            "result_release": (
                "a stage emits no semantic extraction, prediction, partial metric, "
                "action count, return, gate, or direction before its sealed joint "
                "report; failure never authorizes another attempt or a sibling chosen "
                "from the revealed result, and publication-pending emits no semantic "
                "result and blocks every later stage until exact recovery completes"
            ),
            "publication_pending": {
                "is_terminal": False,
                "is_attempt_rerun": False,
                "research_effects_may_resume": False,
                "semantic_result_release_blocked": True,
                "next_stage_blocked": True,
                "only_exact_publication_recovery_permitted": True,
            },
        },
        "artifacts": {
            "predictions_and_actions_sealed_before_outcomes": True,
            "append_only_lessons": True,
            "same_ledger_benchmark": True,
            "independent_no_leverage_verification": True,
            "complete_metrics": True,
            "checksums": True,
            "failed_attempts_preserved": True,
            "failed_gate_complete_joint_diagnostic_externally_pinned": True,
            "later_stage_joint_release_only": True,
            "input_source_hashes_bound_before_stage_access": True,
            "pending_actions_and_lessons_preserved_across_boundaries": True,
            "raw_acquisition_bytes_never_exposed_as_public_objects": True,
            "terminal_pass_binds_existing_store_artifact_receipts": True,
            "terminal_reconstruction_material_is_exact_and_durable_before_intent": True,
            "publication_intent_is_fsynced_self_hashed_and_append_only": True,
            "publication_intent_prepare_freezes_exact_elapsed_hex": True,
            "publication_pending_releases_no_semantic_artifact": True,
            "publication_recovery_start_precedes_every_remote_action": True,
            "publication_push_requires_durable_single_push_marker": True,
            "normal_and_recovery_push_share_exact_marker_protocol": True,
            "remote_readback_evidence_schema_and_sentinels_are_frozen": True,
            "readback_and_push_bind_config_isolated_literal_remote_url": True,
            "observation_sequence_and_completion_phase_are_exact": True,
            "decisive_observation_precedes_fresh_capability_on_reopen": True,
            "recovery_receipt_requires_receipt_eligible_completion": True,
            "publication_worker_tree_is_killed_on_parent_exit": True,
            "publication_recovery_never_reopens_research_effects": True,
            "terminalization_capability_is_single_use": True,
        },
    }
    return copy.deepcopy(manifest)


CONTRACT_SHA256: Final[str] = (
    "64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d"
)
if canonical_sha256(build_contract_manifest()) != CONTRACT_SHA256:
    raise RuntimeError(
        "SEC/Gemma online-overlay manifest no longer matches its literal preregistration"
    )


def validate_contract_manifest(value: Any) -> dict[str, Any]:
    """Accept only the exact preregistered manifest and return a detached copy."""

    if not isinstance(value, dict):
        raise SecGemmaOnlineRiskOverlayContractError(
            "SEC/Gemma online-overlay contract must be a mapping"
        )
    expected = build_contract_manifest()
    try:
        observed_bytes = canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise SecGemmaOnlineRiskOverlayContractError(
            "SEC/Gemma online-overlay contract is not canonical JSON"
        ) from exc
    if observed_bytes != canonical_json_bytes(expected):
        raise SecGemmaOnlineRiskOverlayContractError(
            "SEC/Gemma online-overlay contract differs from preregistration"
        )
    return copy.deepcopy(expected)


__all__ = [
    "ACQUISITION_TERMINAL_RECONSTRUCTION_FIELDS",
    "BASELINE_POLICY_ID",
    "BASELINE_SOURCE_FILE",
    "BASELINE_SOURCE_SHA256",
    "BRANCH_NAME",
    "CONTRACT_SHA256",
    "CONTRACT_VERSION",
    "CONFIRMATION_ATTEMPT_ID",
    "CONTROL_FEATURES",
    "DEVELOPMENT_ACQUISITION_ID",
    "DEVELOPMENT_ATTEMPT_ID",
    "DEVELOPMENT_BLOCKS",
    "EMPTY_BYTES_SHA256",
    "EXPECTED_EDGE_GATE",
    "EXTERNAL_TAG_REF_TEMPLATE",
    "FEATURES",
    "FINAL_ATTEMPT_ID",
    "FINAL_REGISTRY_TAG_REF_TEMPLATE",
    "GOVERNANCE_CONTINGENCY_SECONDS",
    "HORIZON_SESSIONS",
    "MARKET_FEATURES",
    "MAX_PUBLICATION_RECOVERY_SECONDS",
    "MEANING_FEATURES",
    "MAX_MARKET_REQUESTS_PER_STAGE",
    "MAX_MARKET_SECONDS",
    "MAX_TOTAL_RUNTIME_SECONDS",
    "MINIMUM_CLASS_ROWS",
    "MINIMUM_TRAINING_ROWS",
    "MODEL_CONFIG_DIGEST",
    "MODEL_LAYER_DIGESTS",
    "MODEL_MANIFEST_SHA256",
    "MODEL_NAME",
    "OLLAMA_VERSION",
    "PROBABILITY_GATE",
    "PUBLICATION_CONFLICT_FIELDS",
    "PUBLICATION_FINALIZATION_PENDING_SECONDS",
    "PUBLICATION_INTENT_FIELDS",
    "PUBLICATION_INTENT_PREPARATION_SECONDS",
    "PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256",
    "PUBLICATION_NORMAL_NO_RECOVERY_COMPLETION_SHA256",
    "PUBLICATION_NORMAL_OPERATION_SHA256",
    "PUBLICATION_PRE_PUSH_AUTHORIZATION_FIELDS",
    "PUBLICATION_PUSH_COMMAND_PROFILE_SHA256",
    "PUBLICATION_RECEIPT_FIELDS",
    "PUBLICATION_RECOVERY_COMPLETION_GENESIS_SHA256",
    "PUBLICATION_RECOVERY_INVOCATION_COMPLETION_FIELDS",
    "PUBLICATION_RECOVERY_INVOCATION_START_FIELDS",
    "PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256",
    "PUBLICATION_RECOVERY_NO_REMOTE_OBSERVATION_SHA256",
    "PUBLICATION_REMOTE_READBACK_EVIDENCE_FIELDS",
    "PUBLICATION_REMOTE_READBACK_COMMAND_PROFILE_SHA256",
    "PUBLICATION_REMOTE_REF_ABSENT_SENTINEL",
    "PUBLICATION_REMOTE_OBSERVATION_FIELDS",
    "PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL",
    "PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL",
    "PUBLICATION_WORKER_OWNERSHIP_FIELDS",
    "PUBLICATION_WORKER_QUIESCENCE_FIELDS",
    "QUALITY_FEATURE",
    "RUNTIME_FINGERPRINT_SHA256",
    "SCORED_TERMINAL_RECONSTRUCTION_FIELDS",
    "SEMANTIC_FEATURES",
    "SOURCE_PIN_FILES",
    "SOURCE_PINS",
    "SUPERVISED_PUBLICATION_SECONDS",
    "TERMINALIZATION_CLAIM_FIELDS",
    "SecGemmaOnlineRiskOverlayContractError",
    "build_contract_manifest",
    "build_runtime_fingerprint_material",
    "canonical_json_bytes",
    "canonical_sha256",
    "validate_contract_manifest",
]
