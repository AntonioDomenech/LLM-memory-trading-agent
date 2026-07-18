from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path

import pytest

from agent_benchmark.sec_session_calendar import (
    CALENDAR_ID,
    EXPECTED_MARKET_HISTORY_SESSIONS,
    EXPECTED_SESSIONS,
    MARKET_HISTORY_CALENDAR_ID,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    ACQUISITION_TERMINAL_EVIDENCE_FIELDS,
    ACQUISITION_TERMINAL_RECONSTRUCTION_FIELDS,
    ACQUISITION_VALIDATION_FIELDS,
    BRANCH_NAME,
    CONFIRMATION_ATTEMPT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_ACQUISITION_ID,
    DEVELOPMENT_ATTEMPT_ID,
    EMPTY_BYTES_SHA256,
    EXTERNAL_TAG_REF_TEMPLATE,
    FEATURES,
    FINAL_ATTEMPT_ID,
    FINAL_REGISTRY_AUTHORIZATION_FIELDS,
    FINAL_REGISTRY_SUCCESSOR_FIELDS,
    FINAL_REGISTRY_TAG_REF_TEMPLATE,
    GOVERNANCE_CONTINGENCY_SECONDS,
    HORIZON_SESSIONS,
    MAX_PUBLICATION_RECOVERY_SECONDS,
    MAX_TOTAL_RUNTIME_SECONDS,
    MODEL_MANIFEST_SHA256,
    NEW_SOURCE_FILES,
    PUBLICATION_FINALIZATION_PENDING_SECONDS,
    PUBLICATION_CONFLICT_FIELDS,
    PUBLICATION_INTENT_FIELDS,
    PUBLICATION_INTENT_PREPARATION_SECONDS,
    PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256,
    PUBLICATION_NORMAL_OPERATION_SHA256,
    PUBLICATION_NORMAL_NO_RECOVERY_COMPLETION_SHA256,
    PUBLICATION_PRE_PUSH_AUTHORIZATION_FIELDS,
    PUBLICATION_PUSH_COMMAND_PROFILE_SHA256,
    PUBLICATION_RECEIPT_FIELDS,
    PUBLICATION_RECOVERY_COMPLETION_GENESIS_SHA256,
    PUBLICATION_RECOVERY_INVOCATION_COMPLETION_FIELDS,
    PUBLICATION_RECOVERY_INVOCATION_START_FIELDS,
    PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256,
    PUBLICATION_RECOVERY_NO_REMOTE_OBSERVATION_SHA256,
    PUBLICATION_REMOTE_READBACK_EVIDENCE_FIELDS,
    PUBLICATION_REMOTE_READBACK_COMMAND_PROFILE_SHA256,
    PUBLICATION_REMOTE_REF_ABSENT_SENTINEL,
    PUBLICATION_REMOTE_OBSERVATION_FIELDS,
    PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
    PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL,
    PUBLICATION_WORKER_OWNERSHIP_FIELDS,
    PUBLICATION_WORKER_QUIESCENCE_FIELDS,
    RUNTIME_FINGERPRINT_SHA256,
    SCORED_TERMINAL_EVIDENCE_FIELDS,
    SCORED_TERMINAL_RECONSTRUCTION_FIELDS,
    SOURCE_PIN_FILES,
    SOURCE_PINS,
    SUPERVISED_PUBLICATION_SECONDS,
    TERMINALIZATION_CLAIM_FIELDS,
    SecGemmaOnlineRiskOverlayContractError,
    build_contract_manifest,
    build_runtime_fingerprint_material,
    canonical_sha256,
    validate_contract_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
V22_PREREGISTRATION_REVISION = "a849b9d704ffd98547e570735a221b2b75f7db86"


def test_manifest_is_exact_deterministic_and_detached() -> None:
    first = build_contract_manifest()
    second = build_contract_manifest()

    assert first == second
    assert first is not second
    assert canonical_sha256(first) == CONTRACT_SHA256
    assert CONTRACT_SHA256 == (
        "64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d"
    )
    assert first["contract_version"] == CONTRACT_VERSION
    assert CONTRACT_VERSION == "aapl-sec-gemma-online-risk-overlay-v2-2"
    assert BRANCH_NAME == "codex/aapl-sec-gemma-online-risk-overlay-v2-2"
    assert first["features"]["ordered_names"] == list(FEATURES)
    assert first["features"]["count"] == 12
    assert first["policy"]["overlay_horizon_sessions"] == HORIZON_SESSIONS
    assert first["runtime"]["total_seconds_strictly_below"] == 3600
    assert MAX_TOTAL_RUNTIME_SECONDS == 3600

    first["features"]["ordered_names"][0] = "mutated"
    assert build_contract_manifest() == second


def test_contract_freezes_live_causal_learning_and_controls() -> None:
    manifest = validate_contract_manifest(build_contract_manifest())
    chronology = manifest["chronology"]
    learner = manifest["learner"]

    assert learner["training_mode"] == (
        "continuous expanding causal refit before each filing"
    )
    assert "t+21" in learner["training_rows"]
    assert "incremental" in learner["label"]
    assert "fixed baseline" in learner["probability_head"]
    assert chronology["updates_inside_every_period"] == (
        "permitted only after the complete 20-session outcome matures"
    )
    assert "no annual cutoff" in chronology["every_2025_decision_state"]
    assert "through-2023" in chronology["controls"]["final"]
    assert learner["same_session_matured_label_admission"] == (
        "maturity_session <= decision_session"
    )


def test_contract_freezes_long_cash_only_and_same_ledger_costs() -> None:
    manifest = build_contract_manifest()
    objective = manifest["objective"]
    policy = manifest["policy"]

    assert objective["allowed_target_exposures"] == [0, 1]
    assert objective["maximum_target_exposure"] == 1
    assert objective["maximum_realized_exposure"] == 1
    assert objective["shorting"] is False
    assert objective["leverage"] is False
    assert objective["borrowing"] is False
    assert objective["negative_cash"] is False
    assert objective["paid_api_calls"] == 0
    assert policy["costs_bps_per_changing_leg"] == [5, 10]
    assert policy["same_action_stream_at_both_costs"] is True


def test_contract_requires_learning_and_semantics_to_change_actions() -> None:
    gates = build_contract_manifest()["gates"]

    assert gates["zero_action_difference_is_rejection"] is True
    assert (
        gates["development"][
            "online_vs_block_frozen_action_differences_at_least"
        ]
        == 5
    )
    assert (
        gates["development"][
            "semantic_vs_no_filing_meaning_action_differences_at_least"
        ]
        == 5
    )
    assert (
        gates["final"]["post_2023_online_vs_frozen_action_differences_at_least"]
        == 3
    )
    assert gates["final"]["online_vs_frozen_continuous_10bps_strictly_positive"]


def test_bound_source_files_match_literal_sha256_pins() -> None:
    source_roles = set(SOURCE_PIN_FILES)
    pinned_roles = set(SOURCE_PINS)
    source_paths = set(SOURCE_PIN_FILES.values())

    assert source_roles == pinned_roles
    assert len(source_paths) == len(SOURCE_PIN_FILES)

    verified_roles: set[str] = set()
    verified_paths: set[str] = set()
    for role, relative_path in SOURCE_PIN_FILES.items():
        try:
            result = subprocess.run(
                [
                    "git",
                    "show",
                    f"{V22_PREREGISTRATION_REVISION}:{relative_path}",
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            pytest.fail(
                "failed to read frozen v2.2 source blob "
                f"for role {role!r} at path {relative_path!r}: {exc}",
                pytrace=False,
            )

        payload = result.stdout
        assert hashlib.sha256(payload).hexdigest() == SOURCE_PINS[role]
        verified_roles.add(role)
        verified_paths.add(relative_path)

    assert verified_roles == source_roles == pinned_roles
    assert verified_paths == source_paths


def test_final_reveal_registry_predecessor_pin_matches_literal_bytes() -> None:
    final_access = build_contract_manifest()["stage_access"]["final"]
    relative_path = final_access["predecessor_registry_pin_file"]
    payload = (REPO_ROOT / relative_path).read_bytes()

    assert hashlib.sha256(payload).hexdigest() == final_access[
        "predecessor_registry_pin_file_sha256"
    ]
    assert final_access["historical_final_reveal_count_lower_bound"] == 10
    assert len(final_access["predecessor_registry_sha256"]) == 64
    assert len(final_access["predecessor_registry_tip_sha256"]) == 64


def test_model_runtime_and_horizon_are_literal_not_placeholders() -> None:
    manifest = build_contract_manifest()
    gemma = manifest["gemma"]
    horizon = manifest["data"]["market"]["horizon"]

    assert gemma["model_manifest_sha256"] == MODEL_MANIFEST_SHA256
    assert len(MODEL_MANIFEST_SHA256) == 64
    assert gemma["runtime_fingerprint_sha256"] == RUNTIME_FINGERPRINT_SHA256
    assert len(RUNTIME_FINGERPRINT_SHA256) == 64
    assert canonical_sha256(build_runtime_fingerprint_material()) == (
        RUNTIME_FINGERPRINT_SHA256
    )
    assert RUNTIME_FINGERPRINT_SHA256 == (
        "816a7c1a6b1e87d083f8e0f85654f80ba0db124bf09fe8dedce960c8124e1c77"
    )
    assert build_runtime_fingerprint_material()[
        "show_semantic_sha256"
    ] == "5ccdf8b9a40bb762ea998dc9f5691ab2855d96833d60940b1d93686d08eb47e6"
    assert build_runtime_fingerprint_material()[
        "show_semantic_excluded_keys"
    ] == ["modified_at"]
    assert horizon == {
        "decision_close_index": "t",
        "entry_open_index": "t+1",
        "exit_open_index": "t+21",
        "held_open_to_open_intervals": 20,
        "label_maturity_session_index": "t+21",
    }


def test_calendar_ids_and_canonical_date_sequences_match_literal_pins() -> None:
    calendar = build_contract_manifest()["data"]["market"]["calendar"]

    assert calendar["calendar_id"] == CALENDAR_ID
    assert calendar["market_calendar_id"] == MARKET_HISTORY_CALENDAR_ID
    assert canonical_sha256(list(EXPECTED_SESSIONS)) == calendar[
        "calendar_dates_sha256"
    ]
    assert canonical_sha256(list(EXPECTED_MARKET_HISTORY_SESSIONS)) == calendar[
        "market_calendar_dates_sha256"
    ]


def test_baseline_is_prefix_invariant_raw_union_not_actionable_mask() -> None:
    policy = build_contract_manifest()["policy"]

    assert policy["baseline_kind"] == (
        "fixed raw expert union, not binary-regime selector"
    )
    assert policy["baseline_signal_column"] == "unfiltered_union_signal"
    assert policy["baseline_forbidden_column"] == (
        "unfiltered_union_target_exposure"
    )
    assert "may not rewrite" in policy["baseline_prefix_invariance"]
    assert "never suppresses" in policy["baseline_cooldown_independence"]


def test_stage_counts_attempts_and_metric_math_are_unambiguous() -> None:
    manifest = build_contract_manifest()

    assert manifest["data"]["sec"][
        "minimum_development_corpus_filings_2000_2018"
    ] == 72
    assert manifest["chronology"]["development_corpus"] == [
        "2000-01-01",
        "2018-12-31",
    ]
    assert manifest["stage_access"]["confirmation"]["one_shot"] is True
    assert manifest["stage_access"]["final"]["one_shot"] is True
    assert manifest["metric_definitions"]["positive_tolerance"] == 1e-12
    assert "max(ablation_brier, 1e-12)" in manifest["metric_definitions"][
        "brier_relative_improvement"
    ]


def test_all_effectful_stages_are_one_shot_and_locked_before_reveal() -> None:
    access = build_contract_manifest()["stage_access"]

    assert access["development_acquisition"]["attempt_id"] == (
        DEVELOPMENT_ACQUISITION_ID
    )
    assert access["development_acquisition"]["one_shot"] is True
    assert access["development_acquisition"][
        "consume_before_first_official_sec_or_market_network_request"
    ] is True
    assert access["development_acquisition"][
        "failed_or_pre_intent_indeterminate_acquisition_is_terminal"
    ] is True
    assert access["development_acquisition"][
        "post_intent_failure_remains_publication_pending"
    ] is True
    assert access["development"]["attempt_id"] == DEVELOPMENT_ATTEMPT_ID
    assert access["development"]["one_shot"] is True
    assert access["development"][
        "consume_before_first_real_gemma_call_or_first_canonical_market_value_read"
    ] is True
    assert access["development"][
        "latency_preflight_calls_occur_after_consumption"
    ] is True
    assert access["development"][
        "pre_intent_indeterminate_execution_is_terminal"
    ] is True
    assert access["development"][
        "post_intent_failure_remains_publication_pending"
    ] is True
    assert access["confirmation"]["attempt_id"] == CONFIRMATION_ATTEMPT_ID
    assert access["confirmation"]["consume_before_first_stage_network_request"] is True
    assert access["confirmation"][
        "consume_before_first_2019_feature_or_outcome_read"
    ] is True
    assert access["confirmation"][
        "post_intent_failure_remains_publication_pending"
    ] is True
    assert access["final"]["attempt_id"] == FINAL_ATTEMPT_ID
    assert access["final"]["consume_before_first_stage_network_request"] is True
    assert access["final"][
        "consume_before_first_2024_feature_or_outcome_read"
    ] is True
    assert access["final"][
        "post_intent_failure_remains_publication_pending"
    ] is True
    assert "semantic extraction" in access["result_release"]


def test_market_provider_windows_and_prefix_continuity_are_frozen() -> None:
    market = build_contract_manifest()["data"]["market"]
    source = market["source_acquisition"]

    assert source["provider_family"] == (
        "yahoo-finance-chart-v8-public-unauthenticated"
    )
    assert source["endpoint"] == (
        "https://query1.finance.yahoo.com/v8/finance/chart"
    )
    assert source["request_order"] == [
        "AAPL",
        "SPY",
        "QQQ",
        "IWM",
        "VIX",
        "TNX",
    ]
    assert source["request_count_each_stage"] == 6
    assert source["request_windows"]["development"]["period2_utc"] == 1546300800
    assert source["request_windows"]["confirmation"]["period2_utc"] == 1704067200
    assert source["request_windows"]["final"]["period2_utc"] == 1783728000
    assert source["request_windows"]["final"][
        "last_exposed_market_value_session"
    ] == "2026-07-09"
    assert source["request_windows"]["final"][
        "last_scored_fill_origin_decision_session"
    ] == "2026-07-08"
    assert source["request_windows"]["final"][
        "last_pending_prediction_session"
    ] == "2026-07-09"
    assert market["ledger_price_fields"] == [
        "raw_open",
        "raw_close",
        "adjusted_close",
    ]
    assert "float bit pattern" in source["canonical_prefix_rule"]
    assert "terminally fails" in source["canonical_prefix_rule"]
    assert "only after their durable stage locks" in source[
        "visibility_and_lock_rule"
    ]
    assert "never returned as a public mapping" in source["opaque_vault_rule"]
    assert "adjusted_open" in market["ledger_price_validation"]
    assert "terminally fails the stage" in market["ledger_price_validation"]
    coverage = market["required_market_coverage"]
    assert coverage["aapl_exact_expected_session_counts"] == {
        "development_through_2018_12_31": 5283,
        "confirmation_through_2023_12_29": 6541,
        "final_exposed_through_2026_07_09": 7172,
        "final_transport_through_2026_07_10": 7173,
    }
    assert coverage["context_first_accepted_session"]["IWM"] == "2000-05-26"
    assert coverage["context_allowed_missing_sessions"]["SPY"] == []
    assert coverage["context_allowed_missing_sessions"]["TNX"][-1] == (
        "2016-11-11"
    )
    assert "truncated 253-row tail" in coverage["coverage_rule"]


def test_sec_acquisition_is_replayed_and_future_metadata_stays_opaque() -> None:
    sec = build_contract_manifest()["data"]["sec"]
    replay = sec["detached_replay"]

    assert "validate_detached_catalog_replay" in replay["catalogue"]
    assert "validate_detached_stage_content_replay" in replay["stage_content"]
    assert "caller-supplied metadata is never an authority" in replay[
        "universe_membership"
    ]
    assert "only inside the opaque acquisition vault" in replay[
        "future_metadata_boundary"
    ]


def test_terminal_reports_require_exact_evidence_and_external_pins() -> None:
    integrity = build_contract_manifest()["execution_integrity"]
    acquisition = integrity["acquisition_terminal_pass"]
    scored = integrity["scored_terminal_pass"]
    pin = integrity["external_report_pin"]

    assert acquisition["exact_checks"] == [
        "exact_raw_bytes_replayed_sha256",
        "request_receipts_reconciled_sha256",
        "stage_and_attempt_scope_bound_sha256",
        "private_identity_digest_only_sha256",
        "market_prefix_continuity_replayed_sha256",
        "blinded_model_requests_replayed_sha256",
        "request_byte_retry_redirect_caps_reconciled_sha256",
    ]
    assert acquisition["arbitrary_all_true_mapping_forbidden"] is True
    assert acquisition["validation_fields"] == list(
        ACQUISITION_VALIDATION_FIELDS
    )
    assert acquisition["terminal_evidence_fields"] == list(
        ACQUISITION_TERMINAL_EVIDENCE_FIELDS
    )
    assert "publication_intent_sha256" in ACQUISITION_TERMINAL_EVIDENCE_FIELDS
    assert (
        "publication_intent_store_receipt_sha256"
        in ACQUISITION_TERMINAL_EVIDENCE_FIELDS
    )
    assert "publication_receipt_sha256" in ACQUISITION_TERMINAL_EVIDENCE_FIELDS
    assert (
        "publication_receipt_store_receipt_sha256"
        in ACQUISITION_TERMINAL_EVIDENCE_FIELDS
    )
    assert scored["every_literal_gate_true"] is True
    assert scored["terminal_evidence_fields"] == list(
        SCORED_TERMINAL_EVIDENCE_FIELDS
    )
    assert "publication_intent_sha256" in SCORED_TERMINAL_EVIDENCE_FIELDS
    assert (
        "publication_intent_store_receipt_sha256"
        in SCORED_TERMINAL_EVIDENCE_FIELDS
    )
    assert "publication_receipt_sha256" in SCORED_TERMINAL_EVIDENCE_FIELDS
    assert (
        "publication_receipt_store_receipt_sha256"
        in SCORED_TERMINAL_EVIDENCE_FIELDS
    )
    assert scored["arbitrary_all_true_mapping_forbidden"] is True
    assert "externally pin its hash" in integrity["failed_scored_gate"]
    assert pin["ordinary_terminal_ref_template"] == (
        "refs/tags/sec-gemma-online-risk-overlay-v2-2/"
        "attempts/{attempt_id}/terminal"
    )
    assert pin["final_registry_ref_template"] == (
        "refs/tags/sec-gemma-online-risk-overlay-v2-2/"
        "registry/{attempt_id}/successor"
    )
    assert pin["ordinary_terminal_ref_template"] == EXTERNAL_TAG_REF_TEMPLATE
    assert pin["final_registry_ref_template"] == (
        FINAL_REGISTRY_TAG_REF_TEMPLATE
    )
    assert "{report_kind}" not in EXTERNAL_TAG_REF_TEMPLATE
    assert "{artifact_sha256}" not in EXTERNAL_TAG_REF_TEMPLATE
    assert "shares the final attempt ID" in pin["stable_ref_rule"]
    assert pin["deletion_force_or_reuse_forbidden"] is True
    registry = integrity["final_registry_authorization"]
    assert registry["successor_fields"] == list(
        FINAL_REGISTRY_SUCCESSOR_FIELDS
    )
    assert registry["authorization_fields"] == list(
        FINAL_REGISTRY_AUTHORIZATION_FIELDS
    )
    assert registry["tag_ref_template"] == FINAL_REGISTRY_TAG_REF_TEMPLATE
    assert "arbitrary hexadecimal string" in registry[
        "attempt_plan_input"
    ]
    assert "can never register or consume" in integrity["test_double_boundary"]
    assert "strictly below 3600 seconds" in integrity["attempt_deadline_scope"]


def test_publication_intent_and_recovery_protocol_is_exact() -> None:
    manifest = build_contract_manifest()
    integrity = manifest["execution_integrity"]
    intent = integrity["publication_intent"]
    recovery = integrity["publication_recovery"]
    pending = manifest["stage_access"]["publication_pending"]

    assert integrity["opaque_types"]["publication_receipt"] == (
        "VerifiedDurablePublicationReceipt"
    )
    assert integrity["opaque_types"]["publication_remote_observation"] == (
        "VerifiedDurablePublicationRemoteObservation"
    )
    assert integrity["opaque_types"]["publication_capability"] == (
        "PublicationRecoveryCapability"
    )
    assert integrity["opaque_types"]["terminalization_capability"] == (
        "TerminalizationCapability"
    )
    assert PUBLICATION_INTENT_FIELDS == (
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
    assert intent["fields"] == list(PUBLICATION_INTENT_FIELDS)
    assert intent["fixed_intent_status"] == "publication_pending"
    assert intent["fixed_values"] == {
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
    }
    assert "before building the publication intent" in intent[
        "canonical_tag_preparation_order"
    ]
    assert "publication_intent_prepare" in intent[
        "durability_before_remote_effect"
    ]
    assert intent["append_protocol"]["order"] == [
        "normal_elapsed_hex_frozen_in_exact_intent_bytes",
        "prepare_anchor_fsynced",
        "exact_journal_row_committed",
        "committed_anchor_fsynced",
        "opaque_store_receipt_issued",
    ]
    assert "exact fully committed intent is idempotent" in intent[
        "append_protocol"
    ]["exact_reconciliation"]
    assert "normal_attempt_elapsed_at_intent_prepare_hex" in intent[
        "append_protocol"
    ]["pending_elapsed_binding"]
    assert "strictly below 3600" in intent["append_protocol"][
        "pending_elapsed_binding"
    ]
    assert "before the database append" in intent["append_protocol"][
        "pending_elapsed_binding"
    ]
    assert "complete exact canonical self-hashed intent row bytes" in intent[
        "append_protocol"
    ]["prepare_anchor_frozen_payload"]
    assert intent["remote_push_without_intent_forbidden"] is True
    assert "exact committed database tip immediately before" in intent[
        "noncircular_store_binding"
    ]
    assert "outside the self-hashed intent body" in intent[
        "noncircular_store_binding"
    ]
    assert "excluded from semantic record_counts" in intent[
        "noncircular_store_binding"
    ]
    assert intent["live_monotonic_clock_or_deadline_field_forbidden"] is True
    assert "destroys and invalidates" in intent["authority_cutoff"]
    assert "no extraction, prediction, action, metric, gate" in intent[
        "pending_receipt_rule"
    ]
    assert "arbitrary downtime" in recovery["restart_boot_and_downtime_rule"]
    assert "may issue at most one push command" in recovery[
        "exact_publish_or_recover"
    ]
    assert "its own new exact durable observation again proves the ref absent" in recovery[
        "exact_publish_or_recover"
    ]
    assert recovery["push_commands_per_invocation_at_most"] == 1
    assert "poisons the publication state" in recovery[
        "conflicting_tag_rule"
    ]
    assert recovery["poison_is_irreversible"] is True
    assert "preserves the exact durable publication-pending intent" in recovery[
        "transient_failure_rule"
    ]
    assert "reconcile an exact durable publication intent and then" in recovery[
        "store_reopen_precedence"
    ]
    assert "existing terminal-indeterminate recovery" in recovery[
        "store_reopen_precedence"
    ]
    sweep = recovery["observation_reopen_precedence"]
    assert sweep["ordered_sweep"][0] == (
        "reconcile_every_partial_transport_manifest_prepare_row_and_commit"
    )
    assert sweep["ordered_sweep"][1] == (
        "validate_exact_transport_manifest_and_reject_extras"
    )
    assert sweep["ordered_sweep"][2] == (
        "reconcile_every_partial_observation_prepare_row_and_commit"
    )
    assert sweep["ordered_sweep"][-1] == (
        "only_then_consider_fresh_capability"
    )
    assert "first valid terminal observation" in sweep[
        "first_terminal_observation_rule"
    ]
    assert "post_push absent maps to post_push_ref_absent" in sweep[
        "recovery_completion_reconstruction"
    ]
    assert "0x1.2c00000000000p+8" in sweep[
        "recovery_completion_reconstruction"
    ]
    assert "normal exact_expected observation" in sweep[
        "normal_terminal_observation_reconciliation"
    ]
    assert "lone pre_push absent observation" in sweep[
        "pre_push_absent_crash_rule"
    ]
    assert "can never be discarded" in sweep["capability_block"]
    assert "every transport manifest" in sweep["capability_block"]
    assert "existing terminal_intent and terminal_committed" in recovery[
        "completion_rule"
    ]
    assert recovery["each_invocation_seconds_at_most"] == 300
    assert recovery["killable_supervised_process_required"] is True
    assert recovery["process_tree_termination_on_deadline_required"] is True
    assert recovery["zero_model_market_or_data_effects"] is True
    assert recovery["unknown_descendant_record_poisons"] is True
    assert recovery["whitelisted_descendant_governance_records"] == [
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
    ]
    capabilities = recovery["capability_state_machine"]
    assert "new store-session nonce" in capabilities["restart_reissue"]
    assert "earlier process, store session, or nonce is stale" in capabilities[
        "restart_reissue"
    ]
    assert "fresh opaque capability" in capabilities["publication_capability"]
    assert "unreconciled observation" in capabilities[
        "publication_capability"
    ]
    assert "fresh opaque capability" in capabilities[
        "terminalization_capability"
    ]
    assert pending == {
        "is_terminal": False,
        "is_attempt_rerun": False,
        "research_effects_may_resume": False,
        "semantic_result_release_blocked": True,
        "next_stage_blocked": True,
        "only_exact_publication_recovery_permitted": True,
    }


def test_publication_receipt_reconstruction_and_terminal_recovery_are_exact() -> None:
    integrity = build_contract_manifest()["execution_integrity"]
    reconstruction = integrity["publication_pending_terminal_reconstruction"]
    transport_isolation = integrity["publication_transport_isolation"]
    observation = integrity["publication_remote_observation"]
    receipt = integrity["publication_receipt"]
    recovery = integrity["publication_recovery"]
    claim = integrity["terminalization_claim"]
    terminal = integrity["terminal_completion_recovery"]

    assert PUBLICATION_RECEIPT_FIELDS == (
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
    assert PUBLICATION_REMOTE_OBSERVATION_FIELDS == (
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
    assert PUBLICATION_REMOTE_READBACK_EVIDENCE_FIELDS == (
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
    assert observation["fields"] == list(PUBLICATION_REMOTE_OBSERVATION_FIELDS)
    assert observation["readback_evidence_fields"] == list(
        PUBLICATION_REMOTE_READBACK_EVIDENCE_FIELDS
    )
    assert observation["readback_command_profile"]["argv_template"] == [
        "ls-remote",
        "--tags",
        "{remote_url}",
        "{tag_ref}",
        "{tag_ref}^{}",
    ]
    assert observation["readback_command_profile"][
        "profile_template_sha256"
    ] == PUBLICATION_REMOTE_READBACK_COMMAND_PROFILE_SHA256
    assert observation["readback_command_profile"]["command_count"] == 1
    assert observation["readback_command_profile"]["executable"] == (
        "{verified_git_executable_path}"
    )
    assert observation["readback_command_profile"][
        "child_environment_mode"
    ] == "exact_allowlist_no_inheritance"
    assert observation["readback_command_profile"][
        "inherited_environment_forbidden"
    ] is True
    exact_child_environment = observation["readback_command_profile"][
        "exact_child_environment"
    ]
    assert exact_child_environment == {
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
    }
    environment_keys = observation["readback_command_profile"][
        "child_environment_key_set"
    ]
    assert set(environment_keys) == set(exact_child_environment)
    assert len({name.casefold() for name in environment_keys}) == len(
        environment_keys
    )
    readback_profile_hash_material = copy.deepcopy(
        observation["readback_command_profile"]
    )
    readback_profile_hash_material.pop("profile_template_sha256")
    assert canonical_sha256(readback_profile_hash_material) == (
        PUBLICATION_REMOTE_READBACK_COMMAND_PROFILE_SHA256
    )
    assert observation["readback_command_profile"]["working_directory"] == (
        "verified_isolated_transport_git_directory"
    )
    assert "no alternate option order" in observation[
        "readback_command_profile"
    ]["sequence_hash_rule"]
    assert observation["scope"] == [
        "normal_publication",
        "publication_recovery",
    ]
    assert observation["observed_ref_states"] == [
        "absent",
        "exact_expected",
        "conflicting",
    ]
    assert observation["literal_observed_value_sentinels"] == {
        "ref_absent": PUBLICATION_REMOTE_REF_ABSENT_SENTINEL,
        "value_missing": PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL,
        "value_malformed": PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
        "no_prior_push_command_sha256": (
            PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256
        ),
    }
    assert observation["accepted_observation_operations"][
        "normal_publication"
    ]["observation_operation_sha256"] == PUBLICATION_NORMAL_OPERATION_SHA256
    assert "exact committed recovery start hash" in observation[
        "operation_binding_rule"
    ]
    evidence_matrix = observation["readback_evidence_cross_field_matrix"]
    assert evidence_matrix["completed_absent"] == {
        "process_exit_status": "exited",
        "process_exit_code": 0,
        "transport_status": "completed",
        "ref_lookup_status": "absent",
        "stdout_byte_count": 0,
        "stdout_sha256": EMPTY_BYTES_SHA256,
        "stderr_byte_count": 0,
        "stderr_sha256": EMPTY_BYTES_SHA256,
        "raw_ref_object_value": PUBLICATION_REMOTE_REF_ABSENT_SENTINEL,
        "raw_peeled_value": PUBLICATION_REMOTE_REF_ABSENT_SENTINEL,
        "raw_tag_message_sha256": PUBLICATION_REMOTE_REF_ABSENT_SENTINEL,
        "observation_permitted": True,
        "projected_observed_ref_state": "absent",
    }
    assert evidence_matrix["exited_nonzero"][
        "process_exit_code"
    ] == "any_integer_except_zero"
    assert evidence_matrix["exited_nonzero"][
        "ref_lookup_status"
    ] == "unknown"
    assert evidence_matrix["exited_nonzero"][
        "observation_permitted"
    ] is False
    assert evidence_matrix["deadline"]["process_exit_code"] is None
    assert evidence_matrix["deadline"]["observation_permitted"] is False
    absent_authorizers = [
        name
        for name, row in evidence_matrix.items()
        if row.get("observation_permitted") is True
        and row.get("projected_observed_ref_state") == "absent"
    ]
    assert absent_authorizers == ["completed_absent"]
    for name in (
        "completed_malformed_output",
        "exited_nonzero",
        "deadline",
        "parent_interrupted",
        "spawn_failed",
    ):
        assert evidence_matrix[name]["ref_lookup_status"] == "unknown"
        assert evidence_matrix[name]["observation_permitted"] is False
    assert "only completed_absent may project" in observation[
        "evidence_matrix_exhaustiveness_rule"
    ]
    assert "annotated-tag object OID content-addresses" in observation[
        "present_observation_projection_rule"
    ]
    assert "literal remote_url" in observation[
        "transport_isolation_binding_rule"
    ]
    assert transport_isolation["readback_command_profile_sha256"] == (
        PUBLICATION_REMOTE_READBACK_COMMAND_PROFILE_SHA256
    )
    assert transport_isolation["push_command_profile_sha256"] == (
        PUBLICATION_PUSH_COMMAND_PROFILE_SHA256
    )
    assert transport_isolation["local_config_exact_entries"] == [
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
    ]
    assert transport_isolation["local_config_allowlist_only"] is True
    assert transport_isolation["child_environment_mode"] == (
        "exact_allowlist_no_inheritance"
    )
    assert transport_isolation["inherited_environment_forbidden"] is True
    assert transport_isolation["allowed_remote_url"] == (
        "https://github.com/AntonioDomenech/LLM-memory-trading-agent.git"
    )
    assert transport_isolation["remote_url_scheme"] == "https"
    assert transport_isolation["path_lookup_forbidden"] is True
    for forbidden_name in (
        "GIT_CONFIG_PARAMETERS",
        "GIT_DIR",
        "GIT_COMMON_DIR",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_INDEX_FILE",
        "GIT_SSH",
        "GIT_PROXY_COMMAND",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "GIT_SSL_NO_VERIFY",
        "GIT_SSL_CAINFO",
        "GIT_SSL_CAPATH",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        assert forbidden_name in transport_isolation[
            "unlisted_environment_rule"
        ]
    manifest_fields = transport_isolation[
        "isolated_git_directory_manifest_fields"
    ]
    for identity_field in (
        "git_executable_path",
        "git_executable_sha256",
        "git_exec_path",
        "git_exec_path_directory_manifest_sha256",
        "git_remote_https_executable_path",
        "git_remote_https_executable_sha256",
        "credential_helper_executable_path",
        "credential_helper_executable_sha256",
        "command_interpreter_executable_path",
        "command_interpreter_executable_sha256",
        "transport_executable_closure_manifest_sha256",
        "child_environment_policy_id",
        "child_environment_sha256",
        "empty_hooks_directory_path",
        "empty_hooks_directory_identity_sha256",
        "empty_hooks_directory_listing_sha256",
        "path_lookup_forbidden",
        "remote_url_scheme",
    ):
        assert identity_field in manifest_fields
    assert "path lookup is forbidden" in transport_isolation[
        "executable_identity_rule"
    ]
    assert "leading exclamation mark" in transport_isolation[
        "credential_helper_rule"
    ]
    assert "double-quoted" in transport_isolation["credential_helper_rule"]
    assert "core.hooksPath" in transport_isolation["hook_disable_rule"]
    assert "canonical empty listing" in transport_isolation[
        "hook_disable_rule"
    ]
    assert "rechecked immediately before every Git process" in (
        transport_isolation["hook_disable_rule"]
    )
    assert "pre-push" in transport_isolation["hook_disable_rule"]
    assert "never remote_name" in transport_isolation[
        "literal_endpoint_rule"
    ]
    assert "remote, URL rewrite" in transport_isolation[
        "literal_endpoint_rule"
    ]
    assert "objects/info/alternates" in transport_isolation[
        "object_access_rule"
    ]
    assert len(observation["allowed_observation_sequences"]) == 7
    assert observation["allowed_observation_sequences"][0] == []
    assert observation["allowed_observation_sequences"][1] == [
        {
            "observation_ordinal": 1,
            "observation_phase": "pre_push",
            "observed_ref_state": "absent",
        }
    ]
    assert observation["allowed_observation_sequences"][-1] == [
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
    ]
    assert "empty sequence is valid only" in observation[
        "observation_sequence_rule"
    ]
    assert "no gap, duplicate, third observation" in observation[
        "observation_sequence_rule"
    ]
    assert "every pre_push observation stores" in observation[
        "push_phase_binding_rule"
    ]
    assert "post_push observation is valid only after" in observation[
        "push_phase_binding_rule"
    ]
    assert "transport unavailable" in observation["classification_rule"]
    assert "integer exit code zero" in observation["classification_rule"]
    assert "can never be encoded as absence" in observation[
        "classification_rule"
    ]
    assert "exactly the listed evidence fields" in observation[
        "evidence_binding_rule"
    ]
    assert observation["append_protocol"]["order"][2] == (
        "prepare_anchor_with_exact_evidence_and_row_bytes_fsynced"
    )
    assert "exact canonical remote-readback evidence bytes" in observation[
        "append_protocol"
    ]["exact_reconciliation"]
    assert "no remote readback may authorize a push" in observation[
        "durability_gate"
    ]
    assert "becomes proven only when" in observation[
        "conflict_proof_boundary"
    ]
    assert "before the observation prepare anchor" in observation[
        "conflict_proof_boundary"
    ]
    assert receipt["fields"] == list(PUBLICATION_RECEIPT_FIELDS)
    assert receipt["append_protocol"]["order"] == [
        "verified_external_publication_observation_committed",
        "matching_receipt_eligible_recovery_completion_committed_if_recovery",
        "prepare_anchor_with_exact_row_bytes_fsynced",
        "exact_journal_row_committed",
        "committed_anchor_fsynced",
        "opaque_store_receipt_issued",
    ]
    assert "exact canonical self-hashed receipt row bytes" in receipt[
        "append_protocol"
    ]["exact_reconciliation"]
    assert "exact fully committed receipt is idempotent" in receipt[
        "append_protocol"
    ]["exact_reconciliation"]
    assert "durably committed exact_expected observation" in receipt[
        "observation_binding_rule"
    ]
    assert "for normal_publication" in receipt[
        "completion_and_authorization_binding_rule"
    ]
    assert "must exist before receipt prepare" in receipt[
        "completion_and_authorization_binding_rule"
    ]
    assert receipt["normal_no_recovery_completion_sha256"] == (
        PUBLICATION_NORMAL_NO_RECOVERY_COMPLETION_SHA256
    )
    assert "publication_receipt_store_receipt_sha256" in receipt[
        "terminal_evidence_precondition"
    ]
    assert "skips all remote publication" in receipt[
        "receipt_committed_restart_rule"
    ]

    assert ACQUISITION_TERMINAL_RECONSTRUCTION_FIELDS == (
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
    assert SCORED_TERMINAL_RECONSTRUCTION_FIELDS == (
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
    assert reconstruction["acquisition_fields"] == list(
        ACQUISITION_TERMINAL_RECONSTRUCTION_FIELDS
    )
    assert reconstruction["scored_fields"] == list(
        SCORED_TERMINAL_RECONSTRUCTION_FIELDS
    )
    assert reconstruction["journal_record_table"] == (
        "terminal_reconstruction_materials"
    )
    assert reconstruction["append_protocol"]["order"][-1] == (
        "publication_intent_prepare_permitted"
    )
    assert "exact canonical self-hashed row bytes" in reconstruction[
        "append_protocol"
    ]["exact_reconciliation"]
    assert "terminal_reconstruction_material_store_receipt_sha256" in (
        reconstruction["intent_binding"]
    )
    assert "sealed acquisition phase evidence" in reconstruction[
        "acquisition_rehydration"
    ]
    assert "without chronological replay" in reconstruction[
        "scored_reconstruction"
    ]
    assert reconstruction["new_research_or_recomputation_forbidden"] is True
    assert "intent-bound terminal artifact payload" in reconstruction[
        "read_only_local_material_permitted"
    ]

    assert "may convert the attempt to terminal-indeterminate" in terminal[
        "publication_intent_precludes_indeterminate"
    ]
    assert "claim_without_terminal_intent window" in terminal[
        "receipt_without_terminal_intent"
    ]
    assert "append the one exact predeclared terminal_intent" in terminal[
        "claim_without_terminal_intent"
    ]
    assert "without issuing another capability or claim" in terminal[
        "claim_without_terminal_intent"
    ]
    assert "terminal_intent anchor exists" in terminal[
        "terminal_intent_without_database_transition"
    ]
    assert "terminal database transition exists" in terminal[
        "database_transition_without_terminal_committed_anchor"
    ]
    assert "append only the exact terminal_committed anchor" in terminal[
        "database_transition_without_terminal_committed_anchor"
    ]
    assert "read-only validation and rehydration" in terminal[
        "fully_committed_idempotency"
    ]
    assert "with no new research computation" in terminal[
        "fully_committed_idempotency"
    ]
    assert "single-use terminalization capability" in recovery[
        "completion_rule"
    ]
    assert claim["fields"] == list(TERMINALIZATION_CLAIM_FIELDS)
    assert "atomically invalidated" in claim["atomic_single_use_rule"]
    assert "a second terminal_intent" in claim[
        "concurrent_or_repeated_entry_rule"
    ]


def test_recovery_and_poison_governance_records_are_exact() -> None:
    recovery = build_contract_manifest()["execution_integrity"][
        "publication_recovery"
    ]
    start = recovery["recovery_invocation_start_record"]
    authorization = recovery["pre_push_authorization_record"]
    completion = recovery["recovery_invocation_completion_record"]
    interruption = recovery["interrupted_invocation_reconciliation"]
    conflict = recovery["conflict_poison_record"]

    assert PUBLICATION_RECOVERY_INVOCATION_START_FIELDS == (
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
    assert PUBLICATION_PRE_PUSH_AUTHORIZATION_FIELDS == (
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
    assert PUBLICATION_RECOVERY_INVOCATION_COMPLETION_FIELDS == (
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
    assert PUBLICATION_CONFLICT_FIELDS == (
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
    assert start["fields"] == list(
        PUBLICATION_RECOVERY_INVOCATION_START_FIELDS
    )
    assert start["self_hashed_and_externally_anchored"] is True
    assert start["fixed_values"]["invocation_seconds_cap"] == 300
    assert start["record_identity_fields"] == [
        "store_instance_id",
        "attempt_id",
        "invocation_ordinal",
    ]
    assert start["first_prior_completion_sha256"] == (
        PUBLICATION_RECOVERY_COMPLETION_GENESIS_SHA256
    )
    assert start["first_prior_cumulative_recovery_seconds"] == "0x0.0p+0"
    assert "binary64 hexadecimal value" in start["time_encoding_rule"]
    assert "before any remote readback" in start["before_remote_action_rule"]
    assert authorization["fields"] == list(
        PUBLICATION_PRE_PUSH_AUTHORIZATION_FIELDS
    )
    assert authorization["fixed_values"]["push_command_limit"] == 1
    assert authorization["record_identity_fields"][-1] == (
        "pre_push_authorization_marker_key"
    )
    assert authorization["accepted_authorization_operations"][
        "normal_publication"
    ] == {
        "authorization_operation_sha256": PUBLICATION_NORMAL_OPERATION_SHA256,
        "authorization_operation_ordinal": 0,
        "required_recovery_start": False,
    }
    assert authorization["accepted_authorization_operations"][
        "publication_recovery"
    ]["required_recovery_start"] is True
    assert "normal_publication binds" in authorization[
        "operation_binding_rule"
    ]
    assert "only after the exact operation foundation" in authorization[
        "marker_key_derivation_rule"
    ]
    assert "never a recovery-start field" in authorization[
        "marker_key_derivation_rule"
    ]
    assert "no normal or recovery push is legal without this marker" in authorization[
        "issuance_rule"
    ]
    assert authorization["push_command_profile"]["profile_template_sha256"] == (
        PUBLICATION_PUSH_COMMAND_PROFILE_SHA256
    )
    assert authorization["push_command_profile"]["argv_template"] == [
        "push",
        "--porcelain",
        "--no-verify",
        "{remote_url}",
        "{expected_tag_object_sha1}:{tag_ref}",
    ]
    assert authorization["push_command_profile"]["force_forbidden"] is True
    assert authorization["push_command_profile"]["executable"] == (
        "{verified_git_executable_path}"
    )
    assert authorization["push_command_profile"][
        "exact_child_environment"
    ] == (
        build_contract_manifest()["execution_integrity"][
            "publication_remote_observation"
        ]["readback_command_profile"]["exact_child_environment"]
    )
    assert authorization["push_command_profile"][
        "child_environment_key_set"
    ] == build_contract_manifest()["execution_integrity"][
        "publication_remote_observation"
    ][
        "readback_command_profile"
    ][
        "child_environment_key_set"
    ]
    push_profile_hash_material = copy.deepcopy(
        authorization["push_command_profile"]
    )
    push_profile_hash_material.pop("profile_template_sha256")
    assert canonical_sha256(push_profile_hash_material) == (
        PUBLICATION_PUSH_COMMAND_PROFILE_SHA256
    )
    assert "literal allowed HTTPS remote_url" in authorization[
        "push_command_evidence_rule"
    ]
    assert "top-level Git subprocess" in authorization[
        "push_command_evidence_rule"
    ]
    assert "only internal shell use" in authorization[
        "push_command_evidence_rule"
    ]
    assert "repository hook" in authorization["push_command_evidence_rule"]
    assert "--no-verify" in authorization["push_command_evidence_rule"]
    assert "uses no remote name" in authorization[
        "push_command_evidence_rule"
    ]
    assert "before the sole push command" in authorization["issuance_rule"]
    assert completion["fields"] == list(
        PUBLICATION_RECOVERY_INVOCATION_COMPLETION_FIELDS
    )
    assert completion["self_hashed_and_externally_anchored"] is True
    assert completion["record_identity_fields"] == [
        "store_instance_id",
        "attempt_id",
        "invocation_ordinal",
    ]
    matrix = completion["outcome_cross_field_matrix"]
    assert list(matrix) == [
        "remote_unavailable_before_observation",
        "remote_exact_without_push",
        "remote_conflict_poisoned_without_push",
        "remote_absent_authorization_not_committed",
        "authorized_push_not_issued",
        "push_issued_unconfirmed",
        "post_push_ref_absent",
        "published_exact_after_push",
        "post_push_conflict_poisoned",
        "interrupted_without_authorization",
        "interrupted_with_authorization",
    ]
    assert {
        row["outcome"] for row in matrix.values()
    } == {
        "remote_unavailable_before_observation",
        "remote_exact_without_push",
        "remote_conflict_poisoned_without_push",
        "remote_absent_authorization_not_committed",
        "authorized_push_not_issued",
        "push_issued_unconfirmed",
        "post_push_ref_absent",
        "published_exact_after_push",
        "post_push_conflict_poisoned",
        "interrupted_before_completion",
    }
    no_authorization = PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
    no_observation = PUBLICATION_RECOVERY_NO_REMOTE_OBSERVATION_SHA256
    assert matrix["remote_unavailable_before_observation"] == {
        "outcome": "remote_unavailable_before_observation",
        "pre_push_authorization_sha256": no_authorization,
        "remote_observation_sha256": no_observation,
        "push_command_count_upper_bound": 0,
        "conflict_poison_requirement": "forbidden",
        "publication_receipt_eligible": False,
    }
    assert matrix["remote_exact_without_push"][
        "publication_receipt_eligible"
    ] is True
    assert matrix["published_exact_after_push"][
        "publication_receipt_eligible"
    ] is True
    assert matrix["authorized_push_not_issued"][
        "push_command_count_upper_bound"
    ] == 0
    assert matrix["authorized_push_not_issued"][
        "pre_push_authorization_sha256"
    ] == "exact_committed_pre_push_authorization_sha256"
    assert matrix["push_issued_unconfirmed"][
        "push_command_count_upper_bound"
    ] == 1
    assert matrix["push_issued_unconfirmed"][
        "remote_observation_sha256"
    ] == "exact_absent_remote_observation_sha256"
    assert matrix["post_push_ref_absent"] == {
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
    }
    assert matrix["remote_conflict_poisoned_without_push"][
        "conflict_poison_requirement"
    ] == "exact_committed_matching_conflict_poison"
    assert matrix["post_push_conflict_poisoned"][
        "conflict_poison_requirement"
    ] == "exact_committed_matching_conflict_poison"
    assert matrix["interrupted_without_authorization"][
        "pre_push_authorization_sha256"
    ] == no_authorization
    assert matrix["interrupted_without_authorization"][
        "remote_observation_sha256"
    ] == no_observation
    assert matrix["interrupted_without_authorization"][
        "push_command_count_upper_bound"
    ] == 0
    assert matrix["interrupted_with_authorization"][
        "pre_push_authorization_sha256"
    ] == "exact_committed_pre_push_authorization_sha256"
    assert matrix["interrupted_with_authorization"][
        "remote_observation_sha256"
    ] == no_observation
    assert matrix["interrupted_with_authorization"][
        "push_command_count_upper_bound"
    ] == 1
    assert "matches exactly one matrix row" in completion[
        "outcome_matrix_rule"
    ]
    requirements = completion["outcome_observation_requirements"]
    assert requirements["remote_exact_without_push"] == {
        "observation_phase": "pre_push",
        "observation_ordinal": 1,
        "observed_ref_state": "exact_expected",
        "required_sequence": ["pre_push:exact_expected"],
    }
    assert requirements["push_issued_unconfirmed"][
        "forbidden_committed_post_push_observation"
    ] is True
    assert requirements["post_push_ref_absent"] == {
        "observation_phase": "post_push",
        "observation_ordinal": 2,
        "observed_ref_state": "absent",
        "required_sequence": [
            "pre_push:absent",
            "post_push:absent",
        ],
    }
    assert requirements["published_exact_after_push"][
        "observation_phase"
    ] == "post_push"
    assert requirements["post_push_conflict_poisoned"][
        "observation_phase"
    ] == "post_push"
    assert "can never reference a pre_push observation" in completion[
        "outcome_observation_binding_rule"
    ]
    assert "last exact durably committed observation" in completion[
        "clean_completion_observation_rule"
    ]
    assert "authorized_push_not_issued" in completion[
        "authorization_and_push_rule"
    ]
    assert "interrupted completion may conservatively overstate" in completion[
        "authorization_and_push_rule"
    ]
    assert "already committed irreversible conflict poison" in completion[
        "conflict_completion_rule"
    ]
    assert "cumulative time" in completion["cumulative_time_rule"]
    assert "round-to-nearest-ties-to-even" in completion[
        "time_encoding_rule"
    ]
    assert interruption["no_observation_sentinel_sha256"] == (
        PUBLICATION_RECOVERY_NO_REMOTE_OBSERVATION_SHA256
    )
    assert interruption["no_pre_push_authorization_sentinel_sha256"] == (
        PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
    )
    assert "outcome interrupted_before_completion" in interruption["rule"]
    assert "elapsed_seconds exactly 0x1.2c00000000000p+8" in interruption[
        "rule"
    ]
    assert "if and only if" in interruption["rule"]
    assert "explicitly a conservative upper bound" in interruption["rule"]
    assert "never reported as a known actual command count" in interruption[
        "rule"
    ]
    assert "completion prepare anchor" in interruption["rule"]
    assert conflict["fields"] == list(PUBLICATION_CONFLICT_FIELDS)
    assert conflict["self_hashed_and_externally_anchored"] is True
    assert conflict["accepted_observation_operations"] == {
        "normal_publication": {
            "observation_operation_sha256": PUBLICATION_NORMAL_OPERATION_SHA256,
            "required_recovery_start": False,
        },
        "publication_recovery": {
            "observation_operation_sha256": (
                "the exact committed recovery_invocation_start_sha256"
            ),
            "required_recovery_start": True,
        },
    }
    assert "accept exactly one operation pair" in conflict[
        "operation_binding_rule"
    ]
    assert "exact committed worker ownership record" in conflict[
        "operation_binding_rule"
    ]
    assert "normal publication never waits for a later recovery" in conflict[
        "proof_timing_rule"
    ]
    assert "exact committed conflicting publication-remote-observation" in conflict[
        "observation_precondition"
    ]
    assert "exact canonical self-hashed conflict row bytes" in conflict[
        "exact_reconciliation"
    ]
    assert "foreign or conflicting remote tag is later deleted" in conflict[
        "irreversibility_rule"
    ]
    assert "exact committed poison is idempotent" in conflict[
        "exact_reconciliation"
    ]


def test_worker_containment_and_terminalization_claim_are_exact() -> None:
    integrity = build_contract_manifest()["execution_integrity"]
    containment = integrity["publication_worker_containment"]
    ownership = containment["ownership_record"]
    quiescence = containment["quiescence_record"]
    claim = integrity["terminalization_claim"]

    assert PUBLICATION_WORKER_OWNERSHIP_FIELDS == (
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
    assert PUBLICATION_WORKER_QUIESCENCE_FIELDS == (
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
    assert TERMINALIZATION_CLAIM_FIELDS == (
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
    assert containment["scope"] == [
        "normal_publication",
        "publication_recovery",
    ]
    assert containment["normal_operation_sha256"] == (
        PUBLICATION_NORMAL_OPERATION_SHA256
    )
    assert "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE" in containment["os_mechanism"]
    assert "created suspended" in containment["os_mechanism"]
    assert ownership["fields"] == list(PUBLICATION_WORKER_OWNERSHIP_FIELDS)
    assert ownership["record_identity_fields"][-1] == "owner_nonce_sha256"
    assert ownership["fixed_values"]["kill_on_parent_exit"] is True
    assert "before creating any Git, SSH" in ownership["before_child_rule"]
    assert quiescence["fields"] == list(PUBLICATION_WORKER_QUIESCENCE_FIELDS)
    assert quiescence["record_identity_fields"][-1] == "verification_mode"
    assert quiescence["fixed_values"]["job_object_active_process_count"] == 0
    assert "prior owner PID plus creation FILETIME identity to be dead" in (
        quiescence["post_restart_rule"]
    )
    assert "old Git or SSH tree can never overlap" in containment[
        "capability_gate"
    ]
    assert "unexpected parent exit" in containment[
        "deadline_and_parent_exit_rule"
    ]
    assert claim["fields"] == list(TERMINALIZATION_CLAIM_FIELDS)
    assert "unique attempt identity" in claim["atomic_single_use_rule"]
    assert "cannot" in claim["concurrent_or_repeated_entry_rule"]
    assert "a second terminal_intent" in claim[
        "concurrent_or_repeated_entry_rule"
    ]


def test_new_source_inventory_is_exact_and_dependency_closed() -> None:
    gemma = build_contract_manifest()["gemma"]

    assert gemma["new_v2_2_sources"] == dict(sorted(NEW_SOURCE_FILES.items()))
    assert set(NEW_SOURCE_FILES) == {
        "acquisition",
        "attempt",
        "baseline",
        "features",
        "learner",
        "ledger",
        "market_verifier",
        "metrics",
        "no_leverage",
        "policy",
        "production",
        "publisher",
        "registry",
        "replay",
        "runner",
        "runtime",
        "source_verifier",
        "store",
        "vault",
    }
    assert "dependency-closed local imports" in gemma["source_inventory_rule"]
    assert gemma["allowed_external_python_distributions"] == [
        "requests",
        "urllib3",
        "certifi",
        "charset-normalizer",
        "idna",
    ]


def test_event_unavailability_never_becomes_an_implicit_imputation() -> None:
    availability = build_contract_manifest()["event_availability"]

    assert "not_comparable" in availability["first_same_form_filing"]
    assert "quality_risk to 1" in availability[
        "authenticated_schema_invalid_output"
    ]
    assert "exclude it from learner membership" in availability[
        "missing_or_unauthenticated_model_output"
    ]
    assert "no imputation" in availability["market_unavailable"]
    assert "audit-only" in availability["unavailable_label_rule"]
    assert "no fitted prediction" in availability["learner_unready"]
    assert "effective schedule flag false" in availability["active_overlay_event"]
    assert availability["undefined_or_nonfinite_metric"] == "terminal stage failure"


def test_filing_meaning_is_isolated_from_extraction_quality() -> None:
    manifest = build_contract_manifest()
    features = manifest["features"]
    gates = manifest["gates"]

    assert features["meaning_names"] == [
        "commercial_deterioration",
        "financial_deterioration",
        "risk_outlook_deterioration",
        "adverse_flag_fraction",
    ]
    assert features["quality_name"] == "semantic_quality_risk"
    assert "set only the four filing-meaning features to zero" in features[
        "no_filing_meaning_ablation"
    ]
    assert "preserve semantic_quality_risk exactly" in features[
        "no_filing_meaning_ablation"
    ]
    assert "cannot satisfy or rescue" in features["no_gemma_channel_diagnostic"]
    assert "schema-invalid outputs remain in the denominator" in features[
        "schema_valid_extraction_rate"
    ]
    assert "earlier cumulative rows cannot satisfy" in features[
        "extraction_coverage_stage_assignment"
    ]
    assert gates["development"]["schema_valid_extraction_rate_at_least"] == 0.90
    assert gates["development"]["nonzero_filing_meaning_rows_at_least"] == 24
    assert gates["confirmation"]["schema_valid_extraction_rate_at_least"] == 0.90
    assert gates["confirmation"]["nonzero_filing_meaning_rows_at_least"] == 6
    assert gates["final"]["schema_valid_extraction_rate_at_least"] == 0.90
    assert gates["final"]["nonzero_filing_meaning_rows_at_least"] == 3
    assert gates["final"][
        "semantic_vs_no_filing_meaning_action_differences_at_least"
    ] == 2


def test_semantic_arms_and_frozen_controls_have_exact_continuity() -> None:
    chronology = build_contract_manifest()["chronology"]
    controls = chronology["controls"]

    assert "start at portfolio genesis" in controls["semantic_arm_continuity"]
    assert "without reset" in controls["semantic_arm_continuity"]
    assert "same matured counterfactual labels" in controls["semantic_arm_labels"]
    assert "preceding session" in controls["fork_boundary_order"]
    assert "before any label maturing on the boundary session" in controls[
        "fork_boundary_order"
    ]
    assert "before any 2019-session admission" in controls["confirmation"]
    assert "before any 2024-session admission" in controls["final"]
    assert "acceptance timestamp ascending" in chronology["same_session_event_order"]
    assert "scheduled overlay blocks" in chronology["same_close_pending_overlay_rule"]


def test_metric_boundaries_drawdown_and_counting_are_literal() -> None:
    manifest = build_contract_manifest()
    chronology = manifest["chronology"]
    metrics = manifest["metric_definitions"]

    assert "performance and both terminal valuations end on 2026-07-09" in chronology[
        "final_cutoff_rule"
    ]
    assert "Only decisions through 2026-07-08" in chronology["final_cutoff_rule"]
    assert "append exactly one" in metrics["terminal_adjusted_close_report_variant"]
    assert "MDD = min(wealth/running_peak - 1)" in metrics["maximum_drawdown"]
    assert "strategy_MDD" in metrics["drawdown_comparison"]
    assert "count each block once" in metrics["difference_block"]
    assert "count each year once" in metrics["semantic_difference_year"]
    assert "zero and negative contributions are non-wins" in metrics[
        "episode_win_rate"
    ]
    assert "Semantic arms never fork after genesis" in metrics[
        "comparison_fork_rule"
    ]


def test_runtime_limit_applies_to_each_attempt_not_only_the_lifecycle() -> None:
    runtime = build_contract_manifest()["runtime"]
    partition = runtime["normal_governance_partition"]
    recovery = runtime["publication_recovery"]

    assert "each scored stage attempt" in runtime["scope"]
    assert "multi-stage research lifecycle" in runtime["cumulative_lifecycle_rule"]
    assert runtime["market_seconds_at_most_within_acquisition"] == 210
    assert runtime["market_requests_each_stage"] == 6
    assert runtime["sec_plus_market_combined_seconds_at_most"] == 720
    assert runtime["contingency_seconds"] == 239
    assert runtime["contingency_seconds"] == GOVERNANCE_CONTINGENCY_SECONDS
    assert runtime["maximum_phase_caps_plus_contingency_seconds"] == 3599
    assert PUBLICATION_INTENT_PREPARATION_SECONDS == 89
    assert SUPERVISED_PUBLICATION_SECONDS == 90
    assert PUBLICATION_FINALIZATION_PENDING_SECONDS == 60
    assert MAX_PUBLICATION_RECOVERY_SECONDS == 300
    assert partition == {
        "local_tag_and_intent_preparation_seconds": 89,
        "supervised_publication_seconds": 90,
        "terminal_finalization_or_pending_seconds": 60,
        "partition_sum_seconds": 239,
        "borrowing_between_partitions_forbidden": True,
        "remote_push_before_durable_intent_forbidden": True,
    }
    assert sum(
        (
            partition["local_tag_and_intent_preparation_seconds"],
            partition["supervised_publication_seconds"],
            partition["terminal_finalization_or_pending_seconds"],
        )
    ) == runtime["contingency_seconds"]
    assert recovery == {
        "each_non_effectful_invocation_seconds_at_most": 300,
        "separately_reported_from_effectful_attempt": True,
        "arbitrary_downtime_between_invocations_permitted": True,
        "research_effects_permitted": False,
        "model_calls_permitted": 0,
        "market_or_sec_requests_permitted": 0,
    }
    assert "explicitly returns the exact durable" in runtime[
        "scope"
    ]
    assert "separately reported publication recovery time" in runtime[
        "cumulative_lifecycle_rule"
    ]
    assert "all belong to the same original sub-hour normal invocation" in runtime[
        "normal_to_recovery_transition"
    ]
    assert "only after that invocation returns the explicit pending receipt" in runtime[
        "normal_to_recovery_transition"
    ]
    assert (
        runtime["sec_plus_market_combined_seconds_at_most"]
        + runtime["gemma_seconds_at_most"]
        + runtime["deterministic_seconds_at_most"]
        + runtime["contingency_seconds"]
        == 3599
    )
    assert runtime["per_attempt_phase_budgets"]["development_acquisition"][
        "gemma_seconds_at_most"
    ] == 0
    assert runtime["per_attempt_phase_budgets"]["development_scored"][
        "acquisition_seconds_at_most"
    ] == 0
    for stage_budget in runtime["per_attempt_phase_budgets"].values():
        assert stage_budget["total_seconds_strictly_below"] == 3600
        assert (
            stage_budget["acquisition_seconds_at_most"]
            + stage_budget["gemma_seconds_at_most"]
            + stage_budget["deterministic_seconds_at_most"]
            + runtime["contingency_seconds"]
            < stage_budget["total_seconds_strictly_below"]
        )
    assert runtime["model_call_caps"] == {
        "development_2000_2018": 80,
        "confirmation_2019_2023": 20,
        "final_2024_2026_ytd": 12,
    }


def test_ledger_and_final_gates_freeze_both_terminal_valuations() -> None:
    manifest = build_contract_manifest()
    ledger = manifest["ledger"]
    final = manifest["gates"]["final"]

    assert ledger["buy_cost_factor"] == "1/(1+cost_bps/10000)"
    assert ledger["sell_cost_factor"] == "1-cost_bps/10000"
    assert "shares = pre_fill_cash" in ledger["buy_fill_arithmetic"]
    assert "shares = 0" in ledger["sell_fill_arithmetic"]
    assert ledger["adjusted_open_valuation"] == "cash + shares * adjusted_open"
    assert "no synthetic liquidation" in ledger[
        "terminal_adjusted_close_valuation"
    ]
    assert final["terminal_valuation_methods_required"] == [
        "adjusted_open",
        "terminal_adjusted_close",
    ]
    assert final[
        "all_return_edge_and_drawdown_gates_pass_under_both_terminal_valuations"
    ] is True
    assert final["undefined_metric_fails"] is True


def test_every_ambiguous_gate_has_an_explicit_cost_scope() -> None:
    gates = build_contract_manifest()["gates"]
    development = gates["development"]
    confirmation = gates["confirmation"]
    final = gates["final"]

    for key in (
        "combined_edge_without_best_block_10bps_at_least",
        "positive_combined_blocks_10bps_at_least",
        "annual_win_rate_10bps_at_least",
        "negative_aapl_year_win_rate_10bps_at_least",
        "overlay_episode_win_rate_10bps_at_least",
        "overlay_median_edge_10bps_strictly_positive",
        "largest_positive_episode_share_10bps_at_most",
        "incremental_vs_baseline_without_best_block_10bps_strictly_positive",
        "semantic_edge_without_best_xor_10bps_strictly_positive",
    ):
        assert key in development
    assert confirmation[
        "combined_positive_years_at_least_3_at_both_5_and_10bps"
    ] is True
    assert "incremental_without_best_episode_10bps_strictly_positive" in confirmation
    assert "incremental_vs_baseline_positive_periods_10bps_at_least" in final
    assert "overlay_episode_win_rate_10bps_at_least" in final
    assert "largest_positive_episode_share_10bps_at_most" in final
    assert (
        final["max_drawdown_not_worse_than_aapl_by_more_than_at_5_and_10bps"]
        == 0.01
    )
    assert gates["undefined_or_nonfinite_metric_fails_every_dependent_gate"] is True


def test_document_matches_key_machine_contract_facts() -> None:
    document = (
        REPO_ROOT / "docs/aapl_sec_gemma_online_risk_overlay_v2_2.md"
    ).read_text(encoding="utf-8")
    normalized_document = " ".join(document.split())

    for expected in (
        "same trading thesis",
        "v2.1 rejected no-effect preflight branch at commit `9c2fbb0`",
        "long one unit of AAPL or in cash only",
        "keeps learning during later unseen periods",
        "development on 2000-2018",
        "untouched confirmation on 2019-2023",
        "2024, 2025, and 2026 year to date",
        "removes only that field",
        "raw response hash is recorded as diagnostic evidence",
        "durable opaque quarantine vault",
        "complete frozen history, not merely the latest 253 rows",
        "arbitrary all-true mapping cannot pass",
        "externally pinned",
        "non-force annotated Git tag",
        "3,599 seconds",
        "publication-pending",
        "arbitrary downtime",
        "one stable per-attempt terminal tag ref",
        "publication_intent_prepare",
        "binary64 hexadecimal value",
        "terminal_reconstruction_prepare",
        "commits and externally anchors a separate self-hashed start record",
        "one-push authorization marker",
        "`REF_ABSENT`, `VALUE_MISSING`, and `VALUE_MALFORMED`",
        "never use a mutable Git remote name",
            "one allowed literal GitHub HTTPS `remote_url` directly in their argv",
        "exactly one of seven observation sequences",
        "No normal or recovery push is legal without this marker",
        "The first valid terminal observation is authoritative",
        "A recovery receipt additionally binds",
        "interrupted_before_completion",
        "Windows Job Object",
        "prior owner PID and creation identity dead",
        "publication_receipt_committed",
        "new store-session nonce",
        "terminalization capability is single-use",
        "`claim_without_terminal_intent` recovery window",
        "read-only validation and rehydration",
        "`terminal_intent` but no database transition",
        "89 seconds",
        "90 seconds",
        "60 seconds",
        "300 seconds",
    ):
        assert expected in normalized_document


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("objective", "leverage"), True),
        (("gemma", "temperature"), 0.1),
        (("learner", "action_gate", "probability_at_least"), 0.5),
        (("policy", "overlay_horizon_sessions"), 21),
        (("runtime", "total_seconds_strictly_below"), 3601),
        (
            (
                "runtime",
                "normal_governance_partition",
                "supervised_publication_seconds",
            ),
            91,
        ),
        (
            (
                "execution_integrity",
                "publication_recovery",
                "zero_model_market_or_data_effects",
            ),
            False,
        ),
        (
            (
                "execution_integrity",
                "publication_recovery",
                "recovery_invocation_start_record",
                "before_remote_action_rule",
            ),
            "remote readback first",
        ),
        (
            (
                "execution_integrity",
                "publication_worker_containment",
                "ownership_record",
                "fixed_values",
                "kill_on_parent_exit",
            ),
            False,
        ),
        (
            (
                "execution_integrity",
                "terminalization_claim",
                "atomic_single_use_rule",
            ),
            "reusable",
        ),
        (
            (
                "execution_integrity",
                "external_report_pin",
                "ordinary_terminal_ref_template",
            ),
            "refs/tags/forked/{artifact_sha256}",
        ),
        (
            (
                "execution_integrity",
                "terminal_completion_recovery",
                "publication_intent_precludes_indeterminate",
            ),
            "convert to indeterminate",
        ),
        (("gates", "zero_action_difference_is_rejection"), False),
        (("chronology", "every_2025_decision_state"), "future-aware"),
    ],
)
def test_any_material_mutation_is_rejected(
    path: tuple[str, ...], replacement: object
) -> None:
    value = copy.deepcopy(build_contract_manifest())
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(
        SecGemmaOnlineRiskOverlayContractError,
        match="differs from preregistration",
    ):
        validate_contract_manifest(value)


def test_non_mapping_and_noncanonical_json_are_rejected() -> None:
    with pytest.raises(SecGemmaOnlineRiskOverlayContractError, match="mapping"):
        validate_contract_manifest([])

    value = build_contract_manifest()
    value["not_json"] = float("nan")
    with pytest.raises(
        SecGemmaOnlineRiskOverlayContractError,
        match="not canonical JSON",
    ):
        validate_contract_manifest(value)
