from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
    CONSUMED,
    TERMINAL_PASS,
    build_attempt_plan,
    build_implementation_manifest,
    store_record_receipt_material,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    BRANCH_NAME,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_ACQUISITION_ID,
    EXTERNAL_TAG_REF_TEMPLATE,
    PUBLICATION_NORMAL_NO_RECOVERY_COMPLETION_SHA256,
    PUBLICATION_NORMAL_OPERATION_SHA256,
    PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256,
    PUBLICATION_PUSH_COMMAND_PROFILE_SHA256,
    PUBLICATION_RECOVERY_COMPLETION_GENESIS_SHA256,
    PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256,
    PUBLICATION_RECOVERY_NO_REMOTE_OBSERVATION_SHA256,
    PUBLICATION_REMOTE_READBACK_COMMAND_PROFILE_SHA256,
    PUBLICATION_REMOTE_REF_ABSENT_SENTINEL,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
    EXPECTED_ORIGIN_URL,
    PREREGISTRATION_COMMIT,
    REQUIRED_NEW_SOURCE_PATHS,
    verify_live_source_tree,
)
from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
    _issue_verified_external_publication,
    prepare_external_publication,
)
from agent_benchmark.sec_gemma_online_risk_overlay_store import (
    ANCHOR_FILENAME,
    ANCHOR_RELATIVE_DIRECTORY,
    DATABASE_FILENAME,
    GOVERNANCE_RECORD_TABLES,
    STATE_RELATIVE_DIRECTORY,
    PublicationRecoveryCapability,
    SecGemmaOnlineRiskOverlayStore,
    SecGemmaOnlineRiskOverlayStoreConflict,
    SecGemmaOnlineRiskOverlayStoreError,
    TerminalizationCapability,
    VerifiedDurablePublicationReceipt,
    VerifiedPublicationIntent,
    VerifiedTerminalReconstructionMaterial,
    VerifiedTerminalizationClaim,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _implementation_repo(destination: Path) -> Path:
    destination.mkdir()
    shutil.copytree(
        WORKSPACE_ROOT / ".git",
        destination / ".git",
        ignore=shutil.ignore_patterns("turn-diffs"),
    )
    _git(destination, "config", "core.longpaths", "true")
    _git(destination, "config", "core.autocrlf", "false")
    _git(destination, "reset", "--quiet", "--hard", PREREGISTRATION_COMMIT)
    _git(destination, "checkout", "--quiet", "--detach", PREREGISTRATION_COMMIT)
    _git(destination, "switch", "--quiet", "-C", BRANCH_NAME)
    for relative in REQUIRED_NEW_SOURCE_PATHS.values():
        source = WORKSPACE_ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copy2(source, target)
        else:
            target.write_text('"""Replay placeholder."""\n', encoding="utf-8")
    _git(destination, "add", "--", *REQUIRED_NEW_SOURCE_PATHS.values())
    _git(destination, "config", "user.name", "Store Test")
    _git(destination, "config", "user.email", "store@example.invalid")
    _git(destination, "commit", "--quiet", "-m", "test implementation")
    _git(destination, "remote", "set-url", "origin", EXPECTED_ORIGIN_URL)
    head = _git(destination, "rev-parse", "HEAD")
    _git(
        destination,
        "update-ref",
        f"refs/remotes/origin/{BRANCH_NAME}",
        head,
    )
    return destination.resolve()


@pytest.fixture(scope="module")
def repo_and_manifest(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any]]:
    repo = _implementation_repo(tmp_path_factory.mktemp("store-v22") / "repo")
    manifest = build_implementation_manifest(
        verified_sources=verify_live_source_tree(repo)
    )
    return repo, manifest


@pytest.fixture(autouse=True)
def clean_contract_state(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    repo, _ = repo_and_manifest
    shutil.rmtree(repo / STATE_RELATIVE_DIRECTORY, ignore_errors=True)
    shutil.rmtree(repo / ANCHOR_RELATIVE_DIRECTORY, ignore_errors=True)
    yield
    shutil.rmtree(repo / STATE_RELATIVE_DIRECTORY, ignore_errors=True)
    shutil.rmtree(repo / ANCHOR_RELATIVE_DIRECTORY, ignore_errors=True)


def _store(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> SecGemmaOnlineRiskOverlayStore:
    return SecGemmaOnlineRiskOverlayStore(
        repo_and_manifest[0],
        implementation_manifest=repo_and_manifest[1],
    )


def _consumed(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> tuple[
    SecGemmaOnlineRiskOverlayStore,
    dict[str, Any],
    Any,
]:
    manifest = repo_and_manifest[1]
    plan = build_attempt_plan(
        implementation_manifest=manifest,
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
    )
    store = _store(repo_and_manifest)
    store.register_attempt(plan)
    return store, plan, store.consume_attempt(plan["attempt_id"])


def _self_hashed(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: canonical_sha256(body)}


def _sealed_reconstruction_and_intent(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> tuple[
    SecGemmaOnlineRiskOverlayStore,
    dict[str, Any],
    VerifiedTerminalReconstructionMaterial,
    VerifiedPublicationIntent,
]:
    store, plan, capability = _consumed(repo_and_manifest)
    manifest = repo_and_manifest[1]
    store.append_evidence(
        capability=capability,
        effect="official_sec_network",
        identity=f"evidence:{plan['attempt_id']}",
        payload={"sealed": True},
    )
    artifact = {"sealed_terminal_artifact": True}
    artifact_receipt = store.append_artifact(
        capability=capability,
        effect="deterministic_private_quarantine",
        identity=f"terminal_artifact:{plan['attempt_id']}",
        payload=artifact,
    )
    artifact_receipt_sha256 = canonical_sha256(
        store_record_receipt_material(artifact_receipt)
    )
    evidence_material = store.terminal_evidence_material(capability)
    reconstruction_body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "acquisition-terminal-reconstruction-v1"
        ),
        "reconstruction_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-test-reconstructor-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": manifest[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": manifest["implementation_commit"],
        "store_instance_id": store.store_instance_id,
        "attempt_id": plan["attempt_id"],
        "attempt_kind": plan["attempt_kind"],
        "attempt_plan_sha256": plan["attempt_plan_sha256"],
        "stage": "development",
        "terminal_status": TERMINAL_PASS,
        "report_kind": "acquisition_pass",
        "terminal_artifact_sha256": canonical_sha256(artifact),
        "terminal_artifact_store_receipt_sha256": (
            artifact_receipt_sha256
        ),
        "acquisition_validation_sha256": "1" * 64,
        "bundle_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "private_index_sha256": "4" * 64,
        "check_set_sha256": "5" * 64,
        "record_counts": evidence_material["record_counts"],
        "record_commitment_sha256": evidence_material[
            "record_commitment_sha256"
        ],
        "acquisition_artifact_receipt_sha256": (
            artifact_receipt_sha256
        ),
        "sealed_acquisition_phase_evidence_sha256": "6" * 64,
        "sealed_vault_commitments_sha256": "7" * 64,
    }
    reconstruction = store.commit_terminal_reconstruction_material(
        capability,
        _self_hashed(
            reconstruction_body,
            "terminal_reconstruction_material_sha256",
        ),
    )
    snapshot = store.snapshot()
    prepared = prepare_external_publication(
        implementation_manifest=manifest,
        attempt_id=plan["attempt_id"],
        terminal_status=TERMINAL_PASS,
        report_kind="acquisition_pass",
        artifact_sha256=canonical_sha256(artifact),
        predecessor_publication_sha256="8" * 64,
    )
    intent_body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-publication-intent-v1"
        ),
        "intent_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-intent-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": manifest[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": manifest["implementation_commit"],
        "attempt_id": plan["attempt_id"],
        "attempt_kind": plan["attempt_kind"],
        "attempt_plan_sha256": plan["attempt_plan_sha256"],
        "terminal_status": TERMINAL_PASS,
        "report_kind": "acquisition_pass",
        "artifact_sha256": canonical_sha256(artifact),
        "artifact_store_receipt_sha256": artifact_receipt_sha256,
        "terminal_reconstruction_material_sha256": reconstruction[
            "terminal_reconstruction_material_sha256"
        ],
        "terminal_reconstruction_material_store_receipt_sha256": (
            canonical_sha256(
                store_record_receipt_material(reconstruction.store_receipt)
            )
        ),
        "record_counts": evidence_material["record_counts"],
        "record_commitment_sha256": evidence_material[
            "record_commitment_sha256"
        ],
        "store_instance_id": store.store_instance_id,
        "store_journal_sequence": snapshot["journal_entry_count"],
        "store_journal_tip_sha256": snapshot["journal_tip_sha256"],
        "normal_attempt_elapsed_at_intent_prepare_hex": (1.0).hex(),
        "predecessor_publication_sha256": "8" * 64,
        **prepared.intent_material,
        "intent_status": "publication_pending",
        "research_effect_authority_invalidated": True,
        "semantic_result_release_blocked": True,
        "next_stage_authority_blocked": True,
        "external_cost_usd": 0,
    }
    intent = store.commit_publication_intent(
        capability,
        _self_hashed(intent_body, "publication_intent_sha256"),
    )
    return store, plan, reconstruction, intent


def _transport_manifest(
    store: SecGemmaOnlineRiskOverlayStore,
    plan: dict[str, Any],
    intent: VerifiedPublicationIntent,
    capability: PublicationRecoveryCapability,
) -> dict[str, Any]:
    snapshot = store.snapshot()
    digest = lambda name: canonical_sha256({"transport": name})
    body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-transport-isolation-v1"
        ),
        "profile_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-transport-isolation-profile-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": (
            store._implementation_manifest["implementation_manifest_sha256"]
        ),
        "implementation_commit": store._implementation_manifest[
            "implementation_commit"
        ],
        "store_instance_id": store.store_instance_id,
        "store_session_nonce_sha256": store.store_session_nonce_sha256,
        "attempt_id": plan["attempt_id"],
        "publication_intent_sha256": intent["publication_intent_sha256"],
        "operation_kind": capability.operation_kind,
        "operation_sha256": capability.operation_sha256,
        "isolated_git_directory_identity_sha256": digest("git-dir"),
        "local_config_entries_sha256": digest("config"),
        "empty_hooks_directory_path": "C:/isolated/empty-hooks",
        "empty_hooks_directory_identity_sha256": digest("hooks-id"),
        "empty_hooks_directory_listing_sha256": digest("hooks-listing"),
        "alternates_file_bytes_sha256": digest("alternates"),
        "alternate_object_directory_identity_sha256": digest("objects"),
        "git_executable_path": "C:/Program Files/Git/cmd/git.exe",
        "git_executable_sha256": digest("git"),
        "git_version_stdout_sha256": digest("git-version"),
        "git_exec_path": "C:/Program Files/Git/mingw64/libexec/git-core",
        "git_exec_path_directory_manifest_sha256": digest("git-exec"),
        "git_remote_https_executable_path": (
            "C:/Program Files/Git/mingw64/libexec/git-core/"
            "git-remote-https.exe"
        ),
        "git_remote_https_executable_sha256": digest("remote-https"),
        "credential_helper_config_value": (
            "!\"C:/Program Files/Git/mingw64/bin/"
            "git-credential-manager.exe\""
        ),
        "credential_helper_executable_path": (
            "C:/Program Files/Git/mingw64/bin/"
            "git-credential-manager.exe"
        ),
        "credential_helper_executable_sha256": digest("gcm"),
        "credential_helper_version_stdout_sha256": digest("gcm-version"),
        "command_interpreter_executable_path": (
            "C:/Program Files/Git/usr/bin/sh.exe"
        ),
        "command_interpreter_executable_sha256": digest("sh"),
        "transport_executable_closure_manifest_sha256": digest("closure"),
        "child_environment_policy_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-exact-child-environment-v1"
        ),
        "child_environment_sha256": digest("environment"),
        "path_lookup_forbidden": True,
        "remote_url": intent["remote_url"],
        "remote_url_scheme": "https",
        "readback_command_profile_sha256": (
            PUBLICATION_REMOTE_READBACK_COMMAND_PROFILE_SHA256
        ),
        "push_command_profile_sha256": (
            PUBLICATION_PUSH_COMMAND_PROFILE_SHA256
        ),
        "pre_transport_store_journal_sequence": snapshot[
            "journal_entry_count"
        ],
        "pre_transport_store_journal_tip_sha256": snapshot[
            "journal_tip_sha256"
        ],
    }
    return _self_hashed(
        body, "isolated_transport_git_directory_manifest_sha256"
    )


def _ownership(
    store: SecGemmaOnlineRiskOverlayStore,
    plan: dict[str, Any],
    intent: VerifiedPublicationIntent,
    capability: PublicationRecoveryCapability,
) -> dict[str, Any]:
    body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-worker-ownership-v1"
        ),
        "owner_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-worker-owner-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": (
            store._implementation_manifest["implementation_manifest_sha256"]
        ),
        "implementation_commit": store._implementation_manifest[
            "implementation_commit"
        ],
        "store_instance_id": store.store_instance_id,
        "store_session_nonce_sha256": store.store_session_nonce_sha256,
        "attempt_id": plan["attempt_id"],
        "publication_intent_sha256": intent["publication_intent_sha256"],
        "operation_kind": capability.operation_kind,
        "operation_sha256": capability.operation_sha256,
        "owner_nonce_sha256": canonical_sha256({"owner": "nonce"}),
        "owner_process_id": 1234,
        "owner_process_creation_filetime_hex": "0x1",
        "job_object_name_sha256": canonical_sha256({"job": "name"}),
        "owner_mutex_name_sha256": canonical_sha256({"mutex": "name"}),
        "kill_on_parent_exit": True,
        "child_assignment_before_resume_required": True,
        "ownership_status": "claimed",
    }
    return _self_hashed(body, "worker_ownership_sha256")


def _quiescence(
    store: SecGemmaOnlineRiskOverlayStore,
    plan: dict[str, Any],
    intent: VerifiedPublicationIntent,
    ownership: dict[str, Any],
    *,
    verification_mode: str = "same_session_clean_release",
) -> dict[str, Any]:
    body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-worker-quiescence-v1"
        ),
        "quiescence_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-worker-quiescence-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": (
            store._implementation_manifest["implementation_manifest_sha256"]
        ),
        "implementation_commit": store._implementation_manifest[
            "implementation_commit"
        ],
        "store_instance_id": store.store_instance_id,
        "store_session_nonce_sha256": store.store_session_nonce_sha256,
        "attempt_id": plan["attempt_id"],
        "publication_intent_sha256": intent["publication_intent_sha256"],
        "worker_ownership_sha256": ownership["worker_ownership_sha256"],
        "verification_mode": verification_mode,
        "prior_owner_process_dead": True,
        "owner_mutex_unowned": True,
        "job_object_active_process_count": 0,
        "recorded_git_ssh_processes_alive_count": 0,
        "quiescence_status": "verified_no_live_owner_or_worker",
    }
    return _self_hashed(body, "worker_quiescence_sha256")


def _readback_and_observation(
    store: SecGemmaOnlineRiskOverlayStore,
    plan: dict[str, Any],
    intent: VerifiedPublicationIntent,
    capability: PublicationRecoveryCapability,
    ownership: dict[str, Any],
    transport_manifest: dict[str, Any],
    *,
    observed_state: str,
    exit_code: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if observed_state == "absent":
        ref_status = "absent"
        raw = PUBLICATION_REMOTE_REF_ABSENT_SENTINEL
        observed_object = raw
        observed_peeled = raw
        observed_message = raw
    elif observed_state == "exact_expected":
        ref_status = "present"
        raw = intent["expected_tag_object_sha1"]
        observed_object = intent["expected_tag_object_sha1"]
        observed_peeled = intent["expected_peeled_commit"]
        observed_message = intent["tag_message_sha256"]
    else:
        ref_status = "present"
        raw = "f" * 40
        observed_object = raw
        observed_peeled = intent["expected_peeled_commit"]
        observed_message = "e" * 64
    common = {
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": (
            store._implementation_manifest["implementation_manifest_sha256"]
        ),
        "implementation_commit": store._implementation_manifest[
            "implementation_commit"
        ],
        "store_instance_id": store.store_instance_id,
        "store_session_nonce_sha256": store.store_session_nonce_sha256,
        "attempt_id": plan["attempt_id"],
        "publication_intent_sha256": intent["publication_intent_sha256"],
        "observation_operation_kind": capability.operation_kind,
        "observation_operation_sha256": capability.operation_sha256,
        "worker_ownership_sha256": ownership["worker_ownership_sha256"],
        "observation_ordinal": 1,
        "observation_phase": "pre_push",
        "prior_push_command_sha256": (
            PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256
        ),
        "tag_ref": intent["tag_ref"],
        "remote_name": intent["remote_name"],
        "remote_url": intent["remote_url"],
    }
    evidence_body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-remote-readback-evidence-v1"
        ),
        "evidence_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-remote-readback-evidence-verifier-v1"
        ),
        **common,
        "transport_isolation_profile_sha256": canonical_sha256(
            {"profile": "frozen"}
        ),
        "isolated_transport_git_directory_manifest_sha256": (
            transport_manifest[
                "isolated_transport_git_directory_manifest_sha256"
            ]
        ),
        "command_profile_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-remote-readback-command-profile-v1"
        ),
        "command_sequence_sha256": canonical_sha256({"command": "readback"}),
        "process_exit_status": "exited",
        "process_exit_code": exit_code,
        "stdout_byte_count": 0,
        "stdout_sha256": canonical_sha256(""),
        "stderr_byte_count": 0,
        "stderr_sha256": canonical_sha256(""),
        "transport_status": "completed",
        "ref_lookup_status": ref_status,
        "raw_ref_object_value": raw,
        "raw_peeled_value": (
            PUBLICATION_REMOTE_REF_ABSENT_SENTINEL
            if observed_state == "absent"
            else observed_peeled
        ),
        "raw_tag_message_sha256": (
            PUBLICATION_REMOTE_REF_ABSENT_SENTINEL
            if observed_state == "absent"
            else observed_message
        ),
    }
    evidence = _self_hashed(
        evidence_body, "remote_readback_evidence_sha256"
    )
    snapshot = store.snapshot()
    observation_body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-remote-observation-v1"
        ),
        "observation_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-remote-observation-verifier-v1"
        ),
        **common,
        "expected_tag_object_sha1": intent["expected_tag_object_sha1"],
        "expected_peeled_commit": intent["expected_peeled_commit"],
        "observed_ref_state": observed_state,
        "observed_tag_object_sha1": observed_object,
        "observed_peeled_commit": observed_peeled,
        "observed_tag_message_sha256": observed_message,
        "remote_readback_evidence_sha256": evidence[
            "remote_readback_evidence_sha256"
        ],
        "pre_observation_store_journal_sequence": snapshot[
            "journal_entry_count"
        ],
        "pre_observation_store_journal_tip_sha256": snapshot[
            "journal_tip_sha256"
        ],
    }
    return evidence, _self_hashed(
        observation_body, "publication_remote_observation_sha256"
    )


def test_store_uses_v22_namespace_schema_and_exclusive_lock(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    repo = repo_and_manifest[0]
    store = _store(repo_and_manifest)
    try:
        assert "v2_2" in STATE_RELATIVE_DIRECTORY.as_posix()
        assert "v2_2" in ANCHOR_RELATIVE_DIRECTORY.as_posix()
        assert store.path == repo / STATE_RELATIVE_DIRECTORY / DATABASE_FILENAME
        assert store.anchor_path == (
            repo / ANCHOR_RELATIVE_DIRECTORY / ANCHOR_FILENAME
        )
        assert set(GOVERNANCE_RECORD_TABLES) <= set(
            store.snapshot()["table_counts"]
        )
        with pytest.raises(SecGemmaOnlineRiskOverlayStoreConflict):
            _store(repo_and_manifest)
    finally:
        store.close()


def test_intent_invalidates_research_and_survives_reopen(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    store, plan, reconstruction, intent = (
        _sealed_reconstruction_and_intent(repo_and_manifest)
    )
    with pytest.raises(SecGemmaOnlineRiskOverlayStoreError, match="stale"):
        store.authorize_effect(
            next(iter(store._active_capabilities.values()), object()),
            "official_sec_network",
        )
    assert isinstance(reconstruction, VerifiedTerminalReconstructionMaterial)
    assert isinstance(intent, VerifiedPublicationIntent)
    assert store.attempt_history(plan["attempt_id"])[-1]["status"] == CONSUMED
    store.close()

    with _store(repo_and_manifest) as reopened:
        assert reopened.attempt_plan(plan["attempt_id"]) == plan
        assert reopened.attempt_history(plan["attempt_id"])[-1][
            "status"
        ] == CONSUMED
        assert len(
            reopened.governance_records(
                "terminal_reconstruction_materials",
                attempt_id=plan["attempt_id"],
            )
        ) == 1
        assert len(
            reopened.governance_records(
                "publication_intents", attempt_id=plan["attempt_id"]
            )
        ) == 1
        assert reopened.terminal_reconstruction_authority(
            plan["attempt_id"]
        ).material == reconstruction.material
        assert reopened.publication_intent_authority(
            plan["attempt_id"]
        ).material == intent.material
        artifact, artifact_receipt = (
            reopened.terminal_artifact_payload_and_receipt(
                plan["attempt_id"]
            )
        )
        assert artifact == {"sealed_terminal_artifact": True}
        assert artifact_receipt.payload_sha256 == canonical_sha256(artifact)


def test_same_session_reconciliation_closes_exact_orphan_observation(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    store, plan, _, intent = _sealed_reconstruction_and_intent(
        repo_and_manifest
    )
    snapshot = store.snapshot()
    start_body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-recovery-start-v1"
        ),
        "start_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-recovery-start-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": (
            store._implementation_manifest["implementation_manifest_sha256"]
        ),
        "implementation_commit": store._implementation_manifest[
            "implementation_commit"
        ],
        "store_instance_id": store.store_instance_id,
        "store_session_nonce_sha256": store.store_session_nonce_sha256,
        "attempt_id": plan["attempt_id"],
        "publication_intent_sha256": intent["publication_intent_sha256"],
        "invocation_ordinal": 1,
        "prior_recovery_completion_sha256": (
            PUBLICATION_RECOVERY_COMPLETION_GENESIS_SHA256
        ),
        "prior_cumulative_recovery_seconds": "0x0.0p+0",
        "start_status": "started",
        "invocation_seconds_cap": 300,
        "durable_pre_push_authorization_required": True,
        "pre_start_store_journal_sequence": snapshot[
            "journal_entry_count"
        ],
        "pre_start_store_journal_tip_sha256": snapshot[
            "journal_tip_sha256"
        ],
    }
    start = _self_hashed(
        start_body, "recovery_invocation_start_sha256"
    )
    capability = store.issue_publication_recovery_capability(
        plan["attempt_id"],
        operation_kind="publication_recovery",
        operation_sha256=start["recovery_invocation_start_sha256"],
    )
    store.commit_recovery_start(capability, start)
    transport = _transport_manifest(store, plan, intent, capability)
    store.commit_transport_manifest(capability, transport)
    ownership = _ownership(store, plan, intent, capability)
    store.claim_publication_worker_ownership(capability, ownership)
    evidence, observation = _readback_and_observation(
        store,
        plan,
        intent,
        capability,
        ownership,
        transport,
        observed_state="exact_expected",
    )
    durable = store.commit_remote_observation(
        capability, evidence, observation
    )
    store.commit_worker_quiescence(
        capability,
        _quiescence(store, plan, intent, ownership),
    )
    store.release_publication_capability(capability)

    store.reconcile_publication_recovery_state(plan["attempt_id"])

    completions = store.governance_records(
        "publication_recovery_invocation_completions",
        attempt_id=plan["attempt_id"],
    )
    assert len(completions) == 1
    assert completions[0]["outcome"] == "remote_exact_without_push"
    assert completions[0]["remote_observation_sha256"] == durable[
        "publication_remote_observation_sha256"
    ]
    store.close()


def test_reopen_completes_exact_intent_row_after_committed_anchor_crash(
    repo_and_manifest: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, plan, capability = _consumed(repo_and_manifest)
    manifest = repo_and_manifest[1]
    artifact = {"sealed_terminal_artifact": True}
    artifact_receipt = store.append_artifact(
        capability=capability,
        effect="deterministic_private_quarantine",
        identity=f"terminal_artifact:{plan['attempt_id']}",
        payload=artifact,
    )
    material = store.terminal_evidence_material(capability)
    artifact_receipt_hash = canonical_sha256(
        store_record_receipt_material(artifact_receipt)
    )
    reconstruction_body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "acquisition-terminal-reconstruction-v1"
        ),
        "reconstruction_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-test-reconstructor-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": manifest[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": manifest["implementation_commit"],
        "store_instance_id": store.store_instance_id,
        "attempt_id": plan["attempt_id"],
        "attempt_kind": plan["attempt_kind"],
        "attempt_plan_sha256": plan["attempt_plan_sha256"],
        "stage": "development",
        "terminal_status": TERMINAL_PASS,
        "report_kind": "acquisition_pass",
        "terminal_artifact_sha256": canonical_sha256(artifact),
        "terminal_artifact_store_receipt_sha256": artifact_receipt_hash,
        "acquisition_validation_sha256": "1" * 64,
        "bundle_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "private_index_sha256": "4" * 64,
        "check_set_sha256": "5" * 64,
        "record_counts": material["record_counts"],
        "record_commitment_sha256": material["record_commitment_sha256"],
        "acquisition_artifact_receipt_sha256": artifact_receipt_hash,
        "sealed_acquisition_phase_evidence_sha256": "6" * 64,
        "sealed_vault_commitments_sha256": "7" * 64,
    }
    reconstruction = store.commit_terminal_reconstruction_material(
        capability,
        _self_hashed(
            reconstruction_body,
            "terminal_reconstruction_material_sha256",
        ),
    )
    snapshot = store.snapshot()
    intent_body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-publication-intent-v1"
        ),
        "intent_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-intent-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": manifest[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": manifest["implementation_commit"],
        "attempt_id": plan["attempt_id"],
        "attempt_kind": plan["attempt_kind"],
        "attempt_plan_sha256": plan["attempt_plan_sha256"],
        "terminal_status": TERMINAL_PASS,
        "report_kind": "acquisition_pass",
        "artifact_sha256": canonical_sha256(artifact),
        "artifact_store_receipt_sha256": artifact_receipt_hash,
        "terminal_reconstruction_material_sha256": reconstruction[
            "terminal_reconstruction_material_sha256"
        ],
        "terminal_reconstruction_material_store_receipt_sha256": (
            canonical_sha256(
                store_record_receipt_material(reconstruction.store_receipt)
            )
        ),
        "record_counts": material["record_counts"],
        "record_commitment_sha256": material["record_commitment_sha256"],
        "store_instance_id": store.store_instance_id,
        "store_journal_sequence": snapshot["journal_entry_count"],
        "store_journal_tip_sha256": snapshot["journal_tip_sha256"],
        "normal_attempt_elapsed_at_intent_prepare_hex": (1.0).hex(),
        "predecessor_publication_sha256": "8" * 64,
        "tag_ref": EXTERNAL_TAG_REF_TEMPLATE.format(
            attempt_id=plan["attempt_id"]
        ),
        "tag_target_commit": manifest["implementation_commit"],
        "tag_message_sha256": "9" * 64,
        "expected_tag_object_sha1": "a" * 40,
        "expected_peeled_commit": manifest["implementation_commit"],
        "expected_publication_sha256": "b" * 64,
        "remote_name": "origin",
        "remote_url": manifest["origin_url"],
        "intent_status": "publication_pending",
        "research_effect_authority_invalidated": True,
        "semantic_result_release_blocked": True,
        "next_stage_authority_blocked": True,
        "external_cost_usd": 0,
    }
    intent = _self_hashed(intent_body, "publication_intent_sha256")
    original = store._append_anchor

    def fail_committed(
        event: str,
        database_snapshot: dict[str, Any],
        *,
        attempt_id: str | None,
        event_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if event == "publication_intent_committed":
            raise SecGemmaOnlineRiskOverlayStoreError(
                "injected intent anchor crash"
            )
        return original(
            event,
            database_snapshot,
            attempt_id=attempt_id,
            event_payload=event_payload,
        )

    monkeypatch.setattr(store, "_append_anchor", fail_committed)
    with pytest.raises(
        SecGemmaOnlineRiskOverlayStoreError,
        match="injected intent anchor crash",
    ):
        store.commit_publication_intent(capability, intent)
    store._close_resources(recover=False)

    with _store(repo_and_manifest) as reopened:
        assert reopened.governance_records(
            "publication_intents", attempt_id=plan["attempt_id"]
        ) == [intent]
        assert reopened.attempt_history(plan["attempt_id"])[-1][
            "status"
        ] == CONSUMED
        events = [entry["event"] for entry in reopened._anchor_entries()]
        assert events.count("publication_intent_prepare") == 1
        assert events.count("publication_intent_committed") == 1


def test_failed_readback_cannot_masquerade_as_absent(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    store, plan, _, intent = _sealed_reconstruction_and_intent(
        repo_and_manifest
    )
    pre_owner = store.issue_publication_recovery_capability(
        plan["attempt_id"],
        operation_kind="normal_publication",
    )
    store.release_publication_capability(pre_owner)
    capability = store.issue_publication_recovery_capability(
        plan["attempt_id"],
        operation_kind="normal_publication",
    )
    transport = _transport_manifest(store, plan, intent, capability)
    store.commit_transport_manifest(capability, transport)
    ownership = _ownership(store, plan, intent, capability)
    store.claim_publication_worker_ownership(capability, ownership)
    evidence, observation = _readback_and_observation(
        store,
        plan,
        intent,
        capability,
        ownership,
        transport,
        observed_state="absent",
        exit_code=1,
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayStoreError,
        match="cannot create an observation",
    ):
        store.commit_remote_observation(
            capability, evidence, observation
        )
    assert store.governance_records(
        "publication_remote_observations",
        attempt_id=plan["attempt_id"],
    ) == []
    store._close_resources(recover=False)


@dataclass(frozen=True)
class _ExternalPublication:
    publication: dict[str, Any]


def test_exact_remote_receipt_issues_single_use_terminalization_claim(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    store, plan, _, intent = _sealed_reconstruction_and_intent(
        repo_and_manifest
    )
    capability = store.issue_publication_recovery_capability(
        plan["attempt_id"],
        operation_kind="normal_publication",
    )
    transport = _transport_manifest(store, plan, intent, capability)
    store.commit_transport_manifest(capability, transport)
    ownership = _ownership(store, plan, intent, capability)
    store.claim_publication_worker_ownership(capability, ownership)
    evidence, observation = _readback_and_observation(
        store,
        plan,
        intent,
        capability,
        ownership,
        transport,
        observed_state="exact_expected",
    )
    durable_observation = store.commit_remote_observation(
        capability, evidence, observation
    )
    store.commit_worker_quiescence(
        capability,
        _quiescence(store, plan, intent, ownership),
    )
    prepared = prepare_external_publication(
        implementation_manifest=store._implementation_manifest,
        attempt_id=plan["attempt_id"],
        terminal_status=intent["terminal_status"],
        report_kind=intent["report_kind"],
        artifact_sha256=intent["artifact_sha256"],
        predecessor_publication_sha256=(
            intent["predecessor_publication_sha256"]
        ),
    )
    publication = prepared.expected_publication
    opaque_publication = _issue_verified_external_publication(
        publication,
        implementation_manifest=store._implementation_manifest,
    )
    snapshot = store.snapshot()
    intent_receipt = intent.store_receipt
    receipt_body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-publication-receipt-v1"
        ),
        "receipt_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-receipt-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": (
            store._implementation_manifest["implementation_manifest_sha256"]
        ),
        "implementation_commit": store._implementation_manifest[
            "implementation_commit"
        ],
        "store_instance_id": store.store_instance_id,
        "attempt_id": plan["attempt_id"],
        "publication_intent_sha256": intent["publication_intent_sha256"],
        "publication_intent_store_receipt_sha256": canonical_sha256(
            store_record_receipt_material(intent_receipt)
        ),
        "publication_remote_observation_sha256": durable_observation[
            "publication_remote_observation_sha256"
        ],
        "recovery_invocation_completion_sha256": (
            PUBLICATION_NORMAL_NO_RECOVERY_COMPLETION_SHA256
        ),
        "pre_push_authorization_sha256": (
            PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
        ),
        "external_publication_sha256": publication["publication_sha256"],
        "remote_tag_object_sha1": intent["expected_tag_object_sha1"],
        "remote_peeled_commit": intent["expected_peeled_commit"],
        "pre_receipt_store_journal_sequence": snapshot[
            "journal_entry_count"
        ],
        "pre_receipt_store_journal_tip_sha256": snapshot[
            "journal_tip_sha256"
        ],
        "receipt_status": "publication_verified",
        "publication_capability_invalidated": True,
        "terminalization_capability_required": True,
    }
    receipt_material = _self_hashed(
        receipt_body, "publication_receipt_sha256"
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayStoreError,
        match="opaque external publication",
    ):
        store.commit_publication_receipt(
            capability,
            receipt_material,
            _ExternalPublication(publication),
        )
    durable_receipt = store.commit_publication_receipt(
        capability,
        receipt_material,
        opaque_publication,
    )
    assert isinstance(durable_receipt, VerifiedDurablePublicationReceipt)
    reissued_receipt = store.publication_receipt_authority(
        plan["attempt_id"],
        opaque_publication,
    )
    assert reissued_receipt.material == durable_receipt.material
    terminalization = store.issue_terminalization_capability(
        plan["attempt_id"]
    )
    assert isinstance(terminalization, TerminalizationCapability)
    terminal_body = {
        "terminal_status": TERMINAL_PASS,
        "attempt_id": plan["attempt_id"],
    }
    terminal_evidence = _self_hashed(
        terminal_body, "terminal_evidence_sha256"
    )
    claim = store.commit_terminalization_claim(
        terminalization,
        terminal_evidence,
        TERMINAL_PASS,
    )
    assert isinstance(claim, VerifiedTerminalizationClaim)
    with pytest.raises(SecGemmaOnlineRiskOverlayStoreConflict):
        store.issue_terminalization_capability(plan["attempt_id"])
    store.close()


def test_quiescent_normal_capability_can_be_released_for_recovery(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    store, plan, _, intent = _sealed_reconstruction_and_intent(
        repo_and_manifest
    )
    capability = store.issue_publication_recovery_capability(
        plan["attempt_id"],
        operation_kind="normal_publication",
    )
    transport = _transport_manifest(store, plan, intent, capability)
    store.commit_transport_manifest(capability, transport)
    ownership = _ownership(store, plan, intent, capability)
    store.claim_publication_worker_ownership(capability, ownership)
    with pytest.raises(
        SecGemmaOnlineRiskOverlayStoreError,
        match="requires exact committed worker quiescence",
    ):
        store.release_publication_capability(capability)
    store.commit_worker_quiescence(
        capability,
        _quiescence(store, plan, intent, ownership),
    )
    store.release_publication_capability(capability)
    recovery_operation = canonical_sha256({"recovery": 1})
    recovery = store.issue_publication_recovery_capability(
        plan["attempt_id"],
        operation_kind="publication_recovery",
        operation_sha256=recovery_operation,
    )
    assert recovery.operation_sha256 == recovery_operation
    store._close_resources(recover=False)


def test_normal_conflict_is_poisoned_before_quiescent_release(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    store, plan, _, intent = _sealed_reconstruction_and_intent(
        repo_and_manifest
    )
    capability = store.issue_publication_recovery_capability(
        plan["attempt_id"],
        operation_kind="normal_publication",
    )
    transport = _transport_manifest(store, plan, intent, capability)
    store.commit_transport_manifest(capability, transport)
    ownership = _ownership(store, plan, intent, capability)
    store.claim_publication_worker_ownership(capability, ownership)
    evidence, observation = _readback_and_observation(
        store,
        plan,
        intent,
        capability,
        ownership,
        transport,
        observed_state="conflicting",
    )
    durable = store.commit_remote_observation(
        capability, evidence, observation
    )
    conflict_body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-publication-conflict-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": (
            store._implementation_manifest["implementation_manifest_sha256"]
        ),
        "store_instance_id": store.store_instance_id,
        "attempt_id": plan["attempt_id"],
        "publication_intent_sha256": intent["publication_intent_sha256"],
        "observation_operation_kind": capability.operation_kind,
        "observation_operation_sha256": capability.operation_sha256,
        "tag_ref": intent["tag_ref"],
        "remote_observation_sha256": durable[
            "publication_remote_observation_sha256"
        ],
        "conflict_reason": "durable_remote_ref_conflict",
        "poisoned": True,
        "prior_governance_record_sha256": ownership[
            "worker_ownership_sha256"
        ],
    }
    store.commit_publication_conflict(
        capability,
        _self_hashed(conflict_body, "publication_conflict_sha256"),
    )
    store.commit_worker_quiescence(
        capability,
        _quiescence(store, plan, intent, ownership),
    )
    store.release_publication_capability(capability)
    with pytest.raises(
        SecGemmaOnlineRiskOverlayStoreError,
        match="Receipt or conflict forbids",
    ):
        store.issue_publication_recovery_capability(
            plan["attempt_id"],
            operation_kind="publication_recovery",
            operation_sha256=canonical_sha256({"recovery": "poisoned"}),
        )
    store.close()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("conflict_reason", "remote_conflict"),
        ("tag_ref", "refs/tags/foreign-terminal"),
    ),
)
def test_conflict_poison_requires_exact_reason_and_observation_tag(
    repo_and_manifest: tuple[Path, dict[str, Any]],
    field: str,
    value: str,
) -> None:
    store, plan, _, intent = _sealed_reconstruction_and_intent(
        repo_and_manifest
    )
    capability = store.issue_publication_recovery_capability(
        plan["attempt_id"],
        operation_kind="normal_publication",
    )
    transport = _transport_manifest(store, plan, intent, capability)
    store.commit_transport_manifest(capability, transport)
    ownership = _ownership(store, plan, intent, capability)
    store.claim_publication_worker_ownership(capability, ownership)
    evidence, observation = _readback_and_observation(
        store,
        plan,
        intent,
        capability,
        ownership,
        transport,
        observed_state="conflicting",
    )
    durable = store.commit_remote_observation(
        capability, evidence, observation
    )
    conflict_body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-publication-conflict-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": (
            store._implementation_manifest["implementation_manifest_sha256"]
        ),
        "store_instance_id": store.store_instance_id,
        "attempt_id": plan["attempt_id"],
        "publication_intent_sha256": intent["publication_intent_sha256"],
        "observation_operation_kind": capability.operation_kind,
        "observation_operation_sha256": capability.operation_sha256,
        "tag_ref": intent["tag_ref"],
        "remote_observation_sha256": durable[
            "publication_remote_observation_sha256"
        ],
        "conflict_reason": "durable_remote_ref_conflict",
        "poisoned": True,
        "prior_governance_record_sha256": ownership[
            "worker_ownership_sha256"
        ],
    }
    conflict_body[field] = value

    with pytest.raises(
        SecGemmaOnlineRiskOverlayStoreError,
        match="exact durable conflict proof",
    ):
        store.commit_publication_conflict(
            capability,
            _self_hashed(
                conflict_body, "publication_conflict_sha256"
            ),
        )

    assert store.governance_records(
        "publication_conflicts", attempt_id=plan["attempt_id"]
    ) == []
    store._close_resources(recover=False)


def test_reopen_can_prove_prior_owner_quiescent_before_new_capability(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    store, plan, _, intent = _sealed_reconstruction_and_intent(
        repo_and_manifest
    )
    capability = store.issue_publication_recovery_capability(
        plan["attempt_id"],
        operation_kind="normal_publication",
    )
    transport = _transport_manifest(store, plan, intent, capability)
    store.commit_transport_manifest(capability, transport)
    ownership_material = _ownership(store, plan, intent, capability)
    store.claim_publication_worker_ownership(
        capability, ownership_material
    )
    store._close_resources(recover=False)

    with _store(repo_and_manifest) as reopened:
        unresolved = (
            reopened.unresolved_publication_worker_ownership_authorities(
                plan["attempt_id"]
            )
        )
        assert len(unresolved) == 1
        with pytest.raises(
            SecGemmaOnlineRiskOverlayStoreError,
            match="not quiescent",
        ):
            reopened.issue_publication_recovery_capability(
                plan["attempt_id"],
                operation_kind="publication_recovery",
                operation_sha256=canonical_sha256({"recovery": "blocked"}),
            )
        rehydrated_intent = reopened.publication_intent_authority(
            plan["attempt_id"]
        )
        restarted_quiescence = _quiescence(
            reopened,
            plan,
            rehydrated_intent,
            unresolved[0].material,
            verification_mode="post_restart_prior_owner_dead",
        )
        reopened.commit_restarted_worker_quiescence(
            unresolved[0], restarted_quiescence
        )
        recovery_hash = canonical_sha256({"recovery": "unblocked"})
        recovery = reopened.issue_publication_recovery_capability(
            plan["attempt_id"],
            operation_kind="publication_recovery",
            operation_sha256=recovery_hash,
        )
        assert recovery.operation_sha256 == recovery_hash


def test_reopen_closes_stale_recovery_start_without_research(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    store, plan, _, intent = _sealed_reconstruction_and_intent(
        repo_and_manifest
    )
    snapshot = store.snapshot()
    start_body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-recovery-start-v1"
        ),
        "start_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-recovery-start-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": (
            store._implementation_manifest["implementation_manifest_sha256"]
        ),
        "implementation_commit": store._implementation_manifest[
            "implementation_commit"
        ],
        "store_instance_id": store.store_instance_id,
        "store_session_nonce_sha256": store.store_session_nonce_sha256,
        "attempt_id": plan["attempt_id"],
        "publication_intent_sha256": intent["publication_intent_sha256"],
        "invocation_ordinal": 1,
        "prior_recovery_completion_sha256": (
            PUBLICATION_RECOVERY_COMPLETION_GENESIS_SHA256
        ),
        "prior_cumulative_recovery_seconds": "0x0.0p+0",
        "start_status": "started",
        "invocation_seconds_cap": 300,
        "durable_pre_push_authorization_required": True,
        "pre_start_store_journal_sequence": snapshot[
            "journal_entry_count"
        ],
        "pre_start_store_journal_tip_sha256": snapshot[
            "journal_tip_sha256"
        ],
    }
    start = _self_hashed(
        start_body, "recovery_invocation_start_sha256"
    )
    capability = store.issue_publication_recovery_capability(
        plan["attempt_id"],
        operation_kind="publication_recovery",
        operation_sha256=start["recovery_invocation_start_sha256"],
    )
    store.commit_recovery_start(capability, start)
    transport = _transport_manifest(store, plan, intent, capability)
    store.commit_transport_manifest(capability, transport)
    ownership = _ownership(store, plan, intent, capability)
    store.claim_publication_worker_ownership(capability, ownership)
    store._close_resources(recover=False)

    with _store(repo_and_manifest) as reopened:
        assert reopened.governance_records(
            "publication_recovery_invocation_completions",
            attempt_id=plan["attempt_id"],
        ) == []
        unresolved = (
            reopened.unresolved_publication_worker_ownership_authorities(
                plan["attempt_id"]
            )
        )
        assert len(unresolved) == 1
        reopened_intent = reopened.publication_intent_authority(
            plan["attempt_id"]
        )
        reopened.commit_restarted_worker_quiescence(
            unresolved[0],
            _quiescence(
                reopened,
                plan,
                reopened_intent,
                unresolved[0].material,
                verification_mode="post_restart_prior_owner_dead",
            ),
        )
        completions = reopened.governance_records(
            "publication_recovery_invocation_completions",
            attempt_id=plan["attempt_id"],
        )
        assert len(completions) == 1
        assert completions[0]["outcome"] == "interrupted_before_completion"
        assert completions[0]["remote_observation_sha256"] == (
            PUBLICATION_RECOVERY_NO_REMOTE_OBSERVATION_SHA256
        )
        assert completions[0]["pre_push_authorization_sha256"] == (
            PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
        )
        assert completions[0]["push_command_count_upper_bound"] == 0
        assert completions[0]["elapsed_seconds"] == (
            "0x1.2c00000000000p+8"
        )
        reissued = reopened.recovery_completion_authority(
            plan["attempt_id"],
            completions[0]["recovery_invocation_completion_sha256"],
        )
        assert reissued.material == completions[0]
        assert reopened.attempt_history(plan["attempt_id"])[-1][
            "status"
        ] == CONSUMED
