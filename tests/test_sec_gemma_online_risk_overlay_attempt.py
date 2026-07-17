from __future__ import annotations

import copy
from pathlib import Path
import shutil
import sqlite3
import subprocess
from typing import Any

import pytest

import agent_benchmark.sec_gemma_online_risk_overlay_acquisition as acquisition
import agent_benchmark.sec_gemma_online_risk_overlay_publisher as publisher
import agent_benchmark.sec_gemma_online_risk_overlay_store as store_module
import agent_benchmark.sec_gemma_online_risk_overlay_vault as vault_module
from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
    _external_distributions,
    _numerical_time_distributions,
    ATTEMPT_IDS,
    CONSUMED,
    DETERMINISTIC_EVALUATION_SCHEMA_VERSION,
    JOINT_STAGE_REPORT_SCHEMA_VERSION,
    PLANNED,
    REPORT_RECORD_TABLES,
    TERMINAL_FAIL,
    TERMINAL_PASS,
    SecGemmaOnlineRiskOverlayAttemptError,
    build_attempt_plan,
    build_attempt_transition,
    build_implementation_manifest,
    issue_verified_acquisition_terminal_evidence,
    issue_verified_scored_terminal_evidence,
    validate_acquisition_terminal_evidence,
    validate_implementation_manifest,
    validate_scored_terminal_evidence,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    ACQUISITION_TERMINAL_EVIDENCE_FIELDS,
    ACQUISITION_VALIDATION_CHECKS,
    BRANCH_NAME,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_ACQUISITION_ID,
    DEVELOPMENT_ATTEMPT_ID,
    EXTERNAL_TAG_REF_TEMPLATE,
    PUBLICATION_NORMAL_NO_RECOVERY_COMPLETION_SHA256,
    PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256,
    SCORED_TERMINAL_EVIDENCE_FIELDS,
    build_contract_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
    EXPECTED_ORIGIN_URL,
    PREREGISTRATION_COMMIT,
    REQUIRED_NEW_SOURCE_PATHS,
    verify_live_source_tree,
)
from agent_benchmark.sec_gemma_online_risk_overlay_store import (
    StoreRecordReceipt,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def test_external_distribution_manifest_binds_complete_module_inventory() -> None:
    module_files = [
        {"path": "requests/__init__.py", "sha256": "a" * 64},
        {"path": "requests/adapters.py", "sha256": "b" * 64},
    ]
    value = [
        {
            "import_name": "requests",
            "distribution": "requests",
            "version": "2.32.5",
            "module_files": module_files,
            "module_files_sha256": canonical_sha256(module_files),
            "direct_url_sha256": None,
        }
    ]

    assert _external_distributions(value) == value

    changed = copy.deepcopy(value)
    changed[0]["module_files_sha256"] = "c" * 64
    with pytest.raises(
        SecGemmaOnlineRiskOverlayAttemptError,
        match="inventory hash",
    ):
        _external_distributions(changed)

    reordered = copy.deepcopy(value)
    reordered[0]["module_files"].reverse()
    reordered[0]["module_files_sha256"] = canonical_sha256(
        reordered[0]["module_files"]
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayAttemptError,
        match="canonically ordered",
    ):
        _external_distributions(reordered)


def test_numerical_time_manifest_binds_origins_native_and_data_files() -> None:
    value = []
    for import_name, origin, extra_path in (
        (
            "numpy",
            "numpy/__init__.py",
            "numpy/_core/_multiarray_umath.pyd",
        ),
        (
            "tzdata",
            "tzdata/__init__.py",
            "tzdata/zoneinfo/America/New_York",
        ),
    ):
        runtime_files = [
            {"path": origin, "sha256": "a" * 64},
            {"path": extra_path, "sha256": "b" * 64},
        ]
        runtime_files.sort(key=lambda item: item["path"])
        value.append(
            {
                "import_name": import_name,
                "distribution": import_name,
                "version": "1.0",
                "module_origin": {
                    "path": origin,
                    "sha256": "a" * 64,
                },
                "runtime_files": runtime_files,
                "runtime_files_sha256": canonical_sha256(runtime_files),
                "direct_url_sha256": None,
            }
        )
    value.sort(key=lambda item: item["import_name"])

    assert _numerical_time_distributions(value) == value

    changed = copy.deepcopy(value)
    changed[0]["module_origin"]["sha256"] = "c" * 64
    with pytest.raises(
        SecGemmaOnlineRiskOverlayAttemptError,
        match="outside its runtime file inventory",
    ):
        _numerical_time_distributions(changed)

    missing = copy.deepcopy(value[:-1])
    with pytest.raises(
        SecGemmaOnlineRiskOverlayAttemptError,
        match="roots differ",
    ):
        _numerical_time_distributions(missing)


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
    _git(destination, "config", "user.name", "Attempt Test")
    _git(destination, "config", "user.email", "attempt@example.invalid")
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
def manifest(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    repo = _implementation_repo(tmp_path_factory.mktemp("attempt-v21") / "repo")
    return build_implementation_manifest(
        verified_sources=verify_live_source_tree(repo)
    )


def _terminal_pass(
    manifest: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    planned = build_attempt_transition(
        attempt_plan=plan,
        implementation_manifest=manifest,
        status=PLANNED,
    )
    consumed = build_attempt_transition(
        attempt_plan=plan,
        implementation_manifest=manifest,
        status=CONSUMED,
        prior_transition=planned,
    )
    return build_attempt_transition(
        attempt_plan=plan,
        implementation_manifest=manifest,
        status=TERMINAL_PASS,
        prior_transition=consumed,
    )


def _material(
    plan: dict[str, Any],
    *,
    counts: dict[str, int] | None = None,
    commitment: str = "c" * 64,
) -> dict[str, Any]:
    return {
        "attempt_id": plan["attempt_id"],
        "attempt_plan_sha256": plan["attempt_plan_sha256"],
        "record_counts": (
            counts
            if counts is not None
            else {table: 1 for table in REPORT_RECORD_TABLES}
        ),
        "record_commitment_sha256": commitment,
    }


def _receipt(
    *,
    attempt_id: str,
    payload: dict[str, Any],
    identity: str,
) -> StoreRecordReceipt:
    return StoreRecordReceipt(
        table="artifacts",
        identity=identity,
        attempt_id=attempt_id,
        payload_sha256=canonical_sha256(payload),
        journal_sequence=9,
        journal_entry_sha256="d" * 64,
    )


def _publication(
    manifest: dict[str, Any],
    *,
    attempt_id: str,
    terminal_status: str,
    report_kind: str,
    artifact_sha256: str,
) -> publisher.VerifiedExternalPublication:
    message = publisher.build_external_tag_message(
        implementation_manifest=manifest,
        attempt_id=attempt_id,
        terminal_status=terminal_status,
        report_kind=report_kind,
        artifact_sha256=artifact_sha256,
        predecessor_publication_sha256=publisher.PUBLICATION_GENESIS_SHA256,
    )
    tag_ref = EXTERNAL_TAG_REF_TEMPLATE.format(
        attempt_id=attempt_id,
        report_kind=report_kind,
        artifact_sha256=artifact_sha256,
    )
    payload = publisher.build_external_publication(
        implementation_manifest=manifest,
        tag_message=message,
        tag_ref=tag_ref,
        remote_name="origin",
        remote_url=manifest["origin_url"],
        remote_tag_object_sha1="1" * 40,
        remote_peeled_commit=manifest["implementation_commit"],
    )
    return publisher._issue_verified_external_publication(
        payload,
        implementation_manifest=manifest,
    )


def _store_receipt_material(
    receipt: StoreRecordReceipt,
) -> dict[str, Any]:
    return {
        "table": receipt.table,
        "identity": receipt.identity,
        "attempt_id": receipt.attempt_id,
        "payload_sha256": receipt.payload_sha256,
        "journal_sequence": receipt.journal_sequence,
        "journal_entry_sha256": receipt.journal_entry_sha256,
    }


def _self_hashed(
    body: dict[str, Any],
    hash_field: str,
) -> dict[str, Any]:
    return {
        **body,
        hash_field: canonical_sha256(body),
    }


def _publication_authorities(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    *,
    terminal_status: str,
    report_kind: str,
    artifact_sha256: str,
    artifact_receipt: StoreRecordReceipt,
    report_material: dict[str, Any],
    external_publication: publisher.VerifiedExternalPublication,
) -> tuple[
    store_module.VerifiedPublicationIntent,
    store_module.VerifiedDurablePublicationReceipt,
]:
    publication = external_publication.publication
    store_instance_id = "2" * 64
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
        "terminal_status": terminal_status,
        "report_kind": report_kind,
        "artifact_sha256": artifact_sha256,
        "artifact_store_receipt_sha256": canonical_sha256(
            _store_receipt_material(artifact_receipt)
        ),
        "terminal_reconstruction_material_sha256": "3" * 64,
        "terminal_reconstruction_material_store_receipt_sha256": "4" * 64,
        "record_counts": copy.deepcopy(report_material["record_counts"]),
        "record_commitment_sha256": report_material[
            "record_commitment_sha256"
        ],
        "store_instance_id": store_instance_id,
        "store_journal_sequence": 10,
        "store_journal_tip_sha256": "5" * 64,
        "normal_attempt_elapsed_at_intent_prepare_hex": (1.0).hex(),
        "predecessor_publication_sha256": publication[
            "predecessor_publication_sha256"
        ],
        "tag_ref": publication["tag_ref"],
        "tag_target_commit": publication["tag_target_commit"],
        "tag_message_sha256": publication["tag_message_sha256"],
        "expected_tag_object_sha1": publication[
            "remote_tag_object_sha1"
        ],
        "expected_peeled_commit": publication["remote_peeled_commit"],
        "expected_publication_sha256": publication["publication_sha256"],
        "remote_name": publication["remote_name"],
        "remote_url": publication["remote_url"],
        "intent_status": "publication_pending",
        "research_effect_authority_invalidated": True,
        "semantic_result_release_blocked": True,
        "next_stage_authority_blocked": True,
        "external_cost_usd": 0,
    }
    intent_material = _self_hashed(
        intent_body,
        "publication_intent_sha256",
    )
    intent_store_receipt = StoreRecordReceipt(
        table="publication_intents",
        identity=f"publication_intent:{plan['attempt_id']}",
        attempt_id=plan["attempt_id"],
        payload_sha256=canonical_sha256(intent_material),
        journal_sequence=11,
        journal_entry_sha256="6" * 64,
    )
    intent = store_module.VerifiedPublicationIntent(
        material=intent_material,
        store_receipt=intent_store_receipt,
        _sentinel=store_module._GOVERNANCE_SENTINEL,
    )
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
        "implementation_manifest_sha256": manifest[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": manifest["implementation_commit"],
        "store_instance_id": store_instance_id,
        "attempt_id": plan["attempt_id"],
        "publication_intent_sha256": intent_material[
            "publication_intent_sha256"
        ],
        "publication_intent_store_receipt_sha256": canonical_sha256(
            _store_receipt_material(intent_store_receipt)
        ),
        "publication_remote_observation_sha256": "7" * 64,
        "recovery_invocation_completion_sha256": (
            PUBLICATION_NORMAL_NO_RECOVERY_COMPLETION_SHA256
        ),
        "pre_push_authorization_sha256": (
            PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
        ),
        "external_publication_sha256": publication[
            "publication_sha256"
        ],
        "remote_tag_object_sha1": publication[
            "remote_tag_object_sha1"
        ],
        "remote_peeled_commit": publication["remote_peeled_commit"],
        "pre_receipt_store_journal_sequence": 12,
        "pre_receipt_store_journal_tip_sha256": "8" * 64,
        "receipt_status": "publication_verified",
        "publication_capability_invalidated": True,
        "terminalization_capability_required": True,
    }
    receipt_material = _self_hashed(
        receipt_body,
        "publication_receipt_sha256",
    )
    receipt_store_receipt = StoreRecordReceipt(
        table="publication_receipts",
        identity=f"publication_receipt:{plan['attempt_id']}",
        attempt_id=plan["attempt_id"],
        payload_sha256=canonical_sha256(receipt_material),
        journal_sequence=13,
        journal_entry_sha256="9" * 64,
    )
    durable_receipt = store_module.VerifiedDurablePublicationReceipt(
        material=receipt_material,
        store_receipt=receipt_store_receipt,
        external_publication=external_publication,
        _sentinel=store_module._GOVERNANCE_SENTINEL,
    )
    return intent, durable_receipt


def _acquisition_report(
    vault_path: Path,
) -> acquisition.VerifiedAcquisitionReport:
    checks = {
        name: canonical_sha256({"check": name})
        for name in ACQUISITION_VALIDATION_CHECKS
    }
    body = {
        "schema_version": acquisition.ACQUISITION_VALIDATION_SCHEMA_VERSION,
        "verifier_id": acquisition.ACQUISITION_VALIDATION_VERIFIER_ID,
        "verdict": "pass",
        "stage": "development",
        "attempt_id": DEVELOPMENT_ACQUISITION_ID,
        "attempt_kind": "development_acquisition",
        "acquisition_plan_sha256": acquisition.build_acquisition_plan(
            "development"
        )["acquisition_plan_sha256"],
        "bundle_sha256": "3" * 64,
        "manifest_sha256": "4" * 64,
        "private_index_sha256": "5" * 64,
        "predecessor_chain_bundle_sha256s": [],
        "checks": checks,
        "check_set_sha256": canonical_sha256(checks),
    }
    payload = {
        **body,
        "validation_sha256": canonical_sha256(body),
    }
    vault = vault_module.open_test_acquisition_vault(vault_path)
    capability = store_module.EffectCapability(
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
        allowed_effects=("official_sec_network", "market_network"),
        receipt=StoreRecordReceipt(
            table="attempts",
            identity="attempt:test",
            attempt_id=DEVELOPMENT_ACQUISITION_ID,
            payload_sha256="a" * 64,
            journal_sequence=1,
            journal_entry_sha256="b" * 64,
        ),
        store_instance_id="c" * 64,
        store_nonce="d" * 64,
        transition_sha256="e" * 64,
        _sentinel=store_module._CAPABILITY_SENTINEL,
    )

    class FakeStore:
        @staticmethod
        def authorize_effect(
            supplied: store_module.EffectCapability, effect: str
        ) -> None:
            if supplied is not capability or effect not in supplied.allowed_effects:
                raise RuntimeError("unauthorized")

    handle = vault_module._seal_quarantine(
        vault,
        store=FakeStore(),
        capability=capability,
        stage="development",
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
        bundle_sha256=body["bundle_sha256"],
        manifest_sha256=body["manifest_sha256"],
        private_index_sha256=body["private_index_sha256"],
        predecessor_handles=(),
        quarantine={},
    )
    return acquisition.VerifiedAcquisitionReport(
        payload,
        vault=vault,
        handle=handle,
        _sentinel=acquisition._VERIFIED_REPORT_SENTINEL,
    )


def _gate_checks(stage: str, *, fail: str | None = None) -> dict[str, bool]:
    checks = {
        name: True
        for name in build_contract_manifest()["gates"][stage]
    }
    if fail is not None:
        checks[fail] = False
    return checks


def _joint_report(
    plan: dict[str, Any],
    *,
    checks: dict[str, bool],
) -> dict[str, Any]:
    stage = "development"
    failures = [name for name in checks if not checks[name]]
    gate_body = {
        "checks": copy.deepcopy(checks),
        "passed": not failures,
        "failed_checks": failures,
    }
    gate = {
        **gate_body,
        "gate_report_sha256": canonical_sha256(gate_body),
    }
    metrics_input = {
        "stage_metrics_input_sha256": "6" * 64,
    }
    metrics = {"stage_metrics_sha256": "7" * 64}
    proofs = {"proof": {"proof_sha256": "8" * 64}}
    evaluation_body = {
        "schema_version": DETERMINISTIC_EVALUATION_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": stage,
        "stage_input_bundle_sha256": "9" * 64,
        "metrics_input": metrics_input,
        "metrics_input_sha256": metrics_input[
            "stage_metrics_input_sha256"
        ],
        "stage_metrics": metrics,
        "stage_metrics_sha256": metrics["stage_metrics_sha256"],
        "gate_report": gate,
        "gate_report_sha256": gate["gate_report_sha256"],
        "no_leverage_proofs": proofs,
        "no_leverage_proofs_sha256": canonical_sha256(proofs),
    }
    evaluation = {
        **evaluation_body,
        "deterministic_evaluation_sha256": canonical_sha256(
            evaluation_body
        ),
    }
    joint_body = {
        "schema_version": JOINT_STAGE_REPORT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "metric_stage": stage,
        "attempt_id": plan["attempt_id"],
        "attempt_plan_sha256": plan["attempt_plan_sha256"],
        "implementation_manifest_sha256": plan[
            "implementation_manifest_sha256"
        ],
        "deterministic_evaluation": evaluation,
        "deterministic_evaluation_sha256": evaluation[
            "deterministic_evaluation_sha256"
        ],
    }
    return {
        **joint_body,
        "joint_stage_report_sha256": canonical_sha256(joint_body),
    }


def test_manifest_binds_v21_dependency_closure(
    manifest: dict[str, Any],
) -> None:
    assert validate_implementation_manifest(manifest) == manifest
    assert manifest["schema_version"].startswith(f"{CONTRACT_VERSION}-")
    assert manifest["dependency_sources"]
    assert manifest["dependency_closure_sha256"] == canonical_sha256(
        {
            "dependency_sources": manifest["dependency_sources"],
            "external_distributions": manifest["external_distributions"],
            "numerical_time_distributions": manifest[
                "numerical_time_distributions"
            ],
        }
    )
    changed = copy.deepcopy(manifest)
    changed["schema_version"] = (
        "aapl-sec-gemma-online-risk-overlay-v2-implementation-manifest-v2"
    )
    changed["implementation_manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "implementation_manifest_sha256"
        }
    )
    with pytest.raises(SecGemmaOnlineRiskOverlayAttemptError):
        validate_implementation_manifest(changed)


def test_plans_are_one_shot_and_final_rejects_bare_registry_input(
    manifest: dict[str, Any],
) -> None:
    first = build_attempt_plan(
        implementation_manifest=manifest,
        attempt_id=ATTEMPT_IDS[0],
    )
    second = build_attempt_plan(
        implementation_manifest=manifest,
        attempt_id=ATTEMPT_IDS[1],
        prerequisite_terminal_transition=_terminal_pass(manifest, first),
    )
    assert first["attempt_ordinal"] == 1
    assert second["attempt_ordinal"] == 2
    assert first["one_shot"] is True
    assert first["retry_permitted"] is False
    with pytest.raises(SecGemmaOnlineRiskOverlayAttemptError):
        build_attempt_plan(
            implementation_manifest=manifest,
            attempt_id=ATTEMPT_IDS[-1],
            prerequisite_terminal_transition=_terminal_pass(
                manifest, second
            ),
            final_registry_authorization={"authorization_sha256": "a" * 64},
        )
    with pytest.raises(TypeError):
        build_attempt_plan(
            implementation_manifest=manifest,
            attempt_id=ATTEMPT_IDS[-1],
            final_registry_pin_sha256="a" * 64,  # type: ignore[call-arg]
        )


def test_acquisition_terminal_evidence_binds_exact_report_receipt_and_publication(
    manifest: dict[str, Any],
    tmp_path: Path,
) -> None:
    plan = build_attempt_plan(
        implementation_manifest=manifest,
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
    )
    report = _acquisition_report(tmp_path / "vault.sqlite3")
    report_payload = report.as_dict()
    receipt = _receipt(
        attempt_id=plan["attempt_id"],
        payload=report_payload,
        identity="terminal.acquisition",
    )
    report_material = _material(plan)
    publication_proof = _publication(
        manifest,
        attempt_id=plan["attempt_id"],
        terminal_status=TERMINAL_PASS,
        report_kind=publisher.ACQUISITION_PASS,
        artifact_sha256=report_payload["validation_sha256"],
    )
    publication_intent, publication_receipt = _publication_authorities(
        manifest,
        plan,
        terminal_status=TERMINAL_PASS,
        report_kind=publisher.ACQUISITION_PASS,
        artifact_sha256=report_payload["validation_sha256"],
        artifact_receipt=receipt,
        report_material=report_material,
        external_publication=publication_proof,
    )
    verified = issue_verified_acquisition_terminal_evidence(
        implementation_manifest=manifest,
        attempt_plan=plan,
        acquisition_report=report,
        report_material=report_material,
        acquisition_artifact_receipt=receipt,
        publication_intent=publication_intent,
        publication_receipt=publication_receipt,
    )
    evidence = validate_acquisition_terminal_evidence(
        verified,
        implementation_manifest=manifest,
        attempt_plan=plan,
    )
    assert set(evidence) == set(ACQUISITION_TERMINAL_EVIDENCE_FIELDS)
    assert evidence["terminal_status"] == TERMINAL_PASS
    assert len(ACQUISITION_VALIDATION_CHECKS) == 7

    with pytest.raises(SecGemmaOnlineRiskOverlayAttemptError):
        issue_verified_acquisition_terminal_evidence(
            implementation_manifest=manifest,
            attempt_plan=plan,
            acquisition_report=report.as_dict(),
            report_material=report_material,
            acquisition_artifact_receipt=receipt,
            publication_intent=publication_intent,
            publication_receipt=publication_receipt,
        )
    with pytest.raises(SecGemmaOnlineRiskOverlayAttemptError):
        issue_verified_acquisition_terminal_evidence(
            implementation_manifest=manifest,
            attempt_plan=plan,
            acquisition_report=report,
            report_material=report_material,
            acquisition_artifact_receipt=receipt,
            publication_intent=publication_intent.as_dict(),
            publication_receipt=publication_receipt,
        )
    with pytest.raises(SecGemmaOnlineRiskOverlayAttemptError):
        issue_verified_acquisition_terminal_evidence(
            implementation_manifest=manifest,
            attempt_plan=plan,
            acquisition_report=report,
            report_material=report_material,
            acquisition_artifact_receipt=receipt,
            publication_intent=publication_intent,
            publication_receipt=publication_receipt.as_dict(),
        )
    connection = sqlite3.connect(report._vault._database_path)
    try:
        connection.execute(
            "UPDATE quarantine_entries SET generation=generation+1"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(SecGemmaOnlineRiskOverlayAttemptError):
        validate_acquisition_terminal_evidence(
            verified,
            implementation_manifest=manifest,
            attempt_plan=plan,
        )


@pytest.mark.parametrize("failed", [False, True])
def test_scored_terminal_evidence_uses_literal_gate_set_and_failed_gate_verdict(
    manifest: dict[str, Any],
    failed: bool,
) -> None:
    acquisition_plan = build_attempt_plan(
        implementation_manifest=manifest,
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
    )
    plan = build_attempt_plan(
        implementation_manifest=manifest,
        attempt_id=DEVELOPMENT_ATTEMPT_ID,
        prerequisite_terminal_transition=_terminal_pass(
            manifest, acquisition_plan
        ),
    )
    names = list(build_contract_manifest()["gates"]["development"])
    failing_name = names[-1] if failed else None
    checks = _gate_checks("development", fail=failing_name)
    joint = _joint_report(plan, checks=checks)
    receipt = _receipt(
        attempt_id=plan["attempt_id"],
        payload=joint,
        identity=f"terminal.scored.{failed}",
    )
    report_material = _material(plan)
    terminal_status = TERMINAL_FAIL if failed else TERMINAL_PASS
    report_kind = (
        publisher.SCORED_FAILED_GATE if failed else publisher.SCORED_PASS
    )
    publication_proof = _publication(
        manifest,
        attempt_id=plan["attempt_id"],
        terminal_status=terminal_status,
        report_kind=report_kind,
        artifact_sha256=joint["joint_stage_report_sha256"],
    )
    publication_intent, publication_receipt = _publication_authorities(
        manifest,
        plan,
        terminal_status=terminal_status,
        report_kind=report_kind,
        artifact_sha256=joint["joint_stage_report_sha256"],
        artifact_receipt=receipt,
        report_material=report_material,
        external_publication=publication_proof,
    )
    verified = issue_verified_scored_terminal_evidence(
        implementation_manifest=manifest,
        attempt_plan=plan,
        joint_stage_report=joint,
        report_material=report_material,
        joint_artifact_receipt=receipt,
        gate_checks=checks,
        publication_intent=publication_intent,
        publication_receipt=publication_receipt,
    )
    evidence = validate_scored_terminal_evidence(
        verified,
        implementation_manifest=manifest,
        attempt_plan=plan,
    )
    assert set(evidence) == set(SCORED_TERMINAL_EVIDENCE_FIELDS)
    assert evidence["terminal_status"] == terminal_status
    assert evidence["verdict"] == ("failed_gate" if failed else "pass")
    assert evidence["failed_gate_names"] == (
        [failing_name] if failed else []
    )
    assert max(map(len, checks)) > 64

    renamed = dict(checks)
    renamed["caller_selected_pass"] = renamed.pop(names[0])
    with pytest.raises(SecGemmaOnlineRiskOverlayAttemptError):
        issue_verified_scored_terminal_evidence(
            implementation_manifest=manifest,
            attempt_plan=plan,
            joint_stage_report=joint,
            report_material=report_material,
            joint_artifact_receipt=receipt,
            gate_checks=renamed,
            publication_intent=publication_intent,
            publication_receipt=publication_receipt,
        )

    wrong_material = _material(plan, commitment="d" * 64)
    wrong_intent, wrong_receipt = _publication_authorities(
        manifest,
        plan,
        terminal_status=terminal_status,
        report_kind=report_kind,
        artifact_sha256=joint["joint_stage_report_sha256"],
        artifact_receipt=receipt,
        report_material=wrong_material,
        external_publication=publication_proof,
    )
    with pytest.raises(SecGemmaOnlineRiskOverlayAttemptError):
        issue_verified_scored_terminal_evidence(
            implementation_manifest=manifest,
            attempt_plan=plan,
            joint_stage_report=joint,
            report_material=report_material,
            joint_artifact_receipt=receipt,
            gate_checks=checks,
            publication_intent=wrong_intent,
            publication_receipt=wrong_receipt,
        )


def test_scored_terminal_rejects_v2_namespace_and_receipt_payload_mismatch(
    manifest: dict[str, Any],
) -> None:
    acquisition_plan = build_attempt_plan(
        implementation_manifest=manifest,
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
    )
    plan = build_attempt_plan(
        implementation_manifest=manifest,
        attempt_id=DEVELOPMENT_ATTEMPT_ID,
        prerequisite_terminal_transition=_terminal_pass(
            manifest, acquisition_plan
        ),
    )
    checks = _gate_checks("development")
    joint = _joint_report(plan, checks=checks)
    old_joint = copy.deepcopy(joint)
    old_joint["schema_version"] = (
        "aapl-sec-gemma-online-risk-overlay-v2-joint-stage-report-v1"
    )
    old_joint["joint_stage_report_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in old_joint.items()
            if key != "joint_stage_report_sha256"
        }
    )
    publication_proof = _publication(
        manifest,
        attempt_id=plan["attempt_id"],
        terminal_status=TERMINAL_PASS,
        report_kind=publisher.SCORED_PASS,
        artifact_sha256=joint["joint_stage_report_sha256"],
    )
    valid_receipt = _receipt(
        attempt_id=plan["attempt_id"],
        payload=joint,
        identity="terminal.valid-v22",
    )
    report_material = _material(plan)
    publication_intent, publication_receipt = _publication_authorities(
        manifest,
        plan,
        terminal_status=TERMINAL_PASS,
        report_kind=publisher.SCORED_PASS,
        artifact_sha256=joint["joint_stage_report_sha256"],
        artifact_receipt=valid_receipt,
        report_material=report_material,
        external_publication=publication_proof,
    )
    with pytest.raises(SecGemmaOnlineRiskOverlayAttemptError):
        issue_verified_scored_terminal_evidence(
            implementation_manifest=manifest,
            attempt_plan=plan,
            joint_stage_report=old_joint,
            report_material=report_material,
            joint_artifact_receipt=_receipt(
                attempt_id=plan["attempt_id"],
                payload=old_joint,
                identity="terminal.old-v2",
            ),
            gate_checks=checks,
            publication_intent=publication_intent,
            publication_receipt=publication_receipt,
        )

    wrong_receipt = StoreRecordReceipt(
        table="artifacts",
        identity="terminal.wrong-payload",
        attempt_id=plan["attempt_id"],
        payload_sha256="f" * 64,
        journal_sequence=9,
        journal_entry_sha256="e" * 64,
    )
    with pytest.raises(SecGemmaOnlineRiskOverlayAttemptError):
        issue_verified_scored_terminal_evidence(
            implementation_manifest=manifest,
            attempt_plan=plan,
            joint_stage_report=joint,
            report_material=report_material,
            joint_artifact_receipt=wrong_receipt,
            gate_checks=checks,
            publication_intent=publication_intent,
            publication_receipt=publication_receipt,
        )
