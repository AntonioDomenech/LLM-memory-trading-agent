from __future__ import annotations

import copy
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

import agent_benchmark.sec_gemma_online_risk_overlay_publisher as publisher
import agent_benchmark.sec_gemma_online_risk_overlay_vault as vault_module
from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
    CONSUMED,
    DETERMINISTIC_EVALUATION_SCHEMA_VERSION,
    JOINT_STAGE_REPORT_SCHEMA_VERSION,
    REPORT_RECORD_TABLES,
    TERMINAL_FAIL,
    TERMINAL_INDETERMINATE,
    TERMINAL_PASS,
    build_attempt_plan,
    build_implementation_manifest,
    issue_verified_acquisition_terminal_evidence,
    issue_verified_scored_terminal_evidence,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    ACQUISITION_VALIDATION_CHECKS,
    BRANCH_NAME,
    CONFIRMATION_ATTEMPT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_ACQUISITION_ID,
    DEVELOPMENT_ATTEMPT_ID,
    EXTERNAL_TAG_REF_TEMPLATE,
    build_contract_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_features import (
    FEATURE_ROW_SCHEMA_VERSION,
)
from agent_benchmark.sec_gemma_online_risk_overlay_runner import (
    build_phase_output,
)
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
    EXPECTED_ORIGIN_URL,
    PREREGISTRATION_COMMIT,
    REQUIRED_NEW_SOURCE_PATHS,
    verify_live_source_tree,
)
from agent_benchmark.sec_gemma_online_risk_overlay_store import (
    ANCHOR_FILENAME,
    ANCHOR_RELATIVE_DIRECTORY,
    DATABASE_FILENAME,
    STATE_RELATIVE_DIRECTORY,
    EffectCapability,
    SecGemmaOnlineRiskOverlayStore,
    SecGemmaOnlineRiskOverlayStoreConflict,
    SecGemmaOnlineRiskOverlayStoreError,
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
    repo = _implementation_repo(tmp_path_factory.mktemp("store-v21") / "repo")
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


def _publication(
    manifest: dict[str, Any],
    *,
    attempt_id: str,
    terminal_status: str,
    report_kind: str,
    artifact_sha256: str,
    predecessor_publication_sha256: str = (
        publisher.PUBLICATION_GENESIS_SHA256
    ),
) -> publisher.VerifiedExternalPublication:
    message = publisher.build_external_tag_message(
        implementation_manifest=manifest,
        attempt_id=attempt_id,
        terminal_status=terminal_status,
        report_kind=report_kind,
        artifact_sha256=artifact_sha256,
        predecessor_publication_sha256=predecessor_publication_sha256,
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


def _acquisition_report(
    store: SecGemmaOnlineRiskOverlayStore,
    capability: EffectCapability,
) -> Any:
    import agent_benchmark.sec_gemma_online_risk_overlay_acquisition as acquisition

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
    vault = vault_module.open_test_acquisition_vault(
        store.path.parent / "test-quarantine.sqlite3"
    )
    handle = vault_module._seal_quarantine(
        vault,
        store=store,
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


def _first_consumed(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> tuple[
    SecGemmaOnlineRiskOverlayStore,
    dict[str, Any],
    EffectCapability,
]:
    manifest = repo_and_manifest[1]
    plan = build_attempt_plan(
        implementation_manifest=manifest,
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
    )
    store = _store(repo_and_manifest)
    store.register_attempt(plan)
    return store, plan, store.consume_attempt(plan["attempt_id"])


def _acquisition_phase_evidence(
    report: dict[str, Any],
    *,
    command: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import agent_benchmark.sec_gemma_online_risk_overlay_acquisition as acquisition

    summary_body = {
        "schema_version": acquisition.ACQUISITION_PUBLIC_SUMMARY_SCHEMA_VERSION,
        "stage": report["stage"],
        "attempt_id": report["attempt_id"],
        "attempt_kind": report["attempt_kind"],
        "acquisition_plan_sha256": report["acquisition_plan_sha256"],
        "predecessor_bundle_sha256": (
            report["predecessor_chain_bundle_sha256s"][-1]
            if report["predecessor_chain_bundle_sha256s"]
            else None
        ),
        "bundle_sha256": report["bundle_sha256"],
        "manifest_sha256": report["manifest_sha256"],
        "private_index_sha256": report["private_index_sha256"],
        "sec_catalog_source_count": 1,
        "sec_primary_document_count": 1,
        "market_response_count": 6,
        "model_request_count": 1,
        "model_slice_sha256": "1" * 64,
        "stage_slice_sha256": "2" * 64,
        "market_elapsed_seconds_hex": (1.5).hex(),
        "total_raw_byte_count": 160,
        "complete_batch": True,
        "quarantine_only": True,
        "production_authority": True,
    }
    summary = {
        **summary_body,
        "public_summary_sha256": canonical_sha256(summary_body),
    }
    accounting_body = {
        "schema_version": (
            acquisition.ACQUISITION_REQUEST_ACCOUNTING_SCHEMA_VERSION
        ),
        "stage": report["stage"],
        "attempt_id": report["attempt_id"],
        "sec_request_count": 2,
        "market_request_count": 6,
        "sec_bytes": 100,
        "market_bytes": 60,
        "network_request_count": 8,
        "retry_count": 0,
        "redirect_count": 0,
        "market_elapsed_seconds_hex": (1.5).hex(),
    }
    accounting = {
        **accounting_body,
        "accounting_sha256": canonical_sha256(accounting_body),
    }
    safe = {
        "verified_acquisition_report": copy.deepcopy(report),
        "public_summary": summary,
        "request_accounting": accounting,
    }
    counters = {
        "sec_request_count": 2,
        "market_request_count": 6,
        "model_call_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "model_pull_count": 0,
        "paid_api_call_count": 0,
    }
    output = build_phase_output(
        command=command,
        phase="acquisition",
        counters=counters,
        payload=safe,
    )
    receipt_body = {
        "phase": "acquisition",
        "phase_output_sha256": output["phase_output_sha256"],
        "payload_sha256": output["payload_sha256"],
        "counters": counters,
        "elapsed_seconds_hex": (1.0).hex(),
        "market_elapsed_seconds_hex": (1.5).hex(),
        "deadline_monotonic_hex": (10.0).hex(),
    }
    receipt = {
        **receipt_body,
        "phase_receipt_sha256": canonical_sha256(receipt_body),
    }
    evidence = {
        "command": command,
        "phase": "acquisition",
        "phase_output_sha256": output["phase_output_sha256"],
        "payload_sha256": output["payload_sha256"],
        "phase_receipt": receipt,
        **safe,
    }
    return safe, evidence


def _fake_acquisition_report(
    *,
    stage: str,
    attempt_id: str,
    attempt_kind: str,
    predecessor_bundles: list[str],
) -> dict[str, Any]:
    import agent_benchmark.sec_gemma_online_risk_overlay_acquisition as acquisition

    marker = {
        "confirmation": ("a", "b", "c"),
        "final": ("d", "e", "f"),
    }[stage]
    checks = {
        name: canonical_sha256(
            {
                "stage": stage,
                "check": name,
            }
        )
        for name in ACQUISITION_VALIDATION_CHECKS
    }
    body = {
        "schema_version": acquisition.ACQUISITION_VALIDATION_SCHEMA_VERSION,
        "verifier_id": acquisition.ACQUISITION_VALIDATION_VERIFIER_ID,
        "verdict": "pass",
        "stage": stage,
        "attempt_id": attempt_id,
        "attempt_kind": attempt_kind,
        "acquisition_plan_sha256": acquisition.build_acquisition_plan(
            stage
        )["acquisition_plan_sha256"],
        "bundle_sha256": marker[0] * 64,
        "manifest_sha256": marker[1] * 64,
        "private_index_sha256": marker[2] * 64,
        "predecessor_chain_bundle_sha256s": list(predecessor_bundles),
        "checks": checks,
        "check_set_sha256": canonical_sha256(checks),
    }
    return {**body, "validation_sha256": canonical_sha256(body)}


def _seal_acquisition(
    store: SecGemmaOnlineRiskOverlayStore,
    plan: dict[str, Any],
    capability: EffectCapability,
    manifest: dict[str, Any],
) -> Any:
    store.append_evidence(
        capability=capability,
        effect="official_sec_network",
        identity=f"evidence:{plan['attempt_id']}",
        payload={"request_count": 1},
    )
    report = _acquisition_report(store, capability)
    _, phase_evidence = _acquisition_phase_evidence(
        report.as_dict(),
        command="development_acquisition",
    )
    store.append_evidence(
        capability=capability,
        effect="official_sec_network",
        identity=f"phase:{plan['attempt_id']}:acquisition",
        payload=phase_evidence,
    )
    receipt = store.append_artifact(
        capability=capability,
        effect="deterministic_private_quarantine",
        identity=f"terminal_artifact:{plan['attempt_id']}",
        payload=report.as_dict(),
    )
    material = store.terminal_evidence_material(capability)
    publication_proof = _publication(
        manifest,
        attempt_id=plan["attempt_id"],
        terminal_status=TERMINAL_PASS,
        report_kind=publisher.ACQUISITION_PASS,
        artifact_sha256=report["validation_sha256"],
    )
    evidence = issue_verified_acquisition_terminal_evidence(
        implementation_manifest=manifest,
        attempt_plan=plan,
        acquisition_report=report,
        report_material=material,
        acquisition_artifact_receipt=receipt,
        external_publication=publication_proof,
    )
    store.finish_attempt(
        capability,
        terminal_status=TERMINAL_PASS,
        verified_terminal_evidence=evidence,
    )
    return evidence


def _gate_checks(
    *,
    failed: bool,
    stage: str = "development",
) -> dict[str, bool]:
    checks = {
        name: True
        for name in build_contract_manifest()["gates"][stage]
    }
    if failed:
        checks[next(iter(checks))] = False
    return checks


def _joint_report(
    plan: dict[str, Any],
    *,
    checks: dict[str, bool],
    stage: str = "development",
) -> dict[str, Any]:
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
    metrics_input = {"stage_metrics_input_sha256": "6" * 64}
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
    body = {
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
    return {**body, "joint_stage_report_sha256": canonical_sha256(body)}


def _second_consumed(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> tuple[
    SecGemmaOnlineRiskOverlayStore,
    dict[str, Any],
    EffectCapability,
]:
    manifest = repo_and_manifest[1]
    store, first_plan, first_capability = _first_consumed(
        repo_and_manifest
    )
    _seal_acquisition(
        store,
        first_plan,
        first_capability,
        manifest,
    )
    predecessor = store.attempt_history(first_plan["attempt_id"])[-1]
    plan = build_attempt_plan(
        implementation_manifest=manifest,
        attempt_id=DEVELOPMENT_ATTEMPT_ID,
        prerequisite_terminal_transition=predecessor,
    )
    store.register_attempt(plan)
    return store, plan, store.consume_attempt(plan["attempt_id"])


def _append_scored_minimums(
    store: SecGemmaOnlineRiskOverlayStore,
    capability: EffectCapability,
) -> None:
    store.append_evidence(
        capability=capability,
        effect="chronological_replay",
        identity=f"evidence:{capability.attempt_id}",
        payload={"complete": True},
    )
    store.append_feature(
        capability=capability,
        effect="chronological_replay",
        identity=f"feature:{capability.attempt_id}",
        payload={"complete": True},
    )
    store.append_prediction(
        capability=capability,
        effect="chronological_replay",
        identity=f"prediction:{capability.attempt_id}",
        payload={"complete": True},
    )
    store.append_lesson(
        capability=capability,
        effect="chronological_replay",
        identity=f"lesson:{capability.attempt_id}",
        payload={"complete": True},
    )
    store.append_ledger(
        capability=capability,
        effect="chronological_replay",
        identity=f"ledger:{capability.attempt_id}",
        payload={"complete": True},
    )


def _feature_row(
    *,
    stage: str,
    accession: str,
    decision_session: str,
    acceptance_datetime: str,
) -> dict[str, Any]:
    bindings = {
        "stage": stage,
        "accession": accession,
    }
    body = {
        "schema_version": FEATURE_ROW_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "accession_number": accession,
        "form": "10-Q",
        "decision_session": decision_session,
        "acceptance_datetime": acceptance_datetime,
        "artifact_stage": stage,
        "test_marker": True,
        "upstream_bindings": bindings,
        "upstream_bindings_sha256": canonical_sha256(bindings),
    }
    return {**body, "feature_row_sha256": canonical_sha256(body)}


def _append_scoped_scored_records(
    store: SecGemmaOnlineRiskOverlayStore,
    capability: EffectCapability,
    *,
    stage: str,
    feature_rows: list[dict[str, Any]],
    acquisition_phase_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attempt_id = capability.attempt_id
    if acquisition_phase_evidence is not None:
        store.append_evidence(
            capability=capability,
            effect="official_sec_network",
            identity=f"phase:{attempt_id}:acquisition",
            payload=acquisition_phase_evidence,
        )
    store.append_evidence(
        capability=capability,
        effect="ollama_runtime_identity",
        identity=f"phase:{attempt_id}:runtime_identity",
        payload={
            "stage": stage,
            "runtime": "pinned",
        },
    )
    for row in feature_rows:
        store.append_feature(
            capability=capability,
            effect="chronological_replay",
            identity=f"feature:{row['feature_row_sha256']}",
            payload=row,
        )
    prediction = {
        "stage": stage,
        "prediction": "cash",
    }
    prediction_hash = canonical_sha256(prediction)
    prediction_identity = (
        f"prediction:{stage}:semantic-online:{prediction_hash}"
    )
    store.append_prediction(
        capability=capability,
        effect="chronological_replay",
        identity=prediction_identity,
        payload=prediction,
    )
    lesson = {
        "stage": stage,
        "lesson": "chronological",
    }
    lesson_hash = canonical_sha256(lesson)
    lesson_identity = f"lesson:{stage}:{lesson_hash}"
    store.append_lesson(
        capability=capability,
        effect="chronological_replay",
        identity=lesson_identity,
        payload=lesson,
    )
    ledger = {
        "stage": stage,
        "ledger_id": "semantic-online-5bps",
    }
    ledger_identity = f"ledger:{stage}:semantic-online-5bps"
    store.append_ledger(
        capability=capability,
        effect="chronological_replay",
        identity=ledger_identity,
        payload=ledger,
    )
    return {
        "prediction": prediction_identity,
        "lesson": lesson_identity,
        "ledger": ledger_identity,
    }


def _finish_scored_pass(
    store: SecGemmaOnlineRiskOverlayStore,
    plan: dict[str, Any],
    capability: EffectCapability,
    manifest: dict[str, Any],
    *,
    stage: str,
) -> Any:
    checks = _gate_checks(failed=False, stage=stage)
    joint = _joint_report(plan, checks=checks, stage=stage)
    receipt = store.append_artifact(
        capability=capability,
        effect="joint_report_seal",
        identity=f"terminal_artifact:{plan['attempt_id']}",
        payload=joint,
    )
    evidence = issue_verified_scored_terminal_evidence(
        implementation_manifest=manifest,
        attempt_plan=plan,
        joint_stage_report=joint,
        report_material=store.terminal_evidence_material(capability),
        joint_artifact_receipt=receipt,
        gate_checks=checks,
        external_publication=_publication(
            manifest,
            attempt_id=plan["attempt_id"],
            terminal_status=TERMINAL_PASS,
            report_kind=publisher.SCORED_PASS,
            artifact_sha256=joint["joint_stage_report_sha256"],
        ),
    )
    store.finish_attempt(
        capability,
        terminal_status=TERMINAL_PASS,
        verified_terminal_evidence=evidence,
    )
    return evidence


def test_store_uses_v21_namespace_and_exclusive_lock(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    repo = repo_and_manifest[0]
    store = _store(repo_and_manifest)
    try:
        assert "v2_1" in STATE_RELATIVE_DIRECTORY.as_posix()
        assert "v2_1" in ANCHOR_RELATIVE_DIRECTORY.as_posix()
        assert store.path == repo / STATE_RELATIVE_DIRECTORY / DATABASE_FILENAME
        assert store.anchor_path == (
            repo / ANCHOR_RELATIVE_DIRECTORY / ANCHOR_FILENAME
        )
        with pytest.raises(
            SecGemmaOnlineRiskOverlayStoreError,
            match="not preregistered",
        ):
            store.terminal_acquisition_phase_evidence(
                DEVELOPMENT_ATTEMPT_ID
            )
        with pytest.raises(
            SecGemmaOnlineRiskOverlayStoreError,
            match="did not terminal-pass",
        ):
            store.terminal_acquisition_phase_evidence(
                DEVELOPMENT_ACQUISITION_ID
            )
        with pytest.raises(
            SecGemmaOnlineRiskOverlayStoreError,
            match="not preregistered",
        ):
            store.predecessor_feature_rows("unknown")
        with pytest.raises(SecGemmaOnlineRiskOverlayStoreConflict):
            _store(repo_and_manifest)
    finally:
        store.close()


def test_terminal_pass_rejects_missing_or_bare_evidence(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    store, plan, capability = _first_consumed(repo_and_manifest)
    try:
        with pytest.raises(
            SecGemmaOnlineRiskOverlayStoreError,
            match="opaque verifier-issued",
        ):
            store.finish_attempt(
                capability,
                terminal_status=TERMINAL_PASS,
            )
        store.append_evidence(
            capability=capability,
            effect="official_sec_network",
            identity="evidence:bare",
            payload={"complete": True},
        )
        report = _acquisition_report(store, capability)
        receipt = store.append_artifact(
            capability=capability,
            effect="deterministic_private_quarantine",
            identity="terminal_artifact:bare",
            payload=report.as_dict(),
        )
        evidence = issue_verified_acquisition_terminal_evidence(
            implementation_manifest=repo_and_manifest[1],
            attempt_plan=plan,
            acquisition_report=report,
            report_material=store.terminal_evidence_material(capability),
            acquisition_artifact_receipt=receipt,
            external_publication=_publication(
                repo_and_manifest[1],
                attempt_id=plan["attempt_id"],
                terminal_status=TERMINAL_PASS,
                report_kind=publisher.ACQUISITION_PASS,
                artifact_sha256=report["validation_sha256"],
            ),
        )
        with pytest.raises(
            SecGemmaOnlineRiskOverlayStoreError,
            match="opaque verifier-issued",
        ):
            store.finish_attempt(
                capability,
                terminal_status=TERMINAL_PASS,
                verified_terminal_evidence=evidence.evidence,  # type: ignore[arg-type]
            )
    finally:
        store.close()


def test_acquisition_pass_anchor_binds_entire_evidence_publication_and_receipt(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    store, plan, capability = _first_consumed(repo_and_manifest)
    evidence = _seal_acquisition(
        store,
        plan,
        capability,
        repo_and_manifest[1],
    )
    binding = store.terminal_anchor_binding(plan["attempt_id"])
    assert binding["terminal_evidence"] == evidence.evidence
    assert (
        binding["external_publication"]
        == evidence.external_publication.publication
    )
    assert binding["artifact_receipt"]["payload_sha256"] == (
        evidence.artifact_receipt.payload_sha256
    )
    assert store.attempt_history(plan["attempt_id"])[-1]["status"] == (
        TERMINAL_PASS
    )
    recovered = store.terminal_acquisition_phase_evidence(
        plan["attempt_id"]
    )
    assert set(recovered) == {
        "verified_acquisition_report",
        "public_summary",
        "request_accounting",
    }
    assert (
        recovered["verified_acquisition_report"]["validation_sha256"]
        == binding["terminal_evidence"]["acquisition_validation_sha256"]
    )
    recovered["public_summary"]["complete_batch"] = False
    assert store.terminal_acquisition_phase_evidence(plan["attempt_id"])[
        "public_summary"
    ]["complete_batch"] is True
    with pytest.raises(SecGemmaOnlineRiskOverlayStoreError, match="stale"):
        store.authorize_effect(capability, "official_sec_network")
    store.close()


def test_scoped_stage_records_and_predecessor_rows_survive_reopen(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    manifest = repo_and_manifest[1]
    store, development_plan, development_capability = _second_consumed(
        repo_and_manifest
    )
    development_rows = [
        _feature_row(
            stage="development",
            accession="0000320193-10-000001",
            decision_session="2010-01-04",
            acceptance_datetime="20100104120000",
        ),
        _feature_row(
            stage="development",
            accession="0000320193-15-000001",
            decision_session="2015-01-05",
            acceptance_datetime="20150105120000",
        ),
    ]
    development_identities = _append_scoped_scored_records(
        store,
        development_capability,
        stage="development",
        feature_rows=development_rows,
    )
    _finish_scored_pass(
        store,
        development_plan,
        development_capability,
        manifest,
        stage="development",
    )
    development_acquisition = (
        store.terminal_acquisition_phase_evidence(
            DEVELOPMENT_ACQUISITION_ID
        )
    )

    confirmation_plan = build_attempt_plan(
        implementation_manifest=manifest,
        attempt_id=CONFIRMATION_ATTEMPT_ID,
        prerequisite_terminal_transition=store.attempt_history(
            DEVELOPMENT_ATTEMPT_ID
        )[-1],
    )
    store.register_attempt(confirmation_plan)
    confirmation_capability = store.consume_attempt(
        CONFIRMATION_ATTEMPT_ID
    )
    confirmation_report = _fake_acquisition_report(
        stage="confirmation",
        attempt_id=CONFIRMATION_ATTEMPT_ID,
        attempt_kind="confirmation_scoring",
        predecessor_bundles=[
            development_acquisition["verified_acquisition_report"][
                "bundle_sha256"
            ]
        ],
    )
    confirmation_safe, confirmation_phase = (
        _acquisition_phase_evidence(
            confirmation_report,
            command="confirmation",
        )
    )
    confirmation_rows = [
        _feature_row(
            stage="confirmation",
            accession="0000320193-20-000001",
            decision_session="2020-01-06",
            acceptance_datetime="20200106120000",
        )
    ]
    confirmation_identities = _append_scoped_scored_records(
        store,
        confirmation_capability,
        stage="confirmation",
        feature_rows=confirmation_rows,
        acquisition_phase_evidence=confirmation_phase,
    )
    _finish_scored_pass(
        store,
        confirmation_plan,
        confirmation_capability,
        manifest,
        stage="confirmation",
    )
    store.close()

    with _store(repo_and_manifest) as reopened:
        assert reopened.predecessor_feature_rows("development") == []
        assert reopened.predecessor_feature_rows(
            "confirmation"
        ) == development_rows
        final_rows = reopened.predecessor_feature_rows("final")
        assert final_rows == development_rows + confirmation_rows
        final_rows[0]["artifact_stage"] = "changed"
        assert reopened.predecessor_feature_rows("final")[0][
            "artifact_stage"
        ] == "development"

        recovered_confirmation = (
            reopened.terminal_acquisition_phase_evidence(
                CONFIRMATION_ATTEMPT_ID
            )
        )
        assert recovered_confirmation == confirmation_safe
        assert set(recovered_confirmation) == {
            "verified_acquisition_report",
            "public_summary",
            "request_accounting",
        }

        for table in ("predictions", "lessons", "ledgers"):
            development_receipt = reopened.record_receipt(
                table,
                development_identities[table[:-1]],
            )
            confirmation_receipt = reopened.record_receipt(
                table,
                confirmation_identities[table[:-1]],
            )
            assert development_receipt.identity != (
                confirmation_receipt.identity
            )
            assert development_receipt.attempt_id == (
                DEVELOPMENT_ATTEMPT_ID
            )
            assert confirmation_receipt.attempt_id == (
                CONFIRMATION_ATTEMPT_ID
            )


def test_terminal_evidence_fails_after_current_record_commitment_changes(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    store, plan, capability = _first_consumed(repo_and_manifest)
    manifest = repo_and_manifest[1]
    store.append_evidence(
        capability=capability,
        effect="official_sec_network",
        identity="evidence:first",
        payload={"complete": True},
    )
    report = _acquisition_report(store, capability)
    receipt = store.append_artifact(
        capability=capability,
        effect="deterministic_private_quarantine",
        identity="terminal_artifact:stale",
        payload=report.as_dict(),
    )
    evidence = issue_verified_acquisition_terminal_evidence(
        implementation_manifest=manifest,
        attempt_plan=plan,
        acquisition_report=report,
        report_material=store.terminal_evidence_material(capability),
        acquisition_artifact_receipt=receipt,
        external_publication=_publication(
            manifest,
            attempt_id=plan["attempt_id"],
            terminal_status=TERMINAL_PASS,
            report_kind=publisher.ACQUISITION_PASS,
            artifact_sha256=report["validation_sha256"],
        ),
    )
    store.append_evidence(
        capability=capability,
        effect="market_network",
        identity="evidence:after-evidence",
        payload={"complete": True},
    )
    try:
        with pytest.raises(
            SecGemmaOnlineRiskOverlayStoreError,
            match="current post-consumption",
        ):
            store.finish_attempt(
                capability,
                terminal_status=TERMINAL_PASS,
                verified_terminal_evidence=evidence,
            )
    finally:
        store.close()


def test_valid_failed_gate_is_anchored_and_invalid_failure_releases_nothing(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    manifest = repo_and_manifest[1]
    store, plan, capability = _second_consumed(repo_and_manifest)
    _append_scored_minimums(store, capability)
    checks = _gate_checks(failed=True)
    joint = _joint_report(plan, checks=checks)
    receipt = store.append_artifact(
        capability=capability,
        effect="joint_report_seal",
        identity=f"terminal_artifact:{plan['attempt_id']}",
        payload=joint,
    )
    evidence = issue_verified_scored_terminal_evidence(
        implementation_manifest=manifest,
        attempt_plan=plan,
        joint_stage_report=joint,
        report_material=store.terminal_evidence_material(capability),
        joint_artifact_receipt=receipt,
        gate_checks=checks,
        external_publication=_publication(
            manifest,
            attempt_id=plan["attempt_id"],
            terminal_status=TERMINAL_FAIL,
            report_kind=publisher.SCORED_FAILED_GATE,
            artifact_sha256=joint["joint_stage_report_sha256"],
        ),
    )
    store.finish_attempt(
        capability,
        terminal_status=TERMINAL_FAIL,
        verified_terminal_evidence=evidence,
    )
    binding = store.terminal_anchor_binding(plan["attempt_id"])
    assert binding["terminal_evidence"]["verdict"] == "failed_gate"
    assert binding["terminal_evidence"]["failed_gate_names"]
    store.close()

    shutil.rmtree(
        repo_and_manifest[0] / STATE_RELATIVE_DIRECTORY,
        ignore_errors=True,
    )
    shutil.rmtree(
        repo_and_manifest[0] / ANCHOR_RELATIVE_DIRECTORY,
        ignore_errors=True,
    )
    store, plan, capability = _second_consumed(repo_and_manifest)
    with pytest.raises(
        SecGemmaOnlineRiskOverlayStoreError,
        match="Terminal-fail requires opaque scored failed-gate evidence",
    ):
        store.finish_attempt(
            capability,
            terminal_status=TERMINAL_FAIL,
        )
    store.finish_attempt(
        capability,
        terminal_status=TERMINAL_INDETERMINATE,
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayStoreError,
        match="no releasable evidence",
    ):
        store.terminal_anchor_binding(plan["attempt_id"])
    store.close()


def test_indeterminate_never_accepts_or_releases_terminal_evidence(
    repo_and_manifest: tuple[Path, dict[str, Any]],
) -> None:
    store, plan, capability = _first_consumed(repo_and_manifest)
    store.finish_attempt(
        capability,
        terminal_status=TERMINAL_INDETERMINATE,
    )
    assert store.attempt_history(plan["attempt_id"])[-1]["status"] == (
        TERMINAL_INDETERMINATE
    )
    with pytest.raises(SecGemmaOnlineRiskOverlayStoreError):
        store.terminal_anchor_binding(plan["attempt_id"])
    store.close()


def test_reopen_completes_exact_evidence_bearing_terminal_intent(
    repo_and_manifest: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, plan, capability = _first_consumed(repo_and_manifest)
    manifest = repo_and_manifest[1]
    store.append_evidence(
        capability=capability,
        effect="official_sec_network",
        identity="evidence:recovery",
        payload={"complete": True},
    )
    report = _acquisition_report(store, capability)
    receipt = store.append_artifact(
        capability=capability,
        effect="deterministic_private_quarantine",
        identity="terminal_artifact:recovery",
        payload=report.as_dict(),
    )
    evidence = issue_verified_acquisition_terminal_evidence(
        implementation_manifest=manifest,
        attempt_plan=plan,
        acquisition_report=report,
        report_material=store.terminal_evidence_material(capability),
        acquisition_artifact_receipt=receipt,
        external_publication=_publication(
            manifest,
            attempt_id=plan["attempt_id"],
            terminal_status=TERMINAL_PASS,
            report_kind=publisher.ACQUISITION_PASS,
            artifact_sha256=report["validation_sha256"],
        ),
    )
    original = store._append_anchor

    def fail_committed(
        event: str,
        database_snapshot: dict[str, Any],
        *,
        attempt_id: str | None,
        event_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if event == "terminal_committed":
            raise SecGemmaOnlineRiskOverlayStoreError(
                "injected terminal commit crash"
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
        match="injected terminal commit crash",
    ):
        store.finish_attempt(
            capability,
            terminal_status=TERMINAL_PASS,
            verified_terminal_evidence=evidence,
        )
    store._close_resources(recover=False)

    with _store(repo_and_manifest) as reopened:
        binding = reopened.terminal_anchor_binding(plan["attempt_id"])
        assert binding["terminal_evidence"] == evidence.evidence
        assert reopened.attempt_history(plan["attempt_id"])[-1][
            "status"
        ] == TERMINAL_PASS
