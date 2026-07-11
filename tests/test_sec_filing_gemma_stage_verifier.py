from __future__ import annotations

import base64
import copy
from datetime import date
import hashlib
import json
from pathlib import Path
import tracemalloc
from types import SimpleNamespace

import pytest

import agent_benchmark.sec_filing_gemma_stage_verifier as verifier_module
from agent_benchmark.sec_filing_gemma_contract import (
    CALENDAR_SOURCE_URLS,
    CONTRACT_VERSION,
    REQUIRED_SOURCE_HASHES,
    REQUIRED_STAGE_VERIFIER_CHECKS,
    build_candidate_manifest,
    build_calendar_source_evidence_manifest,
    build_contract_manifest,
    build_corpus_universe_manifest,
    canonical_sha256,
    session_calendar_sha256,
)
from agent_benchmark.sec_filing_gemma_learner import (
    SecFilingGemmaTwoHeadLearner,
)
from agent_benchmark.sec_filing_gemma_prediction_evidence import (
    AVAILABLE_PREDICTION_STATUS,
)
from agent_benchmark.sec_filing_gemma_stage_verifier import (
    OWNED_HARDENED_TRANSPORT_MODE,
    STAGE_EVIDENCE_SCHEMA_VERSION,
    STAGE_RUNTIME_RECEIPT_SCHEMA_VERSION,
    TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION,
    SecFilingGemmaStageVerifierBlocked,
    SecFilingGemmaStageVerifierError,
    audit_stage_evidence,
    authoritative_prerequisite_validator,
    validate_candidate_source_bytes,
    validate_candidate_source_role_audit,
    validate_calendar_and_universe_snapshot,
    validate_detached_catalog_evidence,
    validate_detached_stage_content_evidence,
    validate_learner_refit_replays,
    validate_model_attempt_batch,
    validate_raw_scores_gates_and_ranking,
    validate_parent_stage_lineage,
    validate_stage_runtime_receipt,
    validate_trusted_stage_content_pin,
)
from agent_benchmark.sec_filing_gemma_source_identity import (
    CANONICAL_SOURCE_ROLE_PATHS,
    canonical_source_tree_sha256,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _detached_boundary() -> dict[str, bool]:
    return {
        "authorizing": False,
        "fresh_network_provenance_verified": False,
        "network_receipt_claims_replayed_not_observed": True,
    }


def _trusted_content_pin(
    stage: str,
    *,
    content_hash: str | None = None,
    artifact_hash: str | None = None,
) -> tuple[dict, dict]:
    content = content_hash or _h(f"{stage} content")
    artifact = artifact_hash or _h(f"{stage} artifact")
    seal = _h(f"{stage} external seal receipt")
    access = {
        "prerequisite_evidence_pin": {
            "stage": stage,
            "content_manifest_sha256": content,
            "stage_artifact_sha256": artifact,
            "external_seal_receipt_sha256": seal,
        }
    }
    body = {
        "schema_version": TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION,
        "stage": stage,
        "content_manifest_sha256": content,
        "stage_artifact_sha256": artifact,
        "external_seal_receipt_sha256": seal,
        "trusted_store_state_sha256": _h(f"{stage} trusted store state"),
    }
    return access, {**body, "pin_sha256": canonical_sha256(body)}


def _candidate_fixture() -> tuple[dict, dict[str, str]]:
    source_payloads: dict[str, bytes] = {}
    for role in REQUIRED_SOURCE_HASHES:
        path = CANONICAL_SOURCE_ROLE_PATHS[role]
        source_payloads[role] = (
            f"unresolved source bytes for {role}".encode("utf-8")
            if path is None
            else (REPOSITORY_ROOT / path).read_bytes()
        )
    source_hashes = {
        role: hashlib.sha256(payload).hexdigest()
        for role, payload in source_payloads.items()
    }
    candidate = build_candidate_manifest(
        model_digest=_h("model"),
        ollama_runtime_fingerprint_sha256=_h("runtime"),
        sec_audit_checksums_json_sha256=_h("audit"),
        sec_catalog_artifact_sha256=_h("catalog"),
        sec_audit_source_commit=_h("audit commit"),
        calendar_source_evidence_sha256=_h("calendar evidence"),
        calendar_sessions_sha256=session_calendar_sha256(EXPECTED_SESSIONS),
        corpus_universe_sha256=_h("universe"),
        corpus_universe_semantic_sha256=_h("universe semantic"),
        identity_lexicon_sha256=_h("identity lexicon"),
        predecessor_reveal_registry_sha256=_h("predecessor registry"),
        holdout_attempt_id=f"{CONTRACT_VERSION}-attempt-001",
        experiment_source_commit=_h("experiment commit"),
        source_tree_sha256=canonical_source_tree_sha256(source_hashes),
        source_hashes=source_hashes,
    )
    encoded = {role: _b64(payload) for role, payload in source_payloads.items()}
    return candidate, encoded


def test_candidate_contract_and_every_source_byte_pin_replay() -> None:
    candidate, sources = _candidate_fixture()
    summary = validate_candidate_source_bytes(
        contract_manifest=build_contract_manifest(),
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate["candidate_sha256"],
        source_bytes_base64_by_role=sources,
    )

    assert summary["candidate_sha256"] == candidate["candidate_sha256"]
    assert summary["stage_verifier_source_sha256"] == candidate["bindings"][
        "source_hashes"
    ]["stage_verifier"]


def test_source_byte_omission_mutation_and_cross_candidate_pin_fail_closed() -> None:
    candidate, sources = _candidate_fixture()
    omitted = copy.deepcopy(sources)
    omitted.pop("runner")
    with pytest.raises(SecFilingGemmaStageVerifierError, match="keys changed"):
        validate_candidate_source_bytes(
            contract_manifest=build_contract_manifest(),
            candidate_manifest=candidate,
            expected_candidate_sha256=candidate["candidate_sha256"],
            source_bytes_base64_by_role=omitted,
        )

    changed = copy.deepcopy(sources)
    changed["runner"] = _b64(b"different runner")
    with pytest.raises(SecFilingGemmaStageVerifierError, match="runner"):
        validate_candidate_source_bytes(
            contract_manifest=build_contract_manifest(),
            candidate_manifest=candidate,
            expected_candidate_sha256=candidate["candidate_sha256"],
            source_bytes_base64_by_role=changed,
        )

    with pytest.raises(SecFilingGemmaStageVerifierError, match="candidate"):
        validate_candidate_source_bytes(
            contract_manifest=build_contract_manifest(),
            candidate_manifest=candidate,
            expected_candidate_sha256=_h("other candidate"),
            source_bytes_base64_by_role=sources,
        )


def test_source_role_audit_uses_frozen_mapping_and_keeps_five_roles_unresolved() -> None:
    candidate, sources = _candidate_fixture()
    receipt = validate_candidate_source_role_audit(
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate["candidate_sha256"],
        source_bytes_base64_by_role=sources,
    )
    assert receipt["complete"] is False
    assert receipt["authorizes"] is False
    assert receipt["unresolved_role_count"] == 5
    assert receipt["unresolved_roles"] == [
        "extractor_prompt",
        "extractor_schema",
        "ledger",
        "market_acquirer",
        "runner",
    ]


def test_runtime_source_audit_accepts_no_caller_paths_or_bytes(monkeypatch) -> None:
    candidate, _ = _candidate_fixture()
    captured: dict[str, object] = {}

    def runtime_audit(**kwargs):
        captured.update(kwargs)
        body = {
            "candidate_sha256": candidate["candidate_sha256"],
            "candidate_source_tree_sha256": candidate["bindings"][
                "source_tree_sha256"
            ],
            "computed_source_tree_sha256": candidate["bindings"][
                "source_tree_sha256"
            ],
            "required_source_roles": list(REQUIRED_SOURCE_HASHES),
            "declared_static_local_import_closure_complete": True,
            "runtime_source_files_verified": True,
            "runtime_module_paths_verified": True,
            "runtime_module_files_attested": True,
            "runtime_executing_code_bytes_attested": False,
            "caller_supplied_paths_or_bytes_accepted": False,
            "complete": False,
            "authorizes": False,
        }
        return {
            **body,
            "source_identity_receipt_sha256": canonical_sha256(body),
        }

    monkeypatch.setattr(
        verifier_module.source_identity_module,
        "audit_runtime_candidate_source_identity",
        runtime_audit,
    )
    receipt = verifier_module.validate_candidate_runtime_source_audit(
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate["candidate_sha256"],
    )
    assert set(captured) == {"candidate_manifest", "expected_candidate_sha256"}
    assert receipt["runtime_module_files_attested"] is True
    assert receipt["runtime_executing_code_bytes_attested"] is False

    forged = runtime_audit()
    forged["caller_supplied_paths_or_bytes_accepted"] = True
    forged_body = {
        key: value
        for key, value in forged.items()
        if key != "source_identity_receipt_sha256"
    }
    forged["source_identity_receipt_sha256"] = canonical_sha256(forged_body)
    monkeypatch.setattr(
        verifier_module.source_identity_module,
        "audit_runtime_candidate_source_identity",
        lambda **kwargs: forged,
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="non-substitutable"):
        verifier_module.validate_candidate_runtime_source_audit(
            candidate_manifest=candidate,
            expected_candidate_sha256=candidate["candidate_sha256"],
        )


def _catalogue_wrapper_evidence(candidate: dict) -> tuple[dict, dict]:
    universe = {
        "universe_sha256": candidate["bindings"]["corpus_universe_sha256"],
        "catalog_artifact_sha256": candidate["bindings"][
            "sec_catalog_artifact_sha256"
        ],
        "calendar_artifact_sha256": candidate["bindings"][
            "calendar_source_evidence_sha256"
        ],
    }
    evidence = {
        "source_payloads": [
            {"name": "CIK0000320193.json", "payload_base64": _b64(b"catalogue")}
        ],
        "request_receipts": [],
        "catalog_artifact": {
            "catalog_artifact_sha256": candidate["bindings"][
                "sec_catalog_artifact_sha256"
            ]
        },
        "expected_source_payload_sha256s": {
            "CIK0000320193.json": hashlib.sha256(b"catalogue").hexdigest()
        },
        "expected_request_receipt_sha256s": {},
        "expected_request_receipts_sha256": _h("catalogue receipts"),
        "expected_catalog_artifact_sha256": candidate["bindings"][
            "sec_catalog_artifact_sha256"
        ],
        "expected_corpus_universe_sha256": candidate["bindings"][
            "corpus_universe_sha256"
        ],
        "expected_calendar_artifact_sha256": candidate["bindings"][
            "calendar_source_evidence_sha256"
        ],
    }
    return evidence, universe


def test_catalogue_wrapper_decodes_bytes_and_rejects_substituted_identity(monkeypatch) -> None:
    candidate, _ = _candidate_fixture()
    evidence, universe = _catalogue_wrapper_evidence(candidate)
    captured: dict[str, object] = {}

    def replay(**kwargs):
        captured.update(kwargs)
        return {
            **_detached_boundary(),
            "schema_version": verifier_module.DETACHED_CATALOG_REPLAY_RECEIPT_SCHEMA_VERSION,
            "catalog_artifact_sha256": evidence["expected_catalog_artifact_sha256"],
            "corpus_universe_sha256": evidence[
                "expected_corpus_universe_sha256"
            ],
            "eligible_records_sha256": _h("eligible"),
            "request_receipts_sha256": evidence[
                "expected_request_receipts_sha256"
            ],
            "source_payload_sha256s": evidence[
                "expected_source_payload_sha256s"
            ],
            "request_receipt_sha256s": {},
            "replay_validation_sha256": _h("catalogue replay"),
        }

    monkeypatch.setattr(verifier_module, "validate_detached_catalog_replay", replay)
    receipt = validate_detached_catalog_evidence(
        evidence,
        candidate_manifest=candidate,
        corpus_universe_manifest=universe,
    )
    assert captured["source_payloads"][0]["payload"] == b"catalogue"
    assert (
        verifier_module.DETACHED_CATALOG_REPLAY_RECEIPT_SCHEMA_VERSION
        != verifier_module.DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION
    )
    assert receipt["schema_version"] == (
        verifier_module.DETACHED_CATALOG_REPLAY_RECEIPT_SCHEMA_VERSION
    )
    assert receipt["replay_validation_sha256"] == _h("catalogue replay")

    substituted = copy.deepcopy(evidence)
    substituted["expected_catalog_artifact_sha256"] = _h("substituted catalog")
    with pytest.raises(SecFilingGemmaStageVerifierError, match="candidate"):
        validate_detached_catalog_evidence(
            substituted,
            candidate_manifest=candidate,
            corpus_universe_manifest=universe,
        )

    wrong_schema = replay()
    wrong_schema["schema_version"] = (
        verifier_module.DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_detached_catalog_replay",
        lambda **kwargs: wrong_schema,
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="schema"):
        validate_detached_catalog_evidence(
            evidence,
            candidate_manifest=candidate,
            corpus_universe_manifest=universe,
        )


def _content_wrapper_evidence(candidate: dict, stage: str = "development") -> tuple[dict, dict]:
    universe_hash = candidate["bindings"]["corpus_universe_sha256"]
    content_hash = _h(f"{stage} content")
    artifact_hash = _h(f"{stage} artifact")
    universe = {
        "universe_sha256": universe_hash,
        "records": [
            {
                "accession_number": "0000320193-05-000001",
                "artifact_stage": stage,
            }
        ],
    }
    evidence = {
        "stage": stage,
        "document_payloads": [
            {
                "accession_number": "0000320193-05-000001",
                "payload_base64": _b64(b"document"),
            }
        ],
        "request_receipts": [],
        "content_manifest": {
            "artifact_stage": stage,
            "corpus_universe_sha256": universe_hash,
            "content_manifest_sha256": content_hash,
        },
        "stage_artifact": {
            "artifact_stage": stage,
            "corpus_universe_sha256": universe_hash,
            "content_manifest_sha256": content_hash,
            "stage_artifact_sha256": artifact_hash,
        },
        "expected_document_sha256s": {
            "0000320193-05-000001": hashlib.sha256(b"document").hexdigest()
        },
        "expected_normalized_text_sha256s": {
            "0000320193-05-000001": _h("normalized")
        },
        "expected_request_receipt_sha256s": {},
        "expected_request_receipts_sha256": _h(f"{stage} receipts"),
        "expected_content_manifest_sha256": content_hash,
        "expected_stage_artifact_sha256": artifact_hash,
    }
    return evidence, universe


def test_content_wrapper_rejects_cross_stage_and_substituted_manifest(monkeypatch) -> None:
    candidate, _ = _candidate_fixture()
    evidence, universe = _content_wrapper_evidence(candidate)

    def replay(**kwargs):
        return {
            **_detached_boundary(),
            "schema_version": verifier_module.DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION,
            "artifact_stage": kwargs["authorized_stage"],
            "corpus_universe_sha256": kwargs[
                "expected_corpus_universe_sha256"
            ],
            "content_manifest_sha256": kwargs[
                "expected_content_manifest_sha256"
            ],
            "stage_artifact_sha256": kwargs["expected_stage_artifact_sha256"],
            "request_receipts_sha256": kwargs[
                "expected_request_receipts_sha256"
            ],
            "primary_document_sha256s": kwargs["expected_document_sha256s"],
            "normalized_text_sha256s": kwargs[
                "expected_normalized_text_sha256s"
            ],
            "request_receipt_sha256s": kwargs[
                "expected_request_receipt_sha256s"
            ],
            "replay_validation_sha256": _h("content replay"),
        }

    monkeypatch.setattr(
        verifier_module, "validate_detached_stage_content_replay", replay
    )
    receipt = validate_detached_stage_content_evidence(
        evidence,
        expected_stage="development",
        candidate_manifest=candidate,
        corpus_universe_manifest=universe,
        externally_pinned_content_manifest_sha256=evidence[
            "expected_content_manifest_sha256"
        ],
        externally_pinned_stage_artifact_sha256=evidence[
            "expected_stage_artifact_sha256"
        ],
        maximum_total_document_bytes=1024,
    )
    assert receipt["replay_validation_sha256"] == _h("content replay")
    assert receipt["schema_version"] == (
        verifier_module.DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION
    )

    crossed = copy.deepcopy(evidence)
    crossed["stage"] = "intermediate"
    with pytest.raises(SecFilingGemmaStageVerifierError, match="stage slot"):
        validate_detached_stage_content_evidence(
            crossed,
            expected_stage="development",
            candidate_manifest=candidate,
            corpus_universe_manifest=universe,
            externally_pinned_content_manifest_sha256=evidence[
                "expected_content_manifest_sha256"
            ],
            externally_pinned_stage_artifact_sha256=evidence[
                "expected_stage_artifact_sha256"
            ],
            maximum_total_document_bytes=1024,
        )

    substituted = copy.deepcopy(evidence)
    substituted["content_manifest"]["content_manifest_sha256"] = _h("other")
    with pytest.raises(SecFilingGemmaStageVerifierError, match="binding"):
        validate_detached_stage_content_evidence(
            substituted,
            expected_stage="development",
            candidate_manifest=candidate,
            corpus_universe_manifest=universe,
            externally_pinned_content_manifest_sha256=evidence[
                "expected_content_manifest_sha256"
            ],
            externally_pinned_stage_artifact_sha256=evidence[
                "expected_stage_artifact_sha256"
            ],
            maximum_total_document_bytes=1024,
        )

    forged_boundary = replay(
        authorized_stage="development",
        expected_corpus_universe_sha256=candidate["bindings"][
            "corpus_universe_sha256"
        ],
        expected_content_manifest_sha256=evidence[
            "expected_content_manifest_sha256"
        ],
        expected_stage_artifact_sha256=evidence[
            "expected_stage_artifact_sha256"
        ],
        expected_request_receipts_sha256=evidence[
            "expected_request_receipts_sha256"
        ],
        expected_document_sha256s=evidence["expected_document_sha256s"],
        expected_normalized_text_sha256s=evidence[
            "expected_normalized_text_sha256s"
        ],
        expected_request_receipt_sha256s=evidence[
            "expected_request_receipt_sha256s"
        ],
    )
    forged_boundary["fresh_network_provenance_verified"] = True
    monkeypatch.setattr(
        verifier_module,
        "validate_detached_stage_content_replay",
        lambda **kwargs: forged_boundary,
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="trust boundary"):
        validate_detached_stage_content_evidence(
            evidence,
            expected_stage="development",
            candidate_manifest=candidate,
            corpus_universe_manifest=universe,
            externally_pinned_content_manifest_sha256=evidence[
                "expected_content_manifest_sha256"
            ],
            externally_pinned_stage_artifact_sha256=evidence[
                "expected_stage_artifact_sha256"
            ],
            maximum_total_document_bytes=1024,
        )

    with pytest.raises(SecFilingGemmaStageVerifierError, match="pre-existing trusted pins"):
        validate_detached_stage_content_evidence(
            evidence,
            expected_stage="development",
            candidate_manifest=candidate,
            corpus_universe_manifest=universe,
            externally_pinned_content_manifest_sha256=_h("substituted external pin"),
            externally_pinned_stage_artifact_sha256=evidence[
                "expected_stage_artifact_sha256"
            ],
            maximum_total_document_bytes=1024,
        )


def test_trusted_content_pin_must_match_preexisting_stage_access() -> None:
    access, trusted = _trusted_content_pin("development")
    summary = validate_trusted_stage_content_pin(
        stage_access_manifest=access,
        trusted_stage_content_pin=trusted,
        expected_stage="development",
    )
    assert summary["pin_sha256"] == trusted["pin_sha256"]

    substituted = copy.deepcopy(trusted)
    substituted["stage_artifact_sha256"] = _h("same-envelope artifact substitution")
    substituted_body = {
        key: value for key, value in substituted.items() if key != "pin_sha256"
    }
    substituted["pin_sha256"] = canonical_sha256(substituted_body)
    with pytest.raises(SecFilingGemmaStageVerifierError, match="stage_artifact_sha256"):
        validate_trusted_stage_content_pin(
            stage_access_manifest=access,
            trusted_stage_content_pin=substituted,
            expected_stage="development",
        )


def _complete_calendar_universe_fixture() -> tuple[dict, dict, dict, dict[str, str]]:
    calendar_payloads = {
        name: f"official fixture bytes {name}".encode("utf-8")
        for name in CALENDAR_SOURCE_URLS
    }
    calendar = build_calendar_source_evidence_manifest(
        retrieved_at_utc="2026-07-11T10:00:00Z",
        source_records={
            name: {
                "url": CALENDAR_SOURCE_URLS[name],
                "content_sha256": hashlib.sha256(payload).hexdigest(),
                "byte_count": len(payload),
            }
            for name, payload in calendar_payloads.items()
        },
    )
    records: list[dict] = []
    serial = 1
    for year in range(2000, 2026):
        for form, month in (("10-K", 2), ("10-Q", 5), ("10-Q", 8), ("10-Q", 11)):
            evidence_date = date(year, month, 15)
            records.append(
                {
                    "accession_number": f"0000320193-{year % 100:02d}-{serial:06d}",
                    "subject_cik": "0000320193",
                    "form": form,
                    "acceptance_datetime": evidence_date.strftime("%Y%m%d") + "160000",
                    "filing_date": evidence_date.isoformat(),
                    "filing_date_change": None,
                    "primary_document": f"filing-{serial}.htm",
                    "source_record_sha256": f"{serial + 70_000:064x}",
                }
            )
            serial += 1
    for form, month in (("10-Q", 2), ("10-Q", 5)):
        evidence_date = date(2026, month, 15)
        records.append(
            {
                "accession_number": f"0000320193-26-{serial:06d}",
                "subject_cik": "0000320193",
                "form": form,
                "acceptance_datetime": evidence_date.strftime("%Y%m%d") + "160000",
                "filing_date": evidence_date.isoformat(),
                "filing_date_change": None,
                "primary_document": f"filing-{serial}.htm",
                "source_record_sha256": f"{serial + 70_000:064x}",
            }
        )
        serial += 1
    universe = build_corpus_universe_manifest(
        catalog_artifact_sha256=_h("complete catalog"),
        calendar_artifact_sha256=calendar["calendar_source_evidence_sha256"],
        catalog_total_record_count=1_000,
        catalog_eligible_record_count=len(records),
        session_dates=EXPECTED_SESSIONS,
        records=records,
    )
    _, encoded_sources = _candidate_fixture()
    source_hashes = {
        role: hashlib.sha256(base64.b64decode(encoded)).hexdigest()
        for role, encoded in encoded_sources.items()
    }
    candidate = build_candidate_manifest(
        model_digest=_h("model"),
        ollama_runtime_fingerprint_sha256=_h("runtime"),
        sec_audit_checksums_json_sha256=_h("audit"),
        sec_catalog_artifact_sha256=universe["catalog_artifact_sha256"],
        sec_audit_source_commit=_h("audit commit"),
        calendar_source_evidence_sha256=calendar[
            "calendar_source_evidence_sha256"
        ],
        calendar_sessions_sha256=universe["calendar_sessions_sha256"],
        corpus_universe_sha256=universe["universe_sha256"],
        corpus_universe_semantic_sha256=universe["universe_semantic_sha256"],
        identity_lexicon_sha256=_h("identity lexicon"),
        predecessor_reveal_registry_sha256=_h("predecessor registry"),
        holdout_attempt_id=f"{CONTRACT_VERSION}-attempt-001",
        experiment_source_commit=_h("experiment commit"),
        source_tree_sha256=_h("source tree"),
        source_hashes=source_hashes,
    )
    calendar_bytes = {name: _b64(payload) for name, payload in calendar_payloads.items()}
    return calendar, universe, candidate, calendar_bytes


def test_calendar_bytes_and_complete_universe_are_candidate_bound() -> None:
    calendar, universe, candidate, calendar_bytes = _complete_calendar_universe_fixture()
    result = validate_calendar_and_universe_snapshot(
        calendar_evidence_manifest=calendar,
        calendar_source_bytes_base64_by_name=calendar_bytes,
        corpus_universe_manifest=universe,
        candidate_manifest=candidate,
    )
    assert result["calendar_source_evidence_sha256"] == calendar[
        "calendar_source_evidence_sha256"
    ]
    assert result["corpus_universe_sha256"] == universe["universe_sha256"]

    changed = copy.deepcopy(calendar_bytes)
    changed["nyse_hours_calendars"] = _b64(b"substituted calendar")
    with pytest.raises(SecFilingGemmaStageVerifierError, match="do not match"):
        validate_calendar_and_universe_snapshot(
            calendar_evidence_manifest=calendar,
            calendar_source_bytes_base64_by_name=changed,
            corpus_universe_manifest=universe,
            candidate_manifest=candidate,
        )

def _learner_replays() -> tuple[str, dict, list[dict]]:
    candidate_hash = _h("learner candidate")
    semantic_names = ["x", "y"]
    semantic_training = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 0.5],
        [0.5, 2.0],
    ]
    ablation_names = ["x"]
    ablation_training = [[row[0]] for row in semantic_training]
    binary = [0, 1, 0, 1, 1, 0]
    edges = [-0.2, 0.1, -0.1, 0.2, 0.3, -0.3]
    edge_targets_hex = [float(value).hex() for value in edges]
    membership_hash = _h("training membership")
    encoded_features = {
        "semantic": [
            [float(value).hex() for value in values]
            for values in semantic_training
        ],
        "ablation": [
            [float(value).hex() for value in values]
            for values in ablation_training
        ],
    }
    row_hash = _h("prediction row")
    row = {
        "prediction_row_sha256": row_hash,
        "fold_id": "fold_1",
        "prediction_status": AVAILABLE_PREDICTION_STATUS,
        "fold_context": {
            "fold_train_cutoff_session": "2004-12-31",
            "training_set_count": len(binary),
            "training_positive_count": sum(binary),
            "training_set_membership_sha256": membership_hash,
            "semantic_training_feature_matrix_sha256": canonical_sha256(
                encoded_features["semantic"]
            ),
            "ablation_training_feature_matrix_sha256": canonical_sha256(
                encoded_features["ablation"]
            ),
            "training_binary_target_sha256": canonical_sha256(binary),
            "training_edge_target_sha256": canonical_sha256(edge_targets_hex),
            "training_set_max_label_maturity_session": "2004-12-30",
        },
    }
    replays: list[dict] = []
    for variant in ("semantic", "ablation"):
        names = semantic_names if variant == "semantic" else ablation_names
        training = semantic_training if variant == "semantic" else ablation_training
        prediction = [[0.75, 0.25]] if variant == "semantic" else [[0.75]]
        metadata = {
            "candidate_sha256": candidate_hash,
            "head_variant": variant,
            "fold_id": "fold_1",
            "train_label_maturity_through": "2004-12-31",
            "training_set_sha256": membership_hash,
            "training_row_count": len(training),
            "maximum_training_label_maturity_session": "2004-12-30",
            "feature_schema_sha256": canonical_sha256(names),
        }
        learner = SecFilingGemmaTwoHeadLearner().fit(
            training,
            binary,
            edges,
            feature_names=names,
            fit_metadata=metadata,
        )
        output = learner.predict_components(prediction)
        row[f"{variant}_cash_probability_hex"] = float(
            output["cash_win_probability_10bps"][0]
        ).hex()
        row[f"{variant}_expected_edge_hex"] = float(
            output["expected_active_log_edge_10bps"][0]
        ).hex()
        row["fold_context"][f"{variant}_fold_state_sha256"] = learner.model_sha256
        replays.append(
            {
                "fold_id": "fold_1",
                "variant": variant,
                "feature_names": names,
                "training_features_hex": encoded_features[variant],
                "binary_targets": binary,
                "edge_targets_hex": edge_targets_hex,
                "fit_metadata": metadata,
                "expected_state": learner.to_state(),
                "prediction_cases": [
                    {
                        "prediction_row_sha256": row_hash,
                        "features_hex": [float(value).hex() for value in prediction[0]],
                    }
                ],
            }
        )
    return candidate_hash, {"rows": [row]}, replays


def test_deterministic_learner_refit_state_and_prediction_replay() -> None:
    candidate_hash, prefix, replays = _learner_replays()
    result = validate_learner_refit_replays(
        replays,
        candidate_sha256=candidate_hash,
        prediction_prefix=prefix,
    )
    assert result["prediction_case_count"] == 2
    assert set(result["learner_state_sha256s_by_fold"]["fold_1"]) == {
        "semantic",
        "ablation",
    }
    assert set(result["learner_input_identity_sha256s_by_fold"]["fold_1"]) == {
        "semantic",
        "ablation",
    }


def test_learner_state_mutation_case_omission_and_cross_binding_are_rejected() -> None:
    candidate_hash, prefix, replays = _learner_replays()
    mutated = copy.deepcopy(replays)
    mutated[0]["expected_state"]["training_positive_count"] += 1
    with pytest.raises(SecFilingGemmaStageVerifierError, match="state"):
        validate_learner_refit_replays(
            mutated, candidate_sha256=candidate_hash, prediction_prefix=prefix
        )

    omitted = copy.deepcopy(replays)
    omitted[1]["prediction_cases"] = []
    with pytest.raises(SecFilingGemmaStageVerifierError, match="omit"):
        validate_learner_refit_replays(
            omitted, candidate_sha256=candidate_hash, prediction_prefix=prefix
        )

    crossed = copy.deepcopy(replays)
    crossed[0]["fit_metadata"]["candidate_sha256"] = _h("other")
    with pytest.raises(SecFilingGemmaStageVerifierError, match="binding"):
        validate_learner_refit_replays(
            crossed, candidate_sha256=candidate_hash, prediction_prefix=prefix
        )

    duplicated = copy.deepcopy(replays)
    extra = copy.deepcopy(duplicated[0])
    extra["prediction_cases"] = []
    duplicated.append(extra)
    with pytest.raises(SecFilingGemmaStageVerifierError, match="duplicated"):
        validate_learner_refit_replays(
            duplicated, candidate_sha256=candidate_hash, prediction_prefix=prefix
        )


def test_learner_refit_inputs_must_match_the_prediction_fold_context() -> None:
    candidate_hash, prefix, replays = _learner_replays()

    changed_matrix = copy.deepcopy(replays)
    changed_matrix[0]["training_features_hex"][0][0] = float(0.25).hex()
    with pytest.raises(SecFilingGemmaStageVerifierError, match="feature matrix"):
        validate_learner_refit_replays(
            changed_matrix,
            candidate_sha256=candidate_hash,
            prediction_prefix=prefix,
        )

    changed_membership = copy.deepcopy(replays)
    changed_membership[0]["fit_metadata"]["training_set_sha256"] = _h(
        "another membership"
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="membership"):
        validate_learner_refit_replays(
            changed_membership,
            candidate_sha256=candidate_hash,
            prediction_prefix=prefix,
        )

    changed_targets = copy.deepcopy(replays)
    changed_targets[0]["binary_targets"][0] = 1
    with pytest.raises(SecFilingGemmaStageVerifierError, match="targets"):
        validate_learner_refit_replays(
            changed_targets,
            candidate_sha256=candidate_hash,
            prediction_prefix=prefix,
        )

    changed_counts = copy.deepcopy(prefix)
    changed_counts["rows"][0]["fold_context"]["training_set_count"] += 1
    with pytest.raises(SecFilingGemmaStageVerifierError, match="counts"):
        validate_learner_refit_replays(
            replays,
            candidate_sha256=candidate_hash,
            prediction_prefix=changed_counts,
        )

    changed_maturity = copy.deepcopy(replays)
    changed_maturity[0]["fit_metadata"]["train_label_maturity_through"] = (
        "2004-12-29"
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="maturity"):
        validate_learner_refit_replays(
            changed_maturity,
            candidate_sha256=candidate_hash,
            prediction_prefix=prefix,
        )


def test_model_batch_hardcodes_owned_transport_and_exact_guard_order(monkeypatch) -> None:
    candidate, _ = _candidate_fixture()
    observed: dict[str, object] = {}

    def fake_attempt(manifest, **kwargs):
        observed.setdefault("modes", []).append(kwargs["expected_transport_mode"])
        observed.setdefault("candidates", []).append(kwargs["expected_candidate_sha256"])
        payload = manifest["payload"]
        return SimpleNamespace(
            request_bytes=json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            receipt_sha256=manifest["receipt_sha256"],
            elapsed_nanoseconds=7,
        )

    def fake_guard(value, **kwargs):
        observed["guard_order"] = kwargs["model_call_receipt_sha256s"]
        observed["guard_stage"] = kwargs["stage"]
        return value["runtime_guard_sha256"]

    monkeypatch.setattr(verifier_module, "validate_ollama_model_attempt_receipt", fake_attempt)
    monkeypatch.setattr(verifier_module, "validate_runtime_identity_guard", fake_guard)
    accessions = ["0000320193-05-000001", "0000320193-05-000002"]
    attempts = []
    for index, accession in enumerate(accessions):
        payload = {"messages": [index]}
        attempts.append(
            {
                "accession_number": accession,
                "expected_model_payload": payload,
                "expected_sentence_ids": ["C0001"],
                "receipt": {
                    "payload": payload,
                    "receipt_sha256": _h(f"receipt {index}"),
                },
            }
        )
    batch = {
        "stage": "development",
        "before_runtime_evidence": {"probe": "same"},
        "after_runtime_evidence": {"probe": "same"},
        "runtime_guard": {"runtime_guard_sha256": _h("guard")},
        "attempts": attempts,
    }
    result = validate_model_attempt_batch(
        batch, candidate_manifest=candidate, expected_accessions=accessions
    )
    assert observed["modes"] == [OWNED_HARDENED_TRANSPORT_MODE] * 2
    assert observed["guard_order"] == result["model_attempt_receipt_sha256s"]
    assert observed["guard_stage"] == "development"

    reordered = copy.deepcopy(batch)
    reordered["attempts"].reverse()
    with pytest.raises(SecFilingGemmaStageVerifierError, match="reordered"):
        validate_model_attempt_batch(
            reordered, candidate_manifest=candidate, expected_accessions=accessions
        )


def _score_replay() -> tuple[dict, dict, dict]:
    prefix = {"prediction_prefix_sha256": _h("prefix")}
    market = {"market_stage_manifest_sha256": _h("market")}
    receipt = {
        "configuration": {
            "selected_candidate_id": "p50_e0",
            "selected_variant": "semantic",
            "cost_bps": 10,
            "score_cutoff_session": "2023-12-29",
            "terminal_convention": "adjusted_open",
        },
        "score_receipt_sha256": _h("score"),
    }
    gate = {"gate_receipt_sha256": _h("gate"), "passed": True}
    replay = {
        "label_release_evidence": {
            "label_release_ledger_sha256": _h("labels")
        },
        "evaluations": [
            {
                "candidate_id": "p50_e0",
                "score_receipts": [receipt],
                "gate_receipt": gate,
            }
        ],
        "ranking_receipt": None,
        "selected_candidate_id": "p50_e0",
    }
    return replay, prefix, market


def test_raw_score_and_no_leverage_replay_precede_gate(monkeypatch) -> None:
    replay, prefix, market = _score_replay()
    calls: list[str] = []

    monkeypatch.setattr(
        verifier_module,
        "validate_score_receipt",
        lambda *args, **kwargs: calls.append("score") or kwargs[
            "expected_score_receipt_sha256"
        ],
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_sec_gemma_no_leverage_proof",
        lambda *args, **kwargs: calls.append("no_leverage")
        or {"proof_sha256": _h("proof")},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_stage_gate_receipt",
        lambda *args, **kwargs: calls.append("gate")
        or kwargs["expected_gate_receipt_sha256"],
    )
    result = validate_raw_scores_gates_and_ranking(
        replay,
        prediction_prefix=prefix,
        market_stage=market,
        prerequisite_stage="intermediate",
    )
    assert calls == ["score", "no_leverage", "gate"]
    assert result["no_leverage_proof_sha256s"] == [_h("proof")]


def test_score_cross_binding_and_raw_failure_never_reach_gate(monkeypatch) -> None:
    replay, prefix, market = _score_replay()
    replay["evaluations"][0]["score_receipts"][0]["configuration"][
        "selected_candidate_id"
    ] = "p55_e0"
    with pytest.raises(SecFilingGemmaStageVerifierError, match="crossed"):
        validate_raw_scores_gates_and_ranking(
            replay,
            prediction_prefix=prefix,
            market_stage=market,
            prerequisite_stage="intermediate",
        )

    replay, prefix, market = _score_replay()
    called_gate = False

    def fail_score(*args, **kwargs):
        raise ValueError("raw mismatch")

    def gate(*args, **kwargs):
        nonlocal called_gate
        called_gate = True

    monkeypatch.setattr(verifier_module, "validate_score_receipt", fail_score)
    monkeypatch.setattr(verifier_module, "validate_stage_gate_receipt", gate)
    with pytest.raises(SecFilingGemmaStageVerifierError, match="Raw score"):
        validate_raw_scores_gates_and_ranking(
            replay,
            prediction_prefix=prefix,
            market_stage=market,
            prerequisite_stage="intermediate",
        )
    assert called_gate is False


def _runtime_receipt(candidate_hash: str, model_summary: dict) -> dict:
    body = {
        "schema_version": STAGE_RUNTIME_RECEIPT_SCHEMA_VERSION,
        "stage": "development",
        "candidate_sha256": candidate_hash,
        "phase_elapsed_seconds": {
            "sec": 1.0.hex(),
            "model": 2.0.hex(),
            "fit_simulation_sealing": 3.0.hex(),
            "total": 6.0.hex(),
        },
        "sec_request_count": 4,
        "sec_response_bytes": 100,
        "model_attempt_receipt_sha256s": model_summary[
            "model_attempt_receipt_sha256s"
        ],
        "runtime_guard_sha256": model_summary["runtime_guard_sha256"],
        "paid_api_calls": 0,
        "estimated_cost_usd_hex": 0.0.hex(),
        "pull_attempts": 0,
        "retries": 0,
        "repair_attempts": 0,
    }
    return {**body, "runtime_receipt_sha256": canonical_sha256(body)}


def test_stage_specific_runtime_receipt_is_exact_zero_cost_and_bound() -> None:
    candidate_hash = _h("candidate")
    model = {
        "model_attempt_receipt_sha256s": [_h("one"), _h("two")],
        "runtime_guard_sha256": _h("guard"),
        "model_elapsed_nanoseconds": 1_000_000_000,
    }
    receipt = _runtime_receipt(candidate_hash, model)
    assert validate_stage_runtime_receipt(
        receipt,
        prerequisite_stage="development",
        candidate_sha256=candidate_hash,
        model_batch_summary=model,
    ) == receipt["runtime_receipt_sha256"]

    paid = copy.deepcopy(receipt)
    paid["paid_api_calls"] = 1
    paid["runtime_receipt_sha256"] = canonical_sha256(
        {key: paid[key] for key in paid if key != "runtime_receipt_sha256"}
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="paid_api_calls"):
        validate_stage_runtime_receipt(
            paid,
            prerequisite_stage="development",
            candidate_sha256=candidate_hash,
            model_batch_summary=model,
        )

    understated = copy.deepcopy(model)
    understated["model_elapsed_nanoseconds"] = 2_100_000_000
    with pytest.raises(SecFilingGemmaStageVerifierError, match="shorter"):
        validate_stage_runtime_receipt(
            receipt,
            prerequisite_stage="development",
            candidate_sha256=candidate_hash,
            model_batch_summary=understated,
        )


def _minimal_audit_envelope(candidate: dict, sources: dict[str, str]) -> dict:
    content_manifest = {"content_manifest_sha256": _h("content")}
    body = {
        "schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "prerequisite_stage": "development",
        "parent_stage_evidence_sha256": None,
        "parent_stage_lineage": None,
        "contract_manifest": build_contract_manifest(),
        "candidate_manifest": candidate,
        "source_bytes_base64_by_role": sources,
        "calendar_evidence_manifest": {},
        "calendar_source_bytes_base64_by_name": {},
        "corpus_universe_manifest": {
            "records": [
                {
                    "accession_number": "0000320193-05-000001",
                    "artifact_stage": "development",
                    "availability_session": "2005-01-03",
                }
            ]
        },
        "catalog_replay": {},
        "content_replays_by_stage": {
            "development": {
                "stage": "development",
                "content_manifest": content_manifest,
            }
        },
        "prerequisite_content_manifest": content_manifest,
        "model_batches_by_stage": {"development": {}},
        "market_replays_by_stage": {
            "development": {"market_stage_manifest": {}}
        },
        "prediction_replay": {},
        "learner_replays": [],
        "score_replay": {},
        "stage_runtime_receipt": {},
        "reveal_registry": {},
        "registry_external_pin": {},
    }
    return {**body, "stage_evidence_sha256": canonical_sha256(body)}


def test_audit_lists_every_required_check_and_never_claims_authorization(monkeypatch) -> None:
    candidate, sources = _candidate_fixture()
    evidence = _minimal_audit_envelope(candidate, sources)
    stage_access, trusted_pin = _trusted_content_pin(
        "development", content_hash=_h("content")
    )
    candidate_hash = candidate["candidate_sha256"]
    monkeypatch.setattr(
        verifier_module,
        "validate_candidate_source_bytes",
        lambda **kwargs: {
            "candidate_sha256": candidate_hash,
            "stage_verifier_source_sha256": candidate["bindings"]["source_hashes"][
                "stage_verifier"
            ],
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_candidate_source_role_audit",
        lambda **kwargs: {
            "source_identity_receipt_sha256": _h("source identity")
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_candidate_runtime_source_audit",
        lambda **kwargs: {
            "source_identity_receipt_sha256": _h("runtime source identity"),
            "complete": False,
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_calendar_and_universe_snapshot",
        lambda **kwargs: {"corpus_universe_sha256": _h("universe")},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_detached_catalog_evidence",
        lambda *args, **kwargs: {
            **_detached_boundary(),
            "schema_version": verifier_module.DETACHED_CATALOG_REPLAY_RECEIPT_SCHEMA_VERSION,
            "corpus_universe_sha256": _h("universe"),
            "replay_validation_sha256": _h("catalog replay"),
            "request_receipts_sha256": _h("catalog receipts"),
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_detached_stage_content_evidence",
        lambda value, expected_stage, **kwargs: {
            **_detached_boundary(),
            "schema_version": verifier_module.DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION,
            "artifact_stage": expected_stage,
            "content_manifest_sha256": value["content_manifest"][
                "content_manifest_sha256"
            ],
            "replay_validation_sha256": _h(f"{expected_stage} content replay"),
            "request_receipts_sha256": _h(f"{expected_stage} content receipts"),
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_model_attempt_batch",
        lambda *args, **kwargs: {
            "stage": "development",
            "model_attempt_receipt_sha256s": [_h("attempt")],
            "runtime_guard_sha256": _h("guard"),
            "model_elapsed_nanoseconds": 1,
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_market_snapshot_stage_replay",
        lambda *args, **kwargs: {"artifact_stage": "development"},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_prediction_artifact_replay",
        lambda *args, **kwargs: {"stage": "development", "prefix": {}},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_learner_refit_replays",
        lambda *args, **kwargs: {"learner_replay_sha256": _h("learner")},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_raw_scores_gates_and_ranking",
        lambda *args, **kwargs: {"selected_candidate_id": "p50_e0"},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_stage_runtime_receipt",
        lambda *args, **kwargs: _h("runtime"),
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_registry_request_and_stage_access",
        lambda **kwargs: {"request_sha256": _h("request")},
    )
    receipt = audit_stage_evidence(
        evidence,
        stage_access,
        {},
        trusted_stage_content_pin=trusted_pin,
    )
    assert receipt["semantic_checks"] == list(REQUIRED_STAGE_VERIFIER_CHECKS)
    assert set(receipt["check_status"]) == set(REQUIRED_STAGE_VERIFIER_CHECKS)
    assert receipt["check_status"]["artifact_seal_cas"]["status"] == "blocked"
    assert receipt["check_status"]["no_leverage"]["status"] == "replayed"
    assert receipt["check_status"]["artifact_replay"]["status"] == "blocked"
    assert receipt["check_status"]["prediction_replay"]["status"] == "blocked"
    assert receipt["check_status"]["stage_access_identity"]["status"] == "blocked"
    assert receipt["check_status"]["stage_identity"]["status"] == "replayed"
    assert receipt["blocking_check_count"] > 0
    assert receipt["all_checks_completed"] is False
    assert receipt["authorizes_outcome_access"] is False
    assert receipt["source_identity_receipt_sha256"] == _h("source identity")
    assert receipt["source_runtime_trust_boundary"] == {
        "candidate_pinned_detached_source_bytes_verified": True,
        "current_files_at_loaded_module_paths_verified": True,
        "executing_validator_source_bytes_attested": False,
        "runtime_module_paths_verified": True,
        "caller_supplied_runtime_paths_or_bytes_accepted": False,
        "role_ownership_complete": False,
        "identity_scope": "current_files_at_already_loaded_module_paths",
    }
    assert "current_file_at_loaded_module_path" in receipt[
        "validator_source_sha256_role"
    ]
    assert receipt["trusted_content_pin_boundary"] == {
        "stage_access_pin_cross_bound": True,
        "separately_supplied_store_pin_claim_present": True,
        "separately_supplied_store_pin_authenticated": False,
        "trusted_store_state_authenticated_by_verifier": False,
        "authorizing": False,
    }
    assert receipt["trusted_stage_content_pin_sha256"] == trusted_pin["pin_sha256"]
    assert receipt["catalog_replay_receipt_sha256"] == _h("catalog replay")
    assert receipt["content_replay_receipt_sha256s_by_stage"] == {
        "development": _h("development content replay")
    }
    assert receipt["corpus_replay_trust_boundary"] == {
        "catalog": _detached_boundary(),
        "content_by_stage": {"development": _detached_boundary()},
    }
    assert receipt["corpus_replay_receipt_schema_versions"] == {
        "catalog": verifier_module.DETACHED_CATALOG_REPLAY_RECEIPT_SCHEMA_VERSION,
        "content_by_stage": {
            "development": verifier_module.DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION
        },
    }

    monkeypatch.setattr(
        verifier_module,
        "validate_model_attempt_batch",
        lambda *args, **kwargs: {
            "stage": "intermediate",
            "model_attempt_receipt_sha256s": [_h("attempt")],
            "runtime_guard_sha256": _h("guard"),
            "model_elapsed_nanoseconds": 1,
        },
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="stage slot"):
        audit_stage_evidence(
            evidence,
            stage_access,
            {},
            trusted_stage_content_pin=trusted_pin,
        )

    evidence["content_replays_by_stage"] = {}
    evidence_body = {
        key: evidence[key] for key in evidence if key != "stage_evidence_sha256"
    }
    evidence["stage_evidence_sha256"] = canonical_sha256(evidence_body)
    with pytest.raises(SecFilingGemmaStageVerifierError, match="content replay stage coverage"):
        audit_stage_evidence(
            evidence,
            stage_access,
            {},
            trusted_stage_content_pin=trusted_pin,
        )


def test_intermediate_stage_rejects_an_opaque_parent_hash() -> None:
    candidate, sources = _candidate_fixture()
    evidence = _minimal_audit_envelope(candidate, sources)
    evidence["prerequisite_stage"] = "intermediate"
    evidence["parent_stage_evidence_sha256"] = _h("opaque parent")
    evidence_body = {
        key: evidence[key] for key in evidence if key != "stage_evidence_sha256"
    }
    evidence["stage_evidence_sha256"] = canonical_sha256(evidence_body)
    with pytest.raises(SecFilingGemmaStageVerifierError, match="complete parent lineage"):
        audit_stage_evidence(evidence, {}, {})


def test_parent_lineage_replays_actual_evidence_and_exact_receipt(monkeypatch) -> None:
    candidate, sources = _candidate_fixture()
    parent = _minimal_audit_envelope(candidate, sources)
    parent_hash = parent["stage_evidence_sha256"]
    audit_body = {
        "stage_evidence_sha256": parent_hash,
        "candidate_sha256": candidate["candidate_sha256"],
        "prerequisite_stage": "development",
        "requested_stage": "intermediate",
        "authorizes_outcome_access": False,
        "trusted_stage_content_pin_sha256": _h("parent trusted pin"),
        "trusted_content_manifest_sha256": _h("content"),
        "trusted_stage_artifact_sha256": _h("development artifact"),
        "trusted_external_seal_receipt_sha256": _h("development seal"),
        "trusted_store_state_sha256": _h("development store"),
    }
    computed = {
        **audit_body,
        "audit_receipt_sha256": canonical_sha256(audit_body),
    }
    monkeypatch.setattr(
        verifier_module,
        "audit_stage_evidence",
        lambda *args, **kwargs: copy.deepcopy(computed),
    )
    lineage = {
        "evidence": parent,
        "stage_access_manifest": {"parent": "access"},
        "expected_context": {"parent": "context"},
        "trusted_stage_content_pin": {"parent": "pin"},
        "audit_receipt": copy.deepcopy(computed),
    }
    summary = validate_parent_stage_lineage(
        lineage,
        prerequisite_stage="intermediate",
        declared_parent_stage_evidence_sha256=parent_hash,
        current_candidate_sha256=candidate["candidate_sha256"],
    )
    assert summary["parent_stage_evidence_sha256"] == parent_hash

    substituted = copy.deepcopy(lineage)
    substituted["audit_receipt"]["trusted_content_manifest_sha256"] = _h(
        "substituted parent receipt"
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="authoritative parent replay"):
        validate_parent_stage_lineage(
            substituted,
            prerequisite_stage="intermediate",
            declared_parent_stage_evidence_sha256=parent_hash,
            current_candidate_sha256=candidate["candidate_sha256"],
        )


def test_real_recursive_parent_lineage_replays_nested_v3_audit(monkeypatch) -> None:
    candidate, sources = _candidate_fixture()
    candidate_hash = candidate["candidate_sha256"]
    parent = _minimal_audit_envelope(candidate, sources)
    parent["model_batches_by_stage"] = {
        "development": {"stage": "development"}
    }
    parent["market_replays_by_stage"] = {
        "development": {
            "stage": "development",
            "market_stage_manifest": {},
        }
    }
    parent["prediction_replay"] = {"stage": "development"}
    parent_body = {
        key: parent[key] for key in parent if key != "stage_evidence_sha256"
    }
    parent["stage_evidence_sha256"] = canonical_sha256(parent_body)
    parent_access, parent_pin = _trusted_content_pin(
        "development", content_hash=_h("content")
    )

    monkeypatch.setattr(
        verifier_module,
        "validate_candidate_source_bytes",
        lambda **kwargs: {
            "candidate_sha256": candidate_hash,
            "stage_verifier_source_sha256": candidate["bindings"][
                "source_hashes"
            ]["stage_verifier"],
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_candidate_source_role_audit",
        lambda **kwargs: {
            "source_identity_receipt_sha256": _h("source identity")
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_candidate_runtime_source_audit",
        lambda **kwargs: {
            "source_identity_receipt_sha256": _h("runtime source identity"),
            "complete": False,
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_calendar_and_universe_snapshot",
        lambda **kwargs: {"corpus_universe_sha256": _h("universe")},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_detached_catalog_evidence",
        lambda *args, **kwargs: {
            **_detached_boundary(),
            "schema_version": (
                verifier_module.DETACHED_CATALOG_REPLAY_RECEIPT_SCHEMA_VERSION
            ),
            "corpus_universe_sha256": _h("universe"),
            "replay_validation_sha256": _h("catalog replay"),
            "request_receipts_sha256": _h("catalog receipts"),
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_detached_stage_content_evidence",
        lambda value, expected_stage, **kwargs: {
            **_detached_boundary(),
            "schema_version": (
                verifier_module.DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION
            ),
            "artifact_stage": expected_stage,
            "content_manifest_sha256": value["content_manifest"][
                "content_manifest_sha256"
            ],
            "replay_validation_sha256": _h(
                f"{expected_stage} content replay"
            ),
            "request_receipts_sha256": _h(
                f"{expected_stage} content receipts"
            ),
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_model_attempt_batch",
        lambda batch, **kwargs: {
            "stage": batch["stage"],
            "model_attempt_receipt_sha256s": [_h(f"{batch['stage']} attempt")],
            "runtime_guard_sha256": _h(f"{batch['stage']} guard"),
            "model_elapsed_nanoseconds": 1,
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_market_snapshot_stage_replay",
        lambda replay: {"artifact_stage": replay["stage"]},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_prediction_artifact_replay",
        lambda value, **kwargs: {"stage": value["stage"], "prefix": {}},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_learner_refit_replays",
        lambda *args, **kwargs: {"learner_replay_sha256": _h("learner")},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_raw_scores_gates_and_ranking",
        lambda *args, **kwargs: {"selected_candidate_id": "p50_e0"},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_stage_runtime_receipt",
        lambda *args, **kwargs: _h("runtime"),
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_registry_request_and_stage_access",
        lambda **kwargs: {"request_sha256": _h("request")},
    )

    parent_receipt = audit_stage_evidence(
        parent,
        parent_access,
        {},
        trusted_stage_content_pin=parent_pin,
    )
    child = _minimal_audit_envelope(candidate, sources)
    child["prerequisite_stage"] = "intermediate"
    child["parent_stage_evidence_sha256"] = parent[
        "stage_evidence_sha256"
    ]
    child["parent_stage_lineage"] = {
        "evidence": parent,
        "stage_access_manifest": parent_access,
        "expected_context": {},
        "trusted_stage_content_pin": parent_pin,
        "audit_receipt": parent_receipt,
    }
    intermediate_manifest = {
        "content_manifest_sha256": _h("intermediate content")
    }
    child["content_replays_by_stage"] = {
        "development": copy.deepcopy(parent["content_replays_by_stage"]["development"]),
        "intermediate": {
            "stage": "intermediate",
            "content_manifest": intermediate_manifest,
        },
    }
    child["prerequisite_content_manifest"] = intermediate_manifest
    child["model_batches_by_stage"] = {
        "development": {"stage": "development"},
        "intermediate": {"stage": "intermediate"},
    }
    child["market_replays_by_stage"] = {
        "development": {
            "stage": "development",
            "market_stage_manifest": {},
        },
        "intermediate": {
            "stage": "intermediate",
            "market_stage_manifest": {},
        },
    }
    child["prediction_replay"] = {"stage": "intermediate"}
    child_body = {
        key: child[key] for key in child if key != "stage_evidence_sha256"
    }
    child["stage_evidence_sha256"] = canonical_sha256(child_body)
    child_access, child_pin = _trusted_content_pin(
        "intermediate", content_hash=_h("intermediate content")
    )

    receipt = audit_stage_evidence(
        child,
        child_access,
        {},
        trusted_stage_content_pin=child_pin,
    )
    assert receipt["parent_stage_evidence_sha256"] == parent[
        "stage_evidence_sha256"
    ]
    assert receipt["parent_audit_receipt_sha256"] == parent_receipt[
        "audit_receipt_sha256"
    ]
    assert receipt["check_status"]["stage_identity"]["status"] == "blocked"

    substituted = copy.deepcopy(child)
    substituted["content_replays_by_stage"]["development"]["content_manifest"][
        "content_manifest_sha256"
    ] = _h("substituted development content")
    substituted_body = {
        key: substituted[key]
        for key in substituted
        if key != "stage_evidence_sha256"
    }
    substituted["stage_evidence_sha256"] = canonical_sha256(substituted_body)
    with pytest.raises(
        SecFilingGemmaStageVerifierError, match="Earlier-stage content"
    ):
        audit_stage_evidence(
            substituted,
            child_access,
            {},
            trusted_stage_content_pin=child_pin,
        )


def test_intermediate_stage_identity_stays_blocked_after_exact_parent_replay(
    monkeypatch,
) -> None:
    candidate, sources = _candidate_fixture()
    evidence = _minimal_audit_envelope(candidate, sources)
    evidence["prerequisite_stage"] = "intermediate"
    evidence["parent_stage_evidence_sha256"] = _h("development evidence")
    evidence["parent_stage_lineage"] = {"replayed": True}
    evidence["model_batches_by_stage"] = {
        "development": {"stage": "development"},
        "intermediate": {"stage": "intermediate"},
    }
    evidence["market_replays_by_stage"] = {
        "development": {"stage": "development", "market_stage_manifest": {}},
        "intermediate": {"stage": "intermediate", "market_stage_manifest": {}},
    }
    intermediate_content = {"content_manifest_sha256": _h("intermediate content")}
    evidence["content_replays_by_stage"] = {
        "development": {
            "stage": "development",
            "content_manifest": {"content_manifest_sha256": _h("content")},
        },
        "intermediate": {
            "stage": "intermediate",
            "content_manifest": intermediate_content,
        },
    }
    evidence["prerequisite_content_manifest"] = intermediate_content
    evidence_body = {
        key: evidence[key] for key in evidence if key != "stage_evidence_sha256"
    }
    evidence["stage_evidence_sha256"] = canonical_sha256(evidence_body)
    candidate_hash = candidate["candidate_sha256"]
    stage_access, trusted_pin = _trusted_content_pin(
        "intermediate", content_hash=_h("intermediate content")
    )
    parent_access, parent_pin = _trusted_content_pin(
        "development", content_hash=_h("content")
    )
    parent_content_pin = validate_trusted_stage_content_pin(
        stage_access_manifest=parent_access,
        trusted_stage_content_pin=parent_pin,
        expected_stage="development",
    )

    monkeypatch.setattr(
        verifier_module,
        "validate_parent_stage_lineage",
        lambda *args, **kwargs: {
            "parent_stage_evidence_sha256": _h("development evidence"),
            "parent_audit_receipt_sha256": _h("parent audit"),
            "parent_trusted_stage_content_pin_sha256": parent_pin["pin_sha256"],
            "trusted_stage_content_pin": parent_content_pin,
            "content_replays_by_stage": {
                "development": copy.deepcopy(
                    evidence["content_replays_by_stage"]["development"]
                )
            },
        },
    )

    monkeypatch.setattr(
        verifier_module,
        "validate_candidate_source_bytes",
        lambda **kwargs: {
            "candidate_sha256": candidate_hash,
            "stage_verifier_source_sha256": candidate["bindings"]["source_hashes"][
                "stage_verifier"
            ],
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_candidate_source_role_audit",
        lambda **kwargs: {
            "source_identity_receipt_sha256": _h("source identity")
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_candidate_runtime_source_audit",
        lambda **kwargs: {
            "source_identity_receipt_sha256": _h("runtime source identity"),
            "complete": False,
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_calendar_and_universe_snapshot",
        lambda **kwargs: {"corpus_universe_sha256": _h("universe")},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_detached_catalog_evidence",
        lambda *args, **kwargs: {
            **_detached_boundary(),
            "schema_version": verifier_module.DETACHED_CATALOG_REPLAY_RECEIPT_SCHEMA_VERSION,
            "corpus_universe_sha256": _h("universe"),
            "replay_validation_sha256": _h("catalog replay"),
            "request_receipts_sha256": _h("catalog receipts"),
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_detached_stage_content_evidence",
        lambda value, expected_stage, **kwargs: {
            **_detached_boundary(),
            "schema_version": verifier_module.DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION,
            "artifact_stage": expected_stage,
            "content_manifest_sha256": value["content_manifest"][
                "content_manifest_sha256"
            ],
            "replay_validation_sha256": _h(f"{expected_stage} content replay"),
            "request_receipts_sha256": _h(f"{expected_stage} content receipts"),
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_model_attempt_batch",
        lambda batch, **kwargs: {
            "stage": batch["stage"],
            "model_attempt_receipt_sha256s": [_h(f"{batch['stage']} attempt")],
            "runtime_guard_sha256": _h(f"{batch['stage']} guard"),
            "model_elapsed_nanoseconds": 1,
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_market_snapshot_stage_replay",
        lambda replay: {"artifact_stage": replay["stage"]},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_prediction_artifact_replay",
        lambda *args, **kwargs: {"stage": "intermediate", "prefix": {}},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_learner_refit_replays",
        lambda *args, **kwargs: {"learner_replay_sha256": _h("learner")},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_raw_scores_gates_and_ranking",
        lambda *args, **kwargs: {"selected_candidate_id": "p50_e0"},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_stage_runtime_receipt",
        lambda *args, **kwargs: _h("runtime"),
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_registry_request_and_stage_access",
        lambda **kwargs: {"request_sha256": _h("request")},
    )

    receipt = audit_stage_evidence(
        evidence,
        stage_access,
        {},
        trusted_stage_content_pin=trusted_pin,
    )
    stage_identity = receipt["check_status"]["stage_identity"]
    assert stage_identity["status"] == "blocked"
    assert "development winner" in stage_identity["reason"]
    assert "parent evidence and audit receipt are replayed" in stage_identity["reason"]
    assert "learner output state" in stage_identity["reason"]
    assert receipt["all_checks_completed"] is False
    assert receipt["authorizes_outcome_access"] is False

    substituted = copy.deepcopy(evidence)
    substituted["content_replays_by_stage"]["development"]["content_manifest"][
        "content_manifest_sha256"
    ] = _h("substituted earlier-stage content")
    substituted_body = {
        key: substituted[key]
        for key in substituted
        if key != "stage_evidence_sha256"
    }
    substituted["stage_evidence_sha256"] = canonical_sha256(substituted_body)
    with pytest.raises(SecFilingGemmaStageVerifierError, match="Earlier-stage content"):
        audit_stage_evidence(
            substituted,
            stage_access,
            {},
            trusted_stage_content_pin=trusted_pin,
        )


def test_authorizing_entrypoint_always_fails_closed(monkeypatch) -> None:
    audit = {
        "check_status": {
            check: {
                "status": "blocked" if check == "runtime_budget" else "replayed"
            }
            for check in REQUIRED_STAGE_VERIFIER_CHECKS
        }
    }
    monkeypatch.setattr(
        verifier_module, "audit_stage_evidence", lambda *args, **kwargs: audit
    )
    with pytest.raises(SecFilingGemmaStageVerifierBlocked, match="runtime_budget"):
        authoritative_prerequisite_validator({}, {}, {})


def test_stage_evidence_schema_omission_is_rejected_before_any_replay() -> None:
    candidate, sources = _candidate_fixture()
    evidence = _minimal_audit_envelope(candidate, sources)
    evidence.pop("learner_replays")
    with pytest.raises(SecFilingGemmaStageVerifierError, match="keys changed"):
        audit_stage_evidence(evidence, {}, {})


def test_base64_encoded_length_is_rejected_before_decoder_call(monkeypatch) -> None:
    monkeypatch.setattr(
        verifier_module.base64,
        "b64decode",
        lambda *args, **kwargs: pytest.fail("decoder must not be called"),
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="encoded length"):
        verifier_module._decode_base64(_b64(b"12345678"), "oversized", maximum=4)


def test_catalog_aggregate_bytes_are_preflighted_before_any_decode(monkeypatch) -> None:
    candidate, _ = _candidate_fixture()
    evidence, universe = _catalogue_wrapper_evidence(candidate)
    evidence["source_payloads"] = [
        {"name": "one.json", "payload_base64": _b64(b"12345678")},
        {"name": "two.json", "payload_base64": _b64(b"abcdefgh")},
    ]
    monkeypatch.setattr(verifier_module, "MAX_CATALOG_SOURCE_BYTES", 10)
    monkeypatch.setattr(
        verifier_module.base64,
        "b64decode",
        lambda *args, **kwargs: pytest.fail("decoder must not be called"),
    )
    monkeypatch.setattr(
        verifier_module,
        "_bounded_plain_json_copy",
        lambda *args, **kwargs: pytest.fail("envelope must not be copied"),
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="aggregate byte ceiling"):
        validate_detached_catalog_evidence(
            evidence,
            candidate_manifest=candidate,
            corpus_universe_manifest=universe,
        )


def test_stage_document_aggregate_is_preflighted_before_copy_or_decode(
    monkeypatch,
) -> None:
    candidate, _ = _candidate_fixture()
    evidence, universe = _content_wrapper_evidence(candidate)
    evidence["document_payloads"] = [
        {
            "accession_number": "0000320193-05-000001",
            "payload_base64": _b64(b"12345678"),
        },
        {
            "accession_number": "0000320193-05-000002",
            "payload_base64": _b64(b"abcdefgh"),
        },
    ]
    monkeypatch.setattr(
        verifier_module.base64,
        "b64decode",
        lambda *args, **kwargs: pytest.fail("decoder must not be called"),
    )
    monkeypatch.setattr(
        verifier_module,
        "_bounded_plain_json_copy",
        lambda *args, **kwargs: pytest.fail("envelope must not be copied"),
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="aggregate bytes"):
        validate_detached_stage_content_evidence(
            evidence,
            expected_stage="development",
            candidate_manifest=candidate,
            corpus_universe_manifest=universe,
            externally_pinned_content_manifest_sha256=evidence[
                "expected_content_manifest_sha256"
            ],
            externally_pinned_stage_artifact_sha256=evidence[
                "expected_stage_artifact_sha256"
            ],
            maximum_total_document_bytes=10,
        )


@pytest.mark.parametrize(
    ("limit_name", "limit", "payload", "message"),
    [
        (
            "MAX_PREFLIGHT_CONTAINER_ITEMS",
            3,
            {"too_many": [1, 2, 3, 4]},
            "maximum item count",
        ),
        (
            "MAX_PREFLIGHT_TOTAL_ELEMENTS",
            4,
            {"too_many_elements": [1, 2, 3, 4]},
            "maximum element count",
        ),
        (
            "MAX_PREFLIGHT_NESTING_DEPTH",
            3,
            {"deep": {"a": {"b": {"c": 1}}}},
            "maximum nesting depth",
        ),
    ],
)
def test_envelope_shape_limits_precede_copy_and_replay(
    monkeypatch,
    limit_name: str,
    limit: int,
    payload: dict,
    message: str,
) -> None:
    monkeypatch.setattr(verifier_module, limit_name, limit)
    monkeypatch.setattr(
        verifier_module,
        "_bounded_plain_json_copy",
        lambda *args, **kwargs: pytest.fail("envelope must not be copied"),
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match=message):
        audit_stage_evidence(payload, {}, {})


@pytest.mark.parametrize(
    ("limit_name", "limit", "message"),
    [
        ("MAX_PREFLIGHT_BASE64_ITEMS", 1, "Base64 item count"),
        (
            "MAX_PREFLIGHT_BASE64_DECODED_BYTES",
            10,
            "aggregate Base64 byte ceiling",
        ),
    ],
)
def test_base64_item_and_aggregate_limits_precede_envelope_copy(
    monkeypatch,
    limit_name: str,
    limit: int,
    message: str,
) -> None:
    payload = {
        "first_base64": _b64(b"12345678"),
        "second_base64": _b64(b"abcdefgh"),
    }
    monkeypatch.setattr(verifier_module, limit_name, limit)
    monkeypatch.setattr(
        verifier_module,
        "_bounded_plain_json_copy",
        lambda *args, **kwargs: pytest.fail("envelope must not be copied"),
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match=message):
        audit_stage_evidence(payload, {}, {})


def test_combined_caller_bundle_has_one_estimated_json_budget(monkeypatch) -> None:
    first = {"value": "a" * 40}
    second = {"value": "b" * 40}
    monkeypatch.setattr(
        verifier_module, "MAX_PREFLIGHT_ESTIMATED_JSON_BYTES", 100
    )

    verifier_module.preflight_untrusted_stage_json(first, "first envelope")
    verifier_module.preflight_untrusted_stage_json(second, "second envelope")
    with pytest.raises(
        SecFilingGemmaStageVerifierError,
        match="estimated canonical-JSON byte ceiling",
    ):
        verifier_module.preflight_untrusted_stage_json(
            {"first": first, "second": second}, "combined envelope"
        )


@pytest.mark.parametrize("value", [1 << 256, -(1 << 256)])
def test_integer_size_limit_precedes_copy_and_serialization(
    monkeypatch, value: int
) -> None:
    monkeypatch.setattr(
        verifier_module.json,
        "dumps",
        lambda *args, **kwargs: pytest.fail("JSON serialization must not run"),
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="integer size"):
        verifier_module.detach_untrusted_stage_json(
            {"oversized_integer": value}, "integer envelope"
        )


def test_padded_base64_preflight_does_not_copy_the_payload() -> None:
    payload = _b64(b"x" * (1024 * 1024 + 1))
    tracemalloc.start()
    try:
        decoded_length = verifier_module._base64_decoded_length_preflight(
            payload,
            "allocation probe",
            maximum=2 * 1024 * 1024,
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert decoded_length == 1024 * 1024 + 1
    assert peak < len(payload) // 8
