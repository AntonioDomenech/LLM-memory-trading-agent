from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import MappingProxyType

import pytest

from agent_benchmark import sec_gemma_lean_acquisition as lean
from agent_benchmark.sec_audit_transport import ResponseAudit
from agent_benchmark.sec_filing_gemma_corpus import SecCorpusBudget
from agent_benchmark.sec_point_in_time import content_sha256, validate_sec_user_agent


def _request(ordinal: int, size: int) -> dict[str, object]:
    return {
        "accession_number": f"0000320193-00-{ordinal:06d}",
        "request_sha256": f"{ordinal:064x}"[-64:],
        "request_bytes": bytes([ordinal]) * size,
    }


def _fake_checkpoint_inputs() -> dict[str, object]:
    plan = lean.validate_acquisition_plan(
        lean.build_acquisition_plan(lean.DEVELOPMENT),
        expected_stage=lean.DEVELOPMENT,
    )
    filing_count = plan["sec"]["minimum_filings"]
    bundle = {
        "bundle_sha256": "a" * 64,
        "private_index_sha256": "c" * 64,
        "public_manifest": {
            "manifest_sha256": "b" * 64,
            "private_index_sha256": "c" * 64,
            "sec_identity_sha256": "sha256:" + "2" * 64,
            "sec_primary_document_count": filing_count,
            "model_request_count": filing_count,
            "market_response_count": lean.MAX_MARKET_REQUESTS_PER_STAGE,
        },
        "private_quarantine": {
            "model_requests": [
                _request(index, 1000 + index) for index in range(1, 6)
            ],
            "sec_replay_evidence": {
                "stage_artifact": {
                    "documents": [
                        {
                            "accession_number": f"item-{index}",
                            "normalized_text_usable": True,
                        }
                        for index in range(filing_count)
                    ]
                }
            }
        },
    }
    readiness = {
        "preflight_sha256": "d" * 64,
        "head_commit": "e" * 40,
    }
    accounting = {
        "sec_request_count": filing_count + 2,
        "market_request_count": lean.MAX_MARKET_REQUESTS_PER_STAGE,
        "network_request_count": filing_count
        + 2
        + lean.MAX_MARKET_REQUESTS_PER_STAGE,
        "retry_count": 0,
        "redirect_count": 0,
        "sec_bytes": 1234,
        "market_bytes": 5678,
        "market_elapsed_seconds_hex": (1.0).hex(),
    }
    pilot = [
        {
            "ordinal": index,
            "accession_number": f"item-{index}",
            "request_sha256": f"{index:064x}",
            "request_utf8_bytes": 1000 + index,
        }
        for index in range(1, 6)
    ]
    source_resume_body = {
        "resume_contract": {
            "aggregate_sec_deadline_applied": False,
            "final_evidence_rebuilt_as_exact_frozen_batch": True,
        },
        "sec_prefix_checkpoint_sha256": "3" * 64,
        "unit_manifest_count": 2,
        "unit_manifests": [],
        "unit_manifest_set_sha256": "4" * 64,
        "all_units_immutable_and_canonical": True,
    }
    source_resume = {
        **source_resume_body,
        "source_resume_sha256": lean.canonical_sha256(source_resume_body),
    }
    return {
        "plan": plan,
        "bundle": bundle,
        "readiness": readiness,
        "accounting": accounting,
        "pilot": pilot,
        "source_resume": source_resume,
    }


def _fake_sec_prefix(plan: dict[str, object]) -> dict[str, object]:
    contact_sha256 = "sha256:" + "2" * 64
    filing_count = plan["sec"]["minimum_filings"]
    catalog_receipts = [
        {
            "size_bytes": 10,
            "network_requests": 1,
            "retries": 0,
            "redirects": 0,
            "user_agent_sha256": contact_sha256,
        }
    ]
    document_receipts = [
        {
            "size_bytes": 20,
            "network_requests": 1,
            "retries": 0,
            "redirects": 0,
            "user_agent_sha256": contact_sha256,
        }
        for _ in range(filing_count)
    ]
    return {
        "sec_replay_evidence": {
            "catalog_request_receipts": catalog_receipts,
            "authenticated_stage_request_receipts": document_receipts,
            "stage_artifact": {
                "user_agent_sha256": contact_sha256,
                "documents": [
                    {
                        "accession_number": f"item-{index}",
                        "normalized_text_usable": True,
                    }
                    for index in range(filing_count)
                ],
            },
        },
        "sec_catalog_sources": [
            {"name": "main", "url": "https://data.sec.gov/a", "body": b"catalog"}
        ],
        "sec_primary_documents": [
            {
                "accession_number": f"item-{index}",
                "body": f"document-{index}".encode("ascii"),
            }
            for index in range(filing_count)
        ],
    }


def test_public_api_has_no_test_or_sealing_switch() -> None:
    signature = inspect.signature(lean.run_development_acquisition)
    assert list(signature.parameters) == ["repo_root"]


def test_error_and_private_contact_reprs_are_redacted() -> None:
    secret = "Real Research Team secret-sentinel@reachable-domain.co"
    contact = lean._PrivateSecContact(secret)
    assert secret not in repr(contact)
    assert contact.sha256.startswith("sha256:")
    assert contact.encoded_for_echo_check() == secret.encode("utf-8")
    assert lean.SecGemmaLeanAcquisitionError("BAD!").code == "invalid_error_code"


def test_frozen_helper_identity_check_rejects_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lean._verify_frozen_helper_identities()
    monkeypatch.setattr(lean, "_build_model_requests", lambda *args, **kwargs: [])
    with pytest.raises(
        lean.SecGemmaLeanAcquisitionError,
        match="frozen_helper_identity_changed",
    ):
        lean._verify_frozen_helper_identities()


def test_local_production_identity_check_rejects_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lean._verify_local_function_identities()
    monkeypatch.setattr(lean, "_acquire_bundle", lambda **kwargs: None)
    with pytest.raises(
        lean.SecGemmaLeanAcquisitionError,
        match="production_function_identity_changed",
    ):
        lean._verify_local_function_identities()


def test_pilot_selection_uses_frozen_size_desc_accession_order() -> None:
    requests = [
        _request(1, 10),
        _request(2, 70),
        _request(3, 50),
        _request(4, 70),
        _request(5, 20),
        _request(6, 60),
        _request(7, 40),
    ]
    selected = lean._pilot_selection(requests)
    assert [item["ordinal"] for item in selected] == [2, 4, 6, 3, 7]
    assert [item["request_utf8_bytes"] for item in selected] == [70, 70, 60, 50, 40]


def test_external_byte_tree_round_trip_deduplicates_and_detects_corruption(
    tmp_path: Path,
) -> None:
    blob_root = tmp_path / "blobs"
    blob_root.mkdir()
    inventory: dict[str, int] = {}
    original = {
        "alpha": b"same bytes",
        "nested": [b"same bytes", {"value": 3, "flag": True}],
    }
    encoded = lean._encode_external_tree(
        original,
        blob_root=blob_root,
        inventory=inventory,
    )
    assert len(inventory) == 1
    seen: set[str] = set()
    decoded = lean._decode_external_tree(
        encoded,
        blob_root=blob_root,
        expected_inventory=inventory,
        seen=seen,
    )
    assert decoded == original
    assert seen == set(inventory)

    digest = next(iter(inventory))
    (blob_root / f"{digest}.bin").write_bytes(b"changed")
    with pytest.raises(lean.SecGemmaLeanAcquisitionError):
        lean._decode_external_tree(
            encoded,
            blob_root=blob_root,
            expected_inventory=inventory,
            seen=set(),
        )


def test_checkpoint_manifest_is_redacted_and_fail_closed() -> None:
    values = _fake_checkpoint_inputs()
    checkpoint = lean._build_checkpoint_manifest(
        readiness=values["readiness"],
        plan=values["plan"],
        bundle=values["bundle"],
        validation={"stage": "development", "validation_sha256": "f" * 64},
        accounting=values["accounting"],
        tree_bytes=b"tree",
        inventory={"1" * 64: 4},
        pilot_selection=values["pilot"],
        source_resume=values["source_resume"],
        contact_sha256="sha256:" + "2" * 64,
        started_at_utc="2026-07-17T00:00:00Z",
        finished_at_utc="2026-07-17T00:01:00Z",
        elapsed_seconds=60.0,
    )
    assert checkpoint["model_authorized"] is False
    assert checkpoint["effect_counts"]["model_generation_calls"] == 0
    assert checkpoint["effect_counts"]["performance_results_opened"] == 0
    assert checkpoint["checkpoint_sha256"] == lean.canonical_sha256(
        {
            key: value
            for key, value in checkpoint.items()
            if key != "checkpoint_sha256"
        }
    )

    values["bundle"]["private_quarantine"]["sec_replay_evidence"][
        "stage_artifact"
    ]["documents"][0]["normalized_text_usable"] = False
    with pytest.raises(
        lean.SecGemmaLeanAcquisitionError,
        match="filing_text_unusable",
    ):
        lean._build_checkpoint_manifest(
            readiness=values["readiness"],
            plan=values["plan"],
            bundle=values["bundle"],
            validation={},
            accounting=values["accounting"],
            tree_bytes=b"tree",
            inventory={},
            pilot_selection=values["pilot"],
            source_resume=values["source_resume"],
            contact_sha256="sha256:" + "2" * 64,
            started_at_utc="2026-07-17T00:00:00Z",
            finished_at_utc="2026-07-17T00:01:00Z",
            elapsed_seconds=60.0,
        )


def test_rehashed_checkpoint_claim_tamper_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fake_checkpoint_inputs()
    validation = {"validation_sha256": "f" * 64}
    monkeypatch.setattr(lean, "_safe_private_stage_root", lambda root: tmp_path)
    monkeypatch.setattr(
        lean,
        "_validate_one_bundle",
        lambda bundle, *, plan, predecessor: validation,
    )
    monkeypatch.setattr(
        lean,
        "_execution_request_accounting",
        lambda bundle, *, market_elapsed_seconds_hex: values["accounting"],
    )
    monkeypatch.setattr(
        lean,
        "_load_sec_prefix",
        lambda root: (
            {
                "repository": values["readiness"],
                "sec_contact_sha256": "sha256:" + "2" * 64,
                "acquisition_plan_sha256": values["plan"][
                    "acquisition_plan_sha256"
                ],
            },
            {},
            {},
        ),
    )
    monkeypatch.setattr(
        lean,
        "_source_resume_evidence",
        lambda root, prefix_manifest, prefix: values["source_resume"],
    )

    def factory(tree_bytes: bytes, inventory: dict[str, int]) -> dict[str, object]:
        return lean._build_checkpoint_manifest(
            readiness=values["readiness"],
            plan=values["plan"],
            bundle=values["bundle"],
            validation=validation,
            accounting=values["accounting"],
            tree_bytes=tree_bytes,
            inventory=inventory,
            pilot_selection=lean._pilot_selection(
                values["bundle"]["private_quarantine"]["model_requests"]
            ),
            source_resume=values["source_resume"],
            contact_sha256="sha256:" + "2" * 64,
            started_at_utc="2026-07-17T00:00:00Z",
            finished_at_utc="2026-07-17T00:01:00Z",
            elapsed_seconds=60.0,
        )

    lean._seal_private_checkpoint(
        tmp_path,
        bundle=values["bundle"],
        checkpoint_body_factory=factory,
        contact=lean._PrivateSecContact(
            "Real Research Team secret-sentinel@reachable-domain.co"
        ),
    )
    loaded, _, _ = lean._load_completed_checkpoint(tmp_path)
    assert loaded["effect_counts"]["model_generation_calls"] == 0

    path = tmp_path / lean.PRIVATE_STAGE_DIRECTORY / lean.PRIVATE_MANIFEST_NAME
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["effect_counts"]["model_generation_calls"] = 1
    tampered["checkpoint_sha256"] = lean.canonical_sha256(
        {
            key: value
            for key, value in tampered.items()
            if key != "checkpoint_sha256"
        }
    )
    path.write_bytes(lean.canonical_json_bytes(tampered))
    with pytest.raises(lean.SecGemmaLeanAcquisitionError):
        lean._load_completed_checkpoint(tmp_path)


def test_sec_prefix_round_trip_and_rehashed_tamper_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fake_checkpoint_inputs()
    prefix = _fake_sec_prefix(values["plan"])
    validation = {
        "catalog_replay_validation_sha256": "a" * 64,
        "stage_replay_validation_sha256": "b" * 64,
    }
    monkeypatch.setattr(lean, "_safe_private_stage_root", lambda root: tmp_path)
    monkeypatch.setattr(
        lean,
        "_validate_sec_detached_replay",
        lambda value, *, stage: validation,
    )

    def factory(tree_bytes: bytes, inventory: dict[str, int]) -> dict[str, object]:
        return lean._build_sec_prefix_manifest(
            readiness=values["readiness"],
            plan=values["plan"],
            prefix=prefix,
            validation=validation,
            tree_bytes=tree_bytes,
            inventory=inventory,
            contact_sha256="sha256:" + "2" * 64,
            started_at_utc="2026-07-17T00:00:00Z",
            finished_at_utc="2026-07-17T00:01:00Z",
            elapsed_seconds=60.0,
        )

    lean._seal_sec_prefix(
        tmp_path,
        prefix=prefix,
        manifest_factory=factory,
        contact=lean._PrivateSecContact(
            "Real Research Team secret-sentinel@reachable-domain.co"
        ),
    )
    loaded, reloaded, replayed = lean._load_sec_prefix(tmp_path)
    assert reloaded == prefix
    assert replayed == validation
    assert loaded["effect_counts"]["market_data_requests"] == 0

    path = (
        tmp_path
        / lean.PRIVATE_SEC_PREFIX_DIRECTORY
        / lean.PRIVATE_MANIFEST_NAME
    )
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["effect_counts"]["market_data_requests"] = 1
    tampered["sec_prefix_sha256"] = lean.canonical_sha256(
        {
            key: value
            for key, value in tampered.items()
            if key != "sec_prefix_sha256"
        }
    )
    path.write_bytes(lean.canonical_json_bytes(tampered))
    with pytest.raises(lean.SecGemmaLeanAcquisitionError):
        lean._load_sec_prefix(tmp_path)


def test_public_receipt_contains_only_fingerprint_and_zero_effect_gates() -> None:
    values = _fake_checkpoint_inputs()
    checkpoint = lean._build_checkpoint_manifest(
        readiness=values["readiness"],
        plan=values["plan"],
        bundle=values["bundle"],
        validation={"validation_sha256": "f" * 64},
        accounting=values["accounting"],
        tree_bytes=b"tree",
        inventory={"1" * 64: 4},
        pilot_selection=values["pilot"],
        source_resume=values["source_resume"],
        contact_sha256="sha256:" + "2" * 64,
        started_at_utc="2026-07-17T00:00:00Z",
        finished_at_utc="2026-07-17T00:01:00Z",
        elapsed_seconds=60.0,
    )
    receipt = lean._public_receipt(
        checkpoint=checkpoint,
        readiness_after=values["readiness"],
        journal_summary={
            "terminal_outcome_recorded": True,
            "head_event_sha256": "9" * 64,
            "event_count": 2,
            "lifetime_network_accounting": {
                "sec": {
                    "known_network_requests": values["accounting"][
                        "sec_request_count"
                    ]
                },
                "yahoo": {
                    "known_network_requests": values["accounting"][
                        "market_request_count"
                    ]
                },
            },
        },
    )
    encoded = json.dumps(receipt, sort_keys=True)
    assert "@" not in encoded
    assert receipt["model_authorized"] is False
    assert receipt["execution_authorized"] is False
    assert receipt["performance_results_opened"] is False
    assert receipt["privacy"]["readable_sec_contact_published"] is False
    assert receipt["effect_counts"]["model_generation_calls"] == 0


def test_sec_session_disables_environment_and_retries() -> None:
    session = lean._build_sec_session()
    try:
        assert session.trust_env is False
        assert session.auth is None
        assert not session.cookies
        assert not session.headers
        assert session.proxies == {}
        assert session.get_adapter("https://").max_retries.total == 0
    finally:
        session.close()


def test_catalog_manifest_recursively_detaches_validator_mapping_views() -> None:
    plan = lean.validate_acquisition_plan(
        lean.build_acquisition_plan(lean.DEVELOPMENT),
        expected_stage=lean.DEVELOPMENT,
    )
    state = {
        "catalog_sources": [
            {
                "name": "main",
                "url": "https://data.sec.gov/submissions/CIK0000320193.json",
                "body": b"{}",
            }
        ],
        "catalog_request_receipts": [
            {
                "size_bytes": 2,
                "network_requests": 1,
                "retries": 0,
                "redirects": 0,
            }
        ],
        "catalog_artifact": {"catalog_artifact_sha256": "a" * 64},
        "corpus_universe_manifest": {"universe_sha256": "b" * 64},
        "selected_records": [],
        "document_plan": [],
    }
    validation = {
        "source_payload_sha256s": MappingProxyType({"main": "c" * 64}),
        "request_receipt_sha256s": MappingProxyType({"main": "d" * 64}),
        "nested_sequence": (MappingProxyType({"value": 1}),),
    }
    manifest = lean._build_sec_catalog_manifest(
        readiness={"preflight_sha256": "e" * 64},
        plan=plan,
        state=state,
        validation=validation,
        tree_bytes=b"{}",
        inventory={},
        contact_sha256="sha256:" + "f" * 64,
        started_at_utc="2026-07-17T09:00:00Z",
        finished_at_utc="2026-07-17T09:00:01Z",
        elapsed_seconds=1.0,
    )
    detached = manifest["detached_validation"]
    assert type(detached["source_payload_sha256s"]) is dict
    assert type(detached["request_receipt_sha256s"]) is dict
    assert type(detached["nested_sequence"]) is list
    assert type(detached["nested_sequence"][0]) is dict
    lean._plain_json_bytes(manifest)


def test_stale_lock_recovery_removes_only_valid_dead_owner(tmp_path: Path) -> None:
    lock = tmp_path / "ACTIVE.lock"
    live = {
        "schema_version": "sec-gemma-lean-v3-local-lease-v1",
        "pid": os.getpid(),
        "nonce_sha256": "a" * 64,
    }
    lock.write_text(
        json.dumps(live, sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )
    assert lean._recover_stale_acquisition_lock(lock) is False
    assert lock.exists()

    dead = {**live, "pid": 2_147_000_000}
    lock.write_text(
        json.dumps(dead, sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )
    assert lean._recover_stale_acquisition_lock(lock) is True
    assert not lock.exists()


def test_stale_lock_recovery_rejects_malformed_marker(tmp_path: Path) -> None:
    lock = tmp_path / "ACTIVE.lock"
    lock.write_text("not-json", encoding="ascii")
    with pytest.raises(
        lean.SecGemmaLeanAcquisitionError,
        match="stale_lock_invalid",
    ):
        lean._recover_stale_acquisition_lock(lock)
    assert lock.exists()


def test_public_publisher_is_no_overwrite(tmp_path: Path) -> None:
    checkpoint = tmp_path / "data" / "aapl_sec_gemma_lean_evidence_v3"
    checkpoint.mkdir(parents=True)
    receipt = {"status": "passed", "secret": False}
    lean._publish_public_receipt(tmp_path, checkpoint, receipt)
    target = tmp_path / Path(lean.PUBLIC_ARTIFACT_PATH)
    first = target.read_bytes()
    with pytest.raises(
        lean.SecGemmaLeanAcquisitionError,
        match="public_artifact_already_exists",
    ):
        lean._publish_public_receipt(tmp_path, checkpoint, receipt)
    assert target.read_bytes() == first
    assert not list(checkpoint.glob(".acquisition-seal-*.tmp"))


def test_source_has_no_old_authority_model_or_scoring_path() -> None:
    source = Path(lean.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "acquire_stage_to_quarantine",
        "EffectCapability",
        "ProductionAcquisitionVault",
        "TestAcquisitionVault",
        "VerifiedProductionAuthority",
        "api/chat",
        "ollama",
        "score_stage",
    ):
        assert forbidden not in source


def test_recursive_privacy_scan_rejects_before_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "Real Research Team secret-sentinel@reachable-domain.co"
    contact = lean._PrivateSecContact(secret)
    monkeypatch.setattr(lean, "_safe_private_stage_root", lambda root: tmp_path)

    with pytest.raises(lean.SecGemmaLeanAcquisitionError, match="private_contact_leaked"):
        lean._seal_external_tree_unit(
            tmp_path,
            directory_name="privacy-test",
            value={"nested": [{"receipt": f"prefix {secret} suffix"}]},
            manifest_factory=lambda tree, inventory: {},
            contact=contact,
            publish_error_code="checkpoint_publish_failed",
        )

    assert not (tmp_path / "privacy-test").exists()
    assert not list(tmp_path.glob(".privacy-test-*.tmp"))


def test_attempt_journal_is_append_only_redacted_and_accounts_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fake_checkpoint_inputs()
    secret = "Real Research Team secret-sentinel@reachable-domain.co"
    contact = lean._PrivateSecContact(secret)
    monkeypatch.setattr(lean, "_safe_private_stage_root", lambda root: tmp_path)

    journal = lean._AttemptJournal(
        tmp_path,
        readiness=values["readiness"],
        plan=values["plan"],
        contact=contact,
    )
    assert journal.begin_attempt() == 1
    settled = journal.request_intent(
        provider="sec",
        logical_purpose="official_sec_catalog",
        request_key_sha256="1" * 64,
    )
    journal.request_settled(
        intent_event_sha256=settled,
        outcome="response_started",
        status_code=200,
    )
    journal.request_intent(
        provider="sec",
        logical_purpose="official_sec_catalog",
        request_key_sha256="2" * 64,
    )
    journal.finish_attempt(
        disposition="resumable_transient",
        code="sec_catalog_acquisition_failed",
    )

    summary = journal.summary()
    sec = summary["lifetime_network_accounting"]["sec"]
    assert sec["known_network_requests"] == 1
    assert sec["indeterminate_request_intents"] == 1
    assert sec["exact_network_request_count"] is None
    assert sec["network_request_count_lower_bound"] == 1
    assert sec["network_request_count_upper_bound"] == 2
    assert all(
        secret.encode("utf-8") not in path.read_bytes()
        for path in (tmp_path / lean.PRIVATE_ATTEMPT_JOURNAL_DIRECTORY).glob("*.json")
    )

    reopened = lean._AttemptJournal(
        tmp_path,
        readiness=values["readiness"],
        plan=values["plan"],
        contact=contact,
    )
    assert reopened.begin_attempt() == 2


def test_attempt_journal_rejects_self_rehashed_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fake_checkpoint_inputs()
    contact = lean._PrivateSecContact(
        "Real Research Team secret-sentinel@reachable-domain.co"
    )
    monkeypatch.setattr(lean, "_safe_private_stage_root", lambda root: tmp_path)
    journal = lean._AttemptJournal(
        tmp_path,
        readiness=values["readiness"],
        plan=values["plan"],
        contact=contact,
    )
    journal.begin_attempt()
    path = tmp_path / lean.PRIVATE_ATTEMPT_JOURNAL_DIRECTORY / "00000001.json"
    event = json.loads(path.read_text(encoding="utf-8"))
    event["payload"]["authority"]["repository"]["head_commit"] = "f" * 40
    event["event_sha256"] = lean.canonical_sha256(
        {key: value for key, value in event.items() if key != "event_sha256"}
    )
    path.write_bytes(lean.canonical_json_bytes(event))
    with pytest.raises(lean.SecGemmaLeanAcquisitionError):
        lean._AttemptJournal(
            tmp_path,
            readiness=values["readiness"],
            plan=values["plan"],
            contact=contact,
        )


def test_transport_error_remains_indeterminate_in_attempt_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fake_checkpoint_inputs()
    contact = lean._PrivateSecContact(
        "Real Research Team secret-sentinel@reachable-domain.co"
    )
    monkeypatch.setattr(lean, "_safe_private_stage_root", lambda root: tmp_path)
    journal = lean._AttemptJournal(
        tmp_path,
        readiness=values["readiness"],
        plan=values["plan"],
        contact=contact,
    )
    journal.begin_attempt()
    intent = journal.request_intent(
        provider="sec",
        logical_purpose="official_sec_catalog",
        request_key_sha256="7" * 64,
    )
    journal.request_settled(
        intent_event_sha256=intent,
        outcome="transport_error",
        status_code=None,
    )

    sec = journal.summary()["lifetime_network_accounting"]["sec"]
    assert sec["known_network_requests"] == 0
    assert sec["confirmed_response_started_requests"] == 0
    assert sec["ambiguous_settled_request_intents"] == 1
    assert sec["indeterminate_request_intents"] == 1
    assert sec["exact_network_request_count"] is None
    assert sec["network_request_count_lower_bound"] == 0
    assert sec["network_request_count_upper_bound"] == 1


def test_terminal_failure_receipt_is_redacted_honest_and_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fake_checkpoint_inputs()
    secret = "Real Research Team secret-sentinel@reachable-domain.co"
    contact = lean._PrivateSecContact(secret)
    monkeypatch.setattr(lean, "_safe_private_stage_root", lambda root: tmp_path)
    journal = lean._AttemptJournal(
        tmp_path,
        readiness=values["readiness"],
        plan=values["plan"],
        contact=contact,
    )
    journal.begin_attempt()
    intent = journal.request_intent(
        provider="sec",
        logical_purpose="official_sec_catalog",
        request_key_sha256="8" * 64,
    )
    journal.request_settled(
        intent_event_sha256=intent,
        outcome="transport_error",
        status_code=None,
    )
    evidence = lean._failure_evidence_commitments(tmp_path)
    journal.seal_terminal_failure(
        disposition="terminal_suitability_rejection",
        code="sec_prefix_suitability_failed",
        evidence_commitments=evidence,
    )
    terminal = journal.terminal_event()
    assert terminal is not None
    receipt = lean._public_failure_receipt(
        terminal_event=terminal,
        readiness=values["readiness"],
        plan=values["plan"],
        contact=contact,
        journal_summary=journal.summary(),
    )
    assert receipt["status"] == "rejected"
    assert receipt["request_accounting"]["all_request_effects_exact"] is False
    assert receipt["effect_counts"]["official_sec_requests"] is None
    assert receipt["effect_counts"]["official_sec_requests_lower_bound"] == 0
    assert receipt["effect_counts"]["official_sec_requests_upper_bound"] == 1
    assert receipt["effect_counts"]["model_generation_calls"] == 0
    assert secret not in json.dumps(receipt, sort_keys=True)

    reopened = lean._AttemptJournal(
        tmp_path,
        readiness=values["readiness"],
        plan=values["plan"],
        contact=contact,
    )
    rebuilt = lean._public_failure_receipt(
        terminal_event=reopened.terminal_event(),
        readiness=values["readiness"],
        plan=values["plan"],
        contact=contact,
        journal_summary=reopened.summary(),
    )
    assert rebuilt == receipt


class _ExactBatchTransport:
    def __init__(self, payloads: dict[str, bytes], user_agent: str) -> None:
        self.payloads = payloads
        self.user_agent_audit = validate_sec_user_agent(user_agent)

    def acquisition_security_state(self) -> dict[str, object]:
        return {
            "trust_env": False,
            "proxies": False,
            "follow_redirects": False,
            "max_retries": 0,
            "max_redirects": 0,
            "allow_cache_reads": False,
            "allow_cache_writes": False,
            "streaming_body": True,
            "content_length_preflight": True,
            "incremental_byte_budget": True,
            "transport_max_requests": lean.SEC_REQUEST_CAP,
            "transport_max_bytes": lean.SEC_BYTE_CAP,
            "transport_max_seconds": float(lean.MAX_SEC_SECONDS),
        }

    def fetch(self, url: str) -> tuple[bytes, ResponseAudit]:
        payload = self.payloads[url]
        return payload, ResponseAudit(
            url=url,
            status_code=200,
            content_type="text/html; charset=iso-8859-1",
            size_bytes=len(payload),
            content_sha256=content_sha256(payload),
            cache_hit=False,
            user_agent_sha256=self.user_agent_audit.sha256,
            network_requests=1,
            retries=0,
            redirects=0,
        )


def test_one_document_chunks_rebuild_exact_monolithic_batch() -> None:
    user_agent = "Real Research Team secret-sentinel@reachable-domain.co"
    plan = []
    payloads: dict[str, bytes] = {}
    for index in range(1, 8):
        accession = f"0000320193-24-{index:06d}"
        url = (
            "https://www.sec.gov/Archives/edgar/data/320193/"
            f"{accession.replace('-', '')}/aapl-{index}.htm"
        )
        plan.append({"accession_number": accession, "official_url": url})
        payloads[url] = (
            f"<html><body>Document {index} says demand improved and costs fell."
            " Services remained durable.</body></html>"
        ).encode("ascii")

    monolithic = lean._acquire_authenticated_stage_access_document_batch(
        authenticated_document_plan=plan,
        transport=_ExactBatchTransport(payloads, user_agent),
        user_agent=user_agent,
        budget=SecCorpusBudget(
            clock=lambda: 0.0,
            max_requests=lean.SEC_REQUEST_CAP,
            max_bytes=lean.SEC_BYTE_CAP,
            max_seconds=float(lean.MAX_SEC_SECONDS),
        ),
    )
    chunks = [
        lean._acquire_authenticated_stage_access_document_batch(
            authenticated_document_plan=[planned],
            transport=_ExactBatchTransport(payloads, user_agent),
            user_agent=user_agent,
            budget=SecCorpusBudget(
                clock=lambda: 0.0,
                max_requests=lean.SEC_REQUEST_CAP,
                max_bytes=lean.SEC_BYTE_CAP,
                max_seconds=float(lean.MAX_SEC_SECONDS),
            ),
        )
        for planned in plan
    ]
    rebuilt = lean._aggregate_document_batches(
        chunks,
        document_plan=plan,
        contact_sha256=validate_sec_user_agent(user_agent).sha256,
    )
    assert lean.SEC_DOCUMENT_CHUNK_SIZE == 1
    assert rebuilt.request_receipts_json == monolithic.request_receipts_json
    assert rebuilt.byte_manifest_json == monolithic.byte_manifest_json
    assert rebuilt.documents == monolithic.documents


def test_cli_reports_keyboard_interrupt_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        lean,
        "run_development_acquisition",
        lambda root: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    assert lean.main(["--repo-root", "."]) == 130
    assert json.loads(capsys.readouterr().out) == {
        "status": "interrupted",
        "code": "user_interrupt",
    }


def test_cli_reports_terminal_rejection_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "status": "rejected",
        "acquisition_sha256": "a" * 64,
        "rejection": {
            "disposition": "terminal_suitability_rejection",
            "code": "sec_prefix_suitability_failed",
            "later_stage_access_stopped": True,
        },
        "effect_counts": {
            "model_generation_calls": 0,
            "performance_results_opened": 0,
        },
        "next_step": "commit_rejection_and_update_approach_comparison",
    }
    monkeypatch.setattr(
        lean,
        "run_development_acquisition",
        lambda root: report,
    )
    assert lean.main(["--repo-root", "."]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "rejected"
    assert output["rejection"] == report["rejection"]
    assert "filing_count" not in output
