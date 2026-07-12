from __future__ import annotations

import copy
from collections.abc import Mapping
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
from types import MappingProxyType
from unittest.mock import patch

import pytest

import agent_benchmark.sec_filing_gemma_reveal_store as reveal_store_module

from agent_benchmark.sec_filing_gemma_contract import (
    CONTRACT_VERSION,
    REQUIRED_SOURCE_HASHES,
    build_candidate_manifest,
    canonical_sha256,
    session_calendar_sha256,
)
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    append_candidate_attempt,
    build_registry_pin_transition,
    build_single_candidate_reveal_request,
    candidate_design_sha256,
    historical_reveal_declaration,
)
from agent_benchmark.sec_filing_gemma_reveal_store import (
    AUTHORITATIVE_VALIDATOR_ID,
    CURRENT_TIP_PENDING_SCHEMA_VERSION,
    MAX_CURRENT_TIP_ANCHOR_FILE_BYTES,
    MAX_STATE_FILE_BYTES,
    REQUIRED_SEMANTIC_CHECKS,
    STATE_FILENAME,
    SecFilingGemmaRevealStore,
    SecFilingGemmaRevealStoreError,
    SemanticPrerequisiteValidation,
)
from agent_benchmark.sec_filing_gemma_stage_access import (
    STAGE_ACCESS_MANIFEST_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_stage_authorization import (
    CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION,
    SecFilingGemmaStageAuthorizationError,
    build_consumed_stage_authorization_grant,
    validate_consumed_stage_authorization_grant,
)
from agent_benchmark.sec_filing_gemma_stage_verifier import (
    STAGE_AUDIT_RECEIPT_SCHEMA_VERSION,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_ID = AUTHORITATIVE_VALIDATOR_ID
VALIDATOR_SOURCE_SHA256 = hashlib.sha256(b"test semantic verifier").hexdigest()
SEMANTIC_CHECKS = REQUIRED_SEMANTIC_CHECKS


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _commit(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _candidate(registry: dict, sequence: int, *, salt: str) -> dict:
    return build_candidate_manifest(
        model_digest=_digest(f"{salt}:model"),
        ollama_runtime_fingerprint_sha256=_digest(f"{salt}:runtime"),
        sec_audit_checksums_json_sha256=_digest(f"{salt}:audit"),
        sec_catalog_artifact_sha256=_digest(f"{salt}:catalog"),
        sec_audit_source_commit=_commit(f"{salt}:audit-commit"),
        calendar_source_evidence_sha256=_digest(f"{salt}:calendar-evidence"),
        calendar_sessions_sha256=session_calendar_sha256(EXPECTED_SESSIONS),
        corpus_universe_sha256=_digest(f"{salt}:universe"),
        corpus_universe_semantic_sha256=_digest(f"{salt}:semantic-universe"),
        identity_lexicon_sha256=_digest(f"{salt}:lexicon"),
        predecessor_reveal_registry_sha256=registry["registry_sha256"],
        holdout_attempt_id=(
            f"aapl-sec-filing-gemma-v1-attempt-{sequence:03d}"
        ),
        experiment_source_commit=_commit(f"{salt}:experiment-commit"),
        source_tree_sha256=_digest(f"{salt}:tree"),
        source_hashes={
            name: (
                VALIDATOR_SOURCE_SHA256
                if name == "stage_verifier"
                else _digest(f"{salt}:source:{name}")
            )
            for name in REQUIRED_SOURCE_HASHES
        },
    )


def _store(tmp_path: Path) -> SecFilingGemmaRevealStore:
    return SecFilingGemmaRevealStore(
        repository_root=REPO_ROOT,
        store_directory=tmp_path / "reveal-store",
    )


def _register(
    store: SecFilingGemmaRevealStore, *, salt: str = "registered"
) -> tuple[dict, dict]:
    initial = store.load()
    prior_registry = initial["latest_registry"]
    prior_pin = initial["latest_registry_pin"]
    candidate = _candidate(
        prior_registry,
        len(prior_registry["entries"]) + 1,
        salt=salt,
    )
    appended = append_candidate_attempt(
        prior_registry,
        external_prior_pin=prior_pin,
        candidate_manifest=candidate,
    )
    transition = build_registry_pin_transition(
        prior_registry,
        external_prior_pin=prior_pin,
        appended_registry=appended,
    )
    state = store.compare_and_swap_append(
        transition=transition,
        appended_registry=appended,
    )
    return state, candidate


def _evidence(stage: str, candidate: dict, *, salt: str) -> dict:
    return {
        "schema_version": "sec-gemma-test-stage-evidence-v1",
        "stage": stage,
        "attempt_id": candidate["bindings"]["holdout_attempt_id"],
        "candidate_sha256": candidate["candidate_sha256"],
        "semantic_payload_sha256": _digest(f"{salt}:{stage}:semantic-payload"),
    }


def _request(
    state: dict,
    candidate: dict,
    *,
    stage: str,
    evidence: dict,
    salt: str,
) -> tuple[dict, dict]:
    prerequisite = "development" if stage == "intermediate" else "intermediate"
    attempt = candidate["bindings"]["holdout_attempt_id"]
    access_body = {
        "schema_version": STAGE_ACCESS_MANIFEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "transition": {
            "prerequisite_stage": prerequisite,
            "requested_stage": stage,
            "transition_ordinal": 1 if stage == "intermediate" else 2,
            "single_use_consumption_required": True,
            "stage_reuse_permitted": False,
        },
        "candidate": {
            "candidate_sha256": candidate["candidate_sha256"],
            "candidate_design_sha256": candidate_design_sha256(candidate),
            "attempt_id": attempt,
        },
        "prerequisite_evidence_pin": {
            "stage": prerequisite,
            "content_manifest_sha256": _digest(f"{salt}:{prerequisite}:content"),
            "stage_artifact_sha256": _digest(f"{salt}:{prerequisite}:artifact"),
            "external_seal_receipt_sha256": _digest(
                f"{salt}:{prerequisite}:seal"
            ),
        },
        "scope_id": f"{salt}:{stage}:access",
    }
    access_hash = canonical_sha256(access_body)
    access_manifest = {
        **access_body,
        "stage_access_manifest_sha256": access_hash,
    }
    request = build_single_candidate_reveal_request(
        state["latest_registry"],
        external_pin=state["latest_registry_pin"],
        candidate_manifest=candidate,
        stage=stage,
        stage_access_manifest_sha256=access_hash,
        prerequisite_stage_evidence_sha256=canonical_sha256(evidence),
    )
    return request, access_manifest


def _grant_request(
    state: dict,
    candidate: dict,
    *,
    stage: str,
    evidence: dict,
) -> tuple[dict, dict]:
    prerequisite = "development" if stage == "intermediate" else "intermediate"
    attempt = candidate["bindings"]["holdout_attempt_id"]
    access_body = {
        "schema_version": STAGE_ACCESS_MANIFEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "transition": {
            "prerequisite_stage": prerequisite,
            "requested_stage": stage,
            "transition_ordinal": 1 if stage == "intermediate" else 2,
            "single_use_consumption_required": True,
            "stage_reuse_permitted": False,
        },
        "candidate": {
            "candidate_sha256": candidate["candidate_sha256"],
            "candidate_design_sha256": candidate_design_sha256(candidate),
            "attempt_id": attempt,
        },
        "output": {
            "namespace": f"aapl-sec-gemma-{attempt}-{stage}",
            "write_mode": "create_new_exclusive",
            "existing_namespace_reuse_permitted": False,
            "cross_stage_write_permitted": False,
        },
        "scope": {
            "authorized_stage": stage,
            "future_stage_access_permitted": False,
            "outcome_access_before_atomic_request_consumption_permitted": False,
        },
        "prerequisite_evidence_pin": {
            "stage": prerequisite,
            "content_manifest_sha256": _digest(
                f"grant:{attempt}:{prerequisite}:content"
            ),
            "stage_artifact_sha256": _digest(
                f"grant:{attempt}:{prerequisite}:artifact"
            ),
            "external_seal_receipt_sha256": _digest(
                f"grant:{attempt}:{prerequisite}:seal"
            ),
        },
    }
    access_manifest = {
        **access_body,
        "stage_access_manifest_sha256": canonical_sha256(access_body),
    }
    request = build_single_candidate_reveal_request(
        state["latest_registry"],
        external_pin=state["latest_registry_pin"],
        candidate_manifest=candidate,
        stage=stage,
        stage_access_manifest_sha256=access_manifest[
            "stage_access_manifest_sha256"
        ],
        prerequisite_stage_evidence_sha256=canonical_sha256(evidence),
    )
    return request, access_manifest


def _semantic_validator(
    evidence: dict,
    access_manifest: dict,
    context: dict,
    *,
    authenticated_store_context: dict,
) -> SemanticPrerequisiteValidation:
    assert evidence["stage"] == context["prerequisite_stage"]
    assert evidence["candidate_sha256"] == context["candidate_sha256"]
    assert access_manifest["stage_access_manifest_sha256"] == context[
        "stage_access_manifest_sha256"
    ]
    audit_body = {
        "schema_version": STAGE_AUDIT_RECEIPT_SCHEMA_VERSION,
        "prerequisite_stage": context["prerequisite_stage"],
        "requested_stage": context["stage"],
        "stage_evidence_sha256": context[
            "prerequisite_stage_evidence_sha256"
        ],
        "candidate_sha256": context["candidate_sha256"],
        "trusted_stage_content_pin_sha256": context[
            "trusted_stage_content_pin_sha256"
        ],
        "trusted_stage_content_authentication": authenticated_store_context[
            "trusted_stage_content_authentication"
        ],
        "trusted_stage_content_authentication_receipt_sha256": context[
            "trusted_stage_content_authentication_receipt_sha256"
        ],
        "parent_consumption_binding_sha256": context[
            "parent_consumption_binding_sha256"
        ],
        "authenticated_store_context_sha256": context[
            "authenticated_store_context_sha256"
        ],
        "replayed_without_outcome_access": True,
        "authorizes_outcome_access": False,
    }
    semantic_receipt = {
        **audit_body,
        "audit_receipt_sha256": canonical_sha256(audit_body),
    }
    return SemanticPrerequisiteValidation.success(
        context,
        validator_id=VALIDATOR_ID,
        validator_source_sha256=VALIDATOR_SOURCE_SHA256,
        semantic_checks=SEMANTIC_CHECKS,
        semantic_receipt=semantic_receipt,
    )


def _store_bound_receipt(context: Mapping[str, object], **extra: object) -> dict:
    return {
        "trusted_stage_content_pin_sha256": context[
            "trusted_stage_content_pin_sha256"
        ],
        "trusted_stage_content_authentication_receipt_sha256": context[
            "trusted_stage_content_authentication_receipt_sha256"
        ],
        "parent_consumption_binding_sha256": context[
            "parent_consumption_binding_sha256"
        ],
        "authenticated_store_context_sha256": context[
            "authenticated_store_context_sha256"
        ],
        **extra,
    }


def _consume(
    store: SecFilingGemmaRevealStore,
    request: dict,
    candidate: dict,
    evidence: dict,
    *,
    stage: str,
    access_hash: dict,
    validator=_semantic_validator,
) -> dict:
    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "authoritative_prerequisite_validator",
        validator,
    ), patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "AUTHORITATIVE_STAGE_PROMOTION_ENABLED",
        True,
    ):
        return store.consume_request(
            request,
            candidate_manifest=candidate,
            stage=stage,
            stage_access_manifest=access_hash,
            prerequisite_stage_evidence=evidence,
        )


def _consume_with_grant(
    store: SecFilingGemmaRevealStore,
    request: dict,
    candidate: dict,
    evidence: dict,
    *,
    stage: str,
    access_manifest: dict,
    validator=_semantic_validator,
) -> dict:
    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "authoritative_prerequisite_validator",
        validator,
    ), patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "AUTHORITATIVE_STAGE_PROMOTION_ENABLED",
        True,
    ):
        return store.consume_request_and_issue_authorization_grant(
            request,
            candidate_manifest=candidate,
            stage=stage,
            stage_access_manifest=access_manifest,
            prerequisite_stage_evidence=evidence,
        )


def test_fresh_store_uses_only_tracked_genesis_and_is_idempotent(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    first = store.initialize()
    repeated = store.initialize()

    assert repeated == first == store.load()
    assert first["latest_registry"]["entries"] == []
    assert first["latest_registry_pin"]["registered_entry_count"] == 0
    assert first["anchor"]["migration_source_canonical_sha256"] == (
        historical_reveal_declaration()["migration_source_canonical_sha256"]
    )
    chain = first["consumption_ledger"]["chain"]
    assert chain["consumed_request_count"] == 0
    assert chain["actual_final_touch_count"] == 0
    assert chain["historical_final_reveal_count_lower_bound"] == 10
    assert chain["repository_final_touch_count_lower_bound"] == 10
    assert store.load_current_tip_anchor()["trusted_stage_content_pins"] == {}


def test_interrupted_genesis_pending_anchor_recovers_on_initialize(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    real_atomic_replace = reveal_store_module._atomic_replace
    stopped = False

    def stop_after_genesis_pending(path: Path, payload: bytes) -> None:
        nonlocal stopped
        real_atomic_replace(path, payload)
        if path == store.current_tip_anchor_path and not stopped:
            parsed = json.loads(payload)
            if parsed.get("schema_version") == CURRENT_TIP_PENDING_SCHEMA_VERSION:
                stopped = True
                raise RuntimeError("simulated stop after genesis pending anchor")

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store._atomic_replace",
        stop_after_genesis_pending,
    ), pytest.raises(RuntimeError, match="genesis pending anchor"):
        store.initialize()

    assert stopped is True
    assert not store.state_path.exists()
    assert store.current_tip_anchor_path.exists()
    recovered = store.initialize()
    assert recovered["consumption_ledger"]["entries"] == []
    assert store.load() == recovered
    assert (
        json.loads(store.current_tip_anchor_path.read_bytes())["schema_version"]
        != CURRENT_TIP_PENDING_SCHEMA_VERSION
    )


def test_registry_registration_is_exact_cas_and_does_not_count_as_final_touch(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()

    registered, candidate = _register(store)

    assert registered["latest_registry_pin"]["registered_entry_count"] == 1
    assert registered["latest_registry"]["entries"][0]["candidate_sha256"] == (
        candidate["candidate_sha256"]
    )
    assert registered["consumption_ledger"]["entries"] == []
    assert registered["consumption_ledger"]["chain"][
        "actual_final_touch_count"
    ] == 0


def test_registry_cas_deep_input_fails_before_copy_and_preserves_store(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()
    transition: dict[str, object] = {}
    cursor = transition
    for _ in range(40):
        nested: dict[str, object] = {}
        cursor["nested"] = nested
        cursor = nested

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="CAS inputs exceed fixed allocation bounds",
    ):
        store.compare_and_swap_append(
            transition=transition,
            appended_registry={},
        )

    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes


def test_stale_or_forked_registry_cas_is_rejected_without_state_change(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    initial = store.initialize()
    prior_registry = initial["latest_registry"]
    prior_pin = initial["latest_registry_pin"]

    children: list[tuple[dict, dict]] = []
    for salt in ("winning-cas", "stale-fork"):
        candidate = _candidate(prior_registry, 1, salt=salt)
        appended = append_candidate_attempt(
            prior_registry,
            external_prior_pin=prior_pin,
            candidate_manifest=candidate,
        )
        transition = build_registry_pin_transition(
            prior_registry,
            external_prior_pin=prior_pin,
            appended_registry=appended,
        )
        children.append((appended, transition))

    store.compare_and_swap_append(
        appended_registry=children[0][0], transition=children[0][1]
    )
    before_rejection = store.load()
    with pytest.raises(
        SecFilingGemmaRevealStoreError, match="stale, forked, or rollback"
    ):
        store.compare_and_swap_append(
            appended_registry=children[1][0], transition=children[1][1]
        )
    assert store.load() == before_rejection


def test_tampered_authoritative_state_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    state = store.initialize()
    changed = copy.deepcopy(state)
    changed["consumption_ledger"]["chain"]["actual_final_touch_count"] = 7
    changed["state_sha256"] = canonical_sha256(
        {key: changed[key] for key in changed if key != "state_sha256"}
    )
    store.state_path.write_text(
        json.dumps(changed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SecFilingGemmaRevealStoreError, match="Consumption ledger count"
    ):
        store.load()


def test_intermediate_request_is_single_use_and_does_not_touch_final(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store)
    evidence = _evidence("development", candidate, salt="intermediate")
    request, access_hash = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="intermediate",
    )
    tip_before = store.load_current_tip_anchor()

    consumed = _consume(
        store,
        request,
        candidate,
        evidence,
        stage="intermediate",
        access_hash=access_hash,
    )
    chain = consumed["consumption_ledger"]["chain"]
    assert chain["consumed_request_count"] == 1
    assert chain["actual_final_touch_count"] == 0
    assert consumed["consumption_ledger"]["entries"][0][
        "final_touch_delta"
    ] == 0
    tip_after = store.load_current_tip_anchor()
    assert tip_after["revision"] == tip_before["revision"] + 2
    assert set(tip_after["trusted_stage_content_pins"]) == {
        request["request_sha256"]
    }

    before_replay = store.load()
    with pytest.raises(SecFilingGemmaRevealStoreError, match="already consumed"):
        _consume(
            store,
            request,
            candidate,
            evidence,
            stage="intermediate",
            access_hash=access_hash,
        )
    assert store.load() == before_replay


def test_public_consume_uses_only_the_fixed_verifier_and_fails_closed(
    tmp_path: Path,
) -> None:
    signature = inspect.signature(SecFilingGemmaRevealStore.consume_request)
    assert "prerequisite_validator" not in signature.parameters

    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="fixed-verifier-only")
    evidence = _evidence("development", candidate, salt="fixed-verifier-only")
    request, access_manifest = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="fixed-verifier-only",
    )
    before = store.load()

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="Fixed semantic prerequisite verifier failed",
    ):
        store.consume_request(
            request,
            candidate_manifest=candidate,
            stage="intermediate",
            stage_access_manifest=access_manifest,
            prerequisite_stage_evidence=evidence,
        )
    assert store.load() == before


def test_blocked_verifier_observes_precommitted_pin_and_retry_reuses_it(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="blocked-pin-retry")
    evidence = _evidence("development", candidate, salt="blocked-pin-retry")
    request, access_manifest = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="blocked-pin-retry",
    )
    state_bytes = store.state_path.read_bytes()
    state_before = store.load()
    tip_before = store.load_current_tip_anchor()
    observed_pins: list[dict] = []

    def blocked_validator(
        _evidence,
        _access,
        _context,
        *,
        authenticated_store_context,
    ):
        on_disk_tip = json.loads(store.current_tip_anchor_path.read_bytes())
        persisted_pin = on_disk_tip["trusted_stage_content_pins"][
            request["request_sha256"]
        ]
        assert persisted_pin == authenticated_store_context[
            "trusted_stage_content_pin"
        ]
        observed_pins.append(copy.deepcopy(persisted_pin))
        raise RuntimeError("synthetic blocked verifier")

    for attempt in range(2):
        with pytest.raises(
            SecFilingGemmaRevealStoreError,
            match="Fixed semantic prerequisite verifier failed",
        ):
            _consume(
                store,
                request,
                candidate,
                evidence,
                stage="intermediate",
                access_hash=access_manifest,
                validator=blocked_validator,
            )
        tip_after_attempt = store.load_current_tip_anchor()
        if attempt == 0:
            first_tip_bytes = store.current_tip_anchor_path.read_bytes()
            first_tip = tip_after_attempt
        else:
            assert store.current_tip_anchor_path.read_bytes() == first_tip_bytes
            assert tip_after_attempt == first_tip

    assert len(observed_pins) == 2
    assert observed_pins[0] == observed_pins[1]
    assert first_tip["revision"] == tip_before["revision"] + 1
    assert set(first_tip["trusted_stage_content_pins"]) == {
        request["request_sha256"]
    }
    assert first_tip["authorization_bundles"] == tip_before[
        "authorization_bundles"
    ]
    assert store.state_path.read_bytes() == state_bytes
    assert store.load() == state_before
    assert store.load()["consumption_ledger"]["entries"] == []


def test_substituted_successful_verifier_cannot_bypass_store_promotion_gate(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="independent-promotion-gate")
    evidence = _evidence(
        "development", candidate, salt="independent-promotion-gate"
    )
    request, access_manifest = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="independent-promotion-gate",
    )
    state_bytes = store.state_path.read_bytes()
    tip_before = store.load_current_tip_anchor()

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "authoritative_prerequisite_validator",
        _semantic_validator,
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="stage promotion remains independently disabled",
    ):
        store.consume_request(
            request,
            candidate_manifest=candidate,
            stage="intermediate",
            stage_access_manifest=access_manifest,
            prerequisite_stage_evidence=evidence,
        )

    assert store.state_path.read_bytes() == state_bytes
    tip_after = store.load_current_tip_anchor()
    assert tip_after["revision"] == tip_before["revision"] + 1
    assert set(tip_after["trusted_stage_content_pins"]) == {
        request["request_sha256"]
    }
    assert tip_after["trusted_stage_content_pins"][request["request_sha256"]][
        "request_sha256"
    ] == request["request_sha256"]
    assert store.load()["consumption_ledger"]["entries"] == []


def test_substituted_verifier_cannot_enable_the_store_promotion_gate(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="verifier-gate-flip")
    evidence = _evidence("development", candidate, salt="verifier-gate-flip")
    request, access_manifest = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="verifier-gate-flip",
    )
    state_bytes = store.state_path.read_bytes()
    tip_before = store.load_current_tip_anchor()

    def flip_gate_then_succeed(
        evidence_value,
        access_value,
        context_value,
        *,
        authenticated_store_context,
    ):
        reveal_store_module.AUTHORITATIVE_STAGE_PROMOTION_ENABLED = True
        return _semantic_validator(
            evidence_value,
            access_value,
            context_value,
            authenticated_store_context=authenticated_store_context,
        )

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "authoritative_prerequisite_validator",
        flip_gate_then_succeed,
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="stage promotion remains independently disabled",
    ):
        store.consume_request(
            request,
            candidate_manifest=candidate,
            stage="intermediate",
            stage_access_manifest=access_manifest,
            prerequisite_stage_evidence=evidence,
        )

    assert reveal_store_module.AUTHORITATIVE_STAGE_PROMOTION_ENABLED is False
    assert store.state_path.read_bytes() == state_bytes
    tip_after = store.load_current_tip_anchor()
    assert tip_after["revision"] == tip_before["revision"] + 1
    assert set(tip_after["trusted_stage_content_pins"]) == {
        request["request_sha256"]
    }
    assert store.load()["consumption_ledger"]["entries"] == []


def test_grant_issuing_consume_remains_locked_by_the_fixed_verifier(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="fixed-grant-verifier")
    evidence = _evidence("development", candidate, salt="fixed-grant-verifier")
    request, access_manifest = _grant_request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
    )
    before = store.load()

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="Fixed semantic prerequisite verifier failed",
    ):
        store.consume_request_and_issue_authorization_grant(
            request,
            candidate_manifest=candidate,
            stage="intermediate",
            stage_access_manifest=access_manifest,
            prerequisite_stage_evidence=evidence,
        )
    assert store.load() == before


def test_successful_consumption_issues_exact_post_append_grant_under_the_lock(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="issued-grant")
    evidence = _evidence("development", candidate, salt="issued-grant")
    request, access_manifest = _grant_request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
    )
    observed: dict[str, object] = {}

    def checking_builder(**kwargs):
        on_disk_bytes = store.state_path.read_bytes()
        on_disk = json.loads(on_disk_bytes)
        snapshot = kwargs["authenticated_store_snapshot"]
        # Grant construction happens before the write-ahead transaction; the
        # old state must still be authoritative until state+bundle can commit
        # as one recoverable unit.
        assert on_disk["consumption_ledger"]["chain"]["consumed_request_count"] == 0
        entry = snapshot["consumption_ledger"]["entries"][-1]
        assert entry["entry_sha256"] == kwargs[
            "expected_new_consumption_entry_sha256"
        ]
        assert snapshot["consumption_ledger"]["chain"]["tip_sha256"] == entry[
            "entry_sha256"
        ]
        observed["entry_sha256"] = entry["entry_sha256"]
        expected_bytes = (
            json.dumps(
                snapshot,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        observed["state_bytes_sha256"] = hashlib.sha256(expected_bytes).hexdigest()
        observed["state_byte_count"] = len(expected_bytes)
        return build_consumed_stage_authorization_grant(**kwargs)

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "authoritative_prerequisite_validator",
        _semantic_validator,
    ), patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "AUTHORITATIVE_STAGE_PROMOTION_ENABLED",
        True,
    ), patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "build_consumed_stage_authorization_grant",
        checking_builder,
    ):
        bundle = store.consume_request_and_issue_authorization_grant(
            request,
            candidate_manifest=candidate,
            stage="intermediate",
            stage_access_manifest=access_manifest,
            prerequisite_stage_evidence=evidence,
        )

    snapshot = bundle["authenticated_store_snapshot"]
    entry = snapshot["consumption_ledger"]["entries"][-1]
    grant = bundle["authorization_grant"]
    assert bundle["schema_version"] == (
        CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION
    )
    assert canonical_sha256(
        {key: bundle[key] for key in bundle if key != "bundle_sha256"}
    ) == bundle["bundle_sha256"]
    assert snapshot == store.load()
    assert observed["entry_sha256"] == entry["entry_sha256"]
    assert grant["consumption_entry_sha256"] == entry["entry_sha256"]
    assert grant["consumption_ledger_tip_sha256"] == entry["entry_sha256"]
    assert grant["store_state_sha256"] == snapshot["state_sha256"]
    assert grant["store_snapshot_bytes_sha256"] == observed["state_bytes_sha256"]
    assert grant["store_snapshot_byte_count"] == observed["state_byte_count"]
    assert grant["outcomes_included"] is False
    assert validate_consumed_stage_authorization_grant(
        grant,
        authenticated_store_snapshot=snapshot,
        external_store_state_pin=bundle["store_state_pin"],
        independent_current_tip_anchor=store.load_current_tip_anchor(),
        expected_consumption_entry_sha256=entry["entry_sha256"],
        expected_request_sha256=request["request_sha256"],
        expected_candidate_sha256=candidate["candidate_sha256"],
        expected_stage="intermediate",
        expected_prerequisite_stage_evidence_sha256=canonical_sha256(evidence),
        expected_stage_access_manifest_sha256=access_manifest[
            "stage_access_manifest_sha256"
        ],
        expected_output_namespace=access_manifest["output"]["namespace"],
    ) == grant["authorization_grant_sha256"]


def test_grant_failure_rolls_back_the_consumption_before_unlocking(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="grant-rollback")
    evidence = _evidence("development", candidate, salt="grant-rollback")
    request, access_manifest = _grant_request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
    )
    before = store.load()
    before_bytes = store.state_path.read_bytes()

    def fail_grant(**_kwargs):
        raise SecFilingGemmaStageAuthorizationError("synthetic grant failure")

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "authoritative_prerequisite_validator",
        _semantic_validator,
    ), patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "AUTHORITATIVE_STAGE_PROMOTION_ENABLED",
        True,
    ), patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "build_consumed_stage_authorization_grant",
        fail_grant,
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="could not produce an exact authorization grant",
    ):
        store.consume_request_and_issue_authorization_grant(
            request,
            candidate_manifest=candidate,
            stage="intermediate",
            stage_access_manifest=access_manifest,
            prerequisite_stage_evidence=evidence,
        )

    assert store.state_path.read_bytes() == before_bytes
    assert store.load() == before


def test_state_write_crash_recovers_exact_bundle_and_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="crash-retry")
    evidence = _evidence("development", candidate, salt="crash-retry")
    request, access_manifest = _grant_request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
    )
    real_atomic_replace = reveal_store_module._atomic_replace
    crashed = False

    def crash_after_state_replace(path: Path, payload: bytes) -> None:
        nonlocal crashed
        real_atomic_replace(path, payload)
        if (
            path == store.state_path
            and json.loads(payload)["consumption_ledger"]["chain"][
                "consumed_request_count"
            ]
            == 1
            and not crashed
        ):
            crashed = True
            raise RuntimeError("simulated process stop after state replace")

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "authoritative_prerequisite_validator",
        _semantic_validator,
    ), patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "AUTHORITATIVE_STAGE_PROMOTION_ENABLED",
        True,
    ), patch(
        "agent_benchmark.sec_filing_gemma_reveal_store._atomic_replace",
        crash_after_state_replace,
    ), pytest.raises(RuntimeError, match="simulated process stop"):
        store.consume_request_and_issue_authorization_grant(
            request,
            candidate_manifest=candidate,
            stage="intermediate",
            stage_access_manifest=access_manifest,
            prerequisite_stage_evidence=evidence,
        )

    pending = json.loads(store.current_tip_anchor_path.read_bytes())
    assert pending["schema_version"] == CURRENT_TIP_PENDING_SCHEMA_VERSION
    exact_persisted_bundle = pending["next_tip_anchor"][
        "authorization_bundles"
    ][request["request_sha256"]]

    def verifier_must_not_run(*_args, **_kwargs):
        raise AssertionError("an idempotent retry must not re-run the verifier")

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "authoritative_prerequisite_validator",
        verifier_must_not_run,
    ):
        recovered = store.consume_request_and_issue_authorization_grant(
            request,
            candidate_manifest=candidate,
            stage="intermediate",
            stage_access_manifest=access_manifest,
            prerequisite_stage_evidence=evidence,
        )

    assert recovered == exact_persisted_bundle
    assert store.load()["consumption_ledger"]["chain"][
        "consumed_request_count"
    ] == 1
    current_tip = store.load_current_tip_anchor()
    assert current_tip["schema_version"] != CURRENT_TIP_PENDING_SCHEMA_VERSION
    assert current_tip["authorization_bundles"][request["request_sha256"]] == recovered


def test_old_snapshot_and_bundled_pin_fail_after_any_newer_store_tip(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="stale-bundle")
    evidence = _evidence("development", candidate, salt="stale-bundle")
    request, access_manifest = _grant_request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
    )
    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "authoritative_prerequisite_validator",
        _semantic_validator,
    ), patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "AUTHORITATIVE_STAGE_PROMOTION_ENABLED",
        True,
    ):
        old_bundle = store.consume_request_and_issue_authorization_grant(
            request,
            candidate_manifest=candidate,
            stage="intermediate",
            stage_access_manifest=access_manifest,
            prerequisite_stage_evidence=evidence,
        )
    old_state_bytes = store.state_path.read_bytes()
    old_state = old_bundle["authenticated_store_snapshot"]
    old_entry = old_state["consumption_ledger"]["entries"][-1]

    _register(store, salt="newer-registry-tip")
    new_state = store.load()
    new_tip = store.load_current_tip_anchor()
    assert new_state["state_sha256"] != old_state["state_sha256"]

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="current-tip anchor does not authenticate",
    ):
        validate_consumed_stage_authorization_grant(
            old_bundle["authorization_grant"],
            authenticated_store_snapshot=old_state,
            external_store_state_pin=old_bundle["store_state_pin"],
            independent_current_tip_anchor=new_tip,
            expected_consumption_entry_sha256=old_entry["entry_sha256"],
            expected_request_sha256=request["request_sha256"],
            expected_candidate_sha256=candidate["candidate_sha256"],
            expected_stage="intermediate",
            expected_prerequisite_stage_evidence_sha256=canonical_sha256(evidence),
            expected_stage_access_manifest_sha256=access_manifest[
                "stage_access_manifest_sha256"
            ],
            expected_output_namespace=access_manifest["output"]["namespace"],
        )

    # Rolling back only the state to the old bundled snapshot cannot pass the
    # separately persisted newer tip anchor on the next load.
    store.state_path.write_bytes(old_state_bytes)
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="current-tip anchor does not authenticate",
    ):
        store.load()


def test_malicious_mapping_cannot_execute_a_state_and_anchor_rollback(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    stale_state_bytes = store.state_path.read_bytes()
    stale_tip_bytes = store.current_tip_anchor_path.read_bytes()
    _register(store, salt="mapping-rollback-current")
    current_state_bytes = store.state_path.read_bytes()
    current_tip_bytes = store.current_tip_anchor_path.read_bytes()
    hooks_executed = False

    class RollbackMapping(Mapping):
        def _rollback(self) -> None:
            nonlocal hooks_executed
            hooks_executed = True
            store.state_path.write_bytes(stale_state_bytes)
            store.current_tip_anchor_path.write_bytes(stale_tip_bytes)

        def __getitem__(self, key):
            self._rollback()
            raise KeyError(key)

        def __iter__(self):
            self._rollback()
            return iter(())

        def __len__(self):
            self._rollback()
            return 0

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="exact built-in dict",
    ):
        store.compare_and_swap_append(
            transition=RollbackMapping(),
            appended_registry={},
        )

    assert hooks_executed is False
    assert store.state_path.read_bytes() == current_state_bytes
    assert store.current_tip_anchor_path.read_bytes() == current_tip_bytes
    store.load()


def test_nested_mapping_proxy_cannot_execute_state_and_anchor_rollback(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    stale_state_bytes = store.state_path.read_bytes()
    stale_tip_bytes = store.current_tip_anchor_path.read_bytes()
    registered, candidate = _register(store, salt="nested-proxy-current")
    evidence = _evidence("development", candidate, salt="nested-proxy-current")
    request, access_manifest = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="nested-proxy-current",
    )
    current_state_bytes = store.state_path.read_bytes()
    current_tip_bytes = store.current_tip_anchor_path.read_bytes()
    hooks_executed = False

    class RollbackMapping(Mapping):
        def _rollback(self) -> None:
            nonlocal hooks_executed
            hooks_executed = True
            store.state_path.write_bytes(stale_state_bytes)
            store.current_tip_anchor_path.write_bytes(stale_tip_bytes)

        def __getitem__(self, key):
            self._rollback()
            raise KeyError(key)

        def __iter__(self):
            self._rollback()
            return iter(())

        def __len__(self):
            self._rollback()
            return 0

    evidence["nested_hostile_proxy"] = MappingProxyType(RollbackMapping())
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="fixed verifier allocation bounds",
    ):
        store.consume_request(
            request,
            candidate_manifest=candidate,
            stage="intermediate",
            stage_access_manifest=access_manifest,
            prerequisite_stage_evidence=evidence,
        )

    assert hooks_executed is False
    assert store.state_path.read_bytes() == current_state_bytes
    assert store.current_tip_anchor_path.read_bytes() == current_tip_bytes
    store.load()


def test_oversized_state_and_current_tip_anchor_fail_before_unbounded_reads(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()

    with store.state_path.open("r+b") as handle:
        handle.truncate(MAX_STATE_FILE_BYTES + 1)
    with pytest.raises(SecFilingGemmaRevealStoreError, match="exceeds"):
        store.load()

    store.state_path.write_bytes(state_bytes)
    store.load()
    with store.current_tip_anchor_path.open("r+b") as handle:
        handle.truncate(MAX_CURRENT_TIP_ANCHOR_FILE_BYTES + 1)
    with pytest.raises(SecFilingGemmaRevealStoreError, match="exceeds"):
        store.load()

    store.current_tip_anchor_path.write_bytes(tip_bytes)
    assert store.load()["state_sha256"] == json.loads(state_bytes)["state_sha256"]


def test_deep_caller_evidence_fails_before_copy_or_verifier_and_preserves_store(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="deep-preflight")
    evidence = _evidence("development", candidate, salt="deep-preflight")
    request, access_manifest = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="deep-preflight",
    )
    cursor: dict[str, object] = evidence
    for _ in range(40):
        nested: dict[str, object] = {}
        cursor["nested"] = nested
        cursor = nested

    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store._exact_caller_dict",
        side_effect=AssertionError("caller evidence must not be copied"),
    ), patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "authoritative_prerequisite_validator",
        side_effect=AssertionError("verifier must not run"),
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="exceed the fixed verifier allocation bounds",
    ):
        store.consume_request(
            request,
            candidate_manifest=candidate,
            stage="intermediate",
            stage_access_manifest=access_manifest,
            prerequisite_stage_evidence=evidence,
        )

    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes
    store.load()


def test_evidence_mutation_after_preflight_is_rechecked_during_bounded_copy(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="preflight-mutation")
    evidence = _evidence("development", candidate, salt="preflight-mutation")
    request, access_manifest = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="preflight-mutation",
    )
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()
    real_preflight = reveal_store_module.preflight_untrusted_stage_json
    mutated = False

    def mutate_after_preflight(value, location):
        nonlocal mutated
        totals = real_preflight(value, location)
        cursor: dict[str, object] = evidence
        for _ in range(40):
            nested: dict[str, object] = {}
            cursor["inserted_after_check"] = nested
            cursor = nested
        mutated = True
        return totals

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "preflight_untrusted_stage_json",
        mutate_after_preflight,
    ), patch(
        "agent_benchmark.sec_filing_gemma_reveal_store."
        "authoritative_prerequisite_validator",
        side_effect=AssertionError("verifier must not run"),
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="exceed the fixed verifier allocation bounds",
    ):
        store.consume_request(
            request,
            candidate_manifest=candidate,
            stage="intermediate",
            stage_access_manifest=access_manifest,
            prerequisite_stage_evidence=evidence,
        )

    assert mutated is True
    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes
    store.load()


def test_verifier_exception_after_state_mutation_restores_exact_original_bytes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="mutate-then-raise")
    evidence = _evidence("development", candidate, salt="mutate-then-raise")
    request, access_manifest = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="mutate-then-raise",
    )
    before = store.load()
    original_bytes = store.state_path.read_bytes()

    def mutate_then_raise(
        _evidence,
        _access,
        _context,
        *,
        authenticated_store_context,
    ):
        assert authenticated_store_context["trusted_stage_content_pin"][
            "request_sha256"
        ] == request["request_sha256"]
        assert store.restore_pending_path.exists()
        store.state_path.write_bytes(b'{"verifier_mutation":true}\n')
        assert store.state_path.read_bytes() != original_bytes
        raise RuntimeError("verifier failed after mutation")

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="Fixed semantic prerequisite verifier failed",
    ):
        _consume(
            store,
            request,
            candidate,
            evidence,
            stage="intermediate",
            access_hash=access_manifest,
            validator=mutate_then_raise,
        )

    assert store.state_path.read_bytes() == original_bytes
    assert store.load() == before


def test_interrupted_failed_verifier_restore_recovers_both_files_on_next_load(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="restore-transaction-crash")
    evidence = _evidence(
        "development", candidate, salt="restore-transaction-crash"
    )
    request, access_manifest = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="restore-transaction-crash",
    )
    original_state = store.load()
    original_state_bytes = store.state_path.read_bytes()
    original_tip_bytes = store.current_tip_anchor_path.read_bytes()
    post_pin_tip_bytes: bytes | None = None

    def mutate_both_then_raise(
        _evidence,
        _access,
        _context,
        *,
        authenticated_store_context,
    ):
        nonlocal post_pin_tip_bytes
        assert authenticated_store_context["trusted_stage_content_pin"][
            "request_sha256"
        ] == request["request_sha256"]
        post_pin_tip_bytes = store.current_tip_anchor_path.read_bytes()
        store.state_path.write_bytes(b'{"mutated_state":true}\n')
        store.current_tip_anchor_path.write_bytes(b'{"mutated_tip":true}\n')
        raise RuntimeError("verifier failed after mutating both files")

    real_atomic_replace = reveal_store_module._atomic_replace
    stopped = False

    def stop_after_recovered_state(path: Path, payload: bytes) -> None:
        nonlocal stopped
        real_atomic_replace(path, payload)
        if (
            path == store.state_path
            and store.restore_pending_path.exists()
            and not stopped
        ):
            stopped = True
            raise RuntimeError("simulated stop between restore targets")

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store._atomic_replace",
        stop_after_recovered_state,
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="could not be restored",
    ):
        _consume(
            store,
            request,
            candidate,
            evidence,
            stage="intermediate",
            access_hash=access_manifest,
            validator=mutate_both_then_raise,
        )

    assert stopped is True
    assert post_pin_tip_bytes is not None
    assert post_pin_tip_bytes != original_tip_bytes
    assert store.restore_pending_path.exists()
    assert store.state_path.read_bytes() == original_state_bytes
    assert store.current_tip_anchor_path.read_bytes() != original_tip_bytes

    assert store.load() == original_state
    assert store.state_path.read_bytes() == original_state_bytes
    assert store.current_tip_anchor_path.read_bytes() == post_pin_tip_bytes
    assert not store.restore_pending_path.exists()


def test_invalid_verifier_result_after_state_mutation_restores_exact_original_bytes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="mutate-then-invalid")
    evidence = _evidence("development", candidate, salt="mutate-then-invalid")
    request, access_manifest = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="mutate-then-invalid",
    )
    before = store.load()
    original_bytes = store.state_path.read_bytes()

    def mutate_then_return_invalid(
        _evidence,
        _access,
        _context,
        *,
        authenticated_store_context,
    ):
        assert authenticated_store_context["trusted_stage_content_pin"][
            "request_sha256"
        ] == request["request_sha256"]
        store.state_path.write_bytes(b'{"verifier_mutation":true}\n')
        assert store.state_path.read_bytes() != original_bytes
        return True

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="not an arbitrary truthy result",
    ):
        _consume(
            store,
            request,
            candidate,
            evidence,
            stage="intermediate",
            access_hash=access_manifest,
            validator=mutate_then_return_invalid,
        )

    assert store.state_path.read_bytes() == original_bytes
    assert store.load() == before


@pytest.mark.parametrize(
    "validator",
    [
        lambda _evidence, _access, _context, *, authenticated_store_context: True,
        lambda _evidence, _access, _context, *, authenticated_store_context: {
            "semantic_validation_completed": True
        },
    ],
)
def test_arbitrary_truthy_prerequisite_results_are_rejected_without_consumption(
    tmp_path: Path,
    validator,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="arbitrary-prerequisite")
    evidence = _evidence("development", candidate, salt="arbitrary")
    request, access_hash = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="arbitrary",
    )

    before = store.load()
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="not an arbitrary truthy result",
    ):
        _consume(
            store,
            request,
            candidate,
            evidence,
            stage="intermediate",
            access_hash=access_hash,
            validator=validator,
        )
    assert store.load() == before


def test_semantic_result_with_wrong_candidate_binding_is_rejected(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="wrong-binding")
    evidence = _evidence("development", candidate, salt="wrong-binding")
    request, access_hash = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="wrong-binding",
    )

    def wrong_validator(
        _evidence,
        _access,
        context,
        *,
        authenticated_store_context,
    ):
        changed = dict(context)
        changed["candidate_sha256"] = _digest("another candidate")
        return SemanticPrerequisiteValidation.success(
            changed,
            validator_id=VALIDATOR_ID,
            validator_source_sha256=VALIDATOR_SOURCE_SHA256,
            semantic_checks=SEMANTIC_CHECKS,
            semantic_receipt=_store_bound_receipt(
                context, semantic_replay="wrong candidate"
            ),
        )

    before = store.load()
    with pytest.raises(
        SecFilingGemmaRevealStoreError, match="candidate_sha256"
    ):
        _consume(
            store,
            request,
            candidate,
            evidence,
            stage="intermediate",
            access_hash=access_hash,
            validator=wrong_validator,
        )
    assert store.load() == before


def test_semantic_result_cannot_be_reused_for_another_access_manifest(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="cross-access-result")
    evidence = _evidence("development", candidate, salt="cross-access-result")
    request, access_hash = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="cross-access-result",
    )

    def stale_access_validator(
        _evidence,
        _access,
        context,
        *,
        authenticated_store_context,
    ):
        changed = dict(context)
        changed["stage_access_manifest_sha256"] = _digest("another access manifest")
        return SemanticPrerequisiteValidation.success(
            changed,
            validator_id=VALIDATOR_ID,
            validator_source_sha256=VALIDATOR_SOURCE_SHA256,
            semantic_checks=SEMANTIC_CHECKS,
            semantic_receipt=_store_bound_receipt(
                context, semantic_replay="stale access manifest"
            ),
        )

    before = store.load()
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="stage_access_manifest_sha256",
    ):
        _consume(
            store,
            request,
            candidate,
            evidence,
            stage="intermediate",
            access_hash=access_hash,
            validator=stale_access_validator,
        )
    assert store.load() == before


def test_consumption_hashes_the_actual_stage_access_manifest(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="actual-access-manifest")
    evidence = _evidence("development", candidate, salt="actual-access-manifest")
    request, access_manifest = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="actual-access-manifest",
    )
    forged_access = copy.deepcopy(access_manifest)
    forged_access["scope_id"] = "another:intermediate:access"
    forged_access["stage_access_manifest_sha256"] = canonical_sha256(
        {
            key: forged_access[key]
            for key in forged_access
            if key != "stage_access_manifest_sha256"
        }
    )

    before = store.load()
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="altered, stale, non-authorizing, or not current-tip bound",
    ):
        _consume(
            store,
            request,
            candidate,
            evidence,
            stage="intermediate",
            access_hash=forged_access,
        )
    assert store.load() == before


def test_semantic_result_requires_the_exact_frozen_replay_checklist(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="weak-checklist")
    evidence = _evidence("development", candidate, salt="weak-checklist")
    request, access_hash = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="weak-checklist",
    )

    def weak_validator(
        _evidence,
        _access,
        context,
        *,
        authenticated_store_context,
    ):
        return SemanticPrerequisiteValidation.success(
            context,
            validator_id=VALIDATOR_ID,
            validator_source_sha256=VALIDATOR_SOURCE_SHA256,
            semantic_checks=("ok",),
            semantic_receipt=_store_bound_receipt(context, claimed=True),
        )

    before = store.load()
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="exact frozen verifier checklist",
    ):
        _consume(
            store,
            request,
            candidate,
            evidence,
            stage="intermediate",
            access_hash=access_hash,
            validator=weak_validator,
        )
    assert store.load() == before


def test_validator_source_identity_is_derived_from_the_candidate(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="wrong-validator-source")
    evidence = _evidence("development", candidate, salt="wrong-validator-source")
    request, access_manifest = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="wrong-validator-source",
    )

    def wrong_source_validator(
        _evidence,
        _access,
        context,
        *,
        authenticated_store_context,
    ):
        return SemanticPrerequisiteValidation.success(
            context,
            validator_id=VALIDATOR_ID,
            validator_source_sha256=_digest("not candidate-bound"),
            semantic_checks=SEMANTIC_CHECKS,
            semantic_receipt=_store_bound_receipt(context, claimed=True),
        )

    before = store.load()
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="independently pinned validator",
    ):
        _consume(
            store,
            request,
            candidate,
            evidence,
            stage="intermediate",
            access_hash=access_manifest,
            validator=wrong_source_validator,
        )
    assert store.load() == before


def test_fixed_verifier_cannot_redirect_the_authoritative_store_path(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="path-redirection")
    evidence = _evidence("development", candidate, salt="path-redirection")
    request, access_hash = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="path-redirection",
    )
    original_path = store.state_path
    original_bytes = original_path.read_bytes()
    redirected_directory = tmp_path / "redirected-store"
    redirected_directory.mkdir()
    redirected_state = redirected_directory / STATE_FILENAME
    redirected_state.write_bytes(original_bytes)

    def redirecting_validator(
        callback_evidence,
        callback_access,
        context,
        *,
        authenticated_store_context,
    ):
        store._store_directory = redirected_directory
        return _semantic_validator(
            callback_evidence,
            callback_access,
            context,
            authenticated_store_context=authenticated_store_context,
        )

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="paths changed",
    ):
        _consume(
            store,
            request,
            candidate,
            evidence,
            stage="intermediate",
            access_hash=access_hash,
            validator=redirecting_validator,
        )
    assert store.state_path == original_path
    assert original_path.read_bytes() == original_bytes
    assert redirected_state.read_bytes() == original_bytes
    assert store.load()["consumption_ledger"]["entries"] == []


def test_reloading_rehash_consistent_state_replays_semantic_invariants(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="reload-semantic-invariants")
    evidence = _evidence(
        "development", candidate, salt="reload-semantic-invariants"
    )
    request, access_hash = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=evidence,
        salt="reload-semantic-invariants",
    )
    state = _consume(
        store,
        request,
        candidate,
        evidence,
        stage="intermediate",
        access_hash=access_hash,
    )
    forged = copy.deepcopy(state)
    ledger = forged["consumption_ledger"]
    entry = ledger["entries"][0]
    validation = entry["prerequisite_validation"]
    validation["semantic_checks"] = list(REQUIRED_SEMANTIC_CHECKS[:-1])
    validation["result_sha256"] = canonical_sha256(
        {key: validation[key] for key in validation if key != "result_sha256"}
    )
    entry["entry_sha256"] = canonical_sha256(
        {key: entry[key] for key in entry if key != "entry_sha256"}
    )
    ledger["chain"]["tip_sha256"] = entry["entry_sha256"]
    ledger["ledger_sha256"] = canonical_sha256(
        {key: ledger[key] for key in ledger if key != "ledger_sha256"}
    )
    forged["state_sha256"] = canonical_sha256(
        {key: forged[key] for key in forged if key != "state_sha256"}
    )
    store.state_path.write_text(
        json.dumps(forged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="Stored semantic checks",
    ):
        store.load()


def test_final_consumption_increments_actual_touch_exactly_once(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="final-count")
    assert registered["consumption_ledger"]["chain"][
        "actual_final_touch_count"
    ] == 0

    development_evidence = _evidence(
        "development", candidate, salt="final-count-development"
    )
    intermediate_request, intermediate_access = _grant_request(
        registered,
        candidate,
        stage="intermediate",
        evidence=development_evidence,
    )
    intermediate_bundle = _consume_with_grant(
        store,
        intermediate_request,
        candidate,
        development_evidence,
        stage="intermediate",
        access_manifest=intermediate_access,
    )
    after_intermediate = intermediate_bundle["authenticated_store_snapshot"]
    assert after_intermediate["consumption_ledger"]["chain"][
        "actual_final_touch_count"
    ] == 0

    intermediate_evidence = _evidence(
        "intermediate", candidate, salt="final-count-final"
    )
    final_request, final_access = _request(
        after_intermediate,
        candidate,
        stage="final",
        evidence=intermediate_evidence,
        salt="final-count-final",
    )
    after_final = _consume(
        store,
        final_request,
        candidate,
        intermediate_evidence,
        stage="final",
        access_hash=final_access,
    )
    chain = after_final["consumption_ledger"]["chain"]
    assert chain["consumed_request_count"] == 2
    assert chain["actual_final_touch_count"] == 1
    assert chain["repository_final_touch_count_lower_bound"] == 11
    assert after_final["consumption_ledger"]["entries"][-1][
        "final_touch_delta"
    ] == 1
    assert after_final["consumption_ledger"]["entries"][-1][
        "cumulative_actual_final_touch_count"
    ] == 1

    with pytest.raises(SecFilingGemmaRevealStoreError, match="already consumed"):
        _consume(
            store,
            final_request,
            candidate,
            intermediate_evidence,
            stage="final",
            access_hash=final_access,
        )
    assert store.load()["consumption_ledger"]["chain"][
        "actual_final_touch_count"
    ] == 1


def test_final_verifier_receives_exact_internal_parent_consumption_binding(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="final-parent-context")
    development_evidence = _evidence(
        "development", candidate, salt="final-parent-context-development"
    )
    intermediate_request, intermediate_access = _grant_request(
        registered,
        candidate,
        stage="intermediate",
        evidence=development_evidence,
    )
    intermediate_bundle = _consume_with_grant(
        store,
        intermediate_request,
        candidate,
        development_evidence,
        stage="intermediate",
        access_manifest=intermediate_access,
    )
    intermediate_state = intermediate_bundle["authenticated_store_snapshot"]
    intermediate_evidence = _evidence(
        "intermediate", candidate, salt="final-parent-context-final"
    )
    final_request, final_access = _request(
        intermediate_state,
        candidate,
        stage="final",
        evidence=intermediate_evidence,
        salt="final-parent-context-final",
    )

    for public_method in (
        SecFilingGemmaRevealStore.consume_request,
        SecFilingGemmaRevealStore.consume_request_and_issue_authorization_grant,
    ):
        public_signature = inspect.signature(public_method)
        assert "authenticated_store_context" not in public_signature.parameters
        assert all(
            parameter.kind is not inspect.Parameter.VAR_KEYWORD
            for parameter in public_signature.parameters.values()
        )
    state_before_override = store.state_path.read_bytes()
    tip_before_override = store.current_tip_anchor_path.read_bytes()
    with pytest.raises(TypeError, match="authenticated_store_context"):
        store.consume_request(
            final_request,
            candidate_manifest=candidate,
            stage="final",
            stage_access_manifest=final_access,
            prerequisite_stage_evidence=intermediate_evidence,
            authenticated_store_context={"forged": True},
        )
    assert store.state_path.read_bytes() == state_before_override
    assert store.current_tip_anchor_path.read_bytes() == tip_before_override

    tip_before_final = store.load_current_tip_anchor()
    observed: dict[str, dict] = {}

    def capture_final_context(
        evidence,
        access_manifest,
        expected_context,
        *,
        authenticated_store_context,
    ):
        assert expected_context["stage"] == "final"
        observed["expected_context"] = copy.deepcopy(expected_context)
        observed["authenticated_store_context"] = copy.deepcopy(
            authenticated_store_context
        )
        observed["post_final_pin_state"] = json.loads(
            store.state_path.read_bytes()
        )
        observed["post_final_pin_tip"] = json.loads(
            store.current_tip_anchor_path.read_bytes()
        )
        return _semantic_validator(
            evidence,
            access_manifest,
            expected_context,
            authenticated_store_context=authenticated_store_context,
        )

    consumed = _consume(
        store,
        final_request,
        candidate,
        intermediate_evidence,
        stage="final",
        access_hash=final_access,
        validator=capture_final_context,
    )
    assert consumed["consumption_ledger"]["chain"]["consumed_request_count"] == 2

    expected_context = observed["expected_context"]
    store_context = observed["authenticated_store_context"]
    post_pin_state = observed["post_final_pin_state"]
    post_pin_tip = observed["post_final_pin_tip"]
    assert post_pin_state == intermediate_state
    assert post_pin_tip["revision"] == tip_before_final["revision"] + 1
    assert post_pin_tip["authorization_bundles"][
        intermediate_request["request_sha256"]
    ] == intermediate_bundle
    assert store_context["trusted_stage_content_pin"] == post_pin_tip[
        "trusted_stage_content_pins"
    ][final_request["request_sha256"]]
    store_context_body = {
        key: store_context[key]
        for key in store_context
        if key != "authenticated_store_context_sha256"
    }
    assert canonical_sha256(store_context_body) == store_context[
        "authenticated_store_context_sha256"
    ]
    assert expected_context["authenticated_store_context_sha256"] == (
        store_context["authenticated_store_context_sha256"]
    )

    binding = store_context["parent_consumption_binding"]
    parent_entry = post_pin_state["consumption_ledger"]["entries"][-1]
    parent_request = parent_entry["request"]
    parent_validation = parent_entry["prerequisite_validation"]
    parent_audit = parent_validation["semantic_receipt"]
    parent_pin = post_pin_tip["trusted_stage_content_pins"][
        parent_request["request_sha256"]
    ]
    persisted_bundle = post_pin_tip["authorization_bundles"][
        parent_request["request_sha256"]
    ]
    parent_grant = persisted_bundle["authorization_grant"]
    parent_store_pin = persisted_bundle["store_state_pin"]
    assert parent_entry["entry_sha256"] == post_pin_state[
        "consumption_ledger"
    ]["chain"]["tip_sha256"]
    assert parent_validation["semantic_receipt_sha256"] == canonical_sha256(
        parent_audit
    )
    assert parent_audit["audit_receipt_sha256"] == canonical_sha256(
        {
            key: parent_audit[key]
            for key in parent_audit
            if key != "audit_receipt_sha256"
        }
    )
    assert parent_audit["trusted_stage_content_pin_sha256"] == parent_pin[
        "pin_sha256"
    ]
    assert parent_audit[
        "trusted_stage_content_authentication_receipt_sha256"
    ] == parent_audit["trusted_stage_content_authentication"][
        "authentication_receipt_sha256"
    ]

    parent_expected_context = {
        "prerequisite_stage": parent_request["prerequisite_stage"],
        "prerequisite_stage_evidence_sha256": parent_request[
            "prerequisite_stage_evidence_sha256"
        ],
        "attempt_id": parent_request["attempt_id"],
        "candidate_sha256": parent_request["candidate_sha256"],
        "candidate_design_sha256": parent_request[
            "candidate_design_sha256"
        ],
        "registry_entry_sha256": parent_request["registry_entry_sha256"],
        "request_sha256": parent_request["request_sha256"],
        "stage": parent_request["stage"],
        "stage_access_manifest_sha256": parent_request[
            "stage_access_manifest_sha256"
        ],
        "registry_sha256": parent_request["registry_sha256"],
        "registry_tip_sha256": parent_request["registry_tip_sha256"],
        "trusted_stage_content_pin_sha256": parent_pin["pin_sha256"],
        "trusted_stage_content_authentication_receipt_sha256": parent_audit[
            "trusted_stage_content_authentication_receipt_sha256"
        ],
        "parent_consumption_binding_sha256": None,
        "authenticated_store_context_sha256": parent_audit[
            "authenticated_store_context_sha256"
        ],
    }
    child_fields = (
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
    expected_child = {field: final_request[field] for field in child_fields}
    expected_parent = {
        "entry_sha256": parent_entry["entry_sha256"],
        "sequence": parent_entry["sequence"],
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
        "candidate_design_sha256": parent_request[
            "candidate_design_sha256"
        ],
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
        "audit_receipt_sha256": parent_audit["audit_receipt_sha256"],
        "trusted_stage_content_pin": parent_pin,
        "trusted_stage_content_pin_sha256": parent_pin["pin_sha256"],
        "authorization_bundle_sha256": persisted_bundle["bundle_sha256"],
        "authorization_grant_sha256": parent_grant[
            "authorization_grant_sha256"
        ],
        "store_pin_sha256": parent_store_pin["store_pin_sha256"],
    }
    expected_authenticated_tip = {
        "store_state_sha256": post_pin_state["state_sha256"],
        "state_snapshot_bytes_sha256": post_pin_tip[
            "state_snapshot_bytes_sha256"
        ],
        "state_snapshot_byte_count": post_pin_tip[
            "state_snapshot_byte_count"
        ],
        "consumption_ledger_sha256": post_pin_state["consumption_ledger"][
            "ledger_sha256"
        ],
        "consumption_ledger_tip_sha256": post_pin_state[
            "consumption_ledger"
        ]["chain"]["tip_sha256"],
        "consumed_request_count": post_pin_state["consumption_ledger"][
            "chain"
        ]["consumed_request_count"],
        "current_tip_anchor_sha256": post_pin_tip["tip_anchor_sha256"],
        "current_tip_revision": post_pin_tip["revision"],
    }
    expected_binding_body = {
        "schema_version": (
            reveal_store_module.PARENT_CONSUMPTION_BINDING_SCHEMA_VERSION
        ),
        "contract_version": CONTRACT_VERSION,
        "binding_kind": "exact_prior_intermediate_consumption_and_grant",
        "child_request": expected_child,
        "parent_consumption": expected_parent,
        "authenticated_preconsumption_tip": expected_authenticated_tip,
    }
    expected_binding = {
        **expected_binding_body,
        "parent_consumption_binding_sha256": canonical_sha256(
            expected_binding_body
        ),
    }
    assert binding == expected_binding
    assert expected_context["parent_consumption_binding_sha256"] == binding[
        "parent_consumption_binding_sha256"
    ]


def test_final_requires_exact_predecessor_authorization_bundle_before_pin(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="final-parent-bundle")
    development_evidence = _evidence(
        "development", candidate, salt="final-parent-bundle-development"
    )
    intermediate_request, intermediate_access = _request(
        registered,
        candidate,
        stage="intermediate",
        evidence=development_evidence,
        salt="final-parent-bundle-intermediate",
    )
    after_intermediate = _consume(
        store,
        intermediate_request,
        candidate,
        development_evidence,
        stage="intermediate",
        access_hash=intermediate_access,
    )
    assert intermediate_request["request_sha256"] not in store.load_current_tip_anchor()[
        "authorization_bundles"
    ]

    intermediate_evidence = _evidence(
        "intermediate", candidate, salt="final-parent-bundle-final"
    )
    final_request, final_access = _request(
        after_intermediate,
        candidate,
        stage="final",
        evidence=intermediate_evidence,
        salt="final-parent-bundle-final",
    )
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()

    def verifier_must_not_run(*_args, **_kwargs):
        raise AssertionError(
            "final predecessor authorization must fail before verifier execution"
        )

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="persisted authorization bundle",
    ):
        _consume(
            store,
            final_request,
            candidate,
            intermediate_evidence,
            stage="final",
            access_hash=final_access,
            validator=verifier_must_not_run,
        )

    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes
    assert final_request["request_sha256"] not in store.load_current_tip_anchor()[
        "trusted_stage_content_pins"
    ]


def test_final_request_cannot_be_consumed_before_intermediate_request(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(store, salt="final-before-intermediate")
    evidence = _evidence("intermediate", candidate, salt="too-early-final")
    request, access_hash = _request(
        registered,
        candidate,
        stage="final",
        evidence=evidence,
        salt="too-early-final",
    )
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()

    with pytest.raises(
        SecFilingGemmaRevealStoreError, match="intermediate predecessor"
    ):
        _consume(
            store,
            request,
            candidate,
            evidence,
            stage="final",
            access_hash=access_hash,
        )
    assert store.load()["consumption_ledger"]["chain"][
        "actual_final_touch_count"
    ] == 0
    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes
    assert request["request_sha256"] not in store.load_current_tip_anchor()[
        "trusted_stage_content_pins"
    ]


def test_interrupted_atomic_temp_file_is_never_authoritative(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.store_directory.mkdir(parents=True)
    interrupted = store.store_directory / (
        f".{STATE_FILENAME}.{'a' * 32}.tmp"
    )
    interrupted.write_text('{"forged":true}\n', encoding="utf-8")

    state = store.initialize()

    assert not interrupted.exists()
    assert state == store.load()
    assert state["latest_registry_pin"]["registered_entry_count"] == 0


def _run_git(repository: Path, *arguments: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    if completed.returncode != 0:
        pytest.fail(completed.stderr.decode("utf-8", errors="replace"))


def test_worktree_anchor_mismatch_from_tracked_head_is_rejected(
    tmp_path: Path,
) -> None:
    fake_repo = tmp_path / "tracked-anchor-repo"
    fake_repo.mkdir()
    declaration = historical_reveal_declaration()
    relative_paths = [
        "docs/protocol_evidence/sec_gemma_reveal_registry_initial_pin.json",
        declaration["migration_source"],
    ]
    for relative in relative_paths:
        source = REPO_ROOT.joinpath(*Path(relative).parts)
        destination = fake_repo.joinpath(*Path(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    _run_git(fake_repo, "init", "--quiet")
    _run_git(fake_repo, "add", "--", *relative_paths)
    _run_git(
        fake_repo,
        "-c",
        "user.name=Reveal Store Test",
        "-c",
        "user.email=reveal-store@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "track protocol anchors",
    )

    pin_path = fake_repo / relative_paths[0]
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    pin["registry_pin"]["historical_final_reveal_count_lower_bound"] = 9
    pin_path.write_text(
        json.dumps(pin, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    store = SecFilingGemmaRevealStore(
        repository_root=fake_repo,
        store_directory=tmp_path / "mismatch-store",
    )

    with pytest.raises(
        SecFilingGemmaRevealStoreError, match="does not match Git HEAD"
    ):
        store.initialize()


def test_store_directory_link_is_rejected_when_supported(tmp_path: Path) -> None:
    real_directory = tmp_path / "real-store"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked-store"
    try:
        linked_directory.symlink_to(real_directory, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable for this test user")

    store = SecFilingGemmaRevealStore(
        repository_root=REPO_ROOT,
        store_directory=linked_directory,
    )
    with pytest.raises(
        SecFilingGemmaRevealStoreError, match="link or reparse point"
    ):
        store.initialize()
