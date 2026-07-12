from __future__ import annotations

import copy
import hashlib
from unittest.mock import patch

import pytest

from agent_benchmark.sec_filing_gemma_contract import (
    CONTRACT_VERSION,
    REQUIRED_STAGE_VERIFIER_CHECKS,
    canonical_sha256,
)
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    REVEAL_REQUEST_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_stage_access import (
    STAGE_ACCESS_MANIFEST_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_stage_authorization import (
    CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION,
    CONSUMED_STAGE_AUTHORIZATION_GRANT_SCHEMA_VERSION,
    CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION,
    CONSUMED_STAGE_STORE_PIN_SCHEMA_VERSION,
    CONSUMPTION_ENTRY_SCHEMA_VERSION,
    CONSUMPTION_LEDGER_SCHEMA_VERSION,
    SEMANTIC_PREREQUISITE_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
    TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION,
    TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION,
    SecFilingGemmaStageAuthorizationError,
    authenticate_reveal_store_trusted_stage_content_pin,
    build_reveal_store_current_tip_anchor,
    build_consumed_stage_authorization_grant,
    build_consumed_stage_output_receipt,
    derive_consumed_stage_store_state_pin,
    derive_reveal_store_trusted_stage_content_pin,
    validate_consumed_stage_authorization_grant,
    validate_consumed_stage_output_receipt,
    validate_consumed_stage_store_state_pin,
    validate_reveal_store_current_tip_anchor_transition,
    validate_trusted_stage_content_authentication_receipt,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _genesis(anchor: dict) -> str:
    return canonical_sha256(
        {
            "schema_version": "aapl-sec-gemma-consumed-request-genesis-v1",
            "contract_version": CONTRACT_VERSION,
            "store_anchor_sha256": canonical_sha256(anchor),
        }
    )


def _access(*, attempt: str, candidate: str, stage: str) -> dict:
    prerequisite = "development" if stage == "intermediate" else "intermediate"
    body = {
        "schema_version": STAGE_ACCESS_MANIFEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "transition": {
            "prerequisite_stage": prerequisite,
            "requested_stage": stage,
            "transition_ordinal": 1 if stage == "intermediate" else 2,
            "single_use_consumption_required": True,
            "stage_reuse_permitted": False,
        },
        "prerequisite_evidence_pin": {
            "stage": prerequisite,
            "content_manifest_sha256": _h(
                f"{attempt}:{prerequisite}:content-manifest"
            ),
            "stage_artifact_sha256": _h(
                f"{attempt}:{prerequisite}:stage-artifact"
            ),
            "external_seal_receipt_sha256": _h(
                f"{attempt}:{prerequisite}:external-seal"
            ),
        },
        "candidate": {
            "candidate_sha256": candidate,
            "candidate_design_sha256": _h(f"{attempt}:design"),
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
    }
    return {**body, "stage_access_manifest_sha256": canonical_sha256(body)}


def _entry(
    *,
    sequence: int,
    prior_tip: str,
    stage: str = "intermediate",
) -> dict:
    attempt = f"{CONTRACT_VERSION}-attempt-{sequence:03d}"
    candidate = _h(f"candidate:{sequence}")
    access = _access(attempt=attempt, candidate=candidate, stage=stage)
    prerequisite = "development" if stage == "intermediate" else "intermediate"
    request_body = {
        "schema_version": REVEAL_REQUEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "registry_sha256": _h("registry"),
        "registry_tip_sha256": _h("registry tip"),
        "registered_entry_count": sequence,
        "historical_final_reveal_count_lower_bound": 10,
        "stage": stage,
        "stage_access_manifest_sha256": access["stage_access_manifest_sha256"],
        "prerequisite_stage": prerequisite,
        "prerequisite_stage_evidence_sha256": _h(
            f"{sequence}:{prerequisite}:evidence"
        ),
        "attempt_id": attempt,
        "candidate_sha256": candidate,
        "candidate_design_sha256": access["candidate"][
            "candidate_design_sha256"
        ],
        "registry_entry_sha256": _h(f"registry entry:{sequence}"),
        "request_scope": "single_candidate_single_stage",
        "authorizes_outcome_access": False,
        "effectful_atomic_single_use_consumption_required": True,
        "cross_attempt_comparison_permitted": False,
        "cross_attempt_winner_selection_permitted": False,
        "globally_pristine_claim": False,
    }
    request = {
        **request_body,
        "request_sha256": canonical_sha256(request_body),
    }
    semantic_receipt = {
        "schema_version": "synthetic-semantic-receipt-v1",
        "request_sha256": request["request_sha256"],
    }
    validation_body = {
        "schema_version": SEMANTIC_PREREQUISITE_SCHEMA_VERSION,
        "validation_kind": "independent_semantic_prerequisite_replay",
        "validator_id": "synthetic-authoritative-verifier-v1",
        "validator_source_sha256": _h("validator source"),
        "prerequisite_stage": prerequisite,
        "prerequisite_stage_evidence_sha256": request[
            "prerequisite_stage_evidence_sha256"
        ],
        "attempt_id": attempt,
        "candidate_sha256": candidate,
        "candidate_design_sha256": request["candidate_design_sha256"],
        "registry_entry_sha256": request["registry_entry_sha256"],
        "request_sha256": request["request_sha256"],
        "requested_stage": stage,
        "stage_access_manifest_sha256": access["stage_access_manifest_sha256"],
        "registry_sha256": request["registry_sha256"],
        "registry_tip_sha256": request["registry_tip_sha256"],
        "semantic_checks": list(REQUIRED_STAGE_VERIFIER_CHECKS),
        "semantic_receipt": semantic_receipt,
        "semantic_receipt_sha256": canonical_sha256(semantic_receipt),
        "semantic_validation_completed": True,
        "authorizes_outcome_access": False,
    }
    validation = {
        **validation_body,
        "result_sha256": canonical_sha256(validation_body),
    }
    entry_body = {
        "schema_version": CONSUMPTION_ENTRY_SCHEMA_VERSION,
        "sequence": sequence,
        "request_sha256": request["request_sha256"],
        "request": request,
        "stage_access_manifest": access,
        "stage": stage,
        "attempt_id": attempt,
        "candidate_sha256": candidate,
        "registry_entry_sha256": request["registry_entry_sha256"],
        "prerequisite_validation": validation,
        "prior_tip_sha256": prior_tip,
        "final_touch_delta": 1 if stage == "final" else 0,
        "cumulative_actual_final_touch_count": 0,
    }
    return {**entry_body, "entry_sha256": canonical_sha256(entry_body)}


def _snapshot(*, entry_count: int = 1) -> dict:
    anchor = {"schema_version": "synthetic-store-anchor-v1", "root": _h("anchor")}
    current_tip = _genesis(anchor)
    entries: list[dict] = []
    for sequence in range(1, entry_count + 1):
        entry = _entry(sequence=sequence, prior_tip=current_tip)
        entries.append(entry)
        current_tip = entry["entry_sha256"]
    ledger_body = {
        "schema_version": CONSUMPTION_LEDGER_SCHEMA_VERSION,
        "entries": entries,
        "chain": {
            "genesis_tip_sha256": _genesis(anchor),
            "tip_sha256": current_tip,
            "consumed_request_count": len(entries),
            "actual_final_touch_count": 0,
            "historical_final_reveal_count_lower_bound": 10,
            "repository_final_touch_count_lower_bound": 10,
        },
    }
    ledger = {**ledger_body, "ledger_sha256": canonical_sha256(ledger_body)}
    registry = {
        "schema_version": "synthetic-registry-v1",
        "chain": {
            "tip_sha256": _h("registry tip"),
            "registered_entry_count": max(1, entry_count),
        },
        "registry_sha256": _h("registry"),
    }
    registry_pin = {
        "schema_version": "synthetic-registry-pin-v1",
        "registry_sha256": registry["registry_sha256"],
        "tip_sha256": registry["chain"]["tip_sha256"],
        "registered_entry_count": registry["chain"]["registered_entry_count"],
    }
    state_body = {
        "schema_version": STORE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "anchor": anchor,
        "latest_registry": registry,
        "latest_registry_pin": registry_pin,
        "consumption_ledger": ledger,
    }
    return {**state_body, "state_sha256": canonical_sha256(state_body)}


def _grant_context(entry_count: int = 1) -> tuple[dict, dict, dict, dict, dict]:
    state = _snapshot(entry_count=entry_count)
    pin = derive_consumed_stage_store_state_pin(state)
    entry = state["consumption_ledger"]["entries"][-1]
    grant = build_consumed_stage_authorization_grant(
        authenticated_store_snapshot=state,
        external_store_state_pin=pin,
        expected_new_consumption_entry_sha256=entry["entry_sha256"],
    )
    bundle_body = {
        "schema_version": CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION,
        "authenticated_store_snapshot": state,
        "store_state_pin": pin,
        "authorization_grant": grant,
    }
    bundle = {**bundle_body, "bundle_sha256": canonical_sha256(bundle_body)}
    current_tip_anchor = build_reveal_store_current_tip_anchor(
        state,
        revision=entry_count,
        previous_tip_anchor_sha256=(None if entry_count == 0 else _h("prior tip")),
        authorization_bundles={entry["request_sha256"]: bundle},
    )
    return state, pin, entry, grant, current_tip_anchor


def _rehash(value: dict, field: str) -> None:
    value[field] = canonical_sha256(
        {key: child for key, child in value.items() if key != field}
    )


def _trusted_pin_context() -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    state = _snapshot(entry_count=0)
    prospective_entry = _entry(
        sequence=1,
        prior_tip=state["consumption_ledger"]["chain"]["tip_sha256"],
    )
    request = prospective_entry["request"]
    access = prospective_entry["stage_access_manifest"]
    pin = derive_reveal_store_trusted_stage_content_pin(
        state,
        reveal_request=request,
        stage_access_manifest=access,
    )
    prior_tip = build_reveal_store_current_tip_anchor(
        state,
        revision=0,
        previous_tip_anchor_sha256=None,
        authorization_bundles={},
        trusted_stage_content_pins={},
    )
    pinned_tip = build_reveal_store_current_tip_anchor(
        state,
        revision=1,
        previous_tip_anchor_sha256=prior_tip["tip_anchor_sha256"],
        authorization_bundles={},
        trusted_stage_content_pins={request["request_sha256"]: pin},
    )
    receipt = authenticate_reveal_store_trusted_stage_content_pin(
        state,
        pinned_tip,
        reveal_request=request,
        stage_access_manifest=access,
    )
    return state, request, access, pin, prior_tip, pinned_tip, receipt


def _validate(
    state: dict, pin: dict, entry: dict, grant: dict, current_tip_anchor: dict
) -> str:
    request = entry["request"]
    return validate_consumed_stage_authorization_grant(
        grant,
        authenticated_store_snapshot=state,
        external_store_state_pin=pin,
        independent_current_tip_anchor=current_tip_anchor,
        expected_consumption_entry_sha256=entry["entry_sha256"],
        expected_request_sha256=request["request_sha256"],
        expected_candidate_sha256=request["candidate_sha256"],
        expected_stage=request["stage"],
        expected_prerequisite_stage_evidence_sha256=request[
            "prerequisite_stage_evidence_sha256"
        ],
        expected_stage_access_manifest_sha256=request[
            "stage_access_manifest_sha256"
        ],
        expected_output_namespace=entry["stage_access_manifest"]["output"][
            "namespace"
        ],
    )


def _output_binding(grant: dict, *, salt: str = "first output") -> dict:
    return {
        "output_stage_evidence_schema_version": "stage-evidence-test-v1",
        "output_stage_evidence_sha256": _h(f"{salt}:evidence"),
        "output_stage_evidence_document_sha256": _h(f"{salt}:document"),
        "output_stage_evidence_canonical_byte_count": 4096,
        "output_stage_evidence_prerequisite_stage": grant["stage"],
        "output_parent_stage_evidence_sha256": grant[
            "prerequisite_stage_evidence_sha256"
        ],
        "output_candidate_sha256": grant["candidate_sha256"],
    }


def test_exact_ledger_tip_mints_compact_no_outcome_grant() -> None:
    state, pin, entry, grant, current_tip = _grant_context()
    assert pin["schema_version"] == CONSUMED_STAGE_STORE_PIN_SCHEMA_VERSION
    assert grant["schema_version"] == CONSUMED_STAGE_AUTHORIZATION_GRANT_SCHEMA_VERSION
    assert grant["consumption_entry_sha256"] == entry["entry_sha256"]
    assert grant["consumption_ledger_tip_sha256"] == entry["entry_sha256"]
    assert grant["output_namespace"] == entry["stage_access_manifest"]["output"][
        "namespace"
    ]
    assert grant["outcomes_included"] is False
    assert grant["market_values_included"] is False
    assert all(not isinstance(value, (dict, list)) for value in grant.values())
    assert _validate(state, pin, entry, grant, current_tip) == grant[
        "authorization_grant_sha256"
    ]


def test_first_output_receipt_is_exact_current_tip_append_and_replays() -> None:
    state, _pin, entry, grant, current_tip = _grant_context()
    request_hash = entry["request_sha256"]
    bundle = current_tip["authorization_bundles"][request_hash]
    output_binding = _output_binding(grant)
    receipt = build_consumed_stage_output_receipt(bundle, **output_binding)
    next_tip = build_reveal_store_current_tip_anchor(
        state,
        revision=current_tip["revision"] + 1,
        previous_tip_anchor_sha256=current_tip["tip_anchor_sha256"],
        authorization_bundles=current_tip["authorization_bundles"],
        trusted_stage_content_pins=current_tip["trusted_stage_content_pins"],
        consumed_stage_output_receipts={request_hash: receipt},
    )
    prior, validated_next = validate_reveal_store_current_tip_anchor_transition(
        current_tip,
        next_tip,
    )
    assert prior == current_tip
    assert validated_next == next_tip
    assert receipt["schema_version"] == CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION
    assert receipt["consumption_entry_sha256"] == entry["entry_sha256"]
    assert receipt["authorization_grant_sha256"] == grant[
        "authorization_grant_sha256"
    ]
    assert receipt["output_namespace"] == grant["output_namespace"]
    assert validate_consumed_stage_output_receipt(
        receipt,
        authenticated_store_snapshot=state,
        independent_current_tip_anchor=next_tip,
        authorization_bundle=bundle,
        **output_binding,
    ) == receipt["output_receipt_sha256"]


def test_output_receipt_rejects_substitution_and_non_dedicated_append() -> None:
    state, _pin, entry, grant, current_tip = _grant_context()
    request_hash = entry["request_sha256"]
    bundle = current_tip["authorization_bundles"][request_hash]
    output_binding = _output_binding(grant)
    receipt = build_consumed_stage_output_receipt(bundle, **output_binding)
    persisted_tip = build_reveal_store_current_tip_anchor(
        state,
        revision=current_tip["revision"] + 1,
        previous_tip_anchor_sha256=current_tip["tip_anchor_sha256"],
        authorization_bundles=current_tip["authorization_bundles"],
        trusted_stage_content_pins=current_tip["trusted_stage_content_pins"],
        consumed_stage_output_receipts={request_hash: receipt},
    )
    changed_binding = dict(output_binding)
    changed_binding["output_stage_evidence_document_sha256"] = _h(
        "substituted document"
    )
    with pytest.raises(SecFilingGemmaStageAuthorizationError, match="differs"):
        validate_consumed_stage_output_receipt(
            receipt,
            authenticated_store_snapshot=state,
            independent_current_tip_anchor=persisted_tip,
            authorization_bundle=bundle,
            **changed_binding,
        )

    # A receipt may not hitchhike on a state/consumption transition.
    newer_state = _snapshot(entry_count=2)
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="dedicated tip-only",
    ):
        mixed_tip = build_reveal_store_current_tip_anchor(
            newer_state,
            revision=current_tip["revision"] + 1,
            previous_tip_anchor_sha256=current_tip["tip_anchor_sha256"],
            authorization_bundles=current_tip["authorization_bundles"],
            trusted_stage_content_pins=current_tip["trusted_stage_content_pins"],
            consumed_stage_output_receipts={request_hash: receipt},
        )
        validate_reveal_store_current_tip_anchor_transition(current_tip, mixed_tip)


def test_output_receipt_cannot_append_after_its_grant_tip_is_stale() -> None:
    state, _pin, entry, grant, current_tip = _grant_context()
    request_hash = entry["request_sha256"]
    bundle = current_tip["authorization_bundles"][request_hash]
    receipt = build_consumed_stage_output_receipt(
        bundle,
        **_output_binding(grant),
    )

    advanced_state = _snapshot(entry_count=2)
    advanced_tip = build_reveal_store_current_tip_anchor(
        advanced_state,
        revision=current_tip["revision"] + 1,
        previous_tip_anchor_sha256=current_tip["tip_anchor_sha256"],
        authorization_bundles=current_tip["authorization_bundles"],
        trusted_stage_content_pins=current_tip["trusted_stage_content_pins"],
        consumed_stage_output_receipts={},
    )
    validate_reveal_store_current_tip_anchor_transition(current_tip, advanced_tip)

    polluted_tip = build_reveal_store_current_tip_anchor(
        advanced_state,
        revision=advanced_tip["revision"] + 1,
        previous_tip_anchor_sha256=advanced_tip["tip_anchor_sha256"],
        authorization_bundles=advanced_tip["authorization_bundles"],
        trusted_stage_content_pins=advanced_tip["trusted_stage_content_pins"],
        consumed_stage_output_receipts={request_hash: receipt},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="exact current grant tip",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            advanced_tip,
            polluted_tip,
        )


def test_pin_snapshot_and_grant_tampering_fail_closed() -> None:
    state, pin, entry, grant, current_tip = _grant_context()

    changed_pin = copy.deepcopy(pin)
    changed_pin["store_state_sha256"] = _h("another state")
    changed_pin["store_pin_sha256"] = canonical_sha256(
        {key: changed_pin[key] for key in changed_pin if key != "store_pin_sha256"}
    )
    with pytest.raises(SecFilingGemmaStageAuthorizationError, match="authenticate"):
        validate_consumed_stage_store_state_pin(state, changed_pin)

    changed_state = copy.deepcopy(state)
    changed_state["anchor"]["root"] = _h("tampered anchor")
    changed_state["state_sha256"] = canonical_sha256(
        {key: changed_state[key] for key in changed_state if key != "state_sha256"}
    )
    with pytest.raises(SecFilingGemmaStageAuthorizationError):
        _validate(changed_state, pin, entry, grant, current_tip)

    changed_grant = copy.deepcopy(grant)
    changed_grant["output_namespace"] = "aapl-sec-gemma-substituted-intermediate"
    changed_grant["authorization_grant_sha256"] = canonical_sha256(
        {
            key: changed_grant[key]
            for key in changed_grant
            if key != "authorization_grant_sha256"
        }
    )
    with pytest.raises(SecFilingGemmaStageAuthorizationError, match="differs"):
        _validate(state, pin, entry, changed_grant, current_tip)


def test_historical_entry_and_stale_grant_replay_are_rejected() -> None:
    old_state, old_pin, old_entry, old_grant, old_current_tip = _grant_context()
    new_state = _snapshot(entry_count=2)
    new_pin = derive_consumed_stage_store_state_pin(new_state)

    with pytest.raises(SecFilingGemmaStageAuthorizationError, match="newly appended"):
        build_consumed_stage_authorization_grant(
            authenticated_store_snapshot=new_state,
            external_store_state_pin=new_pin,
            expected_new_consumption_entry_sha256=old_entry["entry_sha256"],
        )

    with pytest.raises(SecFilingGemmaStageAuthorizationError):
        validate_consumed_stage_authorization_grant(
            old_grant,
            authenticated_store_snapshot=new_state,
            external_store_state_pin=new_pin,
            independent_current_tip_anchor=old_current_tip,
            expected_consumption_entry_sha256=old_entry["entry_sha256"],
            expected_request_sha256=old_entry["request_sha256"],
            expected_candidate_sha256=old_entry["candidate_sha256"],
            expected_stage=old_entry["stage"],
            expected_prerequisite_stage_evidence_sha256=old_entry["request"][
                "prerequisite_stage_evidence_sha256"
            ],
            expected_stage_access_manifest_sha256=old_entry["request"][
                "stage_access_manifest_sha256"
            ],
            expected_output_namespace=old_entry["stage_access_manifest"]["output"][
                "namespace"
            ],
        )
    assert validate_consumed_stage_store_state_pin(old_state, old_pin) == old_pin


def test_old_snapshot_and_its_bundled_pin_fail_against_a_newer_current_tip() -> None:
    old_state, old_pin, old_entry, old_grant, old_current_tip = _grant_context()
    new_state = _snapshot(entry_count=2)
    new_current_tip = build_reveal_store_current_tip_anchor(
        new_state,
        revision=old_current_tip["revision"] + 1,
        previous_tip_anchor_sha256=old_current_tip["tip_anchor_sha256"],
        authorization_bundles=old_current_tip["authorization_bundles"],
    )
    request = old_entry["request"]

    # The old pin still agrees with its old snapshot, which is exactly why it
    # is not allowed to act as the independent trust root.
    assert validate_consumed_stage_store_state_pin(old_state, old_pin) == old_pin
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="current-tip anchor does not authenticate",
    ):
        validate_consumed_stage_authorization_grant(
            old_grant,
            authenticated_store_snapshot=old_state,
            external_store_state_pin=old_pin,
            independent_current_tip_anchor=new_current_tip,
            expected_consumption_entry_sha256=old_entry["entry_sha256"],
            expected_request_sha256=request["request_sha256"],
            expected_candidate_sha256=request["candidate_sha256"],
            expected_stage=request["stage"],
            expected_prerequisite_stage_evidence_sha256=request[
                "prerequisite_stage_evidence_sha256"
            ],
            expected_stage_access_manifest_sha256=request[
                "stage_access_manifest_sha256"
            ],
            expected_output_namespace=old_entry["stage_access_manifest"]["output"][
                "namespace"
            ],
        )


def test_authorization_inputs_reject_mapping_subclasses_without_running_hooks() -> None:
    state, pin, entry, grant, current_tip = _grant_context()
    hook_ran = False

    class HostileMapping(dict):
        def items(self):
            nonlocal hook_ran
            hook_ran = True
            raise AssertionError("caller mapping hook executed")

        def __iter__(self):
            nonlocal hook_ran
            hook_ran = True
            raise AssertionError("caller mapping hook executed")

    changed = dict(grant)
    changed["authorization_scope"] = HostileMapping()
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="exact built-in",
    ):
        _validate(state, pin, entry, changed, current_tip)
    assert hook_ran is False


@pytest.mark.parametrize("attack", ["deep", "wide", "huge_int"])
def test_public_authorization_boundary_rejects_unbounded_inputs_before_hashing(
    attack: str,
) -> None:
    hostile: dict = {}
    if attack == "deep":
        cursor = hostile
        for _ in range(64):
            child: dict = {}
            cursor["nested"] = child
            cursor = child
    elif attack == "wide":
        hostile["wide"] = list(range(20_001))
    else:
        hostile["huge_integer"] = 1 << 512

    with patch(
        "agent_benchmark.sec_filing_gemma_stage_authorization.canonical_sha256",
        side_effect=AssertionError("unbounded input reached canonical hashing"),
    ), pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="bounded exact-JSON authorization limits",
    ):
        derive_consumed_stage_store_state_pin(hostile)


def test_cross_stage_candidate_request_and_access_substitution_are_rejected() -> None:
    state, pin, entry, grant, current_tip = _grant_context()
    request = entry["request"]
    common = {
        "grant": grant,
        "authenticated_store_snapshot": state,
        "external_store_state_pin": pin,
        "independent_current_tip_anchor": current_tip,
        "expected_consumption_entry_sha256": entry["entry_sha256"],
        "expected_request_sha256": request["request_sha256"],
        "expected_candidate_sha256": request["candidate_sha256"],
        "expected_stage": request["stage"],
        "expected_prerequisite_stage_evidence_sha256": request[
            "prerequisite_stage_evidence_sha256"
        ],
        "expected_stage_access_manifest_sha256": request[
            "stage_access_manifest_sha256"
        ],
        "expected_output_namespace": entry["stage_access_manifest"]["output"][
            "namespace"
        ],
    }
    for key, value in (
        ("expected_candidate_sha256", _h("other candidate")),
        ("expected_request_sha256", _h("other request")),
        ("expected_stage", "final"),
        ("expected_stage_access_manifest_sha256", _h("other access")),
        (
            "expected_prerequisite_stage_evidence_sha256",
            _h("other stage evidence"),
        ),
        ("expected_output_namespace", "aapl-sec-gemma-other-final"),
    ):
        changed = dict(common)
        changed[key] = value
        with pytest.raises(SecFilingGemmaStageAuthorizationError, match="expected"):
            validate_consumed_stage_authorization_grant(**changed)


def test_trusted_stage_content_pin_derivation_is_exact_and_deterministic() -> None:
    state, request, access, pin, _prior_tip, _pinned_tip, _receipt = (
        _trusted_pin_context()
    )
    repeated = derive_reveal_store_trusted_stage_content_pin(
        copy.deepcopy(state),
        reveal_request=copy.deepcopy(request),
        stage_access_manifest=copy.deepcopy(access),
    )

    assert repeated == pin
    assert pin["schema_version"] == TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION
    assert pin["request_sha256"] == request["request_sha256"]
    assert pin["prerequisite_stage_evidence_sha256"] == request[
        "prerequisite_stage_evidence_sha256"
    ]
    assert pin["stage_access_manifest_sha256"] == access[
        "stage_access_manifest_sha256"
    ]
    assert pin["prerequisite_stage"] == "development"
    assert pin["requested_stage"] == "intermediate"
    assert pin["trusted_store_state_sha256"] == state["state_sha256"]
    assert pin["content_manifest_sha256"] == access[
        "prerequisite_evidence_pin"
    ]["content_manifest_sha256"]
    assert pin["stage_artifact_sha256"] == access["prerequisite_evidence_pin"][
        "stage_artifact_sha256"
    ]
    assert pin["external_seal_receipt_sha256"] == access[
        "prerequisite_evidence_pin"
    ]["external_seal_receipt_sha256"]
    assert pin["pin_sha256"] == canonical_sha256(
        {key: value for key, value in pin.items() if key != "pin_sha256"}
    )
    assert "trusted_current_tip_anchor_sha256" not in pin

    substituted_request = copy.deepcopy(request)
    substituted_request["candidate_sha256"] = _h("substituted candidate")
    _rehash(substituted_request, "request_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="crossed its request or stage boundary",
    ):
        derive_reveal_store_trusted_stage_content_pin(
            state,
            reveal_request=substituted_request,
            stage_access_manifest=access,
        )

    substituted_access = copy.deepcopy(access)
    substituted_access["prerequisite_evidence_pin"][
        "stage_artifact_sha256"
    ] = _h("substituted stage artifact")
    _rehash(substituted_access, "stage_access_manifest_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="crossed its request or stage boundary",
    ):
        derive_reveal_store_trusted_stage_content_pin(
            state,
            reveal_request=request,
            stage_access_manifest=substituted_access,
        )


def test_trusted_stage_content_pin_authenticates_exact_persisted_membership() -> None:
    state, request, access, pin, prior_tip, pinned_tip, receipt = (
        _trusted_pin_context()
    )

    prior, current = validate_reveal_store_current_tip_anchor_transition(
        prior_tip,
        pinned_tip,
    )
    assert prior == prior_tip
    assert current == pinned_tip
    assert current["trusted_stage_content_pins"] == {
        request["request_sha256"]: pin
    }
    assert receipt["schema_version"] == (
        TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION
    )
    assert receipt["request_sha256"] == request["request_sha256"]
    assert receipt["trusted_stage_content_pin_sha256"] == pin["pin_sha256"]
    assert receipt["trusted_store_state_sha256"] == state["state_sha256"]
    assert receipt["trusted_current_tip_anchor_sha256"] == pinned_tip[
        "tip_anchor_sha256"
    ]
    assert receipt["trusted_current_tip_revision"] == 1
    assert validate_trusted_stage_content_authentication_receipt(receipt) == receipt


def test_trusted_stage_content_authentication_rejects_unpersisted_and_substituted_inputs() -> None:
    state, request, access, pin, prior_tip, pinned_tip, _receipt = (
        _trusted_pin_context()
    )

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="not exactly persisted",
    ):
        authenticate_reveal_store_trusted_stage_content_pin(
            state,
            prior_tip,
            reveal_request=request,
            stage_access_manifest=access,
        )

    substituted_pin = copy.deepcopy(pin)
    substituted_pin["content_manifest_sha256"] = _h("substituted content")
    _rehash(substituted_pin, "pin_sha256")
    substituted_tip = build_reveal_store_current_tip_anchor(
        state,
        revision=1,
        previous_tip_anchor_sha256=prior_tip["tip_anchor_sha256"],
        authorization_bundles={},
        trusted_stage_content_pins={request["request_sha256"]: substituted_pin},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="not exactly persisted",
    ):
        authenticate_reveal_store_trusted_stage_content_pin(
            state,
            substituted_tip,
            reveal_request=request,
            stage_access_manifest=access,
        )

    substituted_request = copy.deepcopy(request)
    substituted_request["prerequisite_stage_evidence_sha256"] = _h(
        "substituted prerequisite evidence"
    )
    _rehash(substituted_request, "request_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="not exactly persisted",
    ):
        authenticate_reveal_store_trusted_stage_content_pin(
            state,
            pinned_tip,
            reveal_request=substituted_request,
            stage_access_manifest=access,
        )

    substituted_access = copy.deepcopy(access)
    substituted_access["prerequisite_evidence_pin"][
        "external_seal_receipt_sha256"
    ] = _h("substituted seal")
    _rehash(substituted_access, "stage_access_manifest_sha256")
    request_for_substituted_access = copy.deepcopy(request)
    request_for_substituted_access["stage_access_manifest_sha256"] = (
        substituted_access["stage_access_manifest_sha256"]
    )
    _rehash(request_for_substituted_access, "request_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="not exactly persisted",
    ):
        authenticate_reveal_store_trusted_stage_content_pin(
            state,
            pinned_tip,
            reveal_request=request_for_substituted_access,
            stage_access_manifest=substituted_access,
        )


def test_trusted_stage_content_authentication_rejects_stale_store_state() -> None:
    state, request, access, pin, _prior_tip, pinned_tip, _receipt = (
        _trusted_pin_context()
    )
    newer_state = _snapshot(entry_count=1)
    newer_tip = build_reveal_store_current_tip_anchor(
        newer_state,
        revision=2,
        previous_tip_anchor_sha256=pinned_tip["tip_anchor_sha256"],
        authorization_bundles={},
        trusted_stage_content_pins={request["request_sha256"]: pin},
    )

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="current-tip anchor does not authenticate",
    ):
        authenticate_reveal_store_trusted_stage_content_pin(
            state,
            newer_tip,
            reveal_request=request,
            stage_access_manifest=access,
        )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="not exactly persisted",
    ):
        authenticate_reveal_store_trusted_stage_content_pin(
            newer_state,
            newer_tip,
            reveal_request=request,
            stage_access_manifest=access,
        )


def test_current_tip_transition_rejects_trusted_content_pin_deletion() -> None:
    state, _request, _access, _pin, _prior_tip, pinned_tip, _receipt = (
        _trusted_pin_context()
    )
    deleted = build_reveal_store_current_tip_anchor(
        state,
        revision=2,
        previous_tip_anchor_sha256=pinned_tip["tip_anchor_sha256"],
        authorization_bundles={},
        trusted_stage_content_pins={},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="removed or changed a trusted content pin",
    ):
        validate_reveal_store_current_tip_anchor_transition(pinned_tip, deleted)


def test_current_tip_transition_rejects_trusted_content_pin_replacement() -> None:
    state, request, _access, pin, _prior_tip, pinned_tip, _receipt = (
        _trusted_pin_context()
    )
    replacement = copy.deepcopy(pin)
    replacement["stage_artifact_sha256"] = _h("replacement artifact")
    _rehash(replacement, "pin_sha256")
    replaced = build_reveal_store_current_tip_anchor(
        state,
        revision=2,
        previous_tip_anchor_sha256=pinned_tip["tip_anchor_sha256"],
        authorization_bundles={},
        trusted_stage_content_pins={request["request_sha256"]: replacement},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="removed or changed a trusted content pin",
    ):
        validate_reveal_store_current_tip_anchor_transition(pinned_tip, replaced)


def test_current_tip_transition_rejects_two_trusted_content_pin_appends() -> None:
    state, request, access, pin, prior_tip, _pinned_tip, _receipt = (
        _trusted_pin_context()
    )
    second_request = copy.deepcopy(request)
    second_request["prerequisite_stage_evidence_sha256"] = _h(
        "second prerequisite evidence"
    )
    _rehash(second_request, "request_sha256")
    second_pin = derive_reveal_store_trusted_stage_content_pin(
        state,
        reveal_request=second_request,
        stage_access_manifest=access,
    )
    two_pins = build_reveal_store_current_tip_anchor(
        state,
        revision=1,
        previous_tip_anchor_sha256=prior_tip["tip_anchor_sha256"],
        authorization_bundles={},
        trusted_stage_content_pins={
            request["request_sha256"]: pin,
            second_request["request_sha256"]: second_pin,
        },
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="at most one trusted content pin",
    ):
        validate_reveal_store_current_tip_anchor_transition(prior_tip, two_pins)


def test_current_tip_transition_rejects_state_changing_pin_append() -> None:
    state, request, access, _pin, prior_tip, _pinned_tip, _receipt = (
        _trusted_pin_context()
    )
    changed_state = copy.deepcopy(state)
    changed_registry_hash = _h("changed registry")
    changed_state["latest_registry"]["registry_sha256"] = changed_registry_hash
    changed_state["latest_registry_pin"]["registry_sha256"] = changed_registry_hash
    _rehash(changed_state, "state_sha256")
    changed_state_pin = derive_reveal_store_trusted_stage_content_pin(
        changed_state,
        reveal_request=request,
        stage_access_manifest=access,
    )
    changed_state_tip = build_reveal_store_current_tip_anchor(
        changed_state,
        revision=1,
        previous_tip_anchor_sha256=prior_tip["tip_anchor_sha256"],
        authorization_bundles={},
        trusted_stage_content_pins={
            request["request_sha256"]: changed_state_pin,
        },
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="changed the authenticated store state",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            prior_tip,
            changed_state_tip,
        )


def test_current_tip_transition_rejects_pin_append_with_consumption() -> None:
    _state, request, _access, pin, prior_tip, _pinned_tip, _receipt = (
        _trusted_pin_context()
    )
    consumed_state = _snapshot(entry_count=1)
    combined = build_reveal_store_current_tip_anchor(
        consumed_state,
        revision=1,
        previous_tip_anchor_sha256=prior_tip["tip_anchor_sha256"],
        authorization_bundles={},
        trusted_stage_content_pins={request["request_sha256"]: pin},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="dedicated tip-only transition",
    ):
        validate_reveal_store_current_tip_anchor_transition(prior_tip, combined)
