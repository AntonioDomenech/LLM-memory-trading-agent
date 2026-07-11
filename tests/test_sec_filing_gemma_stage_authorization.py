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
    CONSUMED_STAGE_STORE_PIN_SCHEMA_VERSION,
    CONSUMPTION_ENTRY_SCHEMA_VERSION,
    CONSUMPTION_LEDGER_SCHEMA_VERSION,
    SEMANTIC_PREREQUISITE_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
    SecFilingGemmaStageAuthorizationError,
    build_reveal_store_current_tip_anchor,
    build_consumed_stage_authorization_grant,
    derive_consumed_stage_store_state_pin,
    validate_consumed_stage_authorization_grant,
    validate_consumed_stage_store_state_pin,
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
