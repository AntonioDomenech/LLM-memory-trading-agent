from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import agent_benchmark.sec_filing_gemma_reveal_registry as registry_module
from agent_benchmark.sec_filing_gemma_contract import (
    REQUIRED_SOURCE_HASHES,
    build_candidate_manifest,
    canonical_sha256,
    session_calendar_sha256,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND,
    HISTORICAL_REGISTRY_CANONICAL_SHA256,
    SecFilingGemmaRevealRegistryError,
    append_candidate_attempt,
    build_initial_reveal_registry,
    build_registry_pin_transition,
    build_single_candidate_reveal_request,
    candidate_design_sha256,
    derive_registry_pin,
    historical_reveal_declaration,
    validate_registry_pin_transition,
    validate_reveal_registry,
    validate_single_candidate_reveal_request,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _commit(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _candidate(
    predecessor_registry: dict,
    sequence: int,
    *,
    salt: str = "candidate",
    provenance_salt: str | None = None,
    audit_wrapper_salt: str | None = None,
    evidence_salt: str | None = None,
    bound_registry_sha256: str | None = None,
) -> dict:
    provenance = salt if provenance_salt is None else provenance_salt
    audit_wrapper = salt if audit_wrapper_salt is None else audit_wrapper_salt
    evidence = salt if evidence_salt is None else evidence_salt
    return build_candidate_manifest(
        model_digest=_digest(f"{salt}:model"),
        ollama_runtime_fingerprint_sha256=_digest(f"{salt}:runtime"),
        sec_audit_checksums_json_sha256=_digest(f"{audit_wrapper}:audit"),
        sec_catalog_artifact_sha256=_digest(f"{evidence}:catalog"),
        sec_audit_source_commit=_commit(f"{provenance}:audit-commit"),
        calendar_source_evidence_sha256=_digest(f"{evidence}:calendar-evidence"),
        calendar_sessions_sha256=session_calendar_sha256(EXPECTED_SESSIONS),
        corpus_universe_sha256=_digest(f"{evidence}:universe"),
        corpus_universe_semantic_sha256=_digest(f"{salt}:semantic-universe"),
        identity_lexicon_sha256=_digest(f"{salt}:lexicon"),
        predecessor_reveal_registry_sha256=(
            predecessor_registry["registry_sha256"]
            if bound_registry_sha256 is None
            else bound_registry_sha256
        ),
        holdout_attempt_id=(
            f"aapl-sec-filing-gemma-v1-attempt-{sequence:03d}"
        ),
        experiment_source_commit=_commit(f"{provenance}:experiment-commit"),
        source_tree_sha256=_digest(f"{provenance}:tree"),
        source_hashes={
            name: _digest(f"{provenance}:source:{name}")
            for name in REQUIRED_SOURCE_HASHES
        },
    )


def _append(
    registry: dict, sequence: int, *, salt: str
) -> tuple[dict, dict, dict]:
    pin = derive_registry_pin(registry)
    candidate = _candidate(registry, sequence, salt=salt)
    appended = append_candidate_attempt(
        registry,
        external_prior_pin=pin,
        candidate_manifest=candidate,
    )
    return appended, derive_registry_pin(appended), candidate


def _stage_args(stage: str = "intermediate", *, salt: str = "stage") -> dict:
    return {
        "stage": stage,
        "stage_access_manifest_sha256": _digest(f"{salt}:{stage}:access"),
        "prerequisite_stage_evidence_sha256": _digest(
            f"{salt}:{stage}:prerequisite"
        ),
    }


def test_historical_declaration_preserves_the_known_repository_lower_bound() -> None:
    declaration = historical_reveal_declaration()

    assert HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND == 10
    assert declaration["historical_final_reveal_count_lower_bound"] == 10
    assert declaration["unattributed_pre_registry_reveal_count_lower_bound"] == 4
    assert declaration["historical_candidate_identities_complete"] is False
    assert declaration["count_semantics"] == "lower_bound_not_exhaustive_history"
    assert declaration["migration_source_canonical_sha256"] == (
        HISTORICAL_REGISTRY_CANONICAL_SHA256
    )
    source = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / declaration["migration_source"]
        ).read_text(encoding="utf-8")
    )
    assert canonical_sha256(source) == HISTORICAL_REGISTRY_CANONICAL_SHA256
    assert [
        entry["reveal_index"] for entry in declaration["known_registered_reveals"]
    ] == [5, 6, 7, 8, 9, 10]
    assert len(
        {
            entry["candidate_sha256"]
            for entry in declaration["known_registered_reveals"]
        }
    ) == 6


def test_initial_registry_is_canonical_and_requires_exact_external_pin() -> None:
    registry = build_initial_reveal_registry()
    pin = derive_registry_pin(registry)

    assert registry["entries"] == []
    assert registry["chain"]["genesis_tip_sha256"] == (
        "10b0ecce18437a9c0e7f03f27882a3d7ba8836c1bcc48a75c3a64d0623de879a"
    )
    assert registry["registry_sha256"] == (
        "5853fcfc8f9ddb651981cb4b1c9f9426e57b214898ab6f080c0485eb1b40fe61"
    )
    assert registry["chain"]["registered_entry_count"] == 0
    assert registry["chain"]["historical_final_reveal_count_lower_bound"] == 10
    assert pin["registered_entry_count"] == 0
    assert pin["historical_final_reveal_count_lower_bound"] == 10
    external_pin_artifact = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/protocol_evidence/sec_gemma_reveal_registry_initial_pin.json"
        ).read_text(encoding="utf-8")
    )
    assert external_pin_artifact["registry_pin"] == pin
    assert external_pin_artifact["migration_source_canonical_sha256"] == (
        HISTORICAL_REGISTRY_CANONICAL_SHA256
    )
    assert canonical_sha256(external_pin_artifact) == (
        "fd28b077da1ab7acbd1e42f124886d50b525bb8fa7ecb8583e46b4613fd0da46"
    )
    assert validate_reveal_registry(registry, external_pin=pin) == registry[
        "registry_sha256"
    ]


def test_first_candidate_binds_attempt_001_without_claiming_a_new_reveal() -> None:
    initial = build_initial_reveal_registry()
    initial_copy = copy.deepcopy(initial)
    initial_pin = derive_registry_pin(initial)
    candidate = _candidate(initial, 1, salt="first")

    appended = append_candidate_attempt(
        initial,
        external_prior_pin=initial_pin,
        candidate_manifest=candidate,
    )
    appended_pin = derive_registry_pin(appended)

    assert initial == initial_copy
    assert appended["registry_sha256"] != initial["registry_sha256"]
    assert appended["chain"]["registered_entry_count"] == 1
    assert appended["chain"]["historical_final_reveal_count_lower_bound"] == 10
    entry = appended["entries"][0]
    assert entry["attempt_id"] == "aapl-sec-filing-gemma-v1-attempt-001"
    assert entry["candidate_sha256"] == candidate["candidate_sha256"]
    assert entry["candidate_design_sha256"] == candidate_design_sha256(candidate)
    assert entry["candidate_manifest"] == candidate
    assert "repository_reveal_index_lower_bound" not in entry
    assert (
        entry["candidate_bound_predecessor_registry_sha256"]
        == initial["registry_sha256"]
    )
    assert entry["prior_tip_sha256"] == initial["chain"]["tip_sha256"]
    assert entry["outcome_payload_included"] is False
    assert entry["cross_attempt_winner_selection_permitted"] is False
    assert validate_reveal_registry(
        appended, external_pin=appended_pin
    ) == appended["registry_sha256"]


def test_append_produces_an_exact_external_compare_and_swap_transition() -> None:
    initial = build_initial_reveal_registry()
    initial_pin = derive_registry_pin(initial)
    candidate = _candidate(initial, 1, salt="cas")
    appended = append_candidate_attempt(
        initial,
        external_prior_pin=initial_pin,
        candidate_manifest=candidate,
    )

    transition = build_registry_pin_transition(
        initial,
        external_prior_pin=initial_pin,
        appended_registry=appended,
    )

    assert transition["expected_prior_pin"] == initial_pin
    assert transition["next_pin"] == derive_registry_pin(appended)
    assert transition["next_pin"]["registered_entry_count"] == 1
    assert transition["stale_or_forked_prior_permitted"] is False
    assert transition["derive_external_pin_from_presented_registry_permitted"] is False
    assert validate_registry_pin_transition(
        transition,
        prior_registry=initial,
        external_prior_pin=initial_pin,
        appended_registry=appended,
    ) == transition["transition_sha256"]


def test_authoritative_newer_pin_rejects_a_stale_or_forked_transition() -> None:
    initial = build_initial_reveal_registry()
    initial_pin = derive_registry_pin(initial)
    winner = append_candidate_attempt(
        initial,
        external_prior_pin=initial_pin,
        candidate_manifest=_candidate(initial, 1, salt="cas-winner"),
    )
    losing_fork = append_candidate_attempt(
        initial,
        external_prior_pin=initial_pin,
        candidate_manifest=_candidate(initial, 1, salt="cas-loser"),
    )
    authoritative_latest_pin = derive_registry_pin(winner)

    with pytest.raises(
        SecFilingGemmaRevealRegistryError, match="externally pinned"
    ):
        build_registry_pin_transition(
            initial,
            external_prior_pin=authoritative_latest_pin,
            appended_registry=losing_fork,
        )


@pytest.mark.parametrize(
    "field",
    [
        "registry_sha256",
        "tip_sha256",
        "registered_entry_count",
        "historical_final_reveal_count_lower_bound",
    ],
)
def test_registry_validation_fails_if_any_external_pin_component_changes(
    field: str,
) -> None:
    registry, pin, _ = _append(
        build_initial_reveal_registry(), 1, salt="external-pin"
    )
    changed = copy.deepcopy(pin)
    if isinstance(changed[field], int):
        changed[field] += 1
    else:
        changed[field] = _digest(f"changed:{field}")

    with pytest.raises(
        SecFilingGemmaRevealRegistryError, match="externally pinned"
    ):
        validate_reveal_registry(registry, external_pin=changed)


def test_candidate_must_bind_the_exact_pinned_predecessor_registry() -> None:
    initial = build_initial_reveal_registry()
    candidate = _candidate(
        initial,
        1,
        salt="wrong-predecessor",
        bound_registry_sha256=_digest("unrelated-registry"),
    )

    with pytest.raises(
        SecFilingGemmaRevealRegistryError, match="predecessor registry"
    ):
        append_candidate_attempt(
            initial,
            external_prior_pin=derive_registry_pin(initial),
            candidate_manifest=candidate,
        )


def test_attempt_ids_cannot_skip_or_be_selected_arbitrarily() -> None:
    initial = build_initial_reveal_registry()
    skipped = _candidate(initial, 2, salt="skipped")

    with pytest.raises(
        SecFilingGemmaRevealRegistryError, match="next contiguous"
    ):
        append_candidate_attempt(
            initial,
            external_prior_pin=derive_registry_pin(initial),
            candidate_manifest=skipped,
        )


def test_exact_candidate_retry_cannot_create_another_attempt() -> None:
    initial = build_initial_reveal_registry()
    appended, appended_pin, candidate = _append(initial, 1, salt="same-candidate")

    with pytest.raises(
        SecFilingGemmaRevealRegistryError, match="Duplicate holdout attempt"
    ):
        append_candidate_attempt(
            appended,
            external_prior_pin=appended_pin,
            candidate_manifest=candidate,
        )


def test_same_underlying_design_cannot_hide_behind_a_new_attempt_hash() -> None:
    initial = build_initial_reveal_registry()
    appended, appended_pin, first = _append(
        initial, 1, salt="stable-duplicate"
    )
    second = _candidate(
        appended,
        2,
        salt="stable-duplicate",
        provenance_salt="new-commit-and-tree",
    )

    assert first["candidate_sha256"] != second["candidate_sha256"]
    assert candidate_design_sha256(first) == candidate_design_sha256(second)
    with pytest.raises(
        SecFilingGemmaRevealRegistryError, match="identical frozen candidate design"
    ):
        append_candidate_attempt(
            appended,
            external_prior_pin=appended_pin,
            candidate_manifest=second,
        )


def test_fresh_audit_wrapper_cannot_turn_same_design_into_another_attempt() -> None:
    initial = build_initial_reveal_registry()
    appended, appended_pin, first = _append(
        initial, 1, salt="same-audit-design"
    )
    second = _candidate(
        appended,
        2,
        salt="same-audit-design",
        audit_wrapper_salt="fresh-runtime-ledger",
    )

    assert first["candidate_sha256"] != second["candidate_sha256"]
    assert candidate_design_sha256(first) == candidate_design_sha256(second)
    with pytest.raises(
        SecFilingGemmaRevealRegistryError, match="identical frozen candidate design"
    ):
        append_candidate_attempt(
            appended,
            external_prior_pin=appended_pin,
            candidate_manifest=second,
        )


def test_fresh_evidence_wrappers_keep_the_same_semantic_design_identity() -> None:
    initial = build_initial_reveal_registry()
    appended, appended_pin, first = _append(
        initial, 1, salt="same-semantic-data"
    )
    second = _candidate(
        appended,
        2,
        salt="same-semantic-data",
        evidence_salt="fresh-catalog-calendar-universe-wrappers",
    )

    assert first["candidate_sha256"] != second["candidate_sha256"]
    assert candidate_design_sha256(first) == candidate_design_sha256(second)
    with pytest.raises(
        SecFilingGemmaRevealRegistryError, match="identical frozen candidate design"
    ):
        append_candidate_attempt(
            appended,
            external_prior_pin=appended_pin,
            candidate_manifest=second,
        )


def test_single_candidate_request_is_hashed_but_does_not_authorize_access() -> None:
    registry, pin, candidate = _append(
        build_initial_reveal_registry(), 1, salt="authorized"
    )

    request = build_single_candidate_reveal_request(
        registry,
        external_pin=pin,
        candidate_manifest=candidate,
        **_stage_args(),
    )

    assert request["attempt_id"].endswith("001")
    assert request["candidate_sha256"] == candidate["candidate_sha256"]
    assert request["stage"] == "intermediate"
    assert request["prerequisite_stage"] == "development"
    assert request["request_scope"] == (
        "one_current_tip_candidate_and_one_stage_only"
    )
    assert request["authorizes_outcome_access"] is False
    assert request["effectful_atomic_single_use_consumption_required"] is True
    assert request["cross_attempt_comparison_permitted"] is False
    assert request["cross_attempt_winner_selection_permitted"] is False
    assert request["globally_pristine_claim"] is False
    assert validate_single_candidate_reveal_request(
        request,
        registry=registry,
        external_pin=pin,
        candidate_manifest=candidate,
        **_stage_args(),
    ) == request["request_sha256"]


def test_reveal_request_cannot_be_reused_for_another_stage_or_dataset() -> None:
    registry, pin, candidate = _append(
        build_initial_reveal_registry(), 1, salt="stage-bound"
    )
    intermediate_args = _stage_args("intermediate", salt="intermediate")
    request = build_single_candidate_reveal_request(
        registry,
        external_pin=pin,
        candidate_manifest=candidate,
        **intermediate_args,
    )

    with pytest.raises(SecFilingGemmaRevealRegistryError):
        validate_single_candidate_reveal_request(
            request,
            registry=registry,
            external_pin=pin,
            candidate_manifest=candidate,
            **_stage_args("final", salt="final"),
        )
    changed_scope = dict(intermediate_args)
    changed_scope["stage_access_manifest_sha256"] = _digest("other-stage-data")
    with pytest.raises(SecFilingGemmaRevealRegistryError):
        validate_single_candidate_reveal_request(
            request,
            registry=registry,
            external_pin=pin,
            candidate_manifest=candidate,
            **changed_scope,
        )
    with pytest.raises(SecFilingGemmaRevealRegistryError, match="intermediate or final"):
        build_single_candidate_reveal_request(
            registry,
            external_pin=pin,
            candidate_manifest=candidate,
            **_stage_args("development"),
        )


def test_older_candidate_cannot_be_requested_after_a_new_tip_is_pinned() -> None:
    first_registry, first_pin, first_candidate = _append(
        build_initial_reveal_registry(), 1, salt="first-tip"
    )
    second_candidate = _candidate(first_registry, 2, salt="second-tip")
    second_registry = append_candidate_attempt(
        first_registry,
        external_prior_pin=first_pin,
        candidate_manifest=second_candidate,
    )
    second_pin = derive_registry_pin(second_registry)

    with pytest.raises(
        SecFilingGemmaRevealRegistryError, match="current tip"
    ):
        build_single_candidate_reveal_request(
            second_registry,
            external_pin=second_pin,
            candidate_manifest=first_candidate,
            **_stage_args(),
        )

    current = build_single_candidate_reveal_request(
        second_registry,
        external_pin=second_pin,
        candidate_manifest=second_candidate,
        **_stage_args(),
    )
    assert current["attempt_id"].endswith("002")
    assert current["registered_entry_count"] == 2
    assert current["historical_final_reveal_count_lower_bound"] == 10


@pytest.mark.parametrize("mutation", ["rank_field", "boolean_as_integer", "hash"])
def test_request_rejects_ranking_and_noncanonical_mutations(
    mutation: str,
) -> None:
    registry, pin, candidate = _append(
        build_initial_reveal_registry(), 1, salt="no-ranking"
    )
    request = build_single_candidate_reveal_request(
        registry,
        external_pin=pin,
        candidate_manifest=candidate,
        **_stage_args(),
    )
    if mutation == "rank_field":
        request["winner_rank"] = 1
    elif mutation == "boolean_as_integer":
        request["cross_attempt_winner_selection_permitted"] = 0
    else:
        request["request_sha256"] = _digest("forged-request")

    with pytest.raises(SecFilingGemmaRevealRegistryError):
        validate_single_candidate_reveal_request(
            request,
            registry=registry,
            external_pin=pin,
            candidate_manifest=candidate,
            **_stage_args(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "historical_count",
        "winner_semantics",
        "winner_semantics_integer",
        "entry_candidate",
        "entry_boolean_integer",
        "chain_count",
        "registry_hash",
    ],
)
def test_registry_tampering_is_fail_closed_even_before_external_pin_comparison(
    mutation: str,
) -> None:
    registry, pin, _ = _append(
        build_initial_reveal_registry(), 1, salt=f"tamper:{mutation}"
    )
    changed = copy.deepcopy(registry)
    if mutation == "historical_count":
        changed["historical_reveal_declaration"][
            "historical_final_reveal_count_lower_bound"
        ] = 9
    elif mutation == "winner_semantics":
        changed["governance"]["cross_attempt_winner_selection_permitted"] = True
    elif mutation == "winner_semantics_integer":
        # Python considers False == 0, but canonical JSON does not.  The
        # registry must therefore validate bytes/types as well as equality.
        changed["governance"]["cross_attempt_winner_selection_permitted"] = 0
    elif mutation == "entry_candidate":
        changed["entries"][0]["candidate_sha256"] = _digest("replacement")
    elif mutation == "entry_boolean_integer":
        changed["entries"][0]["outcome_payload_included"] = 0
    elif mutation == "chain_count":
        changed["chain"]["registered_entry_count"] = 2
    else:
        changed["registry_sha256"] = _digest("changed-registry")

    with pytest.raises(SecFilingGemmaRevealRegistryError):
        validate_reveal_registry(changed, external_pin=pin)


def test_structurally_rehashed_duplicate_candidate_is_still_rejected() -> None:
    first_registry, first_pin, _ = _append(
        build_initial_reveal_registry(), 1, salt="duplicate-one"
    )
    second_candidate = _candidate(first_registry, 2, salt="duplicate-two")
    second_registry = append_candidate_attempt(
        first_registry,
        external_prior_pin=first_pin,
        candidate_manifest=second_candidate,
    )
    forged = copy.deepcopy(second_registry)
    forged["entries"][1]["candidate_sha256"] = forged["entries"][0][
        "candidate_sha256"
    ]
    second_body = {
        key: value
        for key, value in forged["entries"][1].items()
        if key != "entry_sha256"
    }
    forged["entries"][1]["entry_sha256"] = canonical_sha256(second_body)
    forged["chain"]["tip_sha256"] = forged["entries"][1]["entry_sha256"]
    registry_body = {
        key: value for key, value in forged.items() if key != "registry_sha256"
    }
    forged["registry_sha256"] = canonical_sha256(registry_body)

    with pytest.raises(SecFilingGemmaRevealRegistryError):
        derive_registry_pin(forged)


def test_chain_removal_and_reordering_cannot_be_rehashed_into_valid_history() -> None:
    first_registry, first_pin, _ = _append(
        build_initial_reveal_registry(), 1, salt="ordered-one"
    )
    second_candidate = _candidate(first_registry, 2, salt="ordered-two")
    second_registry = append_candidate_attempt(
        first_registry,
        external_prior_pin=first_pin,
        candidate_manifest=second_candidate,
    )

    for changed_entries in (
        list(reversed(copy.deepcopy(second_registry["entries"]))),
        copy.deepcopy(second_registry["entries"][1:]),
    ):
        changed = copy.deepcopy(second_registry)
        changed["entries"] = changed_entries
        with pytest.raises(SecFilingGemmaRevealRegistryError):
            derive_registry_pin(changed)


def test_invalid_candidate_manifest_is_rejected_at_the_registry_boundary() -> None:
    initial = build_initial_reveal_registry()
    candidate = _candidate(initial, 1, salt="invalid-candidate")
    candidate["candidate_sha256"] = _digest("tampered-candidate")

    with pytest.raises(
        SecFilingGemmaRevealRegistryError, match="not a valid frozen"
    ):
        append_candidate_attempt(
            initial,
            external_prior_pin=derive_registry_pin(initial),
            candidate_manifest=candidate,
        )


def test_registry_module_has_no_effectful_imports_or_io_calls() -> None:
    tree = ast.parse(inspect.getsource(registry_module))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert imported_modules <= {
        "__future__",
        "collections.abc",
        "copy",
        "hmac",
        "re",
        "typing",
        "agent_benchmark.sec_filing_gemma_contract",
    }
    forbidden_calls = {
        "open",
        "exec",
        "eval",
        "compile",
        "input",
        "getattr",
        "setattr",
        "delattr",
        "__import__",
    }
    observed_direct_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert observed_direct_calls.isdisjoint(forbidden_calls)
