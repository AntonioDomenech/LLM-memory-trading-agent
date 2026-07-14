from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import date
from functools import lru_cache
import hashlib
import json
from unittest.mock import patch

import pytest

import agent_benchmark.sec_filing_gemma_stage_verifier as verifier_module

from agent_benchmark.sec_filing_gemma_contract import (
    CANONICAL_IDENTITY_LEXICON_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_FOLD_SPECS,
    REQUIRED_SOURCE_HASHES,
    REQUIRED_STAGE_VERIFIER_CHECKS,
    build_candidate_manifest,
    build_contract_manifest,
    build_corpus_universe_manifest,
    build_stage_content_manifest,
    canonical_sha256,
    market_session_calendar_sha256,
    session_calendar_sha256,
)
from agent_benchmark.sec_session_calendar import (
    EXPECTED_MARKET_HISTORY_SESSIONS,
    EXPECTED_SESSIONS,
)
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    REVEAL_REQUEST_SCHEMA_VERSION,
    candidate_design_sha256,
)
from agent_benchmark.sec_filing_gemma_market_acquirer import (
    build_development_market_acquisition_plan,
)
from agent_benchmark.sec_filing_gemma_market_evidence import MARKET_SYMBOLS
from agent_benchmark.sec_filing_gemma_learner import (
    MODEL_TYPE as LEARNER_MODEL_TYPE,
    STATE_SCHEMA_VERSION as LEARNER_STATE_SCHEMA_VERSION,
    SecFilingGemmaLearnerConfig,
)
from agent_benchmark.sec_filing_gemma_stage_access import (
    DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID,
    STAGE_ACCESS_MANIFEST_SCHEMA_VERSION,
    build_development_content_root_plan,
)
from agent_benchmark.sec_filing_gemma_stage_verifier import (
    STAGE_EVIDENCE_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_stage_authorization import (
    CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION,
    CONSUMED_STAGE_AUTHORIZATION_GRANT_SCHEMA_VERSION,
    CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION,
    CONSUMED_STAGE_STORE_PIN_SCHEMA_VERSION,
    CONSUMPTION_ENTRY_SCHEMA_VERSION,
    CONSUMPTION_LEDGER_SCHEMA_VERSION,
    DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_SCHEMA_VERSION,
    DEVELOPMENT_LABEL_ASSEMBLY_PLAN_SCHEMA_VERSION,
    DEVELOPMENT_OOF_LEARNER_FIT_PLAN_SCHEMA_VERSION,
    DEVELOPMENT_OOF_PREDICTION_PLAN_SCHEMA_VERSION,
    DEVELOPMENT_POLICY_REPLAY_PLAN_SCHEMA_VERSION,
    DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_SCHEMA_VERSION,
    DEVELOPMENT_MARKET_EXECUTION_ABORT_SCHEMA_VERSION,
    DEVELOPMENT_MARKET_EXECUTION_CLAIM_SCHEMA_VERSION,
    DEVELOPMENT_MARKET_READER_RECEIPT_SCHEMA_VERSION,
    DEVELOPMENT_SEC_EXECUTION_ABORT_SCHEMA_VERSION,
    DEVELOPMENT_SEC_EXECUTION_CLAIM_SCHEMA_VERSION,
    DEVELOPMENT_SEC_READER_RECEIPT_SCHEMA_VERSION,
    DEVELOPMENT_MODEL_EXECUTION_ABORT_SCHEMA_VERSION,
    DEVELOPMENT_MODEL_EXECUTION_CLAIM_SCHEMA_VERSION,
    DEVELOPMENT_MODEL_READER_RECEIPT_SCHEMA_VERSION,
    DEVELOPMENT_ROOT_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION,
    MARKET_EXECUTION_SOURCE_ROLES,
    MODEL_EXECUTION_SOURCE_ROLES,
    OWNED_SEC_RAW_BATCH_MAX_BYTES,
    SEC_EXECUTION_RESOLVED_SOURCE_PATHS,
    SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID,
    SEMANTIC_PREREQUISITE_SCHEMA_VERSION,
    STAGE_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION,
    STAGE_MODEL_BATCH_COMPONENT_ID,
    STAGE_MODEL_EXECUTION_ABORT_SCHEMA_VERSION,
    STAGE_MODEL_EXECUTION_CLAIM_SCHEMA_VERSION,
    STAGE_MODEL_READER_RECEIPT_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
    TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION,
    TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION,
    SecFilingGemmaStageAuthorizationError,
    authenticate_reveal_store_trusted_stage_content_pin,
    build_reveal_store_current_tip_anchor,
    build_consumed_stage_authorization_grant,
    build_consumed_stage_output_receipt,
    build_development_market_execution_abort,
    build_development_market_execution_claim,
    build_development_market_reader_receipt,
    build_development_feature_assembly_plan,
    build_development_label_assembly_plan,
    build_development_oof_learner_fit_plan,
    build_development_oof_prediction_plan,
    build_development_policy_replay_plan,
    build_development_training_membership_assembly_plan,
    build_development_sec_execution_abort,
    build_development_sec_execution_claim,
    build_development_sec_reader_receipt,
    build_development_model_execution_abort,
    build_development_model_execution_claim,
    build_development_model_reader_receipt,
    build_development_root_carry_in_reader_receipt,
    build_stage_carry_in_reader_receipt,
    build_stage_model_execution_abort,
    build_stage_model_execution_claim,
    build_stage_model_reader_receipt,
    build_stage_sec_execution_abort,
    build_stage_sec_execution_claim,
    build_stage_sec_reader_receipt,
    derive_consumed_stage_store_state_pin,
    derive_reveal_store_trusted_stage_content_pin,
    validate_consumed_stage_authorization_grant,
    validate_consumed_stage_output_receipt,
    validate_consumed_stage_store_state_pin,
    validate_development_market_execution_abort,
    validate_development_market_execution_claim,
    validate_development_market_reader_receipt,
    validate_development_feature_assembly_plan,
    validate_development_label_assembly_plan,
    validate_development_oof_learner_fit_plan,
    validate_development_oof_prediction_plan,
    validate_development_policy_replay_plan,
    validate_development_training_membership_assembly_plan,
    validate_development_root_carry_in_reader_receipt,
    validate_development_model_execution_abort,
    validate_development_model_execution_claim,
    validate_development_model_reader_receipt,
    validate_reveal_store_current_tip_anchor_transition,
    validate_stage_carry_in_reader_receipt,
    validate_stage_model_execution_abort,
    validate_stage_model_execution_claim,
    validate_stage_model_reader_receipt,
    validate_trusted_stage_content_authentication_receipt,
    _sec_component_plan_from_bundle,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


SEC_USER_AGENT_SHA256 = f"sha256:{_h('private SEC contact')}"


def _commit(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def _sec_source_hashes(sequence: int) -> dict[str, str]:
    source_hashes = {name: _h(f"source:{sequence}:{name}") for name in REQUIRED_SOURCE_HASHES}
    source_hashes["runner"] = _h("runner source")
    source_hashes["sec_corpus_selector"] = _h("corpus source")
    return source_hashes


def _execution_source_hashes(sequence: int = 1) -> dict[str, str]:
    source_hashes = _sec_source_hashes(sequence)
    return {
        role: source_hashes[role]
        for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
    }


def _model_execution_source_hashes(sequence: int = 1) -> dict[str, str]:
    source_hashes = _sec_source_hashes(sequence)
    return {role: source_hashes[role] for role in MODEL_EXECUTION_SOURCE_ROLES}


def _development_model_source_hashes(state: dict) -> dict[str, str]:
    source_hashes = state["latest_registry"]["entries"][-1][
        "candidate_manifest"
    ]["bindings"]["source_hashes"]
    return {role: source_hashes[role] for role in MODEL_EXECUTION_SOURCE_ROLES}


def _development_market_source_hashes(state: dict) -> dict[str, str]:
    source_hashes = state["latest_registry"]["entries"][-1][
        "candidate_manifest"
    ]["bindings"]["source_hashes"]
    return {role: source_hashes[role] for role in MARKET_EXECUTION_SOURCE_ROLES}


_MARKET_PLAN_SOURCE_ROLES = {
    "agent_benchmark/sec_filing_gemma_market_acquirer.py": "market_acquirer",
    "agent_benchmark/sec_filing_gemma_market_evidence.py": "market_evidence",
    "agent_benchmark/sec_filing_gemma_market_source_bytes.py": (
        "market_source_bytes"
    ),
    "agent_benchmark/sec_filing_gemma_contract.py": "contract",
    "agent_benchmark/sec_session_calendar.py": "calendar",
}


def _bundle_model_source_hashes(bundle: dict) -> dict[str, str]:
    source_hashes = bundle["authenticated_store_snapshot"]["latest_registry"][
        "entries"
    ][-1]["candidate_manifest"]["bindings"]["source_hashes"]
    return {role: source_hashes[role] for role in MODEL_EXECUTION_SOURCE_ROLES}


def test_model_execution_source_roles_cover_every_resolved_candidate_source() -> None:
    assert MODEL_EXECUTION_SOURCE_ROLES == tuple(
        role for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
    )
    assert set(MODEL_EXECUTION_SOURCE_ROLES) == {
        role for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
    }


def _development_source_record(
    year: int,
    serial: int,
    form: str,
    month: int,
) -> dict:
    evidence_date = date(year, month, 15)
    return {
        "accession_number": f"0000320193-{year % 100:02d}-{serial:06d}",
        "subject_cik": "0000320193",
        "form": form,
        "acceptance_datetime": evidence_date.strftime("%Y%m%d") + "160000",
        "filing_date": evidence_date.isoformat(),
        "filing_date_change": None,
        "primary_document": f"filing-{serial}.htm",
        "source_record_sha256": f"{serial + 70_000:064x}",
    }


@lru_cache(maxsize=1)
def _development_universe() -> dict:
    records: list[dict] = []
    serial = 1
    for year in range(2000, 2026):
        # Deliberately assign accession serials out of event-time order.  SEC
        # acquisition is accession-sorted, while model events must be ordered
        # by conservative availability session.
        for form, month in (
            ("10-Q", 5),
            ("10-K", 2),
            ("10-Q", 8),
            ("10-Q", 11),
        ):
            records.append(
                _development_source_record(year, serial, form, month)
            )
            serial += 1
    for form, month in (("10-Q", 5), ("10-Q", 2)):
        records.append(_development_source_record(2026, serial, form, month))
        serial += 1
    return build_corpus_universe_manifest(
        catalog_artifact_sha256=_h("development catalog"),
        calendar_artifact_sha256=_h("development calendar evidence"),
        catalog_total_record_count=1_000,
        catalog_eligible_record_count=len(records),
        session_dates=EXPECTED_SESSIONS,
        records=records,
    )


def _development_root_context() -> tuple[dict, dict, dict, dict]:
    universe = copy.deepcopy(_development_universe())
    source_hashes = {
        role: _h(f"development source:{role}")
        for role in REQUIRED_SOURCE_HASHES
    }
    market_plan = build_development_market_acquisition_plan()
    for path, digest in market_plan["source_code_sha256s"].items():
        source_hashes[_MARKET_PLAN_SOURCE_ROLES[path]] = digest
    attempt_id = f"{CONTRACT_VERSION}-attempt-001"
    candidate = build_candidate_manifest(
        model_digest=_h("development model"),
        ollama_runtime_fingerprint_sha256=_h("development runtime"),
        sec_audit_checksums_json_sha256=_h("development audit"),
        sec_catalog_artifact_sha256=universe["catalog_artifact_sha256"],
        sec_audit_source_commit=_commit("development audit source"),
        calendar_source_evidence_sha256=universe[
            "calendar_artifact_sha256"
        ],
        calendar_sessions_sha256=universe["calendar_sessions_sha256"],
        corpus_universe_sha256=universe["universe_sha256"],
        corpus_universe_semantic_sha256=universe[
            "universe_semantic_sha256"
        ],
        identity_lexicon_sha256=CANONICAL_IDENTITY_LEXICON_SHA256,
        predecessor_reveal_registry_sha256=_h(
            "development predecessor registry"
        ),
        holdout_attempt_id=attempt_id,
        experiment_source_commit=_commit("development experiment source"),
        source_tree_sha256=_h("development source tree"),
        source_hashes=source_hashes,
    )
    design_hash = candidate_design_sha256(candidate)
    plan = build_development_content_root_plan(
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate["candidate_sha256"],
        expected_candidate_design_sha256=design_hash,
        expected_attempt_id=attempt_id,
        base_corpus_universe_sha256=universe["universe_sha256"],
        corpus_universe_manifest=universe,
        session_calendar_sha256=universe["calendar_sessions_sha256"],
    )
    state = _snapshot(entry_count=0)
    registry_entry = {
        "entry_sha256": _h("development current registry entry"),
        "sequence": 1,
        "attempt_id": attempt_id,
        "candidate_sha256": candidate["candidate_sha256"],
        "candidate_design_sha256": design_hash,
        "candidate_manifest": candidate,
    }
    registry_hash = _h("development current registry")
    registry_tip = _h("development current registry tip")
    state["latest_registry"] = {
        "schema_version": "synthetic-development-registry-v1",
        "entries": [registry_entry],
        "chain": {
            "tip_sha256": registry_tip,
            "registered_entry_count": 1,
        },
        "registry_sha256": registry_hash,
    }
    state["latest_registry_pin"] = {
        "schema_version": "synthetic-development-registry-pin-v1",
        "registry_sha256": registry_hash,
        "tip_sha256": registry_tip,
        "registered_entry_count": 1,
    }
    _rehash(state, "state_sha256")
    current_tip = build_reveal_store_current_tip_anchor(
        state,
        revision=0,
        previous_tip_anchor_sha256=None,
        authorization_bundles={},
    )
    execution_sources = {
        role: source_hashes[role]
        for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
    }
    return state, plan, current_tip, execution_sources


def _sec_candidate(sequence: int) -> dict:
    source_hashes = _sec_source_hashes(sequence)
    return build_candidate_manifest(
        model_digest=_h(f"model:{sequence}"),
        ollama_runtime_fingerprint_sha256=_h(f"runtime:{sequence}"),
        sec_audit_checksums_json_sha256=_h(f"audit:{sequence}"),
        sec_catalog_artifact_sha256=_h(f"catalog:{sequence}"),
        sec_audit_source_commit=_commit(f"audit-source:{sequence}"),
        calendar_source_evidence_sha256=_h(f"calendar:{sequence}"),
        calendar_sessions_sha256=session_calendar_sha256(EXPECTED_SESSIONS),
        corpus_universe_sha256=_h(f"universe:{sequence}"),
        corpus_universe_semantic_sha256=_h(f"universe-semantic:{sequence}"),
        identity_lexicon_sha256=CANONICAL_IDENTITY_LEXICON_SHA256,
        predecessor_reveal_registry_sha256=_h("registry"),
        holdout_attempt_id=f"{CONTRACT_VERSION}-attempt-{sequence:03d}",
        experiment_source_commit=_commit(f"experiment:{sequence}"),
        source_tree_sha256=_h(f"source-tree:{sequence}"),
        source_hashes=source_hashes,
    )


def _genesis(anchor: dict) -> str:
    return canonical_sha256(
        {
            "schema_version": "aapl-sec-gemma-consumed-request-genesis-v1",
            "contract_version": CONTRACT_VERSION,
            "store_anchor_sha256": canonical_sha256(anchor),
        }
    )


def _access(
    *,
    attempt: str,
    candidate: str,
    stage: str,
    include_sec_plan: bool = False,
) -> dict:
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
    if include_sec_plan:
        accession = "0000320193-24-000123"
        official_url = (
            "https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019324000123/aapl-20240928.htm"
        )
        documents = [
            {"accession_number": accession, "official_url": official_url}
        ]
        body["sec_access_plan"] = {
            "selection_policy": (
                "all_and_only_requested_stage_universe_primary_documents"
            ),
            "method": "GET",
            "network_scope": "official_sec_https_only",
            "redirects_permitted": False,
            "retries_permitted": False,
            "cache_substitution_permitted": False,
            "document_count": 1,
            "accessions_sha256": canonical_sha256([accession]),
            "official_urls_sha256": canonical_sha256([official_url]),
            "documents": documents,
        }
        body["budgets"] = {
            "max_sec_requests": 1,
            "max_sec_response_bytes": 1_000_000,
            "max_sec_acquisition_seconds": 30.0,
        }
    return {**body, "stage_access_manifest_sha256": canonical_sha256(body)}


def _entry(
    *,
    sequence: int,
    prior_tip: str,
    stage: str = "intermediate",
    include_sec_plan: bool = False,
    candidate_sha256_override: str | None = None,
) -> dict:
    attempt = f"{CONTRACT_VERSION}-attempt-{sequence:03d}"
    candidate = (
        _h(f"candidate:{sequence}")
        if candidate_sha256_override is None
        else candidate_sha256_override
    )
    access = _access(
        attempt=attempt,
        candidate=candidate,
        stage=stage,
        include_sec_plan=include_sec_plan,
    )
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


def _snapshot(*, entry_count: int = 1, include_sec_plan: bool = False) -> dict:
    anchor = {"schema_version": "synthetic-store-anchor-v1", "root": _h("anchor")}
    current_tip = _genesis(anchor)
    entries: list[dict] = []
    registry_entries: list[dict] = []
    for sequence in range(1, entry_count + 1):
        candidate_manifest = _sec_candidate(sequence) if include_sec_plan else None
        entry = _entry(
            sequence=sequence,
            prior_tip=current_tip,
            include_sec_plan=include_sec_plan,
            candidate_sha256_override=(
                None
                if candidate_manifest is None
                else candidate_manifest["candidate_sha256"]
            ),
        )
        entries.append(entry)
        if candidate_manifest is not None:
            registry_entries.append(
                {
                    "entry_sha256": entry["registry_entry_sha256"],
                    "candidate_manifest": candidate_manifest,
                }
            )
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
        "entries": registry_entries,
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


def _grant_context(
    entry_count: int = 1,
    *,
    include_sec_plan: bool = False,
) -> tuple[dict, dict, dict, dict, dict]:
    state = _snapshot(
        entry_count=entry_count,
        include_sec_plan=include_sec_plan,
    )
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
        "output_stage_evidence_schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "output_stage_evidence_sha256": _h(f"{salt}:evidence"),
        "output_stage_evidence_document_sha256": _h(f"{salt}:document"),
        "output_stage_evidence_complete_marker_sha256": _h(
            f"{salt}:complete-marker"
        ),
        "output_stage_evidence_canonical_byte_count": 4096,
        "output_stage_evidence_prerequisite_stage": grant["stage"],
        "output_parent_stage_evidence_sha256": grant[
            "prerequisite_stage_evidence_sha256"
        ],
        "output_candidate_sha256": grant["candidate_sha256"],
    }


def _next_tip(
    state: dict,
    prior_tip: dict,
    **map_overrides: dict,
) -> dict:
    maps = {
        name: copy.deepcopy(prior_tip[name])
        for name in (
            "authorization_bundles",
            "trusted_stage_content_pins",
            "consumed_stage_output_receipts",
            "stage_carry_in_reader_receipts",
            "stage_sec_execution_claims",
            "stage_sec_reader_receipts",
            "stage_sec_execution_aborts",
            "development_sec_execution_claims",
            "development_sec_reader_receipts",
            "development_sec_execution_aborts",
            "development_market_execution_claims",
            "development_market_reader_receipts",
            "development_market_execution_aborts",
            "development_root_carry_in_reader_receipts",
            "stage_model_execution_claims",
            "stage_model_reader_receipts",
            "stage_model_execution_aborts",
            "development_model_execution_claims",
            "development_model_reader_receipts",
            "development_model_execution_aborts",
        )
    }
    maps.update(map_overrides)
    return build_reveal_store_current_tip_anchor(
        state,
        revision=prior_tip["revision"] + 1,
        previous_tip_anchor_sha256=prior_tip["tip_anchor_sha256"],
        **maps,
    )


def _market_byte_index(
    *,
    raw_response_sha256s: dict[str, str],
    artifact_sha256s: dict[str, str],
    window_sha256s: dict[str, str],
) -> list[dict]:
    layout: list[tuple[str, str, str]] = []
    for symbol in MARKET_SYMBOLS:
        layout.append(
            (
                f"raw-response-{symbol}",
                f"raw-response-{symbol}.json",
                raw_response_sha256s[symbol],
            )
        )
    for symbol in MARKET_SYMBOLS:
        layout.append(
            (
                f"artifact-{symbol}",
                f"artifact-{symbol}.json",
                artifact_sha256s[symbol],
            )
        )
    for symbol in MARKET_SYMBOLS:
        layout.append(
            (
                f"window-{symbol}",
                f"window-{symbol}.json",
                window_sha256s[symbol],
            )
        )
    layout.extend(
        (
            (
                "source-manifest",
                "source-manifest.json",
                _h("development market source manifest file bytes"),
            ),
            (
                "stage-manifest",
                "stage-manifest.json",
                _h("development market stage manifest file bytes"),
            ),
            (
                "reconciliation-receipt",
                "reconciliation-receipt.json",
                _h("development market reconciliation receipt file bytes"),
            ),
            (
                "acquisition-receipt",
                "acquisition-receipt.json",
                _h("development market acquisition receipt file bytes"),
            ),
        )
    )
    return [
        {
            "ordinal": ordinal,
            "logical_id": logical_id,
            "relative_path": relative_path,
            "byte_count": 100 + ordinal,
            "sha256": digest,
        }
        for ordinal, (logical_id, relative_path, digest) in enumerate(
            layout,
            start=1,
        )
    ]


def _development_market_receipt_inputs() -> dict:
    raw_response_sha256s = {
        symbol: _h(f"development market raw response:{symbol}")
        for symbol in MARKET_SYMBOLS
    }
    artifact_sha256s = {
        symbol: _h(f"development market artifact:{symbol}")
        for symbol in MARKET_SYMBOLS
    }
    window_sha256s = {
        symbol: _h(f"development market window:{symbol}")
        for symbol in MARKET_SYMBOLS
    }
    values = {
        "acquisition_receipt_sha256": _h(
            "development market acquisition receipt"
        ),
        "acquisition_bundle_sha256": _h("development market acquisition bundle"),
        "acquisition_validation_sha256": _h(
            "development market acquisition validation"
        ),
        "source_manifest_sha256": _h("development market source manifest"),
        "market_stage_manifest_sha256": _h(
            "development market stage manifest"
        ),
        "source_reconciliation_sha256": _h(
            "development market source reconciliation"
        ),
        "raw_response_sha256s": raw_response_sha256s,
        "artifact_sha256s": artifact_sha256s,
        "window_sha256s": window_sha256s,
        "complete_marker_sha256": _h("development market complete marker"),
        "owned_transport_attested_by_store": True,
    }
    values["byte_index"] = _market_byte_index(
        raw_response_sha256s=raw_response_sha256s,
        artifact_sha256s=artifact_sha256s,
        window_sha256s=window_sha256s,
    )
    return values


def _completed_development_market_lifecycle(
    state: dict,
    root_reader_tip: dict,
    *,
    scope_hash: str,
) -> tuple[dict, dict, dict, dict, dict]:
    acquisition_plan = build_development_market_acquisition_plan()
    claim = build_development_market_execution_claim(
        state,
        development_root_scope_sha256=scope_hash,
        independent_current_tip_anchor=root_reader_tip,
        market_acquisition_plan=acquisition_plan,
        execution_source_hashes=_development_market_source_hashes(state),
    )
    claim_tip = _next_tip(
        state,
        root_reader_tip,
        development_market_execution_claims={scope_hash: claim},
    )
    validate_reveal_store_current_tip_anchor_transition(
        root_reader_tip,
        claim_tip,
        authenticated_store_snapshot=state,
    )
    receipt_inputs = _development_market_receipt_inputs()
    reader = build_development_market_reader_receipt(claim, **receipt_inputs)
    reader_tip = _next_tip(
        state,
        claim_tip,
        development_market_reader_receipts={scope_hash: reader},
    )
    validate_reveal_store_current_tip_anchor_transition(claim_tip, reader_tip)
    return claim_tip, reader_tip, claim, reader, {
        "scope_sha256": scope_hash,
        "acquisition_plan": acquisition_plan,
        "reader": reader,
        "receipt_inputs": receipt_inputs,
    }


def _completed_sec_output_ancestry(
    state: dict,
    entry: dict,
    prior_tip: dict,
) -> tuple[dict, dict, dict]:
    request_hash = entry["request_sha256"]
    bundle = prior_tip["authorization_bundles"][request_hash]
    claim = build_stage_sec_execution_claim(
        bundle,
        independent_current_tip_anchor=prior_tip,
        execution_source_hashes=_execution_source_hashes(entry["sequence"]),
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    claimed_tip = _next_tip(
        state,
        prior_tip,
        stage_sec_execution_claims={request_hash: claim},
    )
    validate_reveal_store_current_tip_anchor_transition(prior_tip, claimed_tip)
    reader = build_stage_sec_reader_receipt(
        claim,
        byte_index=[
            {
                "ordinal": 1,
                "logical_id": "synthetic-sec-byte",
                "relative_path": "synthetic-sec-byte.raw",
                "byte_count": 1,
                "sha256": _h("synthetic SEC byte"),
            }
        ],
        complete_marker_sha256=_h("synthetic SEC complete marker"),
    )
    reader_tip = _next_tip(
        state,
        claimed_tip,
        stage_sec_reader_receipts={request_hash: reader},
    )
    validate_reveal_store_current_tip_anchor_transition(claimed_tip, reader_tip)
    return claim, reader, reader_tip


def _rewrite_entry(
    raw_entry: dict,
    *,
    access: dict,
    identity_request: dict | None = None,
    prerequisite_evidence_sha256: str | None = None,
) -> dict:
    entry = copy.deepcopy(raw_entry)
    access_body = {
        key: value for key, value in access.items() if key != "stage_access_manifest_sha256"
    }
    access = {
        **access_body,
        "stage_access_manifest_sha256": canonical_sha256(access_body),
    }
    request = copy.deepcopy(entry["request"])
    request["stage_access_manifest_sha256"] = access[
        "stage_access_manifest_sha256"
    ]
    if prerequisite_evidence_sha256 is not None:
        request["prerequisite_stage_evidence_sha256"] = prerequisite_evidence_sha256
    if identity_request is not None:
        for field in (
            "registry_sha256",
            "registry_tip_sha256",
            "registered_entry_count",
            "historical_final_reveal_count_lower_bound",
            "attempt_id",
            "candidate_sha256",
            "candidate_design_sha256",
            "registry_entry_sha256",
        ):
            request[field] = identity_request[field]
        access["candidate"] = copy.deepcopy(
            entry["stage_access_manifest"]["candidate"]
        )
        access["candidate"].update(
            {
                "attempt_id": request["attempt_id"],
                "candidate_sha256": request["candidate_sha256"],
                "candidate_design_sha256": request["candidate_design_sha256"],
            }
        )
        access["output"]["namespace"] = (
            f"aapl-sec-gemma-{request['attempt_id']}-{request['stage']}"
        )
        access_body = {
            key: value
            for key, value in access.items()
            if key != "stage_access_manifest_sha256"
        }
        access["stage_access_manifest_sha256"] = canonical_sha256(access_body)
        request["stage_access_manifest_sha256"] = access[
            "stage_access_manifest_sha256"
        ]
    request_body = {
        key: value for key, value in request.items() if key != "request_sha256"
    }
    request = {**request_body, "request_sha256": canonical_sha256(request_body)}
    validation = copy.deepcopy(entry["prerequisite_validation"])
    validation.update(
        {
            "request_sha256": request["request_sha256"],
            "prerequisite_stage_evidence_sha256": request[
                "prerequisite_stage_evidence_sha256"
            ],
            "attempt_id": request["attempt_id"],
            "candidate_sha256": request["candidate_sha256"],
            "candidate_design_sha256": request["candidate_design_sha256"],
            "registry_entry_sha256": request["registry_entry_sha256"],
            "stage_access_manifest_sha256": request[
                "stage_access_manifest_sha256"
            ],
            "registry_sha256": request["registry_sha256"],
            "registry_tip_sha256": request["registry_tip_sha256"],
        }
    )
    validation["semantic_receipt"] = {
        **validation["semantic_receipt"],
        "request_sha256": request["request_sha256"],
    }
    validation["semantic_receipt_sha256"] = canonical_sha256(
        validation["semantic_receipt"]
    )
    validation_body = {
        key: value for key, value in validation.items() if key != "result_sha256"
    }
    validation = {
        **validation_body,
        "result_sha256": canonical_sha256(validation_body),
    }
    entry.update(
        {
            "request_sha256": request["request_sha256"],
            "request": request,
            "stage_access_manifest": access,
            "attempt_id": request["attempt_id"],
            "candidate_sha256": request["candidate_sha256"],
            "registry_entry_sha256": request["registry_entry_sha256"],
            "prerequisite_validation": validation,
        }
    )
    if entry["stage"] == "final":
        entry["final_touch_delta"] = 1
        entry["cumulative_actual_final_touch_count"] = 1
    entry_body = {key: value for key, value in entry.items() if key != "entry_sha256"}
    return {**entry_body, "entry_sha256": canonical_sha256(entry_body)}


def _state_from_entries(entries: list[dict], candidate: dict) -> dict:
    anchor = {"schema_version": "synthetic-store-anchor-v1", "root": _h("anchor")}
    actual_final = sum(entry["stage"] == "final" for entry in entries)
    ledger_body = {
        "schema_version": CONSUMPTION_LEDGER_SCHEMA_VERSION,
        "entries": entries,
        "chain": {
            "genesis_tip_sha256": _genesis(anchor),
            "tip_sha256": entries[-1]["entry_sha256"],
            "consumed_request_count": len(entries),
            "actual_final_touch_count": actual_final,
            "historical_final_reveal_count_lower_bound": 10,
            "repository_final_touch_count_lower_bound": 10 + actual_final,
        },
    }
    ledger = {**ledger_body, "ledger_sha256": canonical_sha256(ledger_body)}
    registry = {
        "schema_version": "synthetic-registry-v1",
        "entries": [
            {
                "entry_sha256": entries[0]["registry_entry_sha256"],
                "candidate_manifest": candidate,
            }
        ],
        "chain": {"tip_sha256": _h("registry tip"), "registered_entry_count": 1},
        "registry_sha256": _h("registry"),
    }
    state_body = {
        "schema_version": STORE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "anchor": anchor,
        "latest_registry": registry,
        "latest_registry_pin": {
            "schema_version": "synthetic-registry-pin-v1",
            "registry_sha256": registry["registry_sha256"],
            "tip_sha256": registry["chain"]["tip_sha256"],
            "registered_entry_count": 1,
        },
        "consumption_ledger": ledger,
    }
    return {**state_body, "state_sha256": canonical_sha256(state_body)}


def _bundle_for_state(state: dict) -> tuple[dict, dict, dict]:
    pin = derive_consumed_stage_store_state_pin(state)
    entry = state["consumption_ledger"]["entries"][-1]
    grant = build_consumed_stage_authorization_grant(
        authenticated_store_snapshot=state,
        external_store_state_pin=pin,
        expected_new_consumption_entry_sha256=entry["entry_sha256"],
    )
    body = {
        "schema_version": CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION,
        "authenticated_store_snapshot": state,
        "store_state_pin": pin,
        "authorization_grant": grant,
    }
    return entry, grant, {**body, "bundle_sha256": canonical_sha256(body)}


def _final_carry_in_fixture() -> tuple[dict, dict, dict, dict]:
    _root_child_state, _root_child_tip, _root_carry, root_binding = (
        _development_root_carry_in_fixture()
    )
    root_state = root_binding["root_state"]
    root_claim = root_binding["root_claim"]
    root_reader = root_binding["root_reader"]
    root_scope_hash = root_binding["root_scope_sha256"]
    root_market_claim = root_binding["root_market_claim"]
    root_market_reader = root_binding["root_market_reader"]
    root_market_reader_tip = root_binding["root_market_reader_tip"]
    root_execution_sources = root_binding["root_execution_sources"]
    candidate = root_state["latest_registry"]["entries"][0][
        "candidate_manifest"
    ]
    root_registry_entry = root_state["latest_registry"]["entries"][0]
    identity_request = {
        "registry_sha256": root_state["latest_registry"]["registry_sha256"],
        "registry_tip_sha256": root_state["latest_registry_pin"]["tip_sha256"],
        "registered_entry_count": root_state["latest_registry_pin"][
            "registered_entry_count"
        ],
        "historical_final_reveal_count_lower_bound": 10,
        "attempt_id": root_claim["attempt_id"],
        "candidate_sha256": root_claim["candidate_sha256"],
        "candidate_design_sha256": root_claim["candidate_design_sha256"],
        "registry_entry_sha256": root_registry_entry["entry_sha256"],
    }
    parent_documents = [
        {
            "accession_number": "0000320193-23-000111",
            "official_url": (
                "https://www.sec.gov/Archives/edgar/data/320193/"
                "000032019323000111/aapl-20230930.htm"
            ),
        },
        {
            "accession_number": "0000320193-23-000222",
            "official_url": (
                "https://www.sec.gov/Archives/edgar/data/320193/"
                "000032019323000222/aapl-20230701.htm"
            ),
        },
    ]
    parent_raw = _entry(
        sequence=1,
        prior_tip=_genesis(
            {"schema_version": "synthetic-store-anchor-v1", "root": _h("anchor")}
        ),
        stage="intermediate",
        candidate_sha256_override=candidate["candidate_sha256"],
    )
    parent_access = copy.deepcopy(parent_raw["stage_access_manifest"])
    parent_access["sec_access_plan"] = {
        "selection_policy": "all_and_only_requested_stage_universe_primary_documents",
        "method": "GET",
        "network_scope": "official_sec_https_only",
        "redirects_permitted": False,
        "retries_permitted": False,
        "cache_substitution_permitted": False,
        "document_count": 2,
        "accessions_sha256": canonical_sha256(
            [item["accession_number"] for item in parent_documents]
        ),
        "official_urls_sha256": canonical_sha256(
            [item["official_url"] for item in parent_documents]
        ),
        "documents": parent_documents,
    }
    parent_access["budgets"] = {
        "max_sec_requests": 2,
        "max_sec_response_bytes": 1_000_000,
        "max_sec_acquisition_seconds": 30.0,
    }
    parent_entry = _rewrite_entry(
        parent_raw,
        access=parent_access,
        identity_request=identity_request,
    )
    parent_state = _state_from_entries([parent_entry], candidate)
    parent_state["latest_registry"] = copy.deepcopy(root_state["latest_registry"])
    parent_state["latest_registry_pin"] = copy.deepcopy(
        root_state["latest_registry_pin"]
    )
    _rehash(parent_state, "state_sha256")
    parent_entry, parent_grant, parent_bundle = _bundle_for_state(parent_state)
    parent_request_hash = parent_entry["request_sha256"]
    parent_tip = build_reveal_store_current_tip_anchor(
        parent_state,
        revision=root_market_reader_tip["revision"] + 1,
        previous_tip_anchor_sha256=root_market_reader_tip["tip_anchor_sha256"],
        authorization_bundles={parent_request_hash: parent_bundle},
        development_sec_execution_claims={root_scope_hash: root_claim},
        development_sec_reader_receipts={root_scope_hash: root_reader},
        development_market_execution_claims={
            root_scope_hash: root_market_claim
        },
        development_market_reader_receipts={
            root_scope_hash: root_market_reader
        },
    )
    parent_claim = build_stage_sec_execution_claim(
        parent_bundle,
        independent_current_tip_anchor=parent_tip,
        execution_source_hashes=root_execution_sources,
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    parent_claim_tip = _next_tip(
        parent_state,
        parent_tip,
        stage_sec_execution_claims={parent_request_hash: parent_claim},
    )
    normalized = [
        (101, _h("parent 10-K normalized")),
        (102, _h("parent 10-Q normalized")),
    ]
    parent_reader = build_stage_sec_reader_receipt(
        parent_claim,
        byte_index=[
            {
                "ordinal": ordinal,
                "logical_id": f"document-{ordinal:04d}-normalized",
                "relative_path": f"document-{ordinal:04d}.normalized.txt",
                "byte_count": byte_count,
                "sha256": digest,
            }
            for ordinal, (byte_count, digest) in enumerate(normalized, start=1)
        ],
        complete_marker_sha256=_h("parent SEC marker"),
    )
    parent_reader_tip = _next_tip(
        parent_state,
        parent_claim_tip,
        stage_sec_reader_receipts={parent_request_hash: parent_reader},
    )
    parent_evidence_hash = _h("exact intermediate output evidence")
    parent_binding = _output_binding(parent_grant, salt="intermediate parent")
    parent_binding["output_stage_evidence_sha256"] = parent_evidence_hash
    parent_output = build_consumed_stage_output_receipt(
        parent_bundle,
        stage_sec_execution_claim=parent_claim,
        stage_sec_reader_receipt=parent_reader,
        **parent_binding,
    )

    records = [
        {
            "accession_number": document["accession_number"],
            "form": form,
            "availability_session": session,
            "artifact_stage": "intermediate",
            "source_record_sha256": _h(f"source record {form}"),
            "normalized_text_sha256": normalized[index][1],
            "normalized_text_bytes": normalized[index][0],
            "content_record_sha256": _h(f"content record {form}"),
            "content_manifest_sha256": _h("intermediate content manifest"),
        }
        for index, (document, form, session) in enumerate(
            zip(parent_documents, ("10-K", "10-Q"), ("2023-11-03", "2023-08-04"))
        )
    ]
    child_raw = _entry(
        sequence=2,
        prior_tip=parent_entry["entry_sha256"],
        stage="final",
        candidate_sha256_override=candidate["candidate_sha256"],
    )
    child_access = copy.deepcopy(child_raw["stage_access_manifest"])
    final_records = [
        record
        for record in root_binding["root_plan"]["corpus_universe_manifest"][
            "records"
        ]
        if record["artifact_stage"] == "final"
    ]
    child_documents = sorted(
        (
            {
                "accession_number": record["accession_number"],
                "official_url": (
                    "https://www.sec.gov/Archives/edgar/data/320193/"
                    f"{record['accession_number'].replace('-', '')}/"
                    f"{record['primary_document']}"
                ),
            }
            for record in final_records
        ),
        key=lambda document: document["accession_number"],
    )
    child_access["sec_access_plan"] = {
        "selection_policy": "all_and_only_requested_stage_universe_primary_documents",
        "method": "GET",
        "network_scope": "official_sec_https_only",
        "redirects_permitted": False,
        "retries_permitted": False,
        "cache_substitution_permitted": False,
        "document_count": len(child_documents),
        "accessions_sha256": canonical_sha256(
            [document["accession_number"] for document in child_documents]
        ),
        "official_urls_sha256": canonical_sha256(
            [document["official_url"] for document in child_documents]
        ),
        "documents": child_documents,
    }
    child_access["budgets"] = {
        "max_sec_requests": len(child_documents),
        "max_sec_response_bytes": 1_000_000,
        "max_sec_acquisition_seconds": 30.0,
    }
    child_access["prerequisite_evidence_pin"] = {
        "stage": "intermediate",
        "content_manifest_sha256": _h("intermediate content manifest"),
        "stage_artifact_sha256": _h("intermediate stage artifact"),
        "external_seal_receipt_sha256": _h("intermediate external seal"),
    }
    child_access["prior_same_form_carry_in"] = {
        "selection_policy": "latest_prerequisite_stage_filing_of_each_requested_stage_form",
        "artifact_scope": "sealed_normalized_text_only",
        "network_refetch_permitted": False,
        "write_permitted": False,
        "bound_by_prerequisite_stage_evidence_sha256": parent_evidence_hash,
        "prerequisite_content_manifest_sha256": _h(
            "intermediate content manifest"
        ),
        "record_count": 2,
        "records_sha256": canonical_sha256(records),
        "records": records,
    }
    child_access["scope"].update(
        {
            "general_cross_stage_access_permitted": False,
            "exact_prior_same_form_carry_in_read_permitted": True,
        }
    )
    child_entry = _rewrite_entry(
        child_raw,
        access=child_access,
        identity_request=parent_entry["request"],
        prerequisite_evidence_sha256=parent_evidence_hash,
    )
    child_state = _state_from_entries([parent_entry, child_entry], candidate)
    child_state["latest_registry"] = copy.deepcopy(root_state["latest_registry"])
    child_state["latest_registry_pin"] = copy.deepcopy(
        root_state["latest_registry_pin"]
    )
    _rehash(child_state, "state_sha256")
    child_entry, child_grant, child_bundle = _bundle_for_state(child_state)
    child_request_hash = child_entry["request_sha256"]
    child_tip = build_reveal_store_current_tip_anchor(
        child_state,
        revision=parent_reader_tip["revision"] + 1,
        previous_tip_anchor_sha256=parent_reader_tip["tip_anchor_sha256"],
        authorization_bundles={
            parent_request_hash: parent_bundle,
            child_request_hash: child_bundle,
        },
        consumed_stage_output_receipts={parent_request_hash: parent_output},
        stage_sec_execution_claims={parent_request_hash: parent_claim},
        stage_sec_reader_receipts={parent_request_hash: parent_reader},
        development_sec_execution_claims={root_scope_hash: root_claim},
        development_sec_reader_receipts={root_scope_hash: root_reader},
        development_market_execution_claims={
            root_scope_hash: root_market_claim
        },
        development_market_reader_receipts={
            root_scope_hash: root_market_reader
        },
    )
    child_claim = build_stage_sec_execution_claim(
        child_bundle,
        independent_current_tip_anchor=child_tip,
        execution_source_hashes=root_execution_sources,
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    child_claim_tip = _next_tip(
        child_state,
        child_tip,
        stage_sec_execution_claims={
            parent_request_hash: parent_claim,
            child_request_hash: child_claim,
        },
    )
    child_reader = build_stage_sec_reader_receipt(
        child_claim,
        byte_index=[
            {
                "ordinal": 1,
                "logical_id": "final-stage-byte",
                "relative_path": "final-stage-byte.raw",
                "byte_count": 1,
                "sha256": _h("final stage byte"),
            }
        ],
        complete_marker_sha256=_h("child SEC marker"),
    )
    child_reader_tip = _next_tip(
        child_state,
        child_claim_tip,
        stage_sec_reader_receipts={
            parent_request_hash: parent_reader,
            child_request_hash: child_reader,
        },
    )
    copied_index = [
        {
            "ordinal": ordinal,
            "logical_id": f"carry-in-{ordinal:04d}-normalized",
            "relative_path": f"carry-in-{ordinal:04d}.normalized.txt",
            "byte_count": byte_count,
            "sha256": digest,
        }
        for ordinal, (byte_count, digest) in enumerate(normalized, start=1)
    ]
    receipt = build_stage_carry_in_reader_receipt(
        child_bundle,
        stage_sec_execution_claim=child_claim,
        stage_sec_reader_receipt=child_reader,
        parent_authorization_bundle=parent_bundle,
        parent_stage_sec_execution_claim=parent_claim,
        parent_stage_sec_reader_receipt=parent_reader,
        parent_consumed_stage_output_receipt=parent_output,
        carry_in_byte_index=copied_index,
        carry_in_complete_marker_sha256=_h("carry-in complete marker"),
        reader_source_sha256=child_claim["execution_source_hashes"]["reveal_store"],
    )
    return child_state, child_reader_tip, receipt, {
        "byte_index": copied_index,
        "complete_marker_sha256": _h("carry-in complete marker"),
        "reader_source_sha256": child_claim["execution_source_hashes"][
            "reveal_store"
        ],
        "request_sha256": child_request_hash,
        "root_scope_sha256": root_scope_hash,
        "root_claim": root_claim,
        "root_reader": root_reader,
    }


@lru_cache(maxsize=1)
def _development_root_carry_in_fixture() -> tuple[dict, dict, dict, dict]:
    root_state, plan, root_tip, execution_sources = _development_root_context()
    root_claim = build_development_sec_execution_claim(
        root_state,
        development_content_root_plan=plan,
        independent_current_tip_anchor=root_tip,
        execution_source_hashes=execution_sources,
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    root_scope_hash = plan["development_root_scope_sha256"]
    root_claim_tip = _next_tip(
        root_state,
        root_tip,
        development_sec_execution_claims={root_scope_hash: root_claim},
    )
    content_inputs: list[dict] = []
    root_byte_index: list[dict] = []
    root_normalized_by_accession: dict[str, tuple[int, str]] = {}
    for document_ordinal, document in enumerate(
        plan["sec_access_plan"]["documents"],
        start=1,
    ):
        raw_bytes = 200 + document_ordinal
        normalized_bytes = 100 + document_ordinal
        raw_hash = _h(f"development raw {document['accession_number']}")
        normalized_hash = _h(
            f"development normalized {document['accession_number']}"
        )
        content_inputs.append(
            {
                "accession_number": document["accession_number"],
                "primary_document_sha256": raw_hash,
                "normalized_text_sha256": normalized_hash,
                "primary_document_bytes": raw_bytes,
                "normalized_text_bytes": normalized_bytes,
            }
        )
        root_normalized_by_accession[document["accession_number"]] = (
            normalized_bytes,
            normalized_hash,
        )
        for logical_suffix, path_suffix, byte_count, digest in (
            ("raw", "raw", raw_bytes, raw_hash),
            ("normalized", "normalized.txt", normalized_bytes, normalized_hash),
        ):
            root_byte_index.append(
                {
                    "ordinal": len(root_byte_index) + 1,
                    "logical_id": (
                        f"document-{document_ordinal:04d}-{logical_suffix}"
                    ),
                    "relative_path": (
                        f"document-{document_ordinal:04d}.{path_suffix}"
                    ),
                    "byte_count": byte_count,
                    "sha256": digest,
                }
            )
    content_manifest = build_stage_content_manifest(
        artifact_stage="development",
        corpus_universe_sha256=plan["corpus_universe_manifest"]["universe_sha256"],
        documents=content_inputs,
        universe_manifest=plan["corpus_universe_manifest"],
    )

    def encoded(value: dict) -> bytes:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    for logical_id, relative_path, payload in (
        (
            "request-receipts-json",
            "request-receipts.json",
            b"synthetic request receipts",
        ),
        ("byte-manifest-json", "byte-manifest.json", b"synthetic byte manifest"),
        (
            "corpus-universe-json",
            "corpus-universe.json",
            encoded(plan["corpus_universe_manifest"]),
        ),
        (
            "development-content-manifest-json",
            "development-content-manifest.json",
            encoded(content_manifest),
        ),
    ):
        root_byte_index.append(
            {
                "ordinal": len(root_byte_index) + 1,
                "logical_id": logical_id,
                "relative_path": relative_path,
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    root_reader = build_development_sec_reader_receipt(
        root_claim,
        content_manifest_sha256=content_manifest["content_manifest_sha256"],
        byte_index=root_byte_index,
        complete_marker_sha256=_h("development root complete marker"),
    )
    root_reader_tip = _next_tip(
        root_state,
        root_claim_tip,
        development_sec_reader_receipts={root_scope_hash: root_reader},
    )
    (
        root_market_claim_tip,
        root_market_reader_tip,
        root_market_claim,
        root_market_reader,
        root_market_binding,
    ) = _completed_development_market_lifecycle(
        root_state,
        root_reader_tip,
        scope_hash=root_scope_hash,
    )

    candidate = root_state["latest_registry"]["entries"][0]["candidate_manifest"]
    root_registry_entry = root_state["latest_registry"]["entries"][0]
    child_raw = _entry(
        sequence=1,
        prior_tip=root_state["consumption_ledger"]["chain"]["tip_sha256"],
        stage="intermediate",
        candidate_sha256_override=candidate["candidate_sha256"],
    )
    child_access = copy.deepcopy(child_raw["stage_access_manifest"])
    universe = plan["corpus_universe_manifest"]
    intermediate_records = sorted(
        (
            record
            for record in universe["records"]
            if record["artifact_stage"] == "intermediate"
        ),
        key=lambda record: (
            record["availability_session"],
            record["accession_number"],
        ),
    )
    child_documents = sorted(
        (
            {
                "accession_number": record["accession_number"],
                "official_url": (
                    "https://www.sec.gov/Archives/edgar/data/320193/"
                    f"{record['accession_number'].replace('-', '')}/"
                    f"{record['primary_document']}"
                ),
            }
            for record in intermediate_records
        ),
        key=lambda document: document["accession_number"],
    )
    child_access["sec_access_plan"] = {
        "selection_policy": "all_and_only_requested_stage_universe_primary_documents",
        "method": "GET",
        "network_scope": "official_sec_https_only",
        "redirects_permitted": False,
        "retries_permitted": False,
        "cache_substitution_permitted": False,
        "document_count": len(child_documents),
        "accessions_sha256": canonical_sha256(
            [document["accession_number"] for document in child_documents]
        ),
        "official_urls_sha256": canonical_sha256(
            [document["official_url"] for document in child_documents]
        ),
        "documents": child_documents,
    }
    child_access["budgets"] = {
        "max_sec_requests": len(child_documents),
        "max_sec_response_bytes": 1_000_000,
        "max_sec_acquisition_seconds": 30.0,
    }
    prerequisite_evidence_hash = _h("development stage evidence")
    child_access["prerequisite_evidence_pin"] = {
        "stage": "development",
        "content_manifest_sha256": content_manifest["content_manifest_sha256"],
        "stage_artifact_sha256": _h("development stage artifact"),
        "external_seal_receipt_sha256": _h("development external seal"),
    }
    child_access["corpus_provenance"] = {
        "frozen_base_universe": {
            "corpus_universe_sha256": universe["universe_sha256"],
            "corpus_universe_semantic_sha256": universe[
                "universe_semantic_sha256"
            ],
            "sec_catalog_artifact_sha256": universe["catalog_artifact_sha256"],
            "calendar_source_evidence_sha256": universe[
                "calendar_artifact_sha256"
            ],
            "session_calendar_sha256": universe["calendar_sessions_sha256"],
            "identity_role": "immutable_holdout_base",
        }
    }
    content_by_accession = {
        document["accession_number"]: document
        for document in content_manifest["documents"]
    }
    carry_records: list[dict] = []
    for form in ("10-K", "10-Q"):
        first_requested = next(
            record for record in intermediate_records if record["form"] == form
        )
        selected = max(
            (
                record
                for record in universe["records"]
                if record["form"] == form
                and (
                    record["availability_session"],
                    record["accession_number"],
                )
                < (
                    first_requested["availability_session"],
                    first_requested["accession_number"],
                )
            ),
            key=lambda record: (
                record["availability_session"],
                record["accession_number"],
            ),
        )
        content_record = content_by_accession[selected["accession_number"]]
        carry_records.append(
            {
                "accession_number": selected["accession_number"],
                "form": selected["form"],
                "availability_session": selected["availability_session"],
                "artifact_stage": selected["artifact_stage"],
                "source_record_sha256": selected["source_record_sha256"],
                "normalized_text_sha256": content_record[
                    "normalized_text_sha256"
                ],
                "normalized_text_bytes": content_record[
                    "normalized_text_bytes"
                ],
                "content_record_sha256": canonical_sha256(content_record),
                "content_manifest_sha256": content_manifest[
                    "content_manifest_sha256"
                ],
            }
        )
    child_access["prior_same_form_carry_in"] = {
        "selection_policy": "latest_prerequisite_stage_filing_of_each_requested_stage_form",
        "artifact_scope": "sealed_normalized_text_only",
        "network_refetch_permitted": False,
        "write_permitted": False,
        "bound_by_prerequisite_stage_evidence_sha256": (
            prerequisite_evidence_hash
        ),
        "prerequisite_content_manifest_sha256": content_manifest[
            "content_manifest_sha256"
        ],
        "record_count": 2,
        "records_sha256": canonical_sha256(carry_records),
        "records": carry_records,
    }
    child_access["scope"].update(
        {
            "prohibited_stages": ["development", "final"],
            "general_cross_stage_access_permitted": False,
            "exact_prior_same_form_carry_in_read_permitted": True,
        }
    )
    identity_request = {
        "registry_sha256": root_state["latest_registry"]["registry_sha256"],
        "registry_tip_sha256": root_state["latest_registry_pin"]["tip_sha256"],
        "registered_entry_count": root_state["latest_registry_pin"][
            "registered_entry_count"
        ],
        "historical_final_reveal_count_lower_bound": 10,
        "attempt_id": root_claim["attempt_id"],
        "candidate_sha256": root_claim["candidate_sha256"],
        "candidate_design_sha256": root_claim["candidate_design_sha256"],
        "registry_entry_sha256": root_registry_entry["entry_sha256"],
    }
    child_entry = _rewrite_entry(
        child_raw,
        access=child_access,
        identity_request=identity_request,
        prerequisite_evidence_sha256=prerequisite_evidence_hash,
    )
    child_state = _state_from_entries([child_entry], candidate)
    child_state["latest_registry"] = copy.deepcopy(root_state["latest_registry"])
    child_state["latest_registry_pin"] = copy.deepcopy(
        root_state["latest_registry_pin"]
    )
    _rehash(child_state, "state_sha256")
    child_entry, child_grant, child_bundle = _bundle_for_state(child_state)
    child_request_hash = child_entry["request_sha256"]
    child_tip = build_reveal_store_current_tip_anchor(
        child_state,
        revision=root_market_reader_tip["revision"] + 1,
        previous_tip_anchor_sha256=root_market_reader_tip["tip_anchor_sha256"],
        authorization_bundles={child_request_hash: child_bundle},
        development_sec_execution_claims={root_scope_hash: root_claim},
        development_sec_reader_receipts={root_scope_hash: root_reader},
        development_market_execution_claims={
            root_scope_hash: root_market_claim
        },
        development_market_reader_receipts={
            root_scope_hash: root_market_reader
        },
    )
    child_claim = build_stage_sec_execution_claim(
        child_bundle,
        independent_current_tip_anchor=child_tip,
        execution_source_hashes=execution_sources,
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    child_claim_tip = _next_tip(
        child_state,
        child_tip,
        stage_sec_execution_claims={child_request_hash: child_claim},
    )
    child_reader = build_stage_sec_reader_receipt(
        child_claim,
        byte_index=[
            {
                "ordinal": 1,
                "logical_id": "intermediate-stage-byte",
                "relative_path": "intermediate-stage-byte.raw",
                "byte_count": 1,
                "sha256": _h("intermediate stage byte"),
            }
        ],
        complete_marker_sha256=_h("intermediate SEC marker"),
    )
    child_reader_tip = _next_tip(
        child_state,
        child_claim_tip,
        stage_sec_reader_receipts={child_request_hash: child_reader},
    )
    copied_index = [
        {
            "ordinal": ordinal,
            "logical_id": f"carry-in-{ordinal:04d}-normalized",
            "relative_path": f"carry-in-{ordinal:04d}.normalized.txt",
            "byte_count": root_normalized_by_accession[record["accession_number"]][
                0
            ],
            "sha256": root_normalized_by_accession[record["accession_number"]][1],
        }
        for ordinal, record in enumerate(carry_records, start=1)
    ]
    receipt = build_development_root_carry_in_reader_receipt(
        child_bundle,
        stage_sec_execution_claim=child_claim,
        stage_sec_reader_receipt=child_reader,
        development_sec_execution_claim=root_claim,
        development_sec_reader_receipt=root_reader,
        carry_in_byte_index=copied_index,
        carry_in_complete_marker_sha256=_h(
            "development-root carry-in complete marker"
        ),
        reader_source_sha256=child_claim["execution_source_hashes"][
            "reveal_store"
        ],
    )
    return child_state, child_reader_tip, receipt, {
        "byte_index": copied_index,
        "complete_marker_sha256": _h(
            "development-root carry-in complete marker"
        ),
        "reader_source_sha256": child_claim["execution_source_hashes"][
            "reveal_store"
        ],
        "request_sha256": child_request_hash,
        "root_scope_sha256": root_scope_hash,
        "root_claim": root_claim,
        "root_reader": root_reader,
        "child_claim": child_claim,
        "child_reader": child_reader,
        "child_claim_tip": child_claim_tip,
        "root_state": root_state,
        "root_reader_tip": root_reader_tip,
        "root_market_claim_tip": root_market_claim_tip,
        "root_market_reader_tip": root_market_reader_tip,
        "root_market_claim": root_market_claim,
        "root_market_reader": root_market_reader,
        "root_market_binding": root_market_binding,
        "root_plan": plan,
        "root_execution_sources": execution_sources,
    }


def _model_byte_index(salt: str) -> list[dict]:
    return [
        {
            "ordinal": 1,
            "logical_id": "model-batch-manifest",
            "relative_path": "model-batch-manifest.json",
            "byte_count": 137,
            "sha256": _h(f"{salt}:model batch manifest"),
        }
    ]


_MODEL_MARKET_SHA_FIELDS = (
    "development_market_execution_claim_sha256",
    "development_market_reader_receipt_sha256",
    "development_market_acquisition_receipt_sha256",
    "development_market_acquisition_bundle_sha256",
    "development_market_acquisition_validation_sha256",
    "development_market_source_manifest_sha256",
    "development_market_stage_manifest_sha256",
    "development_market_source_reconciliation_sha256",
    "development_market_byte_index_sha256",
)

_DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_KEYS = {
    "schema_version",
    "contract_version",
    "plan_kind",
    "artifact_stage",
    "development_root_scope_sha256",
    "development_content_root_plan_sha256",
    "candidate_sha256",
    "candidate_design_sha256",
    "corpus_universe_sha256",
    "development_cutoff_session",
    "start_consumed_request_count",
    "development_sec_execution_claim_sha256",
    "development_sec_reader_receipt_sha256",
    "development_market_execution_claim_sha256",
    "development_market_reader_receipt_sha256",
    "development_market_acquisition_receipt_sha256",
    "development_market_acquisition_bundle_sha256",
    "development_market_acquisition_validation_sha256",
    "development_market_source_manifest_sha256",
    "development_market_stage_manifest_sha256",
    "development_market_source_reconciliation_sha256",
    "development_market_byte_index_sha256",
    "development_model_execution_claim_sha256",
    "development_model_reader_receipt_sha256",
    "event_count",
    "event_plan",
    "event_plan_sha256",
    "execution_source_hashes_sha256",
    "canonical_market_rows_required",
    "raw_market_output_permitted",
    "normalized_filing_text_output_permitted",
    "model_transport_envelope_output_permitted",
    "feature_rows_output_permitted",
    "outcome_access_permitted",
    "label_access_permitted",
    "training_membership_access_permitted",
    "learner_fit_permitted",
    "prediction_access_permitted",
    "holdout_access_permitted",
    "ledger_mutation_permitted",
    "stage_promotion_permitted",
    "feature_assembly_plan_sha256",
}

_DEVELOPMENT_LABEL_ASSEMBLY_PLAN_KEYS = {
    "schema_version",
    "contract_version",
    "plan_kind",
    "artifact_stage",
    "development_root_scope_sha256",
    "start_consumed_request_count",
    "source_feature_assembly_plan",
    "source_feature_assembly_plan_sha256",
    "calendar_sessions_sha256",
    "development_cutoff_session",
    "label_horizon_sessions",
    "label_entry_session_offset",
    "label_maturity_session_offset",
    "maturity_rule",
    "event_count",
    "maturity_plan",
    "maturity_plan_sha256",
    "matured_event_count",
    "unmatured_event_count",
    "canonical_market_rows_required",
    "development_outcome_derivation_permitted",
    "development_label_rows_output_permitted",
    "post_cutoff_market_access_permitted",
    "raw_market_output_permitted",
    "normalized_filing_text_output_permitted",
    "model_transport_envelope_output_permitted",
    "training_membership_access_permitted",
    "learner_fit_permitted",
    "prediction_access_permitted",
    "holdout_access_permitted",
    "ledger_mutation_permitted",
    "stage_promotion_permitted",
    "production_permitted",
    "label_assembly_plan_sha256",
}

_DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_KEYS = {
    "schema_version",
    "contract_version",
    "plan_kind",
    "artifact_stage",
    "development_root_scope_sha256",
    "start_consumed_request_count",
    "source_label_assembly_plan",
    "source_label_assembly_plan_sha256",
    "source_feature_assembly_plan_sha256",
    "candidate_sha256",
    "corpus_universe_sha256",
    "calendar_sessions_sha256",
    "development_cutoff_session",
    "label_horizon_sessions",
    "label_entry_session_offset",
    "label_maturity_session_offset",
    "event_count",
    "matured_event_count",
    "unmatured_event_count",
    "membership_view_count",
    "membership_view_specs",
    "membership_view_specs_sha256",
    "model_variant_count",
    "model_variant_specs",
    "model_variant_specs_sha256",
    "membership_maturity_rule",
    "feature_eligibility_rule",
    "membership_order_rule",
    "shared_variant_support_rule",
    "target_cost_bps",
    "binary_target_field",
    "binary_target_encoding",
    "edge_target_field",
    "minimum_training_row_count",
    "both_binary_classes_required",
    "semantic_ablation_feature_matrices_must_differ",
    "canonical_source_feature_rows_required",
    "canonical_source_label_rows_required",
    "source_feature_rows_access_permitted",
    "source_label_rows_access_permitted",
    "development_outcome_values_access_permitted",
    "development_outcome_derivation_permitted",
    "training_membership_access_permitted",
    "training_membership_rows_output_permitted",
    "training_feature_matrices_output_permitted",
    "training_target_vectors_output_permitted",
    "outcome_based_membership_filtering_permitted",
    "row_rebalancing_permitted",
    "post_cutoff_market_access_permitted",
    "raw_market_output_permitted",
    "compact_adjusted_open_paths_output_permitted",
    "normalized_filing_text_output_permitted",
    "model_transport_envelope_output_permitted",
    "learner_fit_permitted",
    "prediction_access_permitted",
    "holdout_access_permitted",
    "ledger_mutation_permitted",
    "stage_promotion_permitted",
    "production_permitted",
    "training_membership_assembly_plan_sha256",
}

_DEVELOPMENT_OOF_LEARNER_FIT_PLAN_KEYS = {
    "schema_version",
    "contract_version",
    "contract_sha256",
    "plan_kind",
    "artifact_stage",
    "development_root_scope_sha256",
    "start_consumed_request_count",
    "source_training_membership_assembly_plan",
    "source_training_membership_assembly_plan_sha256",
    "source_training_membership_projection_sha256",
    "source_training_membership_batch_sha256",
    "candidate_sha256",
    "corpus_universe_sha256",
    "calendar_sessions_sha256",
    "development_cutoff_session",
    "source_training_view_count",
    "source_training_view_ids",
    "source_training_view_specs_sha256",
    "authorized_training_view_count",
    "authorized_training_view_ids",
    "deferred_training_view_count",
    "deferred_training_views",
    "deferred_training_views_sha256",
    "model_variant_count",
    "model_variant_ids",
    "learner_fit_input_count",
    "learner_fit_input_specs",
    "learner_fit_input_specs_sha256",
    "learner_state_output_count",
    "learner_model_type",
    "learner_state_schema_version",
    "learner_config",
    "learner_config_sha256",
    "fit_order_rule",
    "maximum_fit_seconds",
    "canonical_source_training_membership_batch_required",
    "source_training_membership_batch_access_permitted",
    "authorized_training_feature_matrices_access_permitted",
    "authorized_training_target_vectors_access_permitted",
    "deterministic_learner_fit_permitted",
    "learner_state_output_permitted",
    "compact_fit_audit_output_permitted",
    "deterministic_refit_validation_permitted",
    "source_feature_batch_access_permitted",
    "source_label_batch_access_permitted",
    "development_outcome_derivation_permitted",
    "training_membership_derivation_permitted",
    "training_membership_mutation_permitted",
    "outcome_based_membership_filtering_permitted",
    "row_rebalancing_permitted",
    "deferred_training_view_fit_permitted",
    "hyperparameter_change_permitted",
    "solver_retry_permitted",
    "feature_selection_permitted",
    "model_transport_access_permitted",
    "network_access_permitted",
    "prediction_access_permitted",
    "candidate_selection_permitted",
    "threshold_action_access_permitted",
    "holdout_access_permitted",
    "ledger_mutation_permitted",
    "stage_promotion_permitted",
    "production_permitted",
    "development_oof_learner_fit_plan_sha256",
}

_DEVELOPMENT_OOF_FIT_INPUT_SPEC_KEYS = {
    "fit_ordinal",
    "source_training_view_ordinal",
    "training_view_id",
    "source_training_view_sha256",
    "head_variant",
    "training_row_count",
    "training_positive_count",
    "training_set_membership_sha256",
    "training_feature_matrix_sha256",
    "training_binary_target_sha256",
    "training_edge_target_sha256",
    "learner_input_context_sha256",
    "fit_metadata_template_sha256",
    "feature_schema_sha256",
    "train_label_maturity_through",
    "maximum_training_label_maturity_session",
    "fit_input_spec_sha256",
}

_DEVELOPMENT_OOF_ALLOWED_CAPABILITIES = (
    "canonical_source_training_membership_batch_required",
    "source_training_membership_batch_access_permitted",
    "authorized_training_feature_matrices_access_permitted",
    "authorized_training_target_vectors_access_permitted",
    "deterministic_learner_fit_permitted",
    "learner_state_output_permitted",
    "compact_fit_audit_output_permitted",
    "deterministic_refit_validation_permitted",
)

_DEVELOPMENT_OOF_DENIED_CAPABILITIES = (
    "source_feature_batch_access_permitted",
    "source_label_batch_access_permitted",
    "development_outcome_derivation_permitted",
    "training_membership_derivation_permitted",
    "training_membership_mutation_permitted",
    "outcome_based_membership_filtering_permitted",
    "row_rebalancing_permitted",
    "deferred_training_view_fit_permitted",
    "hyperparameter_change_permitted",
    "solver_retry_permitted",
    "feature_selection_permitted",
    "model_transport_access_permitted",
    "network_access_permitted",
    "prediction_access_permitted",
    "candidate_selection_permitted",
    "threshold_action_access_permitted",
    "holdout_access_permitted",
    "ledger_mutation_permitted",
    "stage_promotion_permitted",
    "production_permitted",
)

_DEVELOPMENT_OOF_PREDICTION_PLAN_KEYS = {
    "schema_version",
    "contract_version",
    "contract_sha256",
    "plan_kind",
    "artifact_stage",
    "development_root_scope_sha256",
    "start_consumed_request_count",
    "source_development_oof_learner_fit_plan_sha256",
    "source_development_oof_learner_fit_projection_sha256",
    "source_development_oof_learner_fit_batch_sha256",
    "source_training_membership_assembly_plan_sha256",
    "source_training_membership_projection_sha256",
    "source_training_membership_batch_sha256",
    "source_feature_assembly_plan_sha256",
    "source_feature_batch_sha256",
    "prediction_feature_batch_sha256",
    "candidate_sha256",
    "corpus_universe_sha256",
    "calendar_sessions_sha256",
    "development_cutoff_session",
    "source_event_count",
    "authorized_fold_count",
    "authorized_fold_ids",
    "prediction_fold_model_count",
    "prediction_fold_model_bundle_sha256",
    "prediction_fold_model_specs",
    "prediction_fold_model_specs_sha256",
    "model_variant_count",
    "model_variant_ids",
    "learner_state_count",
    "learner_model_type",
    "learner_state_schema_version",
    "learner_config_sha256",
    "feature_schema_sha256",
    "prediction_input_count",
    "available_prediction_input_count",
    "unavailable_prediction_input_count",
    "prediction_input_specs",
    "prediction_input_specs_sha256",
    "prediction_population_rule",
    "prediction_input_order_rule",
    "fold_state_usage_rule",
    "maximum_prediction_calls",
    "maximum_prediction_seconds",
    "canonical_prediction_fold_model_bundle_required",
    "canonical_prediction_feature_batch_required",
    "authorized_learner_state_access_permitted",
    "authorized_prediction_feature_row_access_permitted",
    "learner_state_deserialization_permitted",
    "deterministic_numeric_prediction_permitted",
    "raw_prediction_component_output_permitted",
    "unavailable_prediction_output_permitted",
    "compact_prediction_audit_output_permitted",
    "source_development_oof_learner_fit_batch_access_permitted",
    "source_training_membership_batch_access_permitted",
    "training_membership_rows_access_permitted",
    "training_feature_matrices_access_permitted",
    "training_target_vectors_access_permitted",
    "source_feature_batch_access_permitted",
    "source_label_batch_access_permitted",
    "label_access_permitted",
    "outcome_access_permitted",
    "post_decision_market_data_access_permitted",
    "post_2018_data_access_permitted",
    "deferred_training_view_access_permitted",
    "deferred_training_view_state_access_permitted",
    "learner_fit_permitted",
    "learner_state_update_permitted",
    "online_learning_permitted",
    "feature_mutation_permitted",
    "row_drop_permitted",
    "row_reordering_permitted",
    "prediction_retry_permitted",
    "model_transport_access_permitted",
    "network_access_permitted",
    "threshold_action_access_permitted",
    "candidate_selection_permitted",
    "policy_state_transition_permitted",
    "prediction_sealing_permitted",
    "label_release_permitted",
    "holdout_access_permitted",
    "ledger_mutation_permitted",
    "stage_promotion_permitted",
    "production_permitted",
    "development_oof_prediction_plan_sha256",
}

_DEVELOPMENT_OOF_PREDICTION_FOLD_MODEL_SPEC_KEYS = {
    "fold_ordinal",
    "fold_id",
    "prediction_window_first_date",
    "prediction_window_last_date",
    "source_learner_fit_view_sha256",
    "prediction_fold_context_sha256",
    "semantic_fit_record_sha256",
    "semantic_learner_state_sha256",
    "ablation_fit_record_sha256",
    "ablation_learner_state_sha256",
    "feature_schema_sha256",
    "prediction_fold_model_sha256",
    "prediction_fold_model_spec_sha256",
}

_DEVELOPMENT_OOF_PREDICTION_INPUT_SPEC_KEYS = {
    "prediction_ordinal",
    "source_event_ordinal",
    "fold_ordinal",
    "fold_id",
    "decision_session",
    "accession_number",
    "source_feature_row_sha256",
    "event_binding_sha256",
    "prediction_available",
    "unavailable_reason",
    "semantic_feature_schema_sha256",
    "semantic_feature_values_sha256",
    "ablation_feature_schema_sha256",
    "ablation_feature_values_sha256",
    "prediction_feature_input_sha256",
    "prediction_input_spec_sha256",
}

_DEVELOPMENT_OOF_PREDICTION_ALLOWED_CAPABILITIES = (
    "canonical_prediction_fold_model_bundle_required",
    "canonical_prediction_feature_batch_required",
    "authorized_learner_state_access_permitted",
    "authorized_prediction_feature_row_access_permitted",
    "learner_state_deserialization_permitted",
    "deterministic_numeric_prediction_permitted",
    "raw_prediction_component_output_permitted",
    "unavailable_prediction_output_permitted",
    "compact_prediction_audit_output_permitted",
)

_DEVELOPMENT_OOF_PREDICTION_DENIED_CAPABILITIES = (
    "source_development_oof_learner_fit_batch_access_permitted",
    "source_training_membership_batch_access_permitted",
    "training_membership_rows_access_permitted",
    "training_feature_matrices_access_permitted",
    "training_target_vectors_access_permitted",
    "source_feature_batch_access_permitted",
    "source_label_batch_access_permitted",
    "label_access_permitted",
    "outcome_access_permitted",
    "post_decision_market_data_access_permitted",
    "post_2018_data_access_permitted",
    "deferred_training_view_access_permitted",
    "deferred_training_view_state_access_permitted",
    "learner_fit_permitted",
    "learner_state_update_permitted",
    "online_learning_permitted",
    "feature_mutation_permitted",
    "row_drop_permitted",
    "row_reordering_permitted",
    "prediction_retry_permitted",
    "model_transport_access_permitted",
    "network_access_permitted",
    "threshold_action_access_permitted",
    "candidate_selection_permitted",
    "policy_state_transition_permitted",
    "prediction_sealing_permitted",
    "label_release_permitted",
    "holdout_access_permitted",
    "ledger_mutation_permitted",
    "stage_promotion_permitted",
    "production_permitted",
)


def _assert_model_market_bindings(value: dict, tip: dict, scope_hash: str) -> None:
    market_claim = tip["development_market_execution_claims"][scope_hash]
    market_reader = tip["development_market_reader_receipts"][scope_hash]
    expected = {
        "development_market_execution_claim_sha256": market_claim["claim_sha256"],
        "development_market_reader_receipt_sha256": market_reader[
            "receipt_sha256"
        ],
        "development_market_acquisition_receipt_sha256": market_reader[
            "acquisition_receipt_sha256"
        ],
        "development_market_acquisition_bundle_sha256": market_reader[
            "acquisition_bundle_sha256"
        ],
        "development_market_acquisition_validation_sha256": market_reader[
            "acquisition_validation_sha256"
        ],
        "development_market_source_manifest_sha256": market_reader[
            "source_manifest_sha256"
        ],
        "development_market_stage_manifest_sha256": market_reader[
            "market_stage_manifest_sha256"
        ],
        "development_market_source_reconciliation_sha256": market_reader[
            "source_reconciliation_sha256"
        ],
        "development_market_byte_index_sha256": market_reader[
            "byte_index_sha256"
        ],
    }
    assert {field: value[field] for field in _MODEL_MARKET_SHA_FIELDS} == expected


def _stage_model_fixture(
    stage: str = "intermediate",
) -> tuple[dict, dict, dict, dict, dict]:
    if stage == "intermediate":
        state, sec_reader_tip, carry, binding = _development_root_carry_in_fixture()
        carry_tip = _next_tip(
            state,
            sec_reader_tip,
            development_root_carry_in_reader_receipts={
                binding["request_sha256"]: carry
            },
        )
    elif stage == "final":
        state, sec_reader_tip, carry, binding = _final_carry_in_fixture()
        carry_tip = _next_tip(
            state,
            sec_reader_tip,
            stage_carry_in_reader_receipts={binding["request_sha256"]: carry},
        )
    else:
        raise AssertionError(stage)
    validate_reveal_store_current_tip_anchor_transition(sec_reader_tip, carry_tip)
    request_hash = binding["request_sha256"]
    bundle = carry_tip["authorization_bundles"][request_hash]
    claim = build_stage_model_execution_claim(
        bundle,
        independent_current_tip_anchor=carry_tip,
        execution_source_hashes=_bundle_model_source_hashes(bundle),
    )
    claim_tip = _next_tip(
        state,
        carry_tip,
        stage_model_execution_claims={request_hash: claim},
    )
    validate_reveal_store_current_tip_anchor_transition(carry_tip, claim_tip)
    byte_index = _model_byte_index(stage)
    reader = build_stage_model_reader_receipt(
        claim,
        byte_index=byte_index,
        complete_marker_sha256=_h(f"{stage}:model complete marker"),
    )
    return state, carry_tip, claim_tip, claim, {
        "request_sha256": request_hash,
        "reader": reader,
        "byte_index": byte_index,
        "complete_marker_sha256": _h(f"{stage}:model complete marker"),
    }


def _development_model_fixture() -> tuple[dict, dict, dict, dict, dict]:
    _state, _tip, _carry, binding = _development_root_carry_in_fixture()
    root_state = binding["root_state"]
    root_reader_tip = binding["root_market_reader_tip"]
    scope_hash = binding["root_scope_sha256"]
    claim = build_development_model_execution_claim(
        root_state,
        development_root_scope_sha256=scope_hash,
        independent_current_tip_anchor=root_reader_tip,
        execution_source_hashes=_development_model_source_hashes(root_state),
    )
    claim_tip = _next_tip(
        root_state,
        root_reader_tip,
        development_model_execution_claims={scope_hash: claim},
    )
    validate_reveal_store_current_tip_anchor_transition(
        root_reader_tip,
        claim_tip,
        authenticated_store_snapshot=root_state,
    )
    byte_index = _model_byte_index("development")
    reader = build_development_model_reader_receipt(
        claim,
        byte_index=byte_index,
        complete_marker_sha256=_h("development:model complete marker"),
    )
    return root_state, root_reader_tip, claim_tip, claim, {
        "scope_sha256": scope_hash,
        "reader": reader,
        "byte_index": byte_index,
        "complete_marker_sha256": _h("development:model complete marker"),
    }


def _development_feature_assembly_plan_fixture() -> tuple[dict, dict, dict, dict]:
    state, _root_tip, claim_tip, claim, binding = _development_model_fixture()
    scope_hash = binding["scope_sha256"]
    reader_tip = _next_tip(
        state,
        claim_tip,
        development_model_reader_receipts={scope_hash: binding["reader"]},
    )
    validate_reveal_store_current_tip_anchor_transition(claim_tip, reader_tip)
    plan = build_development_feature_assembly_plan(
        state,
        development_root_scope_sha256=scope_hash,
        independent_current_tip_anchor=reader_tip,
    )
    return state, reader_tip, claim, plan


def _development_label_assembly_plan_fixture() -> tuple[dict, dict, dict, dict]:
    state, reader_tip, _claim, feature_plan = (
        _development_feature_assembly_plan_fixture()
    )
    label_plan = build_development_label_assembly_plan(
        state,
        development_root_scope_sha256=feature_plan[
            "development_root_scope_sha256"
        ],
        source_feature_assembly_plan=feature_plan,
        independent_current_tip_anchor=reader_tip,
    )
    return state, reader_tip, feature_plan, label_plan


def _development_training_membership_assembly_plan_fixture() -> tuple[
    dict, dict, dict, dict
]:
    state, reader_tip, _feature_plan, label_plan = (
        _development_label_assembly_plan_fixture()
    )
    membership_plan = build_development_training_membership_assembly_plan(
        state,
        development_root_scope_sha256=label_plan[
            "development_root_scope_sha256"
        ],
        source_label_assembly_plan=label_plan,
        independent_current_tip_anchor=reader_tip,
    )
    return state, reader_tip, label_plan, membership_plan


def _development_oof_fit_input_specs(membership_plan: dict) -> list[dict]:
    specs: list[dict] = []
    for view in membership_plan["membership_view_specs"][:5]:
        row_count = view["view_ordinal"] + 5
        common = {
            "source_training_view_ordinal": view["view_ordinal"],
            "training_view_id": view["training_view_id"],
            "source_training_view_sha256": _h(
                f"OOF source view {view['training_view_id']}"
            ),
            "training_row_count": row_count,
            "training_positive_count": 2,
            "training_set_membership_sha256": _h(
                f"OOF membership {view['training_view_id']}"
            ),
            "training_binary_target_sha256": _h(
                f"OOF binary target {view['training_view_id']}"
            ),
            "training_edge_target_sha256": _h(
                f"OOF edge target {view['training_view_id']}"
            ),
            "learner_input_context_sha256": _h(
                f"OOF learner context {view['training_view_id']}"
            ),
            "feature_schema_sha256": _h("OOF frozen feature schema"),
            "train_label_maturity_through": view[
                "train_label_maturity_through"
            ],
            "maximum_training_label_maturity_session": view[
                "train_label_maturity_through"
            ],
        }
        for head_variant in ("semantic", "ablation"):
            body = {
                "fit_ordinal": len(specs) + 1,
                **common,
                "head_variant": head_variant,
                "training_feature_matrix_sha256": _h(
                    f"OOF {view['training_view_id']} {head_variant} matrix"
                ),
                "fit_metadata_template_sha256": _h(
                    f"OOF {view['training_view_id']} {head_variant} metadata"
                ),
            }
            specs.append(
                {**body, "fit_input_spec_sha256": canonical_sha256(body)}
            )
    return specs


def _development_oof_learner_fit_plan_fixture() -> tuple[
    dict, dict, dict, list[dict], dict
]:
    state, reader_tip, _label_plan, membership_plan = (
        _development_training_membership_assembly_plan_fixture()
    )
    fit_specs = _development_oof_fit_input_specs(membership_plan)
    plan = build_development_oof_learner_fit_plan(
        state,
        development_root_scope_sha256=membership_plan[
            "development_root_scope_sha256"
        ],
        source_training_membership_assembly_plan=membership_plan,
        source_training_membership_projection_sha256=_h(
            "OOF source membership projection"
        ),
        source_training_membership_batch_sha256=_h(
            "OOF source membership batch"
        ),
        fit_input_specs=fit_specs,
        independent_current_tip_anchor=reader_tip,
    )
    return state, reader_tip, membership_plan, fit_specs, plan


def _development_oof_prediction_fold_model_specs(
    source_fit_plan: dict,
) -> list[dict]:
    specs: list[dict] = []
    feature_schema_sha256 = source_fit_plan["learner_fit_input_specs"][0][
        "feature_schema_sha256"
    ]
    for fold_ordinal, (fold_id, _cutoff, first, last) in enumerate(
        DEVELOPMENT_FOLD_SPECS,
        start=1,
    ):
        semantic_fit = source_fit_plan["learner_fit_input_specs"][
            2 * (fold_ordinal - 1)
        ]
        body = {
            "fold_ordinal": fold_ordinal,
            "fold_id": fold_id,
            "prediction_window_first_date": first,
            "prediction_window_last_date": last,
            "source_learner_fit_view_sha256": semantic_fit[
                "source_training_view_sha256"
            ],
            "prediction_fold_context_sha256": _h(
                f"OOF prediction fold context {fold_id}"
            ),
            "semantic_fit_record_sha256": _h(
                f"OOF semantic fit record {fold_id}"
            ),
            "semantic_learner_state_sha256": _h(
                f"OOF semantic learner state {fold_id}"
            ),
            "ablation_fit_record_sha256": _h(
                f"OOF ablation fit record {fold_id}"
            ),
            "ablation_learner_state_sha256": _h(
                f"OOF ablation learner state {fold_id}"
            ),
            "feature_schema_sha256": feature_schema_sha256,
            "prediction_fold_model_sha256": _h(
                f"OOF prediction fold model {fold_id}"
            ),
        }
        specs.append(
            {
                **body,
                "prediction_fold_model_spec_sha256": canonical_sha256(body),
            }
        )
    return specs


def _development_oof_prediction_fold_for_session(
    session: str,
) -> tuple[int, str]:
    for fold_ordinal, (fold_id, _cutoff, first, last) in enumerate(
        DEVELOPMENT_FOLD_SPECS,
        start=1,
    ):
        if first <= session <= last:
            return fold_ordinal, fold_id
    raise AssertionError(session)


def _development_oof_prediction_input_specs(
    source_fit_plan: dict,
) -> list[dict]:
    feature_plan = source_fit_plan[
        "source_training_membership_assembly_plan"
    ]["source_label_assembly_plan"]["source_feature_assembly_plan"]
    feature_schema_sha256 = source_fit_plan["learner_fit_input_specs"][0][
        "feature_schema_sha256"
    ]
    events = [
        event
        for event in feature_plan["event_plan"]
        if any(
            first <= event["availability_session"] <= last
            for _fold_id, _cutoff, first, last in DEVELOPMENT_FOLD_SPECS
        )
    ]
    unavailable_reasons = (
        None,
        "missing_required_market_features",
        "missing_required_extraction_features",
        "missing_required_market_and_extraction_features",
    )
    specs: list[dict] = []
    for prediction_ordinal, event in enumerate(events, start=1):
        fold_ordinal, fold_id = _development_oof_prediction_fold_for_session(
            event["availability_session"]
        )
        reason = unavailable_reasons[(prediction_ordinal - 1) % 4]
        available = reason is None
        body = {
            "prediction_ordinal": prediction_ordinal,
            "source_event_ordinal": event["event_ordinal"],
            "fold_ordinal": fold_ordinal,
            "fold_id": fold_id,
            "decision_session": event["availability_session"],
            "accession_number": event["accession_number"],
            "source_feature_row_sha256": _h(
                f"OOF source feature row {event['accession_number']}"
            ),
            "event_binding_sha256": _h(
                f"OOF prediction event binding {event['accession_number']}"
            ),
            "prediction_available": available,
            "unavailable_reason": reason,
            "semantic_feature_schema_sha256": feature_schema_sha256,
            "semantic_feature_values_sha256": (
                _h(f"OOF semantic feature values {event['accession_number']}")
                if available
                else None
            ),
            "ablation_feature_schema_sha256": feature_schema_sha256,
            "ablation_feature_values_sha256": (
                _h(f"OOF ablation feature values {event['accession_number']}")
                if available
                else None
            ),
            "prediction_feature_input_sha256": _h(
                f"OOF compact prediction input {event['accession_number']}"
            ),
        }
        specs.append(
            {**body, "prediction_input_spec_sha256": canonical_sha256(body)}
        )
    return specs


def _development_oof_prediction_plan_fixture() -> tuple[
    dict, dict, dict, list[dict], list[dict], dict
]:
    state, reader_tip, _membership_plan, _fit_specs, source_fit_plan = (
        _development_oof_learner_fit_plan_fixture()
    )
    fold_specs = _development_oof_prediction_fold_model_specs(source_fit_plan)
    input_specs = _development_oof_prediction_input_specs(source_fit_plan)
    plan = _build_development_oof_prediction_plan_for_test(
        state,
        reader_tip,
        source_fit_plan,
        fold_specs,
        input_specs,
    )
    return state, reader_tip, source_fit_plan, fold_specs, input_specs, plan


def _build_development_oof_prediction_plan_for_test(
    state: dict,
    reader_tip: dict,
    source_fit_plan: dict,
    fold_specs: list[dict],
    input_specs: list[dict],
) -> dict:
    return build_development_oof_prediction_plan(
        state,
        development_root_scope_sha256=source_fit_plan[
            "development_root_scope_sha256"
        ],
        source_development_oof_learner_fit_plan=source_fit_plan,
        source_development_oof_learner_fit_projection_sha256=_h(
            "OOF learner-fit projection"
        ),
        source_development_oof_learner_fit_batch_sha256=_h(
            "OOF learner-fit batch"
        ),
        prediction_fold_model_bundle_sha256=_h(
            "OOF prediction fold-model bundle"
        ),
        prediction_fold_model_specs=fold_specs,
        source_feature_batch_sha256=_h("OOF source feature batch"),
        prediction_feature_batch_sha256=_h(
            "OOF compact prediction feature batch"
        ),
        prediction_input_specs=input_specs,
        independent_current_tip_anchor=reader_tip,
    )


def _development_policy_replay_input_specs(
    prediction_plan: dict,
) -> list[dict]:
    fold_specs = {
        item["fold_id"]: item
        for item in prediction_plan["prediction_fold_model_specs"]
    }
    result: list[dict] = []
    for ordinal, prediction_input in enumerate(
        prediction_plan["prediction_input_specs"], start=1
    ):
        fold = fold_specs[prediction_input["fold_id"]]
        body = {
            "schema_version": (
                "aapl-sec-gemma-development-policy-replay-input-spec-v1"
            ),
            "input_ordinal": ordinal,
            "source_raw_prediction_row_sha256": _h(
                f"policy replay raw row {ordinal}"
            ),
            "source_feature_row_sha256": prediction_input[
                "source_feature_row_sha256"
            ],
            "event_binding_sha256": prediction_input[
                "event_binding_sha256"
            ],
            "decision_session": prediction_input["decision_session"],
            "accession_number": prediction_input["accession_number"],
            "fold_id": prediction_input["fold_id"],
            "prediction_fold_context_sha256": fold[
                "prediction_fold_context_sha256"
            ],
            "semantic_learner_state_sha256": fold[
                "semantic_learner_state_sha256"
            ],
            "ablation_learner_state_sha256": fold[
                "ablation_learner_state_sha256"
            ],
            "prediction_status": (
                "available_pre_label"
                if prediction_input["prediction_available"]
                else "unavailable_pre_label"
            ),
            "unavailable_reason": prediction_input["unavailable_reason"],
            "numerical_components_sha256": _h(
                f"policy replay numerical components {ordinal}"
            ),
        }
        result.append(
            {
                **body,
                "policy_replay_input_spec_sha256": canonical_sha256(body),
            }
        )
    return result


def _development_policy_replay_plan_fixture() -> tuple[
    dict, dict, dict, list[dict], dict
]:
    state, reader_tip, _fit_plan, _fold_specs, _prediction_specs, prediction_plan = (
        _development_oof_prediction_plan_fixture()
    )
    replay_specs = _development_policy_replay_input_specs(prediction_plan)
    encoded_state = (
        json.dumps(
            state,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    encoded_tip = (
        json.dumps(
            reader_tip,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    plan = build_development_policy_replay_plan(
        state,
        development_root_scope_sha256=prediction_plan[
            "development_root_scope_sha256"
        ],
        authenticated_store_state_bytes_sha256=hashlib.sha256(
            encoded_state
        ).hexdigest(),
        source_development_oof_prediction_plan=prediction_plan,
        source_development_oof_prediction_projection_sha256=_h(
            "policy replay source prediction projection"
        ),
        source_development_oof_prediction_batch_sha256=_h(
            "policy replay source prediction batch"
        ),
        source_raw_prediction_rows_sha256=_h(
            "policy replay source raw rows"
        ),
        source_raw_prediction_tip_sha256=_h(
            "policy replay source raw tip"
        ),
        source_raw_prediction_row_count=len(replay_specs),
        policy_replay_input_specs=replay_specs,
        independent_current_tip_anchor=reader_tip,
        independent_current_tip_anchor_bytes_sha256=hashlib.sha256(
            encoded_tip
        ).hexdigest(),
    )
    return state, reader_tip, prediction_plan, replay_specs, plan


def _development_market_fixture() -> tuple[dict, dict, dict, dict, dict]:
    _state, _tip, _carry, root_binding = _development_root_carry_in_fixture()
    root_state = root_binding["root_state"]
    root_reader_tip = root_binding["root_reader_tip"]
    return (
        root_state,
        root_reader_tip,
        root_binding["root_market_claim_tip"],
        root_binding["root_market_claim"],
        copy.deepcopy(root_binding["root_market_binding"]),
    )


def test_development_market_claim_and_reader_are_exact_dedicated_lifecycle() -> None:
    state, prior_tip, claim_tip, claim, binding = _development_market_fixture()
    scope_hash = binding["scope_sha256"]
    assert claim["schema_version"] == DEVELOPMENT_MARKET_EXECUTION_CLAIM_SCHEMA_VERSION
    assert claim["authorized_stage"] == "development"
    assert claim["fixed_request_count"] == 6
    assert claim["market_access_permitted"] is True
    assert claim["external_network_access_permitted"] is True
    assert claim["paid_api_access_permitted"] is False
    assert claim["outcome_access_permitted"] is False
    assert claim["future_stage_access_permitted"] is False
    assert claim["reveal_request_consumption_permitted"] is False
    assert claim["consumption_ledger_mutation_permitted"] is False
    assert claim["effect_may_be_repeated_after_indeterminate_crash"] is False
    assert claim["market_acquisition_plan"] == binding["acquisition_plan"]
    assert claim_tip["state_sha256"] == prior_tip["state_sha256"]
    assert claim_tip["consumed_request_count"] == prior_tip["consumed_request_count"]
    assert (
        validate_development_market_execution_claim(
            claim,
            independent_current_tip_anchor=claim_tip,
        )
        == claim["claim_sha256"]
    )

    reader = binding["reader"]
    reader_tip = _next_tip(
        state,
        claim_tip,
        development_market_reader_receipts={scope_hash: reader},
    )
    prior, validated = validate_reveal_store_current_tip_anchor_transition(
        claim_tip,
        reader_tip,
    )
    assert prior == claim_tip
    assert validated == reader_tip
    assert reader["schema_version"] == DEVELOPMENT_MARKET_READER_RECEIPT_SCHEMA_VERSION
    assert reader["fresh_network_provenance_claimed"] is False
    assert reader["provider_response_normalization_replayed_by_store"] is True
    assert reader["owned_transport_attested_by_store"] is True
    assert len(reader["byte_index"]) == 22
    index_by_path = {
        item["relative_path"]: item for item in reader["byte_index"]
    }
    assert (
        index_by_path["source-manifest.json"]["sha256"]
        != reader["source_manifest_sha256"]
    )
    assert (
        validate_development_market_reader_receipt(
            reader,
            independent_current_tip_anchor=reader_tip,
            **binding["receipt_inputs"],
        )
        == reader["receipt_sha256"]
    )


def test_development_market_reader_rejects_previous_synthetic_one_file_index() -> None:
    _state, _prior_tip, _claim_tip, claim, binding = (
        _development_market_fixture()
    )
    receipt_inputs = copy.deepcopy(binding["receipt_inputs"])
    receipt_inputs["byte_index"] = [
        {
            "ordinal": 1,
            "logical_id": "market-acquisition-bundle",
            "relative_path": "market-acquisition-bundle.json",
            "byte_count": 379,
            "sha256": _h("development market acquisition bundle bytes"),
        }
    ]
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="exact ordered layout",
    ):
        build_development_market_reader_receipt(claim, **receipt_inputs)

    receipt_inputs = copy.deepcopy(binding["receipt_inputs"])
    receipt_inputs["owned_transport_attested_by_store"] = False
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="owned transport attestation",
    ):
        build_development_market_reader_receipt(claim, **receipt_inputs)


def test_development_market_active_claim_only_allows_exact_terminal_abort() -> None:
    state, _prior_tip, claim_tip, claim, binding = _development_market_fixture()
    scope_hash = binding["scope_sha256"]
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="Active development market execution claim",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            claim_tip,
            _next_tip(state, claim_tip),
        )

    abort = build_development_market_execution_abort(
        claim,
        reason="external_effect_failed_or_completion_unknown",
    )
    abort_tip = _next_tip(
        state,
        claim_tip,
        development_market_execution_aborts={scope_hash: abort},
    )
    validate_reveal_store_current_tip_anchor_transition(claim_tip, abort_tip)
    assert abort["schema_version"] == DEVELOPMENT_MARKET_EXECUTION_ABORT_SCHEMA_VERSION
    assert abort["external_effect_retry_permitted"] is False
    assert (
        validate_development_market_execution_abort(
            abort,
            independent_current_tip_anchor=abort_tip,
        )
        == abort["abort_sha256"]
    )


def test_development_market_claim_rejects_unpinned_or_changed_authority() -> None:
    state, prior_tip, claim_tip, claim, binding = _development_market_fixture()
    scope_hash = binding["scope_sha256"]
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="authenticated prior store snapshot",
    ):
        validate_reveal_store_current_tip_anchor_transition(prior_tip, claim_tip)

    changed_plan = copy.deepcopy(binding["acquisition_plan"])
    changed_plan["transport_authority"]["estimated_cost_usd"] = "0.01"
    _rehash(changed_plan, "acquisition_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="changed its exact authority",
    ):
        build_development_market_execution_claim(
            state,
            development_root_scope_sha256=scope_hash,
            independent_current_tip_anchor=prior_tip,
            market_acquisition_plan=changed_plan,
            execution_source_hashes=_development_market_source_hashes(state),
        )

    changed_sources = _development_market_source_hashes(state)
    changed_sources["market_acquirer"] = _h("unregistered market acquirer")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="acquisition plan differs from its execution source closure",
    ):
        build_development_market_execution_claim(
            state,
            development_root_scope_sha256=scope_hash,
            independent_current_tip_anchor=prior_tip,
            market_acquisition_plan=binding["acquisition_plan"],
            execution_source_hashes=changed_sources,
        )

    changed_reader = copy.deepcopy(binding["reader"])
    changed_reader["fresh_network_provenance_claimed"] = True
    _rehash(changed_reader, "receipt_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="reader receipt semantics changed",
    ):
        _next_tip(
            state,
            claim_tip,
            development_market_reader_receipts={scope_hash: changed_reader},
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


def test_final_carry_in_receipt_is_exact_dedicated_current_tip_append() -> None:
    state, reader_tip, receipt, binding = _final_carry_in_fixture()
    request_hash = binding["request_sha256"]
    next_tip = _next_tip(
        state,
        reader_tip,
        stage_carry_in_reader_receipts={request_hash: receipt},
    )
    prior, validated = validate_reveal_store_current_tip_anchor_transition(
        reader_tip,
        next_tip,
    )
    assert prior == reader_tip
    assert validated == next_tip
    assert receipt["schema_version"] == (
        STAGE_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION
    )
    assert receipt["authorized_stage"] == "final"
    assert receipt["input_prerequisite_stage"] == "intermediate"
    assert receipt["carry_in_record_count"] == 2
    assert [record["form"] for record in receipt["carry_in_records"]] == [
        "10-K",
        "10-Q",
    ]
    assert receipt["reader_output_recomputed_by_store"] is True
    assert receipt["network_refetch_permitted"] is False
    assert receipt["write_permitted"] is False
    assert receipt["general_cross_stage_access_permitted"] is False
    assert receipt["fresh_carry_in_provenance_claimed"] is False
    assert validate_stage_carry_in_reader_receipt(
        receipt,
        authenticated_store_snapshot=state,
        independent_current_tip_anchor=next_tip,
        carry_in_byte_index=binding["byte_index"],
        carry_in_complete_marker_sha256=binding[
            "complete_marker_sha256"
        ],
        reader_source_sha256=binding["reader_source_sha256"],
    ) == receipt["receipt_sha256"]
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="differs from revalidated durable inputs",
    ):
        validate_stage_carry_in_reader_receipt(
            receipt,
            authenticated_store_snapshot=state,
            independent_current_tip_anchor=next_tip,
            carry_in_byte_index=binding["byte_index"],
            carry_in_complete_marker_sha256=_h("substituted marker"),
            reader_source_sha256=binding["reader_source_sha256"],
        )


def test_development_root_carry_in_is_exact_dedicated_intermediate_append() -> None:
    state, reader_tip, receipt, binding = copy.deepcopy(
        _development_root_carry_in_fixture()
    )
    request_hash = binding["request_sha256"]
    next_tip = _next_tip(
        state,
        reader_tip,
        development_root_carry_in_reader_receipts={request_hash: receipt},
    )
    prior, validated = validate_reveal_store_current_tip_anchor_transition(
        reader_tip,
        next_tip,
    )

    assert prior == reader_tip
    assert validated == next_tip
    assert receipt["schema_version"] == (
        DEVELOPMENT_ROOT_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION
    )
    assert receipt["authorized_stage"] == "intermediate"
    assert receipt["input_prerequisite_stage"] == "development"
    assert receipt["development_root_scope_sha256"] == binding[
        "root_scope_sha256"
    ]
    assert receipt["development_sec_execution_claim_sha256"] == binding[
        "root_claim"
    ]["claim_sha256"]
    assert receipt["development_sec_reader_receipt_sha256"] == binding[
        "root_reader"
    ]["receipt_sha256"]
    assert receipt["child_sec_execution_claim_sha256"] == binding["child_claim"][
        "claim_sha256"
    ]
    assert receipt["child_sec_reader_receipt_sha256"] == binding["child_reader"][
        "receipt_sha256"
    ]
    assert [record["form"] for record in receipt["carry_in_records"]] == [
        "10-K",
        "10-Q",
    ]
    assert receipt["carry_in_record_count"] == 2
    assert receipt["network_refetch_permitted"] is False
    assert receipt["write_permitted"] is False
    assert receipt["general_cross_stage_access_permitted"] is False
    assert receipt["fresh_carry_in_provenance_claimed"] is False
    assert next_tip["stage_carry_in_reader_receipts"] == {}
    assert next_tip["development_root_carry_in_reader_receipts"] == {
        request_hash: receipt
    }
    assert validate_development_root_carry_in_reader_receipt(
        receipt,
        authenticated_store_snapshot=state,
        independent_current_tip_anchor=next_tip,
        carry_in_byte_index=binding["byte_index"],
        carry_in_complete_marker_sha256=binding["complete_marker_sha256"],
        reader_source_sha256=binding["reader_source_sha256"],
    ) == receipt["receipt_sha256"]


def test_development_root_carry_in_reconstructs_content_and_rejects_substitution() -> None:
    state, reader_tip, receipt, binding = copy.deepcopy(
        _development_root_carry_in_fixture()
    )
    child_bundle = reader_tip["authorization_bundles"][binding["request_sha256"]]
    substituted_root_reader = copy.deepcopy(binding["root_reader"])
    substituted_root_reader["content_manifest_sha256"] = _h(
        "substituted development content manifest"
    )
    _rehash(substituted_root_reader, "receipt_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="content identity differs from its root receipt",
    ):
        build_development_root_carry_in_reader_receipt(
            child_bundle,
            stage_sec_execution_claim=binding["child_claim"],
            stage_sec_reader_receipt=binding["child_reader"],
            development_sec_execution_claim=binding["root_claim"],
            development_sec_reader_receipt=substituted_root_reader,
            carry_in_byte_index=binding["byte_index"],
            carry_in_complete_marker_sha256=binding[
                "complete_marker_sha256"
            ],
            reader_source_sha256=binding["reader_source_sha256"],
        )

    substituted_index = copy.deepcopy(binding["byte_index"])
    substituted_index[0]["sha256"] = _h("substituted carry-in bytes")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="copied-byte index differs from its root bytes",
    ):
        build_development_root_carry_in_reader_receipt(
            child_bundle,
            stage_sec_execution_claim=binding["child_claim"],
            stage_sec_reader_receipt=binding["child_reader"],
            development_sec_execution_claim=binding["root_claim"],
            development_sec_reader_receipt=binding["root_reader"],
            carry_in_byte_index=substituted_index,
            carry_in_complete_marker_sha256=binding[
                "complete_marker_sha256"
            ],
            reader_source_sha256=binding["reader_source_sha256"],
        )


def test_development_root_carry_in_is_append_only_and_cannot_share_transition() -> None:
    state, reader_tip, receipt, binding = copy.deepcopy(
        _development_root_carry_in_fixture()
    )
    request_hash = binding["request_sha256"]
    persisted_tip = _next_tip(
        state,
        reader_tip,
        development_root_carry_in_reader_receipts={request_hash: receipt},
    )
    validate_reveal_store_current_tip_anchor_transition(reader_tip, persisted_tip)

    replaced = copy.deepcopy(receipt)
    replaced["carry_in_complete_marker_sha256"] = _h("replacement carry marker")
    _rehash(replaced, "receipt_sha256")
    replacement_tip = _next_tip(
        state,
        persisted_tip,
        development_root_carry_in_reader_receipts={request_hash: replaced},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="removed or changed a development-root carry-in receipt",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            persisted_tip,
            replacement_tip,
        )

    combined_tip = _next_tip(
        state,
        binding["child_claim_tip"],
        stage_sec_reader_receipts={request_hash: binding["child_reader"]},
        development_root_carry_in_reader_receipts={request_hash: receipt},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="dedicated tip-only transition",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            binding["child_claim_tip"],
            combined_tip,
        )


def test_development_root_carry_in_must_precede_same_request_stage_output() -> None:
    state, reader_tip, receipt, binding = copy.deepcopy(
        _development_root_carry_in_fixture()
    )
    request_hash = binding["request_sha256"]
    child_bundle = reader_tip["authorization_bundles"][request_hash]
    child_output = build_consumed_stage_output_receipt(
        child_bundle,
        stage_sec_execution_claim=binding["child_claim"],
        stage_sec_reader_receipt=binding["child_reader"],
        **_output_binding(
            child_bundle["authorization_grant"],
            salt="intermediate output before development-root carry",
        ),
    )
    output_first_tip = _next_tip(
        state,
        reader_tip,
        consumed_stage_output_receipts={request_hash: child_output},
    )
    validate_reveal_store_current_tip_anchor_transition(
        reader_tip,
        output_first_tip,
    )
    late_carry_tip = _next_tip(
        state,
        output_first_tip,
        development_root_carry_in_reader_receipts={request_hash: receipt},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="must precede the same request's consumed-stage output receipt",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            output_first_tip,
            late_carry_tip,
        )

    carry_first_tip = _next_tip(
        state,
        reader_tip,
        development_root_carry_in_reader_receipts={request_hash: receipt},
    )
    validate_reveal_store_current_tip_anchor_transition(
        reader_tip,
        carry_first_tip,
    )
    output_after_carry_tip = _next_tip(
        state,
        carry_first_tip,
        consumed_stage_output_receipts={request_hash: child_output},
    )
    validate_reveal_store_current_tip_anchor_transition(
        carry_first_tip,
        output_after_carry_tip,
    )
    assert validate_development_root_carry_in_reader_receipt(
        receipt,
        authenticated_store_snapshot=state,
        independent_current_tip_anchor=output_after_carry_tip,
        carry_in_byte_index=binding["byte_index"],
        carry_in_complete_marker_sha256=binding["complete_marker_sha256"],
        reader_source_sha256=binding["reader_source_sha256"],
    ) == receipt["receipt_sha256"]


def test_development_root_carry_in_rejects_abort_and_current_byte_mismatch() -> None:
    state, reader_tip, receipt, binding = copy.deepcopy(
        _development_root_carry_in_fixture()
    )
    request_hash = binding["request_sha256"]
    persisted_tip = _next_tip(
        state,
        reader_tip,
        development_root_carry_in_reader_receipts={request_hash: receipt},
    )
    abort = build_development_sec_execution_abort(
        binding["root_claim"],
        reason="durable_output_verification_failed",
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="cannot be both completed and aborted",
    ):
        _next_tip(
            state,
            reader_tip,
            development_sec_execution_aborts={
                binding["root_scope_sha256"]: abort
            },
            development_root_carry_in_reader_receipts={request_hash: receipt},
        )

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="differs from revalidated durable inputs",
    ):
        validate_development_root_carry_in_reader_receipt(
            receipt,
            authenticated_store_snapshot=state,
            independent_current_tip_anchor=persisted_tip,
            carry_in_byte_index=binding["byte_index"],
            carry_in_complete_marker_sha256=_h("substituted current marker"),
            reader_source_sha256=binding["reader_source_sha256"],
        )


def test_development_root_carry_in_rejects_transition_valid_reverse_chronology() -> None:
    child_state, normal_reader_tip, _receipt, binding = copy.deepcopy(
        _development_root_carry_in_fixture()
    )
    root_state, plan, root_tip, execution_sources = _development_root_context()
    request_hash = binding["request_sha256"]
    child_bundle = normal_reader_tip["authorization_bundles"][request_hash]
    child_tip = build_reveal_store_current_tip_anchor(
        child_state,
        revision=root_tip["revision"] + 1,
        previous_tip_anchor_sha256=root_tip["tip_anchor_sha256"],
        authorization_bundles={request_hash: child_bundle},
    )
    validate_reveal_store_current_tip_anchor_transition(root_tip, child_tip)
    child_claim = build_stage_sec_execution_claim(
        child_bundle,
        independent_current_tip_anchor=child_tip,
        execution_source_hashes=execution_sources,
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    child_claim_tip = _next_tip(
        child_state,
        child_tip,
        stage_sec_execution_claims={request_hash: child_claim},
    )
    validate_reveal_store_current_tip_anchor_transition(
        child_tip,
        child_claim_tip,
    )
    child_reader = build_stage_sec_reader_receipt(
        child_claim,
        byte_index=binding["child_reader"]["byte_index"],
        complete_marker_sha256=binding["child_reader"][
            "complete_marker_sha256"
        ],
    )
    child_reader_tip = _next_tip(
        child_state,
        child_claim_tip,
        stage_sec_reader_receipts={request_hash: child_reader},
    )
    validate_reveal_store_current_tip_anchor_transition(
        child_claim_tip,
        child_reader_tip,
    )

    reverse_root_claim = build_development_sec_execution_claim(
        child_state,
        development_content_root_plan=plan,
        independent_current_tip_anchor=child_reader_tip,
        execution_source_hashes=execution_sources,
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    root_scope_hash = reverse_root_claim["development_root_scope_sha256"]
    assert reverse_root_claim["start_consumed_request_count"] == 1
    assert reverse_root_claim["start_consumption_ledger_tip_sha256"] == (
        child_state["consumption_ledger"]["chain"]["tip_sha256"]
    )
    reverse_root_claim_tip = _next_tip(
        child_state,
        child_reader_tip,
        development_sec_execution_claims={root_scope_hash: reverse_root_claim},
    )
    validate_reveal_store_current_tip_anchor_transition(
        child_reader_tip,
        reverse_root_claim_tip,
    )
    reverse_root_reader = build_development_sec_reader_receipt(
        reverse_root_claim,
        content_manifest_sha256=binding["root_reader"][
            "content_manifest_sha256"
        ],
        byte_index=binding["root_reader"]["byte_index"],
        complete_marker_sha256=binding["root_reader"][
            "complete_marker_sha256"
        ],
    )
    reverse_root_reader_tip = _next_tip(
        child_state,
        reverse_root_claim_tip,
        development_sec_reader_receipts={root_scope_hash: reverse_root_reader},
    )
    validate_reveal_store_current_tip_anchor_transition(
        reverse_root_claim_tip,
        reverse_root_reader_tip,
    )

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="root did not predate child consumption",
    ):
        build_development_root_carry_in_reader_receipt(
            child_bundle,
            stage_sec_execution_claim=child_claim,
            stage_sec_reader_receipt=child_reader,
            development_sec_execution_claim=reverse_root_claim,
            development_sec_reader_receipt=reverse_root_reader,
            carry_in_byte_index=binding["byte_index"],
            carry_in_complete_marker_sha256=binding[
                "complete_marker_sha256"
            ],
            reader_source_sha256=binding["reader_source_sha256"],
        )


def test_output_receipt_schema_is_synchronized_with_parent_verifier() -> None:
    assert CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION == (
        "aapl-sec-gemma-consumed-stage-output-receipt-v2"
    )
    assert CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION == (
        verifier_module.CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION
    )


def test_output_receipt_builder_rejects_cross_request_sec_reader() -> None:
    state_one, _pin_one, entry_one, grant_one, tip_one = _grant_context(
        entry_count=1,
        include_sec_plan=True,
    )
    claim_one, _reader_one, _reader_tip_one = _completed_sec_output_ancestry(
        state_one,
        entry_one,
        tip_one,
    )
    state_two, _pin_two, entry_two, _grant_two, tip_two = _grant_context(
        entry_count=2,
        include_sec_plan=True,
    )
    _claim_two, reader_two, _reader_tip_two = _completed_sec_output_ancestry(
        state_two,
        entry_two,
        tip_two,
    )
    bundle_one = tip_one["authorization_bundles"][entry_one["request_sha256"]]
    with pytest.raises(SecFilingGemmaStageAuthorizationError):
        build_consumed_stage_output_receipt(
            bundle_one,
            stage_sec_execution_claim=claim_one,
            stage_sec_reader_receipt=reader_two,
            **_output_binding(grant_one, salt="cross-request reader"),
        )


def test_first_output_receipt_is_exact_current_tip_append_and_replays() -> None:
    state, _pin, entry, grant, grant_tip = _grant_context(include_sec_plan=True)
    request_hash = entry["request_sha256"]
    claim, reader, current_tip = _completed_sec_output_ancestry(
        state,
        entry,
        grant_tip,
    )
    bundle = current_tip["authorization_bundles"][request_hash]
    output_binding = _output_binding(grant)
    receipt = build_consumed_stage_output_receipt(
        bundle,
        stage_sec_execution_claim=claim,
        stage_sec_reader_receipt=reader,
        **output_binding,
    )
    next_tip = _next_tip(
        state,
        current_tip,
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


@pytest.mark.parametrize(
    "ancestry_field",
    ("sec_execution_claim_sha256", "sec_reader_receipt_sha256"),
)
def test_rehashed_output_receipt_cannot_substitute_sec_ancestry(
    ancestry_field: str,
) -> None:
    state, _pin, entry, grant, grant_tip = _grant_context(
        include_sec_plan=True
    )
    request_hash = entry["request_sha256"]
    claim, reader, reader_tip = _completed_sec_output_ancestry(
        state,
        entry,
        grant_tip,
    )
    bundle = reader_tip["authorization_bundles"][request_hash]
    receipt = build_consumed_stage_output_receipt(
        bundle,
        stage_sec_execution_claim=claim,
        stage_sec_reader_receipt=reader,
        **_output_binding(grant, salt=f"tamper {ancestry_field}"),
    )
    receipt[ancestry_field] = _h(f"substituted {ancestry_field}")
    _rehash(receipt, "output_receipt_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="crossed its grant or output boundary",
    ):
        _next_tip(
            state,
            reader_tip,
            consumed_stage_output_receipts={request_hash: receipt},
        )


def test_output_receipt_rejects_substitution_and_non_dedicated_append() -> None:
    state, _pin, entry, grant, grant_tip = _grant_context(include_sec_plan=True)
    request_hash = entry["request_sha256"]
    claim, reader, current_tip = _completed_sec_output_ancestry(
        state,
        entry,
        grant_tip,
    )
    bundle = current_tip["authorization_bundles"][request_hash]
    output_binding = _output_binding(grant)
    receipt = build_consumed_stage_output_receipt(
        bundle,
        stage_sec_execution_claim=claim,
        stage_sec_reader_receipt=reader,
        **output_binding,
    )
    persisted_tip = _next_tip(
        state,
        current_tip,
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
            stage_sec_execution_claims=current_tip[
                "stage_sec_execution_claims"
            ],
            stage_sec_reader_receipts=current_tip["stage_sec_reader_receipts"],
        )
        validate_reveal_store_current_tip_anchor_transition(current_tip, mixed_tip)


def test_output_receipt_cannot_append_after_its_grant_tip_is_stale() -> None:
    state, _pin, entry, grant, grant_tip = _grant_context(include_sec_plan=True)
    request_hash = entry["request_sha256"]
    claim, reader, current_tip = _completed_sec_output_ancestry(
        state,
        entry,
        grant_tip,
    )
    bundle = current_tip["authorization_bundles"][request_hash]
    receipt = build_consumed_stage_output_receipt(
        bundle,
        stage_sec_execution_claim=claim,
        stage_sec_reader_receipt=reader,
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
        stage_sec_execution_claims=current_tip["stage_sec_execution_claims"],
        stage_sec_reader_receipts=current_tip["stage_sec_reader_receipts"],
    )
    validate_reveal_store_current_tip_anchor_transition(current_tip, advanced_tip)

    polluted_tip = build_reveal_store_current_tip_anchor(
        advanced_state,
        revision=advanced_tip["revision"] + 1,
        previous_tip_anchor_sha256=advanced_tip["tip_anchor_sha256"],
        authorization_bundles=advanced_tip["authorization_bundles"],
        trusted_stage_content_pins=advanced_tip["trusted_stage_content_pins"],
        consumed_stage_output_receipts={request_hash: receipt},
        stage_sec_execution_claims=advanced_tip["stage_sec_execution_claims"],
        stage_sec_reader_receipts=advanced_tip["stage_sec_reader_receipts"],
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


def test_sec_execution_claim_is_an_exact_dedicated_current_grant_transition() -> None:
    state, _pin, entry, grant, prior_tip = _grant_context(
        include_sec_plan=True
    )
    request_hash = entry["request_sha256"]
    bundle = prior_tip["authorization_bundles"][request_hash]
    claim = build_stage_sec_execution_claim(
        bundle,
        independent_current_tip_anchor=prior_tip,
        execution_source_hashes=_execution_source_hashes(),
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    claimed_tip = _next_tip(
        state,
        prior_tip,
        stage_sec_execution_claims={request_hash: claim},
    )

    validated_prior, validated_claimed = (
        validate_reveal_store_current_tip_anchor_transition(
            prior_tip,
            claimed_tip,
        )
    )

    assert validated_prior == prior_tip
    assert validated_claimed == claimed_tip
    assert claimed_tip["revision"] == prior_tip["revision"] + 1
    assert claimed_tip["state_sha256"] == prior_tip["state_sha256"]
    assert claimed_tip["stage_sec_execution_claims"] == {
        request_hash: claim
    }
    assert claim["request_sha256"] == request_hash
    assert claim["start_current_tip_anchor_sha256"] == prior_tip[
        "tip_anchor_sha256"
    ]
    assert claim["authorization_grant_sha256"] == grant[
        "authorization_grant_sha256"
    ]
    assert claim["sec_component_id"] == SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID
    assert claim["sec_user_agent_sha256"] == SEC_USER_AGENT_SHA256
    assert claim["effect_may_be_repeated_after_indeterminate_crash"] is False

    _exact_bundle, _exact_grant, component_plan = _sec_component_plan_from_bundle(
        bundle
    )
    assert component_plan["authorized_max_sec_response_bytes"] == 1_000_000
    assert component_plan["owned_sec_raw_batch_max_bytes"] == (
        OWNED_SEC_RAW_BATCH_MAX_BYTES
    )
    assert component_plan["max_sec_response_bytes"] == 1_000_000


@pytest.mark.parametrize("substituted_role", ("runner", "sec_corpus_selector", "sec_acquirer"))
def test_sec_execution_claim_rejects_source_bytes_outside_candidate_pins(
    substituted_role: str,
) -> None:
    _state, _pin, entry, _grant, prior_tip = _grant_context(
        include_sec_plan=True
    )
    bundle = prior_tip["authorization_bundles"][entry["request_sha256"]]

    source_hashes = _execution_source_hashes()
    source_hashes[substituted_role] = _h(f"substituted {substituted_role}")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="differ from the registered candidate",
    ):
        build_stage_sec_execution_claim(
            bundle,
            independent_current_tip_anchor=prior_tip,
            execution_source_hashes=source_hashes,
            sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
        )


def test_active_sec_claim_blocks_nonterminal_and_second_claim_transitions() -> None:
    state, _pin, entry, _grant, prior_tip = _grant_context(
        include_sec_plan=True
    )
    request_hash = entry["request_sha256"]
    bundle = prior_tip["authorization_bundles"][request_hash]
    claim = build_stage_sec_execution_claim(
        bundle,
        independent_current_tip_anchor=prior_tip,
        execution_source_hashes=_execution_source_hashes(),
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    claimed_tip = _next_tip(
        state,
        prior_tip,
        stage_sec_execution_claims={request_hash: claim},
    )
    validate_reveal_store_current_tip_anchor_transition(prior_tip, claimed_tip)

    nonterminal_tip = _next_tip(state, claimed_tip)
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="Active SEC execution claim blocks every transition",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            claimed_tip,
            nonterminal_tip,
        )

    reader = build_stage_sec_reader_receipt(
        claim,
        byte_index=[
            {
                "ordinal": 1,
                "logical_id": "active-claim-byte",
                "relative_path": "active-claim-byte.raw",
                "byte_count": 1,
                "sha256": _h("active claim byte"),
            }
        ],
        complete_marker_sha256=_h("active claim complete marker"),
    )
    output_receipt = build_consumed_stage_output_receipt(
        bundle,
        stage_sec_execution_claim=claim,
        stage_sec_reader_receipt=reader,
        **_output_binding(bundle["authorization_grant"], salt="active claim"),
    )
    output_tip = _next_tip(
        state,
        claimed_tip,
        consumed_stage_output_receipts={request_hash: output_receipt},
        stage_sec_reader_receipts={request_hash: reader},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="Active SEC execution claim blocks every transition",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            claimed_tip,
            output_tip,
        )

    second_claim = copy.deepcopy(claim)
    second_request_hash = _h("cross-request second claim")
    second_claim["request_sha256"] = second_request_hash
    _rehash(second_claim, "claim_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="lacks its persisted grant bundle|stored under another request",
    ):
        _next_tip(
            state,
            claimed_tip,
            stage_sec_execution_claims={
                request_hash: claim,
                second_request_hash: second_claim,
            },
        )


def test_sec_claim_rejects_a_stale_start_tip_and_cross_request_substitution() -> None:
    state, _pin, entry, _grant, prior_tip = _grant_context(
        include_sec_plan=True
    )
    request_hash = entry["request_sha256"]
    bundle = prior_tip["authorization_bundles"][request_hash]
    stale_claim = build_stage_sec_execution_claim(
        bundle,
        independent_current_tip_anchor=prior_tip,
        execution_source_hashes=_execution_source_hashes(),
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    advanced_tip = _next_tip(state, prior_tip)
    validate_reveal_store_current_tip_anchor_transition(prior_tip, advanced_tip)
    stale_claim_tip = _next_tip(
        state,
        advanced_tip,
        stage_sec_execution_claims={request_hash: stale_claim},
    )

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="does not bind the exact current grant tip",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            advanced_tip,
            stale_claim_tip,
        )

    crossed = copy.deepcopy(stale_claim)
    crossed["request_sha256"] = _h("other request")
    _rehash(crossed, "claim_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="stored under another request|lacks its persisted grant bundle",
    ):
        _next_tip(
            state,
            prior_tip,
            stage_sec_execution_claims={request_hash: crossed},
        )


def test_active_sec_claim_allows_only_its_exact_terminal_receipt_or_abort() -> None:
    state, _pin, entry, _grant, prior_tip = _grant_context(
        include_sec_plan=True
    )
    request_hash = entry["request_sha256"]
    bundle = prior_tip["authorization_bundles"][request_hash]
    claim = build_stage_sec_execution_claim(
        bundle,
        independent_current_tip_anchor=prior_tip,
        execution_source_hashes=_execution_source_hashes(),
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    claimed_tip = _next_tip(
        state,
        prior_tip,
        stage_sec_execution_claims={request_hash: claim},
    )
    validate_reveal_store_current_tip_anchor_transition(prior_tip, claimed_tip)
    payload = b"fixed SEC filing bytes"
    byte_index = [
        {
            "ordinal": 1,
            "logical_id": "0000320193-24-000123",
            "relative_path": "0001-primary.htm",
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    receipt = build_stage_sec_reader_receipt(
        claim,
        byte_index=byte_index,
        complete_marker_sha256=_h("canonical complete marker"),
    )
    assert receipt["sec_user_agent_sha256"] == SEC_USER_AGENT_SHA256
    completed_tip = _next_tip(
        state,
        claimed_tip,
        stage_sec_reader_receipts={request_hash: receipt},
    )
    validate_reveal_store_current_tip_anchor_transition(
        claimed_tip,
        completed_tip,
    )
    assert completed_tip["stage_sec_reader_receipts"][request_hash] == receipt

    abort = build_stage_sec_execution_abort(
        claim,
        reason="external_effect_failed_or_completion_unknown",
    )
    aborted_tip = _next_tip(
        state,
        claimed_tip,
        stage_sec_execution_aborts={request_hash: abort},
    )
    validate_reveal_store_current_tip_anchor_transition(
        claimed_tip,
        aborted_tip,
    )
    assert abort["external_effect_retry_permitted"] is False

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="cannot be both completed and aborted",
    ):
        _next_tip(
            state,
            completed_tip,
            stage_sec_execution_aborts={request_hash: abort},
        )


def test_development_sec_claim_is_request_free_exact_and_dedicated() -> None:
    state, plan, prior_tip, execution_sources = _development_root_context()
    claim = build_development_sec_execution_claim(
        state,
        development_content_root_plan=plan,
        independent_current_tip_anchor=prior_tip,
        execution_source_hashes=execution_sources,
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    repeated = build_development_sec_execution_claim(
        state,
        development_content_root_plan=plan,
        independent_current_tip_anchor=prior_tip,
        execution_source_hashes=execution_sources,
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    scope_hash = plan["development_root_scope_sha256"]
    claimed_tip = _next_tip(
        state,
        prior_tip,
        development_sec_execution_claims={scope_hash: claim},
    )

    validate_reveal_store_current_tip_anchor_transition(prior_tip, claimed_tip)

    assert repeated == claim
    assert claim["schema_version"] == DEVELOPMENT_SEC_EXECUTION_CLAIM_SCHEMA_VERSION
    assert claim["development_content_root_plan"] == plan
    assert claim["development_content_root_plan_sha256"] == plan[
        "development_content_root_plan_sha256"
    ]
    assert claim["development_root_scope_sha256"] == scope_hash
    assert claim["sec_component_id"] == DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID
    assert claim["start_current_tip_anchor_sha256"] == prior_tip[
        "tip_anchor_sha256"
    ]
    assert claim["start_consumption_ledger_tip_sha256"] == prior_tip[
        "consumption_ledger_tip_sha256"
    ]
    for field in (
        "authorizes_outcome_access",
        "market_access_permitted",
        "model_access_permitted",
        "future_stage_access_permitted",
        "reveal_request_consumption_permitted",
        "consumption_ledger_mutation_permitted",
        "effect_may_be_repeated_after_indeterminate_crash",
    ):
        assert claim[field] is False
    assert claimed_tip["development_sec_execution_claims"] == {
        scope_hash: claim
    }
    assert claimed_tip["development_sec_reader_receipts"] == {}
    assert claimed_tip["development_sec_execution_aborts"] == {}


def test_active_development_sec_claim_allows_only_exact_terminal_artifact() -> None:
    state, plan, prior_tip, execution_sources = _development_root_context()
    claim = build_development_sec_execution_claim(
        state,
        development_content_root_plan=plan,
        independent_current_tip_anchor=prior_tip,
        execution_source_hashes=execution_sources,
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    scope_hash = plan["development_root_scope_sha256"]
    claimed_tip = _next_tip(
        state,
        prior_tip,
        development_sec_execution_claims={scope_hash: claim},
    )
    validate_reveal_store_current_tip_anchor_transition(prior_tip, claimed_tip)
    byte_index = [
        {
            "ordinal": 1,
            "logical_id": "development-content-manifest",
            "relative_path": "development-content-manifest.json",
            "byte_count": 128,
            "sha256": _h("development content manifest bytes"),
        }
    ]
    receipt = build_development_sec_reader_receipt(
        claim,
        content_manifest_sha256=_h("development content manifest"),
        byte_index=byte_index,
        complete_marker_sha256=_h("development complete marker"),
    )
    assert receipt == build_development_sec_reader_receipt(
        claim,
        content_manifest_sha256=_h("development content manifest"),
        byte_index=byte_index,
        complete_marker_sha256=_h("development complete marker"),
    )
    assert receipt["schema_version"] == DEVELOPMENT_SEC_READER_RECEIPT_SCHEMA_VERSION
    completed_tip = _next_tip(
        state,
        claimed_tip,
        development_sec_reader_receipts={scope_hash: receipt},
    )
    validate_reveal_store_current_tip_anchor_transition(
        claimed_tip,
        completed_tip,
    )

    abort = build_development_sec_execution_abort(
        claim,
        reason="external_effect_failed_or_completion_unknown",
    )
    assert abort["schema_version"] == DEVELOPMENT_SEC_EXECUTION_ABORT_SCHEMA_VERSION
    aborted_tip = _next_tip(
        state,
        claimed_tip,
        development_sec_execution_aborts={scope_hash: abort},
    )
    validate_reveal_store_current_tip_anchor_transition(
        claimed_tip,
        aborted_tip,
    )

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="cannot be both completed and aborted",
    ):
        _next_tip(
            state,
            completed_tip,
            development_sec_execution_aborts={scope_hash: abort},
        )


def test_active_development_claim_blocks_noop_and_combined_append() -> None:
    state, plan, prior_tip, execution_sources = _development_root_context()
    claim = build_development_sec_execution_claim(
        state,
        development_content_root_plan=plan,
        independent_current_tip_anchor=prior_tip,
        execution_source_hashes=execution_sources,
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    scope_hash = plan["development_root_scope_sha256"]
    claimed_tip = _next_tip(
        state,
        prior_tip,
        development_sec_execution_claims={scope_hash: claim},
    )
    validate_reveal_store_current_tip_anchor_transition(prior_tip, claimed_tip)

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="Active development SEC execution claim blocks every transition",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            claimed_tip,
            _next_tip(state, claimed_tip),
        )

    receipt = build_development_sec_reader_receipt(
        claim,
        content_manifest_sha256=_h("combined content manifest"),
        byte_index=[
            {
                "ordinal": 1,
                "logical_id": "combined-root-byte",
                "relative_path": "combined-root-byte.json",
                "byte_count": 1,
                "sha256": _h("combined root byte"),
            }
        ],
        complete_marker_sha256=_h("combined root marker"),
    )
    combined_tip = _next_tip(
        state,
        prior_tip,
        development_sec_execution_claims={scope_hash: claim},
        development_sec_reader_receipts={scope_hash: receipt},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="require separate transitions",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            prior_tip,
            combined_tip,
        )


def test_development_claim_rejects_stale_tip_source_and_plan_substitution() -> None:
    state, plan, prior_tip, execution_sources = _development_root_context()
    claim = build_development_sec_execution_claim(
        state,
        development_content_root_plan=plan,
        independent_current_tip_anchor=prior_tip,
        execution_source_hashes=execution_sources,
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    scope_hash = plan["development_root_scope_sha256"]
    advanced_tip = _next_tip(state, prior_tip)
    validate_reveal_store_current_tip_anchor_transition(prior_tip, advanced_tip)
    stale_tip = _next_tip(
        state,
        advanced_tip,
        development_sec_execution_claims={scope_hash: claim},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="does not bind the exact current registry and ledger tip",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            advanced_tip,
            stale_tip,
        )

    substituted_sources = copy.deepcopy(execution_sources)
    substituted_sources["runner"] = _h("substituted development runner")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="differ from the registered candidate",
    ):
        build_development_sec_execution_claim(
            state,
            development_content_root_plan=plan,
            independent_current_tip_anchor=prior_tip,
            execution_source_hashes=substituted_sources,
            sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
        )

    changed_plan = copy.deepcopy(plan)
    changed_plan["scope"]["future_stage_access_permitted"] = True
    _rehash(changed_plan, "development_content_root_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="authorization scope changed",
    ):
        build_development_sec_execution_claim(
            state,
            development_content_root_plan=changed_plan,
            independent_current_tip_anchor=prior_tip,
            execution_source_hashes=execution_sources,
            sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
        )


def test_development_sec_maps_are_append_only_after_terminal_receipt() -> None:
    state, plan, prior_tip, execution_sources = _development_root_context()
    claim = build_development_sec_execution_claim(
        state,
        development_content_root_plan=plan,
        independent_current_tip_anchor=prior_tip,
        execution_source_hashes=execution_sources,
        sec_user_agent_sha256=SEC_USER_AGENT_SHA256,
    )
    scope_hash = plan["development_root_scope_sha256"]
    claimed_tip = _next_tip(
        state,
        prior_tip,
        development_sec_execution_claims={scope_hash: claim},
    )
    receipt = build_development_sec_reader_receipt(
        claim,
        content_manifest_sha256=_h("append-only content manifest"),
        byte_index=[
            {
                "ordinal": 1,
                "logical_id": "append-only-root-byte",
                "relative_path": "append-only-root-byte.json",
                "byte_count": 1,
                "sha256": _h("append-only root byte"),
            }
        ],
        complete_marker_sha256=_h("append-only root marker"),
    )
    completed_tip = _next_tip(
        state,
        claimed_tip,
        development_sec_reader_receipts={scope_hash: receipt},
    )
    validate_reveal_store_current_tip_anchor_transition(
        claimed_tip,
        completed_tip,
    )

    replaced_receipt = copy.deepcopy(receipt)
    replaced_receipt["content_manifest_sha256"] = _h(
        "replacement content manifest"
    )
    _rehash(replaced_receipt, "receipt_sha256")
    replacement_tip = _next_tip(
        state,
        completed_tip,
        development_sec_reader_receipts={scope_hash: replaced_receipt},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="removed or changed a persisted development SEC reader receipt",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            completed_tip,
            replacement_tip,
        )

    deleted_tip = _next_tip(
        state,
        completed_tip,
        development_sec_execution_claims={},
        development_sec_reader_receipts={},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="removed or changed a persisted development SEC execution claim",
    ):
        validate_reveal_store_current_tip_anchor_transition(
            completed_tip,
            deleted_tip,
        )


def test_intermediate_model_claim_binds_terminal_sec_carry_candidate_and_limits() -> None:
    state, carry_tip, claim_tip, claim, binding = _stage_model_fixture()
    request_hash = binding["request_sha256"]
    bundle = carry_tip["authorization_bundles"][request_hash]
    replay = build_stage_model_execution_claim(
        bundle,
        independent_current_tip_anchor=carry_tip,
        execution_source_hashes=_bundle_model_source_hashes(bundle),
    )
    assert replay == claim
    assert claim["schema_version"] == STAGE_MODEL_EXECUTION_CLAIM_SCHEMA_VERSION
    assert claim["authorized_stage"] == "intermediate"
    assert claim["carry_in_kind"] == "development_root_carry_in"
    assert claim["model_component_id"] == STAGE_MODEL_BATCH_COMPONENT_ID
    assert claim["model_runtime_limits"]["model_call_count"] == len(
        claim["sec_acquisition_accession_order"]
    )
    assert claim["model_runtime_limits"]["model_call_cap"] == 20
    assert claim["identity_lexicon_sha256"] == CANONICAL_IDENTITY_LEXICON_SHA256
    assert claim["event_count"] == len(claim["event_plan"])
    assert [
        (event["availability_session"], event["accession_number"])
        for event in claim["event_plan"]
    ] == sorted(
        (event["availability_session"], event["accession_number"])
        for event in claim["event_plan"]
    )
    assert [
        event["accession_number"] for event in claim["event_plan"]
    ] != claim["sec_acquisition_accession_order"]
    acquisition_ordinal = {
        accession: ordinal
        for ordinal, accession in enumerate(
            claim["sec_acquisition_accession_order"], start=1
        )
    }
    assert [event["event_ordinal"] for event in claim["event_plan"]] == list(
        range(1, claim["event_count"] + 1)
    )
    assert all(
        event["sec_document_ordinal"]
        == acquisition_ordinal[event["accession_number"]]
        for event in claim["event_plan"]
    )
    assert claim["development_root_scope_sha256"] in carry_tip[
        "development_sec_execution_claims"
    ]
    _assert_model_market_bindings(
        claim,
        carry_tip,
        claim["development_root_scope_sha256"],
    )
    for field in (
        "caller_supplied_path_permitted",
        "filing_text_included",
        "market_access_permitted",
        "outcome_access_permitted",
        "future_stage_access_permitted",
        "paid_api_access_permitted",
        "external_network_access_permitted",
        "effect_may_be_repeated_after_indeterminate_crash",
    ):
        assert claim[field] is False
    assert validate_stage_model_execution_claim(
        claim,
        independent_current_tip_anchor=claim_tip,
    ) == claim["claim_sha256"]

    no_carry_tip = _next_tip(
        state,
        carry_tip,
        development_root_carry_in_reader_receipts={},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="exactly one stage-correct carry",
    ):
        build_stage_model_execution_claim(
            bundle,
            independent_current_tip_anchor=no_carry_tip,
            execution_source_hashes=_bundle_model_source_hashes(bundle),
        )


@pytest.mark.parametrize("stage", ["intermediate", "final"])
@pytest.mark.parametrize("market_terminal", ["missing", "aborted"])
def test_stage_model_requires_successful_same_root_market_terminal(
    stage: str,
    market_terminal: str,
) -> None:
    state, carry_tip, _claim_tip, claim, binding = _stage_model_fixture(stage)
    request_hash = binding["request_sha256"]
    scope_hash = claim["development_root_scope_sha256"]
    market_claim = carry_tip["development_market_execution_claims"][scope_hash]
    market_aborts: dict[str, dict] = {}
    if market_terminal == "aborted":
        market_aborts[scope_hash] = build_development_market_execution_abort(
            market_claim,
            reason="external_effect_failed_or_completion_unknown",
        )
    no_success_tip = _next_tip(
        state,
        carry_tip,
        development_market_reader_receipts={},
        development_market_execution_aborts=market_aborts,
    )
    bundle = no_success_tip["authorization_bundles"][request_hash]
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="exactly one successful, non-aborted",
    ):
        build_stage_model_execution_claim(
            bundle,
            independent_current_tip_anchor=no_success_tip,
            execution_source_hashes=_bundle_model_source_hashes(bundle),
        )


def test_stage_model_reader_and_abort_are_exact_distinct_terminals() -> None:
    state, _carry_tip, claim_tip, claim, binding = _stage_model_fixture()
    request_hash = binding["request_sha256"]
    reader = binding["reader"]
    assert reader["schema_version"] == STAGE_MODEL_READER_RECEIPT_SCHEMA_VERSION
    for field in (
        "development_root_scope_sha256",
        "development_sec_execution_claim_sha256",
        "development_sec_reader_receipt_sha256",
        "corpus_universe_sha256",
        "sec_document_count",
        "sec_acquisition_accession_order_sha256",
        "event_count",
        "event_plan_sha256",
        "identity_lexicon_sha256",
        *_MODEL_MARKET_SHA_FIELDS,
    ):
        assert reader[field] == claim[field]
    reader_tip = _next_tip(
        state,
        claim_tip,
        stage_model_reader_receipts={request_hash: reader},
    )
    validate_reveal_store_current_tip_anchor_transition(claim_tip, reader_tip)
    assert validate_stage_model_reader_receipt(
        reader,
        independent_current_tip_anchor=reader_tip,
        byte_index=binding["byte_index"],
        complete_marker_sha256=binding["complete_marker_sha256"],
    ) == reader["receipt_sha256"]

    crossed_reader = copy.deepcopy(reader)
    crossed_reader["development_market_byte_index_sha256"] = _h(
        "crossed stage model market byte index"
    )
    _rehash(crossed_reader, "receipt_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="crossed its execution claim",
    ):
        _next_tip(
            state,
            claim_tip,
            stage_model_reader_receipts={request_hash: crossed_reader},
        )

    abort = build_stage_model_execution_abort(
        claim,
        reason="external_effect_failed_or_completion_unknown",
    )
    assert abort == build_stage_model_execution_abort(
        claim,
        reason="external_effect_failed_or_completion_unknown",
    )
    assert abort["schema_version"] == STAGE_MODEL_EXECUTION_ABORT_SCHEMA_VERSION
    abort_tip = _next_tip(
        state,
        claim_tip,
        stage_model_execution_aborts={request_hash: abort},
    )
    validate_reveal_store_current_tip_anchor_transition(claim_tip, abort_tip)
    assert validate_stage_model_execution_abort(
        abort,
        independent_current_tip_anchor=abort_tip,
    ) == abort["abort_sha256"]
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="both completed and aborted",
    ):
        _next_tip(
            state,
            claim_tip,
            stage_model_reader_receipts={request_hash: reader},
            stage_model_execution_aborts={request_hash: abort},
        )


def test_stage_model_rejects_source_and_claim_mutation() -> None:
    state, carry_tip, _claim_tip, claim, binding = _stage_model_fixture()
    request_hash = binding["request_sha256"]
    bundle = carry_tip["authorization_bundles"][request_hash]
    changed_sources = _bundle_model_source_hashes(bundle)
    changed_sources[MODEL_EXECUTION_SOURCE_ROLES[0]] = _h("wrong model source")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="differ from the registered candidate",
    ):
        build_stage_model_execution_claim(
            bundle,
            independent_current_tip_anchor=carry_tip,
            execution_source_hashes=changed_sources,
        )

    changed_claim = copy.deepcopy(claim)
    changed_claim["model_runtime_limits"]["retries"] = 1
    _rehash(changed_claim, "claim_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="frozen local-model limits",
    ):
        _next_tip(
            state,
            carry_tip,
            stage_model_execution_claims={request_hash: changed_claim},
        )


def test_stage_model_rejects_rehashed_event_identity_and_root_mutations() -> None:
    _state, carry_tip, claim_tip, claim, binding = _stage_model_fixture()
    request_hash = binding["request_sha256"]

    def reverse_chronology(value: dict) -> None:
        value["event_plan"].reverse()
        for ordinal, event in enumerate(value["event_plan"], start=1):
            event["event_ordinal"] = ordinal

    event_mutators = (
        reverse_chronology,
        lambda value: value["event_plan"][0].update(event_ordinal=99),
        lambda value: value["event_plan"][0].update(
            accession_number="0000320193-24-999999"
        ),
        lambda value: value["event_plan"][0].update(
            form="10-Q" if value["event_plan"][0]["form"] == "10-K" else "10-K"
        ),
        lambda value: value["event_plan"][0].update(
            availability_session="2024-12-31"
        ),
        lambda value: value["event_plan"][0].update(sec_document_ordinal=99),
    )
    mutators = [
        *event_mutators,
        lambda value: value.update(identity_lexicon_sha256=_h("forged lexicon")),
        lambda value: value.update(
            development_root_scope_sha256=_h("forged development root scope")
        ),
        lambda value: value.update(
            development_sec_execution_claim_sha256=_h(
                "forged development SEC root claim"
            )
        ),
        lambda value: value.update(
            development_sec_reader_receipt_sha256=_h(
                "forged development SEC root reader"
            )
        ),
        *[
            (
                lambda value, field=field: value.update(
                    {field: _h(f"forged stage model {field}")}
                )
            )
            for field in _MODEL_MARKET_SHA_FIELDS
        ],
    ]
    for mutate in mutators:
        forged_claim = copy.deepcopy(claim)
        mutate(forged_claim)
        if forged_claim["event_plan"] != claim["event_plan"]:
            forged_claim["event_plan_sha256"] = canonical_sha256(
                forged_claim["event_plan"]
            )
        _rehash(forged_claim, "claim_sha256")
        forged_tip = copy.deepcopy(claim_tip)
        forged_tip["stage_model_execution_claims"] = {
            request_hash: forged_claim
        }
        _rehash(forged_tip, "tip_anchor_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_reveal_store_current_tip_anchor_transition(
                carry_tip,
                forged_tip,
            )


def test_active_stage_model_claim_blocks_output_and_wrong_terminal() -> None:
    state, carry_tip, claim_tip, claim, binding = _stage_model_fixture()
    request_hash = binding["request_sha256"]
    bundle = carry_tip["authorization_bundles"][request_hash]
    output = build_consumed_stage_output_receipt(
        bundle,
        stage_sec_execution_claim=carry_tip["stage_sec_execution_claims"][
            request_hash
        ],
        stage_sec_reader_receipt=carry_tip["stage_sec_reader_receipts"][
            request_hash
        ],
        **_output_binding(bundle["authorization_grant"], salt="model active"),
    )
    output_tip = _next_tip(
        state,
        claim_tip,
        consumed_stage_output_receipts={request_hash: output},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="Active model execution claim blocks every transition",
    ):
        validate_reveal_store_current_tip_anchor_transition(claim_tip, output_tip)

    wrong_reader = copy.deepcopy(binding["reader"])
    wrong_reader["request_sha256"] = _h("wrong request")
    _rehash(wrong_reader, "receipt_sha256")
    with pytest.raises(SecFilingGemmaStageAuthorizationError):
        _next_tip(
            state,
            claim_tip,
            stage_model_reader_receipts={_h("wrong request"): wrong_reader},
        )
    assert claim["start_current_tip_anchor_sha256"] == carry_tip[
        "tip_anchor_sha256"
    ]


def test_final_model_claim_uses_only_final_stage_carry() -> None:
    _state, carry_tip, claim_tip, claim, binding = _stage_model_fixture("final")
    assert claim["authorized_stage"] == "final"
    assert claim["carry_in_kind"] == "stage_carry_in"
    assert claim["model_runtime_limits"]["model_call_cap"] == 12
    assert claim["carry_in_reader_receipt_sha256"] == carry_tip[
        "stage_carry_in_reader_receipts"
    ][binding["request_sha256"]]["receipt_sha256"]
    assert validate_stage_model_execution_claim(
        claim,
        independent_current_tip_anchor=claim_tip,
    ) == claim["claim_sha256"]


def test_development_model_lifecycle_is_request_free_and_has_no_carry() -> None:
    state, root_tip, claim_tip, claim, binding = _development_model_fixture()
    scope_hash = binding["scope_sha256"]
    replay = build_development_model_execution_claim(
        state,
        development_root_scope_sha256=scope_hash,
        independent_current_tip_anchor=root_tip,
        execution_source_hashes=_development_model_source_hashes(state),
    )
    assert replay == claim
    assert claim["schema_version"] == DEVELOPMENT_MODEL_EXECUTION_CLAIM_SCHEMA_VERSION
    assert claim["authorized_stage"] == "development"
    assert claim["carry_in_required"] is False
    assert claim["reveal_request_consumption_permitted"] is False
    assert claim["consumption_ledger_mutation_permitted"] is False
    assert claim["model_runtime_limits"]["model_call_cap"] == 80
    assert claim["identity_lexicon_sha256"] == CANONICAL_IDENTITY_LEXICON_SHA256
    _assert_model_market_bindings(claim, root_tip, scope_hash)
    assert claim["event_count"] == len(claim["event_plan"])
    assert [
        (event["availability_session"], event["accession_number"])
        for event in claim["event_plan"]
    ] == sorted(
        (event["availability_session"], event["accession_number"])
        for event in claim["event_plan"]
    )
    assert [
        event["accession_number"] for event in claim["event_plan"]
    ] != claim["sec_acquisition_accession_order"]
    development_acquisition_ordinal = {
        accession: ordinal
        for ordinal, accession in enumerate(
            claim["sec_acquisition_accession_order"], start=1
        )
    }
    assert all(
        event["sec_document_ordinal"]
        == development_acquisition_ordinal[event["accession_number"]]
        for event in claim["event_plan"]
    )
    assert validate_development_model_execution_claim(
        claim,
        independent_current_tip_anchor=claim_tip,
    ) == claim["claim_sha256"]

    reader = binding["reader"]
    assert reader["schema_version"] == DEVELOPMENT_MODEL_READER_RECEIPT_SCHEMA_VERSION
    for field in (
        "sec_document_count",
        "sec_acquisition_accession_order_sha256",
        "event_count",
        "event_plan_sha256",
        "identity_lexicon_sha256",
        *_MODEL_MARKET_SHA_FIELDS,
    ):
        assert reader[field] == claim[field]
    reader_tip = _next_tip(
        state,
        claim_tip,
        development_model_reader_receipts={scope_hash: reader},
    )
    validate_reveal_store_current_tip_anchor_transition(claim_tip, reader_tip)
    assert validate_development_model_reader_receipt(
        reader,
        independent_current_tip_anchor=reader_tip,
        byte_index=binding["byte_index"],
        complete_marker_sha256=binding["complete_marker_sha256"],
    ) == reader["receipt_sha256"]

    crossed_reader = copy.deepcopy(reader)
    crossed_reader["development_market_acquisition_bundle_sha256"] = _h(
        "crossed development model market bundle"
    )
    _rehash(crossed_reader, "receipt_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="crossed its execution claim",
    ):
        _next_tip(
            state,
            claim_tip,
            development_model_reader_receipts={scope_hash: crossed_reader},
        )


@pytest.mark.parametrize("market_terminal", ["missing", "aborted"])
def test_development_model_requires_successful_same_root_market_terminal(
    market_terminal: str,
) -> None:
    _child_state, _child_tip, _carry, root_binding = (
        _development_root_carry_in_fixture()
    )
    state = root_binding["root_state"]
    scope_hash = root_binding["root_scope_sha256"]
    if market_terminal == "missing":
        no_success_tip = root_binding["root_reader_tip"]
    else:
        market_claim = root_binding["root_market_claim"]
        market_abort = build_development_market_execution_abort(
            market_claim,
            reason="external_effect_failed_or_completion_unknown",
        )
        no_success_tip = _next_tip(
            state,
            root_binding["root_market_claim_tip"],
            development_market_execution_aborts={scope_hash: market_abort},
        )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="exactly one successful, non-aborted",
    ):
        build_development_model_execution_claim(
            state,
            development_root_scope_sha256=scope_hash,
            independent_current_tip_anchor=no_success_tip,
            execution_source_hashes=_development_model_source_hashes(state),
        )


def test_development_model_claim_transition_rebuilds_every_candidate_model_source_binding() -> None:
    state, root_tip, claim_tip, claim, binding = _development_model_fixture()
    scope_hash = binding["scope_sha256"]
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="requires the authenticated prior store snapshot",
    ):
        validate_reveal_store_current_tip_anchor_transition(root_tip, claim_tip)

    def change_source_map(value: dict) -> None:
        value["execution_source_hashes"]["extractor"] = _h(
            "forged development model extractor source"
        )
        value["execution_source_hashes_sha256"] = canonical_sha256(
            value["execution_source_hashes"]
        )

    def change_runtime_limits(value: dict) -> None:
        value["model_runtime_limits"]["read_timeout_seconds"] = 29
        value["model_runtime_limits_sha256"] = canonical_sha256(
            value["model_runtime_limits"]
        )

    def reverse_event_plan(value: dict) -> None:
        value["event_plan"].reverse()
        for ordinal, event in enumerate(value["event_plan"], start=1):
            event["event_ordinal"] = ordinal
        value["event_plan_sha256"] = canonical_sha256(value["event_plan"])

    def change_event_field(value: dict, field: str, replacement: object) -> None:
        value["event_plan"][0][field] = replacement
        value["event_plan_sha256"] = canonical_sha256(value["event_plan"])

    mutators = (
        lambda value: value.update(candidate_sha256=_h("forged candidate")),
        lambda value: value.update(
            candidate_design_sha256=_h("forged candidate design")
        ),
        lambda value: value.update(
            registry_entry_sha256=_h("forged candidate registry entry")
        ),
        lambda value: value.update(
            candidate_source_hashes_sha256=_h("forged candidate sources")
        ),
        lambda value: value.update(model_name="gemma4:99b"),
        lambda value: value.update(model_digest=_h("forged model digest")),
        lambda value: value.update(
            runtime_fingerprint_sha256=_h("forged runtime fingerprint")
        ),
        lambda value: value.update(
            model_transport_sha256=_h("forged model transport")
        ),
        change_source_map,
        lambda value: value.update(
            execution_source_hashes_sha256=_h("forged source closure hash")
        ),
        lambda value: value.update(
            execution_source_role_count=value["execution_source_role_count"] + 1
        ),
        change_runtime_limits,
        reverse_event_plan,
        lambda value: change_event_field(value, "event_ordinal", 99),
        lambda value: change_event_field(
            value,
            "accession_number",
            "0000320193-23-999999",
        ),
        lambda value: change_event_field(
            value,
            "form",
            "10-Q" if value["event_plan"][0]["form"] == "10-K" else "10-K",
        ),
        lambda value: change_event_field(
            value,
            "availability_session",
            "2023-12-29",
        ),
        lambda value: change_event_field(value, "sec_document_ordinal", 99),
        lambda value: value.update(
            identity_lexicon_sha256=_h("forged development identity lexicon")
        ),
        *(
            (
                lambda value, field=field: value.update(
                    {field: _h(f"forged development model {field}")}
                )
            )
            for field in _MODEL_MARKET_SHA_FIELDS
        ),
    )
    for mutate in mutators:
        forged_claim = copy.deepcopy(claim)
        mutate(forged_claim)
        _rehash(forged_claim, "claim_sha256")
        forged_tip = copy.deepcopy(claim_tip)
        forged_tip["development_model_execution_claims"] = {
            scope_hash: forged_claim
        }
        _rehash(forged_tip, "tip_anchor_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_reveal_store_current_tip_anchor_transition(
                root_tip,
                forged_tip,
                authenticated_store_snapshot=state,
            )


def test_development_model_abort_and_chronology_are_fail_closed() -> None:
    state, _root_tip, claim_tip, claim, binding = _development_model_fixture()
    scope_hash = binding["scope_sha256"]
    abort = build_development_model_execution_abort(
        claim,
        reason="claim_recovered_without_terminal_receipt",
    )
    assert abort["schema_version"] == DEVELOPMENT_MODEL_EXECUTION_ABORT_SCHEMA_VERSION
    abort_tip = _next_tip(
        state,
        claim_tip,
        development_model_execution_aborts={scope_hash: abort},
    )
    validate_reveal_store_current_tip_anchor_transition(claim_tip, abort_tip)
    assert validate_development_model_execution_abort(
        abort,
        independent_current_tip_anchor=abort_tip,
    ) == abort["abort_sha256"]

    child_state, child_tip, _carry, child_binding = (
        _development_root_carry_in_fixture()
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="precede reveal-request consumption",
    ):
        build_development_model_execution_claim(
            child_state,
            development_root_scope_sha256=child_binding["root_scope_sha256"],
            independent_current_tip_anchor=child_tip,
            execution_source_hashes=_development_model_source_hashes(
                child_state
            ),
        )


def test_stage_and_development_model_claims_share_one_global_effect_slot() -> None:
    state, carry_tip, _claim_tip, _claim, binding = _stage_model_fixture()
    _root_state, _root_tip, development_claim_tip, development_claim, dev_binding = (
        _development_model_fixture()
    )
    active_cross_tip = _next_tip(
        state,
        carry_tip,
        development_model_execution_claims={
            dev_binding["scope_sha256"]: development_claim
        },
    )
    request_hash = binding["request_sha256"]
    bundle = active_cross_tip["authorization_bundles"][request_hash]
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="Another owned external effect is already active",
    ):
        build_stage_model_execution_claim(
            bundle,
            independent_current_tip_anchor=active_cross_tip,
            execution_source_hashes=_bundle_model_source_hashes(bundle),
        )
    assert development_claim_tip["development_model_execution_claims"][
        dev_binding["scope_sha256"]
    ] == development_claim


def test_development_feature_assembly_plan_is_exact_feature_only_and_deterministic() -> None:
    state, reader_tip, model_claim, plan = (
        _development_feature_assembly_plan_fixture()
    )
    scope_hash = plan["development_root_scope_sha256"]
    state_before = copy.deepcopy(state)
    tip_before = copy.deepcopy(reader_tip)

    assert set(plan) == _DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_KEYS
    assert (
        plan["schema_version"]
        == DEVELOPMENT_FEATURE_ASSEMBLY_PLAN_SCHEMA_VERSION
    )
    assert plan["plan_kind"] == "request_free_development_feature_assembly"
    assert plan["artifact_stage"] == "development"
    assert plan["development_cutoff_session"] == "2018-12-31"
    assert plan["start_consumed_request_count"] == 0
    assert plan["canonical_market_rows_required"] is True
    assert plan["feature_rows_output_permitted"] is True
    for field in (
        "raw_market_output_permitted",
        "normalized_filing_text_output_permitted",
        "model_transport_envelope_output_permitted",
        "outcome_access_permitted",
        "label_access_permitted",
        "training_membership_access_permitted",
        "learner_fit_permitted",
        "prediction_access_permitted",
        "holdout_access_permitted",
        "ledger_mutation_permitted",
        "stage_promotion_permitted",
    ):
        assert plan[field] is False
    assert plan["event_count"] == len(plan["event_plan"])
    assert plan["event_plan"] == model_claim["event_plan"]
    assert plan["event_plan_sha256"] == canonical_sha256(plan["event_plan"])
    assert [
        (event["availability_session"], event["accession_number"])
        for event in plan["event_plan"]
    ] == sorted(
        (event["availability_session"], event["accession_number"])
        for event in plan["event_plan"]
    )
    assert all(
        event["availability_session"] <= plan["development_cutoff_session"]
        for event in plan["event_plan"]
    )

    sec_claim = reader_tip["development_sec_execution_claims"][scope_hash]
    sec_reader = reader_tip["development_sec_reader_receipts"][scope_hash]
    market_claim = reader_tip["development_market_execution_claims"][scope_hash]
    market_reader = reader_tip["development_market_reader_receipts"][scope_hash]
    model_reader = reader_tip["development_model_reader_receipts"][scope_hash]
    assert plan["development_sec_execution_claim_sha256"] == sec_claim["claim_sha256"]
    assert (
        plan["development_sec_reader_receipt_sha256"]
        == sec_reader["receipt_sha256"]
    )
    assert (
        plan["development_market_execution_claim_sha256"]
        == market_claim["claim_sha256"]
    )
    assert (
        plan["development_market_reader_receipt_sha256"]
        == market_reader["receipt_sha256"]
    )
    assert (
        plan["development_model_execution_claim_sha256"]
        == model_claim["claim_sha256"]
    )
    assert (
        plan["development_model_reader_receipt_sha256"]
        == model_reader["receipt_sha256"]
    )
    assert validate_development_feature_assembly_plan(
        plan,
        expected_feature_assembly_plan_sha256=plan[
            "feature_assembly_plan_sha256"
        ],
    ) == plan["feature_assembly_plan_sha256"]
    assert build_development_feature_assembly_plan(
        state,
        development_root_scope_sha256=scope_hash,
        independent_current_tip_anchor=reader_tip,
    ) == plan
    assert state == state_before
    assert reader_tip == tip_before


def test_development_feature_assembly_plan_requires_terminal_non_aborted_model() -> None:
    state, _root_tip, claim_tip, claim, binding = _development_model_fixture()
    scope_hash = binding["scope_sha256"]
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="terminal non-aborted SEC, market, and model ancestry",
    ):
        build_development_feature_assembly_plan(
            state,
            development_root_scope_sha256=scope_hash,
            independent_current_tip_anchor=claim_tip,
        )

    abort = build_development_model_execution_abort(
        claim,
        reason="claim_recovered_without_terminal_receipt",
    )
    abort_tip = _next_tip(
        state,
        claim_tip,
        development_model_execution_aborts={scope_hash: abort},
    )
    validate_reveal_store_current_tip_anchor_transition(claim_tip, abort_tip)
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="terminal non-aborted SEC, market, and model ancestry",
    ):
        build_development_feature_assembly_plan(
            state,
            development_root_scope_sha256=scope_hash,
            independent_current_tip_anchor=abort_tip,
        )


def test_development_feature_assembly_plan_rejects_consumption_and_active_effects() -> None:
    root_state, _root_tip, model_claim_tip, model_claim, model_binding = (
        _development_model_fixture()
    )
    scope_hash = model_binding["scope_sha256"]
    model_reader_tip = _next_tip(
        root_state,
        model_claim_tip,
        development_model_reader_receipts={scope_hash: model_binding["reader"]},
    )
    child_state, child_reader_tip, _carry, child_binding = (
        _development_root_carry_in_fixture()
    )
    consumed_tip = _next_tip(
        child_state,
        child_reader_tip,
        development_model_execution_claims={scope_hash: model_claim},
        development_model_reader_receipts={scope_hash: model_binding["reader"]},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="precede every reveal-request consumption",
    ):
        build_development_feature_assembly_plan(
            child_state,
            development_root_scope_sha256=scope_hash,
            independent_current_tip_anchor=consumed_tip,
        )

    active_tip = _next_tip(
        child_state,
        child_binding["child_claim_tip"],
        development_model_execution_claims={scope_hash: model_claim},
        development_model_reader_receipts={scope_hash: model_binding["reader"]},
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="requires no active owned external effect",
    ):
        build_development_feature_assembly_plan(
            child_state,
            development_root_scope_sha256=scope_hash,
            independent_current_tip_anchor=active_tip,
        )
    assert model_reader_tip["consumed_request_count"] == 0


def test_development_feature_assembly_plan_validation_fails_closed() -> None:
    _state, _reader_tip, _model_claim, plan = (
        _development_feature_assembly_plan_fixture()
    )

    for field, replacement in (
        ("outcome_access_permitted", True),
        ("label_access_permitted", True),
        ("learner_fit_permitted", True),
        ("feature_rows_output_permitted", False),
        ("development_cutoff_session", "2019-01-02"),
        ("start_consumed_request_count", 1),
        ("start_consumed_request_count", False),
    ):
        changed = copy.deepcopy(plan)
        changed[field] = replacement
        _rehash(changed, "feature_assembly_plan_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_development_feature_assembly_plan(
                changed,
                expected_feature_assembly_plan_sha256=changed[
                    "feature_assembly_plan_sha256"
                ],
            )

    reordered = copy.deepcopy(plan)
    reordered["event_plan"].reverse()
    for ordinal, event in enumerate(reordered["event_plan"], start=1):
        event["event_ordinal"] = ordinal
    reordered["event_plan_sha256"] = canonical_sha256(reordered["event_plan"])
    _rehash(reordered, "feature_assembly_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="not in exact chronological order",
    ):
        validate_development_feature_assembly_plan(
            reordered,
            expected_feature_assembly_plan_sha256=reordered[
                "feature_assembly_plan_sha256"
            ],
        )

    with_extra = copy.deepcopy(plan)
    with_extra["unexpected"] = False
    _rehash(with_extra, "feature_assembly_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="keys changed",
    ):
        validate_development_feature_assembly_plan(
            with_extra,
            expected_feature_assembly_plan_sha256=with_extra[
                "feature_assembly_plan_sha256"
            ],
        )

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="not externally pinned",
    ):
        validate_development_feature_assembly_plan(
            plan,
            expected_feature_assembly_plan_sha256=_h("other feature plan"),
        )


def test_development_label_assembly_plan_is_exact_narrow_and_deterministic() -> None:
    state, reader_tip, feature_plan, plan = (
        _development_label_assembly_plan_fixture()
    )
    state_before = copy.deepcopy(state)
    tip_before = copy.deepcopy(reader_tip)

    assert set(plan) == _DEVELOPMENT_LABEL_ASSEMBLY_PLAN_KEYS
    assert plan["schema_version"] == DEVELOPMENT_LABEL_ASSEMBLY_PLAN_SCHEMA_VERSION
    assert plan["plan_kind"] == "request_free_development_label_assembly"
    assert plan["artifact_stage"] == "development"
    assert plan["development_root_scope_sha256"] == feature_plan[
        "development_root_scope_sha256"
    ]
    assert plan["start_consumed_request_count"] == 0
    assert plan["source_feature_assembly_plan"] == feature_plan
    assert plan["source_feature_assembly_plan_sha256"] == feature_plan[
        "feature_assembly_plan_sha256"
    ]
    assert plan["calendar_sessions_sha256"] == market_session_calendar_sha256(
        EXPECTED_MARKET_HISTORY_SESSIONS
    )
    assert plan["development_cutoff_session"] == "2018-12-31"
    assert plan["label_horizon_sessions"] == 20
    assert plan["label_entry_session_offset"] == 1
    assert plan["label_maturity_session_offset"] == 21
    assert (
        plan["maturity_rule"]
        == "t_plus_21_session_lte_development_cutoff_inclusive"
    )
    assert plan["canonical_market_rows_required"] is True
    assert plan["development_outcome_derivation_permitted"] is True
    assert plan["development_label_rows_output_permitted"] is True
    for field in (
        "post_cutoff_market_access_permitted",
        "raw_market_output_permitted",
        "normalized_filing_text_output_permitted",
        "model_transport_envelope_output_permitted",
        "training_membership_access_permitted",
        "learner_fit_permitted",
        "prediction_access_permitted",
        "holdout_access_permitted",
        "ledger_mutation_permitted",
        "stage_promotion_permitted",
        "production_permitted",
    ):
        assert plan[field] is False
    assert feature_plan["outcome_access_permitted"] is False
    assert feature_plan["label_access_permitted"] is False
    assert plan["event_count"] == len(feature_plan["event_plan"])
    assert plan["event_count"] == len(plan["maturity_plan"])
    assert plan["maturity_plan_sha256"] == canonical_sha256(
        plan["maturity_plan"]
    )
    assert plan["matured_event_count"] + plan["unmatured_event_count"] == plan[
        "event_count"
    ]
    assert validate_development_label_assembly_plan(
        plan,
        expected_label_assembly_plan_sha256=plan[
            "label_assembly_plan_sha256"
        ],
    ) == plan["label_assembly_plan_sha256"]
    assert build_development_label_assembly_plan(
        state,
        development_root_scope_sha256=feature_plan[
            "development_root_scope_sha256"
        ],
        source_feature_assembly_plan=feature_plan,
        independent_current_tip_anchor=reader_tip,
    ) == plan
    assert state == state_before
    assert reader_tip == tip_before


def test_development_label_maturity_boundary_is_cutoff_inclusive() -> None:
    state, reader_tip, feature_plan, _plan = (
        _development_label_assembly_plan_fixture()
    )
    boundary_feature_plan = copy.deepcopy(feature_plan)
    cutoff_index = EXPECTED_MARKET_HISTORY_SESSIONS.index("2018-12-31")
    boundary_decision = EXPECTED_MARKET_HISTORY_SESSIONS[cutoff_index - 21]
    post_cutoff_decision = EXPECTED_MARKET_HISTORY_SESSIONS[cutoff_index - 20]
    boundary_feature_plan["event_plan"][-2][
        "availability_session"
    ] = boundary_decision
    boundary_feature_plan["event_plan"][-1][
        "availability_session"
    ] = post_cutoff_decision
    boundary_feature_plan["event_plan_sha256"] = canonical_sha256(
        boundary_feature_plan["event_plan"]
    )
    _rehash(boundary_feature_plan, "feature_assembly_plan_sha256")

    with patch(
        "agent_benchmark.sec_filing_gemma_stage_authorization."
        "build_development_feature_assembly_plan",
        return_value=boundary_feature_plan,
    ):
        plan = build_development_label_assembly_plan(
            state,
            development_root_scope_sha256=boundary_feature_plan[
                "development_root_scope_sha256"
            ],
            source_feature_assembly_plan=boundary_feature_plan,
            independent_current_tip_anchor=reader_tip,
        )

    boundary, post_cutoff = plan["maturity_plan"][-2:]
    assert boundary["decision_session"] == boundary_decision
    assert boundary["label_maturity_session"] == "2018-12-31"
    assert boundary["matured_by_development_cutoff"] is True
    assert post_cutoff["decision_session"] == post_cutoff_decision
    assert post_cutoff["label_maturity_session"] > "2018-12-31"
    assert post_cutoff["matured_by_development_cutoff"] is False
    assert plan["matured_event_count"] == plan["event_count"] - 1
    assert plan["unmatured_event_count"] == 1
    assert validate_development_label_assembly_plan(
        plan,
        expected_label_assembly_plan_sha256=plan[
            "label_assembly_plan_sha256"
        ],
    ) == plan["label_assembly_plan_sha256"]


def test_development_label_plan_rejects_a_different_supplied_feature_plan() -> None:
    state, reader_tip, feature_plan, _plan = (
        _development_label_assembly_plan_fixture()
    )
    changed = copy.deepcopy(feature_plan)
    changed["candidate_sha256"] = _h("different candidate")
    _rehash(changed, "feature_assembly_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="differs from the terminal store replay",
    ):
        build_development_label_assembly_plan(
            state,
            development_root_scope_sha256=feature_plan[
                "development_root_scope_sha256"
            ],
            source_feature_assembly_plan=changed,
            independent_current_tip_anchor=reader_tip,
        )


def test_development_label_assembly_plan_validation_fails_closed() -> None:
    _state, _reader_tip, _feature_plan, plan = (
        _development_label_assembly_plan_fixture()
    )

    for field, replacement in (
        ("canonical_market_rows_required", False),
        ("development_outcome_derivation_permitted", False),
        ("development_label_rows_output_permitted", False),
        ("post_cutoff_market_access_permitted", True),
        ("training_membership_access_permitted", True),
        ("learner_fit_permitted", True),
        ("prediction_access_permitted", True),
        ("holdout_access_permitted", True),
        ("ledger_mutation_permitted", True),
        ("stage_promotion_permitted", True),
        ("production_permitted", True),
        ("calendar_sessions_sha256", _h("changed market calendar")),
        ("label_horizon_sessions", False),
        ("label_entry_session_offset", False),
        ("label_maturity_session_offset", False),
        ("event_count", False),
        ("matured_event_count", False),
        ("unmatured_event_count", False),
        ("start_consumed_request_count", False),
    ):
        changed = copy.deepcopy(plan)
        changed[field] = replacement
        _rehash(changed, "label_assembly_plan_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_development_label_assembly_plan(
                changed,
                expected_label_assembly_plan_sha256=changed[
                    "label_assembly_plan_sha256"
                ],
            )

    changed_maturity = copy.deepcopy(plan)
    changed_maturity["maturity_plan"][-1][
        "label_maturity_session"
    ] = "2018-12-31"
    changed_maturity["maturity_plan_sha256"] = canonical_sha256(
        changed_maturity["maturity_plan"]
    )
    _rehash(changed_maturity, "label_assembly_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="frozen calendar derivation",
    ):
        validate_development_label_assembly_plan(
            changed_maturity,
            expected_label_assembly_plan_sha256=changed_maturity[
                "label_assembly_plan_sha256"
            ],
        )

    changed_boundary = copy.deepcopy(plan)
    changed_boundary["maturity_plan"][-1][
        "matured_by_development_cutoff"
    ] = not changed_boundary["maturity_plan"][-1][
        "matured_by_development_cutoff"
    ]
    changed_boundary["maturity_plan_sha256"] = canonical_sha256(
        changed_boundary["maturity_plan"]
    )
    _rehash(changed_boundary, "label_assembly_plan_sha256")
    with pytest.raises(SecFilingGemmaStageAuthorizationError):
        validate_development_label_assembly_plan(
            changed_boundary,
            expected_label_assembly_plan_sha256=changed_boundary[
                "label_assembly_plan_sha256"
            ],
        )

    reordered = copy.deepcopy(plan)
    reordered["maturity_plan"].reverse()
    reordered["maturity_plan_sha256"] = canonical_sha256(
        reordered["maturity_plan"]
    )
    _rehash(reordered, "label_assembly_plan_sha256")
    with pytest.raises(SecFilingGemmaStageAuthorizationError):
        validate_development_label_assembly_plan(
            reordered,
            expected_label_assembly_plan_sha256=reordered[
                "label_assembly_plan_sha256"
            ],
        )

    extra_item_key = copy.deepcopy(plan)
    extra_item_key["maturity_plan"][0]["unexpected"] = False
    extra_item_key["maturity_plan_sha256"] = canonical_sha256(
        extra_item_key["maturity_plan"]
    )
    _rehash(extra_item_key, "label_assembly_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="keys changed",
    ):
        validate_development_label_assembly_plan(
            extra_item_key,
            expected_label_assembly_plan_sha256=extra_item_key[
                "label_assembly_plan_sha256"
            ],
        )

    extra_plan_key = copy.deepcopy(plan)
    extra_plan_key["unexpected"] = False
    _rehash(extra_plan_key, "label_assembly_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="keys changed",
    ):
        validate_development_label_assembly_plan(
            extra_plan_key,
            expected_label_assembly_plan_sha256=extra_plan_key[
                "label_assembly_plan_sha256"
            ],
        )

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="not externally pinned",
    ):
        validate_development_label_assembly_plan(
            plan,
            expected_label_assembly_plan_sha256=_h("other label plan"),
        )


def test_development_training_membership_plan_is_exact_narrow_and_deterministic() -> None:
    state, reader_tip, label_plan, plan = (
        _development_training_membership_assembly_plan_fixture()
    )
    state_before = copy.deepcopy(state)
    tip_before = copy.deepcopy(reader_tip)
    feature_plan = label_plan["source_feature_assembly_plan"]

    assert set(plan) == _DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_KEYS
    assert (
        plan["schema_version"]
        == DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_SCHEMA_VERSION
    )
    assert (
        plan["plan_kind"]
        == "request_free_development_training_membership_assembly"
    )
    assert plan["artifact_stage"] == "development"
    assert plan["development_root_scope_sha256"] == label_plan[
        "development_root_scope_sha256"
    ]
    assert plan["source_label_assembly_plan"] == label_plan
    assert plan["source_label_assembly_plan_sha256"] == label_plan[
        "label_assembly_plan_sha256"
    ]
    assert plan["source_feature_assembly_plan_sha256"] == feature_plan[
        "feature_assembly_plan_sha256"
    ]
    assert plan["candidate_sha256"] == feature_plan["candidate_sha256"]
    assert plan["corpus_universe_sha256"] == feature_plan[
        "corpus_universe_sha256"
    ]
    assert plan["event_count"] == label_plan["event_count"]
    assert plan["matured_event_count"] == label_plan["matured_event_count"]
    assert plan["unmatured_event_count"] == label_plan["unmatured_event_count"]

    expected_views = [
        {
            "view_ordinal": 1,
            "training_view_id": "fold_1",
            "view_kind": "development_out_of_fold",
            "training_source_stage": "development",
            "prediction_stage": "development",
            "train_label_maturity_through": "2004-12-31",
            "prediction_window_first_date": "2005-01-03",
            "prediction_window_last_date": "2007-12-31",
            "state_updates_inside_prediction_window": False,
        },
        {
            "view_ordinal": 2,
            "training_view_id": "fold_2",
            "view_kind": "development_out_of_fold",
            "training_source_stage": "development",
            "prediction_stage": "development",
            "train_label_maturity_through": "2007-12-31",
            "prediction_window_first_date": "2008-01-02",
            "prediction_window_last_date": "2010-12-31",
            "state_updates_inside_prediction_window": False,
        },
        {
            "view_ordinal": 3,
            "training_view_id": "fold_3",
            "view_kind": "development_out_of_fold",
            "training_source_stage": "development",
            "prediction_stage": "development",
            "train_label_maturity_through": "2010-12-31",
            "prediction_window_first_date": "2011-01-03",
            "prediction_window_last_date": "2013-12-31",
            "state_updates_inside_prediction_window": False,
        },
        {
            "view_ordinal": 4,
            "training_view_id": "fold_4",
            "view_kind": "development_out_of_fold",
            "training_source_stage": "development",
            "prediction_stage": "development",
            "train_label_maturity_through": "2013-12-31",
            "prediction_window_first_date": "2014-01-02",
            "prediction_window_last_date": "2016-12-30",
            "state_updates_inside_prediction_window": False,
        },
        {
            "view_ordinal": 5,
            "training_view_id": "fold_5",
            "view_kind": "development_out_of_fold",
            "training_source_stage": "development",
            "prediction_stage": "development",
            "train_label_maturity_through": "2016-12-30",
            "prediction_window_first_date": "2017-01-03",
            "prediction_window_last_date": "2018-12-31",
            "state_updates_inside_prediction_window": False,
        },
        {
            "view_ordinal": 6,
            "training_view_id": "intermediate_frozen_through_2018",
            "view_kind": "intermediate_frozen_refit",
            "training_source_stage": "development",
            "prediction_stage": "intermediate",
            "train_label_maturity_through": "2018-12-31",
            "prediction_window_first_date": "2019-01-01",
            "prediction_window_last_date": "2023-12-31",
            "state_updates_inside_prediction_window": False,
        },
    ]
    assert plan["membership_view_count"] == 6
    assert plan["membership_view_specs"] == expected_views
    assert plan["membership_view_specs_sha256"] == canonical_sha256(
        expected_views
    )
    expected_variants = [
        {
            "variant_ordinal": 1,
            "variant_id": "semantic",
            "feature_names_field": "semantic_feature_names",
            "feature_values_field": "semantic_feature_values_hex",
        },
        {
            "variant_ordinal": 2,
            "variant_id": "ablation",
            "feature_names_field": "ablation_feature_names",
            "feature_values_field": "ablation_feature_values_hex",
        },
    ]
    assert plan["model_variant_count"] == 2
    assert plan["model_variant_specs"] == expected_variants
    assert plan["model_variant_specs_sha256"] == canonical_sha256(
        expected_variants
    )
    assert plan["target_cost_bps"] == 10
    assert plan["binary_target_field"] == "cash_beats_long_10bps"
    assert plan["binary_target_encoding"] == "false_to_0_true_to_1"
    assert plan["edge_target_field"] == "cash_active_log_edge_10bps_hex"
    assert plan["minimum_training_row_count"] == 2
    assert plan["both_binary_classes_required"] is True
    assert plan["semantic_ablation_feature_matrices_must_differ"] is True

    for field in (
        "canonical_source_feature_rows_required",
        "canonical_source_label_rows_required",
        "source_feature_rows_access_permitted",
        "source_label_rows_access_permitted",
        "development_outcome_values_access_permitted",
        "training_membership_access_permitted",
        "training_membership_rows_output_permitted",
        "training_feature_matrices_output_permitted",
        "training_target_vectors_output_permitted",
    ):
        assert plan[field] is True
    for field in (
        "development_outcome_derivation_permitted",
        "outcome_based_membership_filtering_permitted",
        "row_rebalancing_permitted",
        "post_cutoff_market_access_permitted",
        "raw_market_output_permitted",
        "compact_adjusted_open_paths_output_permitted",
        "normalized_filing_text_output_permitted",
        "model_transport_envelope_output_permitted",
        "learner_fit_permitted",
        "prediction_access_permitted",
        "holdout_access_permitted",
        "ledger_mutation_permitted",
        "stage_promotion_permitted",
        "production_permitted",
    ):
        assert plan[field] is False
    assert validate_development_training_membership_assembly_plan(
        plan,
        expected_training_membership_assembly_plan_sha256=plan[
            "training_membership_assembly_plan_sha256"
        ],
    ) == plan["training_membership_assembly_plan_sha256"]
    assert build_development_training_membership_assembly_plan(
        state,
        development_root_scope_sha256=plan["development_root_scope_sha256"],
        source_label_assembly_plan=label_plan,
        independent_current_tip_anchor=reader_tip,
    ) == plan
    assert state == state_before
    assert reader_tip == tip_before


def test_development_training_membership_plan_requires_exact_terminal_sources() -> None:
    state, reader_tip, label_plan, _plan = (
        _development_training_membership_assembly_plan_fixture()
    )
    changed = copy.deepcopy(label_plan)
    changed_feature = changed["source_feature_assembly_plan"]
    changed_feature["candidate_sha256"] = _h("different membership candidate")
    _rehash(changed_feature, "feature_assembly_plan_sha256")
    changed["source_feature_assembly_plan_sha256"] = changed_feature[
        "feature_assembly_plan_sha256"
    ]
    _rehash(changed, "label_assembly_plan_sha256")

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="terminal store replay",
    ):
        build_development_training_membership_assembly_plan(
            state,
            development_root_scope_sha256=label_plan[
                "development_root_scope_sha256"
            ],
            source_label_assembly_plan=changed,
            independent_current_tip_anchor=reader_tip,
        )


def test_development_training_membership_plan_rejects_view_and_variant_tampering() -> None:
    _state, _reader_tip, _label_plan, plan = (
        _development_training_membership_assembly_plan_fixture()
    )
    changed_view = copy.deepcopy(plan)
    changed_view["membership_view_specs"][0][
        "train_label_maturity_through"
    ] = "2005-01-03"
    changed_view["membership_view_specs_sha256"] = canonical_sha256(
        changed_view["membership_view_specs"]
    )
    _rehash(changed_view, "training_membership_assembly_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="frozen contract",
    ):
        validate_development_training_membership_assembly_plan(
            changed_view,
            expected_training_membership_assembly_plan_sha256=changed_view[
                "training_membership_assembly_plan_sha256"
            ],
        )

    reordered = copy.deepcopy(plan)
    reordered["membership_view_specs"].reverse()
    reordered["membership_view_specs_sha256"] = canonical_sha256(
        reordered["membership_view_specs"]
    )
    _rehash(reordered, "training_membership_assembly_plan_sha256")
    with pytest.raises(SecFilingGemmaStageAuthorizationError):
        validate_development_training_membership_assembly_plan(
            reordered,
            expected_training_membership_assembly_plan_sha256=reordered[
                "training_membership_assembly_plan_sha256"
            ],
        )

    changed_variant = copy.deepcopy(plan)
    changed_variant["model_variant_specs"][0][
        "feature_values_field"
    ] = "ablation_feature_values_hex"
    changed_variant["model_variant_specs_sha256"] = canonical_sha256(
        changed_variant["model_variant_specs"]
    )
    _rehash(changed_variant, "training_membership_assembly_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="variant differs",
    ):
        validate_development_training_membership_assembly_plan(
            changed_variant,
            expected_training_membership_assembly_plan_sha256=changed_variant[
                "training_membership_assembly_plan_sha256"
            ],
        )

    for nested_field, replacement in (
        (("membership_view_specs", 0, "view_ordinal"), True),
        (
            (
                "membership_view_specs",
                0,
                "state_updates_inside_prediction_window",
            ),
            0,
        ),
        (("model_variant_specs", 0, "variant_ordinal"), True),
    ):
        changed_type = copy.deepcopy(plan)
        collection, index, field = nested_field
        changed_type[collection][index][field] = replacement
        changed_type[f"{collection}_sha256"] = canonical_sha256(
            changed_type[collection]
        )
        _rehash(
            changed_type,
            "training_membership_assembly_plan_sha256",
        )
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_development_training_membership_assembly_plan(
                changed_type,
                expected_training_membership_assembly_plan_sha256=(
                    changed_type[
                        "training_membership_assembly_plan_sha256"
                    ]
                ),
            )


def test_development_training_membership_plan_validation_fails_closed() -> None:
    _state, _reader_tip, _label_plan, plan = (
        _development_training_membership_assembly_plan_fixture()
    )
    allowed_capabilities = (
        "canonical_source_feature_rows_required",
        "canonical_source_label_rows_required",
        "source_feature_rows_access_permitted",
        "source_label_rows_access_permitted",
        "development_outcome_values_access_permitted",
        "training_membership_access_permitted",
        "training_membership_rows_output_permitted",
        "training_feature_matrices_output_permitted",
        "training_target_vectors_output_permitted",
    )
    denied_capabilities = (
        "development_outcome_derivation_permitted",
        "outcome_based_membership_filtering_permitted",
        "row_rebalancing_permitted",
        "post_cutoff_market_access_permitted",
        "raw_market_output_permitted",
        "compact_adjusted_open_paths_output_permitted",
        "normalized_filing_text_output_permitted",
        "model_transport_envelope_output_permitted",
        "learner_fit_permitted",
        "prediction_access_permitted",
        "holdout_access_permitted",
        "ledger_mutation_permitted",
        "stage_promotion_permitted",
        "production_permitted",
    )
    for field, replacement in (
        *((field, False) for field in allowed_capabilities),
        *((field, True) for field in denied_capabilities),
        ("both_binary_classes_required", False),
        ("semantic_ablation_feature_matrices_must_differ", False),
    ):
        changed = copy.deepcopy(plan)
        changed[field] = replacement
        _rehash(changed, "training_membership_assembly_plan_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_development_training_membership_assembly_plan(
                changed,
                expected_training_membership_assembly_plan_sha256=changed[
                    "training_membership_assembly_plan_sha256"
                ],
            )

    for field in (
        "label_horizon_sessions",
        "label_entry_session_offset",
        "label_maturity_session_offset",
        "event_count",
        "matured_event_count",
        "unmatured_event_count",
        "membership_view_count",
        "model_variant_count",
        "target_cost_bps",
        "minimum_training_row_count",
    ):
        changed = copy.deepcopy(plan)
        changed[field] = True
        _rehash(changed, "training_membership_assembly_plan_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_development_training_membership_assembly_plan(
                changed,
                expected_training_membership_assembly_plan_sha256=changed[
                    "training_membership_assembly_plan_sha256"
                ],
            )

    for field, replacement in (
        ("membership_maturity_rule", "outcomes_choose_membership"),
        ("feature_eligibility_rule", "semantic_only"),
        ("membership_order_rule", "sort_by_outcome"),
        ("shared_variant_support_rule", "variant_specific_support"),
        ("binary_target_field", "cash_beats_long_5bps"),
        ("binary_target_encoding", "true_to_0_false_to_1"),
        ("edge_target_field", "cash_active_log_edge_5bps_hex"),
    ):
        changed = copy.deepcopy(plan)
        changed[field] = replacement
        _rehash(changed, "training_membership_assembly_plan_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_development_training_membership_assembly_plan(
                changed,
                expected_training_membership_assembly_plan_sha256=changed[
                    "training_membership_assembly_plan_sha256"
                ],
            )

    extra = copy.deepcopy(plan)
    extra["unexpected"] = False
    _rehash(extra, "training_membership_assembly_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="keys changed",
    ):
        validate_development_training_membership_assembly_plan(
            extra,
            expected_training_membership_assembly_plan_sha256=extra[
                "training_membership_assembly_plan_sha256"
            ],
        )

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="not externally pinned",
    ):
        validate_development_training_membership_assembly_plan(
            plan,
            expected_training_membership_assembly_plan_sha256=_h(
                "other membership plan"
            ),
        )

    inconsistent_self_hash = copy.deepcopy(plan)
    inconsistent_self_hash[
        "training_membership_assembly_plan_sha256"
    ] = _h("internally inconsistent membership plan")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="self-hash is inconsistent",
    ):
        validate_development_training_membership_assembly_plan(
            inconsistent_self_hash,
            expected_training_membership_assembly_plan_sha256=(
                inconsistent_self_hash[
                    "training_membership_assembly_plan_sha256"
                ]
            ),
        )

    class DictSubclass(dict):
        pass

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="exact built-in dict",
    ):
        validate_development_training_membership_assembly_plan(
            DictSubclass(plan),
            expected_training_membership_assembly_plan_sha256=plan[
                "training_membership_assembly_plan_sha256"
            ],
        )


def test_development_oof_learner_fit_plan_is_exact_narrow_and_deterministic() -> None:
    state, reader_tip, membership_plan, fit_specs, plan = (
        _development_oof_learner_fit_plan_fixture()
    )
    state_before = copy.deepcopy(state)
    tip_before = copy.deepcopy(reader_tip)

    assert set(plan) == _DEVELOPMENT_OOF_LEARNER_FIT_PLAN_KEYS
    assert (
        plan["schema_version"]
        == DEVELOPMENT_OOF_LEARNER_FIT_PLAN_SCHEMA_VERSION
    )
    assert plan["contract_version"] == CONTRACT_VERSION
    assert plan["contract_sha256"] == canonical_sha256(
        build_contract_manifest()
    )
    assert plan["plan_kind"] == "request_free_development_oof_learner_fit"
    assert plan["artifact_stage"] == "development"
    assert plan["source_training_membership_assembly_plan"] == membership_plan
    assert plan["source_training_membership_assembly_plan_sha256"] == (
        membership_plan["training_membership_assembly_plan_sha256"]
    )
    expected_source_ids = [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "fold_5",
        "intermediate_frozen_through_2018",
    ]
    assert plan["source_training_view_count"] == 6
    assert plan["source_training_view_ids"] == expected_source_ids
    assert plan["source_training_view_specs_sha256"] == membership_plan[
        "membership_view_specs_sha256"
    ]
    assert plan["authorized_training_view_count"] == 5
    assert plan["authorized_training_view_ids"] == expected_source_ids[:5]
    expected_deferred = [
        {
            "source_training_view_ordinal": 6,
            "training_view_id": "intermediate_frozen_through_2018",
            "reason": (
                "requires_passed_development_ranking_receipt_and_frozen_"
                "candidate_selection"
            ),
        }
    ]
    assert plan["deferred_training_view_count"] == 1
    assert plan["deferred_training_views"] == expected_deferred
    assert plan["deferred_training_views_sha256"] == canonical_sha256(
        expected_deferred
    )
    assert plan["model_variant_count"] == 2
    assert plan["model_variant_ids"] == ["semantic", "ablation"]
    assert plan["learner_fit_input_count"] == 10
    assert plan["learner_fit_input_specs"] == fit_specs
    assert plan["learner_fit_input_specs_sha256"] == canonical_sha256(
        fit_specs
    )
    assert plan["learner_state_output_count"] == 10
    assert plan["learner_model_type"] == LEARNER_MODEL_TYPE
    assert plan["learner_state_schema_version"] == LEARNER_STATE_SCHEMA_VERSION
    assert plan["learner_config"] == asdict(SecFilingGemmaLearnerConfig())
    assert plan["learner_config_sha256"] == canonical_sha256(
        plan["learner_config"]
    )
    assert plan["fit_order_rule"] == (
        "view_ordinal_ascending_then_semantic_then_ablation_exactly_once"
    )
    assert plan["maximum_fit_seconds"] == 60
    for field in _DEVELOPMENT_OOF_ALLOWED_CAPABILITIES:
        assert plan[field] is True
    for field in _DEVELOPMENT_OOF_DENIED_CAPABILITIES:
        assert plan[field] is False
    for ordinal, spec in enumerate(plan["learner_fit_input_specs"], start=1):
        assert set(spec) == _DEVELOPMENT_OOF_FIT_INPUT_SPEC_KEYS
        assert spec["fit_ordinal"] == ordinal
        assert spec["head_variant"] == (
            "semantic" if ordinal % 2 else "ablation"
        )

    assert validate_development_oof_learner_fit_plan(
        plan,
        expected_development_oof_learner_fit_plan_sha256=plan[
            "development_oof_learner_fit_plan_sha256"
        ],
    ) == plan["development_oof_learner_fit_plan_sha256"]
    assert build_development_oof_learner_fit_plan(
        state,
        development_root_scope_sha256=membership_plan[
            "development_root_scope_sha256"
        ],
        source_training_membership_assembly_plan=membership_plan,
        source_training_membership_projection_sha256=plan[
            "source_training_membership_projection_sha256"
        ],
        source_training_membership_batch_sha256=plan[
            "source_training_membership_batch_sha256"
        ],
        fit_input_specs=fit_specs,
        independent_current_tip_anchor=reader_tip,
    ) == plan
    assert state == state_before
    assert reader_tip == tip_before


def test_development_oof_learner_fit_plan_independently_rebuilds_membership_source() -> None:
    state, reader_tip, membership_plan, fit_specs, _plan = (
        _development_oof_learner_fit_plan_fixture()
    )
    changed = copy.deepcopy(membership_plan)
    changed_label = changed["source_label_assembly_plan"]
    changed_feature = changed_label["source_feature_assembly_plan"]
    changed_feature["candidate_sha256"] = _h("different OOF candidate")
    _rehash(changed_feature, "feature_assembly_plan_sha256")
    changed_label["source_feature_assembly_plan_sha256"] = changed_feature[
        "feature_assembly_plan_sha256"
    ]
    _rehash(changed_label, "label_assembly_plan_sha256")
    changed["source_label_assembly_plan_sha256"] = changed_label[
        "label_assembly_plan_sha256"
    ]
    changed["source_feature_assembly_plan_sha256"] = changed_feature[
        "feature_assembly_plan_sha256"
    ]
    changed["candidate_sha256"] = changed_feature["candidate_sha256"]
    _rehash(changed, "training_membership_assembly_plan_sha256")

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="terminal store replay",
    ):
        build_development_oof_learner_fit_plan(
            state,
            development_root_scope_sha256=membership_plan[
                "development_root_scope_sha256"
            ],
            source_training_membership_assembly_plan=changed,
            source_training_membership_projection_sha256=_h(
                "changed OOF projection"
            ),
            source_training_membership_batch_sha256=_h("changed OOF batch"),
            fit_input_specs=fit_specs,
            independent_current_tip_anchor=reader_tip,
        )


def test_development_oof_learner_fit_plan_rejects_order_view6_and_support_tampering() -> None:
    _state, _reader_tip, _membership_plan, _fit_specs, plan = (
        _development_oof_learner_fit_plan_fixture()
    )

    reordered = copy.deepcopy(plan)
    reordered["learner_fit_input_specs"].reverse()
    reordered["learner_fit_input_specs_sha256"] = canonical_sha256(
        reordered["learner_fit_input_specs"]
    )
    _rehash(reordered, "development_oof_learner_fit_plan_sha256")
    with pytest.raises(SecFilingGemmaStageAuthorizationError):
        validate_development_oof_learner_fit_plan(
            reordered,
            expected_development_oof_learner_fit_plan_sha256=reordered[
                "development_oof_learner_fit_plan_sha256"
            ],
        )

    view6 = copy.deepcopy(plan)
    source_view6 = view6[
        "source_training_membership_assembly_plan"
    ]["membership_view_specs"][5]
    for spec in view6["learner_fit_input_specs"][-2:]:
        spec["source_training_view_ordinal"] = 6
        spec["training_view_id"] = "intermediate_frozen_through_2018"
        spec["train_label_maturity_through"] = source_view6[
            "train_label_maturity_through"
        ]
        spec["maximum_training_label_maturity_session"] = source_view6[
            "train_label_maturity_through"
        ]
        _rehash(spec, "fit_input_spec_sha256")
    view6["learner_fit_input_specs_sha256"] = canonical_sha256(
        view6["learner_fit_input_specs"]
    )
    _rehash(view6, "development_oof_learner_fit_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="frozen view",
    ):
        validate_development_oof_learner_fit_plan(
            view6,
            expected_development_oof_learner_fit_plan_sha256=view6[
                "development_oof_learner_fit_plan_sha256"
            ],
        )

    different_target = copy.deepcopy(plan)
    different_target["learner_fit_input_specs"][1][
        "training_binary_target_sha256"
    ] = _h("variant-specific forbidden target")
    _rehash(
        different_target["learner_fit_input_specs"][1],
        "fit_input_spec_sha256",
    )
    different_target["learner_fit_input_specs_sha256"] = canonical_sha256(
        different_target["learner_fit_input_specs"]
    )
    _rehash(different_target, "development_oof_learner_fit_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="causal training support",
    ):
        validate_development_oof_learner_fit_plan(
            different_target,
            expected_development_oof_learner_fit_plan_sha256=(
                different_target["development_oof_learner_fit_plan_sha256"]
            ),
        )

    identical_matrices = copy.deepcopy(plan)
    identical_matrices["learner_fit_input_specs"][1][
        "training_feature_matrix_sha256"
    ] = identical_matrices["learner_fit_input_specs"][0][
        "training_feature_matrix_sha256"
    ]
    _rehash(
        identical_matrices["learner_fit_input_specs"][1],
        "fit_input_spec_sha256",
    )
    identical_matrices["learner_fit_input_specs_sha256"] = canonical_sha256(
        identical_matrices["learner_fit_input_specs"]
    )
    _rehash(identical_matrices, "development_oof_learner_fit_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="causal training support",
    ):
        validate_development_oof_learner_fit_plan(
            identical_matrices,
            expected_development_oof_learner_fit_plan_sha256=(
                identical_matrices["development_oof_learner_fit_plan_sha256"]
            ),
        )


def test_development_oof_learner_fit_plan_capabilities_and_types_fail_closed() -> None:
    _state, _reader_tip, _membership_plan, _fit_specs, plan = (
        _development_oof_learner_fit_plan_fixture()
    )
    for field, replacement in (
        *((field, False) for field in _DEVELOPMENT_OOF_ALLOWED_CAPABILITIES),
        *((field, True) for field in _DEVELOPMENT_OOF_DENIED_CAPABILITIES),
    ):
        changed = copy.deepcopy(plan)
        changed[field] = replacement
        _rehash(changed, "development_oof_learner_fit_plan_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_development_oof_learner_fit_plan(
                changed,
                expected_development_oof_learner_fit_plan_sha256=changed[
                    "development_oof_learner_fit_plan_sha256"
                ],
            )

    for field in (
        "start_consumed_request_count",
        "source_training_view_count",
        "authorized_training_view_count",
        "deferred_training_view_count",
        "model_variant_count",
        "learner_fit_input_count",
        "learner_state_output_count",
        "learner_state_schema_version",
        "maximum_fit_seconds",
    ):
        changed = copy.deepcopy(plan)
        changed[field] = True
        _rehash(changed, "development_oof_learner_fit_plan_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_development_oof_learner_fit_plan(
                changed,
                expected_development_oof_learner_fit_plan_sha256=changed[
                    "development_oof_learner_fit_plan_sha256"
                ],
            )

    bool_fit_ordinal = copy.deepcopy(plan)
    bool_fit_ordinal["learner_fit_input_specs"][0]["fit_ordinal"] = True
    _rehash(
        bool_fit_ordinal["learner_fit_input_specs"][0],
        "fit_input_spec_sha256",
    )
    bool_fit_ordinal["learner_fit_input_specs_sha256"] = canonical_sha256(
        bool_fit_ordinal["learner_fit_input_specs"]
    )
    _rehash(bool_fit_ordinal, "development_oof_learner_fit_plan_sha256")
    with pytest.raises(SecFilingGemmaStageAuthorizationError):
        validate_development_oof_learner_fit_plan(
            bool_fit_ordinal,
            expected_development_oof_learner_fit_plan_sha256=(
                bool_fit_ordinal["development_oof_learner_fit_plan_sha256"]
            ),
        )

    noncanonical_config = copy.deepcopy(plan)
    noncanonical_config["learner_config"]["raw_z_clip"] = 4
    noncanonical_config["learner_config_sha256"] = canonical_sha256(
        noncanonical_config["learner_config"]
    )
    _rehash(noncanonical_config, "development_oof_learner_fit_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="frozen learner",
    ):
        validate_development_oof_learner_fit_plan(
            noncanonical_config,
            expected_development_oof_learner_fit_plan_sha256=(
                noncanonical_config["development_oof_learner_fit_plan_sha256"]
            ),
        )

    changed_deferred = copy.deepcopy(plan)
    changed_deferred["deferred_training_views"][0]["reason"] = "fit_now"
    changed_deferred["deferred_training_views_sha256"] = canonical_sha256(
        changed_deferred["deferred_training_views"]
    )
    _rehash(changed_deferred, "development_oof_learner_fit_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="deferred training view changed",
    ):
        validate_development_oof_learner_fit_plan(
            changed_deferred,
            expected_development_oof_learner_fit_plan_sha256=(
                changed_deferred["development_oof_learner_fit_plan_sha256"]
            ),
        )


def test_development_oof_learner_fit_plan_checksum_and_exact_mapping_fail_closed() -> None:
    _state, _reader_tip, _membership_plan, _fit_specs, plan = (
        _development_oof_learner_fit_plan_fixture()
    )
    extra = copy.deepcopy(plan)
    extra["unexpected"] = False
    _rehash(extra, "development_oof_learner_fit_plan_sha256")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="keys changed",
    ):
        validate_development_oof_learner_fit_plan(
            extra,
            expected_development_oof_learner_fit_plan_sha256=extra[
                "development_oof_learner_fit_plan_sha256"
            ],
        )

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="not externally pinned",
    ):
        validate_development_oof_learner_fit_plan(
            plan,
            expected_development_oof_learner_fit_plan_sha256=_h(
                "other development OOF learner-fit plan"
            ),
        )

    inconsistent = copy.deepcopy(plan)
    inconsistent["development_oof_learner_fit_plan_sha256"] = _h(
        "inconsistent development OOF learner-fit plan"
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="self-hash is inconsistent",
    ):
        validate_development_oof_learner_fit_plan(
            inconsistent,
            expected_development_oof_learner_fit_plan_sha256=inconsistent[
                "development_oof_learner_fit_plan_sha256"
            ],
        )

    class DictSubclass(dict):
        pass

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="exact built-in dict",
    ):
        validate_development_oof_learner_fit_plan(
            DictSubclass(plan),
            expected_development_oof_learner_fit_plan_sha256=plan[
                "development_oof_learner_fit_plan_sha256"
            ],
        )


def test_development_oof_prediction_plan_is_exact_private_and_deterministic() -> None:
    state, reader_tip, source_fit_plan, fold_specs, input_specs, plan = (
        _development_oof_prediction_plan_fixture()
    )
    state_before = copy.deepcopy(state)
    tip_before = copy.deepcopy(reader_tip)

    assert set(plan) == _DEVELOPMENT_OOF_PREDICTION_PLAN_KEYS
    assert plan["schema_version"] == DEVELOPMENT_OOF_PREDICTION_PLAN_SCHEMA_VERSION
    assert plan["contract_version"] == CONTRACT_VERSION
    assert plan["contract_sha256"] == canonical_sha256(build_contract_manifest())
    assert plan["plan_kind"] == "request_free_development_oof_prediction"
    assert plan["artifact_stage"] == "development"
    assert "source_development_oof_learner_fit_plan" not in plan
    assert plan["source_development_oof_learner_fit_plan_sha256"] == (
        source_fit_plan["development_oof_learner_fit_plan_sha256"]
    )
    assert plan["source_event_count"] == 76
    assert plan["authorized_fold_count"] == 5
    assert plan["authorized_fold_ids"] == [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "fold_5",
    ]
    assert plan["prediction_fold_model_count"] == 5
    assert plan["learner_state_count"] == 10
    assert plan["model_variant_ids"] == ["semantic", "ablation"]
    assert plan["prediction_input_count"] == 56
    assert plan["available_prediction_input_count"] == 14
    assert plan["unavailable_prediction_input_count"] == 42
    assert plan["maximum_prediction_calls"] == 28
    assert plan["maximum_prediction_seconds"] == 60
    assert plan["prediction_fold_model_specs"] == fold_specs
    assert plan["prediction_input_specs"] == input_specs
    assert all(
        "2005-01-01" <= spec["decision_session"] <= "2018-12-31"
        for spec in plan["prediction_input_specs"]
    )
    compact_json = json.dumps(plan, sort_keys=True)
    assert "intermediate_frozen_through_2018" not in compact_json
    assert "2019-" not in compact_json
    for field in _DEVELOPMENT_OOF_PREDICTION_ALLOWED_CAPABILITIES:
        assert plan[field] is True
    for field in _DEVELOPMENT_OOF_PREDICTION_DENIED_CAPABILITIES:
        assert plan[field] is False
    for ordinal, spec in enumerate(plan["prediction_fold_model_specs"], start=1):
        assert set(spec) == _DEVELOPMENT_OOF_PREDICTION_FOLD_MODEL_SPEC_KEYS
        assert spec["fold_ordinal"] == ordinal
    for ordinal, spec in enumerate(plan["prediction_input_specs"], start=1):
        assert set(spec) == _DEVELOPMENT_OOF_PREDICTION_INPUT_SPEC_KEYS
        assert spec["prediction_ordinal"] == ordinal

    assert validate_development_oof_prediction_plan(
        plan,
        expected_development_oof_prediction_plan_sha256=plan[
            "development_oof_prediction_plan_sha256"
        ],
    ) == plan["development_oof_prediction_plan_sha256"]
    assert _build_development_oof_prediction_plan_for_test(
        state,
        reader_tip,
        source_fit_plan,
        fold_specs,
        input_specs,
    ) == plan
    assert state == state_before
    assert reader_tip == tip_before


def test_development_oof_prediction_plan_rebuilds_source_and_exact_population() -> None:
    state, reader_tip, source_fit_plan, fold_specs, input_specs, _plan = (
        _development_oof_prediction_plan_fixture()
    )

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="omit or add",
    ):
        _build_development_oof_prediction_plan_for_test(
            state,
            reader_tip,
            source_fit_plan,
            fold_specs,
            input_specs[:-1],
        )

    changed_source = copy.deepcopy(source_fit_plan)
    changed_source["development_oof_learner_fit_plan_sha256"] = _h(
        "forged source fit plan"
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="self-hash is inconsistent",
    ):
        _build_development_oof_prediction_plan_for_test(
            state,
            reader_tip,
            changed_source,
            fold_specs,
            input_specs,
        )

    changed_tip = copy.deepcopy(reader_tip)
    changed_tip["development_model_reader_receipts"] = {}
    _rehash(changed_tip, "tip_anchor_sha256")
    with pytest.raises(SecFilingGemmaStageAuthorizationError):
        _build_development_oof_prediction_plan_for_test(
            state,
            changed_tip,
            source_fit_plan,
            fold_specs,
            input_specs,
        )


def test_development_oof_prediction_plan_rejects_fold_and_state_tampering() -> None:
    _state, _reader_tip, _source_fit_plan, _fold_specs, _input_specs, plan = (
        _development_oof_prediction_plan_fixture()
    )

    changed_plans: list[dict] = []

    reordered = copy.deepcopy(plan)
    reordered["prediction_fold_model_specs"].reverse()
    changed_plans.append(reordered)

    view6 = copy.deepcopy(plan)
    view6_spec = view6["prediction_fold_model_specs"][-1]
    view6_spec["fold_id"] = "intermediate_frozen_through_2018"
    _rehash(view6_spec, "prediction_fold_model_spec_sha256")
    changed_plans.append(view6)

    duplicate_state = copy.deepcopy(plan)
    duplicate_state_spec = duplicate_state["prediction_fold_model_specs"][1]
    duplicate_state_spec["semantic_learner_state_sha256"] = duplicate_state[
        "prediction_fold_model_specs"
    ][0]["semantic_learner_state_sha256"]
    _rehash(duplicate_state_spec, "prediction_fold_model_spec_sha256")
    changed_plans.append(duplicate_state)

    crossed_schema = copy.deepcopy(plan)
    crossed_schema_spec = crossed_schema["prediction_fold_model_specs"][1]
    crossed_schema_spec["feature_schema_sha256"] = _h(
        "crossed prediction feature schema"
    )
    _rehash(crossed_schema_spec, "prediction_fold_model_spec_sha256")
    changed_plans.append(crossed_schema)

    for changed in changed_plans:
        changed["prediction_fold_model_specs_sha256"] = canonical_sha256(
            changed["prediction_fold_model_specs"]
        )
        _rehash(changed, "development_oof_prediction_plan_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_development_oof_prediction_plan(
                changed,
                expected_development_oof_prediction_plan_sha256=changed[
                    "development_oof_prediction_plan_sha256"
                ],
            )


def test_development_oof_prediction_plan_rejects_input_chronology_and_null_tampering() -> None:
    _state, _reader_tip, _source_fit_plan, _fold_specs, _input_specs, plan = (
        _development_oof_prediction_plan_fixture()
    )

    changed_plans: list[dict] = []

    reordered = copy.deepcopy(plan)
    reordered["prediction_input_specs"][:2] = reversed(
        reordered["prediction_input_specs"][:2]
    )
    changed_plans.append(reordered)

    duplicated_decision = copy.deepcopy(plan)
    duplicate_spec = duplicated_decision["prediction_input_specs"][1]
    first_spec = duplicated_decision["prediction_input_specs"][0]
    duplicate_spec["decision_session"] = first_spec["decision_session"]
    duplicate_spec["accession_number"] = first_spec["accession_number"]
    _rehash(duplicate_spec, "prediction_input_spec_sha256")
    changed_plans.append(duplicated_decision)

    pre_2005 = copy.deepcopy(plan)
    pre_2005_spec = pre_2005["prediction_input_specs"][0]
    pre_2005_spec["decision_session"] = "2004-12-31"
    _rehash(pre_2005_spec, "prediction_input_spec_sha256")
    changed_plans.append(pre_2005)

    unavailable_vector = copy.deepcopy(plan)
    unavailable_spec = unavailable_vector["prediction_input_specs"][1]
    assert unavailable_spec["prediction_available"] is False
    unavailable_spec["semantic_feature_values_sha256"] = _h(
        "forbidden unavailable vector"
    )
    _rehash(unavailable_spec, "prediction_input_spec_sha256")
    changed_plans.append(unavailable_vector)

    available_reason = copy.deepcopy(plan)
    available_spec = available_reason["prediction_input_specs"][0]
    assert available_spec["prediction_available"] is True
    available_spec["unavailable_reason"] = "missing_required_market_features"
    _rehash(available_spec, "prediction_input_spec_sha256")
    changed_plans.append(available_reason)

    for changed in changed_plans:
        changed["prediction_input_specs_sha256"] = canonical_sha256(
            changed["prediction_input_specs"]
        )
        _rehash(changed, "development_oof_prediction_plan_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_development_oof_prediction_plan(
                changed,
                expected_development_oof_prediction_plan_sha256=changed[
                    "development_oof_prediction_plan_sha256"
                ],
            )


def test_development_oof_prediction_plan_capabilities_types_and_pins_fail_closed() -> None:
    _state, _reader_tip, _source_fit_plan, _fold_specs, _input_specs, plan = (
        _development_oof_prediction_plan_fixture()
    )
    for field, replacement in (
        *((field, False) for field in _DEVELOPMENT_OOF_PREDICTION_ALLOWED_CAPABILITIES),
        *((field, True) for field in _DEVELOPMENT_OOF_PREDICTION_DENIED_CAPABILITIES),
    ):
        changed = copy.deepcopy(plan)
        changed[field] = replacement
        _rehash(changed, "development_oof_prediction_plan_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_development_oof_prediction_plan(
                changed,
                expected_development_oof_prediction_plan_sha256=changed[
                    "development_oof_prediction_plan_sha256"
                ],
            )

    for field in (
        "source_event_count",
        "authorized_fold_count",
        "prediction_fold_model_count",
        "learner_state_count",
        "prediction_input_count",
        "maximum_prediction_calls",
        "maximum_prediction_seconds",
    ):
        changed = copy.deepcopy(plan)
        changed[field] = True
        _rehash(changed, "development_oof_prediction_plan_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_development_oof_prediction_plan(
                changed,
                expected_development_oof_prediction_plan_sha256=changed[
                    "development_oof_prediction_plan_sha256"
                ],
            )

    extra = copy.deepcopy(plan)
    extra["unexpected"] = False
    _rehash(extra, "development_oof_prediction_plan_sha256")
    with pytest.raises(SecFilingGemmaStageAuthorizationError, match="keys changed"):
        validate_development_oof_prediction_plan(
            extra,
            expected_development_oof_prediction_plan_sha256=extra[
                "development_oof_prediction_plan_sha256"
            ],
        )

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="not externally pinned",
    ):
        validate_development_oof_prediction_plan(
            plan,
            expected_development_oof_prediction_plan_sha256=_h(
                "other OOF prediction plan"
            ),
        )

    inconsistent = copy.deepcopy(plan)
    inconsistent["development_oof_prediction_plan_sha256"] = _h(
        "inconsistent OOF prediction plan"
    )
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="self-hash is inconsistent",
    ):
        validate_development_oof_prediction_plan(
            inconsistent,
            expected_development_oof_prediction_plan_sha256=inconsistent[
                "development_oof_prediction_plan_sha256"
            ],
        )

    class DictSubclass(dict):
        pass

    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="exact built-in dict",
    ):
        validate_development_oof_prediction_plan(
            DictSubclass(plan),
            expected_development_oof_prediction_plan_sha256=plan[
                "development_oof_prediction_plan_sha256"
            ],
        )


def test_development_policy_replay_plan_is_exact_private_and_deterministic() -> None:
    state, reader_tip, prediction_plan, replay_specs, plan = (
        _development_policy_replay_plan_fixture()
    )
    state_before = copy.deepcopy(state)
    tip_before = copy.deepcopy(reader_tip)
    allowed = {
        "source_development_oof_prediction_batch_access_permitted",
        "raw_prediction_components_access_permitted",
        "threshold_evaluation_permitted",
        "policy_state_transition_permitted",
        "compact_policy_replay_output_permitted",
    }
    denied = {
        "numeric_prediction_permitted",
        "learner_state_access_permitted",
        "source_feature_batch_access_permitted",
        "source_label_batch_access_permitted",
        "label_access_permitted",
        "outcome_access_permitted",
        "post_decision_market_data_access_permitted",
        "post_2018_data_access_permitted",
        "deferred_training_view_access_permitted",
        "deferred_training_view_state_access_permitted",
        "learner_fit_permitted",
        "learner_refit_permitted",
        "learner_state_update_permitted",
        "online_learning_permitted",
        "candidate_selection_permitted",
        "scoring_permitted",
        "prediction_sealing_permitted",
        "policy_replay_sealing_permitted",
        "label_release_permitted",
        "holdout_access_permitted",
        "model_transport_access_permitted",
        "network_access_permitted",
        "raw_prediction_mutation_permitted",
        "row_drop_permitted",
        "row_reordering_permitted",
        "policy_retry_permitted",
        "ledger_mutation_permitted",
        "stage_promotion_permitted",
        "production_permitted",
    }
    identity = {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "plan_kind",
        "artifact_stage",
        "development_root_scope_sha256",
        "start_store_state_bytes_sha256",
        "start_store_state_sha256",
        "start_current_tip_anchor_bytes_sha256",
        "start_current_tip_anchor_sha256",
        "start_current_tip_revision",
        "start_consumed_request_count",
        "source_development_oof_prediction_plan_sha256",
        "source_development_oof_prediction_projection_sha256",
        "source_development_oof_prediction_batch_sha256",
        "source_raw_prediction_rows_sha256",
        "source_raw_prediction_tip_sha256",
        "source_raw_prediction_row_count",
        "policy_replay_input_count",
        "policy_replay_input_specs",
        "policy_replay_input_specs_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "development_cutoff_session",
        "candidate_count",
        "candidate_ids",
        "candidate_threshold_specs",
        "candidate_threshold_specs_sha256",
        "model_variant_count",
        "model_variant_ids",
        "candidate_gate_comparison_rule",
        "cash_episode_sessions",
        "cash_episode_rule",
        "unavailable_prediction_rule",
        "policy_replay_order_rule",
        "development_policy_replay_plan_sha256",
    }
    assert set(plan) == identity | allowed | denied
    assert plan["schema_version"] == DEVELOPMENT_POLICY_REPLAY_PLAN_SCHEMA_VERSION
    assert plan["start_store_state_sha256"] == state["state_sha256"]
    assert plan["start_current_tip_anchor_sha256"] == reader_tip[
        "tip_anchor_sha256"
    ]
    assert plan["start_current_tip_revision"] == reader_tip["revision"]
    assert plan["start_consumed_request_count"] == 0
    assert plan["source_development_oof_prediction_plan_sha256"] == (
        prediction_plan["development_oof_prediction_plan_sha256"]
    )
    assert plan["source_raw_prediction_row_count"] == len(replay_specs)
    assert plan["policy_replay_input_count"] == len(replay_specs)
    assert plan["policy_replay_input_specs"] == replay_specs
    assert plan["candidate_ids"] == [
        "p50_e0",
        "p55_e0",
        "p50_e25",
        "p55_e25",
    ]
    assert [
        spec["probability_gate_hex"]
        for spec in plan["candidate_threshold_specs"]
    ] == [(0.50).hex(), (0.55).hex(), (0.50).hex(), (0.55).hex()]
    assert [
        spec["expected_edge_gate_hex"]
        for spec in plan["candidate_threshold_specs"]
    ] == [(0.0).hex(), (0.0).hex(), (0.0025).hex(), (0.0025).hex()]
    assert plan["model_variant_ids"] == ["semantic", "ablation"]
    assert plan["candidate_gate_comparison_rule"] == (
        "probability_gte_and_expected_edge_gte"
    )
    assert plan["cash_episode_sessions"] == 20
    assert "never_extend" in plan["cash_episode_rule"]
    assert "starts_no_new_cash_episode" in plan["unavailable_prediction_rule"]
    for field in allowed:
        assert plan[field] is True
    for field in denied:
        assert plan[field] is False
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    for forbidden in (
        '"source_development_oof_prediction_plan":',
        '"source_label_batch":',
        '"labels":',
        '"outcomes":',
        '"prices":',
        "intermediate_frozen_through_2018",
        "2019-01-01",
        "2023-12-31",
    ):
        assert forbidden not in encoded
    assert validate_development_policy_replay_plan(
        plan,
        expected_development_policy_replay_plan_sha256=plan[
            "development_policy_replay_plan_sha256"
        ],
    ) == plan["development_policy_replay_plan_sha256"]
    assert state == state_before
    assert reader_tip == tip_before


def test_development_policy_replay_plan_rejects_crossed_bytes_and_consumed_store() -> None:
    state, reader_tip, prediction_plan, replay_specs, _plan = (
        _development_policy_replay_plan_fixture()
    )
    encoded_tip = (
        json.dumps(reader_tip, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="byte pins crossed",
    ):
        build_development_policy_replay_plan(
            state,
            development_root_scope_sha256=prediction_plan[
                "development_root_scope_sha256"
            ],
            authenticated_store_state_bytes_sha256=_h("crossed state bytes"),
            source_development_oof_prediction_plan=prediction_plan,
            source_development_oof_prediction_projection_sha256=_h(
                "policy replay source prediction projection"
            ),
            source_development_oof_prediction_batch_sha256=_h(
                "policy replay source prediction batch"
            ),
            source_raw_prediction_rows_sha256=_h(
                "policy replay source raw rows"
            ),
            source_raw_prediction_tip_sha256=_h(
                "policy replay source raw tip"
            ),
            source_raw_prediction_row_count=len(replay_specs),
            policy_replay_input_specs=replay_specs,
            independent_current_tip_anchor=reader_tip,
            independent_current_tip_anchor_bytes_sha256=hashlib.sha256(
                encoded_tip
            ).hexdigest(),
        )

    consumed_state, _pin, _entry, _grant, consumed_tip = _grant_context(1)
    encoded_state = (
        json.dumps(consumed_state, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    encoded_consumed_tip = (
        json.dumps(consumed_tip, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    with pytest.raises(
        SecFilingGemmaStageAuthorizationError,
        match="zero-consumption",
    ):
        build_development_policy_replay_plan(
            consumed_state,
            development_root_scope_sha256=prediction_plan[
                "development_root_scope_sha256"
            ],
            authenticated_store_state_bytes_sha256=hashlib.sha256(
                encoded_state
            ).hexdigest(),
            source_development_oof_prediction_plan=prediction_plan,
            source_development_oof_prediction_projection_sha256=_h(
                "policy replay source prediction projection"
            ),
            source_development_oof_prediction_batch_sha256=_h(
                "policy replay source prediction batch"
            ),
            source_raw_prediction_rows_sha256=_h(
                "policy replay source raw rows"
            ),
            source_raw_prediction_tip_sha256=_h(
                "policy replay source raw tip"
            ),
            source_raw_prediction_row_count=len(replay_specs),
            policy_replay_input_specs=replay_specs,
            independent_current_tip_anchor=consumed_tip,
            independent_current_tip_anchor_bytes_sha256=hashlib.sha256(
                encoded_consumed_tip
            ).hexdigest(),
        )


def test_development_policy_replay_plan_tampering_fails_closed() -> None:
    _state, _tip, _prediction_plan, _replay_specs, plan = (
        _development_policy_replay_plan_fixture()
    )
    for field, replacement in (
        ("threshold_evaluation_permitted", False),
        ("label_access_permitted", True),
        ("scoring_permitted", True),
        ("learner_refit_permitted", True),
        ("deferred_training_view_access_permitted", True),
        ("post_2018_data_access_permitted", True),
        ("network_access_permitted", True),
        ("production_permitted", True),
    ):
        changed = copy.deepcopy(plan)
        changed[field] = replacement
        _rehash(changed, "development_policy_replay_plan_sha256")
        with pytest.raises(SecFilingGemmaStageAuthorizationError):
            validate_development_policy_replay_plan(
                changed,
                expected_development_policy_replay_plan_sha256=changed[
                    "development_policy_replay_plan_sha256"
                ],
            )

    reordered = copy.deepcopy(plan)
    reordered["candidate_ids"].reverse()
    _rehash(reordered, "development_policy_replay_plan_sha256")
    with pytest.raises(SecFilingGemmaStageAuthorizationError):
        validate_development_policy_replay_plan(
            reordered,
            expected_development_policy_replay_plan_sha256=reordered[
                "development_policy_replay_plan_sha256"
            ],
        )

    crossed_input = copy.deepcopy(plan)
    crossed_input["policy_replay_input_specs"][0][
        "decision_session"
    ] = "2019-01-02"
    _rehash(
        crossed_input["policy_replay_input_specs"][0],
        "policy_replay_input_spec_sha256",
    )
    crossed_input["policy_replay_input_specs_sha256"] = canonical_sha256(
        crossed_input["policy_replay_input_specs"]
    )
    _rehash(crossed_input, "development_policy_replay_plan_sha256")
    with pytest.raises(SecFilingGemmaStageAuthorizationError):
        validate_development_policy_replay_plan(
            crossed_input,
            expected_development_policy_replay_plan_sha256=crossed_input[
                "development_policy_replay_plan_sha256"
            ],
        )
