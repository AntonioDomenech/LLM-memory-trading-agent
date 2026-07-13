from __future__ import annotations

import copy
from datetime import date
from functools import lru_cache
import hashlib
from unittest.mock import patch

import pytest

import agent_benchmark.sec_filing_gemma_stage_verifier as verifier_module

from agent_benchmark.sec_filing_gemma_contract import (
    CONTRACT_VERSION,
    REQUIRED_SOURCE_HASHES,
    REQUIRED_STAGE_VERIFIER_CHECKS,
    build_candidate_manifest,
    build_corpus_universe_manifest,
    canonical_sha256,
    session_calendar_sha256,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    REVEAL_REQUEST_SCHEMA_VERSION,
    candidate_design_sha256,
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
    DEVELOPMENT_SEC_EXECUTION_ABORT_SCHEMA_VERSION,
    DEVELOPMENT_SEC_EXECUTION_CLAIM_SCHEMA_VERSION,
    DEVELOPMENT_SEC_READER_RECEIPT_SCHEMA_VERSION,
    OWNED_SEC_RAW_BATCH_MAX_BYTES,
    SEC_EXECUTION_RESOLVED_SOURCE_PATHS,
    SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID,
    SEMANTIC_PREREQUISITE_SCHEMA_VERSION,
    STAGE_CARRY_IN_READER_RECEIPT_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
    TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION,
    TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION,
    SecFilingGemmaStageAuthorizationError,
    authenticate_reveal_store_trusted_stage_content_pin,
    build_reveal_store_current_tip_anchor,
    build_consumed_stage_authorization_grant,
    build_consumed_stage_output_receipt,
    build_development_sec_execution_abort,
    build_development_sec_execution_claim,
    build_development_sec_reader_receipt,
    build_stage_carry_in_reader_receipt,
    build_stage_sec_execution_abort,
    build_stage_sec_execution_claim,
    build_stage_sec_reader_receipt,
    derive_consumed_stage_store_state_pin,
    derive_reveal_store_trusted_stage_content_pin,
    validate_consumed_stage_authorization_grant,
    validate_consumed_stage_output_receipt,
    validate_consumed_stage_store_state_pin,
    validate_reveal_store_current_tip_anchor_transition,
    validate_stage_carry_in_reader_receipt,
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
        for form, month in (
            ("10-K", 2),
            ("10-Q", 5),
            ("10-Q", 8),
            ("10-Q", 11),
        ):
            records.append(
                _development_source_record(year, serial, form, month)
            )
            serial += 1
    for form, month in (("10-Q", 2), ("10-Q", 5)):
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
        identity_lexicon_sha256=_h("development identity lexicon"),
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
        identity_lexicon_sha256=_h(f"lexicon:{sequence}"),
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
        )
    }
    maps.update(map_overrides)
    return build_reveal_store_current_tip_anchor(
        state,
        revision=prior_tip["revision"] + 1,
        previous_tip_anchor_sha256=prior_tip["tip_anchor_sha256"],
        **maps,
    )


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
    candidate = _sec_candidate(1)
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
    parent_entry = _rewrite_entry(parent_raw, access=parent_access)
    parent_state = _state_from_entries([parent_entry], candidate)
    parent_entry, parent_grant, parent_bundle = _bundle_for_state(parent_state)
    parent_request_hash = parent_entry["request_sha256"]
    parent_tip = build_reveal_store_current_tip_anchor(
        parent_state,
        revision=1,
        previous_tip_anchor_sha256=_h("parent prior tip"),
        authorization_bundles={parent_request_hash: parent_bundle},
    )
    parent_claim = build_stage_sec_execution_claim(
        parent_bundle,
        independent_current_tip_anchor=parent_tip,
        execution_source_hashes=_execution_source_hashes(),
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
    child_document = {
        "accession_number": "0000320193-24-000333",
        "official_url": (
            "https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019324000333/aapl-20240928.htm"
        ),
    }
    child_access["sec_access_plan"] = {
        "selection_policy": "all_and_only_requested_stage_universe_primary_documents",
        "method": "GET",
        "network_scope": "official_sec_https_only",
        "redirects_permitted": False,
        "retries_permitted": False,
        "cache_substitution_permitted": False,
        "document_count": 1,
        "accessions_sha256": canonical_sha256(
            [child_document["accession_number"]]
        ),
        "official_urls_sha256": canonical_sha256([child_document["official_url"]]),
        "documents": [child_document],
    }
    child_access["budgets"] = {
        "max_sec_requests": 1,
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
    )
    child_claim = build_stage_sec_execution_claim(
        child_bundle,
        independent_current_tip_anchor=child_tip,
        execution_source_hashes=_execution_source_hashes(),
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
