from __future__ import annotations

import copy
from collections.abc import Mapping
from datetime import date
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import MappingProxyType
from unittest.mock import patch

import pytest

import agent_benchmark.sec_filing_gemma_reveal_store as reveal_store_module

from agent_benchmark.sec_audit_transport import ResponseAudit
from agent_benchmark.sec_filing_gemma_contract import (
    CONTRACT_VERSION,
    REQUIRED_SOURCE_HASHES,
    build_candidate_manifest,
    build_corpus_universe_manifest,
    build_stage_content_manifest,
    canonical_sha256,
    session_calendar_sha256,
)
from agent_benchmark.sec_filing_gemma_corpus import (
    SecCorpusBudget,
    _acquire_authenticated_stage_access_document_batch,
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
    SEC_BATCH_COMPLETE_MARKER_FILENAME,
    SEC_BATCH_COMPLETE_MARKER_SCHEMA_VERSION,
    SEC_STAGE_COMPONENT_DIRECTORY_NAME,
    STATE_FILENAME,
    STAGE_EVIDENCE_COMPLETE_MARKER_FILENAME,
    STAGE_EVIDENCE_COMPLETE_MARKER_SCHEMA_VERSION,
    STAGE_EVIDENCE_COMPONENT_DIRECTORY_NAME,
    STAGE_EVIDENCE_FILENAME,
    STAGE_OUTPUTS_DIRECTORY_NAME,
    SecFilingGemmaRevealStore,
    SecFilingGemmaRevealStoreError,
    SemanticPrerequisiteValidation,
)
from agent_benchmark.sec_filing_gemma_stage_access import (
    STAGE_ACCESS_MANIFEST_SCHEMA_VERSION,
    _prior_same_form_carry_ins,
)
from agent_benchmark.sec_filing_gemma_stage_authorization import (
    CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION,
    SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID,
    STAGE_EVIDENCE_OUTPUT_COMPONENT_ID,
    STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH,
    SecFilingGemmaStageAuthorizationError,
    build_consumed_stage_authorization_grant,
    validate_consumed_stage_authorization_grant,
    _sec_component_plan_from_bundle,
)
from agent_benchmark.sec_filing_gemma_source_identity import (
    CANONICAL_SOURCE_ROLE_PATHS,
)
from agent_benchmark.sec_filing_gemma_stage_verifier import (
    STAGE_AUDIT_RECEIPT_SCHEMA_VERSION,
    STAGE_EVIDENCE_KEYS,
    STAGE_EVIDENCE_SCHEMA_VERSION,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS
from agent_benchmark.sec_point_in_time import content_sha256, validate_sec_user_agent


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_ID = AUTHORITATIVE_VALIDATOR_ID
VALIDATOR_SOURCE_SHA256 = hashlib.sha256(
    (REPO_ROOT / "agent_benchmark/sec_filing_gemma_stage_verifier.py").read_bytes()
).hexdigest()
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
            name: hashlib.sha256(
                (REPO_ROOT / CANONICAL_SOURCE_ROLE_PATHS[name]).read_bytes()
            ).hexdigest()
            if CANONICAL_SOURCE_ROLE_PATHS[name] is not None
            else _digest(f"{salt}:source:{name}")
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


def _stage_evidence_hash(evidence: dict) -> str:
    return evidence.get("stage_evidence_sha256", canonical_sha256(evidence))


def _owned_v3_stage_evidence(
    candidate: dict,
    grant: dict,
    *,
    salt: str,
) -> dict:
    body = {
        "schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "prerequisite_stage": grant["stage"],
        "parent_stage_evidence_sha256": grant[
            "prerequisite_stage_evidence_sha256"
        ],
        "parent_stage_lineage": None,
        "contract_manifest": {"fixture": salt},
        "candidate_manifest": copy.deepcopy(candidate),
        "source_bytes_base64_by_role": {},
        "calendar_evidence_manifest": {},
        "calendar_source_bytes_base64_by_name": {},
        "corpus_universe_manifest": {},
        "catalog_replay": {},
        "content_replays_by_stage": {},
        "prerequisite_content_manifest": {},
        "model_batches_by_stage": {},
        "market_replays_by_stage": {},
        "prediction_replay": {},
        "learner_replays": {},
        "score_replay": {},
        "stage_runtime_receipt": {},
        "reveal_registry": {},
        "registry_external_pin": {},
    }
    assert set(body) | {"stage_evidence_sha256"} == STAGE_EVIDENCE_KEYS
    return {**body, "stage_evidence_sha256": canonical_sha256(body)}


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
        prerequisite_stage_evidence_sha256=_stage_evidence_hash(evidence),
    )
    return request, access_manifest


def _grant_request(
    state: dict,
    candidate: dict,
    *,
    stage: str,
    evidence: dict,
    include_sec_plan: bool = False,
    sec_documents: list[dict] | None = None,
    prior_same_form_carry_in: dict | None = None,
    prerequisite_evidence_pin: dict | None = None,
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
    if prerequisite_evidence_pin is not None:
        access_body["prerequisite_evidence_pin"] = copy.deepcopy(
            prerequisite_evidence_pin
        )
    if include_sec_plan:
        documents = (
            [
                {
                    "accession_number": "0000320193-24-000123",
                    "official_url": (
                        "https://www.sec.gov/Archives/edgar/data/320193/"
                        "000032019324000123/aapl-20240928.htm"
                    ),
                }
            ]
            if sec_documents is None
            else copy.deepcopy(sec_documents)
        )
        accessions = [document["accession_number"] for document in documents]
        official_urls = [document["official_url"] for document in documents]
        access_body["sec_access_plan"] = {
            "selection_policy": (
                "all_and_only_requested_stage_universe_primary_documents"
            ),
            "method": "GET",
            "network_scope": "official_sec_https_only",
            "redirects_permitted": False,
            "retries_permitted": False,
            "cache_substitution_permitted": False,
            "document_count": len(documents),
            "accessions_sha256": canonical_sha256(accessions),
            "official_urls_sha256": canonical_sha256(official_urls),
            "documents": documents,
        }
        access_body["budgets"] = {
            "max_sec_requests": len(documents),
            "max_sec_response_bytes": 1_000_000,
            "max_sec_acquisition_seconds": 30.0,
        }
    if prior_same_form_carry_in is not None:
        access_body["prior_same_form_carry_in"] = copy.deepcopy(
            prior_same_form_carry_in
        )
        access_body["scope"] = {
            "authorized_stage": stage,
            "prohibited_stages": [
                value
                for value in ("development", "intermediate", "final")
                if value != stage
            ],
            "prerequisite_stage_data_scope": (
                "sealed_evidence_identity_plus_exact_read_only_prior_same_form_"
                "normalized_text_carry_in"
            ),
            "general_cross_stage_access_permitted": False,
            "exact_prior_same_form_carry_in_read_permitted": True,
            "future_stage_access_permitted": False,
            "outcome_access_before_atomic_request_consumption_permitted": False,
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
        prerequisite_stage_evidence_sha256=_stage_evidence_hash(evidence),
    )
    return request, access_manifest


def _semantic_validator(
    evidence: dict,
    access_manifest: dict,
    context: dict,
    *,
    authenticated_store_context: dict,
) -> SemanticPrerequisiteValidation:
    evidence_stage = evidence.get("stage", evidence.get("prerequisite_stage"))
    candidate_manifest = evidence.get("candidate_manifest")
    evidence_candidate = evidence.get("candidate_sha256")
    if type(candidate_manifest) is dict:
        evidence_candidate = candidate_manifest.get("candidate_sha256")
    assert evidence_stage == context["prerequisite_stage"]
    assert evidence_candidate == context["candidate_sha256"]
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


def _issued_intermediate_grant(
    store: SecFilingGemmaRevealStore,
    *,
    salt: str,
    include_sec_plan: bool = False,
) -> tuple[dict, dict, dict]:
    store.initialize()
    registered, candidate = _register(store, salt=salt)
    development_evidence = _evidence(
        "development",
        candidate,
        salt=f"{salt}-development",
    )
    request, access_manifest = _grant_request(
        registered,
        candidate,
        stage="intermediate",
        evidence=development_evidence,
        include_sec_plan=include_sec_plan,
    )
    bundle = _consume_with_grant(
        store,
        request,
        candidate,
        development_evidence,
        stage="intermediate",
        access_manifest=access_manifest,
    )
    return candidate, request, bundle


def _issued_sec_claim(
    store: SecFilingGemmaRevealStore,
    *,
    salt: str,
) -> tuple[dict, dict, dict, dict]:
    candidate, request, bundle = _issued_intermediate_grant(
        store,
        salt=salt,
        include_sec_plan=True,
    )
    claim_result = store.claim_authorized_sec_stage_execution(
        request_sha256=request["request_sha256"],
        sec_user_agent_sha256=SEC_TEST_USER_AGENT_SHA256,
    )
    assert claim_result["created"] is True
    return candidate, request, bundle, claim_result["claim"]


SEC_TEST_USER_AGENT = "Private Owner owner-contact@real-domain-for-tests.dev"
SEC_TEST_USER_AGENT_SHA256 = validate_sec_user_agent(SEC_TEST_USER_AGENT).sha256


class _FixedSecTransport:
    def __init__(self, payloads: dict[str, bytes], component_plan: dict) -> None:
        self.payloads = payloads
        self.component_plan = component_plan
        self.user_agent_audit = validate_sec_user_agent(SEC_TEST_USER_AGENT)

    def acquisition_security_state(self) -> dict:
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
            "transport_max_requests": self.component_plan["max_sec_requests"],
            "transport_max_bytes": self.component_plan["max_sec_response_bytes"],
            "transport_max_seconds": float(
                self.component_plan["max_sec_acquisition_seconds"]
            ),
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


def _write_fixed_sec_component(
    store: SecFilingGemmaRevealStore,
    claim: dict,
    *,
    payloads: tuple[bytes, ...] = (
        b"<html><body><p>Fixed SEC filing bytes.</p></body></html>",
    ),
) -> tuple[Path, Path, list[dict]]:
    tip = store.load_current_tip_anchor()
    bundle = tip["authorization_bundles"][claim["request_sha256"]]
    _exact_bundle, _grant, component_plan = _sec_component_plan_from_bundle(bundle)
    documents_plan = component_plan["sec_access_plan"]["documents"]
    assert len(payloads) == len(documents_plan)
    transport_payloads = {
        document["official_url"]: payload
        for document, payload in zip(documents_plan, payloads)
    }
    batch = _acquire_authenticated_stage_access_document_batch(
        authenticated_document_plan=documents_plan,
        transport=_FixedSecTransport(transport_payloads, component_plan),
        user_agent=SEC_TEST_USER_AGENT,
        budget=SecCorpusBudget(
            clock=lambda: 0.0,
            max_requests=component_plan["max_sec_requests"],
            max_bytes=component_plan["max_sec_response_bytes"],
            max_seconds=float(component_plan["max_sec_acquisition_seconds"]),
        ),
    )
    component_directory = (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / claim["claim_sha256"]
        / SEC_STAGE_COMPONENT_DIRECTORY_NAME
    )
    component_directory.mkdir(parents=True)
    evidence_payloads: list[tuple[str, str, bytes]] = []
    for document_ordinal, document in enumerate(batch.documents, start=1):
        prefix = f"document-{document_ordinal:04d}"
        evidence_payloads.extend(
            (
                (f"{prefix}-raw", f"{prefix}.raw", document.raw_primary_document),
                (
                    f"{prefix}-normalized",
                    f"{prefix}.normalized.txt",
                    document.normalized_text,
                ),
            )
        )
    evidence_payloads.extend(
        (
            (
                "request-receipts-json",
                "request-receipts.json",
                batch.request_receipts_json,
            ),
            (
                "byte-manifest-json",
                "byte-manifest.json",
                batch.byte_manifest_json,
            ),
        )
    )
    byte_index: list[dict] = []
    for ordinal, (logical_id, relative_path, payload) in enumerate(
        evidence_payloads, start=1
    ):
        (component_directory / relative_path).write_bytes(payload)
        byte_index.append(
            {
                "ordinal": ordinal,
                "logical_id": logical_id,
                "relative_path": relative_path,
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    marker_body = {
        "schema_version": SEC_BATCH_COMPLETE_MARKER_SCHEMA_VERSION,
        "request_sha256": claim["request_sha256"],
        "claim_sha256": claim["claim_sha256"],
        "component_id": SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID,
        "byte_index": byte_index,
        "byte_index_sha256": canonical_sha256(byte_index),
    }
    marker = {
        **marker_body,
        "marker_sha256": canonical_sha256(marker_body),
    }
    marker_path = component_directory / SEC_BATCH_COMPLETE_MARKER_FILENAME
    marker_path.write_bytes(reveal_store_module._encoded_state(marker))
    return component_directory, marker_path, byte_index


def _rewrite_complete_marker(marker_path: Path, marker: dict) -> None:
    marker["byte_index_sha256"] = canonical_sha256(marker["byte_index"])
    marker["marker_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in marker.items()
            if key != "marker_sha256"
        }
    )
    marker_path.write_bytes(reveal_store_module._encoded_state(marker))


def _complete_fixed_sec_ancestry(
    store: SecFilingGemmaRevealStore,
    request_sha256: str,
    *,
    payloads: tuple[bytes, ...] | None = None,
) -> tuple[dict, dict]:
    tip = store.load_current_tip_anchor()
    claim = tip["stage_sec_execution_claims"].get(request_sha256)
    if claim is None:
        claim_result = store.claim_authorized_sec_stage_execution(
            request_sha256=request_sha256,
            sec_user_agent_sha256=SEC_TEST_USER_AGENT_SHA256,
        )
        claim = claim_result["claim"]
    tip = store.load_current_tip_anchor()
    reader = tip["stage_sec_reader_receipts"].get(request_sha256)
    if reader is None:
        if payloads is None:
            _write_fixed_sec_component(store, claim)
        else:
            _write_fixed_sec_component(store, claim, payloads=payloads)
        reader = store._record_authorized_sec_stage_reader_output(
            request_sha256=request_sha256,
        )
    return claim, reader


def _write_fixed_stage_evidence_component(
    store: SecFilingGemmaRevealStore,
    claim: dict,
    sec_reader: dict,
    stage_evidence: dict,
) -> tuple[Path, Path]:
    component_directory = (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / claim["claim_sha256"]
        / STAGE_EVIDENCE_COMPONENT_DIRECTORY_NAME
    )
    component_directory.mkdir(parents=True, exist_ok=True)
    evidence_bytes = json.dumps(
        stage_evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    evidence_path = component_directory / STAGE_EVIDENCE_FILENAME
    evidence_path.write_bytes(evidence_bytes)
    marker_body = {
        "schema_version": STAGE_EVIDENCE_COMPLETE_MARKER_SCHEMA_VERSION,
        "request_sha256": claim["request_sha256"],
        "claim_sha256": claim["claim_sha256"],
        "sec_reader_receipt_sha256": sec_reader["receipt_sha256"],
        "component_id": STAGE_EVIDENCE_OUTPUT_COMPONENT_ID,
        "relative_path": STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH,
        "byte_count": len(evidence_bytes),
        "document_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "stage_evidence_sha256": stage_evidence["stage_evidence_sha256"],
    }
    marker = {**marker_body, "marker_sha256": canonical_sha256(marker_body)}
    marker_path = component_directory / STAGE_EVIDENCE_COMPLETE_MARKER_FILENAME
    marker_path.write_bytes(reveal_store_module._encoded_state(marker))
    return evidence_path, marker_path


def _rewrite_fixed_stage_evidence_component(
    prepared: dict,
    evidence: dict,
    *,
    recompute_self_hash: bool = True,
) -> None:
    body = {
        key: value for key, value in evidence.items() if key != "stage_evidence_sha256"
    }
    if recompute_self_hash:
        evidence["stage_evidence_sha256"] = canonical_sha256(body)
    evidence_bytes = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    prepared["evidence_path"].write_bytes(evidence_bytes)
    marker = json.loads(prepared["marker_path"].read_bytes())
    marker["byte_count"] = len(evidence_bytes)
    marker["document_sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    marker["stage_evidence_sha256"] = evidence["stage_evidence_sha256"]
    marker["marker_sha256"] = canonical_sha256(
        {key: value for key, value in marker.items() if key != "marker_sha256"}
    )
    prepared["marker_path"].write_bytes(
        reveal_store_module._encoded_state(marker)
    )


def _prepare_fixed_stage_evidence_output(
    store: SecFilingGemmaRevealStore,
    request: dict,
    candidate: dict,
    *,
    salt: str,
    stage_evidence: dict | None = None,
    sec_payloads: tuple[bytes, ...] | None = None,
) -> dict:
    claim, sec_reader = _complete_fixed_sec_ancestry(
        store,
        request["request_sha256"],
        payloads=sec_payloads,
    )
    tip = store.load_current_tip_anchor()
    grant = tip["authorization_bundles"][request["request_sha256"]][
        "authorization_grant"
    ]
    evidence = (
        _owned_v3_stage_evidence(candidate, grant, salt=salt)
        if stage_evidence is None
        else copy.deepcopy(stage_evidence)
    )
    evidence_path, marker_path = _write_fixed_stage_evidence_component(
        store,
        claim,
        sec_reader,
        evidence,
    )
    return {
        "stage_evidence": evidence,
        "claim": claim,
        "sec_reader_receipt": sec_reader,
        "evidence_path": evidence_path,
        "marker_path": marker_path,
    }


def _sec_plan_document(accession_number: str, filename: str) -> dict:
    return {
        "accession_number": accession_number,
        "official_url": (
            "https://www.sec.gov/Archives/edgar/data/320193/"
            f"{accession_number.replace('-', '')}/{filename}"
        ),
    }


def _complete_test_carry_in_universe(*, salt: str) -> dict:
    records: list[dict] = []
    serial = 1
    for year in range(2000, 2026):
        for form, month in (
            ("10-K", 2),
            ("10-Q", 5),
            ("10-Q", 8),
            ("10-Q", 11),
        ):
            evidence_date = date(year, month, 15)
            records.append(
                {
                    "accession_number": (
                        f"0000320193-{year % 100:02d}-{serial:06d}"
                    ),
                    "subject_cik": "0000320193",
                    "form": form,
                    "acceptance_datetime": (
                        evidence_date.strftime("%Y%m%d") + "160000"
                    ),
                    "filing_date": evidence_date.isoformat(),
                    "filing_date_change": None,
                    "primary_document": f"filing-{serial}.htm",
                    "source_record_sha256": _digest(
                        f"{salt}:universe:{serial}"
                    ),
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
                "source_record_sha256": _digest(
                    f"{salt}:universe:{serial}"
                ),
            }
        )
        serial += 1
    return build_corpus_universe_manifest(
        catalog_artifact_sha256=_digest(f"{salt}:catalog"),
        calendar_artifact_sha256=_digest(f"{salt}:calendar"),
        catalog_total_record_count=1_000,
        catalog_eligible_record_count=len(records),
        session_dates=EXPECTED_SESSIONS,
        records=records,
    )


def _prepare_final_owned_carry_in_chain(
    store: SecFilingGemmaRevealStore,
    *,
    salt: str,
    tamper_carry_scope: bool = False,
) -> dict:
    store.initialize()
    registered, candidate = _register(store, salt=salt)
    universe = _complete_test_carry_in_universe(salt=salt)
    final_universe_records = sorted(
        (
            record
            for record in universe["records"]
            if record["artifact_stage"] == "final"
        ),
        key=lambda record: (
            record["availability_session"],
            record["accession_number"],
        ),
    )
    selected_parent_records = []
    for form in ("10-K", "10-Q"):
        first_final = next(
            record for record in final_universe_records if record["form"] == form
        )
        selected_parent_records.append(
            max(
                (
                    record
                    for record in universe["records"]
                    if record["form"] == form
                    and record["artifact_stage"] == "intermediate"
                    and (
                        record["availability_session"],
                        record["accession_number"],
                    )
                    < (
                        first_final["availability_session"],
                        first_final["accession_number"],
                    )
                ),
                key=lambda record: (
                    record["availability_session"],
                    record["accession_number"],
                ),
            )
        )
    selected_parent_records.sort(key=lambda record: record["accession_number"])
    parent_documents = [
        _sec_plan_document(
            record["accession_number"],
            record["primary_document"],
        )
        for record in selected_parent_records
    ]
    final_documents = [
        _sec_plan_document("0000320193-25-000201", "aapl-20250927.htm"),
        _sec_plan_document("0000320193-26-000202", "aapl-20251227.htm"),
    ]
    parent_payloads = (
        b"<html><body><p>Owned intermediate annual carry in.</p></body></html>",
        b"<html><body><p>Owned intermediate quarterly carry in.</p></body></html>",
    )
    development_evidence = _evidence(
        "development", candidate, salt=f"{salt}-development"
    )
    parent_request, parent_access = _grant_request(
        registered,
        candidate,
        stage="intermediate",
        evidence=development_evidence,
        include_sec_plan=True,
        sec_documents=parent_documents,
    )
    parent_bundle = _consume_with_grant(
        store,
        parent_request,
        candidate,
        development_evidence,
        stage="intermediate",
        access_manifest=parent_access,
    )
    parent_claim_result = store.claim_authorized_sec_stage_execution(
        request_sha256=parent_request["request_sha256"],
        sec_user_agent_sha256=SEC_TEST_USER_AGENT_SHA256,
    )
    parent_claim = parent_claim_result["claim"]
    _write_fixed_sec_component(
        store,
        parent_claim,
        payloads=parent_payloads,
    )
    parent_reader = store._record_authorized_sec_stage_reader_output(
        request_sha256=parent_request["request_sha256"],
    )
    parent_sec_directory = (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / parent_claim["claim_sha256"]
        / SEC_STAGE_COMPONENT_DIRECTORY_NAME
    )
    normalized_payloads = tuple(
        (
            parent_sec_directory
            / f"document-{ordinal:04d}.normalized.txt"
        ).read_bytes()
        for ordinal in range(1, len(parent_documents) + 1)
    )
    normalized_by_accession = {
        document["accession_number"]: payload
        for document, payload in zip(parent_documents, normalized_payloads)
    }
    content_documents = [
        {
            "accession_number": record["accession_number"],
            "primary_document_sha256": _digest(
                f"{salt}:{record['accession_number']}:primary"
            ),
            "normalized_text_sha256": (
                hashlib.sha256(
                    normalized_by_accession[record["accession_number"]]
                ).hexdigest()
                if record["accession_number"] in normalized_by_accession
                else _digest(
                    f"{salt}:{record['accession_number']}:normalized"
                )
            ),
            "primary_document_bytes": 2_000,
            "normalized_text_bytes": (
                len(normalized_by_accession[record["accession_number"]])
                if record["accession_number"] in normalized_by_accession
                else 1_000
            ),
        }
        for record in universe["records"]
        if record["artifact_stage"] == "intermediate"
    ]
    parent_content_manifest = build_stage_content_manifest(
        artifact_stage="intermediate",
        corpus_universe_sha256=universe["universe_sha256"],
        documents=content_documents,
        universe_manifest=universe,
    )
    parent_tip = store.load_current_tip_anchor()
    parent_grant = parent_tip["authorization_bundles"][
        parent_request["request_sha256"]
    ]["authorization_grant"]
    parent_evidence = _owned_v3_stage_evidence(
        candidate,
        parent_grant,
        salt=f"{salt}-parent-output",
    )
    parent_evidence["corpus_universe_manifest"] = universe
    parent_evidence["prerequisite_content_manifest"] = (
        parent_content_manifest
    )
    parent_evidence["stage_evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in parent_evidence.items()
            if key != "stage_evidence_sha256"
        }
    )
    parent_evidence_path, parent_marker_path = (
        _write_fixed_stage_evidence_component(
            store,
            parent_claim,
            parent_reader,
            parent_evidence,
        )
    )
    parent_output_receipt = store._record_owned_stage_evidence_output(
        request_sha256=parent_request["request_sha256"],
    )
    carry_records = _prior_same_form_carry_ins(
        parent_evidence["corpus_universe_manifest"],
        parent_content_manifest,
        prerequisite_stage="intermediate",
        requested_stage="final",
        prerequisite_content_manifest_sha256=parent_content_manifest[
            "content_manifest_sha256"
        ],
    )
    carry_scope = {
        "selection_policy": (
            "latest_prerequisite_stage_filing_of_each_requested_stage_form"
        ),
        "artifact_scope": "sealed_normalized_text_only",
        "network_refetch_permitted": False,
        "write_permitted": False,
        "bound_by_prerequisite_stage_evidence_sha256": parent_evidence[
            "stage_evidence_sha256"
        ],
        "prerequisite_content_manifest_sha256": parent_content_manifest[
            "content_manifest_sha256"
        ],
        "record_count": len(carry_records),
        "records_sha256": canonical_sha256(carry_records),
        "records": carry_records,
    }
    if tamper_carry_scope:
        carry_scope["records"] = copy.deepcopy(carry_records)
        carry_scope["records"][0]["normalized_text_sha256"] = _digest(
            f"{salt}:cross-scope-normalized-text"
        )
        carry_scope["records_sha256"] = canonical_sha256(
            carry_scope["records"]
        )
    final_request, final_access = _grant_request(
        parent_bundle["authenticated_store_snapshot"],
        candidate,
        stage="final",
        evidence=parent_evidence,
        include_sec_plan=True,
        sec_documents=final_documents,
        prior_same_form_carry_in=carry_scope,
        prerequisite_evidence_pin={
            "stage": "intermediate",
            "content_manifest_sha256": parent_content_manifest[
                "content_manifest_sha256"
            ],
            "stage_artifact_sha256": _digest(
                f"{salt}:intermediate:stage-artifact"
            ),
            "external_seal_receipt_sha256": _digest(
                f"{salt}:intermediate:external-seal"
            ),
        },
    )
    final_bundle = _consume_with_grant(
        store,
        final_request,
        candidate,
        parent_evidence,
        stage="final",
        access_manifest=final_access,
    )
    child_claim, child_reader = _complete_fixed_sec_ancestry(
        store,
        final_request["request_sha256"],
        payloads=(
            b"<html><body><p>Owned final annual filing.</p></body></html>",
            b"<html><body><p>Owned final quarterly filing.</p></body></html>",
        ),
    )
    return {
        "candidate": candidate,
        "parent_request": parent_request,
        "parent_bundle": parent_bundle,
        "parent_claim": parent_claim,
        "parent_reader": parent_reader,
        "parent_output_receipt": parent_output_receipt,
        "parent_evidence": parent_evidence,
        "parent_evidence_path": parent_evidence_path,
        "parent_marker_path": parent_marker_path,
        "parent_normalized_payloads": normalized_payloads,
        "parent_sec_directory": parent_sec_directory,
        "final_request": final_request,
        "final_access": final_access,
        "final_bundle": final_bundle,
        "child_claim": child_claim,
        "child_reader": child_reader,
        "carry_records": carry_records,
    }


def test_sec_execution_claim_is_current_tip_only_and_idempotently_reports_created(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _candidate_value, request, _bundle = _issued_intermediate_grant(
        store,
        salt="sec-claim-idempotent",
        include_sec_plan=True,
    )
    state_bytes = store.state_path.read_bytes()
    tip_before = store.load_current_tip_anchor()

    first = store.claim_authorized_sec_stage_execution(
        request_sha256=request["request_sha256"],
        sec_user_agent_sha256=SEC_TEST_USER_AGENT_SHA256,
    )

    tip_after = store.load_current_tip_anchor()
    assert first["created"] is True
    assert first["reader_receipt"] is None
    assert first["abort"] is None
    assert store.state_path.read_bytes() == state_bytes
    assert tip_after["revision"] == tip_before["revision"] + 1
    assert tip_after["previous_tip_anchor_sha256"] == tip_before[
        "tip_anchor_sha256"
    ]
    assert tip_after["state_sha256"] == tip_before["state_sha256"]
    assert tip_after["stage_sec_execution_claims"] == {
        request["request_sha256"]: first["claim"]
    }
    assert first["claim"]["start_current_tip_anchor_sha256"] == tip_before[
        "tip_anchor_sha256"
    ]
    assert first["claim"]["sec_user_agent_sha256"] == SEC_TEST_USER_AGENT_SHA256
    stable_tip_bytes = store.current_tip_anchor_path.read_bytes()

    repeated = store.claim_authorized_sec_stage_execution(
        request_sha256=request["request_sha256"],
        sec_user_agent_sha256=SEC_TEST_USER_AGENT_SHA256,
    )

    assert repeated == {**first, "created": False}
    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == stable_tip_bytes

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="contact differs",
    ):
        store.claim_authorized_sec_stage_execution(
            request_sha256=request["request_sha256"],
            sec_user_agent_sha256=content_sha256(b"another private contact"),
        )
    assert store.current_tip_anchor_path.read_bytes() == stable_tip_bytes


@pytest.mark.parametrize("attack", ("delete", "byte_change", "marker_change"))
def test_completed_sec_reader_receipt_requires_unchanged_durable_batch(
    tmp_path: Path,
    attack: str,
) -> None:
    store = _store(tmp_path)
    _candidate_value, request, _bundle, claim = _issued_sec_claim(
        store,
        salt=f"completed-sec-reader-{attack}",
    )
    component_directory, marker_path, byte_index = _write_fixed_sec_component(
        store,
        claim,
    )
    receipt = store._record_authorized_sec_stage_reader_output(
        request_sha256=request["request_sha256"],
    )
    stable_state_bytes = store.state_path.read_bytes()
    stable_tip_bytes = store.current_tip_anchor_path.read_bytes()
    payload_path = component_directory / byte_index[0]["relative_path"]

    if attack == "delete":
        payload_path.unlink()
    elif attack == "byte_change":
        payload_path.write_bytes(b"changed after receipt")
    else:
        marker_path.write_bytes(marker_path.read_bytes() + b" ")

    with pytest.raises(SecFilingGemmaRevealStoreError):
        store._record_authorized_sec_stage_reader_output(
            request_sha256=request["request_sha256"],
        )

    assert receipt == store.load_current_tip_anchor()["stage_sec_reader_receipts"][
        request["request_sha256"]
    ]
    assert store.state_path.read_bytes() == stable_state_bytes
    assert store.current_tip_anchor_path.read_bytes() == stable_tip_bytes


@pytest.mark.parametrize(
    "substituted_source_path",
    (
        "agent_benchmark/sec_filing_gemma_stage_runner.py",
        "agent_benchmark/sec_audit_transport.py",
        "agent_benchmark/sec_filing_content.py",
        "agent_benchmark/sec_filing_gemma_stage_authorization.py",
    ),
)
def test_sec_execution_claim_rejects_substituted_source_bytes_without_tip_mutation(
    tmp_path: Path,
    substituted_source_path: str,
) -> None:
    store = _store(tmp_path)
    _candidate_value, request, _bundle = _issued_intermediate_grant(
        store,
        salt="sec-claim-source-substitution",
        include_sec_plan=True,
    )
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()
    real_read = reveal_store_module._read_regular_bytes

    def substituted_read(
        path: Path,
        location: str,
        *,
        max_bytes: int = reveal_store_module.MAX_TRACKED_ANCHOR_FILE_BYTES,
    ) -> bytes:
        payload = real_read(path, location, max_bytes=max_bytes)
        if path.as_posix().endswith(substituted_source_path):
            return payload + b"\n# substituted after candidate registration\n"
        return payload

    with patch.object(
        reveal_store_module,
        "_read_regular_bytes",
        side_effect=substituted_read,
    ):
        with pytest.raises(
            SecFilingGemmaRevealStoreError,
            match="Could not claim the exact current SEC stage grant",
        ):
            store.claim_authorized_sec_stage_execution(
                request_sha256=request["request_sha256"],
                sec_user_agent_sha256=SEC_TEST_USER_AGENT_SHA256,
            )

    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes
    assert store.load_current_tip_anchor()["stage_sec_execution_claims"] == {}


def test_sec_reader_finalizer_is_not_a_public_store_api() -> None:
    assert not hasattr(
        SecFilingGemmaRevealStore,
        "record_authorized_sec_stage_reader_output",
    )
    assert hasattr(
        SecFilingGemmaRevealStore,
        "_record_authorized_sec_stage_reader_output",
    )


def test_sec_reader_receipt_rehashes_fixed_bytes_and_exact_retry_is_noop(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _candidate_value, request, _bundle, claim = _issued_sec_claim(
        store,
        salt="sec-reader-rehash-success",
    )
    payloads = (
        b"<html><body><p>First and second fixed SEC filing.</p></body></html>",
    )
    _directory, marker_path, expected_index = _write_fixed_sec_component(
        store,
        claim,
        payloads=payloads,
    )
    state_bytes = store.state_path.read_bytes()
    tip_before = store.load_current_tip_anchor()

    receipt = store._record_authorized_sec_stage_reader_output(
        request_sha256=request["request_sha256"],
    )

    tip_after = store.load_current_tip_anchor()
    assert store.state_path.read_bytes() == state_bytes
    assert tip_after["revision"] == tip_before["revision"] + 1
    assert tip_after["stage_sec_reader_receipts"] == {
        request["request_sha256"]: receipt
    }
    assert receipt["claim_sha256"] == claim["claim_sha256"]
    assert receipt["byte_index"] == expected_index
    assert receipt["byte_index_sha256"] == canonical_sha256(expected_index)
    assert receipt["byte_count_total"] == sum(
        item["byte_count"] for item in expected_index
    )
    assert receipt["complete_marker_sha256"] == hashlib.sha256(
        marker_path.read_bytes()
    ).hexdigest()
    assert receipt["reader_output_recomputed_by_store"] is True
    assert receipt["fresh_network_provenance_claimed"] is False
    assert receipt["sec_user_agent_sha256"] == SEC_TEST_USER_AGENT_SHA256
    stable_tip_bytes = store.current_tip_anchor_path.read_bytes()

    repeated = store._record_authorized_sec_stage_reader_output(
        request_sha256=request["request_sha256"],
    )

    assert repeated == receipt
    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == stable_tip_bytes


@pytest.mark.parametrize(
    "attack",
    (
        "byte_flip",
        "semantic_rehash",
        "manifest_extra_field",
        "manifest_bool_count",
        "extra",
        "missing",
        "unsafe_path",
        "rehashed_marker",
        "symlink",
    ),
)
def test_sec_reader_rejects_directory_substitution_without_tip_mutation(
    tmp_path: Path,
    attack: str,
) -> None:
    store = _store(tmp_path)
    _candidate_value, request, _bundle, claim = _issued_sec_claim(
        store,
        salt=f"sec-reader-substitution-{attack}",
    )
    component_directory, marker_path, byte_index = _write_fixed_sec_component(
        store,
        claim,
    )
    payload_path = component_directory / byte_index[0]["relative_path"]
    if attack == "byte_flip":
        payload_path.write_bytes(b"fixed SEC filing byteS")
    elif attack == "semantic_rehash":
        substituted = b"<html><body><p>Forged but rehashed filing.</p></body></html>"
        payload_path.write_bytes(substituted)
        marker = json.loads(marker_path.read_bytes())
        marker["byte_index"][0]["byte_count"] = len(substituted)
        marker["byte_index"][0]["sha256"] = hashlib.sha256(substituted).hexdigest()
        _rewrite_complete_marker(marker_path, marker)
    elif attack in {"manifest_extra_field", "manifest_bool_count"}:
        manifest_path = component_directory / "byte-manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        if attack == "manifest_extra_field":
            manifest["forged_provenance"] = "self-issued"
        else:
            manifest["document_count"] = True
        manifest_body = {
            key: value
            for key, value in manifest.items()
            if key != "byte_manifest_sha256"
        }
        manifest["byte_manifest_sha256"] = canonical_sha256(manifest_body)
        manifest_bytes = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        marker = json.loads(marker_path.read_bytes())
        manifest_item = next(
            item
            for item in marker["byte_index"]
            if item["relative_path"] == "byte-manifest.json"
        )
        manifest_item["byte_count"] = len(manifest_bytes)
        manifest_item["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        _rewrite_complete_marker(marker_path, marker)
    elif attack == "extra":
        (component_directory / "unlisted.htm").write_bytes(b"unlisted")
    elif attack == "missing":
        payload_path.unlink()
    elif attack == "unsafe_path":
        marker = json.loads(marker_path.read_bytes())
        marker["byte_index"][0]["relative_path"] = "../outside.htm"
        _rewrite_complete_marker(marker_path, marker)
    elif attack == "rehashed_marker":
        marker = json.loads(marker_path.read_bytes())
        marker["request_sha256"] = _digest("substituted marker request")
        _rewrite_complete_marker(marker_path, marker)
    else:
        original_payload = payload_path.read_bytes()
        payload_path.unlink()
        external_target = tmp_path / "external-sec-payload.htm"
        external_target.write_bytes(original_payload)
        try:
            payload_path.symlink_to(external_target)
        except (OSError, NotImplementedError):
            pytest.skip("File symlinks are unavailable for this test user")
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()
    tip_before = store.load_current_tip_anchor()

    with pytest.raises(SecFilingGemmaRevealStoreError):
        store._record_authorized_sec_stage_reader_output(
            request_sha256=request["request_sha256"],
        )

    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes
    assert store.load_current_tip_anchor() == tip_before
    assert request["request_sha256"] not in tip_before[
        "stage_sec_reader_receipts"
    ]


def test_sec_execution_abort_is_terminal_idempotent_and_forbids_retry(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _candidate_value, request, _bundle, claim = _issued_sec_claim(
        store,
        salt="sec-execution-abort",
    )
    state_bytes = store.state_path.read_bytes()
    tip_before = store.load_current_tip_anchor()

    abort = store.abort_authorized_sec_stage_execution(
        request_sha256=request["request_sha256"],
        reason="external_effect_failed_or_completion_unknown",
    )

    tip_after = store.load_current_tip_anchor()
    assert store.state_path.read_bytes() == state_bytes
    assert tip_after["revision"] == tip_before["revision"] + 1
    assert tip_after["stage_sec_execution_aborts"] == {
        request["request_sha256"]: abort
    }
    assert abort["claim_sha256"] == claim["claim_sha256"]
    assert abort["external_effect_retry_permitted"] is False
    stable_tip_bytes = store.current_tip_anchor_path.read_bytes()

    repeated_abort = store.abort_authorized_sec_stage_execution(
        request_sha256=request["request_sha256"],
        reason="external_effect_failed_or_completion_unknown",
    )
    recovered_claim = store.claim_authorized_sec_stage_execution(
        request_sha256=request["request_sha256"],
        sec_user_agent_sha256=SEC_TEST_USER_AGENT_SHA256,
    )

    assert repeated_abort == abort
    assert recovered_claim["created"] is False
    assert recovered_claim["claim"] == claim
    assert recovered_claim["abort"] == abort
    assert recovered_claim["reader_receipt"] is None
    assert store.current_tip_anchor_path.read_bytes() == stable_tip_bytes

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="another terminal abort",
    ):
        store.abort_authorized_sec_stage_execution(
            request_sha256=request["request_sha256"],
            reason="durable_output_verification_failed",
        )
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="Aborted SEC execution cannot publish",
    ):
        store._record_authorized_sec_stage_reader_output(
            request_sha256=request["request_sha256"],
        )
    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == stable_tip_bytes


def test_active_sec_claim_blocks_registry_consumption_output_and_other_claim(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _candidate_value, intermediate_request, _intermediate_bundle = (
        _issued_intermediate_grant(
            store,
            salt="active-sec-consumption",
            include_sec_plan=True,
        )
    )
    claimed = store.claim_authorized_sec_stage_execution(
        request_sha256=intermediate_request["request_sha256"],
        sec_user_agent_sha256=SEC_TEST_USER_AGENT_SHA256,
    )
    assert claimed["created"] is True
    stable_state_bytes = store.state_path.read_bytes()
    stable_tip_bytes = store.current_tip_anchor_path.read_bytes()

    with pytest.raises(SecFilingGemmaRevealStoreError):
        _register(store, salt="active-sec-blocked-registry")
    assert store.state_path.read_bytes() == stable_state_bytes
    assert store.current_tip_anchor_path.read_bytes() == stable_tip_bytes

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="lacks its current grant bundle",
    ):
        store.claim_authorized_sec_stage_execution(
            request_sha256=_digest("cross-request claim"),
            sec_user_agent_sha256=SEC_TEST_USER_AGENT_SHA256,
        )
    assert store.state_path.read_bytes() == stable_state_bytes
    assert store.current_tip_anchor_path.read_bytes() == stable_tip_bytes

    output_store = _store(tmp_path / "output-block")
    output_candidate, output_request, _output_bundle = (
        _issued_intermediate_grant(
            output_store,
            salt="active-sec-output",
            include_sec_plan=True,
        )
    )
    output_store.claim_authorized_sec_stage_execution(
        request_sha256=output_request["request_sha256"],
        sec_user_agent_sha256=SEC_TEST_USER_AGENT_SHA256,
    )
    output_state_bytes = output_store.state_path.read_bytes()
    output_tip_bytes = output_store.current_tip_anchor_path.read_bytes()
    with pytest.raises(SecFilingGemmaRevealStoreError):
        output_store._record_owned_stage_evidence_output(
            request_sha256=output_request["request_sha256"],
        )
    assert output_store.state_path.read_bytes() == output_state_bytes
    assert output_store.current_tip_anchor_path.read_bytes() == output_tip_bytes


@pytest.mark.parametrize("artifact", ("claim", "reader_receipt", "abort"))
@pytest.mark.parametrize("crash_point", ("pending_tip", "state_replace"))
def test_sec_tip_only_artifact_crash_recovers_once(
    tmp_path: Path,
    artifact: str,
    crash_point: str,
) -> None:
    store = _store(tmp_path)
    _candidate_value, request, _bundle = _issued_intermediate_grant(
        store,
        salt=f"sec-crash-{artifact}-{crash_point}",
        include_sec_plan=True,
    )
    if artifact != "claim":
        claim_result = store.claim_authorized_sec_stage_execution(
            request_sha256=request["request_sha256"],
            sec_user_agent_sha256=SEC_TEST_USER_AGENT_SHA256,
        )
        if artifact == "reader_receipt":
            _write_fixed_sec_component(store, claim_result["claim"])

    def operation():
        if artifact == "claim":
            return store.claim_authorized_sec_stage_execution(
                request_sha256=request["request_sha256"],
                sec_user_agent_sha256=SEC_TEST_USER_AGENT_SHA256,
            )
        if artifact == "reader_receipt":
            return store._record_authorized_sec_stage_reader_output(
                request_sha256=request["request_sha256"],
            )
        return store.abort_authorized_sec_stage_execution(
            request_sha256=request["request_sha256"],
            reason="external_effect_failed_or_completion_unknown",
        )

    state_bytes = store.state_path.read_bytes()
    tip_before = store.load_current_tip_anchor()
    real_atomic_replace = reveal_store_module._atomic_replace
    crashed = False

    def crash_during_tip_only_cas(path: Path, payload: bytes) -> None:
        nonlocal crashed
        real_atomic_replace(path, payload)
        parsed = json.loads(payload)
        is_target = (
            crash_point == "pending_tip"
            and path == store.current_tip_anchor_path
            and parsed.get("schema_version") == CURRENT_TIP_PENDING_SCHEMA_VERSION
        ) or (
            crash_point == "state_replace"
            and path == store.state_path
        )
        if is_target and not crashed:
            crashed = True
            raise RuntimeError(
                f"simulated {artifact} crash at {crash_point}"
            )

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store._atomic_replace",
        crash_during_tip_only_cas,
    ), pytest.raises(RuntimeError, match=f"simulated {artifact} crash"):
        operation()

    pending = json.loads(store.current_tip_anchor_path.read_bytes())
    assert pending["schema_version"] == CURRENT_TIP_PENDING_SCHEMA_VERSION
    map_name = {
        "claim": "stage_sec_execution_claims",
        "reader_receipt": "stage_sec_reader_receipts",
        "abort": "stage_sec_execution_aborts",
    }[artifact]
    expected = pending["next_tip_anchor"][map_name][request["request_sha256"]]

    recovered = operation()

    if artifact == "claim":
        assert recovered["created"] is False
        assert recovered["claim"] == expected
    else:
        assert recovered == expected
    assert store.state_path.read_bytes() == state_bytes
    recovered_tip = store.load_current_tip_anchor()
    assert recovered_tip["revision"] == tip_before["revision"] + 1
    assert recovered_tip[map_name][request["request_sha256"]] == expected
    stable_tip_bytes = store.current_tip_anchor_path.read_bytes()
    repeated = operation()
    if artifact == "claim":
        assert repeated["created"] is False
        assert repeated["claim"] == expected
    else:
        assert repeated == expected
    assert store.current_tip_anchor_path.read_bytes() == stable_tip_bytes


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


def test_first_consumed_stage_output_is_persisted_once_and_exact_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate, request, _bundle = _issued_intermediate_grant(
        store,
        salt="first-stage-output",
        include_sec_plan=True,
    )
    prepared = _prepare_fixed_stage_evidence_output(
        store,
        request,
        candidate,
        salt="first-stage-output-evidence",
    )
    stage_evidence = prepared["stage_evidence"]
    state_bytes_before = store.state_path.read_bytes()
    tip_before = store.load_current_tip_anchor()

    receipt = store._record_owned_stage_evidence_output(
        request_sha256=request["request_sha256"],
    )

    tip_after_first = store.load_current_tip_anchor()
    assert store.state_path.read_bytes() == state_bytes_before
    assert tip_after_first["revision"] == tip_before["revision"] + 1
    assert tip_after_first["previous_tip_anchor_sha256"] == tip_before[
        "tip_anchor_sha256"
    ]
    assert tip_after_first["consumed_stage_output_receipts"] == {
        request["request_sha256"]: receipt
    }
    assert receipt["request_sha256"] == request["request_sha256"]
    assert receipt["output_stage"] == "intermediate"
    assert receipt["output_stage_evidence_sha256"] == stage_evidence[
        "stage_evidence_sha256"
    ]
    assert receipt["output_stage_evidence_document_sha256"] == hashlib.sha256(
        prepared["evidence_path"].read_bytes()
    ).hexdigest()
    assert receipt["output_stage_evidence_complete_marker_sha256"] == hashlib.sha256(
        prepared["marker_path"].read_bytes()
    ).hexdigest()
    assert receipt["sec_execution_claim_sha256"] == prepared["claim"][
        "claim_sha256"
    ]
    assert receipt["sec_reader_receipt_sha256"] == prepared[
        "sec_reader_receipt"
    ]["receipt_sha256"]
    assert receipt["output_stage_evidence_recomputed_by_store"] is True
    assert receipt["fresh_stage_evidence_provenance_claimed"] is False
    tip_bytes_after_first = store.current_tip_anchor_path.read_bytes()

    repeated = store._record_owned_stage_evidence_output(
        request_sha256=request["request_sha256"],
    )

    assert repeated == receipt
    assert store.state_path.read_bytes() == state_bytes_before
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes_after_first
    assert store.load_current_tip_anchor()["revision"] == tip_after_first["revision"]


def test_owned_stage_output_finalizer_has_no_public_caller_mapping_api() -> None:
    assert not hasattr(
        SecFilingGemmaRevealStore,
        "record_consumed_stage_output_evidence",
    )
    signature = inspect.signature(
        SecFilingGemmaRevealStore._record_owned_stage_evidence_output
    )
    assert tuple(signature.parameters) == ("self", "request_sha256")
    assert signature.parameters["request_sha256"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )


def test_owned_final_carry_in_reader_is_request_only_durable_and_idempotent(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    prepared = _prepare_final_owned_carry_in_chain(
        store,
        salt="owned-final-carry-in-happy",
    )
    request_hash = prepared["final_request"]["request_sha256"]
    state_bytes = store.state_path.read_bytes()
    tip_before = store.load_current_tip_anchor()

    receipt = store._record_owned_stage_carry_in_reader_output(
        request_sha256=request_hash,
    )

    tip_after = store.load_current_tip_anchor()
    assert store.state_path.read_bytes() == state_bytes
    assert tip_after["revision"] == tip_before["revision"] + 1
    assert tip_after["stage_carry_in_reader_receipts"] == {
        request_hash: receipt
    }
    assert receipt["request_sha256"] == request_hash
    assert receipt["parent_request_sha256"] == prepared["parent_request"][
        "request_sha256"
    ]
    assert receipt["carry_in_records"] == prepared["carry_records"]
    assert receipt["carry_in_byte_count_total"] == sum(
        len(payload) for payload in prepared["parent_normalized_payloads"]
    )
    component_directory = (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / prepared["child_claim"]["claim_sha256"]
        / reveal_store_module.PRIOR_SAME_FORM_CARRY_IN_COMPONENT_DIRECTORY_NAME
    )
    assert sorted(path.name for path in component_directory.iterdir()) == [
        "carry-in-0001.normalized.txt",
        "carry-in-0002.normalized.txt",
        reveal_store_module.PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_FILENAME,
    ]
    assert tuple(
        (component_directory / f"carry-in-{ordinal:04d}.normalized.txt").read_bytes()
        for ordinal in range(1, 3)
    ) == prepared["parent_normalized_payloads"]
    tip_bytes_after = store.current_tip_anchor_path.read_bytes()

    repeated = store._record_owned_stage_carry_in_reader_output(
        request_sha256=request_hash,
    )

    assert repeated == receipt
    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes_after
    public_names = {
        name
        for name, _member in inspect.getmembers(
            SecFilingGemmaRevealStore,
            predicate=inspect.isfunction,
        )
        if not name.startswith("_")
    }
    assert "record_owned_stage_carry_in_reader_output" not in public_names
    signature = inspect.signature(
        SecFilingGemmaRevealStore._record_owned_stage_carry_in_reader_output
    )
    assert tuple(signature.parameters) == ("self", "request_sha256")
    assert signature.parameters["request_sha256"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )


def test_owned_carry_in_reader_rejects_intermediate_before_reader_mutation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _candidate_value, request, _bundle, claim = _issued_sec_claim(
        store,
        salt="owned-carry-in-invalid-intermediate",
    )
    _write_fixed_sec_component(store, claim)
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="requires a consumed final child",
    ):
        store._record_owned_stage_carry_in_reader_output(
            request_sha256=request["request_sha256"],
        )

    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes
    assert request["request_sha256"] not in store.load_current_tip_anchor()[
        "stage_sec_reader_receipts"
    ]


@pytest.mark.parametrize("mutated_artifact", ("child_marker", "parent_raw"))
def test_owned_final_carry_in_replays_full_sec_batches_under_final_lock(
    tmp_path: Path,
    mutated_artifact: str,
) -> None:
    store = _store(tmp_path)
    prepared = _prepare_final_owned_carry_in_chain(
        store,
        salt=f"owned-final-carry-in-sec-closure-{mutated_artifact}",
    )
    request_hash = prepared["final_request"]["request_sha256"]
    child_sec_directory = (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / prepared["child_claim"]["claim_sha256"]
        / SEC_STAGE_COMPONENT_DIRECTORY_NAME
    )
    original_replayer = store._record_authorized_sec_stage_reader_output
    call_count = 0

    def replay_then_mutate(*, request_sha256: str):
        nonlocal call_count
        receipt = original_replayer(request_sha256=request_sha256)
        call_count += 1
        if call_count == 2:
            if mutated_artifact == "child_marker":
                path = child_sec_directory / SEC_BATCH_COMPLETE_MARKER_FILENAME
                path.write_bytes(path.read_bytes() + b" ")
            else:
                path = prepared["parent_sec_directory"] / "document-0001.raw"
                path.write_bytes(b"changed parent raw bytes after outside replay")
        return receipt

    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()
    with patch.object(
        store,
        "_record_authorized_sec_stage_reader_output",
        side_effect=replay_then_mutate,
    ), pytest.raises(SecFilingGemmaRevealStoreError):
        store._record_owned_stage_carry_in_reader_output(
            request_sha256=request_hash,
        )

    assert call_count == 2
    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes
    assert request_hash not in store.load_current_tip_anchor()[
        "stage_carry_in_reader_receipts"
    ]


def test_owned_final_carry_in_reader_retry_rejects_copied_byte_tamper(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    prepared = _prepare_final_owned_carry_in_chain(
        store,
        salt="owned-final-carry-in-tamper",
    )
    request_hash = prepared["final_request"]["request_sha256"]
    store._record_owned_stage_carry_in_reader_output(
        request_sha256=request_hash,
    )
    component_directory = (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / prepared["child_claim"]["claim_sha256"]
        / reveal_store_module.PRIOR_SAME_FORM_CARRY_IN_COMPONENT_DIRECTORY_NAME
    )
    (component_directory / "carry-in-0001.normalized.txt").write_bytes(
        b"coherently wrong copied carry-in bytes"
    )
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="differs from the exact owned replay",
    ):
        store._record_owned_stage_carry_in_reader_output(
            request_sha256=request_hash,
        )

    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes


def test_owned_final_carry_in_reader_recovers_partial_file_before_marker(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    prepared = _prepare_final_owned_carry_in_chain(
        store,
        salt="owned-final-carry-in-partial-recovery",
    )
    request_hash = prepared["final_request"]["request_sha256"]
    component_directory = (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / prepared["child_claim"]["claim_sha256"]
        / reveal_store_module.PRIOR_SAME_FORM_CARRY_IN_COMPONENT_DIRECTORY_NAME
    )
    component_directory.mkdir()
    partial_path = component_directory / "carry-in-0001.normalized.txt"
    partial_path.write_bytes(b"interrupted pre-marker bytes")
    tip_before = store.load_current_tip_anchor()

    receipt = store._record_owned_stage_carry_in_reader_output(
        request_sha256=request_hash,
    )

    assert partial_path.read_bytes() == prepared["parent_normalized_payloads"][0]
    assert (
        component_directory
        / reveal_store_module.PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_FILENAME
    ).is_file()
    tip_after = store.load_current_tip_anchor()
    assert tip_after["revision"] == tip_before["revision"] + 1
    assert tip_after["stage_carry_in_reader_receipts"][request_hash] == receipt


def test_owned_final_carry_in_reader_recovers_partial_marker_before_receipt(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    prepared = _prepare_final_owned_carry_in_chain(
        store,
        salt="owned-final-carry-in-partial-marker",
    )
    request_hash = prepared["final_request"]["request_sha256"]
    component_directory = (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / prepared["child_claim"]["claim_sha256"]
        / reveal_store_module.PRIOR_SAME_FORM_CARRY_IN_COMPONENT_DIRECTORY_NAME
    )
    component_directory.mkdir()
    marker_path = (
        component_directory
        / reveal_store_module.PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_FILENAME
    )
    marker_path.write_bytes(b'{"schema_version":')

    receipt = store._record_owned_stage_carry_in_reader_output(
        request_sha256=request_hash,
    )

    marker = json.loads(marker_path.read_bytes())
    assert marker["schema_version"] == (
        reveal_store_module.PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_SCHEMA_VERSION
    )
    assert hashlib.sha256(marker_path.read_bytes()).hexdigest() == receipt[
        "carry_in_complete_marker_sha256"
    ]


def test_owned_final_carry_in_does_not_repair_after_marker_before_receipt(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    prepared = _prepare_final_owned_carry_in_chain(
        store,
        salt="owned-final-carry-in-marker-before-receipt",
    )
    request_hash = prepared["final_request"]["request_sha256"]
    with patch.object(
        store,
        "_commit_state_and_tip_locked",
        side_effect=RuntimeError("simulated crash before carry receipt CAS"),
    ), pytest.raises(RuntimeError, match="simulated crash"):
        store._record_owned_stage_carry_in_reader_output(
            request_sha256=request_hash,
        )
    component_directory = (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / prepared["child_claim"]["claim_sha256"]
        / reveal_store_module.PRIOR_SAME_FORM_CARRY_IN_COMPONENT_DIRECTORY_NAME
    )
    assert (
        component_directory
        / reveal_store_module.PRIOR_SAME_FORM_CARRY_IN_COMPLETE_MARKER_FILENAME
    ).is_file()
    assert request_hash not in store.load_current_tip_anchor()[
        "stage_carry_in_reader_receipts"
    ]
    (component_directory / "carry-in-0001.normalized.txt").write_bytes(
        b"wrong bytes after committed marker before receipt"
    )
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="differs from the exact owned replay",
    ):
        store._record_owned_stage_carry_in_reader_output(
            request_sha256=request_hash,
        )

    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes


def test_owned_final_carry_in_replays_valid_marker_after_pre_receipt_crash(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    prepared = _prepare_final_owned_carry_in_chain(
        store,
        salt="owned-final-carry-in-valid-marker-retry",
    )
    request_hash = prepared["final_request"]["request_sha256"]
    tip_before = store.load_current_tip_anchor()
    with patch.object(
        store,
        "_commit_state_and_tip_locked",
        side_effect=RuntimeError("simulated crash before carry receipt CAS"),
    ), pytest.raises(RuntimeError, match="simulated crash"):
        store._record_owned_stage_carry_in_reader_output(
            request_sha256=request_hash,
        )
    tip_after_crash = store.load_current_tip_anchor()
    assert tip_after_crash == tip_before

    receipt = store._record_owned_stage_carry_in_reader_output(
        request_sha256=request_hash,
    )

    tip_after_retry = store.load_current_tip_anchor()
    assert tip_after_retry["revision"] == tip_before["revision"] + 1
    assert tip_after_retry["stage_carry_in_reader_receipts"][request_hash] == receipt


@pytest.mark.parametrize("attack", ("extra_file", "hardlink"))
def test_owned_final_carry_in_rejects_component_namespace_attacks(
    tmp_path: Path,
    attack: str,
) -> None:
    store = _store(tmp_path)
    prepared = _prepare_final_owned_carry_in_chain(
        store,
        salt=f"owned-final-carry-in-namespace-{attack}",
    )
    request_hash = prepared["final_request"]["request_sha256"]
    component_directory = (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / prepared["child_claim"]["claim_sha256"]
        / reveal_store_module.PRIOR_SAME_FORM_CARRY_IN_COMPONENT_DIRECTORY_NAME
    )
    component_directory.mkdir()
    if attack == "extra_file":
        (component_directory / "unexpected.txt").write_bytes(b"unexpected")
    else:
        external = tmp_path / "external-carry-in-hardlink-target.txt"
        external.write_bytes(b"linked bytes")
        os.link(
            external,
            component_directory / "carry-in-0001.normalized.txt",
        )
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()

    with pytest.raises(SecFilingGemmaRevealStoreError):
        store._record_owned_stage_carry_in_reader_output(
            request_sha256=request_hash,
        )

    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes
    assert request_hash not in store.load_current_tip_anchor()[
        "stage_carry_in_reader_receipts"
    ]


def test_owned_final_carry_in_reader_rejects_coherently_rehashed_cross_scope(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    prepared = _prepare_final_owned_carry_in_chain(
        store,
        salt="owned-final-carry-in-cross-scope",
        tamper_carry_scope=True,
    )
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="cannot be rederived from durable parent evidence",
    ):
        store._record_owned_stage_carry_in_reader_output(
            request_sha256=prepared["final_request"]["request_sha256"],
        )

    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes
    component_directory = (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / prepared["child_claim"]["claim_sha256"]
        / reveal_store_module.PRIOR_SAME_FORM_CARRY_IN_COMPONENT_DIRECTORY_NAME
    )
    assert not component_directory.exists()


@pytest.mark.parametrize(
    "attack",
    (
        "missing_evidence",
        "missing_marker",
        "extra_file",
        "duplicate_json",
        "noncanonical_json",
        "missing_v3_key",
        "extra_v3_key",
        "wrong_schema",
        "wrong_parent",
        "null_parent",
        "wrong_candidate",
        "wrong_stage",
        "bad_self_hash",
        "evidence_over_cap",
        "marker_request",
        "marker_claim",
        "marker_reader",
        "marker_component",
        "marker_path",
        "marker_count",
        "marker_document",
        "marker_evidence",
        "marker_missing_key",
        "marker_extra_key",
        "marker_wrong_schema",
        "marker_bad_self_hash",
        "marker_duplicate_json",
        "marker_noncanonical_json",
        "marker_over_cap",
        "marker_hardlink",
        "marker_symlink",
        "wrong_case_evidence",
        "case_collision",
        "directory_symlink",
        "hardlink",
        "symlink",
    ),
)
def test_owned_stage_output_rejects_durable_substitution_without_tip_mutation(
    tmp_path: Path,
    attack: str,
) -> None:
    store = _store(tmp_path)
    candidate, request, _bundle = _issued_intermediate_grant(
        store,
        salt=f"durable-stage-attack-{attack}",
        include_sec_plan=True,
    )
    prepared = _prepare_fixed_stage_evidence_output(
        store,
        request,
        candidate,
        salt=f"durable-stage-attack-{attack}",
    )
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()
    tip_before = store.load_current_tip_anchor()
    evidence_path = prepared["evidence_path"]
    marker_path = prepared["marker_path"]

    if attack == "missing_evidence":
        evidence_path.unlink()
    elif attack == "missing_marker":
        marker_path.unlink()
    elif attack == "extra_file":
        (evidence_path.parent / "unlisted.json").write_bytes(b"{}")
    elif attack == "duplicate_json":
        duplicate = (
            b'{"schema_version":"'
            + STAGE_EVIDENCE_SCHEMA_VERSION.encode("ascii")
            + b'",'
            + evidence_path.read_bytes()[1:]
        )
        evidence_path.write_bytes(duplicate)
    elif attack == "noncanonical_json":
        evidence_path.write_bytes(
            (json.dumps(prepared["stage_evidence"], indent=2) + "\n").encode(
                "utf-8"
            )
        )
    elif attack in {
        "missing_v3_key",
        "extra_v3_key",
        "wrong_schema",
        "wrong_parent",
        "null_parent",
        "wrong_candidate",
        "wrong_stage",
        "bad_self_hash",
    }:
        evidence = copy.deepcopy(prepared["stage_evidence"])
        if attack == "missing_v3_key":
            evidence.pop("learner_replays")
        elif attack == "extra_v3_key":
            evidence["forged_stage_field"] = {}
        elif attack == "wrong_schema":
            evidence["schema_version"] = "aapl-sec-gemma-stage-evidence-audit-v4"
        elif attack == "wrong_parent":
            evidence["parent_stage_evidence_sha256"] = _digest(
                "cross-grant parent evidence"
            )
        elif attack == "null_parent":
            evidence["parent_stage_evidence_sha256"] = None
        elif attack == "wrong_candidate":
            evidence["candidate_manifest"]["candidate_sha256"] = _digest(
                "cross-grant candidate"
            )
        elif attack == "wrong_stage":
            evidence["prerequisite_stage"] = "final"
        else:
            evidence["stage_evidence_sha256"] = _digest(
                "invalid evidence self hash"
            )
        _rewrite_fixed_stage_evidence_component(
            prepared,
            evidence,
            recompute_self_hash=attack != "bad_self_hash",
        )
    elif attack == "evidence_over_cap":
        pass
    elif attack in {
        "marker_missing_key",
        "marker_extra_key",
        "marker_wrong_schema",
        "marker_bad_self_hash",
        "marker_duplicate_json",
        "marker_noncanonical_json",
        "marker_over_cap",
        "marker_hardlink",
        "marker_symlink",
    }:
        marker = json.loads(marker_path.read_bytes())
        if attack == "marker_missing_key":
            marker.pop("relative_path")
        elif attack == "marker_extra_key":
            marker["forged_marker_field"] = True
        elif attack == "marker_wrong_schema":
            marker["schema_version"] = "aapl-sec-gemma-stage-evidence-complete-v2"
        elif attack == "marker_bad_self_hash":
            marker["marker_sha256"] = _digest("invalid marker self hash")
        elif attack == "marker_duplicate_json":
            marker_path.write_bytes(
                b'{"schema_version":"'
                + STAGE_EVIDENCE_COMPLETE_MARKER_SCHEMA_VERSION.encode("ascii")
                + b'",'
                + marker_path.read_bytes().lstrip()[1:]
            )
        elif attack == "marker_noncanonical_json":
            marker_path.write_bytes(
                (json.dumps(marker, indent=4, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            )
        elif attack == "marker_over_cap":
            marker_path.write_bytes(
                b" " * (reveal_store_module.MAX_TRACKED_ANCHOR_FILE_BYTES + 1)
            )
        else:
            original = marker_path.read_bytes()
            marker_path.unlink()
            external = tmp_path / f"external-{attack}.json"
            external.write_bytes(original)
            try:
                if attack == "marker_hardlink":
                    os.link(external, marker_path)
                else:
                    marker_path.symlink_to(external)
            except (OSError, NotImplementedError):
                pytest.skip(f"{attack} creation is unavailable for this test user")
        if attack in {
            "marker_missing_key",
            "marker_extra_key",
            "marker_wrong_schema",
        }:
            marker["marker_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in marker.items()
                    if key != "marker_sha256"
                }
            )
            marker_path.write_bytes(reveal_store_module._encoded_state(marker))
        elif attack == "marker_bad_self_hash":
            marker_path.write_bytes(reveal_store_module._encoded_state(marker))
    elif attack == "wrong_case_evidence":
        temporary = evidence_path.parent / "rename-temporary.json"
        evidence_path.rename(temporary)
        wrong_case = evidence_path.parent / "STAGE_EVIDENCE.JSON"
        temporary.rename(wrong_case)
        if wrong_case.name not in {item.name for item in wrong_case.parent.iterdir()}:
            pytest.skip("filesystem did not preserve the wrong-case filename")
    elif attack == "case_collision":
        collision = evidence_path.parent / "STAGE_EVIDENCE.JSON"
        collision.write_bytes(evidence_path.read_bytes())
        names = [item.name for item in evidence_path.parent.iterdir()]
        if len(names) == len({name.casefold() for name in names}):
            pytest.skip("case-colliding filenames are unavailable on this filesystem")
    elif attack == "directory_symlink":
        component_directory = evidence_path.parent
        external_directory = tmp_path / "external-stage-evidence-directory"
        component_directory.rename(external_directory)
        try:
            component_directory.symlink_to(
                external_directory,
                target_is_directory=True,
            )
        except (OSError, NotImplementedError):
            pytest.skip("directory symlinks are unavailable for this test user")
    elif attack in {
        "marker_request",
        "marker_claim",
        "marker_reader",
        "marker_component",
        "marker_path",
        "marker_count",
        "marker_document",
        "marker_evidence",
    }:
        marker = json.loads(marker_path.read_bytes())
        marker_field = {
            "marker_request": "request_sha256",
            "marker_claim": "claim_sha256",
            "marker_reader": "sec_reader_receipt_sha256",
            "marker_component": "component_id",
            "marker_path": "relative_path",
            "marker_count": "byte_count",
            "marker_document": "document_sha256",
            "marker_evidence": "stage_evidence_sha256",
        }[attack]
        marker[marker_field] = (
            marker["byte_count"] + 1
            if marker_field == "byte_count"
            else _digest(f"substituted {marker_field}")
        )
        marker["marker_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in marker.items()
                if key != "marker_sha256"
            }
        )
        marker_path.write_bytes(reveal_store_module._encoded_state(marker))
    else:
        original = evidence_path.read_bytes()
        evidence_path.unlink()
        external = tmp_path / f"external-{attack}.json"
        external.write_bytes(original)
        try:
            if attack == "hardlink":
                os.link(external, evidence_path)
            else:
                evidence_path.symlink_to(external)
        except (OSError, NotImplementedError):
            pytest.skip(f"{attack} creation is unavailable for this test user")

    evidence_cap = (
        len(evidence_path.read_bytes()) - 1
        if attack == "evidence_over_cap"
        else 1
    )
    cap_patch = (
        patch.object(
            reveal_store_module,
            "MAX_STAGE_EVIDENCE_FILE_BYTES",
            evidence_cap,
        )
        if attack == "evidence_over_cap"
        else None
    )
    try:
        if cap_patch is not None:
            cap_patch.start()
        with pytest.raises(SecFilingGemmaRevealStoreError):
            store._record_owned_stage_evidence_output(
                request_sha256=request["request_sha256"],
            )
    finally:
        if cap_patch is not None:
            cap_patch.stop()

    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes
    assert store.load_current_tip_anchor() == tip_before


@pytest.mark.parametrize("race_target", ("evidence", "marker", "directory"))
def test_owned_stage_output_detects_same_size_mutation_during_complete_replay(
    tmp_path: Path,
    race_target: str,
) -> None:
    store = _store(tmp_path)
    candidate, request, _bundle = _issued_intermediate_grant(
        store,
        salt=f"durable-stage-race-{race_target}",
        include_sec_plan=True,
    )
    prepared = _prepare_fixed_stage_evidence_output(
        store,
        request,
        candidate,
        salt=f"durable-stage-race-{race_target}",
    )
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()
    real_read = reveal_store_module._read_regular_bytes
    mutated = False

    def racing_read(
        path,
        location,
        *,
        max_bytes=reveal_store_module.MAX_TRACKED_ANCHOR_FILE_BYTES,
    ):
        nonlocal mutated
        payload = real_read(path, location, max_bytes=max_bytes)
        should_mutate = (
            race_target == "evidence"
            and location == "owned stage-evidence complete marker"
        ) or (
            race_target == "marker"
            and location == "owned stage-evidence document closure replay"
        ) or (
            race_target == "directory"
            and location == "owned stage-evidence complete marker"
        )
        if should_mutate and not mutated:
            if race_target == "directory":
                component_directory = prepared["evidence_path"].parent
                replaced_directory = tmp_path / "replaced-stage-evidence-directory"
                component_directory.rename(replaced_directory)
                shutil.copytree(replaced_directory, component_directory)
            else:
                target = (
                    prepared["evidence_path"]
                    if race_target == "evidence"
                    else prepared["marker_path"]
                )
                changed = bytearray(target.read_bytes())
                changed[len(changed) // 2] ^= 1
                target.write_bytes(bytes(changed))
            mutated = True
        return payload

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store._read_regular_bytes",
        racing_read,
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="changed during complete replay|directory changed",
    ):
        store._record_owned_stage_evidence_output(
            request_sha256=request["request_sha256"],
        )

    assert mutated is True
    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes


def test_shifted_post_closure_mutation_is_snapshot_only_and_blocks_retry(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate, request, _bundle = _issued_intermediate_grant(
        store,
        salt="durable-stage-shifted-race",
        include_sec_plan=True,
    )
    prepared = _prepare_fixed_stage_evidence_output(
        store,
        request,
        candidate,
        salt="durable-stage-shifted-race",
    )
    real_read = reveal_store_module._read_regular_bytes
    mutated = False

    def shifted_read(
        path,
        location,
        *,
        max_bytes=reveal_store_module.MAX_TRACKED_ANCHOR_FILE_BYTES,
    ):
        nonlocal mutated
        payload = real_read(path, location, max_bytes=max_bytes)
        if (
            location == "owned stage-evidence complete marker closure replay"
            and not mutated
        ):
            evidence_path = prepared["evidence_path"]
            before = evidence_path.read_bytes()
            after = before.replace(b"shifted-race", b"shifted-racf", 1)
            assert len(after) == len(before)
            assert after != before
            evidence_path.write_bytes(after)
            mutated = True
        return payload

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store._read_regular_bytes",
        shifted_read,
    ):
        receipt = store._record_owned_stage_evidence_output(
            request_sha256=request["request_sha256"],
        )

    assert mutated is True
    assert receipt["fresh_stage_evidence_provenance_claimed"] is False
    assert receipt["output_stage_evidence_document_sha256"] != hashlib.sha256(
        prepared["evidence_path"].read_bytes()
    ).hexdigest()
    stable_tip_bytes = store.current_tip_anchor_path.read_bytes()
    with pytest.raises(SecFilingGemmaRevealStoreError):
        store._record_owned_stage_evidence_output(
            request_sha256=request["request_sha256"],
        )
    assert store.current_tip_anchor_path.read_bytes() == stable_tip_bytes


def test_different_second_consumed_stage_output_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate, request, _bundle = _issued_intermediate_grant(
        store,
        salt="different-second-stage-output",
        include_sec_plan=True,
    )
    _prepare_fixed_stage_evidence_output(
        store,
        request,
        candidate,
        salt="different-second-stage-output-first",
    )
    first_receipt = store._record_owned_stage_evidence_output(
        request_sha256=request["request_sha256"],
    )
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()
    tip_before_rejection = store.load_current_tip_anchor()
    _prepare_fixed_stage_evidence_output(
        store,
        request,
        candidate,
        salt="different-second-stage-output-second",
    )

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="already has a different durable first output",
    ):
        store._record_owned_stage_evidence_output(
            request_sha256=request["request_sha256"],
        )

    assert store.state_path.read_bytes() == state_bytes
    assert store.current_tip_anchor_path.read_bytes() == tip_bytes
    tip_after_rejection = store.load_current_tip_anchor()
    assert tip_after_rejection == tip_before_rejection
    assert tip_after_rejection["consumed_stage_output_receipts"] == {
        request["request_sha256"]: first_receipt
    }


@pytest.mark.parametrize("tamper_kind", ("receipt", "map_key"))
def test_rehashed_consumed_stage_output_receipt_or_map_tamper_fails_load(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    store = _store(tmp_path)
    candidate, request, _bundle = _issued_intermediate_grant(
        store,
        salt=f"output-tamper-{tamper_kind}",
        include_sec_plan=True,
    )
    _prepare_fixed_stage_evidence_output(
        store,
        request,
        candidate,
        salt=f"output-tamper-{tamper_kind}-evidence",
    )
    store._record_owned_stage_evidence_output(
        request_sha256=request["request_sha256"],
    )
    tip = store.load_current_tip_anchor()
    receipts = tip["consumed_stage_output_receipts"]
    request_hash = request["request_sha256"]

    if tamper_kind == "receipt":
        receipt = receipts[request_hash]
        receipt["output_candidate_sha256"] = _digest(
            "forged-output-candidate"
        )
        receipt["output_receipt_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "output_receipt_sha256"
            }
        )
    else:
        receipts[_digest("forged-output-map-key")] = receipts.pop(request_hash)
    tip["tip_anchor_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tip.items()
            if key != "tip_anchor_sha256"
        }
    )
    store.current_tip_anchor_path.write_text(
        json.dumps(tip, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SecFilingGemmaRevealStoreError):
        store.load_current_tip_anchor()


@pytest.mark.parametrize("crash_point", ("pending_tip", "state_replace"))
def test_consumed_stage_output_receipt_cas_crash_recovers_once(
    tmp_path: Path,
    crash_point: str,
) -> None:
    store = _store(tmp_path)
    candidate, request, _bundle = _issued_intermediate_grant(
        store,
        salt=f"output-crash-{crash_point}",
        include_sec_plan=True,
    )
    _prepare_fixed_stage_evidence_output(
        store,
        request,
        candidate,
        salt=f"output-crash-{crash_point}-evidence",
    )
    state_bytes_before = store.state_path.read_bytes()
    tip_before = store.load_current_tip_anchor()
    real_atomic_replace = reveal_store_module._atomic_replace
    crashed = False

    def crash_during_output_receipt_cas(path: Path, payload: bytes) -> None:
        nonlocal crashed
        real_atomic_replace(path, payload)
        parsed = json.loads(payload)
        is_target = (
            crash_point == "pending_tip"
            and path == store.current_tip_anchor_path
            and parsed.get("schema_version") == CURRENT_TIP_PENDING_SCHEMA_VERSION
        ) or (
            crash_point == "state_replace"
            and path == store.state_path
        )
        if is_target and not crashed:
            crashed = True
            raise RuntimeError(f"simulated output receipt crash at {crash_point}")

    with patch(
        "agent_benchmark.sec_filing_gemma_reveal_store._atomic_replace",
        crash_during_output_receipt_cas,
    ), pytest.raises(RuntimeError, match="simulated output receipt crash"):
        store._record_owned_stage_evidence_output(
            request_sha256=request["request_sha256"],
        )

    pending = json.loads(store.current_tip_anchor_path.read_bytes())
    assert pending["schema_version"] == CURRENT_TIP_PENDING_SCHEMA_VERSION
    expected_receipt = pending["next_tip_anchor"][
        "consumed_stage_output_receipts"
    ][request["request_sha256"]]

    recovered = store._record_owned_stage_evidence_output(
        request_sha256=request["request_sha256"],
    )

    assert recovered == expected_receipt
    assert store.state_path.read_bytes() == state_bytes_before
    recovered_tip = store.load_current_tip_anchor()
    assert recovered_tip["schema_version"] != CURRENT_TIP_PENDING_SCHEMA_VERSION
    assert recovered_tip["revision"] == tip_before["revision"] + 1
    assert recovered_tip["consumed_stage_output_receipts"] == {
        request["request_sha256"]: expected_receipt
    }
    stable_tip_bytes = store.current_tip_anchor_path.read_bytes()
    assert store._record_owned_stage_evidence_output(
        request_sha256=request["request_sha256"],
    ) == expected_receipt
    assert store.current_tip_anchor_path.read_bytes() == stable_tip_bytes


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


@pytest.mark.parametrize("deleted_parent_file", ("evidence", "marker"))
def test_final_consumption_replays_durable_parent_before_pin_or_verifier(
    tmp_path: Path,
    deleted_parent_file: str,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    registered, candidate = _register(
        store,
        salt=f"final-parent-replay-{deleted_parent_file}",
    )
    development_evidence = _evidence(
        "development",
        candidate,
        salt=f"final-parent-replay-{deleted_parent_file}",
    )
    intermediate_request, intermediate_access = _grant_request(
        registered,
        candidate,
        stage="intermediate",
        evidence=development_evidence,
        include_sec_plan=True,
    )
    intermediate_bundle = _consume_with_grant(
        store,
        intermediate_request,
        candidate,
        development_evidence,
        stage="intermediate",
        access_manifest=intermediate_access,
    )
    prepared = _prepare_fixed_stage_evidence_output(
        store,
        intermediate_request,
        candidate,
        salt=f"final-parent-replay-{deleted_parent_file}",
    )
    intermediate_evidence = prepared["stage_evidence"]
    store._record_owned_stage_evidence_output(
        request_sha256=intermediate_request["request_sha256"],
    )
    final_request, final_access = _request(
        intermediate_bundle["authenticated_store_snapshot"],
        candidate,
        stage="final",
        evidence=intermediate_evidence,
        salt=f"final-parent-replay-{deleted_parent_file}",
    )
    prepared[f"{deleted_parent_file}_path"].unlink()
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()

    def verifier_must_not_run(*_args, **_kwargs):
        raise AssertionError("durable parent failure must precede verifier execution")

    with pytest.raises(SecFilingGemmaRevealStoreError):
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
        include_sec_plan=True,
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

    prepared = _prepare_fixed_stage_evidence_output(
        store,
        intermediate_request,
        candidate,
        salt="final-count-final",
    )
    intermediate_evidence = prepared["stage_evidence"]
    store._record_owned_stage_evidence_output(
        request_sha256=intermediate_request["request_sha256"],
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
        include_sec_plan=True,
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
    prepared = _prepare_fixed_stage_evidence_output(
        store,
        intermediate_request,
        candidate,
        salt="final-parent-context-final",
    )
    intermediate_evidence = prepared["stage_evidence"]
    intermediate_output_receipt = store._record_owned_stage_evidence_output(
        request_sha256=intermediate_request["request_sha256"],
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
        "consumed_stage_output_receipt": intermediate_output_receipt,
        "consumed_stage_output_receipt_sha256": intermediate_output_receipt[
            "output_receipt_sha256"
        ],
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
        "consumed_stage_output_receipts": post_pin_tip[
            "consumed_stage_output_receipts"
        ],
            "consumed_stage_output_receipts_sha256": canonical_sha256(
                post_pin_tip["consumed_stage_output_receipts"]
            ),
            "stage_sec_execution_claims": post_pin_tip[
                "stage_sec_execution_claims"
            ],
            "stage_sec_execution_claims_sha256": canonical_sha256(
                post_pin_tip["stage_sec_execution_claims"]
            ),
            "stage_sec_reader_receipts": post_pin_tip[
                "stage_sec_reader_receipts"
            ],
            "stage_sec_reader_receipts_sha256": canonical_sha256(
                post_pin_tip["stage_sec_reader_receipts"]
            ),
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


def test_final_requires_parent_first_output_receipt_before_verifier_or_pin(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate, intermediate_request, intermediate_bundle = (
        _issued_intermediate_grant(
            store,
            salt="final-parent-output-receipt",
        )
    )
    intermediate_evidence = _evidence(
        "intermediate",
        candidate,
        salt="final-parent-output-receipt-evidence",
    )
    final_request, final_access = _request(
        intermediate_bundle["authenticated_store_snapshot"],
        candidate,
        stage="final",
        evidence=intermediate_evidence,
        salt="final-parent-output-receipt-final",
    )
    tip_before = store.load_current_tip_anchor()
    assert intermediate_request["request_sha256"] in tip_before[
        "authorization_bundles"
    ]
    assert intermediate_request["request_sha256"] not in tip_before[
        "consumed_stage_output_receipts"
    ]
    state_bytes = store.state_path.read_bytes()
    tip_bytes = store.current_tip_anchor_path.read_bytes()

    def verifier_must_not_run(*_args, **_kwargs):
        raise AssertionError(
            "missing parent output receipt must fail before verifier execution"
        )

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="persisted first-output receipt",
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
    assert store.load_current_tip_anchor() == tip_before
    assert final_request["request_sha256"] not in tip_before[
        "trusted_stage_content_pins"
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
