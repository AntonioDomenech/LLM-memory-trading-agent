from __future__ import annotations

import copy
from datetime import date
from functools import lru_cache
import hashlib
import re
from types import MappingProxyType

import pytest

from agent_benchmark.sec_filing_gemma_contract import (
    CANONICAL_IDENTITY_LEXICON_SHA256,
    CONTRACT_VERSION,
    REQUIRED_SOURCE_HASHES,
    build_candidate_manifest,
    build_corpus_universe_manifest,
    build_stage_content_manifest,
    canonical_sha256,
    session_calendar_sha256,
)
from agent_benchmark.sec_filing_gemma_market_evidence import MARKET_SYMBOLS
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND,
    REVEAL_REQUEST_SCHEMA_VERSION,
    candidate_design_sha256,
)
from agent_benchmark.sec_filing_gemma_stage_access import (
    DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID,
    DEVELOPMENT_CONTENT_ROOT_PLAN_SCHEMA_VERSION,
    DEVELOPMENT_CONTENT_ROOT_RAW_BATCH_CAP_BYTES,
    MODEL_ENDPOINT,
    STAGE_ACCESS_MANIFEST_SCHEMA_VERSION,
    SecFilingGemmaStageAccessError,
    build_development_content_root_plan,
    build_stage_access_manifest,
    validate_development_content_root_plan,
    validate_prior_same_form_carry_in_scope,
    validate_stage_access_manifest,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS


_MISSING = object()


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source_record(year: int, serial: int, form: str, month: int) -> dict:
    evidence_date = date(year, month, 15)
    return {
        "accession_number": f"0000320193-{year % 100:02d}-{serial:06d}",
        "subject_cik": "0000320193",
        "form": form,
        "acceptance_datetime": evidence_date.strftime("%Y%m%d") + "160000",
        "filing_date": evidence_date.isoformat(),
        "filing_date_change": None,
        "primary_document": f"filing-{serial}.htm",
        "source_record_sha256": f"{serial + 60_000:064x}",
    }


@lru_cache(maxsize=1)
def _frozen_universe() -> dict:
    records: list[dict] = []
    serial = 1
    for year in range(2000, 2026):
        for form, month in (
            ("10-K", 2),
            ("10-Q", 5),
            ("10-Q", 8),
            ("10-Q", 11),
        ):
            records.append(_source_record(year, serial, form, month))
            serial += 1
    for form, month in (("10-Q", 2), ("10-Q", 5)):
        records.append(_source_record(2026, serial, form, month))
        serial += 1
    return build_corpus_universe_manifest(
        catalog_artifact_sha256=_h("catalog"),
        calendar_artifact_sha256=_h("calendar-evidence"),
        catalog_total_record_count=1_000,
        catalog_eligible_record_count=len(records),
        session_dates=EXPECTED_SESSIONS,
        records=records,
    )


def _candidate(universe: dict) -> dict:
    return build_candidate_manifest(
        model_digest=_h("model-digest"),
        ollama_runtime_fingerprint_sha256=_h("runtime-fingerprint"),
        sec_audit_checksums_json_sha256=_h("sec-audit"),
        sec_catalog_artifact_sha256=universe["catalog_artifact_sha256"],
        sec_audit_source_commit=_h("audit-commit"),
        calendar_source_evidence_sha256=universe["calendar_artifact_sha256"],
        calendar_sessions_sha256=universe["calendar_sessions_sha256"],
        corpus_universe_sha256=universe["universe_sha256"],
        corpus_universe_semantic_sha256=universe[
            "universe_semantic_sha256"
        ],
        identity_lexicon_sha256=CANONICAL_IDENTITY_LEXICON_SHA256,
        predecessor_reveal_registry_sha256=_h("predecessor-registry"),
        holdout_attempt_id=f"{CONTRACT_VERSION}-attempt-001",
        experiment_source_commit=_h("experiment-commit"),
        source_tree_sha256=_h("source-tree"),
        source_hashes={name: _h(f"source-{name}") for name in REQUIRED_SOURCE_HASHES},
    )


def _documents(stage: str, universe: dict) -> list[dict[str, str]]:
    documents = [
        {
            "accession_number": record["accession_number"],
            "official_url": (
                "https://www.sec.gov/Archives/edgar/data/320193/"
                f"{record['accession_number'].replace('-', '')}/"
                f"{record['primary_document']}"
            ),
        }
        for record in universe["records"]
        if record["artifact_stage"] == stage
    ]
    return sorted(documents, key=lambda item: item["accession_number"])


def _content_manifest(stage: str, universe: dict) -> dict:
    documents = [
        {
            "accession_number": record["accession_number"],
            "primary_document_sha256": _h(
                f"{stage}:{record['accession_number']}:primary"
            ),
            "normalized_text_sha256": _h(
                f"{stage}:{record['accession_number']}:normalized"
            ),
            "primary_document_bytes": 2_000,
            "normalized_text_bytes": 1_000,
        }
        for record in universe["records"]
        if record["artifact_stage"] == stage
    ]
    return build_stage_content_manifest(
        artifact_stage=stage,
        corpus_universe_sha256=universe["universe_sha256"],
        documents=documents,
        universe_manifest=universe,
    )


def _development_root_context(universe: dict | None = None) -> dict:
    exact_universe = copy.deepcopy(_frozen_universe() if universe is None else universe)
    candidate = _candidate(exact_universe)
    kwargs = {
        "candidate_manifest": candidate,
        "expected_candidate_sha256": candidate["candidate_sha256"],
        "expected_candidate_design_sha256": candidate_design_sha256(candidate),
        "expected_attempt_id": candidate["bindings"]["holdout_attempt_id"],
        "base_corpus_universe_sha256": exact_universe["universe_sha256"],
        "corpus_universe_manifest": exact_universe,
        "session_calendar_sha256": exact_universe["calendar_sessions_sha256"],
    }
    plan = build_development_content_root_plan(**kwargs)
    return {**kwargs, "plan": plan}


def _validate_development_root(
    context: dict,
    *,
    plan: dict | None = None,
    expected_plan_hash: str | None = None,
    **overrides: object,
) -> str:
    observed = context["plan"] if plan is None else plan
    kwargs = {
        key: context[key]
        for key in (
            "candidate_manifest",
            "expected_candidate_sha256",
            "expected_candidate_design_sha256",
            "expected_attempt_id",
            "base_corpus_universe_sha256",
            "corpus_universe_manifest",
            "session_calendar_sha256",
        )
    }
    kwargs.update(overrides)
    return validate_development_content_root_plan(
        observed,
        expected_development_content_root_plan_sha256=(
            expected_plan_hash
            or observed["development_content_root_plan_sha256"]
        ),
        **kwargs,
    )


def _rehash_development_root_plan(plan: dict) -> dict:
    value = copy.deepcopy(plan)
    body = {
        key: value[key]
        for key in value
        if key != "development_content_root_plan_sha256"
    }
    value["development_content_root_plan_sha256"] = canonical_sha256(body)
    return value


def _fully_rehash_development_document_scope(plan: dict) -> dict:
    value = copy.deepcopy(plan)
    documents = value["sec_access_plan"]["documents"]
    accessions = [document["accession_number"] for document in documents]
    urls = [document["official_url"] for document in documents]
    value["sec_access_plan"].update(
        {
            "document_count": len(documents),
            "accessions_sha256": canonical_sha256(accessions),
            "official_urls_sha256": canonical_sha256(urls),
        }
    )
    value["root_scope"].update(
        {
            "document_count": len(documents),
            "accessions_sha256": canonical_sha256(accessions),
            "official_urls_sha256": canonical_sha256(urls),
        }
    )
    value["budgets"]["max_sec_requests"] = len(documents)
    value["development_root_scope_sha256"] = canonical_sha256(
        value["root_scope"]
    )
    return _rehash_development_root_plan(value)


def test_development_content_root_plan_is_complete_request_free_and_valid() -> None:
    context = _development_root_context()
    plan = context["plan"]
    documents = _documents("development", context["corpus_universe_manifest"])

    assert _validate_development_root(context) == plan[
        "development_content_root_plan_sha256"
    ]
    assert plan["schema_version"] == DEVELOPMENT_CONTENT_ROOT_PLAN_SCHEMA_VERSION
    assert plan["sec_access_plan"]["documents"] == documents
    assert plan["sec_access_plan"]["document_count"] == len(documents)
    assert plan["corpus_universe_manifest"] == context[
        "corpus_universe_manifest"
    ]
    assert plan["root_scope"]["artifact_stage"] == "development"
    assert plan["root_scope"]["document_count"] == len(documents)
    assert plan["root_scope"]["component_id"] == (
        DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID
    )
    assert plan["development_root_scope_sha256"] == canonical_sha256(
        plan["root_scope"]
    )
    assert plan["output"] == {
        "namespace": (
            f"aapl-sec-gemma-{context['expected_attempt_id']}-"
            "development-content-root"
        ),
        "component_id": DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID,
        "write_mode": "create_new_exclusive",
        "existing_namespace_reuse_permitted": False,
    }
    assert plan["budgets"]["max_raw_batch_bytes"] == (
        DEVELOPMENT_CONTENT_ROOT_RAW_BATCH_CAP_BYTES
    )
    assert plan["budgets"]["max_paid_api_calls"] == 0
    assert plan["budgets"]["max_estimated_cost_usd"] == 0.0
    assert "reveal_request" not in plan
    assert "market_access" not in plan
    assert "model_access" not in plan
    assert plan["scope"]["reveal_request_required"] is False
    assert plan["scope"]["outcome_access_permitted"] is False
    assert plan["scope"]["market_access_permitted"] is False
    assert plan["scope"]["model_access_permitted"] is False
    assert plan["scope"]["consumption_ledger_mutation_permitted"] is False

    universe_by_accession = {
        record["accession_number"]: record
        for record in context["corpus_universe_manifest"]["records"]
    }
    assert {
        universe_by_accession[document["accession_number"]]["artifact_stage"]
        for document in documents
    } == {"development"}
    assert set(plan["scope"]["prohibited_artifact_stages"]) == {
        "intermediate",
        "final",
    }


def test_development_content_root_plan_detaches_complete_universe() -> None:
    context = _development_root_context()
    embedded = copy.deepcopy(context["plan"]["corpus_universe_manifest"])

    context["corpus_universe_manifest"]["records"][0][
        "primary_document"
    ] = "caller-mutated.htm"

    assert context["plan"]["corpus_universe_manifest"] == embedded


def test_development_content_root_rejects_omission_extra_reorder_and_wrong_url() -> None:
    context = _development_root_context()
    base_documents = context["plan"]["sec_access_plan"]["documents"]
    extra = _documents("intermediate", context["corpus_universe_manifest"])[0]

    forged_document_sets = []
    omitted = copy.deepcopy(base_documents[:-1])
    forged_document_sets.append(omitted)
    with_extra = copy.deepcopy(base_documents)
    with_extra.append(copy.deepcopy(extra))
    forged_document_sets.append(with_extra)
    forged_document_sets.append(list(reversed(copy.deepcopy(base_documents))))
    wrong_url = copy.deepcopy(base_documents)
    wrong_url[0]["official_url"] = wrong_url[0]["official_url"].replace(
        "www.sec.gov",
        "evil.example",
    )
    forged_document_sets.append(wrong_url)

    for documents in forged_document_sets:
        forged = copy.deepcopy(context["plan"])
        forged["sec_access_plan"]["documents"] = documents
        forged = _fully_rehash_development_document_scope(forged)
        with pytest.raises(
            SecFilingGemmaStageAccessError,
            match="exact candidate-bound construction",
        ):
            _validate_development_root(
                context,
                plan=forged,
                expected_plan_hash=forged[
                    "development_content_root_plan_sha256"
                ],
            )


def test_development_content_root_uses_availability_stage_at_2018_boundary() -> None:
    frozen = copy.deepcopy(_frozen_universe())
    source_keys = (
        "accession_number",
        "subject_cik",
        "form",
        "acceptance_datetime",
        "filing_date",
        "filing_date_change",
        "primary_document",
        "source_record_sha256",
    )
    source_records = [
        {key: record[key] for key in source_keys}
        for record in frozen["records"]
    ]
    accession_2018 = next(
        record
        for record in source_records
        if record["accession_number"].startswith("0000320193-18-")
        and record["form"] == "10-K"
    )
    accession_2019 = next(
        record
        for record in source_records
        if record["accession_number"].startswith("0000320193-19-")
        and record["form"] == "10-K"
    )
    accession_2018.update(
        {
            "acceptance_datetime": "20181231160000",
            "filing_date": "2018-12-31",
        }
    )
    accession_2019.update(
        {
            "acceptance_datetime": "20181228160000",
            "filing_date": "2018-12-28",
        }
    )
    universe = build_corpus_universe_manifest(
        catalog_artifact_sha256=frozen["catalog_artifact_sha256"],
        calendar_artifact_sha256=frozen["calendar_artifact_sha256"],
        catalog_total_record_count=frozen["catalog_total_record_count"],
        catalog_eligible_record_count=len(source_records),
        session_dates=EXPECTED_SESSIONS,
        records=source_records,
    )
    context = _development_root_context(universe)
    plan_accessions = {
        document["accession_number"]
        for document in context["plan"]["sec_access_plan"]["documents"]
    }
    records_by_accession = {
        record["accession_number"]: record for record in universe["records"]
    }

    assert records_by_accession[accession_2019["accession_number"]][
        "availability_session"
    ] == "2018-12-31"
    assert records_by_accession[accession_2019["accession_number"]][
        "artifact_stage"
    ] == "development"
    assert accession_2019["accession_number"] in plan_accessions
    assert records_by_accession[accession_2018["accession_number"]][
        "availability_session"
    ] == "2019-01-02"
    assert records_by_accession[accession_2018["accession_number"]][
        "artifact_stage"
    ] == "intermediate"
    assert accession_2018["accession_number"] not in plan_accessions
    assert _validate_development_root(context) == context["plan"][
        "development_content_root_plan_sha256"
    ]


def test_development_content_root_requires_scope_plan_and_external_self_hashes() -> None:
    context = _development_root_context()
    changed = copy.deepcopy(context["plan"])
    changed["budgets"]["max_raw_batch_bytes"] -= 1
    with pytest.raises(SecFilingGemmaStageAccessError, match="not canonical"):
        _validate_development_root(context, plan=changed)

    changed = _rehash_development_root_plan(changed)
    with pytest.raises(
        SecFilingGemmaStageAccessError,
        match="exact candidate-bound construction",
    ):
        _validate_development_root(
            context,
            plan=changed,
            expected_plan_hash=changed["development_content_root_plan_sha256"],
        )

    changed_scope = copy.deepcopy(context["plan"])
    changed_scope["root_scope"]["candidate_sha256"] = _h("forged candidate")
    changed_scope = _rehash_development_root_plan(changed_scope)
    with pytest.raises(SecFilingGemmaStageAccessError, match="scope is not"):
        _validate_development_root(
            context,
            plan=changed_scope,
            expected_plan_hash=changed_scope[
                "development_content_root_plan_sha256"
            ],
        )

    with pytest.raises(SecFilingGemmaStageAccessError, match="externally pinned"):
        _validate_development_root(
            context,
            expected_plan_hash=_h("wrong development root plan"),
        )


def _request_identity(
    *,
    candidate: dict,
    design_hash: str,
    prerequisite_stage: str,
    requested_stage: str,
) -> dict:
    return {
        "schema_version": REVEAL_REQUEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "registry_sha256": _h("current-registry"),
        "registry_tip_sha256": _h("current-tip"),
        "registered_entry_count": 1,
        "historical_final_reveal_count_lower_bound": (
            HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND
        ),
        "stage": requested_stage,
        "prerequisite_stage": prerequisite_stage,
        "prerequisite_stage_evidence_sha256": _h(
            f"{prerequisite_stage}-evidence"
        ),
        "attempt_id": candidate["bindings"]["holdout_attempt_id"],
        "candidate_sha256": candidate["candidate_sha256"],
        "candidate_design_sha256": design_hash,
        "registry_entry_sha256": _h("registry-entry"),
        "request_scope": "one_current_tip_candidate_and_one_stage_only",
        "authorizes_outcome_access": False,
        "effectful_atomic_single_use_consumption_required": True,
        "cross_attempt_comparison_permitted": False,
        "cross_attempt_winner_selection_permitted": False,
        "globally_pristine_claim": False,
    }


def _completed_request(identity: dict, manifest_hash: str) -> dict:
    body = {
        **copy.deepcopy(identity),
        "stage_access_manifest_sha256": manifest_hash,
    }
    return {**body, "request_sha256": canonical_sha256(body)}


def _context(stage: str = "intermediate") -> dict:
    prerequisite = "development" if stage == "intermediate" else "intermediate"
    universe = copy.deepcopy(_frozen_universe())
    candidate = _candidate(universe)
    design_hash = candidate_design_sha256(candidate)
    identity = _request_identity(
        candidate=candidate,
        design_hash=design_hash,
        prerequisite_stage=prerequisite,
        requested_stage=stage,
    )
    documents = _documents(stage, universe)
    prerequisite_content = _content_manifest(prerequisite, universe)
    artifacts = {symbol: _h(f"artifact-{symbol}") for symbol in MARKET_SYMBOLS}
    windows = {symbol: _h(f"{stage}-window-{symbol}") for symbol in MARKET_SYMBOLS}
    output_namespace = (
        f"aapl-sec-gemma-{candidate['bindings']['holdout_attempt_id']}-{stage}"
    )
    build_kwargs = {
        "prerequisite_stage": prerequisite,
        "requested_stage": stage,
        "candidate_manifest": candidate,
        "expected_candidate_sha256": candidate["candidate_sha256"],
        "expected_candidate_design_sha256": design_hash,
        "expected_attempt_id": candidate["bindings"]["holdout_attempt_id"],
        "expected_stage_verifier_source_sha256": candidate["bindings"][
            "source_hashes"
        ]["stage_verifier"],
        "reveal_request_identity": identity,
        "registry_sha256": identity["registry_sha256"],
        "registry_tip_sha256": identity["registry_tip_sha256"],
        "registered_entry_count": identity["registered_entry_count"],
        "registry_entry_sha256": identity["registry_entry_sha256"],
        "base_corpus_universe_sha256": candidate["bindings"][
            "corpus_universe_sha256"
        ],
        "corpus_universe_manifest": universe,
        "prerequisite_content_manifest": prerequisite_content,
        "expected_prerequisite_content_manifest_sha256": prerequisite_content[
            "content_manifest_sha256"
        ],
        "prerequisite_stage_artifact_sha256": _h(
            f"{prerequisite}-stage-artifact"
        ),
        "prerequisite_external_seal_receipt_sha256": _h(
            f"{prerequisite}-external-seal-receipt"
        ),
        "session_calendar_sha256": candidate["bindings"][
            "calendar_sessions_sha256"
        ],
        "authorized_documents": documents,
        "market_source_manifest_sha256": _h(f"{stage}-market-manifest"),
        "market_source_artifact_sha256s": artifacts,
        "market_source_window_sha256s": windows,
        "max_sec_response_bytes": 8_000_000,
        "output_namespace": output_namespace,
    }
    manifest = build_stage_access_manifest(**build_kwargs)
    request = _completed_request(identity, manifest["stage_access_manifest_sha256"])
    return {
        **build_kwargs,
        "identity": identity,
        "manifest": manifest,
        "request": request,
    }


def _validate(
    context: dict,
    *,
    manifest: dict | None = None,
    request=_MISSING,
    expected_manifest_hash: str | None = None,
    expected_request_hash: str | None = None,
    **overrides: object,
) -> str:
    observed_manifest = context["manifest"] if manifest is None else manifest
    observed_request = context["request"] if request is _MISSING else request
    kwargs = {
        key: context[key]
        for key in (
            "prerequisite_stage",
            "requested_stage",
            "candidate_manifest",
            "expected_candidate_sha256",
            "expected_candidate_design_sha256",
            "expected_attempt_id",
            "expected_stage_verifier_source_sha256",
            "registry_sha256",
            "registry_tip_sha256",
            "registered_entry_count",
            "registry_entry_sha256",
            "base_corpus_universe_sha256",
            "corpus_universe_manifest",
            "prerequisite_content_manifest",
            "expected_prerequisite_content_manifest_sha256",
            "prerequisite_stage_artifact_sha256",
            "prerequisite_external_seal_receipt_sha256",
            "session_calendar_sha256",
            "authorized_documents",
            "market_source_manifest_sha256",
            "market_source_artifact_sha256s",
            "market_source_window_sha256s",
            "max_sec_response_bytes",
            "output_namespace",
        )
    }
    kwargs.update(overrides)
    return validate_stage_access_manifest(
        observed_manifest,
        expected_stage_access_manifest_sha256=(
            expected_manifest_hash
            or observed_manifest["stage_access_manifest_sha256"]
        ),
        reveal_request=observed_request,
        expected_reveal_request_sha256=(
            expected_request_hash
            or (
                context["request"]["request_sha256"]
                if observed_request is None
                else observed_request["request_sha256"]
            )
        ),
        **kwargs,
    )


def _rehash_manifest(manifest: dict) -> dict:
    value = copy.deepcopy(manifest)
    body = {
        key: value[key]
        for key in value
        if key != "stage_access_manifest_sha256"
    }
    value["stage_access_manifest_sha256"] = canonical_sha256(body)
    return value


def _request_for(context: dict, manifest: dict) -> dict:
    return _completed_request(
        context["identity"], manifest["stage_access_manifest_sha256"]
    )


@pytest.mark.parametrize("stage", ["intermediate", "final"])
def test_exact_two_permitted_transitions_build_and_validate(stage: str) -> None:
    context = _context(stage)
    manifest = context["manifest"]
    observed = _validate(context)

    assert observed == manifest["stage_access_manifest_sha256"]
    assert manifest["schema_version"] == STAGE_ACCESS_MANIFEST_SCHEMA_VERSION
    assert manifest["model_access"]["endpoint"] == MODEL_ENDPOINT
    assert manifest["model_access"]["paid_api_calls"] == 0
    assert manifest["model_access"]["estimated_cost_usd"] == 0.0
    assert manifest["sec_access_plan"]["documents"] == context[
        "authorized_documents"
    ]
    assert manifest["reveal_request_binding"][
        "request_body_identity_sha256"
    ] == canonical_sha256(context["identity"])
    assert manifest["verifier"]["stage_verifier_source_sha256"] == context[
        "expected_stage_verifier_source_sha256"
    ]
    base = manifest["corpus_provenance"]["frozen_base_universe"]
    live = manifest["corpus_provenance"]["live_append_only_extension"]
    assert base["corpus_universe_sha256"] == context[
        "base_corpus_universe_sha256"
    ]
    assert live["identity_sha256"] is None
    assert live["entry_count"] == 0
    assert live["included_in_base_universe_identity"] is False
    assert live["authorized_for_holdout_stage"] is False
    carry_in = manifest["prior_same_form_carry_in"]
    assert carry_in["artifact_scope"] == "sealed_normalized_text_only"
    assert carry_in["network_refetch_permitted"] is False
    assert carry_in["write_permitted"] is False
    assert carry_in["record_count"] == 2
    assert carry_in["prerequisite_content_manifest_sha256"] == context[
        "expected_prerequisite_content_manifest_sha256"
    ]
    assert manifest["prerequisite_evidence_pin"] == {
        "stage": context["prerequisite_stage"],
        "content_manifest_sha256": context[
            "expected_prerequisite_content_manifest_sha256"
        ],
        "stage_artifact_sha256": context[
            "prerequisite_stage_artifact_sha256"
        ],
        "external_seal_receipt_sha256": context[
            "prerequisite_external_seal_receipt_sha256"
        ],
    }
    assert {record["form"] for record in carry_in["records"]} == {"10-K", "10-Q"}
    assert {
        record["artifact_stage"] for record in carry_in["records"]
    } == {context["prerequisite_stage"]}
    assert carry_in["bound_by_prerequisite_stage_evidence_sha256"] == context[
        "identity"
    ]["prerequisite_stage_evidence_sha256"]
    prerequisite_documents = {
        document["accession_number"]: document
        for document in context["prerequisite_content_manifest"]["documents"]
    }
    for record in carry_in["records"]:
        content = prerequisite_documents[record["accession_number"]]
        assert record["normalized_text_sha256"] == content[
            "normalized_text_sha256"
        ]
        assert record["normalized_text_bytes"] == content[
            "normalized_text_bytes"
        ]
        assert record["content_record_sha256"] == canonical_sha256(content)
    assert manifest["scope"]["general_cross_stage_access_permitted"] is False
    assert (
        manifest["scope"]["exact_prior_same_form_carry_in_read_permitted"]
        is True
    )
    assert stage not in manifest["scope"]["prohibited_stages"]
    assert set(manifest["scope"]["prohibited_stages"]) == {
        value for value in ("development", "intermediate", "final") if value != stage
    }


@pytest.mark.parametrize("stage", ["intermediate", "final"])
def test_carry_in_scope_is_rederived_as_detached_latest_10k_and_10q(
    stage: str,
) -> None:
    context = _context(stage)
    manifest = context["manifest"]

    records = validate_prior_same_form_carry_in_scope(
        MappingProxyType(manifest),
        corpus_universe_manifest=MappingProxyType(
            context["corpus_universe_manifest"]
        ),
        prerequisite_content_manifest=MappingProxyType(
            context["prerequisite_content_manifest"]
        ),
        expected_prerequisite_stage=context["prerequisite_stage"],
        expected_requested_stage=context["requested_stage"],
        expected_prerequisite_stage_evidence_sha256=context["identity"][
            "prerequisite_stage_evidence_sha256"
        ],
    )

    assert type(records) is list
    assert records == manifest["prior_same_form_carry_in"]["records"]
    assert [record["form"] for record in records] == ["10-K", "10-Q"]
    for record in records:
        first_requested = min(
            (
                item
                for item in context["corpus_universe_manifest"]["records"]
                if item["artifact_stage"] == stage and item["form"] == record["form"]
            ),
            key=lambda item: (
                item["availability_session"],
                item["accession_number"],
            ),
        )
        expected_prior = max(
            (
                item
                for item in context["corpus_universe_manifest"]["records"]
                if item["form"] == record["form"]
                and (
                    item["availability_session"],
                    item["accession_number"],
                )
                < (
                    first_requested["availability_session"],
                    first_requested["accession_number"],
                )
            ),
            key=lambda item: (
                item["availability_session"],
                item["accession_number"],
            ),
        )
        assert record["accession_number"] == expected_prior["accession_number"]

    records[0]["form"] = "forged"
    assert manifest["prior_same_form_carry_in"]["records"][0]["form"] == "10-K"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda section: section.pop("selection_policy"),
        lambda section: section.update({"extra": "scope"}),
        lambda section: section.update({"selection_policy": "caller_selected"}),
        lambda section: section.update({"artifact_scope": "raw_or_normalized"}),
        lambda section: section.update({"network_refetch_permitted": True}),
        lambda section: section.update({"write_permitted": True}),
        lambda section: section.update(
            {"bound_by_prerequisite_stage_evidence_sha256": _h("other evidence")}
        ),
        lambda section: section.update(
            {"prerequisite_content_manifest_sha256": _h("other content")}
        ),
        lambda section: section.update({"record_count": True}),
        lambda section: section.update({"record_count": 1}),
        lambda section: section.update({"records_sha256": _h("other records")}),
        lambda section: section["records"].reverse(),
        lambda section: section["records"][0].update(
            {"accession_number": "0000320193-18-999999"}
        ),
    ],
    ids=[
        "missing-key",
        "extra-key",
        "selection-policy",
        "artifact-scope",
        "network-refetch",
        "write",
        "evidence-binding",
        "content-hash",
        "boolean-count",
        "count",
        "records-hash",
        "record-order",
        "record-selection",
    ],
)
def test_carry_in_scope_rejects_every_mutated_declaration(mutator) -> None:
    context = _context()
    forged = copy.deepcopy(context["manifest"])
    mutator(forged["prior_same_form_carry_in"])

    with pytest.raises(SecFilingGemmaStageAccessError):
        validate_prior_same_form_carry_in_scope(
            forged,
            corpus_universe_manifest=context["corpus_universe_manifest"],
            prerequisite_content_manifest=context["prerequisite_content_manifest"],
            expected_prerequisite_stage=context["prerequisite_stage"],
            expected_requested_stage=context["requested_stage"],
            expected_prerequisite_stage_evidence_sha256=context["identity"][
                "prerequisite_stage_evidence_sha256"
            ],
        )


@pytest.mark.parametrize(
    "scope_change",
    [
        {"exact_prior_same_form_carry_in_read_permitted": False},
        {"general_cross_stage_access_permitted": True},
        {"prohibited_stages": ["development"]},
        {"prohibited_stages": ["development", "intermediate", "final"]},
    ],
    ids=[
        "carry-read-disabled",
        "general-cross-stage-enabled",
        "another-stage-not-prohibited",
        "requested-stage-prohibited",
    ],
)
def test_carry_in_scope_rejects_broadened_or_disabled_manifest_scope(
    scope_change: dict,
) -> None:
    context = _context()
    forged = copy.deepcopy(context["manifest"])
    forged["scope"].update(scope_change)

    with pytest.raises(SecFilingGemmaStageAccessError, match="exact carry-in read"):
        validate_prior_same_form_carry_in_scope(
            forged,
            corpus_universe_manifest=context["corpus_universe_manifest"],
            prerequisite_content_manifest=context["prerequisite_content_manifest"],
            expected_prerequisite_stage=context["prerequisite_stage"],
            expected_requested_stage=context["requested_stage"],
            expected_prerequisite_stage_evidence_sha256=context["identity"][
                "prerequisite_stage_evidence_sha256"
            ],
        )


def test_carry_in_scope_rejects_canonical_parent_content_drift() -> None:
    context = _context()
    content_document_keys = {
        "accession_number",
        "primary_document_sha256",
        "normalized_text_sha256",
        "primary_document_bytes",
        "normalized_text_bytes",
    }
    documents = [
        {
            key: copy.deepcopy(value)
            for key, value in document.items()
            if key in content_document_keys
        }
        for document in context["prerequisite_content_manifest"]["documents"]
    ]
    carry_accession = context["manifest"]["prior_same_form_carry_in"]["records"][0][
        "accession_number"
    ]
    next(
        document
        for document in documents
        if document["accession_number"] == carry_accession
    )["normalized_text_sha256"] = _h("changed normalized carry-in")
    changed_content = build_stage_content_manifest(
        artifact_stage=context["prerequisite_stage"],
        corpus_universe_sha256=context["corpus_universe_manifest"]["universe_sha256"],
        documents=documents,
        universe_manifest=context["corpus_universe_manifest"],
    )

    with pytest.raises(
        SecFilingGemmaStageAccessError,
        match="exact rederivation",
    ):
        validate_prior_same_form_carry_in_scope(
            context["manifest"],
            corpus_universe_manifest=context["corpus_universe_manifest"],
            prerequisite_content_manifest=changed_content,
            expected_prerequisite_stage=context["prerequisite_stage"],
            expected_requested_stage=context["requested_stage"],
            expected_prerequisite_stage_evidence_sha256=context["identity"][
                "prerequisite_stage_evidence_sha256"
            ],
        )


@pytest.mark.parametrize(
    ("prerequisite", "requested"),
    [
        ("development", "final"),
        ("intermediate", "intermediate"),
        ("final", "intermediate"),
        ("final", "development"),
    ],
)
def test_other_or_reused_stage_transitions_are_rejected(
    prerequisite: str, requested: str
) -> None:
    context = _context()
    kwargs = {
        key: value
        for key, value in context.items()
        if key
        in {
            "prerequisite_stage",
            "requested_stage",
            "candidate_manifest",
            "expected_candidate_sha256",
            "expected_candidate_design_sha256",
            "expected_attempt_id",
            "expected_stage_verifier_source_sha256",
            "reveal_request_identity",
            "registry_sha256",
            "registry_tip_sha256",
            "registered_entry_count",
            "registry_entry_sha256",
            "base_corpus_universe_sha256",
            "corpus_universe_manifest",
            "prerequisite_content_manifest",
            "expected_prerequisite_content_manifest_sha256",
            "prerequisite_stage_artifact_sha256",
            "prerequisite_external_seal_receipt_sha256",
            "session_calendar_sha256",
            "authorized_documents",
            "market_source_manifest_sha256",
            "market_source_artifact_sha256s",
            "market_source_window_sha256s",
            "max_sec_response_bytes",
            "output_namespace",
        }
    }
    kwargs["prerequisite_stage"] = prerequisite
    kwargs["requested_stage"] = requested
    with pytest.raises(SecFilingGemmaStageAccessError, match="Only development"):
        build_stage_access_manifest(**kwargs)


def test_stage_manifest_requires_both_canonical_self_hash_and_external_pin() -> None:
    context = _context()
    tampered = copy.deepcopy(context["manifest"])
    tampered["budgets"]["max_sec_requests"] += 1
    with pytest.raises(SecFilingGemmaStageAccessError, match="not canonical"):
        _validate(context, manifest=tampered)

    with pytest.raises(SecFilingGemmaStageAccessError, match="externally pinned"):
        _validate(context, expected_manifest_hash=_h("wrong-external-pin"))


def test_validator_accepts_read_only_mapping_views_from_the_reveal_store() -> None:
    context = _context()
    assert _validate(
        context,
        manifest=MappingProxyType(context["manifest"]),
        request=MappingProxyType(context["request"]),
    ) == context["manifest"]["stage_access_manifest_sha256"]


def test_validator_can_reconstruct_the_request_from_store_context_hash() -> None:
    context = _context()
    assert _validate(context, request=None) == context["manifest"][
        "stage_access_manifest_sha256"
    ]


def test_arbitrary_self_hashed_manifest_cannot_replace_externally_bound_inputs() -> None:
    context = _context()
    forged = copy.deepcopy(context["manifest"])
    forged["corpus_provenance"]["frozen_base_universe"][
        "corpus_universe_sha256"
    ] = _h("alternate-universe")
    forged = _rehash_manifest(forged)
    request = _request_for(context, forged)
    with pytest.raises(SecFilingGemmaStageAccessError, match="exact externally bound"):
        _validate(context, manifest=forged, request=request)


@pytest.mark.parametrize(
    "field",
    [
        "content_manifest_sha256",
        "stage_artifact_sha256",
        "external_seal_receipt_sha256",
    ],
)
def test_prerequisite_evidence_pin_is_exact_and_manifest_bound(field: str) -> None:
    context = _context()
    forged = copy.deepcopy(context["manifest"])
    forged["prerequisite_evidence_pin"][field] = _h(f"forged-{field}")
    forged = _rehash_manifest(forged)
    request = _request_for(context, forged)
    with pytest.raises(SecFilingGemmaStageAccessError, match="exact externally bound"):
        _validate(context, manifest=forged, request=request)


def test_stage_verifier_source_must_equal_the_candidate_immutable_pin() -> None:
    context = _context()
    kwargs = {
        key: context[key]
        for key in (
            "prerequisite_stage",
            "requested_stage",
            "candidate_manifest",
            "expected_candidate_sha256",
            "expected_candidate_design_sha256",
            "expected_attempt_id",
            "expected_stage_verifier_source_sha256",
            "reveal_request_identity",
            "registry_sha256",
            "registry_tip_sha256",
            "registered_entry_count",
            "registry_entry_sha256",
            "base_corpus_universe_sha256",
            "corpus_universe_manifest",
            "prerequisite_content_manifest",
            "expected_prerequisite_content_manifest_sha256",
            "prerequisite_stage_artifact_sha256",
            "prerequisite_external_seal_receipt_sha256",
            "session_calendar_sha256",
            "authorized_documents",
            "market_source_manifest_sha256",
            "market_source_artifact_sha256s",
            "market_source_window_sha256s",
            "max_sec_response_bytes",
            "output_namespace",
        )
    }
    kwargs["expected_stage_verifier_source_sha256"] = _h("substitute-verifier")
    with pytest.raises(SecFilingGemmaStageAccessError, match="immutable candidate pin"):
        build_stage_access_manifest(**kwargs)


@pytest.mark.parametrize("forbidden_key", ["result", "passed", "score"])
def test_result_pass_and_score_fields_are_forbidden_even_when_rehashed(
    forbidden_key: str,
) -> None:
    context = _context()
    forged = copy.deepcopy(context["manifest"])
    forged["scope"][forbidden_key] = True
    forged = _rehash_manifest(forged)
    request = _request_for(context, forged)
    with pytest.raises(SecFilingGemmaStageAccessError, match="result, pass, or score"):
        _validate(context, manifest=forged, request=request)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("candidate_sha256", _h("other-candidate")),
        ("registry_sha256", _h("other-registry")),
        ("registry_tip_sha256", _h("other-tip")),
    ],
)
def test_candidate_registry_or_request_body_mismatch_is_rejected(
    field: str, replacement: str
) -> None:
    context = _context()
    identity = copy.deepcopy(context["identity"])
    identity[field] = replacement
    request = _completed_request(
        identity, context["manifest"]["stage_access_manifest_sha256"]
    )
    with pytest.raises(SecFilingGemmaStageAccessError, match="not bound"):
        _validate(context, request=request)


def test_request_hash_requires_its_own_external_pin() -> None:
    context = _context()
    with pytest.raises(SecFilingGemmaStageAccessError, match="external request pin"):
        _validate(context, expected_request_hash=_h("wrong-request-pin"))


def test_valid_manifest_cannot_be_reused_for_another_stage() -> None:
    context = _context("intermediate")
    with pytest.raises(SecFilingGemmaStageAccessError):
        _validate(
            context,
            prerequisite_stage="intermediate",
            requested_stage="final",
            output_namespace=(
                f"aapl-sec-gemma-{context['expected_attempt_id']}-final"
            ),
        )


def test_omitted_or_extra_document_fails_all_and_only_external_plan() -> None:
    context = _context()
    omitted_kwargs = {
        key: context[key]
        for key in (
            "prerequisite_stage",
            "requested_stage",
            "candidate_manifest",
            "expected_candidate_sha256",
            "expected_candidate_design_sha256",
            "expected_attempt_id",
            "expected_stage_verifier_source_sha256",
            "reveal_request_identity",
            "registry_sha256",
            "registry_tip_sha256",
            "registered_entry_count",
            "registry_entry_sha256",
            "base_corpus_universe_sha256",
            "corpus_universe_manifest",
            "prerequisite_content_manifest",
            "expected_prerequisite_content_manifest_sha256",
            "prerequisite_stage_artifact_sha256",
            "prerequisite_external_seal_receipt_sha256",
            "session_calendar_sha256",
            "authorized_documents",
            "market_source_manifest_sha256",
            "market_source_artifact_sha256s",
            "market_source_window_sha256s",
            "max_sec_response_bytes",
            "output_namespace",
        )
    }
    omitted_kwargs["authorized_documents"] = context["authorized_documents"][:-1]
    with pytest.raises(SecFilingGemmaStageAccessError, match="all and only"):
        build_stage_access_manifest(**omitted_kwargs)

    extra_documents = copy.deepcopy(context["authorized_documents"])
    extra_documents.append(
        {
            "accession_number": "0000320193-20-000003",
            "official_url": (
                "https://www.sec.gov/Archives/edgar/data/320193/"
                "000032019320000003/a20q3.htm"
            ),
        }
    )
    extra_kwargs = dict(omitted_kwargs)
    extra_kwargs["authorized_documents"] = extra_documents
    with pytest.raises(
        SecFilingGemmaStageAccessError, match="all and only|sorted and deduplicated"
    ):
        build_stage_access_manifest(**extra_kwargs)


def test_same_accession_wrong_primary_filename_is_rejected() -> None:
    context = _context()
    kwargs = {
        key: context[key]
        for key in (
            "prerequisite_stage",
            "requested_stage",
            "candidate_manifest",
            "expected_candidate_sha256",
            "expected_candidate_design_sha256",
            "expected_attempt_id",
            "expected_stage_verifier_source_sha256",
            "reveal_request_identity",
            "registry_sha256",
            "registry_tip_sha256",
            "registered_entry_count",
            "registry_entry_sha256",
            "base_corpus_universe_sha256",
            "corpus_universe_manifest",
            "prerequisite_content_manifest",
            "expected_prerequisite_content_manifest_sha256",
            "prerequisite_stage_artifact_sha256",
            "prerequisite_external_seal_receipt_sha256",
            "session_calendar_sha256",
            "authorized_documents",
            "market_source_manifest_sha256",
            "market_source_artifact_sha256s",
            "market_source_window_sha256s",
            "max_sec_response_bytes",
            "output_namespace",
        )
    }
    changed = copy.deepcopy(context["authorized_documents"])
    changed[0]["official_url"] = (
        changed[0]["official_url"].rsplit("/", 1)[0]
        + "/not-the-primary-document.htm"
    )
    kwargs["authorized_documents"] = changed
    with pytest.raises(SecFilingGemmaStageAccessError, match="all and only"):
        build_stage_access_manifest(**kwargs)


def test_duplicate_or_reordered_accessions_and_urls_are_rejected() -> None:
    context = _context()
    kwargs = {
        key: context[key]
        for key in (
            "prerequisite_stage",
            "requested_stage",
            "candidate_manifest",
            "expected_candidate_sha256",
            "expected_candidate_design_sha256",
            "expected_attempt_id",
            "expected_stage_verifier_source_sha256",
            "reveal_request_identity",
            "registry_sha256",
            "registry_tip_sha256",
            "registered_entry_count",
            "registry_entry_sha256",
            "base_corpus_universe_sha256",
            "corpus_universe_manifest",
            "prerequisite_content_manifest",
            "expected_prerequisite_content_manifest_sha256",
            "prerequisite_stage_artifact_sha256",
            "prerequisite_external_seal_receipt_sha256",
            "session_calendar_sha256",
            "authorized_documents",
            "market_source_manifest_sha256",
            "market_source_artifact_sha256s",
            "market_source_window_sha256s",
            "max_sec_response_bytes",
            "output_namespace",
        )
    }
    for documents in (
        [
            context["authorized_documents"][0],
            context["authorized_documents"][0],
        ],
        list(reversed(context["authorized_documents"])),
    ):
        changed = dict(kwargs)
        changed["authorized_documents"] = documents
        with pytest.raises(SecFilingGemmaStageAccessError, match="sorted and deduplicated"):
            build_stage_access_manifest(**changed)


def test_nonofficial_or_nonmatching_sec_url_is_rejected_before_authorization() -> None:
    context = _context()
    documents = copy.deepcopy(context["authorized_documents"])
    documents[0]["official_url"] = documents[0]["official_url"].replace(
        "www.sec.gov", "evil.example"
    )
    kwargs = {
        key: context[key]
        for key in (
            "prerequisite_stage",
            "requested_stage",
            "candidate_manifest",
            "expected_candidate_sha256",
            "expected_candidate_design_sha256",
            "expected_attempt_id",
            "expected_stage_verifier_source_sha256",
            "reveal_request_identity",
            "registry_sha256",
            "registry_tip_sha256",
            "registered_entry_count",
            "registry_entry_sha256",
            "base_corpus_universe_sha256",
            "corpus_universe_manifest",
            "prerequisite_content_manifest",
            "expected_prerequisite_content_manifest_sha256",
            "prerequisite_stage_artifact_sha256",
            "prerequisite_external_seal_receipt_sha256",
            "session_calendar_sha256",
            "market_source_manifest_sha256",
            "market_source_artifact_sha256s",
            "market_source_window_sha256s",
            "max_sec_response_bytes",
            "output_namespace",
        )
    }
    kwargs["authorized_documents"] = documents
    with pytest.raises(SecFilingGemmaStageAccessError, match="official SEC HTTPS"):
        build_stage_access_manifest(**kwargs)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["scope"].update(
            {"prohibited_stages": ["development"]}
        ),
        lambda value: value["model_access"].update(
            {"endpoint": "https://api.example/v1/chat"}
        ),
        lambda value: value["model_access"].update({"estimated_cost_usd": 0.01}),
        lambda value: value["budgets"].update({"max_sec_requests": 999}),
        lambda value: value["market_access"].update(
            {"source_family": "unsealed-remote-market-data"}
        ),
        lambda value: value["verifier"].update(
            {"stage_verifier_source_sha256": _h("substituted-verifier")}
        ),
        lambda value: value["corpus_provenance"][
            "live_append_only_extension"
        ].update(
            {
                "identity_sha256": _h("live-extension"),
                "entry_count": 1,
            }
        ),
        lambda value: value["prior_same_form_carry_in"]["records"][0].update(
            {"accession_number": "0000320193-18-999999"}
        ),
    ],
    ids=[
        "protected-scope",
        "remote-model",
        "paid-cost",
        "budget",
        "market-source",
        "verifier-source",
        "live-extension",
        "prior-same-form-carry-in",
    ],
)
def test_rehashed_protected_scope_model_cost_budget_or_market_changes_fail(
    mutator,
) -> None:
    context = _context()
    forged = copy.deepcopy(context["manifest"])
    mutator(forged)
    forged = _rehash_manifest(forged)
    request = _request_for(context, forged)
    with pytest.raises(SecFilingGemmaStageAccessError, match="exact externally bound"):
        _validate(context, manifest=forged, request=request)


def test_manifest_schema_has_no_result_pass_or_score_field_names() -> None:
    manifest = _context()["manifest"]

    def walk(value) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                tokens = {
                    token
                    for token in re.split(r"[^A-Za-z0-9]+", key.lower())
                    if token
                }
                assert not tokens & {
                    "result",
                    "results",
                    "pass",
                    "passed",
                    "score",
                    "scores",
                }
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(manifest)
