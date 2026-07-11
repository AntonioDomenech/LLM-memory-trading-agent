from __future__ import annotations

import copy
from datetime import date
from functools import lru_cache
import hashlib
import re
from types import MappingProxyType

import pytest

from agent_benchmark.sec_filing_gemma_contract import (
    CONTRACT_VERSION,
    REQUIRED_SOURCE_HASHES,
    build_candidate_manifest,
    build_corpus_universe_manifest,
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
    MODEL_ENDPOINT,
    STAGE_ACCESS_MANIFEST_SCHEMA_VERSION,
    SecFilingGemmaStageAccessError,
    build_stage_access_manifest,
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
        identity_lexicon_sha256=_h("identity-lexicon"),
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
    assert stage not in manifest["scope"]["prohibited_stages"]
    assert set(manifest["scope"]["prohibited_stages"]) == {
        value for value in ("development", "intermediate", "final") if value != stage
    }


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
    ],
    ids=[
        "protected-scope",
        "remote-model",
        "paid-cost",
        "budget",
        "market-source",
        "verifier-source",
        "live-extension",
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
