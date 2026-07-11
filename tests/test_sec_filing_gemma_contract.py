from __future__ import annotations

import ast
import copy
from datetime import date, timedelta
import inspect
import json

import pytest

import agent_benchmark.sec_filing_gemma_contract as contract_module
from agent_benchmark.sec_filing_gemma_contract import (
    DIMENSION_NAMES,
    EXTRACTOR_REQUEST_VERSION,
    EXTRACTOR_SCHEMA_VERSION,
    FLAG_NAMES,
    LABEL_MATURITY_OFFSET,
    LIVE_LESSON_SCHEMA_VERSION,
    MANDATORY_IDENTITY_TERMS,
    PREPROCESSOR_VERSION,
    REQUIRED_SOURCE_HASHES,
    SecFilingGemmaContractError,
    authorize_stage_access,
    build_calendar_extension_manifest,
    build_candidate_manifest,
    build_contract_manifest,
    build_corpus_universe_manifest,
    build_extractor_model_payload,
    build_redacted_input_manifest,
    canonical_sha256,
    session_calendar_sha256,
    validate_candidate_manifest,
    validate_complete_corpus,
    validate_contract_manifest,
    validate_corpus_universe_manifest,
    validate_execution_policy,
    validate_final_extraction_coverage,
    validate_extractor_output,
    validate_extractor_request,
    validate_live_lessons,
    validate_final_runtime_summary,
    validate_stage_extraction_coverage,
    validate_training_rows,
)


def _hash(character: str = "a") -> str:
    return character * 64


def _sessions() -> list[str]:
    current = date(2000, 1, 3)
    end = date(2026, 7, 10)
    result: list[str] = []
    while current <= end:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return result


SESSIONS = _sessions()
CALENDAR_SESSIONS_SHA256 = session_calendar_sha256(SESSIONS)


def _offset_session(value: str, offset: int) -> str:
    return SESSIONS[SESSIONS.index(value) + offset]


def _identity_terms() -> list[str]:
    return sorted(
        set(MANDATORY_IDENTITY_TERMS)
        | {
            "luca maestri",
        }
    )


IDENTITY_TERMS = _identity_terms()
IDENTITY_LEXICON_SHA256 = canonical_sha256(IDENTITY_TERMS)


def _sources() -> dict[str, str]:
    characters = "123456789abcdef"
    assert len(REQUIRED_SOURCE_HASHES) <= len(characters)
    return {
        name: _hash(characters[index])
        for index, name in enumerate(REQUIRED_SOURCE_HASHES)
    }


def _source_record(
    year: int,
    serial: int,
    form: str,
    month: int,
    *,
    normalized_text_sha256: str | None = None,
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
        "primary_document_sha256": f"{serial + 2_000:064x}",
        "normalized_text_sha256": normalized_text_sha256
        or f"{serial + 4_000:064x}",
        "source_record_sha256": f"{serial + 6_000:064x}",
    }


def _universe_source_records() -> list[dict]:
    records: list[dict] = []
    serial = 1
    for year in range(2000, 2026):
        for form, month in (("10-K", 2), ("10-Q", 5), ("10-Q", 8), ("10-Q", 11)):
            records.append(_source_record(year, serial, form, month))
            serial += 1
    for form, month in (("10-Q", 2), ("10-Q", 5)):
        records.append(_source_record(2026, serial, form, month))
        serial += 1
    return records


def _universe(records: list[dict] | None = None) -> dict:
    source_records = records or _universe_source_records()
    return build_corpus_universe_manifest(
        catalog_artifact_sha256=_hash("b"),
        calendar_artifact_sha256=_hash("c"),
        catalog_total_record_count=1_000,
        catalog_eligible_record_count=len(source_records),
        session_dates=SESSIONS,
        records=source_records,
    )


def _placeholder_redacted_manifest(universe: dict) -> dict:
    return build_redacted_input_manifest(
        corpus_universe_sha256=universe["universe_sha256"],
        payload_hashes_by_accession={
            record["accession_number"]: record["normalized_text_sha256"]
            for record in universe["records"]
        },
        universe_manifest=universe,
    )


def _candidate(universe: dict, redacted_manifest: dict | None = None) -> dict:
    manifest = redacted_manifest or _placeholder_redacted_manifest(universe)
    return build_candidate_manifest(
        model_digest=_hash("d"),
        ollama_runtime_fingerprint_sha256=_hash("e"),
        sec_audit_checksums_sha256=_hash("f"),
        sec_audit_source_commit="a" * 40,
        calendar_sha256=_hash("c"),
        corpus_universe_sha256=universe["universe_sha256"],
        identity_lexicon_sha256=IDENTITY_LEXICON_SHA256,
        redacted_input_manifest_sha256=manifest["redacted_input_manifest_sha256"],
        reveal_registry_sha256=_hash("7"),
        holdout_attempt_id="aapl-sec-filing-gemma-v1-attempt-001",
        experiment_source_commit="b" * 40,
        source_tree_sha256=_hash("c"),
        source_hashes=_sources(),
    )


def _extractor_context(universe: dict) -> tuple[dict, dict, dict]:
    current = next(
        record
        for record in universe["records"]
        if record["form"] == "10-Q"
        and record["availability_session"].startswith("2001-")
    )
    prior = [
        record
        for record in universe["records"]
        if record["form"] == current["form"]
        and (record["availability_session"], record["accession_number"])
        < (current["availability_session"], current["accession_number"])
    ][-1]
    sentences = [
        {"id": "C0001", "text": "Demand improved while costs declined."},
        {"id": "C0002", "text": "Management described uncertainty as elevated."},
        {"id": "P0001", "text": "Demand had weakened in the prior period."},
    ]
    byte_count = len("\n".join(item["text"] for item in sentences).encode("utf-8"))
    model_payload = build_extractor_model_payload(sentences)
    model_payload_sha256 = canonical_sha256(model_payload)
    payload_hashes = {
        record["accession_number"]: record["normalized_text_sha256"]
        for record in universe["records"]
    }
    payload_hashes[current["accession_number"]] = model_payload_sha256
    redacted_manifest = build_redacted_input_manifest(
        corpus_universe_sha256=universe["universe_sha256"],
        payload_hashes_by_accession=payload_hashes,
        universe_manifest=universe,
    )
    candidate = _candidate(universe, redacted_manifest)
    request = {
        "request_version": EXTRACTOR_REQUEST_VERSION,
        "preprocessor_version": PREPROCESSOR_VERSION,
        "corpus_universe_sha256": universe["universe_sha256"],
        "identity_lexicon_sha256": IDENTITY_LEXICON_SHA256,
        "redacted_input_manifest_sha256": redacted_manifest[
            "redacted_input_manifest_sha256"
        ],
        "stage": current["artifact_stage"],
        "current_accession_number": current["accession_number"],
        "current_form": current["form"],
        "current_availability_session": current["availability_session"],
        "current_filing_sha256": current["normalized_text_sha256"],
        "prior_accession_number": prior["accession_number"],
        "prior_availability_session": prior["availability_session"],
        "prior_same_form_filing_sha256": prior["normalized_text_sha256"],
        "model_payload": model_payload,
        "model_payload_sha256": model_payload_sha256,
        "redaction_report": {
            "identity_matches_remaining": 0,
            "numeric_matches_remaining": 0,
            "date_matches_remaining": 0,
            "forbidden_context_fields": 0,
            "sentence_count": len(sentences),
            "utf8_bytes": byte_count,
            "maximum_sentence_characters": max(len(item["text"]) for item in sentences),
        },
    }
    return request, redacted_manifest, candidate


def _validate_request(
    request: dict,
    universe: dict,
    redacted_manifest: dict,
    candidate: dict,
) -> dict:
    return validate_extractor_request(
        request,
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate["candidate_sha256"],
        universe_manifest=universe,
        session_dates=SESSIONS,
        forbidden_identity_terms=IDENTITY_TERMS,
        redacted_input_manifest=redacted_manifest,
    )


def _replace_first_model_sentence(
    request: dict,
    universe: dict,
    redacted_manifest: dict,
    text: str,
) -> tuple[dict, dict, dict]:
    changed = copy.deepcopy(request)
    user_content = json.loads(changed["model_payload"]["messages"][1]["content"])
    user_content["sentences"][0]["text"] = text
    sentences = user_content["sentences"]
    changed["model_payload"] = build_extractor_model_payload(sentences)
    payload_hash = canonical_sha256(changed["model_payload"])
    changed["model_payload_sha256"] = payload_hash
    changed["redaction_report"]["utf8_bytes"] = len(
        "\n".join(item["text"] for item in sentences).encode("utf-8")
    )
    changed["redaction_report"]["maximum_sentence_characters"] = max(
        len(item["text"]) for item in sentences
    )
    payload_hashes = dict(redacted_manifest["payload_hashes_by_accession"])
    payload_hashes[changed["current_accession_number"]] = payload_hash
    new_manifest = build_redacted_input_manifest(
        corpus_universe_sha256=universe["universe_sha256"],
        payload_hashes_by_accession=payload_hashes,
        universe_manifest=universe,
    )
    changed["redacted_input_manifest_sha256"] = new_manifest[
        "redacted_input_manifest_sha256"
    ]
    return changed, new_manifest, _candidate(universe, new_manifest)


def _extractor_output() -> dict:
    return {
        "schema_version": EXTRACTOR_SCHEMA_VERSION,
        "document_quality": "usable",
        "dimensions": {
            name: {
                "current_impact": "not_stated",
                "change_vs_prior": "not_stated",
                "evidence_sentence_ids": [],
            }
            for name in DIMENSION_NAMES
        },
        "flags": {
            name: {"present": False, "evidence_sentence_ids": []}
            for name in FLAG_NAMES
        },
    }


def _runtime_evidence(candidate: dict, universe: dict) -> dict:
    called = {
        stage: sorted(
            record["accession_number"]
            for record in universe["records"]
            if record["artifact_stage"] == stage
        )
        for stage in ("development", "intermediate", "final")
    }
    return {
        "complete": True,
        "partial": False,
        "candidate_sha256": candidate["candidate_sha256"],
        "corpus_universe_sha256": universe["universe_sha256"],
        "total_elapsed_seconds": 3_000.0,
        "projected_total_seconds": 3_200.0,
        "sec_acquisition_seconds": 600.0,
        "gemma_extraction_seconds": 2_000.0,
        "fit_simulation_sealing_seconds": 400.0,
        "sec_requests": 900,
        "sec_bytes": 1_000_000,
        "sec_hosts": ["data.sec.gov", "www.sec.gov"],
        "model_calls_by_stage": {stage: len(values) for stage, values in called.items()},
        "called_accessions_by_stage": called,
        "model_call_ledger_sha256": _hash("1"),
        "artifact_sha256s": {
            "sec_corpus": _hash("2"),
            "development_extraction": _hash("3"),
            "intermediate_extraction": _hash("4"),
            "final_extraction": _hash("5"),
            "fit_simulation_evaluation": _hash("6"),
        },
        "market_data_sha256s_by_stage": {
            stage: {
                name: f"{50_000 + stage_index * 10 + name_index:064x}"
                for name_index, name in enumerate(
                    ("AAPL", "SPY", "QQQ", "IWM", "VIX", "TNX", "canonical_frame")
                )
            }
            for stage_index, stage in enumerate(
                ("development", "intermediate", "final"), start=1
            )
        },
        "model_endpoint": "http://127.0.0.1:11434/api/chat",
        "model_name": "gemma4:12b",
        "model_digest": candidate["model"]["digest"],
        "model_runtime_fingerprint_sha256": candidate["model"][
            "runtime_fingerprint_sha256"
        ],
        "model_hosts": ["127.0.0.1"],
        "paid_api_calls": 0,
        "estimated_cost_usd": 0.0,
        "pull_attempts": 0,
        "retries": 0,
        "repair_attempts": 0,
    }


def _validate_runtime(evidence: dict, candidate: dict, universe: dict) -> None:
    validate_final_runtime_summary(
        evidence,
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate["candidate_sha256"],
        universe_manifest=universe,
        expected_universe_sha256=universe["universe_sha256"],
        session_dates=SESSIONS,
    )


def _training_rows(universe: dict, candidate: dict, phase: str) -> list[dict]:
    cutoff = "2018-12-31" if phase == "development_fit" else "2023-12-31"
    allowed = {"development"} if phase == "development_fit" else {
        "development",
        "intermediate",
    }
    rows: list[dict] = []
    for index, record in enumerate(universe["records"], start=1):
        if record["artifact_stage"] not in allowed:
            continue
        maturity = _offset_session(
            record["availability_session"], LABEL_MATURITY_OFFSET
        )
        if maturity > cutoff:
            continue
        rows.append(
            {
                "accession_number": record["accession_number"],
                "candidate_sha256": candidate["candidate_sha256"],
                "corpus_universe_sha256": universe["universe_sha256"],
                "artifact_stage": record["artifact_stage"],
                "feature_availability_session": record["availability_session"],
                "label_maturity_session": maturity,
                "horizon_sessions": 20,
                "feature_sha256": f"{10_000 + index:064x}",
                "extraction_output_sha256": f"{20_000 + index:064x}",
                "market_feature_row_sha256": f"{30_000 + index:064x}",
                "label_ledger_row_sha256": f"{40_000 + index:064x}",
                "semantic_available": True,
            }
        )
    return rows


def _validate_training(
    phase: str, rows: list[dict], universe: dict, candidate: dict
) -> str:
    return validate_training_rows(
        phase,
        rows,
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate["candidate_sha256"],
        universe_manifest=universe,
        expected_universe_sha256=universe["universe_sha256"],
        extraction_artifact_sha256=_hash("1"),
        market_data_manifest_sha256=_hash("2"),
        label_ledger_sha256=_hash("3"),
        session_dates=SESSIONS,
        expected_calendar_sessions_sha256=CALENDAR_SESSIONS_SHA256,
    )


def test_contract_is_deterministic_and_freezes_the_real_goal() -> None:
    first = build_contract_manifest()
    second = build_contract_manifest()

    assert first == second
    assert first is not second
    assert validate_contract_manifest(first) == canonical_sha256(first)
    assert first["objective"]["positions"] == ["LONG", "CASH"]
    assert first["objective"]["leverage_allowed"] is False
    assert first["chronology"]["development"]["last_availability_session"] == "2018-12-31"
    assert first["chronology"]["final"]["training_permitted"] is False
    assert first["model"]["role"] == "grounded_structured_text_extractor_only"
    assert first["model"]["parametric_contamination_risk"] == "unresolved_but_bounded"
    assert first["predictor"]["signal_during_active_episode"] == "ignored_no_extension"
    assert len(first["predictor"]["price_regime_features"]) == 24
    assert len(first["predictor"]["market_sentiment_features"]) == 15
    assert first["gates"]["final"]["minimum_periods_beating_ablation"] == 2


def test_any_contract_mutation_is_rejected() -> None:
    manifest = build_contract_manifest()
    manifest["gates"]["final"]["minimum_total_episode_count"] = 1
    with pytest.raises(SecFilingGemmaContractError, match="does not match"):
        validate_contract_manifest(manifest)


def test_stage_access_is_fail_closed_until_authoritative_evidence_exists() -> None:
    with pytest.raises(SecFilingGemmaContractError, match="disabled"):
        authorize_stage_access(
            stage="intermediate",
            candidate_manifest={},
            expected_candidate_sha256=_hash("a"),
            prior_receipts=[],
        )


def test_candidate_binds_model_runtime_universe_lexicon_and_source_closure() -> None:
    universe = _universe()
    candidate = _candidate(universe)
    assert validate_candidate_manifest(
        candidate, expected_candidate_sha256=candidate["candidate_sha256"]
    ) == candidate["candidate_sha256"]

    changed = copy.deepcopy(candidate)
    changed["model"]["runtime_fingerprint_sha256"] = _hash("9")
    with pytest.raises(SecFilingGemmaContractError, match="canonical"):
        validate_candidate_manifest(changed)
    with pytest.raises(SecFilingGemmaContractError, match="externally pinned"):
        validate_candidate_manifest(candidate, expected_candidate_sha256=_hash("9"))


def test_candidate_rejects_missing_transitive_source_hash() -> None:
    universe = _universe()
    sources = _sources()
    sources.pop("ledger")
    with pytest.raises(SecFilingGemmaContractError, match="source_hashes"):
        build_candidate_manifest(
            model_digest=_hash("d"),
            ollama_runtime_fingerprint_sha256=_hash("e"),
            sec_audit_checksums_sha256=_hash("f"),
            sec_audit_source_commit="a" * 40,
            calendar_sha256=_hash("c"),
            corpus_universe_sha256=universe["universe_sha256"],
            identity_lexicon_sha256=IDENTITY_LEXICON_SHA256,
            redacted_input_manifest_sha256=_placeholder_redacted_manifest(universe)[
                "redacted_input_manifest_sha256"
            ],
            reveal_registry_sha256=_hash("7"),
            holdout_attempt_id="aapl-sec-filing-gemma-v1-attempt-001",
            experiment_source_commit="b" * 40,
            source_tree_sha256=_hash("c"),
            source_hashes=sources,
        )


def test_authoritative_universe_recomputes_availability_and_passes_coverage() -> None:
    universe = _universe()
    assert validate_corpus_universe_manifest(
        universe,
        session_dates=SESSIONS,
        expected_universe_sha256=universe["universe_sha256"],
    ) == universe["universe_sha256"]
    assert universe["stage_counts"] == {
        "development": 76,
        "intermediate": 20,
        "final": 10,
    }
    assert universe["exact_acceptance_timestamp_rate"] == 1.0
    assert all(
        date.fromisoformat(record["availability_session"]).weekday() < 5
        for record in universe["records"]
    )


def test_universe_rejects_non_apple_and_cross_stage_document_reuse() -> None:
    records = _universe_source_records()
    records[0]["subject_cik"] = "0000000001"
    with pytest.raises(SecFilingGemmaContractError, match="subject CIK"):
        _universe(records)

    records = _universe_source_records()
    records[-1]["normalized_text_sha256"] = records[0]["normalized_text_sha256"]
    with pytest.raises(SecFilingGemmaContractError, match="different stage"):
        _universe(records)


def test_shortened_universe_cannot_match_the_external_pin() -> None:
    original = _universe()
    shortened = _universe(_universe_source_records()[:-1])
    with pytest.raises(SecFilingGemmaContractError, match="external pin"):
        validate_corpus_universe_manifest(
            shortened,
            session_dates=SESSIONS,
            expected_universe_sha256=original["universe_sha256"],
        )


def test_forged_availability_and_missing_yearly_form_fail() -> None:
    universe = _universe()
    forged = copy.deepcopy(universe)
    forged["records"][0]["availability_session"] = "2000-02-20"
    with pytest.raises(SecFilingGemmaContractError, match="not canonical"):
        validate_corpus_universe_manifest(
            forged,
            session_dates=SESSIONS,
            expected_universe_sha256=universe["universe_sha256"],
        )

    missing_k = [
        record
        for record in _universe_source_records()
        if not (record["filing_date"].startswith("2005-") and record["form"] == "10-K")
    ]
    incomplete = _universe(missing_k)
    with pytest.raises(SecFilingGemmaContractError, match="Completed year 2005"):
        validate_complete_corpus(incomplete)


def test_valid_extractor_request_is_bound_to_exact_current_and_prior_filings() -> None:
    universe = _universe()
    request, redacted_manifest, candidate = _extractor_context(universe)
    validated = _validate_request(request, universe, redacted_manifest, candidate)
    assert validated["sentence_ids"] == ("C0001", "C0002", "P0001")
    assert set(validated["model_payload"]) == {
        "model",
        "messages",
        "format",
        "stream",
        "think",
        "options",
    }


def test_extractor_rejects_stage_relabel_and_wrong_or_self_prior() -> None:
    universe = _universe()
    request, redacted_manifest, candidate = _extractor_context(universe)
    request["stage"] = "final"
    with pytest.raises(SecFilingGemmaContractError, match="stage boundary"):
        _validate_request(request, universe, redacted_manifest, candidate)

    request, redacted_manifest, candidate = _extractor_context(universe)
    request["prior_same_form_filing_sha256"] = request["current_filing_sha256"]
    with pytest.raises(SecFilingGemmaContractError, match="immediately preceding"):
        _validate_request(request, universe, redacted_manifest, candidate)


@pytest.mark.parametrize(
    "text, message",
    [
        (
            "Luca Maestri discussed AirPods and iOS in Cupertino while revenue rose.",
            "identity",
        ),
        ("App Store and Mac sales improved in Greater China.", "identity"),
        ("Revenue exceeded twelve billion dollars.", "Spelled absolute"),
        ("Revenue increased by 12 percent.", "Absolute numeric"),
        ("The period ended September thirtieth while demand improved.", "Spelled date"),
    ],
)
def test_extractor_rejects_identity_and_absolute_value_leakage(
    text: str, message: str
) -> None:
    universe = _universe()
    request, redacted_manifest, candidate = _extractor_context(universe)
    request, redacted_manifest, candidate = _replace_first_model_sentence(
        request, universe, redacted_manifest, text
    )
    with pytest.raises(SecFilingGemmaContractError, match=message):
        _validate_request(request, universe, redacted_manifest, candidate)


def test_identity_and_date_scanners_do_not_destroy_directional_risk_language() -> None:
    universe = _universe()
    request, redacted_manifest, _ = _extractor_context(universe)
    request, redacted_manifest, candidate = _replace_first_model_sentence(
        request,
        universe,
        redacted_manifest,
        "Results may vary as macroeconomic ratios deteriorate.",
    )
    assert _validate_request(request, universe, redacted_manifest, candidate)[
        "model_payload_sha256"
    ] == request["model_payload_sha256"]


def test_metadata_cannot_be_injected_into_the_model_payload() -> None:
    universe = _universe()
    request, redacted_manifest, candidate = _extractor_context(universe)
    request["model_payload"]["stage"] = "development"
    request["model_payload_sha256"] = canonical_sha256(request["model_payload"])
    with pytest.raises(SecFilingGemmaContractError, match="metadata"):
        _validate_request(request, universe, redacted_manifest, candidate)


def test_valid_grounded_extractor_output_and_evidence() -> None:
    output = _extractor_output()
    output["dimensions"]["demand"] = {
        "current_impact": "favorable",
        "change_vs_prior": "improving",
        "evidence_sentence_ids": ["C0001", "P0001"],
    }
    output["flags"]["management_transition"] = {
        "present": True,
        "evidence_sentence_ids": ["C0002"],
    }
    validate_extractor_output(
        output, supplied_sentence_ids=("C0001", "C0002", "P0001")
    )


def test_extractor_output_rejects_outcomes_and_unsupported_claims() -> None:
    output = _extractor_output()
    output["predicted_return"] = 0.1
    with pytest.raises(SecFilingGemmaContractError, match="extra"):
        validate_extractor_output(output, supplied_sentence_ids=("C0001", "P0001"))

    output = _extractor_output()
    output["dimensions"]["demand"] = {
        "current_impact": "favorable",
        "change_vs_prior": "improving",
        "evidence_sentence_ids": ["C0001"],
    }
    with pytest.raises(SecFilingGemmaContractError, match="current and prior"):
        validate_extractor_output(output, supplied_sentence_ids=("C0001",))

    output = _extractor_output()
    output["flags"]["liquidity_stress"]["evidence_sentence_ids"] = ["C0001"]
    with pytest.raises(SecFilingGemmaContractError, match="False flag"):
        validate_extractor_output(output, supplied_sentence_ids=("C0001",))


def test_extraction_coverage_is_exact_and_stage_bounded() -> None:
    universe = _universe()
    qualities = {
        record["accession_number"]: "usable" for record in universe["records"]
    }
    validate_final_extraction_coverage(qualities, universe_manifest=universe)

    development_qualities = {
        record["accession_number"]: "usable"
        for record in universe["records"]
        if record["artifact_stage"] == "development"
    }
    validate_stage_extraction_coverage(
        "development", development_qualities, universe_manifest=universe
    )

    development_accessions = [
        record["accession_number"]
        for record in universe["records"]
        if record["artifact_stage"] == "development"
    ]
    for accession in development_accessions[:8]:
        qualities[accession] = "invalid"
        development_qualities[accession] = "invalid"
    with pytest.raises(SecFilingGemmaContractError, match="development"):
        validate_stage_extraction_coverage(
            "development", development_qualities, universe_manifest=universe
        )


def test_training_rows_require_exact_pinned_session_maturity_and_cutoff() -> None:
    universe = _universe()
    candidate = _candidate(universe)
    rows = _training_rows(universe, candidate, "development_fit")
    assert _validate_training("development_fit", rows, universe, candidate)

    changed = copy.deepcopy(rows)
    changed[0]["label_maturity_session"] = _offset_session(
        changed[0]["feature_availability_session"], 1
    )
    with pytest.raises(SecFilingGemmaContractError, match="exact 20-session"):
        _validate_training("development_fit", changed, universe, candidate)

    omitted = rows[:-1]
    with pytest.raises(SecFilingGemmaContractError, match="omit"):
        _validate_training("development_fit", omitted, universe, candidate)

    arbitrary = copy.deepcopy(rows)
    arbitrary[0]["accession_number"] = "not-a-filing"
    with pytest.raises(SecFilingGemmaContractError, match="exact unique"):
        _validate_training("development_fit", arbitrary, universe, candidate)

    with pytest.raises(SecFilingGemmaContractError, match="forbidden"):
        validate_training_rows(
            "final",
            rows,
            candidate_manifest=candidate,
            expected_candidate_sha256=candidate["candidate_sha256"],
            universe_manifest=universe,
            expected_universe_sha256=universe["universe_sha256"],
            extraction_artifact_sha256=_hash("1"),
            market_data_manifest_sha256=_hash("2"),
            label_ledger_sha256=_hash("3"),
            session_dates=SESSIONS,
            expected_calendar_sessions_sha256=CALENDAR_SESSIONS_SHA256,
        )


def test_runtime_evidence_reconciles_every_filing_and_bound_artifact() -> None:
    universe = _universe()
    candidate = _candidate(universe)
    evidence = _runtime_evidence(candidate, universe)
    _validate_runtime(evidence, candidate, universe)

    zero_work = copy.deepcopy(evidence)
    zero_work["sec_requests"] = 0
    zero_work["model_calls_by_stage"] = {
        "development": 0,
        "intermediate": 0,
        "final": 0,
    }
    zero_work["called_accessions_by_stage"] = {
        "development": [],
        "intermediate": [],
        "final": [],
    }
    with pytest.raises(SecFilingGemmaContractError):
        _validate_runtime(zero_work, candidate, universe)

    wrong_model = copy.deepcopy(evidence)
    wrong_model["model_digest"] = _hash("9")
    with pytest.raises(SecFilingGemmaContractError, match="model digest"):
        _validate_runtime(wrong_model, candidate, universe)

    impossible_timing = copy.deepcopy(evidence)
    impossible_timing["total_elapsed_seconds"] = 2_000
    with pytest.raises(SecFilingGemmaContractError, match="phase seconds"):
        _validate_runtime(impossible_timing, candidate, universe)


def test_runtime_rejects_paid_external_or_over_budget_work() -> None:
    universe = _universe()
    candidate = _candidate(universe)
    evidence = _runtime_evidence(candidate, universe)
    for field, value in (
        ("paid_api_calls", 1),
        ("estimated_cost_usd", 0.01),
        ("pull_attempts", 1),
        ("total_elapsed_seconds", 3_601),
    ):
        changed = copy.deepcopy(evidence)
        changed[field] = value
        with pytest.raises(SecFilingGemmaContractError):
            _validate_runtime(changed, candidate, universe)

    external = copy.deepcopy(evidence)
    external["model_endpoint"] = "https://api.example.com/chat"
    external["model_hosts"] = ["api.example.com"]
    with pytest.raises(SecFilingGemmaContractError, match="loopback"):
        _validate_runtime(external, candidate, universe)


def test_execution_policy_rejects_any_leverage_related_change() -> None:
    policy = build_contract_manifest()["execution"]
    validate_execution_policy(policy)
    policy["binary_target_only"] = False
    with pytest.raises(SecFilingGemmaContractError, match="policy changed"):
        validate_execution_policy(policy)


def _live_lesson(*, edge: float = 0.01, beats: bool = True) -> tuple[dict, str]:
    decision = "2026-01-02"
    maturity = _offset_session(decision, LABEL_MATURITY_OFFSET)
    first_use = _offset_session(maturity, 1)
    body = {
        "schema_version": LIVE_LESSON_SCHEMA_VERSION,
        "sequence_number": 1,
        "lesson_id": "lesson-one",
        "accession_number": "0000320193-26-000001",
        "decision_session": decision,
        "label_maturity_session": maturity,
        "ingested_session": maturity,
        "first_eligible_decision_session": first_use,
        "horizon_sessions": 20,
        "feature_sha256": _hash("7"),
        "prediction_receipt_sha256": _hash("1"),
        "market_data_manifest_sha256": _hash("2"),
        "strategy_ledger_slice_sha256": _hash("3"),
        "benchmark_ledger_slice_sha256": _hash("4"),
        "cash_beats_long_10bps": beats,
        "cash_active_log_edge_10bps": edge,
        "parent_lesson_sha256": None,
        "input_model_state_sha256": _hash("8"),
        "output_model_state_sha256": _hash("9"),
    }
    return {**body, "lesson_sha256": canonical_sha256(body)}, _offset_session(first_use, 5)


def test_live_lessons_have_exact_maturity_label_and_state_hash_chain() -> None:
    lesson, as_of = _live_lesson()
    eligible_ids = ["lesson-one"]
    anchor = validate_live_lessons(
        [lesson],
        as_of_session=as_of,
        session_dates=SESSIONS,
        expected_calendar_sessions_sha256=CALENDAR_SESSIONS_SHA256,
        expected_prior_tip_sha256=None,
        expected_prior_count=0,
        expected_prior_model_state_sha256=_hash("8"),
        expected_prior_last_sort_key=None,
        eligible_lesson_ids=eligible_ids,
        expected_eligible_lesson_set_sha256=canonical_sha256(eligible_ids),
    )
    assert anchor["lesson_count"] == 1
    assert anchor["tip_sha256"] == lesson["lesson_sha256"]

    unchanged = validate_live_lessons(
        [],
        as_of_session=as_of,
        session_dates=SESSIONS,
        expected_calendar_sessions_sha256=CALENDAR_SESSIONS_SHA256,
        expected_prior_tip_sha256=anchor["tip_sha256"],
        expected_prior_count=anchor["lesson_count"],
        expected_prior_model_state_sha256=anchor["model_state_sha256"],
        expected_prior_last_sort_key=anchor["last_sort_key"],
        eligible_lesson_ids=[],
        expected_eligible_lesson_set_sha256=canonical_sha256([]),
    )
    assert unchanged["lesson_count"] == 1
    assert unchanged["tip_sha256"] == anchor["tip_sha256"]

    backdated = copy.deepcopy(lesson)
    backdated.update(
        {
            "sequence_number": 2,
            "lesson_id": "lesson-two",
            "decision_session": "2001-01-03",
            "label_maturity_session": _offset_session(
                "2001-01-03", LABEL_MATURITY_OFFSET
            ),
            "parent_lesson_sha256": anchor["tip_sha256"],
            "input_model_state_sha256": anchor["model_state_sha256"],
            "output_model_state_sha256": _hash("a"),
        }
    )
    backdated["ingested_session"] = backdated["label_maturity_session"]
    backdated["first_eligible_decision_session"] = _offset_session(
        backdated["ingested_session"], 1
    )
    backdated["lesson_sha256"] = canonical_sha256(
        {key: value for key, value in backdated.items() if key != "lesson_sha256"}
    )
    with pytest.raises(SecFilingGemmaContractError, match="chronological"):
        validate_live_lessons(
            [backdated],
            as_of_session=as_of,
            session_dates=SESSIONS,
            expected_calendar_sessions_sha256=CALENDAR_SESSIONS_SHA256,
            expected_prior_tip_sha256=anchor["tip_sha256"],
            expected_prior_count=anchor["lesson_count"],
            expected_prior_model_state_sha256=anchor["model_state_sha256"],
            expected_prior_last_sort_key=anchor["last_sort_key"],
            eligible_lesson_ids=["lesson-two"],
            expected_eligible_lesson_set_sha256=canonical_sha256(["lesson-two"]),
        )

    wrong_label, as_of = _live_lesson(edge=-0.01, beats=True)
    with pytest.raises(SecFilingGemmaContractError, match="Boolean"):
        validate_live_lessons(
            [wrong_label],
            as_of_session=as_of,
            session_dates=SESSIONS,
            expected_calendar_sessions_sha256=CALENDAR_SESSIONS_SHA256,
            expected_prior_tip_sha256=None,
            expected_prior_count=0,
            expected_prior_model_state_sha256=_hash("8"),
            expected_prior_last_sort_key=None,
            eligible_lesson_ids=eligible_ids,
            expected_eligible_lesson_set_sha256=canonical_sha256(eligible_ids),
        )

    premature, as_of = _live_lesson()
    premature["label_maturity_session"] = _offset_session(
        premature["decision_session"], 1
    )
    premature["lesson_sha256"] = canonical_sha256(
        {key: value for key, value in premature.items() if key != "lesson_sha256"}
    )
    with pytest.raises(SecFilingGemmaContractError, match="exact 20-session"):
        validate_live_lessons(
            [premature],
            as_of_session=as_of,
            session_dates=SESSIONS,
            expected_calendar_sessions_sha256=CALENDAR_SESSIONS_SHA256,
            expected_prior_tip_sha256=None,
            expected_prior_count=0,
            expected_prior_model_state_sha256=_hash("8"),
            expected_prior_last_sort_key=None,
            eligible_lesson_ids=eligible_ids,
            expected_eligible_lesson_set_sha256=canonical_sha256(eligible_ids),
        )


def test_future_calendar_extension_preserves_the_complete_historical_prefix() -> None:
    legacy = [value for value in SESSIONS if value <= "2025-01-10"]
    legacy_manifest = build_calendar_extension_manifest(
        prior_session_dates=legacy,
        extended_session_dates=SESSIONS,
        expected_prior_sessions_sha256=canonical_sha256(legacy),
    )
    assert legacy_manifest["prior_last_session"] == "2025-01-10"
    assert legacy_manifest["extended_last_session"] == "2026-07-10"

    extended = list(SESSIONS)
    current = date.fromisoformat(SESSIONS[-1]) + timedelta(days=1)
    while len(extended) < len(SESSIONS) + 5:
        if current.weekday() < 5:
            extended.append(current.isoformat())
        current += timedelta(days=1)
    manifest = build_calendar_extension_manifest(
        prior_session_dates=SESSIONS,
        extended_session_dates=extended,
        expected_prior_sessions_sha256=CALENDAR_SESSIONS_SHA256,
    )
    assert manifest["prior_session_count"] == len(SESSIONS)
    assert manifest["extended_session_count"] == len(SESSIONS) + 5

    rewritten = list(extended)
    rewritten[100] = rewritten[100 + 1]
    with pytest.raises(SecFilingGemmaContractError):
        build_calendar_extension_manifest(
            prior_session_dates=SESSIONS,
            extended_session_dates=rewritten,
            expected_prior_sessions_sha256=CALENDAR_SESSIONS_SHA256,
        )


def test_contract_module_imports_only_effect_free_standard_library_modules() -> None:
    tree = ast.parse(inspect.getsource(contract_module))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__",
        "collections",
        "copy",
        "datetime",
        "hashlib",
        "hmac",
        "json",
        "math",
        "re",
        "typing",
    }
