from __future__ import annotations

import ast
import copy
from datetime import date, timedelta
import inspect
import json

import pytest

import agent_benchmark.sec_filing_gemma_contract as contract_module
import agent_benchmark.sec_filing_gemma_extractor_prompt as extractor_prompt_module
import agent_benchmark.sec_filing_gemma_extractor_schema as extractor_schema_module
from agent_benchmark.sec_filing_gemma_contract import (
    CALENDAR_SOURCE_URLS,
    CANONICAL_IDENTITY_LEXICON,
    CANONICAL_IDENTITY_LEXICON_SHA256,
    DIMENSION_NAMES,
    EXTRACTOR_REQUEST_VERSION,
    EXTRACTOR_SCHEMA_VERSION,
    FLAG_NAMES,
    LABEL_MATURITY_OFFSET,
    LIVE_LESSON_SCHEMA_VERSION,
    PREPROCESSOR_VERSION,
    REDACTED_INPUT_SCHEMA_VERSION,
    REQUIRED_SOURCE_HASHES,
    SecFilingGemmaContractError,
    authorize_stage_access,
    build_calendar_extension_manifest,
    build_calendar_source_evidence_manifest,
    build_candidate_manifest,
    build_contract_manifest,
    build_corpus_universe_manifest,
    build_extractor_model_payload,
    build_redacted_input_manifest,
    build_stage_content_manifest,
    canonical_sha256,
    session_calendar_sha256,
    validate_candidate_manifest,
    validate_calendar_extension_manifest,
    validate_calendar_source_evidence_manifest,
    validate_complete_corpus,
    validate_contract_manifest,
    validate_corpus_universe_manifest,
    validate_execution_policy,
    validate_final_extraction_coverage,
    validate_extractor_output,
    validate_extractor_request,
    validate_live_lessons,
    validate_redacted_input_manifest,
    validate_final_runtime_summary,
    validate_stage_extraction_coverage,
    validate_stage_content_manifest,
    validate_training_rows,
)
from agent_benchmark.sec_session_calendar import (
    EXPECTED_SESSIONS,
    EXPECTED_MARKET_HISTORY_SESSIONS,
    LEGACY_EXPECTED_SESSIONS,
)


def _hash(character: str = "a") -> str:
    return character * 64


def _sessions() -> list[str]:
    return list(EXPECTED_SESSIONS)


SESSIONS = _sessions()
CALENDAR_SESSIONS_SHA256 = session_calendar_sha256(SESSIONS)


def _offset_session(value: str, offset: int) -> str:
    return SESSIONS[SESSIONS.index(value) + offset]


IDENTITY_TERMS = list(CANONICAL_IDENTITY_LEXICON)
IDENTITY_LEXICON_SHA256 = CANONICAL_IDENTITY_LEXICON_SHA256


def _sources() -> dict[str, str]:
    return {
        name: f"{index + 1:064x}"
        for index, name in enumerate(REQUIRED_SOURCE_HASHES)
    }


def _source_record(
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


def _stage_content(universe: dict, stage: str) -> dict:
    documents = []
    for index, record in enumerate(universe["records"], start=1):
        if record["artifact_stage"] != stage:
            continue
        documents.append(
            {
                "accession_number": record["accession_number"],
                "primary_document_sha256": f"{20_000 + index:064x}",
                "normalized_text_sha256": f"{30_000 + index:064x}",
                "primary_document_bytes": 10_000 + index,
                "normalized_text_bytes": 8_000 + index,
            }
        )
    return build_stage_content_manifest(
        artifact_stage=stage,
        corpus_universe_sha256=universe["universe_sha256"],
        documents=documents,
        universe_manifest=universe,
    )


def _candidate(
    universe: dict,
    *,
    calendar_source_evidence_sha256: str | None = None,
    catalog_sha256: str | None = None,
    identity_lexicon_sha256: str = IDENTITY_LEXICON_SHA256,
) -> dict:
    return build_candidate_manifest(
        model_digest=_hash("d"),
        ollama_runtime_fingerprint_sha256=_hash("e"),
        sec_audit_checksums_json_sha256=_hash("f"),
        sec_catalog_artifact_sha256=(
            universe["catalog_artifact_sha256"]
            if catalog_sha256 is None
            else catalog_sha256
        ),
        sec_audit_source_commit="a" * 40,
        calendar_source_evidence_sha256=(
            universe["calendar_artifact_sha256"]
            if calendar_source_evidence_sha256 is None
            else calendar_source_evidence_sha256
        ),
        calendar_sessions_sha256=CALENDAR_SESSIONS_SHA256,
        corpus_universe_sha256=universe["universe_sha256"],
        corpus_universe_semantic_sha256=universe["universe_semantic_sha256"],
        identity_lexicon_sha256=identity_lexicon_sha256,
        predecessor_reveal_registry_sha256=_hash("7"),
        holdout_attempt_id="aapl-sec-filing-gemma-v1-attempt-001",
        experiment_source_commit="b" * 40,
        source_tree_sha256=_hash("c"),
        source_hashes=_sources(),
    )


def _extractor_context(universe: dict) -> tuple[dict, dict, dict, dict]:
    content_manifest = _stage_content(universe, "development")
    content_by_accession = {
        document["accession_number"]: document
        for document in content_manifest["documents"]
    }
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
    redacted_manifest = build_redacted_input_manifest(
        artifact_stage=current["artifact_stage"],
        accession_number=current["accession_number"],
        corpus_universe_sha256=universe["universe_sha256"],
        model_payload_sha256=model_payload_sha256,
        preprocessed_event_sha256=_hash("1"),
        owned_preprocessing_receipt_sha256=_hash("2"),
        sec_reader_receipt_sha256=_hash("3"),
        carry_in_reader_receipt_sha256=None,
        universe_manifest=universe,
        stage_content_manifest=content_manifest,
    )
    candidate = _candidate(universe)
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
        "current_filing_sha256": content_by_accession[current["accession_number"]][
            "normalized_text_sha256"
        ],
        "prior_accession_number": prior["accession_number"],
        "prior_availability_session": prior["availability_session"],
        "prior_same_form_filing_sha256": content_by_accession[prior["accession_number"]][
            "normalized_text_sha256"
        ],
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
    return request, redacted_manifest, candidate, {"development": content_manifest}


def _validate_request(
    request: dict,
    universe: dict,
    redacted_manifest: dict,
    candidate: dict,
    content_manifests: dict,
) -> dict:
    return validate_extractor_request(
        request,
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate["candidate_sha256"],
        universe_manifest=universe,
        content_manifests_by_stage=content_manifests,
        expected_content_manifest_sha256s={
            stage: manifest["content_manifest_sha256"]
            for stage, manifest in content_manifests.items()
        },
        session_dates=SESSIONS,
        forbidden_identity_terms=IDENTITY_TERMS,
        redacted_input_manifest=redacted_manifest,
        expected_redacted_input_manifest_sha256=redacted_manifest[
            "redacted_input_manifest_sha256"
        ],
        expected_preprocessed_event_sha256=_hash("1"),
        expected_owned_preprocessing_receipt_sha256=_hash("2"),
        expected_sec_reader_receipt_sha256=_hash("3"),
        expected_carry_in_reader_receipt_sha256=None,
    )


def _replace_first_model_sentence(
    request: dict,
    universe: dict,
    redacted_manifest: dict,
    content_manifests: dict,
    text: str,
) -> tuple[dict, dict, dict, dict]:
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
    new_manifest = build_redacted_input_manifest(
        artifact_stage=redacted_manifest["artifact_stage"],
        accession_number=changed["current_accession_number"],
        corpus_universe_sha256=universe["universe_sha256"],
        model_payload_sha256=payload_hash,
        preprocessed_event_sha256=redacted_manifest[
            "preprocessed_event_sha256"
        ],
        owned_preprocessing_receipt_sha256=redacted_manifest[
            "owned_preprocessing_receipt_sha256"
        ],
        sec_reader_receipt_sha256=redacted_manifest[
            "sec_reader_receipt_sha256"
        ],
        carry_in_reader_receipt_sha256=redacted_manifest[
            "carry_in_reader_receipt_sha256"
        ],
        universe_manifest=universe,
        stage_content_manifest=content_manifests[redacted_manifest["artifact_stage"]],
    )
    changed["redacted_input_manifest_sha256"] = new_manifest[
        "redacted_input_manifest_sha256"
    ]
    return changed, new_manifest, _candidate(universe), content_manifests


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
                "feature_row_sha256": f"{10_000 + index:064x}",
                "extraction_identity_sha256": f"{20_000 + index:064x}",
                "market_prefix_chain_identity_sha256": f"{25_000 + index:064x}",
                "market_feature_row_sha256": f"{30_000 + index:064x}",
                "label_evidence_sha256": f"{40_000 + index:064x}",
                "semantic_available": True,
                "market_available": True,
                "prediction_available": True,
                "fit_eligible": True,
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
    assert first["chronology"]["calendar_id"].endswith("2026_07_10_v2")
    assert first["model"]["role"] == "grounded_structured_text_extractor_only"
    assert first["model"]["parametric_contamination_risk"] == "unresolved_but_bounded"
    assert first["predictor"]["signal_during_active_episode"] == "ignored_no_extension"
    assert first["predictor"]["market_history_start"] == "1998-01-01"
    assert first["predictor"]["market_history_calendar_id"].endswith(
        "1998_01_01_2026_07_10_v1"
    )
    assert sum(value < "2000-01-01" for value in EXPECTED_MARKET_HISTORY_SESSIONS) >= 252
    assert first["predictor"]["maximum_market_lookback_sessions"] == 252
    assert (
        first["predictor"]["market_context_timing"]
        == "completed_decision_session_close_only"
    )
    assert first["predictor"]["prediction_timeline"]["fill_session_offset"] == 1
    assert (
        first["predictor"]["prediction_timeline"][
            "cash_exit_and_label_maturity_session_offset"
        ]
        == 21
    )
    assert all(
        fold["state_updates_inside_test_window"] is False
        for fold in first["predictor"]["development_folds"]
    )
    assert all(
        fold["train_label_maturity_through"] in EXPECTED_SESSIONS
        and fold["test_first_session"] in EXPECTED_SESSIONS
        and fold["test_last_session"] in EXPECTED_SESSIONS
        for fold in first["predictor"]["development_folds"]
    )
    assert first["predictor"]["binary_cash_win_target"] == {
        "cost_bps": 10,
        "positive_if": "cash_active_log_edge_strictly_above_comparison_tolerance",
        "comparison_tolerance": 1e-12,
        "used_by": [
            "probability_head",
            "brier_score",
            "causal_climatology",
            "episode_win_rate",
            "live_lessons",
        ],
    }
    learner = first["predictor"]["learner_configuration"]
    assert learner["feature_selection"] is False
    assert learner["interactions"] is False
    assert learner["intercept_regularized"] is False
    assert learner["frozen_numerical_constants"] == {
        "raw_mad_multiplier": 1.4826,
        "raw_scale_floor": 1e-6,
        "raw_z_clip": 4.0,
        "ridge_lambda": 0.1,
        "logistic_max_iterations": 50,
        "logistic_tolerance": 1e-10,
        "newton_line_search_max_steps": 50,
        "newton_armijo_constant": 1e-4,
        "logistic_curvature_floor": 1e-15,
        "huber_delta": 1.5,
        "huber_max_iterations": 50,
        "huber_tolerance": 1e-10,
        "target_mad_multiplier": 1.4826,
        "target_scale_floor": 1e-6,
        "edge_clip_lower": -0.5,
        "edge_clip_upper": 0.5,
    }
    assert "reveal_registry" in REQUIRED_SOURCE_HASHES
    assert "stage_verifier" in REQUIRED_SOURCE_HASHES
    assert "market_source_bytes" in REQUIRED_SOURCE_HASHES
    assert "market_acquirer" in REQUIRED_SOURCE_HASHES
    assert "stage_access" in REQUIRED_SOURCE_HASHES
    assert "training_membership" in REQUIRED_SOURCE_HASHES
    assert "learner_prediction" in REQUIRED_SOURCE_HASHES
    assert {
        "cftc_cot_policy",
        "content_normalizer",
        "deterministic_aapl",
        "direct_edge_features",
        "downside_features",
        "market_evidence",
        "no_leverage",
        "package_init",
        "prediction_evidence",
        "reveal_store",
        "sec_audit_artifact",
        "sec_audit_evaluation",
        "sec_audit_plan",
        "sec_audit_selection",
        "sec_point_in_time",
        "source_identity",
        "stage_authorization",
        "unleveraged_aapl",
    }.issubset(REQUIRED_SOURCE_HASHES)
    assert (
        first["holdout_governance"][
            "candidate_binds_predecessor_registry_snapshot"
        ]
        is True
    )
    assert first["holdout_governance"][
        "required_semantic_prerequisite_checks"
    ] == list(contract_module.REQUIRED_STAGE_VERIFIER_CHECKS)
    assert len(first["predictor"]["price_regime_features"]) == 24
    assert len(first["predictor"]["market_sentiment_features"]) == 15
    feature_semantics = first["predictor"]["feature_semantics"]
    assert feature_semantics["adverse_flag_names"] == list(
        contract_module.FLAG_NAMES[:-1]
    )
    assert feature_semantics[
        "adverse_flag_count_excludes_management_transition"
    ] is True
    assert feature_semantics["management_transition_is_separate_feature"] is True
    assert feature_semantics["invalid_model_output"].startswith("neutral_semantics")
    assert feature_semantics[
        "missing_or_unauthenticated_extraction_evidence"
    ] == "prediction_unavailable_integrity_failure"
    assert feature_semantics["label_cost_application"].endswith(
        "entry_and_exit_position_changing_fills"
    )
    assert first["model"]["completed_call_attempt_receipt"].startswith(
        "exact_request_response_and_output_bytes"
    )
    assert first["model"]["invalid_attempt_semantics"].startswith(
        "sealed_invalid_status"
    )
    assert first["model"]["per_call_client_receipt_trust"] == (
        "unattested_until_stage_runner_replay"
    )
    assert first["model"]["runtime_identity_guard"].startswith(
        "candidate_pins_checked_immediately_before_and_after"
    )
    assert first["gates"]["final"]["minimum_periods_beating_ablation"] == 2


def test_contract_reexports_owned_extractor_prompt_schema_and_exact_payload() -> None:
    assert (
        contract_module.EXTRACTOR_SYSTEM_PROMPT
        is extractor_prompt_module.EXTRACTOR_SYSTEM_PROMPT
    )
    assert (
        contract_module.build_extractor_json_schema
        is extractor_schema_module.build_extractor_json_schema
    )
    assert contract_module.EXTRACTOR_SCHEMA_VERSION == (
        extractor_schema_module.EXTRACTOR_SCHEMA_VERSION
    )
    assert contract_module.DIMENSION_NAMES is extractor_schema_module.DIMENSION_NAMES
    assert contract_module.FLAG_NAMES is extractor_schema_module.FLAG_NAMES
    assert (
        contract_module.ADVERSE_FLAG_NAMES
        is extractor_schema_module.ADVERSE_FLAG_NAMES
    )
    assert contract_module.CURRENT_IMPACTS is extractor_schema_module.CURRENT_IMPACTS
    assert (
        contract_module.COMPARATIVE_CHANGES
        is extractor_schema_module.COMPARATIVE_CHANGES
    )
    assert (
        contract_module.DOCUMENT_QUALITIES
        is extractor_schema_module.DOCUMENT_QUALITIES
    )

    sentences = [
        {"id": "C0001", "text": "Demand improved while costs declined."},
        {"id": "P0001", "text": "Demand had weakened in the prior period."},
    ]
    payload = build_extractor_model_payload(sentences)
    assert payload["messages"][0] == {
        "role": "system",
        "content": extractor_prompt_module.EXTRACTOR_SYSTEM_PROMPT,
    }
    assert payload["format"] == extractor_schema_module.build_extractor_json_schema()

    # Every call returns detached schema containers; a caller cannot mutate the
    # next request or the owned enum constants through a prior return value.
    payload["format"]["properties"]["dimensions"]["required"].append("forged")
    replay = build_extractor_model_payload(sentences)
    assert "forged" not in replay["format"]["properties"]["dimensions"]["required"]


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


def test_intermediate_structural_receipt_cannot_expose_partial_results() -> None:
    with pytest.raises(SecFilingGemmaContractError, match="cannot be exposed partially"):
        contract_module._build_structural_stage_receipt(
            stage="intermediate",
            candidate_sha256=_hash("1"),
            corpus_manifest_sha256=_hash("2"),
            extraction_artifact_sha256=_hash("3"),
            input_model_state_sha256=_hash("4"),
            output_model_state_sha256=_hash("5"),
            selected_candidate_id="p50_e0",
            passed=True,
            parent_receipt_sha256=_hash("6"),
            design_change_count=0,
            partial_results_exposed=True,
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

    with pytest.raises(SecFilingGemmaContractError, match="frozen production lexicon"):
        _candidate(universe, identity_lexicon_sha256=_hash("9"))


def test_candidate_rejects_missing_transitive_source_hash() -> None:
    universe = _universe()
    sources = _sources()
    sources.pop("ledger")
    with pytest.raises(SecFilingGemmaContractError, match="source_hashes"):
        build_candidate_manifest(
            model_digest=_hash("d"),
            ollama_runtime_fingerprint_sha256=_hash("e"),
            sec_audit_checksums_json_sha256=_hash("f"),
            sec_catalog_artifact_sha256=_hash("a"),
            sec_audit_source_commit="a" * 40,
            calendar_source_evidence_sha256=_hash("c"),
            calendar_sessions_sha256=CALENDAR_SESSIONS_SHA256,
            corpus_universe_sha256=universe["universe_sha256"],
            corpus_universe_semantic_sha256=universe[
                "universe_semantic_sha256"
            ],
            identity_lexicon_sha256=IDENTITY_LEXICON_SHA256,
            predecessor_reveal_registry_sha256=_hash("7"),
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


def test_universe_semantic_hash_excludes_nonbehavioral_source_wrappers() -> None:
    records = _universe_source_records()
    first = _universe(records)
    changed_records = copy.deepcopy(records)
    changed_records[0]["source_record_sha256"] = _hash("9")
    changed_records[0]["acceptance_datetime"] = (
        changed_records[0]["acceptance_datetime"][:8] + "170000"
    )
    second = _universe(changed_records)

    assert first["universe_sha256"] != second["universe_sha256"]
    assert first["universe_semantic_sha256"] == second[
        "universe_semantic_sha256"
    ]


def test_universe_rejects_non_apple_and_stage_content_rejects_cross_stage_accession() -> None:
    records = _universe_source_records()
    records[0]["subject_cik"] = "0000000001"
    with pytest.raises(SecFilingGemmaContractError, match="subject CIK"):
        _universe(records)

    universe = _universe()
    content = _stage_content(universe, "development")
    assert validate_stage_content_manifest(
        content,
        universe_manifest=universe,
        expected_content_manifest_sha256=content["content_manifest_sha256"],
    ) == content["content_manifest_sha256"]
    source_documents = [
        {
            key: document[key]
            for key in (
                "accession_number",
                "primary_document_sha256",
                "normalized_text_sha256",
                "primary_document_bytes",
                "normalized_text_bytes",
            )
        }
        for document in content["documents"]
    ]
    source_documents[0]["accession_number"] = next(
        record["accession_number"]
        for record in universe["records"]
        if record["artifact_stage"] == "final"
    )
    with pytest.raises(SecFilingGemmaContractError, match="authorized accession"):
        build_stage_content_manifest(
            artifact_stage="development",
            corpus_universe_sha256=universe["universe_sha256"],
            documents=source_documents,
            universe_manifest=universe,
        )


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
    request, redacted_manifest, candidate, contents = _extractor_context(universe)
    validated = _validate_request(
        request, universe, redacted_manifest, candidate, contents
    )
    assert validated["sentence_ids"] == ("C0001", "C0002", "P0001")
    assert set(validated["model_payload"]) == {
        "model",
        "messages",
        "format",
        "stream",
        "think",
        "options",
    }


def test_redacted_input_v3_binds_owned_preprocessing_and_reader_ancestry() -> None:
    universe = _universe()
    _request, manifest, _candidate_manifest, contents = _extractor_context(universe)
    assert manifest["schema_version"] == REDACTED_INPUT_SCHEMA_VERSION
    assert manifest["preprocessed_event_sha256"] == _hash("1")
    assert manifest["owned_preprocessing_receipt_sha256"] == _hash("2")
    assert manifest["sec_reader_receipt_sha256"] == _hash("3")
    assert manifest["carry_in_reader_receipt_sha256"] is None
    assert validate_redacted_input_manifest(
        manifest,
        universe_manifest=universe,
        stage_content_manifest=contents["development"],
        expected_manifest_sha256=manifest["redacted_input_manifest_sha256"],
        expected_preprocessed_event_sha256=_hash("1"),
        expected_owned_preprocessing_receipt_sha256=_hash("2"),
        expected_sec_reader_receipt_sha256=_hash("3"),
        expected_carry_in_reader_receipt_sha256=None,
    ) == manifest["redacted_input_manifest_sha256"]

    current = next(
        record
        for record in universe["records"]
        if record["artifact_stage"] == "development"
    )
    with pytest.raises(
        SecFilingGemmaContractError,
        match="cannot claim a carry-in",
    ):
        build_redacted_input_manifest(
            artifact_stage="development",
            accession_number=current["accession_number"],
            corpus_universe_sha256=universe["universe_sha256"],
            model_payload_sha256=_hash("4"),
            preprocessed_event_sha256=_hash("5"),
            owned_preprocessing_receipt_sha256=_hash("6"),
            sec_reader_receipt_sha256=_hash("7"),
            carry_in_reader_receipt_sha256=_hash("8"),
            universe_manifest=universe,
            stage_content_manifest=contents["development"],
        )

    intermediate_content = _stage_content(universe, "intermediate")
    intermediate = next(
        record
        for record in universe["records"]
        if record["artifact_stage"] == "intermediate"
    )
    with pytest.raises(
        SecFilingGemmaContractError,
        match="carry_in_reader_receipt_sha256",
    ):
        build_redacted_input_manifest(
            artifact_stage="intermediate",
            accession_number=intermediate["accession_number"],
            corpus_universe_sha256=universe["universe_sha256"],
            model_payload_sha256=_hash("4"),
            preprocessed_event_sha256=_hash("5"),
            owned_preprocessing_receipt_sha256=_hash("6"),
            sec_reader_receipt_sha256=_hash("7"),
            carry_in_reader_receipt_sha256=None,
            universe_manifest=universe,
            stage_content_manifest=intermediate_content,
        )
    intermediate_manifest = build_redacted_input_manifest(
        artifact_stage="intermediate",
        accession_number=intermediate["accession_number"],
        corpus_universe_sha256=universe["universe_sha256"],
        model_payload_sha256=_hash("4"),
        preprocessed_event_sha256=_hash("5"),
        owned_preprocessing_receipt_sha256=_hash("6"),
        sec_reader_receipt_sha256=_hash("7"),
        carry_in_reader_receipt_sha256=_hash("8"),
        universe_manifest=universe,
        stage_content_manifest=intermediate_content,
    )
    assert intermediate_manifest["carry_in_reader_receipt_sha256"] == _hash("8")
    assert validate_redacted_input_manifest(
        intermediate_manifest,
        universe_manifest=universe,
        stage_content_manifest=intermediate_content,
        expected_manifest_sha256=intermediate_manifest[
            "redacted_input_manifest_sha256"
        ],
        expected_preprocessed_event_sha256=_hash("5"),
        expected_owned_preprocessing_receipt_sha256=_hash("6"),
        expected_sec_reader_receipt_sha256=_hash("7"),
        expected_carry_in_reader_receipt_sha256=_hash("8"),
    ) == intermediate_manifest["redacted_input_manifest_sha256"]


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("preprocessed_event_sha256", _hash("4")),
        ("owned_preprocessing_receipt_sha256", _hash("5")),
        ("sec_reader_receipt_sha256", _hash("6")),
    ],
)
def test_redacted_input_rejects_rehashed_but_unpinned_owned_ancestry(
    field: str,
    replacement: str,
) -> None:
    universe = _universe()
    request, manifest, candidate, contents = _extractor_context(universe)
    changed = copy.deepcopy(manifest)
    changed[field] = replacement
    body = {
        key: value
        for key, value in changed.items()
        if key != "redacted_input_manifest_sha256"
    }
    changed["redacted_input_manifest_sha256"] = canonical_sha256(body)
    request["redacted_input_manifest_sha256"] = changed[
        "redacted_input_manifest_sha256"
    ]

    with pytest.raises(
        SecFilingGemmaContractError,
        match="independently pinned owned ancestry",
    ):
        _validate_request(request, universe, changed, candidate, contents)


@pytest.mark.parametrize("binding", ["calendar", "catalog"])
def test_extractor_rejects_candidate_universe_provenance_mismatch(
    binding: str,
) -> None:
    universe = _universe()
    request, redacted_manifest, _, contents = _extractor_context(universe)
    candidate = _candidate(
        universe,
        calendar_source_evidence_sha256=(
            _hash("9") if binding == "calendar" else None
        ),
        catalog_sha256=_hash("8") if binding == "catalog" else None,
    )

    with pytest.raises(SecFilingGemmaContractError, match="candidate-bound"):
        _validate_request(
            request,
            universe,
            redacted_manifest,
            candidate,
            contents,
        )


def test_extractor_rejects_stage_relabel_and_wrong_or_self_prior() -> None:
    universe = _universe()
    request, redacted_manifest, candidate, contents = _extractor_context(universe)
    request["stage"] = "final"
    with pytest.raises(SecFilingGemmaContractError, match="stage boundary"):
        _validate_request(request, universe, redacted_manifest, candidate, contents)

    request, redacted_manifest, candidate, contents = _extractor_context(universe)
    request["prior_same_form_filing_sha256"] = request["current_filing_sha256"]
    with pytest.raises(SecFilingGemmaContractError, match="immediately preceding"):
        _validate_request(request, universe, redacted_manifest, candidate, contents)

    request, redacted_manifest, candidate, contents = _extractor_context(universe)
    with_later_stage = {
        **contents,
        "final": _stage_content(universe, "final"),
    }
    with pytest.raises(SecFilingGemmaContractError, match="content_manifests_by_stage"):
        _validate_request(
            request,
            universe,
            redacted_manifest,
            candidate,
            with_later_stage,
        )

    other_accession = next(
        document["accession_number"]
        for document in contents["development"]["documents"]
        if document["accession_number"] != request["current_accession_number"]
    )
    other_event_manifest = build_redacted_input_manifest(
        artifact_stage="development",
        accession_number=other_accession,
        corpus_universe_sha256=universe["universe_sha256"],
        model_payload_sha256=request["model_payload_sha256"],
        preprocessed_event_sha256=redacted_manifest[
            "preprocessed_event_sha256"
        ],
        owned_preprocessing_receipt_sha256=redacted_manifest[
            "owned_preprocessing_receipt_sha256"
        ],
        sec_reader_receipt_sha256=redacted_manifest[
            "sec_reader_receipt_sha256"
        ],
        carry_in_reader_receipt_sha256=redacted_manifest[
            "carry_in_reader_receipt_sha256"
        ],
        universe_manifest=universe,
        stage_content_manifest=contents["development"],
    )
    request["redacted_input_manifest_sha256"] = other_event_manifest[
        "redacted_input_manifest_sha256"
    ]
    with pytest.raises(SecFilingGemmaContractError, match="another filing event"):
        _validate_request(
            request,
            universe,
            other_event_manifest,
            candidate,
            contents,
        )


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
        (
            "The period ended September thirtieth while demand improved.",
            "Calendar date language",
        ),
        ("Revenue was １２ in the current period.", "canonical ASCII"),
        ("Revenue was ١٢ in the current period.", "canonical ASCII"),
        ("A\u0440ple demand improved.", "canonical ASCII"),
        ("Revenue improved in September.", "Calendar date language"),
        ("Demand weakened on Tuesday.", "Calendar date language"),
        ("Revenue improved in May.", "Calendar date language"),
    ],
)
def test_extractor_rejects_identity_and_absolute_value_leakage(
    text: str, message: str
) -> None:
    universe = _universe()
    request, redacted_manifest, candidate, contents = _extractor_context(universe)
    request, redacted_manifest, candidate, contents = _replace_first_model_sentence(
        request, universe, redacted_manifest, contents, text
    )
    with pytest.raises(SecFilingGemmaContractError, match=message):
        _validate_request(request, universe, redacted_manifest, candidate, contents)


def test_identity_and_date_scanners_do_not_destroy_directional_risk_language() -> None:
    universe = _universe()
    request, redacted_manifest, _, contents = _extractor_context(universe)
    request, redacted_manifest, candidate, contents = _replace_first_model_sentence(
        request,
        universe,
        redacted_manifest,
        contents,
        "Results may vary as macroeconomic ratios deteriorate.",
    )
    assert _validate_request(request, universe, redacted_manifest, candidate, contents)[
        "model_payload_sha256"
    ] == request["model_payload_sha256"]


def test_metadata_cannot_be_injected_into_the_model_payload() -> None:
    universe = _universe()
    request, redacted_manifest, candidate, contents = _extractor_context(universe)
    request["model_payload"]["stage"] = "development"
    request["model_payload_sha256"] = canonical_sha256(request["model_payload"])
    with pytest.raises(SecFilingGemmaContractError, match="metadata"):
        _validate_request(request, universe, redacted_manifest, candidate, contents)


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

    changed_support = copy.deepcopy(rows)
    changed_support[0]["fit_eligible"] = False
    with pytest.raises(SecFilingGemmaContractError, match="Learner support"):
        _validate_training(
            "development_fit", changed_support, universe, candidate
        )

    invalid_causal_market_identity = copy.deepcopy(rows)
    invalid_causal_market_identity[0]["market_prefix_chain_identity_sha256"] = "not-a-hash"
    with pytest.raises(
        SecFilingGemmaContractError,
        match="market_prefix_chain_identity_sha256",
    ):
        _validate_training(
            "development_fit", invalid_causal_market_identity, universe, candidate
        )

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

    reused = copy.deepcopy(evidence)
    reused["artifact_sha256s"]["final_extraction"] = reused[
        "artifact_sha256s"
    ]["development_extraction"]
    with pytest.raises(SecFilingGemmaContractError, match="physically distinct"):
        _validate_runtime(reused, candidate, universe)

    reused_market = copy.deepcopy(evidence)
    reused_market["market_data_sha256s_by_stage"]["final"]["AAPL"] = (
        reused_market["market_data_sha256s_by_stage"]["development"]["AAPL"]
    )
    with pytest.raises(SecFilingGemmaContractError, match="stage-bounded"):
        _validate_runtime(reused_market, candidate, universe)


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


def _lesson_binding(lesson: dict) -> dict:
    return {
        key: lesson[key]
        for key in (
            "lesson_id",
            "accession_number",
            "decision_session",
            "feature_sha256",
            "prediction_receipt_sha256",
            "market_data_manifest_sha256",
            "strategy_ledger_slice_sha256",
            "benchmark_ledger_slice_sha256",
        )
    }


def _lesson_binding_args(
    *, prior: list[dict], eligible: list[dict]
) -> dict:
    normalized_prior = sorted(prior, key=lambda item: item["lesson_id"])
    normalized_eligible = sorted(eligible, key=lambda item: item["lesson_id"])
    return {
        "prior_lesson_bindings": normalized_prior,
        "expected_prior_lesson_bindings_sha256": canonical_sha256(
            normalized_prior
        ),
        "eligible_lesson_bindings": normalized_eligible,
        "expected_eligible_lesson_bindings_sha256": canonical_sha256(
            normalized_eligible
        ),
    }


def test_live_lessons_have_exact_maturity_label_and_state_hash_chain() -> None:
    lesson, as_of = _live_lesson()
    initial_binding_args = _lesson_binding_args(
        prior=[], eligible=[_lesson_binding(lesson)]
    )
    anchor = validate_live_lessons(
        [lesson],
        as_of_session=as_of,
        session_dates=SESSIONS,
        expected_calendar_sessions_sha256=CALENDAR_SESSIONS_SHA256,
        expected_prior_tip_sha256=None,
        expected_prior_count=0,
        expected_prior_model_state_sha256=_hash("8"),
        expected_prior_last_sort_key=None,
        **initial_binding_args,
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
        **_lesson_binding_args(prior=anchor["lesson_bindings"], eligible=[]),
    )
    assert unchanged["lesson_count"] == 1
    assert unchanged["tip_sha256"] == anchor["tip_sha256"]

    backdated = copy.deepcopy(lesson)
    backdated.update(
        {
            "sequence_number": 2,
            "lesson_id": "lesson-two",
            "accession_number": "0000320193-01-000002",
            "decision_session": "2001-01-03",
            "label_maturity_session": _offset_session(
                "2001-01-03", LABEL_MATURITY_OFFSET
            ),
            "parent_lesson_sha256": anchor["tip_sha256"],
            "input_model_state_sha256": anchor["model_state_sha256"],
            "output_model_state_sha256": _hash("a"),
            "feature_sha256": _hash("b"),
            "prediction_receipt_sha256": _hash("c"),
            "market_data_manifest_sha256": _hash("d"),
            "strategy_ledger_slice_sha256": _hash("e"),
            "benchmark_ledger_slice_sha256": _hash("f"),
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
            **_lesson_binding_args(
                prior=anchor["lesson_bindings"],
                eligible=[_lesson_binding(backdated)],
            ),
        )


def test_live_lessons_reject_invalid_accessions_and_cross_batch_replay() -> None:
    lesson, as_of = _live_lesson()
    invalid = copy.deepcopy(lesson)
    invalid["accession_number"] = "0000320193-not-an-accession"
    invalid["lesson_sha256"] = canonical_sha256(
        {key: value for key, value in invalid.items() if key != "lesson_sha256"}
    )
    with pytest.raises(SecFilingGemmaContractError, match="non-Apple accession"):
        validate_live_lessons(
            [invalid],
            as_of_session=as_of,
            session_dates=SESSIONS,
            expected_calendar_sessions_sha256=CALENDAR_SESSIONS_SHA256,
            expected_prior_tip_sha256=None,
            expected_prior_count=0,
            expected_prior_model_state_sha256=_hash("8"),
            expected_prior_last_sort_key=None,
            **_lesson_binding_args(
                prior=[], eligible=[_lesson_binding(invalid)]
            ),
        )

    anchor = validate_live_lessons(
        [lesson],
        as_of_session=as_of,
        session_dates=SESSIONS,
        expected_calendar_sessions_sha256=CALENDAR_SESSIONS_SHA256,
        expected_prior_tip_sha256=None,
        expected_prior_count=0,
        expected_prior_model_state_sha256=_hash("8"),
        expected_prior_last_sort_key=None,
        **_lesson_binding_args(prior=[], eligible=[_lesson_binding(lesson)]),
    )
    replay = copy.deepcopy(lesson)
    replay["sequence_number"] = 2
    replay["lesson_id"] = "lesson-replayed"
    replay["parent_lesson_sha256"] = anchor["tip_sha256"]
    replay["input_model_state_sha256"] = anchor["model_state_sha256"]
    replay["output_model_state_sha256"] = _hash("a")
    replay["lesson_sha256"] = canonical_sha256(
        {key: value for key, value in replay.items() if key != "lesson_sha256"}
    )
    with pytest.raises(SecFilingGemmaContractError, match="replays a prior"):
        validate_live_lessons(
            [replay],
            as_of_session=as_of,
            session_dates=SESSIONS,
            expected_calendar_sessions_sha256=CALENDAR_SESSIONS_SHA256,
            expected_prior_tip_sha256=anchor["tip_sha256"],
            expected_prior_count=anchor["lesson_count"],
            expected_prior_model_state_sha256=anchor["model_state_sha256"],
            expected_prior_last_sort_key=anchor["last_sort_key"],
            **_lesson_binding_args(
                prior=anchor["lesson_bindings"],
                eligible=[_lesson_binding(replay)],
            ),
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
            **_lesson_binding_args(
                prior=[], eligible=[_lesson_binding(wrong_label)]
            ),
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
            **_lesson_binding_args(
                prior=[], eligible=[_lesson_binding(premature)]
            ),
        )


def test_future_calendar_extension_preserves_the_complete_historical_prefix() -> None:
    legacy = list(LEGACY_EXPECTED_SESSIONS)
    legacy_manifest = build_calendar_extension_manifest(
        prior_session_dates=legacy,
        extended_session_dates=SESSIONS,
        expected_prior_sessions_sha256=canonical_sha256(legacy),
    )
    assert legacy_manifest["prior_last_session"] == "2025-01-10"
    assert legacy_manifest["extended_last_session"] == "2026-07-10"
    assert legacy_manifest["prior_calendar_id"].endswith("2025_01_10_v1")
    assert legacy_manifest["extended_calendar_id"].endswith("2026_07_10_v2")
    assert validate_calendar_extension_manifest(
        legacy_manifest,
        expected_prior_sessions_sha256=canonical_sha256(legacy),
        expected_extension_manifest_sha256=legacy_manifest[
            "extension_manifest_sha256"
        ],
    ) == legacy_manifest["extension_manifest_sha256"]

    rewritten = list(SESSIONS)
    rewritten[100] = rewritten[101]
    with pytest.raises(SecFilingGemmaContractError):
        build_calendar_extension_manifest(
            prior_session_dates=legacy,
            extended_session_dates=rewritten,
            expected_prior_sessions_sha256=canonical_sha256(legacy),
        )

    changed = copy.deepcopy(legacy_manifest)
    changed["appended_sessions"][0] = "2025-01-14"
    with pytest.raises(SecFilingGemmaContractError, match="noncanonical"):
        validate_calendar_extension_manifest(
            changed,
            expected_prior_sessions_sha256=canonical_sha256(legacy),
            expected_extension_manifest_sha256=legacy_manifest[
                "extension_manifest_sha256"
            ],
        )


def test_calendar_source_evidence_binds_exact_official_bytes_and_external_pin() -> None:
    records = {
        name: {
            "url": url,
            "content_sha256": f"{index + 1:064x}",
            "byte_count": 1_000 + index,
        }
        for index, (name, url) in enumerate(CALENDAR_SOURCE_URLS.items())
    }
    evidence = build_calendar_source_evidence_manifest(
        retrieved_at_utc="2026-07-11T12:00:00Z",
        source_records=records,
    )
    assert evidence["current_session_count"] == len(SESSIONS)
    assert evidence["market_history_session_count"] == len(
        EXPECTED_MARKET_HISTORY_SESSIONS
    )
    assert evidence["semantic_reconciliation_required_before_candidate_use"] is True
    assert evidence["authorizes_corpus_or_model_access"] is False
    assert validate_calendar_source_evidence_manifest(
        evidence,
        expected_calendar_source_evidence_sha256=evidence[
            "calendar_source_evidence_sha256"
        ],
    ) == evidence["calendar_source_evidence_sha256"]

    changed = copy.deepcopy(evidence)
    changed["source_records"]["nyse_2026_calendar_pdf"][
        "content_sha256"
    ] = _hash("9")
    with pytest.raises(SecFilingGemmaContractError, match="noncanonical"):
        validate_calendar_source_evidence_manifest(
            changed,
            expected_calendar_source_evidence_sha256=evidence[
                "calendar_source_evidence_sha256"
            ],
        )


def test_contract_module_imports_only_effect_free_modules() -> None:
    tree = ast.parse(inspect.getsource(contract_module))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__",
        "agent_benchmark",
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
