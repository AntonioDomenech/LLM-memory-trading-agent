from __future__ import annotations

import base64
import copy
from datetime import date
import hashlib
import json
from types import SimpleNamespace

import pytest

import agent_benchmark.sec_filing_gemma_stage_verifier as verifier_module
from agent_benchmark.sec_filing_gemma_contract import (
    CALENDAR_SOURCE_URLS,
    CONTRACT_VERSION,
    REQUIRED_SOURCE_HASHES,
    REQUIRED_STAGE_VERIFIER_CHECKS,
    build_candidate_manifest,
    build_calendar_source_evidence_manifest,
    build_contract_manifest,
    build_corpus_universe_manifest,
    canonical_sha256,
    session_calendar_sha256,
)
from agent_benchmark.sec_filing_gemma_learner import (
    SecFilingGemmaTwoHeadLearner,
)
from agent_benchmark.sec_filing_gemma_prediction_evidence import (
    AVAILABLE_PREDICTION_STATUS,
)
from agent_benchmark.sec_filing_gemma_stage_verifier import (
    OWNED_HARDENED_TRANSPORT_MODE,
    STAGE_EVIDENCE_SCHEMA_VERSION,
    STAGE_RUNTIME_RECEIPT_SCHEMA_VERSION,
    SecFilingGemmaStageVerifierBlocked,
    SecFilingGemmaStageVerifierError,
    audit_stage_evidence,
    authoritative_prerequisite_validator,
    validate_candidate_source_bytes,
    validate_calendar_and_universe_snapshot,
    validate_learner_refit_replays,
    validate_model_attempt_batch,
    validate_raw_scores_gates_and_ranking,
    validate_stage_runtime_receipt,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _candidate_fixture() -> tuple[dict, dict[str, str]]:
    source_payloads = {
        role: f"exact source bytes for {role}".encode("utf-8")
        for role in REQUIRED_SOURCE_HASHES
    }
    source_hashes = {
        role: hashlib.sha256(payload).hexdigest()
        for role, payload in source_payloads.items()
    }
    candidate = build_candidate_manifest(
        model_digest=_h("model"),
        ollama_runtime_fingerprint_sha256=_h("runtime"),
        sec_audit_checksums_json_sha256=_h("audit"),
        sec_catalog_artifact_sha256=_h("catalog"),
        sec_audit_source_commit=_h("audit commit"),
        calendar_source_evidence_sha256=_h("calendar evidence"),
        calendar_sessions_sha256=session_calendar_sha256(EXPECTED_SESSIONS),
        corpus_universe_sha256=_h("universe"),
        corpus_universe_semantic_sha256=_h("universe semantic"),
        identity_lexicon_sha256=_h("identity lexicon"),
        predecessor_reveal_registry_sha256=_h("predecessor registry"),
        holdout_attempt_id=f"{CONTRACT_VERSION}-attempt-001",
        experiment_source_commit=_h("experiment commit"),
        source_tree_sha256=_h("source tree"),
        source_hashes=source_hashes,
    )
    encoded = {role: _b64(payload) for role, payload in source_payloads.items()}
    return candidate, encoded


def test_candidate_contract_and_every_source_byte_pin_replay() -> None:
    candidate, sources = _candidate_fixture()
    summary = validate_candidate_source_bytes(
        contract_manifest=build_contract_manifest(),
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate["candidate_sha256"],
        source_bytes_base64_by_role=sources,
    )

    assert summary["candidate_sha256"] == candidate["candidate_sha256"]
    assert summary["stage_verifier_source_sha256"] == candidate["bindings"][
        "source_hashes"
    ]["stage_verifier"]


def test_source_byte_omission_mutation_and_cross_candidate_pin_fail_closed() -> None:
    candidate, sources = _candidate_fixture()
    omitted = copy.deepcopy(sources)
    omitted.pop("runner")
    with pytest.raises(SecFilingGemmaStageVerifierError, match="keys changed"):
        validate_candidate_source_bytes(
            contract_manifest=build_contract_manifest(),
            candidate_manifest=candidate,
            expected_candidate_sha256=candidate["candidate_sha256"],
            source_bytes_base64_by_role=omitted,
        )

    changed = copy.deepcopy(sources)
    changed["runner"] = _b64(b"different runner")
    with pytest.raises(SecFilingGemmaStageVerifierError, match="runner"):
        validate_candidate_source_bytes(
            contract_manifest=build_contract_manifest(),
            candidate_manifest=candidate,
            expected_candidate_sha256=candidate["candidate_sha256"],
            source_bytes_base64_by_role=changed,
        )

    with pytest.raises(SecFilingGemmaStageVerifierError, match="candidate"):
        validate_candidate_source_bytes(
            contract_manifest=build_contract_manifest(),
            candidate_manifest=candidate,
            expected_candidate_sha256=_h("other candidate"),
            source_bytes_base64_by_role=sources,
        )


def _complete_calendar_universe_fixture() -> tuple[dict, dict, dict, dict[str, str]]:
    calendar_payloads = {
        name: f"official fixture bytes {name}".encode("utf-8")
        for name in CALENDAR_SOURCE_URLS
    }
    calendar = build_calendar_source_evidence_manifest(
        retrieved_at_utc="2026-07-11T10:00:00Z",
        source_records={
            name: {
                "url": CALENDAR_SOURCE_URLS[name],
                "content_sha256": hashlib.sha256(payload).hexdigest(),
                "byte_count": len(payload),
            }
            for name, payload in calendar_payloads.items()
        },
    )
    records: list[dict] = []
    serial = 1
    for year in range(2000, 2026):
        for form, month in (("10-K", 2), ("10-Q", 5), ("10-Q", 8), ("10-Q", 11)):
            evidence_date = date(year, month, 15)
            records.append(
                {
                    "accession_number": f"0000320193-{year % 100:02d}-{serial:06d}",
                    "subject_cik": "0000320193",
                    "form": form,
                    "acceptance_datetime": evidence_date.strftime("%Y%m%d") + "160000",
                    "filing_date": evidence_date.isoformat(),
                    "filing_date_change": None,
                    "primary_document": f"filing-{serial}.htm",
                    "source_record_sha256": f"{serial + 70_000:064x}",
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
                "source_record_sha256": f"{serial + 70_000:064x}",
            }
        )
        serial += 1
    universe = build_corpus_universe_manifest(
        catalog_artifact_sha256=_h("complete catalog"),
        calendar_artifact_sha256=calendar["calendar_source_evidence_sha256"],
        catalog_total_record_count=1_000,
        catalog_eligible_record_count=len(records),
        session_dates=EXPECTED_SESSIONS,
        records=records,
    )
    _, encoded_sources = _candidate_fixture()
    source_hashes = {
        role: hashlib.sha256(base64.b64decode(encoded)).hexdigest()
        for role, encoded in encoded_sources.items()
    }
    candidate = build_candidate_manifest(
        model_digest=_h("model"),
        ollama_runtime_fingerprint_sha256=_h("runtime"),
        sec_audit_checksums_json_sha256=_h("audit"),
        sec_catalog_artifact_sha256=universe["catalog_artifact_sha256"],
        sec_audit_source_commit=_h("audit commit"),
        calendar_source_evidence_sha256=calendar[
            "calendar_source_evidence_sha256"
        ],
        calendar_sessions_sha256=universe["calendar_sessions_sha256"],
        corpus_universe_sha256=universe["universe_sha256"],
        corpus_universe_semantic_sha256=universe["universe_semantic_sha256"],
        identity_lexicon_sha256=_h("identity lexicon"),
        predecessor_reveal_registry_sha256=_h("predecessor registry"),
        holdout_attempt_id=f"{CONTRACT_VERSION}-attempt-001",
        experiment_source_commit=_h("experiment commit"),
        source_tree_sha256=_h("source tree"),
        source_hashes=source_hashes,
    )
    calendar_bytes = {name: _b64(payload) for name, payload in calendar_payloads.items()}
    return calendar, universe, candidate, calendar_bytes


def test_calendar_bytes_and_complete_universe_are_candidate_bound() -> None:
    calendar, universe, candidate, calendar_bytes = _complete_calendar_universe_fixture()
    result = validate_calendar_and_universe_snapshot(
        calendar_evidence_manifest=calendar,
        calendar_source_bytes_base64_by_name=calendar_bytes,
        corpus_universe_manifest=universe,
        candidate_manifest=candidate,
    )
    assert result["calendar_source_evidence_sha256"] == calendar[
        "calendar_source_evidence_sha256"
    ]
    assert result["corpus_universe_sha256"] == universe["universe_sha256"]

    changed = copy.deepcopy(calendar_bytes)
    changed["nyse_hours_calendars"] = _b64(b"substituted calendar")
    with pytest.raises(SecFilingGemmaStageVerifierError, match="do not match"):
        validate_calendar_and_universe_snapshot(
            calendar_evidence_manifest=calendar,
            calendar_source_bytes_base64_by_name=changed,
            corpus_universe_manifest=universe,
            candidate_manifest=candidate,
        )

def _learner_replays() -> tuple[str, dict, list[dict]]:
    candidate_hash = _h("learner candidate")
    semantic_names = ["x", "y"]
    semantic_training = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 0.5],
        [0.5, 2.0],
    ]
    ablation_names = ["x"]
    ablation_training = [[row[0]] for row in semantic_training]
    binary = [0, 1, 0, 1, 1, 0]
    edges = [-0.2, 0.1, -0.1, 0.2, 0.3, -0.3]
    edge_targets_hex = [float(value).hex() for value in edges]
    membership_hash = _h("training membership")
    encoded_features = {
        "semantic": [
            [float(value).hex() for value in values]
            for values in semantic_training
        ],
        "ablation": [
            [float(value).hex() for value in values]
            for values in ablation_training
        ],
    }
    row_hash = _h("prediction row")
    row = {
        "prediction_row_sha256": row_hash,
        "fold_id": "fold_1",
        "prediction_status": AVAILABLE_PREDICTION_STATUS,
        "fold_context": {
            "fold_train_cutoff_session": "2004-12-31",
            "training_set_count": len(binary),
            "training_positive_count": sum(binary),
            "training_set_membership_sha256": membership_hash,
            "semantic_training_feature_matrix_sha256": canonical_sha256(
                encoded_features["semantic"]
            ),
            "ablation_training_feature_matrix_sha256": canonical_sha256(
                encoded_features["ablation"]
            ),
            "training_binary_target_sha256": canonical_sha256(binary),
            "training_edge_target_sha256": canonical_sha256(edge_targets_hex),
            "training_set_max_label_maturity_session": "2004-12-30",
        },
    }
    replays: list[dict] = []
    for variant in ("semantic", "ablation"):
        names = semantic_names if variant == "semantic" else ablation_names
        training = semantic_training if variant == "semantic" else ablation_training
        prediction = [[0.75, 0.25]] if variant == "semantic" else [[0.75]]
        metadata = {
            "candidate_sha256": candidate_hash,
            "head_variant": variant,
            "fold_id": "fold_1",
            "train_label_maturity_through": "2004-12-31",
            "training_set_sha256": membership_hash,
            "training_row_count": len(training),
            "maximum_training_label_maturity_session": "2004-12-30",
            "feature_schema_sha256": canonical_sha256(names),
        }
        learner = SecFilingGemmaTwoHeadLearner().fit(
            training,
            binary,
            edges,
            feature_names=names,
            fit_metadata=metadata,
        )
        output = learner.predict_components(prediction)
        row[f"{variant}_cash_probability_hex"] = float(
            output["cash_win_probability_10bps"][0]
        ).hex()
        row[f"{variant}_expected_edge_hex"] = float(
            output["expected_active_log_edge_10bps"][0]
        ).hex()
        row["fold_context"][f"{variant}_fold_state_sha256"] = learner.model_sha256
        replays.append(
            {
                "fold_id": "fold_1",
                "variant": variant,
                "feature_names": names,
                "training_features_hex": encoded_features[variant],
                "binary_targets": binary,
                "edge_targets_hex": edge_targets_hex,
                "fit_metadata": metadata,
                "expected_state": learner.to_state(),
                "prediction_cases": [
                    {
                        "prediction_row_sha256": row_hash,
                        "features_hex": [float(value).hex() for value in prediction[0]],
                    }
                ],
            }
        )
    return candidate_hash, {"rows": [row]}, replays


def test_deterministic_learner_refit_state_and_prediction_replay() -> None:
    candidate_hash, prefix, replays = _learner_replays()
    result = validate_learner_refit_replays(
        replays,
        candidate_sha256=candidate_hash,
        prediction_prefix=prefix,
    )
    assert result["prediction_case_count"] == 2
    assert set(result["learner_state_sha256s_by_fold"]["fold_1"]) == {
        "semantic",
        "ablation",
    }
    assert set(result["learner_input_identity_sha256s_by_fold"]["fold_1"]) == {
        "semantic",
        "ablation",
    }


def test_learner_state_mutation_case_omission_and_cross_binding_are_rejected() -> None:
    candidate_hash, prefix, replays = _learner_replays()
    mutated = copy.deepcopy(replays)
    mutated[0]["expected_state"]["training_positive_count"] += 1
    with pytest.raises(SecFilingGemmaStageVerifierError, match="state"):
        validate_learner_refit_replays(
            mutated, candidate_sha256=candidate_hash, prediction_prefix=prefix
        )

    omitted = copy.deepcopy(replays)
    omitted[1]["prediction_cases"] = []
    with pytest.raises(SecFilingGemmaStageVerifierError, match="omit"):
        validate_learner_refit_replays(
            omitted, candidate_sha256=candidate_hash, prediction_prefix=prefix
        )

    crossed = copy.deepcopy(replays)
    crossed[0]["fit_metadata"]["candidate_sha256"] = _h("other")
    with pytest.raises(SecFilingGemmaStageVerifierError, match="binding"):
        validate_learner_refit_replays(
            crossed, candidate_sha256=candidate_hash, prediction_prefix=prefix
        )

    duplicated = copy.deepcopy(replays)
    extra = copy.deepcopy(duplicated[0])
    extra["prediction_cases"] = []
    duplicated.append(extra)
    with pytest.raises(SecFilingGemmaStageVerifierError, match="duplicated"):
        validate_learner_refit_replays(
            duplicated, candidate_sha256=candidate_hash, prediction_prefix=prefix
        )


def test_learner_refit_inputs_must_match_the_prediction_fold_context() -> None:
    candidate_hash, prefix, replays = _learner_replays()

    changed_matrix = copy.deepcopy(replays)
    changed_matrix[0]["training_features_hex"][0][0] = float(0.25).hex()
    with pytest.raises(SecFilingGemmaStageVerifierError, match="feature matrix"):
        validate_learner_refit_replays(
            changed_matrix,
            candidate_sha256=candidate_hash,
            prediction_prefix=prefix,
        )

    changed_membership = copy.deepcopy(replays)
    changed_membership[0]["fit_metadata"]["training_set_sha256"] = _h(
        "another membership"
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="membership"):
        validate_learner_refit_replays(
            changed_membership,
            candidate_sha256=candidate_hash,
            prediction_prefix=prefix,
        )

    changed_targets = copy.deepcopy(replays)
    changed_targets[0]["binary_targets"][0] = 1
    with pytest.raises(SecFilingGemmaStageVerifierError, match="targets"):
        validate_learner_refit_replays(
            changed_targets,
            candidate_sha256=candidate_hash,
            prediction_prefix=prefix,
        )

    changed_counts = copy.deepcopy(prefix)
    changed_counts["rows"][0]["fold_context"]["training_set_count"] += 1
    with pytest.raises(SecFilingGemmaStageVerifierError, match="counts"):
        validate_learner_refit_replays(
            replays,
            candidate_sha256=candidate_hash,
            prediction_prefix=changed_counts,
        )

    changed_maturity = copy.deepcopy(replays)
    changed_maturity[0]["fit_metadata"]["train_label_maturity_through"] = (
        "2004-12-29"
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="maturity"):
        validate_learner_refit_replays(
            changed_maturity,
            candidate_sha256=candidate_hash,
            prediction_prefix=prefix,
        )


def test_model_batch_hardcodes_owned_transport_and_exact_guard_order(monkeypatch) -> None:
    candidate, _ = _candidate_fixture()
    observed: dict[str, object] = {}

    def fake_attempt(manifest, **kwargs):
        observed.setdefault("modes", []).append(kwargs["expected_transport_mode"])
        observed.setdefault("candidates", []).append(kwargs["expected_candidate_sha256"])
        payload = manifest["payload"]
        return SimpleNamespace(
            request_bytes=json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            receipt_sha256=manifest["receipt_sha256"],
            elapsed_nanoseconds=7,
        )

    def fake_guard(value, **kwargs):
        observed["guard_order"] = kwargs["model_call_receipt_sha256s"]
        observed["guard_stage"] = kwargs["stage"]
        return value["runtime_guard_sha256"]

    monkeypatch.setattr(verifier_module, "validate_ollama_model_attempt_receipt", fake_attempt)
    monkeypatch.setattr(verifier_module, "validate_runtime_identity_guard", fake_guard)
    accessions = ["0000320193-05-000001", "0000320193-05-000002"]
    attempts = []
    for index, accession in enumerate(accessions):
        payload = {"messages": [index]}
        attempts.append(
            {
                "accession_number": accession,
                "expected_model_payload": payload,
                "expected_sentence_ids": ["C0001"],
                "receipt": {
                    "payload": payload,
                    "receipt_sha256": _h(f"receipt {index}"),
                },
            }
        )
    batch = {
        "stage": "development",
        "before_runtime_evidence": {"probe": "same"},
        "after_runtime_evidence": {"probe": "same"},
        "runtime_guard": {"runtime_guard_sha256": _h("guard")},
        "attempts": attempts,
    }
    result = validate_model_attempt_batch(
        batch, candidate_manifest=candidate, expected_accessions=accessions
    )
    assert observed["modes"] == [OWNED_HARDENED_TRANSPORT_MODE] * 2
    assert observed["guard_order"] == result["model_attempt_receipt_sha256s"]
    assert observed["guard_stage"] == "development"

    reordered = copy.deepcopy(batch)
    reordered["attempts"].reverse()
    with pytest.raises(SecFilingGemmaStageVerifierError, match="reordered"):
        validate_model_attempt_batch(
            reordered, candidate_manifest=candidate, expected_accessions=accessions
        )


def _score_replay() -> tuple[dict, dict, dict]:
    prefix = {"prediction_prefix_sha256": _h("prefix")}
    market = {"market_stage_manifest_sha256": _h("market")}
    receipt = {
        "configuration": {
            "selected_candidate_id": "p50_e0",
            "selected_variant": "semantic",
            "cost_bps": 10,
            "score_cutoff_session": "2023-12-29",
            "terminal_convention": "adjusted_open",
        },
        "score_receipt_sha256": _h("score"),
    }
    gate = {"gate_receipt_sha256": _h("gate"), "passed": True}
    replay = {
        "label_release_evidence": {
            "label_release_ledger_sha256": _h("labels")
        },
        "evaluations": [
            {
                "candidate_id": "p50_e0",
                "score_receipts": [receipt],
                "gate_receipt": gate,
            }
        ],
        "ranking_receipt": None,
        "selected_candidate_id": "p50_e0",
    }
    return replay, prefix, market


def test_raw_score_and_no_leverage_replay_precede_gate(monkeypatch) -> None:
    replay, prefix, market = _score_replay()
    calls: list[str] = []

    monkeypatch.setattr(
        verifier_module,
        "validate_score_receipt",
        lambda *args, **kwargs: calls.append("score") or kwargs[
            "expected_score_receipt_sha256"
        ],
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_sec_gemma_no_leverage_proof",
        lambda *args, **kwargs: calls.append("no_leverage")
        or {"proof_sha256": _h("proof")},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_stage_gate_receipt",
        lambda *args, **kwargs: calls.append("gate")
        or kwargs["expected_gate_receipt_sha256"],
    )
    result = validate_raw_scores_gates_and_ranking(
        replay,
        prediction_prefix=prefix,
        market_stage=market,
        prerequisite_stage="intermediate",
    )
    assert calls == ["score", "no_leverage", "gate"]
    assert result["no_leverage_proof_sha256s"] == [_h("proof")]


def test_score_cross_binding_and_raw_failure_never_reach_gate(monkeypatch) -> None:
    replay, prefix, market = _score_replay()
    replay["evaluations"][0]["score_receipts"][0]["configuration"][
        "selected_candidate_id"
    ] = "p55_e0"
    with pytest.raises(SecFilingGemmaStageVerifierError, match="crossed"):
        validate_raw_scores_gates_and_ranking(
            replay,
            prediction_prefix=prefix,
            market_stage=market,
            prerequisite_stage="intermediate",
        )

    replay, prefix, market = _score_replay()
    called_gate = False

    def fail_score(*args, **kwargs):
        raise ValueError("raw mismatch")

    def gate(*args, **kwargs):
        nonlocal called_gate
        called_gate = True

    monkeypatch.setattr(verifier_module, "validate_score_receipt", fail_score)
    monkeypatch.setattr(verifier_module, "validate_stage_gate_receipt", gate)
    with pytest.raises(SecFilingGemmaStageVerifierError, match="Raw score"):
        validate_raw_scores_gates_and_ranking(
            replay,
            prediction_prefix=prefix,
            market_stage=market,
            prerequisite_stage="intermediate",
        )
    assert called_gate is False


def _runtime_receipt(candidate_hash: str, model_summary: dict) -> dict:
    body = {
        "schema_version": STAGE_RUNTIME_RECEIPT_SCHEMA_VERSION,
        "stage": "development",
        "candidate_sha256": candidate_hash,
        "phase_elapsed_seconds": {
            "sec": 1.0.hex(),
            "model": 2.0.hex(),
            "fit_simulation_sealing": 3.0.hex(),
            "total": 6.0.hex(),
        },
        "sec_request_count": 4,
        "sec_response_bytes": 100,
        "model_attempt_receipt_sha256s": model_summary[
            "model_attempt_receipt_sha256s"
        ],
        "runtime_guard_sha256": model_summary["runtime_guard_sha256"],
        "paid_api_calls": 0,
        "estimated_cost_usd_hex": 0.0.hex(),
        "pull_attempts": 0,
        "retries": 0,
        "repair_attempts": 0,
    }
    return {**body, "runtime_receipt_sha256": canonical_sha256(body)}


def test_stage_specific_runtime_receipt_is_exact_zero_cost_and_bound() -> None:
    candidate_hash = _h("candidate")
    model = {
        "model_attempt_receipt_sha256s": [_h("one"), _h("two")],
        "runtime_guard_sha256": _h("guard"),
        "model_elapsed_nanoseconds": 1_000_000_000,
    }
    receipt = _runtime_receipt(candidate_hash, model)
    assert validate_stage_runtime_receipt(
        receipt,
        prerequisite_stage="development",
        candidate_sha256=candidate_hash,
        model_batch_summary=model,
    ) == receipt["runtime_receipt_sha256"]

    paid = copy.deepcopy(receipt)
    paid["paid_api_calls"] = 1
    paid["runtime_receipt_sha256"] = canonical_sha256(
        {key: paid[key] for key in paid if key != "runtime_receipt_sha256"}
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="paid_api_calls"):
        validate_stage_runtime_receipt(
            paid,
            prerequisite_stage="development",
            candidate_sha256=candidate_hash,
            model_batch_summary=model,
        )

    understated = copy.deepcopy(model)
    understated["model_elapsed_nanoseconds"] = 2_100_000_000
    with pytest.raises(SecFilingGemmaStageVerifierError, match="shorter"):
        validate_stage_runtime_receipt(
            receipt,
            prerequisite_stage="development",
            candidate_sha256=candidate_hash,
            model_batch_summary=understated,
        )


def _minimal_audit_envelope(candidate: dict, sources: dict[str, str]) -> dict:
    body = {
        "schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "prerequisite_stage": "development",
        "parent_stage_evidence_sha256": None,
        "contract_manifest": build_contract_manifest(),
        "candidate_manifest": candidate,
        "source_bytes_base64_by_role": sources,
        "calendar_evidence_manifest": {},
        "calendar_source_bytes_base64_by_name": {},
        "corpus_universe_manifest": {
            "records": [
                {
                    "accession_number": "0000320193-05-000001",
                    "artifact_stage": "development",
                    "availability_session": "2005-01-03",
                }
            ]
        },
        "prerequisite_content_manifest": {},
        "model_batches_by_stage": {"development": {}},
        "market_replays_by_stage": {
            "development": {"market_stage_manifest": {}}
        },
        "prediction_replay": {},
        "learner_replays": [],
        "score_replay": {},
        "stage_runtime_receipt": {},
        "reveal_registry": {},
        "registry_external_pin": {},
    }
    return {**body, "stage_evidence_sha256": canonical_sha256(body)}


def test_audit_lists_every_required_check_and_never_claims_authorization(monkeypatch) -> None:
    candidate, sources = _candidate_fixture()
    evidence = _minimal_audit_envelope(candidate, sources)
    candidate_hash = candidate["candidate_sha256"]
    monkeypatch.setattr(
        verifier_module,
        "validate_candidate_source_bytes",
        lambda **kwargs: {
            "candidate_sha256": candidate_hash,
            "stage_verifier_source_sha256": candidate["bindings"]["source_hashes"][
                "stage_verifier"
            ],
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_calendar_and_universe_snapshot",
        lambda **kwargs: {"corpus_universe_sha256": _h("universe")},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_model_attempt_batch",
        lambda *args, **kwargs: {
            "stage": "development",
            "model_attempt_receipt_sha256s": [_h("attempt")],
            "runtime_guard_sha256": _h("guard"),
            "model_elapsed_nanoseconds": 1,
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_market_snapshot_stage_replay",
        lambda *args, **kwargs: {"artifact_stage": "development"},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_prediction_artifact_replay",
        lambda *args, **kwargs: {"stage": "development", "prefix": {}},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_learner_refit_replays",
        lambda *args, **kwargs: {"learner_replay_sha256": _h("learner")},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_raw_scores_gates_and_ranking",
        lambda *args, **kwargs: {"selected_candidate_id": "p50_e0"},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_stage_runtime_receipt",
        lambda *args, **kwargs: _h("runtime"),
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_registry_request_and_stage_access",
        lambda **kwargs: {"request_sha256": _h("request")},
    )
    receipt = audit_stage_evidence(evidence, {}, {})
    assert receipt["semantic_checks"] == list(REQUIRED_STAGE_VERIFIER_CHECKS)
    assert set(receipt["check_status"]) == set(REQUIRED_STAGE_VERIFIER_CHECKS)
    assert receipt["check_status"]["artifact_seal_cas"]["status"] == "blocked"
    assert receipt["check_status"]["no_leverage"]["status"] == "replayed"
    assert receipt["check_status"]["artifact_replay"]["status"] == "blocked"
    assert receipt["check_status"]["prediction_replay"]["status"] == "blocked"
    assert receipt["check_status"]["stage_access_identity"]["status"] == "blocked"
    assert receipt["check_status"]["stage_identity"]["status"] == "replayed"
    assert receipt["blocking_check_count"] > 0
    assert receipt["all_checks_completed"] is False
    assert receipt["authorizes_outcome_access"] is False

    monkeypatch.setattr(
        verifier_module,
        "validate_model_attempt_batch",
        lambda *args, **kwargs: {
            "stage": "intermediate",
            "model_attempt_receipt_sha256s": [_h("attempt")],
            "runtime_guard_sha256": _h("guard"),
            "model_elapsed_nanoseconds": 1,
        },
    )
    with pytest.raises(SecFilingGemmaStageVerifierError, match="stage slot"):
        audit_stage_evidence(evidence, {}, {})


def test_intermediate_stage_identity_is_blocked_without_parent_winner_and_state_lineage(
    monkeypatch,
) -> None:
    candidate, sources = _candidate_fixture()
    evidence = _minimal_audit_envelope(candidate, sources)
    evidence["prerequisite_stage"] = "intermediate"
    evidence["parent_stage_evidence_sha256"] = _h("development evidence")
    evidence["model_batches_by_stage"] = {
        "development": {"stage": "development"},
        "intermediate": {"stage": "intermediate"},
    }
    evidence["market_replays_by_stage"] = {
        "development": {"stage": "development", "market_stage_manifest": {}},
        "intermediate": {"stage": "intermediate", "market_stage_manifest": {}},
    }
    evidence_body = {
        key: evidence[key] for key in evidence if key != "stage_evidence_sha256"
    }
    evidence["stage_evidence_sha256"] = canonical_sha256(evidence_body)
    candidate_hash = candidate["candidate_sha256"]

    monkeypatch.setattr(
        verifier_module,
        "validate_candidate_source_bytes",
        lambda **kwargs: {
            "candidate_sha256": candidate_hash,
            "stage_verifier_source_sha256": candidate["bindings"]["source_hashes"][
                "stage_verifier"
            ],
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_calendar_and_universe_snapshot",
        lambda **kwargs: {"corpus_universe_sha256": _h("universe")},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_model_attempt_batch",
        lambda batch, **kwargs: {
            "stage": batch["stage"],
            "model_attempt_receipt_sha256s": [_h(f"{batch['stage']} attempt")],
            "runtime_guard_sha256": _h(f"{batch['stage']} guard"),
            "model_elapsed_nanoseconds": 1,
        },
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_market_snapshot_stage_replay",
        lambda replay: {"artifact_stage": replay["stage"]},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_prediction_artifact_replay",
        lambda *args, **kwargs: {"stage": "intermediate", "prefix": {}},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_learner_refit_replays",
        lambda *args, **kwargs: {"learner_replay_sha256": _h("learner")},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_raw_scores_gates_and_ranking",
        lambda *args, **kwargs: {"selected_candidate_id": "p50_e0"},
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_stage_runtime_receipt",
        lambda *args, **kwargs: _h("runtime"),
    )
    monkeypatch.setattr(
        verifier_module,
        "validate_registry_request_and_stage_access",
        lambda **kwargs: {"request_sha256": _h("request")},
    )

    receipt = audit_stage_evidence(evidence, {}, {})
    stage_identity = receipt["check_status"]["stage_identity"]
    assert stage_identity["status"] == "blocked"
    assert "development winner" in stage_identity["reason"]
    assert "learner output state" in stage_identity["reason"]
    assert receipt["all_checks_completed"] is False
    assert receipt["authorizes_outcome_access"] is False


def test_authorizing_entrypoint_always_fails_closed(monkeypatch) -> None:
    audit = {
        "check_status": {
            check: {
                "status": "blocked" if check == "runtime_budget" else "replayed"
            }
            for check in REQUIRED_STAGE_VERIFIER_CHECKS
        }
    }
    monkeypatch.setattr(verifier_module, "audit_stage_evidence", lambda *args: audit)
    with pytest.raises(SecFilingGemmaStageVerifierBlocked, match="runtime_budget"):
        authoritative_prerequisite_validator({}, {}, {})


def test_stage_evidence_schema_omission_is_rejected_before_any_replay() -> None:
    candidate, sources = _candidate_fixture()
    evidence = _minimal_audit_envelope(candidate, sources)
    evidence.pop("learner_replays")
    with pytest.raises(SecFilingGemmaStageVerifierError, match="keys changed"):
        audit_stage_evidence(evidence, {}, {})
