from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import agent_benchmark.sec_filing_gemma_reveal_store as reveal_store_module
import agent_benchmark.sec_filing_gemma_stage_runner as stage_runner_module
from agent_benchmark.sec_filing_gemma_contract import (
    DIMENSION_NAMES,
    EXTRACTOR_SCHEMA_VERSION,
    FLAG_NAMES,
    canonical_sha256,
)
from agent_benchmark.sec_filing_gemma_features import (
    OWNED_DEVELOPMENT_FEATURE_INPUTS_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_reveal_store import (
    STAGE_OUTPUTS_DIRECTORY_NAME,
    SecFilingGemmaRevealStore,
    SecFilingGemmaRevealStoreError,
)
from agent_benchmark.sec_filing_gemma_stage_authorization import (
    validate_development_feature_assembly_plan,
)
from tests import test_sec_filing_gemma_reveal_store as scaffold


def _valid_extractor_output() -> dict:
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


def _attempt_response_bytes(output: dict) -> bytes:
    payload = {
        "model": scaffold.ollama_module.OLLAMA_MODEL,
        "created_at": "2026-07-13T10:11:12.123456789Z",
        "message": {
            "role": "assistant",
            "content": json.dumps(
                output,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 100,
        "load_duration": 10,
        "prompt_eval_count": 20,
        "prompt_eval_duration": 30,
        "eval_count": 40,
        "eval_duration": 50,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _file_tree_sha256s(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _walk_json(value: object, path: tuple[object, ...] = ()):
    yield path, value
    if type(value) is dict:
        for key, child in value.items():
            yield from _walk_json(child, (*path, key))
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from _walk_json(child, (*path, index))


def _complete_mixed_feature_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SecFilingGemmaRevealStore, dict, dict]:
    store = scaffold._store(tmp_path)
    digest, fingerprint, version_bytes, show_bytes = (
        scaffold._synthetic_runtime_probe_context()
    )
    _registered, _candidate, _universe, plan = (
        scaffold._registered_development_root(
            store,
            salt="owned-development-feature-inputs",
            model_digest=digest,
            runtime_fingerprint_sha256=fingerprint,
        )
    )
    scope_hash = plan["development_root_scope_sha256"]
    sec_claim = store.claim_owned_development_sec_root_execution(
        development_content_root_plan=plan,
        sec_user_agent_sha256=scaffold.SEC_TEST_USER_AGENT_SHA256,
    )["claim"]
    scaffold._write_fixed_development_sec_root(store, sec_claim, plan)
    store._record_owned_development_sec_root_reader_output(
        development_root_scope_sha256=scope_hash,
    )
    scaffold._complete_development_market_without_network(
        store,
        development_root_scope_sha256=scope_hash,
    )

    valid_output = _valid_extractor_output()
    invalid_response = scaffold._synthetic_attempt_response_bytes
    response_number = 0

    def mixed_attempt_response() -> bytes:
        nonlocal response_number
        response_number += 1
        if response_number == 1:
            return _attempt_response_bytes(valid_output)
        return invalid_response()

    monkeypatch.setattr(
        scaffold,
        "_synthetic_attempt_response_bytes",
        mixed_attempt_response,
    )
    scaffold._install_synthetic_owned_ollama(
        monkeypatch,
        version_bytes=version_bytes,
        show_bytes=show_bytes,
    )
    claim_result = store.claim_owned_development_model_execution(
        development_root_scope_sha256=scope_hash,
    )
    loaded_inputs = store._load_owned_development_model_event_inputs(
        development_root_scope_sha256=scope_hash,
    )
    stage_runner_module._execute_owned_model_batch(
        store,
        lifecycle_kind="development_root",
        lifecycle_sha256=scope_hash,
        claim=claim_result["claim"],
        loaded_inputs=loaded_inputs,
    )
    store._record_owned_development_model_reader_output(
        development_root_scope_sha256=scope_hash,
    )
    return store, plan, valid_output


def test_terminal_loader_projects_only_causal_safe_feature_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, development_plan, valid_output = _complete_mixed_feature_store(
        tmp_path,
        monkeypatch,
    )
    scope_hash = development_plan["development_root_scope_sha256"]
    signature = inspect.signature(
        store._load_owned_development_feature_inputs
    )
    assert list(signature.parameters) == ["development_root_scope_sha256"]
    assert signature.parameters[
        "development_root_scope_sha256"
    ].kind is inspect.Parameter.KEYWORD_ONLY

    state_before = store.state_path.read_bytes()
    tip_before = store.current_tip_anchor_path.read_bytes()
    tree_before = _file_tree_sha256s(store.store_directory)
    result = store._load_owned_development_feature_inputs(
        development_root_scope_sha256=scope_hash,
    )

    assert store.state_path.read_bytes() == state_before
    assert store.current_tip_anchor_path.read_bytes() == tip_before
    assert _file_tree_sha256s(store.store_directory) == tree_before
    assert set(result) == {
        "schema_version",
        "feature_assembly_plan",
        "events",
        "feature_inputs_sha256",
    }
    assert (
        result["schema_version"]
        == OWNED_DEVELOPMENT_FEATURE_INPUTS_SCHEMA_VERSION
    )
    result_body = {
        key: value
        for key, value in result.items()
        if key != "feature_inputs_sha256"
    }
    assert result["feature_inputs_sha256"] == canonical_sha256(result_body)

    feature_plan = result["feature_assembly_plan"]
    assert validate_development_feature_assembly_plan(
        feature_plan,
        expected_feature_assembly_plan_sha256=feature_plan[
            "feature_assembly_plan_sha256"
        ],
    ) == feature_plan["feature_assembly_plan_sha256"]
    assert feature_plan["development_root_scope_sha256"] == scope_hash
    assert feature_plan["start_consumed_request_count"] == 0
    assert feature_plan["canonical_market_rows_required"] is True
    assert feature_plan["feature_rows_output_permitted"] is True
    for denied_permission in (
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
        assert feature_plan[denied_permission] is False

    events = result["events"]
    assert len(events) == feature_plan["event_count"]
    assert [event["event_plan_item"] for event in events] == feature_plan[
        "event_plan"
    ]
    for ordinal, event in enumerate(events, start=1):
        assert set(event) == {
            "event_ordinal",
            "event_plan_item",
            "market_prefix",
            "market_prefix_proof",
            "universe_event_proof",
            "extraction_event_proof",
        }
        assert event["event_ordinal"] == ordinal
        event_plan_item = event["event_plan_item"]
        prefix = event["market_prefix"]
        proof = event["market_prefix_proof"]
        rows = prefix["lookback_rows"]
        decision_session = event_plan_item["availability_session"]
        assert prefix["decision_event_id"] == event_plan_item[
            "accession_number"
        ]
        assert prefix["decision_session"] == decision_session
        assert prefix["market_cutoff_session"] == decision_session
        assert prefix["full_prefix_last_session"] == decision_session
        assert prefix["lookback_last_session"] == decision_session
        assert prefix["lookback_row_count"] == 253
        assert prefix["maximum_feature_lookback_sessions"] == 252
        assert len(rows) == 253
        assert (
            prefix["lookback_end_row_index"]
            - prefix["lookback_start_row_index"]
            == 252
        )
        assert [row["session"] for row in rows] == sorted(
            row["session"] for row in rows
        )
        assert all(row["session"] <= decision_session for row in rows)
        assert rows[-1]["session"] == decision_session
        assert rows[-1]["row_sha256"] == prefix[
            "lookback_row_chain_tip_sha256"
        ]
        assert proof["decision_event_id"] == event_plan_item[
            "accession_number"
        ]
        assert proof["decision_session"] == decision_session
        assert proof["market_prefix_sha256"] == prefix[
            "market_prefix_sha256"
        ]
        assert proof["lookback_row_count"] == 253

    valid_proof = events[0]["extraction_event_proof"]
    assert valid_proof["extraction_status"] == "valid"
    assert valid_proof["extraction_evidence_authenticated"] is True
    assert valid_proof["validated_output"] == valid_output
    assert valid_proof["document_quality"] == "usable"
    assert valid_proof["extraction_output_sha256"] is not None
    assert valid_proof["extraction_output_canonical_sha256"] == canonical_sha256(
        valid_output
    )
    for event in events[1:]:
        invalid_proof = event["extraction_event_proof"]
        assert invalid_proof["extraction_status"] == "invalid"
        assert invalid_proof["extraction_evidence_authenticated"] is True
        assert invalid_proof["validated_output"] is None
        assert invalid_proof["document_quality"] is None
        assert invalid_proof["extraction_output_sha256"] is None
        assert invalid_proof["extraction_output_canonical_sha256"] is None

    forbidden_keys = {
        "authenticated_store_snapshot",
        "candidate_manifest",
        "content_manifests_by_stage",
        "corpus_universe_manifest",
        "current_normalized_text",
        "prior_same_form_normalized_text",
        "normalized_text",
        "raw_primary_document",
        "raw_response_bytes_by_symbol",
        "artifact_bytes_by_symbol",
        "window_bytes_by_symbol",
        "source_manifest",
        "stage_manifest",
        "request_bytes",
        "response_bytes",
        "extractor_output_bytes",
        "request_bytes_base64",
        "response_bytes_base64",
        "extractor_output_bytes_base64",
        "model_transport_envelope",
        "labels",
        "label_evidence",
        "outcomes",
        "outcome_rows",
        "holdout_payload",
        "training_membership",
        "training_membership_payload",
        "training_rows",
        "feature_rows",
        "predictions",
    }
    observed_key_paths: dict[str, list[tuple[object, ...]]] = {}
    valid_output_occurrences = 0
    for path, value in _walk_json(result):
        assert type(value) is not bytes
        if type(value) is dict:
            if value == valid_output:
                valid_output_occurrences += 1
            for key in value:
                observed_key_paths.setdefault(key, []).append((*path, key))
                assert key not in forbidden_keys
    assert valid_output_occurrences == 1
    assert observed_key_paths["validated_output"] == [
        ("events", ordinal, "extraction_event_proof", "validated_output")
        for ordinal in range(len(events))
    ]
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    assert "Synthetic development filing" not in encoded
    assert "raw-response-" not in encoded
    assert "regularMarketTime" not in encoded
    assert "request_bytes_base64" not in encoded
    assert "response_bytes_base64" not in encoded
    assert "extractor_output_bytes_base64" not in encoded

    feature_batch = stage_runner_module.run_owned_development_feature_batch(
        reveal_store=store,
        development_root_scope_sha256=scope_hash,
    )
    assert feature_batch["event_count"] == feature_plan["event_count"]
    assert len(feature_batch["feature_rows"]) == feature_plan["event_count"]
    assert feature_batch["feature_row_sha256s"] == [
        row["feature_row_sha256"] for row in feature_batch["feature_rows"]
    ]
    assert [row["accession_number"] for row in feature_batch["feature_rows"]] == [
        event["accession_number"] for event in feature_plan["event_plan"]
    ]
    assert [row["decision_session"] for row in feature_batch["feature_rows"]] == [
        event["availability_session"] for event in feature_plan["event_plan"]
    ]
    assert all(
        row["artifact_stage"] == "development"
        and row["decision_session"] <= feature_plan["development_cutoff_session"]
        for row in feature_batch["feature_rows"]
    )
    for nonauthorizing_flag in (
        "labels_included",
        "outcomes_included",
        "post_decision_market_rows_included",
        "training_membership_included",
        "learner_fit_authorized",
        "stage_promotion_authorized",
        "production_authorized",
    ):
        assert feature_batch[nonauthorizing_flag] is False
    forbidden_batch_keys = {
        "raw_response_bytes_by_symbol",
        "artifact_bytes_by_symbol",
        "window_bytes_by_symbol",
        "lookback_rows",
        "observations",
        "current_normalized_text",
        "prior_same_form_normalized_text",
        "primary_document_bytes",
        "normalized_text_bytes",
        "model_payload",
        "request_bytes_base64",
        "response_bytes_base64",
        "extractor_output_bytes_base64",
        "call_intent",
        "model_attempt_receipt",
        "runtime_guard",
        "label_evidence",
        "label_evidence_sha256",
        "outcome_rows",
        "future_market_rows",
        "holdout_payload",
        "training_set_membership",
        "learner_state",
        "prediction_rows",
    }
    batch_keys: set[str] = set()
    for _path, value in _walk_json(feature_batch):
        assert type(value) is not bytes
        if type(value) is dict:
            batch_keys.update(value)
    assert forbidden_batch_keys.isdisjoint(batch_keys)
    assert store.state_path.read_bytes() == state_before
    assert store.current_tip_anchor_path.read_bytes() == tip_before
    assert _file_tree_sha256s(store.store_directory) == tree_before

    current = store.load()
    current_tip = store.load_current_tip_anchor()
    consumed_tip = copy.deepcopy(current_tip)
    consumed_tip["consumed_request_count"] = 1
    with patch.object(
        store,
        "_read_state_and_tip_locked",
        return_value=(current, consumed_tip, b"", b""),
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="zero consumed reveals",
    ):
        store._load_owned_development_feature_inputs(
            development_root_scope_sha256=scope_hash,
        )

    active_tip = copy.deepcopy(current_tip)
    active_tip["stage_sec_execution_claims"]["f" * 64] = {}
    with patch.object(
        store,
        "_read_state_and_tip_locked",
        return_value=(current, active_tip, b"", b""),
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="active stage SEC effect",
    ):
        store._load_owned_development_feature_inputs(
            development_root_scope_sha256=scope_hash,
        )

    cross_root_tip = copy.deepcopy(current_tip)
    cross_root_tip["development_market_execution_claims"][scope_hash][
        "development_root_scope_sha256"
    ] = "e" * 64
    with patch.object(
        store,
        "_read_state_and_tip_locked",
        return_value=(current, cross_root_tip, b"", b""),
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="crossed their root scope",
    ):
        store._load_owned_development_feature_inputs(
            development_root_scope_sha256=scope_hash,
        )

    assert store.state_path.read_bytes() == state_before
    assert store.current_tip_anchor_path.read_bytes() == tip_before
    tip = store.load_current_tip_anchor()
    claim = tip["development_model_execution_claims"][scope_hash]
    component = (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / claim["claim_sha256"]
        / reveal_store_module.MODEL_EXTRACTION_COMPONENT_DIRECTORY_NAME
    )
    scaffold._coherently_mutate_model_artifact(
        component,
        "events/000001/model_attempt_receipt.json",
    )
    with pytest.raises(SecFilingGemmaRevealStoreError):
        store._load_owned_development_feature_inputs(
            development_root_scope_sha256=scope_hash,
        )
    assert store.state_path.read_bytes() == state_before
    assert store.current_tip_anchor_path.read_bytes() == tip_before


def test_feature_loader_rejects_active_and_aborted_model_effects(
    tmp_path: Path,
) -> None:
    store = scaffold._store(tmp_path)
    prepared = scaffold._prepare_development_model_claim(
        store,
        salt="feature-inputs-active-and-aborted",
    )
    scope_hash = prepared["plan"]["development_root_scope_sha256"]
    stable_state = store.state_path.read_bytes()
    tip_with_active_claim = store.current_tip_anchor_path.read_bytes()

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="active development model effect",
    ):
        store._load_owned_development_feature_inputs(
            development_root_scope_sha256=scope_hash,
        )
    assert store.state_path.read_bytes() == stable_state
    assert store.current_tip_anchor_path.read_bytes() == tip_with_active_claim

    store.abort_owned_development_model_execution(
        development_root_scope_sha256=scope_hash,
        reason="external_effect_failed_or_completion_unknown",
    )
    tip_with_abort = store.current_tip_anchor_path.read_bytes()
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="cannot use an aborted effect",
    ):
        store._load_owned_development_feature_inputs(
            development_root_scope_sha256=scope_hash,
        )
    assert store.state_path.read_bytes() == stable_state
    assert store.current_tip_anchor_path.read_bytes() == tip_with_abort
