from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

import agent_benchmark.sec_filing_gemma_features as features_module
import agent_benchmark.sec_filing_gemma_reveal_store as reveal_store_module
from agent_benchmark.sec_filing_gemma_contract import (
    SecFilingGemmaContractError,
    canonical_sha256,
)
from agent_benchmark.sec_filing_gemma_features import (
    OWNED_DEVELOPMENT_LABEL_PROJECTION_SCHEMA_VERSION,
    SecFilingGemmaFeatureError,
    validate_owned_development_feature_batch,
)
from agent_benchmark.sec_filing_gemma_reveal_store import (
    SecFilingGemmaRevealStore,
    SecFilingGemmaRevealStoreError,
)
from agent_benchmark.sec_filing_gemma_stage_authorization import (
    validate_development_label_assembly_plan,
)
from tests import test_sec_filing_gemma_feature_assembly_store as feature_scaffold
from tests import test_sec_filing_gemma_reveal_store as store_scaffold


def _walk_json(value: object, path: tuple[object, ...] = ()):
    yield path, value
    if type(value) is dict:
        for key, child in value.items():
            yield from _walk_json(child, (*path, key))
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from _walk_json(child, (*path, index))


def test_terminal_label_projection_is_mature_safe_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, development_plan, _valid_output = (
        feature_scaffold._complete_mixed_feature_store(
            tmp_path,
            monkeypatch,
        )
    )
    scope_hash = development_plan["development_root_scope_sha256"]
    signature = inspect.signature(
        store._load_owned_development_label_projection
    )
    assert list(signature.parameters) == ["development_root_scope_sha256"]
    assert signature.parameters[
        "development_root_scope_sha256"
    ].kind is inspect.Parameter.KEYWORD_ONLY

    state_before = store.state_path.read_bytes()
    tip_before = store.current_tip_anchor_path.read_bytes()
    tree_before = feature_scaffold._file_tree_sha256s(store.store_directory)

    original_builder = features_module.build_twenty_session_label_evidence
    builder_calls: list[tuple[str, str]] = []

    def observing_builder(**kwargs: Any) -> dict[str, Any]:
        evidence = original_builder(**kwargs)
        builder_calls.append(
            (
                kwargs["market_prefix"]["decision_session"],
                evidence["label_maturity_session"],
            )
        )
        return evidence

    # The store calls its imported alias once, and the validator rebuilds via
    # the feature module once.  Observing both proves neither path opens an
    # immature event.
    monkeypatch.setattr(
        reveal_store_module,
        "build_twenty_session_label_evidence",
        observing_builder,
    )
    monkeypatch.setattr(
        features_module,
        "build_twenty_session_label_evidence",
        observing_builder,
    )

    result = store._load_owned_development_label_projection(
        development_root_scope_sha256=scope_hash,
    )

    assert store.state_path.read_bytes() == state_before
    assert store.current_tip_anchor_path.read_bytes() == tip_before
    assert feature_scaffold._file_tree_sha256s(store.store_directory) == tree_before
    assert set(result) == {
        "schema_version",
        "label_assembly_plan",
        "source_feature_batch",
        "maturity_audit_rows",
        "label_evidence_rows",
        "label_projection_sha256",
    }
    assert (
        result["schema_version"]
        == OWNED_DEVELOPMENT_LABEL_PROJECTION_SCHEMA_VERSION
    )
    projection_body = {
        key: value
        for key, value in result.items()
        if key != "label_projection_sha256"
    }
    assert result["label_projection_sha256"] == canonical_sha256(
        projection_body
    )

    label_plan = result["label_assembly_plan"]
    assert validate_development_label_assembly_plan(
        label_plan,
        expected_label_assembly_plan_sha256=label_plan[
            "label_assembly_plan_sha256"
        ],
    ) == label_plan["label_assembly_plan_sha256"]
    source_feature_plan = label_plan["source_feature_assembly_plan"]
    source_feature_batch = result["source_feature_batch"]
    assert validate_owned_development_feature_batch(
        source_feature_batch,
        feature_assembly_plan=source_feature_plan,
        expected_feature_assembly_plan_sha256=source_feature_plan[
            "feature_assembly_plan_sha256"
        ],
        expected_feature_batch_sha256=source_feature_batch[
            "feature_batch_sha256"
        ],
    ) == source_feature_batch["feature_batch_sha256"]
    assert source_feature_batch["labels_included"] is False
    assert source_feature_batch["outcomes_included"] is False
    assert source_feature_batch["learner_fit_authorized"] is False
    assert source_feature_batch["stage_promotion_authorized"] is False
    assert source_feature_batch["production_authorized"] is False

    audit_rows = result["maturity_audit_rows"]
    evidence_rows = result["label_evidence_rows"]
    assert len(audit_rows) == label_plan["event_count"]
    assert len(evidence_rows) == label_plan["matured_event_count"]
    assert len(builder_calls) == 2 * label_plan["matured_event_count"]
    assert all(maturity <= "2018-12-31" for _decision, maturity in builder_calls)
    assert [row["event_ordinal"] for row in audit_rows] == list(
        range(1, len(audit_rows) + 1)
    )
    evidence_by_hash = {
        row["label_evidence_sha256"]: row for row in evidence_rows
    }
    assert len(evidence_by_hash) == len(evidence_rows)
    source_feature_hashes = source_feature_batch["feature_row_sha256s"]
    for audit, maturity_item, feature_hash in zip(
        audit_rows,
        label_plan["maturity_plan"],
        source_feature_hashes,
        strict=True,
    ):
        assert set(audit) == {
            "event_ordinal",
            "accession_number",
            "decision_session",
            "feature_row_sha256",
            "label_maturity_session",
            "matured_by_development_cutoff",
            "label_evidence_sha256",
        }
        assert audit["event_ordinal"] == maturity_item["event_ordinal"]
        assert audit["accession_number"] == maturity_item["accession_number"]
        assert audit["decision_session"] == maturity_item["decision_session"]
        assert audit["feature_row_sha256"] == feature_hash
        assert audit["label_maturity_session"] == maturity_item[
            "label_maturity_session"
        ]
        assert audit["matured_by_development_cutoff"] is maturity_item[
            "matured_by_development_cutoff"
        ]
        if audit["matured_by_development_cutoff"]:
            assert audit["label_maturity_session"] <= "2018-12-31"
            evidence = evidence_by_hash[audit["label_evidence_sha256"]]
            assert evidence["accession_number"] == audit["accession_number"]
            assert evidence["decision_session"] == audit["decision_session"]
            assert evidence["feature_row_sha256"] == audit["feature_row_sha256"]
            assert evidence["label_maturity_session"] == audit[
                "label_maturity_session"
            ]
            assert evidence["future_market_row_count"] == 21
            assert all(
                item["session"] <= "2018-12-31"
                for item in evidence["adjusted_open_path"]
            )
        else:
            assert audit["label_maturity_session"] > "2018-12-31"
            assert audit["label_evidence_sha256"] is None

    forbidden_keys = {
        "events",
        "market_prefix",
        "market_prefix_proof",
        "universe_event_proof",
        "extraction_event_proof",
        "lookback_rows",
        "observations",
        "stage_manifest",
        "source_manifest",
        "raw_response_bytes_by_symbol",
        "artifact_bytes_by_symbol",
        "window_bytes_by_symbol",
        "current_normalized_text",
        "prior_same_form_normalized_text",
        "model_payload",
        "request_bytes_base64",
        "response_bytes_base64",
        "model_attempt_receipt",
        "training_membership",
        "learner_state",
        "prediction_rows",
        "consumption_ledger",
    }
    observed_keys: set[str] = set()
    for _path, value in _walk_json(result):
        assert type(value) is not bytes
        if type(value) is dict:
            observed_keys.update(value)
    assert forbidden_keys.isdisjoint(observed_keys)
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    assert "Synthetic development filing" not in encoded
    assert "raw-response-" not in encoded
    assert "regularMarketTime" not in encoded
    assert "request_bytes_base64" not in encoded
    assert "response_bytes_base64" not in encoded


def _install_synthetic_label_projection(
    store: SecFilingGemmaRevealStore,
    monkeypatch: pytest.MonkeyPatch,
    *,
    label_builder: Any,
) -> tuple[str, dict[str, Any]]:
    scope_hash = "9" * 64
    source_hash = "1" * 64
    stage_hash = "2" * 64
    feature_plan_hash = "3" * 64
    feature_plan = {
        "feature_assembly_plan_sha256": feature_plan_hash,
        "event_count": 2,
        "development_market_source_manifest_sha256": source_hash,
        "development_market_stage_manifest_sha256": stage_hash,
    }
    event_plan_items = [
        {
            "event_ordinal": 1,
            "accession_number": "0000320193-18-000001",
            "form": "10-K",
            "availability_session": "2018-01-02",
            "sec_document_ordinal": 1,
        },
        {
            "event_ordinal": 2,
            "accession_number": "0000320193-18-000002",
            "form": "10-Q",
            "availability_session": "2018-12-20",
            "sec_document_ordinal": 2,
        },
    ]
    events = [
        {
            "event_ordinal": item["event_ordinal"],
            "event_plan_item": copy.deepcopy(item),
            "market_prefix": {
                "decision_session": item["availability_session"],
            },
            "market_prefix_proof": {
                "market_prefix_proof_sha256": str(item["event_ordinal"]) * 64,
            },
            "universe_event_proof": {
                "universe_event_proof_sha256": chr(
                    ord("a") + item["event_ordinal"] - 1
                )
                * 64,
            },
            "extraction_event_proof": {
                "extraction_event_proof_sha256": chr(
                    ord("c") + item["event_ordinal"] - 1
                )
                * 64,
            },
        }
        for item in event_plan_items
    ]
    feature_inputs = {
        "schema_version": "synthetic-feature-inputs",
        "feature_assembly_plan": feature_plan,
        "events": events,
        "feature_inputs_sha256": "4" * 64,
    }
    feature_rows = [
        {
            "accession_number": item["accession_number"],
            "decision_session": item["availability_session"],
            "feature_row_sha256": chr(ord("e") + index) * 64,
        }
        for index, item in enumerate(event_plan_items)
    ]
    feature_batch = {
        "event_count": 2,
        "feature_rows": feature_rows,
        "feature_batch_sha256": "7" * 64,
    }
    maturity_plan = [
        {
            "event_ordinal": 1,
            "accession_number": event_plan_items[0]["accession_number"],
            "form": event_plan_items[0]["form"],
            "decision_session": event_plan_items[0]["availability_session"],
            "sec_document_ordinal": 1,
            "label_maturity_session": "2018-02-01",
            "matured_by_development_cutoff": True,
        },
        {
            "event_ordinal": 2,
            "accession_number": event_plan_items[1]["accession_number"],
            "form": event_plan_items[1]["form"],
            "decision_session": event_plan_items[1]["availability_session"],
            "sec_document_ordinal": 2,
            "label_maturity_session": "2019-01-23",
            "matured_by_development_cutoff": False,
        },
    ]
    label_plan = {
        "development_root_scope_sha256": scope_hash,
        "start_consumed_request_count": 0,
        "source_feature_assembly_plan": feature_plan,
        "source_feature_assembly_plan_sha256": feature_plan_hash,
        "event_count": 2,
        "maturity_plan": maturity_plan,
        "matured_event_count": 1,
        "unmatured_event_count": 1,
        "label_assembly_plan_sha256": "8" * 64,
    }
    market_claim = {
        "development_root_scope_sha256": scope_hash,
        "claim_sha256": "a" * 64,
    }
    market_reader = {
        "development_root_scope_sha256": scope_hash,
        "claim_sha256": market_claim["claim_sha256"],
    }
    model_claim = {
        "development_root_scope_sha256": scope_hash,
    }
    current_tip = {
        "development_market_execution_claims": {scope_hash: market_claim},
        "development_market_reader_receipts": {scope_hash: market_reader},
        "development_market_execution_aborts": {},
        "development_model_execution_claims": {scope_hash: model_claim},
    }
    market_replay = {
        "source_manifest": {"source_manifest_sha256": source_hash},
        "stage_manifest": {"market_stage_manifest_sha256": stage_hash},
    }

    monkeypatch.setattr(
        store,
        "_load_owned_development_feature_inputs_locked",
        lambda **_kwargs: copy.deepcopy(feature_inputs),
    )
    monkeypatch.setattr(
        store,
        "_build_owned_development_feature_batch_from_inputs_locked",
        lambda **_kwargs: copy.deepcopy(feature_batch),
    )
    monkeypatch.setattr(
        reveal_store_module,
        "_load_tracked_anchor",
        lambda _root: {},
    )
    monkeypatch.setattr(
        store,
        "_read_state_and_tip_locked",
        lambda _anchor: ({}, copy.deepcopy(current_tip), b"", b""),
    )
    monkeypatch.setattr(
        reveal_store_module,
        "build_development_label_assembly_plan",
        lambda *_args, **_kwargs: copy.deepcopy(label_plan),
    )
    monkeypatch.setattr(
        reveal_store_module,
        "validate_development_label_assembly_plan",
        lambda *_args, **_kwargs: label_plan["label_assembly_plan_sha256"],
    )
    monkeypatch.setattr(
        store,
        "_replay_owned_development_market_component_locked",
        lambda **_kwargs: copy.deepcopy(market_replay),
    )
    monkeypatch.setattr(
        store,
        "_revalidate_authorized_market_execution_sources",
        lambda _claim: None,
    )
    monkeypatch.setattr(
        store,
        "_revalidate_authorized_model_execution_sources",
        lambda _claim: None,
    )
    monkeypatch.setattr(
        reveal_store_module,
        "build_twenty_session_label_evidence",
        label_builder,
    )
    monkeypatch.setattr(
        reveal_store_module,
        "validate_twenty_session_label_evidence",
        lambda evidence, **_kwargs: evidence["label_evidence_sha256"],
    )
    return scope_hash, label_plan


def test_label_projection_never_opens_an_immature_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_scaffold._store(tmp_path)
    calls: list[str] = []

    def one_mature_label(**kwargs: Any) -> dict[str, Any]:
        decision = kwargs["market_prefix"]["decision_session"]
        calls.append(decision)
        assert decision == "2018-01-02"
        body = {
            "accession_number": "0000320193-18-000001",
            "decision_session": decision,
            "label_maturity_session": "2018-02-01",
            "feature_row_sha256": "e" * 64,
        }
        return {**body, "label_evidence_sha256": canonical_sha256(body)}

    scope_hash, _label_plan = _install_synthetic_label_projection(
        store,
        monkeypatch,
        label_builder=one_mature_label,
    )
    result = store._load_owned_development_label_projection(
        development_root_scope_sha256=scope_hash,
    )

    assert calls == ["2018-01-02"]
    assert len(result["label_evidence_rows"]) == 1
    assert result["maturity_audit_rows"][0]["label_evidence_sha256"] is not None
    assert result["maturity_audit_rows"][1] == {
        "event_ordinal": 2,
        "accession_number": "0000320193-18-000002",
        "decision_session": "2018-12-20",
        "feature_row_sha256": "f" * 64,
        "label_maturity_session": "2019-01-23",
        "matured_by_development_cutoff": False,
        "label_evidence_sha256": None,
    }


@pytest.mark.parametrize("changed_source", ("feature_inputs", "market_replay"))
def test_label_projection_rejects_changed_post_label_source_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_source: str,
) -> None:
    store = store_scaffold._store(tmp_path)

    def one_mature_label(**kwargs: Any) -> dict[str, Any]:
        body = {
            "accession_number": "0000320193-18-000001",
            "decision_session": kwargs["market_prefix"]["decision_session"],
            "label_maturity_session": "2018-02-01",
            "feature_row_sha256": "e" * 64,
        }
        return {**body, "label_evidence_sha256": canonical_sha256(body)}

    scope_hash, _label_plan = _install_synthetic_label_projection(
        store,
        monkeypatch,
        label_builder=one_mature_label,
    )
    if changed_source == "feature_inputs":
        stable_loader = store._load_owned_development_feature_inputs_locked
        load_count = 0

        def changing_feature_inputs(**kwargs: Any) -> dict[str, Any]:
            nonlocal load_count
            load_count += 1
            value = stable_loader(**kwargs)
            if load_count == 2:
                value["feature_inputs_sha256"] = "0" * 64
            return value

        monkeypatch.setattr(
            store,
            "_load_owned_development_feature_inputs_locked",
            changing_feature_inputs,
        )
    else:
        stable_replay = store._replay_owned_development_market_component_locked
        replay_count = 0

        def changing_market_replay(**kwargs: Any) -> dict[str, Any]:
            nonlocal replay_count
            replay_count += 1
            value = stable_replay(**kwargs)
            if replay_count == 2:
                value["closure_marker"] = "changed"
            return value

        monkeypatch.setattr(
            store,
            "_replay_owned_development_market_component_locked",
            changing_market_replay,
        )

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="source ancestry changed",
    ):
        store._load_owned_development_label_projection(
            development_root_scope_sha256=scope_hash,
        )


@pytest.mark.parametrize(
    ("error_type", "failure"),
    (
        (
            SecFilingGemmaFeatureError,
            "AAPL label path contains a missing open",
        ),
        (
            SecFilingGemmaFeatureError,
            "AAPL label path contains an invalid open",
        ),
        (
            SecFilingGemmaFeatureError,
            "Market row checksum does not reconcile",
        ),
        (
            SecFilingGemmaContractError,
            "Market stage source manifest checksum changed",
        ),
    ),
)
def test_label_target_failure_is_whole_projection_and_contextual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    failure: str,
) -> None:
    store = store_scaffold._store(tmp_path)

    def failed_label(**_kwargs: Any) -> dict[str, Any]:
        raise error_type(failure)

    scope_hash, _label_plan = _install_synthetic_label_projection(
        store,
        monkeypatch,
        label_builder=failed_label,
    )
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match=(
            r"event 1 \(0000320193-18-000001\) "
            r"decision=2018-01-02 maturity=2018-02-01"
        ),
    ) as caught:
        store._load_owned_development_label_projection(
            development_root_scope_sha256=scope_hash,
        )
    assert failure in str(caught.value)
