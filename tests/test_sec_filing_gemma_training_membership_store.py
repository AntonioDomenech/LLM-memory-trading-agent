from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import pytest

import agent_benchmark.sec_filing_gemma_reveal_store as reveal_store_module
from agent_benchmark.sec_filing_gemma_contract import canonical_sha256
from agent_benchmark.sec_filing_gemma_features import (
    OWNED_DEVELOPMENT_LABEL_PROJECTION_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_reveal_store import (
    SecFilingGemmaRevealStore,
    SecFilingGemmaRevealStoreError,
)
from agent_benchmark.sec_filing_gemma_training_membership import (
    OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_PROJECTION_SCHEMA_VERSION,
)
from tests import test_sec_filing_gemma_reveal_store as store_scaffold


def _tree_sha256s(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _walk_json(value: object):
    yield value
    if type(value) is dict:
        for child in value.values():
            yield from _walk_json(child)
    elif type(value) is list:
        for child in value:
            yield from _walk_json(child)


def _synthetic_membership_sources() -> tuple[
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    scope_hash = "9" * 64
    feature_plan_hash = "1" * 64
    label_plan_hash = "2" * 64
    feature_batch_hash = "3" * 64
    label_batch_hash = "4" * 64
    candidate_hash = "5" * 64
    universe_hash = "6" * 64
    feature_plan = {
        "development_root_scope_sha256": scope_hash,
        "feature_assembly_plan_sha256": feature_plan_hash,
        "candidate_sha256": candidate_hash,
        "corpus_universe_sha256": universe_hash,
    }
    label_plan = {
        "development_root_scope_sha256": scope_hash,
        "start_consumed_request_count": 0,
        "source_feature_assembly_plan": copy.deepcopy(feature_plan),
        "source_feature_assembly_plan_sha256": feature_plan_hash,
        "label_assembly_plan_sha256": label_plan_hash,
        "development_cutoff_session": "2018-12-31",
        "event_count": 3,
        "matured_event_count": 2,
        "unmatured_event_count": 1,
    }
    feature_rows = [
        {
            "accession_number": "0000320193-04-000001",
            "decision_session": "2004-01-02",
            "prediction_available": True,
            "fit_eligible": True,
            "unavailable_reasons": [],
            "feature_row_sha256": "a" * 64,
        },
        {
            "accession_number": "0000320193-08-000002",
            "decision_session": "2008-01-02",
            "prediction_available": False,
            "fit_eligible": False,
            "unavailable_reasons": ["unauthenticated_extraction_evidence"],
            "feature_row_sha256": "b" * 64,
        },
        {
            "accession_number": "0000320193-18-000003",
            "decision_session": "2018-12-20",
            "prediction_available": True,
            "fit_eligible": True,
            "unavailable_reasons": [],
            "feature_row_sha256": "c" * 64,
        },
    ]
    source_feature_batch = {
        "development_root_scope_sha256": scope_hash,
        "feature_assembly_plan_sha256": feature_plan_hash,
        "candidate_sha256": candidate_hash,
        "corpus_universe_sha256": universe_hash,
        "event_count": 3,
        "feature_rows": feature_rows,
        "feature_batch_sha256": feature_batch_hash,
    }
    maturity_audit_rows = [
        {
            "event_ordinal": 1,
            "accession_number": feature_rows[0]["accession_number"],
            "decision_session": feature_rows[0]["decision_session"],
            "feature_row_sha256": feature_rows[0]["feature_row_sha256"],
            "label_maturity_session": "2004-02-03",
            "matured_by_development_cutoff": True,
            "label_evidence_sha256": "d" * 64,
        },
        {
            "event_ordinal": 2,
            "accession_number": feature_rows[1]["accession_number"],
            "decision_session": feature_rows[1]["decision_session"],
            "feature_row_sha256": feature_rows[1]["feature_row_sha256"],
            "label_maturity_session": "2008-02-01",
            "matured_by_development_cutoff": True,
            "label_evidence_sha256": "e" * 64,
        },
        {
            "event_ordinal": 3,
            "accession_number": feature_rows[2]["accession_number"],
            "decision_session": feature_rows[2]["decision_session"],
            "feature_row_sha256": feature_rows[2]["feature_row_sha256"],
            "label_maturity_session": "2019-01-23",
            "matured_by_development_cutoff": False,
            "label_evidence_sha256": None,
        },
    ]
    label_evidence_rows = [
        {
            "accession_number": audit["accession_number"],
            "feature_row_sha256": audit["feature_row_sha256"],
            "label_evidence_sha256": audit["label_evidence_sha256"],
            "adjusted_open_path": [
                {
                    "session": audit["label_maturity_session"],
                    "adjusted_open_hex": float(100 + index).hex(),
                    "source_market_row_sha256": str(index + 1) * 64,
                }
            ],
        }
        for index, audit in enumerate(maturity_audit_rows[:2])
    ]
    source_label_batch = {
        "development_root_scope_sha256": scope_hash,
        "label_assembly_plan_sha256": label_plan_hash,
        "source_feature_assembly_plan_sha256": feature_plan_hash,
        "source_feature_batch_sha256": feature_batch_hash,
        "candidate_sha256": candidate_hash,
        "corpus_universe_sha256": universe_hash,
        "development_cutoff_session": "2018-12-31",
        "event_count": 3,
        "matured_label_count": 2,
        "unmatured_event_count": 1,
        "maturity_audit_rows": maturity_audit_rows,
        "label_evidence_rows": label_evidence_rows,
        "label_batch_sha256": label_batch_hash,
    }
    label_projection_body = {
        "schema_version": OWNED_DEVELOPMENT_LABEL_PROJECTION_SCHEMA_VERSION,
        "label_assembly_plan": label_plan,
        "source_feature_batch": source_feature_batch,
        "maturity_audit_rows": maturity_audit_rows,
        "label_evidence_rows": label_evidence_rows,
    }
    label_projection = {
        **label_projection_body,
        "label_projection_sha256": canonical_sha256(label_projection_body),
    }
    membership_plan_body = {
        "schema_version": (
            "aapl-sec-gemma-development-training-membership-assembly-plan-v1"
        ),
        "development_root_scope_sha256": scope_hash,
        "start_consumed_request_count": 0,
        "source_label_assembly_plan": copy.deepcopy(label_plan),
        "source_label_assembly_plan_sha256": label_plan_hash,
        "source_feature_assembly_plan_sha256": feature_plan_hash,
        "candidate_sha256": candidate_hash,
        "corpus_universe_sha256": universe_hash,
        "development_cutoff_session": "2018-12-31",
        "event_count": 3,
        "matured_event_count": 2,
        "unmatured_event_count": 1,
        "training_membership_access_permitted": True,
        "training_membership_rows_output_permitted": True,
        "training_feature_matrices_output_permitted": True,
        "training_target_vectors_output_permitted": True,
        "outcome_based_membership_filtering_permitted": False,
        "learner_fit_permitted": False,
        "prediction_access_permitted": False,
        "holdout_access_permitted": False,
        "ledger_mutation_permitted": False,
        "stage_promotion_permitted": False,
        "production_permitted": False,
    }
    membership_plan = {
        **membership_plan_body,
        "training_membership_assembly_plan_sha256": canonical_sha256(
            membership_plan_body
        ),
    }
    return scope_hash, label_projection, source_label_batch, membership_plan


def _install_synthetic_membership_projection(
    store: SecFilingGemmaRevealStore,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None]
    | None = None,
) -> tuple[str, dict[str, int]]:
    scope_hash, label_projection, source_label_batch, membership_plan = (
        _synthetic_membership_sources()
    )
    if mutate is not None:
        mutate(label_projection, source_label_batch, membership_plan)
    calls = {"locked_label": 0, "label_batch": 0, "plan": 0, "validate": 0}

    def load_locked_label(**kwargs: Any) -> dict[str, Any]:
        calls["locked_label"] += 1
        assert kwargs == {"development_root_scope_sha256": scope_hash}
        return copy.deepcopy(label_projection)

    def build_label_batch(**kwargs: Any) -> dict[str, Any]:
        calls["label_batch"] += 1
        assert kwargs == {"label_projection": label_projection}
        return copy.deepcopy(source_label_batch)

    def build_plan(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["plan"] += 1
        assert kwargs["development_root_scope_sha256"] == scope_hash
        assert kwargs["source_label_assembly_plan"] == label_projection[
            "label_assembly_plan"
        ]
        return copy.deepcopy(membership_plan)

    def validate_plan(
        plan: Any,
        *,
        expected_training_membership_assembly_plan_sha256: str,
    ) -> str:
        calls["validate"] += 1
        assert plan == membership_plan
        assert expected_training_membership_assembly_plan_sha256 == plan[
            "training_membership_assembly_plan_sha256"
        ]
        return expected_training_membership_assembly_plan_sha256

    def forbidden_loader(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("membership projection attempted a recursive public loader")

    monkeypatch.setattr(
        store,
        "_load_owned_development_label_projection_locked",
        load_locked_label,
    )
    monkeypatch.setattr(
        store,
        "_build_owned_development_label_batch_from_projection_locked",
        build_label_batch,
    )
    monkeypatch.setattr(
        store,
        "_load_owned_development_label_projection",
        forbidden_loader,
    )
    monkeypatch.setattr(
        store,
        "_load_owned_development_feature_inputs",
        forbidden_loader,
    )
    monkeypatch.setattr(
        store,
        "_load_owned_development_feature_inputs_locked",
        forbidden_loader,
    )
    monkeypatch.setattr(
        reveal_store_module,
        "build_development_training_membership_assembly_plan",
        build_plan,
    )
    monkeypatch.setattr(
        reveal_store_module,
        "validate_development_training_membership_assembly_plan",
        validate_plan,
    )
    return scope_hash, calls


def test_membership_projection_is_exact_read_only_and_preserves_complete_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    scope_hash, calls = _install_synthetic_membership_projection(
        store,
        monkeypatch,
    )
    signature = inspect.signature(
        store._load_owned_development_training_membership_projection
    )
    assert list(signature.parameters) == ["development_root_scope_sha256"]
    assert signature.parameters[
        "development_root_scope_sha256"
    ].kind is inspect.Parameter.KEYWORD_ONLY
    state_before = store.state_path.read_bytes()
    tip_before = store.current_tip_anchor_path.read_bytes()
    tree_before = _tree_sha256s(store.store_directory)

    result = store._load_owned_development_training_membership_projection(
        development_root_scope_sha256=scope_hash,
    )

    assert calls == {
        "locked_label": 1,
        "label_batch": 1,
        "plan": 1,
        "validate": 1,
    }
    assert set(result) == {
        "schema_version",
        "training_membership_assembly_plan",
        "source_feature_batch",
        "source_label_batch",
        "membership_projection_sha256",
    }
    assert (
        result["schema_version"]
        == OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_PROJECTION_SCHEMA_VERSION
    )
    body = {
        key: result[key]
        for key in result
        if key != "membership_projection_sha256"
    }
    assert result["membership_projection_sha256"] == canonical_sha256(body)
    assert result["training_membership_assembly_plan"][
        "learner_fit_permitted"
    ] is False
    feature_rows = result["source_feature_batch"]["feature_rows"]
    audit_rows = result["source_label_batch"]["maturity_audit_rows"]
    label_rows = result["source_label_batch"]["label_evidence_rows"]
    assert len(feature_rows) == len(audit_rows) == 3
    assert len(label_rows) == 2
    assert feature_rows[1]["fit_eligible"] is False
    assert feature_rows[1]["unavailable_reasons"] == [
        "unauthenticated_extraction_evidence"
    ]
    assert audit_rows[1]["matured_by_development_cutoff"] is True
    assert audit_rows[1]["label_evidence_sha256"] == label_rows[1][
        "label_evidence_sha256"
    ]
    assert audit_rows[2]["matured_by_development_cutoff"] is False
    assert audit_rows[2]["label_evidence_sha256"] is None

    forbidden_keys = {
        "lookback_rows",
        "observations",
        "stage_manifest",
        "source_manifest",
        "raw_response_bytes_by_symbol",
        "artifact_bytes_by_symbol",
        "window_bytes_by_symbol",
        "current_normalized_text",
        "prior_same_form_normalized_text",
        "request_bytes_base64",
        "response_bytes_base64",
        "model_attempt_receipt",
        "learner_state",
        "prediction_rows",
        "consumption_ledger",
    }
    observed_keys: set[str] = set()
    for value in _walk_json(result):
        assert type(value) is not bytes
        if type(value) is dict:
            observed_keys.update(value)
    assert forbidden_keys.isdisjoint(observed_keys)
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    assert "raw-response-" not in encoded
    assert "request_bytes_base64" not in encoded
    assert store.state_path.read_bytes() == state_before
    assert store.current_tip_anchor_path.read_bytes() == tip_before
    assert _tree_sha256s(store.store_directory) == tree_before


@pytest.mark.parametrize(
    "mutate",
    (
        lambda _projection, label_batch, _plan: label_batch.__setitem__(
            "candidate_sha256", "0" * 64
        ),
        lambda _projection, label_batch, _plan: label_batch.__setitem__(
            "source_feature_batch_sha256", "0" * 64
        ),
        lambda _projection, label_batch, _plan: label_batch.__setitem__(
            "matured_label_count", 1
        ),
        lambda _projection, _label_batch, plan: plan.__setitem__(
            "development_root_scope_sha256", "0" * 64
        ),
    ),
)
def test_membership_projection_rejects_crossed_source_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None],
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    scope_hash, _calls = _install_synthetic_membership_projection(
        store,
        monkeypatch,
        mutate=mutate,
    )

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="crossed its feature or label ancestry",
    ):
        store._load_owned_development_training_membership_projection(
            development_root_scope_sha256=scope_hash,
        )


def test_membership_projection_rejects_changed_authorization_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    scope_hash, _calls = _install_synthetic_membership_projection(
        store,
        monkeypatch,
    )
    stable_reader = store._read_state_and_tip_locked
    read_count = 0

    def changing_reader(anchor: Any):
        nonlocal read_count
        read_count += 1
        current, tip, state_bytes, tip_bytes = stable_reader(anchor)
        if read_count == 2:
            tip_bytes += b"changed"
        return current, tip, state_bytes, tip_bytes

    monkeypatch.setattr(store, "_read_state_and_tip_locked", changing_reader)
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="authorization ancestry changed",
    ):
        store._load_owned_development_training_membership_projection(
            development_root_scope_sha256=scope_hash,
        )
    assert read_count == 2


@pytest.mark.parametrize("extra_key", (False, True))
def test_label_batch_helper_rejects_nonexact_or_changed_projection(
    tmp_path: Path,
    extra_key: bool,
) -> None:
    store = store_scaffold._store(tmp_path)
    _scope, projection, _label_batch, _plan = _synthetic_membership_sources()
    if extra_key:
        projection["unexpected"] = None
        match = "not exact"
    else:
        projection["label_projection_sha256"] = "0" * 64
        match = "checksum changed"

    with pytest.raises(SecFilingGemmaRevealStoreError, match=match):
        store._build_owned_development_label_batch_from_projection_locked(
            label_projection=projection,
        )
