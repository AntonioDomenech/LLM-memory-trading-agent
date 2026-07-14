from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Iterator

import pytest

import agent_benchmark.sec_filing_gemma_reveal_store as reveal_store_module
from agent_benchmark.sec_filing_gemma_contract import canonical_sha256
from agent_benchmark.sec_filing_gemma_learner_prediction import (
    OWNED_DEVELOPMENT_OOF_PREDICTION_PROJECTION_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_policy_replay import (
    OWNED_DEVELOPMENT_POLICY_REPLAY_PROJECTION_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_reveal_store import (
    SecFilingGemmaRevealStore,
    SecFilingGemmaRevealStoreError,
)
from tests import test_sec_filing_gemma_reveal_store as store_scaffold


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _walk_json(value: object):
    yield value
    if type(value) is dict:
        for child in value.values():
            yield from _walk_json(child)
    elif type(value) is list:
        for child in value:
            yield from _walk_json(child)


def _tree_sha256s(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _install_nonreentrant_lock(
    store: SecFilingGemmaRevealStore,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    state = {"depth": 0, "entries": 0}

    @contextmanager
    def locked(_store: SecFilingGemmaRevealStore) -> Iterator[None]:
        assert state["depth"] == 0, "Policy projection nested its store lock"
        state["depth"] += 1
        state["entries"] += 1
        try:
            yield
        finally:
            state["depth"] -= 1

    monkeypatch.setattr(SecFilingGemmaRevealStore, "_locked", locked)
    return state


def _synthetic_sources() -> dict[str, Any]:
    scope = _digest("policy scope")
    contract = _digest("policy contract")
    candidate = _digest("policy candidate")
    universe = _digest("policy universe")
    calendar = _digest("policy calendar")
    prediction_plan = {
        "development_root_scope_sha256": scope,
        "contract_sha256": contract,
        "candidate_sha256": candidate,
        "corpus_universe_sha256": universe,
        "calendar_sessions_sha256": calendar,
        "development_cutoff_session": "2018-12-31",
        "development_oof_prediction_plan_sha256": _digest(
            "source prediction plan"
        ),
    }
    fold_model_bundle = {
        "prediction_fold_model_bundle_sha256": _digest("fold model bundle"),
        "private_fold_state": "FOLD-STATE-MUST-NOT-ESCAPE",
    }
    prediction_feature_batch = {
        "prediction_feature_batch_sha256": _digest(
            "prediction feature batch"
        ),
        "private_feature_vector": "FEATURE-VECTOR-MUST-NOT-ESCAPE",
    }
    projection_body = {
        "schema_version": (
            OWNED_DEVELOPMENT_OOF_PREDICTION_PROJECTION_SCHEMA_VERSION
        ),
        "development_oof_prediction_plan": prediction_plan,
        "prediction_fold_model_bundle": fold_model_bundle,
        "prediction_feature_batch": prediction_feature_batch,
    }
    prediction_projection = {
        **projection_body,
        "prediction_projection_sha256": canonical_sha256(projection_body),
    }
    raw_rows = [
        {
            "prediction_ordinal": 1,
            "raw_prediction_row_sha256": _digest("raw row 1"),
            "event_binding_sha256": _digest("event 1"),
            "prediction_fold_context_sha256": _digest("fold context 1"),
            "prediction_status": "available_pre_label",
            "unavailable_reason": None,
            "semantic_cash_probability_hex": (0.75).hex(),
            "semantic_expected_edge_hex": (0.01).hex(),
            "ablation_cash_probability_hex": (0.25).hex(),
            "ablation_expected_edge_hex": (-0.01).hex(),
        }
    ]
    prediction_batch = {
        "development_root_scope_sha256": scope,
        "development_oof_prediction_plan_sha256": prediction_plan[
            "development_oof_prediction_plan_sha256"
        ],
        "contract_sha256": contract,
        "candidate_sha256": candidate,
        "corpus_universe_sha256": universe,
        "calendar_sessions_sha256": calendar,
        "development_cutoff_session": "2018-12-31",
        "prediction_event_count": 1,
        "model_variant_count": 2,
        "model_variant_ids": ["semantic", "ablation"],
        "raw_prediction_rows": raw_rows,
        "raw_prediction_rows_sha256": canonical_sha256(raw_rows),
        "raw_prediction_tip_sha256": raw_rows[-1][
            "raw_prediction_row_sha256"
        ],
        "labels_included": False,
        "outcomes_included": False,
        "post_2018_data_included": False,
    }
    prediction_batch["prediction_batch_sha256"] = canonical_sha256(
        prediction_batch
    )
    policy_input_body = {
        "schema_version": (
            "aapl-sec-gemma-development-policy-replay-input-spec-v1"
        ),
        "input_ordinal": 1,
        "source_raw_prediction_row_sha256": raw_rows[0][
            "raw_prediction_row_sha256"
        ],
        "source_feature_row_sha256": _digest("feature row 1"),
        "event_binding_sha256": raw_rows[0]["event_binding_sha256"],
        "decision_session": "2010-01-04",
        "accession_number": "0000320193-10-000001",
        "fold_id": "fold_3",
        "prediction_fold_context_sha256": raw_rows[0][
            "prediction_fold_context_sha256"
        ],
        "semantic_learner_state_sha256": _digest("semantic state"),
        "ablation_learner_state_sha256": _digest("ablation state"),
        "prediction_status": "available_pre_label",
        "unavailable_reason": None,
        "numerical_components_sha256": _digest("components 1"),
    }
    policy_input_specs = [
        {
            **policy_input_body,
            "policy_replay_input_spec_sha256": canonical_sha256(
                policy_input_body
            ),
        }
    ]
    return {
        "scope": scope,
        "prediction_plan": prediction_plan,
        "prediction_projection": prediction_projection,
        "prediction_batch": prediction_batch,
        "policy_input_specs": policy_input_specs,
        "policy_plan_overrides": {},
    }


def _install_projection(
    store: SecFilingGemmaRevealStore,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, int], list[str]]:
    sources = _synthetic_sources()
    names = (
        "prediction_projection_locked",
        "validate_prediction_plan",
        "build_prediction_batch",
        "validate_prediction_batch",
        "derive_policy_specs",
        "build_policy_plan",
        "validate_policy_plan",
    )
    calls = {name: 0 for name in names}
    order: list[str] = []

    def called(name: str) -> None:
        calls[name] += 1
        order.append(name)

    def load_prediction_locked(**kwargs: Any) -> dict[str, Any]:
        called("prediction_projection_locked")
        assert kwargs == {"development_root_scope_sha256": sources["scope"]}
        return sources["prediction_projection"]

    def validate_prediction_plan(plan: Any, **kwargs: Any) -> str:
        called("validate_prediction_plan")
        assert plan == sources["prediction_plan"]
        assert kwargs["expected_development_oof_prediction_plan_sha256"] == (
            plan["development_oof_prediction_plan_sha256"]
        )
        return plan["development_oof_prediction_plan_sha256"]

    def build_prediction_batch(**kwargs: Any) -> dict[str, Any]:
        called("build_prediction_batch")
        assert kwargs["development_oof_prediction_plan"] == sources[
            "prediction_plan"
        ]
        assert kwargs["prediction_fold_model_bundle"] == sources[
            "prediction_projection"
        ]["prediction_fold_model_bundle"]
        assert kwargs["prediction_feature_batch"] == sources[
            "prediction_projection"
        ]["prediction_feature_batch"]
        return sources["prediction_batch"]

    def validate_prediction_batch(batch: Any, **kwargs: Any) -> str:
        called("validate_prediction_batch")
        assert batch is sources["prediction_batch"]
        assert kwargs["expected_prediction_batch_sha256"] == batch[
            "prediction_batch_sha256"
        ]
        return batch["prediction_batch_sha256"]

    def derive_policy_specs(batch: Any, **kwargs: Any) -> list[dict[str, Any]]:
        called("derive_policy_specs")
        assert batch is sources["prediction_batch"]
        assert kwargs["expected_source_prediction_batch_sha256"] == batch[
            "prediction_batch_sha256"
        ]
        return sources["policy_input_specs"]

    def build_policy_plan(snapshot: Any, **kwargs: Any) -> dict[str, Any]:
        called("build_policy_plan")
        prediction_batch = sources["prediction_batch"]
        prediction_plan = sources["prediction_plan"]
        tip = kwargs["independent_current_tip_anchor"]
        plan = {
            "development_root_scope_sha256": sources["scope"],
            "start_store_state_bytes_sha256": kwargs[
                "authenticated_store_state_bytes_sha256"
            ],
            "start_store_state_sha256": snapshot["state_sha256"],
            "start_current_tip_anchor_bytes_sha256": kwargs[
                "independent_current_tip_anchor_bytes_sha256"
            ],
            "start_current_tip_anchor_sha256": tip["tip_anchor_sha256"],
            "start_current_tip_revision": tip["revision"],
            "start_consumed_request_count": tip["consumed_request_count"],
            "source_development_oof_prediction_plan_sha256": prediction_plan[
                "development_oof_prediction_plan_sha256"
            ],
            "source_development_oof_prediction_projection_sha256": kwargs[
                "source_development_oof_prediction_projection_sha256"
            ],
            "source_development_oof_prediction_batch_sha256": prediction_batch[
                "prediction_batch_sha256"
            ],
            "source_raw_prediction_rows_sha256": prediction_batch[
                "raw_prediction_rows_sha256"
            ],
            "source_raw_prediction_tip_sha256": prediction_batch[
                "raw_prediction_tip_sha256"
            ],
            "source_raw_prediction_row_count": prediction_batch[
                "prediction_event_count"
            ],
            "policy_replay_input_count": len(sources["policy_input_specs"]),
            "policy_replay_input_specs": copy.deepcopy(
                sources["policy_input_specs"]
            ),
            "policy_replay_input_specs_sha256": canonical_sha256(
                sources["policy_input_specs"]
            ),
            "contract_sha256": prediction_plan["contract_sha256"],
            "candidate_sha256": prediction_plan["candidate_sha256"],
            "corpus_universe_sha256": prediction_plan[
                "corpus_universe_sha256"
            ],
            "calendar_sessions_sha256": prediction_plan[
                "calendar_sessions_sha256"
            ],
            "development_cutoff_session": prediction_plan[
                "development_cutoff_session"
            ],
            "model_variant_count": 2,
            "model_variant_ids": ["semantic", "ablation"],
            "threshold_evaluation_permitted": True,
            "policy_state_transition_permitted": True,
            "label_access_permitted": False,
            "outcome_access_permitted": False,
            "scoring_permitted": False,
            "production_permitted": False,
        }
        plan.update(sources["policy_plan_overrides"])
        plan["development_policy_replay_plan_sha256"] = canonical_sha256(plan)
        sources["policy_plan"] = plan
        return plan

    def validate_policy_plan(plan: Any, **kwargs: Any) -> str:
        called("validate_policy_plan")
        assert plan is sources["policy_plan"]
        assert kwargs["expected_development_policy_replay_plan_sha256"] == (
            plan["development_policy_replay_plan_sha256"]
        )
        return plan["development_policy_replay_plan_sha256"]

    monkeypatch.setattr(
        store,
        "_load_owned_development_oof_prediction_projection_locked",
        load_prediction_locked,
    )
    monkeypatch.setattr(
        store,
        "_load_owned_development_oof_prediction_projection",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Policy replay called the public prediction loader")
        ),
    )
    replacements = {
        "validate_development_oof_prediction_plan": validate_prediction_plan,
        "build_owned_development_oof_prediction_batch": build_prediction_batch,
        "validate_owned_development_oof_prediction_batch": (
            validate_prediction_batch
        ),
        "derive_development_policy_replay_input_specs": derive_policy_specs,
        "build_development_policy_replay_plan": build_policy_plan,
        "validate_development_policy_replay_plan": validate_policy_plan,
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(reveal_store_module, name, replacement)
    return sources, calls, order


def test_policy_replay_projection_uses_one_lock_and_one_raw_prediction_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    sources, calls, order = _install_projection(store, monkeypatch)
    lock_state = _install_nonreentrant_lock(store, monkeypatch)
    stable_reader = store._read_state_and_tip_locked

    def observed_reader(anchor: Any):
        order.append("state_tip_read")
        return stable_reader(anchor)

    monkeypatch.setattr(store, "_read_state_and_tip_locked", observed_reader)
    signature = inspect.signature(
        store._load_owned_development_policy_replay_projection
    )
    assert list(signature.parameters) == ["development_root_scope_sha256"]
    assert signature.parameters[
        "development_root_scope_sha256"
    ].kind is inspect.Parameter.KEYWORD_ONLY

    result = store._load_owned_development_policy_replay_projection(
        development_root_scope_sha256=sources["scope"]
    )

    assert lock_state == {"depth": 0, "entries": 1}
    assert calls == {name: 1 for name in calls}
    assert order == [
        "state_tip_read",
        "prediction_projection_locked",
        "validate_prediction_plan",
        "build_prediction_batch",
        "validate_prediction_batch",
        "derive_policy_specs",
        "build_policy_plan",
        "validate_policy_plan",
        "state_tip_read",
    ]
    assert set(result) == {
        "schema_version",
        "development_policy_replay_plan",
        "source_development_oof_prediction_batch",
        "policy_replay_projection_sha256",
    }


def test_policy_replay_projection_is_exact_private_detached_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    sources, _calls, _order = _install_projection(store, monkeypatch)
    interrupted = store.store_directory / (
        ".sec_gemma_reveal_store.json." + "c" * 32 + ".tmp"
    )
    interrupted.write_bytes(b"policy projection must not clean this file")
    tree_before = _tree_sha256s(store.store_directory)

    result = store._load_owned_development_policy_replay_projection(
        development_root_scope_sha256=sources["scope"]
    )
    source_snapshot = copy.deepcopy(sources)

    assert result["schema_version"] == (
        OWNED_DEVELOPMENT_POLICY_REPLAY_PROJECTION_SCHEMA_VERSION
    )
    body = {
        key: result[key]
        for key in result
        if key != "policy_replay_projection_sha256"
    }
    assert result["policy_replay_projection_sha256"] == canonical_sha256(body)
    assert result["source_development_oof_prediction_batch"][
        "raw_prediction_rows"
    ][0]["semantic_cash_probability_hex"] == (0.75).hex()
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    for forbidden in (
        "FOLD-STATE-MUST-NOT-ESCAPE",
        "FEATURE-VECTOR-MUST-NOT-ESCAPE",
        "intermediate_frozen_through_2018",
        "2019-01-01",
        "SOURCE-LABEL-BATCH-MUST-NOT-ESCAPE",
        "PRICES-MUST-NOT-ESCAPE",
    ):
        assert forbidden not in encoded
    forbidden_keys = {
        "prediction_fold_model_bundle",
        "prediction_feature_batch",
        "source_feature_batch",
        "source_label_batch",
        "label_evidence_rows",
        "adjusted_open_hex",
        "adjusted_close_hex",
        "observations",
        "policy_prefix",
        "raw_to_policy_bindings",
    }
    observed_keys = {
        key
        for value in _walk_json(result)
        if type(value) is dict
        for key in value
    }
    assert forbidden_keys.isdisjoint(observed_keys)
    assert _tree_sha256s(store.store_directory) == tree_before
    assert interrupted.read_bytes() == (
        b"policy projection must not clean this file"
    )
    result["source_development_oof_prediction_batch"]["model_variant_ids"][
        0
    ] = "changed"
    result["development_policy_replay_plan"]["policy_replay_input_specs"][0][
        "fold_id"
    ] = "changed"
    assert sources == source_snapshot


@pytest.mark.parametrize(
    "field",
    (
        "development_root_scope_sha256",
        "source_development_oof_prediction_plan_sha256",
        "source_development_oof_prediction_projection_sha256",
        "source_development_oof_prediction_batch_sha256",
        "source_raw_prediction_rows_sha256",
        "source_raw_prediction_tip_sha256",
        "source_raw_prediction_row_count",
        "policy_replay_input_specs_sha256",
        "candidate_sha256",
        "calendar_sessions_sha256",
    ),
)
def test_policy_replay_projection_rejects_crossed_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    sources, _calls, _order = _install_projection(store, monkeypatch)
    sources["policy_plan_overrides"][field] = (
        2 if field == "source_raw_prediction_row_count" else "0" * 64
    )

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="crossed its owned prediction ancestry",
    ):
        store._load_owned_development_policy_replay_projection(
            development_root_scope_sha256=sources["scope"]
        )


@pytest.mark.parametrize(
    "changed_part",
    ("state_object", "tip_object", "state_bytes", "tip_bytes"),
)
def test_policy_replay_projection_rejects_changed_four_part_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_part: str,
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    sources, _calls, _order = _install_projection(store, monkeypatch)
    stable_reader = store._read_state_and_tip_locked
    read_count = 0

    def changing_reader(anchor: Any):
        nonlocal read_count
        read_count += 1
        current, tip, state_bytes, tip_bytes = stable_reader(anchor)
        if read_count == 2:
            if changed_part == "state_object":
                current = {**current, "changed": True}
            elif changed_part == "tip_object":
                tip = {**tip, "changed": True}
            elif changed_part == "state_bytes":
                state_bytes += b"changed"
            else:
                tip_bytes += b"changed"
        return current, tip, state_bytes, tip_bytes

    monkeypatch.setattr(store, "_read_state_and_tip_locked", changing_reader)
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="authorization ancestry changed",
    ):
        store._load_owned_development_policy_replay_projection(
            development_root_scope_sha256=sources["scope"]
        )
    assert read_count == 2


@pytest.mark.parametrize(
    ("private_key", "private_value"),
    (
        ("source_label_batch", {"rows": ["LABELS-MUST-NOT-ESCAPE"]}),
        ("adjusted_open_hex", (123.0).hex()),
        ("observations", [{"price": "PRICES-MUST-NOT-ESCAPE"}]),
        ("private_window", "intermediate_frozen_through_2018"),
        ("private_date", "2019-01-01"),
    ),
)
def test_policy_replay_projection_rejects_private_source_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    private_key: str,
    private_value: Any,
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    sources, _calls, _order = _install_projection(store, monkeypatch)
    sources["prediction_batch"][private_key] = private_value

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="exposed a private source|exposed a forbidden window",
    ):
        store._load_owned_development_policy_replay_projection(
            development_root_scope_sha256=sources["scope"]
        )


def test_policy_replay_store_core_has_no_forbidden_effect_or_recursive_lock() -> None:
    source = inspect.getsource(
        SecFilingGemmaRevealStore._load_owned_development_policy_replay_projection_locked
    )
    for forbidden in (
        "with self._locked()",
        "self._load_owned_development_oof_prediction_projection(",
        "self._load_owned_development_label_projection",
        "self._load_owned_development_market",
        "build_owned_development_policy_replay_batch",
        "build_prediction_ledger",
        "self._record_owned_",
        "self._claim_owned_",
        "self._atomic_write",
        "_cleanup_interrupted_temporaries",
        "urlopen(",
        "requests.",
        "subprocess.",
        "stage_runner",
    ):
        assert forbidden not in source
