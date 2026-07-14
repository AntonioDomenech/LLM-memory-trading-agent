from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

import agent_benchmark.sec_filing_gemma_reveal_store as reveal_store_module
from agent_benchmark.sec_filing_gemma_contract import canonical_sha256
from agent_benchmark.sec_filing_gemma_learner_fit import (
    OWNED_DEVELOPMENT_OOF_LEARNER_FIT_PROJECTION_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_learner_prediction import (
    OWNED_DEVELOPMENT_OOF_PREDICTION_PROJECTION_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_reveal_store import (
    SecFilingGemmaRevealStore,
    SecFilingGemmaRevealStoreError,
)
from agent_benchmark.sec_filing_gemma_training_membership import (
    OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_PROJECTION_SCHEMA_VERSION,
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


def _synthetic_sources() -> dict[str, Any]:
    scope = _digest("prediction scope")
    contract = _digest("contract")
    candidate = _digest("candidate")
    universe = _digest("universe")
    calendar = _digest("calendar")
    feature_plan_hash = _digest("feature plan")
    view_ids = [f"fold_{index}" for index in range(1, 6)]
    source_feature_batch = {
        "development_root_scope_sha256": scope,
        "feature_assembly_plan_sha256": feature_plan_hash,
        "feature_batch_sha256": _digest("source feature batch"),
        "candidate_sha256": candidate,
        "corpus_universe_sha256": universe,
        "event_count": 11,
        "private_marker": "SOURCE-FEATURE-BATCH-MUST-NOT-ESCAPE",
    }
    membership_plan = {
        "development_root_scope_sha256": scope,
        "start_consumed_request_count": 0,
        "training_membership_assembly_plan_sha256": _digest(
            "membership plan"
        ),
        "source_feature_assembly_plan_sha256": feature_plan_hash,
        "candidate_sha256": candidate,
        "corpus_universe_sha256": universe,
        "calendar_sessions_sha256": calendar,
        "development_cutoff_session": "2018-12-31",
    }
    membership_projection_body = {
        "schema_version": (
            OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_PROJECTION_SCHEMA_VERSION
        ),
        "training_membership_assembly_plan": membership_plan,
        "source_feature_batch": source_feature_batch,
        "source_label_batch": {
            "label_batch_sha256": _digest("source label batch"),
            "private_marker": "SOURCE-LABEL-BATCH-MUST-NOT-ESCAPE",
        },
    }
    membership_projection = {
        **membership_projection_body,
        "membership_projection_sha256": canonical_sha256(
            membership_projection_body
        ),
    }
    membership_batch = {
        "development_root_scope_sha256": scope,
        "training_membership_assembly_plan_sha256": membership_plan[
            "training_membership_assembly_plan_sha256"
        ],
        "source_feature_assembly_plan_sha256": feature_plan_hash,
        "training_membership_batch_sha256": _digest("membership batch"),
        "candidate_sha256": candidate,
        "corpus_universe_sha256": universe,
        "calendar_sessions_sha256": calendar,
        "development_cutoff_session": "2018-12-31",
        "private_membership": "MEMBERSHIP-MUST-NOT-ESCAPE",
    }
    fit_input_specs = [
        {
            "fit_ordinal": ordinal,
            "fit_input_spec_sha256": _digest(f"fit input {ordinal}"),
        }
        for ordinal in range(1, 11)
    ]
    learner_fit_plan = {
        "development_root_scope_sha256": scope,
        "start_consumed_request_count": 0,
        "source_training_membership_assembly_plan": membership_plan,
        "source_training_membership_assembly_plan_sha256": membership_plan[
            "training_membership_assembly_plan_sha256"
        ],
        "source_training_membership_projection_sha256": membership_projection[
            "membership_projection_sha256"
        ],
        "source_training_membership_batch_sha256": membership_batch[
            "training_membership_batch_sha256"
        ],
        "contract_sha256": contract,
        "candidate_sha256": candidate,
        "corpus_universe_sha256": universe,
        "calendar_sessions_sha256": calendar,
        "development_cutoff_session": "2018-12-31",
        "learner_config_sha256": _digest("learner config"),
        "learner_fit_input_specs": fit_input_specs,
        "learner_fit_input_specs_sha256": canonical_sha256(fit_input_specs),
    }
    learner_fit_plan["development_oof_learner_fit_plan_sha256"] = (
        canonical_sha256(learner_fit_plan)
    )
    learner_fit_batch = {
        "development_root_scope_sha256": scope,
        "development_oof_learner_fit_plan_sha256": learner_fit_plan[
            "development_oof_learner_fit_plan_sha256"
        ],
        "source_training_membership_batch_sha256": membership_batch[
            "training_membership_batch_sha256"
        ],
        "contract_sha256": contract,
        "candidate_sha256": candidate,
        "corpus_universe_sha256": universe,
        "calendar_sessions_sha256": calendar,
        "development_cutoff_session": "2018-12-31",
        "learner_fit_batch_sha256": _digest("learner fit batch"),
        "private_fit_context": "FIT-CONTEXT-MUST-NOT-ESCAPE",
    }
    fold_models = [
        {
            "fold_ordinal": index,
            "fold_id": fold_id,
            "semantic_learner_state": {
                "state_sha256": _digest(f"{fold_id} semantic state")
            },
            "ablation_learner_state": {
                "state_sha256": _digest(f"{fold_id} ablation state")
            },
            "prediction_fold_model_sha256": _digest(f"{fold_id} models"),
        }
        for index, fold_id in enumerate(view_ids, start=1)
    ]
    fold_bundle = {
        "schema_version": "synthetic-fold-model-bundle-v1",
        "artifact_stage": "development",
        "source_learner_fit_batch_sha256": learner_fit_batch[
            "learner_fit_batch_sha256"
        ],
        "contract_sha256": contract,
        "candidate_sha256": candidate,
        "corpus_universe_sha256": universe,
        "calendar_sessions_sha256": calendar,
        "development_cutoff_session": "2018-12-31",
        "fold_model_count": 5,
        "fold_ids": view_ids,
        "model_variant_count": 2,
        "model_variant_ids": ["semantic", "ablation"],
        "learner_model_type": "synthetic-two-head-v1",
        "learner_state_schema_version": 1,
        "feature_names": ["x", "y"],
        "feature_schema_sha256": canonical_sha256(["x", "y"]),
        "fold_models": fold_models,
        "fold_models_sha256": canonical_sha256(fold_models),
    }
    fold_bundle["prediction_fold_model_bundle_sha256"] = canonical_sha256(
        fold_bundle
    )
    feature_rows = [
        {
            "prediction_ordinal": index,
            "source_event_ordinal": index + 1,
            "fold_ordinal": index,
            "fold_id": fold_id,
            "decision_session": f"{2002 + 3 * index}-01-03",
            "prediction_feature_input_sha256": _digest(
                f"{fold_id} prediction feature"
            ),
        }
        for index, fold_id in enumerate(view_ids, start=1)
    ]
    prediction_feature_batch = {
        "schema_version": "synthetic-prediction-feature-batch-v1",
        "artifact_stage": "development",
        "source_feature_batch_sha256": source_feature_batch[
            "feature_batch_sha256"
        ],
        "contract_sha256": contract,
        "candidate_sha256": candidate,
        "corpus_universe_sha256": universe,
        "source_event_count": source_feature_batch["event_count"],
        "prediction_event_count": 5,
        "prediction_fold_count": 5,
        "prediction_fold_ids": view_ids,
        "feature_names": ["x", "y"],
        "feature_schema_sha256": canonical_sha256(["x", "y"]),
        "prediction_feature_rows": feature_rows,
        "prediction_feature_rows_sha256": canonical_sha256(feature_rows),
    }
    prediction_feature_batch["prediction_feature_batch_sha256"] = (
        canonical_sha256(prediction_feature_batch)
    )
    fold_specs = [
        {
            "fold_ordinal": index,
            "fold_id": fold_id,
            "prediction_fold_model_sha256": fold_models[index - 1][
                "prediction_fold_model_sha256"
            ],
        }
        for index, fold_id in enumerate(view_ids, start=1)
    ]
    input_specs = [
        {
            "prediction_ordinal": index,
            "fold_id": fold_id,
            "prediction_available": True,
            "prediction_feature_input_sha256": feature_rows[index - 1][
                "prediction_feature_input_sha256"
            ],
        }
        for index, fold_id in enumerate(view_ids, start=1)
    ]
    return {
        "scope": scope,
        "contract": contract,
        "candidate": candidate,
        "universe": universe,
        "calendar": calendar,
        "feature_plan_hash": feature_plan_hash,
        "membership_projection": membership_projection,
        "membership_batch": membership_batch,
        "fit_input_specs": fit_input_specs,
        "learner_fit_plan": learner_fit_plan,
        "learner_fit_batch": learner_fit_batch,
        "fold_bundle": fold_bundle,
        "prediction_feature_batch": prediction_feature_batch,
        "fold_specs": fold_specs,
        "input_specs": input_specs,
        "prediction_plan_overrides": {},
    }


def _install_nonreentrant_lock(
    store: SecFilingGemmaRevealStore,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    state = {"depth": 0, "entries": 0}

    @contextmanager
    def locked() -> Iterator[None]:
        assert state["depth"] == 0, "OOF prediction nested its store lock"
        state["depth"] += 1
        state["entries"] += 1
        try:
            yield
        finally:
            state["depth"] -= 1

    monkeypatch.setattr(store, "_locked", locked)
    return state


def _install_projection(
    store: SecFilingGemmaRevealStore,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, int], list[str]]:
    sources = _synthetic_sources()
    names = (
        "membership_locked",
        "membership_batch",
        "derive_fit_specs",
        "build_fit_plan",
        "validate_fit_plan",
        "build_fit_batch",
        "validate_fit_batch",
        "derive_fold_bundle",
        "derive_feature_batch",
        "derive_fold_specs",
        "derive_input_specs",
        "build_prediction_plan",
        "validate_prediction_plan",
    )
    calls = {name: 0 for name in names}
    order: list[str] = []

    def called(name: str) -> None:
        calls[name] += 1
        order.append(name)

    def load_membership_locked(**kwargs: Any) -> dict[str, Any]:
        called("membership_locked")
        assert kwargs == {"development_root_scope_sha256": sources["scope"]}
        return sources["membership_projection"]

    def build_membership_batch(**kwargs: Any) -> dict[str, Any]:
        called("membership_batch")
        assert kwargs["training_membership_projection"] == sources[
            "membership_projection"
        ]
        return sources["membership_batch"]

    def derive_fit_specs(batch: Any) -> list[dict[str, Any]]:
        called("derive_fit_specs")
        assert batch is sources["membership_batch"]
        return sources["fit_input_specs"]

    def build_fit_plan(snapshot: Any, **kwargs: Any) -> dict[str, Any]:
        called("build_fit_plan")
        assert type(snapshot) is dict
        assert kwargs["development_root_scope_sha256"] == sources["scope"]
        assert kwargs["fit_input_specs"] == sources["fit_input_specs"]
        return sources["learner_fit_plan"]

    def validate_fit_plan(plan: Any, **kwargs: Any) -> str:
        called("validate_fit_plan")
        assert plan is sources["learner_fit_plan"]
        assert kwargs[
            "expected_development_oof_learner_fit_plan_sha256"
        ] == plan["development_oof_learner_fit_plan_sha256"]
        return plan["development_oof_learner_fit_plan_sha256"]

    def build_fit_batch(**kwargs: Any) -> dict[str, Any]:
        called("build_fit_batch")
        assert kwargs["development_oof_learner_fit_plan"] is sources[
            "learner_fit_plan"
        ]
        assert kwargs["source_training_membership_batch"] is sources[
            "membership_batch"
        ]
        return sources["learner_fit_batch"]

    def validate_fit_batch(batch: Any, **kwargs: Any) -> str:
        called("validate_fit_batch")
        assert batch is sources["learner_fit_batch"]
        assert kwargs["expected_learner_fit_batch_sha256"] == batch[
            "learner_fit_batch_sha256"
        ]
        return batch["learner_fit_batch_sha256"]

    def derive_fold_bundle(batch: Any) -> dict[str, Any]:
        called("derive_fold_bundle")
        assert batch is sources["learner_fit_batch"]
        return sources["fold_bundle"]

    def derive_feature_batch(batch: Any) -> dict[str, Any]:
        called("derive_feature_batch")
        assert batch == sources["membership_projection"]["source_feature_batch"]
        return sources["prediction_feature_batch"]

    def derive_fold_specs(bundle: Any) -> list[dict[str, Any]]:
        called("derive_fold_specs")
        assert bundle is sources["fold_bundle"]
        return sources["fold_specs"]

    def derive_input_specs(batch: Any) -> list[dict[str, Any]]:
        called("derive_input_specs")
        assert batch is sources["prediction_feature_batch"]
        return sources["input_specs"]

    def build_prediction_plan(snapshot: Any, **kwargs: Any) -> dict[str, Any]:
        called("build_prediction_plan")
        assert type(snapshot) is dict
        fit_projection_body = {
            "schema_version": (
                OWNED_DEVELOPMENT_OOF_LEARNER_FIT_PROJECTION_SCHEMA_VERSION
            ),
            "development_oof_learner_fit_plan": sources["learner_fit_plan"],
            "source_training_membership_batch": sources["membership_batch"],
        }
        assert kwargs[
            "source_development_oof_learner_fit_projection_sha256"
        ] == canonical_sha256(fit_projection_body)
        assert kwargs["prediction_fold_model_specs"] == sources["fold_specs"]
        assert kwargs["prediction_input_specs"] == sources["input_specs"]
        plan = {
            "development_root_scope_sha256": sources["scope"],
            "start_consumed_request_count": 0,
            "source_development_oof_learner_fit_plan_sha256": sources[
                "learner_fit_plan"
            ]["development_oof_learner_fit_plan_sha256"],
            "source_development_oof_learner_fit_projection_sha256": kwargs[
                "source_development_oof_learner_fit_projection_sha256"
            ],
            "source_development_oof_learner_fit_batch_sha256": sources[
                "learner_fit_batch"
            ]["learner_fit_batch_sha256"],
            "source_training_membership_assembly_plan_sha256": sources[
                "membership_batch"
            ]["training_membership_assembly_plan_sha256"],
            "source_training_membership_projection_sha256": sources[
                "membership_projection"
            ]["membership_projection_sha256"],
            "source_training_membership_batch_sha256": sources[
                "membership_batch"
            ]["training_membership_batch_sha256"],
            "prediction_fold_model_bundle_sha256": sources["fold_bundle"][
                "prediction_fold_model_bundle_sha256"
            ],
            "prediction_fold_model_specs": copy.deepcopy(sources["fold_specs"]),
            "prediction_fold_model_specs_sha256": canonical_sha256(
                sources["fold_specs"]
            ),
            "source_feature_assembly_plan_sha256": sources["feature_plan_hash"],
            "source_feature_batch_sha256": sources["membership_projection"][
                "source_feature_batch"
            ]["feature_batch_sha256"],
            "prediction_feature_batch_sha256": sources[
                "prediction_feature_batch"
            ]["prediction_feature_batch_sha256"],
            "prediction_input_specs": copy.deepcopy(sources["input_specs"]),
            "prediction_input_specs_sha256": canonical_sha256(
                sources["input_specs"]
            ),
            "contract_sha256": sources["contract"],
            "candidate_sha256": sources["candidate"],
            "corpus_universe_sha256": sources["universe"],
            "calendar_sessions_sha256": sources["calendar"],
            "development_cutoff_session": "2018-12-31",
            "source_event_count": sources["prediction_feature_batch"][
                "source_event_count"
            ],
            "authorized_fold_count": 5,
            "authorized_fold_ids": [
                "fold_1",
                "fold_2",
                "fold_3",
                "fold_4",
                "fold_5",
            ],
            "prediction_fold_model_count": 5,
            "model_variant_count": 2,
            "model_variant_ids": ["semantic", "ablation"],
            "learner_state_count": 10,
            "learner_model_type": sources["fold_bundle"]["learner_model_type"],
            "learner_state_schema_version": sources["fold_bundle"][
                "learner_state_schema_version"
            ],
            "learner_config_sha256": sources["learner_fit_plan"][
                "learner_config_sha256"
            ],
            "feature_schema_sha256": sources["fold_bundle"][
                "feature_schema_sha256"
            ],
            "prediction_input_count": len(sources["input_specs"]),
            "available_prediction_input_count": len(sources["input_specs"]),
            "unavailable_prediction_input_count": 0,
        }
        plan.update(sources["prediction_plan_overrides"])
        plan["development_oof_prediction_plan_sha256"] = canonical_sha256(plan)
        sources["prediction_plan"] = plan
        return plan

    def validate_prediction_plan(plan: Any, **kwargs: Any) -> str:
        called("validate_prediction_plan")
        assert plan is sources["prediction_plan"]
        assert kwargs["expected_development_oof_prediction_plan_sha256"] == plan[
            "development_oof_prediction_plan_sha256"
        ]
        return plan["development_oof_prediction_plan_sha256"]

    monkeypatch.setattr(
        store,
        "_load_owned_development_training_membership_projection_locked",
        load_membership_locked,
    )
    monkeypatch.setattr(
        store,
        "_build_owned_development_training_membership_batch_from_projection_locked",
        build_membership_batch,
    )
    monkeypatch.setattr(
        store,
        "_load_owned_development_training_membership_projection",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Prediction projection called a public loader")
        ),
    )
    replacements = {
        "derive_development_oof_learner_fit_input_specs": derive_fit_specs,
        "build_development_oof_learner_fit_plan": build_fit_plan,
        "validate_development_oof_learner_fit_plan": validate_fit_plan,
        "build_owned_development_oof_learner_fit_batch": build_fit_batch,
        "validate_owned_development_oof_learner_fit_batch": validate_fit_batch,
        "derive_development_oof_prediction_fold_model_bundle": derive_fold_bundle,
        "derive_development_oof_prediction_feature_batch": derive_feature_batch,
        "derive_development_oof_prediction_fold_model_specs": derive_fold_specs,
        "derive_development_oof_prediction_input_specs": derive_input_specs,
        "build_development_oof_prediction_plan": build_prediction_plan,
        "validate_development_oof_prediction_plan": validate_prediction_plan,
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(reveal_store_module, name, replacement)
    return sources, calls, order


def test_prediction_projection_uses_one_lock_and_exact_owned_order(
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
        store._load_owned_development_oof_prediction_projection
    )
    assert list(signature.parameters) == ["development_root_scope_sha256"]
    assert signature.parameters[
        "development_root_scope_sha256"
    ].kind is inspect.Parameter.KEYWORD_ONLY

    result = store._load_owned_development_oof_prediction_projection(
        development_root_scope_sha256=sources["scope"]
    )

    assert lock_state == {"depth": 0, "entries": 1}
    assert calls == {name: 1 for name in calls}
    assert order == [
        "state_tip_read",
        "membership_locked",
        "membership_batch",
        "derive_fit_specs",
        "build_fit_plan",
        "validate_fit_plan",
        "build_fit_batch",
        "validate_fit_batch",
        "derive_fold_bundle",
        "derive_feature_batch",
        "derive_fold_specs",
        "derive_input_specs",
        "build_prediction_plan",
        "validate_prediction_plan",
        "state_tip_read",
    ]
    assert set(result) == {
        "schema_version",
        "development_oof_prediction_plan",
        "prediction_fold_model_bundle",
        "prediction_feature_batch",
        "prediction_projection_sha256",
    }


def test_prediction_projection_is_exact_private_detached_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    sources, _calls, _order = _install_projection(store, monkeypatch)
    interrupted = store.store_directory / (
        ".sec_gemma_reveal_store.json." + "b" * 32 + ".tmp"
    )
    interrupted.write_bytes(b"projection must not clean this file")
    state_before = store.state_path.read_bytes()
    tip_before = store.current_tip_anchor_path.read_bytes()
    tree_before = _tree_sha256s(store.store_directory)

    result = store._load_owned_development_oof_prediction_projection(
        development_root_scope_sha256=sources["scope"]
    )
    source_snapshot = copy.deepcopy(sources)

    assert result["schema_version"] == (
        OWNED_DEVELOPMENT_OOF_PREDICTION_PROJECTION_SCHEMA_VERSION
    )
    body = {
        key: result[key]
        for key in result
        if key != "prediction_projection_sha256"
    }
    assert result["prediction_projection_sha256"] == canonical_sha256(body)
    assert result["prediction_fold_model_bundle"]["fold_ids"] == [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "fold_5",
    ]
    assert [
        row["fold_id"]
        for row in result["prediction_feature_batch"]["prediction_feature_rows"]
    ] == ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    for forbidden in (
        "SOURCE-FEATURE-BATCH-MUST-NOT-ESCAPE",
        "SOURCE-LABEL-BATCH-MUST-NOT-ESCAPE",
        "MEMBERSHIP-MUST-NOT-ESCAPE",
        "FIT-CONTEXT-MUST-NOT-ESCAPE",
        "intermediate_frozen_through_2018",
        "2019-01-01",
    ):
        assert forbidden not in encoded
    forbidden_keys = {
        "source_training_membership_batch",
        "training_set_membership",
        "semantic_training_features_hex",
        "ablation_training_features_hex",
        "training_binary_targets",
        "training_edge_targets_hex",
        "source_feature_batch",
        "source_label_batch",
        "learner_fit_views",
        "learner_fit_records",
    }
    observed_keys = {
        key
        for value in _walk_json(result)
        if type(value) is dict
        for key in value
    }
    assert forbidden_keys.isdisjoint(observed_keys)
    assert store.state_path.read_bytes() == state_before
    assert store.current_tip_anchor_path.read_bytes() == tip_before
    assert _tree_sha256s(store.store_directory) == tree_before
    assert interrupted.read_bytes() == b"projection must not clean this file"
    result["prediction_fold_model_bundle"]["fold_ids"][0] = "changed"
    result["prediction_feature_batch"]["prediction_feature_rows"][0][
        "fold_id"
    ] = "changed"
    assert sources == source_snapshot


@pytest.mark.parametrize(
    "tamper",
    (
        "root",
        "fit-plan",
        "fit-projection",
        "fit-batch",
        "fold-bundle",
        "source-feature",
        "prediction-feature",
        "fold-specs",
        "input-specs",
        "candidate",
        "calendar",
        "feature-plan",
    ),
)
def test_prediction_projection_rejects_crossed_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    sources, _calls, _order = _install_projection(store, monkeypatch)
    changed = "0" * 64
    overrides = sources["prediction_plan_overrides"]
    if tamper == "root":
        overrides["development_root_scope_sha256"] = changed
    elif tamper == "fit-plan":
        overrides["source_development_oof_learner_fit_plan_sha256"] = changed
    elif tamper == "fit-projection":
        overrides["source_development_oof_learner_fit_projection_sha256"] = changed
    elif tamper == "fit-batch":
        overrides["source_development_oof_learner_fit_batch_sha256"] = changed
    elif tamper == "fold-bundle":
        overrides["prediction_fold_model_bundle_sha256"] = changed
    elif tamper == "source-feature":
        overrides["source_feature_batch_sha256"] = changed
    elif tamper == "prediction-feature":
        overrides["prediction_feature_batch_sha256"] = changed
    elif tamper == "fold-specs":
        overrides["prediction_fold_model_specs_sha256"] = changed
    elif tamper == "input-specs":
        overrides["prediction_input_specs_sha256"] = changed
    elif tamper == "candidate":
        overrides["candidate_sha256"] = changed
    elif tamper == "calendar":
        overrides["calendar_sessions_sha256"] = changed
    else:
        overrides["source_feature_assembly_plan_sha256"] = changed

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="crossed its owned ancestry",
    ):
        store._load_owned_development_oof_prediction_projection(
            development_root_scope_sha256=sources["scope"]
        )


@pytest.mark.parametrize(
    "changed_part",
    ("state_object", "tip_object", "state_bytes", "tip_bytes"),
)
def test_prediction_projection_rejects_changed_four_part_closure(
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
        store._load_owned_development_oof_prediction_projection(
            development_root_scope_sha256=sources["scope"]
        )
    assert read_count == 2


def test_prediction_projection_rejects_nonexact_membership_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DictSubclass(dict):
        pass

    store = store_scaffold._store(tmp_path)
    store.initialize()
    scope = _digest("strict scope")
    monkeypatch.setattr(
        store,
        "_load_owned_development_training_membership_projection_locked",
        lambda **_kwargs: DictSubclass({"schema_version": "untrusted"}),
    )
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="exact built-in JSON values",
    ):
        store._load_owned_development_oof_prediction_projection(
            development_root_scope_sha256=scope
        )


def test_prediction_projection_rejects_nested_fit_plan_or_deferred_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    sources, _calls, _order = _install_projection(store, monkeypatch)
    sources["prediction_plan_overrides"][
        "source_development_oof_learner_fit_plan"
    ] = {
        "deferred_training_views": [
            {"training_view_id": "intermediate_frozen_through_2018"}
        ]
    }
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="exposed a private source",
    ):
        store._load_owned_development_oof_prediction_projection(
            development_root_scope_sha256=sources["scope"]
        )


def test_prediction_store_core_has_no_predict_effect_or_recursive_lock() -> None:
    source = inspect.getsource(
        SecFilingGemmaRevealStore._load_owned_development_oof_prediction_projection_locked
    )
    for forbidden in (
        "with self._locked()",
        "self._load_owned_development_training_membership_projection(",
        "self._load_owned_development_oof_learner_fit_projection(",
        ".predict(",
        ".predict_components(",
        "self._record_owned_",
        "self._claim_owned_",
        "self._atomic_write",
        "_cleanup_interrupted_temporaries",
        "urlopen(",
        "requests.",
        "subprocess.",
    ):
        assert forbidden not in source
