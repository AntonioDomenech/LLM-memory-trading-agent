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


def _synthetic_sources() -> dict[str, Any]:
    scope = _digest("scope")
    view_ids = [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "fold_5",
        "intermediate_frozen_through_2018",
    ]
    view_specs = [
        {"view_ordinal": index, "training_view_id": view_id}
        for index, view_id in enumerate(view_ids, start=1)
    ]
    variant_specs = [
        {"variant_ordinal": 1, "variant_id": "semantic"},
        {"variant_ordinal": 2, "variant_id": "ablation"},
    ]
    membership_plan = {
        "development_root_scope_sha256": scope,
        "start_consumed_request_count": 0,
        "training_membership_assembly_plan_sha256": _digest("membership-plan"),
        "candidate_sha256": _digest("candidate"),
        "corpus_universe_sha256": _digest("universe"),
        "calendar_sessions_sha256": _digest("calendar"),
        "development_cutoff_session": "2018-12-31",
        "event_count": 42,
        "matured_event_count": 40,
        "unmatured_event_count": 2,
        "membership_view_count": 6,
        "membership_view_specs": view_specs,
        "membership_view_specs_sha256": canonical_sha256(view_specs),
        "model_variant_count": 2,
        "model_variant_specs": variant_specs,
        "model_variant_specs_sha256": canonical_sha256(variant_specs),
    }
    feature_batch = {
        "feature_batch_sha256": _digest("feature-batch"),
        "private_marker": "FEATURE-SOURCE-MUST-NOT-ESCAPE",
    }
    label_batch = {
        "label_batch_sha256": _digest("label-batch"),
        "private_marker": "LABEL-SOURCE-MUST-NOT-ESCAPE",
    }
    projection_body = {
        "schema_version": (
            OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_PROJECTION_SCHEMA_VERSION
        ),
        "training_membership_assembly_plan": membership_plan,
        "source_feature_batch": feature_batch,
        "source_label_batch": label_batch,
    }
    membership_projection = {
        **projection_body,
        "membership_projection_sha256": canonical_sha256(projection_body),
    }
    membership_batch = {
        "development_root_scope_sha256": scope,
        "training_membership_assembly_plan_sha256": membership_plan[
            "training_membership_assembly_plan_sha256"
        ],
        "training_membership_batch_sha256": _digest("membership-batch"),
        "candidate_sha256": membership_plan["candidate_sha256"],
        "corpus_universe_sha256": membership_plan["corpus_universe_sha256"],
        "calendar_sessions_sha256": membership_plan["calendar_sessions_sha256"],
        "development_cutoff_session": "2018-12-31",
        "event_count": 42,
        "matured_label_count": 40,
        "unmatured_event_count": 2,
        "training_view_count": 6,
        "training_view_ids": view_ids,
        "membership_view_specs_sha256": membership_plan[
            "membership_view_specs_sha256"
        ],
        "model_variant_specs_sha256": membership_plan[
            "model_variant_specs_sha256"
        ],
        "training_views_sha256": _digest("training-views"),
    }
    specs = [
        {
            "fit_ordinal": fit_ordinal,
            "training_view_id": view_ids[(fit_ordinal - 1) // 2],
            "variant_id": ("semantic" if fit_ordinal % 2 else "ablation"),
            "fit_input_spec_sha256": _digest(f"fit-spec-{fit_ordinal}"),
        }
        for fit_ordinal in range(1, 11)
    ]
    deferred = [
        {
            "source_training_view_ordinal": 6,
            "training_view_id": view_ids[5],
            "reason": (
                "requires_passed_development_ranking_receipt_and_frozen_"
                "candidate_selection"
            ),
        }
    ]
    plan = {
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
        "candidate_sha256": membership_plan["candidate_sha256"],
        "corpus_universe_sha256": membership_plan["corpus_universe_sha256"],
        "calendar_sessions_sha256": membership_plan["calendar_sessions_sha256"],
        "development_cutoff_session": "2018-12-31",
        "source_training_view_count": 6,
        "source_training_view_ids": view_ids,
        "source_training_view_specs_sha256": membership_plan[
            "membership_view_specs_sha256"
        ],
        "authorized_training_view_count": 5,
        "authorized_training_view_ids": view_ids[:5],
        "deferred_training_view_count": 1,
        "deferred_training_views": deferred,
        "deferred_training_views_sha256": canonical_sha256(deferred),
        "model_variant_count": 2,
        "model_variant_ids": ["semantic", "ablation"],
        "learner_fit_input_count": 10,
        "learner_fit_input_specs": specs,
        "learner_fit_input_specs_sha256": canonical_sha256(specs),
        "learner_state_output_count": 10,
    }
    plan["development_oof_learner_fit_plan_sha256"] = canonical_sha256(plan)
    return {
        "scope": scope,
        "membership_projection": membership_projection,
        "membership_batch": membership_batch,
        "specs": specs,
        "plan": plan,
    }


def _install_nonreentrant_lock(
    store: SecFilingGemmaRevealStore,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    state = {"depth": 0, "entries": 0}

    @contextmanager
    def locked() -> Iterator[None]:
        assert state["depth"] == 0, "OOF learner fit nested its store lock"
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
    *,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, int], list[str]]:
    sources = _synthetic_sources()
    if mutate is not None:
        mutate(sources)
    calls = {
        "membership_locked": 0,
        "membership_batch": 0,
        "derive_specs": 0,
        "build_plan": 0,
        "validate_plan": 0,
    }
    order: list[str] = []

    def load_membership_locked(**kwargs: Any) -> dict[str, Any]:
        calls["membership_locked"] += 1
        order.append("membership_locked")
        assert kwargs == {
            "development_root_scope_sha256": sources["scope"]
        }
        return sources["membership_projection"]

    def build_membership_batch(**kwargs: Any) -> dict[str, Any]:
        calls["membership_batch"] += 1
        order.append("membership_batch")
        assert kwargs == {
            "training_membership_projection": sources["membership_projection"]
        }
        return sources["membership_batch"]

    def derive_specs(batch: Any) -> list[dict[str, Any]]:
        calls["derive_specs"] += 1
        order.append("derive_specs")
        assert batch is sources["membership_batch"]
        return sources["specs"]

    def build_plan(snapshot: Any, **kwargs: Any) -> dict[str, Any]:
        calls["build_plan"] += 1
        order.append("build_plan")
        assert type(snapshot) is dict
        assert kwargs["development_root_scope_sha256"] == sources["scope"]
        assert kwargs["source_training_membership_assembly_plan"] == sources[
            "membership_projection"
        ]["training_membership_assembly_plan"]
        assert kwargs["source_training_membership_projection_sha256"] == sources[
            "membership_projection"
        ]["membership_projection_sha256"]
        assert kwargs["source_training_membership_batch_sha256"] == sources[
            "membership_batch"
        ]["training_membership_batch_sha256"]
        assert kwargs["fit_input_specs"] == sources["specs"]
        assert type(kwargs["independent_current_tip_anchor"]) is dict
        return sources["plan"]

    def validate_plan(plan: Any, **kwargs: Any) -> str:
        calls["validate_plan"] += 1
        order.append("validate_plan")
        assert plan is sources["plan"]
        assert kwargs == {
            "expected_development_oof_learner_fit_plan_sha256": plan[
                "development_oof_learner_fit_plan_sha256"
            ]
        }
        return plan["development_oof_learner_fit_plan_sha256"]

    def forbidden_public_loader(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("OOF learner fit called the public membership loader")

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
        forbidden_public_loader,
    )
    monkeypatch.setattr(
        reveal_store_module,
        "derive_development_oof_learner_fit_input_specs",
        derive_specs,
    )
    monkeypatch.setattr(
        reveal_store_module,
        "build_development_oof_learner_fit_plan",
        build_plan,
    )
    monkeypatch.setattr(
        reveal_store_module,
        "validate_development_oof_learner_fit_plan",
        validate_plan,
    )
    return sources, calls, order


def test_public_oof_fit_projection_uses_one_lock_and_exact_owned_order(
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
        store._load_owned_development_oof_learner_fit_projection
    )
    assert list(signature.parameters) == ["development_root_scope_sha256"]
    assert signature.parameters[
        "development_root_scope_sha256"
    ].kind is inspect.Parameter.KEYWORD_ONLY

    result = store._load_owned_development_oof_learner_fit_projection(
        development_root_scope_sha256=sources["scope"]
    )

    assert lock_state == {"depth": 0, "entries": 1}
    assert calls == {
        "membership_locked": 1,
        "membership_batch": 1,
        "derive_specs": 1,
        "build_plan": 1,
        "validate_plan": 1,
    }
    assert order == [
        "state_tip_read",
        "membership_locked",
        "membership_batch",
        "derive_specs",
        "build_plan",
        "validate_plan",
        "state_tip_read",
    ]
    assert set(result) == {
        "schema_version",
        "development_oof_learner_fit_plan",
        "source_training_membership_batch",
        "learner_fit_projection_sha256",
    }


def test_oof_fit_projection_is_detached_exact_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    sources, _calls, _order = _install_projection(store, monkeypatch)
    source_snapshot = copy.deepcopy(sources)
    interrupted = store.store_directory / (
        ".sec_gemma_reveal_store.json." + "a" * 32 + ".tmp"
    )
    interrupted.write_bytes(b"read-only projection must not clean this file")
    state_before = store.state_path.read_bytes()
    tip_before = store.current_tip_anchor_path.read_bytes()
    tree_before = _tree_sha256s(store.store_directory)

    result = store._load_owned_development_oof_learner_fit_projection(
        development_root_scope_sha256=sources["scope"]
    )

    assert result["schema_version"] == (
        OWNED_DEVELOPMENT_OOF_LEARNER_FIT_PROJECTION_SCHEMA_VERSION
    )
    body = {
        key: result[key]
        for key in result
        if key != "learner_fit_projection_sha256"
    }
    assert result["learner_fit_projection_sha256"] == canonical_sha256(body)
    plan = result["development_oof_learner_fit_plan"]
    assert plan["authorized_training_view_ids"] == [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "fold_5",
    ]
    assert plan["learner_fit_input_count"] == 10
    assert plan["learner_state_output_count"] == 10
    assert plan["deferred_training_views"] == [
        {
            "source_training_view_ordinal": 6,
            "training_view_id": "intermediate_frozen_through_2018",
            "reason": (
                "requires_passed_development_ranking_receipt_and_frozen_"
                "candidate_selection"
            ),
        }
    ]
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    assert "FEATURE-SOURCE-MUST-NOT-ESCAPE" not in encoded
    assert "LABEL-SOURCE-MUST-NOT-ESCAPE" not in encoded
    forbidden_keys = {
        "source_feature_batch",
        "source_label_batch",
        "lookback_rows",
        "observations",
        "raw_response_bytes_by_symbol",
        "current_normalized_text",
        "prior_same_form_normalized_text",
        "request_bytes_base64",
        "response_bytes_base64",
        "model_transport_envelope",
        "artifact_path",
        "relative_path",
        "resolved_path",
        "learner_state",
        "prediction_rows",
    }
    observed_keys: set[str] = set()
    for value in _walk_json(result):
        assert type(value) is not bytes
        if type(value) is dict:
            observed_keys.update(value)
    assert forbidden_keys.isdisjoint(observed_keys)
    assert store.state_path.read_bytes() == state_before
    assert store.current_tip_anchor_path.read_bytes() == tip_before
    assert _tree_sha256s(store.store_directory) == tree_before
    assert interrupted.read_bytes() == b"read-only projection must not clean this file"
    result["development_oof_learner_fit_plan"]["candidate_sha256"] = "0" * 64
    result["source_training_membership_batch"]["training_view_ids"][0] = "changed"
    assert sources == source_snapshot


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda s: s["plan"].__setitem__("development_root_scope_sha256", "0" * 64), id="root"),
        pytest.param(lambda s: s["plan"].__setitem__("start_consumed_request_count", 1), id="consumed-count"),
        pytest.param(lambda s: s["plan"].__setitem__("source_training_membership_assembly_plan", {}), id="source-plan"),
        pytest.param(lambda s: s["plan"].__setitem__("source_training_membership_assembly_plan_sha256", "0" * 64), id="source-plan-hash"),
        pytest.param(lambda s: s["plan"].__setitem__("source_training_membership_projection_sha256", "0" * 64), id="source-projection-hash"),
        pytest.param(lambda s: s["plan"].__setitem__("source_training_membership_batch_sha256", "0" * 64), id="source-batch-hash"),
        pytest.param(lambda s: s["plan"].__setitem__("candidate_sha256", "0" * 64), id="candidate"),
        pytest.param(lambda s: s["plan"].__setitem__("corpus_universe_sha256", "0" * 64), id="universe"),
        pytest.param(lambda s: s["plan"].__setitem__("calendar_sessions_sha256", "0" * 64), id="calendar"),
        pytest.param(lambda s: s["plan"].__setitem__("development_cutoff_session", "2018-12-28"), id="cutoff"),
        pytest.param(lambda s: s["membership_batch"].__setitem__("event_count", 41), id="event-count"),
        pytest.param(lambda s: s["membership_batch"].__setitem__("matured_label_count", 39), id="matured-count"),
        pytest.param(lambda s: s["membership_batch"].__setitem__("unmatured_event_count", 3), id="unmatured-count"),
        pytest.param(lambda s: s["plan"].__setitem__("source_training_view_count", 5), id="source-view-count"),
        pytest.param(lambda s: s["plan"]["source_training_view_ids"].reverse(), id="source-view-ids"),
        pytest.param(lambda s: s["plan"].__setitem__("source_training_view_specs_sha256", "0" * 64), id="view-spec-hash"),
        pytest.param(lambda s: s["plan"].__setitem__("authorized_training_view_count", 6), id="authorized-count"),
        pytest.param(lambda s: s["plan"]["authorized_training_view_ids"].reverse(), id="authorized-ids"),
        pytest.param(lambda s: s["plan"].__setitem__("deferred_training_view_count", 0), id="deferred-count"),
        pytest.param(lambda s: s["plan"]["deferred_training_views"][0].__setitem__("source_training_view_ordinal", 5), id="deferred-ordinal"),
        pytest.param(lambda s: s["plan"]["deferred_training_views"][0].__setitem__("training_view_id", "fold_5"), id="deferred-id"),
        pytest.param(lambda s: s["plan"]["deferred_training_views"][0].__setitem__("reason", "fit-now"), id="deferred-reason"),
        pytest.param(lambda s: s["plan"].__setitem__("deferred_training_views_sha256", "0" * 64), id="deferred-hash"),
        pytest.param(lambda s: s["plan"].__setitem__("model_variant_count", 3), id="variant-count"),
        pytest.param(lambda s: s["plan"]["model_variant_ids"].reverse(), id="variant-ids"),
        pytest.param(lambda s: s["membership_batch"].__setitem__("model_variant_specs_sha256", "0" * 64), id="variant-spec-hash"),
        pytest.param(lambda s: s["plan"].__setitem__("learner_fit_input_count", 9), id="fit-count"),
        pytest.param(lambda s: s["plan"].__setitem__("learner_state_output_count", 9), id="state-count"),
        pytest.param(lambda s: s["plan"]["learner_fit_input_specs"].pop(), id="input-specs"),
        pytest.param(lambda s: s["plan"].__setitem__("learner_fit_input_specs_sha256", "0" * 64), id="input-specs-hash"),
    ],
)
def test_oof_fit_projection_rejects_crossed_membership_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    sources, _calls, _order = _install_projection(
        store,
        monkeypatch,
        mutate=mutate,
    )

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="crossed its membership ancestry",
    ):
        store._load_owned_development_oof_learner_fit_projection(
            development_root_scope_sha256=sources["scope"]
        )


@pytest.mark.parametrize(
    "changed_part",
    ("state_object", "tip_object", "state_bytes", "tip_bytes"),
)
def test_oof_fit_projection_rejects_changed_authorization_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_part: str,
) -> None:
    store = store_scaffold._store(tmp_path)
    store.initialize()
    sources, _calls, order = _install_projection(store, monkeypatch)
    stable_reader = store._read_state_and_tip_locked
    read_count = 0

    def changing_reader(anchor: Any):
        nonlocal read_count
        read_count += 1
        current, tip, state_bytes, tip_bytes = stable_reader(anchor)
        order.append("state_tip_read")
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
        store._load_owned_development_oof_learner_fit_projection(
            development_root_scope_sha256=sources["scope"]
        )
    assert read_count == 2
    assert order[0:2] == ["state_tip_read", "membership_locked"]
    assert order[-1] == "state_tip_read"


@pytest.mark.parametrize("failure", ("extra", "checksum"))
def test_membership_batch_helper_rejects_nonexact_projection(
    tmp_path: Path,
    failure: str,
) -> None:
    store = store_scaffold._store(tmp_path)
    projection = _synthetic_sources()["membership_projection"]
    if failure == "extra":
        projection["unexpected"] = None
        match = "not exact"
    else:
        projection["membership_projection_sha256"] = "0" * 64
        match = "checksum changed"
    with pytest.raises(SecFilingGemmaRevealStoreError, match=match):
        store._build_owned_development_training_membership_batch_from_projection_locked(
            training_membership_projection=projection
        )


def test_membership_batch_helper_performs_full_exact_replay_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = store_scaffold._store(tmp_path)
    sources = _synthetic_sources()
    projection = sources["membership_projection"]
    batch = sources["membership_batch"]
    calls = {"plan": 0, "build": 0, "batch": 0}

    def validate_plan(plan: Any, **kwargs: Any) -> str:
        calls["plan"] += 1
        assert plan == projection["training_membership_assembly_plan"]
        assert kwargs["expected_training_membership_assembly_plan_sha256"] == plan[
            "training_membership_assembly_plan_sha256"
        ]
        return plan["training_membership_assembly_plan_sha256"]

    def build_batch(**kwargs: Any) -> dict[str, Any]:
        calls["build"] += 1
        assert kwargs == {
            "training_membership_assembly_plan": projection[
                "training_membership_assembly_plan"
            ],
            "source_feature_batch": projection["source_feature_batch"],
            "source_label_batch": projection["source_label_batch"],
        }
        return batch

    def validate_batch(observed: Any, **kwargs: Any) -> str:
        calls["batch"] += 1
        assert observed is batch
        assert kwargs["expected_training_membership_batch_sha256"] == batch[
            "training_membership_batch_sha256"
        ]
        assert kwargs["expected_source_feature_batch_sha256"] == projection[
            "source_feature_batch"
        ]["feature_batch_sha256"]
        assert kwargs["expected_source_label_batch_sha256"] == projection[
            "source_label_batch"
        ]["label_batch_sha256"]
        return batch["training_membership_batch_sha256"]

    monkeypatch.setattr(
        reveal_store_module,
        "validate_development_training_membership_assembly_plan",
        validate_plan,
    )
    monkeypatch.setattr(
        reveal_store_module,
        "build_owned_development_training_membership_batch",
        build_batch,
    )
    monkeypatch.setattr(
        reveal_store_module,
        "validate_owned_development_training_membership_batch",
        validate_batch,
    )
    observed = (
        store._build_owned_development_training_membership_batch_from_projection_locked(
            training_membership_projection=projection
        )
    )
    assert observed is batch
    assert calls == {"plan": 1, "build": 1, "batch": 1}


def test_oof_fit_store_core_has_no_fit_prediction_recursive_lock_or_writes() -> None:
    source = inspect.getsource(
        SecFilingGemmaRevealStore._load_owned_development_oof_learner_fit_projection_locked
    )
    helper_source = inspect.getsource(
        SecFilingGemmaRevealStore._build_owned_development_training_membership_batch_from_projection_locked
    )
    feature_projection_source = inspect.getsource(
        SecFilingGemmaRevealStore._load_owned_development_feature_inputs_locked
    )
    for forbidden in (
        "self._load_owned_development_training_membership_projection(",
        "with self._locked()",
        "self._record_owned_development_",
        "self._claim_owned_development_",
        "self._atomic_write",
        ".fit(",
        ".predict(",
        ".predict_components(",
    ):
        assert forbidden not in source
        assert forbidden not in helper_source
    assert "_cleanup_interrupted_temporaries" not in feature_projection_source
