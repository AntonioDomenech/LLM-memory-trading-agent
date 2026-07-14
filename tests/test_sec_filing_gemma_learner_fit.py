from __future__ import annotations

import copy
import json
import time
from typing import Any

import pytest

import agent_benchmark.sec_filing_gemma_learner_fit as fit_module
from agent_benchmark.sec_filing_gemma_contract import (
    BRIER_TARGET_COST_BPS,
    build_contract_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_filing_gemma_learner import (
    SecFilingGemmaLearnerError,
    SecFilingGemmaTwoHeadLearner,
)
from agent_benchmark.sec_filing_gemma_learner_fit import (
    DEVELOPMENT_OOF_LEARNER_FIT_RECORD_SCHEMA_VERSION,
    DEVELOPMENT_OOF_LEARNER_FIT_VIEW_SCHEMA_VERSION,
    OWNED_DEVELOPMENT_OOF_LEARNER_FIT_BATCH_SCHEMA_VERSION,
    SecFilingGemmaLearnerFitError,
    build_owned_development_oof_learner_fit_batch,
    derive_development_oof_learner_fit_input_specs,
    validate_owned_development_oof_learner_fit_batch,
)
from agent_benchmark.sec_filing_gemma_stage_authorization import (
    build_development_oof_learner_fit_plan,
)
from agent_benchmark.sec_filing_gemma_training_membership import (
    DEVELOPMENT_TRAINING_VIEW_SCHEMA_VERSION,
    OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_BATCH_SCHEMA_VERSION,
)
from tests.test_sec_filing_gemma_stage_authorization import (
    _development_training_membership_assembly_plan_fixture,
)


AUTHORIZED_VIEW_IDS = [f"fold_{ordinal}" for ordinal in range(1, 6)]
DEFERRED_VIEW_ID = "intermediate_frozen_through_2018"
FIT_RECORD_KEYS = {
    "schema_version",
    "fit_ordinal",
    "source_training_view_ordinal",
    "training_view_id",
    "head_variant",
    "source_training_view_sha256",
    "training_input_identity",
    "training_input_identity_sha256",
    "fit_metadata",
    "fit_metadata_sha256",
    "learner_state",
    "learner_state_sha256",
    "learner_parameters_sha256",
    "learner_fit_record_sha256",
}
LEARNER_CONTEXT_KEYS = {
    "fold_train_cutoff_session",
    "training_set_count",
    "training_positive_count",
    "training_set_membership_sha256",
    "semantic_training_feature_matrix_sha256",
    "ablation_training_feature_matrix_sha256",
    "training_binary_target_sha256",
    "training_edge_target_sha256",
    "training_set_max_label_maturity_session",
}


def _digest(label: str) -> str:
    return canonical_sha256({"learner-fit-test": label})


def _metadata(
    *,
    candidate_sha256: str,
    view: dict[str, Any],
    variant: str,
    membership_sha256: str,
    row_count: int,
    maximum_maturity: str,
    feature_schema_sha256: str,
) -> dict[str, Any]:
    return {
        "candidate_sha256": candidate_sha256,
        "head_variant": variant,
        "fold_id": view["training_view_id"],
        "train_label_maturity_through": view[
            "train_label_maturity_through"
        ],
        "training_set_sha256": membership_sha256,
        "training_row_count": row_count,
        "maximum_training_label_maturity_session": maximum_maturity,
        "feature_schema_sha256": feature_schema_sha256,
    }


def _training_view(
    *,
    view_spec: dict[str, Any],
    candidate_sha256: str,
    feature_schema_sha256: str,
) -> dict[str, Any]:
    ordinal = view_spec["view_ordinal"]
    row_count = 4
    members = [
        {
            "training_row_ordinal": row_ordinal,
            "source_event_ordinal": row_ordinal,
            "accession_number": (
                f"0000320193-04-{ordinal:02d}{row_ordinal:04d}"
            ),
            "form": "10-K" if row_ordinal % 2 else "10-Q",
            "decision_session": f"2003-0{row_ordinal}-03",
            "label_maturity_session": f"2003-0{row_ordinal}-28",
            "training_event_identity_sha256": _digest(
                f"event-{ordinal}-{row_ordinal}"
            ),
        }
        for row_ordinal in range(1, row_count + 1)
    ]
    semantic = [
        [
            float(ordinal / 100 + row_ordinal).hex(),
            float(1 if row_ordinal % 2 else -1).hex(),
        ]
        for row_ordinal in range(row_count)
    ]
    ablation = [
        [float(ordinal / 200 + row_ordinal).hex(), float(0).hex()]
        for row_ordinal in range(row_count)
    ]
    binary = [0, 1, 0, 1]
    edge = [float(value).hex() for value in (-0.03, 0.02, -0.01, 0.04)]
    membership_sha256 = canonical_sha256(members)
    semantic_sha256 = canonical_sha256(semantic)
    ablation_sha256 = canonical_sha256(ablation)
    binary_sha256 = canonical_sha256(binary)
    edge_sha256 = canonical_sha256(edge)
    maximum_maturity = members[-1]["label_maturity_session"]
    context = {
        "fold_train_cutoff_session": view_spec[
            "train_label_maturity_through"
        ],
        "training_set_count": row_count,
        "training_positive_count": sum(binary),
        "training_set_membership_sha256": membership_sha256,
        "semantic_training_feature_matrix_sha256": semantic_sha256,
        "ablation_training_feature_matrix_sha256": ablation_sha256,
        "training_binary_target_sha256": binary_sha256,
        "training_edge_target_sha256": edge_sha256,
        "training_set_max_label_maturity_session": maximum_maturity,
    }
    semantic_metadata = _metadata(
        candidate_sha256=candidate_sha256,
        view=view_spec,
        variant="semantic",
        membership_sha256=membership_sha256,
        row_count=row_count,
        maximum_maturity=maximum_maturity,
        feature_schema_sha256=feature_schema_sha256,
    )
    ablation_metadata = _metadata(
        candidate_sha256=candidate_sha256,
        view=view_spec,
        variant="ablation",
        membership_sha256=membership_sha256,
        row_count=row_count,
        maximum_maturity=maximum_maturity,
        feature_schema_sha256=feature_schema_sha256,
    )
    body = {
        "schema_version": DEVELOPMENT_TRAINING_VIEW_SCHEMA_VERSION,
        **copy.deepcopy(view_spec),
        "training_set_count": row_count,
        "training_positive_count": sum(binary),
        "training_set_max_label_maturity_session": maximum_maturity,
        "training_set_membership": members,
        "training_set_membership_sha256": membership_sha256,
        "semantic_training_features_hex": semantic,
        "semantic_training_feature_matrix_sha256": semantic_sha256,
        "ablation_training_features_hex": ablation,
        "ablation_training_feature_matrix_sha256": ablation_sha256,
        "training_binary_targets": binary,
        "training_binary_target_sha256": binary_sha256,
        "training_edge_targets_hex": edge,
        "training_edge_target_sha256": edge_sha256,
        "learner_input_context": context,
        "learner_input_context_sha256": canonical_sha256(context),
        "semantic_fit_metadata_template": semantic_metadata,
        "semantic_fit_metadata_template_sha256": canonical_sha256(
            semantic_metadata
        ),
        "ablation_fit_metadata_template": ablation_metadata,
        "ablation_fit_metadata_template_sha256": canonical_sha256(
            ablation_metadata
        ),
    }
    return {**body, "training_view_sha256": canonical_sha256(body)}


def _membership_batch(membership_plan: dict[str, Any]) -> dict[str, Any]:
    feature_names = ["feature_a", "feature_b"]
    feature_schema_sha256 = canonical_sha256(feature_names)
    views = [
        _training_view(
            view_spec=view,
            candidate_sha256=membership_plan["candidate_sha256"],
            feature_schema_sha256=feature_schema_sha256,
        )
        for view in membership_plan["membership_view_specs"]
    ]
    event_audits: list[dict[str, Any]] = []
    body = {
        "schema_version": (
            OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_BATCH_SCHEMA_VERSION
        ),
        "development_root_scope_sha256": membership_plan[
            "development_root_scope_sha256"
        ],
        "training_membership_assembly_plan_sha256": membership_plan[
            "training_membership_assembly_plan_sha256"
        ],
        "source_feature_assembly_plan_sha256": membership_plan[
            "source_feature_assembly_plan_sha256"
        ],
        "source_label_assembly_plan_sha256": membership_plan[
            "source_label_assembly_plan_sha256"
        ],
        "source_feature_batch_sha256": _digest("feature-batch"),
        "source_label_batch_sha256": _digest("label-batch"),
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "candidate_sha256": membership_plan["candidate_sha256"],
        "corpus_universe_sha256": membership_plan[
            "corpus_universe_sha256"
        ],
        "calendar_sessions_sha256": membership_plan[
            "calendar_sessions_sha256"
        ],
        "development_cutoff_session": membership_plan[
            "development_cutoff_session"
        ],
        "event_count": membership_plan["event_count"],
        "matured_label_count": membership_plan["matured_event_count"],
        "unmatured_event_count": membership_plan["unmatured_event_count"],
        "training_view_count": 6,
        "training_view_ids": [*AUTHORIZED_VIEW_IDS, DEFERRED_VIEW_ID],
        "membership_view_specs_sha256": membership_plan[
            "membership_view_specs_sha256"
        ],
        "model_variant_specs_sha256": membership_plan[
            "model_variant_specs_sha256"
        ],
        "feature_names": feature_names,
        "feature_schema_sha256": feature_schema_sha256,
        "target_cost_bps": BRIER_TARGET_COST_BPS,
        "binary_target_field": "cash_beats_long_10bps",
        "binary_target_encoding": "false_to_0_true_to_1",
        "edge_target_field": "cash_active_log_edge_10bps_hex",
        "binary_comparison_tolerance_hex": float(1e-12).hex(),
        "membership_maturity_rule": membership_plan[
            "membership_maturity_rule"
        ],
        "feature_eligibility_rule": membership_plan[
            "feature_eligibility_rule"
        ],
        "membership_order_rule": membership_plan["membership_order_rule"],
        "shared_variant_support_rule": membership_plan[
            "shared_variant_support_rule"
        ],
        "event_audit_rows": event_audits,
        "event_audit_rows_sha256": canonical_sha256(event_audits),
        "training_views": views,
        "training_views_sha256": canonical_sha256(views),
        "development_labels_included": True,
        "development_outcomes_included": True,
        "training_membership_included": True,
        "learner_input_matrices_included": True,
        "learner_targets_included": True,
        "compact_adjusted_open_paths_included": False,
        "full_market_rows_included": False,
        "post_cutoff_market_data_included": False,
        "outcome_based_membership_filtering_permitted": False,
        "row_rebalancing_permitted": False,
        "learner_fit_authorized": False,
        "prediction_authorized": False,
        "holdout_access_authorized": False,
        "ledger_mutation_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    return {**body, "training_membership_batch_sha256": canonical_sha256(body)}


def _rebind_view(
    view: dict[str, Any],
    *,
    candidate_sha256: str,
    feature_schema_sha256: str,
) -> None:
    members = view["training_set_membership"]
    binary = view["training_binary_targets"]
    edge = view["training_edge_targets_hex"]
    row_count = len(members)
    maximum_maturity = max(item["label_maturity_session"] for item in members)
    membership_sha256 = canonical_sha256(members)
    semantic_sha256 = canonical_sha256(
        view["semantic_training_features_hex"]
    )
    ablation_sha256 = canonical_sha256(
        view["ablation_training_features_hex"]
    )
    binary_sha256 = canonical_sha256(binary)
    edge_sha256 = canonical_sha256(edge)
    view["training_set_count"] = row_count
    view["training_positive_count"] = sum(binary)
    view["training_set_max_label_maturity_session"] = maximum_maturity
    view["training_set_membership_sha256"] = membership_sha256
    view["semantic_training_feature_matrix_sha256"] = semantic_sha256
    view["ablation_training_feature_matrix_sha256"] = ablation_sha256
    view["training_binary_target_sha256"] = binary_sha256
    view["training_edge_target_sha256"] = edge_sha256
    context = {
        "fold_train_cutoff_session": view["train_label_maturity_through"],
        "training_set_count": row_count,
        "training_positive_count": sum(binary),
        "training_set_membership_sha256": membership_sha256,
        "semantic_training_feature_matrix_sha256": semantic_sha256,
        "ablation_training_feature_matrix_sha256": ablation_sha256,
        "training_binary_target_sha256": binary_sha256,
        "training_edge_target_sha256": edge_sha256,
        "training_set_max_label_maturity_session": maximum_maturity,
    }
    view["learner_input_context"] = context
    view["learner_input_context_sha256"] = canonical_sha256(context)
    for variant in ("semantic", "ablation"):
        metadata = _metadata(
            candidate_sha256=candidate_sha256,
            view=view,
            variant=variant,
            membership_sha256=membership_sha256,
            row_count=row_count,
            maximum_maturity=maximum_maturity,
            feature_schema_sha256=feature_schema_sha256,
        )
        view[f"{variant}_fit_metadata_template"] = metadata
        view[f"{variant}_fit_metadata_template_sha256"] = canonical_sha256(
            metadata
        )
    body = {
        key: value for key, value in view.items() if key != "training_view_sha256"
    }
    view["training_view_sha256"] = canonical_sha256(body)


def _rehash_membership(batch: dict[str, Any]) -> None:
    batch["training_views_sha256"] = canonical_sha256(batch["training_views"])
    batch["event_audit_rows_sha256"] = canonical_sha256(
        batch["event_audit_rows"]
    )
    body = {
        key: value
        for key, value in batch.items()
        if key != "training_membership_batch_sha256"
    }
    batch["training_membership_batch_sha256"] = canonical_sha256(body)


def _plan_for(
    *,
    state: dict[str, Any],
    tip: dict[str, Any],
    membership_plan: dict[str, Any],
    membership_batch: dict[str, Any],
) -> dict[str, Any]:
    specs = derive_development_oof_learner_fit_input_specs(membership_batch)
    return build_development_oof_learner_fit_plan(
        state,
        development_root_scope_sha256=membership_plan[
            "development_root_scope_sha256"
        ],
        source_training_membership_assembly_plan=membership_plan,
        source_training_membership_projection_sha256=_digest(
            "membership-projection"
        ),
        source_training_membership_batch_sha256=membership_batch[
            "training_membership_batch_sha256"
        ],
        fit_input_specs=specs,
        independent_current_tip_anchor=tip,
    )


def _build(case: dict[str, Any]) -> dict[str, Any]:
    return build_owned_development_oof_learner_fit_batch(
        development_oof_learner_fit_plan=case["plan"],
        expected_development_oof_learner_fit_plan_sha256=case["plan"][
            "development_oof_learner_fit_plan_sha256"
        ],
        source_training_membership_batch=case["membership_batch"],
        expected_source_training_membership_batch_sha256=case[
            "membership_batch"
        ]["training_membership_batch_sha256"],
    )


@pytest.fixture(scope="module")
def learner_fit_case() -> dict[str, Any]:
    state, tip, _label_plan, membership_plan = (
        _development_training_membership_assembly_plan_fixture()
    )
    membership_batch = _membership_batch(membership_plan)
    plan = _plan_for(
        state=state,
        tip=tip,
        membership_plan=membership_plan,
        membership_batch=membership_batch,
    )
    return {
        "state": state,
        "tip": tip,
        "membership_plan": membership_plan,
        "membership_batch": membership_batch,
        "plan": plan,
    }


def _changed_case(
    base: dict[str, Any], membership_batch: dict[str, Any]
) -> dict[str, Any]:
    return {
        **base,
        "membership_batch": membership_batch,
        "plan": _plan_for(
            state=base["state"],
            tip=base["tip"],
            membership_plan=base["membership_plan"],
            membership_batch=membership_batch,
        ),
    }


def _state_hashes(batch: dict[str, Any]) -> list[str]:
    return [
        record["learner_state_sha256"]
        for record in batch["learner_fit_records"]
    ]


def _all_keys(value: Any) -> set[str]:
    if type(value) is dict:
        return set(value) | set().union(
            *(_all_keys(item) for item in value.values()), set()
        )
    if type(value) is list:
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def test_input_specs_are_exact_ten_view_major_semantic_then_ablation(
    learner_fit_case: dict[str, Any],
) -> None:
    specs = derive_development_oof_learner_fit_input_specs(
        learner_fit_case["membership_batch"]
    )
    assert len(specs) == 10
    assert [item["fit_ordinal"] for item in specs] == list(range(1, 11))
    assert [item["training_view_id"] for item in specs] == [
        view_id for view_id in AUTHORIZED_VIEW_IDS for _ in range(2)
    ]
    assert [item["head_variant"] for item in specs] == [
        variant
        for _ in AUTHORIZED_VIEW_IDS
        for variant in ("semantic", "ablation")
    ]
    assert all(
        item["fit_input_spec_sha256"]
        == canonical_sha256(
            {
                key: value
                for key, value in item.items()
                if key != "fit_input_spec_sha256"
            }
        )
        for item in specs
    )
    assert DEFERRED_VIEW_ID not in {
        item["training_view_id"] for item in specs
    }


def test_build_fits_ten_fresh_states_without_predicting_and_emits_compact_views(
    learner_fit_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_instances: list[SecFilingGemmaTwoHeadLearner] = []
    original_fit = SecFilingGemmaTwoHeadLearner.fit

    def tracked_fit(self: SecFilingGemmaTwoHeadLearner, *args: Any, **kwargs: Any):
        fit_instances.append(self)
        return original_fit(self, *args, **kwargs)

    def forbidden_prediction(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("The learner-fit boundary must not predict")

    monkeypatch.setattr(SecFilingGemmaTwoHeadLearner, "fit", tracked_fit)
    monkeypatch.setattr(
        SecFilingGemmaTwoHeadLearner,
        "predict_components",
        forbidden_prediction,
    )
    started = time.monotonic()
    batch = _build(learner_fit_case)
    elapsed = time.monotonic() - started

    assert elapsed < 60
    assert len(fit_instances) == 10
    assert len({id(item) for item in fit_instances}) == 10
    assert batch["schema_version"] == (
        OWNED_DEVELOPMENT_OOF_LEARNER_FIT_BATCH_SCHEMA_VERSION
    )
    assert batch["learner_fit_count"] == batch["learner_state_count"] == 10
    assert batch["authorized_training_view_ids"] == AUTHORIZED_VIEW_IDS
    assert len(batch["learner_fit_views"]) == 5
    assert len(batch["learner_fit_records"]) == 10
    assert [item["training_view_id"] for item in batch["learner_fit_records"]] == [
        view_id for view_id in AUTHORIZED_VIEW_IDS for _ in range(2)
    ]
    assert DEFERRED_VIEW_ID not in {
        item["training_view_id"] for item in batch["learner_fit_records"]
    }
    assert batch["deferred_training_view_state_included"] is False

    for ordinal, record in enumerate(batch["learner_fit_records"], start=1):
        assert set(record) == FIT_RECORD_KEYS
        assert record["schema_version"] == (
            DEVELOPMENT_OOF_LEARNER_FIT_RECORD_SCHEMA_VERSION
        )
        assert record["fit_ordinal"] == ordinal
        assert record["learner_state_sha256"] == record["learner_state"][
            "state_sha256"
        ]
        assert record["learner_parameters_sha256"] == record["learner_state"][
            "parameters_sha256"
        ]
        identity = record["training_input_identity"]
        for field in (
            "development_oof_learner_fit_plan_sha256",
            "source_training_membership_batch_sha256",
            "training_set_membership_sha256",
            "training_feature_matrix_sha256",
            "training_binary_target_sha256",
            "training_edge_target_sha256",
            "learner_input_context_sha256",
            "fit_metadata_template_sha256",
            "feature_schema_sha256",
            "learner_config_sha256",
        ):
            assert len(identity[field]) == 64
        assert record["training_input_identity_sha256"] == canonical_sha256(
            identity
        )

    for view in batch["learner_fit_views"]:
        assert view["schema_version"] == (
            DEVELOPMENT_OOF_LEARNER_FIT_VIEW_SCHEMA_VERSION
        )
        assert set(view["learner_input_context"]) == LEARNER_CONTEXT_KEYS
        assert set(view["prediction_fold_context"]) == (
            LEARNER_CONTEXT_KEYS
            | {"semantic_fold_state_sha256", "ablation_fold_state_sha256"}
        )
        assert view["prediction_fold_context"][
            "semantic_fold_state_sha256"
        ] == view["semantic_learner_state_sha256"]
        assert view["prediction_fold_context"][
            "ablation_fold_state_sha256"
        ] == view["ablation_learner_state_sha256"]
        assert view["beta_1_1_climatology_hex"] == float(0.5).hex()

    keys = _all_keys(batch)
    assert not {
        "source_training_membership_batch",
        "training_set_membership",
        "semantic_training_features_hex",
        "ablation_training_features_hex",
        "training_binary_targets",
        "training_edge_targets_hex",
        "event_audit_rows",
    } & keys
    assert batch["prediction_included"] is False
    assert batch["prediction_authorized"] is False
    assert batch["candidate_selection_authorized"] is False
    assert batch["threshold_action_authorized"] is False
    assert batch["holdout_access_authorized"] is False
    assert batch["ledger_mutation_authorized"] is False
    assert batch["stage_promotion_authorized"] is False
    assert batch["production_authorized"] is False
    assert "model_response" not in json.dumps(batch, sort_keys=True)


def test_fit_batch_and_validation_are_byte_deterministic(
    learner_fit_case: dict[str, Any],
) -> None:
    first = _build(learner_fit_case)
    second = _build(learner_fit_case)
    assert first == second
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    assert validate_owned_development_oof_learner_fit_batch(
        first,
        development_oof_learner_fit_plan=learner_fit_case["plan"],
        expected_development_oof_learner_fit_plan_sha256=learner_fit_case[
            "plan"
        ]["development_oof_learner_fit_plan_sha256"],
        source_training_membership_batch=learner_fit_case[
            "membership_batch"
        ],
        expected_source_training_membership_batch_sha256=learner_fit_case[
            "membership_batch"
        ]["training_membership_batch_sha256"],
        expected_learner_fit_batch_sha256=first["learner_fit_batch_sha256"],
    ) == first["learner_fit_batch_sha256"]


def test_matrix_and_target_changes_are_bound_while_membership_stays_fixed(
    learner_fit_case: dict[str, Any],
) -> None:
    baseline = _build(learner_fit_case)

    changed_matrix_batch = copy.deepcopy(learner_fit_case["membership_batch"])
    matrix_view = changed_matrix_batch["training_views"][0]
    original_membership_hash = matrix_view["training_set_membership_sha256"]
    original_target_hash = matrix_view["training_binary_target_sha256"]
    matrix_view["semantic_training_features_hex"][0][0] = float(99).hex()
    _rebind_view(
        matrix_view,
        candidate_sha256=changed_matrix_batch["candidate_sha256"],
        feature_schema_sha256=changed_matrix_batch["feature_schema_sha256"],
    )
    _rehash_membership(changed_matrix_batch)
    matrix_case = _changed_case(learner_fit_case, changed_matrix_batch)
    matrix_result = _build(matrix_case)
    assert matrix_view["training_set_membership_sha256"] == original_membership_hash
    assert matrix_view["training_binary_target_sha256"] == original_target_hash
    assert _state_hashes(matrix_result)[0] != _state_hashes(baseline)[0]
    assert _state_hashes(matrix_result)[1:] == _state_hashes(baseline)[1:]

    changed_target_batch = copy.deepcopy(learner_fit_case["membership_batch"])
    target_view = changed_target_batch["training_views"][0]
    original_membership_hash = target_view["training_set_membership_sha256"]
    original_semantic_matrix_hash = target_view[
        "semantic_training_feature_matrix_sha256"
    ]
    original_ablation_matrix_hash = target_view[
        "ablation_training_feature_matrix_sha256"
    ]
    target_view["training_binary_targets"] = [1, 0, 0, 1]
    _rebind_view(
        target_view,
        candidate_sha256=changed_target_batch["candidate_sha256"],
        feature_schema_sha256=changed_target_batch["feature_schema_sha256"],
    )
    _rehash_membership(changed_target_batch)
    target_case = _changed_case(learner_fit_case, changed_target_batch)
    target_result = _build(target_case)
    assert target_view["training_set_membership_sha256"] == original_membership_hash
    assert target_view["semantic_training_feature_matrix_sha256"] == (
        original_semantic_matrix_hash
    )
    assert target_view["ablation_training_feature_matrix_sha256"] == (
        original_ablation_matrix_hash
    )
    assert _state_hashes(target_result)[:2] != _state_hashes(baseline)[:2]
    assert _state_hashes(target_result)[2:] == _state_hashes(baseline)[2:]


def test_deferred_view_changes_never_create_or_change_an_oof_state(
    learner_fit_case: dict[str, Any],
) -> None:
    baseline = _build(learner_fit_case)
    changed_batch = copy.deepcopy(learner_fit_case["membership_batch"])
    deferred = changed_batch["training_views"][5]
    deferred["semantic_training_features_hex"][0][0] = float(1234).hex()
    deferred["training_binary_targets"] = [1, 0, 0, 1]
    _rebind_view(
        deferred,
        candidate_sha256=changed_batch["candidate_sha256"],
        feature_schema_sha256=changed_batch["feature_schema_sha256"],
    )
    _rehash_membership(changed_batch)
    changed_case = _changed_case(learner_fit_case, changed_batch)
    changed = _build(changed_case)

    assert changed["source_training_membership_batch_sha256"] != baseline[
        "source_training_membership_batch_sha256"
    ]
    assert changed["development_oof_learner_fit_plan_sha256"] != baseline[
        "development_oof_learner_fit_plan_sha256"
    ]
    assert _state_hashes(changed) == _state_hashes(baseline)
    assert all(
        record["training_view_id"] != DEFERRED_VIEW_ID
        for record in changed["learner_fit_records"]
    )


def test_membership_types_and_float_hex_fail_closed(
    learner_fit_case: dict[str, Any],
) -> None:
    class DictSubclass(dict):
        pass

    deepcopy_hooks: list[str] = []

    class NestedHookDict(dict):
        def __deepcopy__(self, _memo: dict[int, Any]) -> "NestedHookDict":
            deepcopy_hooks.append("executed")
            return self

    with pytest.raises(SecFilingGemmaLearnerFitError, match="non-exact JSON"):
        derive_development_oof_learner_fit_input_specs(
            DictSubclass(learner_fit_case["membership_batch"])
        )

    nested_subclass = copy.deepcopy(learner_fit_case["membership_batch"])
    nested_subclass["event_audit_rows"] = [NestedHookDict({"value": True})]
    _rehash_membership(nested_subclass)
    with pytest.raises(SecFilingGemmaLearnerFitError, match="exact built-in"):
        derive_development_oof_learner_fit_input_specs(nested_subclass)
    assert deepcopy_hooks == []

    bool_count = copy.deepcopy(learner_fit_case["membership_batch"])
    bool_count["training_view_count"] = True
    _rehash_membership(bool_count)
    with pytest.raises(SecFilingGemmaLearnerFitError, match="exact integer"):
        derive_development_oof_learner_fit_input_specs(bool_count)

    noncanonical = copy.deepcopy(learner_fit_case["membership_batch"])
    view = noncanonical["training_views"][0]
    view["training_edge_targets_hex"][0] = "0x1.0p+0"
    _rebind_view(
        view,
        candidate_sha256=noncanonical["candidate_sha256"],
        feature_schema_sha256=noncanonical["feature_schema_sha256"],
    )
    _rehash_membership(noncanonical)
    with pytest.raises(SecFilingGemmaLearnerFitError, match="canonical finite"):
        derive_development_oof_learner_fit_input_specs(noncanonical)


def test_first_fit_failure_is_rejected_without_retry_or_prediction(
    learner_fit_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_calls = 0

    def failed_fit(*_args: Any, **_kwargs: Any) -> None:
        nonlocal fit_calls
        fit_calls += 1
        raise SecFilingGemmaLearnerError("synthetic solver failure")

    def forbidden_prediction(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("prediction must remain inaccessible")

    monkeypatch.setattr(SecFilingGemmaTwoHeadLearner, "fit", failed_fit)
    monkeypatch.setattr(
        SecFilingGemmaTwoHeadLearner,
        "predict_components",
        forbidden_prediction,
    )
    with pytest.raises(SecFilingGemmaLearnerFitError, match="fit failed"):
        _build(learner_fit_case)
    assert fit_calls == 1


def test_ten_fit_wall_clock_cap_is_enforced(
    learner_fit_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = iter((100.0, 161.0))
    monkeypatch.setattr(fit_module.time, "monotonic", lambda: next(observed))
    with pytest.raises(SecFilingGemmaLearnerFitError, match="60-second cap"):
        _build(learner_fit_case)
