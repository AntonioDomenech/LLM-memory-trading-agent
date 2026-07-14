from __future__ import annotations

import copy
import inspect
import json
from typing import Any, Mapping

import pytest

import agent_benchmark.sec_filing_gemma_training_membership as membership_module
from agent_benchmark.sec_filing_gemma_contract import (
    BRIER_TARGET_COST_BPS,
    canonical_sha256,
)
from agent_benchmark.sec_filing_gemma_features import (
    ABLATION_FEATURE_COLUMNS,
    SEMANTIC_FEATURE_COLUMNS,
)
from agent_benchmark.sec_filing_gemma_learner import SecFilingGemmaTwoHeadLearner
from agent_benchmark.sec_filing_gemma_training_membership import (
    DEVELOPMENT_TRAINING_VIEW_SCHEMA_VERSION,
    OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_BATCH_SCHEMA_VERSION,
    SecFilingGemmaTrainingMembershipError,
    build_owned_development_training_membership_batch,
    validate_owned_development_training_membership_batch,
)


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


def _view_specs() -> list[dict[str, Any]]:
    rows = [
        (
            "fold_1",
            "development_out_of_fold",
            "2004-12-31",
            "2005-01-03",
            "2007-12-31",
            "development",
        ),
        (
            "fold_2",
            "development_out_of_fold",
            "2007-12-31",
            "2008-01-02",
            "2010-12-31",
            "development",
        ),
        (
            "fold_3",
            "development_out_of_fold",
            "2010-12-31",
            "2011-01-03",
            "2013-12-31",
            "development",
        ),
        (
            "fold_4",
            "development_out_of_fold",
            "2013-12-31",
            "2014-01-02",
            "2016-12-30",
            "development",
        ),
        (
            "fold_5",
            "development_out_of_fold",
            "2016-12-30",
            "2017-01-03",
            "2018-12-31",
            "development",
        ),
        (
            "intermediate_frozen_through_2018",
            "intermediate_frozen_refit",
            "2018-12-31",
            "2019-01-01",
            "2023-12-31",
            "intermediate",
        ),
    ]
    return [
        {
            "view_ordinal": ordinal,
            "training_view_id": view_id,
            "view_kind": view_kind,
            "training_source_stage": "development",
            "prediction_stage": prediction_stage,
            "train_label_maturity_through": cutoff,
            "prediction_window_first_date": first,
            "prediction_window_last_date": last,
            "state_updates_inside_prediction_window": False,
        }
        for ordinal, (
            view_id,
            view_kind,
            cutoff,
            first,
            last,
            prediction_stage,
        ) in enumerate(rows, start=1)
    ]


def _feature_row(ordinal: int, *, eligible: bool) -> dict[str, Any]:
    width = len(SEMANTIC_FEATURE_COLUMNS)
    common = [float(ordinal * 100 + index).hex() for index in range(width - 12)]
    semantic = common + [float(ordinal * 1000 + index + 1).hex() for index in range(12)]
    ablation = common + [float(0.0).hex()] * 12
    body = {
        "accession_number": f"0000320193-0{ordinal}-00000{ordinal}",
        "form": "10-K" if ordinal % 2 else "10-Q",
        "decision_session": f"200{ordinal}-01-02",
        "prediction_available": eligible,
        "fit_eligible": eligible,
        "unavailable_reasons": [] if eligible else ["missing_required_market_history"],
        "bindings": {
            "extraction_identity_sha256": _digest(f"extraction-{ordinal}"),
            "market_prefix_chain_identity_sha256": _digest(f"market-prefix-{ordinal}"),
        },
        "market_feature_row_sha256": _digest(f"market-feature-{ordinal}"),
        "semantic_feature_schema_sha256": canonical_sha256(
            list(SEMANTIC_FEATURE_COLUMNS)
        ),
        "semantic_feature_values_hex": semantic if eligible else None,
        "ablation_feature_schema_sha256": canonical_sha256(
            list(ABLATION_FEATURE_COLUMNS)
        ),
        "ablation_feature_values_hex": ablation if eligible else None,
    }
    return {**body, "feature_row_sha256": canonical_sha256(body)}


def _label_row(
    feature: Mapping[str, Any], ordinal: int, *, positive: bool
) -> dict[str, Any]:
    body = {
        "accession_number": feature["accession_number"],
        "decision_session": feature["decision_session"],
        "label_maturity_session": f"200{ordinal}-02-02",
        "feature_row_sha256": feature["feature_row_sha256"],
        "cash_beats_long_10bps": positive,
        "cash_active_log_edge_10bps_hex": (
            float(0.02 + ordinal / 1000).hex()
            if positive
            else float(-0.02 - ordinal / 1000).hex()
        ),
    }
    return {**body, "label_evidence_sha256": canonical_sha256(body)}


def _rehash_feature(feature: dict[str, Any]) -> None:
    body = {key: value for key, value in feature.items() if key != "feature_row_sha256"}
    feature["feature_row_sha256"] = canonical_sha256(body)


def _rehash_label(label: dict[str, Any]) -> None:
    body = {key: value for key, value in label.items() if key != "label_evidence_sha256"}
    label["label_evidence_sha256"] = canonical_sha256(body)


def _rehash_sources(inputs: dict[str, Any]) -> None:
    feature_batch = inputs["feature_batch"]
    feature_body = {
        key: value for key, value in feature_batch.items() if key != "feature_batch_sha256"
    }
    feature_batch["feature_batch_sha256"] = canonical_sha256(feature_body)
    label_batch = inputs["label_batch"]
    label_body = {
        key: value for key, value in label_batch.items() if key != "label_batch_sha256"
    }
    label_batch["label_batch_sha256"] = canonical_sha256(label_body)


def _synthetic_inputs() -> dict[str, Any]:
    features = [_feature_row(index, eligible=index != 4) for index in range(1, 5)]
    labels = [
        _label_row(feature, index, positive=index % 2 == 0)
        for index, feature in enumerate(features, start=1)
    ]
    audits = [
        {
            "event_ordinal": index,
            "accession_number": feature["accession_number"],
            "decision_session": feature["decision_session"],
            "feature_row_sha256": feature["feature_row_sha256"],
            "label_maturity_session": label["label_maturity_session"],
            "matured_by_development_cutoff": True,
            "label_evidence_sha256": label["label_evidence_sha256"],
        }
        for index, (feature, label) in enumerate(zip(features, labels, strict=True), start=1)
    ]
    feature_batch: dict[str, Any] = {
        "development_root_scope_sha256": _digest("root"),
        "feature_assembly_plan_sha256": _digest("feature-plan"),
        "candidate_sha256": _digest("candidate"),
        "corpus_universe_sha256": _digest("universe"),
        "event_count": len(features),
        "feature_rows": features,
        "private_raw_secret": "FEATURE-SOURCE-MUST-NOT-ESCAPE",
    }
    label_batch: dict[str, Any] = {
        "development_root_scope_sha256": feature_batch[
            "development_root_scope_sha256"
        ],
        "source_feature_assembly_plan_sha256": feature_batch[
            "feature_assembly_plan_sha256"
        ],
        "source_feature_batch_sha256": "0" * 64,
        "label_assembly_plan_sha256": _digest("label-plan"),
        "candidate_sha256": feature_batch["candidate_sha256"],
        "corpus_universe_sha256": feature_batch["corpus_universe_sha256"],
        "event_count": len(features),
        "matured_label_count": len(features),
        "unmatured_event_count": 0,
        "maturity_audit_rows": audits,
        "label_evidence_rows": labels,
        "private_adjusted_open_path": "LABEL-SOURCE-MUST-NOT-ESCAPE",
    }
    specs = _view_specs()
    variants = [
        {"variant_ordinal": 1, "variant_id": "semantic"},
        {"variant_ordinal": 2, "variant_id": "ablation"},
    ]
    plan: dict[str, Any] = {
        "development_root_scope_sha256": feature_batch[
            "development_root_scope_sha256"
        ],
        "source_feature_assembly_plan_sha256": feature_batch[
            "feature_assembly_plan_sha256"
        ],
        "source_label_assembly_plan_sha256": label_batch[
            "label_assembly_plan_sha256"
        ],
        "candidate_sha256": feature_batch["candidate_sha256"],
        "corpus_universe_sha256": feature_batch["corpus_universe_sha256"],
        "calendar_sessions_sha256": _digest("calendar"),
        "development_cutoff_session": "2018-12-31",
        "event_count": len(features),
        "matured_event_count": len(features),
        "unmatured_event_count": 0,
        "membership_view_count": len(specs),
        "membership_view_specs": specs,
        "membership_view_specs_sha256": canonical_sha256(specs),
        "model_variant_specs_sha256": canonical_sha256(variants),
        "target_cost_bps": BRIER_TARGET_COST_BPS,
        "binary_target_field": "cash_beats_long_10bps",
        "binary_target_encoding": "false_to_0_true_to_1",
        "edge_target_field": "cash_active_log_edge_10bps_hex",
        "membership_maturity_rule": "chronological_maturity_only",
        "feature_eligibility_rule": "causal_feature_availability_only",
        "membership_order_rule": "source_event_order_without_reordering",
        "shared_variant_support_rule": "one_support_for_both_variants",
        "minimum_training_row_count": 2,
    }
    plan["training_membership_assembly_plan_sha256"] = canonical_sha256(plan)
    _rehash_sources(
        {"feature_batch": feature_batch, "label_batch": label_batch}
    )
    label_batch["source_feature_batch_sha256"] = feature_batch[
        "feature_batch_sha256"
    ]
    _rehash_sources(
        {"feature_batch": feature_batch, "label_batch": label_batch}
    )
    return {"plan": plan, "feature_batch": feature_batch, "label_batch": label_batch}


@pytest.fixture
def synthetic_membership_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    inputs = _synthetic_inputs()

    def validated_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(dict(plan))

    def validated_sources(
        *,
        plan: Mapping[str, Any],
        source_feature_batch: Mapping[str, Any],
        source_label_batch: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        del plan
        return (
            {},
            {},
            copy.deepcopy(dict(source_feature_batch)),
            copy.deepcopy(dict(source_label_batch)),
        )

    monkeypatch.setattr(membership_module, "_validated_membership_plan", validated_plan)
    monkeypatch.setattr(membership_module, "_validated_sources", validated_sources)
    return inputs


def _build(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return build_owned_development_training_membership_batch(
        training_membership_assembly_plan=inputs["plan"],
        source_feature_batch=inputs["feature_batch"],
        source_label_batch=inputs["label_batch"],
    )


def _change_label(
    inputs: dict[str, Any], index: int, *, positive: bool, edge: float
) -> None:
    label = inputs["label_batch"]["label_evidence_rows"][index]
    label["cash_beats_long_10bps"] = positive
    label["cash_active_log_edge_10bps_hex"] = edge.hex()
    _rehash_label(label)
    inputs["label_batch"]["maturity_audit_rows"][index][
        "label_evidence_sha256"
    ] = label["label_evidence_sha256"]
    _rehash_sources(inputs)


def test_build_emits_six_exact_shared_support_training_views(
    synthetic_membership_sources: dict[str, Any],
) -> None:
    batch = _build(synthetic_membership_sources)
    assert batch["schema_version"] == (
        OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_BATCH_SCHEMA_VERSION
    )
    assert batch["training_view_ids"] == [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "fold_5",
        "intermediate_frozen_through_2018",
    ]
    assert len(batch["training_views"]) == 6
    expected_context_keys = {
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
    expected_metadata_keys = {
        "candidate_sha256",
        "head_variant",
        "fold_id",
        "train_label_maturity_through",
        "training_set_sha256",
        "training_row_count",
        "maximum_training_label_maturity_session",
        "feature_schema_sha256",
    }
    for view in batch["training_views"]:
        assert view["schema_version"] == DEVELOPMENT_TRAINING_VIEW_SCHEMA_VERSION
        assert view["training_set_count"] == 3
        assert [
            member["source_event_ordinal"]
            for member in view["training_set_membership"]
        ] == [1, 2, 3]
        assert view["training_binary_targets"] == [0, 1, 0]
        assert all(
            type(target) is int and target in {0, 1}
            for target in view["training_binary_targets"]
        )
        assert len(view["semantic_training_features_hex"]) == 3
        assert len(view["ablation_training_features_hex"]) == 3
        assert all(
            len(row) == len(SEMANTIC_FEATURE_COLUMNS)
            and all(float.fromhex(value).hex() == value for value in row)
            for row in view["semantic_training_features_hex"]
        )
        assert all(
            len(row) == len(ABLATION_FEATURE_COLUMNS)
            and all(float.fromhex(value).hex() == value for value in row)
            for row in view["ablation_training_features_hex"]
        )
        assert view["semantic_training_features_hex"] != view[
            "ablation_training_features_hex"
        ]
        assert set(view["learner_input_context"]) == expected_context_keys
        assert set(view["semantic_fit_metadata_template"]) == expected_metadata_keys
        assert set(view["ablation_fit_metadata_template"]) == expected_metadata_keys
        assert view["semantic_fit_metadata_template"]["head_variant"] == "semantic"
        assert view["ablation_fit_metadata_template"]["head_variant"] == "ablation"
        assert view["semantic_fit_metadata_template"]["training_set_sha256"] == view[
            "training_set_membership_sha256"
        ]
        assert view["ablation_fit_metadata_template"]["training_set_sha256"] == view[
            "training_set_membership_sha256"
        ]
    excluded = batch["event_audit_rows"][3]["training_view_decisions"]
    assert all(item["included"] is False for item in excluded)
    assert all(item["exclusion_reasons"] == ["feature_not_fit_eligible"] for item in excluded)


def test_training_support_expands_only_at_exact_maturity_cutoffs(
    synthetic_membership_sources: dict[str, Any],
) -> None:
    inputs = copy.deepcopy(synthetic_membership_sources)
    cutoff_maturities = [
        "2004-01-30",
        "2004-06-30",
        "2004-12-31",
        "2007-12-31",
        "2010-12-31",
        "2013-12-31",
        "2016-12-30",
        "2018-12-31",
    ]
    features = [
        _feature_row(ordinal, eligible=True)
        for ordinal in range(1, len(cutoff_maturities) + 2)
    ]
    labels: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for ordinal, (feature, maturity) in enumerate(
        zip(features, cutoff_maturities, strict=False),
        start=1,
    ):
        label = _label_row(feature, ordinal, positive=ordinal % 2 == 0)
        label["label_maturity_session"] = maturity
        _rehash_label(label)
        labels.append(label)
        audits.append(
            {
                "event_ordinal": ordinal,
                "accession_number": feature["accession_number"],
                "decision_session": feature["decision_session"],
                "feature_row_sha256": feature["feature_row_sha256"],
                "label_maturity_session": maturity,
                "matured_by_development_cutoff": True,
                "label_evidence_sha256": label["label_evidence_sha256"],
            }
        )

    unmatured_ordinal = len(features)
    unmatured_feature = features[-1]
    audits.append(
        {
            "event_ordinal": unmatured_ordinal,
            "accession_number": unmatured_feature["accession_number"],
            "decision_session": unmatured_feature["decision_session"],
            "feature_row_sha256": unmatured_feature["feature_row_sha256"],
            "label_maturity_session": "2019-01-02",
            "matured_by_development_cutoff": False,
            "label_evidence_sha256": None,
        }
    )

    feature_batch = inputs["feature_batch"]
    feature_batch["feature_rows"] = features
    feature_batch["event_count"] = len(features)
    label_batch = inputs["label_batch"]
    label_batch["maturity_audit_rows"] = audits
    label_batch["label_evidence_rows"] = labels
    label_batch["event_count"] = len(features)
    label_batch["matured_label_count"] = len(labels)
    label_batch["unmatured_event_count"] = 1
    plan = inputs["plan"]
    plan["event_count"] = len(features)
    plan["matured_event_count"] = len(labels)
    plan["unmatured_event_count"] = 1
    plan_body = {
        key: value
        for key, value in plan.items()
        if key != "training_membership_assembly_plan_sha256"
    }
    plan["training_membership_assembly_plan_sha256"] = canonical_sha256(plan_body)
    _rehash_sources(inputs)
    label_batch["source_feature_batch_sha256"] = feature_batch[
        "feature_batch_sha256"
    ]
    _rehash_sources(inputs)

    batch = _build(inputs)
    expected_supports = [
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7, 8],
    ]
    for view, expected_support in zip(
        batch["training_views"], expected_supports, strict=True
    ):
        assert [
            member["source_event_ordinal"]
            for member in view["training_set_membership"]
        ] == expected_support
        assert view["training_set_max_label_maturity_session"] == (
            cutoff_maturities[expected_support[-1] - 1]
        )
        assert len(view["training_binary_targets"]) == len(expected_support)
        assert 0 < view["training_positive_count"] < len(expected_support)

    transitioning_decisions = batch["event_audit_rows"][3][
        "training_view_decisions"
    ]
    assert transitioning_decisions[0]["included"] is False
    assert transitioning_decisions[0]["exclusion_reasons"] == [
        "label_not_mature_by_view_cutoff"
    ]
    assert all(
        decision["included"] is True for decision in transitioning_decisions[1:]
    )

    unmatured_audit = batch["event_audit_rows"][-1]
    assert unmatured_audit["label_evidence_sha256"] is None
    assert all(
        decision["included"] is False
        and decision["exclusion_reasons"] == [
            "label_not_mature_by_view_cutoff"
        ]
        for decision in unmatured_audit["training_view_decisions"]
    )
    assert all(
        unmatured_ordinal
        not in {
            member["source_event_ordinal"]
            for member in view["training_set_membership"]
        }
        for view in batch["training_views"]
    )


def test_membership_identity_is_independent_of_included_and_excluded_outcomes(
    synthetic_membership_sources: dict[str, Any],
) -> None:
    original = _build(synthetic_membership_sources)
    changed = copy.deepcopy(synthetic_membership_sources)
    _change_label(changed, 0, positive=True, edge=0.123)
    _change_label(changed, 3, positive=False, edge=-0.456)
    rebuilt = _build(changed)

    for before, after in zip(
        original["training_views"], rebuilt["training_views"], strict=True
    ):
        assert before["training_set_membership"] == after["training_set_membership"]
        assert before["training_set_membership_sha256"] == after[
            "training_set_membership_sha256"
        ]
        assert before["semantic_training_features_hex"] == after[
            "semantic_training_features_hex"
        ]
        assert before["ablation_training_features_hex"] == after[
            "ablation_training_features_hex"
        ]
        assert before["training_binary_target_sha256"] != after[
            "training_binary_target_sha256"
        ]
        assert before["training_edge_target_sha256"] != after[
            "training_edge_target_sha256"
        ]
        assert after["training_binary_targets"] == [1, 1, 0]


@pytest.mark.parametrize("failure", ["too_few", "one_class", "identical_matrices"])
def test_training_views_fail_closed_on_unfit_support(
    synthetic_membership_sources: dict[str, Any], failure: str
) -> None:
    inputs = copy.deepcopy(synthetic_membership_sources)
    if failure == "too_few":
        for index in (1, 2):
            feature = inputs["feature_batch"]["feature_rows"][index]
            feature["prediction_available"] = False
            feature["fit_eligible"] = False
            feature["unavailable_reasons"] = ["missing_required_market_history"]
            feature["semantic_feature_values_hex"] = None
            feature["ablation_feature_values_hex"] = None
            _rehash_feature(feature)
            audit = inputs["label_batch"]["maturity_audit_rows"][index]
            audit["feature_row_sha256"] = feature["feature_row_sha256"]
            label = inputs["label_batch"]["label_evidence_rows"][index]
            label["feature_row_sha256"] = feature["feature_row_sha256"]
            _rehash_label(label)
            audit["label_evidence_sha256"] = label["label_evidence_sha256"]
        expected = "fewer than the frozen minimum rows"
    elif failure == "one_class":
        for index in range(3):
            _change_label(inputs, index, positive=True, edge=0.1 + index / 100)
        expected = "does not contain both binary classes"
    else:
        for index in range(3):
            feature = inputs["feature_batch"]["feature_rows"][index]
            feature["semantic_feature_values_hex"] = copy.deepcopy(
                feature["ablation_feature_values_hex"]
            )
            _rehash_feature(feature)
            audit = inputs["label_batch"]["maturity_audit_rows"][index]
            audit["feature_row_sha256"] = feature["feature_row_sha256"]
            label = inputs["label_batch"]["label_evidence_rows"][index]
            label["feature_row_sha256"] = feature["feature_row_sha256"]
            _rehash_label(label)
            audit["label_evidence_sha256"] = label["label_evidence_sha256"]
        expected = "semantic and ablation matrices are identical"
    _rehash_sources(inputs)
    with pytest.raises(SecFilingGemmaTrainingMembershipError, match=expected):
        _build(inputs)


def test_batch_validation_replays_exact_sources_and_external_pins(
    synthetic_membership_sources: dict[str, Any],
) -> None:
    batch = _build(synthetic_membership_sources)
    assert validate_owned_development_training_membership_batch(
        batch,
        training_membership_assembly_plan=synthetic_membership_sources["plan"],
        expected_training_membership_assembly_plan_sha256=(
            synthetic_membership_sources["plan"][
                "training_membership_assembly_plan_sha256"
            ]
        ),
        source_feature_batch=synthetic_membership_sources["feature_batch"],
        expected_source_feature_batch_sha256=synthetic_membership_sources[
            "feature_batch"
        ]["feature_batch_sha256"],
        source_label_batch=synthetic_membership_sources["label_batch"],
        expected_source_label_batch_sha256=synthetic_membership_sources[
            "label_batch"
        ]["label_batch_sha256"],
        expected_training_membership_batch_sha256=batch[
            "training_membership_batch_sha256"
        ],
    ) == batch["training_membership_batch_sha256"]

    changed = copy.deepcopy(batch)
    changed["training_views"][0]["training_binary_targets"][0] = 1
    with pytest.raises(SecFilingGemmaTrainingMembershipError, match="checksum changed"):
        validate_owned_development_training_membership_batch(
            changed,
            training_membership_assembly_plan=synthetic_membership_sources["plan"],
            expected_training_membership_assembly_plan_sha256=(
                synthetic_membership_sources["plan"][
                    "training_membership_assembly_plan_sha256"
                ]
            ),
            source_feature_batch=synthetic_membership_sources["feature_batch"],
            expected_source_feature_batch_sha256=synthetic_membership_sources[
                "feature_batch"
            ]["feature_batch_sha256"],
            source_label_batch=synthetic_membership_sources["label_batch"],
            expected_source_label_batch_sha256=synthetic_membership_sources[
                "label_batch"
            ]["label_batch_sha256"],
            expected_training_membership_batch_sha256=batch[
                "training_membership_batch_sha256"
            ],
        )


def test_public_apis_exercise_real_plan_and_source_validation_glue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_inputs()
    plan = inputs["plan"]
    feature_batch = inputs["feature_batch"]
    label_batch = inputs["label_batch"]
    feature_plan = {
        "feature_assembly_plan_sha256": plan[
            "source_feature_assembly_plan_sha256"
        ]
    }
    label_plan = {
        "label_assembly_plan_sha256": plan[
            "source_label_assembly_plan_sha256"
        ],
        "source_feature_assembly_plan": feature_plan,
    }
    plan["schema_version"] = (
        membership_module.DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_SCHEMA_VERSION
    )
    plan["source_label_assembly_plan"] = label_plan
    plan_body = {
        key: value
        for key, value in plan.items()
        if key != "training_membership_assembly_plan_sha256"
    }
    plan["training_membership_assembly_plan_sha256"] = canonical_sha256(
        plan_body
    )

    calls = {"plan": 0, "feature": 0, "label": 0}

    def validate_plan(
        value: Mapping[str, Any],
        *,
        expected_training_membership_assembly_plan_sha256: str,
    ) -> str:
        calls["plan"] += 1
        observed = value["training_membership_assembly_plan_sha256"]
        body = {
            key: item
            for key, item in value.items()
            if key != "training_membership_assembly_plan_sha256"
        }
        if (
            observed != expected_training_membership_assembly_plan_sha256
            or observed != canonical_sha256(body)
        ):
            raise ValueError("membership plan pin changed")
        return observed

    def validate_feature(
        value: Mapping[str, Any],
        *,
        feature_assembly_plan: Mapping[str, Any],
        expected_feature_assembly_plan_sha256: str,
        expected_feature_batch_sha256: str,
    ) -> str:
        calls["feature"] += 1
        if (
            feature_assembly_plan["feature_assembly_plan_sha256"]
            != expected_feature_assembly_plan_sha256
            or value["feature_batch_sha256"]
            != expected_feature_batch_sha256
        ):
            raise ValueError("feature source pin changed")
        return expected_feature_batch_sha256

    def validate_label(
        value: Mapping[str, Any],
        *,
        label_assembly_plan: Mapping[str, Any],
        expected_label_assembly_plan_sha256: str,
        source_feature_batch: Mapping[str, Any],
        expected_source_feature_batch_sha256: str,
        expected_label_batch_sha256: str,
    ) -> str:
        calls["label"] += 1
        if (
            label_assembly_plan["label_assembly_plan_sha256"]
            != expected_label_assembly_plan_sha256
            or source_feature_batch["feature_batch_sha256"]
            != expected_source_feature_batch_sha256
            or value["label_batch_sha256"] != expected_label_batch_sha256
        ):
            raise ValueError("label source pin changed")
        return expected_label_batch_sha256

    monkeypatch.setattr(
        "agent_benchmark.sec_filing_gemma_stage_authorization."
        "validate_development_training_membership_assembly_plan",
        validate_plan,
    )
    monkeypatch.setattr(
        membership_module,
        "validate_owned_development_feature_batch",
        validate_feature,
    )
    monkeypatch.setattr(
        membership_module,
        "validate_owned_development_label_batch",
        validate_label,
    )

    batch = build_owned_development_training_membership_batch(
        training_membership_assembly_plan=plan,
        source_feature_batch=feature_batch,
        source_label_batch=label_batch,
    )
    assert validate_owned_development_training_membership_batch(
        batch,
        training_membership_assembly_plan=plan,
        expected_training_membership_assembly_plan_sha256=plan[
            "training_membership_assembly_plan_sha256"
        ],
        source_feature_batch=feature_batch,
        expected_source_feature_batch_sha256=feature_batch[
            "feature_batch_sha256"
        ],
        source_label_batch=label_batch,
        expected_source_label_batch_sha256=label_batch["label_batch_sha256"],
        expected_training_membership_batch_sha256=batch[
            "training_membership_batch_sha256"
        ],
    ) == batch["training_membership_batch_sha256"]
    assert calls == {"plan": 3, "feature": 2, "label": 2}

    with pytest.raises(
        SecFilingGemmaTrainingMembershipError,
        match="assembly plan is not externally pinned",
    ):
        validate_owned_development_training_membership_batch(
            batch,
            training_membership_assembly_plan=plan,
            expected_training_membership_assembly_plan_sha256="f" * 64,
            source_feature_batch=feature_batch,
            expected_source_feature_batch_sha256=feature_batch[
                "feature_batch_sha256"
            ],
            source_label_batch=label_batch,
            expected_source_label_batch_sha256=label_batch[
                "label_batch_sha256"
            ],
            expected_training_membership_batch_sha256=batch[
                "training_membership_batch_sha256"
            ],
        )

    for crossed_field, crossed_value, expected_message in (
        ("feature", "f" * 64, "source feature batch is not externally pinned"),
        ("label", "f" * 64, "source label batch is not externally pinned"),
    ):
        with pytest.raises(
            SecFilingGemmaTrainingMembershipError,
            match=expected_message,
        ):
            validate_owned_development_training_membership_batch(
                batch,
                training_membership_assembly_plan=plan,
                expected_training_membership_assembly_plan_sha256=plan[
                    "training_membership_assembly_plan_sha256"
                ],
                source_feature_batch=feature_batch,
                expected_source_feature_batch_sha256=(
                    crossed_value
                    if crossed_field == "feature"
                    else feature_batch["feature_batch_sha256"]
                ),
                source_label_batch=label_batch,
                expected_source_label_batch_sha256=(
                    crossed_value
                    if crossed_field == "label"
                    else label_batch["label_batch_sha256"]
                ),
                expected_training_membership_batch_sha256=batch[
                    "training_membership_batch_sha256"
                ],
            )

    crossed_label_batch = copy.deepcopy(label_batch)
    crossed_label_batch["candidate_sha256"] = "f" * 64
    with pytest.raises(
        SecFilingGemmaTrainingMembershipError,
        match="source batches crossed their plan",
    ):
        build_owned_development_training_membership_batch(
            training_membership_assembly_plan=plan,
            source_feature_batch=feature_batch,
            source_label_batch=crossed_label_batch,
        )


def test_public_batch_does_not_copy_private_sources_or_authorize_learning(
    synthetic_membership_sources: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("membership assembly must not fit or predict")

    monkeypatch.setattr(SecFilingGemmaTwoHeadLearner, "fit", forbidden)
    monkeypatch.setattr(SecFilingGemmaTwoHeadLearner, "predict_components", forbidden)
    batch = _build(synthetic_membership_sources)
    encoded = json.dumps(batch, sort_keys=True)
    assert "FEATURE-SOURCE-MUST-NOT-ESCAPE" not in encoded
    assert "LABEL-SOURCE-MUST-NOT-ESCAPE" not in encoded
    assert "source_feature_batch" not in batch
    assert "source_label_batch" not in batch
    assert "training_membership_assembly_plan" not in batch
    for field in (
        "compact_adjusted_open_paths_included",
        "full_market_rows_included",
        "post_cutoff_market_data_included",
        "outcome_based_membership_filtering_permitted",
        "row_rebalancing_permitted",
        "learner_fit_authorized",
        "prediction_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
    ):
        assert batch[field] is False
    source = inspect.getsource(build_owned_development_training_membership_batch)
    assert ".fit(" not in source
    assert ".predict" not in source
