from __future__ import annotations

import copy
from typing import Any

import pytest

import agent_benchmark.sec_filing_gemma_learner_prediction as prediction_module
from agent_benchmark.direct_edge_features import (
    MARKET_SENTIMENT_FEATURE_COLUMNS,
)
from agent_benchmark.downside_features import PRICE_FEATURE_COLUMNS
from agent_benchmark.sec_filing_gemma_contract import (
    DEVELOPMENT_FOLD_SPECS,
    build_contract_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_filing_gemma_features import (
    ABLATION_FEATURE_COLUMNS,
    FEATURE_ROW_SCHEMA_VERSION,
    FILING_CALENDAR_FEATURE_COLUMNS,
    MARKET_FEATURE_ROW_SCHEMA_VERSION,
    OWNED_DEVELOPMENT_FEATURE_BATCH_SCHEMA_VERSION,
    SEMANTIC_AGGREGATE_FEATURE_COLUMNS,
    SEMANTIC_FEATURE_COLUMNS,
)
from agent_benchmark.sec_filing_gemma_learner import (
    MODEL_TYPE,
    STATE_SCHEMA_VERSION,
    SecFilingGemmaTwoHeadLearner,
)
from agent_benchmark.sec_filing_gemma_learner_fit import (
    DEVELOPMENT_OOF_LEARNER_FIT_RECORD_SCHEMA_VERSION,
    DEVELOPMENT_OOF_LEARNER_FIT_VIEW_SCHEMA_VERSION,
    OWNED_DEVELOPMENT_OOF_LEARNER_FIT_BATCH_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_learner_prediction import (
    SecFilingGemmaLearnerPredictionError,
    build_owned_development_oof_prediction_batch,
    derive_development_oof_prediction_feature_batch,
    derive_development_oof_prediction_fold_model_bundle,
    derive_development_oof_prediction_fold_model_specs,
    derive_development_oof_prediction_input_specs,
    validate_owned_development_oof_prediction_batch,
)


FOLD_IDS = [item[0] for item in DEVELOPMENT_FOLD_SPECS]
VARIANT_IDS = ["semantic", "ablation"]


def _digest(label: str) -> str:
    return canonical_sha256({"prediction-test": label})


def _component_map(names: tuple[str, ...], seed: int) -> dict[str, str]:
    return {
        name: float(seed + (index + 1) / 1000.0).hex()
        for index, name in enumerate(names)
    }


def _source_feature_row(
    *,
    ordinal: int,
    session: str,
    universe_sha256: str,
    market_missing: bool = False,
    extraction_authenticated: bool = True,
    extraction_status: str = "valid",
) -> dict[str, Any]:
    accession = f"test-accession-{ordinal:02d}"
    price = _component_map(PRICE_FEATURE_COLUMNS, 10 * ordinal)
    sentiment = _component_map(MARKET_SENTIMENT_FEATURE_COLUMNS, 20 * ordinal)
    calendar = _component_map(FILING_CALENDAR_FEATURE_COLUMNS, 30 * ordinal)
    semantic = _component_map(SEMANTIC_AGGREGATE_FEATURE_COLUMNS, 40 * ordinal)
    missing: list[str] = []
    if market_missing:
        price[PRICE_FEATURE_COLUMNS[0]] = None  # type: ignore[assignment]
        missing = ["AAPL.close"]
    market_available = not market_missing
    reasons: list[str] = []
    if market_missing:
        reasons.append("missing_required_market_history")
    if not extraction_authenticated:
        reasons.append("missing_or_unauthenticated_extraction_evidence")
    prediction_available = not reasons
    semantic_vector = None
    ablation_vector = None
    if prediction_available:
        semantic_vector = [
            *(price[name] for name in PRICE_FEATURE_COLUMNS),
            *(sentiment[name] for name in MARKET_SENTIMENT_FEATURE_COLUMNS),
            *(calendar[name] for name in FILING_CALENDAR_FEATURE_COLUMNS),
            *(semantic[name] for name in SEMANTIC_AGGREGATE_FEATURE_COLUMNS),
        ]
        ablation_vector = [
            *(price[name] for name in PRICE_FEATURE_COLUMNS),
            *(sentiment[name] for name in MARKET_SENTIMENT_FEATURE_COLUMNS),
            *(calendar[name] for name in FILING_CALENDAR_FEATURE_COLUMNS),
            *(float(0.0).hex() for _ in SEMANTIC_AGGREGATE_FEATURE_COLUMNS),
        ]
    bindings = {
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "corpus_universe_sha256": universe_sha256,
        "universe_event_proof_sha256": _digest(f"universe-proof-{ordinal}"),
        "current_universe_record_sha256": _digest(f"universe-record-{ordinal}"),
        "prior_same_form_universe_record_sha256": _digest(f"prior-universe-{ordinal}"),
        "current_source_record_sha256": _digest(f"source-record-{ordinal}"),
        "prior_same_form_source_record_sha256": _digest(f"prior-source-{ordinal}"),
        "current_filing_sha256": _digest(f"filing-{ordinal}"),
        "prior_same_form_filing_sha256": _digest(f"prior-filing-{ordinal}"),
        "extraction_event_proof_sha256": _digest(f"extraction-proof-{ordinal}"),
        "extraction_identity_sha256": _digest(f"extraction-identity-{ordinal}"),
        "extraction_evidence_sha256": _digest(f"extraction-evidence-{ordinal}"),
        "extraction_output_sha256": _digest(f"extraction-output-{ordinal}"),
        "extraction_output_canonical_sha256": _digest(
            f"extraction-canonical-{ordinal}"
        ),
        "market_prefix_sha256": _digest(f"market-prefix-{ordinal}"),
        "market_prefix_proof_sha256": _digest(f"market-proof-{ordinal}"),
        "market_stage_manifest_sha256": _digest(f"market-stage-{ordinal}"),
        "source_manifest_sha256": _digest(f"source-manifest-{ordinal}"),
        "market_prefix_chain_identity_sha256": _digest(f"market-chain-{ordinal}"),
    }
    missing_hash = canonical_sha256(missing)
    market_body = {
        "schema_version": MARKET_FEATURE_ROW_SCHEMA_VERSION,
        "accession_number": accession,
        "decision_session": session,
        "market_prefix_chain_identity_sha256": bindings[
            "market_prefix_chain_identity_sha256"
        ],
        "market_available": market_available,
        "missing_market_observations": missing,
        "missing_market_observations_sha256": missing_hash,
        "price_regime_features_hex": price,
        "market_sentiment_features_hex": sentiment,
    }
    body = {
        "schema_version": FEATURE_ROW_SCHEMA_VERSION,
        "accession_number": accession,
        "form": "10-K" if ordinal % 2 else "10-Q",
        "artifact_stage": "development",
        "decision_session": session,
        "extraction_status": extraction_status,
        "extraction_evidence_authenticated": extraction_authenticated,
        "document_quality": "thin" if extraction_status == "invalid" else "usable",
        "semantic_available": bool(
            extraction_authenticated and extraction_status == "valid"
        ),
        "market_available": market_available,
        "prediction_available": prediction_available,
        "fit_eligible": prediction_available,
        "unavailable_reasons": reasons,
        "missing_market_observations": missing,
        "missing_market_observations_sha256": missing_hash,
        "bindings": bindings,
        "bindings_sha256": canonical_sha256(bindings),
        "market_feature_row_sha256": canonical_sha256(market_body),
        "price_regime_features_hex": price,
        "market_sentiment_features_hex": sentiment,
        "filing_calendar_features_hex": calendar,
        "semantic_aggregate_features_hex": semantic,
        "semantic_feature_names": list(SEMANTIC_FEATURE_COLUMNS),
        "semantic_feature_schema_sha256": canonical_sha256(
            list(SEMANTIC_FEATURE_COLUMNS)
        ),
        "semantic_feature_values_hex": semantic_vector,
        "ablation_feature_names": list(ABLATION_FEATURE_COLUMNS),
        "ablation_feature_schema_sha256": canonical_sha256(
            list(ABLATION_FEATURE_COLUMNS)
        ),
        "ablation_feature_values_hex": ablation_vector,
    }
    return {**body, "feature_row_sha256": canonical_sha256(body)}


def _rehash_source_feature_batch(batch: dict[str, Any]) -> None:
    for row in batch["feature_rows"]:
        body = {key: value for key, value in row.items() if key != "feature_row_sha256"}
        row["feature_row_sha256"] = canonical_sha256(body)
    batch["feature_row_sha256s"] = [
        row["feature_row_sha256"] for row in batch["feature_rows"]
    ]
    batch["feature_rows_sha256"] = canonical_sha256(batch["feature_rows"])
    body = {key: value for key, value in batch.items() if key != "feature_batch_sha256"}
    batch["feature_batch_sha256"] = canonical_sha256(body)


def _source_feature_batch() -> dict[str, Any]:
    universe = _digest("universe")
    rows = [
        _source_feature_row(ordinal=1, session="2003-01-03", universe_sha256=universe),
        _source_feature_row(
            ordinal=2,
            session="2005-01-03",
            universe_sha256=universe,
            extraction_status="invalid",
        ),
        _source_feature_row(
            ordinal=3,
            session="2008-01-02",
            universe_sha256=universe,
            market_missing=True,
        ),
        _source_feature_row(
            ordinal=4,
            session="2011-01-03",
            universe_sha256=universe,
            extraction_authenticated=False,
            extraction_status="unavailable",
        ),
        _source_feature_row(
            ordinal=5,
            session="2014-01-02",
            universe_sha256=universe,
            market_missing=True,
            extraction_authenticated=False,
            extraction_status="unavailable",
        ),
        _source_feature_row(ordinal=6, session="2017-01-03", universe_sha256=universe),
        _source_feature_row(ordinal=7, session="2019-01-02", universe_sha256=universe),
    ]
    body = {
        "schema_version": OWNED_DEVELOPMENT_FEATURE_BATCH_SCHEMA_VERSION,
        "development_root_scope_sha256": _digest("root"),
        "feature_assembly_plan_sha256": _digest("feature-plan"),
        "candidate_sha256": _digest("candidate"),
        "corpus_universe_sha256": universe,
        "development_sec_reader_receipt_sha256": _digest("sec-reader"),
        "development_market_reader_receipt_sha256": _digest("market-reader"),
        "development_model_reader_receipt_sha256": _digest("model-reader"),
        "event_count": len(rows),
        "event_plan_sha256": _digest("event-plan"),
        "feature_row_schema_version": FEATURE_ROW_SCHEMA_VERSION,
        "feature_row_sha256s": [row["feature_row_sha256"] for row in rows],
        "feature_rows_sha256": canonical_sha256(rows),
        "feature_rows": rows,
        "labels_included": False,
        "outcomes_included": False,
        "post_decision_market_rows_included": False,
        "training_membership_included": False,
        "learner_fit_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    return {**body, "feature_batch_sha256": canonical_sha256(body)}


def _trained_state(
    *,
    fold_id: str,
    cutoff: str,
    variant: str,
    candidate_sha256: str,
    training_set_sha256: str,
    shift: float = 0.0,
) -> dict[str, Any]:
    names = list(SEMANTIC_FEATURE_COLUMNS)
    matrix = [
        [shift + (row + 1) * (column + 1) / 1000.0 for column in range(len(names))]
        for row in range(6)
    ]
    learner = SecFilingGemmaTwoHeadLearner().fit(
        matrix,
        [0, 1, 0, 1, 0, 1],
        [-0.03, 0.02, -0.01, 0.04, -0.02, 0.03],
        feature_names=names,
        fit_metadata={
            "candidate_sha256": candidate_sha256,
            "head_variant": variant,
            "fold_id": fold_id,
            "train_label_maturity_through": cutoff,
            "training_set_sha256": training_set_sha256,
            "training_row_count": 6,
            "maximum_training_label_maturity_session": cutoff,
            "feature_schema_sha256": canonical_sha256(names),
        },
    )
    return learner.to_state()


def _source_fit_batch(candidate_sha256: str, universe_sha256: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    for fold_ordinal, (fold_id, cutoff, _first, _last) in enumerate(
        DEVELOPMENT_FOLD_SPECS, start=1
    ):
        membership_hash = _digest(f"membership-{fold_id}")
        source_view_hash = _digest(f"source-view-{fold_id}")
        states = {
            variant: _trained_state(
                fold_id=fold_id,
                cutoff=cutoff,
                variant=variant,
                candidate_sha256=candidate_sha256,
                training_set_sha256=membership_hash,
            )
            for variant in VARIANT_IDS
        }
        context = {
            "fold_train_cutoff_session": cutoff,
            "training_set_count": 6,
            "training_positive_count": 3,
            "training_set_membership_sha256": membership_hash,
            "semantic_training_feature_matrix_sha256": _digest(
                f"semantic-matrix-{fold_id}"
            ),
            "ablation_training_feature_matrix_sha256": _digest(
                f"ablation-matrix-{fold_id}"
            ),
            "training_binary_target_sha256": _digest(f"binary-{fold_id}"),
            "training_edge_target_sha256": _digest(f"edge-{fold_id}"),
            "training_set_max_label_maturity_session": cutoff,
            "semantic_fold_state_sha256": states["semantic"]["state_sha256"],
            "ablation_fold_state_sha256": states["ablation"]["state_sha256"],
        }
        pair: dict[str, dict[str, Any]] = {}
        for variant in VARIANT_IDS:
            state = states[variant]
            fit_ordinal = 2 * (fold_ordinal - 1) + VARIANT_IDS.index(variant) + 1
            identity = {"fold_id": fold_id, "head_variant": variant}
            record_body = {
                "schema_version": DEVELOPMENT_OOF_LEARNER_FIT_RECORD_SCHEMA_VERSION,
                "fit_ordinal": fit_ordinal,
                "source_training_view_ordinal": fold_ordinal,
                "training_view_id": fold_id,
                "head_variant": variant,
                "source_training_view_sha256": source_view_hash,
                "training_input_identity": identity,
                "training_input_identity_sha256": canonical_sha256(identity),
                "fit_metadata": copy.deepcopy(state["fit_metadata"]),
                "fit_metadata_sha256": canonical_sha256(state["fit_metadata"]),
                "learner_state": state,
                "learner_state_sha256": state["state_sha256"],
                "learner_parameters_sha256": state["parameters_sha256"],
            }
            record = {
                **record_body,
                "learner_fit_record_sha256": canonical_sha256(record_body),
            }
            records.append(record)
            pair[variant] = record
        learner_context = {
            key: value
            for key, value in context.items()
            if key not in {"semantic_fold_state_sha256", "ablation_fold_state_sha256"}
        }
        view_body = {
            "schema_version": DEVELOPMENT_OOF_LEARNER_FIT_VIEW_SCHEMA_VERSION,
            "source_training_view_ordinal": fold_ordinal,
            "training_view_id": fold_id,
            "source_training_view_sha256": source_view_hash,
            "training_row_count": 6,
            "training_positive_count": 3,
            "learner_input_context": learner_context,
            "learner_input_context_sha256": canonical_sha256(learner_context),
            "semantic_fit_record_sha256": pair["semantic"][
                "learner_fit_record_sha256"
            ],
            "ablation_fit_record_sha256": pair["ablation"][
                "learner_fit_record_sha256"
            ],
            "semantic_learner_state_sha256": states["semantic"]["state_sha256"],
            "ablation_learner_state_sha256": states["ablation"]["state_sha256"],
            "prediction_fold_context": context,
            "prediction_fold_context_sha256": canonical_sha256(context),
            "beta_1_1_climatology_hex": float(0.5).hex(),
        }
        views.append(
            {**view_body, "learner_fit_view_sha256": canonical_sha256(view_body)}
        )
    deferred = [{"training_view_id": "intermediate_frozen_through_2018"}]
    body = {
        "schema_version": OWNED_DEVELOPMENT_OOF_LEARNER_FIT_BATCH_SCHEMA_VERSION,
        "artifact_stage": "development",
        "development_root_scope_sha256": _digest("root"),
        "development_oof_learner_fit_plan_sha256": _digest("fit-plan"),
        "source_training_membership_assembly_plan_sha256": _digest("membership-plan"),
        "source_training_membership_projection_sha256": _digest(
            "membership-projection"
        ),
        "source_training_membership_batch_sha256": _digest("membership-batch"),
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "candidate_sha256": candidate_sha256,
        "corpus_universe_sha256": universe_sha256,
        "calendar_sessions_sha256": _digest("calendar"),
        "development_cutoff_session": "2018-12-31",
        "source_training_view_count": 6,
        "source_training_view_ids": [*FOLD_IDS, "intermediate_frozen_through_2018"],
        "authorized_training_view_count": 5,
        "authorized_training_view_ids": FOLD_IDS,
        "deferred_training_view_count": 1,
        "deferred_training_views": deferred,
        "deferred_training_views_sha256": canonical_sha256(deferred),
        "model_variant_count": 2,
        "model_variant_ids": VARIANT_IDS,
        "learner_fit_count": 10,
        "learner_state_count": 10,
        "learner_model_type": MODEL_TYPE,
        "learner_state_schema_version": STATE_SCHEMA_VERSION,
        "learner_config_sha256": canonical_sha256(records[0]["learner_state"]["config"]),
        "fit_order_rule": "view_ordinal_ascending_then_semantic_then_ablation_exactly_once",
        "maximum_fit_seconds": 60,
        "learner_fit_views": views,
        "learner_fit_views_sha256": canonical_sha256(views),
        "learner_fit_records": records,
        "learner_fit_records_sha256": canonical_sha256(records),
        "learner_states_sha256": canonical_sha256(
            [record["learner_state"] for record in records]
        ),
        "source_training_membership_batch_included": False,
        "training_membership_rows_included": False,
        "training_feature_matrices_included": False,
        "training_target_vectors_included": False,
        "deferred_training_view_state_included": False,
        "learner_states_included": True,
        "compact_fit_audit_included": True,
        "prediction_included": False,
        "prediction_authorized": False,
        "candidate_selection_authorized": False,
        "threshold_action_authorized": False,
        "holdout_access_authorized": False,
        "ledger_mutation_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    return {**body, "learner_fit_batch_sha256": canonical_sha256(body)}


def _plan(feature_batch: dict[str, Any], fold_bundle: dict[str, Any]) -> dict[str, Any]:
    input_specs = derive_development_oof_prediction_input_specs(feature_batch)
    fold_specs = derive_development_oof_prediction_fold_model_specs(fold_bundle)
    available = sum(spec["prediction_available"] for spec in input_specs)
    body = {
        "development_root_scope_sha256": _digest("root"),
        "contract_sha256": feature_batch["contract_sha256"],
        "candidate_sha256": feature_batch["candidate_sha256"],
        "corpus_universe_sha256": feature_batch["corpus_universe_sha256"],
        "calendar_sessions_sha256": fold_bundle["calendar_sessions_sha256"],
        "development_cutoff_session": fold_bundle["development_cutoff_session"],
        "source_feature_batch_sha256": feature_batch["source_feature_batch_sha256"],
        "prediction_feature_batch_sha256": feature_batch[
            "prediction_feature_batch_sha256"
        ],
        "source_development_oof_learner_fit_batch_sha256": fold_bundle[
            "source_learner_fit_batch_sha256"
        ],
        "prediction_fold_model_bundle_sha256": fold_bundle[
            "prediction_fold_model_bundle_sha256"
        ],
        "source_event_count": feature_batch["source_event_count"],
        "authorized_fold_count": 5,
        "authorized_fold_ids": FOLD_IDS,
        "prediction_fold_model_count": 5,
        "prediction_fold_model_specs": fold_specs,
        "prediction_fold_model_specs_sha256": canonical_sha256(fold_specs),
        "model_variant_count": 2,
        "model_variant_ids": VARIANT_IDS,
        "learner_state_count": 10,
        "learner_model_type": MODEL_TYPE,
        "learner_state_schema_version": STATE_SCHEMA_VERSION,
        "feature_schema_sha256": feature_batch["feature_schema_sha256"],
        "prediction_input_count": len(input_specs),
        "available_prediction_input_count": available,
        "unavailable_prediction_input_count": len(input_specs) - available,
        "prediction_input_specs": input_specs,
        "prediction_input_specs_sha256": canonical_sha256(input_specs),
        "maximum_prediction_calls": 2 * available,
        "maximum_prediction_seconds": 60,
    }
    return {
        **body,
        "development_oof_prediction_plan_sha256": canonical_sha256(body),
    }


def _rehash_plan(plan: dict[str, Any]) -> None:
    body = {
        key: value
        for key, value in plan.items()
        if key != "development_oof_prediction_plan_sha256"
    }
    plan["development_oof_prediction_plan_sha256"] = canonical_sha256(body)


def _rehash_compact_feature_batch(batch: dict[str, Any]) -> None:
    for row in batch["prediction_feature_rows"]:
        body = {
            key: value
            for key, value in row.items()
            if key != "prediction_feature_input_sha256"
        }
        row["prediction_feature_input_sha256"] = canonical_sha256(body)
    batch["prediction_feature_rows_sha256"] = canonical_sha256(
        batch["prediction_feature_rows"]
    )
    body = {
        key: value
        for key, value in batch.items()
        if key != "prediction_feature_batch_sha256"
    }
    batch["prediction_feature_batch_sha256"] = canonical_sha256(body)


def _rehash_fold_bundle(bundle: dict[str, Any]) -> None:
    for model in bundle["fold_models"]:
        body = {
            key: value
            for key, value in model.items()
            if key != "prediction_fold_model_sha256"
        }
        model["prediction_fold_model_sha256"] = canonical_sha256(body)
    bundle["fold_models_sha256"] = canonical_sha256(bundle["fold_models"])
    body = {
        key: value
        for key, value in bundle.items()
        if key != "prediction_fold_model_bundle_sha256"
    }
    bundle["prediction_fold_model_bundle_sha256"] = canonical_sha256(body)


def _allow_compact_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(
        plan: dict[str, Any],
        *,
        expected_development_oof_prediction_plan_sha256: str,
    ) -> str:
        assert (
            plan["development_oof_prediction_plan_sha256"]
            == expected_development_oof_prediction_plan_sha256
        )
        return expected_development_oof_prediction_plan_sha256

    monkeypatch.setattr(
        prediction_module,
        "validate_development_oof_prediction_plan",
        validate,
    )


@pytest.fixture(scope="module")
def prediction_case() -> dict[str, Any]:
    source_features = _source_feature_batch()
    features = derive_development_oof_prediction_feature_batch(source_features)
    source_fit = _source_fit_batch(
        source_features["candidate_sha256"],
        source_features["corpus_universe_sha256"],
    )
    fold_bundle = derive_development_oof_prediction_fold_model_bundle(source_fit)
    return {
        "source_features": source_features,
        "features": features,
        "source_fit": source_fit,
        "fold_bundle": fold_bundle,
        "plan": _plan(features, fold_bundle),
    }


def _build(case: dict[str, Any]) -> dict[str, Any]:
    return build_owned_development_oof_prediction_batch(
        development_oof_prediction_plan=case["plan"],
        expected_development_oof_prediction_plan_sha256=case["plan"][
            "development_oof_prediction_plan_sha256"
        ],
        prediction_feature_batch=case["features"],
        expected_prediction_feature_batch_sha256=case["features"][
            "prediction_feature_batch_sha256"
        ],
        prediction_fold_model_bundle=case["fold_bundle"],
        expected_prediction_fold_model_bundle_sha256=case["fold_bundle"][
            "prediction_fold_model_bundle_sha256"
        ],
    )


def _validate(case: dict[str, Any], batch: dict[str, Any]) -> str:
    return validate_owned_development_oof_prediction_batch(
        batch,
        development_oof_prediction_plan=case["plan"],
        expected_development_oof_prediction_plan_sha256=case["plan"][
            "development_oof_prediction_plan_sha256"
        ],
        prediction_feature_batch=case["features"],
        expected_prediction_feature_batch_sha256=case["features"][
            "prediction_feature_batch_sha256"
        ],
        prediction_fold_model_bundle=case["fold_bundle"],
        expected_prediction_fold_model_bundle_sha256=case["fold_bundle"][
            "prediction_fold_model_bundle_sha256"
        ],
        expected_prediction_batch_sha256=batch["prediction_batch_sha256"],
    )


def test_projection_is_exact_complete_and_excludes_view_six(
    prediction_case: dict[str, Any],
) -> None:
    features = prediction_case["features"]
    assert prediction_case["source_features"]["feature_rows"][1][
        "semantic_available"
    ] is False
    assert features["prediction_feature_rows"][0]["prediction_available"] is True
    assert [row["event_binding"]["decision_session"] for row in features["prediction_feature_rows"]] == [
        "2005-01-03",
        "2008-01-02",
        "2011-01-03",
        "2014-01-02",
        "2017-01-03",
    ]
    assert [row["unavailable_reason"] for row in features["prediction_feature_rows"]] == [
        None,
        "missing_required_market_features",
        "missing_required_extraction_features",
        "missing_required_market_and_extraction_features",
        None,
    ]
    assert features["prediction_feature_rows"][0]["prediction_available"] is True
    assert "intermediate_frozen_through_2018" not in repr(
        prediction_case["fold_bundle"]
    )
    assert len(prediction_case["fold_bundle"]["fold_models"]) == 5


def test_build_plus_validate_uses_exact_call_budget_and_never_fits(
    prediction_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_compact_plan(monkeypatch)
    calls: list[tuple[str, str]] = []
    original_predict = SecFilingGemmaTwoHeadLearner.predict_components

    def counted(self: SecFilingGemmaTwoHeadLearner, features: Any):
        assert self.fit_metadata is not None
        calls.append((self.fit_metadata.fold_id, self.fit_metadata.head_variant))
        return original_predict(self, features)

    def forbidden_fit(*args: Any, **kwargs: Any):
        raise AssertionError("prediction boundary called fit")

    monkeypatch.setattr(SecFilingGemmaTwoHeadLearner, "predict_components", counted)
    monkeypatch.setattr(SecFilingGemmaTwoHeadLearner, "fit", forbidden_fit)
    batch = _build(prediction_case)
    assert _validate(prediction_case, batch) == batch["prediction_batch_sha256"]
    assert calls == [
        ("fold_1", "semantic"),
        ("fold_1", "ablation"),
        ("fold_5", "semantic"),
        ("fold_5", "ablation"),
    ]
    assert batch["prediction_call_count"] == 4
    assert batch["available_prediction_count"] == 2
    assert batch["unavailable_prediction_count"] == 3
    for row in batch["raw_prediction_rows"]:
        for field in (
            "semantic_cash_probability_hex",
            "semantic_expected_edge_hex",
            "ablation_cash_probability_hex",
            "ablation_expected_edge_hex",
        ):
            if row[field] is not None:
                assert float.fromhex(row[field]).hex() == row[field]


def test_prediction_batch_is_byte_deterministic(
    prediction_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_compact_plan(monkeypatch)
    assert _build(prediction_case) == _build(prediction_case)


def test_duplicate_prediction_session_is_rejected() -> None:
    source = _source_feature_batch()
    source["feature_rows"][2]["decision_session"] = "2005-01-03"
    source["feature_rows"][2]["accession_number"] = "test-accession-99"
    market_body = {
        "schema_version": MARKET_FEATURE_ROW_SCHEMA_VERSION,
        "accession_number": source["feature_rows"][2]["accession_number"],
        "decision_session": "2005-01-03",
        "market_prefix_chain_identity_sha256": source["feature_rows"][2]["bindings"][
            "market_prefix_chain_identity_sha256"
        ],
        "market_available": source["feature_rows"][2]["market_available"],
        "missing_market_observations": source["feature_rows"][2][
            "missing_market_observations"
        ],
        "missing_market_observations_sha256": source["feature_rows"][2][
            "missing_market_observations_sha256"
        ],
        "price_regime_features_hex": source["feature_rows"][2][
            "price_regime_features_hex"
        ],
        "market_sentiment_features_hex": source["feature_rows"][2][
            "market_sentiment_features_hex"
        ],
    }
    source["feature_rows"][2]["market_feature_row_sha256"] = canonical_sha256(
        market_body
    )
    _rehash_source_feature_batch(source)
    with pytest.raises(SecFilingGemmaLearnerPredictionError, match="decision session"):
        derive_development_oof_prediction_feature_batch(source)


def test_prediction_rejects_mutation_after_first_call_without_retry(
    prediction_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_compact_plan(monkeypatch)
    original_predict = SecFilingGemmaTwoHeadLearner.predict_components
    calls = 0

    def mutating(self: SecFilingGemmaTwoHeadLearner, features: Any):
        nonlocal calls
        calls += 1
        result = original_predict(self, features)
        self.target_center += 0.001
        return result

    monkeypatch.setattr(SecFilingGemmaTwoHeadLearner, "predict_components", mutating)
    with pytest.raises(SecFilingGemmaLearnerPredictionError, match="mutated"):
        _build(prediction_case)
    assert calls == 1


def test_later_feature_and_fold_state_changes_do_not_rewrite_earlier_rows(
    prediction_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_compact_plan(monkeypatch)
    baseline = _build(prediction_case)

    later_feature_case = copy.deepcopy(prediction_case)
    later_row = later_feature_case["features"]["prediction_feature_rows"][-1]
    changed = float.fromhex(later_row["semantic_feature_values_hex"][0]) + 0.25
    later_row["semantic_feature_values_hex"][0] = changed.hex()
    later_row["ablation_feature_values_hex"][0] = changed.hex()
    later_row["semantic_feature_values_sha256"] = canonical_sha256(
        later_row["semantic_feature_values_hex"]
    )
    later_row["ablation_feature_values_sha256"] = canonical_sha256(
        later_row["ablation_feature_values_hex"]
    )
    _rehash_compact_feature_batch(later_feature_case["features"])
    later_feature_case["plan"] = _plan(
        later_feature_case["features"], later_feature_case["fold_bundle"]
    )
    changed_features = _build(later_feature_case)
    assert changed_features["raw_prediction_rows"][:4] == baseline[
        "raw_prediction_rows"
    ][:4]
    assert changed_features["raw_prediction_rows"][4] != baseline[
        "raw_prediction_rows"
    ][4]

    later_state_case = copy.deepcopy(prediction_case)
    model = later_state_case["fold_bundle"]["fold_models"][-1]
    context = model["prediction_fold_context"]
    new_state = _trained_state(
        fold_id="fold_5",
        cutoff="2016-12-30",
        variant="semantic",
        candidate_sha256=later_state_case["features"]["candidate_sha256"],
        training_set_sha256=context["training_set_membership_sha256"],
        shift=0.75,
    )
    model["semantic_learner_state"] = new_state
    model["semantic_learner_state_sha256"] = new_state["state_sha256"]
    context["semantic_fold_state_sha256"] = new_state["state_sha256"]
    model["prediction_fold_context_sha256"] = canonical_sha256(context)
    _rehash_fold_bundle(later_state_case["fold_bundle"])
    later_state_case["plan"] = _plan(
        later_state_case["features"], later_state_case["fold_bundle"]
    )
    changed_state = _build(later_state_case)
    assert changed_state["raw_prediction_rows"][:4] == baseline[
        "raw_prediction_rows"
    ][:4]
    assert changed_state["raw_prediction_rows"][4] != baseline[
        "raw_prediction_rows"
    ][4]


def test_compact_ablation_and_source_decomposition_tampering_are_rejected(
    prediction_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_compact_plan(monkeypatch)
    compact_case = copy.deepcopy(prediction_case)
    row = compact_case["features"]["prediction_feature_rows"][0]
    row["ablation_feature_values_hex"][-1] = float(1.0).hex()
    row["ablation_feature_values_sha256"] = canonical_sha256(
        row["ablation_feature_values_hex"]
    )
    _rehash_compact_feature_batch(compact_case["features"])
    with pytest.raises(SecFilingGemmaLearnerPredictionError, match="ablation"):
        _build(compact_case)

    source = copy.deepcopy(prediction_case["source_features"])
    source["feature_rows"][1]["semantic_feature_values_hex"][0] = float(999.0).hex()
    _rehash_source_feature_batch(source)
    with pytest.raises(SecFilingGemmaLearnerPredictionError, match="vector"):
        derive_development_oof_prediction_feature_batch(source)


def test_future_leaking_state_metadata_and_plan_crossbinding_are_rejected(
    prediction_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_compact_plan(monkeypatch)
    bundle = copy.deepcopy(prediction_case["fold_bundle"])
    model = bundle["fold_models"][0]
    state = model["semantic_learner_state"]
    state["fit_metadata"]["train_label_maturity_through"] = "2018-12-31"
    state_body = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = canonical_sha256(state_body)
    model["semantic_learner_state_sha256"] = state["state_sha256"]
    model["prediction_fold_context"]["semantic_fold_state_sha256"] = state[
        "state_sha256"
    ]
    model["prediction_fold_context_sha256"] = canonical_sha256(
        model["prediction_fold_context"]
    )
    _rehash_fold_bundle(bundle)
    with pytest.raises(SecFilingGemmaLearnerPredictionError, match="state|chronology"):
        derive_development_oof_prediction_fold_model_specs(bundle)

    case = copy.deepcopy(prediction_case)
    case["plan"]["prediction_fold_model_bundle_sha256"] = _digest("wrong-bundle")
    _rehash_plan(case["plan"])
    with pytest.raises(SecFilingGemmaLearnerPredictionError, match="bind"):
        _build(case)


def test_validator_rejects_edge_outside_exact_state_clip_without_predicting(
    prediction_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_compact_plan(monkeypatch)
    batch = _build(prediction_case)
    last = batch["raw_prediction_rows"][-1]
    last["semantic_expected_edge_hex"] = float(1.0).hex()
    row_body = {key: value for key, value in last.items() if key != "raw_prediction_row_sha256"}
    last["raw_prediction_row_sha256"] = canonical_sha256(row_body)
    batch["raw_prediction_rows_sha256"] = canonical_sha256(batch["raw_prediction_rows"])
    batch["raw_prediction_tip_sha256"] = last["raw_prediction_row_sha256"]
    batch_body = {key: value for key, value in batch.items() if key != "prediction_batch_sha256"}
    batch["prediction_batch_sha256"] = canonical_sha256(batch_body)

    def forbidden_predict(*args: Any, **kwargs: Any):
        raise AssertionError("validator called predict_components")

    monkeypatch.setattr(
        SecFilingGemmaTwoHeadLearner,
        "predict_components",
        forbidden_predict,
    )
    with pytest.raises(SecFilingGemmaLearnerPredictionError, match="clip bounds"):
        _validate(prediction_case, batch)


def test_nested_subclasses_and_noncanonical_hex_are_rejected(
    prediction_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_compact_plan(monkeypatch)

    class ListSubclass(list):
        pass

    case = copy.deepcopy(prediction_case)
    case["features"]["prediction_feature_rows"][0]["source_unavailable_reasons"] = (
        ListSubclass()
    )
    with pytest.raises(SecFilingGemmaLearnerPredictionError, match="subclass"):
        _build(case)

    case = copy.deepcopy(prediction_case)
    row = case["features"]["prediction_feature_rows"][0]
    row["semantic_feature_values_hex"][0] = "0x1p+0"
    row["semantic_feature_values_sha256"] = canonical_sha256(
        row["semantic_feature_values_hex"]
    )
    row["ablation_feature_values_hex"][0] = "0x1p+0"
    row["ablation_feature_values_sha256"] = canonical_sha256(
        row["ablation_feature_values_hex"]
    )
    _rehash_compact_feature_batch(case["features"])
    with pytest.raises(SecFilingGemmaLearnerPredictionError, match="canonical"):
        _build(case)
