from __future__ import annotations

import copy
from datetime import date, timedelta
from functools import lru_cache
import inspect
import math
import random
from typing import Any

import pytest

from agent_benchmark.sec_filing_gemma_learner import (
    SecFilingGemmaTwoHeadLearner,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    EXPECTED_EDGE_GATE,
    FEATURES,
    MARKET_FEATURES,
    MEANING_FEATURES,
    MINIMUM_CLASS_ROWS,
    MINIMUM_TRAINING_ROWS,
    PROBABILITY_GATE,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_features import (
    build_sec_gemma_online_risk_overlay_feature_row,
)
from agent_benchmark.sec_gemma_online_risk_overlay_learner import (
    ARM_VECTOR_FIELDS,
    FEATURE_ARMS,
    SecFilingGemmaTwoHeadLearner as OverlayTwoHeadLearner,
    SecGemmaOnlineRiskOverlayLearnerError,
    build_online_overlay_fit_audit,
    build_online_overlay_lesson,
    build_online_overlay_prediction,
    build_online_overlay_prediction_from_fit,
    evaluate_online_overlay_gate,
    select_online_overlay_feature_arm,
    validate_online_overlay_feature_row,
    validate_online_overlay_fit_audit,
    validate_online_overlay_lesson,
    validate_online_overlay_prediction_from_fit,
)
from tests import test_sec_filing_gemma_features as v1_helpers


def _session(start: date, offset: int) -> str:
    return (start + timedelta(days=offset)).isoformat()


@lru_cache(maxsize=1)
def _owned_base_feature() -> dict[str, Any]:
    market = v1_helpers._market_evidence(
        v1_helpers._canonical_rows(), "online-overlay-v2"
    )
    universe = v1_helpers._universe_evidence()
    extraction = v1_helpers._extraction_proof(universe["proof"])
    return build_sec_gemma_online_risk_overlay_feature_row(
        **v1_helpers._feature_kwargs(
            market,
            universe["proof"],
            extraction,
        )
    )


def _feature(
    ordinal: int,
    *,
    available: bool = True,
    decision_session: str | None = None,
) -> dict[str, Any]:
    start = date(2000, 1, 3)
    decision = decision_session or _session(start, ordinal * 30)
    row = copy.deepcopy(_owned_base_feature())
    row["accession_number"] = (
        f"0000320193-{ordinal:02d}-{ordinal:06d}"
    )
    row["decision_session"] = decision
    row["acceptance_datetime"] = decision.replace("-", "") + "163000"
    if not available:
        row["extraction_status"] = "unavailable"
        row["extraction_authenticated"] = False
        row["schema_valid_extraction"] = False
        row["document_quality"] = None
        row["prediction_available"] = False
        row["fit_eligible"] = False
        row["meaning_nonzero"] = False
        reasons = ["missing_or_unauthenticated_extraction_evidence"]
        row["unavailable_reasons"] = reasons
        row["unavailable_reasons_sha256"] = canonical_sha256(reasons)
        row["meaning_values_hex"] = {
            name: 0.0.hex() for name in MEANING_FEATURES
        }
        row["semantic_quality_risk_hex"] = 1.0.hex()
        row["semantic_values_hex"] = None
        row["no_filing_meaning_values_hex"] = None
        row["no_gemma_channel_values_hex"] = None
    _rehash_feature(row)
    return row


def _lesson(
    ordinal: int,
    *,
    positive: bool,
    available: bool = True,
    maturity_session: str | None = None,
) -> dict[str, Any]:
    feature = _feature(ordinal, available=available)
    maturity = maturity_session or _session(
        date.fromisoformat(feature["decision_session"]), 22
    )
    return build_online_overlay_lesson(
        feature_row=feature,
        expected_feature_row_sha256=feature["feature_row_sha256"],
        maturity_session=maturity,
        active_log_edge_10bps=0.01 if positive else -0.01,
        label_evidence_sha256=f"{ordinal % 16:x}" * 64,
    )


def _ready_lessons() -> list[dict[str, Any]]:
    return [
        _lesson(index, positive=index % 2 == 0)
        for index in range(MINIMUM_TRAINING_ROWS)
    ]


def _predict(
    current: dict[str, Any],
    lessons: list[dict[str, Any]],
    *,
    arm: str = "semantic",
) -> dict[str, Any]:
    return build_online_overlay_prediction(
        feature_row=current,
        expected_feature_row_sha256=current["feature_row_sha256"],
        matured_lessons=lessons,
        expected_lesson_row_sha256s=[
            item["lesson_row_sha256"] for item in lessons
        ],
        arm=arm,
        decision_session=current["decision_session"],
    )


def _fit(
    lessons: list[dict[str, Any]],
    *,
    arm: str = "semantic",
    as_of_session: str = "2002-06-15",
) -> dict[str, Any]:
    return build_online_overlay_fit_audit(
        matured_lessons=lessons,
        expected_lesson_row_sha256s=[
            item["lesson_row_sha256"] for item in lessons
        ],
        arm=arm,
        as_of_session=as_of_session,
    )


def _rehash_feature(row: dict[str, Any]) -> None:
    body = {
        key: value for key, value in row.items() if key != "feature_row_sha256"
    }
    row["feature_row_sha256"] = canonical_sha256(body)


def _rehash_lesson(row: dict[str, Any]) -> None:
    body = {
        key: value for key, value in row.items() if key != "lesson_row_sha256"
    }
    row["lesson_row_sha256"] = canonical_sha256(body)


def _rehash_fit(fit: dict[str, Any]) -> None:
    body = {
        key: value for key, value in fit.items() if key != "fit_audit_sha256"
    }
    fit["fit_audit_sha256"] = canonical_sha256(body)


def _rehash_learner_state(state: dict[str, Any]) -> None:
    state["parameters_sha256"] = canonical_sha256(state["parameters"])
    body = {
        key: value for key, value in state.items() if key != "state_sha256"
    }
    state["state_sha256"] = canonical_sha256(body)


def test_feature_rows_and_all_three_exact_arms_are_validated() -> None:
    row = _feature(1)
    validated = validate_online_overlay_feature_row(
        row,
        expected_feature_row_sha256=row["feature_row_sha256"],
    )
    assert validated == row
    assert tuple(FEATURE_ARMS) == (
        "semantic",
        "no_filing_meaning",
        "no_gemma_channel",
    )

    selected = {
        arm: select_online_overlay_feature_arm(
            row,
            expected_feature_row_sha256=row["feature_row_sha256"],
            arm=arm,
        )
        for arm in FEATURE_ARMS
    }
    for arm, audit in selected.items():
        assert audit["vector_field"] == ARM_VECTOR_FIELDS[arm]
        assert audit["values_hex"] == row[ARM_VECTOR_FIELDS[arm]]
        assert audit["feature_arm_sha256"] == canonical_sha256(
            {
                key: value
                for key, value in audit.items()
                if key != "feature_arm_sha256"
            }
        )
    semantic = [
        float.fromhex(value) for value in selected["semantic"]["values_hex"]
    ]
    no_meaning = [
        float.fromhex(value)
        for value in selected["no_filing_meaning"]["values_hex"]
    ]
    no_gemma = [
        float.fromhex(value)
        for value in selected["no_gemma_channel"]["values_hex"]
    ]
    market_count = len(MARKET_FEATURES)
    meaning_count = len(MEANING_FEATURES)
    assert semantic[:market_count] == no_meaning[:market_count]
    assert semantic[:market_count] == no_gemma[:market_count]
    assert no_meaning[
        market_count : market_count + meaning_count
    ] == [0.0] * meaning_count
    assert no_gemma[
        market_count : market_count + meaning_count + 1
    ] == [0.0] * (meaning_count + 1)
    assert no_meaning[-2] == semantic[-2]
    assert semantic[-1] == no_meaning[-1] == no_gemma[-1]


def test_lesson_builder_binds_feature_target_edge_and_external_pin() -> None:
    feature = _feature(2)
    lesson = build_online_overlay_lesson(
        feature_row=feature,
        expected_feature_row_sha256=feature["feature_row_sha256"],
        maturity_session=_session(
            date.fromisoformat(feature["decision_session"]), 22
        ),
        active_log_edge_10bps=math.nextafter(1e-12, math.inf),
        label_evidence_sha256="a" * 64,
    )
    assert lesson["binary_cash_win_target"] == 1
    assert lesson["active_log_edge_10bps_hex"] == math.nextafter(
        1e-12, math.inf
    ).hex()
    assert validate_online_overlay_lesson(
        lesson,
        expected_lesson_row_sha256=lesson["lesson_row_sha256"],
    ) == lesson

    changed = copy.deepcopy(lesson)
    changed["binary_cash_win_target"] = 0
    _rehash_lesson(changed)
    with pytest.raises(
        SecGemmaOnlineRiskOverlayLearnerError,
        match="target differs",
    ):
        validate_online_overlay_lesson(
            changed,
            expected_lesson_row_sha256=changed["lesson_row_sha256"],
        )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayLearnerError,
        match="externally pinned",
    ):
        validate_online_overlay_lesson(
            lesson,
            expected_lesson_row_sha256="b" * 64,
        )


def test_fit_checkpoint_can_be_built_and_replayed_without_a_filing() -> None:
    lessons = _ready_lessons()
    first = _fit(lessons, as_of_session="2002-06-15")
    shuffled = copy.deepcopy(lessons)
    random.Random(91).shuffle(shuffled)
    repeated = _fit(shuffled, as_of_session="2002-06-15")

    assert first == repeated
    assert first["as_of_session"] == "2002-06-15"
    assert first["learner_ready"] is True
    assert first["training_row_count"] == MINIMUM_TRAINING_ROWS
    assert len(first["admitted_lessons"]) == len(lessons)
    assert all(
        len(member["feature_values_hex"]) == len(FEATURES)
        for member in first["training_members"]
    )
    assert validate_online_overlay_fit_audit(
        first,
        expected_fit_audit_sha256=first["fit_audit_sha256"],
    ) == first


def test_overlay_uses_exact_pinned_numpy_learner_byte_for_byte() -> None:
    assert OverlayTwoHeadLearner is SecFilingGemmaTwoHeadLearner

    fit = _fit(_ready_lessons(), as_of_session="2002-06-15")
    members = fit["training_members"]
    matrix = [
        [float.fromhex(value) for value in member["feature_values_hex"]]
        for member in members
    ]
    binary = [
        member["binary_cash_win_target"] for member in members
    ]
    edge = [
        float.fromhex(member["active_log_edge_10bps_hex"])
        for member in members
    ]
    pinned = SecFilingGemmaTwoHeadLearner().fit(
        matrix,
        binary,
        edge,
        feature_names=FEATURES,
        fit_metadata=fit["fit_state"]["fit_metadata"],
    )

    assert fit["fit_state"] == pinned.to_state()
    current = _feature(30, decision_session="2002-07-01")
    vector = [
        float.fromhex(value)
        for value in current["semantic_values_hex"]
    ]
    overlay_prediction = OverlayTwoHeadLearner.from_state(
        fit["fit_state"]
    ).predict_components(vector)
    pinned_prediction = pinned.predict_components(vector)
    assert overlay_prediction["cash_win_probability_10bps"] == list(
        pinned_prediction["cash_win_probability_10bps"]
    )
    assert overlay_prediction["expected_active_log_edge_10bps"] == list(
        pinned_prediction["expected_active_log_edge_10bps"]
    )


@pytest.mark.parametrize("seed", range(12))
def test_overlay_numpy_learner_randomized_states_are_exact(seed: int) -> None:
    rng = random.Random(seed)
    row_count = 40 + seed
    feature_count = len(FEATURES)
    matrix = [
        [rng.uniform(-3.0, 3.0) for _ in range(feature_count)]
        for _ in range(row_count)
    ]
    binary = [
        int(sum(row[:4]) + rng.uniform(-1.0, 1.0) > 0.0)
        for row in matrix
    ]
    if len(set(binary)) != 2:
        binary[0] = 0
        binary[-1] = 1
    edge = [
        max(-0.45, min(0.45, 0.02 * sum(row[4:8]) + rng.uniform(-0.01, 0.01)))
        for row in matrix
    ]
    metadata = {
        "candidate_sha256": canonical_sha256(
            {"seed": seed, "kind": "candidate"}
        ),
        "head_variant": "semantic",
        "fold_id": f"randomized-{seed}",
        "train_label_maturity_through": "2023-12-29",
        "training_set_sha256": canonical_sha256(
            {"seed": seed, "kind": "training"}
        ),
        "training_row_count": row_count,
        "maximum_training_label_maturity_session": "2023-12-29",
        "feature_schema_sha256": canonical_sha256(list(FEATURES)),
    }
    overlay = OverlayTwoHeadLearner().fit(
        matrix,
        binary,
        edge,
        feature_names=FEATURES,
        fit_metadata=metadata,
    )
    pinned = SecFilingGemmaTwoHeadLearner().fit(
        matrix,
        binary,
        edge,
        feature_names=FEATURES,
        fit_metadata=metadata,
    )

    assert overlay.to_state() == pinned.to_state()
    for _ in range(5):
        vector = [
            rng.uniform(-4.0, 4.0) for _ in range(feature_count)
        ]
        assert overlay.predict_components(vector) == pinned.predict_components(
            vector
        )


def test_maturity_at_fit_boundary_is_admitted_but_later_is_rejected() -> None:
    as_of = "2002-06-15"
    lessons = _ready_lessons()
    boundary = copy.deepcopy(lessons[-1])
    boundary["maturity_session"] = as_of
    _rehash_lesson(boundary)
    lessons[-1] = boundary

    fit = _fit(lessons, as_of_session=as_of)

    assert fit["training_row_count"] == len(lessons)
    assert fit["fit_state"]["fit_metadata"][
        "maximum_training_label_maturity_session"
    ] == as_of

    future = copy.deepcopy(boundary)
    future["maturity_session"] = "2002-06-16"
    _rehash_lesson(future)
    lessons[-1] = future
    with pytest.raises(
        SecGemmaOnlineRiskOverlayLearnerError,
        match="Future outcome",
    ):
        _fit(lessons, as_of_session=as_of)


def test_prediction_from_fit_reuses_frozen_state_without_refitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit(_ready_lessons(), as_of_session="2002-06-15")
    current = _feature(30, decision_session="2002-07-01")

    def forbidden_fit(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("prediction-from-fit must not refit")

    monkeypatch.setattr(
        SecFilingGemmaTwoHeadLearner,
        "fit",
        forbidden_fit,
    )
    prediction = build_online_overlay_prediction_from_fit(
        feature_row=current,
        expected_feature_row_sha256=current["feature_row_sha256"],
        fit_audit=fit,
        expected_fit_audit_sha256=fit["fit_audit_sha256"],
    )

    assert prediction["frozen_fit_reuse"] is True
    assert prediction["fit_as_of_session"] == fit["as_of_session"]
    assert prediction["fit_audit"] == fit
    assert prediction["fit_audit_sha256"] == fit["fit_audit_sha256"]
    assert validate_online_overlay_prediction_from_fit(
        prediction,
        expected_prediction_row_sha256=prediction[
            "prediction_row_sha256"
        ],
        feature_row=current,
        expected_feature_row_sha256=current["feature_row_sha256"],
        fit_audit=fit,
        expected_fit_audit_sha256=fit["fit_audit_sha256"],
    ) == prediction


def test_same_fit_checkpoint_serves_multiple_filings_and_unavailable_row() -> None:
    fit = _fit(_ready_lessons(), as_of_session="2002-07-01")
    first_current = _feature(30, decision_session="2002-07-01")
    second_current = _feature(31, decision_session="2002-07-01")
    unavailable_current = _feature(
        32,
        available=False,
        decision_session="2002-07-01",
    )

    first = build_online_overlay_prediction_from_fit(
        feature_row=first_current,
        expected_feature_row_sha256=first_current["feature_row_sha256"],
        fit_audit=fit,
        expected_fit_audit_sha256=fit["fit_audit_sha256"],
    )
    second = build_online_overlay_prediction_from_fit(
        feature_row=second_current,
        expected_feature_row_sha256=second_current["feature_row_sha256"],
        fit_audit=fit,
        expected_fit_audit_sha256=fit["fit_audit_sha256"],
    )
    unavailable = build_online_overlay_prediction_from_fit(
        feature_row=unavailable_current,
        expected_feature_row_sha256=unavailable_current[
            "feature_row_sha256"
        ],
        fit_audit=fit,
        expected_fit_audit_sha256=fit["fit_audit_sha256"],
    )

    assert first["fit_audit"] == second["fit_audit"] == fit
    assert first["fit_audit_sha256"] == second["fit_audit_sha256"]
    assert first["frozen_fit_reuse"] is False
    assert second["frozen_fit_reuse"] is False
    assert unavailable["learner_ready"] is True
    assert unavailable["prediction_available"] is False
    assert unavailable["fitted_prediction_available"] is False
    assert unavailable["raw_gate_pass"] is False


def test_compatibility_wrapper_is_exact_fit_then_prediction_composition() -> None:
    current = _feature(30)
    lessons = _ready_lessons()
    fit = _fit(
        lessons,
        as_of_session=current["decision_session"],
    )
    composed = build_online_overlay_prediction_from_fit(
        feature_row=current,
        expected_feature_row_sha256=current["feature_row_sha256"],
        fit_audit=fit,
        expected_fit_audit_sha256=fit["fit_audit_sha256"],
    )

    assert _predict(current, lessons) == composed


def test_fit_and_prediction_validators_reject_semantic_tampering() -> None:
    fit = _fit(_ready_lessons())

    changed = copy.deepcopy(fit)
    coefficient = changed["fit_state"]["parameters"][
        "logistic_coefficients"
    ][0]
    changed["fit_state"]["parameters"]["logistic_coefficients"][0] = (
        math.nextafter(float.fromhex(coefficient), math.inf).hex()
    )
    _rehash_learner_state(changed["fit_state"])
    changed["fit_state_sha256"] = changed["fit_state"]["state_sha256"]
    _rehash_fit(changed)
    with pytest.raises(
        SecGemmaOnlineRiskOverlayLearnerError,
        match="deterministic replay",
    ):
        validate_online_overlay_fit_audit(
            changed,
            expected_fit_audit_sha256=changed["fit_audit_sha256"],
        )

    current = _feature(30)
    prediction = build_online_overlay_prediction_from_fit(
        feature_row=current,
        expected_feature_row_sha256=current["feature_row_sha256"],
        fit_audit=fit,
        expected_fit_audit_sha256=fit["fit_audit_sha256"],
    )
    altered_prediction = copy.deepcopy(prediction)
    altered_prediction["accession_number"] = "tampered"
    with pytest.raises(
        SecGemmaOnlineRiskOverlayLearnerError,
        match="differs",
    ):
        validate_online_overlay_prediction_from_fit(
            altered_prediction,
            expected_prediction_row_sha256=prediction[
                "prediction_row_sha256"
            ],
            feature_row=current,
            expected_feature_row_sha256=current["feature_row_sha256"],
            fit_audit=fit,
            expected_fit_audit_sha256=fit["fit_audit_sha256"],
        )


def test_unready_and_unavailable_events_emit_no_fitted_prediction_or_action() -> None:
    current = _feature(30)
    too_few = _ready_lessons()[:-1]
    unready = _predict(current, too_few)
    assert unready["learner_ready"] is False
    assert unready["prediction_available"] is True
    assert unready["fitted_prediction_available"] is False
    assert unready["action_available"] is False
    assert unready["gate_schedule_overlay"] is False
    assert unready["gate_audit"] is None
    assert unready["fit_audit"]["fit_state"] is None

    class_unready = [
        _lesson(index, positive=index < MINIMUM_CLASS_ROWS - 1)
        for index in range(MINIMUM_TRAINING_ROWS)
    ]
    class_result = _predict(current, class_unready)
    assert class_result["learner_ready"] is False
    assert class_result["fit_audit"]["training_positive_count"] == (
        MINIMUM_CLASS_ROWS - 1
    )

    unavailable_current = _feature(30, available=False)
    unavailable = _predict(unavailable_current, _ready_lessons())
    assert unavailable["learner_ready"] is True
    assert unavailable["current_feature_available"] is False
    assert unavailable["prediction_available"] is False
    assert unavailable["fitted_prediction_available"] is False
    assert unavailable["action_available"] is False
    assert unavailable["gate_schedule_overlay"] is False
    assert unavailable["gate_audit"] is None
    assert unavailable["fit_audit"]["fit_state"] is not None


def test_only_matured_earlier_lessons_can_reach_a_decision() -> None:
    current = _feature(30)
    lessons = _ready_lessons()
    baseline = _predict(current, lessons)
    assert baseline["learner_ready"] is True

    future = _lesson(
        21,
        positive=True,
        maturity_session=_session(
            date.fromisoformat(current["decision_session"]), 1
        ),
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayLearnerError,
        match="Future outcome",
    ):
        _predict(current, [*lessons, future])

    own_feature = current
    own_body = {
        "schema_version": lessons[0]["schema_version"],
        "contract_version": lessons[0]["contract_version"],
        "contract_sha256": lessons[0]["contract_sha256"],
        "accession_number": own_feature["accession_number"],
        "decision_session": own_feature["decision_session"],
        "maturity_session": own_feature["decision_session"],
        "feature_row": own_feature,
        "feature_row_sha256": own_feature["feature_row_sha256"],
        "trainable": True,
        "binary_cash_win_target": 1,
        "active_log_edge_10bps_hex": 0.01.hex(),
        "label_evidence_sha256": "e" * 64,
    }
    own = {
        **own_body,
        "lesson_row_sha256": canonical_sha256(own_body),
    }
    with pytest.raises(
        SecGemmaOnlineRiskOverlayLearnerError,
        match="maturity must follow",
    ):
        _predict(current, [*lessons, own])


def test_same_session_maturity_from_an_earlier_filing_is_admitted() -> None:
    current = _feature(30)
    lessons = _ready_lessons()
    latest = copy.deepcopy(lessons[-1])
    latest["maturity_session"] = current["decision_session"]
    _rehash_lesson(latest)
    lessons[-1] = latest

    result = _predict(current, lessons)

    assert result["learner_ready"] is True
    assert (
        result["fit_audit"]["fit_state"]["fit_metadata"][
            "maximum_training_label_maturity_session"
        ]
        == current["decision_session"]
    )


def test_refit_and_prediction_are_order_invariant_and_byte_deterministic() -> None:
    current = _feature(30)
    lessons = _ready_lessons()
    first = _predict(current, lessons)
    repeated = _predict(current, copy.deepcopy(lessons))
    shuffled = copy.deepcopy(lessons)
    random.Random(417).shuffle(shuffled)
    reordered = _predict(current, shuffled)

    assert first == repeated == reordered
    assert first["prediction_row_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in first.items()
            if key != "prediction_row_sha256"
        }
    )
    fit = first["fit_audit"]
    assert fit["fit_audit_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in fit.items()
            if key != "fit_audit_sha256"
        }
    )
    assert fit["fit_state_sha256"] == fit["fit_state"]["state_sha256"]
    assert first["fitted_prediction_available"] is True
    assert {
        "accession_number",
        "decision_session",
        "acceptance_datetime",
        "prediction_available",
        "learner_ready",
        "raw_gate_pass",
        "prediction_row_sha256",
    }.issubset(first)


def test_same_session_filings_share_fit_and_new_maturity_expands_it() -> None:
    decision = "2002-07-01"
    first_current = _feature(30, decision_session=decision)
    second_current = _feature(31, decision_session=decision)
    lessons = _ready_lessons()

    first = _predict(first_current, lessons)
    second = _predict(second_current, list(reversed(lessons)))
    assert first["fit_audit"] == second["fit_audit"]

    extra = _lesson(20, positive=True)
    expanded = _predict(first_current, [*lessons, extra])
    assert expanded["fit_audit"]["training_row_count"] == (
        first["fit_audit"]["training_row_count"] + 1
    )
    assert (
        expanded["fit_audit"]["fit_state_sha256"]
        != first["fit_audit"]["fit_state_sha256"]
    )


def test_missing_acceptance_timestamp_is_preserved_without_inventing_order() -> None:
    current = _feature(30)
    current["acceptance_datetime"] = None
    _rehash_feature(current)

    result = _predict(current, _ready_lessons())

    assert result["acceptance_datetime"] is None
    assert result["prediction_available"] is True


def test_ambiguous_same_session_lesson_order_is_rejected() -> None:
    lesson_decision = "2000-02-01"
    maturity = "2000-03-01"
    first_feature = _feature(1, decision_session=lesson_decision)
    first_feature["acceptance_datetime"] = None
    _rehash_feature(first_feature)
    second_feature = _feature(2, decision_session=lesson_decision)
    first = build_online_overlay_lesson(
        feature_row=first_feature,
        expected_feature_row_sha256=first_feature["feature_row_sha256"],
        maturity_session=maturity,
        active_log_edge_10bps=0.01,
        label_evidence_sha256="a" * 64,
    )
    second = build_online_overlay_lesson(
        feature_row=second_feature,
        expected_feature_row_sha256=second_feature["feature_row_sha256"],
        maturity_session=maturity,
        active_log_edge_10bps=-0.01,
        label_evidence_sha256="b" * 64,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayLearnerError,
        match="ambiguous",
    ):
        _predict(_feature(30), [first, second])


def test_each_arm_has_a_distinct_bound_fit_identity() -> None:
    current = _feature(30)
    lessons = _ready_lessons()
    results = {
        arm: _predict(current, lessons, arm=arm) for arm in FEATURE_ARMS
    }
    candidates = {
        result["fit_audit"]["candidate_sha256"]
        for result in results.values()
    }
    model_hashes = {
        result["fit_audit"]["fit_state_sha256"]
        for result in results.values()
    }
    assert len(candidates) == len(FEATURE_ARMS)
    assert len(model_hashes) == len(FEATURE_ARMS)
    assert results["semantic"]["fit_audit"]["head_variant"] == "semantic"
    assert (
        results["no_filing_meaning"]["fit_audit"]["head_variant"]
        == "ablation"
    )
    assert (
        results["no_gemma_channel"]["fit_audit"]["head_variant"]
        == "ablation"
    )


@pytest.mark.parametrize(
    "probability, edge, expected",
    [
        (PROBABILITY_GATE, EXPECTED_EDGE_GATE, True),
        (
            math.nextafter(PROBABILITY_GATE, -math.inf),
            EXPECTED_EDGE_GATE,
            False,
        ),
        (
            PROBABILITY_GATE,
            math.nextafter(EXPECTED_EDGE_GATE, -math.inf),
            False,
        ),
        (math.nextafter(PROBABILITY_GATE, math.inf), 0.01, True),
    ],
)
def test_gate_thresholds_are_fixed_and_inclusive(
    probability: float,
    edge: float,
    expected: bool,
) -> None:
    audit = evaluate_online_overlay_gate(
        cash_win_probability=probability,
        expected_incremental_10bps_log_edge=edge,
    )
    assert audit["probability_gate_hex"] == PROBABILITY_GATE.hex()
    assert audit["expected_edge_gate_hex"] == EXPECTED_EDGE_GATE.hex()
    assert audit["overlay_gate_passed"] is expected
    assert audit["gate_audit_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in audit.items()
            if key != "gate_audit_sha256"
        }
    )


def test_rehashed_semantic_feature_tampering_is_rejected() -> None:
    row = _feature(1)
    changed = copy.deepcopy(row)
    changed["no_filing_meaning_values_hex"][len(MARKET_FEATURES)] = 0.5.hex()
    _rehash_feature(changed)
    with pytest.raises(
        SecGemmaOnlineRiskOverlayLearnerError,
        match="arm transform",
    ):
        validate_online_overlay_feature_row(
            changed,
            expected_feature_row_sha256=changed["feature_row_sha256"],
        )

    ordinary = copy.deepcopy(row)
    ordinary["form"] = "10-Q"
    with pytest.raises(
        SecGemmaOnlineRiskOverlayLearnerError,
        match="form control|checksum",
    ):
        validate_online_overlay_feature_row(
            ordinary,
            expected_feature_row_sha256=row["feature_row_sha256"],
        )


def test_audit_only_unavailable_lessons_never_enter_training_membership() -> None:
    current = _feature(30)
    lessons = _ready_lessons()
    unavailable = _lesson(20, positive=True, available=False)
    result = _predict(current, [*lessons, unavailable])

    fit = result["fit_audit"]
    assert fit["supplied_matured_lesson_count"] == len(lessons) + 1
    assert fit["audit_only_lesson_count"] == 1
    assert fit["training_row_count"] == len(lessons)
    assert unavailable["lesson_row_sha256"] not in {
        item["lesson_row_sha256"] for item in fit["training_members"]
    }


def test_prediction_signature_contains_no_raw_outcome_or_external_io_inputs() -> None:
    parameters = inspect.signature(build_online_overlay_prediction).parameters
    assert set(parameters) == {
        "feature_row",
        "expected_feature_row_sha256",
        "matured_lessons",
        "expected_lesson_row_sha256s",
        "arm",
        "decision_session",
    }
    forbidden = {
        "outcome",
        "future",
        "filesystem",
        "path",
        "network",
        "clock",
        "price",
        "return",
    }
    assert not forbidden.intersection(parameters)
    assert set(
        inspect.signature(build_online_overlay_fit_audit).parameters
    ) == {
        "matured_lessons",
        "expected_lesson_row_sha256s",
        "arm",
        "as_of_session",
    }
    assert set(
        inspect.signature(
            build_online_overlay_prediction_from_fit
        ).parameters
    ) == {
        "feature_row",
        "expected_feature_row_sha256",
        "fit_audit",
        "expected_fit_audit_sha256",
    }


def test_readiness_constants_and_feature_schema_are_exact() -> None:
    current = _feature(30)
    result = _predict(current, _ready_lessons())
    fit = result["fit_audit"]
    assert fit["minimum_training_rows"] == MINIMUM_TRAINING_ROWS == 20
    assert fit["minimum_rows_per_binary_class"] == MINIMUM_CLASS_ROWS == 4
    assert fit["feature_names"] == list(FEATURES)
    assert fit["feature_schema_sha256"] == canonical_sha256(list(FEATURES))
