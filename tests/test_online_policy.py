from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from agent_benchmark.online_policy import (
    CalibratedOnlineRiskOffEstimator,
    CounterfactualCostModel,
    PointInTimeSnapshot,
    PricePoint,
    RiskOffEstimatorConfig,
    create_matured_lesson,
)


UTC = timezone.utc


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _snapshot(
    timestamp: datetime,
    *,
    momentum: float = 0.0,
    volatility: float = 0.2,
    as_of: datetime | None = None,
) -> PointInTimeSnapshot:
    cutoff = as_of or timestamp
    return PointInTimeSnapshot(
        symbol="AAPL",
        decision_timestamp=_iso(timestamp),
        as_of_timestamp=_iso(cutoff),
        features={
            "momentum_20d": momentum,
            "volatility_20d": volatility,
        },
    )


def _lesson(
    index: int,
    stock_return: float,
    *,
    momentum: float = 0.0,
    volatility: float = 0.2,
    start: datetime = datetime(2023, 1, 1, tzinfo=UTC),
    costs: CounterfactualCostModel | None = None,
):
    decision = start + timedelta(days=index * 3)
    outcome = decision + timedelta(days=2)
    return create_matured_lesson(
        _snapshot(decision, momentum=momentum, volatility=volatility),
        [
            PricePoint(_iso(decision), 100.0),
            PricePoint(_iso(decision + timedelta(days=1)), 100.0 * (1.0 + stock_return / 2.0)),
            PricePoint(_iso(outcome), 100.0 * (1.0 + stock_return)),
        ],
        cost_model=costs or CounterfactualCostModel(transaction_cost_bps=0.0),
    )


def _query(timestamp: datetime, *, momentum: float = 0.0, volatility: float = 0.2):
    return _snapshot(timestamp, momentum=momentum, volatility=volatility)


def test_counterfactual_long_cash_short_labels_include_all_costs():
    decision = datetime(2024, 1, 1, tzinfo=UTC)
    costs = CounterfactualCostModel(
        transaction_cost_bps=10.0,
        long_turnover=0.0,
        cash_turnover=2.0,
        short_turnover=4.0,
        # 365.25 bps annually is exactly 10 bps over this ten-day horizon.
        annual_short_borrow_bps=365.25,
    )
    lesson = create_matured_lesson(
        _snapshot(decision, momentum=-0.05),
        [
            PricePoint(decision, 100.0),
            PricePoint(decision + timedelta(days=5), 94.0),
            PricePoint(decision + timedelta(days=10), 90.0),
        ],
        cost_model=costs,
    )

    assert lesson.outcomes.long.gross_return == pytest.approx(-0.10)
    assert lesson.outcomes.long.net_return == pytest.approx(-0.10)
    assert lesson.outcomes.cash.gross_return == 0.0
    assert lesson.outcomes.cash.transaction_cost == pytest.approx(0.002)
    assert lesson.outcomes.cash.net_return == pytest.approx(-0.002)
    assert lesson.outcomes.short.gross_return == pytest.approx(0.10)
    assert lesson.outcomes.short.transaction_cost == pytest.approx(0.004)
    assert lesson.outcomes.short.financing_cost == pytest.approx(0.001)
    assert lesson.outcomes.short.net_return == pytest.approx(0.095)
    assert lesson.outcomes.cash_active_return == pytest.approx(0.098)
    assert lesson.outcomes.short_active_return == pytest.approx(0.195)
    assert lesson.risk_off_label == 1
    assert lesson.knowledge_timestamp == _iso(decision + timedelta(days=10))

    bullish = _lesson(0, 0.10)
    assert bullish.risk_off_label == 0
    assert bullish.outcomes.cash_active_return == pytest.approx(-0.10)


def test_snapshot_and_lesson_keep_outcomes_out_of_point_in_time_features():
    decision = datetime(2024, 2, 1, 15, tzinfo=UTC)
    source_features = {"momentum_20d": 0.04, "known_event_flag": 1.0}
    snapshot = PointInTimeSnapshot(
        symbol="aapl",
        decision_timestamp=decision,
        as_of_timestamp=decision - timedelta(minutes=5),
        features=source_features,
        feature_timestamps={
            "momentum_20d": decision - timedelta(hours=1),
            "known_event_flag": decision - timedelta(days=1),
        },
    )
    source_features["momentum_20d"] = 999.0
    lesson = create_matured_lesson(
        snapshot,
        [
            PricePoint(decision, 100.0),
            PricePoint(decision + timedelta(days=20), 80.0),
        ],
    )

    assert snapshot.symbol == "AAPL"
    assert lesson.features == {"known_event_flag": 1.0, "momentum_20d": 0.04}
    assert set(lesson.features) == set(source_features)
    assert "entry_price" not in lesson.features
    assert "exit_price" not in lesson.features
    assert "cash_active_return" not in lesson.features
    assert lesson.to_dict()["snapshot"]["features"] == dict(lesson.features)


@pytest.mark.parametrize(
    "bad_feature",
    [
        "future_return_20d",
        "forward_drawdown",
        "lead_close",
        "outcome_5d",
        "target_cash_beats_long",
        "label_risk_off",
    ],
)
def test_snapshot_rejects_obviously_outcome_derived_feature_names(bad_feature):
    now = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="outcome-derived"):
        PointInTimeSnapshot(
            symbol="AAPL",
            decision_timestamp=now,
            as_of_timestamp=now,
            features={bad_feature: 1.0},
        )


def test_snapshot_rejects_feature_or_common_cutoff_after_decision_time():
    decision = datetime(2024, 1, 2, 16, tzinfo=UTC)
    with pytest.raises(ValueError, match="cannot be after decision"):
        PointInTimeSnapshot(
            symbol="AAPL",
            decision_timestamp=decision,
            as_of_timestamp=decision + timedelta(seconds=1),
            features={"momentum_20d": 0.1},
        )

    with pytest.raises(ValueError, match="was not available"):
        PointInTimeSnapshot(
            symbol="AAPL",
            decision_timestamp=decision,
            as_of_timestamp=decision,
            features={"momentum_20d": 0.1},
            feature_timestamps={"momentum_20d": decision + timedelta(seconds=1)},
        )


def test_lesson_rejects_predecision_and_nonchronological_price_paths():
    decision = datetime(2024, 3, 1, tzinfo=UTC)
    snapshot = _snapshot(decision)
    with pytest.raises(ValueError, match="entry price cannot"):
        create_matured_lesson(
            snapshot,
            [
                PricePoint(decision - timedelta(days=1), 100.0),
                PricePoint(decision + timedelta(days=1), 101.0),
            ],
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        create_matured_lesson(
            snapshot,
            [
                PricePoint(decision, 100.0),
                PricePoint(decision + timedelta(days=2), 101.0),
                PricePoint(decision + timedelta(days=1), 102.0),
            ],
        )


def test_historical_prediction_cannot_see_a_later_matured_lesson():
    early = _lesson(0, -0.08, momentum=-0.1)
    late = _lesson(10, 0.15, momentum=-0.1)
    estimator_with_future = CalibratedOnlineRiskOffEstimator(
        RiskOffEstimatorConfig(min_samples=1, max_neighbors=10)
    )
    estimator_with_future.update_many([early, late])

    estimator_at_the_time = CalibratedOnlineRiskOffEstimator(estimator_with_future.config)
    estimator_at_the_time.update(early)
    cutoff = datetime(2023, 1, 20, tzinfo=UTC)
    snapshot = _query(cutoff, momentum=-0.1)

    assert estimator_with_future.decision_support(snapshot).to_dict() == (
        estimator_at_the_time.decision_support(snapshot).to_dict()
    )
    assert estimator_with_future.decision_support(snapshot).eligible_lesson_count == 1


def test_incremental_updates_must_be_chronological_and_batch_is_atomic():
    first = _lesson(0, -0.05)
    second = _lesson(2, -0.06)
    retroactive = _lesson(1, -0.07)
    estimator = CalibratedOnlineRiskOffEstimator()

    with pytest.raises(ValueError, match="knowledge-timestamp order"):
        estimator.update_many([second, retroactive])
    assert estimator.lessons == ()

    estimator.update_many([first, second])
    with pytest.raises(ValueError, match="knowledge-timestamp order"):
        estimator.update(retroactive)
    assert estimator.lessons == (first, second)
    with pytest.raises(ValueError, match="duplicate lesson_id"):
        estimator.update(second)


def test_empirical_bayes_payload_is_bounded_and_deterministic():
    returns = [-0.08, -0.04, 0.03, -0.02, 0.05, -0.06, 0.01, -0.03, 0.04, -0.07]
    lessons = [
        _lesson(
            index,
            stock_return,
            momentum=-0.04 + index * 0.008,
            volatility=0.18 + index * 0.005,
        )
        for index, stock_return in enumerate(returns)
    ]
    config = RiskOffEstimatorConfig(
        max_neighbors=8,
        min_samples=4,
        prior_strength=2.0,
        min_confidence=0.0,
    )
    first = CalibratedOnlineRiskOffEstimator(config)
    second = CalibratedOnlineRiskOffEstimator(config)
    first.update_many(lessons)
    second.update_many(lessons)
    snapshot = _query(datetime(2023, 3, 1, tzinfo=UTC), momentum=-0.015, volatility=0.205)

    payload = first.decision_support(snapshot)
    assert payload.to_dict() == second.decision_support(snapshot).to_dict()
    assert first.to_dict() == second.to_dict()
    assert 0.0 <= payload.cash_outperformance_probability <= 1.0
    assert 0.0 <= payload.loss_probability <= 1.0
    assert payload.loss_probability == pytest.approx(
        1.0 - payload.cash_outperformance_probability
    )
    assert 0.0 <= payload.confidence <= 1.0
    assert 0.0 <= payload.probability_lower_bound <= payload.cash_outperformance_probability
    assert payload.cash_outperformance_probability <= payload.probability_upper_bound <= 1.0
    assert payload.lower_bound <= payload.expected_active_return <= payload.upper_bound
    assert payload.sample_size == 8
    assert 0.0 < payload.effective_sample_size <= payload.sample_size
    assert payload.eligible_lesson_count == len(lessons)
    assert payload.recommended_action in {"BUY_ALL", "CASH_ALL", "HOLD"}


def test_recent_matured_pattern_can_displace_an_ancient_exact_match():
    start = datetime(2020, 1, 1, tzinfo=UTC)
    ancient = _lesson(0, -0.10, momentum=0.0, start=start)
    recent_start = datetime(2023, 12, 1, tzinfo=UTC)
    recent = _lesson(0, 0.10, momentum=0.01, start=recent_start)
    estimator = CalibratedOnlineRiskOffEstimator(
        RiskOffEstimatorConfig(
            max_neighbors=1,
            min_samples=1,
            prior_strength=1.0,
            min_confidence=0.0,
            recency_half_life_days=30.0,
        )
    )
    estimator.update_many([ancient, recent])

    payload = estimator.decision_support(
        _query(datetime(2023, 12, 10, tzinfo=UTC), momentum=0.0)
    )

    assert payload.sample_size == 1
    assert payload.expected_active_return < 0.0
    assert payload.cash_outperformance_probability < 0.5


def test_neighbor_separation_prevents_overlapping_daily_cases_from_inflating_sample_size():
    estimator = CalibratedOnlineRiskOffEstimator(
        RiskOffEstimatorConfig(
            max_neighbors=10,
            min_samples=1,
            min_confidence=0.0,
            min_neighbor_separation_days=7,
        )
    )
    estimator.update_many([_lesson(index, -0.03) for index in range(10)])

    payload = estimator.decision_support(_query(datetime(2023, 3, 1, tzinfo=UTC)))

    # Source decisions are three days apart, so the local sample cannot count
    # all overlapping horizons as ten independent observations.
    assert payload.eligible_lesson_count == 10
    assert payload.sample_size <= 4


def test_neighbor_selection_never_counts_overlapping_outcome_windows():
    start = datetime(2022, 1, 1, tzinfo=UTC)
    lessons = []
    for index in range(5):
        decision = start + timedelta(days=index * 5)
        lessons.append(
            create_matured_lesson(
                _snapshot(decision),
                [
                    PricePoint(decision, 100.0),
                    PricePoint(decision + timedelta(days=20), 95.0),
                ],
            )
        )
    estimator = CalibratedOnlineRiskOffEstimator(
        RiskOffEstimatorConfig(max_neighbors=10, min_samples=1, min_confidence=0.0)
    )
    estimator.update_many(lessons)

    payload = estimator.decision_support(_query(datetime(2022, 3, 1, tzinfo=UTC)))

    assert payload.eligible_lesson_count == 5
    assert payload.sample_size == 1


def test_delayed_price_availability_never_backdates_a_lesson():
    decision = datetime(2023, 1, 1, tzinfo=UTC)
    available = decision + timedelta(days=10)
    lesson = create_matured_lesson(
        _snapshot(decision),
        [
            PricePoint(decision, 100.0),
            PricePoint(decision + timedelta(days=2), 90.0),
        ],
        knowledge_timestamp=available,
    )
    estimator = CalibratedOnlineRiskOffEstimator(
        RiskOffEstimatorConfig(max_neighbors=2, min_samples=1, min_confidence=0.0)
    )
    estimator.update(lesson)

    assert lesson.outcome_timestamp != lesson.knowledge_timestamp
    assert estimator.decision_support(_query(decision + timedelta(days=5))).sample_size == 0
    assert estimator.decision_support(_query(available)).sample_size == 1


def test_recommendations_require_statistically_supported_active_edge():
    config = RiskOffEstimatorConfig(
        max_neighbors=20,
        min_samples=4,
        prior_strength=1.0,
        risk_off_probability=0.55,
        min_confidence=0.10,
        confidence_z=1.0,
    )
    query_time = datetime(2023, 4, 1, tzinfo=UTC)

    risk_off = CalibratedOnlineRiskOffEstimator(config)
    assert risk_off.decision_support(_query(query_time)).recommended_action == "HOLD"
    risk_off.update_many([_lesson(i, -0.06 - i * 0.002) for i in range(8)])
    risk_off_payload = risk_off.decision_support(_query(query_time))
    assert risk_off_payload.lower_bound > 0.0
    assert risk_off_payload.recommended_action == "CASH_ALL"

    stay_long = CalibratedOnlineRiskOffEstimator(config)
    stay_long.update_many([_lesson(i, 0.06 + i * 0.002) for i in range(8)])
    stay_long_payload = stay_long.decision_support(_query(query_time))
    assert stay_long_payload.upper_bound < 0.0
    assert stay_long_payload.recommended_action == "BUY_ALL"

    uncertain = CalibratedOnlineRiskOffEstimator(config)
    uncertain.update_many([_lesson(i, -0.10) for i in range(3)])
    assert uncertain.decision_support(_query(query_time)).recommended_action == "HOLD"


def test_state_round_trip_preserves_predictions_and_incremental_learning(tmp_path):
    config = RiskOffEstimatorConfig(
        max_neighbors=6,
        min_samples=3,
        prior_strength=1.5,
        min_confidence=0.0,
    )
    estimator = CalibratedOnlineRiskOffEstimator(config)
    estimator.update_many(
        [_lesson(i, value, momentum=-0.02 + i * 0.01) for i, value in enumerate([-0.05, 0.03, -0.02, 0.04])]
    )
    path = tmp_path / "nested" / "aapl-online-policy.json"
    estimator.save(path)
    restored = CalibratedOnlineRiskOffEstimator.load(path)

    assert restored.to_dict() == estimator.to_dict()
    first_bytes = path.read_bytes()
    restored.save(path)
    assert path.read_bytes() == first_bytes
    assert json.loads(path.read_text(encoding="utf-8"))["serialization_version"] == 1

    next_lesson = _lesson(5, -0.08, momentum=0.03)
    estimator.update(next_lesson)
    restored.update(next_lesson)
    snapshot = _query(datetime(2023, 3, 1, tzinfo=UTC), momentum=0.025)
    assert restored.to_dict() == estimator.to_dict()
    assert restored.decision_support(snapshot).to_dict() == estimator.decision_support(snapshot).to_dict()


def test_configuration_is_generic_and_contains_no_calendar_year_fields():
    payload = RiskOffEstimatorConfig().to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    assert "2025" not in encoded
    assert not any("year" in key for key in payload)
