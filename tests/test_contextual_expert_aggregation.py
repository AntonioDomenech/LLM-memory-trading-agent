from __future__ import annotations

import copy
import json
import math
import sys
from datetime import date, timedelta

import pytest

import agent_benchmark.contextual_expert_aggregation as aggregation_module
from agent_benchmark.contextual_expert_aggregation import (
    CAUSAL_ONLINE_MODE,
    FROZEN_CUTOFF_MODE,
    FULL_MODE,
    GLOBAL_ONLY_MODE,
    LIFETIME_ONLY_MODE,
    LIFETIME_SCALE,
    LESSON_MEMORY_START_ISO,
    LONG_TIE_TOLERANCE,
    MARKET_STATE_NAMES,
    ROUND_TRIP_LOG_FRICTION,
    SCALE_DISCOUNTS,
    SCALE_NAMES,
    STATE_PRIOR_STRENGTH,
    UNKNOWN_MARKET_STATE,
    ContextualExpertAggregator,
    MarketSession,
    derive_adjusted_open,
    resolve_cash_action,
)


START = date(2001, 1, 1)


def _session(
    index: int,
    *,
    adjusted_open: float = 100.0,
    spy: float = 100.0,
    qqq: float = 100.0,
    contextual: bool = False,
    weak: bool = False,
    opportunity: bool | None = None,
    start: date = START,
) -> MarketSession:
    return MarketSession(
        session_date=start + timedelta(days=index),
        aapl_open=adjusted_open,
        aapl_close=100.0,
        aapl_adj_close=100.0,
        spy_adj_close=spy,
        qqq_adj_close=qqq,
        contextual_signal=contextual,
        weak_trend_signal=weak,
        canonical_union_opportunity=opportunity,
    )


def _warm_model(
    *,
    first: float,
    middle: float,
    current: float,
) -> tuple[ContextualExpertAggregator, object]:
    model = ContextualExpertAggregator()
    result = None
    for index in range(21):
        value = 100.0
        if index == 0:
            value = first
        elif index == 10:
            value = middle
        elif index == 20:
            value = current
        result = model.process_session(_session(index, spy=value, qqq=value))
    assert result is not None
    return model, result


def test_adjusted_open_is_derived_and_inconsistent_supplied_value_is_rejected():
    expected = 80.0 * 50.0 / 100.0
    assert derive_adjusted_open(
        aapl_open=80.0, aapl_close=100.0, aapl_adj_close=50.0
    ) == pytest.approx(expected)
    session = MarketSession(
        session_date="2001-01-01",
        aapl_open=80.0,
        aapl_close=100.0,
        aapl_adj_close=50.0,
        aapl_adj_open=expected,
        spy_adj_close=100.0,
        qqq_adj_close=100.0,
    )
    assert session.aapl_adj_open == pytest.approx(expected)

    with pytest.raises(ValueError, match="inconsistent"):
        MarketSession(
            session_date="2001-01-01",
            aapl_open=80.0,
            aapl_close=100.0,
            aapl_adj_close=50.0,
            aapl_adj_open=expected + 0.01,
            spy_adj_close=100.0,
            qqq_adj_close=100.0,
        )


@pytest.mark.parametrize(
    ("first", "middle", "current", "expected"),
    [
        (100.0, 100.0, 100.0, "fast_off_slow_off"),
        (100.0, 100.0, 110.0, "fast_on_slow_on"),
        (120.0, 100.0, 110.0, "fast_on_slow_off"),
        (100.0, 120.0, 110.0, "fast_off_slow_on"),
    ],
)
def test_exact_spy_qqq_fast_slow_market_states(
    first, middle, current, expected
):
    model, result = _warm_model(
        first=first, middle=middle, current=current
    )
    assert result.market_state == expected
    assert expected in MARKET_STATE_NAMES
    assert len(model.to_dict()["state"]["market_history"]) == 21


def test_missing_lookback_uses_unknown_global_only_state():
    model = ContextualExpertAggregator()
    for index in range(20):
        result = model.process_session(_session(index))
        assert result.market_state == UNKNOWN_MARKET_STATE
        assert all(value == 0.0 for value in result.scale_state_pooling.values())


def test_uniform_cold_start_ties_long_and_two_signal_agreement_selects_cash():
    base, _ = _warm_model(first=100.0, middle=100.0, current=110.0)
    single = base.fork()
    agreed = base.fork()

    single_result = single.process_session(
        _session(21, contextual=True, opportunity=True, spy=111.0, qqq=111.0)
    )
    agreed_result = agreed.process_session(
        _session(
            21,
            contextual=True,
            weak=True,
            opportunity=True,
            spy=111.0,
            qqq=111.0,
        )
    )

    assert single_result.cash_score == pytest.approx(0.5)
    assert single_result.action == "LONG"
    assert agreed_result.cash_score == pytest.approx(0.75)
    assert agreed_result.action == "CASH"


def test_t_plus_two_lesson_updates_before_same_close_prediction():
    model = ContextualExpertAggregator()
    first = model.process_session(
        _session(0, contextual=True, opportunity=True, adjusted_open=100.0)
    )
    assert first.action == "LONG"
    assert model.admitted_lesson_count == 0

    middle = model.process_session(_session(1, adjusted_open=110.0))
    assert middle.matured_lessons == ()
    assert model.admitted_lesson_count == 0

    matured = model.process_session(
        _session(2, contextual=True, opportunity=True, adjusted_open=100.0)
    )
    assert len(matured.matured_lessons) == 1
    assert matured.matured_lessons[0].admitted is True
    assert matured.matured_lessons[0].net_cash_log_edge_10bps == pytest.approx(
        math.log(1.10) + math.log(0.999 / 1.001)
    )
    lesson = matured.matured_lessons[0]
    assert lesson.expert_cash_advice == {
        "always_long": False,
        "union_cash": True,
        "contextual_only": True,
        "weak_trend_only": False,
    }
    assert lesson.expert_rewards["always_long"] == 0.0
    assert lesson.expert_rewards["weak_trend_only"] == 0.0
    assert lesson.expert_rewards["union_cash"] == pytest.approx(
        lesson.net_cash_log_edge_10bps
    )
    assert lesson.expert_rewards["contextual_only"] == pytest.approx(
        lesson.net_cash_log_edge_10bps
    )
    assert lesson.reward_range == pytest.approx(
        abs(lesson.net_cash_log_edge_10bps)
    )
    assert model.admitted_lesson_count == 1
    # The newly matured positive CASH lesson is available before this close's
    # otherwise tied contextual-only opportunity is scored.
    assert matured.cash_score > 0.5
    assert matured.action == "CASH"


def test_online_and_frozen_forks_share_checkpoint_but_admit_different_lessons():
    base = ContextualExpertAggregator()
    base.process_session(
        _session(0, contextual=True, opportunity=True, adjusted_open=100.0)
    )
    online = base.fork(learning_mode=CAUSAL_ONLINE_MODE)
    frozen = base.fork(
        learning_mode=FROZEN_CUTOFF_MODE,
        frozen_cutoff=START,
    )
    assert online.state_dict() == frozen.state_dict()

    for model in (online, frozen):
        model.process_session(_session(1, adjusted_open=110.0))
    online_result = online.process_session(_session(2, adjusted_open=100.0))
    frozen_result = frozen.process_session(_session(2, adjusted_open=100.0))

    assert online_result.matured_lessons[0].admitted is True
    assert frozen_result.matured_lessons[0].admitted is False
    assert online.matured_lesson_count == frozen.matured_lesson_count == 1
    assert online.admitted_lesson_count == 1
    assert frozen.admitted_lesson_count == 0


def test_pre_2000_signal_can_mature_but_cannot_enter_learning_state():
    model = ContextualExpertAggregator()
    start = date(1999, 12, 20)
    model.process_session(
        _session(0, start=start, contextual=True, opportunity=True)
    )
    model.process_session(_session(1, start=start, adjusted_open=110.0))
    result = model.process_session(_session(2, start=start, adjusted_open=100.0))

    assert len(result.matured_lessons) == 1
    assert result.matured_lessons[0].admitted is False
    assert model.matured_lesson_count == 1
    assert model.admitted_lesson_count == 0


def test_full_global_and_lifetime_ablations_share_exact_learning_state():
    base, _ = _warm_model(first=100.0, middle=100.0, current=110.0)
    models = {
        mode: base.fork(ablation_mode=mode)
        for mode in (FULL_MODE, GLOBAL_ONLY_MODE, LIFETIME_ONLY_MODE)
    }
    for model in models.values():
        model.process_session(
            _session(21, contextual=True, opportunity=True, spy=111.0, qqq=111.0)
        )
        model.process_session(
            _session(22, adjusted_open=110.0, spy=112.0, qqq=112.0)
        )
    results = {
        mode: model.process_session(
            _session(
                23,
                adjusted_open=100.0,
                contextual=True,
                opportunity=True,
                spy=113.0,
                qqq=113.0,
            )
        )
        for mode, model in models.items()
    }

    states = [model.state_dict() for model in models.values()]
    assert states[0] == states[1] == states[2]
    assert tuple(results[FULL_MODE].scale_expert_weights) == SCALE_NAMES
    assert tuple(results[GLOBAL_ONLY_MODE].scale_expert_weights) == SCALE_NAMES
    assert tuple(results[LIFETIME_ONLY_MODE].scale_expert_weights) == (
        LIFETIME_SCALE,
    )
    assert all(
        value == 0.0
        for value in results[GLOBAL_ONLY_MODE].scale_state_pooling.values()
    )
    assert any(
        value > 0.0
        for value in results[FULL_MODE].scale_state_pooling.values()
    )
    for result in results.values():
        assert sum(result.aggregate_expert_weights.values()) == pytest.approx(1.0)
        assert 0.0 <= result.cash_score <= 1.0


def test_round_trip_resume_with_pending_lesson_is_byte_deterministic(tmp_path):
    model, _ = _warm_model(first=100.0, middle=100.0, current=110.0)
    model.process_session(
        _session(21, contextual=True, opportunity=True, spy=111.0, qqq=111.0)
    )
    model.process_session(
        _session(22, adjusted_open=110.0, spy=112.0, qqq=112.0)
    )
    payload = json.loads(json.dumps(model.to_dict(), sort_keys=True))
    restored = ContextualExpertAggregator.from_dict(payload)
    uninterrupted = model.fork()

    next_session = _session(
        23,
        adjusted_open=100.0,
        contextual=True,
        opportunity=True,
        spy=113.0,
        qqq=113.0,
    )
    assert restored.process_session(next_session).to_dict() == uninterrupted.process_session(
        next_session
    ).to_dict()
    assert restored.to_dict() == uninterrupted.to_dict()

    path = tmp_path / "nested" / "aggregation.json"
    restored.save(path)
    first_bytes = path.read_bytes()
    loaded = ContextualExpertAggregator.load(path)
    loaded.save(path)
    assert path.read_bytes() == first_bytes
    assert loaded.to_dict() == restored.to_dict()


def test_serialization_rejects_contract_drift_and_sessions_are_strictly_ordered():
    model = ContextualExpertAggregator()
    model.process_session(_session(0))
    bad_constant = copy.deepcopy(model.to_dict())
    bad_constant["constants"]["fast_lookback"] = 9
    with pytest.raises(ValueError, match="constants"):
        ContextualExpertAggregator.from_dict(bad_constant)

    extra = copy.deepcopy(model.to_dict())
    extra["state"]["unexpected"] = True
    with pytest.raises(ValueError, match="missing or unexpected"):
        ContextualExpertAggregator.from_dict(extra)

    with pytest.raises(ValueError, match="strictly increasing"):
        model.process_session(_session(0))


def test_restore_rejects_impossible_pending_advice_and_tail_dates():
    model = ContextualExpertAggregator()
    model.process_session(
        _session(0, contextual=True, opportunity=True)
    )

    no_contributor = copy.deepcopy(model.to_dict())
    pending = no_contributor["state"]["pending_lessons"][0]
    pending["expert_cash_advice"]["contextual_only"] = False
    with pytest.raises(ValueError, match="contributing fixed expert"):
        ContextualExpertAggregator.from_dict(no_contributor)

    future_signal = copy.deepcopy(model.to_dict())
    future_signal["state"]["pending_lessons"][0]["signal_date"] = "2001-01-02"
    with pytest.raises(ValueError, match="absent from the market-history tail"):
        ContextualExpertAggregator.from_dict(future_signal)

    wrong_distance = copy.deepcopy(model.to_dict())
    wrong_distance["state"]["pending_lessons"][0]["sessions_until_maturity"] = 1
    wrong_distance["state"]["pending_lessons"][0]["entry_adjusted_open"] = 100.0
    with pytest.raises(ValueError, match="replay tail"):
        ContextualExpertAggregator.from_dict(wrong_distance)


def test_future_sessions_cannot_change_an_already_emitted_decision():
    first, _ = _warm_model(first=100.0, middle=100.0, current=110.0)
    second = first.fork()
    candidate = _session(
        21,
        contextual=True,
        opportunity=True,
        spy=111.0,
        qqq=111.0,
    )
    first_decision = first.process_session(candidate).to_dict()
    second_decision_object = second.process_session(candidate)
    second_decision = second_decision_object.to_dict()
    assert first_decision == second_decision

    # Only the second replay sees later rows.  The immutable decision already
    # emitted from the identical prefix remains byte-for-byte unchanged.
    second.process_session(
        _session(22, adjusted_open=80.0, spy=90.0, qqq=90.0)
    )
    second.process_session(
        _session(23, adjusted_open=120.0, spy=130.0, qqq=130.0)
    )
    assert second_decision_object.to_dict() == second_decision
    assert first_decision["session_date"] == "2001-01-22"


def test_opportunity_contract_requires_a_contributing_fixed_expert():
    with pytest.raises(ValueError, match="contributing expert"):
        _session(0, opportunity=True)

    # A runner may explicitly suppress a raw signal under the shared canonical
    # cooldown; the model then remains LONG and creates no pending lesson.
    model = ContextualExpertAggregator()
    result = model.process_session(
        _session(0, contextual=True, opportunity=False)
    )
    assert result.opportunity is False
    assert result.action == "LONG"
    assert model.pending_lessons == ()


def test_exact_discount_g_v_n_eta_softmax_rho_and_scale_average_equations():
    model, _ = _warm_model(first=100.0, middle=100.0, current=110.0)
    model.process_session(
        _session(21, contextual=True, opportunity=True, spy=111.0, qqq=111.0)
    )
    model.process_session(
        _session(22, adjusted_open=110.0, spy=112.0, qqq=112.0)
    )
    first_maturity = model.process_session(
        _session(23, adjusted_open=100.0, spy=113.0, qqq=113.0)
    )
    y1 = math.log(110.0 / 100.0) + ROUND_TRIP_LOG_FRICTION
    assert first_maturity.matured_lessons[0].net_cash_log_edge_10bps == pytest.approx(y1)

    model.process_session(
        _session(24, weak=True, opportunity=True, spy=114.0, qqq=114.0)
    )
    model.process_session(
        _session(25, adjusted_open=90.0, spy=115.0, qqq=115.0)
    )
    result = model.process_session(
        _session(
            26,
            adjusted_open=100.0,
            contextual=True,
            weak=True,
            opportunity=True,
            spy=116.0,
            qqq=116.0,
        )
    )
    y2 = math.log(90.0 / 100.0) + ROUND_TRIP_LOG_FRICTION
    state = model.state_dict()
    signal_state = result.market_state
    expected_scale_weights = {}

    for scale in SCALE_NAMES:
        discount = SCALE_DISCOUNTS[scale]
        expected_n = discount + 1.0
        expected_v = discount * y1**2 + y2**2
        expected_g = {
            "always_long": 0.0,
            "union_cash": discount * y1 + y2,
            "contextual_only": discount * y1,
            "weak_trend_only": y2,
        }
        global_pool = state["global_pools"][scale]
        state_pool = state["state_pools"][scale][signal_state]
        assert global_pool["effective_count"] == pytest.approx(expected_n)
        assert global_pool["range_energy"] == pytest.approx(expected_v)
        assert global_pool["cumulative_rewards"] == pytest.approx(expected_g)
        assert state_pool["effective_count"] == pytest.approx(
            global_pool["effective_count"]
        )
        assert state_pool["range_energy"] == pytest.approx(
            global_pool["range_energy"]
        )
        assert state_pool["cumulative_rewards"] == pytest.approx(
            global_pool["cumulative_rewards"]
        )

        eta = math.sqrt(2.0 * math.log(4.0) / expected_v)
        maximum = max(expected_g.values())
        unnormalized = {
            name: math.exp(eta * (value - maximum))
            for name, value in expected_g.items()
        }
        denominator = sum(unnormalized.values())
        expected_weights = {
            name: value / denominator for name, value in unnormalized.items()
        }
        expected_scale_weights[scale] = expected_weights
        assert result.scale_expert_weights[scale] == pytest.approx(expected_weights)
        assert result.scale_state_pooling[scale] == pytest.approx(
            expected_n / (expected_n + STATE_PRIOR_STRENGTH)
        )

    expected_average = {
        name: sum(
            expected_scale_weights[scale][name] for scale in SCALE_NAMES
        )
        / len(SCALE_NAMES)
        for name in expected_scale_weights[SCALE_NAMES[0]]
    }
    assert result.aggregate_expert_weights == pytest.approx(expected_average)
    expected_cash_score = sum(
        expected_average[name]
        for name in ("union_cash", "contextual_only", "weak_trend_only")
    )
    assert result.cash_score == pytest.approx(expected_cash_score)


def test_unknown_state_is_exactly_global_only_for_every_scale():
    base = ContextualExpertAggregator()
    base.process_session(
        _session(0, contextual=True, opportunity=True, adjusted_open=100.0)
    )
    full = base.fork(ablation_mode=FULL_MODE)
    global_only = base.fork(ablation_mode=GLOBAL_ONLY_MODE)
    for model in (full, global_only):
        model.process_session(_session(1, adjusted_open=110.0))
    full_result = full.process_session(
        _session(2, adjusted_open=100.0, contextual=True, opportunity=True)
    )
    global_result = global_only.process_session(
        _session(2, adjusted_open=100.0, contextual=True, opportunity=True)
    )

    assert full_result.market_state == UNKNOWN_MARKET_STATE
    assert full_result.scale_state_pooling == global_result.scale_state_pooling
    assert all(value == 0.0 for value in full_result.scale_state_pooling.values())
    assert full_result.scale_expert_weights == global_result.scale_expert_weights
    assert full_result.aggregate_expert_weights == global_result.aggregate_expert_weights
    assert full_result.cash_score == global_result.cash_score
    assert full_result.action == global_result.action


def test_frozen_cutoff_is_inclusive_at_maturity_close():
    base = ContextualExpertAggregator()
    base.process_session(
        _session(0, contextual=True, opportunity=True, adjusted_open=100.0)
    )
    inclusive = base.fork(
        learning_mode=FROZEN_CUTOFF_MODE,
        frozen_cutoff=START + timedelta(days=2),
    )
    exclusive = base.fork(
        learning_mode=FROZEN_CUTOFF_MODE,
        frozen_cutoff=START + timedelta(days=1),
    )
    for model in (inclusive, exclusive):
        model.process_session(_session(1, adjusted_open=110.0))
    included = inclusive.process_session(_session(2, adjusted_open=100.0))
    excluded = exclusive.process_session(_session(2, adjusted_open=100.0))

    assert included.matured_lessons[0].admitted is True
    assert excluded.matured_lessons[0].admitted is False
    assert inclusive.admitted_lesson_count == 1
    assert exclusive.admitted_lesson_count == 0


def test_lesson_memory_start_is_frozen_and_2000_signal_boundary_is_inclusive():
    assert LESSON_MEMORY_START_ISO == "2000-01-01"
    model = ContextualExpertAggregator()
    assert model.to_dict()["constants"]["lesson_memory_start"] == "2000-01-01"
    with pytest.raises(TypeError):
        ContextualExpertAggregator(lesson_memory_start="2000-01-02")

    before = ContextualExpertAggregator()
    before_start = date(1999, 12, 31)
    before.process_session(
        _session(0, start=before_start, contextual=True, opportunity=True)
    )
    before.process_session(_session(1, start=before_start, adjusted_open=110.0))
    before_result = before.process_session(
        _session(2, start=before_start, adjusted_open=100.0)
    )
    assert before_result.matured_lessons[0].admitted is False

    boundary = ContextualExpertAggregator()
    on_start = date(2000, 1, 1)
    boundary.process_session(
        _session(0, start=on_start, contextual=True, opportunity=True)
    )
    boundary.process_session(_session(1, start=on_start, adjusted_open=110.0))
    boundary_result = boundary.process_session(
        _session(2, start=on_start, adjusted_open=100.0)
    )
    assert boundary_result.matured_lessons[0].admitted is True


def test_exact_cash_threshold_tolerance_and_invalid_cash_scores():
    threshold = 0.5 + LONG_TIE_TOLERANCE
    assert resolve_cash_action(threshold, opportunity=True) == "LONG"
    assert (
        resolve_cash_action(math.nextafter(threshold, math.inf), opportunity=True)
        == "CASH"
    )
    assert resolve_cash_action(1.0, opportunity=False) == "LONG"
    for invalid in (math.nan, math.inf, -math.inf, math.nextafter(0.0, 1.0)):
        with pytest.raises(ValueError, match="cash score"):
            resolve_cash_action(invalid, opportunity=True)


def test_serialized_runtime_is_validated_before_any_override():
    payload = ContextualExpertAggregator().to_dict()
    invalid_mode = copy.deepcopy(payload)
    invalid_mode["runtime"]["learning_mode"] = "invalid"
    with pytest.raises(ValueError, match="learning_mode"):
        ContextualExpertAggregator.from_dict(
            invalid_mode, learning_mode=CAUSAL_ONLINE_MODE
        )

    inconsistent_cutoff = copy.deepcopy(payload)
    inconsistent_cutoff["runtime"]["frozen_cutoff"] = "2001-01-01"
    with pytest.raises(ValueError, match="must not receive"):
        ContextualExpertAggregator.from_dict(
            inconsistent_cutoff, learning_mode=CAUSAL_ONLINE_MODE
        )

    invalid_ablation = copy.deepcopy(payload)
    invalid_ablation["runtime"]["ablation_mode"] = "invalid"
    with pytest.raises(ValueError, match="ablation_mode"):
        ContextualExpertAggregator.from_dict(
            invalid_ablation, ablation_mode=FULL_MODE
        )


def _one_admitted_checkpoint() -> dict:
    model = ContextualExpertAggregator()
    model.process_session(
        _session(0, contextual=True, opportunity=True, adjusted_open=100.0)
    )
    model.process_session(_session(1, adjusted_open=110.0))
    model.process_session(_session(2, adjusted_open=100.0))
    return model.to_dict()


def test_restore_reconciles_global_and_state_counts_with_admitted_lessons():
    payload = _one_admitted_checkpoint()
    restored = ContextualExpertAggregator.from_dict(payload)
    assert restored.admitted_lesson_count == 1
    for scale in SCALE_NAMES:
        assert payload["state"]["global_pools"][scale]["effective_count"] == pytest.approx(1.0)

    bad_global = copy.deepcopy(payload)
    bad_global["state"]["global_pools"][SCALE_NAMES[0]]["effective_count"] = 1.5
    with pytest.raises(ValueError, match="global effective_count"):
        ContextualExpertAggregator.from_dict(bad_global)

    bad_admitted = copy.deepcopy(payload)
    bad_admitted["state"]["admitted_lesson_count"] = 2
    bad_admitted["state"]["matured_lesson_count"] = 2
    with pytest.raises(ValueError, match="global effective_count"):
        ContextualExpertAggregator.from_dict(bad_admitted)

    known_payload = _warm_model(first=100.0, middle=100.0, current=110.0)[0]
    known_payload.process_session(
        _session(21, contextual=True, opportunity=True, spy=111.0, qqq=111.0)
    )
    known_payload.process_session(
        _session(22, adjusted_open=110.0, spy=112.0, qqq=112.0)
    )
    known_payload.process_session(
        _session(23, adjusted_open=100.0, spy=113.0, qqq=113.0)
    )
    bad_state = known_payload.to_dict()
    state_name = "fast_on_slow_on"
    bad_state["state"]["state_pools"][SCALE_NAMES[0]][state_name][
        "effective_count"
    ] = 2.0
    with pytest.raises(ValueError, match="state effective_count"):
        ContextualExpertAggregator.from_dict(bad_state)


def test_restore_rejects_subnormal_nonfinite_and_impossible_pool_values():
    payload = _one_admitted_checkpoint()
    subnormal = math.nextafter(0.0, 1.0)
    bad_reward = copy.deepcopy(payload)
    bad_reward["state"]["global_pools"][SCALE_NAMES[0]][
        "cumulative_rewards"
    ]["always_long"] = subnormal
    with pytest.raises(ValueError, match="normal"):
        ContextualExpertAggregator.from_dict(bad_reward)

    bad_energy = copy.deepcopy(payload)
    bad_energy["state"]["global_pools"][SCALE_NAMES[0]]["range_energy"] = math.inf
    with pytest.raises(ValueError, match="range_energy"):
        ContextualExpertAggregator.from_dict(bad_energy)

    bad_cauchy = copy.deepcopy(payload)
    bad_cauchy["state"]["global_pools"][SCALE_NAMES[0]][
        "cumulative_rewards"
    ]["always_long"] = 1.0
    with pytest.raises(ValueError, match="Cauchy"):
        ContextualExpertAggregator.from_dict(bad_cauchy)

    invalid_eta_pool = aggregation_module._PoolState.empty()
    invalid_eta_pool.range_energy = subnormal
    with pytest.raises(ValueError, match="range_energy"):
        invalid_eta_pool.weights()

    invalid_logit_pool = aggregation_module._PoolState.empty()
    invalid_logit_pool.range_energy = 1.0
    invalid_logit_pool.cumulative_rewards["always_long"] = math.nan
    with pytest.raises(ValueError, match="pool reward"):
        invalid_logit_pool.weights()


def test_pending_signal_state_and_context_are_recomputed_on_restore():
    model, _ = _warm_model(first=100.0, middle=100.0, current=110.0)
    model.process_session(
        _session(21, contextual=True, opportunity=True, spy=111.0, qqq=111.0)
    )
    payload = model.to_dict()
    pending = payload["state"]["pending_lessons"][0]
    assert pending["market_state"] == "fast_on_slow_on"
    assert pending["market_context"] is not None

    bad_state = copy.deepcopy(payload)
    bad_state["state"]["pending_lessons"][0][
        "market_state"
    ] = "fast_off_slow_off"
    with pytest.raises(ValueError, match="signal-time context"):
        ContextualExpertAggregator.from_dict(bad_state)

    bad_context = copy.deepcopy(payload)
    bad_context["state"]["pending_lessons"][0]["market_context"][
        "spy_adj_close_10"
    ] = 101.0
    with pytest.raises(ValueError, match="market-history tail"):
        ContextualExpertAggregator.from_dict(bad_context)

    bad_eligibility = copy.deepcopy(payload)
    bad_eligibility["state"]["pending_lessons"][0]["learning_eligible"] = False
    with pytest.raises(ValueError, match="eligibility"):
        ContextualExpertAggregator.from_dict(bad_eligibility)

    bad_entry = copy.deepcopy(payload)
    bad_entry["state"]["pending_lessons"][0]["sessions_until_maturity"] = 1
    bad_entry["state"]["pending_lessons"][0]["entry_adjusted_open"] = math.nan
    with pytest.raises(ValueError, match="entry_adjusted_open"):
        ContextualExpertAggregator.from_dict(bad_entry)


def test_pending_restore_rejects_deletion_invention_contributor_and_entry_tamper():
    signaled = ContextualExpertAggregator()
    signaled.process_session(
        _session(0, contextual=True, opportunity=True, adjusted_open=100.0)
    )
    new_pending = signaled.to_dict()

    deleted = copy.deepcopy(new_pending)
    deleted["state"]["pending_lessons"] = []
    with pytest.raises(ValueError, match="bounded session replay tail"):
        ContextualExpertAggregator.from_dict(deleted)

    quiet = ContextualExpertAggregator()
    quiet.process_session(_session(0, adjusted_open=100.0))
    invented = quiet.to_dict()
    forged = copy.deepcopy(new_pending["state"]["pending_lessons"][0])
    invented["state"]["pending_lessons"].append(forged)
    with pytest.raises(ValueError, match="bounded session replay tail"):
        ContextualExpertAggregator.from_dict(invented)

    contributor_flip = copy.deepcopy(new_pending)
    advice = contributor_flip["state"]["pending_lessons"][0][
        "expert_cash_advice"
    ]
    advice["contextual_only"] = False
    advice["weak_trend_only"] = True
    with pytest.raises(ValueError, match="bounded session replay tail"):
        ContextualExpertAggregator.from_dict(contributor_flip)

    signaled.process_session(_session(1, adjusted_open=110.0))
    observed_entry = signaled.to_dict()
    assert observed_entry["state"]["pending_lessons"][0][
        "entry_adjusted_open"
    ] == 110.0
    changed_entry = copy.deepcopy(observed_entry)
    changed_entry["state"]["pending_lessons"][0]["entry_adjusted_open"] = 111.0
    with pytest.raises(ValueError, match="bounded session replay tail"):
        ContextualExpertAggregator.from_dict(changed_entry)

    changed_tail_opportunity = copy.deepcopy(new_pending)
    changed_tail_opportunity["state"]["market_history"][-1][
        "canonical_union_opportunity"
    ] = False
    with pytest.raises(ValueError, match="bounded session replay tail"):
        ContextualExpertAggregator.from_dict(changed_tail_opportunity)


def test_pending_advice_property_cannot_mutate_internal_model_state():
    model = ContextualExpertAggregator()
    model.process_session(_session(0, contextual=True, opportunity=True))
    before = model.to_dict()
    exposed = model.pending_lessons[0].expert_cash_advice

    with pytest.raises(TypeError):
        exposed["contextual_only"] = False  # type: ignore[index]

    assert model.to_dict() == before
    assert model.pending_lessons[0].expert_cash_advice["contextual_only"] is True


def test_pending_and_full_session_replay_tail_schemas_are_strict():
    model = ContextualExpertAggregator()
    model.process_session(_session(0, contextual=True, opportunity=True))

    extra_tail_field = copy.deepcopy(model.to_dict())
    extra_tail_field["state"]["market_history"][-1]["unexpected"] = True
    with pytest.raises(ValueError, match="missing or unexpected"):
        ContextualExpertAggregator.from_dict(extra_tail_field)

    missing_tail_field = copy.deepcopy(model.to_dict())
    del missing_tail_field["state"]["market_history"][-1]["aapl_adj_open"]
    with pytest.raises(ValueError, match="missing or unexpected"):
        ContextualExpertAggregator.from_dict(missing_tail_field)

    extra_pending_field = copy.deepcopy(model.to_dict())
    extra_pending_field["state"]["pending_lessons"][0]["unexpected"] = True
    with pytest.raises(ValueError, match="missing or unexpected"):
        ContextualExpertAggregator.from_dict(extra_pending_field)

    truncated_tail = copy.deepcopy(model.to_dict())
    truncated_tail["state"]["market_history"] = []
    with pytest.raises(ValueError, match="market history|market-history"):
        ContextualExpertAggregator.from_dict(truncated_tail)

    changed_processed_count = copy.deepcopy(model.to_dict())
    changed_processed_count["state"]["processed_session_count"] = 2
    with pytest.raises(ValueError, match="market-history length"):
        ContextualExpertAggregator.from_dict(changed_processed_count)


@pytest.mark.parametrize(
    "mutation",
    [
        "reward_nan",
        "energy_nan",
        "count_nan",
        "reward_without_energy",
        "extra_reward",
    ],
)
def test_uniform_weight_fast_path_validates_every_g_v_n_component(mutation: str):
    pool = aggregation_module._PoolState.empty()
    if mutation == "reward_nan":
        pool.cumulative_rewards["always_long"] = math.nan
    elif mutation == "energy_nan":
        pool.range_energy = math.nan
    elif mutation == "count_nan":
        pool.effective_count = math.nan
    elif mutation == "reward_without_energy":
        pool.cumulative_rewards["always_long"] = 1e-6
    elif mutation == "extra_reward":
        pool.cumulative_rewards["unexpected"] = 0.0
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises((TypeError, ValueError)):
        pool.weights()


def test_derived_adjusted_open_rejects_positive_subnormal_result():
    with pytest.raises(ValueError, match="derived adjusted open.*normal"):
        derive_adjusted_open(
            aapl_open=sys.float_info.min,
            aapl_close=1.0,
            aapl_adj_close=0.5,
        )
    with pytest.raises(ValueError, match="derived adjusted open.*normal"):
        MarketSession(
            session_date="2001-01-01",
            aapl_open=sys.float_info.min,
            aapl_close=1.0,
            aapl_adj_close=0.5,
            spy_adj_close=1.0,
            qqq_adj_close=1.0,
        )


@pytest.mark.parametrize("pool_kind", ["global", "state"])
@pytest.mark.parametrize("component", ["count", "energy", "reward"])
def test_cold_checkpoint_requires_exact_zero_for_every_pool_component(
    pool_kind: str, component: str
):
    payload = ContextualExpertAggregator().to_dict()
    scale = SCALE_NAMES[0]
    if pool_kind == "global":
        pool = payload["state"]["global_pools"][scale]
    else:
        pool = payload["state"]["state_pools"][scale][MARKET_STATE_NAMES[0]]

    pool["effective_count"] = 1e-13
    if component in {"energy", "reward"}:
        pool["range_energy"] = 1e-13
    if component == "reward":
        pool["cumulative_rewards"]["union_cash"] = 1e-13

    with pytest.raises(ValueError, match="cold state"):
        ContextualExpertAggregator.from_dict(payload)


def test_active_global_count_rejects_even_one_ulp_of_checkpoint_drift():
    payload = _one_admitted_checkpoint()
    scale = SCALE_NAMES[0]
    original = payload["state"]["global_pools"][scale]["effective_count"]
    payload["state"]["global_pools"][scale]["effective_count"] = math.nextafter(
        original, math.inf
    )
    with pytest.raises(ValueError, match="global effective_count"):
        ContextualExpertAggregator.from_dict(payload)
