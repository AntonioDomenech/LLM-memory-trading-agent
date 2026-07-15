from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

import agent_benchmark.binary_regime_union_selector as selector


def _market_frame(
    periods: int = 420, *, start: str = "2000-01-03"
) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=periods)
    aapl = np.linspace(220.0, 100.0, periods)
    spy = np.linspace(500.0, 250.0, periods)
    qqq = np.linspace(400.0, 180.0, periods)
    return pd.DataFrame(
        {
            "aapl_open": aapl.copy(),
            "aapl_close": aapl,
            "aapl_adj_close": aapl,
            "spy_adj_close": spy,
            "qqq_adj_close": qqq,
        },
        index=index,
    )


def _set_intraday_return(frame: pd.DataFrame, position: int, value: float) -> None:
    close = float(frame.iloc[position]["aapl_close"])
    frame.iloc[position, frame.columns.get_loc("aapl_open")] = close / (1.0 + value)


def _negative_shadow_frame(
    lessons: int = 17,
) -> tuple[pd.DataFrame, list[int]]:
    """Isolated risk-off union opportunities whose cash labels are harmful."""

    positions = [130 + 5 * offset for offset in range(lessons)]
    frame = _market_frame(positions[-1] + 8)
    open_column = frame.columns.get_loc("aapl_open")
    close_column = frame.columns.get_loc("aapl_close")
    for offset, signal in enumerate(positions):
        intraday_return = 0.20 + 0.01 * offset
        signal_close = float(frame.iloc[signal, close_column])
        frame.iloc[signal, open_column] = signal_close / (1.0 + intraday_return)
        # Deliberately make t+1 another raw signal. The fixed expert cooldown
        # must consume it before the selector is applied.
        next_close = float(frame.iloc[signal + 1, close_column])
        frame.iloc[signal + 1, open_column] = next_close / (
            1.0 + intraday_return
        )
        # A high t+2 re-entry open makes the 10-bps cash edge negative.
        exit_close = float(frame.iloc[signal + 2, close_column])
        frame.iloc[signal + 2, open_column] = exit_close * 1.20
    return frame, positions


def _same_close_update_frame() -> tuple[pd.DataFrame, int]:
    """A union signal at t and another canonical candidate at its t+2 close."""

    frame = _market_frame(150)
    signal = 130
    open_column = frame.columns.get_loc("aapl_open")
    close_column = frame.columns.get_loc("aapl_close")
    frame.iloc[signal, open_column] = float(frame.iloc[signal, close_column]) / 1.20
    frame.iloc[signal + 1, open_column] = (
        float(frame.iloc[signal + 1, close_column]) * 1.20
    )
    frame.iloc[signal + 2, open_column] = (
        float(frame.iloc[signal + 2, close_column]) / 1.20
    )
    return frame, signal


def _forecast(
    frame: pd.DataFrame,
    *,
    learning_mode: str = selector.CAUSAL_ONLINE_MODE,
    frozen_cutoff: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    return selector.build_binary_regime_union_selector_forecast(
        frame,
        learning_mode=learning_mode,
        frozen_cutoff=frozen_cutoff,
    )


def test_frozen_constants_and_structural_defaults_are_exact():
    assert selector.LESSON_DISCOUNT == pytest.approx(0.995)
    assert selector.MIN_EFFECTIVE_LESSONS == pytest.approx(12.0)
    assert selector.POSITIVE_MEAN_THRESHOLD == pytest.approx(0.001)
    assert selector.NEGATIVE_MEAN_THRESHOLD == pytest.approx(-0.001)
    assert set(selector.REGIME_NAMES) == {
        selector.RISK_ON_REGIME,
        selector.NOT_RISK_ON_REGIME,
    }

    states = selector.initialize_regime_states()
    assert set(states) == set(selector.REGIME_NAMES)
    assert states[selector.RISK_ON_REGIME].cash_selected is False
    assert states[selector.NOT_RISK_ON_REGIME].cash_selected is True


def test_regime_features_use_exact_completed_twenty_session_returns():
    frame = _market_frame(50)
    position = 30
    spy = frame.columns.get_loc("spy_adj_close")
    qqq = frame.columns.get_loc("qqq_adj_close")
    frame.iloc[position - 20, spy] = 100.0
    frame.iloc[position, spy] = 110.0
    frame.iloc[position - 20, qqq] = 200.0
    frame.iloc[position, qqq] = 230.0

    features = selector.build_binary_regime_features(frame)
    row = features.iloc[position]

    assert row["spy_return_20"] == pytest.approx(0.10)
    assert row["qqq_return_20"] == pytest.approx(0.15)
    assert bool(row["risk_on"])
    assert set(selector.REGIME_FEATURE_COLUMNS).issubset(features.columns)
    assert tuple(features.attrs["regime_feature_order"]) == tuple(
        selector.REGIME_FEATURE_COLUMNS
    )


def test_zero_or_mixed_twenty_session_return_is_not_risk_on():
    frame = _market_frame(55)
    position = 30
    spy = frame.columns.get_loc("spy_adj_close")
    qqq = frame.columns.get_loc("qqq_adj_close")
    frame.iloc[position - 20, spy] = 100.0
    frame.iloc[position, spy] = 100.0
    frame.iloc[position - 20, qqq] = 100.0
    frame.iloc[position, qqq] = 120.0
    equality = selector.build_binary_regime_features(frame)
    assert equality.iloc[position]["spy_return_20"] == pytest.approx(0.0)
    assert not bool(equality.iloc[position]["risk_on"])

    frame.iloc[position, spy] = 110.0
    frame.iloc[position, qqq] = 90.0
    mixed = selector.build_binary_regime_features(frame)
    assert not bool(mixed.iloc[position]["risk_on"])


def test_future_market_mutation_cannot_change_an_earlier_regime():
    frame = _market_frame(70)
    decision = 35
    original = selector.build_binary_regime_features(frame)
    changed = frame.copy()
    changed.iloc[decision + 1 :, changed.columns.get_loc("spy_adj_close")] *= 4.0
    changed.iloc[decision + 1 :, changed.columns.get_loc("qqq_adj_close")] *= 0.25
    replay = selector.build_binary_regime_features(changed)

    pd.testing.assert_frame_equal(
        original.iloc[: decision + 1], replay.iloc[: decision + 1]
    )


def test_one_update_matches_discounted_equations_and_does_not_mutate_input():
    state = selector.initialize_regime_state(structural_default_cash=False)
    original = copy.deepcopy(state)
    updated = selector.update_ew_regime_state(
        state, 0.02, structural_default_cash=False
    )

    assert state == original
    assert updated.n_raw == 1
    assert updated.n_eff == pytest.approx(1.0)
    assert updated.weighted_label_sum == pytest.approx(0.02)
    assert updated.weighted_squared_label_sum == pytest.approx(0.02**2)
    # Readiness is mandatory: a positive first result cannot override LONG.
    assert updated.cash_selected is False

    second = selector.update_ew_regime_state(
        updated, -0.01, structural_default_cash=False
    )
    assert second.n_raw == 2
    assert second.n_eff == pytest.approx(1.995)
    assert second.weighted_label_sum == pytest.approx(0.995 * 0.02 - 0.01)
    assert second.weighted_squared_label_sum == pytest.approx(
        0.995 * 0.02**2 + 0.01**2
    )


def test_only_the_signal_time_regime_state_is_updated():
    states = selector.initialize_regime_states()
    untouched = states[selector.NOT_RISK_ON_REGIME]
    updated_risk_on = selector.update_ew_regime_state(
        states[selector.RISK_ON_REGIME],
        -0.03,
        structural_default_cash=False,
    )
    next_states = dict(states)
    next_states[selector.RISK_ON_REGIME] = updated_risk_on

    assert next_states[selector.NOT_RISK_ON_REGIME] == untouched
    assert next_states[selector.RISK_ON_REGIME].n_raw == 1
    assert next_states[selector.NOT_RISK_ON_REGIME].n_raw == 0


def test_effective_count_readiness_requires_thirteen_discounted_lessons():
    state = selector.initialize_regime_state(structural_default_cash=False)
    for _ in range(12):
        state = selector.update_ew_regime_state(
            state, 0.01, structural_default_cash=False
        )
    assert state.n_eff == pytest.approx(sum(0.995**i for i in range(12)))
    assert state.n_eff < selector.MIN_EFFECTIVE_LESSONS
    assert state.cash_selected is False

    state = selector.update_ew_regime_state(
        state, 0.01, structural_default_cash=False
    )
    assert state.n_eff >= selector.MIN_EFFECTIVE_LESSONS
    assert state.cash_selected is True


def test_ready_latch_keeps_learning_and_can_switch_back_in_the_future():
    state = selector.initialize_regime_state(structural_default_cash=True)
    for _ in range(13):
        state = selector.update_ew_regime_state(
            state, -0.02, structural_default_cash=True
        )
    assert state.n_eff >= selector.MIN_EFFECTIVE_LESSONS
    assert state.mean < selector.NEGATIVE_MEAN_THRESHOLD
    assert state.cash_selected is False

    negative_count = state.n_raw
    while not state.cash_selected:
        state = selector.update_ew_regime_state(
            state, 0.05, structural_default_cash=True
        )
        assert state.n_raw < 200
    assert state.n_raw > negative_count
    assert state.mean > selector.POSITIVE_MEAN_THRESHOLD


@pytest.mark.parametrize(
    ("mean", "previous_cash", "expected_cash"),
    [
        (0.0010000001, False, True),
        (0.001, False, False),
        (0.0005, True, True),
        (-0.0005, False, False),
        (-0.001, True, True),
        (-0.0010000001, True, False),
    ],
)
def test_hysteresis_uses_strict_thresholds_and_retains_inside_band(
    mean: float, previous_cash: bool, expected_cash: bool
):
    n_eff = (1.0 - selector.LESSON_DISCOUNT**13) / (
        1.0 - selector.LESSON_DISCOUNT
    )
    state = selector.EwRegimeState(
        n_raw=13,
        n_eff=n_eff,
        weighted_label_sum=n_eff * mean,
        weighted_squared_label_sum=0.01,
        cash_selected=previous_cash,
    )
    assert selector.resolve_regime_cash_selection(
        state, structural_default_cash=previous_cash
    ) is expected_cash


def test_unready_state_is_forced_to_its_structural_default():
    n_eff = (1.0 - selector.LESSON_DISCOUNT**12) / (
        1.0 - selector.LESSON_DISCOUNT
    )
    risk_on = selector.EwRegimeState(
        n_raw=12,
        n_eff=n_eff,
        weighted_label_sum=1.0,
        weighted_squared_label_sum=1.0,
        cash_selected=True,
    )
    risk_off = selector.EwRegimeState(
        n_raw=12,
        n_eff=n_eff,
        weighted_label_sum=-1.0,
        weighted_squared_label_sum=1.0,
        cash_selected=False,
    )
    assert selector.resolve_regime_cash_selection(
        risk_on, structural_default_cash=False
    ) is False
    assert selector.resolve_regime_cash_selection(
        risk_off, structural_default_cash=True
    ) is True


@pytest.mark.parametrize("bad_label", [np.nan, np.inf, -np.inf])
def test_nonfinite_labels_fail_closed(bad_label: float):
    state = selector.initialize_regime_state(structural_default_cash=False)
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        selector.update_ew_regime_state(
            state, bad_label, structural_default_cash=False
        )


def test_state_and_two_regime_checkpoint_round_trip_is_exact():
    states = selector.initialize_regime_states()
    states[selector.RISK_ON_REGIME] = selector.update_ew_regime_state(
        states[selector.RISK_ON_REGIME],
        -0.02,
        structural_default_cash=False,
    )
    states[selector.NOT_RISK_ON_REGIME] = selector.update_ew_regime_state(
        states[selector.NOT_RISK_ON_REGIME],
        0.03,
        structural_default_cash=True,
    )

    structural_defaults = {
        selector.RISK_ON_REGIME: False,
        selector.NOT_RISK_ON_REGIME: True,
    }
    for name, state in states.items():
        assert selector.restore_ew_regime_state(
            selector.serialize_ew_regime_state(state),
            structural_default_cash=structural_defaults[name],
        ) == state, name
    payload = selector.serialize_regime_states(states)
    assert selector.restore_regime_states(payload) == states

    tampered = copy.deepcopy(payload)
    tampered["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="schema|incompatible"):
        selector.restore_regime_states(tampered)


def test_state_restore_rejects_effective_count_inconsistent_with_raw_count():
    state = selector.initialize_regime_state(structural_default_cash=False)
    for _ in range(13):
        state = selector.update_ew_regime_state(
            state, 0.0, structural_default_cash=False
        )
    payload = selector.serialize_ew_regime_state(state)
    payload["n_eff"] += 1e-8

    with pytest.raises(ValueError, match="n_eff is inconsistent with n_raw"):
        selector.restore_ew_regime_state(
            payload, structural_default_cash=False
        )


def test_state_restore_rejects_moments_that_violate_cauchy_schwarz():
    state = selector.initialize_regime_state(structural_default_cash=False)
    for _ in range(13):
        state = selector.update_ew_regime_state(
            state, 0.0, structural_default_cash=False
        )
    payload = selector.serialize_ew_regime_state(state)
    payload["weighted_label_sum"] = 1.0

    with pytest.raises(ValueError, match=r"S\^2 <= n_eff \* Q"):
        selector.restore_ew_regime_state(
            payload, structural_default_cash=False
        )


def test_pending_lesson_round_trip_is_strict_and_preserves_regime():
    pending = (
        selector.PendingRegimeLesson(
            signal_close=pd.Timestamp("2017-12-28"),
            sessions_until_maturity=1,
            risk_on=True,
            entry_adjusted_open=123.5,
            learning_eligible=True,
        ),
        selector.PendingRegimeLesson(
            signal_close=pd.Timestamp("2017-12-29"),
            sessions_until_maturity=2,
            risk_on=False,
            entry_adjusted_open=None,
            learning_eligible=True,
        ),
    )
    payload = selector.serialize_pending_regime_lessons(pending)
    assert selector.restore_pending_regime_lessons(payload) == pending

    bad = copy.deepcopy(payload)
    bad["lessons"][0]["risk_on"] = "yes"
    with pytest.raises(ValueError, match="risk_on|boolean|invalid"):
        selector.restore_pending_regime_lessons(bad)


def test_shadow_label_matures_at_t_plus_two_before_same_close_prediction():
    frame, signal = _same_close_update_frame()
    forecast = _forecast(frame)
    maturity = signal + 2
    adjusted_open = (
        frame["aapl_open"] * frame["aapl_adj_close"] / frame["aapl_close"]
    )
    expected = math.log(
        adjusted_open.iloc[signal + 1] / adjusted_open.iloc[maturity]
    ) + math.log(0.999 / 1.001)

    assert forecast.iloc[signal]["shadow_matures_on_close"] == frame.index[maturity]
    assert not bool(forecast.iloc[signal + 1]["shadow_matured_now"])
    assert bool(forecast.iloc[maturity]["shadow_matured_now"])
    assert bool(forecast.iloc[maturity]["shadow_lesson_added_now"])
    assert forecast.iloc[maturity]["shadow_signal_close"] == frame.index[signal]
    assert forecast.iloc[maturity]["shadow_label_10bps"] == pytest.approx(expected)
    assert bool(forecast.iloc[maturity]["canonical_union_cash_signal"])
    # The maturity row is itself a raw/canonical candidate and must see the
    # update before it is scored.
    assert forecast.iloc[maturity]["not_risk_on_n_raw"] == 1
    assert forecast.iloc[maturity]["selector_regime_n_eff"] == pytest.approx(1.0)


def test_lesson_updates_signal_time_regime_even_if_market_flips_by_maturity():
    frame = _market_frame(155)
    signal = 130
    for column in ("spy_adj_close", "qqq_adj_close"):
        location = frame.columns.get_loc(column)
        frame.iloc[signal - 20, location] = 100.0
        frame.iloc[signal - 10, location] = 130.0
        frame.iloc[signal, location] = 110.0
        frame.iloc[signal + 2, location] = 80.0
    _set_intraday_return(frame, signal, 0.30)

    forecast = _forecast(frame)
    maturity = signal + 2
    assert bool(forecast.iloc[signal]["canonical_union_cash_signal"])
    assert bool(forecast.iloc[signal]["risk_on"])
    assert not bool(forecast.iloc[maturity]["risk_on"])
    assert bool(forecast.iloc[maturity]["shadow_matured_signal_risk_on"])
    assert int(forecast.iloc[maturity]["risk_on_n_raw"]) == 1
    assert int(forecast.iloc[maturity]["not_risk_on_n_raw"]) == 0


def test_pre_2000_shadow_opportunity_is_warmup_only():
    frame = _market_frame(330, start="1999-01-04")
    before = 200
    after = 270
    assert frame.index[before] < pd.Timestamp("2000-01-01")
    assert frame.index[after] >= pd.Timestamp("2000-01-01")
    for signal in (before, after):
        _set_intraday_return(frame, signal, 0.30)
        frame.iloc[signal + 1, frame.columns.get_loc("aapl_open")] *= 1.20
        frame.iloc[signal + 2, frame.columns.get_loc("aapl_open")] *= 0.80

    forecast = _forecast(frame)
    assert bool(forecast.iloc[before]["shadow_canonical_union_signal"])
    assert bool(forecast.iloc[before + 2]["shadow_matured_now"])
    assert not bool(forecast.iloc[before + 2]["shadow_lesson_added_now"])
    assert bool(forecast.iloc[after + 2]["shadow_lesson_added_now"])
    total = int(forecast.iloc[after + 2]["risk_on_n_raw"]) + int(
        forecast.iloc[after + 2]["not_risk_on_n_raw"]
    )
    assert total == 1
    assert forecast.attrs["lesson_memory_start"] == "2000-01-01"


def test_selector_is_a_pure_union_filter_with_exact_binary_exposure():
    frame, _ = _negative_shadow_frame(17)
    forecast = _forecast(frame)
    union = forecast["canonical_union_cash_signal"].astype(bool)
    learner = forecast["learner_cash_signal"].astype(bool)

    assert union.any()
    assert not bool((learner & ~union).any())
    assert np.array_equal(
        forecast["target_exposure"].to_numpy(dtype=float),
        np.where(learner, 0.0, 1.0),
    )
    assert set(forecast["target_exposure"].unique()).issubset({0.0, 1.0})


def test_risk_off_default_takes_union_until_negative_lessons_flip_latch():
    frame, positions = _negative_shadow_frame(17)
    forecast = _forecast(frame)

    assert not bool(forecast.iloc[positions[0]]["risk_on"])
    assert bool(forecast.iloc[positions[0]]["selector_cash_prediction"])
    assert bool(forecast.iloc[positions[0]]["learner_cash_signal"])
    skipped = [
        position
        for position in positions
        if bool(forecast.iloc[position]["selector_skip_signal"])
    ]
    assert skipped
    first = skipped[0]
    assert forecast.iloc[first]["not_risk_on_n_eff"] >= 12.0
    assert forecast.iloc[first]["not_risk_on_mean"] < -0.001
    assert not bool(forecast.iloc[first]["not_risk_on_cash_selected"])
    assert not bool(forecast.iloc[first]["learner_cash_signal"])


def test_selector_skip_cannot_resurrect_t_plus_one_suppressed_candidate():
    frame, _ = _negative_shadow_frame(17)
    forecast = _forecast(frame)
    skipped = np.flatnonzero(forecast["selector_skip_signal"].to_numpy(dtype=bool))
    assert len(skipped) > 0
    first = int(skipped[0])

    assert bool(forecast.iloc[first]["canonical_union_cash_signal"])
    assert not bool(forecast.iloc[first]["learner_cash_signal"])
    assert bool(forecast.iloc[first + 1]["contextual_raw_signal"])
    assert not bool(forecast.iloc[first + 1]["union_candidate_signal"])
    assert not bool(forecast.iloc[first + 1]["canonical_union_cash_signal"])
    assert not bool(forecast.iloc[first + 1]["learner_cash_signal"])


def test_skipped_opportunity_still_matures_into_its_signal_time_regime():
    frame, _ = _negative_shadow_frame(17)
    forecast = _forecast(frame)
    first = int(
        np.flatnonzero(forecast["selector_skip_signal"].to_numpy(dtype=bool))[0]
    )
    maturity = first + 2
    prior_count = int(forecast.iloc[maturity - 1]["not_risk_on_n_raw"])

    assert not bool(forecast.iloc[first]["risk_on"])
    assert bool(forecast.iloc[maturity]["shadow_matured_now"])
    assert not bool(forecast.iloc[maturity]["shadow_matured_signal_risk_on"])
    assert bool(forecast.iloc[maturity]["shadow_lesson_added_now"])
    assert int(forecast.iloc[maturity]["not_risk_on_n_raw"]) == prior_count + 1


def test_mutating_an_opportunitys_future_outcome_cannot_change_its_action():
    frame, _ = _negative_shadow_frame(17)
    original = _forecast(frame)
    decision = int(
        np.flatnonzero(original["selector_skip_signal"].to_numpy(dtype=bool))[0]
    )
    changed = frame.copy()
    changed.iloc[decision + 1, changed.columns.get_loc("aapl_open")] *= 3.0
    changed.iloc[decision + 2, changed.columns.get_loc("aapl_open")] *= 0.25
    replay = _forecast(changed)

    predictive = [
        "risk_on",
        "not_risk_on_n_raw",
        "not_risk_on_n_eff",
        "not_risk_on_weighted_label_sum",
        "not_risk_on_weighted_squared_label_sum",
        "not_risk_on_mean",
        "not_risk_on_ready",
        "not_risk_on_cash_selected",
        "selector_regime_n_eff",
        "selector_regime_mean",
        "selector_regime_ready",
        "selector_cash_prediction",
        "selector_skip_prediction",
        "selector_skip_signal",
        "learner_cash_signal",
        "target_exposure",
    ]
    pd.testing.assert_series_equal(
        original.iloc[decision][predictive], replay.iloc[decision][predictive]
    )
    assert original.iloc[decision + 2]["shadow_label_10bps"] != pytest.approx(
        replay.iloc[decision + 2]["shadow_label_10bps"]
    )


def test_frozen_cutoff_stops_admission_by_maturity_while_online_continues():
    frame, positions = _negative_shadow_frame(17)
    cutoff = frame.index[positions[12] + 2]
    frozen = _forecast(
        frame,
        learning_mode=selector.FROZEN_CUTOFF_MODE,
        frozen_cutoff=cutoff,
    )
    online = _forecast(frame)

    state_columns = [
        f"{regime}_{suffix}"
        for regime in selector.REGIME_NAMES
        for suffix in (
            "n_raw",
            "n_eff",
            "weighted_label_sum",
            "weighted_squared_label_sum",
            "mean",
            "ready",
            "cash_selected",
        )
    ]
    pd.testing.assert_frame_equal(
        frozen.loc[:cutoff, state_columns], online.loc[:cutoff, state_columns]
    )
    frozen_total = frozen["risk_on_n_raw"] + frozen["not_risk_on_n_raw"]
    online_total = online["risk_on_n_raw"] + online["not_risk_on_n_raw"]
    assert frozen_total.loc[cutoff:].nunique() == 1
    assert int(online_total.iloc[-1]) > int(frozen_total.iloc[-1])
    assert not frozen.loc[
        frozen.index > cutoff, "shadow_lesson_added_now"
    ].astype(bool).any()
    assert frozen.attrs["post_cutoff_outcomes_used_for_learning"] is False
    assert online.attrs["post_cutoff_outcomes_used_for_learning"] is True


def test_unresolved_prefix_lesson_stays_pending_until_t_plus_two_exists():
    full, positions = _negative_shadow_frame(1)
    signal = positions[0]
    prefix = full.iloc[: signal + 2].copy()
    pending = selector.replay_binary_regime_union_selector(
        prefix, learning_mode=selector.CAUSAL_ONLINE_MODE
    )

    assert bool(pending.forecast.iloc[signal]["shadow_pending"])
    assert len(pending.pending_lessons) == 1
    assert pending.pending_lessons[0].sessions_until_maturity == 1
    assert sum(state.n_raw for state in pending.final_states.values()) == 0

    complete = selector.replay_binary_regime_union_selector(
        full.iloc[: signal + 3], learning_mode=selector.CAUSAL_ONLINE_MODE
    )
    assert len(complete.pending_lessons) == 0
    assert bool(complete.forecast.iloc[signal + 2]["shadow_lesson_added_now"])
    assert sum(state.n_raw for state in complete.final_states.values()) == 1


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_market_input_fails_closed(bad_value: float):
    frame, positions = _negative_shadow_frame(1)
    frame.iloc[positions[0], frame.columns.get_loc("spy_adj_close")] = bad_value
    with pytest.raises((ValueError, RuntimeError), match="finite|nonfinite"):
        _forecast(frame)


def test_repeated_replay_is_bitwise_deterministic():
    frame, _ = _negative_shadow_frame(17)
    first = _forecast(frame)
    second = _forecast(frame.copy())
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert first.attrs == second.attrs


def test_learning_mode_contract_fails_closed_on_ambiguous_cutoffs():
    frame = _market_frame(150)
    with pytest.raises(ValueError, match="frozen_cutoff"):
        _forecast(frame, learning_mode=selector.FROZEN_CUTOFF_MODE)
    with pytest.raises(ValueError, match="must not receive|frozen_cutoff"):
        _forecast(
            frame,
            learning_mode=selector.CAUSAL_ONLINE_MODE,
            frozen_cutoff=frame.index[-1],
        )
    with pytest.raises(ValueError, match="unsupported learning_mode"):
        _forecast(frame, learning_mode="look_ahead")
