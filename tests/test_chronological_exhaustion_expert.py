from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.chronological_exhaustion_expert import (
    CAUSAL_ONLINE_MODE,
    EDGE_PRIOR_SCALE,
    EDGE_PRIOR_STRENGTH,
    FROZEN_CUTOFF_MODE,
    MIN_MATURED_EPISODES,
    PROBABILITY_AND_EDGE_Z,
    ROUND_TRIP_LOG_FRICTION,
    build_chronological_exhaustion_forecast,
    build_fixed_expert_signals,
    build_unfiltered_expert_targets,
    canonicalize_one_session_signals,
    posterior_from_sufficient_statistics,
)


def _market_frame(periods: int = 340, *, start: str = "2000-01-03") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=periods)
    aapl_close = np.linspace(220.0, 100.0, periods)
    spy_close = np.linspace(500.0, 250.0, periods)
    qqq_close = np.linspace(400.0, 180.0, periods)
    return pd.DataFrame(
        {
            "aapl_open": aapl_close.copy(),
            "aapl_close": aapl_close,
            "aapl_adj_close": aapl_close,
            "spy_adj_close": spy_close,
            "qqq_adj_close": qqq_close,
        },
        index=index,
    )


def _profitable_episode_frame() -> tuple[pd.DataFrame, list[int]]:
    frame = _market_frame()
    signals = list(range(130, 311, 15))
    assert len(signals) == 13
    open_column = frame.columns.get_loc("aapl_open")
    close_column = frame.columns.get_loc("aapl_close")
    for signal in signals:
        close = float(frame.iloc[signal, close_column])
        frame.iloc[signal, open_column] = close / 1.20
        next_close = float(frame.iloc[signal + 1, close_column])
        frame.iloc[signal + 1, open_column] = next_close * 1.10
    return frame, signals


def test_canonicalization_skips_only_the_session_after_each_acceptance():
    raw = pd.Series([True, True, True, False, True, True], dtype=bool)

    accepted = canonicalize_one_session_signals(raw)

    assert accepted.tolist() == [True, False, True, False, True, False]


def test_fixed_experts_use_exact_prior_percentiles_market_windows_and_sma():
    frame = _market_frame(150)
    decision = 130
    for offset in (0, 1, 2):
        close = float(frame.iloc[decision + offset]["aapl_close"])
        frame.iloc[
            decision + offset, frame.columns.get_loc("aapl_open")
        ] = close / 1.20

    signals = build_fixed_expert_signals(frame)

    assert signals.iloc[decision]["contextual_prior_intraday_percentile"] == pytest.approx(0.0)
    assert signals.iloc[decision]["weak_trend_prior_intraday_percentile"] == pytest.approx(0.0)
    assert bool(signals.iloc[decision]["contextual_raw_signal"])
    assert bool(signals.iloc[decision]["weak_trend_raw_signal"])
    assert signals.iloc[decision]["aapl_intraday_return"] == pytest.approx(0.20)
    assert signals.iloc[decision]["contextual_spy_return_10"] < 0.0
    assert signals.iloc[decision]["contextual_qqq_return_10"] < 0.0
    assert signals.iloc[decision]["weak_trend_spy_return_20"] < 0.0
    assert signals.iloc[decision]["weak_trend_qqq_return_20"] < 0.0
    assert (
        frame.iloc[decision]["aapl_adj_close"]
        < signals.iloc[decision]["weak_trend_aapl_sma_20"]
    )
    assert signals.iloc[decision : decision + 3][
        "contextual_virtual_signal"
    ].tolist() == [True, False, True]
    assert signals.iloc[decision : decision + 3][
        "weak_trend_virtual_signal"
    ].tolist() == [True, False, True]


def test_unfiltered_union_nonstacks_signals_from_alternating_experts():
    frame = _market_frame(150)
    first = 130
    second = 131
    for decision in (first, second):
        close = float(frame.iloc[decision]["aapl_close"])
        frame.iloc[decision, frame.columns.get_loc("aapl_open")] = close / 1.20

    # At first: 10-session market return is negative and 20-session is
    # positive, so only contextual signals. At second the signs reverse, so
    # only weak-trend signals. Use the same construction for SPY and QQQ.
    for column in ("spy_adj_close", "qqq_adj_close"):
        location = frame.columns.get_loc(column)
        frame.iloc[first - 20, location] = 80.0
        frame.iloc[first - 19, location] = 120.0
        frame.iloc[first - 10, location] = 120.0
        frame.iloc[first - 9, location] = 80.0
        frame.iloc[first, location] = 100.0
        frame.iloc[second, location] = 90.0

    signals = build_fixed_expert_signals(frame)
    targets = build_unfiltered_expert_targets(frame)

    assert bool(signals.iloc[first]["contextual_virtual_signal"])
    assert not bool(signals.iloc[first]["weak_trend_virtual_signal"])
    assert not bool(signals.iloc[second]["contextual_virtual_signal"])
    assert bool(signals.iloc[second]["weak_trend_virtual_signal"])
    assert signals.iloc[first : second + 1][
        "unfiltered_union_candidate_signal"
    ].tolist() == [True, True]
    assert signals.iloc[first : second + 1]["unfiltered_union_signal"].tolist() == [
        True,
        False,
    ]
    assert targets.iloc[first : second + 1][
        "unfiltered_union_target_exposure"
    ].tolist() == [0.0, 1.0]
    assert set(np.unique(targets.to_numpy(dtype=float))) <= {0.0, 1.0}


def test_posterior_matches_the_frozen_equations_and_all_four_trust_gates():
    count = 12
    wins = 12
    edges = np.full(count, 0.05)
    posterior = posterior_from_sufficient_statistics(
        count=count,
        wins=wins,
        sum_net_edge=float(edges.sum()),
        sum_squared_net_edge=float(np.square(edges).sum()),
    )
    alpha = wins + 1.0
    beta = count - wins + 1.0
    probability = alpha / (alpha + beta)
    probability_lower = probability - PROBABILITY_AND_EDGE_Z * math.sqrt(
        alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    )
    edge_mean = float(edges.sum()) / (count + EDGE_PRIOR_STRENGTH)
    edge_second = (
        float(np.square(edges).sum())
        + EDGE_PRIOR_STRENGTH * EDGE_PRIOR_SCALE**2
    ) / (count + EDGE_PRIOR_STRENGTH)
    edge_se = math.sqrt(edge_second / (count + EDGE_PRIOR_STRENGTH))

    assert posterior.cash_win_probability == pytest.approx(probability)
    assert posterior.cash_win_probability_lower == pytest.approx(probability_lower)
    assert posterior.edge_mean == pytest.approx(edge_mean)
    assert posterior.edge_second_moment == pytest.approx(edge_second)
    assert posterior.edge_standard_error == pytest.approx(edge_se)
    assert posterior.edge_lower_score == pytest.approx(
        edge_mean - PROBABILITY_AND_EDGE_Z * edge_se
    )
    assert posterior.count_gate
    assert posterior.probability_gate
    assert posterior.probability_lower_gate
    assert posterior.edge_lower_score_gate
    assert posterior.trusted
    assert "edge_lower_score" in posterior.to_dict()
    assert "edge_lower" not in posterior.to_dict()

    too_early = posterior_from_sufficient_statistics(
        count=MIN_MATURED_EPISODES - 1,
        wins=MIN_MATURED_EPISODES - 1,
        sum_net_edge=0.05 * (MIN_MATURED_EPISODES - 1),
        sum_squared_net_edge=0.05**2 * (MIN_MATURED_EPISODES - 1),
    )
    assert not too_early.count_gate
    assert not too_early.trusted


def test_lessons_enter_at_close_t_plus_two_before_that_closes_prediction():
    frame, signal_positions = _profitable_episode_frame()
    forecast = build_chronological_exhaustion_forecast(
        frame,
        learning_mode=CAUSAL_ONLINE_MODE,
    )
    first = signal_positions[0]
    thirteenth = signal_positions[12]
    adjusted_open = frame["aapl_open"].to_numpy(dtype=float)
    expected_raw_edge = math.log(adjusted_open[first + 1] / adjusted_open[first + 2])

    assert forecast.iloc[first]["contextual_matured_count"] == 0
    assert forecast.iloc[first + 1]["contextual_matured_count"] == 0
    assert forecast.iloc[first + 2]["contextual_matured_count"] == 1
    assert bool(forecast.iloc[first + 2]["contextual_matured_episode_now"])
    assert bool(forecast.iloc[first + 2]["contextual_lesson_added_now"])
    assert forecast.iloc[first + 2]["contextual_matured_raw_edge"] == pytest.approx(
        expected_raw_edge
    )
    assert forecast.iloc[first + 2]["contextual_matured_net_edge"] == pytest.approx(
        expected_raw_edge + ROUND_TRIP_LOG_FRICTION
    )
    assert forecast.iloc[first + 2]["contextual_matured_signal_close"] == frame.index[first]
    assert forecast.iloc[first]["contextual_episode_matures_on_close"] == frame.index[
        first + 2
    ]

    # Twelve profitable episodes have matured before the thirteenth signal.
    # Both experts are then trusted and the combined policy takes one cash day.
    assert forecast.iloc[thirteenth]["contextual_matured_count"] == 12
    assert forecast.iloc[thirteenth]["weak_trend_matured_count"] == 12
    assert bool(forecast.iloc[thirteenth]["contextual_trusted_signal"])
    assert bool(forecast.iloc[thirteenth]["weak_trend_trusted_signal"])
    assert bool(forecast.iloc[thirteenth]["combined_cash_signal"])
    assert forecast.iloc[thirteenth]["target_exposure"] == 0.0
    assert forecast.iloc[thirteenth + 1]["target_exposure"] == 1.0
    assert set(forecast["target_exposure"].unique()) <= {0.0, 1.0}


def test_frozen_mode_admits_only_lessons_matured_by_its_inclusive_cutoff():
    frame, signal_positions = _profitable_episode_frame()
    fifth_maturity = signal_positions[4] + 2
    cutoff = frame.index[fifth_maturity]
    frozen = build_chronological_exhaustion_forecast(
        frame,
        learning_mode=FROZEN_CUTOFF_MODE,
        frozen_cutoff=cutoff,
    )
    online = build_chronological_exhaustion_forecast(
        frame,
        learning_mode=CAUSAL_ONLINE_MODE,
    )

    assert frozen.iloc[-1]["contextual_matured_count"] == 5
    assert frozen.iloc[-1]["weak_trend_matured_count"] == 5
    assert online.iloc[-1]["contextual_matured_count"] == 13
    assert frozen.attrs["post_cutoff_outcomes_used_for_learning"] is False
    assert online.attrs["post_cutoff_outcomes_used_for_learning"] is True
    sixth_maturity = signal_positions[5] + 2
    assert bool(frozen.iloc[sixth_maturity]["contextual_matured_episode_now"])
    assert not bool(frozen.iloc[sixth_maturity]["contextual_lesson_added_now"])
    pd.testing.assert_series_equal(
        frozen.loc[:cutoff, "contextual_matured_count"],
        online.loc[:cutoff, "contextual_matured_count"],
    )
    pd.testing.assert_series_equal(
        frozen.loc[:cutoff, "target_exposure"],
        online.loc[:cutoff, "target_exposure"],
    )


def test_1999_signals_are_warmup_only_and_never_enter_lesson_memory():
    frame = _market_frame(330, start="1999-01-04")
    pre_start = 200
    post_start = 270
    assert frame.index[pre_start] < pd.Timestamp("2000-01-01")
    assert frame.index[post_start] >= pd.Timestamp("2000-01-01")
    for signal in (pre_start, post_start):
        close = float(frame.iloc[signal]["aapl_close"])
        frame.iloc[signal, frame.columns.get_loc("aapl_open")] = close / 1.20
        next_close = float(frame.iloc[signal + 1]["aapl_close"])
        frame.iloc[signal + 1, frame.columns.get_loc("aapl_open")] = (
            next_close * 1.10
        )

    forecast = build_chronological_exhaustion_forecast(
        frame,
        learning_mode=CAUSAL_ONLINE_MODE,
    )

    assert bool(forecast.iloc[pre_start]["contextual_virtual_signal"])
    assert bool(forecast.iloc[pre_start + 2]["contextual_matured_episode_now"])
    assert not bool(forecast.iloc[pre_start + 2]["contextual_lesson_added_now"])
    assert forecast.iloc[post_start]["contextual_matured_count"] == 0
    assert bool(forecast.iloc[post_start + 2]["contextual_lesson_added_now"])
    assert forecast.iloc[post_start + 2]["contextual_matured_count"] == 1
    assert forecast.attrs["lesson_memory_start"] == "2000-01-01"


@pytest.mark.parametrize("supplied_rows", [311, 312])
def test_pending_end_of_stage_signal_is_diagnostic_but_every_target_stays_long(
    supplied_rows,
):
    frame, signal_positions = _profitable_episode_frame()
    pending = signal_positions[-1]
    stage = frame.iloc[:supplied_rows].copy()
    assert pending >= len(stage) - 2

    signals = build_fixed_expert_signals(stage)
    unfiltered = build_unfiltered_expert_targets(stage)
    forecast = build_chronological_exhaustion_forecast(
        stage,
        learning_mode=CAUSAL_ONLINE_MODE,
    )

    assert bool(signals.iloc[pending]["contextual_virtual_signal"])
    assert bool(
        signals.iloc[pending]["contextual_virtual_signal_pending_stage_outcome"]
    )
    assert bool(signals.iloc[pending]["weak_trend_virtual_signal"])
    assert bool(
        signals.iloc[pending]["unfiltered_union_signal_pending_stage_outcome"]
    )
    assert (unfiltered.iloc[pending] == 1.0).all()
    assert bool(forecast.iloc[pending]["combined_trusted_candidate_signal"])
    assert bool(
        forecast.iloc[pending][
            "combined_trusted_candidate_pending_stage_outcome"
        ]
    )
    assert not bool(forecast.iloc[pending]["combined_cash_signal"])
    assert forecast.iloc[pending]["target_exposure"] == 1.0


def test_combined_nonstacking_does_not_suppress_an_ignored_experts_lesson():
    frame, signal_positions = _profitable_episode_frame()
    first = signal_positions[-1]
    second = first + 1
    second_close = float(frame.iloc[second]["aapl_close"])
    frame.iloc[second, frame.columns.get_loc("aapl_open")] = second_close / 1.20
    for column in ("spy_adj_close", "qqq_adj_close"):
        location = frame.columns.get_loc(column)
        frame.iloc[first - 20, location] = 80.0
        frame.iloc[first - 19, location] = 120.0
        frame.iloc[first - 10, location] = 120.0
        frame.iloc[first - 9, location] = 80.0
        frame.iloc[first, location] = 100.0
        frame.iloc[second, location] = 90.0

    forecast = build_chronological_exhaustion_forecast(
        frame,
        learning_mode=CAUSAL_ONLINE_MODE,
    )

    assert bool(forecast.iloc[first]["contextual_trusted_signal"])
    assert not bool(forecast.iloc[first]["weak_trend_virtual_signal"])
    assert not bool(forecast.iloc[second]["contextual_virtual_signal"])
    assert bool(forecast.iloc[second]["weak_trend_trusted_signal"])
    assert forecast.iloc[first : second + 1][
        "combined_trusted_candidate_signal"
    ].tolist() == [True, True]
    assert forecast.iloc[first : second + 1]["combined_cash_signal"].tolist() == [
        True,
        False,
    ]
    assert bool(
        forecast.iloc[second]["combined_signal_ignored_due_to_nonstacking"]
    )
    assert bool(forecast.iloc[second + 2]["weak_trend_matured_episode_now"])
    assert bool(forecast.iloc[second + 2]["weak_trend_lesson_added_now"])


def test_learning_mode_contract_rejects_ambiguous_cutoff_use():
    frame = _market_frame(140)
    with pytest.raises(ValueError, match="requires an inclusive frozen_cutoff"):
        build_chronological_exhaustion_forecast(
            frame,
            learning_mode=FROZEN_CUTOFF_MODE,
        )
    with pytest.raises(ValueError, match="must not receive a frozen_cutoff"):
        build_chronological_exhaustion_forecast(
            frame,
            learning_mode=CAUSAL_ONLINE_MODE,
            frozen_cutoff="2000-12-31",
        )


def test_changing_a_new_signals_future_outcome_cannot_change_its_prediction():
    frame, signal_positions = _profitable_episode_frame()
    decision = signal_positions[-1]
    original = build_chronological_exhaustion_forecast(
        frame,
        learning_mode=CAUSAL_ONLINE_MODE,
    )
    changed = frame.copy()
    changed.iloc[decision + 1, changed.columns.get_loc("aapl_open")] *= 0.25
    changed.iloc[decision + 2, changed.columns.get_loc("aapl_open")] *= 4.0
    replay = build_chronological_exhaustion_forecast(
        changed,
        learning_mode=CAUSAL_ONLINE_MODE,
    )

    predictive_columns = [
        "contextual_matured_count",
        "contextual_posterior_cash_win_probability",
        "contextual_posterior_edge_lower_score",
        "contextual_trusted_signal",
        "weak_trend_matured_count",
        "weak_trend_posterior_cash_win_probability",
        "weak_trend_posterior_edge_lower_score",
        "weak_trend_trusted_signal",
        "combined_cash_signal",
        "target_exposure",
    ]
    pd.testing.assert_series_equal(
        original.iloc[decision][predictive_columns],
        replay.iloc[decision][predictive_columns],
    )
