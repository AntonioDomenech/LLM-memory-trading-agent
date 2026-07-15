from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest

import agent_benchmark.union_contextual_veto as veto_model


FEATURE_COLUMNS = (
    "feature_intercept",
    "feature_weak_only",
    "feature_expert_overlap",
    "feature_tail_strength",
    "feature_market_sentiment_10",
    "feature_market_sentiment_20",
    "feature_aapl_trend",
)


def _market_frame(
    periods: int = 420, *, start: str = "2000-01-03"
) -> pd.DataFrame:
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


def _set_intraday_return(frame: pd.DataFrame, position: int, value: float) -> None:
    close = float(frame.iloc[position]["aapl_close"])
    frame.iloc[position, frame.columns.get_loc("aapl_open")] = close / (1.0 + value)


def _negative_shadow_frame(
    lessons: int = 46,
) -> tuple[pd.DataFrame, list[int]]:
    """Create isolated union opportunities with strongly harmful cash labels."""

    positions = [130 + 5 * offset for offset in range(lessons)]
    frame = _market_frame(positions[-1] + 8)
    open_column = frame.columns.get_loc("aapl_open")
    close_column = frame.columns.get_loc("aapl_close")
    for offset, signal in enumerate(positions):
        # The intended signal is a strong positive intraday return. The next
        # row is also a raw signal but must be suppressed by the union cooldown.
        # Increase the synthetic tail monotonically so the prior rolling 90th
        # percentile cannot eventually equal and suppress later opportunities.
        intraday_return = 0.20 + 0.01 * offset
        signal_close = float(frame.iloc[signal, close_column])
        frame.iloc[signal, open_column] = signal_close / (1.0 + intraday_return)
        next_close = float(frame.iloc[signal + 1, close_column])
        frame.iloc[signal + 1, open_column] = next_close / (
            1.0 + intraday_return
        )
        # A much higher re-entry open makes the union cash episode harmful.
        exit_close = float(frame.iloc[signal + 2, close_column])
        frame.iloc[signal + 2, open_column] = exit_close * 1.20
    return frame, positions


def _same_close_update_frame() -> tuple[pd.DataFrame, int]:
    """Make a signal at t and another candidate exactly at its t+2 maturity."""

    frame = _market_frame(150)
    signal = 130
    open_column = frame.columns.get_loc("aapl_open")
    close_column = frame.columns.get_loc("aapl_close")
    signal_close = float(frame.iloc[signal, close_column])
    frame.iloc[signal, open_column] = signal_close / 1.20
    # High t+1 open and low t+2 open make a profitable cash label while the
    # low t+2 open independently creates the next raw/canonical candidate.
    next_close = float(frame.iloc[signal + 1, close_column])
    frame.iloc[signal + 1, open_column] = next_close * 1.20
    maturity_close = float(frame.iloc[signal + 2, close_column])
    frame.iloc[signal + 2, open_column] = maturity_close / 1.20
    return frame, signal


def _forecast(
    frame: pd.DataFrame,
    *,
    learning_mode: str = veto_model.CAUSAL_ONLINE_MODE,
    frozen_cutoff: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    return veto_model.build_union_contextual_veto_forecast(
        frame,
        learning_mode=learning_mode,
        frozen_cutoff=frozen_cutoff,
    )


def _state_matrix(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    matrix_items: list[tuple[int, int, float]] = []
    vector_items: list[tuple[int, float]] = []
    for name, value in row.items():
        matrix_match = re.fullmatch(r"state_A_(\d+)_(\d+)", str(name))
        if matrix_match:
            matrix_items.append(
                (int(matrix_match.group(1)), int(matrix_match.group(2)), float(value))
            )
            continue
        vector_match = re.fullmatch(r"state_b_(\d+)", str(name))
        if vector_match:
            vector_items.append((int(vector_match.group(1)), float(value)))
    assert matrix_items, "forecast must expose the upper-triangle state_A checkpoint"
    assert vector_items, "forecast must expose the state_b checkpoint"
    offset = min(
        min(first, second) for first, second, _ in matrix_items
    )
    size = max(max(first, second) for first, second, _ in matrix_items) - offset + 1
    assert size == len(FEATURE_COLUMNS)
    matrix = np.zeros((size, size), dtype=float)
    for first, second, value in matrix_items:
        i = first - offset
        j = second - offset
        matrix[i, j] = value
        matrix[j, i] = value
    vector_offset = min(index for index, _ in vector_items)
    vector = np.zeros(size, dtype=float)
    for index, value in vector_items:
        vector[index - vector_offset] = value
    return matrix, vector


def test_close_time_features_match_the_frozen_point_in_time_transforms():
    frame = _market_frame(150)
    decision = 130
    # A non-degenerate prior window makes the exact rank test sensitive to the
    # 126-row lookback and to excluding the current session. Descending order
    # keeps the immediately preceding row from creating a cooldown collision.
    prior_returns = np.arange(126, dtype=float)[::-1] / 10_000.0
    for position, value in zip(
        range(decision - 126, decision), prior_returns, strict=True
    ):
        _set_intraday_return(frame, position, float(value))
    current_return = 0.01195
    _set_intraday_return(frame, decision, current_return)

    forecast = _forecast(frame)
    row = forecast.iloc[decision]
    rank = float(np.count_nonzero(prior_returns <= current_return)) / 126.0
    expected_tail = float(np.clip((rank - 0.90) / 0.10, 0.0, 1.0))
    spy_10 = frame["spy_adj_close"].iloc[decision] / frame["spy_adj_close"].iloc[
        decision - 10
    ] - 1.0
    qqq_10 = frame["qqq_adj_close"].iloc[decision] / frame["qqq_adj_close"].iloc[
        decision - 10
    ] - 1.0
    spy_20 = frame["spy_adj_close"].iloc[decision] / frame["spy_adj_close"].iloc[
        decision - 20
    ] - 1.0
    qqq_20 = frame["qqq_adj_close"].iloc[decision] / frame["qqq_adj_close"].iloc[
        decision - 20
    ] - 1.0
    sma_20 = float(frame["aapl_adj_close"].iloc[decision - 19 : decision + 1].mean())
    expected_trend = float(
        np.clip(
            (frame["aapl_adj_close"].iloc[decision] / sma_20 - 1.0) / 0.10,
            -1.0,
            1.0,
        )
    )

    assert bool(row["union_candidate_signal"])
    assert row["feature_intercept"] == pytest.approx(1.0)
    assert row["feature_weak_only"] == pytest.approx(0.0)
    assert row["feature_expert_overlap"] == pytest.approx(1.0)
    assert row["feature_tail_strength"] == pytest.approx(expected_tail)
    assert row["feature_market_sentiment_10"] == pytest.approx(
        np.clip(np.mean([spy_10, qqq_10]) / 0.10, -1.0, 1.0)
    )
    assert row["feature_market_sentiment_20"] == pytest.approx(
        np.clip(np.mean([spy_20, qqq_20]) / 0.15, -1.0, 1.0)
    )
    assert row["feature_aapl_trend"] == pytest.approx(expected_trend)


def test_weak_only_identity_is_distinct_from_the_overlap_reference():
    frame = _market_frame(150)
    decision = 130
    _set_intraday_return(frame, decision, 0.20)
    # A positive 10-session return disables contextual, while a negative
    # 20-session return and the declining AAPL trend keep weak-trend active.
    for column in ("spy_adj_close", "qqq_adj_close"):
        location = frame.columns.get_loc(column)
        frame.iloc[decision - 20, location] = 120.0
        frame.iloc[decision - 10, location] = 80.0
        frame.iloc[decision, location] = 100.0

    row = _forecast(frame).iloc[decision]

    assert bool(row["union_candidate_signal"])
    assert row["feature_weak_only"] == pytest.approx(1.0)
    assert row["feature_expert_overlap"] == pytest.approx(0.0)


def test_shadow_label_matures_at_t_plus_two_before_same_close_scoring():
    frame, signal = _same_close_update_frame()
    forecast = _forecast(frame)
    maturity = signal + 2
    adjusted_open = (
        frame["aapl_open"] * frame["aapl_adj_close"] / frame["aapl_close"]
    )
    expected_label = math.log(
        adjusted_open.iloc[signal + 1] / adjusted_open.iloc[maturity]
    ) + math.log(0.999 / 1.001)

    assert forecast.iloc[signal]["shadow_matures_on_close"] == frame.index[maturity]
    assert not bool(forecast.iloc[signal]["shadow_matured_now"])
    assert not bool(forecast.iloc[signal + 1]["shadow_matured_now"])
    assert bool(forecast.iloc[maturity]["shadow_matured_now"])
    assert bool(forecast.iloc[maturity]["shadow_lesson_added_now"])
    assert forecast.iloc[maturity]["shadow_signal_close"] == frame.index[signal]
    assert forecast.iloc[maturity]["shadow_label_10bps"] == pytest.approx(
        expected_label
    )
    assert forecast.iloc[signal]["n_raw"] == 0
    assert forecast.iloc[signal + 1]["n_raw"] == 0
    assert forecast.iloc[maturity]["n_raw"] == 1
    assert forecast.iloc[maturity]["n_eff"] == pytest.approx(1.0)
    # The maturity row is itself a canonical opportunity. Its score must expose
    # the already-updated state, though warmup still forbids a veto.
    assert bool(forecast.iloc[maturity]["canonical_union_cash_signal"])
    assert np.isfinite(float(forecast.iloc[maturity]["model_upper"]))
    assert not bool(forecast.iloc[maturity]["model_ready"])
    assert not bool(forecast.iloc[maturity]["veto"])


def test_1999_shadow_opportunity_is_warmup_only_and_never_enters_memory():
    frame = _market_frame(330, start="1999-01-04")
    before_start = 200
    after_start = 270
    assert frame.index[before_start] < pd.Timestamp("2000-01-01")
    assert frame.index[after_start] >= pd.Timestamp("2000-01-01")
    for signal in (before_start, after_start):
        _set_intraday_return(frame, signal, 0.20)
        # Complete each label without introducing a consecutive raw signal.
        frame.iloc[signal + 1, frame.columns.get_loc("aapl_open")] *= 1.20
        frame.iloc[signal + 2, frame.columns.get_loc("aapl_open")] *= 0.80

    forecast = _forecast(frame)

    assert bool(forecast.iloc[before_start]["shadow_canonical_union_signal"])
    assert bool(forecast.iloc[before_start + 2]["shadow_matured_now"])
    assert not bool(forecast.iloc[before_start + 2]["shadow_lesson_added_now"])
    assert forecast.iloc[after_start]["n_raw"] == 0
    assert bool(forecast.iloc[after_start + 2]["shadow_lesson_added_now"])
    assert forecast.iloc[after_start + 2]["n_raw"] == 1
    assert forecast.attrs["lesson_memory_start"] == "2000-01-01"


def test_first_bayesian_update_and_same_close_prediction_match_frozen_equations():
    frame, signal = _same_close_update_frame()
    forecast = _forecast(frame)
    maturity = signal + 2
    signal_x = forecast.iloc[signal].loc[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    current_x = forecast.iloc[maturity].loc[list(FEATURE_COLUMNS)].to_numpy(
        dtype=float
    )
    y = float(forecast.iloc[maturity]["shadow_label_10bps"])
    prior_precision = np.eye(len(FEATURE_COLUMNS), dtype=float) / 0.02**2
    expected_a = prior_precision + np.outer(signal_x, signal_x) / 0.04**2
    expected_b = signal_x * y / 0.04**2
    actual_a, actual_b = _state_matrix(forecast.iloc[maturity])
    np.testing.assert_allclose(actual_a, expected_a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_b, expected_b, rtol=1e-12, atol=1e-12)
    beta = np.linalg.solve(expected_a, expected_b)
    covariance = np.linalg.inv(expected_a)
    expected_mu = float(current_x @ beta)
    expected_se = float(math.sqrt(current_x @ covariance @ current_x))
    assert forecast.iloc[maturity]["model_mu"] == pytest.approx(expected_mu)
    assert forecast.iloc[maturity]["model_se"] == pytest.approx(expected_se)
    assert forecast.iloc[maturity]["model_upper"] == pytest.approx(
        expected_mu + 1.282 * expected_se
    )
    assert forecast.iloc[maturity]["label_sum"] == pytest.approx(y)
    assert forecast.iloc[maturity]["squared_label_sum"] == pytest.approx(y**2)
    assert forecast.iloc[maturity]["wins"] == int(y > 0.0)


def test_second_bayesian_update_discounts_only_learned_precision_and_vector():
    frame, positions = _negative_shadow_frame(lessons=2)
    forecast = _forecast(frame)
    first, second = positions
    second_maturity = second + 2
    x_first = forecast.iloc[first].loc[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    x_second = forecast.iloc[second].loc[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y_first = float(forecast.iloc[first + 2]["shadow_label_10bps"])
    y_second = float(forecast.iloc[second_maturity]["shadow_label_10bps"])
    a0 = np.eye(len(FEATURE_COLUMNS), dtype=float) / 0.02**2
    a_first = a0 + np.outer(x_first, x_first) / 0.04**2
    b_first = x_first * y_first / 0.04**2
    expected_a = a0 + 0.995 * (a_first - a0) + np.outer(
        x_second, x_second
    ) / 0.04**2
    expected_b = 0.995 * b_first + x_second * y_second / 0.04**2

    actual_a, actual_b = _state_matrix(forecast.iloc[second_maturity])
    np.testing.assert_allclose(actual_a, expected_a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_b, expected_b, rtol=1e-12, atol=1e-12)
    assert forecast.iloc[second_maturity]["n_raw"] == 2
    assert forecast.iloc[second_maturity]["n_eff"] == pytest.approx(1.995)


def test_warmup_is_default_on_and_learner_is_always_a_pure_union_veto():
    frame, _ = _negative_shadow_frame(lessons=25)
    forecast = _forecast(frame)
    warmup = forecast["n_raw"] < 40
    canonical = forecast["canonical_union_cash_signal"].astype(bool)
    learner = forecast["learner_cash_signal"].astype(bool)

    assert canonical.any()
    assert not forecast.loc[warmup, "veto"].astype(bool).any()
    pd.testing.assert_series_equal(
        learner.loc[warmup], canonical.loc[warmup], check_names=False
    )
    assert not (learner & ~canonical).any()
    assert (
        forecast["target_exposure"].to_numpy(dtype=float)
        == np.where(learner, 0.0, 1.0)
    ).all()


def test_veto_does_not_resurrect_the_next_raw_candidate_suppressed_by_union():
    frame, _ = _negative_shadow_frame(lessons=46)
    forecast = _forecast(frame)
    vetoed = np.flatnonzero(forecast["veto"].to_numpy(dtype=bool))
    assert len(vetoed) > 0
    first = int(vetoed[0])

    assert bool(forecast.iloc[first]["union_candidate_signal"])
    assert bool(forecast.iloc[first]["canonical_union_cash_signal"])
    assert not bool(forecast.iloc[first]["learner_cash_signal"])
    assert forecast.iloc[first]["target_exposure"] == 1.0
    # The synthetic t+1 row is deliberately another raw expert signal. The
    # expert's independent cooldown suppresses it before union construction;
    # vetoing t therefore cannot resurrect it.
    assert bool(forecast.iloc[first + 1]["contextual_raw_signal"])
    assert bool(forecast.iloc[first + 1]["weak_trend_raw_signal"])
    assert not bool(forecast.iloc[first + 1]["union_candidate_signal"])
    assert not bool(forecast.iloc[first + 1]["canonical_union_cash_signal"])
    assert not bool(forecast.iloc[first + 1]["learner_cash_signal"])


def test_vetoed_opportunity_still_matures_as_one_shadow_lesson():
    frame, _ = _negative_shadow_frame(lessons=46)
    forecast = _forecast(frame)
    first = int(np.flatnonzero(forecast["veto"].to_numpy(dtype=bool))[0])
    maturity = first + 2

    assert bool(forecast.iloc[first]["veto"])
    assert not bool(forecast.iloc[first]["learner_cash_signal"])
    assert bool(forecast.iloc[maturity]["shadow_matured_now"])
    assert bool(forecast.iloc[maturity]["shadow_lesson_added_now"])
    assert forecast.iloc[maturity]["shadow_signal_close"] == frame.index[first]
    assert forecast.iloc[maturity]["n_raw"] == forecast.iloc[maturity - 1]["n_raw"] + 1
    assert float(forecast.iloc[maturity]["shadow_label_10bps"]) < 0.0


def test_veto_boolean_is_exactly_the_frozen_readiness_and_strict_harm_rule():
    frame, _ = _negative_shadow_frame(lessons=46)
    forecast = _forecast(frame)
    expected = (
        forecast["canonical_union_cash_signal"].astype(bool)
        & (forecast["n_raw"].astype(int) >= 40)
        & (forecast["n_eff"].astype(float) >= 30.0)
        & (forecast["model_upper"].astype(float) < -0.001)
    )
    pd.testing.assert_series_equal(
        forecast["veto"].astype(bool), expected.astype(bool), check_names=False
    )
    assert forecast["veto"].astype(bool).any()

    assert veto_model.should_veto(n_raw=40, n_eff=30.0, upper=-0.0010000001)
    assert not veto_model.should_veto(n_raw=40, n_eff=30.0, upper=-0.001)
    assert not veto_model.should_veto(
        n_raw=40, n_eff=30.0, upper=-0.0009999999
    )
    assert not veto_model.should_veto(n_raw=39, n_eff=30.0, upper=-1.0)
    assert not veto_model.should_veto(n_raw=40, n_eff=29.999, upper=-1.0)


def test_frozen_cutoff_applies_to_maturity_date_while_online_keeps_learning():
    frame, positions = _negative_shadow_frame(lessons=46)
    cutoff_position = positions[39] + 2
    cutoff = frame.index[cutoff_position]
    frozen = _forecast(
        frame,
        learning_mode=veto_model.FROZEN_CUTOFF_MODE,
        frozen_cutoff=cutoff,
    )
    online = _forecast(frame)

    columns = [
        *FEATURE_COLUMNS,
        "n_raw",
        "n_eff",
        "wins",
        "label_sum",
        "squared_label_sum",
        "model_mu",
        "model_se",
        "model_upper",
        "model_ready",
        "veto",
        "learner_cash_signal",
        "target_exposure",
    ]
    pd.testing.assert_frame_equal(
        frozen.loc[:cutoff, columns], online.loc[:cutoff, columns]
    )
    assert frozen.loc[cutoff:, "n_raw"].nunique() == 1
    assert int(online.iloc[-1]["n_raw"]) > int(frozen.iloc[-1]["n_raw"])
    post_cutoff_maturities = frozen.index > cutoff
    assert frozen.loc[post_cutoff_maturities, "shadow_matured_now"].astype(bool).any()
    assert not frozen.loc[
        post_cutoff_maturities, "shadow_lesson_added_now"
    ].astype(bool).any()
    assert frozen.attrs["post_cutoff_outcomes_used_for_learning"] is False
    assert online.attrs["post_cutoff_outcomes_used_for_learning"] is True


def test_mutating_a_new_opportunitys_future_outcome_cannot_change_its_action():
    frame, _ = _negative_shadow_frame(lessons=46)
    original = _forecast(frame)
    decision = int(np.flatnonzero(original["veto"].to_numpy(dtype=bool))[0])
    changed = frame.copy()
    changed.iloc[decision + 1, changed.columns.get_loc("aapl_open")] *= 3.0
    changed.iloc[decision + 2, changed.columns.get_loc("aapl_open")] *= 0.25
    replay = _forecast(changed)

    predictive_columns = [
        *FEATURE_COLUMNS,
        "n_raw",
        "n_eff",
        "wins",
        "label_sum",
        "squared_label_sum",
        "model_mu",
        "model_se",
        "model_upper",
        "model_ready",
        "veto",
        "learner_cash_signal",
        "target_exposure",
    ]
    pd.testing.assert_series_equal(
        original.iloc[decision][predictive_columns],
        replay.iloc[decision][predictive_columns],
    )
    assert original.iloc[decision + 2]["shadow_label_10bps"] != pytest.approx(
        replay.iloc[decision + 2]["shadow_label_10bps"]
    )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_candidate_input_fails_closed(bad_value):
    frame, positions = _negative_shadow_frame(lessons=2)
    frame.iloc[positions[0], frame.columns.get_loc("spy_adj_close")] = bad_value
    with pytest.raises((ValueError, RuntimeError), match="finite|nonfinite"):
        _forecast(frame)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_model_update_and_prediction_fail_closed(bad_value):
    state = veto_model.initialize_bayesian_veto_state()
    valid = np.zeros(len(FEATURE_COLUMNS), dtype=float)
    valid[0] = 1.0
    bad_features = valid.copy()
    bad_features[-1] = bad_value

    with pytest.raises((ValueError, RuntimeError), match="finite|nonfinite"):
        veto_model.update_bayesian_veto_state(state, bad_features, 0.0)
    with pytest.raises((ValueError, RuntimeError), match="finite|nonfinite"):
        veto_model.update_bayesian_veto_state(state, valid, bad_value)
    with pytest.raises((ValueError, RuntimeError), match="finite|nonfinite"):
        veto_model.predict_bayesian_veto(state, bad_features)


def test_bayesian_state_checkpoint_round_trip_preserves_next_prediction():
    state = veto_model.initialize_bayesian_veto_state()
    for offset in range(6):
        features = np.asarray(
            [
                1.0,
                float(offset % 2),
                float(offset % 3 == 0),
                0.20 + 0.05 * offset,
                -0.30 + 0.04 * offset,
                -0.20 + 0.03 * offset,
                -0.10 + 0.02 * offset,
            ],
            dtype=float,
        )
        state = veto_model.update_bayesian_veto_state(
            state, features, -0.02 + 0.005 * offset
        )
    payload = veto_model.serialize_bayesian_veto_state(state)
    restored = veto_model.restore_bayesian_veto_state(payload)

    np.testing.assert_array_equal(restored.A, state.A)
    np.testing.assert_array_equal(restored.b, state.b)
    for field in (
        "n_raw",
        "n_eff",
        "wins",
        "label_sum",
        "squared_label_sum",
    ):
        assert getattr(restored, field) == getattr(state, field)
    probe = np.asarray([1.0, 1.0, 0.0, 0.7, -0.2, -0.1, 0.05])
    before = veto_model.predict_bayesian_veto(state, probe)
    after = veto_model.predict_bayesian_veto(restored, probe)
    np.testing.assert_array_equal(after.beta, before.beta)
    assert after.mu == before.mu
    assert after.se == before.se
    assert after.upper == before.upper
    assert after.model_ready == before.model_ready
    assert after.veto == before.veto
    assert veto_model.serialize_bayesian_veto_state(restored) == payload


def test_prefix_pending_lesson_is_not_used_until_extension_supplies_t_plus_two():
    frame, positions = _negative_shadow_frame(lessons=3)
    pending = positions[-1]
    truncated = frame.iloc[: pending + 2].copy()
    short = _forecast(truncated)
    full = _forecast(frame)

    assert bool(short.iloc[pending]["union_candidate_signal"])
    assert bool(short.iloc[pending]["shadow_pending"])
    # Core replay preserves the continuous canonical/default-on stream. The
    # bounded runner separately masks unresolved opportunities before ledger
    # execution, while retaining this pending checkpoint context.
    assert bool(short.iloc[pending]["canonical_union_cash_signal"])
    assert bool(short.iloc[pending]["learner_cash_signal"])
    assert short.iloc[pending]["target_exposure"] == 0.0
    assert not bool(short.iloc[pending]["shadow_lesson_added_now"])
    # Rows whose complete t+2 outcomes were present in both prefixes replay
    # identically. Only the two unresolved tail rows may differ.
    replay_columns = [
        *FEATURE_COLUMNS,
        "n_raw",
        "n_eff",
        "wins",
        "label_sum",
        "squared_label_sum",
        "model_mu",
        "model_se",
        "model_upper",
        "model_ready",
        "veto",
        "learner_cash_signal",
        "target_exposure",
    ]
    pd.testing.assert_frame_equal(
        short.iloc[:-2][replay_columns], full.loc[short.index[:-2], replay_columns]
    )
    maturity = pending + 2
    assert bool(full.iloc[maturity]["shadow_matured_now"])
    assert bool(full.iloc[maturity]["shadow_lesson_added_now"])
    assert full.iloc[maturity]["shadow_signal_close"] == frame.index[pending]


def test_repeated_replay_is_bitwise_deterministic():
    frame, _ = _negative_shadow_frame(lessons=46)
    first = _forecast(frame)
    second = _forecast(frame.copy())
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert first.attrs == second.attrs


def test_learning_mode_contract_fails_closed_on_ambiguous_cutoffs():
    frame = _market_frame(140)
    with pytest.raises(ValueError, match="requires.*cutoff|frozen_cutoff"):
        _forecast(frame, learning_mode=veto_model.FROZEN_CUTOFF_MODE)
    with pytest.raises(ValueError, match="must not.*cutoff|frozen_cutoff"):
        _forecast(
            frame,
            learning_mode=veto_model.CAUSAL_ONLINE_MODE,
            frozen_cutoff="2000-06-30",
        )
