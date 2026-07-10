from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.downside_features import (
    CFTC_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    chronological_training_mask,
)
from agent_benchmark.downside_walkforward import (
    MODEL_FAMILIES,
    RISK_MULTIPLE_GATES,
    WALK_FORWARD_FOLDS,
    build_pre2019_walkforward_predictions,
    candidate_cash_target_column,
    five_session_cash_target,
)


def _synthetic_feature_label_frame() -> pd.DataFrame:
    dates = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.bdate_range(f"{year}-01-03", periods=80).to_numpy()
                for year in range(2000, 2019)
            ]
        )
    )
    position = np.arange(len(dates), dtype=float)
    data: dict[str, object] = {}
    for feature_index, name in enumerate(PRICE_FEATURE_COLUMNS):
        data[name] = (
            np.sin(position / (7.0 + feature_index))
            + 0.35 * np.cos(position / (19.0 + 2.0 * feature_index))
            + 0.0002 * feature_index * position
        )
    for feature_index, name in enumerate(CFTC_FEATURE_COLUMNS):
        data[name] = (
            np.cos(position / (13.0 + 3.0 * feature_index))
            - 0.25 * np.sin(position / (31.0 + feature_index))
        )

    future_return = 0.011 + 0.003 * np.sin(position / 17.0)
    price_zero = np.asarray(data[PRICE_FEATURE_COLUMNS[0]], dtype=float)
    price_one = np.asarray(data[PRICE_FEATURE_COLUMNS[1]], dtype=float)
    downside = (price_zero > 0.75) & (price_one < 0.55)
    future_return[downside] = -0.062 - 0.004 * np.cos(position[downside])
    maturity = pd.Series(dates, index=dates).shift(-6)
    crash = pd.Series(
        (future_return <= np.log(0.96)).astype(np.int8),
        index=dates,
        dtype="Int8",
    )
    crash.iloc[-6:] = pd.NA
    future_return[-6:] = np.nan

    data.update(
        {
            "cftc_price_only_fallback": np.zeros(len(dates), dtype=bool),
            "label_maturity_date": maturity.to_numpy(),
            "crash_label_5session": crash,
            "aapl_forward_log_return_5": future_return,
        }
    )
    frame = pd.DataFrame(data, index=dates)
    frame.index.name = "decision_date"

    first_fill = frame.index[frame.index.year == 2005][0]
    last_2004 = frame.index[frame.index.year == 2004][-1]
    # This case is only revealed on the first fill date and must be purged.
    frame.loc[last_2004, "label_maturity_date"] = first_fill

    first_position = int(np.flatnonzero(frame.index == first_fill)[0])
    # Claimed-ready but corrupt CFTC data: never impute, never fall back.
    frame.iloc[
        first_position, frame.columns.get_loc(CFTC_FEATURE_COLUMNS[0])
    ] = np.nan
    # Explicit fallback: all three unavailable values must copy price exactly.
    frame.iloc[
        first_position + 1,
        frame.columns.get_indexer(list(CFTC_FEATURE_COLUMNS)),
    ] = np.nan
    frame.iloc[
        first_position + 1,
        frame.columns.get_loc("cftc_price_only_fallback"),
    ] = True
    # A missing price feature must make both predictions unavailable.
    frame.iloc[
        first_position + 2,
        frame.columns.get_loc(PRICE_FEATURE_COLUMNS[0]),
    ] = np.nan
    return frame


@pytest.fixture(scope="module")
def synthetic_run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = _synthetic_feature_label_frame()
    first = build_pre2019_walkforward_predictions(frame)
    repeated = build_pre2019_walkforward_predictions(frame.copy())
    return frame, first, repeated


def test_fixed_folds_use_strictly_purged_training_rows(synthetic_run) -> None:
    frame, predictions, _ = synthetic_run
    assert tuple(fold.fold_id for fold in WALK_FORWARD_FOLDS) == (
        "2005-2006",
        "2007-2008",
        "2009-2010",
        "2011-2012",
        "2013-2014",
        "2015-2016",
        "2017-2018",
    )
    for fold in WALK_FORWARD_FOLDS:
        rows = predictions.loc[predictions["fold_id"] == fold.fold_id]
        first_decision = rows.index.min()
        assert (
            rows["price_only_training_max_label_maturity_date"] < first_decision
        ).all()
        assert (rows["price_only_training_max_decision_date"] < first_decision).all()
        expected = chronological_training_mask(
            frame,
            as_of_date=first_decision,
            strictly_before_as_of=True,
        )
        expected &= np.isfinite(frame["aapl_forward_log_return_5"])
        assert rows["price_only_training_count"].iloc[0] == int(expected.sum())

    first_2005 = predictions.index[predictions.index.year == 2005][0]
    assert (
        predictions.loc[first_2005, "price_only_training_max_label_maturity_date"]
        < first_2005
    )


def test_models_are_never_updated_inside_a_fill_block(synthetic_run) -> None:
    _, predictions, _ = synthetic_run
    for _, rows in predictions.groupby("fold_id", sort=False):
        for family in MODEL_FAMILIES:
            assert rows[f"{family}_model_sha256"].nunique() == 1
            assert rows[f"{family}_training_count"].nunique() == 1
            assert rows[f"{family}_training_max_decision_date"].nunique() == 1
            assert rows[f"{family}_training_max_label_maturity_date"].nunique() == 1
            assert rows[f"{family}_baseline_downside_probability"].nunique() == 1


def test_explicit_cftc_fallback_is_exact_price_identity(synthetic_run) -> None:
    frame, predictions, _ = synthetic_run
    first_2005 = predictions.index[predictions.index.year == 2005][0]
    fallback_date = predictions.index[
        predictions.index.get_loc(first_2005) + 1
    ]
    assert frame.loc[fallback_date, "cftc_price_only_fallback"]
    assert frame.loc[fallback_date, list(CFTC_FEATURE_COLUMNS)].isna().all()
    assert predictions.loc[fallback_date, "price_cftc_used_fallback"]
    assert (
        predictions.loc[fallback_date, "price_cftc_downside_probability"]
        == predictions.loc[fallback_date, "price_only_downside_probability"]
    )
    assert (
        predictions.loc[
            fallback_date, "price_cftc_predicted_mean_clipped_return"
        ]
        == predictions.loc[
            fallback_date, "price_only_predicted_mean_clipped_return"
        ]
    )
    assert (
        predictions.loc[
            fallback_date,
            "price_cftc_effective_baseline_downside_probability",
        ]
        == predictions.loc[
            fallback_date, "price_only_baseline_downside_probability"
        ]
    )


def test_insufficient_early_cftc_training_falls_back_without_weakening_model() -> None:
    frame = _synthetic_feature_label_frame()
    early = frame.index < pd.Timestamp("2005-01-03")
    frame.loc[early, list(CFTC_FEATURE_COLUMNS)] = np.nan
    frame.loc[early, "cftc_price_only_fallback"] = True

    predictions = build_pre2019_walkforward_predictions(frame)
    first_fold = predictions.loc[predictions["fold_id"] == "2005-2006"]

    assert (first_fold["price_cftc_model_status"] == (
        "insufficient_training_price_fallback"
    )).all()
    assert first_fold["price_cftc_model_sha256"].isna().all()
    assert first_fold["price_cftc_used_fallback"].all()
    assert (
        first_fold["price_cftc_fallback_reason"]
        == "insufficient_fold_training"
    ).all()
    np.testing.assert_array_equal(
        first_fold["price_cftc_downside_probability"].to_numpy(),
        first_fold["price_only_downside_probability"].to_numpy(),
    )
    np.testing.assert_array_equal(
        first_fold["price_cftc_effective_baseline_downside_probability"].to_numpy(),
        first_fold["price_only_baseline_downside_probability"].to_numpy(),
    )


def test_missing_values_are_never_treated_as_zero(synthetic_run) -> None:
    _, predictions, _ = synthetic_run
    first_2005 = predictions.index[predictions.index.year == 2005][0]
    # This row says CFTC is ready but contains NaN.  The augmented model must
    # remain unavailable, not receive zero and not silently use price-only.
    assert not predictions.loc[first_2005, "price_cftc_used_fallback"]
    assert np.isfinite(
        predictions.loc[first_2005, "price_only_downside_probability"]
    )
    assert pd.isna(
        predictions.loc[first_2005, "price_cftc_downside_probability"]
    )
    for gate in RISK_MULTIPLE_GATES:
        assert predictions.loc[
            first_2005, candidate_cash_target_column("price_cftc", gate)
        ] == 0

    missing_price_date = predictions.index[
        predictions.index.get_loc(first_2005) + 2
    ]
    assert pd.isna(
        predictions.loc[missing_price_date, "price_only_downside_probability"]
    )
    assert pd.isna(
        predictions.loc[missing_price_date, "price_cftc_downside_probability"]
    )


def test_cash_state_machine_is_one_continuous_non_overlapping_sequence() -> None:
    trigger = np.array(
        [True, True, False, True, False, True, False, False, False, True, False],
        dtype=object,
    )
    # Triggers at 1 and 3 are ignored during the first episode.  Row 5 can
    # begin the next episode immediately after the first five rows complete.
    assert five_session_cash_target(trigger).tolist() == [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
    ]
    assert five_session_cash_target(
        np.array([False, None, pd.NA, True, False, False, False, False], dtype=object)
    ).tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="booleans"):
        five_session_cash_target([False, 1, False])


def test_outputs_and_model_hashes_are_deterministic(synthetic_run) -> None:
    _, first, repeated = synthetic_run
    pd.testing.assert_frame_equal(first, repeated, check_exact=True)
    assert first.index.min().year == 2005
    assert first.index.max().year == 2018
    for family in MODEL_FAMILIES:
        assert first[f"{family}_model_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
        for gate in RISK_MULTIPLE_GATES:
            target = first[candidate_cash_target_column(family, gate)]
            assert target.dtype == np.int8
            assert set(target.unique()).issubset({0, 1})


def test_any_post_2018_decision_row_is_rejected() -> None:
    frame = _synthetic_feature_label_frame()
    forbidden = frame.iloc[[-1]].copy()
    forbidden.index = pd.DatetimeIndex(["2019-01-02"], name="decision_date")
    contaminated = pd.concat([frame, forbidden])
    with pytest.raises(ValueError, match="post-2018"):
        build_pre2019_walkforward_predictions(contaminated)
