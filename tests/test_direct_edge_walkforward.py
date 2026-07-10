from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

import agent_benchmark.direct_edge_walkforward as walkforward
from agent_benchmark.direct_edge_features import (
    DIRECT_EDGE_LABEL_COLUMNS,
    MARKET_SENTIMENT_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    price_only_chronological_training_mask,
)


class _FakeDirectEdgeGAM:
    fail_sentiment_before_2007 = False

    def __init__(self) -> None:
        self.feature_names: tuple[str, ...] = ()
        self._hash = ""

    @property
    def model_sha256(self) -> str:
        return self._hash

    def fit(
        self,
        features,
        cash_beats_long_10bps,
        edge_10bps,
        *,
        feature_names=None,
        fit_metadata,
    ):
        self.feature_names = tuple(feature_names or features.columns)
        if (
            self.fail_sentiment_before_2007
            and len(self.feature_names)
            == len(PRICE_FEATURE_COLUMNS) + len(MARKET_SENTIMENT_FEATURE_COLUMNS)
            and fit_metadata["training_end_date"] < "2007-01-01"
        ):
            raise ValueError("synthetic insufficient sentiment fit")
        binary = np.asarray(cash_beats_long_10bps, dtype=float)
        edge = np.asarray(edge_10bps, dtype=float)
        if not len(binary) or len(np.unique(binary)) != 2:
            raise ValueError("both classes are required")
        material = "|".join(
            [
                *self.feature_names,
                *(f"{key}={fit_metadata[key]}" for key in sorted(fit_metadata)),
                f"n={len(binary)}",
                f"p={binary.mean().hex()}",
                f"e={edge.mean().hex()}",
            ]
        )
        self._hash = hashlib.sha256(material.encode("ascii")).hexdigest()
        return self

    def predict_components(self, features):
        values = features.to_numpy(dtype=float)
        shared = np.tanh(values[:, 0])
        sentiment = (
            np.tanh(values[:, -1])
            if values.shape[1] > len(PRICE_FEATURE_COLUMNS)
            else np.zeros(len(values))
        )
        return {
            "cash_beats_long_probability": np.clip(
                0.53 + 0.04 * shared + 0.03 * sentiment,
                0.0,
                1.0,
            ),
            "expected_edge_10bps": 0.001 + 0.003 * shared + 0.002 * sentiment,
        }


def _synthetic_frame() -> pd.DataFrame:
    dates = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.bdate_range(f"{year}-01-03", periods=36).to_numpy()
                for year in range(2000, 2019)
            ]
        ),
        name="decision_date",
    )
    position = np.arange(len(dates), dtype=float)
    data: dict[str, object] = {}
    for feature_index, name in enumerate(PRICE_FEATURE_COLUMNS):
        data[name] = (
            np.sin(position / (5.0 + feature_index))
            + 0.25 * np.cos(position / (17.0 + feature_index))
        )
    for feature_index, name in enumerate(MARKET_SENTIMENT_FEATURE_COLUMNS):
        data[name] = (
            np.cos(position / (7.0 + feature_index))
            - 0.20 * np.sin(position / (23.0 + feature_index))
        )

    edge = 0.008 * np.sin(position / 9.0) - 0.001
    binary = (edge > 0.0).astype(np.int8)
    maturity = pd.Series(dates, index=dates).shift(-6)
    binary_series = pd.Series(binary, index=dates, dtype="Int8")
    binary_series.iloc[-6:] = pd.NA
    edge[-6:] = np.nan
    entry_date = pd.Series(dates, index=dates).shift(-1)
    data.update(
        {
            "label_entry_date": entry_date.to_numpy(),
            "label_maturity_date": maturity.to_numpy(),
            "label_entry_adjusted_open": np.full(len(dates), 100.0),
            "label_exit_adjusted_open": np.full(len(dates), 101.0),
            "aapl_forward_log_return_5": -np.nan_to_num(edge, nan=0.0),
            "cash_active_log_edge_5bps": np.where(np.isfinite(edge), edge + 0.001, np.nan),
            "cash_active_log_edge_10bps": edge,
            "cash_beats_long_10bps": binary_series,
            "price_features_ready": np.ones(len(dates), dtype=bool),
            "sentiment_features_ready": np.ones(len(dates), dtype=bool),
            "sentiment_price_only_fallback": np.zeros(len(dates), dtype=bool),
        }
    )
    frame = pd.DataFrame(data, index=dates)

    first_2005 = frame.index[frame.index.year == 2005][0]
    last_2004 = frame.index[frame.index.year == 2004][-1]
    frame.loc[last_2004, "label_maturity_date"] = first_2005

    # Historical sentiment gaps make its causal baseline genuinely different.
    historical_gap = (frame.index.year == 2001) & (
        frame["cash_beats_long_10bps"].fillna(0).to_numpy(dtype=int) == 1
    )
    frame.loc[historical_gap, "sentiment_features_ready"] = False
    frame.loc[historical_gap, "sentiment_price_only_fallback"] = True
    frame.loc[historical_gap, list(MARKET_SENTIMENT_FEATURE_COLUMNS)] = np.nan

    first_position = int(np.flatnonzero(frame.index == first_2005)[0])
    frame.iloc[
        first_position, frame.columns.get_loc(PRICE_FEATURE_COLUMNS[0])
    ] = np.nan
    frame.iloc[
        first_position, frame.columns.get_loc("price_features_ready")
    ] = False
    frame.iloc[
        first_position + 1,
        frame.columns.get_indexer(list(MARKET_SENTIMENT_FEATURE_COLUMNS)),
    ] = np.nan
    frame.iloc[
        first_position + 1,
        frame.columns.get_loc("sentiment_features_ready"),
    ] = False
    frame.iloc[
        first_position + 1,
        frame.columns.get_loc("sentiment_price_only_fallback"),
    ] = True
    return frame


@pytest.fixture()
def fake_model(monkeypatch):
    _FakeDirectEdgeGAM.fail_sentiment_before_2007 = False
    monkeypatch.setattr(walkforward, "DirectEdgeGAM", _FakeDirectEdgeGAM)


@pytest.fixture()
def synthetic_predictions(fake_model):
    frame = _synthetic_frame()
    first = walkforward.build_pre2019_direct_edge_predictions(frame)
    repeated = walkforward.build_pre2019_direct_edge_predictions(frame.copy())
    return frame, first, repeated


def test_strict_fold_purge_and_models_frozen_inside_each_block(
    synthetic_predictions,
) -> None:
    frame, predictions, _ = synthetic_predictions
    assert tuple(fold.fold_id for fold in walkforward.WALK_FORWARD_FOLDS) == (
        "2005-2006",
        "2007-2008",
        "2009-2010",
        "2011-2012",
        "2013-2014",
        "2015-2016",
        "2017-2018",
    )
    for fold_id, rows in predictions.groupby("fold_id", sort=False):
        first_decision = rows.index.min()
        assert (rows["price_only_training_max_decision_date"] < first_decision).all()
        assert (
            rows["price_only_training_max_label_maturity_date"] < first_decision
        ).all()
        expected = price_only_chronological_training_mask(
            frame, as_of_date=first_decision
        )
        assert rows["price_only_training_count"].iloc[0] == int(expected.sum())
        for family in walkforward.MODEL_FAMILIES:
            assert rows[f"{family}_model_sha256"].nunique() == 1
            assert rows[f"{family}_training_count"].nunique() == 1
        assert rows["fold_id"].eq(fold_id).all()


def test_row_fallback_copies_predictions_hash_and_causal_baselines_exactly(
    synthetic_predictions,
) -> None:
    _, predictions, _ = synthetic_predictions
    first = predictions.index[predictions.index.year == 2005][0]
    fallback_date = predictions.index[predictions.index.get_loc(first) + 1]
    assert predictions.loc[fallback_date, "market_sentiment_used_price_fallback"]
    assert (
        predictions.loc[fallback_date, "market_sentiment_cash_win_probability"]
        == predictions.loc[fallback_date, "price_only_cash_win_probability"]
    )
    assert (
        predictions.loc[fallback_date, "market_sentiment_expected_net_edge"]
        == predictions.loc[fallback_date, "price_only_expected_net_edge"]
    )
    assert (
        predictions.loc[
            fallback_date, "market_sentiment_prediction_model_sha256"
        ]
        == predictions.loc[fallback_date, "price_only_model_sha256"]
    )
    assert (
        predictions.loc[
            fallback_date, "market_sentiment_baseline_cash_win_probability"
        ]
        == predictions.loc[
            fallback_date, "price_only_baseline_cash_win_probability"
        ]
    )
    assert (
        predictions.loc[fallback_date, "market_sentiment_baseline_mean_net_edge"]
        == predictions.loc[fallback_date, "price_only_baseline_mean_net_edge"]
    )

    ready_date = predictions.index[predictions.index.get_loc(first) + 2]
    assert not predictions.loc[ready_date, "market_sentiment_used_price_fallback"]
    assert (
        predictions.loc[
            ready_date, "market_sentiment_baseline_cash_win_probability"
        ]
        != predictions.loc[ready_date, "price_only_baseline_cash_win_probability"]
    )


def test_missing_price_means_long_and_all_candidate_ids_are_exact(
    synthetic_predictions,
) -> None:
    _, predictions, _ = synthetic_predictions
    first = predictions.index[predictions.index.year == 2005][0]
    assert pd.isna(predictions.loc[first, "price_only_cash_win_probability"])
    assert pd.isna(predictions.loc[first, "market_sentiment_cash_win_probability"])
    expected_ids = ("p50_e0", "p55_e0", "p50_e25", "p55_e25")
    assert tuple(item.candidate_id for item in walkforward.DIRECT_EDGE_CANDIDATES) == expected_ids
    for family in walkforward.MODEL_FAMILIES:
        for candidate_id in expected_ids:
            column = walkforward.candidate_cash_target_column(family, candidate_id)
            start_column = walkforward.candidate_cash_block_start_column(
                family, candidate_id
            )
            assert predictions.loc[first, column] == 0
            assert predictions.loc[first, start_column] == 0
            assert predictions[column].dtype == np.int8
            assert predictions[start_column].dtype == np.int8


def test_failed_sentiment_fit_routes_the_whole_fold_to_price(fake_model) -> None:
    _FakeDirectEdgeGAM.fail_sentiment_before_2007 = True
    predictions = walkforward.build_pre2019_direct_edge_predictions(_synthetic_frame())
    rows = predictions.loc[predictions["fold_id"] == "2005-2006"]
    assert rows["market_sentiment_whole_fold_fallback"].all()
    assert rows["market_sentiment_fitted_model_sha256"].eq("").all()
    assert (
        rows["market_sentiment_model_sha256"]
        == rows["price_only_model_sha256"]
    ).all()
    np.testing.assert_array_equal(
        rows["market_sentiment_cash_win_probability"].to_numpy(),
        rows["price_only_cash_win_probability"].to_numpy(),
    )
    np.testing.assert_array_equal(
        rows["market_sentiment_expected_net_edge"].to_numpy(),
        rows["price_only_expected_net_edge"].to_numpy(),
    )
    np.testing.assert_array_equal(
        rows["market_sentiment_baseline_cash_win_probability"].to_numpy(),
        rows["price_only_baseline_cash_win_probability"].to_numpy(),
    )
    assert rows["market_sentiment_attempted_training_count"].iloc[0] > 0


def test_cash_state_is_continuous_and_non_overlapping() -> None:
    triggers = [
        True,
        True,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ]
    target, starts = walkforward.five_session_cash_policy(triggers)
    assert target.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    assert starts.tolist() == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    np.testing.assert_array_equal(
        walkforward.five_session_cash_target(triggers), target
    )
    assert walkforward.five_session_cash_target(
        [False, None, pd.NA, True, False, False, False, False]
    ).tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="booleans"):
        walkforward.five_session_cash_target([False, 1, False])


def test_outputs_hashes_targets_and_diagnostics_are_deterministic(
    synthetic_predictions,
) -> None:
    _, first, repeated = synthetic_predictions
    pd.testing.assert_frame_equal(first, repeated, check_exact=True)
    assert first.index.min().year == 2005
    assert first.index.max().year == 2018
    for name in DIRECT_EDGE_LABEL_COLUMNS:
        assert name in first.columns
    assert first["price_only_model_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert (
        first.loc[
            ~first["market_sentiment_used_price_fallback"],
            "market_sentiment_prediction_model_sha256",
        ]
        == first.loc[
            ~first["market_sentiment_used_price_fallback"],
            "market_sentiment_model_sha256",
        ]
    ).all()


def test_walkforward_exports_exact_sentiment_readiness_for_common_support(
    synthetic_predictions,
) -> None:
    from agent_benchmark.direct_edge_experiment import _common_support_mask

    frame, predictions, _ = synthetic_predictions
    expected = frame.loc[predictions.index, "sentiment_features_ready"].astype(bool)
    pd.testing.assert_series_equal(
        predictions["sentiment_features_ready"],
        expected.rename("sentiment_features_ready"),
        check_exact=True,
    )
    support = _common_support_mask(predictions)
    assert support.shape == (len(predictions),)
    assert support.dtype == bool


def test_post_2018_decisions_are_refused(fake_model) -> None:
    frame = _synthetic_frame()
    forbidden = frame.iloc[[-1]].copy()
    forbidden.index = pd.DatetimeIndex(["2019-01-02"], name="decision_date")
    with pytest.raises(ValueError, match="post-2018"):
        walkforward.build_pre2019_direct_edge_predictions(
            pd.concat([frame, forbidden])
        )
