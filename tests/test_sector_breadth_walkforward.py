from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

import agent_benchmark.sector_breadth_walkforward as walkforward
from agent_benchmark.direct_edge_features import (
    DIRECT_EDGE_LABEL_COLUMNS,
    PRICE_FEATURE_COLUMNS,
)
from agent_benchmark.sector_breadth_features import SECTOR_BREADTH_FEATURE_COLUMNS


class _FakeDirectEdgeGAM:
    fit_calls: list[dict[str, object]] = []

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
        binary = np.asarray(cash_beats_long_10bps, dtype=float)
        edge = np.asarray(edge_10bps, dtype=float)
        if len(np.unique(binary)) != 2:
            raise ValueError("both classes are required")
        call = {
            "feature_names": self.feature_names,
            "count": len(features),
            "fit_metadata": dict(fit_metadata),
            "binary": binary.copy(),
            "edge": edge.copy(),
        }
        type(self).fit_calls.append(call)
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
        breadth = (
            np.tanh(values[:, -1])
            if values.shape[1] > len(PRICE_FEATURE_COLUMNS)
            else np.zeros(len(values))
        )
        return {
            "cash_beats_long_probability": np.clip(
                0.54 + 0.04 * shared + 0.03 * breadth,
                0.0,
                1.0,
            ),
            "expected_edge_10bps": 0.001 + 0.003 * shared + 0.002 * breadth,
        }

    def to_state(self):
        payload = {
            "feature_names": list(self.feature_names),
            "model_sha256": self._hash,
        }
        return {"payload": payload}

    @classmethod
    def from_state(cls, state):
        model = cls()
        model.feature_names = tuple(state["payload"]["feature_names"])
        model._hash = state["payload"]["model_sha256"]
        return model


def _synthetic_frame() -> pd.DataFrame:
    dates = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.bdate_range(f"{year}-01-03", periods=20).to_numpy()
                for year in range(2000, 2024)
            ]
        ),
        name="decision_date",
    )
    position = np.arange(len(dates), dtype=float)
    data: dict[str, object] = {}
    for feature_index, name in enumerate(PRICE_FEATURE_COLUMNS):
        data[name] = (
            np.sin(position / (5.0 + feature_index))
            + 0.2 * np.cos(position / (13.0 + feature_index))
        )
    for feature_index, name in enumerate(SECTOR_BREADTH_FEATURE_COLUMNS):
        data[name] = (
            np.cos(position / (7.0 + feature_index))
            - 0.15 * np.sin(position / (19.0 + feature_index))
        )

    edge = 0.008 * np.sin(position / 8.0) - 0.0005
    binary = (edge > 0.0).astype(np.int8)
    maturity = pd.Series(dates, index=dates).shift(-6)
    entry = pd.Series(dates, index=dates).shift(-1)
    binary_series = pd.Series(binary, index=dates, dtype="Int8")
    binary_series.iloc[-6:] = pd.NA
    edge[-6:] = np.nan
    data.update(
        {
            "label_entry_date": entry.to_numpy(),
            "label_maturity_date": maturity.to_numpy(),
            "label_entry_adjusted_open": np.where(entry.notna(), 100.0, np.nan),
            "label_exit_adjusted_open": np.where(maturity.notna(), 101.0, np.nan),
            "aapl_forward_log_return_5": np.where(
                maturity.notna(), -np.nan_to_num(edge, nan=0.0), np.nan
            ),
            "cash_active_log_edge_5bps": np.where(
                np.isfinite(edge), edge + 0.001, np.nan
            ),
            "cash_active_log_edge_10bps": edge,
            "cash_beats_long_10bps": binary_series,
            "price_features_ready": np.ones(len(dates), dtype=bool),
            "sector_breadth_features_ready": np.ones(len(dates), dtype=bool),
            "sector_breadth_price_only_fallback": np.zeros(len(dates), dtype=bool),
            "sector_breadth_status": np.full(len(dates), "ready", dtype=object),
            "label_available": maturity.notna().to_numpy(),
        }
    )
    frame = pd.DataFrame(data, index=dates)

    missing_context_dates = [
        frame.index[frame.index.year == year][0] for year in (2005, 2019)
    ]
    frame.loc[
        missing_context_dates, list(SECTOR_BREADTH_FEATURE_COLUMNS)
    ] = np.nan
    frame.loc[missing_context_dates, "sector_breadth_features_ready"] = False
    frame.loc[missing_context_dates, "sector_breadth_price_only_fallback"] = True
    frame.loc[missing_context_dates, "sector_breadth_status"] = "missing_context"
    return frame


@pytest.fixture()
def fake_model(monkeypatch):
    _FakeDirectEdgeGAM.fit_calls = []
    monkeypatch.setattr(walkforward, "DirectEdgeGAM", _FakeDirectEdgeGAM)


def test_seven_expanding_purged_folds_emit_exact_frozen_artifacts(fake_model) -> None:
    frame = _synthetic_frame()
    first = walkforward.build_sector_breadth_walkforward_predictions(frame)
    repeated = walkforward.build_sector_breadth_walkforward_predictions(frame.copy())

    assert tuple(
        fold.fold_id for fold in walkforward.SECTOR_BREADTH_WALK_FORWARD_FOLDS
    ) == (
        "2005-2006",
        "2007-2008",
        "2009-2010",
        "2011-2012",
        "2013-2014",
        "2015-2016",
        "2017-2018",
    )
    assert len(_FakeDirectEdgeGAM.fit_calls) == 28
    assert first.index.min().year == 2005
    assert first.index.max().year == 2018
    pd.testing.assert_frame_equal(first, repeated, check_exact=True)

    prior_count = 0
    for fold_id, rows in first.groupby("fold_id", sort=False):
        boundary = rows["fold_training_boundary"].iloc[0]
        for family in walkforward.SECTOR_BREADTH_MODEL_FAMILIES:
            assert rows[f"{family}_model_sha256"].nunique() == 1
            assert rows[f"{family}_training_sha256"].nunique() == 1
            assert (
                rows[f"{family}_training_max_decision_date"] < boundary
            ).all()
            assert (
                rows[f"{family}_training_max_label_maturity_date"] < boundary
            ).all()
        count = int(rows["price_only_training_count"].iloc[0])
        assert count == int(rows["sector_breadth_training_count"].iloc[0])
        assert count > prior_count
        prior_count = count
        assert rows["fold_id"].eq(fold_id).all()

    first_2005 = first.index[first.index.year == 2005][0]
    assert np.isfinite(first.loc[first_2005, "price_only_cash_win_probability"])
    assert pd.isna(first.loc[first_2005, "sector_breadth_cash_win_probability"])
    for name in DIRECT_EDGE_LABEL_COLUMNS:
        assert name in first


def test_exact_four_gates_and_five_session_binary_policy_are_reused(fake_model) -> None:
    predictions = walkforward.build_sector_breadth_walkforward_predictions(
        _synthetic_frame()
    )
    expected_ids = ("p50_e0", "p55_e0", "p50_e25", "p55_e25")
    assert tuple(
        candidate.candidate_id for candidate in walkforward.DIRECT_EDGE_CANDIDATES
    ) == expected_ids
    for family in walkforward.SECTOR_BREADTH_MODEL_FAMILIES:
        for candidate_id in expected_ids:
            target_name = walkforward.sector_breadth_candidate_cash_target_column(
                family, candidate_id
            )
            start_name = (
                walkforward.sector_breadth_candidate_cash_block_start_column(
                    family, candidate_id
                )
            )
            assert predictions[target_name].dtype == np.int8
            assert predictions[start_name].dtype == np.int8
            walkforward.validate_sector_breadth_binary_exposure(
                predictions, target_columns=(target_name,)
            )


def test_intermediate_is_one_frozen_fit_and_final_cannot_change_selection(
    fake_model,
) -> None:
    frame = _synthetic_frame()
    policy = walkforward.fit_intermediate_sector_breadth_policy(
        frame,
        selected_family="sector_breadth",
        selected_candidate="p55_e25",
    )
    calls_after_fit = len(_FakeDirectEdgeGAM.fit_calls)
    validation = walkforward.predict_intermediate_sector_breadth_validation(
        frame, policy
    )
    repeated = walkforward.predict_intermediate_sector_breadth_validation(
        frame.copy(), policy
    )
    assert len(_FakeDirectEdgeGAM.fit_calls) == calls_after_fit == 1
    pd.testing.assert_frame_equal(validation, repeated, check_exact=True)
    assert validation.index.min().year == 2019
    assert validation.index.max().year == 2023
    assert validation["model_sha256"].nunique() == 1
    assert validation["training_sha256"].nunique() == 1
    assert validation["candidate_id"].eq("sector_breadth_p55_e25").all()
    assert (
        validation["training_max_label_maturity_date"]
        < walkforward.INTERMEDIATE_TRAINING_BOUNDARY
    ).all()
    for name in (
        "baseline_cash_win_probability",
        "baseline_mean_net_edge",
        "cash_beats_long_10bps",
        "cash_active_log_edge_10bps",
        "label_maturity_date",
        "cash_win_probability",
        "expected_net_edge",
        "cash_target",
        "cash_block_start",
    ):
        assert name in validation
    missing_context_date = validation.index[validation.index.year == 2019][0]
    assert pd.isna(validation.loc[missing_context_date, "cash_win_probability"])
    assert pd.isna(validation.loc[missing_context_date, "expected_net_edge"])
    assert validation.loc[missing_context_date, "cash_block_start"] == 0
    assert validation.loc[missing_context_date, "cash_target"] == 0

    final = walkforward.fit_final_sector_breadth_policy(
        frame, selected_policy=policy
    )
    assert len(_FakeDirectEdgeGAM.fit_calls) == 2
    assert final.training_boundary == walkforward.FINAL_TRAINING_BOUNDARY
    assert final.model_family == policy.model_family
    assert final.candidate == policy.candidate
    assert final.training_max_label_maturity_date < pd.Timestamp("2024-01-01")


def test_frozen_policy_state_round_trip_and_checksum_rejection(fake_model) -> None:
    policy = walkforward.fit_intermediate_sector_breadth_policy(
        _synthetic_frame(),
        selected_family="price_only",
        selected_candidate="price_only_p50_e0",
    )
    state = policy.to_state()
    restored = walkforward.FrozenSectorBreadthPolicy.from_state(state)
    assert restored.canonical_json() == policy.canonical_json()
    assert restored.model_sha256 == policy.model_sha256

    tampered = policy.to_state()
    tampered["payload"]["candidate"]["candidate_id"] = "p55_e0"
    with pytest.raises(ValueError, match="checksum"):
        walkforward.FrozenSectorBreadthPolicy.from_state(tampered)


def test_incomplete_development_or_validation_period_is_refused(fake_model) -> None:
    frame = _synthetic_frame()
    missing_development_year = frame.loc[frame.index.year != 2012]
    with pytest.raises(ValueError, match="Incomplete fixed fold 2011-2012"):
        walkforward.build_sector_breadth_walkforward_predictions(
            missing_development_year
        )

    policy = walkforward.fit_intermediate_sector_breadth_policy(
        frame,
        selected_family="price_only",
        selected_candidate="p50_e0",
    )
    missing_validation_year = frame.loc[frame.index.year != 2021]
    with pytest.raises(ValueError, match="Incomplete frozen intermediate"):
        walkforward.predict_intermediate_sector_breadth_validation(
            missing_validation_year, policy
        )


def test_future_decisions_bad_labels_and_nonbinary_exposure_are_refused(
    fake_model,
) -> None:
    frame = _synthetic_frame()
    forbidden = frame.iloc[[-1]].copy()
    forbidden.index = pd.DatetimeIndex(["2024-01-02"], name="decision_date")
    with pytest.raises(ValueError, match="after 2023-12-31"):
        walkforward.fit_intermediate_sector_breadth_policy(
            pd.concat([frame, forbidden]),
            selected_family="price_only",
            selected_candidate="p50_e0",
        )

    nonbinary_label = frame.copy()
    date = nonbinary_label.index[10]
    nonbinary_label["cash_beats_long_10bps"] = nonbinary_label[
        "cash_beats_long_10bps"
    ].astype(float)
    nonbinary_label.loc[date, "cash_beats_long_10bps"] = 0.25
    with pytest.raises(ValueError, match="must be binary"):
        walkforward.build_sector_breadth_walkforward_predictions(nonbinary_label)

    backwards_maturity = frame.copy()
    backwards_maturity.loc[date, "label_maturity_date"] = date
    with pytest.raises(ValueError, match="strictly after"):
        walkforward.build_sector_breadth_walkforward_predictions(
            backwards_maturity
        )

    policy = walkforward.fit_intermediate_sector_breadth_policy(
        frame,
        selected_family="price_only",
        selected_candidate="p50_e0",
    )
    validation = walkforward.predict_intermediate_sector_breadth_validation(
        frame, policy
    )
    validation["cash_target"] = validation["cash_target"].astype(float)
    validation.loc[validation.index[0], "cash_target"] = 0.5
    with pytest.raises(ValueError, match="exactly binary"):
        walkforward.validate_sector_breadth_binary_exposure(
            validation, target_columns=("cash_target",)
        )


def test_boundary_maturity_is_purged_not_leaked(fake_model) -> None:
    frame = _synthetic_frame()
    boundary = pd.Timestamp("2005-01-01")
    candidate_rows = frame.loc[
        (frame.index.year == 2004) & frame["label_available"]
    ]
    date = candidate_rows.index[-1]
    frame.loc[date, "label_maturity_date"] = boundary
    predictions = walkforward.build_sector_breadth_walkforward_predictions(frame)
    first_fold = predictions.loc[predictions["fold_id"] == "2005-2006"]
    expected_count = int(
        (
            (frame.index < boundary)
            & frame["label_available"].to_numpy(dtype=bool)
            & (
                pd.to_datetime(frame["label_maturity_date"]).to_numpy(
                    dtype="datetime64[ns]"
                )
                < boundary.to_datetime64()
            )
            & frame["price_features_ready"].to_numpy(dtype=bool)
        ).sum()
    )
    assert first_fold["price_only_training_count"].iloc[0] == expected_count
    assert (
        first_fold["price_only_training_max_label_maturity_date"] < boundary
    ).all()


def test_available_label_maturing_after_2023_is_refused(fake_model) -> None:
    frame = _synthetic_frame()
    decision = frame.index[frame.index.year == 2023][0]
    frame.loc[decision, "label_maturity_date"] = pd.Timestamp("2024-01-02")

    with pytest.raises(ValueError, match="labels maturing after 2023-12-31"):
        walkforward.build_sector_breadth_walkforward_predictions(frame)


def test_both_model_families_use_identical_common_training_rows(fake_model) -> None:
    frame = _synthetic_frame()
    common_support_gap = frame.index[frame.index.year == 2016][0]
    frame.loc[
        common_support_gap, list(SECTOR_BREADTH_FEATURE_COLUMNS)
    ] = np.nan
    frame.loc[common_support_gap, "sector_breadth_features_ready"] = False
    frame.loc[common_support_gap, "sector_breadth_price_only_fallback"] = True
    frame.loc[common_support_gap, "sector_breadth_status"] = "missing_context"

    predictions = walkforward.build_sector_breadth_walkforward_predictions(frame)
    for _, rows in predictions.groupby("fold_id", sort=False):
        assert (
            rows["price_only_training_count"].iloc[0]
            == rows["sector_breadth_training_count"].iloc[0]
        )
        assert (
            rows["price_only_training_max_decision_date"].iloc[0]
            == rows["sector_breadth_training_max_decision_date"].iloc[0]
        )
        assert (
            rows["price_only_training_max_label_maturity_date"].iloc[0]
            == rows["sector_breadth_training_max_label_maturity_date"].iloc[0]
        )

    price_policy = walkforward.fit_intermediate_sector_breadth_policy(
        frame,
        selected_family="price_only",
        selected_candidate="p50_e0",
    )
    breadth_policy = walkforward.fit_intermediate_sector_breadth_policy(
        frame,
        selected_family="sector_breadth",
        selected_candidate="p50_e0",
    )
    assert price_policy.training_count == breadth_policy.training_count
    assert (
        price_policy.training_max_decision_date
        == breadth_policy.training_max_decision_date
    )
    assert (
        price_policy.training_max_label_maturity_date
        == breadth_policy.training_max_label_maturity_date
    )
