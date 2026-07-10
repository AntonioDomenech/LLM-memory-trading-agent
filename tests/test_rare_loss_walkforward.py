from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

import agent_benchmark.rare_loss_walkforward as walkforward
from agent_benchmark.rare_loss_features import (
    CASH_EDGE_SLIPPAGE_BPS,
    CORE_FEATURE_COLUMNS,
    RARE_LOSS_FEATURE_COLUMNS,
    REGRESSION_EDGE_CLIP,
    SENTIMENT_FEATURE_COLUMNS,
    SEVERE_SIMPLE_RETURN_THRESHOLD,
    rare_loss_feature_readiness,
)
from agent_benchmark.rare_loss_forest import MINIMUM_OOB_TREE_COUNT


class _FakeRareLossForest:
    """Fast deterministic stand-in; the forest itself has focused real tests."""

    instances: list["_FakeRareLossForest"] = []
    force_low_oob = False

    def __init__(self) -> None:
        self.namespace = ""
        self.feature_names: tuple[str, ...] = ()
        self.fit_metadata: dict[str, Any] = {}
        self.training_index = pd.DatetimeIndex([])
        self._oob_counts = np.empty(0, dtype=np.int64)
        self._oob_severe = np.empty(0, dtype=float)
        self._hash = ""

    @property
    def model_sha256(self) -> str:
        return self._hash

    def fit(
        self,
        features,
        severe_labels,
        ordinary_labels,
        clipped_edges,
        *,
        feature_names=None,
        namespace,
        fit_metadata,
    ):
        assert feature_names is None
        matrix = features.to_numpy(dtype=float)
        assert np.isfinite(matrix).all()
        severe = np.asarray(severe_labels, dtype=float)
        ordinary = np.asarray(ordinary_labels, dtype=float)
        edge = np.asarray(clipped_edges, dtype=float)
        assert len(features) == len(severe) == len(ordinary) == len(edge)
        assert np.isin(severe, (0.0, 1.0)).all()
        assert np.isin(ordinary, (0.0, 1.0)).all()
        assert np.isfinite(edge).all()
        self.namespace = str(namespace)
        self.feature_names = tuple(str(column) for column in features.columns)
        self.fit_metadata = dict(fit_metadata)
        self.training_index = pd.DatetimeIndex(features.index)
        count = len(features)
        self._oob_counts = np.full(count, MINIMUM_OOB_TREE_COUNT + 4, dtype=np.int64)
        if self.force_low_oob:
            self._oob_counts[0] = MINIMUM_OOB_TREE_COUNT - 1
        # Increasing values make the frozen higher-quantile rule exact and easy
        # to audit without coupling these tests to tree construction internals.
        self._oob_severe = np.linspace(0.05, 0.45, count, dtype=float)
        material = json.dumps(
            {
                "namespace": self.namespace,
                "feature_names": self.feature_names,
                "fit_metadata": self.fit_metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._hash = hashlib.sha256(material).hexdigest()
        self.instances.append(self)
        return self

    def oob_components(self) -> dict[str, np.ndarray]:
        count = len(self._oob_counts)
        return {
            "severe_probability": self._oob_severe.copy(),
            "ordinary_cash_win_probability": np.full(count, 0.60),
            "expected_clipped_edge": np.full(count, 0.002),
            "oob_tree_count": self._oob_counts.copy(),
        }

    def oob_severe_threshold(self, quantile: float) -> float:
        eligible = self._oob_counts >= MINIMUM_OOB_TREE_COUNT
        return float(
            np.quantile(self._oob_severe[eligible], quantile, method="higher")
        )

    def predict_components(self, features) -> dict[str, np.ndarray]:
        values = features.to_numpy(dtype=float)
        available = np.isfinite(values).all(axis=1)
        count = len(features)
        return {
            "severe_probability": np.full(count, 0.80),
            "ordinary_cash_win_probability": np.full(count, 0.70),
            "expected_clipped_edge": np.full(count, 0.01),
            "observation_available": available,
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "feature_names": list(self.feature_names),
            "fit_metadata": self.fit_metadata,
            "model_sha256": self._hash,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_state(), sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "_FakeRareLossForest":
        model = object.__new__(cls)
        model.namespace = str(state["namespace"])
        model.feature_names = tuple(state["feature_names"])
        model.fit_metadata = dict(state["fit_metadata"])
        model.training_index = pd.DatetimeIndex([])
        model._oob_counts = np.empty(0, dtype=np.int64)
        model._oob_severe = np.empty(0, dtype=float)
        model._hash = str(state["model_sha256"])
        return model


def _synthetic_feature_label_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2000-01-03", "2018-12-31", name="decision_date")
    position = np.arange(len(dates), dtype=float)
    frame = pd.DataFrame(index=dates)
    for feature_number, column in enumerate(RARE_LOSS_FEATURE_COLUMNS, start=1):
        frame[column] = (
            np.sin(position / (5.0 + feature_number))
            + 0.2 * np.cos(position / (13.0 + feature_number))
            + feature_number / 100.0
        )

    entry = pd.Series(dates, index=dates).shift(-1)
    maturity = pd.Series(dates, index=dates).shift(-2)
    simple_return = np.where(
        (position.astype(int) % 10) == 0,
        -0.04,
        np.where((position.astype(int) % 3) == 0, -0.01, 0.006),
    ).astype(float)
    simple_return[-2:] = np.nan
    log_return = np.log1p(simple_return)
    cost = math.log1p(-CASH_EDGE_SLIPPAGE_BPS / 10_000.0) - math.log1p(
        CASH_EDGE_SLIPPAGE_BPS / 10_000.0
    )
    active_edge = cost - log_return
    entry_open = np.where(entry.notna(), 100.0, np.nan)
    exit_open = np.where(maturity.notna(), 100.0 * (1.0 + simple_return), np.nan)
    severe = pd.Series(
        (np.nan_to_num(simple_return) <= SEVERE_SIMPLE_RETURN_THRESHOLD).astype(
            np.int8
        ),
        index=dates,
        dtype="Int8",
    )
    ordinary = pd.Series(
        (np.nan_to_num(active_edge, nan=-1.0) > 0.0).astype(np.int8),
        index=dates,
        dtype="Int8",
    )
    severe.iloc[-2:] = pd.NA
    ordinary.iloc[-2:] = pd.NA
    frame["label_entry_date"] = entry
    frame["label_maturity_date"] = maturity
    frame["label_entry_adjusted_open"] = entry_open
    frame["label_exit_adjusted_open"] = exit_open
    frame["aapl_forward_simple_return_1"] = simple_return
    frame["aapl_forward_log_return_1"] = log_return
    frame["severe_loss_label_1session"] = severe
    frame["cash_active_log_edge_10bps"] = active_edge
    frame["cash_beats_long_10bps"] = ordinary
    frame["cash_active_log_edge_10bps_clipped"] = np.clip(
        active_edge, -REGRESSION_EDGE_CLIP, REGRESSION_EDGE_CLIP
    )

    missing_date = pd.Timestamp("2005-06-15")
    frame.loc[missing_date, SENTIMENT_FEATURE_COLUMNS[0]] = np.nan
    frame["features_ready"] = rare_loss_feature_readiness(frame)
    frame["label_available"] = (
        frame["label_maturity_date"].notna()
        & frame["cash_active_log_edge_10bps_clipped"].notna()
    )
    frame["training_ready"] = frame["features_ready"] & frame["label_available"]
    frame["entry_fill_year"] = pd.to_datetime(frame["label_entry_date"]).dt.year.astype(
        "Int16"
    )
    return frame


@pytest.fixture()
def fake_forest(monkeypatch: pytest.MonkeyPatch):
    _FakeRareLossForest.instances = []
    _FakeRareLossForest.force_low_oob = False
    monkeypatch.setattr(walkforward, "RareLossForest", _FakeRareLossForest)


@pytest.fixture()
def walkforward_result(fake_forest):
    frame = _synthetic_feature_label_frame()
    result = walkforward.build_pre2019_rare_loss_walkforward(frame)
    instances = list(_FakeRareLossForest.instances)
    return frame, result, instances


def test_frozen_folds_candidates_variants_and_gates_are_exact() -> None:
    assert tuple(fold.fold_id for fold in walkforward.WALK_FORWARD_FOLDS) == (
        "2005-2006",
        "2007-2008",
        "2009-2010",
        "2011-2012",
        "2013-2014",
        "2015-2016",
        "2017-2018",
    )
    assert walkforward.MODEL_VARIANTS == ("full", "core")
    assert [
        (candidate.candidate_id, candidate.oob_tail_quantile)
        for candidate in walkforward.RARE_LOSS_CANDIDATES
    ] == [("tail975", 0.975), ("tail990", 0.99)]
    assert walkforward.SEVERE_PREVALENCE_MULTIPLIER == 2.0
    assert walkforward.ORDINARY_PROBABILITY_GATE == 0.55
    assert walkforward.EXPECTED_EDGE_GATE_10BPS == 0.001
    with pytest.raises(ValueError, match="Unknown rare-loss variant"):
        walkforward.candidate_cash_target_column("price", "tail975")
    with pytest.raises(ValueError, match="Unknown rare-loss candidate"):
        walkforward.candidate_cash_target_column("full", "tail950")


def test_post_2018_decision_or_target_date_and_bad_entry_year_are_refused(
    fake_forest,
) -> None:
    frame = _synthetic_feature_label_frame()
    future = frame.reindex(
        frame.index.append(pd.DatetimeIndex([pd.Timestamp("2019-01-02")]))
    )
    with pytest.raises(ValueError, match="post-2018 rows"):
        walkforward.build_pre2019_rare_loss_walkforward(future)

    future_label = frame.copy()
    future_label.loc[future_label.index[-3], "label_maturity_date"] = pd.Timestamp(
        "2019-01-02"
    )
    with pytest.raises(ValueError, match="post-2018 label_maturity_date"):
        walkforward.build_pre2019_rare_loss_walkforward(future_label)

    wrong_year = frame.copy()
    wrong_year.loc[wrong_year.index[1000], "entry_fill_year"] += 1
    with pytest.raises(ValueError, match="entry_fill_year is inconsistent"):
        walkforward.build_pre2019_rare_loss_walkforward(wrong_year)


def test_training_is_strictly_mature_and_identical_for_full_and_core(
    walkforward_result,
) -> None:
    frame, result, instances = walkforward_result
    assert len(instances) == 14
    by_namespace = {model.namespace: model for model in instances}
    missing_date = pd.Timestamp("2005-06-15")
    for fold in walkforward.WALK_FORWARD_FOLDS:
        fold_rows = result.predictions.loc[
            result.predictions["fold_id"] == fold.fold_id
        ]
        first_decision = fold_rows.index.min()
        full = by_namespace[f"{fold.fold_id}|full"]
        core = by_namespace[f"{fold.fold_id}|core"]
        assert full.training_index.equals(core.training_index)
        assert (full.training_index < first_decision).all()
        maturity = pd.to_datetime(
            frame.loc[full.training_index, "label_maturity_date"]
        )
        assert (maturity < first_decision).all()
        assert full.fit_metadata["maximum_label_maturity_date"] == (
            maturity.max().date().isoformat()
        )
        if fold.first_entry_year >= 2007:
            assert missing_date not in full.training_index
        assert full.feature_names == RARE_LOSS_FEATURE_COLUMNS
        assert core.feature_names == CORE_FEATURE_COLUMNS


def test_full_and_core_prediction_support_is_identical_and_missing_forces_long(
    walkforward_result,
) -> None:
    _, result, _ = walkforward_result
    predictions = result.predictions
    full_available = predictions["full_severe_probability"].notna().to_numpy()
    core_available = predictions["core_severe_probability"].notna().to_numpy()
    np.testing.assert_array_equal(full_available, core_available)
    np.testing.assert_array_equal(
        full_available,
        predictions["features_ready"].to_numpy(dtype=bool),
    )
    missing_date = pd.Timestamp("2005-06-15")
    assert missing_date in predictions.index
    assert not predictions.loc[missing_date, "features_ready"]
    for variant in walkforward.MODEL_VARIANTS:
        for candidate in walkforward.RARE_LOSS_CANDIDATES:
            column = walkforward.candidate_cash_target_column(
                variant, candidate.candidate_id
            )
            assert predictions.loc[missing_date, column] == 0
            assert predictions[column].sum() > 0


def test_oob_minimum_counts_and_higher_quantile_thresholds_are_preserved(
    walkforward_result,
) -> None:
    _, result, _ = walkforward_result
    assert len(result.model_states) == 14
    for state in result.model_states:
        counts = np.asarray(state["oob_tree_counts"], dtype=np.int64)
        assert len(counts) == state["training_row_count"]
        assert counts.min() == MINIMUM_OOB_TREE_COUNT + 4
        assert state["oob_minimum_tree_count"] == MINIMUM_OOB_TREE_COUNT + 4
        source = np.linspace(0.05, 0.45, len(counts), dtype=float)
        for candidate in walkforward.RARE_LOSS_CANDIDATES:
            expected = float(
                np.quantile(source, candidate.oob_tail_quantile, method="higher")
            )
            assert state["oob_tail_thresholds"][
                candidate.candidate_id
            ] == pytest.approx(expected)
            threshold_column = (
                f"{state['variant']}_{candidate.candidate_id}_oob_severe_threshold"
            )
            fold_values = result.predictions.loc[
                result.predictions["fold_id"] == state["fold_id"], threshold_column
            ]
            assert fold_values.nunique() == 1
            assert float(fold_values.iloc[0]) == pytest.approx(expected)


def test_oob_below_sixteen_fails_closed(fake_forest) -> None:
    _FakeRareLossForest.force_low_oob = True
    with pytest.raises(ValueError, match="minimum 16-tree OOB contract"):
        walkforward.build_pre2019_rare_loss_walkforward(
            _synthetic_feature_label_frame()
        )


def test_first_and_final_fill_targets_are_long(walkforward_result) -> None:
    _, result, _ = walkforward_result
    predictions = result.predictions
    assert predictions.iloc[0]["label_entry_date"].year == 2005
    assert predictions.iloc[-1]["label_entry_date"].year == 2018
    for variant in walkforward.MODEL_VARIANTS:
        for candidate in walkforward.RARE_LOSS_CANDIDATES:
            column = walkforward.candidate_cash_target_column(
                variant, candidate.candidate_id
            )
            assert predictions.iloc[0][column] == 0
            assert predictions.iloc[-1][column] == 0


def test_fourteen_states_and_predictions_are_deterministic(fake_forest) -> None:
    frame = _synthetic_feature_label_frame()
    first = walkforward.build_pre2019_rare_loss_walkforward(frame)
    first_states = json.dumps(first.model_states, sort_keys=True, separators=(",", ":"))
    _FakeRareLossForest.instances = []
    repeated = walkforward.build_pre2019_rare_loss_walkforward(frame.copy())
    pd.testing.assert_frame_equal(first.predictions, repeated.predictions, check_exact=True)
    repeated_states = json.dumps(
        repeated.model_states, sort_keys=True, separators=(",", ":")
    )
    assert first_states == repeated_states
    assert len(first.model_states) == len(repeated.model_states) == 14
