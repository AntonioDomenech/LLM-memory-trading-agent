from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import agent_benchmark.regime_consensus_walkforward as walkforward
from agent_benchmark.regime_consensus_features import (
    ALL_HEADS_READY_COLUMN,
    HEAD_FEATURE_COLUMNS,
    HEAD_NAMES,
    HEAD_ORIENTATIONS,
    HEAD_READINESS_COLUMNS,
    REGIME_FEATURE_COLUMNS,
    REGIME_LABEL_COLUMNS,
    regime_head_readiness,
    strict_pre_fold_head_training_mask,
    strict_pre_fold_label_mask,
)


class _FakeRegimeHeadModel:
    instances: list["_FakeRegimeHeadModel"] = []
    profiles = {
        "aapl_state": (0.62, 0.00080),
        "market_state": (0.56, 0.00080),
        "sentiment_state": (0.475, -0.00005),
    }

    def __init__(self) -> None:
        self.head_name = ""
        self.feature_names: tuple[str, ...] = ()
        self.fit_features = pd.DataFrame()
        self.fit_binary = np.empty(0)
        self.fit_edge = np.empty(0)
        self.fit_metadata: dict[str, object] = {}
        self._hash = ""
        self.prediction_calls: list[dict[str, object]] = []
        self.instances.append(self)

    @property
    def model_sha256(self) -> str:
        return self._hash

    def fit(
        self,
        features,
        cash_beats_long_10bps,
        edge_10bps,
        *,
        head_name,
        feature_names=None,
        stress_orientations,
        fit_metadata,
    ):
        self.head_name = str(head_name)
        self.feature_names = tuple(feature_names or features.columns)
        self.fit_features = features.copy()
        self.fit_binary = np.asarray(cash_beats_long_10bps, dtype=float).copy()
        self.fit_edge = np.asarray(edge_10bps, dtype=float).copy()
        self.fit_metadata = dict(fit_metadata)
        assert tuple(features.columns) == self.feature_names
        assert len(features) == len(self.fit_binary) == len(self.fit_edge)
        assert tuple(stress_orientations) == HEAD_ORIENTATIONS[
            HEAD_NAMES.index(self.head_name)
        ]
        assert set(self.fit_metadata) == {
            "training_start_date",
            "training_end_date",
            "maximum_label_maturity_date",
            "training_dates_sha256",
            "binary_target_sha256",
            "edge_target_sha256",
        }
        material = json.dumps(
            {
                "head_name": self.head_name,
                "feature_names": self.feature_names,
                "fit_metadata": self.fit_metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self._hash = hashlib.sha256(material.encode("ascii")).hexdigest()
        return self

    def predict_components(
        self,
        features,
        *,
        initial_state_probability=None,
        continue_from_training=True,
    ):
        values = features.to_numpy(dtype=float)
        available = np.isfinite(values).all(axis=1)
        stress = np.full(len(values), np.nan, dtype=float)
        run_length = 0
        for position, complete in enumerate(available):
            if not complete:
                run_length = 0
                continue
            run_length += 1
            stress[position] = 0.40 + min(run_length, 10) * 0.01
        probability = np.full(len(values), np.nan, dtype=float)
        edge = np.full(len(values), np.nan, dtype=float)
        probability[available], edge[available] = self.profiles[self.head_name]
        state = np.column_stack((1.0 - stress, stress))
        self.prediction_calls.append(
            {
                "features": features.copy(),
                "initial_state_probability": initial_state_probability,
                "continue_from_training": continue_from_training,
            }
        )
        return {
            "state_probabilities": state,
            "stress_probability": stress,
            "cash_beats_long_probability": probability,
            "expected_edge_10bps": edge,
            "observation_available": available,
        }

    def to_state(self):
        return {
            "payload": {
                "head_name": self.head_name,
                "feature_names": list(self.feature_names),
                "fit_metadata": self.fit_metadata,
            },
            "payload_sha256": self._hash,
        }


def _synthetic_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2000-01-03", "2018-12-31", name="decision_date")
    position = np.arange(len(dates), dtype=float)
    data: dict[str, object] = {}
    for feature_number, column in enumerate(REGIME_FEATURE_COLUMNS, start=1):
        data[column] = (
            np.sin(position / (7.0 + feature_number))
            + 0.3 * np.cos(position / (19.0 + feature_number))
            + feature_number / 100.0
        )

    entry_date = pd.Series(dates, index=dates).shift(-1)
    maturity_date = pd.Series(dates, index=dates).shift(-2)
    edge_10 = 0.003 * np.sin(position / 11.0) - 0.0002
    edge_10[-2:] = np.nan
    edge_5 = np.where(np.isfinite(edge_10), edge_10 + 0.001, np.nan)
    binary_10 = pd.Series((np.nan_to_num(edge_10) > 0.0).astype(np.int8), index=dates, dtype="Int8")
    binary_5 = pd.Series((np.nan_to_num(edge_5) > 0.0).astype(np.int8), index=dates, dtype="Int8")
    binary_10.iloc[-2:] = pd.NA
    binary_5.iloc[-2:] = pd.NA
    data.update(
        {
            "label_entry_date": entry_date.to_numpy(),
            "label_maturity_date": maturity_date.to_numpy(),
            "label_entry_adjusted_open": np.where(entry_date.notna(), 100.0, np.nan),
            "label_exit_adjusted_open": np.where(maturity_date.notna(), 101.0, np.nan),
            "aapl_forward_log_return_1": np.where(np.isfinite(edge_10), -edge_10, np.nan),
            "cash_active_log_edge_5bps": edge_5,
            "cash_active_log_edge_10bps": edge_10,
            "cash_beats_long_5bps": binary_5,
            "cash_beats_long_10bps": binary_10,
        }
    )
    frame = pd.DataFrame(data, index=dates)

    # One historical gap must be preserved in the sentiment HMM chronology.
    frame.loc[pd.Timestamp("2003-06-16"), HEAD_FEATURE_COLUMNS[2][0]] = np.nan
    # One validation gap proves reset/no-partial-consensus behavior.
    frame.loc[pd.Timestamp("2005-06-15"), HEAD_FEATURE_COLUMNS[0][0]] = np.nan
    readiness = regime_head_readiness(frame)
    for column in readiness:
        frame[column] = readiness[column]
    frame["label_available"] = frame["label_maturity_date"].notna()
    frame["entry_fill_year"] = pd.to_datetime(frame["label_entry_date"]).dt.year.astype(
        "Int16"
    )
    return frame


@pytest.fixture()
def fake_model(monkeypatch):
    _FakeRegimeHeadModel.instances = []
    monkeypatch.setattr(walkforward, "RegimeHeadModel", _FakeRegimeHeadModel)


@pytest.fixture()
def synthetic_result(fake_model):
    frame = _synthetic_frame()
    first = walkforward.build_pre2019_regime_consensus_walkforward(frame)
    first_instances = list(_FakeRegimeHeadModel.instances)
    repeated = walkforward.build_pre2019_regime_consensus_walkforward(frame.copy())
    return frame, first, repeated, first_instances


def test_entry_fill_folds_purge_outcomes_and_freeze_exactly_three_heads(
    synthetic_result,
) -> None:
    frame, result, _, _ = synthetic_result
    predictions = result.predictions
    assert tuple(fold.fold_id for fold in walkforward.WALK_FORWARD_FOLDS) == (
        "2005-2006",
        "2007-2008",
        "2009-2010",
        "2011-2012",
        "2013-2014",
        "2015-2016",
        "2017-2018",
    )
    assert predictions.index.min() == pd.Timestamp("2004-12-31")
    assert pd.Timestamp(predictions.iloc[0]["label_entry_date"]).year == 2005

    for fold in walkforward.WALK_FORWARD_FOLDS:
        rows = predictions.loc[predictions["fold_id"] == fold.fold_id]
        first_decision = rows.index.min()
        baseline_mask = strict_pre_fold_label_mask(
            frame, fold_first_decision_date=first_decision
        )
        assert rows["causal_baseline_training_count"].eq(int(baseline_mask.sum())).all()
        assert (rows["causal_baseline_max_label_maturity_date"] < first_decision).all()
        for head_name in HEAD_NAMES:
            expected = strict_pre_fold_head_training_mask(
                frame,
                fold_first_decision_date=first_decision,
                head_name=head_name,
            )
            assert rows[f"{head_name}_outcome_training_rows"].eq(
                int(expected.sum())
            ).all()
            assert (rows[f"{head_name}_training_emission_end_date"] < first_decision).all()
            assert (
                rows[f"{head_name}_training_max_label_maturity_date"]
                < first_decision
            ).all()
            assert rows[f"{head_name}_model_sha256"].nunique() == 1

    assert len(result.model_states) == 21
    assert [(item["fold_id"], item["head_name"]) for item in result.model_states] == [
        (fold.fold_id, head_name)
        for fold in walkforward.WALK_FORWARD_FOLDS
        for head_name in HEAD_NAMES
    ]


def test_full_chronology_keeps_gaps_and_validation_filter_continues_once(
    synthetic_result,
) -> None:
    _, result, _, first_instances = synthetic_result
    first_sentiment = next(
        instance
        for instance in first_instances
        if instance.head_name == "sentiment_state"
        and instance.fit_metadata["training_end_date"] < "2005-01-01"
    )
    assert first_sentiment.fit_features.index.min() == pd.Timestamp("2000-01-03")
    assert first_sentiment.fit_features.index.max() == pd.Timestamp("2004-12-30")
    assert first_sentiment.fit_features.loc[pd.Timestamp("2003-06-16")].isna().any()
    assert len(first_sentiment.prediction_calls) == 1
    call = first_sentiment.prediction_calls[0]
    assert call["continue_from_training"] is True
    assert call["initial_state_probability"] is None
    assert call["features"].index.min() == pd.Timestamp("2004-12-31")

    predictions = result.predictions
    gap = pd.Timestamp("2005-06-15")
    after_gap = predictions.index[predictions.index.get_loc(gap) + 1]
    assert not predictions.loc[gap, "aapl_state_ready"]
    assert not predictions.loc[gap, ALL_HEADS_READY_COLUMN]
    assert pd.isna(predictions.loc[gap, "aapl_state_stress_probability"])
    assert predictions.loc[after_gap, "aapl_state_stress_probability"] == pytest.approx(
        0.41
    )
    assert pd.isna(
        predictions.loc[gap, "regime_consensus_cash_win_probability_10bps"]
    )
    for candidate in walkforward.REGIME_CANDIDATES:
        for variant in walkforward.MODEL_VARIANTS:
            assert predictions.loc[
                gap,
                walkforward.candidate_cash_target_column(
                    variant, candidate.candidate_id
                ),
            ] == 0


def test_validation_outcome_mutation_does_not_change_same_fold_predictions(
    fake_model,
) -> None:
    frame = _synthetic_frame()
    baseline = walkforward.build_pre2019_regime_consensus_walkforward(frame)
    first_fold = baseline.predictions["fold_id"].eq("2005-2006")
    validation_dates = baseline.predictions.index[first_fold]
    changed = frame.copy()
    mature = changed.loc[validation_dates, "cash_beats_long_10bps"].notna()
    changed_dates = validation_dates[mature.to_numpy()]
    changed.loc[changed_dates, "cash_beats_long_10bps"] = (
        1 - changed.loc[changed_dates, "cash_beats_long_10bps"].astype(int)
    ).astype("Int8")
    changed.loc[changed_dates, "cash_active_log_edge_10bps"] *= -17.0
    mutated = walkforward.build_pre2019_regime_consensus_walkforward(changed)

    ignored = set(REGIME_LABEL_COLUMNS)
    comparable = [
        column for column in baseline.predictions.columns if column not in ignored
    ]
    pd.testing.assert_frame_equal(
        baseline.predictions.loc[first_fold, comparable],
        mutated.predictions.loc[first_fold, comparable],
        check_exact=True,
    )
    assert baseline.model_states[:3] == mutated.model_states[:3]


def test_exact_frozen_thresholds_support_and_long_boundary_fills(
    synthetic_result,
) -> None:
    _, result, _, _ = synthetic_result
    predictions = result.predictions
    assert tuple(item.candidate_id for item in walkforward.REGIME_CANDIDATES) == (
        "p55_e5",
        "p60_e5",
    )
    ready = predictions[ALL_HEADS_READY_COLUMN].to_numpy(dtype=bool)
    first_fill = int(
        np.argmin(pd.to_datetime(predictions["label_entry_date"]).to_numpy())
    )
    final_fill = int(
        np.argmax(pd.to_datetime(predictions["label_entry_date"]).to_numpy())
    )
    for variant in walkforward.MODEL_VARIANTS:
        for candidate in walkforward.REGIME_CANDIDATES:
            target = predictions[
                walkforward.candidate_cash_target_column(
                    variant, candidate.candidate_id
                )
            ]
            assert target.dtype == np.int8
            assert target.iloc[first_fill] == 0
            assert target.iloc[final_fill] == 0
            assert not target.loc[~predictions[ALL_HEADS_READY_COLUMN]].any()

    full_p55 = predictions[
        walkforward.candidate_cash_target_column("regime_consensus", "p55_e5")
    ]
    full_p60 = predictions[
        walkforward.candidate_cash_target_column("regime_consensus", "p60_e5")
    ]
    aapl_p60 = predictions[
        walkforward.candidate_cash_target_column("aapl_only", "p60_e5")
    ]
    assert full_p55.sum() == int(ready.sum()) - 2
    assert full_p60.sum() == 0
    assert aapl_p60.sum() == int(ready.sum()) - 2

    # Targets are formed after concatenating folds, so a CASH run may remain
    # continuous across a refit boundary.
    first_rows = predictions.loc[predictions["fold_id"] == "2005-2006"]
    second_rows = predictions.loc[predictions["fold_id"] == "2007-2008"]
    assert full_p55.loc[first_rows.index[-1]] == 1
    assert full_p55.loc[second_rows.index[0]] == 1


def test_outputs_states_and_hashes_are_byte_deterministic(synthetic_result) -> None:
    _, first, repeated, _ = synthetic_result
    pd.testing.assert_frame_equal(
        first.predictions, repeated.predictions, check_exact=True
    )
    assert first.model_states == repeated.model_states
    encoded = json.dumps(
        first.model_states,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert json.loads(encoded)
    for state in first.model_states:
        assert len(state["model_sha256"]) == 64
        assert len(state["canonical_state_sha256"]) == 64


def test_convenience_wrapper_and_post_2018_refusal(fake_model) -> None:
    frame = _synthetic_frame()
    expected = walkforward.build_pre2019_regime_consensus_walkforward(frame)
    actual = walkforward.build_pre2019_regime_consensus_predictions(frame.copy())
    pd.testing.assert_frame_equal(expected.predictions, actual, check_exact=True)

    forbidden = frame.iloc[[-1]].copy()
    forbidden.index = pd.DatetimeIndex(["2019-01-02"], name="decision_date")
    with pytest.raises(ValueError, match="post-2018"):
        walkforward.build_pre2019_regime_consensus_predictions(
            pd.concat([frame, forbidden])
        )
