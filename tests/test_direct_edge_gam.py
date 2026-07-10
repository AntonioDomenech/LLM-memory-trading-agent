from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from agent_benchmark.direct_edge_gam import (
    ANCHOR_FEATURE_NAMES,
    DirectEdgeGAM,
    DirectEdgeGAMConfig,
)


def _metadata() -> dict[str, str]:
    return {
        "training_start_date": "2000-01-03",
        "training_end_date": "2018-12-20",
        "maximum_label_maturity_date": "2018-12-31",
        "training_dates_sha256": "0" * 64,
        "binary_target_sha256": "1" * 64,
        "edge_target_sha256": "2" * 64,
    }


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _synthetic_data(
    sample_count: int = 420,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(78122)
    features = rng.normal(size=(sample_count, 6))
    latent = (
        -0.35
        + 1.15 * features[:, 0]
        - 0.70 * np.maximum(-features[:, 1] - 0.5, 0.0)
        + 0.55 * features[:, 0] * features[:, 2]
    )
    probability = 1.0 / (1.0 + np.exp(-latent))
    binary = rng.binomial(1, probability).astype(float)
    edge = (
        -0.004
        + 0.018 * features[:, 0]
        - 0.012 * np.maximum(-features[:, 1] - 0.5, 0.0)
        + 0.010 * features[:, 0] * features[:, 2]
        + rng.normal(scale=0.007, size=sample_count)
    )
    names = (
        "aapl_drawdown_252",
        "aapl_rv_20",
        "price_feature",
        "qqq_lr_60",
        "fourth_feature",
        "fifth_feature",
    )
    return features, binary, edge, names


def _fit_synthetic() -> tuple[DirectEdgeGAM, np.ndarray, np.ndarray, np.ndarray]:
    features, binary, edge, names = _synthetic_data()
    model = DirectEdgeGAM().fit(
        features,
        binary,
        edge,
        feature_names=names,
        fit_metadata=_metadata(),
    )
    return model, features, binary, edge


def test_frozen_config_and_basis_preprocessing_are_exact():
    config = DirectEdgeGAMConfig()
    config.validate()
    assert config.ridge_lambdas == (0.001, 0.01, 0.1)
    assert config.raw_mad_multiplier == pytest.approx(1.4826)
    assert config.raw_z_clip == pytest.approx(4.0)
    assert config.hinge_knot == pytest.approx(1.0)
    assert config.logistic_max_iterations == 50
    assert config.huber_delta == pytest.approx(1.5)
    with pytest.raises(ValueError, match="frozen"):
        DirectEdgeGAMConfig(ridge_lambdas=(0.01,)).validate()

    features = np.array(
        [
            [-100.0, -2.0],
            [-1.0, -1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [100.0, 3.0],
        ]
    )
    binary = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    edge = np.array([-0.03, -0.02, -0.01, 0.01, 0.02, 0.03])
    names = ("aapl_drawdown_252", "ordinary")
    model = DirectEdgeGAM().fit(
        features,
        binary,
        edge,
        feature_names=names,
        fit_metadata=_metadata(),
    )

    medians = np.median(features, axis=0)
    mads = np.median(np.abs(features - medians), axis=0)
    scales = np.maximum(1.4826 * mads, 1e-6)
    raw_z = np.clip((features - medians) / scales, -4.0, 4.0)
    unscaled = np.column_stack(
        (
            raw_z[:, 0],
            np.maximum(raw_z[:, 0] - 1.0, 0.0),
            np.maximum(-raw_z[:, 0] - 1.0, 0.0),
            raw_z[:, 1],
            np.maximum(raw_z[:, 1] - 1.0, 0.0),
            np.maximum(-raw_z[:, 1] - 1.0, 0.0),
            raw_z[:, 0] * raw_z[:, 1],
        )
    )
    basis_means = unscaled.mean(axis=0)
    basis_scales = np.maximum(unscaled.std(axis=0, ddof=0), 1e-6)
    expected = (unscaled - basis_means) / basis_scales

    np.testing.assert_array_equal(model.raw_medians, medians)
    np.testing.assert_array_equal(model.raw_scales, scales)
    np.testing.assert_allclose(model.transform_basis(features), expected, atol=1e-15)
    assert model.basis_names == (
        "linear:aapl_drawdown_252",
        "hinge_positive:aapl_drawdown_252",
        "hinge_negative:aapl_drawdown_252",
        "linear:ordinary",
        "hinge_positive:ordinary",
        "hinge_negative:ordinary",
        "interaction:aapl_drawdown_252*ordinary",
    )
    assert set(ANCHOR_FEATURE_NAMES) == {
        "aapl_drawdown_252",
        "aapl_rv_20",
        "qqq_lr_60",
        "vix_z_252",
    }


def test_both_solvers_converge_with_frozen_limits():
    model, _, _, _ = _fit_synthetic()
    assert all(head["converged"] for head in model.logistic_heads)
    assert all(0 <= head["iterations"] <= 50 for head in model.logistic_heads)
    assert all(head["converged"] for head in model.huber_heads)
    assert all(0 <= head["iterations"] <= 50 for head in model.huber_heads)


def test_predictions_are_bounded_and_learn_the_synthetic_signal():
    model, features, binary, edge = _fit_synthetic()
    components = model.predict_components(features)
    probability = components["cash_beats_long_probability"]
    predicted_edge = components["expected_edge_10bps"]
    assert np.isfinite(probability).all()
    assert np.isfinite(predicted_edge).all()
    assert ((0.0 <= probability) & (probability <= 1.0)).all()
    assert ((-0.2 <= predicted_edge) & (predicted_edge <= 0.2)).all()
    assert probability[binary == 1.0].mean() > probability[binary == 0.0].mean()
    prevalence_brier = float(np.mean((binary.mean() - binary) ** 2))
    model_brier = float(np.mean((probability - binary) ** 2))
    assert model_brier < prevalence_brier
    assert np.corrcoef(predicted_edge, edge)[0, 1] > 0.80

    extreme = np.full((2, features.shape[1]), 1e12)
    extreme_prediction = model.predict_components(extreme)
    assert ((0.0 <= extreme_prediction["cash_beats_long_probability"]) & (
        extreme_prediction["cash_beats_long_probability"] <= 1.0
    )).all()
    assert ((-0.2 <= extreme_prediction["expected_edge_10bps"]) & (
        extreme_prediction["expected_edge_10bps"] <= 0.2
    )).all()


def test_fit_and_serialized_bytes_are_deterministic(tmp_path):
    features, binary, edge, names = _synthetic_data()
    first = DirectEdgeGAM().fit(
        features,
        binary,
        edge,
        feature_names=names,
        fit_metadata=_metadata(),
    )
    second = DirectEdgeGAM().fit(
        features.copy(),
        binary.copy(),
        edge.copy(),
        feature_names=names,
        fit_metadata=dict(_metadata()),
    )
    assert first.canonical_json() == second.canonical_json()
    assert first.model_sha256 == second.model_sha256
    np.testing.assert_array_equal(first.predict_proba(features), second.predict_proba(features))
    np.testing.assert_array_equal(
        first.predict_expected_edge(features), second.predict_expected_edge(features)
    )

    path = tmp_path / "gam.json"
    first.save(path)
    assert path.read_bytes() == (first.canonical_json() + "\n").encode("utf-8")
    loaded = DirectEdgeGAM.load(path)
    assert loaded.canonical_json() == first.canonical_json()
    assert loaded.model_sha256 == first.model_sha256
    state = first.to_state()
    preprocessing = state["payload"]["parameters"]["preprocessing"]
    for key in (
        "raw_medians",
        "raw_scales",
        "basis_means",
        "basis_scales",
    ):
        assert all(value == float.fromhex(value).hex() for value in preprocessing[key])
    for group in state["payload"]["parameters"]["heads"].values():
        assert all(
            value == float.fromhex(value).hex()
            for head in group
            for value in head["coefficients"]
        )


def test_checksum_schema_and_semantic_tampering_are_rejected():
    model, _, _, _ = _fit_synthetic()
    state = model.to_state()

    ordinary_tamper = copy.deepcopy(state)
    ordinary_tamper["payload"]["training"]["sample_count"] += 1
    with pytest.raises(ValueError, match="checksum"):
        DirectEdgeGAM.from_state(ordinary_tamper)

    parameter_tamper = copy.deepcopy(state)
    coefficient = parameter_tamper["payload"]["parameters"]["heads"]["logistic"][0][
        "coefficients"
    ][0]
    parameter_tamper["payload"]["parameters"]["heads"]["logistic"][0][
        "coefficients"
    ][0] = (float.fromhex(coefficient) + 0.01).hex()
    parameter_tamper["payload_sha256"] = _canonical_hash(parameter_tamper["payload"])
    with pytest.raises(ValueError, match="parameter checksum"):
        DirectEdgeGAM.from_state(parameter_tamper)

    basis_tamper = copy.deepcopy(state)
    basis_tamper["payload"]["basis_names"][0] = "linear:invented"
    basis_tamper["payload_sha256"] = _canonical_hash(basis_tamper["payload"])
    with pytest.raises(ValueError, match="basis schema"):
        DirectEdgeGAM.from_state(basis_tamper)

    scaler_tamper = copy.deepcopy(state)
    scaler_tamper["payload"]["parameters"]["preprocessing"]["raw_scales"][0] = 0.0.hex()
    scaler_tamper["payload"]["parameters_sha256"] = _canonical_hash(
        scaler_tamper["payload"]["parameters"]
    )
    scaler_tamper["payload_sha256"] = _canonical_hash(scaler_tamper["payload"])
    with pytest.raises(ValueError, match="positive floor"):
        DirectEdgeGAM.from_state(scaler_tamper)

    schema_tamper = copy.deepcopy(state)
    schema_tamper["payload"]["unexpected"] = True
    schema_tamper["payload_sha256"] = _canonical_hash(schema_tamper["payload"])
    with pytest.raises(ValueError, match="Invalid payload keys"):
        DirectEdgeGAM.from_state(schema_tamper)


def test_nonfinite_inputs_bad_metadata_and_wrong_schemas_fail_closed():
    features, binary, edge, names = _synthetic_data(120)
    broken = features.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        DirectEdgeGAM().fit(
            broken,
            binary,
            edge,
            feature_names=names,
            fit_metadata=_metadata(),
        )
    broken_edge = edge.copy()
    broken_edge[-1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        DirectEdgeGAM().fit(
            features,
            binary,
            broken_edge,
            feature_names=names,
            fit_metadata=_metadata(),
        )
    bad_metadata = _metadata()
    bad_metadata["training_dates_sha256"] = "ABC"
    with pytest.raises(ValueError, match="SHA-256"):
        DirectEdgeGAM().fit(
            features,
            binary,
            edge,
            feature_names=names,
            fit_metadata=bad_metadata,
        )

    model = DirectEdgeGAM().fit(
        features,
        binary,
        edge,
        feature_names=names,
        fit_metadata=_metadata(),
    )
    with pytest.raises(ValueError, match="schema"):
        model.predict_proba(features[:, :-1])
    corrupted_prediction = features[:2].copy()
    corrupted_prediction[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        model.predict_expected_edge(corrupted_prediction)


def test_implementation_uses_only_the_local_numeric_stack():
    source = (
        Path(__file__).resolve().parents[1]
        / "agent_benchmark"
        / "direct_edge_gam.py"
    ).read_text(encoding="utf-8").lower()
    forbidden = (("sk" + "learn"), ("sci" + "py"))
    assert not any(name in source for name in forbidden)

