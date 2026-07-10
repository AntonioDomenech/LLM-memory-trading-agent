from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.regime_consensus_features import (
    HEAD_FEATURE_COLUMNS,
    HEAD_ORIENTATIONS,
    build_regime_consensus_feature_label_frame,
)
from agent_benchmark.regime_hmm import (
    RegimeHeadModel,
    RegimeHMMConfig,
    RegimeOutcomeHead,
    _likelihood_decreased_beyond_guard,
    _oriented_order,
    _regularized_log_objective,
    _sequence_slices,
)


def _metadata() -> dict[str, str]:
    return {
        "training_start_date": "2000-01-03",
        "training_end_date": "2004-01-02",
        "maximum_label_maturity_date": "2003-12-31",
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


def _synthetic(
    sample_count: int = 760,
    *,
    terminal_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[int, ...]]:
    rng = np.random.default_rng(90210)
    latent_state = np.zeros(sample_count, dtype=int)
    for index in range(1, sample_count):
        if rng.random() < 0.055:
            latent_state[index] = 1 - latent_state[index - 1]
        else:
            latent_state[index] = latent_state[index - 1]
    means = np.asarray([[-0.75, 0.55, -0.35], [0.90, -0.70, 0.45]])
    features = means[latent_state] + rng.normal(scale=0.52, size=(sample_count, 3))
    features[175:179] = np.nan
    if sample_count > 521:
        features[521] = np.nan
    if terminal_missing:
        features[-1] = np.nan
    cash_probability = np.where(latent_state == 1, 0.73, 0.28)
    binary = (rng.random(sample_count) < cash_probability).astype(float)
    edge = (
        np.where(latent_state == 1, 0.006, -0.003)
        + rng.normal(scale=0.003, size=sample_count)
    )
    binary[60:66] = np.nan
    edge[60:66] = np.nan
    names = ("fear", "trend", "volatility")
    orientations = (1, -1, 1)
    return features, binary, edge, names, orientations


def _fit(*, terminal_missing: bool = False) -> tuple[RegimeHeadModel, np.ndarray]:
    features, binary, edge, names, orientations = _synthetic(
        terminal_missing=terminal_missing
    )
    model = RegimeHeadModel().fit(
        features,
        binary,
        edge,
        head_name="synthetic_state",
        feature_names=names,
        stress_orientations=orientations,
        fit_metadata=_metadata(),
    )
    return model, features


def test_frozen_configuration_and_oriented_rank_fit_are_exact() -> None:
    config = RegimeHMMConfig()
    config.validate()
    assert config.n_states == 2
    assert config.raw_mad_multiplier == pytest.approx(1.4826)
    assert config.raw_scale_floor == pytest.approx(1e-6)
    assert config.raw_z_clip == pytest.approx(6.0)
    assert config.variance_floor == pytest.approx(0.0025)
    assert config.transition_additive == ((5.0, 1.0), (1.0, 5.0))
    assert config.pi_additive == (1.0, 1.0)
    assert config.max_iterations == 100
    assert config.min_iterations == 5
    assert config.relative_tolerance == pytest.approx(1e-10)
    assert config.min_complete_emissions == 500
    assert config.min_effective_occupancy == pytest.approx(50.0)
    assert config.outcome_prior == pytest.approx(100.0)
    with pytest.raises(ValueError, match="frozen"):
        RegimeHMMConfig(raw_z_clip=5.0).validate()

    model, features = _fit()
    complete = np.isfinite(features).all(axis=1)
    observations = features[complete]
    medians = np.median(observations, axis=0)
    scales = np.maximum(
        1.4826 * np.median(np.abs(observations - medians), axis=0), 1e-6
    )
    np.testing.assert_array_equal(model.raw_medians, medians)
    np.testing.assert_array_equal(model.raw_scales, scales)
    assert model.converged
    assert 5 <= model.iterations <= 100
    assert (model.variances >= 0.0025).all()
    np.testing.assert_allclose(model.pi.sum(), 1.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        model.transition.sum(axis=1), np.ones(2), rtol=0.0, atol=1e-14
    )
    oriented = model.means @ np.asarray((1.0, -1.0, 1.0)) / 3.0
    assert oriented[0] <= oriented[1]
    assert (model.effective_occupancy >= 50.0).all()
    assert model.effective_occupancy.sum() == pytest.approx(complete.sum())


def test_outcome_head_uses_laplace_global_and_exact_prior_100_shrinkage() -> None:
    rows = 600
    state = np.column_stack(
        (np.linspace(0.9, 0.1, rows), np.linspace(0.1, 0.9, rows))
    )
    binary = (np.arange(rows) % 3 == 0).astype(float)
    edge = np.linspace(-0.02, 0.03, rows)
    head = RegimeOutcomeHead().fit(state, binary, edge)
    occupancy = state.sum(axis=0)
    p0 = (binary.sum() + 1.0) / (rows + 2.0)
    expected_probability = (state.T @ binary + 100.0 * p0) / (
        occupancy + 100.0
    )
    global_edge = edge.mean()
    expected_edge = (state.T @ edge + 100.0 * global_edge) / (
        occupancy + 100.0
    )
    assert head.global_probability == pytest.approx(p0)
    np.testing.assert_allclose(head.state_probabilities, expected_probability)
    np.testing.assert_allclose(head.state_edges, expected_edge)
    loaded = RegimeOutcomeHead.from_state(head.to_state())
    assert loaded.canonical_json() == head.canonical_json()


def test_forward_filter_is_causal_resets_on_missing_and_continues_training_state() -> None:
    model, _ = _fit()
    validation = np.asarray(
        [
            [0.6, -0.4, 0.5],
            [0.7, -0.5, 0.6],
            [0.8, -0.6, 0.7],
            [0.9, -0.7, 0.8],
        ]
    )
    default = model.filter_state_probabilities(validation)
    explicit = model.filter_state_probabilities(
        validation,
        initial_state_probability=model.training_terminal_state_probability,
        continue_from_training=False,
    )
    np.testing.assert_array_equal(default, explicit)

    changed_future = validation.copy()
    changed_future[-1] = [-4.0, 4.0, -4.0]
    changed = model.filter_state_probabilities(changed_future)
    np.testing.assert_array_equal(default[:-1], changed[:-1])

    with_gap = np.vstack((validation[:2], np.full((1, 3), np.nan), validation[2:]))
    gap_filtered = model.filter_state_probabilities(with_gap)
    assert np.isnan(gap_filtered[2]).all()
    fresh_suffix = model.filter_state_probabilities(
        validation[2:], continue_from_training=False
    )
    np.testing.assert_array_equal(gap_filtered[3:], fresh_suffix)

    missing_terminal, _ = _fit(terminal_missing=True)
    assert not missing_terminal.training_terminal_contiguous
    np.testing.assert_array_equal(
        missing_terminal.training_terminal_state_probability,
        missing_terminal.initial_probabilities,
    )
    np.testing.assert_array_equal(
        missing_terminal.filter_state_probabilities(validation),
        missing_terminal.filter_state_probabilities(
            validation, continue_from_training=False
        ),
    )


def test_prediction_contract_returns_nan_fallback_rows_and_bounded_probabilities() -> None:
    model, _ = _fit()
    rows = np.asarray([[0.1, -0.1, 0.2], [np.nan, 0.0, 0.0], [0.3, -0.4, 0.5]])
    prediction = model.predict_components(rows, continue_from_training=False)
    assert set(prediction) == {
        "state_probabilities",
        "stress_probability",
        "cash_beats_long_probability",
        "expected_edge_10bps",
        "observation_available",
    }
    np.testing.assert_array_equal(prediction["observation_available"], [True, False, True])
    assert np.isnan(prediction["stress_probability"][1])
    assert np.isnan(prediction["cash_beats_long_probability"][1])
    assert np.isnan(prediction["expected_edge_10bps"][1])
    available = prediction["observation_available"]
    probability = prediction["cash_beats_long_probability"][available]
    assert ((0.0 <= probability) & (probability <= 1.0)).all()


def test_double_fit_serialization_and_predictions_are_byte_deterministic(tmp_path: Path) -> None:
    features, binary, edge, names, orientations = _synthetic()
    first = RegimeHeadModel().fit(
        features,
        binary,
        edge,
        head_name="synthetic_state",
        feature_names=names,
        stress_orientations=orientations,
        fit_metadata=_metadata(),
    )
    second = RegimeHeadModel().fit(
        features.copy(),
        binary.copy(),
        edge.copy(),
        head_name="synthetic_state",
        feature_names=names,
        stress_orientations=orientations,
        fit_metadata=dict(_metadata()),
    )
    assert first.canonical_json() == second.canonical_json()
    assert first.model_sha256 == second.model_sha256
    sample = features[-20:].copy()
    for key, values in first.predict_components(sample).items():
        np.testing.assert_array_equal(values, second.predict_components(sample)[key])

    path = tmp_path / "regime.json"
    first.save(path)
    assert path.read_bytes() == (first.canonical_json() + "\n").encode("utf-8")
    loaded = RegimeHeadModel.load(path)
    assert loaded.canonical_json() == first.canonical_json()
    assert loaded.model_sha256 == first.model_sha256
    state = first.to_state()
    parameters = state["payload"]["parameters"]
    encoded = [
        *parameters["preprocessing"]["raw_medians"],
        *parameters["preprocessing"]["raw_scales"],
        *parameters["hmm"]["initial_probabilities"],
        *parameters["hmm"]["training_terminal_state_probability"],
    ]
    assert all(value == float.fromhex(value).hex() for value in encoded)


def test_checksum_schema_and_rehashed_semantic_tampering_fail_closed() -> None:
    model, _ = _fit()
    state = model.to_state()
    ordinary = copy.deepcopy(state)
    ordinary["payload"]["training"]["row_count"] += 1
    with pytest.raises(ValueError, match="checksum"):
        RegimeHeadModel.from_state(ordinary)

    parameter = copy.deepcopy(state)
    parameter["payload"]["parameters"]["hmm"]["means"][0][0] = 99.0.hex()
    parameter["payload_sha256"] = _canonical_hash(parameter["payload"])
    with pytest.raises(ValueError, match="parameter checksum"):
        RegimeHeadModel.from_state(parameter)

    scale = copy.deepcopy(state)
    scale["payload"]["parameters"]["preprocessing"]["raw_scales"][0] = 0.0.hex()
    scale["payload"]["parameters_sha256"] = _canonical_hash(
        scale["payload"]["parameters"]
    )
    scale["payload_sha256"] = _canonical_hash(scale["payload"])
    with pytest.raises(ValueError, match="scale violates"):
        RegimeHeadModel.from_state(scale)

    convergence = copy.deepcopy(state)
    convergence["payload"]["training"]["converged"] = False
    convergence["payload_sha256"] = _canonical_hash(convergence["payload"])
    with pytest.raises(ValueError, match="converged"):
        RegimeHeadModel.from_state(convergence)


def test_minimum_emissions_labels_occupancy_and_nonfinite_inputs_are_rejected() -> None:
    features, binary, edge, names, orientations = _synthetic(499)
    with pytest.raises(ValueError, match="at least 500 complete emissions"):
        RegimeHeadModel().fit(
            features,
            binary,
            edge,
            head_name="too_short",
            feature_names=names,
            stress_orientations=orientations,
            fit_metadata=_metadata(),
        )

    features, binary, edge, names, orientations = _synthetic()
    binary[:270] = np.nan
    edge[:270] = np.nan
    with pytest.raises(ValueError, match="at least 500 labeled rows"):
        RegimeHeadModel().fit(
            features,
            binary,
            edge,
            head_name="too_few_labels",
            feature_names=names,
            stress_orientations=orientations,
            fit_metadata=_metadata(),
        )

    features, binary, edge, names, orientations = _synthetic()
    features[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinities"):
        RegimeHeadModel().fit(
            features,
            binary,
            edge,
            head_name="infinite",
            feature_names=names,
            stress_orientations=orientations,
            fit_metadata=_metadata(),
        )

    state = np.tile([0.99, 0.01], (600, 1))
    target = (np.arange(600) % 2).astype(float)
    with pytest.raises(ValueError, match="occupancy"):
        RegimeOutcomeHead().fit(state, target, target)


def test_likelihood_guard_and_state_tie_order_match_frozen_contract() -> None:
    previous = -1000.0
    allowance = 1e-9 * (1.0 + abs(previous))
    assert not _likelihood_decreased_beyond_guard(
        previous, previous - 0.99 * allowance
    )
    assert _likelihood_decreased_beyond_guard(previous, previous - 1.01 * allowance)

    pi = np.asarray([0.4, 0.6])
    transition = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    means = np.asarray([[2.0, 0.0], [1.0, 1.0]])
    variances = np.ones((2, 2))
    # Equal oriented averages; raw lexicographic means put [1, 1] first.
    ordered = _oriented_order(
        pi, transition, means, variances, np.asarray([1.0, 1.0])
    )
    np.testing.assert_array_equal(ordered[2], means[[1, 0]])


def test_long_realistic_multifeature_fit_guards_map_objective_not_raw_likelihood() -> None:
    dates = pd.bdate_range("2000-01-03", "2004-12-30")
    position = np.arange(len(dates), dtype=float)
    market_state = ((position // 90) % 2).astype(int)
    aapl_return = np.where(market_state == 0, 0.0008, -0.0004) + 0.0002 * np.sin(
        position / 7.0
    )
    qqq_return = np.where(market_state == 0, 0.0005, -0.0002) + 0.0001 * np.cos(
        position / 11.0
    )
    spy_return = 0.7 * qqq_return
    iwm_return = np.where(market_state == 0, 0.0004, -0.0003) + 0.0001 * np.sin(
        position / 9.0
    )
    aapl = 50.0 * np.exp(np.cumsum(aapl_return))
    prices = pd.DataFrame(
        {
            "aapl_open": aapl * np.exp(0.0001 * np.sin(position / 5.0)),
            "aapl_close": aapl,
            "aapl_adj_close": aapl,
            "spy_adj_close": 100.0 * np.exp(np.cumsum(spy_return)),
            "qqq_adj_close": 80.0 * np.exp(np.cumsum(qqq_return)),
        },
        index=dates,
    )
    context = pd.DataFrame(
        {
            "iwm_adj_close": 70.0 * np.exp(np.cumsum(iwm_return)),
            "vix_close": np.where(market_state == 0, 15.0, 29.0)
            + np.sin(position / 13.0),
        },
        index=dates,
    )
    feature_label = build_regime_consensus_feature_label_frame(prices, context)
    feature_columns = HEAD_FEATURE_COLUMNS[0]
    orientations = HEAD_ORIENTATIONS[0]
    features = feature_label.loc[:, list(feature_columns)]
    row_number = np.arange(len(features))
    binary = ((row_number % 5) < 2).astype(float)
    edge = np.where(binary == 1.0, 0.002, -0.001)
    binary[-2:] = np.nan
    edge[-2:] = np.nan

    # Replay the actual M-step path and prove the old raw-likelihood guard
    # would reject it even though the exact pseudo-count MAP objective rises.
    probe = RegimeHeadModel()
    matrix = features.to_numpy(dtype=float)
    complete = np.isfinite(matrix).all(axis=1)
    observations = matrix[complete]
    medians = np.median(observations, axis=0)
    scales = np.maximum(
        1.4826 * np.median(np.abs(observations - medians), axis=0), 1e-6
    )
    scaled = np.full_like(matrix, np.nan)
    scaled[complete] = np.clip((observations - medians) / scales, -6.0, 6.0)
    slices = _sequence_slices(complete)
    orientation_array = np.asarray(orientations, dtype=float)
    pi, transition, means, variances = probe._initial_parameters(
        scaled, complete, slices, orientation_array, probe.config
    )
    previous_raw, _, _, _ = probe._expectation(
        scaled, slices, pi, transition, means, variances
    )
    previous_objective = _regularized_log_objective(
        previous_raw, pi, transition, probe.config
    )
    raw_guard_would_fail = False
    for _ in range(20):
        _, gammas, transition_counts, pi_counts = probe._expectation(
            scaled, slices, pi, transition, means, variances
        )
        pi, transition, means, variances, _ = probe._maximization(
            scaled,
            slices,
            gammas,
            transition_counts,
            pi_counts,
            orientation_array,
            probe.config,
        )
        current_raw, _, _, _ = probe._expectation(
            scaled, slices, pi, transition, means, variances
        )
        current_objective = _regularized_log_objective(
            current_raw, pi, transition, probe.config
        )
        if _likelihood_decreased_beyond_guard(previous_raw, current_raw):
            raw_guard_would_fail = True
            assert current_objective > previous_objective
        assert not _likelihood_decreased_beyond_guard(
            previous_objective, current_objective
        )
        previous_raw = current_raw
        previous_objective = current_objective
    assert raw_guard_would_fail

    metadata = _metadata()
    metadata["training_end_date"] = dates[-1].date().isoformat()
    model = RegimeHeadModel().fit(
        features,
        binary,
        edge,
        head_name="aapl_state",
        feature_names=feature_columns,
        stress_orientations=orientations,
        fit_metadata=metadata,
    )
    assert model.converged
    assert model.final_regularized_objective == pytest.approx(
        _regularized_log_objective(
            model.final_log_likelihood,
            model.initial_probabilities,
            model.transition_matrix,
            model.config,
        ),
        abs=1e-12,
    )


def test_implementation_uses_only_the_local_numeric_stack() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "agent_benchmark" / "regime_hmm.py"
    ).read_text(encoding="utf-8").lower()
    forbidden = (("sk" + "learn"), ("sci" + "py"), "hmmlearn")
    assert not any(name in source for name in forbidden)
