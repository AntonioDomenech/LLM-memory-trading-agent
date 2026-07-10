from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from agent_benchmark.downside_forest import (
    DownsideForest,
    ForestConfig,
    LABEL_LOG_RETURN_CUTOFF,
    LABEL_SIMPLE_RETURN_CUTOFF,
    downside_labels,
    five_session_label_available_indices,
    five_session_open_log_returns,
)


def _synthetic_training_data(
    observation_count: int = 640,
    feature_count: int = 6,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(9281)
    features = rng.normal(size=(observation_count, feature_count))
    returns = rng.normal(0.008, 0.018, size=observation_count)
    downside = (
        ((features[:, 0] > 0.55) & (features[:, 1] < -0.10))
        | ((features[:, 2] < -1.0) & (features[:, 3] > 0.25))
    )
    returns[downside] = rng.normal(-0.065, 0.006, size=int(downside.sum()))
    names = tuple(f"feature_{index}" for index in range(feature_count))
    return features, returns, names


def _canonical_payload_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _walk_nodes(node: dict, depth: int = 0):
    yield node, depth
    if node["kind"] == "split":
        yield from _walk_nodes(node["left"], depth + 1)
        yield from _walk_nodes(node["right"], depth + 1)


def _first_leaf(node: dict) -> dict:
    current = node
    while current["kind"] == "split":
        current = current["left"]
    return current


def test_frozen_config_is_the_exact_small_architecture():
    config = ForestConfig()
    config.validate()
    assert config.tree_count == 31
    assert config.max_depth == 2
    assert config.block_length == 20
    assert config.bootstrap_fraction == pytest.approx(0.75)
    assert config.features_per_node == 4
    assert config.split_quantiles == (0.10, 0.25, 0.50, 0.75, 0.90)
    assert config.min_samples_leaf == 64
    assert config.prior_strength == pytest.approx(24.0)
    assert config.seed == 1729
    with pytest.raises(ValueError, match="frozen"):
        ForestConfig(tree_count=32).validate()


def test_five_session_label_uses_open_t_plus_1_through_open_t_plus_6():
    opens = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 94.0, 95.0, 96.0, 97.0])
    returns = five_session_open_log_returns(opens)
    available = five_session_label_available_indices(len(opens))
    assert returns[0] == pytest.approx(math.log(94.0 / 101.0))
    assert returns[1] == pytest.approx(math.log(95.0 / 102.0))
    assert np.isnan(returns[-6:]).all()
    assert available.tolist() == [6, 7, 8, 9, -1, -1, -1, -1, -1, -1]

    assert LABEL_SIMPLE_RETURN_CUTOFF == pytest.approx(-0.04)
    assert LABEL_LOG_RETURN_CUTOFF == pytest.approx(math.log(0.96))
    labels = downside_labels(
        np.array(
            [
                math.log(0.9599999),
                math.log(0.96),
                math.log(0.9600001),
                np.nan,
            ]
        )
    )
    assert labels[:3].tolist() == [1.0, 1.0, 0.0]
    assert np.isnan(labels[3])


def test_fit_is_byte_deterministic_and_sha_seed_namespace_is_bound():
    features, returns, names = _synthetic_training_data()
    first = DownsideForest().fit(
        features,
        returns,
        feature_names=names,
        namespace="fold-2005-2006",
    )
    repeated = DownsideForest().fit(
        features,
        returns,
        feature_names=names,
        namespace="fold-2005-2006",
    )
    different_namespace = DownsideForest().fit(
        features,
        returns,
        feature_names=names,
        namespace="fold-2007-2008",
    )
    assert first.canonical_json() == repeated.canonical_json()
    assert first.model_sha256 == repeated.model_sha256
    np.testing.assert_array_equal(
        first.predict_proba(features[:40]), repeated.predict_proba(features[:40])
    )
    assert first.model_sha256 != different_namespace.model_sha256


def test_fitted_trees_obey_depth_leaf_bootstrap_and_schema_contracts():
    features, returns, names = _synthetic_training_data()
    forest = DownsideForest().fit(features, returns, feature_names=names)
    state = forest.to_state()["payload"]
    expected_root_count = math.ceil(len(features) * 0.75)
    assert len(state["trees"]) == 31
    saw_split = False
    for tree in state["trees"]:
        assert tree["sample_count"] == expected_root_count
        for node, depth in _walk_nodes(tree):
            assert depth <= 2
            if depth:
                assert node["sample_count"] >= 64
            if node["kind"] == "split":
                saw_split = True
                assert node["feature_name"] == names[node["feature_index"]]
                assert node["weighted_gini_gain"] >= 1e-6
            else:
                assert 0.0 <= node["downside_probability"] <= 1.0
                assert -0.15 <= node["mean_clipped_return"] <= 0.15
    assert saw_split


def test_leaf_beta_probability_and_clipped_mean_shrinkage_are_exact():
    observation_count = 320
    features = np.zeros((observation_count, 4), dtype=float)
    returns = np.full(observation_count, 0.02, dtype=float)
    returns[:64] = -0.05
    returns[-1] = 9.0  # Must contribute only +0.15 to a leaf return sum.
    forest = DownsideForest().fit(
        features,
        returns,
        feature_names=("a", "b", "c", "d"),
        namespace="constant-features",
    )
    state = forest.to_state()["payload"]
    assert state["training"]["positive_count"] == 64
    assert state["training"]["global_prevalence"] == pytest.approx(0.20)
    tree_probabilities = []
    tree_returns = []
    for tree in state["trees"]:
        assert tree["kind"] == "leaf"
        expected_probability = (
            tree["positive_count"] + 24.0 * 0.20
        ) / (tree["sample_count"] + 24.0)
        expected_mean = (
            tree["clipped_return_sum"]
            + 24.0 * state["training"]["global_mean_clipped_return"]
        ) / (tree["sample_count"] + 24.0)
        assert tree["downside_probability"] == pytest.approx(expected_probability)
        assert tree["mean_clipped_return"] == pytest.approx(expected_mean)
        tree_probabilities.append(expected_probability)
        tree_returns.append(expected_mean)
    assert forest.predict_proba(features[:1])[0] == pytest.approx(
        np.mean(tree_probabilities)
    )
    assert forest.predict_mean_clipped_return(features[:1])[0] == pytest.approx(
        np.mean(tree_returns)
    )


@pytest.mark.parametrize("constant_return, expected", [(0.01, 0.0), (-0.05, 1.0)])
def test_constant_class_training_is_finite(constant_return, expected):
    features = np.zeros((320, 4), dtype=float)
    returns = np.full(320, constant_return, dtype=float)
    forest = DownsideForest().fit(
        features,
        returns,
        feature_names=("a", "b", "c", "d"),
        namespace=f"constant-{expected}",
    )
    probability = forest.predict_proba(features[:5])
    assert np.isfinite(probability).all()
    assert probability == pytest.approx(np.full(5, expected))


def test_probabilities_and_predicted_clipped_returns_are_finite_and_bounded():
    features, returns, names = _synthetic_training_data()
    forest = DownsideForest().fit(features, returns, feature_names=names)
    prediction = forest.predict_components(features)
    probability = prediction["downside_probability"]
    mean_return = prediction["mean_clipped_return"]
    assert np.isfinite(probability).all()
    assert np.isfinite(mean_return).all()
    assert ((0.0 <= probability) & (probability <= 1.0)).all()
    assert ((-0.15 <= mean_return) & (mean_return <= 0.15)).all()
    assert probability[returns <= LABEL_LOG_RETURN_CUTOFF].mean() > probability[
        returns > LABEL_LOG_RETURN_CUTOFF
    ].mean()


def test_save_load_is_byte_stable_and_detects_checksum_tampering(tmp_path):
    features, returns, names = _synthetic_training_data()
    forest = DownsideForest().fit(features, returns, feature_names=names)
    path = tmp_path / "forest.json"
    forest.save(path)
    assert path.read_bytes() == (forest.canonical_json() + "\n").encode("utf-8")
    loaded = DownsideForest.load(path)
    assert loaded.canonical_json() == forest.canonical_json()
    assert loaded.model_sha256 == forest.model_sha256
    np.testing.assert_array_equal(
        loaded.predict_proba(features[:25]), forest.predict_proba(features[:25])
    )

    tampered = copy.deepcopy(forest.to_state())
    leaf = _first_leaf(tampered["payload"]["trees"][0])
    leaf["downside_probability"] += 0.01
    with pytest.raises(ValueError, match="checksum"):
        DownsideForest.from_state(tampered)


def test_rehashed_semantic_tampering_is_rejected():
    features, returns, names = _synthetic_training_data()
    forest = DownsideForest().fit(features, returns, feature_names=names)
    tampered = copy.deepcopy(forest.to_state())
    leaf = _first_leaf(tampered["payload"]["trees"][0])
    leaf["downside_probability"] += 0.01
    tampered["payload_sha256"] = _canonical_payload_hash(tampered["payload"])
    with pytest.raises(ValueError, match="probability is inconsistent"):
        DownsideForest.from_state(tampered)

    small_tamper = copy.deepcopy(forest.to_state())
    small_leaf = _first_leaf(small_tamper["payload"]["trees"][0])
    small_leaf["downside_probability"] += 1e-11
    small_tamper["payload_sha256"] = _canonical_payload_hash(
        small_tamper["payload"]
    )
    with pytest.raises(ValueError, match="probability is inconsistent"):
        DownsideForest.from_state(small_tamper)

    altered_contract = copy.deepcopy(forest.to_state())
    altered_contract["payload"]["config"]["max_depth"] = 3
    altered_contract["payload_sha256"] = _canonical_payload_hash(
        altered_contract["payload"]
    )
    with pytest.raises(ValueError, match="frozen"):
        DownsideForest.from_state(altered_contract)

    altered_type = copy.deepcopy(forest.to_state())
    altered_type["payload"]["config"]["tree_count"] = 31.0
    altered_type["payload_sha256"] = _canonical_payload_hash(altered_type["payload"])
    with pytest.raises(ValueError, match="frozen"):
        DownsideForest.from_state(altered_type)


def test_fit_rejects_unmatured_or_nonfinite_inputs_and_wrong_schema():
    features, returns, names = _synthetic_training_data()
    with pytest.raises(ValueError, match="causally mature"):
        DownsideForest().fit(
            features,
            np.where(np.arange(len(returns)) == len(returns) - 1, np.nan, returns),
            feature_names=names,
        )
    corrupted = features.copy()
    corrupted[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        DownsideForest().fit(corrupted, returns, feature_names=names)
    forest = DownsideForest().fit(features, returns, feature_names=names)
    with pytest.raises(ValueError, match="schema"):
        forest.predict_proba(features[:3, :-1])


def test_implementation_has_no_sklearn_dependency():
    source = (
        Path(__file__).resolve().parents[1]
        / "agent_benchmark"
        / "downside_forest.py"
    ).read_text(encoding="utf-8")
    forbidden = "sk" + "learn"
    assert forbidden not in source.lower()
