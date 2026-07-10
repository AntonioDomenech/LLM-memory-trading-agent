from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from agent_benchmark.rare_loss_forest import (
    MINIMUM_OOB_TREE_COUNT,
    OOB_GATE_QUANTILES,
    SEVERE_LOG_RETURN_CUTOFF,
    SEVERE_SIMPLE_RETURN_CUTOFF,
    RareLossForest,
    RareLossForestConfig,
    _block_bootstrap_indices,
    _higher_quantile,
    _lower_rank_thresholds,
    _seed_for_tree,
    cash_win_labels,
    rare_loss_labels,
)


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
    rows: int = 900,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(70123)
    features = rng.normal(size=(rows, 8))
    # A nonlinear, interaction-only rare event.  Nuisance columns ensure the
    # random feature subsets and bagged trees are genuinely exercised.
    severe = (
        ((features[:, 0] > 0.55) & (features[:, 1] < -0.30))
        | ((features[:, 2] < -1.05) & (features[:, 3] > 0.55))
    )
    ordinary = severe | (
        (features[:, 0] > 0.10)
        & (features[:, 4] > 0.20)
        & (features[:, 5] < 0.70)
    )
    edge = (
        np.where(severe, 0.035, np.where(ordinary, 0.004, -0.006))
        + rng.normal(scale=0.0015, size=rows)
    )
    names = tuple(f"feature_{index}" for index in range(features.shape[1]))
    return (
        features,
        severe.astype(float),
        ordinary.astype(float),
        edge,
        names,
    )


@pytest.fixture(scope="module")
def fitted_forest():
    features, severe, ordinary, edge, names = _synthetic()
    model = RareLossForest().fit(
        features,
        severe,
        ordinary,
        edge,
        feature_names=names,
        namespace="synthetic-core-fold-2005-2006",
        fit_metadata={"fold": "2005-2006", "support": "common-full-ready"},
    )
    return model, features, severe, ordinary, edge, names


def test_frozen_configuration_labels_and_quantile_conventions_are_exact() -> None:
    config = RareLossForestConfig()
    config.validate()
    assert config.tree_count == 127
    assert config.max_depth == 3
    assert config.block_length == 20
    assert config.bootstrap_fraction == 0.75
    assert config.features_per_node == 5
    assert config.split_quantiles == (0.10, 0.25, 0.50, 0.75, 0.90)
    assert config.min_samples_leaf == 64
    assert config.shrinkage_strength == 64.0
    assert config.clipped_edge_lower == -0.08
    assert config.clipped_edge_upper == 0.08
    assert config.seed == 2741
    assert OOB_GATE_QUANTILES == (0.975, 0.99)
    assert MINIMUM_OOB_TREE_COUNT == 16
    assert SEVERE_SIMPLE_RETURN_CUTOFF == -0.025
    assert SEVERE_LOG_RETURN_CUTOFF == pytest.approx(math.log1p(-0.025))
    with pytest.raises(ValueError, match="frozen"):
        RareLossForestConfig(seed=2742).validate()

    returns = np.asarray(
        [SEVERE_LOG_RETURN_CUTOFF - 1e-6, SEVERE_LOG_RETURN_CUTOFF, 0.0, np.nan]
    )
    np.testing.assert_array_equal(
        rare_loss_labels(returns)[:3], np.asarray([1.0, 1.0, 0.0])
    )
    assert np.isnan(rare_loss_labels(returns)[-1])
    edge = np.asarray([-0.1, 0.0, 1e-12, np.nan])
    np.testing.assert_array_equal(cash_win_labels(edge)[:3], [0.0, 0.0, 1.0])
    assert np.isnan(cash_win_labels(edge)[-1])

    values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    assert _lower_rank_thresholds(values, (0.25, 0.50, 0.90)) == (1.0, 2.0, 3.0)
    assert _higher_quantile(values, 0.975) == 4.0


def test_fit_predicts_all_three_outputs_and_learns_nonlinear_tail(
    fitted_forest,
) -> None:
    model, features, severe, _, _, _ = fitted_forest
    assert model.is_fitted
    assert len(model.trees) == 127
    assert model.severe_positive_count == int(severe.sum())
    assert model.global_severe_probability == pytest.approx(severe.mean())
    prediction = model.predict_components(features)
    assert set(prediction) == {
        "severe_probability",
        "ordinary_cash_win_probability",
        "expected_clipped_edge",
        "observation_available",
    }
    assert prediction["observation_available"].all()
    assert np.isfinite(prediction["expected_clipped_edge"]).all()
    for name in ("severe_probability", "ordinary_cash_win_probability"):
        values = prediction[name]
        assert ((0.0 <= values) & (values <= 1.0)).all()
    assert prediction["severe_probability"][severe == 1.0].mean() > (
        prediction["severe_probability"][severe == 0.0].mean() + 0.10
    )
    assert prediction["expected_clipped_edge"][severe == 1.0].mean() > (
        prediction["expected_clipped_edge"][severe == 0.0].mean() + 0.005
    )

    missing = features[:3].copy()
    missing[1, 2] = np.nan
    fallback = model.predict_components(missing)
    assert fallback["observation_available"].tolist() == [True, False, True]
    assert np.isnan(fallback["severe_probability"][1])
    assert np.isnan(fallback["ordinary_cash_win_probability"][1])
    assert np.isnan(fallback["expected_clipped_edge"][1])


def test_leaf_outputs_use_exact_independent_strength_64_shrinkage(
    fitted_forest,
) -> None:
    model, *_ = fitted_forest

    def visit(node: dict) -> int:
        if node["kind"] == "split":
            return visit(node["left"]) + visit(node["right"])
        count = node["sample_count"]
        denominator = count + 64.0
        assert node["severe_loss_probability"] == pytest.approx(
            (
                node["severe_positive_count"]
                + 64.0 * model.global_severe_probability
            )
            / denominator
        )
        assert node["cash_win_probability_10bps"] == pytest.approx(
            (
                node["cash_win_positive_count"]
                + 64.0 * model.global_cash_win_probability
            )
            / denominator
        )
        assert node["expected_edge_10bps"] == pytest.approx(
            (node["clipped_edge_sum"] + 64.0 * model.global_expected_edge)
            / denominator
        )
        return 1

    assert sum(visit(tree) for tree in model.trees) >= 127


def test_oob_predictions_average_only_trees_where_row_was_never_sampled(
    fitted_forest,
) -> None:
    model, features, _, _, _, _ = fitted_forest
    oob = model.oob_components()
    assert set(oob) == {
        "severe_probability",
        "ordinary_cash_win_probability",
        "expected_clipped_edge",
        "oob_tree_count",
    }
    counts = oob["oob_tree_count"]
    assert counts.shape == (len(features),)
    assert (counts > 0).all()
    row = 17
    expected: list[tuple[float, float, float]] = []
    target_count = math.ceil(0.75 * len(features))
    for tree_index, tree in enumerate(model.trees):
        rng = np.random.default_rng(
            _seed_for_tree(model.config.seed, model.namespace, tree_index)
        )
        bootstrap = _block_bootstrap_indices(
            len(features),
            target_count=target_count,
            block_length=20,
            rng=rng,
        )
        if row not in bootstrap:
            expected.append(model._predict_tree(tree, features[row]))
    assert counts[row] == len(expected)
    expected_array = np.asarray(expected)
    assert oob["severe_probability"][row] == pytest.approx(expected_array[:, 0].mean())
    assert oob["ordinary_cash_win_probability"][row] == pytest.approx(
        expected_array[:, 1].mean()
    )
    assert oob["expected_clipped_edge"][row] == pytest.approx(
        expected_array[:, 2].mean()
    )

    eligible = oob["severe_probability"][counts >= 16]
    ordered = np.sort(eligible, kind="mergesort")
    for quantile in OOB_GATE_QUANTILES:
        expected_threshold = ordered[
            math.ceil(quantile * (len(ordered) - 1))
        ]
        assert model.oob_severe_threshold(quantile) == expected_threshold
    with pytest.raises(ValueError, match="one of"):
        model.oob_severe_threshold(0.98)
    with pytest.raises(ValueError, match="frozen at 16"):
        model.oob_severe_threshold(0.975, minimum_tree_count=15)


def test_double_refit_serialization_and_predictions_are_byte_deterministic(
    fitted_forest,
    tmp_path: Path,
) -> None:
    first, features, severe, ordinary, edge, names = fitted_forest
    second = RareLossForest().fit(
        features.copy(),
        severe.copy(),
        ordinary.copy(),
        edge.copy(),
        feature_names=names,
        namespace=first.namespace,
        fit_metadata=dict(first.fit_metadata),
    )
    assert first.canonical_json() == second.canonical_json()
    assert first.model_sha256 == second.model_sha256
    for name, values in first.predict_components(features[:25]).items():
        np.testing.assert_array_equal(values, second.predict_components(features[:25])[name])
    for name, values in first.oob_components().items():
        np.testing.assert_array_equal(values, second.oob_components()[name])

    path = tmp_path / "rare-loss.json"
    first.save(path)
    assert path.read_bytes() == (first.canonical_json() + "\n").encode("utf-8")
    loaded = RareLossForest.load(path)
    assert loaded.canonical_json() == first.canonical_json()
    assert loaded.model_sha256 == first.model_sha256
    for name, values in first.predict_components(features[:25]).items():
        np.testing.assert_array_equal(values, loaded.predict_components(features[:25])[name])


def test_checksum_and_rehashed_semantic_tampering_fail_closed(fitted_forest) -> None:
    model, *_ = fitted_forest
    state = model.to_state()
    ordinary = copy.deepcopy(state)
    ordinary["payload"]["training"]["row_count"] += 1
    with pytest.raises(ValueError, match="checksum"):
        RareLossForest.from_state(ordinary)

    global_probability = copy.deepcopy(state)
    global_probability["payload"]["training"]["global_severe_probability"] = (
        0.9.hex()
    )
    global_probability["payload_sha256"] = _canonical_hash(
        global_probability["payload"]
    )
    with pytest.raises(ValueError, match="raw training priors"):
        RareLossForest.from_state(global_probability)

    leaf = copy.deepcopy(state)
    node = leaf["payload"]["trees"][0]
    while node["kind"] == "split":
        node = node["left"]
    node["expected_edge_10bps"] = 0.079.hex()
    leaf["payload_sha256"] = _canonical_hash(leaf["payload"])
    with pytest.raises(ValueError, match="shrinkage"):
        RareLossForest.from_state(leaf)


def test_invalid_training_schema_classes_and_nonfinite_values_are_rejected() -> None:
    features, severe, ordinary, edge, names = _synthetic(180)
    with pytest.raises(ValueError, match="finite"):
        changed = features.copy()
        changed[0, 0] = np.nan
        RareLossForest().fit(
            changed,
            severe,
            ordinary,
            edge,
            feature_names=names,
            namespace="invalid",
        )
    with pytest.raises(ValueError, match="both classes"):
        RareLossForest().fit(
            features,
            np.zeros(len(features)),
            ordinary,
            edge,
            feature_names=names,
            namespace="invalid",
        )
    with pytest.raises(ValueError, match="both classes"):
        RareLossForest().fit(
            features,
            severe,
            np.ones(len(features)),
            edge,
            feature_names=names,
            namespace="invalid",
        )
    with pytest.raises(ValueError, match="feature_names"):
        RareLossForest().fit(
            features,
            severe,
            ordinary,
            edge,
            namespace="invalid",
        )
    model = RareLossForest().fit(
        features,
        severe,
        ordinary,
        edge,
        feature_names=names,
        namespace="valid-small",
    )
    with pytest.raises(ValueError, match="schema"):
        model.predict_components(features[:, :-1])
    with pytest.raises(ValueError, match="infinite"):
        changed = features[:2].copy()
        changed[0, 0] = np.inf
        model.predict_components(changed)


def test_implementation_uses_only_numpy_and_standard_library() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "agent_benchmark"
        / "rare_loss_forest.py"
    ).read_text(encoding="utf-8").lower()
    forbidden = (("sk" + "learn"), ("sci" + "py"), "xgboost", "lightgbm")
    assert not any(name in source for name in forbidden)

