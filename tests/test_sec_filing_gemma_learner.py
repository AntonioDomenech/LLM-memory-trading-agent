from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from agent_benchmark.sec_filing_gemma_contract import canonical_sha256
from agent_benchmark.sec_filing_gemma_learner import (
    SecFilingGemmaFitMetadata,
    SecFilingGemmaLearnerConfig,
    SecFilingGemmaLearnerError,
    SecFilingGemmaTwoHeadLearner,
    beta_1_1_climatology,
)


def _synthetic_data(
    row_count: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    trend = np.linspace(-2.0, 2.0, row_count)
    wave = np.sin(np.arange(row_count, dtype=float) * 0.37)
    calendar = (np.arange(row_count) % 5).astype(float) / 4.0
    features = np.column_stack((trend, wave, calendar))
    binary = (
        1.2 * trend - 0.7 * wave + 0.2 * (np.arange(row_count) % 3) > 0.0
    ).astype(float)
    edge = np.clip(-0.04 * trend + 0.015 * wave - 0.005, -0.1, 0.1)
    return features, binary, edge, ["trend", "wave", "calendar"]


def _metadata(
    names: list[str],
    row_count: int,
    *,
    variant: str = "semantic",
    candidate: str = "a",
) -> dict[str, object]:
    return {
        "candidate_sha256": candidate * 64,
        "head_variant": variant,
        "fold_id": "development_fold_1",
        "train_label_maturity_through": "2004-12-31",
        "training_set_sha256": "b" * 64,
        "training_row_count": row_count,
        "maximum_training_label_maturity_session": "2004-12-30",
        "feature_schema_sha256": canonical_sha256(names),
    }


def _fit(
    *, variant: str = "semantic"
) -> tuple[SecFilingGemmaTwoHeadLearner, np.ndarray, np.ndarray, list[str]]:
    features, binary, edge, names = _synthetic_data()
    model = SecFilingGemmaTwoHeadLearner().fit(
        features,
        binary,
        edge,
        feature_names=names,
        fit_metadata=_metadata(names, len(features), variant=variant),
    )
    return model, features, edge, names


def _rehash_state(state: dict[str, object], *, parameters: bool = False) -> None:
    if parameters:
        state["parameters_sha256"] = canonical_sha256(state["parameters"])
    body = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = canonical_sha256(body)


def test_fit_predictions_and_state_are_byte_deterministic() -> None:
    features, binary, edge, names = _synthetic_data()
    metadata = _metadata(names, len(features))
    first = SecFilingGemmaTwoHeadLearner().fit(
        features,
        binary,
        edge,
        feature_names=names,
        fit_metadata=metadata,
    )
    repeated = SecFilingGemmaTwoHeadLearner().fit(
        features.copy(),
        binary.copy(),
        edge.copy(),
        feature_names=list(names),
        fit_metadata=dict(metadata),
    )

    assert first.canonical_json() == repeated.canonical_json()
    assert first.model_sha256 == repeated.model_sha256
    assert json.loads(first.canonical_json()) == first.to_state()
    assert first.logistic_iterations <= 50
    assert 1 <= first.huber_iterations <= 50
    for key, value in first.to_state()["config"].items():
        if key.endswith("iterations") or key == "newton_line_search_max_steps":
            assert isinstance(value, int) and not isinstance(value, bool)
        else:
            assert isinstance(value, str)
            assert float.fromhex(value).hex() == value
    for values in first.to_state()["parameters"].values():
        encoded = values if isinstance(values, list) else [values]
        assert all(float.fromhex(value).hex() == value for value in encoded)

    first_predictions = first.predict_components(features)
    repeated_predictions = repeated.predict_components(features)
    for key in first_predictions:
        np.testing.assert_array_equal(
            first_predictions[key], repeated_predictions[key]
        )
    assert np.all(
        (first_predictions["cash_win_probability_10bps"] > 0.0)
        & (first_predictions["cash_win_probability_10bps"] < 1.0)
    )
    assert np.max(np.abs(first_predictions["expected_active_log_edge_10bps"])) <= 0.5


def test_round_trip_and_metadata_bind_semantic_identity() -> None:
    semantic, features, _, names = _fit(variant="semantic")
    restored = SecFilingGemmaTwoHeadLearner.from_state(semantic.to_state())
    assert restored.canonical_json() == semantic.canonical_json()
    assert restored.model_sha256 == semantic.model_sha256
    for key, values in semantic.predict_components(features).items():
        np.testing.assert_array_equal(restored.predict_components(features)[key], values)

    ablation = SecFilingGemmaTwoHeadLearner().fit(
        features,
        (_synthetic_data()[1]),
        (_synthetic_data()[2]),
        feature_names=names,
        fit_metadata=_metadata(names, len(features), variant="ablation"),
    )
    assert semantic.fit_metadata is not None and ablation.fit_metadata is not None
    assert semantic.fit_metadata.training_row_count == ablation.fit_metadata.training_row_count
    assert (
        semantic.fit_metadata.train_label_maturity_through
        == ablation.fit_metadata.train_label_maturity_through
    )
    assert semantic.model_sha256 != ablation.model_sha256
    for key, values in semantic.predict_components(features).items():
        np.testing.assert_array_equal(ablation.predict_components(features)[key], values)

    rebound = copy.deepcopy(semantic.to_state())
    rebound["fit_metadata"]["candidate_sha256"] = "c" * 64
    _rehash_state(rebound)
    rebound_model = SecFilingGemmaTwoHeadLearner.from_state(rebound)
    assert rebound_model.model_sha256 != semantic.model_sha256
    for key, values in semantic.predict_components(features).items():
        np.testing.assert_array_equal(rebound_model.predict_components(features)[key], values)


def test_robust_scaling_target_clipping_and_climatology_are_frozen() -> None:
    features, binary, edge, names = _synthetic_data()
    features[:, 2] = 7.0
    edge[0], edge[-1] = -9.0, 9.0
    model = SecFilingGemmaTwoHeadLearner().fit(
        features,
        binary,
        edge,
        feature_names=names,
        fit_metadata=_metadata(names, len(features)),
    )
    expected_medians = np.median(features, axis=0)
    expected_scales = np.maximum(
        1.4826 * np.median(np.abs(features - expected_medians), axis=0), 1e-6
    )
    clipped = np.clip(edge, -0.5, 0.5)
    expected_center = float(np.median(clipped))
    expected_target_scale = max(
        1.4826 * float(np.median(np.abs(clipped - expected_center))), 1e-6
    )
    np.testing.assert_array_equal(model.raw_medians, expected_medians)
    np.testing.assert_array_equal(model.raw_scales, expected_scales)
    assert model.raw_scales[2] == 1e-6
    assert model.target_center == expected_center
    assert model.target_scale == expected_target_scale
    assert beta_1_1_climatology(0, 0) == 0.5
    assert beta_1_1_climatology(3, 8) == 0.4
    with pytest.raises(SecFilingGemmaLearnerError, match="counts"):
        beta_1_1_climatology(4, 3)


def test_training_and_prediction_inputs_fail_closed() -> None:
    features, binary, edge, names = _synthetic_data()
    metadata = _metadata(names, len(features))

    bad_features = features.copy()
    bad_features[0, 0] = np.nan
    with pytest.raises(SecFilingGemmaLearnerError, match="finite"):
        SecFilingGemmaTwoHeadLearner().fit(
            bad_features, binary, edge, feature_names=names, fit_metadata=metadata
        )
    with pytest.raises(SecFilingGemmaLearnerError, match="both target classes"):
        SecFilingGemmaTwoHeadLearner().fit(
            features,
            np.zeros(len(features)),
            edge,
            feature_names=names,
            fit_metadata=metadata,
        )
    with pytest.raises(SecFilingGemmaLearnerError, match="zero or one"):
        SecFilingGemmaTwoHeadLearner().fit(
            features,
            np.full(len(features), 0.5),
            edge,
            feature_names=names,
            fit_metadata=metadata,
        )
    wrong_rows = dict(metadata)
    wrong_rows["training_row_count"] = len(features) - 1
    with pytest.raises(SecFilingGemmaLearnerError, match="row count"):
        SecFilingGemmaTwoHeadLearner().fit(
            features, binary, edge, feature_names=names, fit_metadata=wrong_rows
        )
    wrong_schema = dict(metadata)
    wrong_schema["feature_schema_sha256"] = "f" * 64
    with pytest.raises(SecFilingGemmaLearnerError, match="ordered feature schema"):
        SecFilingGemmaTwoHeadLearner().fit(
            features, binary, edge, feature_names=names, fit_metadata=wrong_schema
        )
    late_metadata = dict(metadata)
    late_metadata["maximum_training_label_maturity_session"] = "2005-01-03"
    with pytest.raises(SecFilingGemmaLearnerError, match="exceeds"):
        SecFilingGemmaTwoHeadLearner().fit(
            features, binary, edge, feature_names=names, fit_metadata=late_metadata
        )

    fitted = SecFilingGemmaTwoHeadLearner().fit(
        features, binary, edge, feature_names=names, fit_metadata=metadata
    )
    with pytest.raises(SecFilingGemmaLearnerError, match="schema"):
        fitted.predict_components(features[:, :2])
    with pytest.raises(SecFilingGemmaLearnerError, match="finite"):
        prediction = features[:1].copy()
        prediction[0, 0] = np.inf
        fitted.predict_components(prediction)


def test_dataframe_column_reordering_and_nonfrozen_config_are_rejected() -> None:
    pandas = pytest.importorskip("pandas")
    features, binary, edge, names = _synthetic_data()
    frame = pandas.DataFrame(features, columns=names)
    model = SecFilingGemmaTwoHeadLearner().fit(
        frame,
        binary,
        edge,
        feature_names=names,
        fit_metadata=_metadata(names, len(frame)),
    )
    with pytest.raises(SecFilingGemmaLearnerError, match="reordered"):
        model.predict_components(frame[["wave", "trend", "calendar"]])
    with pytest.raises(SecFilingGemmaLearnerError, match="frozen"):
        SecFilingGemmaTwoHeadLearner(
            SecFilingGemmaLearnerConfig(ridge_lambda=0.2)
        )
    with pytest.raises(SecFilingGemmaLearnerError, match="frozen"):
        SecFilingGemmaTwoHeadLearner(
            SecFilingGemmaLearnerConfig(logistic_max_iterations=50.0)  # type: ignore[arg-type]
        )


def test_checksum_and_semantic_state_tampering_are_rejected() -> None:
    model, _, _, _ = _fit()
    state = model.to_state()

    ordinary = copy.deepcopy(state)
    ordinary["training_positive_count"] += 1
    with pytest.raises(SecFilingGemmaLearnerError, match="checksum"):
        SecFilingGemmaTwoHeadLearner.from_state(ordinary)

    zero_scale = copy.deepcopy(state)
    zero_scale["parameters"]["raw_scales"][0] = 0.0.hex()
    _rehash_state(zero_scale, parameters=True)
    with pytest.raises(SecFilingGemmaLearnerError, match="scale"):
        SecFilingGemmaTwoHeadLearner.from_state(zero_scale)

    target_scale = copy.deepcopy(state)
    target_scale["parameters"]["target_scale"] = (-1.0).hex()
    _rehash_state(target_scale, parameters=True)
    with pytest.raises(SecFilingGemmaLearnerError, match="target scaling"):
        SecFilingGemmaTwoHeadLearner.from_state(target_scale)

    excessive_iterations = copy.deepcopy(state)
    excessive_iterations["huber_iterations"] = 51
    _rehash_state(excessive_iterations)
    with pytest.raises(SecFilingGemmaLearnerError, match="convergence"):
        SecFilingGemmaTwoHeadLearner.from_state(excessive_iterations)

    decimal_config = copy.deepcopy(state)
    decimal_config["config"]["ridge_lambda"] = 0.1
    _rehash_state(decimal_config)
    with pytest.raises(SecFilingGemmaLearnerError, match="hexadecimal"):
        SecFilingGemmaTwoHeadLearner.from_state(decimal_config)

    wrong_schema = copy.deepcopy(state)
    wrong_schema["feature_names"][0] = "invented"
    _rehash_state(wrong_schema)
    with pytest.raises(SecFilingGemmaLearnerError, match="schema hash"):
        SecFilingGemmaTwoHeadLearner.from_state(wrong_schema)


def test_fit_metadata_rejects_noncanonical_hashes_dates_and_variants() -> None:
    _, _, _, names = _synthetic_data()
    base = _metadata(names, 80)
    for key, value, match in (
        ("candidate_sha256", "A" * 64, "SHA-256"),
        ("head_variant", "unknown", "variant"),
        ("train_label_maturity_through", "2004-1-1", "ISO"),
        ("training_row_count", True, "integer"),
    ):
        changed = dict(base)
        changed[key] = value
        with pytest.raises(SecFilingGemmaLearnerError, match=match):
            SecFilingGemmaFitMetadata.coerce(changed)
