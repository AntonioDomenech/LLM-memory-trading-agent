from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_TYPE = "aapl_five_session_downside_forest"
STATE_SCHEMA_VERSION = 1
LABEL_SIMPLE_RETURN_CUTOFF = -0.04
LABEL_LOG_RETURN_CUTOFF = math.log1p(LABEL_SIMPLE_RETURN_CUTOFF)
FORWARD_SESSIONS = 5
ENTRY_OFFSET = 1
EXIT_OFFSET = ENTRY_OFFSET + FORWARD_SESSIONS


@dataclass(frozen=True)
class ForestConfig:
    """Frozen architecture for the first downside-forest family.

    The fields are serialized to make the fitted artifact self-describing, but
    ``validate`` deliberately rejects alternative values. A new architecture
    therefore needs a new model type/version rather than an in-place tweak.
    """

    tree_count: int = 31
    max_depth: int = 2
    block_length: int = 20
    bootstrap_fraction: float = 0.75
    features_per_node: int = 4
    split_quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    min_samples_leaf: int = 64
    prior_strength: float = 24.0
    clipped_return_lower: float = -0.15
    clipped_return_upper: float = 0.15
    label_log_return_cutoff: float = LABEL_LOG_RETURN_CUTOFF
    min_weighted_gini_gain: float = 1e-6
    seed: int = 1729

    def validate(self) -> None:
        expected = ForestConfig()
        # Canonical JSON distinguishes an integer from a numerically equal
        # float, which ordinary dataclass/dict equality does not. This makes
        # type-changing state edits fail the frozen-contract check too.
        if _canonical_json(asdict(self)) != _canonical_json(asdict(expected)):
            raise ValueError(
                "Forest configuration does not match the frozen downside-forest-v1 contract"
            )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _expect_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"Invalid {location} keys; missing={missing}, extra={extra}"
        )


def _finite_float(value: Any, location: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    return result


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def five_session_open_log_returns(adjusted_opens: Sequence[float]) -> np.ndarray:
    """Return the close-t label measured from open t+1 through open t+6.

    The final six decision rows remain NaN because their exit open is not in
    the supplied bounded snapshot. This explicit alignment lets callers purge
    on the label-availability row rather than guessing with calendar days.
    """

    opens = np.asarray(adjusted_opens, dtype=float)
    if opens.ndim != 1:
        raise ValueError("Adjusted opens must be one-dimensional")
    if len(opens) < EXIT_OFFSET + 1:
        raise ValueError("At least seven adjusted opens are required")
    if not np.isfinite(opens).all() or (opens <= 0.0).any():
        raise ValueError("Adjusted opens must be finite and strictly positive")
    result = np.full(len(opens), np.nan, dtype=float)
    mature_count = len(opens) - EXIT_OFFSET
    entry = opens[ENTRY_OFFSET : ENTRY_OFFSET + mature_count]
    exit_ = opens[EXIT_OFFSET:]
    result[:mature_count] = np.log(exit_ / entry)
    return result


def five_session_label_available_indices(observation_count: int) -> np.ndarray:
    """Map each decision row to the row at which its exit open is observable."""

    if isinstance(observation_count, bool) or not isinstance(observation_count, int):
        raise ValueError("observation_count must be an integer")
    if observation_count < EXIT_OFFSET + 1:
        raise ValueError("At least seven observations are required")
    result = np.full(observation_count, -1, dtype=np.int64)
    mature_count = observation_count - EXIT_OFFSET
    result[:mature_count] = np.arange(EXIT_OFFSET, observation_count, dtype=np.int64)
    return result


def downside_labels(future_log_returns: Sequence[float]) -> np.ndarray:
    """Convert mature five-session returns to the frozen binary class label."""

    returns = np.asarray(future_log_returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError("Future log returns must be one-dimensional")
    result = np.full(len(returns), np.nan, dtype=float)
    mature = np.isfinite(returns)
    result[mature] = (returns[mature] <= LABEL_LOG_RETURN_CUTOFF).astype(float)
    return result


def _seed_for_tree(seed: int, namespace: str, tree_index: int) -> int:
    material = f"{MODEL_TYPE}|{seed}|{namespace}|{tree_index}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _block_bootstrap_indices(
    observation_count: int,
    *,
    target_count: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    remaining = target_count
    maximum_start = observation_count - block_length
    while remaining > 0:
        start = int(rng.integers(0, maximum_start + 1))
        take = min(block_length, remaining)
        chunks.append(np.arange(start, start + take, dtype=np.int64))
        remaining -= take
    return np.concatenate(chunks)


def _rank_thresholds(values: np.ndarray, quantiles: Sequence[float]) -> tuple[float, ...]:
    ordered = np.sort(values.astype(float, copy=False), kind="mergesort")
    thresholds: list[float] = []
    for quantile in quantiles:
        rank = int(math.floor(float(quantile) * (len(ordered) - 1)))
        candidate = float(ordered[rank])
        if candidate <= ordered[0] or candidate >= ordered[-1]:
            continue
        if not thresholds or candidate != thresholds[-1]:
            thresholds.append(candidate)
    return tuple(thresholds)


def _weighted_gini(labels: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    positive = float(weights[labels].sum())
    probability = positive / total
    return 2.0 * probability * (1.0 - probability)


class DownsideForest:
    """Deterministic, shallow, moving-block bagged classifier/regressor."""

    def __init__(self, config: ForestConfig | None = None) -> None:
        self.config = config or ForestConfig()
        self.config.validate()
        self.feature_names: tuple[str, ...] = ()
        self.fit_namespace = ""
        self.training_sample_count = 0
        self.training_positive_count = 0
        self.global_prevalence = 0.0
        self.global_mean_clipped_return = 0.0
        self.trees: list[dict[str, Any]] = []

    @property
    def is_fitted(self) -> bool:
        return bool(self.trees)

    @property
    def model_sha256(self) -> str:
        self._require_fitted()
        return _sha256_json(self._payload())

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Downside forest is not fitted")

    @staticmethod
    def _coerce_training_matrix(
        features: Any,
        feature_names: Sequence[str] | None,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        inferred_names: tuple[str, ...] | None = None
        if hasattr(features, "columns") and hasattr(features, "to_numpy"):
            inferred_names = tuple(str(value) for value in features.columns)
            matrix = np.asarray(features.to_numpy(dtype=float), dtype=float)
        else:
            matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Feature matrix must be two-dimensional")
        if feature_names is None:
            if inferred_names is None:
                raise ValueError("feature_names are required for an array feature matrix")
            names = inferred_names
        else:
            names = tuple(str(value) for value in feature_names)
            if inferred_names is not None and names != inferred_names:
                raise ValueError("Provided feature_names do not match the table columns")
        if len(names) != matrix.shape[1]:
            raise ValueError("Feature-name count does not match the feature matrix")
        if len(names) < ForestConfig().features_per_node:
            raise ValueError("At least four features are required by the frozen forest")
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Feature names must be unique, non-empty strings")
        if not np.isfinite(matrix).all():
            raise ValueError("Training features must all be finite")
        return matrix, names

    def _coerce_prediction_matrix(self, features: Any) -> np.ndarray:
        inferred_names: tuple[str, ...] | None = None
        if hasattr(features, "columns") and hasattr(features, "to_numpy"):
            inferred_names = tuple(str(value) for value in features.columns)
            matrix = np.asarray(features.to_numpy(dtype=float), dtype=float)
        else:
            matrix = np.asarray(features, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("Prediction feature matrix does not match the fitted schema")
        if inferred_names is not None and inferred_names != self.feature_names:
            raise ValueError("Prediction table columns do not match the fitted feature schema")
        if not np.isfinite(matrix).all():
            raise ValueError("Prediction features must all be finite")
        return matrix

    def fit(
        self,
        features: Any,
        future_log_returns: Sequence[float],
        *,
        feature_names: Sequence[str] | None = None,
        namespace: str = "development",
    ) -> "DownsideForest":
        self.config.validate()
        matrix, names = self._coerce_training_matrix(features, feature_names)
        returns = np.asarray(future_log_returns, dtype=float)
        if returns.ndim != 1 or len(returns) != len(matrix):
            raise ValueError("Future log returns must have one value per training row")
        if not np.isfinite(returns).all():
            raise ValueError("Only causally mature, finite future returns may train the forest")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("A non-empty deterministic fit namespace is required")
        target_count = int(math.ceil(len(matrix) * self.config.bootstrap_fraction))
        if target_count < 2 * self.config.min_samples_leaf:
            raise ValueError(
                "Training data is too small for two minimum-size leaves after bootstrapping"
            )

        labels = returns <= self.config.label_log_return_cutoff
        clipped_returns = np.clip(
            returns,
            self.config.clipped_return_lower,
            self.config.clipped_return_upper,
        )
        positive_count = int(labels.sum())
        negative_count = int(len(labels) - positive_count)
        if positive_count and negative_count:
            positive_weight = len(labels) / (2.0 * positive_count)
            negative_weight = len(labels) / (2.0 * negative_count)
            weights = np.where(labels, positive_weight, negative_weight).astype(float)
        else:
            weights = np.ones(len(labels), dtype=float)

        self.feature_names = names
        self.fit_namespace = namespace.strip()
        self.training_sample_count = int(len(matrix))
        self.training_positive_count = positive_count
        self.global_prevalence = float(positive_count / len(labels))
        self.global_mean_clipped_return = float(clipped_returns.mean())
        self.trees = []
        for tree_index in range(self.config.tree_count):
            rng = np.random.default_rng(
                _seed_for_tree(self.config.seed, self.fit_namespace, tree_index)
            )
            indices = _block_bootstrap_indices(
                len(matrix),
                target_count=target_count,
                block_length=self.config.block_length,
                rng=rng,
            )
            tree = self._grow_tree(
                matrix,
                labels,
                clipped_returns,
                weights,
                indices,
                depth=0,
                rng=rng,
            )
            self.trees.append(tree)
        self._validate_fitted_semantics()
        return self

    def _node_counts(
        self,
        labels: np.ndarray,
        clipped_returns: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[int, int, float]:
        return (
            int(len(indices)),
            int(labels[indices].sum()),
            float(clipped_returns[indices].sum()),
        )

    def _leaf(
        self,
        labels: np.ndarray,
        clipped_returns: np.ndarray,
        indices: np.ndarray,
    ) -> dict[str, Any]:
        sample_count, positive_count, clipped_sum = self._node_counts(
            labels, clipped_returns, indices
        )
        denominator = sample_count + self.config.prior_strength
        probability = (
            positive_count + self.config.prior_strength * self.global_prevalence
        ) / denominator
        mean_return = (
            clipped_sum
            + self.config.prior_strength * self.global_mean_clipped_return
        ) / denominator
        return {
            "kind": "leaf",
            "sample_count": sample_count,
            "positive_count": positive_count,
            "clipped_return_sum": clipped_sum,
            "downside_probability": float(probability),
            "mean_clipped_return": float(mean_return),
        }

    def _grow_tree(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
        clipped_returns: np.ndarray,
        weights: np.ndarray,
        indices: np.ndarray,
        *,
        depth: int,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        if (
            depth >= self.config.max_depth
            or len(indices) < 2 * self.config.min_samples_leaf
            or labels[indices].all()
            or not labels[indices].any()
        ):
            return self._leaf(labels, clipped_returns, indices)

        node_labels = labels[indices]
        node_weights = weights[indices]
        parent_weight = float(node_weights.sum())
        parent_impurity = _weighted_gini(node_labels, node_weights)
        feature_indices = np.sort(
            rng.choice(
                matrix.shape[1],
                size=self.config.features_per_node,
                replace=False,
            )
        )
        best: tuple[float, int, float, np.ndarray] | None = None
        for feature_index_value in feature_indices:
            feature_index = int(feature_index_value)
            node_values = matrix[indices, feature_index]
            for threshold in _rank_thresholds(
                node_values, self.config.split_quantiles
            ):
                left_mask = node_values <= threshold
                left_count = int(left_mask.sum())
                right_count = int(len(indices) - left_count)
                if (
                    left_count < self.config.min_samples_leaf
                    or right_count < self.config.min_samples_leaf
                ):
                    continue
                left_weights = node_weights[left_mask]
                right_weights = node_weights[~left_mask]
                left_weight = float(left_weights.sum())
                right_weight = float(right_weights.sum())
                impurity = (
                    left_weight
                    * _weighted_gini(node_labels[left_mask], left_weights)
                    + right_weight
                    * _weighted_gini(node_labels[~left_mask], right_weights)
                ) / parent_weight
                gain = float(parent_impurity - impurity)
                if gain < self.config.min_weighted_gini_gain:
                    continue
                if best is None:
                    best = (gain, feature_index, threshold, left_mask)
                    continue
                best_gain, best_feature, best_threshold, _ = best
                better_gain = gain > best_gain + 1e-15
                tied_gain = abs(gain - best_gain) <= 1e-15
                better_tie = (feature_index, threshold) < (
                    best_feature,
                    best_threshold,
                )
                if better_gain or (tied_gain and better_tie):
                    best = (gain, feature_index, threshold, left_mask)

        if best is None:
            return self._leaf(labels, clipped_returns, indices)
        gain, feature_index, threshold, left_mask = best
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        sample_count, positive_count, clipped_sum = self._node_counts(
            labels, clipped_returns, indices
        )
        return {
            "kind": "split",
            "sample_count": sample_count,
            "positive_count": positive_count,
            "clipped_return_sum": clipped_sum,
            "feature_index": feature_index,
            "feature_name": self.feature_names[feature_index],
            "threshold": float(threshold),
            "weighted_gini_gain": float(gain),
            "left": self._grow_tree(
                matrix,
                labels,
                clipped_returns,
                weights,
                left_indices,
                depth=depth + 1,
                rng=rng,
            ),
            "right": self._grow_tree(
                matrix,
                labels,
                clipped_returns,
                weights,
                right_indices,
                depth=depth + 1,
                rng=rng,
            ),
        }

    @staticmethod
    def _predict_tree(node: Mapping[str, Any], row: np.ndarray) -> tuple[float, float]:
        current = node
        while current["kind"] == "split":
            feature_index = int(current["feature_index"])
            current = (
                current["left"]
                if row[feature_index] <= float(current["threshold"])
                else current["right"]
            )
        return (
            float(current["downside_probability"]),
            float(current["mean_clipped_return"]),
        )

    def predict_components(self, features: Any) -> dict[str, np.ndarray]:
        self._require_fitted()
        matrix = self._coerce_prediction_matrix(features)
        probabilities = np.zeros(len(matrix), dtype=float)
        mean_returns = np.zeros(len(matrix), dtype=float)
        for tree in self.trees:
            for row_index, row in enumerate(matrix):
                probability, mean_return = self._predict_tree(tree, row)
                probabilities[row_index] += probability
                mean_returns[row_index] += mean_return
        probabilities /= len(self.trees)
        mean_returns /= len(self.trees)
        return {
            "downside_probability": probabilities,
            "mean_clipped_return": mean_returns,
        }

    def predict_proba(self, features: Any) -> np.ndarray:
        return self.predict_components(features)["downside_probability"]

    def predict_mean_clipped_return(self, features: Any) -> np.ndarray:
        return self.predict_components(features)["mean_clipped_return"]

    def _config_payload(self) -> dict[str, Any]:
        payload = asdict(self.config)
        payload["split_quantiles"] = list(self.config.split_quantiles)
        return payload

    def _payload(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "label_contract": {
                "decision_time": "completed_close_t",
                "entry_open_offset": ENTRY_OFFSET,
                "exit_open_offset": EXIT_OFFSET,
                "forward_sessions": FORWARD_SESSIONS,
                "positive_when_simple_return_lte": LABEL_SIMPLE_RETURN_CUTOFF,
                "positive_when_log_return_lte": LABEL_LOG_RETURN_CUTOFF,
            },
            "config": self._config_payload(),
            "feature_names": list(self.feature_names),
            "training": {
                "fit_namespace": self.fit_namespace,
                "sample_count": self.training_sample_count,
                "positive_count": self.training_positive_count,
                "global_prevalence": self.global_prevalence,
                "global_mean_clipped_return": self.global_mean_clipped_return,
            },
            "trees": self.trees,
        }

    def to_state(self) -> dict[str, Any]:
        payload = self._payload()
        return json.loads(
            _canonical_json(
                {
                    "payload": payload,
                    "payload_sha256": _sha256_json(payload),
                }
            )
        )

    def canonical_json(self) -> str:
        return _canonical_json(self.to_state())

    def save(self, path: str | os.PathLike[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes((self.canonical_json() + "\n").encode("utf-8"))
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "DownsideForest":
        source = Path(path)
        try:
            state = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Could not read a valid downside-forest state") from exc
        return cls.from_state(state)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DownsideForest":
        if not isinstance(state, Mapping):
            raise ValueError("Downside-forest state must be an object")
        _expect_keys(state, {"payload", "payload_sha256"}, "state")
        payload = state["payload"]
        if not isinstance(payload, Mapping):
            raise ValueError("Downside-forest payload must be an object")
        expected_hash = state["payload_sha256"]
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError("Downside-forest payload hash is invalid")
        if not hmac.compare_digest(_sha256_json(payload), expected_hash):
            raise ValueError("Downside-forest state checksum mismatch")
        _expect_keys(
            payload,
            {
                "state_schema_version",
                "model_type",
                "label_contract",
                "config",
                "feature_names",
                "training",
                "trees",
            },
            "payload",
        )
        if (
            isinstance(payload["state_schema_version"], bool)
            or not isinstance(payload["state_schema_version"], int)
            or payload["state_schema_version"] != STATE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported downside-forest state schema")
        if payload["model_type"] != MODEL_TYPE:
            raise ValueError("Unexpected downside-forest model type")

        label_contract = payload["label_contract"]
        expected_label_contract = {
            "decision_time": "completed_close_t",
            "entry_open_offset": ENTRY_OFFSET,
            "exit_open_offset": EXIT_OFFSET,
            "forward_sessions": FORWARD_SESSIONS,
            "positive_when_simple_return_lte": LABEL_SIMPLE_RETURN_CUTOFF,
            "positive_when_log_return_lte": LABEL_LOG_RETURN_CUTOFF,
        }
        if _canonical_json(label_contract) != _canonical_json(expected_label_contract):
            raise ValueError("Downside-forest label contract was altered")

        raw_config = payload["config"]
        if not isinstance(raw_config, Mapping):
            raise ValueError("Downside-forest config must be an object")
        _expect_keys(raw_config, set(asdict(ForestConfig())), "config")
        try:
            config = ForestConfig(
                **{
                    **dict(raw_config),
                    "split_quantiles": tuple(raw_config["split_quantiles"]),
                }
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Downside-forest config could not be decoded") from exc
        config.validate()

        raw_names = payload["feature_names"]
        if not isinstance(raw_names, list):
            raise ValueError("Downside-forest feature_names must be a list")
        names = tuple(raw_names)
        if (
            len(names) < config.features_per_node
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Downside-forest feature schema is invalid")

        training = payload["training"]
        if not isinstance(training, Mapping):
            raise ValueError("Downside-forest training metadata must be an object")
        _expect_keys(
            training,
            {
                "fit_namespace",
                "sample_count",
                "positive_count",
                "global_prevalence",
                "global_mean_clipped_return",
            },
            "training metadata",
        )
        fit_namespace = training["fit_namespace"]
        if not isinstance(fit_namespace, str) or not fit_namespace:
            raise ValueError("Downside-forest fit namespace is invalid")
        sample_count = _strict_int(training["sample_count"], "training sample_count", minimum=1)
        positive_count = _strict_int(
            training["positive_count"], "training positive_count", minimum=0
        )
        if positive_count > sample_count:
            raise ValueError("Training positive_count exceeds sample_count")
        global_prevalence = _finite_float(
            training["global_prevalence"], "training global_prevalence"
        )
        expected_prevalence = positive_count / sample_count
        if not math.isclose(
            global_prevalence,
            expected_prevalence,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Training prevalence is inconsistent with counts")
        global_mean = _finite_float(
            training["global_mean_clipped_return"],
            "training global_mean_clipped_return",
        )
        if not config.clipped_return_lower <= global_mean <= config.clipped_return_upper:
            raise ValueError("Training clipped-return mean is outside the frozen bounds")

        trees = payload["trees"]
        if not isinstance(trees, list) or len(trees) != config.tree_count:
            raise ValueError("Downside-forest state must contain exactly 31 trees")
        forest = cls(config)
        forest.feature_names = names
        forest.fit_namespace = fit_namespace
        forest.training_sample_count = sample_count
        forest.training_positive_count = positive_count
        forest.global_prevalence = global_prevalence
        forest.global_mean_clipped_return = global_mean
        forest.trees = json.loads(_canonical_json(trees))
        forest._validate_fitted_semantics()
        if forest.model_sha256 != expected_hash:
            raise ValueError("Decoded downside-forest model hash changed unexpectedly")
        return forest

    def _validate_fitted_semantics(self) -> None:
        self.config.validate()
        if len(self.trees) != self.config.tree_count:
            raise ValueError("Fitted forest must contain exactly 31 trees")
        if len(self.feature_names) < self.config.features_per_node:
            raise ValueError("Fitted forest has too few features")
        expected_root_count = int(
            math.ceil(self.training_sample_count * self.config.bootstrap_fraction)
        )
        for tree_index, tree in enumerate(self.trees):
            sample_count, _, _ = self._validate_node(
                tree,
                depth=0,
                location=f"tree[{tree_index}]",
            )
            if sample_count != expected_root_count:
                raise ValueError("Tree root sample_count violates the bootstrap fraction")

    def _validate_node(
        self,
        node: Mapping[str, Any],
        *,
        depth: int,
        location: str,
    ) -> tuple[int, int, float]:
        if not isinstance(node, Mapping):
            raise ValueError(f"{location} must be an object")
        kind = node.get("kind")
        common = {"kind", "sample_count", "positive_count", "clipped_return_sum"}
        if kind == "leaf":
            _expect_keys(
                node,
                common | {"downside_probability", "mean_clipped_return"},
                location,
            )
        elif kind == "split":
            _expect_keys(
                node,
                common
                | {
                    "feature_index",
                    "feature_name",
                    "threshold",
                    "weighted_gini_gain",
                    "left",
                    "right",
                },
                location,
            )
        else:
            raise ValueError(f"{location} has an invalid node kind")

        sample_count = _strict_int(node["sample_count"], f"{location}.sample_count", minimum=1)
        positive_count = _strict_int(
            node["positive_count"], f"{location}.positive_count", minimum=0
        )
        if positive_count > sample_count:
            raise ValueError(f"{location} has more positives than samples")
        clipped_sum = _finite_float(
            node["clipped_return_sum"], f"{location}.clipped_return_sum"
        )
        lower_sum = sample_count * self.config.clipped_return_lower
        upper_sum = sample_count * self.config.clipped_return_upper
        if not lower_sum - 1e-12 <= clipped_sum <= upper_sum + 1e-12:
            raise ValueError(f"{location} clipped-return sum is outside its bounds")
        if depth > 0 and sample_count < self.config.min_samples_leaf:
            raise ValueError(f"{location} violates min_samples_leaf")

        if kind == "leaf":
            probability = _finite_float(
                node["downside_probability"], f"{location}.downside_probability"
            )
            mean_return = _finite_float(
                node["mean_clipped_return"], f"{location}.mean_clipped_return"
            )
            denominator = sample_count + self.config.prior_strength
            expected_probability = (
                positive_count
                + self.config.prior_strength * self.global_prevalence
            ) / denominator
            expected_mean = (
                clipped_sum
                + self.config.prior_strength * self.global_mean_clipped_return
            ) / denominator
            if not math.isclose(
                probability,
                expected_probability,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"{location} probability is inconsistent with its leaf data")
            if not math.isclose(
                mean_return,
                expected_mean,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"{location} mean return is inconsistent with its leaf data")
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{location} probability is outside [0, 1]")
            if not self.config.clipped_return_lower <= mean_return <= self.config.clipped_return_upper:
                raise ValueError(f"{location} mean return is outside clipped bounds")
            return sample_count, positive_count, clipped_sum

        if depth >= self.config.max_depth:
            raise ValueError(f"{location} exceeds max_depth")
        feature_index = _strict_int(
            node["feature_index"], f"{location}.feature_index", minimum=0
        )
        if feature_index >= len(self.feature_names):
            raise ValueError(f"{location} feature index is outside the schema")
        if node["feature_name"] != self.feature_names[feature_index]:
            raise ValueError(f"{location} feature name/index do not agree")
        _finite_float(node["threshold"], f"{location}.threshold")
        gain = _finite_float(
            node["weighted_gini_gain"], f"{location}.weighted_gini_gain"
        )
        if gain < self.config.min_weighted_gini_gain:
            raise ValueError(f"{location} split gain is below the frozen minimum")
        left = self._validate_node(
            node["left"], depth=depth + 1, location=f"{location}.left"
        )
        right = self._validate_node(
            node["right"], depth=depth + 1, location=f"{location}.right"
        )
        if left[0] + right[0] != sample_count:
            raise ValueError(f"{location} child sample counts do not sum to the parent")
        if left[1] + right[1] != positive_count:
            raise ValueError(f"{location} child positive counts do not sum to the parent")
        if not math.isclose(
            left[2] + right[2],
            clipped_sum,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{location} child return sums do not sum to the parent")
        return sample_count, positive_count, clipped_sum


__all__ = [
    "DownsideForest",
    "ForestConfig",
    "FORWARD_SESSIONS",
    "LABEL_LOG_RETURN_CUTOFF",
    "LABEL_SIMPLE_RETURN_CUTOFF",
    "downside_labels",
    "five_session_label_available_indices",
    "five_session_open_log_returns",
]
