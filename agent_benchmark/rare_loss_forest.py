"""Deterministic pure-NumPy one-session rare-loss forest.

The forest is deliberately small and frozen.  Class-balanced severe-loss
Gini selects nonlinear partitions; leaves separately estimate severe-loss
probability, ordinary 10-bps CASH-win probability, and expected 10-bps CASH
edge with shrinkage toward full causally eligible training priors.

Every tree uses a contiguous-block bootstrap.  Exact out-of-bag predictions
and counts are retained and serialized so the later 97.5/99 percent tail
gates can be reproduced without looking at validation predictions.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_TYPE = "aapl_one_session_rare_loss_forest"
STATE_SCHEMA_VERSION = 1
SEVERE_SIMPLE_RETURN_CUTOFF = -0.025
SEVERE_LOG_RETURN_CUTOFF = math.log1p(SEVERE_SIMPLE_RETURN_CUTOFF)
OOB_GATE_QUANTILES: tuple[float, float] = (0.975, 0.99)
MINIMUM_OOB_TREE_COUNT = 16
_TIE_TOLERANCE = 1e-15
_MISSING_TOKEN = "missing"


@dataclass(frozen=True)
class RareLossForestConfig:
    tree_count: int = 127
    max_depth: int = 3
    block_length: int = 20
    bootstrap_fraction: float = 0.75
    features_per_node: int = 5
    split_quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    min_samples_leaf: int = 64
    shrinkage_strength: float = 64.0
    clipped_edge_lower: float = -0.08
    clipped_edge_upper: float = 0.08
    severe_log_return_cutoff: float = SEVERE_LOG_RETURN_CUTOFF
    seed: int = 2741

    def validate(self) -> None:
        if _canonical_json(asdict(self)) != _canonical_json(
            asdict(RareLossForestConfig())
        ):
            raise ValueError(
                "Forest configuration does not match the frozen rare-loss-v1 contract"
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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _expect_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"Invalid {location} keys; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _float_hex(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Only finite floats can be encoded")
    return number.hex()


def _decode_float_hex(value: Any, location: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a canonical hexadecimal float")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{location} must be a canonical hexadecimal float") from exc
    if not math.isfinite(result) or result.hex() != value:
        raise ValueError(f"{location} must be a canonical finite hexadecimal float")
    return result


def _encode_optional_vector(values: np.ndarray) -> list[str]:
    return [
        _float_hex(value) if math.isfinite(float(value)) else _MISSING_TOKEN
        for value in np.asarray(values, dtype=float)
    ]


def _decode_optional_vector(values: Any, location: str, length: int) -> np.ndarray:
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{location} must contain exactly {length} values")
    result = np.full(length, np.nan, dtype=float)
    for index, value in enumerate(values):
        if value == _MISSING_TOKEN:
            continue
        result[index] = _decode_float_hex(value, f"{location}[{index}]")
    return result


def _json_safe_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    raw: Mapping[str, Any] = {} if value is None else value
    if not isinstance(raw, Mapping):
        raise ValueError("fit_metadata must be a mapping")
    try:
        return json.loads(_canonical_json(dict(raw)))
    except (TypeError, ValueError) as exc:
        raise ValueError("fit_metadata must be canonical JSON-compatible") from exc


def rare_loss_labels(forward_log_returns: Sequence[float]) -> np.ndarray:
    """Return the frozen severe-loss label, preserving immature NaNs."""

    values = np.asarray(forward_log_returns, dtype=float)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError("forward_log_returns must be one-dimensional without infinity")
    result = np.full(len(values), np.nan, dtype=float)
    mature = np.isfinite(values)
    result[mature] = (values[mature] <= SEVERE_LOG_RETURN_CUTOFF).astype(float)
    return result


def cash_win_labels(edge_10bps: Sequence[float]) -> np.ndarray:
    """Return ordinary CASH-win labels from exact 10-bps active log edge."""

    values = np.asarray(edge_10bps, dtype=float)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError("edge_10bps must be one-dimensional without infinity")
    result = np.full(len(values), np.nan, dtype=float)
    mature = np.isfinite(values)
    result[mature] = (values[mature] > 0.0).astype(float)
    return result


def _seed_for_tree(seed: int, namespace: str, tree_index: int) -> int:
    material = f"{MODEL_TYPE}|{seed}|{namespace}|{tree_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _block_bootstrap_indices(
    observation_count: int,
    *,
    target_count: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if observation_count < block_length:
        raise ValueError("Training rows must cover at least one complete block")
    chunks: list[np.ndarray] = []
    remaining = int(target_count)
    maximum_start = observation_count - block_length
    while remaining:
        start = int(rng.integers(0, maximum_start + 1))
        take = min(block_length, remaining)
        chunks.append(np.arange(start, start + take, dtype=np.int64))
        remaining -= take
    return np.concatenate(chunks)


def _lower_rank_thresholds(
    values: np.ndarray,
    quantiles: Sequence[float],
) -> tuple[float, ...]:
    ordered = np.sort(np.asarray(values, dtype=float), kind="mergesort")
    if not len(ordered):
        return ()
    thresholds: list[float] = []
    for quantile in quantiles:
        rank = int(math.floor(float(quantile) * (len(ordered) - 1)))
        candidate = float(ordered[rank])
        if candidate <= ordered[0] or candidate >= ordered[-1]:
            continue
        if not thresholds or candidate != thresholds[-1]:
            thresholds.append(candidate)
    return tuple(thresholds)


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float), kind="mergesort")
    if not len(ordered):
        raise ValueError("OOB tail threshold requires at least one eligible row")
    rank = int(math.ceil(float(quantile) * (len(ordered) - 1)))
    return float(ordered[rank])


def _weighted_gini(labels: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    positive = float(weights[np.asarray(labels, dtype=bool)].sum())
    probability = positive / total
    return 2.0 * probability * (1.0 - probability)


def _balanced_weights(labels: np.ndarray, sample_indices: np.ndarray) -> np.ndarray:
    binary = np.asarray(labels, dtype=bool)
    sampled = binary[np.asarray(sample_indices, dtype=np.int64)]
    positive = int(sampled.sum())
    negative = len(sampled) - positive
    weights = np.zeros(len(binary), dtype=float)
    if positive:
        weights[binary] = 0.5 / positive
    if negative:
        weights[~binary] = 0.5 / negative
    return weights


class RareLossForest:
    """Frozen deterministic block-bagged rare-loss forest."""

    def __init__(self, config: RareLossForestConfig | None = None) -> None:
        self.config = config or RareLossForestConfig()
        self.config.validate()
        self.namespace = ""
        self.feature_names: tuple[str, ...] = ()
        self.fit_metadata: dict[str, Any] = {}
        self.training_row_count = 0
        self.severe_positive_count = 0
        self.cash_win_positive_count = 0
        self.global_severe_probability = math.nan
        self.global_cash_win_probability = math.nan
        self.global_expected_edge = math.nan
        self.trees: list[dict[str, Any]] = []
        self.oob_tree_counts = np.empty(0, dtype=np.int64)
        self.oob_severe_loss_probability = np.empty(0, dtype=float)
        self.oob_cash_win_probability_10bps = np.empty(0, dtype=float)
        self.oob_expected_edge_10bps = np.empty(0, dtype=float)

    @property
    def is_fitted(self) -> bool:
        return (
            bool(self.namespace)
            and bool(self.feature_names)
            and len(self.trees) == self.config.tree_count
            and self.training_row_count > 0
            and self.oob_tree_counts.shape == (self.training_row_count,)
        )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Rare-loss forest is not fitted")

    @property
    def model_sha256(self) -> str:
        self._require_fitted()
        return _sha256_json(self._payload())

    @staticmethod
    def _coerce_training_matrix(
        features: Any,
        feature_names: Sequence[str] | None,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        inferred: tuple[str, ...] | None = None
        if hasattr(features, "columns") and hasattr(features, "to_numpy"):
            inferred = tuple(str(column) for column in features.columns)
            matrix = np.asarray(features.to_numpy(dtype=float), dtype=float)
        else:
            matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] < 1:
            raise ValueError("Rare-loss features must be a two-dimensional matrix")
        if feature_names is None:
            if inferred is None:
                raise ValueError("feature_names are required for an array matrix")
            names = inferred
        else:
            names = tuple(str(value) for value in feature_names)
            if inferred is not None and inferred != names:
                raise ValueError("Provided feature_names do not match table columns")
        if (
            len(names) != matrix.shape[1]
            or len(names) < RareLossForestConfig().features_per_node
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Feature names must be unique, non-empty, and match the matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("Rare-loss training features must be finite")
        return matrix, names

    def _coerce_prediction_matrix(self, features: Any) -> np.ndarray:
        inferred: tuple[str, ...] | None = None
        if hasattr(features, "columns") and hasattr(features, "to_numpy"):
            inferred = tuple(str(column) for column in features.columns)
            matrix = np.asarray(features.to_numpy(dtype=float), dtype=float)
        else:
            matrix = np.asarray(features, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("Prediction features do not match the fitted schema")
        if inferred is not None and inferred != self.feature_names:
            raise ValueError("Prediction table columns do not match the fitted schema")
        if np.isinf(matrix).any():
            raise ValueError("Prediction features may be NaN but not infinite")
        return matrix

    @staticmethod
    def _binary_target(values: Sequence[float], *, name: str, length: int) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if result.shape != (length,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must contain one finite value per training row")
        if not np.isin(result, (0.0, 1.0)).all():
            raise ValueError(f"{name} must contain exactly zero or one")
        if len(np.unique(result)) != 2:
            raise ValueError(f"{name} requires both classes")
        return result.astype(bool)

    def fit(
        self,
        features: Any,
        severe_labels: Sequence[float],
        ordinary_labels: Sequence[float],
        clipped_edges: Sequence[float],
        *,
        feature_names: Sequence[str] | None = None,
        namespace: str,
        fit_metadata: Mapping[str, Any] | None = None,
    ) -> "RareLossForest":
        self.config.validate()
        matrix, names = self._coerce_training_matrix(features, feature_names)
        if not isinstance(namespace, str) or not namespace or namespace.strip() != namespace:
            raise ValueError("namespace must be a non-empty canonical string")
        count = len(matrix)
        if count < max(
            self.config.block_length,
            2 * self.config.min_samples_leaf,
        ):
            raise ValueError("Rare-loss forest has too few training rows")
        severe = self._binary_target(
            severe_labels, name="severe_labels", length=count
        )
        cash_win = self._binary_target(
            ordinary_labels, name="ordinary_labels", length=count
        )
        raw_edge = np.asarray(clipped_edges, dtype=float)
        if raw_edge.shape != (count,) or not np.isfinite(raw_edge).all():
            raise ValueError("clipped_edges must contain one finite value per training row")
        clipped_edge = np.clip(
            raw_edge,
            self.config.clipped_edge_lower,
            self.config.clipped_edge_upper,
        )
        severe_count = int(severe.sum())
        cash_count = int(cash_win.sum())
        self.namespace = namespace
        self.feature_names = names
        self.fit_metadata = _json_safe_mapping(fit_metadata)
        self.training_row_count = count
        self.severe_positive_count = severe_count
        self.cash_win_positive_count = cash_count
        self.global_severe_probability = float(severe_count / count)
        self.global_cash_win_probability = float(cash_count / count)
        self.global_expected_edge = float(clipped_edge.mean())
        self.trees = []

        oob_severe_sum = np.zeros(count, dtype=float)
        oob_cash_sum = np.zeros(count, dtype=float)
        oob_edge_sum = np.zeros(count, dtype=float)
        oob_counts = np.zeros(count, dtype=np.int64)
        target_count = int(math.ceil(self.config.bootstrap_fraction * count))
        for tree_index in range(self.config.tree_count):
            rng = np.random.default_rng(
                _seed_for_tree(self.config.seed, namespace, tree_index)
            )
            bootstrap = _block_bootstrap_indices(
                count,
                target_count=target_count,
                block_length=self.config.block_length,
                rng=rng,
            )
            weights = _balanced_weights(severe, bootstrap)
            tree = self._build_node(
                matrix,
                severe,
                cash_win,
                clipped_edge,
                weights,
                bootstrap,
                depth=0,
                rng=rng,
            )
            self.trees.append(tree)
            in_bag = np.zeros(count, dtype=bool)
            in_bag[np.unique(bootstrap)] = True
            oob = np.flatnonzero(~in_bag)
            for row_index in oob:
                severe_probability, cash_probability, expected_edge = self._predict_tree(
                    tree, matrix[row_index]
                )
                oob_severe_sum[row_index] += severe_probability
                oob_cash_sum[row_index] += cash_probability
                oob_edge_sum[row_index] += expected_edge
                oob_counts[row_index] += 1

        available = oob_counts > 0
        self.oob_tree_counts = oob_counts
        self.oob_severe_loss_probability = np.full(count, np.nan, dtype=float)
        self.oob_cash_win_probability_10bps = np.full(count, np.nan, dtype=float)
        self.oob_expected_edge_10bps = np.full(count, np.nan, dtype=float)
        self.oob_severe_loss_probability[available] = (
            oob_severe_sum[available] / oob_counts[available]
        )
        self.oob_cash_win_probability_10bps[available] = (
            oob_cash_sum[available] / oob_counts[available]
        )
        self.oob_expected_edge_10bps[available] = (
            oob_edge_sum[available] / oob_counts[available]
        )
        self._validate_fitted_semantics()
        return self

    def _leaf(
        self,
        severe: np.ndarray,
        cash_win: np.ndarray,
        clipped_edge: np.ndarray,
        indices: np.ndarray,
    ) -> dict[str, Any]:
        count = int(len(indices))
        severe_count = int(severe[indices].sum())
        cash_count = int(cash_win[indices].sum())
        edge_sum = float(clipped_edge[indices].sum())
        strength = self.config.shrinkage_strength
        denominator = count + strength
        return {
            "kind": "leaf",
            "sample_count": count,
            "severe_positive_count": severe_count,
            "cash_win_positive_count": cash_count,
            "clipped_edge_sum": edge_sum,
            "severe_loss_probability": float(
                (severe_count + strength * self.global_severe_probability)
                / denominator
            ),
            "cash_win_probability_10bps": float(
                (cash_count + strength * self.global_cash_win_probability)
                / denominator
            ),
            "expected_edge_10bps": float(
                (edge_sum + strength * self.global_expected_edge) / denominator
            ),
        }

    def _build_node(
        self,
        matrix: np.ndarray,
        severe: np.ndarray,
        cash_win: np.ndarray,
        clipped_edge: np.ndarray,
        weights: np.ndarray,
        indices: np.ndarray,
        *,
        depth: int,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        labels = severe[indices]
        if (
            depth >= self.config.max_depth
            or len(indices) < 2 * self.config.min_samples_leaf
            or labels.all()
            or not labels.any()
        ):
            return self._leaf(severe, cash_win, clipped_edge, indices)

        # Rebalance within every splittable node, as frozen: when both
        # classes are present each class contributes total weight 0.5 at that
        # node.  Reusing root weights here would cease to be class-balanced
        # after the first split.
        node_weights = _balanced_weights(severe, indices)[indices]
        parent_weight = float(node_weights.sum())
        parent_impurity = _weighted_gini(labels, node_weights)
        selected_features = np.sort(
            rng.choice(
                matrix.shape[1],
                size=self.config.features_per_node,
                replace=False,
            )
        )
        best: tuple[float, int, float, np.ndarray] | None = None
        for feature_index in selected_features:
            node_values = matrix[indices, feature_index]
            for threshold in _lower_rank_thresholds(
                node_values, self.config.split_quantiles
            ):
                left = node_values <= threshold
                left_count = int(left.sum())
                right_count = len(indices) - left_count
                if (
                    left_count < self.config.min_samples_leaf
                    or right_count < self.config.min_samples_leaf
                ):
                    continue
                left_weights = node_weights[left]
                right_weights = node_weights[~left]
                weighted_children = (
                    float(left_weights.sum())
                    * _weighted_gini(labels[left], left_weights)
                    + float(right_weights.sum())
                    * _weighted_gini(labels[~left], right_weights)
                ) / parent_weight
                gain = float(parent_impurity - weighted_children)
                if gain <= 0.0:
                    continue
                candidate = (gain, int(feature_index), float(threshold), left)
                if best is None:
                    best = candidate
                    continue
                best_gain, best_feature, best_threshold, _ = best
                if gain > best_gain + _TIE_TOLERANCE or (
                    abs(gain - best_gain) <= _TIE_TOLERANCE
                    and (
                        int(feature_index) < best_feature
                        or (
                            int(feature_index) == best_feature
                            and float(threshold) < best_threshold
                        )
                    )
                ):
                    best = candidate

        if best is None:
            return self._leaf(severe, cash_win, clipped_edge, indices)
        gain, feature_index, threshold, left = best
        return {
            "kind": "split",
            "sample_count": int(len(indices)),
            "severe_positive_count": int(labels.sum()),
            "cash_win_positive_count": int(cash_win[indices].sum()),
            "clipped_edge_sum": float(clipped_edge[indices].sum()),
            "feature_index": feature_index,
            "threshold": threshold,
            "weighted_gini_gain": gain,
            "left": self._build_node(
                matrix,
                severe,
                cash_win,
                clipped_edge,
                weights,
                indices[left],
                depth=depth + 1,
                rng=rng,
            ),
            "right": self._build_node(
                matrix,
                severe,
                cash_win,
                clipped_edge,
                weights,
                indices[~left],
                depth=depth + 1,
                rng=rng,
            ),
        }

    @staticmethod
    def _predict_tree(tree: Mapping[str, Any], row: np.ndarray) -> tuple[float, float, float]:
        current = tree
        while current["kind"] == "split":
            current = (
                current["left"]
                if row[int(current["feature_index"])] <= float(current["threshold"])
                else current["right"]
            )
        return (
            float(current["severe_loss_probability"]),
            float(current["cash_win_probability_10bps"]),
            float(current["expected_edge_10bps"]),
        )

    def predict_components(self, features: Any) -> dict[str, np.ndarray]:
        self._require_fitted()
        matrix = self._coerce_prediction_matrix(features)
        available = np.isfinite(matrix).all(axis=1)
        severe = np.full(len(matrix), np.nan, dtype=float)
        cash = np.full(len(matrix), np.nan, dtype=float)
        edge = np.full(len(matrix), np.nan, dtype=float)
        for row_index in np.flatnonzero(available):
            for tree in self.trees:
                tree_severe, tree_cash, tree_edge = self._predict_tree(
                    tree, matrix[row_index]
                )
                severe[row_index] = (
                    tree_severe
                    if math.isnan(severe[row_index])
                    else severe[row_index] + tree_severe
                )
                cash[row_index] = (
                    tree_cash
                    if math.isnan(cash[row_index])
                    else cash[row_index] + tree_cash
                )
                edge[row_index] = (
                    tree_edge
                    if math.isnan(edge[row_index])
                    else edge[row_index] + tree_edge
                )
            severe[row_index] /= len(self.trees)
            cash[row_index] /= len(self.trees)
            edge[row_index] /= len(self.trees)
        return {
            "severe_probability": severe,
            "ordinary_cash_win_probability": cash,
            "expected_clipped_edge": edge,
            "observation_available": available,
        }

    def oob_components(self) -> dict[str, np.ndarray]:
        self._require_fitted()
        return {
            "severe_probability": self.oob_severe_loss_probability.copy(),
            "ordinary_cash_win_probability": self.oob_cash_win_probability_10bps.copy(),
            "expected_clipped_edge": self.oob_expected_edge_10bps.copy(),
            "oob_tree_count": self.oob_tree_counts.copy(),
        }

    def oob_severe_threshold(
        self,
        quantile: float,
        *,
        minimum_tree_count: int = MINIMUM_OOB_TREE_COUNT,
    ) -> float:
        self._require_fitted()
        if isinstance(quantile, bool) or float(quantile) not in OOB_GATE_QUANTILES:
            raise ValueError(f"quantile must be one of {OOB_GATE_QUANTILES}")
        if minimum_tree_count != MINIMUM_OOB_TREE_COUNT:
            raise ValueError(
                f"minimum_tree_count is frozen at {MINIMUM_OOB_TREE_COUNT}"
            )
        eligible = (
            self.oob_tree_counts >= minimum_tree_count
        ) & np.isfinite(self.oob_severe_loss_probability)
        return _higher_quantile(
            self.oob_severe_loss_probability[eligible], float(quantile)
        )

    def _node_state(self, node: Mapping[str, Any]) -> dict[str, Any]:
        common = {
            "kind": node["kind"],
            "sample_count": int(node["sample_count"]),
            "severe_positive_count": int(node["severe_positive_count"]),
            "cash_win_positive_count": int(node["cash_win_positive_count"]),
            "clipped_edge_sum": _float_hex(float(node["clipped_edge_sum"])),
        }
        if node["kind"] == "leaf":
            return {
                **common,
                "severe_loss_probability": _float_hex(
                    float(node["severe_loss_probability"])
                ),
                "cash_win_probability_10bps": _float_hex(
                    float(node["cash_win_probability_10bps"])
                ),
                "expected_edge_10bps": _float_hex(
                    float(node["expected_edge_10bps"])
                ),
            }
        return {
            **common,
            "feature_index": int(node["feature_index"]),
            "threshold": _float_hex(float(node["threshold"])),
            "weighted_gini_gain": _float_hex(float(node["weighted_gini_gain"])),
            "left": self._node_state(node["left"]),
            "right": self._node_state(node["right"]),
        }

    @classmethod
    def _node_from_state(cls, state: Any) -> dict[str, Any]:
        if not isinstance(state, Mapping):
            raise ValueError("Serialized tree node must be an object")
        common = {
            "kind",
            "sample_count",
            "severe_positive_count",
            "cash_win_positive_count",
            "clipped_edge_sum",
        }
        kind = state.get("kind")
        if kind == "leaf":
            _expect_keys(
                state,
                common
                | {
                    "severe_loss_probability",
                    "cash_win_probability_10bps",
                    "expected_edge_10bps",
                },
                "leaf",
            )
            node = {
                "kind": "leaf",
                "sample_count": _strict_int(
                    state["sample_count"], "leaf.sample_count", minimum=1
                ),
                "severe_positive_count": _strict_int(
                    state["severe_positive_count"],
                    "leaf.severe_positive_count",
                ),
                "cash_win_positive_count": _strict_int(
                    state["cash_win_positive_count"],
                    "leaf.cash_win_positive_count",
                ),
                "clipped_edge_sum": _decode_float_hex(
                    state["clipped_edge_sum"], "leaf.clipped_edge_sum"
                ),
                "severe_loss_probability": _decode_float_hex(
                    state["severe_loss_probability"],
                    "leaf.severe_loss_probability",
                ),
                "cash_win_probability_10bps": _decode_float_hex(
                    state["cash_win_probability_10bps"],
                    "leaf.cash_win_probability_10bps",
                ),
                "expected_edge_10bps": _decode_float_hex(
                    state["expected_edge_10bps"], "leaf.expected_edge_10bps"
                ),
            }
            return node
        if kind == "split":
            _expect_keys(
                state,
                common
                | {
                    "feature_index",
                    "threshold",
                    "weighted_gini_gain",
                    "left",
                    "right",
                },
                "split",
            )
            return {
                "kind": "split",
                "sample_count": _strict_int(
                    state["sample_count"], "split.sample_count", minimum=1
                ),
                "severe_positive_count": _strict_int(
                    state["severe_positive_count"],
                    "split.severe_positive_count",
                ),
                "cash_win_positive_count": _strict_int(
                    state["cash_win_positive_count"],
                    "split.cash_win_positive_count",
                ),
                "clipped_edge_sum": _decode_float_hex(
                    state["clipped_edge_sum"], "split.clipped_edge_sum"
                ),
                "feature_index": _strict_int(
                    state["feature_index"], "split.feature_index"
                ),
                "threshold": _decode_float_hex(
                    state["threshold"], "split.threshold"
                ),
                "weighted_gini_gain": _decode_float_hex(
                    state["weighted_gini_gain"], "split.weighted_gini_gain"
                ),
                "left": cls._node_from_state(state["left"]),
                "right": cls._node_from_state(state["right"]),
            }
        raise ValueError("Serialized tree node kind must be leaf or split")

    def _payload(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "config": asdict(self.config),
            "namespace": self.namespace,
            "feature_names": list(self.feature_names),
            "fit_metadata": self.fit_metadata,
            "training": {
                "row_count": self.training_row_count,
                "severe_positive_count": self.severe_positive_count,
                "cash_win_positive_count": self.cash_win_positive_count,
                "global_severe_probability": _float_hex(
                    self.global_severe_probability
                ),
                "global_cash_win_probability": _float_hex(
                    self.global_cash_win_probability
                ),
                "global_expected_edge": _float_hex(self.global_expected_edge),
            },
            "trees": [self._node_state(tree) for tree in self.trees],
            "oob": {
                "tree_counts": self.oob_tree_counts.tolist(),
                "severe_loss_probability": _encode_optional_vector(
                    self.oob_severe_loss_probability
                ),
                "cash_win_probability_10bps": _encode_optional_vector(
                    self.oob_cash_win_probability_10bps
                ),
                "expected_edge_10bps": _encode_optional_vector(
                    self.oob_expected_edge_10bps
                ),
            },
        }

    def to_state(self) -> dict[str, Any]:
        payload = self._payload()
        return json.loads(
            _canonical_json(
                {"payload": payload, "payload_sha256": _sha256_json(payload)}
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
    def load(cls, path: str | os.PathLike[str]) -> "RareLossForest":
        try:
            state = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Could not read a valid rare-loss forest state") from exc
        return cls.from_state(state)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "RareLossForest":
        if not isinstance(state, Mapping):
            raise ValueError("Rare-loss forest state must be an object")
        _expect_keys(state, {"payload", "payload_sha256"}, "state")
        payload = state["payload"]
        checksum = state["payload_sha256"]
        if not isinstance(payload, Mapping) or not _is_sha256(checksum):
            raise ValueError("Rare-loss forest payload or checksum is invalid")
        if not hmac.compare_digest(_sha256_json(payload), checksum):
            raise ValueError("Rare-loss forest state checksum mismatch")
        _expect_keys(
            payload,
            {
                "state_schema_version",
                "model_type",
                "config",
                "namespace",
                "feature_names",
                "fit_metadata",
                "training",
                "trees",
                "oob",
            },
            "payload",
        )
        if (
            isinstance(payload["state_schema_version"], bool)
            or payload["state_schema_version"] != STATE_SCHEMA_VERSION
            or payload["model_type"] != MODEL_TYPE
        ):
            raise ValueError("Unsupported rare-loss forest state schema or model type")
        raw_config = payload["config"]
        if not isinstance(raw_config, Mapping):
            raise ValueError("Serialized forest config must be an object")
        _expect_keys(raw_config, set(asdict(RareLossForestConfig())), "config")
        config_values = dict(raw_config)
        config_values["split_quantiles"] = tuple(config_values["split_quantiles"])
        config = RareLossForestConfig(**config_values)
        config.validate()
        namespace = payload["namespace"]
        if not isinstance(namespace, str) or not namespace or namespace.strip() != namespace:
            raise ValueError("Serialized namespace is invalid")
        raw_names = payload["feature_names"]
        if (
            not isinstance(raw_names, list)
            or len(raw_names) < config.features_per_node
            or any(not isinstance(value, str) or not value for value in raw_names)
            or len(set(raw_names)) != len(raw_names)
        ):
            raise ValueError("Serialized feature schema is invalid")
        metadata = _json_safe_mapping(payload["fit_metadata"])
        training = payload["training"]
        if not isinstance(training, Mapping):
            raise ValueError("Serialized training metadata must be an object")
        _expect_keys(
            training,
            {
                "row_count",
                "severe_positive_count",
                "cash_win_positive_count",
                "global_severe_probability",
                "global_cash_win_probability",
                "global_expected_edge",
            },
            "training",
        )
        row_count = _strict_int(training["row_count"], "row_count", minimum=1)
        severe_count = _strict_int(
            training["severe_positive_count"], "severe_positive_count", minimum=1
        )
        cash_count = _strict_int(
            training["cash_win_positive_count"], "cash_win_positive_count", minimum=1
        )
        trees = payload["trees"]
        if not isinstance(trees, list) or len(trees) != config.tree_count:
            raise ValueError("Serialized forest must contain exactly 127 trees")
        decoded_trees = [cls._node_from_state(tree) for tree in trees]
        oob = payload["oob"]
        if not isinstance(oob, Mapping):
            raise ValueError("Serialized OOB state must be an object")
        _expect_keys(
            oob,
            {
                "tree_counts",
                "severe_loss_probability",
                "cash_win_probability_10bps",
                "expected_edge_10bps",
            },
            "oob",
        )
        raw_counts = oob["tree_counts"]
        if not isinstance(raw_counts, list) or len(raw_counts) != row_count:
            raise ValueError("Serialized OOB tree counts have invalid length")
        counts = np.asarray(
            [
                _strict_int(value, f"oob.tree_counts[{index}]")
                for index, value in enumerate(raw_counts)
            ],
            dtype=np.int64,
        )
        model = cls(config)
        model.namespace = namespace
        model.feature_names = tuple(raw_names)
        model.fit_metadata = metadata
        model.training_row_count = row_count
        model.severe_positive_count = severe_count
        model.cash_win_positive_count = cash_count
        model.global_severe_probability = _decode_float_hex(
            training["global_severe_probability"], "global_severe_probability"
        )
        model.global_cash_win_probability = _decode_float_hex(
            training["global_cash_win_probability"],
            "global_cash_win_probability",
        )
        model.global_expected_edge = _decode_float_hex(
            training["global_expected_edge"], "global_expected_edge"
        )
        model.trees = decoded_trees
        model.oob_tree_counts = counts
        model.oob_severe_loss_probability = _decode_optional_vector(
            oob["severe_loss_probability"], "oob.severe_loss_probability", row_count
        )
        model.oob_cash_win_probability_10bps = _decode_optional_vector(
            oob["cash_win_probability_10bps"],
            "oob.cash_win_probability_10bps",
            row_count,
        )
        model.oob_expected_edge_10bps = _decode_optional_vector(
            oob["expected_edge_10bps"], "oob.expected_edge_10bps", row_count
        )
        model._validate_fitted_semantics()
        if model.model_sha256 != checksum:
            raise ValueError("Decoded rare-loss forest hash changed unexpectedly")
        return model

    def _validate_node(self, node: Mapping[str, Any], *, depth: int) -> None:
        count = int(node["sample_count"])
        severe_count = int(node["severe_positive_count"])
        cash_count = int(node["cash_win_positive_count"])
        edge_sum = float(node["clipped_edge_sum"])
        if count < 1 or not 0 <= severe_count <= count or not 0 <= cash_count <= count:
            raise ValueError("Serialized tree node counts are inconsistent")
        if depth > 0 and count < self.config.min_samples_leaf:
            raise ValueError("Serialized tree node violates frozen minimum leaf size")
        if not math.isfinite(edge_sum) or not (
            count * self.config.clipped_edge_lower - 1e-12
            <= edge_sum
            <= count * self.config.clipped_edge_upper + 1e-12
        ):
            raise ValueError("Serialized tree node edge sum is invalid")
        if node["kind"] == "leaf":
            denominator = count + self.config.shrinkage_strength
            expected = (
                (severe_count + self.config.shrinkage_strength * self.global_severe_probability)
                / denominator,
                (cash_count + self.config.shrinkage_strength * self.global_cash_win_probability)
                / denominator,
                (edge_sum + self.config.shrinkage_strength * self.global_expected_edge)
                / denominator,
            )
            actual = (
                float(node["severe_loss_probability"]),
                float(node["cash_win_probability_10bps"]),
                float(node["expected_edge_10bps"]),
            )
            if any(
                not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)
                for left, right in zip(expected, actual, strict=True)
            ):
                raise ValueError("Serialized leaf estimates violate frozen shrinkage")
            if not 0.0 <= actual[0] <= 1.0 or not 0.0 <= actual[1] <= 1.0:
                raise ValueError("Serialized leaf probability is outside [0, 1]")
            return
        if depth >= self.config.max_depth:
            raise ValueError("Serialized split exceeds frozen maximum depth")
        feature_index = int(node["feature_index"])
        if not 0 <= feature_index < len(self.feature_names):
            raise ValueError("Serialized split feature index is outside schema")
        if float(node["weighted_gini_gain"]) <= 0.0:
            raise ValueError("Serialized split gain must be positive")
        self._validate_node(node["left"], depth=depth + 1)
        self._validate_node(node["right"], depth=depth + 1)
        left = node["left"]
        right = node["right"]
        for key in (
            "sample_count",
            "severe_positive_count",
            "cash_win_positive_count",
        ):
            if int(node[key]) != int(left[key]) + int(right[key]):
                raise ValueError("Serialized split child counts do not reconcile")
        if not math.isclose(
            float(node["clipped_edge_sum"]),
            float(left["clipped_edge_sum"]) + float(right["clipped_edge_sum"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Serialized split child edge sums do not reconcile")

    def _validate_fitted_semantics(self) -> None:
        self.config.validate()
        if not self.is_fitted:
            raise ValueError("Rare-loss forest state is incomplete")
        if (
            self.training_row_count < 2 * self.config.min_samples_leaf
            or not 0 < self.severe_positive_count < self.training_row_count
            or not 0 < self.cash_win_positive_count < self.training_row_count
        ):
            raise ValueError("Rare-loss training counts are inconsistent")
        expected_severe = self.severe_positive_count / self.training_row_count
        expected_cash = self.cash_win_positive_count / self.training_row_count
        if not math.isclose(
            self.global_severe_probability, expected_severe, rel_tol=0.0, abs_tol=1e-15
        ) or not math.isclose(
            self.global_cash_win_probability, expected_cash, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("Serialized global probabilities violate raw training priors")
        if not (
            self.config.clipped_edge_lower
            <= self.global_expected_edge
            <= self.config.clipped_edge_upper
        ):
            raise ValueError("Serialized global expected edge is outside clip bounds")
        expected_root_count = int(
            math.ceil(self.config.bootstrap_fraction * self.training_row_count)
        )
        for tree in self.trees:
            if int(tree["sample_count"]) != expected_root_count:
                raise ValueError("Serialized tree root violates frozen bootstrap size")
            self._validate_node(tree, depth=0)
        if (self.oob_tree_counts < 0).any() or (
            self.oob_tree_counts > self.config.tree_count
        ).any():
            raise ValueError("Serialized OOB counts are outside valid bounds")
        available = self.oob_tree_counts > 0
        for name, values in (
            ("severe", self.oob_severe_loss_probability),
            ("cash", self.oob_cash_win_probability_10bps),
            ("edge", self.oob_expected_edge_10bps),
        ):
            if values.shape != (self.training_row_count,):
                raise ValueError(f"Serialized OOB {name} vector has invalid shape")
            if not np.isfinite(values[available]).all() or not np.isnan(
                values[~available]
            ).all():
                raise ValueError(f"Serialized OOB {name} availability is inconsistent")
        if (
            ((self.oob_severe_loss_probability[available] < 0.0)
             | (self.oob_severe_loss_probability[available] > 1.0)).any()
            or ((self.oob_cash_win_probability_10bps[available] < 0.0)
                | (self.oob_cash_win_probability_10bps[available] > 1.0)).any()
        ):
            raise ValueError("Serialized OOB probabilities are outside [0, 1]")


__all__ = [
    "MINIMUM_OOB_TREE_COUNT",
    "MODEL_TYPE",
    "OOB_GATE_QUANTILES",
    "SEVERE_LOG_RETURN_CUTOFF",
    "SEVERE_SIMPLE_RETURN_CUTOFF",
    "STATE_SCHEMA_VERSION",
    "RareLossForest",
    "RareLossForestConfig",
    "cash_win_labels",
    "rare_loss_labels",
]
