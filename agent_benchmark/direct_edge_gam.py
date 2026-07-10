from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_TYPE = "aapl_direct_edge_regularized_gam"
STATE_SCHEMA_VERSION = 1
ANCHOR_FEATURE_NAMES = (
    "aapl_drawdown_252",
    "aapl_rv_20",
    "qqq_lr_60",
    "vix_z_252",
)


@dataclass(frozen=True)
class DirectEdgeGAMConfig:
    """Frozen configuration for the deterministic direct-edge GAM."""

    raw_mad_multiplier: float = 1.4826
    raw_scale_floor: float = 1e-6
    raw_z_clip: float = 4.0
    hinge_knot: float = 1.0
    basis_scale_floor: float = 1e-6
    ridge_lambdas: tuple[float, ...] = (0.001, 0.01, 0.1)
    logistic_max_iterations: int = 50
    logistic_tolerance: float = 1e-10
    newton_line_search_max_steps: int = 50
    huber_delta: float = 1.5
    huber_max_iterations: int = 50
    huber_tolerance: float = 1e-10
    edge_clip_lower: float = -0.2
    edge_clip_upper: float = 0.2
    target_mad_multiplier: float = 1.4826
    target_scale_floor: float = 1e-6

    def validate(self) -> None:
        expected = DirectEdgeGAMConfig()
        if _canonical_json(asdict(self)) != _canonical_json(asdict(expected)):
            raise ValueError(
                "GAM configuration does not match the frozen direct-edge-v1 contract"
            )


GAMConfig = DirectEdgeGAMConfig


@dataclass(frozen=True)
class GAMFitMetadata:
    training_start_date: str
    training_end_date: str
    maximum_label_maturity_date: str
    training_dates_sha256: str
    binary_target_sha256: str
    edge_target_sha256: str

    def validate(self) -> None:
        parsed_dates: dict[str, date] = {}
        for field_name in (
            "training_start_date",
            "training_end_date",
            "maximum_label_maturity_date",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be an ISO date")
            try:
                parsed = date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"{field_name} must be an ISO date") from exc
            if parsed.isoformat() != value:
                raise ValueError(f"{field_name} must use canonical YYYY-MM-DD form")
            parsed_dates[field_name] = parsed
        if parsed_dates["training_start_date"] > parsed_dates["training_end_date"]:
            raise ValueError("training_start_date must not follow training_end_date")
        if (
            parsed_dates["training_end_date"]
            > parsed_dates["maximum_label_maturity_date"]
        ):
            raise ValueError(
                "maximum_label_maturity_date must not precede training_end_date"
            )
        for field_name in (
            "training_dates_sha256",
            "binary_target_sha256",
            "edge_target_sha256",
        ):
            if not _is_sha256(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")

    @classmethod
    def coerce(cls, value: Mapping[str, Any] | "GAMFitMetadata") -> "GAMFitMetadata":
        if isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            expected = set(cls.__dataclass_fields__)
            _expect_keys(value, expected, "fit metadata")
            try:
                result = cls(**dict(value))
            except TypeError as exc:
                raise ValueError("Could not decode GAM fit metadata") from exc
        else:
            raise ValueError("fit_metadata must be a GAMFitMetadata or mapping")
        result.validate()
        return result


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
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Invalid {location} keys; missing={missing}, extra={extra}")


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
        number = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{location} must be a canonical hexadecimal float") from exc
    if not math.isfinite(number) or number.hex() != value:
        raise ValueError(f"{location} must be a canonical finite hexadecimal float")
    return number


def _encode_float_vector(values: np.ndarray) -> list[str]:
    return [_float_hex(value) for value in np.asarray(values, dtype=float)]


def _decode_float_vector(
    values: Any,
    location: str,
    *,
    expected_length: int,
) -> np.ndarray:
    if not isinstance(values, list) or len(values) != expected_length:
        raise ValueError(f"{location} must contain exactly {expected_length} values")
    return np.asarray(
        [_decode_float_hex(value, f"{location}[{index}]") for index, value in enumerate(values)],
        dtype=float,
    )


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _feature_basis_spec(feature_names: Sequence[str]) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    names: list[str] = []
    for feature_name in feature_names:
        names.extend(
            (
                f"linear:{feature_name}",
                f"hinge_positive:{feature_name}",
                f"hinge_negative:{feature_name}",
            )
        )
    anchors = set(ANCHOR_FEATURE_NAMES).intersection(feature_names)
    interactions: list[tuple[int, int]] = []
    for left in range(len(feature_names)):
        for right in range(left + 1, len(feature_names)):
            if feature_names[left] in anchors or feature_names[right] in anchors:
                interactions.append((left, right))
                names.append(
                    f"interaction:{feature_names[left]}*{feature_names[right]}"
                )
    return tuple(names), tuple(interactions)


def _raw_basis(
    raw_z: np.ndarray,
    interactions: Sequence[tuple[int, int]],
    *,
    hinge_knot: float,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for feature_index in range(raw_z.shape[1]):
        values = raw_z[:, feature_index]
        columns.extend(
            (
                values,
                np.maximum(values - hinge_knot, 0.0),
                np.maximum(-values - hinge_knot, 0.0),
            )
        )
    for left, right in interactions:
        columns.append(raw_z[:, left] * raw_z[:, right])
    return np.column_stack(columns)


def _design_with_intercept(basis: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(basis), dtype=float), basis))


def _penalty_diagonal(dimension: int, ridge_lambda: float) -> np.ndarray:
    penalty = np.full(dimension, ridge_lambda, dtype=float)
    penalty[0] = 0.0
    return penalty


def _logistic_objective(
    design: np.ndarray,
    target: np.ndarray,
    coefficients: np.ndarray,
    ridge_lambda: float,
) -> float:
    linear = design @ coefficients
    data_loss = float(np.mean(np.logaddexp(0.0, linear) - target * linear))
    penalty = 0.5 * ridge_lambda * float(coefficients[1:] @ coefficients[1:])
    return data_loss + penalty


def _fit_logistic_head(
    design: np.ndarray,
    target: np.ndarray,
    ridge_lambda: float,
    config: DirectEdgeGAMConfig,
) -> tuple[np.ndarray, int, bool]:
    prevalence = float(target.mean())
    coefficients = np.zeros(design.shape[1], dtype=float)
    coefficients[0] = math.log(prevalence / (1.0 - prevalence))
    penalty = _penalty_diagonal(design.shape[1], ridge_lambda)
    identity_penalty = np.diag(penalty)
    converged = False
    completed_iterations = 0

    for iteration in range(1, config.logistic_max_iterations + 1):
        linear = design @ coefficients
        probability = _stable_sigmoid(linear)
        gradient = design.T @ (probability - target) / len(target)
        gradient += penalty * coefficients
        if float(np.max(np.abs(gradient))) <= config.logistic_tolerance:
            converged = True
            completed_iterations = iteration - 1
            break
        curvature = np.maximum(probability * (1.0 - probability), 1e-15)
        hessian = (design.T @ (design * curvature[:, None])) / len(target)
        hessian += identity_penalty
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        current_objective = _logistic_objective(
            design, target, coefficients, ridge_lambda
        )
        directional_derivative = float(gradient @ direction)
        step = 1.0
        accepted = False
        candidate = coefficients
        for _ in range(config.newton_line_search_max_steps):
            candidate = coefficients - step * direction
            candidate_objective = _logistic_objective(
                design, target, candidate, ridge_lambda
            )
            if candidate_objective <= current_objective - 1e-4 * step * directional_derivative:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            completed_iterations = iteration
            if float(np.max(np.abs(gradient))) <= 10.0 * config.logistic_tolerance:
                converged = True
            break
        change = float(np.max(np.abs(candidate - coefficients)))
        coefficients = candidate
        completed_iterations = iteration
        if change <= config.logistic_tolerance:
            converged = True
            break
    if not np.isfinite(coefficients).all():
        raise RuntimeError("Logistic GAM solver produced non-finite coefficients")
    return coefficients, completed_iterations, converged


def _fit_huber_head(
    design: np.ndarray,
    standardized_target: np.ndarray,
    ridge_lambda: float,
    config: DirectEdgeGAMConfig,
) -> tuple[np.ndarray, int, bool]:
    coefficients = np.zeros(design.shape[1], dtype=float)
    coefficients[0] = float(np.median(standardized_target))
    penalty = np.diag(_penalty_diagonal(design.shape[1], ridge_lambda))
    converged = False
    completed_iterations = 0

    for iteration in range(1, config.huber_max_iterations + 1):
        residual = standardized_target - design @ coefficients
        absolute = np.abs(residual)
        weights = np.ones(len(residual), dtype=float)
        outside = absolute > config.huber_delta
        weights[outside] = config.huber_delta / absolute[outside]
        system = (design.T @ (design * weights[:, None])) / len(design)
        system += penalty
        right_hand_side = design.T @ (weights * standardized_target) / len(design)
        try:
            candidate = np.linalg.solve(system, right_hand_side)
        except np.linalg.LinAlgError:
            candidate = np.linalg.lstsq(system, right_hand_side, rcond=None)[0]
        if not np.isfinite(candidate).all():
            raise RuntimeError("Huber GAM solver produced non-finite coefficients")
        change = float(np.max(np.abs(candidate - coefficients)))
        coefficients = candidate
        completed_iterations = iteration
        if change <= config.huber_tolerance:
            converged = True
            break
    return coefficients, completed_iterations, converged


class DirectEdgeGAM:
    """Deterministic two-head additive model for five-session cash edge."""

    def __init__(self, config: DirectEdgeGAMConfig | None = None) -> None:
        self.config = config or DirectEdgeGAMConfig()
        self.config.validate()
        self.feature_names: tuple[str, ...] = ()
        self.basis_names: tuple[str, ...] = ()
        self.interactions: tuple[tuple[int, int], ...] = ()
        self.raw_medians = np.empty(0, dtype=float)
        self.raw_scales = np.empty(0, dtype=float)
        self.basis_means = np.empty(0, dtype=float)
        self.basis_scales = np.empty(0, dtype=float)
        self.target_center = 0.0
        self.target_scale = 1.0
        self.fit_metadata: GAMFitMetadata | None = None
        self.training_sample_count = 0
        self.training_positive_count = 0
        self.logistic_heads: list[dict[str, Any]] = []
        self.huber_heads: list[dict[str, Any]] = []

    @property
    def is_fitted(self) -> bool:
        return (
            self.fit_metadata is not None
            and len(self.logistic_heads) == len(self.config.ridge_lambdas)
            and len(self.huber_heads) == len(self.config.ridge_lambdas)
        )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Direct-edge GAM is not fitted")

    @property
    def model_sha256(self) -> str:
        self._require_fitted()
        return _sha256_json(self._payload())

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
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
            raise ValueError("Training features must be a non-empty two-dimensional matrix")
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
        cash_beats_long_10bps: Sequence[float],
        edge_10bps: Sequence[float],
        *,
        feature_names: Sequence[str] | None = None,
        fit_metadata: Mapping[str, Any] | GAMFitMetadata,
    ) -> "DirectEdgeGAM":
        self.config.validate()
        matrix, names = self._coerce_training_matrix(features, feature_names)
        binary = np.asarray(cash_beats_long_10bps, dtype=float)
        edge = np.asarray(edge_10bps, dtype=float)
        if binary.ndim != 1 or len(binary) != len(matrix):
            raise ValueError("Binary target must have one value per training row")
        if edge.ndim != 1 or len(edge) != len(matrix):
            raise ValueError("Edge target must have one value per training row")
        if not np.isfinite(binary).all() or not np.isfinite(edge).all():
            raise ValueError("Training targets must all be finite and causally mature")
        if not np.isin(binary, (0.0, 1.0)).all():
            raise ValueError("cash_beats_long_10bps must be binary zero/one")
        positive_count = int(binary.sum())
        if positive_count == 0 or positive_count == len(binary):
            raise ValueError("Logistic training requires both binary target classes")
        metadata = GAMFitMetadata.coerce(fit_metadata)

        raw_medians = np.median(matrix, axis=0)
        raw_mads = np.median(np.abs(matrix - raw_medians), axis=0)
        raw_scales = np.maximum(
            self.config.raw_mad_multiplier * raw_mads,
            self.config.raw_scale_floor,
        )
        raw_z = np.clip(
            (matrix - raw_medians) / raw_scales,
            -self.config.raw_z_clip,
            self.config.raw_z_clip,
        )
        basis_names, interactions = _feature_basis_spec(names)
        unscaled_basis = _raw_basis(
            raw_z,
            interactions,
            hinge_knot=self.config.hinge_knot,
        )
        basis_means = np.mean(unscaled_basis, axis=0)
        basis_scales = np.maximum(
            np.std(unscaled_basis, axis=0, ddof=0),
            self.config.basis_scale_floor,
        )
        basis = (unscaled_basis - basis_means) / basis_scales
        design = _design_with_intercept(basis)

        clipped_edge = np.clip(
            edge, self.config.edge_clip_lower, self.config.edge_clip_upper
        )
        target_center = float(np.median(clipped_edge))
        target_mad = float(np.median(np.abs(clipped_edge - target_center)))
        target_scale = max(
            self.config.target_mad_multiplier * target_mad,
            self.config.target_scale_floor,
        )
        standardized_edge = (clipped_edge - target_center) / target_scale

        logistic_heads: list[dict[str, Any]] = []
        huber_heads: list[dict[str, Any]] = []
        for ridge_lambda in self.config.ridge_lambdas:
            coefficients, iterations, converged = _fit_logistic_head(
                design, binary, ridge_lambda, self.config
            )
            if not converged:
                raise RuntimeError(
                    "Logistic GAM solver did not converge for "
                    f"ridge_lambda={ridge_lambda}"
                )
            logistic_heads.append(
                {
                    "ridge_lambda": float(ridge_lambda),
                    "coefficients": coefficients,
                    "iterations": iterations,
                    "converged": converged,
                }
            )
            coefficients, iterations, converged = _fit_huber_head(
                design, standardized_edge, ridge_lambda, self.config
            )
            if not converged:
                raise RuntimeError(
                    "Huber GAM solver did not converge for "
                    f"ridge_lambda={ridge_lambda}"
                )
            huber_heads.append(
                {
                    "ridge_lambda": float(ridge_lambda),
                    "coefficients": coefficients,
                    "iterations": iterations,
                    "converged": converged,
                }
            )

        self.feature_names = names
        self.basis_names = basis_names
        self.interactions = interactions
        self.raw_medians = raw_medians.astype(float, copy=True)
        self.raw_scales = raw_scales.astype(float, copy=True)
        self.basis_means = basis_means.astype(float, copy=True)
        self.basis_scales = basis_scales.astype(float, copy=True)
        self.target_center = target_center
        self.target_scale = target_scale
        self.fit_metadata = metadata
        self.training_sample_count = int(len(matrix))
        self.training_positive_count = positive_count
        self.logistic_heads = logistic_heads
        self.huber_heads = huber_heads
        self._validate_fitted_semantics()
        return self

    def _transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        raw_z = np.clip(
            (matrix - self.raw_medians) / self.raw_scales,
            -self.config.raw_z_clip,
            self.config.raw_z_clip,
        )
        unscaled = _raw_basis(
            raw_z,
            self.interactions,
            hinge_knot=self.config.hinge_knot,
        )
        return (unscaled - self.basis_means) / self.basis_scales

    def transform_basis(self, features: Any) -> np.ndarray:
        self._require_fitted()
        matrix = self._coerce_prediction_matrix(features)
        return self._transform_matrix(matrix)

    def predict_components(self, features: Any) -> dict[str, np.ndarray]:
        self._require_fitted()
        matrix = self._coerce_prediction_matrix(features)
        design = _design_with_intercept(self._transform_matrix(matrix))
        probabilities = np.zeros(len(matrix), dtype=float)
        expected_edges = np.zeros(len(matrix), dtype=float)
        for head in self.logistic_heads:
            probabilities += _stable_sigmoid(design @ head["coefficients"])
        for head in self.huber_heads:
            standardized = design @ head["coefficients"]
            expected_edges += self.target_center + self.target_scale * standardized
        probabilities /= len(self.logistic_heads)
        expected_edges /= len(self.huber_heads)
        expected_edges = np.clip(
            expected_edges,
            self.config.edge_clip_lower,
            self.config.edge_clip_upper,
        )
        return {
            "cash_beats_long_probability": probabilities,
            "expected_edge_10bps": expected_edges,
        }

    def predict_proba(self, features: Any) -> np.ndarray:
        return self.predict_components(features)["cash_beats_long_probability"]

    def predict_cash_probability(self, features: Any) -> np.ndarray:
        return self.predict_proba(features)

    def predict_expected_edge(self, features: Any) -> np.ndarray:
        return self.predict_components(features)["expected_edge_10bps"]

    @staticmethod
    def _encoded_head(head: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "ridge_lambda": _float_hex(head["ridge_lambda"]),
            "coefficients": _encode_float_vector(head["coefficients"]),
            "iterations": int(head["iterations"]),
            "converged": bool(head["converged"]),
        }

    def _parameter_payload(self) -> dict[str, Any]:
        return {
            "preprocessing": {
                "raw_medians": _encode_float_vector(self.raw_medians),
                "raw_scales": _encode_float_vector(self.raw_scales),
                "basis_means": _encode_float_vector(self.basis_means),
                "basis_scales": _encode_float_vector(self.basis_scales),
                "target_center": _float_hex(self.target_center),
                "target_scale": _float_hex(self.target_scale),
            },
            "heads": {
                "logistic": [self._encoded_head(head) for head in self.logistic_heads],
                "huber": [self._encoded_head(head) for head in self.huber_heads],
            },
        }

    def _payload(self) -> dict[str, Any]:
        self._require_fitted()
        assert self.fit_metadata is not None
        parameters = self._parameter_payload()
        return {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "config": asdict(self.config),
            "feature_names": list(self.feature_names),
            "basis_names": list(self.basis_names),
            "fit_metadata": asdict(self.fit_metadata),
            "training": {
                "sample_count": self.training_sample_count,
                "positive_count": self.training_positive_count,
            },
            "parameters": parameters,
            "parameters_sha256": _sha256_json(parameters),
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
    def load(cls, path: str | os.PathLike[str]) -> "DirectEdgeGAM":
        try:
            state = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Could not read a valid direct-edge GAM state") from exc
        return cls.from_state(state)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DirectEdgeGAM":
        if not isinstance(state, Mapping):
            raise ValueError("Direct-edge GAM state must be an object")
        _expect_keys(state, {"payload", "payload_sha256"}, "state")
        payload = state["payload"]
        if not isinstance(payload, Mapping):
            raise ValueError("Direct-edge GAM payload must be an object")
        payload_hash = state["payload_sha256"]
        if not _is_sha256(payload_hash):
            raise ValueError("Direct-edge GAM payload hash is invalid")
        if not hmac.compare_digest(_sha256_json(payload), payload_hash):
            raise ValueError("Direct-edge GAM state checksum mismatch")
        _expect_keys(
            payload,
            {
                "state_schema_version",
                "model_type",
                "config",
                "feature_names",
                "basis_names",
                "fit_metadata",
                "training",
                "parameters",
                "parameters_sha256",
            },
            "payload",
        )
        if (
            isinstance(payload["state_schema_version"], bool)
            or not isinstance(payload["state_schema_version"], int)
            or payload["state_schema_version"] != STATE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported direct-edge GAM state schema")
        if payload["model_type"] != MODEL_TYPE:
            raise ValueError("Unexpected direct-edge GAM model type")

        raw_config = payload["config"]
        if not isinstance(raw_config, Mapping):
            raise ValueError("Direct-edge GAM config must be an object")
        _expect_keys(raw_config, set(asdict(DirectEdgeGAMConfig())), "config")
        try:
            config = DirectEdgeGAMConfig(
                **{**dict(raw_config), "ridge_lambdas": tuple(raw_config["ridge_lambdas"])}
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Could not decode direct-edge GAM config") from exc
        config.validate()

        raw_names = payload["feature_names"]
        if (
            not isinstance(raw_names, list)
            or not raw_names
            or any(not isinstance(name, str) or not name for name in raw_names)
            or len(set(raw_names)) != len(raw_names)
        ):
            raise ValueError("Direct-edge GAM feature schema is invalid")
        names = tuple(raw_names)
        expected_basis_names, interactions = _feature_basis_spec(names)
        if payload["basis_names"] != list(expected_basis_names):
            raise ValueError("Direct-edge GAM basis schema is inconsistent")
        metadata = GAMFitMetadata.coerce(payload["fit_metadata"])

        training = payload["training"]
        if not isinstance(training, Mapping):
            raise ValueError("Direct-edge GAM training metadata must be an object")
        _expect_keys(training, {"sample_count", "positive_count"}, "training")
        sample_count = _strict_int(training["sample_count"], "sample_count", minimum=2)
        positive_count = _strict_int(
            training["positive_count"], "positive_count", minimum=1
        )
        if positive_count >= sample_count:
            raise ValueError("Training metadata must contain both binary classes")

        parameters = payload["parameters"]
        if not isinstance(parameters, Mapping):
            raise ValueError("Direct-edge GAM parameters must be an object")
        parameter_hash = payload["parameters_sha256"]
        if not _is_sha256(parameter_hash) or not hmac.compare_digest(
            _sha256_json(parameters), parameter_hash
        ):
            raise ValueError("Direct-edge GAM parameter checksum mismatch")
        _expect_keys(parameters, {"preprocessing", "heads"}, "parameters")
        preprocessing = parameters["preprocessing"]
        if not isinstance(preprocessing, Mapping):
            raise ValueError("Direct-edge GAM preprocessing must be an object")
        _expect_keys(
            preprocessing,
            {
                "raw_medians",
                "raw_scales",
                "basis_means",
                "basis_scales",
                "target_center",
                "target_scale",
            },
            "preprocessing",
        )
        raw_medians = _decode_float_vector(
            preprocessing["raw_medians"], "raw_medians", expected_length=len(names)
        )
        raw_scales = _decode_float_vector(
            preprocessing["raw_scales"], "raw_scales", expected_length=len(names)
        )
        basis_means = _decode_float_vector(
            preprocessing["basis_means"],
            "basis_means",
            expected_length=len(expected_basis_names),
        )
        basis_scales = _decode_float_vector(
            preprocessing["basis_scales"],
            "basis_scales",
            expected_length=len(expected_basis_names),
        )
        target_center = _decode_float_hex(
            preprocessing["target_center"], "target_center"
        )
        target_scale = _decode_float_hex(preprocessing["target_scale"], "target_scale")
        if (raw_scales < config.raw_scale_floor).any():
            raise ValueError("Raw scaler violates the frozen positive floor")
        if (basis_scales < config.basis_scale_floor).any():
            raise ValueError("Basis scaler violates the frozen positive floor")
        if target_scale < config.target_scale_floor:
            raise ValueError("Target scaler violates the frozen positive floor")
        if not config.edge_clip_lower <= target_center <= config.edge_clip_upper:
            raise ValueError("Target center is outside the frozen edge bounds")

        heads = parameters["heads"]
        if not isinstance(heads, Mapping):
            raise ValueError("Direct-edge GAM heads must be an object")
        _expect_keys(heads, {"logistic", "huber"}, "heads")

        def decode_heads(raw_heads: Any, location: str) -> list[dict[str, Any]]:
            if not isinstance(raw_heads, list) or len(raw_heads) != len(config.ridge_lambdas):
                raise ValueError(f"{location} must contain one head per frozen lambda")
            decoded: list[dict[str, Any]] = []
            for index, (raw_head, expected_lambda) in enumerate(
                zip(raw_heads, config.ridge_lambdas)
            ):
                if not isinstance(raw_head, Mapping):
                    raise ValueError(f"{location}[{index}] must be an object")
                _expect_keys(
                    raw_head,
                    {"ridge_lambda", "coefficients", "iterations", "converged"},
                    f"{location}[{index}]",
                )
                ridge_lambda = _decode_float_hex(
                    raw_head["ridge_lambda"], f"{location}[{index}].ridge_lambda"
                )
                if ridge_lambda.hex() != float(expected_lambda).hex():
                    raise ValueError(f"{location} lambda order violates the frozen ensemble")
                coefficients = _decode_float_vector(
                    raw_head["coefficients"],
                    f"{location}[{index}].coefficients",
                    expected_length=len(expected_basis_names) + 1,
                )
                maximum = (
                    config.logistic_max_iterations
                    if location == "logistic"
                    else config.huber_max_iterations
                )
                iterations = _strict_int(
                    raw_head["iterations"], f"{location}[{index}].iterations", minimum=0
                )
                if iterations > maximum:
                    raise ValueError(f"{location} iterations exceed the frozen maximum")
                converged = raw_head["converged"]
                if not isinstance(converged, bool):
                    raise ValueError(f"{location} converged flag must be boolean")
                if not converged:
                    raise ValueError(f"{location}[{index}] did not converge")
                decoded.append(
                    {
                        "ridge_lambda": ridge_lambda,
                        "coefficients": coefficients,
                        "iterations": iterations,
                        "converged": converged,
                    }
                )
            return decoded

        logistic_heads = decode_heads(heads["logistic"], "logistic")
        huber_heads = decode_heads(heads["huber"], "huber")

        model = cls(config)
        model.feature_names = names
        model.basis_names = expected_basis_names
        model.interactions = interactions
        model.raw_medians = raw_medians
        model.raw_scales = raw_scales
        model.basis_means = basis_means
        model.basis_scales = basis_scales
        model.target_center = target_center
        model.target_scale = target_scale
        model.fit_metadata = metadata
        model.training_sample_count = sample_count
        model.training_positive_count = positive_count
        model.logistic_heads = logistic_heads
        model.huber_heads = huber_heads
        model._validate_fitted_semantics()
        if model.model_sha256 != payload_hash:
            raise ValueError("Decoded direct-edge GAM model hash changed unexpectedly")
        return model

    def _validate_fitted_semantics(self) -> None:
        self.config.validate()
        if self.fit_metadata is None:
            raise ValueError("Fitted GAM is missing fit metadata")
        self.fit_metadata.validate()
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Fitted GAM feature schema is invalid")
        expected_basis_names, expected_interactions = _feature_basis_spec(self.feature_names)
        if self.basis_names != expected_basis_names or self.interactions != expected_interactions:
            raise ValueError("Fitted GAM basis schema is invalid")
        if self.raw_medians.shape != (len(self.feature_names),):
            raise ValueError("Fitted GAM raw median shape is invalid")
        if self.raw_scales.shape != (len(self.feature_names),):
            raise ValueError("Fitted GAM raw scale shape is invalid")
        if self.basis_means.shape != (len(self.basis_names),):
            raise ValueError("Fitted GAM basis mean shape is invalid")
        if self.basis_scales.shape != (len(self.basis_names),):
            raise ValueError("Fitted GAM basis scale shape is invalid")
        for values in (
            self.raw_medians,
            self.raw_scales,
            self.basis_means,
            self.basis_scales,
        ):
            if not np.isfinite(values).all():
                raise ValueError("Fitted GAM preprocessing contains non-finite values")
        if (self.raw_scales < self.config.raw_scale_floor).any():
            raise ValueError("Fitted GAM raw scale violates the frozen floor")
        if (self.basis_scales < self.config.basis_scale_floor).any():
            raise ValueError("Fitted GAM basis scale violates the frozen floor")
        if not math.isfinite(self.target_center) or not math.isfinite(self.target_scale):
            raise ValueError("Fitted GAM target scaler is non-finite")
        if self.target_scale < self.config.target_scale_floor:
            raise ValueError("Fitted GAM target scale violates the frozen floor")
        if self.training_sample_count < 2:
            raise ValueError("Fitted GAM has too few training samples")
        if not 0 < self.training_positive_count < self.training_sample_count:
            raise ValueError("Fitted GAM training classes are invalid")
        dimension = len(self.basis_names) + 1
        for group in (self.logistic_heads, self.huber_heads):
            if len(group) != len(self.config.ridge_lambdas):
                raise ValueError("Fitted GAM has the wrong number of ensemble heads")
            for head, expected_lambda in zip(group, self.config.ridge_lambdas):
                if float(head["ridge_lambda"]).hex() != float(expected_lambda).hex():
                    raise ValueError("Fitted GAM ensemble lambda order is invalid")
                if head.get("converged") is not True:
                    raise ValueError("Fitted GAM contains a non-converged ensemble head")
                coefficients = np.asarray(head["coefficients"], dtype=float)
                if coefficients.shape != (dimension,) or not np.isfinite(coefficients).all():
                    raise ValueError("Fitted GAM coefficients are invalid")


DirectEdgeGam = DirectEdgeGAM


__all__ = [
    "ANCHOR_FEATURE_NAMES",
    "DirectEdgeGAM",
    "DirectEdgeGAMConfig",
    "DirectEdgeGam",
    "GAMConfig",
    "GAMFitMetadata",
]
